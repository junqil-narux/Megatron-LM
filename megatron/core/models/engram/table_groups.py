# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process groups that shard Engram tables, decoupled from the MoE expert-parallel group.

Two layouts are supported:

* ``"ep"`` (default): rows are sharded over the MoE EP group and replicated over expert-DP.
  DDP reduces the table gradients over expert-DP and the expert DistributedOptimizer
  shards the Adam states over it.
* ``"ep_edp"``: rows are sharded over EP x expert-DP inside one pipeline stage (still
  replicated over TP), so every row has exactly one owner per TP rank. The lookup's
  backward all-to-all already delivers every data-parallel contribution to that owner, so
  the table needs no data-parallel reduction: it gets its own DDP buffer and its own
  DistributedOptimizer over a single-rank replica group, tagged on the parameter as
  ``optimizer_sharding_group``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ENGRAM_TABLE_SHARDINGS = ("ep", "ep_edp")


@dataclass(frozen=True)
class EngramTableGroups:
    """Groups for one table layout.

    Attributes:
        table: Ranks the rows are sharded over (the lookup all-to-all group).
        replica: Ranks holding the same row shard.
        optimizer_sharding_group: Replica group to tag on the table parameters when the
            table leaves the expert-DP buffer, else None.
    """

    table: torch.distributed.ProcessGroup
    replica: torch.distributed.ProcessGroup
    optimizer_sharding_group: torch.distributed.ProcessGroup | None


_EP_EDP_GROUP_CACHE: dict[tuple[int, ...], EngramTableGroups] = {}


def ep_edp_table_ranks(ep_ranks: list[int], expt_dp_ranks: list[int], rank: int) -> list[int]:
    """Global ranks of the EP x expert-DP table group containing ``rank``, EP innermost.

    The expert rank grid is a product of axes, so ``rank``'s EP peers and expert-DP peers
    span a rectangle: the peer at (EP index i, expert-DP index j) is
    ``ep_ranks[i] + expt_dp_ranks[j] - rank``. EP is the inner axis, so table rank
    ``j * EP + i`` and each EP group owns one contiguous block of every table.
    """
    if rank not in ep_ranks or rank not in expt_dp_ranks:
        raise ValueError(f"rank {rank} must belong to both its EP and expert-DP groups.")
    return [ep_rank + edp_rank - rank for edp_rank in expt_dp_ranks for ep_rank in ep_ranks]


def get_engram_table_groups(
    table_sharding: str,
    ep_group: torch.distributed.ProcessGroup,
    expt_dp_group: torch.distributed.ProcessGroup,
) -> EngramTableGroups:
    """Return the table and replica groups for ``table_sharding``.

    ``"ep_edp"`` builds a dedicated communicator rather than reusing the data-parallel one
    (same ranks when CP=1): the per-microbatch lookup must not queue behind DDP's bucketed
    reduce-scatters and all-gathers. The first call is collective over the whole job (the
    layout requires PP=1, so every rank builds its Engram layers at the same point); later
    Engram layers reuse the groups.
    """
    if table_sharding == "ep":
        return EngramTableGroups(
            table=ep_group, replica=expt_dp_group, optimizer_sharding_group=None
        )
    if table_sharding != "ep_edp":
        raise ValueError(
            f"Unknown Engram table sharding {table_sharding!r}; "
            f"expected one of {ENGRAM_TABLE_SHARDINGS}."
        )

    from megatron.core.parallel_state import create_group

    rank = torch.distributed.get_rank()
    table_ranks = ep_edp_table_ranks(
        torch.distributed.get_process_group_ranks(ep_group),
        torch.distributed.get_process_group_ranks(expt_dp_group),
        rank,
    )
    key = tuple(table_ranks)
    groups = _EP_EDP_GROUP_CACHE.get(key)
    if groups is None:
        # Every rank creates every group in the same order, as parallel_state does. Creating
        # only one's own groups (use_local_synchronization) is not safe: PyTorch up to at
        # least 2.13 salts those group names with the size of the rank's group registry, which
        # differs across ranks once any earlier group had uneven membership, and the members'
        # store rendezvous then deadlocks.
        backend = torch.distributed.get_backend(ep_group)
        world_size = torch.distributed.get_world_size()
        all_table_ranks = [None] * world_size
        torch.distributed.all_gather_object(all_table_ranks, table_ranks)
        table_group = replica_group = None
        for ranks in sorted({tuple(group_ranks) for group_ranks in all_table_ranks}):
            group = create_group(list(ranks), backend=backend, group_desc="ENGRAM_TABLE_GROUP")
            if rank in ranks:
                table_group = group
        for other_rank in range(world_size):
            group = create_group(
                [other_rank], backend=backend, group_desc="ENGRAM_TABLE_REPLICA_GROUP"
            )
            if other_rank == rank:
                replica_group = group
        groups = EngramTableGroups(
            table=table_group, replica=replica_group, optimizer_sharding_group=replica_group
        )
        _EP_EDP_GROUP_CACHE[key] = groups
    return groups


def tag_engram_table_parameters(embedding: torch.nn.Module, groups: EngramTableGroups) -> None:
    """Mark the table shards that leave the expert-DP buffer for DDP and the optimizer."""
    if groups.optimizer_sharding_group is None:
        return
    for table in embedding.tables:
        table.weight.optimizer_sharding_group = groups.optimizer_sharding_group


@torch.no_grad()
def initialize_like_ep_layout(embedding: torch.nn.Module, ep_size: int, init_method) -> None:
    """Overwrite every table shard with the values the ``"ep"`` layout draws for its rows.

    Each layout seeds a shard by its own shard rank, so ``"ep"`` and ``"ep_edp"`` start from
    different (equally distributed) tables. For A/B runs this redraws, for every ``"ep"``
    shard overlapping this rank's rows, that shard on the ``"ep"`` seed and copies the overlap,
    so both layouts start from bit-identical tables. The transient is one ``"ep"`` shard.
    """
    from .distributed_embedding import get_contiguous_row_range, initialize_row_shard

    for table in embedding.tables:
        rows = table.global_num_embeddings
        for ep_rank in range(ep_size):
            ep_start, ep_end = get_contiguous_row_range(rows, ep_rank, ep_size)
            start, end = max(ep_start, table.row_start), min(ep_end, table.row_end)
            if start >= end:
                continue
            ep_shard = torch.empty(
                (ep_end - ep_start, table.embedding_dim),
                dtype=table.weight.dtype,
                device=table.weight.device,
            )
            initialize_row_shard(ep_shard, init_method, rows, ep_rank)
            table.weight[start - table.row_start : end - table.row_start].copy_(
                ep_shard[start - ep_start : end - ep_start]
            )


def destroy_engram_table_groups() -> None:
    """Forget cached groups (tests re-initialize model parallelism between cases)."""
    _EP_EDP_GROUP_CACHE.clear()
