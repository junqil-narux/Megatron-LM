# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Routed lookup for EP-sharded Engram tables over NCCL all-to-alls.

``EPShardedMultiTableEmbedding.forward`` (``distributed_embedding``) maps the requested rows
to their owners, exchanges the request counts, copies them to the host and then runs the
request all-to-all, the owner-side gather and the differentiable return all-to-all, all in one
call. The functions here run the same operations in the same order on an unchanged embedding,
as three steps::

    routed = route(embedding, hash_ids)                 # device work only
    exchange_route_counts([routed], embedding.ep_size, len(embedding.tables), group)
    rows = lookup_routed(embedding, routed, group)      # requests, owner gather, rows back

so the rows and the table gradients are those of ``embedding(hash_ids)``, bit for bit. The
split lets a caller route several lookups, exchange all their counts with one all-to-all and
one host transfer, and issue each lookup later, on another stream or on another communicator
over the same ranks.

The requests are counted with :func:`count_requests_per_bucket` instead of ``torch.bincount``,
which reads its input's minimum and maximum on the host. A routed lookup therefore
synchronizes with the host once, for the count transfer; the embedding's forward does so three
times.

The owner mapping, the exchanges and their order follow
``EPShardedMultiTableEmbedding.forward`` of NVIDIA/Megatron-LM#8225, adapted there from
NVIDIA/Megatron-LM#7231 by BestJuly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from .distributed_embedding import EPShardedMultiTableEmbedding, _DifferentiableEngramAllToAll


def count_requests_per_bucket(buckets: Tensor, num_buckets: int) -> Tensor:
    """Return the int64 request count of every bucket in ``[0, num_buckets)``.

    Equals ``torch.bincount(buckets, minlength=num_buckets)``, but ``bincount`` reads the input's
    minimum and maximum on the host (two device synchronizations per call); a scatter-add into a
    buffer of known size needs none.

    Args:
        buckets: 1-D int64 bucket of every request; each must lie in ``[0, num_buckets)``.
        num_buckets: Number of buckets.
    """
    counts = torch.zeros(num_buckets, dtype=torch.int64, device=buckets.device)
    return counts.scatter_add_(0, buckets, torch.ones_like(buckets))


@dataclass(eq=False)
class EngramRoute:
    """Owner routing of one lookup.

    :func:`route` fills the device tensors and :func:`exchange_route_counts` the host-side
    splits. On a row group of one rank no row leaves the rank: the route keeps the row IDs
    (``hash_ids``) and nothing else. Routes hold device tensors and compare by identity.

    Attributes:
        shape: Shape of the routed ``hash_ids`` (``[..., num_tables]``).
        owner_order: Permutation that sorts the flattened requests by owner.
        requests: ``[num_requests, 2]`` (table, owner-local row) pairs in owner order.
        send_counts: Requests per (owner, table), owner-major.
        send_splits: Requests sent to each owner.
        recv_splits: Requests received from each peer.
        table_counts: Received requests per table.
        hash_ids: The routed row IDs, on a row group of one rank only.
    """

    shape: torch.Size
    owner_order: Optional[Tensor] = None
    requests: Optional[Tensor] = None
    send_counts: Optional[Tensor] = None
    send_splits: Optional[list[int]] = None
    recv_splits: Optional[list[int]] = None
    table_counts: Optional[list[int]] = None
    hash_ids: Optional[Tensor] = None


def route(embedding: EPShardedMultiTableEmbedding, hash_ids: Tensor) -> EngramRoute:
    """Map global table rows to their owners and count the requests per (owner, table).

    Device work only, without host synchronization. The counts still have to be exchanged
    (:func:`exchange_route_counts`) before :func:`lookup_routed` can run, except on a row
    group of one rank.

    Args:
        embedding: The tables and their row group.
        hash_ids: Per-table row IDs with shape ``[..., num_tables]``.
    """
    num_tables = len(embedding.tables)
    if hash_ids.shape[-1] != num_tables:
        raise ValueError(f"Expected {num_tables} Engram hash heads, got {hash_ids.shape[-1]}.")
    original_shape = hash_ids.shape
    if embedding.ep_size == 1:
        return EngramRoute(shape=original_shape, hash_ids=hash_ids)

    rows = hash_ids.reshape(-1).to(torch.int64)
    table_ids = (
        torch.arange(num_tables, device=hash_ids.device, dtype=torch.int64)
        .view(*([1] * (hash_ids.ndim - 1)), -1)
        .expand(original_shape)
        .reshape(-1)
    )

    message = "engram.lookup.owner-map"
    nvtx_range_push(message)
    try:
        # Balanced contiguous partition: ranks below `remainder` own `base + 1` rows and the
        # rest own `base`, so the owner of a global row follows in O(1) per request.
        base = embedding.table_row_base.index_select(0, table_ids)
        remainder = embedding.table_row_remainder.index_select(0, table_ids)
        large_region_end = (base + 1) * remainder
        owners = torch.where(
            rows < large_region_end,
            rows // (base + 1),
            remainder + (rows - large_region_end) // base.clamp_min(1),
        )
        owner_starts = owners * base + torch.minimum(owners, remainder)
        local_rows = rows - owner_starts

        owner_order = torch.argsort(owners, stable=True)
        sorted_owners = owners.index_select(0, owner_order)
        requests = torch.stack(
            (table_ids.index_select(0, owner_order), local_rows.index_select(0, owner_order)),
            dim=-1,
        )
        # Count per (peer, table) rather than per peer: the same exchange then yields the
        # all-to-all splits and the owner-side per-table counts from one host transfer.
        send_counts = count_requests_per_bucket(
            sorted_owners * num_tables + requests[:, 0], embedding.ep_size * num_tables
        )
    finally:
        nvtx_range_pop(message)
    return EngramRoute(
        shape=original_shape, owner_order=owner_order, requests=requests, send_counts=send_counts
    )


def exchange_route_counts(routes: list[EngramRoute], ep_size: int, num_tables: int, group) -> None:
    """Exchange the request counts of ``routes`` and fill in their host-side splits.

    One all-to-all and one host transfer serve every route, so the counts of several lookups
    (the microbatches of a step, the Engram layers of a model) cost a single host
    synchronization. The transfer synchronizes only with the current stream. A row group of one
    rank has no counts to exchange.

    Args:
        routes: Routes on this rank, in the same order on every rank of ``group``; all of them
            route ``num_tables`` tables over ``ep_size`` owners.
        ep_size: Number of owners (the size of ``group``).
        num_tables: Tables per route.
        group: Group with the owners' ranks, in owner order.
    """
    if ep_size == 1 or not routes:
        return
    for route in routes:
        if route.send_counts is None or route.send_counts.numel() != ep_size * num_tables:
            raise ValueError(
                f"Engram exchange_route_counts needs routes of {num_tables} tables over "
                f"{ep_size} owners."
            )
    if len(routes) == 1:
        send_counts = routes[0].send_counts
    else:
        # [routes, peers, tables] -> [peers, routes, tables]: each peer receives one contiguous
        # block holding its counts of every route.
        send_counts = torch.stack(
            [route.send_counts.view(ep_size, num_tables) for route in routes], dim=1
        ).reshape(-1)
    recv_counts = torch.empty_like(send_counts)

    message = "engram.lookup.count-a2a"
    nvtx_range_push(message)
    try:
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=group)
    finally:
        nvtx_range_pop(message)
    sent, received = torch.stack((send_counts, recv_counts)).tolist()

    per_peer = len(routes) * num_tables
    for index, route in enumerate(routes):
        offsets = [peer * per_peer + index * num_tables for peer in range(ep_size)]
        route.send_splits = [sum(sent[offset : offset + num_tables]) for offset in offsets]
        route.recv_splits = [sum(received[offset : offset + num_tables]) for offset in offsets]
        route.table_counts = [
            sum(received[offset + table] for offset in offsets) for table in range(num_tables)
        ]


def lookup_routed(
    embedding: EPShardedMultiTableEmbedding, route: EngramRoute, group=None
) -> Tensor:
    """Fetch the routed rows from their owners and restore token/head ordering.

    Args:
        embedding: The embedding that ``route`` was made for.
        route: A route whose counts were exchanged.
        group: Group for the request and return all-to-alls (and the return all-to-all's
            backward); defaults to the embedding's row group. Any other group must have the
            same ranks in the same order. Unused on a row group of one rank.

    Returns:
        Retrieved embeddings with shape ``[..., num_tables, embedding_dim]``.
    """
    if route.requests is None:
        # A row group of one rank: the embedding looks its own rows up without routing,
        # communication or host synchronization.
        return embedding(route.hash_ids)
    if route.send_splits is None:
        raise RuntimeError("Engram lookup_routed needs a route whose counts were exchanged.")
    group = embedding.ep_group if group is None else group

    received_requests = route.requests.new_empty((sum(route.recv_splits), 2))
    message = "engram.lookup.request-a2a"
    nvtx_range_push(message)
    try:
        torch.distributed.all_to_all_single(
            received_requests,
            route.requests.contiguous(),
            output_split_sizes=route.recv_splits,
            input_split_sizes=route.send_splits,
            group=group,
        )
    finally:
        nvtx_range_pop(message)

    message = "engram.lookup.local-embedding"
    nvtx_range_push(message)
    try:
        owner_embeddings = embedding._lookup_received_requests(
            received_requests, route.table_counts
        )
    finally:
        nvtx_range_pop(message)
    sorted_embeddings = _DifferentiableEngramAllToAll.apply(
        group, owner_embeddings, route.send_splits, route.recv_splits
    )
    output = torch.empty_like(sorted_embeddings)
    output.index_copy_(0, route.owner_order, sorted_embeddings)
    return output.view(*route.shape, embedding.embedding_dim)
