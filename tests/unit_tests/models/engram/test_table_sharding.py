# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram tables sharded over EP x expert-DP (``ep_edp``) against the EP layout (``ep``).

The distributed cases need exactly four ranks, laid out as the expert grid EP2 x expert-DP2
(``rank = ep + 2 * edp``). They run with NCCL on GPUs or with Gloo on CPU::

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/models/engram/test_table_sharding.py

``ep`` keeps two replicas of every row shard and sums them over expert-DP after backward, as
DDP's expert buffer does; ``ep_edp`` gives every row one owner that receives all four ranks'
gradients through the lookup's backward all-to-all and only applies the 1/dp scaling. With
exactly representable (integer) gradients the two layouts must agree bit for bit. With random
bf16 gradients they differ only on rows that both expert-DP replicas touch in the same
microbatch: ``ep`` rounds each replica's partial sum to bf16 before adding them, ``ep_edp``
rounds the whole sum once.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.models.engram.table_groups import (
    destroy_engram_table_groups,
    ep_edp_table_ranks,
    get_engram_table_groups,
    initialize_like_ep_layout,
    tag_engram_table_parameters,
)
from megatron.core.parallel_state import RankGenerator
from megatron.core.utils import get_pg_rank

WORLD_SIZE = 4
EP_SIZE = 2
EXPERT_DP_SIZE = 2
TABLE_SIZES = (11, 13, 5)
EMBEDDING_DIM = 4
NUM_MICROBATCHES = 2


@pytest.mark.parametrize(
    "tp, ep, dp, pp", [(1, 2, 4, 1), (2, 2, 4, 1), (1, 4, 8, 2), (2, 4, 8, 1), (1, 8, 8, 1)]
)
def test_table_group_is_stage_data_parallel_group_with_ep_innermost(tp, ep, dp, pp):
    order = "tp-cp-ep-dp-pp"
    dense = RankGenerator(tp=tp, ep=1, dp=dp, pp=pp, cp=1, order=order)
    expert = RankGenerator(tp=tp, ep=ep, dp=dp // ep, pp=pp, cp=1, order=order)
    ep_groups, expt_dp_groups = expert.get_ranks("ep"), expert.get_ranks("dp")
    for rank in range(tp * dp * pp):
        ep_ranks = next(group for group in ep_groups if rank in group)
        expt_dp_ranks = next(group for group in expt_dp_groups if rank in group)
        table_ranks = ep_edp_table_ranks(ep_ranks, expt_dp_ranks, rank)
        # Same ranks and order as the expert grid's EP x expert-DP group and, with CP=1, as the
        # dense data-parallel group: the 1/dp gradient scaling covers exactly the table group.
        assert table_ranks == next(g for g in expert.get_ranks("ep-dp") if rank in g)
        assert table_ranks == next(g for g in dense.get_ranks("dp") if rank in g)
        # EP innermost: every EP group is one contiguous block of table ranks.
        for edp_index, edp_peer in enumerate(expt_dp_ranks):
            block = table_ranks[edp_index * ep : (edp_index + 1) * ep]
            assert block == next(group for group in ep_groups if edp_peer in group)


def test_ep_edp_rejects_unsupported_layouts():
    supported = dict(
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        gtp_weight_remat_size=1,
        expert_gtp_weight_remat_size=1,
    )
    holder = SimpleNamespace(table_sharding="ep_edp")
    EngramConfig.validate_table_sharding(
        holder, SimpleNamespace(**supported), use_distributed_optimizer=True
    )
    for field in supported:
        with pytest.raises(ValueError, match=field):
            EngramConfig.validate_table_sharding(
                holder, SimpleNamespace(**{**supported, field: 2}), use_distributed_optimizer=True
            )
    with pytest.raises(ValueError, match="use-distributed-optimizer"):
        EngramConfig.validate_table_sharding(
            holder, SimpleNamespace(**supported), use_distributed_optimizer=False
        )
    with pytest.raises(ValueError, match="num_distributed_optimizer_instances"):
        EngramConfig.validate_table_sharding(
            holder,
            SimpleNamespace(**supported),
            use_distributed_optimizer=True,
            num_distributed_optimizer_instances=2,
        )
    # The EP layout keeps every combination it supported before.
    EngramConfig.validate_table_sharding(
        SimpleNamespace(table_sharding="ep"),
        SimpleNamespace(**{**supported, "context_parallel_size": 2}),
        use_distributed_optimizer=False,
    )


_GROUPS = {}


def _distributed():
    """Initialize four ranks (NCCL on GPU, Gloo on CPU) and the EP2 x expert-DP2 groups."""
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        pytest.skip(f"requires torchrun with exactly {WORLD_SIZE} ranks")
    if not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            from tests.unit_tests.test_utilities import Utils

            Utils.initialize_distributed()
        else:
            torch.distributed.init_process_group("gloo")
    if not _GROUPS:
        rank = torch.distributed.get_rank()
        for edp in range(EXPERT_DP_SIZE):
            ranks = [ep + EP_SIZE * edp for ep in range(EP_SIZE)]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _GROUPS["ep"] = group
        for ep in range(EP_SIZE):
            ranks = [ep + EP_SIZE * edp for edp in range(EXPERT_DP_SIZE)]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _GROUPS["expt_dp"] = group
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return torch.distributed.get_rank(), device, _GROUPS["ep"], _GROUPS["expt_dp"]


def test_table_groups_survive_uneven_group_membership():
    """Group creation must not depend on how many groups each rank already belongs to.

    PyTorch (2.13 at least) names members-only groups after the size of the rank's group
    registry; after a group with uneven membership, the ranks disagree on that name and a
    members-only rendezvous deadlocks. Every rank must therefore create every table group.
    """
    rank, device, ep_group, expt_dp_group = _distributed()
    # Ranks 2 and 3 are not members, so their group registries end up smaller.
    torch.distributed.new_group([0, 1])
    destroy_engram_table_groups()
    groups = get_engram_table_groups("ep_edp", ep_group, expt_dp_group)
    probe = torch.ones(1, device=device)
    torch.distributed.all_reduce(probe, group=groups.table)
    torch.distributed.all_reduce(probe, group=groups.replica)
    assert probe.item() == WORLD_SIZE
    assert torch.distributed.get_process_group_ranks(groups.replica) == [rank]


def _full_tables(device, dtype):
    """Global tables with small dyadic values (exact in bf16)."""
    tables = []
    for table_id, rows in enumerate(TABLE_SIZES):
        row = torch.arange(rows, device=device).unsqueeze(1)
        column = torch.arange(EMBEDDING_DIM, device=device).unsqueeze(0)
        tables.append((((table_id * 7 + row * 3 + column) % 17 - 8) / 8.0).to(dtype))
    return tables


def _build(table_sharding, ep_group, expt_dp_group, device, dtype, deterministic=False):
    config = SimpleNamespace(
        use_cpu_initialization=device.type == "cpu",
        params_dtype=dtype,
        perform_initialization=False,
        sequence_parallel=False,
        deterministic_mode=deterministic,
    )
    groups = get_engram_table_groups(table_sharding, ep_group, expt_dp_group)
    embedding = EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=TABLE_SIZES,
        embedding_dim=EMBEDDING_DIM,
        init_method=lambda _: None,
        ep_group=groups.table,
        tp_group=None,
        expt_dp_group=groups.replica,
    )
    tag_engram_table_parameters(embedding, groups)
    with torch.no_grad():
        for table, full in zip(embedding.tables, _full_tables(device, dtype)):
            table.weight.copy_(full[table.row_start : table.row_end])
    return embedding


def _hash_ids(rank, microbatch, empty_microbatch):
    """Per-(rank, microbatch) lookups with heavy repetition and one optional empty microbatch."""
    if empty_microbatch and rank == WORLD_SIZE - 1 and microbatch == 0:
        return torch.zeros((1, 0, len(TABLE_SIZES)), dtype=torch.int64)
    generator = torch.Generator().manual_seed(1000 * rank + microbatch)
    num_tokens = 6 + 3 * rank
    ids = torch.stack(
        [torch.randint(0, rows, (num_tokens,), generator=generator) for rows in TABLE_SIZES], dim=-1
    )
    # Hot rows: every rank of both expert-DP replicas hits row 0 of every table.
    ids[:2] = 0
    return ids.view(1, num_tokens, len(TABLE_SIZES))


def _output_gradients(rank, microbatch, num_tokens, exact):
    generator = torch.Generator().manual_seed(5000 + 1000 * rank + microbatch)
    shape = (1, num_tokens, len(TABLE_SIZES), EMBEDDING_DIM)
    if exact:
        return torch.randint(-3, 4, shape, generator=generator).float()
    return torch.randn(shape, generator=generator)


def _train_microbatches(embedding, rank, device, exact, empty_microbatch):
    """Forward/backward every microbatch, accumulating into fp32 main grads as DDP does."""
    main_grads = [
        torch.zeros(table.weight.shape, dtype=torch.float32, device=device)
        for table in embedding.tables
    ]
    outputs = []
    for microbatch in range(NUM_MICROBATCHES):
        hash_ids = _hash_ids(rank, microbatch, empty_microbatch).to(device)
        output = embedding(hash_ids)
        output_grad = _output_gradients(rank, microbatch, hash_ids.shape[1], exact)
        (output * output_grad.to(device=device, dtype=output.dtype)).sum().backward()
        outputs.append(output.detach())
        for main_grad, table in zip(main_grads, embedding.tables):
            main_grad.add_(table.weight.grad)
            table.weight.grad = None
    return outputs, main_grads


def _finish_grad_sync(embedding, main_grads, expt_dp_group):
    """DDP's reduction: scale by 1/dp, then sum over the buffer's data-parallel group."""
    for table, main_grad in zip(embedding.tables, main_grads):
        main_grad.mul_(1.0 / WORLD_SIZE)
        sharding_group = getattr(table.weight, "optimizer_sharding_group", None)
        group = expt_dp_group if sharding_group is None else sharding_group
        torch.distributed.all_reduce(main_grad, group=group)


def _assemble(embedding, shards, device):
    """Full tables from the row shards, taking each row from replica 0 of its shard only."""
    full_tables = []
    for table, shard in zip(embedding.tables, shards):
        full = torch.zeros(
            (table.global_num_embeddings, EMBEDDING_DIM), dtype=torch.float32, device=device
        )
        if get_pg_rank(table.expt_dp_group) == 0:
            full[table.row_start : table.row_end] = shard.float()
        torch.distributed.all_reduce(full)
        full_tables.append(full)
    return full_tables


def _reference_gradients(device, exact, empty_microbatch):
    """Full-table gradients of every rank's lookups, scaled by 1/dp, in fp32."""
    gradients = [torch.zeros((rows, EMBEDDING_DIM), device=device) for rows in TABLE_SIZES]
    for rank in range(WORLD_SIZE):
        for microbatch in range(NUM_MICROBATCHES):
            hash_ids = _hash_ids(rank, microbatch, empty_microbatch)
            output_grad = _output_gradients(rank, microbatch, hash_ids.shape[1], exact)
            for table_id, gradient in enumerate(gradients):
                gradient.index_add_(
                    0,
                    hash_ids[..., table_id].reshape(-1).to(device),
                    output_grad[..., table_id, :].reshape(-1, EMBEDDING_DIM).to(device),
                )
    return [gradient / WORLD_SIZE for gradient in gradients]


def _rows_shared_by_replicas(empty_microbatch):
    """Per table, the rows both expert-DP replicas look up within one microbatch."""
    shared = [set() for _ in TABLE_SIZES]
    for microbatch in range(NUM_MICROBATCHES):
        for table_id in range(len(TABLE_SIZES)):
            touched = []
            for edp in range(EXPERT_DP_SIZE):
                rows = set()
                for ep in range(EP_SIZE):
                    hash_ids = _hash_ids(ep + EP_SIZE * edp, microbatch, empty_microbatch)
                    rows.update(hash_ids[..., table_id].reshape(-1).tolist())
                touched.append(rows)
            shared[table_id] |= touched[0] & touched[1]
    return shared


def _assert_equal(actual, expected, what):
    """torch.equal with a per-rank diagnostic printed before raising (ranks may abort later)."""
    if torch.equal(actual, expected):
        return
    diff = (actual.float() - expected.float()).abs()
    print(
        f"[rank {torch.distributed.get_rank()}] {what}: dtype {actual.dtype}/{expected.dtype} "
        f"shape {tuple(actual.shape)}/{tuple(expected.shape)} "
        f"max|diff| {diff.max().item() if diff.numel() else 0.0}",
        flush=True,
    )
    raise AssertionError(what)


@pytest.mark.parametrize("exact", [True, False], ids=["exact", "random"])
@pytest.mark.parametrize("empty_microbatch", [False, True], ids=["full", "empty-microbatch"])
@pytest.mark.parametrize("deterministic", [False, True], ids=["default", "deterministic"])
def test_lookup_and_gradients_match_ep_layout(exact, empty_microbatch, deterministic):
    rank, device, ep_group, expt_dp_group = _distributed()
    dtype = torch.bfloat16
    layouts = {
        name: _build(name, ep_group, expt_dp_group, device, dtype, deterministic)
        for name in ("ep", "ep_edp")
    }
    groups = get_engram_table_groups("ep_edp", ep_group, expt_dp_group)
    assert torch.distributed.get_process_group_ranks(groups.table) == list(range(WORLD_SIZE))
    assert groups.replica.size() == 1
    for table in layouts["ep_edp"].tables:
        assert table.weight.optimizer_sharding_group is groups.replica
    for table in layouts["ep"].tables:
        assert not hasattr(table.weight, "optimizer_sharding_group")

    results = {}
    for name, embedding in layouts.items():
        outputs, main_grads = _train_microbatches(embedding, rank, device, exact, empty_microbatch)
        _finish_grad_sync(embedding, main_grads, expt_dp_group)
        results[name] = (outputs, _assemble(embedding, main_grads, device))

    full_tables = _full_tables(device, dtype)
    for microbatch, (output_ep, output_ep_edp) in enumerate(
        zip(results["ep"][0], results["ep_edp"][0])
    ):
        hash_ids = _hash_ids(rank, microbatch, empty_microbatch).to(device)
        expected = torch.stack(
            [full[hash_ids[..., t]] for t, full in enumerate(full_tables)], dim=-2
        )
        _assert_equal(output_ep, expected, f"ep output, microbatch {microbatch}")
        _assert_equal(output_ep_edp, expected, f"ep_edp output, microbatch {microbatch}")

    reference = _reference_gradients(device, exact, empty_microbatch)
    shared_rows = _rows_shared_by_replicas(empty_microbatch)
    for table_id, (grad_ep, grad_ep_edp) in enumerate(zip(results["ep"][1], results["ep_edp"][1])):
        if exact:
            _assert_equal(grad_ep, reference[table_id], f"ep gradient, table {table_id}")
            _assert_equal(grad_ep_edp, reference[table_id], f"ep_edp gradient, table {table_id}")
            continue
        exclusive = torch.ones(TABLE_SIZES[table_id], dtype=torch.bool, device=device)
        exclusive[list(shared_rows[table_id])] = False
        _assert_equal(
            grad_ep_edp[exclusive], grad_ep[exclusive], f"exclusive rows, table {table_id}"
        )
        # Shared rows differ by bf16 rounding of the per-replica partial sums.
        torch.testing.assert_close(grad_ep_edp, reference[table_id], rtol=2**-6, atol=2**-6)
        torch.testing.assert_close(grad_ep, reference[table_id], rtol=2**-6, atol=2**-6)


@pytest.mark.parametrize("table_sharding", ["ep", "ep_edp"])
def test_sharded_state_dict_covers_every_row_once(table_sharding):
    rank, device, ep_group, expt_dp_group = _distributed()
    embedding = _build(table_sharding, ep_group, expt_dp_group, device, torch.float32)
    state = embedding.sharded_state_dict()
    entries = [
        (key, tensor.global_offset[0], tensor.local_shape[0], tensor.replica_id)
        for key, tensor in state.items()
    ]
    gathered = [None] * WORLD_SIZE
    torch.distributed.all_gather_object(gathered, entries)
    for table_id, rows in enumerate(TABLE_SIZES):
        key = f"tables.{table_id}.weight"
        intervals = sorted(
            (offset, offset + count)
            for rank_entries in gathered
            for entry_key, offset, count, replica_id in rank_entries
            if entry_key == key and all(index == 0 for index in replica_id)
        )
        assert intervals[0][0] == 0 and intervals[-1][1] == rows
        assert all(end == start for (_, end), (start, _) in zip(intervals, intervals[1:]))
        replicas = [
            entry[3] for rank_entries in gathered for entry in rank_entries if entry[0] == key
        ]
        if table_sharding == "ep_edp":
            # One owner per row: four disjoint main shards, no replicas.
            assert len(intervals) == WORLD_SIZE and all(r == (0, 0, 0) for r in replicas)
        else:
            assert len(intervals) == EP_SIZE


def test_init_like_ep_reproduces_the_ep_layout_tables():
    _, device, ep_group, expt_dp_group = _distributed()
    torch.manual_seed(1234)
    init_method = lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.02)
    config = SimpleNamespace(
        use_cpu_initialization=device.type == "cpu",
        params_dtype=torch.bfloat16,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=False,
    )
    full_tables = {}
    for table_sharding in ("ep", "ep_edp"):
        groups = get_engram_table_groups(table_sharding, ep_group, expt_dp_group)
        embedding = EPShardedMultiTableEmbedding(
            config=config,
            table_sizes=TABLE_SIZES,
            embedding_dim=EMBEDDING_DIM,
            init_method=init_method,
            ep_group=groups.table,
            tp_group=None,
            expt_dp_group=groups.replica,
        )
        if table_sharding == "ep_edp":
            own = _assemble(embedding, [table.weight for table in embedding.tables], device)
            initialize_like_ep_layout(embedding, EP_SIZE, init_method)
        full_tables[table_sharding] = _assemble(
            embedding, [table.weight for table in embedding.tables], device
        )
    for ep_table, own_table, like_ep_table in zip(full_tables["ep"], own, full_tables["ep_edp"]):
        assert torch.equal(like_ep_table, ep_table)
        assert not torch.equal(own_table, ep_table)


def _shared_checkpoint_dir():
    path = [
        (
            tempfile.mkdtemp(prefix="engram-table-sharding-")
            if torch.distributed.get_rank() == 0
            else None
        )
    ]
    torch.distributed.broadcast_object_list(path, src=0)
    return path[0]


@pytest.mark.parametrize("source, target", [("ep", "ep_edp"), ("ep_edp", "ep")])
def test_checkpoint_loads_across_layouts(source, target):
    from megatron.core.dist_checkpointing import load, save

    rank, device, ep_group, expt_dp_group = _distributed()
    if device.type != "cuda":
        pytest.skip("dist_checkpointing save needs CUDA")
    saved = _build(source, ep_group, expt_dp_group, device, torch.float32)
    loaded = _build(target, ep_group, expt_dp_group, device, torch.float32)
    with torch.no_grad():
        for table in loaded.tables:
            table.weight.zero_()

    checkpoint_dir = _shared_checkpoint_dir()
    save(saved.sharded_state_dict(), checkpoint_dir)
    torch.distributed.barrier()
    loaded.load_state_dict(load(loaded.sharded_state_dict(), checkpoint_dir))

    for table, full in zip(loaded.tables, _full_tables(device, torch.float32)):
        assert torch.equal(table.weight, full[table.row_start : table.row_end])
    torch.distributed.barrier()
    if rank == 0:
        import shutil

        shutil.rmtree(checkpoint_dir, ignore_errors=True)
