# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Routed Engram lookup against the lookup of the embedding's own forward.

``routed_lookup`` runs the operations of ``EPShardedMultiTableEmbedding.forward`` as three
calls: route, count exchange and routed lookup. The cases therefore demand the embedding's
rows and table gradients bit for bit, through the same all-to-alls, and pin down what the
split is for: one host synchronization per lookup where the embedding's forward makes three,
and one count all-to-all and one host synchronization for any number of routes.

The multi-rank cases need GPUs. They run under the CI unit-test launcher
(torch.distributed.run at 8 ranks) as well as local torchrun with 2 or 4 ranks: each case
builds a row group of the first ``ep_size`` ranks and the remaining ranks sit the case out
after the collective group creation. The other cases run on CPU in any process.
"""

import os
import warnings
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from megatron.core.models.engram.distributed_embedding import (
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from megatron.core.models.engram.routed_lookup import exchange_route_counts, lookup_routed, route
from tests.unit_tests.test_utilities import Utils

from ._test_utils import deterministic_algorithms, sync_debug_mode

_multi_rank = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8),
    reason="requires torchrun with two, four, or eight ranks",
)

# One tiny table (a few rows per shard) and two prime-sized ones.
_TABLE_SIZES = (13, 1009, 997)
_EMBEDDING_DIM = 16

_BIT_VIEWS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}


class _Lookup(NamedTuple):
    """What one lookup and its backward produced."""

    rows: torch.Tensor
    table_grads: list
    # (output splits, input splits, received buffer) of every all-to-all, backward included.
    all_to_alls: list
    # Synchronizing CUDA calls of the forward.
    synchronizations: int


def _bits(tensor):
    tensor = tensor.detach().contiguous()
    view = _BIT_VIEWS.get(tensor.dtype)
    return tensor if view is None else tensor.view(view)


def _assert_bitwise_equal(actual, expected, what):
    assert actual.dtype == expected.dtype, f"{what}: {actual.dtype} vs {expected.dtype}"
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    assert torch.equal(_bits(actual), _bits(expected)), f"{what} differs"


def _assert_same_lookup(actual, expected):
    """Same rows, table gradients and all-to-alls (splits and received buffers)."""
    _assert_bitwise_equal(actual.rows, expected.rows, "rows")
    for table_id, (grad, ref_grad) in enumerate(zip(actual.table_grads, expected.table_grads)):
        _assert_bitwise_equal(grad, ref_grad, f"table {table_id} gradient")
    assert len(actual.all_to_alls) == len(expected.all_to_alls)
    for index, ((splits_out, splits_in, received), (ref_out, ref_in, ref_received)) in enumerate(
        zip(actual.all_to_alls, expected.all_to_alls)
    ):
        assert splits_out == ref_out, f"all-to-all {index} output splits"
        assert splits_in == ref_in, f"all-to-all {index} input splits"
        _assert_bitwise_equal(received, ref_received, f"all-to-all {index} received buffer")


def _row_group(ep_size):
    """Initialize the world; return this rank, its device and a row group of the first
    ``ep_size`` ranks, or no group on the ranks that sit the case out."""
    Utils.initialize_distributed()
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if world_size < ep_size:
        pytest.skip(f"requires at least {ep_size} ranks")
    # Collective on the default group: every rank must participate in group creation.
    group = torch.distributed.new_group(ranks=list(range(ep_size)))
    device = torch.device("cuda", torch.cuda.current_device())
    return rank, device, group if rank < ep_size else None


def _embedding(group, device, deterministic=False, seed=1234):
    # The tables draw their rows from a stream keyed by the seed, the table and the rank.
    torch.manual_seed(seed)
    config = SimpleNamespace(
        use_cpu_initialization=device.type == "cpu",
        params_dtype=torch.float32,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=deterministic,
    )
    return EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=_TABLE_SIZES,
        embedding_dim=_EMBEDDING_DIM,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1),
        ep_group=group,
        tp_group=None,
        expt_dp_group=None,
    )


def _hash_ids(case, ep_size, rank, device, seed=0):
    """Global row IDs ``[batch, seq, tables]`` of one lookup, different on every rank."""
    if case == "empty" or (case == "empty_on_first_rank" and rank == 0):
        return torch.empty((1, 0, len(_TABLE_SIZES)), dtype=torch.int64, device=device)
    generator = torch.Generator().manual_seed(4321 + rank + 1000 * seed)
    batch, seq = 2, 37
    # "first_owner_only": every request lands on rank 0, so the other peers receive and
    # return nothing and every rank sends empty splits to them.
    limits = [
        get_contiguous_row_range(size, 0, ep_size)[1] if case == "first_owner_only" else size
        for size in _TABLE_SIZES
    ]
    columns = [torch.randint(0, limit, (batch, seq), generator=generator) for limit in limits]
    return torch.stack(columns, dim=-1).to(device)


def _record_all_to_alls(monkeypatch):
    """Record the splits and the received buffer of every all-to-all from here on."""
    calls = []
    real = torch.distributed.all_to_all_single

    def recording(output, input, output_split_sizes=None, input_split_sizes=None, **kwargs):
        result = real(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            **kwargs,
        )
        calls.append(
            (
                None if output_split_sizes is None else list(output_split_sizes),
                None if input_split_sizes is None else list(input_split_sizes),
                output.detach().clone(),
            )
        )
        return result

    monkeypatch.setattr(torch.distributed, "all_to_all_single", recording)
    return calls


def _count_synchronizations(device, fn):
    """Run ``fn``; return its result and the number of synchronizing CUDA calls it made."""
    if device.type != "cuda":
        return fn(), 0
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught, sync_debug_mode("warn"):
        warnings.simplefilter("always")
        result = fn()
    torch.cuda.synchronize()
    return result, sum("synchroniz" in str(warning.message) for warning in caught)


def _run(lookup, embedding, hash_ids, grad_seed, calls):
    """Look ``hash_ids`` up with ``lookup`` and back-propagate a random gradient of the rows."""
    for table in embedding.tables:
        table.weight.grad = None
    first_call = len(calls)
    rows, synchronizations = _count_synchronizations(hash_ids.device, lambda: lookup(hash_ids))
    grad_rows = torch.randn(
        rows.shape,
        generator=torch.Generator(device=rows.device).manual_seed(grad_seed),
        device=rows.device,
        dtype=rows.dtype,
    )
    rows.backward(grad_rows)
    table_grads = [table.weight.grad.detach().clone() for table in embedding.tables]
    return _Lookup(rows.detach().clone(), table_grads, calls[first_call:], synchronizations)


def _routed(embedding):
    """The embedding's lookup as its three routed steps."""

    def lookup(hash_ids):
        routed = route(embedding, hash_ids)
        exchange_route_counts(
            [routed], embedding.ep_size, len(embedding.tables), embedding.ep_group
        )
        return lookup_routed(embedding, routed)

    return lookup


@_multi_rank
@pytest.mark.parametrize("case", ["random", "first_owner_only", "empty", "empty_on_first_rank"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("ep_size", [2, 4])
def test_routed_lookup_matches_the_embedding(ep_size, deterministic, case, monkeypatch):
    """The embedding's all-to-alls, rows and table gradients, with one host synchronization
    where the embedding's forward makes three."""
    rank, device, group = _row_group(ep_size)
    if group is None:
        return
    embedding = _embedding(group, device, deterministic)
    hash_ids = _hash_ids(case, ep_size, rank, device)
    calls = _record_all_to_alls(monkeypatch)

    with deterministic_algorithms(deterministic):
        expected = _run(embedding, embedding, hash_ids, rank, calls)
        actual = _run(_routed(embedding), embedding, hash_ids, rank, calls)

    _assert_same_lookup(actual, expected)
    # Count, request and return exchanges forward, return exchange backward.
    assert len(actual.all_to_alls) == 4
    # The exchanged counts sum to the requests this rank received.
    assert int(actual.all_to_alls[0][2].sum()) == sum(actual.all_to_alls[1][0])
    # The transfer of the exchanged counts is the only host read of a routed lookup. The
    # embedding's bincount adds two unless it has no requests to count; the first lookup of a
    # process can report one more.
    assert actual.synchronizations == 1
    assert expected.synchronizations >= (3 if hash_ids.numel() > 0 else 1)


@_multi_rank
@pytest.mark.parametrize("ep_size", [2, 4])
def test_routes_share_one_count_exchange(ep_size, monkeypatch):
    """The counts of every microbatch and Engram layer take one all-to-all and one host
    transfer; the lookups that follow read nothing on the host."""
    rank, device, group = _row_group(ep_size)
    if group is None:
        return
    # Two layers with tables of their own over one row group, three microbatches each.
    embeddings = [_embedding(group, device, seed=1234 + layer) for layer in range(2)]
    lookups = [
        (embedding, _hash_ids(case, ep_size, rank, device, seed=2 * microbatch + layer))
        for microbatch, case in enumerate(("random", "empty", "first_owner_only"))
        for layer, embedding in enumerate(embeddings)
    ]
    calls = _record_all_to_alls(monkeypatch)
    expected = [
        _run(embedding, embedding, hash_ids, rank + index, calls)
        for index, (embedding, hash_ids) in enumerate(lookups)
    ]

    first_call = len(calls)
    routes, route_synchronizations = _count_synchronizations(
        device, lambda: [route(embedding, hash_ids) for embedding, hash_ids in lookups]
    )
    route_calls = len(calls) - first_call
    _, exchange_synchronizations = _count_synchronizations(
        device, lambda: exchange_route_counts(routes, ep_size, len(_TABLE_SIZES), group)
    )
    exchange_calls = len(calls) - first_call
    actual = [
        _run(
            lambda _, embedding=embedding, routed=routed: lookup_routed(embedding, routed),
            embedding,
            hash_ids,
            rank + index,
            calls,
        )
        for index, ((embedding, hash_ids), routed) in enumerate(zip(lookups, routes))
    ]

    # Checked once every collective has run, so a failing rank leaves no peer waiting.
    assert (route_synchronizations, route_calls) == (0, 0)
    assert (exchange_synchronizations, exchange_calls) == (1, 1)
    for result, inline in zip(actual, expected):
        # The lookup's own count exchange is the one all-to-all the routed lookup lacks.
        _assert_same_lookup(result, inline._replace(all_to_alls=inline.all_to_alls[1:]))
        assert result.synchronizations == 0


def test_single_rank_row_group_routes_nothing(monkeypatch):
    """Without peers the routed steps are the embedding's local lookup: no all-to-all."""
    device = torch.device("cpu")
    embedding = _embedding(None, device)
    assert embedding.ep_size == 1
    hash_ids = _hash_ids("random", 1, 0, device)
    calls = _record_all_to_alls(monkeypatch)

    expected = _run(embedding, embedding, hash_ids, 0, calls)
    actual = _run(_routed(embedding), embedding, hash_ids, 0, calls)

    _assert_same_lookup(actual, expected)
    assert calls == []


def test_route_maps_rows_to_owners_without_communication(monkeypatch):
    """Requests in stable owner order with owner-local rows, counted per (owner, table)."""
    ep_size, num_tables = 4, len(_TABLE_SIZES)
    # A stand-in for a four-rank row group; nothing here communicates over it.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    group = SimpleNamespace(rank=lambda: 1, size=lambda: ep_size)
    device = torch.device("cpu")
    embedding = _embedding(group, device)
    hash_ids = _hash_ids("random", ep_size, 1, device)
    calls = _record_all_to_alls(monkeypatch)

    routed = route(embedding, hash_ids)

    assert calls == []
    assert routed.shape == hash_ids.shape
    rows = hash_ids.reshape(-1).tolist()
    tables = [index % num_tables for index in range(len(rows))]
    ranges = [
        [get_contiguous_row_range(size, owner, ep_size) for owner in range(ep_size)]
        for size in _TABLE_SIZES
    ]
    owners = [
        next(owner for owner, (start, end) in enumerate(ranges[table]) if start <= row < end)
        for row, table in zip(rows, tables)
    ]
    order = sorted(range(len(rows)), key=owners.__getitem__)
    assert routed.owner_order.tolist() == order
    assert routed.requests.tolist() == [
        [tables[index], rows[index] - ranges[tables[index]][owners[index]][0]] for index in order
    ]
    requests = list(zip(owners, tables))
    assert routed.send_counts.view(ep_size, num_tables).tolist() == [
        [requests.count((owner, table)) for table in range(num_tables)] for owner in range(ep_size)
    ]

    with pytest.raises(RuntimeError, match="counts were exchanged"):
        lookup_routed(embedding, routed)
    with pytest.raises(ValueError, match="hash heads"):
        route(embedding, hash_ids[..., :-1])
