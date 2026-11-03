# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The Engram request counts come from a device-side scatter-add instead of ``torch.bincount``.

``torch.bincount`` reads its input's minimum and maximum on the host, two device
synchronizations per lookup. ``count_requests_per_bucket`` must return the same counts without
them, so the EP lookup exchanges the same splits and returns the same rows and gradients bit for
bit while synchronizing once (the ``.tolist()`` of the exchanged counts) instead of three times.

The count tests run on a single process (CPU, and CUDA when present); the lookup tests run
under torchrun with two, four or eight ranks.
"""

import os
import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram import distributed_embedding
from megatron.core.models.engram.distributed_embedding import (
    EPShardedMultiTableEmbedding,
    count_requests_per_bucket,
    get_contiguous_row_range,
)

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_DISTRIBUTED = int(os.environ.get("WORLD_SIZE", "1")) in (2, 4, 8)

_BIT_VIEWS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}


def _bincount_reference(buckets, num_buckets):
    """The replaced implementation."""
    return torch.bincount(buckets, minlength=num_buckets).to(torch.int64)


@contextmanager
def _deterministic_algorithms(enabled):
    """Mirror --deterministic-mode, which turns on torch.use_deterministic_algorithms."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@contextmanager
def _sync_debug_mode(mode):
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(mode)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def _bucket_cases(device):
    """Bucket tensors covering skew, unused buckets, both ends of the range and no requests."""
    generator = torch.Generator().manual_seed(1234)
    num_buckets = 4 * 16  # e.g. EP4 x 16 tables
    uniform = torch.randint(0, num_buckets, (65536,), generator=generator)
    # Half the requests land in three buckets; most other buckets get a few or none.
    skewed = torch.cat(
        (
            torch.tensor([0, 17, num_buckets - 1]).repeat(20000),
            torch.randint(0, num_buckets, (500,), generator=generator),
        )
    )
    cases = {
        "uniform": uniform,
        "skewed": skewed,
        "one_bucket": torch.full((4096,), 5),
        "last_bucket_only": torch.full((7,), num_buckets - 1),
        "first_bucket_only": torch.zeros(3, dtype=torch.int64),
        "empty": torch.empty(0, dtype=torch.int64),
    }
    return {name: (tensor.to(device), num_buckets) for name, tensor in cases.items()}


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("device", _DEVICES)
def test_counts_match_bincount(device, deterministic):
    with _deterministic_algorithms(deterministic):
        for name, (buckets, num_buckets) in _bucket_cases(device).items():
            counts = count_requests_per_bucket(buckets, num_buckets)
            expected = _bincount_reference(buckets, num_buckets)
            assert counts.dtype == torch.int64, name
            assert counts.shape == (num_buckets,), name
            assert counts.device == buckets.device, name
            assert torch.equal(counts, expected), name
            assert int(counts.sum()) == buckets.numel(), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("deterministic", [False, True])
def test_counts_do_not_synchronize(deterministic):
    cases = _bucket_cases("cuda")
    torch.cuda.synchronize()
    with _deterministic_algorithms(deterministic), _sync_debug_mode("error"):
        for name, (buckets, num_buckets) in cases.items():
            count_requests_per_bucket(buckets, num_buckets)
        # The check is live: the replaced bincount does synchronize.
        buckets, num_buckets = cases["uniform"]
        with pytest.raises(RuntimeError, match="synchroniz"):
            _bincount_reference(buckets, num_buckets)
    torch.cuda.synchronize()


def _bits(tensor):
    tensor = tensor.detach().contiguous()
    view = _BIT_VIEWS.get(tensor.dtype)
    return tensor if view is None else tensor.view(view)


def _assert_bitwise_equal(actual, expected, what):
    assert actual.dtype == expected.dtype, f"{what}: {actual.dtype} vs {expected.dtype}"
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    assert torch.equal(_bits(actual), _bits(expected)), f"{what} differs"


def _hash_ids(case, table_sizes, ep_size, rank, device):
    """Global row IDs ``[batch, seq, tables]`` for a lookup case, different on every rank."""
    if case == "empty":
        return torch.empty((1, 0, len(table_sizes)), dtype=torch.int64, device=device)
    generator = torch.Generator().manual_seed(4321 + rank)
    batch, seq = 2, 37
    # "first_owner_only": every request lands on EP rank 0, so the other peers receive and
    # return nothing and every rank sends empty splits to them.
    limits = [
        get_contiguous_row_range(size, 0, ep_size)[1] if case == "first_owner_only" else size
        for size in table_sizes
    ]
    columns = [torch.randint(0, limit, (batch, seq), generator=generator) for limit in limits]
    return torch.stack(columns, dim=-1).to(device)


@contextmanager
def _record_all_to_all(calls):
    """Record every all-to-all's splits and received buffer (count, request and return)."""
    real = torch.distributed.all_to_all_single

    def spy(output, input, output_split_sizes=None, input_split_sizes=None, **kwargs):
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

    torch.distributed.all_to_all_single = spy
    try:
        yield
    finally:
        torch.distributed.all_to_all_single = real


def _count_synchronizations(fn):
    """Run ``fn`` and return (result, number of synchronizing CUDA calls it made)."""
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught, _sync_debug_mode("warn"):
        warnings.simplefilter("always")
        result = fn()
    torch.cuda.synchronize()
    return result, sum("synchroniz" in str(warning.message) for warning in caught)


def _run_lookup(embedding, hash_ids, grad_seed):
    for table in embedding.tables:
        table.weight.grad = None
    calls = []
    with _record_all_to_all(calls):
        output, forward_syncs = _count_synchronizations(lambda: embedding(hash_ids))
        grad_output = torch.randn(
            output.shape,
            generator=torch.Generator(device=output.device).manual_seed(grad_seed),
            device=output.device,
            dtype=output.dtype,
        )
        output.backward(grad_output)
    grads = [table.weight.grad.detach().clone() for table in embedding.tables]
    return output.detach().clone(), grads, calls, forward_syncs


@pytest.mark.skipif(not _DISTRIBUTED, reason="requires torchrun with two, four, or eight ranks")
@pytest.mark.parametrize("case", ["random", "first_owner_only", "empty"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("ep_size", [2, 4])
def test_lookup_matches_bincount_path(ep_size, deterministic, case, monkeypatch):
    """Same splits, counts, rows and table gradients as the bincount path, with 2 fewer syncs."""
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_distributed()
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if world_size < ep_size:
        pytest.skip(f"requires at least {ep_size} ranks")
    # Collective on the default group: every rank must participate in group creation.
    group = torch.distributed.new_group(ranks=list(range(ep_size)))
    if rank >= ep_size:
        return

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    # One tiny table (a few rows per shard) and two prime-sized ones.
    table_sizes = (13, 1009, 997)
    config = SimpleNamespace(
        use_cpu_initialization=False,
        params_dtype=torch.float32,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=deterministic,
    )
    embedding = EPShardedMultiTableEmbedding(
        config=config,
        table_sizes=table_sizes,
        embedding_dim=16,
        init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1),
        ep_group=group,
        tp_group=None,
        expt_dp_group=None,
    )
    hash_ids = _hash_ids(case, table_sizes, ep_size, rank, device)

    with _deterministic_algorithms(deterministic):
        output, grads, calls, forward_syncs = _run_lookup(embedding, hash_ids, grad_seed=rank)
        monkeypatch.setattr(distributed_embedding, "count_requests_per_bucket", _bincount_reference)
        ref_output, ref_grads, ref_calls, ref_forward_syncs = _run_lookup(
            embedding, hash_ids, grad_seed=rank
        )

    _assert_bitwise_equal(output, ref_output, "lookup output")
    for table_id, (grad, ref_grad) in enumerate(zip(grads, ref_grads)):
        _assert_bitwise_equal(grad, ref_grad, f"table {table_id} gradient")
    # Count, request and return exchanges forward, return exchange backward.
    assert len(calls) == len(ref_calls) == 4
    for index, ((splits_out, splits_in, received), (ref_out, ref_in, ref_received)) in enumerate(
        zip(calls, ref_calls)
    ):
        assert splits_out == ref_out, f"all-to-all {index} output splits"
        assert splits_in == ref_in, f"all-to-all {index} input splits"
        _assert_bitwise_equal(received, ref_received, f"all-to-all {index} received buffer")
    # The exchanged counts themselves (first all-to-all) sum to this rank's received requests.
    assert int(calls[0][2].sum()) == sum(calls[1][0])
    # bincount synchronized twice per lookup (it returns early, without reading the input, when
    # there are no requests); only the counts' .tolist() remains.
    expected_saved = 2 if hash_ids.numel() > 0 else 0
    assert ref_forward_syncs - forward_syncs == expected_saved, (forward_syncs, ref_forward_syncs)
    assert forward_syncs == 1, forward_syncs
