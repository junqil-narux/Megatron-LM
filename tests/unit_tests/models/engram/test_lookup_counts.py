# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram request counts from a device-side scatter-add instead of ``torch.bincount``.

``torch.bincount`` reads its input's minimum and maximum on the host, two device
synchronizations per call. ``count_requests_per_bucket`` must return the same counts without
them.

The cases run in a single process, on CPU and on CUDA when present.
"""

import pytest
import torch

from megatron.core.models.engram.routed_lookup import count_requests_per_bucket

from ._test_utils import deterministic_algorithms, sync_debug_mode

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bincount_reference(buckets, num_buckets):
    return torch.bincount(buckets, minlength=num_buckets).to(torch.int64)


def _bucket_cases(device):
    """Bucket tensors covering skew, unused buckets, both ends of the range and no requests."""
    generator = torch.Generator().manual_seed(1234)
    num_buckets = 4 * 16  # e.g. four owners x 16 tables
    uniform = torch.randint(0, num_buckets, (65536,), generator=generator)
    # Three buckets hold 60000 of the 60500 requests; the other 500 are uniform over all buckets.
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
    with deterministic_algorithms(deterministic):
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
    with deterministic_algorithms(deterministic), sync_debug_mode("error"):
        for name, (buckets, num_buckets) in cases.items():
            count_requests_per_bucket(buckets, num_buckets)
        # The check is live: bincount does synchronize.
        buckets, num_buckets = cases["uniform"]
        with pytest.raises(RuntimeError, match="synchroniz"):
            _bincount_reference(buckets, num_buckets)
    torch.cuda.synchronize()
