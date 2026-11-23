# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Device-side request counts for the lookups of EP-sharded Engram tables.

A lookup over a row group sends every requested row to its owner and needs the number of
requests per (owner, table) for the all-to-all splits. ``EPShardedMultiTableEmbedding.forward``
(``distributed_embedding``) counts them with ``torch.bincount``;
:func:`count_requests_per_bucket` returns the same counts without the host reads of
``bincount``.
"""

from __future__ import annotations

import torch
from torch import Tensor


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
