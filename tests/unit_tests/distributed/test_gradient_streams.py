# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gradient streams in DDP.

A parameter whose gradient is produced on a CUDA stream of its own names that stream
(``grad_accumulation_stream``). DDP creates the parameter's gradient accumulator on it, so the
accumulation and DDP's hook run there, and on the stream DDP was built on for the other
parameters.

The cases need CUDA; they run with any number of ranks (one GPU per rank)::

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/distributed/test_gradient_streams.py
"""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils, release_process_groups

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")

_SHAPE = (24, 11)


@pytest.fixture(autouse=True)
def _release_process_groups():
    """Every case sets up the model-parallel groups; see release_process_groups."""
    with release_process_groups():
        yield


class _TwoWeights(torch.nn.Module):
    """Two weights whose gradients do not depend on each other."""

    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(_SHAPE))
        self.second = torch.nn.Parameter(torch.zeros(_SHAPE))


class _RecordingDDP(DistributedDataParallel):
    """DDP that records each call of its backward hook: the parameter and the current stream."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_calls = []

    def _make_backward_post_hook(self, param):
        hook = super()._make_backward_post_hook(param)

        def recording_hook(*unused):
            self.hook_calls.append((param, torch.cuda.current_stream()))
            return hook(*unused)

        return recording_hook


def test_gradient_accumulator_runs_on_the_stream_its_parameter_names():
    """DDP's hook runs on the stream a parameter names; without a name, on the stream DDP was
    built on. Neither is the stream the backward is called on."""
    Utils.initialize_model_parallel()
    gradient_stream, construction_stream = torch.cuda.Stream(), torch.cuda.Stream()
    model = _TwoWeights().cuda()
    model.first.grad_accumulation_stream = gradient_stream
    with torch.cuda.stream(construction_stream):
        ddp_model = _RecordingDDP(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(),
            model,
        )

    (model.first.sum() + model.second.sum()).backward()

    assert len(ddp_model.hook_calls) == 2
    hook_streams = dict(ddp_model.hook_calls)
    assert hook_streams[model.first] == gradient_stream
    assert hook_streams[model.second] == construction_stream
    assert torch.cuda.current_stream() not in (gradient_stream, construction_stream)
