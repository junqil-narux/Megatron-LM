# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Gradient streams in DDP.

A parameter whose gradient is produced on a CUDA stream of its own names that stream
(``grad_accumulation_stream``). DDP creates the parameter's gradient accumulator on it, so the
accumulation and DDP's hook run there, and on the stream DDP was built on for the other
parameters. A bucket group whose parameters accumulate on several streams waits for all of
them before it reduces its buckets, and again before ``finish_grad_sync`` returns, so that its
caller reads complete gradients also when the reduction is the identity and launches nothing.

The cases need CUDA; they run with any number of ranks (one GPU per rank), the reduction over
the data-parallel group with two or more::

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/distributed/test_gradient_streams.py
"""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils, release_process_groups

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")

_SHAPE = (24, 11)
_NAMES = ("first", "second")
_STEPS = 2


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


def _coefficients(rank, step):
    """Integer coefficients of the two weights on one rank for one step: their gradients."""
    generator = torch.Generator().manual_seed(100 * step + rank)
    return {
        name: torch.randint(-8, 9, _SHAPE, generator=generator).float().cuda() for name in _NAMES
    }


def _occupy(stream):
    """Queue work on ``stream`` that outlasts the dispatch of a gradient reduction by far."""
    with torch.cuda.stream(stream):
        work = torch.eye(4096, device="cuda")
        for _ in range(16):
            work = work @ work


def _reduced_gradients(construction_stream, named=None, gradient_stream=None, identity=False):
    """Reduce the gradients of ``_STEPS`` steps over the data-parallel group, overlapped.

    DDP is built on ``construction_stream``. The weight ``named`` names ``gradient_stream`` and
    has its loss term, hence its gradient, computed there; the other weights are computed on
    ``construction_stream``. With ``identity`` the bucket group skips its collectives, so each
    rank keeps its own scaled gradients. Returns each step's reduced gradients, read on the
    calling stream right after ``finish_grad_sync``.

    The backward is called on a stream of its own. At its end, autograd makes the stream it
    was called on wait for the streams of the gradient accumulators; called on the stream that
    reads the gradients, it would order that stream after them whatever DDP does.
    """
    rank = torch.distributed.get_rank()
    streams = {name: gradient_stream if name == named else construction_stream for name in _NAMES}
    model = _TwoWeights().cuda()
    if named is not None:
        getattr(model, named).grad_accumulation_stream = gradient_stream
    with torch.cuda.stream(construction_stream):
        ddp_model = _RecordingDDP(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(overlap_grad_reduce=True),
            model,
        )
    (bucket_group,) = ddp_model.bucket_groups
    assert bucket_group.params == set(model.parameters())
    if identity:
        # What DDP sets on the bucket groups of a parameter domain whose replica group has a
        # single rank. Set here, so that the case needs no such domain and runs on one rank.
        bucket_group.collectives_are_identity = True

    backward_stream = torch.cuda.Stream()
    # The gradient the backward starts from. The host waits for it before the first backward.
    one = torch.ones((), device="cuda")
    reduced = []
    for step in range(_STEPS):
        coefficients = _coefficients(rank, step)
        ddp_model.zero_grad_buffer()
        terms = []
        for name in _NAMES:
            with torch.cuda.stream(streams[name]):
                terms.append((getattr(model, name) * coefficients[name]).sum())
        # Nothing is pending on any stream: what they run next is the backward.
        torch.cuda.synchronize()
        if step > 0:
            # From the second step on, the hook of the weight whose gradient comes last
            # dispatches the reduction, on its stream. Hold the other stream back, so that its
            # gradient is not in the bucket yet when that hook runs.
            dispatching_stream = ddp_model.hook_calls[-1][1]
            held_back = [streams[name] for name in _NAMES if streams[name] != dispatching_stream]
            if identity and gradient_stream is not None and gradient_stream not in held_back:
                # An identity reduction launches no collective whose handle the caller could
                # wait for. Hold the named stream back whether it dispatches or not, so that
                # its gradient is still to be accumulated when finish_grad_sync is called.
                held_back.append(gradient_stream)
            for stream in held_back:
                _occupy(stream)
        with torch.cuda.stream(backward_stream):
            torch.autograd.backward(terms, [one] * len(terms))
        ddp_model.finish_grad_sync()
        reduced.append({name: getattr(model, name).main_grad.clone() for name in _NAMES})
    return reduced


@pytest.mark.parametrize("named", _NAMES)
def test_bucket_is_reduced_after_every_stream_accumulated_into_it(named):
    """A bucket group whose weights accumulate on two streams reduces like one on one stream.

    The weight whose gradient comes last dispatches the reduction on its own stream, while the
    gradient of the other weight is still held back on the other stream. Over the two cases,
    the weight that names its stream is once the last one and once the other one.
    """
    if Utils.world_size < 2:
        pytest.skip("a reduction over one rank is the identity")
    Utils.initialize_model_parallel()
    reference = _reduced_gradients(torch.cuda.current_stream())
    reduced = _reduced_gradients(torch.cuda.Stream(), named, torch.cuda.Stream())
    for step in range(_STEPS):
        for name in _NAMES:
            assert torch.equal(reduced[step][name], reference[step][name]), (step, name)


@pytest.mark.parametrize("named", _NAMES)
def test_identity_reduction_is_complete_when_finish_grad_sync_returns(named):
    """A bucket group that skips its collectives hands its caller complete gradients.

    Nothing is communicated, so no handle orders the calling stream after the streams that
    accumulate and scale the gradients: ``finish_grad_sync`` waits for them itself. The stream
    the weight names is held back when the backward starts, the backward is called on a third
    stream, and the gradients are read on the calling stream right after ``finish_grad_sync``,
    with no synchronization in between. Each rank expects its own gradients, scaled as for a
    sum over the data-parallel group; one rank is enough.
    """
    Utils.initialize_model_parallel()
    reduced = _reduced_gradients(torch.cuda.Stream(), named, torch.cuda.Stream(), identity=True)
    scale = 1.0 / torch.distributed.get_world_size()
    for step in range(_STEPS):
        expected = _coefficients(torch.distributed.get_rank(), step)
        for name in _NAMES:
            assert torch.equal(reduced[step][name], expected[name] * scale), (step, name)
