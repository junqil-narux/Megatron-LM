# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Asynchronous Engram lookup (``--engram-async-lookup``).

The memory an Engram layer adds depends only on the token IDs; only its dense half reads the
hidden states. So the lookup (hashing, owner routing, the request and return all-to-alls and
the owner-side gather) is issued before the embedding, on a dedicated stream and a dedicated
communicator, and runs while the layers in front of the Engram layer compute:

* ``plan_engram_lookups`` routes every microbatch of a step at the start of the step and
  exchanges all their request counts with one all-to-all, so a step pays one host
  synchronization for the counts instead of one per microbatch and Engram layer.
* ``prefetch_engram_lookups`` issues each local Engram layer's lookup on the lookup stream at
  the start of ``HybridModel.forward``; ``use_prefetched_lookup`` hands the result to the layer,
  whose ``Engram.forward`` then runs only its dense half. Under full recompute the prefetched
  memory stays bound to its microbatch's checkpoint, so the recompute does not repeat the
  lookup or its all-to-alls.
* In backward, the table-gradient branch (return all-to-all, owner-side embedding backward,
  gradient accumulation) is launched on the lookup stream as soon as the Engram layer's
  backward produced the memory's gradient, and overlaps the backward of the layers in front.
  ``join_lookup_stream_in_backward`` makes the compute stream wait for it once the backward
  reaches the embedding.

Every result is bitwise identical to the inline lookup: the same operations run on the same
data, only on another stream and communicator.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

import torch
from torch import Tensor

from megatron.core.utils import nvtx_range_pop, nvtx_range_push

from .distributed_embedding import EngramRoute, exchange_route_counts

_LOOKUP_STREAM: Optional[torch.cuda.Stream] = None
_LOOKUP_GROUPS: dict[tuple[int, ...], torch.distributed.ProcessGroup] = {}


def get_lookup_stream() -> torch.cuda.Stream:
    """The stream every asynchronous Engram lookup and its backward run on."""
    global _LOOKUP_STREAM
    if _LOOKUP_STREAM is None:
        # High priority: the lookup's kernels are few and short, and the Engram layer waits on
        # them, so let them through ahead of the layers they overlap.
        _LOOKUP_STREAM = torch.cuda.Stream(priority=-1)
    return _LOOKUP_STREAM


def create_lookup_groups(group: torch.distributed.ProcessGroup) -> None:
    """Create the lookup communicator: a second group over the ranks of ``group``.

    The lookup's all-to-alls run on the lookup stream. NCCL serializes operations on one
    communicator, so sharing the EP communicator with the MoE all-to-alls on the compute stream
    would make the lookup wait behind them; a dedicated communicator lets both proceed.

    Collective over the whole job: every rank must call this, with the same kind of group, the
    same number of times (every rank builds the group of every peer group, in the same order,
    as ``parallel_state`` does). Later calls with an already created group are no-ops. Each
    new communicator is initialized right away, on all ranks together, rather than lazily in
    the middle of a step while other all-to-alls are in flight.
    """
    from megatron.core.parallel_state import create_group

    ranks = tuple(torch.distributed.get_process_group_ranks(group))
    if ranks in _LOOKUP_GROUPS:
        return
    all_ranks = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(all_ranks, ranks)
    backend = torch.distributed.get_backend(group)
    own_group = None
    for group_ranks in sorted(set(all_ranks)):
        new_group = create_group(
            list(group_ranks), backend=backend, group_desc="ENGRAM_LOOKUP_GROUP"
        )
        if group_ranks == ranks:
            own_group = new_group
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if str(backend) == "nccl"
        else torch.device("cpu")
    )
    torch.distributed.all_reduce(torch.zeros(1, device=device), group=own_group)
    _LOOKUP_GROUPS[ranks] = own_group


def get_lookup_group(group: torch.distributed.ProcessGroup) -> torch.distributed.ProcessGroup:
    """The lookup communicator over the ranks of ``group`` (see ``create_lookup_groups``)."""
    ranks = tuple(torch.distributed.get_process_group_ranks(group))
    try:
        return _LOOKUP_GROUPS[ranks]
    except KeyError:
        raise RuntimeError(
            "The Engram lookup communicator was not created; HybridModel creates it when "
            "engram_async_lookup is set."
        ) from None


def destroy_lookup_groups() -> None:
    """Forget the cached lookup communicators (tests re-initialize model parallelism)."""
    _LOOKUP_GROUPS.clear()


class _EngramLookupHandoff(torch.autograd.Function):
    """Identity that places a prefetched lookup at the Engram layer in the autograd graph.

    The lookup's own graph was recorded before the embedding, so autograd would run its
    backward (``sequence_nr`` order) only after the backward of every layer in front of the
    Engram layer, and the end of that backward would then make the compute stream wait for it.
    This node is created where the Engram layer consumes the memory instead. Its input is a
    detached alias of the memory; its backward runs the lookup's graph itself, with the lookup
    stream current, as soon as the Engram layer's backward produced the memory's gradient.
    """

    @staticmethod
    def forward(ctx, memory: Tensor, handle: "EngramLookupHandle") -> Tensor:
        """Wait for the lookup on the current stream and return the memory."""
        handle.wait()
        ctx.handle = handle
        return memory.view_as(memory)

    @staticmethod
    def backward(ctx, grad_memory: Tensor):
        """Launch the lookup's backward on the lookup stream."""
        handle, ctx.handle = ctx.handle, None
        stream = get_lookup_stream()
        stream.wait_stream(torch.cuda.current_stream())
        grad_memory.record_stream(stream)
        message = "engram.lookup.backward"
        nvtx_range_push(message)
        try:
            with torch.cuda.stream(stream):
                torch.autograd.backward(handle.memory, grad_memory)
        finally:
            nvtx_range_pop(message)
        handle.memory = None
        return None, None


class EngramLookupHandle:
    """A lookup issued on the lookup stream, for one Engram layer and one microbatch.

    Attributes:
        memory: ``[seq, batch, memory_dim]`` memory with the lookup's autograd graph.
        ready: Event recorded on the lookup stream once ``memory`` is complete.
    """

    def __init__(self, memory: Tensor, ready: torch.cuda.Event) -> None:
        self.memory = memory
        self.ready = ready

    def wait(self) -> None:
        """Make the current stream wait for the lookup and keep ``memory`` alive for it."""
        stream = torch.cuda.current_stream()
        stream.wait_event(self.ready)
        self.memory.record_stream(stream)

    def attach(self) -> Tensor:
        """Return the memory for the Engram layer's dense half, on the current stream.

        With autograd recording, the result's backward launches the lookup's backward (see
        ``_EngramLookupHandoff``). Under full recompute this runs twice: without autograd in
        the checkpointed forward, and with it in the recompute.
        """
        if self.memory is None:
            raise RuntimeError("This Engram lookup was already consumed by a backward pass.")
        if torch.is_grad_enabled() and self.memory.requires_grad:
            return _EngramLookupHandoff.apply(self.memory.detach().requires_grad_(), self)
        self.wait()
        return self.memory


def _engram_of(layer: torch.nn.Module):
    """The Engram module of a decoder layer (wrapped by a hyper-connection layer or not)."""
    inner = getattr(layer, "inner_layer", layer)
    return getattr(inner, "engram", None)


def local_engram_modules(modules: Iterable[torch.nn.Module]) -> list:
    """The Engram modules of ``modules`` (decoder layers), in layer order."""
    return [engram for engram in map(_engram_of, modules) if engram is not None]


@contextmanager
def use_prefetched_lookup(layer: torch.nn.Module, lookups: Optional[dict]):
    """Hand ``layer``'s prefetched lookup (if any) to its Engram module for the call."""
    engram = _engram_of(layer)
    handle = None if engram is None or not lookups else lookups.get(engram.layer_number)
    if handle is None:
        yield
        return
    previous = engram.prefetched_lookup
    engram.prefetched_lookup = handle
    try:
        yield
    finally:
        engram.prefetched_lookup = previous


def prefetch_engram_lookups(
    engrams: list, input_ids: Tensor, packed_seq_params=None
) -> dict[int, EngramLookupHandle]:
    """Issue the lookup of every Engram module in ``engrams`` for this microbatch.

    Uses the routes ``plan_engram_lookups`` prepared for the step, in microbatch order; a
    module without a planned route routes and exchanges its counts now (one synchronization
    with the lookup stream). Returns the handles keyed by layer number.
    """
    return {
        engram.layer_number: engram.prefetch(
            input_ids,
            packed_seq_params,
            route=engram.planned_routes.popleft() if engram.planned_routes else None,
        )
        for engram in engrams
    }


def join_lookup_stream_in_backward(tensor: Tensor) -> None:
    """Make the compute stream wait for the lookup stream when backward reaches ``tensor``.

    Register on the embedding output: by then every Engram layer's backward has launched its
    lookup's backward on the lookup stream, and the gradient accumulation it does must be
    complete before the gradients are reduced and the optimizer reads them.
    """
    if not tensor.requires_grad:
        raise RuntimeError(
            "Asynchronous Engram lookups need the embedding output to require grad: the "
            "compute stream joins the lookup stream when backward reaches it."
        )

    def _join(grad: Tensor) -> Tensor:
        torch.cuda.current_stream().wait_stream(get_lookup_stream())
        return grad

    tensor.register_hook(_join)


def plan_engram_lookups(model_chunks: Iterable[torch.nn.Module], token_batches: list) -> None:
    """Route every microbatch of a step for every training Engram module in ``model_chunks``.

    All the request counts go through one all-to-all and one host transfer; the routes are
    queued on each module and consumed, in microbatch order, by ``prefetch_engram_lookups``.
    Collective over the EP group: every EP rank calls this with the same number of batches.

    Args:
        model_chunks: This rank's model chunks.
        token_batches: The step's ``[batch, seq]`` token IDs, one per microbatch, before any
            context-parallel split (the token prefetch's rows).
    """
    engrams = [
        module
        for chunk in model_chunks
        for module in chunk.modules()
        if getattr(module, "supports_async_lookup", False) and module.training
    ]
    for engram in engrams:
        # A re-run step (rerun state machine) starts over.
        engram.planned_routes.clear()
    engrams = [engram for engram in engrams if engram.embedding.ep_size > 1]
    if not engrams or not token_batches:
        return
    first = engrams[0].embedding
    for engram in engrams[1:]:
        if engram.embedding.ep_size != first.ep_size or len(engram.embedding.tables) != len(
            first.tables
        ):
            raise ValueError("All Engram layers must route the same tables over the same EP group.")

    stream = get_lookup_stream()
    stream.wait_stream(torch.cuda.current_stream())
    routes: list[tuple[object, EngramRoute]] = []
    message = "engram.lookup.plan"
    nvtx_range_push(message)
    try:
        with torch.cuda.stream(stream):
            for tokens in token_batches:
                tokens.record_stream(stream)
            for engram in engrams:
                for tokens in token_batches:
                    routes.append((engram, engram.route_global_tokens(tokens)))
            exchange_route_counts(
                [route for _, route in routes],
                first.ep_size,
                len(first.tables),
                get_lookup_group(first.ep_group),
            )
    finally:
        nvtx_range_pop(message)
    for engram, route in routes:
        engram.planned_routes.append(route)
