# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Asynchronous Engram lookup (``EngramConfig.async_lookup``).

The memory an Engram layer adds depends only on the token IDs; only the fusion reads the hidden
states. So the lookup (hashing, owner routing, the request and return all-to-alls and the
owner-side table lookup) can be issued ahead of the layer, on a stream and a communicator of
its own, and run while the layers in front of the Engram layer compute:

* :func:`plan_engram_lookups` routes every microbatch of a step at the start of the step and
  exchanges all their request counts with one all-to-all, so a step pays one host
  synchronization for the counts instead of at least one per microbatch and Engram layer.
* :func:`prefetch_engram_lookups` issues each local Engram layer's lookup on the lookup stream,
  and :func:`use_prefetched_lookup` binds the result to the layer for its call, where
  ``Engram.forward`` then runs only the fusion. A layer that is recomputed is called under the
  same binding again, so the recompute repeats neither the lookup nor its all-to-alls.
* In backward, the table-gradient branch (return all-to-all, owner-side table backward,
  gradient accumulation) is launched on the lookup stream as soon as the Engram layer's
  backward produced the memory's gradient, and overlaps the backward of the layers in front.
  :func:`join_lookup_stream_in_backward` makes the compute stream wait for it once the backward
  reaches the tensor it is registered on.

Every result is bitwise identical to the inline lookup: the same operations run on the same
data, on another stream and communicator. The lookups are the routed NCCL lookups of
``routed_lookup``, over the row group of the tables in either table layout; the lookup
communicators are created by :func:`initialize_engram_lookup_groups` before the model is built.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, nvtx_range_pop, nvtx_range_push

from .config import EngramConfig
from .routed_lookup import EngramRoute, exchange_route_counts
from .table_sharding import _group_timeout, resolve_engram_groups

_LOOKUP_STREAM: Optional[torch.cuda.Stream] = None

# Lookup communicators by the global ranks of the row group they serve, in group order. Every
# rank creates them in the same call, so every rank holds its key from that call on.
_LOOKUP_GROUPS: Dict[Tuple[int, ...], ProcessGroup] = {}

_GROUP_DESCRIPTION = "ENGRAM_LOOKUP"


def get_lookup_stream() -> torch.cuda.Stream:
    """Return the stream every asynchronous Engram lookup and its backward run on."""
    global _LOOKUP_STREAM
    if _LOOKUP_STREAM is None:
        # High priority: the lookup's kernels are few and short, and the Engram layer waits on
        # them, so let them through ahead of the layers they overlap.
        _LOOKUP_STREAM = torch.cuda.Stream(priority=-1)
    return _LOOKUP_STREAM


def initialize_engram_lookup_groups(
    pg_collection: ProcessGroupCollection,
    engram_config: Optional[EngramConfig],
    *,
    timeout: Optional[timedelta] = None,
) -> None:
    """Create the communicator of the asynchronous lookups: a second group over the row group.

    The lookup's all-to-alls run on the lookup stream. NCCL serializes the operations of one
    communicator, so on the row group's own communicator they would queue behind what that
    group carries on the compute stream: the MoE all-to-alls of an expert-parallel group, the
    gradient reductions of a dp_cp group. A second communicator over the same ranks lets both
    proceed.

    Collective over the default group when ``engram_config.async_lookup`` is set, and a no-op
    otherwise: every rank calls it at the same point, after
    :func:`~megatron.core.models.engram.table_sharding.initialize_engram_parallel_state` and
    before it builds any Engram layer, including pipeline stages without one. The ranks
    exchange their row groups first, so a bad set raises on every rank, and every rank then
    creates the lookup group of every row group in the same order, as
    ``torch.distributed.new_group`` requires.

    The ranks of the first pipeline stage run one collective on their new group right away:
    NCCL creates a communicator at the first collective of a group, and that must not be a
    lookup in the middle of a step, while other all-to-alls are in flight. Only that stage can
    hold the embedding, where a model issues the lookups ahead, and all ranks of a row group
    are on one stage, so a group runs the collective as a whole or not at all. A model issues
    no lookup from another stage, so the groups of the other stages never communicate and
    never get a communicator.

    A row group of one rank looks its rows up locally and gets no lookup group. Calling this
    again for row groups that have their lookup groups creates nothing.

    Args:
        pg_collection (ProcessGroupCollection): The groups the model is built with.
        engram_config (EngramConfig, optional): The Engram configuration; None (no Engram)
            returns at once.
        timeout (timedelta, optional): Timeout of the new groups. None gives them the timeout
            of the row group, or the backend's default when this torch does not expose it.
    """
    if engram_config is None or not engram_config.async_lookup:
        return
    if not torch.distributed.is_initialized():
        return
    if torch.distributed.get_backend() == "fake":
        # The lookup communicator spans the ranks of the row group, which a fake group never
        # starts: its first collective would wait forever.
        raise ValueError(
            "The asynchronous Engram lookup creates a communicator of its own and cannot run on "
            "a fake process group."
        )
    row_group = resolve_engram_groups(pg_collection, engram_config)[0]
    ranks = _group_ranks(row_group)
    if ranks in _LOOKUP_GROUPS:
        return
    backend = torch.distributed.get_backend(row_group)
    world_size = torch.distributed.get_world_size()
    descriptions: list = [None] * world_size
    torch.distributed.all_gather_object(descriptions, (ranks, str(backend)))

    # Validate only after the exchange: a bad set then raises on every rank instead of leaving
    # its peers waiting in the group creation below.
    row_groups = sorted({description[0] for description in descriptions})
    if (
        any(description[1] != descriptions[0][1] for description in descriptions)
        or len({len(group_ranks) for group_ranks in row_groups}) != 1
        or sorted(rank for group_ranks in row_groups for rank in group_ranks)
        != list(range(world_size))
    ):
        raise ValueError(
            "The Engram row groups must have one backend and one size and partition all "
            f"{world_size} ranks, got {sorted(set(descriptions))}."
        )
    if len(ranks) == 1:
        return
    if timeout is None:
        timeout = _group_timeout(row_group)
    lookup_group = None
    for group_ranks in row_groups:
        group = parallel_state.create_group(
            list(group_ranks), timeout=timeout, backend=backend, group_desc=_GROUP_DESCRIPTION
        )
        if group_ranks == ranks:
            lookup_group = group
    if get_pg_rank(getattr(pg_collection, "pp", None)) == 0:
        # This rank's stage can hold the embedding: have NCCL create the communicator now.
        device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else None
        torch.distributed.all_reduce(torch.zeros(1, device=device), group=lookup_group)
    # Every rank keeps its group, whether it communicated on it or not, so that a later call
    # returns above on every rank alike.
    _LOOKUP_GROUPS[ranks] = lookup_group


def get_lookup_group(row_group: ProcessGroup) -> ProcessGroup:
    """Return the lookup communicator over the ranks of ``row_group``.

    :func:`initialize_engram_lookup_groups` must have created it.
    """
    try:
        return _LOOKUP_GROUPS[_group_ranks(row_group)]
    except KeyError:
        raise RuntimeError(
            "The asynchronous Engram lookup has no communicator over this row group: call "
            "initialize_engram_lookup_groups(pg_collection, engram_config) on every rank before "
            "building the model."
        ) from None


def destroy_lookup_groups() -> None:
    """Forget the cached lookup communicators.

    ``clear_engram_parallel_state`` calls it with the teardown of the model-parallel state, so
    the next setup creates its own. Modules that already hold a group keep their reference.
    """
    _LOOKUP_GROUPS.clear()


def _group_ranks(group: Optional[ProcessGroup]) -> Tuple[int, ...]:
    """Global ranks of ``group`` in group order; this rank alone without a group."""
    if group is None:
        return (torch.distributed.get_rank(),)
    return tuple(torch.distributed.get_process_group_ranks(group))


class _EngramLookupHandoff(torch.autograd.Function):
    """Identity that places a prefetched lookup at the Engram layer in the autograd graph.

    The lookup's own graph is recorded ahead of the layers in front of the Engram layer, so
    autograd would run its backward only after theirs, and the end of that backward would then
    make the calling stream wait for it. This node is created where the Engram layer consumes
    the memory instead. Its input is a detached alias of the memory; its backward runs the
    lookup's graph itself, with the lookup stream current, as soon as the Engram layer's
    backward produced the memory's gradient.
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

    It carries the fields of an ``EngramLookupResult``; ``Engram.forward`` rebuilds the result
    from :meth:`attach`, ``valid`` and ``cu_seqlens``.

    Attributes:
        memory: The ``[S_local, B, total_memory_dim]`` rows, with the lookup's autograd graph
            when the tables train. None once a backward pass consumed the lookup.
        valid: The ``[1, S_local, 1]`` mask of the positions that hold a token, or None.
        cu_seqlens: The positions of the documents of a packed row, or None.
        ready: Event recorded on the lookup stream once ``memory`` and ``valid`` are complete.
    """

    def __init__(
        self,
        memory: Tensor,
        valid: Optional[Tensor],
        cu_seqlens: Optional[Tensor],
        ready: torch.cuda.Event,
    ) -> None:
        self.memory = memory
        self.valid = valid
        self.cu_seqlens = cu_seqlens
        self.ready = ready

    def wait(self) -> None:
        """Make the current stream wait for the lookup and keep its tensors alive for it."""
        stream = torch.cuda.current_stream()
        stream.wait_event(self.ready)
        self.memory.record_stream(stream)
        if self.valid is not None and self.valid.is_cuda:
            self.valid.record_stream(stream)

    def attach(self) -> Tensor:
        """Return the memory for the Engram layer's fusion, on the current stream.

        With autograd recording, the result's backward launches the lookup's backward (see
        :class:`_EngramLookupHandoff`). Under full recompute this runs twice: without autograd
        in the checkpointed forward, and with it in the recompute. A memory without a graph
        (frozen tables, a lookup made without gradients) is returned as it is.
        """
        if self.memory is None:
            raise RuntimeError("This Engram lookup was already consumed by a backward pass.")
        if torch.is_grad_enabled() and self.memory.requires_grad:
            return _EngramLookupHandoff.apply(self.memory.detach().requires_grad_(), self)
        self.wait()
        return self.memory


def _engram_of(layer: torch.nn.Module):
    """The Engram module of a decoder layer (wrapped by a hyper-connection layer or not)."""
    inner_layer = getattr(layer, "inner_layer", layer)
    return getattr(inner_layer, "engram", None)


def local_engram_modules(layers: Iterable[torch.nn.Module]) -> list:
    """Return the Engram modules of ``layers`` (decoder layers), in layer order."""
    return [engram for engram in map(_engram_of, layers) if engram is not None]


@contextmanager
def use_prefetched_lookup(layer: torch.nn.Module, lookups: Optional[dict]):
    """Bind ``layer``'s prefetched lookup, if it has one, to its Engram module for the call.

    Args:
        layer: A decoder layer: one that holds an Engram module, the hyper-connection wrapper
            of one, or any other layer, for which nothing is bound.
        lookups: The handles of :func:`prefetch_engram_lookups`, keyed by layer number.
    """
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
    engrams: Iterable[torch.nn.Module], input_ids: Tensor, packed_seq_params=None
) -> dict[int, EngramLookupHandle]:
    """Issue the lookup of every Engram module in ``engrams`` for this microbatch.

    A module takes the next route :func:`plan_engram_lookups` queued for it; without one it
    routes and exchanges its counts now, which synchronizes the host with the lookup stream
    once. Collective over the row group: every rank issues the lookups of the same modules in
    the same order.

    Args:
        engrams: This rank's Engram modules, in layer order.
        input_ids: This rank's ``[B, S]`` token rows.
        packed_seq_params: The packed rows' document boundaries, if any.

    Returns:
        The handles keyed by layer number. A module that issues no lookup for this row
        (``Engram.prefetch`` returns None) has no entry, and its layer looks its rows up inline.
    """
    lookups = {}
    for engram in engrams:
        route = engram.planned_routes.popleft() if engram.planned_routes else None
        handle = engram.prefetch(input_ids, packed_seq_params, route=route)
        if handle is not None:
            lookups[engram.layer_number] = handle
    return lookups


def join_lookup_stream_in_backward(tensor: Tensor) -> None:
    """Make the calling stream wait for the lookup stream when backward reaches ``tensor``.

    Registered on a tensor in front of every Engram layer, such as the embedding output: when
    the backward reaches it, every Engram layer's backward has launched its lookup's backward
    on the lookup stream, and the gradient accumulation there must be complete before the
    gradients are reduced and the optimizer reads them. A tensor that does not require grad is
    in no backward pass, and nothing is registered on it.
    """
    if not tensor.requires_grad:
        return

    def _join(grad: Tensor) -> None:
        torch.cuda.current_stream().wait_stream(get_lookup_stream())

    tensor.register_hook(_join)


def plan_engram_lookups(
    model_chunks: Iterable[torch.nn.Module], token_batches: list[Tensor]
) -> None:
    """Route every microbatch of a step for every training Engram module of ``model_chunks``.

    All the request counts go through one all-to-all and one host transfer; the routes are
    queued on each module and consumed, in microbatch order, by
    :func:`prefetch_engram_lookups`. Routes still queued from an earlier call are dropped
    first, so a step that is run again starts over. Collective over the row group: every rank
    of it calls this with the same number of microbatches.

    Args:
        model_chunks: This rank's model chunks.
        token_batches: The step's unpacked ``[B, S]`` token rows, one tensor per microbatch,
            in sequence order and before any context-parallel split.
    """
    # Engram imports this module.
    from .engram import Engram

    engrams = [
        module for chunk in model_chunks for module in chunk.modules() if isinstance(module, Engram)
    ]
    for engram in engrams:
        engram.planned_routes.clear()
    # A row group of one rank has no counts to exchange.
    engrams = [
        engram
        for engram in engrams
        if engram.training and engram.engram_config.async_lookup and engram.embedding.ep_size > 1
    ]
    if not engrams or not token_batches:
        return
    first = engrams[0]
    num_tables = len(first.embedding.tables)
    for engram in engrams[1:]:
        same_tables = len(engram.embedding.tables) == num_tables
        if engram.lookup_group is not first.lookup_group or not same_tables:
            raise ValueError(
                "All Engram layers must look up the same number of tables over the same row "
                "group to share one count exchange."
            )

    stream = get_lookup_stream()
    stream.wait_stream(torch.cuda.current_stream())
    routes: list[tuple[torch.nn.Module, EngramRoute]] = []
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
                first.embedding.ep_size,
                num_tables,
                first.lookup_group,
            )
    finally:
        nvtx_range_pop(message)
    for engram, route in routes:
        engram.planned_routes.append(route)
