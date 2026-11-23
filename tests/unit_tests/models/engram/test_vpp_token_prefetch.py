# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Virtual pipeline parallelism: one token prefetch feeding every model chunk.

Under VPP the scheduler gives each model chunk its own data iterator and indexes that list
itself before calling ``forward_step``, so each chunk ends up holding its own wrapper. Tokens
depend only on the microbatch, never on the chunk, so a single prefetch has to serve all of
them -- each with an independent cursor, because the interleaved schedule consumes chunks out
of lockstep.

A caller can also look at the tokens queued for the microbatches still to be scheduled without
consuming them, and have a single rank, which distributes nothing, read its step ahead for
that. Such a rank has no peer to tell whether its data arrived, so it reads no tensor on the
host for it.
"""

from contextlib import contextmanager

import pytest
import torch

from megatron.training.utils import get_pipeline_prefetched_tokens, prepare_tokens_for_pipeline


@pytest.fixture
def single_rank(monkeypatch):
    """Collapse PP and TP so the helper takes its no-collective path."""
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 1)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)


@pytest.fixture
def two_rank(monkeypatch):
    """Two peers so the helper takes the prefetch-and-broadcast path, with comms stubbed."""
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 2)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: rank)


@contextmanager
def _no_host_read(monkeypatch):
    """Fail on ``item``, ``tolist`` and ``bool()`` of a tensor inside the block.

    Each of them reads the tensor's values on the host and so waits for the device. The
    availability flag that the source of a multi-rank prefetch broadcasts is read with
    ``item``.
    """

    def host_read(*unused, **unused_options):
        raise AssertionError("the token prefetch read a tensor on the host")

    with monkeypatch.context() as patch:
        for name in ("item", "tolist", "__bool__"):
            patch.setattr(torch.Tensor, name, host_read)
        yield


def _batches(count, seq=4, mbs=1):
    return [
        {"tokens": torch.arange(i * seq * mbs, (i + 1) * seq * mbs).view(mbs, seq), "id": i}
        for i in range(count)
    ]


def test_a_list_of_iterators_yields_one_wrapper_per_chunk(two_rank):
    chunks = [iter(_batches(2)), iter(_batches(2)), iter(_batches(2))]
    wrappers = prepare_tokens_for_pipeline(chunks, 2, 1, 4, object(), object())

    assert isinstance(wrappers, list) and len(wrappers) == 3
    # Each chunk owns an independent cursor over the same tokens.
    assert wrappers[0] is not wrappers[1]


def test_every_chunk_sees_the_same_tokens_for_the_same_microbatch(two_rank):
    expected = _batches(3)
    chunks = [iter(_batches(3)) for _ in range(2)]
    wrappers = prepare_tokens_for_pipeline(chunks, 3, 1, 4, object(), object())

    for microbatch in range(3):
        per_chunk = []
        for wrapper in wrappers:
            next(wrapper)
            per_chunk.append(get_pipeline_prefetched_tokens(wrapper))
        torch.testing.assert_close(
            per_chunk[0], expected[microbatch]["tokens"].to(per_chunk[0].device)
        )
        # Chunk 1 must see byte-identical tokens for the same microbatch.
        torch.testing.assert_close(per_chunk[1], per_chunk[0])


def test_chunks_may_be_consumed_out_of_lockstep(two_rank):
    """The interleaved schedule runs chunk 0 ahead of chunk 1; cursors must not interfere."""
    expected = _batches(3)
    chunks = [iter(_batches(3)) for _ in range(2)]
    first, second = prepare_tokens_for_pipeline(chunks, 3, 1, 4, object(), object())

    # Chunk 0 runs all three microbatches before chunk 1 runs any.
    seen_first = []
    for _ in range(3):
        next(first)
        seen_first.append(get_pipeline_prefetched_tokens(first))
    seen_second = []
    for _ in range(3):
        next(second)
        seen_second.append(get_pipeline_prefetched_tokens(second))

    for microbatch in range(3):
        torch.testing.assert_close(
            seen_first[microbatch], expected[microbatch]["tokens"].to(seen_first[0].device)
        )
        torch.testing.assert_close(seen_second[microbatch], seen_first[microbatch])


def test_each_chunk_keeps_its_own_exhaustion_guard(two_rank):
    chunks = [iter(_batches(1)), iter(_batches(1))]
    first, second = prepare_tokens_for_pipeline(chunks, 1, 1, 4, object(), object())

    next(first)
    get_pipeline_prefetched_tokens(first)
    with pytest.raises(RuntimeError, match="queue is exhausted"):
        get_pipeline_prefetched_tokens(first)
    # Draining chunk 0 must not have consumed chunk 1's tokens.
    next(second)
    get_pipeline_prefetched_tokens(second)


def test_single_rank_fast_path_is_also_per_chunk(single_rank):
    """With PP=TP=1 nothing is broadcast, but each chunk still needs its own wrapper."""
    chunks = [iter(_batches(2)), iter(_batches(2))]
    wrappers = prepare_tokens_for_pipeline(chunks, 2, 1, 4, object(), object())

    assert isinstance(wrappers, list) and len(wrappers) == 2
    for wrapper in wrappers:
        batch = next(wrapper)
        tokens = get_pipeline_prefetched_tokens(wrapper)
        torch.testing.assert_close(tokens, batch["tokens"].to(tokens.device))


def test_a_bare_iterator_still_returns_a_bare_wrapper(two_rank):
    """Non-VPP callers must keep the old contract: one iterator in, one wrapper out."""
    wrapper = prepare_tokens_for_pipeline(iter(_batches(2)), 2, 1, 4, object(), object())
    assert not isinstance(wrapper, list)
    next(wrapper)
    assert get_pipeline_prefetched_tokens(wrapper).shape == (1, 4)


def test_a_middle_chunk_without_data_still_gets_every_microbatch(two_rank):
    """Middle virtual stages build no dataset, so their chunk iterator is None.

    get_batch takes their tokens without advancing the wrapper, which has no batch to give.
    """
    expected = _batches(2)
    _, middle = prepare_tokens_for_pipeline([iter(_batches(2)), None], 2, 1, 4, object(), object())

    for microbatch in range(2):
        tokens = get_pipeline_prefetched_tokens(middle)
        torch.testing.assert_close(tokens, expected[microbatch]["tokens"].to(tokens.device))
    with pytest.raises(StopIteration):
        next(middle)


def test_queued_tokens_can_be_read_without_consuming_them(two_rank):
    expected = _batches(3)
    wrapper = prepare_tokens_for_pipeline(iter(_batches(3)), 3, 1, 4, object(), object())

    for scheduled in range(3):
        # Reading twice gives the same microbatches: the ones still to be scheduled, in order.
        for _ in range(2):
            queued = wrapper.peek_pipeline_tokens()
            assert len(queued) == 3 - scheduled
            for tokens, batch in zip(queued, expected[scheduled:]):
                assert torch.equal(tokens, batch["tokens"].to(tokens.device))
        next(wrapper)
        assert get_pipeline_prefetched_tokens(wrapper) is queued[0]
    assert wrapper.peek_pipeline_tokens() == []


def test_a_single_rank_reads_its_step_ahead_on_request(single_rank, monkeypatch):
    """With PP=TP=1 nothing is distributed, but ``prefetch_all`` still reads the step ahead: its
    tokens are queued before the schedule, which then gets the same batches in the same order.
    No peer waits for an availability flag, so the rank reads no tensor on the host."""
    batches = _batches(4)
    with _no_host_read(monkeypatch):
        wrapper = prepare_tokens_for_pipeline(
            iter(batches), 3, 1, 4, object(), object(), prefetch_all=True
        )
        queued = wrapper.peek_pipeline_tokens()

    assert len(queued) == 3
    for microbatch in range(3):
        assert next(wrapper) is batches[microbatch]
        tokens = get_pipeline_prefetched_tokens(wrapper)
        assert tokens is queued[microbatch]
        assert torch.equal(tokens, batches[microbatch]["tokens"].to(tokens.device))
    # Only the step was read ahead: what follows comes from the data iterator as before.
    assert next(wrapper) is batches[3]


def test_a_single_rank_gets_its_iterators_own_exhaustion(single_rank):
    """A data iterator that ends within the step raises its ``StopIteration`` to the caller of
    the prefetch, as it does to the schedule without ``prefetch_all``; so does any other
    failure of the read. With peers, the flag turns an exhausted iterator into a RuntimeError
    on every rank; any other failure is raised as it is on the source rank and as a
    RuntimeError on its peers."""
    with pytest.raises(StopIteration):
        prepare_tokens_for_pipeline(
            iter(_batches(2)), 3, 1, 4, object(), object(), prefetch_all=True
        )
    with pytest.raises(ValueError, match="expected token shape"):
        prepare_tokens_for_pipeline(
            iter(_batches(3, seq=5)), 3, 1, 4, object(), object(), prefetch_all=True
        )


def test_a_single_rank_queues_nothing_unless_asked(single_rank):
    """Without ``prefetch_all`` the tokens of a single rank are derived from each batch as the
    schedule consumes it, so none are queued and the data iterator is not read ahead."""
    batches = _batches(2)
    source = iter(batches)
    wrapper = prepare_tokens_for_pipeline(source, 2, 1, 4, object(), object())

    assert wrapper.peek_pipeline_tokens() == []
    assert next(source) is batches[0]
    assert next(wrapper) is batches[1]
    assert wrapper.peek_pipeline_tokens() == []
