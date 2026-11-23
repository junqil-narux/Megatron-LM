# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Asynchronous Engram lookup (``EngramConfig.async_lookup``) against the inline lookup.

The asynchronous path runs the operations of the inline lookup on the same data, on the lookup
stream and communicator and at other points of the step. So the multi-rank cases demand what
the inline lookup gives, bit for bit: the output of every microbatch and every parameter's
gradient after DDP's reduction. They also pin down what the path is for: no host
synchronization in a planned microbatch, no lookup replay under full recompute, and the
table-gradient branch on the lookup stream right after the Engram layer's backward. Two
further cases hold the lookup stream back during the backward and read the table gradients on
the compute stream with no synchronization in between: they fail when nothing makes the
compute stream wait for the lookup stream.

The multi-rank cases need GPUs: torchrun with two, four or eight ranks for the expert-parallel
table layout, and exactly four for the sharded layouts::

    torchrun --nproc-per-node 4 -m pytest tests/unit_tests/models/engram/test_async_lookup.py

The other cases run on CPU in any process. Those that issue a lookup do so on a table that
one rank owns whole, with stand-ins for the CUDA streams.
"""

import os
import warnings
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.engram import async_lookup
from megatron.core.models.engram import engram as engram_module
from megatron.core.models.engram import hashing as hashing_module
from megatron.core.models.engram.async_lookup import (
    get_lookup_group,
    get_lookup_stream,
    initialize_engram_lookup_groups,
    join_lookup_stream_in_backward,
    local_engram_modules,
    plan_engram_lookups,
    prefetch_engram_lookups,
    use_prefetched_lookup,
)
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.cp_layout import (
    select_contiguous,
    select_zigzag,
    thd_partition_index,
)
from megatron.core.models.engram.engram import Engram
from megatron.core.models.engram.table_sharding import (
    clear_engram_parallel_state,
    initialize_engram_parallel_state,
)
from megatron.core.models.engram.variants import GATED_FUSION
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

from ._test_utils import deterministic_algorithms, make_module_config, make_pg_collection
from ._test_utils import packed_seq_params as tagged_packed_seq_params
from ._test_utils import sync_debug_mode, write_tokenizer_map

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_multi_rank = pytest.mark.skipif(
    _WORLD_SIZE not in (2, 4, 8), reason="requires torchrun with two, four, or eight ranks"
)
_four_ranks = pytest.mark.skipif(
    _WORLD_SIZE != 4, reason="requires torchrun with exactly four ranks"
)

_HIDDEN = 128
_SEQ = 64
_BATCH = 2
_VOCAB = 64
_MICROBATCHES = 3
_ENGRAM_LAYER = 2
_DTYPE = torch.bfloat16

_BIT_VIEWS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}


def _assert_bitwise_equal(actual, expected, what):
    assert actual.dtype == expected.dtype, f"{what}: {actual.dtype} vs {expected.dtype}"
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    view = _BIT_VIEWS.get(actual.dtype, actual.dtype)
    mismatches = (actual.contiguous().view(view) != expected.contiguous().view(view)).sum().item()
    assert mismatches == 0, f"{what}: {mismatches} of {actual.numel()} elements differ"


def _assert_same_step(result, reference):
    """Same outputs of the microbatches and same reduced gradient of every parameter."""
    outputs, grads = result
    reference_outputs, reference_grads = reference
    assert len(outputs) == len(reference_outputs)
    for index, (output, expected) in enumerate(zip(outputs, reference_outputs)):
        _assert_bitwise_equal(output, expected, f"microbatch {index} output")
    assert grads.keys() == reference_grads.keys()
    for name, grad in grads.items():
        _assert_bitwise_equal(grad, reference_grads[name], f"{name} gradient")


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _occupy(stream):
    """Queue work on ``stream`` that outlasts by far what the host launches elsewhere meanwhile.

    Whatever is queued on the stream afterwards runs late: another stream that reads its
    results without waiting for the stream reads them before they are written. That holds
    while the host itself does not wait for the stream. The table backward of deterministic
    mode reads a count on the host, so the cases that hold a stream back build their blocks
    without that mode.
    """
    with torch.cuda.stream(stream):
        # Elementwise passes over 512 MB: tens of milliseconds, and no library workspace.
        work = torch.zeros(1 << 27, device=_device())
        for _ in range(256):
            work.add_(1.0)


@contextmanager
def _record_stream_waits():
    """Record the calls of ``torch.cuda.Stream.wait_stream`` in order, as (waiting, awaited)."""
    waits = []
    wait_stream = torch.cuda.Stream.wait_stream

    def recording_wait_stream(stream, awaited):
        waits.append((stream, awaited))
        return wait_stream(stream, awaited)

    torch.cuda.Stream.wait_stream = recording_wait_stream
    try:
        yield waits
    finally:
        torch.cuda.Stream.wait_stream = wait_stream


def _assert_compute_stream_waits_for_the_last_lookup_backward(waits):
    """The compute stream is made to wait for the lookup stream after the last lookup backward.

    The handoff makes the lookup stream wait for the compute stream and queues the lookup's
    backward behind that wait. A wait of the compute stream for the lookup stream orders the
    compute stream after the table gradients of that backward only if it is issued later.

    The plan and every prefetch make the lookup stream wait for the compute stream as well,
    so such a wait alone does not show a launched backward. The last one recorded is the
    handoff of the last backward, which runs after the last prefetch.

    A comparison of gradients that are read early does not always tell: the device can run the
    work of the lookup stream before the read although nothing orders the two, for example
    while the compute stream waits for the reductions of other buckets.
    """
    compute_stream, lookup_stream = torch.cuda.current_stream(), get_lookup_stream()
    launched = [
        index for index, wait in enumerate(waits) if wait == (lookup_stream, compute_stream)
    ]
    joined = [index for index, wait in enumerate(waits) if wait == (compute_stream, lookup_stream)]
    assert launched, "the lookup stream never waited for the compute stream"
    assert (
        joined and joined[-1] > launched[-1]
    ), "nothing makes the compute stream wait for the table gradients of the last backward"


@contextmanager
def _count_synchronizations(counts, key):
    """Add the synchronizing CUDA calls made inside the block to ``counts[key]``."""
    with warnings.catch_warnings(record=True) as caught, sync_debug_mode("warn"):
        warnings.simplefilter("always")
        yield
    counts[key] += sum("synchroniz" in str(warning.message) for warning in caught)


def _record_all_to_all_groups(monkeypatch):
    """Record the group of every all-to-all from here on, those of the backward included."""
    groups = []
    real = torch.distributed.all_to_all_single

    def recording(*args, group=None, **kwargs):
        groups.append(group)
        return real(*args, group=group, **kwargs)

    monkeypatch.setattr(torch.distributed, "all_to_all_single", recording)
    return groups


@contextmanager
def _deterministic_mode(enabled, monkeypatch):
    """Mirror --deterministic-mode: deterministic torch algorithms and the cuBLAS workspace
    they ask for."""
    if enabled:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with deterministic_algorithms(enabled):
        yield


@pytest.fixture
def expert_parallel(request):
    """Model-parallel groups with expert parallelism ``request.param``; the rest is expert-DP."""
    ep_size = request.param
    if _WORLD_SIZE % ep_size != 0:
        pytest.skip(f"world size {_WORLD_SIZE} is not a multiple of EP {ep_size}")
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    model_parallel_cuda_manual_seed(123)
    yield ep_size
    Utils.destroy_model_parallel()


def _engram_config(tmp_path, **overrides):
    artifact = tmp_path / "map.json"
    if not artifact.exists():
        write_tokenizer_map(artifact, vocab_size=_VOCAB, layer_ids=(_ENGRAM_LAYER,))
    return EngramConfig(
        global_vocab_sizes=(1009, 997),
        layer_ids=(_ENGRAM_LAYER,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=64,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
        **overrides,
    )


class _Block(torch.nn.Module):
    """An embedding stand-in, the layer in front of the Engram layer, and the Engram layer."""

    def __init__(self, engram_config, pg_collection, deterministic=False, batch_size=_BATCH):
        super().__init__()
        device = _device()
        config = make_module_config(dtype=_DTYPE, deterministic_mode=deterministic)
        config.use_cpu_initialization = device.type == "cpu"
        config.hidden_size = _HIDDEN
        self.embedding = torch.nn.Parameter(
            0.5 * torch.randn(_SEQ, batch_size, _HIDDEN, dtype=_DTYPE, device=device)
        )
        self.front = torch.nn.Linear(_HIDDEN, _HIDDEN, bias=False, dtype=_DTYPE, device=device)
        self.engram = Engram(
            config=config,
            engram_config=engram_config,
            layer_number=_ENGRAM_LAYER,
            pg_collection=pg_collection,
        )
        with torch.no_grad():
            # Away from their initial values, so that every dense parameter gets a gradient.
            if engram_config.variant_spec.fusion == GATED_FUSION:
                self.engram.q_weight.normal_(mean=1.0, std=0.2)
                self.engram.k_weight.normal_(mean=1.0, std=0.2)
            else:
                self.engram.short_conv.weight.normal_(mean=0.0, std=0.05)

    def layers(self, hidden, input_ids, packed_seq_params=None):
        hidden = hidden + self.front(hidden)
        return self.engram(hidden, input_ids, packed_seq_params, add_to_hidden_states=True)


def _wrap_with_ddp(block, pg_collection):
    from megatron.training.training import wrap_model_chunks_with_ddp

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
        # Small buckets: a table shard fills buckets of its own or shares one with the weights
        # of the other layers, and the hook of a bucket's last gradient launches its reduction.
        bucket_size=8192,
    )
    model = wrap_model_chunks_with_ddp(
        [block],
        TransformerConfig(num_attention_heads=1, num_layers=1),
        ddp_config,
        pg_collection=pg_collection,
    )[0]
    model.broadcast_params()
    return model


def _build(engram_config, pg_collection, deterministic=False, batch_size=_BATCH, ddp=True):
    """The block of one Engram configuration, under DDP with overlapping gradient reduction
    unless ``ddp`` is off."""
    torch.manual_seed(1234)
    model = _Block(engram_config, pg_collection, deterministic, batch_size)
    if ddp:
        model = _wrap_with_ddp(model, pg_collection)
    model.train()
    return model


def _build_pair(tmp_path, deterministic=False, batch_size=_BATCH, ddp=True, **engram_options):
    """The block twice, with the same parameters: without and with ``async_lookup``."""
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    asynchronous_config = _engram_config(tmp_path, async_lookup=True, **engram_options)
    # Collective, on every rank before a model is built, as in the model builder.
    initialize_engram_parallel_state(pg_collection, asynchronous_config)
    initialize_engram_lookup_groups(pg_collection, asynchronous_config)
    inline = _build(
        _engram_config(tmp_path, **engram_options), pg_collection, deterministic, batch_size, ddp
    )
    asynchronous = _build(asynchronous_config, pg_collection, deterministic, batch_size, ddp)
    with torch.no_grad():
        for source, target in zip(inline.parameters(), asynchronous.parameters()):
            target.copy_(source)
    return inline, asynchronous


def _token_batches(batch_size=_BATCH):
    device = _device()
    generator = torch.Generator(device=device).manual_seed(77 + torch.distributed.get_rank())
    return [
        torch.randint(0, _VOCAB, (batch_size, _SEQ), generator=generator, device=device)
        for _ in range(_MICROBATCHES)
    ]


def _run_step(
    model,
    token_batches,
    *,
    asynchronous,
    planned=True,
    recompute=False,
    packed_seq_params=None,
    hold_lookup_stream=False,
):
    """One step over ``token_batches``, in the order a training step uses.

    Returns the outputs of the microbatches, every trained parameter's reduced gradient, and
    the host synchronizations of the per-step plan and of the forward passes.

    With ``hold_lookup_stream`` the lookup stream is busy when the backward of the last
    microbatch starts, and the gradients are read on the compute stream as soon as
    ``finish_grad_sync`` returns, as the optimizer of a training run reads them. Otherwise
    the host waits for every stream first.
    """
    block = model.module
    synchronizations = {"plan": 0, "forward": 0}
    model.zero_grad_buffer()
    if asynchronous and planned:
        with _count_synchronizations(synchronizations, "plan"):
            plan_engram_lookups([model], token_batches)
    outputs = []
    for index, input_ids in enumerate(token_batches):
        last = index == len(token_batches) - 1
        with nullcontext() if last else model.no_sync():
            with _count_synchronizations(synchronizations, "forward"):
                lookups = None
                if asynchronous:
                    lookups = prefetch_engram_lookups([block.engram], input_ids, packed_seq_params)
                hidden = block.embedding * 1.0
                if asynchronous:
                    join_lookup_stream_in_backward(hidden)

                def run(hidden_states):
                    # What checkpointed_forward does with its layer_context: the binding is
                    # entered around the layer in the forward and again in the recompute.
                    with use_prefetched_lookup(block, lookups):
                        return block.layers(hidden_states, input_ids, packed_seq_params)

                output = (
                    tensor_parallel.checkpoint(run, False, hidden) if recompute else run(hidden)
                )
            generator = torch.Generator(device=output.device).manual_seed(
                1000 * index + torch.distributed.get_rank()
            )
            upstream = torch.randn(
                output.shape, generator=generator, device=output.device, dtype=output.dtype
            )
            if hold_lookup_stream and last:
                # Nothing is pending on any stream: the lookup stream runs this work next, and
                # the table-gradient branch of the backward only behind it.
                torch.cuda.synchronize()
                _occupy(get_lookup_stream())
            output.backward(upstream)
            outputs.append(output.detach().clone())
    model.finish_grad_sync()
    if not hold_lookup_stream:
        torch.cuda.synchronize()
    grads = {
        name: param.main_grad.detach().clone()
        for name, param in block.named_parameters()
        if param.requires_grad
    }
    return (outputs, grads), synchronizations


def _compare_steps(models, tokens, monkeypatch, *, planned=True, recompute=False, **step_options):
    """Run the inline step on the first model and the asynchronous one on the second.

    Returns both results, the groups of the all-to-alls of each step, and the host
    synchronizations of the asynchronous step. Each model runs one step before its measured
    one, which sets up the communicators and kernels a first step initializes.
    """
    inline, asynchronous = models
    groups = _record_all_to_all_groups(monkeypatch)
    step_options.update(recompute=recompute)
    _run_step(inline, tokens, asynchronous=False, **step_options)
    _run_step(asynchronous, tokens, asynchronous=True, planned=planned, **step_options)
    first = len(groups)
    reference, _ = _run_step(inline, tokens, asynchronous=False, **step_options)
    inline_groups, first = groups[first:], len(groups)
    result, synchronizations = _run_step(
        asynchronous, tokens, asynchronous=True, planned=planned, **step_options
    )
    return result, reference, inline_groups, groups[first:], synchronizations


def _assert_all_to_alls(models, inline_groups, async_groups, *, planned, recompute):
    """The asynchronous step runs on the lookup communicator and replays no lookup.

    Inline, every forward of the Engram layer exchanges counts, requests and rows, and the
    recompute is one more forward; the backward returns the row gradients when the tables
    train. The asynchronous step exchanges the counts of a planned step once and looks every
    microbatch up once.
    """
    inline, asynchronous = (model.module.engram for model in models)
    row_group = inline.embedding.ep_group
    lookup_group = asynchronous.lookup_group
    assert lookup_group is not None and lookup_group is not row_group
    assert set(inline_groups) == {row_group}
    assert set(async_groups) == {lookup_group}
    backward = 0 if inline.engram_config.freeze_embedding else 1
    forwards = 2 if recompute else 1
    assert len(inline_groups) == _MICROBATCHES * (3 * forwards + backward)
    count_exchanges = 1 if planned else _MICROBATCHES
    assert len(async_groups) == count_exchanges + _MICROBATCHES * (2 + backward)


@_multi_rank
@pytest.mark.parametrize("expert_parallel", [2, 4], indirect=True)
@pytest.mark.parametrize("deterministic", [False, True], ids=["nondeterministic", "deterministic"])
@pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
@pytest.mark.parametrize("planned", [True, False], ids=["planned", "per_microbatch"])
def test_async_lookup_matches_inline_bitwise(
    tmp_path, monkeypatch, expert_parallel, deterministic, recompute, planned
):
    """Expert-parallel layout: same outputs and reduced gradients, one count exchange for a
    planned step and no host synchronization in its microbatches, no replay under recompute."""
    with _deterministic_mode(deterministic, monkeypatch):
        models = _build_pair(tmp_path, deterministic)
        result, reference, inline_groups, async_groups, synchronizations = _compare_steps(
            models, _token_batches(), monkeypatch, planned=planned, recompute=recompute
        )

    _assert_same_step(result, reference)
    _assert_all_to_alls(models, inline_groups, async_groups, planned=planned, recompute=recompute)
    # The transfer of the exchanged counts is the only host read of the lookups: once for a
    # planned step, once per microbatch otherwise.
    expected = {"plan": 1, "forward": 0} if planned else {"plan": 0, "forward": _MICROBATCHES}
    assert synchronizations == expected


@_four_ranks
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("expert_parallel", [2], indirect=True)
@pytest.mark.parametrize("parallel_size", [4, 2])
@pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
def test_async_lookup_matches_inline_bitwise_with_sharded_tables(
    tmp_path, monkeypatch, expert_parallel, parallel_size, recompute
):
    """Sharded layouts with the NCCL lookup: the row group is dp_cp itself (one owner per row,
    single-rank replica groups) or a slice of it, and the tables are a parameter domain."""
    models = _build_pair(tmp_path, parallel_size=parallel_size, communication_backend="nccl")
    result, reference, inline_groups, async_groups, synchronizations = _compare_steps(
        models, _token_batches(), monkeypatch, recompute=recompute
    )

    _assert_same_step(result, reference)
    _assert_all_to_alls(models, inline_groups, async_groups, planned=True, recompute=recompute)
    assert synchronizations == {"plan": 1, "forward": 0}
    assert torch.distributed.get_process_group_ranks(models[1].module.engram.lookup_group) == (
        torch.distributed.get_process_group_ranks(models[0].module.engram.embedding.ep_group)
    )


@_four_ranks
@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("expert_parallel", [2], indirect=True)
def test_table_gradients_are_complete_when_finish_grad_sync_returns(tmp_path, expert_parallel):
    """One owner per row: no collective orders the compute stream after the lookup stream.

    The replica group of the tables has a single rank, so DDP skips their reductions, and the
    handle of a collective is not there to wait on. The table gradients are accumulated into
    their buckets on the lookup stream, which is held back here, and from the second step on
    the hook of the last one finishes the buckets there. The compute stream reads complete
    gradients once ``finish_grad_sync`` returns only if it was made to wait for the lookup
    stream. Two waits do that, and the case fails when both are missing: the join at the
    embedding output (``join_lookup_stream_in_backward``), and the wait of
    ``finish_grad_sync`` for the stream the tables name. The case cannot tell which of the
    two is missing; the join alone is checked without DDP below.

    The gradients read early are compared, but that comparison alone passes without either
    wait on the devices it was tried on: the compute stream waits for the reductions of the
    other buckets, and the lookup stream gets through its work meanwhile. The recorded stream
    waits are what fails then.
    """
    models = _build_pair(tmp_path, parallel_size=4, communication_backend="nccl")
    tokens = _token_batches()
    # The first step of a model reduces its buckets from finish_grad_sync, on the compute
    # stream; the hooks dispatch the reductions from the second step on.
    for asynchronous, model in enumerate(models):
        _run_step(model, tokens, asynchronous=bool(asynchronous))
    reference, _ = _run_step(models[0], tokens, asynchronous=False)
    with _record_stream_waits() as waits:
        result, _ = _run_step(models[1], tokens, asynchronous=True, hold_lookup_stream=True)

    _assert_same_step(result, reference)
    assert any("embedding.tables" in name for name in result[1])
    _assert_compute_stream_waits_for_the_last_lookup_backward(waits)


def _table_gradients(block, input_ids, *, asynchronous, hold_lookup_stream=False):
    """One forward and backward of a block that DDP does not wrap.

    Returns each table's ``.grad``, read on the compute stream as soon as the backward
    returns. The gradients are zeroed in place first, so that from the second call on the
    backward accumulates into them and a gradient that is read too early reads as zero. With
    ``hold_lookup_stream`` the lookup stream is busy when the backward starts.
    """
    block.zero_grad(set_to_none=False)
    lookups = None
    if asynchronous:
        lookups = prefetch_engram_lookups([block.engram], input_ids)
    hidden = block.embedding * 1.0
    if asynchronous:
        join_lookup_stream_in_backward(hidden)
    with use_prefetched_lookup(block, lookups):
        output = block.layers(hidden, input_ids)
    generator = torch.Generator(device=output.device).manual_seed(torch.distributed.get_rank())
    upstream = torch.randn(
        output.shape, generator=generator, device=output.device, dtype=output.dtype
    )
    # Nothing is pending on any stream, the zeroed gradients included.
    torch.cuda.synchronize()
    if hold_lookup_stream:
        _occupy(get_lookup_stream())
    output.backward(upstream)
    grads = [table.weight.grad.clone() for table in block.engram.embedding.tables]
    torch.cuda.synchronize()
    return grads


@_multi_rank
@pytest.mark.parametrize("expert_parallel", [2], indirect=True)
def test_backward_waits_for_the_table_gradients_without_ddp(tmp_path, expert_parallel):
    """The join is all that orders the compute stream after the table gradients.

    The blocks are not wrapped by DDP: nothing reduces their gradients, and no bucket group
    waits for the lookup stream. The table-gradient branch accumulates each table's ``.grad``
    on the lookup stream, behind the work that holds that stream back, and the backward of the
    block returns without autograd waiting for it. The gradients read on the compute stream
    right then equal the inline block's only if the backward waited for the lookup stream at
    the embedding output (``join_lookup_stream_in_backward``).
    """
    inline, asynchronous = _build_pair(tmp_path, ddp=False)
    input_ids = _token_batches()[0]
    # A first pass each: it creates the gradients that the measured pass accumulates into.
    _table_gradients(inline, input_ids, asynchronous=False)
    _table_gradients(asynchronous, input_ids, asynchronous=True)
    reference = _table_gradients(inline, input_ids, asynchronous=False)
    with _record_stream_waits() as waits:
        grads = _table_gradients(
            asynchronous, input_ids, asynchronous=True, hold_lookup_stream=True
        )

    assert len(grads) == len(reference) > 0
    for index, (grad, expected) in enumerate(zip(grads, reference)):
        assert torch.count_nonzero(expected) > 0, f"table {index} got no gradient"
        _assert_bitwise_equal(grad, expected, f"table {index} gradient")
    _assert_compute_stream_waits_for_the_last_lookup_backward(waits)


@_multi_rank
@pytest.mark.parametrize("expert_parallel", [2], indirect=True)
@pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
def test_async_lookup_of_a_packed_row_carries_its_valid_mask(
    tmp_path, monkeypatch, expert_parallel, recompute
):
    """A packed row of the gated fusion: its counts are exchanged per microbatch, and the
    handle carries the document bounds and the mask that closes the gate after the last
    document."""
    models = _build_pair(tmp_path, batch_size=1, variant="deepseek_v41")
    cu_seqlens = torch.tensor([0, 23, _SEQ - 9], dtype=torch.int32, device=_device())
    packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens)
    tokens = _token_batches(batch_size=1)
    result, reference, inline_groups, async_groups, _ = _compare_steps(
        models,
        tokens,
        monkeypatch,
        planned=False,
        recompute=recompute,
        packed_seq_params=packed_seq_params,
    )
    # One more lookup, for the fields of its handle: a collective, so it comes before the
    # assertions.
    handle = models[1].module.engram.prefetch(tokens[0], packed_seq_params)
    torch.cuda.synchronize()

    _assert_same_step(result, reference)
    _assert_all_to_alls(models, inline_groups, async_groups, planned=False, recompute=recompute)
    assert torch.equal(handle.cu_seqlens, cu_seqlens)
    holds_token = torch.arange(_SEQ, device=handle.valid.device) < _SEQ - 9
    assert torch.equal(handle.valid.flatten(), holds_token)


@_multi_rank
@pytest.mark.parametrize("expert_parallel", [2], indirect=True)
def test_async_lookup_with_frozen_tables(tmp_path, monkeypatch, expert_parallel):
    """Frozen tables give a memory without a graph: the layer takes it as it is, in the
    checkpointed forward and in the recompute, and no row gradient is returned."""
    models = _build_pair(tmp_path, freeze_embedding=True)
    result, reference, inline_groups, async_groups, synchronizations = _compare_steps(
        models, _token_batches(), monkeypatch, recompute=True
    )

    _assert_same_step(result, reference)
    _assert_all_to_alls(models, inline_groups, async_groups, planned=True, recompute=True)
    assert synchronizations == {"plan": 1, "forward": 0}
    for table in models[1].module.engram.embedding.tables:
        assert not hasattr(table.weight, "grad_accumulation_stream")


@_multi_rank
@pytest.mark.parametrize("expert_parallel", [2], indirect=True)
@pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
def test_table_gradient_branch_runs_early_on_the_lookup_stream(
    tmp_path, expert_parallel, recompute
):
    """The table gradients are accumulated before the layer in front runs its backward.

    Inline, the lookup's graph sits inside the Engram layer, so autograd runs the table branch
    before the layer in front. The asynchronous lookup records its graph ahead of that layer,
    and only the handoff node keeps the order; its branch runs on the lookup stream, where
    DDP accumulates the table gradients, and everything else stays on the compute stream.
    """
    models = _build_pair(tmp_path)
    tokens = _token_batches()[:1]
    lookup_stream = get_lookup_stream()
    events = []
    for asynchronous, model in enumerate(models):
        block = model.module
        recorded = []
        events.append(recorded)

        def record(name, recorded=recorded):
            return lambda param: recorded.append(
                (name, torch.cuda.current_stream() == lookup_stream)
            )

        hooks = [
            table.weight.register_post_accumulate_grad_hook(record("table"))
            for table in block.engram.embedding.tables
        ]
        hooks.append(block.front.weight.register_post_accumulate_grad_hook(record("front")))
        hooks.append(block.embedding.register_post_accumulate_grad_hook(record("embedding")))
        _run_step(model, tokens, asynchronous=bool(asynchronous), recompute=recompute)
        for hook in hooks:
            hook.remove()

    for asynchronous, recorded in enumerate(events):
        names = [name for name, _ in recorded]
        tables = [index for index, name in enumerate(names) if name == "table"]
        assert len(tables) == len(models[0].module.engram.embedding.tables), recorded
        assert max(tables) < names.index("front") < names.index("embedding"), recorded
        for name, on_lookup_stream in recorded:
            assert on_lookup_stream == (bool(asynchronous) and name == "table"), recorded


@_multi_rank
@pytest.mark.parametrize("expert_parallel", [2], indirect=True)
def test_lookup_communicator_is_a_second_group_over_the_row_group(
    tmp_path, monkeypatch, expert_parallel
):
    """A module with ``async_lookup`` needs the lookup communicator when it is built; the
    setup creates one per row group, once, and tags the trained tables with the lookup stream."""
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    config = _engram_config(tmp_path, async_lookup=True)
    # Raised on every rank alike, ahead of any collective.
    with pytest.raises(RuntimeError, match="initialize_engram_lookup_groups"):
        _Block(config, pg_collection)

    created = []
    new_group = torch.distributed.new_group
    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        lambda *args, **kwargs: created.append(new_group(*args, **kwargs)) or created[-1],
    )
    # No setup without the option.
    initialize_engram_lookup_groups(pg_collection, _engram_config(tmp_path))
    assert not created
    initialize_engram_lookup_groups(pg_collection, config)
    num_row_groups = _WORLD_SIZE // expert_parallel
    num_created = len(created)
    initialize_engram_lookup_groups(pg_collection, config)

    assert num_created == len(created) == num_row_groups
    lookup_group = get_lookup_group(pg_collection.ep)
    assert lookup_group in created
    ranks = torch.distributed.get_process_group_ranks
    assert ranks(lookup_group) == ranks(pg_collection.ep)
    block = _Block(config, pg_collection)
    assert block.engram.lookup_group is lookup_group
    for table in block.engram.embedding.tables:
        assert table.weight.grad_accumulation_stream == get_lookup_stream()


_CPU_SEQ = 24


def _cpu_module(tmp_path, cp_group=None, cp_layouts=None, **engram_options):
    """An Engram module on CPU whose tables one rank owns whole.

    ``cp_layouts`` sets the layouts of the stack's linear and attention layers; the module is a
    linear layer's.
    """
    config = make_module_config()
    if cp_layouts is not None:
        config.linear_cp_layout, config.attention_cp_layout = cp_layouts
    torch.manual_seed(0)
    return Engram(
        config=config,
        engram_config=_engram_config(tmp_path, **engram_options),
        layer_number=_ENGRAM_LAYER,
        pg_collection=make_pg_collection(cp=cp_group),
    )


def _cpu_inputs(batch_size=1):
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, _VOCAB, (batch_size, _CPU_SEQ), generator=generator)
    hidden = torch.randn(_CPU_SEQ, batch_size, 8, generator=generator, dtype=torch.float64)
    return tokens, hidden


class _LookedUp:
    """A stand-in for a handle whose lookup is complete: the fields of a lookup result."""

    def __init__(self, result):
        self._memory, self.valid, self.cu_seqlens = result
        self.attached = 0

    def attach(self):
        self.attached += 1
        return self._memory


@pytest.mark.parametrize("wrapped", [False, True], ids=["layer", "hyper_connection_wrapper"])
def test_forward_takes_the_lookup_bound_to_its_layer(tmp_path, wrapped):
    """Under ``use_prefetched_lookup`` the layer's module fuses the bound lookup and looks
    nothing up; outside it, and for a layer without a lookup, nothing is bound."""
    module = _cpu_module(tmp_path, variant="deepseek_v41")
    layer = SimpleNamespace(engram=module)
    if wrapped:
        layer = SimpleNamespace(inner_layer=layer)
    tokens, hidden = _cpu_inputs()
    packed_seq_params = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=torch.tensor([0, 7, 19], dtype=torch.int32)
    )
    expected = module(hidden, tokens, packed_seq_params)
    handle = _LookedUp(module.lookup(tokens, packed_seq_params))
    assert handle.valid is not None and handle.cu_seqlens is not None
    lookups = []
    table_lookup = module._lookup
    module._lookup = lambda hash_ids: lookups.append(1) or table_lookup(hash_ids)

    with use_prefetched_lookup(layer, {_ENGRAM_LAYER: handle}):
        assert module.prefetched_lookup is handle
        # Neither the tokens nor the document bounds are read.
        assert torch.equal(module(hidden), expected)
        # An explicit lookup result takes precedence over the bound one.
        result = module.lookup(tokens, packed_seq_params)
        assert torch.equal(module(hidden, lookup=result), expected)
    assert (handle.attached, len(lookups)) == (1, 1)
    assert module.prefetched_lookup is None

    for other_layer, other_lookups in (
        (layer, None),
        (layer, {_ENGRAM_LAYER + 1: handle}),
        (SimpleNamespace(), {_ENGRAM_LAYER: handle}),
    ):
        with use_prefetched_lookup(other_layer, other_lookups):
            assert module.prefetched_lookup is None
    with pytest.raises(ZeroDivisionError):
        with use_prefetched_lookup(layer, {_ENGRAM_LAYER: handle}):
            raise ZeroDivisionError
    assert module.prefetched_lookup is None
    assert local_engram_modules([SimpleNamespace(), layer]) == [module]


def test_unsupported_async_lookup_fails_when_the_module_is_built(tmp_path, monkeypatch):
    """The option needs the NCCL lookup and CUDA; a module without it issues no lookup ahead."""
    config = _engram_config(tmp_path, async_lookup=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="needs CUDA"):
        Engram(make_module_config(), config, _ENGRAM_LAYER, make_pg_collection())
    # A configuration changed after EngramConfig checked it.
    config.communication_backend = "symmetric"
    with pytest.raises(ValueError, match="requires communication_backend='nccl'"):
        Engram(make_module_config(), config, _ENGRAM_LAYER, make_pg_collection())

    module = _cpu_module(tmp_path)
    assert module.lookup_group is None
    with pytest.raises(RuntimeError, match="built without async_lookup"):
        module.prefetch(_cpu_inputs()[0])


def test_plan_and_join_leave_nothing_behind_without_work(tmp_path):
    """A new plan drops the routes of an earlier one, also from a module it plans nothing
    for, and a tensor outside the backward pass gets no hook."""
    module = _cpu_module(tmp_path)
    module.planned_routes.append("stale")
    plan_engram_lookups([module], [_cpu_inputs()[0]])
    assert not module.planned_routes

    constant = torch.zeros(3)
    join_lookup_stream_in_backward(constant)
    assert not constant._backward_hooks


def test_teardown_forgets_the_lookup_communicators(monkeypatch):
    monkeypatch.setitem(async_lookup._LOOKUP_GROUPS, (0, 1), object())
    clear_engram_parallel_state()
    assert not async_lookup._LOOKUP_GROUPS


@pytest.mark.parametrize("pipeline_rank", [0, 1], ids=["embedding_stage", "later_stage"])
def test_only_the_embedding_stage_communicates_on_its_lookup_group(monkeypatch, pipeline_rank):
    """Two pipeline stages with one row group of two ranks each, on stand-in groups.

    Every rank creates the lookup group of every row group and keeps its own. Only the ranks
    of the first stage, which can hold the embedding, run a collective on theirs: that makes
    NCCL create the communicator, which a stage that issues no lookup ahead does not need.
    """
    row_groups = [(0, 1), (2, 3)]
    ranks = row_groups[pipeline_rank]
    row_group = SimpleNamespace(ranks=ranks)
    pg_collection = SimpleNamespace(pp=SimpleNamespace(rank=lambda: pipeline_rank))
    created, communicated = [], []

    def all_gather_object(descriptions, description):
        # Each rank describes its row group; both ranks of a group describe the same one.
        descriptions[:] = [(group, description[1]) for group in row_groups for _ in group]

    def create_group(group_ranks, **options):
        created.append(SimpleNamespace(ranks=tuple(group_ranks)))
        return created[-1]

    monkeypatch.setattr(async_lookup, "_LOOKUP_GROUPS", {})
    monkeypatch.setattr(async_lookup, "resolve_engram_groups", lambda *unused: (row_group,))
    monkeypatch.setattr(async_lookup, "_group_timeout", lambda group: None)
    monkeypatch.setattr(async_lookup.parallel_state, "create_group", create_group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group=None: "gloo")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda tensor, group: communicated.append(group)
    )

    initialize_engram_lookup_groups(pg_collection, SimpleNamespace(async_lookup=True))

    assert [group.ranks for group in created] == row_groups
    lookup_group = get_lookup_group(row_group)
    assert lookup_group in created and lookup_group.ranks == ranks
    assert communicated == ([lookup_group] if pipeline_rank == 0 else [])
    # A second call finds the group on either stage and creates nothing.
    initialize_engram_lookup_groups(pg_collection, SimpleNamespace(async_lookup=True))
    assert len(created) == len(row_groups)


class _Stream:
    """A stand-in for a CUDA stream: it records what it is made to wait for."""

    def __init__(self, priority=0):
        self.waited_for = []

    def wait_stream(self, stream):
        self.waited_for.append(stream)

    def wait_event(self, event):
        self.waited_for.append(event)


class _Event:
    """A stand-in for a CUDA event: it records the stream it is recorded on."""

    stream = None

    def record(self, stream):
        self.stream = stream


@pytest.fixture
def compute_stream(monkeypatch):
    """Replace the CUDA streams by stand-ins, for lookups issued on CPU tensors.

    The lookup stream is created anew from the stand-in. Returns the stand-in of the stream
    that is current outside the lookup.
    """
    current = [_Stream()]

    @contextmanager
    def stream(new_stream):
        current.append(new_stream)
        try:
            yield
        finally:
            current.pop()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", _Stream)
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current[-1])
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    monkeypatch.setattr(async_lookup, "_LOOKUP_STREAM", None)
    return current[0]


def test_prefetched_lookup_runs_its_backward_on_the_lookup_stream(tmp_path, compute_stream):
    """One rank, CPU: the lookup and the table-gradient branch run with the lookup stream
    current and give the inline lookup's output and gradients; a lookup made without
    gradients is fused as it is."""
    module = _cpu_module(tmp_path, async_lookup=True)
    with torch.no_grad():
        module.short_conv.weight.normal_(mean=0.0, std=0.05)
    lookup_stream = get_lookup_stream()
    assert lookup_stream is not compute_stream
    for table in module.embedding.tables:
        assert table.weight.grad_accumulation_stream is lookup_stream
    tokens, hidden = _cpu_inputs(batch_size=2)
    hidden.requires_grad_(True)
    weights = torch.linspace(-1.0, 1.0, hidden.numel(), dtype=hidden.dtype).view_as(hidden)

    def gradients(output):
        (output * weights).sum().backward()
        collected = [hidden.grad.clone()] + [param.grad.clone() for param in module.parameters()]
        hidden.grad = None
        module.zero_grad()
        return collected

    expected = module(hidden, tokens)
    expected_gradients = gradients(expected)

    streams = []
    module.embedding.register_forward_hook(
        lambda *unused: streams.append(("lookup", torch.cuda.current_stream()))
    )
    module.embedding.tables[0].weight.register_post_accumulate_grad_hook(
        lambda param: streams.append(("table gradient", torch.cuda.current_stream()))
    )
    handle = module.prefetch(tokens)
    # The lookup waits for what the compute stream holds, and its event is recorded behind it.
    assert lookup_stream.waited_for == [compute_stream]
    assert handle.ready.stream is lookup_stream
    with use_prefetched_lookup(SimpleNamespace(engram=module), {_ENGRAM_LAYER: handle}):
        output = module(hidden)
    assert compute_stream.waited_for == [handle.ready]
    assert torch.equal(output, expected)
    for actual, wanted in zip(gradients(output), expected_gradients):
        assert torch.equal(actual, wanted)
    assert streams == [("lookup", lookup_stream), ("table gradient", lookup_stream)]
    # The backward consumed the lookup and its graph.
    assert handle.memory is None
    with pytest.raises(RuntimeError, match="already consumed"):
        handle.attach()

    with torch.no_grad():
        handle = module.prefetch(tokens)
    assert not handle.memory.requires_grad
    assert handle.attach() is handle.memory
    with use_prefetched_lookup(SimpleNamespace(engram=module), {_ENGRAM_LAYER: handle}):
        assert torch.equal(module(hidden), expected)


class _Group:
    """A stand-in process group that only carries a size and this rank's index."""

    def __init__(self, size, rank):
        self.size, self.rank = size, rank


def _patch_groups(monkeypatch):
    """Let the module read the size and rank of :class:`_Group` stand-ins."""

    def size(group):
        return group.size if isinstance(group, _Group) else 1

    def rank(group):
        return group.rank if isinstance(group, _Group) else 0

    for namespace in (engram_module, hashing_module):
        monkeypatch.setattr(namespace, "get_pg_size", size)
        monkeypatch.setattr(namespace, "get_pg_rank", rank)


@pytest.mark.parametrize("layout", ["unpacked", "packed"])
@pytest.mark.parametrize("cp_layout", ["contiguous", "zigzag"])
def test_prefetch_follows_the_lookup_under_context_parallelism(
    tmp_path, monkeypatch, compute_stream, cp_layout, layout
):
    """Each context-parallel rank's lookup issued ahead against its inline lookup.

    Without a planned route the tokens are gathered on the compute stream and the row is
    hashed on the lookup stream; a planned route of the whole row needs no gather. In a stack
    that mixes layouts only a planned route gives a lookup: the row handed to ``prefetch`` is
    the model's, not the one the stack converts for the layer.
    """
    _patch_groups(monkeypatch)
    cp_size = 2
    packed = layout == "packed"
    tokens, _ = _cpu_inputs(batch_size=1 if packed else 2)
    packed_seq_params = row_cu_seqlens = None
    if packed:
        # The contiguous layout splits a row whose last document ends before the row does; the
        # per-document zigzag splits the documents, which then fill the row.
        document_ends = (7, _CPU_SEQ - 5) if cp_layout == "contiguous" else (8, _CPU_SEQ)
        row_cu_seqlens = torch.tensor([0, *document_ends], dtype=torch.int32)
        packed_seq_params = tagged_packed_seq_params(
            qkv_format="thd", cu_seqlens_q=row_cu_seqlens, cp_partition_mode=cp_layout
        )

    def shard(cp_rank):
        if cp_layout == "contiguous":
            return select_contiguous(tokens, cp_size, cp_rank, 1)
        if row_cu_seqlens is None:
            return select_zigzag(tokens, cp_size, cp_rank, 1)
        return tokens.index_select(1, thd_partition_index(row_cu_seqlens, cp_size, cp_rank))

    gather_streams = []

    def all_gather(parts, tensor, group):
        gather_streams.append(torch.cuda.current_stream())
        for rank, part in enumerate(parts):
            part.copy_(shard(rank))

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    for cp_rank in range(cp_size):
        options = dict(async_lookup=True, variant="deepseek_v41", cp_group=_Group(cp_size, cp_rank))
        module = _cpu_module(tmp_path, cp_layouts=(cp_layout, cp_layout), **options)
        assert module.cp_layout == cp_layout
        expected = module.lookup(shard(cp_rank), packed_seq_params)
        gather_streams.clear()

        handle = module.prefetch(shard(cp_rank), packed_seq_params)
        assert gather_streams == [compute_stream]
        assert torch.equal(handle.memory, expected.memory)
        if packed:
            assert torch.equal(handle.valid, expected.valid)
            assert torch.equal(handle.cu_seqlens, expected.cu_seqlens)
        else:
            assert handle.valid is None and handle.cu_seqlens is None

        # The stack's attention layers run in the other layout.
        other_layout = "zigzag" if cp_layout == "contiguous" else "contiguous"
        mixed = _cpu_module(tmp_path, cp_layouts=(cp_layout, other_layout), **options)
        assert mixed.prefetch(shard(cp_rank), packed_seq_params) is None
        route = module.route_global_tokens(tokens)
        if packed:
            with pytest.raises(RuntimeError, match="planned route"):
                module.prefetch(shard(cp_rank), packed_seq_params, route=route)
        else:
            for planned_module in (module, mixed):
                handle = planned_module.prefetch(shard(cp_rank), route=route)
                assert torch.equal(handle.memory, expected.memory)
        assert gather_streams == [compute_stream]
