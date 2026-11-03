# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Asynchronous Engram lookup (``--engram-async-lookup``) against the inline lookup.

The asynchronous path runs the same operations on the same data, only on the lookup stream and
communicator and at other points of the step, so every test demands bitwise equality with the
inline path: layer outputs, and every parameter's gradient after DDP's reduction. The tests
also pin down what the path is for: no host synchronization in a planned microbatch, no lookup
replay under full recompute, and the table-gradient branch launched on the lookup stream right
after the Engram layer's backward rather than after the backward of the layers in front.

Runs under torchrun with two, four or eight ranks.
"""

import os
import warnings
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.engram.async_lookup import (
    create_lookup_groups,
    destroy_lookup_groups,
    get_lookup_stream,
    join_lookup_stream_in_backward,
    local_engram_modules,
    plan_engram_lookups,
    prefetch_engram_lookups,
    use_prefetched_lookup,
)
from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.engram import Engram
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

from ._test_utils import make_pg_collection, write_tokenizer_map

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8),
    reason="requires torchrun with two, four, or eight ranks",
)

_HIDDEN = 128
_SEQ = 64
_BATCH = 2
_VOCAB = 64
_MICROBATCHES = 3
_ENGRAM_LAYER = 2

_BIT_VIEWS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}


def _bits(tensor):
    tensor = tensor.detach().contiguous()
    return tensor.view(_BIT_VIEWS.get(tensor.dtype, tensor.dtype))


def _assert_bitwise_equal(actual, expected, what):
    assert actual.dtype == expected.dtype, f"{what}: {actual.dtype} vs {expected.dtype}"
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    mismatches = (_bits(actual) != _bits(expected)).sum().item()
    assert mismatches == 0, f"{what}: {mismatches} of {actual.numel()} elements differ"


@contextmanager
def _torch_determinism(enabled):
    """Mirror --deterministic-mode, which also turns on torch.use_deterministic_algorithms."""
    if not enabled:
        yield
        return
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)
        if previous is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG")
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous


@contextmanager
def _count_synchronizations(counter):
    """Count the synchronizing CUDA calls made inside the block into ``counter[0]``."""
    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    counter[0] += sum("synchroniz" in str(warning.message) for warning in caught)


@contextmanager
def _count_all_to_alls(counter):
    real = torch.distributed.all_to_all_single

    def spy(*args, **kwargs):
        counter[0] += 1
        return real(*args, **kwargs)

    torch.distributed.all_to_all_single = spy
    try:
        yield
    finally:
        torch.distributed.all_to_all_single = real


@pytest.fixture
def model_parallel(request):
    """Initialize EP = ``request.param`` over the world; the rest is expert-DP."""
    ep_size = request.param
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size % ep_size != 0:
        pytest.skip(f"world size {world_size} is not a multiple of EP {ep_size}")
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    model_parallel_cuda_manual_seed(123)
    create_lookup_groups(parallel_state.get_expert_model_parallel_group())
    yield ep_size
    destroy_lookup_groups()
    Utils.destroy_model_parallel()


# Row-sparse table gradients (--engram-row-sparse-grad) come with a separate change; their
# cases run where EngramConfig has the option.
_HAS_ROW_SPARSE_GRAD = "row_sparse_grad" in EngramConfig.__dataclass_fields__


def _engram_config(tmp_path, row_sparse_grad=False):
    artifact = tmp_path / "map.json"
    if not artifact.exists():
        write_tokenizer_map(artifact, vocab_size=_VOCAB, layer_ids=(_ENGRAM_LAYER,))
    options = {"row_sparse_grad": True} if row_sparse_grad else {}
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
        **options,
    )


class _Block(torch.nn.Module):
    """An embedding stand-in, the layer in front of the Engram layer, and the Engram layer."""

    def __init__(self, tmp_path, deterministic, row_sparse_grad=False):
        super().__init__()
        config = SimpleNamespace(
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            perform_initialization=True,
            init_method=lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1),
            hidden_size=_HIDDEN,
            enable_hyper_connections=False,
            num_residual_streams=1,
            layernorm_epsilon=1e-5,
            sequence_parallel=False,
            deterministic_mode=deterministic,
        )
        self.embedding = torch.nn.Parameter(
            0.5 * torch.randn(_SEQ, _BATCH, _HIDDEN, dtype=torch.bfloat16, device="cuda")
        )
        self.front = torch.nn.Linear(
            _HIDDEN, _HIDDEN, bias=False, dtype=torch.bfloat16, device="cuda"
        )
        self.engram = Engram(
            config=config,
            engram_config=_engram_config(tmp_path, row_sparse_grad),
            layer_number=_ENGRAM_LAYER,
            pg_collection=make_pg_collection(
                ep=parallel_state.get_expert_model_parallel_group(),
                expt_dp=parallel_state.get_expert_data_parallel_group(),
            ),
        )
        with torch.no_grad():
            # A nonzero convolution branch, so every dense parameter gets a gradient.
            self.engram.short_conv.weight.normal_(mean=0.0, std=0.05)

    def layers(self, hidden, input_ids):
        hidden = hidden + self.front(hidden)
        return hidden + self.engram(hidden, input_ids)


def _build(tmp_path, deterministic, row_sparse_grad=False):
    from megatron.training.training import wrap_model_chunks_with_ddp

    torch.manual_seed(1234)
    block = _Block(tmp_path, deterministic, row_sparse_grad)
    for table in block.engram.embedding.tables:
        # As HybridModel does with engram_async_lookup, before DDP builds the accumulators.
        table.weight.grad_accumulation_stream = get_lookup_stream()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
        # Small buckets give each table shard its own bucket, so overlap_grad_reduce launches a
        # bucket (from the lookup stream) as soon as its table's gradient is ready.
        bucket_size=8192,
    )
    model = wrap_model_chunks_with_ddp(
        [block], TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config
    )[0]
    model.broadcast_params()
    model.train()
    return model


def _token_batches():
    rank = torch.distributed.get_rank()
    generator = torch.Generator(device="cuda").manual_seed(77 + rank)
    return [
        torch.randint(0, _VOCAB, (_BATCH, _SEQ), generator=generator, device="cuda")
        for _ in range(_MICROBATCHES)
    ]


def _run_step(
    model, token_batches, *, asynchronous, planned=True, recompute=False, events=None, stats=None
):
    """One step of ``len(token_batches)`` microbatches, in the order ``train_step`` uses.

    Returns the per-microbatch outputs and every parameter's reduced gradient.
    """
    block = model.module
    stats = (
        {"plan_syncs": [0], "forward_syncs": [0], "all_to_alls": [0]} if stats is None else stats
    )
    model.zero_grad_buffer()
    outputs = []
    with _count_all_to_alls(stats["all_to_alls"]):
        if asynchronous and planned:
            with _count_synchronizations(stats["plan_syncs"]):
                plan_engram_lookups([model], token_batches)
        for index, input_ids in enumerate(token_batches):
            last = index == len(token_batches) - 1
            with model.no_sync() if not last else nullcontext():
                lookups = None
                with _count_synchronizations(stats["forward_syncs"]):
                    if asynchronous:
                        lookups = prefetch_engram_lookups([block.engram], input_ids)
                    hidden = block.embedding * 1.0
                    if asynchronous:
                        join_lookup_stream_in_backward(hidden)

                    def run(hidden_states):
                        with use_prefetched_lookup(block, lookups):
                            return block.layers(hidden_states, input_ids)

                    if recompute:
                        output = tensor_parallel.checkpoint(run, False, hidden)
                    else:
                        output = run(hidden)
                generator = torch.Generator(device="cuda").manual_seed(
                    1000 * index + torch.distributed.get_rank()
                )
                upstream = torch.randn(
                    output.shape, generator=generator, device="cuda", dtype=output.dtype
                )
                output.backward(upstream)
                outputs.append(output.detach().clone())
    model.finish_grad_sync()
    torch.cuda.synchronize()
    grads = {
        name: param.main_grad.detach().clone() for name, param in model.module.named_parameters()
    }
    return outputs, grads


def _assert_same(result, reference):
    outputs, grads = result
    reference_outputs, reference_grads = reference
    for index, (output, expected) in enumerate(zip(outputs, reference_outputs)):
        _assert_bitwise_equal(output, expected, f"microbatch {index} output")
    assert grads.keys() == reference_grads.keys()
    for name, grad in grads.items():
        _assert_bitwise_equal(grad, reference_grads[name], f"{name} gradient")


@pytest.mark.parametrize("model_parallel", [2, 4], indirect=True)
@pytest.mark.parametrize("deterministic", [False, True], ids=["nondeterministic", "deterministic"])
@pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
@pytest.mark.parametrize("planned", [True, False], ids=["planned", "per_microbatch"])
@pytest.mark.parametrize("row_sparse_grad", [False, True], ids=["dense_grad", "row_sparse_grad"])
def test_async_lookup_matches_inline_bitwise(
    tmp_path, model_parallel, deterministic, recompute, planned, row_sparse_grad
):
    if row_sparse_grad and not _HAS_ROW_SPARSE_GRAD:
        pytest.skip("needs the row-sparse Engram table gradient (--engram-row-sparse-grad)")
    model = _build(tmp_path, deterministic, row_sparse_grad)
    tokens = _token_batches()
    with _torch_determinism(deterministic):
        # Warm up once (Triton compilation, NCCL setup), then compare fresh steps.
        _run_step(model, tokens, asynchronous=False, recompute=recompute)
        inline_stats = {"plan_syncs": [0], "forward_syncs": [0], "all_to_alls": [0]}
        reference = _run_step(
            model, tokens, asynchronous=False, recompute=recompute, stats=inline_stats
        )
        async_stats = {"plan_syncs": [0], "forward_syncs": [0], "all_to_alls": [0]}
        result = _run_step(
            model,
            tokens,
            asynchronous=True,
            planned=planned,
            recompute=recompute,
            stats=async_stats,
        )
    _assert_same(result, reference)

    # Inline: count, request and return all-to-alls forward (and again in the recompute), and
    # the return all-to-all backward. Asynchronous: the recompute does not repeat the lookup,
    # and a planned step exchanges all its counts at once.
    forwards = 2 if recompute else 1
    assert inline_stats["all_to_alls"][0] == _MICROBATCHES * (3 * forwards + 1)
    expected = 1 + 3 * _MICROBATCHES if planned else 4 * _MICROBATCHES
    assert async_stats["all_to_alls"][0] == expected

    # Inline: the counts' .tolist() and bincount-free routing, one synchronization per lookup
    # (and again in the recompute's lookup, which happens in backward). Asynchronous planned:
    # one for the whole step, none in the microbatches.
    assert inline_stats["forward_syncs"][0] == _MICROBATCHES
    if planned:
        assert async_stats["plan_syncs"][0] == 1
        assert async_stats["forward_syncs"][0] == 0
    else:
        assert async_stats["forward_syncs"][0] == _MICROBATCHES


@pytest.mark.parametrize("model_parallel", [2], indirect=True)
@pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
def test_table_gradient_branch_runs_early_on_the_lookup_stream(tmp_path, model_parallel, recompute):
    """The branch starts right after the Engram layer's backward, on the lookup stream.

    Inline, the lookup's graph sits inside the Engram layer, so autograd runs the table branch
    before the layer in front; the asynchronous lookup records its graph before the embedding,
    and only the handoff node keeps that order. Both paths are checked, so the test also shows
    that the order it asserts is the inline order.
    """
    model = _build(tmp_path, deterministic=False)
    tokens = _token_batches()[:1]
    block = model.module
    lookup_stream = get_lookup_stream()
    for asynchronous in (False, True):
        events = []

        def record(name):
            def hook(param):
                events.append((name, torch.cuda.current_stream() == lookup_stream))

            return hook

        handles = [
            table.weight.register_post_accumulate_grad_hook(record("table"))
            for table in block.engram.embedding.tables
        ]
        handles.append(block.front.weight.register_post_accumulate_grad_hook(record("front")))
        handles.append(block.embedding.register_post_accumulate_grad_hook(record("embedding")))
        try:
            _run_step(model, tokens, asynchronous=asynchronous, recompute=recompute)
        finally:
            for handle in handles:
                handle.remove()
        names = [name for name, _ in events]
        tables = [index for index, name in enumerate(names) if name == "table"]
        assert len(tables) == len(block.engram.embedding.tables), events
        assert max(tables) < names.index("front") < names.index("embedding"), events
        # The table accumulators live on the lookup stream (built there for DDP); the rest of
        # the step stays on the compute stream.
        for name, on_lookup_stream in events:
            assert on_lookup_stream == (name == "table"), events


def _hybrid_model(tmp_path, asynchronous, recompute, hyper_connections):
    from megatron.core.models.engram import apply_engram_to_hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_layer_specs import (
        gated_residual_hybrid_stack_spec,
        hybrid_stack_spec,
    )
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    pattern = "----"  # Engram on MLP layer 2, between MLP layers 1 and 3.
    kwargs = dict(
        num_layers=len(pattern),
        hidden_size=_HIDDEN,
        ffn_hidden_size=2 * _HIDDEN,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        engram_enabled=True,
        engram_async_lookup=asynchronous,
    )
    if recompute:
        kwargs.update(
            recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
        )
    if hyper_connections:
        kwargs.update(
            enable_hyper_connections=True,
            num_residual_streams=4,
            hyper_connection_variant="gated_residual",
            hc_lowrank=32,
            normalization="RMSNorm",
        )
    config = TransformerConfig(**kwargs)
    base_spec = gated_residual_hybrid_stack_spec(config) if hyper_connections else hybrid_stack_spec
    spec = apply_engram_to_hybrid_stack_spec(base_spec, _engram_config(tmp_path), pattern, config)
    torch.manual_seed(4321)
    model_parallel_cuda_manual_seed(4321)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=spec,
        vocab_size=_VOCAB,
        max_sequence_length=_SEQ,
        hybrid_layer_pattern=pattern,
    )
    for engram in local_engram_modules(model.decoder.layers):
        with torch.no_grad():
            engram.short_conv.weight.normal_(mean=0.0, std=0.05)
    return model.cuda().train()


def _hybrid_step(model, token_batches, asynchronous):
    if asynchronous:
        plan_engram_lookups([model], token_batches)
    losses = []
    for index, tokens in enumerate(token_batches):
        position_ids = torch.arange(_SEQ, device="cuda").unsqueeze(0).expand(_BATCH, -1)
        labels = torch.roll(tokens, shifts=-1, dims=1)
        loss = model(tokens, position_ids, None, labels=labels)
        generator = torch.Generator(device="cuda").manual_seed(500 + index)
        weights = torch.rand(loss.shape, generator=generator, device="cuda")
        (loss.float() * weights).sum().backward()
        losses.append(loss.detach().clone())
    torch.cuda.synchronize()
    return losses, {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


@pytest.mark.parametrize("model_parallel", [2], indirect=True)
@pytest.mark.parametrize("recompute", [False, True], ids=["no_recompute", "full_recompute"])
@pytest.mark.parametrize("hyper_connections", [False, True], ids=["plain", "hyper_connections"])
def test_hybrid_model_async_lookup_matches_inline_bitwise(
    tmp_path, model_parallel, recompute, hyper_connections
):
    """End to end through HybridModel: prefetch, HybridStack hand-over, recompute, join."""
    tokens = _token_batches()
    inline = _hybrid_model(tmp_path, False, recompute, hyper_connections)
    asynchronous = _hybrid_model(tmp_path, True, recompute, hyper_connections)
    assert [engram.layer_number for engram in asynchronous._engram_modules] == [_ENGRAM_LAYER]
    for name, param in inline.state_dict().items():
        if torch.is_tensor(param):
            _assert_bitwise_equal(asynchronous.state_dict()[name], param, f"initial {name}")

    reference_losses, reference_grads = _hybrid_step(inline, tokens, asynchronous=False)
    all_to_alls = [0]
    with _count_all_to_alls(all_to_alls):
        losses, grads = _hybrid_step(asynchronous, tokens, asynchronous=True)
    for index, (loss, expected) in enumerate(zip(losses, reference_losses)):
        _assert_bitwise_equal(loss, expected, f"microbatch {index} loss")
    assert grads.keys() == reference_grads.keys()
    assert any("engram.embedding.tables" in name for name in grads)
    for name, grad in grads.items():
        _assert_bitwise_equal(grad, reference_grads[name], f"{name} gradient")
    # One count exchange for the step; request and return forward, return backward per
    # microbatch: nothing is replayed by the recompute.
    assert all_to_alls[0] == 1 + 3 * _MICROBATCHES
