# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Row-sparse Engram table gradients (``--engram-row-sparse-grad``) against the dense path.

The dense backward hands autograd a gradient as large as the table shard and the DDP hook adds
all of it into ``main_grad``. The row-sparse backward adds only the touched rows, with the same
rounding, so every test here demands bitwise equality with the dense path: ``main_grad`` during
accumulation, the reduced gradient buffer, the gradient norm and the parameters after optimizer
steps. Runs under torchrun with two, four or eight ranks; every case wraps the lookup in MCore
DDP and builds the Megatron optimizer, as training does.
"""

import os
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8),
    reason="requires torchrun with two, four, or eight ranks",
)

# Two prime-sized tables and one tiny table whose EP shards hold only a few rows each.
_TABLE_SIZES = (1009, 997, 13)
_EMBEDDING_DIM = 32
_OUTPUT_DIM = 8
# Above 3072 requests per table, embedding_dense_backward switches from its feature kernel to
# its sort-based kernel; the large microbatch puts every owner above it, the small one below.
_LARGE_TOKENS = 4096
_SMALL_TOKENS = 24

_BIT_VIEWS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
}


def _bits(tensor):
    tensor = tensor.detach()
    return tensor.contiguous().view(_BIT_VIEWS[tensor.dtype])


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


@pytest.fixture
def model_parallel(request):
    """Initialize EP = ``request.param`` over the world; the rest is expert-DP."""
    ep_size = request.param
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size % ep_size != 0:
        pytest.skip(f"world size {world_size} is not a multiple of EP {ep_size}")
    Utils.initialize_model_parallel(expert_model_parallel_size=ep_size)
    model_parallel_cuda_manual_seed(123)
    yield ep_size
    Utils.destroy_model_parallel()


class _LookupModel(torch.nn.Module):
    """An Engram-style multi-table lookup feeding a projection, as one hidden-state block."""

    def __init__(self, table_sizes, row_sparse_grad, deterministic, recompute=False):
        super().__init__()
        config = SimpleNamespace(
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            perform_initialization=True,
            sequence_parallel=False,
            deterministic_mode=deterministic,
        )
        self.embedding = EPShardedMultiTableEmbedding(
            config=config,
            table_sizes=table_sizes,
            embedding_dim=_EMBEDDING_DIM,
            init_method=lambda weight: torch.nn.init.normal_(weight, std=0.5),
            ep_group=parallel_state.get_expert_model_parallel_group(),
            tp_group=parallel_state.get_tensor_model_parallel_group(),
            expt_dp_group=parallel_state.get_expert_data_parallel_group(),
            row_sparse_grad=row_sparse_grad,
        )
        self.proj = torch.nn.Linear(
            _EMBEDDING_DIM * len(table_sizes),
            _OUTPUT_DIM,
            bias=False,
            dtype=torch.bfloat16,
            device="cuda",
        )
        # Full recompute checkpoints a block whose input is the hidden state, which requires
        # grad; this bias plays that role so the checkpointed lookup is replayed in backward.
        self.hidden_bias = torch.nn.Parameter(
            torch.zeros(_OUTPUT_DIM, dtype=torch.bfloat16, device="cuda")
        )
        self.recompute = recompute

    def _block(self, hidden_states, hash_ids):
        return hidden_states + self.proj(self.embedding(hash_ids).flatten(start_dim=-2))

    def forward(self, hash_ids):
        hidden_states = self.hidden_bias.expand(*hash_ids.shape[:-1], _OUTPUT_DIM)
        if self.recompute:
            return tensor_parallel.checkpoint(self._block, False, hidden_states, hash_ids)
        return self._block(hidden_states, hash_ids)


def _build(
    row_sparse_grad,
    *,
    deterministic=True,
    grad_reduce_in_fp32=True,
    overlap_grad_reduce=True,
    optimizer="adam",
    recompute=False,
    table_sizes=_TABLE_SIZES,
    **optimizer_overrides,
):
    """Build the DDP-wrapped lookup model and its Megatron optimizer.

    Both sides of a comparison are built from the same seed, so their parameters start equal.
    """
    from megatron.training.training import wrap_model_chunks_with_ddp

    torch.manual_seed(1234)
    module = _LookupModel(table_sizes, row_sparse_grad, deterministic, recompute=recompute)
    layer_wise = optimizer == "muon"
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        overlap_grad_reduce=overlap_grad_reduce,
        use_distributed_optimizer=not layer_wise,
        # Small buckets give each table shard its own bucket, so overlap_grad_reduce launches a
        # bucket as soon as its table's gradient is ready.
        bucket_size=8192,
    )
    model = wrap_model_chunks_with_ddp(
        [module],
        TransformerConfig(num_attention_heads=1, num_layers=1),
        ddp_config,
        use_layer_wise_distributed_optimizer=layer_wise,
    )[0]
    model.broadcast_params()

    if layer_wise:
        optimizer_kwargs = dict(
            optimizer="muon",
            use_distributed_optimizer=False,
            muon_tp_mode="duplicated",
            use_layer_wise_distributed_optimizer=True,
        )
    else:
        optimizer_kwargs = dict(optimizer="adam", use_distributed_optimizer=True)
    optimizer_config = OptimizerConfig(
        lr=1.0e-2,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        clip_grad=1.0,
        **optimizer_kwargs,
        **optimizer_overrides,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()
    megatron_optimizer = get_megatron_optimizer(
        optimizer_config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
    )
    return model, megatron_optimizer


def _tables(model):
    return [table.weight for table in model.module.embedding.tables]


def _microbatches(iteration, table_sizes=_TABLE_SIZES):
    """Per-rank microbatches covering repeated rows, empty owners and empty microbatches.

    0. Large: half the requests hit 48 hot rows spread over every owner, so each touched row
       repeats many times; the rest are uniform.
    1. No tokens at all on any rank.
    2. Small, and only rows owned by EP rank 0: every other owner gets zero requests.
    3. No tokens at all, as the last microbatch: overlap_grad_reduce must still launch.
    """
    rank = torch.distributed.get_rank()
    generator = torch.Generator(device="cuda").manual_seed(1000 * iteration + rank)
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    sizes = torch.tensor(table_sizes, device="cuda")

    uniform = (
        torch.rand(_LARGE_TOKENS, len(table_sizes), generator=generator, device="cuda") * sizes
    ).long()
    hot = torch.randint(
        0, 48, (_LARGE_TOKENS, len(table_sizes)), generator=generator, device="cuda"
    ) * (sizes // 48).clamp_min(1)
    use_hot = torch.rand(_LARGE_TOKENS, 1, generator=generator, device="cuda") < 0.5
    large = torch.where(use_hot, hot % sizes, uniform)

    owner0_rows = (sizes + ep_size - 1) // ep_size
    small = (
        torch.rand(_SMALL_TOKENS, len(table_sizes), generator=generator, device="cuda")
        * owner0_rows
    ).long()
    empty = torch.empty(0, len(table_sizes), dtype=torch.long, device="cuda")

    coefficients = [1.0 + 0.37 * rank + 0.11 * index for index in range(4)]
    batches = [large, empty, small, empty]
    return [(rows.unsqueeze(0), coefficient) for rows, coefficient in zip(batches, coefficients)]


def _train_iteration(model, optimizer, microbatches, offload=False, after_microbatch=None):
    """One iteration in the order ``train_step`` uses (no_sync on all but the last microbatch)."""
    if offload:
        optimizer.offload_optimizer_state_for_forward()
    model.zero_grad_buffer()
    optimizer.zero_grad()
    for index, (hash_ids, coefficient) in enumerate(microbatches):
        is_last = index == len(microbatches) - 1
        with nullcontext() if is_last else model.no_sync():
            output = model(hash_ids)
            (output.float() * coefficient).sum().backward()
        if after_microbatch is not None and not is_last:
            after_microbatch(index)
    if offload:
        optimizer.prefetch_optimizer_state_for_gradient_finalization()
    model.finish_grad_sync()
    update_successful, grad_norm, _ = optimizer.step()
    assert update_successful
    return grad_norm


def _run_pair(iterations=2, offload=False, table_sizes=_TABLE_SIZES, **build):
    """Train the dense and row-sparse models in lockstep and compare everything bitwise."""
    with _torch_determinism(build.get("deterministic", True)):
        _run_pair_impl(iterations, offload, table_sizes, **build)


def _run_pair_impl(iterations, offload, table_sizes, **build):
    dense_model, dense_optimizer = _build(False, table_sizes=table_sizes, **build)
    sparse_model, sparse_optimizer = _build(True, table_sizes=table_sizes, **build)

    for iteration in range(iterations):
        microbatches = _microbatches(iteration, table_sizes)
        dense_snapshots = {}

        def snapshot_dense(index):
            dense_snapshots[index] = [weight.main_grad.clone() for weight in _tables(dense_model)]

        def compare_sparse(index):
            for table_id, weight in enumerate(_tables(sparse_model)):
                _assert_bitwise_equal(
                    weight.main_grad,
                    dense_snapshots[index][table_id],
                    f"iteration {iteration} microbatch {index} table {table_id} main_grad",
                )
                assert weight.grad is None, "the row-sparse sentinel must not survive the hook"

        dense_norm = _train_iteration(
            dense_model, dense_optimizer, microbatches, offload, snapshot_dense
        )
        sparse_norm = _train_iteration(
            sparse_model, sparse_optimizer, microbatches, offload, compare_sparse
        )

        assert dense_norm == sparse_norm, f"iteration {iteration}: grad norm differs"
        for (name, dense_param), (_, sparse_param) in zip(
            dense_model.module.named_parameters(), sparse_model.module.named_parameters()
        ):
            _assert_bitwise_equal(
                sparse_param.main_grad, dense_param.main_grad, f"iteration {iteration} {name} grad"
            )
            _assert_bitwise_equal(sparse_param, dense_param, f"iteration {iteration} {name}")
    for weight in _tables(sparse_model):
        assert weight.grad_added_to_main_grad, "row-sparse path was not taken"


@pytest.mark.parametrize("model_parallel", [2, 4], indirect=True)
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "default"])
@pytest.mark.parametrize("grad_reduce_in_fp32", [True, False], ids=["fp32_grad", "bf16_grad"])
@pytest.mark.parametrize("overlap_grad_reduce", [True, False], ids=["overlap", "no_overlap"])
def test_row_sparse_grad_matches_dense_bitwise(
    model_parallel, deterministic, grad_reduce_in_fp32, overlap_grad_reduce
):
    """main_grad per microbatch, reduced grads, grad norm and Adam steps match bit for bit."""
    _run_pair(
        deterministic=deterministic,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        overlap_grad_reduce=overlap_grad_reduce,
    )


@pytest.mark.parametrize("model_parallel", [2], indirect=True)
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "default"])
def test_full_recompute_matches_dense_bitwise(model_parallel, deterministic):
    """Recompute replays the lookup forward in backward; the gradient is still added once."""
    _run_pair(deterministic=deterministic, recompute=True)


@pytest.mark.parametrize("model_parallel", [2], indirect=True)
@pytest.mark.parametrize(
    "variant", ["adam_offload", "adam_precision_aware", "muon", "muon_offload"]
)
def test_optimizer_variants_match_dense_bitwise(model_parallel, variant):
    """The table gradient feeds the expert Adam sibling under Muon, offload and PA unchanged."""
    offload = variant.endswith("_offload")
    overrides = {}
    if offload:
        overrides.update(
            chunked_optimizer_state_offload=True, optimizer_state_offload_chunk_size_mb=1
        )
    if variant == "adam_precision_aware":
        overrides.update(
            use_precision_aware_optimizer=True,
            main_params_dtype=torch.float32,
            main_grads_dtype=torch.float32,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
        )
    optimizer = "muon" if variant.startswith("muon") else "adam"
    _run_pair(iterations=3, offload=offload, optimizer=optimizer, **overrides)


@pytest.mark.parametrize("model_parallel", [2], indirect=True)
@pytest.mark.parametrize("row_sparse_grad", [False, True], ids=["dense", "row_sparse"])
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "default"])
def test_backward_peak_memory(model_parallel, row_sparse_grad, deterministic):
    """Only the dense path allocates a shard-sized gradient during backward.

    This also proves autograd keeps the sentinel instead of cloning it into ``.grad``.
    """
    with _torch_determinism(deterministic):
        _check_backward_peak_memory(model_parallel, row_sparse_grad, deterministic)


def _check_backward_peak_memory(ep_size, row_sparse_grad, deterministic):
    rows_per_rank = 1 << 20
    model, _ = _build(
        row_sparse_grad, deterministic=deterministic, table_sizes=(rows_per_rank * ep_size,)
    )
    table = _tables(model)[0]
    shard_bytes = table.numel() * table.element_size()
    hash_ids = torch.randint(0, rows_per_rank * ep_size, (1, 2048, 1), device="cuda")

    model.zero_grad_buffer()
    output = model(hash_ids)
    loss = output.float().sum()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    loss.backward()
    torch.cuda.synchronize()
    backward_peak = torch.cuda.max_memory_allocated() - baseline

    if torch.distributed.get_rank() == 0:
        print(
            f"\n[engram backward peak] row_sparse_grad={row_sparse_grad} "
            f"deterministic={deterministic} peak={backward_peak} shard={shard_bytes}"
        )
    assert table.grad is None
    # Activations saved for backward are freed before the table gradient exists, so the dense
    # path peaks a little below the full shard over the post-forward baseline.
    if row_sparse_grad:
        assert backward_peak < shard_bytes // 8, (backward_peak, shard_bytes)
    else:
        assert backward_peak > shard_bytes // 2, (backward_peak, shard_bytes)
    model.finish_grad_sync()


@pytest.mark.parametrize("model_parallel", [2], indirect=True)
def test_zero_out_wgrad_keeps_dense_path(model_parallel):
    """A zero_out_wgrad table keeps the dense gradient, which the DDP hook always adds."""

    def mark(model):
        for weight in _tables(model):
            weight.zero_out_wgrad = True

    dense_model, dense_optimizer = _build(False)
    sparse_model, sparse_optimizer = _build(True)
    mark(dense_model)
    mark(sparse_model)
    microbatches = _microbatches(0)
    _train_iteration(dense_model, dense_optimizer, microbatches)
    _train_iteration(sparse_model, sparse_optimizer, microbatches)
    for dense_weight, sparse_weight in zip(_tables(dense_model), _tables(sparse_model)):
        assert not sparse_weight.grad_added_to_main_grad, "dense fallback was not taken"
        _assert_bitwise_equal(sparse_weight.main_grad, dense_weight.main_grad, "main_grad")
        _assert_bitwise_equal(sparse_weight, dense_weight, "table")


@pytest.mark.parametrize("model_parallel", [2], indirect=True)
@pytest.mark.parametrize("deterministic", [True, False], ids=["deterministic", "default"])
def test_without_ddp_falls_back_to_dense_grad(model_parallel, deterministic):
    """Outside MCore DDP there is no main_grad; the table gets the ordinary dense .grad."""
    torch.manual_seed(1234)
    dense = _LookupModel(_TABLE_SIZES, False, deterministic)
    torch.manual_seed(1234)
    sparse = _LookupModel(_TABLE_SIZES, True, deterministic)
    hash_ids = _microbatches(0)[0][0]
    with _torch_determinism(deterministic):
        dense(hash_ids).float().sum().backward()
        sparse(hash_ids).float().sum().backward()
    for dense_table, sparse_table in zip(dense.embedding.tables, sparse.embedding.tables):
        assert sparse_table.weight.grad is not None
        _assert_bitwise_equal(sparse_table.weight.grad, dense_table.weight.grad, "dense .grad")
