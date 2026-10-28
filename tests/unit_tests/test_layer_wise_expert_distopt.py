# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Expert-parallel non-emerging (Adam) param groups on the layer-wise + DistOpt path.

In layer-wise mode the emerging optimizer (Muon) owns the matrix parameters through
``LayerWiseDistributedOptimizer``, and everything else is handed to a sibling
``DistributedOptimizer``. That sibling used to be dense-only, so a parameter that is both
expert-parallel and non-Muon had nowhere to go. It is not a hypothetical combination: a hashed
n-gram / PLE memory is EP-sharded (``allreduce=False``) *and* tagged
``is_embedding_or_output_parameter``, so it is expert-parallel and Adam-managed at once.

Two things are tested, and the second matters as much as the first:

1. such a model builds a three-way chain and the expert sibling shards over the expert-DP group;
2. a Muon model *without* expert-parallel non-Muon params builds exactly the chain it built
   before -- the containment property that keeps this change off every other training route.
"""

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_size
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip layer-wise optimizer for LTS test",
)


class _MatrixOnlyModel(nn.Module):
    """Plain 2-D weights: every weight is Muon-eligible, biases go to the scalar optimizer."""

    def __init__(self, input_size=32, hidden_size=16, output_size=8):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class _ExpertTableModel(_MatrixOnlyModel):
    """Adds an EP-sharded, embedding-tagged table -- expert-parallel *and* Adam-managed.

    The two tags are what a real hashed-memory module sets (see
    ``megatron/core/models/engram/distributed_embedding.py``): ``allreduce=False`` makes the
    parameter expert-parallel, and ``is_embedding_or_output_parameter=True`` keeps it out of
    Newton-Schulz orthogonalization.
    """

    def __init__(self, table_rows=12, table_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.table = nn.Parameter(torch.empty(table_rows, table_dim))
        nn.init.normal_(self.table, std=0.02)
        self.table.allreduce = False
        self.table.is_embedding_or_output_parameter = True

    def forward(self, x):
        return super().forward(x) + self.table.sum()


def _build_optimizer(model_class, **optimizer_overrides):
    """Wrap ``model_class`` in DDP the way the layer-wise path requires, and build the optimizer.

    ``optimizer_overrides`` are extra ``OptimizerConfig`` fields (e.g. chunked state offload).
    """
    from megatron.training.training import wrap_model_chunks_with_ddp

    model = model_class().bfloat16().cuda()
    model.requires_grad_(True)

    transformer_config = TransformerConfig(num_attention_heads=1, num_layers=1)
    # use_layer_wise_distributed_optimizer=True makes wrap_model_chunks_with_ddp force
    # ddp_config.use_distributed_optimizer=True, which is what selects the split path.
    model = wrap_model_chunks_with_ddp(
        [model],
        transformer_config,
        DistributedDataParallelConfig(),
        use_layer_wise_distributed_optimizer=True,
    )[0]
    model.broadcast_params()

    optimizer_config = OptimizerConfig(
        optimizer='muon',
        lr=0.01,
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=False,
        clip_grad=1.0,
        muon_tp_mode="duplicated",
        use_layer_wise_distributed_optimizer=True,
        **optimizer_overrides,
    )

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

    optimizer = get_megatron_optimizer(
        optimizer_config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
    )
    return model, optimizer


class TestLayerWiseExpertDistOpt:
    """Expert-parallel Adam groups get their own DistOpt instance on the layer-wise path."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    def test_expert_parallel_adam_group_gets_its_own_distopt(self):
        """An EP-sharded Adam table builds a third chain member on the expert-DP group.

        Before the fix this raised "Non-emerging expert-parallel param groups are not yet
        supported"; there was nowhere to put the table.
        """
        _, optimizer = _build_optimizer(_ExpertTableModel)

        assert isinstance(optimizer, ChainedOptimizer)
        children = optimizer.chained_optimizers
        assert len(children) == 3, (
            "expected LayerWise + dense DistOpt + expert DistOpt, got "
            f"{[type(c).__name__ for c in children]}"
        )
        assert isinstance(children[0], LayerWiseDistributedOptimizer)

        # The two siblings must shard over different data-parallel groups: the dense one over
        # DP, the expert one over expert-DP. With EP=2 those have different sizes, which is the
        # cheapest way to tell them apart without reaching into private state.
        dense, expert = children[1], children[2]
        dp_size = get_pg_size(parallel_state.get_data_parallel_group(with_context_parallel=True))
        expt_dp_size = get_pg_size(parallel_state.get_expert_data_parallel_group())
        assert get_pg_size(dense.data_parallel_group) == dp_size
        assert get_pg_size(expert.data_parallel_group) == expt_dp_size

    def test_expert_table_state_is_sharded_over_expert_dp(self):
        """The point of the exercise: the table's optimizer state is split, not replicated."""
        model, optimizer = _build_optimizer(_ExpertTableModel)

        expt_dp_size = get_pg_size(parallel_state.get_expert_data_parallel_group())
        if expt_dp_size == 1:
            pytest.skip("expert-DP size 1 shards nothing; run with EP < world size")

        expert = optimizer.chained_optimizers[2]
        # Sum of this rank's shard across the param groups the expert sibling owns.
        owned = sum(p.numel() for group in expert.optimizer.param_groups for p in group['params'])
        table_numel = model.module.table.numel()
        assert owned < table_numel, (
            f"expert sibling holds {owned} elements of a {table_numel}-element table; "
            "expected a shard, so the state is still replicated"
        )

    def test_step_runs(self):
        """A full step must work end to end, not just construction."""
        model, optimizer = _build_optimizer(_ExpertTableModel)
        optimizer.zero_grad()
        out = model(torch.randn(4, 32, dtype=torch.bfloat16, device='cuda'))
        out.sum().backward()
        model.finish_grad_sync()
        _, grad_norm, _ = optimizer.step()
        assert grad_norm is None or torch.isfinite(torch.as_tensor(grad_norm))


class _ExpertLookupTableModel(_ExpertTableModel):
    """Engram-like table use: rows are gathered by a hash of the input, so per step only the
    touched rows get a gradient while dense Adam still updates every row of the shard."""

    def __init__(self, table_rows=4096, table_dim=128, **kwargs):
        super().__init__(table_rows=table_rows, table_dim=table_dim, **kwargs)

    def forward(self, x):
        rows = (x.float().abs().sum(-1) * 1000).long() % self.table.shape[0]
        lookup = F.embedding(rows, self.table).sum(-1, keepdim=True)
        return _MatrixOnlyModel.forward(self, x) + lookup


# A small chunk target so the table shard is its own (oversized, atomic) state chunk, as a
# 300-400B table shard is in production.
_STATE_OFFLOAD = dict(chunked_optimizer_state_offload=True, optimizer_state_offload_chunk_size_mb=1)


def _train_iteration(model, optimizer, inputs, offload):
    """One iteration in the order ``train_step`` uses when chunked state offload is enabled."""
    if offload:
        optimizer.offload_optimizer_state_for_forward()
    model.zero_grad_buffer()
    optimizer.zero_grad()
    model(inputs).sum().backward()
    if offload:
        # ``train_step`` wraps finalize_model_grads to start this prefetch before the reduction.
        optimizer.prefetch_optimizer_state_for_gradient_finalization()
    model.finish_grad_sync()
    update_successful, _, _ = optimizer.step()
    assert update_successful


class TestLayerWiseExpertDistOptOffload:
    """Chunked optimizer-state offload reaches the expert sibling that owns an n-gram table.

    The global ``--chunked-optimizer-state-offload`` flag is inherited by the expert sibling
    through its copied config, so the table's fp32 master and Adam moments leave the GPU between
    updates while the bf16 table and its fp32 main_grad stay resident.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    def test_expert_table_bundle_is_selected_for_offload(self):
        """Every bundle the expert sibling owns (the table shard) is selected, and the Muon and
        dense siblings get managers from the same flag."""
        _, optimizer = _build_optimizer(_ExpertLookupTableModel, **_STATE_OFFLOAD)
        _, dense, expert = optimizer.chained_optimizers

        manager = expert._optimizer_state_offloader
        assert manager is not None, "expert sibling did not inherit chunked state offload"
        owned = [p for group in expert.optimizer.param_groups for p in group['params']]
        assert owned, "expert sibling owns no table shard on this rank"
        assert {id(p) for p in manager.selected_params} == {id(p) for p in owned}
        assert all(len(chunk.params) == 1 for chunk in manager.chunks), (
            "a table shard larger than the chunk target must form its own atomic chunk"
        )
        # The flag is global: the dense sibling gets a manager too (LayerWise only does when this
        # rank owns a Muon matrix, which whole-parameter sharding does not guarantee).
        assert dense._optimizer_state_offloader is not None

    def test_table_master_and_moments_are_pinned_host_between_steps(self):
        """After an update the table's master and moments are pinned CPU tensors; the bf16 table
        and its main_grad remain on the GPU for the next forward/backward."""
        model, optimizer = _build_optimizer(_ExpertLookupTableModel, **_STATE_OFFLOAD)
        expert = optimizer.chained_optimizers[2]
        manager = expert._optimizer_state_offloader

        torch.manual_seed(7)
        for _ in range(2):
            inputs = torch.randn(16, 32, dtype=torch.bfloat16, device='cuda')
            _train_iteration(model, optimizer, inputs, offload=True)
        optimizer.offload_optimizer_state_for_forward()
        torch.cuda.synchronize()

        for param in manager.selected_params:
            assert param.device.type == 'cpu' and param.is_pinned(), "table master not offloaded"
            moments = {
                key: value
                for key, value in expert.optimizer.state[param].items()
                if isinstance(value, torch.Tensor) and value.shape == param.shape
            }
            assert set(moments) == {'exp_avg', 'exp_avg_sq'}, sorted(moments)
            for key, value in moments.items():
                assert value.device.type == 'cpu' and value.is_pinned(), f"{key} not offloaded"

        table = model.module.table
        assert table.is_cuda and table.main_grad.is_cuda

    def test_offloaded_table_training_matches_resident(self):
        """Five iterations with the table's state offloaded give bitwise the same parameters and
        table moments as the resident run: offload moves bytes, it must not change the math."""
        torch.manual_seed(1234)
        model_off, optimizer_off = _build_optimizer(_ExpertLookupTableModel, **_STATE_OFFLOAD)
        torch.manual_seed(1234)
        model_ref, optimizer_ref = _build_optimizer(_ExpertLookupTableModel)

        torch.manual_seed(100 + torch.distributed.get_rank())
        for _ in range(5):
            inputs = torch.randn(16, 32, dtype=torch.bfloat16, device='cuda')
            _train_iteration(model_off, optimizer_off, inputs, offload=True)
            _train_iteration(model_ref, optimizer_ref, inputs, offload=False)
        optimizer_off.offload_optimizer_state_for_forward()
        torch.cuda.synchronize()

        for (name, param_off), (_, param_ref) in zip(
            model_off.module.named_parameters(), model_ref.module.named_parameters()
        ):
            assert torch.equal(param_off, param_ref), f"{name} differs from the resident run"

        expert_off = optimizer_off.chained_optimizers[2]
        expert_ref = optimizer_ref.chained_optimizers[2]
        params_off = [p for g in expert_off.optimizer.param_groups for p in g['params']]
        params_ref = [p for g in expert_ref.optimizer.param_groups for p in g['params']]
        for p_off, p_ref in zip(params_off, params_ref):
            assert torch.equal(p_off.cpu(), p_ref.cpu()), "table master differs"
            for key in ('exp_avg', 'exp_avg_sq'):
                state_off = expert_off.optimizer.state[p_off][key]
                state_ref = expert_ref.optimizer.state[p_ref][key]
                assert torch.equal(state_off.cpu(), state_ref.cpu()), f"table {key} differs"


class TestLayerWiseExpertDistOptContainment:
    """A Muon model with no expert-parallel Adam params must be unaffected by the change."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    def test_no_expert_group_builds_the_original_two_member_chain(self):
        """Without an expert-parallel non-Muon param the chain is LayerWise + one dense DistOpt.

        This is the regression guard for the containment property: the expert partition must
        stay empty and contribute no optimizer, exactly as before the change.
        """
        _, optimizer = _build_optimizer(_MatrixOnlyModel)

        assert isinstance(optimizer, ChainedOptimizer)
        children = optimizer.chained_optimizers
        assert len(children) == 2, (
            "a model with no expert-parallel non-Muon params must build the same two-member "
            f"chain it built before the change, got {[type(c).__name__ for c in children]}"
        )
        assert isinstance(children[0], LayerWiseDistributedOptimizer)
        dp_size = get_pg_size(parallel_state.get_data_parallel_group(with_context_parallel=True))
        assert get_pg_size(children[1].data_parallel_group) == dp_size
