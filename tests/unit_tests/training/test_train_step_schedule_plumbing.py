# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for explicit objects threaded through `train_step`."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.models.engram import EngramConfig
from megatron.training import training as training_mod
from megatron.training.models import HybridModelConfig, T5ModelConfig
from tests.unit_tests.models.engram._test_utils import write_tokenizer_map


@pytest.fixture(autouse=True)
def training_config(run_config):
    run_config.train.micro_batch_size = 1


class _Rerun:
    """Run the forward/backward body once, then ask train_step to exit before optimizer.step."""

    _ran = False

    def should_run_forward_backward(self, data_iterator):
        run, self._ran = not self._ran, True
        return run

    def should_checkpoint_and_exit(self):
        return False, True, 0  # (checkpoint, exit, code)


class _RerunThroughOptimizer(_Rerun):
    """Run one forward/backward body and continue through the optimizer step."""

    def should_checkpoint_and_exit(self):
        return False, False, 0


def _run(
    *,
    args_overrides=None,
    model=None,
    optimizer=None,
    config=None,
    distillation=None,
    data_iterator=None,
    num_microbatches=1,
    forward_backward_func=None,
    **kwargs,
):
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
    )
    for name, value in (args_overrides or {}).items():
        setattr(args, name, value)
    captured = {}
    model = model or [SimpleNamespace(force_all_reduce=False, zero_grad_buffer=lambda: None)]
    optimizer = optimizer or SimpleNamespace(zero_grad=lambda: None, chained_optimizers=[])
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=_Rerun()),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=num_microbatches),
        mock.patch.object(training_mod, "has_nvidia_modelopt", distillation is not None),
        mock.patch.object(
            training_mod,
            "is_distillation_model",
            return_value=distillation is not None,
            create=True,
        ),
        mock.patch.object(
            training_mod, "get_tensor_shapes_adjust_fn_for_distillation", distillation, create=True
        ),
    ):
        result = training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]) if data_iterator is None else data_iterator,
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=None,
            config=config or SimpleNamespace(),
            forward_backward_func=forward_backward_func or (lambda **kw: captured.update(kw) or []),
            iteration=0,
            **kwargs,
        )
    # Preserve main's samples-seen result when appending the scheduled count.
    assert len(result) == 10
    assert result[-2] is None  # early exit before optimizer/sample accounting
    captured['returned_num_microbatches'] = result[-1]
    return captured


def test_train_step_returns_scheduled_microbatch_count():
    packed_iterator = iter([1, 2, 3])
    with (
        mock.patch.object(
            training_mod, 'wrap_data_iterator', return_value=(packed_iterator, 3, 60, 1200)
        ),
        mock.patch.object(training_mod, 'get_attr_wrapped_model', return_value=object()),
        mock.patch.object(training_mod, 'set_seqlen_stats_in_iteration'),
    ):
        captured = _run(config=SimpleNamespace(sequence_packing_scheduler='default_dynamic_cp'))
    assert captured['num_microbatches'] == captured['returned_num_microbatches'] == 3
    assert captured['data_iterator'] is packed_iterator


@pytest.mark.parametrize("encoder_decoder", [False, True])
def test_train_step_forwards_schedule_plumbing(run_config, encoder_decoder):
    run_config.model.transformer.max_seq_length_per_sample = 32
    if encoder_decoder:
        run_config.model = T5ModelConfig(
            transformer=run_config.model.transformer,
            seq_length=128,
            encoder_num_layers=2,
            encoder_seq_length=32,
            decoder_seq_length=16,
        )
    p2p, pg = object(), object()
    captured = _run(p2p_communicator=p2p, pg_collection=pg)
    assert captured["p2p_communicator"] is p2p and captured["pg_collection"] is pg
    assert captured["seq_length"] == 32
    assert "decoder_seq_length" not in captured


def test_distillation_shape_kwargs_remain_compatible_with_core(run_config):
    from megatron.core.pipeline_parallel.schedules import get_tensor_shapes

    run_config.model.transformer.max_seq_length_per_sample = 32

    def distillation(model, **kwargs):
        assert get_tensor_shapes(
            **kwargs,
            config=run_config.model.transformer,
            tp_group=mock.Mock(size=lambda: 1),
            cp_group=mock.Mock(size=lambda: 1),
        ) == [(32, 1, run_config.model.transformer.hidden_size)]
        return None

    _run(args_overrides={"modelopt_enabled": True}, distillation=distillation)


def test_train_step_defaults_to_none():
    captured = _run()
    assert captured["p2p_communicator"] is None and captured["pg_collection"] is None


def _run_engram_step(run_config, tmp_path, *, async_lookup):
    """A step of two microbatches for a hybrid model with an Engram memory, on one rank.

    Returns the batches, the model, the stand-in of the lookup plan, and the events of the
    step: the plan, then the token rows queued when the schedule starts and the batches the
    schedule draws from its iterator.
    """
    transformer = run_config.model.transformer
    transformer.engram_enabled = True
    transformer.max_seq_length_per_sample = 4
    run_config.model = HybridModelConfig(
        transformer=transformer,
        vocab_size=32,
        engram_config=EngramConfig(
            global_vocab_sizes=(17, 19),
            layer_ids=(1,),
            max_ngram_order=3,
            num_hash_heads=2,
            memory_dim=8,
            kernel_size=4,
            hash_seed=0,
            boundary_token_id=0,
            tokenizer_map_path=str(write_tokenizer_map(tmp_path / "map.json")),
            async_lookup=async_lookup,
        ),
    )
    batches = [{"tokens": 4 * index + torch.arange(4).view(1, 4), "id": index} for index in (0, 1)]
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: None,
            # No pipeline or tensor parallel peers: the token rows are not distributed.
            pg_collection=SimpleNamespace(pp=None, tp=None),
        )
    ]
    events = []

    def schedule(data_iterator, **kwargs):
        events.append(("queued rows", len(data_iterator.peek_pipeline_tokens())))
        events.extend(batch["id"] for batch in data_iterator)
        return []

    with mock.patch.object(
        training_mod, "plan_engram_lookups", side_effect=lambda *args: events.append("plan")
    ) as plan:
        _run(
            model=model,
            data_iterator=iter(batches),
            num_microbatches=len(batches),
            forward_backward_func=schedule,
        )
    return batches, model, plan, events


def test_train_step_plans_the_engram_lookups_of_a_step_before_the_schedule(run_config, tmp_path):
    batches, model, plan, events = _run_engram_step(run_config, tmp_path, async_lookup=True)

    # The plan sees the rows of every microbatch, and the schedule still draws every batch.
    assert events == ["plan", ("queued rows", 2), 0, 1]
    planned_model, planned_rows = plan.call_args.args
    assert planned_model is model
    assert len(planned_rows) == len(batches)
    for row, batch in zip(planned_rows, batches):
        assert torch.equal(row.cpu(), batch["tokens"])


@pytest.mark.parametrize(
    "async_lookup, rows_have_cu_seqlens",
    [pytest.param(False, False, id="inline_lookup"), pytest.param(True, True, id="packed_rows")],
)
def test_train_step_plans_no_engram_lookup(
    run_config, tmp_path, async_lookup, rows_have_cu_seqlens
):
    """Without the option the Engram layers look their rows up themselves. Rows with document
    boundaries are hashed with the cu_seqlens of their batch, so their lookups cannot be
    routed from the token rows ahead of the schedule. Neither reads the step ahead."""
    with mock.patch.object(
        type(run_config.data), "has_cu_seqlens", return_value=rows_have_cu_seqlens
    ):
        _, _, plan, events = _run_engram_step(run_config, tmp_path, async_lookup=async_lookup)

    plan.assert_not_called()
    assert events == [("queued rows", 0), 0, 1]


def test_train_step_uses_optimizer_ddp_config_for_mxfp8_staging():
    class _DistributedOptimizer:
        def __init__(self, overlap_param_gather):
            self.ddp_config = SimpleNamespace(
                reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=overlap_param_gather
            )
            self._copy_main_params_to_param_buffer = mock.Mock()

    overlapped = _DistributedOptimizer(overlap_param_gather=True)
    nonoverlapped = _DistributedOptimizer(overlap_param_gather=False)
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, chained_optimizers=[overlapped, nonoverlapped]
    )
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: None,
            remove_forward_pre_hook_handles={object(): object()},
        )
    ]

    with mock.patch.object(training_mod, "DistributedOptimizer", _DistributedOptimizer):
        # Global args intentionally disagree; the optimizer DDP config is authoritative.
        _run(
            args_overrides={
                "reuse_grad_buf_for_mxfp8_param_ag": False,
                "overlap_param_gather": False,
            },
            model=model,
            optimizer=optimizer,
        )

    overlapped._copy_main_params_to_param_buffer.assert_called_once_with()
    nonoverlapped._copy_main_params_to_param_buffer.assert_not_called()


def test_train_step_supports_bare_distributed_optimizer_for_mxfp8_staging():
    class _DistributedOptimizer:
        def __init__(self):
            self.ddp_config = SimpleNamespace(
                reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
            )
            self._copy_main_params_to_param_buffer = mock.Mock()

        def zero_grad(self):
            pass

    optimizer = _DistributedOptimizer()
    model = [
        SimpleNamespace(
            force_all_reduce=False,
            zero_grad_buffer=lambda: None,
            remove_forward_pre_hook_handles={object(): object()},
        )
    ]

    with mock.patch.object(training_mod, "DistributedOptimizer", _DistributedOptimizer):
        _run(model=model, optimizer=optimizer)

    optimizer._copy_main_params_to_param_buffer.assert_called_once_with()


def test_model_process_groups_fall_back_when_attribute_is_missing_or_none():
    fallback = object()
    with mock.patch.object(
        training_mod.ProcessGroupCollection, "use_mpu_process_groups", return_value=fallback
    ) as use_mpu_process_groups:
        assert training_mod._get_model_pg_collection(SimpleNamespace()) is fallback
        assert (
            training_mod._get_model_pg_collection(SimpleNamespace(pg_collection=None)) is fallback
        )

    assert use_mpu_process_groups.call_count == 2


def test_model_process_groups_are_found_through_model_wrappers():
    pg_collection = object()
    wrapped_model = SimpleNamespace(module=SimpleNamespace(pg_collection=pg_collection))

    assert training_mod._get_model_pg_collection(wrapped_model) is pg_collection


def test_train_step_invokes_tensor_metric_observer_before_optimizer():
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        qk_clip=False,
        log_num_zeros_in_grad=False,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
    )
    events = []
    pg_collection = SimpleNamespace(mp=object(), pp=object(), dp_cp=object(), dp_cp_gtp_remat=None)
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=pg_collection
        )
    ]
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, step=lambda: events.append("optimizer") or (True, 1.0, 0)
    )
    scheduler = mock.MagicMock()

    def observer(**kwargs):
        events.append("observer")
        assert model[0].force_all_reduce is False
        assert kwargs == {
            "model": model,
            "optimizer": optimizer,
            "iteration": 7,
            "pg_collection": pg_collection,
        }

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod, "get_rerun_state_machine", return_value=_RerunThroughOptimizer()
        ),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=False),
        mock.patch.object(training_mod, "_get_samples_seen_in_iteration", return_value=1),
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            config=SimpleNamespace(barrier_with_L1_time=False, log_max_attention_logit=False),
            forward_backward_func=lambda **kw: [],
            iteration=7,
            tensor_metric_observer=observer,
        )

    assert events == ["observer", "optimizer"]
    assert model[0].force_all_reduce is False
    scheduler.step.assert_called_once_with(increment=1)


def test_train_step_scopes_forward_tensor_observation_around_schedule():
    args = SimpleNamespace(
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        overlap_param_gather=False,
        seq_length=8,
        micro_batch_size=1,
        decoder_seq_length=None,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        qk_clip=False,
        log_num_zeros_in_grad=False,
        data_parallel_size=1,
        gtp_weight_remat_size=1,
    )
    events = []
    pg_collection = SimpleNamespace(mp=object(), pp=object(), dp_cp=object(), dp_cp_gtp_remat=None)
    model = [
        SimpleNamespace(
            force_all_reduce=False, zero_grad_buffer=lambda: None, pg_collection=pg_collection
        )
    ]
    optimizer = SimpleNamespace(
        zero_grad=lambda: None, step=lambda: events.append("optimizer") or (True, 1.0, 0)
    )

    class Observer:
        @contextmanager
        def observe_forward_backward(self, **kwargs):
            assert kwargs == {"model": model, "iteration": 7, "pg_collection": pg_collection}
            events.append("observe-enter")
            yield
            events.append("observe-exit")

        def __call__(self, **kwargs):
            events.append("commit")

    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod, "get_rerun_state_machine", return_value=_RerunThroughOptimizer()
        ),
        mock.patch.object(training_mod, "get_num_microbatches", return_value=1),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=False),
        mock.patch.object(training_mod, "_get_samples_seen_in_iteration", return_value=1),
        mock.patch.object(
            training_mod,
            "logical_and_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
        mock.patch.object(
            training_mod,
            "reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, group: value,
        ),
    ):
        training_mod.train_step(
            forward_step_func=lambda *a, **k: None,
            data_iterator=iter([]),
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=mock.MagicMock(),
            config=SimpleNamespace(barrier_with_L1_time=False, log_max_attention_logit=False),
            forward_backward_func=lambda **kw: events.append("forward-backward") or [],
            iteration=7,
            tensor_metric_observer=Observer(),
        )

    assert events == ["observe-enter", "forward-backward", "observe-exit", "commit", "optimizer"]
