"""Tests for PhaseScheduler.

Uses a mock model that mimics the surface PhaseScheduler reads/writes
on :class:`WaveFieldMemoryLLM`: ``_global_step``, ``current_phase``,
sub-layer trainable flags, and a ``warmup_memory_keys`` method.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import keras

from dl_techniques.models.memory_bank.phase_scheduler import (
    PhaseScheduler,
    PHASE_WARMUP,
    PHASE_FREEZE_BACKBONE,
    PHASE_FULL,
    PHASE_EXTEND,
)


# ---------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------


class _MockModel:
    """Minimal stand-in for WaveFieldMemoryLLM exposing the surface
    PhaseScheduler interacts with."""

    def __init__(self):
        self._global_step = MagicMock()
        self._global_step.numpy = MagicMock(return_value=0)

        self.current_phase = MagicMock()
        self.current_phase.assign = MagicMock()

        self.token_embeddings = MagicMock(trainable=True)
        self.position_embeddings = MagicMock(trainable=True)
        self.embed_norm = MagicMock(trainable=True)
        self.final_norm = MagicMock(trainable=True)
        self.lm_head = None
        self.blocks = [MagicMock(trainable=True) for _ in range(3)]

        self.lt_memory = MagicMock(trainable=True)
        self.wm_memory = MagicMock(trainable=True)
        self.read_controller = MagicMock(
            enable_gate_entropy=False,
            enable_load_balance=False,
            enable_z_loss=False,
            enable_diversity=False,
            enable_infonce=False,
        )
        self.write_controller = MagicMock(trainable=True)

        self.warmup_memory_keys = MagicMock()

    def set_step(self, step: int) -> None:
        self._global_step.numpy = MagicMock(return_value=step)


# ---------------------------------------------------------------------


class TestPhaseScheduler:

    def _make(self, **kwargs):
        defaults = dict(
            phase1_steps=10, phase2_steps=10, phase3_steps=10,
            warmup_dataset="dummy_dataset",
            warmup_num_batches=4,
        )
        defaults.update(kwargs)
        sched = PhaseScheduler(**defaults)
        sched.set_model(_MockModel())
        return sched

    def test_phase_boundaries(self):
        s = self._make()
        assert s._step_to_phase(0) == 1
        assert s._step_to_phase(9) == 1
        assert s._step_to_phase(10) == 2
        assert s._step_to_phase(19) == 2
        assert s._step_to_phase(20) == 3
        assert s._step_to_phase(29) == 3
        assert s._step_to_phase(30) == 4

    def test_phase1_on_train_begin(self):
        s = self._make()
        s.on_train_begin()
        assert s._last_phase == 1
        # Backbone trainable, aux losses disabled, no warmup yet.
        assert s.model.blocks[0].trainable is True
        assert s.model.read_controller.enable_load_balance is False
        s.model.warmup_memory_keys.assert_not_called()
        # current_phase.assign called with 1.
        s.model.current_phase.assign.assert_called_with(1)

    def test_phase_1_to_2_freezes_and_warms_up_once(self):
        s = self._make()
        s.on_train_begin()  # phase 1
        s.model.set_step(10)
        s.on_train_batch_begin(0)
        # Backbone now frozen.
        assert s.model.token_embeddings.trainable is False
        for blk in s.model.blocks:
            assert blk.trainable is False
        # Memory still trainable.
        assert s.model.lt_memory.trainable is True
        # Aux losses enabled.
        assert s.model.read_controller.enable_load_balance is True
        assert s.model.read_controller.enable_gate_entropy is True
        # Warmup invoked exactly once.
        s.model.warmup_memory_keys.assert_called_once_with(
            "dummy_dataset", num_batches=4,
        )

        # Crossing more batches in phase 2 should NOT re-call warmup.
        s.model.set_step(11)
        s.on_train_batch_begin(0)
        s.model.warmup_memory_keys.assert_called_once()

    def test_phase_2_to_3_unfreezes_backbone_aux_stays_on(self):
        s = self._make()
        s.on_train_begin()
        s.model.set_step(10)
        s.on_train_batch_begin(0)  # P2
        s.model.set_step(20)
        s.on_train_batch_begin(0)  # P3
        for blk in s.model.blocks:
            assert blk.trainable is True
        assert s.model.token_embeddings.trainable is True
        assert s.model.read_controller.enable_load_balance is True

    def test_phase4_no_op_extension(self):
        s = self._make()
        s.on_train_begin()
        s.model.set_step(30)
        s.on_train_batch_begin(0)
        # Phase 4: same surface as phase 3.
        for blk in s.model.blocks:
            assert blk.trainable is True
        assert s.model.read_controller.enable_load_balance is True
        s.model.current_phase.assign.assert_called_with(4)

    def test_init_from_skip_phase1(self):
        # phase1_steps=0 forces immediate Phase 2.
        s = self._make(phase1_steps=0, phase2_steps=10, phase3_steps=10)
        s.on_train_begin()
        assert s._last_phase == 2
        s.model.warmup_memory_keys.assert_called_once()

    def test_get_config(self):
        s = self._make(phase1_steps=5, phase2_steps=7, phase3_steps=11)
        cfg = s.get_config()
        assert cfg["phase1_steps"] == 5
        assert cfg["phase2_steps"] == 7
        assert cfg["phase3_steps"] == 11
        assert cfg["warmup_num_batches"] == 4


class TestPhaseConstants:
    """D2: module-level phase constants are exported and used by the
    scheduler's `_step_to_phase`."""

    def test_constants_have_expected_int_values(self):
        assert PHASE_WARMUP == 1
        assert PHASE_FREEZE_BACKBONE == 2
        assert PHASE_FULL == 3
        assert PHASE_EXTEND == 4

    def test_step_to_phase_uses_constants(self):
        s = PhaseScheduler(
            phase1_steps=10, phase2_steps=10, phase3_steps=10,
        )
        assert s._step_to_phase(0) == PHASE_WARMUP
        assert s._step_to_phase(10) == PHASE_FREEZE_BACKBONE
        assert s._step_to_phase(20) == PHASE_FULL
        assert s._step_to_phase(30) == PHASE_EXTEND


class TestEmbedDropoutWalked:
    """R2: embed_dropout was missing from the trainable walk in the
    original parallel-list implementation. The new set-driven walk
    includes it; verify by flipping phase and observing the flag."""

    def _make_model_with_layers(self):
        from unittest.mock import MagicMock

        class _LayerStub:
            def __init__(self):
                self.trainable = True

        m = _MockModel()
        m.token_embeddings = _LayerStub()
        m.position_embeddings = _LayerStub()
        m.embed_norm = _LayerStub()
        m.embed_dropout = _LayerStub()
        m.final_norm = _LayerStub()
        m.lm_head = _LayerStub()
        m.lt_memory = _LayerStub()
        m.write_controller = _LayerStub()
        m.read_controller = _LayerStub()
        # blocks (decoder list)
        m.blocks = [_LayerStub() for _ in range(2)]
        return m

    def test_embed_dropout_is_walked_through_phase_changes(self):
        m = self._make_model_with_layers()
        s = PhaseScheduler(phase1_steps=10, phase2_steps=10, phase3_steps=10)
        s.set_model(m)

        # Phase 2: backbone frozen, memory trainable.
        s._apply_phase(PHASE_FREEZE_BACKBONE)
        assert m.embed_dropout.trainable is False
        assert m.token_embeddings.trainable is False
        assert m.lt_memory.trainable is True

        # Phase 3: everything trainable.
        s._apply_phase(PHASE_FULL)
        assert m.embed_dropout.trainable is True
        assert m.token_embeddings.trainable is True
        assert m.lt_memory.trainable is True

    def test_phase_1_all_trainable_no_aux(self):
        m = self._make_model_with_layers()
        s = PhaseScheduler(phase1_steps=10, phase2_steps=10, phase3_steps=10)
        s.set_model(m)
        s._apply_phase(PHASE_WARMUP)
        assert m.embed_dropout.trainable is True
        assert m.lt_memory.trainable is True
        # aux losses off
        assert m.read_controller.enable_gate_entropy is False


class TestTopKSchedule:
    """O7: PhaseScheduler applies `model.top_k_schedule(step)` to
    `read_controller.top_k` on phase transitions."""

    def _make_model_with_schedule(self, schedule):
        from unittest.mock import MagicMock

        class _LayerStub:
            def __init__(self):
                self.trainable = True

        m = _MockModel()
        m.token_embeddings = _LayerStub()
        m.position_embeddings = _LayerStub()
        m.embed_norm = _LayerStub()
        m.embed_dropout = _LayerStub()
        m.final_norm = _LayerStub()
        m.lm_head = _LayerStub()
        m.lt_memory = _LayerStub()
        m.write_controller = _LayerStub()
        # read_controller needs `.top_k` attr (not a MagicMock so we can
        # mutate it cleanly).
        rc = _LayerStub()
        rc.top_k = 32
        rc.enable_gate_entropy = False
        rc.enable_load_balance = False
        rc.enable_z_loss = False
        rc.enable_diversity = False
        rc.enable_infonce = False
        rc.enable_v_diversity = False
        m.read_controller = rc
        m.blocks = [_LayerStub() for _ in range(2)]
        m.top_k_schedule = schedule
        return m

    def test_schedule_applied_on_phase_transition(self):
        applied_steps = []
        def schedule(step):
            applied_steps.append(step)
            return 16 if step < 10 else 8

        m = self._make_model_with_schedule(schedule)
        s = PhaseScheduler(phase1_steps=10, phase2_steps=10, phase3_steps=10)
        s.set_model(m)

        # Phase 1 transition (on_train_begin doesn't call schedule per
        # current design; only batch-begin transitions do).
        s.on_train_begin()
        assert m.read_controller.top_k == 32  # unchanged at P1 start

        # Cross into Phase 2.
        m.set_step(10)
        s.on_train_batch_begin(0)
        assert 10 in applied_steps
        assert m.read_controller.top_k == 8

        # Same phase: schedule NOT re-applied.
        applied_steps.clear()
        m.set_step(11)
        s.on_train_batch_begin(0)
        assert applied_steps == []
        assert m.read_controller.top_k == 8

    def test_schedule_handles_invalid_returns(self):
        def bad_schedule(step):
            return -1

        m = self._make_model_with_schedule(bad_schedule)
        s = PhaseScheduler(phase1_steps=5, phase2_steps=5, phase3_steps=5)
        s.set_model(m)
        s.on_train_begin()
        m.set_step(5)
        # Should NOT raise; logs warning and keeps current top_k.
        s.on_train_batch_begin(0)
        assert m.read_controller.top_k == 32


class TestLinearTopKAnneal:
    """O7 helper: `linear_top_k_anneal` produces the expected schedule."""

    def test_linear_schedule(self):
        from dl_techniques.models.memory_bank.wave_field_memory_llm import (
            linear_top_k_anneal,
        )
        sched = linear_top_k_anneal(start=64, end=8, end_step=100)
        assert sched(0) == 64
        assert sched(50) in (35, 36)  # 64 + (-56)*0.5 = 36
        assert sched(100) == 8
        assert sched(200) == 8  # past end_step => clamp
