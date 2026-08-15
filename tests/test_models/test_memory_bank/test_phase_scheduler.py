"""Tests for PhaseScheduler.

Two layers of testing live here, and the second one is the point.

The first uses a stand-in model exposing the surface PhaseScheduler
reads/writes (``_global_step``, ``current_phase``,
``set_backbone_optimizer_active``, ``warmup_memory_keys``) and checks the
boundary arithmetic.

The second (:class:`TestCurriculumReachesTheTracedGraph`) runs a real
``fit()``, because a stand-in model cannot see the defect this module was
rewritten for: ``fit()`` traces ``train_function`` **before**
``on_train_begin``, and rebuilds it only from ``compile()``, so a
curriculum built out of Python attribute flips (``layer.trainable``,
``read_controller.enable_gate_entropy``) is inert from the first batch
onward. Only a phase transition that happens *after* the trace exposes
that, which is why those tests use ``phase1_steps=1`` and at least three
steps. ``phase1_steps=0`` is **forbidden** in them: it applies phase 2
from ``on_train_begin``, before any transition, and the suite that used it
was green against a completely inert curriculum.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import keras
import tensorflow as tf
from keras import ops

from dl_techniques.losses import MaskedCausalLMLoss
from dl_techniques.models.memory_bank.phase_scheduler import (
    PhaseScheduler,
    PHASE_WARMUP,
    PHASE_FREEZE_BACKBONE,
    PHASE_FULL,
    PHASE_EXTEND,
)
from dl_techniques.models.memory_bank.wave_field_memory_llm import (
    WaveFieldMemoryLLM,
)


# ---------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------


class _LayerStub:
    """Records every write to a curriculum-relevant Python attribute.

    Recording the writes — rather than reading the flags at the end — is
    load-bearing: the old curriculum flipped `trainable` off at phase 2
    and back on at phase 3, so a stub inspected after a full phase cycle
    shows every flag back at its initial value and the pin passes against
    the defect.
    """

    _TRACKED = ("trainable",)

    def __init__(self):
        object.__setattr__(self, "writes", [])
        self.trainable = True
        self.writes.clear()

    def __setattr__(self, name, value):
        if name in self._TRACKED:
            self.writes.append((name, value))
        object.__setattr__(self, name, value)


class _ReadControllerStub(_LayerStub):
    _TRACKED = (
        "trainable", "enable_gate_entropy", "enable_load_balance",
        "enable_z_loss", "enable_diversity", "enable_infonce",
        "enable_v_diversity",
    )

    def __init__(self):
        super().__init__()
        self.top_k = 32
        self.enable_gate_entropy = False
        self.enable_load_balance = False
        self.enable_z_loss = False
        self.enable_diversity = False
        self.enable_infonce = False
        self.enable_v_diversity = False
        self.writes.clear()


class _MockModel:
    """Minimal stand-in for WaveFieldMemoryLLM exposing the surface
    PhaseScheduler interacts with, plus the Python attributes the
    curriculum must NOT touch."""

    def __init__(self):
        self._global_step = MagicMock()
        self._global_step.numpy = MagicMock(return_value=0)

        self.current_phase = MagicMock()
        self.current_phase.assign = MagicMock()

        # Recorded, not asserted-on-a-mock: the scheduler's only eager
        # side effect besides `current_phase.assign`.
        self.backbone_active_calls = []
        self.set_backbone_optimizer_active = self.backbone_active_calls.append

        self.token_embeddings = _LayerStub()
        self.position_embeddings = _LayerStub()
        self.embed_norm = _LayerStub()
        self.embed_dropout = _LayerStub()
        self.final_norm = _LayerStub()
        self.lm_head = _LayerStub()
        self.blocks = [_LayerStub() for _ in range(3)]

        self.lt_memory = _LayerStub()
        self.wm_memory = _LayerStub()
        self.read_controller = _ReadControllerStub()
        self.write_controller = _LayerStub()

        self.warmup_memory_keys = MagicMock()

    def all_layers(self):
        return [
            self.token_embeddings, self.position_embeddings, self.embed_norm,
            self.embed_dropout, self.final_norm, self.lm_head,
            self.lt_memory, self.wm_memory, self.read_controller,
            self.write_controller, *self.blocks,
        ]

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
        s.model.current_phase.assign.assert_called_with(1)
        # Backbone optimizer left running; no warmup yet.
        assert s.model.backbone_active_calls == [True]
        s.model.warmup_memory_keys.assert_not_called()

    def test_phase_1_to_2_freezes_and_warms_up_once(self):
        s = self._make()
        s.on_train_begin()  # phase 1
        s.model.set_step(10)
        s.on_train_batch_begin(0)

        s.model.current_phase.assign.assert_called_with(2)
        # Backbone optimizer frozen at the boundary.
        assert s.model.backbone_active_calls == [True, False]
        s.model.warmup_memory_keys.assert_called_once_with(
            "dummy_dataset", num_batches=4,
        )

        # Crossing more batches in phase 2 should NOT re-call warmup.
        s.model.set_step(11)
        s.on_train_batch_begin(0)
        s.model.warmup_memory_keys.assert_called_once()

    def test_phase_2_to_3_unfreezes_backbone(self):
        s = self._make()
        s.on_train_begin()
        s.model.set_step(10)
        s.on_train_batch_begin(0)  # P2
        s.model.set_step(20)
        s.on_train_batch_begin(0)  # P3
        s.model.current_phase.assign.assert_called_with(3)
        assert s.model.backbone_active_calls == [True, False, True]

    def test_phase4_no_op_extension(self):
        s = self._make()
        s.on_train_begin()
        s.model.set_step(30)
        s.on_train_batch_begin(0)
        s.model.current_phase.assign.assert_called_with(4)
        assert s.model.backbone_active_calls[-1] is True

    def test_init_from_skip_phase1(self):
        # phase1_steps=0 forces immediate Phase 2. Kept as a boundary
        # case ONLY; it can never observe a post-trace transition, which
        # is why TestCurriculumReachesTheTracedGraph uses phase1_steps=1.
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

    def test_missing_backbone_gate_is_tolerated(self):
        """A model without `set_backbone_optimizer_active` (another
        architecture reusing the callback) must not raise."""
        s = PhaseScheduler(phase1_steps=1, phase2_steps=1, phase3_steps=1)
        m = _MockModel()
        del m.set_backbone_optimizer_active
        s.set_model(m)
        s.on_train_begin()
        m.set_step(1)
        s.on_train_batch_begin(0)
        m.current_phase.assign.assert_called_with(2)


class TestCurriculumMutatesNoPythonState:
    """C-28 regression pin. The curriculum is a `Variable` assignment and
    nothing else; a `layer.trainable` / `enable_*` flip cannot reach an
    already-traced `train_function`, so its reappearance here is a
    defect, not a style question."""

    def test_no_trainable_flag_or_enable_flag_is_touched(self):
        s = PhaseScheduler(phase1_steps=10, phase2_steps=10, phase3_steps=10)
        m = _MockModel()
        s.set_model(m)

        for phase in (
            PHASE_WARMUP, PHASE_FREEZE_BACKBONE, PHASE_FULL, PHASE_EXTEND,
        ):
            s._apply_phase(phase)

        writes = [
            (layer.__class__.__name__, w)
            for layer in m.all_layers() for w in layer.writes
        ]
        assert writes == [], (
            "PhaseScheduler wrote a Python curriculum attribute "
            f"({writes[:4]}...). `trainable` and the `enable_*` aux flags "
            "are invisible to an already-traced train_step, which is how "
            "the whole curriculum came to be inert (C-28)"
        )


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


class TestTopKSchedule:
    """O7: PhaseScheduler applies `model.top_k_schedule(step)` to
    `read_controller.top_k` on phase transitions."""

    def _make_model_with_schedule(self, schedule):
        m = _MockModel()
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


# ---------------------------------------------------------------------
# C-28 — the curriculum inside a real traced fit()
# ---------------------------------------------------------------------


_VOCAB = 64
_SEQ = 8


def _tiny_kwargs():
    return dict(
        vocab_size=_VOCAB,
        embed_dim=32,
        depth=3,
        num_heads=4,
        max_seq_len=_SEQ,
        field_size=16,
        d_k=8,
        d_v=16,
        s_lt=16,
        top_k=4,
        diversity_subsample=8,
        infonce_negatives=8,
    )


class AuxProbeLLM(WaveFieldMemoryLLM):
    """Records the read controller's total |aux loss| for the step, from
    inside the traced graph.

    The probe is a ``Variable`` assigned during ``compute_loss``, i.e. on
    the traced path — reading ``model.losses`` eagerly after ``fit()``
    would measure a *different* graph than the one that trained, which is
    exactly the confusion C-28 lived in. It records the sum of absolute
    values because the gate-entropy term is negative by construction and a
    signed sum could cancel to ~0 while every term is live.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aux_probe = self.add_weight(
            name="aux_probe_total", shape=(), initializer="zeros",
            trainable=False, dtype="float32",
        )

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None,
                     **kwargs):
        aux = list(self.read_controller.losses)
        if aux:
            total = ops.cast(0.0, "float32")
            for term in aux:
                total = total + ops.cast(ops.sum(ops.abs(term)), "float32")
            self.aux_probe.assign(total)
        return super().compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight,
        )


class _StepRecorder(keras.callbacks.Callback):
    """Snapshots the probe + two watched weights after every batch."""

    def __init__(self, backbone_var, memory_var):
        super().__init__()
        self._backbone_var = backbone_var
        self._memory_var = memory_var
        self.aux = []
        self.phase = []
        self.backbone = []
        self.memory = []

    def on_train_batch_end(self, batch, logs=None):
        self.aux.append(float(ops.convert_to_numpy(self.model.aux_probe)))
        self.phase.append(float(ops.convert_to_numpy(self.model.current_phase)))
        self.backbone.append(np.array(ops.convert_to_numpy(self._backbone_var)))
        self.memory.append(np.array(ops.convert_to_numpy(self._memory_var)))


def _compiled_probe_model(weight_decay: float = 0.01, **overrides):
    kwargs = _tiny_kwargs()
    kwargs.update(overrides)
    m = AuxProbeLLM(**kwargs)
    m(np.random.randint(0, _VOCAB, size=(1, _SEQ)).astype(np.int32),
      training=False)
    m.compile(
        backbone_optimizer=keras.optimizers.AdamW(
            learning_rate=1e-2, weight_decay=weight_decay,
        ),
        memory_optimizer=keras.optimizers.AdamW(
            learning_rate=1e-2, weight_decay=weight_decay,
        ),
        loss={"logits": MaskedCausalLMLoss()},
    )
    m.output_names = ["logits"]
    return m


def _one_batch_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, _VOCAB, size=(2, _SEQ)).astype(np.int32)
    y = rng.integers(0, _VOCAB, size=(2, _SEQ)).astype(np.int32)
    return tf.data.Dataset.from_tensor_slices((x, {"logits": y})).batch(2)


class TestCurriculumReachesTheTracedGraph:
    """C-28 RED proof.

    Regime, and why it is not negotiable: ``phase1_steps=1`` with three
    steps, so the phase 1 -> 2 transition happens on the SECOND step —
    after ``make_train_function()`` has already traced the graph. At
    ``phase1_steps=0`` phase 2 is applied from ``on_train_begin`` and no
    transition ever occurs, so both assertions below pass against a
    curriculum that is completely inert. Do not "simplify" this to
    ``phase1_steps=0`` or to a single ``fit`` step.
    """

    def _run(self, epochs=3):
        m = _compiled_probe_model()
        rec = _StepRecorder(
            backbone_var=m.final_norm.gamma,
            memory_var=m.read_controller.W_g.kernel,
        )
        backbone_at_start = np.array(
            ops.convert_to_numpy(m.final_norm.gamma)
        )
        m.fit(
            _one_batch_dataset(),
            epochs=epochs,
            verbose=0,
            callbacks=[
                PhaseScheduler(
                    phase1_steps=1, phase2_steps=1_000, phase3_steps=1_000,
                ),
                rec,
            ],
        )
        return m, rec, backbone_at_start

    def test_transition_reaches_the_phase_gate(self):
        _, rec, _ = self._run()
        assert rec.phase == [1.0, 2.0, 2.0], rec.phase

    def test_aux_losses_start_contributing_after_the_transition(self):
        """The anti-collapse losses are exactly 0 in phase 1 and non-zero
        from the transition on."""
        _, rec, _ = self._run()
        assert rec.aux[0] == 0.0, (
            f"phase-1 aux total should be exactly 0, got {rec.aux[0]}"
        )
        assert rec.aux[1] > 0.0, (
            "no anti-collapse aux loss contributed anything on the step "
            "after the phase 1 -> 2 transition; the curriculum never "
            "reached the traced graph (C-28). "
            f"aux per step = {rec.aux}"
        )
        assert rec.aux[2] > 0.0, rec.aux

    def test_backbone_freezes_while_memory_keeps_training(self):
        """After the transition a backbone weight stops moving entirely,
        while a memory weight keeps moving. The second half is the
        isolating arm: a global stop (a broken optimizer, a zero loss)
        would satisfy the first assertion on its own."""
        _, rec, backbone_at_start = self._run()

        moved_in_phase1 = np.max(np.abs(rec.backbone[0] - backbone_at_start))
        assert moved_in_phase1 > 0.0, (
            "liveness arm failed: the backbone weight did not move during "
            "phase 1, so 'it stops moving in phase 2' proves nothing"
        )

        frozen_delta = max(
            float(np.max(np.abs(rec.backbone[1] - rec.backbone[0]))),
            float(np.max(np.abs(rec.backbone[2] - rec.backbone[1]))),
        )
        assert frozen_delta == 0.0, (
            "the backbone kept training after the phase 1 -> 2 transition "
            f"(max |delta| = {frozen_delta}); the freeze never reached the "
            "traced graph (C-28)"
        )

        memory_delta = float(np.max(np.abs(rec.memory[2] - rec.memory[1])))
        assert memory_delta > 0.0, (
            "the memory gate stopped training too — this is a global stop, "
            "not a backbone freeze"
        )

    def test_gradient_freeze_holds_with_no_callback_attached(self):
        """The gradient half of the freeze is a property of the saved
        `current_phase`, not of the callback: a model reloaded
        mid-curriculum and trained with no PhaseScheduler still stops
        gradient reaching its backbone. This is what the in-graph mask
        buys over zeroing the optimizer's learning rate.

        `weight_decay=0.0` is load-bearing here, and measured, not
        assumed: AdamW's decoupled decay is not a gradient-driven term
        (`variable -= variable * wd * lr`), so NO gradient mask can freeze
        a weight under it — at `weight_decay=0.01, lr=1e-2` the backbone
        still drifts exactly `1e-4` per step. Only the callback's
        learning-rate gate closes that, which is why both halves exist.
        """
        m = _compiled_probe_model(weight_decay=0.0)
        m.current_phase.assign(float(PHASE_FREEZE_BACKBONE))

        backbone_before = np.array(ops.convert_to_numpy(m.final_norm.gamma))
        memory_before = np.array(
            ops.convert_to_numpy(m.read_controller.W_g.kernel)
        )
        m.fit(_one_batch_dataset(), epochs=2, verbose=0)

        backbone_after = np.array(ops.convert_to_numpy(m.final_norm.gamma))
        memory_after = np.array(
            ops.convert_to_numpy(m.read_controller.W_g.kernel)
        )
        assert float(np.max(np.abs(backbone_after - backbone_before))) == 0.0
        assert float(np.max(np.abs(memory_after - memory_before))) > 0.0


class TestBackboneOptimizerGate:

    def test_schedule_learning_rate_is_refused_by_name(self):
        """A LearningRateSchedule cannot be zeroed, so the phase-2 freeze
        would silently not happen. It must raise, naming the constraint."""
        m = _compiled_probe_model()
        m.compile(
            backbone_optimizer=keras.optimizers.AdamW(
                learning_rate=keras.optimizers.schedules.ExponentialDecay(
                    1e-2, decay_steps=10, decay_rate=0.9,
                ),
            ),
            memory_optimizer=keras.optimizers.AdamW(learning_rate=1e-2),
            loss={"logits": MaskedCausalLMLoss()},
        )
        with pytest.raises(ValueError, match="assignable learning rate"):
            m.set_backbone_optimizer_active(False)

    def test_gate_restores_the_compiled_rate(self):
        m = _compiled_probe_model()
        base = float(ops.convert_to_numpy(m.backbone_optimizer.learning_rate))
        m.set_backbone_optimizer_active(False)
        assert float(
            ops.convert_to_numpy(m.backbone_optimizer.learning_rate)
        ) == 0.0
        m.set_backbone_optimizer_active(True)
        assert float(
            ops.convert_to_numpy(m.backbone_optimizer.learning_rate)
        ) == pytest.approx(base)
