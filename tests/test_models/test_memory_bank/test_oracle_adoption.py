"""
Oracle adoption for ``models/memory_bank`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

THE MEASUREMENT THAT MAKES THIS PACKAGE'S NUMBER MEAN ANYTHING
---------------------------------------------------------------
``WaveFieldMemoryLLM`` is a 4-phase curriculum model. ``current_phase`` is a
non-trainable ``Variable`` that starts at ``PHASE_WARMUP`` (1), and the memory
pathway is gated off in that phase. A gradient report taken on a freshly
constructed model therefore convicts **fourteen** weights, and every one of them
is gated off BY DESIGN. Measured 2026-08-21 (GPU 1), tiny geometry
(``embed_dim=32, depth=4, s_lt=32, d_k=8, d_v=16, top_k=4``), one real Adam
step, ``(2, 16)`` int32 tokens:

=================================  ========  ==================================
what was measured                  weights   dead
=================================  ========  ==================================
phase 1, ``default_loss``          92        14 (13 memory + ``log_temp_nce``)
phase 2 / 3 / 4, ``default_loss``  92        1  (``log_temp_nce``)
phase 3, ``+ sum(model.losses)``   92        **0**
=================================  ========  ==================================

Two separate facts, and each is pinned two-sided below:

1. **The phase gate.** The 13 memory weights are off the graph at phase 1 and
   live at phase 2. A one-sided waiver would have hidden a genuinely broken
   gate; the phase-2 arm is what makes the phase-1 reading a gate rather than a
   defect.
2. **``log_temp_nce`` is an AUX-LOSS weight.** It is the learnable InfoNCE
   temperature (``tau = softplus(log_temp_nce) + 1e-3``), and the InfoNCE term
   reaches the tape through ``add_loss``, never through the LM head. It is
   ``None`` -- not on the backward graph at all -- for ANY loss built only from
   the forward OUTPUT, at every phase. Add ``sum(model.losses)`` and it measures
   ``1.42e-03``. The main assertion therefore uses a loss that INCLUDES the
   model's own aux losses, and asserts a clean 0-dead 92-weight report, rather
   than waiving a weight that is perfectly alive under the loss the model is
   actually trained with.

NOT re-opened here: ``wave_field_memory_llm.py``'s custom ``train_step`` is
fp16-unreachable and inert, anchored and CLOSED under decisions.md D-034. This
file never calls it -- every measurement is a plain tape over ``model(...)``.
"""

from typing import Any

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.memory_bank.phase_scheduler import PHASE_WARMUP
from dl_techniques.models.memory_bank.wave_field_memory_llm import (
    WaveFieldMemoryLLM,
)

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

VOCAB = 128
SEQ_LEN = 16
EMBED_DIM = 32
BATCH = 2

#: Measured 2026-08-21 at the tiny geometry below.
GF_N_WEIGHTS = 92

#: The 13 memory weights the phase-1 gate holds off the graph, matched by path
#: SUFFIX. Suffixes, not absolute ``Variable.path`` strings: Keras uniquifies a
#: model's name per process, so a second model of the same class built in one
#: pytest session is ``..._1/...`` and an absolute pin becomes order-dependent
#: -- green run alone, red behind any other test that builds this model.
PHASE1_GATED = frozenset({
    "memory_lt_bank/memory_K_lt",
    "memory_lt_bank/memory_V_lt",
    "memory_write_controller/memory_wm_bank/memory_wm_W_K/kernel",
    "memory_write_controller/memory_wm_bank/memory_wm_W_V/kernel",
    "memory_write_controller/memory_wm_bank/memory_wm_W_V/bias",
    "memory_read_controller/memory_read_log_temp",
    "memory_read_controller/memory_read_W_Q/kernel",
    "memory_read_controller/memory_read_W_out/kernel",
    "memory_read_controller/memory_read_W_out/bias",
    "memory_read_controller/memory_read_out_norm/gamma",
    "memory_read_controller/memory_read_out_norm/beta",
    "memory_read_controller/gate_W_g/kernel",
    "memory_read_controller/gate_W_g/bias",
})

#: The InfoNCE temperature -- an ``add_loss`` weight, never on the LM-head graph.
AUX_ONLY = "memory_read_controller/memory_read_log_temp_nce"

#: The phase at which the memory pathway is active. Any value above
#: ``PHASE_WARMUP`` does; 3 is the longest phase of the shipped curriculum.
MEMORY_PHASE = 3


def _tiny_kwargs(**overrides) -> dict:
    kwargs = dict(
        vocab_size=VOCAB, embed_dim=EMBED_DIM, depth=4, num_heads=4,
        max_seq_len=SEQ_LEN, field_size=32, d_k=8, d_v=16, s_lt=32, top_k=4,
        diversity_subsample=8, infonce_negatives=8,
        dropout_rate=0.0, attention_dropout_rate=0.0,
    )
    kwargs.update(overrides)
    return kwargs


def _tokens(batch: int = BATCH, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, VOCAB, size=(batch, SEQ_LEN)).astype("int32")


def _model(phase: int = MEMORY_PHASE, **overrides) -> WaveFieldMemoryLLM:
    model = WaveFieldMemoryLLM(**_tiny_kwargs(**overrides))
    model(_tokens(1), training=False)
    model.current_phase.assign(float(phase))
    return model


def _total_loss(model: keras.Model):
    """LM head + the model's own ``add_loss`` terms -- what it trains on."""

    def loss_fn(outputs: Any) -> Any:
        total = default_loss(outputs)
        for term in model.losses:
            total = total + keras.ops.cast(term, total.dtype)
        return total

    return loss_fn


def _assert_matches_exactly(paths, expected_suffixes) -> None:
    """``paths`` is exactly ``expected_suffixes``, matched by suffix.

    The count is asserted as well as the membership: suffix matching alone
    would accept a report containing two weights ending in the same suffix.
    """
    paths = list(paths)
    unmatched = [p for p in paths
                 if not any(p.endswith(s) for s in expected_suffixes)]
    assert not unmatched, f"unexpected weights in the set: {sorted(unmatched)}"
    missing = [s for s in expected_suffixes
               if not any(p.endswith(s) for p in paths)]
    assert not missing, f"expected weights absent from the set: {sorted(missing)}"
    assert len(paths) == len(expected_suffixes), (
        f"expected {len(expected_suffixes)} weights, got {len(paths)}: "
        f"{sorted(paths)}"
    )


def _lookup(report, suffix):
    """The single report entry whose path ends with ``suffix``."""
    matches = [p for p in report if p.endswith(suffix)]
    assert len(matches) == 1, f"{suffix!r} matched {matches}"
    return report[matches[0]]


def _one_adam_step(model: keras.Model, inputs, loss_fn=None) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = (default_loss if loss_fn is None else loss_fn)(outputs)
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestThePhaseGateIsAGateAndNotADefect:
    """Fact 1, pinned two-sided. See the module docstring."""

    def test_a_fresh_model_starts_in_the_warmup_phase(self):
        model = WaveFieldMemoryLLM(**_tiny_kwargs())
        model(_tokens(1), training=False)
        assert float(keras.ops.convert_to_numpy(model.current_phase)) == float(
            PHASE_WARMUP)

    def test_at_phase_1_exactly_the_gated_memory_weights_are_off_the_graph(self):
        model = _model(phase=PHASE_WARMUP)
        x = _tokens()
        _one_adam_step(model, x)
        report = gradient_report(model, x)
        dead = [p for p, v in report.items() if v is None or v == 0.0]
        _assert_matches_exactly(dead, set(PHASE1_GATED) | {AUX_ONLY})

    def test_at_the_memory_phase_every_one_of_them_is_live(self):
        """The discriminating half: without it, a broken gate reads the same."""
        model = _model(phase=MEMORY_PHASE)
        x = _tokens()
        _one_adam_step(model, x)
        report = gradient_report(model, x)
        for suffix in PHASE1_GATED:
            value = _lookup(report, suffix)
            assert value is not None and value > 0.0, (
                f"{suffix} is STILL dead at phase {MEMORY_PHASE}; the phase-1 "
                f"reading is then not a gate and this explanation is wrong"
            )


class TestTheInfoNCETemperatureIsAnAuxLossWeight:
    """Fact 2, pinned two-sided. See the module docstring."""

    def test_it_is_off_the_forward_output_graph_at_the_memory_phase(self):
        model = _model(phase=MEMORY_PHASE)
        x = _tokens()
        _one_adam_step(model, x)
        report = gradient_report(model, x)
        value = _lookup(report, AUX_ONLY)
        assert value is None, (
            f"expected NO gradient from an LM-head-only loss; got {value}"
        )

    def test_it_is_live_once_the_model_s_own_aux_losses_are_included(self):
        model = _model(phase=MEMORY_PHASE)
        x = _tokens()
        loss_fn = _total_loss(model)
        _one_adam_step(model, x, loss_fn)
        report = gradient_report(model, x, loss_fn=loss_fn)
        value = _lookup(report, AUX_ONLY)
        assert value is not None and value > 0.0, (
            f"the InfoNCE temperature must be live under the loss the model is "
            f"actually trained with; got {value}"
        )


class TestMemoryBankGradientFlow:

    def test_no_layer_is_stochastic(self):
        model = _model()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], (
            f"a non-zero stochastic rate is live: {stochastic}"
        )

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        """Zero waivers: the training loss reaches all 92 weights."""
        model = _model(phase=MEMORY_PHASE)
        x = _tokens()
        loss_fn = _total_loss(model)
        _one_adam_step(model, x, loss_fn)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=loss_fn)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _model(phase=MEMORY_PHASE)
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _tokens())


class TestMemoryBankKnobSensitivity:

    def test_depth_changes_the_parameterisation(self):
        # `depth` must satisfy the tap topology `L_write < L_read < depth`
        # (`wave_field_memory_llm.py:322`), which rules out `depth=2`.
        builders = {d: (lambda d=d: _model(depth=d)) for d in (4, 6, 8)}
        assert_structural_knob_changes_weights(builders, knob="depth")

    def test_s_lt_changes_the_long_term_bank(self):
        """A knob that reaches ONLY the long-term memory bank.

        ``depth`` above would still pass with no memory bank at all. This one
        would not: ``s_lt`` is the bank's row count and touches nothing in the
        transformer stack.
        """
        builders = {s: (lambda s=s: _model(s_lt=s)) for s in (16, 32, 64)}
        assert_structural_knob_changes_weights(builders, knob="s_lt")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="depth")


class TestMemoryBankSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _tokens()

        def contract(out):
            assert isinstance(out, dict), (
                f"WaveFieldMemoryLLM returns a dict, got {type(out)}"
            )
            assert {"logits", "last_hidden_state"} <= set(out), (
                f"missing an output key: {sorted(set(out))}"
            )
            assert tuple(out["logits"].shape) == (BATCH, SEQ_LEN, VOCAB), (
                f"logits: expected {(BATCH, SEQ_LEN, VOCAB)}, got "
                f"{tuple(out['logits'].shape)}"
            )
            assert tuple(out["last_hidden_state"].shape) == (
                BATCH, SEQ_LEN, EMBED_DIM), (
                f"last_hidden_state: expected {(BATCH, SEQ_LEN, EMBED_DIM)}, "
                f"got {tuple(out['last_hidden_state'].shape)}"
            )
            assert_finite(out["logits"])
            assert_finite(out["last_hidden_state"])

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
