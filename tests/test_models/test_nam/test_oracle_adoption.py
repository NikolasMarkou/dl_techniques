"""
Oracle adoption for ``models/nam`` -- Phase 5 batch B.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored.

Measured 2026-08-21 (GPU 1), one ACT step of ``NAM`` at ``hidden_size=32,
num_tree_layers=1, memory_size=8, num_read_heads=2, halt_max_steps=4``, both
dropout rates pinned to 0.0, on the three-expression batch the package's own
suite uses: **83** trainable weights, **0** dead, **0** disconnected.

This package is GREEN and was repaired, so a failure here is a REGRESSION
---------------------------------------------------------------------------
``nam``'s ULP-void guards were repaired under decisions.md D-050, and two dead
config knobs (``shift_range``, ``num_write_heads``) were removed after audits
found nothing read them. If any assertion in this file goes red, the first
hypothesis is a regression in this package, not a flaw in the instrument -- say
so loudly rather than widening a tolerance.

Why an adapter, and why it is not an oracle
-------------------------------------------
``NAM.call(carry, batch, training=...)`` takes TWO positional arguments, so it
does not fit the ``model(inputs, training=...)`` shape all three shared
instruments call. :class:`_OneACTStep` is a five-line call-site adapter that
runs ``initial_carry`` plus exactly one ACT step and returns the outputs dict.
It adds NO weights of its own -- ``len(adapter.trainable_weights) ==
len(nam.trainable_weights)`` is asserted below, because an adapter that
accidentally owned parameters would make every count in this file a claim about
the adapter.

``outputs["batch"]`` is dropped from the adapter's return value: it carries
integer token ids, which are not differentiable, and keeping them would put a
non-float leaf in front of every breaker in the smoke arm for no gain.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.nam.config import NAMConfig
from dl_techniques.models.nam.model import NAM
from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

MAX_LEN = 16
EXPRESSIONS = ["1 + 2", "3 * 4", "10 / 2"]
BATCH = len(EXPRESSIONS)

#: Measured 2026-08-21 at the config below.
GF_N_WEIGHTS = 83

#: Weight-path fragments that must be PRESENT. NAM is a tree parser feeding an
#: NTM-style memory feeding an ACT halting head; the count alone would be met
#: by any one of the three.
NAM_PATH_FRAGMENTS = ("token_embedding", "numeric_proj")

#: Every output key and its shape, at BATCH rows. The ACT head's two q-logits
#: are rank 1 on purpose.
EXPECTED_SHAPES: Dict[str, tuple] = {
    "result": (BATCH, 1),
    "valid": (BATCH, 1),
    "step_result": (BATCH, 1),
    "step_valid": (BATCH, 1),
    "op_logits": (BATCH, 4),
    "q_halt_logits": (BATCH,),
    "q_continue_logits": (BATCH,),
    "step_left_val": (BATCH, 1),
    "step_right_val": (BATCH, 1),
    "reduction_weights": (BATCH, MAX_LEN),
}


class _OneACTStep(keras.Model):
    """Adapter: ``ids -> one ACT step's outputs``. Owns no weights.

    Interface contract: ``__call__(input_ids, training=...)`` takes an
    ``(B, L)`` int array and returns the ACT step's outputs dict with the
    non-differentiable ``"batch"`` entry removed. Raises whatever ``NAM``
    raises; it adds no validation.
    """

    def __init__(self, nam: NAM, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.nam = nam

    def call(self, inputs: Any, training: Any = None) -> Dict[str, Any]:
        batch = {"input_ids": inputs}
        carry = self.nam.initial_carry(batch)
        _, outputs = self.nam(carry, batch, training=training)
        return {k: v for k, v in outputs.items() if k != "batch"}


def _config(**overrides) -> NAMConfig:
    kwargs = dict(
        hidden_size=32, num_heads=4, num_tree_layers=1, intermediate_size=64,
        memory_size=8, num_read_heads=2, max_expression_len=MAX_LEN,
        halt_max_steps=4, hidden_dropout_rate=0.0, attention_dropout_rate=0.0,
    )
    kwargs.update(overrides)
    return NAMConfig(**kwargs)


def _ids() -> np.ndarray:
    return np.asarray(ArithmeticTokenizer(max_len=MAX_LEN).encode_batch(EXPRESSIONS))


def _model(**config_overrides) -> _OneACTStep:
    model = _OneACTStep(NAM(config=_config(**config_overrides)))
    model(_ids(), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = default_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestTheAdapterAddsNothing:
    """The premise every count below rests on."""

    def test_the_adapter_owns_no_weights_of_its_own(self):
        nam = NAM(config=_config())
        adapter = _OneACTStep(nam)
        adapter(_ids(), training=False)
        assert len(adapter.trainable_weights) == len(nam.trainable_weights)
        assert {w.path for w in adapter.trainable_weights} == {
            w.path for w in nam.trainable_weights}

    def test_both_dropout_rates_are_pinned_to_zero(self):
        """A report taken under an unpinned rate reports the DRAW, not NAM."""
        config = _config()
        assert config.hidden_dropout_rate == 0.0
        assert config.attention_dropout_rate == 0.0


class TestNAMGradientFlow:

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        """A regression here is a NAM regression -- see the docstring."""
        model = _model()
        x = _ids()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(model, x)

        assert len(report) == GF_N_WEIGHTS == len(model.trainable_weights)
        for fragment in NAM_PATH_FRAGMENTS:
            assert any(fragment in path for path in report), (
                f"no weight under {fragment!r} -- a component the count above "
                f"rests on is not in the trainable set"
            )

    def test_the_gradient_assertion_can_fail(self):
        """RED proof: detach the forward and every weight must be convicted."""
        model = _model()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _ids())


class TestNAMKnobSensitivity:

    def test_hidden_size_changes_the_parameterisation(self):
        builders = {h: (lambda h=h: _model(hidden_size=h)) for h in (16, 32, 64)}
        assert_structural_knob_changes_weights(builders, knob="hidden_size")

    def test_num_read_heads_changes_the_memory_parameterisation(self):
        """A knob that reaches ONLY the NTM memory heads.

        ``hidden_size`` above would still pass if the memory were removed. This
        one would not -- and this package has already had TWO config fields
        (``shift_range``, ``num_write_heads``) deleted for being exactly this
        kind of no-op, so the live one is worth an explicit guard.
        """
        builders = {
            h: (lambda h=h: _model(num_read_heads=h)) for h in (1, 2, 4)
        }
        assert_structural_knob_changes_weights(builders, knob="num_read_heads")

    def test_num_tree_layers_changes_the_parser_depth(self):
        builders = {
            n: (lambda n=n: _model(num_tree_layers=n)) for n in (1, 2, 3)
        }
        assert_structural_knob_changes_weights(builders, knob="num_tree_layers")

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _model()), "b": (lambda: _model())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="hidden_size")


class TestNAMSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _model()
        x = _ids()

        def contract(out):
            assert isinstance(out, dict), (
                f"one ACT step returns a dict of outputs, got {type(out)}"
            )
            assert set(out) == set(EXPECTED_SHAPES), (
                f"output key set changed: {sorted(set(out))}"
            )
            for key, shape in EXPECTED_SHAPES.items():
                assert tuple(out[key].shape) == shape, (
                    f"{key}: expected {shape}, got {tuple(out[key].shape)}"
                )
                assert_finite(out[key])

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
