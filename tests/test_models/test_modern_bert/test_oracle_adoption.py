"""
Oracle adoption for ``models/modern_bert`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

THE BRIEF'S ENTERING-STATE CLAIM WAS STALE, AND IT WAS MEASURED
---------------------------------------------------------------
The batch-C brief carried "``test_modern_bert`` carries **3 RED node ids** in
``test_the_block_norms_use_the_models_epsilon.py``" and instructed this step to
quote them as its entering state. **Measured 2026-08-21 at HEAD, before this
file existed**::

    pytest tests/test_models/test_modern_bert -q -p no:randomly
    -> 57 passed, 1 warning in 52.16s

The directory is GREEN, 57 passed / 57 collected. Iteration-2 step 5.6 repaired
those three node ids and the carried figure was never re-derived after it. This
is the fifth carried count in this plan to fail re-derivation (D-094 records the
first four). The three node ids are green here and are re-run alongside this
file on every scoped regression.

Measured 2026-08-21 (GPU 1), one Adam step, on a 4-layer / 32-hidden encoder:

===============================  ==========  ======
arm                              weights     dead
===============================  ==========  ======
ModernBERT (dict input)          42          0
===============================  ==========  ======

Both the symmetric ``default_loss`` and the ramp loss report zero dead weights
here, so this package does NOT sit on the D-059 uniform-softmax saddle (it has
no softmax classifier head -- ``call`` returns ``last_hidden_state``). The ramp
loss is used anyway, for one reason: it is the loss every other file in this
batch uses, and a single shared choice is one thing that can be checked rather
than ten.

Every stochastic rate is pinned to ``0.0`` and every build is seeded. That is
not decoration: batch A had a BEiT arm flaky 1 run in 4 and batch B had a
MobileNetV3 arm flaky 2 runs in 5, both because an unpinned rate or an unseeded
build made the report describe the DRAW rather than the model.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.modern_bert.model import ModernBERT

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
# Imported, never re-typed: this is the ramp D-059 installed, and a second copy
# is a second thing that can drift back into a symmetric loss.
from ..precision_arm_oracle import _asymmetric_loss, flatten_tensors
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)

VOCAB = 64
SEQ_LEN = 16
HIDDEN = 32
BUILD_SEED = 0

#: Measured 2026-08-21, one Adam step, ramp loss.
GF_WEIGHTS = 42


def ramp_loss(outputs: Any) -> Any:
    return sum(_asymmetric_loss(t) for t in flatten_tensors(outputs))


def _ids(batch: int = 2, seq_len: int = SEQ_LEN, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, VOCAB, (batch, seq_len)).astype("int32")


def _inputs(**o) -> Dict[str, np.ndarray]:
    return {"input_ids": _ids(**o)}


def _bert(**o) -> ModernBERT:
    kwargs: Dict[str, Any] = dict(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_layers=4, num_heads=2,
        intermediate_size=48,
        # Pinned to 0.0, not defaulted: the shipped defaults are 0.1/0.1, and a
        # gradient report taken through live dropout reports the draw.
        hidden_dropout_rate=0.0, attention_probs_dropout_rate=0.0,
        global_attention_interval=2, local_attention_window_size=8,
        max_position_embeddings=32,
    )
    kwargs.update(o)
    return ModernBERT(**kwargs)


def _built(build_fn=_bert, seed: int = BUILD_SEED) -> ModernBERT:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_inputs(batch=1), training=False)
    return model


def _one_adam_step(model: keras.Model, inputs) -> None:
    optimizer = keras.optimizers.Adam(1e-3)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    with tf.GradientTape() as tape:
        loss = ramp_loss(model(inputs, training=True))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        [(g, v) for g, v in zip(grads, variables) if g is not None]
    )


class TestModernBERTGradientFlow:

    def test_no_layer_is_stochastic(self):
        """The premise of every measurement below."""
        model = _built()
        stochastic = [
            (layer.name, attr, getattr(layer, attr))
            for layer in model._flatten_layers(include_self=False)
            for attr in ("rate", "drop_path_rate", "dropout_rate")
            if isinstance(getattr(layer, attr, None), float)
            and getattr(layer, attr) > 0.0
        ]
        assert stochastic == [], f"a non-zero stochastic rate is live: {stochastic}"

    def test_gradients_reach_every_trainable_weight_after_one_step(self):
        model = _built()
        x = _inputs()
        _one_adam_step(model, x)

        report = assert_gradients_reach_every_trainable_weight(
            model, x, loss_fn=ramp_loss)

        assert len(report) == GF_WEIGHTS == len(model.trainable_weights)

    def test_the_symmetric_loss_agrees_no_weight_is_dead(self):
        """This package is NOT on the D-059 saddle, and that is measured.

        ``ModernBERT.call`` returns ``last_hidden_state``, not a softmax over
        classes, so the symmetric ``default_loss`` has no uniform distribution
        to sit on. Asserted rather than assumed: if a classifier head is ever
        put on this path, this test says so instead of the file silently
        changing meaning.
        """
        model = _built()
        x = _inputs()
        _one_adam_step(model, x)
        report = gradient_report(model, x)  # default_loss, the symmetric one
        dead = {p for p, v in report.items() if v is None or v == 0.0}
        assert dead == set(), f"unexpected dead weights under default_loss: {dead}"

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(
                    model, _inputs(), loss_fn=ramp_loss)


class TestModernBERTKnobSensitivity:

    def test_num_layers_changes_the_parameterisation(self):
        builders = {
            n: (lambda n=n: _built(lambda: _bert(num_layers=n)))
            for n in (2, 4, 6)
        }
        assert_structural_knob_changes_weights(builders, knob="num_layers")

    def test_intermediate_size_changes_the_parameterisation(self):
        builders = {
            s: (lambda s=s: _built(lambda: _bert(intermediate_size=s)))
            for s in (48, 64, 96)
        }
        assert_structural_knob_changes_weights(builders, knob="intermediate_size")

    def test_global_attention_interval_changes_the_parameterisation(self):
        """MEASURED: this knob is STRUCTURAL here, not a value knob.

        It was written as a value knob first, on the reasoning that deciding
        *which* blocks attend globally changes no weight shape. The value
        instrument rejected that: at ``num_layers=4`` the two settings hold
        **46 tensors / 39428 parameters** vs **43 / 39366** -- a global block
        and a local block are not the same parameterisation in this
        implementation, so the two configurations draw different random
        numbers and an output-difference claim between them would prove
        nothing. That rejection is exactly what the oracle's signature
        pre-check exists for, and it is recorded here rather than worked
        around.
        """
        builders = {
            i: (lambda i=i: _built(lambda: _bert(global_attention_interval=i)))
            for i in (2, 4)
        }
        signatures = assert_structural_knob_changes_weights(
            builders, knob="global_attention_interval")
        assert len(signatures[2]) == 46 and len(signatures[4]) == 43, (
            f"the measured tensor counts moved: "
            f"{ {k: len(v) for k, v in signatures.items()} }"
        )

    def test_global_rope_theta_reaches_the_forward_pass(self):
        """The VALUE knob, and the one most able to be silently dropped.

        ``global_rope_theta`` is the RoPE base frequency of the global-attention
        blocks. It changes NO weight shape -- both settings hold bit-identical
        weights under a fixed seed, which the oracle asserts first -- so a
        shape-only sweep cannot see it and a build that never routed the
        argument into the rotary embedding would pass one. Measured
        ``max|delta| = 1.487e-01`` between the two shipped-ish values, four
        orders of magnitude above the instrument's 1e-5 floor.
        """
        x = _inputs()
        builders = {
            theta: (lambda theta=theta: _bert(global_rope_theta=theta))
            for theta in (1e4, 1.6e5)
        }
        deltas = assert_value_knob_changes_output(
            builders, x, knob="global_rope_theta",
            extract=lambda o: o["last_hidden_state"],
        )
        assert all(d > 1e-3 for d in deltas.values()), deltas

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_layers")

    def test_the_value_knob_assertion_can_fail(self):
        """The same theta passed twice must be reported inert."""
        x = _inputs()
        builders = {
            k: (lambda: _bert(global_rope_theta=1.6e5)) for k in ("a", "b")
        }
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_value_knob_changes_output(
                builders, x, knob="global_rope_theta",
                extract=lambda o: o["last_hidden_state"],
            )


class TestModernBERTSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        x = _inputs()

        def contract(out):
            assert isinstance(out, dict), (
                f"ModernBERT.call returns a dict, got {type(out)}")
            assert set(out) == {"last_hidden_state", "attention_mask"}, (
                f"unexpected key set {sorted(out)}")
            hidden = out["last_hidden_state"]
            assert tuple(hidden.shape) == (2, SEQ_LEN, HIDDEN), (
                f"expected {(2, SEQ_LEN, HIDDEN)}, got {tuple(hidden.shape)}")
            assert_finite(hidden)

        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }

    def test_the_omitted_attention_mask_is_returned_as_all_ones(self):
        """The documented contract that makes ``predict()`` work on one key.

        ``call``'s docstring promises the output structure is independent of
        the input -- an omitted mask comes back as all-ones rather than as
        ``None``. Pinned here because the smoke contract above only checks the
        key is PRESENT.
        """
        model = _built()
        out = model(_inputs(), training=False)
        mask = keras.ops.convert_to_numpy(out["attention_mask"])
        np.testing.assert_array_equal(mask, np.ones_like(mask))

    def test_a_sequence_longer_than_the_positional_table_is_refused(self):
        """A build-time contract, not an output contract.

        ``max_position_embeddings=32``; a 64-token batch has no positional row
        to read. Asserted so the package's own limit is a test rather than a
        docstring.
        """
        model = _built()
        with pytest.raises(Exception):
            model(_inputs(seq_len=64), training=False)
