"""The predictor's LayerNorm epsilon, and why it is not a cosmetic constant.

``CausalSelfAttnMLPBlock`` built its two ``LayerNormalization`` layers without
an ``epsilon``, inheriting Keras' default of **1e-3**, while every other
normalization in the same forward pass — the ``CliffordNetBlock`` and
``CausalCliffordNetBlock`` this predictor is assembled from — runs at **1e-6**
(``layers/geometric/clifford_block.py:1443``, chosen there to agree with
``layers/norms/factory.py``'s ``setdefault('epsilon', 1e-6)``). A 1000x spread
inside one stack.

An epsilon is invisible to every shape, dtype, count and finiteness assertion
in the suite, so the config assertions below are honestly labelled as *config*
assertions. The forward arm is what makes them worth having: it measures how
large the difference actually is at a realistic activation scale.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import zipfile

import keras
import numpy as np

from dl_techniques.models.video_jepa.predictor import (
    CausalSelfAttnMLPBlock,
    _NORM_EPSILON,
)

REFERENCE_EPSILON = 1e-6
KERAS_DEFAULT_EPSILON = 1e-3


def _block(**kwargs):
    return CausalSelfAttnMLPBlock(
        dim=16, num_heads=2, dim_head=8, mlp_dim=32, **kwargs
    )


class TestEpsilonReachesTheConstructedLayer:
    """Config assertions — necessary, not sufficient. See the forward arm."""

    def test_module_constant_is_the_clifford_block_value(self):
        assert _NORM_EPSILON == REFERENCE_EPSILON

    def test_both_layer_norms_carry_it(self):
        block = _block()
        assert block.ln1.epsilon == REFERENCE_EPSILON
        assert block.ln2.epsilon == REFERENCE_EPSILON
        assert block.ln1.epsilon != KERAS_DEFAULT_EPSILON, (
            "the Keras default is what this test exists to keep out"
        )

    def test_it_survives_a_config_round_trip(self):
        block = _block()
        clone = CausalSelfAttnMLPBlock.from_config(block.get_config())
        assert clone.ln1.epsilon == REFERENCE_EPSILON
        assert clone.ln2.epsilon == REFERENCE_EPSILON


class TestTheEpsilonIsLiveInTheForwardPass:
    """How much the value is worth, measured rather than asserted in the abstract."""

    def test_layer_norm_output_moves_by_order_one_at_a_realistic_scale(self):
        """At an activation scale near the epsilon, 1e-3 dominates the variance.

        Measured 2026-08-18 (CPU, float32): inputs at scale 1e-2 (variance
        ~1e-4, i.e. BELOW the 1e-3 that was in force) give a max absolute
        difference of ~1.54 between the two epsilons — the normalization is
        being done largely by the epsilon rather than by the data. This is the
        mechanism by which a wrong epsilon is "silent": nothing is NaN, nothing
        is the wrong shape, the outputs are simply scaled wrong.
        """
        x = (np.random.RandomState(0).randn(2, 4, 16) * 1e-2).astype("float32")

        reference = np.asarray(
            keras.layers.LayerNormalization(epsilon=REFERENCE_EPSILON)(x)
        )
        keras_default = np.asarray(
            keras.layers.LayerNormalization(epsilon=KERAS_DEFAULT_EPSILON)(x)
        )

        delta = float(np.max(np.abs(reference - keras_default)))
        assert delta > 1.0, (
            f"expected an order-one divergence at this scale, measured {delta}"
        )

    def test_the_block_output_moves_too_but_layer_scale_damps_it(self):
        """The same A/B through the whole block, for honest magnitudes.

        ``layer_scale_init=1e-5`` makes the block near-identity at init, so the
        block-level delta is ~1e-5 relative even though the LayerNorm's own
        output moved by ~1.5. Recorded so nobody reads the small number above
        as "the epsilon does not matter": it is LayerScale that is small here,
        and LayerScale is trained away.
        """
        def run(epsilon):
            keras.utils.set_random_seed(0)
            block = _block()
            block.ln1.epsilon = epsilon
            block.ln2.epsilon = epsilon
            x = (np.random.RandomState(0).randn(2, 4, 16) * 1e-2).astype("float32")
            return np.asarray(block(x, training=False))

        delta = float(np.max(np.abs(run(REFERENCE_EPSILON) - run(KERAS_DEFAULT_EPSILON))))
        assert delta > 0.0, "the epsilon must reach the forward pass at all"


class TestCheckpointImpactIsWhatTheDecisionEntryClaims:
    """Pins the serialization fact D-028's impact statement rests on."""

    def test_the_inner_layer_norm_epsilon_is_NOT_stored_in_a_saved_model(self, tmp_path):
        """So a pre-existing checkpoint reloads with the NEW epsilon.

        ``CausalSelfAttnMLPBlock`` creates its LayerNorms in ``__init__``, and a
        subclassed layer's saved config lists only its own constructor
        arguments. On load, ``__init__`` runs against the CURRENT source, so the
        sublayer is rebuilt with whatever epsilon the code says today — the
        saved file cannot pin it. Weights still load; numerics change slightly.

        If this ever fails, D-028's checkpoint-impact paragraph is wrong and
        should be corrected, not this test.
        """
        inputs = keras.Input(shape=(4, 16))
        outputs = _block(name="blk")(inputs)
        model = keras.Model(inputs, outputs)

        path = tmp_path / "m.keras"
        model.save(path)

        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read("config.json"))

        block_config = config["config"]["layers"][1]["config"]
        assert block_config["name"] == "blk"
        assert "epsilon" not in json.dumps(block_config), (
            "the block's saved config now carries an epsilon; if it became a "
            "constructor argument, old checkpoints DO pin the old value"
        )
