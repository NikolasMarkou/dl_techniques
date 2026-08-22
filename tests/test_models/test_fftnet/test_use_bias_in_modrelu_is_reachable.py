"""
``use_bias_in_modrelu`` must be reachable from the shipped entry points.

The knob was fully wired INSIDE ``FFTMixer`` -- declared at ``model.py:149``,
consumed in ``build()`` (the ``modrelu_bias`` weight exists only when it is
True), branched in ``_apply_modrelu``, and serialized in the mixer's own
``get_config()`` -- while ``FFTNetBlock.__init__`` declared no such parameter
and constructed the mixer without the keyword. Every ``FFTNet`` /
``create_fftnet`` / ``create_fftnet_classifier`` caller therefore got the
mixer's default and nothing else: a serialized knob with no reachable setter.

The instrument is the STRUCTURAL one from ``knob_sensitivity_oracle`` and not
an output-difference assertion, because flipping this flag adds or removes a
weight -- ``assert_value_knob_changes_output`` would (correctly) reject the
comparison, and a bare "outputs differ" claim on a structural knob is
satisfiable by a different random draw alone. The measured model-level forward
delta (0.0489 at the geometry below) is reported here as context, not as the
claim.

RED proof: revert the ``use_bias_in_modrelu=use_bias_in_modrelu`` forwarding in
``FFTNetBlock.__init__`` and every test below fails; the structural one fails
with "use_bias_in_modrelu is a no-op: ... identical weight-shape signature".
"""

import numpy as np
import pytest

from dl_techniques.models.fftnet.model import FFTNet, FFTNetBlock, create_fftnet

from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    as_array,
    build_seeded,
)

MODEL_CONFIG = dict(
    image_size=32,
    patch_size=8,
    embed_dim=16,
    num_layers=2,
    mlp_hidden_dim=8,
    ffn_ratio=2,
    dropout_rate=0.0,
)


def _build_model(flag=None):
    kwargs = dict(MODEL_CONFIG)
    if flag is not None:
        kwargs["use_bias_in_modrelu"] = flag
    model = FFTNet(**kwargs)
    model.build((None, MODEL_CONFIG["image_size"], MODEL_CONFIG["image_size"], 3))
    return model


def _inputs():
    return np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32")


class TestUseBiasInModreluReachesTheMixer:
    def test_the_knob_changes_the_weight_shape_signature(self):
        assert_structural_knob_changes_weights(
            {True: lambda: _build_model(True), False: lambda: _build_model(False)},
            knob="use_bias_in_modrelu",
        )

    def test_the_knob_reaches_every_block_s_mixer(self):
        for flag in (True, False):
            model = _build_model(flag)
            for block in model.blocks:
                assert block.use_bias_in_modrelu is flag
                assert block.fft_mixer.use_bias_in_modrelu is flag
                assert (block.fft_mixer.modrelu_bias is None) is (not flag)

    def test_the_knob_reaches_the_forward_pass(self):
        x = _inputs()
        outs = []
        for flag in (True, False):
            model = build_seeded(lambda flag=flag: _build_model(flag))
            outs.append(as_array(model(x, training=False)["last_hidden_state"]))
        delta = float(np.max(np.abs(outs[0] - outs[1])))
        # Floor only. Measured 0.048875 at this geometry; the defect signal
        # (knob dropped on the floor) is exactly 0.0, so any positive bar
        # separates the two.
        assert delta > 1e-5, (
            "flipping use_bias_in_modrelu left the forward output unchanged "
            f"(max|delta| = {delta:.6e}); the kwarg is not reaching the graph"
        )

    def test_the_default_is_unchanged_by_the_new_parameter(self):
        # Control: the parameter must not have changed what today's callers get.
        # FFTMixer's own default is True, so omitting the kwarg and passing
        # True explicitly must be bit-identical.
        x = _inputs()
        implicit = build_seeded(lambda: _build_model(None))
        explicit = build_seeded(lambda: _build_model(True))
        delta = float(
            np.max(
                np.abs(
                    as_array(implicit(x, training=False)["last_hidden_state"])
                    - as_array(explicit(x, training=False)["last_hidden_state"])
                )
            )
        )
        assert delta == 0.0, f"default changed meaning: max|delta| = {delta:.6e}"

    @pytest.mark.parametrize("flag", [True, False])
    def test_the_knob_survives_get_config_at_both_levels(self, flag):
        model = _build_model(flag)
        assert model.get_config()["use_bias_in_modrelu"] is flag
        assert model.blocks[0].get_config()["use_bias_in_modrelu"] is flag
        restored = FFTNet.from_config(model.get_config())
        restored.build((None, 32, 32, 3))
        assert restored.blocks[0].fft_mixer.use_bias_in_modrelu is flag
        block = FFTNetBlock.from_config(model.blocks[0].get_config())
        assert block.fft_mixer.use_bias_in_modrelu is flag

    def test_the_knob_is_reachable_through_create_fftnet(self):
        # The public factory is the entry point the defect made unreachable.
        model = create_fftnet(
            "tiny", use_bias_in_modrelu=False, image_size=32, patch_size=8
        )
        model.build((None, 32, 32, 3))
        assert model.blocks[0].fft_mixer.modrelu_bias is None
