"""F-63 and the darkir half of F-64.

F-63 -- `middle_blk_num_dec == 0` was accepted and turned
`layers.Add(name="middle_residual")([x, x_light])` into `2 * x_light`, because
the middle-decoder loop never runs and `x` is still `x_light` when the Add
fires. MEASURED at HEAD-before-fix, CPU:
`max|middle_residual - 2*x_light| = 0.0` against `max|2*x_light| = 4.427` --
an exact scalar doubling, not a residual. `middle_blk_num_enc == 0` is NOT
degenerate and stays legal.

F-64 -- both block docstrings called beta/gamma "learnable scalar parameters"
while `build` allocates `(1, 1, 1, channels)`. The shape assertion below is what
makes the corrected docstring load-bearing.

CPU only.
"""

import numpy as np
import pytest
import keras

from dl_techniques.models.darkir.model import (
    create_darkir_model,
    DarkIREncoderBlock,
    DarkIRDecoderBlock,
)

TINY = dict(img_channels=3, width=8, enc_blk_nums=[1], dec_blk_nums=[1], dilations=[1])


class TestDegenerateMiddleSection:
    def test_zero_middle_decoder_blocks_is_refused(self):
        with pytest.raises(ValueError, match="middle_blk_num_dec must be >= 1"):
            create_darkir_model(middle_blk_num_enc=1, middle_blk_num_dec=0, **TINY)

    def test_zero_middle_encoder_blocks_is_still_legal(self):
        """The asymmetry is the point: only the DECODER count is degenerate."""
        model = create_darkir_model(middle_blk_num_enc=0, middle_blk_num_dec=1, **TINY)
        out = model(np.zeros((1, 32, 32, 3), dtype="float32"))
        assert np.all(np.isfinite(np.asarray(out)))

    def test_a_real_middle_residual_is_not_a_doubling(self):
        """Anti-vacuity: with dec >= 1 the Add sees two genuinely different tensors.

        The gates MUST be perturbed first. Every darkir block initializes its
        `beta`/`gamma` to exactly zero, so at initialization the block IS the
        identity and this assertion reads 0.0 for a reason that has nothing to
        do with F-63 -- the first draft of this test failed for exactly that
        reason and would have "proved" the degenerate case at every block count.
        """
        keras.utils.set_random_seed(0)
        model = create_darkir_model(middle_blk_num_enc=1, middle_blk_num_dec=1, **TINY)
        for w in model.get_layer("mid_dec_0").weights:
            if w.path.rsplit("/", 1)[-1] in ("beta", "gamma") and tuple(w.shape)[:3] == (1, 1, 1):
                w.assign(np.full(w.shape, 0.5, dtype="float32"))
        x = np.random.RandomState(0).randn(1, 32, 32, 3).astype("float32")
        light = np.asarray(keras.Model(model.inputs, model.get_layer("mid_enc_0").output)(x))
        res = np.asarray(keras.Model(model.inputs, model.get_layer("middle_residual").output)(x))
        assert float(np.max(np.abs(res - 2 * light))) > 1e-6


@pytest.mark.parametrize("cls", [DarkIREncoderBlock, DarkIRDecoderBlock], ids=["enc", "dec"])
class TestGatesArePerChannel:
    def test_beta_and_gamma_are_per_channel_not_scalar(self, cls):
        block = cls(channels=6, dilations=[1])
        block.build((None, 8, 8, 6))
        # By ATTRIBUTE, not by name: the block's own LayerNormalizations each
        # carry a `gamma` and a `beta` too, so a name sweep matches six weights.
        gates = [block.gamma, block.beta]
        for w in gates:
            assert tuple(w.shape) == (1, 1, 1, 6), (w.path, tuple(w.shape))
