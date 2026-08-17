"""R-4: `Mamba2Layer`'s documented escape hatch was unreachable.

Commit ``f5a0939f5`` flipped ``Mamba2Layer.__init__``'s ``norm_before_gate``
from ``True`` to ``False`` and wrote the remedy into the docstring: "a
checkpoint trained under the old default must set ``norm_before_gate=True``
explicitly." But ``Mamba2ResidualBlock`` accepted no such parameter, did not
forward one, and carried none in ``get_config()`` -- and ``Mamba2`` builds
its entire stack out of those blocks. There was no way to reach the layer's
constructor from a real model, so the remedy could not be applied to the thing
it was written for.

The knob-changes-output template is `test_xlstm/test_multivariate_denorm.py`:
prove the flag MOVES the output, not merely that the attribute is stored.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest

from dl_techniques.models.mamba.components_v2 import (
    Mamba2Layer,
    Mamba2ResidualBlock,
)
from dl_techniques.models.mamba.mamba_v2 import Mamba2

D_MODEL = 32
SEQ = 8
BATCH = 2


def _x():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).random((BATCH, SEQ, D_MODEL)).astype("float32")
    )


def _block(norm_before_gate):
    keras.utils.set_random_seed(7)
    block = Mamba2ResidualBlock(
        d_model=D_MODEL, d_state=8, d_conv=4, expand=2, headdim=16,
        d_ssm=2 * D_MODEL, norm_before_gate=norm_before_gate,
    )
    block.build((None, SEQ, D_MODEL))
    return block


class TestNormBeforeGateReachesTheLayer:

    def test_block_accepts_and_forwards_it(self):
        block = _block(True)
        assert block.norm_before_gate is True
        assert block.mamba2.norm_before_gate is True

    def test_block_default_matches_the_layer_default(self):
        """The block must not quietly impose a different default from the layer
        it wraps -- that would be a second, invisible flip."""
        assert (
            Mamba2ResidualBlock(
                d_model=D_MODEL, d_state=8, d_conv=4, expand=2, headdim=16,
                d_ssm=2 * D_MODEL,
            ).norm_before_gate
            is Mamba2Layer(d_model=D_MODEL, d_state=8, headdim=16).norm_before_gate
        )

    def test_it_is_serialized_by_the_block(self):
        assert _block(True).get_config()["norm_before_gate"] is True

    def test_flipping_it_actually_changes_the_block_output(self):
        """The knob must MOVE something. Both branches share the same `norm`
        weights, so a stored-attribute assertion would pass against a block that
        forwards nothing."""
        x = _x()
        out_false = keras.ops.convert_to_numpy(_block(False)(x)[0])
        out_true = keras.ops.convert_to_numpy(_block(True)(x)[0])

        assert out_false.shape == out_true.shape
        delta = float(np.max(np.abs(out_false - out_true)))
        assert delta > 1e-4, (
            f"norm_before_gate changed nothing: max|delta| = {delta:.6e}. "
            "The two seeded blocks are otherwise bit-identical, so any "
            "non-forwarding of the flag shows up here as exactly 0.0."
        )

    def test_two_identically_configured_blocks_are_bit_identical(self):
        """Control for the assertion above: the seeding really does make the
        comparison a one-variable one."""
        x = _x()
        a = keras.ops.convert_to_numpy(_block(True)(x)[0])
        b = keras.ops.convert_to_numpy(_block(True)(x)[0])
        np.testing.assert_array_equal(a, b)


class TestNormBeforeGateReachesTheWholeModel:
    """The remedy is written for a trained checkpoint, so it has to be settable
    where checkpoints live: on the model."""

    @staticmethod
    def _model(norm_before_gate):
        keras.utils.set_random_seed(11)
        return Mamba2(
            vocab_size=16, d_model=D_MODEL, num_layers=2, d_state=8,
            headdim=16, norm_before_gate=norm_before_gate,
        )

    def test_model_forwards_it_to_every_block(self):
        model = self._model(True)
        assert [b.mamba2.norm_before_gate for b in model.encoder_layers] == [True, True]

    def test_model_serializes_it(self):
        assert self._model(True).get_config()["norm_before_gate"] is True

    def test_it_changes_the_model_output(self):
        ids = keras.ops.convert_to_tensor(
            np.random.default_rng(3).integers(0, 16, (BATCH, SEQ)).astype("int32")
        )
        a = self._model(False)(ids)
        b = self._model(True)(ids)
        key = "last_hidden_state" if "last_hidden_state" in a else list(a)[0]
        delta = float(np.max(np.abs(
            keras.ops.convert_to_numpy(a[key]) - keras.ops.convert_to_numpy(b[key])
        )))
        assert delta > 1e-4, f"norm_before_gate is inert at model level: {delta:.6e}"
