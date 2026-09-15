"""Tests for ``PatchPooling`` (``dl_techniques.layers.blt.patch_pooling``)."""

import keras
import numpy as np

from dl_techniques.layers.blt.patch_pooling import PatchPooling

SEQ, HID, NP_, GDIM = 10, 16, 8, 16
B = 2


def _patch_ids():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(1).integers(0, NP_, size=(B, SEQ)).astype("int32")
    )


# ---------------------------------------------------------------------
# PatchPooling (multi-arg call)
# ---------------------------------------------------------------------

class TestPatchPooling:

    def test_forward_pass(self):
        pool = PatchPooling(output_dim=GDIM, num_queries=2, max_patches=NP_)
        byte_hiddens = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((B, SEQ, HID)).astype("float32")
        )
        out = pool(byte_hiddens, _patch_ids())
        assert out.shape[0] == B and out.shape[-1] == GDIM

    def test_get_config_round_trip(self):
        pool = PatchPooling(output_dim=GDIM, num_queries=2, max_patches=NP_)
        rebuilt = PatchPooling.from_config(pool.get_config())
        assert rebuilt.output_dim == GDIM
