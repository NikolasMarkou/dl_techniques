"""The PFT layer partitions and merges its windows with ``utils.tensors``.

``blocks.py`` once carried private ``window_partition`` / ``window_reverse``
copies (the first flattened to ``(B * nW, ws * ws, C)``). They are gone: the layer
calls the shared helpers and flattens the partition itself. Two things must hold
and are pinned here on the running layer, not on the helpers in isolation:

* the names in ``blocks`` ARE the ``utils.tensors`` functions, so a private copy
  cannot come back unseen;
* what the layer hands ``_attend`` is the independent numpy oracle's partition
  (``pft_reference_numpy.window_partition``, official :172) of the map it
  windows, flattened row-major per window, and what it merges is the oracle's
  ``window_reverse`` (:186). The map is a non-square ``8 x 12`` at
  ``window_size=4`` (6 windows per image, batch 2) at float64, so a transposed
  window order, an H/W swap or a square-only shortcut cannot pass. Plain and
  shifted layers both run (the cyclic roll surrounds the windowing).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.super_resolution.pft_sr import blocks
from dl_techniques.utils import tensors as shared

from . import pft_reference_numpy as ref
from ..precision_arm_oracle import float64_scope

DIM, HEADS, WS, H, W = 16, 2, 4, 8, 12


def test_the_blocks_module_uses_the_shared_window_helpers():
    assert blocks.window_partition is shared.window_partition
    assert blocks.window_reverse is shared.window_reverse


@pytest.mark.parametrize("shift_size", [0, WS // 2])
def test_the_layer_windows_equal_the_numpy_oracle_windows(shift_size, monkeypatch):
    seen = {}

    real_partition, real_reverse = blocks.window_partition, blocks.window_reverse

    def partition(x, window_size):
        seen["tokens"] = np.asarray(x)
        return real_partition(x, window_size)

    def reverse(windows, window_size, height, width):
        out = real_reverse(windows, window_size, height, width)
        seen["merged_in"], seen["merged_out"] = np.asarray(windows), np.asarray(out)
        return out

    with float64_scope():
        keras.utils.set_random_seed(0)
        layer = blocks.PFTLayer(dim=DIM, num_heads=HEADS, window_size=WS,
                                shift_size=shift_size, convffn_kernel_size=5,
                                mlp_ratio=2.0, keep_topk=None)
        layer.build((None, H, W, DIM))
        real_attend = layer._attend

        def attend(windows, *args, **kwargs):
            seen["attended"] = np.asarray(windows)
            return real_attend(windows, *args, **kwargs)

        monkeypatch.setattr(blocks, "window_partition", partition)
        monkeypatch.setattr(blocks, "window_reverse", reverse)
        monkeypatch.setattr(layer, "_attend", attend)
        x = np.random.default_rng(1).standard_normal((2, H, W, DIM))
        layer(x)

    tokens, attended = seen["tokens"], seen["attended"]
    assert tokens.shape == (2, H, W, 4 * DIM) and tokens.dtype == np.float64
    expect = ref.window_partition(tokens, WS)
    assert expect.shape[0] == 2 * (H // WS) * (W // WS)
    assert attended.shape == (expect.shape[0], WS * WS, 4 * DIM)
    np.testing.assert_array_equal(attended, expect.reshape(-1, WS * WS, 4 * DIM))
    # The merge: the oracle's reverse of the attended-layout windows.
    merged_in = seen["merged_in"]
    np.testing.assert_array_equal(
        seen["merged_out"],
        ref.window_reverse(merged_in.reshape(-1, WS, WS, merged_in.shape[-1]), WS, H, W))
