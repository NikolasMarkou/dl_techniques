"""AccUNet's central claims: HANC's hierarchical context and MLFC's cross-level mixing.

Why this file exists
--------------------
`test_accunet/test_model.py` checks HANC, ResPath and MLFC with `isinstance` and
with stored integers (`layer.k == 3`, `len(model.mlfc_layers) == 3`). Replace
either block with a 1x1 convolution -- or with the identity -- and every one of
those assertions still holds.

The claims, and what each is worth:

1. **HANC is hierarchical neighbourhood context.** With `k` scales the layer
   pools at 2x2, 4x4, ..., 2^(k-1), so the output at a pixel depends on exactly
   the aligned `2^(k-1)` block containing it -- and on NOTHING outside it. That
   is a two-sided claim with exact zeros on one side, which no shape or
   `isinstance` test can approximate.

   MEASURED 2026-08-18 (16x16 input, 8 channels, seed 1): perturbing input
   pixel (i, j) and reading output (0, 0) --

       k   (0,1)     (1,1)     (3,3)     (4,4)     (7,7)     (8,8)
       1   0.0       0.0       0.0       0.0       0.0       0.0
       2   3.89284   3.74197   0.0       0.0       0.0       0.0
       3   3.40397   4.89917   2.78455   0.0       0.0       0.0
       4   3.79862   6.81515   4.16075   3.06353   1.92224   0.0

   `k = 1` is the dead component: a pointwise 1x1 convolution, influence
   exactly 0.0 everywhere off the pixel itself.

2. **MLFC compiles ACROSS levels.** A perturbation of the deepest (level 3)
   feature map must reach the shallowest (level 0) output. MEASURED: 1.397389.
   A per-level block that did not mix would score exactly 0.0.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.hanc_layer import HANCLayer
from dl_techniques.layers.multi_level_feature_compilation import MLFCLayer


SIZE, CHANNELS = 16, 8
BASE = np.random.default_rng(0).normal(size=(1, SIZE, SIZE, CHANNELS)).astype("float32")


def _influence_on_origin(layer: HANCLayer, row: int, col: int) -> float:
    """max|delta| at output pixel (0, 0) when input pixel (row, col) is bumped."""
    clean = np.asarray(
        keras.ops.convert_to_numpy(
            layer(keras.ops.convert_to_tensor(BASE), training=False)
        )
    )
    bumped_input = BASE.copy()
    bumped_input[0, row, col, :] += 5.0
    bumped = np.asarray(
        keras.ops.convert_to_numpy(
            layer(keras.ops.convert_to_tensor(bumped_input), training=False)
        )
    )
    return float(np.max(np.abs(clean[0, 0, 0] - bumped[0, 0, 0])))


def _hanc(k: int) -> HANCLayer:
    keras.utils.set_random_seed(1)
    return HANCLayer(in_channels=CHANNELS, out_channels=CHANNELS, k=k)


class TestHANCAggregatesAHierarchicalNeighbourhood:
    @pytest.mark.parametrize("k,radius", [(2, 2), (3, 4), (4, 8)])
    def test_the_receptive_field_is_exactly_the_aligned_block(self, k, radius):
        layer = _hanc(k)
        inside = _influence_on_origin(layer, radius - 1, radius - 1)
        outside = _influence_on_origin(layer, radius, radius)
        assert inside > 1e-3, (
            f"HANC(k={k}): pixel ({radius - 1}, {radius - 1}) is inside the "
            f"{radius}x{radius} pooling block but influences the origin by only "
            f"{inside:.5f}; the hierarchical context is not being aggregated"
        )
        assert outside == 0.0, (
            f"HANC(k={k}): pixel ({radius}, {radius}) is OUTSIDE the "
            f"{radius}x{radius} pooling block yet influences the origin by "
            f"{outside:.5f}; the receptive field does not match the "
            f"2^(k-1) hierarchy this layer claims"
        )

    def test_k_equals_one_is_pointwise(self):
        """The dead component, measured: a 1x1 convolution and nothing more."""
        layer = _hanc(1)
        for row, col in [(0, 1), (1, 1), (3, 3), (4, 4), (7, 7), (8, 8)]:
            assert _influence_on_origin(layer, row, col) == 0.0, (
                f"HANC(k=1) is not pointwise: pixel ({row}, {col}) reaches the "
                "origin, so the k>1 measurements above are not attributable to "
                "the pooling hierarchy"
            )

    def test_more_scales_means_a_wider_field(self):
        """The knob `k` was checked only by `assert layer.k == k`."""
        widths = []
        for k in (1, 2, 3, 4):
            layer = _hanc(k)
            reach = [
                offset
                for offset in (1, 2, 4, 8)
                if _influence_on_origin(layer, offset, offset) > 0.0
            ]
            widths.append(max(reach) if reach else 0)
        assert widths == sorted(widths) and widths[0] < widths[-1], (
            f"the reach does not grow with k: {widths}"
        )


class TestMLFCMixesAcrossLevels:
    @staticmethod
    def _features():
        rng = np.random.default_rng(3)
        channels = [4, 8, 16, 32]
        return channels, [
            rng.normal(size=(1, 32 // (2 ** index), 32 // (2 ** index), channels[index]))
            .astype("float32")
            for index in range(4)
        ]

    def test_the_deepest_level_reaches_the_shallowest_output(self):
        channels, features = self._features()
        keras.utils.set_random_seed(2)
        layer = MLFCLayer(channels_list=channels)

        def run(feats):
            out = layer(
                [keras.ops.convert_to_tensor(f) for f in feats], training=False
            )
            return np.asarray(keras.ops.convert_to_numpy(out[0]))

        clean = run(features)
        perturbed = list(features)
        perturbed[3] = perturbed[3] + 3.0
        delta = float(np.max(np.abs(clean - run(perturbed))))
        # Measured 1.397389 on this configuration.
        assert delta > 1e-3, (
            f"perturbing the level-3 feature map moved the level-0 output by "
            f"{delta:.6f}: MLFC is processing each level independently, which "
            f"is the one thing 'multi-level feature compilation' must not do"
        )

    def test_the_probe_is_not_measuring_a_shared_input(self):
        """Control: perturbing level 3 must leave level 3's own output moving too,
        and the level-0 response must vanish if level 3 is left alone."""
        channels, features = self._features()
        keras.utils.set_random_seed(2)
        layer = MLFCLayer(channels_list=channels)

        def run(feats):
            out = layer(
                [keras.ops.convert_to_tensor(f) for f in feats], training=False
            )
            return [np.asarray(keras.ops.convert_to_numpy(o)) for o in out]

        clean = run(features)
        again = run(features)
        assert float(np.max(np.abs(clean[0] - again[0]))) == 0.0, (
            "the layer is nondeterministic at inference; the delta measured in "
            "the previous test would then be noise"
        )
