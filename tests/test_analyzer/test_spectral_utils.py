"""`spectral_utils.py` had NO direct test module — only an incidental CONV2D path.

Why (F-066). `spectral_utils` is the layer-shape adapter every spectral metric depends
on: `infer_layer_type` decides whether a layer is analyzed at all, and
`get_weight_matrices` decides what matrix the metrics describe. Nothing tested either
directly. `test_spectral_metrics.py` exercises exactly one path
(`TestGlorotFactorMatchesKerasOwnFans`, CONV2D only), so the other eight `LayerType`
branches, the `NORM` skip, the RNN "first kernel only" choice, and the `get_weights()`
exception return were all unguarded.

Two findings from this file's own gaps are now pinned:

- `keras.layers.RMSNormalization` and `keras.layers.Normalization` fell through
  `infer_layer_type`'s substring list to `UNKNOWN` and were dropped from the spectral
  frame, instead of reaching the `NORM` branch that exists to skip them deliberately
  (D-010). Fixed at F-065; pinned below.
- LSTM/GRU contribute only `weights_list[0]`, the INPUT kernel `(input_dim, 4*units)`;
  the recurrent kernel is dropped. That is a deliberate simplification, but
  `_describe_model` then reports `num_params` from that one tensor, so a Keras LSTM's
  parameter count in the spectral frame is roughly a quarter of the layer's real count.
  Pinned as DOCUMENTED behaviour so a future change to it is deliberate.
"""

import keras
import numpy as np
import pytest

from dl_techniques.analyzer import spectral_metrics, spectral_utils
from dl_techniques.analyzer.constants import LayerType

# Input shapes each layer family needs to BUILD. A Conv/RNN layer built against a 2-D
# input raises `Kernel shape must have the same length as input`, so the family, not the
# layer, decides the shape.
_INPUT_SHAPE = {
    LayerType.DENSE: (None, 4),
    LayerType.EMBEDDING: (None,),
    LayerType.CONV1D: (None, 8, 4),
    LayerType.CONV2D: (None, 6, 6, 4),
    LayerType.CONV3D: (None, 4, 4, 4, 4),
    LayerType.NORM: (None, 4),
    LayerType.LSTM: (None, 3, 5),
    LayerType.GRU: (None, 3, 5),
}


class TestInferLayerType:
    """Every branch, both the isinstance path and the string fallback."""

    @pytest.mark.parametrize("layer_factory,expected", [
        (lambda: keras.layers.Dense(8), LayerType.DENSE),
        (lambda: keras.layers.Conv1D(4, 3), LayerType.CONV1D),
        (lambda: keras.layers.Conv2D(4, 3), LayerType.CONV2D),
        (lambda: keras.layers.Conv3D(4, 3), LayerType.CONV3D),
        (lambda: keras.layers.Embedding(10, 4), LayerType.EMBEDDING),
        (lambda: keras.layers.LSTM(4), LayerType.LSTM),
        (lambda: keras.layers.GRU(4), LayerType.GRU),
        (lambda: keras.layers.BatchNormalization(), LayerType.NORM),
        (lambda: keras.layers.LayerNormalization(), LayerType.NORM),
    ])
    def test_a_keras_layer_maps_to_its_type(self, layer_factory, expected):
        layer = layer_factory()
        layer.build(_INPUT_SHAPE[expected])
        assert spectral_utils.infer_layer_type(layer) is expected

    @pytest.mark.parametrize("layer_factory", [
        lambda: keras.layers.Normalization(),
        lambda: keras.layers.GroupNormalization(2),
        lambda: keras.layers.UnitNormalization(),
    ])
    def test_normalisation_variants_reach_the_norm_branch(self, layer_factory):
        """F-065. These were UNKNOWN, so they were dropped silently.

        The outcome a reader sees is identical to D-010's deliberate skip, which is
        exactly why the miss went unnoticed: nothing recorded that the layer was skipped
        for the WRONG reason. `keras.layers.RMSNormalization` does not exist in keras
        3.8, so `keras.layers.Normalization` and `UnitNormalization` stand in for it —
        what matters is that a normalisation layer whose name is not one of the three
        originally listed still reaches the NORM branch.
        """
        layer = layer_factory()
        layer.build((None, 4))
        assert spectral_utils.infer_layer_type(layer) is LayerType.NORM, (
            "a normalisation layer must be classified NORM so D-010 skips it "
            "deliberately, not UNKNOWN so it falls out of the frame unnoticed"
        )

    def test_an_activation_layer_is_unknown(self):
        layer = keras.layers.ReLU()
        layer.build((None, 4))
        assert spectral_utils.infer_layer_type(layer) is LayerType.UNKNOWN

    def test_an_unknown_subclass_is_unknown(self):
        class Mystery(keras.layers.Layer):
            def call(self, inputs):
                return inputs

        layer = Mystery()
        layer.build((None, 4))
        assert spectral_utils.infer_layer_type(layer) is LayerType.UNKNOWN


class TestGetLayerWeightsAndBias:
    def test_a_dense_layer_reports_kernel_and_bias(self):
        layer = keras.layers.Dense(6, use_bias=True)
        layer.build((None, 5))
        has_weights, weights, has_bias, bias = \
            spectral_utils.get_layer_weights_and_bias(layer)
        assert has_weights and has_bias
        assert weights.shape == (5, 6)
        assert bias.shape == (6,)

    def test_a_biasless_dense_layer_reports_no_bias(self):
        layer = keras.layers.Dense(6, use_bias=False)
        layer.build((None, 5))
        _, _, has_bias, bias = spectral_utils.get_layer_weights_and_bias(layer)
        assert not has_bias
        assert bias is None

    def test_a_normalisation_layer_is_skipped_with_no_weights(self):
        """D-010: its 1-D gamma/beta vectors have a degenerate ESD, so it is skipped."""
        layer = keras.layers.BatchNormalization()
        layer.build((None, 4))
        has_weights, weights, has_bias, bias = \
            spectral_utils.get_layer_weights_and_bias(layer)
        assert not has_weights
        assert weights is None

    def test_a_weightless_layer_reports_nothing(self):
        layer = keras.layers.ReLU()
        layer.build((None, 4))
        has_weights, weights, has_bias, bias = \
            spectral_utils.get_layer_weights_and_bias(layer)
        assert not has_weights and not has_bias
        assert weights is None and bias is None

    def test_an_rnn_reports_only_its_input_kernel(self):
        """Documented simplification: the recurrent kernel is dropped.

        Pinned so that changing it is a deliberate act. It also means `_describe_model`'s
        `num_params` under-reports an LSTM by roughly 4x, which is a property of the
        frame a reader should know about rather than discover.
        """
        layer = keras.layers.LSTM(4)
        layer.build((None, 3, 5))
        assert layer.built
        has_weights, weights, has_bias, bias = \
            spectral_utils.get_layer_weights_and_bias(layer)
        assert has_weights
        assert weights.shape == (5, 16), (
            f"expected the INPUT kernel (input_dim=5, 4*units=16), got {weights.shape}"
        )
        assert len(layer.get_weights()) == 3, (
            "the layer really does hold three tensors; only the first is analyzed"
        )

    def test_a_layer_whose_get_weights_raises_is_reported_as_weightless(self):
        """The `except Exception: return` path, which had no coverage at all."""

        class Hostile(keras.layers.Layer):
            def call(self, inputs):
                return inputs

            def get_weights(self):
                raise RuntimeError("not built in a reachable scope")

        has_weights, weights, has_bias, bias = \
            spectral_utils.get_layer_weights_and_bias(Hostile())
        assert (has_weights, weights, has_bias, bias) == (False, None, False, None)


class TestGetWeightMatrices:
    def test_a_dense_kernel_passes_through_with_rf_one(self):
        layer = keras.layers.Dense(6)
        layer.build((None, 5))
        has_weights, weights, _, _ = spectral_utils.get_layer_weights_and_bias(layer)
        matrices, n, m, rf = spectral_utils.get_weight_matrices(
            weights, LayerType.DENSE)
        assert len(matrices) == 1
        assert matrices[0].shape == (5, 6)
        assert (n, m, rf) == (6, 5, 1.0)
        assert n >= m, "N is documented as the LARGER dimension"

    def test_a_conv2d_kernel_is_folded_to_one_matrix(self):
        """F-057: the fold is WeightWatcher's convention and is kept deliberately.

        The test pins the SHAPE only. It does not claim the folded matrix's spectrum is
        the convolution operator's — `spectral_utils`'s docstring and README.md both
        corrected that claim (F-057), and the correction is that it is NOT.
        """
        layer = keras.layers.Conv2D(8, (3, 3))
        layer.build((None, 6, 6, 4))
        _, weights, _, _ = spectral_utils.get_layer_weights_and_bias(layer)
        matrices, n, m, rf = spectral_utils.get_weight_matrices(
            weights, LayerType.CONV2D)
        assert len(matrices) == 1
        assert matrices[0].shape == (3 * 3 * 4, 8)   # (kh*kw*in_c, out_c) = (36, 8)
        assert rf == 9.0
        assert (n, m) == (36, 8)
        assert n >= m, "N is documented as the LARGER dimension"

    def test_an_embedding_matrix_is_returned_as_is(self):
        layer = keras.layers.Embedding(10, 4)
        layer.build((None,))
        _, weights, _, _ = spectral_utils.get_layer_weights_and_bias(layer)
        matrices, n, m, rf = spectral_utils.get_weight_matrices(
            weights, LayerType.EMBEDDING)
        assert matrices[0].shape == (10, 4)
        assert rf == 1.0

    def test_a_norm_layer_yields_no_matrices(self):
        _, weights, _, _ = spectral_utils.get_layer_weights_and_bias(
            keras.layers.BatchNormalization())
        matrices, n, m, rf = spectral_utils.get_weight_matrices(
            np.zeros((4,)), LayerType.NORM)
        assert matrices == []

    def test_an_unknown_layer_type_yields_no_matrices(self):
        matrices, n, m, rf = spectral_utils.get_weight_matrices(
            np.zeros((4, 4)), LayerType.UNKNOWN)
        assert matrices == []

    def test_no_input_means_no_matrices_rather_than_an_exception(self):
        matrices, _, _, _ = spectral_utils.get_weight_matrices(
            None, LayerType.DENSE)
        assert matrices == []


class TestGlorotNormalizationUsesKerasOwnFans:
    """The D-035 anchor: `matrix_shape` must arrive ORDERED, never as `(N, M)`.

    `N = max(shape)` and `M = min(shape)` destroy which axis is `fan_in`, so for the
    canonical first conv `(3,3,3,64)` the folded matrix is `(27, 64)` and `N` is the
    OUTPUT axis. Spelling it `N + M*rf` computed `64 + 27*9 = 307` against the true
    `27 + 64*9 = 603`.
    """

    def test_a_conv_kernel_gets_keras_own_fans(self):
        kappa = spectral_metrics.calculate_glorot_normalization_factor(
            (27, 64), rf=9.0)
        expected = np.sqrt(2.0 / (27 + 64 * 9))
        assert kappa == pytest.approx(expected, rel=1e-12)

    def test_the_sorted_spelling_would_have_been_wrong(self):
        """The defect D-035 fixed, asserted rather than merely described.

        `(27, 64)` sorted to `(N, M) = (64, 27)` and the old `N + M*rf` spelling gave
        `64 + 27*9 = 307` against the true `27 + 64*9 = 603`. Because kappa is
        proportional to `1/sqrt(fan_in + fan_out)`, the resulting error in kappa is
        `sqrt(603/307) = 1.4015` — the figure D-035 measured.
        """
        correct = spectral_metrics.calculate_glorot_normalization_factor(
            (27, 64), rf=9.0)
        wrong = np.sqrt(2.0 / (64 + 27 * 9))
        assert wrong / correct == pytest.approx(np.sqrt(603 / 307), rel=1e-12)
        assert wrong > correct

    def test_a_dense_layer_reduces_to_the_familiar_form(self):
        kappa = spectral_metrics.calculate_glorot_normalization_factor(
            (784, 256), rf=1.0)
        assert kappa == pytest.approx(np.sqrt(2.0 / (784 + 256)), rel=1e-12)

    def test_a_degenerate_shape_returns_one(self):
        assert spectral_metrics.calculate_glorot_normalization_factor(
            (0, 0), rf=1.0) == 1.0
