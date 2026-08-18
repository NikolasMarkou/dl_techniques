"""Tests for the PRISM time-series blocks.

The top-level ``PRISMLayer`` composes ``PRISMTimeTree`` / ``PRISMNode`` /
``FrequencyBandRouter`` / ``FrequencyBandStatistics``, so its forward pass and
``.keras`` round-trip exercise those sub-layers. ``FrequencyBandStatistics`` is
also tested directly.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.prism_blocks import (
    PRISMLayer,
    PRISMNode,
    PRISMTimeTree,
    FrequencyBandStatistics,
)

# seq_len must be long enough that the multi-level wavelet decomposition does
# not drive the deepest band to length 1 or 0. There is no "band-length
# normalization" to collapse: the real mechanism is that a length-1 band has an
# EMPTY first-difference tensor (mean/var over an empty axis) and a length-0
# band has nothing to reduce at all. Both are handled by the guard in
# ``FrequencyBandStatistics.call``, exercised deliberately in
# ``TestFrequencyBandStatisticsDegenerateLengths`` below; the default here stays
# non-degenerate so the ordinary tests exercise the ordinary path.
B, SEQ, CH = 4, 32, 4


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, SEQ, CH)).astype("float32")


class TestFrequencyBandStatistics:
    def test_forward_and_shape(self, sample):
        layer = FrequencyBandStatistics()
        out = layer(sample)
        n_stats = layer._num_stats
        assert tuple(out.shape) == (B, CH, n_stats)
        assert layer.compute_output_shape((B, SEQ, CH)) == (B, CH, n_stats)

    def test_serialization(self, sample, tmp_path):
        inp = keras.Input(shape=(SEQ, CH))
        out = FrequencyBandStatistics(name="fbs")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "fbs.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"FrequencyBandStatistics": FrequencyBandStatistics}
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(loaded(sample)),
            rtol=1e-5, atol=1e-5,
        )


class TestPRISMLayer:
    def _make(self, **kw):
        defaults = dict(tree_depth=2, num_wavelet_levels=2, router_hidden_dim=16,
                        dropout_rate=0.0)
        defaults.update(kw)
        return PRISMLayer(**defaults)

    def test_forward_pass(self, sample):
        out = self._make()(sample)
        assert tuple(out.shape) == (B, SEQ, CH)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, SEQ, CH)) == (B, SEQ, CH)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(SEQ, CH))
        out = self._make(name="prism")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "prism.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"PRISMLayer": PRISMLayer}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        rebuilt = PRISMLayer.from_config(self._make(use_residual=False).get_config())
        assert rebuilt.use_residual is False


class TestPRISMTimeTreeSegmentArithmetic:
    """The build-time and runtime segment-length formulas must agree.

    ``PRISMTimeTree.build`` sizes every node from one formula while
    ``PRISMTimeTree._split_with_overlap`` feeds it segments sized by another;
    the two drifted apart (they differ in the SIGN of the overlap term).  This
    pins them together: whatever ``build`` promises a node, the forward pass
    must actually hand it.
    """

    @pytest.mark.parametrize("context_len", [96, 256])
    def test_build_and_runtime_segment_lengths_agree(self, context_len, monkeypatch):
        channels = 7
        depth = 4

        recorded = []
        original_build = PRISMNode.build

        def recording_build(self, input_shape):
            recorded.append(input_shape[1])
            return original_build(self, input_shape)

        monkeypatch.setattr(PRISMNode, "build", recording_build)

        tree = PRISMTimeTree(
            tree_depth=depth, num_wavelet_levels=1, router_hidden_dim=8,
            dropout_rate=0.0, overlap_ratio=0.25,
        )
        tree.build((None, context_len, channels))

        # One entry per node, in level order: 1 + 2 + 4 + 8 + 16 nodes.
        assert len(recorded) == sum(2 ** level for level in range(depth + 1))

        x = np.zeros((2, context_len, channels), dtype="float32")

        offset = 1  # skip level 0 (num_segments == 1, no split)
        mismatches = []
        for level in range(1, depth + 1):
            num_segments = 2 ** level
            build_len = recorded[offset]
            offset += num_segments

            runtime_len = int(
                tree._split_with_overlap(x, num_segments)[0].shape[1]
            )
            if build_len != runtime_len:
                mismatches.append(
                    f"ctx={context_len} level={level} "
                    f"({num_segments} segments): build={build_len} "
                    f"runtime={runtime_len}"
                )

        assert not mismatches, (
            "build-time and runtime segment lengths disagree:\n  "
            + "\n  ".join(mismatches)
        )


class TestFrequencyBandStatisticsDegenerateLengths:
    """A frequency band can collapse to length 1 or 0 under wavelet decimation.

    ``PRISMTimeTree`` splits its input into ``2 ** tree_depth`` segments and each
    ``PRISMNode`` runs ``num_wavelet_levels`` of floor-halving Haar decomposition
    over that segment, so the deepest band has length
    ``segment_len // 2 ** num_wavelet_levels`` -- which reaches 1, then 0, at
    configurations the model accepts today.  At length 1 the first-difference
    tensor is EMPTY and ``ops.mean``/``ops.var`` return NaN *silently*, which the
    router's single joint softmax then spreads across every band; at length 0
    ``ops.min``/``ops.max`` raise ``InvalidArgumentError`` instead.  Both must
    yield finite statistics, in eager AND in traced (``tf.function``) execution.
    """

    CHANNELS = 7
    BATCH = 4

    @staticmethod
    def _call_traced(layer, x):
        """Invoke ``layer`` inside a ``tf.function`` (the regime ``fit`` uses)."""
        import tensorflow as tf

        @tf.function
        def traced(t):
            return layer(t)

        return traced(tf.convert_to_tensor(x))

    @pytest.mark.parametrize("mode", ["eager", "graph"])
    def test_length_one_band_diff_features_are_zero(self, mode):
        layer = FrequencyBandStatistics()
        x = np.random.default_rng(0).standard_normal(
            (self.BATCH, 1, self.CHANNELS)
        ).astype("float32")

        out = layer(x) if mode == "eager" else self._call_traced(layer, x)
        out = keras.ops.convert_to_numpy(out)

        assert out.shape == (self.BATCH, self.CHANNELS, 6)
        assert np.isfinite(out).all(), (
            f"[{mode}] non-finite statistics on a length-1 band; "
            f"nan_frac={np.isnan(out).mean()}"
        )
        # Channels 4 and 5 are diff_mean / diff_std: the first difference of a
        # single sample is undefined, and 0.0 is the defined stand-in.
        np.testing.assert_array_equal(out[:, :, 4], np.zeros((self.BATCH, self.CHANNELS), "float32"))
        np.testing.assert_array_equal(out[:, :, 5], np.zeros((self.BATCH, self.CHANNELS), "float32"))

    @pytest.mark.parametrize("mode", ["eager", "graph"])
    def test_length_one_band_non_diff_features_are_unchanged(self, mode):
        layer = FrequencyBandStatistics()
        x = np.random.default_rng(1).standard_normal(
            (self.BATCH, 1, self.CHANNELS)
        ).astype("float32")

        out = layer(x) if mode == "eager" else self._call_traced(layer, x)
        out = keras.ops.convert_to_numpy(out)

        # mean == min == max == the single sample; std == sqrt(0 + epsilon).
        single = x[:, 0, :]
        np.testing.assert_allclose(out[:, :, 0], single, rtol=0, atol=1e-6)
        np.testing.assert_allclose(out[:, :, 2], single, rtol=0, atol=1e-6)
        np.testing.assert_allclose(out[:, :, 3], single, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            out[:, :, 1], np.full_like(single, np.sqrt(layer.epsilon)), rtol=1e-4, atol=1e-6
        )

    @pytest.mark.parametrize("mode", ["eager", "graph"])
    def test_empty_band_returns_zeros(self, mode):
        layer = FrequencyBandStatistics()
        x = np.zeros((self.BATCH, 0, self.CHANNELS), dtype="float32")

        out = layer(x) if mode == "eager" else self._call_traced(layer, x)
        out = keras.ops.convert_to_numpy(out)

        assert out.shape == (self.BATCH, self.CHANNELS, 6)
        assert np.isfinite(out).all(), f"[{mode}] non-finite statistics on a length-0 band"
        np.testing.assert_array_equal(
            out, np.zeros((self.BATCH, self.CHANNELS, 6), dtype="float32")
        )

    @pytest.mark.parametrize("mode", ["eager", "graph"])
    def test_length_two_band_is_untouched_by_the_guard(self, mode):
        """The guard must not reach the non-degenerate path every variant uses."""
        layer = FrequencyBandStatistics()
        x = np.random.default_rng(2).standard_normal(
            (self.BATCH, 2, self.CHANNELS)
        ).astype("float32")

        out = keras.ops.convert_to_numpy(
            layer(x) if mode == "eager" else self._call_traced(layer, x)
        )
        diff = x[:, 1:, :] - x[:, :-1, :]
        np.testing.assert_allclose(out[:, :, 4], diff.mean(axis=1), rtol=1e-6, atol=1e-6)
        assert np.isfinite(out).all()


class TestFrequencyBandStatisticsDynamicTimeAxis:
    """The degenerate-band guard must also hold when the time axis is UNKNOWN.

    ``FrequencyBandStatistics`` branches on the STATIC time length, so a trace
    with ``TensorSpec([None, None, C])`` -- the shape an ONNX/SavedModel export
    or a ragged-length ``tf.data`` pipeline produces -- used to fall through the
    guard entirely.  Under a graph the failure is silent rather than loud:
    ``ops.min``/``ops.max`` over a zero-length axis return ``+/-inf`` instead of
    raising, and ``ops.mean``/``ops.var`` return NaN, which the router's single
    joint softmax then spreads across every band.
    """

    CHANNELS = 7
    BATCH = 4

    @staticmethod
    def _call_dynamic(layer, x):
        """Trace ``layer`` with the time axis left unknown."""
        import tensorflow as tf

        fn = tf.function(
            lambda t: layer(t),
            input_signature=[tf.TensorSpec([None, None, x.shape[-1]], tf.float32)],
        )
        return fn(tf.convert_to_tensor(x))

    @pytest.mark.parametrize("length", [0, 1, 2, 8])
    def test_dynamic_time_axis_statistics_are_finite(self, length):
        layer = FrequencyBandStatistics()
        x = np.random.default_rng(length).standard_normal(
            (self.BATCH, length, self.CHANNELS)
        ).astype("float32")

        out = keras.ops.convert_to_numpy(self._call_dynamic(layer, x))

        assert out.shape == (self.BATCH, self.CHANNELS, 6)
        assert np.isfinite(out).all(), (
            f"non-finite statistics on a dynamic-axis band of length {length}; "
            f"non_finite_frac={float(np.mean(~np.isfinite(out)))}"
        )

    def test_dynamic_length_zero_matches_the_static_guard(self):
        """A length-0 band is all zeros -- exactly what the static branch returns."""
        layer = FrequencyBandStatistics()
        x = np.zeros((self.BATCH, 0, self.CHANNELS), dtype="float32")

        out = keras.ops.convert_to_numpy(self._call_dynamic(layer, x))
        np.testing.assert_array_equal(
            out, np.zeros((self.BATCH, self.CHANNELS, 6), dtype="float32")
        )

    def test_dynamic_length_one_matches_the_static_guard(self):
        """A length-1 band gets diff features of exactly 0.0 and real mean/min/max."""
        layer = FrequencyBandStatistics()
        x = np.random.default_rng(11).standard_normal(
            (self.BATCH, 1, self.CHANNELS)
        ).astype("float32")

        out = keras.ops.convert_to_numpy(self._call_dynamic(layer, x))
        static = keras.ops.convert_to_numpy(layer(x))

        np.testing.assert_array_equal(out, static)
        np.testing.assert_array_equal(out[:, :, 4], np.zeros((self.BATCH, self.CHANNELS), "float32"))
        np.testing.assert_array_equal(out[:, :, 5], np.zeros((self.BATCH, self.CHANNELS), "float32"))

    @pytest.mark.parametrize("length", [2, 8])
    def test_dynamic_non_degenerate_band_equals_the_static_result(self, length):
        """The dynamic path must not perturb an ordinary band: bit-identical."""
        layer = FrequencyBandStatistics()
        x = np.random.default_rng(100 + length).standard_normal(
            (self.BATCH, length, self.CHANNELS)
        ).astype("float32")

        dynamic = keras.ops.convert_to_numpy(self._call_dynamic(layer, x))
        static = keras.ops.convert_to_numpy(layer(x))
        np.testing.assert_array_equal(dynamic, static)

    def test_static_path_still_propagates_genuine_nan_inputs(self):
        """Sanitization is confined to the dynamic path -- it must not launder data.

        A real NaN in user data is a data defect the caller must see, not
        something the statistics layer silently rewrites to 0.0.
        """
        layer = FrequencyBandStatistics()
        x = np.random.default_rng(3).standard_normal(
            (self.BATCH, 8, self.CHANNELS)
        ).astype("float32")
        x[0, 0, 0] = np.nan

        out = keras.ops.convert_to_numpy(layer(x))
        assert np.isnan(out[0, 0, :]).any(), (
            "a genuine NaN input was laundered away on the STATIC path"
        )


class TestPRISMModelDynamicTimeAxis:
    """Model-level closure: a traced PRISM with an unknown time axis is finite."""

    def test_depth_three_model_traced_with_unknown_time_axis_is_finite(self):
        import tensorflow as tf
        from dl_techniques.models.time_series.prism.model import PRISMModel

        keras.utils.set_random_seed(1234)
        model = PRISMModel(
            context_len=96,
            forecast_len=24,
            num_features=7,
            tree_depth=3,
            hidden_dim=32,
            num_layers=1,
            router_hidden_dim=16,
        )
        x = np.random.default_rng(0).standard_normal((2, 96, 7)).astype("float32")
        model(x)  # build

        traced = tf.function(
            lambda t: model(t, training=False),
            input_signature=[tf.TensorSpec([None, None, 7], tf.float32)],
        )
        y = keras.ops.convert_to_numpy(traced(tf.convert_to_tensor(x)))

        nan_frac = float(np.mean(~np.isfinite(y)))
        assert nan_frac == 0.0, f"traced dynamic-axis PRISM produced nan_frac={nan_frac}"
