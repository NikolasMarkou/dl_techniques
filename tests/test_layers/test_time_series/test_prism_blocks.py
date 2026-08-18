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

# seq_len must be long enough for the multi-level wavelet decomposition not to
# degenerate (very short bands collapse the band-length normalization).
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
