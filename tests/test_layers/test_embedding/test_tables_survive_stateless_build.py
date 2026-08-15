"""Frequency/selector tables must hold their intended values when the layer is
built through a PARENT layer's ``call()``.

Why the parent: Keras 3 runs a symbolic build pass inside a ``StatelessScope``
whenever a sublayer is first reached from a parent's ``call()`` -- which is what
happens in every real model -- and that scope RECORDS a ``.assign()`` and then
DISCARDS it. Four embedding layers used ``add_weight(initializer='zeros')``
followed by ``.assign()`` inside ``build()``, so their tables were all zeros in
every real model while a direct ``layer.build(...)`` in a unit test showed the
correct values. **A test that calls ``.build(...)`` directly is exactly the test
that missed this defect for as long as it existed** -- every assertion below
therefore goes through ``_Parent.call()``.

Anti-vacuity: each assertion pins a VALUE against its closed form, not a shape
and not mere non-zero-ness. ``cos[0] == 1`` and ``inv_freq[0] == 1`` are the
discriminating entries (an all-zero table reads 0 there); the ``sin`` tables are
legitimately 0 at position 0, so they are pinned at a LATER position instead.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.embedding.continuous_sin_cos_embedding import (
    ContinuousSinCosEmbed,
)
from dl_techniques.layers.embedding.continuous_rope_embedding import ContinuousRoPE
from dl_techniques.layers.embedding.dual_rotary_position_embedding import (
    DualRotaryPositionEmbedding,
)
from dl_techniques.layers.embedding.multi_axis_rope import Ideogram4MRoPE


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call()`` is the only path that builds the child."""

    def __init__(self, child: keras.layers.Layer, **kwargs) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs):
        return self.child(inputs)


def _build_through_parent(child, input_shape, dtype="float32"):
    """Build ``child`` ONLY by reaching it from a parent layer's ``call()``."""
    parent = _Parent(child)
    parent(keras.Input(shape=input_shape[1:], dtype=dtype))
    return child


def _np(variable) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(variable))


class TestTablesSurviveTheStatelessBuildPass:
    """One assertion per fixed site, each with its own isolating mutation."""

    def test_continuous_sincos_omega_is_the_closed_form_frequency_ladder(self):
        layer = ContinuousSinCosEmbed(dim=32, ndim=2)
        _build_through_parent(layer, (2, 5, 2))

        eff = layer.effective_dim_per_wave
        expected = 1.0 / (
            layer.max_wavelength ** (np.arange(0, eff, 2, dtype=np.float32) / eff)
        )
        omega = _np(layer.omega)

        assert omega[0] == pytest.approx(1.0), (
            "omega[0] must be 1.0 (theta ** 0); reading 0.0 means the build-time "
            ".assign() was discarded by the StatelessScope"
        )
        np.testing.assert_allclose(omega, expected, rtol=0, atol=1e-7)

    def test_continuous_rope_omega_is_the_closed_form_frequency_ladder(self):
        layer = ContinuousRoPE(dim=32, ndim=2)
        _build_through_parent(layer, (2, 5, 2))

        eff = layer.effective_dim_per_wave
        expected = 1.0 / (
            layer.max_wavelength ** (np.arange(0, eff, 2, dtype=np.float32) / eff)
        )
        omega = _np(layer.omega)

        assert omega[0] == pytest.approx(1.0)
        np.testing.assert_allclose(omega, expected, rtol=0, atol=1e-7)

    def test_dual_rope_global_cos_is_the_closed_form_cosine_table(self):
        layer = DualRotaryPositionEmbedding(
            head_dim=8, max_seq_len=16, global_theta_base=10000.0
        )
        _build_through_parent(layer, (2, 4, 6, 8), dtype="float32")

        cos = _np(layer.cos_global_cached)
        freq_dim = layer.head_dim // 2
        inv_freq = 1.0 / (
            layer.global_theta_base
            ** (np.arange(0, freq_dim, dtype=np.float32) * 2.0 / layer.head_dim)
        )
        expected = np.cos(
            np.concatenate(
                [np.outer(np.arange(16, dtype=np.float32), inv_freq)] * 2, axis=1
            )
        )

        assert cos[0, 0] == pytest.approx(1.0), "cos(0) must be 1.0, not 0.0"
        np.testing.assert_allclose(cos, expected, rtol=0, atol=1e-6)

    def test_dual_rope_global_sin_is_the_closed_form_sine_table(self):
        layer = DualRotaryPositionEmbedding(
            head_dim=8, max_seq_len=16, global_theta_base=10000.0
        )
        _build_through_parent(layer, (2, 4, 6, 8), dtype="float32")

        sin = _np(layer.sin_global_cached)
        # sin(0) is legitimately 0, so pin position 1 of the fastest frequency
        # (inv_freq[0] == 1.0 => sin(1 * 1.0) == sin(1)).
        assert sin[1, 0] == pytest.approx(np.sin(1.0), abs=1e-6), (
            "sin[1, 0] must be sin(1); an all-zero table also reads 0 at "
            "position 0, which is why position 1 is the discriminating entry"
        )
        assert sin[0, 0] == pytest.approx(0.0, abs=1e-7)

    def test_dual_rope_local_tables_use_the_local_theta_base(self):
        layer = DualRotaryPositionEmbedding(
            head_dim=8,
            max_seq_len=16,
            global_theta_base=10000.0,
            local_theta_base=100.0,
        )
        _build_through_parent(layer, (2, 4, 6, 8), dtype="float32")

        cos_local = _np(layer.cos_local_cached)
        sin_local = _np(layer.sin_local_cached)
        freq_dim = layer.head_dim // 2
        inv_freq = 1.0 / (
            layer.local_theta_base
            ** (np.arange(0, freq_dim, dtype=np.float32) * 2.0 / layer.head_dim)
        )
        freqs = np.concatenate(
            [np.outer(np.arange(16, dtype=np.float32), inv_freq)] * 2, axis=1
        )

        assert cos_local[0, 0] == pytest.approx(1.0)
        np.testing.assert_allclose(cos_local, np.cos(freqs), rtol=0, atol=1e-6)
        np.testing.assert_allclose(sin_local, np.sin(freqs), rtol=0, atol=1e-6)

    def test_mrope_inv_freq_is_the_closed_form_inverse_frequency(self):
        layer = Ideogram4MRoPE(
            head_dim=16, rope_theta=10000.0, mrope_section=(2, 2, 2)
        )
        _build_through_parent(layer, (2, 5, 3), dtype="int32")

        expected = 1.0 / (
            layer.rope_theta
            ** (
                np.arange(0, layer.head_dim, 2, dtype=np.float32) / layer.head_dim
            )
        )
        inv_freq = _np(layer.inv_freq)

        assert inv_freq[0] == pytest.approx(1.0), "inv_freq[0] must be theta ** 0 == 1"
        np.testing.assert_allclose(inv_freq, expected, rtol=0, atol=1e-7)

    def test_mrope_select_onehot_is_a_one_hot_row_per_frequency_slot(self):
        layer = Ideogram4MRoPE(
            head_dim=16, rope_theta=10000.0, mrope_section=(2, 2, 2)
        )
        _build_through_parent(layer, (2, 5, 3), dtype="int32")

        expected = np.eye(3, dtype="float32")[layer._source_axis]
        onehot = _np(layer._select_onehot)

        # Every row must sum to exactly 1: an all-zero selector sums to 0 and
        # silently zeroes the selected frequency for EVERY slot.
        np.testing.assert_allclose(onehot.sum(axis=1), np.ones(layer._half), atol=0)
        np.testing.assert_array_equal(onehot, expected)

    def test_mrope_output_actually_depends_on_the_position_ids(self):
        """End-to-end consequence: with a zeroed table mRoPE is the identity."""
        layer = Ideogram4MRoPE(
            head_dim=16, rope_theta=10000.0, mrope_section=(2, 2, 2)
        )
        _build_through_parent(layer, (2, 5, 3), dtype="int32")

        pos = np.tile(np.arange(5, dtype="int32")[None, :, None], (1, 1, 3))
        cos, sin = layer(keras.ops.convert_to_tensor(pos))
        cos = _np(cos)

        assert cos[0, 0, 0] == pytest.approx(1.0)
        # Position 4 at the fastest frequency must rotate: cos(4) != cos(0).
        assert abs(cos[0, 4, 0] - 1.0) > 1e-3, (
            "cos is constant across positions -- the frequency table is dead"
        )
