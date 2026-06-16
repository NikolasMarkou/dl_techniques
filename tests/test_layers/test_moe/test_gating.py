"""Direct unit tests for the MoE gating layers and FFNExpert.

These concrete layers are also exercised through the MixtureOfExperts composite
in ``test_layer.py``; this module adds direct construction / validation /
forward / serialization coverage (notably a CosineGating round-trip, which the
composite tests do not cover directly).

The gating layers return ``(expert_weights, expert_indices, aux_dict)`` — the
dict output makes a functional ``.keras`` model awkward, so serialization is
verified via a ``get_config`` -> ``from_config`` + weight-transfer round-trip.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.moe.gating import LinearGating, CosineGating, SoftMoEGating
from dl_techniques.layers.moe.experts import FFNExpert

B, D = 4, 16
NUM_EXPERTS = 4


def _f32(*shape):
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).standard_normal(shape).astype("float32")
    )


def _gating_weight_round_trip(layer, data):
    """Build, serialize via config + weight transfer, assert identical weights output."""
    w0, _, _ = layer(data)
    rebuilt = type(layer).from_config(layer.get_config())
    rebuilt(data)  # build the clone
    rebuilt.set_weights(layer.get_weights())
    w1, _, _ = rebuilt(data)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(w0), keras.ops.convert_to_numpy(w1),
        rtol=1e-5, atol=1e-5,
    )


class TestLinearGating:
    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            LinearGating(num_experts=NUM_EXPERTS, top_k=NUM_EXPERTS + 1)

    def test_forward_and_shape(self):
        layer = LinearGating(num_experts=NUM_EXPERTS, top_k=2, add_noise=False)
        weights, indices, aux = layer(_f32(B, D))
        assert tuple(weights.shape) == (B, NUM_EXPERTS)
        assert tuple(indices.shape) == (B, 2)
        w_shape, i_shape, _ = layer.compute_output_shape((B, D))
        assert w_shape == (B, NUM_EXPERTS) and i_shape == (B, 2)

    def test_serialization(self):
        _gating_weight_round_trip(
            LinearGating(num_experts=NUM_EXPERTS, top_k=2, add_noise=False), _f32(B, D)
        )


class TestCosineGating:
    def test_invalid_args(self):
        with pytest.raises(ValueError):
            CosineGating(num_experts=NUM_EXPERTS, embedding_dim=0)
        with pytest.raises(ValueError):
            CosineGating(num_experts=NUM_EXPERTS, temperature=0.0)

    def test_forward_and_shape(self):
        layer = CosineGating(num_experts=NUM_EXPERTS, embedding_dim=8, top_k=2)
        weights, indices, aux = layer(_f32(B, D))
        assert tuple(weights.shape) == (B, NUM_EXPERTS)
        assert tuple(indices.shape) == (B, 2)

    def test_serialization(self):
        _gating_weight_round_trip(
            CosineGating(num_experts=NUM_EXPERTS, embedding_dim=8, top_k=2,
                         learnable_temperature=True),
            _f32(B, D),
        )

    def test_get_config_round_trip(self):
        layer = CosineGating(num_experts=NUM_EXPERTS, embedding_dim=8, temperature=2.0)
        rebuilt = CosineGating.from_config(layer.get_config())
        assert rebuilt.embedding_dim == 8 and rebuilt.temperature == 2.0


class TestSoftMoEGating:
    def test_invalid_num_slots(self):
        with pytest.raises(ValueError):
            SoftMoEGating(num_experts=NUM_EXPERTS, num_slots=0)

    def test_forward(self):
        layer = SoftMoEGating(num_experts=NUM_EXPERTS, num_slots=2)
        out = layer(_f32(B, 5, D))  # SoftMoE needs a sequence dim
        assert out is not None

    def test_get_config_round_trip(self):
        layer = SoftMoEGating(num_experts=NUM_EXPERTS, num_slots=3)
        rebuilt = SoftMoEGating.from_config(layer.get_config())
        assert rebuilt.num_slots == 3


class TestFFNExpert:
    def _expert(self):
        return FFNExpert(ffn_config={"type": "mlp", "hidden_dim": 32, "output_dim": D})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError):
            FFNExpert(ffn_config={"hidden_dim": 32, "output_dim": D})

    def test_forward_and_shape(self):
        out = self._expert()(_f32(B, D))
        assert tuple(out.shape) == (B, D)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(D,))
        out = self._expert()(inp)
        model = keras.Model(inp, out)
        data = np.random.default_rng(0).standard_normal((B, D)).astype("float32")
        y0 = model(data)
        path = os.path.join(tmp_path, "expert.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1 = loaded(data)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
