"""
Test suite for VectorQuantizerRotationTrick.

Covers:
- Initialization + ctor validation (negative paths)
- Build (single-head + multi-head + dim mismatch)
- Forward shape on 1D + 2D inputs
- Per-gradient-mode forward (rotation / reflection / no_grad_scale / ste)
- STE back-compat vs existing VectorQuantizer (atol<=1e-6)
- Cosine vs euclidean lookup
- EMA update + no-update-on-inference
- Dead-code expiration changes codebook entries
- K-means init runs once + changes codebook
- Diversity + orthogonal aux losses present when coefficient > 0
- Gradient flow under tf.GradientTape
- .keras save/load round-trip for every gradient_mode
- compute_output_shape == input shape
"""

import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.vector_quantizer import VectorQuantizer
from dl_techniques.layers.vector_quantizer_rotation_trick import (
    VectorQuantizerRotationTrick,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_config() -> Dict[str, Any]:
    return {
        "num_embeddings": 16,
        "embedding_dim": 32,
        "commitment_cost": 0.25,
        "gradient_mode": "rotation",
        "distance_mode": "euclidean",
        "use_ema": False,
        "ema_decay": 0.99,
        "epsilon": 1e-5,
        "num_heads": 1,
    }


@pytest.fixture
def sample_input_2d() -> keras.KerasTensor:
    keras.utils.set_random_seed(42)
    return ops.cast(keras.random.normal((4, 8, 8, 32)), "float32")


@pytest.fixture
def sample_input_1d() -> keras.KerasTensor:
    keras.utils.set_random_seed(42)
    return ops.cast(keras.random.normal((4, 16, 32)), "float32")


# ============================================================================
# Init + validation
# ============================================================================

class TestInitialization:
    def test_default_init(self, base_config):
        layer = VectorQuantizerRotationTrick(**base_config)
        assert layer.num_embeddings == 16
        assert layer.embedding_dim == 32
        assert layer.head_dim == 32
        assert layer.gradient_mode == "rotation"
        assert layer.distance_mode == "euclidean"
        assert not layer.built

    def test_invalid_num_embeddings(self):
        with pytest.raises(ValueError, match="num_embeddings must be positive"):
            VectorQuantizerRotationTrick(num_embeddings=0, embedding_dim=32)

    def test_invalid_embedding_dim(self):
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            VectorQuantizerRotationTrick(num_embeddings=16, embedding_dim=-1)

    def test_invalid_commitment_cost(self):
        with pytest.raises(ValueError, match="commitment_cost must be non-negative"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=32, commitment_cost=-0.1
            )

    def test_invalid_gradient_mode(self):
        with pytest.raises(ValueError, match="gradient_mode must be"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=32, gradient_mode="nope"
            )

    def test_invalid_distance_mode(self):
        with pytest.raises(ValueError, match="distance_mode must be"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=32, distance_mode="manhattan"
            )

    def test_multi_head_dim_mismatch(self):
        with pytest.raises(ValueError, match="must be divisible by"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=30, num_heads=4
            )

    def test_invalid_ema_decay(self):
        with pytest.raises(ValueError, match="ema_decay must be in"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=32, use_ema=True, ema_decay=0.0
            )

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=32, num_heads=0
            )

    def test_invalid_dead_code_threshold(self):
        with pytest.raises(ValueError, match="dead_code_threshold must be non-negative"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=32, dead_code_threshold=-1
            )

    def test_invalid_diversity(self):
        with pytest.raises(ValueError, match="diversity_coefficient must be non-negative"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=32, diversity_coefficient=-0.1
            )

    def test_invalid_orthogonal(self):
        with pytest.raises(ValueError, match="orthogonal_reg_coefficient must be non-negative"):
            VectorQuantizerRotationTrick(
                num_embeddings=16, embedding_dim=32, orthogonal_reg_coefficient=-1.0
            )


# ============================================================================
# Build
# ============================================================================

class TestBuild:
    def test_build_single_head(self, base_config, sample_input_2d):
        layer = VectorQuantizerRotationTrick(**base_config)
        layer.build(sample_input_2d.shape)
        assert layer.built
        assert layer.embeddings.shape == (1, 16, 32)
        assert layer.embeddings.trainable is True

    def test_build_multi_head(self, base_config, sample_input_2d):
        base_config["num_heads"] = 4
        layer = VectorQuantizerRotationTrick(**base_config)
        layer.build(sample_input_2d.shape)
        assert layer.embeddings.shape == (4, 16, 8)

    def test_build_ema(self, base_config, sample_input_2d):
        base_config["use_ema"] = True
        layer = VectorQuantizerRotationTrick(**base_config)
        layer.build(sample_input_2d.shape)
        assert layer.embeddings.trainable is False
        assert layer.ema_cluster_size is not None
        assert layer.ema_embeddings is not None
        assert layer.ema_cluster_size.shape == (1, 16)

    def test_build_dim_mismatch(self, base_config):
        layer = VectorQuantizerRotationTrick(**base_config)
        with pytest.raises(ValueError, match="must match embedding_dim"):
            layer.build((4, 8, 8, 64))

    def test_build_dead_code_buffer(self, base_config, sample_input_2d):
        base_config["dead_code_threshold"] = 3
        layer = VectorQuantizerRotationTrick(**base_config)
        layer.build(sample_input_2d.shape)
        assert layer.dead_code_unused is not None
        assert layer.dead_code_unused.shape == (1, 16)


# ============================================================================
# Forward + shape
# ============================================================================

class TestForward:
    def test_forward_2d(self, base_config, sample_input_2d):
        layer = VectorQuantizerRotationTrick(**base_config)
        out = layer(sample_input_2d)
        assert out.shape == sample_input_2d.shape
        assert len(layer.losses) == 2  # codebook + commitment

    def test_forward_1d(self, base_config, sample_input_1d):
        layer = VectorQuantizerRotationTrick(**base_config)
        out = layer(sample_input_1d)
        assert out.shape == sample_input_1d.shape

    @pytest.mark.parametrize(
        "mode", ["rotation", "reflection", "no_grad_scale", "ste"]
    )
    def test_forward_each_grad_mode(self, base_config, sample_input_2d, mode):
        base_config["gradient_mode"] = mode
        layer = VectorQuantizerRotationTrick(**base_config)
        out = layer(sample_input_2d, training=True)
        out_np = ops.convert_to_numpy(out)
        assert out.shape == sample_input_2d.shape
        assert np.all(np.isfinite(out_np))

    def test_forward_multi_head(self, base_config, sample_input_2d):
        base_config["num_heads"] = 4
        layer = VectorQuantizerRotationTrick(**base_config)
        out = layer(sample_input_2d, training=True)
        assert out.shape == sample_input_2d.shape

    def test_compute_output_shape(self, base_config):
        layer = VectorQuantizerRotationTrick(**base_config)
        shape = (None, 8, 8, 32)
        assert layer.compute_output_shape(shape) == shape


# ============================================================================
# STE back-compat
# ============================================================================

class TestSTEBackCompat:
    def test_ste_matches_existing_vq(self, sample_input_2d):
        """gradient_mode='ste' must be bit-equivalent to VectorQuantizer."""
        init_weights = np.random.RandomState(0).uniform(
            -1, 1, (16, 32)
        ).astype("float32")

        vq = VectorQuantizer(num_embeddings=16, embedding_dim=32)
        vq(sample_input_2d)
        vq.embeddings.assign(init_weights)
        out_vq = vq(sample_input_2d)

        vqr = VectorQuantizerRotationTrick(
            num_embeddings=16,
            embedding_dim=32,
            gradient_mode="ste",
            num_heads=1,
            distance_mode="euclidean",
        )
        vqr(sample_input_2d)
        vqr.embeddings.assign(init_weights[None, ...])
        out_vqr = vqr(sample_input_2d)

        np.testing.assert_allclose(
            ops.convert_to_numpy(out_vq),
            ops.convert_to_numpy(out_vqr),
            atol=1e-6, rtol=1e-6,
            err_msg="STE mode must match VectorQuantizer bit-equivalently",
        )


# ============================================================================
# Cosine vs euclidean
# ============================================================================

class TestDistanceModes:
    def test_cosine_produces_valid_indices(self, base_config, sample_input_2d):
        base_config["distance_mode"] = "cosine"
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d)
        idx = layer.get_codebook_indices(sample_input_2d)
        idx_np = ops.convert_to_numpy(idx)
        assert np.all(idx_np >= 0)
        assert np.all(idx_np < base_config["num_embeddings"])

    def test_cosine_output_shape(self, base_config, sample_input_2d):
        base_config["distance_mode"] = "cosine"
        layer = VectorQuantizerRotationTrick(**base_config)
        out = layer(sample_input_2d)
        assert out.shape == sample_input_2d.shape


# ============================================================================
# EMA
# ============================================================================

class TestEMA:
    def test_ema_updates_in_training(self, base_config, sample_input_2d):
        base_config["use_ema"] = True
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d, training=True)
        c0 = ops.convert_to_numpy(layer.ema_cluster_size).copy()

        new_x = ops.cast(keras.random.normal((4, 8, 8, 32)), "float32")
        _ = layer(new_x, training=True)
        c1 = ops.convert_to_numpy(layer.ema_cluster_size)
        assert not np.allclose(c0, c1)

    def test_ema_no_updates_in_inference(self, base_config, sample_input_2d):
        base_config["use_ema"] = True
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d, training=True)
        c0 = ops.convert_to_numpy(layer.ema_cluster_size).copy()
        e0 = ops.convert_to_numpy(layer.ema_embeddings).copy()

        new_x = ops.cast(keras.random.normal((4, 8, 8, 32)), "float32")
        _ = layer(new_x, training=False)
        c1 = ops.convert_to_numpy(layer.ema_cluster_size)
        e1 = ops.convert_to_numpy(layer.ema_embeddings)
        np.testing.assert_allclose(c0, c1, atol=1e-7)
        np.testing.assert_allclose(e0, e1, atol=1e-7)


# ============================================================================
# Dead-code expiration
# ============================================================================

class TestDeadCode:
    def test_dead_code_changes_codebook(self, base_config, sample_input_2d):
        base_config["num_embeddings"] = 64  # plenty of unused codes
        base_config["dead_code_threshold"] = 1
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d, training=True)  # first training call
        cb_before = ops.convert_to_numpy(layer.embeddings).copy()
        _ = layer(sample_input_2d, training=True)  # second call: unused codes get reinit
        cb_after = ops.convert_to_numpy(layer.embeddings)
        # at least one code must have changed
        assert not np.allclose(cb_before, cb_after)


# ============================================================================
# K-means init
# ============================================================================

class TestKMeansInit:
    def test_kmeans_init_changes_codebook(self, base_config, sample_input_2d):
        base_config["kmeans_init"] = True
        base_config["kmeans_init_steps"] = 1
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d, training=False)  # build
        cb_before = ops.convert_to_numpy(layer.embeddings).copy()
        assert float(ops.convert_to_numpy(layer.kmeans_init_done)) < 0.5
        _ = layer(sample_input_2d, training=True)
        cb_after = ops.convert_to_numpy(layer.embeddings)
        assert float(ops.convert_to_numpy(layer.kmeans_init_done)) > 0.5
        assert not np.allclose(cb_before, cb_after)

    def test_kmeans_init_runs_only_once(self, base_config, sample_input_2d):
        base_config["kmeans_init"] = True
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d, training=True)
        cb_after_first = ops.convert_to_numpy(layer.embeddings).copy()
        # second training call must NOT re-run k-means (gradient updates may
        # change codebook but only marginally; we check the flag).
        assert float(ops.convert_to_numpy(layer.kmeans_init_done)) > 0.5
        _ = layer(sample_input_2d, training=True)
        # flag stays at 1
        assert float(ops.convert_to_numpy(layer.kmeans_init_done)) > 0.5


# ============================================================================
# Aux losses
# ============================================================================

class TestAuxLosses:
    def test_diversity_loss_present(self, base_config, sample_input_2d):
        base_config["diversity_coefficient"] = 0.5
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d, training=True)
        # codebook + commitment + diversity = 3
        assert len(layer.losses) == 3

    def test_orthogonal_loss_present(self, base_config, sample_input_2d):
        base_config["orthogonal_reg_coefficient"] = 0.5
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d, training=True)
        assert len(layer.losses) == 3

    def test_aux_losses_absent_when_zero(self, base_config, sample_input_2d):
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d, training=True)
        assert len(layer.losses) == 2


# ============================================================================
# Indices API
# ============================================================================

class TestIndicesAPI:
    def test_get_indices_single_head(self, base_config, sample_input_2d):
        layer = VectorQuantizerRotationTrick(**base_config)
        idx = layer.get_codebook_indices(sample_input_2d)
        assert idx.shape == sample_input_2d.shape[:-1]

    def test_get_indices_multi_head(self, base_config, sample_input_2d):
        base_config["num_heads"] = 4
        layer = VectorQuantizerRotationTrick(**base_config)
        idx = layer.get_codebook_indices(sample_input_2d)
        assert idx.shape == sample_input_2d.shape[:-1] + (4,)

    def test_indices_roundtrip_single_head(self, base_config, sample_input_2d):
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d)
        idx = layer.get_codebook_indices(sample_input_2d)
        q = layer.quantize_from_indices(idx)
        assert q.shape == sample_input_2d.shape

    def test_indices_roundtrip_multi_head(self, base_config, sample_input_2d):
        base_config["num_heads"] = 4
        layer = VectorQuantizerRotationTrick(**base_config)
        _ = layer(sample_input_2d)
        idx = layer.get_codebook_indices(sample_input_2d)
        q = layer.quantize_from_indices(idx)
        assert q.shape == sample_input_2d.shape


# ============================================================================
# Gradient flow
# ============================================================================

class TestGradientFlow:
    @pytest.mark.parametrize(
        "mode", ["rotation", "reflection", "no_grad_scale", "ste"]
    )
    def test_gradients_flow(self, base_config, sample_input_2d, mode):
        base_config["gradient_mode"] = mode
        layer = VectorQuantizerRotationTrick(**base_config)
        x_var = tf.Variable(sample_input_2d)
        with tf.GradientTape() as tape:
            out = layer(x_var, training=True)
            loss = ops.mean(ops.square(out)) + ops.sum(layer.losses)
        grads = tape.gradient(loss, [x_var] + layer.trainable_variables)
        assert all(g is not None for g in grads)
        for g in grads:
            g_np = ops.convert_to_numpy(g)
            assert not np.allclose(g_np, 0.0)


# ============================================================================
# Serialization
# ============================================================================

class TestSerialization:
    def test_config_completeness(self, base_config):
        layer = VectorQuantizerRotationTrick(**base_config)
        cfg = layer.get_config()
        for key in (
                "num_embeddings", "embedding_dim", "commitment_cost",
                "gradient_mode", "distance_mode", "initializer", "use_ema",
                "ema_decay", "epsilon", "num_heads", "kmeans_init",
                "kmeans_init_steps", "kmeans_seed", "dead_code_threshold",
                "diversity_coefficient", "orthogonal_reg_coefficient",
        ):
            assert key in cfg, f"missing key: {key}"

    @pytest.mark.parametrize(
        "mode", ["rotation", "reflection", "no_grad_scale", "ste"]
    )
    @pytest.mark.parametrize("num_heads", [1, 4])
    def test_save_load_round_trip(
            self, base_config, sample_input_2d, mode, num_heads,
    ):
        base_config["gradient_mode"] = mode
        base_config["num_heads"] = num_heads

        inp = keras.Input(shape=sample_input_2d.shape[1:])
        out = VectorQuantizerRotationTrick(**base_config)(inp)
        model = keras.Model(inp, out)

        orig = model(sample_input_2d, training=False)

        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, f"layer_{mode}_h{num_heads}.keras")
            model.save(fp)
            loaded = keras.models.load_model(fp)
            new = loaded(sample_input_2d, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(orig),
                ops.convert_to_numpy(new),
                atol=1e-5, rtol=1e-5,
            )

    def test_save_load_with_ema(self, base_config, sample_input_2d):
        base_config["use_ema"] = True
        inp = keras.Input(shape=sample_input_2d.shape[1:])
        out = VectorQuantizerRotationTrick(**base_config)(inp)
        model = keras.Model(inp, out)
        # warm-up EMA
        for _ in range(3):
            _ = model(sample_input_2d, training=True)
        orig = model(sample_input_2d, training=False)
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "ema.keras")
            model.save(fp)
            loaded = keras.models.load_model(fp)
            new = loaded(sample_input_2d, training=False)
            np.testing.assert_allclose(
                ops.convert_to_numpy(orig),
                ops.convert_to_numpy(new),
                atol=1e-5, rtol=1e-5,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
