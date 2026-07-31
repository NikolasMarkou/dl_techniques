import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models
import tempfile
import os
from typing import Any, Dict

from dl_techniques.layers.transformers.vision_encoder import (
    VisionEncoder,
    create_vision_encoder,
    create_vit_encoder,
    create_siglip_encoder,
)

# G-02: `MASK_INCOMPATIBLE_OUTPUT_MODES` no longer exists. It named the three
# pooling strategies `VisionEncoder.call` used to refuse with a mask (D-013);
# F-24 is fixed in `layers/sequence_pooling/` and they now isolate a masked
# patch exactly, so the constant, the raise and this import are gone and the
# three modes are asserted here as ISOLATING rather than as refused.
FORMERLY_REFUSED_OUTPUT_MODES = ['top_k_max', 'top_k_mean', 'weighted']


# --- Test Class ---
class TestVisionEncoder:
    """
    Comprehensive and modern test suite for the VisionEncoder.
    This suite follows modern Keras 3 testing best practices and covers all
    architectural variations and factory patterns.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Provides a basic configuration for a small, testable encoder."""
        return {
            'img_size': 32,
            'patch_size': 8,
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 4,
        }

    @pytest.fixture
    def vit_config(self) -> Dict[str, Any]:
        """Provides ViT-style configuration."""
        return {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'use_cls_token': True,
            'output_mode': 'cls'
        }

    @pytest.fixture
    def modern_config(self) -> Dict[str, Any]:
        """Provides modern encoder configuration with advanced features."""
        # Ensure num_patches matches window_size^2 for WindowAttention compatibility
        window_size = 4
        num_patches_per_dim = window_size
        patch_size = 8
        img_size = num_patches_per_dim * patch_size  # 4 * 8 = 32

        return {
            'img_size': img_size,
            'patch_size': patch_size,
            'embed_dim': 128,
            'depth': 4,
            'num_heads': 4,
            'patch_embed_type': 'siglip',
            'attention_type': 'window',
            'normalization_type': 'rms_norm',
            'normalization_position': 'pre',
            'ffn_type': 'swiglu',
            'stochastic_depth_rate': 0.1,
            'output_mode': 'mean',
            'use_cls_token': False,
            'attention_args': {'window_size': window_size}
        }

    @pytest.fixture
    def sample_images(self) -> tf.Tensor:
        """Provides a batch of sample images for testing."""
        return tf.random.uniform(
            shape=(2, 32, 32, 3), minval=0.0, maxval=1.0, dtype=tf.float32
        )

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, basic_config):
        """Tests encoder initialization with default parameters."""
        encoder = VisionEncoder(**basic_config)
        assert not encoder.built
        assert encoder.patch_embed_type == 'linear'
        assert encoder.attention_type == 'multi_head'
        assert encoder.normalization_type == 'layer_norm'
        assert encoder.ffn_type == 'mlp'
        assert encoder.output_mode == 'cls'
        assert encoder.use_cls_token

    @pytest.mark.parametrize("patch_embed_type", ['linear', 'siglip', 'conv', 'hybrid'])
    def test_initialization_patch_embed_types(self, basic_config, patch_embed_type):
        """Tests initialization with different patch embedding types."""
        config = {**basic_config, 'patch_embed_type': patch_embed_type}
        encoder = VisionEncoder(**config)
        assert encoder.patch_embed_type == patch_embed_type
        assert hasattr(encoder, 'patch_embed')

    @pytest.mark.parametrize("attention_type", [
        'multi_head', 'window', 'group_query', 'differential'
    ])
    def test_initialization_attention_types(self, basic_config, attention_type):
        """Tests initialization with different attention mechanisms."""
        config = {**basic_config, 'attention_type': attention_type}
        if attention_type == 'window':
            # This makes num_patches match window_size^2
            config['img_size'] = 16
            config['patch_size'] = 8
            config['attention_args'] = {'window_size': 2}
        encoder = VisionEncoder(**config)
        assert encoder.attention_type == attention_type

    def test_build_process(self, basic_config, sample_images):
        """Tests that encoder and all sub-layers are built correctly."""
        encoder = VisionEncoder(**basic_config)
        assert not encoder.built
        output = encoder(sample_images)
        assert encoder.built
        assert hasattr(encoder.patch_embed, 'built')
        assert encoder.patch_embed.built
        assert encoder.cls_token is not None
        assert encoder.cls_token.shape == (1, 1, basic_config['embed_dim'])

    def test_build_without_cls_token(self, basic_config, sample_images):
        """Tests that CLS token is not created when use_cls_token is False."""
        config = {**basic_config, 'use_cls_token': False, 'output_mode': 'mean'}
        encoder = VisionEncoder(**config)
        encoder(sample_images)
        assert encoder.cls_token is None

    # ===============================================
    # 2. Parameter Validation Tests
    # ===============================================
    def test_invalid_img_size_patch_size(self):
        """Tests validation of img_size and patch_size compatibility."""
        with pytest.raises(ValueError, match="img_size .* must be divisible by patch_size"):
            VisionEncoder(img_size=32, patch_size=7)

    def test_invalid_embed_dim_num_heads(self):
        """Tests validation of embed_dim and num_heads compatibility."""
        with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
            VisionEncoder(embed_dim=64, num_heads=5)

    def test_invalid_output_mode_cls_without_cls_token(self):
        """Tests validation of output_mode='cls' requiring use_cls_token=True."""
        with pytest.raises(ValueError, match="output_mode='cls' requires use_cls_token=True"):
            VisionEncoder(output_mode='cls', use_cls_token=False)

    # ===============================================
    # 3. Forward Pass and Core Behavior Tests
    # ===============================================
    @pytest.mark.parametrize("output_mode", ['cls', 'mean', 'max'])
    def test_forward_pass_pooled_output_modes(self, basic_config, sample_images, output_mode):
        """Tests forward pass with pooled output modes."""
        config = {**basic_config, 'output_mode': output_mode}
        encoder = VisionEncoder(**config)
        output = encoder(sample_images, training=False)
        expected_shape = (sample_images.shape[0], basic_config['embed_dim'])
        assert output.shape == expected_shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_forward_pass_sequence_output(self, basic_config, sample_images):
        """Tests forward pass with 'none' output mode."""
        config = {**basic_config, 'output_mode': 'none'}
        encoder = VisionEncoder(**config)
        output = encoder(sample_images, training=False)
        expected_shape = (sample_images.shape[0], encoder.seq_len, basic_config['embed_dim'])
        assert output.shape == expected_shape

    def test_forward_pass_no_cls_token(self, basic_config, sample_images):
        """Tests forward pass without CLS token."""
        config = {**basic_config, 'use_cls_token': False, 'output_mode': 'mean'}
        encoder = VisionEncoder(**config)
        output = encoder(sample_images, training=False)
        expected_shape = (sample_images.shape[0], basic_config['embed_dim'])
        assert output.shape == expected_shape
        assert encoder.seq_len == encoder.num_patches

    def test_training_vs_inference_modes(self, basic_config, sample_images):
        """Tests behavior difference between training and inference modes."""
        config = {**basic_config, 'dropout_rate': 0.5, 'pos_dropout_rate': 0.3}
        encoder = VisionEncoder(**config)
        output_train = encoder(sample_images, training=True)
        output_infer = encoder(sample_images, training=False)
        assert output_train.shape == output_infer.shape
        assert not np.allclose(ops.convert_to_numpy(output_train), ops.convert_to_numpy(output_infer))

    def test_get_cls_features(self, basic_config, sample_images):
        """Tests get_cls_features method."""
        encoder = VisionEncoder(**basic_config)
        cls_features = encoder.get_cls_features(sample_images, training=False)
        expected_shape = (sample_images.shape[0], basic_config['embed_dim'])
        assert cls_features.shape == expected_shape

    def test_get_patch_features(self, basic_config, sample_images):
        """Tests get_patch_features method."""
        encoder = VisionEncoder(**basic_config)
        patch_features = encoder.get_patch_features(sample_images, training=False)
        expected_shape = (sample_images.shape[0], encoder.num_patches, basic_config['embed_dim'])
        assert patch_features.shape == expected_shape

    def test_get_spatial_features(self, basic_config, sample_images):
        """Tests get_spatial_features method."""
        encoder = VisionEncoder(**basic_config)
        spatial_features = encoder.get_spatial_features(sample_images, training=False)
        patches_per_dim = basic_config['img_size'] // basic_config['patch_size']
        expected_shape = (sample_images.shape[0], patches_per_dim, patches_per_dim, basic_config['embed_dim'])
        assert spatial_features.shape == expected_shape

    # ===============================================
    # 4. Serialization Tests (The Gold Standard)
    # ===============================================
    def test_full_serialization_cycle_basic(self, basic_config, sample_images):
        """Tests full serialization cycle with basic configuration."""
        inputs = layers.Input(shape=sample_images.shape[1:])
        outputs = VisionEncoder(**basic_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_images, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_basic_vision_encoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_images, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    def test_full_serialization_cycle_modern(self, modern_config):
        """Tests full serialization cycle with modern configuration."""
        img_size = modern_config['img_size']
        sample_images_modern = tf.random.uniform((2, img_size, img_size, 3))

        inputs = layers.Input(shape=sample_images_modern.shape[1:])
        outputs = VisionEncoder(**modern_config)(inputs)
        model = models.Model(inputs, outputs)
        original_prediction = model(sample_images_modern, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_modern_vision_encoder.keras")
            model.save(filepath)
            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_images_modern, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    # ===============================================
    # 5. Gradient and Training Integration Tests
    # ===============================================
    def test_gradient_flow(self, basic_config, sample_images):
        """Tests gradient flow through encoder."""
        encoder = VisionEncoder(**basic_config)
        x_var = tf.Variable(sample_images)

        with tf.GradientTape() as tape:
            output = encoder(x_var, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, encoder.trainable_variables)
        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "A gradient is None."

    def test_model_training_loop_integration(self, basic_config):
        """Tests encoder integration in a standard training loop."""
        img_size = basic_config['img_size']
        model = models.Sequential([
            layers.InputLayer(shape=(img_size, img_size, 3)),
            VisionEncoder(**basic_config),
            layers.Dense(10)
        ])
        model.compile("adam", "sparse_categorical_crossentropy")
        x_train = tf.random.uniform((8, img_size, img_size, 3))
        y_train = tf.random.uniform((8,), maxval=10, dtype=tf.int32)
        history = model.fit(x_train, y_train, epochs=1, batch_size=4, verbose=0)
        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0])

    # ===============================================
    # 6. Factory Functions Tests
    # ===============================================
    def test_create_vision_encoder_factory(self):
        """Tests the create_vision_encoder factory function."""
        encoder = create_vision_encoder(
            img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4
        )
        assert isinstance(encoder, VisionEncoder)
        assert encoder.img_size == 32
        assert encoder.embed_dim == 64

    def test_create_vit_encoder_factory(self):
        """Tests the create_vit_encoder factory function."""
        encoder = create_vit_encoder(img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4)
        assert isinstance(encoder, VisionEncoder)
        assert encoder.patch_embed_type == 'linear'
        assert encoder.attention_type == 'multi_head'
        assert encoder.use_cls_token
        assert encoder.output_mode == 'cls'

    def test_create_siglip_encoder_factory(self):
        """Tests the create_siglip_encoder factory function."""
        encoder = create_siglip_encoder(img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4)
        assert isinstance(encoder, VisionEncoder)
        assert encoder.patch_embed_type == 'siglip'

    def test_factory_parameter_validation(self):
        """Tests that factory functions validate parameters properly."""
        with pytest.raises(ValueError, match="img_size .* must be divisible by patch_size"):
            create_vision_encoder(img_size=32, patch_size=7)

    # ===============================================
    # 7. Configuration and Get Config Tests
    # ===============================================
    def test_get_config_completeness(self, basic_config):
        """Tests that get_config contains all initialization parameters."""
        encoder = VisionEncoder(**basic_config)
        config = encoder.get_config()
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"
        assert 'patch_embed_type' in config

    def test_config_reconstruction(self, basic_config):
        """Tests that an encoder can be reconstructed from its config."""
        original_encoder = VisionEncoder(**basic_config)
        config = original_encoder.get_config()
        reconstructed_encoder = VisionEncoder.from_config(config)
        assert reconstructed_encoder.img_size == original_encoder.img_size
        assert reconstructed_encoder.embed_dim == original_encoder.embed_dim
        assert reconstructed_encoder.depth == original_encoder.depth

    def test_compute_output_shape(self, basic_config):
        """Tests the compute_output_shape method."""
        encoder = VisionEncoder(**basic_config)
        input_shape = (None, 32, 32, 3)
        output_shape = encoder.compute_output_shape(input_shape)
        expected_shape = (None, basic_config['embed_dim'])  # Default is 'cls'
        assert output_shape == expected_shape

    def test_compute_output_shape_sequence(self, basic_config):
        """Tests compute_output_shape with 'none' output mode."""
        config = {**basic_config, 'output_mode': 'none'}
        encoder = VisionEncoder(**config)
        input_shape = (None, 32, 32, 3)
        output_shape = encoder.compute_output_shape(input_shape)
        expected_shape = (None, encoder.seq_len, basic_config['embed_dim'])
        assert output_shape == expected_shape

    # ===============================================
    # 8. Mixed Precision Compatibility Test
    # ===============================================
    def test_mixed_precision_compatibility(self, basic_config, sample_images):
        """Tests encoder compatibility with mixed precision training."""
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        try:
            encoder = VisionEncoder(**basic_config)
            output = encoder(sample_images, training=False)
            assert output.dtype == tf.float16
        finally:
            keras.mixed_precision.set_global_policy('float32')


# =====================================================================
# F-04 — `attention_mask` must reach the TransformerLayer stack.
#
# Before plan-2026-07-31T042809-ddc92265/iter-1/step-5, ``VisionEncoder.call``
# forwarded ``attention_mask`` *only* to ``self.pooling_layer``; the helper that
# actually runs the transformer stack never received it. Two separate defects
# followed, and both are pinned below:
#
#   1. Masked patches still participated in every self-attention layer, so they
#      influenced the representation of every *unmasked* token (and, for
#      ``output_mode='cls'``, the mask had no observable effect whatsoever).
#   2. The mask handed to ``self.pooling_layer`` was over PATCHES only, while the
#      pooled sequence carries a CLS token at position 0. With
#      ``use_cls_token=True`` that is a length mismatch: 13 of the 18 pooling
#      strategies raised ``InvalidArgumentError`` at runtime, and ``'last'``
#      silently selected a position one off.
# =====================================================================

# Small, fast geometry: 16/8 -> 2x2 = 4 patches.
MASK_CFG: Dict[str, Any] = {
    'img_size': 16,
    'patch_size': 8,
    'embed_dim': 32,
    'depth': 2,
    'num_heads': 4,
    'use_bias': True,
}
MASK_PATCHES_PER_DIM = MASK_CFG['img_size'] // MASK_CFG['patch_size']  # 2
MASK_NUM_PATCHES = MASK_PATCHES_PER_DIM ** 2  # 4
MASKED_PATCH = 3  # the LAST patch; chosen so no positional strategy selects it

# Every pooling strategy `SequencePooling` accepts, as reachable through
# `VisionEncoder(output_mode=...)`.
ALL_OUTPUT_MODES = [
    'cls', 'first', 'last', 'middle',
    'mean', 'max', 'min', 'sum',
    'mean_max', 'mean_std', 'mean_max_min',
    'attention', 'multi_head_attention', 'weighted',
    'top_k_mean', 'top_k_max',
    'none', 'flatten',
]

# Strategies whose output excludes the masked patch AT `MASKED_PATCH = 3`, so a
# perturbation of that patch must leave the output BIT-IDENTICAL.
#
# SCOPE — this list is MASK-PATTERN-DEPENDENT, not a universal property.
# `MASKED_PATCH = 3` is the LAST patch, i.e. a contiguous-prefix keep-mask, and
# the four POSITIONAL modes (`cls`, `first`, `last`, `middle`) are in this list
# only because of that. MEASURED with patch 2 masked instead (4 patches,
# non-prefix): `use_cls_token=True` `last` leaks 9.1e-01; `use_cls_token=False`
# `last` and `middle` leak 1.3e+00; with patch 0 masked, `cls`/`first` leak.
# `cls`/`first`/`middle` select a POSITION and never consult the mask, which is
# their contract; `last` DOES consult it (`sum(mask) - 1`) and is wrong off
# prefix — a genuine defect carried forward as F-25. Do NOT read the name of
# this constant as "these modes isolate under any mask".
#
# `weighted`, `top_k_mean` and `top_k_max` are in this list as of G-02. They
# used to sit in the exclusions because `SequencePooling` leaked a masked
# position into them (F-24: `weighted` multiplied the position weights by the
# mask BEFORE the softmax, so a masked position kept weight `softmax(0) != 0`;
# `top_k_*` ranked by the norms of the MASKED inputs but gathered from the
# UNMASKED ones, leaking whenever `k` exceeded a row's kept count -- and
# `VisionEncoder` never exposes `top_k`, so `k` was always the default 10).
# That defect is FIXED in `layers/sequence_pooling/`, `VisionEncoder`'s
# `NotImplementedError` containment is gone, and all three now measure exactly
# 0.0 here (see `TestMaskedPoolingIsIsolated` below).
#
# `flatten` is the ONE remaining exclusion, and it is not a defect: the output
# literally contains every token, which is what the mode means.
ISOLATING_OUTPUT_MODES = [
    'cls', 'first', 'last', 'middle',
    'mean', 'max', 'min', 'sum',
    'mean_max', 'mean_std', 'mean_max_min',
    'attention', 'multi_head_attention', 'weighted',
    'top_k_mean', 'top_k_max',
    'none',
]
NON_ISOLATING_OUTPUT_MODES = ['flatten']
assert sorted(ISOLATING_OUTPUT_MODES + NON_ISOLATING_OUTPUT_MODES) == sorted(ALL_OUTPUT_MODES)


def _seeded_encoder(seed: int = 1234, **overrides: Any) -> VisionEncoder:
    """Build a `VisionEncoder` and assign EVERY weight from a seeded RNG.

    Fresh Keras initialisers leave biases at zero, which (per decision D-008 of
    this plan, recorded for `GatedLinearAttentionBlock`) can make a masking site
    unobservable by construction. These fixtures therefore put the layer in the
    state a *trained* model is in: non-zero biases, non-unit norm gains.

    :param seed: RNG seed for the weight assignment.
    :param overrides: Keyword overrides merged over ``MASK_CFG``.
    :return: A built encoder with fully randomised, non-degenerate weights.
    """
    cfg = {**MASK_CFG, **overrides}
    encoder = VisionEncoder(**cfg)
    encoder.build((None, cfg['img_size'], cfg['img_size'], 3))

    rng = np.random.default_rng(seed)
    saw_nonzero_bias = False
    for w in encoder.weights:
        shape = tuple(w.shape)
        name = w.path.split('/')[-1]
        if 'gamma' in name or 'scale' in name:
            # Keep normalisation gains near 1 - a near-zero gain would collapse
            # the signal and make the whole probe vacuous.
            value = 1.0 + 0.05 * rng.normal(size=shape)
        elif 'beta' in name or 'bias' in name:
            value = 0.05 * rng.normal(size=shape)
            saw_nonzero_bias = True
        else:
            value = 0.1 * rng.normal(size=shape)
        w.assign(ops.cast(ops.convert_to_tensor(value), w.dtype))

    assert saw_nonzero_bias, (
        "Fixture is degenerate: no bias weight was assigned a non-zero value, so "
        "masking sites downstream of a zeroed activation cannot be observed."
    )
    return encoder


def _mask_images(seed: int = 7, batch: int = 2) -> np.ndarray:
    """Return a deterministic batch of images."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(batch, MASK_CFG['img_size'], MASK_CFG['img_size'], 3)).astype('float32')


def _perturb_patch(images: np.ndarray, patch_index: int, seed: int = 99) -> np.ndarray:
    """Perturb one patch's PIXELS with a large, NON-UNIFORM signal.

    A uniform per-channel offset is mean-centred away by the first LayerNorm
    (measured in step 1 of this plan: a real leak read as 3.58e-07 of
    reassociation noise). The perturbation here varies per pixel and per channel.

    :param images: Batch of images ``(B, H, W, C)``.
    :param patch_index: Row-major patch index to perturb.
    :param seed: RNG seed for the perturbation.
    :return: A copy of ``images`` with that patch perturbed.
    """
    ps = MASK_CFG['patch_size']
    r = (patch_index // MASK_PATCHES_PER_DIM) * ps
    c = (patch_index % MASK_PATCHES_PER_DIM) * ps
    rng = np.random.default_rng(seed)
    out = np.array(images, copy=True)
    out[:, r:r + ps, c:c + ps, :] += (
        5.0 * rng.normal(size=(images.shape[0], ps, ps, images.shape[-1]))
    ).astype(images.dtype)
    return out


def _patch_mask(masked_patch: int = MASKED_PATCH, batch: int = 2, dtype: str = 'float32') -> np.ndarray:
    """Return a ``(B, num_patches)`` keep-mask (1 = attend) with one patch masked."""
    m = np.ones((batch, MASK_NUM_PATCHES), dtype=dtype)
    m[:, masked_patch] = 0
    return m


def _np(x: Any) -> np.ndarray:
    return ops.convert_to_numpy(x)


class TestAttentionMaskReachesSelfAttention:
    """F-04: the caller's ``attention_mask`` must gate self-attention, not just pooling."""

    def test_masked_patch_perturbation_cannot_reach_cls(self):
        """Perturbing a MASKED patch's pixels must leave the CLS output bit-identical.

        This is the SC-5 guard. It carries its own in-band live control: the same
        perturbation applied to an UNMASKED patch must move CLS by a wide margin,
        so a test that passes because *nothing* moves is impossible.
        """
        encoder = _seeded_encoder(output_mode='cls', use_cls_token=True)
        images = _mask_images()
        mask = ops.convert_to_tensor(_patch_mask())

        base = _np(encoder(ops.convert_to_tensor(images), attention_mask=mask, training=False))

        # LIVE CONTROL: an unmasked patch must move the output a lot.
        live = _np(encoder(
            ops.convert_to_tensor(_perturb_patch(images, 0)),
            attention_mask=mask, training=False,
        ))
        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-2, (
            f"Vacuous probe: perturbing UNMASKED patch 0 moved CLS by only "
            f"{live_delta:.3e}; the isolation assertion below would prove nothing."
        )

        # THE GUARD: the masked patch must be unable to reach CLS at all.
        masked = _np(encoder(
            ops.convert_to_tensor(_perturb_patch(images, MASKED_PATCH)),
            attention_mask=mask, training=False,
        ))
        np.testing.assert_allclose(masked, base, rtol=0, atol=0)

    def test_mask_is_not_vacuous_for_cls_output(self):
        """A mask must CHANGE the CLS output relative to no mask at all.

        Pre-fix this was measured byte-identical (max abs diff 0.0) - the mask
        never reached self-attention and, for ``output_mode='cls'``, never even
        reached pooling.
        """
        encoder = _seeded_encoder(output_mode='cls', use_cls_token=True)
        images = ops.convert_to_tensor(_mask_images())

        unmasked = _np(encoder(images, attention_mask=None, training=False))
        masked = _np(encoder(images, attention_mask=ops.convert_to_tensor(_patch_mask()), training=False))
        delta = float(np.max(np.abs(masked - unmasked)))
        assert delta > 1e-3, (
            f"attention_mask has no observable effect on the CLS output "
            f"(max abs diff {delta:.6e}) - it is being dropped before self-attention."
        )

    @pytest.mark.parametrize("use_cls_token", [True, False])
    @pytest.mark.parametrize("output_mode", ISOLATING_OUTPUT_MODES)
    def test_masked_patch_is_isolated_for_every_output_mode(self, output_mode, use_cls_token):
        """Bit-identity under masked-patch perturbation, across pooling modes.

        Covering both ``use_cls_token`` settings is what stops a CLS-splice
        off-by-one from hiding: with the mask misaligned by one position the
        WRONG patch would be masked and this assertion would fire.
        """
        if output_mode == 'cls' and not use_cls_token:
            pytest.skip("output_mode='cls' requires use_cls_token=True")

        encoder = _seeded_encoder(output_mode=output_mode, use_cls_token=use_cls_token)
        images = _mask_images()
        mask = ops.convert_to_tensor(_patch_mask())

        base = _np(encoder(ops.convert_to_tensor(images), attention_mask=mask, training=False))
        masked = _np(encoder(
            ops.convert_to_tensor(_perturb_patch(images, MASKED_PATCH)),
            attention_mask=mask, training=False,
        ))
        live = _np(encoder(
            ops.convert_to_tensor(_perturb_patch(images, 0)),
            attention_mask=mask, training=False,
        ))

        if output_mode == 'none':
            # The full sequence is returned, so the masked patch's OWN row does
            # change; every other row must not.
            masked_pos = MASKED_PATCH + (1 if use_cls_token else 0)
            keep = [i for i in range(base.shape[1]) if i != masked_pos]
            base, masked, live = base[:, keep], masked[:, keep], live[:, keep]

        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-3, (
            f"Vacuous probe for output_mode={output_mode!r}, use_cls_token="
            f"{use_cls_token}: the live control moved only {live_delta:.3e}."
        )
        np.testing.assert_allclose(masked, base, rtol=0, atol=0)

    def test_helper_methods_honour_attention_mask(self):
        """``get_cls_features``/``get_patch_features``/``get_spatial_features`` pass the mask through."""
        encoder = _seeded_encoder(output_mode='cls', use_cls_token=True)
        images = _mask_images()
        pert = _perturb_patch(images, MASKED_PATCH)
        mask = ops.convert_to_tensor(_patch_mask())

        cls_base = _np(encoder.get_cls_features(ops.convert_to_tensor(images), attention_mask=mask))
        cls_pert = _np(encoder.get_cls_features(ops.convert_to_tensor(pert), attention_mask=mask))
        np.testing.assert_allclose(cls_pert, cls_base, rtol=0, atol=0)

        keep = [i for i in range(MASK_NUM_PATCHES) if i != MASKED_PATCH]
        patch_base = _np(encoder.get_patch_features(ops.convert_to_tensor(images), attention_mask=mask))
        patch_pert = _np(encoder.get_patch_features(ops.convert_to_tensor(pert), attention_mask=mask))
        np.testing.assert_allclose(patch_pert[:, keep], patch_base[:, keep], rtol=0, atol=0)
        # Live control: the masked patch's own representation DOES change.
        assert float(np.max(np.abs(patch_pert[:, MASKED_PATCH] - patch_base[:, MASKED_PATCH]))) > 1e-3

        spat_base = _np(encoder.get_spatial_features(ops.convert_to_tensor(images), attention_mask=mask))
        spat_pert = _np(encoder.get_spatial_features(ops.convert_to_tensor(pert), attention_mask=mask))
        spat_base = spat_base.reshape(spat_base.shape[0], -1, MASK_CFG['embed_dim'])
        spat_pert = spat_pert.reshape(spat_pert.shape[0], -1, MASK_CFG['embed_dim'])
        np.testing.assert_allclose(spat_pert[:, keep], spat_base[:, keep], rtol=0, atol=0)

    def test_boolean_mask_is_accepted(self):
        """A boolean keep-mask must survive the CLS splice (dtype-preserving concat)."""
        encoder = _seeded_encoder(output_mode='cls', use_cls_token=True)
        images = _mask_images()
        bool_mask = ops.convert_to_tensor(_patch_mask(dtype='bool'))
        base = _np(encoder(ops.convert_to_tensor(images), attention_mask=bool_mask, training=False))
        pert = _np(encoder(
            ops.convert_to_tensor(_perturb_patch(images, MASKED_PATCH)),
            attention_mask=bool_mask, training=False,
        ))
        np.testing.assert_allclose(pert, base, rtol=0, atol=0)

    def test_rank_3_mask_fails_loud(self):
        """The documented contract is rank-2 ``(B, num_patches)``; anything else must raise."""
        encoder = _seeded_encoder(output_mode='cls', use_cls_token=True)
        images = ops.convert_to_tensor(_mask_images())
        bad = ops.convert_to_tensor(
            np.ones((2, MASK_NUM_PATCHES, MASK_NUM_PATCHES), dtype='float32')
        )
        with pytest.raises(ValueError, match="rank-2"):
            encoder(images, attention_mask=bad, training=False)


class TestPoolingMaskIsClsExtended:
    """Second defect: the un-extended patch mask was mis-sized at the pooling call.

    With ``use_cls_token=True`` the pooled sequence is ``1 + num_patches`` long
    while the caller's mask is ``num_patches`` long. Measured at HEAD before this
    step: 13 of the 18 strategies raised ``InvalidArgumentError`` and ``'last'``
    silently picked a position one short.
    """

    @pytest.mark.parametrize("use_cls_token", [True, False])
    @pytest.mark.parametrize("output_mode", ALL_OUTPUT_MODES)
    def test_every_pooling_strategy_accepts_a_patch_mask(self, output_mode, use_cls_token):
        if output_mode == 'cls' and not use_cls_token:
            pytest.skip("output_mode='cls' requires use_cls_token=True")

        encoder = _seeded_encoder(output_mode=output_mode, use_cls_token=use_cls_token)

        # G-02: `weighted`/`top_k_mean`/`top_k_max` used to take a
        # `pytest.raises(NotImplementedError)` branch here (F-24, D-013). F-24 is
        # fixed, the refusal is gone, and all 18 modes now go down the SAME path
        # -- accepted, correctly shaped and finite. Their masked-patch isolation
        # is asserted in `TestMaskedPoolingIsIsolated` below.
        out = encoder(
            ops.convert_to_tensor(_mask_images()),
            attention_mask=ops.convert_to_tensor(_patch_mask()),
            training=False,
        )
        expected = encoder.compute_output_shape((2, MASK_CFG['img_size'], MASK_CFG['img_size'], 3))
        assert tuple(out.shape)[1:] == tuple(expected)[1:]
        assert np.all(np.isfinite(_np(out)))

    def test_last_strategy_indexes_the_cls_extended_sequence(self):
        """``'last'`` picks ``sum(mask) - 1``; with the un-extended mask that is off by one."""
        encoder = _seeded_encoder(output_mode='last', use_cls_token=True)
        images = ops.convert_to_tensor(_mask_images())
        mask = ops.convert_to_tensor(_patch_mask())

        pooled = _np(encoder(images, attention_mask=mask, training=False))
        sequence = _np(encoder._get_full_sequence_features(
            images, attention_mask=mask, training=False
        ))

        # 4 valid positions in the CLS-extended mask -> index 3.
        np.testing.assert_allclose(pooled, sequence[:, 3, :], rtol=0, atol=0)
        # And it must NOT be the un-extended answer (index 2).
        assert float(np.max(np.abs(pooled - sequence[:, 2, :]))) > 1e-4


class TestAttentionMaskSerialization:
    """I3: a ``.keras`` round-trip must restore VALUES on a config using the mask path."""

    def test_roundtrip_preserves_values_on_the_mask_path(self):
        img_size = MASK_CFG['img_size']
        image_in = layers.Input(shape=(img_size, img_size, 3), name='image')
        mask_in = layers.Input(shape=(MASK_NUM_PATCHES,), name='patch_mask')
        encoder = VisionEncoder(**MASK_CFG, output_mode='mean', use_cls_token=True)
        out = encoder(image_in, attention_mask=mask_in)
        model = models.Model([image_in, mask_in], out)

        images = _mask_images()
        mask = _patch_mask()
        before = _np(model([images, mask], training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'vision_encoder_mask.keras')
            model.save(path)
            loaded = models.load_model(path)
            after = _np(loaded([images, mask], training=False))

        np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)
        # Non-vacuity: the reloaded model must still honour the mask.
        after_nomask = _np(loaded([images, np.ones_like(mask)], training=False))
        assert float(np.max(np.abs(after_nomask - after))) > 1e-4


class TestMaskedPoolingIsIsolated:
    """G-02: the three formerly-REFUSED pooling modes now ISOLATE a masked patch.

    This class REPLACES ``TestMaskIncompatiblePooling`` at the same site. That
    class asserted the opposite contract — that ``VisionEncoder.call`` raises
    ``NotImplementedError`` for ``weighted`` / ``top_k_mean`` / ``top_k_max``
    whenever an ``attention_mask`` is supplied (D-013 containment of finding
    F-24, which lived in ``layers/sequence_pooling/``).

    Its ``test_the_leak_the_guard_prevents_is_real`` was a deliberate
    SELF-OBSOLETING guard: it asserted the leak stayed ``> 1e-3`` and said in its
    own docstring that if the leak ever measured ``0.0``, the containment was
    obsolete and should be REMOVED rather than left green. F-24 is now fixed at
    the root, the leak measures exactly ``0.0``, and this suite is that test
    honouring its own instruction.

    Leak measured through this exact probe BEFORE the fix (one patch masked,
    that patch's pixels perturbed, seeded weights; required movement ``0.0``):

    =============  ==================  ===================
    output_mode    use_cls_token=True  use_cls_token=False
    =============  ==================  ===================
    weighted       2.317689e-01        1.119100e-01
    top_k_mean     2.349049e-01        1.266325e-01
    top_k_max      4.473200e-01        2.670361e-01
    =============  ==================  ===================
    """

    @pytest.mark.parametrize("use_cls_token", [True, False])
    @pytest.mark.parametrize("output_mode", FORMERLY_REFUSED_OUTPUT_MODES)
    def test_a_mask_is_accepted_rather_than_refused(self, output_mode, use_cls_token):
        """The combination must no longer raise, and must return a sane tensor.

        SCOPE PIN, inverted from ``test_a_mask_is_refused_rather_than_silently_leaked``:
        the old assertion was ``pytest.raises(NotImplementedError)`` at this very
        parametrization. Over-refusal is now the regression to catch.
        """
        encoder = _seeded_encoder(output_mode=output_mode, use_cls_token=use_cls_token)
        out = encoder(
            ops.convert_to_tensor(_mask_images()),
            attention_mask=ops.convert_to_tensor(_patch_mask()),
            training=False,
        )
        expected = encoder.compute_output_shape(
            (2, MASK_CFG['img_size'], MASK_CFG['img_size'], 3)
        )
        assert tuple(out.shape)[1:] == tuple(expected)[1:]
        assert np.all(np.isfinite(_np(out)))

    @pytest.mark.parametrize("use_cls_token", [True, False])
    @pytest.mark.parametrize("output_mode", FORMERLY_REFUSED_OUTPUT_MODES)
    def test_the_masked_patch_is_isolated_through_call(self, output_mode, use_cls_token):
        """SC-3: perturbing the masked patch must leave the output bit-identical.

        Carries its own live control, so a pass caused by *nothing* moving is
        impossible.
        """
        encoder = _seeded_encoder(output_mode=output_mode, use_cls_token=use_cls_token)
        images = _mask_images()
        mask = ops.convert_to_tensor(_patch_mask())

        base = _np(encoder(ops.convert_to_tensor(images), attention_mask=mask, training=False))

        live = _np(encoder(
            ops.convert_to_tensor(_perturb_patch(images, 0)),
            attention_mask=mask, training=False,
        ))
        live_delta = float(np.max(np.abs(live - base)))
        assert live_delta > 1e-2, (
            f"Vacuous probe for output_mode={output_mode!r}, use_cls_token="
            f"{use_cls_token}: the UNMASKED live control moved only "
            f"{live_delta:.6e}."
        )

        leaked = _np(encoder(
            ops.convert_to_tensor(_perturb_patch(images, MASKED_PATCH)),
            attention_mask=mask, training=False,
        ))
        np.testing.assert_allclose(
            leaked, base, rtol=0, atol=0,
            err_msg=(
                f"output_mode={output_mode!r}, use_cls_token={use_cls_token}: "
                f"the masked patch moved the pooled output by "
                f"{float(np.max(np.abs(leaked - base))):.6e}; required 0.0."
            ),
        )

    @pytest.mark.parametrize("use_cls_token", [True, False])
    @pytest.mark.parametrize("output_mode", FORMERLY_REFUSED_OUTPUT_MODES)
    def test_isolation_holds_at_the_pooling_layer_itself(self, output_mode, use_cls_token):
        """The direct replacement for ``test_the_leak_the_guard_prevents_is_real``.

        Same probe, same site, INVERTED assertion: it reaches past ``call()``
        straight into the sequence features and the pooling layer — i.e. exactly
        what ``call()`` used to refuse to do — and requires ``0.0`` movement
        where the old test required ``> 1e-3``. Keeping it distinct from the
        ``call()``-level test above matters: it proves the isolation comes from
        the POOLING fix and not merely from self-attention already excluding the
        masked patch upstream.
        """
        encoder = _seeded_encoder(output_mode=output_mode, use_cls_token=use_cls_token)
        images = _mask_images()
        mask = ops.convert_to_tensor(_patch_mask())

        def _pooled(arr: np.ndarray) -> np.ndarray:
            features = encoder._get_full_sequence_features(
                ops.convert_to_tensor(arr), attention_mask=mask, training=False
            )
            pooling_mask = encoder._extend_mask_for_cls(mask, 2)
            return _np(encoder.pooling_layer(
                features, mask=pooling_mask, training=False
            ))

        base = _pooled(images)
        live = _pooled(_perturb_patch(images, 0))
        assert float(np.max(np.abs(live - base))) > 1e-2, (
            f"Vacuous probe for output_mode={output_mode!r}: the live control "
            f"did not move at the pooling layer."
        )

        leaked = _pooled(_perturb_patch(images, MASKED_PATCH))
        delta = float(np.max(np.abs(leaked - base)))
        assert delta == 0.0, (
            f"output_mode={output_mode!r}, use_cls_token={use_cls_token}: the "
            f"masked patch leaked into the pooled output by {delta:.6e}; "
            f"F-24 has regressed in `layers/sequence_pooling/`."
        )

    def test_the_formerly_refused_modes_still_work_without_a_mask(self):
        """Retained scope pin: removing the guard must not disturb the mask-free path."""
        for output_mode in FORMERLY_REFUSED_OUTPUT_MODES:
            encoder = _seeded_encoder(output_mode=output_mode, use_cls_token=True)
            out = encoder(ops.convert_to_tensor(_mask_images()), training=False)
            explicit = encoder(
                ops.convert_to_tensor(_mask_images()),
                attention_mask=None,
                training=False,
            )
            assert np.all(np.isfinite(_np(out))), output_mode
            np.testing.assert_allclose(_np(explicit), _np(out), rtol=0, atol=0)

    def test_no_output_mode_is_refused_with_a_mask(self):
        """Scope pin: after G-02 NOTHING raises on the mask path.

        The old ``test_the_isolating_modes_are_not_refused`` proved the guard bit
        only three modes; this proves the guard is gone for all 18.
        """
        for output_mode in ALL_OUTPUT_MODES:
            encoder = _seeded_encoder(output_mode=output_mode, use_cls_token=True)
            out = encoder(
                ops.convert_to_tensor(_mask_images()),
                attention_mask=ops.convert_to_tensor(_patch_mask()),
                training=False,
            )
            assert np.all(np.isfinite(_np(out))), output_mode

    def test_helper_methods_still_accept_a_mask(self):
        """``get_*_features`` do not pool; they were never refused and still are not."""
        encoder = _seeded_encoder(output_mode='weighted', use_cls_token=True)
        images = ops.convert_to_tensor(_mask_images())
        mask = ops.convert_to_tensor(_patch_mask())

        assert np.all(np.isfinite(_np(
            encoder.get_cls_features(images, attention_mask=mask)
        )))
        assert np.all(np.isfinite(_np(
            encoder.get_patch_features(images, attention_mask=mask)
        )))
