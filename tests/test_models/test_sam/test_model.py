"""
Comprehensive Test Suite for SAM (Segment Anything Model)
=========================================================

This test suite provides thorough coverage of the SAM model, including:
- Model instantiation and variant creation
- Forward pass with different prompt types
- Serialization and deserialization
- Input validation and error handling
- Shape consistency and output validation
- Preprocessing and postprocessing
- Training mode compatibility

Run with: pytest test_sam_model.py -v
"""

import gc

import pytest
import keras
import numpy as np
import tempfile
import os
from typing import Dict, Any

# Import SAM components
from dl_techniques.models.SAM.SAM1.model import SAM
from dl_techniques.models.SAM.SAM1.image_encoder import ImageEncoderViT
from dl_techniques.models.SAM.SAM1.prompt_encoder import PromptEncoder
from dl_techniques.models.SAM.SAM1.mask_decoder import MaskDecoder
from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer


class TestSAMInstantiation:
    """Test suite for SAM model instantiation and configuration."""

    def test_from_variant_vit_b(self):
        """Test creating SAM with vit_b variant."""
        model = SAM.from_variant('vit_b')

        assert isinstance(model, SAM)
        assert isinstance(model.image_encoder, ImageEncoderViT)
        assert isinstance(model.prompt_encoder, PromptEncoder)
        assert isinstance(model.mask_decoder, MaskDecoder)

        # Check encoder configuration
        assert model.image_encoder.embed_dim == 768
        assert model.image_encoder.depth == 12
        assert model.image_encoder.num_heads == 12

        del model
        keras.backend.clear_session()
        gc.collect()

    def test_from_variant_vit_l(self):
        """Test creating SAM with vit_l variant."""
        model = SAM.from_variant('vit_l')

        assert model.image_encoder.embed_dim == 1024
        assert model.image_encoder.depth == 24
        assert model.image_encoder.num_heads == 16

        del model
        keras.backend.clear_session()
        gc.collect()

    def test_from_variant_vit_h(self):
        """Test creating SAM with vit_h variant."""
        model = SAM.from_variant('vit_h')

        assert model.image_encoder.embed_dim == 1280
        assert model.image_encoder.depth == 32
        assert model.image_encoder.num_heads == 16

        del model
        keras.backend.clear_session()
        gc.collect()

    def test_from_variant_invalid(self):
        """Test that invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            SAM.from_variant('invalid_variant')

    def test_from_variant_with_custom_params(self):
        """Test creating SAM with custom parameters."""
        model = SAM.from_variant(
            'vit_b',
            mask_threshold=0.5,
            pixel_mean=[120.0, 115.0, 100.0],
            pixel_std=[60.0, 60.0, 60.0]
        )

        assert model.mask_threshold == 0.5
        assert np.allclose(
            keras.ops.convert_to_numpy(model.pixel_mean),
            [120.0, 115.0, 100.0]
        )

        del model
        keras.backend.clear_session()
        gc.collect()

    def test_direct_instantiation(self):
        """Test direct instantiation with custom components."""
        # Create minimal components
        image_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=16,
            embed_dim=256,
            depth=2,
            num_heads=4,
            out_chans=128,
            global_attn_indexes=(1,),
        )

        prompt_encoder = PromptEncoder(
            embed_dim=128,
            image_embedding_size=(16, 16),
            input_image_size=(256, 256),
            mask_in_chans=16
        )

        transformer = TwoWayTransformer(
            depth=1,
            embedding_dim=128,
            num_heads=4,
            mlp_dim=256
        )

        mask_decoder = MaskDecoder(
            transformer_dim=128,
            transformer=transformer,
            num_multimask_outputs=2
        )

        model = SAM(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder
        )

        assert isinstance(model, SAM)

    def test_invalid_pixel_mean_length(self):
        """Test that invalid pixel_mean length raises error."""
        image_encoder = ImageEncoderViT(
            img_size=256, patch_size=16, embed_dim=256, depth=2, num_heads=4,
            global_attn_indexes=(1,),
        )
        prompt_encoder = PromptEncoder(
            embed_dim=256, image_embedding_size=(16, 16),
            input_image_size=(256, 256)
        )
        transformer = TwoWayTransformer(depth=1, embedding_dim=256, num_heads=4)
        mask_decoder = MaskDecoder(transformer_dim=256, transformer=transformer)

        with pytest.raises(ValueError, match="pixel_mean must have 3 values"):
            SAM(
                image_encoder=image_encoder,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                pixel_mean=[123.0, 116.0]  # Only 2 values
            )

    def test_invalid_image_format(self):
        """Test that invalid image format raises error."""
        image_encoder = ImageEncoderViT(
            img_size=256, patch_size=16, embed_dim=256, depth=2, num_heads=4,
            global_attn_indexes=(1,),
        )
        prompt_encoder = PromptEncoder(
            embed_dim=256, image_embedding_size=(16, 16),
            input_image_size=(256, 256)
        )
        transformer = TwoWayTransformer(depth=1, embedding_dim=256, num_heads=4)
        mask_decoder = MaskDecoder(transformer_dim=256, transformer=transformer)

        with pytest.raises(ValueError, match="Only 'RGB' image format is supported"):
            SAM(
                image_encoder=image_encoder,
                prompt_encoder=prompt_encoder,
                mask_decoder=mask_decoder,
                image_format='BGR'
            )


class TestSAMForwardPass:
    """Test suite for SAM forward pass with different prompt types."""

    @pytest.fixture
    def small_model(self):
        """Create a small SAM model for testing."""
        image_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=16,
            embed_dim=128,
            depth=2,
            num_heads=4,
            out_chans=64,
            global_attn_indexes=(1,),
        )

        prompt_encoder = PromptEncoder(
            embed_dim=64,
            image_embedding_size=(16, 16),
            input_image_size=(256, 256),
            mask_in_chans=8
        )

        transformer = TwoWayTransformer(
            depth=1,
            embedding_dim=64,
            num_heads=4,
            mlp_dim=128
        )

        mask_decoder = MaskDecoder(
            transformer_dim=64,
            transformer=transformer,
            num_multimask_outputs=3
        )

        model = SAM(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder
        )
        yield model
        del model
        keras.backend.clear_session()
        gc.collect()

    def test_forward_with_points(self, small_model):
        """Test forward pass with point prompts."""
        image = keras.random.normal(shape=(2, 256, 256, 3))
        points_coords = keras.ops.convert_to_tensor([
            [[100.0, 100.0], [150.0, 150.0]],
            [[120.0, 120.0], [180.0, 180.0]]
        ])
        points_labels = keras.ops.convert_to_tensor([[1, 0], [1, 1]])

        outputs = small_model({
            'image': image,
            'points': (points_coords, points_labels),
            'original_size': keras.ops.convert_to_tensor((256, 256))
        })

        assert 'masks' in outputs
        assert 'iou_predictions' in outputs
        assert 'low_res_logits' in outputs

        # Check shapes
        assert keras.ops.shape(outputs['masks'])[0] == 2  # batch size
        assert keras.ops.shape(outputs['masks'])[2] == 256  # height
        assert keras.ops.shape(outputs['masks'])[3] == 256  # width

        # Check masks are binary
        masks_tensor = outputs['masks']
        is_binary = keras.ops.logical_or(masks_tensor == 0, masks_tensor == 1)
        assert keras.ops.all(is_binary)

    def test_forward_with_boxes(self, small_model):
        """Test forward pass with box prompts."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        boxes = keras.ops.convert_to_tensor([[[50.0, 50.0, 200.0, 200.0]]])

        outputs = small_model({
            'image': image,
            'boxes': boxes,
            'original_size': keras.ops.convert_to_tensor((256, 256))
        })

        assert 'masks' in outputs
        assert 'iou_predictions' in outputs
        assert keras.ops.shape(outputs['masks'])[0] == 1

    def test_forward_with_masks(self, small_model):
        """Test forward pass with mask prompts."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        mask_prompt = keras.random.normal(shape=(1, 1, 64, 64))

        outputs = small_model({
            'image': image,
            'masks': mask_prompt,
            'original_size': keras.ops.convert_to_tensor((256, 256))
        })

        assert 'masks' in outputs
        assert keras.ops.shape(outputs['masks'])[0] == 1

    def test_forward_with_combined_prompts(self, small_model):
        """Test forward pass with multiple prompt types."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([[[100.0, 100.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )
        boxes = keras.ops.convert_to_tensor([[[50.0, 50.0, 200.0, 200.0]]])

        outputs = small_model({
            'image': image,
            'points': points,
            'boxes': boxes,
            'original_size': keras.ops.convert_to_tensor((256, 256))
        })

        assert 'masks' in outputs
        assert 'iou_predictions' in outputs

    def test_forward_multimask_output(self, small_model):
        """Test forward pass with multimask_output=True."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([[[100.0, 100.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )

        outputs = small_model(
            {'image': image, 'points': points, 'original_size': keras.ops.convert_to_tensor((256, 256))},
            multimask_output=True
        )

        # Should return 3 masks (num_multimask_outputs)
        assert keras.ops.shape(outputs['masks'])[1] == 3
        assert keras.ops.shape(outputs['iou_predictions'])[1] == 3

    def test_forward_single_mask_output(self, small_model):
        """Test forward pass with multimask_output=False."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([[[100.0, 100.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )

        outputs = small_model(
            {'image': image, 'points': points, 'original_size': keras.ops.convert_to_tensor((256, 256))},
            multimask_output=False
        )

        # Should return 1 mask
        assert keras.ops.shape(outputs['masks'])[1] == 1
        assert keras.ops.shape(outputs['iou_predictions'])[1] == 1

    def test_forward_missing_image(self, small_model):
        """Test that missing image raises error."""
        with pytest.raises(ValueError, match="must contain 'image' key"):
            small_model({'original_size': keras.ops.convert_to_tensor((256, 256))})

    def test_forward_missing_original_size(self, small_model):
        """Test that missing original_size raises error."""
        image = keras.random.normal(shape=(1, 256, 256, 3))

        with pytest.raises(ValueError, match="must contain 'original_size' key"):
            small_model({'image': image})

    def test_forward_training_mode(self, small_model):
        """Test forward pass in training mode."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([[[100.0, 100.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )

        outputs_train = small_model(
            {'image': image, 'points': points, 'original_size': keras.ops.convert_to_tensor((256, 256))},
            training=True
        )

        outputs_eval = small_model(
            {'image': image, 'points': points, 'original_size': keras.ops.convert_to_tensor((256, 256))},
            training=False
        )

        assert 'masks' in outputs_train
        assert 'masks' in outputs_eval


class TestSAMSerialization:
    """Test suite for SAM model serialization and deserialization."""

    @pytest.fixture
    def small_model(self):
        """Create a small SAM model for testing."""
        image_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=16,
            embed_dim=128,
            depth=2,
            num_heads=4,
            out_chans=64,
            global_attn_indexes=(1,),
        )

        prompt_encoder = PromptEncoder(
            embed_dim=64,
            image_embedding_size=(16, 16),
            input_image_size=(256, 256),
            mask_in_chans=8
        )

        transformer = TwoWayTransformer(
            depth=1,
            embedding_dim=64,
            num_heads=4,
            mlp_dim=128
        )

        mask_decoder = MaskDecoder(
            transformer_dim=64,
            transformer=transformer,
            num_multimask_outputs=3
        )

        model = SAM(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder
        )
        yield model
        del model
        keras.backend.clear_session()
        gc.collect()

    def test_get_config(self, small_model):
        """Test that get_config returns complete configuration."""
        config = small_model.get_config()

        assert 'image_encoder' in config
        assert 'prompt_encoder' in config
        assert 'mask_decoder' in config
        assert 'pixel_mean' in config
        assert 'pixel_std' in config
        assert 'mask_threshold' in config
        assert 'image_format' in config

        # Check pixel_mean is a list
        assert isinstance(config['pixel_mean'], list)
        assert len(config['pixel_mean']) == 3

    def test_from_config(self, small_model):
        """Test reconstruction from config."""
        config = small_model.get_config()
        reconstructed = SAM.from_config(config)

        assert isinstance(reconstructed, SAM)
        assert isinstance(reconstructed.image_encoder, ImageEncoderViT)
        assert isinstance(reconstructed.prompt_encoder, PromptEncoder)
        assert isinstance(reconstructed.mask_decoder, MaskDecoder)

    def test_save_and_load(self, small_model):
        """Save a **BUILT** model and prove the round-trip on VALUES.

        Until iteration 1 / step 12 this test saved an *unbuilt* model — Keras
        emitted "You are saving a model that has not yet been built", the
        archive stored no weights, and the only assertion was `isinstance`.
        That combination passes even when every weight is re-initialized on
        load, which is exactly the failure mode a `.keras` layout change
        produces. Build first, then compare `low_res_logits` values.
        """
        inputs = {
            'image': keras.random.normal(shape=(1, 256, 256, 3), seed=17),
            'points': (
                keras.ops.convert_to_tensor([[[100.0, 100.0]]]),
                keras.ops.convert_to_tensor([[1]])
            ),
            'original_size': keras.ops.convert_to_tensor((256, 256)),
        }

        # BUILD before saving. Without this the archive carries no weights.
        reference = keras.ops.convert_to_numpy(
            small_model(inputs, multimask_output=True)['low_res_logits']
        )
        n_weights = len(small_model.weights)
        assert n_weights > 0, (
            "the fixture must be BUILT before saving; an unbuilt save stores "
            "no weights and makes the comparison below meaningless"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'sam_model.keras')

            small_model.save(model_path)
            assert os.path.exists(model_path)

            loaded_model = keras.models.load_model(model_path)
            assert isinstance(loaded_model, SAM)
            assert len(loaded_model.weights) == n_weights, (
                f"weight-list length moved on load: {n_weights} -> "
                f"{len(loaded_model.weights)}"
            )

            restored = keras.ops.convert_to_numpy(
                loaded_model(inputs, multimask_output=True)['low_res_logits']
            )

        max_abs_diff = float(np.max(np.abs(reference - restored)))
        assert max_abs_diff == 0.0, (
            f"low_res_logits moved across the .keras round-trip: max abs diff "
            f"{max_abs_diff!r}. An `isinstance` assertion cannot see this."
        )

    def test_output_consistency_after_loading(self, small_model):
        """Test that loaded model produces identical outputs.

        Compares `low_res_logits`, NOT the binarized `masks`. The uint8
        comparison this test used to make is quantifiably blind: it saturates
        at an absolute difference of 1, and it is entirely insensitive to any
        logit change that does not move a pixel across `mask_threshold` —
        scaling every logit by a positive constant leaves it bit-identical.
        """
        image = keras.random.normal(shape=(1, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([[[100.0, 100.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )
        inputs = {'image': image, 'points': points, 'original_size': keras.ops.convert_to_tensor((256, 256))}

        # Get outputs from original model
        outputs_original = small_model(inputs, multimask_output=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'sam_model.keras')
            small_model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            # Get outputs from loaded model
            outputs_loaded = loaded_model(inputs, multimask_output=True)

            # Compare outputs on the differentiable float logits.
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(outputs_original['low_res_logits']),
                keras.ops.convert_to_numpy(outputs_loaded['low_res_logits']),
                rtol=1e-5, atol=1e-5,
                err_msg="low_res_logits should match after loading"
            )

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(outputs_original['iou_predictions']),
                keras.ops.convert_to_numpy(outputs_loaded['iou_predictions']),
                rtol=1e-5, atol=1e-5,
                err_msg="IoU predictions should match after loading"
            )


class TestSAMPreprocessing:
    """Test suite for SAM preprocessing functionality."""

    @pytest.fixture
    def model(self):
        """Create a small SAM model for testing."""
        image_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=16,
            embed_dim=64,
            depth=1,
            num_heads=4,
            out_chans=32,
            global_attn_indexes=(0,),
        )
        prompt_encoder = PromptEncoder(
            embed_dim=32,
            image_embedding_size=(16, 16),
            input_image_size=(256, 256)
        )
        transformer = TwoWayTransformer(
            depth=1, embedding_dim=32, num_heads=4, mlp_dim=64
        )
        mask_decoder = MaskDecoder(transformer_dim=32, transformer=transformer)

        model = SAM(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder
        )
        yield model
        del model
        keras.backend.clear_session()
        gc.collect()

    def test_preprocess_normalization(self, model):
        """Test that preprocessing normalizes images correctly."""
        image = keras.ops.ones((1, 128, 128, 3)) * 128.0
        preprocessed = model.preprocess(image)

        # Check shape (should be padded to encoder size)
        assert keras.ops.shape(preprocessed)[1] == model.image_encoder.img_size
        assert keras.ops.shape(preprocessed)[2] == model.image_encoder.img_size

        # Check normalization (approximately zero mean after normalization)
        mean_val = keras.ops.mean(preprocessed)
        assert abs(keras.ops.convert_to_numpy(mean_val)) < 1.0

    def test_preprocess_padding(self, model):
        """Test that preprocessing pads images correctly."""
        # Test various sizes
        for h, w in [(100, 100), (200, 150), (256, 128)]:
            image = keras.random.normal(shape=(1, h, w, 3))
            preprocessed = model.preprocess(image)

            # Should always pad to encoder size
            assert keras.ops.shape(preprocessed)[1] == model.image_encoder.img_size
            assert keras.ops.shape(preprocessed)[2] == model.image_encoder.img_size

    def test_preprocess_no_padding_needed(self, model):
        """Test preprocessing when image already matches encoder size."""
        image = keras.random.normal(
            shape=(1, model.image_encoder.img_size, model.image_encoder.img_size, 3)
        )
        preprocessed = model.preprocess(image)

        assert keras.ops.shape(preprocessed)[1] == model.image_encoder.img_size
        assert keras.ops.shape(preprocessed)[2] == model.image_encoder.img_size


class TestSAMPostprocessing:
    """Test suite for SAM postprocessing functionality."""

    @pytest.fixture
    def model(self):
        """Create a small SAM model for testing."""
        image_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=16,
            embed_dim=64,
            depth=1,
            num_heads=4,
            out_chans=32,
            global_attn_indexes=(0,),
        )
        prompt_encoder = PromptEncoder(
            embed_dim=32,
            image_embedding_size=(16, 16),
            input_image_size=(256, 256)
        )
        transformer = TwoWayTransformer(
            depth=1, embedding_dim=32, num_heads=4, mlp_dim=64
        )
        mask_decoder = MaskDecoder(transformer_dim=32, transformer=transformer)

        model = SAM(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder
        )
        yield model
        del model
        keras.backend.clear_session()
        gc.collect()

    def test_postprocess_upscaling(self, model):
        """Test that postprocessing upscales masks correctly."""
        low_res_masks = keras.random.normal(shape=(1, 3, 64, 64))
        input_size = (200, 200)
        original_size = (400, 400)

        upscaled = model.postprocess_masks(low_res_masks, input_size, original_size)

        # Should match original size
        assert keras.ops.shape(upscaled)[2] == original_size[0]
        assert keras.ops.shape(upscaled)[3] == original_size[1]

    def test_postprocess_cropping(self, model):
        """Test that postprocessing handles padding removal."""
        low_res_masks = keras.random.normal(shape=(1, 2, 32, 32))
        input_size = (200, 200)  # Original input before padding
        original_size = (200, 200)  # No resize needed

        processed = model.postprocess_masks(low_res_masks, input_size, original_size)

        assert keras.ops.shape(processed)[2] == original_size[0]
        assert keras.ops.shape(processed)[3] == original_size[1]

    def test_postprocess_different_aspect_ratios(self, model):
        """Test postprocessing with different aspect ratios."""
        low_res_masks = keras.random.normal(shape=(1, 1, 32, 32))

        # Test various aspect ratios
        test_cases = [
            ((256, 128), (512, 256)),  # 2:1 ratio
            ((128, 256), (256, 512)),  # 1:2 ratio
            ((200, 200), (400, 400)),  # 1:1 ratio
        ]

        for input_size, original_size in test_cases:
            processed = model.postprocess_masks(low_res_masks, input_size, original_size)
            assert keras.ops.shape(processed)[2] == original_size[0]
            assert keras.ops.shape(processed)[3] == original_size[1]


class TestSAMShapeConsistency:
    """Test suite for shape consistency throughout the pipeline."""

    @pytest.fixture
    def model(self):
        """Create a small SAM model for testing."""
        image_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=16,
            embed_dim=128,
            depth=2,
            num_heads=4,
            out_chans=64,
            global_attn_indexes=(1,),
        )
        prompt_encoder = PromptEncoder(
            embed_dim=64,
            image_embedding_size=(16, 16),
            input_image_size=(256, 256)
        )
        transformer = TwoWayTransformer(
            depth=1, embedding_dim=64, num_heads=4, mlp_dim=128
        )
        mask_decoder = MaskDecoder(transformer_dim=64, transformer=transformer)

        model = SAM(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder
        )
        yield model
        del model
        keras.backend.clear_session()
        gc.collect()

    def test_batch_size_consistency(self, model):
        """Test that batch size is preserved throughout."""
        for batch_size in [1, 2, 4]:
            image = keras.random.normal(shape=(batch_size, 256, 256, 3))
            points = (
                keras.ops.convert_to_tensor(
                    np.random.rand(batch_size, 2, 2).astype(np.float32) * 256
                ),
                keras.ops.convert_to_tensor(np.ones((batch_size, 2), dtype=np.int32))
            )

            outputs = model({
                'image': image,
                'points': points,
                'original_size': keras.ops.convert_to_tensor((256, 256))
            })

            assert keras.ops.shape(outputs['masks'])[0] == batch_size
            assert keras.ops.shape(outputs['iou_predictions'])[0] == batch_size
            assert keras.ops.shape(outputs['low_res_logits'])[0] == batch_size

    def test_output_shape_matches_original_size(self, model):
        """Test that output mask size matches requested original size."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([[[100.0, 100.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )

        # Test various output sizes
        for original_size in [(256, 256), (512, 512), (384, 384)]:
            outputs = model({
                'image': image,
                'points': points,
                'original_size': keras.ops.convert_to_tensor(original_size)
            })

            assert keras.ops.shape(outputs['masks'])[2] == original_size[0]
            assert keras.ops.shape(outputs['masks'])[3] == original_size[1]

    def test_iou_predictions_shape(self, model):
        """Test that IoU predictions have correct shape."""
        image = keras.random.normal(shape=(2, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([[[100.0, 100.0]], [[150.0, 150.0]]]),
            keras.ops.convert_to_tensor([[1], [0]])
        )

        # Multimask output
        outputs_multi = model(
            {'image': image, 'points': points, 'original_size': keras.ops.convert_to_tensor((256, 256))},
            multimask_output=True
        )
        assert keras.ops.shape(outputs_multi['iou_predictions'])[0] == 2  # batch
        assert keras.ops.shape(outputs_multi['iou_predictions'])[1] == 3  # num masks

        # Single mask output
        outputs_single = model(
            {'image': image, 'points': points, 'original_size': keras.ops.convert_to_tensor((256, 256))},
            multimask_output=False
        )
        assert keras.ops.shape(outputs_single['iou_predictions'])[0] == 2  # batch
        assert keras.ops.shape(outputs_single['iou_predictions'])[1] == 1  # num masks


class TestSAMEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.fixture
    def model(self):
        """Create a small SAM model for testing."""
        image_encoder = ImageEncoderViT(
            img_size=256,
            patch_size=16,
            embed_dim=64,
            depth=1,
            num_heads=4,
            out_chans=32,
            global_attn_indexes=(0,),
        )
        prompt_encoder = PromptEncoder(
            embed_dim=32,
            image_embedding_size=(16, 16),
            input_image_size=(256, 256)
        )
        transformer = TwoWayTransformer(
            depth=1, embedding_dim=32, num_heads=4, mlp_dim=64
        )
        mask_decoder = MaskDecoder(transformer_dim=32, transformer=transformer)

        model = SAM(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder
        )
        yield model
        del model
        keras.backend.clear_session()
        gc.collect()

    def test_single_point(self, model):
        """Test with a single point prompt."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([[[128.0, 128.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )

        outputs = model({
            'image': image,
            'points': points,
            'original_size': keras.ops.convert_to_tensor((256, 256))
        })

        assert 'masks' in outputs

    def test_many_points(self, model):
        """Test with many point prompts."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        num_points = 10
        points = (
            keras.ops.convert_to_tensor(
                np.random.rand(1, num_points, 2).astype(np.float32) * 256
            ),
            keras.ops.convert_to_tensor(np.ones((1, num_points), dtype=np.int32))
        )

        outputs = model({
            'image': image,
            'points': points,
            'original_size': keras.ops.convert_to_tensor((256, 256))
        })

        assert 'masks' in outputs

    def test_extreme_coordinates(self, model):
        """Test with points at image boundaries."""
        image = keras.random.normal(shape=(1, 256, 256, 3))
        points = (
            keras.ops.convert_to_tensor([
                [[0.0, 0.0], [255.0, 255.0], [0.0, 255.0], [255.0, 0.0]]
            ]),
            keras.ops.convert_to_tensor([[1, 1, 0, 0]])
        )

        outputs = model({
            'image': image,
            'points': points,
            'original_size': keras.ops.convert_to_tensor((256, 256))
        })

        assert 'masks' in outputs

    def test_different_image_sizes(self, model):
        """Test with various input image sizes."""
        points = (
            keras.ops.convert_to_tensor([[[100.0, 100.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )

        # Test various sizes smaller than encoder size
        for h, w in [(128, 128), (200, 200), (256, 128), (128, 256)]:
            image = keras.random.normal(shape=(1, h, w, 3))

            outputs = model({
                'image': image,
                'points': points,
                'original_size': keras.ops.convert_to_tensor((h, w))
            })

            # Output should match original size
            assert keras.ops.shape(outputs['masks'])[2] == h
            assert keras.ops.shape(outputs['masks'])[3] == w

    def test_mask_threshold_effect(self, model):
        """`mask_threshold` must actually MOVE the binary mask.

        The previous body asserted only that both outputs were binary, which
        is tautologically true of any `cast(x > t, 'uint8')` for every `t` —
        including a hardcoded one that ignores `mask_threshold` entirely. It
        never checked that the two outputs DIFFER. The thresholds below are
        derived from the model's own measured logit range, so the test does
        not depend on where random weights happen to put the logits.
        """
        image = keras.random.normal(shape=(1, 256, 256, 3), seed=23)
        points = (
            keras.ops.convert_to_tensor([[[128.0, 128.0]]]),
            keras.ops.convert_to_tensor([[1]])
        )
        inputs = {'image': image, 'points': points, 'original_size': keras.ops.convert_to_tensor((256, 256))}

        # Read the full-resolution float logits the threshold is applied to.
        model.binarize_masks = False
        logits = keras.ops.convert_to_numpy(model(inputs)['masks'])
        model.binarize_masks = True

        lo, hi = float(logits.min()), float(logits.max())
        assert lo < hi, (
            f"degenerate probe: the mask logits are constant at {lo!r}, so no "
            f"threshold could distinguish anything"
        )

        # Below every logit -> all ones. Above every logit -> all zeros.
        model.mask_threshold = lo - 1.0
        masks_all_on = keras.ops.convert_to_numpy(model(inputs)['masks'])
        model.mask_threshold = hi + 1.0
        masks_all_off = keras.ops.convert_to_numpy(model(inputs)['masks'])

        # Still binary (the original, weak property — kept as a regression pin).
        for m in (masks_all_on, masks_all_off):
            assert np.all((m == 0) | (m == 1))

        # ...and the threshold is LIVE: every pixel flips between the two.
        assert masks_all_on.min() == 1, (
            "a threshold below every logit must mark every pixel foreground; "
            "got some zeros, so mask_threshold is not being applied"
        )
        assert masks_all_off.max() == 0, (
            "a threshold above every logit must mark every pixel background; "
            "got some ones, so mask_threshold is not being applied"
        )
        n_diff = int(np.count_nonzero(masks_all_on != masks_all_off))
        assert n_diff == masks_all_on.size, (
            f"only {n_diff} of {masks_all_on.size} pixels responded to the "
            f"threshold change"
        )

        # An intermediate threshold must binarize at exactly that boundary.
        mid = 0.5 * (lo + hi)
        model.mask_threshold = mid
        masks_mid = keras.ops.convert_to_numpy(model(inputs)['masks'])
        np.testing.assert_array_equal(
            masks_mid, (logits > mid).astype(masks_mid.dtype),
            err_msg=(
                "the binary mask is not `low-res-upscaled logits > "
                "mask_threshold`; the threshold is being ignored or applied "
                "to a different tensor"
            )
        )


# Run tests with: pytest test_sam_model.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])