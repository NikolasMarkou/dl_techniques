"""
Masked Autoencoder (MAE) Framework
===================================

A flexible framework for training masked autoencoders with any encoder architecture.
Supports patch-based masking and works seamlessly with ConvNeXt V2, ViT, and ResNet.

Key Features:
- Asymmetric Encoder-Decoder architecture
- Efficient masked training (loss computed only on masked patches)
- Mixed Precision compatible (explicit casting in loss)
- **Deep Supervision Support**: Handles encoders returning lists of outputs.
"""

import keras
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .conv_decoder import ConvDecoder
from .patch_masking import PatchMasking

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MaskedAutoencoder(keras.Model):
    """Masked Autoencoder (MAE) model.

    A self-supervised learning model that masks random patches of input images
    and trains to reconstruct them.

    Args:
        encoder: A `keras.Model` instance (feature extractor).
        patch_size: Integer, size of patches for masking. Defaults to 16.
        mask_ratio: Float, ratio of patches to mask (0 to 1). Defaults to 0.75.
        decoder_dims: List of integers, decoder layer dimensions.
        decoder_depth: Integer, number of decoder layers (if decoder_dims=None).
        norm_pix_loss: Boolean, whether to normalize pixel values for loss.
        mask_value: "learnable", "zero", "noise", or float.
        input_shape: Tuple, input image shape (H, W, C).
        non_mask_value: Float, small value weight to apply to non-masked patches
    """

    def __init__(
        self,
        encoder: keras.Model,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_dims: Optional[List[int]] = None,
        decoder_depth: int = 4,
        norm_pix_loss: bool = False,
        mask_value: Union[str, float] = "learnable",
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        non_mask_value: float = 0.0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(encoder, keras.Model):
            raise TypeError("encoder must be a keras.Model instance.")

        self.encoder = encoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.decoder_dims = decoder_dims
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.mask_value = mask_value
        self.input_shape_config = input_shape
        self.non_mask_value = non_mask_value

        # Ensure encoder is built to determine shapes
        if not self.encoder.built:
            self.encoder.build((None,) + input_shape)

        # Compute output shape to determine channels
        encoder_output_shape = self.encoder.compute_output_shape((None,) + input_shape)

        # Handle Deep Supervision (List of shapes)
        # We assume the first output is the main feature map for the default decoder
        if isinstance(encoder_output_shape, list):
            main_shape = encoder_output_shape[0]
        else:
            main_shape = encoder_output_shape

        if len(main_shape) != 4:
             # Fallback for flattened outputs, though ConvDecoder expects 4D
            raise ValueError(
                f"Encoder main output must be 4D tensor (B, H, W, C). "
                f"Got: {main_shape} (Full output: {encoder_output_shape})"
            )

        self.encoder_channels = main_shape[-1]

        # 1. Masking Layer
        self.masking = PatchMasking(
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            name="patch_masking"
        )

        # 2. Decoder
        self.decoder = self._create_decoder()

        # Metrics
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")

    def _create_decoder(self) -> ConvDecoder:
        """Create or configure the decoder."""
        if self.decoder_dims is None:
            # Auto-configure decoder dimensions: gradually reduce from encoder dim
            decoder_dims = []
            current_dim = self.encoder_channels
            for _ in range(self.decoder_depth):
                current_dim = max(current_dim // 2, 64)
                decoder_dims.append(current_dim)
            self.decoder_dims = decoder_dims

        return ConvDecoder(
            decoder_dims=self.decoder_dims,
            output_channels=self.input_shape_config[-1],
            name="conv_decoder"
        )

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute output shapes for all model outputs."""
        batch_size = input_shape[0]
        height, width, channels = input_shape[1:]

        # Handle dynamic shapes if possible
        if height is None: height = self.input_shape_config[0]
        if width is None: width = self.input_shape_config[1]

        num_patches = (height // self.patch_size) * (width // self.patch_size)

        return {
            "reconstruction": (batch_size, height, width, channels),
            "mask": (batch_size, num_patches),
            "masked_input": (batch_size, height, width, channels),
            "encoded": self.encoder.compute_output_shape(input_shape)
        }

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass: Mask -> Encode -> Decode."""
        # Masking
        masked_images, mask, _ = self.masking(inputs, training=training)

        # CRITICAL FIX for Mixed Precision:
        # PatchMasking often returns float32 (due to scatter ops).
        # We must cast it to float16 if the global policy mandates it,
        # otherwise it propagates float32 into the encoder, causing mismatch errors.
        policy = keras.mixed_precision.dtype_policy()
        if getattr(policy, "name", "") == "mixed_float16":
             masked_images = keras.ops.cast(masked_images, "float16")
        elif self.compute_dtype:
             masked_images = keras.ops.cast(masked_images, self.compute_dtype)

        # Encoding
        encoded = self.encoder(masked_images, training=training)

        # Decoding
        reconstruction = self.decoder(encoded, training=training)

        return {
            "reconstruction": reconstruction,
            "mask": mask,
            "masked_input": masked_images,
            "encoded": encoded
        }

    def compute_loss(
        self,
        x: keras.KerasTensor,
        y: Optional[keras.KerasTensor] = None,
        y_pred: Optional[Dict[str, keras.KerasTensor]] = None,
        sample_weight: Optional[keras.KerasTensor] = None,
        **kwargs: Any
    ) -> keras.KerasTensor:
        """Compute reconstruction loss only on masked patches."""
        if y_pred is None:
            return keras.ops.convert_to_tensor(0.0)

        # CRITICAL: Cast inputs and outputs to float32 for loss stability
        # This resolves Mixed Precision mismatches (float16 output vs float32 input)
        target = keras.ops.cast(x, "float32")
        reconstruction = keras.ops.cast(y_pred["reconstruction"], "float32")
        mask = keras.ops.cast(y_pred["mask"], "float32")

        # Pixel Normalization (optional per-patch normalization for loss targets)
        if self.norm_pix_loss:
            target_patches = self._extract_patches_for_loss(target)
            mean = keras.ops.mean(target_patches, axis=-1, keepdims=True)
            var = keras.ops.var(target_patches, axis=-1, keepdims=True)
            target_normalized = (target_patches - mean) / keras.ops.sqrt(var + 1e-6)
            target = self._reconstruct_patches_for_loss(target_normalized)

        # MSE Loss
        loss = keras.ops.square(target - reconstruction)
        loss = keras.ops.mean(loss, axis=-1)  # [batch, H, W]

        # Reshape mask to match spatial dimensions
        mask_img = self._reshape_mask_for_loss(mask, target)
        mask_img = keras.ops.maximum(mask_img, self.non_mask_value)

        # Apply mask: Loss = 0 for unmasked pixels
        loss = loss * mask_img

        # Normalize by number of masked elements
        num_masked = keras.ops.sum(mask, axis=-1) + 1e-6  # [batch]

        # Sum over spatial dims, then divide by num_masked patches * patch_pixels
        # Note: mask_img is 1s and 0s.
        loss_sum = keras.ops.sum(loss, axis=[1, 2]) # [batch]

        # Adjust denominator: num_masked is patches, we need pixels
        pixels_per_patch = self.patch_size * self.patch_size
        loss = loss_sum / (num_masked * pixels_per_patch)

        return keras.ops.mean(loss) # Global mean

    def _extract_patches_for_loss(self, images: keras.KerasTensor) -> keras.KerasTensor:
        """Helper to extract patches for pixel normalization."""
        # Implementation assumes fixed patch size logic
        B = keras.ops.shape(images)[0]
        H, W, C = self.input_shape_config
        P = self.patch_size

        # [B, H//P, P, W//P, P, C]
        patches = keras.ops.reshape(images, (B, H // P, P, W // P, P, C))
        # [B, H//P, W//P, P, P, C] -> [B, N_patches, P*P*C]
        patches = keras.ops.transpose(patches, (0, 1, 3, 2, 4, 5))
        return keras.ops.reshape(patches, (B, -1, P * P * C))

    def _reconstruct_patches_for_loss(self, patches: keras.KerasTensor) -> keras.KerasTensor:
        """Helper to reverse patch extraction."""
        B = keras.ops.shape(patches)[0]
        H, W, C = self.input_shape_config
        P = self.patch_size

        # [B, H//P, W//P, P, P, C]
        patches = keras.ops.reshape(patches, (B, H//P, W//P, P, P, C))
        patches = keras.ops.transpose(patches, (0, 1, 3, 2, 4, 5))
        return keras.ops.reshape(patches, (B, H, W, C))

    def _reshape_mask_for_loss(self, mask: keras.KerasTensor, target: keras.KerasTensor) -> keras.KerasTensor:
        """Expands (B, NumPatches) mask to (B, H, W)."""
        B = keras.ops.shape(mask)[0]
        H, W = keras.ops.shape(target)[1], keras.ops.shape(target)[2]
        P = self.patch_size

        # [B, H//P, W//P]
        mask_grid = keras.ops.reshape(mask, (B, H // P, W // P))

        # Upsample nearest neighbor to patch size
        mask_img = keras.ops.repeat(mask_grid, P, axis=1)
        mask_img = keras.ops.repeat(mask_img, P, axis=2)

        return mask_img

    def train_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, float]:
        import tensorflow as tf  # Backend-specific optimization import

        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=None, y_pred=y_pred)

        # Gradient Application
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.reconstruction_loss_tracker.update_state(loss)
        return {
            "loss": loss,
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }

    def test_step(self, data: Union[keras.KerasTensor, Tuple]) -> Dict[str, float]:
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        # Always mask during validation to calculate reconstruction loss
        y_pred = self(x, training=True)
        loss = self.compute_loss(x=x, y=None, y_pred=y_pred)

        self.reconstruction_loss_tracker.update_state(loss)
        return {
            "loss": loss,
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        return [self.reconstruction_loss_tracker]

    def visualize(self, image: np.ndarray, return_arrays: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Visualize a single image reconstruction."""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        outputs = self(image, training=True)
        masked = outputs["masked_input"]
        reconstructed = outputs["reconstruction"]

        if return_arrays:
            image = keras.ops.convert_to_numpy(image[0])
            masked = keras.ops.convert_to_numpy(masked[0])
            reconstructed = keras.ops.convert_to_numpy(reconstructed[0])

        return image, masked, reconstructed

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "decoder_dims": self.decoder_dims,
            "decoder_depth": self.decoder_depth,
            "norm_pix_loss": self.norm_pix_loss,
            "mask_value": self.mask_value,
            "input_shape": self.input_shape_config,
            "non_mask_value": self.non_mask_value
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskedAutoencoder":
        # Deserialization ensures nested model is restored
        config["encoder"] = keras.saving.deserialize_keras_object(config["encoder"])
        return cls(**config)