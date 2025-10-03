"""
Masked Autoencoder (MAE) Framework
===================================

A flexible framework for training masked autoencoders with any encoder architecture.
Supports patch-based masking, various decoder types, and works seamlessly with
ConvNeXt V2 and other architectures.

Based on: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
https://arxiv.org/abs/2111.06377

And: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (Woo et al., 2023)
https://arxiv.org/abs/2301.00808

Key Features:
------------
- Random patch masking with configurable ratio
- Support for any encoder architecture (ConvNeXt, ResNet, ViT, etc.)
- Multiple decoder architectures (Conv-based, Transformer-based)
- Efficient training with loss computed only on masked patches
- Visualization utilities for qualitative assessment

Usage Examples:
--------------
```python
# Create MAE with ConvNeXt V2 encoder
encoder = ConvNeXtV2.from_variant("tiny", include_top=False, input_shape=(224, 224, 3))
mae = MaskedAutoencoder(
    encoder=encoder,
    patch_size=16,
    mask_ratio=0.75,
    decoder_dims=[512, 256, 128, 64],
    input_shape=(224, 224, 3)
)

# Train the model
mae.compile(optimizer="adam")
mae.fit(train_dataset, epochs=100)

# Visualize reconstructions
original, masked, reconstructed = mae.visualize(test_images[0])
```
"""

import keras
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .patch_masking import PatchMasking
from .conv_decoder import ConvDecoder

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MaskedAutoencoder(keras.Model):
    """Masked Autoencoder (MAE) model.

    A self-supervised learning model that masks random patches of input images
    and trains to reconstruct them. The encoder processes masked images and
    the decoder reconstructs the original image.

    **Architecture**:
    ```
    Input(shape=[batch, H, W, C])
           ↓
    PatchMasking(patch_size, mask_ratio)
           ↓
    Encoder(feature extractor) → [batch, H', W', C']
           ↓
    Decoder(transposed convs) → [batch, H, W, C]

    Loss computed only on masked patches
    ```

    Args:
        encoder: A `keras.Model` instance to be used as the feature extractor.
            It should be a feature extractor (e.g., `include_top=False`).
        patch_size: Integer, size of patches for masking. Defaults to 16.
        mask_ratio: Float, ratio of patches to mask (0 to 1). Defaults to 0.75.
        decoder_dims: List of integers, decoder layer dimensions.
            If None, auto-determined from encoder. Defaults to None.
        decoder_depth: Integer, number of decoder layers (if decoder_dims=None).
            Defaults to 4.
        norm_pix_loss: Boolean, whether to normalize pixel values for loss.
            Defaults to False.
        mask_value: String or float, value for masked patches. Options:
            "learnable", "zero", "noise", or a float. Defaults to "learnable".
        input_shape: Tuple, input image shape (H, W, C). Defaults to (224, 224, 3).
        **kwargs: Additional arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

    Output shape:
        Dictionary with keys:
            - "reconstruction": Reconstructed images (batch, H, W, C)
            - "mask": Binary mask (batch, num_patches)
            - "masked_input": Input with masked patches (batch, H, W, C)
            - "encoded": Encoded features (batch, H', W', C')

    Attributes:
        masking: PatchMasking layer for creating and applying masks.
        encoder: Encoder model (feature extractor).
        decoder: ConvDecoder for reconstructing images.
        reconstruction_loss_tracker: Metric for tracking reconstruction loss.

    Example:
        >>> # Create MAE with a pre-built encoder
        >>> encoder = ConvNeXtV2.from_variant("tiny", include_top=False)
        >>> mae = MaskedAutoencoder(
        ...     encoder=encoder,
        ...     patch_size=16,
        ...     mask_ratio=0.75,
        ...     decoder_dims=[512, 256, 128, 64],
        ...     input_shape=(224, 224, 3)
        ... )
        >>>
        >>> # Compile and train
        >>> mae.compile(optimizer="adam")
        >>> mae.fit(train_data, epochs=100)
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
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(encoder, keras.Model):
            raise TypeError("encoder must be a keras.Model instance.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if not 0 <= mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        if decoder_depth <= 0:
            raise ValueError(f"decoder_depth must be positive, got {decoder_depth}")
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D (H, W, C), got {input_shape}")

        # Store configuration
        self.encoder = encoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.decoder_dims = decoder_dims
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        self.mask_value = mask_value
        self.input_shape_config = input_shape

        # Determine encoder's output shape to configure the decoder
        if not self.encoder.built:
            self.encoder.build((None,) + input_shape)
        encoder_output_shape = self.encoder.compute_output_shape((None,) + input_shape)

        if len(encoder_output_shape) != 4:
            raise ValueError(
                f"Expected encoder to have 4D output (batch, H, W, C), "
                f"but got shape {encoder_output_shape}"
            )

        self.encoder_height = encoder_output_shape[1]
        self.encoder_width = encoder_output_shape[2]
        self.encoder_channels = encoder_output_shape[3]

        # CREATE sub-layers in __init__

        # 1. Masking layer
        self.masking = PatchMasking(
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            name="patch_masking"
        )

        # 2. Decoder
        self.decoder = self._create_decoder()

        # Loss tracker
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

    def _create_decoder(self) -> ConvDecoder:
        """Create decoder based on configuration.

        Returns:
            ConvDecoder instance.
        """
        # Determine decoder dimensions if not provided
        if self.decoder_dims is None:
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

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of MAE.

        Args:
            inputs: Input images tensor.
            training: Boolean or None, whether in training mode.

        Returns:
            Dictionary with keys:
                - "reconstruction": Reconstructed images
                - "mask": Binary mask (1=masked, 0=visible)
                - "masked_input": Input with masked patches
                - "encoded": Encoded features
        """
        # Apply masking
        masked_images, mask, num_patches = self.masking(inputs, training=training)

        # Encode
        encoded = self.encoder(masked_images, training=training)

        # Decode
        reconstruction = self.decoder(encoded, training=training)

        return {
            "reconstruction": reconstruction,
            "mask": mask,
            "masked_input": masked_images,
            "encoded": encoded
        }

    def compute_loss(
            self,
            x: Optional[keras.KerasTensor] = None,
            y: Optional[keras.KerasTensor] = None,
            y_pred: Optional[Dict[str, keras.KerasTensor]] = None,
            sample_weight: Optional[keras.KerasTensor] = None,
            **kwargs: Any
    ) -> keras.KerasTensor:
        """Compute MAE reconstruction loss.

        Loss is computed only on masked patches.

        Args:
            x: Input images (used as reconstruction target).
            y: Not used (MAE is self-supervised).
            y_pred: Dictionary with reconstruction and mask.
            sample_weight: Optional sample weights.
            **kwargs: Additional arguments.

        Returns:
            Scalar loss tensor.
        """
        if y_pred is None:
            return keras.ops.convert_to_tensor(0.0)

        target = x
        reconstruction = y_pred["reconstruction"]
        mask = y_pred["mask"]

        # Compute per-pixel loss
        if self.norm_pix_loss:
            # Normalize target per patch
            target_patches = self._extract_patches_for_loss(target)
            mean = keras.ops.mean(target_patches, axis=[-3, -2, -1], keepdims=True)
            var = keras.ops.var(target_patches, axis=[-3, -2, -1], keepdims=True)
            target_normalized = (target_patches - mean) / keras.ops.sqrt(var + 1e-6)
            target = self._reconstruct_patches_for_loss(target_normalized)

        # MSE loss
        loss = keras.ops.square(target - reconstruction)
        loss = keras.ops.mean(loss, axis=-1)  # Average over channels

        # Apply mask - only compute loss on masked regions
        mask_reshaped = self._reshape_mask_for_loss(mask, target)
        loss = loss * mask_reshaped

        # Average over masked patches
        num_masked = keras.ops.sum(mask, axis=-1, keepdims=True) + 1e-6
        loss = keras.ops.sum(loss, axis=[1, 2]) / num_masked

        # Average over batch
        loss = keras.ops.mean(loss)

        return loss

    def _extract_patches_for_loss(
            self,
            images: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Extract patches for loss computation.

        Args:
            images: Image tensor.

        Returns:
            Patches tensor.
        """
        batch_size = keras.ops.shape(images)[0]
        height, width, channels = self.input_shape_config
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        patches = keras.ops.reshape(
            images,
            (batch_size, num_patches_h, self.patch_size,
             num_patches_w, self.patch_size, channels)
        )
        patches = keras.ops.transpose(patches, (0, 1, 3, 2, 4, 5))

        return patches

    def _reconstruct_patches_for_loss(
            self,
            patches: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Reconstruct image from patches for loss computation.

        Args:
            patches: Patches tensor.

        Returns:
            Image tensor.
        """
        batch_size = keras.ops.shape(patches)[0]
        height, width, channels = self.input_shape_config

        patches = keras.ops.transpose(patches, (0, 1, 3, 2, 4, 5))
        images = keras.ops.reshape(
            patches,
            (batch_size, height, width, channels)
        )

        return images

    def _reshape_mask_for_loss(
            self,
            mask: keras.KerasTensor,
            target: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Reshape patch mask to match image dimensions for loss.

        Args:
            mask: Patch mask tensor (batch, num_patches).
            target: Target image tensor.

        Returns:
            Reshaped mask tensor matching target spatial dimensions.
        """
        batch_size = keras.ops.shape(mask)[0]
        height, width = keras.ops.shape(target)[1], keras.ops.shape(target)[2]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Reshape to (batch, num_patches_h, num_patches_w)
        mask_2d = keras.ops.reshape(mask, (batch_size, num_patches_h, num_patches_w))

        # Expand to match patch size
        mask_2d = keras.ops.expand_dims(mask_2d, axis=-1)
        mask_2d = keras.ops.repeat(mask_2d, self.patch_size, axis=2)
        mask_2d = keras.ops.expand_dims(mask_2d, axis=3)
        mask_2d = keras.ops.repeat(mask_2d, self.patch_size, axis=3)

        # Flatten spatial dimensions
        mask_img = keras.ops.reshape(mask_2d, (batch_size, height, width))

        return mask_img

    def train_step(
            self,
            data: Union[keras.KerasTensor, Tuple]
    ) -> Dict[str, float]:
        """Custom training step for MAE.

        Args:
            data: Input data (images only, no labels needed).

        Returns:
            Dictionary of metrics.
        """
        # Import TF for GradientTape
        import tensorflow as tf

        # Unpack data
        if isinstance(data, tuple):
            x, _ = data
        else:
            x = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Compute loss
            loss = self.compute_loss(x=x, y=None, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.reconstruction_loss_tracker.update_state(loss)

        return {
            "loss": loss,
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }

    def test_step(
            self,
            data: Union[keras.KerasTensor, Tuple]
    ) -> Dict[str, float]:
        """Custom test step for MAE.

        Args:
            data: Input data (images only).

        Returns:
            Dictionary of metrics.
        """
        # Unpack data
        if isinstance(data, tuple):
            # When validation_data=(x_val, y_val) is passed to fit,
            # y_val is ignored here as MAE is self-supervised.
            x, _ = data
        else:
            x = data

        # Forward pass.
        # We use training=True to enable patch masking during validation.
        # This is the correct way to evaluate an MAE, as the task is to
        # reconstruct a masked image. Keras internally ensures that layers
        # like Dropout or BatchNormalization remain in inference mode.
        y_pred = self(x, training=True)

        # Compute loss
        loss = self.compute_loss(x=x, y=None, y_pred=y_pred)

        # Update metrics
        self.reconstruction_loss_tracker.update_state(loss)

        return {
            "loss": loss,
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return model metrics."""
        return [self.reconstruction_loss_tracker]

    def visualize(
            self,
            image: np.ndarray,
            return_arrays: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Visualize MAE reconstruction for a single image.

        Args:
            image: Input image array of shape (H, W, C).
            return_arrays: Boolean, whether to return numpy arrays.

        Returns:
            Tuple of (original, masked, reconstructed) images.
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Forward pass with training=True to enable masking
        outputs = self(image, training=True)

        masked = outputs["masked_input"]
        reconstructed = outputs["reconstruction"]

        if return_arrays:
            image = keras.ops.convert_to_numpy(image[0])
            masked = keras.ops.convert_to_numpy(masked[0])
            reconstructed = keras.ops.convert_to_numpy(reconstructed[0])

        return image, masked, reconstructed

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = {
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "decoder_dims": self.decoder_dims,
            "decoder_depth": self.decoder_depth,
            "norm_pix_loss": self.norm_pix_loss,
            "mask_value": self.mask_value,
            "input_shape": self.input_shape_config,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MaskedAutoencoder":
        config["encoder"] = keras.saving.deserialize_keras_object(config["encoder"])
        return cls(**config)

# ---------------------------------------------------------------------
