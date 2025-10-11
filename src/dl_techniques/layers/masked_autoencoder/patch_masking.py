"""
A patch-based random masking strategy for self-supervised learning.

This layer is the core data corruption mechanism for self-supervised models
like Masked Autoencoders (MAE). It performs a structured, patch-based masking
of the input image to create a challenging pretext task for a neural network.

Architecture and Design Philosophy:
The fundamental idea is to force a model to learn high-level semantic
representations by predicting large missing regions of an image, rather than
just interpolating low-level textures from adjacent pixels. To achieve this,
the layer first deconstructs the image into a non-overlapping grid of
patches. A random subset of these patches is then selected for masking.

A key design choice, popularized by the MAE paper, is the use of a shared,
learnable `mask_token`. Instead of simply zeroing out the masked patches,
they are replaced with this single, trainable vector. This provides a clear
signal to the downstream model that information is missing and allows the
model to learn a dedicated representation for "absence," which can be more
informative than a simple null value.

Foundational Algorithm:
The masking process is designed to be efficient and fully vectorized. To
randomly select a fixed ratio of patches for masking without replacement, an
indirect sorting algorithm is employed:
1. A vector of random noise is generated, with one value for each patch in
   the image. The shape of this vector is `(batch_size, num_patches)`.
2. The indices that would sort this noise vector are found using an `argsort`
   operation. This effectively produces a random permutation of the patch
   indices for each image in the batch.
3. The first `k` indices from this random permutation are chosen as the
   indices to be masked, where `k` is determined by `mask_ratio`.
This method avoids iterative sampling and is highly parallelizable on modern
hardware accelerators like GPUs and TPUs.

References:
    - [He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022).
      Masked Autoencoders Are Scalable Vision Learners. In CVPR.](
      https://arxiv.org/abs/2111.06377)
"""

import keras
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PatchMasking(keras.layers.Layer):
    """Layer for creating patches and applying random masking.

    This layer divides the input image into non-overlapping patches and
    randomly masks a specified ratio of patches. Masked patches can be
    replaced with a learnable mask token, zeros, or noise.

    **Architecture**:
    ```
    Input(shape=[batch, H, W, C])
           ↓
    Extract Patches → [batch, num_patches_h, num_patches_w, patch_h, patch_w, C]
           ↓
    Generate Random Mask → [batch, num_patches] (0=visible, 1=masked)
           ↓
    Apply Mask Token/Value to Masked Patches
           ↓
    Reconstruct Image → [batch, H, W, C]
    ```

    Args:
        patch_size: Integer, size of each square patch. Must divide image dimensions evenly.
        mask_ratio: Float between 0 and 1, ratio of patches to mask during training.
        mask_value: String or float, value to use for masked patches.
            Options: "learnable" (trainable mask token), "zero", "noise", or a float value.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        Height and width must be divisible by patch_size.

    Output shape:
        Tuple of:
            - masked_images: 4D tensor, same shape as input with masked patches
            - mask: 2D tensor of shape (batch_size, num_patches), binary mask
            - num_patches: Integer, total number of patches

    Attributes:
        mask_token: Learnable mask token (if mask_value="learnable"), created in build().

    Example:
        >>> masking = PatchMasking(patch_size=16, mask_ratio=0.75)
        >>> masked_img, mask, num_patches = masking(images, training=True)
    """

    def __init__(
            self,
            patch_size: int = 16,
            mask_ratio: float = 0.75,
            mask_value: Union[str, float] = "learnable",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if not 0 <= mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        if isinstance(mask_value, str) and mask_value not in ["learnable", "zero", "noise"]:
            raise ValueError(
                f"mask_value must be 'learnable', 'zero', 'noise', or a float, got {mask_value}"
            )

        # Store configuration
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

        # Weights created in build()
        self.mask_token = None

        # Shape attributes computed in build()
        self.num_patches_h = None
        self.num_patches_w = None
        self.num_patches = None
        self.channels = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Shape tuple (batch, height, width, channels).
        """
        _, height, width, channels = input_shape

        if height is None or width is None:
            raise ValueError("Height and width must be known at build time")

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Image dimensions ({height}x{width}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )

        self.num_patches_h = height // self.patch_size
        self.num_patches_w = width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.channels = channels

        # Create learnable mask token if needed
        if self.mask_value == "learnable":
            self.mask_token = self.add_weight(
                name="mask_token",
                shape=(1, self.patch_size, self.patch_size, channels),
                initializer="zeros",
                trainable=True
            )

        super().build(input_shape)

    def _create_mask(
            self,
            batch_size: keras.KerasTensor,
            training: bool
    ) -> keras.KerasTensor:
        """Create random binary mask for patches.

        Args:
            batch_size: Symbolic tensor for the batch size.
            training: Boolean, whether in training mode.

        Returns:
            Binary mask tensor of shape (batch_size, num_patches).
        """
        if training and self.mask_ratio > 0:
            num_masked = int(self.num_patches * self.mask_ratio)
            if num_masked == 0:
                return keras.ops.zeros((batch_size, self.num_patches), dtype="float32")

            # Generate random noise for each patch in each batch sample
            noise = keras.random.uniform(shape=(batch_size, self.num_patches))

            # Find the indices that would sort the noise. This gives a random permutation of indices.
            rand_indices = keras.ops.argsort(noise, axis=-1)

            # Determine the rank of each patch. Patches with a rank less than num_masked will be masked.
            rank = keras.ops.argsort(rand_indices, axis=-1)

            # Create the mask: 1 for masked, 0 for visible.
            mask = keras.ops.cast(rank < num_masked, dtype="float32")

            return mask
        else:
            # No masking during inference or if mask_ratio is 0
            return keras.ops.zeros((batch_size, self.num_patches), dtype="float32")

    def _extract_patches(
            self,
            images: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Extract patches from images.

        Args:
            images: Input tensor of shape (batch, height, width, channels).

        Returns:
            Patches tensor of shape (batch, num_patches_h, num_patches_w,
                                    patch_size, patch_size, channels).
        """
        batch_size = keras.ops.shape(images)[0]

        # Reshape to patches
        # (batch, H, W, C) -> (batch, num_h, patch_h, num_w, patch_w, C)
        patches = keras.ops.reshape(
            images,
            (batch_size, self.num_patches_h, self.patch_size,
             self.num_patches_w, self.patch_size, self.channels)
        )
        # -> (batch, num_h, num_w, patch_h, patch_w, C)
        patches = keras.ops.transpose(patches, (0, 1, 3, 2, 4, 5))

        return patches

    def _reconstruct_from_patches(
            self,
            patches: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Reconstruct image from patches.

        Args:
            patches: Tensor of shape (batch, num_h, num_w, patch_h, patch_w, C).

        Returns:
            Image tensor of shape (batch, height, width, channels).
        """
        batch_size = keras.ops.shape(patches)[0]

        # (batch, num_h, num_w, patch_h, patch_w, C)
        # -> (batch, num_h, patch_h, num_w, patch_w, C)
        patches = keras.ops.transpose(patches, (0, 1, 3, 2, 4, 5))

        # -> (batch, H, W, C)
        height = self.num_patches_h * self.patch_size
        width = self.num_patches_w * self.patch_size
        images = keras.ops.reshape(
            patches,
            (batch_size, height, width, self.channels)
        )

        return images

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, int]:
        """Apply patch masking to inputs.

        Args:
            inputs: Input tensor of shape (batch, height, width, channels).
            training: Boolean or None, whether in training mode.

        Returns:
            Tuple of:
                - masked_images: Images with masked patches
                - mask: Binary mask (1=masked, 0=visible)
                - num_patches: Total number of patches
        """
        batch_size = keras.ops.shape(inputs)[0]

        # Create mask
        mask = self._create_mask(batch_size, training)

        # Extract patches
        patches = self._extract_patches(inputs)

        # Apply masking
        mask_reshaped = keras.ops.reshape(
            mask,
            (batch_size, self.num_patches_h, self.num_patches_w, 1, 1, 1)
        )

        if self.mask_value == "learnable":
            mask_token = keras.ops.tile(
                self.mask_token,
                (batch_size, self.num_patches_h, self.num_patches_w, 1, 1, 1)
            )
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * mask_token
        elif self.mask_value == "zero":
            masked_patches = (1 - mask_reshaped) * patches
        elif self.mask_value == "noise":
            noise = keras.random.normal(keras.ops.shape(patches))
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * noise
        else:
            # Use constant value
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * self.mask_value

        # Reconstruct image from masked patches
        masked_images = self._reconstruct_from_patches(masked_patches)

        return masked_images, mask, self.num_patches

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "mask_value": self.mask_value,
        })
        return config

# ---------------------------------------------------------------------
