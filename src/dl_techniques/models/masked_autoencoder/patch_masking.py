"""
A patch-based random masking strategy for self-supervised learning.
"""

import keras
from keras import ops, layers, random
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PatchMasking(keras.layers.Layer):
    """Layer for creating patches and applying random masking.

    Attributes:
        mask_token: Learnable mask token (if mask_value="learnable").
    """

    def __init__(
        self,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        mask_value: Union[str, float] = "learnable",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if not 0 <= mask_ratio <= 1:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.mask_token = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        _, height, width, channels = input_shape

        if height is None or width is None:
             # Defer strict shape check to call if dynamic
             pass
        else:
            if height % self.patch_size != 0 or width % self.patch_size != 0:
                raise ValueError(
                    f"Dimensions ({height}x{width}) not divisible by patch_size {self.patch_size}"
                )

        self.channels = channels

        # Initialize Learnable Token
        if self.mask_value == "learnable":
            self.mask_token = self.add_weight(
                name="mask_token",
                shape=(1, self.patch_size, self.patch_size, channels),
                initializer="zeros",
                trainable=True,
                dtype=self.dtype
            )

        super().build(input_shape)

    def _create_mask(self, batch_size: int, num_patches: int, training: bool) -> keras.KerasTensor:
        """Generates the binary mask using argsort."""
        if not training or self.mask_ratio == 0:
            return ops.zeros((batch_size, num_patches), dtype="float32")

        num_masked = int(num_patches * self.mask_ratio)

        # 1. Random noise
        noise = random.uniform(shape=(batch_size, num_patches))

        # 2. Sort noise to get random indices
        rand_indices = ops.argsort(noise, axis=-1)

        # 3. Find rank of each patch
        rank = ops.argsort(rand_indices, axis=-1)

        # 4. Mask patches with low rank (1 = masked, 0 = visible)
        mask = ops.cast(rank < num_masked, dtype="float32")
        return mask

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, int]:

        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # 1. Create Mask (float32)
        mask = self._create_mask(batch_size, num_patches, training)

        # 2. Extract Patches: (B, H, W, C) -> (B, num_h, patch_h, num_w, patch_w, C)
        patches = ops.reshape(
            inputs,
            (batch_size, num_patches_h, self.patch_size,
             num_patches_w, self.patch_size, self.channels)
        )
        # -> (B, num_h, num_w, patch_h, patch_w, C)
        patches = ops.transpose(patches, (0, 1, 3, 2, 4, 5))

        # 3. Apply Mask
        # Reshape mask to broadcast: (B, num_h, num_w, 1, 1, 1)
        mask_reshaped = ops.reshape(mask, (batch_size, num_patches_h, num_patches_w, 1, 1, 1))

        # Cast mask to input dtype (e.g., float16) to allow mixed precision multiplication
        mask_reshaped = ops.cast(mask_reshaped, inputs.dtype)

        if self.mask_value == "learnable":
            # Broadcast token to (B, num_h, num_w, patch, patch, C)
            token = ops.cast(self.mask_token, inputs.dtype)
            token = ops.broadcast_to(token, ops.shape(patches))
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * token
        elif self.mask_value == "noise":
            noise = random.normal(ops.shape(patches), dtype=inputs.dtype)
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * noise
        elif self.mask_value == "zero":
            masked_patches = (1 - mask_reshaped) * patches
        else:
             # Constant float
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * ops.cast(self.mask_value, inputs.dtype)

        # 4. Reconstruct: (B, num_h, num_w, patch_h, patch_w, C) -> (B, H, W, C)
        masked_patches = ops.transpose(masked_patches, (0, 1, 3, 2, 4, 5))
        masked_images = ops.reshape(masked_patches, (batch_size, height, width, self.channels))

        return masked_images, mask, num_patches

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "mask_value": self.mask_value,
        })
        return config

# ---------------------------------------------------------------------
