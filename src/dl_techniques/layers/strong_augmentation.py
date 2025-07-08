import keras
from keras import ops
from typing import Dict, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class StrongAugmentation(keras.layers.Layer):
    """Strong augmentation layer for unlabeled data.

    Implements color jittering and CutMix augmentations using Keras operations
    for backend compatibility.

    Args:
        cutmix_prob: Float, probability of applying CutMix augmentation.
        cutmix_ratio_range: Tuple of floats, range for CutMix cut ratio.
        color_jitter_strength: Float, strength of color jittering.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            cutmix_prob: float = 0.5,
            cutmix_ratio_range: Tuple[float, float] = (0.1, 0.5),
            color_jitter_strength: float = 0.2,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.cutmix_prob = cutmix_prob
        self.cutmix_ratio_range = cutmix_ratio_range
        self.color_jitter_strength = color_jitter_strength

    def call(self, inputs: keras.KerasTensor, training: bool = None) -> keras.KerasTensor:
        """Apply strong augmentations to input images.

        Args:
            inputs: Input images tensor with shape (batch_size, height, width, channels).
            training: Boolean indicating whether in training mode.

        Returns:
            Augmented images tensor with same shape as input.
        """
        if not training:
            return inputs

        # Apply color jittering
        x = self._apply_color_jitter(inputs)

        # Apply CutMix with probability
        x = self._apply_cutmix(x)

        return x

    def _apply_color_jitter(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Apply color jittering augmentation.

        Args:
            x: Input images tensor.

        Returns:
            Color-jittered images tensor.
        """
        # Brightness adjustment
        brightness_factor = ops.random.uniform(
            shape=(),
            minval=1.0 - self.color_jitter_strength,
            maxval=1.0 + self.color_jitter_strength
        )
        x = ops.multiply(x, brightness_factor)

        # Contrast adjustment
        contrast_factor = ops.random.uniform(
            shape=(),
            minval=1.0 - self.color_jitter_strength,
            maxval=1.0 + self.color_jitter_strength
        )
        mean_val = ops.mean(x, axis=[1, 2, 3], keepdims=True)
        x = ops.multiply(ops.subtract(x, mean_val), contrast_factor) + mean_val

        # Clip to valid range
        x = ops.clip(x, 0.0, 1.0)

        return x

    def _apply_cutmix(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Apply CutMix augmentation.

        Args:
            x: Input images tensor.

        Returns:
            CutMix-augmented images tensor.
        """
        # Apply CutMix with probability
        should_apply = ops.random.uniform(shape=()) < self.cutmix_prob

        if not should_apply:
            return x

        batch_size = ops.shape(x)[0]
        height, width = ops.shape(x)[1], ops.shape(x)[2]

        # Generate random permutation
        perm_indices = ops.random.shuffle(ops.arange(batch_size))
        x_perm = ops.take(x, perm_indices, axis=0)

        # Generate random cut ratio
        cut_ratio = ops.random.uniform(
            shape=(),
            minval=self.cutmix_ratio_range[0],
            maxval=self.cutmix_ratio_range[1]
        )

        # Calculate cut dimensions
        cut_h = ops.cast(ops.cast(height, "float32") * cut_ratio, "int32")
        cut_w = ops.cast(ops.cast(width, "float32") * cut_ratio, "int32")

        # Generate random cut position
        cut_y = ops.random.uniform(
            shape=(),
            minval=0,
            maxval=height - cut_h,
            dtype="int32"
        )
        cut_x = ops.random.uniform(
            shape=(),
            minval=0,
            maxval=width - cut_w,
            dtype="int32"
        )

        # Create mask
        mask = ops.zeros((height, width, 1))
        ones_patch = ops.ones((cut_h, cut_w, 1))

        # Apply patch to mask (simplified approach)
        # In practice, you might want to use more sophisticated masking
        mask = ops.where(
            ops.logical_and(
                ops.logical_and(
                    ops.arange(height)[:, None] >= cut_y,
                    ops.arange(height)[:, None] < cut_y + cut_h
                ),
                ops.logical_and(
                    ops.arange(width)[None, :] >= cut_x,
                    ops.arange(width)[None, :] < cut_x + cut_w
                )
            )[:, :, None],
            ops.ones_like(mask),
            mask
        )

        # Apply mask to all channels
        mask = ops.tile(mask, [1, 1, 3])

        # Mix images
        x = ops.multiply(x, ops.subtract(1.0, mask)) + ops.multiply(x_perm, mask)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "cutmix_prob": self.cutmix_prob,
            "cutmix_ratio_range": self.cutmix_ratio_range,
            "color_jitter_strength": self.color_jitter_strength,
        })
        return config

# ---------------------------------------------------------------------
