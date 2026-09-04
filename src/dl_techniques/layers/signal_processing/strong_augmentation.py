"""
Strong data augmentation for consistency-based regularization.

Applies color jittering (per-sample brightness and contrast) followed by
CutMix to produce a heavily distorted view of an input image — the "strong"
view a consistency method such as FixMatch compares against a weakly
augmented view. CutMix pastes a random rectangular patch from another image
in the batch: ``x_new = mask * x_perm + (1 - mask) * x``.

``call()`` returns the mixed image only, which is correct for a target-free
consumer. A supervised caller must use
:meth:`StrongAugmentation.augment_with_mix` and
:meth:`StrongAugmentation.apply_mix_to_target` instead, so the pasted box is
also pasted into the target — otherwise the cut rectangle carries another
sample's label as noise. ``input_value_range`` declares the input's valid
range for clipping; pass ``None`` for standardized or ``[-1, +1]`` inputs,
where clipping to ``[0, 1]`` would zero out negative pixels on the training
path only.

References:
    - Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong
      Classifiers with Localizable Features. *ICCV*.
    - Sohn, K., et al. (2020). FixMatch: Simplifying Semi-Supervised Learning
      with Consistency and Confidence. *NeurIPS*.
    - Krizhevsky, A., et al. (2012). ImageNet Classification with Deep
      Convolutional Neural Networks. *NeurIPS*. (Early use of color
      augmentation).
"""

import keras
from keras import ops
from typing import Dict, Tuple, Any, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.signal_processing.strong_augmentation")
class StrongAugmentation(keras.layers.Layer):
    """
    Strong augmentation layer for consistency-based regularization.

    Applies sequential color jittering (brightness and contrast adjustment) and
    CutMix augmentation during training. During inference the input is passed
    through unchanged. Color jittering modifies pixel values via random scaling
    factors, while CutMix pastes a random rectangular patch from a shuffled
    batch image onto the original: ``x_new = M * x_perm + (1-M) * x``.

    Supervised callers must use :meth:`augment_with_mix` +
    :meth:`apply_mix_to_target` rather than ``call``, so the CutMix box that was
    pasted into the image is also pasted into the target. See the module
    docstring.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [B, H, W, C]              │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Color Jitter                    │
        │  brightness: I * alpha           │
        │  contrast: (I-mu)*beta + mu      │
        │  clip to input_value_range        │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  CutMix (with prob cutmix_prob)  │
        │  paste random patch from         │
        │  shuffled batch                  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Output [B, H, W, C]             │
        └──────────────────────────────────┘

    :param cutmix_prob: Probability of applying CutMix augmentation.
    :type cutmix_prob: float
    :param cutmix_ratio_range: Range for CutMix cut ratio ``(min, max)``.
    :type cutmix_ratio_range: tuple[float, float]
    :param color_jitter_strength: Strength of color jittering.
    :type color_jitter_strength: float
    :param input_value_range: Declared ``(min, max)`` range of the input images,
        used to clip the color-jitter result back into a valid range. Pass
        ``None`` to disable clipping entirely, which is what standardized or
        ``[-1, +1]`` inputs require. Defaults to ``(0.0, 1.0)``.
    :type input_value_range: tuple[float, float] or None
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any
    """

    def __init__(
            self,
            cutmix_prob: float = 0.5,
            cutmix_ratio_range: Tuple[float, float] = (0.1, 0.5),
            color_jitter_strength: float = 0.2,
            input_value_range: Optional[Tuple[float, float]] = (0.0, 1.0),
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not (0.0 <= cutmix_prob <= 1.0):
            raise ValueError(f"cutmix_prob must be in [0, 1], got {cutmix_prob}")
        if color_jitter_strength < 0.0:
            raise ValueError(
                f"color_jitter_strength must be non-negative, got {color_jitter_strength}"
            )
        if (len(cutmix_ratio_range) != 2
                or not (0.0 <= cutmix_ratio_range[0] <= cutmix_ratio_range[1] <= 1.0)):
            raise ValueError(
                f"cutmix_ratio_range must be (min, max) with 0 <= min <= max <= 1, "
                f"got {cutmix_ratio_range}"
            )
        if input_value_range is not None:
            if (len(input_value_range) != 2
                    or not (input_value_range[0] < input_value_range[1])):
                raise ValueError(
                    f"input_value_range must be (min, max) with min < max, or None "
                    f"to disable clipping; got {input_value_range}"
                )

        self.cutmix_prob = cutmix_prob
        self.cutmix_ratio_range = cutmix_ratio_range
        self.color_jitter_strength = color_jitter_strength
        self.input_value_range = (
            None if input_value_range is None else tuple(input_value_range)
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Apply strong augmentations to input images.

        Returns the mixed image only. A caller that has a supervision target for
        ``inputs`` must use :meth:`augment_with_mix` instead — CutMix mixes across
        batch rows, and a target left unmixed becomes label noise.

        :param inputs: Input images tensor with shape ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Augmented images tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        x, _ = self.augment_with_mix(inputs, training=training)
        return x

    def augment_with_mix(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]]:
        """Augment images and return the CutMix descriptor alongside them.

        Interface contract (this method has 2+ callers — ``call`` and every
        supervised consumer):

        * Returns ``(x_aug, mix)``. ``x_aug`` has the same shape and dtype as
          ``inputs``.
        * ``mix`` is ``None`` when ``training`` is falsy (nothing was applied),
          otherwise the pair ``(mix_mask, perm_indices)``: ``mix_mask`` has shape
          ``(height, width, 1)`` and is ``1.0`` inside the pasted box and ``0.0``
          elsewhere (identically ``0.0`` when the per-batch probability gate did
          not fire); ``perm_indices`` is the ``(batch_size,)`` donor permutation.
        * Feed that pair unchanged to :meth:`apply_mix_to_target` for every
          target tensor that shares ``inputs``' batch and spatial layout.
        * Failure mode: none — the method never raises for a rank-4 input; the
          probability gate is symbolic, so the returned shapes are static and the
          call is graph-traceable inside ``model.fit``.

        :param inputs: Input images tensor ``(batch_size, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Tuple of augmented images and the mix descriptor (or ``None``).
        :rtype: tuple
        """
        if not training:
            return inputs, None

        x = self._apply_color_jitter(inputs)
        x, mix_mask, perm_indices = self._apply_cutmix(x)
        return x, (mix_mask, perm_indices)

    def apply_mix_to_target(
            self,
            target: keras.KerasTensor,
            mix: Optional[Tuple[keras.KerasTensor, keras.KerasTensor]],
    ) -> keras.KerasTensor:
        """Apply the CutMix box from :meth:`augment_with_mix` to a target tensor.

        Interface contract: ``target`` must share the batch and spatial axes of
        the images the descriptor came from; its channel count is free (a depth
        map, a depth+validity-mask pair, a one-hot label map). ``mix=None``
        returns ``target`` unchanged, so callers need no branch of their own.

        :param target: Supervision target ``(batch_size, height, width, ...)``.
        :type target: keras.KerasTensor
        :param mix: The ``(mix_mask, perm_indices)`` pair, or ``None``.
        :type mix: tuple or None
        :return: Target with the same rectangle replaced by the donor's target.
        :rtype: keras.KerasTensor
        """
        if mix is None:
            return target
        mix_mask, perm_indices = mix
        mask = ops.cast(mix_mask, target.dtype)
        target_perm = ops.take(target, perm_indices, axis=0)
        return ops.multiply(target, ops.subtract(1.0, mask)) + ops.multiply(
            target_perm, mask
        )

    def _apply_color_jitter(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply color jittering augmentation.

        :param x: Input images tensor.
        :type x: keras.KerasTensor
        :return: Color-jittered images tensor.
        :rtype: keras.KerasTensor
        """
        # Per-sample factor (B, 1, 1, 1) broadcasts over (H, W, C). See
        # README Known Issues #9.
        batch_size = ops.shape(x)[0]
        per_sample_shape = (batch_size, 1, 1, 1)

        # Brightness adjustment
        brightness_factor = keras.random.uniform(
            shape=per_sample_shape,
            minval=1.0 - self.color_jitter_strength,
            maxval=1.0 + self.color_jitter_strength
        )
        x = ops.multiply(x, brightness_factor)

        # Contrast adjustment (per-sample)
        contrast_factor = keras.random.uniform(
            shape=per_sample_shape,
            minval=1.0 - self.color_jitter_strength,
            maxval=1.0 + self.color_jitter_strength
        )
        mean_val = ops.mean(x, axis=[1, 2, 3], keepdims=True)
        x = ops.multiply(ops.subtract(x, mean_val), contrast_factor) + mean_val

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-014: clip into the declared
        # range only -- an unconditional [0, 1] clip zeroes `depth_anything`'s [-1, +1] inputs during training only.
        if self.input_value_range is not None:
            x = ops.clip(x, self.input_value_range[0], self.input_value_range[1])

        return x

    def _apply_cutmix(
            self,
            x: keras.KerasTensor,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Apply CutMix augmentation.

        :param x: Input images tensor.
        :type x: keras.KerasTensor
        :return: Tuple of the CutMix-augmented images, the ``(H, W, 1)`` mix mask
            (already multiplied by the symbolic probability gate) and the
            ``(batch_size,)`` donor permutation. The last two are what lets a
            caller mix the supervision target by the same box.
        :rtype: tuple
        """
        # Apply CutMix with probability. Use a symbolic gate (no Python `if`)
        # so the layer is graph-traceable inside `model.fit`.
        should_apply = keras.random.uniform(shape=()) < self.cutmix_prob
        gate = ops.cast(should_apply, "float32")  # 0.0 or 1.0

        batch_size = ops.shape(x)[0]
        height, width = ops.shape(x)[1], ops.shape(x)[2]

        # Generate random permutation
        perm_indices = keras.random.shuffle(ops.arange(batch_size))
        x_perm = ops.take(x, perm_indices, axis=0)

        # Generate random cut ratio
        cut_ratio = keras.random.uniform(
            shape=(),
            minval=self.cutmix_ratio_range[0],
            maxval=self.cutmix_ratio_range[1]
        )

        # Calculate cut dimensions
        cut_h = ops.cast(ops.cast(height, "float32") * cut_ratio, "int32")
        cut_w = ops.cast(ops.cast(width, "float32") * cut_ratio, "int32")

        # Cut position: draw float in [0, 1) and cast to int32, since
        # keras.random.uniform requires a floating dtype. (D-005 follow-up.)
        cut_y_f = keras.random.uniform(shape=(), minval=0.0, maxval=1.0)
        cut_x_f = keras.random.uniform(shape=(), minval=0.0, maxval=1.0)
        cut_y = ops.cast(cut_y_f * ops.cast(height - cut_h, "float32"), "int32")
        cut_x = ops.cast(cut_x_f * ops.cast(width - cut_w, "float32"), "int32")

        # Create mask
        mask = ops.zeros((height, width, 1))

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

        # Gate by the symbolic apply-probability so mask=0 when skipped. The
        # trailing singleton channel axis broadcasts onto any target channel count.
        mask = mask * gate

        # Mix images
        mask_x = ops.cast(mask, x.dtype)
        x = ops.multiply(x, ops.subtract(1.0, mask_x)) + ops.multiply(x_perm, mask_x)

        return x, mask, perm_indices

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape equals input shape (augmentation preserves dimensions).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (identical to input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "cutmix_prob": self.cutmix_prob,
            "cutmix_ratio_range": self.cutmix_ratio_range,
            "color_jitter_strength": self.color_jitter_strength,
            "input_value_range": self.input_value_range,
        })
        return config

# ---------------------------------------------------------------------
