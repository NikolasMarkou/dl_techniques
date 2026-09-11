"""``PatchMasking``, a layer that splits an image into patches and randomly masks a
fraction of them.

Defines :class:`PatchMasking`, which returns the masked image alongside the mask it
drew. Patches are chosen by a per-batch random ranking rather than a fixed grid, so
the masked set differs on every call: uniform noise is sorted twice to turn it into a
rank per patch, and the lowest-ranked patches are masked. A masked patch is replaced
by a learnable token, Gaussian noise, zero, or a constant, depending on
``mask_value``. Masking happens only when ``training`` is truthy; otherwise the layer
returns the image untouched and an all-zero mask. ``call`` returns three values, not
one, and height and width have to divide by ``patch_size``.
"""

import keras
from keras import ops, random
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.masked_autoencoder.patch_masking")
class PatchMasking(keras.layers.Layer):
    """Split an image into patches and randomly mask a fraction of them.

    The image is reshaped into a patch grid, the masked patches are substituted, and
    the grid is folded back to the original resolution, so the output is the same
    shape as the input.

    Architecture:

    .. code-block:: text

        image [B, H, W, C]
                 │
                 ▼
        ┌──────────────────────────┐
        │ split into patches       │  [B, nH, nW, pH, pW, C]
        └──────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ replace masked patches   │ ◄── mask [B, nH*nW]
        └──────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ fold back to an image    │
        └──────────────────────────┘
                 │
                 ├──► masked_images [B, H, W, C]
                 ├──► mask [B, nH*nW], 1 masked and 0 visible
                 └──► num_patches

    Mask construction:

    .. code-block:: text

        training and mask_ratio > 0?
                    │
            ┌───────┴───────┐
            ▼               ▼
           no              yes
        zeros [B, N]    uniform noise [B, N]
                        argsort -> shuffled indices
                        argsort -> per-patch rank
                        rank < int(N * mask_ratio)

    Fill:

    .. code-block:: text

        mask_value
             │
             ├─ "learnable"  ──► mask_token, broadcast over patches
             ├─ "noise"      ──► gaussian noise per patch
             ├─ "zero"       ──► the patch drops to 0
             └─ float        ──► that constant

    Visible patches pass through as (1 - mask) * patches in every case.

    :param patch_size: side length of a square patch.
    :param mask_ratio: fraction of patches to mask, in [0, 1]. At 0 the mask is
        all zeros whatever ``training`` says.
    :param mask_value: how to fill a masked patch: ``"learnable"`` (a
        trained token), ``"noise"``, ``"zero"``, or a constant float.
    :param **kwargs: passthrough to `keras.layers.Layer`.

    :raises ValueError: If ``patch_size`` is not positive, ``mask_ratio`` is outside
        [0, 1], or the built height and width do not divide by ``patch_size``.

    :ivar mask_token: The learnable mask token, set only when
        ``mask_value="learnable"``.
    :vartype mask_token: keras.Variable or None

    Input shape:
        4D tensor `(batch, height, width, channels)`, with height and width
        divisible by `patch_size`.

    Output shape:
        A 3-tuple: masked images `(batch, height, width, channels)`, the mask
        `(batch, num_patches)`, and `num_patches` itself.
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
        """Record the channel count and create the mask token if one is needed.

        The channel count is taken from here, so ``call`` reuses it rather than
        reading it back off the input.

        :param input_shape: Shape tuple `(batch, height, width, channels)`.
        :raises ValueError: If a static height or width does not divide by
            ``patch_size``.
        """
        _, height, width, channels = input_shape

        if height is None or width is None:
             # Dynamic spatial dims: divisibility by patch_size is not checked here.
             pass
        else:
            if height % self.patch_size != 0 or width % self.patch_size != 0:
                raise ValueError(
                    f"Dimensions ({height}x{width}) not divisible by patch_size {self.patch_size}"
                )

        self.channels = channels

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
        """Draw the binary patch mask by ranking uniform noise.

        :param batch_size: Rows to draw a mask for.
        :param num_patches: Patches per row.
        :param training: When false, the mask is all zeros.
        :return: Float mask `(batch_size, num_patches)`, 1 where masked.
        """
        if not training or self.mask_ratio == 0:
            return ops.zeros((batch_size, num_patches), dtype="float32")

        num_masked = int(num_patches * self.mask_ratio)

        noise = random.uniform(shape=(batch_size, num_patches))
        rand_indices = ops.argsort(noise, axis=-1)
        # Ranking the shuffled indices recovers each patch's random rank.
        rank = ops.argsort(rand_indices, axis=-1)
        # 1 = masked, 0 = visible.
        mask = ops.cast(rank < num_masked, dtype="float32")
        return mask

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, int]:
        """Mask a fraction of the image's patches and fold the grid back.

        :param inputs: Input images `(B, H, W, C)`.
        :param training: When false, nothing is masked and the mask is all zeros.
        :return: `(masked_images, mask, num_patches)`, with `mask` shaped
            `(B, num_patches)` and 1 marking a masked patch.
        """
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        mask = self._create_mask(batch_size, num_patches, training)

        patches = ops.reshape(
            inputs,
            (batch_size, num_patches_h, self.patch_size,
             num_patches_w, self.patch_size, self.channels)
        )
        patches = ops.transpose(patches, (0, 1, 3, 2, 4, 5))

        # Cast to the input dtype so the multiply below stays in one precision.
        mask_reshaped = ops.reshape(mask, (batch_size, num_patches_h, num_patches_w, 1, 1, 1))
        mask_reshaped = ops.cast(mask_reshaped, inputs.dtype)

        if self.mask_value == "learnable":
            token = ops.cast(self.mask_token, inputs.dtype)
            token = ops.broadcast_to(token, ops.shape(patches))
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * token
        elif self.mask_value == "noise":
            noise = random.normal(ops.shape(patches), dtype=inputs.dtype)
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * noise
        elif self.mask_value == "zero":
            masked_patches = (1 - mask_reshaped) * patches
        else:
            masked_patches = (1 - mask_reshaped) * patches + mask_reshaped * ops.cast(self.mask_value, inputs.dtype)

        masked_patches = ops.transpose(masked_patches, (0, 1, 3, 2, 4, 5))
        masked_images = ops.reshape(masked_patches, (batch_size, height, width, self.channels))

        return masked_images, mask, num_patches

    def compute_output_shape(self, input_shape):
        """Return the shape of the masked image, which equals the input shape.

        This covers the first return value only; ``call`` also returns the mask and
        the patch count.

        :param input_shape: Shape tuple `(batch, height, width, channels)`.
        :return: The same shape tuple.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dict holding every constructor argument.
        """
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "mask_ratio": self.mask_ratio,
            "mask_value": self.mask_value,
        })
        return config

# ---------------------------------------------------------------------
