"""Tube mask generator for V-JEPA-style masked latent prediction.

`TubeMaskGenerator` produces a binary spatial mask `M` of shape `(B, H_p,
W_p)` per sample, where exactly `K = round(mask_ratio * H_p * W_p)`
positions are masked. The mask is a spatial tube: the model broadcasts it
across every frame at call time, so a masked position stays masked for the
whole clip and cannot leak information from a future frame.

The generator takes a batch size, not pixel tensors: masking is applied by
the model after encoding, and each sample gets an independently sampled
mask. Exact per-row cardinality reuses the argsort-of-uniform-noise idiom
from :mod:`dl_techniques.models.vision.masked_autoencoder.patch_masking`.

This layer is stateless: it holds no weights. The learned mask token lives
on :class:`dl_techniques.models.vision.video_jepa.model.VideoJEPA`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import keras
from keras import ops, random
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.video_jepa.masking")
class TubeMaskGenerator(keras.layers.Layer):
    """Sample a per-sample spatial tube mask.

    Per-row cardinality is exact, not approximate, thanks to the
    argsort-of-uniform-noise sampler.

    :param mask_ratio: Fraction of the `H_p * W_p` spatial grid to mask per
        sample, in `[0.0, 1.0]`. `VideoJEPAConfig` additionally enforces a
        strict upper bound of `1.0` so the next-frame loss keeps unmasked
        positions. `0.0` yields an all-zeros mask.
    :type mask_ratio: float
    :param patches_per_side: `H_p = W_p = img_size // patch_size`, used at
        construction time so `K = round(mask_ratio * H_p * W_p)` is a plain
        Python int.
    :type patches_per_side: int
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        mask_ratio: float,
        patches_per_side: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not (0.0 <= mask_ratio <= 1.0):
            raise ValueError(
                f"mask_ratio must be in [0.0, 1.0], got {mask_ratio}"
            )
        if patches_per_side <= 0:
            raise ValueError(
                f"patches_per_side must be positive, got {patches_per_side}"
            )

        self.mask_ratio = float(mask_ratio)
        self.patches_per_side = int(patches_per_side)

        # call() takes batch_size as a Python/scalar int, not a tensor; Keras 3
        # rejects non-tensor positional args by default, so opt out.
        self._allow_non_tensor_positional_args = True

        self._num_patches = self.patches_per_side * self.patches_per_side
        self._num_masked = int(round(self.mask_ratio * self._num_patches))

    @property
    def num_masked(self) -> int:
        """Exact number of positions masked per sample, ``K``."""
        return self._num_masked

    @property
    def num_patches(self) -> int:
        """``H_p * W_p`` — static grid size."""
        return self._num_patches

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def call(
        self,
        batch_size: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Sample a fresh tube mask.

        :param batch_size: Scalar tensor or Python int, the number of samples.
        :param training: Accepted for API consistency; ignored, since masks
            are always sampled the same way regardless of mode.
        :return: `(B, H_p, W_p)` float32 mask, `1.0` masked and `0.0`
            visible, with the per-row sum exactly `self.num_masked`.
        :rtype: keras.KerasTensor
        """
        del training

        # mask_ratio == 0.0: an all-zeros mask is exact, no sampling needed.
        if self._num_masked == 0:
            return ops.zeros(
                (batch_size, self.patches_per_side, self.patches_per_side),
                dtype="float32",
            )

        noise = random.uniform(shape=(batch_size, self._num_patches))

        # rank[i, j] is where patch j falls in sample i's random ordering;
        # the smallest K ranks are masked.
        rand_indices = ops.argsort(noise, axis=-1)
        rank = ops.argsort(rand_indices, axis=-1)

        mask_flat = ops.cast(rank < self._num_masked, dtype="float32")

        # Reshape to the spatial grid; the model broadcasts this tube over T.
        mask = ops.reshape(
            mask_flat,
            (batch_size, self.patches_per_side, self.patches_per_side),
        )
        return mask

    def compute_output_shape(self, input_shape):
        """Output spatial tube mask shape ``(B, H_p, W_p)``."""
        return (None, self.patches_per_side, self.patches_per_side)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "mask_ratio": self.mask_ratio,
                "patches_per_side": self.patches_per_side,
            }
        )
        return config
