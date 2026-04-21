"""Tube mask generator for V-JEPA-style masked latent prediction (iter-2).

Produces a binary spatial mask ``M ∈ {0, 1}^(B, H_p, W_p)`` per sample where
exactly ``K = round(mask_ratio * H_p * W_p)`` positions are masked (=1) and
the rest are visible (=0).

The mask is a **pure spatial tube**: at model-call time, ``M`` is broadcast
across all ``T`` frames (so masked ``(h, w)`` positions stay masked for the
entire clip). Time-invariance of the mask is what keeps causality intact
under the iter-2 dual-loss training objective — a masked position cannot
leak information from a future frame if it is masked at **every** frame.

Decisions anchored here:
- **D-009**: latent-space masking (the generator is fed a batch size, not
  pixel tensors; masking is applied by the model *after* encoding).
- **D-011**: each sample receives an independently-sampled tube mask.

Implementation reuses the well-tested ``argsort``-of-uniform-noise idiom
from :mod:`dl_techniques.models.masked_autoencoder.patch_masking` to sample
exactly ``K`` distinct positions per row.

.. note::

    This layer is *stateless*: it holds no weights and no persistent
    state. The learned mask token lives on
    :class:`dl_techniques.models.video_jepa.model.VideoJEPA`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import keras
from keras import ops, random


@keras.saving.register_keras_serializable(package="video_jepa")
class TubeMaskGenerator(keras.layers.Layer):
    """Sample a per-sample spatial tube mask.

    :param mask_ratio: Fraction of the ``H_p * W_p`` spatial grid to mask
        per sample. Must be in ``[0.0, 1.0]`` inclusive — caller (the
        :class:`VideoJEPAConfig`) additionally enforces strict upper
        bound ``< 1.0`` so the next-frame loss has unmasked positions.
        Value ``0.0`` yields an all-zeros mask (regression-guard path).
    :param patches_per_side: Integer ``H_p = W_p = img_size // patch_size``.
        Used at construction time so ``K = round(mask_ratio * H_p * W_p)``
        is a Python int (simpler broadcasts).

    :returns: ``mask: (B, H_p, W_p)`` float32 in ``{0, 1}``, where
        exactly ``K`` positions are 1 per batch row.

    .. note::

        Per-row cardinality is **exact** (not approximate) thanks to the
        argsort-of-uniform-noise sampler. Verified by
        ``test_mask_ratio_exact`` in
        ``tests/test_models/test_video_jepa/test_video_jepa.py``.
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

        # ``call`` takes ``batch_size`` (a Python/scalar int), not a tensor.
        # Keras 3 defaults to rejecting non-tensor positional args — opt out.
        self._allow_non_tensor_positional_args = True

        # Precompute static counts.
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

        :param batch_size: Scalar tensor or Python int; number of samples.
        :param training: Standard Keras arg. Present for API consistency;
            ignored — masks are always sampled identically.

        :returns: ``(B, H_p, W_p)`` float32 mask. ``1.0`` = masked,
            ``0.0`` = visible. Per-row sum equals ``self.num_masked``
            exactly.
        """
        del training  # unused — deterministic per-call sampling

        # Early-out for mask_ratio == 0.0 (regression-guard path):
        # all-zeros mask is exactly correct, no sampling needed.
        if self._num_masked == 0:
            return ops.zeros(
                (batch_size, self.patches_per_side, self.patches_per_side),
                dtype="float32",
            )

        # 1. Uniform noise per (sample, patch).
        noise = random.uniform(shape=(batch_size, self._num_patches))

        # 2. Argsort-rank: rank[i, j] = where patch j falls in sample i's
        #    random ordering. Smallest K ranks -> masked.
        rand_indices = ops.argsort(noise, axis=-1)
        rank = ops.argsort(rand_indices, axis=-1)

        # 3. Boolean -> float mask, flat (B, N).
        mask_flat = ops.cast(rank < self._num_masked, dtype="float32")

        # 4. Reshape to spatial grid (B, H_p, W_p). This is the tube:
        #    the model will broadcast over T at call time.
        mask = ops.reshape(
            mask_flat,
            (batch_size, self.patches_per_side, self.patches_per_side),
        )
        return mask

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "mask_ratio": self.mask_ratio,
                "patches_per_side": self.patches_per_side,
            }
        )
        return config
