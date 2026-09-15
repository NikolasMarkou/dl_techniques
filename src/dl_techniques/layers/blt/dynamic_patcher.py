"""DynamicPatcher: segment a byte sequence into patches by thresholding
entropy, for the Byte Latent Transformer (BLT).

A position ``t`` opens a new patch when ``H(x_t) > entropy_threshold``. Each
byte is assigned the number of boundaries at or before it, saturated at
``max_patches - 1``; the returned patch lengths are the occupancy counts of
that assignment. The layer holds no weights.
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.blt.dynamic_patcher")
class DynamicPatcher(keras.layers.Layer):
    """Segment a byte sequence into patches by thresholding entropy.

    A position ``t`` opens a new patch when ``H(x_t) > entropy_threshold``.
    Each byte is assigned the number of boundaries at or before it, saturated
    at ``max_patches - 1``; the returned patch lengths are the occupancy
    counts of that assignment. The layer holds no weights.

    Architecture:

    .. code-block:: text

        entropy [B, S]
              │
              ▼
        ┌──────────────────────────────┐
        │ entropy > threshold          │
        └──────────────────────────────┘
              │ is_boundary [B, S]
              ▼
        ┌──────────────────────────────┐
        │ cumsum, min at P-1           │
        └──────────────────────────────┘
              │ patch_index [B, S]
              ▼
        ┌──────────────────────────────┐
        │ one-hot, then sum over S     │
        └──────────────────────────────┘
              │
              ▼
        patch_lengths [B, max_patches]

    Segmentation, with max_patches 4:

    .. code-block:: text

        position           0   1   2   3   4   5
        is_boundary        0   1   0   0   1   0
        patch_index        0   1   1   1   2   2
        patch_lengths      1   3   2   0

    Rows sum to ``seq_len`` by construction, since every byte is counted into
    exactly one patch; ``compute_patch_ids`` does not re-validate that sum. The
    cap truncates by position, not by entropy magnitude, so everything after
    the ``(max_patches - 1)``-th boundary merges into the final patch. Keeping
    the highest-entropy boundaries instead would let a late high-entropy byte
    displace an earlier one, making an earlier byte's patch id depend on a
    later byte.

    A leading zero-length patch is legal when position 0 is itself a
    boundary; trailing patches are zero-length whenever a sequence produces
    fewer boundaries than ``max_patches - 1``. Both leave patch ids
    non-decreasing, which ``LocalDecoder``'s preceding-patch gather requires.

    :param entropy_threshold: Entropy in nats above which a byte opens a new
        patch. For a vocabulary of size ``V`` the entropy of a uniform
        distribution is ``ln(V)``; a threshold at or below the model's
        typical entropy makes every position a boundary, and one above
        ``ln(V)`` makes none.
    :type entropy_threshold: float
    :param max_patches: Maximum number of patches to create.
    :type max_patches: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
            self,
            entropy_threshold: float = 1.5,
            max_patches: int = 512,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.entropy_threshold = entropy_threshold
        self.max_patches = max_patches

    def call(
            self,
            entropy: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Derive patch lengths from entropy values, per row.

        Each row is segmented independently, so sequences with different
        content get different boundaries.

        :param entropy: Entropy, shape ``(batch_size, seq_len)``, in nats (as
            produced by ``EntropyModel.compute_entropy``).
        :type entropy: keras.KerasTensor
        :param training: Unused — the segmentation is deterministic.
        :type training: Optional[bool]
        :return: Patch lengths, shape ``(batch_size, max_patches)``,
            ``int32``, non-negative, each row summing to exactly ``seq_len``.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-14T183218-f4c612aa/D-012: cap by position, not by
        # entropy magnitude; a top-k cap breaks causality, moving logits 4.85e-01.
        # See decisions.md.
        is_boundary = ops.cast(entropy > self.entropy_threshold, 'int32')

        # The saturating cumsum depends only on entropy[..., :t + 1].
        patch_index = ops.cumsum(is_boundary, axis=1)
        patch_index = ops.minimum(patch_index, self.max_patches - 1)

        # compute_patch_ids already materializes a tensor of this exact shape, so
        # the one-hot is not a new memory regime.
        occupancy = ops.one_hot(patch_index, self.max_patches, dtype='int32')

        return ops.sum(occupancy, axis=1)

    def warn_if_segmentation_is_degenerate(
            self,
            entropy: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
    ) -> bool:
        """Warn if a concrete batch's segmentation is degenerate.

        Pure except for the log record; returns ``True`` only if a warning
        was emitted, so a caller can assert on the decision rather than on log
        text. Never raises. Requires an eager tensor, since it reads the
        entropy values.

        Pass ``mask`` whenever the batch is padded. The rate is a mean over
        positions, and a trained entropy model drives padding to near-zero
        entropy, so a mostly-padding batch can hide a fully degenerate real
        region behind a low unmasked rate.

        Degenerate means one of two ends:

        - boundary rate 1.0 — every position opens a patch, so patch 0 is
          empty, one byte lands in each patch after it, and the remaining
          tail merges into the final patch. An untrained entropy model
          (output near the uniform ceiling ``ln(vocab_size)``) produces this
          against any threshold below that ceiling.
        - boundary rate 0.0 — no position opens a patch, so the whole
          sequence is one patch and ``max_patches`` is inert.

        Rates strictly between the ends are an ordinary segmentation and are
        not reported.

        :param entropy: Concrete (eager) entropy tensor, ``(batch, seq_len)``,
            in nats — the same tensor ``call`` consumes.
        :type entropy: keras.KerasTensor
        :param mask: Optional concrete (eager) tensor broadcastable to
            ``entropy``, non-zero at real positions and zero at padding. When
            omitted, every position counts, which is correct only for an
            unpadded batch.
        :type mask: Optional[keras.KerasTensor]
        :return: ``True`` if a warning was logged. This includes the
            no-real-positions case (an all-zero ``mask``), reported as a
            defect in the caller's probe rather than a degenerate
            segmentation.
        :rtype: bool
        """
        # DECISION plan-2026-08-14T183218-f4c612aa/D-018: opt-in, never called from
        # call(), which is keras.ops-only; a branch on a value needs eager data.
        # DECISION plan-2026-08-14T183218-f4c612aa/D-024: measure over real positions
        # when a mask is given; unmasked hides content behind padding (0.1250 vs
        # 1.0000). See decisions.md.
        is_boundary = ops.cast(entropy > self.entropy_threshold, 'float32')

        if mask is None:
            rate = float(ops.convert_to_numpy(ops.mean(is_boundary)))
            scope = "this batch"
        else:
            weights = ops.cast(ops.cast(mask, 'bool'), 'float32')
            counted = float(ops.convert_to_numpy(ops.sum(weights)))
            if counted == 0.0:
                logger.warning(
                    f"{type(self).__name__}: the supplied mask selects NO "
                    f"positions, so no boundary rate could be measured and this "
                    f"diagnostic saw nothing. Check the caller's mask."
                )
                return True
            rate = float(
                ops.convert_to_numpy(ops.sum(is_boundary * weights))
            ) / counted
            scope = (
                f"this batch (measured over its {int(counted)} non-padding "
                f"positions; padding excluded)"
            )

        if rate == 1.0:
            logger.warning(
                f"{type(self).__name__}: entropy_threshold="
                f"{self.entropy_threshold:.4g} nats is below the entropy at "
                f"EVERY position of {scope} (observed boundary rate 1.0), "
                f"so patch 0 is empty, each patch after it holds one byte, and "
                f"the whole remaining sequence collapses into the final patch "
                f"of max_patches={self.max_patches}. Raise the threshold, or "
                f"pretrain the entropy model before relying on the "
                f"segmentation."
            )
            return True

        if rate == 0.0:
            logger.warning(
                f"{type(self).__name__}: entropy_threshold="
                f"{self.entropy_threshold:.4g} nats is above the entropy at "
                f"every position of {scope} (observed boundary rate 0.0), "
                f"so the whole sequence is a single patch and "
                f"max_patches={self.max_patches} is inert. Lower the threshold."
            )
            return True

        return False

    def compute_patch_ids(
            self,
            patch_lengths: keras.KerasTensor,
            seq_len: Optional[int] = None
    ) -> keras.KerasTensor:
        """Convert patch lengths to a patch id for each byte position.

        A position's id is the number of cumulative patch lengths at or below
        it, clamped to ``max_patches - 1``, so a zero-length patch consumes no
        position.

        :param patch_lengths: Patch lengths, shape ``(batch_size, max_patches)``.
        :type patch_lengths: keras.KerasTensor
        :param seq_len: Sequence length to expand to. Pass this from the
            caller's own byte tensor whenever known. When ``None``, it is
            recovered from the data as ``max(sum(patch_lengths))``, which
            makes the layer's output shape data-dependent and XLA-incompatible.
        :type seq_len: Optional[int]
        :return: Patch ids, shape ``(batch_size, seq_len)``, ``int32``.
        :rtype: keras.KerasTensor
        """
        max_patches = ops.shape(patch_lengths)[1]

        # DECISION plan-2026-08-19T163559-499b6f0e/D-034: pass seq_len in; deriving it
        # makes the output shape data-dependent, which XLA rejects. See decisions.md.
        if seq_len is None:
            max_seq_len = ops.max(ops.sum(patch_lengths, axis=1))
        else:
            max_seq_len = seq_len

        cumulative_lengths = ops.cumsum(patch_lengths, axis=1)

        positions = ops.arange(max_seq_len)
        positions = ops.expand_dims(ops.expand_dims(positions, 0), -1)
        cum_expanded = ops.expand_dims(cumulative_lengths, 1)

        boundary_passed = ops.cast(cum_expanded <= positions, 'int32')
        patch_ids = ops.sum(boundary_passed, axis=-1)
        patch_ids = ops.minimum(patch_ids, max_patches - 1)

        return ops.cast(patch_ids, 'int32')

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Entropy shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, max_patches)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], self.max_patches)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'entropy_threshold': self.entropy_threshold,
            'max_patches': self.max_patches
        })
        return config
