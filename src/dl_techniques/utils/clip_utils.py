"""Shared CLIP dual-encoder helpers.

Small, backend-agnostic (``keras.ops``) building blocks reused by more than
one CLIP-family model so the last-non-pad-token gather and the temperature-
scaled logits epilogue are defined once instead of being copy-pasted per
model file.

These are plain functions, not registered Keras layers or losses, so there
is no serialization concern; they hold no state and add no trainable
weights. They exist purely to remove duplicated inline blocks from
:mod:`dl_techniques.models.clip.clifford_clip` and
:mod:`dl_techniques.models.clip.model`.

Interface contract (both functions):
    * Pure, stateless, side-effect free.
    * Operate through ``keras.ops`` only (no raw TensorFlow) so they run on
      any Keras 3 backend.
    * Dtype-preserving: outputs follow the dtype of their tensor inputs.
"""

from typing import Any, Callable, Optional

from keras import ops

# ---------------------------------------------------------------------


def apply_clifford_head(
    head_kind: str,
    anchor: Any,
    z_det: Any,
    z_ctx: Any,
    geo_layer: Callable[[Any, Any], Any],
    scale_layer: Callable[[Any], Any],
    training: Optional[bool] = None,
) -> Any:
    """Mix a Clifford geometric-product head into the canonical CLIP anchor.

    Shared by CliffordCLIP's vision and text towers (they differ only in which
    tensors feed anchor/z_det/z_ctx). Reproduces the per-tower head_kind dispatch:

    - ``"plain"``: return the anchor unchanged (no geometric product).
    - ``"mean_max"`` / ``"learned_query"``: return ``geo = geo_layer(z_det, z_ctx)``.
    - ``"learned_query_residual"``: return ``anchor + scale_layer(geo_layer(z_det, z_ctx))``.

    The dispatch is pure Python; the tensor work happens inside ``geo_layer``
    (a ``SparseRollingGeometricProduct``-like callable) and ``scale_layer`` (a
    ``LayerScale``/``LearnableMultiplier`` gate). Neither takes a ``training``
    flag in the current CliffordCLIP call sites, so it is not forwarded;
    ``training`` is kept purely for API symmetry with the encode_* signatures.

    Args:
        head_kind: One of ``plain`` | ``mean_max`` | ``learned_query`` |
            ``learned_query_residual``.
        anchor: Canonical CLIP anchor ``(B, D)`` — GAP (vision) or last-non-pad
            (text).
        z_det: Deterministic pool ``(B, D)`` — the geometric product's first
            operand.
        z_ctx: Context pool ``(B, D)`` — the geometric product's second operand.
        geo_layer: A ``SparseRollingGeometricProduct``-like callable
            ``(z_det, z_ctx) -> (B, D)``.
        scale_layer: A ``LayerScale``/``LearnableMultiplier`` callable
            ``geo -> (B, D)`` for the residual gate.
        training: Forwarded if the layers need it (kept for API symmetry;
            unused by the current layers).

    Returns:
        Mixed features ``(B, D)``.
    """
    if head_kind == "plain":
        return anchor
    geo = geo_layer(z_det, z_ctx)
    if head_kind == "learned_query_residual":
        return anchor + scale_layer(geo)
    return geo


def last_non_pad_token(features, input_ids, pad_token_id):
    """Gather the last non-pad token's feature vector for each sequence.

    This is the canonical CLIP text anchor: after a causal text tower, the
    representation used for the pooled sentence embedding is the hidden
    state at the position of the final real (non-pad) token.

    The gather is expressed as a one-hot matmul (rather than an integer
    ``gather``/``take_along_axis``) so it stays fully differentiable and
    backend-agnostic. Assumes **right-padding**: the non-pad tokens form a
    contiguous prefix, so ``sum(input_ids != pad_token_id) - 1`` is the index
    of the last real token. Sequences that are entirely padding clip to
    index ``0``.

    Args:
        features: Feature tensor of shape ``(B, L, D)`` from the text tower.
        input_ids: Integer token-id tensor of shape ``(B, L)``.
        pad_token_id: Integer id used for padding positions.

    Returns:
        Tensor of shape ``(B, D)`` holding, per sequence, the feature vector
        at the last non-pad position.
    """
    seq_len = ops.shape(features)[1]
    non_pad = ops.cast(ops.not_equal(input_ids, pad_token_id), "int32")
    lengths = ops.sum(non_pad, axis=1)
    last_idx = ops.clip(lengths - 1, 0, seq_len - 1)
    one_hot = ops.one_hot(last_idx, num_classes=seq_len, dtype=features.dtype)
    return ops.squeeze(
        ops.matmul(ops.expand_dims(one_hot, axis=1), features), axis=1
    )


def compute_clip_logits(image_features, text_features, logit_scale):
    """Compute the symmetric CLIP similarity logits.

    Given L2-normalized image and text features and a (positive) temperature
    ``logit_scale``, returns the temperature-scaled cosine-similarity matrix
    and its transpose::

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text  = logits_per_image.T

    The caller owns computing ``logit_scale`` (e.g. ``exp(log_temperature)``
    and any dtype handling); this helper only performs the matmul epilogue,
    so it is byte-identical to the inlined blocks it replaces.

    Args:
        image_features: Tensor of shape ``(B_img, D)``.
        text_features: Tensor of shape ``(B_txt, D)``.
        logit_scale: Scalar temperature broadcast over the logits. Must be
            in a dtype compatible with the features for the multiply.

    Returns:
        Tuple ``(logits_per_image, logits_per_text)`` of shapes
        ``(B_img, B_txt)`` and ``(B_txt, B_img)``.
    """
    logits_per_image = logit_scale * ops.matmul(
        image_features, ops.transpose(text_features)
    )
    logits_per_text = ops.transpose(logits_per_image)
    return logits_per_image, logits_per_text
