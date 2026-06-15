"""
Replace masked positions of a token sequence with a single learnable mask token.

This layer implements the iBOT / BEiT "mask token" mechanism used by
self-supervised vision transformers (DINOv2, iBOT, BEiT, MAE-style masking):
a single learnable vector substitutes the patch embedding at every position
flagged by a boolean mask, while unmasked positions pass through unchanged.

Architecture:
    Given a patch-embedding sequence ``X`` of shape ``(batch, seq_len, dim)`` and
    a boolean mask ``M`` of shape ``(batch, seq_len)`` (``True`` = this position
    is masked / should be replaced), the layer owns a single trainable weight
    ``mask_token`` of shape ``(1, 1, dim)`` and returns::

        Y[b, i, :] = mask_token[0, 0, :]   if M[b, i]      (masked)
                   = X[b, i, :]            otherwise       (kept)

Why this is a dedicated layer:
    The weight is created in ``build()`` (i.e. ``add_weight`` runs only when the
    layer is built, AFTER its ``super().__init__``), and the layer is invoked
    inside a functional graph. This lets a host Functional ``keras.Model`` build
    its symbolic graph without ever calling ``add_weight`` on *itself* before its
    own ``super().__init__(inputs=, outputs=)`` — the Keras-3 rule that a
    Functional model creates no weights at graph-construction time. This mirrors
    ``ClassTokenPrepend`` (same package) and supersedes the degenerate
    "Dense-on-ones with zeros init" pattern that produces a constant-zero (not
    learnable) mask vector.

Mathematics:
    Let ``X in R^{B x L x D}``, ``M in {0,1}^{B x L}`` and ``m in R^{1 x 1 x D}``
    (the learnable mask token). With ``M' = expand_dims(M, -1)`` broadcast over
    the feature axis::

        Y = where(M', m, X)   in R^{B x L x D}
"""

import keras
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MaskTokenApply(keras.layers.Layer):
    """Replace masked positions of ``(B, L, D)`` with a learnable mask token.

    Call signature: ``layer((patch_embeddings, mask))`` where ``patch_embeddings``
    is ``(B, L, D)`` and ``mask`` is a boolean ``(B, L)`` tensor (``True`` marks a
    position to be replaced by the learnable mask token, iBOT convention).

    Args:
        initializer: Initializer for the mask-token weight. Defaults to a
            ``TruncatedNormal(stddev=0.02)`` (matching the DINO / ViT / iBOT
            convention for token initialization).
        **kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    Input shape:
        Tuple/list of two tensors:
            - ``patch_embeddings``: 3D ``(batch_size, sequence_length, dim)``.
            - ``mask``: 2D boolean ``(batch_size, sequence_length)``.

    Output shape:
        3D tensor ``(batch_size, sequence_length, dim)`` (same as the embeddings).

    Attributes:
        mask_token: The learnable weight of shape ``(1, 1, dim)`` created in
            ``build()``.
    """

    def __init__(
            self,
            initializer: Any = "truncated_normal",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Default to the DINO/iBOT stddev=0.02 truncated-normal when the caller
        # passes the bare string default; honor any explicit initializer object.
        if initializer == "truncated_normal":
            initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self.initializer = keras.initializers.get(initializer)
        self.mask_token = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        # input_shape is a list/tuple: [patch_embeddings_shape, mask_shape]
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "MaskTokenApply expects two inputs (patch_embeddings, mask); "
                f"got input_shape={input_shape}"
            )
        emb_shape = input_shape[0]
        if len(emb_shape) != 3:
            raise ValueError(
                "MaskTokenApply expects 3D patch embeddings (batch, seq, dim); "
                f"got embeddings shape={emb_shape}"
            )
        dim = emb_shape[-1]
        if dim is None:
            raise ValueError(
                "MaskTokenApply requires a static feature dimension "
                "(embeddings shape[-1] must be known)."
            )
        # Weight is created here (in build, after super().__init__), so a host
        # Functional Model never runs add_weight before its own super().__init__.
        self.mask_token = self.add_weight(
            shape=(1, 1, dim),
            initializer=self.initializer,
            trainable=True,
            name="mask_token",
        )
        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        patch_embeddings, mask = inputs
        # Broadcast: cond (B, L, 1), mask_token (1, 1, D), embeddings (B, L, D).
        mask_expanded = keras.ops.expand_dims(mask, -1)
        return keras.ops.where(mask_expanded, self.mask_token, patch_embeddings)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        # Output is the embeddings shape, unchanged.
        return tuple(input_shape[0])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "initializer": keras.initializers.serialize(self.initializer),
        })
        return config

# ---------------------------------------------------------------------
