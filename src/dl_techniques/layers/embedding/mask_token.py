"""
Replace masked positions of a token sequence with a single learnable mask token.

This is the iBOT / BEiT mask token used by self-supervised vision transformers
(DINOv2, iBOT, BEiT, MAE-style masking). One learnable vector replaces the patch
embedding at every position the boolean mask flags. Unmasked positions pass
through untouched.

Architecture:
    The inputs are a patch-embedding sequence ``X`` of shape
    ``(batch, seq_len, dim)`` and a boolean mask ``M`` of shape
    ``(batch, seq_len)``, where ``True`` means "this position is masked". The
    layer owns one trainable weight ``mask_token`` of shape ``(1, 1, dim)`` and
    returns::

        Y[b, i, :] = mask_token[0, 0, :]   if M[b, i]      (masked)
                   = X[b, i, :]            otherwise       (kept)

Why this is a dedicated layer:
    The weight is created in ``build()``, so ``add_weight`` runs only once the
    layer is built, after its ``super().__init__``. That lets a host Functional
    ``keras.Model`` build its symbolic graph without ever calling ``add_weight``
    on *itself* before its own ``super().__init__(inputs=, outputs=)``. Keras 3
    requires that a Functional model create no weights at graph-construction
    time. ``ClassTokenPrepend`` in this package exists for the same reason.

    This layer also replaces a broken idiom: a ``Dense`` applied to a tensor of
    ones with a zeros initializer. That produces a constant-zero vector that
    never learns, not a learnable mask token.

Mathematics:
    Let ``X in R^{B x L x D}``, ``M in {0,1}^{B x L}`` and ``m in R^{1 x 1 x D}``
    be the learnable mask token. With ``M' = expand_dims(M, -1)`` broadcast over
    the feature axis::

        Y = where(M', m, X)   in R^{B x L x D}
"""

import keras
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MaskTokenApply(keras.layers.Layer):
    """Replace masked positions of ``(B, L, D)`` with a learnable mask token.

    Call the layer on a pair: ``layer((patch_embeddings, mask))``, where
    ``patch_embeddings`` is ``(B, L, D)`` and ``mask`` is a boolean ``(B, L)``
    tensor. ``True`` marks a position to replace, which is the iBOT convention.
    The sequence length does not change; only the values at masked positions do.

    **Architecture Overview:**

    .. code-block:: text

        mask M  [B, L]            embeddings X  [B, L, D]
              │                             │
              ▼                             │
        ┌──────────────────────┐            │
        │ expand_dims(M, -1)   │            │
        │ -> [B, L, 1]         │            │
        └──────────┬───────────┘            │
                   │   mask_token [1, 1, D] │
                   │   (weight)             │
                   │        │               │
                   └────────┼───────────────┘
                            ▼
              where(cond, mask_token, X)
              broadcast over B, L and D
                            │
                            ▼
             output Y  [B, L, D]   (L unchanged)

    :param initializer: Initializer for the mask-token weight. The default
        string ``"truncated_normal"`` is replaced by
        ``TruncatedNormal(stddev=0.02)``, the DINO / ViT / iBOT convention for
        token initialization. Any other value is passed to
        ``keras.initializers.get`` unchanged.
    :type initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar initializer: The resolved initializer object.
    :vartype initializer: keras.initializers.Initializer
    :ivar mask_token: The learnable token of shape ``(1, 1, dim)``. ``None``
        until ``build()`` runs.
    :vartype mask_token: keras.Variable or None

    :raises ValueError: From ``build()``, if the input is not a pair of
        shapes, if the embeddings shape is not rank 3, or if the embeddings'
        last dimension is not statically known.

    Input shape:
        Tuple or list of two tensors:

        - ``patch_embeddings``: 3D ``(batch_size, sequence_length, dim)``.
        - ``mask``: 2D boolean ``(batch_size, sequence_length)``.

    Output shape:
        3D tensor with shape ``(batch_size, sequence_length, dim)``, the same
        shape as the embeddings.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding.mask_token import (
            MaskTokenApply,
        )

        x = keras.random.normal((4, 196, 384))
        m = keras.random.uniform((4, 196)) < 0.4
        y = MaskTokenApply()((x, m))
        y.shape  # (4, 196, 384)
    """

    def __init__(
            self,
            initializer: Any = "truncated_normal",
            **kwargs: Any
    ) -> None:
        """Resolve the initializer. No weight is created here.

        :param initializer: Initializer for the mask-token weight. The bare
            string default is upgraded to ``TruncatedNormal(stddev=0.02)``.
        :type initializer: str or keras.initializers.Initializer
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        """
        super().__init__(**kwargs)
        # The bare string default becomes the DINO/iBOT stddev=0.02 truncated
        # normal. An explicit initializer object is honored as given.
        if initializer == "truncated_normal":
            initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        self.initializer = keras.initializers.get(initializer)
        self.mask_token = None

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """Create the ``(1, 1, dim)`` mask-token weight.

        :param input_shape: A list or tuple of two shapes, in the order
            ``[patch_embeddings_shape, mask_shape]``.
        :type input_shape: list of tuple
        :raises ValueError: If the layer did not receive exactly two inputs,
            if the embeddings are not rank 3, or if their last axis is
            ``None``.
        """
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
        # The weight is created here, in build, after super().__init__. That is
        # what keeps a host Functional Model from running add_weight before its
        # own super().__init__.
        self.mask_token = self.add_weight(
            shape=(1, 1, dim),
            initializer=self.initializer,
            trainable=True,
            name="mask_token",
        )
        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """Substitute the mask token at every flagged position.

        :param inputs: The pair ``(patch_embeddings, mask)``, with shapes
            ``(B, L, D)`` and ``(B, L)``.
        :type inputs: list of keras.KerasTensor
        :return: The embeddings with masked positions replaced, shape
            ``(B, L, D)``.
        :rtype: keras.KerasTensor
        """
        patch_embeddings, mask = inputs
        # The three operands broadcast together: cond (B, L, 1), mask_token
        # (1, 1, D), embeddings (B, L, D).
        mask_expanded = keras.ops.expand_dims(mask, -1)
        return keras.ops.where(mask_expanded, self.mask_token, patch_embeddings)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Return the embeddings shape, unchanged.

        :param input_shape: The pair of input shapes.
        :type input_shape: list of tuple
        :return: ``input_shape[0]`` as a tuple.
        :rtype: tuple
        """
        return tuple(input_shape[0])

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus the serialized
            ``initializer``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "initializer": keras.initializers.serialize(self.initializer),
        })
        return config

# ---------------------------------------------------------------------
