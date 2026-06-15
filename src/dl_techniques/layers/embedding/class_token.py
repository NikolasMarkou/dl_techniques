"""
Prepend a learnable [CLS] (class) token to a sequence of token embeddings.

This layer implements the standard Vision-Transformer "class token" mechanism:
a single learnable vector is prepended to the sequence of patch (or word)
embeddings, increasing the sequence length by one. The representation at this
leading position, after the transformer encoder, is conventionally used as the
aggregate sequence representation for classification.

Architecture:
    Given an input sequence ``X`` of shape ``(batch, seq_len, dim)``, the layer
    owns a single trainable weight ``cls_token`` of shape ``(1, 1, dim)``. At
    call time the token is broadcast across the batch and concatenated in front
    of ``X`` along the sequence axis, producing ``(batch, seq_len + 1, dim)``.

Why this is a dedicated layer:
    The weight is created in ``build()`` (i.e. ``add_weight`` runs only when the
    layer is built, AFTER its ``super().__init__``), and the layer is invoked
    inside a functional graph. This lets a host Functional ``keras.Model`` build
    its symbolic graph without ever calling ``add_weight`` on *itself* before its
    own ``super().__init__(inputs=, outputs=)`` — the Keras-3 rule that a
    Functional model creates no weights at graph-construction time. Reused by the
    DINO model family (v1/v3, and optionally v2) which previously created the CLS
    token directly inside the model body before ``super().__init__``.

Mathematics:
    Let ``X in R^{B x L x D}`` and ``c in R^{1 x 1 x D}`` (the learnable token).
    The output ``Y in R^{B x (L+1) x D}`` is::

        Y[b, 0, :]   = c[0, 0, :]            (broadcast over batch b)
        Y[b, i, :]   = X[b, i-1, :]          for i = 1 .. L
"""

import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ClassTokenPrepend(keras.layers.Layer):
    """Prepend a single learnable [CLS] token to a ``(B, L, D)`` sequence.

    Args:
        initializer: Initializer for the class-token weight. Defaults to
            ``"truncated_normal"`` (matching the DINO / ViT convention).
        **kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    Input shape:
        3D tensor ``(batch_size, sequence_length, dim)``.

    Output shape:
        3D tensor ``(batch_size, sequence_length + 1, dim)``.

    Attributes:
        cls_token: The learnable weight of shape ``(1, 1, dim)`` created in
            ``build()``.
    """

    def __init__(
            self,
            initializer: str = "truncated_normal",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.initializer = initializer
        self.cls_token = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 3:
            raise ValueError(
                f"ClassTokenPrepend expects a 3D input (batch, seq, dim); "
                f"got input_shape={input_shape}"
            )
        dim = input_shape[-1]
        if dim is None:
            raise ValueError(
                "ClassTokenPrepend requires a static feature dimension "
                "(input_shape[-1] must be known)."
            )
        # Weight is created here (in build, after super().__init__), so a host
        # Functional Model never runs add_weight before its own super().__init__.
        self.cls_token = self.add_weight(
            shape=(1, 1, dim),
            initializer=self.initializer,
            trainable=True,
            name="cls_token",
        )
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        batch_size = keras.ops.shape(inputs)[0]
        cls_tokens = keras.ops.broadcast_to(
            self.cls_token, (batch_size, 1, keras.ops.shape(inputs)[2])
        )
        return keras.ops.concatenate([cls_tokens, inputs], axis=1)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        batch, seq, dim = input_shape
        new_seq = seq + 1 if seq is not None else None
        return (batch, new_seq, dim)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"initializer": self.initializer})
        return config

# ---------------------------------------------------------------------
