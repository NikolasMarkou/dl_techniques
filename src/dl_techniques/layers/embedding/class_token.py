"""
Prepend a learnable [CLS] (class) token to a sequence of token embeddings.

This is the standard Vision-Transformer class token. One learnable vector is
prepended to a sequence of patch or word embeddings, so the sequence gets one
position longer. After the encoder, the vector at that leading position is the
usual aggregate representation used for classification.

Architecture:
    The input is a sequence ``X`` of shape ``(batch, seq_len, dim)``. The layer
    owns one trainable weight ``cls_token`` of shape ``(1, 1, dim)``. At call
    time that weight is broadcast across the batch and concatenated in front of
    ``X`` on the sequence axis. The result is ``(batch, seq_len + 1, dim)``.

Why this is a dedicated layer:
    The weight is created in ``build()``, so ``add_weight`` runs only once the
    layer is built, after its ``super().__init__``. That lets a host Functional
    ``keras.Model`` build its symbolic graph without ever calling ``add_weight``
    on *itself* before its own ``super().__init__(inputs=, outputs=)``. Keras 3
    requires that a Functional model create no weights at graph-construction
    time. The DINO family (v1/v3, and optionally v2) used to create the CLS
    token inside the model body before ``super().__init__``; it uses this layer
    instead.

Mathematics:
    Let ``X in R^{B x L x D}`` and ``c in R^{1 x 1 x D}``, the learnable token.
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

    The layer holds one weight of shape ``(1, 1, D)``, broadcasts it to the
    batch, and concatenates it in front of the input on the sequence axis. The
    sequence grows by exactly one position: ``L`` in, ``L + 1`` out. Nothing
    else about the input changes.

    **Architecture Overview:**

    .. code-block:: text

        cls_token  [1, 1, D]          inputs X  [B, L, D]
        (weight, created in build)              │
                  │                             │
                  ▼                             │
        ┌──────────────────────┐                │
        │ broadcast_to         │                │
        │ (B, 1, D)            │                │
        └──────────┬───────────┘                │
          [B, 1, D]│                            │
                   └─────────────┬──────────────┘
                                 ▼
                   concatenate(axis=1), token first
                                 │
                                 ▼
                    output Y  [B, L + 1, D]

    :param initializer: Initializer for the class-token weight. Defaults to
        ``"truncated_normal"``, the DINO / ViT convention. Passed straight to
        ``add_weight``, so any value ``add_weight`` accepts works.
    :type initializer: str
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar initializer: The configured weight initializer, stored as given.
    :vartype initializer: str
    :ivar cls_token: The learnable token of shape ``(1, 1, dim)``. ``None``
        until ``build()`` runs.
    :vartype cls_token: keras.Variable or None

    :raises ValueError: From ``build()``, if the input is not rank 3, or if
        its last dimension is not statically known.

    Input shape:
        3D tensor with shape ``(batch_size, sequence_length, dim)``.

    Output shape:
        3D tensor with shape ``(batch_size, sequence_length + 1, dim)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding.class_token import (
            ClassTokenPrepend,
        )

        x = keras.random.normal((4, 196, 384))
        y = ClassTokenPrepend()(x)
        y.shape  # (4, 197, 384)
    """

    def __init__(
            self,
            initializer: str = "truncated_normal",
            **kwargs: Any
    ) -> None:
        """Store the initializer. No weight is created here.

        :param initializer: Initializer for the class-token weight.
        :type initializer: str
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        """
        super().__init__(**kwargs)
        self.initializer = initializer
        self.cls_token = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the ``(1, 1, dim)`` class-token weight.

        :param input_shape: Shape of the input sequence, expected to be
            ``(batch_size, sequence_length, dim)``.
        :type input_shape: tuple
        :raises ValueError: If the input is not rank 3, or if its last axis
            is ``None``.
        """
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
        # The weight is created here, in build, after super().__init__. That is
        # what keeps a host Functional Model from running add_weight before its
        # own super().__init__.
        self.cls_token = self.add_weight(
            shape=(1, 1, dim),
            initializer=self.initializer,
            trainable=True,
            name="cls_token",
        )
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Broadcast the token to the batch and prepend it.

        :param inputs: Input sequence of shape ``(batch, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :return: The sequence with the class token at position 0, of shape
            ``(batch, seq_len + 1, dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(inputs)[0]
        cls_tokens = keras.ops.broadcast_to(
            self.cls_token, (batch_size, 1, keras.ops.shape(inputs)[2])
        )
        return keras.ops.concatenate([cls_tokens, inputs], axis=1)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Add one to the sequence axis.

        :param input_shape: Shape of the input sequence.
        :type input_shape: tuple
        :return: ``(batch, seq_len + 1, dim)``. The sequence entry stays
            ``None`` if it was ``None``.
        :rtype: tuple
        """
        batch, seq, dim = input_shape
        new_seq = seq + 1 if seq is not None else None
        return (batch, new_seq, dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus ``initializer``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({"initializer": self.initializer})
        return config

# ---------------------------------------------------------------------
