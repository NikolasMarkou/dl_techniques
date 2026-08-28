"""
Emit a bank of ``R`` independent learnable "register" tokens, broadcast to the
batch of a reference sequence.

Register tokens (Darcet et al. 2023, "Vision Transformers Need Registers") are
extra learnable tokens appended to a ViT's token sequence. They give the model
somewhere to park global state instead of hijacking high-norm patch tokens. The
property that matters is independence: the ``R`` tokens are ``R`` separate
learnable vectors, ``nn.Parameter(zeros(1, R, D))`` in the reference
implementation.

Why this is a dedicated layer:
    Same reason as ``ClassTokenPrepend`` (see ``class_token.py``): the weight is
    created in ``build()``, so a host Functional ``keras.Model`` can build its
    symbolic graph without ever calling ``add_weight`` on *itself* before its own
    ``super().__init__(inputs=, outputs=)``.

    It also replaces a broken idiom. A ``Dense(D, use_bias=False)`` applied to
    ``ones((1, R, 1))`` has an input feature dim of 1, so its kernel is ``(1, D)``
    and every one of the ``R`` output rows equals ``1.0 * kernel[0]``. That is
    ``R`` bit-identical copies of ONE learnable vector sharing a single gradient
    accumulator: ``D`` parameters where the architecture calls for ``R*D``. The
    forward pass is finite and input-sensitive either way, which is why the
    usual smoke assertions never see the defect.

Interface contract:
    Input:  a reference sequence ``(batch, seq_len, dim)``. Only its batch size
            is read; its values do not enter the output. ``dim`` must match
            ``embed_dim``.
    Output: ``(batch, num_tokens, embed_dim)``.
    Raises: ``ValueError`` on ``num_tokens < 1`` or ``embed_dim < 1`` at
            construction, and on a non-3D input or a reference sequence whose
            feature dim disagrees with ``embed_dim`` at build.

    This layer does not concatenate. The caller decides where the tokens go, so
    the host graph keeps control of the insertion position. For DINOv2 that
    position is after the positional embedding.
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RegisterTokens(keras.layers.Layer):
    """Produce ``num_tokens`` independent learnable tokens for a batch.

    The layer holds one weight of shape ``(1, R, D)`` whose ``R`` rows are
    independent, casts it to the compute dtype, and broadcasts it across the
    batch. The input sequence is a reference only: its batch size is read, its
    values are discarded, and the output sequence length is ``R``, not the
    input's ``L``.

    **Architecture Overview:**

    .. code-block:: text

        inputs  [B, L, D]       register_tokens  [1, R, D]
        (reference sequence;    (weight, R independent rows)
         only B is read,                    │
         values discarded)                  ▼
              │                  ┌────────────────────────┐
              └─ shape[0] = B ─┐ │ cast to compute_dtype  │
                               │ └───────────┬────────────┘
                               │             ▼
                               │ ┌────────────────────────┐
                               └►│ broadcast_to (B, R, D) │
                                 └───────────┬────────────┘
                                             ▼
                                  output  [B, R, D]
                                  seq length L -> R

    :param num_tokens: Number of register tokens ``R``. Must be at least 1.
    :type num_tokens: int
    :param embed_dim: Token width ``D``. Must be at least 1, and must match the
        last axis of the reference sequence.
    :type embed_dim: int
    :param initializer: Initializer for the token bank. Defaults to
        ``"truncated_normal"``, the DINO convention for extra tokens. Resolved
        through ``keras.initializers.get``.
    :type initializer: str or keras.initializers.Initializer
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If ``num_tokens`` or ``embed_dim`` is below 1.

    :ivar num_tokens: The configured token count ``R``.
    :vartype num_tokens: int
    :ivar embed_dim: The configured token width ``D``.
    :vartype embed_dim: int
    :ivar initializer: The resolved initializer object.
    :vartype initializer: keras.initializers.Initializer
    :ivar register_tokens: The learnable bank of shape ``(1, R, D)``, one
        independent row per token. ``None`` until ``build()`` runs.
    :vartype register_tokens: keras.Variable or None

    Input shape:
        3D tensor with shape ``(batch_size, sequence_length, embed_dim)``, a
        reference sequence used only for its batch size.

    Output shape:
        3D tensor with shape ``(batch_size, num_tokens, embed_dim)``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding.register_tokens import (
            RegisterTokens,
        )

        x = keras.random.normal((4, 196, 384))
        regs = RegisterTokens(num_tokens=4, embed_dim=384)(x)
        regs.shape  # (4, 4, 384)

        # The caller decides the insertion point.
        seq = keras.ops.concatenate([x, regs], axis=1)

    .. note::

       The reference sequence may have a dynamic feature dim. A ``None`` last
       axis skips the width check in ``build()``; a known width that differs
       from ``embed_dim`` raises.
    """

    def __init__(
            self,
            num_tokens: int,
            embed_dim: int,
            initializer: Any = "truncated_normal",
            **kwargs: Any
    ) -> None:
        """Validate the token count and width, then resolve the initializer.

        :param num_tokens: Number of register tokens ``R``. Must be at least 1.
        :type num_tokens: int
        :param embed_dim: Token width ``D``. Must be at least 1.
        :type embed_dim: int
        :param initializer: Initializer for the token bank.
        :type initializer: str or keras.initializers.Initializer
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``num_tokens`` or ``embed_dim`` is below 1.
        """
        super().__init__(**kwargs)
        if num_tokens < 1:
            raise ValueError(
                f"RegisterTokens requires num_tokens >= 1; got {num_tokens}."
            )
        if embed_dim < 1:
            raise ValueError(
                f"RegisterTokens requires embed_dim >= 1; got {embed_dim}."
            )
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.initializer = keras.initializers.get(initializer)
        self.register_tokens = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Check the reference shape and create the ``(1, R, D)`` token bank.

        :param input_shape: Shape of the reference sequence, expected to be
            ``(batch_size, sequence_length, embed_dim)``.
        :type input_shape: tuple
        :raises ValueError: If the reference input is not rank 3, or if its
            known feature dim differs from ``embed_dim``.
        """
        if len(input_shape) != 3:
            raise ValueError(
                f"RegisterTokens expects a 3D reference input "
                f"(batch, seq, dim); got input_shape={input_shape}"
            )
        dim = input_shape[-1]
        if dim is not None and int(dim) != self.embed_dim:
            raise ValueError(
                f"RegisterTokens was configured with embed_dim={self.embed_dim} "
                f"but received a reference sequence of width {dim}."
            )
        # Shape (1, R, D) gives R independent rows. A (1, 1, D) weight
        # broadcast to R rows would reintroduce the defect this layer exists
        # to remove.
        self.register_tokens = self.add_weight(
            shape=(1, self.num_tokens, self.embed_dim),
            initializer=self.initializer,
            trainable=True,
            name="register_tokens",
        )
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Broadcast the token bank across the batch of the reference input.

        :param inputs: Reference sequence of shape ``(batch, seq_len, dim)``.
            Only its batch size is used.
        :type inputs: keras.KerasTensor
        :return: The token bank as ``(batch, num_tokens, embed_dim)``, in the
            layer's compute dtype.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(inputs)[0]
        return keras.ops.broadcast_to(
            keras.ops.cast(self.register_tokens, self.compute_dtype),
            (batch_size, self.num_tokens, self.embed_dim),
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Replace the sequence and feature axes with the token bank's.

        :param input_shape: Shape of the reference sequence.
        :type input_shape: tuple
        :return: ``(batch, num_tokens, embed_dim)``.
        :rtype: tuple
        """
        return (input_shape[0], self.num_tokens, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base ``Layer`` config plus ``num_tokens``, ``embed_dim``
            and the serialized ``initializer``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "embed_dim": self.embed_dim,
            "initializer": keras.initializers.serialize(self.initializer),
        })
        return config

# ---------------------------------------------------------------------
