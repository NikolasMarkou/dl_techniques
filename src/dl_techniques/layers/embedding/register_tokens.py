"""
Emit a bank of ``R`` independent learnable "register" tokens, broadcast to the
batch of a reference sequence.

Register tokens (Darcet et al. 2023, "Vision Transformers Need Registers") are
extra learnable tokens appended to a ViT's token sequence to give the model
somewhere to park global state instead of hijacking high-norm patch tokens. The
defining property is that the ``R`` tokens are **independent** learnable vectors:
``nn.Parameter(zeros(1, R, D))`` in the reference implementation.

Why this is a dedicated layer:
    Same reason as ``ClassTokenPrepend`` (see ``class_token.py``): the weight is
    created in ``build()``, so a host Functional ``keras.Model`` can build its
    symbolic graph without ever calling ``add_weight`` on *itself* before its own
    ``super().__init__(inputs=, outputs=)``.

    It also replaces a broken idiom. A ``Dense(D, use_bias=False)`` applied to
    ``ones((1, R, 1))`` has an input feature dim of 1, so its kernel is ``(1, D)``
    and every one of the ``R`` output rows equals ``1.0 * kernel[0]`` -- ``R``
    bit-identical copies of ONE learnable vector sharing a single gradient
    accumulator, i.e. ``D`` parameters where the architecture calls for ``R*D``.
    The forward pass is finite and input-sensitive either way, which is why the
    defect is invisible to the usual smoke assertions.

Interface contract:
    Input:  a reference sequence ``(batch, seq_len, dim)``. Only its batch size
            is read; its values do not enter the output. ``dim`` must match
            ``embed_dim``.
    Output: ``(batch, num_tokens, embed_dim)``.
    Raises: ``ValueError`` on a non-3D input, on ``num_tokens < 1``, or on a
            reference sequence whose feature dim disagrees with ``embed_dim``.

    The caller decides where the tokens go (concatenation is deliberately NOT
    done here, so the host graph keeps control of the insertion position -- for
    DINOv2 that position is after the positional embedding, on purpose).
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RegisterTokens(keras.layers.Layer):
    """Produce ``num_tokens`` independent learnable tokens for a batch.

    Args:
        num_tokens: Integer, number of register tokens ``R``. Must be >= 1.
        embed_dim: Integer, token width ``D``.
        initializer: Initializer for the token bank. Defaults to a small
            truncated normal, matching the DINO convention for extra tokens.
        **kwargs: Standard ``keras.layers.Layer`` keyword arguments.

    Input shape:
        3D tensor ``(batch_size, sequence_length, embed_dim)`` -- a reference
        sequence, used only for its batch size.

    Output shape:
        3D tensor ``(batch_size, num_tokens, embed_dim)``.

    Attributes:
        register_tokens: The learnable weight of shape ``(1, R, D)`` created in
            ``build()``. Each of the ``R`` rows is independent.
    """

    def __init__(
            self,
            num_tokens: int,
            embed_dim: int,
            initializer: Any = "truncated_normal",
            **kwargs: Any
    ) -> None:
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
        # (1, R, D): R INDEPENDENT rows. A (1, 1, D) weight broadcast to R rows
        # would reintroduce the very defect this layer exists to remove.
        self.register_tokens = self.add_weight(
            shape=(1, self.num_tokens, self.embed_dim),
            initializer=self.initializer,
            trainable=True,
            name="register_tokens",
        )
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        batch_size = keras.ops.shape(inputs)[0]
        return keras.ops.broadcast_to(
            keras.ops.cast(self.register_tokens, self.compute_dtype),
            (batch_size, self.num_tokens, self.embed_dim),
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return (input_shape[0], self.num_tokens, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "embed_dim": self.embed_dim,
            "initializer": keras.initializers.serialize(self.initializer),
        })
        return config

# ---------------------------------------------------------------------
