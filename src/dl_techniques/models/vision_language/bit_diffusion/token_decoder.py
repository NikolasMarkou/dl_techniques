"""``SharedTokenDecoder`` reads text back out of a sampled bridge tensor.

The bridge carries a caption as raw token embeddings, not token ids, and the
diffusion model never sees a vocabulary. This decoder turns the sampler's
output back into words: one small MLP applied to every token position
independently, sharing weights across positions the way a language-model
head shares weights across time.

It is a separate model, not part of :class:`DiTXA` and not touched by the
diffusion training loop. It consumes the reverse direction's output
(image to text) after that output has been unpacked by
:func:`~.token_bridge.bridge_to_token_flat`.

.. code-block:: text

    bridge tensor            (B, H, W, C)        e.g. (B, 32, 32, 4)
        |
        |  bridge_to_token_flat()   -- exact inverse of the packing
        v
    token_flat               (B, token_seq_len * token_emb_dim)
        |
        |  reshape, then L2-normalize each token row
        v
    unit tokens              (B, token_seq_len, token_emb_dim)
        |
        |  Dense(hidden_dim) -> GELU -> Dense(hidden_dim) -> GELU
        |  -> Dense(vocab_size)                 [exact-erf GELU, see below]
        v
    logits                   (B, token_seq_len, vocab_size)

Normalizing each token row makes the decoder invariant to the absolute
scale of its input: ``decode(x)`` and ``decode(k * x)`` give the same
logits for any ``k > 0``. This makes it robust to a sampler whose output
drifts in magnitude, and means the decoder does not need to know
``token_scale``.

This decoder uses exact-erf GELU, while the DiTXA transformer blocks use
the tanh approximation; the two formulas differ by up to 4.732e-04 and are
pinned by a value-level guard, not to be unified.

References:
    - Upstream ``token_decoder.py``, the source of this port.
    - Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*.
      arXiv:1606.08415. Section 2 gives both the exact ``x * Phi(x)`` form used
      here and the tanh approximation used by the transformer blocks.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

#: ``torch.nn.functional.normalize``'s default lower bound on the norm, the
#: only thing standing between an all-zero (padding) token row and a 0/0.
DEFAULT_NORMALIZE_EPSILON: float = 1e-12

#: Upstream's ``torch.nn.GELU()`` default. ``False`` is the exact
#: ``0.5 * x * (1 + erf(x / sqrt(2)))``; ``True`` is the tanh approximation the
#: DiTXA block's MLP uses. This package uses both.
# DECISION plan-2026-09-02T094601-77d4a04e/D-026: do not unify this with the
# gelu_tanh FFN factory key blocks.py uses -- the two formulas differ by up to 4.732e-04, invisible to shape/dtype/round-trip tests. See decisions.md.
GELU_APPROXIMATE: bool = False

#: Number of ``Dense`` layers in the shared per-token head.
NUM_MLP_LAYERS: int = 3


def _normalize_epsilon_for(epsilon: float, dtype: str) -> float:
    """Raise ``epsilon`` to the smallest normal of ``dtype`` when it underflows.

    # DECISION plan-2026-09-02T094601-77d4a04e/D-025: floor epsilon at the compute
    # dtype's smallest normal -- under float16, 1e-12 is itself zero, so a padding row's norm divides by zero. See decisions.md.

    :param epsilon: The configured lower bound on the token norm.
    :type epsilon: float
    :param dtype: The compute dtype the normalization runs in.
    :type dtype: str
    :return: ``epsilon``, or the dtype's smallest normal when that is larger.
    :rtype: float
    """
    try:
        tiny = float(np.finfo(np.dtype(dtype)).tiny)
    except (TypeError, ValueError):  # a non-float policy; nothing to floor
        return float(epsilon)
    return max(float(epsilon), tiny)


@register_dl_technique(package="dl_techniques.models.bit_diffusion.token_decoder")
class SharedTokenDecoder(keras.Model):
    """Per-token MLP head mapping bridge token embeddings to vocabulary logits.

    :param vocab_size: Number of output logits per token position.
    :type vocab_size: int
    :param hidden_dim: Width of both hidden ``Dense`` layers.
    :type hidden_dim: int
    :param token_seq_len: Token positions carried by one bridge tensor; must
        match :attr:`~.config.BridgeConfig.token_seq_len`.
    :type token_seq_len: int
    :param token_emb_dim: Width of one token embedding; must match
        :attr:`~.config.BridgeConfig.token_emb_dim`.
    :type token_emb_dim: int
    :param normalize_epsilon: Lower bound on a token's L2 norm, so an all-zero
        padding token decodes rather than dividing by zero.
    :type normalize_epsilon: float
    :param use_bias: Whether the three ``Dense`` layers carry biases; upstream
        does (``torch.nn.Linear``'s default).
    :type use_bias: bool
    :raises ValueError: If any size argument is not positive.

    Example:
        >>> decoder = SharedTokenDecoder(vocab_size=1000, token_seq_len=8,
        ...                              token_emb_dim=32, hidden_dim=16)
        >>> logits = decoder(keras.ops.zeros((2, 8 * 32)))
        >>> tuple(logits.shape)
        (2, 8, 1000)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        token_seq_len: int = 64,
        token_emb_dim: int = 64,
        normalize_epsilon: float = DEFAULT_NORMALIZE_EPSILON,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (
            ("vocab_size", vocab_size),
            ("hidden_dim", hidden_dim),
            ("token_seq_len", token_seq_len),
            ("token_emb_dim", token_emb_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if float(normalize_epsilon) <= 0.0:
            raise ValueError(
                f"normalize_epsilon must be positive, got {normalize_epsilon}"
            )

        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.token_seq_len = int(token_seq_len)
        self.token_emb_dim = int(token_emb_dim)
        self.normalize_epsilon = float(normalize_epsilon)
        self.use_bias = bool(use_bias)

        # Three separate initializer instances: one Initializer object shared
        # across layers draws bit-identically wherever shapes agree.
        self.mlp_in = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.GlorotUniform(),
            bias_initializer=keras.initializers.Zeros(),
            name="mlp_in",
        )
        self.mlp_hidden = keras.layers.Dense(
            self.hidden_dim,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.GlorotUniform(),
            bias_initializer=keras.initializers.Zeros(),
            name="mlp_hidden",
        )
        self.mlp_out = keras.layers.Dense(
            self.vocab_size,
            use_bias=self.use_bias,
            kernel_initializer=keras.initializers.GlorotUniform(),
            bias_initializer=keras.initializers.Zeros(),
            name="mlp_out",
        )

        logger.debug(
            "SharedTokenDecoder: %d tokens x %d dims -> %d logits (hidden %d)",
            self.token_seq_len,
            self.token_emb_dim,
            self.vocab_size,
            self.hidden_dim,
        )

    @property
    def token_flat_dim(self) -> int:
        """``token_seq_len * token_emb_dim`` -- the expected input width.

        :return: Number of numbers in one flattened token sequence.
        :rtype: int
        """
        return self.token_seq_len * self.token_emb_dim

    def build(self, input_shape: Any) -> None:
        """Build the three ``Dense`` layers against the per-token shape.

        :param input_shape: ``(B, token_flat_dim)``.
        :type input_shape: Any
        :raises ValueError: If the input is not rank 2, or its last axis is not
            :attr:`token_flat_dim`.
        """
        if self.built:
            return
        shape = tuple(input_shape)
        if len(shape) != 2:
            raise ValueError(
                "SharedTokenDecoder expects a rank-2 (batch, token_flat_dim) "
                f"input; got shape {shape!r}"
            )
        if shape[-1] is not None and int(shape[-1]) != self.token_flat_dim:
            raise ValueError(
                f"input width {shape[-1]} does not match token_seq_len * "
                f"token_emb_dim ({self.token_seq_len} * {self.token_emb_dim} = "
                f"{self.token_flat_dim})"
            )

        token_shape = (shape[0], self.token_seq_len, self.token_emb_dim)
        hidden_shape = (shape[0], self.token_seq_len, self.hidden_dim)
        self.mlp_in.build(token_shape)
        self.mlp_hidden.build(hidden_shape)
        self.mlp_out.build(hidden_shape)

        super().build(input_shape)

    def normalize_tokens(self, token_flat: Any) -> Any:
        """Reshape to ``(B, T, D)`` and L2-normalize each token row.

        Exposed as a method so a guard can drive the scale-invariance and the
        zero-row (padding) behaviour directly, without the MLP in the way.

        :param token_flat: ``(B, token_flat_dim)`` embeddings, at whatever
            absolute scale the sampler produced.
        :return: ``(B, token_seq_len, token_emb_dim)`` rows of unit norm; an
            all-zero row stays exactly zero.
        """
        batch = keras.ops.shape(token_flat)[0]
        tokens = keras.ops.reshape(
            token_flat, (batch, self.token_seq_len, self.token_emb_dim)
        )
        # The sanctioned two-step for naming a backend dtype; `keras.backend.*`
        # is a Keras-2 residue banned across all of `src/` by
        # `tests/test_the_keras2_backend_calls_are_gone.py`.
        dtype_name = getattr(tokens.dtype, "name", None) or str(tokens.dtype)
        epsilon = _normalize_epsilon_for(self.normalize_epsilon, dtype_name)
        return keras.ops.normalize(tokens, axis=-1, order=2, epsilon=epsilon)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Decode flattened token embeddings into per-token vocabulary logits.

        :param inputs: ``(B, token_flat_dim)`` token embeddings.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the ``Dense`` sub-layers.
        :type training: Optional[bool]
        :return: ``(B, token_seq_len, vocab_size)`` logits.
        :rtype: keras.KerasTensor
        """
        x = self.normalize_tokens(inputs)
        x = self.mlp_in(x, training=training)
        x = keras.ops.gelu(x, approximate=GELU_APPROXIMATE)
        x = self.mlp_hidden(x, training=training)
        x = keras.ops.gelu(x, approximate=GELU_APPROXIMATE)
        return self.mlp_out(x, training=training)

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        """Return ``(B, token_seq_len, vocab_size)`` without building anything.

        :param input_shape: ``(B, token_flat_dim)``.
        :type input_shape: Any
        :return: The per-token logits shape.
        :rtype: Tuple[Optional[int], ...]
        """
        batch = tuple(input_shape)[0]
        return (batch, self.token_seq_len, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: A JSON-serializable configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_dim": self.hidden_dim,
                "token_seq_len": self.token_seq_len,
                "token_emb_dim": self.token_emb_dim,
                "normalize_epsilon": self.normalize_epsilon,
                "use_bias": self.use_bias,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SharedTokenDecoder":
        """Rebuild from :meth:`get_config`.

        :param config: A configuration dictionary.
        :type config: Dict[str, Any]
        :return: The reconstructed decoder.
        :rtype: SharedTokenDecoder
        """
        return cls(**config)


def create_shared_token_decoder(
    vocab_size: int,
    hidden_dim: int = 128,
    token_seq_len: int = 64,
    token_emb_dim: int = 64,
    **kwargs: Any,
) -> SharedTokenDecoder:
    """Build a :class:`SharedTokenDecoder`.

    There is no ``variant`` argument and no ``MODEL_VARIANTS`` table: the
    decoder's geometry is fully determined by the
    :class:`~.config.BridgeConfig` it reads and by the tokenizer's vocabulary,
    so a named-variant table here would be an invented axis. See
    ``models/CLAUDE.md`` "When the shape does not apply".

    :param vocab_size: Number of output logits per token position.
    :type vocab_size: int
    :param hidden_dim: Width of both hidden layers.
    :type hidden_dim: int
    :param token_seq_len: Token positions per bridge tensor.
    :type token_seq_len: int
    :param token_emb_dim: Width of one token embedding.
    :type token_emb_dim: int
    :param kwargs: Forwarded to the constructor.
    :type kwargs: Any
    :return: The configured decoder.
    :rtype: SharedTokenDecoder

    Example:
        >>> decoder = create_shared_token_decoder(vocab_size=32000)
    """
    return SharedTokenDecoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        token_seq_len=token_seq_len,
        token_emb_dim=token_emb_dim,
        **kwargs,
    )
