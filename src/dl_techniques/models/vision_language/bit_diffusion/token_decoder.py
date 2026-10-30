"""
SharedTokenDecoder: reads caption tokens back out of a sampled bridge tensor.

The bridge carries a caption as raw token embeddings, and the diffusion model
never sees a vocabulary. :class:`SharedTokenDecoder` turns the image-to-text
sampler's output, unpacked by :func:`~.token_bridge.bridge_to_token_flat`, into
per-token vocabulary logits with one small MLP shared across positions, like a
language-model head. It is a separate model, outside :class:`DiTXA` and the
diffusion training loop. Each token row is L2-normalized first, so
``decode(x)`` equals ``decode(k * x)`` for any ``k > 0`` and the decoder needs no
``token_scale``. Its GELU is the exact erf form, not the tanh approximation of
the DiTXA blocks; the two differ by up to 4.7e-4.

References:
    - Upstream ``token_decoder.py``, the source of this port.
    - Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*.
      arXiv:1606.08415. Section 2 gives both the exact ``x * Phi(x)`` form used
      here and the tanh approximation used by the transformer blocks.
"""

from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np

from dl_techniques.layers.activations.exact_activations import exact_gelu
from dl_techniques.layers.activations.gelu_tanh import gelu_tanh
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import refuse_pretrained

#: ``torch.nn.functional.normalize``'s default epsilon; it keeps an all-zero
#: (padding) token row from becoming 0/0.
DEFAULT_NORMALIZE_EPSILON: float = 1e-12

# DECISION plan-2026-09-02T094601-77d4a04e/D-026: keep exact GELU, separate from blocks.py.
# The tanh form differs by up to 4.7e-4, invisible to shape and dtype checks. See decisions.md.
#: ``False`` (upstream ``torch.nn.GELU()``) selects exact erf GELU; ``True`` the tanh form.
GELU_APPROXIMATE: bool = False

#: Number of ``Dense`` layers in the shared per-token head.
NUM_MLP_LAYERS: int = 3


# DECISION plan-2026-09-02T094601-77d4a04e/D-025: floor epsilon at the dtype's smallest normal.
# In float16, 1e-12 is zero, and a padding row would divide by zero. See decisions.md.
def _normalize_epsilon_for(epsilon: float, dtype: str) -> float:
    """Raise ``epsilon`` to the smallest normal of ``dtype`` when it underflows.

    A dtype NumPy cannot read as a float (an integer or unknown name) returns
    ``epsilon`` unchanged.

    :param epsilon: The configured normalization epsilon.
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
    """Map flattened bridge token embeddings to per-token vocabulary logits.

    Each token row is L2-normalized, then one three-layer MLP runs on every
    position with shared weights. An all-zero (padding) row stays zero after
    normalization and decodes to the MLP's response to a zero input.

    .. code-block:: text

        token_flat [B, T * D]
                 │
                 ▼
        reshape [B, T, D]
                 │
                 ▼
        L2-normalize each token row
                 │  unit rows; zero rows stay zero
                 ▼
        ┌─────────────────────┐
        │ mlp_in  Dense(H)    │
        └─────────────────────┘
                 │
                 ▼
        GELU (exact erf)
                 │
                 ▼
        ┌─────────────────────┐
        │ mlp_hidden  Dense(H)│
        └─────────────────────┘
                 │
                 ▼
        GELU (exact erf)
                 │
                 ▼
        ┌─────────────────────┐
        │ mlp_out  Dense(V)   │
        └─────────────────────┘
                 │
                 ▼
        logits [B, T, V]

    T is token_seq_len, D token_emb_dim, H hidden_dim, V vocab_size.

    :param vocab_size: Number of output logits per token position. Must be
        positive.
    :type vocab_size: int
    :param hidden_dim: Width of both hidden ``Dense`` layers. Must be positive.
    :type hidden_dim: int
    :param token_seq_len: Token positions carried by one bridge tensor; must
        match :attr:`~.config.BridgeConfig.token_seq_len`, which is not
        checked. Must be positive.
    :type token_seq_len: int
    :param token_emb_dim: Width of one token embedding; must match
        :attr:`~.config.BridgeConfig.token_emb_dim`, which is not checked. Must
        be positive.
    :type token_emb_dim: int
    :param normalize_epsilon: Epsilon of the L2 normalization, floored at the
        compute dtype's smallest normal. Must be positive.
    :type normalize_epsilon: float
    :param use_bias: Whether the three ``Dense`` layers carry biases; upstream
        does (``torch.nn.Linear``'s default).
    :type use_bias: bool
    :param **kwargs: Standard ``keras.Model`` keyword arguments.

    :ivar mlp_in: ``Dense(hidden_dim)`` on the normalized tokens.
    :ivar mlp_hidden: ``Dense(hidden_dim)``.
    :ivar mlp_out: ``Dense(vocab_size)`` producing the logits.

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
        """Validate the sizes and create the three ``Dense`` layers.

        Arguments are documented on the class. Sizes are converted with
        ``int()``, so a float is truncated before the check.

        :raises ValueError: If a size argument or ``normalize_epsilon`` is not
            positive.
        """
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
        """Return ``token_seq_len * token_emb_dim``, the expected input width.

        :return: Number of numbers in one flattened token sequence.
        :rtype: int
        """
        return self.token_seq_len * self.token_emb_dim

    def build(self, input_shape: Any) -> None:
        """Build the three ``Dense`` layers on the per-token shape.

        Returns at once if already built.

        :param input_shape: ``(B, token_flat_dim)``. An unknown width is
            accepted here; a wrong one then fails in the reshape at call time.
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

        :param token_flat: ``(B, token_flat_dim)`` embeddings, at whatever
            absolute scale the sampler produced.
        :return: ``(B, token_seq_len, token_emb_dim)`` rows of unit norm; an
            all-zero row stays exactly zero.
        """
        batch = keras.ops.shape(token_flat)[0]
        tokens = keras.ops.reshape(
            token_flat, (batch, self.token_seq_len, self.token_emb_dim)
        )
        # Read the dtype name directly: keras.backend.* is banned across src/.
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
        :param training: Forwarded to the ``Dense`` sub-layers, which ignore it.
        :type training: Optional[bool]
        :return: ``(B, token_seq_len, vocab_size)`` logits.
        :rtype: keras.KerasTensor
        """
        x = self.normalize_tokens(inputs)
        # Both GELU forms use float64-exact constants (D-277) and match
        # keras.ops.gelu bit for bit at float32.
        gelu = gelu_tanh if GELU_APPROXIMATE else exact_gelu
        x = self.mlp_in(x, training=training)
        x = gelu(x)
        x = self.mlp_hidden(x, training=training)
        x = gelu(x)
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
    pretrained: bool = False,
    **kwargs: Any,
) -> SharedTokenDecoder:
    """Build a :class:`SharedTokenDecoder`.

    There is no ``variant`` argument and no ``MODEL_VARIANTS`` table: the
    decoder's geometry comes from the :class:`~.config.BridgeConfig` and the
    tokenizer's vocabulary.

    :param vocab_size: Number of output logits per token position.
    :type vocab_size: int
    :param hidden_dim: Width of both hidden layers.
    :type hidden_dim: int
    :param token_seq_len: Token positions per bridge tensor.
    :type token_seq_len: int
    :param token_emb_dim: Width of one token embedding.
    :type token_emb_dim: int
    :param pretrained: ``False`` (default) for random init. Any truthy value,
        a path string included, raises ``NotImplementedError``; restore a saved
        decoder whole with ``keras.models.load_model(path)``.
    :type pretrained: bool
    :param **kwargs: Forwarded to the constructor.
    :type **kwargs: Any
    :return: The configured decoder.
    :rtype: SharedTokenDecoder
    :raises NotImplementedError: If ``pretrained`` is truthy.
    :raises ValueError: If the constructor rejects an argument.

    Example:
        >>> decoder = create_shared_token_decoder(vocab_size=32000)
    """
    refuse_pretrained(pretrained, "SharedTokenDecoder")
    return SharedTokenDecoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        token_seq_len=token_seq_len,
        token_emb_dim=token_emb_dim,
        **kwargs,
    )