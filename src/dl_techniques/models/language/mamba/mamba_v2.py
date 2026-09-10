"""
Mamba-2 encoder: state-space-duality blocks with head-scalar decay and grouped B/C.

Defines :class:`Mamba2`, a stack of ``Mamba2ResidualBlock`` layers returning hidden
states. Where Mamba-1 gives every inner channel its own ``d_inner x d_state``
transition, Mamba-2 restricts the transition ``A`` to one scalar per head, so
``d_state`` can reach 128 at little cost and the unrolled recurrence becomes a
lower-triangular matrix with a scalar decay mask in place of softmax. ``B`` and ``C``
are shared across ``ngroups`` head groups and broadcast to the heads each group
serves, as in grouped-query attention, and ``z``, ``x``, ``B``, ``C`` and ``dt`` all
come from one ``in_proj`` above the convolution. Setting ``d_ssm < d_inner`` routes
the leading channels around the SSM as a gated MLP, concatenated back before the
output projection. The scan is a sequential ``while_loop`` rather than the paper's
chunked-matmul algorithm, so this is a correctness reference and not a speed
benchmark. The model returns only ``{'last_hidden_state'}``, so a task head is
attached externally; ``rmsnorm`` governs the in-block SSM-output norm while the final
norm is always ``LayerNormalization``; and each block returns
``(output, running_residual)`` with the single add in the model tail.

References:
    - Dao and Gu, 2024. Transformers are SSMs: Generalized Models and Efficient
      Algorithms Through Structured State Space Duality.
      (https://arxiv.org/abs/2405.21060)
    - Gu and Dao, 2023. Mamba: Linear-Time Sequence Modeling with Selective State
      Spaces. (https://arxiv.org/abs/2312.00752)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer Models
      from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
"""

import keras
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .components_v2 import Mamba2ResidualBlock
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.mamba.mamba_v2")
class Mamba2(keras.Model):
    """Encode token ids into hidden states with a stack of Mamba-2 blocks.

    Every constructor argument past ``pad_token_id`` is a pass-through to the blocks,
    which forward it to their ``Mamba2Layer``. The encoder has no task head; attach
    one to ``last_hidden_state``.

    Architecture:

    .. code-block:: text

        input_ids [B, L]  (tensor, or a dict under "input_ids")
                 │
                 ▼
        ┌───────────────────┐
        │ embedding         │
        └───────────────────┘
                 │  [B, L, d_model]
                 ▼
        ┌───────────────────┐
        │ mamba2_block_0    │
        └───────────────────┘
                 │
                 ▼
                ...
                 │
                 ▼
        ┌───────────────────┐
        │ mamba2_block_{n-1}│
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ final_norm        │  layernorm, always
        └───────────────────┘
                 │
                 ▼
        {"last_hidden_state": [B, L, d_model]}

    Residual wiring:

    .. code-block:: text

              hidden                residual
                │                      │
                ▼                      ▼
        ┌──────────────────────────────────┐
        │ mamba2_block_i(hidden, residual) │
        └──────────────────────────────────┘
                │                      │
                ▼                      ▼
              hidden                residual
                │                      │
                └──────────┬───────────┘
                           ▼
                          add
                           │
                           ▼
                      final_norm

    The first block receives ``residual=None`` and the add is skipped if it stays None.

    Variants:

    .. code-block:: text

        variant  d_model  num_layers  checkpoint     alias
        130m       768        24      mamba2-130m    base
        370m      1024        48      mamba2-370m
        780m      1536        48      mamba2-780m
        1.3b      2048        48      mamba2-1.3b    1.4b
        2.7b      2560        64      mamba2-2.7b    2.8b

    Checkpoints are the ``state-spaces/`` releases; the aliases are the v1 names.

    :param vocab_size: Size of the vocabulary. Must be positive.
    :param d_model: Dimensionality of the model's hidden states. Must be positive.
    :param num_layers: Number of Mamba residual blocks. Must be positive.
    :param d_state: Dimensionality of SSM latent state.
    :param d_conv: Kernel size for causal convolutions.
    :param expand: Expansion factor for internal dimensions.
    :param headdim: Dimensionality of each SSM head.
    :param norm_epsilon: Epsilon for all normalization layers.
    :param pad_token_id: ID of the padding token. Stored for callers that build a
        mask; this model does not read it.
    :param rmsnorm: If True, the in-block SSM-output norm is an RMSNorm. The model's
        final norm is a ``LayerNormalization`` either way.
    :param d_ssm: Dimensionality of the SSM. Defaults to ``d_model * expand``. A
        smaller value routes the leading ``d_inner - d_ssm`` channels around the SSM
        as a gated MLP.
    :param norm_before_gate: Forwarded to every
        :class:`~dl_techniques.models.language.mamba.components_v2.Mamba2Layer` in the
        stack; see that class for the semantics and for which checkpoints need
        ``True``.
    :param ngroups: Number of head groups sharing one ``B``/``C``. Forwarded to every
        ``Mamba2Layer`` in the stack.
    :param dt_min: Forwarded to every ``Mamba2Layer`` in the stack.
    :param dt_max: Forwarded to every ``Mamba2Layer`` in the stack.
    :param dt_init_floor: Forwarded to every ``Mamba2Layer`` in the stack.
    :param bias: Forwarded to every ``Mamba2Layer`` in the stack.
    :param conv_bias: Forwarded to every ``Mamba2Layer`` in the stack.
    :param **kwargs: Additional keyword arguments for ``keras.Model``.

    :raises ValueError: If ``vocab_size``, ``d_model`` or ``num_layers`` is not
        positive.

    Input shape:
        A 2D tensor ``(batch_size, sequence_length)`` of token IDs, or a dictionary
        holding that tensor under ``'input_ids'``.

    Output shape:
        Dictionary with ``'last_hidden_state'``, a 3D tensor
        ``(batch_size, sequence_length, d_model)``.

    Note:
        Every default here matches the corresponding `Mamba2Layer` default,
        so the default construction path is unchanged. See decisions.md
        plan-2026-08-18T140459-7991552f/D-036.
    """

    # DECISION plan-2026-08-18T140459-7991552f/D-024: shapes come from the Mamba-2
    # release configs (Dao and Gu 2024), not the Mamba-1 paper. See decisions.md.
    MODEL_VARIANTS = {
        "2.7b": {"d_model": 2560, "num_layers": 64},
        "1.3b": {"d_model": 2048, "num_layers": 48},
        "780m": {"d_model": 1536, "num_layers": 48},
        "370m": {"d_model": 1024, "num_layers": 48},
        "130m": {"d_model": 768, "num_layers": 24, "name": "base"},
    }

    # Mamba-1 series names, resolving to the v2 rows with the same d_model and
    # num_layers, so no caller silently changes model.
    VARIANT_ALIASES = {
        "base": "130m",
        "1.4b": "1.3b",
        "2.8b": "2.7b",
    }

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int,
            d_state: int = 128,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 64,
            norm_epsilon: float = 1e-5,
            pad_token_id: int = 0,
            rmsnorm: bool = True,
            d_ssm: Optional[int] = None,
            norm_before_gate: bool = False,
            ngroups: int = 1,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init_floor: float = 1e-4,
            bias: bool = False,
            conv_bias: bool = True,
            **kwargs: Any,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.norm_epsilon = norm_epsilon
        self.pad_token_id = pad_token_id
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        # DECISION plan-2026-08-18T140459-7991552f/D-036: pure pass-throughs to
        # Mamba2Layer; defaults must equal Mamba2Layer's. See decisions.md.
        self.ngroups = ngroups
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias

        d_inner = d_model * expand
        if d_ssm is None:
            d_ssm = d_inner
        self.d_ssm = d_ssm

        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=d_model, name="embedding"
        )
        self.encoder_layers = []
        for i in range(num_layers):
            block = Mamba2ResidualBlock(
                d_model=d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                headdim=self.headdim,
                d_ssm=self.d_ssm,
                rmsnorm=self.rmsnorm,
                norm_epsilon=self.norm_epsilon,
                norm_before_gate=self.norm_before_gate,
                ngroups=self.ngroups,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                dt_init_floor=self.dt_init_floor,
                bias=self.bias,
                conv_bias=self.conv_bias,
                name=f"mamba2_block_{i}",
            )
            self.encoder_layers.append(block)

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=norm_epsilon, name="final_norm"
        )

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Embed the ids, run every block, then add the residual and normalize.

        :param inputs: A tensor of token ids ``(batch, seq_len)``, or a dictionary
            holding one under ``'input_ids'``.
        :param training: Accepted for the Keras signature. It is not forwarded to the
            sub-layers, which have no training-dependent behaviour here.
        :return: Dictionary with ``'last_hidden_state'`` of shape
            ``(batch, seq_len, d_model)``.
        :raises ValueError: If a dictionary input has no ``'input_ids'`` key, or the
            ids are ``None``.
        """
        if isinstance(inputs, dict):
            if "input_ids" not in inputs:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs

        if input_ids is None:
            raise ValueError("Input 'input_ids' cannot be None.")

        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.encoder_layers:
            hidden_states, residual = layer(hidden_states, residual)

        final_residual = hidden_states + residual if residual is not None else hidden_states
        last_hidden_state = self.final_norm(final_residual)

        return {"last_hidden_state": last_hidden_state}

    @classmethod
    def from_variant(cls, variant: str, vocab_size: int, **kwargs: Any) -> "Mamba2":
        """Create a Mamba-2 model from a variant or alias name.

        The variant row sets ``d_model`` and ``num_layers``, and the ``130m`` row also
        names the model ``"base"``. Anything in ``kwargs`` overrides those values.
        ``vocab_size`` stays a caller argument, since the released checkpoints use
        50277 padded up to a multiple of 16.

        :param variant: A key of ``MODEL_VARIANTS`` or of ``VARIANT_ALIASES``.
        :param vocab_size: Size of the vocabulary.
        :param **kwargs: Additional arguments, overriding the variant's values.
        :return: A configured model.
        :rtype: Mamba2
        :raises ValueError: If ``variant`` is in neither table; the message lists both.
        """
        variant = cls.VARIANT_ALIASES.get(variant, variant)
        if variant not in cls.MODEL_VARIANTS:
            available = list(cls.MODEL_VARIANTS.keys()) + list(cls.VARIANT_ALIASES.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available}")

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)
        config["vocab_size"] = vocab_size
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument for serialization.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "num_layers": self.num_layers, "d_state": self.d_state,
            "d_conv": self.d_conv, "expand": self.expand,
            "headdim": self.headdim, "norm_epsilon": self.norm_epsilon,
            "pad_token_id": self.pad_token_id,
            "rmsnorm": self.rmsnorm,
            "d_ssm": self.d_ssm,
            "norm_before_gate": self.norm_before_gate,
            "ngroups": self.ngroups,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
            "dt_init_floor": self.dt_init_floor,
            "bias": self.bias,
            "conv_bias": self.conv_bias,
        })
        return config

# ---------------------------------------------------------------------