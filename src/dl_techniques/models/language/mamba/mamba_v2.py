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
benchmark. By default the model returns only ``{'last_hidden_state'}``; with
``include_top=True`` and a ``task_config`` it owns ONE task head from
``layers/heads/nlp``, named ``task_head``, which pools the last real token under the
input's ``attention_mask``, else ``input_ids != pad_token_id`` (all ones when it is None), and returns only that
head's output (as :class:`~dl_techniques.models.language.mamba.mamba_v1.Mamba`
does). :func:`create_mamba2` is the variant factory. ``rmsnorm`` governs the
in-block SSM-output norm while the final norm is always ``LayerNormalization``; and
each block returns
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

from dl_techniques.layers.ssm.mamba2 import Mamba2ResidualBlock
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    build_model_task_head,
    finalize_head_outputs,
    head_attention_mask,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.models.language.mamba.mamba_v1 import load_local_checkpoint
from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.mamba.mamba_v2")
class Mamba2(keras.Model):
    """Encode token ids into hidden states with a stack of Mamba-2 blocks.

    Every constructor argument from ``rmsnorm`` to ``conv_bias`` is a pass-through to
    the blocks, which forward it to their ``Mamba2Layer``. By default the encoder has
    no task head. With ``include_top=True`` it owns one, fed ``{"hidden_states":
    last_hidden_state, "attention_mask": <mask>}``, where the mask is the input dict's
    ``attention_mask`` or, without one, ``input_ids != pad_token_id`` (all ones when it is None), cast to the
    hidden-state dtype, and ``call`` returns the head's output reduced by
    :func:`~dl_techniques.layers.heads.nlp.finalize_head_outputs`. The blocks never
    see the mask, so the hidden states do not depend on the head.

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

        include_top=True: the dict is replaced by
        ┌───────────────────┐
        │ task_head         │──► finalized head output (only output)
        │ (pools 'last')    │
        └───────────────────┘

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
    :param pad_token_id: Id treated as padding by the task head when the input
        carries no ``attention_mask``. Read only with ``include_top=True``.
        Defaults to None, which derives no padding (the head sees every
        position): id 0 is a real token, so with right-padded batches pass the
        tokenizer's pad id or an ``attention_mask``.
    :param rmsnorm: If True, the in-block SSM-output norm is an RMSNorm. The model's
        final norm is a ``LayerNormalization`` either way.
    :param d_ssm: Dimensionality of the SSM. Defaults to ``d_model * expand``. A
        smaller value routes the leading ``d_inner - d_ssm`` channels around the SSM
        as a gated MLP.
    :param norm_before_gate: Forwarded to every
        :class:`~dl_techniques.layers.ssm.mamba2.Mamba2Layer` in the
        stack; see that class for the semantics and for which checkpoints need
        ``True``.
    :param ngroups: Number of head groups sharing one ``B``/``C``. Forwarded to every
        ``Mamba2Layer`` in the stack.
    :param dt_min: Forwarded to every ``Mamba2Layer`` in the stack.
    :param dt_max: Forwarded to every ``Mamba2Layer`` in the stack.
    :param dt_init_floor: Forwarded to every ``Mamba2Layer`` in the stack.
    :param bias: Forwarded to every ``Mamba2Layer`` in the stack.
    :param conv_bias: Forwarded to every ``Mamba2Layer`` in the stack.
    :param include_top: Whether the model owns a task head. Defaults to False: no
        head layer and no head weight exist.
    :param task_config: The head's task, as an ``NLPTaskConfig`` or a dict of its
        fields. Required when ``include_top`` is True, refused otherwise; stored as a
        plain dict. ``MULTIPLE_CHOICE`` is refused. Its ``vocabulary_size`` is not
        read: the vocabulary is ``vocab_size``.
    :param head_config_overrides: Extra keyword arguments for the head constructor
        (e.g. ``{"pooling_type": "mean"}``, which replaces the ``'last'`` default).
        Only valid with ``include_top=True``.
    :param **kwargs: Additional keyword arguments for ``keras.Model``.

    :raises ValueError: If ``vocab_size``, ``d_model`` or ``num_layers`` is not
        positive, if ``include_top=True`` comes without a ``task_config``, or if a
        ``task_config`` or ``head_config_overrides`` comes with
        ``include_top=False``.

    Input shape:
        A 2D tensor ``(batch_size, sequence_length)`` of token IDs, or a dictionary
        holding that tensor under ``'input_ids'`` (and optionally an
        ``'attention_mask'``, read only by a task head).

    Output shape:
        With ``include_top=False``, a dictionary with ``'last_hidden_state'``, a 3D
        tensor ``(batch_size, sequence_length, d_model)``. With ``include_top=True``,
        the head's output: a bare tensor, or a dict of two or more tensors.

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
            pad_token_id: Optional[int] = None,
            rmsnorm: bool = True,
            d_ssm: Optional[int] = None,
            norm_before_gate: bool = False,
            ngroups: int = 1,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init_floor: float = 1e-4,
            bias: bool = False,
            conv_bias: bool = True,
            include_top: bool = False,
            task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        validate_task_head_knobs(include_top, task_config, head_config_overrides)
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
        self.include_top = include_top
        self.task_config = (
            task_config_to_dict(task_config) if task_config is not None else None
        )
        self.head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides is not None else None
        )

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

        # Created after the backbone and under its own fixed name, so the
        # backbone's layer names, weight paths and RNG draws are the same with
        # or without a head.
        self.task_head = None
        if self.include_top:
            # DECISION plan-2026-08-17T183311-79c63e38/D-023: pool 'last', not 'cls'; Mamba-2 is
            # causal, and 'last' needs the attention_mask below wired in. See decisions.md.
            self.task_head = build_model_task_head(
                self.task_config,
                input_dim=self.d_model,
                head_config_overrides=self.head_config_overrides,
                default_pooling="last",
            )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method an explicit ``build`` marked the model built with no
        weight created (Keras warns "does not have a ``build()`` method"), so a
        weight load into it raised; only a forward pass built it. The shared
        helper traces ``call()`` on symbolic inputs, as in
        :class:`~dl_techniques.models.language.mamba.mamba_v1.Mamba`.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Embed the ids, run every block, then add the residual and normalize.

        :param inputs: A tensor of token ids ``(batch, seq_len)``, or a dictionary
            holding one under ``'input_ids'`` (and optionally ``'attention_mask'``,
            read only by a task head).
        :param training: Forwarded to the task head only. The backbone sub-layers
            have no training-dependent behaviour here.
        :return: With ``include_top=False``, a dictionary with
            ``'last_hidden_state'`` of shape ``(batch, seq_len, d_model)``. With
            ``include_top=True``, only the head's output reduced by
            ``finalize_head_outputs``.
        :raises ValueError: If a dictionary input has no ``'input_ids'`` key, or the
            ids are ``None``.
        """
        attention_mask = None
        if isinstance(inputs, dict):
            if "input_ids" not in inputs:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
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

        if self.include_top:
            head_outputs = self.task_head(
                {
                    "hidden_states": last_hidden_state,
                    "attention_mask": head_attention_mask(
                        input_ids, attention_mask, self.pad_token_id,
                        last_hidden_state.dtype,
                    ),
                },
                training=training,
            )
            return finalize_head_outputs(head_outputs)

        return {"last_hidden_state": last_hidden_state}

    @property
    def hidden_size(self) -> int:
        """Alias for :attr:`d_model`, for callers expecting the common name.

        Additive only: ``d_model`` is not renamed or removed, and this is not
        a constructor argument, so ``get_config``/``from_config`` are
        unaffected. See
        ``dl_techniques.models.common.masked_language_model.clm.CausalLanguageModel``,
        whose ``__init__`` requires a ``hidden_size`` attribute on any
        backbone built with ``skip_head=False``.

        :return: :attr:`d_model`.
        :rtype: int
        """
        return self.d_model

    @classmethod
    def from_variant(
            cls,
            variant: str,
            vocab_size: int,
            pretrained: Union[bool, str] = False,
            **kwargs: Any,
    ) -> "Mamba2":
        """Create a Mamba-2 model from a variant or alias name.

        The variant row sets ``d_model`` and ``num_layers``, and the ``130m`` row also
        names the model ``"base"``. Anything in ``kwargs`` overrides those values.
        ``vocab_size`` stays a caller argument, since the released checkpoints use
        50277 padded up to a multiple of 16.

        :param variant: A key of ``MODEL_VARIANTS`` or of ``VARIANT_ALIASES``.
        :param vocab_size: Size of the vocabulary.
        :param pretrained: If a string, loads weights from that local path; a model
            with a head (``include_top=True``) receives the backbone only, from a
            ``.keras`` file, and keeps its own fresh head (see
            :meth:`~dl_techniques.models.language.mamba.mamba_v1.Mamba.from_variant`).
            If True, raises ``NotImplementedError``. Defaults to False.
        :param **kwargs: Additional arguments, overriding the variant's values,
            including ``include_top``, ``task_config`` and ``head_config_overrides``.
        :return: A configured model.
        :rtype: Mamba2
        :raises ValueError: If ``variant`` is in neither table; the message lists both.
        :raises NotImplementedError: If ``pretrained is True``.
        """
        variant = cls.VARIANT_ALIASES.get(variant, variant)
        if variant not in cls.MODEL_VARIANTS:
            available = list(cls.MODEL_VARIANTS.keys()) + list(cls.VARIANT_ALIASES.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available}")
        if pretrained is True:
            # Raising keeps a caller from training on weights they think are
            # pretrained.
            raise NotImplementedError(
                f"No pretrained weights are distributed with dl_techniques "
                f"for Mamba-2 variant '{variant}'. Pass a local checkpoint "
                f"instead: Mamba2.from_variant('{variant}', "
                f"vocab_size=..., pretrained='/path/to/weights.keras'), "
                f"or use pretrained=False (default) for random init."
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)
        config["vocab_size"] = vocab_size
        model = cls(**config)
        if isinstance(pretrained, str):
            load_local_checkpoint(model, pretrained)
            logger.info(f"Loaded pretrained weights from {pretrained}")
        return model

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
            "include_top": self.include_top,
            "task_config": (
                dict(self.task_config) if self.task_config is not None else None
            ),
            "head_config_overrides": (
                dict(self.head_config_overrides)
                if self.head_config_overrides is not None else None
            ),
        })
        return config


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_mamba2(
        variant: str,
        vocab_size: int,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
) -> Mamba2:
    """Create a Mamba-2 model from a named variant or alias.

    Delegates to :meth:`Mamba2.from_variant`, whose ``vocab_size`` contract it
    keeps: the variants fix ``d_model`` and ``num_layers`` only, so the
    vocabulary is always the caller's.

    :param variant: A key of ``Mamba2.MODEL_VARIANTS`` ("2.7b", "1.3b", "780m",
        "370m", "130m") or of ``Mamba2.VARIANT_ALIASES`` ("base", "1.4b", "2.8b").
    :type variant: str
    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param pretrained: If a string, loads weights from that local path (a model
        with a head receives the backbone only). If True, raises
        ``NotImplementedError``. Defaults to False.
    :type pretrained: Union[bool, str]
    :param kwargs: Forwarded to :meth:`Mamba2.from_variant`, e.g. ``headdim``,
        or ``include_top=True`` with a ``task_config`` for a model with a task
        head.
    :type kwargs: Any
    :return: The configured model.
    :rtype: Mamba2
    :raises ValueError: If ``variant`` is unknown, or a forwarded argument is
        invalid (including inconsistent head knobs, see :class:`Mamba2`).
    :raises NotImplementedError: If ``pretrained is True``.

    Example:
        .. code-block:: python

            encoder = create_mamba2("130m", vocab_size=50288)

            # The same encoder with a sentence-classification head (pools 'last')
            clf = create_mamba2(
                "130m", vocab_size=50288, include_top=True,
                task_config=NLPTaskConfig(
                    name="sst2", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                    num_classes=2))
    """
    return Mamba2.from_variant(
        variant, vocab_size=vocab_size, pretrained=pretrained, **kwargs
    )


# ---------------------------------------------------------------------