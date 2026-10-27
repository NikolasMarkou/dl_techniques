"""
Mamba (v1) selective state space encoder, optionally topped by one NLP task head.

Defines :class:`Mamba`, a stack of selective-SSM residual blocks returning hidden
states, and :func:`create_mamba`, its variant factory.
A classical state space model is linear and time-invariant, so the same decay applies
to every token; Mamba projects its discretization parameters (delta, B, C) from the
input at each position, so a large delta wipes the state and a small one holds it.
Residual addition is deferred: each block returns ``(output, running_residual)`` and
the one add happens in the model's tail, so discarding the second return value drops
every skip connection. The scan runs sequentially through ``keras.ops.while_loop``
instead of the paper's hardware-parallel scan, so this is a faithful reference rather
than a fast one. ``pretrained=True`` raises ``NotImplementedError``; pass a local
``.keras`` path instead. The embedding sets ``mask_zero=False``; the blocks never
see a mask. With ``include_top=True`` and a ``task_config`` the model owns ONE task
head from ``layers/heads/nlp``, named ``task_head``, which pools the last real token
(the only position of a causal model that has seen the whole sequence) under the
input's ``attention_mask``, else ``input_ids != pad_token_id`` (all ones when it is None), and ``call`` returns
only that head's output. The model is causal, so right-padding leaves the valid prefix
intact and left-padding does not.

References:
    - Gu and Dao, 2023. Mamba: Linear-Time Sequence Modeling with Selective State
      Spaces. (https://arxiv.org/abs/2312.00752)
    - Gu et al., 2021. Efficiently Modeling Long Sequences with Structured State
      Spaces. (https://arxiv.org/abs/2111.00396)
    - Gu et al., 2022. On the Parameterization and Initialization of Diagonal State
      Space Models. (https://arxiv.org/abs/2206.11893)
    - Fu et al., 2023. Hungry Hungry Hippos: Towards Language Modeling with State
      Space Models. (https://arxiv.org/abs/2212.14052)
    - Smith et al., 2023. Simplified State Space Layers for Sequence Modeling.
      (https://arxiv.org/abs/2208.04933)
"""

import keras
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    TASK_HEAD_NAME,
    build_model_task_head,
    finalize_head_outputs,
    head_attention_mask,
    head_output_shape,
    ids_shape_of,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
)
from dl_techniques.layers.ssm.selective_ssm import MambaResidualBlock
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.mamba.mamba_v1")
class Mamba(keras.Model):
    """Encode token ids into hidden states with a stack of selective SSM blocks.

    By default the encoder carries no task head and returns hidden states. With
    ``include_top=True`` it owns one NLP task head, fed ``{"hidden_states":
    last_hidden_state, "attention_mask": <mask>}``, where the mask is the input
    dict's ``attention_mask`` or, without one, ``input_ids != pad_token_id`` (all ones when it is None), cast
    to the hidden-state dtype; ``call`` then returns what the head returns, reduced
    by :func:`~dl_techniques.layers.heads.nlp.finalize_head_outputs`. The blocks
    never see the mask, so the hidden states do not depend on the head. Cost is linear in sequence length rather than quadratic, because the
    discretization parameters that make the state space selective are computed from
    the input instead of being fixed.

    Architecture:

    .. code-block:: text

        input_ids [B, L]  (tensor, or a dict under "input_ids")
                 │
                 ▼
        ┌───────────────────┐
        │ embedding         │  mask_zero False
        └───────────────────┘
                 │  [B, L, d_model]
                 ▼
        ┌───────────────────┐
        │ mamba_block_0     │
        └───────────────────┘
                 │
                 ▼
                ...
                 │
                 ▼
        ┌───────────────────┐
        │ mamba_block_{n-1} │
        └───────────────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ final_norm        │
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
        │ mamba_block_i(hidden, residual)  │
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

        variant  d_model  num_layers
        2.8b     2560     64
        1.4b     2048     48
        790m     1536     48
        370m     1024     48
        130m      768     24
        base      768     24

    base is an alias for 130m.

    :param vocab_size: Size of the vocabulary. Must be positive.
    :type vocab_size: int
    :param d_model: Dimensionality of the model's hidden states. Must be positive.
    :type d_model: int
    :param num_layers: Number of Mamba residual blocks to stack. Must be positive.
    :type num_layers: int
    :param d_state: Dimensionality of SSM latent state. Defaults to 16.
    :type d_state: int
    :param d_conv: Kernel size for causal convolutions. Defaults to 4.
    :type d_conv: int
    :param expand: Expansion factor for internal dimensions. Defaults to 2.
    :type expand: int
    :param dt_rank: Rank for step size projection. 'auto' uses ceil(d_model/16).
        Defaults to "auto".
    :type dt_rank: Union[str, int]
    :param norm_epsilon: Epsilon for all normalization layers. Defaults to 1e-5.
    :type norm_epsilon: float
    :param pad_token_id: Id treated as padding by the task head when the input
        carries no ``attention_mask``. Read only with ``include_top=True``.
        Defaults to None, which derives no padding (the head sees every
        position): id 0 is a real token, so with right-padded batches pass the
        tokenizer's pad id or an ``attention_mask``.
    :type pad_token_id: Optional[int]
    :param include_top: Whether the model owns a task head. Defaults to False: no
        head layer and no head weight exist.
    :type include_top: bool
    :param task_config: The head's task, as an ``NLPTaskConfig`` or a dict of its
        fields. Required when ``include_top`` is True, refused otherwise; stored as
        a plain dict. ``MULTIPLE_CHOICE`` is refused. Its ``vocabulary_size`` is not
        read: the vocabulary is ``vocab_size``.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Extra keyword arguments for the head constructor
        (e.g. ``{"pooling_type": "mean"}``, which replaces the ``'last'`` default).
        Only valid with ``include_top=True``. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param **kwargs: Additional keyword arguments for Model base class.

    Input shape:
        A 2D tensor ``(batch_size, sequence_length)`` of token IDs, or a dictionary
        holding that tensor under ``'input_ids'``.

    Output shape:
        With ``include_top=False``, a dictionary containing:
        - 'last_hidden_state': 3D tensor (batch_size, sequence_length, d_model)
        With ``include_top=True``, the head's output: a bare tensor, or a dict of
        two or more tensors (``start_logits`` / ``end_logits`` for QA).

    :ivar embedding: Token embedding layer.
    :vartype embedding: keras.layers.Embedding
    :ivar encoder_layers: List of MambaResidualBlock layers.
    :vartype encoder_layers: List[MambaResidualBlock]
    :ivar final_norm: Final layer normalization.
    :vartype final_norm: keras.layers.LayerNormalization
    :ivar task_head: The task head named ``"task_head"``, or None when
        ``include_top`` is False.
    :vartype task_head: Optional[dl_techniques.layers.heads.nlp.base.BaseNLPHead]

    :raises ValueError: If ``vocab_size``, ``d_model`` or ``num_layers`` is not
        positive, if ``include_top=True`` comes without a ``task_config``, or if a
        ``task_config`` or ``head_config_overrides`` comes with
        ``include_top=False``.

    Example:
        .. code-block:: python

            # Create a base Mamba model
            model = Mamba.from_variant("base", vocab_size=50257)

            # Custom configuration
            model = Mamba(
                vocab_size=50257,
                d_model=1024,
                num_layers=32,
                d_state=16,
                expand=2
            )

            # Use the model
            inputs = {
                "input_ids": keras.random.randint(
                    (2, 512), 0, 50257, dtype="int32"
                )
            }
            outputs = model(inputs)
            hidden_states = outputs["last_hidden_state"]  # (2, 512, 1024)

            # The same encoder with a sentence-classification head
            clf = Mamba(
                vocab_size=50257, d_model=1024, num_layers=32,
                include_top=True,
                task_config={"name": "sst2",
                             "task_type": "text_classification",
                             "num_classes": 2},
            )
            logits = clf(inputs)  # (2, 2), pooled at the last real token

    Note:
        Unlike BERT, Mamba doesn't use positional embeddings or token type
        embeddings - all positional information is captured implicitly through
        the causal convolutions and recurrent state space mechanism.
    """

    # Matches Gu and Dao 2023, Table 9. Layer counts run double the GPT-3
    # equivalents — one Mamba block replaces an attention+MLP pair.
    MODEL_VARIANTS = {
        "2.8b": {
            "d_model": 2560,
            "num_layers": 64,
            "description": "Mamba-2.8B: Largest variant with 2.8B parameters"
        },
        "1.4b": {
            "d_model": 2048,
            "num_layers": 48,
            "description": "Mamba-1.4B: Large variant with ~1.4B parameters"
        },
        "790m": {
            "d_model": 1536,
            "num_layers": 48,
            "description": "Mamba-790M: Medium variant with ~790M parameters"
        },
        "370m": {
            "d_model": 1024,
            "num_layers": 48,
            "description": "Mamba-370M: Small variant with ~370M parameters"
        },
        "130m": {
            "d_model": 768,
            "num_layers": 24,
            "description": "Mamba-130M: Base variant with ~130M parameters"
        },
        "base": {
            "d_model": 768,
            "num_layers": 24,
            "description": "Mamba-Base: Alias for 130M variant"
        },
    }

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[str, int] = "auto",
        norm_epsilon: float = 1e-5,
        pad_token_id: Optional[int] = None,
        include_top: bool = False,
        task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        validate_task_head_knobs(include_top, task_config, head_config_overrides)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.norm_epsilon = norm_epsilon
        self.pad_token_id = pad_token_id
        self.include_top = include_top
        self.task_config = (
            task_config_to_dict(task_config, vocab_size=vocab_size)
            if task_config is not None else None
        )
        self.head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides is not None else None
        )

        # Padding never reaches the blocks; only a task head reads a mask.
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            mask_zero=False,
            name="embedding"
        )

        self.encoder_layers = []
        for i in range(num_layers):
            block = MambaResidualBlock(
                d_model=d_model,
                norm_epsilon=norm_epsilon,
                mamba_kwargs={
                    "d_state": self.d_state,
                    "d_conv": self.d_conv,
                    "expand": self.expand,
                    "dt_rank": self.dt_rank,
                    "layer_idx": i,
                },
                name=f"mamba_block_{i}"
            )
            self.encoder_layers.append(block)

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=norm_epsilon,
            name="final_norm"
        )

        # Created after the backbone and under its own fixed name, so the
        # backbone's layer names, weight paths and RNG draws are the same with
        # or without a head.
        self.task_head = None
        if self.include_top:
            # DECISION plan-2026-08-17T183311-79c63e38/D-023: pool 'last', not 'cls'; Mamba is
            # causal, and 'last' needs the attention_mask below wired in. See decisions.md.
            self.task_head = build_model_task_head(
                self.task_config,
                input_dim=self.d_model,
                head_config_overrides=self.head_config_overrides,
                vocab_size=self.vocab_size,
                default_pooling="last",
            )

        logger.info(
            f"Created Mamba foundation model: {self.num_layers} layers, "
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"vocab_size={self.vocab_size}"
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method Mamba inherits ``Layer.build``, which marks the model
        built while its sub-layers are still unbuilt, and Keras warns about it. The
        shared helper traces ``call()`` on symbolic inputs, so what gets built matches
        what gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Compute the shapes of what ``call`` returns.

        Keras builds a symbolic call of this model from this method, never
        from ``call``, so it must report what ``call`` returns in both modes.

        :param input_shape: Shape of ``input_ids`` ``(batch, seq_len)``, or a
            dict holding it under ``'input_ids'``.
        :type input_shape: Any
        :return: Headless, a dictionary with ``last_hidden_state`` (the only
            key ``call`` returns). With a head, the head's output shapes after the
            reduction ``call`` applies: one shape, or a mapping of two or more.
        :rtype: Any
        """
        ids_shape = ids_shape_of(input_shape)
        if self.include_top:
            return head_output_shape(
                self.task_head, ids_shape, self.d_model)
        return {"last_hidden_state": (*ids_shape, self.d_model)}

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Embed the ids, run every block, then add the residual and normalize.

        :param inputs: Either a tensor of input IDs or a dictionary containing
            'input_ids' (and optionally 'attention_mask', read only by a task
            head). Shape: (batch_size, sequence_length).
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether in training mode. Defaults to None.
        :type training: Optional[bool]
        :return: With ``include_top=False``, a dictionary with 'last_hidden_state'
            key containing the final hidden states of shape (batch_size,
            sequence_length, d_model). With ``include_top=True``, only the head's
            output reduced by ``finalize_head_outputs``.
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :raises ValueError: If a dictionary input has no 'input_ids' key.
        """
        attention_mask = None
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask")
        else:
            input_ids = inputs

        hidden_states = self.embedding(input_ids, training=training)

        residual = None
        for layer in self.encoder_layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                training=training
            )

        final_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )
        last_hidden_state = self.final_norm(final_residual, training=training)

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
        unaffected. Mirrors
        ``dl_techniques.models.language.mamba.mamba_v2.Mamba2.hidden_size``
        (plan-2026-09-12T195532-422091c3/D-009). See
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
    ) -> "Mamba":
        """Create a Mamba model from a predefined variant.

        The variant sets ``d_model`` and ``num_layers`` to the paper's values;
        anything in ``kwargs`` overrides the rest of the defaults.

        :param variant: Name of the variant. One of: "2.8b", "1.4b", "790m",
            "370m", "130m", "base".
        :type variant: str
        :param vocab_size: Size of the vocabulary. Must be specified.
        :type vocab_size: int
        :param pretrained: If a string, loads weights from that local path. If
            True, raises `NotImplementedError` — no public checkpoints ship with
            this package. Defaults to False. Headed or headless, the model
            receives the backbone only: a ``task_head`` in the checkpoint is
            never copied and the model's own head keeps its fresh
            initialization, so a checkpoint of any Mamba, headless or trained
            under any head, warm-starts a model with any other head or none.
            Every backbone layer loads, and every checkpoint layer finds its
            place, or the call raises (another depth, deeper or shallower, or
            ``d_model`` is refused); only the ``embedding`` table stays fresh,
            and only when the checkpoint's own ``vocab_size`` differs. Needs a
            ``.keras`` file (see :func:`load_local_checkpoint`). To restore a whole model, head
            included, use ``keras.models.load_model``.
        :type pretrained: Union[bool, str]
        :param **kwargs: Additional arguments to override variant defaults,
            including ``include_top``, ``task_config`` and
            ``head_config_overrides``.
        :return: A Mamba model instance configured for the specified variant.
        :rtype: Mamba
        :raises ValueError: If ``variant`` is unknown, or a resolved argument is
            invalid.
        :raises NotImplementedError: If ``pretrained is True``.

        Example:
            .. code-block:: python

                # Create base model
                model = Mamba.from_variant("base", vocab_size=50257)

                # Create large model with custom parameters
                model = Mamba.from_variant(
                    "1.4b",
                    vocab_size=50257,
                    d_state=32,  # Override default
                    expand=3     # Override default
                )

                # Load from weights file
                model = Mamba.from_variant(
                    "base",
                    vocab_size=50257,
                    pretrained="path/to/weights.keras"
                )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")

        logger.info(f"Creating Mamba-{variant.upper()} model")
        logger.info(f"Configuration: {description}")

        config.update(kwargs)
        config["vocab_size"] = vocab_size

        model = cls(**config)

        if pretrained:
            if isinstance(pretrained, str):
                try:
                    load_local_checkpoint(model, pretrained)
                    logger.info(f"Loaded pretrained weights from {pretrained}")
                except Exception as e:
                    logger.error(f"Failed to load weights: {e}")
                    raise
            elif pretrained is True:
                # Raising keeps a caller from training on weights they think are
                # pretrained.
                raise NotImplementedError(
                    f"No pretrained weights are distributed with dl_techniques "
                    f"for Mamba variant '{variant}'. Pass a local checkpoint "
                    f"instead: Mamba.from_variant('{variant}', "
                    f"vocab_size=..., pretrained='/path/to/weights.keras'), "
                    f"or use pretrained=False (default) for random init."
                )

        return model

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "dt_rank": self.dt_rank,
            "norm_epsilon": self.norm_epsilon,
            "pad_token_id": self.pad_token_id,
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Mamba":
        """Create model instance from configuration.

        :param config: Dictionary containing model configuration.
        :type config: Dict[str, Any]
        :return: New Mamba model instance.
        :rtype: Mamba
        """
        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """Print the Keras summary, then log the state space settings.

        :param **kwargs: Additional arguments passed to keras.Model.summary.
        """
        super().summary(**kwargs)
        logger.info("Mamba Foundation Model Configuration:")
        logger.info(
            f"  - Architecture: {self.num_layers} layers, "
            f"{self.d_model} hidden size"
        )
        logger.info(f"  - State space: d_state={self.d_state}")
        logger.info(
            f"  - Convolution: kernel_size={self.d_conv}, "
            f"expand={self.expand}"
        )
        logger.info(f"  - Vocabulary: {self.vocab_size} tokens")
        logger.info(
            f"  - Internal dimension: {int(self.expand * self.d_model)}"
        )

# ---------------------------------------------------------------------
# Pretrained loading and factory
# ---------------------------------------------------------------------


def load_local_checkpoint(model: keras.Model, path: str) -> None:
    """Restore a Mamba or Mamba-2 model from a local checkpoint file.

    Interface contract (call sites: ``Mamba.from_variant`` and
    ``Mamba2.from_variant``): ``model`` is a freshly constructed ``Mamba`` or
    ``Mamba2``, built here if it is not built yet (by ``build``, so no RNG is
    drawn and no forward pass runs). The backbone is copied by layer name
    through the shared language load rule
    (:func:`dl_techniques.utils.weight_transfer.load_backbone_from_checkpoint`):
    every backbone layer loads and every checkpoint layer finds its place, or
    this raises (a deeper checkpoint is refused, never truncated); the
    checkpoint's ``task_head`` is never copied. The one exception is the
    ``embedding`` table when the checkpoint's own ``vocab_size`` differs from
    the model's: it then keeps its fresh initialization. Returns nothing;
    every failure raises.

    :param model: The model to restore into.
    :type model: keras.Model
    :param path: Path to a ``.keras`` checkpoint.
    :type path: str
    :raises ValueError: If the file is not ``.keras`` or the checkpoint does
        not fit the backbone.
    :raises FileNotFoundError: If the checkpoint does not exist.
    """
    if not model.built:
        # A build, not a dummy forward: weight shapes do not depend on the
        # length, and nothing reads the RNG. HEAD loaded into the unbuilt model,
        # which Keras refuses, so this route raised for every model.
        model.build((None, None))
    # DECISION plan-2026-10-11T004932-b83935e1/D-013: by-name and strict; do NOT send a headed
    # model through whole-file load_weights (raises on a headless checkpoint, copies an equal-shape
    # head) nor pass strict=False (a backbone that does not fit would load partly or not at all,
    # and return normally). See decisions.md.
    # DECISION plan-2026-10-11T004932-b83935e1/D-025: the embedding exemption is keyed on the CHECKPOINT's
    # vocab_size (inside the shared rule); do NOT pass skip_mismatch=True to a whole-file load for it --
    # that also returns normally for a deeper model with fresh blocks. See decisions.md.
    # DECISION plan-2026-10-11T004932-b83935e1/D-031: by name for a headless model too; do NOT go back
    # to whole-file load_weights there -- Keras ignores a deeper checkpoint's extra blocks. See decisions.md.
    # DECISION plan-2026-10-11T004932-b83935e1/D-007: skip the checkpoint's task_head; do NOT
    # drop TASK_HEAD_NAME from this list -- a head of another task raises on shape, one of equal shape is copied silently. See decisions.md.
    load_backbone_from_checkpoint(
        model,
        path,
        head_prefixes=(TASK_HEAD_NAME,),
        vocab_layers=("embedding",),
        tables=("embedding/embeddings",),
    )


def create_mamba(
        variant: str,
        vocab_size: int,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
) -> Mamba:
    """Create a Mamba model from a named variant.

    Delegates to :meth:`Mamba.from_variant`, whose ``vocab_size`` contract it
    keeps: the variants fix ``d_model`` and ``num_layers`` only, so the
    vocabulary is always the caller's.

    :param variant: One of ``Mamba.MODEL_VARIANTS``: "2.8b", "1.4b", "790m",
        "370m", "130m", "base".
    :type variant: str
    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param pretrained: If a string, loads weights from that local path (a model
        with a head receives the backbone only). If True, raises
        ``NotImplementedError``. Defaults to False.
    :type pretrained: Union[bool, str]
    :param kwargs: Forwarded to :meth:`Mamba.from_variant`, e.g. ``d_state``,
        or ``include_top=True`` with a ``task_config`` for a model with a task
        head.
    :type kwargs: Any
    :return: The configured model.
    :rtype: Mamba
    :raises ValueError: If ``variant`` is unknown, or a forwarded argument is
        invalid (including inconsistent head knobs, see :class:`Mamba`).
    :raises NotImplementedError: If ``pretrained is True``.

    Example:
        .. code-block:: python

            encoder = create_mamba("130m", vocab_size=50257)

            # The same encoder with a sentence-classification head (pools 'last')
            clf = create_mamba(
                "130m", vocab_size=50257, include_top=True,
                task_config=NLPTaskConfig(
                    name="sst2", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                    num_classes=2))
    """
    return Mamba.from_variant(
        variant, vocab_size=vocab_size, pretrained=pretrained, **kwargs
    )

# ---------------------------------------------------------------------