"""
Mamba (v1) selective state space encoder, and a factory that attaches an NLP head.

Defines :class:`Mamba`, a stack of selective-SSM residual blocks returning hidden
states, and :func:`create_mamba_with_head`, which wires that encoder to a task head.
A classical state space model is linear and time-invariant, so the same decay applies
to every token; Mamba projects its discretization parameters (delta, B, C) from the
input at each position, so a large delta wipes the state and a small one holds it.
Residual addition is deferred: each block returns ``(output, running_residual)`` and
the one add happens in the model's tail, so discarding the second return value drops
every skip connection. The scan runs sequentially through ``keras.ops.while_loop``
instead of the paper's hardware-parallel scan, so this is a faithful reference rather
than a fast one. ``pretrained=True`` raises ``NotImplementedError``; pass a local
``.keras`` path instead. The embedding sets ``mask_zero=False`` and
:func:`create_mamba_with_head` builds the mask from ``input_ids != pad_token_id``.
The model is causal, so right-padding leaves the valid prefix intact and left-padding
does not.

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
from dl_techniques.layers.heads.nlp import NLPTaskConfig, create_nlp_head
from dl_techniques.utils.model_build import materialize_sublayers
from .components import MambaResidualBlock
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.mamba.mamba_v1")
class Mamba(keras.Model):
    """Encode token ids into hidden states with a stack of selective SSM blocks.

    The encoder carries no task head; combine it with one the way BERT is used
    elsewhere in this codebase, or call :func:`create_mamba_with_head` for the common
    case. Cost is linear in sequence length rather than quadratic, because the
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
    :param pad_token_id: ID of padding token, used by
        :func:`create_mamba_with_head` to build the mask. Defaults to 0.
    :type pad_token_id: int
    :param **kwargs: Additional keyword arguments for Model base class.

    Input shape:
        A 2D tensor ``(batch_size, sequence_length)`` of token IDs, or a dictionary
        holding that tensor under ``'input_ids'``.

    Output shape:
        Dictionary containing:
        - 'last_hidden_state': 3D tensor (batch_size, sequence_length, d_model)

    :ivar embedding: Token embedding layer.
    :vartype embedding: keras.layers.Embedding
    :ivar encoder_layers: List of MambaResidualBlock layers.
    :vartype encoder_layers: List[MambaResidualBlock]
    :ivar final_norm: Final layer normalization.
    :vartype final_norm: keras.layers.LayerNormalization

    :raises ValueError: If ``vocab_size``, ``d_model`` or ``num_layers`` is not
        positive.

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

            # Add a task head (e.g., language modeling)
            lm_head = keras.layers.Dense(vocab_size, name="lm_head")
            logits = lm_head(hidden_states)

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
        pad_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.norm_epsilon = norm_epsilon
        self.pad_token_id = pad_token_id

        # Padding is handled by the mask create_mamba_with_head builds, not here.
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

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Embed the ids, run every block, then add the residual and normalize.

        :param inputs: Either a tensor of input IDs or a dictionary containing
            'input_ids'. Shape: (batch_size, sequence_length).
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether in training mode. Defaults to None.
        :type training: Optional[bool]
        :return: Dictionary with 'last_hidden_state' key containing the final
            hidden states of shape (batch_size, sequence_length, d_model).
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If a dictionary input has no 'input_ids' key.
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
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

        return {"last_hidden_state": last_hidden_state}

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
            this package. Defaults to False.
        :type pretrained: Union[bool, str]
        :param **kwargs: Additional arguments to override variant defaults.
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
                    model.load_weights(pretrained)
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
# Integration with NLP Task Heads
# ---------------------------------------------------------------------

def create_mamba_with_head(
        mamba_variant: str,
        task_config: NLPTaskConfig,
        pretrained: Union[bool, str] = False,
        mamba_config_overrides: Optional[Dict[str, Any]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """Build an end-to-end model: a Mamba encoder plus an NLP task head.

    Takes a variant name, instantiates the encoder, builds a head from the
    ``dl_techniques.layers.heads.nlp`` factory, and joins them into one functional
    ``keras.Model``. The only input is ``input_ids``; the padding mask is derived
    here from ``pad_token_id``, since Mamba uses neither an attention mask nor token
    type ids of its own. The head pools the last position by default, which
    ``head_config_overrides`` can change.

    .. code-block:: text

        {"input_ids": [B, L] int32}
                 │
                 ├─────────────────────┐
                 ▼                     ▼
        ┌───────────────────┐     input_ids != pad_token_id
        │ Mamba encoder     │          │
        └───────────────────┘          │
                 │ last_hidden_state   │
                 ▼                     ▼
        ┌─────────────────────────────────┐
        │ nlp head  pooling_type 'last'   │
        └─────────────────────────────────┘
                 │
                 ▼
            task outputs

    :param mamba_variant: The Mamba variant to use (e.g., "130m", "base").
    :type mamba_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task, which must set
        ``vocabulary_size``.
    :type task_config: NLPTaskConfig
    :param pretrained: If a string, path to a local weights file. If True, raises
        `NotImplementedError`. Defaults to False.
    :type pretrained: Union[bool, str]
    :param mamba_config_overrides: Optional dictionary to override default Mamba
        configuration for the chosen variant. Defaults to None.
    :type mamba_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary to override default head
        configuration, including ``pooling_type``. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete `keras.Model` ready for the specified task.
    :rtype: keras.Model
    :raises ValueError: If ``task_config`` has no ``vocabulary_size``, or the variant
        is unknown.
    :raises NotImplementedError: If ``pretrained is True``.

    Example:
        .. code-block:: python

            from dl_techniques.layers.heads.nlp import NLPTaskType

            # Define a task for sequence classification
            seq_cls_task = NLPTaskConfig(
                name="sentiment_analysis",
                task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=3,
                vocabulary_size=50257  # Mamba needs the vocabulary at creation
            )

            # Create the full model with a Mamba-130m encoder
            model = create_mamba_with_head(
                mamba_variant="130m",
                task_config=seq_cls_task,
                pretrained=False, # No public weights yet
                head_config_overrides={"dropout_rate": 0.15}
            )
            model.summary()
    """
    mamba_config_overrides = mamba_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(
        f"Creating Mamba-{mamba_variant} with a '{task_config.name}' head."
    )

    # NLPTaskConfig's field is `vocabulary_size`, not `vocab_size`.
    if not getattr(task_config, 'vocabulary_size', None):
        raise ValueError(
            "The `task_config` must set 'vocabulary_size' "
            "to create a Mamba model."
        )

    mamba_encoder = Mamba.from_variant(
        mamba_variant,
        vocab_size=task_config.vocabulary_size,
        pretrained=pretrained,
        **mamba_config_overrides,
    )

    # DECISION plan-2026-08-17T183311-79c63e38/D-023: pool 'last', not 'cls'; Mamba is
    # causal, and 'last' needs the attention_mask below wired in. See decisions.md.
    head_kwargs = {'pooling_type': 'last'}
    head_kwargs.update(head_config_overrides)
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=mamba_encoder.d_model,
        **head_kwargs,
    )

    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        ),
    }

    encoder_outputs = mamba_encoder(inputs)

    attention_mask = keras.ops.not_equal(
        inputs["input_ids"], mamba_encoder.pad_token_id
    )

    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": attention_mask,
    }
    task_outputs = task_head(head_inputs)

    model_name = f"mamba_{mamba_variant}_with_{task_config.name}_head"
    model = keras.Model(
        inputs=inputs,
        outputs=task_outputs,
        name=model_name
    )

    logger.info(
        f"Successfully created model with {model.count_params():,} parameters."
    )
    return model

# ---------------------------------------------------------------------