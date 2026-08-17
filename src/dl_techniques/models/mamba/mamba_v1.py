"""
Mamba (v1) selective state-space encoder producing contextual hidden states, plus a
factory that joins it to any NLP task head.

Attention buys unrestricted token-to-token routing at a price that is quadratic in
sequence length; a recurrent state buys `O(L)` at the price of squeezing the entire
past through a fixed-size vector. Classical structured state-space models take the
recurrent side of that trade and make it work by being linear and time-*invariant*:

`h'(t) = A h(t) + B x(t)`,  `y(t) = C h(t) + D x(t)`

discretized with a step `Δ` into `h_k = Ā h_{k-1} + B̄ x_k`, `y_k = C h_k + D x_k`.
Time-invariance is exactly what makes such a model fast — with fixed `Ā, B̄` the
whole recurrence is a convolution and can be evaluated by FFT — and it is also
exactly what makes it weak on language. A system whose dynamics do not depend on
what it is reading cannot decide to remember one token and discard the next; it
applies the same decay to everything.

Mamba's selection mechanism gives up time-invariance to buy that decision. `Δ`, `B`
and `C` are projected from the input at every position, so the dynamics are
re-chosen per token: a large `Δ` drives `exp(ΔA)` toward zero and wipes the state
(reset on a delimiter), a small `Δ` drives it toward one and holds the state across
irrelevant filler. `A` and `D` remain input-independent, which is what keeps the
recurrence a *linear* system for any fixed input and therefore keeps it cheap and
stable; the content-dependence enters only through the coefficients.

Discretization here follows the reference implementation's asymmetric choice:
`Ā = exp(ΔA)` is the exact zero-order-hold solution, while `B̄x` is formed as the
first-order `Δ · B · x` rather than the exact `(ΔA)^-1 (exp(ΔA) - I) ΔB`. The
approximation is harmless because `B` is itself a learned function of the input and
can absorb the difference. `A` is stored as `A_log` and used as `-exp(A_log)`, so
every eigenvalue is negative by construction and `exp(ΔA)` lies in `(0, 1)` for any
positive `Δ` — the recurrence cannot diverge no matter what the selection network
emits. The S4D-real initialization fills row `d` of `A` with `[1, 2, ..., d_state]`,
giving each state channel a distinct decay rate; `dt_proj`'s bias is initialized to
the inverse-softplus of a log-uniform draw in `[dt_min, dt_max]`, so channels also
start with a spread of timescales rather than all forgetting at the same rate.

**The scan in this implementation is sequential, not parallel.** `_selective_scan`
runs a `keras.ops.while_loop` over the time axis, one `scatter_update` per step.
That is `O(L)` in arithmetic, as the paper promises, but it has no parallelism
across time and no fused kernel, so wall-clock training throughput is far below
attention at the sequence lengths where Mamba's asymptotics ought to win. The
hardware-aware parallel scan the paper relies on is not implemented here; earlier
revisions of this docstring claimed a "hardware-optimized selective scan" and were
wrong. Treat this package as an architecturally faithful reference, not a
performance one.

A block projects to `expand * d_model`, splits into a signal path `x` and a gate
path `z`, runs `x` through a depthwise `Conv1D` with `padding='causal'` and SiLU,
computes `Δ, B, C` from the convolved signal, scans, adds the `D * x` skip, and
gates by `silu(z)` before projecting back. Causality is structural rather than
enforced: the convolution is causal and the recurrence only ever reads `h_{k-1}`,
so there is no attention mask to get wrong and no way for a future token to leak
backwards. For the same reason there are no positional embeddings — order is
carried by the recurrence itself.

Residual addition is deferred rather than performed inside the block. Each
`MambaResidualBlock` returns `(mamba_output, running_residual)` and adds the
*previous* residual before normalizing, with the final addition done once in the
model's tail. This mirrors the reference implementation's fused add-norm and keeps
the residual stream unnormalized end to end. The consequence for anyone composing
blocks by hand is sharp: calling `block(x)` and discarding the second return value
produces a network with no skip connections at all, and it will still run.

The embedding is created with `mask_zero=False` deliberately. Nothing in the
encoder consumes a padding mask — the recurrence has no notion of an ignorable
position — so `create_mamba_with_head` builds the mask from `input_ids !=
pad_token_id` at the boundary and hands it to the task head, which is the only
component that can act on it (masked pooling). Padded positions still update the
state, so a batch's results depend on its padding; right-padding a causal model
leaves the valid prefix intact, left-padding does not.

Two things were wrong here until 2026-08-14 and are worth stating as fixed rather
than leaving a reader to re-derive. `pretrained=True` used to log a warning and
return a randomly initialized model; it now raises `NotImplementedError`, because no
public Mamba checkpoints ship with `dl_techniques` and a caller who asks for trained
weights must not silently receive untrained ones. Pass a local `.keras` path to
`pretrained` instead. And `MODEL_VARIANTS` did not reproduce the paper's size table:
370M carried 24 layers instead of 48, 790M carried `d_model` 1024 instead of 1536 and
1.4B carried 1536 instead of 2048, so three of the six rows built a model
substantially smaller than the parameter count in its own name. The table now matches
Gu and Dao 2023: 130M 768x24, 370M 1024x48, 790M 1536x48, 1.4B 2048x48, 2.8B
2560x64.

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
from .components import MambaResidualBlock


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Mamba(keras.Model):
    """
    Mamba (v1) foundation model for efficient sequence modeling.

    This is a complete Mamba model implementing the selective state space
    architecture described in "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces". It provides a pure encoder that produces
    contextual representations, separating the core architecture from any
    task-specific layers.

    The Mamba architecture achieves linear-time complexity O(L) compared to
    quadratic O(L²) for attention-based models, while maintaining competitive
    or superior performance on long-range dependency tasks. The key innovation
    is the selective state space mechanism where the discretization parameters
    are computed from input data, allowing the model to selectively filter
    and propagate information.

    **Intent**:
    Provide an efficient foundational model for sequence modeling that can be
    easily adapted for various tasks (language modeling, classification, etc.)
    by adding task-specific heads, similar to how BERT is used in dl_techniques.

    **Architecture Overview**:

    .. code-block:: text

        Input (token IDs)
               │
               ▼
        Token Embedding
               │
               ▼
        MambaResidualBlock₁
               │
               ▼
              ...
               │
               ▼
        MambaResidualBlockₙ
               │
               ▼
        Final LayerNorm
               │
               ▼
        Output (hidden states)

    **Key Features**:
    - Linear-time complexity: O(BLD²) vs O(BL²D) for attention
    - Selective state space: Data-dependent state transitions
    - Hardware-efficient: Optimized for modern accelerators
    - Long-range modeling: Effective on sequences up to 1M tokens
    - Modular design: Easy to extend with task-specific heads

    :param vocab_size: Size of the vocabulary. Must be specified.
    :type vocab_size: int
    :param d_model: Dimensionality of the model's hidden states.
    :type d_model: int
    :param num_layers: Number of Mamba residual blocks to stack.
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
    :param pad_token_id: ID of padding token. Defaults to 0.
    :type pad_token_id: int
    :param kwargs: Additional keyword arguments for Model base class.

    Input shape:
        Dictionary containing:
        - 'input_ids': 2D tensor (batch_size, sequence_length) with token IDs

    Output shape:
        Dictionary containing:
        - 'last_hidden_state': 3D tensor (batch_size, sequence_length, d_model)

    :ivar embedding: Token embedding layer.
    :vartype embedding: keras.layers.Embedding
    :ivar encoder_layers: List of MambaResidualBlock layers.
    :vartype encoder_layers: List[MambaResidualBlock]
    :ivar final_norm: Final layer normalization.
    :vartype final_norm: keras.layers.LayerNormalization

    :raises ValueError: If vocab_size is not provided or invalid parameters.

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

    # Model variants following the original Mamba paper's size table (Gu and Dao
    # 2023, Table 9). Corrected 2026-08-14: 370m carried 24 layers, 790m carried
    # d_model 1024 and 1.4b carried d_model 1536, so three of the six rows were
    # smaller than the parameter count in their own name. The layer counts are
    # double the GPT-3 equivalents by design — one Mamba block replaces an
    # attention+MLP pair, so 130M is 24 blocks where GPT-3 small is 12 layers.
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

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.norm_epsilon = norm_epsilon
        self.pad_token_id = pad_token_id

        # CREATE sub-layers in __init__
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            mask_zero=False,  # Mamba handles padding differently than attention models
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

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Forward pass through the Mamba model.

        :param inputs: Either a tensor of input IDs or a dictionary containing
            'input_ids'. Shape: (batch_size, sequence_length).
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether in training mode. Defaults to None.
        :type training: Optional[bool]
        :return: Dictionary with 'last_hidden_state' key containing the final
            hidden states of shape (batch_size, sequence_length, d_model).
        :rtype: Dict[str, keras.KerasTensor]
        :raises ValueError: If input_ids is not provided.
        """
        # Handle both tensor and dictionary inputs
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
        else:
            input_ids = inputs

        # Token embedding
        hidden_states = self.embedding(input_ids, training=training)

        # Process through Mamba blocks with residual connections
        residual = None
        for layer in self.encoder_layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                training=training
            )

        # Final residual addition and normalization
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
        """
        Create a Mamba model from a predefined variant.

        This factory method instantiates a model with architecture parameters
        matching the original Mamba paper's specifications. Additional parameters
        can be provided to override defaults.

        :param variant: Name of the variant. One of: "2.8b", "1.4b", "790m",
            "370m", "130m", "base".
        :type variant: str
        :param vocab_size: Size of the vocabulary. Must be specified.
        :type vocab_size: int
        :param pretrained: If a string, loads weights from that local path. If
            True, raises `NotImplementedError` — no public checkpoints ship with
            this package. Defaults to False.
        :type pretrained: Union[bool, str]
        :param kwargs: Additional arguments to override variant defaults.
        :return: A Mamba model instance configured for the specified variant.
        :rtype: Mamba
        :raises ValueError: If unknown variant or invalid parameters.
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

        # Merge variant config with user overrides
        config.update(kwargs)
        config["vocab_size"] = vocab_size

        # Create model
        model = cls(**config)

        # Load pretrained weights if specified
        if pretrained:
            if isinstance(pretrained, str):
                # Load from file path
                try:
                    model.load_weights(pretrained)
                    logger.info(f"Loaded pretrained weights from {pretrained}")
                except Exception as e:
                    logger.error(f"Failed to load weights: {e}")
                    raise
            elif pretrained is True:
                # Do NOT reinstate a warn-and-return branch here. It made
                # `pretrained=True` hand back a randomly initialized model that
                # a caller had every reason to believe was trained; the house
                # rule (models/CLAUDE.md, Axis 3) is that an unavailable
                # checkpoint fails loudly.
                raise NotImplementedError(
                    f"No pretrained weights are distributed with dl_techniques "
                    f"for Mamba variant '{variant}'. Pass a local checkpoint "
                    f"instead: Mamba.from_variant('{variant}', "
                    f"vocab_size=..., pretrained='/path/to/weights.keras'), "
                    f"or use pretrained=False (default) for random init."
                )

        return model

    def get_config(self) -> Dict[str, Any]:
        """
        Return model configuration for serialization.

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
        """
        Create model instance from configuration.

        :param config: Dictionary containing model configuration.
        :type config: Dict[str, Any]
        :return: New Mamba model instance.
        :rtype: Mamba
        """
        return cls(**config)

    def summary(self, **kwargs: Any) -> None:
        """
        Print model summary with Mamba-specific information.

        :param kwargs: Additional arguments passed to keras.Model.summary.
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
    """Factory function to create a Mamba model with a task-specific head.

    This function demonstrates the intended integration pattern for Mamba:
    1. Instantiate a foundational `Mamba` model (optionally pretrained).
    2. Instantiate a task-specific head from the `dl_techniques.nlp.heads`
       factory.
    3. Combine them into a single, end-to-end `keras.Model`.

    Unlike BERT, Mamba does not inherently use an attention mask or token type
    IDs. This function only requires `input_ids` and creates a padding mask
    on-the-fly for compatibility with heads that might use it (e.g., for pooling).

    :param mamba_variant: The Mamba variant to use (e.g., "130m", "base").
    :type mamba_variant: str
    :param task_config: An `NLPTaskConfig` object defining the task, which must
        include `vocab_size`.
    :type task_config: NLPTaskConfig
    :param pretrained: If True, attempts to load pretrained weights (not yet
        implemented). If string, path to local weights file.
    :type pretrained: Union[bool, str]
    :param mamba_config_overrides: Optional dictionary to override default Mamba
        configuration for the chosen variant. Defaults to None.
    :type mamba_config_overrides: Optional[Dict[str, Any]]
    :param head_config_overrides: Optional dictionary to override default head
        configuration. Defaults to None.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :return: A complete `keras.Model` ready for the specified task.
    :rtype: keras.Model

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

    # The field is `vocabulary_size`. `NLPTaskConfig` is a dataclass and has
    # never had a `vocab_size` field, so the previous `hasattr(task_config,
    # 'vocab_size')` guard could not be satisfied by ANY config object and this
    # function raised on every call; the docstring and README examples passed
    # `vocab_size=...` to `NLPTaskConfig`, which is a `TypeError` before this
    # line is ever reached. Both examples are corrected above / in README § 9.
    if not getattr(task_config, 'vocabulary_size', None):
        raise ValueError(
            "The `task_config` must set 'vocabulary_size' "
            "to create a Mamba model."
        )

    # 1. Create the foundational Mamba model
    mamba_encoder = Mamba.from_variant(
        mamba_variant,
        vocab_size=task_config.vocabulary_size,
        pretrained=pretrained,
        **mamba_config_overrides,
    )

    # 2. Create the task head
    # DECISION plan-2026-08-17T183311-79c63e38/D-023: pool the LAST token, not
    # the first. `BaseNLPHead` defaults to `pooling_type='cls'`, which is right
    # for the bidirectional encoders that make up most of its consumers and
    # WRONG here: `Mamba` is a strictly causal selective SSM, so the hidden state
    # at position 0 is a function of token 0 alone and a 'cls'-pooled classifier
    # is a function of the first token id and nothing else. Measured on CPU with
    # an 8-token input before this line existed: perturbing token 5 moved the
    # logits by exactly 0.000e+00, while perturbing token 0 moved them by
    # 6.205e-02. The failure is SILENT -- the loss falls and accuracy plateaus at
    # the first-token prior. Same defect and same remedy as qwen3's D-029.
    # `head_config_overrides` still wins, so a caller can opt back out.
    # Do NOT "simplify" 'last' to `inputs[:, -1, :]`: SequencePooling's 'last'
    # resolves the last position KEPT BY THE MASK, which is why the
    # `attention_mask` built below is load-bearing rather than decorative.
    head_kwargs = {'pooling_type': 'last'}
    head_kwargs.update(head_config_overrides)
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=mamba_encoder.d_model,  # Pass Mamba's hidden size
        **head_kwargs,
    )

    # 3. Define inputs and build the end-to-end model
    # Mamba only requires input_ids
    inputs = {
        "input_ids": keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        ),
    }

    # Get hidden states from the encoder
    encoder_outputs = mamba_encoder(inputs)

    # Create a mask for compatibility with heads that might need it
    # (e.g., for masked pooling).
    attention_mask = keras.ops.not_equal(
        inputs["input_ids"], mamba_encoder.pad_token_id
    )

    # Pass encoder outputs to the task head
    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": attention_mask,
    }
    task_outputs = task_head(head_inputs)

    # Create the final model
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