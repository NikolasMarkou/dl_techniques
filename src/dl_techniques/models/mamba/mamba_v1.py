"""
Mamba State Space Model Implementation
============================================

Mamba provides an alternative to the popular Transformer architecture for
modeling long sequences. It is built upon the structured state space
model (SSM) paradigm, but introduces a critical "selection" mechanism that
allows it to operate with linear-time complexity in sequence length,
a significant improvement over the quadratic complexity of attention.

Architecturally, the model consists of a stack of Mamba blocks. Each block
applies a selective SSM to its input, allowing it to modulate how much
information is propagated or forgotten at each step based on the content
of the input sequence itself.

The foundational mathematical concept is the linear time-invariant (LTI)
state space model, a cornerstone of control theory. A continuous-time SSM
is defined by the differential equations:
    1. h'(t) = Ah(t) + Bx(t)  (State equation)
    2. y(t) = Ch(t) + Dx(t)  (Output equation)
Here, x(t) is the input, h(t) is a latent state vector, and y(t) is the
output. The matrices (A, B, C, D) are the model parameters that govern
the system's dynamics.

To be used in a deep learning model, this continuous system is discretized
into a recurrent formulation using a step size Δ:
    h_k = Āh_{k-1} + B̄x_k
    y_k = Ch_k + Dx_k

where Ā and B̄ are derived from A, B, and Δ. This recurrent form can
model sequences, but its time-invariant nature (fixed Ā, B̄) limits its
expressiveness for complex data like language.

Mamba's key innovation is to make the SSM parameters input-dependent,
thereby breaking the time-invariance and enabling selectivity. The crucial
parameters (step size Δ, and projections B and C) are no longer fixed but
are instead computed dynamically from the input tokens x_k at each time
step. This allows the model to selectively remember relevant information
and forget irrelevant details by changing the system dynamics on the fly.
For instance, a large Δ allows the state to be forgotten quickly, while a
small Δ preserves it. This content-aware reasoning is what allows Mamba
to effectively manage long-range dependencies.

While this selection mechanism makes the model powerful, it prevents the
use of a fast parallel convolutional computation method available to LTI
SSMs. To maintain efficiency, Mamba employs a hardware-aware parallel
scan algorithm. This algorithm allows the recurrent computation to be
executed efficiently on modern accelerators like GPUs, achieving linear-time
complexity while preserving the model's selective capabilities.

The Mamba architecture uses a selective state space mechanism that allows
the model to efficiently capture long-range dependencies in sequences while
maintaining linear-time complexity. Unlike attention-based models, Mamba
uses a data-dependent state space model with learned selective parameters.

Usage Examples:
--------------

.. code-block:: python

    import keras
    from dl_techniques.models.mamba import Mamba

    # 1. Create Mamba model with predefined variant
    mamba_model = Mamba.from_variant("base", vocab_size=50257)

    # 2. Load from local weights file
    mamba_model = Mamba.from_variant("large", vocab_size=50257,
                                      pretrained="path/to/weights.keras")

    # 3. Create Mamba with custom configuration
    mamba_model = Mamba(
        vocab_size=50257,
        d_model=768,
        num_layers=24,
        d_state=16,
        d_conv=4,
        expand=2
    )

    # 4. Use the model
    inputs = {
        "input_ids": keras.random.randint((2, 512), 0, 50257, dtype="int32")
    }
    outputs = mamba_model(inputs)
    print(outputs["last_hidden_state"].shape)  # (2, 512, 768)

Key Features:
- Linear-time complexity O(L) vs O(L²) for attention
- Selective state space mechanism with data-dependent parameters
- Efficient long-range dependency modeling
- Hardware-optimized selective scan implementation
- Modular architecture separating encoding from task-specific heads

Architecture:
    Input (tokens) → Embedding → [MambaLayer × N] → Output (hidden states)

    Each MambaLayer contains:
    - Input projection (splits into x and z paths)
    - Causal 1D convolution
    - Selective SSM with data-dependent Δ, B, C parameters
    - Output gating and projection

References:
    - Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with
      Selective State Spaces. arXiv preprint arXiv:2312.00752.
      https://arxiv.org/abs/2312.00752
"""

import keras
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.nlp_heads import NLPTaskConfig, create_nlp_head
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

    # Model variants following the original Mamba paper specifications
    MODEL_VARIANTS = {
        "2.8b": {
            "d_model": 2560,
            "num_layers": 64,
            "description": "Mamba-2.8B: Largest variant with 2.8B parameters"
        },
        "1.4b": {
            "d_model": 1536,
            "num_layers": 48,
            "description": "Mamba-1.4B: Large variant with ~1.4B parameters"
        },
        "790m": {
            "d_model": 1024,
            "num_layers": 48,
            "description": "Mamba-790M: Medium variant with ~790M parameters"
        },
        "370m": {
            "d_model": 1024,
            "num_layers": 24,
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
        :param pretrained: If True, attempts to load pretrained weights (not yet
            implemented). If string, loads weights from the specified path.
            Defaults to False.
        :type pretrained: Union[bool, str]
        :param kwargs: Additional arguments to override variant defaults.
        :return: A Mamba model instance configured for the specified variant.
        :rtype: Mamba
        :raises ValueError: If unknown variant or invalid parameters.

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
                # Future: load from hub or default location
                logger.warning(
                    "Pretrained weights not yet available. "
                    "Model initialized with random weights."
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

            from dl_techniques.layers.nlp_heads import NLPTaskType

            # Define a task for sequence classification
            seq_cls_task = NLPTaskConfig(
                name="sentiment_analysis",
                task_type=NLPTaskType.SEQUENCE_CLASSIFICATION,
                num_classes=3,
                vocab_size=50257  # Mamba needs vocab_size at creation
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

    if not hasattr(task_config, 'vocab_size') or not task_config.vocab_size:
        raise ValueError(
            "The `task_config` must have a 'vocab_size' attribute "
            "to create a Mamba model."
        )

    # 1. Create the foundational Mamba model
    mamba_encoder = Mamba.from_variant(
        mamba_variant,
        vocab_size=task_config.vocab_size,
        pretrained=pretrained,
        **mamba_config_overrides,
    )

    # 2. Create the task head
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=mamba_encoder.d_model,  # Pass Mamba's hidden size
        **head_config_overrides,
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