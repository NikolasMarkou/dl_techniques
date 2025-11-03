"""
Mamba (v1) State Space Model Implementation
============================================

A complete implementation of the Mamba (v1) architecture as described in
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
(Gu & Dao, 2023) https://arxiv.org/abs/2312.00752

This implementation provides a pure foundation model separating the core
selective state space logic from task-specific heads, following the same
architectural principles as BERT in dl_techniques.

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
"""

import keras
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
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
