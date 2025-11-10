import keras
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .components_v2 import Mamba2ResidualBlock


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Mamba2(keras.Model):
    """
    Mamba v2 foundation model for efficient sequence modeling.

    This model stacks Mamba2ResidualBlocks to form a deep sequence model,
    implementing the architecture from "Mamba: Linear-Time Sequence
    Modeling with Selective State Spaces" with V2 block enhancements.

    :param vocab_size: Size of the vocabulary.
    :param d_model: Dimensionality of the model's hidden states.
    :param num_layers: Number of Mamba residual blocks.
    :param d_state: Dimensionality of SSM latent state.
    :param d_conv: Kernel size for causal convolutions.
    :param expand: Expansion factor for internal dimensions.
    :param headdim: Dimensionality of each SSM head.
    :param norm_epsilon: Epsilon for all normalization layers.
    :param pad_token_id: ID of the padding token.
    :param rmsnorm: If True, use RMSNorm instead of LayerNormalization.
    :param d_ssm: Dimensionality of the SSM. Defaults to `d_model * expand`.
    """

    MODEL_VARIANTS = {
        "2.8b": {"d_model": 2560, "num_layers": 64},
        "1.4b": {"d_model": 2048, "num_layers": 48},
        "780m": {"d_model": 1536, "num_layers": 36},
        "370m": {"d_model": 1024, "num_layers": 24},
        "130m": {"d_model": 768, "num_layers": 24, "name": "base"},
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

        # If d_ssm is not provided, it should default to d_inner.
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
        if variant == "base":
            variant = "130m"
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}")

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)
        config["vocab_size"] = vocab_size
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "num_layers": self.num_layers, "d_state": self.d_state,
            "d_conv": self.d_conv, "expand": self.expand,
            "headdim": self.headdim, "norm_epsilon": self.norm_epsilon,
            "pad_token_id": self.pad_token_id,
            "rmsnorm": self.rmsnorm,
            "d_ssm": self.d_ssm,
        })
        return config

# ---------------------------------------------------------------------
