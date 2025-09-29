"""
Expert network implementations for Mixture of Experts (MoE) models.

This module provides FFN expert implementations that leverage the dl_techniques
FFN factory system, eliminating code duplication and ensuring consistency with
the broader framework's FFN implementations.
"""

import keras
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..ffn import create_ffn_from_config, validate_ffn_config

# ---------------------------------------------------------------------

class BaseExpert(keras.layers.Layer, ABC):
    """
    Abstract base class for MoE expert networks.

    This class defines the interface that all expert implementations must follow,
    ensuring consistency across different expert types and enabling polymorphic
    usage in MoE layers.

    Args:
        name: Name for the expert layer.
        **kwargs: Additional keyword arguments for the base Layer class.

    Methods:
        call: Abstract method for forward computation.
        compute_output_shape: Abstract method for output shape computation.
    """

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the base expert layer."""
        super().__init__(name=name, **kwargs)
        self._built_input_shape = None

    @abstractmethod
    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward computation for the expert.

        Args:
            inputs: Input tensor.
            training: Whether the layer is in training mode.

        Returns:
            Expert output tensor.
        """
        pass

    @abstractmethod
    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the expert.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Shape of the output tensor.
        """
        pass

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._built_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the expert from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


@keras.saving.register_keras_serializable()
class FFNExpert(BaseExpert):
    """
    Feed-Forward Network expert for MoE layers using dl_techniques FFN factory.

    This expert acts as a container that leverages the existing dl_techniques FFN
    factory system to create various FFN architectures (MLP, SwiGLU, GeGLU, etc.).
    All FFN validation, parameter handling, and layer creation is delegated to
    the FFN factory, eliminating code duplication.

    Args:
        ffn_config: Dictionary containing FFN configuration that will be passed
            to create_ffn_from_config(). Must include 'type' and appropriate
            parameters for the specified FFN type.
        **kwargs: Additional keyword arguments for the base Layer class.

    Example:
        ```python
        # Create SwiGLU expert
        expert = FFNExpert(
            ffn_config={
                "type": "swiglu",
                "d_model": 768,
                "ffn_expansion_factor": 4
            }
        )

        # Create MLP expert
        expert = FFNExpert(
            ffn_config={
                "type": "mlp",
                "hidden_dim": 2048,
                "output_dim": 768,
                "activation": "gelu",
                "dropout_rate": 0.1
            }
        )

        # Create GeGLU expert
        expert = FFNExpert(
            ffn_config={
                "type": "geglu",
                "hidden_dim": 3072,
                "output_dim": 768
            }
        )
        ```

    Note:
        All FFN creation, validation, and parameter handling is performed by the
        dl_techniques FFN factory system. This ensures consistency across the
        framework and automatic support for new FFN types as they're added.
    """

    def __init__(self, ffn_config: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize the FFN expert using the FFN factory system."""
        super().__init__(**kwargs)

        self.ffn_config = ffn_config

        # Validate FFN configuration using the factory system
        if 'type' not in ffn_config:
            raise ValueError("ffn_config must contain 'type' field specifying FFN type")

        # Extract type for validation
        ffn_type = ffn_config['type']
        ffn_params = {k: v for k, v in ffn_config.items() if k != 'type'}

        # Validate using FFN factory validation
        try:
            validate_ffn_config(ffn_type, **ffn_params)
        except ValueError as e:
            raise ValueError(f"Invalid FFN configuration for expert: {e}")

        self.ffn_block = None  # Created in build()

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the FFN expert using the factory system."""
        self._built_input_shape = input_shape

        try:
            # Create FFN block using the factory system
            # This handles all validation, parameter defaults, and layer creation
            self.ffn_block = create_ffn_from_config(self.ffn_config)
            logger.debug(f"Created {self.ffn_config['type']} expert using FFN factory")

        except Exception as e:
            raise ValueError(
                f"Failed to create FFN expert using factory. "
                f"FFN config: {self.ffn_config}. "
                f"Error: {e}"
            )

        # Build the FFN block
        self.ffn_block.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the FFN expert."""
        return self.ffn_block(inputs, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape by delegating to the FFN block."""
        if self.ffn_block is None:
            # If not built yet, estimate based on common FFN patterns
            if 'output_dim' in self.ffn_config:
                output_shape = list(input_shape)
                output_shape[-1] = self.ffn_config['output_dim']
                return tuple(output_shape)
            elif 'd_model' in self.ffn_config:
                # SwiGLU case
                output_shape = list(input_shape)
                output_shape[-1] = self.ffn_config['d_model']
                return tuple(output_shape)
            else:
                # Default: same as input
                return input_shape

        return self.ffn_block.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'ffn_config': self.ffn_config
        })
        return config


def create_expert(expert_type: str, **kwargs) -> BaseExpert:
    """
    Factory function to create FFN expert networks.

    This simplified factory only supports FFN experts, leveraging the existing
    dl_techniques FFN factory system for all FFN creation and validation.

    Args:
        expert_type: Type of expert to create. Must be 'ffn'.
        **kwargs: Configuration parameters for the expert. Should include
            'ffn_config' dictionary with FFN parameters.

    Returns:
        Configured FFN expert network.

    Raises:
        ValueError: If expert_type is not 'ffn' or if configuration is invalid.

    Example:
        ```python
        # Create SwiGLU expert
        expert = create_expert(
            'ffn',
            ffn_config={
                "type": "swiglu",
                "d_model": 768,
                "ffn_expansion_factor": 4
            }
        )

        # Create MLP expert with custom parameters
        expert = create_expert(
            'ffn',
            ffn_config={
                "type": "mlp",
                "hidden_dim": 2048,
                "output_dim": 768,
                "activation": "relu",
                "dropout_rate": 0.2
            }
        )
        ```
    """
    if expert_type == 'ffn':
        return FFNExpert(**kwargs)
    else:
        raise ValueError(
            f"Unsupported expert type: {expert_type}. "
            f"Only 'ffn' experts are supported in the simplified MoE implementation."
        )