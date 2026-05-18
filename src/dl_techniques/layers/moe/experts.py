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
from ..norms import create_normalization_layer

# ---------------------------------------------------------------------

class BaseExpert(keras.layers.Layer, ABC):
    """
    Abstract base class for MoE expert networks.

    Defines the interface that all expert implementations must follow,
    ensuring consistency across different expert types and enabling polymorphic
    usage in MoE layers. Subclasses must implement ``call`` and
    ``compute_output_shape``.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────┐
        │     BaseExpert (ABC)    │
        │                         │
        │  call(inputs) ──────►  output
        │  compute_output_shape() │
        └─────────────────────────┘

    :param name: Name for the expert layer.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for the base Layer class.
    :type kwargs: Any
    """

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        """Initialize the base expert layer."""
        super().__init__(name=name, **kwargs)
        self._built_input_shape = None

    @abstractmethod
    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward computation for the expert.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Expert output tensor.
        :rtype: keras.KerasTensor
        """
        pass

    @abstractmethod
    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the expert.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape of the output tensor.
        :rtype: Tuple[Optional[int], ...]
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
    Feed-Forward Network expert for MoE layers using the dl_techniques FFN factory.

    Wraps the dl_techniques FFN factory system to create various FFN architectures
    (MLP, SwiGLU, GeGLU, etc.) as expert networks within a Mixture-of-Experts layer.
    The FFN block is constructed lazily in ``build()`` via ``create_ffn_from_config()``,
    with all validation delegated to the factory, ensuring consistency and zero code
    duplication.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────┐
        │           FFNExpert               │
        │                                   │
        │  Input ──► FFN Block (factory) ──► Output
        │            (MLP / SwiGLU / GeGLU) │
        └───────────────────────────────────┘

    :param ffn_config: Dictionary containing FFN configuration passed to
        ``create_ffn_from_config()``. Must include ``'type'`` and appropriate
        parameters for the specified FFN type.
    :type ffn_config: Dict[str, Any]
    :param norm_type: Optional normalization type to apply pre- and/or
        post-FFN, instantiated through ``create_normalization_layer``. If
        ``None`` (default), no extra norms are added.
    :type norm_type: Optional[str]
    :param norm_config: Extra kwargs forwarded to ``create_normalization_layer``.
    :type norm_config: Optional[Dict[str, Any]]
    :param pre_norm: If True (default when ``norm_type`` is set), wrap the FFN
        with a pre-normalization layer.
    :type pre_norm: bool
    :param post_norm: If True, append a post-normalization layer after the FFN.
        Defaults to False. Only valid when the FFN preserves the last-dim size.
    :type post_norm: bool
    :param kwargs: Additional keyword arguments for the base Layer class.
    :type kwargs: Any
    """

    def __init__(
            self,
            ffn_config: Dict[str, Any],
            norm_type: Optional[str] = None,
            norm_config: Optional[Dict[str, Any]] = None,
            pre_norm: bool = True,
            post_norm: bool = False,
            **kwargs: Any,
    ) -> None:
        """Initialize the FFN expert using the FFN factory system."""
        super().__init__(**kwargs)

        self.ffn_config = ffn_config
        self.norm_type = norm_type
        self.norm_config = dict(norm_config) if norm_config else {}
        self.pre_norm_flag = pre_norm
        self.post_norm_flag = post_norm

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

        # CREATE the FFN block in __init__ (Golden Rule: create sub-layers here)
        try:
            self.ffn_block = create_ffn_from_config(self.ffn_config)
            logger.debug(f"Created {self.ffn_config['type']} expert using FFN factory")
        except Exception as e:
            raise ValueError(
                f"Failed to create FFN expert using factory. "
                f"FFN config: {self.ffn_config}. "
                f"Error: {e}"
            )

        # Optional pre/post normalization via factory.
        if self.norm_type is not None and self.pre_norm_flag:
            self.pre_norm = create_normalization_layer(
                self.norm_type, name='pre_norm', **self.norm_config
            )
        else:
            self.pre_norm = None

        if self.norm_type is not None and self.post_norm_flag:
            self.post_norm = create_normalization_layer(
                self.norm_type, name='post_norm', **self.norm_config
            )
        else:
            self.post_norm = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """BUILD the FFN expert sub-layers explicitly."""
        self._built_input_shape = input_shape

        if self.pre_norm is not None:
            self.pre_norm.build(input_shape)

        # BUILD the FFN block (sub-layer created in __init__)
        self.ffn_block.build(input_shape)

        if self.post_norm is not None:
            ffn_out_shape = self.ffn_block.compute_output_shape(input_shape)
            self.post_norm.build(ffn_out_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the FFN expert."""
        x = inputs
        if self.pre_norm is not None:
            x = self.pre_norm(x, training=training)
        x = self.ffn_block(x, training=training)
        if self.post_norm is not None:
            x = self.post_norm(x, training=training)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape by delegating to the FFN block."""
        assert self.ffn_block is not None, "compute_output_shape called before build/init"
        return self.ffn_block.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'ffn_config': self.ffn_config,
            'norm_type': self.norm_type,
            'norm_config': self.norm_config,
            'pre_norm': self.pre_norm_flag,
            'post_norm': self.post_norm_flag,
        })
        return config


def create_expert(expert_type: str, **kwargs) -> BaseExpert:
    """
    Factory function to create FFN expert networks.

    Only supports FFN experts, leveraging the dl_techniques FFN factory system
    for all FFN creation and validation.

    :param expert_type: Type of expert to create. Must be ``'ffn'``.
    :type expert_type: str
    :param kwargs: Configuration parameters for the expert. Should include
        ``'ffn_config'`` dictionary with FFN parameters.
    :type kwargs: Any
    :return: Configured FFN expert network.
    :rtype: BaseExpert
    :raises ValueError: If expert_type is not ``'ffn'`` or if configuration is invalid.
    """
    if expert_type == 'ffn':
        return FFNExpert(**kwargs)
    else:
        raise ValueError(
            f"Unsupported expert type: {expert_type}. "
            f"Only 'ffn' experts are supported in the simplified MoE implementation."
        )

