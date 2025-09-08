"""
Kolmogorov-Arnold Network (KAN) Model Implementation
===========================================================

A complete implementation of the KAN architecture using modern Keras 3 patterns.
KAN uses learnable activation functions on edges rather than nodes, providing
a more flexible alternative to traditional MLPs with fixed activation functions.

Based on: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
https://arxiv.org/abs/2404.19756

Key Features:
------------
- Modular design using KANLinear layers as building blocks
- Learnable spline-based activation functions on edges
- Support for multiple KAN variants for different tasks
- Comprehensive debugging and monitoring capabilities
- Complete serialization support with modern Keras 3 patterns
- Production-ready implementation with robust error handling

Architecture Concept:
-------------------
KAN replaces the linear transformations of traditional MLPs with learnable
univariate functions. Each edge has a learnable activation function parameterized
by B-splines, making the network more expressive and interpretable.

Model Variants:
--------------
- KAN-Micro: [16, 8] layers, grid_size=3 (for simple tasks)
- KAN-Small: [64, 32, 16] layers, grid_size=5 (MNIST, small datasets)
- KAN-Medium: [128, 64, 32] layers, grid_size=7 (CIFAR-10, medium datasets)
- KAN-Large: [256, 128, 64, 32] layers, grid_size=10 (complex tasks)
- KAN-XLarge: [512, 256, 128, 64] layers, grid_size=12 (large datasets)

Usage Examples:
-------------
```python
# MNIST classification (784 -> 10)
model = KAN.from_variant("small", input_features=784, num_classes=10)

# Regression task with custom architecture
model = KAN.from_variant("medium", input_features=100, num_classes=1)

# Custom configuration
config = [
    {"in_features": 784, "out_features": 128, "grid_size": 8},
    {"in_features": 128, "out_features": 64, "grid_size": 6},
    {"in_features": 64, "out_features": 10, "grid_size": 5}
]
model = KAN(layers_configurations=config)
```
"""

import keras
from typing import Optional, Dict, Any, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.kan_linear import KANLinear


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class KAN(keras.Model):
    """Kolmogorov-Arnold Network model using modern Keras 3 patterns.

    KAN stacks multiple KANLinear layers to create a deep network that can
    approximate complex multivariate functions using learnable spline-based
    activation functions on edges rather than nodes.

    Args:
        layers_configurations: List of dictionaries, each containing configuration
            for a KANLinear layer. Each dict should have 'in_features' and 'out_features'
            keys, and optionally other KANLinear parameters.
        enable_debugging: Boolean, whether to enable extra validation during forward pass.
            When True, logs layer outputs and checks for numerical issues.
        input_features: Optional integer, input feature dimension. If provided and
            layers_configurations is None, will be used with num_classes to create default config.
        num_classes: Optional integer, output dimension. Only used with input_features
            when layers_configurations is None.
        include_top: Boolean, whether to include the final output layer.
        name: Optional string name for the model.
        **kwargs: Additional arguments passed to the Model base class.

    Raises:
        ValueError: If layer configurations are invalid or incompatible.

    Example:
        >>> # Custom configuration
        >>> config = [
        ...     {"in_features": 784, "out_features": 128, "grid_size": 8},
        ...     {"in_features": 128, "out_features": 64, "grid_size": 6},
        ...     {"in_features": 64, "out_features": 10, "grid_size": 5}
        ... ]
        >>> model = KAN(layers_configurations=config)
        >>>
        >>> # Simple configuration
        >>> model = KAN(input_features=784, num_classes=10)
        >>>
        >>> # From variant
        >>> model = KAN.from_variant("small", input_features=784, num_classes=10)
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "micro": {
            "hidden_sizes": [16, 8],
            "grid_size": 3,
            "spline_order": 3,
            "activation": "swish"
        },
        "small": {
            "hidden_sizes": [64, 32, 16],
            "grid_size": 5,
            "spline_order": 3,
            "activation": "swish"
        },
        "medium": {
            "hidden_sizes": [128, 64, 32],
            "grid_size": 7,
            "spline_order": 3,
            "activation": "gelu"
        },
        "large": {
            "hidden_sizes": [256, 128, 64, 32],
            "grid_size": 10,
            "spline_order": 3,
            "activation": "gelu"
        },
        "xlarge": {
            "hidden_sizes": [512, 256, 128, 64],
            "grid_size": 12,
            "spline_order": 3,
            "activation": "gelu"
        }
    }

    # Architecture constants
    DEFAULT_GRID_SIZE = 5
    DEFAULT_SPLINE_ORDER = 3
    DEFAULT_ACTIVATION = "swish"

    def __init__(
            self,
            layers_configurations: Optional[List[Dict[str, Any]]] = None,
            enable_debugging: bool = False,
            input_features: Optional[int] = None,
            num_classes: Optional[int] = None,
            include_top: bool = True,
            name: Optional[str] = None,
            **kwargs: Any
    ):
        # Handle configuration creation
        if layers_configurations is None:
            if input_features is None or num_classes is None:
                raise ValueError(
                    "Either layers_configurations must be provided, or both "
                    "input_features and num_classes must be specified"
                )
            layers_configurations = [
                {
                    "in_features": input_features,
                    "out_features": num_classes,
                    "grid_size": self.DEFAULT_GRID_SIZE,
                    "spline_order": self.DEFAULT_SPLINE_ORDER,
                    "activation": self.DEFAULT_ACTIVATION
                }
            ]

        # Validate and store configuration
        self.layers_configurations = self._deep_copy_configurations(layers_configurations)
        self.enable_debugging = enable_debugging
        self.include_top = include_top
        self._validate_configurations(self.layers_configurations)

        # Store computed properties
        self.input_features = self.layers_configurations[0]["in_features"]
        self.output_features = self.layers_configurations[-1]["out_features"] if include_top else None

        # Track model statistics for debugging
        self._layer_stats = {}

        # Build the model using functional API
        inputs = keras.Input(shape=(self.input_features,), name="input")
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, name=name or "kan_model", **kwargs)

        logger.info(
            f"Created KAN model with {len(self.layers_configurations)} layers: "
            f"{self.input_features} -> {self.output_features if include_top else 'features'}"
        )

    def _deep_copy_configurations(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a deep copy of configurations to avoid mutations.

        Args:
            configs: Original configuration list.

        Returns:
            Deep copy of the configuration list.

        Raises:
            ValueError: If configs is not a list.
        """
        if not isinstance(configs, list):
            raise ValueError("Layer configurations must be a list")
        return [dict(config) for config in configs]

    def _validate_configurations(self, configs: List[Dict[str, Any]]) -> None:
        """Validate layer configurations for compatibility.

        Args:
            configs: List of layer configuration dictionaries.

        Raises:
            ValueError: If configurations are invalid or incompatible.
        """
        if not configs:
            raise ValueError("Layer configurations cannot be empty")

        if not isinstance(configs, list):
            raise ValueError("Layer configurations must be a list")

        # Check that each config has required keys and valid values
        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                raise ValueError(f"Layer {i} configuration must be a dictionary")

            required_keys = ['in_features', 'out_features']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Layer {i} missing required key: '{key}'")

                if not isinstance(config[key], int) or config[key] <= 0:
                    raise ValueError(f"Layer {i} '{key}' must be a positive integer, got {config[key]}")

            # Validate optional parameters if present
            if 'grid_size' in config:
                if not isinstance(config['grid_size'], int) or config['grid_size'] < 3:
                    raise ValueError(f"Layer {i} 'grid_size' must be an integer >= 3, got {config['grid_size']}")

            if 'spline_order' in config:
                if not isinstance(config['spline_order'], int) or config['spline_order'] < 1:
                    raise ValueError(
                        f"Layer {i} 'spline_order' must be a positive integer, got {config['spline_order']}")

        # Check that consecutive layers are compatible
        for i in range(len(configs) - 1):
            current_out = configs[i]['out_features']
            next_in = configs[i + 1]['in_features']

            if current_out != next_in:
                raise ValueError(
                    f"Incompatible layer dimensions: Layer {i} output features ({current_out}) "
                    f"don't match Layer {i + 1} input features ({next_in})"
                )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete KAN model architecture.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        x = inputs

        # Apply KAN layers sequentially
        for i, layer_config in enumerate(self.layers_configurations):
            x = self._build_kan_layer(x, i, layer_config)

        return x

    def _build_kan_layer(
            self,
            x: keras.KerasTensor,
            layer_idx: int,
            config: Dict[str, Any]
    ) -> keras.KerasTensor:
        """Build a single KAN layer.

        Args:
            x: Input tensor
            layer_idx: Index of the layer
            config: Configuration dictionary for the layer

        Returns:
            Output tensor from the KAN layer
        """
        # Create KAN layer with debugging wrapper if needed
        kan_layer = KANLinear(name=f"kan_layer_{layer_idx}", **config)

        x = kan_layer(x)

        # Add debugging monitoring if enabled
        if self.enable_debugging:
            # This will be executed during the forward pass
            def debug_monitor(tensor):
                # Log layer information
                logger.info(f"KAN Layer {layer_idx}: shape={keras.ops.shape(tensor)}")

                # Check for numerical issues
                if keras.ops.any(keras.ops.isnan(tensor)):
                    logger.warning(f"NaN detected in KAN layer {layer_idx} output")
                if keras.ops.any(keras.ops.isinf(tensor)):
                    logger.warning(f"Infinite values detected in KAN layer {layer_idx} output")

                return tensor

            x = keras.layers.Lambda(debug_monitor, name=f"debug_monitor_{layer_idx}")(x)

        logger.info(f"Built KAN layer {layer_idx}: {config['in_features']} -> {config['out_features']}")

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_features: int,
            num_classes: int,
            enable_debugging: bool = False,
            **kwargs
    ) -> "KAN":
        """Create a KAN model from a predefined variant.

        Args:
            variant: String, one of "micro", "small", "medium", "large", "xlarge"
            input_features: Integer, number of input features
            num_classes: Integer, number of output classes
            enable_debugging: Boolean, whether to enable debugging
            **kwargs: Additional arguments passed to the constructor

        Returns:
            KAN model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # MNIST classification
            >>> model = KAN.from_variant("small", input_features=784, num_classes=10)
            >>> # Binary classification
            >>> model = KAN.from_variant("medium", input_features=100, num_classes=1)
            >>> # Multi-class problem
            >>> model = KAN.from_variant("large", input_features=256, num_classes=50)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]
        hidden_sizes = config["hidden_sizes"]

        # Create layer sizes by combining input, hidden, and output
        layer_sizes = [input_features] + hidden_sizes + [num_classes]

        # Create layer configurations
        layers_configurations = []
        for i in range(len(layer_sizes) - 1):
            layer_config = {
                "in_features": layer_sizes[i],
                "out_features": layer_sizes[i + 1],
                "grid_size": config["grid_size"],
                "spline_order": config["spline_order"],
                "activation": config["activation"]
            }
            layers_configurations.append(layer_config)

        logger.info(f"Creating KAN-{variant.upper()} model")
        logger.info(f"Architecture: {' -> '.join(map(str, layer_sizes))}")

        return cls(
            layers_configurations=layers_configurations,
            enable_debugging=enable_debugging,
            **kwargs
        )

    def get_architecture_summary(self) -> str:
        """Get a human-readable summary of the model architecture.

        Returns:
            String description of the model architecture.
        """
        if not self.layers_configurations:
            return "Empty KAN model"

        summary_lines = ["KAN Model Architecture:"]
        summary_lines.append("=" * 50)

        total_params = 0
        for i, config in enumerate(self.layers_configurations):
            in_feat = config['in_features']
            out_feat = config['out_features']
            grid_size = config.get('grid_size', 5)
            activation = config.get('activation', 'swish')

            # Estimate parameters (base + spline weights + scaler)
            base_params = in_feat * out_feat
            spline_params = in_feat * out_feat * (grid_size + 2)  # Approximate
            scaler_params = in_feat * out_feat
            layer_params = base_params + spline_params + scaler_params
            total_params += layer_params

            summary_lines.append(
                f"Layer {i:2d}: {in_feat:4d} -> {out_feat:4d} "
                f"(grid={grid_size}, act={activation}, params≈{layer_params:,})"
            )

        summary_lines.append("=" * 50)
        summary_lines.append(f"Total parameters (approx): {total_params:,}")
        summary_lines.append(f"Number of layers: {len(self.layers_configurations)}")
        summary_lines.append(f"Debugging enabled: {self.enable_debugging}")

        return "\n".join(summary_lines)

    def get_layer_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics from the last forward pass (debug mode only).

        Returns:
            Dictionary containing layer-wise statistics.
        """
        return dict(self._layer_stats) if self.enable_debugging else {}

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = {
            "layers_configurations": self.layers_configurations,
            "enable_debugging": self.enable_debugging,
            "include_top": self.include_top,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KAN":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            KAN model instance
        """
        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        total_layers = len(self.layers_configurations)
        logger.info(f"KAN configuration:")
        logger.info(f"  - Input features: {self.input_features}")
        logger.info(f"  - Output features: {self.output_features}")
        logger.info(f"  - Total KAN layers: {total_layers}")
        logger.info(f"  - Include top: {self.include_top}")
        logger.info(f"  - Debugging enabled: {self.enable_debugging}")

        # Print architecture summary
        print("\n" + self.get_architecture_summary())


# ---------------------------------------------------------------------

def create_kan_model(
        variant: str = "small",
        input_features: int = 784,
        num_classes: int = 10,
        optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
        learning_rate: float = 0.001,
        loss: Union[str, keras.losses.Loss] = "sparse_categorical_crossentropy",
        metrics: List[Union[str, keras.metrics.Metric]] = None,
        **kwargs
) -> KAN:
    """Convenience function to create and compile KAN models.

    Args:
        variant: String, model variant ("micro", "small", "medium", "large", "xlarge")
        input_features: Integer, number of input features
        num_classes: Integer, number of output classes
        optimizer: String name or optimizer instance. Default is "adam"
        learning_rate: Float, learning rate for optimizer. Default is 0.001
        loss: String name or loss function. Default is "sparse_categorical_crossentropy"
        metrics: List of metrics to track. Default is ["accuracy"]
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        Compiled KAN model ready for training

    Example:
        >>> # MNIST classification
        >>> model = create_kan_model("small", input_features=784, num_classes=10)
        >>>
        >>> # Binary classification with custom settings
        >>> model = create_kan_model(
        ...     "medium",
        ...     input_features=100,
        ...     num_classes=1,
        ...     loss="binary_crossentropy",
        ...     metrics=["binary_accuracy"]
        ... )
        >>>
        >>> # Regression task
        >>> model = create_kan_model(
        ...     "large",
        ...     input_features=50,
        ...     num_classes=1,
        ...     loss="mse",
        ...     metrics=["mae"]
        ... )
    """
    if metrics is None:
        metrics = ["accuracy"]

    # Create the model
    model = KAN.from_variant(
        variant=variant,
        input_features=input_features,
        num_classes=num_classes,
        **kwargs
    )

    # Set up optimizer
    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, 'learning_rate'):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    # Compile the model
    model.compile(
        optimizer=optimizer_instance,
        loss=loss,
        metrics=metrics
    )

    logger.info(f"Created and compiled KAN-{variant.upper()} with input_features={input_features}, "
                f"num_classes={num_classes}")

    return model


def create_kan_from_sizes(
        layer_sizes: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        activation: str = 'swish',
        enable_debugging: bool = False,
        **kan_layer_kwargs: Any
) -> KAN:
    """Create a KAN model with uniform configuration across layers.

    This is a convenience function for creating KAN models where all layers
    share the same hyperparameters except for input/output dimensions.

    Args:
        layer_sizes: List of layer sizes including input and output dimensions.
            E.g., [784, 128, 64, 10] creates a 3-layer network.
        grid_size: Grid size for all KANLinear layers.
        spline_order: Spline order for all KANLinear layers.
        activation: Activation function for all layers.
        enable_debugging: Whether to enable debugging in the model.
        **kan_layer_kwargs: Additional arguments passed to all KANLinear layers.

    Returns:
        Configured KAN model.

    Raises:
        ValueError: If layer_sizes has fewer than 2 elements.

    Example:
        >>> # 3-layer KAN: 784->128->64->10
        >>> model = create_kan_from_sizes([784, 128, 64, 10], grid_size=8, activation='gelu')
        >>>
        >>> # Simple 1-layer network for binary classification
        >>> model = create_kan_from_sizes([100, 1], grid_size=5, activation='swish')
    """
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must have at least 2 elements (input and output)")

    # Create layer configurations
    layer_configs = []
    for i in range(len(layer_sizes) - 1):
        config = {
            'in_features': layer_sizes[i],
            'out_features': layer_sizes[i + 1],
            'grid_size': grid_size,
            'spline_order': spline_order,
            'activation': activation,
            **kan_layer_kwargs
        }
        layer_configs.append(config)

    return KAN(
        layers_configurations=layer_configs,
        enable_debugging=enable_debugging
    )

# ---------------------------------------------------------------------