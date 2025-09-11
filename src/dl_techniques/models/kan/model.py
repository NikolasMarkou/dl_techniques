"""
Kolmogorov-Arnold Network (KAN) Model Implementation - Modern Keras 3
====================================================================

A complete implementation of the KAN architecture using modern Keras 3 patterns.
KAN uses learnable activation functions on edges rather than nodes, providing
a more flexible alternative to traditional MLPs with fixed activation functions.

Based on: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
https://arxiv.org/abs/2404.19756

Key Features:
------------
- Modern Keras 3 functional API patterns for robust serialization
- Modular design using KANLinear layers as building blocks
- Learnable spline-based activation functions on edges
- Support for multiple KAN variants for different tasks
- Complete serialization support with proper layer building
- Production-ready implementation with comprehensive error handling
- Full type safety with modern type hints

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
layer_configs = [
    {"features": 128, "grid_size": 8, "activation": "gelu"},
    {"features": 64, "grid_size": 6, "activation": "gelu"},
    {"features": 10, "grid_size": 5, "activation": "softmax"}
]
model = KAN(
    layer_configs=layer_configs,
    input_features=784,
    name="custom_kan"
)
```
"""

import keras
from keras import backend
from typing import Optional, Dict, Any, List, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.kan_linear import KANLinear

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KAN(keras.Model):
    """Modern Kolmogorov-Arnold Network model using Keras 3 functional API patterns.

    KAN stacks multiple KANLinear layers to create a deep network that can
    approximate complex multivariate functions using learnable spline-based
    activation functions on edges rather than nodes.

    **Intent**: Provide a clean, serializable KAN implementation that leverages
    the modern Keras 3 patterns for robust model creation, training, and deployment.
    The model uses the functional API internally for proper layer building and
    weight management.

    **Architecture**:
    ```
    Input(shape=[input_features])
           ↓
    KANLinear(layer_configs[0])
           ↓
    KANLinear(layer_configs[1])
           ↓
         ...
           ↓
    KANLinear(layer_configs[-1])
           ↓
    Output(shape=[final_features])
    ```

    **Configuration Pattern**:
    Each layer_config dict supports all KANLinear parameters:
    - features (required): output features
    - grid_size: B-spline grid size
    - spline_order: B-spline order
    - activation: activation function
    - regularization_factor: L2 regularization
    - grid_range: spline grid range
    - use_residual: residual connections
    - kernel_initializer/spline_initializer: weight initializers
    - kernel_regularizer/spline_regularizer: weight regularizers

    Args:
        layer_configs: List of dictionaries, each containing KANLinear configuration.
            Each dict must have 'features' key and optionally other KANLinear parameters.
        input_features: Integer, number of input features. Must be positive.
        enable_debugging: Boolean, whether to enable debug logging during training.
        name: Optional string name for the model.
        **kwargs: Additional arguments passed to the Model base class.

    Input shape:
        Tensor with shape `(batch_size, input_features)`.

    Output shape:
        Tensor with shape `(batch_size, layer_configs[-1]['features'])`.

    Example:
        ```python
        # Simple 3-layer KAN
        layer_configs = [
            {"features": 128, "grid_size": 8, "activation": "gelu"},
            {"features": 64, "grid_size": 6, "activation": "gelu"},
            {"features": 10, "grid_size": 5, "activation": "softmax"}
        ]
        model = KAN(
            layer_configs=layer_configs,
            input_features=784,
            enable_debugging=True
        )

        # Compile and use
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        ```

    Raises:
        ValueError: If layer_configs is empty or contains invalid configurations.
        ValueError: If input_features is not positive.
    """

    # Model variant configurations for easy setup
    VARIANT_CONFIGS = {
        "micro": {
            "hidden_features": [16, 8],
            "grid_size": 3,
            "spline_order": 3,
            "activation": "swish",
            "regularization_factor": 0.01
        },
        "small": {
            "hidden_features": [64, 32, 16],
            "grid_size": 5,
            "spline_order": 3,
            "activation": "swish",
            "regularization_factor": 0.01
        },
        "medium": {
            "hidden_features": [128, 64, 32],
            "grid_size": 7,
            "spline_order": 3,
            "activation": "gelu",
            "regularization_factor": 0.005
        },
        "large": {
            "hidden_features": [256, 128, 64, 32],
            "grid_size": 10,
            "spline_order": 3,
            "activation": "gelu",
            "regularization_factor": 0.001
        },
        "xlarge": {
            "hidden_features": [512, 256, 128, 64],
            "grid_size": 12,
            "spline_order": 3,
            "activation": "gelu",
            "regularization_factor": 0.0005
        }
    }

    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        input_features: int,
        enable_debugging: bool = False,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize KAN model using modern Keras 3 functional API patterns.

        Args:
            layer_configs: List of KANLinear configuration dictionaries.
            input_features: Number of input features.
            enable_debugging: Whether to enable debug logging.
            name: Optional model name.
            **kwargs: Additional Model arguments.

        Raises:
            ValueError: If configurations are invalid.
        """
        # Validate inputs first
        if not isinstance(layer_configs, list) or not layer_configs:
            raise ValueError("layer_configs must be a non-empty list")

        if not isinstance(input_features, int) or input_features <= 0:
            raise ValueError(f"input_features must be positive integer, got {input_features}")

        # Store configuration for serialization
        self.layer_configs = self._validate_and_copy_configs(layer_configs)
        self.input_features = input_features
        self.enable_debugging = enable_debugging
        self.num_layers = len(self.layer_configs)

        # Build model using functional API pattern
        inputs, outputs = self._build_functional_model()

        # Initialize parent Model class
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=name or "kan_model",
            **kwargs
        )

        logger.info(
            f"Created KAN model: {input_features} -> "
            f"{' -> '.join([str(cfg['features']) for cfg in self.layer_configs])} "
            f"({self.num_layers} layers)"
        )

    def _validate_and_copy_configs(
        self,
        configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate and create deep copy of layer configurations.

        Args:
            configs: Original configuration list.

        Returns:
            Validated and copied configuration list.

        Raises:
            ValueError: If any configuration is invalid.
        """
        validated_configs = []

        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                raise ValueError(f"Layer {i} config must be a dictionary, got {type(config)}")

            # Check required 'features' parameter
            if 'features' not in config:
                raise ValueError(f"Layer {i} config missing required 'features' key")

            features = config['features']
            if not isinstance(features, int) or features <= 0:
                raise ValueError(
                    f"Layer {i} 'features' must be positive integer, got {features}"
                )

            # Validate optional parameters
            if 'grid_size' in config:
                grid_size = config['grid_size']
                if not isinstance(grid_size, int) or grid_size < 3:
                    raise ValueError(
                        f"Layer {i} 'grid_size' must be integer >= 3, got {grid_size}"
                    )

            if 'spline_order' in config:
                spline_order = config['spline_order']
                if not isinstance(spline_order, int) or spline_order < 1:
                    raise ValueError(
                        f"Layer {i} 'spline_order' must be positive integer, got {spline_order}"
                    )

            # Create deep copy to avoid mutations
            validated_configs.append(dict(config))

        return validated_configs

    def _build_functional_model(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Build the model using Keras functional API for proper serialization.

        This method follows the modern Keras 3 pattern of using functional API
        in __init__ to ensure proper layer creation and building.

        Returns:
            Tuple of (inputs, outputs) tensors for Model initialization.
        """
        # Create input layer
        inputs = keras.Input(shape=(self.input_features,), name="kan_input")

        # Build sequential KAN layers
        x = inputs
        for i, config in enumerate(self.layer_configs):
            layer_name = f"kan_layer_{i}"

            # Add debugging wrapper if enabled
            if self.enable_debugging:
                x = self._add_debug_layer(x, i)

            # Create and apply KANLinear layer
            kan_layer = KANLinear(name=layer_name, **config)
            x = kan_layer(x)

            logger.info(f"Added {layer_name}: features={config['features']}")

        # Final debug layer if enabled
        if self.enable_debugging:
            x = self._add_debug_layer(x, self.num_layers, is_final=True)

        return inputs, x

    def _add_debug_layer(
        self,
        x: keras.KerasTensor,
        layer_idx: int,
        is_final: bool = False
    ) -> keras.KerasTensor:
        """Add debugging monitoring layer.

        Args:
            x: Input tensor to monitor.
            layer_idx: Index of the layer for naming.
            is_final: Whether this is the final output monitoring.

        Returns:
            Monitored tensor (unchanged data, just adds monitoring).
        """
        def debug_monitor(tensor: keras.KerasTensor) -> keras.KerasTensor:
            """Monitor tensor for debugging information using XLA-compatible ops."""
            # The original implementation with Python `if` and `logger` is not
            # compatible with graph compilation (e.g., tf.function) or XLA.
            # We use tf.debugging.check_numerics, which is the robust and
            # XLA-compatible way to check for NaN and Inf values.
            if backend.backend() != "tensorflow":
                return tensor  # No-op for other backends

            import tensorflow as tf

            # This op will raise an InvalidArgumentError if the tensor contains
            # NaN or Inf values, halting execution. This is a more effective
            # debugging strategy than logging warnings, as it prevents bad
            # values from propagating through the network.
            return tf.debugging.check_numerics(
                tensor,
                message=f"Numerical issue (NaN/Inf) detected in tensor before layer {layer_idx}"
            )

        suffix = "_final" if is_final else ""
        debug_layer = keras.layers.Lambda(
            debug_monitor,
            output_shape=lambda shape: shape,
            name=f"debug_monitor_{layer_idx}{suffix}"
        )

        return debug_layer(x)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        input_features: int,
        num_classes: int,
        enable_debugging: bool = False,
        override_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> "KAN":
        """Create KAN model from predefined variant configuration.

        Args:
            variant: String, one of "micro", "small", "medium", "large", "xlarge".
            input_features: Integer, number of input features.
            num_classes: Integer, number of output classes/features.
            enable_debugging: Boolean, whether to enable debugging.
            override_config: Optional dict to override variant defaults.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            Configured KAN model instance.

        Raises:
            ValueError: If variant is not recognized.

        Example:
            ```python
            # Standard MNIST classification
            model = KAN.from_variant("small", input_features=784, num_classes=10)

            # Custom activation for all layers
            model = KAN.from_variant(
                "medium",
                input_features=256,
                num_classes=50,
                override_config={"activation": "gelu"}
            )
            ```
        """
        if variant not in cls.VARIANT_CONFIGS:
            available = list(cls.VARIANT_CONFIGS.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available}")

        variant_config = cls.VARIANT_CONFIGS[variant].copy()

        # Apply overrides if provided
        if override_config:
            variant_config.update(override_config)

        # Extract hidden features and build layer configs
        hidden_features = variant_config.pop("hidden_features")
        all_features = hidden_features + [num_classes]

        # Create layer configurations
        layer_configs = []
        for features in all_features:
            config = variant_config.copy()
            config["features"] = features
            layer_configs.append(config)

        logger.info(f"Creating KAN-{variant.upper()} model")
        logger.info(f"Architecture: {input_features} -> {' -> '.join(map(str, all_features))}")

        return cls(
            layer_configs=layer_configs,
            input_features=input_features,
            enable_debugging=enable_debugging,
            **kwargs
        )

    @classmethod
    def from_layer_sizes(
        cls,
        layer_sizes: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        activation: str = "swish",
        regularization_factor: float = 0.01,
        enable_debugging: bool = False,
        **kan_layer_kwargs: Any
    ) -> "KAN":
        """Create KAN model from layer sizes with uniform configuration.

        This is a convenience method for creating KAN models where all layers
        share the same hyperparameters except for the number of features.

        Args:
            layer_sizes: List of layer sizes including input and output.
                E.g., [784, 128, 64, 10] creates input->128->64->10 architecture.
            grid_size: Grid size for all KANLinear layers.
            spline_order: Spline order for all layers.
            activation: Activation function for all layers.
            regularization_factor: L2 regularization factor.
            enable_debugging: Whether to enable debugging.
            **kan_layer_kwargs: Additional KANLinear parameters.

        Returns:
            Configured KAN model.

        Raises:
            ValueError: If layer_sizes has fewer than 2 elements.

        Example:
            ```python
            # Simple 3-layer network: 784->128->64->10
            model = KAN.from_layer_sizes(
                [784, 128, 64, 10],
                grid_size=8,
                activation="gelu"
            )
            ```
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements")

        input_features = layer_sizes[0]
        output_features = layer_sizes[1:]

        # Create uniform layer configurations
        layer_configs = []
        for features in output_features:
            config = {
                "features": features,
                "grid_size": grid_size,
                "spline_order": spline_order,
                "activation": activation,
                "regularization_factor": regularization_factor,
                **kan_layer_kwargs
            }
            layer_configs.append(config)

        return cls(
            layer_configs=layer_configs,
            input_features=input_features,
            enable_debugging=enable_debugging
        )

    def get_architecture_summary(self) -> str:
        """Get human-readable summary of model architecture.

        Returns:
            Multi-line string describing the model architecture.
        """
        lines = ["KAN Model Architecture Summary"]
        lines.append("=" * 50)

        # Model overview
        total_features = [self.input_features] + [cfg["features"] for cfg in self.layer_configs]
        architecture_str = " -> ".join(map(str, total_features))
        lines.append(f"Architecture: {architecture_str}")
        lines.append(f"Total layers: {self.num_layers}")
        lines.append(f"Debugging enabled: {self.enable_debugging}")
        lines.append("")

        # Layer details
        for i, config in enumerate(self.layer_configs):
            features = config["features"]
            grid_size = config.get("grid_size", "default")
            activation = config.get("activation", "default")
            spline_order = config.get("spline_order", "default")

            lines.append(
                f"Layer {i:2d}: features={features:4d}, grid_size={grid_size}, "
                f"activation={activation}, spline_order={spline_order}"
            )

        # Parameter estimation
        total_params = self._estimate_parameters()
        lines.append("")
        lines.append(f"Estimated parameters: ~{total_params:,}")

        return "\n".join(lines)

    def _estimate_parameters(self) -> int:
        """Estimate total number of parameters in the model.

        Returns:
            Approximate parameter count.
        """
        total_params = 0
        current_features = self.input_features

        for config in self.layer_configs:
            out_features = config["features"]
            grid_size = config.get("grid_size", 5)
            spline_order = config.get("spline_order", 3)

            # KANLinear parameter estimation
            # base_weight: in_features * out_features
            # spline_weight: in_features * out_features * (grid_size + spline_order - 1)
            # spline_scaler: in_features * out_features

            base_params = current_features * out_features
            spline_params = current_features * out_features * (grid_size + spline_order - 1)
            scaler_params = current_features * out_features

            layer_params = base_params + spline_params + scaler_params
            total_params += layer_params

            current_features = out_features

        return total_params

    def summary(self, **kwargs: Any) -> None:
        """Print model summary with additional KAN-specific information.

        Args:
            **kwargs: Arguments passed to parent summary method.
        """
        # Print standard Keras summary
        super().summary(**kwargs)

        # Print KAN-specific architecture summary
        print("\n" + self.get_architecture_summary())

    def get_config(self) -> Dict[str, Any]:
        """Get complete model configuration for serialization.

        Returns:
            Dictionary containing all configuration needed to reconstruct model.
        """
        config = super().get_config()
        config.update({
            "layer_configs": self.layer_configs,
            "input_features": self.input_features,
            "enable_debugging": self.enable_debugging,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KAN":
        """Reconstruct model from configuration.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            Reconstructed KAN model instance.
        """
        return cls(**config)


def create_compiled_kan(
    variant: str = "small",
    input_features: int = 784,
    num_classes: int = 10,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    loss: Union[str, keras.losses.Loss] = "sparse_categorical_crossentropy",
    metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
    **model_kwargs: Any
) -> KAN:
    """Create and compile KAN model ready for training.

    Args:
        variant: Model variant ("micro", "small", "medium", "large", "xlarge").
        input_features: Number of input features.
        num_classes: Number of output classes.
        optimizer: Optimizer name or instance.
        learning_rate: Learning rate for optimizer.
        loss: Loss function name or instance.
        metrics: List of metrics to track.
        **model_kwargs: Additional arguments for model creation.

    Returns:
        Compiled KAN model ready for training.

    Example:
        ```python
        # MNIST classification
        model = create_compiled_kan(
            variant="small",
            input_features=784,
            num_classes=10,
            learning_rate=0.001
        )

        # Binary classification
        model = create_compiled_kan(
            variant="medium",
            input_features=100,
            num_classes=1,
            loss="binary_crossentropy",
            metrics=["binary_accuracy"]
        )
        ```
    """
    if metrics is None:
        metrics = ["accuracy"]

    # Create model
    model = KAN.from_variant(
        variant=variant,
        input_features=input_features,
        num_classes=num_classes,
        **model_kwargs
    )

    # Configure optimizer with learning rate
    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, "learning_rate"):
            optimizer_instance.learning_rate = learning_rate
    else:
        optimizer_instance = optimizer

    # Compile model
    model.compile(
        optimizer=optimizer_instance,
        loss=loss,
        metrics=metrics
    )

    logger.info(
        f"Created and compiled KAN-{variant.upper()}: "
        f"{input_features}->{num_classes}, lr={learning_rate}"
    )

    return model