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
- Modular design using the corrected KANLinear layer as building blocks
- Learnable spline-based activation functions on edges
- Support for multiple KAN variants for different tasks
- Complete serialization support with proper layer building
- Production-ready implementation with comprehensive error handling
- Full type safety with modern type hints

Architecture Concept:
-------------------
KAN replaces the linear transformations of traditional MLPs with learnable
univariate functions. Each edge has a learnable activation function parameterized
by B-splines. The final activation (e.g., softmax) is applied separately after
the last KAN layer.

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
    {"features": 10, "grid_size": 5, "activation": "softmax"} # Final activation
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
    activation functions on edges. The final activation is applied separately.

    **Intent**: Provide a clean, serializable KAN implementation that leverages
    modern Keras 3 patterns for robust model creation, training, and deployment.
    The model uses the functional API internally for proper layer building and
    weight management.

    **Architecture**:
    ```
    Input(shape=[input_features])
           ↓
    KANLinear(layer_configs)
           ↓
         ...
           ↓
    KANLinear(layer_configs[-1]) -> Produces logits
           ↓
    Activation(final_activation)
           ↓
    Output(shape=[final_features])
    ```

    **Configuration Pattern**:
    Each layer_config dict supports all KANLinear parameters:
    - features (required): output features
    - grid_size, spline_order, grid_range: B-spline parameters
    - activation: The fixed activation for the base component in KANLinear.
    - activation: (Only for the last layer) The final activation function
      (e.g., 'softmax', 'sigmoid', 'linear').

    Args:
        layer_configs: List of dictionaries, each containing KANLinear configuration.
        input_features: Integer, number of input features. Must be positive.
        enable_debugging: Boolean, whether to enable debug logging during training.
        name: Optional string name for the model.
        **kwargs: Additional arguments passed to the Model base class.
    """

    # Model variant configurations for easy setup
    VARIANT_CONFIGS = {
        "micro": {
            "hidden_features": [16, 8],
            "grid_size": 3,
            "spline_order": 3,
            "activation": "swish",
        },
        "small": {
            "hidden_features": [64, 32, 16],
            "grid_size": 5,
            "spline_order": 3,
            "activation": "swish",
        },
        "medium": {
            "hidden_features": [128, 64, 32],
            "grid_size": 7,
            "spline_order": 3,
            "activation": "gelu",
        },
        "large": {
            "hidden_features": [256, 128, 64, 32],
            "grid_size": 10,
            "spline_order": 3,
            "activation": "gelu",
        },
        "xlarge": {
            "hidden_features": [512, 256, 128, 64],
            "grid_size": 12,
            "spline_order": 3,
            "activation": "gelu",
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
        if not isinstance(layer_configs, list) or not layer_configs:
            raise ValueError("layer_configs must be a non-empty list")
        if not isinstance(input_features, int) or input_features <= 0:
            raise ValueError(f"input_features must be positive integer, got {input_features}")

        self.layer_configs = self._validate_and_copy_configs(layer_configs)
        self.input_features = input_features
        self.enable_debugging = enable_debugging
        self.num_layers = len(self.layer_configs)

        inputs, outputs = self._build_functional_model()

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
        validated_configs = []
        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                raise ValueError(f"Layer {i} config must be a dict, got {type(config)}")
            if 'features' not in config:
                raise ValueError(f"Layer {i} config missing required 'features' key")
            if not isinstance(config['features'], int) or config['features'] <= 0:
                raise ValueError(f"Layer {i} 'features' must be positive int, got {config['features']}")
            validated_configs.append(dict(config))
        return validated_configs

    def _build_functional_model(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        inputs = keras.Input(shape=(self.input_features,), name="kan_input")
        x = inputs
        final_activation = None

        for i, config in enumerate(self.layer_configs):
            layer_name = f"kan_layer_{i}"
            kan_config = config.copy()

            if self.enable_debugging:
                x = self._add_debug_layer(x, i)

            # Handle activation logic
            if 'activation' in kan_config and i < self.num_layers - 1:
                # For intermediate layers, 'activation' means 'activation'
                kan_config['activation'] = kan_config.pop('activation')
            elif i == self.num_layers - 1:
                # For the last layer, 'activation' is the final model activation
                final_activation = kan_config.pop('activation', 'linear')
                # The last KAN layer should produce raw logits
                kan_config['activation'] = 'linear'

            kan_layer = KANLinear(name=layer_name, **kan_config)
            x = kan_layer(x)
            logger.info(f"Added {layer_name}: features={config['features']}")

        # Apply the final activation function after the last KAN layer
        if final_activation:
            x = keras.layers.Activation(final_activation, name="final_activation")(x)
            logger.info(f"Added final activation: {final_activation}")
            
        if self.enable_debugging:
            x = self._add_debug_layer(x, self.num_layers, is_final=True)

        return inputs, x

    def _add_debug_layer(
        self,
        x: keras.KerasTensor,
        layer_idx: int,
        is_final: bool = False
    ) -> keras.KerasTensor:
        def debug_monitor(tensor: keras.KerasTensor) -> keras.KerasTensor:
            if backend.backend() != "tensorflow":
                return tensor
            import tensorflow as tf
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
        if variant not in cls.VARIANT_CONFIGS:
            available = list(cls.VARIANT_CONFIGS.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available}")

        variant_config = cls.VARIANT_CONFIGS[variant].copy()
        if override_config:
            variant_config.update(override_config)

        hidden_features = variant_config.pop("hidden_features")
        layer_configs = []

        # Hidden layers
        for features in hidden_features:
            config = variant_config.copy()
            config["features"] = features
            layer_configs.append(config)

        # Output layer
        output_config = variant_config.copy()
        output_config["features"] = num_classes
        # Determine final activation based on num_classes
        if num_classes > 1:
            output_config["activation"] = "softmax"
        elif num_classes == 1:
            output_config["activation"] = "sigmoid" # Common for binary classification/regression
        layer_configs.append(output_config)

        logger.info(f"Creating KAN-{variant.upper()} model")
        arch_str = ' -> '.join(map(str, [input_features] + hidden_features + [num_classes]))
        logger.info(f"Architecture: {arch_str}")

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
        final_activation: Optional[str] = None,
        enable_debugging: bool = False,
        **kan_layer_kwargs: Any
    ) -> "KAN":
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements")

        input_features = layer_sizes[0]
        output_feature_sizes = layer_sizes[1:]

        layer_configs = []
        for i, features in enumerate(output_feature_sizes):
            config = {
                "features": features,
                "grid_size": grid_size,
                "spline_order": spline_order,
                "activation": activation,
                **kan_layer_kwargs
            }
            if i == len(output_feature_sizes) - 1: # Last layer
                if final_activation:
                    config["activation"] = final_activation
                elif features > 1:
                    config["activation"] = "softmax"
                else:
                    config["activation"] = "linear"

            layer_configs.append(config)

        return cls(
            layer_configs=layer_configs,
            input_features=input_features,
            enable_debugging=enable_debugging
        )

    def get_architecture_summary(self) -> str:
        lines = ["KAN Model Architecture Summary"]
        lines.append("=" * 50)
        total_features = [self.input_features] + [cfg["features"] for cfg in self.layer_configs]
        lines.append(f"Architecture: {' -> '.join(map(str, total_features))}")
        lines.append(f"Total layers: {self.num_layers}")
        lines.append(f"Debugging enabled: {self.enable_debugging}")
        lines.append("")

        for i, config in enumerate(self.layer_configs):
            is_last = (i == self.num_layers - 1)
            act_key = 'activation' if is_last and 'activation' in config else 'activation'
            activation = config.get(act_key, "linear")
            lines.append(
                f"Layer {i:2d}: features={config['features']:4d}, "
                f"grid_size={config.get('grid_size', 'def')}, "
                f"spline_order={config.get('spline_order', 'def')}, "
                f"{act_key}='{activation}'"
            )

        lines.append("")
        lines.append(f"Estimated parameters: ~{self._estimate_parameters():,}")
        return "\n".join(lines)

    def _estimate_parameters(self) -> int:
        total_params = 0
        current_features = self.input_features
        for config in self.layer_configs:
            out_features = config["features"]
            grid_size = config.get("grid_size", 5)
            spline_order = config.get("spline_order", 3)
            
            num_basis_fns = grid_size + spline_order
            
            # Parameters for one KANLinear layer:
            # spline_weight: in * out * num_basis_fns
            # spline_scaler: in * out
            # base_scaler:   in * out
            layer_params = current_features * out_features * (num_basis_fns + 2)
            total_params += layer_params
            current_features = out_features
        return total_params

    def summary(self, **kwargs: Any) -> None:
        super().summary(**kwargs)
        print("\n" + self.get_architecture_summary())

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        # The functional model is rebuilt from inputs/outputs, so we only need
        # to store the original construction parameters.
        return {
            "layer_configs": self.layer_configs,
            "input_features": self.input_features,
            "enable_debugging": self.enable_debugging,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KAN":
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
    if metrics is None:
        metrics = ["accuracy"]

    model = KAN.from_variant(
        variant=variant,
        input_features=input_features,
        num_classes=num_classes,
        **model_kwargs
    )

    if isinstance(optimizer, str):
        optimizer_instance = keras.optimizers.get(optimizer)
        if hasattr(optimizer_instance, "learning_rate"):
            optimizer_instance.learning_rate.assign(learning_rate)
    else:
        optimizer_instance = optimizer

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

# ---------------------------------------------------------------------
