"""
Kolmogorov-Arnold Network (KAN)
====================================================================

A complete implementation of the KAN architecture using modern Keras 3 patterns.
KAN uses learnable activation functions on edges rather than nodes, providing
a more flexible alternative to traditional MLPs with fixed activation functions.

Based on: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
https://arxiv.org/abs/2404.19756

Usage Examples:
-------------
```python
# Create standard KAN
model = create_kan_model("small", input_features=784, output_features=10)

# Update grids with training data (Critical step for KANs)
model.update_kan_grids(x_train[:1000])

# Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
```
"""

import os
import keras
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np

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
    activation functions on edges.

    **Architecture**:
    ```
    Input(shape=[input_features])
           ↓
    KANLinear(layer_configs[0])
           ↓
         ...
           ↓
    KANLinear(layer_configs[-1]) -> Produces logits/raw output
           ↓
    Activation(final_activation) -> Optional final transform
           ↓
    Output
    ```

    Args:
        layer_configs: List of dictionaries, each containing KANLinear configuration.
        input_features: Integer, number of input features. Must be positive.
        name: Optional string name for the model.
        **kwargs: Additional arguments passed to the Model base class.
    """

    # Model variant configurations
    VARIANT_CONFIGS = {
        "micro":  {"hidden_features": [16, 8], "grid_size": 3, "spline_order": 3, "activation": "swish"},
        "small":  {"hidden_features": [64, 32, 16], "grid_size": 5, "spline_order": 3, "activation": "swish"},
        "medium": {"hidden_features": [128, 64, 32], "grid_size": 7, "spline_order": 3, "activation": "gelu"},
        "large":  {"hidden_features": [256, 128, 64, 32], "grid_size": 10, "spline_order": 3, "activation": "gelu"},
        "xlarge": {"hidden_features": [512, 256, 128, 64], "grid_size": 12, "spline_order": 3, "activation": "gelu"},
    }

    # Pretrained weights URLs (Placeholders - update with actual URLs when available)
    PRETRAINED_WEIGHTS = {
        "micro": {
            "mnist": "https://example.com/kan_micro_mnist.keras",
        },
        "small": {
            "mnist": "https://example.com/kan_small_mnist.keras",
            "cifar10": "https://example.com/kan_small_cifar10.keras",
        },
        "medium": {
            "mnist": "https://example.com/kan_medium_mnist.keras",
            "cifar10": "https://example.com/kan_medium_cifar10.keras",
        },
        "large": {
            "cifar10": "https://example.com/kan_large_cifar10.keras",
        },
        "xlarge": {
            "cifar100": "https://example.com/kan_xlarge_cifar100.keras",
        },
    }

    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        input_features: int,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        if not isinstance(layer_configs, list) or not layer_configs:
            raise ValueError("layer_configs must be a non-empty list")
        if not isinstance(input_features, int) or input_features <= 0:
            raise ValueError(f"input_features must be positive integer, got {input_features}")

        self.layer_configs = self._validate_and_copy_configs(layer_configs)
        self.input_features = input_features
        self.num_layers = len(self.layer_configs)

        # Build the functional graph
        inputs, outputs = self._build_functional_model()

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=name or "kan_model",
            **kwargs
        )

        self._log_model_creation()

    def _log_model_creation(self):
        structure = [str(self.input_features)] + [str(cfg['features']) for cfg in self.layer_configs]
        logger.info(f"Created KAN model: {' -> '.join(structure)} ({self.num_layers} layers)")

    def _validate_and_copy_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated_configs = []
        for i, config in enumerate(configs):
            if not isinstance(config, dict):
                raise ValueError(f"Layer {i} config must be a dict, got {type(config)}")
            if 'features' not in config:
                raise ValueError(f"Layer {i} config missing required 'features' key")
            validated_configs.append(config.copy())
        return validated_configs

    def _build_functional_model(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Construct the Functional API graph."""
        inputs = keras.Input(shape=(self.input_features,), name="kan_input")
        x = inputs
        final_activation_fn = None

        for i, config in enumerate(self.layer_configs):
            layer_name = f"kan_layer_{i}"
            # Use a local copy to avoid modifying self.layer_configs during build
            kan_args = config.copy()
            is_last_layer = (i == self.num_layers - 1)

            if is_last_layer:
                # Extract the final network activation (e.g., 'softmax')
                # and force the KAN layer itself to be linear to avoid double activation.
                final_activation_fn = kan_args.pop('activation', 'linear')
                kan_args['activation'] = 'linear'

            kan_layer = KANLinear(name=layer_name, **kan_args)
            x = kan_layer(x)

        # Apply the distinct final activation (Softmax, Sigmoid, etc.) separately
        if final_activation_fn and final_activation_fn != 'linear':
            x = keras.layers.Activation(final_activation_fn, name="final_activation")(x)

        return inputs, x

    def update_kan_grids(self, x_data: Union[keras.KerasTensor, np.ndarray, Any]) -> None:
        """
        Update the B-spline grids of all KANLinear layers using the provided data.

        This is a critical step for KAN training. It performs a forward pass to
        collect the input distribution seen by each hidden layer, then adapts
        that layer's grid to match the distribution (quantile matching).

        Args:
            x_data: Batch of input data (numpy array or tensor). Should be a
                representative sample of the training data (e.g., 100-1000 samples).
        """
        kan_layers = [layer for layer in self.layers if isinstance(layer, KANLinear)]
        if not kan_layers:
            logger.warning("No KANLinear layers found to update.")
            return

        # To update hidden layers, we need their inputs.
        # We build a temporary model to extract intermediate activations.
        # For a functional model, layer.input gives the symbolic tensor feeding the layer.
        layer_inputs = [layer.input for layer in kan_layers]
        
        # Create a temporary extraction model
        # Note: self.input corresponds to the model's main input
        extraction_model = keras.Model(inputs=self.input, outputs=layer_inputs)
        
        # Run inference to get actual values
        # verbose=0 prevents progress bars for this utility op
        intermediate_values = extraction_model.predict(x_data, verbose=0)
        
        # Handle singleton case (predict returns array instead of list if 1 output)
        if len(kan_layers) == 1:
            intermediate_values = [intermediate_values]
            
        # Update each layer with its corresponding input distribution
        for layer, data in zip(kan_layers, intermediate_values):
            layer.update_grid_from_samples(data)
            
        logger.info(f"Updated grids for {len(kan_layers)} KAN layers.")

    def load_pretrained_weights(
        self,
        weights_path: str,
        skip_mismatch: bool = True,
        by_name: bool = True
    ) -> None:
        """Load pretrained weights into the model.

        Args:
            weights_path: String, path to the weights file (.keras format).
            skip_mismatch: Boolean, whether to skip layers with mismatched shapes.
            by_name: Boolean, whether to load weights by layer name.

        Raises:
            FileNotFoundError: If weights_path doesn't exist.
            ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            logger.info(f"Loading pretrained weights from {weights_path}")

            self.load_weights(
                weights_path,
                skip_mismatch=skip_mismatch,
                by_name=by_name
            )

            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped (e.g., output layer)."
                )
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    @staticmethod
    def _download_weights(
        variant: str,
        dataset: str = "mnist",
        cache_dir: Optional[str] = None
    ) -> str:
        """Download pretrained weights from URL."""
        if variant not in KAN.PRETRAINED_WEIGHTS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'. "
                f"Available variants: {list(KAN.PRETRAINED_WEIGHTS.keys())}"
            )

        if dataset not in KAN.PRETRAINED_WEIGHTS[variant]:
            raise ValueError(
                f"No pretrained weights available for dataset '{dataset}'. "
                f"Available datasets for {variant}: "
                f"{list(KAN.PRETRAINED_WEIGHTS[variant].keys())}"
            )

        url = KAN.PRETRAINED_WEIGHTS[variant][dataset]

        logger.info(f"Downloading KAN-{variant} weights from {dataset}...")

        weights_path = keras.utils.get_file(
            fname=f"kan_{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/kan"
        )

        logger.info(f"Weights downloaded to: {weights_path}")
        return weights_path

    @classmethod
    def from_variant(
        cls,
        variant: str,
        input_features: int,
        output_features: int,
        output_activation: Optional[str] = None,
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "mnist",
        weights_input_features: Optional[int] = None,
        cache_dir: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> "KAN":
        """Factory method to create KAN models from standard presets.

        Args:
            variant: "micro", "small", "medium", "large", or "xlarge".
            input_features: Dimension of input data.
            output_features: Dimension of output.
            output_activation: Final activation (e.g., "softmax", "sigmoid").
            pretrained: Boolean or string. If True, loads weights from default URL.
                        If string, treats as path to local weights file.
            weights_dataset: Dataset the weights were trained on (e.g., "mnist").
                             Only used if pretrained=True and not a local path.
            weights_input_features: Input dimension of the pretrained model.
                                    Used to detect mismatches.
            cache_dir: Directory to cache downloaded weights.
            override_config: Dictionary to override variant defaults.
            **kwargs: Arguments passed to KAN constructor.
        """
        if variant not in cls.VARIANT_CONFIGS:
            available = list(cls.VARIANT_CONFIGS.keys())
            raise ValueError(f"Unknown variant '{variant}'. Available: {available}")

        config_base = cls.VARIANT_CONFIGS[variant].copy()
        if override_config:
            config_base.update(override_config)

        hidden_features = config_base.pop("hidden_features")
        layer_configs = []

        # Build hidden layers configuration
        for features in hidden_features:
            config = config_base.copy()
            config["features"] = features
            layer_configs.append(config)

        # Build output layer configuration
        output_config = config_base.copy()
        output_config["features"] = output_features

        # Determine final activation
        if output_activation:
            output_config["activation"] = output_activation
        elif output_features > 1:
            output_config["activation"] = "softmax"
        else:
            output_config["activation"] = "linear"

        layer_configs.append(output_config)

        # Handle pretrained weights logic
        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                try:
                    load_weights_path = cls._download_weights(
                        variant=variant,
                        dataset=weights_dataset,
                        cache_dir=cache_dir
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {str(e)}. "
                        f"Continuing with random initialization."
                    )
                    load_weights_path = None

            # Detect mismatches
            # 1. Input mismatch
            if weights_input_features and weights_input_features != input_features:
                logger.info(
                    f"Pretrained input features ({weights_input_features}) differ from "
                    f"current ({input_features}). Input layer weights may be skipped."
                )
                skip_mismatch = True

            # 2. Output mismatch
            pretrained_classes = 10
            if weights_dataset == "cifar100":
                pretrained_classes = 100
            elif weights_dataset in ["mnist", "cifar10"]:
                pretrained_classes = 10

            if output_features != pretrained_classes:
                logger.info(
                    f"Output features ({output_features}) differ from pretrained "
                    f"({pretrained_classes}). Output layer weights will be skipped."
                )
                skip_mismatch = True

        # Create model instance
        model = cls(layer_configs=layer_configs, input_features=input_features, **kwargs)

        # Load weights if available
        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch,
                    by_name=True
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {str(e)}")
                raise

        return model

    @classmethod
    def from_layer_sizes(
        cls,
        layer_sizes: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        activation: str = "swish",
        final_activation: Optional[str] = None,
        **kan_layer_kwargs: Any
    ) -> "KAN":
        """Create a KAN by defining a list of node counts per layer."""
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements (input -> output)")

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

            # Last layer logic
            if i == len(output_feature_sizes) - 1:
                if final_activation:
                    config["activation"] = final_activation
                elif features > 1:
                    config["activation"] = "softmax"
                else:
                    config["activation"] = "linear"

            layer_configs.append(config)

        return cls(layer_configs=layer_configs, input_features=input_features)

    def get_architecture_summary(self) -> str:
        """Returns a formatted string summarizing the KAN architecture details."""
        lines = ["KAN Model Architecture Summary"]
        lines.append("=" * 50)

        total_features = [self.input_features] + [cfg["features"] for cfg in self.layer_configs]
        lines.append(f"Flow: {' -> '.join(map(str, total_features))}")
        lines.append(f"Total layers: {self.num_layers}")
        lines.append("-" * 50)

        for i, config in enumerate(self.layer_configs):
            # Determine what the activation effectively is for display
            is_last = (i == self.num_layers - 1)
            if is_last:
                act_display = config.get('activation', 'linear')
            else:
                act_display = config.get('activation', 'swish')

            lines.append(
                f"Layer {i:2d}: "
                f"Units={config['features']:<4d} | "
                f"Grid={config.get('grid_size', 'def'):<2} | "
                f"Order={config.get('spline_order', 'def'):<1} | "
                f"Act='{act_display}'"
            )

        lines.append("=" * 50)
        lines.append(f"Est. Parameters: ~{self._estimate_parameters():,}")
        return "\n".join(lines)

    def _estimate_parameters(self) -> int:
        """Estimate number of trainable parameters."""
        total_params = 0
        curr_in = self.input_features

        for config in self.layer_configs:
            curr_out = config["features"]
            grid = config.get("grid_size", 5)
            order = config.get("spline_order", 3)
            num_basis = grid + order

            # Param count logic:
            # 1. Spline weights: in * out * basis
            # 2. Spline scalers: in * out
            # 3. Base scalers:   in * out
            # Note: KANLinear does not currently implement a bias vector.
            
            layer_params = (curr_in * curr_out * num_basis) + (2 * curr_in * curr_out)

            total_params += layer_params
            curr_in = curr_out

        return total_params

    def summary(self, **kwargs: Any) -> None:
        super().summary(**kwargs)
        print("\n" + self.get_architecture_summary())

    def get_config(self) -> Dict[str, Any]:
        # Functional model only needs construction args to be serializable
        return {
            "layer_configs": self.layer_configs,
            "input_features": self.input_features,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KAN":
        return cls(**config)

# ---------------------------------------------------------------------
# factory method
# ---------------------------------------------------------------------

def create_kan_model(
    variant: str = "small",
    input_features: int = 784,
    output_features: int = 10,
    output_activation: Optional[str] = None,
    pretrained: Union[bool, str] = False,
    weights_dataset: str = "mnist",
    weights_input_features: Optional[int] = None,
    cache_dir: Optional[str] = None,
    **model_kwargs: Any
) -> KAN:
    """Helper to create a standard KAN model configuration.

    Args:
        variant: "micro", "small", "medium", "large", or "xlarge".
        input_features: Input dimension.
        output_features: Output dimension.
        output_activation: Final activation.
        pretrained: Load pretrained weights (bool or path string).
        weights_dataset: Dataset for pretrained weights (e.g. "mnist").
        weights_input_features: Original input features of pretrained model.
        cache_dir: Download cache location.
        **model_kwargs: Additional model arguments.

    Returns:
        Uncompiled KAN model.
    """
    model = KAN.from_variant(
        variant=variant,
        input_features=input_features,
        output_features=output_features,
        output_activation=output_activation,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_features=weights_input_features,
        cache_dir=cache_dir,
        **model_kwargs
    )
    return model
