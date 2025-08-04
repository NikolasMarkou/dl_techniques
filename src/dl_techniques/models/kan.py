import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.kan_linear import KANLinear


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KAN(keras.Model):
    """Kolmogorov-Arnold Network model with stability enhancements.

    This model stacks multiple KANLinear layers to create a deep network that can
    approximate complex multivariate functions using the Kolmogorov-Arnold representation.

    The model automatically handles layer building, provides comprehensive validation,
    and includes debugging capabilities for monitoring training progress.

    Args:
        layers_configurations: List of dictionaries, each containing configuration
            for a KANLinear layer. Each dict should have 'in_features' and 'out_features'
            keys, and optionally other KANLinear parameters like 'grid_size', 'spline_order',
            'activation', etc.
        enable_debugging: Whether to enable extra validation during forward pass.
            When True, logs layer outputs and checks for numerical issues.
        name: Optional name for the model. If None, defaults to 'kan_model'.
        **kwargs: Additional arguments passed to the Model constructor.

    Raises:
        ValueError: If layer configurations are empty, missing required keys,
            or have incompatible dimensions between consecutive layers.

    Example:
        >>> config = [
        ...     {"in_features": 10, "out_features": 20, "grid_size": 8, "activation": "swish"},
        ...     {"in_features": 20, "out_features": 10, "grid_size": 6, "activation": "gelu"},
        ...     {"in_features": 10, "out_features": 1, "grid_size": 5, "activation": "linear"}
        ... ]
        >>> model = KAN(layers_configurations=config, enable_debugging=True)
        >>> x = keras.random.normal((32, 10))
        >>> y = model(x)
        >>> print(y.shape)  # (32, 1)

        >>> # Model can be saved and loaded
        >>> model.save('kan_model.keras')
        >>> loaded_model = keras.models.load_model('kan_model.keras')
    """

    def __init__(
            self,
            layers_configurations: List[Dict[str, Any]],
            enable_debugging: bool = False,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize KAN model.

        Args:
            layers_configurations: List of layer configurations.
            enable_debugging: Whether to enable extra validation.
            name: Optional name for the model.
            **kwargs: Additional arguments for Model constructor.

        Raises:
            ValueError: If layer configurations are invalid or incompatible.
        """
        super().__init__(name=name or 'kan_model', **kwargs)

        # Validate and store configuration
        self.layers_configurations = self._deep_copy_configurations(layers_configurations)
        self.enable_debugging = enable_debugging
        self._validate_configurations(self.layers_configurations)

        # Initialize layers list - will be populated in build()
        self.kan_layers = []

        # Store build information for serialization
        self._build_input_shape = None
        self._is_built = False

        # Track model statistics for debugging
        self._layer_stats = {}

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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the KAN layers based on input shape.

        Args:
            input_shape: Shape of the input tensor.

        Raises:
            ValueError: If input shape is incompatible with first layer configuration.
        """
        # Handle None input_shape
        if input_shape is None:
            raise ValueError("Input shape cannot be None")

        # Store for serialization
        self._build_input_shape = input_shape

        # Validate input shape compatibility
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        input_features = input_shape[-1]
        expected_features = self.layers_configurations[0]['in_features']

        if input_features != expected_features:
            raise ValueError(
                f"Input features ({input_features}) don't match first layer "
                f"in_features ({expected_features})"
            )

        # Clear existing layers if rebuilding
        self.kan_layers = []

        # Create KAN layers with enhanced error handling
        for i, layer_config in enumerate(self.layers_configurations):
            try:
                layer_name = f"kan_layer_{i}"
                kan_layer = KANLinear(name=layer_name, **layer_config)
                self.kan_layers.append(kan_layer)

                if self.enable_debugging:
                    logger.info(
                        f"Created KAN layer {i}: {layer_config['in_features']} -> {layer_config['out_features']}")

            except Exception as e:
                raise ValueError(f"Failed to create KAN layer {i}: {str(e)}") from e

        # Build each layer sequentially
        current_shape = input_shape
        for i, layer in enumerate(self.kan_layers):
            try:
                layer.build(current_shape)
                # Update shape for next layer
                current_shape = layer.compute_output_shape(current_shape)

                if self.enable_debugging:
                    logger.info(f"Built KAN layer {i} with input shape {current_shape}")

            except Exception as e:
                raise ValueError(f"Failed to build KAN layer {i}: {str(e)}") from e

        self._is_built = True
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass with comprehensive validation and debugging.

        Args:
            inputs: Input tensor with shape (..., in_features).
            training: Whether the model is in training mode. If None, uses the model's
                current training mode.

        Returns:
            Output tensor after passing through all KAN layers.

        Raises:
            ValueError: If inputs have incompatible shape or contain invalid values.
        """
        # Input validation - check for None first
        if inputs is None:
            raise ValueError("Inputs cannot be None")

        # Check input shape compatibility
        input_shape = ops.shape(inputs)
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        # Handle empty batch gracefully
        batch_size = input_shape[0]
        if batch_size == 0:
            # Return appropriately shaped empty tensor
            final_out_features = self.layers_configurations[-1]['out_features']
            output_shape = list(input_shape[:-1]) + [final_out_features]
            return ops.zeros(output_shape, dtype=inputs.dtype)

        # Numerical validation in debug mode
        if self.enable_debugging and training:
            if ops.any(ops.isnan(inputs)):
                logger.warning("NaN values detected in model inputs")
            if ops.any(ops.isinf(inputs)):
                logger.warning("Infinite values detected in model inputs")

            logger.info(f"KAN forward pass - input shape: {input_shape}, training: {training}")

        # Forward pass through layers
        outputs = inputs

        for i, layer in enumerate(self.kan_layers):
            try:
                # Apply layer transformation
                outputs = layer(outputs, training=training)

                # Debug monitoring
                if self.enable_debugging:
                    output_shape = ops.shape(outputs)
                    logger.info(f"Layer {i} output shape: {output_shape}")

                    # Track statistics for debugging
                    if training:
                        output_mean = ops.mean(outputs)
                        output_std = ops.std(outputs)
                        self._layer_stats[f'layer_{i}'] = {
                            'output_mean': float(output_mean),
                            'output_std': float(output_std)
                        }

                    # Check for numerical issues
                    if ops.any(ops.isnan(outputs)):
                        logger.warning(f"NaN detected in layer {i} output")
                    if ops.any(ops.isinf(outputs)):
                        logger.warning(f"Infinite values detected in layer {i} output")

                    # Check for gradient explosion indicators
                    if training:
                        max_abs_value = ops.max(ops.abs(outputs))
                        if max_abs_value > 1e6:
                            logger.warning(f"Large values detected in layer {i}: max_abs = {float(max_abs_value)}")

            except Exception as e:
                logger.error(f"Error in KAN layer {i}: {str(e)}")
                raise ValueError(f"Forward pass failed at layer {i}: {str(e)}") from e

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the KAN model.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape after passing through all layers.
        """
        if not self.layers_configurations:
            return input_shape

        # Output shape has same dimensions as input except last (features) dimension
        input_shape_list = list(input_shape)
        final_out_features = self.layers_configurations[-1]['out_features']

        return tuple(input_shape_list[:-1] + [final_out_features])

    def get_layer_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics from the last forward pass (debug mode only).

        Returns:
            Dictionary containing layer-wise statistics.
        """
        return dict(self._layer_stats) if self.enable_debugging else {}

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

    def get_config(self) -> Dict[str, Any]:
        """Return the config of the model.

        Returns:
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            "layers_configurations": self.layers_configurations,
            "enable_debugging": self.enable_debugging,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the model from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the model from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "KAN":
        """Create a KAN model from a config dictionary.

        Args:
            config: Configuration dictionary containing at minimum
                'layers_configurations' and optionally other parameters.

        Returns:
            KAN model instance.

        Raises:
            ValueError: If config is missing required keys.
        """
        if 'layers_configurations' not in config:
            raise ValueError("Config must contain 'layers_configurations'")

        return cls(**config)


# ---------------------------------------------------------------------

def create_kan_model(
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
            E.g., [10, 20, 15, 1] creates a 3-layer network with input dim 10,
            hidden layers of size 20 and 15, and output dim 1.
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
        >>> model = create_kan_model([784, 128, 64, 10], grid_size=8, activation='gelu')
        >>> # Creates a 3-layer KAN: 784->128->64->10
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