"""
PowerMLP, a dual-branch feedforward network that replaces KAN's B-spline bases with
ReLU-k activations, with configurable power, dropout and batch normalization.

Kolmogorov-Arnold Networks move the nonlinearity from the nodes to the edges: each
connection carries its own learned univariate function, parameterized as a B-spline
over a grid. That is what gives KAN its expressiveness per parameter, and it is
also what makes it slow — every forward pass must locate each input in the spline
grid and evaluate the basis polynomials there, an irregular, memory-bound
computation that does not reduce to a matrix multiply. PowerMLP starts from the
observation that the *shape* KAN buys with splines can be approximated by a fixed
nonlinearity of the right order combined with a linear map, recovering dense
GEMM-shaped compute.

The layer is two branches summed:

`y = ReLU_k(W_main x + b) + W_basis * swish(x)`

The main branch is a dense projection followed by `ReLU_k(z) = max(0, z)^k`. Raising
ReLU to an integer power `k` makes the branch piecewise-polynomial of degree `k`
rather than piecewise-linear, so a single layer can bend where a ReLU layer would
need several to approximate the same curvature — the same degree-of-freedom KAN
gets from its splines, but as an elementwise power on an already-projected vector.
The branch ordering matters and is easy to state backwards: the dense map comes
*first* and the power is applied to its output, so `k` acts on learned features
rather than on raw inputs.

The basis branch applies `swish(x) = x * sigmoid(x)` to the *input* and then
projects it. Swish is smooth, non-monotonic and unbounded above, which makes it a
complementary shape to `ReLU_k`: it is nonzero for negative inputs, where `ReLU_k`
is identically zero and its gradient vanishes. Summing the two means a unit is
never fully dead — whatever the main branch gates off, the basis branch still
passes a signal and a gradient. The basis projection is deliberately bias-free;
both branches carrying a bias would be redundant, and the main branch already has
one.

Note that the basis branch is a *stateless* activation followed by a linear map. It
adds no learned nonlinearity of its own, unlike a KAN edge function; the learning
in that branch lives entirely in `W_basis`. This is the trade PowerMLP makes, and
it is why it is faster rather than merely cheaper.

The model stacks these layers according to `hidden_units`, which is read as
`[input_dim, hidden_1, ..., hidden_n, output_dim]` — the first entry describes the
expected input width rather than creating a layer, and the last entry sizes the
output. Optional batch normalization and dropout are applied after each hidden
layer, in that order. The output layer is a plain `Dense`, not a PowerMLP layer:
the final map needs an arbitrary activation (softmax, sigmoid, or none for
regression), and `ReLU_k` on the logits would clamp them non-negative and destroy
the parameterization every downstream loss expects.

`k` is a fixed integer hyperparameter, validated as such at construction — it is
not learned. The preset variants raise it with model size (2 for micro, 3 through
base, 4 for the two largest), since higher-degree units are only worth their
conditioning cost when there is enough width to use them. Large `k` sharpens the
activation's gradient near the origin and grows its outputs fast, which is why
batch normalization becomes worth enabling as `k` rises.

References:
    - Liu et al., 2024. KAN: Kolmogorov-Arnold Networks.
      (https://arxiv.org/abs/2404.19756)
    - Ramachandran et al., 2017. Searching for Activation Functions.
      (https://arxiv.org/abs/1710.05941)
    - Ioffe & Szegedy, 2015. Batch Normalization: Accelerating Deep Network
      Training by Reducing Internal Covariate Shift.
      (https://arxiv.org/abs/1502.03167)
"""


import os
import keras
from typing import List, Optional, Union, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn.power_mlp_layer import PowerMLPLayer


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PowerMLP(keras.Model):
    """
    PowerMLP model: Efficient alternative to Kolmogorov-Arnold Networks.

    This model provides a complete Keras Model implementation of PowerMLP, offering
    superior computational efficiency compared to KAN while maintaining competitive
    performance across various tasks. The dual-branch architecture combines the
    expressiveness of nonlinear transformations with computational efficiency.

    **Intent**: Provide a production-ready, efficient alternative to KAN that can be
    easily integrated into existing Keras workflows while offering significant
    computational advantages for practical deep learning applications. The model is
    designed to be drop-in compatible with standard Keras workflows while providing
    ~40x faster training and ~10x fewer FLOPs compared to equivalent KAN networks.

    **Architecture**:
    The model consists of a sequence of PowerMLPLayer instances, each implementing
    a dual-branch design that combines dense transformations with basis functions.
    Optional regularization techniques (dropout, batch normalization) can be applied
    between layers for improved generalization.

    ```
    Input(shape=[batch, input_dim])
           ↓
    PowerMLPLayer_1(units=hidden_1)
           ↓ [main: Dense→ReLU-k] + [basis: BasisFunc→Dense]
    [BatchNorm_1] (optional)
           ↓
    [Dropout_1] (optional)
           ↓
    PowerMLPLayer_2(units=hidden_2)
           ↓ [main: Dense→ReLU-k] + [basis: BasisFunc→Dense]
    [BatchNorm_2] (optional)
           ↓
    [Dropout_2] (optional)
           ↓
    ...
           ↓
    Dense(units=output_dim, activation=output_activation)
           ↓
    Output(shape=[batch, output_dim])
    ```

    **Component Details**:
    - **PowerMLPLayer**: Dual-branch layer with main (Dense→ReLU-k) and basis branches
    - **BatchNormalization**: Optional normalization for training stability (especially for k > 3)
    - **Dropout**: Optional regularization to prevent overfitting
    - **Output Layer**: Standard Dense layer with configurable activation
    - **Flexibility**: Support for arbitrary depth and width configurations

    **Configuration Format**:
    The `hidden_units` list is interpreted as `[input_dim, hidden_1_units, ..., hidden_n_units, output_units]`:
    - First element: Input dimension (must match input data)
    - Middle elements: Hidden layer dimensions (PowerMLPLayers)
    - Last element: Output dimension (standard Dense layer)

    Args:
        hidden_units: List of integers specifying the number of units for the
            input, hidden, and output layers. The first element is the input
            dimension, the last is the output dimension. Must have at least two
            elements (input and output). For example, `[784, 128, 64, 10]` creates
            a network with 784 input features, two hidden PowerMLP layers with
            128 and 64 units respectively, and 10 output units.
        k: Integer, power exponent for the ReLU-k activation function in main branch.
            Must be positive. Higher values create more aggressive nonlinearities.
            Recommended range: 2-5. Higher k values may require batch normalization
            and gradient clipping. Defaults to 3.
        kernel_initializer: Initializer for kernel weights in all layers.
            Can be string name ('glorot_uniform', 'he_normal') or Initializer instance.
            Defaults to "he_normal" which is appropriate for ReLU-like activations.
        bias_initializer: Initializer for bias vectors in all layers.
            Can be string name or Initializer instance. Defaults to "zeros".
        kernel_regularizer: Optional regularizer function applied to kernel weights.
            Can be string name ('l1', 'l2', 'l1_l2') or Regularizer instance.
            Helps prevent overfitting by penalizing large weights. Defaults to None.
        bias_regularizer: Optional regularizer function applied to bias vectors.
            Can be string name or Regularizer instance. Defaults to None.
        use_bias: Boolean, whether to use bias in the main branch dense layers.
            The basis branch never uses bias by design. Defaults to True.
        output_activation: Optional activation function for the final output layer.
            Can be string name ('softmax', 'sigmoid') or callable. None for linear
            output (regression). Use 'softmax' for multi-class classification,
            'sigmoid' for binary classification, None for regression. Defaults to None.
        dropout_rate: Float between 0 and 1, dropout rate applied after each
            hidden layer. Set to 0.0 to disable dropout. Higher values provide
            stronger regularization. Typical values: 0.1-0.5. Defaults to 0.0.
        batch_normalization: Boolean, whether to apply batch normalization
            after each hidden layer. Recommended when k > 3 for training stability.
            Can improve convergence and generalization. Defaults to False.
        name: Optional string name for the model. Defaults to "power_mlp".
        **kwargs: Additional keyword arguments passed to the Model parent class,
            such as `trainable`, `dtype`, etc.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        `input_dim` should match the first element of `hidden_units`.
        Typically 2D: `(batch_size, input_dim)` for flat feature vectors.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., output_dim)` where output_dim is the
        last element in `hidden_units`. For classification with softmax, values sum to 1.

    Attributes:
        hidden_units: List of layer dimensions including input and output.
        k: Power exponent for ReLU-k activation.
        hidden_layers: List of PowerMLPLayer instances for feature transformation.
        dropout_layers: List of optional Dropout layers for regularization.
        batch_norm_layers: List of optional BatchNormalization layers.
        output_layer: Final Dense layer for output generation.
        kernel_initializer: Initializer for kernel weights (serialized form).
        bias_initializer: Initializer for bias vectors (serialized form).
        kernel_regularizer: Regularizer for kernel weights (serialized form).
        bias_regularizer: Regularizer for bias vectors (serialized form).
        use_bias: Whether bias is used in layers.
        output_activation: Output activation function (serialized form).
        dropout_rate: Dropout probability.
        batch_normalization: Whether batch normalization is enabled.

    Raises:
        ValueError: If hidden_units has fewer than two elements or contains non-positive values.
        TypeError: If k is not an integer.
        ValueError: If k is not a positive integer.
        ValueError: If dropout_rate is not in [0, 1].

    Example:
        ```python
        import keras
        import numpy as np

        # Standard classification model with input_dim=784 (MNIST)
        model = PowerMLP(
            hidden_units=[784, 128, 64, 10],
            k=3,
            dropout_rate=0.2,
            output_activation="softmax"
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Regression model with batch normalization
        model = PowerMLP(
            hidden_units=[100, 256, 128, 1],
            k=4,
            batch_normalization=True,
            output_activation=None
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Using model variants for CIFAR-10 (input_dim=3072)
        model = PowerMLP.from_variant("base", num_classes=10, input_dim=3072)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Train the model
        model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
        ```

    Note:
        For models, Keras automatically handles sub-layer building, so no custom
        build() method is needed. The model can be compiled and trained directly
        after instantiation. All sub-layers are created in __init__ and built
        automatically on first call.
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "micro": {"hidden_units": [32, 16], "k": 2},
        "tiny": {"hidden_units": [64, 32], "k": 3},
        "small": {"hidden_units": [128, 64, 32], "k": 3},
        "base": {"hidden_units": [256, 128, 64], "k": 3},
        "large": {"hidden_units": [512, 256, 128], "k": 4},
        "xlarge": {"hidden_units": [1024, 512, 256, 128], "k": 4},
    }

    def __init__(
        self,
        hidden_units: List[int],
        k: int = 3,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        use_bias: bool = True,
        output_activation: Optional[Union[str, callable]] = None,
        dropout_rate: float = 0.0,
        batch_normalization: bool = False,
        name: Optional[str] = "power_mlp",
        **kwargs: Any
    ) -> None:
        """
        Initialize the PowerMLP model.

        Creates all sub-layers in the network including PowerMLPLayers, optional
        dropout and batch normalization layers, and the output layer. All layers
        are instantiated but not yet built (building happens on first call).

        Args:
            hidden_units: List of integers for layer sizes, including input and output.
            k: Power for ReLU-k activation.
            kernel_initializer: Initializer for kernel weights.
            bias_initializer: Initializer for bias vector.
            kernel_regularizer: Regularizer for kernel weights.
            bias_regularizer: Regularizer for bias vector.
            use_bias: Whether to use bias in main branch.
            output_activation: Activation function for output layer.
            dropout_rate: Dropout rate for regularization.
            batch_normalization: Whether to use batch normalization.
            name: Optional name for the model.
            **kwargs: Additional keyword arguments for Model parent class.

        Raises:
            ValueError: If parameters are invalid.
            TypeError: If k is not an integer.
        """
        super().__init__(name=name, **kwargs)

        # Validate parameters
        self._validate_parameters(hidden_units, k, dropout_rate)

        # Store configuration parameters for serialization
        self.hidden_units = list(hidden_units)  # Make a copy
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.use_bias = use_bias
        self.output_activation = keras.activations.get(output_activation)
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization

        # CREATE all sub-layers in __init__ (Golden Rule)
        self.hidden_layers: List[PowerMLPLayer] = []
        self.dropout_layers: List[Optional[keras.layers.Dropout]] = []
        self.batch_norm_layers: List[Optional[keras.layers.BatchNormalization]] = []

        # Create hidden layers
        self._create_hidden_layers()

        # Create output layer
        self._create_output_layer()

        logger.info(
            f"Initialized PowerMLP model '{self.name}' with architecture "
            f"{self.hidden_units}, k={self.k}, dropout={self.dropout_rate}, "
            f"batch_norm={self.batch_normalization}"
        )

    def _validate_parameters(
        self,
        hidden_units: List[int],
        k: int,
        dropout_rate: float
    ) -> None:
        """
        Validate initialization parameters.

        Ensures all parameters are within valid ranges and of correct types.
        Raises descriptive errors for invalid configurations.

        Args:
            hidden_units: List of layer sizes.
            k: Power for ReLU-k activation.
            dropout_rate: Dropout rate.

        Raises:
            ValueError: If any parameter is invalid.
            TypeError: If k is not an integer.
        """
        if not hidden_units or len(hidden_units) < 2:
            raise ValueError(
                "hidden_units must contain at least an input and output size, "
                f"got {len(hidden_units)} elements"
            )
        if any(units <= 0 for units in hidden_units):
            raise ValueError(
                f"All hidden_units must be positive, got {hidden_units}"
            )
        if not isinstance(k, int):
            raise TypeError(
                f"k must be an integer, got type {type(k).__name__}"
            )
        if k <= 0:
            raise ValueError(
                f"k must be a positive integer, got {k}"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )

    def _create_hidden_layers(self) -> None:
        """
        Create the hidden PowerMLP layers with optional dropout and batch norm.

        Instantiates all hidden layers (PowerMLPLayer) along with their optional
        regularization layers (Dropout, BatchNormalization). The hidden layers
        correspond to elements from the second to the second-to-last in hidden_units.
        """
        # Hidden layers: hidden_units[1:-1]
        for i, units in enumerate(self.hidden_units[1:-1]):
            # PowerMLP layer
            power_mlp_layer = PowerMLPLayer(
                units=units,
                k=self.k,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                use_bias=self.use_bias,
                name=f"powermlp_hidden_{i + 1}"
            )
            self.hidden_layers.append(power_mlp_layer)

            # Batch normalization layer (optional)
            if self.batch_normalization:
                bn_layer = keras.layers.BatchNormalization(
                    name=f"batch_norm_{i + 1}"
                )
                self.batch_norm_layers.append(bn_layer)
            else:
                self.batch_norm_layers.append(None)

            # Dropout layer (optional)
            if self.dropout_rate > 0.0:
                dropout_layer = keras.layers.Dropout(
                    rate=self.dropout_rate,
                    name=f"dropout_{i + 1}"
                )
                self.dropout_layers.append(dropout_layer)
            else:
                self.dropout_layers.append(None)

    def _create_output_layer(self) -> None:
        """
        Create the output layer.

        Uses a standard Dense layer for the output to allow flexible activation
        functions. The output units correspond to the last element in hidden_units.
        """
        output_units = self.hidden_units[-1]

        # Use regular Dense layer for output to allow flexible activation
        self.output_layer = keras.layers.Dense(
            units=output_units,
            activation=self.output_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            use_bias=self.use_bias,
            name="output_layer"
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass for the PowerMLP model.

        Processes inputs through all hidden layers with optional regularization,
        then through the output layer. Keras automatically builds sub-layers on
        first call.

        Args:
            inputs: Input tensor with shape `(batch_size, ..., input_dim)` where
                input_dim must match the first element of hidden_units.
            training: Boolean indicating whether the layer should behave in
                training mode (apply dropout) or inference mode (no dropout).
                If None, Keras infers from the current execution context.

        Returns:
            Output tensor with shape `(batch_size, ..., output_dim)` where
            output_dim is the last element in hidden_units.
        """
        x = inputs

        # Pass through hidden layers with optional regularization
        for i, layer in enumerate(self.hidden_layers):
            # PowerMLP layer
            x = layer(x, training=training)

            # Optional batch normalization
            if self.batch_normalization and self.batch_norm_layers[i] is not None:
                x = self.batch_norm_layers[i](x, training=training)

            # Optional dropout
            if self.dropout_rate > 0.0 and self.dropout_layers[i] is not None:
                x = self.dropout_layers[i](x, training=training)

        # Output layer
        outputs = self.output_layer(x, training=training)

        return outputs

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the model.

        Args:
            input_shape: Shape tuple of input tensor, typically
                `(batch_size, input_dim)`.

        Returns:
            Output shape tuple: `(batch_size, output_dim)` where output_dim
            is the last element in hidden_units.
        """
        return input_shape[:-1] + (self.hidden_units[-1],)

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns a dictionary containing all constructor parameters needed to
        recreate this model. Serializes complex objects like initializers and
        regularizers properly.

        Returns:
            Dictionary containing the complete model configuration including
            all constructor parameters and their serialized forms.
        """
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "k": self.k,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "use_bias": self.use_bias,
            "output_activation": keras.activations.serialize(self.output_activation),
            "dropout_rate": self.dropout_rate,
            "batch_normalization": self.batch_normalization,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PowerMLP":
        """
        Create PowerMLP model from configuration dictionary.

        Deserializes complex objects (initializers, regularizers, activations)
        from their serialized forms and instantiates the model.

        Args:
            config: Configuration dictionary as returned by get_config().

        Returns:
            PowerMLP model instance reconstructed from the configuration.
        """
        # Deserialize complex objects
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "bias_initializer" in config and isinstance(config["bias_initializer"], dict):
            config["bias_initializer"] = keras.initializers.deserialize(
                config["bias_initializer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if "bias_regularizer" in config and config["bias_regularizer"]:
            config["bias_regularizer"] = keras.regularizers.deserialize(
                config["bias_regularizer"]
            )
        if "output_activation" in config and isinstance(config["output_activation"], dict):
            config["output_activation"] = keras.activations.deserialize(
                config["output_activation"]
            )

        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int,
        input_dim: int,
        **kwargs: Any
    ) -> "PowerMLP":
        """
        Create a PowerMLP model from a predefined variant.

        Provides convenient access to pre-configured model architectures
        optimized for different scales and use cases. Variants differ in
        depth, width, and k value.

        Args:
            variant: String identifier for the model variant. Must be one of:
                "micro", "tiny", "small", "base", "large", "xlarge".
            num_classes: Integer, number of output classes or output dimension.
                For classification, this is the number of classes. For regression,
                typically 1.
            input_dim: Integer, input feature dimension. Must match the dimensionality
                of your input data (e.g., 784 for flattened MNIST, 3072 for flattened CIFAR-10).
            **kwargs: Additional arguments passed to the constructor to override
                variant defaults. Can include dropout_rate, batch_normalization,
                output_activation, etc.

        Returns:
            PowerMLP model instance configured according to the specified variant.

        Raises:
            ValueError: If variant is not recognized.

        Example:
            >>> # CIFAR-10 model (flattened 32x32x3 = 3072)
            >>> model = PowerMLP.from_variant("base", num_classes=10, input_dim=3072)
            >>>
            >>> # MNIST model (flattened 28x28 = 784)
            >>> model = PowerMLP.from_variant("small", num_classes=10, input_dim=784)
            >>>
            >>> # Custom regression model with k override
            >>> model = PowerMLP.from_variant(
            ...     "large",
            ...     num_classes=1,
            ...     input_dim=100,
            ...     k=5,
            ...     batch_normalization=True
            ... )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # Start with variant defaults
        config = cls.MODEL_VARIANTS[variant].copy()

        # Allow kwargs to override variant defaults
        config.update(kwargs)

        # Construct the full hidden_units list
        base_hidden_units = cls.MODEL_VARIANTS[variant]["hidden_units"]
        config["hidden_units"] = [input_dim] + base_hidden_units + [num_classes]

        logger.info(f"Creating PowerMLP-{variant.upper()} model")
        logger.info(f"Architecture: {config['hidden_units']}")

        return cls(**config)

    def save_model(
        self,
        filepath: str,
        overwrite: bool = True,
        save_format: str = "keras"
    ) -> None:
        """
        Save the model to a file.

        Convenience method that ensures the directory exists before saving.
        Uses the standard Keras save format (.keras) by default.

        Args:
            filepath: Path where to save the model. Should end with '.keras'.
            overwrite: Whether to overwrite existing file. Defaults to True.
            save_format: Format to save the model in. Defaults to "keras".
        """
        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save model
        self.save(filepath, overwrite=overwrite, save_format=save_format)
        logger.info(f"PowerMLP model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "PowerMLP":
        """
        Load a saved PowerMLP model.

        Args:
            filepath: Path to the saved model file.

        Returns:
            Loaded PowerMLP model ready for inference or continued training.
        """
        # Note: With proper @keras.saving.register_keras_serializable() decorator,
        # custom_objects may not be strictly necessary, but we include them for robustness
        custom_objects = {
            "PowerMLP": cls,
            "PowerMLPLayer": PowerMLPLayer,
        }

        model = keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info(f"PowerMLP model loaded from {filepath}")
        return model

    def summary(self, **kwargs: Any) -> None:
        """
        Print model summary with additional PowerMLP-specific information.

        Extends the standard Keras summary with PowerMLP configuration details
        including architecture, parameter counts, and regularization settings.

        Args:
            **kwargs: Additional keyword arguments passed to Model.summary(),
                such as `line_length`, `positions`, `print_fn`.
        """
        # Build the model first if it hasn't been built
        if not self.built:
            # We need an input shape to build. We can infer from hidden_units.
            input_dim = self.hidden_units[0]
            self.build((None, input_dim))

        # Print standard Keras summary
        super().summary(**kwargs)

        # Print PowerMLP-specific configuration
        logger.info("\nPowerMLP Configuration:")
        logger.info(f"  - Architecture (input→hidden→output): {self.hidden_units}")
        logger.info(f"  - ReLU-k power: {self.k}")
        logger.info(f"  - Total parameters: {self.count_params():,}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Batch normalization: {self.batch_normalization}")
        logger.info(f"  - Output activation: {keras.activations.serialize(self.output_layer.activation)}")
        logger.info(f"  - Use bias: {self.use_bias}")

    def __repr__(self) -> str:
        """
        Return string representation of the model.

        Provides a concise, readable representation useful for debugging
        and logging.

        Returns:
            String representation including key parameters.
        """
        return (
            f"PowerMLP(hidden_units={self.hidden_units}, k={self.k}, "
            f"dropout_rate={self.dropout_rate}, name='{self.name}')"
        )


# ---------------------------------------------------------------------
# Helper functions to create and compile PowerMLP models
# ---------------------------------------------------------------------

def create_power_mlp(
    hidden_units: List[int],
    k: int = 3,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    loss: Union[str, keras.losses.Loss] = "categorical_crossentropy",
    metrics: Optional[List[Union[str, keras.metrics.Metric]]] = None,
    **kwargs: Any
) -> PowerMLP:
    """
    Create and compile a PowerMLP model.

    This is a convenience function that creates a PowerMLP model and compiles it
    with the specified optimizer, loss, and metrics in a single call.

    Args:
        hidden_units: List of integers specifying the number of units in each layer.
            Format: [input_dim, hidden_1, ..., hidden_n, output_dim].
        k: Power exponent for the ReLU-k activation function. Defaults to 3.
        optimizer: Optimizer name or instance. If string, learning_rate will be
            applied. Defaults to "adam".
        learning_rate: Learning rate for optimizer. Only used if optimizer is
            a string. Defaults to 0.001.
        loss: Loss function name or instance. Defaults to "categorical_crossentropy".
        metrics: List of metric names or instances. If None, defaults to ['accuracy'].
        **kwargs: Additional arguments for PowerMLP constructor, such as
            dropout_rate, batch_normalization, output_activation, etc.

    Returns:
        Compiled PowerMLP model ready for training with model.fit().

    Example:
        >>> model = create_power_mlp(
        ...     hidden_units=[784, 128, 64, 10],
        ...     k=3,
        ...     optimizer='adam',
        ...     learning_rate=0.001,
        ...     loss='categorical_crossentropy',
        ...     metrics=['accuracy'],
        ...     dropout_rate=0.2
        ... )
        >>> model.fit(x_train, y_train, epochs=10, validation_split=0.2)
    """
    # Create model
    model = PowerMLP(hidden_units=hidden_units, k=k, **kwargs)

    # Handle optimizer
    if isinstance(optimizer, str):
        optimizer = keras.optimizers.get(optimizer)
    if hasattr(optimizer, 'learning_rate'):
        optimizer.learning_rate = learning_rate

    # Default metrics
    if metrics is None:
        metrics = ['accuracy']

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    logger.info(
        f"Created and compiled PowerMLP model with architecture "
        f"{hidden_units[1:-1]} (hidden layers)"
    )
    return model


def create_power_mlp_regressor(
    hidden_units: List[int],
    k: int = 3,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs: Any
) -> PowerMLP:
    """
    Create and compile a PowerMLP model for regression tasks.

    Convenience function that sets up appropriate loss and metrics for
    regression problems (MSE loss, MAE metric, no output activation).

    Args:
        hidden_units: List of integers specifying the number of units in each layer.
            The last element should typically be 1 for single-target regression,
            or N for multi-target regression.
        k: Power exponent for the ReLU-k activation function. Defaults to 3.
        optimizer: Optimizer name or instance. Defaults to "adam".
        learning_rate: Learning rate for optimizer. Defaults to 0.001.
        **kwargs: Additional arguments for PowerMLP constructor.

    Returns:
        Compiled PowerMLP model configured for regression.

    Example:
        >>> model = create_power_mlp_regressor(
        ...     hidden_units=[100, 256, 128, 1],
        ...     k=4,
        ...     learning_rate=0.001,
        ...     batch_normalization=True
        ... )
        >>> model.fit(x_train, y_train, epochs=50)
    """
    return create_power_mlp(
        hidden_units=hidden_units,
        k=k,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss="mse",
        metrics=["mae", "mse"],
        output_activation=None,  # Linear output for regression
        **kwargs
    )


def create_power_mlp_binary_classifier(
    hidden_units: List[int],
    k: int = 3,
    optimizer: Union[str, keras.optimizers.Optimizer] = "adam",
    learning_rate: float = 0.001,
    **kwargs: Any
) -> PowerMLP:
    """
    Create and compile a PowerMLP model for binary classification.

    Convenience function that sets up appropriate loss, metrics, and activation
    for binary classification problems (BCE loss, sigmoid activation).

    Args:
        hidden_units: List of integers specifying the number of units in each layer.
            The last element should be 1 for binary classification.
        k: Power exponent for the ReLU-k activation function. Defaults to 3.
        optimizer: Optimizer name or instance. Defaults to "adam".
        learning_rate: Learning rate for optimizer. Defaults to 0.001.
        **kwargs: Additional arguments for PowerMLP constructor.

    Returns:
        Compiled PowerMLP model configured for binary classification.

    Example:
        >>> model = create_power_mlp_binary_classifier(
        ...     hidden_units=[200, 512, 256, 128, 1],
        ...     k=3,
        ...     dropout_rate=0.3,
        ...     learning_rate=0.0005
        ... )
        >>> model.fit(x_train, y_train, epochs=20)
    """
    # Ensure output is configured for binary classification
    if hidden_units[-1] != 1:
        logger.warning(
            f"For binary classification, output should be 1 unit, got {hidden_units[-1]}. "
            "Consider adjusting hidden_units to end with 1."
        )

    return create_power_mlp(
        hidden_units=hidden_units,
        k=k,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"],
        output_activation="sigmoid",
        **kwargs
    )

# ---------------------------------------------------------------------