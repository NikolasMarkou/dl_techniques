"""
A Random Fourier Features (RFF) mapping to approximate kernel methods.

This layer provides an efficient, scalable alternative to traditional kernel
machines (e.g., Support Vector Machines with an RBF kernel) by leveraging a
randomized feature map. The core problem that RFF addresses is the
computational and memory cost of kernel methods, which typically require the
computation and storage of an `N x N` kernel matrix, where `N` is the number
of data points. This `O(N^2)` complexity makes them intractable for large-scale
datasets.

This layer circumvents that limitation by creating an explicit, low-dimensional
feature mapping `φ(x)` such that the inner product in the new feature space
approximates the desired kernel function: `k(x, y) ≈ φ(x)^T φ(y)`. This
transforms the non-linear kernel problem into a linear one in the randomized
feature space, which can then be solved efficiently with standard methods like
linear regression or logistic regression.

Architecture and Core Concepts:

The architecture is composed of two main parts: a fixed, non-trainable random
feature mapping, followed by a standard trainable linear layer.

1.  **Random Feature Mapping:** This is the heart of the RFF method. The input
    data `x` is projected into a higher-dimensional space using a randomly
    generated and subsequently fixed weight matrix `ω` and bias vector `b`.
    This projection is then passed through a cosine non-linearity. The matrix
    `ω` and vector `b` are sampled from specific distributions dictated by the
    kernel one wishes to approximate. They are *not* learned during training;
    their role is to define a static, randomized basis that effectively
    approximates the kernel's feature space.

2.  **Trainable Linear Transformation:** The resulting random features `φ(x)` are
    then fed into a standard dense (fully-connected) layer. This is where all
    the learning occurs. The model learns to perform its task (e.g.,
    classification or regression) by finding the optimal linear combination of
    these fixed random features.

By separating the non-linear feature mapping (which is randomized and fixed)
from the model training (which becomes a simple linear problem), RFF achieves
significant computational savings while maintaining strong performance.

Mathematical Foundation:

The theoretical underpinning of this method is Bochner's theorem, which states
that any continuous, shift-invariant kernel `k(x, y) = k(x - y)` is the
Fourier transform of a non-negative measure. For the widely used Gaussian
RBF kernel, `k(x,y) = exp(-γ²||x-y||²)`, this corresponding measure is a
Gaussian distribution.

The RFF method is a Monte Carlo approximation of this Fourier integral. To
approximate the RBF kernel, we sample `D` random frequency vectors `ω_i` from a
Gaussian distribution `N(0, γ²I)` and `D` random phase shifts `b_i` from a
uniform distribution `Uniform(0, 2π)`.

The feature map `φ(x)` for an input `x` is then constructed as:
`φ(x) = sqrt(2/D) * [cos(ω_1^T x + b_1), ..., cos(ω_D^T x + b_D)]`

This mapping has the remarkable property that the expected value of the inner
product `E[φ(x)^T φ(y)]` is equal to the RBF kernel `k(x, y)`. For a
sufficiently large number of features `D`, the inner product provides a
high-quality approximation of the kernel, allowing a linear model trained on
`φ(x)` to effectively replicate the behavior of a powerful kernel machine.

References:

The foundational work that introduced Random Fourier Features is:
-   Rahimi, A., & Recht, B. (2007). "Random Features for Large-Scale Kernel
    Machines." This paper demonstrated how this randomized approach could
    drastically scale up kernel methods to datasets with millions of examples.

"""

import keras
import numpy as np
from typing import Optional, Union, Tuple, Any, Dict
from keras import ops, layers, initializers, regularizers, constraints

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RFFKernelLayer(keras.layers.Layer):
    """
    Random Fourier Features layer for efficient kernel approximation.

    This layer implements Random Fourier Features (RFF) which provide an efficient
    approximation to translation-invariant kernels using random projections. It achieves
    significant memory reduction (up to 1200x) compared to full kernel matrices while
    maintaining comparable accuracy.

    **Mathematical Foundation**:

    For translation-invariant kernels k(x, y) = k(x - y), the approximation is:

    .. math::
        k(x, y) ≈ \\frac{1}{D} \\sum_{i=1}^{D} \\cos(\\omega_i^T x + b_i) \\cos(\\omega_i^T y + b_i)

    Where:
    - ω_i ~ N(0, γ²I) are random frequencies sampled from a Gaussian
    - b_i ~ Uniform(0, 2π) are random phase shifts
    - D is the number of random features

    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           ↓
    Random Projection: z = x @ ω + b
           ↓
    Feature Map: φ(x) = √(2/D) * cos(z)
           ↓
    Linear Transform: output = φ(x) @ W
           ↓
    Output(shape=[..., output_dim])
    ```

    **Key Benefits**:
    - **Memory Efficiency**: O(D) memory instead of O(N²) for kernel matrices
    - **Linear Scaling**: Computational cost scales linearly with dataset size
    - **Accuracy Preservation**: Maintains kernel approximation quality

    Args:
        input_dim: Integer, dimensionality of the input features. Must be positive.
            This determines the size of the random projection matrix ω.
        output_dim: Integer, dimensionality of the output space. Must be positive.
            If None, defaults to input_dim for autoencoder-like architectures.
        n_features: Integer, number of random Fourier features. Must be positive.
            Higher values give better kernel approximation but increase memory.
            Typical values: 100-10000. Defaults to 1000.
        gamma: Float, bandwidth parameter for the RBF kernel. Must be positive.
            Controls the width of the kernel. Smaller values create wider kernels.
            Related to RBF kernel as k(x,y) = exp(-γ²||x-y||²). Defaults to 1.0.
        use_bias: Boolean, whether to use bias in the output linear layer.
            Defaults to True.
        activation: Optional activation function to apply after the linear transform.
            Can be string name ('relu', 'gelu') or callable. None means linear.
            Defaults to None.
        kernel_initializer: Initializer for the output weight matrix.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the output bias vector.
            Only used when use_bias=True. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for the output weight matrix.
        bias_regularizer: Optional regularizer for the output bias vector.
        kernel_constraint: Optional constraint for the output weight matrix.
        bias_constraint: Optional constraint for the output bias vector.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        Most common: 2D tensor with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., output_dim)`.
        Same rank as input, but last dimension changed to `output_dim`.

    Attributes:
        omega: Non-trainable weight matrix of shape (input_dim, n_features).
            Random frequencies sampled from N(0, gamma²).
        b: Non-trainable bias vector of shape (n_features,).
            Random phase shifts sampled from Uniform(0, 2π).
        linear: Dense layer for output transformation.

    Example:
        ```python
        # Basic RFF layer for kernel approximation
        rff_layer = RFFKernelLayer(
            input_dim=784,
            output_dim=128,
            n_features=2000,
            gamma=0.1
        )

        # Using in a model
        inputs = keras.Input(shape=(784,))
        features = rff_layer(inputs)  # Shape: (batch, 128)

        # For classification with RFF features
        rff = RFFKernelLayer(input_dim=256, output_dim=512, n_features=5000)
        classifier = keras.Sequential([
            rff,
            keras.layers.Dense(10, activation='softmax')
        ])

        # Memory-efficient kernel machine
        rff_kernel = RFFKernelLayer(
            input_dim=1024,
            output_dim=1024,  # Same as input for kernel-like behavior
            n_features=100,    # Much smaller than O(N²) kernel matrix
            gamma=0.01         # Wide kernel
        )
        ```

    References:
        - Random Features for Large-Scale Kernel Machines (Rahimi & Recht, 2007)
        - Weighted Sums of Random Kitchen Sinks (Rahimi & Recht, 2008)

    Note:
        The random features (omega and b) are fixed after initialization and not
        learned during training. This is intentional and key to the method's
        efficiency. Only the output linear transformation is learned.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: Optional[int] = None,
            n_features: int = 1000,
            gamma: float = 1.0,
            use_bias: bool = True,
            activation: Optional[Union[str, callable]] = None,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            kernel_constraint: Optional[constraints.Constraint] = None,
            bias_constraint: Optional[constraints.Constraint] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        # Store configuration
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.n_features = n_features
        self.gamma = gamma
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Validate output_dim
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")

        # Create output linear layer in __init__
        self.linear = layers.Dense(
            units=self.output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name='output_projection'
        )

        # Initialize weight attributes (created in build)
        self.omega = None
        self.b = None
        self._scale_factor = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights including random features and output projection.

        This method creates the non-trainable random features (omega and b) that
        form the basis of the RFF approximation, and builds the output Dense layer.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Validate input shape
        if input_shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match "
                f"input_dim ({self.input_dim})"
            )

        # Create random frequency matrix omega ~ N(0, gamma²I)
        # Non-trainable as these are fixed random features
        omega_init = keras.random.normal(
            shape=(self.input_dim, self.n_features),
            mean=0.0,
            stddev=self.gamma,
            seed=None  # Will use global seed
        )
        self.omega = self.add_weight(
            name='omega',
            shape=(self.input_dim, self.n_features),
            initializer=lambda shape, dtype: omega_init,
            trainable=False,  # Fixed random features
            dtype=self.compute_dtype
        )

        # Create random phase shifts b ~ Uniform(0, 2π)
        # Non-trainable as these are fixed random features
        b_init = keras.random.uniform(
            shape=(self.n_features,),
            minval=0.0,
            maxval=2.0 * np.pi,
            seed=None
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.n_features,),
            initializer=lambda shape, dtype: b_init,
            trainable=False,  # Fixed random features
            dtype=self.compute_dtype
        )

        # Pre-compute scale factor for efficiency
        self._scale_factor = ops.sqrt(2.0 / self.n_features)

        # Build the output Dense layer
        # The input to Dense will be the random features of shape (..., n_features)
        feature_shape = input_shape[:-1] + (self.n_features,)
        self.linear.build(feature_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply Random Fourier Features transformation.

        Computes the RFF approximation by:
        1. Projecting inputs through random frequencies
        2. Applying cosine transformation with phase shifts
        3. Scaling and passing through linear output layer

        Args:
            inputs: Input tensor of shape (batch_size, ..., input_dim).
            training: Boolean indicating training mode (unused but kept for compatibility).

        Returns:
            Output tensor of shape (batch_size, ..., output_dim).
        """
        # Random projection: z = x @ omega + b
        # Shape: (..., input_dim) @ (input_dim, n_features) -> (..., n_features)
        projection = ops.matmul(inputs, self.omega) + self.b

        # Generate random Fourier features: φ(x) = sqrt(2/D) * cos(z)
        # This creates the feature representation that approximates the kernel
        features = self._scale_factor * ops.cos(projection)

        # Apply linear transformation to get output
        # Shape: (..., n_features) -> (..., output_dim)
        output = self.linear(features, training=training)

        # Apply activation if specified
        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Shape tuple of the output.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing all configuration parameters.
        """
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'n_features': self.n_features,
            'gamma': self.gamma,
            'use_bias': self.use_bias,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config

# ---------------------------------------------------------------------
