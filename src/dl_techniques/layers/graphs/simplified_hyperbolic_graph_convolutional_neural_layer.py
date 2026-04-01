"""
Simplified Hyperbolic Graph Convolutional Neural Network Layer.

This module implements the sHGCN layer as defined in Arevalo et al., specifically
following Equation 14. The layer performs efficient graph convolution by switching
between Euclidean and hyperbolic spaces strategically: Euclidean for aggregation,
hyperbolic only for bias addition.

Mathematical Flow (Eq 14):
    H^l = σ(Ã log₀^c(exp₀^c(W H^{l-1}) ⊕_c exp₀^c(b)))

where:
    - W: Euclidean weight matrix
    - ⊕_c: Möbius addition (hyperbolic bias)
    - exp₀^c: Exponential map (tangent → manifold)
    - log₀^c: Logarithmic map (manifold → tangent)
    - Ã: Normalized adjacency matrix
    - σ: Euclidean activation function
"""

import keras
import tensorflow as tf
from typing import Optional, Union, List, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.geometry.poincare_math import PoincareMath

# ---------------------------------------------------------------------

# Initialize global math utility instance
_poincare_math = PoincareMath(eps=1e-5)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SHGCNLayer(keras.layers.Layer):
    """Simplified Hyperbolic Graph Convolutional Layer.

    Implements efficient hyperbolic graph convolution (Eq. 14 of Arevalo et al.)
    by performing neighbourhood aggregation in Euclidean tangent space while
    using hyperbolic geometry only for bias addition. The full forward pass is
    H^l = sigma(A_tilde log_0^c(exp_0^c(W H^{l-1}) oplus_c exp_0^c(b))), where
    oplus_c is Mobius addition, exp_0^c / log_0^c are the exponential / logarithmic
    maps at the origin, A_tilde is the normalised adjacency, and sigma is the
    Euclidean activation.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────┐
        │  Input: [features, adjacency]         │
        │         [N, D_in]    [N, N] (sparse)  │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  1. Euclidean Linear   Z = X @ W      │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  2. Exp Map   Z_hyp = exp₀^c(Z)      │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  3. Moebius Bias  H = Z_hyp ⊕_c b    │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  4. Project (numerical stability)     │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  5. Log Map   H_tan = log₀^c(H)      │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  6. Aggregate   Y = Ã @ H_tan         │
        └──────────────────┬────────────────────┘
                           ▼
        ┌───────────────────────────────────────┐
        │  7. Activation  Output = σ(Y)         │
        │     [N, units]                        │
        └───────────────────────────────────────┘

    :param units: Output dimensionality (features per node). Must be positive.
    :type units: int
    :param activation: Euclidean activation function. Defaults to ``'relu'``.
    :type activation: Union[str, callable]
    :param use_bias: Whether to add Mobius bias in hyperbolic space.
        Defaults to ``True``.
    :type use_bias: bool
    :param use_curvature: Whether curvature *c* is learnable. Defaults to ``True``.
    :type use_curvature: bool
    :param dropout_rate: Dropout probability in ``[0, 1)``. Defaults to 0.0.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for weight matrix *W*.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vector *b*.
        Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    """

    def __init__(
            self,
            units: int,
            activation: Union[str, callable] = 'relu',
            use_bias: bool = True,
            use_curvature: bool = True,
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            **kwargs: Any
    ) -> None:
        """Initialise the sHGCN layer."""
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")

        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.use_curvature = use_curvature
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Create dropout layer
        self.dropout = keras.layers.Dropout(dropout_rate)

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        """Create layer weights based on input shape.

        :param input_shape: List of ``[features_shape, adjacency_shape]``.
        :type input_shape: Union[Tuple, List[Tuple]]
        """
        # Input is [features, adjacency]
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"Input must be [features, adjacency], got {input_shape}"
            )

        feat_shape = input_shape[0]
        input_dim = feat_shape[-1]

        # Weight matrix W (Euclidean linear transformation)
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name='kernel',
            trainable=True
        )

        # Bias vector b (will be mapped to hyperbolic space)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='bias',
                trainable=True
            )

        # Curvature parameter c = softplus(c_theta) ensures c > 0
        # Initialize to ~0.54 so softplus(0.54) ≈ 1.0
        if self.use_curvature:
            self.c_theta = self.add_weight(
                shape=(),
                initializer=keras.initializers.Constant(0.54),
                name='curvature_theta',
                trainable=True
            )
        else:
            # Fixed curvature
            self.c_theta = tf.constant(0.54, dtype=tf.float32)

        # Build dropout layer
        self.dropout.build(feat_shape)

        super().build(input_shape)

    @property
    def curvature(self) -> keras.KerasTensor:
        """Get current curvature value c > 0.

        :return: Scalar tensor representing curvature ``c = softplus(c_theta)``.
        :rtype: keras.KerasTensor
        """
        return keras.ops.softplus(self.c_theta)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass implementing Equation 14.

        :param inputs: List of ``[features, adjacency]`` tensors.
        :type inputs: List[keras.KerasTensor]
        :param training: Whether in training mode (affects dropout).
        :type training: Optional[bool]
        :return: Updated node embeddings of shape ``[num_nodes, units]``.
        :rtype: keras.KerasTensor
        """
        x, adj = inputs

        # Step 0: Apply dropout to input features
        if training and self.dropout_rate > 0:
            x = self.dropout(x, training=training)

        # Step 1: Euclidean linear transformation
        # Z = X @ W: [N, input_dim] @ [input_dim, units] -> [N, units]
        z_euclid = keras.ops.matmul(x, self.kernel)

        # Get current curvature
        c = self.curvature

        # Step 2: Map to hyperbolic space (Poincaré ball)
        # Z_hyp = exp₀^c(Z): [N, units] -> [N, units]
        z_hyp = _poincare_math.exp_map_0(z_euclid, c)

        # Step 3: Möbius bias addition in hyperbolic space
        if self.use_bias:
            # Map bias to hyperbolic space: b_hyp = exp₀^c(b)
            b_hyp = _poincare_math.exp_map_0(self.bias, c)

            # Hyperbolic addition: H_hyp = Z_hyp ⊕_c b_hyp
            h_hyp = _poincare_math.mobius_add(z_hyp, b_hyp, c)
        else:
            h_hyp = z_hyp

        # Step 4: Project to ensure numerical stability
        # Prevents points from getting too close to boundary before log_map
        h_hyp = _poincare_math.project(h_hyp, c)

        # Step 5: Map back to tangent space (Euclidean)
        # H_tan = log₀^c(H_hyp): [N, units] -> [N, units]
        h_tangent = _poincare_math.log_map_0(h_hyp, c)

        # Step 6: Neighborhood aggregation using sparse matrix multiplication
        # Y = Ã @ H_tan: [N, N] (sparse) @ [N, units] -> [N, units]
        # This is the key simplification: aggregation in Euclidean space
        aggregated = tf.sparse.sparse_dense_matmul(adj, h_tangent)

        # Step 7: Apply Euclidean activation function
        # Out = σ(Y): [N, units] -> [N, units]
        output = self.activation(aggregated)

        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple, List[Tuple]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape.

        :param input_shape: List of ``[features_shape, adjacency_shape]``.
        :type input_shape: Union[Tuple, List[Tuple]]
        :return: Output shape tuple ``(num_nodes, units)``.
        :rtype: Tuple[Optional[int], ...]
        """
        feat_shape = input_shape[0]
        return (feat_shape[0], self.units)

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'use_curvature': self.use_curvature,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
        })
        return config

# ---------------------------------------------------------------------
