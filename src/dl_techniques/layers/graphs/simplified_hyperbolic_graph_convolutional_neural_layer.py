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
    """
    Simplified Hyperbolic Graph Convolutional Layer.

    This layer implements efficient hyperbolic graph convolution by performing
    neighbor aggregation in Euclidean tangent space while using hyperbolic geometry
    specifically for bias addition, capturing hierarchical structure.

    **Key Innovation**: Unlike standard HGCN which performs expensive aggregation
    in hyperbolic space (Fréchet means), sHGCN performs aggregation in Euclidean
    space after geometric bias transformation, achieving better speed and stability.

    **Architecture Flow**:
    ```
    Input Features [N, D_in]
            ↓
    1. Euclidean Transform: Z = X @ W
            ↓
    2. Map to Hyperbolic: Z_hyp = exp₀^c(Z)
            ↓
    3. Möbius Bias: H_hyp = Z_hyp ⊕_c b_hyp
            ↓
    4. Project (stability): H_hyp = project(H_hyp)
            ↓
    5. Map to Tangent: H_tan = log₀^c(H_hyp)
            ↓
    6. Aggregate Neighbors: Y = Ã @ H_tan
            ↓
    7. Activation: Output = σ(Y)
            ↓
    Output [N, D_out]
    ```

    **Mathematical Operations**:
        1. Linear: Z = XW where Z ∈ ℝ^{N×units}
        2. Exponential: Z_hyp = exp₀^c(Z) maps to 𝔻^n_c
        3. Bias: H_hyp = Z_hyp ⊕_c exp₀^c(b)
        4. Logarithmic: H_tan = log₀^c(H_hyp) maps back to tangent space
        5. Aggregation: Y = ÃH_tan leverages sparse matrix operations
        6. Activation: Out = σ(Y) applies element-wise nonlinearity

    Args:
        units: Output dimensionality (number of features per node). Must be positive.
        activation: Euclidean activation function name or callable. Applied after
            aggregation in tangent space. Defaults to 'relu'.
        use_bias: Whether to include learnable hyperbolic bias. When True, adds
            Möbius translation in hyperbolic space. Defaults to True.
        use_curvature: Whether curvature c is learnable. When True, c is optimized
            during training. When False, fixed at c=1. Defaults to True.
        dropout_rate: Dropout probability applied to input features before
            transformation. Range [0, 1]. Defaults to 0.0 (no dropout).
        kernel_initializer: Initializer for weight matrix W. Defaults to
            'glorot_uniform'.
        bias_initializer: Initializer for bias vector b. Only used when
            use_bias=True. Defaults to 'zeros'.
        **kwargs: Additional keyword arguments for Layer base class.

    Input:
        List of two tensors:
        - features: Dense tensor of shape (num_nodes, input_dim) with node features
        - adjacency: Sparse tensor of shape (num_nodes, num_nodes) representing
            normalized adjacency matrix Ã. Must be a tf.sparse.SparseTensor.

    Output:
        Dense tensor of shape (num_nodes, units) with updated node embeddings
        in Euclidean tangent space.

    Attributes:
        kernel: Weight matrix of shape (input_dim, units).
        bias: Bias vector of shape (units,) if use_bias=True.
        c_theta: Curvature parameter θ where c = softplus(θ).
        dropout: Dropout layer instance.

    Example:
        ```python
        # Single sHGCN layer
        layer = SHGCNLayer(units=64, dropout_rate=0.3)

        # Prepare inputs
        features = ops.random.normal((100, 32))  # 100 nodes, 32 features
        adj_sparse = create_normalized_adjacency(...)  # Sparse tensor

        # Forward pass
        output = layer([features, adj_sparse], training=True)
        print(output.shape)  # (100, 64)

        # Multi-layer graph network
        x = features
        for units in [64, 32, 16]:
            layer = SHGCNLayer(units=units, activation='relu', dropout_rate=0.5)
            x = layer([x, adj_sparse], training=True)
        ```

    Note:
        - Adjacency matrix should be pre-normalized (e.g., D^{-1/2}AD^{-1/2})
        - Sparse adjacency is required for scalability on large graphs
        - Output embeddings are in Euclidean space, not hyperbolic
        - Curvature c is constrained > 0 via softplus transformation

    References:
        Arevalo et al. "Simplified Hyperbolic Graph Convolutional Neural Networks"
        Equation 14 (main forward pass)
        Equation 17 (Möbius addition)
        Equation 19 (exponential/logarithmic maps)
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
        """Initialize sHGCN layer with configuration."""
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
        """
        Create layer weights based on input shape.

        Args:
            input_shape: List of two shapes:
                - features_shape: (num_nodes, input_dim)
                - adjacency_shape: (num_nodes, num_nodes)
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
        """
        Get current curvature value c > 0.

        Returns:
            Scalar tensor representing curvature c = softplus(c_theta).
        """
        return keras.ops.softplus(self.c_theta)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass implementing Equation 14.

        Args:
            inputs: List containing:
                - features: [num_nodes, input_dim] dense tensor
                - adjacency: [num_nodes, num_nodes] sparse tensor (normalized)
            training: Whether in training mode (affects dropout).

        Returns:
            Updated node embeddings of shape [num_nodes, units] in Euclidean space.
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
        """
        Compute output shape given input shape.

        Args:
            input_shape: List of [features_shape, adjacency_shape].

        Returns:
            Output shape tuple (num_nodes, units).
        """
        feat_shape = input_shape[0]
        return (feat_shape[0], self.units)

    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all constructor arguments.
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
