"""
Holonomic Transformer Layer.

This module provides a HolonomicTransformerLayer that integrates all the
field-based components into a complete transformer layer. This layer
provides the full benefits of holonomic processing:
- Gauge-invariant attention
- Parallel transport of information
- Curvature-based representations
- Manifold stress for anomaly detection

The HolonomicTransformerLayer serves as a drop-in replacement for standard
transformer layers while providing enhanced robustness and geometric awareness.

Key Features:
1. Field embeddings instead of point vectors
2. Connection-aware information flow
3. Holonomy-based global features
4. Built-in anomaly detection via manifold stress
5. O(n) complexity for sequence length (vs O(n²) for standard attention)
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Dict, Any, Tuple, Literal

from .field_embedding import FieldEmbedding
from .connection_layer import ConnectionLayer
from .parallel_transport import ParallelTransportLayer
from .holonomy_layer import HolonomyLayer
from .gauge_invariant_attention import GaugeInvariantAttention
from .manifold_stress import ManifoldStressLayer

NormalizationType = Literal['layer_norm', 'rms_norm', 'field_norm']


@keras.saving.register_keras_serializable(package='holonomic')
class FieldNormalization(keras.layers.Layer):
    """
    Field-aware normalization that respects curvature.

    Unlike standard layer normalization, this normalization accounts for
    the local curvature of the field, providing more geometrically
    meaningful normalization.

    Args:
        epsilon: Small constant for numerical stability.
        use_curvature_scaling: Whether to scale by local curvature.
        center: Whether to center the outputs.
        scale: Whether to scale the outputs.
    """

    def __init__(
            self,
            epsilon: float = 1e-6,
            use_curvature_scaling: bool = True,
            center: bool = True,
            scale: bool = True,
            **kwargs: Any
    ) -> None:
        """Initialize FieldNormalization."""
        super().__init__(**kwargs)

        self.epsilon = epsilon
        self.use_curvature_scaling = use_curvature_scaling
        self.center = center
        self.scale = scale

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the layer weights."""
        if isinstance(input_shape, list):
            feature_dim = input_shape[0][-1]
        else:
            feature_dim = input_shape[-1]

        if self.scale:
            self.gamma = self.add_weight(
                name='gamma',
                shape=(feature_dim,),
                initializer='ones',
                trainable=True
            )

        if self.center:
            self.beta = self.add_weight(
                name='beta',
                shape=(feature_dim,),
                initializer='zeros',
                trainable=True
            )

        if self.use_curvature_scaling:
            self.curvature_scale = self.add_weight(
                name='curvature_scale',
                shape=(1,),
                initializer=initializers.Constant(0.1),
                trainable=True
            )

        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply field-aware normalization.

        Args:
            inputs: Either embeddings tensor or tuple of (embeddings, curvature).
            training: Whether in training mode.

        Returns:
            Normalized embeddings.
        """
        if isinstance(inputs, (list, tuple)):
            embeddings, curvature = inputs
        else:
            embeddings = inputs
            curvature = None

        # Standard normalization
        mean = ops.mean(embeddings, axis=-1, keepdims=True)
        variance = ops.var(embeddings, axis=-1, keepdims=True)

        normalized = (embeddings - mean) / ops.sqrt(variance + self.epsilon)

        # Apply curvature-based scaling if available
        if self.use_curvature_scaling and curvature is not None:
            # Flatten curvature to match embedding dimensions
            curv_shape = ops.shape(curvature)
            if len(curv_shape) > 3:
                curvature = ops.reshape(
                    curvature,
                    (curv_shape[0], curv_shape[1], -1)
                )

            # Compute curvature magnitude
            curv_mag = ops.sqrt(ops.sum(curvature ** 2, axis=-1, keepdims=True) + self.epsilon)

            # Scale normalization by inverse curvature (high curvature → less aggressive norm)
            curvature_factor = 1.0 / (1.0 + self.curvature_scale * curv_mag)
            normalized = normalized * curvature_factor

        # Apply learned scale and shift
        if self.scale:
            normalized = normalized * self.gamma
        if self.center:
            normalized = normalized + self.beta

        return normalized

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'use_curvature_scaling': self.use_curvature_scaling,
            'center': self.center,
            'scale': self.scale,
        })
        return config


@keras.saving.register_keras_serializable(package='holonomic')
class HolonomicTransformerLayer(keras.layers.Layer):
    """
    Complete Holonomic Transformer Layer.

    This layer implements a full transformer block using holonomic principles:
    1. Input passes through connection computation
    2. Gauge-invariant attention processes the field
    3. Parallel transport ensures geometric consistency
    4. Holonomy features capture global structure
    5. FFN with field-aware normalization
    6. Manifold stress provides anomaly detection

    The layer provides several advantages over standard transformers:
    - Natural robustness to adversarial inputs
    - Geometric regularization through curvature constraints
    - Built-in anomaly detection
    - Richer representations through field structure

    Args:
        hidden_dim: Hidden dimension size.
        num_heads: Number of attention heads.
        ffn_dim: Dimension of feed-forward network. If None, uses 4 * hidden_dim.
        curvature_type: Type of curvature ('ricci', 'scalar', 'metric').
        connection_type: Type of connection ('yang_mills', 'levi_civita', 'affine').
        attention_metric: Metric for attention ('hybrid', 'holonomy', 'geodesic').
        use_holonomy_features: Whether to include holonomy features.
        use_anomaly_detection: Whether to compute manifold stress.
        dropout_rate: Dropout rate.
        normalization_type: Type of normalization.
        activation: Activation function for FFN.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Regularizer for kernel weights.

    Example:
        >>> layer = HolonomicTransformerLayer(
        ...     hidden_dim=256,
        ...     num_heads=8,
        ...     use_holonomy_features=True,
        ...     use_anomaly_detection=True
        ... )
        >>> x = keras.ops.random.normal((2, 16, 256))
        >>> output, anomaly_scores = layer(x)
        >>> print(output.shape)  # (2, 16, 256)
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            ffn_dim: Optional[int] = None,
            curvature_type: str = 'ricci',
            connection_type: str = 'yang_mills',
            attention_metric: str = 'hybrid',
            use_holonomy_features: bool = True,
            use_anomaly_detection: bool = True,
            dropout_rate: float = 0.1,
            normalization_type: NormalizationType = 'field_norm',
            activation: str = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the HolonomicTransformerLayer."""
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or (4 * hidden_dim)
        self.curvature_type = curvature_type
        self.connection_type = connection_type
        self.attention_metric = attention_metric
        self.use_holonomy_features = use_holonomy_features
        self.use_anomaly_detection = use_anomaly_detection
        self.dropout_rate = dropout_rate
        self.normalization_type = normalization_type
        self.activation = activation

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Create sub-layers in __init__
        self._create_sublayers()

    def _create_sublayers(self) -> None:
        """Create all sub-layers."""
        # Connection computation
        self.connection_layer = ConnectionLayer(
            hidden_dim=self.hidden_dim,
            connection_type=self.connection_type,
            num_generators=self.num_heads,
            name='connection'
        )

        # Curvature projection (maps input to curvature)
        self.curvature_dense = keras.layers.Dense(
            self.hidden_dim,
            activation='tanh',
            kernel_initializer=self.kernel_initializer,
            name='curvature_projection'
        )

        # Gauge-invariant attention
        self.attention = GaugeInvariantAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            attention_metric=self.attention_metric,
            use_curvature_gating=True,
            use_parallel_transport=True,
            dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='gauge_attention'
        )

        # Parallel transport
        self.transport = ParallelTransportLayer(
            transport_dim=self.hidden_dim,
            num_steps=5,
            transport_method='iterative',
            name='parallel_transport'
        )

        # Holonomy features
        if self.use_holonomy_features:
            self.holonomy = HolonomyLayer(
                hidden_dim=self.hidden_dim,
                loop_sizes=[2, 4, 8],
                loop_type='rectangular',
                use_trace=True,
                name='holonomy'
            )
            self.holonomy_projection = keras.layers.Dense(
                self.hidden_dim,
                kernel_initializer=self.kernel_initializer,
                name='holonomy_projection'
            )

        # Manifold stress for anomaly detection
        if self.use_anomaly_detection:
            self.stress_layer = ManifoldStressLayer(
                hidden_dim=self.hidden_dim,
                stress_types=['curvature', 'connection', 'combined'],
                use_learnable_baseline=True,
                name='manifold_stress'
            )

        # Normalization layers
        if self.normalization_type == 'field_norm':
            self.attn_norm = FieldNormalization(name='attn_norm')
            self.ffn_norm = FieldNormalization(name='ffn_norm')
        elif self.normalization_type == 'rms_norm':
            # Use standard RMS norm
            self.attn_norm = keras.layers.LayerNormalization(
                epsilon=1e-6, center=False, name='attn_norm'
            )
            self.ffn_norm = keras.layers.LayerNormalization(
                epsilon=1e-6, center=False, name='ffn_norm'
            )
        else:
            self.attn_norm = keras.layers.LayerNormalization(
                epsilon=1e-6, name='attn_norm'
            )
            self.ffn_norm = keras.layers.LayerNormalization(
                epsilon=1e-6, name='ffn_norm'
            )

        # Feed-forward network
        self.ffn_dense1 = keras.layers.Dense(
            self.ffn_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='ffn_dense1'
        )
        self.ffn_dense2 = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='ffn_dense2'
        )

        # Dropout
        self.dropout = keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        # Build all sub-layers
        self.connection_layer.build([input_shape, input_shape])
        self.curvature_dense.build(input_shape)

        # Attention expects tuple of shapes
        curv_shape = list(input_shape)
        curv_shape[-1] = self.hidden_dim
        conn_shape = list(input_shape)
        conn_shape.append(self.hidden_dim)

        self.attention.build([input_shape, tuple(curv_shape), tuple(conn_shape)])
        self.transport.build([input_shape, tuple(conn_shape)])

        if self.use_holonomy_features:
            self.holonomy.build([input_shape, tuple(conn_shape)])
            holonomy_out_shape = list(input_shape)
            holonomy_out_shape[-1] = self.hidden_dim
            self.holonomy_projection.build(tuple(holonomy_out_shape))

        if self.use_anomaly_detection:
            self.stress_layer.build([input_shape, tuple(curv_shape), tuple(conn_shape)])

        # Build normalization and FFN
        self.attn_norm.build(input_shape)
        self.ffn_norm.build(input_shape)
        self.ffn_dense1.build(input_shape)
        ffn_mid_shape = list(input_shape)
        ffn_mid_shape[-1] = self.ffn_dim
        self.ffn_dense2.build(tuple(ffn_mid_shape))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass through the holonomic transformer layer.

        Args:
            inputs: Input tensor of shape (batch, seq_len, hidden_dim).
            training: Whether in training mode.
            attention_mask: Optional attention mask.
            return_attention_weights: Whether to return attention weights.

        Returns:
            If use_anomaly_detection:
                Tuple of (output, anomaly_scores)
            Else:
                Output tensor
        """
        # Step 1: Compute curvature from input
        curvature = self.curvature_dense(inputs, training=training)
        curvature = curvature * 0.1  # Scale for stability

        # Step 2: Compute connection
        connection = self.connection_layer([inputs, curvature], training=training)

        # Step 3: Pre-normalization
        if self.normalization_type == 'field_norm':
            normed = self.attn_norm([inputs, curvature], training=training)
        else:
            normed = self.attn_norm(inputs, training=training)

        # Step 4: Gauge-invariant attention
        attn_output = self.attention(
            [normed, curvature, connection],
            training=training,
            attention_mask=attention_mask
        )

        # Step 5: Parallel transport for residual (ensures geometric consistency)
        transported_inputs = self.transport([inputs, connection], training=training)

        # Step 6: First residual connection
        x = transported_inputs + self.dropout(attn_output, training=training)

        # Step 7: Add holonomy features if enabled
        if self.use_holonomy_features:
            holonomy_features = self.holonomy([x, connection], training=training)
            holonomy_proj = self.holonomy_projection(holonomy_features, training=training)
            x = x + 0.1 * holonomy_proj  # Scaled addition

        # Step 8: FFN with pre-norm
        if self.normalization_type == 'field_norm':
            # Recompute curvature for updated representation
            curvature_ffn = self.curvature_dense(x, training=training) * 0.1
            normed_ffn = self.ffn_norm([x, curvature_ffn], training=training)
        else:
            normed_ffn = self.ffn_norm(x, training=training)

        ffn_output = self.ffn_dense1(normed_ffn, training=training)
        ffn_output = self.ffn_dense2(ffn_output, training=training)

        # Step 9: Second residual connection
        output = x + self.dropout(ffn_output, training=training)

        # Step 10: Anomaly detection if enabled
        if self.use_anomaly_detection:
            # Recompute curvature and connection for stress computation
            final_curvature = self.curvature_dense(output, training=training) * 0.1
            final_connection = self.connection_layer(
                [output, final_curvature], training=training
            )

            stress, anomaly_mask = self.stress_layer(
                [output, final_curvature, final_connection],
                training=training
            )

            return output, stress

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]:
        """
        Compute output shape.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Output shape(s).
        """
        if self.use_anomaly_detection:
            # Returns (output, stress)
            stress_shape = (input_shape[0], input_shape[1], 1)
            return input_shape, stress_shape

        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'ffn_dim': self.ffn_dim,
            'curvature_type': self.curvature_type,
            'connection_type': self.connection_type,
            'attention_metric': self.attention_metric,
            'use_holonomy_features': self.use_holonomy_features,
            'use_anomaly_detection': self.use_anomaly_detection,
            'dropout_rate': self.dropout_rate,
            'normalization_type': self.normalization_type,
            'activation': self.activation,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config