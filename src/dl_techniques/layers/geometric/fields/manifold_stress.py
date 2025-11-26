"""
Manifold Stress Layer

This module provides a ManifoldStressLayer that computes the "stress" or
inconsistency in the field representation. High stress indicates that the
input violates the expected geometric structure, which can be used to detect:
- Adversarial perturbations
- Poisoned training data
- Out-of-distribution inputs
- Semantically inconsistent content

Mathematical Foundation:
    In differential geometry, stress can be measured through various quantities:
    - Curvature deviation: ||R - R_expected||
    - Connection inconsistency: ||∇Γ||
    - Metric distortion: ||g - g_expected||
    - Torsion: antisymmetric part of connection

    These quantities measure how much the local geometry deviates from
    a "healthy" manifold structure.

In Holonomic AI:
    - Poisoned data increases manifold stress → automatically downweighted
    - Adversarial inputs have inconsistent geometry → detected and flagged
    - The stress provides a natural anomaly score based on geometric principles
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Dict, Any, Tuple, List, Literal

StressType = Literal['curvature', 'connection', 'holonomy', 'metric', 'combined']


@keras.saving.register_keras_serializable(package='holonomic')
class ManifoldStressLayer(keras.layers.Layer):
    """
    Computes manifold stress for anomaly and adversarial detection.

    This layer measures the geometric stress in the field representation,
    which indicates how well the input conforms to the expected manifold
    structure. High stress values indicate potential anomalies.

    Stress types:
    - 'curvature': Deviation of curvature from expected values
    - 'connection': Smoothness/consistency of the connection
    - 'holonomy': Deviation of holonomy from identity (flat space)
    - 'metric': Distortion of the induced metric
    - 'combined': Weighted combination of all stress types

    The stress can be used to:
    1. Flag potentially adversarial inputs
    2. Downweight anomalous data during training
    3. Provide confidence scores for predictions

    Args:
        hidden_dim: Hidden dimension size.
        stress_types: List of stress types to compute.
        stress_threshold: Threshold for anomaly flagging.
        use_learnable_baseline: Whether to learn baseline (expected) values.
        return_components: Whether to return individual stress components.
        kernel_initializer: Initializer for kernel weights.

    Example:
        >>> stress_layer = ManifoldStressLayer(
        ...     hidden_dim=256,
        ...     stress_types=['curvature', 'connection'],
        ...     stress_threshold=0.5
        ... )
        >>> embeddings = keras.ops.random.normal((2, 16, 256))
        >>> curvature = keras.ops.random.normal((2, 16, 256))
        >>> connection = keras.ops.random.normal((2, 16, 256, 256)) * 0.01
        >>> stress, anomaly_mask = stress_layer([embeddings, curvature, connection])
        >>> print(stress.shape)  # (2, 16, 1)
    """

    def __init__(
            self,
            hidden_dim: int,
            stress_types: List[str] = ['curvature', 'connection', 'combined'],
            stress_threshold: float = 0.5,
            use_learnable_baseline: bool = True,
            return_components: bool = False,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            **kwargs: Any
    ) -> None:
        """Initialize the ManifoldStressLayer."""
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        valid_stress_types = {'curvature', 'connection', 'holonomy', 'metric', 'combined'}
        for st in stress_types:
            if st not in valid_stress_types:
                raise ValueError(
                    f"Invalid stress type '{st}'. Must be one of {valid_stress_types}"
                )

        self.hidden_dim = hidden_dim
        self.stress_types = list(stress_types)
        self.stress_threshold = stress_threshold
        self.use_learnable_baseline = use_learnable_baseline
        self.return_components = return_components

        self.kernel_initializer = initializers.get(kernel_initializer)

        # Number of stress components
        self.num_components = len(stress_types)

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        """
        Build the layer weights.

        Args:
            input_shape: Tuple of shapes for inputs.
        """
        # Learnable stress weights for combining components
        self.stress_weights = self.add_weight(
            name='stress_weights',
            shape=(self.num_components,),
            initializer=initializers.Constant(1.0 / self.num_components),
            trainable=True
        )

        # Learnable baselines for expected values
        if self.use_learnable_baseline:
            # Expected curvature (typically near zero for "healthy" data)
            self.baseline_curvature = self.add_weight(
                name='baseline_curvature',
                shape=(self.hidden_dim,),
                initializer='zeros',
                trainable=True
            )

            # Expected connection smoothness
            self.baseline_connection = self.add_weight(
                name='baseline_connection',
                shape=(1,),
                initializer='zeros',
                trainable=True
            )

            # Expected holonomy (identity → trace = dim)
            self.baseline_holonomy = self.add_weight(
                name='baseline_holonomy',
                shape=(1,),
                initializer=initializers.Constant(float(self.hidden_dim)),
                trainable=True
            )

        # Projection for combining stress features
        self.stress_projection = self.add_weight(
            name='stress_projection',
            shape=(self.num_components, 1),
            initializer=self.kernel_initializer,
            trainable=True
        )

        # Adaptive threshold
        self.adaptive_threshold = self.add_weight(
            name='adaptive_threshold',
            shape=(1,),
            initializer=initializers.Constant(self.stress_threshold),
            trainable=True
        )

        super().build(input_shape)

    def _compute_curvature_stress(
            self,
            curvature: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute stress from curvature deviation.

        High curvature or curvature inconsistent with the baseline
        indicates potential anomalies.

        Args:
            curvature: Curvature tensor.

        Returns:
            Curvature stress, shape (batch, seq_len, 1).
        """
        # Flatten curvature if needed
        curv_shape = ops.shape(curvature)
        batch_size = curv_shape[0]
        seq_len = curv_shape[1]

        if len(curv_shape) > 3:
            curvature = ops.reshape(curvature, (batch_size, seq_len, -1))

        # Match dimensions with baseline
        curv_dim = ops.shape(curvature)[-1]
        if self.use_learnable_baseline:
            # Adjust baseline dimension if needed
            if curv_dim <= self.hidden_dim:
                baseline = self.baseline_curvature[:curv_dim]
            else:
                # Tile baseline
                reps = (curv_dim // self.hidden_dim) + 1
                baseline = ops.tile(self.baseline_curvature, (reps,))[:curv_dim]

            # Deviation from baseline
            deviation = curvature - baseline
        else:
            # Default baseline is zero
            deviation = curvature

        # L2 norm of deviation
        stress = ops.sqrt(ops.sum(deviation ** 2, axis=-1, keepdims=True) + 1e-8)

        return stress

    def _compute_connection_stress(
            self,
            connection: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute stress from connection inconsistency.

        A smooth, consistent connection has low variation between
        adjacent positions. High variation indicates stress.

        Args:
            connection: Connection tensor, shape (batch, seq_len, dim, dim).

        Returns:
            Connection stress, shape (batch, seq_len, 1).
        """
        batch_size = ops.shape(connection)[0]
        seq_len = ops.shape(connection)[1]

        # Flatten connection for easier computation
        conn_flat = ops.reshape(connection, (batch_size, seq_len, -1))

        # Compute local variation (gradient-like)
        # Forward difference
        conn_shift = ops.concatenate([
            conn_flat[:, 1:, :],
            conn_flat[:, -1:, :]
        ], axis=1)
        variation = conn_shift - conn_flat

        # L2 norm of variation
        local_stress = ops.sqrt(ops.sum(variation ** 2, axis=-1, keepdims=True) + 1e-8)

        # Also penalize large connection values (should be near zero for flat space)
        magnitude_stress = ops.sqrt(ops.sum(conn_flat ** 2, axis=-1, keepdims=True) + 1e-8)

        # Combine
        if self.use_learnable_baseline:
            stress = local_stress + 0.1 * (magnitude_stress - ops.abs(self.baseline_connection))
        else:
            stress = local_stress + 0.1 * magnitude_stress

        return stress

    def _compute_holonomy_stress(
            self,
            connection: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute stress from holonomy deviation.

        For a flat space (no curvature), holonomy should be the identity.
        Deviation from identity indicates curvature and potential anomalies.

        Args:
            connection: Connection tensor.

        Returns:
            Holonomy stress, shape (batch, seq_len, 1).
        """
        batch_size = ops.shape(connection)[0]
        seq_len = ops.shape(connection)[1]
        conn_dim = ops.shape(connection)[2]

        # Compute approximate holonomy around small loops
        # Using small rectangular loops at each position

        stress_list = []

        for pos in range(seq_len):
            # Get connections at pos and pos+1 (if available)
            gamma_here = connection[:, pos, :, :]

            if pos < seq_len - 1:
                gamma_next = connection[:, pos + 1, :, :]
            else:
                gamma_next = gamma_here

            # Approximate holonomy: [Γ_here, Γ_next] (commutator)
            # For flat space, this should be zero
            commutator = ops.matmul(gamma_here, gamma_next) - ops.matmul(gamma_next, gamma_here)

            # Frobenius norm of commutator
            holonomy_deviation = ops.sqrt(ops.sum(commutator ** 2, axis=(-2, -1)) + 1e-8)
            stress_list.append(holonomy_deviation)

        # Stack: (batch, seq_len)
        stress = ops.stack(stress_list, axis=1)
        stress = ops.expand_dims(stress, -1)

        # Compare to expected holonomy
        if self.use_learnable_baseline:
            # Baseline holonomy trace deviation
            expected_deviation = ops.abs(stress - self.baseline_holonomy / conn_dim)
            stress = expected_deviation

        return stress

    def _compute_metric_stress(
            self,
            embeddings: keras.KerasTensor,
            curvature: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Compute stress from metric distortion.

        The induced metric from embeddings should be smooth and consistent.
        Distortions indicate geometric anomalies.

        Args:
            embeddings: Embedding tensor.
            curvature: Optional curvature tensor.

        Returns:
            Metric stress, shape (batch, seq_len, 1).
        """
        batch_size = ops.shape(embeddings)[0]
        seq_len = ops.shape(embeddings)[1]

        # Compute local metric from embeddings
        # g_ij ∝ <∂_i e, ∂_j e> where e is embedding

        # Approximate partial derivatives using finite differences
        embed_shift = ops.concatenate([
            embeddings[:, 1:, :],
            embeddings[:, -1:, :]
        ], axis=1)
        local_derivative = embed_shift - embeddings

        # Local metric tensor: (batch, seq_len, dim, dim)
        # Approximated by outer product
        metric_diag = ops.sum(local_derivative ** 2, axis=-1)  # (batch, seq_len)

        # Metric stress: variation in local metric
        metric_shift = ops.concatenate([
            metric_diag[:, 1:],
            metric_diag[:, -1:]
        ], axis=1)
        metric_variation = ops.abs(metric_shift - metric_diag)

        stress = ops.expand_dims(metric_variation, -1)

        return stress

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, ...],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Compute manifold stress and anomaly mask.

        Args:
            inputs: Tuple of (embeddings, curvature, connection).
                - embeddings: shape (batch, seq_len, dim)
                - curvature: shape (batch, seq_len, ...) - optional
                - connection: shape (batch, seq_len, dim, dim) - optional
            training: Whether in training mode.

        Returns:
            Tuple of:
                - stress: Total stress tensor, shape (batch, seq_len, 1) or
                  (batch, seq_len, num_components) if return_components=True
                - anomaly_mask: Boolean mask, True where stress > threshold
        """
        # Parse inputs
        if isinstance(inputs, (list, tuple)):
            embeddings = inputs[0]
            curvature = inputs[1] if len(inputs) > 1 else None
            connection = inputs[2] if len(inputs) > 2 else None
        else:
            embeddings = inputs
            curvature = None
            connection = None

        batch_size = ops.shape(embeddings)[0]
        seq_len = ops.shape(embeddings)[1]

        # Compute individual stress components
        stress_components = []

        for stress_type in self.stress_types:
            if stress_type == 'curvature' and curvature is not None:
                stress = self._compute_curvature_stress(curvature)
            elif stress_type == 'connection' and connection is not None:
                stress = self._compute_connection_stress(connection)
            elif stress_type == 'holonomy' and connection is not None:
                stress = self._compute_holonomy_stress(connection)
            elif stress_type == 'metric':
                stress = self._compute_metric_stress(embeddings, curvature)
            elif stress_type == 'combined':
                # Combine available stresses
                sub_stresses = []
                if curvature is not None:
                    sub_stresses.append(self._compute_curvature_stress(curvature))
                if connection is not None:
                    sub_stresses.append(self._compute_connection_stress(connection))
                sub_stresses.append(self._compute_metric_stress(embeddings, curvature))

                if sub_stresses:
                    stress = ops.mean(ops.concatenate(sub_stresses, axis=-1), axis=-1, keepdims=True)
                else:
                    stress = ops.zeros((batch_size, seq_len, 1))
            else:
                # Fallback: zero stress
                stress = ops.zeros((batch_size, seq_len, 1))

            stress_components.append(stress)

        # Stack components: (batch, seq_len, num_components)
        all_stresses = ops.concatenate(stress_components, axis=-1)

        # Combine with learned weights
        # Ensure weights are positive
        weights = ops.softmax(self.stress_weights)
        total_stress = ops.sum(all_stresses * weights, axis=-1, keepdims=True)

        # Compute anomaly mask using adaptive threshold
        threshold = ops.abs(self.adaptive_threshold)
        anomaly_mask = total_stress > threshold

        # Return results
        if self.return_components:
            return all_stresses, anomaly_mask
        else:
            return total_stress, anomaly_mask

    def compute_output_shape(
            self,
            input_shape: Tuple[Tuple[int, ...], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """
        Compute output shapes.

        Args:
            input_shape: Tuple of input shapes.

        Returns:
            Tuple of (stress_shape, mask_shape).
        """
        if isinstance(input_shape, list):
            embed_shape = input_shape[0]
        else:
            embed_shape = input_shape

        batch_size = embed_shape[0]
        seq_len = embed_shape[1] if len(embed_shape) > 1 else None

        if self.return_components:
            stress_shape = (batch_size, seq_len, self.num_components)
        else:
            stress_shape = (batch_size, seq_len, 1)

        mask_shape = (batch_size, seq_len, 1)

        return stress_shape, mask_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'stress_types': self.stress_types,
            'stress_threshold': self.stress_threshold,
            'use_learnable_baseline': self.use_learnable_baseline,
            'return_components': self.return_components,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        })
        return config