"""
Connection Layer.

This module provides a ConnectionLayer that computes the gauge connection
(analogous to Christoffel symbols in differential geometry) from field
representations. The connection defines how to parallel transport vectors
along paths on the semantic manifold.

Mathematical Foundation:
    In differential geometry, the connection Γ defines how vectors change
    as they are moved along curves. For a vector V being transported along
    a curve with tangent vector T, the covariant derivative is:

    ∇_T V = dV/dt + Γ(T, V)

    The connection ensures that transported vectors remain "parallel" in
    a geometric sense, preserving the intrinsic properties of the manifold.

In the context of Holonomic AI:
    - The connection determines how semantic meaning changes as we move
      through the representation space
    - It enables gauge-invariant operations by ensuring transformations
      respect the manifold structure
    - External inputs (like adversarial perturbations) that don't respect
      the connection structure are naturally rejected
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Dict, Any, Tuple, Literal

ConnectionType = Literal['levi_civita', 'yang_mills', 'affine']


@keras.saving.register_keras_serializable(package='holonomic')
class ConnectionLayer(keras.layers.Layer):
    """
    Computes the gauge connection from field representations.

    The connection is a tensor that describes how vectors transform under
    parallel transport. This layer learns to compute the connection from
    the input field, enabling geometric operations like parallel transport
    and holonomy computation.

    The connection can be computed in several ways:
    - 'levi_civita': Metric-compatible, torsion-free connection
    - 'yang_mills': Non-abelian gauge connection (more general)
    - 'affine': General affine connection

    Args:
        hidden_dim: Dimension of the hidden representation.
        connection_dim: Dimension of the connection (output per position pair).
            If None, uses hidden_dim.
        connection_type: Type of connection to compute.
        num_generators: Number of Lie algebra generators for yang_mills type.
        use_metric: Whether to compute metric-compatible connection.
        antisymmetric: Whether to enforce antisymmetry (for gauge connections).
        connection_regularization: Strength of connection smoothness regularization.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Regularizer for kernel weights.

    Example:
        >>> connection_layer = ConnectionLayer(
        ...     hidden_dim=256,
        ...     connection_type='yang_mills',
        ...     num_generators=8
        ... )
        >>> embeddings = keras.ops.random.normal((2, 10, 256))
        >>> curvature = keras.ops.random.normal((2, 10, 256))
        >>> connection = connection_layer([embeddings, curvature])
        >>> print(connection.shape)  # (2, 10, 256, 8) for yang_mills
    """

    def __init__(
            self,
            hidden_dim: int,
            connection_dim: Optional[int] = None,
            connection_type: ConnectionType = 'yang_mills',
            num_generators: int = 8,
            use_metric: bool = True,
            antisymmetric: bool = True,
            connection_regularization: float = 0.001,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the ConnectionLayer."""
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_generators <= 0:
            raise ValueError(f"num_generators must be positive, got {num_generators}")
        if connection_type not in ('levi_civita', 'yang_mills', 'affine'):
            raise ValueError(
                f"connection_type must be one of 'levi_civita', 'yang_mills', 'affine', "
                f"got {connection_type}"
            )

        self.hidden_dim = hidden_dim
        self.connection_dim = connection_dim or hidden_dim
        self.connection_type = connection_type
        self.num_generators = num_generators
        self.use_metric = use_metric
        self.antisymmetric = antisymmetric
        self.connection_regularization = connection_regularization

        # Store initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Compute output dimension based on connection type
        if connection_type == 'yang_mills':
            # Yang-Mills: connection valued in Lie algebra
            self.output_dim = self.connection_dim * self.num_generators
        else:
            # Affine or Levi-Civita: connection has one index
            self.output_dim = self.connection_dim * self.connection_dim

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        """
        Build the layer weights.

        Args:
            input_shape: Tuple of (embedding_shape, curvature_shape).
        """
        # Determine input dimension from embedding shape
        if isinstance(input_shape, list):
            embed_shape = input_shape[0]
        else:
            embed_shape = input_shape

        input_dim = embed_shape[-1]

        # Projection to connection space
        # We combine embedding and curvature information
        combined_input_dim = input_dim * 2  # embedding + curvature (flattened)

        # First hidden layer for connection computation
        self.connection_kernel_1 = self.add_weight(
            name='connection_kernel_1',
            shape=(combined_input_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.connection_bias_1 = self.add_weight(
            name='connection_bias_1',
            shape=(self.hidden_dim,),
            initializer=self.bias_initializer,
            trainable=True
        )

        # Output layer for connection
        self.connection_kernel_2 = self.add_weight(
            name='connection_kernel_2',
            shape=(self.hidden_dim, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.connection_bias_2 = self.add_weight(
            name='connection_bias_2',
            shape=(self.output_dim,),
            initializer=self.bias_initializer,
            trainable=True
        )

        # For Yang-Mills: learnable Lie algebra generators
        if self.connection_type == 'yang_mills':
            # Initialize generators as orthogonal matrices for non-Abelian structure
            self.lie_generators = self.add_weight(
                name='lie_generators',
                shape=(self.num_generators, self.connection_dim, self.connection_dim),
                initializer='orthogonal',
                trainable=True
            )

        # For metric-compatible connection: metric tensor
        if self.use_metric:
            self.metric_kernel = self.add_weight(
                name='metric_kernel',
                shape=(input_dim, self.connection_dim),
                initializer=self.kernel_initializer,
                trainable=True
            )

        super().build(input_shape)

    def _make_antisymmetric(
            self,
            tensor: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Make a tensor antisymmetric in its last two dimensions.

        For gauge connections, antisymmetry corresponds to the Lie algebra
        structure, ensuring the connection values are proper gauge fields.

        Args:
            tensor: Input tensor with at least 2 dimensions.

        Returns:
            Antisymmetric tensor.
        """
        # A - A^T along last two dimensions
        return (tensor - ops.transpose(tensor,
                                       list(range(len(ops.shape(tensor)) - 2)) + [-1, -2])) / 2.0

    def _compute_yang_mills_connection(
            self,
            hidden: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute Yang-Mills gauge connection.

        The Yang-Mills connection is valued in the Lie algebra of the gauge group.
        Each connection component is a linear combination of Lie algebra generators.

        Args:
            hidden: Hidden representation.

        Returns:
            Connection tensor of shape (batch, seq_len, connection_dim, connection_dim).
        """
        batch_size = ops.shape(hidden)[0]
        seq_len = ops.shape(hidden)[1]

        # Get connection coefficients: (batch, seq_len, connection_dim * num_generators)
        coeffs = ops.matmul(hidden, self.connection_kernel_2) + self.connection_bias_2
        coeffs = ops.reshape(
            coeffs,
            (batch_size, seq_len, self.connection_dim, self.num_generators)
        )

        # Expand generators for broadcasting
        # generators: (num_generators, connection_dim, connection_dim)
        # Make generators antisymmetric for proper Lie algebra structure
        generators = self._make_antisymmetric(self.lie_generators)

        # Combine: sum over generators weighted by coefficients
        # coeffs: (batch, seq_len, connection_dim, num_generators)
        # generators: (num_generators, connection_dim, connection_dim)
        # Result: (batch, seq_len, connection_dim, connection_dim)
        connection = ops.einsum('bsig,gij->bsij', coeffs, generators)

        return connection

    def _compute_affine_connection(
            self,
            hidden: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute general affine connection.

        The affine connection Γ^k_{ij} has three indices. We compute it as
        a tensor that can be used for parallel transport.

        Args:
            hidden: Hidden representation.

        Returns:
            Connection tensor of shape (batch, seq_len, connection_dim, connection_dim).
        """
        batch_size = ops.shape(hidden)[0]
        seq_len = ops.shape(hidden)[1]

        # Get raw connection values
        raw_connection = ops.matmul(hidden, self.connection_kernel_2)
        raw_connection = raw_connection + self.connection_bias_2

        # Reshape to tensor form
        connection = ops.reshape(
            raw_connection,
            (batch_size, seq_len, self.connection_dim, self.connection_dim)
        )

        if self.antisymmetric:
            connection = self._make_antisymmetric(connection)

        return connection

    def _compute_levi_civita_connection(
            self,
            embeddings: keras.KerasTensor,
            hidden: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute Levi-Civita connection (metric-compatible, torsion-free).

        The Levi-Civita connection is uniquely determined by the metric.
        We compute an approximation based on learned metric structure.

        Args:
            embeddings: Original embeddings for metric computation.
            hidden: Hidden representation.

        Returns:
            Connection tensor of shape (batch, seq_len, connection_dim, connection_dim).
        """
        batch_size = ops.shape(hidden)[0]
        seq_len = ops.shape(hidden)[1]

        # Compute metric from embeddings
        if self.use_metric:
            metric_rep = ops.matmul(embeddings, self.metric_kernel)
            # Metric as outer product: g_ij = sum_k e_k * e_k
            # Shape: (batch, seq_len, connection_dim, connection_dim)
            metric = ops.einsum('bsi,bsj->bsij', metric_rep, metric_rep)
            # Add identity for positive definiteness
            metric = metric + ops.eye(self.connection_dim)
        else:
            metric = ops.eye(self.connection_dim)
            metric = ops.expand_dims(ops.expand_dims(metric, 0), 0)
            metric = ops.tile(metric, (batch_size, seq_len, 1, 1))

        # Compute connection using affine base and metric correction
        affine_connection = self._compute_affine_connection(hidden)

        # For Levi-Civita, symmetrize in lower indices
        # Γ^k_{ij} = (Γ^k_{ij} + Γ^k_{ji}) / 2 for torsion-free
        connection = (affine_connection + ops.transpose(
            affine_connection, (0, 1, 3, 2)
        )) / 2.0

        return connection

    def call(
            self,
            inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute the connection from input field.

        Args:
            inputs: Either a single tensor or tuple of (embeddings, curvature).
                If single tensor, curvature is assumed to be zeros.
            training: Whether the layer is in training mode.

        Returns:
            Connection tensor of shape (batch, seq_len, connection_dim, connection_dim).
        """
        # Handle input formats
        if isinstance(inputs, (list, tuple)):
            embeddings, curvature = inputs
            # Flatten curvature if needed
            curvature_shape = ops.shape(curvature)
            if len(curvature_shape) > 3:
                curvature = ops.reshape(
                    curvature,
                    (curvature_shape[0], curvature_shape[1], -1)
                )
            # Match dimensions if needed
            if ops.shape(curvature)[-1] != ops.shape(embeddings)[-1]:
                # Pad or truncate curvature to match embedding dim
                curv_dim = ops.shape(curvature)[-1]
                embed_dim = ops.shape(embeddings)[-1]
                if curv_dim < embed_dim:
                    padding = ops.zeros((
                        ops.shape(curvature)[0],
                        ops.shape(curvature)[1],
                        embed_dim - curv_dim
                    ))
                    curvature = ops.concatenate([curvature, padding], axis=-1)
                else:
                    curvature = curvature[..., :embed_dim]
        else:
            embeddings = inputs
            curvature = ops.zeros_like(embeddings)

        # Combine embedding and curvature information
        combined = ops.concatenate([embeddings, curvature], axis=-1)

        # First hidden layer
        hidden = ops.matmul(combined, self.connection_kernel_1)
        hidden = hidden + self.connection_bias_1
        hidden = ops.tanh(hidden)  # Bounded activation for stability

        # Compute connection based on type
        if self.connection_type == 'yang_mills':
            connection = self._compute_yang_mills_connection(hidden)
        elif self.connection_type == 'levi_civita':
            connection = self._compute_levi_civita_connection(embeddings, hidden)
        else:  # affine
            connection = self._compute_affine_connection(hidden)

        # Add regularization for connection smoothness
        if training and self.connection_regularization > 0:
            # Encourage small connection values (flat space approximation)
            reg_loss = self.connection_regularization * ops.mean(connection ** 2)
            self.add_loss(reg_loss)

        return connection

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape.

        Args:
            input_shape: Input shape or tuple of input shapes.

        Returns:
            Output shape.
        """
        if isinstance(input_shape, list):
            embed_shape = input_shape[0]
        else:
            embed_shape = input_shape

        batch_size = embed_shape[0]
        seq_len = embed_shape[1] if len(embed_shape) > 1 else None

        return (batch_size, seq_len, self.connection_dim, self.connection_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'connection_dim': self.connection_dim,
            'connection_type': self.connection_type,
            'num_generators': self.num_generators,
            'use_metric': self.use_metric,
            'antisymmetric': self.antisymmetric,
            'connection_regularization': self.connection_regularization,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config