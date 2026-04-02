"""
Field Embedding Layer.

This module provides a FieldEmbedding layer that represents tokens not as point vectors
but as fields with associated curvature and connection information. This enables
geometric reasoning about semantic relationships and provides natural resistance
to adversarial perturbations.

The key insight is that traditional embeddings map tokens to points in a vector space,
while field embeddings map tokens to local sections of a fiber bundle with associated
curvature tensors. This allows the model to capture not just position but also how
meaning varies locally in the representation space.

Mathematical Foundation:
    - Standard embedding: token -> R^d (point vector)
    - Field embedding: token -> (R^d, C^{d x d}) where C is local curvature

The curvature tensor captures how the semantic "landscape" curves around each token,
providing richer geometric structure for downstream processing.
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Dict, Any, Tuple, Literal

CurvatureType = Literal['metric', 'riemann', 'ricci', 'scalar']


@keras.saving.register_keras_serializable(package='holonomic')
class FieldEmbedding(keras.layers.Layer):
    """Field Embedding layer that maps tokens to fields with curvature.

    Instead of mapping tokens to point vectors in R^d, maps them to
    (R^d, C) pairs where C is a local curvature tensor capturing how
    semantic meaning varies in the neighbourhood of each token. The
    curvature is computed as tanh(embedding @ W_curv + b) * scale, with
    shape depending on curvature_type: metric and riemann yield (B, S, D, D),
    ricci yields (B, S, D), and scalar yields (B, S, 1). Curvature
    smoothness regularisation encourages adjacent tokens to have similar
    local geometry.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────┐
        │ Token IDs  [B, S]       │
        └────────────┬────────────┘
                     ▼
        ┌─────────────────────────┐
        │ Embedding Lookup        │
        │ → embeddings [B, S, D]  │
        └────────┬────────────────┘
                 ├──────────────────────┐
                 ▼                      ▼
        ┌────────────────┐   ┌──────────────────────┐
        │ Output:        │   │ Curvature Projection │
        │ embeddings     │   │ emb @ W + b → tanh   │
        │ [B, S, D]      │   │ × scale              │
        └────────────────┘   │ → curvature          │
                             └──────────────────────┘

    :param vocab_size: Size of the vocabulary. Must be positive.
    :type vocab_size: int
    :param embed_dim: Dimension of embedding vectors. Must be positive.
    :type embed_dim: int
    :param curvature_dim: Dimension for curvature. Defaults to ``embed_dim``.
    :type curvature_dim: Optional[int]
    :param curvature_type: Type of curvature
        (``'metric'``, ``'riemann'``, ``'ricci'``, ``'scalar'``).
        Defaults to ``'ricci'``.
    :type curvature_type: CurvatureType
    :param curvature_scale: Initial scale for curvature values. Defaults to 0.1.
    :type curvature_scale: float
    :param curvature_regularization: Smoothness regularization strength.
        Defaults to 0.01.
    :type curvature_regularization: float
    :param embed_initializer: Initializer for embedding weights.
    :type embed_initializer: Union[str, initializers.Initializer]
    :param curvature_initializer: Initializer for curvature projection.
    :type curvature_initializer: Union[str, initializers.Initializer]
    :param embed_regularizer: Regularizer for embedding weights.
    :type embed_regularizer: Optional[regularizers.Regularizer]
    :param curvature_regularizer: Regularizer for curvature weights.
    :type curvature_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for the ``Layer`` base class.
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            curvature_dim: Optional[int] = None,
            curvature_type: CurvatureType = 'ricci',
            curvature_scale: float = 0.1,
            curvature_regularization: float = 0.01,
            embed_initializer: Union[str, initializers.Initializer] = 'uniform',
            curvature_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            embed_regularizer: Optional[regularizers.Regularizer] = None,
            curvature_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the FieldEmbedding layer."""
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if curvature_type not in ('metric', 'riemann', 'ricci', 'scalar'):
            raise ValueError(
                f"curvature_type must be one of 'metric', 'riemann', 'ricci', 'scalar', "
                f"got {curvature_type}"
            )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.curvature_dim = curvature_dim or embed_dim
        self.curvature_type = curvature_type
        self.curvature_scale = curvature_scale
        self.curvature_regularization = curvature_regularization

        # Store initializers
        self.embed_initializer = initializers.get(embed_initializer)
        self.curvature_initializer = initializers.get(curvature_initializer)
        self.embed_regularizer = regularizers.get(embed_regularizer)
        self.curvature_regularizer = regularizers.get(curvature_regularizer)

        # Compute curvature output dimension based on type
        if curvature_type == 'metric':
            self.curvature_output_dim = self.curvature_dim * self.curvature_dim
        elif curvature_type == 'riemann':
            self.curvature_output_dim = self.curvature_dim * self.curvature_dim
        elif curvature_type == 'ricci':
            self.curvature_output_dim = self.curvature_dim
        else:  # scalar
            self.curvature_output_dim = 1

        # Weights will be created in build()
        self.embedding_weights = None
        self.curvature_projection = None
        self.curvature_bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Main embedding weights
        self.embedding_weights = self.add_weight(
            name='embedding_weights',
            shape=(self.vocab_size, self.embed_dim),
            initializer=self.embed_initializer,
            regularizer=self.embed_regularizer,
            trainable=True
        )

        # Curvature projection: maps embedding to curvature representation
        self.curvature_projection = self.add_weight(
            name='curvature_projection',
            shape=(self.embed_dim, self.curvature_output_dim),
            initializer=self.curvature_initializer,
            regularizer=self.curvature_regularizer,
            trainable=True
        )

        self.curvature_bias = self.add_weight(
            name='curvature_bias',
            shape=(self.curvature_output_dim,),
            initializer='zeros',
            trainable=True
        )

        super().build(input_shape)

    def _compute_curvature(
            self,
            embeddings: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute curvature from embeddings.

        The curvature is computed as a nonlinear function of the embedding,
        ensuring that it captures the local geometric structure of the
        representation space.

        :param embeddings: Embedding tensor of shape (batch, seq_len, embed_dim).
        :type embeddings: keras.KerasTensor
        :return: Curvature tensor with shape depending on curvature_type.
        :rtype: keras.KerasTensor
        """
        # Project embedding to curvature space
        # Shape: (batch, seq_len, curvature_output_dim)
        raw_curvature = ops.matmul(embeddings, self.curvature_projection)
        raw_curvature = raw_curvature + self.curvature_bias

        # Apply nonlinearity and scaling
        # Using tanh to bound curvature values
        raw_curvature = ops.tanh(raw_curvature) * self.curvature_scale

        # Reshape based on curvature type
        batch_size = ops.shape(embeddings)[0]
        seq_len = ops.shape(embeddings)[1]

        if self.curvature_type == 'metric':
            # Full metric tensor: ensure symmetry and positive definiteness
            curvature = ops.reshape(
                raw_curvature,
                (batch_size, seq_len, self.curvature_dim, self.curvature_dim)
            )
            # Make symmetric: (A + A^T) / 2
            curvature = (curvature + ops.transpose(curvature, (0, 1, 3, 2))) / 2.0
            # Add identity to ensure positive definiteness
            identity = ops.eye(self.curvature_dim)
            identity = ops.expand_dims(ops.expand_dims(identity, 0), 0)
            curvature = curvature + identity

        elif self.curvature_type == 'riemann':
            # Riemann-like curvature tensor (antisymmetric)
            curvature = ops.reshape(
                raw_curvature,
                (batch_size, seq_len, self.curvature_dim, self.curvature_dim)
            )
            # Make antisymmetric: (A - A^T) / 2
            curvature = (curvature - ops.transpose(curvature, (0, 1, 3, 2))) / 2.0

        elif self.curvature_type == 'ricci':
            # Ricci curvature (diagonal elements)
            curvature = raw_curvature  # Already shape (batch, seq_len, curvature_dim)

        else:  # scalar
            # Scalar curvature
            curvature = raw_curvature  # Shape (batch, seq_len, 1)

        return curvature

    def _curvature_regularization_loss(
            self,
            curvature: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute curvature smoothness regularization loss.

        Encourages neighboring tokens to have similar curvature,
        providing implicit regularization and preventing sharp
        discontinuities in the geometric structure.

        :param curvature: Curvature tensor.
        :type curvature: keras.KerasTensor
        :return: Scalar regularization loss.
        :rtype: keras.KerasTensor
        """
        if self.curvature_regularization <= 0:
            return ops.zeros(())

        # Compute curvature differences between adjacent positions
        if len(ops.shape(curvature)) >= 2:
            # Flatten curvature to (batch, seq_len, -1) for simpler computation
            flat_shape = (
                ops.shape(curvature)[0],
                ops.shape(curvature)[1],
                -1
            )
            flat_curvature = ops.reshape(curvature, flat_shape)

            # Adjacent differences
            diff = flat_curvature[:, 1:, :] - flat_curvature[:, :-1, :]

            # L2 norm of differences
            smoothness_loss = ops.mean(ops.sum(diff ** 2, axis=-1))

            return self.curvature_regularization * smoothness_loss

        return ops.zeros(())

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass: embed tokens as fields with curvature.

        :param inputs: Integer tensor of token indices with shape (batch, seq_len).
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Tuple of (embeddings, curvature):
            - embeddings: shape (batch, seq_len, embed_dim)
            - curvature: shape depends on curvature_type
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Look up embeddings
        embeddings = ops.take(self.embedding_weights, inputs, axis=0)

        # Compute curvature
        curvature = self._compute_curvature(embeddings)

        # Add regularization loss during training
        if training:
            reg_loss = self._curvature_regularization_loss(curvature)
            self.add_loss(reg_loss)

        return embeddings, curvature

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Compute the output shapes.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Tuple of (embedding_shape, curvature_shape).
        :rtype: Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]
        """
        batch_size = input_shape[0]
        seq_len = input_shape[1] if len(input_shape) > 1 else None

        embedding_shape = (batch_size, seq_len, self.embed_dim)

        if self.curvature_type in ('metric', 'riemann'):
            curvature_shape = (
                batch_size, seq_len, self.curvature_dim, self.curvature_dim
            )
        elif self.curvature_type == 'ricci':
            curvature_shape = (batch_size, seq_len, self.curvature_dim)
        else:  # scalar
            curvature_shape = (batch_size, seq_len, 1)

        return embedding_shape, curvature_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'curvature_dim': self.curvature_dim,
            'curvature_type': self.curvature_type,
            'curvature_scale': self.curvature_scale,
            'curvature_regularization': self.curvature_regularization,
            'embed_initializer': initializers.serialize(self.embed_initializer),
            'curvature_initializer': initializers.serialize(self.curvature_initializer),
            'embed_regularizer': regularizers.serialize(self.embed_regularizer),
            'curvature_regularizer': regularizers.serialize(self.curvature_regularizer),
        })
        return config