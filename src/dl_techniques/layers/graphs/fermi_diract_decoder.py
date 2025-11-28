"""
Fermi-Dirac Decoder for Link Prediction.

This module implements the decoder used specifically for link prediction tasks
in hyperbolic graph neural networks, as defined in Equation 30 of Arevalo et al.

Key Distinction: Unlike standard HGCN decoders that use hyperbolic distance, this
decoder uses Euclidean distance in accordance with the sHGCN simplifications.
"""

import keras
from typing import List, Any, Union, Tuple, Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class FermiDiracDecoder(keras.layers.Layer):
    """
    Fermi-Dirac decoder for edge probability prediction using Euclidean distances.

    This decoder computes the probability of an edge existing between two nodes
    based on the Euclidean distance between their embeddings. The formulation is
    inspired by the Fermi-Dirac distribution from physics, providing a smooth
    sigmoid-like transition around a learnable threshold.

    **Intent**: Provide a differentiable link prediction decoder that maps pairwise
    node distances to edge probabilities via learnable threshold and temperature
    parameters, enabling end-to-end training of graph neural networks for link
    prediction tasks.

    **Architecture**:
    ::

        Inputs: [u_embeddings, v_embeddings]
                      │              │
                      └──────┬───────┘
                             │
                             ▼
               ┌─────────────────────────┐
               │  Squared Euclidean Dist │
               │    d² = ||u - v||²      │
               └─────────────────────────┘
                             │
                             ▼
               ┌─────────────────────────┐
               │    Score Computation    │
               │    s = (d² - r) / t     │
               │                         │
               │  r: learnable threshold │
               │  t: learnable temp      │
               └─────────────────────────┘
                             │
                             ▼
               ┌─────────────────────────┐
               │   Sigmoid Activation    │
               │    p = sigmoid(-s)      │
               └─────────────────────────┘
                             │
                             ▼
                   Output: probabilities
                     shape (batch,)

    **Mathematical Operations**:
        Given embeddings u, v in R^D:

        1. **Squared Distance**: d² = sum((u_i - v_i)²)
        2. **Score**: s = (d² - r) / t
        3. **Probability**: p = sigmoid(-s) = 1 / (1 + exp(s))

    **Intuition**:
        - When d² < r: Nodes are close, p -> 1 (likely edge)
        - When d² > r: Nodes are far, p -> 0 (unlikely edge)
        - t controls how sharp this transition is

    Args:
        r_initializer: Initializer for threshold parameter r. Accepts string names
            ('zeros', 'ones') or Initializer instances. Defaults to Constant(2.0),
            assuming normalized embeddings.
        t_initializer: Initializer for temperature parameter t. Accepts string names
            or Initializer instances. Defaults to Constant(1.0).
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        List of two tensors:
            - embeddings_u: Shape ``(batch_size, feature_dim)`` for source nodes.
            - embeddings_v: Shape ``(batch_size, feature_dim)`` for target nodes.
        Both tensors must have matching feature dimensions.

    Output shape:
        1D tensor with shape ``(batch_size,)`` containing edge probabilities
        in range [0, 1].

    Attributes:
        r: Threshold parameter, scalar tensor. Represents optimal distance for edges.
        t: Temperature parameter, scalar tensor. Controls transition sharpness.

    Raises:
        ValueError: If input is not a list/tuple of exactly two tensors.
        ValueError: If embedding dimensions do not match between u and v.

    References:
        Arevalo et al., Equation 30: Link prediction with Euclidean distance.

    Note:
        - Embeddings should be from the same model to ensure consistency.
        - Works with any embedding dimension.
        - Both r and t are learned during training.
        - Output can be directly used with binary cross-entropy loss.
    """

    def __init__(
            self,
            r_initializer: Union[str, keras.initializers.Initializer] = None,
            t_initializer: Union[str, keras.initializers.Initializer] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize Fermi-Dirac decoder with learnable parameters.

        Args:
            r_initializer: Initializer for threshold parameter r.
            t_initializer: Initializer for temperature parameter t.
            **kwargs: Additional keyword arguments for Layer base class.
        """
        super().__init__(**kwargs)

        # Store initializers with defaults
        self.r_initializer = keras.initializers.get(
            r_initializer if r_initializer is not None
            else keras.initializers.Constant(2.0)
        )
        self.t_initializer = keras.initializers.get(
            t_initializer if t_initializer is not None
            else keras.initializers.Constant(1.0)
        )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Create learnable parameters.

        Args:
            input_shape: List of two shapes for [u_embeddings, v_embeddings].
                Both should have the same feature dimension.

        Raises:
            ValueError: If input_shape is not a list/tuple of two shapes.
            ValueError: If embedding dimensions do not match.
        """
        # Validate input shapes
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"Input must be [u_embeddings, v_embeddings], got {input_shape}"
            )

        u_shape, v_shape = input_shape

        if u_shape[-1] != v_shape[-1]:
            raise ValueError(
                f"Embedding dimensions must match: u={u_shape[-1]}, v={v_shape[-1]}"
            )

        # Threshold parameter r: optimal distance for edges
        self.r = self.add_weight(
            shape=(),
            initializer=self.r_initializer,
            name='threshold',
            trainable=True
        )

        # Temperature parameter t: controls transition sharpness
        self.t = self.add_weight(
            shape=(),
            initializer=self.t_initializer,
            name='temperature',
            trainable=True
        )

        super().build(input_shape)

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute edge probabilities from embedding pairs.

        Args:
            inputs: List of [u_embeddings, v_embeddings], both shape (batch, dim).
            training: Boolean flag for training mode (unused but included for
                API consistency).

        Returns:
            Edge probabilities of shape (batch,) with values in [0, 1].
        """
        u, v = inputs

        # Compute squared Euclidean distance
        # d² = ||u - v||² = sum((u_i - v_i)²)
        # Shape: [batch, dim] -> [batch,]
        dist_squared = keras.ops.sum(keras.ops.square(u - v), axis=-1)

        # Compute Fermi-Dirac score
        # s = (d² - r) / t
        # Shape: [batch,]
        scores = (dist_squared - self.r) / self.t

        # Apply sigmoid: p = 1 / (1 + exp(s)) = sigmoid(-s)
        # When d² < r: scores < 0 -> sigmoid(-scores) -> 1 (likely edge)
        # When d² > r: scores > 0 -> sigmoid(-scores) -> 0 (unlikely edge)
        # Shape: [batch,]
        probabilities = keras.ops.sigmoid(-scores)

        return probabilities

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int]]:
        """
        Compute output shape given input shapes.

        Args:
            input_shape: List of [u_shape, v_shape].

        Returns:
            Output shape tuple (batch_size,).
        """
        u_shape = input_shape[0]
        return (u_shape[0],)

    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all constructor arguments needed to
            reconstruct the layer.
        """
        config = super().get_config()
        config.update({
            'r_initializer': keras.initializers.serialize(self.r_initializer),
            't_initializer': keras.initializers.serialize(self.t_initializer),
        })
        return config

# ---------------------------------------------------------------------
