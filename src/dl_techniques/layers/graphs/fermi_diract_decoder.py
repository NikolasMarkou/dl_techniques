"""
Fermi-Dirac Decoder for Link Prediction.

This module implements the decoder used specifically for link prediction tasks
in hyperbolic graph neural networks, as defined in Equation 30 of Arevalo et al.

Key Distinction: Unlike standard HGCN decoders that use hyperbolic distance, this
decoder uses Euclidean distance in accordance with the sHGCN simplifications.

Mathematical Formula:
    p(u,v) = sigmoid(-(d²(u,v) - r) / t)

where:
    - d²(u,v): Squared Euclidean distance between node embeddings
    - r: Learnable threshold parameter
    - t: Learnable temperature parameter
"""

import keras
from keras import layers, initializers, ops
from typing import List, Optional, Any


@keras.saving.register_keras_serializable()
class FermiDiracDecoder(layers.Layer):
    """
    Fermi-Dirac decoder for edge probability prediction using Euclidean distances.

    This decoder computes the probability of an edge existing between two nodes
    based on the Euclidean distance between their embeddings. The formulation is
    inspired by the Fermi-Dirac distribution from physics, providing a smooth
    sigmoid-like transition around a learnable threshold.

    **Key Feature**: Uses Euclidean distance rather than hyperbolic distance,
    consistent with sHGCN's philosophy of operating in tangent space where possible.

    **Mathematical Operation**:
        Given embeddings u, v ∈ ℝ^D:

        1. Compute squared distance: d² = ||u - v||²
        2. Compute score: s = (d² - r) / t
        3. Compute probability: p = sigmoid(-s) = 1 / (1 + exp(s))

    where:
        - r is a learnable threshold (optimal distance for edges)
        - t is a learnable temperature (controls transition sharpness)

    **Intuition**:
        - When d² < r: Nodes are close, p → 1 (likely edge)
        - When d² > r: Nodes are far, p → 0 (unlikely edge)
        - t controls how sharp this transition is

    Args:
        r_initializer: Initializer for threshold parameter r. Defaults to
            Constant(2.0), assuming normalized embeddings.
        t_initializer: Initializer for temperature parameter t. Defaults to
            Constant(1.0).
        **kwargs: Additional keyword arguments for Layer base class.

    Input:
        List of two tensors:
        - embeddings_u: Shape (batch_size, feature_dim) for source nodes
        - embeddings_v: Shape (batch_size, feature_dim) for target nodes

    Output:
        Tensor of shape (batch_size,) with edge probabilities in range [0, 1].

    Attributes:
        r: Threshold parameter, scalar tensor.
        t: Temperature parameter, scalar tensor.

    Example:
        ```python
        # Create decoder
        decoder = FermiDiracDecoder()

        # Get embeddings for node pairs
        embeddings = model([features, adj])  # [num_nodes, embed_dim]

        # Predict edges for specific pairs
        u_indices = [0, 1, 2]  # Source nodes
        v_indices = [5, 6, 7]  # Target nodes

        u_embed = tf.gather(embeddings, u_indices)  # [3, embed_dim]
        v_embed = tf.gather(embeddings, v_indices)  # [3, embed_dim]

        # Get probabilities
        probs = decoder([u_embed, v_embed])  # [3,]
        print(probs)  # e.g., [0.89, 0.12, 0.67]

        # For training with binary cross-entropy
        loss = keras.losses.binary_crossentropy(
            y_true=[1, 0, 1],  # Ground truth edges
            y_pred=probs
        )
        ```

    Note:
        - Embeddings should be from the same model to ensure consistency
        - Works with any embedding dimension
        - Both r and t are learned during training
        - Output can be directly used with binary cross-entropy loss

    References:
        Arevalo et al., Equation 30: Link prediction with Euclidean distance
    """

    def __init__(
            self,
            r_initializer: Union[str, initializers.Initializer] = initializers.Constant(2.0),
            t_initializer: Union[str, initializers.Initializer] = initializers.Constant(1.0),
            **kwargs: Any
    ) -> None:
        """Initialize Fermi-Dirac decoder with learnable parameters."""
        super().__init__(**kwargs)

        self.r_initializer = keras.initializers.get(r_initializer)
        self.t_initializer = keras.initializers.get(t_initializer)

    def build(self, input_shape: List[tuple]) -> None:
        """
        Create learnable parameters.

        Args:
            input_shape: List of two shapes for [u_embeddings, v_embeddings].
                Both should have the same feature dimension.
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
            inputs: List[keras.KerasTensor]
    ) -> keras.KerasTensor:
        """
        Compute edge probabilities from embedding pairs.

        Args:
            inputs: List of [u_embeddings, v_embeddings], both shape (batch, dim).

        Returns:
            Edge probabilities of shape (batch,) with values in [0, 1].
        """
        u, v = inputs

        # Compute squared Euclidean distance
        # d² = ||u - v||² = sum((u_i - v_i)²)
        # Shape: [batch, dim] -> [batch,]
        dist_squared = ops.sum(ops.square(u - v), axis=-1)

        # Compute Fermi-Dirac score
        # s = (d² - r) / t
        # Shape: [batch,]
        scores = (dist_squared - self.r) / self.t

        # Apply sigmoid: p = 1 / (1 + exp(s)) = sigmoid(-s)
        # When d² < r: scores < 0 → sigmoid(-scores) → 1 (likely edge)
        # When d² > r: scores > 0 → sigmoid(-scores) → 0 (unlikely edge)
        # Shape: [batch,]
        probabilities = ops.sigmoid(-scores)

        return probabilities

    def compute_output_shape(
            self,
            input_shape: List[tuple]
    ) -> tuple:
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
            Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            'r_initializer': keras.initializers.serialize(self.r_initializer),
            't_initializer': keras.initializers.serialize(self.t_initializer),
        })
        return config