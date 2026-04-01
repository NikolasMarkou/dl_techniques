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
    """Fermi-Dirac decoder for edge probability prediction.

    Computes the probability of an edge between two nodes from the Euclidean
    distance of their embeddings, inspired by the Fermi-Dirac distribution.
    Given embeddings u, v in R^D the probability is
    p = sigmoid(-(d^2 - r) / t) where d^2 = ||u - v||^2, *r* is a learnable
    threshold and *t* is a learnable temperature controlling transition
    sharpness. When d^2 < r the probability tends to 1 (likely edge);
    when d^2 > r it tends to 0 (Eq. 30, Arevalo et al.).

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐   ┌──────────────┐
        │ u embeddings │   │ v embeddings │
        │  [B, D]      │   │  [B, D]      │
        └──────┬───────┘   └──────┬───────┘
               └──────────┬───────┘
                          ▼
               ┌─────────────────────┐
               │  d² = ||u - v||²    │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │  s = (d² - r) / t   │
               │  r, t: learnable    │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │  p = sigmoid(-s)    │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │  Output  [B]        │
               └─────────────────────┘

    :param r_initializer: Initializer for threshold *r*.
        Defaults to ``Constant(2.0)``.
    :type r_initializer: Union[str, keras.initializers.Initializer]
    :param t_initializer: Initializer for temperature *t*.
        Defaults to ``Constant(1.0)``.
    :type t_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    """

    def __init__(
            self,
            r_initializer: Union[str, keras.initializers.Initializer] = None,
            t_initializer: Union[str, keras.initializers.Initializer] = None,
            **kwargs: Any
    ) -> None:
        """Initialise Fermi-Dirac decoder with learnable parameters."""
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
        """Create learnable parameters.

        :param input_shape: List of two shapes for ``[u_embeddings, v_embeddings]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
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
        """Compute edge probabilities from embedding pairs.

        :param inputs: List of ``[u_embeddings, v_embeddings]``, both shape ``(batch, dim)``.
        :type inputs: List[keras.KerasTensor]
        :param training: Whether in training mode (unused, kept for API consistency).
        :type training: Optional[bool]
        :return: Edge probabilities of shape ``(batch,)`` in ``[0, 1]``.
        :rtype: keras.KerasTensor
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
        """Compute output shape given input shapes.

        :param input_shape: List of ``[u_shape, v_shape]``.
        :type input_shape: List[Tuple[Optional[int], ...]]
        :return: Output shape tuple ``(batch_size,)``.
        :rtype: Tuple[Optional[int]]
        """
        u_shape = input_shape[0]
        return (u_shape[0],)

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'r_initializer': keras.initializers.serialize(self.r_initializer),
            't_initializer': keras.initializers.serialize(self.t_initializer),
        })
        return config

# ---------------------------------------------------------------------
