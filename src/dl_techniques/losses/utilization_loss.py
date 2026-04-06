"""
Component Utilization Losses
============================

Auxiliary losses that encourage model sub-components (memory modules,
graph neural networks) to actively use their capacity. Without these
regularizers, components can "collapse" — producing near-constant outputs
while the main pathway does all the work.

Example::

    from dl_techniques.losses import MANNUtilizationLoss, GNNUtilizationLoss

    mann_loss = MANNUtilizationLoss(entropy_weight=0.01)
    gnn_loss = GNNUtilizationLoss(diversity_weight=0.01)
"""

import keras
from keras import ops
from typing import Any, Dict


@keras.saving.register_keras_serializable(package="dl_techniques.losses")
class MANNUtilizationLoss(keras.losses.Loss):
    """Encourages a Memory-Augmented Neural Network to use its memory.

    Combines three regularization terms to prevent memory collapse:

    1. **Entropy loss**: Encourages diverse memory access patterns by
       maximizing the entropy of softmax-normalized memory vectors.
    2. **Variance loss**: Encourages temporal variation in memory usage
       by penalizing low variance of consecutive memory state differences.
    3. **Magnitude loss**: Encourages non-trivial memory activation by
       penalizing low memory vector norms.

    The ``call`` method expects a third argument ``memory_vectors`` with
    shape ``(batch, time, memory_dim)`` representing the raw memory state
    across time steps. During training this is typically passed as an
    auxiliary output from the model.

    Args:
        entropy_weight: Weight for the entropy regularization term.
        write_weight: Weight for the magnitude (write activation) term.
        variance_weight: Weight for the temporal variance term.
        name: Loss name for logging.
        **kwargs: Additional arguments passed to ``keras.losses.Loss``.
    """

    def __init__(
        self,
        entropy_weight: float = 0.01,
        write_weight: float = 0.01,
        variance_weight: float = 0.01,
        name: str = "mann_utilization_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.entropy_weight = entropy_weight
        self.write_weight = write_weight
        self.variance_weight = variance_weight

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute MANN utilization loss.

        Treats ``y_pred`` as the memory state tensor of shape
        ``(batch, time, memory_dim)``. ``y_true`` is ignored (pass a
        dummy tensor of matching batch size).

        Args:
            y_true: Ignored (required by Keras Loss API).
            y_pred: Memory state tensor ``(batch, time, memory_dim)``.

        Returns:
            Scalar utilization loss.
        """
        memory_vectors = y_pred

        # Entropy: encourage diverse memory access
        memory_probs = ops.softmax(memory_vectors, axis=-1)
        entropy = -ops.sum(
            memory_probs * ops.log(memory_probs + 1e-10), axis=-1
        )
        entropy_loss = -ops.mean(entropy)

        # Temporal variance: encourage changing memory patterns
        temporal_diff = memory_vectors[:, 1:, :] - memory_vectors[:, :-1, :]
        variance = ops.var(temporal_diff)
        variance_loss = -ops.log(variance + 1e-10)

        # Magnitude: encourage non-trivial memory activation
        memory_norm = ops.sqrt(
            ops.sum(ops.square(memory_vectors), axis=-1)
        )
        magnitude_loss = -ops.mean(memory_norm)

        return (
            self.entropy_weight * entropy_loss
            + self.variance_weight * variance_loss
            + self.write_weight * magnitude_loss
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "entropy_weight": self.entropy_weight,
            "write_weight": self.write_weight,
            "variance_weight": self.variance_weight,
        })
        return config


@keras.saving.register_keras_serializable(package="dl_techniques.losses")
class GNNUtilizationLoss(keras.losses.Loss):
    """Encourages a Graph Neural Network to use its entity representations.

    Combines three regularization terms to prevent GNN component collapse:

    1. **Diversity loss**: Penalizes high pairwise cosine similarity between
       entity embeddings, encouraging each entity to be distinct.
    2. **Activation loss**: Penalizes low entity embedding norms, encouraging
       non-trivial activations.
    3. **Variance loss**: Penalizes low variance across entities within a
       batch, encouraging spread in the representation space.

    The ``call`` method expects a third argument ``entity_embeddings`` with
    shape ``(batch, num_entities, embed_dim)``.

    Args:
        diversity_weight: Weight for the diversity regularization term.
        attention_weight: Weight for the variance (attention spread) term.
        activation_weight: Weight for the activation magnitude term.
        name: Loss name for logging.
        **kwargs: Additional arguments passed to ``keras.losses.Loss``.
    """

    def __init__(
        self,
        diversity_weight: float = 0.01,
        attention_weight: float = 0.01,
        activation_weight: float = 0.01,
        name: str = "gnn_utilization_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.diversity_weight = diversity_weight
        self.attention_weight = attention_weight
        self.activation_weight = activation_weight

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute GNN utilization loss.

        Treats ``y_pred`` as the entity embedding tensor of shape
        ``(batch, num_entities, embed_dim)``. ``y_true`` is ignored
        (pass a dummy tensor of matching batch size).

        Args:
            y_true: Ignored (required by Keras Loss API).
            y_pred: Entity embeddings ``(batch, num_entities, embed_dim)``.

        Returns:
            Scalar utilization loss.
        """
        entity_embeddings = y_pred

        # Diversity: penalize high pairwise cosine similarity
        normalized = ops.normalize(entity_embeddings, axis=-1)
        similarity = ops.einsum("bnd,bmd->bnm", normalized, normalized)
        num_entities = ops.shape(entity_embeddings)[1]
        mask = 1.0 - ops.eye(num_entities)
        diversity_loss = ops.mean(ops.abs(similarity * mask))

        # Activation: encourage non-zero entity norms
        entity_norm = ops.sqrt(
            ops.sum(ops.square(entity_embeddings), axis=-1)
        )
        activation_loss = -ops.mean(entity_norm)

        # Variance: encourage entity spread
        entity_var = ops.var(entity_embeddings, axis=1)
        variance_loss = -ops.mean(entity_var)

        return (
            self.diversity_weight * diversity_loss
            + self.activation_weight * activation_loss
            + self.attention_weight * variance_loss
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "diversity_weight": self.diversity_weight,
            "attention_weight": self.attention_weight,
            "activation_weight": self.activation_weight,
        })
        return config
