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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.losses.utilization_loss")
class MANNUtilizationLoss(keras.losses.Loss):
    """Encourages a Memory-Augmented Neural Network to use its memory.

    Combines three regularization terms to prevent memory collapse:

    1. **Entropy loss**: Encourages diverse memory access patterns by
       maximizing the entropy of softmax-normalized memory vectors.
    2. **Variance loss**: Encourages temporal variation in memory usage
       by penalizing low variance of consecutive memory state differences.
    3. **Magnitude loss**: Encourages non-trivial memory activation by
       penalizing low memory vector norms.

    ``call(y_true, y_pred)`` takes the memory state as ``y_pred``, shape
    ``(batch, time, memory_dim)``; ``y_true`` is ignored. There is no third
    argument.

    The objective is unbounded below: the entropy, log-variance and norm terms
    are negated, so scaling the memory up lowers the loss without limit
    (measured: input scale 1 / 10 / 100 / 1e4 gives -7 / -43 / -417 / -40053).
    A bounded-below hinge redesign is NOT done (plan-2026-11-11T131349-13b28b37
    D-013); keep the weights small and treat this as a regularizer.

    The loss is one value per sample (``call`` returns ``(batch,)``), so
    ``sample_weight`` and ``reduction=`` act on rows. It is batch-coupled through
    the variance term ONLY: the temporal variance is a batch statistic, handled by a
    detached first-order surrogate (see ``call``), so a row with a zero
    ``sample_weight`` still shifts the other rows through the batch variance and
    mean. Zeroing a weight therefore does NOT equal dropping the row, except with
    ``variance_weight=0``; the entropy and magnitude terms are plain per-row means.

    Direct ``call`` limits, documented and not fixed (``call`` does not cast; the
    loss object does): it needs ``time >= 2`` (``time == 1`` gives NaN, an empty
    temporal difference), and on float16 the norm overflows past a scale of about
    300 (NaN). Through ``__call__`` the inputs are cast to float32 first.

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
        """Compute MANN utilization loss, one value per sample.

        Treats ``y_pred`` as the memory state tensor of shape
        ``(batch, time, memory_dim)``. ``y_true`` is ignored (pass a
        dummy tensor of matching batch size).

        Entropy and magnitude are means over time of one row, so they are
        per-sample. The temporal variance is a BATCH statistic (``ops.var`` over
        every axis of the temporal difference, including batch, inside a ``-log``),
        so it has no exact per-row form. Returning ``-log(V)`` broadcast to every
        row is not an option: Keras multiplies ``values * sample_weight`` before it
        reduces, so a constant keeps a zero-weighted row's gradient alive. This uses
        the first-order expansion of ``-log`` about the detached batch variance, as
        ``MultistepLoss._gtmse`` does::

            row_b = -log(V + eps) - (v_b - V) / (V + eps)

        with ``v_b`` the row's mean squared deviation from the detached batch mean
        of the temporal difference and ``V = mean_b(v_b)`` detached. Two identities
        hold: the mean over rows of the variance rows is ``-log(V + eps)`` (value of
        the old scalar), and its gradient is that of the old scalar. A row with a
        zero ``sample_weight`` receives exactly zero gradient.

        Args:
            y_true: Ignored (required by Keras Loss API).
            y_pred: Memory state tensor ``(batch, time, memory_dim)``.

        Returns:
            Tensor of shape ``(batch,)``, one utilization loss per sample. With the
            default reduction the loss value equals the scalar this method returned
            before it was made per-sample.
        """
        memory_vectors = y_pred

        # Entropy: encourage diverse memory access (mean over time, per row)
        memory_probs = ops.softmax(memory_vectors, axis=-1)
        entropy = -ops.sum(
            memory_probs * ops.log(memory_probs + 1e-10), axis=-1
        )
        entropy_loss = -ops.mean(entropy, axis=1)

        # Magnitude: encourage non-trivial memory activation (mean over time)
        memory_norm = ops.sqrt(
            ops.sum(ops.square(memory_vectors), axis=-1)
        )
        magnitude_loss = -ops.mean(memory_norm, axis=1)

        # Temporal variance: encourage changing memory patterns.
        # DECISION plan-2026-11-11T223825-64d1b401/D-005
        # The variance is a BATCH statistic inside a log, so it is the detached
        # first-order surrogate of MultistepLoss._gtmse. Do NOT broadcast
        # -log(V) to every row: a zero-weighted row would keep a gradient
        # (measured 2.4e-4). Do NOT use -log(var_b) per row: that is a different
        # objective (Jensen gap, default value -0.0701 against -0.0809).
        temporal_diff = memory_vectors[:, 1:, :] - memory_vectors[:, :-1, :]
        batch_mean = ops.stop_gradient(ops.mean(temporal_diff))
        row_var = ops.mean(ops.square(temporal_diff - batch_mean), axis=(1, 2))
        batch_var = ops.stop_gradient(ops.mean(row_var))
        variance_row = (
            -ops.log(batch_var + 1e-10)
            - (row_var - batch_var) / (batch_var + 1e-10)
        )

        return (
            self.entropy_weight * entropy_loss
            + self.variance_weight * variance_row
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


@register_dl_technique("dl_techniques.losses.utilization_loss")
class GNNUtilizationLoss(keras.losses.Loss):
    """Encourages a Graph Neural Network to use its entity representations.

    Combines three regularization terms to prevent GNN component collapse:

    1. **Diversity loss**: Penalizes high pairwise cosine similarity between
       entity embeddings, encouraging each entity to be distinct.
    2. **Activation loss**: Penalizes low entity embedding norms, encouraging
       non-trivial activations.
    3. **Variance loss**: Penalizes low variance across entities within each
       sample, encouraging spread in the representation space.

    ``call(y_true, y_pred)`` takes the entity embeddings as ``y_pred``, shape
    ``(batch, num_entities, embed_dim)``; ``y_true`` is ignored. There is no
    third argument.

    Every term is a plain mean over the entity and feature axes of one sample,
    so ``call`` returns one value per sample, shape ``(batch,)``, and no row
    reads another: the loss is not batch-coupled. ``reduction="none"`` returns
    those rows, ``reduction="sum"`` returns ``batch`` times the default, and
    ``sample_weight`` charges each row its own value times its own weight.

    The activation and variance terms are negated norms and variances, so the
    objective is unbounded below: scaling the embeddings up lowers the loss
    without limit. A bounded-below hinge redesign is NOT done
    (plan-2026-11-11T131349-13b28b37 D-013).

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
        """Compute GNN utilization loss, one value per sample.

        Treats ``y_pred`` as the entity embedding tensor of shape
        ``(batch, num_entities, embed_dim)``. ``y_true`` is ignored
        (pass a dummy tensor of matching batch size). A direct call does not
        cast; the diversity mask follows the dtype of ``y_pred``.

        Args:
            y_true: Ignored (required by Keras Loss API).
            y_pred: Entity embeddings ``(batch, num_entities, embed_dim)``.

        Returns:
            Per-sample utilization loss, shape ``(batch,)``. Row ``b`` depends
            on sample ``b`` only.
        """
        entity_embeddings = y_pred

        # Diversity: penalize high pairwise cosine similarity
        normalized = ops.normalize(entity_embeddings, axis=-1)
        similarity = ops.einsum("bnd,bmd->bnm", normalized, normalized)
        num_entities = ops.shape(entity_embeddings)[1]
        mask = ops.cast(1.0 - ops.eye(num_entities), similarity.dtype)
        diversity_loss = ops.mean(ops.abs(similarity * mask), axis=(1, 2))

        # Activation: encourage non-zero entity norms
        entity_norm = ops.sqrt(
            ops.sum(ops.square(entity_embeddings), axis=-1)
        )
        activation_loss = -ops.mean(entity_norm, axis=1)

        # Variance: encourage entity spread (across entities, within each sample)
        entity_var = ops.var(entity_embeddings, axis=1)
        variance_loss = -ops.mean(entity_var, axis=-1)

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

# ---------------------------------------------------------------------
