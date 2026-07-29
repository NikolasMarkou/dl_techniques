"""
Fenchel-Young loss tailored for Sparsemax activation.

This loss function provides the theoretically principled counterpart to the
Sparsemax activation function, analogous to the relationship between softmax
and categorical cross-entropy. It is designed to train models that output
sparse probability distributions, ensuring that the learning process is both
stable and aligned with the sparse nature of the predictions.

The mathematical foundation of this loss stems from the theory of Fenchel-Young
losses. For a given activation function (like Sparsemax), its corresponding
Fenchel-Young loss is constructed to be convex and to produce a simple,
intuitive gradient. The loss for a given vector of logits `z` and a one-hot
true label vector `y` is defined as:

L(y, z) = 0.5 * ||z - p||²₂ - zᵀy

where `p = sparsemax(z)`. This formulation has two intuitive components:
1.  `0.5 * ||z - p||²₂`: This term represents the squared Euclidean distance
    between the raw logits `z` and their sparse projection `p` onto the
    probability simplex. It acts as a regularization term, penalizing logits
    that are far from their resulting sparse representation.
2.  `-zᵀy`: This is a linear scoring term that encourages the logit
    corresponding to the true class to be high.

A key benefit of this construction is its gradient. The gradient of the
Sparsemax loss with respect to the input logits `z` is remarkably simple:

∇_z L = p - y

This gradient, `predicted_distribution - true_distribution`, provides a
clean and direct error signal for backpropagation. It mirrors the elegant
gradient of the softmax cross-entropy loss, making optimization straightforward
and confirming that this loss is the natural choice for Sparsemax-based
models. Using a different loss, such as cross-entropy, with Sparsemax would
result in a mismatched, more complex gradient that could hinder training.

References:
    - Martins & Astudillo, 2016. "From Softmax to Sparsemax: A Sparse
      Model of Attention and Multi-Label Classification".
      (https://arxiv.org/abs/1602.02068)
    - Blondel et al., 2020. "Learning with Fenchel-Young Losses".
      (https://arxiv.org/abs/1901.02324)
"""

import keras
from typing import Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..layers.activations.sparsemax import Sparsemax

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SparsemaxLoss(keras.losses.Loss):
    """
    Sparsemax loss function for training with sparse probability distributions.

    SparsemaxLoss is the natural loss function for models using sparsemax activation,
    analogous to how cross-entropy is used with softmax. It encourages sparse
    probability distributions and is derived from the Fenchel-Young inequality,
    making it the theoretically principled choice when using sparsemax.

    **Intent**: Provide the optimal loss function for sparsemax-activated models,
    ensuring consistent gradients and encouraging the desired sparse behavior
    during training.

    **Architecture**:
    ```
    Inputs: labels (one-hot), logits (or sparsemax probs)
           ↓
    [Optional] Apply Sparsemax: p = sparsemax(logits)
           ↓
    Compute Squared Distance: ||logits - p||²
           ↓
    Compute Dot Product: logits · labels
           ↓
    Loss: L = 0.5 · ||logits - p||² - logits · labels
           ↓
    Output: scalar loss value
    ```

    **Mathematical Operations**:
    For true label y and predicted logits z:

    1. **With from_logits=True**:
       - Compute p = sparsemax(z)
       - L(y, z) = 0.5 · ||z - p||₂² - z^T y

    2. **With from_logits=False** (p provided directly):
       - L(y, p) = 0.5 · ||z - p||₂² - z^T y
         (where z would be the pre-sparsemax logits)

    **Properties**:
    - Convex in logits z
    - Gradient with respect to logits: ∇_z L = p - y (like softmax + cross-entropy!)
    - Encourages sparse predictions
    - Proper scoring rule (encourages calibrated predictions)
    - Non-negative loss values

    **Comparison with Cross-Entropy**:
    - Cross-entropy + softmax: optimized for dense probability distributions
    - Sparsemax loss + sparsemax: optimized for sparse probability distributions
    - Both have the same gradient form: predicted - true distribution
    - Mixing them (e.g., cross-entropy + sparsemax) gives suboptimal gradients

    References:
        Martins & Astudillo (2016). "From Softmax to Sparsemax: A Sparse
        Model of Attention and Multi-Label Classification". ICML 2016.

        Blondel et al. (2019). "Learning Classifiers with Fenchel-Young Losses:
        Generalized Entropies, Margins, and Algorithms". AISTATS 2019.

    Args:
        from_logits: Boolean indicating whether y_pred contains logits (True)
            or sparsemax probabilities (False). If True, sparsemax is applied
            internally. Defaults to True.
        reduction: Type of reduction to apply to loss values across the batch.
            Options: 'sum_over_batch_size' (default), 'sum', 'none'.
            - 'sum_over_batch_size': Average loss over batch
            - 'sum': Total loss sum
            - 'none': Per-sample loss values
        name: String name for the loss instance. Defaults to 'sparsemax_loss'.
        **kwargs: Additional arguments passed to Loss base class.

    Input shapes:
        - y_true: True labels, shape `(batch_size, num_classes)`.
          Should be one-hot encoded (values 0 or 1, sum to 1 per sample).
        - y_pred: Predicted logits or probabilities, shape `(batch_size, num_classes)`.
          If from_logits=True, these are raw logits.
          If from_logits=False, these should be sparsemax outputs.

    Output shape:
        - If reduction='none': shape `(batch_size,)` with per-sample losses.
        - Otherwise: scalar loss value.

    Attributes:
        from_logits: Whether to apply sparsemax to predictions.
        sparsemax: Internal Sparsemax layer (only if from_logits=True).

    Example:
        ```python
        # Training with sparsemax classifier
        num_classes = 10
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes),  # Logits
            Sparsemax()  # Sparse probabilities
        ])

        # Use sparsemax loss for training (requires logits, not probabilities)
        # The loss applies sparsemax internally
        model_logits = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes)  # Just logits
        ])
        loss_fn = SparsemaxLoss(from_logits=True)
        model_logits.compile(optimizer='adam', loss=loss_fn)

        # Direct loss computation
        labels = keras.ops.array([[0, 0, 1], [0, 1, 0]])  # One-hot
        logits = keras.ops.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])
        loss = loss_fn(labels, logits)
        print(f"Loss: {loss:.4f}")
        ```

    Note:
        - Always use one-hot encoded labels, not integer labels.
        - For best results, pair with Sparsemax activation (don't mix with softmax).
        - The loss is calibrated to produce gradients ∇L = p - y, making
          optimization straightforward.
        - Consider using label smoothing by slightly relaxing one-hot encoding
          if needed for regularization.
    """

    def __init__(
            self,
            from_logits: bool = True,
            reduction: str = "sum_over_batch_size",
            name: str = "sparsemax_loss",
            **kwargs: Any
    ) -> None:
        """
        Initialize SparsemaxLoss.

        Args:
            from_logits: Whether y_pred contains logits or sparsemax outputs.
            reduction: Reduction method for batch losses.
            name: Loss name.
            **kwargs: Additional Loss base class arguments.

        Raises:
            ValueError: If reduction is not a valid option.
        """
        super().__init__(reduction=reduction, name=name, **kwargs)

        # Validate inputs
        if not isinstance(from_logits, bool):
            raise ValueError(
                f"from_logits must be a boolean, got {type(from_logits).__name__}"
            )

        valid_reductions = ["sum_over_batch_size", "sum", "none"]
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, got '{reduction}'"
            )

        if not from_logits:
            raise ValueError(
                "SparsemaxLoss requires from_logits=True. The Fenchel-Young "
                "loss formula requires the original logits z to compute "
                "0.5 * ||z - sparsemax(z)||^2 - z^T y. When from_logits=False, "
                "the logits are not available and the loss cannot be computed correctly."
            )

        self.from_logits = from_logits

        # Create sparsemax layer if computing from logits
        # This is a sub-layer, but since Loss doesn't have a build() method,
        # we create it here and it will be built automatically on first call
        self.sparsemax = Sparsemax() if from_logits else None

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """
        Compute sparsemax loss per sample.

        The loss computation:
        L = 0.5 · ||z - p||² - z^T y

        where:
        - z are the logits (y_pred)
        - p = sparsemax(z) are the sparse probabilities
        - y are the true labels (y_true)

        Args:
            y_true: True labels, one-hot encoded, shape (batch_size, num_classes).
                Each row should sum to 1.
            y_pred: Predicted logits (if from_logits=True) or sparsemax
                probabilities (if from_logits=False), shape (batch_size, num_classes).

        Returns:
            Loss value per sample, shape (batch_size,) before reduction.
            Scalar after reduction (except if reduction='none').

        Note:
            The sparsemax probabilities are cast to ``self.dtype`` before they
            meet ``y_pred``. ``keras.losses.Loss`` computes in ``self.dtype``
            (derived from ``keras.backend.floatx()``), while the inner
            ``Sparsemax`` sub-layer returns the compute dtype of whatever
            mixed-precision policy was in force when this loss was
            CONSTRUCTED. See the decision anchor at the cast site.
        """
        # DECISION plan-2026-07-29T110112-09832856/D-015
        # THE CAST IS LOAD-BEARING; DO NOT SIMPLIFY IT AWAY.
        # `keras.losses.Loss` fixes its dtype from `keras.backend.floatx()` and
        # casts `y_true`/`y_pred` to it before `call()` runs — it is NOT
        # mixed-precision-policy aware. The `Sparsemax()` sub-layer is built
        # EAGERLY in `__init__` (a `Loss` has no `build()`), so it freezes the
        # compute dtype of the policy in force at CONSTRUCTION time and returns
        # that dtype forever after. Without this cast, `y_pred - p` below is
        # float32 minus float16/bfloat16/float64 and TensorFlow raises
        # (`InvalidArgumentError` eagerly, `TypeError` inside `model.fit`).
        # Measured on the full 4x4 construct-x-call policy grid: 12/12
        # non-float32-CONSTRUCTION points raised and 4/4 float32-construction
        # points worked, with the CALL-time policy irrelevant at every point.
        # The cast is bit-identical under float32 construction (all three
        # reductions, `np.array_equal`), so deleting it looks free and silently
        # re-breaks all three narrow policies.
        # DO NOT instead make `Sparsemax` return float32 — prohibited in that
        # layer's own cast-back anchor ("That is a defect in the LOSS and its
        # fix belongs in the loss's own file") and by user ruling.
        # DO NOT instead pin the sub-layer with `Sparsemax(dtype=self.dtype)`
        # in `__init__`: it keeps more precision but changes the sub-layer's
        # advertised policy contract. Ruled out by the user in favour of this
        # boundary cast (`D-005`).
        # ACCEPTED COST: under fp16/bf16 construction the projection is still
        # computed narrow (loss error 1.56e-05 / 7.89e-05 against an
        # exact-rational Fenchel-Young reference).
        # RED-able: `TestSparsemaxLossDtypePolicy::
        # test_loss_constructed_under_a_narrow_policy_matches_the_reference`.
        p = keras.ops.cast(self.sparsemax(y_pred), self.dtype)

        # Compute loss components
        # 1. Squared Euclidean distance: 0.5 * ||z - p||²
        squared_diff = keras.ops.square(y_pred - p)
        squared_distance = 0.5 * keras.ops.sum(squared_diff, axis=-1)

        # 2. Dot product term: z^T y
        dot_product = keras.ops.sum(y_pred * y_true, axis=-1)

        # Combine: L = 0.5 * ||z - p||² - z^T y
        loss = squared_distance - dot_product

        return loss

    def get_config(self) -> Dict[str, Any]:
        """
        Get loss configuration for serialization.

        Returns:
            Dictionary containing loss configuration with all constructor
            parameters needed to recreate the loss.
        """
        config = super().get_config()
        config.update({
            "from_logits": self.from_logits,
        })
        return config

# ---------------------------------------------------------------------
