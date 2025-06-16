"""
GoodhartAwareLoss: Information-Theoretic Loss Function for Robust Classification

This module implements a comprehensive loss function that addresses Goodhart's Law
("When a measure becomes a target, it ceases to be a good measure") in machine
learning classification tasks through information-theoretic principles.

Goodhart's Law manifests in ML when models optimize for proxy metrics (like accuracy
or cross-entropy) in ways that don't improve the true underlying objective. This can
lead to overconfident predictions, exploitation of spurious correlations, and poor
generalization to out-of-distribution data.

The GoodhartAwareLoss combines three information-theoretic mechanisms:

1. **Temperature Scaling** (T = 1.5-3.0, default: 2.0): Prevents overconfident
   predictions by increasing output entropy through logit scaling. This addresses the
   common failure mode where models become artificially certain to minimize cross-entropy loss.

   **Common Values & Rationale:**
   - T=1.5-2.0: Light calibration for well-behaved datasets (CIFAR-10, MNIST)
   - T=2.0-3.0: Standard calibration for complex datasets (ImageNet, NLP tasks)
   - T=3.0-5.0: Heavy calibration for noisy labels or small datasets
   - Higher T values increase uncertainty but may hurt accuracy if too extreme

2. **Entropy Regularization** (λ = 0.01-0.2, default: 0.1): Explicitly encourages
   prediction uncertainty by maximizing the entropy H(p) = -Σ p_i log p_i of output
   distributions. This prevents the model from collapsing to overly confident solutions.

   **Common Values & Rationale:**
   - λ=0.01-0.05: Light regularization for large, clean datasets (>100k samples)
   - λ=0.05-0.15: Standard regularization for typical supervised learning
   - λ=0.15-0.3: Heavy regularization for small/noisy datasets (<10k samples)
   - Higher λ maintains more uncertainty but may reduce task performance
   - Lower λ allows more confidence but risks overfitting to spurious patterns

3. **Mutual Information Regularization** (β = 0.001-0.05, default: 0.01): Constrains
   the mutual information I(X;Y) between inputs and predictions to prevent memorization
   of spurious patterns while preserving task-relevant information.

   **Common Values & Rationale:**
   - β=0.001-0.005: Light compression for high-capacity models (ResNets, Transformers)
   - β=0.005-0.02: Standard compression for balanced generalization
   - β=0.02-0.1: Heavy compression for preventing overfitting on small datasets
   - Higher β forces more compression but may lose important information
   - Lower β allows more information flow but risks memorizing dataset artifacts

Mathematical Foundation:
- Temperature Scaling: p_i = exp(z_i/T) / Σ exp(z_j/T), where T > 1
- Entropy: H(p) = -Σ p_i log p_i
- Mutual Information: I(X;Y) ≈ H(Y) - H(Y|X)

Usage:
    loss_fn = GoodhartAwareLoss(
        temperature=2.0,
        entropy_weight=0.1,
        mi_weight=0.01
    )
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

References:
- Goodhart's Law: https://en.wikipedia.org/wiki/Goodhart's_law
- Information Bottleneck: Tishby et al. (2000)
- Temperature Scaling: Guo et al. (2017)
- MINE: Belghazi et al. (2018)
"""

import keras
import warnings
from keras import ops
from typing import Dict, Any, Tuple, Union


# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GoodhartAwareLoss(keras.losses.Loss):
    """
    Information-theoretic loss function that resists Goodhart's Law effects.

    This loss function combines multiple information-theoretic principles to create
    robust classification objectives that don't suffer from metric gaming:

    1. Temperature scaling for calibrated confidence
    2. Entropy regularization for appropriate uncertainty
    3. Mutual information constraints for robust representations

    Args:
        temperature: Float > 1.0. Temperature parameter for scaling logits.
            Higher values increase entropy and reduce overconfidence.
            Typical range: [1.1, 5.0]. Default: 2.0
        entropy_weight: Float >= 0. Weight for entropy regularization term.
            Controls how much the model is encouraged to maintain uncertainty.
            Typical range: [0.001, 0.5]. Default: 0.1
        mi_weight: Float >= 0. Weight for mutual information regularization.
            Controls compression of input-output information flow.
            Typical range: [0.001, 0.1]. Default: 0.01
        epsilon: Float > 0. Small constant for numerical stability in log operations.
            Should be much smaller than 1/num_classes. Default: 1e-8
        name: String name for the loss function.
            Default: 'goodhart_aware_loss'
        reduction: Type of reduction to apply to loss.
            Default: 'sum_over_batch_size'

    Raises:
        ValueError: If any parameter is outside valid range.

    Example:
        >>> # Basic usage
        >>> loss_fn = GoodhartAwareLoss(temperature=1.5, entropy_weight=0.05)
        >>>
        >>> # With model
        >>> model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        >>>
        >>> # Manual computation
        >>> y_true = ops.one_hot(ops.array([0, 1]), 3)  # shape: (2, 3)
        >>> y_pred = ops.array([[2.0, 1.0, 0.5], [1.5, 3.0, 0.8]])  # logits
        >>> loss = loss_fn(y_true, y_pred)
    """

    def __init__(
            self,
            temperature: float = 2.0,
            entropy_weight: float = 0.1,
            mi_weight: float = 0.01,
            epsilon: float = 1e-8,
            name: str = 'goodhart_aware_loss',
            reduction: str = 'sum_over_batch_size'
    ) -> None:
        super().__init__(name=name, reduction=reduction)

        # Enhanced parameter validation with helpful error messages
        if not isinstance(temperature, (int, float)) or temperature <= 1.0:
            raise ValueError(
                f"Temperature must be a number > 1.0 for entropy increase, "
                f"got {temperature}. Typical range: [1.1, 5.0]"
            )
        if not isinstance(entropy_weight, (int, float)) or entropy_weight < 0:
            raise ValueError(
                f"Entropy weight must be a non-negative number, "
                f"got {entropy_weight}. Typical range: [0.001, 0.5]"
            )
        if not isinstance(mi_weight, (int, float)) or mi_weight < 0:
            raise ValueError(
                f"MI weight must be a non-negative number, "
                f"got {mi_weight}. Typical range: [0.001, 0.1]"
            )
        if not isinstance(epsilon, (int, float)) or epsilon <= 0 or epsilon >= 0.1:
            raise ValueError(
                f"Epsilon must be a small positive number (0, 0.1), "
                f"got {epsilon}. Should be much smaller than 1/num_classes"
            )

        # Warn about potentially problematic parameter combinations
        if temperature > 10.0:
            warnings.warn(
                f"Very high temperature ({temperature}) may lead to overly uniform "
                f"predictions. Consider values in [1.1, 5.0]",
                UserWarning
            )
        if entropy_weight > 1.0:
            warnings.warn(
                f"Very high entropy weight ({entropy_weight}) may dominate training. "
                f"Consider values in [0.001, 0.5]",
                UserWarning
            )

        self.temperature = float(temperature)
        self.entropy_weight = float(entropy_weight)
        self.mi_weight = float(mi_weight)
        self.epsilon = float(epsilon)

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute the Goodhart-aware loss combining all three components.

        The total loss is:
        L_total = L_ce_temp + λ_entropy * L_entropy + λ_mi * L_mi

        Where:
        - L_ce_temp: Temperature-scaled cross-entropy
        - L_entropy: Negative entropy regularization
        - L_mi: Mutual information regularization

        Args:
            y_true: Ground truth labels, shape (batch_size, num_classes).
                Should be one-hot encoded or probability distributions.
            y_pred: Predicted logits, shape (batch_size, num_classes).
                Raw logits (not probabilities).

        Returns:
            Scalar tensor representing the total loss.

        Raises:
            ValueError: If input shapes are incompatible or contain invalid values.
        """
        # Input validation
        y_true = ops.cast(y_true, dtype=y_pred.dtype)

        # Check for NaN or Inf in inputs
        if ops.any(ops.logical_not(ops.isfinite(y_pred))):
            raise ValueError("y_pred contains NaN or Inf values")
        if ops.any(ops.logical_not(ops.isfinite(y_true))):
            raise ValueError("y_true contains NaN or Inf values")

        # Component 1: Temperature-scaled cross-entropy loss
        ce_loss = self._temperature_scaled_cross_entropy(y_true, y_pred)

        # Component 2: Entropy regularization loss (only if weight > 0)
        if self.entropy_weight > 0:
            entropy_loss = self._entropy_regularization(y_pred)
        else:
            entropy_loss = ops.cast(0.0, dtype=y_pred.dtype)

        # Component 3: Mutual information regularization loss (only if weight > 0)
        if self.mi_weight > 0:
            mi_loss = self._mutual_information_regularization(y_true, y_pred)
        else:
            mi_loss = ops.cast(0.0, dtype=y_pred.dtype)

        # Combine all components
        total_loss = (ce_loss +
                      self.entropy_weight * entropy_loss +
                      self.mi_weight * mi_loss)

        return total_loss

    def _temperature_scaled_cross_entropy(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute temperature-scaled cross-entropy loss.

        Temperature scaling addresses Goodhart's Law by preventing overconfident
        predictions. The standard softmax function often produces overly peaked
        distributions to minimize cross-entropy, leading to poor calibration.

        Mathematical Formula:
        p_i = exp(z_i / T) / Σ_j exp(z_j / T)
        L_ce = -Σ_i y_true_i * log(p_i)

        Where:
        - z_i: Raw logit for class i
        - T: Temperature parameter (T > 1 increases entropy)
        - p_i: Temperature-scaled probability for class i

        Why it helps with Goodhart's Law:
        - Higher temperature (T > 1) increases output entropy
        - Prevents the model from becoming artificially overconfident
        - Improves calibration: P(correct | confidence) ≈ confidence
        - Reduces the gap between training loss optimization and true performance

        Implementation Details:
        - Uses numerically stable softmax computation
        - Clips probabilities to prevent log(0)
        - Maintains gradient flow through temperature scaling

        Args:
            y_true: True labels, shape (batch_size, num_classes)
            y_pred: Predicted logits, shape (batch_size, num_classes)

        Returns:
            Temperature-scaled cross-entropy loss (scalar)
        """
        # Scale logits by temperature: z_scaled = z / T
        # This increases entropy when T > 1, making predictions less confident
        scaled_logits = y_pred / self.temperature

        # Compute temperature-scaled probabilities using numerically stable softmax
        # p_i = exp(z_i/T) / Σ exp(z_j/T)
        scaled_probs = ops.softmax(scaled_logits, axis=-1)

        # Clip probabilities for numerical stability in log computation
        # Prevents log(0) which would give -inf
        scaled_probs = ops.clip(scaled_probs, self.epsilon, 1.0 - self.epsilon)

        # Compute cross-entropy: L_ce = -Σ y_true * log(p_scaled)
        # This is the standard categorical cross-entropy but with temperature scaling
        cross_entropy_per_sample = -ops.sum(y_true * ops.log(scaled_probs), axis=-1)

        # Return mean across batch
        return ops.mean(cross_entropy_per_sample)

    def _entropy_regularization(self, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute entropy regularization loss.

        Entropy regularization directly addresses Goodhart's Law by encouraging
        the model to maintain appropriate uncertainty rather than becoming
        overconfident to minimize the primary loss function.

        Mathematical Formula:
        H(p) = -Σ_i p_i * log(p_i)
        L_entropy = -H(p)  [negative because we want to maximize entropy]

        Where:
        - p_i: Predicted probability for class i
        - H(p): Shannon entropy of the probability distribution

        Why it helps with Goodhart's Law:
        - Prevents collapse to overconfident (low-entropy) solutions
        - Encourages the model to "know what it doesn't know"
        - Balances between fitting the data and maintaining uncertainty
        - Reduces overfitting to spurious patterns by preventing memorization

        The regularization acts as a "pressure valve" against overoptimization:
        when the model tries to become too confident to minimize cross-entropy,
        the entropy term pushes back, maintaining calibrated uncertainty.

        Implementation Details:
        - Uses standard temperature (T=1) for entropy computation
        - Clips probabilities to prevent numerical instability
        - Returns negative entropy to encourage high entropy (uncertainty)

        Args:
            y_pred: Predicted logits, shape (batch_size, num_classes)

        Returns:
            Negative entropy loss (scalar). Positive values penalize low entropy.
        """
        # Convert logits to probabilities using standard softmax (T=1)
        # p_i = exp(z_i) / Σ exp(z_j)
        probs = ops.softmax(y_pred, axis=-1)

        # Clip probabilities for numerical stability
        # Prevents log(0) and ensures probabilities sum to ~1
        probs = ops.clip(probs, self.epsilon, 1.0 - self.epsilon)

        # Compute Shannon entropy for each sample: H(p) = -Σ p_i * log(p_i)
        # High entropy = uncertain/uniform predictions
        # Low entropy = confident/peaked predictions
        entropy_per_sample = -ops.sum(probs * ops.log(probs), axis=-1)

        # Return negative entropy (we want to maximize entropy, so minimize -entropy)
        # This encourages the model to maintain uncertainty
        negative_entropy = -ops.mean(entropy_per_sample)

        return negative_entropy

    def _mutual_information_regularization(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Compute mutual information regularization based on information bottleneck principle.

        This component addresses Goodhart's Law by constraining how much information
        flows from inputs to predictions, preventing the model from exploiting
        spurious correlations and encouraging robust, generalizable representations.

        Mathematical Formula:
        I(X;Y) ≈ H(Y) - H(Y|X)

        Where:
        - I(X;Y): Mutual information between inputs X and outputs Y
        - H(Y): Marginal entropy of outputs (diversity of predictions across batch)
        - H(Y|X): Conditional entropy (average uncertainty of predictions)

        Batch-based Approximation:
        - H(Y) ≈ entropy of mean prediction across batch
        - H(Y|X) ≈ mean entropy of individual predictions

        Why it helps with Goodhart's Law:
        - Information bottleneck principle: compress irrelevant information
        - Prevents memorization of spurious input-output correlations
        - Encourages learning of robust, generalizable features
        - Acts as implicit regularization against overfitting

        The information bottleneck creates a "compression bottleneck" that forces
        the model to discard irrelevant information while preserving task-relevant
        patterns. This naturally resists Goodhart's Law by preventing exploitation
        of dataset-specific artifacts.

        Implementation Details:
        - Uses batch-based MI approximation for computational efficiency
        - Only penalizes positive MI (negative MI indicates underfitting)
        - Balances information compression with task performance

        Args:
            y_true: True labels, shape (batch_size, num_classes)
            y_pred: Predicted logits, shape (batch_size, num_classes)

        Returns:
            Mutual information regularization loss (scalar)
        """
        # Convert logits to probabilities
        probs = ops.softmax(y_pred, axis=-1)
        probs = ops.clip(probs, self.epsilon, 1.0 - self.epsilon)

        # Compute H(Y|X): conditional entropy (mean entropy of individual predictions)
        # This measures average uncertainty of the model's predictions
        # Higher values = model is more uncertain about individual predictions
        conditional_entropy_per_sample = -ops.sum(probs * ops.log(probs), axis=-1)
        h_y_given_x = ops.mean(conditional_entropy_per_sample)

        # Compute H(Y): marginal entropy (entropy of average prediction across batch)
        # This measures diversity of predictions across the batch
        # Higher values = model produces diverse predictions across different inputs
        mean_probs = ops.mean(probs, axis=0)  # Average across batch dimension
        mean_probs = ops.clip(mean_probs, self.epsilon, 1.0 - self.epsilon)

        # Ensure mean_probs sums to 1 for valid probability distribution
        mean_probs = mean_probs / ops.sum(mean_probs)
        h_y = -ops.sum(mean_probs * ops.log(mean_probs))

        # Approximate mutual information: I(X;Y) ≈ H(Y) - H(Y|X)
        # High MI: inputs strongly determine outputs (potential overfitting)
        # Low MI: inputs weakly determine outputs (potential underfitting)
        # Negative MI: indicates numerical issues or underfitting
        mutual_information = h_y - h_y_given_x

        # Only regularize positive MI to prevent overfitting
        # Negative MI typically indicates underfitting, which we don't want to penalize
        mi_regularization = ops.maximum(mutual_information, 0.0)

        return mi_regularization

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for serialization.

        Returns:
            Dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'entropy_weight': self.entropy_weight,
            'mi_weight': self.mi_weight,
            'epsilon': self.epsilon
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GoodhartAwareLoss':
        """
        Create instance from configuration dictionary.

        Args:
            config: Configuration dictionary from get_config()

        Returns:
            New GoodhartAwareLoss instance
        """
        return cls(**config)

# ---------------------------------------------------------------------


def analyze_loss_components(
        loss_fn: GoodhartAwareLoss,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
) -> Dict[str, float]:
    """
    Analyze individual components of the GoodhartAwareLoss.

    This utility function breaks down the loss into its constituent parts
    for debugging and hyperparameter tuning purposes.

    Args:
        loss_fn: GoodhartAwareLoss instance
        y_true: True labels, shape (batch_size, num_classes)
        y_pred: Predicted logits, shape (batch_size, num_classes)

    Returns:
        Dictionary with individual loss components and their contributions

    Example:
        >>> loss_fn = GoodhartAwareLoss(temperature=2.0, entropy_weight=0.1)
        >>> analysis = analyze_loss_components(loss_fn, y_true, y_pred)
        >>> logger.info(f"Cross-entropy: {analysis['cross_entropy']:.4f}")
        >>> logger.info(f"Entropy loss: {analysis['entropy_loss']:.4f}")
    """
    # Compute individual components
    ce_loss = loss_fn._temperature_scaled_cross_entropy(y_true, y_pred)
    entropy_loss = loss_fn._entropy_regularization(y_pred)
    mi_loss = loss_fn._mutual_information_regularization(y_true, y_pred)

    # Compute weighted contributions
    weighted_entropy = loss_fn.entropy_weight * entropy_loss
    weighted_mi = loss_fn.mi_weight * mi_loss
    total_loss = ce_loss + weighted_entropy + weighted_mi

    # Convert tensors to Python floats for easy inspection
    return {
        'total_loss': float(ops.convert_to_numpy(total_loss)),
        'cross_entropy': float(ops.convert_to_numpy(ce_loss)),
        'entropy_loss': float(ops.convert_to_numpy(entropy_loss)),
        'mi_loss': float(ops.convert_to_numpy(mi_loss)),
        'weighted_entropy': float(ops.convert_to_numpy(weighted_entropy)),
        'weighted_mi': float(ops.convert_to_numpy(weighted_mi)),
        'entropy_contribution_pct': float(ops.convert_to_numpy(weighted_entropy / total_loss * 100)),
        'mi_contribution_pct': float(ops.convert_to_numpy(weighted_mi / total_loss * 100)),
        'ce_contribution_pct': float(ops.convert_to_numpy(ce_loss / total_loss * 100)),
        'entropy_weight': loss_fn.entropy_weight,
        'mi_weight': loss_fn.mi_weight,
        'temperature': loss_fn.temperature
    }

# ---------------------------------------------------------------------


def suggest_hyperparameters(
        num_classes: int,
        dataset_size: int,
        model_complexity: str = 'medium'
) -> Dict[str, float]:
    """
    Suggest hyperparameters based on problem characteristics.

    Args:
        num_classes: Number of output classes
        dataset_size: Size of training dataset
        model_complexity: 'simple', 'medium', or 'complex'

    Returns:
        Dictionary with suggested hyperparameters

    Example:
        >>> params = suggest_hyperparameters(num_classes=10, dataset_size=50000)
        >>> loss_fn = GoodhartAwareLoss(**params)
    """
    # Base parameters
    base_params = {
        'temperature': 2.0,
        'entropy_weight': 0.1,
        'mi_weight': 0.01
    }

    # Adjust based on number of classes
    if num_classes <= 5:
        base_params['temperature'] = 1.5
        base_params['entropy_weight'] = 0.05
    elif num_classes >= 100:
        base_params['temperature'] = 3.0
        base_params['entropy_weight'] = 0.2

    # Adjust based on dataset size
    if dataset_size < 1000:
        # Small dataset: more regularization
        base_params['entropy_weight'] *= 2.0
        base_params['mi_weight'] *= 2.0
    elif dataset_size > 100000:
        # Large dataset: less regularization
        base_params['entropy_weight'] *= 0.5
        base_params['mi_weight'] *= 0.5

    # Adjust based on model complexity
    complexity_multipliers = {
        'simple': {'entropy_weight': 0.5, 'mi_weight': 0.5},
        'medium': {'entropy_weight': 1.0, 'mi_weight': 1.0},
        'complex': {'entropy_weight': 1.5, 'mi_weight': 1.5}
    }

    if model_complexity in complexity_multipliers:
        multipliers = complexity_multipliers[model_complexity]
        base_params['entropy_weight'] *= multipliers['entropy_weight']
        base_params['mi_weight'] *= multipliers['mi_weight']

    return base_params

# ---------------------------------------------------------------------
