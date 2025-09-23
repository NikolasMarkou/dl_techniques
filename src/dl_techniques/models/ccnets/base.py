import keras
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, List, Any


# ---------------------------------------------------------------------

class CCNetModule(Protocol):
    """
    Protocol defining the interface for CCNet modules.
    Any model that implements this protocol can be used in the framework.
    """

    def __call__(self, *args, **kwargs) -> keras.ops.array:
        """Forward pass of the module."""
        pass

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Get trainable variables of the module."""
        pass

    def save(self, filepath: str):
        """Save the model."""
        pass


# ---------------------------------------------------------------------

@dataclass
class CCNetConfig:
    """
    Configuration for CCNet framework.

    Args:
        explanation_dim: Dimension of the latent explanation vector E.
        loss_fn: Name of the loss function ('l1', 'l2', 'huber', or 'polynomial').
        loss_fn_params: Dictionary of parameters for the chosen loss function.
        learning_rates: Learning rates for each module.
        gradient_clip_norm: Maximum norm for gradient clipping (None to disable).
        use_mixed_precision: Whether to use mixed precision training.
        sequential_data: Whether the data is sequential (enables causal masking).
        explainer_weights: Weights for the explainer's additive error terms.
        reasoner_weights: Weights for the reasoner's additive error terms.
        producer_weights: Weights for the producer's additive error terms.
        dynamic_weighting: [DEPRECATED] If True, dynamically balances loss term weights.
                           Known to cause instability with the causal protocol. Keep False.
    """
    explanation_dim: int = 128
    loss_fn: str = 'l2'
    loss_fn_params: Dict[str, Any] = field(default_factory=dict)
    learning_rates: Dict[str, float] = field(default_factory=lambda: {
        'explainer': 1e-4,
        'reasoner': 1e-4,
        'producer': 1e-4
    })
    gradient_clip_norm: Optional[float] = 1.0
    use_mixed_precision: bool = False
    sequential_data: bool = False

    # CORRECTED: Weight structure aligned with the stable, additive doctrine.
    explainer_weights: Dict[str, float] = field(default_factory=lambda: {
        'inference': 1.0,
        'generation': 1.0,
        'kl_divergence': 0.01  # KL weight is now part of the explainer's responsibility.
    })
    reasoner_weights: Dict[str, float] = field(default_factory=lambda: {
        'reconstruction': 1.0,
        'inference': 1.0
    })
    producer_weights: Dict[str, float] = field(default_factory=lambda: {
        'generation': 1.0,
        'reconstruction': 1.0
    })
    dynamic_weighting: bool = False  # This feature conflicts with the causal protocol and should not be used.


# ---------------------------------------------------------------------

@dataclass
class CCNetLosses:
    """
    Container for the three fundamental CCNet losses.

    Attributes:
        generation_loss: ||X_generated - X_input||
        reconstruction_loss: ||X_reconstructed - X_input||
        inference_loss: ||X_reconstructed - X_generated||
    """
    generation_loss: keras.ops.array
    reconstruction_loss: keras.ops.array
    inference_loss: keras.ops.array

    def to_dict(self) -> Dict[str, float]:
        """Convert losses to dictionary of scalars."""
        return {
            'generation_loss': float(keras.ops.convert_to_numpy(self.generation_loss)),
            'reconstruction_loss': float(keras.ops.convert_to_numpy(self.reconstruction_loss)),
            'inference_loss': float(keras.ops.convert_to_numpy(self.inference_loss))
        }


# ---------------------------------------------------------------------

@dataclass
class CCNetModelErrors:
    """
    Container for the three model-specific error signals.

    Attributes:
        explainer_error: Weighted sum of inference, generation, and KL losses.
        reasoner_error: Weighted sum of reconstruction and inference losses.
        producer_error: Weighted sum of generation and reconstruction losses.
    """
    explainer_error: keras.ops.array
    reasoner_error: keras.ops.array
    producer_error: keras.ops.array

    def to_dict(self) -> Dict[str, float]:
        """Convert errors to dictionary of scalars."""
        return {
            'explainer_error': float(keras.ops.convert_to_numpy(self.explainer_error)),
            'reasoner_error': float(keras.ops.convert_to_numpy(self.reasoner_error)),
            'producer_error': float(keras.ops.convert_to_numpy(self.producer_error))
        }