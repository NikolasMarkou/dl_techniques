import keras
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, List

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

# ---------------------------------------------------------------------

@dataclass
class CCNetConfig:
    """
    Configuration for CCNet framework.

    Args:
        explanation_dim: Dimension of the latent explanation vector E.
        loss_type: Type of loss function ('l1', 'l2', or 'huber').
        learning_rates: Learning rates for each module.
        gradient_clip_norm: Maximum norm for gradient clipping (None to disable).
        use_mixed_precision: Whether to use mixed precision training.
        sequential_data: Whether the data is sequential (enables causal masking).
        verification_weight: Weight for verification losses vs inference losses.
    """
    explanation_dim: int = 128
    loss_type: str = 'l2'
    learning_rates: Dict[str, float] = field(default_factory=lambda: {
        'explainer': 1e-3,
        'reasoner': 1e-3,
        'producer': 1e-3
    })
    gradient_clip_norm: Optional[float] = 1.0
    use_mixed_precision: bool = False
    sequential_data: bool = False
    verification_weight: float = 1.0

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
        explainer_error: Inference + Generation - Reconstruction
        reasoner_error: Reconstruction + Inference - Generation
        producer_error: Generation + Reconstruction - Inference
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

# ---------------------------------------------------------------------
