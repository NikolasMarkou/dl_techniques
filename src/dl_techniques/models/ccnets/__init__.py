"""
CCNets (Causal Cooperative Networks) Framework

A revolutionary three-network architecture for explainable AI inspired by
the Three Kingdoms political philosophy. CCNets implements cooperative learning
through Explainer, Reasoner, and Producer networks that work together rather
than compete.

Main Components:
- ExplainerNetwork: Creates compressed explanations of input data
- ReasonerNetwork: Makes predictions based on input and explanations
- ProducerNetwork: Generates and reconstructs data based on labels and explanations
- CCNetsModel: Main orchestrator implementing cooperative learning
- Training utilities and metrics for monitoring cooperation

Key Features:
- Built-in explainability through explanation vectors
- Cross-verification through multiple validation paths
- Bidirectional inference (prediction and data generation)
- Cooperative rather than adversarial training
- Data simulation capabilities for practical applications

Usage Example:
    ```python
    from dl_techniques.models.ccnets import create_ccnets_model, train_ccnets_classifier_example

    # Create a CCNets model
    model = create_ccnets_model(
        input_dim=20,
        explanation_dim=10,
        output_dim=5
    )

    # Or run a complete example
    results = train_ccnets_classifier_example()
    ```

For detailed examples and tutorials, see:
- ccnets_examples.py: Complete usage examples
- ccnets_training.py: Training utilities and metrics
- ccnets_model.py: Core model implementation
"""

from typing import Dict, List, Optional, Tuple, Any

# Core model components
from .ccnets_base import (
    ExplainerNetwork,
    ReasonerNetwork,
    ProducerNetwork,
    create_explainer_network,
    create_reasoner_network,
    create_producer_network
)

from .ccnets_model import (
    CCNetsModel,
    CCNetsLoss,
    create_ccnets_model
)

# Training and evaluation utilities
from .ccnets_training import (
    CCNetsMetrics,
    CCNetsCallback,
    CCNetsDataGenerator,
    prepare_ccnets_data,
    train_ccnets_model,
    evaluate_ccnets_model,
    ccnets_inference,
    simulate_approval_scenario
)

# Examples and demonstrations
from .ccnets_examples import (
    create_synthetic_dataset,
    create_loan_approval_dataset,
    train_ccnets_classifier_example,
    train_ccnets_loan_approval_example,
    visualize_ccnets_cooperation,
    run_complete_ccnets_demo
)

# Version information
__version__ = "1.0.0"
__author__ = "DL Techniques Framework"
__description__ = "Causal Cooperative Networks for Explainable AI"

# Export main components
__all__ = [
    # Core Networks
    'ExplainerNetwork',
    'ReasonerNetwork',
    'ProducerNetwork',
    'create_explainer_network',
    'create_reasoner_network',
    'create_producer_network',

    # Main Model
    'CCNetsModel',
    'CCNetsLoss',
    'create_ccnets_model',

    # Training Utilities
    'CCNetsMetrics',
    'CCNetsCallback',
    'CCNetsDataGenerator',
    'prepare_ccnets_data',
    'train_ccnets_model',
    'evaluate_ccnets_model',
    'ccnets_inference',
    'simulate_approval_scenario',

    # Examples and Demos
    'create_synthetic_dataset',
    'create_loan_approval_dataset',
    'train_ccnets_classifier_example',
    'train_ccnets_loan_approval_example',
    'visualize_ccnets_cooperation',
    'run_complete_ccnets_demo',

    # Configuration
    'CCNetsConfig',
    'create_default_config',
    'create_classification_config',
    'create_regression_config',
    'create_loan_approval_config'
]


# Configuration utilities
class CCNetsConfig:
    """
    Configuration class for CCNets models.

    Provides structured configuration for different types of problems
    and helps ensure consistent model setup across applications.
    """

    def __init__(
            self,
            input_dim: int,
            explanation_dim: int,
            output_dim: int,
            explainer_config: Optional[Dict] = None,
            reasoner_config: Optional[Dict] = None,
            producer_config: Optional[Dict] = None,
            training_config: Optional[Dict] = None,
            loss_weights: Optional[List[float]] = None
    ):
        self.input_dim = input_dim
        self.explanation_dim = explanation_dim
        self.output_dim = output_dim

        # Network configurations
        self.explainer_config = explainer_config or self._default_explainer_config()
        self.reasoner_config = reasoner_config or self._default_reasoner_config()
        self.producer_config = producer_config or self._default_producer_config()

        # Training configuration
        self.training_config = training_config or self._default_training_config()

        # Loss weights [inference, generation, reconstruction]
        self.loss_weights = loss_weights or [1.0, 1.0, 1.0]

    def _default_explainer_config(self) -> Dict:
        """Default configuration for ExplainerNetwork."""
        return {
            'hidden_dims': [min(512, self.input_dim * 2), min(256, self.input_dim)],
            'activation': 'relu',
            'dropout_rate': 0.3,
            'kernel_initializer': 'glorot_uniform'
        }

    def _default_reasoner_config(self) -> Dict:
        """Default configuration for ReasonerNetwork."""
        return {
            'hidden_dims': [min(512, (self.input_dim + self.explanation_dim) * 2),
                            min(256, self.input_dim + self.explanation_dim)],
            'activation': 'relu',
            'output_activation': 'softmax' if self.output_dim > 1 else 'sigmoid',
            'dropout_rate': 0.3,
            'kernel_initializer': 'glorot_uniform'
        }

    def _default_producer_config(self) -> Dict:
        """Default configuration for ProducerNetwork."""
        return {
            'hidden_dims': [min(512, (self.output_dim + self.explanation_dim) * 2),
                            min(256, self.output_dim + self.explanation_dim)],
            'activation': 'relu',
            'output_activation': 'sigmoid',
            'dropout_rate': 0.3,
            'kernel_initializer': 'glorot_uniform'
        }

    def _default_training_config(self) -> Dict:
        """Default training configuration."""
        return {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001,
            'validation_split': 0.2,
            'patience': 20,
            'verbose': 1
        }

    def create_model(self) -> CCNetsModel:
        """Create a CCNets model from this configuration."""
        return create_ccnets_model(
            input_dim=self.input_dim,
            explanation_dim=self.explanation_dim,
            output_dim=self.output_dim,
            explainer_kwargs=self.explainer_config,
            reasoner_kwargs=self.reasoner_config,
            producer_kwargs=self.producer_config,
            loss_weights=self.loss_weights
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'input_dim': self.input_dim,
            'explanation_dim': self.explanation_dim,
            'output_dim': self.output_dim,
            'explainer_config': self.explainer_config,
            'reasoner_config': self.reasoner_config,
            'producer_config': self.producer_config,
            'training_config': self.training_config,
            'loss_weights': self.loss_weights
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CCNetsConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


def create_default_config(
        input_dim: int,
        explanation_dim: int,
        output_dim: int,
        **kwargs
) -> CCNetsConfig:
    """
    Create a default CCNets configuration.

    Args:
        input_dim: Dimensionality of input data
        explanation_dim: Dimensionality of explanation vectors
        output_dim: Dimensionality of outputs
        **kwargs: Additional configuration parameters

    Returns:
        CCNetsConfig instance with default settings
    """
    return CCNetsConfig(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        output_dim=output_dim,
        **kwargs
    )


def create_classification_config(
        input_dim: int,
        n_classes: int,
        explanation_dim: Optional[int] = None,
        **kwargs
) -> CCNetsConfig:
    """
    Create a CCNets configuration optimized for classification.

    Args:
        input_dim: Dimensionality of input features
        n_classes: Number of classes
        explanation_dim: Explanation vector dimension (auto-computed if None)
        **kwargs: Additional configuration parameters

    Returns:
        CCNetsConfig optimized for classification
    """
    if explanation_dim is None:
        explanation_dim = max(8, min(input_dim // 2, n_classes * 2))

    # Classification-specific configurations
    explainer_config = {
        'hidden_dims': [min(256, input_dim * 2), min(128, input_dim)],
        'activation': 'relu',
        'dropout_rate': 0.2,
        'kernel_initializer': 'he_normal'
    }

    reasoner_config = {
        'hidden_dims': [min(256, input_dim + explanation_dim), min(128, explanation_dim * 2)],
        'activation': 'relu',
        'output_activation': 'softmax',
        'dropout_rate': 0.2,
        'kernel_initializer': 'he_normal'
    }

    producer_config = {
        'hidden_dims': [min(256, n_classes + explanation_dim), min(128, explanation_dim * 2)],
        'activation': 'relu',
        'output_activation': 'sigmoid',
        'dropout_rate': 0.2,
        'kernel_initializer': 'he_normal'
    }

    training_config = {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'patience': 15,
        'verbose': 1
    }

    return CCNetsConfig(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        output_dim=n_classes,
        explainer_config=explainer_config,
        reasoner_config=reasoner_config,
        producer_config=producer_config,
        training_config=training_config,
        loss_weights=[1.0, 1.0, 1.0],
        **kwargs
    )


def create_regression_config(
        input_dim: int,
        output_dim: int = 1,
        explanation_dim: Optional[int] = None,
        **kwargs
) -> CCNetsConfig:
    """
    Create a CCNets configuration optimized for regression.

    Args:
        input_dim: Dimensionality of input features
        output_dim: Dimensionality of output targets
        explanation_dim: Explanation vector dimension (auto-computed if None)
        **kwargs: Additional configuration parameters

    Returns:
        CCNetsConfig optimized for regression
    """
    if explanation_dim is None:
        explanation_dim = max(8, min(input_dim // 3, 32))

    # Regression-specific configurations
    explainer_config = {
        'hidden_dims': [min(128, input_dim * 2), min(64, input_dim)],
        'activation': 'relu',
        'dropout_rate': 0.1,
        'kernel_initializer': 'glorot_normal'
    }

    reasoner_config = {
        'hidden_dims': [min(128, input_dim + explanation_dim), min(64, explanation_dim * 2)],
        'activation': 'relu',
        'output_activation': 'linear',
        'dropout_rate': 0.1,
        'kernel_initializer': 'glorot_normal'
    }

    producer_config = {
        'hidden_dims': [min(128, output_dim + explanation_dim), min(64, explanation_dim * 2)],
        'activation': 'relu',
        'output_activation': 'linear',
        'dropout_rate': 0.1,
        'kernel_initializer': 'glorot_normal'
    }

    training_config = {
        'batch_size': 32,
        'epochs': 150,
        'learning_rate': 0.0005,
        'validation_split': 0.2,
        'patience': 25,
        'verbose': 1
    }

    return CCNetsConfig(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        output_dim=output_dim,
        explainer_config=explainer_config,
        reasoner_config=reasoner_config,
        producer_config=producer_config,
        training_config=training_config,
        loss_weights=[1.0, 1.5, 1.0],  # Emphasize generation for regression
        **kwargs
    )


def create_loan_approval_config(
        input_dim: int = 6,
        explanation_dim: int = 8,
        **kwargs
) -> CCNetsConfig:
    """
    Create a CCNets configuration optimized for loan approval scenarios.

    This configuration is specifically designed for financial applications
    where data simulation and explainability are crucial.

    Args:
        input_dim: Number of financial features (default: 6)
        explanation_dim: Explanation vector dimension
        **kwargs: Additional configuration parameters

    Returns:
        CCNetsConfig optimized for loan approval
    """
    # Loan approval specific configurations
    explainer_config = {
        'hidden_dims': [32, 16],
        'activation': 'relu',
        'dropout_rate': 0.3,
        'kernel_initializer': 'glorot_uniform'
    }

    reasoner_config = {
        'hidden_dims': [32, 16],
        'activation': 'relu',
        'output_activation': 'softmax',
        'dropout_rate': 0.3,
        'kernel_initializer': 'glorot_uniform'
    }

    producer_config = {
        'hidden_dims': [32, 16],
        'activation': 'relu',
        'output_activation': 'sigmoid',
        'dropout_rate': 0.3,
        'kernel_initializer': 'glorot_uniform'
    }

    training_config = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0005,
        'validation_split': 0.2,
        'patience': 20,
        'verbose': 1
    }

    return CCNetsConfig(
        input_dim=input_dim,
        explanation_dim=explanation_dim,
        output_dim=2,  # Binary approval/rejection
        explainer_config=explainer_config,
        reasoner_config=reasoner_config,
        producer_config=producer_config,
        training_config=training_config,
        loss_weights=[1.2, 1.0, 1.0],  # Emphasize inference for consistency
        **kwargs
    )


# Quick start function
def quick_start_example(dataset_type: str = "classification") -> Dict[str, Any]:
    """
    Run a quick start example to demonstrate CCNets capabilities.

    Args:
        dataset_type: Type of example to run ("classification", "loan_approval", or "both")

    Returns:
        Dictionary containing results from the example(s)
    """
    from dl_techniques.utils.logger import logger

    logger.info(f"Running CCNets quick start example: {dataset_type}")

    if dataset_type == "classification":
        return {"classification": train_ccnets_classifier_example()}
    elif dataset_type == "loan_approval":
        return {"loan_approval": train_ccnets_loan_approval_example()}
    elif dataset_type == "both":
        return run_complete_ccnets_demo()
    else:
        raise ValueError("dataset_type must be 'classification', 'loan_approval', or 'both'")


# Add configuration to exports
__all__.extend([
    'CCNetsConfig',
    'create_default_config',
    'create_classification_config',
    'create_regression_config',
    'create_loan_approval_config',
    'quick_start_example'
])