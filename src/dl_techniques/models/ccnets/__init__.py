__version__ = "1.0.0"
__author__ = "Nikolas Markou"
__description__ = "Causal Cooperative Networks for Explainable AI"

from .base import (
    ExplainerNetwork,
    ReasonerNetwork,
    ProducerNetwork
)

from .models import (
    CCNetsModel,
    CCNetsLoss,
    create_ccnets_model
)

# Examples and demonstrations
from .examples import (
    create_synthetic_dataset,
)

__all__ = [
    # Core Networks
    'ExplainerNetwork',
    'ReasonerNetwork',
    'ProducerNetwork',

    # Main Model
    'CCNetsModel',
    'CCNetsLoss',
    'create_ccnets_model',
]