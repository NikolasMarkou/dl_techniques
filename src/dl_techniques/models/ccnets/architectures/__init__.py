"""Concrete CCNet task architectures.

Re-exports the per-task Explainer/Reasoner/Producer networks and their
factory functions. Text architectures are added in a later migration step.
"""

from .mnist import (
    MNISTExplainer,
    MNISTReasoner,
    MNISTProducer,
    create_mnist_ccnet,
)
from .cifar100 import (
    Cifar100Explainer,
    Cifar100Reasoner,
    Cifar100Producer,
    create_cifar100_ccnet,
    HybridCCNetOrchestrator,
)

__all__ = [
    "MNISTExplainer",
    "MNISTReasoner",
    "MNISTProducer",
    "create_mnist_ccnet",
    "Cifar100Explainer",
    "Cifar100Reasoner",
    "Cifar100Producer",
    "create_cifar100_ccnet",
    "HybridCCNetOrchestrator",
]
