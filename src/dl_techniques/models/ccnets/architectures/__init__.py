"""Concrete CCNet task architectures.

Re-exports the per-task Explainer/Reasoner/Producer networks and their
factory functions. CIFAR-100 and text architectures are added in later
migration steps.
"""

from .mnist import (
    MNISTExplainer,
    MNISTReasoner,
    MNISTProducer,
    create_mnist_ccnet,
)

__all__ = [
    "MNISTExplainer",
    "MNISTReasoner",
    "MNISTProducer",
    "create_mnist_ccnet",
]
