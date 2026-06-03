"""Concrete CCNet task architectures.

Re-exports the per-task Explainer/Reasoner/Producer networks and their
factory functions, plus task-specific orchestrators (hybrid image, token-space text).
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
from .text import (
    SentimentExplainer,
    SentimentReasoner,
    SentimentProducer,
    ARSentimentProducer,
    TextCCNetOrchestrator,
    ARTextCCNetOrchestrator,
    create_text_ccnet,
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
    "SentimentExplainer",
    "SentimentReasoner",
    "SentimentProducer",
    "ARSentimentProducer",
    "TextCCNetOrchestrator",
    "ARTextCCNetOrchestrator",
    "create_text_ccnet",
]
