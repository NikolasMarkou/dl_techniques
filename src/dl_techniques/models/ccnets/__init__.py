# --- Framework: config, trainer, orchestrators, utils -----------------------
from .base import CCNetConfig
from .trainer import CCNetTrainer
from .utils import EarlyStoppingCallback, wrap_keras_model
from .orchestrators import CCNetOrchestrator, SequentialCCNetOrchestrator

# --- Shared building blocks --------------------------------------------------
from .blocks import FiLMLayer, ConvBlock, DenseBlock

# --- Concrete task architectures + factories ---------------------------------
from .architectures import (
    MNISTExplainer,
    MNISTReasoner,
    MNISTProducer,
    create_mnist_ccnet,
    Cifar100Explainer,
    Cifar100Reasoner,
    Cifar100Producer,
    create_cifar100_ccnet,
    HybridCCNetOrchestrator,
    SentimentExplainer,
    SentimentReasoner,
    SentimentProducer,
    ARSentimentProducer,
    TextCCNetOrchestrator,
    ARTextCCNetOrchestrator,
    create_text_ccnet,
)

__all__ = [
    # framework
    "CCNetConfig",
    "CCNetTrainer",
    "CCNetOrchestrator",
    "SequentialCCNetOrchestrator",
    "EarlyStoppingCallback",
    "wrap_keras_model",
    # blocks
    "FiLMLayer",
    "ConvBlock",
    "DenseBlock",
    # architectures + factories
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
