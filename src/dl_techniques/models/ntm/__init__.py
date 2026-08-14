"""Neural Turing Machine — public API re-exports."""
from .model import NTMModel, create_ntm_variant
from .model_multitask import NTMMultiTask

__all__ = [
    "NTMModel",
    "NTMMultiTask",
    "create_ntm_variant",
]
