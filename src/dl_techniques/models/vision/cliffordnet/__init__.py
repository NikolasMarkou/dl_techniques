"""CliffordNet public API re-exports."""
from .model import CliffordNet, create_cliffordnet

# DECISION plan-2026-08-10T130454-3649c19e/D-006: surface is exactly these 2 names —
# do not re-export CliffordNetEmbedding/CliffordNetLMRouting, their source files are gone. See decisions.md.
__all__ = [
    "CliffordNet",
    "create_cliffordnet",
]
