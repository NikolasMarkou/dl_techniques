"""CliffordNet public API re-exports."""
from .model import CliffordNet, create_cliffordnet
from .hierarchical import HierarchicalCliffordNet, create_hierarchical_cliffordnet

# DECISION plan-2026-08-10T130454-3649c19e/D-006: surface is exactly these 2 names —
# do not re-export CliffordNetEmbedding/CliffordNetLMRouting, their source files are gone. See decisions.md.
# DECISION plan-2026-10-23T132722-77331e7a/D-002
# The 2-name surface above was a reduction of names whose source files were deleted. The two
# hierarchical names are real and are exported on purpose: the `pretrained=True` behavioural
# sweep in tests/test_models/test_package_api_contract.py finds factories by walking package
# exports, and its own comment calls an unexported factory a package defect. Do not move
# them back behind a waiver there. See decisions.md D-002.
__all__ = [
    "CliffordNet",
    "create_cliffordnet",
    "HierarchicalCliffordNet",
    "create_hierarchical_cliffordnet",
]
