"""CliffordNet and HierarchicalCliffordNet public API re-exports.

The four names are the two image classifiers and their factories; the stem, head-config and
argument helpers they share live in ``cliffordnet.model`` and are not re-exported.
"""
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

# 2026-10-27: the surface is the four names in `__all__` below; the "exactly these 2 names" in the
# first note above is the reduction the second note widened, kept as written.
__all__ = [
    "CliffordNet",
    "create_cliffordnet",
    "HierarchicalCliffordNet",
    "create_hierarchical_cliffordnet",
]
