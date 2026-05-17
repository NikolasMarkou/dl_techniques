"""dl_techniques.training — model-agnostic training utilities.

Public API surface for Token Superposition Training (TST). See
``dl_techniques.training.token_superposition`` module docstring for the
canonical user pattern (two-``model.fit(...)`` invocation per D-007).
"""

from dl_techniques.training.token_superposition import (
    TSTConfig,
    TSTState,
    TSTEmbedding,
    TSTCausalLMLoss,
    TSTPhaseCallback,
    tst_phase1_transform,
    tst_phase2_transform,
    apply_tst,
)

__all__ = [
    "TSTConfig",
    "TSTState",
    "TSTEmbedding",
    "TSTCausalLMLoss",
    "TSTPhaseCallback",
    "tst_phase1_transform",
    "tst_phase2_transform",
    "apply_tst",
]
