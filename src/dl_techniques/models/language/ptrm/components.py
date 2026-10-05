"""
PTRM components — re-exports TRM components since PTRM uses the same
recursive reasoning core. The PTRM innovation is in the inference procedure
(stochastic rollouts with Q-head selection), not in the model architecture.

References:
    - Sghaier et al., 2026. Probabilistic Tiny Recursive Model.
      (https://arxiv.org/abs/XXXX.XXXXX)
    - Jolicoeur-Martineau, 2025. Less is More: Recursive Reasoning with Tiny
      Networks. (https://arxiv.org/abs/2510.04871)
"""

from dl_techniques.models.language.trm.components import (
    TRMReasoningModule,
    TRMInner
)

__all__ = [
    'TRMReasoningModule',
    'TRMInner'
]