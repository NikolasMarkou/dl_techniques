"""
Probabilistic Tiny Recursive Model (PTRM)

A Tiny Recursive Model (TRM) equipped with a stochastic inference procedure
for test-time compute scaling. PTRM runs K parallel stochastic rollouts by
injecting Gaussian noise into the reasoning latent ``z_L`` at each supervision
step, and selects the final answer by the Q-head's halting logit ``q_halt``
(arXiv:2605.19943, Fig. 4 left).

References:
    - Sghaier et al., 2026. Probabilistic Tiny Recursive Model.
      (https://arxiv.org/abs/2605.19943)
    - Jolicoeur-Martineau, 2025. Less is More: Recursive Reasoning with Tiny
      Networks. (https://arxiv.org/abs/2510.04871)
"""

from .components import TRMReasoningModule, TRMInner
from .model import PTRM, create_ptrm
from .inference import PTRMInference, run_ptrm_inference
from .factory import (
    get_sudoku_extreme_config,
    get_ppbench_config,
    get_maze_hard_config,
    get_arc_agi_config,
    PRESET_CONFIGS,
)

__all__ = [
    # Core components (re-exported from TRM)
    'TRMReasoningModule',
    'TRMInner',
    # PTRM model
    'PTRM',
    'create_ptrm',
    # PTRM inference
    'PTRMInference',
    'run_ptrm_inference',
    # Factory presets
    'get_sudoku_extreme_config',
    'get_ppbench_config',
    'get_maze_hard_config',
    'get_arc_agi_config',
    'PRESET_CONFIGS',
]