"""Preset configurations for PTRM, one per benchmark of Sghaier et al. (2026).

Each helper returns the variant and data shape a benchmark uses plus the
inference settings (``K`` rollouts, ``D`` supervision steps, noise ``sigma``).
Build the model with :func:`dl_techniques.models.language.ptrm.create_ptrm`
from the ``variant``, ``vocab_size``, ``seq_len`` and ``puzzle_emb_len`` keys,
and pass the ``"inference"`` entry to
:class:`dl_techniques.models.language.ptrm.PTRMInference`.
"""

from typing import Dict, Any


# Preset configurations matching paper experiments

def get_sudoku_extreme_config() -> Dict[str, Any]:
    """Configuration for Sudoku-Extreme (TRM-MLP, 5M params).

    Paper settings: K=100, D=64, σ=0.3
    """
    return {
        "variant": "mlp_base",
        "vocab_size": 20,  # 0-9 digits + special tokens
        "seq_len": 81,     # 9x9 grid flattened
        "puzzle_emb_len": 16,
        "inference": {
            "num_rollouts": 100,
            "supervision_steps": 64,
            "noise_scale": 0.3,
        }
    }


def get_ppbench_config() -> Dict[str, Any]:
    """Configuration for PPBench puzzles (TRM-Att, 7M params).

    Paper settings: K=100, D=48, σ=0.2
    """
    return {
        "variant": "att_base",
        "vocab_size": 294,  # Unified vocabulary from paper
        "seq_len": 100,     # 10x10 grid padded
        "puzzle_emb_len": 16,
        "inference": {
            "num_rollouts": 100,
            "supervision_steps": 48,
            "noise_scale": 0.2,
        }
    }


def get_maze_hard_config() -> Dict[str, Any]:
    """Configuration for Maze-Hard (TRM-Att, 7M params).

    Paper settings: K=100, D=16, σ=1.0
    """
    return {
        "variant": "att_base",
        "vocab_size": 4,    # wall, empty, start, goal
        "seq_len": 900,     # 30x30 maze flattened
        "puzzle_emb_len": 16,
        "inference": {
            "num_rollouts": 100,
            "supervision_steps": 16,
            "noise_scale": 1.0,
        }
    }


def get_arc_agi_config() -> Dict[str, Any]:
    """Configuration for ARC-AGI-2 (TRM-Att, 7M params).

    Paper settings: K=25, D=16, σ=0.2
    """
    return {
        "variant": "att_base",
        "vocab_size": 10,   # ARC colors 0-9
        "seq_len": 900,     # 30x30 grid
        "puzzle_emb_len": 16,
        "inference": {
            "num_rollouts": 25,
            "supervision_steps": 16,
            "noise_scale": 0.2,
        }
    }


# Mapping for easy access
PRESET_CONFIGS = {
    "sudoku_extreme": get_sudoku_extreme_config,
    "ppbench": get_ppbench_config,
    "maze_hard": get_maze_hard_config,
    "arc_agi": get_arc_agi_config,
}