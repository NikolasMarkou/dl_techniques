"""Factory functions for creating PTRM models with preset configurations."""

from typing import Optional, Dict, Any

from .model import PTRM, create_ptrm


def create_ptrm_from_variant(
    variant: str,
    vocab_size: int,
    seq_len: int,
    puzzle_emb_len: int = 16,
    **kwargs: Any
) -> PTRM:
    """Create a PTRM model from a predefined variant.

    :param variant: One of "mlp_tiny", "mlp_small", "mlp_base", "att_small", "att_base", "att_large".
    :param vocab_size: Vocabulary size.
    :param seq_len: Sequence length.
    :param puzzle_emb_len: Puzzle embedding length.
    :param **kwargs: Additional arguments to override variant config.
    :return: PTRM instance.
    """
    model = PTRM.from_variant(variant, vocab_size, seq_len, puzzle_emb_len, **kwargs)
    if not model.built:
        model.build()
    return model


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