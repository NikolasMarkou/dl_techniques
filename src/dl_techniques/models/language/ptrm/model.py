"""Probabilistic Tiny Recursive Model (PTRM): a Tiny Recursive Model (TRM) equipped
with a stochastic inference procedure for test-time compute scaling.

PTRM shares the exact same architecture as TRM: a single small network applied
recursively under Adaptive Computation Time (ACT), with a hierarchical latent
state (z_H, z_L) and a Q-head for learned halting. The innovation of PTRM is
purely at inference time: instead of a single deterministic rollout, PTRM runs
K parallel stochastic rollouts by injecting Gaussian noise into the latent state
at each supervision step, and selects the final answer using the model's Q-head.

This allows PTRM to escape bad basins in the latent space where deterministic
TRM gets stuck, without any retraining or task-specific augmentations.

References:
    - Sghaier et al., 2026. Probabilistic Tiny Recursive Model.
      (https://arxiv.org/abs/XXXX.XXXXX)
    - Jolicoeur-Martineau, 2025. Less is More: Recursive Reasoning with Tiny
      Networks. (https://arxiv.org/abs/2510.04871)
    - Graves, 2016. Adaptive Computation Time for Recurrent Neural Networks.
      (https://arxiv.org/abs/1603.08983)
    - Wang et al., 2025. Hierarchical Reasoning Model.
      (https://arxiv.org/abs/2506.21734)
"""

import keras
from typing import Optional, Dict, Any, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.models.language.trm.model import TRM, create_trm
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.ptrm.model")
class PTRM(TRM):
    """Probabilistic Tiny Recursive Model.

    Identical architecture to TRM. The PTRM innovation is in the inference
    procedure (see :class:`PTRMInference` in :mod:`dl_techniques.models.language.ptrm.inference`),
    not in the model architecture itself. This class exists to provide a
    distinct registration key and factory for PTRM checkpoints.

    Architecture:

    .. code-block:: text

        (carry, batch)
             │
             ▼
        ┌──────────────────────────┐
        │ TRMInner                 │  n latent recursions of z_L, z_H
        │ (rope attention)         │  updates; all but last under
        └──────────────────────────┘  stop_gradient
             │
             ▼  z_H
             │
      ┌──────┴──────┐
      ▼             ▼
  reasoning_output  q_head -> q_halt, q_continue
      │             │
      ▼             ▼
  carry frozen where halted (ops.where, not Python break)
             │
             ▼
      new_carry, outputs

    :param vocab_size: Size of the vocabulary for token embeddings.
    :param hidden_size: Dimensionality of hidden states. Must be divisible by `num_heads`.
    :param num_heads: Number of attention heads in transformer layers.
    :param expansion: FFN intermediate-size multiplier.
    :param seq_len: Length of the input sequence, excluding the puzzle embedding.
    :param puzzle_emb_len: Length of the puzzle embedding prefix. Default 16.
    :param h_layers: Number of layers in the H-level reasoning module. Default 2.
    :param l_layers: Number of layers in the L-level reasoning module. Default 2.
    :param halt_max_steps: Maximum number of ACT steps allowed. Must be >= 1.
    :param halt_exploration_prob: Probability of forcing extra exploration steps during training. Default 0.1.
    :param no_act_continue: If True, halt on `q_halt > 0`; if False, use Q-learning halting (`q_halt > q_continue`).
    :param rope_theta: RoPE base frequency. Default 10000.0.
    :param attention_type: Attention mechanism. Default 'group_query'.
    :param ffn_type: Feed-forward network type. Default 'swiglu'.
    :param normalization_type: Normalization layer type. Default 'rms_norm'.
    :param normalization_position: 'pre' or 'post'. Default 'post'.
    :param dropout_rate: Dropout rate for transformer layers. Default 0.0.
    :param attention_dropout_rate: Dropout rate for attention. Default 0.0.
    :param **kwargs: Forwarded to `keras.Model`.

    :raises ValueError: If `hidden_size` is not divisible by `num_heads`, if
        `halt_max_steps` is below 1, or if `halt_exploration_prob` is outside [0, 1].
    """

    # Same variant configurations as TRM (paper uses identical backbone)
    MODEL_VARIANTS = {
        "mlp_tiny": {
            "hidden_size": 256,
            "num_heads": 4,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 8,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
        "mlp_small": {
            "hidden_size": 384,
            "num_heads": 6,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 10,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
        # Paper's TRM-MLP for Sudoku-Extreme: 5M params
        "mlp_base": {
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 16,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
        # Paper's TRM-Att for PPBench/Maze/ARC: 7M params
        "att_small": {
            "hidden_size": 384,
            "num_heads": 6,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 16,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
        "att_base": {
            "hidden_size": 512,
            "num_heads": 8,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 16,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
        "att_large": {
            "hidden_size": 768,
            "num_heads": 12,
            "expansion": 2.0,
            "h_layers": 2,
            "l_layers": 2,
            "halt_max_steps": 16,
            "attention_type": "group_query",
            "ffn_type": "swiglu",
        },
    }

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        expansion: float,
        seq_len: int,
        puzzle_emb_len: int = 16,
        h_layers: int = 2,
        l_layers: int = 2,
        halt_max_steps: int = 10,
        halt_exploration_prob: float = 0.1,
        no_act_continue: bool = True,
        rope_theta: float = 10000.0,
        attention_type: str = 'group_query',
        ffn_type: str = 'swiglu',
        normalization_type: str = 'rms_norm',
        normalization_position: str = 'post',
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        **kwargs: Any
    ) -> None:
        # Delegate all initialization to TRM parent
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len,
            h_layers=h_layers,
            l_layers=l_layers,
            halt_max_steps=halt_max_steps,
            halt_exploration_prob=halt_exploration_prob,
            no_act_continue=no_act_continue,
            rope_theta=rope_theta,
            attention_type=attention_type,
            ffn_type=ffn_type,
            normalization_type=normalization_type,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            **kwargs
        )

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        seq_len: int,
        puzzle_emb_len: int = 16,
        **kwargs: Any
    ) -> "PTRM":
        """Create a PTRM from a predefined variant.

        :param variant: String, one of "mlp_tiny", "mlp_small", "mlp_base",
            "att_small", "att_base", "att_large".
        :param vocab_size: Integer, size of vocabulary.
        :param seq_len: Integer, maximum sequence length.
        :param puzzle_emb_len: Integer, puzzle embedding length. Default 16.
        :param **kwargs: Additional arguments passed to the constructor.
        :return: PTRM instance.
        :raises ValueError: If variant is not recognized.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        return cls(
            vocab_size=vocab_size,
            seq_len=seq_len,
            puzzle_emb_len=puzzle_emb_len,
            **config
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        # Ensure the class name is PTRM for serialization
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PTRM":
        """Create model from configuration."""
        return cls(**config)


# ---------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------


def create_ptrm(
    vocab_size: int,
    hidden_size: int,
    num_heads: int,
    expansion: float,
    seq_len: int,
    puzzle_emb_len: int = 16,
    h_layers: int = 2,
    l_layers: int = 2,
    halt_max_steps: int = 10,
    halt_exploration_prob: float = 0.1,
    no_act_continue: bool = True,
    rope_theta: float = 10000.0,
    attention_type: str = 'group_query',
    ffn_type: str = 'swiglu',
    normalization_type: str = 'rms_norm',
    normalization_position: str = 'post',
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    name: Optional[str] = None,
) -> PTRM:
    """Build a PTRM and build its inner layer.

    Returns a PTRM instance with its inner layer built so that ``H_init`` /
    ``L_init`` weights exist before the first ``call``. This mirrors the
    factory convention used elsewhere in ``dl_techniques.models``.

    :param vocab_size: Size of the vocabulary for token embeddings.
    :param hidden_size: Dimensionality of hidden states. Must be divisible by ``num_heads``.
    :param num_heads: Number of attention heads in transformer layers.
    :param expansion: Factor to determine FFN intermediate size.
    :param seq_len: Length of the input sequence (excluding puzzle embedding).
    :param puzzle_emb_len: Length of the puzzle embedding prefix. Default 16.
    :param h_layers: Number of layers in the H-level module. Default 2.
    :param l_layers: Number of layers in the L-level module. Default 2.
    :param halt_max_steps: Maximum ACT steps. Must be >= 1. Default 10.
    :param halt_exploration_prob: Probability of exploration during halting. Must be in [0, 1]. Default 0.1.
    :param no_act_continue: Use simple halting (True) vs Q-learning (False). Default True.
    :param rope_theta: Theta for Rotary Position Embedding. Default 10000.0.
    :param attention_type: Type of attention mechanism. Default 'group_query'.
    :param ffn_type: Type of feed-forward network. Default 'swiglu'.
    :param normalization_type: Type of normalization layer. Default 'rms_norm'.
    :param normalization_position: ``pre`` or ``post`` normalization. Default 'post'.
    :param dropout_rate: Dropout rate for transformer layers. Default 0.0.
    :param attention_dropout_rate: Dropout rate for attention. Default 0.0.
    :param name: Optional Keras model name.

    :return: A built ``PTRM`` instance.
    :raises ValueError: For the same argument checks :class:`PTRM` makes.
    """
    model = PTRM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        expansion=expansion,
        seq_len=seq_len,
        puzzle_emb_len=puzzle_emb_len,
        h_layers=h_layers,
        l_layers=l_layers,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=halt_exploration_prob,
        no_act_continue=no_act_continue,
        rope_theta=rope_theta,
        attention_type=attention_type,
        ffn_type=ffn_type,
        normalization_type=normalization_type,
        normalization_position=normalization_position,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        name=name,
    )
    model.build()
    return model


# ---------------------------------------------------------------------