"""
Configuration for the Neural Arithmetic Module (NAM).

Defines ``NAMConfig`` dataclass and preset model variants.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class NAMConfig:
    """
    Configuration for the Neural Arithmetic Module.

    :param hidden_size: Hidden dimension for embeddings and transformer layers.
    :param num_heads: Number of attention heads in TreeMHA.
    :param num_tree_layers: Number of TreeTransformerBlock layers for parsing.
    :param intermediate_size: FFN intermediate dimension.
    :param memory_size: Number of NTM memory slots.
    :param num_read_heads: Number of NTM read heads (typically 2: left/right operand).
    :param num_write_heads: Number of NTM write heads.
    :param max_expression_len: Maximum token sequence length.
    :param halt_max_steps: Maximum recursive reduction steps (ACT).
    :param halt_exploration_prob: Probability of forcing exploration during training.
    :param vocab_size: Token vocabulary size.
    :param hidden_dropout_rate: Dropout rate for hidden layers.
    :param attention_dropout_rate: Dropout rate for attention scores.
    :param normalization_type: Normalization layer type.
    :param ffn_type: Feed-forward network type.
    :param hidden_act: FFN activation function.
    :param layer_norm_eps: Epsilon for layer normalization.
    :param epsilon: Numerical stability constant for arithmetic.

    .. note::
       There is no ``shift_range``. NAM builds every NTM read and write head
       with ``AddressingMode.CONTENT`` (see :mod:`.cell`), under which the
       circular-shift projection is not created at all, so the field this
       docstring used to advertise configured nothing. Removed 2026-08-18;
       :meth:`from_dict` ignores it, so a config dict serialized before then
       still loads. See decisions.md D-014.
    """

    hidden_size: int = 128
    num_heads: int = 4
    num_tree_layers: int = 4
    intermediate_size: int = 256
    memory_size: int = 32
    num_read_heads: int = 2
    num_write_heads: int = 1
    max_expression_len: int = 64
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    vocab_size: int = 21
    hidden_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    normalization_type: str = "layer_norm"
    ffn_type: str = "mlp"
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    epsilon: float = 1e-7

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be positive "
                f"and divisible by num_heads ({self.num_heads})"
            )
        if self.memory_size <= 0:
            raise ValueError(f"memory_size must be positive, got {self.memory_size}")
        if self.halt_max_steps <= 0:
            raise ValueError(
                f"halt_max_steps must be positive, got {self.halt_max_steps}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NAMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


NAM_VARIANTS: Dict[str, Dict[str, Any]] = {
    "tiny": dict(
        hidden_size=64,
        num_heads=4,
        num_tree_layers=2,
        intermediate_size=128,
        memory_size=16,
        num_read_heads=2,
        num_write_heads=1,
        max_expression_len=32,
        halt_max_steps=8,
    ),
    "small": dict(
        hidden_size=128,
        num_heads=4,
        num_tree_layers=4,
        intermediate_size=256,
        memory_size=32,
        num_read_heads=2,
        num_write_heads=1,
        max_expression_len=64,
        halt_max_steps=16,
    ),
    "base": dict(
        hidden_size=256,
        num_heads=8,
        num_tree_layers=6,
        intermediate_size=512,
        memory_size=64,
        num_read_heads=2,
        num_write_heads=1,
        max_expression_len=128,
        halt_max_steps=32,
    ),
}
