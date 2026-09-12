"""Zamba2: hybrid Mamba2 + shared-attention SSM/Transformer causal language model.

Interleaves per-depth Mamba2 SSM blocks (reused directly from
:mod:`dl_techniques.models.language.mamba`) with a small number of **shared**
attention+MLP "mem-blocks" invoked at multiple depths, where each reuse
("occurrence") of the shared MLP block's up-projection carries its own small
additive LoRA delta so the one physical block can specialize per depth
without multiplying its parameter count (Zyphra's Zamba2 architecture).

See ``plans/plan-2026-09-12T075714-035fd488/plan.md`` for the build sequence
and `README.md` in this package for the architecture writeup.
"""

from dl_techniques.models.language.zamba2.layers import (
    LoRAAdapter,
    Zamba2MambaBlock,
    Zamba2SharedAttentionBlock,
    Zamba2SharedMLPBlock,
)
from dl_techniques.models.language.zamba2.model import (
    MODEL_VARIANTS,
    Zamba2Model,
    create_zamba2,
)

__all__ = [
    "LoRAAdapter",
    "MODEL_VARIANTS",
    "Zamba2MambaBlock",
    "Zamba2Model",
    "Zamba2SharedAttentionBlock",
    "Zamba2SharedMLPBlock",
    "create_zamba2",
]
