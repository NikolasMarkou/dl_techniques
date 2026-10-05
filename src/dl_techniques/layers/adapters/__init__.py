"""Adapter layers: low-rank weight updates, and updates confined to a local support.

This package owns two composable pieces.

:class:`LoRAAdapter` computes a pure additive low-rank **delta**
``(x @ A[i]) @ B[i] * (alpha / rank)`` for a caller-selected adapter slot, with
``num_adapters`` independent ``A``/``B`` pairs. The slot index is a plain Python
``int`` on the ``call`` signature, fixed at model-construction time, so one
physical layer can specialize at many call sites without multiplying its own
parameter count. Two current consumers use the same axis for different reasons:
Zamba2 for depth positions of a weight-shared block, Local Support Learning for
sequential learning phases.

:class:`GatedAdapter` composes that delta with a
:class:`~dl_techniques.layers.statistics.local_support_gate.LocalSupportGate` so
the update reaches only inputs drawn from the distribution that produced it —
the gate half of Local Support Learning.

Neither layer owns the base projection it augments; both return a delta the
caller adds onto its own output. That is the seam that lets one adapter serve a
``Dense`` inside a SwiGLU block and a grouped-query projection inside a decoder
without either call site knowing the other exists.

``factory.py`` registers one key, ``lora``, backed by ``ADAPTER_REGISTRY``.
``create_adapter`` is the strict construction path: a keyword the chosen type
does not declare raises rather than being dropped. ``GatedAdapter`` is
deliberately **not** a factory key — it is a composition, reachable only by
naming the class, so a caller cannot reach a gated adapter by passing a string.

References:
    - Hu, E. J. et al., 2021. LoRA: Low-Rank Adaptation of Large Language
      Models. (https://arxiv.org/abs/2106.09685)
    - Ben-Kish, A. et al., 2026. Local Support Learning.
      (https://arxiv.org/abs/2610.02126)
"""

from .factory import (
    ADAPTER_REGISTRY,
    AdapterType,
    STRICT_DROPPED_KEY_MARKER,
    assemble_adapter_config,
    create_adapter_from_config,
    create_adapter_layer,
    get_adapter_info,
    get_adapter_requirements,
    list_adapter_types,
    validate_adapter_config,
)
from .gated_adapter import GatedAdapter
from .lora import LEGACY_NUM_OCCURRENCES_KEY, LoRAAdapter

__all__ = [
    "ADAPTER_REGISTRY",
    "AdapterType",
    "STRICT_DROPPED_KEY_MARKER",
    "assemble_adapter_config",
    "create_adapter_from_config",
    "create_adapter_layer",
    "get_adapter_info",
    "get_adapter_requirements",
    "list_adapter_types",
    "validate_adapter_config",
    "LEGACY_NUM_OCCURRENCES_KEY",
    "LoRAAdapter",
    "GatedAdapter",
]
