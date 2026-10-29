"""
Fusion layers.

This package holds two public classes. ``HierarchicalScaleFusion``, defined in
``hierarchical_fusion.py``, folds a fine-to-coarse feature pyramid into one
finest-resolution map; it is built from ``ConvBlock`` and ``PixelShuffle2D``,
has no factory and no consumer yet. The rest of this docstring describes the
other class, ``MultiModalFusion``, defined in ``multimodal_fusion.py``. It takes one tensor per modality -- vision, text,
audio and so on -- and combines them into a fused representation. A
``fusion_strategy`` key selects the path. The strategy decides which
sub-layers get built and what shape comes back.

**This package exports nothing.** This file has no imports and no ``__all__``,
so ``from dl_techniques.layers.fusion import MultiModalFusion`` raises
``ImportError``. Import from the submodule instead::

    from dl_techniques.layers.fusion.multimodal_fusion import (
        MultiModalFusion,
        FusionStrategy,
    )
    from dl_techniques.layers.fusion.hierarchical_fusion import (
        HierarchicalScaleFusion,
    )

The one consumer in this repo already does that:
``src/dl_techniques/layers/heads/vlm/``. NanoVLM used it until
2026-10-15 and now runs one causal decoder over its projected image tokens
and the text instead: the symmetric ``'cross_attention'`` strategy lets
every modality attend to the other, which leaks future text into earlier
positions of a causal language model.

**Architecture Overview:**

.. code-block:: text

    modality tensors, one per modality
    each (B, Ti, dim)
                   │
                   ▼
        ┌──────────────────────────┐
        │     MultiModalFusion     │
        │  dispatch on strategy    │
        │  8 keys, 7 call paths    │
        └────────────┬─────────────┘
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     _call_cross  _call_att-   the other
     _attention   ention_      5 call paths
        │         pooling           │
        ▼            │              ▼
     tuple of N      ▼         (B, T, dim)
     (B, Ti, dim)  (B, dim)

    The other 5: _call_concatenation, _call_elementwise
    (shared by 'addition' and 'multiplication', which is
    why 8 keys make 7 paths), _call_gated, _call_bilinear
    and _call_tensor_fusion.

    Only 'cross_attention' and 'attention_pooling' accept
    modalities of different sequence length. The other six
    keys are refused at call time unless every Ti matches.

Per-strategy internals are drawn in ``MultiModalFusion``'s own docstring and
in each ``_call_*`` docstring. They are not repeated here.
"""
