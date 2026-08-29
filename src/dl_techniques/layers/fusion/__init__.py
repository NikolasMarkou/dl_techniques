"""
Multimodal fusion layers.

This package holds one public class, ``MultiModalFusion``, defined in
``multimodal_fusion.py``. It takes one tensor per modality -- vision, text,
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

Both consumers in this repo already do that:
``src/dl_techniques/layers/heads/vlm/factory.py`` and
``src/dl_techniques/models/vision_language/nano_vlm/model.py``.

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
