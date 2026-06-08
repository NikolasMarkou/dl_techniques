"""Task-head layers, organized by domain (``nlp``, ``vision``, ``vlm``).

This package consolidates the formerly-separate ``nlp_heads``, ``vision_heads``,
and ``vlm_heads`` packages. The top-level re-export surface and the
``create_head`` facade are populated in a later step; for now import from the
domain sub-packages directly (e.g. ``from dl_techniques.layers.heads.nlp import
create_nlp_head``).
"""
