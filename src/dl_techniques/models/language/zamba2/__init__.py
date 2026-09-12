"""Zamba2: hybrid Mamba2 + shared-attention SSM/Transformer causal language model.

Interleaves per-depth Mamba2 SSM blocks (reused directly from
:mod:`dl_techniques.models.language.mamba`) with a small number of **shared**
attention+MLP "mem-blocks" invoked at multiple depths, where each reuse
("occurrence") of the shared MLP block's up-projection carries its own small
additive LoRA delta so the one physical block can specialize per depth
without multiplying its parameter count (Zyphra's Zamba2 architecture).

This module is under active construction; public exports are added
incrementally as each piece lands. See
``plans/plan-2026-09-12T075714-035fd488/plan.md`` for the build sequence.
"""
