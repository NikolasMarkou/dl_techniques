"""Neural computers — models with an explicit external memory and a differentiable
controller.

- `nam/` — a tree-transformer parse, NTM memory and TRM halting stack that evaluates
  arithmetic expressions. The name misattributes; see `models/CLAUDE.md`
- `ntm/` — Neural Turing Machine

Import from the leaf package, not from here — this family package carries no re-exports
by design (the reasoning is written out in `models/vision/__init__.py`):

    from dl_techniques.models.neural_computer.ntm import NTMModel
"""
