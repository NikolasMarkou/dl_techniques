"""
SAM 2 (Segment Anything in Images and Videos): the image and streaming paths.
=============================================================================

Fifteen public classes across nine modules: a Hiera trunk with an FPN neck, a
streaming memory (memory attention, memory encoder, memory bank) and a mask
decoder that additionally emits an object score and an object pointer. No
pretrained weights ship here, and this package makes NO accuracy claim --
nothing here has been trained to any quality and no released SAM 2 checkpoint
has ever been loaded in this repository.

Based on:
---------
- Ravi, N. et al. (2024). "SAM 2: Segment Anything in Images and Videos."

Key Features:
------------
- Two entry points, deliberately different in kind: ``SAM2.call`` is the
  traceable image path, ``SAM2.stream_step`` the untraced streaming video path.
- ``SAM2MemoryBank`` is a plain-Python state container the caller constructs
  and drives itself; it owns no weights and is not a Keras layer.
- ``SAM2TrainingModel``, in the ``training_model`` submodule, is the traceable
  multi-frame wrapper stock ``fit()`` can train.
- The exported surface is three names, mirroring ``SAM1``'s. Every component
  stays behind its own submodule.

Architecture Overview:
---------------------
1. **Hiera** -- hierarchical window-attention trunk, four feature levels out.
2. **SAM2FpnNeck** / **SAM2ImageEncoder** -- ``d_model``-wide levels plus one
   sine positional encoding each, then the ``scalp`` level drop.
3. **SAM2MemoryAttention** -- self- then cross-attention against the memory
   sequence, 2D axial RoPE on both.
4. **SAM2MaskDecoder** -- masks, IoU predictions, object score, object pointer.
5. **SAM2MemoryEncoder** -- compresses mask and pixel features to ``mem_dim``.
6. **SAM2MemoryBank** -- frame selection, temporal slots, memory assembly.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM2 import SAM2, SAM2MemoryBank, create_sam2
from dl_techniques.models.SAM.SAM2.hiera import Hiera
model = create_sam2("tiny")
outputs = model({"image": images, "points": (coords, labels)})
```

Measured caveats:
----------------
- Widening ``__all__`` is a deliberate act, not a convenience. Its exact
  contents are asserted in both directions -- nothing missing, nothing extra --
  by ``tests/test_models/test_sam2/test_package_surface.py``, so the surface
  cannot drift open one re-export at a time.
- **Importing this package does NOT register** ``Custom>SAM2TrainingModel``.
  This ``__init__`` imports ``memory_bank`` and ``model`` only, never
  ``training_model``; SAM 1's and SAM 3's inits both import theirs. So a
  ``.keras`` file whose top-level class is ``SAM2TrainingModel`` needs an
  explicit ``import dl_techniques.models.SAM.SAM2.training_model`` BEFORE the
  ``load_model`` call. MEASURED at commit ``96c6a460b`` with the package import
  alone: ``TypeError: Could not deserialize class 'SAM2TrainingModel' because
  its parent module <the pre-move dotted path>.training_model cannot be
  imported`` -- Keras looks the class up in the registry first and, finding it
  absent, falls back to importing the module string the checkpoint recorded.
  Every SAM checkpoint written before this package moved recorded a path that
  no longer exists; that path is deliberately not spelled here, because a
  repo-wide grep asserts it survives nowhere under ``src/``.
- **SAM 2's mask head does not learn under joint training, the cause is known,
  and it is UNFIXED** -- see ``training_model``'s docstring for the arms that
  were measured and the constraint that binds.
"""

from .memory_bank import SAM2MemoryBank
from .model import SAM2, create_sam2

__all__ = [
    'SAM2',
    'SAM2MemoryBank',
    'create_sam2',
]
