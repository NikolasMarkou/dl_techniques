"""DINO self-distillation vision models — public API.

Three independently-authored ViT backbones from the DINO paper line, sharing one
converged factory surface but NOT a common trunk (that unification is a deliberate
non-goal; see ``src/dl_techniques/models/dino/README.md`` § "Why there is no shared
base class"):

- ``dino_v1`` — Caron et al. 2021, "Emerging Properties in Self-Supervised Vision
  Transformers". Exposes ``DINOHead`` (the projection head every version's SSL
  pretraining uses) and ``create_dino_teacher_student_pair``.
- ``dino_v2`` — Oquab et al. 2023. Adds register tokens, LayerScale, iBOT-style
  patch masking on the forward path, and a SwiGLU FFN on ``giant``.
- ``training`` — ``DINOTrainingModel``: student + frozen EMA teacher over a
  multi-crop batch, trainable under stock ``model.fit()``. This is the object the
  SSL pretraining pipeline compiles with ``DINOLoss``.
- ``dino_v3`` — Siméoni et al. 2025, partially. Adds real 1-D RoPE via a
  ``group_query`` attention. Gram anchoring and Sinkhorn-Knopp centering are NOT
  implemented — the README says so explicitly and per-row.

The matching losses live in ``src/dl_techniques/losses/dino_loss.py``
(``DINOLoss``, ``iBOTPatchLoss``, ``KoLeoLoss``) and are exported from
``dl_techniques.losses``, not from here.

**Variant tables are per-CLASS, not module-level.** ``DINOv1.MODEL_VARIANTS``,
``DINOv2VisionTransformer.MODEL_VARIANTS`` and ``DINOv3.MODEL_VARIANTS`` are three
distinct class attributes that happen to share a name. They are reachable through
the exported classes; this package deliberately binds NO module-level
``MODEL_VARIANTS`` alias, because any single name would have to pick one of the
three and would misdescribe the other two (their values genuinely differ — v2's
``giant`` carries ``ffn_type='swiglu'``, v3's carries ``patch_size=(14, 14)`` and
``stochastic_depth_rate=0.4``, v1's carries neither).

Import cost: importing this package eagerly imports Keras and the three model
modules (~4 s cold, dominated by ``import keras``). ``import dl_techniques.models``
is unaffected — that parent ``__init__.py`` is empty and imports no subpackage.
"""

from .dino_v1 import (
    ModelVariant,
    DINOHead,
    DINOv1,
    create_dino_v1,
    create_dino_teacher_student_pair,
)
from .dino_v2 import (
    DINOv2Block,
    DINOv2VisionTransformer,
    DINOv2,
    create_dino_v2,
)
from .dino_v3 import (
    DINOv3,
    create_dino_v3,
)
from .training import (
    N_GLOBAL_VIEWS,
    DINOTrainingModel,
    create_dino_training_model,
)

__all__ = [
    "ModelVariant",
    "DINOHead",
    "DINOv1",
    "DINOv2Block",
    "DINOv2VisionTransformer",
    "DINOv2",
    "DINOv3",
    "create_dino_v1",
    "create_dino_v2",
    "create_dino_v3",
    "create_dino_teacher_student_pair",
    "DINOTrainingModel",
    "create_dino_training_model",
    "N_GLOBAL_VIEWS",
]
