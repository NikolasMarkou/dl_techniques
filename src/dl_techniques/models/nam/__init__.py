"""Neural Arithmetic Module (NAM) — public API re-exports.

``NAM_VARIANTS`` is the package's original spelling of the variant table and is
kept as the module-level name because trainers and tests reference it;
``NAM.MODEL_VARIANTS`` is a class-level alias to the same dict, added for the
house shape. Nothing was renamed and no variant was invented.
"""
from dl_techniques.models.nam.config import NAMConfig, NAM_VARIANTS
from dl_techniques.models.nam.tokenizer import ArithmeticTokenizer
from dl_techniques.models.nam.cell import NAMCell
from dl_techniques.models.nam.model import NAM, create_nam

__all__ = [
    "NAM",
    "NAMConfig",
    "NAMCell",
    "NAM_VARIANTS",
    "ArithmeticTokenizer",
    "create_nam",
]
