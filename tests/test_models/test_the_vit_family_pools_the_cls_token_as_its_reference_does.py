"""Each ViT-family model reads, or skips, the CLS token in pooling as its reference does.

Guard for plan-2026-10-14T062616-81938e81 step 5.6 (D-123 item a, D-205). The
review called the exclusion inconsistent and proposed one uniform rule; the
references were read instead (copies and sha256 in the D-205 entry):

- timm ``VisionTransformer.pool`` / ``global_pool_nlc`` (main, 877d754):
  ``avg``, ``max`` and ``avgmax`` reduce ``x[:, num_prefix_tokens:]``;
  ``token`` reads ``x[:, 0]``.
- unilm BEiT ``modeling_finetune.py``: ``x[:, 1:, :].mean(1)`` with
  ``use_mean_pooling``, else ``x[:, 0]``.
- big_vision ``vit.py`` (the SigLIP tower): a CLS token exists only for
  ``pool_type='tok'``; ``gap`` averages patch tokens only.
- deit ``models_v2.py`` (the hMLP-stem reference): ``x[:, 0]`` only.

So every reduction skips the CLS token and ``cls`` reads it, in all four.
``vit`` and ``beit`` already did; ``vit_siglip`` and ``vit_hmlp`` averaged the
CLS token into ``mean``/``max`` (measured at 69b68fe5a: perturbing position 0
moved the pooled output).
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.models.vision.beit.model import BeitModel
from dl_techniques.models.vision.vit.model import ViT
from dl_techniques.models.vision.vit_hmlp.model import ViTHMLP
from dl_techniques.models.vision.vit_siglip.model import SigLIPVisionTransformer

SHAPE = (32, 32, 3)


def _poolers(model):
    return [l for l in model._flatten_layers() if isinstance(l, SequencePooling)]


def _cls_moves(pool) -> bool:
    """Does perturbing position 0 (the CLS token) move this pooler's output?"""
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(2, 5, 8)).astype("float32")
    base = np.asarray(pool(seq))
    seq[:, 0] += 50.0 * rng.normal(size=(2, 8)).astype("float32")
    return float(np.abs(np.asarray(pool(seq)) - base).max()) > 1e-3


SUBJECTS = {
    "vit": lambda pooling: ViT(
        input_shape=SHAPE, scale="tiny", patch_size=16, include_top=False, pooling=pooling),
    "vit_siglip": lambda pooling: SigLIPVisionTransformer(
        input_shape=SHAPE, scale="tiny", patch_size=16, include_top=False, pooling=pooling),
    "vit_hmlp": lambda pooling: ViTHMLP(
        input_shape=SHAPE, scale="tiny", patch_size=16, include_top=False, pooling=pooling),
}
# Matrix: does the pooled output read the CLS token?
READS_CLS = {"cls": True, "mean": False, "max": False}


@pytest.mark.parametrize("pooling", sorted(READS_CLS))
@pytest.mark.parametrize("subject", sorted(SUBJECTS))
def test_the_pooling_reads_the_cls_token_only_for_cls(subject, pooling):
    model = SUBJECTS[subject](pooling)
    poolers = _poolers(model)
    assert len(poolers) == 1, poolers
    assert _cls_moves(poolers[0]) is READS_CLS[pooling]


@pytest.mark.parametrize("use_mean_pooling, reads_cls", [(True, False)])
def test_beit_mean_pooling_skips_the_cls_token(use_mean_pooling, reads_cls):
    """BEiT's cls path is ``x[:, 0]`` in the shared head, not a SequencePooling."""
    from dl_techniques.models.vision.beit.model import BeitForImageClassification

    backbone = BeitModel(input_shape=SHAPE, scale="tiny", patch_size=16,
                         use_mean_pooling=use_mean_pooling, drop_path_rate=0.0)
    model = BeitForImageClassification(backbone, num_classes=3)
    model(np.zeros((1, 32, 32, 3), "float32"))
    poolers = _poolers(model)
    assert len(poolers) == 1, poolers
    assert _cls_moves(poolers[0]) is reads_cls
