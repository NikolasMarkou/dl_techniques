"""``VisionEncoder`` skips its CLS token in every reduction; ``TextEncoder`` reads it.

Guard for plan-2026-10-14T062616-81938e81 step 5.6 (D-123 item a, D-205).
The review proposed one uniform exclusion rule for both encoders; the
references disagree, so each encoder follows its own:

- ``VisionEncoder`` (a timm-style ViT): timm ``global_pool_nlc`` reduces
  ``x[:, num_prefix_tokens:]`` for every reduction, and its ``map`` attention
  pool also drops the prefix tokens by default (``pool_include_prefix=False``,
  timm main 877d754). Measured at 69b68fe5a: only ``mean`` and ``max``
  excluded position 0; ``min``, ``sum``, ``mean_max``, ``mean_std``,
  ``mean_max_min``, ``attention``, ``multi_head_attention``, ``weighted``,
  ``top_k_mean`` and ``top_k_max`` averaged the CLS token in.
- ``TextEncoder`` (a BERT-style encoder whose CLS token stands in for
  ``[CLS]``): sentence-transformers ``Pooling`` averages every token the
  attention mask keeps, ``[CLS]`` included, so no strategy excludes it.

``cls``/``first`` read position 0 by intent, ``none``/``flatten`` return the
whole sequence, and ``last``/``middle`` never land on position 0 here.
"""

import typing

import numpy as np
import pytest

from dl_techniques.layers.sequence_pooling.sequence_pooling import PoolingStrategy
from dl_techniques.layers.transformers.text_encoder import TextEncoder
from dl_techniques.layers.transformers.vision_encoder import VisionEncoder

STRATEGIES = typing.get_args(PoolingStrategy)
SEQUENCE_READERS = {"cls", "first", "none", "flatten"}
NEVER_POSITION_ZERO = {"last", "middle"}
D = 16


def _cls_moves(pool, seq_len) -> bool:
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(2, seq_len, D)).astype("float32")
    base = np.asarray(pool(seq))
    seq[:, 0] += 50.0 * rng.normal(size=(2, D)).astype("float32")
    return float(np.abs(np.asarray(pool(seq)) - base).max()) > 1e-3


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_vision_encoder_reads_its_cls_token_only_by_intent(strategy):
    encoder = VisionEncoder(
        img_size=32, patch_size=16, embed_dim=D, depth=1, num_heads=2,
        use_cls_token=True, output_mode=strategy,
    )
    assert _cls_moves(encoder.pooling_layer, 5) is (strategy in SEQUENCE_READERS)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_text_encoder_reads_its_cls_token_in_every_reduction(strategy):
    encoder = TextEncoder(
        vocab_size=32, embed_dim=D, depth=1, num_heads=2, max_seq_len=4,
        use_cls_token=True, output_mode=strategy,
    )
    assert _cls_moves(encoder.pooling_layer, 5) is (strategy not in NEVER_POSITION_ZERO)
