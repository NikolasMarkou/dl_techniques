"""
ColBERT: late-interaction retrieval, one shared encoder, two training recipes.

Late interaction runs the encoder **independently** on the query and on the
document -- so every document embedding is computed once, offline -- and scores
a pair with a cheap similarity between two already-computed matrices of
per-token vectors:

.. math::

    S(q, d) = \\sum_{i \\in |E_q|} \\max_{j \\in |E_d|} E_{q_i} \\cdot E_{d_j}^{T}

**v1 and v2 build the same network.** That is a fact about the reference
implementation, not a simplification adopted here: the official
``stanford-futuredata/ColBERT`` repository ships a single
``colbert/modeling/colbert.py`` for both papers, and v1 behaviour is v2's code
with ``use_ib_negatives=False``, ``nway=2``, no distillation scores and no
residual compression. What separates the two here is the *training recipe*
(``ColBERTPairwiseSoftmaxLoss`` versus ``ColBERTDistillationLoss``) plus the
v2-only, index-time :class:`ResidualCompressionCodec`. Both
:func:`create_colbert_v1` and :func:`create_colbert_v2` are therefore real,
exported, documented factories over one class -- not aliases, and not two
architectures.

**No pretrained weights exist**, for ColBERT or for the BERT backbone;
``from_variant(pretrained=True)`` raises on both. Any number produced by
training this model is a **wiring result**, never a retrieval-quality claim.

``README.md`` in this directory carries the usage guide, the variant table with
its two labelled provenance classes, and the full list of deviations from the
reference.

Public surface:

- :class:`ColBERT` -- the model, with ``MODEL_VARIANTS``, ``from_variant`` and
  the ``encode_query`` / ``encode_document`` / ``score`` methods.
- :class:`ColBERTProjection` -- bias-free ``hidden -> dim`` projection, mask
  multiply, L2 normalize.
- :class:`MaxSimScorer` -- the sentinel-masked max-then-sum late interaction.
- :class:`ColBERTTokenizer` -- asymmetric ``[Q]`` / ``[D]`` streams, query
  ``[MASK]`` augmentation, document punctuation skiplist.
- :class:`ResidualCompressionCodec` -- v2 index-time residual compression;
  never part of the forward pass or of any loss.
- :func:`create_colbert`, :func:`create_colbert_v1`, :func:`create_colbert_v2`.

References:
    - Khattab & Zaharia, 2020. ColBERT: Efficient and Effective Passage Search
      via Contextualized Late Interaction over BERT.
      (https://arxiv.org/abs/2004.12832)
    - Santhanam et al., 2021. ColBERTv2: Effective and Efficient Retrieval via
      Lightweight Late Interaction. (https://arxiv.org/abs/2112.01488)
"""

from .components import ColBERTProjection, MaxSimScorer
from .compression import ResidualCompressionCodec
from .tokenization import ColBERTTokenizer
from .model import (
    ColBERT,
    create_colbert,
    create_colbert_v1,
    create_colbert_v2,
)

__all__ = [
    "ColBERT",
    "ColBERTProjection",
    "ColBERTTokenizer",
    "MaxSimScorer",
    "ResidualCompressionCodec",
    "create_colbert",
    "create_colbert_v1",
    "create_colbert_v2",
]
