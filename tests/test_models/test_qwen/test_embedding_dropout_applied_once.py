"""``dropout_rate`` must reach the embedding path EXACTLY ONCE.

Both Qwen3 embedding wrappers were listed as dropped-knob sites (N-04): each
stores ``self.dropout_rate``, serialises it, and called
``create_embedding_layer('positional_learned', ...)`` without it, although
``EMBEDDING_REGISTRY['positional_learned']`` declares ``dropout_rate``.

**That reading was refuted by measurement.** ``PositionalEmbedding`` applies its
own ``dropout_rate`` immediately after adding the positional table, and both
wrappers then applied a second, separately-constructed ``Dropout(dropout_rate)``
to the very same tensor. Forwarding the kwarg -- the "obvious" fix -- would have
applied dropout TWICE back to back, an effective rate of ``1 - (1 - p)^2``
(0.19 for the shipped 0.1), which is a behaviour REGRESSION dressed as a bug
fix. The knob was never dead: it was reaching the embedding path through a
redundant second layer.

The resolution was to forward the kwarg AND delete the redundant ``Dropout``, so
there is exactly one dropout on the embedding sum, at the rate asked for, at the
same point in the graph as before. This module is the guard for that invariant:
it is GREEN at the pre-fix HEAD (one dropout, in ``embedding_dropout``), GREEN
after the fix (one dropout, in the positional embedding), and RED for the naive
forward-and-keep-both fix -- which is the failure mode it exists to catch.
"""

import pytest

from dl_techniques.models.qwen.qwen3_embeddings import (
    Qwen3EmbeddingLayer,
    Qwen3RerankerLayer,
)

COMMON = dict(
    vocab_size=64,
    hidden_size=16,
    num_layers=1,
    num_heads=2,
    intermediate_size=32,
    max_seq_len=16,
)

DROPOUT_RATE = 0.25


def _embedding_path_dropout_rates(layer) -> list:
    """Every dropout rate acting on the embedding sum, in graph order.

    Interface contract: reads the two construction sites the wrappers own --
    the positional embedding's internal dropout and the optional standalone
    ``embedding_dropout`` -- and returns the ACTIVE (non-zero) rates. A rate of
    0.0 is not "a dropout that does nothing to worry about"; it is not a dropout
    at all, and including it would make the "exactly once" claim untestable.
    """
    rates = [getattr(layer.positional_embeddings, "dropout_rate", 0.0)]
    standalone = getattr(layer, "embedding_dropout", None)
    if standalone is not None:
        rates.append(float(standalone.rate))
    return [float(r) for r in rates if float(r) > 0.0]


@pytest.mark.parametrize(
    "cls", [Qwen3EmbeddingLayer, Qwen3RerankerLayer], ids=["embedding", "reranker"]
)
class TestEmbeddingDropoutIsAppliedOnce:
    def test_exactly_one_dropout_acts_on_the_embedding_sum(self, cls):
        layer = cls(dropout_rate=DROPOUT_RATE, **COMMON)
        rates = _embedding_path_dropout_rates(layer)
        assert rates == [DROPOUT_RATE], (
            f"{cls.__name__} applies {len(rates)} dropouts to the embedding sum "
            f"at rates {rates}; two stacked dropouts at rate p give an effective "
            f"rate of 1-(1-p)^2 = {1 - (1 - DROPOUT_RATE) ** 2:.4f}, not "
            f"{DROPOUT_RATE}."
        )

    def test_dropout_rate_zero_leaves_no_active_dropout(self, cls):
        """The other half: the knob must be able to turn dropout OFF entirely."""
        layer = cls(dropout_rate=0.0, **COMMON)
        assert _embedding_path_dropout_rates(layer) == []
