"""``SequencePooling`` consumes no global RNG draw unless a strategy owns a kernel.

Guard for plan-2026-10-14T062616-81938e81 step 12 Part F item 10 (D-061,
D-087). Before the fix ``SequencePooling.__init__`` resolved its
``'glorot_uniform'`` default for every strategy, and a seedless Keras random
initializer takes a seed from the global Python RNG when it is created, so a
weightless ``'mean'`` pooler moved every fixed-seed draw after it. The VLM
heads worked around it with ``kernel_initializer='zeros'``.

Only ``'attention'`` and ``'multi_head_attention'`` hand ``kernel_initializer``
to a sub-layer (``AttentionPooling``); ``'weighted'`` and the ``weighted_sum``
aggregation weights use ``'ones'``. Every other strategy owns no weight.
"""

import random
import typing

import numpy as np
import pytest
import keras

from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.layers.sequence_pooling.sequence_pooling import PoolingStrategy

KERNEL_STRATEGIES = ("attention", "multi_head_attention")
ALL_STRATEGIES = typing.get_args(PoolingStrategy)
NO_KERNEL = tuple(s for s in ALL_STRATEGIES if s not in KERNEL_STRATEGIES)


def _draws(build) -> bool:
    keras.utils.set_random_seed(11)
    expected = random.random()
    keras.utils.set_random_seed(11)
    build()
    return random.random() != expected


def test_the_strategy_list_is_the_literal():
    assert len(ALL_STRATEGIES) == 18
    assert set(KERNEL_STRATEGIES) <= set(ALL_STRATEGIES)


@pytest.mark.parametrize("strategy", NO_KERNEL)
def test_a_strategy_without_a_kernel_draws_nothing(strategy):
    assert not _draws(lambda: SequencePooling(strategy=strategy)), (
        f"SequencePooling(strategy={strategy!r}) consumed a global RNG draw")


def test_a_weighted_sum_of_kernelless_strategies_draws_nothing():
    def build():
        layer = SequencePooling(strategy=["mean", "max", "weighted"],
                                aggregation_method="weighted_sum")
        layer.build((None, 6, 8))
    assert not _draws(build)


@pytest.mark.parametrize("strategy", KERNEL_STRATEGIES)
def test_a_kernel_strategy_still_draws_its_seed(strategy):
    """Liveness: the probe sees a draw where one is expected."""
    assert _draws(lambda: SequencePooling(strategy=strategy, attention_hidden_dim=8))


@pytest.mark.parametrize("strategy", ["mean", ["mean", "max"]])
def test_the_config_round_trips_the_initializer_it_was_given(strategy):
    for given in ("glorot_uniform", "he_normal",
                  keras.initializers.TruncatedNormal(stddev=0.3, seed=5)):
        layer = SequencePooling(strategy=strategy, kernel_initializer=given)
        rebuilt = SequencePooling.from_config(layer.get_config())
        assert rebuilt.get_config() == layer.get_config()


def test_an_attention_pooler_still_uses_the_given_initializer():
    keras.utils.set_random_seed(0)
    layer = SequencePooling(strategy="attention", attention_hidden_dim=8,
                            kernel_initializer="zeros")
    layer(np.ones((2, 5, 8), dtype="float32"))
    kernels = [w for w in layer.weights if "kernel" in w.path]
    assert kernels and all(
        not np.any(keras.ops.convert_to_numpy(w)) for w in kernels)
