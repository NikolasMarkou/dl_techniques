"""No two weights of a freshly built attention pooler replay one initializer draw.

Guard for plan-2026-10-14T062616-81938e81 step 5.2 (W4, D-201). A seedless
``keras.initializers.Initializer`` INSTANCE handed to several weights replays
one sample at every site, and a shorter draw comes out as the longer draw's
prefix (``initializers/clone.py``). ``AttentionPooling`` handed one resolved
instance to its Dense kernel and to ``context_vector``, and ``SequencePooling``
handed one instance to every ``AttentionPooling`` it built. Measured at
bfbded6f0 with the instrument below: ``context_vector ~ attention_transform/kernel
(r=1.000000)`` at seeds 1, 2 and 3 and at 4 heads, and all 6 weight pairs of
``SequencePooling(['attention', 'multi_head_attention'])`` tied.
"""

import keras
import numpy as np
import pytest

from ..test_models.initializer_replay_oracle import replayed_pairs
from dl_techniques.layers.sequence_pooling import AttentionPooling, SequencePooling

D = 32
SEEDS = (1, 2, 3)


def _model(layer) -> keras.Model:
    inputs = keras.Input((6, D))
    return keras.Model(inputs, layer(inputs))


SUBJECTS = {
    # name: (builder, pairs the instrument must compare)
    "attention_1_head": (lambda: AttentionPooling(hidden_dim=32, num_heads=1), 1),
    "attention_4_heads": (lambda: AttentionPooling(hidden_dim=32, num_heads=4), 1),
    "attention_string_init": (
        lambda: AttentionPooling(hidden_dim=32, kernel_initializer="glorot_uniform"), 1
    ),
    "sequence_attention_and_multi_head": (
        lambda: SequencePooling(
            strategy=["attention", "multi_head_attention"],
            attention_hidden_dim=32,
            attention_num_heads=4,
        ),
        6,
    ),
    "sequence_instance_init": (
        lambda: SequencePooling(
            strategy=["attention", "multi_head_attention"],
            attention_hidden_dim=32,
            attention_num_heads=4,
            kernel_initializer=keras.initializers.GlorotUniform(),
        ),
        6,
    ),
}


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("subject", sorted(SUBJECTS))
def test_no_two_pooling_weights_replay_one_draw(subject, seed):
    build, expected_pairs = SUBJECTS[subject]
    keras.utils.set_random_seed(seed)
    replayed, compared = replayed_pairs(_model(build()))
    # Anti-vacuity floor: every kernel and context vector was compared.
    assert compared == expected_pairs, (subject, compared)
    assert replayed == [], replayed


def test_the_detector_sees_a_planted_shared_instance():
    """Positive control: one seedless instance at the two pooler shapes is caught."""
    keras.utils.set_random_seed(1)
    shared = keras.initializers.get("glorot_uniform")
    inputs = keras.Input((6, D))
    kernel = keras.layers.Dense(32, kernel_initializer=shared, name="kernel_site")
    hidden = kernel(inputs)
    context = keras.layers.Dense(1, kernel_initializer=shared, name="context_site")
    model = keras.Model(inputs, context(hidden))
    replayed, compared = replayed_pairs(model)
    assert compared == 1
    assert len(replayed) == 1, replayed


def test_a_seeded_initializer_still_reproduces_a_build():
    """Cloning keeps an explicit seed: two builds with one seeded spec match."""
    def build():
        layer = AttentionPooling(
            hidden_dim=8, kernel_initializer=keras.initializers.GlorotUniform(seed=7)
        )
        layer.build((None, 6, D))
        return [np.asarray(w) for w in layer.weights]

    for a, b in zip(build(), build()):
        np.testing.assert_array_equal(a, b)
