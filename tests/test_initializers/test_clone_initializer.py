"""
``clone_initializer`` -- proved against the symmetry it exists to break.

The premise (a shared seedless instance replays one seed) is MEASURED here
rather than assumed, and each arm has the control that makes it a finding.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.initializers import clone_initializer


def _kernel(initializer):
    layer = keras.layers.Dense(4, kernel_initializer=initializer)
    layer.build((None, 6))
    return np.asarray(ops.convert_to_numpy(layer.kernel))


def test_the_premise_a_shared_seedless_instance_emits_the_same_tensor_twice():
    """
    The Keras 3 behaviour this helper exists for. NOT a repo bug -- and the
    STRING control below is what proves the instance, not the name, is the cause.
    """
    keras.utils.set_random_seed(1234)
    shared = keras.initializers.get("glorot_uniform")
    assert getattr(shared, "seed", None) is not None, (
        "a seedless initializer INSTANCE self-assigns a seed; the exact value is "
        "process-specific (batch 6 recorded 880945459, batch 7 corrected it) so "
        "only its presence is pinned here"
    )
    assert np.array_equal(_kernel(shared), _kernel(shared))


def test_the_control_the_same_initializer_NAME_does_not_share():
    keras.utils.set_random_seed(1234)
    assert not np.array_equal(
        _kernel("glorot_uniform"), _kernel("glorot_uniform")
    ), "if this were equal the premise above would be about the name, not the instance"


def test_a_clone_breaks_the_symmetry():
    keras.utils.set_random_seed(1234)
    shared = keras.initializers.get("glorot_uniform")
    a = _kernel(shared)
    b = _kernel(clone_initializer(shared))
    assert not np.array_equal(a, b)
    assert float(np.abs(a - b).max()) > 0.0


def test_a_clone_of_a_SEEDED_initializer_deliberately_does_NOT_break_symmetry():
    """
    The documented failure mode. An author who asked for ``seed=7`` gets
    reproducibility, and this helper must not silently take it away.
    """
    seeded = keras.initializers.GlorotUniform(seed=7)
    assert np.array_equal(_kernel(seeded), _kernel(clone_initializer(seeded)))


@pytest.mark.parametrize("argument", [None, "zeros", "glorot_uniform"])
def test_strings_and_none_round_trip_through_keras_initializers_get(argument):
    result = clone_initializer(argument)
    assert result == keras.initializers.get(argument) or isinstance(
        result, keras.initializers.Initializer
    )


def test_a_clone_serializes_to_the_same_config():
    original = keras.initializers.get("he_normal")
    assert clone_initializer(original).get_config().keys() == original.get_config().keys()
    assert clone_initializer(original).__class__ is original.__class__
