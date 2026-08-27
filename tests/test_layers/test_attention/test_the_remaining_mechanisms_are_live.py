"""Guards for the remaining mechanism-blind layers.

Completes the set begun in ``test_the_defining_mechanisms_are_live.py``:

    tripse_attention        TripSE3 topology / TripSE4 logit fusion    0 / 66
    wave_field_attention    disable the FFT wave convolution           3 / 64
    capsule_routing         force routing_iterations to 1              0 / 65

Same principle throughout: assert a property derivable without reading the
implementation, and copy weights whenever two separately-constructed layers are
compared -- otherwise the delta measures initialization rather than mechanism
(decisions.md D-020).
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.capsule_routing_attention import (
    CapsuleRoutingSelfAttention,
)
from dl_techniques.layers.attention.tripse_attention import (
    TripSE1,
    TripSE2,
    TripSE3,
    TripSE4,
)
from dl_techniques.layers.attention.wave_field_attention import (
    WaveFieldAttention,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")

VARIANTS = {"TripSE1": TripSE1, "TripSE2": TripSE2, "TripSE3": TripSE3,
            "TripSE4": TripSE4}


@pytest.fixture(name="feature_map")
def _feature_map():
    return (
        np.random.default_rng(0).normal(size=(2, 8, 8, 16)).astype("float32")
    )


def test_the_four_tripse_variants_are_not_the_same_layer(feature_map):
    """Four factory keys must not be four names for one computation.

    TripSE3's parallel topology, TripSE4's logit-domain fusion and `_SEWeights`'s
    no-sigmoid contract are each deletable with 0 of 66 tests failing, so nothing
    pinned what distinguishes the variants. If any pair were accidentally
    equivalent the package would ship a distinction that does not exist.

    Measured pairwise maxima on HEAD (same seed, same input): the closest pair is
    TripSE1/TripSE4 at 0.115, every other pair above 0.27.
    """
    outputs = {}
    for name, cls in VARIANTS.items():
        keras.utils.set_random_seed(0)
        outputs[name] = np.asarray(cls()(feature_map, training=False))

    names = sorted(outputs)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            delta = np.abs(outputs[first] - outputs[second]).max()
            assert delta > 1e-3, (
                f"{first} and {second} produce the same output (max|delta| = "
                f"{delta}): two of the four registered TripSE variants are the "
                "same computation under different names"
            )


def test_the_wave_field_propagates_forward_and_decays(feature_map):
    """The FFT wave convolution is what carries influence along the sequence.

    Disabling it passes 61 of 64 tests. The property is directional and
    distance-dependent: perturbing an EARLY token must move LATER positions, by
    an amount that falls off with distance. Perturbing the LAST token must move
    nothing earlier, which is the layer's documented causality.

    Measured on HEAD after perturbing token 0 (delta at positions 0, 1, 8, 16,
    31): 303.98, 28.09, 4.00, 0.82, 0.045 -- a decaying forward wave.
    """
    tokens = np.random.default_rng(0).normal(size=(2, 32, 64)).astype("float32")
    keras.utils.set_random_seed(0)
    layer = WaveFieldAttention(dim=64, num_heads=4)
    base = np.asarray(layer(tokens, training=False))

    forward = tokens.copy()
    forward[:, 0, :] += 20.0
    forward_delta = np.abs(
        np.asarray(layer(forward, training=False)) - base
    ).max(axis=-1)[0]

    assert forward_delta[1] > 1e-3, (
        f"perturbing token 0 moved token 1 by only {forward_delta[1]}: the wave "
        "convolution is not propagating along the sequence"
    )
    assert forward_delta[8] > 1e-4, (
        "the wave does not reach position 8, so its range has collapsed"
    )
    assert forward_delta[1] > forward_delta[16], (
        "influence does not decay with distance, so this is not a wave field"
    )

    backward = tokens.copy()
    backward[:, -1, :] += 20.0
    backward_delta = np.abs(
        np.asarray(layer(backward, training=False)) - base
    ).max(axis=-1)[0]
    assert backward_delta[:16].max() < 1e-3, (
        f"perturbing the LAST token moved earlier positions by "
        f"{backward_delta[:16].max()}: the field is leaking backwards"
    )


def test_more_routing_iterations_change_the_result():
    """Dynamic routing must actually iterate.

    Hard-coding the loop to one pass leaves all 65 tests green. Weights are
    copied across so this measures the iteration count and not initialization.
    """
    tokens = np.random.default_rng(0).normal(size=(2, 8, 32)).astype("float32")

    keras.utils.set_random_seed(0)
    single = CapsuleRoutingSelfAttention(
        num_heads=4, key_dim=8, routing_iterations=1
    )
    first = np.asarray(single(tokens, training=False))

    keras.utils.set_random_seed(0)
    triple = CapsuleRoutingSelfAttention(
        num_heads=4, key_dim=8, routing_iterations=3
    )
    triple(tokens, training=False)  # build before copying
    triple.set_weights(single.get_weights())
    second = np.asarray(triple(tokens, training=False))

    delta = np.abs(first - second).max()
    assert delta > 1e-3, (
        f"1 and 3 routing iterations differ by only {delta}: the routing loop is "
        "not refining the coupling coefficients"
    )
