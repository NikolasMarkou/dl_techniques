"""Guards for the remaining mechanism-blind layers.

Completes the set begun in ``test_the_defining_mechanisms_are_live.py``:

    tripse_attention        TripSE3 topology / TripSE4 logit fusion    0 / 66
    wave_field_attention    disable the FFT wave convolution           3 / 64
    capsule_routing         force routing_iterations to 1              0 / 65

``kimi_delta_attention`` is not blind: its NumPy oracle catches each deletion
(``beta = 0`` 21 / 109, decay forced to 1 9 / 109, output gate bypassed
13 / 109, short convolution bypassed 14 / 109). Its rows below pin each
mechanism by name, so a failure says which one broke rather than only that the
output moved off the oracle.

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
from dl_techniques.layers.attention.kimi_delta_attention import (
    KimiDeltaAttention,
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


# --------------------------------------------------------------------------
# KimiDeltaAttention -- the delta rule, the decay, the gate and the short conv
# --------------------------------------------------------------------------
def _kimi(tokens, mode="chunk", **kwargs):
    """A built ``KimiDeltaAttention`` at a fixed seed; weights are edited in place."""
    keras.utils.set_random_seed(0)
    layer = KimiDeltaAttention(dim=64, num_heads=4, mode=mode, **kwargs)
    layer(tokens, training=False)
    return layer


def _kimi_tokens():
    return np.random.default_rng(0).normal(size=(2, 32, 64)).astype("float32")


def test_the_kimi_delta_rule_writes_the_error_not_the_value():
    """The delta rule writes ``beta * (v - S^T k)``, so a repeated write saturates.

    Feed one token 64 times with the decay switched off (``A_log = -30``, so
    ``alpha = 1``), ``beta = 1/2`` on every head (zero ``b_proj`` kernel) and no
    output norm. Per head the state is ``c_t k v^T`` with
    ``c_t = c_{t-1} + beta (1 - c_{t-1})``, which converges to 1: the readout
    after 64 writes equals the readout after 32 (relative change 0.5 ** 28).
    A Hebbian write (``beta * v``, no error term) grows ``c_t = t * beta``,
    doubling the readout; ``beta = 0`` writes nothing and reads zero. Both
    modes, since each has its own state update.

    Measured on HEAD: the readout at token 31 is 0.045 in both modes, and the
    relative change from token 31 to token 63 is 1.2e-7 (chunk) and 4.1e-8
    (recurrent).
    """
    constant = np.repeat(_kimi_tokens()[:, :1, :], 64, axis=1)
    for mode in ("chunk", "recurrent"):
        layer = _kimi(constant, mode=mode, output_norm_type=None)
        layer.A_log.assign(np.full(layer.A_log.shape, -30.0, "float32"))
        layer.b_proj.kernel.assign(np.zeros(layer.b_proj.kernel.shape, "float32"))
        out = np.asarray(layer(constant, training=False))

        early = np.abs(out[:, 31]).max()
        assert early > 1e-3, (
            f"{mode}: the readout after 32 identical writes is {early}: nothing "
            "was written to the state, so beta never reaches it"
        )
        change = np.abs(out[:, 63] - out[:, 31]).max() / early
        assert change < 1e-3, (
            f"{mode}: 32 more writes of the same association moved the readout "
            f"by {change:.3f} of its size. The delta rule writes only the error "
            "v - S^T k, which is zero once the association is stored; a write "
            "that keeps growing is a Hebbian (plain linear attention) write"
        )


def test_the_kimi_decay_forgets_and_both_of_its_parameters_reach_it():
    """``alpha_t = exp(-exp(A_log) * softplus(f + dt_bias))`` makes the state forget.

    Perturb token 0 and read the last token, 31 steps later. With the decay off
    (``A_log = -30``) the perturbation is still there; with a strong decay
    (``A_log = 5``) it has decayed away. ``dt_bias`` is the other half of the
    same rate, so moving it must move the output. With the decay forced to 1
    the two arms are equal and ``dt_bias`` is inert.

    Measured on HEAD, max|delta| at token 31: 1.23 (decay off), 0.0039
    (A_log = 5); ``dt_bias + 3`` moves the output by 1.75.
    """
    tokens = _kimi_tokens()
    perturbed = tokens.copy()
    perturbed[:, 0, :] += 20.0

    reach = {}
    for a_log in (-30.0, 5.0):
        layer = _kimi(tokens)
        layer.A_log.assign(np.full(layer.A_log.shape, a_log, "float32"))
        before = np.asarray(layer(tokens, training=False))
        after = np.asarray(layer(perturbed, training=False))
        reach[a_log] = np.abs(after[:, -1] - before[:, -1]).max()

    assert reach[-30.0] > 0.1, (
        f"with the decay off, token 0 moved token 31 by only {reach[-30.0]}: "
        "the state is not carrying information along the sequence"
    )
    assert reach[5.0] < 0.05 * reach[-30.0], (
        f"a strong decay left token 0's effect on token 31 at {reach[5.0]} "
        f"against {reach[-30.0]} with the decay off: A_log does not reach the "
        "state, so the layer never forgets"
    )

    layer = _kimi(tokens)
    before = np.asarray(layer(tokens, training=False))
    layer.dt_bias.assign(layer.dt_bias.numpy() + 3.0)
    delta = np.abs(np.asarray(layer(tokens, training=False)) - before).max()
    assert delta > 1e-3, (
        f"moving dt_bias by 3 moved the output by only {delta}: dt_bias does "
        "not reach the decay"
    )


def test_the_kimi_output_gate_and_short_conv_are_live():
    """The sigmoid output gate and the q/k/v short convolutions reach the output.

    Raising the gate bias, or reversing the three convolution kernels in time,
    must move the output. With the gate multiply or the convolution bypassed,
    that weight is decorative and the delta is exactly 0.0.

    Measured on HEAD: gate bias + 3 moves the output by 1.61, the reversed
    kernels by 2.72.
    """
    tokens = _kimi_tokens()

    layer = _kimi(tokens)
    before = np.asarray(layer(tokens, training=False))
    layer.g_up.bias.assign(layer.g_up.bias.numpy() + 3.0)
    gate_delta = np.abs(np.asarray(layer(tokens, training=False)) - before).max()
    assert gate_delta > 1e-3, (
        f"raising the output gate bias moved the output by only {gate_delta}: "
        "the gate is computed and never applied"
    )

    layer = _kimi(tokens)
    before = np.asarray(layer(tokens, training=False))
    for conv in (layer.q_conv, layer.k_conv, layer.v_conv):
        conv.kernel.assign(np.flip(conv.kernel.numpy(), axis=0).copy())
    conv_delta = np.abs(np.asarray(layer(tokens, training=False)) - before).max()
    assert conv_delta > 1e-3, (
        f"reversing the short-convolution kernels moved the output by only "
        f"{conv_delta}: q, k and v are not passing through the convolution"
    )
