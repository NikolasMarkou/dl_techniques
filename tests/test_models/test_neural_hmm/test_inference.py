"""``neural_hmm.inference``: exact HMM inference against brute-force path enumeration.

Guards:

* the four inference functions (``forward_log_likelihood``, ``path_log_score``,
  ``viterbi_decode``, ``state_posteriors``) equal an independent float64 numpy
  enumeration over all ``K**L`` state paths, for every ``K`` in {1, 2, 3}, ``T`` in
  {1, ..., 5}, a static (broadcast) and a per-step ``log_A``, and right-padded masks of
  lengths ``T``, 1 and 0 in one batch (SC1);
* the same in float32, within a bound derived from ``tests.numerics.reassociation_atol``;
* padding semantics (D-009): whatever the padded positions hold, a padded sequence's
  ``logZ`` and Viterbi output are bit-identical to the truncated sequence's (path score
  and posteriors to float64 reassociation), padded Viterbi entries are
  ``-1``, padded posteriors are exactly 0, and an empty sequence has ``logZ = 0`` and
  path score 0;
* a state outside ``[0, K)`` at a VALID step adds ``mask_sentinel`` to the path score
  per such step (D-022); at a padded step it stays ignored;
* the gradient of ``logZ`` with respect to the emissions equals the brute-force
  posteriors, also when compiled with XLA and traced at a second sequence length;
* the functions run with an unknown batch and time axis: in a ``tf.function`` with a
  ``None`` signature, in a ``reduce_retracing`` function fed three lengths (how Keras
  traces ``predict``; D-014), and inside a ``keras.Input`` functional trace (SC2);
* the package source uses no ``keras.ops.tril``/``triu``/``take_along_axis`` and no
  ``from keras import ops`` (AST scan, SC2);
* float32 stays accurate on long sequences (D-001 of the rescaling plan): posteriors, the
  Viterbi path and ``logZ`` agree with float64 at ``T`` in the thousands, ergodic and banded.
"""

import ast
import itertools
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.probabilistic.neural_hmm import inference
from dl_techniques.utils.dtype_policy import mask_sentinel
from tests.numerics import reassociation_atol
from tests.layer_naming import PublicAutoName

# ---------------------------------------------------------------------
# Brute-force oracle (float64 numpy, independent of the code under test)
# ---------------------------------------------------------------------

#: Float64 tolerance. SC1 asks for 1e-5; the measured float64 error is ~1e-15, so the
#: tighter bound costs nothing and catches smaller defects.
ATOL_F64 = 1e-9


def _path_score(log_pi, log_A, log_emit, path):
    """Score one path of one sequence: ``log_pi[z0] + sum emissions + transitions INTO t``."""
    score = log_pi[path[0]] + log_emit[0, path[0]]
    for t in range(1, len(path)):
        score += log_A[t, path[t - 1], path[t]] + log_emit[t, path[t]]
    return score


def _logsumexp(values):
    peak = np.max(values)
    return peak + np.log(np.sum(np.exp(values - peak)))


def brute_force(log_pi, log_A, log_emit, lengths):
    """Enumerate every state path of each sequence's valid prefix.

    :return: ``(log_z (B,), best_path (B, T) with -1 padding, best_score (B,),
        posteriors (B, T, K) with 0 padding)``. An empty sequence has ``log_z = 0``,
        an all ``-1`` path, score 0 and zero posteriors.
    """
    batch, steps, n_states = log_emit.shape
    log_z = np.zeros(batch)
    best_path = np.full((batch, steps), -1, dtype=np.int64)
    best_score = np.zeros(batch)
    posteriors = np.zeros((batch, steps, n_states))
    for b in range(batch):
        length = int(lengths[b])
        if length == 0:
            continue
        paths = list(itertools.product(range(n_states), repeat=length))
        scores = np.array([_path_score(log_pi[b], log_A[b], log_emit[b], p) for p in paths])
        log_z[b] = _logsumexp(scores)
        top = int(np.argmax(scores))
        best_path[b, :length] = paths[top]
        best_score[b] = scores[top]
        weights = np.exp(scores - log_z[b])
        for path, weight in zip(paths, weights):
            posteriors[b, np.arange(length), list(path)] += weight
    return log_z, best_path, best_score, posteriors


def _make_case(n_states, steps, transition, dtype="float64"):
    """Random parameters, a mask of lengths (T, 1, 0) and a state path with garbage padding."""
    rng = np.random.default_rng(1000 * n_states + 10 * steps + (transition == "dynamic"))
    batch = 3
    log_pi = 2.0 * rng.normal(size=(batch, n_states))
    if transition == "static":
        log_A = np.broadcast_to(
            2.0 * rng.normal(size=(1, 1, n_states, n_states)), (batch, steps, n_states, n_states)
        ).copy()
    else:
        log_A = 2.0 * rng.normal(size=(batch, steps, n_states, n_states))
    log_emit = 2.0 * rng.normal(size=(batch, steps, n_states))
    lengths = np.array([steps, 1, 0])
    mask = (np.arange(steps)[None, :] < lengths[:, None]).astype(dtype)
    states = rng.integers(0, n_states, size=(batch, steps))
    states = np.where(mask > 0, states, np.array([-1, n_states + 3, 0])[:, None])
    arrays = [a.astype(dtype) for a in (log_pi, log_A, log_emit)]
    return arrays + [mask, states, lengths]


CASES = [
    pytest.param(k, t, mode, id=f"K{k}-T{t}-{mode}")
    for k in (1, 2, 3)
    for t in (1, 2, 3, 4, 5)
    for mode in ("static", "dynamic")
]


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _f64_reassociation_atol(n_terms, scale):
    """``tests.numerics.reassociation_atol``'s formula with the float64 unit roundoff.

    ``8 * sqrt(2 * n_terms) * u64 * max(1, scale)``: two evaluations of one formula that
    differ only in summation grouping, an 8-sigma random walk over ``n_terms`` roundings.
    """
    return 8.0 * np.sqrt(2.0 * n_terms) * (np.finfo(np.float64).eps / 2.0) * max(1.0, float(scale))


# ---------------------------------------------------------------------
# SC1: oracle agreement, float64
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n_states, steps, transition", CASES)
def test_forward_log_likelihood_matches_brute_force(n_states, steps, transition):
    log_pi, log_A, log_emit, mask, _, lengths = _make_case(n_states, steps, transition)
    expected, _, _, _ = brute_force(log_pi, log_A, log_emit, lengths)
    got = _np(inference.forward_log_likelihood(log_pi, log_A, log_emit, mask))
    np.testing.assert_allclose(got, expected, rtol=0, atol=ATOL_F64)


@pytest.mark.parametrize("n_states, steps, transition", CASES)
def test_path_log_score_matches_brute_force(n_states, steps, transition):
    log_pi, log_A, log_emit, mask, states, lengths = _make_case(n_states, steps, transition)
    expected = np.array([
        _path_score(log_pi[b], log_A[b], log_emit[b], states[b, : lengths[b]]) if lengths[b] else 0.0
        for b in range(len(lengths))
    ])
    got = _np(inference.path_log_score(log_pi, log_A, log_emit, states, mask))
    np.testing.assert_allclose(got, expected, rtol=0, atol=ATOL_F64)


@pytest.mark.parametrize("n_states, steps, transition", CASES)
def test_viterbi_decode_matches_brute_force(n_states, steps, transition):
    log_pi, log_A, log_emit, mask, _, lengths = _make_case(n_states, steps, transition)
    _, best_path, best_score, _ = brute_force(log_pi, log_A, log_emit, lengths)
    path, score = inference.viterbi_decode(log_pi, log_A, log_emit, mask)
    np.testing.assert_array_equal(_np(path), best_path)
    np.testing.assert_allclose(_np(score), best_score, rtol=0, atol=ATOL_F64)
    # The decoded path's own score is the Viterbi score.
    rescored = _np(inference.path_log_score(log_pi, log_A, log_emit, path, mask))
    np.testing.assert_allclose(rescored, best_score, rtol=0, atol=ATOL_F64)


@pytest.mark.parametrize("n_states, steps, transition", CASES)
def test_state_posteriors_match_brute_force(n_states, steps, transition):
    log_pi, log_A, log_emit, mask, _, lengths = _make_case(n_states, steps, transition)
    _, _, _, expected = brute_force(log_pi, log_A, log_emit, lengths)
    got = _np(inference.state_posteriors(log_pi, log_A, log_emit, mask))
    np.testing.assert_allclose(got, expected, rtol=0, atol=ATOL_F64)


# ---------------------------------------------------------------------
# SC1: oracle agreement, float32
# ---------------------------------------------------------------------


def _assert_float32_matches_brute_force(got, case, n_states, steps):
    """Compare float32 ``(logZ, path, score, posteriors)`` with the oracle on the SAME inputs.

    The oracle runs in float64 on the float32-rounded inputs, so only the float32
    arithmetic is measured. Each step runs a logsumexp of length K forward and one
    backward, so the reduction lengths are ``[K, K]`` over ``T`` steps. Log-space outputs
    are bounded at their own magnitude; a posterior ``exp(alpha + beta - logZ)`` is at
    most 1, and its absolute error is the absolute error of its exponent, which is
    bounded at the magnitude of ``logZ``.
    """
    log_pi, log_A, log_emit, _, _, lengths = case
    log_z, best_path, best_score, post = brute_force(
        log_pi.astype("float64"), log_A.astype("float64"), log_emit.astype("float64"), lengths
    )
    got_z, got_path, got_score, got_post = (_np(g) for g in got)
    assert got_z.dtype == np.float32 and got_post.dtype == np.float32
    reductions = [n_states, n_states]
    z_atol = reassociation_atol(reductions, steps, np.abs(log_z).max())
    np.testing.assert_allclose(got_z, log_z, rtol=0, atol=z_atol)
    np.testing.assert_array_equal(got_path, best_path)
    np.testing.assert_allclose(
        got_score, best_score, rtol=0, atol=reassociation_atol(reductions, steps, np.abs(best_score).max())
    )
    np.testing.assert_allclose(got_post, post, rtol=0, atol=z_atol)


def _run_all_four(log_pi, log_A, log_emit, mask):
    path, score = inference.viterbi_decode(log_pi, log_A, log_emit, mask)
    return (
        inference.forward_log_likelihood(log_pi, log_A, log_emit, mask),
        path,
        score,
        inference.state_posteriors(log_pi, log_A, log_emit, mask),
    )


@pytest.mark.parametrize("n_states, steps, transition", CASES)
def test_float32_inference_matches_brute_force_within_the_reassociation_bound(
    n_states, steps, transition
):
    case = _make_case(n_states, steps, transition, dtype="float32")
    got = _run_all_four(*case[:4])
    _assert_float32_matches_brute_force(got, case, n_states, steps)


# ---------------------------------------------------------------------
# Long sequences in float32 (rescaling)
# ---------------------------------------------------------------------


def _long_chain(steps, emission_offset, topology):
    """Return ``(log_pi, log_A, log_emit)`` of one long sequence, as float64 numpy arrays.

    :param steps: Sequence length ``T``.
    :param emission_offset: Added to every log-emission; it sets how fast an unscaled
        message grows, as a higher-dimensional emission density would.
    :param topology: ``ergodic``, or ``banded`` with sentinel-valued forbidden entries.
    :return: Arrays of shape ``(1, K)``, ``(1, T, K, K)`` and ``(1, T, K)``.
    """
    rng = np.random.default_rng(0)
    n_states = 4
    sentinel = float(mask_sentinel("float32"))
    if topology == "ergodic":
        transition = np.full((n_states, n_states), 0.1 / (n_states - 1))
        np.fill_diagonal(transition, 0.9)
        log_A = np.log(transition)
        log_pi = np.log(np.full(n_states, 1.0 / n_states))
        states = np.zeros(steps, dtype=int)
        for t in range(1, steps):
            states[t] = rng.choice(n_states, p=transition[states[t - 1]])
    else:
        log_A = np.full((n_states, n_states), sentinel)
        for i in range(n_states):
            log_A[i, i] = 0.0 if i == n_states - 1 else np.log(0.999)
            if i < n_states - 1:
                log_A[i, i + 1] = np.log(0.001)
        log_pi = np.array([0.0] + [sentinel] * (n_states - 1))
        states = np.minimum(np.arange(steps) * n_states // steps, n_states - 1)
    distance = rng.normal(size=(steps, 1)) + (states[:, None] - np.arange(n_states)[None]) * 2.0
    log_emit = -0.5 * distance**2 + emission_offset
    return log_pi[None], np.broadcast_to(log_A, (1, steps, n_states, n_states)).copy(), log_emit[None]


def _as_dtype(arrays, dtype):
    return [tf.constant(a, dtype=dtype) for a in arrays] + [tf.ones(arrays[2].shape[:2], dtype=dtype)]


@pytest.mark.parametrize("topology", ["ergodic", "banded"])
@pytest.mark.parametrize("steps", [1000, 5000])
@pytest.mark.parametrize("emission_offset", [-3.0, -30.0])
def test_float32_posteriors_match_float64_on_a_long_sequence(steps, emission_offset, topology):
    arrays = _long_chain(steps, emission_offset, topology)
    run = lambda dtype: tf.function(lambda: inference.state_posteriors(*_as_dtype(arrays, dtype)))().numpy()
    reference = run(tf.float64)
    got = run(tf.float32)
    # Unscaled messages measured a largest error of 0.02 to 36 here. Rescaled: below 1e-5.
    np.testing.assert_allclose(got, reference, rtol=0, atol=2e-5)
    np.testing.assert_allclose(got.sum(axis=-1), 1.0, rtol=0, atol=2e-5)


@pytest.mark.parametrize("topology", ["ergodic", "banded"])
def test_float32_viterbi_matches_float64_at_twenty_thousand_steps(topology):
    arrays = _long_chain(20000, -30.0, topology)
    run = lambda dtype: tf.function(lambda: inference.viterbi_decode(*_as_dtype(arrays, dtype)))()
    reference_path, _ = run(tf.float64)
    path, score = run(tf.float32)
    # The unscaled delta measured 0.075% of the steps decoded differently here (ergodic).
    np.testing.assert_array_equal(path.numpy(), reference_path.numpy())
    float64_score = inference.path_log_score(
        *_as_dtype(arrays[:3], tf.float64)[:3], reference_path, tf.ones_like(reference_path, tf.float64)
    )
    # A float32 running total of 6e5 is only good to its accumulated ulps.
    np.testing.assert_allclose(score.numpy(), float64_score.numpy(), rtol=0, atol=64.0)


def test_rescaling_leaves_the_log_partition_function_and_its_gradient_correct_when_long():
    arrays = _long_chain(2000, -30.0, "ergodic")

    def value_and_gradient(dtype):
        log_pi, log_A, log_emit, mask = _as_dtype(arrays, dtype)
        with tf.GradientTape() as tape:
            tape.watch(log_emit)
            log_z = inference.forward_log_likelihood(log_pi, log_A, log_emit, mask)
        return log_z.numpy(), tape.gradient(log_z, log_emit).numpy()

    reference_z, reference_grad = value_and_gradient(tf.float64)
    got_z, got_grad = value_and_gradient(tf.float32)
    # d logZ / d log_emit is the posterior. A running total near 6e4 in float32 has ulp 0.004.
    np.testing.assert_allclose(got_z, reference_z, rtol=0, atol=1.0)
    np.testing.assert_allclose(got_grad, reference_grad, rtol=0, atol=2e-5)


def test_a_sequence_with_every_path_impossible_keeps_an_infinite_log_z_not_nan():
    n_states, steps = 2, 3
    log_pi = tf.fill((1, n_states), -np.inf)
    log_A = tf.zeros((1, steps, n_states, n_states))
    log_emit = tf.zeros((1, steps, n_states))
    log_z = inference.forward_log_likelihood(log_pi, log_A, log_emit, tf.ones((1, steps))).numpy()
    assert np.isneginf(log_z).all()


# ---------------------------------------------------------------------
# Padding semantics (D-009)
# ---------------------------------------------------------------------


@pytest.mark.parametrize("transition", ["static", "dynamic"])
@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_a_padded_sequence_equals_the_truncated_sequence_bit_for_bit(length, transition):
    """Padded positions hold large garbage; the valid prefix must not see any of it."""
    log_pi, log_A, log_emit, _, states, _ = _make_case(3, 5, transition)
    steps = 5
    mask = np.zeros((3, steps))
    mask[:, :length] = 1.0
    log_emit[:, length:] = 1e3
    log_A[:, length:] = -1e3
    states[:, length:] = 7

    full = {
        "z": inference.forward_log_likelihood(log_pi, log_A, log_emit, mask),
        "score": inference.path_log_score(log_pi, log_A, log_emit, states, mask),
        "viterbi": inference.viterbi_decode(log_pi, log_A, log_emit, mask),
        "post": inference.state_posteriors(log_pi, log_A, log_emit, mask),
    }
    cut = (log_pi, log_A[:, :length], log_emit[:, :length])
    cut_mask = mask[:, :length]
    short = {
        "z": inference.forward_log_likelihood(*cut, cut_mask),
        "score": inference.path_log_score(*cut, states[:, :length], cut_mask),
        "viterbi": inference.viterbi_decode(*cut, cut_mask),
        "post": inference.state_posteriors(*cut, cut_mask),
    }

    # The recursions freeze across padding, so logZ and Viterbi are bit-identical.
    np.testing.assert_array_equal(_np(full["z"]), _np(short["z"]))
    np.testing.assert_array_equal(_np(full["viterbi"][1]), _np(short["viterbi"][1]))
    path = _np(full["viterbi"][0])
    np.testing.assert_array_equal(path[:, :length], _np(short["viterbi"][0]))
    np.testing.assert_array_equal(path[:, length:], -1)
    # The path score is a vectorized sum over T (or L) terms and the posteriors an
    # elementwise exp over a T- (or L-) long axis, so the two shapes reassociate; the
    # garbage above would move them by ~1e3.
    score = _np(full["score"])
    np.testing.assert_allclose(
        score, _np(short["score"]), rtol=0, atol=_f64_reassociation_atol(2 * steps, np.abs(score).max())
    )
    post = _np(full["post"])
    np.testing.assert_allclose(
        post[:, :length], _np(short["post"]), rtol=0,
        atol=_f64_reassociation_atol(2 * steps, np.abs(_np(full["z"])).max()),
    )
    assert np.all(post[:, length:] == 0.0)


def test_an_empty_sequence_has_log_z_zero_score_zero_path_minus_one_and_zero_posteriors():
    log_pi, log_A, log_emit, _, states, _ = _make_case(3, 4, "dynamic")
    mask = np.zeros((3, 4))
    path, score = inference.viterbi_decode(log_pi, log_A, log_emit, mask)
    assert np.all(_np(inference.forward_log_likelihood(log_pi, log_A, log_emit, mask)) == 0.0)
    assert np.all(_np(inference.path_log_score(log_pi, log_A, log_emit, states, mask)) == 0.0)
    assert np.all(_np(score) == 0.0)
    assert np.all(_np(path) == -1)
    assert np.all(_np(inference.state_posteriors(log_pi, log_A, log_emit, mask)) == 0.0)


@pytest.mark.parametrize("bad_state", [-1, 3, 7])
def test_an_out_of_range_state_at_a_valid_step_adds_the_sentinel(bad_state):
    """D-022: an all-zero one-hot used to drop the step's terms silently; now it is loud.

    Row 0 is fully valid and gets the bad state at one step, then at two; rows 1 and 2
    hold ``K + 3`` and ``0`` at their padded steps, which must stay ignored.
    """
    log_pi, log_A, log_emit, mask, states, _ = _make_case(3, 4, "dynamic")
    sentinel = mask_sentinel("float64")
    clean = _np(inference.path_log_score(log_pi, log_A, log_emit, states, mask))
    for bad_steps in ([2], [0, 3]):
        bad = states.copy()
        bad[0, bad_steps] = bad_state
        score = _np(inference.path_log_score(log_pi, log_A, log_emit, bad, mask))
        assert score[0] <= (len(bad_steps) - 0.5) * sentinel, (bad_steps, score[0])
        np.testing.assert_array_equal(score[1:], clean[1:])


def test_a_wrong_rank_input_is_refused_naming_the_argument():
    log_pi, log_A, log_emit, mask, _, _ = _make_case(2, 3, "static")
    with pytest.raises(ValueError, match="log_A"):
        inference.forward_log_likelihood(log_pi, log_A[:, 0], log_emit, mask)


# ---------------------------------------------------------------------
# Gradients, XLA, and unknown batch / time axes (SC2, D-014)
# ---------------------------------------------------------------------


def test_the_xla_gradient_of_log_z_is_the_posterior_at_two_sequence_lengths():
    """d logZ / d log_emit[b, t, k] = p(z_t = k | x) (0 at padding), compiled with XLA.

    Traced with ``reduce_retracing=True`` at T=4 and then T=5, so the second trace sees an
    unknown time axis, as Keras' ``evaluate`` and ``predict`` do.
    """

    @tf.function(reduce_retracing=True, jit_compile=True)
    def grad_of_log_z(log_pi, log_A, log_emit, mask):
        with tf.GradientTape() as tape:
            tape.watch(log_emit)
            total = tf.reduce_sum(inference.forward_log_likelihood(log_pi, log_A, log_emit, mask))
        return tape.gradient(total, log_emit)

    for steps in (4, 5):
        log_pi, log_A, log_emit, mask, _, lengths = _make_case(3, steps, "dynamic")
        _, _, _, posteriors = brute_force(log_pi, log_A, log_emit, lengths)
        args = [tf.constant(a) for a in (log_pi, log_A, log_emit, mask)]
        np.testing.assert_allclose(_np(grad_of_log_z(*args)), posteriors, rtol=0, atol=ATOL_F64)


def test_inference_runs_under_a_tf_function_with_unknown_batch_and_time():
    k = 3
    signature = [
        tf.TensorSpec((None, k), tf.float64),
        tf.TensorSpec((None, None, k, k), tf.float64),
        tf.TensorSpec((None, None, k), tf.float64),
        tf.TensorSpec((None, None), tf.float64),
    ]

    @tf.function(input_signature=signature)
    def run(log_pi, log_A, log_emit, mask):
        path, score = inference.viterbi_decode(log_pi, log_A, log_emit, mask)
        return (
            inference.forward_log_likelihood(log_pi, log_A, log_emit, mask),
            path,
            score,
            inference.state_posteriors(log_pi, log_A, log_emit, mask),
        )

    for steps in (2, 5):
        log_pi, log_A, log_emit, mask, _, lengths = _make_case(k, steps, "dynamic")
        log_z, best_path, best_score, posteriors = brute_force(log_pi, log_A, log_emit, lengths)
        got = run(log_pi, log_A, log_emit, mask)
        np.testing.assert_allclose(_np(got[0]), log_z, rtol=0, atol=ATOL_F64)
        np.testing.assert_array_equal(_np(got[1]), best_path)
        np.testing.assert_allclose(_np(got[2]), best_score, rtol=0, atol=ATOL_F64)
        np.testing.assert_allclose(_np(got[3]), posteriors, rtol=0, atol=ATOL_F64)


def test_a_reduce_retracing_function_accepts_a_second_and_third_sequence_length():
    """Keras traces predict/evaluate this way; from the second length on, T is None."""
    seen_lengths = []

    @tf.function(reduce_retracing=True)
    def run(log_pi, log_A, log_emit, mask):
        seen_lengths.append(log_emit.shape[1])
        return inference.forward_log_likelihood(log_pi, log_A, log_emit, mask)

    for steps in (3, 4, 5):
        log_pi, log_A, log_emit, mask, _, lengths = _make_case(2, steps, "dynamic")
        expected, _, _, _ = brute_force(log_pi, log_A, log_emit, lengths)
        np.testing.assert_allclose(_np(run(log_pi, log_A, log_emit, mask)), expected, rtol=0, atol=ATOL_F64)
    assert None in seen_lengths, f"the probe never saw an unknown time axis: {seen_lengths}"


class _InferenceLayer(PublicAutoName, keras.layers.Layer):
    """Runs all four functions, so a functional model can trace them on keras.Input."""

    def call(self, log_pi, log_A, log_emit, mask):
        return _run_all_four(log_pi, log_A, log_emit, mask)


def test_inference_traces_inside_a_keras_input_graph_and_predicts_the_oracle():
    """Float32 on purpose: ``predict``'s data adapter casts float64 arrays to floatx."""
    k, steps = 3, 5
    inputs = [keras.Input((k,)), keras.Input((None, k, k)), keras.Input((None, k)), keras.Input((None,))]
    outputs = _InferenceLayer()(*inputs)
    assert outputs[0].shape == (None,) and outputs[1].shape == (None, None)
    assert outputs[3].shape == (None, None, k)
    model = keras.Model(inputs, outputs)

    case = _make_case(k, steps, "dynamic", dtype="float32")
    got = model.predict(list(case[:4]), verbose=0, batch_size=2)
    _assert_float32_matches_brute_force(got, case, k, steps)


# ---------------------------------------------------------------------
# AST guard: no XLA-hostile constructs, house import style (SC2)
# ---------------------------------------------------------------------

PACKAGE_DIR = Path(inference.__file__).resolve().parent
_BANNED_OPS = {"tril", "triu", "take_along_axis"}


def _is_ops_namespace(node):
    """True for ``ops`` and ``<anything>.ops`` (e.g. ``keras.ops``)."""
    if isinstance(node, ast.Name):
        return node.id == "ops"
    return isinstance(node, ast.Attribute) and node.attr == "ops"


def _banned_sites(source, filename):
    sites = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_OPS and _is_ops_namespace(node.value):
            sites.append(f"{filename}:{node.lineno} ops.{node.attr}")
        if isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in node.names}
            if node.module == "keras" and "ops" in names:
                sites.append(f"{filename}:{node.lineno} from keras import ops")
            if node.module == "keras.ops":
                sites.append(f"{filename}:{node.lineno} from keras.ops import ...")
    return sites


def test_the_package_uses_no_xla_hostile_ops_and_no_from_keras_import_ops():
    files = sorted(PACKAGE_DIR.glob("*.py"))
    assert any(f.name == "inference.py" for f in files)
    sites = [s for f in files for s in _banned_sites(f.read_text(), f.name)]
    assert sites == [], sites


def test_the_ast_scan_flags_each_banned_construct():
    """Anti-vacuity: the scanner itself must see every form it claims to ban."""
    probe = (
        "import keras\nfrom keras import ops\nfrom keras.ops import tril\n"
        "a = keras.ops.tril(x)\nb = ops.triu(x)\nc = keras.ops.take_along_axis(x, i, 1)\n"
        "d = np.triu(x)\n"
    )
    sites = _banned_sites(probe, "probe.py")
    assert len(sites) == 5, sites
