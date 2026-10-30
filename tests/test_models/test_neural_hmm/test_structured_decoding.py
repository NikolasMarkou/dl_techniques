"""Structural constraints and end potentials: BIO masks, hard end states, a CRF's END tag.

Guards:

* forward, path score, Viterbi and posteriors with ``log_end`` equal an independent float64
  enumeration over every state path, on padded batches (the end potential is read at each
  sequence's LAST VALID step, not at step ``T - 1``);
* ``log_end=None`` is the old behaviour, bit for bit, and an all-zero ``log_end`` equals it;
* a BIO mask is obeyed by Viterbi under any weights, and posteriors put no mass on a
  forbidden start, while the same weights without the mask do decode invalid paths;
* a hard end state is reached by every decoded path; the generative likelihood is the joint
  with that event, so it sums to the probability of ending there;
* a constraint with the wrong shape, or leaving a state with no destination, no start or no
  end, is refused; a learned end potential is refused in generative mode;
* the end logits train, the config round-trips and an archive restores the outputs.
"""

import itertools

import keras
import numpy as np
import pytest

from dl_techniques.models.probabilistic.neural_hmm import NeuralHMM, inference

K, D, V = 3, 2, 3

#: O, B, I: I may only follow B or I, and a sequence may not start in I.
BIO_STATES = 3
BIO_TRANSITIONS = [[True, True, False], [True, True, True], [True, True, True]]
BIO_STARTS = [True, True, False]
SMALL = {"hidden_dim": 8, "num_layers": 1, "encoder_type": "gru"}


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _enumerate(log_pi, log_A, log_emit, log_end, length):
    """Every path of one sequence: ``(logZ, best path, best score, posteriors)``, float64."""
    n_states = log_emit.shape[-1]
    paths = list(itertools.product(range(n_states), repeat=length))
    scores = []
    for path in paths:
        score = log_pi[path[0]] + log_emit[0, path[0]] + log_end[path[-1]]
        for t in range(1, length):
            score += log_A[t, path[t - 1], path[t]] + log_emit[t, path[t]]
        scores.append(score)
    scores = np.array(scores)
    peak = scores.max()
    weights = np.exp(scores - peak)
    log_z = peak + np.log(weights.sum())
    posterior = np.zeros((length, n_states))
    for path, weight in zip(paths, weights):
        posterior[np.arange(length), list(path)] += weight
    return log_z, np.array(paths[int(np.argmax(scores))]), peak, posterior / weights.sum()


@pytest.fixture(name="case")
def _case():
    rng = np.random.default_rng(0)
    batch, steps = 3, 5
    log_pi = rng.normal(size=(batch, K))
    log_A = rng.normal(size=(batch, steps, K, K))
    log_emit = rng.normal(size=(batch, steps, K))
    log_end = rng.normal(size=(batch, K)) * 2.0
    lengths = np.array([5, 3, 1])
    mask = (np.arange(steps)[None] < lengths[:, None]).astype("float64")
    return log_pi, log_A, log_emit, mask, log_end, lengths


def test_every_inference_function_matches_enumeration_with_an_end_potential(case):
    log_pi, log_A, log_emit, mask, log_end, lengths = case
    log_z = _np(inference.forward_log_likelihood(log_pi, log_A, log_emit, mask, log_end))
    path, score = inference.viterbi_decode(log_pi, log_A, log_emit, mask, log_end)
    path, score = _np(path), _np(score)
    posterior = _np(inference.state_posteriors(log_pi, log_A, log_emit, mask, log_end))
    path_score = _np(inference.path_log_score(log_pi, log_A, log_emit, np.maximum(path, 0), mask, log_end))
    for b, length in enumerate(lengths):
        want_z, want_path, want_score, want_post = _enumerate(
            log_pi[b], log_A[b], log_emit[b], log_end[b], int(length))
        np.testing.assert_allclose(log_z[b], want_z, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(path[b, :length], want_path)
        np.testing.assert_allclose(score[b], want_score, rtol=0, atol=1e-9)
        np.testing.assert_allclose(path_score[b], want_score, rtol=0, atol=1e-9)
        np.testing.assert_allclose(posterior[b, :length], want_post, rtol=0, atol=1e-9)


def test_a_zero_end_potential_equals_none(case):
    log_pi, log_A, log_emit, mask, _, _ = case
    zero = np.zeros_like(log_pi)
    for run in (inference.forward_log_likelihood, inference.state_posteriors):
        np.testing.assert_array_equal(
            _np(run(log_pi, log_A, log_emit, mask, zero)), _np(run(log_pi, log_A, log_emit, mask)))
    for got, want in zip(inference.viterbi_decode(log_pi, log_A, log_emit, mask, zero),
                         inference.viterbi_decode(log_pi, log_A, log_emit, mask)):
        np.testing.assert_array_equal(_np(got), _np(want))


def test_an_end_potential_on_an_empty_sequence_scores_zero():
    log_pi, log_A = np.zeros((1, K)), np.zeros((1, 2, K, K))
    log_emit, mask, log_end = np.zeros((1, 2, K)), np.zeros((1, 2)), np.ones((1, K))
    assert _np(inference.forward_log_likelihood(log_pi, log_A, log_emit, mask, log_end)) == 0.0
    assert _np(inference.path_log_score(log_pi, log_A, log_emit, np.zeros((1, 2), "int32"), mask, log_end)) == 0.0
    assert _np(inference.viterbi_decode(log_pi, log_A, log_emit, mask, log_end)[1]) == 0.0


def _tagger(seed=2, **constraints):
    keras.utils.set_random_seed(seed)
    return NeuralHMM(
        n_states=BIO_STATES, mode="discriminative", obs_type="gaussian", obs_dim=D, self_loop_init=0.0, **SMALL,
        **constraints,
    )


def _noisy_batch(batch=32, steps=20, seed=1):
    rng = np.random.default_rng(seed)
    return {"obs": rng.normal(size=(batch, steps, D)).astype("float32"), "mask": np.ones((batch, steps), "float32")}


def _is_valid_bio(paths):
    starts_ok = (paths[:, 0] != 2).all()
    moves_ok = not ((paths[:, :-1] == 0) & (paths[:, 1:] == 2)).any()
    return starts_ok and moves_ok


def test_viterbi_obeys_a_bio_mask_where_the_unconstrained_model_does_not():
    x = _noisy_batch()
    free = _tagger()
    free(x, training=False)
    rng = np.random.default_rng(4)
    for weight in free.trainable_weights:
        weight.assign(rng.normal(0.0, 1.0, size=weight.shape).astype(weight.dtype))
    assert not _is_valid_bio(_np(free.viterbi(x)[0])), "the control must decode an invalid path"

    constrained = _tagger(allowed_transitions=BIO_TRANSITIONS, allowed_starts=BIO_STARTS)
    constrained(x, training=False)
    for weight, source in zip(constrained.trainable_weights, free.trainable_weights):
        weight.assign(source)
    assert _is_valid_bio(_np(constrained.viterbi(x)[0]))
    assert (_np(constrained.posteriors(x))[:, 0, 2] == 0.0).all()


def test_a_bio_tagger_trains_to_a_finite_loss():
    x = _noisy_batch(batch=8, steps=10)
    labels = np.zeros((8, 10), "int32")
    labels[:, 3] = 1
    labels[:, 4] = 2
    model = _tagger(allowed_transitions=BIO_TRANSITIONS, allowed_starts=BIO_STARTS, end_potential=True)
    model.compile(optimizer=keras.optimizers.Adam(1e-2))
    history = model.fit(x, labels, batch_size=8, epochs=8, verbose=0).history["loss"]
    assert np.isfinite(history).all() and history[-1] < history[0]
    assert np.abs(_np(model.transition.end_logits)).max() > 0.0, "the end logits never moved"


def test_a_gold_path_breaking_the_mask_is_loud():
    x = _noisy_batch(batch=2, steps=4)
    labels = np.zeros((2, 4), "int32")
    labels[:, 2] = 2  # O -> I
    model = _tagger(allowed_transitions=BIO_TRANSITIONS, allowed_starts=BIO_STARTS)
    assert float(_np(model.log_likelihood(x, labels)).max()) < -1e8


def _segmenter(**constraints):
    keras.utils.set_random_seed(3)
    return NeuralHMM(n_states=K, obs_type="categorical", vocab_size=V, topology="left_to_right", **constraints)


def test_a_hard_end_state_is_reached_by_every_decoded_path_also_when_padded():
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, V, (6, 8)).astype("int32")
    lengths = np.array([8, 8, 5, 5, 3, 4])
    x = {"obs": symbols, "mask": (np.arange(8)[None] < lengths[:, None]).astype("float32")}
    model = _segmenter(allowed_ends=[False, False, True])
    paths = _np(model.viterbi(x)[0])
    assert all(paths[b, length - 1] == K - 1 for b, length in enumerate(lengths))
    posterior = _np(model.posteriors(x))
    for b, length in enumerate(lengths):
        np.testing.assert_allclose(posterior[b, length - 1, K - 1], 1.0, rtol=0, atol=1e-6)


def test_a_generative_hard_end_likelihood_is_the_joint_with_ending_there():
    """sum_x p(x, z_last = end) = p(z_last = end), which is below 1 and is computed exactly."""
    model = _segmenter(allowed_ends=[False, False, True])
    length = 5
    symbols = np.array(list(itertools.product(range(V), repeat=length)), dtype="int32")
    x = {"obs": symbols, "mask": np.ones(symbols.shape, "float32")}
    total = np.exp(_np(model.log_likelihood(x)).astype("float64")).sum()
    log_A = _np(model(x, training=False)["log_A"])[0, 1].astype("float64")
    start = np.zeros(K)
    start[0] = 1.0
    reach = start @ np.linalg.matrix_power(np.exp(log_A), length - 1)
    assert total < 1.0
    np.testing.assert_allclose(total, reach[K - 1], rtol=0, atol=1e-5)


def test_a_learned_end_potential_is_refused_in_generative_mode():
    with pytest.raises(ValueError, match="end_potential=True needs mode='discriminative'"):
        NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D, end_potential=True)


@pytest.mark.parametrize("constraint,match", [
    ({"allowed_transitions": [[True, True], [True, True]]}, "allowed_transitions must have shape"),
    ({"allowed_starts": [True, True]}, "allowed_starts must have shape"),
    ({"allowed_ends": [True]}, "allowed_ends must have shape"),
    ({"allowed_transitions": [[True] * 3, [False] * 3, [True] * 3]}, "states \\[1\\] have none"),
    ({"allowed_starts": [False] * 3}, "no allowed start state"),
    ({"allowed_ends": [False] * 3}, "no end state"),
])
def test_a_bad_constraint_is_refused(constraint, match):
    with pytest.raises(ValueError, match=match):
        NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D, **constraint)


def test_constraints_narrow_the_topology_rather_than_replace_it():
    model = NeuralHMM(
        n_states=K, obs_type="gaussian", obs_dim=D, topology="left_to_right",
        allowed_transitions=[[True, False, True], [True, True, True], [True, True, True]],
    )
    keep = model.transition._transition_keep
    assert not keep[0, 1] and not keep[1, 0] and keep[0, 2] and keep[1, 1]


def test_an_unconstrained_model_has_no_end_output_and_no_end_weight():
    model = NeuralHMM(n_states=K, obs_type="gaussian", obs_dim=D)
    x = {"obs": np.zeros((1, 2, D), "float32"), "mask": np.ones((1, 2), "float32")}
    assert set(model(x, training=False)) == {"log_pi", "log_A", "log_emit"}
    assert not any("end_logits" in w.path for w in model.weights)


def test_the_output_shapes_agree_with_the_outputs():
    model = _tagger(allowed_ends=[True, True, False], end_potential=True)
    x = _noisy_batch(batch=2, steps=4)
    out = model(x, training=False)
    declared = model.compute_output_shape({"obs": (2, 4, D), "mask": (2, 4)})
    assert {k: tuple(v.shape) for k, v in out.items()} == {k: tuple(v) for k, v in declared.items()}


def test_the_archive_restores_the_constraints_and_the_outputs(tmp_path):
    model = _tagger(allowed_transitions=BIO_TRANSITIONS, allowed_starts=BIO_STARTS,
                    allowed_ends=[True, True, False], end_potential=True)
    x = _noisy_batch(batch=4, steps=6)
    model(x, training=False)
    model.transition.end_logits.assign([0.5, -0.5, 0.0])
    path = tmp_path / "tagger.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    assert restored.allowed_transitions == BIO_TRANSITIONS and restored.end_potential is True
    for key, value in model(x, training=False).items():
        np.testing.assert_array_equal(_np(restored(x, training=False)[key]), _np(value))
