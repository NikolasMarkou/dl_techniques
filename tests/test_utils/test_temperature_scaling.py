"""Tests for ``dl_techniques.utils.temperature_scaling`` (post-hoc temperature fit).

The reference is ``scipy.optimize.minimize_scalar`` run on an independent
negative log-likelihood (``scipy.special.log_softmax``), in four regimes. The
refusals and the edge warning are the behaviours the explorer measured to be
silent or cryptic before the module existed: a perfectly separable or
always-wrong validation set returned the search-range edge with no word, NaN
logits did the same, one-hot labels raised an ``IndexError`` and a length
mismatch a broadcast error. Plan plan-2026-11-11T131349-13b28b37, decision D-043.
"""

import logging

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from dl_techniques.utils.temperature_scaling import fit_temperature

LOGGER_NAME = "dl"


def _sample(n, k, true_temperature, seed, logit_scale=3.0):
    """Logits from a model whose honest temperature is ``true_temperature``.

    Labels are drawn from ``softmax(logits / true_temperature)``, so the fit
    should recover a value near it (above 1: the model is overconfident).
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * logit_scale
    p = np.exp(log_softmax(logits / true_temperature, axis=1))
    labels = np.array([rng.choice(k, p=row) for row in p])
    return logits, labels


def _scipy_temperature(logits, labels):
    """Independent optimum: bounded Brent on log(1/T) over the same range."""
    z = np.asarray(logits, dtype=np.float64)
    rows = np.arange(len(labels))

    def nll(log_beta):
        return -log_softmax(z * np.exp(log_beta), axis=1)[rows, labels].mean()

    res = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded",
                          options={"xatol": 1e-12})
    return float(np.exp(-res.x))


REGIMES = {
    "overconfident": dict(n=2000, k=10, true_temperature=3.0, seed=0),
    "underconfident": dict(n=2000, k=10, true_temperature=0.5, seed=1),
    "n20": dict(n=20, k=5, true_temperature=2.0, seed=2),
    "k100": dict(n=1500, k=100, true_temperature=2.0, seed=3),
}


@pytest.mark.parametrize("regime", list(REGIMES))
def test_matches_scipy_to_five_digits(regime, caplog):
    logits, labels = _sample(**REGIMES[regime])
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        got = fit_temperature(logits, labels)
    want = _scipy_temperature(logits, labels)
    np.testing.assert_allclose(got, want, rtol=1e-5)
    # An interior optimum is not an edge case: no warning.
    assert not [r for r in caplog.records if r.name == LOGGER_NAME]


def test_the_regimes_point_the_right_way():
    over = fit_temperature(*_sample(**REGIMES["overconfident"]))
    under = fit_temperature(*_sample(**REGIMES["underconfident"]))
    assert over > 1.5 > 0.9 > under


def test_the_fit_does_not_change_the_argmax():
    logits, labels = _sample(**REGIMES["overconfident"])
    t = fit_temperature(logits, labels)
    assert (np.argmax(logits / t, axis=1) == np.argmax(logits, axis=1)).all()


def test_float32_logits_are_read_in_float64():
    logits, labels = _sample(**REGIMES["overconfident"])
    got = fit_temperature(logits.astype(np.float32), labels)
    want = _scipy_temperature(logits.astype(np.float32), labels)
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_huge_logits_stay_finite():
    # exp(1e4) overflows a naive softmax; the max-shifted NLL does not.
    logits, labels = _sample(**REGIMES["overconfident"])
    t = fit_temperature(logits * 1e4, labels)
    assert np.isfinite(t) and t > 0


def test_a_list_input_is_accepted():
    logits, labels = _sample(**REGIMES["n20"])
    got = fit_temperature(logits.tolist(), labels.tolist())
    np.testing.assert_allclose(got, fit_temperature(logits, labels))


def test_more_iterations_do_not_change_a_converged_answer():
    logits, labels = _sample(**REGIMES["overconfident"])
    np.testing.assert_allclose(
        fit_temperature(logits, labels, iters=60),
        fit_temperature(logits, labels, iters=200), rtol=1e-6)


# --- degenerate data: the edge is returned AND announced -----------------


def _separable():
    k = 4
    labels = np.arange(40) % k
    logits = np.eye(k)[labels] * 8.0
    return logits, labels


def test_a_separable_set_warns_at_the_lower_edge(caplog):
    # DECISION plan-2026-11-11T131349-13b28b37/D-043 pin: the edge is not silent.
    logits, labels = _separable()
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        t = fit_temperature(logits, labels)
    np.testing.assert_allclose(t, np.exp(-5.0), rtol=1e-6)
    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]
    assert len(messages) == 1
    assert "edge" in messages[0] and "validation" in messages[0]


def test_an_always_wrong_set_warns_at_the_upper_edge(caplog):
    logits, labels = _separable()
    wrong = (labels + 1) % logits.shape[1]
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        t = fit_temperature(logits, wrong)
    np.testing.assert_allclose(t, np.exp(5.0), rtol=1e-6)
    assert len([r for r in caplog.records if r.name == LOGGER_NAME]) == 1


# --- refusals: each used to be silent or cryptic -------------------------


def test_one_hot_labels_raise_a_clear_error():
    # DECISION plan-2026-11-11T131349-13b28b37/D-043 pin: one-hot labels are
    # refused, not flattened into class indices (the audit's reshape(-1) did).
    logits, labels = _sample(**REGIMES["n20"])
    one_hot = np.eye(logits.shape[1])[labels]
    with pytest.raises(ValueError, match="one-hot"):
        fit_temperature(logits, one_hot)


def test_a_length_mismatch_raises():
    logits, labels = _sample(**REGIMES["n20"])
    with pytest.raises(ValueError, match="labels must be"):
        fit_temperature(logits, labels[:-1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_logits_raise(bad):
    logits, labels = _sample(**REGIMES["n20"])
    logits[3, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        fit_temperature(logits, labels)


@pytest.mark.parametrize("shift", [-1, 5])
def test_out_of_range_labels_raise(shift):
    logits, labels = _sample(**REGIMES["n20"])
    labels = labels.copy()
    labels[0] = shift if shift < 0 else logits.shape[1] + shift - 5
    with pytest.raises(ValueError, match="labels must be in"):
        fit_temperature(logits, labels)


def test_float_labels_raise():
    logits, labels = _sample(**REGIMES["n20"])
    with pytest.raises(ValueError, match="integer"):
        fit_temperature(logits, labels.astype(np.float64))


def test_bool_labels_raise():
    logits = np.zeros((4, 2))
    with pytest.raises(ValueError, match="integer"):
        fit_temperature(logits, np.array([True, False, True, False]))


def test_logits_must_be_two_dimensional():
    with pytest.raises(ValueError, match=r"logits must be \(N, C\)"):
        fit_temperature(np.zeros(5), np.zeros(5, dtype=int))


def test_an_empty_validation_set_raises():
    with pytest.raises(ValueError, match="at least one"):
        fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=int))


def test_a_single_class_raises():
    with pytest.raises(ValueError, match="at least 2 classes"):
        fit_temperature(np.zeros((4, 1)), np.zeros(4, dtype=int))


@pytest.mark.parametrize("iters", [0, -3, 2.5, True, None])
def test_iters_must_be_a_positive_int(iters):
    logits, labels = _sample(**REGIMES["n20"])
    with pytest.raises(ValueError, match="iters"):
        fit_temperature(logits, labels, iters=iters)
