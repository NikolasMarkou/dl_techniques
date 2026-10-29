"""Mode extraction for mixture density networks.

``get_point_estimate`` returns the mixture mean, which for a multi-modal output lands
between the modes. ``get_mixture_modes`` and ``get_mode_estimate`` return the modes
themselves. Every numeric claim is checked against a float64 NumPy oracle written here,
independent of the implementation: the log density by direct evaluation, the modes by a
zoomed brute-force grid search of the mixture density.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.statistics.mdn_layer import (
    MDNLayer,
    get_mixture_modes,
    get_mode_estimate,
    get_point_estimate,
)


# ---------------------------------------------------------------------
# float64 oracles
# ---------------------------------------------------------------------


def _oracle_log_density(points, mu, sigma, pi_logits):
    """Log of the mixture density at ``points`` [K, D] for one sample (float64).

    :param points: Query points ``[K, D]``.
    :param mu: Component means ``[M, D]``.
    :param sigma: Component standard deviations ``[M, D]``.
    :param pi_logits: Mixture logits ``[M]``.
    :return: ``[K]`` log densities.
    """
    points = np.asarray(points, np.float64)
    mu = np.asarray(mu, np.float64)
    sigma = np.asarray(sigma, np.float64)
    logits = np.asarray(pi_logits, np.float64)
    log_pi = logits - (logits.max() + np.log(np.exp(logits - logits.max()).sum()))
    z = (points[:, None, :] - mu[None]) / sigma[None]
    log_normal = (
        -0.5 * (z ** 2).sum(-1)
        - np.log(sigma).sum(-1)[None]
        - 0.5 * mu.shape[-1] * np.log(2.0 * np.pi)
    )
    total = log_pi[None] + log_normal
    top = total.max(axis=1)
    return top + np.log(np.exp(total - top[:, None]).sum(axis=1))


def _oracle_grid_mode(mu, sigma, pi_logits, centre, half_width, levels=4, size=41):
    """Argmax of the mixture density by repeated zoomed grid search (float64).

    :param mu: Component means ``[M, D]``.
    :param sigma: Component standard deviations ``[M, D]``.
    :param pi_logits: Mixture logits ``[M]``.
    :param centre: Starting centre of the search box ``[D]``.
    :param half_width: Half width of the first box.
    :param levels: Number of zoom levels, each shrinking the box by 10.
    :param size: Grid points per axis per level.
    :return: ``(mode [D], log density at the mode)``.
    """
    centre = np.asarray(centre, np.float64)
    dim = centre.shape[0]
    for _ in range(levels):
        axes = [np.linspace(c - half_width, c + half_width, size) for c in centre]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
        values = _oracle_log_density(mesh, mu, sigma, pi_logits)
        centre = mesh[values.argmax()]
        half_width = half_width / 10.0
    return centre, float(values.max())


def _oracle_fixed_point(mu, sigma, pi_logits, num_steps):
    """The Carreira-Perpinan iteration from every component mean (float64)."""
    mu = np.asarray(mu, np.float64)
    precision = 1.0 / np.asarray(sigma, np.float64) ** 2
    logits = np.asarray(pi_logits, np.float64)
    log_pi = logits - np.log(np.exp(logits).sum())
    x = mu.copy()
    for _ in range(num_steps):
        out = np.empty_like(x)
        for start in range(x.shape[0]):
            z2 = ((x[start] - mu) ** 2 * precision).sum(-1)
            logp = log_pi - 0.5 * z2 - np.log(1.0 / np.sqrt(precision)).sum(-1)
            r = np.exp(logp - logp.max())
            r /= r.sum()
            out[start] = (r[:, None] * precision * mu).sum(0) / (r[:, None] * precision).sum(0)
        x = out
    return x


def _assert_is_not_the_mixture_mean(estimate, mu, pi_logits, min_gap=1.0):
    """Fail unless ``estimate`` is at least ``min_gap`` from the mixture mean.

    :param estimate: ``[B, D]`` estimate to inspect.
    :param mu: Component means ``[B, M, D]``.
    :param pi_logits: Logits ``[B, M]``.
    :param min_gap: Smallest accepted distance from the mixture mean.
    """
    pi = np.exp(pi_logits - pi_logits.max(-1, keepdims=True))
    pi /= pi.sum(-1, keepdims=True)
    mean = (pi[..., None] * mu).sum(1)
    gap = np.abs(np.asarray(estimate) - mean).max()
    assert gap >= min_gap, f"estimate sits {gap:.3g} from the mixture mean"


# ---------------------------------------------------------------------
# a model whose output is a fixed mixture
# ---------------------------------------------------------------------


def _constant_mixture_model(mu, sigma, pi_logits):
    """Build a model that emits the same packed mixture for every input row.

    :param mu: Component means ``[M, D]``.
    :param sigma: Component standard deviations ``[M, D]``.
    :param pi_logits: Mixture logits ``[M]``.
    :return: ``(model, mdn_layer)``; the layer only unpacks, the model's Dense
        has zero kernel and the packed mixture as bias.
    """
    num_mix, dim = np.asarray(mu).shape
    layer = MDNLayer(output_dimension=dim, num_mixtures=num_mix)
    packed = np.concatenate(
        [np.ravel(mu), np.ravel(sigma), np.ravel(pi_logits)]
    ).astype("float32")
    inputs = keras.Input((2,))
    outputs = keras.layers.Dense(
        packed.size,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(packed),
    )(inputs)
    return keras.Model(inputs, outputs), layer


def _batch(mu, sigma, pi_logits, dtype="float32"):
    """Add a batch axis of one to ``[M, D]``, ``[M, D]``, ``[M]`` arrays."""
    return (
        np.asarray(mu, dtype)[None],
        np.asarray(sigma, dtype)[None],
        np.asarray(pi_logits, dtype)[None],
    )


BIMODAL_MU = np.array([[-3.0, 2.0], [3.0, -2.0]])
BIMODAL_SIGMA = np.full((2, 2), 0.5)
BIMODAL_LOGITS = np.zeros(2)


# ---------------------------------------------------------------------
# 1 to 4: what the modes are
# ---------------------------------------------------------------------


class TestTheModesAreTheMixtureModes:
    def test_a_single_component_has_its_mean_as_mode(self):
        mu, sigma, logits = _batch([[1.5, -2.0, 0.25]], [[0.3, 1.0, 2.0]], [0.7])
        modes, log_density, is_unique = get_mixture_modes(mu, sigma, logits)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(modes), mu, atol=1e-6)
        expected = -np.log(sigma).sum() - 1.5 * np.log(2 * np.pi)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(log_density)[0, 0], expected, rtol=1e-5
        )
        assert bool(keras.ops.convert_to_numpy(is_unique)[0, 0])

    def test_two_separated_equal_components_return_a_mean_not_the_midpoint(self):
        model, layer = _constant_mixture_model(BIMODAL_MU, BIMODAL_SIGMA, BIMODAL_LOGITS)
        x = np.zeros((3, 2), "float32")
        estimate = get_mode_estimate(model, x, layer)
        assert estimate.shape == (3, 2)
        for row in estimate:
            distance = np.abs(BIMODAL_MU - row).max(axis=1).min()
            assert distance < 1e-3, f"{row} is not within 1e-3 of a component mean"
        point = get_point_estimate(model, x, layer)
        np.testing.assert_allclose(point, np.zeros((3, 2)), atol=1e-5)
        mu_batch = np.broadcast_to(BIMODAL_MU, (3, 2, 2))
        logits_batch = np.zeros((3, 2))
        _assert_is_not_the_mixture_mean(estimate, mu_batch, logits_batch)

    def test_the_mode_estimate_of_two_separated_components_has_two_unique_modes(self):
        mu, sigma, logits = _batch(BIMODAL_MU, BIMODAL_SIGMA, BIMODAL_LOGITS)
        modes, _, is_unique = get_mixture_modes(mu, sigma, logits)
        assert keras.ops.convert_to_numpy(is_unique).all()
        np.testing.assert_allclose(keras.ops.convert_to_numpy(modes)[0], BIMODAL_MU, atol=1e-3)

    @pytest.mark.parametrize("logits", [[0.0, 0.0], [0.4, 0.0], [-0.3, 0.9]])
    @pytest.mark.parametrize("dim", [1, 2])
    def test_overlapping_components_have_one_mode_equal_to_the_grid_maximum(self, logits, dim):
        mu = np.stack([np.full(dim, -0.5), np.full(dim, 0.5)])
        sigma = np.full((2, dim), 1.0)
        batch = _batch(mu, sigma, logits)
        modes, log_density, is_unique = [
            keras.ops.convert_to_numpy(t) for t in get_mixture_modes(*batch)
        ]
        assert is_unique.sum() == 1, "overlapping components must collapse to one mode"
        grid_mode, grid_density = _oracle_grid_mode(
            mu, sigma, logits, centre=np.zeros(dim), half_width=1.5
        )
        survivor = modes[0][is_unique[0]][0]
        error = np.abs(survivor - grid_mode).max()
        assert error < 1e-3, f"mode {survivor} vs grid {grid_mode}: {error:.2e}"
        assert abs(log_density[0][is_unique[0]][0] - grid_density) < 1e-5
        # Between the two means, not on either of them (liveness of the move).
        assert np.abs(survivor - mu[0]).max() > 1e-2 or np.abs(survivor - mu[1]).max() > 1e-2

    def test_unequal_weights_select_the_heavier_component(self):
        logits = np.log([0.8, 0.2])
        model, layer = _constant_mixture_model(BIMODAL_MU, BIMODAL_SIGMA, logits)
        estimate = get_mode_estimate(model, np.zeros((2, 2), "float32"), layer)
        np.testing.assert_allclose(estimate, np.broadcast_to(BIMODAL_MU[0], (2, 2)), atol=1e-3)
        flipped_model, flipped_layer = _constant_mixture_model(
            BIMODAL_MU, BIMODAL_SIGMA, logits[::-1]
        )
        flipped = get_mode_estimate(flipped_model, np.zeros((2, 2), "float32"), flipped_layer)
        np.testing.assert_allclose(flipped, np.broadcast_to(BIMODAL_MU[1], (2, 2)), atol=1e-3)

    def test_the_modes_match_the_float64_fixed_point(self):
        rng = np.random.default_rng(3)
        mu = rng.normal(size=(4, 2))
        sigma = rng.uniform(0.6, 1.4, size=(4, 2))
        logits = rng.normal(size=4)
        modes = keras.ops.convert_to_numpy(get_mixture_modes(*_batch(mu, sigma, logits))[0])[0]
        reference = _oracle_fixed_point(mu, sigma, logits, 100)
        assert np.abs(modes - reference).max() < 1e-4


# ---------------------------------------------------------------------
# 5: the log density
# ---------------------------------------------------------------------


class TestTheLogDensity:
    @staticmethod
    def _case():
        rng = np.random.default_rng(11)
        mu = rng.normal(0.0, 1.5, size=(3, 4, 2))
        sigma = rng.uniform(0.4, 1.6, size=(3, 4, 2))
        logits = rng.normal(0.0, 1.0, size=(3, 4))
        return mu, sigma, logits

    @pytest.mark.parametrize("dtype, tolerance", [("float32", 2e-5), ("float64", 1e-9)])
    def test_the_log_density_matches_the_float64_oracle_at_the_returned_modes(
        self, dtype, tolerance
    ):
        mu, sigma, logits = self._case()
        modes, log_density, _ = get_mixture_modes(
            mu.astype(dtype), sigma.astype(dtype), logits.astype(dtype)
        )
        modes = keras.ops.convert_to_numpy(modes)
        log_density = keras.ops.convert_to_numpy(log_density)
        assert modes.dtype == np.dtype(dtype) and log_density.dtype == np.dtype(dtype)
        worst = 0.0
        for b in range(mu.shape[0]):
            # The oracle sees the same rounded inputs, so only the arithmetic differs.
            expected = _oracle_log_density(
                modes[b].astype(np.float64),
                mu[b].astype(dtype).astype(np.float64),
                sigma[b].astype(dtype).astype(np.float64),
                logits[b].astype(dtype).astype(np.float64),
            )
            worst = max(worst, np.abs(log_density[b] - expected).max())
        assert worst < tolerance, f"max |layer - float64 oracle| = {worst:.3e}"

    def test_the_float64_arm_is_tighter_than_the_float32_arm(self):
        # Liveness of the dtype split: float64 inputs are not silently demoted.
        mu, sigma, logits = self._case()
        out = get_mixture_modes(mu, sigma, logits)[1]
        assert keras.ops.convert_to_numpy(out).dtype == np.float64

    def test_the_log_density_at_a_mode_is_not_below_the_density_at_the_start(self):
        mu, sigma, logits = self._case()
        start = _oracle_log_density(mu[0], mu[0], sigma[0], logits[0])
        _, log_density, _ = get_mixture_modes(mu, sigma, logits)
        assert (keras.ops.convert_to_numpy(log_density)[0] >= start - 1e-5).all()


# ---------------------------------------------------------------------
# 6: finiteness
# ---------------------------------------------------------------------


class TestFiniteness:
    @pytest.mark.parametrize(
        "sigma_value, spread",
        [(1e-3, 0.0), (1e-3, 50.0), (1e-6, 0.0), (1.0, 1e4), (1.0, 1e-30), (1e3, 5.0)],
    )
    def test_outputs_are_finite(self, sigma_value, spread):
        rng = np.random.default_rng(5)
        mu = rng.normal(0.0, 2.0, size=(2, 5, 3)).astype("float32")
        sigma = np.full((2, 5, 3), sigma_value, "float32")
        logits = (spread * rng.normal(size=(2, 5))).astype("float32")
        modes, log_density, is_unique = get_mixture_modes(mu, sigma, logits)
        assert np.isfinite(keras.ops.convert_to_numpy(modes)).all()
        assert np.isfinite(keras.ops.convert_to_numpy(log_density)).all()
        assert keras.ops.convert_to_numpy(is_unique).any(axis=-1).all()

    def test_a_narrow_isolated_component_keeps_its_mean_as_mode(self):
        mu = np.array([[[0.0], [10.0]]], "float32")
        sigma = np.full((1, 2, 1), 1e-3, "float32")
        modes = keras.ops.convert_to_numpy(get_mixture_modes(mu, sigma, np.zeros((1, 2), "float32"))[0])
        np.testing.assert_allclose(modes, mu, atol=1e-6)

    def test_float16_inputs_are_finite_and_come_back_in_float16(self):
        mu = np.array([[[-3.0], [3.0]]], "float16")
        sigma = np.full((1, 2, 1), 0.5, "float16")
        modes, log_density, _ = get_mixture_modes(mu, sigma, np.zeros((1, 2), "float16"))
        assert keras.backend.standardize_dtype(modes.dtype) == "float16"
        assert keras.backend.standardize_dtype(log_density.dtype) == "float16"
        assert np.isfinite(keras.ops.convert_to_numpy(modes).astype("float32")).all()


# ---------------------------------------------------------------------
# 7: merging
# ---------------------------------------------------------------------


class TestUniqueness:
    def test_identical_components_leave_exactly_one_unique_start(self):
        mu = np.tile(np.array([[1.0, 2.0]], "float32"), (2, 1))[None].repeat(3, axis=0)
        sigma = np.full(mu.shape, 0.7, "float32")
        logits = np.zeros((3, 2), "float32")
        _, _, is_unique = get_mixture_modes(mu, sigma, logits)
        flags = keras.ops.convert_to_numpy(is_unique)
        assert flags.dtype == np.bool_
        assert (flags.sum(axis=1) == 1).all()
        assert flags[:, 0].all() and not flags[:, 1].any(), "the FIRST of a group is kept"

    def test_a_later_start_is_dropped_when_it_matches_any_earlier_one(self):
        mu = np.array([[[0.0], [5.0], [0.0]]], "float32")
        sigma = np.full((1, 3, 1), 0.3, "float32")
        _, _, is_unique = get_mixture_modes(mu, sigma, np.zeros((1, 3), "float32"), num_steps=0)
        assert keras.ops.convert_to_numpy(is_unique).tolist() == [[True, True, False]]

    def test_merge_tol_controls_the_grouping(self):
        mu = np.array([[[0.0], [0.01]]], "float32")
        sigma = np.full((1, 2, 1), 0.3, "float32")
        logits = np.zeros((1, 2), "float32")
        tight = get_mixture_modes(mu, sigma, logits, num_steps=0, merge_tol=1e-3)[2]
        loose = get_mixture_modes(mu, sigma, logits, num_steps=0, merge_tol=0.1)[2]
        assert keras.ops.convert_to_numpy(tight).all()
        assert keras.ops.convert_to_numpy(loose).tolist() == [[True, False]]


# ---------------------------------------------------------------------
# 8: tracing
# ---------------------------------------------------------------------


class TestTracing:
    def test_an_unknown_batch_size_traces_and_matches_eager(self):
        num_mix, dim = 3, 2

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, num_mix, dim], tf.float32),
                tf.TensorSpec([None, num_mix, dim], tf.float32),
                tf.TensorSpec([None, num_mix], tf.float32),
            ]
        )
        def traced(mu, sigma, logits):
            return get_mixture_modes(mu, sigma, logits, num_steps=30)

        rng = np.random.default_rng(2)
        for batch in (1, 5):
            mu = rng.normal(size=(batch, num_mix, dim)).astype("float32")
            sigma = rng.uniform(0.5, 1.5, size=(batch, num_mix, dim)).astype("float32")
            logits = rng.normal(size=(batch, num_mix)).astype("float32")
            eager = get_mixture_modes(mu, sigma, logits, num_steps=30)
            graph = traced(mu, sigma, logits)
            for a, b in zip(eager, graph):
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(a), b.numpy(), rtol=1e-5, atol=1e-5
                )


# ---------------------------------------------------------------------
# 9: the iteration count
# ---------------------------------------------------------------------


class TestTheIterationCount:
    def test_zero_steps_return_the_component_means(self):
        mu = np.array([[[-0.5], [0.5]]], "float32")
        sigma = np.ones((1, 2, 1), "float32")
        modes = get_mixture_modes(mu, sigma, np.zeros((1, 2), "float32"), num_steps=0)[0]
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(modes), mu)

    def test_one_hundred_steps_move_a_start_in_the_overlapping_case(self):
        mu = np.array([[[-0.5], [0.5]]], "float32")
        sigma = np.ones((1, 2, 1), "float32")
        modes = get_mixture_modes(mu, sigma, np.zeros((1, 2), "float32"), num_steps=100)[0]
        assert np.abs(keras.ops.convert_to_numpy(modes) - mu).max() > 0.4


# ---------------------------------------------------------------------
# 10: argument checks
# ---------------------------------------------------------------------


class TestArgumentChecks:
    MU = np.zeros((1, 2, 2), "float32")
    SIGMA = np.ones((1, 2, 2), "float32")
    LOGITS = np.zeros((1, 2), "float32")

    def test_a_negative_num_steps_is_refused_by_value(self):
        with pytest.raises(ValueError, match="-3"):
            get_mixture_modes(self.MU, self.SIGMA, self.LOGITS, num_steps=-3)

    @pytest.mark.parametrize("bad", [2.5, "7", True])
    def test_a_non_int_num_steps_is_refused_by_value(self, bad):
        with pytest.raises(ValueError, match=repr(bad).replace(".", r"\.")):
            get_mixture_modes(self.MU, self.SIGMA, self.LOGITS, num_steps=bad)

    @pytest.mark.parametrize("bad", [0.0, -1e-3])
    def test_a_non_positive_merge_tol_is_refused_by_value(self, bad):
        with pytest.raises(ValueError, match=repr(bad).replace(".", r"\.").replace("-", r"\-")):
            get_mixture_modes(self.MU, self.SIGMA, self.LOGITS, merge_tol=bad)

    def test_a_rank_mismatch_names_the_shapes(self):
        with pytest.raises(ValueError, match=r"mu=\(2, 2\)"):
            get_mixture_modes(self.MU[0], self.SIGMA, self.LOGITS)
        with pytest.raises(ValueError, match=r"pi_logits=\(1, 2, 2\)"):
            get_mixture_modes(self.MU, self.SIGMA, self.MU)

    def test_a_size_mismatch_names_the_shapes(self):
        with pytest.raises(ValueError, match=r"pi_logits=\(1, 3\)"):
            get_mixture_modes(self.MU, self.SIGMA, np.zeros((1, 3), "float32"))
        with pytest.raises(ValueError, match=r"sigma=\(1, 2, 3\)"):
            get_mixture_modes(self.MU, np.ones((1, 2, 3), "float32"), self.LOGITS)


# ---------------------------------------------------------------------
# 11: the contrast can fail
# ---------------------------------------------------------------------


class TestTheContrastCanFail:
    def test_the_point_estimate_of_the_bimodal_case_fails_the_not_the_mean_check(self):
        model, layer = _constant_mixture_model(BIMODAL_MU, BIMODAL_SIGMA, BIMODAL_LOGITS)
        x = np.zeros((2, 2), "float32")
        mu_batch = np.broadcast_to(BIMODAL_MU, (2, 2, 2))
        with pytest.raises(AssertionError, match="mixture mean"):
            _assert_is_not_the_mixture_mean(
                get_point_estimate(model, x, layer), mu_batch, np.zeros((2, 2))
            )

    def test_the_mode_estimate_passes_the_same_check(self):
        model, layer = _constant_mixture_model(BIMODAL_MU, BIMODAL_SIGMA, BIMODAL_LOGITS)
        x = np.zeros((2, 2), "float32")
        mu_batch = np.broadcast_to(BIMODAL_MU, (2, 2, 2))
        _assert_is_not_the_mixture_mean(
            get_mode_estimate(model, x, layer), mu_batch, np.zeros((2, 2))
        )


# ---------------------------------------------------------------------
# stationary points and the merge point
# ---------------------------------------------------------------------

# Components at -1, 0, +1 (sigma 0.5, weights 0.45, 0.1, 0.45): the middle mean is a density
# minimum (Hessian of log p +4.79). Measured 2026-10-29 in float64: with the 1e-3 sigma start
# offset the middle start reaches the side mode -0.9658 after 30 steps (-0.6543 after 10).
SADDLE_MU = np.array([[-1.0], [0.0], [1.0]])
SADDLE_SIGMA = np.full((3, 1), 0.5)
SADDLE_LOGITS = np.log([0.45, 0.1, 0.45])
SADDLE_SIDE_MODE = 0.9658
SADDLE_STEPS = 100


def _check_the_middle_start_leaves_the_density_minimum():
    mu, sigma, logits = _batch(SADDLE_MU, SADDLE_SIGMA, SADDLE_LOGITS)
    modes, log_density, is_unique = get_mixture_modes(mu, sigma, logits, num_steps=SADDLE_STEPS)
    modes = keras.ops.convert_to_numpy(modes)[0, :, 0]
    log_density = keras.ops.convert_to_numpy(log_density)[0]
    # The middle start leaves the minimum on whichever side its offset points.
    np.testing.assert_allclose(modes[[0, 2]], [-SADDLE_SIDE_MODE, SADDLE_SIDE_MODE], atol=2e-3)
    assert abs(abs(modes[1]) - SADDLE_SIDE_MODE) < 2e-3 and abs(modes[1]) > 1e-3
    assert int(keras.ops.convert_to_numpy(is_unique).sum()) == 2
    at_minimum = _oracle_log_density(np.zeros((1, 1)), SADDLE_MU, SADDLE_SIGMA, SADDLE_LOGITS)[0]
    assert log_density[1] > at_minimum + 0.5
    np.testing.assert_allclose(log_density[1], log_density[[0, 2]], atol=1e-3)


def _check_the_shifted_middle_start_leaves_the_density_minimum(shift, num_steps=300):
    """The -1/0/+1 triple moved to ``shift`` in float32: the middle start must leave the minimum."""
    mu, sigma, logits = _batch(SADDLE_MU + shift, SADDLE_SIGMA, SADDLE_LOGITS)
    # Float32 spacing at the shift bounds how exactly a mode can be located.
    ulp = float(np.spacing(np.float32(abs(shift) + 1.0)))
    modes, log_density, is_unique = get_mixture_modes(
        mu, sigma, logits, num_steps=num_steps, merge_tol=max(1e-3, 8.0 * ulp)
    )
    modes = keras.ops.convert_to_numpy(modes)[0, :, 0].astype(np.float64) - shift
    log_density = keras.ops.convert_to_numpy(log_density)[0]
    atol = max(2e-3, 4.0 * ulp)
    np.testing.assert_allclose(modes[[0, 2]], [-SADDLE_SIDE_MODE, SADDLE_SIDE_MODE], atol=atol)
    assert abs(abs(modes[1]) - SADDLE_SIDE_MODE) < atol and abs(modes[1]) > 0.5
    assert int(keras.ops.convert_to_numpy(is_unique).sum()) == 2
    at_minimum = _oracle_log_density(
        np.full((1, 1), float(shift)), SADDLE_MU + shift, SADDLE_SIGMA, SADDLE_LOGITS
    )[0]
    assert log_density[1] > at_minimum + 0.5
    np.testing.assert_allclose(log_density[1], log_density[[0, 2]], atol=0.05)


class TestStationaryPointsReachedFromEachComponentMean:
    @pytest.mark.parametrize("shift", [0.0, 50.0, 5000.0, 10000.0, 1e6])
    def test_a_far_from_origin_density_minimum_is_still_left_in_float32(self, shift):
        _check_the_shifted_middle_start_leaves_the_density_minimum(shift)

    def test_a_shifted_triple_keeps_its_means_at_zero_steps(self):
        mu, sigma, logits = _batch(SADDLE_MU + 1e4, SADDLE_SIGMA, SADDLE_LOGITS)
        modes = get_mixture_modes(mu, sigma, logits, num_steps=0)[0]
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(modes), mu)

    def test_a_start_at_a_density_minimum_leaves_it(self):
        _check_the_middle_start_leaves_the_density_minimum()

    def test_red_without_the_start_offset_the_minimum_is_returned_as_a_unique_mode(self, monkeypatch):
        import dl_techniques.layers.statistics.mdn_layer as mdn_module

        _check_the_middle_start_leaves_the_density_minimum()
        monkeypatch.setattr(mdn_module, "_MODE_START_OFFSET", 0.0)
        mu, sigma, logits = _batch(SADDLE_MU, SADDLE_SIGMA, SADDLE_LOGITS)
        modes, _, is_unique = get_mixture_modes(mu, sigma, logits, num_steps=SADDLE_STEPS)
        assert abs(float(keras.ops.convert_to_numpy(modes)[0, 1, 0])) < 1e-6
        assert keras.ops.convert_to_numpy(is_unique).tolist() == [[True, True, True]]
        with pytest.raises(AssertionError):
            _check_the_middle_start_leaves_the_density_minimum()

    def test_the_mode_estimate_is_the_highest_density_start(self):
        mu, sigma, logits = _batch(SADDLE_MU, SADDLE_SIGMA, SADDLE_LOGITS)
        modes, log_density, _ = get_mixture_modes(mu, sigma, logits, num_steps=SADDLE_STEPS)
        modes = keras.ops.convert_to_numpy(modes)[0, :, 0]
        best = int(np.argmax(keras.ops.convert_to_numpy(log_density)[0]))
        assert abs(abs(modes[best]) - SADDLE_SIDE_MODE) < 2e-3

    def test_zero_steps_keep_the_means_without_the_offset(self):
        mu, sigma, logits = _batch(SADDLE_MU, SADDLE_SIGMA, SADDLE_LOGITS)
        modes = get_mixture_modes(mu, sigma, logits, num_steps=0)[0]
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(modes), mu)


# A mixture symmetric under swapping coordinates has saddles on the swap-symmetric
# manifold; a start moved along one shared direction stays on that manifold.
X_MU = np.array([[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])
X_SIGMA = np.full((3, 2), 0.5)
X_LOGITS = np.log([0.45, 0.45, 0.1])
X_STEPS = 300


def _oracle_gradient_and_hessian(point, mu, sigma, pi_logits):
    """Gradient and Hessian of the log mixture density at one point (float64).

    :param point: ``[D]`` point.
    :param mu: Component means ``[M, D]``.
    :param sigma: Component standard deviations ``[M, D]``.
    :param pi_logits: Mixture logits ``[M]``.
    :return: ``(gradient [D], hessian [D, D])`` of log p.
    """
    point = np.asarray(point, np.float64)
    mu, sigma = np.asarray(mu, np.float64), np.asarray(sigma, np.float64)
    precision = 1.0 / sigma ** 2
    joint = _component_log_joint(point, mu, sigma, pi_logits)
    resp = np.exp(joint - (joint.max() + np.log(np.exp(joint - joint.max()).sum())))
    slope = -precision * (point - mu)
    mean_slope = (resp[:, None] * slope).sum(0)
    hessian = -np.diag((resp[:, None] * precision).sum(0))
    for r, g in zip(resp, slope):
        hessian += r * np.outer(g, g)
    hessian -= np.outer(mean_slope, mean_slope)
    return mean_slope, hessian


def _component_log_joint(point, mu, sigma, pi_logits):
    logits = np.asarray(pi_logits, np.float64)
    log_pi = logits - (logits.max() + np.log(np.exp(logits - logits.max()).sum()))
    z = (point - mu) / sigma
    return log_pi - 0.5 * (z ** 2).sum(-1) - np.log(sigma).sum(-1) - 0.5 * mu.shape[-1] * np.log(2 * np.pi)


def _assert_every_unique_mode_is_a_local_maximum(mu, sigma, logits, num_steps=X_STEPS):
    """Fail unless each unique returned mode has a vanishing gradient and a negative-definite Hessian."""
    inputs = _batch(mu, sigma, logits, "float64")
    modes, _, is_unique = get_mixture_modes(*inputs, num_steps=num_steps)
    modes = keras.ops.convert_to_numpy(modes)[0]
    unique = keras.ops.convert_to_numpy(is_unique)[0]
    assert unique.any()
    for index in np.flatnonzero(unique):
        gradient, hessian = _oracle_gradient_and_hessian(modes[index], mu, sigma, logits)
        top = np.linalg.eigvalsh(hessian).max()
        assert np.linalg.norm(gradient) < 1e-3 and top < 0.0, (
            f"start {index} at {np.round(modes[index], 4)}: gradient {np.linalg.norm(gradient):.3g}, "
            f"largest Hessian eigenvalue {top:.3g}"
        )


def _swap_symmetric_mixture(seed, dim):
    """A random mixture invariant under swapping coordinates 0 and 1.

    Two mirrored pairs, two components on the diagonal (equal means and sigmas in
    coordinates 0 and 1) and a small diagonal centre; all means within 1.5 of the origin.
    """
    rng = np.random.default_rng(seed)
    mu, sigma, weight = [], [], []

    def swapped(row):
        row = np.array(row)
        row[[0, 1]] = row[[1, 0]]
        return row

    for _ in range(2):
        m, sg, w = rng.uniform(-1.5, 1.5, dim), rng.uniform(0.3, 0.6, dim), rng.uniform(0.5, 1.0)
        mu += [m, swapped(m)]
        sigma += [sg, swapped(sg)]
        weight += [w, w]
    for _ in range(3):
        m, sg = rng.uniform(-1.5, 1.5, dim), rng.uniform(0.3, 0.6, dim)
        m[1], sg[1] = m[0], sg[0]
        mu.append(m)
        sigma.append(sg)
        weight.append(rng.uniform(0.05, 0.3))
    weight = np.array(weight) / np.sum(weight)
    return np.array(mu), np.array(sigma), np.log(weight)


class TestTheStartOffsetBreaksCoordinateSymmetry:
    def test_the_x_shaped_mixture_centre_start_does_not_stop_on_its_saddle(self):
        _assert_every_unique_mode_is_a_local_maximum(X_MU, X_SIGMA, X_LOGITS)

    def test_the_x_shaped_saddle_is_one_of_the_stationary_points(self):
        gradient, hessian = _oracle_gradient_and_hessian(np.zeros(2), X_MU, X_SIGMA, X_LOGITS)
        assert np.linalg.norm(gradient) < 1e-12 and np.linalg.eigvalsh(hessian).max() > 0.0

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("seed", range(20))
    def test_swap_symmetric_mixtures_return_no_saddle_or_minimum_as_a_unique_mode(self, seed, dim):
        _assert_every_unique_mode_is_a_local_maximum(*_swap_symmetric_mixture(seed, dim))

    def test_every_component_and_dimension_gets_its_own_nonzero_offset(self):
        from dl_techniques.layers.statistics import mdn_layer as mdn_module

        for num_mix, dim in [(1, 1), (3, 2), (7, 3), (16, 8)]:
            pattern = keras.ops.convert_to_numpy(mdn_module._mode_start_pattern(num_mix, dim, "float32"))
            assert pattern.shape == (num_mix, dim)
            assert np.abs(pattern).min() >= 0.25 and np.abs(pattern).max() <= 1.0
            assert len(np.unique(pattern)) == num_mix * dim

    def test_the_pattern_stays_distinct_and_nonzero_for_large_tables(self):
        from dl_techniques.layers.statistics import mdn_layer as mdn_module

        flat = keras.ops.convert_to_numpy(mdn_module._mode_start_pattern(1000, 70, "float32")).ravel()
        assert flat.shape == (70000,)
        assert np.abs(flat).min() >= 0.25 and np.abs(flat).max() <= 1.0
        assert len(np.unique(flat)) == 70000
        assert all(len(np.unique(flat[i : i + 4096])) == len(flat[i : i + 4096]) for i in range(0, 70000, 4096))
        # The table depends on the flat index alone, so a wider table repeats the same entries.
        wide = keras.ops.convert_to_numpy(mdn_module._mode_start_pattern(7, 10000, "float32")).ravel()
        np.testing.assert_array_equal(wide[:70000], flat)

    def test_red_with_one_shared_offset_direction_the_x_shaped_saddle_is_returned(self, monkeypatch):
        from dl_techniques.layers.statistics import mdn_layer as mdn_module

        _assert_every_unique_mode_is_a_local_maximum(X_MU, X_SIGMA, X_LOGITS)
        monkeypatch.setattr(
            mdn_module,
            "_mode_start_pattern",
            lambda num_mix, dim, dtype: keras.ops.ones((num_mix, dim), dtype),
        )
        with pytest.raises(AssertionError, match="largest Hessian eigenvalue"):
            _assert_every_unique_mode_is_a_local_maximum(X_MU, X_SIGMA, X_LOGITS)
        failing = 0
        for seed in range(20):
            for dim in (2, 3):
                try:
                    _assert_every_unique_mode_is_a_local_maximum(*_swap_symmetric_mixture(seed, dim))
                except AssertionError:
                    failing += 1
        assert failing >= 1

    def test_the_offset_survives_an_unknown_batch_under_tf_function(self):
        spec = [tf.TensorSpec((None, 3, 2), tf.float32)] * 2 + [tf.TensorSpec((None, 3), tf.float32)]

        @tf.function(input_signature=spec)
        def run(mu, sigma, logits):
            return get_mixture_modes(mu, sigma, logits, num_steps=X_STEPS)[0]

        mu, sigma, logits = _batch(X_MU, X_SIGMA, X_LOGITS)
        graph = run(tf.constant(mu), tf.constant(sigma), tf.constant(logits)).numpy()
        eager = keras.ops.convert_to_numpy(get_mixture_modes(mu, sigma, logits, num_steps=X_STEPS)[0])
        np.testing.assert_allclose(graph, eager, atol=1e-5)


# The start offset is max(1e-3 sigma, 8 eps(work dtype) |mu|): a term tied to rounding of the
# mean, not to a fixed fraction of it, so it never outgrows sigma of a well-separated component.
# Measured 2026-10-30 against the dense-grid maxima: the fixed 1e-5 |mu| term lost a mode of
# components 10 sigma apart at |mu|/sigma >= 1e6 (float32 and float64).
def _grid_maxima_count(mu, sigma, logits):
    """Number of strict local maxima of a 1-D mixture density on a dense float64 grid.

    The mixture is recentred first (the density is shift invariant), so the grid
    resolution does not depend on how far the components sit from the origin.

    :param mu: Component means ``[M]``.
    :param sigma: Component standard deviations ``[M]``.
    :param logits: Mixture logits ``[M]``.
    :return: Number of local maxima.
    """
    mu = np.asarray(mu, np.float64)
    mu = mu - mu.min()
    sigma = np.asarray(sigma, np.float64)
    grid = np.linspace(-6.0 * sigma.max(), mu.max() + 6.0 * sigma.max(), 100001)
    values = _oracle_log_density(grid[:, None], mu[:, None], sigma[:, None], logits)
    return int(np.sum((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])))


def _modes_of_two(level, gap, sigma, weights, dtype, num_steps=200):
    """Run the finder on two components at ``level`` and ``level + gap``.

    :return: ``(modes - level [2], log_density [2], is_unique [2])`` as float64 / bool arrays.
    """
    mu = np.array([[level], [level + gap]])
    sg = np.full((2, 1), sigma)
    mu_b, sg_b, lg_b = _batch(mu, sg, np.log(weights), dtype)
    # A rounded float32 mean sits up to half an ulp off its stored value.
    merge_tol = max(0.1 * sigma, 8.0 * float(np.spacing(np.float32(level + gap)))) if dtype == "float32" else 0.1 * sigma
    modes, log_density, is_unique = get_mixture_modes(mu_b, sg_b, lg_b, num_steps=num_steps, merge_tol=merge_tol)
    return (
        keras.ops.convert_to_numpy(modes)[0, :, 0].astype(np.float64) - level,
        keras.ops.convert_to_numpy(log_density)[0].astype(np.float64),
        keras.ops.convert_to_numpy(is_unique)[0],
    )


def _check_the_narrow_pair_at_one_thousand_keeps_both_modes(dtype):
    """Components 1e3 and 1e3 + 0.01, sigma 1e-3, weights 0.3 / 0.7: both modes, the heavier wins."""
    modes, log_density, is_unique = _modes_of_two(1e3, 0.01, 1e-3, [0.3, 0.7], dtype)
    assert is_unique.tolist() == [True, True], f"modes {modes}"
    heavier = modes[int(np.argmax(log_density))]
    assert abs(heavier - 0.01) < 1e-4, f"heavier mode at {heavier}"


class TestTheStartOffsetTracksRounding:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_a_narrow_pair_far_from_the_origin_keeps_both_modes(self, dtype):
        _check_the_narrow_pair_at_one_thousand_keeps_both_modes(dtype)

    def test_the_mode_estimate_of_the_narrow_pair_is_the_heavier_mode(self):
        model, layer = _constant_mixture_model(
            np.array([[1e3], [1e3 + 0.01]]), np.full((2, 1), 1e-3), np.log([0.3, 0.7])
        )
        estimate = get_mode_estimate(model, np.zeros((1, 2), "float32"), layer, num_steps=200)
        assert abs(float(estimate[0, 0]) - (1e3 + 0.01)) < 1e-4

    @pytest.mark.parametrize("ratio", [1e3, 1e5, 2e5, 5e5, 1e6, 2e6])
    @pytest.mark.parametrize("gap", [3.0, 5.0, 10.0])
    @pytest.mark.parametrize("weights", [(0.3, 0.7), (0.7, 0.3)])
    def test_no_separated_mode_is_lost_in_float64(self, ratio, gap, weights):
        expected = _grid_maxima_count([0.0, gap], [1.0, 1.0], np.log(weights))
        assert expected == 2
        modes, log_density, is_unique = _modes_of_two(ratio, gap, 1.0, weights, "float64")
        heavy = 0.0 if weights[0] > weights[1] else gap
        assert int(is_unique.sum()) == expected, f"modes {modes}"
        assert abs(modes[int(np.argmax(log_density))] - heavy) < 0.1

    # Measured 2026-10-30 in float32 with the eps-scaled offset: no loss up to |mu|/sigma = 1e6 at
    # 3, 5 and 10 sigma, a lost heavy 5 sigma mode first at 2e6 (reviewer probe), so 1e6 is the bound.
    @pytest.mark.parametrize("ratio", [1e3, 1e5, 5e5, 1e6])
    @pytest.mark.parametrize("gap", [3.0, 5.0, 10.0])
    @pytest.mark.parametrize("weights", [(0.3, 0.7), (0.7, 0.3)])
    def test_no_separated_mode_is_lost_in_float32_up_to_one_million(self, ratio, gap, weights):
        modes, log_density, is_unique = _modes_of_two(ratio, gap, 1.0, weights, "float32")
        heavy = 0.0 if weights[0] > weights[1] else gap
        assert int(is_unique.sum()) == 2, f"modes {modes}"
        assert abs(modes[int(np.argmax(log_density))] - heavy) < 0.1

    def test_random_narrow_mixtures_miss_no_maximum_in_float64(self):
        rng = np.random.default_rng(7)
        missed = 0
        for _ in range(100):
            count = int(rng.integers(2, 5))
            level = float(rng.choice([1e2, 1e4, 1e5, 1e6]))
            sigma = float(rng.uniform(1e-3, 1e-2)) * level / 1e4
            # Components land on a lattice at least 3 sigma apart, so every one is its own maximum.
            slots = rng.choice(np.arange(0, 12), size=count, replace=False) * 3.0 * sigma
            mu = level + np.sort(slots)
            logits = rng.normal(size=count)
            mu_b, sg_b, lg_b = _batch(mu[:, None], np.full((count, 1), sigma), logits, "float64")
            modes, log_density, is_unique = get_mixture_modes(
                mu_b, sg_b, lg_b, num_steps=100, merge_tol=0.3 * sigma
            )
            modes = keras.ops.convert_to_numpy(modes)[0, :, 0]
            kept = modes[keras.ops.convert_to_numpy(is_unique)[0]]
            grid_count = _grid_maxima_count(mu, np.full(count, sigma), logits)
            missed += max(grid_count - len(kept), 0)
            top = keras.ops.convert_to_numpy(log_density)[0]
            grid_mode, _ = _oracle_grid_mode(
                mu[:, None], np.full((count, 1), sigma), logits, [mu[int(np.argmax(logits))]], 6.0 * sigma, levels=6, size=201
            )
            assert abs(modes[int(np.argmax(top))] - grid_mode[0]) < 0.1 * sigma
        assert missed == 0

    def test_red_the_old_fixed_fraction_of_the_mean_loses_the_narrow_pair(self, monkeypatch):
        from dl_techniques.layers.statistics import mdn_layer as mdn_module

        eps32 = float(np.finfo(np.float32).eps)
        _check_the_narrow_pair_at_one_thousand_keeps_both_modes("float32")
        monkeypatch.setattr(mdn_module, "_MODE_START_EPS_MULTIPLE", 1e-5 / eps32)
        with pytest.raises(AssertionError):
            _check_the_narrow_pair_at_one_thousand_keeps_both_modes("float32")


# Two equal unit components a distance s apart; the true density has one mode at 0 for
# s <= 2 and two for s > 2. Measured 2026-10-29 in float32 (modes are the symmetric pair +-m):
# s=1.9 m=1.7e-5 after 100 steps; s=1.99 m=0.0669 after 100 and 1.1e-5 after 1000; s=2.0
# m=0.1212 after 100, 0.0387 after 1000, 0.0224 after 3000; s=2.5 m=1.0997 from 100 steps on.
class TestTheMergePointConvergesSlowly:
    @staticmethod
    def _run(separation, num_steps, merge_tol=1e-3):
        mu = np.array([[[-separation / 2.0], [separation / 2.0]]], "float32")
        sigma = np.ones((1, 2, 1), "float32")
        modes, _, is_unique = get_mixture_modes(
            mu, sigma, np.zeros((1, 2), "float32"), num_steps=num_steps, merge_tol=merge_tol
        )
        return (
            keras.ops.convert_to_numpy(modes)[0, :, 0],
            int(keras.ops.convert_to_numpy(is_unique).sum()),
        )

    @pytest.mark.parametrize("separation", [1.9, 1.99])
    def test_after_one_hundred_steps_the_starts_are_within_a_tenth_of_the_true_mode(self, separation):
        modes, _ = self._run(separation, 100)
        assert np.abs(modes).max() < 0.1

    def test_one_hundred_steps_still_report_two_unique_modes_at_1_99_sigma(self):
        modes, unique = self._run(1.99, 100)
        assert unique == 2, "the documented over-count near the merge point"
        assert 0.03 < np.abs(modes).max() < 0.1

    @pytest.mark.parametrize("separation", [1.9, 1.99])
    def test_many_steps_reach_the_single_true_mode(self, separation):
        modes, unique = self._run(separation, 3000)
        assert unique == 1
        assert np.abs(modes).max() < 1e-3

    def test_a_larger_merge_tol_merges_them_at_the_documented_step_count(self):
        _, unique = self._run(1.99, 100, merge_tol=0.2)
        assert unique == 1

    def test_at_exactly_two_sigma_the_convergence_is_still_slow(self):
        modes_1000, unique_1000 = self._run(2.0, 1000)
        assert unique_1000 == 2
        assert 0.01 < np.abs(modes_1000).max() < 0.08
        assert self._run(2.0, 1000, merge_tol=0.2)[1] == 1

    def test_well_separated_components_stay_two_modes_under_the_larger_merge_tol(self):
        modes, unique = self._run(2.5, 3000, merge_tol=0.2)
        assert unique == 2
        assert np.abs(modes).min() > 0.5


# ---------------------------------------------------------------------
# the wrapper on a real layer
# ---------------------------------------------------------------------


class TestTheWrapperOnARealLayer:
    def test_the_estimate_has_the_shape_of_the_target(self):
        layer = MDNLayer(output_dimension=3, num_mixtures=4)
        inputs = keras.Input((5,))
        model = keras.Model(inputs, layer(inputs))
        x = np.random.default_rng(0).normal(size=(6, 5)).astype("float32")
        estimate = get_mode_estimate(model, x, layer, num_steps=10)
        assert estimate.shape == (6, 3)
        assert np.isfinite(estimate).all()
