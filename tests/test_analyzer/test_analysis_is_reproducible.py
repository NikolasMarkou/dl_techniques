"""S12 - nothing in `dl_techniques.analyzer` was seeded.

Complete census of the stochastic sites, all of them on the global NumPy RNG:
`utils.py:95` and `:133` (`DataSampler`), `spectral_analyzer.py:328`
(`np.random.permutation`), `spectral_metrics.py:470` (the goodness-of-fit bootstrap)
and `spectral_metrics.py:1195` (`_power_iteration`). The first two are on the DEFAULT
path via `model_analyzer.py:451`, so the analysis SAMPLE itself changed run to run and
every calibration and confidence number moved with it.

The guards are placed at the DATA PIPELINE, not at a model forward pass: a
reproducibility flag must be proven where the randomness enters.
"""

import numpy as np
import pytest

from dl_techniques.analyzer.config import AnalysisConfig
from dl_techniques.analyzer.data_types import DataInput
from dl_techniques.analyzer.utils import DataSampler


N_TOTAL = 400
N_SAMPLES = 25


@pytest.fixture()
def population() -> DataInput:
    """A population large enough that a repeated draw cannot coincide by luck."""
    x = np.arange(N_TOTAL, dtype="float32").reshape(N_TOTAL, 1)
    y = np.arange(N_TOTAL)
    return DataInput(x_data=x, y_data=y)


def _drawn(data: DataInput, rng=None):
    """Return the drawn population indices - `y_data` IS the index here."""
    return tuple(DataSampler.sample(data, N_SAMPLES, rng=rng).y_data.tolist())


class TestDataSamplerIsSeedable:
    """`DataSampler` is the single sampling chokepoint on the default path."""

    def test_the_same_seed_draws_the_same_indices(self, population):
        first = _drawn(population, np.random.default_rng(1234))
        second = _drawn(population, np.random.default_rng(1234))

        assert len(set(first)) == N_SAMPLES, "the draw was not without replacement"
        assert first == second, (
            f"the same seed drew different samples:\n{first}\n{second}")

    def test_a_different_seed_draws_different_indices(self, population):
        """Anti-vacuity: seeding must not have collapsed the draw to a constant."""
        first = _drawn(population, np.random.default_rng(1234))
        other = _drawn(population, np.random.default_rng(4321))
        assert first != other, f"two different seeds drew the same sample: {first}"

    def test_the_unseeded_path_still_varies(self, population):
        """Anti-vacuity: the default (rng=None) path must remain stochastic."""
        draws = {_drawn(population) for _ in range(5)}
        assert len(draws) > 1, (
            "five unseeded draws were all identical; the unseeded path is no longer "
            "random and the seeded/unseeded distinction is untestable")

    def test_dict_inputs_are_seeded_too(self):
        """`_sample_dict_inputs` is the second `np.random.choice` site."""
        x = {"a": np.arange(N_TOTAL, dtype="float32").reshape(N_TOTAL, 1)}
        data = DataInput(x_data=x, y_data=np.arange(N_TOTAL))

        first = DataSampler.sample(
            data, N_SAMPLES, rng=np.random.default_rng(7)).y_data
        second = DataSampler.sample(
            data, N_SAMPLES, rng=np.random.default_rng(7)).y_data
        other = DataSampler.sample(
            data, N_SAMPLES, rng=np.random.default_rng(8)).y_data

        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first, other), (
            f"two different seeds drew the same dict sample: {first}")


class TestTheSpectralStochasticSitesAreSeedable:
    """The other three census sites take an `rng` too, keyword-only."""

    def test_the_goodness_of_fit_bootstrap_is_seedable(self):
        from dl_techniques.analyzer.spectral_metrics import powerlaw_goodness_of_fit

        rng_draw = np.random.default_rng(3)
        evals = np.sort(rng_draw.pareto(2.0, size=200) + 1.0)[::-1]

        kwargs = dict(alpha=3.0, xmin=float(np.percentile(evals, 50)),
                      n_bootstraps=20)
        first = powerlaw_goodness_of_fit(
            evals, rng=np.random.default_rng(11), **kwargs)
        second = powerlaw_goodness_of_fit(
            evals, rng=np.random.default_rng(11), **kwargs)

        assert first == second, f"same seed gave p={first} then p={second}"
        # Anti-vacuity: the test really ran, rather than returning the
        # "not computed" sentinel on both calls.
        assert first != -1.0, "the bootstrap did not run at all"

    def test_power_iteration_is_seedable(self):
        from dl_techniques.analyzer.spectral_metrics import get_top_eigenvectors

        rng_draw = np.random.default_rng(5)
        w = rng_draw.normal(size=(30, 20))

        first = get_top_eigenvectors(
            w, k=2, method="power_iteration", rng=np.random.default_rng(2))
        second = get_top_eigenvectors(
            w, k=2, method="power_iteration", rng=np.random.default_rng(2))

        # Anti-vacuity: the power-iteration branch really produced something.
        assert first[0].shape == (2,) and first[1].shape == (30, 2)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_the_randomized_spectrum_is_seedable(self):
        import keras

        from dl_techniques.analyzer.analyzers.spectral_analyzer import (
            SpectralAnalyzer,
        )

        inputs = keras.Input(shape=(40,), name="rs_in")
        model = keras.Model(
            inputs, keras.layers.Dense(30, name="rs_d")(inputs), name="rsm")

        def run(seed):
            config = AnalysisConfig(
                spectral_randomize=True, spectral_n_randomizations=2,
                spectral_bootstraps=0, random_state=seed, verbose=False)
            details, _, rand_esds, _, _ = SpectralAnalyzer(
                models={"rsm": model},
                config=config)._analyze_single_model(model)
            return details, rand_esds

        details_a, rand_a = run(21)
        details_b, rand_b = run(21)
        details_c, _ = run(22)

        # Anti-vacuity: randomization really ran.
        assert rand_a and rand_b, "no randomized spectra were produced"

        column = "rand_distance"
        assert column in details_a, f"columns: {list(details_a.columns)}"
        np.testing.assert_allclose(
            details_a[column].to_numpy(), details_b[column].to_numpy(), rtol=0,
            atol=0)
        assert not np.array_equal(
            details_a[column].to_numpy(), details_c[column].to_numpy()), (
            "two different seeds produced identical randomized spectra")


class TestTheConfigSeedReachesTheAnalysis:
    """`analyze()` at a fixed `random_state` must be reproducible end to end."""

    @staticmethod
    def _model():
        import keras

        keras.utils.set_random_seed(3)
        inputs = keras.Input(shape=(4,), name="rp_in")
        hidden = keras.layers.Dense(6, activation="relu", name="rp_d")(inputs)
        out = keras.layers.Dense(3, activation="softmax", name="rp_out")(hidden)
        return keras.Model(inputs, out, name="rpm")

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        return DataInput(
            x_data=rng.standard_normal((N_TOTAL, 4)).astype("float32"),
            y_data=rng.integers(0, 3, size=N_TOTAL),
        )

    def _ece(self, tmp_path, seed, tag):
        from dl_techniques.analyzer.model_analyzer import ModelAnalyzer

        analyzer = ModelAnalyzer(
            models={"rpm": self._model()},
            config=AnalysisConfig(
                analyze_weights=False, analyze_calibration=True,
                analyze_information_flow=False, analyze_training_dynamics=False,
                analyze_spectral=False, n_samples=N_SAMPLES, save_plots=False,
                verbose=False, random_state=seed),
            output_dir=str(tmp_path / tag),
        )
        analyzer.analyze(data=self._data(), analysis_types={"calibration"})
        metrics = analyzer.results.calibration_metrics["rpm"]
        assert metrics, "no calibration metrics were produced"
        return metrics["ece"]

    def test_the_same_random_state_gives_the_same_numbers(self, tmp_path):
        first = self._ece(tmp_path, 99, "a")
        second = self._ece(tmp_path, 99, "b")
        assert first == second, (
            f"ece moved between two runs at random_state=99: {first} vs {second}")

    def test_a_different_random_state_gives_different_numbers(self, tmp_path):
        """Anti-vacuity: the seed must select a genuinely different sample."""
        first = self._ece(tmp_path, 99, "c")
        other = self._ece(tmp_path, 100, "d")
        assert first != other, (
            f"random_state 99 and 100 both gave ece={first}; the seed is not "
            "reaching the sampler, or the sample is not being subsampled at all")
