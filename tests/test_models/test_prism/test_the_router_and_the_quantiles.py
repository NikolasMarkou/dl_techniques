"""PRISM's two central claims: a temperature-scaled band ROUTER, and
non-crossing quantiles.

Why this file exists
--------------------
`router_temperature`, `num_wavelet_levels` and `overlap_ratio` appear in
`test_prism/test_model.py` only inside the fixture and inside
`assert key in config` -- a constructor echo. `enforce_monotonicity` was never
exercised at all, even though "``Q_i <= Q_{i+1}`` holds by construction rather
than by penalty" is the model docstring's own headline claim about the quantile
head.

Two claims, each with its dead component run in the same test:

1. **Router.** `FrequencyBandRouter` emits `softmax(scores / temperature)` over
   the bands. The weights must sum to 1 and must SHARPEN as the temperature
   falls. MEASURED 2026-08-18 (4 bands, hidden 16, seed 5): mean max-weight
   0.6127 / 0.2959 / 0.2545 and entropy 0.8490 / 1.3736 / 1.3862 at
   temperature 0.1 / 1.0 / 10.0. The dead component is the uniform router,
   whose entropy is exactly `log(4) = 1.3863` -- which is what temperature 10
   already almost is, so the discriminating comparison is against temperature
   0.1.
2. **Quantiles.** With `enforce_monotonicity=True` the predicted quantiles must
   be non-decreasing along the quantile axis. MEASURED: zero crossings and a
   minimum successive difference of +0.004229; with the flag OFF, 50.0% of
   successive pairs cross and the minimum difference is -7.2855. That contrast
   is the RED proof.

Scope: `tree_depth` is held at 1-2 throughout, now only because these tests are
about the ROUTER and the quantile head, not about the tree. The depth-3 defect
D-039 recorded (`PRISMModel(tree_depth=3)` all-NaN at initialization) was FIXED
by plan-2026-08-18T073231-52a93f8c: the cause was a length-1 deepest band, whose
empty first-difference tensor returned NaN silently and which the router's
single joint softmax then spread across every band. Depth 3 is finite and swept
in `test_model.py`; depth 4 at `context_len=96` is now refused by `__init__`
with a `ValueError`. The governing quantity is
`min_band_len = deepest_leaf_seg // 2 ** num_wavelet_levels`, NOT a `tree_depth`
range -- a depth-2 config (96/2/4) was equally broken and a depth-4 one
(256/4/3) was always fine.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.prism_blocks import FrequencyBandRouter
from dl_techniques.models.time_series.prism.model import PRISMModel


NUM_BANDS = 4


def _router_weights(temperature: float) -> np.ndarray:
    keras.utils.set_random_seed(5)
    router = FrequencyBandRouter(
        hidden_dim=16, temperature=temperature, dropout_rate=0.0
    )
    rng = np.random.default_rng(0)
    bands = [
        keras.ops.convert_to_tensor(
            (rng.normal(size=(2, 16, 3)) * (1 + index)).astype("float32")
        )
        for index in range(NUM_BANDS)
    ]
    return np.asarray(keras.ops.convert_to_numpy(router(bands, training=False)))


def _entropy(weights: np.ndarray) -> float:
    return float((-(weights * np.log(weights + 1e-12)).sum(-1)).mean())


class TestFrequencyBandRouter:
    def test_the_weights_are_a_distribution_over_bands(self):
        weights = _router_weights(1.0)
        assert weights.shape[-1] == NUM_BANDS
        np.testing.assert_allclose(
            weights.sum(-1), np.ones(weights.shape[:-1], dtype="float32"),
            rtol=0, atol=1e-5,
        )
        assert float(weights.min()) > 0.0

    # DECISION plan-2026-08-22T035419-a11304c8/D-033
    # The sharpening claim is made on a LADDER and at T=0.01, not on a single
    # reading at T=0.1 against `0.9 * log(4)`. That earlier bar was RED at
    # baseline -- `even at temperature 0.1 the router is nearly UNIFORM (entropy
    # 1.2703 against the uniform 1.3863)` -- and the recorded diagnosis ("is the
    # temperature knob reaching the router at all?") is REFUTED by measurement:
    #
    #   * The knob reaches the softmax. `FrequencyBandRouter.call` computes
    #     `ops.softmax(scores / self.temperature, axis=-1)`
    #     (`layers/time_series/prism_blocks.py:428`), and the entropy at the
    #     test's own seed 5 walks 1.3863 -> 1.3850 -> 1.2703 -> 0.1396 across
    #     T = 10 / 1 / 0.1 / 0.01 (mean max-weight 0.2516 -> 0.9661).
    #   * What moved is the DRAW, not the mechanism. Entropy at T=0.1 is a
    #     seed-dependent random variable: measured over 20 seeds it spans
    #     0.4024 .. 1.3254 (mean 1.0135). The 0.8490 this file recorded on
    #     2026-08-18 and the 1.2703 measured at baseline are both ordinary
    #     members, and the `0.9 * log(4)` = 1.2477 bar is INSIDE that population
    #     -- 4 of 20 seeds sit above it. Nothing in
    #     `prism_blocks.py` changed the router since; the only commit touching
    #     the file (`38c7493c6`) rewrote the DYNAMIC-shape NaN predicate in
    #     `FrequencyBandStatistics`, a branch this static-shaped test never
    #     enters.
    #
    # The ladder assertion is a THEOREM, not a sample: for fixed scores the
    # entropy of `softmax(s / T)` is strictly increasing in T unless every score
    # is equal -- and "every score equal" IS the dead router this test exists to
    # convict. Measured strictly monotone in 20 of 20 seeds. The magnitude arm
    # is taken at T=0.01, where the entropy population over 20 seeds is
    # 0.0020 .. 0.3448 (max = 0.2487 of uniform), so the 0.5 * uniform floor
    # clears the worst observed draw by 1.45x.
    def test_a_lower_temperature_sharpens_the_routing(self):
        """The knob was checked only by `assert model.router_temperature == x`."""
        uniform_entropy = float(np.log(NUM_BANDS))  # 1.3863: the dead router
        ladder = [10.0, 1.0, 0.1, 0.01]
        entropies = [_entropy(_router_weights(t)) for t in ladder]

        for hotter, cooler, hot_e, cool_e in zip(
                ladder, ladder[1:], entropies, entropies[1:]
        ):
            assert cool_e < hot_e, (
                f"temperature does not sharpen the routing: entropy {cool_e:.4f} "
                f"at {cooler} is not below {hot_e:.4f} at {hotter}"
            )

        assert entropies[-1] < 0.5 * uniform_entropy, (
            f"even at temperature {ladder[-1]} the router is nearly UNIFORM "
            f"(entropy {entropies[-1]:.4f} against the uniform "
            f"{uniform_entropy:.4f}); it is not expressing a preference "
            "between bands at all"
        )
        # ...and the high-temperature end must approach the uniform router,
        # which is the control that the statistic means what it says.
        # (Measured |entropy(10.0) - log(4)| <= 4.01e-04 over 20 seeds.)
        assert entropies[0] == pytest.approx(uniform_entropy, abs=0.01)


class TestQuantileHeadDoesNotCross:
    """`Q_i <= Q_{i+1}` by construction -- the model docstring's own claim."""

    @staticmethod
    def _model(enforce: bool) -> PRISMModel:
        keras.utils.set_random_seed(9)
        return PRISMModel(
            context_len=48,
            forecast_len=12,
            num_features=3,
            hidden_dim=32,
            num_layers=1,
            # Depth 2 because these tests are about the ROUTER and the
            # quantile head, not the tree. D-039's "depth 3 is all-NaN" is no
            # longer true -- it was fixed by
            # plan-2026-08-18T073231-52a93f8c. At this config depth 3 would
            # still be legal but degenerate: measured, deepest_leaf_seg 6 and
            # min_band_len 6 >> 2 == 1 (depth 2 gives seg 12, min_band_len 3).
            tree_depth=2,
            overlap_ratio=0.25,
            num_wavelet_levels=2,
            router_hidden_dim=16,
            router_temperature=1.0,
            dropout_rate=0.0,
            use_quantile_head=True,
            num_quantiles=5,
            enforce_monotonicity=enforce,
        )

    @staticmethod
    def _quantiles(model: PRISMModel) -> np.ndarray:
        x = np.random.default_rng(0).normal(size=(4, 48, 3)).astype("float32")
        out = model(keras.ops.convert_to_tensor(x), training=False)
        return np.asarray(keras.ops.convert_to_numpy(out))

    def test_enforced_quantiles_never_cross(self):
        quantiles = self._quantiles(self._model(enforce=True))
        assert quantiles.shape == (4, 12, 3, 5)
        assert np.isfinite(quantiles).all()
        differences = np.diff(quantiles, axis=-1)
        # Measured minimum successive difference +0.004229, zero crossings.
        assert float(differences.min()) >= 0.0, (
            f"{float((differences < 0).mean()) * 100:.1f}% of successive "
            f"quantile pairs CROSS (min difference "
            f"{float(differences.min()):.6f}) with enforce_monotonicity=True"
        )

    def test_the_guard_is_red_without_the_enforcement(self):
        """RED proof: the same assertion, with the mechanism switched off."""
        differences = np.diff(self._quantiles(self._model(enforce=False)), axis=-1)
        # Measured: 50.0% crossing, min difference -7.2855.
        assert float(differences.min()) < 0.0, (
            "with enforce_monotonicity=False the quantiles still never cross, "
            "so the enforced case proves nothing about the mechanism"
        )


# ---------------------------------------------------------------------------
# DECISION plan-2026-08-19T163559-499b6f0e/D-117
#
# `predict_quantiles` chose its point-forecast head with
# `if 0.5 in self.quantile_levels: ... else: len(self.quantile_levels) // 2`.
#
# RE-DERIVED, AND THE CARRIED MECHANISM IS ONLY HALF RIGHT. The float `in` test
# really does miss on every even `num_quantiles` -- but for the class's OWN
# `np.linspace(0, 1, n + 2)[1:-1]` fallback the positional middle and the
# nearest-to-0.5 index are always equidistant from 0.5 (measured n=2..11: they
# differ only at n in {4, 8, 10}, and there they straddle 0.5 symmetrically, so
# neither is more correct). The linspace half of the finding is REFUTED.
#
# What is live is the `len(...) // 2` fallback itself: it reads POSITION, never
# VALUE, so an asymmetric caller-supplied level set picks an arbitrary head.
# ---------------------------------------------------------------------------
def _median_head_index(levels):
    """The repaired selection rule, as `PRISMModel.predict_quantiles` runs it."""
    return int(np.argmin(np.abs(np.array(levels) - 0.5)))


def _legacy_median_head_index(levels):
    """The pre-fix rule, verbatim."""
    if 0.5 in levels:
        return levels.index(0.5)
    return len(levels) // 2


def test_an_asymmetric_level_set_picks_the_head_nearest_the_median():
    levels = [0.01, 0.05, 0.1, 0.48, 0.9]
    assert _legacy_median_head_index(levels) == 2   # -> the 0.10 head
    assert _median_head_index(levels) == 3          # -> the 0.48 head
    assert levels[_median_head_index(levels)] == 0.48


@pytest.mark.parametrize("levels", [
    [0.1, 0.5, 0.9],
    [0.25, 0.5, 0.75],
    [0.05, 0.5, 0.95],
])
def test_an_exact_median_still_selects_the_same_head_as_before(levels):
    """Bit-identical on every level set that actually contains 0.5."""
    assert _median_head_index(levels) == _legacy_median_head_index(levels)


def test_the_model_uses_the_value_based_rule(monkeypatch):
    """RED against the real source: pin the rule inside `predict_quantiles`."""
    keras.utils.set_random_seed(9)
    model = PRISMModel(
        context_len=48, forecast_len=12, num_features=3, hidden_dim=32,
        num_layers=1, tree_depth=2, overlap_ratio=0.25, num_wavelet_levels=2,
        router_hidden_dim=16, router_temperature=1.0, dropout_rate=0.0,
        use_quantile_head=True, num_quantiles=5,
        quantile_levels=[0.01, 0.05, 0.1, 0.48, 0.9],
    )
    context = np.zeros((2, 48, 3), dtype="float32")
    captured = {}

    def _fake_predict(self, x, **kwargs):
        out = np.zeros((2, 12, 3, 5), dtype="float32")
        # Tag each quantile head with its own index so the chosen column is
        # identifiable in the returned point forecast.
        for i in range(5):
            out[..., i] = float(i)
        return out

    monkeypatch.setattr(type(model), "predict", _fake_predict, raising=True)
    _quantiles, point = model.predict_quantiles(context, quantile_levels=[0.9])
    captured["idx"] = float(np.unique(point)[0])
    assert captured["idx"] == 3.0, (
        "predict_quantiles took the POSITIONAL middle head "
        f"({captured['idx']:.0f}) instead of the head nearest 0.5 (3)"
    )
