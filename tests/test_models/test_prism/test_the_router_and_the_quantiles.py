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

    def test_a_lower_temperature_sharpens_the_routing(self):
        """The knob was checked only by `assert model.router_temperature == x`."""
        sharp = _router_weights(0.1)
        flat = _router_weights(10.0)
        uniform_entropy = float(np.log(NUM_BANDS))  # 1.3863: the dead router

        sharp_entropy, flat_entropy = _entropy(sharp), _entropy(flat)
        # Measured 0.8490 vs 1.3862.
        assert sharp_entropy < flat_entropy, (
            f"temperature does not sharpen the routing: entropy {sharp_entropy:.4f} "
            f"at 0.1 vs {flat_entropy:.4f} at 10.0"
        )
        assert sharp_entropy < 0.9 * uniform_entropy, (
            f"even at temperature 0.1 the router is nearly UNIFORM "
            f"(entropy {sharp_entropy:.4f} against the uniform {uniform_entropy:.4f}); "
            "it is not expressing a preference between bands at all"
        )
        # ...and the high-temperature end must approach the uniform router,
        # which is the control that the statistic means what it says.
        assert flat_entropy == pytest.approx(uniform_entropy, abs=0.01)


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
            tree_depth=2,  # D-039: depth 3 is all-NaN; not used here
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
