"""RED proof for `adaptive_ema`'s build() override (review finding C-37(a)).

`AdaptiveEMASlopeFilterModel.build()` created only the two threshold scalars
and deliberately did NOT build its sub-layers, on the rule "Keras builds them
lazily on first call". That rule holds only for a class that does not override
`build`. Because this one does, `keras.Model.build_from_config`
(`keras/src/models/model.py:409-412`) takes the `self.build(input_shape)`
branch instead of the build-by-run branch, so on `load_model` the
`slope_featurizer` Conv1D and the `QuantileSequenceHead` did not exist when
weights were restored.

The test gap that hid it: `test_model.py::test_serialization_round_trip` runs
the real `.save`/`load_model` only on `learnable_config`, which has NO head and
therefore no sub-layer weights at all; the quantile-head config is exercised
only through `from_config`, which builds nothing.

Two arms, per `plans/SYSTEM.md`:
- a real `.save()` -> `load_model()` round trip compared on output VALUES;
- an explicit-build vs lazy WEIGHT COUNT comparison, because "build only what
  `call()` runs" is the other half of the contract: building an unused
  sub-layer creates weights the lazy path never created and silently changes
  the `.keras` layout.

CPU only.
"""

import os
import tempfile

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.time_series.adaptive_ema.model import (
    AdaptiveEMASlopeFilterModel,
)

QUANTILE_CONFIG = {
    "ema_period": 10,
    "lookback_period": 5,
    "quantile_head_config": {"num_quantiles": 9},
}
NO_HEAD_CONFIG = {"ema_period": 10, "lookback_period": 5}


def _series(batch=4, length=64):
    rng = np.random.default_rng(0)
    return np.cumsum(rng.standard_normal((batch, length)), axis=1).astype("float32")


class TestQuantileHeadRoundTrip:

    def test_save_load_preserves_output_values(self):
        """The whole model, saved and reloaded, must produce the SAME numbers.

        Not shapes, not `count_params()`: a model whose sub-layers were absent
        at restore time matches on both and differs on values.
        """
        x = _series()
        model = AdaptiveEMASlopeFilterModel(**QUANTILE_CONFIG)
        before = model(x, training=False)
        assert "slope_quantiles" in before

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaptive_ema_quantile.keras")
            model.save(path)
            restored = keras.saving.load_model(path)

        after = restored(x, training=False)
        for key in before:
            a = ops.convert_to_numpy(before[key])
            b = ops.convert_to_numpy(after[key])
            np.testing.assert_allclose(
                a, b, atol=1e-6,
                err_msg=f"round-trip value mismatch on '{key}'",
            )

    def test_reloaded_head_weights_are_the_saved_ones(self):
        """Sharper than the output comparison: the head's kernels must match.

        A restore into non-existent layers leaves freshly-initialized weights,
        which this catches directly rather than through their effect.
        """
        x = _series()
        model = AdaptiveEMASlopeFilterModel(**QUANTILE_CONFIG)
        model(x, training=False)
        saved = [ops.convert_to_numpy(w).copy()
                 for w in model.slope_featurizer.weights]
        assert saved, "the featurizer has no weights to compare"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaptive_ema_quantile.keras")
            model.save(path)
            restored = keras.saving.load_model(path)

        reloaded = [ops.convert_to_numpy(w)
                    for w in restored.slope_featurizer.weights]
        assert len(reloaded) == len(saved)
        for s, r in zip(saved, reloaded):
            np.testing.assert_allclose(
                s, r, atol=0.0,
                err_msg="slope_featurizer weights were not restored",
            )


class TestBuildsExactlyWhatCallRuns:
    """plans/SYSTEM.md: build only what `call()` runs, enforced on COUNTS."""

    @pytest.mark.parametrize("config,label", [
        (QUANTILE_CONFIG, "quantile_head"),
        (NO_HEAD_CONFIG, "no_head"),
    ])
    def test_explicit_build_matches_lazy_weight_count(self, config, label):
        x = _series()

        lazy = AdaptiveEMASlopeFilterModel(**config)
        lazy(x, training=False)

        explicit = AdaptiveEMASlopeFilterModel(**config)
        explicit.build((None, x.shape[1]))

        # Strip the auto-generated model instance name
        # (`adaptive_ema_slope_filter_model_24/...`), which differs per object
        # and says nothing about the weight set.
        def _relative(model):
            return sorted(w.path.split("/", 1)[-1] for w in model.weights)

        lazy_names = _relative(lazy)
        explicit_names = _relative(explicit)
        assert explicit_names == lazy_names, (
            f"[{label}] explicit build created a different weight set than the "
            f"lazy path.\nonly explicit: "
            f"{sorted(set(explicit_names) - set(lazy_names))}\nonly lazy: "
            f"{sorted(set(lazy_names) - set(explicit_names))}"
        )
        assert len(explicit.weights) == len(lazy.weights)

    def test_no_head_config_builds_no_head_weights(self):
        """ANTI-VACUITY: the guard above would pass if BOTH built everything.

        Without a quantile head, `call()` never touches a featurizer, so an
        explicit build must not create one.
        """
        model = AdaptiveEMASlopeFilterModel(**NO_HEAD_CONFIG)
        model.build((None, 64))
        assert model.slope_featurizer is None
        assert model.quantile_head is None
        names = [w.path for w in model.weights]
        assert not any("featurizer" in n or "quantile" in n for n in names), names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
