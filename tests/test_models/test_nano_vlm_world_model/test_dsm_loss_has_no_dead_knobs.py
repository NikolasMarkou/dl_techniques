"""F-23: three advertised knobs on `DenoisingScoreMatchingLoss` reached nothing.

``prediction_type`` ('epsilon' / 'sample' / 'v_prediction'), ``loss_weight_type``
('uniform' / 'snr' / 'truncated_snr', with a Hang et al. 2023 citation) and
``min_snr_gamma`` were stored, serialized and never read: ``call()`` has always
been an unconditional MSE. The aggravating detail is that the class DEFAULTED to
``prediction_type='epsilon'`` on a package whose denoisers are all x_0
predictors, and the one in-repo caller (``VLMDenoisingLoss``) passed
``prediction_type='sample'`` -- two contradictory settings that produced
identical numbers because neither reached anything.

They are removed rather than implemented (decisions.md
plan-2026-08-18T140459-7991552f/D-034): ``call(y_true, y_pred)`` receives no
timesteps and no alphas, so SNR weighting is not implementable at this
signature, and the parameterisation is chosen by ``DiffusionScheduler``, which
has a live ``prediction_type`` of its own.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.nano_vlm_world_model.train import (
    DenoisingScoreMatchingLoss,
    VLMDenoisingLoss,
)


class TestTheDeadKnobsAreGone:

    @pytest.mark.parametrize("key,value", [
        ("prediction_type", "v_prediction"),
        ("loss_weight_type", "snr"),
        ("min_snr_gamma", 2.0),
    ])
    def test_each_is_refused_by_name(self, key, value):
        with pytest.raises((TypeError, ValueError), match=key):
            DenoisingScoreMatchingLoss(**{key: value})

    def test_none_survives_into_the_serialized_config(self):
        config = DenoisingScoreMatchingLoss().get_config()
        for key in DenoisingScoreMatchingLoss._LEGACY_DEAD_KEYS:
            assert key not in config, key

    def test_a_pre_removal_config_still_loads(self):
        """Dropping them is behaviour-preserving by construction, so a config
        saved before the removal must load, not raise."""
        legacy = DenoisingScoreMatchingLoss().get_config()
        legacy.update(
            prediction_type="epsilon", loss_weight_type="snr",
            min_snr_gamma=5.0,
        )
        restored = DenoisingScoreMatchingLoss.from_config(legacy)
        assert isinstance(restored, DenoisingScoreMatchingLoss)
        for key in DenoisingScoreMatchingLoss._LEGACY_DEAD_KEYS:
            assert not hasattr(restored, key), key

    def test_the_internal_caller_no_longer_passes_one(self):
        """``VLMDenoisingLoss`` used to construct its sub-loss with
        ``prediction_type='sample'``."""
        combined = VLMDenoisingLoss()
        for key in DenoisingScoreMatchingLoss._LEGACY_DEAD_KEYS:
            assert not hasattr(combined.dsm_loss, key), key


class TestTheLossIsExactlyAnUnweightedMSE:
    """What the class actually computes, measured -- the reason the knobs were
    dead rather than merely undocumented."""

    def test_it_equals_the_plain_mean_squared_error(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=(4, 5, 3)).astype("float32")
        y_pred = rng.normal(size=(4, 5, 3)).astype("float32")

        got = float(ops.convert_to_numpy(
            DenoisingScoreMatchingLoss()(y_true, y_pred)
        ))
        expected = float(np.mean((y_pred - y_true) ** 2))
        assert got == pytest.approx(expected, rel=1e-6), (
            "the DSM loss is not a plain MSE; if a weighting or a "
            "parameterisation branch has been added, the removed knobs need "
            "to come back with it"
        )

    def test_it_does_not_depend_on_the_timestep_it_never_receives(self):
        """Anti-vacuity for the removal: the signature carries no timestep, so
        no SNR weighting could have been applied even in principle."""
        import inspect
        params = inspect.signature(DenoisingScoreMatchingLoss.call).parameters
        assert set(params) == {"self", "y_true", "y_pred"}, params
