"""RED proof for xLSTMForecaster's multivariate point-mode de-normalization (C-34).

`xLSTMForecaster` applies a reversible instance-norm whose statistics are
per-channel, `(B, 1, F)`. The point head emits `(B, H, F)` — one value per
feature — but the inverse used `_get_target_stats`, which slices the LAST
feature's `(B, 1, 1)` stats and broadcasts them across all F channels. Channels
`0..F-2` therefore came back scaled by another series' std and shifted by its
mean. That is correct for TiRex (one target) and was inherited verbatim.

**The probe.** The instance-norm is exactly shift- and scale-equivariant per
channel: adding a constant `c` to channel `k` of the input leaves the normalized
tensor `x = (v - mean) / std` bit-identical, so the head's raw output is
unchanged. A correct inverse therefore shifts output channel `k` by exactly `c`
and leaves every other channel untouched. Under the defect, shifting a
non-last channel changes nothing at all — the last channel's mean is what gets
added back. Multiplying channel `k` by `a > 0` gives the analogous scale
identity.

This is derived from the RevIN definition, not from the implementation, and it
needs no reference model: the expected output is a function of the model's own
unshifted output.

Both existing point-mode fixtures use `num_features=1`, the one regime in which
`mean[:, :, -1:] == mean`, so they are structurally blind to this.

CPU only. Channel scales are deliberately three orders of magnitude apart so a
cross-channel statistic cannot pass by coincidence.
"""

import numpy as np
import pytest
import keras

from dl_techniques.models.time_series.xlstm.forecaster import xLSTMForecaster

B = 2
L = 32
H = 4
F = 3
EMBED_DIM = 16
NUM_LAYERS = 1
MLSTM_NUM_HEADS = 4

# Deliberately different units per channel: ~1e-2, ~1e0, ~1e3.
CHANNEL_SCALE = np.array([0.01, 1.0, 1000.0], dtype="float32")
CHANNEL_OFFSET = np.array([-5.0, 0.0, 250.0], dtype="float32")


def _point_model() -> xLSTMForecaster:
    keras.utils.set_random_seed(1234)
    return xLSTMForecaster(
        input_length=L,
        prediction_length=H,
        num_features=F,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        mlstm_num_heads=MLSTM_NUM_HEADS,
        use_quantile_head=False,
        use_normalization=True,
        dropout_rate=0.0,
    )


def _context() -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.standard_normal((B, L, F)).astype("float32")
    return base * CHANNEL_SCALE + CHANNEL_OFFSET


def _predict(model, x) -> np.ndarray:
    return keras.ops.convert_to_numpy(model(x, training=False))


class TestMultivariatePointDenormalization:

    def test_shift_of_one_channel_moves_only_that_channel(self):
        """Adding c to input channel k must add exactly c to output channel k.

        RevIN is shift-equivariant per channel, so the normalized tensor — and
        therefore the head's raw output — is unchanged by the shift. Only the
        additive inverse moves.
        """
        model = _point_model()
        x = _context()
        base = _predict(model, x)
        assert base.shape == (B, H, F)

        for k in range(F):
            c = float(50.0 * CHANNEL_SCALE[k])
            shifted = x.copy()
            shifted[:, :, k] += c
            out = _predict(model, shifted)

            delta = out - base
            expected = np.zeros_like(delta)
            expected[:, :, k] = c

            # Compare in each channel's OWN units: channel 2 lives at ~1e3, so
            # a shared absolute tolerance would either be blind on channel 0
            # (~1e-2) or trip on float32 rounding on channel 2.
            np.testing.assert_allclose(
                delta / CHANNEL_SCALE, expected / CHANNEL_SCALE, atol=1e-3,
                err_msg=(
                    f"shifting input channel {k} by {c} must move output "
                    f"channel {k} by exactly {c} and no other channel; "
                    f"got per-channel deltas {delta.mean(axis=(0, 1))}"
                ),
            )

    def test_scaling_one_channel_scales_only_that_channel(self):
        """Multiplying input channel k by a must multiply output channel k by a.

        RevIN is scale-equivariant per channel (mean and std both scale by a),
        so the normalized tensor is unchanged. This arm is independent of the
        shift arm: it pins `std`, where the shift arm pins `mean`.
        """
        model = _point_model()
        x = _context()
        base = _predict(model, x)

        for k in range(F):
            a = 4.0
            scaled = x.copy()
            scaled[:, :, k] *= a
            out = _predict(model, scaled)

            expected = base.copy()
            expected[:, :, k] = base[:, :, k] * a

            np.testing.assert_allclose(
                out / CHANNEL_SCALE, expected / CHANNEL_SCALE,
                rtol=1e-4, atol=1e-2,
                err_msg=(
                    f"scaling input channel {k} by {a} must scale output "
                    f"channel {k} by {a} and leave the rest fixed"
                ),
            )

    def test_anti_vacuity_the_probe_can_see_a_change(self):
        """The probe is not trivially satisfied by a constant output.

        If the model emitted the same numbers regardless of input, both arms
        above would still fail (they demand a specific non-zero delta). This
        arm proves the shift genuinely reaches the output at all, so a
        `delta == 0` result under the defect is a real observation and not a
        dead forward pass.
        """
        model = _point_model()
        x = _context()
        base = _predict(model, x)
        # Shift the LAST channel: the defect and the fix AGREE here, so this
        # must move the output under both, in every channel under the defect
        # and only in the last channel under the fix.
        shifted = x.copy()
        shifted[:, :, F - 1] += 100.0
        out = _predict(model, shifted)
        assert np.abs(out[:, :, F - 1] - base[:, :, F - 1]).max() > 1.0

    def test_quantile_mode_still_uses_the_target_feature_only(self):
        """The quantile head's (B,H,Q) output stays on the LAST feature's stats.

        Guards against "fix" by deleting `_get_target_stats`: a (B,1,F) inverse
        cannot even broadcast against Q quantiles unless Q == F, and where it
        can it would be wrong.
        """
        keras.utils.set_random_seed(1234)
        model = xLSTMForecaster(
            input_length=L,
            prediction_length=H,
            num_features=F,
            embed_dim=EMBED_DIM,
            num_layers=NUM_LAYERS,
            mlstm_num_heads=MLSTM_NUM_HEADS,
            use_quantile_head=True,
            quantile_levels=[0.1, 0.5, 0.9],
            use_normalization=True,
            dropout_rate=0.0,
        )
        x = _context()
        base = _predict(model, x)
        assert base.shape == (B, H, 3)

        # Shifts are sized in each channel's own units. A shift huge relative
        # to a channel's std (e.g. +100 on a channel whose std is 1e-2) makes
        # `(v - mean)` cancel catastrophically in float32 and moves the
        # normalized tensor by ~1e-3 relative — a measurement artifact, not a
        # de-normalization defect. MEASURED: it produced a 1.87e-2 relative
        # drift on CPU and would have made this arm a false positive.
        c0 = float(50.0 * CHANNEL_SCALE[0])
        shifted = x.copy()
        shifted[:, :, 0] += c0
        out = _predict(model, shifted)
        np.testing.assert_allclose(
            out / CHANNEL_SCALE[-1], base / CHANNEL_SCALE[-1], atol=1e-3,
            err_msg="quantile forecasts of the last feature moved when a "
                    "non-target channel was shifted",
        )

        # Shifting the target channel must move them by exactly that amount.
        c_last = float(50.0 * CHANNEL_SCALE[-1])
        shifted = x.copy()
        shifted[:, :, F - 1] += c_last
        out = _predict(model, shifted)
        np.testing.assert_allclose(
            (out - base) / CHANNEL_SCALE[-1],
            np.full_like(base, c_last) / CHANNEL_SCALE[-1], atol=1e-3,
            err_msg="quantile forecasts did not track the target's own mean",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
