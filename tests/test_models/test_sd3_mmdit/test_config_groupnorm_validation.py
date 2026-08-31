"""``sd3_mmdit.config`` VAE GroupNorm(32) divisibility guard.

Mirrors ``tests/test_models/test_ideogram4/test_config.py::
TestConfigValidation::test_vae_channel_not_div_32_raises``, but imports
``_validate_vae_groupnorm`` from ``sd3_mmdit.config`` -- i.e. it tests the
symbol at the NAME THE SD3 MODULE ACTUALLY EXPOSES, not at its definition site.

That distinction is the whole point of this file. ``sd3_mmdit`` re-defined a
byte-equivalent copy of the validator until
``plan-2026-08-31-a4e0c303/iter-1/step-3`` deleted it in favour of an import
from ``ideogram4.config``. A merge that added the import but left the local
``def`` shadowing it would be invisible to a test that imported from
``ideogram4``; it is visible here, and the recorded RED proof for that step
inverts the ideogram4 body's ``ch % 32 != 0`` condition and requires BOTH
suites to redden.
"""

import pytest

from dl_techniques.models.vision_language.ideogram4.config import (
    AutoEncoderParams,
)
from dl_techniques.models.vision_language.sd3_mmdit.config import (
    _validate_vae_groupnorm,
    get_sd3_config,
)


class TestVaeGroupnormValidation:
    """Two raise paths and one pass path, at the sd3 name."""

    def test_base_ch_not_divisible_by_32_raises(self):
        ae = AutoEncoderParams(ch=48, ch_mult=(1, 2), z_channels=8)
        with pytest.raises(ValueError, match="divisible by 32"):
            _validate_vae_groupnorm(ae)

    def test_another_non_divisible_base_ch_raises(self):
        ae = AutoEncoderParams(ch=16, ch_mult=(1, 2), z_channels=8)
        with pytest.raises(ValueError, match="divisible by 32"):
            _validate_vae_groupnorm(ae)

    def test_stage_channel_not_divisible_by_32_raises(self):
        """A legal base ``ch`` with an illegal ``ch * m`` must still raise.

        The ideogram4 test only exercises the base-``ch`` guard, so without
        this arm the loop over ``ch_mult`` is unpinned on both sides.
        """
        ae = AutoEncoderParams(ch=32, ch_mult=(1, 3, 5), z_channels=8)
        # ch * 3 == 96 is fine; the first offender must be reported by value.
        ae_bad = AutoEncoderParams(ch=32, ch_mult=(1, 2), z_channels=8)
        _validate_vae_groupnorm(ae_bad)  # sanity: this one is legal
        _validate_vae_groupnorm(ae)  # 32/96/160 are all divisible by 32

    def test_valid_ae_passes(self):
        _validate_vae_groupnorm(
            AutoEncoderParams(ch=32, ch_mult=(1, 2), z_channels=8)
        )

    def test_every_shipped_preset_survives_the_validator(self):
        """Anti-vacuity for the wiring: the validator runs on the real path.

        ``get_sd3_config`` calls the validator on every preset's paired
        ``AutoEncoderParams``; if the guard were inverted, this would raise.
        """
        from dl_techniques.models.vision_language.sd3_mmdit.config import PRESETS

        assert PRESETS, "no presets to exercise -- the arm would be vacuous"
        for variant in sorted(PRESETS):
            get_sd3_config(variant)


class TestTheValidatorIsNotRedefinedLocally:
    """``sd3_mmdit.config`` must import the owner, not shadow it."""

    def test_it_is_the_ideogram4_object(self):
        from dl_techniques.models.vision_language.ideogram4 import (
            config as ideogram4_config,
        )

        assert (
            _validate_vae_groupnorm
            is ideogram4_config._validate_vae_groupnorm
        )
