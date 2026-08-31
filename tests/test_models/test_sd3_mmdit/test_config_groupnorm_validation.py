"""``sd3_mmdit.config`` VAE GroupNorm(32) divisibility guard.

Mirrors ``tests/test_models/test_ideogram4/test_config.py::
TestConfigValidation::test_vae_channel_not_div_32_raises``, but imports
``validate_vae_groupnorm`` from ``sd3_mmdit.config`` -- i.e. it tests the
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
    validate_vae_groupnorm,
    get_sd3_config,
)


class TestVaeGroupnormValidation:
    """Two raise paths, one pass path, and one unreachability proof."""

    def test_base_ch_not_divisible_by_32_raises(self):
        ae = AutoEncoderParams(ch=48, ch_mult=(1, 2), z_channels=8)
        with pytest.raises(ValueError, match="divisible by 32"):
            validate_vae_groupnorm(ae)

    def test_another_non_divisible_base_ch_raises(self):
        ae = AutoEncoderParams(ch=16, ch_mult=(1, 2), z_channels=8)
        with pytest.raises(ValueError, match="divisible by 32"):
            validate_vae_groupnorm(ae)

    def test_no_integer_ch_mult_can_reach_the_stage_check(self):
        """The ``for m in ch_mult`` branch is UNREACHABLE, proven by exhaustion.

        This replaces an arm that was named ``..._raises`` and asserted that
        nothing raises, over an ``AutoEncoderParams(ch=32, ch_mult=(1, 2))``
        that is perfectly legal -- a test whose body contradicted its own name
        and pinned nothing.

        The proof: the validator checks ``ch % 32 != 0`` FIRST, so anything
        reaching the loop has ``ch == 32 * k``; ``ch_mult`` is declared
        ``Tuple[int, ...]``, so ``ch * m == 32 * k * m`` is divisible by 32 for
        every integer ``m``. No value consistent with the declared type can
        make the second check fire.

        The oracle is the MESSAGE, not the raise: over 256 base widths x 5
        multiplier tuples, every rejection must be the BASE-``ch`` message.
        RED-PROVEN -- weakening the base check to ``ch % 16`` lets ``ch=48``
        through it and the stage loop then rejects it with the stage message,
        failing this arm (an earlier draft asserted only "nothing raises on a
        32-multiple grid" and stayed GREEN under that same mutant).

        The branch IS reachable with a non-integer multiplier (the frozen
        dataclass does no runtime type check: ``ch_mult=(1, 1.5)`` raises
        ``ch * 1.5 = 48.0``), but that violates the annotation, and pinning a
        type-contract violation is not what the deleted arm claimed to do.
        """
        multipliers = [(1,), (1, 2), (1, 2, 4, 4), (1, 3, 5), (2, 3, 7)]
        accepted = 0
        rejected = 0
        for ch in range(1, 257):
            for mult in multipliers:
                ae = AutoEncoderParams(ch=ch, ch_mult=mult, z_channels=8)
                try:
                    validate_vae_groupnorm(ae)
                except ValueError as exc:
                    rejected += 1
                    assert "base ch" in str(exc), (
                        f"ch={ch}, ch_mult={mult} was rejected by the STAGE "
                        f"check ({exc}). For integer ch_mult that branch is "
                        f"unreachable behind the base check -- reaching it "
                        f"means the base check was weakened."
                    )
                else:
                    accepted += 1
        # Anti-vacuity: the grid must exercise BOTH verdicts, or the message
        # assertion above could be satisfied by never running.
        assert accepted > 0 and rejected > 0, (
            f"grid produced {accepted} accepted / {rejected} rejected; it must "
            f"exercise both sides or it certifies nothing"
        )

    def test_valid_ae_passes(self):
        validate_vae_groupnorm(
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
            validate_vae_groupnorm
            is ideogram4_config.validate_vae_groupnorm
        )
