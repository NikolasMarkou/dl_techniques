"""Guard: the `--no-gabor-projection` channel rule is ONE rule with ONE implementation.

Both bfunet trainers used to carry a verbatim copy of the same pre-flight check, and the
rule itself changed when the Gabor stem became a cross-channel `Conv2D`: it is now
`gabor_filters == initial_filters`, NOT `channels * gabor_filters == initial_filters`.
Two hand-maintained copies of a rule that just changed is exactly how a trainer and a
builder end up disagreeing about what a flag means, so the check now lives once in
`train.bfunet.common.validate_gabor_stem_channels` and this module pins:

  1. the predicate itself, including that the OLD rule no longer satisfies it;
  2. that both trainers actually route through it (a shared helper nobody calls is
     worse than a duplicate -- it looks like coverage and provides none); and
  3. that the model builders raise the SAME way as a backstop, so removing the
     trainer-side check would still fail loudly rather than silently.

Run:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \\
        tests/test_train/test_bfunet/test_the_gabor_stem_channel_rule.py -q
"""

import re

import pytest

from train.bfunet.common import validate_gabor_stem_channels

# The retired `channels * gabor_filters` rule, as it would be spelled inside a trainer.
# `\b` after `gabor_filters` is load-bearing: it excludes `gabor_filters_per_channel`,
# the depthwise stem's count, whose width legitimately IS channels-multiplied. See
# `test_trainer_has_no_private_copy_of_the_rule`.
_RETIRED_RULE_RE = re.compile(r"config\.channels\s*\*\s*config\.gabor_filters\b")
import train.bfunet.train_unet_denoiser as unet_trainer
import train.bfunet.train_convunext_denoiser as convunext_trainer

from dl_techniques.models.vision.bias_free_denoisers.bfunet import (
    create_bfunet_denoiser,
)
from dl_techniques.models.vision.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
)


class TestTheSharedPredicate:
    def test_matching_widths_are_admissible(self):
        # Returns None; the whole effect is raising or not raising.
        assert validate_gabor_stem_channels(
            use_gabor_stem=True, gabor_stem_projection=False,
            gabor_filters=32, initial_filters=32) is None

    def test_mismatched_widths_raise(self):
        with pytest.raises(ValueError, match="gabor_filters"):
            validate_gabor_stem_channels(
                use_gabor_stem=True, gabor_stem_projection=False,
                gabor_filters=8, initial_filters=32)

    def test_the_old_channels_times_filters_rule_no_longer_satisfies_it(self):
        """3-channel input x 8 filters == 24 was legal under the depthwise stem.

        It is not legal now, and the failure must be loud. If this test ever goes
        green by returning None, the trainer has silently reverted to the old rule.
        """
        with pytest.raises(ValueError, match="no longer applies"):
            validate_gabor_stem_channels(
                use_gabor_stem=True, gabor_stem_projection=False,
                gabor_filters=8, initial_filters=24)

    @pytest.mark.parametrize("use_stem,projection", [
        (False, False), (False, True), (True, True),
    ])
    def test_inert_unless_the_stem_is_on_and_the_projection_is_off(
            self, use_stem, projection):
        """The rule only binds on the one branch that has no projection to fix a
        width mismatch. On every other branch a mismatch is legal and must not raise."""
        assert validate_gabor_stem_channels(
            use_gabor_stem=use_stem, gabor_stem_projection=projection,
            gabor_filters=8, initial_filters=32) is None


class TestBothTrainersRouteThroughIt:
    """A shared helper that nobody calls is worse than a duplicate."""

    @pytest.mark.parametrize("mod", [unet_trainer, convunext_trainer],
                             ids=["train_unet_denoiser", "train_convunext_denoiser"])
    def test_trainer_imports_the_shared_predicate(self, mod):
        assert mod.validate_gabor_stem_channels is validate_gabor_stem_channels

    @pytest.mark.parametrize("mod", [unet_trainer, convunext_trainer],
                             ids=["train_unet_denoiser", "train_convunext_denoiser"])
    def test_trainer_has_no_private_copy_of_the_rule(self, mod):
        """The deleted duplicate must stay deleted.

        Both copies computed `channels * gabor_filters` inline. That expression
        reappearing in either trainer means the duplicate is back and the two can
        drift again.

        The match is anchored on a WORD BOUNDARY after `gabor_filters`, not a bare
        substring. `standalone-2026-09-07-depthwise-gabor-stem/D-002` added a second,
        legitimately-multiplied count named `gabor_filters_per_channel` -- the depthwise
        stem really does emit `channels * gabor_filters_per_channel`, and that
        identifier merely STARTS with `gabor_filters`. A substring test reports it as
        the retired rule, which is a false positive against a different quantity. The
        boundary keeps the guard on `gabor_filters` ITSELF, where the retired rule is
        still wrong; `\\s*\\*\\s*` additionally catches spacing variants the old literal
        would have missed, so this is strictly stronger on the pattern it targets.
        `test_the_no_private_copy_guard_can_fail` is its RED proof.
        """
        import inspect
        assert _RETIRED_RULE_RE.search(inspect.getsource(mod)) is None, (
            f"{mod.__name__} re-inlined the old channels*gabor_filters rule"
        )

    def test_the_no_private_copy_guard_can_fail(self):
        """RED proof: the tightened pattern must still catch the real re-inlining.

        Guards that only ever run green prove nothing, and this one was just narrowed.
        Both the original spelling and a spacing variant must be caught, while the
        depthwise identifier must not be.
        """
        for offender in ("x = config.channels * config.gabor_filters\n",
                         "x = config.channels*config.gabor_filters\n",
                         "if config.channels * config.gabor_filters != n:\n"):
            assert _RETIRED_RULE_RE.search(offender) is not None, (
                f"the guard no longer catches the retired rule: {offender!r}"
            )
        assert _RETIRED_RULE_RE.search(
            "w = config.channels * config.gabor_filters_per_channel\n"
        ) is None, (
            "the guard fires on gabor_filters_per_channel, a DIFFERENT count whose "
            "width genuinely does multiply by channels"
        )


class TestTheBuildersAreTheBackstop:
    """Both model builders must enforce the same rule independently of the trainers."""

    @pytest.mark.parametrize("build", [create_bfunet_denoiser, create_convunext_denoiser],
                             ids=["bfunet", "convunext"])
    def test_builder_raises_on_width_mismatch(self, build):
        with pytest.raises(ValueError, match="gabor_filters"):
            build(input_shape=(16, 16, 3), depth=2, initial_filters=16,
                  blocks_per_level=1, use_gabor_stem=True, gabor_filters=8,
                  gabor_kernel_size=5, gabor_stem_projection=False)

    @pytest.mark.parametrize("build", [create_bfunet_denoiser, create_convunext_denoiser],
                             ids=["bfunet", "convunext"])
    def test_builder_accepts_matching_widths(self, build):
        model = build(input_shape=(16, 16, 3), depth=2, initial_filters=8,
                      blocks_per_level=1, use_gabor_stem=True, gabor_filters=8,
                      gabor_kernel_size=5, gabor_stem_projection=False)
        assert model.get_layer('gabor_stem').output.shape[-1] == 8
