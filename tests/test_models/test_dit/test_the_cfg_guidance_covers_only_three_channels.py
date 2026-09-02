"""Quirk guard: classifier-free guidance covers THREE channels, not ``in_channels``.

**The line this file pins.**
``src/dl_techniques/models/vision_language/dit/model.py``, ``DiT.forward_with_cfg``::

    eps  = model_out[..., :CFG_GUIDED_CHANNELS]     # CFG_GUIDED_CHANNELS == 3
    rest = model_out[..., CFG_GUIDED_CHANNELS:]

Upstream (``reference/models.py``, ``forward_with_cfg``) writes ``model_out[:, :3]``
on the channel axis of an NCHW tensor and leaves the ``in_channels`` form
COMMENTED OUT one line above, with the note *"for exact reproducibility reasons,
we apply classifier-free guidance on only three channels by default. The
standard approach to cfg applies it to all channels."* See decisions.md D-014 and
the in-code anchor at the slice.

**The plausible WRONG alternative this file is RED against.**
``eps = model_out[..., :self.in_channels]`` -- the "obvious fix". At the
published ``in_channels = 4`` it guides one extra epsilon channel. Every shape,
dtype, parameter count, ``get_config()`` and ``.keras`` round trip is IDENTICAL,
and so is every finiteness check.

**How this file convicts it, and why the obvious probe does not.**
Step 6's ``test_dit_model.py`` measured that injection **INERT: 0 failed / 44
passed**, because its arm compared the two output halves in the guided channels
only -- and both slicings make the guided halves equal. The discriminator is the
UNGUIDED remainder:

* channels ``0..2`` of BOTH halves carry ``uncond + s * (cond - uncond)``, so the
  halves agree there under either slicing;
* channel ``3`` is passed through UNTOUCHED, so its first half is the raw
  CONDITIONAL prediction and its second half the raw UNCONDITIONAL one -- and
  those two differ. Under the ``in_channels`` slicing channel 3 is guided too and
  its two halves become bit-identical.

The arms therefore assert (a) channel ``3``'s two halves DIFFER, and (b)
``out[..., 3:]`` is bit-identical to the raw model output at ``atol=0``. A
precondition arm measures ``cond != uncond`` in channel 3 first, so neither claim
can pass vacuously.

**The model must be woken up first.** A freshly built ``DiT`` emits exactly
``0.0`` everywhere, so ``cond`` and ``uncond`` are equal in every channel and the
whole discriminator collapses. Every arm here runs on
``_dit_helpers.activate(built_model(...))``, which replaces the zero-initialised
trainable weights with real draws; ``TestTheProbeIsNotVacuous`` pins that this
step is load-bearing rather than decorative.

**RED proof (step 10).** Injecting ``eps = model_out[..., :self.in_channels]`` /
``rest = model_out[..., self.in_channels:]`` into ``model.py``:
**6 failed / 9 passed** in this file. The five behavioural arms that fired --
``test_the_remainder_is_bit_identical_to_the_raw_model_output``,
``test_the_ungoverned_channel_still_distinguishes_the_two_halves``,
``test_only_the_first_three_channels_carry_the_guidance_algebra``,
``test_the_number_of_guided_channels_does_not_follow_in_channels``,
``test_a_five_channel_model_still_guides_exactly_three`` -- plus the text arm
``test_the_slice_carries_its_decision_anchor``.
``test_at_in_channels_three_the_two_slicings_coincide`` stayed GREEN, exactly as
it documents: at ``in_channels = 3`` the two slicings ARE the same slicing.
"""

from typing import Any, Tuple

import numpy as np
import pytest

from dl_techniques.models.vision_language.dit.model import CFG_GUIDED_CHANNELS, DiT

from ._dit_helpers import BATCH, TINY, activate, built_model, np_, tiny_inputs

#: Guidance strength used throughout. Not 1.0 and not 0.0: at ``s = 1`` the
#: guided value collapses onto ``cond`` and at ``s = 0`` onto ``uncond``, and
#: either would make the algebra arm satisfiable by the wrong expression.
CFG_SCALE: float = 3.0


def guided_case(
    seed: int = 0, batch: int = BATCH, **overrides: Any
) -> Tuple[DiT, np.ndarray, np.ndarray, np.ndarray]:
    """A woken model plus the ``(x, t, y)`` triple ``forward_with_cfg`` expects.

    Interface contract: ``y``'s first half holds real class labels and its
    second half the null row ``num_classes``, which is what makes the second
    half of the forward pass unconditional. Everything is drawn from a local
    generator; the model's weights come from the seeded helpers, so two calls
    with the same arguments are bit-identical.

    :param seed: Seed for construction, activation and the input draw.
    :type seed: int
    :param batch: Batch size. Must be even.
    :type batch: int
    :param overrides: Constructor kwargs replacing the ``TINY`` entries.
    :type overrides: Any
    :return: ``(model, x, t, y)``.
    :rtype: Tuple[DiT, np.ndarray, np.ndarray, np.ndarray]
    :raises ValueError: If ``batch`` is odd.
    """
    if batch % 2:
        raise ValueError(f"forward_with_cfg needs an even batch, got {batch}")
    model = activate(built_model(seed=seed, **overrides), seed=seed + 5)
    rng = np.random.default_rng(seed + 1)
    n, c = model.input_size, model.in_channels
    x = rng.normal(size=(batch, n, n, c)).astype("float32")
    t = rng.integers(0, 1000, size=(batch,)).astype("float32")
    labels = np.arange(batch // 2, dtype="int32") % model.num_classes
    null = np.full((batch // 2,), model.num_classes, dtype="int32")
    return model, x, t, np.concatenate([labels, null])


def raw_and_guided(
    model: DiT, x: np.ndarray, t: np.ndarray, y: np.ndarray, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_model_output, forward_with_cfg_output)`` for one case.

    Interface contract: reproduces ``forward_with_cfg``'s batch trick -- the
    first half of ``x`` duplicated -- so ``raw`` is exactly the tensor the method
    guides internally. It calls the model directly rather than reaching into the
    method, which is what lets an arm compare the two.
    """
    batch = x.shape[0]
    half = x[: batch // 2]
    combined = np.concatenate([half, half], axis=0)
    raw = np_(model([combined, t, y], training=False))
    out = np_(model.forward_with_cfg(x, t, y, cfg_scale=scale, training=False))
    return raw, out


# ---------------------------------------------------------------------
# The probe has to be capable of seeing anything at all
# ---------------------------------------------------------------------


class TestTheProbeIsNotVacuous:
    """Everything below rests on ``cond != uncond`` in the ungoverned channel."""

    def test_an_unwoken_model_makes_every_channel_indistinguishable(self) -> None:
        """The trap: at init the whole discriminator reads exactly ``0.0``."""
        model = built_model(seed=0)
        x, t, _ = tiny_inputs(seed=1, batch=BATCH)
        y = np.concatenate(
            [
                np.arange(BATCH // 2, dtype="int32"),
                np.full((BATCH // 2,), TINY["num_classes"], dtype="int32"),
            ]
        )
        raw, _ = raw_and_guided(model, x, t, y, CFG_SCALE)
        assert float(np.max(np.abs(raw))) == 0.0

    def test_a_woken_model_separates_the_conditional_from_the_unconditional(
        self,
    ) -> None:
        model, x, t, y = guided_case(seed=0)
        raw, _ = raw_and_guided(model, x, t, y, CFG_SCALE)
        half = raw.shape[0] // 2
        for channel in range(model.out_channels):
            spread = float(
                np.max(np.abs(raw[:half, ..., channel] - raw[half:, ..., channel]))
            )
            assert spread > 0.0, f"channel {channel} is conditioning-blind"


# ---------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------


class TestTheGuidanceAlgebra:
    """``uncond + s * (cond - uncond)`` on channels 0..2, nothing else touched."""

    def test_the_guided_channels_are_exactly_the_upstream_algebra(self) -> None:
        model, x, t, y = guided_case(seed=0)
        raw, out = raw_and_guided(model, x, t, y, CFG_SCALE)
        half = raw.shape[0] // 2

        cond = raw[:half, ..., :CFG_GUIDED_CHANNELS]
        uncond = raw[half:, ..., :CFG_GUIDED_CHANNELS]
        expected = uncond + CFG_SCALE * (cond - uncond)

        np.testing.assert_allclose(
            out[:half, ..., :CFG_GUIDED_CHANNELS], expected, rtol=0, atol=0.0
        )
        # Both halves carry the SAME guided value -- upstream's
        # `torch.cat([half_eps, half_eps], dim=0)`.
        np.testing.assert_allclose(
            out[half:, ..., :CFG_GUIDED_CHANNELS], expected, rtol=0, atol=0.0
        )

    def test_the_remainder_is_bit_identical_to_the_raw_model_output(self) -> None:
        """The discriminator step 6's arm did not have."""
        model, x, t, y = guided_case(seed=0)
        raw, out = raw_and_guided(model, x, t, y, CFG_SCALE)
        assert model.out_channels > CFG_GUIDED_CHANNELS
        np.testing.assert_allclose(
            out[..., CFG_GUIDED_CHANNELS:],
            raw[..., CFG_GUIDED_CHANNELS:],
            rtol=0,
            atol=0.0,
        )

    def test_the_ungoverned_channel_still_distinguishes_the_two_halves(self) -> None:
        """Channel 3 at ``in_channels = 4``: cond in one half, uncond in the other.

        Under ``model_out[..., :in_channels]`` the two halves of channel 3 become
        bit-identical, which is what this arm refuses.
        """
        model, x, t, y = guided_case(seed=0)
        assert model.in_channels == 4 and CFG_GUIDED_CHANNELS == 3
        raw, out = raw_and_guided(model, x, t, y, CFG_SCALE)
        half = out.shape[0] // 2

        first = out[:half, ..., CFG_GUIDED_CHANNELS]
        second = out[half:, ..., CFG_GUIDED_CHANNELS]
        assert float(np.max(np.abs(first - second))) > 0.0

        # And they are the raw conditional / unconditional values, unmixed.
        np.testing.assert_allclose(
            first, raw[:half, ..., CFG_GUIDED_CHANNELS], rtol=0, atol=0.0
        )
        np.testing.assert_allclose(
            second, raw[half:, ..., CFG_GUIDED_CHANNELS], rtol=0, atol=0.0
        )

    def test_only_the_first_three_channels_carry_the_guidance_algebra(self) -> None:
        """A per-channel census, so the count 3 is measured rather than assumed."""
        model, x, t, y = guided_case(seed=0)
        raw, out = raw_and_guided(model, x, t, y, CFG_SCALE)
        half = raw.shape[0] // 2

        guided = []
        for channel in range(model.out_channels):
            cond = raw[:half, ..., channel]
            uncond = raw[half:, ..., channel]
            mixed = uncond + CFG_SCALE * (cond - uncond)
            got = out[:half, ..., channel]
            is_guided = np.allclose(got, mixed, rtol=0, atol=1e-6)
            is_passthrough = np.array_equal(got, cond)
            assert is_guided != is_passthrough, (
                f"channel {channel} is neither cleanly guided nor cleanly "
                "passed through -- the probe cannot classify it"
            )
            if is_guided:
                guided.append(channel)

        assert guided == list(range(CFG_GUIDED_CHANNELS)), guided

    @pytest.mark.parametrize("scale", [0.0, 1.0, 2.5, 7.5])
    def test_the_algebra_holds_at_every_scale(self, scale: float) -> None:
        model, x, t, y = guided_case(seed=2)
        raw, out = raw_and_guided(model, x, t, y, scale)
        half = raw.shape[0] // 2
        cond = raw[:half, ..., :CFG_GUIDED_CHANNELS]
        uncond = raw[half:, ..., :CFG_GUIDED_CHANNELS]
        np.testing.assert_allclose(
            out[:half, ..., :CFG_GUIDED_CHANNELS],
            uncond + scale * (cond - uncond),
            rtol=0,
            atol=0.0,
        )


# ---------------------------------------------------------------------
# The count is THREE, and it does not track in_channels
# ---------------------------------------------------------------------


class TestTheConstantIsThreeAndIndependentOfInChannels:
    """``CFG_GUIDED_CHANNELS`` is a literal 3, not a function of the config."""

    def test_the_constant_is_three(self) -> None:
        assert CFG_GUIDED_CHANNELS == 3

    def test_the_number_of_guided_channels_does_not_follow_in_channels(self) -> None:
        """At ``in_channels = 5`` exactly one MORE channel is left ungoverned."""
        model, x, t, y = guided_case(seed=3, in_channels=5)
        assert model.in_channels == 5
        raw, out = raw_and_guided(model, x, t, y, CFG_SCALE)
        half = raw.shape[0] // 2
        for channel in (CFG_GUIDED_CHANNELS, 4):
            np.testing.assert_allclose(
                out[:half, ..., channel],
                raw[:half, ..., channel],
                rtol=0,
                atol=0.0,
                err_msg=f"channel {channel} was guided",
            )
            assert float(
                np.max(
                    np.abs(out[:half, ..., channel] - out[half:, ..., channel])
                )
            ) > 0.0

    def test_a_five_channel_model_still_guides_exactly_three(self) -> None:
        model, x, t, y = guided_case(seed=3, in_channels=5)
        raw, out = raw_and_guided(model, x, t, y, CFG_SCALE)
        half = raw.shape[0] // 2
        guided = [
            channel
            for channel in range(model.out_channels)
            if np.allclose(
                out[:half, ..., channel],
                raw[half:, ..., channel]
                + CFG_SCALE
                * (raw[:half, ..., channel] - raw[half:, ..., channel]),
                rtol=0,
                atol=1e-6,
            )
            and not np.array_equal(out[:half, ..., channel], raw[:half, ..., channel])
        ]
        assert guided == list(range(CFG_GUIDED_CHANNELS)), guided

    def test_at_in_channels_three_the_two_slicings_coincide(self) -> None:
        """The one configuration where the quirk is INVISIBLE, pinned as such.

        At ``in_channels = 3`` upstream's literal and the commented-out
        alternative select the same channels, so no probe on this configuration
        can distinguish them. Stating that here stops a future reader from
        "simplifying" the guards above onto a 3-channel model and quietly losing
        the whole claim.
        """
        model, x, t, y = guided_case(seed=4, in_channels=3)
        assert model.in_channels == CFG_GUIDED_CHANNELS
        raw, out = raw_and_guided(model, x, t, y, CFG_SCALE)
        # The variance half (channels 3..5) is still untouched -- learn_sigma
        # doubles the width, so `[..., 3:]` is not empty even here.
        np.testing.assert_allclose(
            out[..., CFG_GUIDED_CHANNELS:],
            raw[..., CFG_GUIDED_CHANNELS:],
            rtol=0,
            atol=0.0,
        )


# ---------------------------------------------------------------------
# The anchor exists and says what it must
# ---------------------------------------------------------------------


class TestTheDivergenceIsAnchoredInTheSource:
    """A quirk with no in-code anchor gets "fixed" by the next reader."""

    def test_the_slice_carries_its_decision_anchor(self) -> None:
        import inspect

        source = inspect.getsource(DiT.forward_with_cfg)
        assert "DECISION plan-2026-09-02T170923-1285ed83/D-014" in source
        assert "WHAT NOT TO DO" in source
        assert "self.in_channels" in source, (
            "the anchor must name the wrong alternative explicitly"
        )
        assert "model_out[..., :CFG_GUIDED_CHANNELS]" in source
