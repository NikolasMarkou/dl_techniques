"""``forward_with_cfg`` is ``cond + s*(cond - uncond)``, not the textbook form.

Invariant 6 of the plan, and the single most consequential correction the
EXPLORE pass made to the reconstructed architecture. Upstream (``dit.py:584``)
starts the guidance extrapolation from the CONDITIONAL output, where the DiT
paper starts it from the unconditional one. The two formulas differ by exactly
one unit of guidance -- upstream's ``s`` is the textbook's ``s + 1`` -- so a
"corrected" implementation still produces finite, plausible, well-shaped output
and reproduces none of the reference results at any published ``cfg_scale``.

Nothing structural can see this. Both formulas take the same two forward passes,
return the same shape and dtype, are equally finite, and survive a ``.keras``
round trip identically. Only the arithmetic distinguishes them, so the arms
below pin the arithmetic -- and each one carries an explicit anti-vacuity
partner asserting that the textbook value is a DIFFERENT number for the
constants and scales chosen, because at some ``(cond, uncond, s)`` combinations
the two coincide.

Two independent oracles:

1. :class:`_MaskSensitiveStub` binds the real, unmodified ``DiTXA.forward_with_cfg``
   to a forward pass that returns one constant when the conditioning stream is
   masked and another when it is not. The method's algebra is then readable in
   closed form, with no model, no initialisation and no tolerance.
2. The integration arm runs the real ``tiny`` model and compares
   ``forward_with_cfg`` against two manually issued passes.
"""

from typing import Any, Dict, Optional

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import DiTXA

from ._ditxa_helpers import activate, batch, np_

#: Two scales that provably separate the two formulas for the stub's constants.
#: ``1.0`` is deliberately excluded: the textbook form collapses to ``cond``
#: there, which is a value the correct formula also produces at ``s = 0``, and a
#: guard that used only those two points would be reading a coincidence.
SCALES = (0.5, 3.0)


class _MaskSensitiveStub:
    """The real ``forward_with_cfg`` bound to a two-valued forward pass.

    Returns :data:`COND_VALUE` everywhere when ``cond_mask`` is absent or
    non-zero, and :data:`UNCOND_VALUE` everywhere when it is all-zero. That is
    exactly the distinction ``forward_with_cfg`` is responsible for drawing, so
    a method that forgot to mask the second pass would return ``COND_VALUE``
    twice and every arm below would fail.
    """

    #: The unbound method under test. Not a copy -- an alias, so an edit to
    #: ``model.py`` is visible here immediately.
    forward_with_cfg = DiTXA.forward_with_cfg

    COND_VALUE = 2.0
    UNCOND_VALUE = 0.5

    def __init__(self) -> None:
        self.calls = []

    def __call__(
        self, inputs: Dict[str, Any], training: Optional[bool] = None
    ) -> Any:
        mask = inputs.get("cond_mask")
        masked = mask is not None and float(np.max(np.abs(np_(mask)))) == 0.0
        self.calls.append(
            {
                "masked": masked,
                "mask": None if mask is None else np_(mask),
                "training": training,
            }
        )
        value = self.UNCOND_VALUE if masked else self.COND_VALUE
        return keras.ops.full(keras.ops.shape(inputs["x_t"]), value)


def _stub_inputs(batch_size: int = 3) -> Dict[str, Any]:
    """A minimal input dict; only ``x_t`` and ``t`` are read by the stub."""
    return {
        "x_t": np.zeros((batch_size, 4, 4, 2), dtype="float32"),
        "t": np.full((batch_size,), 0.5, dtype="float32"),
        "y": np.zeros((batch_size,), dtype="int32"),
        "x_cond": np.zeros((batch_size, 4, 4, 2), dtype="float32"),
        "direction": np.zeros((batch_size,), dtype="float32"),
    }


def _ours(s: float) -> float:
    """The ported (upstream) formula on the stub's constants."""
    c, u = _MaskSensitiveStub.COND_VALUE, _MaskSensitiveStub.UNCOND_VALUE
    return c + s * (c - u)


def _textbook(s: float) -> float:
    """The DiT-paper formula on the stub's constants -- the injection's answer."""
    c, u = _MaskSensitiveStub.COND_VALUE, _MaskSensitiveStub.UNCOND_VALUE
    return u + s * (c - u)


class TestTheAlgebraIsUpstreams:
    """Closed-form arms against the stub."""

    @pytest.mark.parametrize("scale", SCALES)
    def test_the_result_is_cond_plus_s_times_the_difference(self, scale):
        """``cond + s*(cond - uncond)``, exactly."""
        stub = _MaskSensitiveStub()
        out = np_(stub.forward_with_cfg(_stub_inputs(), cfg_scale=scale))
        assert out.shape == (3, 4, 4, 2)
        np.testing.assert_allclose(out, _ours(scale), rtol=0, atol=0)

    @pytest.mark.parametrize("scale", SCALES)
    def test_the_textbook_answer_is_a_different_number(self, scale):
        """Anti-vacuity: the injection this guard exists to catch is separable.

        If ``_ours`` and ``_textbook`` agreed at these constants and scales, the
        arm above would pass under BOTH implementations and prove nothing.
        """
        assert not np.isclose(_ours(scale), _textbook(scale)), (
            f"the two formulas coincide at s={scale}; pick different constants"
        )

    def test_zero_guidance_returns_the_conditional_pass(self):
        """``s = 0`` gives ``cond`` here and would give ``uncond`` textbook-side.

        So this is a third discriminating point, not a degenerate one.
        """
        stub = _MaskSensitiveStub()
        out = np_(stub.forward_with_cfg(_stub_inputs(), cfg_scale=0.0))
        np.testing.assert_allclose(
            out, _MaskSensitiveStub.COND_VALUE, rtol=0, atol=0
        )
        assert not np.isclose(
            _MaskSensitiveStub.COND_VALUE, _MaskSensitiveStub.UNCOND_VALUE
        )


class TestTheTwoPasses:
    """The masked pass is a real all-false mask, and there are exactly two."""

    def test_exactly_two_passes_one_conditional_one_masked(self):
        stub = _MaskSensitiveStub()
        stub.forward_with_cfg(_stub_inputs(), cfg_scale=1.5)
        assert [c["masked"] for c in stub.calls] == [False, True]

    def test_the_masked_pass_gets_a_batch_shaped_all_zero_mask(self):
        stub = _MaskSensitiveStub()
        stub.forward_with_cfg(_stub_inputs(batch_size=5), cfg_scale=1.5)
        assert stub.calls[0]["mask"] is None
        mask = stub.calls[1]["mask"]
        assert mask.shape == (5,)
        np.testing.assert_array_equal(mask, np.zeros((5,), dtype=mask.dtype))

    def test_a_caller_supplied_cond_mask_is_ignored(self):
        """The conditional pass is defined as the fully unmasked one.

        A caller who hands in a mask (as the trainer's batches do) must not
        accidentally make the "conditional" pass a second unconditional one.
        """
        inputs = _stub_inputs()
        inputs["cond_mask"] = np.zeros((3,), dtype="float32")
        stub = _MaskSensitiveStub()
        out = np_(stub.forward_with_cfg(inputs, cfg_scale=SCALES[1]))
        assert [c["masked"] for c in stub.calls] == [False, True]
        np.testing.assert_allclose(out, _ours(SCALES[1]), rtol=0, atol=0)

    def test_training_is_threaded_to_both_passes(self):
        stub = _MaskSensitiveStub()
        stub.forward_with_cfg(_stub_inputs(), cfg_scale=1.0, training=False)
        assert [c["training"] for c in stub.calls] == [False, False]


@pytest.fixture(scope="module")
def model() -> DiTXA:
    """A built, non-degenerate ``tiny`` model (see ``_ditxa_helpers.activate``)."""
    m = DiTXA.from_variant("tiny", label_seed=11)
    m(batch(m, batch_size=2))
    return activate(m, seed=6)


class TestTheRealModelAgrees:
    """Integration arm: the same algebra on two genuine forward passes."""

    @pytest.mark.parametrize("scale", SCALES)
    def test_it_equals_two_manual_passes_combined_upstreams_way(
        self, model, scale
    ):
        data = batch(model, batch_size=3, seed=77)
        cond = np_(model(data, training=False))
        masked = dict(data)
        masked["cond_mask"] = np.zeros((3,), dtype="float32")
        uncond = np_(model(masked, training=False))

        # Non-degeneracy: with a zero-init model every one of these is the exact
        # zero tensor and every comparison below is vacuously true.
        assert np.abs(cond).max() > 1e-6
        assert not np.allclose(cond, uncond)

        out = np_(model.forward_with_cfg(data, cfg_scale=scale, training=False))
        np.testing.assert_allclose(
            out, cond + scale * (cond - uncond), rtol=0, atol=1e-6
        )
        assert not np.allclose(
            out, uncond + scale * (cond - uncond), atol=1e-4
        ), "the textbook formula is indistinguishable here; the arm is vacuous"
