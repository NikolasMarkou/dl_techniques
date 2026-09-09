"""The ASSEMBLED H-Net must not read the future.

H-Net is trained and documented as a byte-level *causal* language model, and every one of
its four causal mechanisms is a separate opportunity to get it wrong:

1. the attention keep predicate ``j <= i`` (:func:`components.build_causal_keep_mask`),
2. Mamba-2's sequential scan and its depthwise **causal** convolution,
3. the routing module's pair ``(h_{t-1}, h_t)`` -- a boundary at ``t`` may not depend on
   ``h_{t+1}``,
4. the chunk / dechunk round trip -- position ``t``'s chunk index is the number of
   boundaries at or before ``t``, and the dechunk EMA runs left to right.

A shape test is blind to all four, and so is every component-level test in this
directory: each of those pins ONE layer in isolation, and the guide's headline failure
class is precisely a model whose parts are individually causal while the assembly is not.
This module is therefore run on the assembled :class:`HNet`, at three layouts, so no
single mixer family can carry the claim alone:

======== ================================= =========================================
arm      ``arch_layout``                   what carries causality
======== ================================= =========================================
m_only   ``["m1", ["m1"], "m1"]``          the Mamba-2 scan + causal conv ONLY
T_only   ``["T1", ["T1"], "T1"]``          the attention keep predicate ONLY
mixed    ``["m1", ["T1"], "m1"]``          both, in the shipped sandwich shape
======== ================================= =========================================

Three arms, per plan step 13
----------------------------
* **arm 1** -- perturb the byte at ``t``; every logit at a position ``< t`` is
  **bit-identical** (``atol=0``, ``rtol=0``). The bound is exact and not a tolerance: a
  causal model's arithmetic at the earlier positions is literally independent of the
  perturbed byte, so there is no cancelling sum whose reduction order could move.
* **arm 2** -- at least one logit at a position ``>= t`` moves. Arm 1 alone passes on a
  model that ignores its input entirely, so this is not optional.
* **arm 3** -- arm 1 repeated with the perturbation at the LAST position, where the
  "before" window is the whole sequence bar one, together with the requirement that the
  last position itself moved.

The trap this file had to measure its way out of
------------------------------------------------
**At initialisation the probe is DEAD at the last position, and arm 3 would have passed
for the wrong reason.** ``residual_proj`` is zero-initialised by construction (D-019), so
a freshly built stage is a pure pass-through of the DEchunked inner result: a position
that the routing module did not select as a boundary contributes its own hidden state to
*nothing*, and its output depends only on earlier chunks. MEASURED at init, perturbing the
byte at the last position of a 12-byte input moves the output by exactly
``0.000000e+00`` at EVERY position, all three layouts -- the model is not ignoring the
future, it is ignoring that byte. Arm 3 would then have been satisfied by a disconnected
network, which is trap 1 of ``test_lewm_causality.py`` in H-Net's own clothing.

:func:`_activate_residual` assigns a seeded non-zero ``residual_proj`` kernel, standing in
for a trained model, and
:meth:`TestTheProbeIsLive::test_at_init_the_last_byte_reaches_nothing_at_all` pins that
this step is load-bearing rather than decorative.

Negative controls -- two, because one of them is family-specific
----------------------------------------------------------------
* **The time-reversed comparator** (all three layouts). ``flip(model(flip(x)))`` is
  ANTI-causal by construction, built from the SAME instance and the same weights, with no
  re-initialisation and no RNG draw. Under it a perturbation at ``t`` MUST reach the
  positions ``< t``, which is exactly the window arm 1 watches -- so a probe that reads
  ``0.0`` there for a reason other than causality is caught.
* **The keep predicate removed** (``T_only`` and ``mixed`` only). ``build_causal_keep_mask``
  is monkeypatched to an all-ones predicate on the SAME instance, the leak is required to
  become non-zero, the patch is undone and the leak is required to return to exactly
  ``0.0``. For ``m_only`` the same patch must change **nothing at all** -- that layout
  builds no attention, and asserting the zero states in executable form why the mask
  control cannot be the whole story.

MEASURED (CPU, ``CUDA_VISIBLE_DEVICES=""``), perturbation at ``t = 5`` of a 12-byte input:

========= ============ ============= ================= =============== ==============
layout    leak (< t)   signal (>= t) reversed leak      mask-off leak   restored leak
========= ============ ============= ================= =============== ==============
m_only    0.000000e+00 3.332824e-01  1.992505e-01      0.000000e+00    0.000000e+00
T_only    0.000000e+00 2.350478e-01  2.657645e-01      4.320741e-04    0.000000e+00
mixed     0.000000e+00 2.705553e-01  2.182963e-01      1.964793e-04    0.000000e+00
========= ============ ============= ================= =============== ==============

The mask-off leak is small because at initialisation every residual-WRITING projection --
attention's ``w_o`` included -- is depth-scaled to ``0.02 / sqrt(4) = 0.01`` (D-021), so
attention contributes little to the residual stream. It does not need to be large: the
causal reading is EXACTLY zero, so any non-zero value at all separates the two.

``training=False`` is explicit everywhere and no layer here samples, so the model's own
call-to-call spread is exactly ``0.0``; :data:`SIGNAL_FLOOR` is asserted against that
measured spread inside the tests rather than assumed.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.language.hnet import components
from dl_techniques.models.language.hnet.config import (
    AttnSpec,
    HNetArchConfig,
    SSMSpec,
)
from dl_techniques.models.language.hnet.model import HNet

# ---------------------------------------------------------------------
# Fixtures -- every dimension is tiny; this is a correctness file and the
# Mamba-2 scan is sequential in L.
# ---------------------------------------------------------------------

SEED = 11
RESIDUAL_SEED = 7
DATA_SEED = 0

BATCH = 2
LENGTH = 12
D_MODEL = 16
MAX_CHUNKS = (6,)

#: Where the perturbed byte goes for arms 1 and 2. Strictly inside the sequence so the
#: "before" window is non-empty and the "at or after" window holds more than one position.
PERTURB_AT = 5

#: A byte offset coprime with 256, so ``(b + OFFSET) % 256 != b`` for every byte value --
#: the perturbation can never be a no-op on any row.
BYTE_OFFSET = 137

#: Arm 2's bar. The model is deterministic (asserted, not assumed, by
#: :meth:`TestTheProbeIsLive::test_the_model_repeats_itself_bit_exactly`), so the honest
#: floor is 0.0 and this is a 200x margin over it -- the smallest measured signal is
#: 2.35e-01. It exists so "something changed" cannot be satisfied by float32 dust.
SIGNAL_FLOOR = 1e-3

LAYOUTS = {
    "m_only": ["m1", ["m1"], "m1"],
    "T_only": ["T1", ["T1"], "T1"],
    "mixed": ["m1", ["T1"], "m1"],
}

#: The layouts that build attention at all. ``m_only`` does not, which is why the
#: keep-predicate control is parametrized over the expectation rather than the layout.
LAYOUT_BUILDS_ATTENTION = {"m_only": False, "T_only": True, "mixed": True}


def _config(layout):
    """A two-stage, one-chunking-level architecture at the given layout."""
    return HNetArchConfig(
        arch_layout=layout,
        d_model=[D_MODEL, D_MODEL],
        d_intermediate=[0, 0],
        ssm_cfg=SSMSpec(d_conv=4, expand=2, d_state=8),
        attn_cfg=AttnSpec(
            num_heads=(2, 2), rotary_emb_dim=(4, 4), window_size=(-1, -1)
        ),
    )


def _activate_residual(model, seed=RESIDUAL_SEED):
    """Give every ``residual_proj`` a seeded non-zero kernel, standing in for training.

    Returns the number of kernels assigned, which every caller asserts non-zero: a scope
    that matched nothing would silently restore the at-init dead probe this function
    exists to escape.
    """
    rng = np.random.default_rng(seed)
    n_assigned = 0
    for weight in model.weights:
        if "residual_proj" in weight.path and weight.path.endswith("kernel"):
            weight.assign(rng.standard_normal(weight.shape).astype("float32") * 0.3)
            n_assigned += 1
    return n_assigned


def _built(layout, activate=True):
    """A built model at ``layout``, with ``residual_proj`` activated unless asked not to."""
    keras.utils.set_random_seed(SEED)
    model = HNet(
        _config(LAYOUTS[layout]),
        max_chunks=MAX_CHUNKS,
        max_seq_len=64,
        headdim=8,
    )
    model.build((None, None))
    if activate:
        assert _activate_residual(model) > 0, (
            "no residual_proj kernel was assigned; the probe would be measuring an "
            "at-init model whose non-boundary positions ignore their own byte"
        )
    return model


def _bytes():
    return (
        np.random.default_rng(DATA_SEED)
        .integers(0, 256, (BATCH, LENGTH))
        .astype("int32")
    )


def _perturbed(x, position):
    """``x`` with the byte at ``position`` changed on EVERY row."""
    out = x.copy()
    out[:, position] = (out[:, position] + BYTE_OFFSET) % 256
    assert np.all(out[:, position] != x[:, position])
    return out


def _logits(model, x):
    return np.asarray(model(x, training=False))


def _delta(model, x, position, call=None):
    """``|logits(x) - logits(perturb(x, position))|``, elementwise, as ``(B, L, V)``."""
    call = _logits if call is None else call
    return np.abs(call(model, x) - call(model, _perturbed(x, position)))


def _reversed_call(model, x):
    """``flip(model(flip(x)))`` -- the SAME weights, wired ANTI-causally.

    If ``model`` is causal then this comparator's output at position ``i`` depends on the
    input bytes at ``i..L-1``, so a perturbation at ``t`` reaches every position ``<= t``.
    That is precisely the window arm 1 requires to be untouched, which makes this the
    negative control arm 1 owes: no re-initialisation, no transfer, not one weight
    different.
    """
    return _logits(model, x[:, ::-1])[:, ::-1]


def _all_ones_keep_mask(seq_len, window_size=-1, dtype="int32"):
    """A keep predicate that masks nothing -- the acausal stand-in for the real one."""
    return keras.ops.ones((1, seq_len, seq_len), dtype=dtype)


# ---------------------------------------------------------------------
# 0. The probe itself
# ---------------------------------------------------------------------


class TestTheProbeIsLive:
    """Everything that would make the three arms below unfalsifiable."""

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_at_init_the_last_byte_reaches_nothing_at_all(self, layout):
        """The trap, pinned: an at-init arm 3 passes on a model that read no byte.

        ``residual_proj`` is zero at initialisation, so a non-boundary position's own
        hidden state contributes to nothing and the output there is a function of the
        earlier chunks alone. MEASURED: perturbing the LAST byte moves the output by
        exactly 0.0 at every position -- including that position. Arm 3 must therefore run
        on a model whose residual branch is alive, and this test fails the moment
        ``_activate_residual`` stops being necessary.
        """
        model = _built(layout, activate=False)
        x = _bytes()

        moved = float(np.max(_delta(model, x, LENGTH - 1)))

        assert moved == 0.0, (
            "the at-init model now responds to its last byte, so the reason "
            "_activate_residual exists has changed; re-derive it before deleting it "
            f"(max|delta| = {moved:.6e})"
        )

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_the_activation_makes_the_last_byte_reach_its_own_position(self, layout):
        """The other half: with the residual alive the same perturbation IS visible."""
        model = _built(layout)
        x = _bytes()

        moved = float(np.max(_delta(model, x, LENGTH - 1)))

        assert moved > SIGNAL_FLOOR, f"max|delta| = {moved:.6e}"

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_the_model_repeats_itself_bit_exactly(self, layout):
        """Arm 1's ``atol=0`` is only meaningful if the forward is deterministic.

        Nothing here samples at ``training=False``, so the model's own call-to-call spread
        must be exactly 0.0. If it were not, every ``== 0.0`` below would be measuring the
        RNG and :data:`SIGNAL_FLOOR` would have to be recalibrated against that spread.
        """
        model = _built(layout)
        x = _bytes()

        spread = float(np.max(np.abs(_logits(model, x) - _logits(model, x))))

        assert spread == 0.0, f"the forward is not deterministic: {spread:.6e}"

    def test_the_perturbation_changes_every_row(self):
        """A perturbation that coincided with the original on some row would make arm 1
        vacuous on that row and would never be reported."""
        x = _bytes()
        for position in (0, PERTURB_AT, LENGTH - 1):
            assert np.all(_perturbed(x, position)[:, position] != x[:, position])


# ---------------------------------------------------------------------
# 1-3. The three arms
# ---------------------------------------------------------------------


class TestTheAssembledModelIsCausal:
    """The future-leak probe, on the assembled model, at three mixer layouts."""

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_arm_1_a_perturbed_byte_cannot_reach_an_earlier_position(self, layout):
        """Every logit before ``t`` is BIT-identical: ``atol=0``, ``rtol=0``.

        Exact rather than tolerant because a causal model's arithmetic at those positions
        never touches the perturbed byte. This is not a cancelling sum, so it is not
        reduction-order dependent.
        """
        model = _built(layout)
        x = _bytes()

        before = _delta(model, x, PERTURB_AT)[:, :PERTURB_AT]

        assert before.size > 0, "the 'before' window is empty; arm 1 would be vacuous"
        np.testing.assert_allclose(
            before, np.zeros_like(before), atol=0.0, rtol=0,
            err_msg=(
                f"{layout}: a byte at position {PERTURB_AT} reached at least one logit "
                f"at an EARLIER position (max|delta| = {float(np.max(before)):.6e}); the "
                "model attends to its own future"
            ),
        )

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_arm_2_the_perturbed_byte_does_reach_its_own_position_and_after(
            self, layout
    ):
        """The mandatory twin. Arm 1 alone passes on a model that ignores its input."""
        model = _built(layout)
        x = _bytes()

        after = _delta(model, x, PERTURB_AT)[:, PERTURB_AT:]

        signal = float(np.max(after))
        assert signal > SIGNAL_FLOOR, (
            f"{layout}: perturbing the byte at position {PERTURB_AT} moved NOTHING at or "
            f"after it (max|delta| = {signal:.6e}); arm 1's zero above is a dead model, "
            "not a causal one"
        )

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_arm_3_the_last_byte_cannot_reach_any_earlier_position(self, layout):
        """Arm 1 with the perturbation at the LAST position.

        The "before" window is now the whole sequence bar one, so a model that leaked
        anywhere leaks here; and the last position is required to move, so the arm cannot
        be satisfied by a model that ignores its input.
        """
        model = _built(layout)
        x = _bytes()
        last = LENGTH - 1

        delta = _delta(model, x, last)
        before = delta[:, :last]

        assert before.shape[1] == LENGTH - 1
        np.testing.assert_allclose(
            before, np.zeros_like(before), atol=0.0, rtol=0,
            err_msg=(
                f"{layout}: the LAST byte reached an earlier position "
                f"(max|delta| = {float(np.max(before)):.6e})"
            ),
        )
        assert float(np.max(delta[:, last])) > SIGNAL_FLOOR, (
            f"{layout}: the last byte moved nothing at its own position, so the zero "
            "above is a model ignoring its input"
        )


# ---------------------------------------------------------------------
# 4. Negative controls
# ---------------------------------------------------------------------


class TestTheProbeCanSeeALeak:
    """Two same-weights controls, because the zeros above must be falsifiable."""

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_the_time_reversed_comparator_leaks_into_the_same_window(self, layout):
        """``flip(model(flip(x)))`` is anti-causal, and the probe sees it.

        Same instance, same weights, no re-initialisation. If the leak window read 0.0
        for any reason other than causality -- a dead perturbation, an output slice taken
        on the wrong axis, a comparison of a tensor with itself -- it would read 0.0 here
        too.
        """
        model = _built(layout)
        x = _bytes()

        causal_leak = float(
            np.max(_delta(model, x, PERTURB_AT)[:, :PERTURB_AT])
        )
        reversed_leak = float(
            np.max(_delta(model, x, PERTURB_AT, call=_reversed_call)[:, :PERTURB_AT])
        )

        assert causal_leak == 0.0
        assert reversed_leak > SIGNAL_FLOOR, (
            f"{layout}: the ANTI-causal comparator did not leak either "
            f"({reversed_leak:.6e}); the probe cannot detect a leak at all and arm 1's "
            "zero means nothing"
        )

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_removing_the_keep_predicate_leaks_exactly_where_attention_exists(
            self, layout, monkeypatch
    ):
        """Patch ``build_causal_keep_mask`` to all-ones on the SAME instance, then undo it.

        ``T_only`` and ``mixed`` must start leaking; ``m_only`` must not move by so much
        as a bit, because that layout builds no attention at all -- its causality is the
        Mamba-2 scan and the chunk/dechunk order, which the time-reversed control above is
        what actually covers. Stating the ``m_only`` zero here keeps that division of
        labour executable instead of a comment.
        """
        model = _built(layout)
        x = _bytes()

        assert float(np.max(_delta(model, x, PERTURB_AT)[:, :PERTURB_AT])) == 0.0

        monkeypatch.setattr(
            components, "build_causal_keep_mask", _all_ones_keep_mask
        )
        patched_leak = float(np.max(_delta(model, x, PERTURB_AT)[:, :PERTURB_AT]))
        monkeypatch.undo()

        restored_leak = float(np.max(_delta(model, x, PERTURB_AT)[:, :PERTURB_AT]))

        if LAYOUT_BUILDS_ATTENTION[layout]:
            assert patched_leak > 0.0, (
                f"{layout}: removing the causal keep predicate changed nothing, so the "
                "predicate is not what is masking -- either it never reaches the "
                "attention layer, or something else is silently re-applying causality"
            )
        else:
            assert patched_leak == 0.0, (
                f"{layout} builds no attention, so an attention-mask patch must be "
                f"inert here; it moved the output by {patched_leak:.6e}, which means "
                "this layout is running attention it was never asked for"
            )

        assert restored_leak == 0.0, (
            f"{layout}: the leak did not return to exactly zero after the patch was "
            f"undone ({restored_leak:.6e}); the control left a side effect and the "
            "readings above are not attributable to the predicate"
        )

    def test_the_patch_target_is_the_name_the_stack_actually_calls(self):
        """A monkeypatch on a name nobody looks up is a control that cannot fail.

        ``HNetIsotropic.call`` resolves ``build_causal_keep_mask`` as a module global of
        ``components``, which is what makes the patch above effective. If the call site
        ever switches to a bound method or a local alias, the patch becomes silent and
        the ``T_only`` control would flip from a real measurement to a passing no-op --
        caught here, and by the ``patched_leak > 0.0`` assertion above.
        """
        import inspect

        source = inspect.getsource(components.HNetIsotropic.call)
        assert "build_causal_keep_mask(" in source
        assert components.build_causal_keep_mask.__module__ == components.__name__
