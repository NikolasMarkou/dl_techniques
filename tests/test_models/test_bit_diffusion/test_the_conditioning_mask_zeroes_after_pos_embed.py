"""``cond_mask`` zeroes the conditioning stream AFTER ``pos_embed`` is added.

Invariant 4 of the plan. A masked sample's conditioning tokens must be the
**exact zero tensor** -- not a small tensor, not a learned null token, and above
all not the positional table on its own.

**Why the order is the whole test.** Masking one line earlier, before
``pos_embed`` is added, is a one-token edit that:

* preserves every shape;
* preserves ``get_config()`` and the ``.keras`` round trip;
* leaves the output finite and plausible;
* and leaves the masked sample's conditioning stream carrying the FULL fixed
  positional signal, identical for every masked sample, which the cross-attention
  will happily attend to.

That is the branch classifier-free guidance takes on every single sampler step,
so it is a hot path wearing the costume of a corner case. The RED proof for this
file is exactly that edit, and the arms below are written so it fires on the
zero assertion rather than on some incidental difference: the anti-vacuity arm
first measures that ``pos_embed`` is itself far from zero, which is what makes
"exactly zero" a discriminating claim.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import DiTXA

from ._ditxa_helpers import activate, batch, np_


@pytest.fixture(scope="module")
def model() -> DiTXA:
    """A built, non-degenerate ``tiny`` model shared by every arm."""
    m = DiTXA.from_variant("tiny", class_dropout_rate=0.1, label_seed=7)
    m(batch(m, batch_size=2))
    return activate(m, seed=3)


class TestTheMaskProducesExactZeros:
    """The masked stream is the zero tensor at ``atol = 0``."""

    def test_an_all_false_mask_gives_exactly_zero_conditioning_tokens(self, model):
        """Every element, every sample: exactly ``0.0``.

        ``atol=0`` on purpose. A tolerance here would accept the
        mask-before-pos_embed defect at any tolerance above the positional
        table's own magnitude, which the next arm measures at O(1).
        """
        data = batch(model, batch_size=4, seed=11)
        cond = np_(
            model._embed_conditioning(
                data["x_cond"],
                data["direction"],
                cond_mask=np.zeros((4,), dtype="float32"),
            )
        )
        assert cond.shape == (4, model.num_patches, model.hidden_size)
        assert np.count_nonzero(cond) == 0, (
            "an all-false cond_mask must zero the conditioning stream EXACTLY. "
            f"{np.count_nonzero(cond)} of {cond.size} entries are nonzero, "
            f"max|cond| = {np.abs(cond).max():.10g}. The usual cause is masking "
            "BEFORE pos_embed is added, which leaves the positional table behind."
        )

    def test_a_per_sample_mask_zeroes_only_the_masked_rows(self, model):
        """A mixed mask must not leak across the batch axis."""
        data = batch(model, batch_size=4, seed=12)
        mask = np.array([1.0, 0.0, 1.0, 0.0], dtype="float32")
        cond = np_(
            model._embed_conditioning(
                data["x_cond"], data["direction"], cond_mask=mask
            )
        )
        assert np.count_nonzero(cond[1]) == 0
        assert np.count_nonzero(cond[3]) == 0
        assert np.count_nonzero(cond[0]) > 0
        assert np.count_nonzero(cond[2]) > 0

    def test_the_positional_table_is_large_enough_for_the_claim_to_bite(self, model):
        """Anti-vacuity: ``pos_embed`` is O(1), so a residue would be visible.

        Without this arm "exactly zero" could be true because everything in
        sight is already ~zero. It is not: the fixed sin-cos table has entries
        of order one, and its per-row norm is the residue the defect would
        leave.
        """
        table = np_(model.pos_embed)
        assert np.abs(table).max() > 0.5, (
            "the fixed positional table is nearly zero, so the exact-zero "
            "assertion above would pass under the mask-before-pos_embed defect "
            f"as well. max|pos_embed| = {np.abs(table).max():.10g}"
        )
        row_norm = float(np.linalg.norm(table[0, 0]))
        assert row_norm > 1.0, row_norm


class TestNoneIsAllOnes:
    """``cond_mask=None`` and an explicit all-ones mask must be indistinguishable."""

    def test_none_is_bit_identical_to_an_explicit_ones_mask(self, model):
        """``atol = 0``: this is a default, not an approximation."""
        data = batch(model, batch_size=4, seed=13)
        without = np_(
            model._embed_conditioning(data["x_cond"], data["direction"])
        )
        with_ones = np_(
            model._embed_conditioning(
                data["x_cond"],
                data["direction"],
                cond_mask=np.ones((4,), dtype="float32"),
            )
        )
        assert np.array_equal(without, with_ones), (
            "cond_mask=None must be numerically identical to an all-ones mask; "
            f"max|delta| = {np.abs(without - with_ones).max():.10g}"
        )

    def test_the_full_model_agrees_on_the_two_spellings(self, model):
        """The same claim one level up, through ``call()``."""
        data = batch(model, batch_size=4, seed=14)
        without = np_(model(data, training=False))
        with_ones = np_(
            model({**data, "cond_mask": np.ones((4,), "float32")}, training=False)
        )
        assert np.array_equal(without, with_ones)


class TestTheMaskedModelIgnoresItsConditioning:
    """With every conditioning token zeroed, ``x_cond`` cannot reach the output."""

    def test_the_output_is_independent_of_x_cond_under_an_all_false_mask(
        self, model
    ):
        """The unconditional CFG branch, asserted as a behaviour.

        Two completely different ``x_cond`` tensors, one all-false mask, one
        output. If the mask leaked, the two outputs would differ.
        """
        data = batch(model, batch_size=4, seed=15)
        zeros = np.zeros((4,), dtype="float32")
        first = np_(model({**data, "cond_mask": zeros}, training=False))
        other = batch(model, batch_size=4, seed=99)
        second = np_(
            model(
                {**data, "x_cond": other["x_cond"], "cond_mask": zeros},
                training=False,
            )
        )
        assert np.array_equal(first, second), (
            "under an all-false cond_mask the output must not depend on x_cond; "
            f"max|delta| = {np.abs(first - second).max():.10g}"
        )
        assert np.isfinite(first).all()

    def test_the_model_is_not_degenerate(self, model):
        """Anti-vacuity for the arm above.

        A model whose output is the zero tensor -- which is EXACTLY what a
        freshly initialised DiTXA produces, by design -- satisfies "the two
        outputs are equal" without carrying any information at all. This arm
        proves the fixture is live, and that the conditioning it is being denied
        would otherwise have moved the answer.
        """
        data = batch(model, batch_size=4, seed=16)
        conditioned = np_(model(data, training=False))
        assert np.abs(conditioned).max() > 1e-6, (
            "the fixture model outputs ~zero, so every equality arm in this "
            "file is vacuous. `activate()` did not do its job."
        )
        masked = np_(
            model({**data, "cond_mask": np.zeros((4,), "float32")}, training=False)
        )
        assert np.abs(conditioned - masked).max() > 1e-6, (
            "masking the conditioning changed nothing, so the model never used "
            "it and the independence arm proves nothing"
        )
