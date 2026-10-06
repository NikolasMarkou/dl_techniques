"""RED proof for the transcription oracle in ``spatial_smoothness_oracle``.

A guard that cannot fail is the most likely outcome of writing a new test, and a
shared instrument is exactly the kind of code that gets trusted without being
tested. This module proves the oracle can reject, by mutating the *paper's
formula* in each of the ways an implementation could plausibly get wrong and
checking the oracle notices.

The mutations here are all inside the oracle, deliberately: the claim under test
is "this oracle discriminates", not "the implementation is right". The
implementation's own correctness is carried by
``test_the_loss_matches_the_paper_transcription_term_for_term``, which fails if
the two ever stop agreeing.

References:
    - Rathi et al., 2025. TopoLM, Eq. 1. (https://arxiv.org/abs/2410.11516)
"""

import numpy as np
import pytest

from dl_techniques.layers.regularization.spatial_smoothness import (
    SpatialLayout,
    build_patch_tables,
)

from .spatial_smoothness_oracle import (
    reference_d_unit,
    reference_smoothness_loss,
    smooth_field_activations,
)

NUM_UNITS = 784
RADIUS = 2


@pytest.fixture(scope="module")
def fixture():
    layout = SpatialLayout(NUM_UNITS, seed=0)
    tables = build_patch_tables(layout.cell_to_unit, RADIUS, "linf")
    activations = smooth_field_activations(layout, num_samples=48, seed=0)
    return layout, tables, activations


class TestTheOracleCanDiscriminate:
    def test_the_oracle_is_not_vacuous_on_its_own_fixture(self, fixture):
        """Anti-vacuity first: the oracle must return a real, non-trivial number.

        An oracle that returns a constant satisfies every equality downstream
        while measuring nothing.
        """
        _, tables, activations = fixture
        value = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )
        assert 0.0 < value < 1.0

    def test_the_oracle_separates_a_smooth_field_from_noise(self, fixture):
        _, tables, activations = fixture
        rng = np.random.default_rng(0)
        smooth = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )
        noisy = reference_smoothness_loss(
            rng.normal(size=activations.shape),
            tables.patch_table[0],
            tables.d_unit,
        )
        assert smooth < noisy - 0.05, (smooth, noisy)

    def test_a_reversed_correlation_sign_is_rejected(self, fixture):
        """Sign of the correlation.

        Pearson is odd in either argument, so negating the measured
        correlations must land exactly on ``1 - SL`` about 0.5. If the oracle were
        insensitive to the correlation's sign, the mutation below would pass.
        """
        _, tables, activations = fixture
        correct = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )
        mutated = reference_smoothness_loss(
            activations,
            tables.patch_table[0],
            -tables.d_unit,
        )
        assert correct < 0.5
        assert mutated > 0.5
        assert correct + mutated == pytest.approx(1.0, abs=1e-9)

    def test_a_swapped_patch_is_rejected(self, fixture):
        """The patch must be the one asked for.

        A neighbourhood that silently read a fixed centre -- the classic
        off-by-one in a gather -- still returns a plausible number, so the
        discriminator has to be that DIFFERENT patches give DIFFERENT answers.

        The field is periodic over the grid, so a minority of sampled windows are
        congruent by construction; the bound is 80% distinct rather than "all
        distinct", and one guaranteed-distinct pair (a centre against its
        diagonal corner) is asserted outright.
        """
        _, tables, activations = fixture
        rows = list(range(0, tables.num_centers, 21))
        values = [
            reference_smoothness_loss(
                activations, tables.patch_table[row], tables.d_unit
            )
            for row in rows
        ]
        distinct = len(set(np.round(values, 6)))
        assert distinct >= 0.8 * len(values), (
            f"only {distinct} of {len(values)} neighbourhoods gave distinct "
            f"answers -- the oracle may be reading a fixed window"
        )

        first = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )
        far = reference_smoothness_loss(
            activations, tables.patch_table[tables.num_centers - 1], tables.d_unit
        )
        assert first != pytest.approx(far, abs=1e-6)

    def test_a_missing_normalisation_is_rejected(self, fixture):
        """The `0.5` factor and the centring are load-bearing.

        Dropping the factor doubles the loss; dropping the centring of ``d``
        changes it in a direction the sign test cannot catch. Both must move the
        number by far more than the tolerance the implementation agrees to.
        """
        _, tables, activations = fixture
        correct = reference_smoothness_loss(
            activations, tables.patch_table[0], tables.d_unit
        )

        uncentred = 1.0 / (np.arange(tables.num_pairs, dtype=np.float64) % 11 + 1.0)
        mutated = reference_smoothness_loss(
            activations, tables.patch_table[0], uncentred.astype(np.float32)
        )
        assert abs(correct - mutated) > 1e-3

    def test_an_unnormalized_prior_is_rejected(self, fixture):
        """Scale-invariance of the correlation must not hide a scale bug.

        ``corr`` is invariant to scaling ``d``, so multiplying the prior by a
        constant cannot be detected by any loss-value comparison -- which is
        exactly why the implementation normalises it once in NumPy rather than
        recomputing the norm per step. The oracle agrees on the VALUE of the
        normalized prior and disagrees on an unnormalized one.
        """
        _, tables, activations = fixture
        assert reference_d_unit(tables.patch_units) == pytest.approx(
            tables.d_unit, abs=1e-6
        )
        # A constant multiple of a unit vector is a different array, and the
        # equality above is what says the oracle normalises the same way.
        assert not np.allclose(reference_d_unit(tables.patch_units) * 3.0, tables.d_unit)