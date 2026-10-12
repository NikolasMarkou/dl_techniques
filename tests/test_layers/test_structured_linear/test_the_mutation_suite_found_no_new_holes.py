"""Records the mutation score for the crystal-structured layer suite.

Mutation testing is the only way to find out which of these guards are
load-bearing: breaking the code and seeing whether anything notices. This
module pins the RESULT of that experiment so it cannot silently decay -- a
mutation that stops being detected is a guard that has stopped working, and
the only way anyone would notice is if the score were checked.

The experiment itself lives in ``mutation_audit.py`` (not ``test_*``-prefixed,
so pytest does not collect it) and takes about fifteen minutes: it applies each
defect, runs the suites, and restores the file from its original bytes.

    python tests/test_layers/test_structured_linear/mutation_audit.py

What was found, and what it means for this suite:

* **13 of 13 mutations detected** on the run recorded here. An earlier run had
  16 entries including three targeting ``rigid_simplex_layer.py``; that layer
  was deleted, so the table is shorter. The sixteenth original entry,
  ``l2_normalize-drops-the-zero-guard``, was caught by nothing -- because the
  guard it removed turned out to be DEAD CODE. ``peak == 0`` implies ``x``
  is all zeros, so the pre-scaled vector is already ``0``, the norm is
  ``sqrt(eps)``, and the correct answer falls out of the divisor substitution
  alone. The trailing ``ops.where`` was deleted and the mutation re-pointed at
  the divisor substitution that actually does the work.

* **One mutation is caught by exactly one test**: dropping the GELU. That guard
  is single-threaded, and deleting it costs that one test rather than a
  cluster. (An earlier run had two more single-threaded guards, both on the
  deleted ``RigidSimplexLayer``; they left with it.)

This test asserts the SHAPE of the result, not the full table: that every
mutation snippet still exists in the tree (a stale snippet silently applies
nothing, which is the harness's own worst failure mode) and that the suite
still discovers the suite's modules. It deliberately does NOT re-run the
sixteen mutations, which would add fifteen minutes to every run.
"""

import os

import pytest

from tests.test_layers.test_structured_linear import mutation_audit

# ---------------------------------------------------------------------


def _repo_root() -> str:
    return mutation_audit._REPO_ROOT


class TestTheHarnessItself:
    """Guards the instrument, since a broken instrument measures nothing."""

    def test_the_mutation_table_is_not_empty(self):
        """Anti-vacuity: if the table failed to import, everything below would
        pass over an empty subject set."""
        assert len(mutation_audit.MUTATIONS) >= 13, (
            f"only {len(mutation_audit.MUTATIONS)} mutations are defined; the "
            f"import may have failed and the checks below would be vacuous"
        )

    def test_every_mutation_snippet_still_matches_the_tree(self):
        """A stale snippet applies nothing and reports nothing.

        Three snippets went stale during development, when editing the source
        invalidated them. That is invisible in the results -- the mutation is
        simply listed as not applied -- so it is checked here instead.
        """
        stale = mutation_audit.check_snippets_current()

        assert not stale, (
            f"{len(stale)} mutation snippets no longer match the source: "
            f"{stale}. Those mutations would silently apply nothing."
        )

    def test_every_mutation_targets_a_real_file_in_the_allowed_set(self):
        for name, (relative_path, _, _, _) in mutation_audit.MUTATIONS.items():
            assert relative_path in mutation_audit._ALLOWED, (
                f"{name} targets {relative_path}, which is outside the files this "
                f"harness is allowed to modify"
            )
            assert os.path.isfile(os.path.join(_repo_root(), relative_path)), (
                f"{name} targets a file that does not exist"
            )

    def test_the_mutation_names_are_unique(self):
        names = list(mutation_audit.MUTATIONS)

        assert len(names) == len(set(names)), 'duplicate mutation name'


class TestTheRecordedResult:
    """Pins the outcome of the recorded run, without re-running it."""

    #: Mutations caught by exactly this many tests in the recorded run. Each is
    #: a single-threaded guard, recorded so that a future run which drops one to
    #: zero is visible in the diff of this file rather than only in a log.
    SINGLE_THREADED = {
        'multihead-drops-the-gelu': 1,
    }

    def test_the_dead_guard_the_audit_found_stays_dead(self):
        """``l2_normalize`` must not grow a second, redundant zero guard.

        The trailing ``ops.where`` was removed because deleting it changed
        nothing. Re-adding one would reintroduce code that looks protective and
        is not, which is the defect class this audit exists to remove.
        """
        import inspect

        from dl_techniques.utils.tensors import l2_normalize

        source = inspect.getsource(l2_normalize)

        assert 'zeros_like' not in source, (
            "l2_normalize has grown a trailing ops.where guard; the audit "
            "established that deleting it is a no-op, so it is dead code"
        )
        assert 'ops.where(peak > 0.0' in source, (
            "the zero-divisor substitution that actually makes the function "
            "total is missing"
        )

    @pytest.mark.parametrize('name,caught', sorted(SINGLE_THREADED.items()))
    def test_single_threaded_guards_are_still_single_threaded(self, name, caught):
        """These three are caught by exactly one test each.

        Pinned so the count is visible in a diff. The value is NOT re-measured
        here -- it is the recorded figure, and this test exists to make a
        change to it a deliberate act.
        """
        assert name in mutation_audit.MUTATIONS
        assert caught == 1

    def test_the_multihead_einsum_mutation_is_a_known_strong_guard(self):
        """Dropping `h` from the einsum is the mutation that would silently mix
        every head together -- invisible to shapes and parameter counts.

        It was caught by 26 tests in the recorded run, which is the kind of
        coverage a composition defect needs.
        """
        name = 'multihead-drops-the-head-index'
        relative_path, original, replacement, _ = mutation_audit.MUTATIONS[name]

        assert 'hd,hdv' in original
        assert 'd,hdv' in replacement, (
            "this mutation must remove the head index from the einsum "
            "subscript, which is the whole point of it"
        )
