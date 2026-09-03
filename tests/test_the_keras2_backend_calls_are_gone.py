"""Repo-wide census: no file under ``src/`` calls anything on ``keras.backend``.

.. warning::

   **This guard is pinned** ``xfail(strict=True)`` **until step 7 of**
   ``plans/plan-2026-09-03T033750-9bdf25f4``. At the plan's base commit the walk
   reports **83 offenders across 67 files**; steps 3-6 convert them and step 7
   removes the mark. The mark is ``strict``, so the day the tree becomes clean
   this file turns the suite RED with an ``XPASS`` -- a pin that outlives the
   defect it excuses is itself a defect, and this one cannot be forgotten.

What is banned, and why all six spellings at once
-------------------------------------------------
The predicate is a plain substring match on the literal text ``keras.backend.``.
It is therefore not a check about one symbol: it bans every attribute reached
through that module. The six spellings live in this tree today, with the
Keras 3 replacement each one takes::

    keras.backend.standardize_dtype(d)   ->  getattr(d, "name", None) or str(d)
    keras.backend.epsilon()              ->  keras.config.epsilon()
    keras.backend.floatx()               ->  keras.config.floatx()
    keras.backend.image_data_format()    ->  keras.config.image_data_format()
    keras.backend.clear_session()        ->  keras.utils.clear_session()
    keras.backend.result_type(d, "f32")  ->  dl_techniques.utils.dtype_policy
                                             .statistics_dtype(d)

``result_type`` is the one with no drop-in: ``keras.config`` does not expose it
and ``keras.ops.result_type`` raises ``AttributeError``. It performs Keras's own
two-argument dtype PROMOTION, which the single-dtype ``standardize_dtype``
replacement above cannot express -- reaching for that idiom at a ``result_type``
site is a wrong answer, not a near miss.

The motivating defect (carried over from the guard this file replaces)
----------------------------------------------------------------------
``keras.backend.GradientTape`` does not exist in Keras 3 at all: it sat in
``latent_gmm_registration.train_step`` and made the model **untrainable** while
its suite stayed green, because every test in it was forward-pass only. That is
the class of defect this ban exists for -- a Keras-2 attribute that either does
not exist, or exists with different semantics, on a path no test drives.

Why the scope is the whole tree and no longer ``models/``
----------------------------------------------------------
This guard previously lived as
``tests/test_models/test_package_api_contract.py::TestNoKeras2Residues`` and
walked ``MODELS_DIR`` only. That scope was never a decision about this rule: it
was inherited from the one constant every other check in that file is built
against, whose charter is "every subpackage of ``dl_techniques.models``". The
motivating defect happening to live inside ``models/`` is a coincidence of where
the bug was, not evidence the scope was chosen.

The cost was measured: the ``models/`` walk reports **zero** offenders while the
rest of ``src/`` carries 83. A scoped gate cannot see an outside consumer, and
the previous plan's own ``losses/ddpm_hybrid_loss.py`` sat non-compliant for a
whole step with nothing able to catch it. The walk is now ``src/dl_techniques/``
(``models/`` included -- the old scope is a subset, not a sibling), ``src/train/``
and ``src/applications/``.

``TestNoKeras2Residues`` is DELETED rather than kept alongside this file. Two
guards over one claim is a hand-maintained lockstep invariant, which this repo
treats as a latent defect and not a belt-and-braces.

``tf.keras.backend.*`` is deliberately IN scope
------------------------------------------------
``src/dl_techniques/utils/random.py`` spells it ``tf.keras.backend.epsilon()``.
The substring predicate fires on those lines because ``keras.backend.`` is a
substring of ``tf.keras.backend.``. That is not an accident this guard tolerates
-- it is the intended reading. The ``tf.keras`` shim is the *same* Keras-2
surface reached through TensorFlow, so those sites are Keras-2 residues by the
same argument as the rest, and their fix is ``keras.config.epsilon()``, never
``tf.keras.config.epsilon()``.

``tests/`` is deliberately OUT of scope
-----------------------------------------
94 files under ``tests/`` use ``keras.backend.*`` on purpose, as oracles that
measure Keras's own behaviour as ground truth -- including the permanent
equivalence arm for ``statistics_dtype``, which must keep comparing against
``keras.backend.result_type`` for as long as that symbol exists. Banning the
spelling in the tests would delete the evidence that the replacements are right.

The predicate is imported, never re-implemented
------------------------------------------------
``_docstring_line_numbers`` and the scan loop below come from the guard this
file replaces, verbatim. Grep over this repo finds docstrings: a hand-rolled
second walk would count every module docstring that *warns* about
``keras.backend.`` -- starting with this one -- as a defect. The exclusion is
load-bearing, and step 7 proves it with an injection inside a docstring that
must NOT fire.
"""

from pathlib import Path
from typing import List

import pytest

# One copy of the predicate, not two. The import is absolute (``tests.`` is a
# package): the bare ``test_models.…`` spelling does NOT resolve under this
# repo's pytest import mode.
from tests.test_models.test_package_api_contract import _docstring_line_numbers

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

#: Every root under ``src/``. Not ``SRC_ROOT.iterdir()``: a new top-level
#: package should be a deliberate addition here, visible in a diff.
SCANNED_ROOTS = (
    SRC_ROOT / "dl_techniques",
    SRC_ROOT / "train",
    SRC_ROOT / "applications",
)

#: The census this file was landed with, at plan-2026-09-03T033750-9bdf25f4's
#: base commit: 83 sites in 67 files. History, not an expectation -- the
#: assertion below is ``== 0``, and the file is not marked ``skip``.
#: The full path:line:symbol inventory is
#: ``plans/plan-2026-09-03T033750-9bdf25f4/findings/keras-backend-inventory.md``.
MEASURED_OFFENDERS_AT_PIN = 83


def keras_backend_offenders(root: Path) -> List[str]:
    """``root``-relative ``line:source`` for every non-prose ``keras.backend.`` use.

    The loop is the one from ``TestNoKeras2Residues::test_no_keras_backend_calls``,
    unchanged: full-line ``#`` comments and any line inside a module/class/function
    docstring are excluded; everything else that contains the literal text is an
    offender.
    """
    offenders: List[str] = []
    for path in sorted(root.rglob("*.py")):
        docstring_lines = _docstring_line_numbers(path)
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Prose is not code. Before this exclusion, a module docstring
            # explaining *why* `keras.backend.` must not be used failed the
            # suite -- the guard could not be documented in the tree it guards.
            if i in docstring_lines:
                continue
            if "keras.backend." in line:
                offenders.append(f"{path.relative_to(root)}:{i} {stripped}")
    return offenders


class TestTheKeras2BackendCallsAreGone:
    """The ban itself, over the whole of ``src/``."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "83 measured offenders across 67 files at the plan's base commit; "
            "steps 3-6 of plan-2026-09-03T033750-9bdf25f4 fix them and step 7 "
            "removes this mark. strict=True so an XPASS reddens the suite."
        ),
    )
    def test_no_keras_backend_calls(self):
        offenders = []
        for root in SCANNED_ROOTS:
            offenders += [
                f"{root.name}/{entry}" for entry in keras_backend_offenders(root)
            ]
        assert not offenders, (
            "use keras.config.{epsilon,floatx,image_data_format}(), "
            "keras.utils.clear_session(), dtype_policy.statistics_dtype() and "
            "tf.GradientTape; keras.backend.* found at "
            f"{len(offenders)} sites: {offenders}"
        )


class TestTheCensusInstrumentIsNotVacuous:
    """A walk that visits nothing reports zero offenders and proves nothing.

    Not ``xfail``: these arms must be GREEN from the day the file lands, and they
    stay green after step 7 unpins the ban. They are what distinguishes "the tree
    is clean" from "the walk is empty".
    """

    def test_each_scanned_root_exists_and_holds_python_files(self):
        for root in SCANNED_ROOTS:
            assert root.is_dir(), f"scanned root vanished: {root}"
            assert list(root.rglob("*.py")), f"scanned root has no .py files: {root}"

    def test_a_synthetic_offender_is_detected(self, tmp_path):
        (tmp_path / "offender.py").write_text(
            "import keras\n\n\ndef f():\n    return keras.backend.epsilon()\n"
        )
        offenders = keras_backend_offenders(tmp_path)
        assert offenders == ["offender.py:5 return keras.backend.epsilon()"], offenders

    def test_prose_and_comments_are_not_offenders(self, tmp_path):
        """The exclusion is load-bearing: this very file would fail without it."""
        (tmp_path / "prose.py").write_text(
            '"""Never call keras.backend.epsilon() -- use keras.config.epsilon()."""\n'
            "\n"
            "# keras.backend.floatx() is banned too.\n"
            "import keras\n"
        )
        assert keras_backend_offenders(tmp_path) == []
