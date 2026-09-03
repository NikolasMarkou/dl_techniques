"""Repo-wide census: no file under ``src/`` calls anything on ``keras.backend``.

The walk reports **zero** offenders and the assertion below is unconditional:
there is no ``xfail`` mark, no waiver list and no allow-list. It was landed
pinned ``xfail(strict=True)`` against a measured **83 offenders across 67
files**, and unpinned in the same plan once the last one was converted; that
history lives in ``plans/plan-2026-09-03T033750-9bdf25f4`` and its
``findings/keras-backend-inventory.md``, not in this file's behaviour.

What is banned, exactly
------------------------
Two static checks run per line, after full-line ``#`` comments and docstring
lines have been dropped:

1. the literal text ``keras.backend.`` appears anywhere on the line -- so the
   ban is not about one symbol, it covers **every attribute reached by that
   spelling**, and it fires on ``tf.keras.backend.`` too (see below);
2. the line is an ``import`` statement that binds ``keras.backend`` under any
   name -- ``import keras.backend as kb``, ``from keras import backend as K``,
   ``from keras.backend import epsilon``, and the ``tensorflow.keras``-prefixed
   forms of all three. These are the canonical Keras-2 import idioms; check 1
   is blind to every one of them, because none of them contains a ``.`` after
   ``backend``.

Six call spellings were present in this tree before the migration, each with the
Keras 3 replacement it took::

    keras.backend.standardize_dtype(d)   ->  getattr(d, "name", None) or str(d)
    keras.backend.epsilon()              ->  keras.config.epsilon()
    keras.backend.floatx()               ->  keras.config.floatx()
    keras.backend.image_data_format()    ->  keras.config.image_data_format()
    keras.backend.clear_session()        ->  keras.utils.clear_session()
    keras.backend.result_type(d, "f32")  ->  dl_techniques.utils.dtype_policy
                                             .statistics_dtype(d)

What is NOT banned -- the named residue
----------------------------------------
This is a per-line text scan, not an import-graph or a dataflow analysis. It is
defeated by, and makes no claim about:

* a call through an **alias already bound** -- ``K.epsilon()`` on a later line.
  Check 2 catches the ``import`` that binds ``K``, so the alias cannot be
  introduced without tripping the guard, but a ``K.`` call whose binding came
  from somewhere this walk does not read is invisible;
* dynamic reach: ``getattr(keras, "backend").epsilon()``,
  ``exec("keras." + "backend.epsilon()")``, ``importlib.import_module``;
* a call split across lines, by ``\``-continuation or inside parentheses, such
  that no single line carries the literal text;
* ``keras.src.backend.*``. That is Keras 3's own private module, not the Keras-2
  compatibility surface this ban is about; reaching into it is a different
  (also bad) idea and this guard deliberately says nothing about it.

None of those spellings exists anywhere in ``src/`` today -- measured, not
assumed. An honest narrow guard is worth more than a broad claim: if you widen
the predicate, widen this list too.

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

``TestNoKeras2Residues`` is DELETED rather than kept alongside this file. So is
the hand-rolled second copy of the scan that lived in
``tests/test_models/test_bit_diffusion/test_the_package_keeps_the_repo_contracts.py``
-- this walk strictly subsumes it (that package's 8 modules are a subset of these
1,070 files, and an injected offender is reported by both at the same
``file:line``). Two guards over one claim is a hand-maintained lockstep
invariant, which this repo treats as a latent defect and not a belt-and-braces.

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
``keras.backend.`` -- starting with this one -- as a defect. Full-line ``#``
comments are excluded by the same argument. Both exclusions are load-bearing and
both were proven so, by injections placed in a docstring and on a comment line
that must NOT fire, alongside one real-code injection per banned spelling that
must, and one per import form. If you change the predicate, re-run all eleven
(six call spellings + three import forms, and the two that must NOT fire): a
guard that only ever passes is indistinguishable from a guard that cannot fail.
"""

import re
from pathlib import Path
from typing import List

# One copy of the predicate, not two. The import is absolute (``tests.`` is a
# package): the bare ``test_models.…`` spelling does NOT resolve under this
# repo's pytest import mode.
from tests.test_models.test_package_api_contract import _docstring_line_numbers

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

#: An ``import`` statement that binds ``keras.backend`` under any name. Check 1
#: (the ``keras.backend.`` substring) cannot see any of these: there is no ``.``
#: after ``backend`` in an import line. The optional ``[\w.]+\.`` prefix picks up
#: the ``tensorflow.keras`` spellings, which are the same Keras-2 surface. The
#: trailing ``\b`` keeps it off ``from keras import backend_config``.
_IMPORT_FORM = re.compile(
    r"(?:^|\s)import\s+(?:[\w.]+\.)?keras\.backend\b"
    r"|(?:^|\s)from\s+(?:[\w.]+\.)?keras\s+import\s+backend\b"
    r"|(?:^|\s)from\s+(?:[\w.]+\.)?keras\.backend\s+import\b"
)

#: Every root under ``src/``. Not ``SRC_ROOT.iterdir()``: a new top-level
#: package should be a deliberate addition here, visible in a diff.
SCANNED_ROOTS = (
    SRC_ROOT / "dl_techniques",
    SRC_ROOT / "train",
    SRC_ROOT / "applications",
)


def keras_backend_offenders(root: Path) -> List[str]:
    """``root``-relative ``line:source`` for every non-prose ``keras.backend`` use.

    The exclusions are the ones from
    ``TestNoKeras2Residues::test_no_keras_backend_calls``, unchanged: full-line
    ``#`` comments and any line inside a module/class/function docstring are
    skipped. What counts as an offender is that file's substring check PLUS
    ``_IMPORT_FORM``, which was added after a review measured the three
    canonical Keras-2 import idioms evading the substring alone. See this
    module's docstring for what is still not caught.
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
            if "keras.backend." in line or _IMPORT_FORM.search(line):
                offenders.append(f"{path.relative_to(root)}:{i} {stripped}")
    return offenders


class TestTheKeras2BackendCallsAreGone:
    """The ban itself, over the whole of ``src/``. Unconditional: the count is 0."""

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

These arms were GREEN from the day this file landed, through the whole
    migration, and after the ban was unpinned. They are what distinguishes "the
    tree is clean" from "the walk is empty".
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

    def test_each_keras2_import_form_is_detected(self, tmp_path):
        """The three canonical Keras-2 import idioms, none of which check 1 sees.

        Measured before ``_IMPORT_FORM`` existed: all three were MISSED, because
        an import line carries no ``.`` after ``backend``. ``from keras import
        backend as K`` is *the* historical spelling this ban exists to retire.
        """
        forms = [
            "from keras import backend as K",
            "from keras import backend",
            "import keras.backend as kb",
            "import keras.backend",
            "from keras.backend import epsilon",
            "from tensorflow.keras import backend as K",
            "import tensorflow.keras.backend as kb",
            "from tensorflow.keras.backend import epsilon",
        ]
        for form in forms:
            (tmp_path / "offender.py").write_text(f"{form}\n")
            assert keras_backend_offenders(tmp_path) == [
                f"offender.py:1 {form}"
            ], form

    def test_lookalike_imports_are_not_offenders(self, tmp_path):
        """The widened check must not fire on ordinary Keras 3 or local imports.

        Anti-vacuity in the other direction: a predicate that flags every line
        containing the word ``backend`` would make the census unfalsifiable.
        """
        (tmp_path / "clean.py").write_text(
            "import keras\n"
            "from keras import ops\n"
            "from keras import backend_config\n"
            "from dl_techniques.utils import backend_helpers\n"
            "import numpy as np  # backend, keras, whatever\n"
        )
        assert keras_backend_offenders(tmp_path) == []

    def test_prose_and_comments_are_not_offenders(self, tmp_path):
        """The exclusion is load-bearing: this very file would fail without it."""
        (tmp_path / "prose.py").write_text(
            '"""Never call keras.backend.epsilon() -- use keras.config.epsilon()."""\n'
            "\n"
            "# keras.backend.floatx() is banned too.\n"
            "import keras\n"
        )
        assert keras_backend_offenders(tmp_path) == []
