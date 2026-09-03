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
2. the line is a **single-line** ``import`` statement that binds
   ``keras.backend`` -- ``import keras.backend as kb``, ``from keras import
   backend as K``, ``from keras.backend import epsilon``, and the
   ``tensorflow.keras``-prefixed forms of all three. These are the canonical
   Keras-2 import idioms; check 1 is blind to every one of them, because none of
   them contains a ``.`` after ``backend``. The import's name list is SPLIT, not
   pattern-matched, so the position of ``backend`` in it does not matter (``from
   keras import ops, backend`` and ``import numpy, keras.backend`` both fire) and
   neither does the single-line bracketed form ``from keras import (ops,
   backend)``. Only the first token of each comma-separated item counts as a
   bound name, which is why ``from keras import ops as backend`` -- an alias
   that spells the banned word without importing it -- does NOT fire.

   What check 2 does NOT claim is that it catches every way to bind
   ``keras.backend``: it is one regex pair over one line, and the forms it
   cannot reach are enumerated below rather than left implied.

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
  ``exec("keras." + "backend.epsilon()")``, ``importlib.import_module``, and
  ``__import__("keras.backend", fromlist=["epsilon"])`` -- the last one is a
  *static* line, but ``__import__`` is a call, not an ``import`` statement, and
  the string it takes carries no trailing ``.`` for check 1 to see;
* a call split across lines, by ``\``-continuation or inside parentheses, such
  that no single line carries the literal text;
* an **import** split across lines the same way -- the parenthesised multi-line
  form ``from keras import (\n    backend,\n)`` and the ``\``-continued
  ``from keras import ops, \``. Check 2 reads one line at a time, so the line
  carrying ``backend`` is not the line carrying ``from keras import``. Catching
  these needs cross-line state or an AST walk; this scan deliberately stays
  per-line, and this is the price. The bracketed form on a SINGLE line is
  caught;
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
must, and one per import form. If you change the predicate, re-run every arm of
``TestTheCensusInstrumentIsNotVacuous`` AND re-inject each detected form into a
real ``src/`` module -- a leaf one, so a mid-run failure cannot leave the
library unimportable. Both directions matter: a guard that only ever passes is
indistinguishable from a guard that cannot fail, and a guard that fires on
``from keras import ops`` would make the census unfalsifiable the other way.
The residue list above is deliberately NOT pinned by a must-not-fire arm --
those forms are missed, not permitted, and a test freezing them would redden the
day someone widens the scan.
"""

import re
from pathlib import Path
from typing import List

# One copy of the predicate, not two. The import is absolute (``tests.`` is a
# package): the bare ``test_models.…`` spelling does NOT resolve under this
# repo's pytest import mode.
from tests.test_models.test_package_api_contract import _docstring_line_numbers

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

#: The two import statement shapes, split at the ``#`` so a trailing comment is
#: never read as part of the name list. The optional ``[\w.]+\.`` prefix in the
#: module patterns below picks up the ``tensorflow.keras`` spellings, which are
#: the same Keras-2 surface.
_PLAIN_IMPORT = re.compile(r"(?:^|[\s;])import\s+(?P<names>[^#;]*)")
_FROM_IMPORT = re.compile(
    r"(?:^|[\s;])from\s+(?P<module>[\w.]+)\s+import\s+(?P<names>[^#;]*)"
)

#: ``keras.backend`` itself, or reached through ``tensorflow.``. The trailing
#: ``\b`` keeps it off ``keras.backend_config``; anchoring at ``^`` keeps it off
#: ``keras.src.backend``, which this ban deliberately says nothing about.
_KERAS_BACKEND_MODULE = re.compile(r"^(?:[\w.]+\.)?keras\.backend\b")
#: The ``keras`` package exactly -- not ``keras.src``, not ``mykeras``.
_KERAS_MODULE = re.compile(r"^(?:[\w.]+\.)?keras$")


def _imported_names(names: str) -> List[str]:
    """The names an import list BINDS: ``(ops, backend as K,)`` -> ``[ops, backend]``.

    Parentheses become whitespace rather than being parsed, so the single-line
    bracketed form is handled by the same split. Only the first token of each
    comma-separated item is a bound module name, which is what keeps
    ``from keras import ops as backend`` -- an alias that happens to spell the
    banned word -- out of the result.
    """
    cleaned = names.replace("(", " ").replace(")", " ")
    return [item.split()[0] for item in cleaned.split(",") if item.split()]


def _import_binds_keras_backend(line: str) -> bool:
    """Check 2: does this ONE line bind ``keras.backend``, at ANY list position?

    # DECISION plan-2026-09-03T033750-9bdf25f4/D-011
    Do NOT re-narrow this to the single regex it replaced
    (``from\\s+(?:[\\w.]+\\.)?keras\\s+import\\s+backend\\b``). That form required
    ``backend`` to be the FIRST name in the list, so ``from keras import ops,
    backend`` and ``import numpy, keras.backend`` were both MISSED while the
    module docstring claimed the check fired "under any name" -- measured, and
    the reason this function exists. Splitting the import's name list is what
    makes position irrelevant; a regex that walks the list textually re-acquires
    the same off-by-one the moment someone adds an alternative. Equally: do NOT
    grow this into a multi-line/AST import parser -- the scan is per-line by
    construction and the forms that need more than one line are named residue in
    this module's docstring. See decisions.md D-011.
    """
    from_match = _FROM_IMPORT.search(line)
    if from_match is not None:
        module = from_match.group("module")
        if _KERAS_BACKEND_MODULE.match(module):
            return True  # from keras.backend import epsilon
        if _KERAS_MODULE.match(module):
            return "backend" in _imported_names(from_match.group("names"))
        return False
    plain_match = _PLAIN_IMPORT.search(line)
    if plain_match is None:
        return False
    return any(
        _KERAS_BACKEND_MODULE.match(name)
        for name in _imported_names(plain_match.group("names"))
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
    ``_import_binds_keras_backend``, which was added after a review measured the
    canonical Keras-2 import idioms evading the substring alone, and widened
    again after a second review measured ``backend`` at a non-first list
    position evading the first version of it. See this module's docstring for
    what is still not caught.
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
            if "keras.backend." in line or _import_binds_keras_backend(line):
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
        """Every single-line import spelling check 2 claims, none of which check 1 sees.

        Measured before check 2 existed: the first eight were all MISSED, because
        an import line carries no ``.`` after ``backend``. ``from keras import
        backend as K`` is *the* historical spelling this ban exists to retire.

        The rest were measured MISSED by the *first* version of check 2, which
        pattern-matched ``import\\s+backend`` and so required ``backend`` to be
        the first name in the list. They are here because the position of a name
        in an import list is not a property anyone would think to preserve when
        editing an import -- ``isort`` alone reorders it.
        """
        forms = [
            # the eight canonical single-name idioms
            "from keras import backend as K",
            "from keras import backend",
            "import keras.backend as kb",
            "import keras.backend",
            "from keras.backend import epsilon",
            "from tensorflow.keras import backend as K",
            "import tensorflow.keras.backend as kb",
            "from tensorflow.keras.backend import epsilon",
            # ... and every list POSITION, which the first widening missed
            "from keras import backend, ops",
            "from keras import ops, backend",
            "from keras import ops, backend as K",
            "from keras import ops, backend, layers",
            "from keras import backend as K, ops",
            "from keras import backend_config, backend",
            "from tensorflow.keras import ops, backend",
            "from keras import ops,backend",
            "from keras import (backend, ops)",
            "from keras import (ops, backend)",
            "from keras import (ops, backend,)",
            "import keras.backend, numpy",
            "import numpy, keras.backend",
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
        This arm grew with the check: splitting the import name list means the
        word ``backend`` can now appear ANYWHERE on an import line, so the last
        five below are what stops the widening from becoming that predicate.
        ``keras.src.backend`` is silent on purpose -- see the module docstring;
        it is Keras 3's own private module, not the Keras-2 surface.
        """
        (tmp_path / "clean.py").write_text(
            "import keras\n"
            "from keras import ops\n"
            "from keras import backend_config\n"
            "from dl_techniques.utils import backend_helpers\n"
            "import numpy as np  # backend, keras, whatever\n"
            "from keras import ops as backend\n"
            "from keras import ops  # not backend\n"
            "from keras.src import backend\n"
            "import keras.src.backend\n"
            "from mykeras import backend\n"
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
