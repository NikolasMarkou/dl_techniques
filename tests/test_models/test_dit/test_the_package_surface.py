"""The ``dit`` package's public surface, and the PRESENCE (never the truth) of its docs.

Three separate claims live here, and the difference between them matters:

1. **The export surface is real.** Every name in
   ``dl_techniques.models.vision_language.dit.__all__`` is importable from the
   package, is the same object the defining module holds, and is either
   constructible or a documented module constant. No exported name shadows a
   submodule -- the failure mode there is invisible to a per-package pytest run
   and only shows up at tree-wide collection.

2. **The docstrings carry the shape the plan's SC-7 asks for.** This is a
   **PRESENCE-and-SHAPE check, NOT a truth check.** It asserts that a class
   docstring contains a ``.. code-block:: text`` block, that the block draws with
   the house box characters, that a ``⊕`` and a ``[B, ...]`` shape annotation
   appear, that Sphinx ``:param:`` fields are used rather than Google ``Args:``,
   and that every module docstring ends in a ``References:`` block. A
   beautifully-drawn diagram of the WRONG architecture passes every assertion in
   this file. The truth of each mechanism the diagrams depict is pinned
   elsewhere -- by the per-chunk attribution probe, the unpatchify-orientation
   guard, the pos-embed freeze guard and the sampler's oracle comparisons.

3. **The README's parameter table is re-derived, not trusted.** Only the three
   ``DiT-S/*`` rows are re-derived here: building ``B``/``L``/``XL`` at the
   published geometry costs ~14 s and ~4 GB of RSS, which is not a per-run test
   cost. Those nine rows were measured once, by hand, and the README says so.
   A stale ``S`` row reddens; a stale ``XL`` row does not, and that limitation is
   stated rather than papered over.
"""

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import Any, Dict, List

import keras
import numpy as np
import pytest

import dl_techniques.models.vision_language.dit as dit_pkg
from dl_techniques.models.vision_language.dit import (
    DIT_VARIANTS,
    DiT,
    DiTBlock,
    DiTFinalLayer,
)

# ---------------------------------------------------------------------
# Where the package lives on disk, and which modules it is made of.
# ---------------------------------------------------------------------

PACKAGE_DIR = Path(dit_pkg.__file__).parent
SUBMODULE_NAMES = ("blocks", "config", "diffusion", "model")

#: Exported names that are module-level constants rather than callables. Every
#: one is spelled out so that a NEW constant cannot slip into ``__all__``
#: unnoticed by hiding behind a generic "constants are exempt" rule.
DOCUMENTED_CONSTANTS = {
    "CFG_GUIDED_CHANNELS": int,
    "DEFAULT_CLIP_DENOISED": bool,
    "DIT_ADALN_CHUNK_NAMES": tuple,
    "DIT_FINAL_CHUNK_NAMES": tuple,
    "DIT_VARIANTS": dict,
    "MODEL_INPUT_NAMES": tuple,
    "MODEL_MEAN_TYPES": tuple,
    "MODEL_VAR_TYPES": tuple,
    "VARIANT_FIELDS": tuple,
}

#: The classes SC-7 names, mapped to the module that defines them.
SC7_DIAGRAM_CLASSES = {
    "DiT": "dl_techniques.models.vision_language.dit.model",
    "DiTBlock": "dl_techniques.models.vision_language.dit.blocks",
    "DiTFinalLayer": "dl_techniques.models.vision_language.dit.blocks",
    "TimestepEmbedding": "dl_techniques.layers.embedding.timestep_embedding",
}

BOX_CHARACTERS = "┌─┐│└┘▼"

#: The README's own parameter table, at the geometry the README states.
#: Re-derived below for the S family only.
PUBLISHED_GEOMETRY = {
    "input_size": 32,
    "in_channels": 4,
    "num_classes": 1000,
    "learn_sigma": True,
}
README_PARAMETER_TABLE = {
    "DiT-S/2": 32_963_488,
    "DiT-S/4": 32_945_152,
    "DiT-S/8": 33_148_288,
}


#: The plan directory whose ``decisions.md`` is the source of truth for which
#: ``D-NNN`` ids exist. Derived from the repo root, so the test does not care
#: where pytest was invoked from.
PLAN_DECISIONS = (
    PACKAGE_DIR.parents[4]
    / "plans"
    / "plan-2026-09-02T170923-1285ed83"
    / "decisions.md"
)


def _recorded_decision_ids() -> set:
    """The ``D-NNN`` ids the plan's ``decisions.md`` actually records.

    Interface contract: reads the file, matches only headings (``## D-NNN``), and
    drops the schema example's placeholder. Returns a set of ``"D-NNN"`` strings.
    An EMPTY set is never returned -- a silently empty set would make every
    consumer vacuous -- and a MISSING file SKIPS rather than fails.

    Why skip and not raise. ``plans/`` is gitignored (``.gitignore:212:plans/*``;
    ``git ls-files plans/`` returns only ``plans/ANCHORS.md``) and a plan
    directory is deleted when the plan closes, so a hard requirement on this path
    is red on every fresh clone and red here the day this plan closes. The repo
    already settled this shape the other way at
    ``tests/test_analyzer/test_analyzer_docs.py:1305-1310``: a guard pointed at a
    plan artifact "is live exactly while the artifact it points at exists". The
    tracked ``plans/ANCHORS.md`` was rejected as the source instead: it is
    appended at CLOSE, so it records NOTHING for the plan that is still open --
    exactly the window in which this guard has to catch a dangling citation.

    The skip is inside this helper on purpose, so the anti-vacuity arm
    (:meth:`TestThePortNotesExist.test_the_recorded_decision_probe_can_fail`)
    disappears with the arm it protects rather than passing on an empty set.
    """
    if not PLAN_DECISIONS.is_file():
        pytest.skip(
            f"plan decisions.md not present at {PLAN_DECISIONS}; plans/ is "
            "gitignored and plan directories are deleted at CLOSE, so this "
            "guard is live exactly while the artifact it cites exists"
        )
    text = PLAN_DECISIONS.read_text()
    ids = set(re.findall(r"^## (D-\d{3})\b", text, flags=re.MULTILINE))
    assert ids, "decisions.md records no D-NNN headings -- the probe is broken"
    return ids


#: The number of MUST-WRITE-NEW assets SC-8 claims `PORT_NOTES.md` §3 declares:
#: the sin-cos module, `TimestepEmbedding`, `DDPMSchedule` and `DDPMHybridLoss`.
EXPECTED_NEW_ASSET_ROWS = 4


def _new_asset_rows(port_notes_text: str) -> List[str]:
    """The DATA rows of ``PORT_NOTES.md`` §3's MUST-WRITE-NEW table.

    Interface contract: pure over the passed text -- it reads no file, so the
    anti-vacuity arm can drive the identical parser over a table with a row
    added or removed. Slices §3 by its heading and the next ``## `` heading,
    keeps only pipe-delimited lines, and drops the header row and the ``|---|``
    separator.

    :param port_notes_text: The whole ``PORT_NOTES.md`` body.
    :return: One string per data row, in document order.
    """
    lines = port_notes_text.splitlines()
    try:
        start = next(
            i for i, line in enumerate(lines)
            if line.startswith("## 3. What Was BUILT NEW")
        )
    except StopIteration:
        return []
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    rows = [
        line for line in lines[start + 1:end]
        if line.lstrip().startswith("|")
    ]
    return [
        row for row in rows
        if not set(row.replace("|", "").strip()) <= set("- :")
        and not row.lstrip().startswith("| Upstream component")
    ]


def _module_source_paths() -> List[Path]:
    return sorted(
        p for p in PACKAGE_DIR.glob("*.py") if p.name != "__pycache__"
    )


def _class_docstring(qualified_module: str, class_name: str) -> str:
    module = importlib.import_module(qualified_module)
    return inspect.getdoc(getattr(module, class_name)) or ""


# ---------------------------------------------------------------------
# 1. The export surface
# ---------------------------------------------------------------------


class TestTheExportSurface:
    """``__all__`` is a real, curated, non-shadowing surface."""

    def test_all_is_sorted_and_unique(self) -> None:
        names = list(dit_pkg.__all__)
        assert names == sorted(names), "__all__ must stay alphabetized"
        assert len(names) == len(set(names)), "__all__ has a duplicate"

    @pytest.mark.parametrize("name", sorted(dit_pkg.__all__))
    def test_every_exported_name_resolves(self, name: str) -> None:
        assert hasattr(dit_pkg, name), f"{name} is in __all__ but not bound"
        assert getattr(dit_pkg, name) is not None

    def test_no_exported_name_shadows_a_submodule(self) -> None:
        """A curated ``__init__`` that binds ``model`` would shadow ``model.py``.

        The symptom appears only at tree-wide collection, never in a
        per-package pytest run, which is exactly why it gets a named guard.
        """
        clash = set(dit_pkg.__all__) & set(SUBMODULE_NAMES)
        assert not clash, f"__all__ shadows submodule(s): {sorted(clash)}"

        for sub in SUBMODULE_NAMES:
            mod = importlib.import_module(
                f"dl_techniques.models.vision_language.dit.{sub}"
            )
            assert mod.__name__.endswith(sub)

    def test_the_anti_vacuity_sibling(self) -> None:
        """The shadowing check can actually fail.

        Without this, a bug that made ``SUBMODULE_NAMES`` empty would leave the
        guard permanently green.
        """
        assert set(SUBMODULE_NAMES) & {
            p.stem for p in _module_source_paths()
        } == set(SUBMODULE_NAMES)
        pretend_all = list(dit_pkg.__all__) + ["model"]
        assert set(pretend_all) & set(SUBMODULE_NAMES)

    @pytest.mark.parametrize(
        "name,expected_type", sorted(DOCUMENTED_CONSTANTS.items())
    )
    def test_documented_constants_have_the_declared_type(
        self, name: str, expected_type: type
    ) -> None:
        assert name in dit_pkg.__all__
        assert isinstance(getattr(dit_pkg, name), expected_type)

    def test_every_non_constant_export_is_callable(self) -> None:
        for name in dit_pkg.__all__:
            if name in DOCUMENTED_CONSTANTS:
                continue
            obj = getattr(dit_pkg, name)
            assert callable(obj), f"{name} is neither a documented constant nor callable"

    def test_the_exported_objects_are_the_defining_modules_objects(self) -> None:
        """Re-export, not re-definition."""
        for sub in SUBMODULE_NAMES:
            mod = importlib.import_module(
                f"dl_techniques.models.vision_language.dit.{sub}"
            )
            for name in dit_pkg.__all__:
                if hasattr(mod, name):
                    assert getattr(mod, name) is getattr(dit_pkg, name), (
                        f"{name} exported from the package is not the object "
                        f"{sub}.py defines"
                    )


class TestTheExportsAreConstructible:
    """Constructing, not merely importing -- an export that raises on call is
    still a broken surface."""

    def test_the_model_and_its_factory_construct(self) -> None:
        cfg = dict(
            input_size=4, patch_size=2, in_channels=2, hidden_size=16,
            depth=1, num_heads=2, num_classes=3,
        )
        model = dit_pkg.DiT(**cfg)
        assert isinstance(model, keras.Model)
        assert isinstance(dit_pkg.create_dit("DiT-S/2", **cfg), dit_pkg.DiT)

    def test_the_block_layers_construct(self) -> None:
        assert isinstance(
            dit_pkg.DiTBlock(hidden_size=16, num_heads=2), keras.layers.Layer
        )
        assert isinstance(
            dit_pkg.DiTFinalLayer(hidden_size=16, patch_size=2, out_channels=4),
            keras.layers.Layer,
        )

    def test_the_config_and_sampler_construct(self) -> None:
        cfg = dit_pkg.DiffusionConfig(num_timesteps=20, schedule_name="linear")
        gd = dit_pkg.GaussianDiffusion.from_config(cfg, timestep_respacing=2)
        assert gd.num_timesteps == 2

    def test_the_pure_helpers_run(self) -> None:
        init = dit_pkg.flattened_linear_xavier(fan_in=8, fan_out=16)
        assert init(shape=(2, 2, 2, 16)).shape == (2, 2, 2, 16)

        tokens = keras.ops.zeros((1, 6, 2 * 2 * 3))
        out = dit_pkg.unpatchify_tokens(
            tokens, grid_height=2, grid_width=3, patch_size=2, channels=3
        )
        assert tuple(keras.ops.shape(out)) == (1, 4, 6, 3)

    def test_the_lookup_helpers_run(self) -> None:
        assert dit_pkg.normalize_variant_name("s/2") == "DiT-S/2"
        row = dit_pkg.get_variant_config("DiT-S/2")
        assert set(row) == set(dit_pkg.VARIANT_FIELDS)

    def test_model_variants_is_an_alias_not_a_copy(self) -> None:
        assert DiT.MODEL_VARIANTS is DIT_VARIANTS


# ---------------------------------------------------------------------
# 2. Docstring PRESENCE and SHAPE -- not truth
# ---------------------------------------------------------------------


class TestTheDocstringsCarryTheHouseShape:
    """PRESENCE-and-SHAPE only. A correct-looking diagram of the wrong
    architecture passes every assertion in this class."""

    @pytest.mark.parametrize("path", _module_source_paths(), ids=lambda p: p.name)
    def test_every_module_docstring_has_a_references_block(
        self, path: Path
    ) -> None:
        doc = ast.get_docstring(ast.parse(path.read_text())) or ""
        assert doc.strip(), f"{path.name} has no module docstring"
        assert "References:" in doc, f"{path.name} has no References: block"
        assert "2212.09748" in doc or "arxiv.org" in doc.lower(), (
            f"{path.name}'s References: block cites nothing"
        )

    @pytest.mark.parametrize("path", _module_source_paths(), ids=lambda p: p.name)
    def test_no_module_uses_google_style_sections(self, path: Path) -> None:
        """The house style for a NEW ``models/`` package is Sphinx/reST.

        A bare ``Args:`` line at the start of a line is the Google marker; a
        ``:param x:`` field is the Sphinx one. This asserts the file did not
        drift into the other convention, not that either is universally right.
        """
        src = path.read_text()
        assert not re.search(r"^\s*Args:\s*$", src, re.MULTILINE), (
            f"{path.name} carries a Google 'Args:' section"
        )

    @pytest.mark.parametrize(
        "class_name,module", sorted(SC7_DIAGRAM_CLASSES.items())
    )
    def test_the_sc7_classes_carry_an_ascii_diagram(
        self, class_name: str, module: str
    ) -> None:
        doc = _class_docstring(module, class_name)
        assert ".. code-block:: text" in doc, (
            f"{class_name} has no '.. code-block:: text' block"
        )
        used = [ch for ch in BOX_CHARACTERS if ch in doc]
        assert len(used) >= 5, (
            f"{class_name}'s diagram uses only {used!r} of the house box "
            f"characters {BOX_CHARACTERS!r}"
        )
        assert "⊕" in doc, f"{class_name}'s diagram has no '⊕' combination node"
        assert re.search(r"\[B[,\s]", doc), (
            f"{class_name}'s docstring carries no '[B, ...]' shape annotation"
        )
        assert ":param " in doc, f"{class_name} has no Sphinx :param: fields"

    def test_every_public_class_in_the_package_has_a_diagram(self) -> None:
        """Wider than SC-7's four: every public class defined under ``dit/``.

        ``DiffusionConfig`` and ``GaussianDiffusion`` are not in SC-7's named
        list, so they are held only to the code-block requirement, not to the
        ``⊕`` one -- a value object has nothing to combine.
        """
        missing: List[str] = []
        for path in _module_source_paths():
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                    continue
                doc = ast.get_docstring(node) or ""
                if ".. code-block:: text" not in doc or ":param " not in doc:
                    missing.append(f"{path.name}:{node.name}")
        assert not missing, f"public classes without a Sphinx diagram: {missing}"

    def test_the_diagram_check_is_not_vacuous(self) -> None:
        """The population it iterates is non-empty and contains the real names."""
        found = set()
        for path in _module_source_paths():
            tree = ast.parse(path.read_text())
            found |= {
                n.name
                for n in tree.body
                if isinstance(n, ast.ClassDef) and not n.name.startswith("_")
            }
        assert {"DiT", "DiTBlock", "DiTFinalLayer", "GaussianDiffusion"} <= found
        assert len(found) >= 5


# ---------------------------------------------------------------------
# 3. The README's parameter table, re-derived
# ---------------------------------------------------------------------


def _measured_parameter_count(variant: str) -> int:
    model = DiT.from_variant(variant, **PUBLISHED_GEOMETRY)
    size = PUBLISHED_GEOMETRY["input_size"]
    model.build(
        [(None, size, size, PUBLISHED_GEOMETRY["in_channels"]), (None,), (None,)]
    )
    return sum(int(np.prod(w.shape)) for w in model.weights)


class TestTheReadmeParameterTable:
    """A hand-maintained table in a README goes stale silently. This makes the
    three cheap rows reddening rather than silent.

    Scope, stated rather than implied: the nine ``B``/``L``/``XL`` rows are NOT
    re-derived here (~14 s, ~4 GB), so a stale row there stays stale until
    someone re-measures by hand.
    """

    @pytest.mark.parametrize("variant", sorted(README_PARAMETER_TABLE))
    def test_the_small_rows_still_hold(self, variant: str) -> None:
        assert _measured_parameter_count(variant) == README_PARAMETER_TABLE[variant]

    def test_the_readme_actually_contains_the_numbers_it_is_checked_against(
        self,
    ) -> None:
        """If someone edits the table, this fails before the count arm does --
        which distinguishes 'the README changed' from 'the model changed'."""
        readme = (PACKAGE_DIR / "README.md").read_text()
        for variant, count in README_PARAMETER_TABLE.items():
            assert f"{count:,}" in readme, (
                f"README no longer quotes {count:,} for {variant}"
            )
        for variant in DIT_VARIANTS:
            assert f"`{variant}`" in readme, f"README has no row for {variant}"

    def test_the_geometry_is_stated_in_the_readme(self) -> None:
        """A parameter count without its geometry is not a measurement."""
        readme = (PACKAGE_DIR / "README.md").read_text()
        assert "`input_size=32`" in readme
        assert "`in_channels=4`" in readme
        assert "`num_classes=1000`" in readme

    def test_the_count_is_geometry_sensitive(self) -> None:
        """Anti-vacuity: the pinned numbers are not a constant of the variant.

        If the count did not move with ``input_size`` the table's geometry
        caveat would be decoration.
        """
        smaller = DiT.from_variant("DiT-S/2", **{**PUBLISHED_GEOMETRY, "input_size": 16})
        smaller.build([(None, 16, 16, 4), (None,), (None,)])
        moved = sum(int(np.prod(w.shape)) for w in smaller.weights)
        assert moved != README_PARAMETER_TABLE["DiT-S/2"]


class TestThePortNotesExist:
    """``PORT_NOTES.md`` is the package's divergence log, and every ``D-NNN``
    it cites must be one the plan actually recorded."""

    def test_both_documents_are_present(self) -> None:
        assert (PACKAGE_DIR / "README.md").is_file()
        assert (PACKAGE_DIR / "PORT_NOTES.md").is_file()

    def test_the_port_notes_have_the_four_required_sections(self) -> None:
        text = (PACKAGE_DIR / "PORT_NOTES.md").read_text()
        for heading in (
            "## 1. Overview",
            "## 2. What Was REUSED",
            "## 3. What Was BUILT NEW",
            "## 4. What Does NOT Fit",
        ):
            assert heading in text, f"PORT_NOTES.md is missing '{heading}'"

    def test_section_three_declares_exactly_four_new_assets(self) -> None:
        """SC-8's NUMBER, not just its headings.

        SC-8 claims §3 carries "exactly four MUST-WRITE-NEW rows". The heading
        and citation arms above pass with three rows, or with a fifth shared
        asset quietly added and never justified, so the number itself needs its
        own arm. The four are the sin-cos module, ``TimestepEmbedding``,
        ``DDPMSchedule`` and ``DDPMHybridLoss``; the §3 prose says "Four shared
        assets", so a row change without a prose change is also a contradiction.
        """
        text = (PACKAGE_DIR / "PORT_NOTES.md").read_text()
        rows = _new_asset_rows(text)
        assert len(rows) == EXPECTED_NEW_ASSET_ROWS, (
            f"PORT_NOTES.md §3 declares {len(rows)} MUST-WRITE-NEW rows, "
            f"SC-8 claims {EXPECTED_NEW_ASSET_ROWS}: {rows}"
        )
        for name in (
            "sincos_pos_embed_2d",
            "TimestepEmbedding",
            "DDPMSchedule",
            "DDPMHybridLoss",
        ):
            assert any(name in row for row in rows), (
                f"§3 has {EXPECTED_NEW_ASSET_ROWS} rows but none names {name!r}"
            )

    def test_the_row_count_probe_can_fail(self) -> None:
        """Anti-vacuity: the parser above must MOVE when a row moves.

        Drives the identical pure parser over the real §3 with one row deleted
        and with one row appended. A parser that returned a constant -- or that
        swallowed the whole table into a single element -- passes the arm above
        and is worthless; this one reads 3 and 5.
        """
        text = (PACKAGE_DIR / "PORT_NOTES.md").read_text()
        rows = _new_asset_rows(text)
        assert len(rows) == EXPECTED_NEW_ASSET_ROWS

        without = text.replace(rows[-1] + "\n", "")
        assert len(_new_asset_rows(without)) == EXPECTED_NEW_ASSET_ROWS - 1

        extra = "| a fifth thing | **NEW** `nowhere` | unjustified |"
        with_extra = text.replace(rows[-1], rows[-1] + "\n" + extra)
        assert len(_new_asset_rows(with_extra)) == EXPECTED_NEW_ASSET_ROWS + 1

    def test_every_cited_decision_exists_in_the_plan(self) -> None:
        """A ``D-NNN`` reference that points at nothing is worse than none.

        The recorded set is READ from the plan's ``decisions.md``, never pasted.
        Step 8's first version hard-coded ``D-001..D-018`` and went false the
        moment step 10 appended ``D-020`` -- a pinned population with no source
        of truth is a defect waiting for its own commit.
        """
        recorded = _recorded_decision_ids()
        assert len(recorded) >= 18, recorded

        text = (PACKAGE_DIR / "PORT_NOTES.md").read_text()
        cited = sorted(set(re.findall(r"\bD-\d{3}\b", text)))
        assert cited, "PORT_NOTES.md §4 cites no decisions at all"
        dangling = sorted(set(cited) - recorded)
        assert not dangling, (
            f"PORT_NOTES.md cites decisions the plan never recorded: {dangling}"
        )

    def test_the_recorded_decision_probe_can_fail(self) -> None:
        """Anti-vacuity: an id the plan does not have must be reported."""
        recorded = _recorded_decision_ids()
        assert "D-999" not in recorded

    def test_every_anchor_in_the_source_is_discussed(self) -> None:
        """Each ``# DECISION <plan-id>/D-NNN`` anchor under ``dit/`` must have
        its ``D-NNN`` named somewhere in PORT_NOTES.md."""
        text = (PACKAGE_DIR / "PORT_NOTES.md").read_text()
        anchored = set()
        for path in _module_source_paths():
            anchored |= set(
                re.findall(r"# DECISION [^\s/]+/(D-\d{3})", path.read_text())
            )
        assert anchored, "no DECISION anchors found under dit/ -- probe is broken"
        missing = sorted(d for d in anchored if d not in text)
        assert not missing, f"anchors not discussed in PORT_NOTES.md: {missing}"
