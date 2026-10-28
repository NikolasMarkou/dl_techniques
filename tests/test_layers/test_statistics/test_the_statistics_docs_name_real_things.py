"""The four statistics docs name only things that exist, and name all of them.

``layers/statistics/README.md``, ``AGENTS.md`` and the pair in ``conformal/`` make claims a
reader acts on: a path to open, a class to import, a snippet to paste, a count, a list of
outside importers. A doc nobody tests drifts (this package's old table in ``layers/AGENTS.md``
said "Google-style docstrings" for eleven modules that all use Sphinx). This module derives
each of those claims from the code and compares:

* every repo path a doc names, and every relative link, exists; every bare file name resolves
  to a file somewhere under ``src/`` or ``tests/``;
* every class the package defines is named in the docs of its package (and the catalogue
  names the conformal classes too);
* every ```python block of the two READMEs runs, on CPU, from a fixed seed;
* the registered keys the docs quote are the real ones;
* the counts and lists the docs state (class names, outside importers, the docstring style,
  the ``from keras import`` modules) equal what is derived.

Each check is proven RED by an injection recorded in the commit that adds this file.
"""

import ast
import re
from pathlib import Path

import keras
import pytest

REPO = Path(__file__).resolve().parents[3]
STATS = REPO / "src" / "dl_techniques" / "layers" / "statistics"
CONF = STATS / "conformal"

DOCS = {
    "statistics/README.md": STATS / "README.md",
    "statistics/AGENTS.md": STATS / "AGENTS.md",
    "conformal/README.md": CONF / "README.md",
    "conformal/AGENTS.md": CONF / "AGENTS.md",
}
READMES = ["statistics/README.md", "conformal/README.md"]


def _text(name: str) -> str:
    return DOCS[name].read_text()


def _classes(directory: Path) -> set:
    out = set()
    for path in directory.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                out.add(node.name)
    return out


STATS_CLASSES = _classes(STATS)
CONF_CLASSES = _classes(CONF)
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten"}


def test_the_class_census_is_not_empty():
    # A broken walk would make every "names every class" check below pass vacuously.
    assert len(STATS_CLASSES) >= 11 and len(CONF_CLASSES) >= 4


def test_the_four_docs_exist():
    for name, path in DOCS.items():
        assert path.is_file() and path.stat().st_size > 1000, name


# ---------------------------------------------------------------------
# Paths and links
# ---------------------------------------------------------------------

_BASENAMES = {
    p.name
    for root in ("src", "tests", "research")
    for p in (REPO / root).rglob("*")
    if p.is_file() and "__pycache__" not in p.parts
}


def _code_spans(text: str):
    return re.findall(r"`([^`\n]+)`", text)


@pytest.mark.parametrize("name", DOCS)
def test_every_repo_path_a_doc_names_exists(name):
    checked = 0
    for span in _code_spans(_text(name)):
        token = span.split("::")[0].split(" ")[0].rstrip("/,.;:")
        if "*" in token or "<" in token or "..." in token or not token:
            continue
        if token.startswith(("src/", "tests/", "research/")):
            assert (REPO / token).exists(), f"{name}: {span!r} does not exist"
            checked += 1
        elif re.fullmatch(r"[\w.\-]+\.(py|md|json|keras)", token) and "/" not in token:
            assert token in _BASENAMES or token in {"conformal.json", "conformal_model.keras"}, (
                f"{name}: {span!r} matches no file under src/, tests/ or research/"
            )
            checked += 1
    assert checked >= 3, f"{name}: no path claims found; the extraction is broken"


@pytest.mark.parametrize("name", DOCS)
def test_every_relative_link_resolves(name):
    for target in re.findall(r"\]\(([^)#\s]+)\)", _text(name)):
        if "://" in target:
            continue
        assert (DOCS[name].parent / target).exists(), f"{name}: link {target!r} is dead"


# ---------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------


def test_the_statistics_docs_name_every_class_of_their_package():
    for doc in ("statistics/README.md", "statistics/AGENTS.md"):
        missing = {c for c in STATS_CLASSES if c not in _text(doc)}
        assert not missing, f"{doc} never names {sorted(missing)}"


def test_the_conformal_docs_name_every_class_of_the_subpackage():
    for doc in ("conformal/README.md", "conformal/AGENTS.md", "statistics/README.md"):
        missing = {c for c in CONF_CLASSES if c not in _text(doc)}
        assert not missing, f"{doc} never names {sorted(missing)}"


def test_the_statistics_readme_names_the_mdn_helpers_and_the_attach_function():
    text = _text("statistics/README.md")
    for helper in ("get_point_estimate", "get_uncertainty", "get_prediction_intervals",
                   "check_component_diversity", "attach_conformal_sets"):
        assert helper in text


# ---------------------------------------------------------------------
# Snippets run
# ---------------------------------------------------------------------


def _python_blocks(name: str):
    return re.findall(r"```python\n(.*?)```", _text(name), flags=re.S)


def test_there_are_snippets_to_run():
    assert len(_python_blocks("statistics/README.md")) >= 3
    assert len(_python_blocks("conformal/README.md")) >= 3


@pytest.mark.parametrize(
    "name, index",
    [(n, i) for n in READMES for i in range(len(_python_blocks(n)))],
)
def test_every_readme_snippet_runs(name, index):
    keras.utils.set_random_seed(0)
    code = _python_blocks(name)[index]
    exec(compile(code, f"<{name} block {index}>", "exec"), {"__name__": "__doc_snippet__"})


# ---------------------------------------------------------------------
# Quoted keys and counts
# ---------------------------------------------------------------------


def test_the_registered_keys_the_docs_quote_are_the_real_ones():
    from dl_techniques.layers.statistics.conformal import (
        ConformalInterval,
        ConformalLayer,
        ConformalPredictionSet,
        ConformalQuantileHead,
    )
    from dl_techniques.layers.statistics.scaler import UnifiedScaler

    assert keras.saving.get_registered_name(UnifiedScaler) == (
        "dl_techniques.layers.statistics.scaler>UnifiedScaler")
    quoted = r"dl_techniques\.layers\.statistics\.scaler>UnifiedScaler(?!\w)"
    assert re.search(quoted, _text("statistics/README.md"))
    for cls, module in ((ConformalLayer, "base"), (ConformalInterval, "interval"),
                        (ConformalQuantileHead, "quantile_head"),
                        (ConformalPredictionSet, "prediction_set")):
        key = f"dl_techniques.layers.statistics.conformal.{module}>{cls.__name__}"
        assert keras.saving.get_registered_name(cls) == key
        text = _text("conformal/AGENTS.md")
        assert f"conformal.{module}" in text or f"`.{module}`" in text, module


def test_the_stated_class_count_is_the_derived_one():
    total = len(STATS_CLASSES) + len(CONF_CLASSES)
    assert f"prints {total} names" in _text("statistics/AGENTS.md")


def test_the_stated_outside_importers_are_the_derived_ones():
    pattern = re.compile(r"(layers\.statistics|\.\.statistics)[.a-z_]* import")
    found = set()
    for path in (REPO / "src").rglob("*.py"):
        if STATS in path.parents or "__pycache__" in path.parts:
            continue
        for line in path.read_text().splitlines():
            if pattern.search(line):
                found.add(str(path.relative_to(REPO)))
    expected = {
        "src/dl_techniques/layers/adapters/gated_adapter.py",
        "src/dl_techniques/layers/time_series/forecasting_layers.py",
        "src/dl_techniques/models/time_series/mdn/model.py",
        "src/train/capsnet/train_capsnet.py",
        "src/train/common/conformal.py",
        "src/train/time_series/mdn/export.py",
        "src/train/time_series/mdn/train_mdn.py",
    }
    assert found == expected, (
        f"outside importers changed: added {sorted(found - expected)}, "
        f"removed {sorted(expected - found)}; update the docs and this list"
    )
    readme = _text("statistics/README.md")
    for path in expected:
        short = path.removeprefix("src/dl_techniques/")
        if path.startswith("src/train/"):
            short = path.removeprefix("src/train/")
        assert short.split("/")[-1] in readme or short in readme, f"README never names {path}"


def test_the_docstring_style_claim_is_true():
    modules = sorted(STATS.glob("*.py")) + sorted(CONF.glob("*.py"))
    modules = [m for m in modules if m.name != "__init__.py"]
    assert len(modules) >= 12
    for module in modules:
        text = module.read_text()
        assert ":param" in text, f"{module.name} has no Sphinx :param:"
        assert "Args:" not in text, f"{module.name} uses Google-style Args:"
    assert "Sphinx/reST" in _text("statistics/AGENTS.md")


def test_the_from_keras_import_claim_is_true():
    # The eight older modules say `from keras import ops`; probability_logits.py (new code)
    # says `import keras` only.
    top_level = [m for m in STATS.glob("*.py")
                 if m.name not in ("__init__.py", "probability_logits.py")]
    using = [m for m in top_level if re.search(r"^from keras import", m.read_text(), re.M)]
    assert len(using) == len(top_level), (
        f"{len(using)} of {len(top_level)} modules use `from keras import`; the AGENTS claim "
        f"says all of them do. Update it."
    )
    assert f"All {NUMBER_WORDS[len(top_level)]} older top-level" in _text("statistics/AGENTS.md")
    newer = (STATS / "probability_logits.py").read_text()
    assert not re.search(r"^from keras import", newer, re.M)
    for module in CONF.glob("*.py"):
        assert not re.search(r"^from keras import", module.read_text(), re.M), module.name
