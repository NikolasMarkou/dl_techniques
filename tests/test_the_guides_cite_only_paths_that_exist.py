"""Every test or source path the two GUIDEs cite exists, and every test they name by
function is defined in the file named beside it.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 13 (D-251).
``src/dl_techniques/models/GUIDE.md`` § 10 and ``src/train/GUIDE.md`` give, for each
defect pattern, the guard that now prevents it. A guard path that rots (a renamed or
deleted test) leaves the pattern pointing at nothing, with no symptom. The GUIDE
convention, stated in both files: a test file named without a directory sits in the
directory of the previous path in the same entry (a ``- `` bullet); a bare name with no
earlier path in its entry (the older sections name a template by its name) must exist
somewhere under ``tests/``.

Checked: every backticked ``tests/...`` or ``src/...`` path (a ``::name``, ``:line`` or
trailing ``/`` is stripped; a ``file.py:symbol`` must define the symbol), every bare backticked ``test_*.py`` resolved by that
convention, and every ``` `test_name` in `path` ``` pair (``def test_name`` must be in
the file).
"""

import re
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    "src/dl_techniques/models/GUIDE.md",
    "src/train/GUIDE.md",
)
PATH = re.compile(r"`((?:tests|src)/[^`\s]+)`")
BARE = re.compile(r"`(test_[A-Za-z0-9_]+\.py)`")
SYMBOL = re.compile(r"`((?:tests|src)/[^`\s:]+\.py):([A-Za-z_]\w*)`")
NAMED = re.compile(r"`(test_[A-Za-z0-9_]+)` in\s+`((?:tests|src)/[^`\s]+\.py)`")


def _entries(text: str) -> Iterator[str]:
    """Bullets (``- `` and its continuation lines) and other paragraphs."""
    entry: List[str] = []
    for line in text.splitlines():
        if line.startswith("- ") or not line.strip() or line.startswith("#"):
            if entry:
                yield "\n".join(entry)
            entry = [] if not line.startswith("- ") else [line]
        else:
            entry.append(line)
    if entry:
        yield "\n".join(entry)


def _clean(path: str) -> str:
    path = path.split("::")[0]
    path = re.sub(r":\d+(?:[-,]\d+)*$", "", path)
    path = re.sub(r":[A-Za-z_]\w*$", "", path)  # `file.py:symbol`
    return path.rstrip("/")


def citations(text: str) -> Iterator[Tuple[str, str]]:
    """``(kind, path)`` for every citation in a GUIDE's text.

    ``kind`` is ``"path"`` for a path to check, ``"def:<name>"`` for a test function
    that must be defined in ``path``.
    """
    for entry in _entries(text):
        last_dir = None
        tokens = sorted(
            [(m.start(), "path", m.group(1)) for m in PATH.finditer(entry)]
            + [(m.start(), "bare", m.group(1)) for m in BARE.finditer(entry)])
        for _, kind, token in tokens:
            if kind == "path":
                path = _clean(token)
                yield "path", path
                if path.startswith("tests/"):
                    last_dir = path if (ROOT / path).is_dir() else str(Path(path).parent)
            elif last_dir is not None:
                yield "path", f"{last_dir}/{token}"
            else:
                # A bare name outside a § 10-style entry (the older sections name a
                # template file by its name alone): it must exist somewhere in tests/.
                found = sorted(ROOT.joinpath("tests").rglob(token))
                yield "path", (str(found[0].relative_to(ROOT)) if found
                                 else f"tests/**/{token}")
        for match in NAMED.finditer(entry):
            yield f"def:{match.group(1)}", _clean(match.group(2))
        for match in SYMBOL.finditer(entry):
            yield f"sym:{match.group(2)}", match.group(1)


def _problems(text: str) -> List[str]:
    problems = []
    for kind, path in citations(text):
        target = ROOT / path
        if not target.exists():
            problems.append(f"missing: {path}")
        elif kind.startswith("def:"):
            name = kind[4:]
            if not re.search(rf"^\s*def {name}\(", target.read_text(), re.M):
                problems.append(f"no def {name} in {path}")
        elif kind.startswith("sym:"):
            name = kind[4:]
            if not re.search(rf"^(?:def |class )?{name}\b", target.read_text(), re.M):
                problems.append(f"no {name} defined in {path}")
    return problems


@pytest.mark.parametrize("guide", GUIDES)
def test_every_cited_path_and_test_exists(guide):
    text = (ROOT / guide).read_text()
    found = list(citations(text))
    assert len(found) > 20, f"{guide}: only {len(found)} citations parsed"
    problems = _problems(text)
    assert not problems, f"{guide} cites what does not exist:\n" + "\n".join(problems)


def test_the_guides_link_each_other():
    models = (ROOT / GUIDES[0]).read_text()
    train = (ROOT / GUIDES[1]).read_text()
    assert "src/train/GUIDE.md" in models or "../../train/GUIDE.md" in models
    assert "models/GUIDE.md" in train
    assert "GUIDE.md" in (ROOT / "src/train/AGENTS.md").read_text()


def test_the_checker_flags_what_it_must():
    """Planted text: a missing path, a bare name resolved to a missing file, a test
    function absent from its file, a missing `file.py:symbol`; the present ones pass."""
    planted = (
        "- **x.** *Guard:* `tests/test_train/test_common_fill_batch.py`,\n"
        "  `test_no_such_file.py`. `tests/no/such/dir/test_x.py`.\n"
        "- **y.** `test_no_such_function` in\n"
        "  `tests/test_train/test_common_fill_batch.py`.\n"
        "- **z.** `test_no_such_template.py` is the template.\n"
        "- **w.** `tests/numerics.py:no_such_symbol` and `tests/numerics.py:reassociation_atol`.\n"
    )
    problems = _problems(planted)
    assert "missing: tests/test_train/test_no_such_file.py" in problems
    assert "missing: tests/no/such/dir/test_x.py" in problems
    assert any(p.startswith("no def test_no_such_function") for p in problems)
    assert "missing: tests/**/test_no_such_template.py" in problems
    assert "no no_such_symbol defined in tests/numerics.py" in problems
    assert len(problems) == 5, problems
