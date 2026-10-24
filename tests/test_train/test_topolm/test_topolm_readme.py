"""The TopoLM READMEs may cite only test classes that exist, and none of the figures
that were retracted (plan-2026-10-23T190557-972a9ac5, step 8.2).

A README names the test classes that guard a claim; a class that was renamed or
deleted leaves a sentence pointing at nothing. This reads the two READMEs and
greps the test tree for each cited ``Test...`` name.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_READMES = (
    _ROOT / "src" / "train" / "topolm" / "README.md",
    _ROOT / "src" / "dl_techniques" / "models" / "language" / "topolm" / "README.md",
)

#: Numbers and spellings a review found wrong (an arithmetic slip, a rounded ratio
#: that the table beside it contradicts, a grid that does not exist).
_RETRACTED = ("885,015", "~12x", "(7, 7, 16)")


def _defined_test_classes():
    names = set()
    for path in (_ROOT / "tests").rglob("*.py"):
        names.update(re.findall(r"^class (Test\w+)", path.read_text(encoding="utf-8"), re.M))
    return names


@pytest.mark.parametrize("readme", _READMES, ids=lambda p: p.parent.name)
def test_every_test_class_a_readme_cites_exists(readme):
    cited = set(re.findall(r"`(Test[A-Za-z0-9_]+)`", readme.read_text(encoding="utf-8")))
    assert cited, "the README cites no test class, so this guard would pass vacuously"
    missing = sorted(cited - _defined_test_classes())
    assert missing == [], f"{readme.name} cites test classes that do not exist: {missing}"


@pytest.mark.parametrize("readme", _READMES, ids=lambda p: p.parent.name)
def test_no_retracted_figure_is_back(readme):
    text = readme.read_text(encoding="utf-8")
    assert [literal for literal in _RETRACTED if literal in text] == []
