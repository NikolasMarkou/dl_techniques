"""The TopoLM READMEs may cite only test classes that exist, and none of the figures
that were retracted (plan-2026-10-23T190557-972a9ac5, step 8.2).

A README names the test classes that guard a claim; a class that was renamed or
deleted leaves a sentence pointing at nothing. This reads the two READMEs and
greps the test tree for each cited ``Test...`` name.
"""

import hashlib
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


# ---------------------------------------------------------------------------
# The plain-language guide to the figures (plan-2026-10-23T190557-972a9ac5,
# step 8.3). It has images, relative links and a heading that other text points
# at, and the opening of the README was asked to stay as it was.
# ---------------------------------------------------------------------------

_TRAINER_README = _READMES[0]
_GUIDE_HEADING = "### Reading the figures: a plain-language guide"
_WATCHING_HEADING = "### Watching a run: `--viz-every-steps`"

#: The first line of the README and the sha256 of its first 37 lines (a trailing
#: newline after each), pinned at the value they had before the guide was added.
_FIRST_LINE = "# `train/topolm`: pre-training a topographic language model"
_FIRST_37_LINES_SHA256 = "b90557ba90fa450f8dceb898fe899196e8b2baac32b32316258d684684a33ecb"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_EM_DASH = "—"


def _readme_text() -> str:
    return _TRAINER_README.read_text(encoding="utf-8")


def _guide_text() -> str:
    """The guide: from its heading up to the next ``###`` heading."""
    text = _readme_text()
    start = text.index(_GUIDE_HEADING)
    end = text.index(_WATCHING_HEADING)
    return text[start:end]


def _relative_targets(text: str):
    """Every relative file target of a markdown link or an ``src=`` / ``href=``
    attribute, with a ``#fragment`` removed; absolute URLs and bare anchors are
    skipped."""
    targets = re.findall(r"\]\(([^)\s]+)\)", text)
    targets += re.findall(r"""(?:src|href)=["']([^"']+)["']""", text)
    out = []
    for target in targets:
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        out.append(target.split("#", 1)[0])
    return out


def _github_slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped,
    spaces turned into hyphens."""
    title = heading.lstrip("#").strip().lower()
    title = re.sub(r"[^\w\s-]", "", title)
    return re.sub(r"\s", "-", title)


def test_every_relative_link_and_image_the_readme_uses_exists():
    targets = _relative_targets(_readme_text())
    assert targets, "no relative link at all, so this guard would pass vacuously"
    missing = [t for t in targets if not (_TRAINER_README.parent / t).resolve().exists()]
    assert missing == [], f"links that point at nothing: {missing}"


def test_every_image_the_guide_shows_is_a_real_png():
    guide = _guide_text()
    images = sorted({t for t in _relative_targets(guide) if t.endswith(".png")})
    assert len(images) >= 7, f"the guide references only {images}"
    for image in images:
        path = _TRAINER_README.parent / image
        assert path.is_file(), f"{image} is referenced but does not exist"
        data = path.read_bytes()
        assert len(data) > 0, f"{image} is empty"
        assert data.startswith(_PNG_SIGNATURE), f"{image} is not a PNG"


def test_no_figure_file_sits_unreferenced_in_the_figures_directory():
    shipped = {p.name for p in (_TRAINER_README.parent / "figures").glob("*")}
    referenced = {Path(t).name for t in _relative_targets(_guide_text())}
    assert shipped - referenced == set(), "an image nothing in the guide shows"


def test_the_guide_sits_after_the_figures_table_and_before_watching_a_run():
    text = _readme_text()
    assert text.count(_GUIDE_HEADING) == 1
    figures = text.index("### Figures\n")
    guide = text.index(_GUIDE_HEADING)
    watching = text.index(_WATCHING_HEADING)
    assert figures < guide < watching


def test_the_two_pointers_to_the_guide_resolve_to_its_heading():
    text = _readme_text()
    anchor = "#" + _github_slug(_GUIDE_HEADING)
    figures_intro = text[text.index("### Figures\n"):text.index("| File | Contents |")]
    watching_start = text.index(_WATCHING_HEADING)
    watching_first_sentence = text[watching_start:watching_start + 900]
    assert anchor in figures_intro
    assert anchor in watching_first_sentence


def test_every_in_page_anchor_names_a_heading_that_exists():
    text = _readme_text()
    slugs = {
        _github_slug(line)
        for line in text.splitlines()
        if re.match(r"#{1,6} ", line)
    }
    anchors = re.findall(r"\]\(#([^)\s]+)\)", text)
    assert anchors, "no in-page anchor at all, so this guard would pass vacuously"
    assert sorted(set(anchors) - slugs) == []


def test_the_first_37_lines_of_the_readme_are_unchanged():
    lines = _readme_text().splitlines()
    assert lines[0] == _FIRST_LINE
    digest = hashlib.sha256(("\n".join(lines[:37]) + "\n").encode("utf-8")).hexdigest()
    assert digest == _FIRST_37_LINES_SHA256


def test_the_guide_has_no_em_dash_and_no_retracted_figure():
    guide = _guide_text()
    assert _EM_DASH not in guide
    assert [literal for literal in _RETRACTED if literal in guide] == []
