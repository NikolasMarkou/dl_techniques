"""The TopoLM READMEs may cite only test classes that exist, and none of the figures
that were retracted (plan-2026-10-23T190557-972a9ac5, step 8.2); and the guide to
the figures is held to the DATA of the figures, not only to its own form (steps 8.4
and 8.5, decision D-022).

A README names the test classes that guard a claim; a class that was renamed or
deleted leaves a sentence pointing at nothing. This reads the two READMEs and
greps the test tree for each cited ``Test...`` name.

``src/train/topolm/figures/figure_facts.json`` is written, by a script, from the
real runs the example images came from (and, for the one synthetic image, from the
script that made it) (``metrics.jsonl``,
``topography_report.json``, the PNG files). The tests below pin every image to the
SHA-256, byte size and pixel size recorded there (a swapped figure fails) and require
each caption to print the exact string the facts hold for each number it quotes (a
caption that drifted from the data fails). A review once swapped two figures and
changed a caption number to 42.0 and every form-only test stayed green.
"""

import hashlib
import json
import re
import struct
from pathlib import Path

import pytest
import scipy.stats

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
_FACTS_PATH = _TRAINER_README.parent / "figures" / "figure_facts.json"
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
    assert len(images) >= 6, f"the guide references only {images}"
    for image in images:
        path = _TRAINER_README.parent / image
        assert path.is_file(), f"{image} is referenced but does not exist"
        data = path.read_bytes()
        assert len(data) > 0, f"{image} is empty"
        assert data.startswith(_PNG_SIGNATURE), f"{image} is not a PNG"


def test_no_figure_file_sits_unreferenced_in_the_figures_directory():
    shipped = {p.name for p in (_TRAINER_README.parent / "figures").glob("*")}
    referenced = {Path(t).name for t in _relative_targets(_guide_text())}
    # The facts file is data, not an image: the guide names it in prose, not as a link.
    assert shipped - referenced == {_FACTS_PATH.name}, "a file nothing in the guide shows"
    assert _FACTS_PATH.name in _guide_text()


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


# ---------------------------------------------------------------------------
# Truth, not form (steps 8.4 and 8.5, D-022): the guide against the figures' data.
# ---------------------------------------------------------------------------


def _facts() -> dict:
    return json.loads(_FACTS_PATH.read_text(encoding="utf-8"))


def _shipped(facts: dict = None) -> dict:
    """The facts entries of the images that are shipped (an omitted figure, one no
    run wrote, has an entry with ``omitted`` and no file)."""
    facts = facts or _facts()
    return {name: entry for name, entry in facts["files"].items() if not entry.get("omitted")}


def _squash(text: str) -> str:
    """Whitespace collapsed to single spaces, so a phrase may wrap across lines."""
    return " ".join(text.split())


def _g(value: float) -> str:
    """``0.0`` -> ``0.0``, ``4.25`` -> ``4.25``, ``5.5`` -> ``5.5``: at most three
    decimals, at least one."""
    text = f"{value:.3f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _png_size(data: bytes):
    """Width and height from the IHDR chunk of a PNG (bytes 16 to 24)."""
    return struct.unpack(">II", data[16:24])


def _figure_context(name: str) -> dict:
    """Every ``fmt`` string the facts hold for one image, flat, plus its pixel size.

    ``per_tap`` and ``colour_limit`` are nested; ``colour_limit`` is flattened to
    its own ``fmt`` and the per-tap lists are read by the tests that need them.
    """
    entry = _facts()["files"][name]
    context = {
        key: value["fmt"]
        for key, value in entry["facts"].items()
        if isinstance(value, dict) and "fmt" in value
    }
    context["width"], context["height"] = entry["width"], entry["height"]
    return context


def _caption(name: str, guide: str = None) -> str:
    """The italic paragraph that follows the image tag of ``name`` in the guide,
    whitespace collapsed."""
    guide = _guide_text() if guide is None else guide
    start = guide.index(f"figures/{name}")
    match = re.search(r"\n\n(\*.*?)\n\n", guide[start:], re.S)
    assert match, f"no caption paragraph follows {name}"
    return _squash(match.group(1))


def _pin_failures(directory: Path, entries: dict):
    """Images of ``entries`` whose file in ``directory`` is not the one the facts
    describe (hash, bytes or pixel size), and PNGs there that no entry describes."""
    failures = []
    for name, entry in entries.items():
        path = directory / name
        if not path.is_file():
            failures.append(name)
            continue
        data = path.read_bytes()
        if (
            hashlib.sha256(data).hexdigest() != entry["sha256"]
            or len(data) != entry["bytes"]
            or _png_size(data) != (entry["width"], entry["height"])
        ):
            failures.append(name)
    failures += sorted(p.name for p in directory.glob("*.png") if p.name not in entries)
    return failures


def test_the_facts_file_pins_every_image_by_hash_size_and_pixels():
    assert _pin_failures(_TRAINER_README.parent / "figures", _shipped()) == []


def test_every_cluster_figure_a_run_wrote_is_shipped_and_shown():
    """Both example runs found a condition-A cluster and so wrote ``clusters_raw.png``:
    the facts list both, the files are shipped and the guide shows them. (A run that
    finds none writes no such figure; the facts would then carry an ``omitted`` entry
    with a reason and no file, and this test would say so.)"""
    files = _facts()["files"]
    assert [name for name, entry in files.items() if entry.get("omitted")] == []
    for name, run in (("clusters_2d.png", "2d"), ("clusters_3d.png", "3d")):
        assert files[name]["run"] == run
        assert (_TRAINER_README.parent / "figures" / name).is_file()
        assert f"figures/{name}" in _guide_text()
        assert files[name]["facts"]["total_clusters_a"]["value"] > 0, "a cluster figure needs a cluster of a"


#: What each caption must print, as ``str.format`` templates over the ``fmt``
#: strings of that image's facts (``_figure_context``). Written once, by hand, from
#: the captions; the NUMBERS come from the facts, so a caption that drifts from the
#: data (or a facts file rebuilt while the caption stayed) fails here.
_CAPTION_QUOTES = {
    "t_maps_2d_untrained.png": (
        "step {step}",
        "Mean Moran's I {mean_morans_i}",
        "{mean_significant_units_per_tap} significant units per tap",
        "{total_clusters_a} clusters",
        "t values run from {t_min_over_taps} to {t_max_over_taps}",
        "+/-{colour_limit}",
        "({width} x {height})",
    ),
    "t_maps_2d_trained.png": (
        "Mean Moran's I {mean_morans_i} (per tap {morans_i_min_over_taps} to {morans_i_max_over_taps})",
        "{mean_significant_units_per_tap} significant units per tap ({significant_units_min_over_taps} to "
        "{significant_units_max_over_taps} of {units_per_tap}, depending on the tap)",
        "{total_clusters_a} clusters for `a` and {total_clusters_b} for `b` over the {num_taps} taps",
        "t values run from {t_min_over_taps} to {t_max_over_taps}",
        "+/-{colour_limit}",
        "The unhatched cells inside the outlines are the significant ones",
        "({width} x {height})",
    ),
    "t_maps_3d_trained.png": (
        "({units_per_tap} units per tap on a {grid_shape} box)",
        "Mean Moran's I {mean_morans_i} (per tap {morans_i_min_over_taps} to {morans_i_max_over_taps})",
        "{significant_units_min_over_taps} to {significant_units_max_over_taps} significant units per tap"
        " (mean {mean_significant_units_per_tap} of {units_per_tap})",
        "{total_clusters_a} clusters for `a` and {total_clusters_b} for `b` over the {num_taps} taps",
        "t values from {t_min_over_taps} to {t_max_over_taps}",
        "colour scale +/-{colour_limit}",
        "({width} x {height})",
    ),
    "clusters_2d.png": (
        "Left, the 2D run: `{cluster_title}`, of {cluster_sizes} units ({width} x {height})",
    ),
    "clusters_3d.png": (
        "Right, the 3D run: `{cluster_title}`, of {cluster_sizes} unit",
        "({width} x {height})",
    ),
    "t_maps_2d_alpha0_control.png": (
        "Mean Moran's I {mean_morans_i} (per tap {morans_i_min_over_taps} to {morans_i_max_over_taps})",
        "{mean_significant_units_per_tap} significant units per tap ({significant_units_min_over_taps} to "
        "{significant_units_max_over_taps} of {units_per_tap}, depending on the tap)",
        "{total_clusters_a} clusters for `a` and {total_clusters_b} for `b` over the {num_taps} taps",
        "t values run from {t_min_over_taps} to {t_max_over_taps}",
        "+/-{colour_limit}",
        "({width} x {height})",
    ),
    "t_maps_2d_illustration.png": (
        "SYNTHETIC illustration, made-up numbers, not a run",
        "{grid_shape} maps",
        "one BH-FDR correction across the {panels_word} panels",
        "{width} x {height}",
    ),
    "morans_i_by_depth_2d.png": (
        "depth order: {per_tap_morans_i}",
        "{width} x {height}",
    ),
    "live_curve_2d.png": (
        "Full-size file {width} x {height}",
    ),
}


def _quote_context(name: str, facts_all: dict = None) -> dict:
    facts_all = facts_all or _facts()
    entry = facts_all["files"][name]
    facts = entry["facts"]
    context = {
        key: value["fmt"]
        for key, value in facts.items()
        if isinstance(value, dict) and "fmt" in value
    }
    context["width"], context["height"] = entry["width"], entry["height"]
    context["colour_limit"] = facts["colour_limit"]["fmt"] if "colour_limit" in facts else ""
    if "per_tap" in facts and name != "t_maps_2d_illustration.png":
        context["per_tap_morans_i"] = ", ".join(v["morans_i"]["fmt"] for v in facts["per_tap"].values())
    if "cluster_figure_tap" in facts:
        tap = facts["cluster_figure_tap"]
        context["cluster_title"] = tap["title_as_drawn"]
        context["cluster_sizes"] = " and ".join(str(n) for n in tap["cluster_sizes_a"]["value"])
    if name == "t_maps_2d_illustration.png":
        context["panels_word"] = {4: "four"}[facts["panels"]["value"]]
    return context


def _caption_failures(name: str, guide: str, facts_all: dict = None):
    """The phrases the caption of ``name`` in ``guide`` should print and does not."""
    caption = _caption(name, guide)
    context = _quote_context(name, facts_all)
    return [
        _squash(template.format(**context))
        for template in _CAPTION_QUOTES[name]
        if _squash(template.format(**context)) not in caption
    ]


@pytest.mark.parametrize("name", sorted(_CAPTION_QUOTES))
def test_each_caption_prints_the_numbers_the_facts_hold(name):
    missing = _caption_failures(name, _guide_text())
    assert missing == [], f"{name}: the caption does not say {missing}; it says {_caption(name)!r}"


def test_every_shipped_figure_has_a_caption_row_in_the_quote_table():
    assert sorted(_CAPTION_QUOTES) == sorted(_shipped())


def test_the_trained_examples_have_islands_and_the_guide_says_so():
    """The central fact of both examples: a smooth map in which part is significant,
    in outlined islands, and part is not."""
    facts_all = _facts()["files"]
    for name, units in (("t_maps_2d_trained.png", 256), ("t_maps_3d_trained.png", 343)):
        facts = facts_all[name]["facts"]
        assert facts["units_per_tap"]["value"] == units
        mean = facts["mean_significant_units_per_tap"]["value"]
        assert 0 < mean < units / 2, "part of the map is significant and part is not"
        assert facts["total_clusters_a"]["value"] > 0 and facts["total_clusters_b"]["value"] > 0
        assert facts["mean_morans_i"]["value"] > 0.5
    untrained = facts_all["t_maps_2d_untrained.png"]["facts"]
    assert untrained["significant_units_max_over_taps"]["value"] == 0
    section = _squash(_guide_text()).split("#### Reading a t-map")[1].split("#### Hatching and outlines")[0]
    assert "The map is smooth, and part of it is significant" in section
    assert "Look at the unhatched islands of the trained map" in section
    # The 2D hatch pixel check recorded with the facts: as many cells with a stroke as rejected cells.
    trained = facts_all["t_maps_2d_trained.png"]["facts"]
    rejected = sum(
        trained["units_per_tap"]["value"] - v["significant_units"]["value"] for v in trained["per_tap"].values()
    )
    total = trained["units_per_tap"]["value"] * trained["num_taps"]["value"]
    assert f"{rejected} of {total} cells with a stroke in the trained figure" in _squash(_guide_text())


def test_the_synthetic_illustration_is_synthetic_in_the_facts_and_the_guide():
    entry = _facts()["files"]["t_maps_2d_illustration.png"]
    assert entry["kind"] == "synthetic"
    assert entry["run"] is None and entry["step"] is None
    assert _illustration_failures(_guide_text(), _facts()) == []
    # No run image of the guide is of the synthetic kind, and every other one has a run.
    for name, other in _shipped().items():
        if name != "t_maps_2d_illustration.png":
            assert other.get("kind") != "synthetic" and other["run"] in ("2d", "3d", "2d_alpha0"), name


def _illustration_failures(guide: str, facts_all: dict):
    """What the synthetic illustration must carry: the word in the caption (as the
    sentence ``SYNTHETIC illustration, made-up numbers, not a run``), in the alt text,
    and ``kind: synthetic`` in the facts."""
    problems = []
    if facts_all["files"]["t_maps_2d_illustration.png"].get("kind") != "synthetic":
        problems.append("facts kind")
    caption = _caption("t_maps_2d_illustration.png", guide)
    if "synthetic" not in caption.lower() or "made-up numbers, not a run" not in caption:
        problems.append("caption")
    tag = guide[guide.index("figures/t_maps_2d_illustration.png") - 20:]
    alt = re.search(r'alt="([^"]*)"', tag).group(1)
    if "SYNTHETIC" not in alt:
        problems.append("alt text")
    # The sentence that introduces the image says it too.
    before = guide[: guide.index("figures/t_maps_2d_illustration.png")]
    if "synthetic" not in before[-700:].lower():
        problems.append("no word before the image")
    return problems


def test_the_illustration_panels_are_what_the_caption_says():
    entry = _facts()["files"]["t_maps_2d_illustration.png"]
    per_panel = entry["facts"]["per_panel"]
    assert len(per_panel) == 4
    for stats in per_panel.values():
        assert stats["clusters_a"] >= 1 and stats["clusters_b"] >= 1
        assert all(size >= entry["facts"]["min_cluster_size"]["value"] for size in stats["cluster_sizes_a"] + stats["cluster_sizes_b"])
    first = per_panel["synthetic/tap_1"]["significant_units"]
    assert f"the first has {first} significant units" in _caption("t_maps_2d_illustration.png")
    assert "scored by the same chain as a real run" in _caption("t_maps_2d_illustration.png")


# --- the prose beside the figures --------------------------------------------


def _run_series(run: dict, key: str) -> dict:
    return {int(k): v for k, v in run[key].items()}


#: The framings of the controls' table, in the order the guide prints them, with the
#: wording of their row.
_FRAMING_ROWS = (
    ("context", "the shipped context (60 tokens), then the end-of-text id (the one used here)"),
    ("context:physics_31", "another context of 31 tokens (physics)"),
    ("context:sport_59", "another context of 59 tokens (sport)"),
    ("context:history_123", "another context of 123 tokens (history)"),
    ("eot", "the end-of-text id only"),
    ("none", "nothing before the sentence"),
)


def _prose_phrases(facts: dict):
    """Every phrase the guide's prose must print, built from the facts: ``(in_guide,
    in_readme)``. The assertions inside are the claims the sentences make about the
    data (an ordering, a maximum, an equality), so a sentence cannot outlive the
    numbers that made it true."""
    curve = facts["files"]["live_curve_2d.png"]["facts"]
    run2d, run3d = facts["runs"]["2d"], facts["runs"]["3d"]
    trained = facts["files"]["t_maps_2d_trained.png"]["facts"]
    untrained = facts["files"]["t_maps_2d_untrained.png"]["facts"]
    three = facts["files"]["t_maps_3d_trained.png"]["facts"]
    moran = _run_series(run2d, "mean_morans_i_series")
    sig = {int(k): v for k, v in curve["significant_units_mean_series"].items()}
    ca = {int(k): v for k, v in curve["clusters_a_series"].items()}
    cb = {int(k): v for k, v in curve["clusters_b_series"].items()}
    later = [moran[s] for s in moran if s >= 100]
    last = [moran[s] for s in moran if s >= 250]
    points = list(curve["validation_loss_points_drawn"]["value"].values())
    units = trained["units_per_tap"]["value"]
    taps = trained["num_taps"]["value"]
    comparisons = units * taps
    sig_total = round(trained["mean_significant_units_per_tap"]["value"] * taps)
    rejected = comparisons - sig_total
    need = scipy.stats.t.isf(0.05 / comparisons / 2, 8)
    relaxed = scipy.stats.t.isf(0.05 * sig_total / comparisons / 2, 8)
    counts = run2d["stimuli_framing"]["a"]["text_token_counts"]
    for condition in "abcd":
        assert run2d["stimuli_framing"][condition]["text_token_counts"] == counts
        assert run3d["stimuli_framing"][condition]["text_token_counts"] == counts
    mean_tokens = sum(counts) / len(counts)
    framing2, framing3 = run2d["controls"]["framings"], run3d["controls"]["framings"]

    def cell(framing, key):
        row = framing[key]
        return f"{_g(row['mean_significant_units_per_tap'])} ({row['clusters_a']} / {row['clusters_b']})"

    framing_rows = [
        f"| {label} | {cell(framing2, key)} | {cell(framing3, key)} |" for key, label in _FRAMING_ROWS
    ]
    contexts = [key for key, _ in _FRAMING_ROWS if key.startswith("context")]
    span = lambda framing: [framing[key]["mean_significant_units_per_tap"] for key in contexts]
    moran_framings = {
        "2D": [row["mean_morans_i"] for row in framing2.values()],
        "3D": [row["mean_morans_i"] for row in framing3.values()],
    }
    sig_framings = {
        "2D": [row["mean_significant_units_per_tap"] for row in framing2.values()],
        "3D": [row["mean_significant_units_per_tap"] for row in framing3.values()],
    }
    shuffle2, shuffle3 = (run["controls"]["label_shuffle_null"] for run in (run2d, run3d))
    length2, length3 = (run["controls"]["length_control"] for run in (run2d, run3d))
    for shuffle in (shuffle2, shuffle3):
        top = shuffle["top"]
        assert top[0]["mean_significant_units_per_tap"] == top[1]["mean_significant_units_per_tap"] == shuffle["real"]["mean_significant_units_per_tap"]
        assert top[2]["mean_significant_units_per_tap"] < top[1]["mean_significant_units_per_tap"]
        assert shuffle["n_strictly_above"] == 0 and shuffle["n_at_or_above"] == 2
    best2, best3 = (shuffle["top"][2]["mean_significant_units_per_tap"] for shuffle in (shuffle2, shuffle3))
    real2, real3 = (shuffle["real"]["mean_significant_units_per_tap"] for shuffle in (shuffle2, shuffle3))
    five = [
        length[key]["max"] for length in (length2, length3) for key in ("b_vocab_5v5", "a_vocab_5v5")
    ]
    three_v_seven = [
        length[key] for length in (length2, length3) for key in ("b_vocab_3v7", "a_vocab_3v7")
    ]
    per_tap3 = three["per_tap"]
    sig3 = {name: v["significant_units"]["value"] for name, v in per_tap3.items()}
    most3, fewest3 = max(sig3, key=sig3.get), min(sig3, key=sig3.get)
    moran3 = {name: v["morans_i"]["value"] for name, v in per_tap3.items()}
    assert min(moran3, key=moran3.get) == fewest3
    assert per_tap3[fewest3]["clusters_a"]["value"] == per_tap3[fewest3]["clusters_b"]["value"] == 0
    per_tap2 = trained["per_tap"]
    moran2 = {name: v["morans_i"]["value"] for name, v in per_tap2.items()}
    low2, high2 = min(moran2, key=moran2.get), max(moran2, key=moran2.get)

    def attention_against_feed_forward(per_tap):
        pairs = []
        for block in range(4):
            attention = per_tap[f"block_{block}/attention_tap"]["morans_i"]
            feed_forward = per_tap[f"block_{block}/mlp_tap"]["morans_i"]
            assert attention["value"] > feed_forward["value"], "the sentence says attention is smoother in every block"
            pairs.append(f"{attention['fmt']} against {feed_forward['fmt']}")
        return ", ".join(pairs)

    def percent(count, whole):
        return round(100 * count / whole)

    islands = {}
    for label, facts_of in (("2D", trained), ("3D", three)):
        assert facts_of["taps_with_islands_morans_i_defined"]["value"] == facts_of["num_taps"]["value"]
        islands[label] = (
            f"{facts_of['taps_with_islands_morans_i_defined']['fmt']} of the {facts_of['num_taps']['fmt']} {label} taps, between "
            f"{facts_of['morans_i_islands_min_over_taps']['fmt']} and {facts_of['morans_i_islands_max_over_taps']['fmt']}"
        )
    contexts2, contexts3 = span(framing2), span(framing3)
    in_guide = [
        # the short version
        f"{trained['mean_significant_units_per_tap']['fmt']} of the {units} units per tap pass the significance test "
        f"({trained['significant_units_min_over_taps']['fmt']} to {trained['significant_units_max_over_taps']['fmt']} depending on the tap)",
        f"{trained['total_clusters_a']['fmt']} clusters that prefer the sentences and {trained['total_clusters_b']['fmt']} that prefer the word lists",
        f"in the 3D run {three['mean_significant_units_per_tap']['fmt']} of {three['units_per_tap']['fmt']} units per tap pass, in "
        f"{three['total_clusters_a']['fmt']} and {three['total_clusters_b']['fmt']} clusters",
        f"every one except the real one and its mirror gives at most {_g(best2)} significant units per tap in 2D and {_g(best3)} in 3D",
        f"3D drops to {_g(framing3['eot']['mean_significant_units_per_tap'])} or {_g(framing3['none']['mean_significant_units_per_tap'])} significant units per tap",
        # the framing and the matching
        f"{', '.join(str(c) for c in counts[:-1])} and {counts[-1]} for i = 0 to 4 (mean {mean_tokens:.1f} in every condition)",
        *framing_rows,
        f"between {min(contexts2):.0f} and {max(contexts2):.0f} significant units per tap in 2D and between "
        f"{min(contexts3):.0f} and {max(contexts3):.0f} in 3D",
        f"the end-of-text id alone ({_g(framing2['eot']['mean_significant_units_per_tap'])}) is close to the lowest passage "
        f"({_g(min(contexts2))}) and only the bare sentence at the start of the window loses nearly everything "
        f"({_g(framing2['none']['mean_significant_units_per_tap'])})",
        f"2D {min(moran_framings['2D']):.3f} to {max(moran_framings['2D']):.3f}, 3D {min(moran_framings['3D']):.3f} to {max(moran_framings['3D']):.3f}",
        f"{_g(real3)} here and {three['mean_significant_units_per_tap']['fmt']} in the figure caption",
        # reading the 2D map
        f"of the {comparisons} units ({taps} taps of {units}), {sig_total} are significant "
        f"({trained['mean_significant_units_per_tap']['fmt']} per tap) and {rejected} are not",
        f"{rejected} of {comparisons} cells with a stroke in the trained figure",
        f"over {comparisons} comparisons",
        f"need a |t| of about {need:.1f} to pass",
        f"(about {relaxed:.1f} when {sig_total} of the {comparisons} pass)",
        f"Step 0 is drawn on +/-{untrained['colour_limit']['fmt']} and step 400 on +/-{trained['colour_limit']['fmt']}",
        f"(at step 400 the largest |t| is "
        f"{max(abs(trained['t_min_over_taps']['value']), abs(trained['t_max_over_taps']['value'])):.2f} "
        f"on a scale of +/-{trained['colour_limit']['fmt']})",
        f"(the mean Moran's I went from {untrained['mean_morans_i']['fmt']} to {trained['mean_morans_i']['fmt']})",
        # the 3D map
        f"{three['mean_significant_units_per_tap']['fmt']} of {three['units_per_tap']['fmt']} units per tap are significant on average",
        f"(about {percent(three['mean_significant_units_per_tap']['value'], three['units_per_tap']['value'])} per cent)",
        f"{three['significant_units_min_over_taps']['fmt']} to {three['significant_units_max_over_taps']['fmt']} depending on the tap",
        f"the most are in `{most3}` ({sig3[most3]}), the fewest in `{fewest3}` ({sig3[fewest3]}, which also has the "
        f"lowest Moran's I, {per_tap3[fewest3]['morans_i']['fmt']}, and no cluster at all)",
        # Moran's I
        f"the lowest tap is `{low2}` ({per_tap2[low2]['morans_i']['fmt']}) and the highest `{high2}` ({per_tap2[high2]['morans_i']['fmt']})",
        f"in every one of the four blocks the attention tap is smoother than the feed-forward tap of the same block "
        f"({attention_against_feed_forward(per_tap2)})",
        f"and in the 3D run too ({attention_against_feed_forward(per_tap3)})",
        f"(2D median {shuffle2['morans_i']['median']:.3f}, the real labelling {trained['mean_morans_i']['fmt']}; "
        f"3D median {shuffle3['morans_i']['median']:.3f}, the real labelling {three['mean_morans_i']['fmt']})",
        f"defined in all {islands['2D']}, and in all {islands['3D']}",
        # the label-shuffle table
        f"| significant units per tap, real labelling | {_g(real2)} | {_g(real3)} |",
        f"| shuffled: median, 95th percentile | {_g(shuffle2['sig']['median'])}, {_g(shuffle2['sig']['p95'])} | "
        f"{_g(shuffle3['sig']['median'])}, {_g(shuffle3['sig']['p95'])} |",
        f"| the best other labelling (not the real one or its mirror) | {_g(best2)} | {_g(best3)} |",
        f"| labellings with at least as many as the real one (itself and its mirror included) | {shuffle2['n_at_or_above']} | "
        f"{shuffle3['n_at_or_above']} |",
        f"| labellings with at least one cluster (the real one and its mirror included) | {shuffle2['n_splits_with_a_cluster']} | "
        f"{shuffle3['n_splits_with_a_cluster']} |",
        f"| Moran's I, real labelling | {trained['mean_morans_i']['fmt']} | {three['mean_morans_i']['fmt']} |",
        f"| shuffled: median (minimum to maximum) | {shuffle2['morans_i']['median']:.3f} ({shuffle2['morans_i']['min']:.3f} to "
        f"{shuffle2['morans_i']['max']:.3f}) | {shuffle3['morans_i']['median']:.3f} ({shuffle3['morans_i']['min']:.3f} to "
        f"{shuffle3['morans_i']['max']:.3f}) |",
        f"is {shuffle2['n_at_or_above']} of {shuffle2['n_splits']}, about {shuffle2['n_at_or_above'] / shuffle2['n_splits']:.3f}",
        f"(rank {shuffle2['morans_i']['real_rank_from_top']} and {shuffle3['morans_i']['real_rank_from_top']} from the top out of {shuffle2['n_splits']})",
        f"all {shuffle2['n_splits']} ways to choose 5 of 10",
        # the length-control table
        f"| 3 words against 7 words, `b` vocabulary | {length2['b_vocab_3v7']['median']:.1f} ({length2['b_vocab_3v7']['max']:.1f}) | "
        f"{length3['b_vocab_3v7']['median']:.1f} ({length3['b_vocab_3v7']['max']:.1f}) |",
        f"| 3 words against 7 words, `a` vocabulary | {length2['a_vocab_3v7']['median']:.1f} ({length2['a_vocab_3v7']['max']:.1f}) | "
        f"{length3['a_vocab_3v7']['median']:.1f} ({length3['a_vocab_3v7']['max']:.1f}) |",
        f"| 5 words against 5 words, either vocabulary | {max(length2['b_vocab_5v5']['median'], length2['a_vocab_5v5']['median']):.1f} "
        f"({max(length2['b_vocab_5v5']['max'], length2['a_vocab_5v5']['max']):.1f}) | "
        f"{max(length3['b_vocab_5v5']['median'], length3['a_vocab_5v5']['median']):.1f} "
        f"({max(length3['b_vocab_5v5']['max'], length3['a_vocab_5v5']['max']):.1f}) |",
        f"(at most {max(five):.1f} in any of the {length2['b_vocab_3v7']['draws']} draws)",
        f"a median of at most {max(row['median'] for row in three_v_seven):.1f} and a single draw as large as "
        f"{max(row['max'] for row in three_v_seven):.1f}",
        f"{percent(length2['a_vocab_3v7']['max'], real2)} to {percent(length2['b_vocab_3v7']['max'], real2)} per cent of the real count in 2D "
        f"({length2['a_vocab_3v7']['max']:.1f} and {length2['b_vocab_3v7']['max']:.1f} of {_g(real2)}) and "
        f"{percent(length3['b_vocab_3v7']['max'], real3)} to {percent(length3['a_vocab_3v7']['max'], real3)} per cent in 3D "
        f"({length3['b_vocab_3v7']['max']:.1f} and {length3['a_vocab_3v7']['max']:.1f} of {_g(real3)})",
        # the live curve, panel by panel
        f"from {curve['mean_morans_i_first_step']['fmt']} at step 0 to {moran[50]:.3f} at step 50",
        f"peaks at {curve['mean_morans_i_peak']['fmt']} at step {curve['mean_morans_i_peak_step']['fmt']}",
        f"between {min(later):.3f} and {max(later):.3f} from step 100 on",
        f"({min(last):.3f} to {max(last):.3f} from step 250 on)",
        f"the taps end between {trained['morans_i_min_over_taps']['fmt']} and {trained['morans_i_max_over_taps']['fmt']}",
        f"none at steps 0 and 50, then {ca[100]} and {cb[100]} at step 100, and {ca[400]} and {cb[400]} at step 400",
        f"{sig[0]} at steps 0 and 50, {sig[100]} at step 100, {sig[150]} at step 150, then {sig[200]}, {sig[250]}, {sig[300]}, "
        f"{sig[350]} and {sig[400]} at step 400",
        f"it was flat from step 200 to 250 ({sig[200]} and {sig[250]})",
        ", ".join(f"{v:.3f}" for v in points),
        # the conclusions' scale
        f"Moran's I rises from about 0 to {trained['mean_morans_i']['fmt']} (2D) and {three['mean_morans_i']['fmt']} (3D)",
        f"{trained['mean_significant_units_per_tap']['fmt']} of 256 units per tap in 2D and {three['mean_significant_units_per_tap']['fmt']} of 343 in 3D",
        f"from {min(sig_framings['3D']):.1f} to {max(sig_framings['3D']):.1f} with the framing; in 2D from "
        f"{min(sig_framings['2D']):.1f} to {max(sig_framings['2D']):.1f}",
    ]
    # The run without the penalty (alpha = 0), the fourth control.
    run2c, run3c = facts["runs"]["2d_alpha0"], facts["runs"]["3d_alpha0"]
    assert run2c["config"]["spatial_alpha"] == run3c["config"]["spatial_alpha"] == 0.0
    assert run2d["config"]["spatial_alpha"] == run3d["config"]["spatial_alpha"] == 2.5
    for with_penalty, without in ((run2d, run2c), (run3d, run3c)):
        assert without["config"]["seed"] == with_penalty["config"]["seed"]
        assert without["last_live_row_mean_morans_i"] < 0.1 < 0.5 < with_penalty["last_live_row_mean_morans_i"], "smoothness is the penalty's"
    sig_mean = lambda run: run["significant_units_total"] / 8
    assert sig_mean(run2c) > sig_mean(run2d) and sig_mean(run3c) > 0.9 * sig_mean(run3d), "the count is not the penalty's"
    morans_end = lambda run: run["last_live_row_mean_morans_i"]
    clusters_end = lambda run: run["clusters_ab_series"]["400"]
    val_end = lambda run: run["training"]["final_val_loss"]
    share = lambda run: run["share_of_significant_units_in_clusters"]
    cells2, cells3 = trained["units_per_tap"]["value"], three["units_per_tap"]["value"]
    columns = (run2d, run2c, run3d, run3c)
    assert share(run2d) > share(run2c) and share(run3d) > share(run3c)
    assert val_end(run2c) < val_end(run2d) and val_end(run3c) < val_end(run3d)
    in_guide += [
        f"significant units, {_g(sig_mean(run2c))} per tap in 2D and {_g(sig_mean(run3c))} in 3D",
        f"(Moran's I {morans_end(run2c):.3f} and {morans_end(run3c):.3f} against {morans_end(run2d):.3f} and {morans_end(run3d):.3f})",
        "| mean Moran's I | " + " | ".join(f"{morans_end(run):.3f}" for run in columns) + " |",
        "| significant units per tap | " + " | ".join(_g(sig_mean(run)) for run in columns) + " |",
        "| clusters of `a` / `b` | " + " | ".join(f"{clusters_end(run)[0]} / {clusters_end(run)[1]}" for run in columns) + " |",
        "| share of the significant units that sit in a cluster | " + " | ".join(f"{share(run):.3f}" for run in columns) + " |",
        "| validation loss | " + " | ".join(f"{val_end(run):.3f}" for run in columns) + " |",
        f"({_g(sig_mean(run2c))} against {_g(sig_mean(run2d))}, {percent(sig_mean(run2c), cells2)} per cent of the cells)",
        f"({_g(sig_mean(run3c))} against {_g(sig_mean(run3d))})",
        f"when {percent(sig_mean(run2c), cells2)} per cent (2D) or {percent(sig_mean(run3c), cells3)} per cent (3D) of the cells are significant",
        f"({clusters_end(run3d)[0]} and {clusters_end(run3d)[1]} against {clusters_end(run3c)[0]} and {clusters_end(run3c)[1]})",
        f"({share(run2d) * 100:.1f} per cent of the significant units in the 2D map with the penalty, {share(run2c) * 100:.1f} per cent without; "
        f"{share(run3d) * 100:.1f} and {share(run3c) * 100:.1f} per cent in 3D)",
        f"the validation loss after 400 steps is {val_end(run2d):.3f} with it and {val_end(run2c):.3f} without in 2D, "
        f"{val_end(run3d):.3f} and {val_end(run3c):.3f} in 3D "
        f"({val_end(run2d) - val_end(run2c):.3f} and {val_end(run3d) - val_end(run3c):.3f} nats",
        f"the run without it has {run2c['mean_significant_units_series']['50']} significant units per tap and "
        f"{run2c['clusters_ab_series']['50'][0]} and {run2c['clusters_ab_series']['50'][1]} clusters at step 50",
        f"has as many significant units ({_g(sig_mean(run2c))} per tap in 2D, {_g(sig_mean(run3c))} in 3D)",
        f"(Moran's I {morans_end(run2d):.3f} against {morans_end(run2c):.3f}, {morans_end(run3d):.3f} against {morans_end(run3c):.3f})",
    ]
    assert sig[0] == sig[50] == 0 and sig[100] > 0 and sig[200] > sig[250], "the curve text describes this shape"
    assert ca[0] == ca[50] == cb[0] == cb[50] == 0 and ca[100] > 0 and cb[100] > 0
    # Outside the guide: the sections on watching a run and on training a 3D variant.
    s3 = _run_series(run3d, "mean_morans_i_series")
    s3sig = _run_series(run3d, "mean_significant_units_series")
    w0, h0 = (facts["files"]["t_maps_2d_untrained.png"][k] for k in ("width", "height"))
    w4, h4 = (facts["files"]["t_maps_2d_trained.png"][k] for k in ("width", "height"))
    from_200 = [v for step, v in s3sig.items() if step >= 200]
    val3 = run3d["val_loss_points_in_metrics_jsonl"]
    in_readme = [
        f"on +/-{untrained['colour_limit']['fmt']} at step 0 and +/-{trained['colour_limit']['fmt']} at step 400",
        f"went from {w0}x{h0} at step 0 to {w4}x{h4} at step 400",
        # the 3D training bullet
        f"from {s3[0]:.4f} at step 0 to {s3[50]:.3f} at step 50 and {s3[400]:.3f} at step 400",
        f"{s3sig[50]:.1f} at step 50 and {s3sig[100]:.3f} at step 100, then {s3sig[150]:.1f} at step 150 and between "
        f"{min(from_200):.1f} and {max(from_200):.1f} from step 200 on",
        f"({three['total_clusters_a']['fmt']} clusters for `a` and {three['total_clusters_b']['fmt']} for `b` at step 400)",
        f"words give a median of {length3['b_vocab_3v7']['median']:.1f} significant units per tap on this checkpoint",
        f"are the only two of {shuffle3['n_splits']} that give more than {_g(best3)} significant units per tap",
        # the lag of the last full row behind the final history value
        f"in the 400-step `tiny3d` example run (validation every 100 steps, a draw every 50) the step-400 row held "
        f"{val3['300']:.4f} (the validation of step 300) and the final history value was {run3d['training']['final_val_loss']:.4f}",
        f"in the matching `tiny` run {points_raw(curve, 300):.4f} and {points_raw(curve, 400):.4f}",
        # the cost of a draw
        f"median {run2d['seconds_per_draw_median']['fmt']} s ({run2d['seconds_per_draw_min']['fmt']} to {run2d['seconds_per_draw_max']['fmt']}; "
        f"{run2d['seconds_per_draw_sum']['fmt']} s in all, in a run of {run2d['total_wall_s']['fmt']} s)",
        f"median {run3d['seconds_per_draw_median']['fmt']} s ({run3d['seconds_per_draw_min']['fmt']} to {run3d['seconds_per_draw_max']['fmt']}; "
        f"{run3d['seconds_per_draw_sum']['fmt']} s in all, in a run of {run3d['total_wall_s']['fmt']} s)",
        f"{run2d['peak_own_gpu_mib_nvidia_smi_1s_samples']['value']:,} MiB (`tiny`) and {run3d['peak_own_gpu_mib_nvidia_smi_1s_samples']['value']:,} MiB (`tiny3d`)",
    ]
    return in_guide, in_readme


def points_raw(curve: dict, step: int) -> float:
    """The validation loss the live curve drew at ``step`` (full precision)."""
    return curve["validation_loss_points_drawn"]["value"][str(step)]


def _prose_failures(guide: str, readme: str, facts: dict):
    in_guide, in_readme = _prose_phrases(facts)
    return [p for p in in_guide if _squash(p) not in guide] + [p for p in in_readme if _squash(p) not in readme]


def test_the_prose_beside_the_figures_quotes_the_facts_too():
    """Numbers that sit in the guide's prose rather than in a caption: the controls'
    tables, the framing table, the live curve bullets, the colour limits, the frame
    sizes, the cost of a draw. Every phrase is built from the facts file."""
    absent = _prose_failures(_squash(_guide_text()), _squash(_readme_text()), _facts())
    assert absent == [], f"the prose does not say {absent}"


def test_the_controls_say_what_the_numbers_mean():
    """The words of the controls hold in the data: the real labelling and its mirror
    are the top two of the 252 and every other labelling gives far less, Moran's I of
    the real labelling lies inside the shuffled range in both dimensions, a pure length
    difference gives a median far below the real count in both, the equal-length
    calibration gives nothing, and the framing table shows the dependence on the passage
    in 3D."""
    runs = _facts()["runs"]
    for run in ("2d", "3d"):
        controls = runs[run]["controls"]
        shuffle = controls["label_shuffle_null"]
        real = shuffle["real"]["mean_significant_units_per_tap"]
        assert shuffle["n_splits"] == 252 and shuffle["n_at_or_above"] == 2 and shuffle["n_strictly_above"] == 0
        assert shuffle["sig"]["median"] == 0.0 and shuffle["sig"]["p95"] == 0.0
        others = [entry["mean_significant_units_per_tap"] for entry in shuffle["top"][2:]]
        assert max(others) < real / 4, run
        moran = shuffle["morans_i"]
        assert moran["min"] <= moran["real"] <= moran["max"], run
        observed = runs[run]["last_live_row_mean_morans_i"]
        assert abs(observed - moran["real"]) < 1e-4, "the live run and the CPU rescoring agree on Moran's I"
        length = controls["length_control"]
        for key in ("b_vocab_3v7", "a_vocab_3v7"):
            assert length[key]["median"] < real / 10, (run, key)
            assert length[key]["max"] < real / 2, (run, key)
        for key in ("b_vocab_5v5", "a_vocab_5v5"):
            assert length[key]["median"] == 0.0 and length[key]["max"] < 3, (run, key)
    framing3 = runs["3d"]["controls"]["framings"]
    contexts = [row["mean_significant_units_per_tap"] for key, row in framing3.items() if key.startswith("context")]
    assert min(contexts) > 10 * max(framing3["eot"]["mean_significant_units_per_tap"], framing3["none"]["mean_significant_units_per_tap"])


def test_a_crosshair_index_the_guide_prints_exists_on_the_grid_it_describes():
    """A 3D index runs from 0 to (axis length - 1); the guide once printed
    ``@(3, 4, 9)`` on a 7x7x7 grid."""
    grid = _facts()["files"]["t_maps_3d_trained.png"]["facts"]["grid_shape"]["value"]
    found = re.findall(r"@\((\d+), (\d+), (\d+)\)", _guide_text())
    assert found, "the guide has no crosshair title example"
    for triple in found:
        for index, length in zip(map(int, triple), grid):
            assert index < length, f"@{tuple(map(int, triple))} is outside a {grid} grid"
    # And the panel title's own plane index equals the crosshair coordinate on that axis.
    for axis, plane, x, y, z in re.findall(r"`axis (\d) = (\d)  @\((\d+), (\d+), (\d+)\)`", _guide_text()):
        assert (x, y, z)[int(axis)] == plane


#: A sentence a review found false or a drawing that no longer exists; a README that
#: says it again is wrong. Each is a literal of one specific retraction.
_RETRACTED_SENTENCES = (
    "0.249",                                   # the 3D islands maximum; the run says 0.972
    "@(3, 4, 9)",                              # index 9 does not exist on a 7x7x7 grid
    "contourf",                                # the hatch is one box per rejected cell
    "averaged over all positions",             # the pooling is over the text tokens
    "does not leave the padding out",
    "averaged over the positions of the input window",
    "mean-pooled over token positions",
    "Pooling is a **mean** over token positions",
    "most cells passed",                       # the trained 2D map is hatched everywhere
    "large patches of one\ncolour",            # (the earlier caption of the trained map)
    "187.9 of 256",                            # old-run caption numbers, see _OLDER_RUN_NUMBERS
    "142 to 206",
    "-17.3 to 19.1",
    "fell from 191 at step 50",
    "0.555 at step 50",
    "7.5333",
    "10.0 text tokens on average against 7.8",  # CLS and SEP counted as text (step 8.5)
    "10.0 text tokens on average, `b` 7.8",
    "In 2D the artefact is nearly gone",       # the length control says otherwise
    "the map to trust least on length",        # one ranking of 3D against 2D; none is made
    "a 3D map is the less trustworthy of the two on length",
    "read the plateau at the end",
    "Two taps barely show any island",
    "largest value over the 8 taps is 0.172",
    "ticks stop at 6",                         # the colour bar of the earlier trained map
    # True of the example runs on the previous model, false of the corrected one's:
    "it covers every cell, so nothing in this 2D example is significant",
    "The map is smooth, and it is not significant",
    "Every cell is hatched: none is significant",
    "Neither example run wrote one",
    "no unit passes the\nsignificance test",
    "a smooth map whose units show no detected difference between these\n  stimuli. The curve cannot",
    "the prefix is a choice",                  # the framing is a passage now, not a prefix
    # Review pass 6, concern 5: losses over 100,277 and over 50,304 output rows are not
    # comparable, so no claim about learning speed may stand on them.
    "the new model also learns faster",
    "validation loss 7.16 after 400 steps against about\n7.8 before",
    "so it may simply be further along",
    "also reaches a lower validation loss in the same 400 steps",
    "7.162 against 7.818",
    "7.105 against 7.492",
    "the corrected model trains faster",
)


@pytest.mark.parametrize("readme", _READMES, ids=lambda p: p.parent.name)
def test_no_retracted_sentence_is_back(readme):
    text = readme.read_text(encoding="utf-8")
    squashed = _squash(text)
    assert [s for s in _RETRACTED_SENTENCES if _squash(s) in squashed] == []


#: What the guide must now say, each a sentence the review asked for.
_REQUIRED_SENTENCES = (
    "averaged over the text tokens of the sentence",                 # the pooling (D-020)
    "The framing is a choice, and it moves the result",              # the framing (D-027)
    "matched in length, in tokens",                                  # the stimuli (D-025)
    "learned **absolute** position embeddings",
    "not free of length and position effects",
    "The map is smooth, and part of it is significant",              # the 2D example
    "Look at the unhatched islands of the trained map",
    "a hatched cell did not pass",                                   # literally true since D-021
    "only tell clusters apart",                                      # the cluster colours
    "do not encode the condition",
    "Both example runs wrote one this time",                         # the cluster figure
    "can fall as well as rise",                                      # the significant-units line
    "never as \"the map separates a from b\"",                       # what Moran's I is
    "label-shuffle null",
    "length control",
    "No claim is made that 3D is better or worse than 2D",
    "A real selectivity claim needs a larger stimulus set",
    "A second correction: earlier t-maps were averaged over the padding",
    "A third correction: the stimuli framing and their lengths changed what the t-maps show",
    "But it is not made by the topographic penalty",                # the alpha = 0 control
    "The penalty is **not** what makes units respond differently to sentences and to word lists",
    "They do not show that the topographic penalty made that selectivity",
    "A cluster count is therefore a weak diagnostic here",
    "A fourth correction: the sentence is placed after a fixed passage",
    "A fifth correction: the model was not the model the paper describes",
    "No ablation separates the five differences, so which one mattered is not known",
    # review pass 6, concern 5: the vocabulary is one of the changes, the losses do not compare
    "and an output vocabulary of 100,277 rows instead of 50,304",
    "The validation losses of the earlier and the corrected runs are not comparable",
    "so nothing here says which model learns faster",
    "no claim about learning speed is made",
)


def test_the_guide_says_what_the_review_asked_it_to_say():
    guide = _squash(_readme_text())
    assert [s for s in _REQUIRED_SENTENCES if _squash(s) not in guide] == []


#: Phrases that are false for the examples shown but true of an EARLIER run (the
#: previous model), so they may stand only inside a unit (bullet, table, paragraph)
#: that carries an older-run label.
_ONLY_WITH_A_LABEL = (
    "almost no selectivity",
    "no selectivity at this scale",
)

#: Numbers of the EARLIER example runs (the padded average, the CLS framing, the
#: unmatched lengths) or of controls measured on them. They stay in the README only
#: labelled as such, and each is listed here with where it was measured, so none is
#: mistaken for a number of the figures shown.
_OLDER_RUN_NUMBERS = {
    "187.9": "pooling-fix measurements, mean over 8 taps, 2D old (padded average)",
    "36.5": "same table, 2D new (text tokens), on the earlier checkpoint",
    "72.9": "same table, 3D old",
    "92.8": "same table, 3D new, on the earlier checkpoint",
    "202.9": "dotted control, 2D, old pooling",
    "10.9": "dotted control, 2D, new pooling",
    "1.034": "islands Moran's I of the earlier 2D example, block_0/attention_tap",
    "38.9": "the previous 2D example (CLS framing, unmatched lengths), significant units per tap",
    "89.2": "the previous 3D example, same",
    "38.88": "framing measurement, earlier 2D checkpoint, CLS framing",
    "89.0": "framing measurement, earlier 3D checkpoint, CLS framing",
    "51.12": "framing measurement, earlier 3D checkpoint, end-of-text framing, unmatched",
    "5.25": "length-matched smoke set on the earlier 3D checkpoint",
    "0.786": "Moran's I of the previous 2D example",
    "0.711": "Moran's I of the previous 3D example",
    "0.785": "Moran's I of the example run on the previous model, 2D (fifth correction)",
    "0.670": "the same, 3D",
    "42.4": "length-control median, 3 against 7 words, `b` vocabulary, previous model, 2D",
    "55.6": "the same, `a` vocabulary, 2D",
    "214.4": "the same, `b` vocabulary, previous model, 3D",
}
#: Whole words only (step 9.1): ``old`` used to match inside ``hold``, ``holds`` and
#: ``threshold`` and let an unlabelled number through.
_LABELS = re.compile(r"\b(?:padded|earlier|previous|older|old|before)\b", re.I)

#: A line that opens a new bullet or numbered item. A bullet list is ONE blank-line
#: paragraph, so labelling per paragraph let one bullet's "earlier run" cover
#: another bullet's number. A table is one unit: its header row labels its columns.
_BULLET_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")
_TABLE_ROW = re.compile(r"^\s*\|")


def _label_units(text: str):
    """The stretches a label may cover, each squashed to one line: a bullet or
    numbered item with its continuation lines, a whole table, or a paragraph's
    plain lines. A label counts for the unit it is written in and for no other."""
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        current, in_table = [], False
        for line in paragraph.splitlines():
            row = bool(_TABLE_ROW.match(line))
            if current and (_BULLET_START.match(line) or row != in_table):
                units.append(_squash(" ".join(current)))
                current = []
            in_table = row
            current.append(line)
        if current:
            units.append(_squash(" ".join(current)))
    return units


def _unlabelled_older_numbers(text: str, numbers=_OLDER_RUN_NUMBERS):
    """Numbers of ``numbers`` that some unit quotes without a label of its own.

    Raises ``AssertionError`` for a number no unit quotes at all (the table has gone
    stale), so a vanished figure is not read as a clean pass."""
    units = _label_units(text)
    unlabelled = []
    for number in numbers:
        holders = [u for u in units if re.search(rf"(?<![\d.]){re.escape(number)}(?!\d)", u)]
        assert holders, f"{number} ({numbers[number]}) is no longer quoted; drop it from the table"
        unlabelled += [number for u in holders if not _LABELS.search(u)]
    return unlabelled


def _unlabelled_mentions(text: str, phrases=_ONLY_WITH_A_LABEL):
    """Phrases of ``phrases`` that some unit of ``text`` says without an older-run label."""
    return [
        phrase
        for unit in _label_units(text)
        for phrase in phrases
        if phrase.lower() in unit.lower() and not _LABELS.search(unit)
    ]


def test_numbers_of_the_earlier_runs_are_kept_only_with_their_label():
    assert _unlabelled_older_numbers(_readme_text()) == []


def test_no_selectivity_is_not_said_of_the_current_examples():
    assert _unlabelled_mentions(_readme_text()) == []


_TWO = {"36.5": "t", "92.8": "t"}


class TestTheLabelGuardCanFail:
    """Step 9.1: the guard of the test above, seen RED on texts built to beat it."""

    def test_old_inside_another_word_is_not_a_label(self):
        text = "- the figure holds 36.5 significant units\n- the threshold gives 92.8\n"
        assert _unlabelled_older_numbers(text, _TWO) == ["36.5", "92.8"]

    def test_a_label_in_one_bullet_does_not_cover_another_bullet(self):
        text = "- 36.5 significant units are shown\n- the earlier run had 92.8\n"
        assert _unlabelled_older_numbers(text, _TWO) == ["36.5"]

    def test_a_tables_header_label_covers_its_rows_but_not_the_text_after_it(self):
        text = (
            "| run | earlier units |\n| 2D | 36.5 |\n| 3D | 92.8 |\n"
            "- but now 36.5 is quoted bare\n"
        )
        assert _unlabelled_older_numbers(text, _TWO) == ["36.5"]

    def test_a_table_with_no_label_anywhere_is_flagged(self):
        text = "| run | units |\n| 2D | 36.5 |\n| 3D | 92.8 |\n"
        assert _unlabelled_older_numbers(text, _TWO) == ["36.5", "92.8"]

    def test_a_label_in_the_same_bullet_or_paragraph_is_enough(self):
        text = (
            "- the earlier run had 36.5\n  and, on the previous checkpoint, 92.8\n\n"
            "The old padded average gave 36.5 and\nthe same table has 92.8.\n"
        )
        assert _unlabelled_older_numbers(text, _TWO) == []

    def test_a_number_nobody_quotes_is_reported_not_passed(self):
        with pytest.raises(AssertionError, match="no longer quoted"):
            _unlabelled_older_numbers("- the earlier run had 36.5\n", _TWO)

    def test_the_real_readme_with_an_unlabelled_bullet_slipped_in_is_flagged(self):
        """The README itself, mutated: a two-bullet list is appended whose first
        bullet is labelled and whose second quotes an older-run number bare. The old
        paragraph-level check read the neighbour's label as its own. (The README has
        no such list now, so the list is added to it; a guard that passes the real
        text but misses this one would be blind to a bullet someone adds later.)"""
        text = _readme_text() + (
            "\n\n- the earlier run had 187.9 significant units\n"
            "- the guide reports 72.9 significant units\n"
        )
        flagged = _unlabelled_older_numbers(text)
        assert flagged == ["72.9"], flagged

    def test_an_unlabelled_older_number_on_a_plain_line_of_the_real_readme_is_flagged(self):
        """Step 8.5: the new older-run numbers (38.9, 89.2 ...) are guarded too: one
        of them quoted bare, in a paragraph of its own, is flagged."""
        text = _readme_text() + "\n\nThe 2D map has 38.9 significant units per tap.\n"
        assert "38.9" in _unlabelled_older_numbers(text)


class TestTheReadmeGuardsCanFail:
    """Step 8.5: each guard of the README tests, seen RED on a mutated copy."""

    def test_two_swapped_images_are_named(self, tmp_path):
        figures = _TRAINER_README.parent / "figures"
        for path in figures.glob("*.png"):
            (tmp_path / path.name).write_bytes(path.read_bytes())
        first, second = "t_maps_2d_untrained.png", "t_maps_2d_trained.png"
        (tmp_path / first).write_bytes((figures / second).read_bytes())
        (tmp_path / second).write_bytes((figures / first).read_bytes())
        assert _pin_failures(tmp_path, _shipped()) == [first, second]
        # ... and the unmodified copy is clean, so the failure above is the swap.
        (tmp_path / first).write_bytes((figures / first).read_bytes())
        (tmp_path / second).write_bytes((figures / second).read_bytes())
        assert _pin_failures(tmp_path, _shipped()) == []

    def test_an_extra_png_nobody_describes_is_named(self, tmp_path):
        figures = _TRAINER_README.parent / "figures"
        for path in figures.glob("*.png"):
            (tmp_path / path.name).write_bytes(path.read_bytes())
        (tmp_path / "clusters_2d.png").write_bytes(_PNG_SIGNATURE)
        assert _pin_failures(tmp_path, _shipped()) == ["clusters_2d.png"]

    def test_a_changed_caption_number_is_named(self):
        guide = _guide_text()
        assert _caption_failures("t_maps_2d_trained.png", guide) == []
        trained = _caption("t_maps_2d_trained.png", guide)
        assert "Mean Moran's I 0.771" in trained
        mutated = guide.replace("Mean Moran's I 0.771", "Mean Moran's I 0.781", 1)
        failures = _caption_failures("t_maps_2d_trained.png", mutated)
        assert any("0.771" in failure for failure in failures), failures

    def test_a_changed_prose_number_is_named(self):
        guide, readme = _squash(_guide_text()), _squash(_readme_text())
        assert _prose_failures(guide, readme, _facts()) == []
        mutated = guide.replace("| 3 words against 7 words, `b` vocabulary | 2.5 (28.9) |", "| 3 words against 7 words, `b` vocabulary | 0.5 (28.9) |")
        assert mutated != guide
        failures = _prose_failures(mutated, readme, _facts())
        assert any("2.5 (28.9)" in failure for failure in failures), failures
        # and a changed number in the control table of the run without the penalty
        mutated = guide.replace("| mean Moran's I | 0.771 | -0.019 |", "| mean Moran's I | 0.771 | 0.019 |")
        assert mutated != guide
        assert any("-0.019" in failure for failure in _prose_failures(mutated, readme, _facts()))

    def test_a_reinserted_retracted_claim_about_the_examples_is_flagged(self):
        text = _readme_text()
        assert _unlabelled_mentions(text) == []
        anchor = "**Look at the unhatched islands of the trained map"
        assert anchor in text
        mutated = text.replace(
            anchor,
            "The trained 2D map shows no selectivity at this scale.\n\n" + anchor,
            1,
        )
        assert _unlabelled_mentions(mutated) == ["no selectivity at this scale"]
        # ... and the same words inside an earlier-run sentence are allowed.
        labelled = text.replace(
            anchor,
            "The earlier example read the 2D map as showing no selectivity at this scale.\n\n" + anchor,
            1,
        )
        assert _unlabelled_mentions(labelled) == []
        # ... and a retracted sentence about the examples is caught as a whole.
        squashed = _squash(text + "\nEvery cell is hatched: none is significant.\n")
        assert [r for r in _RETRACTED_SENTENCES if _squash(r) in squashed] == ["Every cell is hatched: none is significant"]

    def test_a_reinserted_retracted_sentence_is_flagged(self):
        squashed = _squash(_readme_text() + "\nIn 2D the artefact is nearly gone.\n")
        assert [s for s in _RETRACTED_SENTENCES if _squash(s) in squashed] == ["In 2D the artefact is nearly gone"]

    def test_an_illustration_caption_without_the_word_synthetic_is_flagged(self):
        guide, facts = _guide_text(), _facts()
        assert _illustration_failures(guide, facts) == []
        caption = _caption("t_maps_2d_illustration.png", guide)
        assert caption.startswith("*SYNTHETIC illustration, made-up numbers, not a run.")
        mutated = guide.replace("*SYNTHETIC illustration, made-up numbers, not a run.", "*Illustration.", 1)
        assert "caption" in _illustration_failures(mutated, facts)
        # The word must also be in the facts: a figure that is not marked there fails.
        unmarked = json.loads(json.dumps(facts))
        unmarked["files"]["t_maps_2d_illustration.png"]["kind"] = "run"
        assert "facts kind" in _illustration_failures(guide, unmarked)
        # ... and in the alt text.
        no_alt = guide.replace("the title says SYNTHETIC", "the title says so", 1)
        assert "alt text" in _illustration_failures(no_alt, facts)


def test_the_uniform_guess_arithmetic_of_the_vocabulary_caveat_is_right():
    """Review pass 6, concern 5: the numbers the caveat quotes are the logarithms it names."""
    import math

    text = _squash(_readme_text())
    assert f"ln 100,277 = {math.log(100277):.2f} nats against ln 50,304 = {math.log(50304):.2f}" in text
    assert f"differs by {math.log(100277) - math.log(50304):.2f} nats" in text
