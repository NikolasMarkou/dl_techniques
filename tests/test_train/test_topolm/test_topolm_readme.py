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


def test_a_figure_no_run_wrote_is_omitted_in_the_facts_and_absent_in_the_guide():
    """No cluster figure was written by either example run: the facts say so, no
    file is shipped and the guide shows none."""
    files = _facts()["files"]
    omitted = sorted(name for name, entry in files.items() if entry.get("omitted"))
    assert omitted == ["clusters_2d.png", "clusters_3d.png"]
    for name in omitted:
        assert files[name]["reason"]
        assert not (_TRAINER_README.parent / "figures" / name).exists()
        assert name not in _readme_text()


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
        "{mean_significant_units_per_tap} significant units per tap ({significant_units_max_over_taps} of {units_per_tap} in every tap)",
        "{total_clusters_a} clusters for `a` and {total_clusters_b} for `b` over the {num_taps} taps",
        "t values run from {t_min_over_taps} to {t_max_over_taps}",
        "+/-{colour_limit}",
        "none is significant",
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
    "t_maps_2d_illustration.png": (
        "SYNTHETIC illustration, made-up numbers, not a run",
        "{grid_shape} maps",
        "|t| is above {threshold_abs_t}",
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
    if "per_tap" in facts:
        context["per_tap_morans_i"] = ", ".join(v["morans_i"]["fmt"] for v in facts["per_tap"].values())
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


def test_the_trained_2d_example_has_no_significant_unit_and_the_guide_says_so():
    """The central fact of the 2D example: smooth and not significant."""
    facts = _facts()["files"]["t_maps_2d_trained.png"]["facts"]
    assert facts["significant_units_min_over_taps"]["value"] == 0
    assert facts["significant_units_max_over_taps"]["value"] == 0
    assert facts["total_clusters_a"]["value"] == 0 and facts["total_clusters_b"]["value"] == 0
    assert facts["mean_morans_i"]["value"] > 0.7
    section = _squash(_guide_text()).split("#### Reading a t-map")[1].split("#### Hatching and outlines")[0]
    assert "The map is smooth, and it is not significant" in section
    assert "it covers every cell, so nothing in this 2D example is significant" in section
    assert "smooth" in section and "not significant" in section


def test_the_synthetic_illustration_is_synthetic_in_the_facts_and_the_guide():
    entry = _facts()["files"]["t_maps_2d_illustration.png"]
    assert entry["kind"] == "synthetic"
    assert entry["run"] is None and entry["step"] is None
    assert _illustration_failures(_guide_text(), _facts()) == []
    # No run image of the guide is of the synthetic kind, and every other one has a run.
    for name, other in _shipped().items():
        if name != "t_maps_2d_illustration.png":
            assert other.get("kind") != "synthetic" and other["run"] in ("2d", "3d"), name


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
    per_panel = _facts()["files"]["t_maps_2d_illustration.png"]["facts"]["per_panel"]
    assert len(per_panel) == 4
    for stats in per_panel.values():
        assert stats["clusters_a"] == 2 and stats["clusters_b"] == 1
    first = per_panel["synthetic/tap_1"]["significant_units"]
    assert f"the first has {first} significant units" in _caption("t_maps_2d_illustration.png")


# --- the prose beside the figures --------------------------------------------


def _run_series(run: dict, key: str) -> dict:
    return {int(k): v for k, v in run[key].items()}


def _prose_phrases(facts: dict):
    """Every phrase the guide's prose must print, built from the facts: ``(in_guide,
    in_readme)``."""
    curve = facts["files"]["live_curve_2d.png"]["facts"]
    run2d, run3d = facts["runs"]["2d"], facts["runs"]["3d"]
    trained = facts["files"]["t_maps_2d_trained.png"]["facts"]
    untrained = facts["files"]["t_maps_2d_untrained.png"]["facts"]
    three = facts["files"]["t_maps_3d_trained.png"]["facts"]
    moran = _run_series(run2d, "mean_morans_i_series")
    sig = {int(k): v for k, v in curve["significant_units_mean_series"].items()}
    ca = {int(k): v for k, v in curve["clusters_a_series"].items()}
    cb = {int(k): v for k, v in curve["clusters_b_series"].items()}
    assert set(ca.values()) == {0} and set(cb.values()) == {0}, "the curve text says 0 and 0 at every step"
    assert [s for s, v in sig.items() if v] == [200], "the curve text says 0.25 at step 200 only"
    later = [moran[s] for s in moran if s >= 100]
    last = [moran[s] for s in moran if s >= 250]
    points = list(curve["validation_loss_points_drawn"]["value"].values())
    units = trained["units_per_tap"]["value"]
    comparisons = units * trained["num_taps"]["value"]
    need = scipy.stats.t.isf(0.05 / comparisons / 2, 8)
    counts = run2d["stimuli_framing"]["a"]["text_token_counts"]
    for condition in "abcd":
        assert run2d["stimuli_framing"][condition]["text_token_counts"] == counts
        assert run3d["stimuli_framing"][condition]["text_token_counts"] == counts
    mean_tokens = sum(counts) / len(counts)
    framing2, framing3 = run2d["controls"]["framings"], run3d["controls"]["framings"]

    def prefix_row(label, key):
        def cell(framing):
            row = framing[key]
            return f"{row['mean_significant_units_per_tap']:.1f} ({row['clusters_a']} / {row['clusters_b']})"
        return f"| {label} | {cell(framing2)} | {cell(framing3)} |"

    moran_by_prefix = {
        "2D": [row["mean_morans_i"] for row in framing2.values()],
        "3D": [row["mean_morans_i"] for row in framing3.values()],
    }
    sig3 = [row["mean_significant_units_per_tap"] for row in framing3.values()]
    shuffle2, shuffle3 = (run["controls"]["label_shuffle_null"] for run in (run2d, run3d))
    length2, length3 = (run["controls"]["length_control_all"] for run in (run2d, run3d))
    five = [
        length[key]["sig_max"]
        for length in (length2, length3)
        for key in ("b_vocab_5v5", "a_vocab_5v5")
    ]
    per_tap3 = three["per_tap"]
    in_guide = [
        # the framing and the matching
        f"{', '.join(str(c) for c in counts[:-1])} and {counts[-1]} for i = 0 to 4 (mean {mean_tokens:.1f} in every condition)",
        prefix_row("CLS marker", "cls_prefix"),
        prefix_row("end-of-text id (the one used here)", "eot_prefix_shipped"),
        prefix_row("none", "no_prefix"),
        f"at most 0.03 across the three prefixes (2D {min(moran_by_prefix['2D']):.3f} to {max(moran_by_prefix['2D']):.3f}, "
        f"3D {min(moran_by_prefix['3D']):.3f} to {max(moran_by_prefix['3D']):.3f})",
        f"factor of about {max(sig3) / min(sig3):.1f} with the framing ({min(sig3):.1f} to {max(sig3):.1f})",
        # reading the 2D map
        f"the largest t of the run, {trained['t_max_over_taps']['fmt']}",
        f"over {comparisons} comparisons",
        f"need a |t| of about {need:.1f} to pass",
        f"Step 0 is drawn on +/-{untrained['colour_limit']['fmt']} and step 400 on +/-{trained['colour_limit']['fmt']}",
        f"(at step 400 the largest |t| is {trained['t_max_over_taps']['fmt']} on a scale of +/-{trained['colour_limit']['fmt']})",
        f"(the mean Moran's I went from {untrained['mean_morans_i']['fmt']} to {trained['mean_morans_i']['fmt']})",
        # the 3D map
        f"{three['mean_significant_units_per_tap']['fmt']} of {three['units_per_tap']['fmt']} units per tap are significant on average",
        f"{three['significant_units_min_over_taps']['fmt']} to {three['significant_units_max_over_taps']['fmt']} depending on the tap",
        f"{per_tap3['block_2/mlp_tap']['significant_units']['fmt']} of them are in `block_2/mlp_tap`",
        "one cluster of `b` of 12 units",
        # Moran's I
        f"the lowest tap is `block_0/mlp_tap` ({trained['per_tap']['block_0/mlp_tap']['morans_i']['fmt']}) and the highest "
        f"`block_3/mlp_tap` ({trained['per_tap']['block_3/mlp_tap']['morans_i']['fmt']})",
        f"(2D median {shuffle2['moran_mean']['median']:.3f}, the real labelling {trained['mean_morans_i']['fmt']}; "
        f"3D median {shuffle3['moran_mean']['median']:.3f}, the real labelling {three['mean_morans_i']['fmt']})",
        f"defined in {three['taps_with_islands_morans_i_defined']['fmt']} of the {three['num_taps']['fmt']} 3D taps, between "
        f"{three['morans_i_islands_min_over_taps']['fmt']} and {three['morans_i_islands_max_over_taps']['fmt']}",
        # the label-shuffle table
        f"| significant units per tap, real labelling | {_g(shuffle2['observed_sig'])} | {three['mean_significant_units_per_tap']['fmt']} |",
        f"| shuffled: median, 95th percentile, maximum | {_g(shuffle2['sig_per_tap']['median'])}, {_g(shuffle2['sig_per_tap']['p95'])}, "
        f"{_g(shuffle2['sig_per_tap']['max'])} | {_g(shuffle3['sig_per_tap']['median'])}, {_g(shuffle3['sig_per_tap']['p95'])}, "
        f"{_g(shuffle3['sig_per_tap']['max'])} |",
        f"| labellings with at least as many as the real one (itself included) | {shuffle2['n_splits_sig_ge_observed']} | "
        f"{shuffle3['n_splits_sig_ge_observed']} |",
        f"| clusters, real labelling | {shuffle2['observed_clusters']} | {shuffle3['observed_clusters']} |",
        f"| labellings with at least one cluster (the real one and its mirror included) | "
        f"{round(shuffle2['n_splits'] * (1 - shuffle2['clusters_total']['fraction_zero']))} | "
        f"{round(shuffle3['n_splits'] * (1 - shuffle3['clusters_total']['fraction_zero']))} |",
        f"| Moran's I, real labelling | {trained['mean_morans_i']['fmt']} | {three['mean_morans_i']['fmt']} |",
        f"| shuffled: median (5th to 95th percentile) | {shuffle2['moran_mean']['median']:.3f} ({shuffle2['moran_mean']['p5']:.3f} to "
        f"{shuffle2['moran_mean']['p95']:.3f}) | {shuffle3['moran_mean']['median']:.3f} ({shuffle3['moran_mean']['p5']:.3f} to "
        f"{shuffle3['moran_mean']['p95']:.3f}) |",
        f"{shuffle3['n_splits_sig_ge_observed']} of {shuffle3['n_splits']} is about {shuffle3['n_splits_sig_ge_observed'] / shuffle3['n_splits']:.3f}",
        f"a strictly larger count ({_g(shuffle3['sig_per_tap']['max'])})",
        f"all {shuffle2['n_splits']} ways to choose 5 of 10",
        # the length-control table
        f"| 3 words against 7 words, `b` vocabulary | {length2['b_vocab_3v7']['sig_median']:.1f} | {length3['b_vocab_3v7']['sig_median']:.1f} |",
        f"| 3 words against 7 words, `a` vocabulary | {length2['a_vocab_3v7']['sig_median']:.1f} | {length3['a_vocab_3v7']['sig_median']:.1f} |",
        f"| 5 words against 5 words, either vocabulary | {max(length2['b_vocab_5v5']['sig_median'], length2['a_vocab_5v5']['sig_median']):.1f} | "
        f"{max(length3['b_vocab_5v5']['sig_median'], length3['a_vocab_5v5']['sig_median']):.1f} |",
        f"(at most {max(five):.1f} in any of the {run2d['controls']['length_control_same_vocabulary_3_vs_7_words_median_significant_units_per_tap']['draws']} draws)",
        f"{three['mean_significant_units_per_tap']['fmt']} significant units per tap, against {length3['b_vocab_3v7']['sig_median']:.1f} for lists four words apart",
        # the live curve, panel by panel
        f"from {curve['mean_morans_i_first_step']['fmt']} at step 0 to {moran[50]:.3f} at step 50",
        f"peaks at {curve['mean_morans_i_peak']['fmt']} at step {curve['mean_morans_i_peak_step']['fmt']}",
        f"between {min(later):.3f} and {max(later):.3f} from step 100 on",
        f"({min(last):.3f} to {max(last):.3f} from step 250 on)",
        f"the taps end between {trained['morans_i_min_over_taps']['fmt']} and {trained['morans_i_max_over_taps']['fmt']}",
        "0 and 0 at every step",
        f"0.0 at every step except {sig[200]:.2f} at step 200",
        ", ".join(f"{v:.3f}" for v in points),
        # the scale of the 3D run's claim in the summary
        f"Moran's I rises from about 0 to {trained['mean_morans_i']['fmt']} (2D) and {three['mean_morans_i']['fmt']} (3D)",
    ]
    # Outside the guide: the sections on watching a run and on training a 3D variant.
    s3 = _run_series(run3d, "mean_morans_i_series")
    s3sig = _run_series(run3d, "mean_significant_units_series")
    w0, h0 = (facts["files"]["t_maps_2d_untrained.png"][k] for k in ("width", "height"))
    w4, h4 = (facts["files"]["t_maps_2d_trained.png"][k] for k in ("width", "height"))
    assert (w0, h0) == (w4, h4), "the frame-size sentence says the two 2D frames have one size"
    in_readme = [
        f"on +/-{untrained['colour_limit']['fmt']} at step 0 and +/-{trained['colour_limit']['fmt']} at step 400",
        f"stayed at {w0}x{h0} at step 0 and at step 400",
        # the 3D training bullet
        f"from {s3[0]:.4f} at step 0 to {s3[50]:.3f} at step 50 and {s3[400]:.3f} at step 400",
        f"{s3sig[250]:.1f} up to step 250, then {s3sig[300]:.3f} at steps 300 and 350 and {s3sig[400]:.1f} at step 400",
        f"{three['total_clusters_a']['fmt']} clusters for `a` and {three['total_clusters_b']['fmt']} for `b` at step 400",
        f"{length3['b_vocab_3v7']['sig_median']:.1f} significant units per tap on this checkpoint",
        # the lag of the last full row behind the final history value (tiny: from the facts)
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
    """The words of the two controls hold in the data: the real 2D value is the null's
    median, the real 3D value is in the top 4 of 252 and two splits beat it that are
    not the real labelling, and Moran's I of the real labelling lies inside the
    shuffled range in both dimensions."""
    runs = _facts()["runs"]
    shuffle2 = runs["2d"]["controls"]["label_shuffle_null"]
    shuffle3 = runs["3d"]["controls"]["label_shuffle_null"]
    assert shuffle2["observed_sig"] == shuffle2["sig_per_tap"]["median"] == 0.0
    assert shuffle2["n_splits_sig_ge_observed"] == shuffle2["n_splits"] == 252
    assert shuffle3["n_splits_sig_ge_observed"] == 4 and shuffle3["n_splits_sig_gt_observed"] == 2
    assert shuffle3["sig_per_tap"]["max"] > shuffle3["observed_sig"]
    for run, shuffle in (("2d", shuffle2), ("3d", shuffle3)):
        observed = runs[run]["last_live_row_mean_morans_i"]
        assert shuffle["moran_mean"]["p5"] <= observed <= shuffle["moran_mean"]["p95"], run
    assert runs["3d"]["controls"]["length_control_all"]["b_vocab_3v7"]["sig_median"] > (
        4 * runs["2d"]["controls"]["length_control_all"]["b_vocab_3v7"]["sig_median"]
    )
    assert runs["2d"]["controls"]["length_control_all"]["b_vocab_3v7"]["sig_median"] > 0


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
    "figures/clusters_2d.png",                 # no run wrote a cluster figure
    "figures/clusters_3d.png",
    "ticks stop at 6",                         # the colour bar of the earlier trained map
)


@pytest.mark.parametrize("readme", _READMES, ids=lambda p: p.parent.name)
def test_no_retracted_sentence_is_back(readme):
    text = readme.read_text(encoding="utf-8")
    squashed = _squash(text)
    assert [s for s in _RETRACTED_SENTENCES if _squash(s) in squashed] == []


#: What the guide must now say, each a sentence the review asked for.
_REQUIRED_SENTENCES = (
    "averaged over the text tokens of the sentence",                 # the pooling (D-020)
    "The prefix is a choice",                                        # the framing (D-024)
    "matched in length, in tokens",                                  # the stimuli (D-025)
    "learned **absolute** position embeddings",
    "not free of length and position effects",
    "The map is smooth, and it is not significant",                  # the 2D example
    "a hatched cell did not pass",                                   # literally true since D-021
    "only tell clusters apart",                                      # the cluster colours
    "do not encode the condition",
    "Neither example run wrote one",                                 # the cluster figure
    "not a growth curve",                                            # the significant-units line
    "never as \"the map separates a from b\"",                       # what Moran's I is
    "label-shuffle null",
    "length control",
    "No claim is made that 3D is better or worse than 2D",
    "A real selectivity claim needs a larger stimulus set",
    "A second correction: earlier t-maps were averaged over the padding",
    "A third correction: the stimuli framing and their lengths changed what the t-maps show",
)


def test_the_guide_says_what_the_review_asked_it_to_say():
    guide = _squash(_readme_text())
    assert [s for s in _REQUIRED_SENTENCES if _squash(s) not in guide] == []


#: Phrases that are false for the 2D example but true of an EARLIER run, so they may
#: stand only inside a unit (bullet, table, paragraph) that carries an older-run label.
_ONLY_WITH_A_LABEL = (
    "islands of significance",
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


def test_islands_of_significance_is_not_said_of_the_current_example():
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
        assert "Mean Moran's I 0.785" in trained
        mutated = guide.replace("Mean Moran's I 0.785", "Mean Moran's I 0.795", 1)
        failures = _caption_failures("t_maps_2d_trained.png", mutated)
        assert any("0.785" in failure for failure in failures), failures

    def test_a_changed_prose_number_is_named(self):
        guide, readme = _squash(_guide_text()), _squash(_readme_text())
        assert _prose_failures(guide, readme, _facts()) == []
        mutated = guide.replace("| 3 words against 7 words, `b` vocabulary | 42.4 |", "| 3 words against 7 words, `b` vocabulary | 4.2 |")
        assert mutated != guide
        failures = _prose_failures(mutated, readme, _facts())
        assert any("42.4" in failure for failure in failures), failures

    def test_a_reinserted_retracted_claim_about_the_2d_example_is_flagged(self):
        text = _readme_text()
        assert _unlabelled_mentions(text) == []
        anchor = "**Look at the hatching of the trained map"
        assert anchor in text
        mutated = text.replace(
            anchor,
            "The trained 2D map is islands of significance in a mostly hatched field.\n\n" + anchor,
            1,
        )
        assert _unlabelled_mentions(mutated) == ["islands of significance"]
        # ... and the same words inside an earlier-run sentence are allowed.
        labelled = text.replace(
            anchor,
            "The earlier example read the 2D map as islands of significance.\n\n" + anchor,
            1,
        )
        assert _unlabelled_mentions(labelled) == []

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
