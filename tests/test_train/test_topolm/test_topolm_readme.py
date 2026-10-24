"""The TopoLM READMEs may cite only test classes that exist, and none of the figures
that were retracted (plan-2026-10-23T190557-972a9ac5, step 8.2); and the guide to
the figures is held to the DATA of the figures, not only to its own form (step 8.4,
decision D-022).

A README names the test classes that guard a claim; a class that was renamed or
deleted leaves a sentence pointing at nothing. This reads the two READMEs and
greps the test tree for each cited ``Test...`` name.

``src/train/topolm/figures/figure_facts.json`` is written, by a script, from the
real runs the seven example images came from (``metrics.jsonl``,
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
# Truth, not form (step 8.4, D-022): the guide against the figures' data.
# ---------------------------------------------------------------------------


def _facts() -> dict:
    return json.loads(_FACTS_PATH.read_text(encoding="utf-8"))


def _squash(text: str) -> str:
    """Whitespace collapsed to single spaces, so a phrase may wrap across lines."""
    return " ".join(text.split())


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


def _caption(name: str) -> str:
    """The italic paragraph that follows the image tag of ``name`` in the guide,
    whitespace collapsed."""
    guide = _guide_text()
    start = guide.index(f"figures/{name}")
    match = re.search(r"\n\n(\*.*?)\n\n", guide[start:], re.S)
    assert match, f"no caption paragraph follows {name}"
    return _squash(match.group(1))


def test_the_facts_file_pins_every_image_by_hash_size_and_pixels():
    entries = _facts()["files"]
    figures = _TRAINER_README.parent / "figures"
    assert sorted(entries) == sorted(p.name for p in figures.glob("*.png"))
    for name, entry in entries.items():
        data = (figures / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], f"{name} is not the figure the facts describe"
        assert len(data) == entry["bytes"], name
        assert _png_size(data) == (entry["width"], entry["height"]), name


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
        "{significant_units_min_over_taps} to {significant_units_max_over_taps} significant units per tap"
        " (mean {mean_significant_units_per_tap} of {units_per_tap})",
        "{total_clusters_a} clusters for `a` and {total_clusters_b} for `b` over the {num_taps} taps",
        "t values run from {t_min_over_taps} to {t_max_over_taps}",
        "+/-{colour_limit}",
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
        "tap `{cluster_figure_tap_name}`, condition `a`, one cluster of {cluster_figure_tap_sizes_a} units",
        "{width} x {height}",
    ),
    "clusters_3d.png": (
        "tap `{cluster_figure_tap_name}`, condition `a` only, one cluster of {cluster_figure_tap_sizes_a} units",
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


def _quote_context(name: str) -> dict:
    context = _figure_context(name)
    facts = _facts()["files"][name]["facts"]
    if "cluster_figure_tap" in facts:
        tap = facts["cluster_figure_tap"]
        context["cluster_figure_tap_name"] = tap["name"]["fmt"]
        context["cluster_figure_tap_sizes_a"] = tap["cluster_sizes_a"]["fmt"]
        context["cluster_figure_tap_sizes_b"] = tap["cluster_sizes_b"]["fmt"]
    context["colour_limit"] = facts["colour_limit"]["fmt"] if "colour_limit" in facts else ""
    context["per_tap_morans_i"] = ", ".join(v["morans_i"]["fmt"] for v in facts["per_tap"].values())
    return context


@pytest.mark.parametrize("name", sorted(_CAPTION_QUOTES))
def test_each_caption_prints_the_numbers_the_facts_hold(name):
    caption = _caption(name)
    context = _quote_context(name)
    missing = []
    for template in _CAPTION_QUOTES[name]:
        phrase = _squash(template.format(**context))
        if phrase not in caption:
            missing.append(phrase)
    assert missing == [], f"{name}: the caption does not say {missing}; it says {caption!r}"


def test_every_figure_has_a_caption_row_in_the_quote_table():
    assert sorted(_CAPTION_QUOTES) == sorted(_facts()["files"])


def test_the_prose_beside_the_figures_quotes_the_facts_too():
    """Numbers that sit in the guide's prose rather than in a caption: the live curve
    bullets, the per-tap sentences, the cluster sizes, the colour limits and the frame
    sizes. Every phrase is built from the facts file."""
    guide = _squash(_guide_text())
    facts = _facts()
    curve = facts["files"]["live_curve_2d.png"]["facts"]
    run2d, run3d = facts["runs"]["2d"], facts["runs"]["3d"]
    trained = facts["files"]["t_maps_2d_trained.png"]["facts"]
    untrained = facts["files"]["t_maps_2d_untrained.png"]["facts"]
    cluster2d = facts["files"]["clusters_2d.png"]["facts"]["cluster_figure_tap"]
    cluster3d = facts["files"]["clusters_3d.png"]["facts"]["cluster_figure_tap"]
    moran = {int(k): v for k, v in run2d["mean_morans_i_series"].items()}
    sig = {int(k): v for k, v in curve["significant_units_mean_series"].items()}
    ca = {int(k): v for k, v in curve["clusters_a_series"].items()}
    cb = {int(k): v for k, v in curve["clusters_b_series"].items()}
    later = [moran[s] for s in moran if s >= 100]
    points = list(curve["validation_loss_points_drawn"]["value"].values())
    per_tap = trained["per_tap"]
    phrases = [
        # the live curve, panel by panel
        f"from {curve['mean_morans_i_first_step']['fmt']} at step 0 to {moran[50]:.3f} at step 50",
        f"peaks at {curve['mean_morans_i_peak']['fmt']} at step {curve['mean_morans_i_peak_step']['fmt']}",
        f"between {min(later):.3f} and {max(later):.3f} from step 100 on",
        f"between {trained['morans_i_min_over_taps']['fmt']} and {trained['morans_i_max_over_taps']['fmt']}",
        f"{ca[200]} and {cb[200]} at step 200",
        f"{ca[250]} and {cb[250]} again at step 250",
        f"{ca[300]} and {cb[300]} at step 300",
        f"{ca[350]} and {cb[350]} at steps 350 and 400",
        f"{sig[200]:.1f} at step 200",
        f"{sig[250]:.1f} at step 250",
        f"{sig[300]:.1f}, {sig[350]:.1f} and {sig[400]:.1f} at steps 300, 350 and 400",
        ", ".join(f"{v:.3f}" for v in points),
        # the colour scale of the two 2D frames
        f"Step 0 is drawn on +/-{untrained['colour_limit']['fmt']} and step 400 on +/-{trained['colour_limit']['fmt']}",
        # the two taps that show almost no island
        f"`block_1/mlp_tap` has {per_tap['block_1/mlp_tap']['significant_units']['fmt']} significant units",
        f"`block_3/mlp_tap` has {per_tap['block_3/mlp_tap']['significant_units']['fmt']},",
        # the 15 per cent: the mean over the units per tap, rounded
        f"{round(100 * trained['mean_significant_units_per_tap']['value'] / trained['units_per_tap']['value'])} per cent of the cells on average",
        # the largest islands value over the taps
        f"{trained['morans_i_islands_max_over_taps']['fmt']} in 2D and "
        f"{facts['files']['t_maps_3d_trained.png']['facts']['morans_i_islands_max_over_taps']['fmt']} in 3D",
        # the cluster that the t-map outlines in magenta beside the one the cluster figure draws
        f"dashed magenta cluster of {cluster2d['cluster_sizes_b']['fmt']} units",
        f"dashed magenta cluster of {cluster3d['cluster_sizes_b']['fmt']} units",
        f"`{cluster2d['title_as_drawn']}`",
    ]
    # Outside the guide: the sections on watching a run and on training a 3D variant.
    elsewhere = [
        f"on +/-{untrained['colour_limit']['fmt']} at step 0 and +/-{trained['colour_limit']['fmt']} at step 400",
        # frame sizes of the 2D example run, quoted as a figure that changes size
        f"{facts['files']['t_maps_2d_untrained.png']['width']}x{facts['files']['t_maps_2d_untrained.png']['height']} at step 0 "
        f"to {facts['files']['t_maps_2d_trained.png']['width']}x{facts['files']['t_maps_2d_trained.png']['height']} at step 400",
        # the 3D training bullet
        f"from {run3d['mean_morans_i_series']['0']:.4f} at step 0 to {run3d['mean_morans_i_series']['50']:.3f} at step 50 "
        f"and {run3d['mean_morans_i_series']['400']:.3f} at step 400",
        f"{run3d['mean_significant_units_series']['250']:.1f}, {run3d['mean_significant_units_series']['300']:.1f}, "
        f"{run3d['mean_significant_units_series']['350']:.1f} and {run3d['mean_significant_units_series']['400']:.1f} at steps 250, 300, 350 and 400",
        f"{facts['files']['t_maps_3d_trained.png']['facts']['total_clusters_a']['fmt']} clusters for `a` and "
        f"{facts['files']['t_maps_3d_trained.png']['facts']['total_clusters_b']['fmt']} for `b` at step 400",
    ]
    readme = _squash(_readme_text())
    cost = [
        f"median {run2d['seconds_per_draw_median']['fmt']} s ({run2d['seconds_per_draw_min']['fmt']} to {run2d['seconds_per_draw_max']['fmt']}; "
        f"{run2d['seconds_per_draw_sum']['fmt']} s in all, in a run of {run2d['total_wall_s']['fmt']} s)",
        f"median {run3d['seconds_per_draw_median']['fmt']} s ({run3d['seconds_per_draw_min']['fmt']} to {run3d['seconds_per_draw_max']['fmt']}; "
        f"{run3d['seconds_per_draw_sum']['fmt']} s in all, in a run of {run3d['total_wall_s']['fmt']} s)",
        f"{run2d['peak_own_gpu_mib_nvidia_smi_1s_samples']['value']:,} MiB (`tiny`) and {run3d['peak_own_gpu_mib_nvidia_smi_1s_samples']['value']:,} MiB (`tiny3d`)",
    ]
    absent = [p for p in phrases if p not in guide] + [p for p in elsewhere + cost if p not in readme]
    assert absent == [], f"the prose does not say {absent}"


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
    "most cells passed",                       # the trained 2D map is mostly hatched
    "large patches of one\ncolour",            # (the earlier caption of the trained map)
    "187.9 of 256",                            # old-run caption numbers, see _OLDER_RUN_NUMBERS
    "142 to 206",
    "-17.3 to 19.1",
    "fell from 191 at step 50",
    "0.555 at step 50",
    "7.5333",
)


@pytest.mark.parametrize("readme", _READMES, ids=lambda p: p.parent.name)
def test_no_retracted_sentence_is_back(readme):
    text = readme.read_text(encoding="utf-8")
    squashed = _squash(text)
    assert [s for s in _RETRACTED_SENTENCES if _squash(s) in squashed] == []


#: What the guide must now say, each a sentence the review asked for.
_REQUIRED_SENTENCES = (
    "averaged over the text tokens of the sentence",                 # the pooling (D-020)
    "excluded by id",
    "not free of length effects",
    "islands of significance in a mostly non-significant field",
    "a hatched cell did not pass",                                  # literally true since D-021
    "only tell clusters apart",                                     # the cluster colours
    "do not encode the condition",
    "not a growth curve",                                           # the significant-units line
    "A second correction: earlier t-maps were averaged over the padding",
)


def test_the_guide_says_what_the_review_asked_it_to_say():
    guide = _squash(_readme_text())
    assert [s for s in _REQUIRED_SENTENCES if s not in guide] == []


#: Numbers of the EARLIER example runs (the padded average) or of controls measured
#: on them. They stay in the README only labelled as such, and each is listed here
#: with where it was measured, so none is mistaken for a number of the figures shown.
_OLDER_RUN_NUMBERS = {
    "187.9": "pooling-fix measurements, mean over 8 taps, 2D old (padded average)",
    "36.5": "same table, 2D new (text tokens), on the earlier checkpoint",
    "72.9": "same table, 3D old",
    "92.8": "same table, 3D new, on the earlier checkpoint",
    "202.9": "dotted control, 2D, old pooling",
    "10.9": "dotted control, 2D, new pooling",
    "111.0": "dotted control, 3D, old pooling",
    "35.9": "dotted control, 3D, new pooling",
    "1.034": "islands Moran's I of the earlier 2D example, block_0/attention_tap",
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the README bullet that starts 'Cutting every `a` sentence to the exact token "
        "lengths' (about line 372) quotes 36.5 and 92.8, numbers of the EARLIER "
        "checkpoints, with no label of its own: the finding of this stricter guard "
        "(step 9.1). Step 8.5 rewrites that bullet; when it does, this test XPASSes "
        "and strict=True fails it, which is the cue to delete this mark."
    ),
)
def test_numbers_of_the_earlier_runs_are_kept_only_with_their_label():
    assert _unlabelled_older_numbers(_readme_text()) == []


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

    def test_the_real_guide_with_an_unlabelled_bullet_slipped_in_is_flagged(self):
        """The README itself, mutated: a bullet quoting an older-run number with no
        label is inserted right BEFORE a labelled bullet, in the same list. The old
        paragraph-level check read the neighbour's label as its own."""
        lines = _readme_text().splitlines()
        number = re.compile(r"(?<![\d.])(?:%s)(?!\d)" % "|".join(map(re.escape, _OLDER_RUN_NUMBERS)))

        def labelled_bullet_at(start):
            end = start + 1
            while end < len(lines) and lines[end].strip() and not _BULLET_START.match(lines[end]):
                end += 1
            unit = " ".join(lines[start:end])
            return bool(number.search(unit) and _LABELS.search(unit))

        target = next(
            i for i, line in enumerate(lines)
            if _BULLET_START.match(line) and labelled_bullet_at(i)
        )
        lines.insert(target, "- the guide reports 72.9 significant units")
        flagged = _unlabelled_older_numbers("\n".join(lines))
        assert "72.9" in flagged, flagged
