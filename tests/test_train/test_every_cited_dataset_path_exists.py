"""Every ``/media/arxwn/data0_4tb/datasets/...`` path that ``src/train`` cites exists on disk.

Guard for plan-2026-11-10T095307-d483ffab step 1 (D-003). After the dataset directories were
renamed (``COCO`` -> ``coco_2017``, ``DIV2K`` -> ``div2k``, ``Megadepth`` -> ``megadepth``,
``bdd_data`` -> ``bdd100k``, ``cbsd68_src/CBSD68`` -> ``CBSD68``), eleven citations went stale and
nothing said so: ``common.TEST_DATASETS["cbsd68"]`` silently recorded CBSD68 as ``skipped`` in
every ``results_summary.json``. A default path that does not exist either crashes a run late or
quietly drops a benchmark.

What is checked, one test per citation (so a failure names the file and line):

* every string constant of every ``src/train/**/*.py`` (an AST walk, so docstrings and argparse
  defaults count, comments do not), and every line of every ``src/train/**/*.md``, is scanned for
  the dataset-root prefix; each path found must exist (``os.path.exists``, symlinks followed);
* the two directories that a script creates, not a rename target, are listed in
  ``CREATED_BY_A_SCRIPT`` with the script that creates them. The table is two-sided: a row whose
  creator does not name its own directory fails, and a row that matches no citation anywhere fails
  (a dead exemption would hide the next rename of that name).

Exact for literal paths only. A path composed at run time (``os.path.join(root, "x")``, an
f-string with a variable directory) and a path in a comment are invisible to it. A path ending in
a glob or a ``{a,b}`` brace set is cut at that character and only its literal prefix is checked.

The module skips, with a reason, on a machine without the dataset root; here it must run with
0 skips.

Separately, each held-out set of the legacy denoiser evaluation lists the image count the docs
state (kodak24 24, CBSD68 68, McMaster 18, Urban100 100) through the very listing function
``bfunet.common._evaluate_held_out`` uses, so a path that exists but is the wrong directory (an
empty dir, a pre-noised sibling) is caught as well.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "src" / "train"
DATASET_ROOT = "/media/arxwn/data0_4tb/datasets"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(DATASET_ROOT),
    reason=f"the dataset root {DATASET_ROOT} is not mounted on this machine",
)

# The root, then path components only: a glob, brace, quote, space or placeholder ends the match.
CITATION = re.compile(re.escape(DATASET_ROOT) + r"(?:/[A-Za-z0-9_.\-]+)*")

# Directories a script creates (not a dataset someone renamed): path -> the script that writes it.
# A citation equal to a row, or below it, is exempt from the existence check.
CREATED_BY_A_SCRIPT: Dict[str, str] = {
    f"{DATASET_ROOT}/fineweb_edu": "src/train/hnet/prepare_hnet_data.py",
    f"{DATASET_ROOT}/omnipoint_synth": "src/train/omnipoint/generate_synthetic_camera_eval_set.py",
}

# name -> (directory, images the docs state), read through the legacy listing function.
# kodak24 and cbsd68 are read from ``common.TEST_DATASETS`` itself; the other two are the
# held-out sets ``bfunet/FINDINGS.md`` names beside them.
STATED_IMAGE_COUNTS: Dict[str, int] = {"kodak24": 24, "cbsd68": 68, "mcmaster": 18, "urban100": 100}
EXTRA_HELD_OUT_DIRS: Dict[str, str] = {
    "mcmaster": f"{DATASET_ROOT}/mcmaster/color",
    "urban100": f"{DATASET_ROOT}/urban100/Urban100_HR",
}


def cited_paths(text: str) -> Iterator[Tuple[int, str]]:
    """``(offset, path)`` for every dataset-root citation in ``text``.

    A trailing ``.`` (the end of a sentence) is not part of the path. The bare root counts.
    """
    for match in CITATION.finditer(text):
        yield match.start(), match.group(0).rstrip(".")


def _citations_of(path: Path) -> Iterator[Tuple[int, str]]:
    """``(line, cited path)`` for one source file; ``.py`` by string constant, ``.md`` by line."""
    text = path.read_text(encoding="utf-8")
    if DATASET_ROOT not in text:
        return
    if path.suffix == ".md":
        for number, line in enumerate(text.splitlines(), start=1):
            for _, cited in cited_paths(line):
                yield number, cited
        return
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for offset, cited in cited_paths(node.value):
                yield node.lineno + node.value.count("\n", 0, offset), cited


def all_citations() -> List[Tuple[str, int, str]]:
    """``(repo-relative file, line, path)`` for every citation under ``src/train``, sorted."""
    found = set()
    files = [p for suffix in ("*.py", "*.md") for p in TRAIN.rglob(suffix) if "__pycache__" not in p.parts]
    for source in files:
        for line, cited in _citations_of(source):
            found.add((source.relative_to(REPO).as_posix(), line, cited))
    return sorted(found)


CITATIONS = all_citations()


def _exempt_row(cited: str) -> str:
    for row in CREATED_BY_A_SCRIPT:
        if cited == row or cited.startswith(row + "/"):
            return row
    return ""


def test_the_scan_finds_citations_in_both_kinds_of_file() -> None:
    """A wrong glob or a changed layout must not turn the guard into a vacuous pass."""
    files = {file for file, _, _ in CITATIONS}
    assert len(CITATIONS) >= 40, f"only {len(CITATIONS)} citations found; the scan is broken"
    assert any(f.endswith(".py") for f in files) and any(f.endswith(".md") for f in files)
    assert "src/train/bfunet/common.py" in files


@pytest.mark.parametrize(
    "file,line,cited", CITATIONS, ids=[f"{f}:{n}:{p.removeprefix(DATASET_ROOT)}" for f, n, p in CITATIONS]
)
def test_a_cited_dataset_path_exists(file: str, line: int, cited: str) -> None:
    if _exempt_row(cited):
        return  # an exempt citation is held to the two-sided table tests below instead
    assert os.path.exists(cited), (
        f"{file}:{line} cites {cited}, which does not exist on disk. If the dataset directory was "
        f"renamed, fix the citation; if a script creates it, add a row to CREATED_BY_A_SCRIPT."
    )


@pytest.mark.parametrize("row,creator", sorted(CREATED_BY_A_SCRIPT.items()))
def test_a_created_by_a_script_row_is_two_sided(row: str, creator: str) -> None:
    creator_file = REPO / creator
    assert creator_file.is_file(), f"{creator} (the creator of {row}) does not exist"
    # the creator may compose the path (``relative_dir="fineweb_edu/sample-10BT"``), so it is
    # held to naming the directory, not to citing the absolute path
    name = row.removeprefix(DATASET_ROOT + "/")
    assert name in creator_file.read_text(encoding="utf-8"), (
        f"{creator} never names {name!r}: the row names a script that does not create it"
    )
    elsewhere = [(f, n) for f, n, p in CITATIONS if _exempt_row(p) == row]
    assert elsewhere, f"no citation under src/train matches the exemption row {row}: delete the dead row"


def test_the_extractor_reads_a_sentence_and_stops_at_a_glob() -> None:
    root = DATASET_ROOT
    text = f"see {root}/div2k/train. Then {root}/imagenet2012/*.jpg and {root}/{{a,b}} and {root}/"
    assert [p for _, p in cited_paths(text)] == [f"{root}/div2k/train", f"{root}/imagenet2012", root, root]


def _listing() -> Tuple[Dict[str, str], object, object]:
    """The held-out directories and the legacy listing function with its extension set."""
    from train.bfunet import common, eval_psnr_vs_noise as ev

    directories = dict(common.TEST_DATASETS)
    missing = {"kodak24", "cbsd68"} - set(directories)
    assert not missing, f"common.TEST_DATASETS lost {sorted(missing)}"
    directories.update(EXTRA_HELD_OUT_DIRS)
    return directories, common.collect_image_paths, ev.IMAGE_EXTENSIONS


@pytest.mark.parametrize("name", sorted(STATED_IMAGE_COUNTS))
def test_a_held_out_set_lists_the_stated_image_count(name: str) -> None:
    directories, collect, extensions = _listing()
    paths = collect([directories[name]], extensions=extensions, sort=True)
    assert len(paths) == STATED_IMAGE_COUNTS[name], (
        f"{name} ({directories[name]}) lists {len(paths)} images through the legacy listing, "
        f"the docs state {STATED_IMAGE_COUNTS[name]}"
    )
