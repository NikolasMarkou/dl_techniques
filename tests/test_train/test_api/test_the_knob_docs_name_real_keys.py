"""The knob documentation of ``python -m train`` names only keys, defaults, hooks and files that exist.

``src/train/README.md`` and ``src/train/AGENTS.md`` each hold ONE region between
``<!-- knobs:begin -->`` and ``<!-- knobs:end -->``, and ``README.md`` holds one cost table between
``<!-- costs:begin -->`` and ``<!-- costs:end -->`` inside it. What this file CHECKS, mechanically,
against the real code (review-iter-1 concern 7; every plant listed in ``PLANTS`` is driven in
``test_every_plant_is_flagged``):

* the region is unique: exactly one begin and one matching end marker per document, never two;
* a key of the task config, anywhere in the DOCUMENT (inside the region or outside it, in backticks
  or not): ``<group>_args.<field>`` and ``data.<field>`` must be a field of the real
  :class:`~train.api.tasks.image_classification.ImageClassificationConfig` (``data`` is the union
  of the three data classes); a backticked ``<name>_args`` alone must be a group; a ``<group>``
  that is not a group is refused; digits count in a field name;
* an ``=value`` suffix (``analysis_args.conformal_alpha=0.1``) must be JSON of the field's real
  type, and, when the sentence also says "default", equal to the field's real default;
* a sentence that says "default", names one key and a ``true``, ``false``, ``null`` or number after
  "is" or "=", states that key's default, and it must equal the real one (JSON spelling);
* a table whose second column is ``Default`` must hold, for every key in its first column, the
  real default, with or without backticks around it;
* a table whose third column is ``Read at``: every ``src/`` or ``tests/`` path in the region must
  exist, and every backticked ``Class.attr`` or ``module.function`` rooted at a class or module this
  file knows (``CLASSES``, ``MODULES``, a ``train.`` or ``dl_techniques.`` dotted path) must resolve;
* ``TaskTrainer.<name>`` and ``ImageClassificationTrainer.<name>``, with or without parentheses,
  must be real attributes, and with parentheses their parameter names must be the real ones;
* ``summary.analyzer.<key>`` must be a key of the analyzer block of ``results_summary.json``;
* every backticked output file name (``.png``, ``.json``, ``.keras``, ``.csv``, ``.log``,
  ``.lock``) must occur in the sources under ``src/``;
* every field of ``AnalysisArgs`` and ``LoggerArgs`` is named in the region;
* the region has no em dash;
* SECONDS: a number followed by ``s`` (a duration) may stand ONLY inside the cost table, and every
  row of that table is recomputed from ``knob_cost_evidence.json`` (the raw wall seconds and the
  output sizes of the one-run-each measurement, copied here from the untracked
  ``plans/.../scratch_step13/measure_out.txt``), so a cost sentence in prose cannot be written
  and a cost number cannot drift from the evidence.

What is NOT checked, and therefore must not be read as checked: the words of a description (what a
knob "writes", "costs nothing", "needs a test split"), any number that is not a duration, a
dotted name rooted at something the resolver does not know (``foo.bar``), a default stated in a
form the "default ... is" rule does not match, the contents of a file a name points to, and the
evidence file itself (a copy of one measurement; this test does not re-measure it).

A region that is missing or empty fails: the test must not pass for want of text.
"""

import dataclasses
import functools
import importlib
import inspect
import json
import re
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from train.api.args import AnalysisArgs, LoggerArgs
from train.api.data_args import BuiltinImageData, CsvImageData, FolderImageData
from train.api.tasks.image_classification import ImageClassificationConfig, ImageClassificationTrainer
from train.api.trainer import TaskTrainer
from train.common.run_summary import skipped_analysis_status

ROOT = Path(__file__).resolve().parents[3]
DOCS = ("src/train/README.md", "src/train/AGENTS.md")
BEGIN, END = "<!-- knobs:begin -->", "<!-- knobs:end -->"
COSTS_BEGIN, COSTS_END = "<!-- costs:begin -->", "<!-- costs:end -->"
EVIDENCE = Path(__file__).with_name("knob_cost_evidence.json")

#: Classes whose attributes a backticked ``Class.attr`` may name, by the name the docs use.
CLASSES = {
    "TaskTrainer": TaskTrainer,
    "ImageClassificationTrainer": ImageClassificationTrainer,
    "AnalysisArgs": AnalysisArgs,
    "LoggerArgs": LoggerArgs,
    "ImageClassificationConfig": ImageClassificationConfig,
}
#: Module aliases the docs use for ``module.function``; a ``train.`` or ``dl_techniques.`` path is
#: imported as written.
MODULES = {
    "classification_run": "train.common.classification_run",
    "run_summary": "train.common.run_summary",
    "conformal": "train.common.conformal",
}
#: ``Class.attr`` roots in other packages, imported lazily (they pull in the analyzer).
LAZY_CLASSES = {
    "AnalysisResults": "dl_techniques.analyzer.data_types:AnalysisResults",
    "ModelAnalyzer": "dl_techniques.analyzer.model_analyzer:ModelAnalyzer",
}

KEY = re.compile(
    r"(?<![\w./-])((?:[a-z]+_)+args|data)\.([A-Za-z0-9_]+)(?:=([^\s`|,;)]+))?"
)
GROUP_ONLY = re.compile(r"`((?:[a-z]+_)+args)`")
SUMMARY = re.compile(r"`summary\.analyzer\.([A-Za-z0-9_]+)`")
HOOK = re.compile(r"`(TaskTrainer|ImageClassificationTrainer)\.([A-Za-z_]\w*)(?:\(([^`]*)\))?`")
DOTTED = re.compile(r"`([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)(?:\([^`]*\))?`")
PATH = re.compile(r"\b((?:src|tests)/[\w/.\-]+\.(?:py|md|json))\b")
FILE_NAME = re.compile(r"`([\w\-]+(?:\.[\w\-]+)*\.(?:png|json|keras|csv|log|lock))`")
FILE_SUFFIXES = ("png", "json", "keras", "csv", "log", "lock", "md", "py", "txt")
SECONDS = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s?s\b(?!\w)")

#: The analyzer block of ``results_summary.json``: the status schema, plus the one key added
#: when an analysis left no result (``classification_run.report_best_weights``).
ANALYZER_BLOCK_KEYS = set(skipped_analysis_status()) | {"failed_analyses"}


# ---------------------------------------------------------------------
# What the code says
# ---------------------------------------------------------------------


def _group_names() -> List[str]:
    return [f.name for f in dataclasses.fields(ImageClassificationConfig) if f.name.endswith("_args") or f.name == "data"]


def _data_fields() -> Dict[str, dataclasses.Field]:
    found: Dict[str, dataclasses.Field] = {}
    for cls in (BuiltinImageData, FolderImageData, CsvImageData):
        for field in dataclasses.fields(cls):
            found.setdefault(field.name, field)
    return found


def _field(group: str, name: str) -> Optional[dataclasses.Field]:
    """Return the dataclass field ``group.name`` of the task config, or ``None``."""
    if group == "data":
        return _data_fields().get(name)
    top = {f.name: f for f in dataclasses.fields(ImageClassificationConfig)}
    if group not in top or top[group].default_factory is dataclasses.MISSING:
        return None
    cls = type(top[group].default_factory())
    inner = {f.name: f for f in dataclasses.fields(cls)}
    return inner.get(name)


def _owner(group: str, name: str) -> Any:
    if group == "data":
        return next(c for c in (BuiltinImageData, FolderImageData, CsvImageData)
                    if name in {f.name for f in dataclasses.fields(c)})
    top = {f.name: f for f in dataclasses.fields(ImageClassificationConfig)}
    return type(top[group].default_factory())


def _default_text(field: dataclasses.Field) -> Optional[str]:
    """Return the field's default in the JSON spelling the docs use, or ``None`` when it has none."""
    if field.default is not dataclasses.MISSING:
        return json.dumps(field.default)
    if field.default_factory is not dataclasses.MISSING:
        return json.dumps(field.default_factory())
    return None


def _accepts(hint: Any, value: Any) -> bool:
    """Whether the JSON ``value`` has the type ``hint`` names."""
    origin = typing.get_origin(hint)
    if origin is typing.Union:
        return any(_accepts(arg, value) for arg in typing.get_args(hint))
    if origin is typing.Literal:
        return value in typing.get_args(hint)
    if hint is type(None):
        return value is None
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is str:
        return isinstance(value, str)
    if hint is dict or origin is dict:
        return isinstance(value, dict)
    if hint is list or origin is list:
        return isinstance(value, list)
    return True  # a type this checker does not model: do not guess


def _hook_params(cls: Any, name: str) -> Optional[List[str]]:
    member = getattr(cls, name, None)
    if member is None or not callable(member):
        return None
    return [p for p in inspect.signature(member).parameters if p != "self"]


@functools.lru_cache(maxsize=1)
def _source_corpus() -> str:
    parts = []
    for path in (ROOT / "src").rglob("*.py"):
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _resolve(token: str) -> Optional[bool]:
    """``True`` if the dotted ``token`` resolves, ``False`` if its root is known and it does not,
    ``None`` if its root is not one this file knows."""
    parts = token.split(".")
    root = parts[0]
    obj: Any
    if root in CLASSES:
        obj, rest = CLASSES[root], parts[1:]
    elif root in LAZY_CLASSES:
        module_name, _, attr = LAZY_CLASSES[root].partition(":")
        obj, rest = getattr(importlib.import_module(module_name), attr), parts[1:]
    elif root in MODULES:
        obj, rest = importlib.import_module(MODULES[root]), parts[1:]
    elif root in ("train", "dl_techniques"):
        for cut in range(len(parts), 0, -1):
            try:
                obj, rest = importlib.import_module(".".join(parts[:cut])), parts[cut:]
                break
            except ImportError:
                continue
        else:
            return False
    else:
        return None
    for part in rest:
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


# ---------------------------------------------------------------------
# What the document says
# ---------------------------------------------------------------------


def _region(text: str) -> str:
    """Return the text between the first begin marker and the first end marker after it."""
    start = text.find(BEGIN)
    end = text.find(END, start + len(BEGIN))
    return text[start + len(BEGIN):end] if start >= 0 and end >= 0 else ""


def _split_costs(region: str):
    """Return ``(region without its cost table, cost table rows or None)``."""
    if COSTS_BEGIN not in region:
        return region, None
    head, _, rest = region.partition(COSTS_BEGIN)
    table, _, tail = rest.partition(COSTS_END)
    rows = [line for line in table.splitlines() if line.startswith("|")]
    return head + tail, rows


def _cells(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _literal(shown: str) -> Any:
    """Parse the value a document writes: JSON (``true``, ``0.1``, ``{}``), else the bare word the
    command line takes (``cifar10`` is the string ``"cifar10"``)."""
    shown = shown.strip("`").rstrip(".")
    try:
        return json.loads(shown)
    except ValueError:
        return shown


def _key_problems(text: str) -> List[str]:
    found: List[str] = []
    groups = set(_group_names())
    for group, name, shown in KEY.findall(text):
        key = f"{group}.{name}"
        if group not in groups:
            found.append(f"`{key}`: `{group}` is not a group of the task config")
            continue
        field = _field(group, name)
        if field is None:
            found.append(f"`{key}` is not a field of the task config")
            continue
        if shown:
            value = _literal(shown)
            hint = typing.get_type_hints(_owner(group, name)).get(name, Any)
            if not _accepts(hint, value):
                found.append(f"`{key}={shown}`: the value is not of the field's type")
    for group in GROUP_ONLY.findall(text):
        if group not in groups:
            found.append(f"`{group}` is not a group of the task config")
    return found


def _sentences(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        if line.startswith("|"):
            continue
        out.extend(part for part in re.split(r"(?<=[.!?])\s+", line) if part.strip())
    return out


def _default_claims(text: str) -> List[str]:
    found: List[str] = []
    for sentence in _sentences(text):
        if not re.search(r"\bdefault\b", sentence, re.I):
            continue
        keys = {(g, n) for g, n, _ in KEY.findall(sentence) if _field(g, n) is not None}
        if len(keys) != 1:
            continue  # not attributable to one key: unchecked (see the module docstring)
        group, name = next(iter(keys))
        real = _default_text(_field(group, name))
        claimed = re.search(
            r"default.*?(?:\bis\b|=)\s*`?(true|false|null|-?\d+(?:\.\d+)?|\{\})`?(?![\w.]*\w)", sentence, re.I
        ) or re.search(rf"{re.escape(group)}\.{re.escape(name)}=`?([^\s`.,;)]+)", sentence)
        if claimed and real is not None and _literal(claimed.group(1).lower()) != json.loads(real):
            found.append(f"`{group}.{name}` is said to default to {claimed.group(1)}, the code has {real}")
    return found


def _table_problems(text: str) -> List[str]:
    found: List[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        header = _cells(lines[index]) if lines[index].startswith("|") else []
        if len(header) >= 2 and header[1].lower() == "default":
            index += 2  # the header and its separator
            while index < len(lines) and lines[index].startswith("|"):
                cells = _cells(lines[index])
                match = re.fullmatch(r"`((?:[a-z]+_)+args|data)\.([A-Za-z0-9_]+)`", cells[0])
                if match and len(cells) >= 2:
                    field = _field(*match.groups())
                    real = None if field is None else _default_text(field)
                    if field is not None and cells[1].strip("`") != real:
                        found.append(
                            f"`{match.group(1)}.{match.group(2)}` is documented with default "
                            f"{cells[1]}, the code has {real}"
                        )
                index += 1
        else:
            index += 1
    # Wherever a row sits, a first-column key whose second cell IS a value must have the real default.
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        match = re.fullmatch(r"`((?:[a-z]+_)+args|data)\.([A-Za-z0-9_]+)`", cells[0])
        if match and len(cells) >= 2 and re.fullmatch(r"`?(true|false|null|-?\d+(?:\.\d+)?|\{\})`?", cells[1]):
            field = _field(*match.groups())
            real = None if field is None else _default_text(field)
            message = (f"`{match.group(1)}.{match.group(2)}` is documented with default "
                       f"{cells[1]}, the code has {real}")
            if field is not None and cells[1].strip("`") != real and message not in found:
                found.append(message)
    return found


def _reference_problems(region: str) -> List[str]:
    found: List[str] = []
    for path in sorted(set(PATH.findall(region))):
        if not (ROOT / path).exists():
            found.append(f"the path {path} does not exist")
    for token in sorted(set(DOTTED.findall(region))):
        if token.split(".")[0] == "summary" or re.match(r"(?:[a-z]+_)*args\.", token):
            continue
        if token.split(".")[0] in ("TaskTrainer", "ImageClassificationTrainer"):
            continue  # the hook rule below owns these, with or without parentheses
        if token.rsplit(".", 1)[1] in FILE_SUFFIXES:
            continue  # a file name, checked against the sources below
        if _resolve(token) is False:
            found.append(f"`{token}` does not resolve")
    for cls_name, name, params in HOOK.findall(region):
        cls = CLASSES[cls_name]
        real = _hook_params(cls, name)
        if real is None:
            found.append(f"`{cls_name}.{name}` is not a method of {cls_name}")
    # With parentheses, the parameter names must be the real ones (without them only the name is).
    for cls_name, name, params in re.findall(
        r"`(TaskTrainer|ImageClassificationTrainer)\.([A-Za-z_]\w*)\(([^`]*)\)`", region
    ):
        real = _hook_params(CLASSES[cls_name], name)
        shown = [p.strip() for p in params.split(",") if p.strip()]
        if real is not None and real != shown:
            found.append(f"`{cls_name}.{name}` takes ({', '.join(real)}), documented ({', '.join(shown)})")
    for key in sorted(set(SUMMARY.findall(region))):
        if key not in ANALYZER_BLOCK_KEYS:
            found.append(f"`summary.analyzer.{key}` is not a key of the analyzer block")
    corpus = _source_corpus()
    for name in sorted(set(FILE_NAME.findall(region))):
        if name not in corpus:
            found.append(f"`{name}` is an output file name that no source under src/ contains")
    return found


def cost_problems(rows: Optional[List[str]]) -> List[str]:
    """Check the cost table against ``knob_cost_evidence.json``; ``None`` rows is a missing table."""
    if rows is None:
        return ["the cost table (between the costs markers) is missing"]
    evidence = json.loads(EVIDENCE.read_text())
    found: List[str] = []
    seen = set()
    for line in rows[2:]:
        cells = _cells(line)
        setting = cells[0].strip("`")
        if setting not in evidence["rows"]:
            found.append(f"cost row `{setting}` has no evidence")
            continue
        seen.add(setting)
        spec = evidence["rows"][setting]
        runs = evidence["runs"]
        want_seconds = runs[spec["run"]]["wall_s"] if spec["against"] is None else round(
            runs[spec["run"]]["wall_s"] - runs[spec["against"]]["wall_s"], 1
        )
        want_mb = None if spec["output_bytes"] is None else round(spec["output_bytes"] / 1e6, 1)
        shown_mb = None if cells[2] in ("-", "") else float(cells[2])
        if float(cells[1]) != want_seconds:
            found.append(f"cost row `{setting}` says {cells[1]} s, the evidence gives {want_seconds}")
        if shown_mb != want_mb:
            found.append(f"cost row `{setting}` says {cells[2]} MB, the evidence gives {want_mb}")
        found.extend(_key_problems(f"`{setting}`"))
        if setting != "base" and not KEY.search(setting):
            found.append(f"cost row `{setting}` is not a knob setting or `base`")
    for setting in evidence["rows"]:
        if setting not in seen:
            found.append(f"the evidence has a row `{setting}` the cost table does not show")
    return found


def problems(text: str) -> List[str]:
    """Name everything ``text`` (a whole document) states that is not true of the code."""
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.find(BEGIN) > text.find(END):
        return [f"the document needs exactly one {BEGIN} ... {END} region, found "
                f"{text.count(BEGIN)} begin and {text.count(END)} end markers"]
    found: List[str] = []
    region = _region(text)
    if not region.strip():
        return [f"no region between {BEGIN} and {END}"]
    found.extend(_key_problems(text))
    outside_costs, cost_rows = _split_costs(region)
    found.extend(_default_claims(text))
    found.extend(_table_problems(text))
    found.extend(_reference_problems(outside_costs))
    for group, cls in (("analysis_args", AnalysisArgs), ("logger_args", LoggerArgs)):
        for field in dataclasses.fields(cls):
            if not re.search(rf"`{group}\.{field.name}(?:=[^`]*)?`", region):
                found.append(f"`{group}.{field.name}` is a knob and the region does not name it")
    if "—" in region:
        found.append("the region has an em dash")
    for match in SECONDS.finditer(outside_costs):
        found.append(
            f"a duration ({match.group(0)!r}) outside the cost table: durations are stated only in "
            "the table, where they are checked against the evidence"
        )
    if COSTS_BEGIN in text:
        found.extend(cost_problems(cost_rows))
    return found


# ---------------------------------------------------------------------
# The documents
# ---------------------------------------------------------------------


@pytest.mark.parametrize("doc", DOCS)
def test_the_knob_region_names_only_what_exists(doc):
    """No stale key, default, summary key, hook, path, file name or duration in the document."""
    assert problems((ROOT / doc).read_text(encoding="utf-8")) == []


def test_the_readme_states_its_costs_in_a_table_pinned_to_the_evidence():
    text = (ROOT / "src/train/README.md").read_text(encoding="utf-8")
    assert COSTS_BEGIN in text, "the README must carry the cost table"
    assert cost_problems(_split_costs(_region(text))[1]) == []


# ---------------------------------------------------------------------
# The checker itself: every plant of the review is flagged
# ---------------------------------------------------------------------


def _good_region() -> str:
    rows = "\n".join(
        f"| `{group}.{f.name}` | `{_default_text(f)}` | text |"
        for group, cls in (("analysis_args", AnalysisArgs), ("logger_args", LoggerArgs))
        for f in dataclasses.fields(cls)
    )
    return f"{BEGIN}\n| Key | Default | What it does |\n|---|---|---|\n{rows}\n{END}\n"


#: The thirteen stale texts of ``scratch_review1/docplant.py``, each appended just before the end
#: marker unless it says otherwise. The first six columns are what the review planted; every one
#: must be flagged. ``foo.bar`` inside the fifth is NOT checkable and is not what flags it.
PLANTS = {
    "false prose cost": "Epoch analysis costs 0.1 s per epoch and figures write nothing.",
    "key with value": "`analysis_args.figures=false` is the default.",
    "bare key, no backtick": "analysis_args.nonexistent is the knob.",
    "wrong default in prose": "The default of `analysis_args.figures` is false.",
    "nonexistent non-args group": "`data.nonexistent` and `metric_args` and `foo.bar`",
    "single-segment group": "`nope_args`",
    "digit key": "`analysis_args.figures2`",
    "hook without parentheses": "`TaskTrainer.no_such_hook`",
    "hook of another class": "`ImageClassificationTrainer.no_such_hook(config)`",
    "row default without backticks": "| `analysis_args.figures` | false | text |",
    "row with a false Read at cell": "| `analysis_args.figures` | `true` | read at src/train/nowhere.py |",
}


def _with(good: str, text: str) -> str:
    return good.replace(END, text + "\n" + END)


@pytest.mark.parametrize("name", sorted(PLANTS))
def test_every_plant_is_flagged(name):
    good = _good_region()
    assert problems(good) == []
    # The row plants need a table whose header names the column they are about.
    text = _with(good, PLANTS[name])
    if name == "row with a false Read at cell":
        text = good.replace(END, "\n| Key | Default | Read at |\n|---|---|---|\n" + PLANTS[name] + "\n" + END)
    assert problems(text), f"the plant {name!r} passes the checker"


def test_a_bad_key_outside_the_markers_is_flagged():
    assert problems(_good_region() + "\n`analysis_args.nonexistent`\n")


def test_a_second_region_is_flagged_even_when_nothing_in_it_is_wrong():
    """The doubled markers are the defect: the second region would never be read."""
    assert problems(_good_region() + "\n" + BEGIN + "nothing wrong here" + END)
    assert problems(_good_region() + "\n" + END)  # a doubled end marker
    assert problems(END + "\n" + _good_region())  # the end marker first


def test_the_checker_flags_what_it_must():
    """Positive control: the good region is clean and each older kind of staleness is named."""
    good = _good_region()
    assert problems(good) == []
    assert any("not a field" in p for p in problems(_with(good, "`analysis_args.figure`")))
    assert any("documented with default" in p
               for p in problems(good.replace("`analysis_args.figures` | `true`", "`analysis_args.figures` | `false`")))
    assert any("not a key of the analyzer block" in p for p in problems(_with(good, "`summary.analyzer.failures`")))
    assert any("not a method" in p for p in problems(_with(good, "`TaskTrainer.no_such_hook(config)`")))
    assert any("takes (" in p for p in problems(_with(good, "`TaskTrainer.figure_callbacks(config, data)`")))
    assert any("does not name it" in p for p in problems(good.replace("`analysis_args.conformal_alpha`", "`x`")))
    assert any("em dash" in p for p in problems(_with(good, "—")))
    assert problems("no region here") != []


def test_a_value_suffix_is_checked_against_the_fields_type_and_a_claimed_default():
    good = _good_region()
    assert problems(_with(good, "`analysis_args.conformal_alpha=0.1` aims at 90% coverage.")) == []
    assert any("not of the field's type" in p for p in problems(_with(good, "`analysis_args.figures=0.5`")))
    assert any("not of the field's type" in p for p in problems(_with(good, "`analysis_args.epoch_analysis_every=true`")))
    assert any("not of the field's type" in p for p in problems(_with(good, "`analysis_args.figures=maybe`")))
    assert problems(_with(good, "`data.dataset=cifar10`, a bare word, is the string the command line takes")) == []
    # A sentence that says "default" and gives the real default is clean; the wrong one is not.
    assert problems(_with(good, "The default of `analysis_args.figures` is true.")) == []
    assert any("default to" in p for p in problems(_with(good, "The default of `analysis_args.epoch_analysis_every` is 3.")))


def test_a_row_is_compared_wherever_it_sits_and_a_default_column_is_compared_whatever_it_holds():
    good = _good_region()
    # Outside any Default table (after a prose line): a value in the second cell is compared.
    loose = good.replace(END, "\nA sentence.\n\n| `analysis_args.figures` | false | text |\n" + END)
    assert any("documented with default false" in p for p in problems(loose))
    assert problems(good.replace(END, "\nA sentence.\n\n| `analysis_args.figures` | true | text |\n" + END)) == []
    # Inside a Default table a second cell that is prose is not the default either.
    assert any("documented with default usually on" in p
               for p in problems(good.replace("`analysis_args.figures` | `true`", "`analysis_args.figures` | usually on")))


def test_a_file_name_a_name_and_a_module_function_must_exist():
    good = _good_region()
    assert any("no source under src/ contains" in p for p in problems(_with(good, "`no_such_figure_anywhere.png`")))
    assert problems(_with(good, "`confusion_matrix.png`")) == []
    assert any("does not resolve" in p for p in problems(_with(good, "`classification_run.no_such_function`")))
    assert any("does not resolve" in p for p in problems(_with(good, "`train.common.conformal.no_such_function`")))
    assert any("does not resolve" in p for p in problems(_with(good, "`LoggerArgs.no_such_field`")))
    assert problems(_with(good, "`classification_run.report_best_weights` and `train.common.conformal.run_conformal`")) == []
    assert problems(_with(good, "`foo.bar` is rooted at nothing this file knows, so it is unchecked")) == []


def test_a_data_key_is_checked_against_the_union_of_the_three_data_classes():
    good = _good_region()
    assert problems(_with(good, "`data.train_dir` and `data.dataset` and `data.max_samples`")) == []
    assert any("not a field" in p for p in problems(_with(good, "`data.train_dirr`")))


def test_a_cost_row_is_recomputed_from_the_evidence():
    evidence = json.loads(EVIDENCE.read_text())
    rows = ["| Setting | Extra wall seconds | Output, MB |", "|---|---|---|"]
    for setting, spec in evidence["rows"].items():
        runs = evidence["runs"]
        seconds = runs[spec["run"]]["wall_s"] if spec["against"] is None else round(
            runs[spec["run"]]["wall_s"] - runs[spec["against"]]["wall_s"], 1
        )
        megabytes = "-" if spec["output_bytes"] is None else round(spec["output_bytes"] / 1e6, 1)
        rows.append(f"| `{setting}` | {seconds} | {megabytes} |")
    assert cost_problems(rows) == []
    drifted = [rows[0], rows[1], *(r.replace(_cells(r)[1], "999.0") if i == 0 else r for i, r in enumerate(rows[2:]))]
    assert any("says 999.0 s" in p for p in cost_problems(drifted))
    wrong_size = [rows[0], rows[1], rows[2], *(r.replace(_cells(r)[2], "123.4") if i == 0 else r for i, r in enumerate(rows[3:]))]
    assert any("says 123.4 MB" in p for p in cost_problems(wrong_size))
    assert any("has no evidence" in p for p in cost_problems(rows[:2] + ["| `analysis_args.figures=false` | 1.0 | 1.0 |"]))
    assert any("does not show" in p for p in cost_problems(rows[:-1]))
    assert cost_problems(None)


def test_the_real_defaults_are_the_ones_the_docs_spell():
    """The JSON spelling used above matches the four analysis defaults, so a row can be checked."""
    assert [_default_text(_field("analysis_args", n)) for n in
            ("figures", "epoch_analysis_every", "model_analysis", "conformal_alpha")] == [
        "true", "0", "true", "null"]
    assert _default_text(_field("logger_args", "tensorboard")) == "{}"
