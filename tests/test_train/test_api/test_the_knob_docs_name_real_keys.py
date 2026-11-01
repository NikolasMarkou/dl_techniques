"""The knob documentation of ``python -m train`` names only keys, defaults and hooks that exist.

``src/train/README.md`` and ``src/train/AGENTS.md`` each hold one region between
``<!-- knobs:begin -->`` and ``<!-- knobs:end -->``. Inside a region, the convention is:

* a backticked dotted key rooted at a group of the task config, such as
  ``analysis_args.figures``, must be a field of the real
  :class:`~train.api.tasks.image_classification.ImageClassificationConfig`;
* a table row whose first cell is such a key and whose second cell is a backticked value
  states the key's default, and it must equal the field's real default (JSON spelling:
  ``true``, ``false``, ``null``, ``0``, ``{}``);
* a backticked ``summary.analyzer.<key>`` must be a key of the ``analyzer`` block of
  ``results_summary.json`` (the status schema, or ``failed_analyses``);
* a backticked ``TaskTrainer.<hook>(<params>)`` must be a real attribute of
  :class:`~train.api.trainer.TaskTrainer`, with exactly those parameter names;
* every field of ``AnalysisArgs`` and ``LoggerArgs`` must be named in the region, so a new
  knob cannot ship undocumented;
* the region has no em dash.

A region that is missing, or empty, fails: the test must not pass for want of text. The
checker is itself driven on a stale key, a stale default, a stale hook signature and a
missing knob (``test_the_checker_flags_what_it_must``).
"""

import dataclasses
import inspect
import json
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

from train.api.args import AnalysisArgs, LoggerArgs
from train.api.tasks.image_classification import ImageClassificationConfig
from train.api.trainer import TaskTrainer
from train.common.run_summary import skipped_analysis_status

ROOT = Path(__file__).resolve().parents[3]
DOCS = ("src/train/README.md", "src/train/AGENTS.md")
BEGIN, END = "<!-- knobs:begin -->", "<!-- knobs:end -->"

KEY = re.compile(r"`([a-z]+(?:_[a-z]+)*_args(?:\.[a-z_]+)+)`")
ROW = re.compile(r"^\|\s*`([a-z_]+_args(?:\.[a-z_]+)+)`\s*\|\s*`([^`]*)`\s*\|", re.M)
SUMMARY = re.compile(r"`summary\.analyzer\.([a-z_]+)`")
HOOK = re.compile(r"`TaskTrainer\.([a-z_]+)\(([^`]*)\)`")

#: The analyzer block of ``results_summary.json``: the status schema, plus the one key added
#: when an analysis left no result (``classification_run.report_best_weights``).
ANALYZER_BLOCK_KEYS = set(skipped_analysis_status()) | {"failed_analyses"}


def _region(text: str) -> str:
    """Return the text between the markers, or ``""`` when either is missing."""
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < start:
        return ""
    return text[start + len(BEGIN):end]


def _field(path: str) -> Optional[dataclasses.Field]:
    """Return the dataclass field a dotted ``path`` names in the task config, or ``None``."""
    cls: Any = ImageClassificationConfig
    parts = path.split(".")
    for index, part in enumerate(parts):
        fields = {f.name: f for f in dataclasses.fields(cls)}
        if part not in fields:
            return None
        field = fields[part]
        if index == len(parts) - 1:
            return field
        if field.default_factory is dataclasses.MISSING:
            return None
        cls = type(field.default_factory())
    return None


def _default_text(field: dataclasses.Field) -> str:
    """Return the field's default in the JSON spelling the docs use."""
    if field.default is not dataclasses.MISSING:
        return json.dumps(field.default)
    return json.dumps(field.default_factory())


def _hook_params(name: str) -> Optional[List[str]]:
    """Return the parameter names of ``TaskTrainer.<name>`` without ``self``, or ``None``."""
    member = getattr(TaskTrainer, name, None)
    if member is None or not callable(member):
        return None
    return [p for p in inspect.signature(member).parameters if p != "self"]


def problems(text: str) -> List[str]:
    """Name everything the region of ``text`` states that is not true of the code."""
    region = _region(text)
    if not region.strip():
        return [f"no region between {BEGIN} and {END}"]
    found: List[str] = []
    for key in sorted(set(KEY.findall(region))):
        if _field(key) is None:
            found.append(f"`{key}` is not a field of the task config")
    for key, shown in ROW.findall(region):
        field = _field(key)
        if field is not None and _default_text(field) != shown:
            found.append(f"`{key}` is documented with default {shown}, the code has {_default_text(field)}")
    for key in sorted(set(SUMMARY.findall(region))):
        if key not in ANALYZER_BLOCK_KEYS:
            found.append(f"`summary.analyzer.{key}` is not a key of the analyzer block")
    for name, params in HOOK.findall(region):
        real = _hook_params(name)
        shown = [p.strip() for p in params.split(",") if p.strip()]
        if real is None:
            found.append(f"`TaskTrainer.{name}` is not a method of TaskTrainer")
        elif real != shown:
            found.append(f"`TaskTrainer.{name}` takes ({', '.join(real)}), documented ({', '.join(shown)})")
    for group, cls in (("analysis_args", AnalysisArgs), ("logger_args", LoggerArgs)):
        for field in dataclasses.fields(cls):
            if f"`{group}.{field.name}`" not in region:
                found.append(f"`{group}.{field.name}` is a knob and the region does not name it")
    if "—" in region:
        found.append("the region has an em dash")
    return found


@pytest.mark.parametrize("doc", DOCS)
def test_the_knob_region_names_only_what_exists(doc):
    """No stale key, default, summary key or hook signature in the region."""
    assert problems((ROOT / doc).read_text(encoding="utf-8")) == []


def _good_region() -> str:
    rows = "\n".join(
        f"| `{group}.{f.name}` | `{_default_text(f)}` | text |"
        for group, cls in (("analysis_args", AnalysisArgs), ("logger_args", LoggerArgs))
        for f in dataclasses.fields(cls)
    )
    return f"{BEGIN}\n{rows}\n{END}\n"


def test_the_checker_flags_what_it_must():
    """Positive control: the good region is clean and each kind of staleness is named."""
    good = _good_region()
    assert problems(good) == []
    assert any("not a field" in p for p in problems(good.replace(END, "`analysis_args.figure`\n" + END)))
    assert any("documented with default" in p
               for p in problems(good.replace("`analysis_args.figures` | `true`",
                                              "`analysis_args.figures` | `false`")))
    assert any("not a key of the analyzer block" in p
               for p in problems(good.replace(END, "`summary.analyzer.failures`\n" + END)))
    assert any("not a method" in p
               for p in problems(good.replace(END, "`TaskTrainer.no_such_hook(config)`\n" + END)))
    assert any("takes (" in p
               for p in problems(good.replace(END, "`TaskTrainer.figure_callbacks(config, data)`\n" + END)))
    assert any("does not name it" in p
               for p in problems(good.replace("`analysis_args.conformal_alpha`", "`x`")))
    assert any("em dash" in p for p in problems(good.replace(END, "—\n" + END)))
    assert problems("no region here") != []


def test_the_real_defaults_are_the_ones_the_docs_spell():
    """The JSON spelling used above matches the four analysis defaults, so a row can be checked."""
    assert [_default_text(_field(f"analysis_args.{n}")) for n in
            ("figures", "epoch_analysis_every", "model_analysis", "conformal_alpha")] == [
        "true", "0", "true", "null"]
    assert _default_text(_field("logger_args.tensorboard")) == "{}"
