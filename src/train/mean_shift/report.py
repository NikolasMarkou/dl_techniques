"""Aggregate the mean shift sweeps: ``rows.jsonl`` into ``summary.json`` and ``summary.md``.

Reads the ``rows.jsonl`` (and ``manifest.json``) of one or more sweep roots written by
``train.mean_shift.common.run_sweep``::

    .venv/bin/python -m train.mean_shift.report \\
        --root results/mean_shift/<exp1 dir> --root results/mean_shift/<exp2 dir> --out <new dir>

Interface contract:

* Rows are keyed by the FULL tuple ``(experiment, arm, config, seed)`` (``common.cell_key``),
  never by the seed alone. A key with an ``ok`` row is a result; a key with only ``failed``
  rows is listed under ``failed``; a manifest cell with no row is listed under ``missing``.
  Nothing is dropped silently.
* Per ``(experiment, dataset)`` group the cells are compared by variant: the ``arm`` for
  ``exp1`` (reference ``mean``), the ``config`` id for ``exp2`` (reference ``mha``). Each
  variant reports mean and sample std (ddof=1, ``train.common.stats.mean_std``) over its
  seeds with the explicit ``n_seeds``, and for every accuracy metric the PAIRED difference
  against the reference, formed seed by seed (only seeds both variants finished).
* ``exp2`` also reports the learned bandwidths per config and ranks the MeanShift configs by
  SELECTION-split accuracy (ties: fewer steps, then config id). The test split never ranks.
  A multi_head variant other than the reference (the ``mha_oinit`` control) is a variant like
  any other (paired against ``mha``) but is not ranked.
* No significance test is applied: with 3 seeds a sign-flip test cannot go below p = 0.25.
* Failure modes: ``FileNotFoundError`` (a root without ``rows.jsonl`` and ``manifest.json``),
  ``ValueError`` (the same ok key in two roots, a malformed ok row, an unknown experiment),
  ``FileExistsError`` (``--out`` already holds a report; nothing is overwritten).

This module does not import TensorFlow or Keras.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dl_techniques.utils.logger import logger
from train.mean_shift import common, corruption


def _load_stats():
    """``train/common/stats.py`` by path: the ``train.common`` package init imports TensorFlow."""
    path = Path(__file__).resolve().parents[1] / "common" / "stats.py"
    spec = importlib.util.spec_from_file_location("_train_common_stats", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = _load_stats()

SUMMARY_NAME = "summary.json"
MARKDOWN_NAME = "summary.md"
SCHEMA = 1

#: Standard deviation convention: the sample std (Bessel), as ``train.common.stats`` defaults.
STD_DDOF = 1

#: The reference variant of each experiment: the arm for exp1, the config id for exp2.
REFERENCES: Dict[str, str] = {"exp1": "mean", "exp2": "mha"}

#: Experiments whose variants are ranked by selection-split accuracy.
RANKED_EXPERIMENTS = ("exp2",)
RANKING_METRIC = "select_accuracy"
FAILED_TAIL_CHARS = 300

CLAIMS_NOTE = (
    "No significance test is applied. With 3 seeds an exact sign-flip test cannot go below "
    "a two-sided p of 0.25, so these tables report mean and sample std over seeds and the "
    "per-seed paired differences, and nothing here supports a claim of significance.")

COST_FIELDS = ("params", "seconds_per_epoch", "throughput_images_per_s")


# ---------------------------------------------------------------------
# small helpers


def _num(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` when it is missing or not finite (strict JSON)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    """``{n, mean, std, values}`` of a list; ``std`` is ``None`` below two values."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "values": []}
    mean, std = stats.mean_std(list(values), ddof=STD_DDOF)
    return {"n": n, "mean": _num(mean), "std": _num(std) if n >= 2 else None,
            "values": [_num(v) for v in values]}


def _key_text(key: Tuple[str, str, str, int]) -> str:
    return "/".join(str(part) for part in key)


# ---------------------------------------------------------------------
# reading the sweeps


def _manifest_keys(root: Path) -> Optional[List[Tuple[str, str, str, int]]]:
    path = root / common.MANIFEST_NAME
    if not path.is_file():
        return None
    return [common.cell_key(cell) for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]]


def collect(roots: Sequence[Path]) -> Dict[str, Any]:
    """Every ok row, every failed key and every missing key of the given sweep roots.

    Returns:
        ``{"ok": {key: (root, row)}, "failed": {key: [rows]}, "missing": [key]}``.
    """
    ok: Dict[Tuple[str, str, str, int], Tuple[Path, Dict[str, Any]]] = {}
    failed: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    expected: List[Tuple[str, str, str, int]] = []
    for root in map(Path, roots):
        manifest = _manifest_keys(root)
        if manifest is None and not (root / common.ROWS_NAME).is_file():
            raise FileNotFoundError(f"{root} has neither {common.MANIFEST_NAME} nor {common.ROWS_NAME}")
        expected.extend(manifest or [])
        for row in common.read_rows(root):
            key = common.cell_key(row)
            if row.get("status") == "ok":
                if key in ok:
                    raise ValueError(
                        f"the ok cell {_key_text(key)} appears twice ({ok[key][0]} and {root}): "
                        "report sweeps with overlapping cells separately")
                ok[key] = (root, row)
            else:
                failed.setdefault(key, []).append(row)
    seen = set(ok) | set(failed)
    return {"ok": ok,
            "failed": {k: rows for k, rows in failed.items() if k not in ok},
            "missing": sorted(k for k in set(expected) if k not in seen)}


def _cell_values(experiment: str, key, row: Mapping[str, Any]) -> Dict[str, Any]:
    """The accuracy metrics, cost fields, dataset and epochs of one ok row; malformed raises."""
    try:
        result = row["metrics"]
        metrics = result["metrics"]
        if experiment == "exp1":
            accuracy = {name: float(metrics["accuracy"][name]) for name in corruption.CONDITIONS}
            accuracy[RANKING_METRIC] = float(metrics[RANKING_METRIC])
        elif experiment == "exp2":
            accuracy = {RANKING_METRIC: float(metrics[RANKING_METRIC]),
                        "test_accuracy": float(metrics["test_accuracy"])}
        else:
            raise ValueError(f"unknown experiment {experiment!r}")
        cost = {"params": result["params"], "seconds_per_epoch": result["seconds_per_epoch"],
                "throughput_images_per_s": metrics.get("throughput_images_per_s")}
        return {"accuracy": accuracy, "cost": cost, "dataset": str(metrics["dataset"]),
                "epochs": int(metrics["epochs"]), "metrics": metrics}
    except (KeyError, TypeError) as error:
        raise ValueError(f"the ok row of {_key_text(key)} lacks {error}") from error


def _variant(experiment: str, arm: str, config: str) -> str:
    return arm if experiment == "exp1" else config


# ---------------------------------------------------------------------
# bandwidths and ranking


def _bandwidth_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Learned bandwidths of one config over its seeds (layer-wise mean of means, extremes)."""
    records = [r["bandwidths"] for r in rows]
    statuses = {r["status"] for r in records}
    if len(statuses) > 1:
        raise ValueError(f"bandwidth status differs between seeds of one config: {sorted(statuses)}")
    status = statuses.pop()
    if status != "extracted":
        return {"status": status, "n_seeds": len(records), "layers": []}
    counts = {len(r["layers"]) for r in records}
    if len(counts) > 1:
        raise ValueError(f"layer counts differ between seeds of one config: {sorted(counts)}")
    layers = []
    for index in range(counts.pop()):
        per_seed = [r["layers"][index] for r in records]
        heads = list(zip(*[s["per_head_mean"] for s in per_seed]))
        layers.append({
            "layer": index,
            "mean": _num(sum(s["mean"] for s in per_seed) / len(per_seed)),
            "min": _num(min(s["min"] for s in per_seed)),
            "max": _num(max(s["max"] for s in per_seed)),
            "per_head_mean": [_num(sum(h) / len(h)) for h in heads],
            "inside_clip_range": all(s["inside_clip_range"] for s in per_seed),
        })
    return {"status": status, "mode": records[0]["mode"], "n_seeds": len(records), "layers": layers}


def _rank(variants: Mapping[str, Mapping[str, Any]], reference: str) -> List[Dict[str, Any]]:
    """MeanShift variants ordered by selection accuracy, then steps, then id.

    The reference and any other multi_head variant (a control such as ``mha_oinit``) are not
    ranked: the ranking chooses the MeanShift configs for the long run.
    """
    rows = []
    for name, cell in variants.items():
        if name == reference or cell["arm"] != "mean_shift":
            continue
        entry = cell["metrics"][RANKING_METRIC]
        steps = cell["config_args"].get("num_steps")
        rows.append({"config": name, "n_seeds": cell["n_seeds"], "num_steps": steps,
                     "select_accuracy_mean": entry["mean"], "select_accuracy_std": entry["std"]})
    rows.sort(key=lambda r: (-r["select_accuracy_mean"],
                             r["num_steps"] if r["num_steps"] is not None else math.inf,
                             r["config"]))
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows


# ---------------------------------------------------------------------
# the summary


def _paired(cell: Mapping[str, Any], reference: Mapping[str, Any], metric: str, ref_name: str):
    mine = dict(zip(cell["seeds"], cell["metrics"][metric]["values"]))
    theirs = dict(zip(reference["seeds"], reference["metrics"][metric]["values"]))
    seeds = sorted(set(mine) & set(theirs))
    diffs = [mine[s] - theirs[s] for s in seeds]
    block = _summary(diffs)
    return {"reference": ref_name, "seeds": seeds, "n_pairs": len(seeds),
            "unpaired_seeds": sorted(set(mine) ^ set(theirs)),
            "diffs": block["values"], "mean": block["mean"], "std": block["std"]}


def _build_group(experiment: str, dataset: str, cells: Mapping[str, Dict[int, Dict[str, Any]]]):
    reference = REFERENCES[experiment]
    names = sorted(cells, key=lambda n: (n != reference, n))
    variants: Dict[str, Dict[str, Any]] = {}
    epochs = set()
    for name in names:
        by_seed = cells[name]
        seeds = sorted(by_seed)
        entries = [by_seed[s] for s in seeds]
        epochs.update(e["epochs"] for e in entries)
        metric_names = list(entries[0]["accuracy"])
        cost = {}
        for field in COST_FIELDS:
            values = [e["cost"][field] for e in entries if e["cost"][field] is not None]
            if values:
                cost[field] = _summary(values)
        variants[name] = {
            "variant": name, "arm": entries[0]["arm"], "config": entries[0]["config"],
            "n_seeds": len(seeds), "seeds": seeds,
            "cell_dirs": [e["cell_dir"] for e in entries],
            "metrics": {m: _summary([e["accuracy"][m] for e in entries]) for m in metric_names},
            "cost": cost,
            "config_args": dict(entries[0]["metrics"].get("config_args") or {}),
        }
        if experiment == "exp2":
            variants[name]["bandwidths"] = _bandwidth_summary([e["metrics"] for e in entries])
    notes = []
    if reference in variants:
        for name, cell in variants.items():
            cell["paired"] = None if name == reference else {
                m: _paired(cell, variants[reference], m, reference) for m in cell["metrics"]}
    else:
        notes.append(f"reference {reference!r} has no ok cell in this group: no paired differences")
        for cell in variants.values():
            cell["paired"] = None
    if len(epochs) > 1:
        notes.append(f"the cells of this group trained for different epochs {sorted(epochs)}")
    seed_counts = {c["n_seeds"] for c in variants.values()}
    if len(seed_counts) > 1:
        notes.append(f"the variants have different seed counts {sorted(seed_counts)}")
    group = {"experiment": experiment, "dataset": dataset, "reference": reference,
             "epochs": sorted(epochs), "variants": list(variants.values()), "notes": notes}
    if experiment in RANKED_EXPERIMENTS:
        group["ranking"] = {"metric": RANKING_METRIC,
                            "tie_break": "fewer num_steps, then config id",
                            "rows": _rank(variants, reference)}
    return group


def build_summary(roots: Sequence[Path]) -> Dict[str, Any]:
    """The whole report as strict-JSON-safe Python (see the module docstring)."""
    found = collect(roots)
    grouped: Dict[Tuple[str, str], Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for key in sorted(found["ok"]):
        root, row = found["ok"][key]
        experiment, arm, config, seed = key
        values = _cell_values(experiment, key, row)
        entry = dict(values, arm=arm, config=config, cell_dir=str(root / row["cell_dir"]))
        slot = grouped.setdefault((experiment, values["dataset"]), {}).setdefault(
            _variant(experiment, arm, config), {})
        if seed in slot:
            raise ValueError(f"two ok cells of one variant share the seed {seed}: {_key_text(key)}")
        slot[seed] = entry
    return {
        "schema": SCHEMA,
        "roots": [str(r) for r in roots],
        "std_convention": f"sample std, ddof={STD_DDOF} (train.common.stats.mean_std); "
                          "null below two seeds",
        "claims": CLAIMS_NOTE,
        "references": dict(REFERENCES),
        "counts": {"ok": len(found["ok"]), "failed": len(found["failed"]),
                   "missing": len(found["missing"])},
        "groups": [_build_group(experiment, dataset, cells)
                   for (experiment, dataset), cells in sorted(grouped.items())],
        "failed": [{
            "key": list(key), "attempts": len(rows),
            "cell_dir": rows[-1].get("cell_dir"),
            "returncode": rows[-1].get("returncode"),
            "error_tail": (rows[-1].get("error_tail") or "")[-FAILED_TAIL_CHARS:]}
            for key, rows in sorted(found["failed"].items())],
        "missing": [list(key) for key in found["missing"]],
    }


# ---------------------------------------------------------------------
# markdown


def _fmt(entry: Mapping[str, Any], decimals: int = 4) -> str:
    if entry is None or entry["mean"] is None:
        return "n/a"
    if entry["std"] is None:
        return f"{entry['mean']:.{decimals}f} +- n/a"
    return f"{entry['mean']:.{decimals}f} +- {entry['std']:.{decimals}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _signed(values: Sequence[Optional[float]]) -> str:
    return ", ".join("n/a" if v is None else f"{v:+.4f}" for v in values)


def _cost_cell(cell: Mapping[str, Any], field: str, decimals: int) -> str:
    entry = cell["cost"].get(field)
    if entry is None:
        return "n/a"
    if field == "params" and not entry["std"]:
        return f"{entry['mean']:.0f}"
    return _fmt(entry, decimals)


def _group_markdown(group: Mapping[str, Any]) -> List[str]:
    reference = group["reference"]
    variants = group["variants"]
    metric_names = list(variants[0]["metrics"]) if variants else []
    has_throughput = any("throughput_images_per_s" in v["cost"] for v in variants)
    lines = [f"## {group['experiment']} on {group['dataset']}", "",
             f"Reference variant: `{reference}`. Epochs: {group['epochs']}.", ""]
    lines += [f"Note: {note}." for note in group["notes"]]
    if group["notes"]:
        lines.append("")
    lines += ["Accuracy, mean +- std over seeds:", ""]
    header = ["variant", "n_seeds", "params", "s/epoch"] + (["img/s"] if has_throughput else []) \
        + metric_names
    rows = []
    for v in variants:
        rows.append([v["variant"], str(v["n_seeds"]), _cost_cell(v, "params", 0),
                     _cost_cell(v, "seconds_per_epoch", 2)]
                    + ([_cost_cell(v, "throughput_images_per_s", 1)] if has_throughput else [])
                    + [_fmt(v["metrics"][m]) for m in metric_names])
    lines += _table(header, rows)
    lines += ["", f"Paired difference against `{reference}` (variant minus reference, seed by seed):", ""]
    paired_rows = []
    for v in variants:
        if v["paired"] is None:
            continue
        for m in metric_names:
            p = v["paired"][m]
            paired_rows.append([v["variant"], m, str(p["n_pairs"]),
                                ",".join(str(s) for s in p["seeds"]),
                                "n/a" if p["mean"] is None else f"{p['mean']:+.4f}",
                                _signed(p["diffs"])])
    if paired_rows:
        lines += _table(["variant", "metric", "n_pairs", "seeds", "mean diff", "per-seed diffs"],
                        paired_rows)
    elif reference in [v["variant"] for v in variants]:
        lines.append("No paired differences: no variant besides the reference has an ok cell.")
    else:
        lines.append("No paired differences: the reference has no ok cell here.")
    if group["experiment"] == "exp2":
        lines += ["", "Learned bandwidths (layer values are means over seeds and heads):", ""]
        band_rows = []
        for v in variants:
            b = v["bandwidths"]
            if b["status"] != "extracted":
                band_rows.append([v["variant"], b["status"], str(b["n_seeds"]), "-", "-", "-", "-"])
                continue
            layers = b["layers"]
            band_rows.append([
                v["variant"], b["mode"], str(b["n_seeds"]),
                ", ".join(f"{layer['mean']:.3f}" for layer in layers),
                f"{min(layer['min'] for layer in layers):.3f}",
                f"{max(layer['max'] for layer in layers):.3f}",
                "yes" if all(layer["inside_clip_range"] for layer in layers) else "NO"])
        lines += _table(["variant", "mode", "n_seeds", "per-layer mean", "min", "max",
                         "inside clip range"], band_rows)
    if "ranking" in group:
        ranking = group["ranking"]
        lines += ["", f"Ranking of the non-reference variants by `{ranking['metric']}` "
                      f"(the test split never ranks; ties: {ranking['tie_break']}):", ""]
        if ranking["rows"]:
            lines += _table(["rank", "config", "n_seeds", "num_steps", "select accuracy"],
                            [[str(r["rank"]), r["config"], str(r["n_seeds"]),
                              "n/a" if r["num_steps"] is None else str(r["num_steps"]),
                              _fmt({"mean": r["select_accuracy_mean"],
                                    "std": r["select_accuracy_std"]})] for r in ranking["rows"]])
        else:
            lines.append("No variant besides the reference to rank.")
    return lines + [""]


def render_markdown(summary: Mapping[str, Any]) -> str:
    """The report as markdown (ASCII only)."""
    counts = summary["counts"]
    lines = ["# Mean shift report", "", "Roots:"]
    lines += [f"- {root}" for root in summary["roots"]]
    lines += ["", f"Cells: {counts['ok']} ok, {counts['failed']} failed, {counts['missing']} missing.",
              "", f"Std convention: {summary['std_convention']}.", "",
              f"Claims: {summary['claims']}", ""]
    for group in summary["groups"]:
        lines += _group_markdown(group)
    lines += ["## Failed and missing cells", ""]
    if not summary["failed"] and not summary["missing"]:
        lines += ["None.", ""]
    for item in summary["failed"]:
        tail = " ".join(item["error_tail"].split())
        lines.append(f"- failed {_key_text(tuple(item['key']))}: {item['attempts']} attempt(s), "
                     f"returncode {item['returncode']}, cell dir {item['cell_dir']}, "
                     f"error tail: {tail or 'none'}")
    for key in summary["missing"]:
        lines.append(f"- missing {_key_text(tuple(key))}: in the manifest, no row")
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------
# writing and the command line


def dumps_strict(summary: Mapping[str, Any]) -> str:
    """``summary`` as strict JSON: a NaN or infinity raises instead of becoming a token."""
    return json.dumps(summary, indent=2, allow_nan=False) + "\n"


def write_report(summary: Mapping[str, Any], out_dir: Path) -> Tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.md`` into ``out_dir``; never overwrite either."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = (out_dir / SUMMARY_NAME, out_dir / MARKDOWN_NAME)
    existing = [p for p in paths if p.exists()]
    if existing:
        raise FileExistsError(f"{existing[0]} exists: a report is never overwritten")
    for path, text in zip(paths, (dumps_strict(summary), render_markdown(summary))):
        with open(path, "x", encoding="utf-8") as handle:
            handle.write(text)
    return paths


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate mean shift sweep roots into summary.json and summary.md.")
    parser.add_argument("--root", action="append", required=True,
                        help="a sweep root holding rows.jsonl (repeatable)")
    parser.add_argument("--out", required=True, help="output directory for the report")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    summary = build_summary([Path(r) for r in args.root])
    json_path, md_path = write_report(summary, Path(args.out))
    counts = summary["counts"]
    logger.info(f"[report] {counts['ok']} ok, {counts['failed']} failed, "
                f"{counts['missing']} missing cells -> {json_path} and {md_path}")
    if counts["failed"] or counts["missing"]:
        logger.warning(f"[report] {counts['failed']} failed and {counts['missing']} missing "
                       "cells are listed in the report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
