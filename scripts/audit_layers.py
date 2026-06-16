#!/usr/bin/env python3
"""Read-only AST compliance scanner for dl_techniques custom Keras layers.

Standalone CLI reporter. Pure stdlib (ast, argparse, pathlib, json, sys).
NO dl_techniques imports. Static analysis only — never mutates, imports, or
executes any scanned file.

Classifies every ClassDef in each .py file under the target path and grades
ONLY concrete keras.layers.Layer subclasses against the mechanical HARD items
of the production-quality rubric (see findings/canonical-standard.md):

    register_decorator, get_config, compute_output_shape,
    super().build() as last statement of build(), no raw tf.* in call(),
    no print() calls.

GHOST constraint: `if self.built: return` is deliberately NOT checked, reported,
or graded anywhere in this scanner. Its absence is not a defect.
"""

import ast
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PATH = "src/dl_techniques/layers"


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #

def _attr_path(node: ast.AST) -> str:
    """Return the dotted attribute path for a Name/Attribute node ('' if other)."""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return ".".join(parts)
    # e.g. a call-result base — not resolvable to a dotted name
    parts.reverse()
    return ".".join(parts)


def _base_name(node: ast.AST) -> str:
    """Last component of a base-class expression (handles Name and Attribute)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):  # e.g. Generic[...] — rare for bases
        return _base_name(node.value)
    return ""


def _decorator_name(node: ast.AST) -> str:
    """Dotted path of a decorator (unwrapping a Call), '' if unresolvable."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _attr_path(node)
    return ""


def _methods(cls: ast.ClassDef) -> Dict[str, ast.AST]:
    """Map method name -> FunctionDef/AsyncFunctionDef for direct-body methods."""
    out: Dict[str, ast.AST] = {}
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[item.name] = item
    return out


def _has_abstractmethod(cls: ast.ClassDef) -> bool:
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in item.decorator_list:
                if _decorator_name(dec).split(".")[-1] == "abstractmethod":
                    return True
    return False


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

LAYER_BASES = {"Layer"}          # match by trailing component
MODEL_BASES = {"Model"}
ENUM_BASES = {"Enum", "IntEnum", "IntFlag", "Flag", "StrEnum"}
ABC_BASES = {"ABC", "ABCMeta"}


def _classify(cls: ast.ClassDef) -> str:
    base_names = [_base_name(b) for b in cls.bases]
    base_names = [b for b in base_names if b]

    decos = [_decorator_name(d).split(".")[-1] for d in cls.decorator_list]
    is_dataclass = "dataclass" in decos

    is_abstract = (
        any(b in ABC_BASES for b in base_names)
        or _has_abstractmethod(cls)
    )

    # Layer detection: trailing component is "Layer" (Layer, keras.layers.Layer,
    # keras.Layer). Not graded if abstract.
    is_layer = any(b in LAYER_BASES for b in base_names)
    is_model = any(b in MODEL_BASES for b in base_names)

    if is_layer and not is_abstract:
        return "CONCRETE-LAYER"
    if any(b in ENUM_BASES for b in base_names):
        return "ENUM"
    if is_dataclass:
        return "DATACLASS"
    if is_abstract:
        return "ABC"
    if is_model:
        return "MODEL"
    if is_layer and is_abstract:
        return "ABC"
    return "OTHER"


def _iter_classdefs(tree: ast.AST):
    """Yield ALL ClassDef nodes (top-level and nested)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node


# --------------------------------------------------------------------------- #
# Per-item checks (CONCRETE-LAYER only)
# --------------------------------------------------------------------------- #

def _has_register_decorator(cls: ast.ClassDef) -> bool:
    for dec in cls.decorator_list:
        if _decorator_name(dec).split(".")[-1] == "register_keras_serializable":
            return True
    return False


def _is_super_build_call(node: ast.AST) -> bool:
    """True if node is an expression statement calling super().build(...)."""
    if isinstance(node, ast.Expr):
        node = node.value
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "build":
        return False
    val = func.value
    return isinstance(val, ast.Call) and isinstance(val.func, ast.Name) \
        and val.func.id == "super"


def _super_build_last(build_fn: ast.AST) -> bool:
    body = list(getattr(build_fn, "body", []))
    # drop a leading docstring if it is the only-or-first statement
    if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        if len(body) > 1:
            body = body[1:]
    if not body:
        return False
    return _is_super_build_call(body[-1])


def _sublayer_build(build_fn: ast.AST) -> bool:
    """Any self.<x>.build(...) call inside the build body (INFO only)."""
    for node in ast.walk(build_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "build":
                val = node.func.value
                # self.<attr>.build(...)  -> val is Attribute on self
                if isinstance(val, ast.Attribute) and isinstance(
                        val.value, ast.Name) and val.value.id == "self":
                    return True
    return False


def _count_raw_tf(call_fn: ast.AST) -> int:
    """Count attribute-call expressions rooted at name `tf` inside call()."""
    count = 0
    for node in ast.walk(call_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "tf":
                count += 1
    return count


def _count_print(fn: ast.AST) -> int:
    count = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "print":
            count += 1
    return count


def _has_type_hints(fn: Optional[ast.AST]) -> bool:
    if fn is None:
        return True  # absent method does not fail the INFO item
    args = fn.args
    pos = list(args.posonlyargs) + list(args.args)
    # ignore 'self'/'cls'
    pos = [a for a in pos if a.arg not in ("self", "cls")]
    args_ok = all(a.annotation is not None for a in pos)
    ret_ok = fn.returns is not None
    return args_ok and ret_ok


def _grade_layer(cls: ast.ClassDef) -> Dict[str, Any]:
    methods = _methods(cls)
    build_fn = methods.get("build")
    call_fn = methods.get("call")
    init_fn = methods.get("__init__")

    items: Dict[str, Any] = {
        "register_decorator": _has_register_decorator(cls),
        "get_config": "get_config" in methods,
        "compute_output_shape": "compute_output_shape" in methods,
        "build_present": build_fn is not None,
        # super_build_last: None when build absent (N/A, not a FAIL)
        "super_build_last": (_super_build_last(build_fn)
                             if build_fn is not None else None),
        "sublayer_build": (_sublayer_build(build_fn)
                           if build_fn is not None else None),  # INFO
        "forward_raw_tf": (_count_raw_tf(call_fn) if call_fn is not None else 0),
        "print_call": sum(_count_print(m) for m in methods.values()),
        "type_hints": (_has_type_hints(init_fn) and _has_type_hints(call_fn)),
    }
    return items


def _failing_items(items: Dict[str, Any]) -> List[str]:
    """HARD failures for one CONCRETE-LAYER."""
    fails: List[str] = []
    if not items["register_decorator"]:
        fails.append("register_decorator")
    if not items["get_config"]:
        fails.append("get_config")
    if not items["compute_output_shape"]:
        fails.append("compute_output_shape")
    if items["build_present"] and items["super_build_last"] is False:
        fails.append("super_build_last")
    if items["forward_raw_tf"] > 0:
        fails.append("forward-raw-tf")
    if items["print_call"] > 0:
        fails.append("print-call")
    return fails


def _failing_items_model(items: Dict[str, Any]) -> List[str]:
    """HARD failures for one CONCRETE-MODEL (Level 2).

    Same universal mechanical items as a layer EXCEPT ``compute_output_shape``:
    Keras infers it for functional/composite models and subclassed models
    commonly (and acceptably) omit it, so its absence is NOT a model FAIL.
    ``super_build_last`` is still enforced when a ``build()`` exists.
    """
    fails: List[str] = []
    if not items["register_decorator"]:
        fails.append("register_decorator")
    if not items["get_config"]:
        fails.append("get_config")
    if items["build_present"] and items["super_build_last"] is False:
        fails.append("super_build_last")
    if items["forward_raw_tf"] > 0:
        fails.append("forward-raw-tf")
    if items["print_call"] > 0:
        fails.append("print-call")
    return fails


# --------------------------------------------------------------------------- #
# Per-file scan
# --------------------------------------------------------------------------- #

def scan_file(path: Path, grade_models: bool = False) -> Dict[str, Any]:
    rel = str(path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        return {
            "file": rel,
            "verdict": "PARSE-ERROR",
            "classes": [],
            "reasons": [f"parse-error: {type(exc).__name__}: {exc}"],
        }

    classes: List[Dict[str, Any]] = []
    concrete: List[Dict[str, Any]] = []
    kinds: List[str] = []

    for cls in _iter_classdefs(tree):
        kind = _classify(cls)
        kinds.append(kind)
        entry: Dict[str, Any] = {"name": cls.name, "kind": kind, "items": {}}
        if kind == "CONCRETE-LAYER":
            items = _grade_layer(cls)
            entry["items"] = items
            entry["failing_items"] = _failing_items(items)
            concrete.append(entry)
        elif kind == "MODEL" and grade_models:
            # Level 2: grade concrete keras.Model subclasses with the
            # model-applicable HARD subset (see _failing_items_model).
            entry["kind"] = "CONCRETE-MODEL"
            kinds[-1] = "CONCRETE-MODEL"
            items = _grade_layer(cls)
            entry["items"] = items
            entry["failing_items"] = _failing_items_model(items)
            concrete.append(entry)
        classes.append(entry)

    if not concrete:
        # N/A — annotate with the dominant non-layer kind for readability
        note = "no-classes"
        if kinds:
            if all(k == "ENUM" for k in kinds):
                note = "enum"
            elif all(k == "DATACLASS" for k in kinds):
                note = "dataclass-config"
            elif all(k == "ABC" for k in kinds):
                note = "abc"
            elif all(k == "MODEL" for k in kinds):
                note = "model"
            else:
                note = "non-layer"
        else:
            note = "pure-functions"
        return {
            "file": rel,
            "verdict": "N/A",
            "classes": classes,
            "reasons": [f"N/A ({note})"],
        }

    reasons: List[str] = []
    for entry in concrete:
        if entry["failing_items"]:
            reasons.append(
                f"{entry['name']}: " + ", ".join(entry["failing_items"]))

    verdict = "FAIL" if reasons else "PASS"
    return {
        "file": rel,
        "verdict": verdict,
        "classes": classes,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Walk + reporting
# --------------------------------------------------------------------------- #

def collect_files(target: Path) -> List[Path]:
    if target.is_file():
        return [target] if target.suffix == ".py" else []
    files: List[Path] = []
    for p in sorted(target.rglob("*.py")):
        if p.name == "__init__.py":
            continue
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    return files


def print_report(results: List[Dict[str, Any]]) -> None:
    name_w = max((len(r["file"]) for r in results), default=4)
    name_w = min(name_w, 90)
    print(f"{'FILE'.ljust(name_w)}  {'VERDICT':<11}  FAILING-ITEMS / NOTE")
    print("-" * (name_w + 13 + 30))
    for r in results:
        note = "; ".join(r["reasons"]) if r["reasons"] else ""
        fname = r["file"]
        if len(fname) > name_w:
            fname = "..." + fname[-(name_w - 3):]
        print(f"{fname.ljust(name_w)}  {r['verdict']:<11}  {note}")

    # --- summary block ---
    total = len(results)
    counts = {"PASS": 0, "FAIL": 0, "N/A": 0, "PARSE-ERROR": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    # aggregate per-HARD-item over CONCRETE-LAYER files only
    agg = {
        "register_decorator": [0, 0],
        "get_config": [0, 0],
        "compute_output_shape": [0, 0],
        "super_build_last": [0, 0],   # over build-present layers only
        "forward_raw_tf_clean": [0, 0],
        "print_clean": [0, 0],
    }
    concrete_layers = 0
    concrete_models = 0
    files_with_concrete = 0
    for r in results:
        clayers = [c for c in r["classes"] if c["kind"] == "CONCRETE-LAYER"]
        cmodels = [c for c in r["classes"] if c["kind"] == "CONCRETE-MODEL"]
        if clayers or cmodels:
            files_with_concrete += 1
        for c in clayers + cmodels:
            is_model = c["kind"] == "CONCRETE-MODEL"
            if is_model:
                concrete_models += 1
            else:
                concrete_layers += 1
            it = c["items"]
            agg["register_decorator"][1] += 1
            agg["register_decorator"][0] += int(it["register_decorator"])
            agg["get_config"][1] += 1
            agg["get_config"][0] += int(it["get_config"])
            # compute_output_shape is a HARD item for layers only.
            if not is_model:
                agg["compute_output_shape"][1] += 1
                agg["compute_output_shape"][0] += int(it["compute_output_shape"])
            if it["build_present"]:
                agg["super_build_last"][1] += 1
                agg["super_build_last"][0] += int(bool(it["super_build_last"]))
            agg["forward_raw_tf_clean"][1] += 1
            agg["forward_raw_tf_clean"][0] += int(it["forward_raw_tf"] == 0)
            agg["print_clean"][1] += 1
            agg["print_clean"][0] += int(it["print_call"] == 0)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files scanned      : {total}")
    print(f"  PASS                   : {counts.get('PASS', 0)}")
    print(f"  FAIL                   : {counts.get('FAIL', 0)}")
    print(f"  N/A                    : {counts.get('N/A', 0)}")
    print(f"  PARSE-ERROR            : {counts.get('PARSE-ERROR', 0)}")
    print(f"Files with concrete cls  : {files_with_concrete}")
    print(f"Concrete layers graded   : {concrete_layers}")
    print(f"Concrete models graded   : {concrete_models}")
    print()
    print("Per-HARD-item (over concrete layers + models):")
    for key, (ok, tot) in agg.items():
        pct = f"{(100.0 * ok / tot):5.1f}%" if tot else "  n/a "
        print(f"  {key:<24}: {ok:4d} / {tot:<4d}  {pct}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only AST compliance scanner for dl_techniques layers.")
    parser.add_argument("--path", default=DEFAULT_PATH,
                        help=f"file or directory to scan (default: {DEFAULT_PATH})")
    parser.add_argument("--subpackage", default=None,
                        help="scan src/dl_techniques/layers/<name>/ instead")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write machine-readable JSON report to this path")
    parser.add_argument("--models", action="store_true",
                        help="ALSO grade concrete keras.Model subclasses "
                             "(Level 2). Off by default (Level 1 = layers only).")
    args = parser.parse_args(argv)

    if args.subpackage:
        target = Path(DEFAULT_PATH) / args.subpackage
    else:
        target = Path(args.path)

    if not target.exists():
        print(f"error: path does not exist: {target}", file=sys.stderr)
        return 2

    files = collect_files(target)
    results = [scan_file(p, grade_models=args.models) for p in files]

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
