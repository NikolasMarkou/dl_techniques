"""Repo guard: no Keras-2 ``input_shape=`` / ``input_dim=`` kwarg on a built-in layer.

R-038 root cause RD-3 (plan ``plan-2026-08-22T035419-a11304c8``). Keras 3 emits

    Do not pass an `input_shape`/`input_dim` argument to a layer. When using
    Sequential models, prefer using an `Input(shape)` object as the first layer
    in the model instead.

whenever a built-in Keras layer is constructed with the Keras-2 shape kwarg. The
warning is not cosmetic: the kwarg is silently discarded on a bare layer, so a
test that believes it declared an input shape declared nothing at all, and under
``-W error::UserWarning`` (R-038's target configuration) every such call site
aborts its test.

The measured population when this guard was written was **23 call sites in 13
files under ``tests/``** and **0 under ``src/``**; all 23 were repaired in the
same commit. This guard exists so the population cannot silently regrow.

``Embedding(input_dim=...)`` is exempt: on ``Embedding`` the name means the
vocabulary size and is the correct modern API.
"""

import ast
import pathlib
from typing import List, Tuple

# Built-in Keras layer names that accept (and warn about) the legacy kwarg.
_KERAS_BUILTIN_LAYERS = {
    "Dense", "Conv1D", "Conv2D", "Conv3D", "Conv1DTranspose", "Conv2DTranspose",
    "DepthwiseConv1D", "DepthwiseConv2D", "SeparableConv1D", "SeparableConv2D",
    "Embedding", "Flatten", "Activation", "LSTM", "GRU", "SimpleRNN",
    "BatchNormalization", "LayerNormalization", "Dropout", "Reshape", "Lambda",
    "MaxPooling1D", "MaxPooling2D", "AveragePooling1D", "AveragePooling2D",
    "GlobalAveragePooling1D", "GlobalAveragePooling2D",
}

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _scan(root: pathlib.Path) -> List[str]:
    """Every ``LayerName(input_shape=...)`` / ``(input_dim=...)`` call under ``root``."""
    offenders: List[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            if name not in _KERAS_BUILTIN_LAYERS:
                continue
            for kw in node.keywords:
                if kw.arg not in ("input_shape", "input_dim"):
                    continue
                # `Embedding(input_dim=V)` is the vocabulary size, not a shape.
                if kw.arg == "input_dim" and name == "Embedding":
                    continue
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{node.lineno} {name}({kw.arg}=...)")
    return offenders


def _guard_self_pair() -> Tuple[int, int]:
    """RED-proof control: the scanner must find the synthetic offender below."""
    source = (
        "import keras\n"
        "keras.layers.Dense(4, input_shape=(2,))\n"
        "keras.layers.Embedding(10, 4, input_dim=10)\n"
    )
    tmp = pathlib.Path(__file__).with_name("__guard_probe_not_a_test.py")
    tmp.write_text(source, encoding="utf-8")
    try:
        found = _scan(tmp.parent)
        hits = [f for f in found if tmp.name in f]
        return len(hits), len(found)
    finally:
        tmp.unlink()


def test_the_scanner_actually_detects_the_kwarg() -> None:
    """Without this, a scanner that finds nothing would pass the guard below."""
    hits, _total = _guard_self_pair()
    assert hits == 1, (
        f"the AST scanner failed its own positive control: expected exactly one "
        f"synthetic offender, found {hits}. The guard below is meaningless."
    )


def test_no_test_passes_a_keras2_shape_kwarg_to_a_builtin_layer() -> None:
    offenders = _scan(_REPO_ROOT / "tests")
    assert offenders == [], (
        "Keras-2 `input_shape=`/`input_dim=` kwarg passed to a built-in Keras "
        "layer. Keras 3 discards it and warns; under `-W error::UserWarning` the "
        "test aborts. Use `keras.Input(shape=...)` as the first Sequential "
        "element, or drop the kwarg on a bare layer and build it explicitly:\n  "
        + "\n  ".join(offenders)
    )


def test_no_library_code_passes_a_keras2_shape_kwarg_to_a_builtin_layer() -> None:
    offenders = _scan(_REPO_ROOT / "src")
    assert offenders == [], (
        "library code passes the Keras-2 shape kwarg to a built-in layer:\n  "
        + "\n  ".join(offenders)
    )
