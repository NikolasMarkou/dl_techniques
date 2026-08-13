"""Readers for the COMMITTED upstream MobileCLIP2 reference material.

The reference files live at ``research/mobileclip2_reference/`` and are Apple's
own release artefacts, committed verbatim (see that directory's ``README.md``).
This module is the single place that knows where they are and how to read them,
so no test hard-codes the path twice.

``mobileclip2.py`` is PyTorch/``timm`` code and CANNOT be imported here — neither
``torch`` nor ``timm`` is installed — so it is read with :mod:`ast`.

Interface contract
------------------
``reference_dir()`` -> ``pathlib.Path``
    Absolute path of ``research/mobileclip2_reference/``. Raises
    ``FileNotFoundError`` if the directory is missing (the oracle is required,
    never optional: a silently skipped oracle is not an oracle).

``load_supplied_json(name)`` -> ``dict``
    Parse ``model_configs/<name>.json``. Raises ``FileNotFoundError`` if absent.

``parse_supplied_mci_model_args()`` -> ``Dict[str, dict]``
    Keys ``'mci3'`` / ``'mci4'``. Each value is the ``model_args = dict(...)``
    literal of the matching ``fastvit_mciN`` factory in ``mobileclip2.py``,
    translated into THIS repo's ``MCI_VARIANTS`` vocabulary, plus the two fields
    that live outside that literal:
    ``stem_use_scale_branch`` (from ``convolutional_stem_timm``) and
    ``norm_layer`` (the ``norm_layer=`` class name, mapped to a string).
    Raises ``AssertionError`` (with the offending construct named) if the source
    has a shape this reader does not understand — never silently returns a
    partial answer.
"""

import ast
import functools
import json
import pathlib
from typing import Any, Dict

#: ``tests/test_models/test_mobile_clip_v2/`` -> repository root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

_REFERENCE_DIRNAME = pathlib.Path("research") / "mobileclip2_reference"

#: `norm_layer=` class name in the supplied source -> this repo's string key.
_NORM_LAYER_NAMES = {
    'LayerNormChannel': 'layer_norm',
    'BatchNorm2d': 'batch_norm',
    'nn.BatchNorm2d': 'batch_norm',
}

#: The `partial(...)` positional callable that marks a positional embedding.
_POS_EMB_CLASS = 'RepConditionalPosEnc'


def reference_dir() -> pathlib.Path:
    """Absolute path of the committed reference directory."""
    path = _REPO_ROOT / _REFERENCE_DIRNAME
    if not path.is_dir():
        raise FileNotFoundError(
            f"The committed upstream reference material is missing: {path}. "
            f"It is a required test oracle, not an optional extra — restore it "
            f"from the Apple MobileCLIP2 release rather than skipping the tests "
            f"that read it."
        )
    return path


def load_supplied_json(name: str) -> Dict[str, Any]:
    """Read one open_clip model config, e.g. ``'MobileCLIP2-S0'``."""
    path = reference_dir() / "model_configs" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Supplied model config not found: {path}")
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _supplied_source_path() -> pathlib.Path:
    path = reference_dir() / "mobileclip2.py"
    if not path.is_file():
        raise FileNotFoundError(f"Supplied source not found: {path}")
    return path


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(
        f"the supplied mobileclip2.py defines no function named {name!r}"
    )


def _model_args_call(func: ast.FunctionDef) -> ast.Call:
    """The ``model_args = dict(...)`` call inside a ``fastvit_mciN`` factory."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if 'model_args' not in targets:
            continue
        value = node.value
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
            and value.func.id == 'dict', (
            f"{func.name}: `model_args` is assigned "
            f"{ast.dump(value)[:120]}, expected a `dict(...)` call"
        )
        return value
    raise AssertionError(f"{func.name}: no `model_args = dict(...)` assignment")


def _pos_embs(node: ast.AST, where: str):
    """Translate the ``pos_embs=`` tuple into ``None`` / spatial-shape tuples."""
    assert isinstance(node, (ast.Tuple, ast.List)), (
        f"{where}: pos_embs is {type(node).__name__}, expected a tuple"
    )
    out = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and element.value is None:
            out.append(None)
            continue
        assert isinstance(element, ast.Call), (
            f"{where}: pos_embs entry is {ast.dump(element)[:100]}, expected "
            f"None or partial({_POS_EMB_CLASS}, spatial_shape=...)"
        )
        assert isinstance(element.func, ast.Name) and element.func.id == 'partial', (
            f"{where}: pos_embs entry is not a `partial(...)` call"
        )
        assert element.args and isinstance(element.args[0], ast.Name) \
            and element.args[0].id == _POS_EMB_CLASS, (
            f"{where}: pos_embs partial does not wrap {_POS_EMB_CLASS}"
        )
        shapes = [
            ast.literal_eval(kw.value) for kw in element.keywords
            if kw.arg == 'spatial_shape'
        ]
        assert len(shapes) == 1, (
            f"{where}: pos_embs partial has {len(shapes)} `spatial_shape` "
            f"keywords, expected exactly 1"
        )
        out.append(tuple(shapes[0]))
    return tuple(out)


def _stem_use_scale_branch(tree: ast.Module) -> bool:
    """The single ``use_scale_branch=`` value of ``convolutional_stem_timm``.

    The supplied file monkey-patches every MCi3/MCi4 stem with this function, so
    its value IS the port's ``stem_use_scale_branch``. All three MobileOneBlocks
    must agree; disagreement is a shape this reader refuses to summarise.
    """
    func = _function_def(tree, 'convolutional_stem_timm')
    values = [
        kw.value.value
        for node in ast.walk(func) if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == 'use_scale_branch' and isinstance(kw.value, ast.Constant)
    ]
    assert values, (
        "convolutional_stem_timm passes no `use_scale_branch=` at all — the "
        "port's stem_use_scale_branch field has no counterpart in the source"
    )
    assert len(set(values)) == 1, (
        f"convolutional_stem_timm's MobileOneBlocks disagree on "
        f"use_scale_branch: {values}; the port models it as ONE field"
    )
    return bool(values[0])


def _uses_the_patched_stem(func: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'convolutional_stem_timm'
        for node in ast.walk(func)
    )


@functools.lru_cache(maxsize=1)
def parse_supplied_mci_model_args() -> Dict[str, Dict[str, Any]]:
    """Parse ``fastvit_mci3`` / ``fastvit_mci4`` out of the supplied source."""
    tree = ast.parse(
        _supplied_source_path().read_text(encoding='utf-8'),
        filename='mobileclip2.py',
    )
    stem_use_scale_branch = _stem_use_scale_branch(tree)

    parsed: Dict[str, Dict[str, Any]] = {}
    for variant in ('mci3', 'mci4'):
        func = _function_def(tree, f'fastvit_{variant}')
        assert _uses_the_patched_stem(func), (
            f"fastvit_{variant} no longer calls convolutional_stem_timm, so its "
            f"stem's use_scale_branch cannot be read from that function"
        )
        keywords = {kw.arg: kw.value for kw in _model_args_call(func).keywords}

        missing = {
            'layers', 'embed_dims', 'mlp_ratios', 'se_downsamples',
            'downsamples', 'pos_embs', 'token_mixers', 'lkc_use_act',
            'norm_layer',
        } - set(keywords)
        assert not missing, (
            f"fastvit_{variant}'s model_args is missing {sorted(missing)}"
        )

        norm_node = keywords['norm_layer']
        norm_name = ast.unparse(norm_node)
        assert norm_name in _NORM_LAYER_NAMES, (
            f"fastvit_{variant}: unknown norm_layer {norm_name!r}; this reader "
            f"knows {sorted(_NORM_LAYER_NAMES)}"
        )

        row = {
            'layers': tuple(ast.literal_eval(keywords['layers'])),
            'embed_dims': tuple(ast.literal_eval(keywords['embed_dims'])),
            'mlp_ratios': tuple(
                float(v) for v in ast.literal_eval(keywords['mlp_ratios'])
            ),
            'se_downsamples': tuple(
                ast.literal_eval(keywords['se_downsamples'])),
            'downsamples': tuple(ast.literal_eval(keywords['downsamples'])),
            'pos_embs': _pos_embs(keywords['pos_embs'], f'fastvit_{variant}'),
            'token_mixers': tuple(ast.literal_eval(keywords['token_mixers'])),
            'lkc_use_act': bool(ast.literal_eval(keywords['lkc_use_act'])),
            'stem_use_scale_branch': stem_use_scale_branch,
            'norm_layer': _NORM_LAYER_NAMES[norm_name],
        }
        parsed[variant] = row
    return parsed
