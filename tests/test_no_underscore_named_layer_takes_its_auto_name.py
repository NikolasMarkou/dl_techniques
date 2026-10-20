"""No Layer or Model whose class name starts with ``_`` is left to its Keras auto-name.

Keras names an unnamed layer after its class, so ``_ContextPoisoner`` became
``_context_poisoner``. TensorFlow refuses a root name scope that starts with
``_`` (``^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$``), and Keras has already pushed
the scope when the refusal comes, so the PARENT's scope stays on Keras'
process-global name-scope stack: every layer built later in the process gets
that prefix in its weight paths. D-219 of plan-2026-10-14T062616-81938e81 found
it as two weight-order guards that failed only after the TripSE file had run;
D-220 measured that a parent called EAGERLY leaks with no error at all, and an
underscore-named top-level model leaks from ``fit``.

The census (AST, ``src/`` and ``tests/``) holds every such class to one of:

* ``PublicAutoName`` among its bases (``tests/layer_naming.py``, the shared
  form for test-private layers);
* its own ``__init__`` supplies a default name that does not start with ``_``
  (``kwargs.setdefault("name", "...")``, a ``name="..."`` parameter default, or
  ``super().__init__(name="...")``);
* every construction passes ``name=``, and the class is otherwise only
  subclassed or type-checked (an abstract base that is never built itself),
  counted over the defining module and every module that imports the class.

The subprocess arm keeps the hazard itself measured: if a Keras or TensorFlow
upgrade stops leaking, that arm fails and the census can be retired on purpose
instead of guarding nothing.
"""

import ast
import functools
import json
import os
import pathlib
import subprocess
import sys
import textwrap
import warnings

import keras
import pytest

from tests.layer_naming import PublicAutoName

ROOT = pathlib.Path(__file__).resolve().parents[1]
_KERAS_ROOTS = {"Layer", "Model", "Sequential", "Functional"}
_MIXIN = "PublicAutoName"

#: Abstract bases whose own ``__init__`` raises, so no instance survives to be
#: built or called (a name scope is pushed only then). The census cannot see a
#: raise, so each entry is checked by execution in
#: ``test_a_never_constructed_class_still_refuses_construction``.
NEVER_CONSTRUCTED = {
    "src/dl_techniques/models/vision/image_restoration/doc_scanner/u2net_blocks.py::_ResidualUBlock",
}


def _base_names(node):
    out = []
    for b in node.bases:
        if isinstance(b, ast.Attribute):
            out.append(b.attr)
        elif isinstance(b, ast.Name):
            out.append(b.id)
    return out


def _resolve_import(path, node):
    """The file an ``ImportFrom`` reads, or ``None`` outside src/ and tests/."""
    parts = (node.module or "").split(".") if node.module else []
    if node.level:
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
    elif parts and parts[0] in ("dl_techniques", "train", "applications"):
        base = ROOT / "src"
    elif parts and parts[0] == "tests":
        base = ROOT
    else:
        return None
    target = base.joinpath(*parts) if parts else base
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _default_name(cls):
    """The literal default name the class's own ``__init__`` supplies, if any."""
    for item in cls.body:
        if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
            continue
        args = item.args
        positional = args.posonlyargs + args.args
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        for arg, default in list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults)):
            if arg.arg == "name" and isinstance(default, ast.Constant) and isinstance(default.value, str):
                return default.value
        for node in ast.walk(item):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (node.func.attr == "setdefault" and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == "name"
                    and isinstance(node.args[1], ast.Constant)):
                return node.args[1].value
            if node.func.attr == "__init__":
                for kw in node.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        return kw.value.value
    return None


@functools.lru_cache(maxsize=1)
def census():
    """Every underscore-named Layer/Model class, and the ones left to their auto-name.

    :return: ``(population, offenders)``: ``population`` maps
        ``"<repo path>::<class>"`` to the form that makes it safe (or
        ``None``); ``offenders`` lists the keys with ``None``, each with the
        constructions that pass no ``name=``.
    """
    trees, classes = {}, {}
    for top in ("src", "tests"):
        for path in sorted((ROOT / top).rglob("*.py")):
            with warnings.catch_warnings():  # a scanned file's own invalid escapes
                warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
                tree = ast.parse(path.read_text(encoding="utf-8"))
            path = path.resolve()
            trees[path] = tree
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.setdefault(node.name, []).append((path, node))

    layerish = set(_KERAS_ROOTS)
    for name, defs in classes.items():
        for _, node in defs:
            if any(isinstance(b, ast.Attribute) and isinstance(b.value, ast.Attribute)
                   and b.value.attr == "layers" for b in node.bases):
                layerish.add(name)
    grown = True
    while grown:
        grown = False
        for name, defs in classes.items():
            if name not in layerish and any(
                    set(_base_names(node)) & layerish for _, node in defs):
                layerish.add(name)
                grown = True

    # Which local names in each file refer to a class defined in which file.
    imported = {}  # (defining path, class name) -> [(importing path, local name)]
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                source = _resolve_import(path, node)
                if source is None:
                    continue
                for alias in node.names:
                    if alias.name.startswith("_"):
                        imported.setdefault((source, alias.name), []).append(
                            (path, alias.asname or alias.name))

    def is_layer(node):
        return bool(set(_base_names(node)) & layerish) or any(
            isinstance(b, ast.Attribute) and isinstance(b.value, ast.Attribute)
            and b.value.attr == "layers" for b in node.bases)

    population, offenders = {}, []
    for name, defs in sorted(classes.items()):
        if not name.startswith("_"):
            continue
        for path, node in defs:
            if not is_layer(node):
                continue
            key = f"{path.relative_to(ROOT)}::{name}"
            default = _default_name(node)
            if _MIXIN in _base_names(node):
                population[key] = "PublicAutoName"
                continue
            if default is not None and not default.startswith("_"):
                population[key] = f"default name {default!r}"
                continue
            unnamed, indirect = [], []
            for where, local in [(path, name)] + imported.get((path, name), []):
                tree = trees[where]
                harmless = set()
                for n in ast.walk(tree):
                    if isinstance(n, ast.ClassDef):
                        harmless.update(id(b) for b in n.bases)
                    elif isinstance(n, (ast.arg, ast.AnnAssign)) and n.annotation is not None:
                        harmless.update(id(a) for a in ast.walk(n.annotation))
                    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns is not None:
                        harmless.update(id(a) for a in ast.walk(n.returns))
                    elif isinstance(n, ast.Call):
                        if isinstance(n.func, ast.Name) and n.func.id == local:
                            harmless.add(id(n.func))
                            if not any(k.arg == "name" for k in n.keywords):
                                unnamed.append(f"{where.relative_to(ROOT)}:{n.lineno}")
                        if (isinstance(n.func, ast.Name) and n.func.id in ("isinstance", "issubclass")
                                and len(n.args) == 2):
                            harmless.update(id(a) for a in ast.walk(n.args[1]))
                for n in ast.walk(tree):
                    if (isinstance(n, ast.Name) and n.id == local and isinstance(n.ctx, ast.Load)
                            and id(n) not in harmless):
                        indirect.append(f"{where.relative_to(ROOT)}:{n.lineno}")
            if not unnamed and not indirect:
                population[key] = "every construction names it"
                continue
            if key in NEVER_CONSTRUCTED:
                population[key] = "never constructed"
                continue
            population[key] = None
            offenders.append(f"{key}: unnamed {unnamed}, passed around {indirect}")
    return population, offenders


def test_every_underscore_named_layer_has_a_name_that_does_not_start_with_an_underscore():
    _, offenders = census()
    assert offenders == [], (
        "Layer/Model classes whose Keras auto-name would start with '_' (a name "
        "TensorFlow refuses as a root scope, which leaves the parent's scope on "
        "Keras' global stack). Add tests.layer_naming.PublicAutoName as the "
        "first base, or give the class a default name:\n" + "\n".join(offenders))


def test_the_census_sees_the_classes_it_was_written_for():
    # A census that stops finding its subjects passes vacuously; one subject
    # per form, plus the population size.
    population, _ = census()
    assert population[
        "tests/test_layers/test_attention/test_tripse_attention.py::_ContextPoisoner"
    ] == "default name 'context_poisoner'"
    assert population[
        "src/dl_techniques/layers/attention/tripse_attention.py::_SEWeights"
    ] == "default name 'se_weights'"
    assert population[
        "src/dl_techniques/layers/heads/vision/factory.py::_DenseVectorHeadBase"
    ] == "every construction names it"
    assert population[
        "tests/test_models/test_precision_arm_oracle.py::_HealthyModel"
    ] == "PublicAutoName"
    # 108 at writing (79 PublicAutoName, 16 default names, 12 always named,
    # 1 never constructed); a drop below 100 means the class walk lost track.
    assert len(population) >= 100, len(population)


def test_a_never_constructed_class_still_refuses_construction():
    from dl_techniques.models.vision.image_restoration.doc_scanner.u2net_blocks import (
        _ResidualUBlock)

    assert set(NEVER_CONSTRUCTED) == {
        "src/dl_techniques/models/vision/image_restoration/doc_scanner/u2net_blocks.py::_ResidualUBlock"}
    with pytest.raises(ValueError, match="_ENCODER_LEVELS"):
        _ResidualUBlock(mid_channels=4, out_channels=4)


def test_public_auto_name_names_an_unnamed_instance_after_its_class():
    class _ProbeLayerForNaming(PublicAutoName, keras.layers.Layer):
        pass

    first, second = _ProbeLayerForNaming(), _ProbeLayerForNaming()
    assert first.name.startswith("probe_layer_for_naming")
    assert second.name != first.name and not second.name.startswith("_")
    assert _ProbeLayerForNaming(name="kept").name == "kept"


_LEAK_PROBE = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import keras, numpy as np
    from keras.src.backend.common import global_state
    from tests.layer_naming import PublicAutoName

    def stack():
        return [getattr(s, "name", s) for s in global_state.get_global_attribute("name_scope_stack") or []]

    def run(child_cls):
        class Outer(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.child = child_cls()
            def call(self, x):
                return self.child(x)
        Outer(name="outer")(np.ones((1, 3), "float32"))
        left = stack()
        global_state.set_global_attribute("name_scope_stack", [])
        return left

    class _Twice(keras.layers.Layer):
        def call(self, x):
            return 2.0 * x

    class _TwicePublic(PublicAutoName, keras.layers.Layer):
        def call(self, x):
            return 2.0 * x

    print(json.dumps({"underscore": run(_Twice), "public": run(_TwicePublic)}))
""")


def test_an_underscore_auto_name_still_leaks_the_parent_scope():
    # Positive control, in a fresh process because the leak is process-global.
    out = subprocess.run(
        [sys.executable, "-c", _LEAK_PROBE, str(ROOT)], capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3"),
        timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == {"underscore": ["outer"], "public": []}, result
