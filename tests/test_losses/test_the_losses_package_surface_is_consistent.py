"""The ``losses`` package says one thing about itself everywhere.

Three censuses, each DERIVED (executed or AST-read), none a hand list
(plan-2026-11-11T131349-13b28b37 D-006, D-007):

1. Every ``keras.losses.Loss`` defined in a ``losses/*.py`` module that can be built
   with no arguments must survive ``from_config(get_config())``. ``ClusteringLoss``
   rejected the emitted ``reduction`` and ``ClassificationFocalLoss`` (a YOLOv12
   sub-loss, not exported from the package) hard-coded ``reduction="none"`` and then
   received it a second time; the other two sub-losses need constructor arguments and
   are pinned parametrically in ``test_yolo12_multitask_loss.py``.
2. Every ``register_dl_technique`` key in ``losses/*.py`` equals
   ``dl_techniques.losses.<module>``. Seven OmniPoint classes used a per-class key.
3. Every name ``losses/__init__.py`` imports is in its ``__all__`` (it imported
   ``match_mask_axis``, which the README and ``sam2_video_loss`` call module-only).
"""

import ast
import importlib
import pathlib

import keras
import pytest

import dl_techniques.losses as losses_pkg

LOSSES_DIR = pathlib.Path(losses_pkg.__file__).parent
MIN_SWEPT = 90  # 97 built at 8ad64d2bd (7 need arguments); far fewer means the sweep is blind


def _every_loss_class_in_the_package():
    """Every ``keras.losses.Loss`` subclass DEFINED in a ``losses/*.py`` module.

    The sweep walks modules, not ``losses.__all__``: the YOLOv12 sub-losses are
    deliberately not exported from the package yet are the very classes that failed.
    """
    classes = {}
    for path in sorted(LOSSES_DIR.glob("*.py")):
        if path.stem == "__init__":
            continue
        module = importlib.import_module(f"dl_techniques.losses.{path.stem}")
        for name, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and issubclass(obj, keras.losses.Loss)
                and obj.__module__ == module.__name__
            ):
                classes[(module.__name__, name)] = obj
    return classes


def _default_constructible_losses():
    swept, skipped = [], []
    for (module, name), cls in _every_loss_class_in_the_package().items():
        try:
            swept.append((f"{module.rsplit('.', 1)[-1]}.{name}", cls()))
        except TypeError:
            skipped.append(name)
    return swept, skipped


def test_every_default_constructible_loss_round_trips_through_its_own_config():
    swept, skipped = _default_constructible_losses()
    assert len(swept) >= MIN_SWEPT, (
        f"the census only built {len(swept)} losses ({len(skipped)} need arguments); "
        "it has gone blind"
    )
    failures = []
    for name, instance in swept:
        try:
            restored = type(instance).from_config(instance.get_config())
        except Exception as error:  # noqa: BLE001 - the census reports every class
            failures.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if type(restored) is not type(instance):
            failures.append(f"{name}: round trip returned {type(restored).__name__}")
    assert not failures, "from_config(get_config()) fails for:\n" + "\n".join(failures)


def _registered_keys():
    """Yield ``(module, class_or_function_name, package_string)`` from the AST."""
    for path in sorted(LOSSES_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and getattr(decorator.func, "id", None) == "register_dl_technique"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    yield path.stem, node.name, decorator.args[0].value


def test_every_registered_loss_key_is_its_module_path():
    rows = list(_registered_keys())
    assert len(rows) >= 100, f"the census saw only {len(rows)} registrations"
    wrong = [
        f"{module}.{name}: {package!r}"
        for module, name, package in rows
        if package != f"dl_techniques.losses.{module}"
    ]
    assert not wrong, "registry key is not dl_techniques.losses.<module>:\n" + "\n".join(wrong)


def test_the_package_init_imports_nothing_it_does_not_export():
    tree = ast.parse((LOSSES_DIR / "__init__.py").read_text(encoding="utf-8"))
    imported = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert len(imported) > 100, "the census read no imports"
    assert imported - set(losses_pkg.__all__) == set()
