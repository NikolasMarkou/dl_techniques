"""Repo-wide guard: no two classes may claim the same Keras serialization key.

`@keras.saving.register_keras_serializable()` with no arguments registers a class under
``"Custom>{ClassName}"``. Two classes with the SAME class name therefore claim the SAME
key, and the one imported LAST silently overwrites the other. Nothing warns.

The consequence is a real serialization-correctness bug, not merely a test annoyance:
saving a model containing the shadowed class and loading it back resolves the key to the
OTHER class, which then fails to deserialize (or, worse, deserializes into the wrong layer
with a compatible-looking config). Because it depends on import order, it reproduces only
when both modules are imported -- so it is invisible to a single-file test run and shows up
as an "unrelated flake" in the full suite.

That is exactly how it was found: `TestConvBlock::test_serialization` passed alone and
failed in the full `tests/test_layers/` run. Four collisions existed:

    Custom>ConvBlock       layers.standard_blocks        vs layers.yolo12_blocks
    Custom>ConvUNextStem   models.bias_free_denoisers    vs models.convunext.model
    Custom>ByteTokenizer   layers.blt_blocks             vs models.modern_bert.components
    Custom>CoverageMetric  metrics.probabilistic_...     vs models.cliffordnet...

Fix: give each colliding class an explicit, distinct ``package=``. This is backward
compatible -- a `.keras` file records ``module`` + ``class_name`` alongside
``registered_name``, and Keras falls back to the module path when the registered key is
absent (verified empirically before the rename).

This test walks every module under `dl_techniques` and asserts no key is claimed twice.
"""

import importlib
import pkgutil

import pytest

import dl_techniques
from keras.src.saving import object_registration


def _walk_all_modules():
    """Import every dl_techniques module, tolerating ones that fail to import.

    A module that cannot be imported cannot register anything, so skipping it is safe for
    this test's purpose. `onerror` is required: at least one unrelated module currently
    raises at import time, and without it `walk_packages` would abort the whole walk and
    silently shrink this guard's subject set to almost nothing.
    """
    for module in pkgutil.walk_packages(
        dl_techniques.__path__, "dl_techniques.", onerror=lambda name: None
    ):
        try:
            importlib.import_module(module.name)
        except Exception:  # noqa: BLE001 - see docstring
            continue


def test_no_duplicate_keras_serialization_keys():
    """No two classes may register under the same Keras custom-object key.

    Detects the collision by watching the registry ACROSS imports: a key whose value
    CHANGES identity as modules are imported has been overwritten. Comparing the final
    registry against itself cannot see this -- by then the loser is already gone.
    """
    registry = object_registration.GLOBAL_CUSTOM_OBJECTS
    collisions = {}

    for module in pkgutil.walk_packages(
        dl_techniques.__path__, "dl_techniques.", onerror=lambda name: None
    ):
        before = dict(registry)
        try:
            importlib.import_module(module.name)
        except Exception:  # noqa: BLE001
            continue
        for key, cls in registry.items():
            previous = before.get(key)
            if previous is not None and previous is not cls:
                collisions.setdefault(key, set()).update({
                    f"{previous.__module__}.{previous.__name__}",
                    f"{cls.__module__}.{cls.__name__}",
                })

    # Guard the guard: if the walk registered almost nothing, the assertion below would
    # pass vacuously. The repo registers ~700 objects; anything near zero means the walk
    # broke and this test is not actually testing anything.
    assert len(registry) > 100, (
        f"only {len(registry)} objects registered -- the module walk likely failed, so "
        f"this guard would pass vacuously"
    )

    assert not collisions, (
        "Two classes registered under the same Keras serialization key. The one imported "
        "LAST silently wins, so saving/loading the other is broken and depends on import "
        "order. Give each an explicit distinct `package=` in "
        "@keras.saving.register_keras_serializable(...):\n"
        + "\n".join(
            f"  {key}\n" + "".join(f"       {c}\n" for c in sorted(classes))
            for key, classes in sorted(collisions.items())
        )
    )
