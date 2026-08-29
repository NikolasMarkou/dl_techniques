"""Repo-wide guard: no two classes may claim the same Keras serialization key.

`@keras.saving.register_keras_serializable()` with no arguments registers a class under
the module-INDEPENDENT key ``Custom>{ClassName}``. Two classes with the SAME class name
therefore claim the SAME key, and the one imported LAST silently overwrites the other.
Nothing warns.

SINCE 2026-08-29 no site under `src/` is spelled that way any more: all 744 of them go
through `dl_techniques.utils.keras_registration.register_dl_technique`, which mints a
package-qualified key and binds the old ``Custom>{ClassName}`` as an ALIAS to the same
object so pre-migration `.keras` archives keep loading. See `MIGRATIONS.md`. This guard is
unchanged in intent and still the right one: an alias is a second key for the SAME object,
which this checker's identity comparison correctly does not count as a collision, and the
four duplicate-name pairs the alias is deliberately WITHHELD from are exactly the pairs
that would otherwise collide in the legacy namespace.

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

The `ConvUNextStem` row is NOT a registry-key collision and never was -- read it as a
PYTHON NAME collision only. Measured on 2026-08-14 (Keras 3.8.0,
plan-2026-08-14-0e3d792d step 1): the two classes held two DISTINCT keys,
``dl_techniques.bias_free_denoisers>ConvUNextStem`` and
``dl_techniques.convunext>ConvUNextStem``, and both resolved simultaneously -- neither
ever shadowed the other. What actually collided was the bare identifier `ConvUNextStem`
in two modules, which is confusing to read but harmless to load. That is also why the
row lists a `package=`-qualified pair while its three neighbours list bare `Custom>`
keys. It is moot now regardless: the two classes were MERGED into one on 2026-08-14,
the `models.convunext.model` twin was deleted, and the survivor lives in
`models/vision/convunext/model.py`. It kept ``package="dl_techniques.bias_free_denoisers"``
so its key stayed byte-stable until 2026-08-29, when the user confirmed there are no
checkpoints -- the exemption's entire basis -- and it was normalized to
``package="dl_techniques.models.convunext.model"`` along with the tree's other 33 ad-hoc
strings (repo-root `MIGRATIONS.md`).

Two of the OTHER right-hand modules have since been DELETED and no longer exist anywhere
in the repo: `models.modern_bert.components` and `models.cliffordnet.confidence_denoiser`
(plan-2026-08-10-3649c19e). The table is kept verbatim as the historical record of why
the `package=` arguments below exist -- do not go looking for those two modules, and do
not drop a `package=` just because its collision partner is gone: the checker below is
derived from what is importable TODAY, so it is the thing that decides, not this table.

Fix: give each colliding class an explicit, distinct ``package=``.

CORRECTED 2026-08-29. This paragraph used to end: "This is backward compatible -- a
`.keras` file records ``module`` + ``class_name`` alongside ``registered_name``, and Keras
falls back to the module path when the registered key is absent (verified empirically
before the rename)." That fallback claim is FALSE for a class whose key moved. Measured
with a control on Keras 3.8.0 (plan-2026-08-29T141252-168933da step 1): an archive written
under ``Custom>X`` and reloaded after X was re-keyed with the alias SUPPRESSED is REFUSED
with ``TypeError``, not silently resolved through its module path. Changing a registered
key IS checkpoint-affecting; what makes it survivable is the legacy alias, not a fallback.

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
    # pass vacuously.
    #
    # The floor was ``> 100`` against a measured population of 728, i.e. ~7x headroom --
    # so seven of every eight registrations could have vanished without this noticing.
    # Measured 2026-08-29 after the registration migration: **1452 keys** (each aliased
    # object holds two, the qualified key and its legacy ``Custom>`` alias). The floor is
    # re-derived on the same 80%-of-population rule the sibling guard in
    # `test_models/test_package_api_contract.py` uses: ``int(0.8 * 1452) == 1161``. A
    # fifth of the tree's registrations may disappear before this guard is allowed to call
    # itself alive; anything tighter would trip on an ordinary refactor and say nothing
    # more about whether the walk still reaches the tree.
    assert len(registry) > 1161, (
        f"only {len(registry)} objects registered (measured 1452 on 2026-08-29) -- the "
        f"module walk likely failed, so this guard would pass vacuously"
    )

    assert not collisions, (
        "Two classes registered under the same Keras serialization key. The one imported "
        "LAST silently wins, so saving/loading the other is broken and depends on import "
        "order. Give each an explicit distinct `package=` via "
        "`register_dl_technique(...)`, and pass `legacy_alias=False` on BOTH sides if "
        "they share a bare class name (see MIGRATIONS.md):\n"
        + "\n".join(
            f"  {key}\n" + "".join(f"       {c}\n" for c in sorted(classes))
            for key, classes in sorted(collisions.items())
        )
    )
