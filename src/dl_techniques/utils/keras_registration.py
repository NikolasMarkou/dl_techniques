"""Shared Keras registration helper: package-qualified key plus a legacy ``Custom>`` alias.

Every ``@keras.saving.register_keras_serializable`` call site under ``src/`` moves here.
Historically the tree registered classes with a *bare* decorator, which produces a key that
is independent of the defining module (``Custom>ClassName``); two classes sharing a name in
different packages therefore claim the same key and whichever module imports last silently
wins. The v2 guide (``research/2026_keras_custom_models_instructions_v2.md`` :sec:`2.2`)
mandates an explicit ``package=`` to close that hole.

Renaming a key is checkpoint-affecting: every ``.keras`` archive written before the rename
stores the *old* key in its ``config.json`` and Keras refuses to deserialize it if that key
no longer resolves. This module therefore registers each object under **both** keys -- the
stable package-qualified one (which new saves write) and, where the bare class name is not
shared with another registered class, the pre-migration ``Custom>ClassName`` one (which
existing archives read). See :sec:`6.3` of the same guide for the migration-path rule, and
``docs/`` / the shipping migration note for the reader-facing record.

Measured behaviour of the mechanism (2026-08-29, keras 3.x, this repository):

- ``keras.saving.get_registered_name(cls)`` returns the **package-qualified** key, so a
  freshly saved archive records the new name.
- ``keras.saving.get_registered_object("Custom>X")`` still returns the class, so a legacy
  archive loads with ``max|delta| = 0.0`` against the pre-migration output.
- **Control**: with ``legacy_alias=False`` the same legacy archive is REFUSED with
  ``TypeError: ... could not be deserialized properly``. The alias is load-bearing, not
  decorative.
- Plain functions (not only classes) register and alias identically.

Example:
    >>> from dl_techniques.utils.keras_registration import register_dl_technique
    >>> @register_dl_technique("dl_techniques.layers.attention.multi_head")
    ... class MultiHeadAttention(keras.layers.Layer):
    ...     ...
"""

from typing import Any, Callable, TypeVar

import keras

# ---------------------------------------------------------------------

__all__ = ["LEGACY_ALIAS_PREFIX", "AliasCollisionError", "register_dl_technique"]

#: Prefix Keras assigns when ``register_keras_serializable`` is called without a package.
#: Every archive written before this migration stores its custom classes under this prefix.
LEGACY_ALIAS_PREFIX = "Custom"

T = TypeVar("T", bound=Callable[..., Any])

# ---------------------------------------------------------------------


class AliasCollisionError(RuntimeError):
    """Raised when two distinct objects would claim the same legacy ``Custom>`` alias.

    This is the failure mode the migration exists to eliminate. Keras itself would accept
    the second write and silently overwrite the first, making the winner a function of
    import order; refusing at definition time makes the collision impossible to miss.

    The fix is never to let one side win by accident: pass ``legacy_alias=False`` on *both*
    sides of the duplicate-name pair, so neither claims the legacy key and each is reachable
    only through its own package-qualified key.
    """

# ---------------------------------------------------------------------


def _same_definition(existing: Any, obj: Any) -> bool:
    """Report whether ``existing`` and ``obj`` are the same definition re-executed.

    A module imported twice under two names (or reloaded by a test harness) produces two
    distinct objects that are nonetheless the *same* source definition. That is not a
    collision and must not raise.

    Args:
        existing: object currently bound to the legacy alias key.
        obj: object that wants to claim the same key.

    Returns:
        ``True`` when the two are the identical object, or carry the same ``__module__``
        and ``__qualname__``; ``False`` otherwise -- in which case they are genuinely two
        different definitions fighting over one key.
    """
    if existing is obj:
        return True
    return (
        getattr(existing, "__module__", None) == getattr(obj, "__module__", object())
        and getattr(existing, "__qualname__", None) == getattr(obj, "__qualname__", object())
    )

# ---------------------------------------------------------------------


def register_dl_technique(package: str, legacy_alias: bool = True) -> Callable[[T], T]:
    """Register a class or function under ``package``, optionally aliasing the legacy key.

    :param package: Package-qualified string handed to
        :func:`keras.saving.register_keras_serializable`. The registered key becomes
        ``f"{package}>{obj.__name__}"``. Derive it from the defining module's dotted path;
        for ``dl_techniques.models`` the family directories (``vision``, ``language``, ...)
        and the subfamily containers (``image_restoration``, ``keypoints``,
        ``super_resolution``, ``sam``) are stripped, because those are a filing decision and
        not a namespace -- they have already been reshuffled once (2026-08-24) and a key
        derived from them would have broken every archive at that moment.
    :type package: str
    :param legacy_alias: When ``True`` (default), also bind ``Custom>{obj.__name__}`` in
        :func:`keras.saving.get_custom_objects` to the same object, so pre-migration
        ``.keras`` archives keep loading. Pass ``False`` for any name shared by two
        registered objects -- aliasing both sides of such a pair would recreate, in the
        legacy namespace, the exact import-order collision this migration removes.
    :type legacy_alias: bool

    :raises AliasCollisionError: if ``legacy_alias`` is ``True`` and the legacy key is
        already bound to a *different* definition. Re-executing the same definition (module
        reloaded, or imported under a second name) is idempotent and does not raise.

    :returns: A decorator that registers its argument and returns it unchanged.
    :rtype: Callable

    .. note::
       The alias is written inside the decorator, at the instant the object is defined, so
       it exists before any load can be attempted. A one-shot registry-walking shim cannot
       offer that guarantee here: most of this package's ``__init__.py`` files are empty, so
       hundreds of registered objects are never imported by ``import dl_techniques`` alone
       and a walk at import time would find nothing to alias.
    """
    if not isinstance(package, str) or not package:
        raise ValueError(
            f"package must be a non-empty string naming the defining module, got {package!r}"
        )

    # DECISION plan-2026-08-29T141252-168933da/D-001
    # The alias is written HERE, inside the decorator, and not by a one-shot shim that walks
    # keras.saving.get_custom_objects() once at import time back-filling `Custom>X` for every
    # `pkg>X`. That shim is simpler and was measured to work -- but only for classes that have
    # already been imported. Most of this package's __init__.py files are empty (layers/,
    # models/, metrics/, callbacks/, constraints/ are all 0 lines), so 393 of the tree's
    # registration sites are never reached by `import dl_techniques`, and a caller doing
    # `keras.models.load_model(path)` without first importing the defining module would find
    # nothing aliased. Registering at definition time is the only ordering-independent option.
    # Do NOT replace this with a post-hoc registry walk.
    def decorator(obj: T) -> T:
        obj = keras.saving.register_keras_serializable(package=package)(obj)
        if legacy_alias:
            key = f"{LEGACY_ALIAS_PREFIX}>{obj.__name__}"
            custom_objects = keras.saving.get_custom_objects()
            existing = custom_objects.get(key)
            # DECISION plan-2026-08-29T141252-168933da/D-002
            # Raise, do not overwrite and do not warn-and-skip. Keras itself accepts the second
            # write silently, which makes the winner a function of import order -- the exact
            # defect this migration exists to remove; a warning would be swallowed by the
            # thousands of lines TF prints at import. Raising costs an ImportError on a genuine
            # collision, which is the intended blast radius. `_same_definition` keeps a module
            # re-imported under a second name (or reloaded by a harness) from tripping it.
            if existing is not None and not _same_definition(existing, obj):
                raise AliasCollisionError(
                    f"legacy alias {key!r} is already claimed by "
                    f"{getattr(existing, '__module__', '?')}.{getattr(existing, '__qualname__', existing)}; "
                    f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', obj)} cannot also claim it. "
                    f"Two registered objects share the bare name {obj.__name__!r}: pass "
                    f"legacy_alias=False on BOTH of them so neither owns the legacy key."
                )
            custom_objects[key] = obj
        return obj

    return decorator

# ---------------------------------------------------------------------
