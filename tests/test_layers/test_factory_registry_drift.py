"""Repo-wide guard: a factory registry must not silently drop a real constructor argument.

The registry-based factories (`attention`, `ffn`, `embedding`, `activations`, `mixtures`,
`logic`, `sequence_pooling`) all share one shape::

    REGISTRY = {'type': {'class': Cls, 'required_params': [...], 'optional_params': {...}}}

and their `create_*` functions FILTER the caller's kwargs against that registry::

    valid = set(required_params) | set(optional_params)
    final = {k: v for k, v in params.items() if k in valid}

**So a constructor argument missing from the registry is SILENTLY DISCARDED.** No exception,
no warning: the layer is built with the class default and the caller believes their value
applied. Measured before this guard existed::

    create_embedding_layer('patch_2d', patch_size=4, embed_dim=32, flatten=False)
    -> layer.flatten is True          # the caller's False vanished

That is a silent-misconfiguration bug, and it is a DIFFERENT (and worse) failure mode than
the one the norms factory had, where a hand-maintained whitelist made the *validator*
reject a parameter the *builder* accepted -- that one at least failed loudly. An audit
found **27** silently-droppable parameters across 9 types in 4 factories.

The inverse direction (PHANTOM: a registry key the class does not accept) raises TypeError
at construction, so it is loud; it is still checked here, cheaply.

NOTE ON THE PREDECESSOR GUARD: `test_registry_keys_are_real_ctor_args`
(tests/test_layers/test_attention/test_attention_factory.py) checks ONLY the phantom
direction, and `pytest.skip`s on `**kwargs` -- so it could not see any of the 27. A guard
that skips its subjects is worse than no guard: it looks like coverage. This one skips
nothing and checks both directions.
"""

import importlib
import inspect

import pytest

# factory label -> (module path, registry attribute)
FACTORIES = {
    "activations": ("dl_techniques.layers.activations.factory", "ACTIVATION_REGISTRY"),
    "attention": ("dl_techniques.layers.attention.factory", "ATTENTION_REGISTRY"),
    "embedding": ("dl_techniques.layers.embedding.factory", "EMBEDDING_REGISTRY"),
    "ffn": ("dl_techniques.layers.ffn.factory", "FFN_REGISTRY"),
    "logic": ("dl_techniques.layers.logic.factory", "LOGIC_REGISTRY"),
    "mixtures": ("dl_techniques.layers.mixtures.factory", "MIXTURE_REGISTRY"),
    "sequence_pooling": (
        "dl_techniques.layers.sequence_pooling.factory",
        "SEQUENCE_POOLING_REGISTRY",
    ),
}

# Keras base-layer kwargs and non-parameters. These are handled by the factories
# separately (or by `keras.layers.Layer`), not declared per-type in the registries.
BASE_PARAMS = {
    "self",
    "kwargs",
    "name",
    "dtype",
    "trainable",
    "autocast",
    "activity_regularizer",
}


def _registry_entries():
    """Yield (factory, type_name, target, info) for every registered type."""
    for factory, (module_path, attr) in FACTORIES.items():
        registry = getattr(importlib.import_module(module_path), attr)
        for type_name, info in sorted(registry.items()):
            target = info.get("class")
            if target is None:
                continue
            yield factory, type_name, target, info


def _signature_params(target):
    """Named params of the target, plus whether it accepts ``**kwargs``.

    A registry entry's ``'class'`` may be a CLASS (inspect ``__init__``) or a factory
    FUNCTION (inspect the function itself) -- e.g. attention's ``'window'`` maps to
    ``create_grid_window_attention``. Inspecting the wrong one reports a whole signature's
    worth of phantom drift that does not exist.
    """
    signature = inspect.signature(
        target.__init__ if inspect.isclass(target) else target
    )
    named = {
        name
        for name, param in signature.parameters.items()
        if name not in BASE_PARAMS
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    accepts_var_kw = any(
        param.kind == param.VAR_KEYWORD for param in signature.parameters.values()
    )
    return named, accepts_var_kw


ENTRIES = list(_registry_entries())
IDS = [f"{f}:{t}" for f, t, _, _ in ENTRIES]


def test_the_guard_has_subjects():
    """Guard the guard: if the registries fail to import, everything below passes vacuously."""
    assert len(ENTRIES) > 50, (
        f"only {len(ENTRIES)} registry entries collected -- the factories likely failed to "
        f"import, so the drift tests below would pass while testing nothing"
    )


@pytest.mark.parametrize("factory,type_name,target,info", ENTRIES, ids=IDS)
def test_registry_declares_every_constructor_param(factory, type_name, target, info):
    """MISSING drift: a ctor param absent from the registry is SILENTLY DROPPED by create_*.

    This is the bug. Nothing raises -- the caller's value is filtered out and the class
    default is used instead.
    """
    named, _ = _signature_params(target)
    declared = set(info.get("required_params", [])) | set(
        info.get("optional_params", {})
    )
    missing = sorted(named - declared)
    assert not missing, (
        f"{factory} registry entry '{type_name}' ({getattr(target, '__name__', target)}) "
        f"does not declare {missing}, which its constructor accepts. create_{factory}_* "
        f"FILTERS kwargs against this registry, so a caller passing any of these has the "
        f"value SILENTLY DISCARDED and gets the class default. Add them to "
        f"'optional_params' with their real constructor defaults."
    )


@pytest.mark.parametrize("factory,type_name,target,info", ENTRIES, ids=IDS)
def test_registry_declares_no_param_the_target_rejects(factory, type_name, target, info):
    """PHANTOM drift: a registry key the target does not accept -> TypeError at build."""
    named, accepts_var_kw = _signature_params(target)
    if accepts_var_kw:
        # The target takes **kwargs, so it accepts anything; no phantom is possible.
        # This is NOT a skip of the subject -- the MISSING check above still applies to it.
        return
    declared = set(info.get("required_params", [])) | set(
        info.get("optional_params", {})
    )
    phantom = sorted(declared - named - BASE_PARAMS)
    assert not phantom, (
        f"{factory} registry entry '{type_name}' declares {phantom}, which "
        f"{getattr(target, '__name__', target)} does not accept -- construction would "
        f"raise TypeError."
    )
