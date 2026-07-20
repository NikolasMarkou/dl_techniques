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

VALUE DRIFT (the third direction, added later)
----------------------------------------------
Names matching is not enough. The `create_*` functions do::

    params.update(info['optional_params'])   # activations/factory.py:503-504

so a registry default is **injected**, not merely documented: if the registry says
`mode='trainable'` and the constructor says `mode='deterministic'`, the factory caller gets
`'trainable'`. A registry default that disagrees with its constructor is therefore
behavioral, not cosmetic.

`test_registry_optional_defaults_match_constructor_defaults` compares the VALUES. Every
`(entry, param)` pair lands in exactly one of four branches:

* **NON-COMPARABLE** - the constructor declares no default (`inspect.Parameter.empty`), or the
  param is not in the resolved signature at all (it reaches the class via `**kwargs`).
  There is no ctor default to disagree with, so this is not drift. Counted, not asserted.
  Example: `embedding:modern_bert_embeddings.initializer_range`.
* **SENTINEL** - the ctor default is `None` and the registry default is not. The ctor resolves
  the `None` elsewhere; a raw `==` would call this drift when the resolved value is identical.
  Requires a recorded entry in `SENTINEL_RESOLUTIONS`, and asserts the registry value still
  equals that recorded resolution.
* **EXEMPT** - the values genuinely differ and the divergence is deliberate (an alias entry that
  specializes the class, e.g. `ffn:reglu` = GLUFFN with `activation='relu'`). Requires an entry
  in `INTENTIONAL_OVERRIDES`, and asserts the registry value still equals the **pinned** expected
  value -- so editing an already-exempt default still reds.
* **COMPARE** - everything else: the values must be equal, with `bool` never satisfying an
  `int`/`float` (Python says `True == 1`; a registry `True` against a ctor `1` is drift).

WHAT THIS GUARD DOES **NOT** SEE (stated, not hidden)
-----------------------------------------------------
1. **Defaults that are `None` on BOTH sides but resolved later, per-mode, inside `build()`.**
   The canonical case is `mixtures:rbf.gamma_init` (`mixtures/radial_basis_function.py:282,313,329`),
   whose real value is fixed per `output_mode` during `build()`. Both sides say `None`, so the
   comparison passes trivially while saying nothing. Catching this needs a mode/shape context the
   registry has not got; resolving it by instantiating every layer was considered and rejected,
   because instantiation cannot reach a per-`output_mode` `build()` resolution either.
2. **Whether a recorded `SENTINEL_RESOLUTIONS` value still matches its resolver.** If
   `adaptive_softmax.py:181` changed `1e-7` to `1e-9` while both the registry and the table below
   still said `1e-7`, this guard would stay green on real drift. The table is hand-maintained;
   that is the price of not instantiating. It is bounded to the 3 entries listed.
3. **Wrapper defaults established by anything other than a literal `kwargs.setdefault(...)`.**
   See `_wrapper_kwarg_setdefaults`. A default set by an `if 'k' not in kwargs:` block or a
   computed expression falls back to the wrapped class's own default, i.e. this guard's behavior
   before wrapper resolution existed.
"""

import ast
import importlib
import inspect
import textwrap

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


# ---------------------------------------------------------------------
# VALUE drift
# ---------------------------------------------------------------------

_EMPTY = inspect.Parameter.empty

# (factory, type_name, param) -> (resolved_value, reason)
#
# The constructor's default is `None`, and the constructor resolves that `None` to a concrete
# value somewhere else. The registry declares that concrete value. Those two agree in effect,
# so this is NOT drift -- but only a recorded, machine-checked resolution makes that claim
# auditable instead of a silent skip. Each reason names the file:line that does the resolving,
# which is the line to re-read if this entry ever reds.
SENTINEL_RESOLUTIONS = {
    ("activations", "adaptive_softmax", "eps"): (
        1e-7,
        "AdaptiveTemperatureSoftmax.__init__ takes eps: Optional[float] = None and resolves it "
        "as `self.eps = eps if eps is not None else 1e-7` "
        "(layers/activations/adaptive_softmax.py:181). The registry declares the resolved value "
        "explicitly so `get_activation_info()` reports something useful; effect is identical.",
    ),
    ("logic", "circuit_depth", "gate_entropy_coefficient"): (
        0.0,
        "The ctor default is None because gate_entropy_coefficient supersedes the deprecated "
        "load_balance_coefficient; the resolver returns `float(load_balance_coefficient or 0.0)` "
        "when both are None (layers/logic/neural_circuit.py:82), i.e. 0.0. Registry declares 0.0.",
    ),
    ("logic", "neural_circuit", "gate_entropy_coefficient"): (
        0.0,
        "Same class and same resolver as logic:circuit_depth "
        "(layers/logic/neural_circuit.py:82): both None -> float(None or 0.0) -> 0.0.",
    ),
}

# (factory, type_name, param) -> (expected_registry_value, reason)
#
# The values genuinely differ and that is the design: the registry entry is an ALIAS door onto a
# more general class, and the alias specializes a default. Pinned to the expected value (not just
# to the key) so that editing an already-exempt default still fails -- an exemption is a record of
# one specific divergence, not a permanent licence for that parameter.
INTENTIONAL_OVERRIDES = {
    ("ffn", "bilinear", "activation"): (
        "linear",
        "ffn:bilinear is GLUFFN specialized to a linear gate; its own registry description "
        "(layers/ffn/factory.py:226) says \"alias of GLUFFN with activation='linear'\". GLUFFN's "
        "own ctor default is 'swish', which belongs to the ffn:glu door.",
    ),
    ("ffn", "reglu", "activation"): (
        "relu",
        "ffn:reglu is GLUFFN specialized to a ReLU gate; its registry description "
        "(layers/ffn/factory.py:211) says \"alias of GLUFFN with activation='relu'\". Same "
        "alias-family pattern as ffn:bilinear.",
    ),
    ("activations", "hierarchical_routing", "mode"): (
        "trainable",
        "activations:hierarchical_routing is the TRAINABLE door onto RoutingProbabilitiesLayer; "
        "the sibling key activations:routing_probabilities maps the SAME class with "
        "mode='deterministic' (layers/activations/factory.py:214-235), which is where the bare "
        "ctor default 'deterministic' (layers/activations/routing_probabilities.py:329-334) "
        "belongs. layers/activations/README.md:23 documents the mapping in a table, and the only "
        "non-factory consumer, ProbabilityOutput._create_strategy_layer "
        "(layers/activations/probability_output.py:201-204), independently constructs "
        "RoutingProbabilitiesLayer(mode='trainable') for this probability_type. Same alias-family "
        "pattern as the two GLUFFN entries above.",
    ),
}


def _wrapper_kwarg_setdefaults(func):
    """Literal ``kwargs.setdefault('name', value)`` defaults established by a wrapper function.

    # DECISION plan-2026-07-20T191713-52a15234/D-005
    A registry entry whose 'class' is a wrapper function has THREE default layers, not two:
    the wrapped class's own ctor default, then whatever the wrapper forces, then the registry.
    `create_zigzag_window_attention` does `kwargs.setdefault("use_relative_position_bias", False)`
    (layers/attention/window_attention.py:762) while its sibling `create_grid_window_attention`
    sets it True (:724) -- so the registry's `False` for attention:window_zigzag is CORRECT even
    though `WindowAttention.__init__` says `True`.

    DO NOT "fix" a red here by adding an INTENTIONAL_OVERRIDES entry. That silences a correct
    value permanently: a later edit of the wrapper's setdefault to `True`, with the registry left
    at `False`, would then never red -- reinstalling the exact blind spot this guard exists to
    close, one layer further down. Model the chain instead. See decisions.md D-005.

    Only literal, top-level `kwargs.setdefault` calls are understood. Anything else (an
    `if 'k' not in kwargs:` block, a computed value) is not seen, and the wrapped class's default
    is used -- the same answer this guard gave before wrapper resolution existed.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return {}

    var_kw = {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind == param.VAR_KEYWORD
    }
    forced = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) != 2:
            continue
        callee = node.func
        if not isinstance(callee, ast.Attribute) or callee.attr != "setdefault":
            continue
        if not isinstance(callee.value, ast.Name) or callee.value.id not in var_kw:
            continue
        try:
            key = ast.literal_eval(node.args[0])
            value = ast.literal_eval(node.args[1])
        except ValueError:
            continue  # not a literal -- fall back to the class default
        if isinstance(key, str):
            forced[key] = value
    return forced


def _effective_defaults(target):
    """param name -> the default a caller gets when the factory passes nothing for it.

    For a CLASS: its ``__init__`` defaults. For a wrapper FUNCTION: the wrapper's own named
    defaults, plus the defaults of the class named by its ``return_annotation`` for params the
    wrapper only forwards via ``**kwargs``, plus anything the wrapper forces via
    ``kwargs.setdefault``. Params with no default map to ``inspect.Parameter.empty``.

    A wrapper whose return annotation is a string (PEP 563) or not a class contributes no class
    defaults, so its forwarded params stay NON-COMPARABLE -- graceful, but blind; see the module
    docstring.
    """
    signature = inspect.signature(
        target.__init__ if inspect.isclass(target) else target
    )
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    defaults = {
        name: param.default
        for name, param in signature.parameters.items()
        if param.kind in kinds
    }
    if inspect.isclass(target):
        return defaults

    wrapped = signature.return_annotation
    if inspect.isclass(wrapped):
        for name, param in inspect.signature(wrapped.__init__).parameters.items():
            if name not in defaults and param.kind in kinds:
                defaults[name] = param.default
    defaults.update(_wrapper_kwarg_setdefaults(target))
    return defaults


def _defaults_equal(registry_value, ctor_default):
    """Value equality that refuses Python's bool/int and bool/float conflation.

    `True == 1` and `False == 0.0` are both true in Python, so a naive `==` would let a registry
    `True` pass against a ctor `1`. That is real drift: the caller gets a bool where the class
    documents an int. Containers compare by value, never by identity, so a mutable default
    introduced later is still compared meaningfully (none exists across all 730 defaults today).
    """
    if isinstance(registry_value, bool) != isinstance(ctor_default, bool):
        return False
    if isinstance(registry_value, (list, dict, set)) or isinstance(
        ctor_default, (list, dict, set)
    ):
        return type(registry_value) is type(ctor_default) and registry_value == ctor_default
    try:
        return bool(registry_value == ctor_default)
    except Exception:  # a default whose __eq__ is not well-behaved
        return registry_value is ctor_default


NON_COMPARABLE, SENTINEL, EXEMPT, MATCH, DRIFT = (
    "NON_COMPARABLE",
    "SENTINEL",
    "EXEMPT",
    "MATCH",
    "DRIFT",
)


def _classify(factory, type_name, param, registry_value, ctor_default):
    """Sort one (entry, param) pair into exactly one branch. Pure; no assertions.

    NOTE THE ORDER: values are compared BEFORE any exemption table is consulted. An exemption is
    reachable only once the values actually disagree, so a table entry can never swallow a value
    that matches -- which is what keeps a mutated production default visible and lets
    `test_no_stale_exemptions` detect a divergence that has gone away.
    """
    if ctor_default is _EMPTY:
        return NON_COMPARABLE
    if ctor_default is None and registry_value is not None:
        return SENTINEL
    if _defaults_equal(registry_value, ctor_default):
        return MATCH
    if (factory, type_name, param) in INTENTIONAL_OVERRIDES:
        return EXEMPT
    return DRIFT


def _classified_pairs():
    """Yield (factory, type_name, param, registry_value, ctor_default, branch) for every pair."""
    for factory, type_name, target, info in ENTRIES:
        defaults = _effective_defaults(target)
        for param, registry_value in (info.get("optional_params") or {}).items():
            ctor_default = defaults.get(param, _EMPTY)
            yield (
                factory,
                type_name,
                param,
                registry_value,
                ctor_default,
                _classify(factory, type_name, param, registry_value, ctor_default),
            )


@pytest.mark.parametrize("factory,type_name,target,info", ENTRIES, ids=IDS)
def test_registry_optional_defaults_match_constructor_defaults(
    factory, type_name, target, info
):
    """VALUE drift: a registry default that disagrees with the ctor is what the caller GETS.

    `create_*` does `params.update(info['optional_params'])` before applying caller kwargs, so
    the registry default wins over the class default. This is the check the name-only guards
    above cannot make.
    """
    optional_params = info.get("optional_params")
    assert optional_params is None or isinstance(optional_params, dict), (
        f"{factory} registry entry '{type_name}' has a non-dict 'optional_params' "
        f"({type(optional_params).__name__}); the registry shape is "
        f"{{param: default}} and every create_* function assumes it."
    )

    defaults = _effective_defaults(target)
    target_name = getattr(target, "__name__", target)
    failures = []

    for param, registry_value in (optional_params or {}).items():
        ctor_default = defaults.get(param, _EMPTY)
        branch = _classify(factory, type_name, param, registry_value, ctor_default)
        key = (factory, type_name, param)

        if branch in (NON_COMPARABLE, MATCH):
            continue

        if branch == SENTINEL:
            if key not in SENTINEL_RESOLUTIONS:
                failures.append(
                    f"  {param}: registry={registry_value!r} but {target_name} defaults to None. "
                    f"If the ctor resolves that None to {registry_value!r} anyway, record it in "
                    f"SENTINEL_RESOLUTIONS[{key!r}] = ({registry_value!r}, 'why, incl. the "
                    f"file:line that resolves it'). If it resolves to something else, the "
                    f"registry default is wrong and callers of create_{factory}_* silently get "
                    f"{registry_value!r}."
                )
                continue
            resolved, reason = SENTINEL_RESOLUTIONS[key]
            if not _defaults_equal(registry_value, resolved):
                failures.append(
                    f"  {param}: registry={registry_value!r} no longer equals the recorded "
                    f"resolution {resolved!r} of {target_name}'s None default. Either the "
                    f"registry default changed (and now diverges) or the resolver did (and "
                    f"SENTINEL_RESOLUTIONS is stale). Recorded reason: {reason}"
                )
            continue

        if branch == EXEMPT:
            expected, reason = INTENTIONAL_OVERRIDES[key]
            if not _defaults_equal(registry_value, expected):
                failures.append(
                    f"  {param}: registry={registry_value!r} but this divergence is allowlisted "
                    f"as {expected!r}. An exemption pins ONE specific value; it is not a licence "
                    f"for this parameter. Confirm the new value is still deliberate and update "
                    f"INTENTIONAL_OVERRIDES[{key!r}], or revert. Recorded reason: {reason}"
                )
            continue

        failures.append(
            f"  {param}: registry default {registry_value!r} != {target_name} default "
            f"{ctor_default!r}. create_{factory}_* INJECTS the registry default "
            f"(params.update(optional_params)), so every caller who does not pass {param} gets "
            f"{registry_value!r}, not {ctor_default!r}. Three legal remedies: (1) fix the "
            f"registry default to {ctor_default!r}; (2) if the ctor default is the wrong one, "
            f"fix the ctor; (3) if this entry is an ALIAS that specializes the class on purpose, "
            f"record INTENTIONAL_OVERRIDES[{key!r}] = ({registry_value!r}, 'why') -- a written "
            f"reason is required, an undocumented exemption is a silenced bug."
        )

    assert not failures, (
        f"{factory} registry entry '{type_name}' ({target_name}) declares "
        f"{len(failures)} optional_params default(s) that disagree with the constructor:\n"
        + "\n".join(failures)
    )


def test_no_stale_exemptions():
    """Anti-rot: an exemption for a divergence that no longer exists must FAIL, not linger.

    The hit set is built by running the real classifier over the real registries here -- not
    populated at import, and not accumulated as a side effect of the parametrized test above
    (which would make this pass or fail depending on how the suite was selected, and would let a
    `-k` run silently invent staleness). A key counts as exercised only when classification
    actually lands on SENTINEL/EXEMPT for it, i.e. only while the divergence it documents is
    still real. Fix the registry to agree with its ctor and the corresponding entry must be
    deleted -- that is the point.
    """
    exercised = {
        (factory, type_name, param)
        for factory, type_name, param, _, _, branch in _classified_pairs()
        if branch in (SENTINEL, EXEMPT)
    }

    stale = sorted(
        set(SENTINEL_RESOLUTIONS) - exercised
    ), sorted(set(INTENTIONAL_OVERRIDES) - exercised)
    assert not stale[0], (
        f"SENTINEL_RESOLUTIONS has {len(stale[0])} entry/entries that no longer describe a live "
        f"registry divergence: {stale[0]}. Either the registry entry, the parameter, or the "
        f"ctor's None default is gone. Delete the stale key -- an exemption nobody can reach is "
        f"documentation that has stopped being true."
    )
    assert not stale[1], (
        f"INTENTIONAL_OVERRIDES has {len(stale[1])} entry/entries that no longer describe a live "
        f"registry divergence: {stale[1]}. The registry and the constructor now agree (or the "
        f"entry/param is gone), so the exemption is obsolete. Delete the stale key -- leaving it "
        f"would silently pre-authorize a FUTURE divergence nobody has reviewed."
    )

    for key, (_, reason) in {**SENTINEL_RESOLUTIONS, **INTENTIONAL_OVERRIDES}.items():
        assert reason and reason.strip(), (
            f"exemption {key} has an empty reason. An exemption without a written justification "
            f"is a silenced bug, not a resolved one."
        )
