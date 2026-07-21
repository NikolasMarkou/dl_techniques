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

WRAPPER ENTRIES: two entries (`attention:window`, `attention:window_zigzag`) register a factory
FUNCTION rather than a class. All three directions resolve such a target through ONE shared
helper, `_wrapper_probe`, which obtains ground truth EMPIRICALLY -- it builds the layer and
observes what happened:

* what the wrapper builds is `type()` of what it returned, not a `return_annotation`;
* what the wrapper FORCES is whatever differs between building the class directly and building
  it through the wrapper with the same arguments;
* what the wrapper PINS is whatever raises `TypeError: got multiple values for keyword
  argument '<name>'` when supplied.

This replaced ~120 lines of `ast` analysis that tried to infer the same three facts from syntax.
Three review passes found four defects in it, every one the same failure wearing a new costume:
confident wrongness about which statements run. Adopting a `setdefault` no caller reaches, or
calling a settable param unsettable, does not merely miss drift -- it certifies drift as GREEN,
which is strictly worse than a guard known to be absent. A probe cannot mis-model control flow
because it does not model control flow. See decisions.md D-010.

Resolving in only SOME directions is not a partial improvement either, it is a blind spot with a
coverage story attached: applying it to the VALUE test alone once left the MISSING test seeing a
3-parameter signature for a 23-parameter class, and nothing failed. See decisions.md D-008.

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

HOW MUCH IS ACTUALLY GUARDED (measured, not rounded up)
-------------------------------------------------------
The value check visits all **730** `(entry, param)` pairs, but visiting is not pinning. Census on
this tree::

    MATCH 719 | NON_COMPARABLE 5 | SENTINEL 3 | EXEMPT 3 | DRIFT 0
    ...of the 719 MATCHes, 231 are `None == None`

A `None == None` match asserts nothing: it records that neither side stated a value, not that they
agree on the resolved one. Subtracting them leaves **494 of 730 (68%) genuinely value-pinned**
(488 non-vacuous MATCH + 3 SENTINEL + 3 EXEMPT). Quote 494, not 730. An inflated coverage number
is the same defect as an undisclosed limit -- it makes a partial guard read as a total one, so the
next reader trusts it further than it has earned.

WHAT THIS GUARD DOES **NOT** SEE (stated, not hidden)
-----------------------------------------------------
1. **Defaults that are `None` on BOTH sides and resolved later -- 231 pairs, not one.** Both sides
   say `None`, so the comparison passes trivially. A conservative scan of the constructing class's
   source (`<param> is None` / `<param> or ...` / `if not <param>`) finds a visible downstream
   resolver for about **54** of the 231 (a looser scan that also counts `is not None` finds ~85) --
   so this is a several-dozen-member class, and the exact size depends on how tightly you match.
   `mixtures:rbf.gamma_init` (`mixtures/radial_basis_function.py:282,313,329`, resolved per
   `output_mode` during `build()`) is one INSTANCE of it, not the whole of it; others include
   `activations:adaptive_softmax.polynomial_coeffs` (resolves to a 5-element list at
   `adaptive_softmax.py:185`), `attention:{energy,gated}.head_dim`, and
   `logic:arithmetic.operation_types`. Catching these needs a mode/shape context the registry has
   not got; resolving them by instantiating every layer was considered and rejected, because
   instantiation cannot reach a per-`output_mode` `build()` resolution either.
2. **Whether a recorded `SENTINEL_RESOLUTIONS` value still matches its resolver.** If
   `adaptive_softmax.py:181` changed `1e-7` to `1e-9` while both the registry and the table below
   still said `1e-7`, this guard would stay green on real drift. The table is hand-maintained;
   that is the price of not instantiating. It is bounded to the 3 entries listed -- but the same
   silent-staleness mechanism applies, unbounded and unmonitored, to the ~54 in item 1.
3. **Wrapper behavior on any path the probe does not take.** `_wrapper_probe` builds each
   wrapper ONCE, with `_PROBE_ARGS`. A wrapper that forces a different default for a different
   argument (`if dim > 128: kwargs.setdefault(...)`) is measured on the probed path only -- which
   is honest for the two wrappers here (both are unconditional) but would not generalize to a
   branchy one. It is a bounded, single-path measurement, not a proof about all inputs.
4. **Wrapper overrides whose values cannot be compared across two constructions.** Only
   `_LITERAL_TYPES` are differenced. If a wrapper forced, say, a specific initializer OBJECT, the
   difference is invisible and the class default stands -- this guard's behavior before wrapper
   resolution existed. The alternative (comparing objects that compare by identity) invents
   drift on every entry, which is worse.
5. **A wrapper that silently OVERWRITES a caller's kwarg** (`kwargs['x'] = ...` before
   delegating). It raises nothing, so it is not a pin and the param stays in the MISSING set;
   the registry declaring it is not wrong, but the caller's value is still discarded and no
   direction here sees it. Fixtured as `_wrapper_overwrites_kwargs_entry` so the limit is
   recorded rather than assumed. None exists in this repo today.
6. **`required_params` VALUES.** Only their names are checked, in both directions. A required
   param whose registry entry disagrees with the ctor in any respect other than presence is out
   of scope by construction.
7. **Params a wrapper pins POSITIONALLY.** A positional pin raises the same TypeError only if
   the wrapper also forwards `**kwargs` into that call; a wrapper that binds the value some
   other way is unseen. None exists in this repo today.
"""

import collections
import functools
import importlib
import inspect

import pytest

from dl_techniques.layers.attention.factory import (
    ATTENTION_REGISTRY,
    create_attention_layer,
)
from dl_techniques.layers.attention.window_attention import (
    WindowAttention,
    create_grid_window_attention,
    create_zigzag_window_attention,
)

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


# ---------------------------------------------------------------------
# Wrapper resolution -- BY CONSTRUCTION, NOT BY STATIC ANALYSIS
#
# Two registry entries (attention:window, attention:window_zigzag) map a FACTORY FUNCTION,
# not a class. Everything below establishes what such a wrapper really does to its caller's
# arguments, and it is used by ALL THREE drift directions -- MISSING, PHANTOM and VALUE.
# Wiring it into only some of them was a real defect (decisions.md D-008); the resolution
# lives here, once, above every consumer.
# ---------------------------------------------------------------------

# Arguments sufficient to construct a wrapper target (and the class it builds) once, for
# probing. Keyed by parameter NAME, and needed only for parameters with no default. Small
# by construction: it covers the REQUIRED params of the 2 wrapper entries out of 97.
#
# This table cannot rot silently, which is what separates it from an exemption table: if a
# new wrapper entry needs a probe argument that is not here, the layer cannot be built and
# `test_every_wrapper_entry_is_probeable` reds by name. Staleness here is LOUD.
_PROBE_ARGS = {"dim": 32, "window_size": 4, "num_heads": 4}

# Value types whose equality is meaningful across two separate constructions. A Keras
# initializer/regularizer object compares by IDENTITY, so two constructions of the same class
# with the same argument produce two unequal objects; treating that as a wrapper override
# would invent drift out of nothing. Restricting the comparison to these types costs only the
# ability to see a wrapper that forces a non-literal default -- which then falls back to the
# class default, i.e. this guard's behavior before wrapper resolution existed.
_LITERAL_TYPES = (bool, int, float, str, bytes, type(None), tuple)


def _named_defaults(callable_obj):
    """``{param: default}`` for every keyword-settable param of a class ctor or a function.

    A param with no default maps to ``inspect.Parameter.empty``.
    """
    signature = inspect.signature(
        callable_obj.__init__ if inspect.isclass(callable_obj) else callable_obj
    )
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.kind in kinds
    }


def _named_and_var_kw(callable_obj):
    """``(named params, accepts **kwargs)`` for a class ctor or a plain function."""
    named = set(_named_defaults(callable_obj)) - BASE_PARAMS
    signature = inspect.signature(
        callable_obj.__init__ if inspect.isclass(callable_obj) else callable_obj
    )
    accepts_var_kw = any(
        param.kind == param.VAR_KEYWORD for param in signature.parameters.values()
    )
    return named, accepts_var_kw


def _probe_arguments(callable_obj):
    """Kwargs that construct ``callable_obj``, or ``None`` if `_PROBE_ARGS` cannot supply them."""
    arguments = {}
    for name, default in _named_defaults(callable_obj).items():
        if name in BASE_PARAMS or default is not inspect.Parameter.empty:
            continue
        if name not in _PROBE_ARGS:
            return None
        arguments[name] = _PROBE_ARGS[name]
    return arguments


_WrapperProbe = collections.namedtuple("_WrapperProbe", "wrapped overrides pinned")


@functools.lru_cache(maxsize=None)
def _wrapper_probe(target):
    """What a wrapper FUNCTION really does, obtained by BUILDING it. ``None`` for a class.

    # DECISION plan-2026-07-20T191713-52a15234/D-010
    Three consecutive review passes found four defects here, every one of them the same
    failure in a new syntactic costume: an `ast` model of the wrapper inferred reachability
    by pattern-matching syntax and got it confidently WRONG -- a `setdefault` after a
    CONDITIONAL early return adopted as unconditional, a param forwarded as
    `partition_mode=partition_mode.lower()` declared unsettable while being plainly settable.
    That is not a coverage gap; a false model certifies real drift as GREEN, while an
    omission merely leaves you where you were.

    DO NOT reintroduce static analysis of the wrapper body -- no `ast.walk`, no
    "unconditional top-level statement" scan, no `kwargs.setdefault` literal extraction, and
    no per-shape special case. If a fifth wrapper shape ever appears to need one, that is the
    signal to extend the PROBE, not to start modelling syntax again. See decisions.md D-010.

    Ground truth is obtained empirically instead, and cannot mis-model control flow because
    it does not model control flow:

    * ``wrapped``   -- ``type()`` of what the wrapper actually returns. Not the
      ``return_annotation``: an annotation can be a string under PEP 563 and lie by omission.
    * ``overrides`` -- DIFFERENTIAL. Build the class directly and build it through the
      wrapper with the same arguments; any attribute that DIFFERS is exactly what the wrapper
      forces. Differencing two real constructions is what makes this immune to attributes the
      ctor transforms (`self.x = normalize(x)`): both sides transform identically, so only a
      genuine override survives.
    * ``pinned``    -- params the wrapper hard-codes in its delegation, so that supplying them
      raises ``TypeError: got multiple values for keyword argument '<name>'``. Matched on that
      SPECIFIC message: a param rejected for any other reason (validation, an unrelated
      TypeError) is NOT a pin, and treating it as one would drop it from the MISSING set and
      reinstate the silent-drop false-green.

    Returns ``None`` when the target is a class (nothing to resolve) or cannot be probed.
    """
    if inspect.isclass(target) or not callable(target):
        return None

    arguments = _probe_arguments(target)
    if arguments is None:
        return None
    try:
        via_wrapper = target(**arguments)
    except Exception:
        return None

    wrapped = type(via_wrapper)
    class_arguments = _probe_arguments(wrapped)
    if class_arguments is None:
        return None
    try:
        direct = wrapped(**class_arguments)
    except Exception:
        return None

    class_defaults = _named_defaults(wrapped)
    missing = object()

    overrides = {}
    for name in class_defaults:
        if name in BASE_PARAMS:
            continue
        theirs = getattr(direct, name, missing)
        ours = getattr(via_wrapper, name, missing)
        if theirs is missing or ours is missing:
            continue
        if not (isinstance(theirs, _LITERAL_TYPES) and isinstance(ours, _LITERAL_TYPES)):
            continue
        if not _defaults_equal(ours, theirs):
            overrides[name] = ours

    pinned = set()
    for name, default in class_defaults.items():
        if name in BASE_PARAMS or name in arguments:
            continue
        value = default if default is not inspect.Parameter.empty else _PROBE_ARGS.get(name)
        if value is None and default is inspect.Parameter.empty:
            continue
        try:
            target(**arguments, **{name: value})
        except TypeError as error:
            if f"multiple values for keyword argument '{name}'" in str(error):
                pinned.add(name)
        except Exception:
            continue  # rejected for an unrelated reason -- that is not a pin
    return _WrapperProbe(wrapped, overrides, frozenset(pinned))


def _registry_entries():
    """Yield (factory, type_name, target, info) for every registered type."""
    for factory, (module_path, attr) in FACTORIES.items():
        registry = getattr(importlib.import_module(module_path), attr)
        for type_name, info in sorted(registry.items()):
            target = info.get("class")
            if target is None:
                continue
            yield factory, type_name, target, info


# named params a caller may set, whether **kwargs still swallows anything, and the params
# a wrapper hard-codes (settable by nobody).
_TargetSignature = collections.namedtuple(
    "_TargetSignature", "named accepts_var_kw pinned"
)


def _signature_params(target):
    """The parameter set a caller of this registry entry can actually set.

    # DECISION plan-2026-07-20T191713-52a15234/D-008
    A registry entry's ``'class'`` may be a CLASS (inspect ``__init__``) or a factory
    FUNCTION -- e.g. attention's ``'window'`` maps to ``create_grid_window_attention``.
    Inspecting the wrapper ALONE is not merely imprecise, it is blind: the wrapper names
    only `dim, window_size, num_heads, **kwargs`, so every one of `WindowAttention`'s other
    20 params looked "not accepted" and the MISSING guard -- the highest-value check in this
    file, the one that found 27 silently-droppable params -- could not see a single one of
    them for these two entries.

    DO NOT resolve wrappers in `_effective_defaults` (the VALUE test) only. That is exactly
    what shipped, and it left this NAME test blind while the plan recorded the blindness as
    fixed; adding a new keyword param to `WindowAttention.__init__` kept all 293 tests green.
    Both directions share the single `_wrapper_probe` above for that reason.
    See decisions.md D-008, D-010.

    Resolution: the wrapper's own named params, PLUS the named params of the class it was
    OBSERVED to build (which reach it through ``**kwargs``), MINUS the params it was OBSERVED
    to pin -- supplying those raises TypeError, so they are settable by nobody and belong in
    ``pinned``, not in ``named``.
    """
    named, accepts_var_kw = _named_and_var_kw(target)
    probe = _wrapper_probe(target)
    if probe is None:
        return _TargetSignature(named, accepts_var_kw, frozenset())

    wrapped_named, wrapped_var_kw = _named_and_var_kw(probe.wrapped)
    pinned = frozenset(probe.pinned - BASE_PARAMS)
    # kwargs flow straight through, so what the CLASS swallows is what the door swallows.
    return _TargetSignature(
        (named | wrapped_named) - pinned, wrapped_var_kw, pinned
    )


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
    signature = _signature_params(target)
    declared = set(info.get("required_params", [])) | set(
        info.get("optional_params", {})
    )
    missing = sorted(signature.named - declared)
    assert not missing, (
        f"{factory} registry entry '{type_name}' ({getattr(target, '__name__', target)}) "
        f"does not declare {missing}, which its constructor accepts. create_{factory}_* "
        f"FILTERS kwargs against this registry, so a caller passing any of these has the "
        f"value SILENTLY DISCARDED and gets the class default. Add them to "
        f"'optional_params' with their real constructor defaults."
    )


@pytest.mark.parametrize("factory,type_name,target,info", ENTRIES, ids=IDS)
def test_registry_declares_no_param_the_target_rejects(factory, type_name, target, info):
    """PHANTOM drift: a registry key the target does not accept -> TypeError at build.

    Two shapes of phantom, and the second only exists because of wrapper resolution:

    1. The target names no such param and swallows no ``**kwargs``.
    2. The target is a WRAPPER that HARD-CODES the param in its delegation call. Declaring
       it makes `create_*` inject it into `**kwargs`, and the class then receives the
       argument twice: ``TypeError: got multiple values for keyword argument``. This one
       fires even though the wrapper takes ``**kwargs``, so it must be checked BEFORE the
       var-kw early return. It is the obligation that pays for `_signature_params` removing
       pinned params from ``named``: excluding them from MISSING without also forbidding
       them in PHANTOM would be a silent blind spot, not a model.
    """
    signature = _signature_params(target)
    declared = set(info.get("required_params", [])) | set(
        info.get("optional_params", {})
    )
    target_name = getattr(target, "__name__", target)

    pinned_phantom = sorted(declared & signature.pinned)
    assert not pinned_phantom, (
        f"{factory} registry entry '{type_name}' declares {pinned_phantom}, which the "
        f"wrapper {target_name} HARD-CODES when it delegates. create_{factory}_* would pass "
        f"the registry value through **kwargs and the wrapped class would receive the "
        f"argument twice -- TypeError: got multiple values for keyword argument. This door "
        f"cannot expose that parameter; if callers need it, they need a different registry "
        f"entry (as attention:window vs attention:window_zigzag already are), or the wrapper "
        f"must stop pinning it."
    )

    if signature.accepts_var_kw:
        # The target takes **kwargs, so it accepts anything; no phantom is possible.
        # This is NOT a skip of the subject -- the MISSING check above still applies to it.
        return
    phantom = sorted(declared - signature.named - BASE_PARAMS)
    assert not phantom, (
        f"{factory} registry entry '{type_name}' declares {phantom}, which "
        f"{target_name} does not accept -- construction would raise TypeError."
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


def _effective_defaults(target):
    """param name -> the default a caller gets when the factory passes nothing for it.

    For a CLASS: its ``__init__`` defaults. For a wrapper FUNCTION: the wrapper's own named
    defaults, plus the defaults of the class it was OBSERVED to build for params it only
    forwards via ``**kwargs``, plus every default that construction showed the wrapper
    OVERRIDES. Params with no default map to ``inspect.Parameter.empty``.

    # DECISION plan-2026-07-20T191713-52a15234/D-010
    The wrapper's contribution is measured (`_wrapper_probe`), never parsed. DO NOT reintroduce
    a `kwargs.setdefault` literal scan on top of these values: three passes of that produced
    four false models, each asserting a default no caller ever receives -- which turns REAL
    drift green while reading as coverage. If the observed override looks wrong, the wrapper
    really does behave that way; read the wrapper. See decisions.md D-010.

    A wrapper that cannot be constructed contributes no class defaults, so its forwarded params
    stay NON-COMPARABLE -- graceful, but blind, and `test_every_wrapper_entry_is_probeable`
    reds so the blindness cannot arrive silently.
    """
    defaults = _named_defaults(target)
    probe = _wrapper_probe(target)
    if probe is None:
        return defaults

    for name, default in _named_defaults(probe.wrapped).items():
        defaults.setdefault(name, default)
    defaults.update(probe.overrides)
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


# ---------------------------------------------------------------------
# REQUIRED/OPTIONAL PLACEMENT drift (the fourth direction)
# ---------------------------------------------------------------------
# The three checks above police the UNION `required_params | optional_params`: MISSING (a
# ctor param in neither), PHANTOM (a declared param the ctor rejects), and VALUE (an
# optional default that disagrees with the ctor). None asks WHICH half of the union a name
# lives in. So a param the constructor REQUIRES (no default) declared under `optional_params`
# slips through every one of them: its name is in the union (MISSING is happy), the ctor
# accepts it (PHANTOM is happy), and `_classify` routes its `_EMPTY` ctor-default straight to
# NON_COMPARABLE (VALUE never compares it). Yet through `create_*` the registry's optional
# default is INJECTED (`params.update(optional_params)`) and silently substitutes for the
# omitted required argument, and `get_*_info()` reports the param as optional-with-default
# when the class cannot be built without it. This is the blind spot D-001 closes.


def _ctor_required_but_optional(target, info):
    """Params declared under `optional_params` that the constructor actually REQUIRES.

    Required-ness is derived from the SIGNATURE, never a name list: a param whose effective
    default is `inspect.Parameter.empty` has no default, so the constructor requires it.
    `_effective_defaults` is the exact source `_classify` uses (and it resolves wrapper
    targets, so a param a wrapper only forwards via ``**kwargs`` still gets the wrapped
    class's real default rather than reading as spuriously-required).

    Loops `optional_params` ONLY -- so it structurally cannot see a param declared under
    `required_params` (e.g. the opposite-direction case
    `activations:hierarchical_routing.output_dim`, required-in-registry but ctor-defaulted).
    That case is a different, non-crashing defect class and is out of scope by construction.
    """
    defaults = _effective_defaults(target)
    return sorted(
        param
        for param in (info.get("optional_params") or {})
        if defaults.get(param, _EMPTY) is _EMPTY
    )


@pytest.mark.parametrize("factory,type_name,target,info", ENTRIES, ids=IDS)
def test_registry_required_params_include_all_ctor_required(
    factory, type_name, target, info
):
    """PLACEMENT drift: a ctor-REQUIRED param (no default) must live in `required_params`.

    # DECISION plan-2026-07-21T044709-6bbefa2a/D-001
    Why this can fail if the implementation is wrong: if a param the constructor requires is
    declared under `optional_params`, `create_*` INJECTS the registry's optional default
    (`params.update(optional_params)`) and silently substitutes it for the omitted required
    argument -- so a direct constructor caller who trusts the registry's implied
    "optional-with-default" contract gets `TypeError: missing N required positional
    arguments`, and every `get_*_info()` consumer mislabels the param. Moving it to
    `required_params` makes `validate_*_config` reject omission with a clear `ValueError`
    naming the missing param instead.

    This is NOT hardcoded to any known offender: required-ness is derived from the ctor
    signature via `_effective_defaults` (`_EMPTY` == no default), the same source `_classify`
    routes to NON_COMPARABLE. `test_required_optional_guard_detects_synthetic_misregistration`
    proves the derivation on a fabricated dummy whose required param is named nothing like the
    real offenders. The check loops `optional_params` only, so it cannot fire on the
    opposite-direction `required_params` case (F6) -- verified by that param never appearing
    in a failure here.
    """
    misregistered = _ctor_required_but_optional(target, info)
    assert not misregistered, (
        f"{factory} registry entry '{type_name}' "
        f"({getattr(target, '__name__', target)}) declares {misregistered} under "
        f"'optional_params', but its constructor requires them (no default). create_{factory}_* "
        f"INJECTS the optional default and silently substitutes it for the omitted required "
        f"argument, so a DIRECT constructor call raises 'missing N required positional "
        f"arguments' while get_{factory}_info() reports these as optional-with-default. Move "
        f"them into 'required_params' (dropping the now-inert optional default) so omission "
        f"raises a clear ValueError instead."
    )


def test_required_optional_guard_detects_synthetic_misregistration():
    """Anti-hardcoding + falsifiability for the PLACEMENT guard, on a fabricated subject.

    Why this can fail if the implementation is wrong: this project has shipped FOUR
    structurally-unfalsifiable checks (LESSONS), so the placement guard must be shown to (a)
    RED on a genuine misregistration and (b) GREEN on its correction, using a param
    (`needed`) named nothing like any real offender -- which is impossible for a check that
    keys off a hardcoded list of the five known names. `opt`, though also under
    `optional_params`, must NOT be flagged: it has a real ctor default, so the guard is not
    merely "everything in optional_params is wrong".
    """

    class _DummyRequiresNeeded:
        def __init__(self, needed, opt=1, **kwargs):
            self.needed = needed
            self.opt = opt

    misregistered_info = {
        "required_params": [],
        "optional_params": {"needed": 5, "opt": 1},
    }
    assert _ctor_required_but_optional(_DummyRequiresNeeded, misregistered_info) == [
        "needed"
    ], (
        "the placement guard failed to flag a ctor-required param declared optional -- it is "
        "not deriving required-ness from the signature"
    )

    corrected_info = {
        "required_params": ["needed"],
        "optional_params": {"opt": 1},
    }
    assert (
        _ctor_required_but_optional(_DummyRequiresNeeded, corrected_info) == []
    ), "moving the required param into required_params must clear the flag (GREEN direction)"


# ---------------------------------------------------------------------
# REQUIRED-GENUINENESS drift (the mirror of the PLACEMENT direction)
# ---------------------------------------------------------------------
# The PLACEMENT check above catches a ctor-REQUIRED param (no default) filed under
# `optional_params`. Its mirror image is a param filed under `required_params` that HAS a ctor
# default. A ctor default normally MEANS "optional", so declaring such a param `required` LOOKS
# like over-declaration -- but it is not always: a class can gate a defaulted param on a MODE the
# registry pins, making it genuinely required at that door. `create_*` cannot see the difference
# (its required-guard raises on ANY omitted required member), and neither can the three checks
# above (a `required_params` name is in the union, the ctor accepts it, and `_classify` routes its
# ctor-defaulted value to a compare, never to a required-ness test). So an over-declared required
# param -- one the class builds fine without -- slips through every existing direction, and
# `get_*_info()` tells callers to pass a value the class does not need.
#
# The sole in-scope member repo-wide today (F4) is `activations:hierarchical_routing.output_dim`
# (ctor `output_dim=None`), and it is CORRECT, NOT a bug -- NO source change (decisions.md D-001).
# That door pins `mode='trainable'`, under which `RoutingProbabilitiesLayer.__init__` raises
# `ValueError` when `output_dim is None` (layers/activations/routing_probabilities.py:378-388): the
# `None` default is a mode-gated SENTINEL, so the param is genuinely required despite the default.
# The sibling door `activations:routing_probabilities` pins `mode='deterministic'` and correctly
# files `output_dim` as optional (layers/activations/factory.py:214-235). This check proves the
# required label by CONSTRUCTION, not by trusting it: build the class DIRECTLY under the entry's
# pinned config, omit the member, and require the ctor to raise.


def _overdeclared_required_params(target, info):
    """`required_params` members that carry a ctor default AND construct fine when omitted.

    The mirror of `_ctor_required_but_optional`. Required-ness is proven EMPIRICALLY -- from the
    SIGNATURE plus one construction -- never from a name list. For each `required_params` member
    whose effective default is NOT `_EMPTY` (it has a ctor default, so it LOOKS optional), build the
    target DIRECTLY under the entry's pinned `optional_params`, supplying `_PROBE_ARGS` for any
    OTHER no-default required param, but OMITTING the member under test. If construction RAISES, the
    member is genuinely required despite its default (a mode-gated sentinel, like
    `hierarchical_routing.output_dim` under `mode='trainable'`) -- correct, not flagged. If
    construction SUCCEEDS, the param is actually optional and the entry over-declares it -- flagged.

    Construction is DIRECT (`target(**kwargs)`), NOT via `create_*` (decisions.md D-002): the
    factory applies its OWN required-params guard and raises on ANY omitted required member
    regardless of ctor genuineness, so a factory-based check would be tautological (always green)
    and thus unfalsifiable -- the project's named "unfalsifiable check" trap. Direct construction
    reds precisely when the registry over-declares an actually-optional param, which is the property
    under test.

    A member with NO ctor default (`_EMPTY`) is genuinely required and skipped -- that is the
    PLACEMENT check's out-of-scope case seen from the other side. A member whose OTHER required
    params need a probe value absent from `_PROBE_ARGS` is unprobeable and skipped (a disclosed
    conservative skip, never a silent flag). Today the sole in-scope member has no other required
    param, so neither skip is exercised by the real tree.

    LIMIT (S2): this is a CONSTRUCTION-time probe. A param required only at `build()`/`call()` --
    validated against an input the ctor never sees -- would construct successfully when omitted and
    be FALSELY flagged as over-declared. None exists today: the sole in-scope member raises in
    `__init__`. Stated, not modelled.

    Returns a sorted list of over-declared member names (empty when every in-scope member is
    genuinely required).
    """
    defaults = _effective_defaults(target)
    required = info.get("required_params", []) or []
    optional = info.get("optional_params") or {}

    overdeclared = []
    for member in required:
        if defaults.get(member, _EMPTY) is _EMPTY:
            continue  # no ctor default -> genuinely required, out of scope

        kwargs = dict(optional)
        probeable = True
        for other in required:
            if other == member or defaults.get(other, _EMPTY) is not _EMPTY:
                continue  # the member itself, or another required param that has its own default
            if other not in _PROBE_ARGS:
                probeable = False
                break
            kwargs[other] = _PROBE_ARGS[other]
        if not probeable:
            continue  # cannot build without an unavailable required value -> skip (disclosed)

        kwargs.pop(member, None)  # OMIT the member under test (defensive: it is not in optional)
        try:
            target(**kwargs)
        except Exception:
            continue  # ctor rejected the omission -> genuinely required
        overdeclared.append(member)  # built without it -> actually optional, over-declared
    return sorted(overdeclared)


@pytest.mark.parametrize("factory,type_name,target,info", ENTRIES, ids=IDS)
def test_registry_required_params_are_genuinely_required(
    factory, type_name, target, info
):
    """REQUIRED-GENUINENESS drift: a `required_params` member with a ctor default must still be
    required-in-fact -- omitting it at direct construction under the entry's pinned config RAISES.

    # DECISION plan-2026-07-21T053622-c786e909/D-001
    Why this can fail if the implementation is wrong: a `required_params` member that carries a
    ctor default LOOKS optional, and if the class builds fine without it then `get_*_info()`
    over-declares it and callers are told to pass a value the class does not need. The one such
    member today, `activations:hierarchical_routing.output_dim` (ctor `output_dim=None`), is NOT a
    bug and the registry is NOT changed (F6, decisions.md D-001): this door pins `mode='trainable'`,
    under which `RoutingProbabilitiesLayer.__init__` raises `ValueError` when `output_dim is None`
    (layers/activations/routing_probabilities.py:378-388). The `None` default is a mode-gated
    sentinel, so the param is genuinely required and this check PASSES it -- proving the required
    label by construction, not asserting it. The sibling door `activations:routing_probabilities`
    pins `mode='deterministic'` and correctly files `output_dim` as optional
    (layers/activations/factory.py:214-235); the two doors mirror the class's mode-gated contract
    correctly, so there is no drift to fix.

    Required-ness is derived from the ctor SIGNATURE (`_effective_defaults`; `_EMPTY` is skipped as
    genuinely-required), never a name list -- proven independent of this tree by
    `test_required_genuineness_guard_detects_synthetic_overdeclaration`, which reds a fabricated
    over-declared entry whose param is named nothing like `output_dim`. Today exactly one entry
    (F4) exercises the non-vacuous branch; the rest pass because they have no ctor-defaulted
    required member, so the synthetic falsifier -- not the tree -- is what keeps the mechanism
    falsifiable.

    LIMIT (S2): a construction-time probe cannot see a param required only at `build()`/`call()`;
    such a param would construct when omitted and be falsely flagged. None exists today (the sole
    in-scope member raises in `__init__`). Disclosed on `_overdeclared_required_params`.
    """
    overdeclared = _overdeclared_required_params(target, info)
    assert not overdeclared, (
        f"{factory} registry entry '{type_name}' ({getattr(target, '__name__', target)}) declares "
        f"{overdeclared} under 'required_params', but the constructor builds successfully with "
        f"{overdeclared} OMITTED under this entry's pinned optional_params -- so the param has a "
        f"real ctor default and is actually OPTIONAL. get_{factory}_info() over-declares it and "
        f"callers are told to supply a value the class does not need. Move it to 'optional_params' "
        f"with its real ctor default, UNLESS a pinned mode makes it genuinely required (as "
        f"activations:hierarchical_routing pins mode='trainable' for output_dim) -- in which case "
        f"the direct construction here would have raised and this test would not fire."
    )


def test_required_genuineness_guard_detects_synthetic_overdeclaration():
    """Anti-hardcoding + falsifiability for the REQUIRED-GENUINENESS guard, on fabricated subjects.

    Why this can fail if the implementation is wrong: this project has shipped FOUR
    structurally-unfalsifiable checks (LESSONS), and the real tree has exactly ONE in-scope member
    that is CORRECT -- so the real-entry test above can never red on this tree and proves nothing on
    its own. This test supplies the RED: `_overdeclared_required_params` must flag a ctor-defaulted
    param declared `required` that CONSTRUCTS when omitted (over-declared), and must NOT flag one
    that RAISES when omitted under its pinned mode (genuine). Both dummies use param names (`gain`,
    `scale`, `mode`) unlike the real offender `output_dim`, which is impossible for a check keyed off
    a hardcoded name list.

    Constructing DIRECTLY (not via `create_*`) is what makes the RED reachable: a factory-based
    check would raise on ANY omitted required member and the over-declared dummy would pass green --
    the tautology D-002 rejects.
    """

    # RED: `gain` has a ctor default and is declared required, but construction omitting it
    # succeeds -> actually optional -> over-declared.
    class _DummyOverdeclaresGain:
        def __init__(self, gain=0.5, **kwargs):
            self.gain = gain

    overdeclared_info = {
        "required_params": ["gain"],
        "optional_params": {},
    }
    assert _overdeclared_required_params(_DummyOverdeclaresGain, overdeclared_info) == [
        "gain"
    ], (
        "the guard failed to flag a ctor-defaulted param that constructs when omitted -- it is not "
        "proving required-ness by direct construction"
    )

    # GREEN: `scale` has a ctor default (None) but is mode-gated -- under the pinned mode='on' the
    # ctor raises when it is omitted, exactly the RoutingProbabilitiesLayer shape -> genuine.
    class _DummyGatesScale:
        def __init__(self, scale=None, mode="off", **kwargs):
            if mode == "on" and scale is None:
                raise ValueError("scale is required when mode='on'")
            self.scale = scale
            self.mode = mode

    genuine_info = {
        "required_params": ["scale"],
        "optional_params": {"mode": "on"},
    }
    assert _overdeclared_required_params(_DummyGatesScale, genuine_info) == [], (
        "a genuinely mode-gated required param (raises when omitted under the pinned mode) must NOT "
        "be flagged -- the guard must not treat every ctor default as over-declaration"
    )

    # SKIP: a required member with NO ctor default is genuinely required and out of scope; it is
    # never flagged even though no value is supplied for it here.
    class _DummyTrulyRequires:
        def __init__(self, needed, **kwargs):
            self.needed = needed

    assert (
        _overdeclared_required_params(
            _DummyTrulyRequires,
            {"required_params": ["needed"], "optional_params": {}},
        )
        == []
    ), (
        "a required param with NO ctor default is genuinely required and must be skipped, not "
        "flagged -- the guard is scoped to ctor-defaulted members only"
    )


# ---------------------------------------------------------------------
# Guard-of-the-guard: the wrapper model itself
#
# Everything above trusts `_signature_params` / `_effective_defaults` to establish what a
# wrapper entry really does. The static model that preceded `_wrapper_probe` was wrong FOUR
# times (decisions.md D-008, D-010), and no error was ever visible from the registries,
# because the only two wrappers in this repo have the SIMPLEST possible shape: one
# unconditional top-level `setdefault` each. Every shape that broke the static model is
# reconstructed below and asserted against the probe.
#
# These fixtures are kept deliberately, even though the mechanism they broke is gone: they are
# the falsification test for D-010's central claim -- that an empirical probe handles all of
# them WITHOUT a single per-shape special case. If any one of them ever needs special-casing
# in `_wrapper_probe`, that claim is false and the retirement of the AST layer must be revisited.
# ---------------------------------------------------------------------


class _SyntheticWrapped:
    """Stand-in for a wrapped layer class. `flag` defaults True; that is the value at stake."""

    def __init__(self, dim, flag=True, partition_mode="grid", **kwargs):
        self.dim = dim
        self.flag = flag
        self.partition_mode = partition_mode


def _wrapper_setdefault_unconditional(dim, **kwargs) -> _SyntheticWrapped:
    """POSITIVE CONTROL. The one shape the old static model got right: effective `flag` IS False."""
    kwargs.setdefault("flag", False)
    return _SyntheticWrapped(dim=dim, **kwargs)


def _wrapper_setdefault_in_if(dim, **kwargs) -> _SyntheticWrapped:
    """`dim` is 32 at every call site below, so this never runs: effective `flag` is True."""
    if dim > 128:
        kwargs.setdefault("flag", False)
    return _SyntheticWrapped(dim=dim, **kwargs)


def _wrapper_setdefault_in_for(dim, **kwargs) -> _SyntheticWrapped:
    """A zero-iteration loop body: effective `flag` is True."""
    for _ in range(0):
        kwargs.setdefault("flag", False)
    return _SyntheticWrapped(dim=dim, **kwargs)


def _wrapper_setdefault_in_nested_def(dim, **kwargs) -> _SyntheticWrapped:
    """A helper that is defined and never called: effective `flag` is True."""

    def _never_called():
        kwargs.setdefault("flag", False)

    return _SyntheticWrapped(dim=dim, **kwargs)


def _wrapper_setdefault_after_return(dim, **kwargs) -> _SyntheticWrapped:
    """Dead code after the return: effective `flag` is True."""
    return _SyntheticWrapped(dim=dim, **kwargs)
    kwargs.setdefault("flag", False)  # noqa - unreachable on purpose


def _wrapper_setdefault_after_conditional_return(dim, **kwargs) -> _SyntheticWrapped:
    """The pass-2 shape: a TOP-LEVEL setdefault skipped by a conditional early return.

    `dim` is 32 at every call site, so the early return always fires and effective `flag` is
    True. The static model stopped only at a top-level `return`, never accounted for control
    flow that SKIPS a later top-level statement, and adopted `False` -- without even reporting
    it as unprovable, so no hint fired either. See decisions.md D-010.
    """
    if dim > 16:
        return _SyntheticWrapped(dim=dim, **kwargs)
    kwargs.setdefault("flag", False)
    return _SyntheticWrapped(dim=dim, **kwargs)


def _wrapper_forwards_partition_mode(
    dim, partition_mode="grid", **kwargs
) -> _SyntheticWrapped:
    """Forwards `partition_mode`; a caller CAN set it, so the registry must declare it."""
    return _SyntheticWrapped(dim=dim, partition_mode=partition_mode, **kwargs)


def _wrapper_renormalizes_partition_mode(
    dim, partition_mode="grid", **kwargs
) -> _SyntheticWrapped:
    """The other pass-2 shape: forwards through an EXPRESSION, and is still settable.

    `w(dim=32, partition_mode='ZIGZAG').partition_mode == 'zigzag'`. The static model asked
    only whether the forwarded value was a bare `Name` and so called this a PIN -- which drops
    the param from the MISSING set and reinstates the silent-drop false-green this whole file
    exists to prevent. See decisions.md D-010.
    """
    return _SyntheticWrapped(
        dim=dim, partition_mode=partition_mode.lower(), **kwargs
    )


def _wrapper_pins_partition_mode(dim, **kwargs) -> _SyntheticWrapped:
    """Hard-codes `partition_mode`; a caller supplying it gets TypeError, so it is not settable."""
    return _SyntheticWrapped(dim=dim, partition_mode="zigzag", **kwargs)


def _wrapper_pins_via_assign_then_return(dim, **kwargs) -> _SyntheticWrapped:
    """A REAL pin the static model missed: the return value is a Name, not a Call.

    Missing a real pin is not harmless -- the MISSING guard would DEMAND the registry declare
    `partition_mode`, and declaring it makes every `create_*` call raise TypeError.
    """
    layer = _SyntheticWrapped(dim=dim, partition_mode="zigzag", **kwargs)
    return layer


def _wrapper_pins_via_dict_splat(dim, **kwargs) -> _SyntheticWrapped:
    """A REAL pin the static model missed: `keyword.arg is None` for a `**dict` splat."""
    pins = {"partition_mode": "zigzag"}
    return _SyntheticWrapped(dim=dim, **pins, **kwargs)


def _wrapper_rejects_flag_with_value_error(dim, **kwargs) -> _SyntheticWrapped:
    """REJECTS `flag`, and a rejection is NOT a pin.

    A pin means "the wrapper already supplied this, so nobody else can". A rejection means
    "this value is unacceptable". Only the first justifies dropping the param from the MISSING
    set. Conflating them is a false-green: an unguarded param that looks guarded.
    """
    if "flag" in kwargs:
        raise ValueError("flag is not supported by this door")
    return _SyntheticWrapped(dim=dim, **kwargs)


def _wrapper_rejects_flag_with_unrelated_type_error(dim, **kwargs) -> _SyntheticWrapped:
    """Rejects `flag` with a TypeError of a DIFFERENT kind -- still not a pin.

    This is why `_wrapper_probe` matches the specific "got multiple values for keyword
    argument" text and not the exception CLASS. `TypeError` is the ordinary way Python reports
    a bad argument type, so "any TypeError means pinned" would silently unguard every param a
    layer type-checks.
    """
    if "flag" in kwargs:
        raise TypeError("flag must be a tensor, not a bool")
    return _SyntheticWrapped(dim=dim, **kwargs)


def _wrapper_overwrites_kwargs_entry(dim, **kwargs) -> _SyntheticWrapped:
    """The one shape NO direction sees: a silent overwrite that raises nothing.

    `kwargs['partition_mode'] = ...` drops the caller's value without a TypeError, so the
    probe correctly reports NO pin (there is none), the param stays in the MISSING set, and a
    registry that declares it is not wrong to. Nothing here is mis-modelled -- but nothing
    catches the drop either. Recorded as module-docstring item 5 and pinned below so the
    limitation is a decision on the record rather than an assumption nobody wrote down.
    """
    kwargs["partition_mode"] = "zigzag"
    return _SyntheticWrapped(dim=dim, **kwargs)


# Every shape whose `setdefault` does NOT establish the effective default. The first four are
# pass-1's; the fifth is pass-2's, and it is the one the static model could not be taught.
_NON_TOP_LEVEL_WRAPPERS = [
    _wrapper_setdefault_in_if,
    _wrapper_setdefault_in_for,
    _wrapper_setdefault_in_nested_def,
    _wrapper_setdefault_after_return,
    _wrapper_setdefault_after_conditional_return,
]


def test_probe_reads_a_real_forced_default():
    """POSITIVE CONTROL for the five negative cases below.

    Why this can fail if the implementation is wrong: a probe that returned no overrides at all
    -- or one restricted so tightly that it observes nothing -- would satisfy every "must fall
    back to the class default" assertion below while establishing nothing. This is what makes
    those five non-vacuous. It also holds the line on the real chain: without it, one could
    "fix" a false model by disabling wrapper resolution entirely, and the suite would stay green
    while attention:window_zigzag's CORRECT `False` started reading as drift.
    """
    probe = _wrapper_probe(_wrapper_setdefault_unconditional)
    assert probe is not None and probe.wrapped is _SyntheticWrapped
    assert probe.overrides.get("flag") is False, (
        f"a wrapper that really does force flag=False was not observed doing so: "
        f"{probe.overrides!r}"
    )
    assert _effective_defaults(_wrapper_setdefault_unconditional)["flag"] is False
    # ...and the same on the two REAL wrappers, so this is not a claim about toys only.
    assert (
        _effective_defaults(create_grid_window_attention)["use_relative_position_bias"]
        is True
    )
    assert (
        _effective_defaults(create_zigzag_window_attention)["use_relative_position_bias"]
        is False
    )
    assert _wrapper_probe(create_zigzag_window_attention).overrides[
        "use_relative_position_bias"
    ] is False, (
        "the zigzag wrapper's False must be OBSERVED, not inherited from the class default True"
    )


def test_a_class_target_is_not_probed():
    """A class wraps nothing, so it must cost zero constructions and yield no probe.

    Why this can fail if the implementation is wrong: probing were it not needed would build
    95 layers at collection time and could turn an unrelated ctor failure into a drift red.
    """
    assert _wrapper_probe(WindowAttention) is None
    assert _signature_params(WindowAttention).pinned == frozenset()


@pytest.mark.parametrize(
    "wrapper", _NON_TOP_LEVEL_WRAPPERS, ids=lambda w: w.__name__
)
def test_setdefault_the_caller_never_reaches_is_not_adopted(wrapper):
    """A `setdefault` that does not run must NOT overwrite the class default.

    Why this can fail if the implementation is wrong: it DID fail -- the first four shapes
    under `ast.walk` (D-008), then the fifth under the "unconditional top-level" scan that
    replaced it (D-010). Each time, `_effective_defaults` returned False: a value no caller of
    these wrappers ever receives. The guard then asserts a registry `False` as correct and the
    true `True` as drift -- real drift green, correct code red, which is a correctness defect in
    the safety mechanism rather than a coverage gap.

    The probe passes all five WITHOUT knowing that `if`, `for`, nested `def`, dead code and
    early returns exist. That is D-010's claim, and this is where it is falsifiable: if any
    shape here ever needs a special case in `_wrapper_probe`, the claim is false.
    """
    assert wrapper(dim=32).flag is True, (
        "the fixture is not modelling what it claims -- this wrapper's real effective `flag` "
        "must be the CLASS default True, or the assertions below prove nothing"
    )

    probe = _wrapper_probe(wrapper)
    assert "flag" not in probe.overrides, (
        f"{wrapper.__name__}: no caller reaches this kwargs.setdefault('flag', False), but the "
        f"guard recorded an override of {probe.overrides.get('flag')!r}"
    )
    assert _effective_defaults(wrapper)["flag"] is True, (
        f"{wrapper.__name__}: effective default must be the _SyntheticWrapped class default "
        f"True, not the unreachable literal False"
    )


def test_probe_ignores_values_that_cannot_be_compared_across_constructions():
    """A ctor-transformed attribute must not read as a wrapper override.

    Why this can fail if the implementation is wrong: `WindowAttention.__init__` does
    `keras.initializers.get(kernel_initializer)`, so two separate constructions hold two
    DISTINCT objects that compare unequal. Comparing them naively would report
    `kernel_initializer` as a forced default on both real wrappers -- pure invention -- and the
    registry's honest `'glorot_uniform'` would red as drift against an object repr.
    """
    for wrapper in (create_grid_window_attention, create_zigzag_window_attention):
        overrides = _wrapper_probe(wrapper).overrides
        assert "kernel_initializer" not in overrides
        assert "bias_initializer" not in overrides
        assert all(isinstance(value, _LITERAL_TYPES) for value in overrides.values())


def test_every_wrapper_entry_is_probeable():
    """Anti-rot: a wrapper the probe cannot build must fail LOUDLY, not degrade in silence.

    Why this can fail if the implementation is wrong: `_wrapper_probe` returns None on any
    construction failure, and every caller then falls back to the wrapper's own bare signature
    -- the pre-resolution blindness that hid 20 of `WindowAttention`'s params from the MISSING
    guard. That fallback is the right behavior, but it must never arrive unannounced. A new
    wrapper entry whose required params are absent from `_PROBE_ARGS` reds here, by name.
    """
    wrappers = [
        (factory, type_name, target)
        for factory, type_name, target, _ in ENTRIES
        if not inspect.isclass(target)
    ]
    assert wrappers, (
        "no function-target registry entries found -- if the wrapper entries were removed, "
        "delete this machinery deliberately rather than leaving it passing vacuously"
    )
    unprobeable = [
        f"{factory}:{type_name} ({target.__name__})"
        for factory, type_name, target in wrappers
        if _wrapper_probe(target) is None
    ]
    assert not unprobeable, (
        f"{unprobeable} could not be constructed for probing, so all three drift directions "
        f"silently fell back to the wrapper's own bare signature. Add the missing required "
        f"argument(s) to _PROBE_ARGS, or fix the wrapper -- do NOT leave this red unread: it "
        f"means those entries are no longer guarded."
    )


def test_name_guard_resolves_wrapper_targets():
    """The MISSING guard must see the params a wrapper FORWARDS, not just the ones it names.

    Why this can fail if the implementation is wrong: it did. D-004's wrapper resolution was
    wired into `_effective_defaults` (values) and never into `_signature_params` (names), so for
    attention:window / window_zigzag the MISSING check -- the check that originally found 27
    silently-droppable params -- saw a 3-parameter signature and could not flag anything. Adding
    a new keyword param to `WindowAttention.__init__` left all 293 tests green.
    """
    wrapper_only, _ = _named_and_var_kw(create_grid_window_attention)
    assert wrapper_only == {"dim", "window_size", "num_heads"}, (
        f"fixture drift: the unresolved wrapper signature is no longer the narrow one this "
        f"test contrasts against ({sorted(wrapper_only)})"
    )

    resolved = _signature_params(create_grid_window_attention)
    class_named, _ = _named_and_var_kw(WindowAttention)
    assert resolved.named >= (class_named - resolved.pinned), (
        f"wrapper resolution did not reach the wrapped class: "
        f"{sorted((class_named - resolved.pinned) - resolved.named)} still invisible"
    )
    assert len(resolved.named) > len(wrapper_only) + 5, (
        f"only {len(resolved.named)} params resolved -- resolution is not doing real work"
    )


def test_name_guard_is_live_for_wrapper_entries():
    """Deleting a declared param from a wrapper entry must make the MISSING guard red.

    Why this can fail if the implementation is wrong: `test_name_guard_resolves_wrapper_targets`
    proves the resolver returns a bigger set; it does NOT prove the assertion consumes it. This
    runs the guard's own `named - declared` computation against the REAL registry entry with one
    declared param withheld, which is the mutation `create_attention_layer` callers would feel.
    """
    for type_name in ("window", "window_zigzag"):
        info = dict(ATTENTION_REGISTRY[type_name])
        target = info["class"]
        named = _signature_params(target).named
        declared = set(info.get("required_params", [])) | set(
            info.get("optional_params", {})
        )
        assert not (named - declared), (
            f"attention:{type_name} is not clean before mutation: {sorted(named - declared)}"
        )
        victim = "qkv_bias"
        assert victim in declared
        assert named - (declared - {victim}) == {victim}, (
            f"withholding {victim!r} from attention:{type_name} did not make the MISSING guard "
            f"red -- the name check is inert for wrapper entries"
        )


@pytest.mark.parametrize(
    "wrapper",
    [
        _wrapper_pins_partition_mode,
        _wrapper_pins_via_assign_then_return,
        _wrapper_pins_via_dict_splat,
    ],
    ids=lambda w: w.__name__,
)
def test_a_real_pin_is_observed_whatever_shape_it_takes(wrapper):
    """If supplying the param really raises TypeError, it must be classified as pinned.

    Why this can fail if the implementation is wrong: MISSING excludes pinned params, so a pin
    the guard cannot see makes the guard DEMAND that the registry declare a param which then
    breaks every `create_*` call with a TypeError -- a red that leads the reader to introduce a
    live bug. The static model saw only the keywords of a top-level `return <Call>(...)`, so it
    missed the last two shapes here entirely; the probe does not care about shape, only about
    what happens.
    """
    with pytest.raises(TypeError, match="multiple values .*partition_mode"):
        wrapper(dim=32, partition_mode="grid")

    signature = _signature_params(wrapper)
    assert "partition_mode" in signature.pinned
    assert "partition_mode" not in signature.named


@pytest.mark.parametrize(
    "wrapper",
    [
        _wrapper_forwards_partition_mode,
        _wrapper_renormalizes_partition_mode,
        _wrapper_overwrites_kwargs_entry,
    ],
    ids=lambda w: w.__name__,
)
def test_a_settable_param_is_never_classified_as_pinned(wrapper):
    """A param that does NOT raise TypeError must stay in the MISSING set.

    Why this can fail if the implementation is wrong: this is the false-green direction, and it
    is the one that matters most. Calling a settable param "pinned" removes it from MISSING, so
    a registry that never declares it stops reding -- and every caller's value is then silently
    dropped by the `create_*` filter, which is the exact bug this whole file exists to catch.
    The static model called `partition_mode=partition_mode.lower()` a pin purely because the
    forwarded value was not a bare `Name` (decisions.md D-010).

    `_wrapper_overwrites_kwargs_entry` is the honest limit: it raises nothing, so the probe
    correctly finds no pin, yet it still discards the caller's value. Nothing in this file
    catches that shape; it is disclosed as module-docstring item 5 rather than papered over.
    """
    wrapper(dim=32, partition_mode="grid")  # settable: must not raise

    signature = _signature_params(wrapper)
    assert "partition_mode" not in signature.pinned
    assert "partition_mode" in signature.named, (
        "a settable param must stay in the MISSING set -- a registry that fails to declare it "
        "silently drops the caller's value, which is the whole point of this file"
    )
    assert "dim" not in signature.pinned, (
        "a plain forward of the wrapper's own parameter (dim=dim) is not a pin"
    )


@pytest.mark.parametrize(
    "wrapper,expected_error",
    [
        (_wrapper_rejects_flag_with_value_error, ValueError),
        (_wrapper_rejects_flag_with_unrelated_type_error, TypeError),
    ],
    ids=lambda x: getattr(x, "__name__", str(x)),
)
def test_a_rejected_param_is_not_mistaken_for_a_pinned_one(wrapper, expected_error):
    """A param that RAISES for an unrelated reason must not be classified as pinned.

    Why this can fail if the implementation is wrong: the probe learns "pinned" from an
    exception, so the obvious shortcut is `except Exception: pinned.add(name)` -- and on this
    repo's real wrappers that shortcut is INDISTINGUISHABLE from the correct implementation,
    because nothing here rejects its own default. These two fixtures are the only subjects that
    tell them apart. Getting it wrong drops a genuinely settable param out of the MISSING set,
    so a registry that never declares it stops reding and every caller's value is silently
    discarded -- the precise false-green D-010 retired the static model to avoid.
    """
    with pytest.raises(expected_error):
        wrapper(dim=32, flag=True)

    signature = _signature_params(wrapper)
    assert "flag" not in signature.pinned, (
        f"{wrapper.__name__} REJECTS flag, it does not pin it -- 'the wrapper already supplied "
        f"this value' and 'this value is unacceptable' are different facts, and only the first "
        f"one means the registry is right to omit the param"
    )
    assert "flag" in signature.named


def test_real_wrapper_pins_are_derived_not_exempted():
    """`partition_mode` is excluded from the MISSING set because it is UNSETTABLE, not waived.

    Why this can fail if the implementation is wrong: a blanket exemption would exclude it
    unconditionally, and would keep passing after a wrapper stopped pinning it. Here the
    exclusion is observed from the wrapper's real behavior, so the day `create_grid_window_
    attention` starts forwarding `partition_mode`, this reds and the registry must declare it.
    """
    for real in (create_grid_window_attention, create_zigzag_window_attention):
        assert "partition_mode" in _signature_params(real).pinned, (
            f"{real.__name__} hard-codes partition_mode; if it no longer does, the registry "
            f"must start declaring it and this test must be re-derived, not deleted"
        )
        assert _signature_params(real).pinned == frozenset({"partition_mode"}), (
            f"{real.__name__} pins more than partition_mode -- an over-broad pin set is a "
            f"silent MISSING blind spot: {sorted(_signature_params(real).pinned)}"
        )


def test_pinned_param_is_genuinely_unsettable():
    """The pin is a real TypeError, not a modelling convenience -- and the factory eats it.

    Why this can fail if the implementation is wrong: if `partition_mode` were in fact settable
    through the wrapper, excluding it from the MISSING set would be exactly the silent-drop bug
    this file exists to catch, and the first assertion would red.

    The second half pins BEHAVIOR THIS GUARD DOES NOT FIX: `create_attention_layer('window',
    partition_mode='zigzag')` silently discards the kwarg and returns a grid layer. That is the
    generic registry filter doing its job -- the 'window' door IS the grid door and
    'window_zigzag' is the zigzag one -- but the caller gets no signal. Recorded here so the
    behavior is a decision on the record rather than an accident, and so that a future change
    making it settable breaks a test instead of drifting in unnoticed.
    """
    with pytest.raises(TypeError, match="multiple values .*partition_mode"):
        create_grid_window_attention(
            dim=32, window_size=4, num_heads=4, partition_mode="zigzag"
        )

    layer = create_attention_layer(
        "window", dim=32, window_size=4, num_heads=4, partition_mode="zigzag"
    )
    assert layer.partition_mode == "grid", (
        "attention:window now honors partition_mode -- if the wrapper stopped pinning it, the "
        "registry must declare it (test_real_wrapper_pins_are_derived_not_exempted covers "
        "the derivation) and this expectation must be rewritten deliberately"
    )
