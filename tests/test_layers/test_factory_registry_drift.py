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
FUNCTION rather than a class. All three directions resolve such a target through the shared
helpers near the top of this file -- `_wrapped_class`, `_wrapper_pinned_params`,
`_wrapper_kwarg_setdefaults` -- so the guard reasons about the class the wrapper actually builds.
Resolving in only SOME directions is not a partial improvement, it is a blind spot with a
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
3. **Wrapper defaults this guard cannot PROVE are unconditional.** See
   `_wrapper_kwarg_setdefaults`. A `kwargs.setdefault(...)` inside a branch, a loop, a nested
   `def`, or after a `return` is deliberately NOT adopted: the wrapped class's own default is used
   instead, which is this guard's behavior before wrapper resolution existed. Non-literal values
   (a computed expression, an `if 'k' not in kwargs:` block) are likewise unseen. This is a
   deliberate downgrade from an earlier version that DID adopt them via `ast.walk` and thereby
   asserted values no caller receives -- see `_unconditional_body` and decisions.md D-008.
4. **`required_params` VALUES.** Only their names are checked, in both directions. A required
   param whose registry entry disagrees with the ctor in any respect other than presence is out
   of scope by construction.
5. **Params a wrapper pins POSITIONALLY.** `_wrapper_pinned_params` models only keyword arguments
   of the delegation call. None exists in this repo today.
"""

import ast
import collections
import importlib
import inspect
import textwrap

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
# Wrapper resolution
#
# Two registry entries (attention:window, attention:window_zigzag) map a FACTORY FUNCTION,
# not a class. Everything below models what such a wrapper really does to its caller's
# arguments, and it is used by ALL THREE drift directions -- MISSING, PHANTOM and VALUE.
# Wiring it into only some of them is precisely the defect this section was rewritten to
# fix (decisions.md D-008); the resolution lives here, once, above every consumer.
# ---------------------------------------------------------------------


def _function_ast(func):
    """The `ast.FunctionDef` of ``func``, or ``None`` if the source is unavailable."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    node = tree.body[0] if tree.body else None
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node
    return None


def _unconditional_body(func_node):
    """The statements of ``func_node`` that are certain to execute exactly once.

    # DECISION plan-2026-07-20T191713-52a15234/D-008
    Iterates the function's OWN top-level body and stops at the first ``return``. It
    deliberately does NOT descend into ``If`` / ``For`` / ``While`` / ``Try`` / ``With`` /
    nested ``def``, and deliberately does not look past a ``return``.

    DO NOT "simplify" this back to `ast.walk(tree)`. `ast.walk` visits the entire subtree,
    so a `kwargs.setdefault('flag', False)` sitting inside `if dim > 128:`, inside
    `for _ in range(0):`, inside a helper that is never called, or after a `return` was
    read as an UNCONDITIONAL default. `_effective_defaults` then overwrote the correct
    class default with it, so the guard asserted a value the caller never actually gets --
    which can turn REAL drift green. A guard that mis-models a case is strictly worse than
    one that omits it: an omission leaves you where you were, a false model certifies wrong
    values as correct while reading as coverage. See decisions.md D-008.
    """
    for stmt in func_node.body:
        if isinstance(stmt, ast.Return):
            return
        yield stmt


def _setdefault_target(node, var_kw_names):
    """``('key', value)`` if ``node`` is a literal ``<var_kw>.setdefault('key', value)``."""
    if not isinstance(node, ast.Call) or len(node.args) != 2:
        return None
    callee = node.func
    if not isinstance(callee, ast.Attribute) or callee.attr != "setdefault":
        return None
    if not isinstance(callee.value, ast.Name) or callee.value.id not in var_kw_names:
        return None
    try:
        key = ast.literal_eval(node.args[0])
        value = ast.literal_eval(node.args[1])
    except ValueError:
        return None  # not a literal -- fall back to the class default
    if not isinstance(key, str):
        return None
    return key, value


def _wrapper_kwarg_setdefaults(func):
    """``(forced, unprovable)`` for a wrapper's ``kwargs.setdefault(...)`` defaults.

    # DECISION plan-2026-07-20T191713-52a15234/D-008
    A registry entry whose 'class' is a wrapper function has THREE default layers, not two:
    the wrapped class's own ctor default, then whatever the wrapper forces, then the registry.
    `create_zigzag_window_attention` does `kwargs.setdefault("use_relative_position_bias", False)`
    (layers/attention/window_attention.py:762) while its sibling `create_grid_window_attention`
    sets it True (:724) -- so the registry's `False` for attention:window_zigzag is CORRECT even
    though `WindowAttention.__init__` says `True`.

    * ``forced`` -- params whose effective default this function can PROVE: a literal
      ``setdefault`` among the unconditional top-level statements (`_unconditional_body`),
      and which appears nowhere else in the function.
    * ``unprovable`` -- params with a ``setdefault`` somewhere the model cannot prove runs
      (inside a branch, a loop, a nested def, or after a ``return``). Callers must fall back
      to the wrapped class's default for these and MUST NOT adopt the literal. A key that is
      unprovable anywhere is unprovable everywhere: a conditional `setdefault` occurring
      *before* a top-level one would win, and `setdefault` is first-write-wins, so a
      top-level literal is not authoritative in that shape.

    DO NOT "fix" a red here by adding an INTENTIONAL_OVERRIDES entry. That silences a correct
    value permanently: a later edit of the wrapper's setdefault to `True`, with the registry left
    at `False`, would then never red -- reinstalling the exact blind spot this guard exists to
    close, one layer further down. Model the chain instead. See decisions.md D-005, D-008.
    """
    func_node = _function_ast(func)
    if func_node is None:
        return {}, set()

    var_kw = {
        name
        for name, param in inspect.signature(func).parameters.items()
        if param.kind == param.VAR_KEYWORD
    }

    top_level = {}
    for stmt in _unconditional_body(func_node):
        value = getattr(stmt, "value", None) if isinstance(stmt, (ast.Expr, ast.Assign)) else None
        hit = _setdefault_target(value, var_kw) if value is not None else None
        if hit is not None:
            top_level.setdefault(*hit)  # first write wins, exactly as setdefault does

    everywhere = {}
    for node in ast.walk(func_node):
        hit = _setdefault_target(node, var_kw)
        if hit is not None:
            everywhere.setdefault(*hit)

    unprovable = set(everywhere) - set(top_level)
    forced = {k: v for k, v in top_level.items() if k not in unprovable}
    return forced, unprovable


def _wrapper_pinned_params(func):
    """Params the wrapper HARD-CODES when it delegates, so a caller cannot set them.

    # DECISION plan-2026-07-20T191713-52a15234/D-008
    `create_grid_window_attention` ends in
    ``return WindowAttention(dim=dim, ..., partition_mode="grid", **kwargs)``. `dim` is a plain
    forward of the wrapper's own parameter, so the caller controls it. `partition_mode` is a
    literal the wrapper chose: passing `partition_mode=...` through `**kwargs` raises
    ``TypeError: got multiple values for keyword argument 'partition_mode'`` (verified).

    So a pinned param is NOT part of the wrapper's accepted-parameter set, and the registry is
    RIGHT not to declare it -- declaring it would be a live TypeError, not a fix. That fact is
    DERIVED from the wrapper's own source here rather than written into an exemption table, so
    it cannot rot: the day a wrapper stops pinning a param, the MISSING guard starts requiring
    it. The counterpart obligation is `test_registry_declares_no_param_the_target_rejects`,
    which reds if a registry ever DOES declare a pinned param.

    Limitation: only keyword arguments of a top-level ``return <Call>(...)`` are modeled.
    A positionally-pinned argument is not seen (none exists in this repo today).
    """
    func_node = _function_ast(func)
    if func_node is None:
        return set()

    own_params = set(inspect.signature(func).parameters)
    pinned = set()
    for stmt in func_node.body:
        if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Call):
            continue
        for keyword in stmt.value.keywords:
            if keyword.arg is None:  # **kwargs pass-through
                continue
            forwarded = (
                isinstance(keyword.value, ast.Name) and keyword.value.id in own_params
            )
            if not forwarded:
                pinned.add(keyword.arg)
        break  # only the first top-level return is reachable
    return pinned


def _wrapped_class(target):
    """The class a wrapper FUNCTION builds, via its ``return_annotation``; else ``None``.

    A class target wraps nothing. A string annotation (PEP 563) or a non-class annotation
    yields ``None``, which degrades this guard to its pre-resolution blindness -- graceful,
    but blind; see the module docstring.
    """
    if inspect.isclass(target) or not callable(target):
        return None
    wrapped = inspect.signature(target).return_annotation
    return wrapped if inspect.isclass(wrapped) else None


def _registry_entries():
    """Yield (factory, type_name, target, info) for every registered type."""
    for factory, (module_path, attr) in FACTORIES.items():
        registry = getattr(importlib.import_module(module_path), attr)
        for type_name, info in sorted(registry.items()):
            target = info.get("class")
            if target is None:
                continue
            yield factory, type_name, target, info


def _named_and_var_kw(callable_obj):
    """``(named params, accepts **kwargs)`` for a class ctor or a plain function."""
    signature = inspect.signature(
        callable_obj.__init__ if inspect.isclass(callable_obj) else callable_obj
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
    Both directions share `_wrapped_class`/`_wrapper_pinned_params` above for that reason.
    See decisions.md D-008.

    Resolution: the wrapper's own named params, PLUS the wrapped class's named params (which
    reach it through ``**kwargs``), MINUS the params the wrapper hard-codes -- those raise
    TypeError if a caller supplies them, so they are settable by nobody and belong in
    ``pinned``, not in ``named``.
    """
    named, accepts_var_kw = _named_and_var_kw(target)
    wrapped = _wrapped_class(target)
    if wrapped is None:
        return _TargetSignature(named, accepts_var_kw, frozenset())

    wrapped_named, wrapped_var_kw = _named_and_var_kw(wrapped)
    pinned = frozenset(_wrapper_pinned_params(target) - BASE_PARAMS)
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
    defaults, plus the defaults of the class named by its ``return_annotation`` for params the
    wrapper only forwards via ``**kwargs``, plus the ``kwargs.setdefault`` values the wrapper
    PROVABLY forces. Params with no default map to ``inspect.Parameter.empty``.

    # DECISION plan-2026-07-20T191713-52a15234/D-008
    A ``setdefault`` that `_wrapper_kwarg_setdefaults` cannot prove unconditional is NOT applied
    here -- the wrapped class's own default stands, which is this guard's pre-resolution answer.
    DO NOT restore the old ``defaults.update(_wrapper_kwarg_setdefaults(target))`` over an
    `ast.walk`-derived dict: that overwrote a CORRECT class default with a literal taken from
    inside an `if`, a `for`, a nested def, or dead code after a `return`, so the guard asserted a
    value the caller never receives and real drift could pass green. See decisions.md D-008.

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
    wrapped = _wrapped_class(target)
    if wrapped is None:
        return defaults

    for name, param in inspect.signature(wrapped.__init__).parameters.items():
        if name not in defaults and param.kind in kinds:
            defaults[name] = param.default
    forced, _unprovable = _wrapper_kwarg_setdefaults(target)
    defaults.update(forced)
    return defaults


def _unprovable_wrapper_params(target):
    """Params whose wrapper ``setdefault`` this guard refused to trust (see D-008).

    Used only to enrich a DRIFT failure message: if a red lands on one of these, the reader
    needs to know the comparator fell back to the class default because the wrapper's own
    ``setdefault`` sits somewhere it cannot prove runs -- otherwise they would "fix" a correct
    registry to match a default the caller never gets. Empty for every entry in this repo today.
    """
    if _wrapped_class(target) is None:
        return frozenset()
    return frozenset(_wrapper_kwarg_setdefaults(target)[1])


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
    unprovable = _unprovable_wrapper_params(target)
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

        hint = (
            f" NOTE: {target_name} calls kwargs.setdefault({param!r}, ...) somewhere this "
            f"guard cannot prove executes (inside a branch/loop/nested def, or after a "
            f"return), so it compared against the WRAPPED CLASS default rather than trusting "
            f"that literal. Read the wrapper before changing the registry -- the registry may "
            f"well be right. See decisions.md D-008."
            if param in unprovable
            else ""
        )
        failures.append(
            f"  {param}:{hint} registry default {registry_value!r} != {target_name} default "
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
# Guard-of-the-guard: the wrapper model itself
#
# Everything above trusts `_signature_params` / `_effective_defaults` to model what a wrapper
# entry really does. Both were wrong once (decisions.md D-008), and neither error was visible
# from the registries, because the only two wrappers in this repo have the SIMPLEST possible
# shape: one unconditional top-level `setdefault` each. So the shapes that broke the model do
# not exist here to be tested against -- they are constructed below.
# ---------------------------------------------------------------------


class _SyntheticWrapped:
    """Stand-in for a wrapped layer class. `flag` defaults True; that is the value at stake."""

    def __init__(self, dim, flag=True, partition_mode="grid", **kwargs):
        self.dim = dim
        self.flag = flag
        self.partition_mode = partition_mode


def _wrapper_setdefault_unconditional(dim, **kwargs) -> _SyntheticWrapped:
    """POSITIVE CONTROL. The one shape the model may trust: effective `flag` really is False."""
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


def _wrapper_forwards_partition_mode(
    dim, partition_mode="grid", **kwargs
) -> _SyntheticWrapped:
    """Forwards `partition_mode`; a caller CAN set it, so the registry must declare it."""
    return _SyntheticWrapped(dim=dim, partition_mode=partition_mode, **kwargs)


def _wrapper_pins_partition_mode(dim, **kwargs) -> _SyntheticWrapped:
    """Hard-codes `partition_mode`; a caller supplying it gets TypeError, so it is not settable."""
    return _SyntheticWrapped(dim=dim, partition_mode="zigzag", **kwargs)


_NON_TOP_LEVEL_WRAPPERS = [
    _wrapper_setdefault_in_if,
    _wrapper_setdefault_in_for,
    _wrapper_setdefault_in_nested_def,
    _wrapper_setdefault_after_return,
]


def test_wrapper_setdefault_reads_a_real_unconditional_default():
    """POSITIVE CONTROL for the four negative cases below.

    Why this can fail if the implementation is wrong: an over-restrictive scan -- or one that
    simply returns `{}` -- would satisfy every "must fall back to the class default" assertion
    below while modelling nothing at all. This is what makes those four non-vacuous. It also
    holds the line on the real chain: without it, "fix" the false-model bug by disabling wrapper
    setdefault resolution entirely and the suite stays green while attention:window_zigzag's
    correct `False` starts reading as drift.
    """
    forced, unprovable = _wrapper_kwarg_setdefaults(_wrapper_setdefault_unconditional)
    assert forced == {"flag": False}, (
        f"an unconditional top-level kwargs.setdefault('flag', False) was not read: {forced!r}"
    )
    assert not unprovable
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


@pytest.mark.parametrize(
    "wrapper", _NON_TOP_LEVEL_WRAPPERS, ids=lambda w: w.__name__
)
def test_wrapper_setdefault_below_top_level_is_not_adopted(wrapper):
    """A `setdefault` this guard cannot prove runs must NOT overwrite the class default.

    Why this can fail if the implementation is wrong: it DID fail, for all four shapes, until
    decisions.md D-008. `ast.walk` visits the whole function tree, so each `setdefault` here was
    read as unconditional and `_effective_defaults` returned False -- a value no caller of these
    wrappers ever receives. The guard would then assert a registry `False` as correct and a
    registry `True` (the truth) as drift: real drift green, correct code red. That is a
    correctness defect in the safety mechanism, not a coverage gap.
    """
    assert wrapper(dim=32).flag is True, (
        "the fixture is not modelling what it claims -- this wrapper's real effective `flag` "
        "must be the CLASS default True, or the assertions below prove nothing"
    )

    forced, unprovable = _wrapper_kwarg_setdefaults(wrapper)
    assert "flag" not in forced, (
        f"{wrapper.__name__}: kwargs.setdefault('flag', False) is not on the unconditional "
        f"top-level path, but the model adopted it as {forced.get('flag')!r}"
    )
    assert "flag" in unprovable, (
        f"{wrapper.__name__}: the setdefault must still be SEEN and reported as unprovable, so "
        f"a DRIFT message can tell the reader why the class default was used instead"
    )
    assert _effective_defaults(wrapper)["flag"] is True, (
        f"{wrapper.__name__}: effective default must fall back to the _SyntheticWrapped class "
        f"default True, not to the unreachable literal False"
    )
    assert "flag" in _unprovable_wrapper_params(wrapper)


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


def test_wrapper_pinned_params_are_derived_not_exempted():
    """`partition_mode` is excluded from the MISSING set because it is UNSETTABLE, not waived.

    Why this can fail if the implementation is wrong: a blanket exemption would exclude it
    unconditionally. Here the exclusion is derived from the wrapper's own delegation call, so a
    wrapper that FORWARDS the param instead of pinning it is required to declare it. If the
    derivation degenerated to "always exclude", the forwarding case below would also be excluded
    and this reds.
    """
    pins = _signature_params(_wrapper_pins_partition_mode)
    assert "partition_mode" in pins.pinned
    assert "partition_mode" not in pins.named

    forwards = _signature_params(_wrapper_forwards_partition_mode)
    assert "partition_mode" not in forwards.pinned
    assert "partition_mode" in forwards.named, (
        "a forwarded param must stay in the MISSING set -- a registry that fails to declare it "
        "silently drops the caller's value, which is the whole point of this file"
    )
    assert "dim" not in forwards.pinned, (
        "a plain forward of the wrapper's own parameter (dim=dim) is not a pin"
    )

    for real in (create_grid_window_attention, create_zigzag_window_attention):
        assert "partition_mode" in _signature_params(real).pinned, (
            f"{real.__name__} hard-codes partition_mode; if it no longer does, the registry "
            f"must start declaring it and this test must be re-derived, not deleted"
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
        "registry must declare it (test_wrapper_pinned_params_are_derived_not_exempted covers "
        "the derivation) and this expectation must be rewritten deliberately"
    )
