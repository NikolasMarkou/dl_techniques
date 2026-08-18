"""
Knob-sensitivity oracle -- proves a ``test_different_X`` sweep can actually fail
===============================================================================

This module is an *instrument*, not a test suite. It is deliberately named
without a ``test_`` prefix so pytest does not collect it, mirroring
``tests/test_models/smoke_contract_oracle.py`` and
``tests/test_models/test_sam/dead_component_oracle.py``. Its own RED proofs live
in ``tests/test_models/test_knob_sensitivity_oracle.py``.

Why it exists
-------------
``grep -rn "def test_(different|various|varying)_" tests/test_models/`` matched
75 functions. Roughly 45 of them sweep a *semantic* knob -- routing iterations,
tree depth, ffn type, normalization type, mLSTM ratio, stack layout, head
layout, depth/activation/kernel size -- and then assert only that the output
SHAPE is unchanged. That is the exact inversion of the claim the sweep exists to
make: the shape is identical whether or not the kwarg ever reached the graph.
The sibling anti-pattern is the *knob echo*, ``assert model.d_state == d_state``,
which proves the constructor stored the argument and nothing else.

The trap in the obvious fix
---------------------------
"Collect the outputs and assert they differ" is the right idea and the wrong
instrument for half of these rows. Two models built with different ``depth``
values have different weight *shapes*, so they consume different draws from the
RNG and their outputs differ **whether or not the kwarg was honoured**. An
output-difference assertion on a structural knob is satisfied by random-init
luck alone; it is a second unfalsifiable test wearing a stronger-looking
assertion.

So the knob has to be classified first, and each class gets its own instrument:

STRUCTURAL knob -- changes the parameterisation
    ``depth``, ``num_blocks``, ``filters``, ``kernel_size``, ``num_heads``,
    ``expand``, ``d_state``, ``layer_configs``, ... The discriminating fact is
    that the model's WEIGHT-SHAPE SIGNATURE must change.
    :func:`assert_structural_knob_changes_weights` asserts exactly that. It
    cannot be satisfied by RNG luck, and it fails precisely when the kwarg is
    dropped on the floor -- which is the defect class this whole sweep is for.

VALUE knob -- same parameterisation, different arithmetic
    ``activation``, ``norm_epsilon``, ``training`` mode, a normalization or FFN
    variant that happens to hold the same weights, ... Here the signature is
    identical, so with the SAME seed the two models hold bit-identical weights
    and any output difference is attributable to the knob ALONE.
    :func:`assert_value_knob_changes_output` asserts both halves: signature
    identical (otherwise the comparison is contaminated by different draws) AND
    outputs differ by more than a stated tolerance.

SCOPED value knob -- honoured in some of the tree, dropped in the rest
    ``kernel_initializer`` on a model that forwards it to its transformer
    blocks but not to its patch embedding; ``initializer_range`` that reaches
    the word embeddings but not the decoder blocks. The whole-model output
    already differs between two configurations at HEAD -- via the parts that DO
    honour the knob -- so :func:`assert_value_knob_changes_output` passes on the
    broken tree and proves nothing about the part under test.
    :func:`assert_scoped_value_knob_changes_weights` compares the WEIGHT VALUES
    of a named subtree instead, with the full signature still pinned identical.

Seeding
-------
Every builder is invoked immediately after ``keras.utils.set_random_seed(seed)``
with the SHIPPED initializers left alone. Without that, two configs differ by
their random draw and the assertion proves nothing; with a hand-substituted
constant initializer, a bias-free or zero-init path becomes structurally
unobservable. ``plans/SYSTEM.md`` additionally records that a statistic reading
the process-global RNG is coupled to pytest COLLECTION ORDER, so the seed is set
inside this module rather than left to a fixture or to import order.

Tolerance
---------
The default ``atol=1e-5`` is a floor, not a derivation. When a comparison lands
near it, measure the defect signal (what the delta is when the knob genuinely is
inert, e.g. by passing the same value twice) and set the bound from THAT with
the margin written at the call site. TF32 has decided a tolerance twice in this
plan already (D-032, D-036): a matmul-heavy forward can differ from itself by
~2e-04 with TF32 enabled and by exactly 0.0 with it disabled.

When a knob measures INERT
--------------------------
That is a finding about the MODEL, not a tolerance to widen. Do not soften the
assertion. Use :func:`knob_output_deltas` to record the measured number, mark
the test ``@pytest.mark.xfail(strict=True, reason="<measured>: ...")`` and log
it in ``plans/.../decisions.md``.

How to use it
-------------
::

    def test_different_depths():
        builders = {d: lambda d=d: create_bfunet_denoiser(depth=d) for d in (3, 4, 5)}
        assert_structural_knob_changes_weights(builders, knob="depth")

    def test_different_activations():
        x = np.random.rand(1, 64, 64, 1).astype("float32")
        builders = {a: lambda a=a: create_bfunet_denoiser(activation=a)
                    for a in ("relu", "gelu")}
        assert_value_knob_changes_output(builders, x, knob="activation")

Note the ``a=a`` default-argument binding: a bare closure over the loop variable
captures the LAST value for every entry, which would make every builder
identical and every one of these assertions vacuous in the quietest possible
way.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Optional, Sequence, Tuple

import keras
import numpy as np

__all__ = [
    "weight_signature",
    "build_seeded",
    "knob_output_deltas",
    "assert_structural_knob_changes_weights",
    "assert_value_knob_changes_output",
    "assert_scoped_value_knob_changes_weights",
    "weights_in_scope",
    "as_array",
]

Builder = Callable[[], Any]
Builders = Dict[Hashable, Builder]

DEFAULT_SEED = 1234
DEFAULT_ATOL = 1e-5


def weight_signature(model: Any) -> Tuple[Tuple[int, ...], ...]:
    """Ordered tuple of every weight's shape.

    This is the parameterisation fingerprint. Two models whose signatures are
    equal consume the same RNG draws under the same seed and therefore hold
    bit-identical weights; two models whose signatures differ do not, and their
    outputs cannot be compared for knob sensitivity.

    Returns an empty tuple for an unbuilt model -- callers must build first. A
    SUBCLASSED ``keras.Model`` is unbuilt until its first ``call()``, so a
    signature taken straight after the constructor is ``()`` and compares equal
    to every other unbuilt model's. See the D-003 anchor below.
    """
    return tuple(tuple(int(d) for d in w.shape) for w in model.weights)


def build_seeded(build_fn: Builder, seed: int = DEFAULT_SEED) -> Any:
    """Seed the global RNG, then build. Shipped initializers are untouched."""
    keras.utils.set_random_seed(seed)
    return build_fn()


def as_array(value: Any) -> np.ndarray:
    """Convert a backend tensor to a numpy array."""
    return np.asarray(keras.ops.convert_to_numpy(value))


def _forward(model: Any, x: Any, extract: Optional[Callable[[Any], Any]]) -> np.ndarray:
    out = model(x, training=False)
    if extract is not None:
        out = extract(out)
    return as_array(out)


def _ordered(builders: Builders) -> Sequence[Hashable]:
    keys = list(builders)
    if len(keys) < 2:
        raise ValueError(
            "a knob sweep needs at least two configurations; got "
            f"{len(keys)} ({keys!r})"
        )
    return keys


def knob_output_deltas(
    builders: Builders,
    x: Any,
    *,
    extract: Optional[Callable[[Any], Any]] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[Tuple[Hashable, Hashable], float]:
    """Measure ``max|out[a] - out[b]|`` for each adjacent pair. Asserts nothing.

    Use this to obtain the number that goes into an xfail reason or a decisions
    entry when a knob turns out to be inert.
    """
    keys = _ordered(builders)
    outs = {k: _forward(build_seeded(builders[k], seed), x, extract) for k in keys}
    return {
        (a, b): float(np.max(np.abs(outs[a] - outs[b])))
        for a, b in zip(keys, keys[1:])
    }


# DECISION plan-2026-08-17T183311-79c63e38/D-037
# Two instruments, not one. Do NOT collapse these into a single
# `assert not np.allclose(outs[0], outs[-1])` helper, however much the 45 rows
# it would replace argue for it: for a STRUCTURAL knob that assertion is
# satisfied by the different random draw alone (two builds with different
# weight shapes consume different RNG values), so it would pass on a model that
# drops the kwarg entirely -- the exact defect this file exists to convict.
# Nor should the value instrument's signature pre-check be relaxed to "compare
# whatever overlaps": the pre-check is what makes its output difference
# attributable to the knob rather than to initialisation. See D-037 in
# plans/plan-2026-08-17T183311-79c63e38/decisions.md.
def assert_structural_knob_changes_weights(
    builders: Builders,
    *,
    knob: str,
    seed: int = DEFAULT_SEED,
) -> Dict[Hashable, Tuple[Tuple[int, ...], ...]]:
    """Assert a structural knob changes the model's weight-shape signature.

    Adjacent configurations are compared, so a sweep of N values yields N-1
    independent claims rather than one first-vs-last claim that a single
    responsive value in the middle could carry.

    Returns the per-value signatures so the caller can make a stronger,
    knob-specific claim on top (e.g. monotone parameter growth in ``depth``).
    """
    keys = _ordered(builders)
    signatures = {}
    for k in keys:
        model = build_seeded(builders[k], seed)
        signatures[k] = weight_signature(model)
        if not signatures[k]:
            raise AssertionError(
                f"{knob}={k!r} produced a model with no weights; the builder "
                "must return a BUILT model for the signature to mean anything"
            )
    for a, b in zip(keys, keys[1:]):
        if signatures[a] == signatures[b]:
            raise AssertionError(
                f"{knob} is a no-op: {knob}={a!r} and {knob}={b!r} produce an "
                f"identical weight-shape signature "
                f"({len(signatures[a])} weights, "
                f"{sum(int(np.prod(s)) for s in signatures[a])} parameters). "
                "The kwarg is not reaching the parameterisation."
            )
    return signatures


def assert_value_knob_changes_output(
    builders: Builders,
    x: Any,
    *,
    knob: str,
    atol: float = DEFAULT_ATOL,
    extract: Optional[Callable[[Any], Any]] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[Tuple[Hashable, Hashable], float]:
    """Assert a value knob changes the output, with the weights held identical.

    Two claims, in order:

    1. Every configuration has the SAME weight-shape signature. Under a fixed
       seed that makes the weights bit-identical, so step 2's difference is
       attributable to the knob and not to a different random draw. If this
       fails the knob is structural -- use
       :func:`assert_structural_knob_changes_weights` instead.
    2. Each adjacent pair's ``max|delta|`` exceeds ``atol``.

    Returns the measured deltas.
    """
    keys = _ordered(builders)
    outs: Dict[Hashable, np.ndarray] = {}
    signatures: Dict[Hashable, Tuple[Tuple[int, ...], ...]] = {}
    for k in keys:
        model = build_seeded(builders[k], seed)
        # DECISION plan-2026-08-18T111512-29569f8b/D-003
        # The signature is captured AFTER the forward pass, and the order is
        # load-bearing -- do NOT "tidy" it back next to the build. A SUBCLASSED
        # keras.Model has len(model.weights) == 0 until its first call(). Most
        # call sites hand-warm the model inside their local _build helper and
        # so were unaffected, but one did not (test_mamba_v1.py's norm_epsilon
        # sweep, measured 0 weight tensors pre-forward and 25 post-forward), and
        # nothing stopped the next one from doing the same. With the capture
        # before _forward, such a pair's signatures were
        # the EMPTY TUPLE, () == () compared equal for free, and clause 1 -- the
        # clause whose entire job is to make clause 2's delta attributable to
        # the KNOB rather than to a different random draw -- could not fail.
        # Proven RED by
        # test_knob_sensitivity_oracle.py::TestSubclassedModelSignatureOrdering
        # ::test_a_structural_knob_on_subclassed_models_is_rejected, which two
        # subclassed arms of genuinely different weight shapes made pass at
        # HEAD. See D-003 in
        # plans/plan-2026-08-18T111512-29569f8b/decisions.md.
        outs[k] = _forward(model, x, extract)
        signatures[k] = weight_signature(model)
        if not signatures[k]:
            raise AssertionError(
                f"{knob}={k!r} produced a model with no weights even after a "
                "forward pass; the signature comparison below would be vacuous"
            )

    first = keys[0]
    for k in keys[1:]:
        if signatures[k] != signatures[first]:
            raise AssertionError(
                f"{knob}={k!r} does not share a weight-shape signature with "
                f"{knob}={first!r} "
                f"({len(signatures[k])} tensors / "
                f"{sum(int(np.prod(w)) for w in signatures[k])} parameters vs "
                f"{len(signatures[first])} tensors / "
                f"{sum(int(np.prod(w)) for w in signatures[first])} parameters; "
                "the counts may match while the shapes do not). "
                "This is a STRUCTURAL knob: "
                "its configurations draw different random numbers, so an output "
                "difference between them proves nothing. Use "
                "assert_structural_knob_changes_weights."
            )

    deltas = {}
    for a, b in zip(keys, keys[1:]):
        delta = float(np.max(np.abs(outs[a] - outs[b])))
        deltas[(a, b)] = delta
        if not (delta > atol):
            raise AssertionError(
                f"{knob} is a no-op: {knob}={a!r} and {knob}={b!r} hold "
                f"bit-identical weights (same signature, same seed) and their "
                f"outputs differ by max|delta| = {delta:.6e} <= atol={atol:.1e}. "
                "The kwarg is not reaching the forward pass. Do NOT widen atol "
                "to make this pass -- record the measurement and xfail(strict)."
            )
    return deltas


def _weight_path(w: Any) -> str:
    """Best-effort stable identifier for a weight, for scope matching."""
    return str(getattr(w, "path", None) or getattr(w, "name", ""))


def weights_in_scope(model: Any, scope: str) -> Sequence[Any]:
    """Every weight of ``model`` whose path contains the substring ``scope``.

    Interface contract (used by
    :func:`assert_scoped_value_knob_changes_weights` and by call sites that want
    to make a stronger claim on the same subtree):

    * ``model`` must already be BUILT -- a subclassed ``keras.Model`` has no
      weights until its first ``call()``, so an unbuilt model yields ``[]``
      here and every downstream comparison is vacuous.
    * Matching is on ``Variable.path`` (falling back to ``.name``), which for a
      sublayer created with ``name="patch_embed"`` contains ``"patch_embed"``.
    * Returns the weights in ``model.weights`` order, which is stable across
      two builds of the same parameterisation.
    """
    return [w for w in model.weights if scope in _weight_path(w)]


# DECISION plan-2026-08-18T140459-7991552f/D-021
# A THIRD instrument, deliberately. Do NOT redirect these call sites to
# assert_value_knob_changes_output because "it is a value knob and the
# signature matches" -- for a knob the model forwards to MOST of its tree and
# drops on ONE sublayer (ViT/Swin `kernel_initializer` reaching every
# transformer block but not the patch embedding; TextDecoder `initializer_range`
# reaching the word embeddings but not the decoder blocks), two arms already
# produce different whole-model outputs AT HEAD, through the parts that do
# honour the knob. The output instrument therefore passes on the broken tree:
# it is unfalsifiable for exactly the defect it would be aimed at. The
# discriminating fact is the VALUES of the dropped subtree's own weights, which
# are bit-identical across arms while the kwarg is dropped and differ once it is
# forwarded. See D-021 in
# plans/plan-2026-08-18T140459-7991552f/decisions.md.
def assert_scoped_value_knob_changes_weights(
    builders: Builders,
    x: Any,
    *,
    knob: str,
    scope: str,
    seed: int = DEFAULT_SEED,
    extract: Optional[Callable[[Any], Any]] = None,
) -> Dict[Tuple[Hashable, Hashable], float]:
    """Assert a value knob reaches the weights of ONE named subtree.

    Three claims, in order:

    1. Every configuration has the same whole-model weight-shape signature, so
       under a fixed seed the arms consume identical RNG draws and step 3's
       difference cannot be an artefact of a different draw.
    2. ``scope`` selects at least one weight, with identical shapes in every
       arm. A scope that matches nothing would make step 3 vacuously true in
       the quietest possible way.
    3. Each adjacent pair's scoped weights differ: ``max|delta| > 0``. The bound
       is exact zero rather than a tolerance because the arms hold BIT-identical
       weights while the kwarg is dropped -- there is no floating-point noise to
       absorb, and a tolerance here would only hide a partially-honoured knob.

    :param builders: ``{knob value: zero-argument builder}``; at least two.
    :param x: Input for the one forward pass that materialises the weights of a
        subclassed model. Its VALUE is irrelevant to the assertion.
    :param knob: Knob name, for the failure message.
    :param scope: Substring matched against each weight's path.
    :param seed: Global seed set immediately before each build.
    :param extract: Optional selector applied to the forward output; supplied
        only so models returning a dict/tuple can be warmed without error.
    :return: The measured per-adjacent-pair maxima.
    """
    keys = _ordered(builders)
    signatures: Dict[Hashable, Tuple[Tuple[int, ...], ...]] = {}
    scoped: Dict[Hashable, Sequence[np.ndarray]] = {}
    for k in keys:
        model = build_seeded(builders[k], seed)
        _forward(model, x, extract)
        signatures[k] = weight_signature(model)
        if not signatures[k]:
            raise AssertionError(
                f"{knob}={k!r} produced a model with no weights even after a "
                "forward pass; every comparison below would be vacuous"
            )
        selected = weights_in_scope(model, scope)
        if not selected:
            raise AssertionError(
                f"{knob}={k!r}: scope {scope!r} matched none of the model's "
                f"{len(signatures[k])} weights. Paths available: "
                f"{sorted({_weight_path(w) for w in model.weights})}"
            )
        scoped[k] = [as_array(w) for w in selected]

    first = keys[0]
    for k in keys[1:]:
        if signatures[k] != signatures[first]:
            raise AssertionError(
                f"{knob}={k!r} does not share a weight-shape signature with "
                f"{knob}={first!r} ({len(signatures[k])} vs "
                f"{len(signatures[first])} tensors). This is a STRUCTURAL knob: "
                "its configurations draw different random numbers. Use "
                "assert_structural_knob_changes_weights."
            )
        if [a.shape for a in scoped[k]] != [a.shape for a in scoped[first]]:
            raise AssertionError(
                f"{knob}={k!r}: scope {scope!r} selected a different set of "
                "weight shapes than "
                f"{knob}={first!r}; the scope is not naming the same subtree."
            )

    deltas: Dict[Tuple[Hashable, Hashable], float] = {}
    for a, b in zip(keys, keys[1:]):
        delta = max(
            float(np.max(np.abs(wa - wb))) for wa, wb in zip(scoped[a], scoped[b])
        )
        deltas[(a, b)] = delta
        if not (delta > 0.0):
            raise AssertionError(
                f"{knob} does not reach {scope!r}: {knob}={a!r} and "
                f"{knob}={b!r} leave the {len(scoped[a])} weight tensors under "
                f"{scope!r} BIT-IDENTICAL (max|delta| = {delta:.6e}) while the "
                "rest of the model's parameterisation is unchanged. The kwarg "
                "is being dropped at that construction site. Do NOT relax this "
                "to a tolerance -- record the measurement and xfail(strict)."
            )
    return deltas
