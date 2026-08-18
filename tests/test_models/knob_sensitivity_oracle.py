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
