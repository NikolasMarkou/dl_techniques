"""
Precision-arm oracle -- the R-088 / R-141 instrument, with its own anti-vacuity
==============================================================================

This module is an *instrument*, not a test suite. Like
``smoke_contract_oracle.py`` and ``gradient_flow_oracle.py`` beside it, it is
deliberately named without a ``test_`` prefix so pytest does not collect it.
Its own RED proofs live in ``tests/test_models/test_precision_arm_oracle.py``.

Why it exists
-------------
Audit rules R-088 ("a mixed-precision arm exists") and R-141 ("that arm has all
four parts") were charged against ~55 of the 73 ``models/`` test directories.
Writing 55 bespoke fp16 tests would be 55 chances to write one that cannot fail,
and the plan has already caught five assertions of exactly that kind. There is
one correct shape for this arm and it lives here once.

The four parts of a precision arm
---------------------------------
1. **The mixed-precision forward runs at all.** Most fp16 defects in this tree
   are a *raise*, not a NaN: an ``fft2`` with no half kernel, a hard-coded
   ``dtype="float32"`` constant meeting an autocast tensor, a mask sentinel
   that overflows. A test that never sets the policy cannot see any of them.
2. **It produces the compute dtype.** A layer that quietly returns float32
   under ``mixed_float16`` has not failed -- Keras promotes and the forward
   completes -- but it has opted its consumer out of mixed precision. Only an
   assertion on the OUTPUT TENSOR's dtype sees this.
3. **It is finite.** Guards written as float32 literals (``1e-9``, ``1e-12``)
   are *exactly zero* in float16, so an fp16 arm that only checks "it ran"
   passes over a division by zero that has not been reached yet.
4. **The BACKWARD pass runs.** Step 5.8 of this plan measured that four of five
   models with a green fp16 forward raised again inside ``train_step``: the
   gradient of a cast island travels a different path from the forward. A
   forward-only arm is half an instrument.

Assertions this oracle deliberately does NOT make
-------------------------------------------------
* ``assert model.dtype == "float16"``. ``Layer.dtype`` reports the *variable*
  dtype, which is ``'float32'`` under ``mixed_float16`` too -- so that assertion
  passes against the very defect it names. :func:`assert_precision_arm` asserts
  ``model.dtype_policy.name`` instead, and separately asserts the dtype of the
  returned tensors.
* A tight numeric comparison against the float32 arm. float16 has ~3 decimal
  digits; a ``rtol=1e-6`` comparison would fail on every correct model. The
  float32 control here exists to prove the fp16 arm is not comparing garbage to
  garbage, at a tolerance stated by the caller.

Vacuity control built in
------------------------
:func:`assert_precision_arm` asserts that the global policy really is
``mixed_float16`` *inside* the arm, and that the model it was handed carries
that policy. A test whose model was built before the policy was set -- the most
common way one of these arms becomes silently a float32 test -- fails here
rather than passing green.

How to use it
-------------
::

    from ..precision_arm_oracle import assert_precision_arm, precision_policy

    def test_the_model_runs_under_mixed_float16():
        assert_precision_arm(
            build=lambda: create_my_model(width=8),
            make_inputs=lambda: np.random.RandomState(0).randn(1, 32, 32, 3),
        )

``build`` is a *callable*, not a model: it is invoked INSIDE the policy context
so the model's variables and sub-layers are created under it.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import keras
import numpy as np
from keras import ops


__all__ = [
    "precision_policy",
    "flatten_tensors",
    "PrecisionArmReport",
    "run_forward",
    "run_backward",
    "assert_precision_arm",
]


@contextlib.contextmanager
def precision_policy(name: str) -> Iterator[None]:
    """
    Set the global Keras dtype policy for the duration of the block.

    The policy is process-global, so it is restored in a ``finally`` -- a test
    that leaves ``mixed_float16`` set would silently re-type every test that
    runs after it in the same process.

    :param name: A policy name, e.g. ``"mixed_float16"`` or ``"float32"``.
    :type name: str
    """
    previous = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy(name)
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


def flatten_tensors(output: Any) -> List[Any]:
    """
    Return every tensor leaf of a model output, in a stable order.

    ``keras.tree`` is deliberately not used: several models in this tree return
    dicts holding a ``None`` value, and the tree libraries disagree about
    whether ``None`` is a leaf.

    :param output: A tensor, a sequence of tensors, or a mapping of them.
    :return: The tensor leaves, in dict-insertion / sequence order.
    :rtype: list
    """
    if isinstance(output, dict):
        items: Sequence[Any] = list(output.values())
    elif isinstance(output, (list, tuple)):
        items = list(output)
    else:
        items = [output]
    return [t for t in items if hasattr(t, "shape") and hasattr(t, "dtype")]


class PrecisionArmReport(dict):
    """A measurement, not a verdict. Keys are documented at the call sites."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - convenience
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def run_forward(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        policy: str,
        training: bool = False,
        seed: Optional[int] = 0,
) -> PrecisionArmReport:
    """
    Build a model and run one forward pass, entirely inside ``policy``.

    :param build: Zero-argument callable returning the model. Called INSIDE the
        policy context so variables are created under it.
    :param make_inputs: Zero-argument callable returning the model's input(s).
    :param policy: Global dtype policy name to run under.
    :param training: Value forwarded to the model's ``training`` argument.
    :param seed: Seeded immediately before ``build()``. This is load-bearing for
        any cross-arm numeric comparison: without it the fp16 and float32 arms
        are two DIFFERENT random models and their ``absmax`` values differ by
        far more than a precision effect (measured: 2.605469 vs 1.821816 on a
        6->4 Dense, a 43% gap that has nothing to do with float16). Pass
        ``None`` only when the comparison is not being made.
    :return: A report with ``policy_seen``, ``model_policy``, ``dtypes``,
        ``n_nan``, ``n_inf``, ``size`` and ``absmax``.
    :rtype: PrecisionArmReport
    """
    with precision_policy(policy):
        seen = keras.mixed_precision.global_policy().name
        if seed is not None:
            keras.utils.set_random_seed(seed)
        model = build()
        tensors = flatten_tensors(model(make_inputs(), training=training))
        arrays = [np.asarray(ops.convert_to_numpy(t)) for t in tensors]
        finite = [a[np.isfinite(a)] for a in arrays]
        return PrecisionArmReport(
            policy_seen=seen,
            model_policy=getattr(model.dtype_policy, "name", None),
            n_tensors=len(tensors),
            dtypes=[str(t.dtype).replace("<dtype: '", "").strip("'>") for t in tensors],
            n_nan=[int(np.isnan(a).sum()) for a in arrays],
            n_inf=[int(np.isinf(a).sum()) for a in arrays],
            size=[int(a.size) for a in arrays],
            absmax=[float(np.abs(f).max()) if f.size else float("nan") for f in finite],
            arrays=arrays,
        )


def run_backward(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        policy: str,
        seed: Optional[int] = 0,
) -> PrecisionArmReport:
    """
    Build a model and take one gradient of a sum-of-squares loss, under ``policy``.

    The loss is accumulated in float32 on purpose: a float16 reduction over a
    large tensor is itself an overflow hazard, and this oracle exists to judge
    the MODEL, not the loss it was handed.

    :return: A report with ``loss``, ``n_vars``, ``n_none``, ``n_nonfinite``
        and ``grad_norm_sum``.
    :rtype: PrecisionArmReport
    """
    import tensorflow as tf

    with precision_policy(policy):
        if seed is not None:
            keras.utils.set_random_seed(seed)
        model = build()
        inputs = make_inputs()
        with tf.GradientTape() as tape:
            tensors = flatten_tensors(model(inputs, training=True))
            loss = sum(
                ops.mean(ops.square(ops.cast(t, "float32"))) for t in tensors
            )
        grads = tape.gradient(loss, model.trainable_variables)
        norms = [
            float(ops.convert_to_numpy(
                ops.sqrt(ops.sum(ops.square(ops.cast(g, "float32"))))
            ))
            for g in grads if g is not None
        ]
        return PrecisionArmReport(
            loss=float(ops.convert_to_numpy(loss)),
            n_vars=len(grads),
            n_none=sum(g is None for g in grads),
            n_nonfinite=sum(not np.isfinite(v) for v in norms),
            grad_norm_sum=float(sum(norms)),
        )


def assert_precision_arm(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        *,
        expected_compute_dtype: str = "float16",
        check_backward: bool = True,
        allowed_none_grads: int = 0,
        rtol_against_float32: Optional[float] = None,
) -> Dict[str, PrecisionArmReport]:
    """
    Assert all four parts of a ``mixed_float16`` arm, with a float32 control.

    :param build: Zero-argument model factory, invoked inside each policy.
    :param make_inputs: Zero-argument input factory. Must be deterministic if
        ``rtol_against_float32`` is used.
    :param expected_compute_dtype: The dtype every returned tensor must carry
        under ``mixed_float16``. Pass ``None`` to skip that part, but only with
        a recorded reason -- part 2 of the arm is the one most often dropped.
    :param check_backward: Run part 4. Only turn this off for a model with no
        trainable variables.
    :param allowed_none_grads: Number of ``None`` gradients that are expected
        AND are equally ``None`` under float32. Must be justified at the call
        site; the default of 0 is the correct value for a healthy model.
    :param rtol_against_float32: If given, the fp16 and float32 arms' ``absmax``
        per tensor must agree to this relative tolerance. ``1e-2`` is a
        realistic value for half precision.
    :return: ``{"mixed_float16": ..., "float32": ..., "backward_mixed_float16":
        ...}``, plus ``"float32_control"`` and ``"float32_build_spread"`` when
        ``rtol_against_float32`` was requested.
    :raises AssertionError: on any failed part.
    """
    fp16 = run_forward(build, make_inputs, "mixed_float16")
    f32 = run_forward(build, make_inputs, "float32")

    # Vacuity control: the arm must actually have run under the policy it names.
    assert fp16["policy_seen"] == "mixed_float16", (
        f"the mixed-precision arm ran under {fp16['policy_seen']!r}; the model "
        "was probably built outside the policy context"
    )
    assert fp16["model_policy"] == "mixed_float16", (
        f"the model carries dtype_policy {fp16['model_policy']!r}, not "
        "'mixed_float16' -- it was constructed before the policy was set, so "
        "this arm is a float32 test wearing an fp16 name"
    )
    assert f32["model_policy"] == "float32", (
        f"the float32 control carries dtype_policy {f32['model_policy']!r}; "
        "the policy context did not restore"
    )
    assert fp16["n_tensors"] > 0, "the model returned no tensors to judge"
    assert fp16["n_tensors"] == f32["n_tensors"], (
        f"arms disagree on output arity: {fp16['n_tensors']} vs {f32['n_tensors']}"
    )

    # Part 2 -- the compute dtype really reached the output.
    if expected_compute_dtype is not None:
        assert all(d == expected_compute_dtype for d in fp16["dtypes"]), (
            f"under mixed_float16 the outputs are {fp16['dtypes']}, expected "
            f"every one to be {expected_compute_dtype!r}; a float32 output means "
            "the model silently opted its consumer out of mixed precision"
        )

    # Part 3 -- finiteness, in BOTH arms (an fp16 NaN is only a finding if the
    # float32 control is clean; an untrained BatchNorm can produce a false NaN
    # reading in either arm).
    assert sum(f32["n_nan"]) == 0 and sum(f32["n_inf"]) == 0, (
        f"the float32 CONTROL is already non-finite (nan={f32['n_nan']}, "
        f"inf={f32['n_inf']}); the fp16 reading below would prove nothing"
    )
    assert sum(fp16["n_nan"]) == 0, (
        f"mixed_float16 forward produced NaN: {fp16['n_nan']} of {fp16['size']}"
    )
    assert sum(fp16["n_inf"]) == 0, (
        f"mixed_float16 forward produced Inf: {fp16['n_inf']} of {fp16['size']}"
    )

    reports = {"mixed_float16": fp16, "float32": f32}

    if rtol_against_float32 is not None:
        # SELF-CALIBRATING TOLERANCE.
        #
        # DECISION plan-2026-08-19T163559-499b6f0e/D-055
        # A cross-arm numeric comparison silently assumes the model rebuilds
        # identically under a fixed seed. Several models here do NOT:
        # ``coshnet`` measured absmax 0.2797602 / 0.3045650 / 0.2811624 over
        # THREE consecutive float32 builds in one process, all after
        # ``keras.utils.set_random_seed(0)``. Comparing fp16 against float32
        # under a flat ``rtol`` therefore reports a precision finding where the
        # real cause is a non-seedable initializer -- this instrument did
        # exactly that before this control existed.
        #
        # The second float32 run is the CONTROL: it measures the model's own
        # build spread, and the fp16 arm is required to stay inside
        # ``max(rtol * |float32|, 3 * spread)``. For a reproducible model the
        # spread is 0.0 and the check is the flat ``rtol`` it looks like; for a
        # non-reproducible one it degrades honestly instead of lying.
        # Do NOT replace this with a fixed larger rtol. See decisions.md D-055.
        f32_again = run_forward(build, make_inputs, "float32")
        reports["float32_control"] = f32_again
        spread = [
            abs(a - b) for a, b in zip(f32["absmax"], f32_again["absmax"])
        ]
        reports["float32_build_spread"] = spread
        for i, (a, b) in enumerate(zip(fp16["absmax"], f32["absmax"])):
            tol = max(rtol_against_float32 * max(abs(b), 1e-8), 3.0 * spread[i])
            assert abs(a - b) <= tol, (
                f"output {i}: fp16 absmax {a:.6e} vs float32 {b:.6e} -- the "
                f"arms disagree by {abs(a - b):.6e}, more than "
                f"tol={tol:.6e} (rtol={rtol_against_float32}, float32 build "
                f"spread {spread[i]:.6e})"
            )

    # Part 4 -- the backward pass.
    if check_backward:
        bwd = run_backward(build, make_inputs, "mixed_float16")
        reports["backward_mixed_float16"] = bwd
        assert bwd["n_vars"] > 0, "the model has no trainable variables"
        assert bwd["n_none"] <= allowed_none_grads, (
            f"mixed_float16 backward left {bwd['n_none']} gradients None "
            f"(allowed {allowed_none_grads}) over {bwd['n_vars']} variables"
        )
        assert bwd["n_nonfinite"] == 0, (
            f"mixed_float16 backward produced {bwd['n_nonfinite']} non-finite "
            f"gradient norms (loss={bwd['loss']})"
        )
        assert bwd["grad_norm_sum"] > 0.0, (
            "every gradient norm is exactly zero under mixed_float16 -- the "
            "backward pass reached no variable, so 'no NaN' is vacuous"
        )

    return reports
