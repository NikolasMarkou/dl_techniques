"""
Arm oracle -- R-088 / R-141 (precision), R-142 (float64) and R-143 (graph/XLA)
=============================================================================

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

Two further arms live here, on the same helpers
-----------------------------------------------
:func:`assert_float64_arm` (R-142) and :func:`assert_xla_equivalence` (R-143)
share this module's ``default_call``, ``flatten_tensors``, seeded single build
and self-calibrating spread control rather than re-deriving them. They are
different *judgements* on the same forward pass -- did the requested precision
reach the tensors, and does the traced graph compute what eager computes -- and
each carries its own anti-vacuity control, argued at its own docstring.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import keras
import numpy as np
from keras import ops


__all__ = [
    "precision_policy",
    "float64_scope",
    "assert_float64_arm",
    "assert_xla_equivalence",
    "default_call",
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


def default_call(model: Any, inputs: Any, training: bool) -> Any:
    """Invoke ``model(inputs, training=training)`` -- the Keras convention.

    Contract for a replacement passed as ``call_fn``: same three positional
    parameters, returns whatever the model returns. It exists because a few
    models in this tree do NOT take a single ``inputs`` argument -- ``TRM.call``
    is ``call(carry, batch, training)`` -- and wrapping such a model in a
    functional adapter would measure the ADAPTER's dtype behaviour, not the
    model's. The replacement must call the real model exactly once.

    :param model: The model under test.
    :param inputs: Whatever ``make_inputs`` returned.
    :param training: Forwarded verbatim.
    :return: The model output.
    """
    return model(inputs, training=training)


def _asymmetric_loss(tensor: Any) -> Any:
    """
    A scalar loss whose gradient is not annihilated by output symmetry.

    DECISION plan-2026-08-19T163559-499b6f0e/D-059
    The obvious ``mean(square(t))`` is WRONG here and was measured wrong: an
    untrained classifier's softmax head emits a uniform distribution, and the
    gradient of a symmetric function of a uniform softmax is EXACTLY zero by
    that symmetry. ``mobilenet`` measured ``grad_norm_sum`` of exactly
    ``0.000000e+00`` under BOTH ``mixed_float16`` AND ``float32`` (loss
    ``0.0625 == 1/16``, i.e. four classes each at ``0.25``), and ``squeezenet``
    measured ``0.000000e+00`` fp16 against ``1.165148e-05`` float32. Neither is
    a model defect; both are this loss meeting a saddle point. A "the backward
    pass reached no variable" assertion that fires on healthy models is an
    instrument that cannot be trusted when it fires on a sick one.

    The fix is to break the symmetry with a FIXED, deterministic ramp over the
    flattened tensor, so no permutation of the outputs leaves the loss
    invariant. Do NOT replace this with ``mean(square(t))``; do NOT make the
    ramp random -- the fp16 and float32 arms must see the same loss surface.
    See decisions.md D-059.

    The reduction is accumulated in float32 on purpose: a float16 reduction
    over a large tensor is itself an overflow hazard, and this oracle exists to
    judge the MODEL, not the loss it was handed.

    :param tensor: One model output tensor.
    :return: A float32 scalar.
    """
    x = ops.cast(tensor, "float32")
    tail = tuple(int(d) for d in x.shape[1:])
    n = int(np.prod(tail)) if tail else 1
    ramp = ops.reshape(
        ops.cast(ops.arange(n), "float32") / float(n) + 0.5, (1,) + tail
    )
    return ops.mean(ops.square(x) * ramp)


def run_forward(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        policy: str,
        training: bool = False,
        seed: Optional[int] = 0,
        call_fn: Optional[Callable[[Any, Any, bool], Any]] = None,
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
        call = call_fn or default_call
        tensors = flatten_tensors(call(model, make_inputs(), training))
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
        call_fn: Optional[Callable[[Any, Any, bool], Any]] = None,
) -> PrecisionArmReport:
    """
    Build a model and take one gradient of :func:`_asymmetric_loss`, under ``policy``.

    The loss is a RAMP-WEIGHTED sum of squares, not a plain one, and float32
    throughout; both properties are load-bearing and are argued at
    :func:`_asymmetric_loss`.

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
        call = call_fn or default_call
        with tf.GradientTape() as tape:
            tensors = flatten_tensors(call(model, inputs, True))
            loss = sum(_asymmetric_loss(t) for t in tensors)
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
        dtype_exempt_outputs: Sequence[int] = (),
        rtol_against_float32: Optional[float] = None,
        call_fn: Optional[Callable[[Any, Any, bool], Any]] = None,
        forward_training: bool = False,
) -> Dict[str, PrecisionArmReport]:
    """
    Assert all four parts of a ``mixed_float16`` arm, with a float32 control.

    :param build: Zero-argument model factory, invoked inside each policy.
    :param make_inputs: Zero-argument input factory. Must be deterministic if
        ``rtol_against_float32`` is used.
    :param expected_compute_dtype: The dtype every returned FLOATING-POINT
        tensor must carry under ``mixed_float16``. Integer outputs (masks,
        BMU indices, token ids) are exempt -- see D-058 -- but at least one
        float output must exist or the check is vacuous. Pass ``None`` to skip
        that part, but only with a recorded reason: part 2 of the arm is the
        one most often dropped.
    :param check_backward: Run part 4. Only turn this off for a model with no
        trainable variables.
    :param allowed_none_grads: Number of ``None`` gradients that are expected
        AND are equally ``None`` under float32. Must be justified at the call
        site; the default of 0 is the correct value for a healthy model.
    :param dtype_exempt_outputs: Indices into the flattened output that part 2
        skips. Reserved for a tensor that is NOT an activation -- a binary
        patch mask, an index map. Every entry must carry a stated reason at
        the call site, because this is the escape hatch that turns part 2 off.
    :param rtol_against_float32: If given, the fp16 and float32 arms' ``absmax``
        per tensor must agree to this relative tolerance. ``1e-2`` is a
        realistic value for half precision.
    :param forward_training: ``training`` for parts 1-3. The default ``False``
        judges the inference path, which is right for almost every model.

        DECISION plan-2026-08-19T163559-499b6f0e/D-065
        Pass ``True`` for a model whose UNTRAINED BatchNorms make the inference
        path meaningless. At initialization ``moving_mean = 0`` and
        ``moving_variance = 1``, so at inference every BN is a near-identity
        and nothing bounds the activation scale; in TRAINING mode BN divides by
        the BATCH statistics and the same graph stays O(1). MEASURED on
        ``yolo12`` scale ``n`` at 64x64: float32 inference ``absmax``
        2.997772e+08 at init (falling to 3.332934e+02 only after 200
        ``training=True`` passes), and the fp16 arm reported NaN in 5712 of
        5712 outputs purely because 1e8 exceeds float16's 65504 ceiling. At
        ``forward_training=True`` the SAME model measures ``absmax`` 4.703125
        fp16 against 4.644949 float32, both NaN-free. This is a statement
        about an untrained model, not about float16 -- the same false reading
        step 16 and step 17 hit. A 200-step warm-up was implemented, measured
        at 106.8s per arm against 7.3s for this, and REJECTED. Do NOT set this
        ``True`` to silence an fp16 finding: the float32 control runs in the
        same mode, so a real dtype defect still fails. See decisions.md D-065.
    :param call_fn: Replacement for :func:`default_call`, for a model whose
        ``call`` does not take a single ``inputs`` argument. Must invoke the
        REAL model -- never a functional adapter, which would measure the
        adapter's dtypes instead.
    :return: ``{"mixed_float16": ..., "float32": ..., "backward_mixed_float16":
        ...}``, plus ``"float32_control"`` and ``"float32_build_spread"`` when
        ``rtol_against_float32`` was requested.
    :raises AssertionError: on any failed part.
    """
    fp16 = run_forward(build, make_inputs, "mixed_float16", call_fn=call_fn,
                       training=forward_training)
    f32 = run_forward(build, make_inputs, "float32", call_fn=call_fn,
                      training=forward_training)

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
    #
    # DECISION plan-2026-08-19T163559-499b6f0e/D-058
    # The rule is applied to the FLOATING-POINT outputs only. Several models
    # here legitimately return an integer tensor beside their float one --
    # ``BERT`` returns an int32 ``attention_mask``, ``SOMModel`` returns int32
    # BMU indices -- and there is no such thing as a float16 index. Do NOT
    # "fix" this by passing ``expected_compute_dtype=None`` at those call
    # sites: that deletes part 2 of the arm entirely, which is the part most
    # often dropped. The float-only rule keeps it live AND is guarded from
    # becoming vacuous by the assertion below that at least one float output
    # exists -- without that, a model returning nothing but integers would
    # satisfy an ``all()`` over an empty list. See decisions.md D-058.
    if expected_compute_dtype is not None:
        float_dtypes = [d for i, d in enumerate(fp16["dtypes"])
                        if i not in set(dtype_exempt_outputs)
                        and (d.startswith("float") or d.startswith("bfloat"))]
        assert float_dtypes, (
            f"the model returned no floating-point tensor ({fp16['dtypes']}), "
            "so the compute-dtype arm would be vacuous"
        )
        assert all(d == expected_compute_dtype for d in float_dtypes), (
            f"under mixed_float16 the float outputs are {float_dtypes} (all "
            f"outputs {fp16['dtypes']}), expected every float one to be "
            f"{expected_compute_dtype!r}; a float32 output means the model "
            "silently opted its consumer out of mixed precision"
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
        f32_again = run_forward(build, make_inputs, "float32",
                                call_fn=call_fn, training=forward_training)
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
        bwd = run_backward(build, make_inputs, "mixed_float16", call_fn=call_fn)
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


# ===========================================================================
# R-142 -- the float64 arm
# ===========================================================================


@contextlib.contextmanager
def float64_scope() -> Iterator[None]:
    """Set ``floatx`` AND the global dtype policy to ``float64``, and restore both.

    DECISION plan-2026-08-19T163559-499b6f0e/D-074
    Both switches are required, and R-142's own wording ("needs
    ``keras.backend.set_floatx('float64')``, not just the mixed-precision
    policy") has it backwards for Keras 3.8. MEASURED, in one process:

    ==========================================  =================  ============
    what the arm does                           layer output       variables
    ==========================================  =================  ============
    ``set_floatx('float64')`` FIRST THING       ``float64``        ``float64``
    ``set_global_policy('float64')`` alone      ``float64``        ``float64``
    ``set_floatx('float64')`` AFTER anything
    has read ``global_policy()``                **``float32``**    **``float32``**
    ==========================================  =================  ============

    ``keras.mixed_precision.global_policy()`` lazily constructs
    ``DTypePolicy(backend.floatx())`` on first read and **caches it**;
    ``set_floatx`` does not invalidate that cache. So a float64 arm written as
    ``set_floatx`` alone is a no-op for every layer dtype the moment it is not
    the first thing in the process -- which it never is, because any earlier
    test, any ``compile()``, and this module's own ``precision_policy`` have
    all read the policy already. That is why the audit found float64 arms
    "silently running at float32", and the mechanism is the cache, not the
    direction of the switch. Do NOT reduce this to one call. See D-074.
    """
    prev_floatx = keras.backend.floatx()
    prev_policy = keras.mixed_precision.global_policy()
    keras.backend.set_floatx("float64")
    keras.mixed_precision.set_global_policy("float64")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(prev_policy)
        keras.backend.set_floatx(prev_floatx)


def _cast_float_leaves(inputs: Any, dtype: str) -> Any:
    """Return ``inputs`` with every FLOATING-point array cast to ``dtype``.

    Integer leaves (token ids, masks, hop distances) are left alone: there is
    no such thing as a float64 index, the same rule part 2 of the precision arm
    follows (D-058).
    """
    if isinstance(inputs, dict):
        return {k: _cast_float_leaves(v, dtype) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(_cast_float_leaves(v, dtype) for v in inputs)
    arr = getattr(inputs, "dtype", None)
    if arr is not None and np.issubdtype(np.dtype(str(arr)), np.floating):
        return np.asarray(inputs).astype(dtype)
    return inputs


def _float_leaf_dtypes(inputs: Any) -> List[str]:
    """Every floating-point leaf dtype of ``inputs``, in a stable order."""
    if isinstance(inputs, dict):
        items: Sequence[Any] = list(inputs.values())
    elif isinstance(inputs, (list, tuple)):
        items = list(inputs)
    else:
        items = [inputs]
    out = []
    for v in items:
        if isinstance(v, (dict, list, tuple)):
            out.extend(_float_leaf_dtypes(v))
            continue
        d = getattr(v, "dtype", None)
        if d is not None and np.issubdtype(np.dtype(str(d)), np.floating):
            out.append(str(d))
    return out


def assert_float64_arm(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        *,
        call_fn: Optional[Callable[[Any, Any, bool], Any]] = None,
        training: bool = False,
        dtype_exempt_outputs: Sequence[int] = (),
        expected_output_dtype: str = "float64",
) -> PrecisionArmReport:
    """Assert a real ``float64`` arm: the policy took, the INPUT took, the OUTPUT took.

    R-142 charges five ``models/`` test directories with naming ``float64`` in a
    test while ``set_floatx`` count is 0 across the directory, so the arm ran at
    float32 and the two "different precisions" agreed for the wrong reason. The
    three assertions here are what make it not that test:

    1. **The scope took.** ``floatx`` and the global policy are BOTH ``float64``
       inside the arm, and the model built under it carries that policy.
    2. **The input took.** Every floating-point input leaf is ``float64``, which
       R-142 names explicitly (``an assertion on inputs[0].dtype``). Integer
       leaves are exempt and are not cast.
    3. **The output took.** Every floating-point output leaf is ``float64``. This
       is the one that catches a hard-coded ``dtype="float32"`` constant or a
       cast island, and it is the only assertion here that can fail on a model
       rather than on the harness.

    :param build: Zero-argument model factory, invoked INSIDE the float64 scope.
    :param make_inputs: Zero-argument input factory; its float leaves are cast.
    :param call_fn: Replacement for :func:`default_call`; same contract.
    :param training: Forwarded to the model.
    :param dtype_exempt_outputs: Indices into the flattened output that
        assertion 3 skips. Same escape hatch as the precision arm's, and it
        carries the same requirement: a stated reason at the call site.
    :param expected_output_dtype: What assertion 3 requires of every float
        output. Leave it at ``"float64"`` unless the model DOCUMENTS a pinned
        dtype, in which case name that dtype here so the assertion keeps
        running against the documented value instead of being switched off --
        the D-058 shape, applied to this arm. ``ideogram4`` is the one such
        subject.
    :return: A report with ``floatx_seen``, ``policy_seen``, ``model_policy``,
        ``input_dtypes``, ``dtypes``, ``n_nan``, ``n_inf`` and ``absmax``.
    :rtype: PrecisionArmReport
    :raises AssertionError: on any failed part.
    """
    with float64_scope():
        floatx_seen = keras.backend.floatx()
        policy_seen = keras.mixed_precision.global_policy().name
        keras.utils.set_random_seed(0)
        model = build()
        inputs = _cast_float_leaves(make_inputs(), "float64")
        input_dtypes = _float_leaf_dtypes(inputs)
        call = call_fn or default_call
        tensors = flatten_tensors(call(model, inputs, training))
        arrays = [np.asarray(ops.convert_to_numpy(t)) for t in tensors]
        report = PrecisionArmReport(
            floatx_seen=floatx_seen,
            policy_seen=policy_seen,
            model_policy=getattr(model.dtype_policy, "name", None),
            input_dtypes=input_dtypes,
            dtypes=[str(t.dtype).replace("<dtype: '", "").strip("'>")
                    for t in tensors],
            n_nan=[int(np.isnan(a).sum()) for a in arrays
                   if a.dtype.kind == "f"],
            n_inf=[int(np.isinf(a).sum()) for a in arrays
                   if a.dtype.kind == "f"],
            absmax=[float(np.abs(a).max()) for a in arrays
                    if a.dtype.kind == "f" and a.size],
        )

    # 1 -- the scope took. Asserted OUTSIDE the context so a failure cannot
    #      leave the process at float64 for every test that follows.
    assert floatx_seen == "float64", (
        f"keras.backend.floatx() was {floatx_seen!r} inside the arm")
    assert policy_seen == "float64", (
        f"the global dtype policy was {policy_seen!r} inside the arm -- "
        "set_floatx alone does not move it once global_policy() has been "
        "read, which is the whole D-074 finding")
    assert report["model_policy"] == "float64", (
        f"the model carries dtype_policy {report['model_policy']!r}; it was "
        "built outside the float64 scope, so this arm is a float32 test "
        "wearing a float64 name")

    # 2 -- the input took.
    assert input_dtypes, (
        "the model was given no floating-point input, so the input-dtype "
        "assertion would be vacuous")
    assert all(d == "float64" for d in input_dtypes), (
        f"the float inputs are {input_dtypes}, expected every one float64")

    # 3 -- the output took.
    float_out = [d for i, d in enumerate(report["dtypes"])
                 if i not in set(dtype_exempt_outputs)
                 and (d.startswith("float") or d.startswith("bfloat"))]
    assert float_out, (
        f"the model returned no floating-point tensor ({report['dtypes']}), "
        "so the output-dtype assertion would be vacuous")
    assert all(d == expected_output_dtype for d in float_out), (
        f"under float64 the float outputs are {float_out} (all outputs "
        f"{report['dtypes']}), expected every one {expected_output_dtype!r}; "
        "an unexpected float32 output means the model pins a dtype somewhere "
        "and the caller's requested precision never reached it")

    assert sum(report["n_nan"]) == 0 and sum(report["n_inf"]) == 0, (
        f"float64 forward is non-finite: nan={report['n_nan']} "
        f"inf={report['n_inf']}")
    return report


# ===========================================================================
# R-143 -- the graph / XLA equivalence arm
# ===========================================================================


def _max_delta(a: Any, b: Any) -> float:
    """``inf`` for an exact-integer mismatch, else the max absolute float delta.

    R-143 asks for ``np.array_equal`` on exact-integer outputs and a tolerance
    on floats. Returning ``inf`` for an integer mismatch lets one comparison
    loop carry both rules without a second code path.
    """
    if a.dtype.kind != "f":
        return 0.0 if np.array_equal(a, b) else float("inf")
    if a.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(a.astype("float64") - b.astype("float64"))))


def assert_xla_equivalence(
        build: Callable[[], Any],
        make_inputs: Callable[[], Any],
        *,
        call_fn: Optional[Callable[[Any, Any, bool], Any]] = None,
        training: bool = False,
        rtol: float = 1e-2,
        seed: Optional[int] = 0,
        expect: str = "compiles",
        expect_reason: Optional[str] = None,
        scope: Optional[str] = None,
) -> PrecisionArmReport:
    """Assert that ``tf.function(jit_compile=True)`` computes what eager does.

    R-143: "an eager-only fix is not a fix". Every defect this plan fixed was
    measured eagerly, and ``fit()`` on a GPU defaults to ``jit_compile='auto'``,
    which is not eager. This arm is the one that notices the difference.

    DECISION plan-2026-08-19T163559-499b6f0e/D-075
    **The eager-vs-eager CONTROL is what makes it usable.** The model is called
    twice eagerly, on the same weights and the same inputs, BEFORE the traced
    call. For a deterministic forward that spread is exactly ``0.0`` and the
    tolerance below is the flat ``rtol`` it looks like. For a STOCHASTIC one it
    is not, and without this control three healthy models convict:

    ==============  ==============  ==============  =========
    package         eager spread    eager-vs-XLA    verdict
    ==============  ==============  ==============  =========
    ``vae``         1.034906e+00    1.029735e+00    the sampler, not XLA
    ``sd3_mmdit``   3.591807e+00    3.483226e+00    the sampler, not XLA
    ``relgt``       9.723109e-02    8.883548e-02    the sampler, not XLA
    ==============  ==============  ==============  =========

    Each of those would have been an "XLA breaks this model" finding under a
    flat tolerance, and each is a model whose forward is documented stochastic
    at ``training=False``. This is the same self-calibration D-055 argues for
    the fp16 arm, applied to a different axis. Do NOT replace the
    ``max(rtol * absmax, 3 * spread)`` bound with a flat ``rtol``, and do NOT
    "fix" a large delta by raising ``rtol`` past the signal -- the two subjects
    that needed it are judged at float64 instead, and the reason is recorded at
    their entries in ``precision_arm_subjects.XLA_OVERRIDES``.
    See decisions.md D-075.

    **``expect='raises'`` is not the same claim as a missing arm.** Some models
    genuinely cannot be XLA-compiled and the repo already ships
    ``jit_compile=False`` for them. Naming that here, with the exception type
    the compiler actually raises, keeps a documented opt-out from silently
    becoming an unnoticed regression in either direction: if the model starts
    compiling, this test fails and the ``jit_compile=False`` becomes removable
    rather than cargo.

    :param build: Zero-argument model factory. Called ONCE; both arms share the
        weights, because two builds would measure the build spread instead
        (D-055 measured a 43% gap from that alone).
    :param make_inputs: Zero-argument input factory. Called once.
    :param call_fn: Replacement for :func:`default_call`; same contract.
    :param training: Forwarded to the model in every arm. Pass ``True`` for a
        model whose untrained BatchNorms make the inference path meaningless --
        the same D-065 reason the precision arm has ``forward_training``.
    :param rtol: Relative tolerance against each output's own ``absmax``. The
        effective bound is ``max(rtol * absmax, 3 * eager_spread)``.
    :param seed: Seeded immediately before ``build()``.
    :param expect: ``"compiles"`` or ``"raises"``.
    :param expect_reason: Required when ``expect="raises"`` -- a substring that
        must appear in the exception text, so "it raised" cannot be satisfied
        by a DIFFERENT failure than the documented one.
    :return: A report with ``absmax``, ``eager_spread``, ``xla_delta``,
        ``dtypes`` and, for ``expect="raises"``, ``exception``.
    :rtype: PrecisionArmReport
    :raises AssertionError: on any failed part.
    """
    import tensorflow as tf

    # ``scope='float64'`` runs BOTH arms at double precision. It is not a way to
    # relax the assertion -- the tolerance below is applied the same way, and a
    # model that computes a genuinely different function under XLA still fails.
    # It is the answer to "is this disagreement XLA or is it the model's own
    # conditioning": if the delta collapses by ten orders of magnitude when the
    # only thing that changed is the mantissa, it was never XLA. See D-075.
    outer = float64_scope() if scope == "float64" else contextlib.nullcontext()
    with outer:
        if seed is not None:
            keras.utils.set_random_seed(seed)
        model = build()
        inputs = make_inputs()
        if scope == "float64":
            inputs = _cast_float_leaves(inputs, "float64")
        call = call_fn or default_call
        return _xla_body(tf, model, inputs, call, training, rtol, expect,
                         expect_reason)


def _xla_body(tf, model, inputs, call, training, rtol, expect, expect_reason):
    """The body of :func:`assert_xla_equivalence`, inside its dtype scope.

    Split out only so the ``float64`` scope can wrap build, call and judge
    together; every argument is exactly the resolved form of the public one.
    """

    def _arrays(out: Any) -> List[Any]:
        return [np.asarray(ops.convert_to_numpy(t)) for t in flatten_tensors(out)]

    eager_1 = _arrays(call(model, inputs, training))
    eager_2 = _arrays(call(model, inputs, training))

    @tf.function(jit_compile=True)
    def _jitted(x: Any) -> Any:
        return call(model, x, training)

    if expect == "raises":
        assert expect_reason, (
            "expect='raises' without an expect_reason would pass on ANY "
            "failure, including one introduced by the test itself")
        try:
            _jitted(inputs)
        except Exception as exc:  # noqa: BLE001 -- the type is the finding
            text = " ".join(str(exc).split())
            assert expect_reason in text, (
                f"XLA raised, but not for the documented reason "
                f"{expect_reason!r}: {type(exc).__name__}: {text[:400]}")
            return PrecisionArmReport(
                exception=type(exc).__name__, message=text[:400],
                absmax=[float(np.abs(a).max()) for a in eager_1
                        if a.dtype.kind == "f" and a.size])
        raise AssertionError(
            "this model is documented as NOT XLA-compilable and it just "
            "compiled. Either the blocker was fixed -- in which case the "
            f"shipped jit_compile=False and this arm are both stale -- or "
            f"{expect_reason!r} no longer describes the blocker.")

    assert expect == "compiles", f"unknown expect={expect!r}"
    xla = _arrays(_jitted(inputs))

    assert eager_1, "the model returned no tensors to judge"
    assert len(xla) == len(eager_1), (
        f"the traced arm returned {len(xla)} tensors, eager returned "
        f"{len(eager_1)}")

    spread = [_max_delta(a, b) for a, b in zip(eager_1, eager_2)]
    delta = [_max_delta(a, b) for a, b in zip(eager_1, xla)]
    absmax = [float(np.abs(a).max()) if a.dtype.kind == "f" and a.size else 0.0
              for a in eager_1]
    report = PrecisionArmReport(
        dtypes=[str(a.dtype) for a in eager_1],
        absmax=absmax, eager_spread=spread, xla_delta=delta)

    assert any(m > 0.0 for m in absmax) or any(
        a.dtype.kind != "f" for a in eager_1), (
        "every output is an all-zero float tensor, so an agreement between "
        "the two arms proves nothing")

    for i, (d, s, m) in enumerate(zip(delta, spread, absmax)):
        if eager_1[i].dtype.kind != "f":
            assert d == 0.0, (
                f"output {i} is {eager_1[i].dtype} and the traced arm returned "
                "DIFFERENT values; an exact-integer output must be "
                "bit-identical under XLA")
            continue
        tol = max(rtol * max(m, 1e-8), 3.0 * s)
        assert d <= tol, (
            f"output {i}: eager-vs-XLA max|delta| {d:.6e} exceeds "
            f"tol={tol:.6e} (rtol={rtol} against absmax {m:.6e}; the model's "
            f"own eager-vs-eager spread is {s:.6e}). A delta far above the "
            "eager spread is XLA computing a different function, not "
            "precision.")
    return report
