"""
R-143: every charged ``models/`` package, under ``tf.function(jit_compile=True)``.

R-143 -- "graph/XLA equivalence: an eager-only fix is not a fix" -- was charged
against ~50 of the 73 ``models/`` test directories, every one of them for the
same reason: ``jit_compile`` count 0 in the directory. It matters here more than
the usual coverage row, because ``Model.fit()`` on Keras 3.8 defaults to
``jit_compile='auto'``, which selects XLA on a GPU, while every defect this plan
found and fixed was measured EAGERLY.

What the sweep found
--------------------
Over the 53 registered subjects, run once each:

* **51 compile and agree with eager.** The largest relative disagreement is
  ``lewm``'s 3.6e-3; the arm's default ``rtol=1e-2`` clears the whole registry
  with ~2.7x margin.
* **1 is documented NOT to compile** -- ``superpoint``, whose bicubic upsample
  has no ``ResizeBicubic`` XLA_GPU_JIT kernel, and which already ships
  ``jit_compile=False`` in its trainer. That row is asserted as a RAISE keyed on
  the op name, so a documented opt-out cannot quietly become stale.
* **1 cannot be traced at all** -- ``SAM``, which fails identically at
  ``jit_compile=False`` and ``True``. That is a graph-mode contract the package
  already documents, and its ``fit()`` path (``SAMTrainingModel``) IS
  XLA-clean, so the subject is the wrapper.

Three readings the eager-vs-eager control REVERSED
--------------------------------------------------
``vae``, ``sd3_mmdit`` and ``relgt`` have stochastic forwards at
``training=False``. Their eager-vs-XLA deltas (1.03e+00, 3.48e+00, 8.88e-02)
are each INSIDE their own eager-vs-eager spread (1.03e+00, 3.59e+00, 9.72e-02).
Under a flat tolerance all three would have been "XLA breaks this model". The
control is argued at :func:`assert_xla_equivalence`.

Two more the arm found that are NOT about XLA
---------------------------------------------
``gemma`` and ``qwen`` read a divergence LARGER than their own output
(3.24e-01 against absmax 3.20e-01) because the subject table fed token ids in
``[0, 256)`` to a 64-row embedding, and eager-GPU and XLA resolve an
out-of-bounds gather differently. That is an instrument defect, fixed in
``precision_arm_subjects.py`` (D-076). ``yolo12``'s 2.59e-01 relative delta is
its own conditioning at initialization, proved by a float64 control that
measures 1.50e-12 on the same graph (D-075).
"""

import pytest

from .precision_arm_oracle import assert_xla_equivalence
from .precision_arm_subjects import (
    CHARGED_PACKAGES, SUBJECTS, XLA_OVERRIDES, subject_names, xla_subject,
)


def test_every_subject_resolves_for_the_xla_arm():
    """No silent dropouts, and no override for a name that does not exist.

    The second half matters more than it looks: an override keyed on a typo
    would be silently ignored, and the subject would run the DEFAULT arm while
    its comment claimed otherwise -- which for ``superpoint`` would mean
    asserting that a model which cannot compile does compile.
    """
    assert set(SUBJECTS) >= set(CHARGED_PACKAGES)
    unknown = sorted(set(XLA_OVERRIDES) - set(SUBJECTS))
    assert not unknown, f"XLA_OVERRIDES keys with no subject: {unknown}"
    for name in subject_names():
        build, make_inputs, kwargs = xla_subject(name)
        assert callable(build) and callable(make_inputs)
        assert set(kwargs) <= {
            "call_fn", "training", "rtol", "seed", "expect", "expect_reason",
            "scope",
        }, f"{name}: unknown kwargs {sorted(kwargs)}"


@pytest.mark.parametrize("name", subject_names())
def test_the_package_computes_the_same_function_under_xla(name):
    """One package, eager against ``tf.function(jit_compile=True)``."""
    build, make_inputs, kwargs = xla_subject(name)
    report = assert_xla_equivalence(build=build, make_inputs=make_inputs,
                                    **kwargs)
    if kwargs.get("expect") == "raises":
        assert report["exception"], "the raise arm reported no exception"
    else:
        assert report["xla_delta"], "no output was compared"


def test_exactly_one_subject_is_documented_as_not_compiling():
    """The census, pinned as a NUMBER.

    "How many of these models actually run under XLA" is the question this
    family exists to answer, and an answer that drifts silently is no answer.
    51 compile, 1 (``superpoint``) is documented not to, and ``SAM`` is
    represented by its traceable ``fit()`` wrapper. If a second package
    acquires an ``expect='raises'`` entry, that is a regression someone must
    justify here, not a table edit.
    """
    raising = sorted(n for n in subject_names()
                     if xla_subject(n)[2].get("expect") == "raises")
    assert raising == ["superpoint"], (
        f"the not-XLA-compilable set changed: {raising}")
    assert len(subject_names()) - len(raising) == 52, (
        "the compiling-subject count moved; re-measure before editing this")


def test_the_hrm_xla_delta_is_conditioning_and_the_probe_says_so():
    """The refutation behind ``hierarchical_reasoning_model``'s float64 scope.

    Its float32 eager-vs-XLA delta (7.26e+00) is LARGER than its own output
    (absmax 5.54e+00) with an eager spread of exactly 0.0, which reads as "XLA
    computes a different function" and is the single most alarming number this
    family produced. It is not that. A **one ULP** change to one element of one
    embedding moves the output by ~1e-1, an amplification of order 1e7, and
    restoring that element returns the output to EXACTLY the original bits.
    XLA reassociates at the ULP level across the whole graph, so a chaotic
    recursive model turns that into a whole-output change.

    Both halves are asserted. The restore arm is what makes the perturbation
    arm mean anything: without it, "the output moved" is equally explained by
    a non-deterministic forward.
    """
    import numpy as np
    import keras
    from .precision_arm_oracle import default_call, flatten_tensors

    build, make_inputs, _ = xla_subject("hierarchical_reasoning_model")
    keras.utils.set_random_seed(0)
    model = build()
    inputs = make_inputs()

    def _logits():
        out = flatten_tensors(default_call(model, inputs, False))[0]
        return np.asarray(keras.ops.convert_to_numpy(out))

    base = _logits()
    var = next(w for w in model.weights
               if w.trainable and np.asarray(w).size > 4)
    original = np.asarray(keras.ops.convert_to_numpy(var)).copy()

    perturbed = original.copy()
    flat = perturbed.reshape(-1)
    step = float(abs(np.nextafter(flat[0], np.float32(np.inf)) - flat[0]))
    flat[0] = np.nextafter(flat[0], np.float32(np.inf))
    var.assign(perturbed)
    moved = float(np.max(np.abs(_logits().astype("float64")
                                - base.astype("float64"))))
    var.assign(original)
    restored = float(np.max(np.abs(_logits().astype("float64")
                                   - base.astype("float64"))))

    assert restored == 0.0, (
        f"restoring the weight left a delta of {restored:.6e}; this forward is "
        "not deterministic, so the perturbation number below proves nothing")
    assert step > 0.0
    assert moved / step > 1e5, (
        f"a {step:.6e} perturbation moved the output by {moved:.6e}, an "
        f"amplification of {moved / step:.3e}. Below 1e5 this model is no "
        "longer chaotic and its float64 scope should be re-derived, because "
        "the float32 arm would then be usable.")
