"""
Gradient-flow oracle -- proves every trainable weight is on the backward graph
=============================================================================

This module is an *instrument*, not a test suite. It is deliberately named
without a ``test_`` prefix so pytest does not collect it, mirroring
``tests/test_models/knob_sensitivity_oracle.py``,
``tests/test_models/smoke_contract_oracle.py`` and
``tests/test_models/test_sam/dead_component_oracle.py``. Its own RED proofs live
in ``tests/test_models/test_gradient_flow_oracle.py``.

Why it exists
-------------
Measured 2026-08-19 over ``tests/test_models/``: 28 package directories plus the
loose ``test_lewm.py`` -- **29 suites** -- contain no ``GradientTape`` and no
``.gradient(`` at all, so for those models "every weight trains" is not a claim
the suite makes. That is 2.6x the carried review's "11" hypothesis.

The suites that DO probe gradients disagree about what the claim is. The two
shapes seen in the tree:

``assert all(g is not None for g in gradients)``
    Satisfied by a model in which nothing trains. ``None`` is returned only for
    a weight that is not on the tape's graph AT ALL; a weight whose gradient is
    a tensor of exact zeros -- the dead-component signature -- passes.
    ``test_power_mlp/test_model.py::test_gradients_flow`` had exactly this body,
    plus a ``grad.shape == var.shape`` loop (which is a property of
    ``tape.gradient`` itself, not of the model).

``assert all(norm >= 0.0 for norm in grad_norms)``
    A Euclidean norm is never negative. ``test_bert`` carried this until
    2026-08-18 and it reported green on a draw where 61 of 61 weights had
    identically-zero gradients.

Two traps this oracle removes from the call site
------------------------------------------------
1. **The from-logits trap.** A ``categorical_crossentropy(..., from_logits=False)``
   loss applied to logits renormalizes and CLIPS to ``[eps, 1-eps]``; for some
   input draws every element lands in the clipped region, where the loss is
   constant and EVERY gradient is exactly 0.0 (measured at ``test_bert``:
   61/61 weights, and the result was pytest-collection-order dependent because
   the draw came from the process-global RNG). :func:`default_loss` is a plain
   mean-of-squares over the output tensors and has no clipped region, so an
   adopting suite that does not need a labelled loss cannot step in this hole.
   Suites that want their real loss pass ``loss_fn=``; if that loss zeroes
   everything, this oracle says so by name instead of reporting green.
2. **The aggregate trap.** A single global-norm assertion hides one dead tensor
   among forty live ones. Every assertion here is PER WEIGHT, keyed by
   ``Variable.path``, and the failure message names the weights.

What "reaches" means
--------------------
Per weight, all three of:

* the gradient is not ``None``       (the disconnected-subgraph case);
* ``np.isfinite(g).all()``           (the NaN/Inf case);
* ``float(np.max(np.abs(g))) > 0.0`` (the dead-component case).

The floor is **exactly zero, not an epsilon**. A correctly initialized weight
may carry a legitimately tiny gradient: ``test_mamba_v1.py`` records
``max|grad_A_log| == 3.17e-08`` for a HEALTHY Mamba block (the discretization
``A_bar = exp(dt * A)`` scales that gradient with ``dt in [1e-3, 1e-1]``), and a
tree_transformer bias measured 1.357e-09. Any absolute floor above 0.0 would
convict those. "Identically zero" is the falsifiable claim, and it is the one
that catches a sublayer that is built, saved, trained around and never executed.

``expect_zero`` -- and why it is an EXPECTATION, not an allowance
-----------------------------------------------------------------
Some weights legitimately receive no gradient. Step 2 of this plan measured one
directly: ``read_controller.W_g.kernel`` in the memory-bank model has an exactly
0.0 gradient during phase 1 BY DESIGN, because the phase scheduler has not yet
switched that path on. A naive oracle calls that a bug; an oracle with no way to
say otherwise gets weakened at adoption time, which is how an instrument rots.

So ``expect_zero`` is a two-sided claim, not a skip list:

* a weight matched by an ``expect_zero`` pattern is EXEMPT from the nonzero
  floor (it may be ``None`` or all-zero -- from the optimizer's point of view
  those are the same statement, "this weight does not learn here");
* but it must ACTUALLY be zero. A waived weight that turns out to have a live
  gradient is an error, because the waiver has become a lie about the model;
* and every pattern must match at least one weight. A pattern that matches
  nothing is a stale waiver from a renamed variable, and stale waivers are how
  a growing allowlist quietly disables an oracle.

``expect_zero`` entries are matched as SUBSTRINGS of ``Variable.path`` so a call
site can waive a subtree (``"read_controller/"``) or one tensor
(``"read_controller/W_g/kernel"``).

There is deliberately **no** ``allow_near_zero``/``atol`` parameter. It was in
the plan's sketch for the mamba ~1e-9 case, but that case needs no waiver at all
once the floor is exact zero (3.17e-08 > 0.0), and a tolerance knob is the one
control that could be widened until the oracle passes anything.

How to use it
-------------
::

    from ..gradient_flow_oracle import assert_gradients_reach_every_trainable_weight

    def test_gradients_reach_every_trainable_weight(self):
        model = create_thing(...)
        x = np.random.default_rng(0).random((2, 32)).astype("float32")
        assert_gradients_reach_every_trainable_weight(model, x)

With a real labelled loss and one documented frozen subtree::

    assert_gradients_reach_every_trainable_weight(
        model,
        inputs,
        loss_fn=lambda out: keras.ops.mean(
            keras.losses.categorical_crossentropy(y, out["logits"], from_logits=True)
        ),
        expect_zero=("read_controller/W_g/kernel",),   # phase 1, by design
    )

A suite where the oracle reports a genuinely dead weight has found a PRODUCT
finding. Log it and fix it -- do not move it into ``expect_zero`` to make the
suite green. ``expect_zero`` is for behaviour the model documents on purpose.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import keras
import numpy as np
import tensorflow as tf

__all__ = [
    "default_loss",
    "gradient_report",
    "assert_gradients_reach_every_trainable_weight",
]

LossFn = Callable[[Any], Any]


def _iter_tensors(outputs: Any) -> Iterable[Any]:
    """Yield every tensor in a possibly-nested model output."""
    if isinstance(outputs, dict):
        for value in outputs.values():
            yield from _iter_tensors(value)
    elif isinstance(outputs, (list, tuple)):
        for value in outputs:
            yield from _iter_tensors(value)
    elif outputs is not None:
        yield outputs


def default_loss(outputs: Any) -> Any:
    """Mean of squares over every floating-point tensor in ``outputs``.

    Chosen over a labelled classification loss on purpose: it is defined for
    every output shape, needs no targets (so a 29-suite adoption costs one line
    per suite), and -- unlike ``categorical_crossentropy(from_logits=False)`` --
    has no clipped region in which every gradient is identically zero.

    Integer outputs (token ids, argmax indices, masks) are skipped: they are not
    differentiable and including them would raise rather than assert.
    """
    terms = []
    for tensor in _iter_tensors(outputs):
        raw = getattr(tensor, "dtype", None)
        if raw is None:
            continue
        # A ``tf.DType`` stringifies as "<dtype: 'float32'>", so read ``.name``
        # when it is there and fall back to ``str`` for a plain-string dtype.
        dtype = getattr(raw, "name", None) or str(raw)
        if not dtype.startswith("float"):
            continue
        terms.append(keras.ops.mean(keras.ops.square(keras.ops.cast(tensor, "float32"))))
    if not terms:
        raise ValueError(
            "default_loss found no floating-point tensor in the model output; "
            "pass an explicit loss_fn=... that selects a differentiable head"
        )
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def gradient_report(
    model: Any,
    inputs: Any,
    *,
    loss_fn: Optional[LossFn] = None,
    training: bool = True,
) -> Dict[str, Optional[float]]:
    """One tape step; return ``{weight.path: max|grad|}``. Asserts nothing.

    ``None`` means the weight received no gradient at all (it is not on the
    backward graph); ``nan`` means the gradient contained a non-finite value.

    Use this to obtain the measured number that goes into a decisions entry or
    an ``expect_zero`` justification.
    """
    weights = list(model.trainable_weights)
    if not weights:
        raise ValueError(
            f"{type(model).__name__} exposes no trainable weights; a "
            "gradient-flow assertion over an empty set is vacuous. Build the "
            "model (a subclassed keras.Model is unbuilt until its first call) "
            "or assert on the sub-model that owns the weights."
        )

    with tf.GradientTape() as tape:
        outputs = model(inputs, training=training)
        loss = default_loss(outputs) if loss_fn is None else loss_fn(outputs)

    gradients = tape.gradient(loss, weights)

    report: Dict[str, Optional[float]] = {}
    for weight, grad in zip(weights, gradients):
        if grad is None:
            report[weight.path] = None
            continue
        if isinstance(grad, tf.IndexedSlices):
            grad = tf.convert_to_tensor(grad)
        array = np.asarray(keras.ops.convert_to_numpy(grad))
        if tuple(array.shape) != tuple(weight.shape):
            raise AssertionError(
                f"gradient at {weight.path} has shape {tuple(array.shape)} but "
                f"the weight has shape {tuple(weight.shape)}"
            )
        report[weight.path] = (
            float("nan") if not np.isfinite(array).all()
            else float(np.max(np.abs(array)))
        )
    return report


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(pattern in path for pattern in patterns)


# DECISION plan-2026-08-19T070627-a616f581/D-010
# The nonzero floor is EXACTLY 0.0 and `expect_zero` is a two-sided claim.
# Do NOT add an `atol`/`allow_near_zero` parameter, however reasonable the
# request looks at a call site: a correctly-initialized Mamba `A_log` carries
# max|grad| = 3.17e-08 and a live tree_transformer bias 1.357e-09, so any
# absolute floor big enough to feel meaningful convicts healthy models, and a
# tolerance knob is the single control that can be widened until this oracle
# passes anything. `expect_zero` is the escape hatch instead -- and it must stay
# two-sided (a waived weight MUST be zero, and every pattern MUST match
# something), because a one-sided skip list is how an oracle rots into silence:
# each adoption that hits a dead weight would append one line and move on.
# See D-010 in plans/plan-2026-08-19T070627-a616f581/decisions.md.
def assert_gradients_reach_every_trainable_weight(
    model: Any,
    inputs: Any,
    *,
    loss_fn: Optional[LossFn] = None,
    expect_zero: Sequence[str] = (),
    training: bool = True,
) -> Dict[str, Optional[float]]:
    """Assert every trainable weight receives a finite, not-identically-zero gradient.

    :param model: a BUILT model (a subclassed ``keras.Model`` is unbuilt until
        its first ``call()``, and this raises rather than passing vacuously).
    :param inputs: whatever ``model(...)`` accepts -- array, list, or dict.
    :param loss_fn: ``outputs -> scalar``. Defaults to :func:`default_loss`.
    :param expect_zero: substrings of ``Variable.path`` naming weights that are
        documented NOT to learn under this input/config. Each pattern must match
        at least one weight, and every matched weight must actually have a
        ``None`` or identically-zero gradient.
    :param training: forwarded to ``model(...)``.
    :returns: the ``{path: max|grad|}`` report, so a caller can make a stronger
        claim on top (e.g. a specific weight's gradient magnitude).
    :raises AssertionError: naming every offending weight path.
    """
    expect_zero = tuple(expect_zero)
    report = gradient_report(model, inputs, loss_fn=loss_fn, training=training)

    dead: List[str] = []
    non_finite: List[str] = []
    disconnected: List[str] = []
    live_but_waived: List[str] = []

    for path, value in report.items():
        waived = _matches(path, expect_zero)
        if value is None:
            if not waived:
                disconnected.append(path)
        elif np.isnan(value):
            non_finite.append(path)
        elif value == 0.0:
            if not waived:
                dead.append(path)
        elif waived:
            live_but_waived.append(f"{path} (max|grad|={value:.3e})")

    unmatched = [p for p in expect_zero if not any(p in path for path in report)]

    problems: List[str] = []
    if disconnected:
        problems.append(
            f"{len(disconnected)} weight(s) received NO gradient (not on the "
            f"backward graph -- built, saved, and never executed):\n  "
            + "\n  ".join(disconnected)
        )
    if non_finite:
        problems.append(
            f"{len(non_finite)} weight(s) received a non-finite (NaN/Inf) "
            f"gradient:\n  " + "\n  ".join(non_finite)
        )
    if dead:
        problems.append(
            f"{len(dead)} of {len(report)} weight(s) received an identically-"
            f"zero gradient:\n  " + "\n  ".join(dead)
        )
    if live_but_waived:
        problems.append(
            f"{len(live_but_waived)} weight(s) listed in expect_zero DO receive "
            f"a gradient -- the waiver is obsolete and must be removed:\n  "
            + "\n  ".join(live_but_waived)
        )
    if unmatched:
        problems.append(
            f"{len(unmatched)} expect_zero pattern(s) matched no weight -- a "
            f"stale waiver (renamed variable?) silently widens this oracle:\n  "
            + "\n  ".join(unmatched)
        )

    if problems:
        live = sum(
            1 for v in report.values() if v is not None and not np.isnan(v) and v > 0.0
        )
        raise AssertionError(
            f"gradient flow is incomplete in {type(model).__name__}: "
            f"{live}/{len(report)} trainable weights receive a live gradient.\n"
            + "\n".join(problems)
        )
    return report
