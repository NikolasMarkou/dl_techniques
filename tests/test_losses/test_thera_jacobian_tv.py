"""Scoped tests for the THERA exact Jacobian-TV regularizer (step 9, SC-9).

This is the **STOP-IF #1 oracle** of the plan's Pre-Mortem: the exact analytic
spatial Jacobian ``d(field)/d(rel_coords)`` (at ``t=0``) must

  1. be produced by ``Thera(..., return_jac=True)`` with the right shape, finite,
     non-zero, non-``None``;
  2. reduce to a finite positive TV penalty (``thera_tv_penalty``);
  3. **differentiate through to the model weights via an OUTER tape** (the nested
     tape composes: inner Jacobian -> TV loss -> outer weight-grads), proving the
     step-11 trainer's nested-tape ``train_step`` is viable;
  4. run on a config where source resolution != coord resolution (16 -> 20).

If 3 fails (None / non-finite weight-grads), that is Pre-Mortem STOP-IF #1 and
must be surfaced -- NOT silently swapped for finite-difference (Q3).
"""

import numpy as np
import tensorflow as tf
import pytest

from dl_techniques.layers.grid_sample import make_grid
from dl_techniques.models.thera import (
    Thera,
    EDSRBackbone,
    build_thera_tail,
)
from dl_techniques.losses.thera_jacobian_tv import (
    thera_tv_penalty,
    thera_total_loss,
)


# ---------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------


def _finite(t) -> bool:
    return bool(np.all(np.isfinite(np.asarray(t))))


def _coords(batch: int, hq: int, wq: int):
    grid = make_grid((hq, wq))  # (hq, wq, 2) numpy, pixel-center [h, w]
    g = tf.convert_to_tensor(grid, dtype=tf.float32)
    return tf.broadcast_to(g[None, ...], (batch, hq, wq, 2))


@pytest.fixture
def small_model() -> Thera:
    """A tiny but real Thera (edsr backbone, air tail) for the Jacobian oracle."""
    return Thera(
        hidden_dim=16,
        out_dim=3,
        backbone=EDSRBackbone(num_feats=32, num_blocks=2),
        tail=build_thera_tail("air"),
    )


@pytest.fixture
def inputs():
    # source 16x16, query grid 20x20 -> source != coord resolution (test 5).
    source = tf.random.normal((2, 16, 16, 3))
    coords = _coords(2, 20, 20)
    t = tf.ones((2, 1))
    return source, coords, t


# ---------------------------------------------------------------------
# 1-2. return_jac shape / finiteness / non-zero
# ---------------------------------------------------------------------


def test_return_jac_shapes_finite_nonzero(small_model, inputs):
    out, jac = small_model(inputs, return_jac=True)

    assert tuple(out.shape) == (2, 20, 20, 3)
    assert tuple(jac.shape) == (2, 20, 20, 3, 2)

    assert out is not None and jac is not None
    assert _finite(out) and _finite(jac)

    # The Jacobian must be NON-ZERO (a zero Jacobian = broken gradient path E5).
    assert float(tf.reduce_max(tf.abs(jac))) > 0.0


# ---------------------------------------------------------------------
# 3. TV penalty is a finite positive scalar
# ---------------------------------------------------------------------


def test_tv_penalty_finite_positive(small_model, inputs):
    _, jac = small_model(inputs, return_jac=True)
    tv = thera_tv_penalty(jac)

    assert tuple(tv.shape) == ()  # scalar
    assert _finite(tv)
    assert float(tv) > 0.0


def test_total_loss_combines(small_model, inputs):
    _, jac = small_model(inputs, return_jac=True)
    recon = tf.constant(0.5, dtype=tf.float32)
    total = thera_total_loss(recon, jac, tv_weight=0.1)
    expected = 0.5 + 0.1 * float(thera_tv_penalty(jac))
    assert _finite(total)
    assert abs(float(total) - expected) < 1e-5


# ---------------------------------------------------------------------
# A7 (review): KNOWN-ANSWER oracle -- thera_tv_penalty == mean(abs(x)) computed
# by an INDEPENDENT numpy path (NOT a recompute via thera_tv_penalty). This is
# the non-tautological replacement for the `expected = thera_tv_penalty(jac)`
# pattern above: the RHS here never calls the function under test.
# ---------------------------------------------------------------------


def test_tv_penalty_known_answer():
    import keras

    # Hand-known 2x2: mean(|[-1, 2, 3, -4]|) = (1 + 2 + 3 + 4) / 4 = 2.5.
    x_np = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    x = keras.ops.convert_to_tensor(x_np)
    oracle = float(np.mean(np.abs(x_np)))
    assert oracle == 2.5  # literal sanity on the hand computation itself
    assert np.isclose(float(thera_tv_penalty(x)), oracle, atol=1e-6)

    # A second tensor of a DIFFERENT (5-D, Jacobian-like) shape against the same
    # independent numpy oracle -- guards against any shape-dependent reduction bug.
    rng = np.random.default_rng(7)
    y_np = rng.standard_normal((2, 3, 4, 5, 2)).astype(np.float32)
    y = keras.ops.convert_to_tensor(y_np)
    assert np.isclose(
        float(thera_tv_penalty(y)), float(np.mean(np.abs(y_np))), atol=1e-6
    )


# ---------------------------------------------------------------------
# 4. STOP-IF #1 ORACLE: nested-tape weight gradients
# ---------------------------------------------------------------------


def test_nested_tape_weight_grads(small_model, inputs):
    """Outer tape differentiates the TV (built from the inner Jacobian) to weights.

    Asserts non-None + finite grads to the heat-field ``components`` AND the
    hypernetwork ``out_conv`` kernel -- the two weights the TV loss must reach.
    ``heat_field.k`` may legitimately get a None/zero grad (the t=0 envelope is
    k-independent), so we do NOT assert on it.
    """
    with tf.GradientTape() as outer:
        _, jac = small_model(inputs, return_jac=True)
        tv = thera_tv_penalty(jac)

    grads = outer.gradient(tv, small_model.trainable_variables)
    grad_by_name = {
        v.path: g for v, g in zip(small_model.trainable_variables, grads)
    }

    # Locate the two target weights by name substring.
    comp_items = [
        (p, g) for p, g in grad_by_name.items() if "components" in p
    ]
    outconv_items = [
        (p, g)
        for p, g in grad_by_name.items()
        if "out_conv" in p and "kernel" in p
    ]

    assert comp_items, "heat_field.components weight not found in trainable vars"
    assert outconv_items, "hypernetwork out_conv kernel not found in trainable vars"

    for p, g in comp_items:
        assert g is not None, f"TV grad to {p} is None (STOP-IF #1: nested tape broke)"
        assert _finite(g), f"TV grad to {p} is non-finite"
        assert float(tf.reduce_max(tf.abs(g))) > 0.0, f"TV grad to {p} is all-zero"

    for p, g in outconv_items:
        assert g is not None, f"TV grad to {p} is None (STOP-IF #1: nested tape broke)"
        assert _finite(g), f"TV grad to {p} is non-finite"
        assert float(tf.reduce_max(tf.abs(g))) > 0.0, f"TV grad to {p} is all-zero"


# ---------------------------------------------------------------------
# 5. Jacobian path runs on source != coord resolution (16 -> 20) without error
# ---------------------------------------------------------------------


def test_jac_path_arbitrary_scale(small_model):
    source = tf.random.normal((1, 16, 16, 3))
    coords = _coords(1, 20, 20)  # upscale 16 -> 20
    t = tf.ones((1, 1))
    out, jac = small_model((source, coords, t), return_jac=True)
    assert tuple(out.shape) == (1, 20, 20, 3)
    assert tuple(jac.shape) == (1, 20, 20, 3, 2)
    assert _finite(out) and _finite(jac)


# ---------------------------------------------------------------------
# 6. import smoke
# ---------------------------------------------------------------------


def test_import_smoke():
    from dl_techniques.losses.thera_jacobian_tv import thera_tv_penalty  # noqa: F401
    assert callable(thera_tv_penalty)
