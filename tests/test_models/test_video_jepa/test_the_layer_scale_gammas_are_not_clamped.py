"""``CausalSelfAttnMLPBlock``'s two LayerScale gammas: init value and sign.

``CausalSelfAttnMLPBlock`` holds two ``LayerScale`` sub-layers, ``gamma_a``
(attention branch) and ``gamma_m`` (MLP branch). Both are constructed with two
NON-DEFAULT arguments, and each ``LayerScale`` default fails SILENTLY and
NUMERICALLY if the argument is dropped:

* ``initializer="ones"`` starts gamma at ``1.0`` instead of ``layer_scale_init``
  (``1e-5`` by default) — 100000x on the residual branch, which erases the
  identity-at-init property the module docstring promises.
* ``constraint="non_neg"`` maps a legitimately negative gamma such as ``-0.5``
  to ``-0.0``.

Neither raises, and neither is visible to a shape, config or round-trip oracle,
so each gets its own guard. See the ``# DECISION`` anchor at
``layers/geometric/clifford_block.py`` and that package's sibling guards in
``tests/test_layers/test_geometric/test_clifford_block.py``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.video_jepa.predictor import (
    CausalSelfAttnMLPBlock,
)

DIM = 8
SEQ = 6


def _built(layer_scale_init: float) -> CausalSelfAttnMLPBlock:
    """Build a small block at a chosen ``layer_scale_init``."""
    block = CausalSelfAttnMLPBlock(
        dim=DIM,
        num_heads=2,
        dim_head=4,
        mlp_dim=16,
        layer_scale_init=layer_scale_init,
        name="blk",
    )
    block.build((None, SEQ, DIM))
    return block


@pytest.mark.parametrize("branch", ["gamma_a", "gamma_m"])
@pytest.mark.parametrize("init_val", [1e-5, 1e-3, 0.25])
def test_layer_scale_gamma_initializes_to_layer_scale_init(branch, init_val):
    """gamma == layer_scale_init, NOT LayerScale's ``ones`` default."""
    block = _built(init_val)
    gamma = keras.ops.convert_to_numpy(getattr(block, branch).gamma)
    assert gamma.shape == (DIM,)
    np.testing.assert_array_equal(
        gamma,
        np.full((DIM,), init_val, dtype=gamma.dtype),
        err_msg=(
            f"{branch} must start at layer_scale_init={init_val}; got "
            f"{gamma[0]!r}. A reading of 1.0 means the `initializer=` argument "
            f"was dropped and LayerScale's `ones` default took over -- "
            f"{1.0 / init_val:.0f}x too large on the residual branch, and the "
            f"block is no longer near-identity at init."
        ),
    )


@pytest.mark.parametrize("branch", ["gamma_a", "gamma_m"])
def test_layer_scale_gamma_is_free_to_go_negative(branch):
    """A negative gamma survives a TRAINING STEP and reaches the output.

    MEASURED: a Keras 3 constraint is NOT applied by ``Variable.assign``. It is
    applied by the optimizer, once per step, as
    ``variable.assign(variable.constraint(variable))``. A guard that only
    assigns and reads back is therefore BLIND to the ``non_neg`` default. This
    one takes one SGD step at ``learning_rate=0.0``, where the constraint is
    the ONLY thing that can move the value, and then checks the forward pass
    too so the negative value is proven to reach the multiply.
    """
    block = _built(1e-5)
    other = "gamma_m" if branch == "gamma_a" else "gamma_a"

    rng = np.random.default_rng(20260831)
    x = rng.standard_normal((2, SEQ, DIM)).astype("float32")
    block(x, training=False)

    negative = np.full((DIM,), -0.5, dtype="float32")
    getattr(block, branch).gamma.assign(negative)
    # Silence the OTHER branch so the residual output isolates this one.
    getattr(block, other).gamma.assign(np.zeros((DIM,), dtype="float32"))

    optimizer = keras.optimizers.SGD(learning_rate=0.0)
    variables = list(block.trainable_variables)
    optimizer.build(variables)
    optimizer.apply_gradients(
        [(keras.ops.zeros(v.shape, dtype=v.dtype), v) for v in variables]
    )

    stored = keras.ops.convert_to_numpy(getattr(block, branch).gamma)
    np.testing.assert_array_equal(
        stored, negative,
        err_msg=(
            f"{branch} was clamped by ONE optimizer step at learning_rate=0.0, "
            "so the only thing that could have moved it is LayerScale's "
            "`constraint='non_neg'` default mapping -0.5 to -0.0. The call "
            "site must pass `constraint=None`."
        ),
    )

    # The block is `x + gamma_a * Attn(LN(x))` then `+ gamma_m * MLP(...)`.
    # With the other branch zeroed, `out - x` is exactly this branch's scaled
    # contribution. Compare it against the same forward pass at gamma = +0.5:
    # a working `constraint=None` gives an EXACTLY sign-flipped residual, a
    # clamp to -0.0 gives a residual of exactly zero.
    residual_neg = keras.ops.convert_to_numpy(block(x, training=False)) - x
    getattr(block, branch).gamma.assign(-negative)
    residual_pos = keras.ops.convert_to_numpy(block(x, training=False)) - x

    assert np.abs(residual_neg).max() > 0.0, (
        f"{branch}'s residual contribution is identically zero, which is what "
        "a gamma clamped to -0.0 produces."
    )
    np.testing.assert_allclose(
        residual_neg, -residual_pos, rtol=1e-6, atol=1e-7,
        err_msg=(
            f"a gamma of -0.5 must produce exactly the negated residual of a "
            f"gamma of +0.5 on branch {branch}; it did not, so the negative "
            f"value did not reach the multiply."
        ),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
