"""Guards for step 26's mamba repairs: the `build` idempotence guard (F-49),
the mislabeled "RMSNorm" (F-50), and the package export gap (F-59).

Two carried premises were REFUTED by measurement here and both refutations are
pinned below, because each one defeats an obvious-looking probe:

1. **F-49's carried RED recipe does not work.** It said to wrap `Mamba2Layer`
   in a parent whose `call()` invokes it twice, expecting
   ``ValueError: Variable .../A_log is already initialized``. Measured: that
   parent succeeds *with no guard at all*, because `Layer.__call__` checks
   `self.built` before `_maybe_build`. A probe that passes identically with and
   without the defect proves nothing. The live path is a **direct** second
   `build(shape)` -- exactly what `Mamba2ResidualBlock.build` does to its child
   -- and the real message is "You cannot add new elements of state ...".

2. **F-50's carried mechanism is wrong.** `LayerNormalization(rms_scaling=True)`
   does NOT mean-centre. It divides by ``sqrt(var(x) + eps)``, i.e. by the
   mean-DEPENDENT standard deviation, rather than by ``sqrt(mean(x**2) + eps)``.
   Not centring and not being RMSNorm are different defects; the assertion below
   pins the arithmetic identity, not the label.

CPU only.
"""

import numpy as np
import pytest
import keras

import dl_techniques.models.mamba as mamba_pkg
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.models.mamba.components import MambaLayer
from dl_techniques.models.mamba.components_v2 import Mamba2Layer, Mamba2ResidualBlock

D_MODEL = 16
SHAPE = (2, 6, D_MODEL)


def _block(**kw):
    return Mamba2ResidualBlock(
        d_model=D_MODEL, d_state=8, headdim=8, d_conv=4, expand=2, d_ssm=None, **kw
    )


class TestBuildIdempotence:
    """F-49. RED against HEAD-before-fix: both assertions raised ValueError."""

    @pytest.mark.parametrize(
        "make",
        [
            lambda: Mamba2Layer(d_model=D_MODEL, d_state=8, headdim=8),
            lambda: MambaLayer(d_model=D_MODEL, d_state=8),
        ],
        ids=["Mamba2Layer", "MambaLayer"],
    )
    def test_a_second_direct_build_is_a_no_op(self, make):
        layer = make()
        layer.build(SHAPE)
        before = len(layer.weights)
        layer.build(SHAPE)
        assert len(layer.weights) == before

    def test_the_residual_block_can_build_an_already_built_child(self):
        """The live path: `Mamba2ResidualBlock.build` calls `self.mamba2.build`
        unconditionally, so a child a forward pass already built used to die."""
        block = _block()
        block.mamba2(np.zeros(SHAPE, dtype="float32"))
        block.build(SHAPE)
        assert len(block.weights) > 0

    def test_the_two_call_parent_probe_is_NOT_a_discriminator(self):
        """Anti-vacuity: documents why the carried RED recipe was discarded.

        This passes both before and after the fix. It is kept so nobody
        re-derives it as a guard.
        """

        class Parent(keras.layers.Layer):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.inner = Mamba2Layer(d_model=D_MODEL, d_state=8, headdim=8)

            def call(self, x):
                return self.inner(self.inner(x))

        assert Parent()(np.zeros(SHAPE, dtype="float32")).shape == SHAPE


class TestTheNormIsReallyRMSNorm:
    """F-50."""

    def test_the_layer_uses_RMSNorm_not_LayerNormalization(self):
        layer = Mamba2Layer(d_model=D_MODEL, d_state=8, headdim=8, rmsnorm=True)
        layer.build(SHAPE)
        assert isinstance(layer.norm, RMSNorm)
        assert isinstance(_block(rmsnorm=True).norm, RMSNorm)

    def test_rms_scaling_is_not_rms_normalisation(self):
        """The measurement that made the swap a code fix and not a doc fix.

        Input carries a per-token mean of 3.0, which is the regime where the two
        differ; `rms_scaling=True` tracks x/std, RMSNorm tracks x/rms.
        """
        x = np.random.RandomState(0).randn(2, 5, 8).astype("float32") + 3.0
        lnrms = np.asarray(keras.layers.LayerNormalization(epsilon=1e-5, rms_scaling=True)(x))
        rms = np.asarray(RMSNorm(epsilon=1e-5)(x))
        closed_form_rms = x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-5)
        closed_form_std = x / np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)

        assert np.max(np.abs(rms - closed_form_rms)) < 1e-5
        assert np.max(np.abs(lnrms - closed_form_std)) < 1e-5
        # ... and it does NOT subtract the mean, contra the carried claim.
        centred = (x - x.mean(-1, keepdims=True)) / np.sqrt(np.var(x, -1, keepdims=True) + 1e-5)
        assert np.max(np.abs(lnrms - centred)) > 1.0
        assert np.max(np.abs(lnrms - rms)) > 1.0


class TestPackageExports:
    """F-59. Every class the package imports must be in `__all__`."""

    @pytest.mark.parametrize(
        "name", ["MambaLayer", "Mamba", "MambaResidualBlock", "Mamba2Layer", "Mamba2", "Mamba2ResidualBlock"]
    )
    def test_public_class_is_exported(self, name):
        assert name in mamba_pkg.__all__
        assert getattr(mamba_pkg, name).__name__ == name
