"""``MambaLayer.dt_proj`` must carry the paper's initialization in a real model.

Keras 3 runs the symbolic build pass of a sublayer that is first reached from a
parent's ``call()`` inside a ``StatelessScope``, which RECORDS an ``.assign()``
and then DISCARDS it. ``MambaLayer.build`` used to perform the whole dt_proj
initialization that way, so in the assembled ``Mamba`` model the kernel stayed at
Dense's default glorot and the bias stayed at ZERO.

**The path matters and a shortcut does not reproduce it.** Building a
``MambaResidualBlock`` alone through a top-level functional call gives the
CORRECT values, because that block's ``build()`` calls ``self.mamba.build(...)``
explicitly — a direct build, where the assign survives. Only the assembled
``Mamba``, whose ``call()`` reaches each block from inside the model's own
``call()``, takes the dead path. So every test here goes through ``Mamba``.

That matters because ``softplus(dt_bias)`` IS the selective-SSM timestep. At a
zero bias it is ``softplus(0) = 0.693147`` for every channel: ~7x above
``dt_max`` and ~660x above ``dt_min``, and identical across channels, which is
the opposite of the log-uniform spread over ``[dt_min, dt_max]`` the
initialization exists to produce.

Measured 2026-08-17, CPU, before the fix, on ``Mamba(num_layers=2)``:

    both MambaLayers: bias all zero, so softplus(bias) == 0.693147 for every
    channel -- ~7x above dt_max (0.1) and ~660x above dt_min (0.001), and
    identical across channels, where the initialization exists to spread it
    log-uniformly over [dt_min, dt_max].

After the fix: dt in [0.001001, 0.094282] and [0.001068, 0.096646], i.e. the two
layers also draw independently.

A test that calls ``MambaLayer.build(...)`` directly is the test that missed this
for the whole life of the package -- it takes the one path where the assign
survives.
"""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.mamba.components import MambaLayer
from dl_techniques.models.mamba.mamba_v1 import Mamba

D_MODEL, SEQ, VOCAB = 16, 8, 64
DT_MIN, DT_MAX = 0.001, 0.1


def _mamba_layers(**kwargs):
    """Build the assembled model and hand back its MambaLayers.

    This is the dead path. Do NOT replace it with `MambaLayer(...).build(...)`
    or with a lone `MambaResidualBlock`: both build the layer DIRECTLY, where
    the discarded-assign defect does not reproduce.
    """
    cfg = dict(vocab_size=VOCAB, d_model=D_MODEL, num_layers=2,
               d_state=8, d_conv=4, expand=2)
    cfg.update(kwargs)
    model = Mamba(**cfg)
    model(ops.zeros((2, SEQ), dtype="int32"))

    layers = [l for l in model._flatten_layers()
              if isinstance(l, MambaLayer)]
    assert len(layers) == cfg["num_layers"], (
        f"found {len(layers)} MambaLayers, expected {cfg['num_layers']}; "
        f"this test is not reaching what it thinks it is"
    )
    return layers


def _layer_direct(**kwargs):
    """A directly-built `MambaLayer`, for the knobs `Mamba` does not forward.

    `Mamba.__init__` exposes `d_state`/`d_conv`/`expand`/`dt_rank` but NOT
    `dt_init`/`dt_scale`, so those two can only be reached at layer level. That
    is sound here only because the fix made both paths share ONE mechanism --
    the initializer -- which `test_a_direct_build_agrees_with_the_assembled_model`
    checks. Before the fix a direct build proved nothing about a real model.
    """
    cfg = dict(d_model=D_MODEL, d_state=8, d_conv=4, expand=2)
    cfg.update(kwargs)
    layer = MambaLayer(**cfg)
    layer.build((2, SEQ, D_MODEL))
    return layer


def _softplus(x):
    return np.log1p(np.exp(x))


class TestDtProjInitializationSurvivesTheStatelessBuild:
    """The dt bias and kernel, asserted against their DEFINITIONS."""

    def test_softplus_of_the_bias_lands_in_the_dt_range(self):
        """The defining property: `softplus(dt_bias)` lies in [dt_min, dt_max].

        Derived from what the initialization is FOR -- `softplus(dt_bias)` is
        the SSM timestep -- not from any value read out of the implementation.
        """
        keras.utils.set_random_seed(0)
        for i, layer in enumerate(_mamba_layers()):
            bias = ops.convert_to_numpy(layer.dt_proj.bias)
            dt = _softplus(bias)

            assert not np.all(bias == 0.0), (
                f"layer {i}: dt_proj.bias is all zero, so the initialization "
                f"was discarded by the stateless build"
            )
            assert dt.min() >= DT_MIN * 0.9, (i, dt.min())
            assert dt.max() <= DT_MAX * 1.1, (i, dt.max())

            # Anti-vacuity: it must be a SPREAD. A discarded init gives
            # softplus(0) = 0.693147 identically for every channel, which a
            # bare "is some value present" check would not catch.
            assert dt.max() / dt.min() > 2.0, (
                f"layer {i}: dt is nearly constant "
                f"({dt.min():.6f}..{dt.max():.6f}); a log-uniform draw is not"
            )
            assert not np.allclose(dt, np.log(2.0)), (
                f"layer {i}: softplus(dt_bias) == softplus(0)"
            )

    def test_each_layer_draws_its_own_dt(self):
        """Two layers sharing one dt vector would mean a closed-over constant."""
        keras.utils.set_random_seed(0)
        a, b = _mamba_layers(num_layers=2)
        ba = ops.convert_to_numpy(a.dt_proj.bias)
        bb = ops.convert_to_numpy(b.dt_proj.bias)
        assert not np.allclose(ba, bb), (
            "both layers got an identical dt bias; the draw is not per-layer"
        )

    def test_the_kernel_is_scaled_by_dt_init_std_not_glorot(self):
        """`dt_init='random'` draws uniform in +/- dt_rank**-0.5 * dt_scale."""
        keras.utils.set_random_seed(0)
        for i, layer in enumerate([_layer_direct(dt_init="random", dt_scale=1.0)]):
            kernel = ops.convert_to_numpy(layer.dt_proj.kernel)
            bound = layer.dt_rank ** -0.5 * layer.dt_scale

            assert np.abs(kernel).max() <= bound * (1.0 + 1e-5), i
            # Over dt_rank*d_inner draws the max approaches the bound; glorot on
            # this shape peaked at 0.391 against a bound of 1.0 (measured).
            assert np.abs(kernel).max() > bound * 0.5, (
                f"layer {i}: max|kernel| = {np.abs(kernel).max():.6f} against a "
                f"bound of {bound:.6f}; this looks like Dense's default glorot"
            )

    def test_constant_dt_init_fills_the_kernel(self):
        """The isolating case: EVERY entry equals dt_init_std, which no default
        initializer produces and no discarded assign could leave behind."""
        keras.utils.set_random_seed(0)
        for layer in [_layer_direct(dt_init="constant", dt_scale=1.0)]:
            kernel = ops.convert_to_numpy(layer.dt_proj.kernel)
            bound = layer.dt_rank ** -0.5 * layer.dt_scale
            np.testing.assert_allclose(kernel, bound, rtol=1e-6)

    def test_dt_scale_actually_scales(self):
        """Anti-vacuity for the whole class: the knob must move the result."""
        keras.utils.set_random_seed(0)
        small = _layer_direct(dt_init="constant", dt_scale=1.0)
        keras.utils.set_random_seed(0)
        large = _layer_direct(dt_init="constant", dt_scale=4.0)

        ks = float(np.abs(ops.convert_to_numpy(small.dt_proj.kernel)).max())
        kl = float(np.abs(ops.convert_to_numpy(large.dt_proj.kernel)).max())
        assert kl == pytest.approx(4.0 * ks, rel=1e-5)

    def test_a_direct_build_agrees_with_the_assembled_model(self):
        """The control that named the defect.

        Before the fix these two paths disagreed completely -- a direct build
        gave the CORRECT values, which is exactly why the existing suite was
        blind to it. Both must now satisfy the same property.
        """
        keras.utils.set_random_seed(0)
        direct = MambaLayer(d_model=D_MODEL, d_state=8, d_conv=4, expand=2)
        direct.build((2, SEQ, D_MODEL))

        keras.utils.set_random_seed(0)
        assembled = _mamba_layers()[0]

        for layer, tag in ((direct, "direct"), (assembled, "assembled")):
            bias = ops.convert_to_numpy(layer.dt_proj.bias)
            dt = _softplus(bias)
            assert not np.all(bias == 0.0), f"{tag}: bias all zero"
            assert dt.min() >= DT_MIN * 0.9, tag
            assert dt.max() <= DT_MAX * 1.1, tag
