"""RED proof for `mamba/components_v2.py`'s build() (review finding C-37(b)).

`Mamba2Layer.build` created only `A_log`, `D` and `dt_bias` and never built
`in_proj`, `conv1d`, `norm` or `out_proj`; `Mamba2ResidualBlock` had no `build`
at all. v1 does it correctly at `components.py:282-300`.

This was masked inside `Mamba2` (no `build` override, so Keras takes the
build-by-run path), but a standalone `.save`/`load_model` -- or embedding in
any parent that overrides `build` -- restores weights into sub-layers that do
not exist.

Per `plans/SYSTEM.md`, the fix has two halves and both are pinned here: the
sub-layers `call()` runs must be built, and the ones it skips must NOT be, so
that the explicit and lazy paths produce the SAME weight set. `rmsnorm=False`
is the discriminating configuration for the second half.

The `np.random` seeding item from the same finding gets its own arm. The
review's premise -- that `keras.utils.set_random_seed` does not reach v2's
initialization -- is REFUTED by measurement: `set_random_seed` calls
`np.random.seed(seed)`, so two seeded builds agree bit-for-bit. The assertion
stays as a regression guard on that fact.

CPU only.
"""

import os
import re
import tempfile

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.models.mamba.components_v2 import (
    Mamba2Layer,
    Mamba2ResidualBlock,
)

D_MODEL = 16
D_STATE = 8
HEADDIM = 4
SEQ_LEN = 6
BATCH = 2


def _inputs():
    rng = np.random.default_rng(0)
    return rng.standard_normal((BATCH, SEQ_LEN, D_MODEL)).astype("float32")


def _layer(**overrides):
    kwargs = dict(d_model=D_MODEL, d_state=D_STATE, headdim=HEADDIM, expand=2)
    kwargs.update(overrides)
    return Mamba2Layer(**kwargs)


def _relative_weight_names(layer):
    """Weight paths, normalized against Keras' auto-generated instance names.

    The outermost segment is dropped, and the `_<n>` uniquifying suffix Keras
    appends to an unnamed layer is stripped from the rest. `Mamba2ResidualBlock`
    constructs its `Mamba2Layer` without an explicit `name`, so two instances in
    the same process get `mamba2_layer_6` and `mamba2_layer_7`; that difference
    is about construction order, not about which weights exist.
    """
    return sorted(
        re.sub(r"_\d+(?=/|$)", "", w.path.split("/", 1)[-1])
        for w in layer.weights
    )


class TestMamba2LayerBuild:

    def test_explicit_build_creates_the_sublayer_weights(self):
        layer = _layer()
        layer.build((None, SEQ_LEN, D_MODEL))
        names = _relative_weight_names(layer)
        for expected in ("in_proj", "conv1d", "out_proj", "rmsnorm"):
            assert any(expected in n for n in names), (
                f"{expected} was not built by build(); names={names}"
            )

    @pytest.mark.parametrize("rmsnorm", [True, False])
    def test_explicit_build_matches_lazy_weight_set(self, rmsnorm):
        """Every sub-layer `call()` runs must exist after `build()` alone.

        MEASURED limitation, stated so nobody over-trusts this arm: Keras runs
        `build()` on the lazy path too, so this comparison catches an
        UNDER-build (the actual defect) but is blind to an OVER-build -- both
        sides would gain the same spurious weight. `rmsnorm=False` over-building
        is covered by `test_rmsnorm_false_creates_no_norm_weights`, which
        asserts the layout directly instead of comparing two paths.
        """
        x = _inputs()

        lazy = _layer(rmsnorm=rmsnorm)
        lazy(x, training=False)

        explicit = _layer(rmsnorm=rmsnorm)
        explicit.build((None, SEQ_LEN, D_MODEL))

        lazy_names = _relative_weight_names(lazy)
        explicit_names = _relative_weight_names(explicit)
        assert explicit_names == lazy_names, (
            f"[rmsnorm={rmsnorm}] weight sets differ.\nonly explicit: "
            f"{sorted(set(explicit_names) - set(lazy_names))}\nonly lazy: "
            f"{sorted(set(lazy_names) - set(explicit_names))}"
        )
        assert len(explicit.weights) == len(lazy.weights)

    def test_rmsnorm_false_creates_no_norm_weights(self):
        """The other half of the SYSTEM.md contract: build no MORE than that.

        `call()` skips `self.norm` when `rmsnorm=False`, so `build()` must not
        create it. An unused sub-layer's weights change the `.keras` layout --
        a silent checkpoint break dressed up as a fix.
        """
        layer = _layer(rmsnorm=False)
        layer.build((None, SEQ_LEN, D_MODEL))
        names = _relative_weight_names(layer)
        assert not any("norm" in n for n in names), names
        # And the sub-layer is not even constructed.
        assert not hasattr(layer, "norm")

    def test_standalone_save_load_preserves_output_values(self):
        """A `.keras` round trip of a MODEL wrapping only this layer."""
        x = _inputs()
        inp = keras.Input(shape=(SEQ_LEN, D_MODEL))
        out = _layer()(inp)
        model = keras.Model(inp, out)
        before = ops.convert_to_numpy(model(x, training=False))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mamba2_layer.keras")
            model.save(path)
            restored = keras.saving.load_model(path)
        after = ops.convert_to_numpy(restored(x, training=False))

        np.testing.assert_allclose(
            before, after, atol=1e-6,
            err_msg="Mamba2Layer output changed across a .keras round trip",
        )


class TestMamba2ResidualBlockBuild:

    def _block(self):
        return Mamba2ResidualBlock(
            d_model=D_MODEL, d_state=D_STATE, d_conv=4, expand=2,
            headdim=HEADDIM, d_ssm=D_MODEL * 2,
        )

    def test_explicit_build_matches_lazy_weight_set(self):
        x = _inputs()

        lazy = self._block()
        lazy(x)

        explicit = self._block()
        explicit.build((None, SEQ_LEN, D_MODEL))

        assert _relative_weight_names(explicit) == _relative_weight_names(lazy)

    def test_block_survives_a_keras_round_trip_by_value(self):
        x = _inputs()
        inp = keras.Input(shape=(SEQ_LEN, D_MODEL))
        hidden, residual = self._block()(inp)
        model = keras.Model(inp, [hidden, residual])
        before = [ops.convert_to_numpy(t) for t in model(x, training=False)]

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mamba2_block.keras")
            model.save(path)
            restored = keras.saving.load_model(path)
        after = [ops.convert_to_numpy(t) for t in restored(x, training=False)]

        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b, atol=1e-6)


class TestInitializationIsReproducible:

    def test_two_seeded_builds_agree(self):
        """`keras.utils.set_random_seed(N)` must pin A_log and dt_bias.

        The review expected this to FAIL, on the theory that drawing from the
        global `np.random` stream escapes Keras' seeding. It does not:
        `keras.utils.set_random_seed` calls `np.random.seed(seed)`. Measured
        difference is exactly 0.0. Kept as a regression guard on that fact --
        if the initialization is ever moved to `keras.random`, or if a
        library-level `np.random` draw is introduced between the seed and the
        build, this is what notices.
        """
        def build_once():
            keras.utils.set_random_seed(1234)
            layer = _layer()
            layer.build((None, SEQ_LEN, D_MODEL))
            return (ops.convert_to_numpy(layer.A_log),
                    ops.convert_to_numpy(layer.dt_bias))

        a1, d1 = build_once()
        a2, d2 = build_once()

        np.testing.assert_array_equal(
            a1, a2, err_msg="A_log is not reproducible under set_random_seed")
        np.testing.assert_array_equal(
            d1, d2, err_msg="dt_bias is not reproducible under set_random_seed")

    def test_different_seeds_give_different_weights(self):
        """ANTI-VACUITY: the weights are genuinely random, not constant."""
        def build_with(seed):
            keras.utils.set_random_seed(seed)
            layer = _layer()
            layer.build((None, SEQ_LEN, D_MODEL))
            return ops.convert_to_numpy(layer.A_log)

        assert not np.array_equal(build_with(1), build_with(2))


class TestNormBeforeGateDefault:

    def test_default_is_gate_then_norm(self):
        """The reference Mamba-2 gated RMSNorm normalizes `y * silu(z)`."""
        assert _layer().norm_before_gate is False

    def test_the_flag_actually_changes_the_output(self):
        """ANTI-VACUITY: the two branches are not numerically the same.

        Both share the same `norm` weights, so this is a value change, not a
        layout change -- which is exactly why it could be flipped.
        """
        x = _inputs()
        keras.utils.set_random_seed(7)
        a = _layer(norm_before_gate=False)
        out_a = ops.convert_to_numpy(a(x, training=False))

        b = _layer(norm_before_gate=True)
        b.build((None, SEQ_LEN, D_MODEL))
        for wa, wb in zip(a.weights, b.weights):
            wb.assign(wa)
        out_b = ops.convert_to_numpy(b(x, training=False))

        assert not np.allclose(out_a, out_b, atol=1e-6), (
            "norm_before_gate has no effect on the output"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
