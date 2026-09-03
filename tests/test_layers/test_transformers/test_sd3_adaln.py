"""
Test suite for the SD3 AdaLN modulation trio.

Covers, for each of :class:`AdaLayerNormZero`, :class:`AdaLayerNormZeroX`,
:class:`AdaLayerNormContinuous`: instantiation + ctor validation, forward
return shapes (``x_norm`` ``(B,N,dim)``; chunks ``(B,dim)``), modulation
broadcast across ``N>1``, AdaLN-Zero identity-at-init (``x_norm`` equals a
plain no-affine LayerNorm of ``x`` because the modulation Dense is
zero-initialized), ``compute_output_shape`` pre/post build, ``get_config`` /
``from_config`` round-trip, a full ``.keras`` save/load round-trip via a
two-input Functional model, and variable batch / variable N.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.sd3_adaln import (
    AdaLayerNormZero,
    AdaLayerNormZeroX,
    AdaLayerNormContinuous,
)


DIM = 64
N = 16
BATCH = 2
EPS = 1e-6


def _no_affine_layernorm(x_np: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Reference no-affine LayerNorm over the last axis."""
    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    return (x_np - mean) / np.sqrt(var + eps)


@pytest.fixture
def sample():
    keras.utils.set_random_seed(42)
    x = keras.random.normal((BATCH, N, DIM))
    cond = keras.random.normal((BATCH, DIM))
    return x, cond


# =====================================================================
# AdaLayerNormZero
# =====================================================================


class TestAdaLayerNormZero:

    @pytest.fixture
    def layer(self):
        return AdaLayerNormZero(dim=DIM, eps=EPS)

    def test_initialization(self, layer):
        assert layer.dim == DIM
        assert layer.eps == EPS
        assert layer.linear.units == 6 * DIM
        assert layer.norm.center is False
        assert layer.norm.scale is False

    def test_ctor_raises_on_bad_dim(self):
        with pytest.raises(ValueError):
            AdaLayerNormZero(dim=0)

    def test_ctor_raises_on_bad_eps(self):
        with pytest.raises(ValueError):
            AdaLayerNormZero(dim=DIM, eps=0.0)

    def test_forward_shapes(self, layer, sample):
        x, cond = sample
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = layer([x, cond])
        assert tuple(x_norm.shape) == (BATCH, N, DIM)
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp):
            assert tuple(chunk.shape) == (BATCH, DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(x_norm)))

    def test_modulation_broadcast_across_n(self, layer):
        """Modulation (B,dim) -> (B,1,dim) must broadcast over arbitrary N."""
        keras.utils.set_random_seed(1)
        for n in (1, 5, 33):
            x = keras.random.normal((BATCH, n, DIM))
            cond = keras.random.normal((BATCH, DIM))
            x_norm = layer([x, cond])[0]
            assert tuple(x_norm.shape) == (BATCH, n, DIM)

    def test_identity_at_init(self, layer, sample):
        """Zero-init Dense => x_norm == no-affine LayerNorm(x); gates ~0."""
        x, cond = sample
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = layer([x, cond])
        ref = _no_affine_layernorm(keras.ops.convert_to_numpy(x))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x_norm), ref, atol=1e-5
        )
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(chunk), 0.0, atol=1e-6
            )

    def test_compute_output_shape_before_build(self):
        layer = AdaLayerNormZero(dim=DIM)
        out = layer.compute_output_shape([(BATCH, N, DIM), (BATCH, DIM)])
        assert out == [
            (BATCH, N, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
        ]

    def test_compute_output_shape_matches_actual(self, layer, sample):
        x, cond = sample
        outs = layer([x, cond])
        computed = layer.compute_output_shape([tuple(x.shape), tuple(cond.shape)])
        for c, o in zip(computed, outs):
            assert c == tuple(o.shape)

    def test_get_config_round_trip(self):
        layer = AdaLayerNormZero(dim=128, eps=1e-5)
        cfg = layer.get_config()
        assert cfg["dim"] == 128
        assert cfg["eps"] == 1e-5
        rebuilt = AdaLayerNormZero.from_config(cfg)
        assert rebuilt.dim == 128
        assert rebuilt.eps == 1e-5

    def test_keras_serialization_round_trip(self, sample):
        x, cond = sample
        x_in = keras.Input(shape=(N, DIM), name="x")
        c_in = keras.Input(shape=(DIM,), name="cond")
        outs = AdaLayerNormZero(dim=DIM)([x_in, c_in])
        model = keras.Model([x_in, c_in], list(outs))
        inputs = {
            "x": keras.ops.convert_to_numpy(x),
            "cond": keras.ops.convert_to_numpy(cond),
        }
        before = model.predict(inputs, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaln_zero.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = reloaded.predict(inputs, verbose=0)
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_variable_batch(self, layer):
        keras.utils.set_random_seed(7)
        for b in (1, 3, 5):
            x = keras.random.normal((b, N, DIM))
            cond = keras.random.normal((b, DIM))
            x_norm, gate_msa = layer([x, cond])[0], layer([x, cond])[1]
            assert tuple(x_norm.shape) == (b, N, DIM)
            assert tuple(gate_msa.shape) == (b, DIM)


# =====================================================================
# AdaLayerNormZeroX
# =====================================================================


class TestAdaLayerNormZeroX:

    @pytest.fixture
    def layer(self):
        return AdaLayerNormZeroX(dim=DIM, eps=EPS)

    def test_initialization(self, layer):
        assert layer.dim == DIM
        assert layer.linear.units == 9 * DIM

    def test_ctor_raises_on_bad_dim(self):
        with pytest.raises(ValueError):
            AdaLayerNormZeroX(dim=-1)

    def test_forward_shapes(self, layer, sample):
        x, cond = sample
        (x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         x_norm2, gate_msa2) = layer([x, cond])
        assert tuple(x_norm.shape) == (BATCH, N, DIM)
        assert tuple(x_norm2.shape) == (BATCH, N, DIM)
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2):
            assert tuple(chunk.shape) == (BATCH, DIM)

    def test_modulation_broadcast_across_n(self, layer):
        keras.utils.set_random_seed(2)
        for n in (1, 5, 33):
            x = keras.random.normal((BATCH, n, DIM))
            cond = keras.random.normal((BATCH, DIM))
            outs = layer([x, cond])
            assert tuple(outs[0].shape) == (BATCH, n, DIM)
            assert tuple(outs[5].shape) == (BATCH, n, DIM)

    def test_identity_at_init(self, layer, sample):
        """Both x_norm and x_norm2 == no-affine LayerNorm(x); gates ~0."""
        x, cond = sample
        (x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         x_norm2, gate_msa2) = layer([x, cond])
        ref = _no_affine_layernorm(keras.ops.convert_to_numpy(x))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x_norm), ref, atol=1e-5
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x_norm2), ref, atol=1e-5
        )
        # x_norm and x_norm2 share norm(x) at init => identical.
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x_norm),
            keras.ops.convert_to_numpy(x_norm2),
            atol=1e-6,
        )
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(chunk), 0.0, atol=1e-6
            )

    def test_compute_output_shape_before_build(self):
        layer = AdaLayerNormZeroX(dim=DIM)
        out = layer.compute_output_shape([(BATCH, N, DIM), (BATCH, DIM)])
        assert out == [
            (BATCH, N, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, N, DIM),
            (BATCH, DIM),
        ]

    def test_compute_output_shape_matches_actual(self, layer, sample):
        x, cond = sample
        outs = layer([x, cond])
        computed = layer.compute_output_shape([tuple(x.shape), tuple(cond.shape)])
        for c, o in zip(computed, outs):
            assert c == tuple(o.shape)

    def test_get_config_round_trip(self):
        layer = AdaLayerNormZeroX(dim=96, eps=1e-4)
        cfg = layer.get_config()
        assert cfg["dim"] == 96
        assert cfg["eps"] == 1e-4
        rebuilt = AdaLayerNormZeroX.from_config(cfg)
        assert rebuilt.dim == 96
        assert rebuilt.eps == 1e-4

    def test_keras_serialization_round_trip(self, sample):
        x, cond = sample
        x_in = keras.Input(shape=(N, DIM), name="x")
        c_in = keras.Input(shape=(DIM,), name="cond")
        outs = AdaLayerNormZeroX(dim=DIM)([x_in, c_in])
        model = keras.Model([x_in, c_in], list(outs))
        inputs = {
            "x": keras.ops.convert_to_numpy(x),
            "cond": keras.ops.convert_to_numpy(cond),
        }
        before = model.predict(inputs, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaln_zerox.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = reloaded.predict(inputs, verbose=0)
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_variable_batch(self, layer):
        keras.utils.set_random_seed(8)
        for b in (1, 3, 5):
            x = keras.random.normal((b, N, DIM))
            cond = keras.random.normal((b, DIM))
            outs = layer([x, cond])
            assert tuple(outs[0].shape) == (b, N, DIM)
            assert tuple(outs[5].shape) == (b, N, DIM)


# =====================================================================
# AdaLayerNormContinuous
# =====================================================================


class TestAdaLayerNormContinuous:

    @pytest.fixture
    def layer(self):
        return AdaLayerNormContinuous(dim=DIM, eps=EPS)

    def test_initialization(self, layer):
        assert layer.dim == DIM
        assert layer.linear.units == 2 * DIM
        assert layer.norm.center is False
        assert layer.norm.scale is False

    def test_ctor_raises_on_bad_dim(self):
        with pytest.raises(ValueError):
            AdaLayerNormContinuous(dim=0)

    def test_forward_shape_single_tensor(self, layer, sample):
        x, cond = sample
        out = layer([x, cond])
        assert not isinstance(out, (list, tuple))
        assert tuple(out.shape) == (BATCH, N, DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_modulation_broadcast_across_n(self, layer):
        keras.utils.set_random_seed(3)
        for n in (1, 5, 33):
            x = keras.random.normal((BATCH, n, DIM))
            cond = keras.random.normal((BATCH, DIM))
            out = layer([x, cond])
            assert tuple(out.shape) == (BATCH, n, DIM)

    def test_identity_at_init(self, layer, sample):
        """Zero-init Dense => output == no-affine LayerNorm(x)."""
        x, cond = sample
        out = layer([x, cond])
        ref = _no_affine_layernorm(keras.ops.convert_to_numpy(x))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out), ref, atol=1e-5
        )

    def test_compute_output_shape_before_build(self):
        layer = AdaLayerNormContinuous(dim=DIM)
        out = layer.compute_output_shape([(BATCH, N, DIM), (BATCH, DIM)])
        assert out == (BATCH, N, DIM)

    def test_compute_output_shape_matches_actual(self, layer, sample):
        x, cond = sample
        out = layer([x, cond])
        computed = layer.compute_output_shape([tuple(x.shape), tuple(cond.shape)])
        assert computed == tuple(out.shape)

    def test_get_config_round_trip(self):
        layer = AdaLayerNormContinuous(dim=80, eps=1e-5)
        cfg = layer.get_config()
        assert cfg["dim"] == 80
        assert cfg["eps"] == 1e-5
        rebuilt = AdaLayerNormContinuous.from_config(cfg)
        assert rebuilt.dim == 80
        assert rebuilt.eps == 1e-5

    def test_keras_serialization_round_trip(self, sample):
        x, cond = sample
        x_in = keras.Input(shape=(N, DIM), name="x")
        c_in = keras.Input(shape=(DIM,), name="cond")
        out = AdaLayerNormContinuous(dim=DIM)([x_in, c_in])
        model = keras.Model([x_in, c_in], out)
        inputs = {
            "x": keras.ops.convert_to_numpy(x),
            "cond": keras.ops.convert_to_numpy(cond),
        }
        before = model.predict(inputs, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaln_continuous.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = reloaded.predict(inputs, verbose=0)
        np.testing.assert_allclose(before, after, atol=1e-6)

    def test_variable_batch(self, layer):
        keras.utils.set_random_seed(9)
        for b in (1, 3, 5):
            x = keras.random.normal((b, N, DIM))
            cond = keras.random.normal((b, DIM))
            out = layer([x, cond])
            assert tuple(out.shape) == (b, N, DIM)


# =====================================================================
# Chunk roles: which learned scalar reaches which sub-op
# =====================================================================
#
# Everything above this line runs at the layers' zero-initialised default,
# where every modulation chunk is exactly 0. That is the regime in which a
# `shift`/`scale` transposition is INVISIBLE: `modulate(n, 0, 0)` is the
# identity whichever name is bound to whichever slot, and the measured
# difference between the correct split and the transposed one is 0.0. At a
# non-zero bias the same transposition moves the output by 1.619. Every arm
# below therefore writes a non-zero bias first; an arm written at init would
# be structurally incapable of failing.
#
# The chunk ORDER is written out by hand here, from the layers' documented
# return contract. It is never read back from the implementation -- a name
# list derived from the code under test cannot disagree with it.

#: The 6-way split consumed by :class:`AdaLayerNormZero`.
ADALN_ZERO_CHUNK_NAMES = (
    "shift_msa", "scale_msa", "gate_msa",
    "shift_mlp", "scale_mlp", "gate_mlp",
)

#: The 9-way split consumed by :class:`AdaLayerNormZeroX`: the 6-way order,
#: then the second (dual) attention triple.
ADALN_ZEROX_CHUNK_NAMES = ADALN_ZERO_CHUNK_NAMES + (
    "shift_msa2", "scale_msa2", "gate_msa2",
)

#: An arbitrary non-zero modulation magnitude. Nothing depends on the value.
VALUE = 0.7


def _bias_with(names, **chunks: float) -> np.ndarray:
    """Build a ``(len(names) * DIM,)`` bias vector with the named chunks set flat.

    :param names: the chunk-name tuple defining slot order (6-way or 9-way).
    :param chunks: ``chunk_name=value`` pairs; every other chunk stays zero.
    :returns: a float32 vector suitable for ``layer.linear.bias.assign``.
    :raises ValueError: if a name is not in ``names``.
    """
    vec = np.zeros((len(names) * DIM,), dtype="float32")
    for name, value in chunks.items():
        if name not in names:
            raise ValueError(f"unknown chunk {name!r}, expected one of {names}")
        index = names.index(name)
        vec[index * DIM:(index + 1) * DIM] = value
    return vec


def _built(layer, x, cond):
    """Run the layer once so its ``Dense`` exists, then return it."""
    layer([x, cond])
    return layer


def _outputs(layer, bias: np.ndarray, x, cond):
    """Set the modulation bias, run the layer once, return numpy outputs."""
    layer.linear.bias.assign(keras.ops.convert_to_tensor(bias))
    return [
        np.asarray(keras.ops.convert_to_numpy(t)) for t in layer([x, cond])
    ]


class TestAdaLayerNormZeroXChunkRolesAreNotInterchangeable:
    """``shift`` adds, ``scale`` multiplies: chunks 0/1 and 6/7 are not pairs.

    ``AdaLayerNormZeroX`` is live -- ``MMDiTBlock(use_dual_attention=True)``
    constructs it, and the SD3.5-medium configuration enables 13 such layers --
    yet every arm above this one runs at zero bias, where transposing
    ``shift_msa`` with ``scale_msa`` (or ``shift_msa2`` with ``scale_msa2``)
    changes the output by exactly nothing.

    The discriminator is read off the modulated stream itself, by comparing two
    runs of the layer against each other. No expected value is ever taken from
    the layer's own formula:

    * an ADDITIVE chunk moves every position by the SAME number, so the delta
      is constant and equal to the value written into the bias;
    * a MULTIPLICATIVE chunk moves each position in proportion to its own
      normalised activation, so the delta is ``value * base``.

    On a non-degenerate input those two are far apart, and each arm asserts the
    spread that makes the other's claim false, so neither can pass under the
    transposed split. The gate of the triple under test is held OPEN throughout
    -- the layer only returns its gates, so a gate must not be able to change
    ``x_norm``, and holding it non-zero proves the reading is not gate-borne.

    The zero-input arm is the same statement without a tolerance: ``norm(0)``
    is ``0``, so ``0 * (1 + scale) == 0`` exactly and a multiplicative chunk has
    nothing to act on, while an additive one still injects its value.
    """

    #: ``(output index, shift name, scale name, gate name)`` per modulated
    #: stream: chunks 0/1 feed ``x_norm`` (output 0), chunks 6/7 feed
    #: ``x_norm2`` (output 5).
    STREAMS = (
        (0, "shift_msa", "scale_msa", "gate_msa"),
        (5, "shift_msa2", "scale_msa2", "gate_msa2"),
    )

    @pytest.fixture
    def probe(self, sample):
        x, cond = sample
        layer = _built(AdaLayerNormZeroX(dim=DIM, eps=EPS), x, cond)
        return layer, x, cond

    @pytest.mark.parametrize("index,shift,scale,gate", STREAMS)
    def test_the_shift_chunk_moves_every_position_by_the_same_amount(
        self, probe, index, shift, scale, gate
    ):
        layer, x, cond = probe
        names = ADALN_ZEROX_CHUNK_NAMES
        base = _outputs(layer, _bias_with(names, **{gate: VALUE}), x, cond)[index]
        moved = _outputs(
            layer, _bias_with(names, **{gate: VALUE, shift: VALUE}), x, cond
        )[index]
        delta = moved - base

        # Anti-vacuity: a multiplicative chunk would give `VALUE * base`, whose
        # spread must be far above the tolerance below -- otherwise the
        # constant-delta assertion does not exclude the other role.
        spread = float(np.std(VALUE * base))
        assert spread > 1e-2, (
            "the normalised activation is nearly constant on this input, so an "
            f"additive and a multiplicative chunk look alike (std = {spread})"
        )

        np.testing.assert_allclose(
            delta,
            np.full_like(delta, VALUE),
            rtol=0,
            atol=1e-5,
            err_msg=(
                f"{shift} did not act additively: its delta is not the constant "
                "written into the bias, so it is being consumed as the "
                "multiplicative scale -- the 9-way chunk order in "
                "AdaLayerNormZeroX.call is wrong"
            ),
        )

    @pytest.mark.parametrize("index,shift,scale,gate", STREAMS)
    def test_the_scale_chunk_moves_each_position_in_proportion_to_itself(
        self, probe, index, shift, scale, gate
    ):
        layer, x, cond = probe
        names = ADALN_ZEROX_CHUNK_NAMES
        base = _outputs(layer, _bias_with(names, **{gate: VALUE}), x, cond)[index]
        moved = _outputs(
            layer, _bias_with(names, **{gate: VALUE, scale: VALUE}), x, cond
        )[index]
        delta = moved - base

        # Anti-vacuity: an additive chunk gives a CONSTANT delta; this one must
        # vary, otherwise the proportionality claim below is unobservable.
        assert float(np.std(delta)) > 1e-2, (
            f"{scale} moved every position by the same amount, i.e. it acted as "
            "an additive shift -- the 9-way chunk order in "
            f"AdaLayerNormZeroX.call is wrong (std = {float(np.std(delta))})"
        )

        np.testing.assert_allclose(
            delta,
            VALUE * base,
            rtol=0,
            atol=1e-5,
            err_msg=(
                f"{scale} is not multiplying the normalised activation -- the "
                "9-way chunk order in AdaLayerNormZeroX.call is wrong"
            ),
        )

    @pytest.mark.parametrize("index,shift,scale,gate", STREAMS)
    def test_on_a_zero_input_only_the_shift_chunk_can_act(
        self, probe, index, shift, scale, gate
    ):
        layer, _, cond = probe
        names = ADALN_ZEROX_CHUNK_NAMES
        x = np.zeros((BATCH, N, DIM), dtype="float32")

        base = _outputs(layer, _bias_with(names, **{gate: VALUE}), x, cond)[index]
        scaled = _outputs(
            layer, _bias_with(names, **{gate: VALUE, scale: VALUE}), x, cond
        )[index]
        shifted = _outputs(
            layer, _bias_with(names, **{gate: VALUE, shift: VALUE}), x, cond
        )[index]

        # atol=0: `0 * (1 + scale)` is exactly `0`, so this is an exact claim.
        np.testing.assert_array_equal(
            scaled,
            base,
            err_msg=(
                f"{scale} changed the output of a ZERO input, so it is being "
                "added rather than multiplied"
            ),
        )
        delta_shift = float(np.max(np.abs(shifted - base)))
        assert delta_shift > 1e-5, (
            f"{shift} could not move a ZERO input, so it is being multiplied "
            f"rather than added (max |delta| = {delta_shift})"
        )


class TestAdaLayerNormZeroXNineSlotAttribution:
    """Every one of the nine chunks carries its own bias slot, and no other.

    The two arms above pin the ROLE of chunks 0/1 and 6/7. They say nothing
    about the remaining five -- ``shift_mlp``/``scale_mlp`` (3/4) and the three
    gates (2/5/8) -- which this layer returns without consuming, so a rotation
    among them is invisible to any numeric claim made inside the layer.
    Closing a mutation family halfway is indistinguishable from closing it, so
    this arm closes all nine at once.

    A distinct value goes into every slot in a single run. With the modulation
    ``Dense``'s kernel zero-initialised the pre-split vector IS the bias, so:

    * the five returned chunks must each be flat at their own value;
    * ``x_norm`` must equal ``base * (1 + v[1]) + v[0]`` and ``x_norm2`` must
      equal ``base * (1 + v[7]) + v[6]``, with ``base`` measured from a
      separate all-zero-bias run rather than recomputed from the formula.

    The values are pairwise distinct and none is 0, so any permutation of the
    nine slots is convicted by at least one of these equalities.
    """

    #: One distinct non-zero value per slot, in split order. Pairwise distinct
    #: is the property that makes a permutation observable.
    SLOT_VALUES = (0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99)

    def test_each_of_the_nine_chunks_carries_its_own_slot(self, sample):
        x, cond = sample
        names = ADALN_ZEROX_CHUNK_NAMES
        assert len(names) == len(self.SLOT_VALUES) == 9
        assert len(set(self.SLOT_VALUES)) == 9, "slot values must be distinct"

        layer = _built(AdaLayerNormZeroX(dim=DIM, eps=EPS), x, cond)
        base = _outputs(layer, np.zeros((9 * DIM,), dtype="float32"), x, cond)
        x_norm_base, x_norm2_base = base[0], base[5]

        # Anti-vacuity: the normalised stream must vary across positions, or
        # the shift/scale reconstruction below cannot separate v[0] from v[1].
        assert float(np.std(x_norm_base)) > 1e-2, (
            "norm(x) is nearly constant on this input, so the reconstruction "
            "below cannot distinguish an additive slot from a multiplicative one"
        )

        bias = _bias_with(names, **dict(zip(names, self.SLOT_VALUES)))
        outs = _outputs(layer, bias, x, cond)

        v = self.SLOT_VALUES
        # The five chunks the layer only returns: output index -> chunk name.
        for out_index, name in (
            (1, "gate_msa"), (2, "shift_mlp"), (3, "scale_mlp"),
            (4, "gate_mlp"), (6, "gate_msa2"),
        ):
            expected = v[names.index(name)]
            np.testing.assert_allclose(
                outs[out_index],
                np.full_like(outs[out_index], expected),
                rtol=0,
                atol=1e-6,
                err_msg=(
                    f"returned chunk {out_index} does not carry slot "
                    f"{names.index(name)} ({name} = {expected}); the 9-way split "
                    "order in AdaLayerNormZeroX.call is permuted"
                ),
            )

        np.testing.assert_allclose(
            outs[0],
            x_norm_base * (1.0 + v[1]) + v[0],
            rtol=0,
            atol=1e-5,
            err_msg=(
                "x_norm is not norm(x)*(1+slot1)+slot0 -- slots 0/1 (shift_msa, "
                "scale_msa) are not reaching the primary modulation in that order"
            ),
        )
        np.testing.assert_allclose(
            outs[5],
            x_norm2_base * (1.0 + v[7]) + v[6],
            rtol=0,
            atol=1e-5,
            err_msg=(
                "x_norm2 is not norm(x)*(1+slot7)+slot6 -- slots 6/7 "
                "(shift_msa2, scale_msa2) are not reaching the dual modulation "
                "in that order"
            ),
        )


class TestAdaLayerNormZeroSixSlotAttribution:
    """The same statement for the 6-way sibling, beside its own layer.

    ``AdaLayerNormZero``'s split is currently pinned only from OUTSIDE this
    package, by ``tests/test_models/test_dit/test_dit_blocks.py``, which
    exercises it through ``DiTBlock``. That leaves the layer's own suite unable
    to see a permutation of its six chunks, and it leaves the coverage filed
    under a consumer rather than the owner. This arm is the 9-slot arm above,
    minus the dual triple.
    """

    SLOT_VALUES = (0.11, 0.22, 0.33, 0.44, 0.55, 0.66)

    def test_each_of_the_six_chunks_carries_its_own_slot(self, sample):
        x, cond = sample
        names = ADALN_ZERO_CHUNK_NAMES
        assert len(names) == len(self.SLOT_VALUES) == 6
        assert len(set(self.SLOT_VALUES)) == 6, "slot values must be distinct"

        layer = _built(AdaLayerNormZero(dim=DIM, eps=EPS), x, cond)
        x_norm_base = _outputs(
            layer, np.zeros((6 * DIM,), dtype="float32"), x, cond
        )[0]
        assert float(np.std(x_norm_base)) > 1e-2, (
            "norm(x) is nearly constant on this input, so the reconstruction "
            "below cannot distinguish an additive slot from a multiplicative one"
        )

        v = self.SLOT_VALUES
        outs = _outputs(
            layer, _bias_with(names, **dict(zip(names, v))), x, cond
        )

        for out_index, name in (
            (1, "gate_msa"), (2, "shift_mlp"), (3, "scale_mlp"), (4, "gate_mlp"),
        ):
            expected = v[names.index(name)]
            np.testing.assert_allclose(
                outs[out_index],
                np.full_like(outs[out_index], expected),
                rtol=0,
                atol=1e-6,
                err_msg=(
                    f"returned chunk {out_index} does not carry slot "
                    f"{names.index(name)} ({name} = {expected}); the 6-way split "
                    "order in AdaLayerNormZero.call is permuted"
                ),
            )

        np.testing.assert_allclose(
            outs[0],
            x_norm_base * (1.0 + v[1]) + v[0],
            rtol=0,
            atol=1e-5,
            err_msg=(
                "x_norm is not norm(x)*(1+slot1)+slot0 -- slots 0/1 (shift_msa, "
                "scale_msa) are not reaching the modulation in that order"
            ),
        )
