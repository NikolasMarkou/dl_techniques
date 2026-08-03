"""Tests for the MannLayer (Memory-Augmented Neural Network / NTM).

The LSTM controller (the default) is the primary, fully-supported path and is
exercised end-to-end. The GRU controller path is currently broken by a keras GRU
``return_state`` quirk in this environment (the returned state tensors lose their
batch dimension), so its forward pass is recorded as an expected failure.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.memory.mann import MannLayer

B, S, F = 2, 5, 3
MEM_LOC, MEM_DIM, CTRL = 8, 4, 6
NR, NW = 1, 1
OUT_DIM = CTRL + NR * MEM_DIM  # 10


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, S, F)).astype("float32")


def _make(**kw):
    defaults = dict(memory_locations=MEM_LOC, memory_dim=MEM_DIM, controller_units=CTRL,
                    num_read_heads=NR, num_write_heads=NW)
    defaults.update(kw)
    return MannLayer(**defaults)


class TestMannLayer:

    def test_construction(self):
        layer = _make()
        assert layer.memory_locations == MEM_LOC
        assert layer.controller_type == "lstm"

    @pytest.mark.parametrize("bad", [
        {"memory_locations": 0},
        {"memory_dim": 0},
        {"controller_units": 0},
        {"num_read_heads": -1},
        {"controller_type": "bogus"},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            _make(**bad)

    def test_forward_pass_lstm(self, sample):
        out = _make(controller_type="lstm")(sample)
        assert tuple(out.shape) == (B, S, OUT_DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        assert _make().compute_output_shape((B, S, F)) == (B, S, OUT_DIM)

    def test_compute_output_shape_matches_call(self, sample):
        layer = _make()
        out = layer(sample)
        assert tuple(out.shape) == tuple(layer.compute_output_shape(sample.shape))

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(S, F))
        out = _make(controller_type="lstm", name="mann")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "mann.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"MannLayer": MannLayer}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = _make(num_read_heads=2, num_write_heads=1)
        rebuilt = MannLayer.from_config(layer.get_config())
        assert rebuilt.num_read_heads == 2 and rebuilt.num_write_heads == 1

    @pytest.mark.xfail(reason="keras GRU return_state drops the batch dim in this "
                              "environment; GRU controller forward is broken upstream.",
                       strict=False)
    def test_forward_pass_gru(self, sample):
        out = _make(controller_type="gru")(sample)
        assert tuple(out.shape) == (B, S, OUT_DIM)


class TestShiftDirection:
    """Delta-impulse guards pinning the DIRECTION of MANN location-based addressing.

    Graves et al. 2014 eq. 8 defines the shift as
    ``w_tilde(i) = sum_j w(j) * s(i - j mod N)``. With ``w`` a delta impulse at
    slot 0 this reduces to ``w_tilde(i) = s(i)``: all shift mass placed on
    offset ``+1`` must land at slot ``1``, and offset ``-1`` at slot ``N-1``.

    ``MannLayer`` inlines its 3-tap shift inside the private
    ``_calculate_head_addressing``. That private method is the target here
    because it is the ONLY home of this math in the layer: the public ``call()``
    path buries it under an LSTM controller, a ``Dense`` parameter generator and
    a per-timestep sequence loop, none of which can be driven to place a clean
    delta impulse on a chosen shift tap. Testing through ``call()`` would only
    re-test shapes, which the suite above already does.

    Parameter layout, derived by reading ``_calculate_head_addressing``
    (``M = memory_dim``)::

        params[:, :M]        -> k      (key vector)
        params[:, M]         -> beta   (softplus)
        params[:, M + 1]     -> g      (sigmoid interpolation gate)
        params[:, M + 2:M+5] -> s      (softmax over the 3 shift taps)
        params[:, M + 5]     -> gamma  (softplus, then + 1.0)

    and the layer's own comment fixes the tap order as ``s = [-1, 0, +1]``, so
    shift-vector index 0 (logit ``M + 2``) is offset ``-1``, index 1 (logit
    ``M + 3``) is offset ``0`` and index 2 (logit ``M + 4``) is offset ``+1``.

    The probe neutralizes every stage that could wash the impulse out:

    * ``g`` is driven to ~0 (logit ``-30`` -> ``sigmoid`` ~9.4e-14), so
      ``w_g == prev_weights == e_0`` and content addressing cannot contribute.
    * ``gamma`` is driven to ~1 (logit ``-30`` -> ``softplus`` ~0), so
      sharpening is the identity and cannot move the argmax.
    * The shift logits are ``+30`` on the tap under test and ``-30`` elsewhere,
      so the softmax puts essentially all mass on one tap.

    The mass threshold is ``0.9``, not ``1.0``: the softmax leaves ~1e-26 on the
    other taps, the gate leaks ~1e-13 of ``w_c``, and step 4 renormalizes by a
    sum plus ``1e-8``, so the delivered mass is near but not exactly 1.0.

    ``memory_locations`` is 8 (the module-level ``MEM_LOC``), never 3 or
    smaller, where ``+1`` and ``-1`` would alias onto the same slot and the
    guard would be vacuous.
    """

    BIG = 30.0

    def _addressing(self, shift_index: int):
        """Run ``_calculate_head_addressing`` with a delta impulse at slot 0.

        Returns the final head weights as a numpy array of shape ``(1, N)``.
        """
        layer = _make()
        # Build so the layer's own weights exist; the addressing math itself
        # reads none of them, but the layer must not be in an unbuilt state.
        layer.build((1, 1, F))

        params = np.full((1, MEM_DIM + 6), 0.0, dtype="float32")
        params[0, MEM_DIM + 1] = -self.BIG          # g -> ~0
        params[0, MEM_DIM + 2:MEM_DIM + 5] = -self.BIG
        params[0, MEM_DIM + 2 + shift_index] = self.BIG
        params[0, MEM_DIM + 5] = -self.BIG          # gamma -> ~1

        prev_weights = np.zeros((1, MEM_LOC), dtype="float32")
        prev_weights[0, 0] = 1.0

        memory = np.zeros((1, MEM_LOC, MEM_DIM), dtype="float32")

        out = layer._calculate_head_addressing(
            keras.ops.convert_to_tensor(params),
            keras.ops.convert_to_tensor(prev_weights),
            keras.ops.convert_to_tensor(memory),
        )
        return keras.ops.convert_to_numpy(out)

    def test_shift_direction_positive_offset_moves_forward(self):
        """Offset +1 (index 2) must move a delta impulse from slot 0 to slot 1."""
        out = self._addressing(shift_index=2)

        assert int(np.argmax(out[0])) == 1, (
            "Shift +1 moved the impulse to slot "
            f"{int(np.argmax(out[0]))}, expected slot 1 "
            f"(Graves eq. 8); full output: {out[0]}"
        )
        assert out[0, 1] >= 0.9, (
            f"Shift +1 delivered only {out[0, 1]:.4f} mass to slot 1, "
            f"expected ~1.0; full output: {out[0]}"
        )

    def test_shift_direction_negative_offset_moves_backward(self):
        """Offset -1 (index 0) must move a delta impulse from slot 0 to slot N-1."""
        out = self._addressing(shift_index=0)
        last = MEM_LOC - 1

        assert int(np.argmax(out[0])) == last, (
            "Shift -1 moved the impulse to slot "
            f"{int(np.argmax(out[0]))}, expected slot {last} "
            f"(Graves eq. 8); full output: {out[0]}"
        )
        assert out[0, last] >= 0.9, (
            f"Shift -1 delivered only {out[0, last]:.4f} mass to slot {last}, "
            f"expected ~1.0; full output: {out[0]}"
        )

    def test_shift_direction_zero_offset_is_identity(self):
        """Offset 0 (index 1) must leave the delta impulse at slot 0.

        This is the CONTROL: it holds under both the mirrored and the correct
        tap ordering, so it proves the probe is not simply always-red and that
        the gate/sharpening neutralization actually works.
        """
        out = self._addressing(shift_index=1)

        assert int(np.argmax(out[0])) == 0, (
            "Shift 0 moved the impulse to slot "
            f"{int(np.argmax(out[0]))}, expected slot 0; full output: {out[0]}"
        )
        assert out[0, 0] >= 0.9, (
            f"Shift 0 delivered only {out[0, 0]:.4f} mass to slot 0, "
            f"expected ~1.0; full output: {out[0]}"
        )
