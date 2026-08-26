"""Direct unit tests for the MoE gating layers and FFNExpert.

These concrete layers are also exercised through the MixtureOfExperts composite
in ``test_layer.py``; this module adds direct construction / validation /
forward / serialization coverage (notably a CosineGating round-trip, which the
composite tests do not cover directly).

The gating layers return ``(expert_weights, expert_indices, aux_dict)`` — the
dict output makes a functional ``.keras`` model awkward, so serialization is
verified via a ``get_config`` -> ``from_config`` + weight-transfer round-trip.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.moe.gating import (
    LinearGating,
    CosineGating,
    SoftMoEGating,
    compute_auxiliary_loss,
    compute_z_loss,
    _mask_neg_inf,
    _min_temperature,
)
from dl_techniques.layers.moe.experts import FFNExpert
from dl_techniques.layers.moe.config import ExpertConfig, GatingConfig, MoEConfig
from dl_techniques.layers.moe.layer import MixtureOfExperts

B, D = 4, 16
NUM_EXPERTS = 4


def _f32(*shape):
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).standard_normal(shape).astype("float32")
    )


def _gating_weight_round_trip(layer, data):
    """Build, serialize via config + weight transfer, assert identical weights output."""
    w0, _, _ = layer(data)
    rebuilt = type(layer).from_config(layer.get_config())
    rebuilt(data)  # build the clone
    rebuilt.set_weights(layer.get_weights())
    w1, _, _ = rebuilt(data)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(w0), keras.ops.convert_to_numpy(w1),
        rtol=1e-5, atol=1e-5,
    )


class TestLinearGating:
    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            LinearGating(num_experts=NUM_EXPERTS, top_k=NUM_EXPERTS + 1)

    def test_forward_and_shape(self):
        layer = LinearGating(num_experts=NUM_EXPERTS, top_k=2, add_noise=False)
        weights, indices, aux = layer(_f32(B, D))
        assert tuple(weights.shape) == (B, NUM_EXPERTS)
        assert tuple(indices.shape) == (B, 2)
        w_shape, i_shape, _ = layer.compute_output_shape((B, D))
        assert w_shape == (B, NUM_EXPERTS) and i_shape == (B, 2)

    def test_serialization(self):
        _gating_weight_round_trip(
            LinearGating(num_experts=NUM_EXPERTS, top_k=2, add_noise=False), _f32(B, D)
        )


class TestCosineGating:
    def test_invalid_args(self):
        with pytest.raises(ValueError):
            CosineGating(num_experts=NUM_EXPERTS, embedding_dim=0)
        with pytest.raises(ValueError):
            CosineGating(num_experts=NUM_EXPERTS, temperature=0.0)

    def test_forward_and_shape(self):
        layer = CosineGating(num_experts=NUM_EXPERTS, embedding_dim=8, top_k=2)
        weights, indices, aux = layer(_f32(B, D))
        assert tuple(weights.shape) == (B, NUM_EXPERTS)
        assert tuple(indices.shape) == (B, 2)

    def test_serialization(self):
        _gating_weight_round_trip(
            CosineGating(num_experts=NUM_EXPERTS, embedding_dim=8, top_k=2,
                         learnable_temperature=True),
            _f32(B, D),
        )

    def test_get_config_round_trip(self):
        layer = CosineGating(num_experts=NUM_EXPERTS, embedding_dim=8, temperature=2.0)
        rebuilt = CosineGating.from_config(layer.get_config())
        assert rebuilt.embedding_dim == 8 and rebuilt.temperature == 2.0


class TestSoftMoEGating:
    def test_invalid_num_slots(self):
        with pytest.raises(ValueError):
            SoftMoEGating(num_experts=NUM_EXPERTS, num_slots=0)

    def test_forward(self):
        layer = SoftMoEGating(num_experts=NUM_EXPERTS, num_slots=2)
        out = layer(_f32(B, 5, D))  # SoftMoE needs a sequence dim
        assert out is not None

    def test_get_config_round_trip(self):
        layer = SoftMoEGating(num_experts=NUM_EXPERTS, num_slots=3)
        rebuilt = SoftMoEGating.from_config(layer.get_config())
        assert rebuilt.num_slots == 3


class TestFFNExpert:
    def _expert(self):
        return FFNExpert(ffn_config={"type": "mlp", "hidden_dim": 32, "output_dim": D})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError):
            FFNExpert(ffn_config={"hidden_dim": 32, "output_dim": D})

    def test_forward_and_shape(self):
        out = self._expert()(_f32(B, D))
        assert tuple(out.shape) == (B, D)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(D,))
        out = self._expert()(inp)
        model = keras.Model(inp, out)
        data = np.random.default_rng(0).standard_normal((B, D)).astype("float32")
        y0 = model(data)
        path = os.path.join(tmp_path, "expert.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1 = loaded(data)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


# ---------------------------------------------------------------------
# Mixed-precision numerics (plan-2026-08-26T100331-f3744602, step 2 / SC-5)
#
# Both classes below were RED against the pre-fix code, MEASURED in a scratch
# worktree at 62adc45d0 under an actual `mixed_float16` policy:
#   * compute_z_loss(logits in [-426.75, 435.0]) -> `inf` (float32 ref 70.707)
#   * CosineGating with temperature_param = 1e-6 -> gate_logits min/max
#     `-inf inf`, 63/80 non-finite logits, 80/80 non-finite expert_weights
# They use `mixed_float16_policy` (tests/test_layers/conftest.py) rather than the
# parametrized `dtype_policy`: an fp16 overflow is unobservable at float32, so
# the other two policies would only add tests that cannot fail.
# ---------------------------------------------------------------------


class TestZLossMixedPrecision:
    """`compute_z_loss` must not overflow to inf under `mixed_float16` (D-009)."""

    # The exact logit range measured to overflow at HEAD.
    LOGIT_LO, LOGIT_HI = -426.75, 435.0

    def _logits_f32(self):
        return np.linspace(
            self.LOGIT_LO, self.LOGIT_HI, 64 * 4, dtype=np.float32
        ).reshape(1, 4, 64)

    def test_large_logits_stay_finite_and_match_float32(self, mixed_float16_policy):
        logits_np = self._logits_f32()
        reference = float(
            compute_z_loss(keras.ops.convert_to_tensor(logits_np), z_loss_weight=1e-3)
        )
        assert np.isfinite(reference), "float32 reference must be finite"

        half = keras.ops.cast(keras.ops.convert_to_tensor(logits_np), "float16")
        assert "float16" in str(half.dtype)
        measured = float(compute_z_loss(half, z_loss_weight=1e-3))

        assert np.isfinite(measured), (
            f"z_loss overflowed in fp16: {measured} (float32 reference {reference})"
        )
        assert abs(measured - reference) <= 0.01 * abs(reference), (
            f"z_loss {measured} differs from the float32 reference {reference} by more than 1%"
        )

    def test_returns_float32_regardless_of_input_dtype(self, mixed_float16_policy):
        half = keras.ops.cast(
            keras.ops.convert_to_tensor(self._logits_f32()), "float16"
        )
        assert "float32" in str(compute_z_loss(half).dtype)

    def test_reaches_add_loss_finite_through_the_gating_layer(self, mixed_float16_policy):
        """The production path: LinearGating -> gate_logits -> compute_z_loss."""
        layer = LinearGating(num_experts=64, top_k=4, add_noise=False)
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(0).standard_normal((2, 8, 32)).astype("float32") * 90.0
        )
        _, _, aux = layer(x, training=True)
        assert "float16" in str(aux["gate_logits"].dtype), (
            "guard is vacuous unless the gate emits fp16 under mixed_float16"
        )
        z_loss = compute_z_loss(aux["gate_logits"], z_loss_weight=1e-3)
        assert np.isfinite(float(z_loss))
        # RED pre-fix: the value handed to `add_loss` was fp16, i.e. one logit
        # excursion away from `inf`. Finiteness alone is not the invariant.
        assert "float32" in str(z_loss.dtype)


class TestCosineGatingTemperatureFloor:
    """`CosineGating`'s temperature floor must be dtype-aware (D-008)."""

    def _layer_and_data(self):
        layer = CosineGating(
            num_experts=8, top_k=2, embedding_dim=16, learnable_temperature=True
        )
        data = keras.ops.convert_to_tensor(
            np.random.default_rng(2).standard_normal((2, 5, 32)).astype("float32")
        )
        return layer, data

    def test_min_temperature_is_dtype_dependent(self):
        assert _min_temperature("float16") == pytest.approx(1e-3)
        assert _min_temperature("bfloat16") == pytest.approx(1e-3)
        assert _min_temperature("float32") == pytest.approx(keras.backend.epsilon())
        # The invariant the floor exists to deliver: bounded logits stay an order
        # of magnitude inside the top-k mask sentinel.
        for dtype in ("float16", "bfloat16", "float32"):
            assert 1.0 / _min_temperature(dtype) <= abs(_mask_neg_inf(dtype)) / 10.0

    def test_tiny_temperature_yields_no_non_finite_values(self, mixed_float16_policy):
        layer, data = self._layer_and_data()
        layer(data)  # build
        layer.temperature_param.assign(1e-6)  # bypasses the constraint on purpose

        weights, _, aux = layer(data, training=False)
        logits = np.asarray(
            keras.ops.convert_to_numpy(aux["gate_logits"])
        ).astype("float32")
        weights = np.asarray(keras.ops.convert_to_numpy(weights)).astype("float32")

        assert int((~np.isfinite(logits)).sum()) == 0, (
            f"{int((~np.isfinite(logits)).sum())}/{logits.size} non-finite gate_logits"
        )
        assert int((~np.isfinite(weights)).sum()) == 0, (
            f"{int((~np.isfinite(weights)).sum())}/{weights.size} non-finite expert_weights"
        )

    def test_temperature_param_carries_a_min_value_constraint(self, mixed_float16_policy):
        layer, data = self._layer_and_data()
        layer(data)
        constraint = layer.temperature_param.constraint
        assert constraint is not None, "temperature_param has no constraint"
        assert float(constraint.min_value) == pytest.approx(
            _min_temperature(layer.compute_dtype)
        )
        clamped = float(
            keras.ops.convert_to_numpy(
                constraint(keras.ops.convert_to_tensor(-5.0, dtype="float32"))
            )
        )
        assert clamped >= _min_temperature(layer.compute_dtype)

    def test_optimizer_cannot_drive_temperature_below_the_floor(self):
        """An SGD step, not Adam — Adam rescales and can hide the raw update."""
        layer, data = self._layer_and_data()
        layer(data)
        floor = _min_temperature(layer.compute_dtype)
        layer.temperature_param.assign(floor * 2.0)

        var = layer.temperature_param
        optimizer = keras.optimizers.SGD(learning_rate=1.0)
        optimizer.build([var])
        # A large positive gradient pushes the temperature far negative.
        optimizer.apply_gradients([(keras.ops.convert_to_tensor(1e3), var)])
        assert float(var) >= floor, (
            f"temperature fell to {float(var)}, below the {floor} floor"
        )

    def test_constraint_survives_config_and_keras_round_trip(self, tmp_path):
        layer, data = self._layer_and_data()
        rebuilt = CosineGating.from_config(layer.get_config())
        rebuilt(data)
        assert rebuilt.temperature_param.constraint is not None
        assert float(rebuilt.temperature_param.constraint.min_value) == pytest.approx(
            _min_temperature(rebuilt.compute_dtype)
        )

        inputs = keras.Input(shape=(32,))
        weights, _, _ = CosineGating(
            num_experts=8, top_k=2, embedding_dim=16, learnable_temperature=True
        )(inputs)
        model = keras.Model(inputs, weights)
        path = os.path.join(tmp_path, "cosine_gating.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        reloaded_layer = [
            sub for sub in loaded.layers if isinstance(sub, CosineGating)
        ][0]
        assert reloaded_layer.temperature_param.constraint is not None, (
            "the min-value constraint did not survive .keras save/load"
        )
        assert float(
            reloaded_layer.temperature_param.constraint.min_value
        ) == pytest.approx(_min_temperature(reloaded_layer.compute_dtype))


class TestAuxiliaryLossMixedPrecision:
    """2c audit outcome: `compute_auxiliary_loss` does NOT share D-009's defect.

    Kept as a REGRESSION pin on the bound that made the float32 upcast
    unnecessary — ``sum_i(f_i * P_i) <= 1`` so the loss is bounded by
    ``num_experts`` — not as a bug guard.
    """

    @pytest.mark.parametrize("num_experts,num_tokens", [(64, 8), (512, 4096), (4096, 64)])
    def test_stays_finite_in_fp16_at_worst_case_imbalance(
        self, mixed_float16_policy, num_experts, num_tokens
    ):
        # Fully imbalanced routing: every token to expert 0 with probability 1.
        weights = np.zeros((1, num_tokens, num_experts), dtype=np.float32)
        weights[..., 0] = 1.0
        half = keras.ops.cast(keras.ops.convert_to_tensor(weights), "float16")
        full = keras.ops.convert_to_tensor(weights)

        measured = float(compute_auxiliary_loss(half, half, num_experts, 0.01))
        reference = float(compute_auxiliary_loss(full, full, num_experts, 0.01))
        assert np.isfinite(measured)
        assert measured == pytest.approx(reference, rel=1e-2)
        # The structural bound: aux <= aux_loss_weight * num_experts (plus fp16 rounding).
        assert measured <= 0.01 * num_experts * 1.001


# ---------------------------------------------------------------------


class TestAuxiliaryLossTopKScaling:
    """D3 / F-5: pin the documented ``top_k`` interaction so the docstring cannot rot.

    `compute_auxiliary_loss` uses the Switch Transformer formula, calibrated for
    ``top_k = 1``, and is DELIBERATELY not normalized (decisions.md D-017,
    anchored in `gating.py`). These tests exist so that a later "cleanup" that
    divides the loss by ``top_k`` fails loudly instead of silently rescaling the
    load-balancing regularizer for Qwen3 and Qwen3-Next.
    """

    AUX_WEIGHT = 0.01
    NUM_TOKENS = 4096

    @staticmethod
    def _balanced(num_experts, top_k, num_tokens):
        """Exactly balanced round-robin dispatch with exactly uniform gate probs."""
        probs = np.full((num_tokens, num_experts), 1.0 / num_experts, dtype="float32")
        weights = np.zeros((num_tokens, num_experts), dtype="float32")
        for t in range(num_tokens):
            for j in range(top_k):
                weights[t, (t * top_k + j) % num_experts] = 1.0 / top_k
        return weights, probs

    @pytest.mark.parametrize("num_experts", [4, 8, 16, 64])
    @pytest.mark.parametrize("top_k", [1, 2, 4])
    def test_balanced_floor_is_weight_times_top_k(self, num_experts, top_k):
        """The floor is ``aux_loss_weight * top_k``, independent of ``num_experts``."""
        weights, probs = self._balanced(num_experts, top_k, self.NUM_TOKENS)
        measured = float(
            compute_auxiliary_loss(
                keras.ops.convert_to_tensor(weights),
                keras.ops.convert_to_tensor(probs),
                num_experts,
                self.AUX_WEIGHT,
            )
        )
        assert measured == pytest.approx(self.AUX_WEIGHT * top_k, rel=1e-5)

    def test_the_floor_actually_moves_with_top_k(self):
        """Guard against a vacuous parametrization: k=1, 2, 4 must give 3 values.

        A normalized implementation would make every row of the table above equal,
        and each individual `approx` above would then be checking one number
        against itself. This asserts the spread the documentation claims.
        """
        floors = []
        for top_k in (1, 2, 4):
            weights, probs = self._balanced(8, top_k, self.NUM_TOKENS)
            floors.append(
                float(
                    compute_auxiliary_loss(
                        keras.ops.convert_to_tensor(weights),
                        keras.ops.convert_to_tensor(probs),
                        8,
                        self.AUX_WEIGHT,
                    )
                )
            )
        assert floors[1] == pytest.approx(2 * floors[0], rel=1e-5)
        assert floors[2] == pytest.approx(4 * floors[0], rel=1e-5)
        assert len(set(round(f, 6) for f in floors)) == 3

    @pytest.mark.parametrize("top_k", [1, 2, 4])
    def test_worst_case_is_weight_times_num_experts_regardless_of_top_k(self, top_k):
        """The ceiling does NOT move with ``top_k`` -- which is why the range shrinks."""
        num_experts = 8
        probs = np.zeros((self.NUM_TOKENS, num_experts), dtype="float32")
        probs[:, :top_k] = 1.0 / top_k
        weights = np.zeros((self.NUM_TOKENS, num_experts), dtype="float32")
        weights[:, :top_k] = 1.0 / top_k
        measured = float(
            compute_auxiliary_loss(
                keras.ops.convert_to_tensor(weights),
                keras.ops.convert_to_tensor(probs),
                num_experts,
                self.AUX_WEIGHT,
            )
        )
        assert measured == pytest.approx(self.AUX_WEIGHT * num_experts, rel=1e-5)

    def test_floor_is_reached_by_a_real_gating_layer(self):
        """The table is not an artefact of hand-built tensors."""
        keras.utils.set_random_seed(0)
        inputs = keras.ops.convert_to_tensor(
            np.random.randn(4, 512, 32).astype("float32")
        )
        for top_k in (1, 2, 4):
            gate = LinearGating(num_experts=8, top_k=top_k, add_noise=False)
            expert_weights, _, info = gate(inputs, training=False)
            measured = float(
                compute_auxiliary_loss(
                    expert_weights, info["raw_gate_probs"], 8, self.AUX_WEIGHT
                )
            )
            assert measured >= self.AUX_WEIGHT * top_k * (1 - 1e-5)
            assert measured <= self.AUX_WEIGHT * 8 * (1 + 1e-5)


# ---------------------------------------------------------------------
# SoftMoE dispatch / combine, against an independent numpy reference
# ---------------------------------------------------------------------


def _np_softmax(x, axis):
    """Numerically stable softmax, written here so the oracle shares no code
    with the implementation under test (``keras.ops.softmax``)."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def _softmoe_reference(x, kernel, bias, num_experts, num_slots):
    """Independent numpy implementation of Puigcerver et al. (2023) SoftMoE.

    Derived from the paper's definition, not from ``SoftMoEGating.call``:

    * ``phi = x W + b``, reshaped to ``[b, s, e, l]``.
    * dispatch ``D = softmax(phi)`` over the **token** axis; ``slot_{e,l} =
      sum_t D_{t,e,l} x_t``.
    * combine ``C = softmax(phi)`` over the flattened **(expert, slot)** axis,
      per token.
    * the per-token/per-expert routing weight is the slot marginal of ``C``.

    :param x: Input tokens, ``[batch, seq, hidden]``.
    :type x: numpy.ndarray
    :param kernel: ``phi_dense`` kernel, ``[hidden, experts * slots]``.
    :type kernel: numpy.ndarray
    :param bias: ``phi_dense`` bias, ``[experts * slots]``.
    :type bias: numpy.ndarray
    :param num_experts: Number of experts.
    :type num_experts: int
    :param num_slots: Slots per expert.
    :type num_slots: int
    :return: Dict of the five tensors ``SoftMoEGating`` exposes.
    :rtype: Dict[str, numpy.ndarray]
    """
    b, s, h = x.shape
    phi = (x.reshape(b * s, h) @ kernel + bias).reshape(b, s, num_experts, num_slots)

    dispatch = _np_softmax(phi, axis=1)
    combine = _np_softmax(
        phi.reshape(b, s, num_experts * num_slots), axis=-1
    ).reshape(b, s, num_experts, num_slots)

    slots = np.einsum('bsel,bsh->belh', dispatch, x)

    return {
        'phi_logits': phi,
        'dispatch_weights': dispatch,
        'combine_weights': combine,
        'soft_slots': slots,
        'expert_inputs': slots.reshape(b, num_experts, num_slots * h),
        'expert_weights': combine.sum(axis=-1),
        'raw_gate_probs': _np_softmax(phi, axis=2).mean(axis=-1),
    }


@pytest.mark.usefixtures("tf32_disabled")
class TestSoftMoEAgainstANumpyReference:
    """Value verification for SoftMoE's two-softmax dispatch/combine.

    Before this class the whole mechanism was covered by ``assert out is not
    None`` plus two shape/row-sum checks -- neither of which can see a
    transposed softmax axis, and both of which a swapped dispatch/combine pair
    passes unchanged.

    All dimensions are mutually distinct (batch 2, seq 5, hidden 6, experts 3,
    slots 4) so that no axis confusion can be masked by a coincidental match.
    """

    B, S, H, E, L = 2, 5, 6, 3, 4

    def _layer_and_inputs(self, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((self.B, self.S, self.H)).astype('float32')
        layer = SoftMoEGating(num_experts=self.E, num_slots=self.L)
        layer(keras.ops.convert_to_tensor(x))  # build
        kernel, bias = [
            keras.ops.convert_to_numpy(w) for w in layer.phi_dense.weights
        ]
        return layer, x, _softmoe_reference(x, kernel, bias, self.E, self.L)

    def test_the_reference_dimensions_are_mutually_distinct(self):
        """Guard the guard: equal axes would hide a transposition."""
        assert len({self.B, self.S, self.H, self.E, self.L}) == 5

    @pytest.mark.parametrize(
        "name",
        ['dispatch_weights', 'combine_weights', 'soft_slots', 'expert_inputs',
         'raw_gate_probs'],
    )
    def test_aux_tensor_matches_the_reference(self, name):
        layer, x, ref = self._layer_and_inputs()
        _, _, aux = layer(keras.ops.convert_to_tensor(x))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(aux[name]), ref[name],
            rtol=1e-5, atol=1e-6,
        )

    def test_expert_weights_match_the_reference(self):
        layer, x, ref = self._layer_and_inputs()
        weights, _, _ = layer(keras.ops.convert_to_tensor(x))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(weights), ref['expert_weights'],
            rtol=1e-5, atol=1e-6,
        )

    def test_the_two_softmaxes_normalize_over_different_axes(self):
        """The distinguishing property of SoftMoE: dispatch sums to 1 over the
        token axis, combine sums to 1 over the (expert, slot) axes. A single
        softmax reused for both cannot satisfy both."""
        layer, x, ref = self._layer_and_inputs()
        _, _, aux = layer(keras.ops.convert_to_tensor(x))
        dispatch = keras.ops.convert_to_numpy(aux['dispatch_weights'])
        combine = keras.ops.convert_to_numpy(aux['combine_weights'])
        np.testing.assert_allclose(
            dispatch.sum(axis=1), np.ones((self.B, self.E, self.L)), atol=1e-5
        )
        np.testing.assert_allclose(
            combine.sum(axis=(2, 3)), np.ones((self.B, self.S)), atol=1e-5
        )
        assert not np.allclose(dispatch, combine, atol=1e-4)

    def test_a_transposed_dispatch_softmax_would_disagree(self):
        """Prove the reference is an oracle and not a tautology: recomputing it
        with the dispatch softmax taken over the expert axis instead of the
        token axis must move the numbers well outside the assertion tolerance."""
        layer, x, ref = self._layer_and_inputs()
        phi = ref['phi_logits']
        mutant_slots = np.einsum('bsel,bsh->belh', _np_softmax(phi, axis=2), x)
        assert np.max(np.abs(mutant_slots - ref['soft_slots'])) > 1e-2

    def test_swapping_dispatch_and_combine_would_disagree(self):
        """The other half of the same argument, for the combine weights."""
        layer, x, ref = self._layer_and_inputs()
        assert np.max(
            np.abs(ref['combine_weights'] - ref['dispatch_weights'])
        ) > 1e-2

    def test_a_second_seed_agrees_too(self):
        """One agreement could be a fixed point; two independent draws is not."""
        for seed in (1, 2, 3):
            layer, x, ref = self._layer_and_inputs(seed=seed)
            _, _, aux = layer(keras.ops.convert_to_tensor(x))
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(aux['soft_slots']), ref['soft_slots'],
                rtol=1e-5, atol=1e-6,
            )


@pytest.mark.usefixtures("tf32_disabled")
class TestSoftMoELayerCombineAgainstTheReference:
    """The other half of SoftMoE lives in ``MixtureOfExperts._process_softmoe``:
    it runs each expert on its slots and combines the results back to token
    positions. That combine was equally unverified.

    The reference below treats each ``FFNExpert`` as a black box -- it
    reimplements the slot construction, the per-expert slot batching and the
    combine contraction, which is where an axis error would live.
    """

    B, S, H, E, L, OUT = 2, 5, 6, 3, 4, 7

    def _build(self, seed=0):
        moe_config = MoEConfig(
            num_experts=self.E,
            expert_config=ExpertConfig(
                ffn_config={'type': 'mlp', 'hidden_dim': 9, 'output_dim': self.OUT}
            ),
            gating_config=GatingConfig(gating_type='softmoe', num_slots=self.L),
            jitter_noise=0.0,
        )
        layer = MixtureOfExperts(config=moe_config)
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((self.B, self.S, self.H)).astype('float32')
        layer(keras.ops.convert_to_tensor(x))  # build
        return layer, x

    def _reference_output(self, layer, x):
        kernel, bias = [
            keras.ops.convert_to_numpy(w)
            for w in layer.gating_network.phi_dense.weights
        ]
        ref = _softmoe_reference(x, kernel, bias, self.E, self.L)
        slots = ref['soft_slots']  # [b, e, l, h]

        expert_out = np.stack(
            [
                keras.ops.convert_to_numpy(
                    layer.experts[e](
                        keras.ops.convert_to_tensor(
                            slots[:, e].reshape(self.B * self.L, self.H)
                        )
                    )
                ).reshape(self.B, self.L, self.OUT)
                for e in range(self.E)
            ],
            axis=1,
        )  # [b, e, l, out]

        return np.einsum('bsel,belo->bso', ref['combine_weights'], expert_out)

    def test_forward_output_matches_the_reference(self):
        layer, x = self._build()
        actual = keras.ops.convert_to_numpy(
            layer(keras.ops.convert_to_tensor(x), training=False)
        )
        np.testing.assert_allclose(
            actual, self._reference_output(layer, x), rtol=1e-4, atol=1e-5
        )

    def test_output_shape_is_the_expert_width_not_the_input_width(self):
        layer, x = self._build()
        out = layer(keras.ops.convert_to_tensor(x), training=False)
        assert tuple(out.shape) == (self.B, self.S, self.OUT)
        assert self.OUT != self.H  # the assertion above would be vacuous otherwise

    def test_the_reference_disagrees_with_a_transposed_combine(self):
        """Combining with the *dispatch* weights instead of the combine weights
        -- the shape-preserving confusion of the two softmaxes, and the exact
        error the old smoke test could not see -- must produce a materially
        different output. That is what gives the equality test above teeth."""
        layer, x = self._build()
        kernel, bias = [
            keras.ops.convert_to_numpy(w)
            for w in layer.gating_network.phi_dense.weights
        ]
        ref = _softmoe_reference(x, kernel, bias, self.E, self.L)
        slots = ref['soft_slots']
        expert_out = np.stack(
            [
                keras.ops.convert_to_numpy(
                    layer.experts[e](
                        keras.ops.convert_to_tensor(
                            slots[:, e].reshape(self.B * self.L, self.H)
                        )
                    )
                ).reshape(self.B, self.L, self.OUT)
                for e in range(self.E)
            ],
            axis=1,
        )
        # Mutant: use the DISPATCH weights (wrong softmax) to combine.
        mutant = np.einsum('bsel,belo->bso', ref['dispatch_weights'], expert_out)
        good = np.einsum('bsel,belo->bso', ref['combine_weights'], expert_out)
        assert np.max(np.abs(mutant - good)) > 1e-3

    def test_a_second_seed_agrees_too(self):
        for seed in (4, 5):
            layer, x = self._build(seed=seed)
            actual = keras.ops.convert_to_numpy(
                layer(keras.ops.convert_to_tensor(x), training=False)
            )
            np.testing.assert_allclose(
                actual, self._reference_output(layer, x), rtol=1e-4, atol=1e-5
            )

# ---------------------------------------------------------------------
