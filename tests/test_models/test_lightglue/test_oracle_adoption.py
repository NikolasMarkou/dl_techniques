"""Shared-oracle adoption for ``models/vision/keypoints/lightglue``.

Gradient flow, dead-component liveness, knob sensitivity, the smoke contract, a
``mixed_float16`` arm, and the stop-gradient guard on the input descriptors. Every guard
is shown able to fail (RED) in the same file.

WHICH LOSS DRIVES WHICH WEIGHT. ``MatchTokenConfidence`` puts ``stop_gradient`` on its INPUT
only, so its own ``token_0`` weights get a gradient exactly when the loss uses
``token_confidences0/1``. ``default_loss`` (mean of squares over every float output) does,
so the "every trainable weight" assertions below use it. A loss over ``log_assignments`` alone
leaves the ``token_0`` weights disconnected; that is asserted explicitly in
:class:`TestWhichLossDrivesWhichWeight`, so the property is documented rather than discovered.
The descriptors themselves never receive a gradient: the model detaches them before
``input_proj`` (the reference does ``descriptors.detach()``).

Weight count (derived, not measured): per layer a self block has 10 variables (Wqkv, out_proj,
ffn_0, ffn_1 with gamma and beta, ffn_3), a cross block 12 (to_qk, to_v, to_out, the same FFN),
an assignment 4 (matchability, final_proj); each of the ``L - 1`` confidence heads 2; the
posenc kernel 1; ``input_proj`` 2 when ``input_dim != descriptor_dim``.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.models.vision.keypoints.lightglue import model as lightglue_model
from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue

from ..gradient_flow_oracle import (
    assert_gradients_reach_every_trainable_weight,
    default_loss,
    gradient_report,
    stop_all_gradients,
)
from ..knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
)
from ..precision_arm_oracle import precision_policy
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
    broken_forward,
)
from ..test_sam.dead_component_oracle import (
    NO_GRADIENTS_MESSAGE,
    component_response,
    fit_one_step_moved_variables,
    no_op_kill,
    outputs_stop_gradient,
    zeroed_variables,
)
from .weight_loading import random_inputs

D, H, L = 16, 2, 3
INPUT_DIM = 24
M, N = 10, 8
BUILD_SEED = 0

# derived, see the module docstring
WEIGHTS = 3 + L * (10 + 12 + 4) + (L - 1) * 2


def _data(batch=2, m=M, n=N, input_dim=INPUT_DIM, seed=0, extra=False) -> Dict[str, Any]:
    return random_inputs(np.random.RandomState(seed), batch, m, n, input_dim, extra=extra)


def _lightglue(**o) -> LightGlue:
    kwargs: Dict[str, Any] = dict(input_dim=INPUT_DIM, descriptor_dim=D, num_layers=L, num_heads=H)
    kwargs.update(o)
    return LightGlue(**kwargs)


def _built(build_fn=_lightglue, seed=BUILD_SEED, extra=False, input_dim=INPUT_DIM) -> LightGlue:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_data(1, input_dim=input_dim, extra=extra), training=False)
    return model


def _as_tensors(data):
    return {k: tf.constant(v) for k, v in data.items()}


class TestLightGlueGradientFlow:

    def test_gradients_reach_every_trainable_weight(self):
        model = _built()
        report = assert_gradients_reach_every_trainable_weight(model, _as_tensors(_data()))
        assert len(report) == WEIGHTS == len(model.trainable_weights)

    def test_the_named_components_are_among_the_live_weights(self):
        report = gradient_report(_built(), _as_tensors(_data()))
        for needle in ("input_proj/kernel", "posenc/kernel", "log_assignment_0/final_proj/kernel",
                       "log_assignment_2/matchability/kernel", "token_confidence_0/token_0/kernel",
                       "self_attn_0/Wqkv/kernel", "cross_attn_2/to_qk/kernel"):
            path = next(p for p in report if p.endswith(needle))
            assert report[path] is not None and report[path] > 0.0, f"{needle} is dead"

    def test_the_gradient_assertion_can_fail(self):
        model = _built()
        with broken_forward(model, stop_all_gradients):
            with pytest.raises(AssertionError, match="received NO gradient"):
                assert_gradients_reach_every_trainable_weight(model, _as_tensors(_data()))


class TestWhichLossDrivesWhichWeight:

    def test_a_loss_over_log_assignments_alone_leaves_only_the_confidence_heads_disconnected(self):
        model = _built()
        report = gradient_report(
            model, _as_tensors(_data()), loss_fn=lambda out: keras.ops.mean(keras.ops.square(out["log_assignments"])))
        disconnected = sorted(p for p, v in report.items() if v is None)
        assert disconnected, "vacuous: nothing disconnected"
        assert all("token_confidence_" in p for p in disconnected), disconnected
        assert len(disconnected) == (L - 1) * 2
        live = [p for p, v in report.items() if v is not None]
        assert len(live) == WEIGHTS - (L - 1) * 2 and all(report[p] > 0.0 for p in live)

    def test_a_confidence_loss_drives_the_heads(self):
        model = _built()
        report = gradient_report(
            model, _as_tensors(_data()),
            loss_fn=lambda out: keras.ops.mean(out["token_confidences0"]) + keras.ops.mean(out["token_confidences1"]))
        heads = [p for p in report if "token_confidence_" in p]
        assert len(heads) == (L - 1) * 2
        assert all(report[p] is not None and report[p] > 0.0 for p in heads)


class TestDescriptorsReceiveNoGradient:
    """The reference detaches descriptors; the model must too."""

    @staticmethod
    def _descriptor_gradient(model):
        data = _as_tensors(_data())
        with tf.GradientTape() as tape:
            tape.watch(data["descriptors0"])
            tape.watch(data["descriptors1"])
            loss = default_loss(model(data, training=True))
        grads = tape.gradient(loss, [data["descriptors0"], data["descriptors1"]],
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return [np.asarray(keras.ops.convert_to_numpy(g)) for g in grads]

    def test_the_gradient_into_the_descriptors_is_exactly_zero(self):
        for g in self._descriptor_gradient(_built()):
            assert np.all(g == 0.0)

    def test_the_guard_is_red_without_the_stop_gradient(self, monkeypatch):
        model = _built()
        monkeypatch.setattr(lightglue_model, "_detach_descriptors", lambda x: x)
        grads = self._descriptor_gradient(model)
        assert all(np.abs(g).max() > 0.0 for g in grads), "the guard could not tell the difference"

    def test_the_guard_is_red_without_the_stop_gradient_for_identity_projection(self, monkeypatch):
        """Same, for input_dim == descriptor_dim where no Dense follows the detach."""
        model = _built(lambda: _lightglue(input_dim=D), input_dim=D)
        data = _as_tensors(_data(input_dim=D))

        def grad_max():
            with tf.GradientTape() as tape:
                tape.watch(data["descriptors0"])
                loss = default_loss(model(data, training=True))
            g = tape.gradient(loss, data["descriptors0"], unconnected_gradients=tf.UnconnectedGradients.ZERO)
            return float(np.abs(keras.ops.convert_to_numpy(g)).max())

        assert grad_max() == 0.0
        monkeypatch.setattr(lightglue_model, "_detach_descriptors", lambda x: x)
        assert grad_max() > 0.0


def _compile_for_fit(model):
    # stock fit, no custom train_step; only the differentiable outputs carry a loss
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={"log_assignments": "mse", "token_confidences0": "mse", "token_confidences1": "mse"},
        jit_compile=False,
    )


def _fit_targets(model, data):
    out = model(data)
    return {k: np.zeros(tuple(out[k].shape), "float32")
            for k in ("log_assignments", "token_confidences0", "token_confidences1")}


class TestLightGlueTrainsUnderStockFit:

    def test_one_fit_step_moves_every_trainable_variable(self):
        model = _built()
        _compile_for_fit(model)
        data = _data(4)
        report = fit_one_step_moved_variables(model, data, _fit_targets(model, data), batch_size=4)
        assert report.unmoved == (), report.summary()
        assert report.total == WEIGHTS and len(report.moved) == WEIGHTS
        assert np.isfinite(report.final_loss)

    def test_the_fit_step_probe_can_fail(self):
        model = _built()
        _compile_for_fit(model)
        data = _data(4)
        with outputs_stop_gradient(model):
            with pytest.raises(ValueError, match=NO_GRADIENTS_MESSAGE):
                fit_one_step_moved_variables(model, data, _fit_targets(model, data), batch_size=4)

    def test_padded_batches_train_with_finite_loss(self):
        model = _built()
        _compile_for_fit(model)
        data = _data(4)
        data["mask0"] = (np.arange(M)[None, :] < 7).repeat(4, 0).astype("float32")
        data["mask1"] = (np.arange(N)[None, :] < 5).repeat(4, 0).astype("float32")
        report = fit_one_step_moved_variables(model, data, _fit_targets(model, data), batch_size=4)
        assert np.isfinite(report.final_loss) and report.unmoved == (), report.summary()


class TestLiveComponentsChangeTheOutput:
    """Positive liveness arms: destroying a component moves the output it feeds."""

    @staticmethod
    def _metric(model, key):
        data = _data(2, seed=3)
        return lambda: float(np.abs(keras.ops.convert_to_numpy(model(data)[key])).sum())

    def test_control_a_no_op_kill_does_not_move_the_metric(self):
        model = _built()
        r = component_response(self._metric(model, "log_assignments"), no_op_kill, name="control")
        assert not r.moved and r.delta == 0.0, r.summary()

    @pytest.mark.parametrize("name, key, pick", [
        ("input_proj", "log_assignments", lambda m: m.input_proj.weights),
        ("posenc", "log_assignments", lambda m: m.posenc.weights),
        ("self_attn_0", "log_assignments", lambda m: m.self_blocks[0].weights),
        ("cross_attn_1", "log_assignments", lambda m: m.cross_blocks[1].weights),
        ("log_assignment_2", "log_assignments", lambda m: m.assignments[2].weights),
        ("token_confidence_0", "token_confidences0", lambda m: m.confidences[0].weights),
    ])
    def test_zeroing_a_component_moves_its_output(self, name, key, pick):
        model = _built()
        r = component_response(self._metric(model, key), lambda: zeroed_variables(pick(model)),
                               name=name, atol=1e-6)
        assert r.moved, r.summary()


class TestLightGlueKnobSensitivity:

    @staticmethod
    def _forward_builders(builder, values, **kw):
        return {v: (lambda v=v: _built(lambda: builder(v), **kw)) for v in values}

    def test_num_layers_changes_the_parameterisation(self):
        assert_structural_knob_changes_weights(
            self._forward_builders(lambda v: _lightglue(num_layers=v), (1, 2, 3)), knob="num_layers")

    def test_descriptor_dim_changes_the_parameterisation(self):
        assert_structural_knob_changes_weights(
            self._forward_builders(lambda v: _lightglue(descriptor_dim=v), (16, 32, 48)), knob="descriptor_dim")

    def test_num_heads_changes_the_parameterisation(self):
        # d = 32 with 2 / 4 / 8 heads: head_dim 16 / 8 / 4, so the posenc kernel changes shape
        assert_structural_knob_changes_weights(
            self._forward_builders(lambda v: _lightglue(descriptor_dim=32, num_heads=v), (2, 4, 8)),
            knob="num_heads")

    def test_input_dim_changes_the_parameterisation(self):
        builders = {
            v: (lambda v=v: _built(lambda: _lightglue(input_dim=v), input_dim=v)) for v in (16, 24, 32)
        }
        assert_structural_knob_changes_weights(builders, knob="input_dim")

    def test_add_scale_ori_changes_the_parameterisation(self):
        builders = {
            v: (lambda v=v: _built(lambda: _lightglue(add_scale_ori=v), extra=True)) for v in (False, True)
        }
        assert_structural_knob_changes_weights(builders, knob="add_scale_ori")

    def test_gamma_reaches_the_forward_pass(self):
        """A VALUE knob: same shapes, different positional-encoding frequencies."""
        builders = {g: (lambda g=g: _lightglue(gamma=g)) for g in (1.0, 0.5)}
        deltas = assert_value_knob_changes_output(
            builders, _data(), knob="gamma", extract=lambda o: o["log_assignments"])
        assert all(d > 1e-4 for d in deltas.values()), deltas

    def test_filter_threshold_reaches_the_matches(self):
        builders = {t: (lambda t=t: _lightglue(filter_threshold=t)) for t in (0.0, 1.0)}
        deltas = assert_value_knob_changes_output(
            builders, _data(), knob="filter_threshold", extract=lambda o: o["matches0"])
        assert all(d >= 1 for d in deltas.values()), deltas

    def test_depth_and_width_confidence_are_inert_in_the_static_path(self):
        """Documented, not a defect: the static path never prunes or exits early. Step 5's
        ``match()`` is where these knobs get a behaviour test."""
        builders = {c: (lambda c=c: _lightglue(depth_confidence=c, width_confidence=c)) for c in (-1, 0.9)}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_value_knob_changes_output(
                builders, _data(), knob="depth_confidence", extract=lambda o: o["log_assignments"])

    def test_the_knob_assertions_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="num_layers")
        value_builders = {k: (lambda: _lightglue(gamma=1.0)) for k in ("a", "b")}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_value_knob_changes_output(
                value_builders, _data(), knob="gamma", extract=lambda o: o["log_assignments"])


class TestLightGlueSmokeContract:

    def test_the_forward_contract_rejects_a_broken_forward(self):
        model = _built()
        data = _data()

        def contract(out):
            assert isinstance(out, dict), f"LightGlue.call returns a dict, got {type(out)}"
            assert set(out) == {
                "log_assignments", "token_confidences0", "token_confidences1",
                "token_logits0", "token_logits1",
                "matches0", "matches1", "matching_scores0", "matching_scores1",
            }, f"unexpected key set {sorted(out)}"
            assert tuple(out["log_assignments"].shape) == (2, L, M + 1, N + 1), tuple(out["log_assignments"].shape)
            assert tuple(out["token_confidences0"].shape) == (2, L - 1, M)
            assert tuple(out["token_confidences1"].shape) == (2, L - 1, N)
            assert tuple(out["matches0"].shape) == (2, M) and tuple(out["matches1"].shape) == (2, N)
            for key, value in out.items():
                assert_finite(value)
            la = keras.ops.convert_to_numpy(out["log_assignments"])
            assert (la <= 1e-5).all(), "log probabilities above 0"
            assert not np.allclose(la, la[:, :1]), "every layer produced the same assignment"
            m0 = keras.ops.convert_to_numpy(out["matches0"])
            assert ((m0 >= -1) & (m0 < N)).all()

        rejections = assert_contract_rejects_a_broken_forward(model, data, contract)
        assert set(rejections) == {"collapse_to_scalar", "slice_leading_axis", "append_trailing_axis"}


class TestLightGlueMixedPrecision:

    def test_mixed_float16_forward_is_finite_with_float32_scores(self):
        with precision_policy("mixed_float16"):
            model = _built()
            out = model(_data(), training=False)
            assert_finite(out)
            assert out["log_assignments"].dtype == tf.float32
            assert out["matches0"].dtype == tf.int32

    def test_mixed_float16_gradients_reach_every_weight(self):
        with precision_policy("mixed_float16"):
            model = _built()
            report = gradient_report(model, _as_tensors(_data()))
        assert [p for p, v in report.items() if v is None] == []
        assert [p for p, v in report.items() if v is not None and not np.isfinite(v)] == []
        assert len(report) == WEIGHTS

    def test_keypoint_positions_are_not_rounded_to_float16(self):
        """Pixel 600.25 is not representable in float16 (spacing 0.5 there). The model turns
        autocast off so the normalisation sees the float32 coordinates."""
        with precision_policy("mixed_float16"):
            model = _built()
            kp = np.full((1, 2, 2), 600.25, "float32")
            seen = {}
            original = lightglue_model.normalize_keypoints

            def spy(keypoints, image_size, dtype=None):
                seen["kp"] = keras.ops.convert_to_numpy(keypoints)
                return original(keypoints, image_size, dtype)

            lightglue_model.normalize_keypoints = spy
            try:
                data = _data(1, 2, 2)
                data["keypoints0"] = kp
                data["keypoints1"] = kp
                data["image_size0"] = data["image_size1"] = np.array([[1200.0, 800.0]], "float32")
                model(data)
            finally:
                lightglue_model.normalize_keypoints = original
        assert seen["kp"].dtype == np.float32
        np.testing.assert_array_equal(seen["kp"], kp)

    def test_posenc_runs_on_float32_inputs_under_mixed_float16(self):
        with precision_policy("mixed_float16"):
            model = _built()
            assert model.posenc.kernel.dtype == "float32"
            table = model.posenc(np.random.RandomState(0).uniform(-1, 1, (1, 5, 2)).astype("float32"))
            assert table.dtype == tf.float32
