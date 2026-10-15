"""Surface normals are unit 3-vectors and optical flow is a signed 2-vector.

Guard for plan-2026-10-14T062616-81938e81 step 11.1 (review semantic item,
plan Step 12 Part E taken over here, decision D-051). Measured at
``9343b0f87``: ``create_vision_head`` built both tasks as a
``DepthEstimationHead`` with 3 or 2 output channels, so each "normal" and each
"flow vector" was a sigmoid squeezed into the depth range and exponentiated:
every component in ``[0.1, 100]`` (measured min 0.47 normals, 0.50 flow on a
random input), never negative, never unit length, under the keys
``'depth'``/``'confidence'``.

The references, read from the official code at this step:

* normals, linear output then per-pixel L2 normalization: Bae et al. 2021,
  ``surface_normal_uncertainty`` ``norm_normalize`` (``norm_x / sqrt(x^2 +
  y^2 + z^2)``), and DSINE (Bae and Davison 2024) ``F.normalize(normal,
  dim=1)`` after a linear prediction head;
* flow, a linear 3x3 conv to 2 channels: RAFT ``FlowHead``
  (``conv2(relu(conv1(x)))``, ``conv2 = nn.Conv2d(hidden_dim, 2, 3,
  padding=1)``) and PWC-Net ``predict_flow`` (``nn.Conv2d(in_planes, 2, 3, 1,
  1, bias=True)``).

The value guards: every normal has unit L2 norm in every dtype policy; scaling
the normals' output conv by ``4`` leaves them unchanged and by ``-1`` flips
them (linear, then normalized); scaling the flow's output conv by ``1024``
scales the flow by exactly ``1024`` (linear, no range map), and the flow takes
both signs. The factors are powers of two (and -1) so the scaling itself is
exact in floating point.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.vision import (
    OpticalFlowHead,
    SurfaceNormalsHead,
    create_vision_head,
)

HIDDEN_DIM = 16
SPATIAL = (2, 6, 10, HIDDEN_DIM)


def _inputs() -> np.ndarray:
    return np.random.default_rng(21).normal(size=SPATIAL).astype("float32")


def _np(x) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(x))


def _scale_output_conv(head, factor: float) -> None:
    kernel, bias = head.output_conv.kernel, head.output_conv.bias
    kernel.assign(kernel * factor)
    bias.assign(bias * factor)


@pytest.mark.parametrize("cls", [SurfaceNormalsHead, OpticalFlowHead],
                         ids=lambda c: c.__name__)
def test_the_head_is_registered_under_its_package_key(cls, registration_contract) -> None:
    registration_contract(cls)


@pytest.mark.parametrize("task,key,channels", [("surface_normals", "normals", 3),
                                               ("optical_flow", "flow", 2)])
def test_the_head_returns_its_own_key_at_eight_times_the_input(task, key, channels) -> None:
    head = create_vision_head(task, hidden_dim=HIDDEN_DIM)
    out = head(_inputs())
    assert sorted(out) == [key]
    assert tuple(out[key].shape) == (2, 48, 80, channels)
    assert head.compute_output_shape(SPATIAL) == {key: (2, 48, 80, channels)}


def test_every_normal_has_unit_length_in_every_dtype_policy(dtype_policy) -> None:
    assert keras.mixed_precision.global_policy().name == dtype_policy
    keras.utils.set_random_seed(5)
    head = create_vision_head("surface_normals", hidden_dim=HIDDEN_DIM)
    normals = head(_inputs())["normals"]
    out_dtype = keras.backend.standardize_dtype(normals.dtype)
    assert out_dtype == keras.mixed_precision.global_policy().compute_dtype
    values = _np(normals).astype("float64")
    norms = np.linalg.norm(values, axis=-1)
    # The normalization (x * rsqrt(sum x^2), float32 or wider) carries a few
    # ulps of its own: one rounding per square, two adds, the rsqrt and the
    # product. Each of the 3 components is then rounded once to the output
    # dtype, a relative error of at most eps/2 of THAT dtype, so the norm moves
    # by at most eps/2 more. 4 * eps of the output dtype bounds both.
    bound = 4 * np.finfo(np.dtype(out_dtype)).eps
    assert np.all(np.abs(norms - 1.0) <= bound), (
        f"{dtype_policy}: max |norm - 1| = {np.max(np.abs(norms - 1.0)):.3e} "
        f"> {bound:.3e}")
    # Unit vectors in every direction, not a positive range map.
    assert values.min() < 0.0 < values.max()


def test_the_normals_are_the_normalized_linear_output(golden_reference_device) -> None:
    # Powers of two and -1 scale a float exactly, so the conv output scales
    # exactly and any difference would be the head's, not rounding's. On the
    # golden-reference device the conv runs one deterministic kernel; a GPU
    # conv measured 1-ulp call-to-call spreads here.
    x = _inputs()
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(5)
        head = create_vision_head("surface_normals", hidden_dim=HIDDEN_DIM)
        before = _np(head(x)["normals"])
        _scale_output_conv(head, 4.0)
        scaled = _np(head(x)["normals"])
        _scale_output_conv(head, -0.25)
        flipped = _np(head(x)["normals"])
    np.testing.assert_array_equal(scaled, before)
    np.testing.assert_array_equal(flipped, -before)


def test_the_flow_is_the_linear_output_and_takes_both_signs(golden_reference_device) -> None:
    x = _inputs()
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(5)
        head = create_vision_head("optical_flow", hidden_dim=HIDDEN_DIM)
        before = _np(head(x)["flow"])
        _scale_output_conv(head, 1024.0)  # a power of two: exact scaling
        after = _np(head(x)["flow"])
    assert before.min() < 0.0 < before.max()
    np.testing.assert_array_equal(after, 1024.0 * before)
    assert np.abs(after).max() > 100.0  # far outside any depth range map


@pytest.mark.parametrize("task", ["surface_normals", "optical_flow"])
def test_the_head_survives_a_keras_round_trip(task, tmp_path, golden_reference_device) -> None:
    x = _inputs()
    with keras.device(golden_reference_device):
        keras.utils.set_random_seed(3)
        inputs = keras.Input(SPATIAL[1:])
        model = keras.Model(inputs, create_vision_head(task, hidden_dim=HIDDEN_DIM)(inputs))
        before = model.predict(x, verbose=0)
        model.save(tmp_path / f"{task}.keras")
        after = keras.models.load_model(tmp_path / f"{task}.keras").predict(x, verbose=0)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key])
