"""Plan step 6: cross-camera synthetic code-path proof (Problem Statement
invariant 5, Success Criterion 6).

**What this test proves, precisely.** OmniPoint's architecture never branches
on camera type -- the only place camera geometry enters is the
`intrinsics_ray_map` conditioning input (`conditioning.py`'s
`ConditioningInputEncoder`, D-013). This test constructs a synthetic pinhole
RGB+depth pair, ACTUALLY REPROJECTS it (a real per-pixel nearest-neighbor
resample, not a stand-in) through `camera_models.py`'s fisheye and
equirectangular projection functions to produce two synthetic non-pinhole
images, then runs the full `OmniPoint` model (`enable_conditioning=True`) on
all three images (pinhole original, fisheye-reprojected, equirect-reprojected)
-- each fed the camera-matched `intrinsics_ray_map` for its own camera model
-- and asserts every output is finite, every predicted ray is unit-norm, and
every predicted distance is strictly positive.

**Why full reprojection, not the simplified ray-map-only framing.** The
Step-6 dispatch brief offered a lighter fallback: skip image reprojection
entirely and only feed a fisheye/equirectangular ray map as the conditioning
input, on a batch that is otherwise a random or pinhole image. That framing
would prove strictly less: it would never exercise the two inverse-projection
functions (`fisheye_unproject_points`, `equirectangular_unproject_points`) at
all, and would leave the RGB input untouched by any actual camera geometry --
i.e. it would show the model tolerates AN ARBITRARY ray map, not that a real
fisheye/equirectangular RE-CAPTURE of the same physical scene the pinhole
camera saw produces a sane forward pass. Success Criterion 6 says "run on a
synthetically-fisheye-reprojected and a synthetically-equirectangular-
reprojected pinhole+depth pair" verbatim -- that is the full-reprojection
claim, so this test builds it for real rather than substituting the weaker
framing.

**What this test does NOT prove** (do not read more into it than this):
    - No trained/generalizing behavior is claimed -- the model is freshly,
      randomly initialized (no pretrained weights exist anywhere for this
      package, D-001/`from_variant`'s own `NotImplementedError` contract).
      "Finite, unit-norm, positive-distance" is a CODE-PATH / CONSTRUCTION
      invariant (enforced by `RayPointCombinator`, Step 2), not a claim that
      the predicted geometry is metrically meaningful for a random-weight
      network.
    - The reprojected fisheye/equirectangular images are not photorealistic
      renders -- they are a real nearest-neighbor resample of the SAME
      camera-space 3D ray directions the pinhole camera observed, through
      each target camera model's own true inverse-projection formula. Pixels
      the resample does not cover are left at zero; this is a valid,
      if incomplete, synthetic image, not a real fisheye/360 photograph.

Reprojection method (all camera-geometry math from `utils/camera_models.py`,
none re-derived here):
    1. Generate the pinhole camera's own per-pixel unit ray map
       (`pinhole_ray_map`). Since ray DIRECTION does not depend on depth, this
       ray map already tells us, for every pinhole pixel, the exact 3D
       camera-space direction a co-located fisheye/equirectangular sensor
       would need to look in to see the same scene point.
    2. Feed those SAME ray directions into `fisheye_unproject_points` /
       `equirectangular_unproject_points` to get where each pinhole pixel's
       content lands on a same-resolution fisheye / equirectangular sensor.
    3. Nearest-neighbor-scatter the pinhole RGB into that target pixel
       coordinate (last-write-wins on collisions; uncovered target pixels
       stay zero).
"""

from typing import Tuple

import numpy as np
import keras

from dl_techniques.models.vision.omnipoint import OmniPoint
from dl_techniques.utils.camera_models import (
    pinhole_ray_map,
    fisheye_ray_map,
    fisheye_unproject_points,
    equirectangular_ray_map,
    equirectangular_unproject_points,
)

# Matches test_conditioning.py's convention: 56 % 14 == 0 -> a 4x4 patch grid
# on ViT-Base/14, small enough to run this test's 3 forward passes quickly.
_H, _W = 56, 56
_IMAGE_SHAPE = (_H, _W, 3)

# Pinhole intrinsics for the synthetic source camera: a moderate FOV so the
# resulting ray directions are well inside every camera model's valid range
# (comfortably below `MAX_INCIDENCE_ANGLE_RAD`).
_FX = _FY = 45.0
_CX = _CY = _W / 2.0

# Fisheye focal length: chosen so the pinhole's own (moderate-FOV) ray cone
# maps to a fisheye sensor region of comparable size, not degenerately small
# or clipped entirely off-canvas.
_FISHEYE_F = 40.0


def _make_synthetic_pinhole_pair() -> Tuple[np.ndarray, np.ndarray]:
    """Build a synthetic pinhole RGB image + positive depth map.

    :return: ``(rgb, depth)``, each ``(H, W, ...)`` numpy arrays -- ``rgb`` is
        ``(H, W, 3)`` in ``[0, 1]``, ``depth`` is ``(H, W)`` and strictly
        positive everywhere.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    rows, cols = np.meshgrid(np.arange(_H), np.arange(_W), indexing="ij")
    # A per-pixel-unique checkerboard-plus-gradient pattern: distinctive
    # enough that a nearest-neighbor scatter produces a visibly non-trivial
    # (not all-zero, not all-identical) reprojected image.
    checker = ((rows // 7 + cols // 7) % 2).astype("float32")
    rgb = np.stack(
        [
            0.5 * checker + 0.5 * (rows / _H),
            0.5 * (1.0 - checker) + 0.5 * (cols / _W),
            checker,
        ],
        axis=-1,
    ).astype("float32")
    # Strictly positive, per-pixel-varying depth (never degenerate/constant).
    depth = (2.0 + 0.5 * np.sin(rows / 4.0) + 0.5 * np.cos(cols / 4.0) + 2.0).astype(
        "float32"
    )
    assert (depth > 0.0).all()
    return rgb, depth


def _scatter_reproject(
        rgb: np.ndarray,
        dst_uv: np.ndarray,
        dst_height: int,
        dst_width: int,
) -> np.ndarray:
    """Nearest-neighbor-scatter ``rgb`` into a ``(dst_height, dst_width, 3)``
    canvas at the target pixel coordinates ``dst_uv``.

    :param rgb: Source image, ``(H, W, 3)``.
    :type rgb: np.ndarray
    :param dst_uv: Target ``(u, v)`` pixel coordinates for every source
        pixel, ``(H, W, 2)``.
    :type dst_uv: np.ndarray
    :param dst_height: Target canvas height.
    :type dst_height: int
    :param dst_width: Target canvas width.
    :type dst_width: int
    :return: Reprojected image, ``(dst_height, dst_width, 3)``, zero where no
        source pixel landed. Collisions resolve last-write-wins (row-major
        source iteration order) -- fine for this synthetic proof, which only
        needs A valid resample, not a specific one.
    :rtype: np.ndarray
    """
    canvas = np.zeros((dst_height, dst_width, 3), dtype="float32")
    u = np.round(dst_uv[..., 0]).astype("int64")
    v = np.round(dst_uv[..., 1]).astype("int64")
    valid = (u >= 0) & (u < dst_width) & (v >= 0) & (v < dst_height)
    canvas[v[valid], u[valid]] = rgb[valid]
    return canvas


def _make_fisheye_reprojected_image(rgb: np.ndarray, pinhole_rays: np.ndarray) -> np.ndarray:
    """Reproject ``rgb`` (seen by the pinhole camera) onto a fisheye sensor.

    :param rgb: Pinhole RGB image, ``(H, W, 3)``.
    :type rgb: np.ndarray
    :param pinhole_rays: The pinhole camera's own unit ray directions,
        ``(H, W, 3)`` -- the same 3D directions a co-located fisheye sensor
        would need to look along to see the same scene points.
    :type pinhole_rays: np.ndarray
    :return: Synthetic fisheye RGB image, ``(H, W, 3)``.
    :rtype: np.ndarray
    """
    dst_uv = keras.ops.convert_to_numpy(
        fisheye_unproject_points(pinhole_rays, f=_FISHEYE_F, cx=_CX, cy=_CY)
    )
    return _scatter_reproject(rgb, dst_uv, _H, _W)


def _make_equirectangular_reprojected_image(
        rgb: np.ndarray, pinhole_rays: np.ndarray
) -> np.ndarray:
    """Reproject ``rgb`` (seen by the pinhole camera) onto an equirectangular sensor.

    :param rgb: Pinhole RGB image, ``(H, W, 3)``.
    :type rgb: np.ndarray
    :param pinhole_rays: The pinhole camera's own unit ray directions,
        ``(H, W, 3)``.
    :type pinhole_rays: np.ndarray
    :return: Synthetic equirectangular RGB image, ``(H, W, 3)``.
    :rtype: np.ndarray
    """
    dst_uv = keras.ops.convert_to_numpy(
        equirectangular_unproject_points(pinhole_rays, height=_H, width=_W)
    )
    return _scatter_reproject(rgb, dst_uv, _H, _W)


def _make_model() -> OmniPoint:
    return OmniPoint.from_variant(
        "omnipoint_base",
        image_shape=_IMAGE_SHAPE,
        enable_conditioning=True,
        intrinsics_channels=2,
        depth_channels=2,
        conditioning_hidden_channels=4,
    )


def _assert_camera_agnostic_invariants(outputs) -> None:
    """Assert Problem Statement invariants 1-2 plus overall finiteness.

    :param outputs: The model's 5-tuple ``(ray, distance, point, mask_logit,
        scale)``.
    """
    ray, distance, point, mask_logit, scale = outputs
    for tensor in outputs:
        arr = keras.ops.convert_to_numpy(tensor)
        assert np.isfinite(arr).all(), "non-finite value in model output"

    ray_np = keras.ops.convert_to_numpy(ray)
    ray_norm = np.linalg.norm(ray_np, axis=-1)
    np.testing.assert_allclose(ray_norm, 1.0, atol=1e-4, rtol=0.0)

    distance_np = keras.ops.convert_to_numpy(distance)
    assert (distance_np > 0.0).all()


class TestCrossCameraSyntheticCodePathProof:
    """Success Criterion 6 / Problem Statement invariant 5."""

    def test_pinhole_fisheye_equirect_all_produce_valid_output(self):
        rgb, depth = _make_synthetic_pinhole_pair()
        pinhole_rays_kt = pinhole_ray_map(_FX, _FY, _CX, _CY, _H, _W)
        pinhole_rays = keras.ops.convert_to_numpy(pinhole_rays_kt)

        fisheye_rgb = _make_fisheye_reprojected_image(rgb, pinhole_rays)
        equirect_rgb = _make_equirectangular_reprojected_image(rgb, pinhole_rays)

        # Sanity on the reprojection itself, before trusting it as a model
        # input: each synthetic image must be a genuinely different, non-
        # trivial (not all-zero) resample -- otherwise this test would be
        # silently exercising the same input three times under three names.
        assert fisheye_rgb.sum() > 0.0
        assert equirect_rgb.sum() > 0.0
        assert not np.allclose(fisheye_rgb, rgb)
        assert not np.allclose(equirect_rgb, rgb)
        assert not np.allclose(fisheye_rgb, equirect_rgb)

        model = _make_model()

        fisheye_ray_map_kt = fisheye_ray_map(_FISHEYE_F, _CX, _CY, _H, _W)
        equirect_ray_map_kt = equirectangular_ray_map(_H, _W)

        cases = {
            "pinhole": (rgb, pinhole_rays_kt),
            "fisheye": (fisheye_rgb, fisheye_ray_map_kt),
            "equirectangular": (equirect_rgb, equirect_ray_map_kt),
        }

        for camera_name, (image, ray_map) in cases.items():
            image_batch = image[None, ...].astype("float32")
            ray_map_batch = keras.ops.expand_dims(ray_map, axis=0)
            outputs = model(image_batch, intrinsics_ray_map=ray_map_batch)
            _assert_camera_agnostic_invariants(outputs)
