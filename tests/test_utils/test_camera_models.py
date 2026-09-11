"""Round-trip and edge-case tests for `dl_techniques.utils.camera_models`."""

import keras
import numpy as np
import pytest
from numpy.testing import assert_allclose

from dl_techniques.utils.camera_models import (
    MAX_INCIDENCE_ANGLE_RAD,
    equirectangular_ray_map,
    equirectangular_unproject_points,
    fisheye_ray_map,
    fisheye_unproject_points,
    pinhole_ray_map,
    pinhole_unproject_points,
)


def _to_numpy(x):
    return keras.ops.convert_to_numpy(x)


# ---------------------------------------------------------------------
# Pinhole
# ---------------------------------------------------------------------


class TestPinholeRoundTrip:
    """project (ray map) -> unproject -> re-project must recover the pixel grid."""

    @pytest.mark.parametrize(
        "fx,fy,cx,cy,height,width",
        [
            (500.0, 500.0, 320.0, 240.0, 480, 640),
            (300.0, 320.0, 128.0, 96.0, 192, 256),
            (900.5, 875.25, 64.0, 64.0, 128, 128),
        ],
    )
    def test_round_trip_recovers_pixel_grid(self, fx, fy, cx, cy, height, width):
        rays = pinhole_ray_map(fx, fy, cx, cy, height, width)
        pixels = pinhole_unproject_points(rays, fx, fy, cx, cy)

        rows = np.arange(height, dtype=np.float32)
        cols = np.arange(width, dtype=np.float32)
        expected_u = cols[None, :] + 0.5
        expected_v = rows[:, None] + 0.5
        expected_u = np.broadcast_to(expected_u, (height, width))
        expected_v = np.broadcast_to(expected_v, (height, width))

        pixels_np = _to_numpy(pixels)
        assert_allclose(pixels_np[..., 0], expected_u, rtol=0, atol=1e-4)
        assert_allclose(pixels_np[..., 1], expected_v, rtol=0, atol=1e-4)

    def test_rays_are_unit_norm_and_finite(self):
        rays = _to_numpy(pinhole_ray_map(500.0, 500.0, 320.0, 240.0, 60, 80))
        norms = np.linalg.norm(rays, axis=-1)
        assert np.isfinite(rays).all()
        assert_allclose(norms, 1.0, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------
# Fisheye (equidistant)
# ---------------------------------------------------------------------


class TestFisheyeRoundTrip:
    @pytest.mark.parametrize(
        "f,cx,cy,height,width",
        [
            (200.0, 320.0, 240.0, 480, 640),
            (150.0, 128.0, 96.0, 192, 256),
            (250.5, 64.0, 64.0, 128, 128),
        ],
    )
    def test_round_trip_recovers_pixel_grid(self, f, cx, cy, height, width):
        rays = fisheye_ray_map(f, cx, cy, height, width)
        pixels = fisheye_unproject_points(rays, f, cx, cy)

        rows = np.arange(height, dtype=np.float32)
        cols = np.arange(width, dtype=np.float32)
        expected_u = cols[None, :] + 0.5
        expected_v = rows[:, None] + 0.5
        expected_u = np.broadcast_to(expected_u, (height, width))
        expected_v = np.broadcast_to(expected_v, (height, width))

        pixels_np = _to_numpy(pixels)

        # Only pixels whose incidence angle stays below the clamp bound have
        # a bijective (non-lossy) round trip; pixels beyond the clamp are
        # covered separately by the edge-case test below.
        du = expected_u - cx
        dv = expected_v - cy
        r = np.sqrt(du**2 + dv**2)
        theta = r / f
        within_bounds = theta < (MAX_INCIDENCE_ANGLE_RAD - 1e-3)

        # float32 round trip through sin/cos/atan2/arccos accumulates more
        # error than the pinhole's purely-linear inverse; 1e-3 pixels is
        # still far tighter than any perceptible/practical difference.
        assert_allclose(
            pixels_np[..., 0][within_bounds],
            expected_u[within_bounds],
            rtol=0,
            atol=1e-3,
        )
        assert_allclose(
            pixels_np[..., 1][within_bounds],
            expected_v[within_bounds],
            rtol=0,
            atol=1e-3,
        )

    def test_rays_are_unit_norm_and_finite(self):
        rays = _to_numpy(fisheye_ray_map(200.0, 320.0, 240.0, 60, 80))
        norms = np.linalg.norm(rays, axis=-1)
        assert np.isfinite(rays).all()
        assert_allclose(norms, 1.0, rtol=0, atol=1e-5)

    def test_pixel_beyond_max_incidence_angle_stays_finite(self):
        """A pixel far outside the fisheye's valid FOV must clamp, not diverge."""
        height, width = 200, 200
        f, cx, cy = 50.0, 100.0, 100.0  # small f -> large images exceed max angle
        rays = _to_numpy(fisheye_ray_map(f, cx, cy, height, width))
        assert np.isfinite(rays).all()
        norms = np.linalg.norm(rays, axis=-1)
        assert_allclose(norms, 1.0, rtol=0, atol=1e-5)

        # The corner pixel is far beyond max incidence angle for this f.
        corner_ray = rays[0, 0]
        assert np.isfinite(corner_ray).all()

        pixels = _to_numpy(
            fisheye_unproject_points(
                keras.ops.convert_to_tensor(rays[None, 0, 0]), f, cx, cy
            )
        )
        assert np.isfinite(pixels).all()


# ---------------------------------------------------------------------
# Equirectangular
# ---------------------------------------------------------------------


class TestEquirectangularRoundTrip:
    @pytest.mark.parametrize(
        "height,width",
        [
            (256, 512),
            (128, 256),
            (64, 128),
        ],
    )
    def test_round_trip_recovers_pixel_grid(self, height, width):
        rays = equirectangular_ray_map(height, width)
        pixels = equirectangular_unproject_points(rays, height, width)

        rows = np.arange(height, dtype=np.float32)
        cols = np.arange(width, dtype=np.float32)
        expected_u = cols[None, :] + 0.5
        expected_v = rows[:, None] + 0.5
        expected_u = np.broadcast_to(expected_u, (height, width))
        expected_v = np.broadcast_to(expected_v, (height, width))

        pixels_np = _to_numpy(pixels)

        # Rows adjacent to the poles are clamped away from the exact pole by
        # design (degenerate longitude there), so exclude the outermost rows
        # from the exact round-trip check; all interior rows must match.
        interior = slice(1, height - 1)
        assert_allclose(
            pixels_np[interior, :, 0], expected_u[interior, :], rtol=0, atol=1e-2
        )
        assert_allclose(
            pixels_np[interior, :, 1], expected_v[interior, :], rtol=0, atol=1e-2
        )

    def test_rays_are_unit_norm_and_finite(self):
        rays = _to_numpy(equirectangular_ray_map(64, 128))
        norms = np.linalg.norm(rays, axis=-1)
        assert np.isfinite(rays).all()
        assert_allclose(norms, 1.0, rtol=0, atol=1e-5)

    def test_pole_does_not_nan(self):
        """Row 0 (north pole) and the last row (south pole) must stay finite."""
        height, width = 64, 128
        rays = _to_numpy(equirectangular_ray_map(height, width))
        assert np.isfinite(rays[0]).all()
        assert np.isfinite(rays[-1]).all()

        norms_top = np.linalg.norm(rays[0], axis=-1)
        norms_bottom = np.linalg.norm(rays[-1], axis=-1)
        assert_allclose(norms_top, 1.0, rtol=0, atol=1e-5)
        assert_allclose(norms_bottom, 1.0, rtol=0, atol=1e-5)

        pixels = _to_numpy(
            equirectangular_unproject_points(
                keras.ops.convert_to_tensor(rays), height, width
            )
        )
        assert np.isfinite(pixels).all()
