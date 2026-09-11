"""Camera-model ray geometry: pinhole, fisheye (equidistant), equirectangular.

This module provides pure, backend-agnostic (``keras.ops`` only) functions for
two directions of camera geometry, for three camera models:

1. **Ray generation** -- given camera intrinsics and an image size, produce a
   per-pixel unit ray-direction map ``(H, W, 3)`` in camera space.
2. **Inverse projection** -- given 3D ray directions in camera space, recover
   the pixel coordinates ``(u, v)`` that would have produced them.

The three camera models are:

- **Pinhole**: the standard linear projection model, parameterized by focal
  lengths ``(fx, fy)`` and a principal point ``(cx, cy)``.
- **Fisheye (equidistant / Kannala-Brandt style)**: the angle ``theta`` between
  a ray and the optical axis is proportional to the ray's radial pixel
  distance from the principal point (``r = f * theta``), which is the
  equidistant special case of the more general Kannala-Brandt polynomial
  model.
- **Equirectangular (360 degree / spherical)**: pixel coordinates map linearly
  to longitude/latitude on the unit sphere, as used for panoramic images.

All ray-generation functions return **L2-normalized, finite** unit vectors
everywhere, including at pixels beyond a fisheye's valid field of view (the
incidence angle is clamped) and at the equirectangular poles (latitude is
clamped strictly inside +/- 90 degrees so the degenerate polar longitude never
produces a NaN).

This module is used both to build ray-map conditioning inputs for models that
need to know their input camera's geometry, and to derive ground-truth
ray/distance targets from depth maps of a known camera model.

References
----------
- Pinhole camera model: R. Hartley and A. Zisserman, "Multiple View Geometry
  in Computer Vision," 2nd ed., Cambridge University Press, 2004, Ch. 6.
- Kannala-Brandt / equidistant fisheye model: J. Kannala and S. S. Brandt, "A
  Generic Camera Model and Calibration Method for Conventional, Wide-Angle,
  and Fish-Eye Lenses," IEEE TPAMI, 2006.
- Equirectangular / spherical panorama projection: standard
  longitude/latitude-to-Cartesian spherical coordinate mapping, as used for
  360 degree panoramic (ERP) imagery.
"""

import keras
from typing import Tuple

# ---------------------------------------------------------------------

# A ray whose incidence angle would exceed this bound is clamped to it. This
# keeps `tan(theta)` (pinhole inverse) and the fisheye/equirectangular pixel
# maps finite at every pixel, including synthetic reprojections that probe
# deliberately out-of-range geometry.
MAX_INCIDENCE_ANGLE_RAD = 1.5533430342749532  # ~89 degrees, strictly < pi / 2

# Numerical floor used wherever a denominator could vanish (radial distance
# in the fisheye inverse, ray norm before normalization).
_EPSILON = 1e-8

# ---------------------------------------------------------------------


def _l2_normalize_rays(rays: keras.KerasTensor) -> keras.KerasTensor:
    """Normalize the last axis of ``rays`` to unit length, safely.

    :param rays: Ray directions, shape ``(..., 3)``.
    :type rays: keras.KerasTensor
    :return: Unit-norm ray directions, same shape, never NaN/Inf -- a
        zero-norm input row is mapped to itself (all zeros) rather than
        dividing by zero.
    :rtype: keras.KerasTensor
    """
    norm = keras.ops.sqrt(
        keras.ops.sum(keras.ops.square(rays), axis=-1, keepdims=True)
    )
    return rays / keras.ops.maximum(norm, _EPSILON)


# ---------------------------------------------------------------------
# Pinhole
# ---------------------------------------------------------------------


def pinhole_ray_map(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    height: int,
    width: int,
) -> keras.KerasTensor:
    """Generate a per-pixel unit ray-direction map for a pinhole camera.

    Pixel ``(u, v)`` (column, row; pixel-center convention ``u = col + 0.5``,
    ``v = row + 0.5``) back-projects to the camera-space direction
    ``(x, y, z) = ((u - cx) / fx, (v - cy) / fy, 1)``, which is then
    L2-normalized.

    :param fx: Focal length in pixels, x axis.
    :type fx: float
    :param fy: Focal length in pixels, y axis.
    :type fy: float
    :param cx: Principal point x coordinate, in pixels.
    :type cx: float
    :param cy: Principal point y coordinate, in pixels.
    :type cy: float
    :param height: Image height in pixels.
    :type height: int
    :param width: Image width in pixels.
    :type width: int
    :return: Unit ray directions, shape ``(height, width, 3)``, camera-space
        (``+z`` is the optical axis, forward).
    :rtype: keras.KerasTensor
    """
    rows = keras.ops.arange(height, dtype="float32")
    cols = keras.ops.arange(width, dtype="float32")
    u = cols[None, :] + 0.5  # (1, W)
    v = rows[:, None] + 0.5  # (H, 1)
    u = keras.ops.broadcast_to(u, (height, width))
    v = keras.ops.broadcast_to(v, (height, width))

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = keras.ops.ones_like(x)

    rays = keras.ops.stack([x, y, z], axis=-1)
    return _l2_normalize_rays(rays)


def pinhole_unproject_points(
    rays: keras.KerasTensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> keras.KerasTensor:
    """Project camera-space ray directions to pinhole pixel coordinates.

    The inverse of :func:`pinhole_ray_map`. Rays whose forward (``z``)
    component is at or below zero (behind or orthogonal to the image plane)
    cannot be projected by a pinhole model; their ``z`` is clamped to a small
    positive floor rather than dividing by zero or a negative number, so the
    output stays finite (callers relying on such rays being valid for a
    pinhole model should mask them separately).

    :param rays: Ray directions, shape ``(..., 3)``. Need not be unit-norm --
        only the direction (ratio of components) matters for projection.
    :type rays: keras.KerasTensor
    :param fx: Focal length in pixels, x axis.
    :type fx: float
    :param fy: Focal length in pixels, y axis.
    :type fy: float
    :param cx: Principal point x coordinate, in pixels.
    :type cx: float
    :param cy: Principal point y coordinate, in pixels.
    :type cy: float
    :return: Pixel coordinates ``(u, v)``, shape ``(..., 2)``.
    :rtype: keras.KerasTensor
    """
    x = rays[..., 0]
    y = rays[..., 1]
    z = keras.ops.maximum(rays[..., 2], _EPSILON)

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return keras.ops.stack([u, v], axis=-1)


# ---------------------------------------------------------------------
# Fisheye (equidistant / Kannala-Brandt)
# ---------------------------------------------------------------------


def fisheye_ray_map(
    f: float,
    cx: float,
    cy: float,
    height: int,
    width: int,
    max_incidence_angle: float = MAX_INCIDENCE_ANGLE_RAD,
) -> keras.KerasTensor:
    """Generate a per-pixel unit ray-direction map for an equidistant fisheye camera.

    Under the equidistant projection model, the incidence angle ``theta``
    between a ray and the optical axis relates to the pixel's radial distance
    ``r`` from the principal point by ``r = f * theta``, i.e.
    ``theta = r / f``. The azimuthal angle ``phi = atan2(dv, du)`` is
    preserved from the pixel offset. ``theta`` is clamped to
    ``max_incidence_angle`` so pixels nominally beyond the lens's valid field
    of view still produce a finite, unit-norm (if physically meaningless past
    the clamp) ray rather than diverging.

    :param f: Focal length in pixels (equidistant model constant).
    :type f: float
    :param cx: Principal point x coordinate, in pixels.
    :type cx: float
    :param cy: Principal point y coordinate, in pixels.
    :type cy: float
    :param height: Image height in pixels.
    :type height: int
    :param width: Image width in pixels.
    :type width: int
    :param max_incidence_angle: Maximum incidence angle (radians) any ray is
        allowed to reach; larger radial distances clamp to this angle instead
        of diverging past it. Defaults to :data:`MAX_INCIDENCE_ANGLE_RAD`
        (~89 degrees).
    :type max_incidence_angle: float
    :return: Unit ray directions, shape ``(height, width, 3)``, camera-space
        (``+z`` is the optical axis, forward).
    :rtype: keras.KerasTensor
    """
    rows = keras.ops.arange(height, dtype="float32")
    cols = keras.ops.arange(width, dtype="float32")
    u = cols[None, :] + 0.5
    v = rows[:, None] + 0.5
    u = keras.ops.broadcast_to(u, (height, width))
    v = keras.ops.broadcast_to(v, (height, width))

    du = u - cx
    dv = v - cy
    r = keras.ops.sqrt(keras.ops.square(du) + keras.ops.square(dv))
    phi = keras.ops.arctan2(dv, du)

    theta = keras.ops.clip(r / f, 0.0, max_incidence_angle)

    x = keras.ops.sin(theta) * keras.ops.cos(phi)
    y = keras.ops.sin(theta) * keras.ops.sin(phi)
    z = keras.ops.cos(theta)

    rays = keras.ops.stack([x, y, z], axis=-1)
    return _l2_normalize_rays(rays)


def fisheye_unproject_points(
    rays: keras.KerasTensor,
    f: float,
    cx: float,
    cy: float,
    max_incidence_angle: float = MAX_INCIDENCE_ANGLE_RAD,
) -> keras.KerasTensor:
    """Project camera-space ray directions to equidistant fisheye pixel coordinates.

    The inverse of :func:`fisheye_ray_map`. The incidence angle
    ``theta = arccos(z / ||ray||)`` is clamped to ``max_incidence_angle``
    before being scaled to a radial pixel distance ``r = f * theta``, so a ray
    pointing beyond the lens's valid field of view (or even behind the
    camera) still maps to a finite pixel coordinate rather than an unbounded
    or NaN one.

    :param rays: Ray directions, shape ``(..., 3)``. Normalized internally, so
        need not already be unit-norm.
    :type rays: keras.KerasTensor
    :param f: Focal length in pixels (equidistant model constant).
    :type f: float
    :param cx: Principal point x coordinate, in pixels.
    :type cx: float
    :param cy: Principal point y coordinate, in pixels.
    :type cy: float
    :param max_incidence_angle: Maximum incidence angle (radians) used to
        clamp ``theta``. Defaults to :data:`MAX_INCIDENCE_ANGLE_RAD`.
    :type max_incidence_angle: float
    :return: Pixel coordinates ``(u, v)``, shape ``(..., 2)``.
    :rtype: keras.KerasTensor
    """
    unit_rays = _l2_normalize_rays(rays)
    x = unit_rays[..., 0]
    y = unit_rays[..., 1]
    z = unit_rays[..., 2]

    # arccos's domain is [-1, 1]; clip against float round-off pushing |z| > 1.
    theta = keras.ops.arccos(keras.ops.clip(z, -1.0, 1.0))
    theta = keras.ops.clip(theta, 0.0, max_incidence_angle)
    phi = keras.ops.arctan2(y, x)

    r = f * theta
    u = r * keras.ops.cos(phi) + cx
    v = r * keras.ops.sin(phi) + cy
    return keras.ops.stack([u, v], axis=-1)


# ---------------------------------------------------------------------
# Equirectangular (360 degree / spherical)
# ---------------------------------------------------------------------


def equirectangular_ray_map(
    height: int,
    width: int,
) -> keras.KerasTensor:
    """Generate a per-pixel unit ray-direction map for an equirectangular (360) camera.

    Pixel column ``u`` (pixel-center, ``u in [0, width)``) maps linearly to
    longitude ``lon in (-pi, pi]``, and pixel row ``v`` (``v in [0, height)``)
    maps linearly to latitude ``lat in (-pi/2, pi/2)`` (row 0 = north pole,
    row ``height - 1`` = south pole). Latitude is kept strictly inside its
    open interval (never exactly +/- pi/2) so the azimuthal component never
    hits the sphere's exact pole, where longitude is degenerate -- this keeps
    the resulting Cartesian direction finite and well-defined even though a
    true pole pixel maps to a single point regardless of longitude.

    Camera-space convention: longitude 0 points along ``+z`` (forward),
    increasing longitude rotates toward ``+x``; latitude 0 is the equator,
    positive latitude points toward ``+y``.

    :param height: Image height in pixels (maps to latitude).
    :type height: int
    :param width: Image width in pixels (maps to longitude).
    :type width: int
    :return: Unit ray directions, shape ``(height, width, 3)``.
    :rtype: keras.KerasTensor
    """
    rows = keras.ops.arange(height, dtype="float32")
    cols = keras.ops.arange(width, dtype="float32")

    u = cols[None, :] + 0.5  # (1, W)
    v = rows[:, None] + 0.5  # (H, 1)
    u = keras.ops.broadcast_to(u, (height, width))
    v = keras.ops.broadcast_to(v, (height, width))

    lon = (u / width) * 2.0 * keras.ops.convert_to_tensor(3.14159265358979) - keras.ops.convert_to_tensor(3.14159265358979)
    lat = keras.ops.convert_to_tensor(1.5707963267948966) - (v / height) * keras.ops.convert_to_tensor(3.14159265358979)

    # Clamp strictly inside (-pi/2, pi/2): never exactly at a pole, so
    # longitude never becomes geometrically degenerate.
    pole_eps = 1e-6
    lat = keras.ops.clip(lat, -1.5707963267948966 + pole_eps, 1.5707963267948966 - pole_eps)

    x = keras.ops.cos(lat) * keras.ops.sin(lon)
    y = keras.ops.sin(lat)
    z = keras.ops.cos(lat) * keras.ops.cos(lon)

    rays = keras.ops.stack([x, y, z], axis=-1)
    return _l2_normalize_rays(rays)


def equirectangular_unproject_points(
    rays: keras.KerasTensor,
    height: int,
    width: int,
) -> keras.KerasTensor:
    """Project camera-space ray directions to equirectangular pixel coordinates.

    The inverse of :func:`equirectangular_ray_map`. Latitude is recovered via
    ``arcsin(y)`` (clipped to the valid ``[-1, 1]`` domain against float
    round-off), then clamped strictly inside ``(-pi/2, pi/2)`` -- so a ray
    pointing exactly at a pole (``y = +/-1``) still resolves to a finite pixel
    row rather than landing exactly on a boundary row with an arbitrary
    (degenerate) longitude/column.

    :param rays: Ray directions, shape ``(..., 3)``. Normalized internally, so
        need not already be unit-norm.
    :type rays: keras.KerasTensor
    :param height: Image height in pixels (maps to latitude).
    :type height: int
    :param width: Image width in pixels (maps to longitude).
    :type width: int
    :return: Pixel coordinates ``(u, v)``, shape ``(..., 2)``.
    :rtype: keras.KerasTensor
    """
    unit_rays = _l2_normalize_rays(rays)
    x = unit_rays[..., 0]
    y = unit_rays[..., 1]
    z = unit_rays[..., 2]

    lon = keras.ops.arctan2(x, z)  # (-pi, pi]
    lat = keras.ops.arcsin(keras.ops.clip(y, -1.0, 1.0))

    pole_eps = 1e-6
    lat = keras.ops.clip(lat, -1.5707963267948966 + pole_eps, 1.5707963267948966 - pole_eps)

    pi = 3.14159265358979
    u = ((lon + pi) / (2.0 * pi)) * width
    v = ((1.5707963267948966 - lat) / pi) * height
    return keras.ops.stack([u, v], axis=-1)
