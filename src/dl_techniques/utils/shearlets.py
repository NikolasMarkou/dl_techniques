import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import matplotlib.cm as cm
from itertools import product
from functools import lru_cache
import matplotlib.colors as colors
from skimage.morphology import medial_axis
from typing import Tuple, Union, List, Iterator
from matplotlib.colors import LinearSegmentedColormap


def cone_orientation(level: int) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Compute shear-indexes for the vertical and horizontal cones of a shearlet system for a specific shear-level.

    This function calculates the orientation indices for vertical and horizontal cones in a shearlet
    system, which is used in directional multi-scale image analysis. The function generates three arrays:
    vertical cone indices, horizontal cone indices, and shear parameters.

    Args:
        level: int
            The shear level determining the resolution of the directional analysis.
            Must be non-negative.

    Returns:
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]
            A tuple containing:
            - vertical_cone: Array of indices for the vertical cone
            - horizontal_cone: Array of indices for the horizontal cone
            - shear_params: Array of shear parameters

    Raises:
        ValueError: If level is negative.

    Examples:
        >>> vertical, horizontal, params = cone_orientation(2)
        >>> print(vertical)  # Example output: [1 2 3 4 5]
        >>> print(horizontal)  # Example output: [3 4]
        >>> print(params)  # Example output: [0 1 2 1 0 -1]

    Notes:
        The function implements a specific indexing scheme for shearlet systems where:
        1. The vertical cone represents coefficients primarily capturing vertical features
        2. The horizontal cone represents coefficients primarily capturing horizontal features
        3. The shear parameters control the directional sensitivity
    """
    # Input validation
    if not isinstance(level, (int, np.integer)):
        raise TypeError(f"Level must be an integer, got {type(level)}")
    if level < 0:
        raise ValueError(f"Level must be non-negative, got {level}")

    # Calculate array size based on level
    array_size: int = 2 ** level + 2

    # Initialize shear parameters array
    shear_params: npt.NDArray[np.int_] = np.zeros(array_size, dtype=np.int_)

    # Calculate initial orientations
    base_level: int = max(level - 2, 0)
    initial_count: int = 2 ** base_level + 1
    initial_indices: npt.NDArray[np.int_] = np.arange(initial_count)

    # Set initial shear parameters
    shear_params[initial_indices] = initial_indices

    # Calculate vertical cone indices
    vertical_indices: npt.NDArray[np.int_] = initial_indices + 1

    # Calculate middle section orientations
    mid_start: int = vertical_indices[-1]
    mid_count: int = 2 ** (level - 1) + 1
    middle_indices: npt.NDArray[np.int_] = np.arange(
        mid_start,
        mid_start + mid_count,
        dtype=np.int_
    )

    # Calculate middle section shear parameters
    mid_center: int = middle_indices[len(middle_indices) // 2]
    shear_params[middle_indices] = (-1 * middle_indices) + mid_center

    # Calculate horizontal cone indices
    horizontal_indices: npt.NDArray[np.int_] = middle_indices + 1

    # Calculate final section
    final_start: int = horizontal_indices[-1]
    final_indices: npt.NDArray[np.int_] = np.arange(final_start, array_size)

    # Update final section if it exists
    if final_indices.size > 0:
        shear_params[final_indices] = final_indices - (final_indices[-1] + 1)
        vertical_indices = np.concatenate((vertical_indices, final_indices + 1))

    return vertical_indices, horizontal_indices, shear_params


"""
Base functions and helpers to construct 2D real-valued even-symmetric shearlet in the frequency domain.

This module provides the core functionality for constructing and manipulating shearlets,
which are directional wavelets used in signal processing and image analysis.
"""


def construct_shearlet(
        rows: int,
        cols: int,
        wavelet_eff_supp: int,
        gaussian_eff_supp: int,
        scales_per_octave: float,
        shear_level: int,
        alpha: float,
        sample_wavelet_off_origin: bool,
        scale: Union[int, float],
        ori: int,
        coneh: npt.NDArray[np.int_],
        ks: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """
    Construct a 2D real-valued even-symmetric shearlet in the frequency domain.

    This function generates a shearlet by combining a wavelet and a Gaussian function,
    applying appropriate scaling and shearing operations to achieve directional sensitivity.

    Args:
        rows: Height of the constructed shearlet.
        cols: Width of the constructed shearlet.
        wavelet_eff_supp: Effective support for wavelet function used in construction.
        gaussian_eff_supp: Effective support for Gauss function used in construction.
        scales_per_octave: Number of scales per octave.
        shear_level: Amount of shearing applied.
        alpha: Scaling parameter for the Gaussian component.
        sample_wavelet_off_origin: Whether to sample the wavelet off-origin.
        scale: Scaling parameter for the shearlet.
        ori: Orientation parameter for directional sensitivity.
        coneh: Array of horizontal cone indices.
        ks: Array of shear parameters.

    Returns:
        npt.NDArray[np.float64]: The constructed shearlet in frequency domain.

    Raises:
        ValueError: If input dimensions or parameters are invalid.
        TypeError: If input types are incorrect.
    """
    # Input validation
    if not all(isinstance(x, int) for x in [rows, cols, wavelet_eff_supp, gaussian_eff_supp, shear_level]):
        raise TypeError("Dimensional parameters must be integers")
    if not all(x > 0 for x in [rows, cols, wavelet_eff_supp, gaussian_eff_supp]):
        raise ValueError("Dimensional parameters must be positive")
    if not isinstance(scales_per_octave, (int, float)) or scales_per_octave <= 0:
        raise ValueError("scales_per_octave must be a positive number")

    # Transpose dimensions if needed based on orientation
    if ori not in coneh:
        rows, cols = cols, rows

    # Compute frequency vectors
    omega_wav: npt.NDArray[np.float64] = (63 * float(wavelet_eff_supp) / 512) * yapuls(rows)
    omega_gau: npt.NDArray[np.float64] = (74 * float(gaussian_eff_supp) / 512) * yapuls(
        cols * (2 ** (shear_level - 2))
    )

    # Apply scaling
    omega_gau = omega_gau / ((2 ** ((scale - 1) / scales_per_octave)) ** alpha)
    omega_wav = omega_wav / (2 ** ((scale - 1) / scales_per_octave))

    # Construct Mexican hat wavelet in frequency domain
    wav_freq: npt.NDArray[np.float64] = np.atleast_2d(
        2 * np.pi * np.multiply(
            np.power(omega_wav, 2),
            np.exp(np.power(omega_wav, 2) / -2)
        )
    )

    # Handle off-origin sampling
    if sample_wavelet_off_origin:
        wav_freq = padarray(
            np.fft.fftshift(wav_freq),
            (1, wav_freq.shape[1] * 2)
        )
        wav_time = np.real(
            np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(wav_freq))
            )
        )

        wav_slice = slice(1, None, 2) if (len(wav_freq) % 2 != 0) else slice(0, -1, 2)
        if ori in coneh:
            wav_freq = np.fft.fft(np.fft.ifftshift(wav_time[::-1, wav_slice]))
        else:
            wav_freq = np.fft.fft(np.fft.ifftshift(wav_time[:, wav_slice]))

    # Construct Gaussian in frequency domain
    gau_freq: npt.NDArray[np.float64] = np.atleast_2d(
        np.exp(-1 * np.power(omega_gau, 2) / 2)
    )

    # Construct final shearlet based on orientation
    if ori in coneh:
        shearlet = np.fft.fftshift(wav_freq.T * gau_freq)
        shearlet = shear(shearlet, -1 * ks[ori - 1], 2)
        shearlet = shearlet[:, ::(2 ** (shear_level - 2))]
    else:
        shearlet = np.fft.fftshift(gau_freq.T * wav_freq)
        shearlet = shear(shearlet, -1 * ks[ori - 1], 1)
        shearlet = shearlet[::(2 ** (shear_level - 2)), :]

    return shearlet


def padarray(
        array: npt.NDArray[np.float64],
        newsize: Tuple[int, int]
) -> npt.NDArray[np.float64]:
    """
    Pad array to specified size with zeros.

    Args:
        array: Input array to be padded.
        newsize: Target shape for padded array (height, width).

    Returns:
        npt.NDArray[np.float64]: Padded array of specified size.

    Raises:
        ValueError: If newsize is smaller than input array size.
    """
    if not all(b >= a for a, b in zip(array.shape, newsize)):
        raise ValueError("Target size must be larger than input array size")

    sizediff: List[int] = [b - a for a, b in zip(array.shape, newsize)]
    pad: List[int] = [
        diff // 2 if (diff % 2 == 0) else (diff // 2) + 1
        for diff in sizediff
    ]
    lshift: List[int] = [
        1 if (size % 2 == 0) and not (diff % 2 == 0) else 0
        for diff, size in zip(sizediff, array.shape)
    ]

    return np.pad(
        array,
        [(a - s, new - (a - s + old))
         for a, s, new, old in zip(pad, lshift, newsize, array.shape)],
        mode='constant'
    )


def shear(
        data: npt.NDArray[np.float64],
        k: int,
        axis: int
) -> npt.NDArray[np.float64]:
    """
    Discretely shear the input data on given axis by k.

    Originally from 'ShearLab 3D'. See http://www.shearlab.org/

    Args:
        data: Input array (e.g. base shearlet).
        k: Amount of shearing to apply.
        axis: Axis to apply shearing (1 for columns, 2 for rows).

    Returns:
        npt.NDArray[np.float64]: Sheared input data.

    Raises:
        ValueError: If axis is not 1 or 2.
    """
    if not isinstance(axis, int) or axis not in {1, 2}:
        raise ValueError("Axis must be either 1 or 2")

    if k == 0:
        return data

    rows, cols = data.shape
    ret: npt.NDArray[np.float64] = np.zeros(data.shape, dtype=data.dtype)

    if axis == 1:
        for col in range(cols):
            ret[:, col] = np.roll(data[:, col], __shift(k, cols, col))
    else:
        for row in range(rows):
            ret[row, :] = np.roll(data[row, :], __shift(k, rows, row))

    return ret


def __shift(k: int, total: int, x: int) -> int:
    """
    Compute (circular) shift for one column during shearing.

    Args:
        k: Shearing parameter.
        total: Total size of dimension being sheared.
        x: Current position.

    Returns:
        int: Amount to shift the current position.
    """
    return k * ((total // 2) - x)


@lru_cache(maxsize=128)
def yapuls(npuls: int) -> npt.NDArray[np.float64]:
    """
    Generate a pulsation vector for wavelet construction.

    Originally from 'Yet Another Wavelet Toolbox (YAWTb)'.
    See http://sites.uclouvain.be/ispgroup/yawtb/

    Args:
        npuls: Length of the pulsation vector. Must be positive.

    Returns:
        npt.NDArray[np.float64]: Pulsation vector containing concatenated
            subvectors in [0, π) and [-π, 0).

    Raises:
        ValueError: If npuls is not positive.
    """
    if not isinstance(npuls, int) or npuls <= 0:
        raise ValueError("npuls must be a positive integer")

    npuls_2: int = (npuls - 1) // 2
    return (2 * np.pi / npuls) * np.concatenate((
        np.arange(npuls_2 + 1),
        np.arange(npuls_2 - npuls + 1, 0)
    ))


def scale_angle(angle: float) -> float:
    """
    Scale an angle to a value between -90 and 90 degrees.

    This function normalizes angles by wrapping them around to ensure they fall
    within the range [-90, 90]. This is useful for orientation calculations
    where angles beyond this range are equivalent to angles within it.

    Args:
        angle: The input angle in degrees to be scaled.

    Returns:
        float: The scaled angle in degrees, guaranteed to be in [-90, 90].

    Examples:
        >>> scale_angle(100)
        -80.0
        >>> scale_angle(-100)
        80.0
        >>> scale_angle(45)
        45.0
    """
    scaled_angle = float(angle)  # Ensure we're working with float

    if scaled_angle > 90:
        scaled_angle -= 180
    if scaled_angle < -90:
        scaled_angle += 180

    return scaled_angle


def curvature(orientations: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Measure local curvature from an array of element-wise orientation measurements.

    This function calculates the local curvature at each point in an orientation field
    by comparing the orientations of neighboring pixels. It is typically used after
    edge or ridge detection to analyze the shape characteristics of detected features.

    Args:
        orientations: 2D array of orientation measurements (in degrees).
            Should be pre-thinned (e.g., using thin_mask).
            Negative values indicate invalid/no orientation.

    Returns:
        npt.NDArray[np.float64]: 2D array containing local curvature measurements.
            - Positive values indicate the measured curvature in degrees
            - -1 indicates points where curvature couldn't be computed

    Raises:
        ValueError: If input array is not 2D or has invalid dimensions.

    Examples:
        >>> # Example usage with edge detection:
        >>> i = Image.open("image.jpg").convert("L")
        >>> img = np.asarray(i)
        >>> edge_sys = EdgeSystem(*img.shape)
        >>> edges, edge_orientations = edge_sys.detect(img, min_contrast=70)
        >>> thinned_edges = mask(edges, thin_mask(edges))
        >>> thinned_orientations = mask(edge_orientations, thin_mask(edges))
        >>> edge_curv = curvature(thinned_orientations)

    Notes:
        - The input orientation array should be pre-processed (thinned) for best results
        - Border pixels (outer 1-pixel margin) will always have curvature = -1
        - Curvature is computed by analyzing orientation differences between neighbors
        - The returned curvature is the average of left and right orientation differences
    """
    # Validate input
    if not isinstance(orientations, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if orientations.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if orientations.shape[0] < 3 or orientations.shape[1] < 3:
        raise ValueError("Input array must be at least 3x3")

    # Initialize output array
    curv: npt.NDArray[np.float64] = np.full_like(orientations, -1.0)

    # Generate neighbor offsets (excluding center point)
    offsets: List[Tuple[int, int]] = [
        (a, b) for a, b in product(range(-1, 2), range(-1, 2))
        if not (a == 0 and b == 0)
    ]

    # Calculate curvature for each interior point
    for row in range(1, orientations.shape[0] - 1):
        for col in range(1, orientations.shape[1] - 1):
            if orientations[row, col] >= 0:
                count = 0
                left = right = 0.0

                # Find left and right neighbors with valid orientations
                for dr, dc in offsets:
                    neighbor_val = orientations[row + dr, col + dc]
                    if neighbor_val >= 0:
                        if count > 0:
                            left = neighbor_val
                        else:
                            right = neighbor_val
                        count += 1

                # Calculate curvature if exactly two valid neighbors found
                if count == 2:
                    d_or_left = scale_angle(orientations[row, col] - left)
                    d_or_right = scale_angle(right - orientations[row, col])
                    curv[row, col] = abs(d_or_left + d_or_right) / 2

    return curv


def overlay(
        img: npt.NDArray[np.float64],
        img_overlay: npt.NDArray[np.float64]
) -> npt.NDArray[np.uint8]:
    """
    Create an RGB overlay of two grayscale images.

    Overlays a red visualization on top of a base grayscale image.
    Both images are normalized before combination.

    Args:
        img: Base grayscale image to be overlaid.
        img_overlay: Overlay image to be displayed in red.

    Returns:
        npt.NDArray[np.uint8]: RGB image with overlay, values in [0, 255].

    Raises:
        ValueError: If input arrays have different shapes or wrong dimensions.

    Examples:
        >>> base = np.ones((100, 100))
        >>> overlay_img = np.zeros((100, 100))
        >>> overlay_img[40:60, 40:60] = 1
        >>> result = overlay(base, overlay_img)
        >>> result.shape
        (100, 100, 3)
    """
    # Validate inputs
    if img.shape != img_overlay.shape:
        raise ValueError("Base image and overlay must have the same shape")
    if img.ndim != 2 or img_overlay.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")

    # Normalize input images
    img_overlay_norm: npt.NDArray[np.float64] = img_overlay / np.maximum(
        img_overlay.max(),
        np.finfo(np.float64).tiny
    )
    img_norm: npt.NDArray[np.float64] = (img / np.maximum(
        img.max(),
        np.finfo(np.float64).tiny
    )) * (1 - img_overlay_norm)

    # Create output RGB arrays
    height, width = img.shape
    red: npt.NDArray[np.float64] = np.zeros((height, width, 3))
    img_rgb: npt.NDArray[np.float64] = np.zeros((height, width, 3))

    # Set red channel for overlay
    red[..., 0] = img_overlay_norm

    # Set all channels for base image
    for i in range(3):
        img_rgb[..., i] = img_norm

    # Combine and convert to uint8
    return np.clip((img_rgb + red) * 255, 0, 255).astype(np.uint8)


def thin_mask(img: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """
    Create a thinning mask using medial axis transform.

    Args:
        img: Binary input image to be thinned.

    Returns:
        npt.NDArray[np.bool_]: Binary mask where True indicates pixels to be masked.

    Raises:
        ValueError: If input is not a binary array.
    """
    if not np.issubdtype(img.dtype, np.bool_):
        raise ValueError("Input must be a binary array")

    return np.invert(medial_axis(img))


def mask(
        img: npt.NDArray,
        mask: npt.NDArray[np.bool_]
) -> ma.MaskedArray:
    """
    Apply a binary mask to an image.

    Creates a masked array with specified fill value for masked regions.

    Args:
        img: Input array to be masked.
        mask: Boolean mask array (True indicates masked values).

    Returns:
        ma.MaskedArray: Masked array with fill_value=-1.

    Raises:
        ValueError: If input arrays have different shapes.
    """
    if img.shape != mask.shape:
        raise ValueError("Image and mask must have the same shape")

    ret = ma.masked_array(img)
    ret.fill_value = -1
    ret.mask = mask
    return ret


def cyclic_cmap() -> LinearSegmentedColormap:
    """
    Create a cyclic colormap for angle visualization.

    Returns a colormap that cycles through colors in a way suitable for
    displaying angular data, with smooth transitions between colors.

    Returns:
        LinearSegmentedColormap: Matplotlib colormap object with 180 colors.

    Notes:
        The colormap cycles through: black -> red -> white -> blue -> black
    """
    colors_list: list[str] = [
        '#000000',  # black
        '#ff0000',  # red
        '#ffffff',  # white
        '#0000ff',  # blue
        '#000000'  # black
    ]

    return colors.LinearSegmentedColormap.from_list(
        'anglemap',
        colors_list,
        N=180,
        gamma=1
    )


def curvature_rgb(
        curvature: npt.NDArray[np.float64],
        max_curvature: float = 10.0
) -> npt.NDArray[np.float64]:
    """
    Convert curvature values to RGB colors using a jet colormap.

    Maps curvature values to colors, with values above max_curvature being
    clamped. Negative curvature values are mapped to transparent pixels.

    Args:
        curvature: Array of curvature values.
        max_curvature: Maximum curvature value for color mapping.
            Values above this are clamped. Defaults to 10.0.

    Returns:
        npt.NDArray[np.float64]: RGBA array where each pixel is
            [red, green, blue, alpha] in [0, 1].

    Raises:
        ValueError: If max_curvature is not positive.
    """
    if max_curvature <= 0:
        raise ValueError("max_curvature must be positive")

    # Create copy to avoid modifying input
    rgb: npt.NDArray[np.float64] = np.copy(curvature)

    # Clamp values above max_curvature
    rgb[curvature > max_curvature] = max_curvature

    # Normalize to [0, 1] range for colormap
    rgb = rgb / max_curvature

    # Set invalid values
    rgb[curvature < 0] = -1

    # Apply jet colormap
    return cm.jet(rgb)
