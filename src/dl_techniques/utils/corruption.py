"""
Keras-only implementation of image corruption functions with severity enum.

This module provides various image corruption functions using only Keras operations,
making them compatible with different backends (TensorFlow, JAX, PyTorch).
All functions use a standardized severity enum for type safety.
"""

import keras
from keras import ops
from enum import Enum
from typing import Union, Callable, Dict, List

# ---------------------------------------------------------------------

class CorruptionSeverity(Enum):
    """
    Enumeration for corruption severity levels.

    Each level represents increasing intensity of corruption:
    - MILD: Level 1 - Barely noticeable corruption
    - MODERATE: Level 2 - Noticeable but not severe
    - MEDIUM: Level 3 - Clearly visible corruption
    - STRONG: Level 4 - Significant corruption
    - SEVERE: Level 5 - Maximum corruption intensity
    """
    MILD = 1
    MODERATE = 2
    MEDIUM = 3
    STRONG = 4
    SEVERE = 5

# ---------------------------------------------------------------------


class CorruptionType(Enum):
    """
    Enumeration for available corruption types.
    """
    GAUSSIAN_NOISE = "gaussian_noise"
    IMPULSE_NOISE = "impulse_noise"
    SHOT_NOISE = "shot_noise"
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"
    PIXELATE = "pixelate"
    QUANTIZE = "quantize"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATE = "saturate"

# ---------------------------------------------------------------------


def _create_gaussian_kernel_1d(size: int, sigma: float) -> keras.KerasTensor:
    """
    Create a 1D Gaussian kernel.

    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation of the Gaussian

    Returns:
        1D Gaussian kernel as KerasTensor
    """
    # Create coordinate array
    half_size = size // 2
    coords = ops.cast(ops.arange(-half_size, half_size + 1), dtype='float32')

    # Calculate Gaussian values
    kernel = ops.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / ops.sum(kernel)

    return kernel

# ---------------------------------------------------------------------

def _apply_separable_gaussian_blur(image: keras.KerasTensor, sigma: float) -> keras.KerasTensor:
    """
    Apply Gaussian blur using separable 1D convolutions.

    Args:
        image: Input image tensor of shape (height, width, channels)
        sigma: Standard deviation for Gaussian blur

    Returns:
        Blurred image tensor
    """
    # Determine kernel size (should be odd and large enough)
    kernel_size = int(2 * ops.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian kernel
    kernel_1d = _create_gaussian_kernel_1d(kernel_size, sigma)
    kernel_1d = ops.cast(kernel_1d, image.dtype)

    # Reshape for convolution
    kernel_h = ops.reshape(kernel_1d, (kernel_size, 1, 1, 1))
    kernel_v = ops.reshape(kernel_1d, (1, kernel_size, 1, 1))

    # Add batch dimension if needed
    original_shape = ops.shape(image)
    if len(original_shape) == 3:
        image = ops.expand_dims(image, axis=0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    # Apply separable convolution to each channel
    channels = ops.shape(image)[-1]
    blurred_channels = []

    for c in range(channels):
        single_channel = image[:, :, :, c:c+1]

        # Apply horizontal blur
        h_blurred = ops.conv(
            single_channel,
            kernel_h,
            strides=1,
            padding='same'
        )

        # Apply vertical blur
        v_blurred = ops.conv(
            h_blurred,
            kernel_v,
            strides=1,
            padding='same'
        )

        blurred_channels.append(v_blurred)

    # Concatenate channels back
    blurred = ops.concatenate(blurred_channels, axis=-1)

    # Remove batch dimension if it was added
    if squeeze_batch:
        blurred = ops.squeeze(blurred, axis=0)

    return blurred

# ---------------------------------------------------------------------

def apply_gaussian_noise(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply Gaussian noise corruption.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Corrupted image tensor
    """
    noise_levels: Dict[CorruptionSeverity, float] = {
        CorruptionSeverity.MILD: 0.04,
        CorruptionSeverity.MODERATE: 0.08,
        CorruptionSeverity.MEDIUM: 0.12,
        CorruptionSeverity.STRONG: 0.16,
        CorruptionSeverity.SEVERE: 0.20,
    }

    noise_std = noise_levels[severity]
    noise = keras.random.normal(ops.shape(image), mean=0.0, stddev=noise_std, dtype=image.dtype)
    return ops.clip(image + noise, ops.cast(0, image.dtype), ops.cast(1, image.dtype))

# ---------------------------------------------------------------------

def apply_impulse_noise(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply impulse (salt and pepper) noise corruption.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Corrupted image tensor
    """
    noise_probs: Dict[CorruptionSeverity, float] = {
        CorruptionSeverity.MILD: 0.01,
        CorruptionSeverity.MODERATE: 0.02,
        CorruptionSeverity.MEDIUM: 0.03,
        CorruptionSeverity.STRONG: 0.05,
        CorruptionSeverity.SEVERE: 0.07,
    }

    noise_prob = noise_probs[severity]
    corrupted = ops.copy(image)

    # Create random masks for salt and pepper noise
    random_vals = keras.random.uniform(ops.shape(image)[:2], dtype=image.dtype)

    # Salt noise (white pixels)
    salt_mask = random_vals < (noise_prob / 2)
    salt_mask = ops.expand_dims(salt_mask, axis=-1)
    salt_mask = ops.broadcast_to(salt_mask, ops.shape(image))

    # Pepper noise (black pixels)
    pepper_mask = random_vals > (1 - noise_prob / 2)
    pepper_mask = ops.expand_dims(pepper_mask, axis=-1)
    pepper_mask = ops.broadcast_to(pepper_mask, ops.shape(image))

    # Apply noise
    corrupted = ops.where(salt_mask, ops.cast(1.0, image.dtype), corrupted)
    corrupted = ops.where(pepper_mask, ops.cast(0.0, image.dtype), corrupted)

    return corrupted

# ---------------------------------------------------------------------

def apply_shot_noise(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply shot noise (Poisson-like noise) corruption.

    Uses normal distribution approximation since Poisson(λ) ≈ Normal(λ, √λ) for large λ.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Corrupted image tensor
    """
    noise_levels: Dict[CorruptionSeverity, int] = {
        CorruptionSeverity.MILD: 500,
        CorruptionSeverity.MODERATE: 250,
        CorruptionSeverity.MEDIUM: 100,
        CorruptionSeverity.STRONG: 75,
        CorruptionSeverity.SEVERE: 50,
    }

    lambda_val = noise_levels[severity]

    # Scale image to appropriate range for shot noise
    scaled_image = image * lambda_val

    # Apply shot noise using normal approximation to Poisson
    # For Poisson(λ), we use Normal(λ, √λ)
    shot_noise = keras.random.normal(
        shape=ops.shape(scaled_image),
        mean=scaled_image,
        stddev=ops.sqrt(ops.maximum(scaled_image, 1e-8)),  # Avoid sqrt(0)
        dtype=image.dtype
    )

    # Ensure non-negative values (characteristic of shot noise)
    noisy_scaled = ops.maximum(shot_noise, ops.cast(0.0, image.dtype))

    # Scale back to [0, 1]
    noisy_image = noisy_scaled / lambda_val

    return ops.clip(noisy_image, 0, 1)

# ---------------------------------------------------------------------

def apply_gaussian_blur(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply Gaussian blur corruption.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Blurred image tensor
    """
    sigma_values: Dict[CorruptionSeverity, float] = {
        CorruptionSeverity.MILD: 0.5,
        CorruptionSeverity.MODERATE: 1.0,
        CorruptionSeverity.MEDIUM: 1.5,
        CorruptionSeverity.STRONG: 2.0,
        CorruptionSeverity.SEVERE: 2.5,
    }

    sigma = sigma_values[severity]
    return _apply_separable_gaussian_blur(image, sigma)

# ---------------------------------------------------------------------

def apply_motion_blur(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply motion blur corruption using convolution.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Motion blurred image tensor
    """
    kernel_sizes: Dict[CorruptionSeverity, int] = {
        CorruptionSeverity.MILD: 3,
        CorruptionSeverity.MODERATE: 5,
        CorruptionSeverity.MEDIUM: 7,
        CorruptionSeverity.STRONG: 9,
        CorruptionSeverity.SEVERE: 11,
    }

    kernel_size = kernel_sizes[severity]

    # Create motion blur kernel
    kernel = ops.zeros((kernel_size, kernel_size), dtype=image.dtype)
    middle_row = kernel_size // 2

    # Create horizontal motion blur kernel
    kernel_values = ops.ones((kernel_size,), dtype=image.dtype) / kernel_size

    # Create the full kernel by setting the middle row
    indices = ops.stack([
        ops.full((kernel_size,), middle_row, dtype='int32'),
        ops.arange(kernel_size, dtype='int32')
    ], axis=1)

    kernel = ops.scatter_update(kernel, indices, kernel_values)

    # Reshape kernel for convolution
    kernel = ops.expand_dims(ops.expand_dims(kernel, axis=-1), axis=-1)

    # Add batch dimension if needed
    original_shape = ops.shape(image)
    if len(original_shape) == 3:
        image = ops.expand_dims(image, axis=0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    # Apply motion blur to each channel separately
    channels = ops.shape(image)[-1]
    blurred_channels = []

    for c in range(channels):
        single_channel = image[:, :, :, c:c+1]
        blurred_channel = ops.conv(
            single_channel,
            kernel,
            strides=1,
            padding='same'
        )
        blurred_channels.append(blurred_channel)

    blurred = ops.concatenate(blurred_channels, axis=-1)

    # Remove batch dimension if it was added
    if squeeze_batch:
        blurred = ops.squeeze(blurred, axis=0)

    return ops.clip(blurred, ops.cast(0, image.dtype), ops.cast(1, image.dtype))

# ---------------------------------------------------------------------

def apply_pixelate(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply pixelate corruption using pooling operations.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Pixelated image tensor
    """
    pixel_sizes: Dict[CorruptionSeverity, int] = {
        CorruptionSeverity.MILD: 28,
        CorruptionSeverity.MODERATE: 24,
        CorruptionSeverity.MEDIUM: 20,
        CorruptionSeverity.STRONG: 16,
        CorruptionSeverity.SEVERE: 12,
    }

    pixel_size = pixel_sizes[severity]

    original_shape = ops.shape(image)
    height, width = original_shape[0], original_shape[1]

    # If image is smaller than pixel_size, return original (no pixelation effect)
    if height < pixel_size or width < pixel_size:
        return image

    # Add batch dimension if needed
    if len(original_shape) == 3:
        image = ops.expand_dims(image, axis=0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    # Calculate how many complete blocks we can fit
    h_blocks = height // pixel_size
    w_blocks = width // pixel_size

    # Calculate the dimensions of the area we'll actually pixelate
    effective_h = h_blocks * pixel_size
    effective_w = w_blocks * pixel_size

    if h_blocks == 0 or w_blocks == 0:
        # No complete blocks fit, return original
        if squeeze_batch:
            return ops.squeeze(image, axis=0)
        return image

    # Extract the area that will be pixelated
    pixelated_area = image[:, :effective_h, :effective_w, :]

    # Reshape to group pixels into blocks for averaging
    # (batch, h_blocks, pixel_size, w_blocks, pixel_size, channels)
    reshaped = ops.reshape(
        pixelated_area,
        (ops.shape(image)[0], h_blocks, pixel_size, w_blocks, pixel_size, ops.shape(image)[3])
    )

    # Average over the pixel_size dimensions (dims 2 and 4)
    averaged = ops.mean(reshaped, axis=(2, 4))  # (batch, h_blocks, w_blocks, channels)

    # Expand back to original block size
    expanded = ops.repeat(ops.repeat(averaged, pixel_size, axis=1), pixel_size, axis=2)

    # Reconstruct the full image
    result = ops.copy(image)

    # Replace the pixelated area - we need to handle this carefully
    # Since we can't directly assign to slices, we'll build the result

    # Split the image into regions
    top_left = expanded  # pixelated area

    # Handle remaining areas
    if effective_w < width:
        # Right edge
        right_edge = image[:, :effective_h, effective_w:, :]
        top_row = ops.concatenate([top_left, right_edge], axis=2)
    else:
        top_row = top_left

    if effective_h < height:
        # Bottom edge
        bottom_edge = image[:, effective_h:, :, :]
        result = ops.concatenate([top_row, bottom_edge], axis=1)
    else:
        result = top_row

    # Remove batch dimension if it was added
    if squeeze_batch:
        result = ops.squeeze(result, axis=0)

    return result

# ---------------------------------------------------------------------

def apply_quantize(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply quantization (bit-depth reduction) corruption.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Quantized image tensor
    """
    bit_depths: Dict[CorruptionSeverity, int] = {
        CorruptionSeverity.MILD: 6,
        CorruptionSeverity.MODERATE: 5,
        CorruptionSeverity.MEDIUM: 4,
        CorruptionSeverity.STRONG: 3,
        CorruptionSeverity.SEVERE: 2,
    }

    bits = bit_depths[severity]

    # Quantize to reduced bit depth
    levels = ops.cast(2 ** bits, image.dtype)
    levels_minus_one = ops.cast(levels - 1, image.dtype)
    quantized = ops.round(image * levels_minus_one) / levels_minus_one

    return ops.clip(quantized, ops.cast(0, image.dtype), ops.cast(1, image.dtype))

# ---------------------------------------------------------------------

def apply_brightness(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply brightness corruption.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Brightness adjusted image tensor
    """
    brightness_factors: Dict[CorruptionSeverity, float] = {
        CorruptionSeverity.MILD: 1.3,
        CorruptionSeverity.MODERATE: 1.6,
        CorruptionSeverity.MEDIUM: 1.9,
        CorruptionSeverity.STRONG: 2.2,
        CorruptionSeverity.SEVERE: 2.5,
    }

    brightness_factor = brightness_factors[severity]
    brightness_factor = ops.cast(brightness_factor, image.dtype)
    return ops.clip(image * brightness_factor, ops.cast(0, image.dtype), ops.cast(1, image.dtype))

# ---------------------------------------------------------------------

def apply_contrast(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply contrast corruption.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Contrast adjusted image tensor
    """
    contrast_factors: Dict[CorruptionSeverity, float] = {
        CorruptionSeverity.MILD: 1.4,
        CorruptionSeverity.MODERATE: 1.8,
        CorruptionSeverity.MEDIUM: 2.2,
        CorruptionSeverity.STRONG: 2.6,
        CorruptionSeverity.SEVERE: 3.0,
    }

    contrast_factor = contrast_factors[severity]
    contrast_factor = ops.cast(contrast_factor, image.dtype)
    midpoint = ops.cast(0.5, image.dtype)
    return ops.clip((image - midpoint) * contrast_factor + midpoint, ops.cast(0, image.dtype), ops.cast(1, image.dtype))

# ---------------------------------------------------------------------

def apply_saturate(image: keras.KerasTensor, severity: CorruptionSeverity) -> keras.KerasTensor:
    """
    Apply saturation corruption.

    Args:
        image: Input image tensor of shape (height, width, channels)
        severity: Corruption severity level

    Returns:
        Saturation adjusted image tensor
    """
    saturation_factors: Dict[CorruptionSeverity, float] = {
        CorruptionSeverity.MILD: 1.4,
        CorruptionSeverity.MODERATE: 1.8,
        CorruptionSeverity.MEDIUM: 2.2,
        CorruptionSeverity.STRONG: 2.6,
        CorruptionSeverity.SEVERE: 3.0,
    }

    saturation_factor = saturation_factors[severity]

    # Convert to grayscale using luminance weights for RGB
    gray_weights = ops.cast(ops.array([0.299, 0.587, 0.114]), image.dtype)
    gray = ops.sum(image * gray_weights, axis=-1, keepdims=True)

    # Adjust saturation
    saturated = gray + (image - gray) * saturation_factor

    return ops.clip(saturated, ops.cast(0, image.dtype), ops.cast(1, image.dtype))

# ---------------------------------------------------------------------

def get_corruption_function(corruption_type: Union[CorruptionType, str]) -> Callable[[keras.KerasTensor, CorruptionSeverity], keras.KerasTensor]:
    """
    Get the corruption function for a given corruption type.

    Args:
        corruption_type: Corruption type enum or string name

    Returns:
        Corruption function that takes (image, severity) and returns corrupted image

    Raises:
        ValueError: If corruption type is not supported
    """
    corruption_functions: Dict[CorruptionType, Callable] = {
        CorruptionType.GAUSSIAN_NOISE: apply_gaussian_noise,
        CorruptionType.IMPULSE_NOISE: apply_impulse_noise,
        CorruptionType.SHOT_NOISE: apply_shot_noise,
        CorruptionType.GAUSSIAN_BLUR: apply_gaussian_blur,
        CorruptionType.MOTION_BLUR: apply_motion_blur,
        CorruptionType.PIXELATE: apply_pixelate,
        CorruptionType.QUANTIZE: apply_quantize,
        CorruptionType.BRIGHTNESS: apply_brightness,
        CorruptionType.CONTRAST: apply_contrast,
        CorruptionType.SATURATE: apply_saturate,
    }

    # Handle string input
    if isinstance(corruption_type, str):
        try:
            corruption_type = CorruptionType(corruption_type)
        except ValueError:
            available_types = [ct.value for ct in CorruptionType]
            raise ValueError(
                f"Unsupported corruption type '{corruption_type}'. "
                f"Available types: {available_types}"
            )

    return corruption_functions[corruption_type]

# ---------------------------------------------------------------------

def apply_corruption(
    image: keras.KerasTensor,
    corruption_type: Union[CorruptionType, str],
    severity: CorruptionSeverity
) -> keras.KerasTensor:
    """
    Apply a specific corruption to an image.

    Args:
        image: Input image tensor of shape (height, width, channels)
        corruption_type: Type of corruption to apply
        severity: Severity level of the corruption

    Returns:
        Corrupted image tensor
    """
    corruption_fn = get_corruption_function(corruption_type)
    return corruption_fn(image, severity)

# ---------------------------------------------------------------------

def get_all_corruption_types() -> List[CorruptionType]:
    """
    Get a list of all available corruption types.

    Returns:
        List of all corruption types
    """
    return list(CorruptionType)

# ---------------------------------------------------------------------

def get_all_severity_levels() -> List[CorruptionSeverity]:
    """
    Get a list of all available severity levels.

    Returns:
        List of all severity levels
    """
    return list(CorruptionSeverity)

# ---------------------------------------------------------------------


