import numpy as np
from typing import Optional, Tuple, List, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .mae import MaskedAutoencoder

# ---------------------------------------------------------------------

def create_mae_model(
        encoder_dims: List[int],
        encoder_output_shape: Tuple[int, int, int],
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_dims: Optional[List[int]] = None,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        **kwargs: Any
) -> MaskedAutoencoder:
    """Convenience function to create MAE model.

    Args:
        encoder_dims: List of integers, encoder channel dimensions.
        encoder_output_shape: Tuple, encoder output shape (H, W, C).
        patch_size: Integer, patch size for masking.
        mask_ratio: Float, ratio of patches to mask.
        decoder_dims: Optional list of decoder dimensions.
        input_shape: Tuple, input image shape.
        **kwargs: Additional arguments for MaskedAutoencoder.

    Returns:
        MaskedAutoencoder model instance.

    Example:
        >>> # For ConvNeXt-Tiny architecture
        >>> mae = create_mae_model(
        ...     encoder_dims=[96, 192, 384, 768],
        ...     encoder_output_shape=(7, 7, 768),
        ...     mask_ratio=0.75
        ... )
        >>> mae.compile(optimizer="adam")
    """
    mae = MaskedAutoencoder(
        encoder_dims=encoder_dims,
        encoder_output_shape=encoder_output_shape,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        decoder_dims=decoder_dims,
        input_shape=input_shape,
        **kwargs
    )

    return mae

# ---------------------------------------------------------------------

def visualize_reconstruction(
        mae: MaskedAutoencoder,
        images: np.ndarray,
        num_samples: int = 4
) -> np.ndarray:
    """Visualize MAE reconstructions for multiple images.

    Creates a grid showing original, masked, and reconstructed images.

    Args:
        mae: MaskedAutoencoder model.
        images: Array of images of shape (N, H, W, C).
        num_samples: Integer, number of samples to visualize.

    Returns:
        Grid of images showing [original, masked, reconstructed] for each sample.

    Example:
        >>> grid = visualize_reconstruction(mae, test_images, num_samples=4)
        >>> plt.imshow(grid)
    """
    num_samples = min(num_samples, len(images))
    samples = images[:num_samples]

    results = []
    for img in samples:
        original, masked, reconstructed = mae.visualize(img)
        results.append([original, masked, reconstructed])

    # Stack into grid: (num_samples, 3, H, W, C)
    grid = np.array(results)

    # Reshape to (num_samples * H, 3 * W, C)
    num_samples, _, H, W, C = grid.shape
    grid = grid.transpose(0, 2, 1, 3, 4)  # (num_samples, H, 3, W, C)
    grid = grid.reshape(num_samples * H, 3 * W, C)

    # Clip to valid range
    grid = np.clip(grid, 0, 1)

    return grid

# ---------------------------------------------------------------------