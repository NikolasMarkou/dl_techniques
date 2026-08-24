"""
Construction and inspection helpers for the masked autoencoder.

Two functions, and neither of them is the architecture: :func:`create_mae_model`
is the package's ``create_*`` factory -- it wires a caller-supplied encoder to
the convolutional decoder and returns a built :class:`MAE` -- and
:func:`visualize_reconstruction` plots an image, its masked view and the
reconstruction side by side for eyeballing a checkpoint.

The model itself, its masking policy and its loss live in
``dl_techniques.models.masked_autoencoder.mae``, which carries the full
architectural description. This module is deliberately thin: anything that
decides what the model IS belongs there, not here.

References:
    - He et al., 2022. Masked Autoencoders Are Scalable Vision Learners.
      CVPR 2022. (https://arxiv.org/abs/2111.06377) -- the mask-a-high-fraction-
      of-patches-and-reconstruct-pixels recipe this package implements.
    - Xie et al., 2022. SimMIM: A Simple Framework for Masked Image Modeling.
      CVPR 2022. (https://arxiv.org/abs/2111.09886) -- the lightweight-decoder
      variant this package's convolutional decoder is closer to than to MAE's
      transformer decoder.
"""
import keras
import numpy as np
from typing import Optional, Tuple, List, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .mae import MaskedAutoencoder

# ---------------------------------------------------------------------

def create_mae_model(
        encoder: keras.Model,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_dims: Optional[List[int]] = None,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        **kwargs: Any
) -> MaskedAutoencoder:
    """Convenience factory to create an MAE model.

    Wraps the MaskedAutoencoder constructor for consistency.

    Args:
        encoder: A built Keras Model to serve as the feature extractor.
        patch_size: Size of patches for masking.
        mask_ratio: Ratio of patches to mask.
        decoder_dims: Optional list of decoder dimensions.
        input_shape: Input image shape.
        **kwargs: Arguments passed to MaskedAutoencoder.

    Returns:
        MaskedAutoencoder model instance.
    """
    mae = MaskedAutoencoder(
        encoder=encoder,
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

    Args:
        mae: Trained MaskedAutoencoder model.
        images: Batch of images (N, H, W, C).
        num_samples: Number of images to plot.

    Returns:
        A large composite image array (Grid) of shape (Total_H, Total_W, C).
        Grid Layout: [Original | Masked | Reconstructed] per row.
    """
    num_samples = min(num_samples, len(images))
    samples = images[:num_samples]

    results = []
    for img in samples:
        # returns single image arrays
        original, masked, reconstructed = mae.visualize(img, return_arrays=True)
        results.append([original, masked, reconstructed])

    # Stack: (num_samples, 3, H, W, C)
    grid = np.array(results)
    N, Cols, H, W, C = grid.shape

    # Transpose to (N, H, Cols, W, C) -> Reshape to (N*H, Cols*W, C)
    grid = grid.transpose(0, 2, 1, 3, 4)
    grid = grid.reshape(N * H, Cols * W, C)

    # Ensure valid visualization range
    grid = np.clip(grid, 0, 1)

    return grid

# ---------------------------------------------------------------------
