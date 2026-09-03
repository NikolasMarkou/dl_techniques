"""Two helpers for the masked autoencoder: a factory and a visualizer.

``create_mae_model`` wires a caller-supplied encoder to the convolutional
decoder and returns a built :class:`MaskedAutoencoder`. ``visualize_reconstruction``
plots an image, its masked view, and the reconstruction side by side.

Neither function defines the architecture, the masking policy, or the loss
those live in ``dl_techniques.models.vision.masked_autoencoder.mae``.

References:
    - He et al., 2022. Masked Autoencoders Are Scalable Vision Learners.
      CVPR 2022. (https://arxiv.org/abs/2111.06377)
    - Xie et al., 2022. SimMIM: A Simple Framework for Masked Image Modeling.
      CVPR 2022. (https://arxiv.org/abs/2111.09886)
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
    """Build a MaskedAutoencoder from a caller-supplied encoder.

    :param encoder: A built Keras model to use as the feature extractor.
    :param patch_size: Patch size used for masking.
    :param mask_ratio: Fraction of patches to mask.
    :param decoder_dims: Decoder channel widths, one per stage.
    :param input_shape: Input image shape.
    :param kwargs: Passed through to `MaskedAutoencoder`.
    :return: A configured `MaskedAutoencoder` instance.
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
    """Build a composite grid of original, masked, and reconstructed images.

    :param mae: A trained `MaskedAutoencoder` model.
    :param images: Batch of images, shape `(N, H, W, C)`.
    :param num_samples: Number of images to plot.
    :return: Composite image array of shape `(N * H, 3 * W, C)`, each row
        laid out as original, masked, reconstructed.
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
