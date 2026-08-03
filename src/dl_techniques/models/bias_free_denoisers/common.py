import keras
from keras import ops
from typing import Optional, Union, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.laplacian_filter import LaplacianPyramidLevel

# ---------------------------------------------------------------------

def _downsample_and_skip(
        x: keras.KerasTensor,
        use_laplacian_pyramid: bool,
        laplacian_kernel_size: Tuple[int, int],
        downsample_name: str,
        pyramid_name: str,
        pool_type: str = "max",
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """Produce ``(skip, downsampled)`` for one encoder junction.

    OFF path (default, byte-identical to the original architecture): the skip is
    the pre-downsample tensor and downsampling is ``MaxPooling2D(2, 2)`` named
    ``downsample_name``. With ``pool_type='average'`` the downsample uses
    ``AveragePooling2D(2, 2)`` instead -- a LINEAR (and bias-free / homogeneous)
    operator, so the encoder path stays linear for the Miyasawa/Tweedie
    residual-as-score interpretation (MaxPooling is non-linear). Pooling layers are
    weightless, so the swap does not affect checkpoint weight transfer.

    ON path: a channel-preserving, bias-free ``LaplacianPyramidLevel`` split. The
    full-resolution high-frequency band becomes the skip; the half-resolution low
    band continues down the encoder. Bias-free and homogeneous by construction
    (fixed blur + average pool + bilinear upsample, zero learnable bias). The
    pyramid already pools linearly, so ``pool_type`` does not apply here.
    """
    if use_laplacian_pyramid:
        low, high = LaplacianPyramidLevel(
            blur_kernel_size=laplacian_kernel_size,
            name=pyramid_name,
        )(x)
        return high, low
    skip = x
    pool_layer = (
        keras.layers.AveragePooling2D if pool_type == "average"
        else keras.layers.MaxPooling2D
    )
    downsampled = pool_layer(
        pool_size=(2, 2),
        name=downsample_name,
    )(x)
    return skip, downsampled

# ---------------------------------------------------------------------
