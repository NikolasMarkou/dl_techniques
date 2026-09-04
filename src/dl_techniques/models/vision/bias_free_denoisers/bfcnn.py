"""A bias-free ResNet-style CNN for image denoising, built by
`create_bfcnn_denoiser` and its `create_bfcnn_variant` factory.

Every additive constant (bias term) is removed from the forward path, which
makes the network exactly homogeneous of degree 1: scaling the input by a
scalar scales the output by the same scalar. A denoiser trained at one noise
level therefore generalizes across levels instead of memorizing one, and its
residual is a scaled score estimate in the Miyasawa sense. The architecture
is a stem convolution, a stack of bias-free residual blocks, and a final
1x1 projection back to the input's channel count; `normalization_type`
defaults to `'batchnorm'`, which resolves to the variance-only
`BiasFreeBatchNorm` rather than stock `BatchNormalization`, whose
`moving_mean` subtraction would break the homogeneity property.

References:
    - Mohan et al., 2020. Robust and Interpretable Blind Image Denoising via
      Bias-Free Convolutional Neural Networks. ICLR 2020.
      (https://arxiv.org/abs/1906.05478)
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385)
    - Zhang et al., 2017. Beyond a Gaussian Denoiser: Residual Learning of Deep
      CNN for Image Denoising (DnCNN). (https://arxiv.org/abs/1608.03981)
    - Miyasawa, 1961. An empirical Bayes estimator of the mean of a normal
      population. Bull. Inst. Internat. Statist. 38, 181-188.
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.conv_blocks.bias_free_conv2d import (
    BiasFreeConv2D,
    BiasFreeResidualBlock,
    resolve_denoiser_normalization,
)

# ---------------------------------------------------------------------
# Model Variant Configurations
# ---------------------------------------------------------------------

BFCNN_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tiny': {
        'num_blocks': 2,
        'filters': 32,
        'description': 'Tiny BFCNN (~ResNet-5) for quick experiments and resource-constrained environments'
    },
    'small': {
        'num_blocks': 5,
        'filters': 48,
        'description': 'Small BFCNN (~ResNet-10) with minimal capacity'
    },
    'base': {
        'num_blocks': 12,
        'filters': 64,
        'description': 'Base BFCNN (~ResNet-25) with standard configuration'
    },
    'large': {
        'num_blocks': 25,
        'filters': 96,
        'description': 'Large BFCNN (~ResNet-50) with high capacity'
    },
    'xlarge': {
        'num_blocks': 50,
        'filters': 128,
        'description': 'Extra-Large BFCNN (~ResNet-100) for maximum performance'
    }
}

# ---------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------

# DECISION plan-2026-08-24T174647-07af0659/D-002: keep layer names, creation
# order, and the caller's initializer/regularizer objects exactly as in the
# original inline builder — a rename or re-resolved initializer silently
# breaks checkpoint loading or RNG draws with no error. See decisions.md.

def _validate_bfcnn_args(
        input_shape: Tuple[int, int, int],
        num_blocks: int,
        filters: int,
        normalization_type: str,
) -> str:
    """Validate the builder arguments and resolve the block normalization name.

    Arguments mirror the identically-named parameters of `create_bfcnn_denoiser`.
    Checks run in the order written; when two arguments are invalid at once,
    the order determines which error message the caller sees.

    :param input_shape: Shape of input images, `(height, width, channels)`.
    :param num_blocks: Number of residual blocks.
    :param filters: Number of filters in residual blocks.
    :param normalization_type: Normalization requested by the caller.
    :return: The resolved normalization name every residual block must receive.
    :rtype: str
    :raises ValueError: If num_blocks is negative or filters is zero or negative.
    :raises TypeError: If input_shape is not a tuple of 3 integers.
    """
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise TypeError("input_shape must be a tuple of 3 integers (height, width, channels)")

    if num_blocks < 0:
        raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")

    if filters <= 0:
        raise ValueError(f"filters must be positive, got {filters}")

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-020: resolve 'batchnorm' to
    # BiasFreeBatchNorm rather than passing it straight through — stock
    # BatchNormalization's moving_mean breaks homogeneity. See decisions.md.
    block_normalization_type = resolve_denoiser_normalization(normalization_type)

    return block_normalization_type

def _build_bfcnn_backbone(
        inputs: keras.KerasTensor,
        num_blocks: int,
        filters: int,
        output_channels: int,
        initial_kernel_size: Union[int, Tuple[int, int]],
        kernel_size: Union[int, Tuple[int, int]],
        activation: Union[str, callable],
        block_normalization_type: str,
        final_activation: Union[str, callable],
        kernel_initializer: Union[str, keras.initializers.Initializer],
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]],
) -> keras.KerasTensor:
    """Build the bias-free backbone: stem, residual stack, final projection.

    Arguments mirror the identically-named parameters of `create_bfcnn_denoiser`,
    except `block_normalization_type` (the resolved name from
    `_validate_bfcnn_args`) and `output_channels` (the final projection's
    channel count). `kernel_initializer` / `kernel_regularizer` are forwarded
    as objects, never re-resolved.

    Note:
        `num_blocks=0` is a permitted degenerate configuration; the loop
        below simply never runs, leaving stem directly followed by the
        final projection.

    :return: The model output tensor.
    :rtype: keras.KerasTensor
    """
    x = BiasFreeConv2D(
        filters=filters,
        kernel_size=initial_kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,
        name='stem'
    )(inputs)

    for i in range(num_blocks):
        x = BiasFreeResidualBlock(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            normalization_type=block_normalization_type,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=f'residual_block_{i}'
        )(x)

    outputs = BiasFreeConv2D(
        filters=output_channels,
        kernel_size=1,
        activation=final_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,
        name='final_conv'
    )(x)

    return outputs

# ---------------------------------------------------------------------
# Core Model Creation Function
# ---------------------------------------------------------------------

def create_bfcnn_denoiser(
        input_shape: Tuple[int, int, int],
        num_blocks: int = 8,
        filters: int = 64,
        initial_kernel_size: Union[int, Tuple[int, int]] = 5,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        activation: Union[str, callable] = 'relu',
        normalization_type: str = 'batchnorm',
        final_activation: Union[str, callable] = 'linear',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        model_name: str = 'bfcnn_denoiser'
) -> keras.Model:
    """Build a bias-free CNN denoiser as a functional `keras.Model`.

    The model is homogeneous of degree 1: scaling the input by a scalar
    scales the output by the same scalar.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
           |
        BiasFreeConv2D (stem)          -> [B, H, W, filters]
           |
        BiasFreeResidualBlock x num_blocks
           |
        BiasFreeConv2D 1x1 (final_conv) -> [B, H, W, C]

    :param input_shape: Shape of input images, `(height, width, channels)`.
    :type input_shape: Tuple[int, int, int]
    :param num_blocks: Number of residual blocks. Defaults to 8.
    :type num_blocks: int
    :param filters: Number of filters in residual blocks. Defaults to 64.
    :type filters: int
    :param initial_kernel_size: Size of the stem convolution kernel. Defaults to 5.
    :param kernel_size: Size of the residual block kernels. Defaults to 3.
    :param activation: Activation function. Defaults to `'relu'`.
    :param normalization_type: Normalization used inside the residual blocks: one
        of ``'batchnorm'``, ``'layernorm'``, ``'bias_free_batchnorm'``. Defaults
        to ``'batchnorm'``, which resolves to the variance-only
        `BiasFreeBatchNorm` (no `moving_mean`, no beta), an exact synonym of
        `'bias_free_batchnorm'`. Stock `BatchNormalization` subtracts
        `moving_mean` at inference and is not reachable from this builder,
        since it breaks degree-1 homogeneity. `'layernorm'` is scale-invariant
        (degree 0), not homogeneous.
    :type normalization_type: str
    :param final_activation: Final activation function. Defaults to `'linear'`.
    :param kernel_initializer: Weight initializer. Defaults to `'glorot_uniform'`.
    :param kernel_regularizer: Optional weight regularizer.
    :param model_name: Model name. Defaults to `'bfcnn_denoiser'`.
    :type model_name: str
    :return: A compiled-ready Keras model.
    :rtype: keras.Model
    :raises ValueError: If num_blocks is negative or filters is zero or negative.
    :raises TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        ```python
        model = create_bfcnn_denoiser(
            input_shape=(None, None, 1),
            num_blocks=10,
            filters=64
        )
        model.compile(optimizer='adam', loss='mse', metrics=['psnr'])
        ```
    """
    block_normalization_type = _validate_bfcnn_args(
        input_shape=input_shape,
        num_blocks=num_blocks,
        filters=filters,
        normalization_type=normalization_type,
    )

    inputs = keras.Input(shape=input_shape, name='input_images')
    output_channels = input_shape[-1]

    outputs = _build_bfcnn_backbone(
        inputs=inputs,
        num_blocks=num_blocks,
        filters=filters,
        output_channels=output_channels,
        initial_kernel_size=initial_kernel_size,
        kernel_size=kernel_size,
        activation=activation,
        block_normalization_type=block_normalization_type,
        final_activation=final_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=model_name
    )

    logger.info(f"Created bias-free CNN model '{model_name}' with {num_blocks} residual blocks and {filters} filters")
    logger.info(f"Model input shape: {input_shape}, output channels: {output_channels}")

    return model

# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_bfcnn_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        **kwargs
) -> keras.Model:
    """Create a BFCNN model from a named variant configuration.

    :param variant: One of `'tiny'`, `'small'`, `'base'`, `'large'`, `'xlarge'`.
    :type variant: str
    :param input_shape: Shape of input images, `(height, width, channels)`.
    :type input_shape: Tuple[int, int, int]
    :param kwargs: Overrides for the variant's default parameters.
    :return: A BFCNN model with the given variant's configuration.
    :rtype: keras.Model
    :raises ValueError: If variant is not recognized.

    Example:
        ```python
        model = create_bfcnn_variant('base', (256, 256, 3))
        model.summary()
        ```
    """
    if variant not in BFCNN_CONFIGS:
        available_variants = list(BFCNN_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available_variants}")

    config = BFCNN_CONFIGS[variant].copy()
    description = config.pop('description')

    config.update(kwargs)

    if 'model_name' not in config:
        config['model_name'] = f'bfcnn_{variant}'

    logger.info(f"Creating BFCNN variant '{variant}': {description}")

    return create_bfcnn_denoiser(
        input_shape=input_shape,
        **config
    )

# ---------------------------------------------------------------------
