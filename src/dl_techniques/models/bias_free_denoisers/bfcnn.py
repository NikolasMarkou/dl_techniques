"""
Bias-Free CNN Denoiser Model with Variants

Implements a ResNet-style denoising CNN where all additive constants (bias terms)
have been removed to enable better generalization across different noise levels.
Provides multiple model variants (tiny, small, base, large, xlarge) for different
computational requirements and performance targets.

Based on "Robust and Interpretable Blind Image Denoising via Bias-Free
Convolutional Neural Networks" (Mohan et al., ICLR 2020).

References:
    - Mohan et al., 2020. Robust and Interpretable Blind Image Denoising via
      Bias-Free Convolutional Neural Networks. ICLR 2020.
      (https://arxiv.org/abs/1906.05478) -- the bias-free result this model IS:
      removing every additive constant makes the network exactly homogeneous of
      degree 1, so a denoiser trained at one noise level generalizes across
      levels rather than memorizing one.
    - He et al., 2015. Deep Residual Learning for Image Recognition.
      (https://arxiv.org/abs/1512.03385) -- the residual block layout the
      variants stack.
    - Zhang et al., 2017. Beyond a Gaussian Denoiser: Residual Learning of Deep
      CNN for Image Denoising (DnCNN). (https://arxiv.org/abs/1608.03981) --
      the residual (noise-predicting) denoiser this is the bias-free form of.
    - Miyasawa, 1961. An empirical Bayes estimator of the mean of a normal
      population. Bull. Inst. Internat. Statist. 38, 181-188 -- with Robbins
      (1956), the identity that makes the residual of a bias-free denoiser a
      scaled score estimate.
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.bias_free_conv2d import (
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

# DECISION plan-2026-08-24T174647-07af0659/D-002: the `_validate_bfcnn_args` /
# `_build_bfcnn_backbone` helpers below are a PURE DECOMPOSITION of `create_bfcnn_denoiser`,
# extracted verbatim. Same three-part extraction contract as the bfunet helpers in this
# package (see the D-002 anchor in bfunet.py for the measurements behind it):
# (1) NOT ONE LAYER IS RENAMED and no `name=` string is edited -- this builder is functional
#     (`keras.Model(inputs, outputs)`), so layer/weight NAMES are the checkpoint contract for
#     every stored `.keras` under `results/`, and a rename breaks them silently at
#     `load_model` time, not at build time. MEASURED for this plan on the sibling builder: a
#     one-word layer rename moved the name arms while weight VALUES and the forward pass
#     stayed bit-identical -- the forward pass CANNOT see this defect class.
# (2) LAYER CREATION ORDER IS PRESERVED EXACTLY (stem -> `num_blocks` residual blocks ->
#     final 1x1 projection). Keras auto-generates names from creation order for any layer
#     built without `name=`, and name scopes are uniquified against a process-global counter.
# (3) THE CALLER'S `kernel_initializer` / `kernel_regularizer` OBJECT IS FORWARDED AS-IS. Do
#     NOT construct, clone, or re-resolve one inside a helper: that changes the number of RNG
#     draws, so every downstream layer initializes differently while every name and shape
#     still matches (the trap recorded for the BEiT restructure as D-017).
# The stem, the residual stack and the final projection live in ONE helper on purpose: they
# always co-change (same filters/kernel/activation/initializer) and the projection is two
# statements, so splitting it out would be a pass-through helper -- an Ousterhout red flag.
# Do NOT collapse the explicitly-forwarded parameters into a shared params object or a
# `**kwargs` bag; that was designed and deliberately rejected (decisions.md D-001). See
# decisions.md D-002.

def _validate_bfcnn_args(
        input_shape: Tuple[int, int, int],
        num_blocks: int,
        filters: int,
        normalization_type: str,
) -> str:
    """
    Validate the builder arguments and resolve the block normalization.

    Args mirror the identically-named parameters of `create_bfcnn_denoiser`. The checks run
    in the order written and that order is part of the contract: when two arguments are
    invalid at once, which message a caller sees is observable behaviour.

    Returns:
        String, the RESOLVED normalization name. This -- not the raw `normalization_type`
        argument -- is what every residual block must receive.

    Raises:
        ValueError: If num_blocks is negative or filters is zero or negative.
        TypeError: If input_shape is not a tuple of 3 integers.
    """
    # Input validation
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        raise TypeError("input_shape must be a tuple of 3 integers (height, width, channels)")

    if num_blocks < 0:
        raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")

    if filters <= 0:
        raise ValueError(f"filters must be positive, got {filters}")

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-020: 'batchnorm' names the homogeneous
    # BiasFreeBatchNorm inside a bias-free denoiser. Do NOT pass `normalization_type`
    # straight through to BiasFreeResidualBlock -- that reaches stock BatchNormalization,
    # whose moving_mean subtraction breaks f(a*x)=a*f(x) once training has moved it off
    # zero. See decisions.md D-020 and the anchor in layers/bias_free_conv2d.py.
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
    """
    Build the whole bias-free backbone: stem, residual stack, final projection.

    Args mirror the identically-named parameters of `create_bfcnn_denoiser`, except
    `block_normalization_type` (the RESOLVED name returned by `_validate_bfcnn_args`) and
    `output_channels` (the channel count the final 1x1 projection must emit).
    `kernel_initializer` / `kernel_regularizer` are forwarded as objects, never re-resolved
    (see the D-002 anchor above).

    Note:
        `num_blocks=0` is a permitted degenerate configuration -- `_validate_bfcnn_args`
        rejects only NEGATIVE block counts -- and the loop below simply never runs, leaving
        stem -> final projection.

    Returns:
        KerasTensor, the model output.
    """
    # Initial convolution to project to feature space
    x = BiasFreeConv2D(
        filters=filters,
        kernel_size=initial_kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,  # First layer typically no batch norm
        name='stem'
    )(inputs)

    # Stack of bias-free residual blocks
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

    # Final convolution to output channels (no activation, no batch norm)
    outputs = BiasFreeConv2D(
        filters=output_channels,
        kernel_size=1,
        activation=final_activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=False,  # Last layer typically no batch norm
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
    """
    Create a bias-free CNN model for image denoising using ResNet architecture.

    This function creates a complete Keras model using bias-free residual blocks.
    The model implements the scaling-invariant property described in the paper:
    if you scale the input by α, the output is scaled by α as well.

    Architecture:
    - Initial bias-free convolution
    - Multiple bias-free residual blocks
    - Final bias-free convolution to output channels

    Args:
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        num_blocks: Integer, number of residual blocks. Defaults to 8.
        filters: Integer, number of filters in residual blocks. Defaults to 64.
        initial_kernel_size: Integer or tuple, size of the first convolutional kernels. Defaults to 5.
        kernel_size: Integer or tuple, size of convolutional kernels. Defaults to 3.
        activation: String or callable, activation function. Defaults to 'relu'.
        normalization_type: String, normalization used inside the residual blocks; one of
            ``'batchnorm'``, ``'layernorm'``, ``'bias_free_batchnorm'``. Defaults to
            ``'batchnorm'``. **In this bias-free denoiser, ``'batchnorm'`` means the
            variance-only ``BiasFreeBatchNorm``** (no ``moving_mean``, no beta), i.e. it is
            an exact synonym of ``'bias_free_batchnorm'``; the resolution happens in
            ``resolve_denoiser_normalization``. Stock ``keras.layers.BatchNormalization``
            subtracts ``moving_mean`` at inference and is NOT degree-1 homogeneous, so it is
            not reachable from this builder. ``'layernorm'`` is a per-input normalization and
            is scale-INVARIANT (degree-0), not homogeneous.
        final_activation: String or callable, final activation function. Defaults to 'linear'.
        kernel_initializer: String or Initializer, weight initializer. Defaults to 'glorot_uniform'.
        kernel_regularizer: String or Regularizer, weight regularizer. Defaults to None.
        model_name: String, name for the model. Defaults to 'bfcnn_denoiser'.

    Returns:
        keras.Model: Compiled Keras model ready for training.

    Raises:
        ValueError: If num_blocks is negative or filters is zero or negative.
        TypeError: If input_shape is not a tuple of 3 integers.

    Example:
        >>> # Create model for grayscale images
        >>> model = create_bfcnn_denoiser(
        ...     input_shape=(None, None, 1),
        ...     num_blocks=10,
        ...     filters=64
        ... )
        >>> model.compile(optimizer='adam', loss='mse', metrics=['psnr'])
        >>>
        >>> # The model exhibits scaling invariance
        >>> # If input is scaled by α, output is also scaled by α
    """

    block_normalization_type = _validate_bfcnn_args(
        input_shape=input_shape,
        num_blocks=num_blocks,
        filters=filters,
        normalization_type=normalization_type,
    )

    # Input layer
    inputs = keras.Input(shape=input_shape, name='input_images')

    # Output same number of channels as input
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

    # Create the model
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
    """
    Create a BFCNN model with a specific variant configuration.

    Args:
        variant: String, one of 'tiny', 'small', 'base', 'large', 'xlarge'.
        input_shape: Tuple of integers, shape of input images (height, width, channels).
        **kwargs: Additional keyword arguments to override default parameters.

    Returns:
        keras.Model: BFCNN model with the specified variant configuration.

    Raises:
        ValueError: If variant is not recognized.

    Example:
        >>> model = create_bfcnn_variant('base', (256, 256, 3))
        >>> model.summary()
    """
    if variant not in BFCNN_CONFIGS:
        available_variants = list(BFCNN_CONFIGS.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available_variants}")

    config = BFCNN_CONFIGS[variant].copy()
    description = config.pop('description')

    # Override config with any provided kwargs
    config.update(kwargs)

    # Set model name if not provided
    if 'model_name' not in config:
        config['model_name'] = f'bfcnn_{variant}'

    logger.info(f"Creating BFCNN variant '{variant}': {description}")

    return create_bfcnn_denoiser(
        input_shape=input_shape,
        **config
    )

# ---------------------------------------------------------------------
