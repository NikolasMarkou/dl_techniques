"""Bias-free ConvUNext: the ``use_bias=False`` arm of the merged ConvUNext builder.

This module holds two things. First, thin wrappers,
``create_convunext_denoiser`` and ``create_convunext_variant``, that pin
``use_bias=False`` and forward to
``dl_techniques.models.vision.convunext.model.create_convunext``, where the
architecture and every parameter are described once. Second, this module is
the Keras registrar for a saved bias-free ConvUNext graph: importing it is
what lets ``keras.models.load_model`` resolve ``ConvUNextStem``,
``ConvNextV1Block``, ``ConvNextV2Block``, ``GlobalResponseNormalization``,
``MatchChannels``, ``StochasticDepth``, ``DownsampleAndSkip``,
``SpatialLinearAttention`` and ``GaborFiltersInitializer``, which
``applications/bias_free_denoiser/denoiser_prior.py`` and the bfunet eval
tools depend on. A bare ``import dl_techniques`` does not register them.

The bias-free design gives scaling invariance: scaling the input by alpha
scales the output by alpha, which lets one denoiser generalize across noise
levels and enables the Miyasawa/Tweedie residual-as-score reading.
``create_convunext``'s docstring names the exceptions that survive: the
non-homogeneous default activations, two hardcoded bias-free sites, and
GRN's ``beta``.

References:
    - Mohan et al., 2020. Robust and Interpretable Blind Image Denoising via
      Bias-Free Convolutional Neural Networks. ICLR 2020.
      (https://arxiv.org/abs/1906.05478)
    - Liu et al., 2022. A ConvNet for the 2020s (ConvNeXt).
      (https://arxiv.org/abs/2201.03545)
    - Ronneberger et al., 2015. U-Net: Convolutional Networks for Biomedical
      Image Segmentation. (https://arxiv.org/abs/1505.04597)
    - Miyasawa, 1961 / Robbins, 1956. The empirical-Bayes identity behind the
      residual-as-score reading named above.
"""

import keras
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

# DECISION plan-2026-08-19T163559-499b6f0e/D-080: every import below keeps its
# `# noqa: F401` — each registers a Keras serializable as a side effect, and a
# static unused-import sweep deleted all twelve once, breaking 7 tests. See decisions.md.
from dl_techniques.layers.conv_blocks.convnext_v1_block import ConvNextV1Block  # noqa: F401
from dl_techniques.layers.conv_blocks.convnext_v2_block import ConvNextV2Block  # noqa: F401
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization  # noqa: F401
from dl_techniques.layers.stochastic_depth import StochasticDepth  # noqa: F401
from dl_techniques.initializers import create_gabor_depthwise_conv2d  # noqa: F401
from dl_techniques.layers.conv_blocks.match_channels import MatchChannels  # noqa: F401
from dl_techniques.layers.downsample_and_skip import DownsampleAndSkip  # noqa: F401

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Registrar re-exports from the shared ConvUNext builder
# ---------------------------------------------------------------------

# DECISION plan-2026-08-14T092357-0e3d792d/D-010: keep importing ConvUNextStem
# under this module's own name; it makes this module the Keras registrar that
# load_model needs to resolve a saved bias-free ConvUNext graph. See decisions.md.
from dl_techniques.models.vision.convunext.model import ConvUNextStem  # noqa: F401

# SpatialLinearAttention registers today via register_dl_technique, which also
# binds the legacy Custom>SpatialLinearAttention alias so old archives still load.
from dl_techniques.models.vision.convunext.model import (  # noqa: F401
    SpatialLinearAttention,
    CONVUNEXT_CONFIGS,
    create_convunext,
    create_convunext_variant as _create_convunext_variant,
)


# ---------------------------------------------------------------------
# Core Model Creation Function (bias-free arm)
# ---------------------------------------------------------------------

def create_convunext_denoiser(
        input_shape: Tuple[int, int, int],
        *,
        block_normalization: Optional[str] = None,
        **kwargs: Any
) -> keras.Model:
    """Create the bias-free ConvUNext denoiser: `create_convunext(..., use_bias=False)`.

    A thin delegator. Every keyword other than `input_shape` and
    `block_normalization` is forwarded verbatim to
    `dl_techniques.models.vision.convunext.model.create_convunext`, which is
    where the architecture and every parameter's meaning are described.

    Two arguments are pinned by this wrapper and cannot be forwarded:
    `use_bias=False` and `stem_normalization='global_response_norm'`.
    Passing either raises `TypeError: ... got multiple values ...`.

    :param input_shape: Shape of input images `(height, width, channels)`.
    :type input_shape: tuple of 3 ints
    :param block_normalization: `None`, the default, is a sentinel meaning
        "not chosen". It resolves to `'layernorm'`, the historical default, so
        the built graph is unchanged, and logs a warning that this builder's
        default is scale-invariant (degree 0) rather than the degree-1
        homogeneity the package name promises. Pass `'batchnorm'` for a
        homogeneous bias-free stack, or `'layernorm'` explicitly to silence
        the warning while keeping the historical graph.
    :type block_normalization: Optional[str]
    :param kwargs: Every remaining parameter of
        :func:`dl_techniques.models.vision.convunext.model.create_convunext`,
        forwarded unchanged.
    :return: A functional, bias-free `keras.Model`.
    :rtype: keras.Model
    :raises TypeError: If a keyword is not a `create_convunext` parameter, or
        duplicates `use_bias`/`stem_normalization`/`input_shape`.
    """
    # DECISION plan-2026-08-14T092357-0e3d792d/D-011: forward every parameter
    # via **kwargs, not a hand-listed argument list — a hand-listed one goes
    # stale silently when create_convunext gains a parameter. See decisions.md.

    # DECISION plan-2026-08-18T140459-7991552f/D-048: block_normalization
    # defaults to a None sentinel, not the string 'layernorm', so an omitted
    # choice can warn while an explicit 'layernorm' choice stays silent. See decisions.md.
    if block_normalization is None:
        block_normalization = 'layernorm'
        logger.warning(
            "create_convunext_denoiser: no block_normalization was passed, so "
            "it resolves to 'layernorm' -- the historical default, which is "
            "scale-INVARIANT (degree 0) and therefore NOT degree-1 "
            "homogeneous. This builder is the one exported bias-free entry "
            "point whose default breaks f(a*x) = a*f(x); the named variants "
            "(create_convunext_variant) select 'batchnorm' and do not. Pass "
            "block_normalization='batchnorm' for a homogeneous bias-free "
            "stack, or pass block_normalization='layernorm' explicitly to "
            "silence this and keep the historical graph."
        )

    # DECISION plan-2026-08-24T120026-64ffd751/D-010: pin use_bias and
    # stem_normalization as explicit arguments here, not kwargs.setdefault —
    # a setdefault lets a caller silently override the bias-free contract. See decisions.md.
    return create_convunext(
        input_shape=input_shape,
        use_bias=False,
        stem_normalization='global_response_norm',
        block_normalization=block_normalization,
        **kwargs,
    )


# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_convunext_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        enable_deep_supervision: bool = True,
        **kwargs
) -> keras.Model:
    """Create a bias-free ConvUNext model from a named variant configuration.

    Forwards to `models.convunext.model.create_convunext_variant` with
    `use_bias=False`. `enable_deep_supervision` defaults to True here,
    differing from the shared builder's False; this is this entry point's
    frozen historical signature.

    The named bias-free variants use `block_normalization='batchnorm'`,
    applied here as a `setdefault` so a caller-supplied value still wins.
    A bare :func:`create_convunext_denoiser` call still gets `'layernorm'`.

    :param variant: One of `'tiny'`, `'small'`, `'base'`, `'large'`, `'xlarge'`, keys of the shared `CONVUNEXT_CONFIGS`.
    :type variant: str
    :param input_shape: Shape of input images `(height, width, channels)`.
    :type input_shape: tuple of 3 ints
    :param enable_deep_supervision: Whether to enable deep-supervision outputs. Defaults to True.
    :type enable_deep_supervision: bool
    :param kwargs: Additional keyword arguments overriding the variant defaults. `use_bias` is not among them; it is pinned to `False` by this wrapper.
    :return: A functional, bias-free `keras.Model`.
    :rtype: keras.Model
    :raises ValueError: If `variant` is not recognized.
    :raises TypeError: If `use_bias` is passed, or a keyword is not a `create_convunext` parameter.

    Example:
        ```python
        model = create_convunext_variant('base', (256, 256, 3),
                                          enable_deep_supervision=True)
        inference = create_convunext_variant('base', (None, None, 3),
                                              enable_deep_supervision=False,
                                              convnext_version='v1')
        ```
    """
    # DECISION plan-2026-08-14T092357-0e3d792d/D-014: select 'batchnorm' here,
    # at the variant wrapper, not in the shared CONVUNEXT_CONFIGS dict — that
    # dict feeds both arms and would flip the bias-ON variants too. See decisions.md.
    kwargs.setdefault('block_normalization', 'batchnorm')

    # DECISION plan-2026-08-24T120026-64ffd751/D-013: pin use_bias=False as an
    # explicit argument, not kwargs.setdefault — a setdefault previously let
    # use_bias=True silently return a biased model. See decisions.md.
    return _create_convunext_variant(
        variant=variant,
        input_shape=input_shape,
        enable_deep_supervision=enable_deep_supervision,
        use_bias=False,
        **kwargs
    )
