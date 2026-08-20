"""Bias-free ConvUNext: the ``use_bias=False`` arm of the merged ConvUNext builder.

This module is TWO things and nothing else:

1. **Thin wrappers.** ``create_convunext_denoiser`` and ``create_convunext_variant``
   keep their exact historical signatures and pin ``use_bias=False``, forwarding to
   ``dl_techniques.models.convunext.model.create_convunext``. The architecture, every
   parameter, the Laplacian-pyramid option and the three documented asymmetries of the
   bias-free arm are described ONCE, there.
2. **The Keras REGISTRAR** (contract H-4). Importing this module is what makes
   ``keras.models.load_model`` able to resolve ``ConvUNextStem``, ``ConvNextV1Block``,
   ``ConvNextV2Block``, ``GlobalResponseNormalization``, ``MatchChannels``,
   ``StochasticDepth``, ``DownsampleAndSkip``, ``SpatialLinearAttention`` and
   ``GaborFiltersInitializer`` in a saved bias-free ConvUNext graph.
   ``applications/bias_free_denoiser/denoiser_prior.py`` and the two bfunet eval tools
   depend on that side effect. A bare ``import dl_techniques`` is NOT enough.

The bias-free design is what gives scaling invariance: if the input is scaled by
alpha, the output is scaled by alpha, which is what lets one denoiser generalize
across noise levels and enables the Miyasawa/Tweedie residual-as-score reading.
``create_convunext``'s docstring names the exceptions that survive on purpose (the
non-homogeneous default activations, two hardcoded bias-free sites, GRN's ``beta``).
"""

import keras
from typing import Optional, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

# REGISTRAR imports (contract H-4). `applications/bias_free_denoiser/denoiser_prior.py`
# and the two bfunet eval tools import THIS module purely so `keras.models.load_model`
# can resolve every custom class a saved bias-free ConvUNext graph names. None of these
# are USED below any more (the builder itself moved to `models/convunext/model.py`);
# they are imported for their registration side effect. Do NOT "clean up" as unused.
# DECISION plan-2026-08-19T163559-499b6f0e/D-080: every import in this block
# carries `# noqa: F401` because this module is a REGISTRAR. Twelve of its
# imports are bound for their side effect -- importing them REGISTERS the Keras
# serializables and re-exports the ONE import path the bf test suite, the bfunet
# trainer and `utils/multiplicative_miyasawa.py` use. A static unused-import
# tool cannot see that, and step 19's sweep DELETED all twelve despite the prose
# comments below each saying "Do NOT delete this re-export"; the bf suite then
# failed 7 tests at `TestRegistrarContract`. The prose is for humans; the
# `# noqa` is for the next tool.
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block  # noqa: F401
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block  # noqa: F401
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization  # noqa: F401
from dl_techniques.layers.stochastic_depth import StochasticDepth  # noqa: F401
from dl_techniques.initializers import create_gabor_depthwise_conv2d  # noqa: F401
from dl_techniques.layers.match_channels import MatchChannels  # noqa: F401
from dl_techniques.layers.downsample_and_skip import DownsampleAndSkip  # noqa: F401

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# ConvUNext Bias-Free Building Blocks (Simple Stem)
# ---------------------------------------------------------------------

# DECISION plan-2026-08-14T092357-0e3d792d/D-010: `ConvUNextStem` no longer LIVES here —
# it was merged with the same-named class in `models/convunext/model.py` and now lives
# there, gaining `use_bias` and `stem_normalization`. This module keeps importing it
# under its own name because it is the Keras REGISTRAR that
# `applications/bias_free_denoiser/denoiser_prior.py` and the two bfunet eval tools
# import purely so `load_model` can resolve this class (contract H-4), and because the
# bf test suite imports it from here. The class's decorator deliberately keeps
# `package="dl_techniques.bias_free_denoisers"` so its registry key does not move.
# Do NOT delete this re-export, and do NOT re-home the class's `package=` string.
from dl_techniques.models.convunext.model import ConvUNextStem  # noqa: F401

# Re-exported from the merged home so this module stays the ONE import path the bf
# test suite, the bfunet trainer and `utils/multiplicative_miyasawa.py` use, and so it
# keeps REGISTERING `SpatialLinearAttention` (registrar contract H-4). The class moved
# with a BARE `@keras.saving.register_keras_serializable()`, whose key
# `Custom>SpatialLinearAttention` was MEASURED to be module-independent on Keras 3.8.0
# (decisions.md D-008), so the move did not change it. Do NOT add a `package=` argument
# "for symmetry" with `ConvUNextStem` — that WOULD change the key.
from dl_techniques.models.convunext.model import (  # noqa: F401
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
        depth: int = 4,
        initial_filters: int = 64,
        filter_multiplier: float = 2.0,
        blocks_per_level: int = 2,
        convnext_version: str = 'v2',
        stem_kernel_size: Union[int, Tuple[int, int]] = 7,
        use_gabor_stem: bool = False,
        gabor_filters: int = 32,
        gabor_kernel_size: Union[int, Tuple[int, int]] = 11,
        gabor_activation: Optional[str] = None,
        gabor_stem_projection: bool = True,
        use_laplacian_pyramid: bool = False,
        laplacian_kernel_size: Tuple[int, int] = (5, 5),
        high_freq_blocks: int = 0,
        bottleneck_attention_blocks: int = 0,
        bottleneck_attention_heads: int = 8,
        zero_pad_channels: bool = False,
        extra_zero_output_channels: bool = False,
        final_projection_groups: int = 1,
        downsample_pool_type: str = "max",
        expose_bottleneck: bool = False,
        block_kernel_size: Union[int, Tuple[int, int]] = 7,
        block_activation: Union[str, keras.layers.Layer] = 'gelu',
        block_normalization: Optional[str] = None,
        stem_activation: Union[str, keras.layers.Layer] = 'gelu',
        drop_path_rate: float = 0.1,
        final_activation: Union[str, callable] = 'linear',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'orthogonal',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        depthwise_initializer: Optional[Union[str, keras.initializers.Initializer]] = None,
        depthwise_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        dropout_rate: float = 0.0,
        enable_deep_supervision: bool = False,
        supervision_norm_scale: bool = True,
        supervision_norm_center: bool = False,
        supervision_activation: Union[str, keras.layers.Layer] = 'gelu',
        model_name: str = 'convunext'
) -> keras.Model:
    """Create the BIAS-FREE ConvUNext denoiser: ``create_convunext(..., use_bias=False)``.

    This is a thin wrapper that pins ``use_bias=False`` and forwards every other
    argument verbatim. The architecture, every parameter's meaning and the three
    documented asymmetries of the bias-free arm are described ONCE, on
    ``dl_techniques.models.convunext.model.create_convunext`` — read that
    docstring; this signature is deliberately identical to it minus ``use_bias``
    and ``stem_normalization`` (the stem is pinned to
    ``'global_response_norm'``, the ConvNeXt-V2 / bias-free choice).

    The signature is frozen: `src/train/bfunet/train_convunext_denoiser.py`,
    `utils/multiplicative_miyasawa.py` and the two bf test suites call it by
    keyword. Parameters are forwarded via a ``locals()`` capture taken as the
    FIRST statement, so a parameter can never be silently dropped from the
    forward (a missing one raises ``TypeError`` at the call instead).

    :param input_shape: Shape of input images ``(height, width, channels)``.
    :type input_shape: tuple of 3 ints
    :param block_normalization: ``None`` (the default) is a SENTINEL meaning
        "not chosen". It resolves to ``'layernorm'`` -- the historical default,
        so the built graph is unchanged -- and logs a warning that this
        builder's default is scale-INVARIANT (degree 0) and therefore breaks
        the ``f(a*x) = a*f(x)`` homogeneity the package name promises. Pass
        ``'batchnorm'`` for a homogeneous bias-free stack, or ``'layernorm'``
        explicitly to keep the historical graph without the warning.
    :type block_normalization: Optional[str]
    :return: A functional, bias-free ``keras.Model``.
    :rtype: keras.Model
    """
    # DECISION plan-2026-08-14T092357-0e3d792d/D-011: forward via a `locals()` capture
    # taken as the FIRST statement of the body, NOT by hand-listing ~40 keyword
    # arguments. The hand-listed form is the tempting "explicit" alternative and it is
    # exactly the failure this repo has already paid for: an omitted parameter becomes a
    # SILENT no-op (the caller's argument is accepted, then dropped on the floor), and no
    # test notices because the model still builds. With `locals()` a name that exists here
    # but not on `create_convunext` raises TypeError at the call — loud, immediate, and
    # impossible to ship. Do NOT "clean this up" into an explicit argument list, and do
    # NOT move any statement above it (that would sweep locals into the forward).
    forwarded = dict(locals())

    # DECISION plan-2026-08-18T140459-7991552f/D-048
    # `block_normalization` defaults to the `None` SENTINEL, not to the string
    # `'layernorm'`, so that "the caller did not choose" is distinguishable
    # from "the caller chose LayerNorm". The sentinel resolves to `'layernorm'`
    # -- the historical default -- so the built graph is byte-identical to
    # before; the only new thing is this warning.
    # Why it is needed even though `create_convunext` ALREADY warns: that guard
    # (`convunext/model.py:665`) fires for `block_normalization == 'layernorm'`
    # under `use_bias=False` whatever the provenance, so a caller who
    # deliberately passed `'layernorm'` gets nagged identically to one who
    # passed nothing at all. That makes the signal unactionable, and it is the
    # unchosen case that matters: this package's `__init__.py:6-8` states
    # "bias-free means ... degree-1 homogeneous", and LayerNorm is degree 0, so
    # the ONE exported builder whose default is not homogeneous is the one that
    # silently invalidates the Miyasawa residual-as-score reading that
    # `applications/bias_free_denoiser/denoiser_prior.py` and `ddnm.py` depend
    # on.
    # WHAT NOT TO DO: do NOT make the sentinel raise, and do NOT resolve it to
    # `'batchnorm'`. Both are behaviour changes on a frozen signature. Raising
    # breaks every omitted-kwarg caller, including
    # `utils/multiplicative_miyasawa.py:835` and ~20 test constructions;
    # resolving to `'batchnorm'` moves the shipped graph (and, at the batch
    # size that caller controls, BatchNorm at batch 1 is a real hazard), and it
    # would erase the very distinction that `create_convunext_variant`'s
    # `setdefault` exists to draw (D-014: batchnorm is selected at the VARIANT
    # wrapper and nowhere else). See decisions.md D-048.
    if block_normalization is None:
        forwarded['block_normalization'] = 'layernorm'
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

    return create_convunext(use_bias=False, **forwarded)


# ---------------------------------------------------------------------
# Variant Creation Functions
# ---------------------------------------------------------------------

def create_convunext_variant(
        variant: str,
        input_shape: Tuple[int, int, int],
        enable_deep_supervision: bool = True,
        **kwargs
) -> keras.Model:
    """Create a BIAS-FREE ConvUNext model from a named variant configuration.

    Forwards to ``models.convunext.model.create_convunext_variant`` with
    ``use_bias=False``. Note the default ``enable_deep_supervision=True`` here,
    which differs from the shared builder's ``False`` — it is the frozen historical
    signature of this bias-free entry point and is deliberately kept.

    **The named bias-free variants use ``block_normalization='batchnorm'``**, which
    is applied HERE as a ``setdefault`` — a caller-supplied value always wins. This
    is the ONE place in the repo where the bias-free arm departs from the shared
    builder's ``'layernorm'`` default; a bare
    :func:`create_convunext_denoiser` call still gets ``'layernorm'``. See the
    ``# DECISION`` anchor below for why the key is not in ``CONVUNEXT_CONFIGS``.

    :param variant: One of ``'tiny'``, ``'small'``, ``'base'``, ``'large'``,
        ``'xlarge'`` (keys of the shared ``CONVUNEXT_CONFIGS``).
    :type variant: str
    :param input_shape: Shape of input images ``(height, width, channels)``.
    :type input_shape: tuple of 3 ints
    :param enable_deep_supervision: Whether to enable deep-supervision outputs.
        Defaults to True.
    :type enable_deep_supervision: bool
    :param kwargs: Additional keyword arguments overriding the variant defaults.
    :return: A functional, bias-free ``keras.Model``.
    :rtype: keras.Model
    :raises ValueError: If ``variant`` is not recognized.

    Example::

        >>> model = create_convunext_variant('base', (256, 256, 3),
        ...                                  enable_deep_supervision=True)
        >>> inference = create_convunext_variant('base', (None, None, 3),
        ...                                      enable_deep_supervision=False,
        ...                                      convnext_version='v1')
    """
    kwargs.setdefault('use_bias', False)
    # DECISION plan-2026-08-14T092357-0e3d792d/D-014: 'batchnorm' is selected HERE, at
    # the bias-free VARIANT wrapper, and nowhere else.
    #   * Do NOT move this key into the shared `CONVUNEXT_CONFIGS` dict — that dict feeds
    #     BOTH arms, so the bias-ON variants would flip to batchnorm too, which the user
    #     explicitly does not want.
    #   * Do NOT change `create_convunext`'s own `block_normalization` default (both arms
    #     stay 'layernorm'). Two things pin that: the byte-identity tripwire in
    #     `test_bfconvunext_denoiser.py` (omitted kwarg must equal an explicit
    #     'layernorm'), and `utils/multiplicative_miyasawa.py`'s omitted-kwarg call, whose
    #     caller controls the batch size — batchnorm at batch 1 is a real hazard there.
    #   * Do NOT promote this to an unconditional assignment. `setdefault` is load-bearing:
    #     a caller passing `block_normalization='layernorm'` must keep it.
    # Rationale: `ddnm.py` documents that a bias-free ConvUNext is degree-1 homogeneous
    # only under batchnorm, so the NAMED bias-free variants get it; the raw builder call
    # keeps its historical graph. decisions.md D-003 (ruling) and D-014 (implementation).
    kwargs.setdefault('block_normalization', 'batchnorm')
    return _create_convunext_variant(
        variant=variant,
        input_shape=input_shape,
        enable_deep_supervision=enable_deep_supervision,
        **kwargs
    )


# ---------------------------------------------------------------------
# Utility Functions for Deep Supervision
# ---------------------------------------------------------------------

from dl_techniques.utils.deep_supervision import (  # noqa: F401
    get_model_output_info,
    create_inference_model_from_training_model,
)

# ---------------------------------------------------------------------
