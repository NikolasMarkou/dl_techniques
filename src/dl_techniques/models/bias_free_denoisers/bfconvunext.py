"""Bias-free ConvUNext: the ``use_bias=False`` arm of the merged ConvUNext builder.

This module is TWO things and nothing else:

1. **Thin wrappers.** ``create_convunext_denoiser`` (a ``**kwargs`` delegator) and
   ``create_convunext_variant`` (its historical signature) pin ``use_bias=False`` and
   forward to ``dl_techniques.models.convunext.model.create_convunext``. The architecture, every
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

References:
    - Mohan et al., 2020. Robust and Interpretable Blind Image Denoising via
      Bias-Free Convolutional Neural Networks. ICLR 2020.
      (https://arxiv.org/abs/1906.05478) -- why ``use_bias=False`` is the whole
      point of this arm.
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
# Registrar re-exports from the shared ConvUNext builder
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

# `SpatialLinearAttention` moved with a BARE `@keras.saving.register_keras_serializable()`,
# whose key `Custom>SpatialLinearAttention` was MEASURED to be module-independent on
# Keras 3.8.0 (decisions.md D-008), so the move did not change it. Do NOT add a
# `package=` argument "for symmetry" with `ConvUNextStem` — that WOULD change the key.
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
        *,
        block_normalization: Optional[str] = None,
        **kwargs: Any
) -> keras.Model:
    """Create the BIAS-FREE ConvUNext denoiser: ``create_convunext(..., use_bias=False)``.

    A thin DELEGATOR. Every keyword other than ``input_shape`` and
    ``block_normalization`` is forwarded VERBATIM to
    ``dl_techniques.models.convunext.model.create_convunext``, which is where the
    architecture, every parameter's meaning and the three documented asymmetries
    of the bias-free arm are described ONCE. That docstring is the full parameter
    reference; it is deliberately not restated here, and this function enumerates
    nothing, so a parameter added there is reachable here the same day.

    Two arguments are pinned by this wrapper and therefore cannot be forwarded:
    ``use_bias=False`` (the whole point of the bias-free arm) and
    ``stem_normalization='global_response_norm'`` (the ConvNeXt-V2 / bias-free
    choice). Passing either raises ``TypeError: ... got multiple values ...`` —
    the pin cannot be overridden by accident, silently or otherwise.

    ``create_convunext`` declares no ``**kwargs``, so an unknown keyword raises
    ``TypeError`` at the delegation, naming the offending keyword and the callee
    (``create_convunext()``, not this wrapper).

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
    :param kwargs: Every remaining parameter of
        :func:`dl_techniques.models.convunext.model.create_convunext`, forwarded
        unchanged — including ``include_top`` and ``output_channels``, which the
        old hand-copied signature did not expose at all.
    :return: A functional, bias-free ``keras.Model``.
    :rtype: keras.Model
    :raises TypeError: If a keyword is not a ``create_convunext`` parameter, or if
        it duplicates ``use_bias``/``stem_normalization``/``input_shape``, which
        this wrapper supplies itself.
    """
    # SUPERSEDED 2026-08-24 (plan-2026-08-24T120026-64ffd751/D-010): the `locals()`
    # capture is GONE, and with it the 38-of-42 hand-copied signature it fed. The
    # guarantee below is NOT weakened — it is now enforced by Python itself, and it
    # now covers the SIGNATURE too, which the capture never could: `create_convunext`
    # declares 42 explicit parameters and NO `**kwargs` (MEASURED at iter-2/step-2:
    # `inspect.signature(create_convunext)` has no VAR_KEYWORD), so an unknown
    # forwarded name raises TypeError at the call — loud, immediate, impossible to
    # ship. Pinned by tests/test_models/test_bias_free_denoisers/
    # test_the_bfconvunext_delegation_contract.py::TestUnknownKwargIsLoud.
    # DECISION plan-2026-08-14T092357-0e3d792d/D-011: forward via a `locals()` capture
    # taken as the FIRST statement of the body, NOT by hand-listing ~40 keyword
    # arguments. The hand-listed form is the tempting "explicit" alternative and it is
    # exactly the failure this repo has already paid for: an omitted parameter becomes a
    # SILENT no-op (the caller's argument is accepted, then dropped on the floor), and no
    # test notices because the model still builds. With `locals()` a name that exists here
    # but not on `create_convunext` raises TypeError at the call — loud, immediate, and
    # impossible to ship. Do NOT "clean this up" into an explicit argument list, and do
    # NOT move any statement above it (that would sweep locals into the forward).

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

    # DECISION plan-2026-08-24T120026-64ffd751/D-010: delegate with `**kwargs`, and pin
    # `use_bias` / `stem_normalization` HERE as explicit arguments of THIS call.
    #   * Do NOT re-add the hand-copied parameter list. It duplicated
    #     `create_convunext`'s signature with no parity test and had already lost
    #     `include_top` and `output_channels` — 38 of 42 — so a bias-free feature
    #     extractor could not be built through this entry point at all.
    #   * Do NOT demote the two pins to `kwargs.setdefault(...)`. A setdefault lets a
    #     caller pass `use_bias=True` and get a BIASED model back from the bias-free
    #     builder silently, which breaks f(a*x) = a*f(x) and the Miyasawa
    #     residual-as-score reading `denoiser_prior.py` and `ddnm.py` depend on. As
    #     arguments they raise `TypeError: got multiple values` instead.
    #   * `stem_normalization` is pinned even though it equals `create_convunext`'s
    #     current default: the docstring's "the stem is pinned" claim was inherited,
    #     not enforced, until this line. See decisions.md D-010.
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
        ``use_bias`` is NOT among them: it is pinned to ``False`` by this wrapper and
        passing it raises ``TypeError: ... got multiple values ...``.
    :return: A functional, bias-free ``keras.Model``.
    :rtype: keras.Model
    :raises ValueError: If ``variant`` is not recognized.
    :raises TypeError: If ``use_bias`` is passed (this wrapper supplies it), or if a
        keyword is not a ``create_convunext`` parameter.

    Example::

        >>> model = create_convunext_variant('base', (256, 256, 3),
        ...                                  enable_deep_supervision=True)
        >>> inference = create_convunext_variant('base', (None, None, 3),
        ...                                      enable_deep_supervision=False,
        ...                                      convnext_version='v1')
    """
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

    # DECISION plan-2026-08-24T120026-64ffd751/D-013: `use_bias=False` is pinned as an
    # explicit ARGUMENT of this call, exactly as in `create_convunext_denoiser` above.
    #   * Do NOT demote it back to `kwargs.setdefault('use_bias', False)`, which is what
    #     it was until 2026-08-24. A setdefault lets a caller-supplied value WIN:
    #     MEASURED at f390e123d, `create_convunext_variant('tiny', shape, use_bias=True)`
    #     returned a model with 54 bias tensors / 29 biased layers out of the BIAS-FREE
    #     entry point, silently, while the sibling arm raised. As an argument it raises
    #     `TypeError: got multiple values` instead. Pinned by
    #     tests/test_models/test_bias_free_denoisers/
    #     test_the_bfconvunext_delegation_contract.py::TestPinnedKwargsCannotBeOverridden
    #     ::test_variant_wrapper_also_refuses_use_bias_true.
    #   * Do NOT "harmonize" this with the `block_normalization` setdefault three lines
    #     above by promoting THAT one too. D-014 states the opposite for that key --
    #     `setdefault` there is load-bearing ("a caller passing
    #     `block_normalization='layernorm'` must keep it"). The two keys have opposite
    #     override semantics on purpose. See decisions.md D-013.
    return _create_convunext_variant(
        variant=variant,
        input_shape=input_shape,
        enable_deep_supervision=enable_deep_supervision,
        use_bias=False,
        **kwargs
    )
