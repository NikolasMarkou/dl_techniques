"""Config-driven construction of the normalization layers in this package.

``create_normalization_layer(type_string, **kwargs)`` builds one of **18**
registered types. Two come from Keras (``layer_norm``, ``batch_norm``); the other
sixteen are classes from this package. The registry is ``_TYPE_TO_CLASS``, and the
same 18 names are spelled out in the ``NormalizationType`` ``Literal``.

Three things a caller has to know before using this module.

**1. The factory imposes its own epsilon.** Every type that takes an ``epsilon``
gets ``1e-6`` unless the caller passes one. For 11 of the 16 bare-constructible
types that is not the class's own default. Rewriting
``keras.layers.BatchNormalization()`` as ``create_normalization_layer('batch_norm')``
divides epsilon by 1000. The full measured table and the reasoning are in
``create_normalization_layer``'s own ``.. warning::``. Instantiate the class
directly if you want its own default.

**2. Three types are not shape-preserving, and two of those do not return a single
tensor.** Measured on a ``(3, 5, 8)`` input at the default ``axis=-1``:
``decoupled_max_logit`` returns a 3-tuple of ``(3, 5)`` tensors,
``dml_plus_center`` a 2-tuple shaped ``(3, 5)`` and ``(3, 5, 1)``, and
``dml_plus_focal`` a single ``(3, 5)`` tensor. A config-driven caller that swaps
``layer_norm`` for one of these does not get a drop-in substitute.
``max_logit_norm`` IS shape-preserving; ``(3, 5, 8)`` stays ``(3, 5, 8)``.

**3. ``get_normalization_info()[t]['parameters']`` is documentation, not a
whitelist.** The validator derives its accepted set from the target class's real
constructor signature, in ``_accepted_params``. Measured at HEAD: for all 18 of 18
types the curated list omits at least one kwarg the factory accepts, 107 such
``(type, kwarg)`` pairs in total. Using the curated list as the whitelist is
exactly the bug that was fixed, twice.

Public surface:

* :func:`create_normalization_layer` -- the builder.
* :func:`create_normalization_from_config` -- the same builder from a ``dict``
  carrying a ``'type'`` key.
* :func:`validate_normalization_config` -- raises on a kwarg the target type does
  not accept. The builder calls it for every known type.
* :func:`get_normalization_info` -- per-type description, curated parameter list
  and use case, for documentation and UI purposes.

The registry key set and the ``NormalizationType`` aliases are public API. Adding,
renaming or removing one is a breaking change, and
``tests/test_layers/test_norms/test_factory.py`` plus
``tests/test_layers/test_norms/test_the_norm_factory_family_is_pinned.py`` assert
it.
"""

import keras
import inspect
from typing import Optional, Dict, Any, Literal, Set

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .rms_norm import RMSNorm
from .bias_free_batch_norm import BiasFreeBatchNorm
from .band_rms import BandRMS
from .adaptive_band_rms import AdaptiveBandRMS
from .band_logit_norm import BandLogitNorm
from .global_response_norm import GlobalResponseNormalization
from .logit_norm import LogitNorm
from .max_logit_norm import MaxLogitNorm, DecoupledMaxLogit, DMLPlus
from .dynamic_tanh import DynamicTanh
from .energy_layer_norm import EnergyLayerNorm
from .zero_centered_rms_norm import ZeroCenteredRMSNorm
from .zero_centered_band_rms_norm import ZeroCenteredBandRMSNorm
from .zero_centered_adaptive_band_rms_norm import ZeroCenteredAdaptiveBandRMS

# ---------------------------------------------------------------------

NormalizationType = Literal[
    'layer_norm', 'batch_norm', 'bias_free_batch_norm', 'rms_norm', 'zero_centered_rms_norm',
    'zero_centered_band_rms_norm', 'zero_centered_adaptive_band_rms_norm',
    'band_rms', 'adaptive_band_rms',
    'band_logit_norm', 'global_response_norm', 'logit_norm', 'max_logit_norm',
    'decoupled_max_logit', 'dml_plus_focal', 'dml_plus_center', 'dynamic_tanh',
    'energy_layer_norm'
]

# ---------------------------------------------------------------------
# Validation whitelist — DERIVED from the real constructors, never hand-maintained.
# ---------------------------------------------------------------------

# The class each type actually instantiates. 18 keys, matching the 18 names in
# `NormalizationType`. Single source of truth, kept honest by
# `test_type_to_class_matches_what_the_builder_returns` in
# tests/test_layers/test_norms/test_factory.py, which builds every type and compares
# `type(layer)` against this map. It therefore cannot silently drift from the
# if/elif chain in `create_normalization_layer`.
_TYPE_TO_CLASS: Dict[str, type] = {
    'layer_norm': keras.layers.LayerNormalization,
    'batch_norm': keras.layers.BatchNormalization,
    'bias_free_batch_norm': BiasFreeBatchNorm,
    'rms_norm': RMSNorm,
    'zero_centered_rms_norm': ZeroCenteredRMSNorm,
    'zero_centered_band_rms_norm': ZeroCenteredBandRMSNorm,
    'zero_centered_adaptive_band_rms_norm': ZeroCenteredAdaptiveBandRMS,
    'band_rms': BandRMS,
    'adaptive_band_rms': AdaptiveBandRMS,
    'band_logit_norm': BandLogitNorm,
    'global_response_norm': GlobalResponseNormalization,
    'logit_norm': LogitNorm,
    'max_logit_norm': MaxLogitNorm,
    'decoupled_max_logit': DecoupledMaxLogit,
    'dml_plus_focal': DMLPlus,
    'dml_plus_center': DMLPlus,
    'dynamic_tanh': DynamicTanh,
    'energy_layer_norm': EnergyLayerNorm,
}

# Named parameters of `create_normalization_layer` ITSELF, valid for every type.
# `epsilon` is universal on purpose. The factory takes it for all 18 types and adapts
# it per layer: aliased to `eps` for global_response_norm, popped for dynamic_tanh.
# Rejecting it for those two types made the validator disagree with the builder.
_FACTORY_LEVEL_PARAMS = frozenset({'name', 'epsilon'})

# Base `keras.layers.Layer` kwargs. Every target class takes `**kwargs` and forwards them,
# so these genuinely build; the validator must not reject what the builder accepts.
_KERAS_BASE_PARAMS = frozenset({'name', 'dtype', 'trainable', 'activity_regularizer', 'autocast'})

# Parameters the factory IGNORES. The governing rule is "the validator must accept
# whatever the builder accepts". But building without raising is not the same as
# having an effect. These two sets are the params a caller may pass, that construct
# fine, and that the factory then throws away. Rejecting them is CORRECT and is not
# the drift that was fixed. It tells a caller their value is doing nothing, instead
# of ignoring it silently.
#
# OVERWRITTEN: `create_normalization_layer` hard-assigns `model_type` for the two DML+
# variants, so a caller's value is clobbered.
_FACTORY_OWNED_PARAMS: Dict[str, frozenset] = {
    'dml_plus_focal': frozenset({'model_type'}),
    'dml_plus_center': frozenset({'model_type'}),
}

# DISCARDED: `DynamicTanh` has no epsilon, and the factory `pop`s it. A config-driven
# caller who sets `epsilon=1e-3` here and is not told would reasonably believe it
# applied. Contrast `global_response_norm`, which ALIASES `epsilon` to `eps`. There the
# value is meaningful, so it is accepted. Rejecting it there WAS drift, and is fixed.
_FACTORY_DROPPED_PARAMS: Dict[str, frozenset] = {
    'dynamic_tanh': frozenset({'epsilon'}),
}


def _accepted_params(normalization_type: str) -> Set[str]:
    """Return every kwarg ``create_normalization_layer`` accepts for one type.

    The set is DERIVED from ``inspect.signature`` of the target class. It is not a
    hand-maintained list. That distinction is the whole reason this function exists.
    A hand-kept whitelist drifts the moment someone adds a constructor argument, and
    the validator then rejects a parameter the builder accepts. That happened twice:
    once on the band and GRN initializers, and once on ``gamma_constraint`` for
    ``energy_layer_norm``. Both originating plan directories have been reaped. The
    second case is still recorded in this file, by the DECISION anchor comment inside
    ``get_normalization_info``'s ``'energy_layer_norm'`` entry.

    The set is built in four steps: the target class's own named constructor
    parameters, plus ``_FACTORY_LEVEL_PARAMS``, plus ``_KERAS_BASE_PARAMS``, minus
    ``_FACTORY_OWNED_PARAMS`` and ``_FACTORY_DROPPED_PARAMS`` for that type.

    ``get_normalization_info()[t]['parameters']`` is a separate, curated
    DOCUMENTATION list of the parameters people commonly pass. It is no longer the
    validation whitelist, so it can be incomplete without breaking a caller. Measured
    at HEAD, it is incomplete for all 18 of 18 types, by 107 ``(type, kwarg)`` pairs.

    :param normalization_type: A registered normalization type. Must be a key of
        ``_TYPE_TO_CLASS``.
    :type normalization_type: str
    :return: The set of accepted keyword-argument names.
    :rtype: Set[str]
    :raises KeyError: If ``normalization_type`` is not a registered type. Callers
        check membership first; ``validate_normalization_config`` raises its own
        ``ValueError`` before reaching here.
    """
    cls = _TYPE_TO_CLASS[normalization_type]
    signature = inspect.signature(cls.__init__)
    named = {
        name for name, param in signature.parameters.items()
        if name != 'self'
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    named |= _FACTORY_LEVEL_PARAMS
    named |= _KERAS_BASE_PARAMS
    named -= _FACTORY_OWNED_PARAMS.get(normalization_type, frozenset())
    named -= _FACTORY_DROPPED_PARAMS.get(normalization_type, frozenset())
    return named


# ---------------------------------------------------------------------

def create_normalization_layer(
        normalization_type: NormalizationType,
        name: Optional[str] = None,
        epsilon: float = 1e-6,
        **kwargs: Any
) -> keras.layers.Layer:
    """Build one of the 18 registered normalization layers.

    One entry point for the two Keras normalization layers and the sixteen from this
    package. Unknown kwargs raise: for every registered type the builder calls
    :func:`validate_normalization_config` first, and that rejects any keyword the
    target class does not declare.

    **Dispatch path:**

    .. code-block:: text

        create_normalization_layer(normalization_type, name,
                                   epsilon, **kwargs)
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ if normalization_type in _TYPE_TO_CLASS:                │
        │     validate_normalization_config(type, **kwargs)       │
        │ an UNKNOWN type skips this and reaches the else arm,    │
        │ which raises with the supported-type list               │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │ layer_kwargs = kwargs.copy()                            │
        │ layer_kwargs['name'] = name   (only when name is given) │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
              the if/elif chain: one arm per registry type
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
          16 arms:            1 arm:                1 arm:
          setdefault          global_               dynamic_tanh
          ('epsilon',         response_norm         pops
           epsilon)           sets eps =            'epsilon'
                              epsilon when          (DynamicTanh
          2 of them           'eps' is not          has none)
          (dml_plus_*)        already given
          also force
          model_type

          Every arm returns <target class>(**layer_kwargs).

    .. warning::

       **THIS IS NOT A DROP-IN REPLACEMENT FOR CONSTRUCTING THE LAYER DIRECTLY.**
       The factory imposes its own ``epsilon=1e-6`` on every type that accepts one
       (via ``setdefault``), which for **11 of the 16 bare-constructible registry
       types** is NOT the value that class would have chosen for itself.
       MEASURED 2026-08-23 on keras 3.8.0, factory default vs class own default:

       ============================================  =========  =========  =========
       ``normalization_type``                        factory    class own  ratio
       ============================================  =========  =========  =========
       ``batch_norm``, ``layer_norm``                ``1e-6``   ``1e-3``   **1000x**
       ``energy_layer_norm``                         ``1e-6``   ``1e-5``   10x
       ``band_rms``, ``adaptive_band_rms``,          ``1e-6``   ``1e-7``   0.1x
       ``band_logit_norm``, ``logit_norm``,
       ``max_logit_norm``, ``decoupled_max_logit``,
       ``zero_centered_band_rms_norm``,
       ``zero_centered_adaptive_band_rms_norm``
       ``rms_norm``, ``zero_centered_rms_norm``,     ``1e-6``   ``1e-6``   1x
       ``bias_free_batch_norm``,
       ``global_response_norm``
       ``dynamic_tanh``                              (popped)   n/a        n/a
       ``dml_plus_focal``, ``dml_plus_center``       ``1e-6``   n/a        not bare
       ============================================  =========  =========  =========

       So **rewriting ``keras.layers.BatchNormalization()`` as
       ``create_normalization_layer('batch_norm')`` divides epsilon by 1000**,
       silently, and changes inference on every affected layer. This is not
       hypothetical. It was proposed for ``mobilenet`` and ``cbam`` (189 layers)
       and rejected on this measurement. The record is
       ``plans/plan-2026-08-22T035419-a11304c8/decisions.md`` D-202. The guard is
       ``tests/test_layers/test_norms/test_the_norm_factory_family_is_pinned.py::
       TestTheFactoryEpsilonIsNotTheLayerDefault``.

       If you want a layer's own default, **instantiate the class directly**.
       There is no ``epsilon=None`` sentinel meaning "use the class default". That
       was proposed and rejected; D-202 records why.

    :param normalization_type: Which of the 18 registered types to build. The names
        are 'layer_norm', 'batch_norm', 'bias_free_batch_norm', 'rms_norm',
        'zero_centered_rms_norm', 'zero_centered_band_rms_norm',
        'zero_centered_adaptive_band_rms_norm', 'band_rms', 'adaptive_band_rms',
        'band_logit_norm', 'global_response_norm', 'logit_norm', 'max_logit_norm',
        'decoupled_max_logit', 'dml_plus_focal', 'dml_plus_center', 'dynamic_tanh'
        and 'energy_layer_norm'. Use 'bias_free_batch_norm' for the variance-only,
        fixed-statistic layer that stays degree-1 homogeneous (``f(a*x) = a*f(x)``)
        at inference, with no ``moving_mean`` and no ``beta``. Bias-free and
        Miyasawa denoisers need that.
    :type normalization_type: NormalizationType
    :param name: Layer name. When ``None`` the layer takes Keras' default naming.
    :type name: Optional[str]
    :param epsilon: Constant for numerical stability, defaulting to 1e-6. Applied
        with ``setdefault``, so an explicit ``epsilon`` in ``kwargs`` wins. This
        1e-6 is the FACTORY's choice and differs from most classes' own defaults.
        Read the warning above before relying on it.
    :type epsilon: float
    :param kwargs: Keyword arguments forwarded to the target class. Common ones are
        ``axis``, ``center``, ``scale``, ``use_scale``, ``momentum`` (batch_norm and
        bias_free_batch_norm), ``max_band_width``, ``temperature``, ``constant``,
        ``alpha_init_value`` and ``eps`` (global_response_norm). Any keyword the
        target class does not declare raises.
    :return: The configured layer. **Three types are not shape-preserving, and two
        of those do not return a single tensor.** Measured on a ``(3, 5, 8)`` input
        at the default ``axis=-1``: ``decoupled_max_logit`` returns a **3-tuple**
        ``(combined score, MaxCosine, MaxNorm)`` of ``(3, 5)`` tensors;
        ``dml_plus_center`` returns a **2-tuple** ``(MaxNorm score, norm factor)``
        shaped ``(3, 5)`` and ``(3, 5, 1)``; ``dml_plus_focal`` returns a
        **single** ``(3, 5)`` tensor. All three reduce the normalized axis away.
        ``max_logit_norm`` does NOT: it reduces with ``keepdims=True`` and then
        divides, so ``(3, 5, 8)`` stays ``(3, 5, 8)``. ``norms/README.md`` carries
        the same table.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``normalization_type`` is not registered. The message
        names every supported type. Measured: ``'definitely_not_a_type'`` gives
        ``Unknown normalization type: 'definitely_not_a_type'. Supported types:
        layer_norm, batch_norm, ...``.
    :raises ValueError: If a kwarg is not accepted for this type. Measured:
        ``create_normalization_layer('layer_norm', bogus=1)`` gives ``Invalid
        parameters for layer_norm: {'bogus'}. Valid parameters: [...]``, listing the
        accepted set. Raised by :func:`validate_normalization_config`.
    :raises ValueError: If a type-specific value is out of range, for example
        ``max_band_width`` outside ``(0, 1)`` or a non-positive ``temperature``.
        Raised by :func:`validate_normalization_config`.
    :raises TypeError: If the target class rejects a kwarg's type.

    Example:

    .. code-block:: python

        from dl_techniques.layers.norms import create_normalization_layer

        norm = create_normalization_layer('rms_norm', epsilon=1e-8, use_scale=True)
        keras_ln = create_normalization_layer('layer_norm', epsilon=1e-3)
    """
    # DECISION plan-2026-08-18T140459-7991552f/D-017
    # The builder must call the validator. `validate_normalization_config` sat in this
    # module for two plans with no caller, so the only thing between a caller and a
    # wrong layer was whatever the target class did with the key. Keras 3 catches most
    # bad keys itself with "Unrecognized keyword arguments". What it does NOT catch is
    # the keys the target class accepts and this factory then throws away, enumerated
    # by `_FACTORY_OWNED_PARAMS` and `_FACTORY_DROPPED_PARAMS`.
    # WITHOUT the two lines below, `create_normalization_layer('dml_plus_focal',
    # model_type='center')` built a layer whose `model_type` was 'focal'. WITH them it
    # raises `ValueError: Invalid parameters for dml_plus_focal: {'model_type'}`, and
    # that is what HEAD does. The raise is the fix working, not a stale comment.
    # DO NOT replace `_accepted_params` with a hand-written whitelist. It derives from
    # the target class's real `inspect.signature`, and a hand-kept list drifted twice.
    # DO NOT move this call into the if/elif chain; it must run for every arm.
    # Unknown types skip validation and reach the `else` arm, whose error names the
    # supported types. The validator's own unknown-type message does not, and tests
    # pin the richer one. Pre-flight before this edit: 200 call sites repo-wide, zero
    # passing a kwarg outside `_accepted_params`, under both a static AST sweep and a
    # runtime recorder over the 11,031 tests of tests/test_layers/. The one runtime hit
    # was a test feeding garbage and asserting a raise. See that plan's decisions.md
    # D-016.
    if normalization_type in _TYPE_TO_CLASS:
        validate_normalization_config(normalization_type, **kwargs)

    # Prepare base parameters
    layer_kwargs = kwargs.copy()
    if name is not None:
        layer_kwargs['name'] = name

    # Create the appropriate normalization layer
    if normalization_type == 'layer_norm':
        # Standard Keras LayerNormalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return keras.layers.LayerNormalization(**layer_kwargs)

    elif normalization_type == 'batch_norm':
        # Standard Keras BatchNormalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return keras.layers.BatchNormalization(**layer_kwargs)

    elif normalization_type == 'bias_free_batch_norm':
        # Variance-only, fixed-statistic normalization; degree-1 homogeneous at
        # inference (no moving_mean, no beta). See bias_free_batch_norm.py.
        layer_kwargs.setdefault('epsilon', epsilon)
        return BiasFreeBatchNorm(**layer_kwargs)

    elif normalization_type == 'rms_norm':
        # Root Mean Square normalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return RMSNorm(**layer_kwargs)

    elif normalization_type == 'zero_centered_rms_norm':
        # Zero-centered RMS normalization with enhanced stability
        layer_kwargs.setdefault('epsilon', epsilon)
        return ZeroCenteredRMSNorm(**layer_kwargs)

    elif normalization_type == 'zero_centered_band_rms_norm':
        # Zero-centered RMS with band constraints
        layer_kwargs.setdefault('epsilon', epsilon)
        return ZeroCenteredBandRMSNorm(**layer_kwargs)

    elif normalization_type == 'zero_centered_adaptive_band_rms_norm':
        # Zero-centered adaptive RMS with log-transformed scaling
        layer_kwargs.setdefault('epsilon', epsilon)
        return ZeroCenteredAdaptiveBandRMS(**layer_kwargs)

    elif normalization_type == 'band_rms':
        # RMS normalization with bounded constraints
        layer_kwargs.setdefault('epsilon', epsilon)
        return BandRMS(**layer_kwargs)

    elif normalization_type == 'adaptive_band_rms':
        # Adaptive RMS with log-transformed scaling
        layer_kwargs.setdefault('epsilon', epsilon)
        return AdaptiveBandRMS(**layer_kwargs)

    elif normalization_type == 'band_logit_norm':
        # Band-constrained logit normalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return BandLogitNorm(**layer_kwargs)

    elif normalization_type == 'global_response_norm':
        # Global Response Normalization (GRN)
        # GRN uses 'eps' instead of 'epsilon'
        if 'eps' not in layer_kwargs:
            layer_kwargs['eps'] = epsilon
        return GlobalResponseNormalization(**layer_kwargs)

    elif normalization_type == 'logit_norm':
        # LogitNorm for classification tasks
        layer_kwargs.setdefault('epsilon', epsilon)
        return LogitNorm(**layer_kwargs)

    elif normalization_type == 'max_logit_norm':
        # MaxLogit normalization for OOD detection
        layer_kwargs.setdefault('epsilon', epsilon)
        return MaxLogitNorm(**layer_kwargs)

    elif normalization_type == 'decoupled_max_logit':
        # Decoupled MaxLogit (DML) normalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return DecoupledMaxLogit(**layer_kwargs)

    elif normalization_type == 'dml_plus_focal':
        # DML+ focal model variant
        layer_kwargs.setdefault('epsilon', epsilon)
        layer_kwargs['model_type'] = 'focal'
        return DMLPlus(**layer_kwargs)

    elif normalization_type == 'dml_plus_center':
        # DML+ center model variant
        layer_kwargs.setdefault('epsilon', epsilon)
        layer_kwargs['model_type'] = 'center'
        return DMLPlus(**layer_kwargs)

    elif normalization_type == 'dynamic_tanh':
        # Dynamic Tanh normalization (normalization-free transformers)
        # DynamicTanh doesn't use epsilon, remove it if present
        layer_kwargs.pop('epsilon', None)
        return DynamicTanh(**layer_kwargs)

    elif normalization_type == 'energy_layer_norm':
        # Energy Transformer layer norm (arXiv:2302.07253 eq. 1-2):
        # SCALAR gamma + VECTOR delta. See energy_layer_norm.py.
        layer_kwargs.setdefault('epsilon', epsilon)
        return EnergyLayerNorm(**layer_kwargs)

    else:
        supported_types = [
            'layer_norm', 'batch_norm', 'bias_free_batch_norm', 'rms_norm',
            'zero_centered_rms_norm',
            'zero_centered_band_rms_norm',
            'zero_centered_adaptive_band_rms_norm',
            'band_rms', 'adaptive_band_rms',
            'band_logit_norm', 'global_response_norm', 'logit_norm',
            'max_logit_norm', 'decoupled_max_logit', 'dml_plus_focal',
            'dml_plus_center', 'dynamic_tanh', 'energy_layer_norm'
        ]
        raise ValueError(
            f"Unknown normalization type: '{normalization_type}'. "
            f"Supported types: {', '.join(supported_types)}"
        )


# ---------------------------------------------------------------------


def get_normalization_info() -> Dict[str, Dict[str, Any]]:
    """Describe all 18 registered normalization types.

    Each entry carries a ``'description'``, a ``'parameters'`` list and a
    ``'use_case'`` string. Intended for documentation, help text and configuration
    UIs.

    .. warning::
       The ``'parameters'`` list is **documentation**, a curated set of the
       parameters callers commonly pass. It is **NOT** the validation whitelist and
       is **NOT** exhaustive. Measured at HEAD: for all 18 of 18 types it omits at
       least one kwarg the factory accepts, 107 such ``(type, kwarg)`` pairs in
       total. ``layer_norm`` alone omits 12 and ``batch_norm`` 14, because both
       accept every Keras ``LayerNormalization`` / ``BatchNormalization`` kwarg.
       Nothing in any list is rejected: measured, 0 documented-but-rejected pairs.

       For the accepted set, call :func:`validate_normalization_config`, whose
       whitelist is DERIVED from the target class's real constructor signature. See
       ``_accepted_params``. A hand-maintained whitelist drifted twice and made the
       validator reject parameters the builder accepts. Deriving it makes that bug
       unrepresentable.

    :return: A dict keyed by normalization type name. Each value is a dict with
        ``'description'``, ``'parameters'`` and ``'use_case'`` keys.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {
        'layer_norm': {
            'description': 'Standard Keras LayerNormalization with learnable scale and bias',
            'parameters': ['axis', 'epsilon', 'center', 'scale'],
            'use_case': 'General purpose normalization for transformers and deep networks'
        },
        'batch_norm': {
            'description': 'Standard Keras BatchNormalization with moving statistics',
            'parameters': ['axis', 'epsilon', 'center', 'scale', 'momentum'],
            'use_case': 'Convolutional networks and batch-based training'
        },
        'bias_free_batch_norm': {
            'description': 'Variance-only, fixed-statistic normalization (no moving_mean, no beta); degree-1 homogeneous at inference',
            'parameters': ['axis', 'epsilon', 'momentum', 'use_scale'],
            'use_case': 'Bias-free / homogeneous architectures (e.g. Miyasawa denoisers) requiring f(a*x)=a*f(x) at inference'
        },
        'rms_norm': {
            'description': 'Root Mean Square normalization without centering',
            'parameters': ['axis', 'epsilon', 'use_scale', 'scale_initializer'],
            'use_case': 'Transformers, especially for faster training and inference'
        },
        'zero_centered_rms_norm': {
            'description': 'Zero-centered RMS normalization combining RMSNorm efficiency with LayerNorm stability',
            'parameters': ['axis', 'epsilon', 'use_scale', 'scale_initializer'],
            'use_case': 'Large language models and transformers requiring enhanced training stability'
        },
        'zero_centered_band_rms_norm': {
            'description': 'Combines zero-centering, RMS, and band constraints for maximum stability',
            'parameters': ['max_band_width', 'axis', 'epsilon', 'band_initializer', 'band_regularizer'],
            'use_case': 'Advanced transformer and LLM architectures for ultimate stability and flexibility'
        },
        'zero_centered_adaptive_band_rms_norm': {
            'description': 'Zero-centered RMS with adaptive log-transformed RMS-based scaling',
            'parameters': ['max_band_width', 'axis', 'epsilon', 'band_initializer', 'band_regularizer'],
            'use_case': 'Advanced training stability combining zero-centering with input-adaptive scaling'
        },
        'band_rms': {
            'description': 'RMS normalization with bounded magnitude constraints',
            'parameters': ['max_band_width', 'axis', 'epsilon', 'band_initializer', 'band_regularizer'],
            'use_case': 'Training stability in deep networks with gradient control'
        },
        'adaptive_band_rms': {
            'description': 'Adaptive RMS with log-transformed RMS-based scaling',
            'parameters': ['max_band_width', 'axis', 'epsilon', 'band_initializer', 'band_regularizer'],
            'use_case': 'Advanced training stability with adaptive scaling'
        },
        'band_logit_norm': {
            'description': 'Band-constrained logit normalization for classification',
            'parameters': ['max_band_width', 'axis', 'epsilon'],
            'use_case': 'Classification tasks with logit magnitude control'
        },
        'global_response_norm': {
            'description': 'Global Response Normalization from ConvNeXt',
            'parameters': ['eps', 'gamma_initializer', 'beta_initializer',
                           'gamma_regularizer', 'beta_regularizer', 'activity_regularizer'],
            'use_case': 'ConvNeXt-style architectures and vision_heads models'
        },
        'logit_norm': {
            'description': 'Temperature-scaled normalization for classification',
            'parameters': ['temperature', 'axis', 'epsilon'],
            'use_case': 'Classification with calibrated confidence estimates'
        },
        'max_logit_norm': {
            'description': 'MaxLogit normalization for out-of-distribution detection',
            'parameters': ['axis', 'epsilon'],
            'use_case': 'OOD detection and uncertainty estimation'
        },
        'decoupled_max_logit': {
            'description': 'Decoupled MaxLogit (DML) with constant decoupling',
            'parameters': ['constant', 'axis', 'epsilon'],
            'use_case': 'Advanced OOD detection with decoupled learning'
        },
        'dml_plus_focal': {
            'description': 'DML+ focal model for separate model training',
            'parameters': ['axis', 'epsilon'],
            'use_case': 'DML+ framework focal model component'
        },
        'dml_plus_center': {
            'description': 'DML+ center model for separate model training',
            'parameters': ['axis', 'epsilon'],
            'use_case': 'DML+ framework center model component'
        },
        'dynamic_tanh': {
            'description': 'Dynamic Tanh normalization for normalization-free transformers',
            'parameters': ['axis', 'alpha_init_value', 'kernel_initializer',
                           'bias_initializer', 'kernel_regularizer', 'bias_regularizer',
                           'kernel_constraint', 'bias_constraint'],
            'use_case': 'Normalization-free transformer architectures'
        },
        'energy_layer_norm': {
            'description': 'Energy Transformer layer norm (arXiv:2302.07253 eq. 1-2): SCALAR gamma + VECTOR delta; g = dL/dx of a Lagrangian with a PSD Hessian',
            # DECISION plan_2026-07-14_e5955791/D-004 (SUPERSEDED in mechanism, kept for
            # history): 'gamma_constraint' looks redundant here. Keep it. It is a real
            # EnergyLayerNorm ctor kwarg that pins gamma > 0, which is what keeps the
            # Lagrangian's Hessian PSD (see 57c9833e/D-010, in energy_layer_norm.py). It
            # was added to the class and not to this list, so the validator REJECTED a
            # parameter the builder accepted. Adding the string here patched the symptom,
            # not the mechanism. The validator now derives its whitelist from the real
            # ctor signature via `_accepted_params`, so this list is documentation only
            # and cannot break a caller by being incomplete. Measured at HEAD: it is
            # incomplete for all 18 of 18 types, by 107 (type, kwarg) pairs, and the
            # validator accepts every one of them anyway.
            # The originating plan directory is gone; this comment is the record.
            'parameters': ['epsilon', 'gamma_initializer', 'delta_initializer',
                           'gamma_constraint'],
            'use_case': 'Energy Transformer blocks, where the norm must be the derivative of a Lagrangian for the energy-descent guarantee to hold'
        }
    }


# ---------------------------------------------------------------------

def validate_normalization_config(
        normalization_type: NormalizationType,
        **kwargs: Any
) -> bool:
    """Check that a set of kwargs is valid for one normalization type.

    Two stages. First, every provided key must be in ``_accepted_params`` for that
    type. Second, a handful of types get a value check: ``max_band_width`` must be
    in ``(0, 1)``, ``temperature`` and ``alpha_init_value`` must be positive,
    ``constant`` must be a positive number, ``momentum`` must be in ``[0, 1]``, and
    ``epsilon`` must be positive for the RMS and bias-free families.

    :func:`create_normalization_layer` calls this for every registered type before
    it builds. Calling it directly is useful for validating a config file before a
    model is assembled.

    The unknown-type message here is the poorer of the two in this module. Measured:
    this function gives ``Unknown normalization type: definitely_not_a_type``, while
    the builder's ``else`` arm gives the same prefix followed by the full supported
    list. The builder therefore skips this function for unknown types on purpose,
    and tests pin the richer message.

    :param normalization_type: The type whose kwargs are being validated. Must be a
        registered type.
    :type normalization_type: NormalizationType
    :param kwargs: The keyword arguments to check.
    :return: ``True``. The function reports failure by raising, never by returning
        ``False``.
    :rtype: bool
    :raises ValueError: If ``normalization_type`` is not registered.
    :raises ValueError: If any key is outside the accepted set for the type. The
        message lists both the offending keys and the sorted accepted set.
    :raises ValueError: If a type-specific value check fails.
    """
    if normalization_type not in _TYPE_TO_CLASS:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    # DERIVED from the target class's real signature — NOT from
    # `get_normalization_info()['parameters']`, which is a curated documentation list and
    # drifts. Using a hand-maintained list as the whitelist made this function reject
    # parameters `create_normalization_layer` accepts, twice (see `_accepted_params`).
    # The invariant, pinned by `TestValidatorAgreesWithBuilder`:
    #     anything the BUILDER accepts, the VALIDATOR must accept.
    valid_params = _accepted_params(normalization_type)
    provided_params = set(kwargs.keys())

    # Check for invalid parameters
    invalid_params = provided_params - valid_params
    if invalid_params:
        raise ValueError(
            f"Invalid parameters for {normalization_type}: {invalid_params}. "
            f"Valid parameters: {sorted(valid_params)}"
        )

    # Type-specific validations
    if normalization_type in ['band_rms', 'adaptive_band_rms', 'band_logit_norm', 'zero_centered_band_rms_norm', 'zero_centered_adaptive_band_rms_norm']:
        if 'max_band_width' in kwargs:
            max_band_width = kwargs['max_band_width']
            if (not isinstance(max_band_width, (int, float))
                    or max_band_width <= 0 or max_band_width >= 1):
                raise ValueError(
                    f"max_band_width must be between 0 and 1, got {max_band_width}"
                )

    if normalization_type == 'logit_norm':
        if 'temperature' in kwargs:
            temperature = kwargs['temperature']
            if not isinstance(temperature, (int, float)) or temperature <= 0:
                raise ValueError("temperature must be a positive number")

    if normalization_type == 'decoupled_max_logit':
        if 'constant' in kwargs:
            constant = kwargs['constant']
            if not isinstance(constant, (int, float)):
                raise ValueError("constant must be a number")
            # Sign check mirrored from DecoupledMaxLogit._validate_inputs (same
            # message), so validate_normalization_config never green-lights a
            # config the class itself refuses.
            if constant <= 0:
                raise ValueError(f"constant must be positive, got {constant}")

    if normalization_type in ['rms_norm', 'zero_centered_rms_norm']:
        if 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']
            if not isinstance(epsilon, (int, float)) or epsilon <= 0:
                raise ValueError("epsilon must be a positive number")

    if normalization_type == 'bias_free_batch_norm':
        if 'momentum' in kwargs:
            momentum = kwargs['momentum']
            if not isinstance(momentum, (int, float)) or not (0.0 <= momentum <= 1.0):
                raise ValueError("momentum must be a number in [0, 1]")
        if 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']
            if not isinstance(epsilon, (int, float)) or epsilon <= 0:
                raise ValueError("epsilon must be a positive number")

    if normalization_type == 'dynamic_tanh':
        if 'alpha_init_value' in kwargs:
            alpha_init_value = kwargs['alpha_init_value']
            if not isinstance(alpha_init_value, (int, float)) or alpha_init_value <= 0:
                raise ValueError("alpha_init_value must be a positive number")

    return True


# ---------------------------------------------------------------------


def create_normalization_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """Build a normalization layer from a configuration dictionary.

    Pops ``'type'`` from a copy of ``config`` and forwards everything else to
    :func:`create_normalization_layer` as keyword arguments. The caller's dict is
    not mutated. Use this when the configuration arrives from a file or a
    hyperparameter sweep rather than as literal arguments.

    :param config: Must contain a ``'type'`` key naming a registered normalization
        type. Any other key is forwarded, including ``'name'`` and ``'epsilon'``.
    :type config: Dict[str, Any]
    :return: The configured layer, exactly as :func:`create_normalization_layer`
        would return it. The same three types return reduced or tuple outputs.
    :rtype: keras.layers.Layer
    :raises KeyError: If ``'type'`` is missing. Measured message:
        ``Configuration dictionary must contain 'type' key``.
    :raises ValueError: If the type is not registered, or a forwarded key or value
        is rejected. Raised downstream by :func:`create_normalization_layer`.

    Example:

    .. code-block:: python

        from dl_techniques.layers.norms import create_normalization_from_config

        norm = create_normalization_from_config(
            {'type': 'band_rms', 'max_band_width': 0.2, 'epsilon': 1e-7}
        )
    """
    if 'type' not in config:
        raise KeyError("Configuration dictionary must contain 'type' key")

    config_copy = config.copy()
    normalization_type = config_copy.pop('type')

    return create_normalization_layer(normalization_type, **config_copy)

# ---------------------------------------------------------------------