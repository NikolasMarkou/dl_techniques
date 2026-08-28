"""
Factory for the embedding layers in this package.

One entry point, :func:`create_embedding_layer`, builds any of the 13
registered embedding types from a type key plus keyword arguments. The type
key selects a row of ``EMBEDDING_REGISTRY``, which names the class, the
parameters that must be supplied and the parameters that have defaults.

The registry covers patch embeddings, learned positional embeddings, the
rotary family (RoPE, dual RoPE, continuous RoPE, multi-axis RoPE), the fixed
sinusoidal family, and the BERT-style token embeddings.

The registry is public API. Its key set, each entry's key names, and each
entry's ``required_params`` and ``optional_params`` are consumed by
config-driven callers and asserted by ``tests/test_layers/test_embedding/
test_embedding_factory.py`` and ``test_factory_ideogram4.py``. Adding,
renaming or removing any of them is a breaking change, not a cleanup.

Architecture Overview:

.. code-block:: text

    create_embedding_layer(embedding_type, name=None, **kwargs)
                    │
                    ▼
        validate_embedding_config(embedding_type, **kwargs)
          ├─ type not in EMBEDDING_REGISTRY ────► ValueError
          ├─ a required_params name missing ────► ValueError
          ├─ a value out of range ──────────────► ValueError
          └─ patch_size of the wrong type ──────► TypeError
                    │
                    ▼
        EMBEDDING_REGISTRY[embedding_type] -> class, params
                    │
                    ▼
        params = optional_params defaults, then kwargs on top
                    │
                    ▼
        strict key check: set(kwargs) - (required | optional)
          └─ non-empty ──► ValueError carrying the substring
                           STRICT_DROPPED_KEY_MARKER
                    │
                    ▼
        embed_class(**final_params) ──────────► the layer

    Every ValueError and TypeError above is caught by one handler at the
    bottom of the function, wrapped with the class name, the required
    params and the provided params, and re-raised as a ValueError. The
    original text survives after "Original error:". Nothing leaves this
    function as a TypeError.

Registered types:
    The table below is generated from ``EMBEDDING_REGISTRY``. The
    ``optional`` column is the number of parameters that carry a default;
    read the defaults themselves with :func:`get_embedding_info`.

.. code-block:: text

    ======================  ===========================  ========
    type key                class                        optional
    ======================  ===========================  ========
    patch_1d                PatchEmbedding1D                    5
    patch_2d                PatchEmbedding2D                    7
    positional_learned      PositionalEmbedding                 3
    rope                    RotaryPositionEmbedding             2
    dual_rope               DualRotaryPositionEmbedding         2
    continuous_rope         ContinuousRoPE                      2
    continuous_sincos       ContinuousSinCosEmbed               2
    bert_embeddings         BertEmbeddings                      8
    modern_bert_embeddings  ModernBertEmbeddings                0
    albert_factorized       AlbertFactorizedEmbedding           3
    positional_sine_2d      PositionEmbeddingSine2D             4
    scalar_sinusoidal       ScalarSinusoidalEmbedding           1
    mrope_ideogram4         Ideogram4MRoPE                      0
    ======================  ===========================  ========

Required parameters per type:
    A separate block, because the longest row holds seven names and would
    not fit beside the two columns above. Truncating it would misstate the
    contract.

.. code-block:: text

    patch_1d
        patch_size, embed_dim
    patch_2d
        patch_size, embed_dim
    positional_learned
        max_seq_len, dim
    rope
        head_dim, max_seq_len
    dual_rope
        head_dim, max_seq_len
    continuous_rope
        dim, ndim
    continuous_sincos
        dim, ndim
    bert_embeddings
        vocab_size, hidden_size, max_position_embeddings
    modern_bert_embeddings
        vocab_size, hidden_size, type_vocab_size, initializer_range,
        layer_norm_eps, dropout_rate, use_bias
    albert_factorized
        vocab_size, bottleneck_dim, output_dim
    positional_sine_2d
        (none)
    scalar_sinusoidal
        dim
    mrope_ideogram4
        head_dim, rope_theta, mrope_section

Two registry facts do not appear in either table:
    - ``bert_embeddings`` needs ``type_vocab_size`` whenever
      ``use_token_type_embeddings`` is true, which is the default. That is a
      computed rule inside :func:`validate_embedding_config`, not a static
      ``required_params`` entry. The anchor at that entry says why.
    - ``positional_sine_2d`` emits channels-FIRST,
      ``(B, 2*num_pos_feats, H, W)``. Callers that work channels-last must
      transpose.
"""

import math
import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .bert_embeddings import (
    BertEmbeddings,
    VALID_NORMALIZATION_TYPES,
    VALID_POSITION_EMBEDDING_TYPES,
)
from .continuous_rope_embedding import ContinuousRoPE
from .dual_rotary_position_embedding import DualRotaryPositionEmbedding
from .continuous_sin_cos_embedding import ContinuousSinCosEmbed
from .patch_embedding import PatchEmbedding1D, PatchEmbedding2D
from .positional_embedding import PositionalEmbedding
from .rotary_position_embedding import RotaryPositionEmbedding
from .multi_axis_rope import Ideogram4MRoPE
from .scalar_sinusoidal_embedding import ScalarSinusoidalEmbedding
from .positional_embedding_sine_2d import PositionEmbeddingSine2D
from .modern_bert_embeddings import ModernBertEmbeddings
from .albert_factorized_embedding import AlbertFactorizedEmbedding

# ---------------------------------------------------------------------
# Type definition for Embedding types
# ---------------------------------------------------------------------

EmbeddingType = Literal[
    'patch_1d',
    'patch_2d',
    'positional_learned',
    'rope',
    'dual_rope',
    'continuous_rope',
    'continuous_sincos',
    'bert_embeddings',
    'modern_bert_embeddings',
    'albert_factorized',
    'positional_sine_2d',
    'scalar_sinusoidal',
    'mrope_ideogram4'
]

# ---------------------------------------------------------------------
# Embedding layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------

EMBEDDING_REGISTRY: Dict[str, Dict[str, Any]] = {
    'patch_1d': {
        'class': PatchEmbedding1D,
        'description': '1D patch embedding for time series data with optional overlap.',
        'required_params': ['patch_size', 'embed_dim'],
        'optional_params': {
            'stride': None,
            'padding': 'causal',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros'
        },
        'use_case': 'Tokenizing time series or other 1D sequential data for transformers.'
    },
    'patch_2d': {
        'class': PatchEmbedding2D,
        'description': '2D image to patch embedding layer for Vision Transformers (ViT).',
        'required_params': ['patch_size', 'embed_dim'],
        'optional_params': {
            'flatten': True,
            'kernel_initializer': 'glorot_normal',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'activation': 'linear',
            'use_bias': True
        },
        'use_case': 'The first layer in ViT-style models to convert images into a sequence of patch embeddings.'
    },
    'positional_learned': {
        'class': PositionalEmbedding,
        'description': 'Adds learned, trainable positional embeddings to a sequence.',
        'required_params': ['max_seq_len', 'dim'],
        'optional_params': {
            'dropout_rate': 0.0,
            'pos_initializer': 'truncated_normal',
            'scale': 0.02
        },
        'use_case': 'Standard positional encoding for transformer models where positions are learned from data.'
    },
    'rope': {
        'class': RotaryPositionEmbedding,
        'description': 'Standard Rotary Position Embedding (RoPE) for relative position encoding.',
        'required_params': ['head_dim', 'max_seq_len'],
        'optional_params': {
            'rope_theta': 10000.0,
            'rope_percentage': 0.5
        },
        'use_case': 'Injecting relative positional information into query/key vectors in attention mechanisms.'
    },
    'dual_rope': {
        'class': DualRotaryPositionEmbedding,
        'description': 'Dual RoPE for Gemma3-style models with separate global and local configurations.',
        'required_params': ['head_dim', 'max_seq_len'],
        'optional_params': {
            'global_theta_base': 1_000_000.0,
            'local_theta_base': 10_000.0
        },
        'use_case': 'Models using both global (full) and local (sliding window) attention patterns.'
    },
    'continuous_rope': {
        'class': ContinuousRoPE,
        'description': 'RoPE extended to handle continuous multi-dimensional coordinates.',
        'required_params': ['dim', 'ndim'],
        'optional_params': {
            'max_wavelength': 10000.0,
            'assert_positive': True
        },
        'use_case': 'Applying rotational position encoding to data with continuous spatial coordinates (e.g., 3D point clouds).'
    },
    'continuous_sincos': {
        'class': ContinuousSinCosEmbed,
        'description': 'Embeds continuous coordinates using fixed sine and cosine functions.',
        'required_params': ['dim', 'ndim'],
        'optional_params': {
            'max_wavelength': 10000.0,
            'assert_positive': True
        },
        'use_case': 'Creating fixed, smooth positional representations for continuous coordinate data.'
    },
    'bert_embeddings': {
        'class': BertEmbeddings,
        'description': 'BERT embeddings combining word, optional position, and optional token type embeddings with configurable normalization.',
        # DECISION plan-2026-08-10T183739-b007f435/D-010
        # type_vocab_size is CONDITIONALLY required, and sits here only because
        # it is meaningless when use_token_type_embeddings is False. Do NOT move
        # it back into required_params: that forces every token-type-free caller,
        # such as models/language/distilbert, to pass a dummy positive integer
        # that is then serialized into every checkpoint. The computed rule in
        # validate_embedding_config covers the case where it IS needed; do not
        # delete that either. See decisions.md D-002 and D-010.
        'required_params': ['vocab_size', 'hidden_size', 'max_position_embeddings'],
        'optional_params': {
            'type_vocab_size': None,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-8,
            'dropout_rate': 0.0,
            'normalization_type': 'layer_norm',
            'use_token_type_embeddings': True,
            'position_embedding_type': 'learned',
            'mask_zero': True
        },
        'use_case': 'BERT-style language models combining word, positional, and segment embeddings with sum aggregation and normalization.'
    },
    'modern_bert_embeddings': {
        'class': ModernBertEmbeddings,
        'description': 'ModernBERT embeddings: word + token-type embeddings, normalized (no learned positional embedding; RoPE is applied in attention).',
        'required_params': ['vocab_size', 'hidden_size', 'type_vocab_size', 'initializer_range', 'layer_norm_eps', 'dropout_rate', 'use_bias'],
        'optional_params': {},
        'use_case': 'ModernBERT-style encoders where positional information is injected by rotary attention rather than a learned position embedding.'
    },
    'albert_factorized': {
        'class': AlbertFactorizedEmbedding,
        'description': 'ALBERT-style factorized embedding: vocab -> bottleneck_dim -> output_dim via a two-matrix decomposition.',
        'required_params': ['vocab_size', 'bottleneck_dim', 'output_dim'],
        'optional_params': {
            'embeddings_initializer': 'uniform',
            'embeddings_regularizer': None,
            'projection_regularizer': None
        },
        'use_case': 'Parameter-efficient token embeddings where a small bottleneck is projected up to the model hidden size.'
    },
    'positional_sine_2d': {
        'class': PositionEmbeddingSine2D,
        'description': 'Fixed 2D sinusoidal positional encoding for image feature maps. NOTE: emits channels-FIRST (B, 2*num_pos_feats, H, W); callers must transpose to channels-last as needed.',
        'required_params': [],
        'optional_params': {
            'num_pos_feats': 64,
            'temperature': 10000.0,
            'normalize': True,
            'scale': 2 * math.pi
        },
        'use_case': 'DETR / ViT-style detectors needing a non-learnable 2D positional grid over a convolutional feature map.'
    },
    'scalar_sinusoidal': {
        'class': ScalarSinusoidalEmbedding,
        'description': 'Sinusoidal scalar (timestep) embedding refined by a 2-layer SiLU MLP, for the Ideogram4 DiT.',
        'required_params': ['dim'],
        'optional_params': {
            'input_range': (0.0, 1.0)
        },
        'use_case': 'Diffusion-model timestep or other continuous scalar conditioning.'
    },
    'mrope_ideogram4': {
        'class': Ideogram4MRoPE,
        'description': '3D multi-axis RoPE (t, h, w) for the Ideogram4 DiT. call() returns a (cos, sin) tuple.',
        'required_params': ['head_dim', 'rope_theta', 'mrope_section'],
        'optional_params': {},
        'use_case': 'Packed-sequence DiT (Ideogram4) where position ids carry (t, h, w) per token.'
    }
}

# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_embedding_info() -> Dict[str, Dict[str, Any]]:
    """Return the registry contents, one entry per embedding type.

    Each entry holds ``class``, ``description``, ``required_params``,
    ``optional_params`` and ``use_case``. The ``optional_params`` mapping is
    the only place the per-type defaults are written down.

    KNOWN DEFECT, DESCRIBED AS IT BEHAVES TODAY. The copy is SHALLOW. The
    outer dict and each type's dict are new, but the nested
    ``required_params`` list and ``optional_params`` dict are the registry's
    own objects: ``get_embedding_info()['patch_2d']['optional_params'] is
    EMBEDDING_REGISTRY['patch_2d']['optional_params']`` is ``True``, and so
    is the same identity for ``required_params``. Mutating either one changes
    the registry for every later call in the process.

    Treat the result as read-only, or deep-copy it. This is a DEFECT
    scheduled for repair, not a design choice; when the entries are deep
    copied this paragraph becomes false and must go with it.

    :return: Mapping from embedding type key to that type's registry entry.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {embed_type: info.copy() for embed_type, info in EMBEDDING_REGISTRY.items()}

def validate_embedding_config(embedding_type: str, **kwargs: Any) -> None:
    """Check an embedding configuration before anything is constructed.

    The checks run in a fixed order: the type must be registered, every name
    in that type's ``required_params`` must be present, then the values are
    range-checked. Value checks are keyed on the parameter NAME, so the
    shared names such as ``dim`` and ``embed_dim`` are checked for every
    type that passes them, and a handful of type-specific rules follow.

    This function does NOT reject unknown keyword names.
    :func:`create_embedding_layer` does that, after calling this.

    :param embedding_type: Type of embedding to validate.
    :type embedding_type: str
    :param kwargs: Parameters to validate against that type's requirements.
    :raises ValueError: If the type is not registered, if a required
        parameter is missing, or if a value is outside its valid range.
    :raises TypeError: If ``patch_size`` is neither an int nor a sequence,
        for the ``patch_2d`` type. This is the one path here that does not
        raise ``ValueError``.
    """
    if embedding_type not in EMBEDDING_REGISTRY:
        available_types = list(EMBEDDING_REGISTRY.keys())
        raise ValueError(
            f"Unknown embedding type '{embedding_type}'. "
            f"Available types: {available_types}"
        )

    embed_info = EMBEDDING_REGISTRY[embedding_type]
    required_params = embed_info['required_params']

    # Check for missing required parameters
    missing_params = [param for param in required_params if param not in kwargs]
    if missing_params:
        raise ValueError(
            f"Required parameters missing for {embedding_type}: {missing_params}. "
            f"Required: {required_params}"
        )

    # --- Parameter Value Validations ---

    # Common positive integer checks
    positive_params = ['dim', 'embed_dim', 'head_dim', 'max_seq_len', 'patch_size', 'ndim',
                      'vocab_size', 'hidden_size', 'max_position_embeddings', 'type_vocab_size',
                      'bottleneck_dim', 'output_dim', 'num_pos_feats']
    for param in positive_params:
        if param in kwargs and kwargs[param] is not None:
            value = kwargs[param]
            # Special case for patch_size in patch_2d
            if param == 'patch_size' and embedding_type == 'patch_2d':
                if isinstance(value, int):
                    if value <= 0:
                        raise ValueError(f"{param} must be a positive integer, got {value}")
                elif isinstance(value, (list, tuple)):
                     if len(value) != 2 or not all(isinstance(p, int) and p > 0 for p in value):
                         raise ValueError(f"{param} must be a tuple of 2 positive integers, got {value}")
                else:
                    raise TypeError(f"{param} must be an int or a tuple of 2 ints, got {type(value)}")
            elif isinstance(value, int) and value <= 0:
                raise ValueError(f"{param} must be positive, got {value}")

    # Common positive float checks
    positive_float_params = ['initializer_range', 'layer_norm_eps', 'scale']
    for param in positive_float_params:
        if param in kwargs and kwargs[param] is not None and kwargs[param] <= 0:
            raise ValueError(f"{param} must be positive, got {kwargs[param]}")

    # Common dropout rate checks
    dropout_params = ['dropout_rate']
    for param in dropout_params:
        if param in kwargs and not (0.0 <= kwargs[param] <= 1.0):
            raise ValueError(f"{param} must be in [0, 1], got {kwargs[param]}")

    # Type-specific validations
    if embedding_type == 'patch_1d':
        if 'stride' in kwargs and kwargs['stride'] is not None and kwargs['stride'] <= 0:
            raise ValueError(f"stride must be positive, got {kwargs['stride']}")
        if 'padding' in kwargs and kwargs['padding'] not in ['same', 'valid', 'causal']:
            raise ValueError(f"padding must be 'same', 'valid', or 'causal', got {kwargs['padding']}")

    if embedding_type == 'rope':
        if 'rope_theta' in kwargs and kwargs['rope_theta'] <= 0:
            raise ValueError(f"rope_theta must be positive, got {kwargs['rope_theta']}")
        if 'rope_percentage' in kwargs and not (0.0 < kwargs['rope_percentage'] <= 1.0):
            raise ValueError(f"rope_percentage must be in (0, 1], got {kwargs['rope_percentage']}")

    if embedding_type == 'dual_rope':
        if 'head_dim' in kwargs and kwargs['head_dim'] % 2 != 0:
            raise ValueError(f"head_dim must be even for dual_rope, got {kwargs['head_dim']}")
        if 'global_theta_base' in kwargs and kwargs['global_theta_base'] <= 0:
            raise ValueError(f"global_theta_base must be positive, got {kwargs['global_theta_base']}")
        if 'local_theta_base' in kwargs and kwargs['local_theta_base'] <= 0:
            raise ValueError(f"local_theta_base must be positive, got {kwargs['local_theta_base']}")

    if embedding_type in ['continuous_rope', 'continuous_sincos']:
        if 'max_wavelength' in kwargs and kwargs['max_wavelength'] <= 0:
            raise ValueError(f"max_wavelength must be positive, got {kwargs['max_wavelength']}")

    if embedding_type == 'bert_embeddings':
        if 'normalization_type' in kwargs:
            valid_norm_types = list(VALID_NORMALIZATION_TYPES)
            if kwargs['normalization_type'] not in valid_norm_types:
                raise ValueError(f"normalization_type must be one of {valid_norm_types}, got {kwargs['normalization_type']}")

        if 'position_embedding_type' in kwargs:
            valid_position_types = list(VALID_POSITION_EMBEDDING_TYPES)
            if kwargs['position_embedding_type'] not in valid_position_types:
                raise ValueError(
                    f"position_embedding_type must be one of {valid_position_types}, "
                    f"got {kwargs['position_embedding_type']}"
                )

        # DECISION plan-2026-08-10T183739-b007f435/D-010
        # The conditional-required rule that replaces the static required_params
        # entry for type_vocab_size. The default below is READ FROM THE REGISTRY,
        # not written as a literal True, so flipping the registry default cannot
        # leave this rule disagreeing with what the factory injects. Delete this
        # block and a caller who omits type_vocab_size with token types enabled
        # reaches the constructor instead of failing here. See decisions.md D-010.
        token_types_default = EMBEDDING_REGISTRY['bert_embeddings'][
            'optional_params']['use_token_type_embeddings']
        if kwargs.get('use_token_type_embeddings', token_types_default):
            type_vocab_size = kwargs.get('type_vocab_size')
            if type_vocab_size is None or type_vocab_size <= 0:
                raise ValueError(
                    f"type_vocab_size is required and must be positive when "
                    f"use_token_type_embeddings is True (the default), got "
                    f"{type_vocab_size!r}. Pass a positive int, or pass "
                    f"use_token_type_embeddings=False for a model without segment "
                    f"embeddings."
                )

    if embedding_type == 'positional_sine_2d':
        if 'temperature' in kwargs and kwargs['temperature'] <= 0:
            raise ValueError(f"temperature must be positive, got {kwargs['temperature']}")


#: The stable substring every strict dropped-key ``ValueError`` from
#: :func:`create_embedding_layer` carries. Guards match on THIS constant rather
#: than re-typing the phrase, so rewording the message cannot silently blind
#: them, and so the phrase has exactly one home. ``layers/ffn/factory.py``
#: defines a constant of the same name with the same wording and the same
#: contract; the match is intentional.
#: NOTE for test authors: the ``(s)`` makes this string a REGEX with a group, so
#: ``pytest.raises(match=...)`` needs ``re.escape()`` around it (or use a plain
#: ``in str(excinfo.value)`` substring check, as the FFN tests do).
STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"


def create_embedding_layer(
    embedding_type: EmbeddingType,
    name: Optional[str] = None,
    **kwargs: Any
) -> keras.layers.Layer:
    """Build one of the 13 registered embedding layers.

    Validates the configuration, fills in that type's defaults, rejects any
    keyword the type does not declare, and constructs the class. See the
    module docstring for the registry table and the dispatch diagram.

    Every failure leaves this function as a ``ValueError``, including the
    ``TypeError`` paths. The handler at the bottom catches both, prepends
    the class name, the required parameters and the parameters you supplied,
    and re-raises. The original message survives after ``Original error:``,
    so a guard matching :data:`STRICT_DROPPED_KEY_MARKER` still matches.

    :param embedding_type: Type key of the layer to create.
    :type embedding_type: EmbeddingType
    :param name: Optional name for the layer.
    :type name: Optional[str]
    :param kwargs: Parameters for that embedding type. Read
        ``get_embedding_info()`` or the module docstring for the names.
    :return: A configured Keras embedding layer instance.
    :rtype: keras.layers.Layer
    :raises ValueError: If the type is not registered, if a required
        parameter is missing, if a value is out of range, if a supplied
        keyword is not a parameter of that type (the message then carries
        :data:`STRICT_DROPPED_KEY_MARKER`), or if the constructor itself
        rejects the arguments.
    """
    try:
        # Validate the provided configuration
        validate_embedding_config(embedding_type, **kwargs)

        # Get layer info and class from the registry
        embed_info = EMBEDDING_REGISTRY[embedding_type]
        embed_class = embed_info['class']

        # Prepare parameters, starting with defaults and overriding with user kwargs
        params = {}
        params.update(embed_info['optional_params'])
        params.update(kwargs)

        # Filter out any unknown parameters to avoid "Unrecognized keyword arguments"
        valid_param_names = set(embed_info['required_params']) | set(embed_info['optional_params'].keys())
        final_params = {key: val for key, val in params.items() if key in valid_param_names}

        # DECISION plan-2026-08-14T042537-ff96c6c6/D-002
        # RAISE on an unrecognized kwarg; do NOT drop it silently and do NOT
        # soften this to a warning. The filter above turned
        # `ViT(pos_dropout_rate=0.5)` into a permanent 0.0 at four production
        # call sites -- MEASURED -- because `dropout=` is not `dropout_rate=`.
        # Every statically resolvable call site was swept clean before this
        # raise landed. Ported from `ffn/factory.py`'s identical predicate
        # (plan-2026-07-30T140922-8af1028f/D-023).
        #
        # BOTH halves of the predicate matter:
        #
        # * subtract `valid_param_names` (required | optional), NEVER just
        #   `required_params`. That narrowing is the tempting "simplification"
        #   and it turns every legitimately passed optional parameter into an
        #   error. MEASURED: the all-optional control then fires for 11 of the
        #   13 registered types and 22 tests go red. The two survivors,
        #   `modern_bert_embeddings` and `mrope_ideogram4`, declare no optional
        #   params and so have nothing to break. The FFN side measured 21/21.
        # * read `kwargs`, what the CALLER supplied, not the merged `params`.
        #   The two agree today, but the `kwargs` form stays correct if a
        #   registry entry ever gains an `optional_params` key its class does
        #   not accept. The `params` form would blame the caller for that.
        #
        # Placement matters too: this runs AFTER `validate_embedding_config`, so
        # unknown-type, missing-required and bad-value calls keep their earlier
        # failure mode, and 3 negative-path tests depend on that order. The
        # message below is quoted by consumers in models/language/distilbert and
        # models/vision/energy_transformer; do not reword its shape.
        # See decisions.md D-002.
        dropped = sorted(set(kwargs) - valid_param_names)
        if dropped:
            raise ValueError(
                f"create_embedding_layer('{embedding_type}'): "
                f"{len(dropped)} {STRICT_DROPPED_KEY_MARKER} {dropped}. "
                f"'{embedding_type}' ({embed_class.__name__}) accepts only "
                f"{sorted(valid_param_names)}. "
                f"Either you mistyped one of those names, or you chose the "
                f"wrong embedding_type for the parameters you are passing."
            )

        # Add layer name if provided
        if name is not None:
            final_params['name'] = name

        # Log final parameters before creating the layer
        logger.info(f"Creating '{embedding_type}' embedding layer with parameters:")
        # A copy, so adding the name for the log cannot touch final_params.
        log_params = {**final_params}
        if name:
            log_params['name'] = name
        for param_name, param_value in sorted(log_params.items()):
            logger.info(f"  {param_name}: {param_value!r}")

        # Create the layer instance
        embedding_layer = embed_class(**final_params)

        logger.debug(f"Successfully created '{embedding_type}' layer: {embedding_layer.name}")
        return embedding_layer

    except (TypeError, ValueError) as e:
        # Provide enhanced error reporting with context
        embed_info = EMBEDDING_REGISTRY.get(embedding_type)
        if embed_info:
            required = embed_info.get('required_params', [])
            provided = list(kwargs.keys())
            class_name = embed_info.get('class', type(None)).__name__
            error_msg = (
                f"Failed to create '{embedding_type}' embedding layer ({class_name}).\n"
                f"  Required params: {required}\n"
                f"  Provided params: {provided}\n"
                f"  Check parameter compatibility and types. "
                f"Use get_embedding_info() for details.\n"
                f"  Original error: {e}"
            )
        else:
            error_msg = f"Failed to create embedding layer. Unknown type '{embedding_type}'. Original error: {e}"

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_embedding_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """Build an embedding layer from a single configuration dictionary.

    The ``'type'`` key selects the embedding type. Every other key is passed
    to :func:`create_embedding_layer` as a keyword argument, so the strict
    unknown-key rule applies to them too. The mapping is copied before the
    ``'type'`` key is popped, so the caller's dictionary is left alone.

    :param config: Dictionary holding a ``'type'`` key plus that type's
        parameters.
    :type config: Dict[str, Any]
    :return: A configured Keras embedding layer instance.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict, if the ``'type'`` key is
        missing, or for any reason :func:`create_embedding_layer` raises.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")

    if 'type' not in config:
        raise ValueError("Configuration dictionary must include a 'type' key.")

    config_copy = config.copy()
    embedding_type = config_copy.pop('type')

    logger.debug(f"Creating embedding from config - type: {embedding_type}, params: {list(config_copy.keys())}")

    return create_embedding_layer(embedding_type, **config_copy)