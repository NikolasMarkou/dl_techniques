"""
One layer that fuses features from several modalities.

``MultiModalFusion`` takes one tensor per modality -- vision, language, audio
and so on -- and combines them into a fused representation. You choose how
with ``fusion_strategy``. The layer builds the sub-layers for that strategy
and no others, so the parameter count and the output shape both depend on the
strategy you pick.

Eight strategy keys, seven implementations: ``'addition'`` and
``'multiplication'`` share one build and one call path. The full list, by the
exact key you pass:

    1. ``'cross_attention'``: every modality attends to every other one, then
       normalization and FFN. Returns a TUPLE, one tensor per modality.
    2. ``'concatenation'``: concatenate on the feature axis, project back to
       ``dim``, then normalization and dropout.
    3. ``'addition'``: align the modalities, sum them, then normalization and
       FFN.
    4. ``'multiplication'``: the same path with an element-wise product in
       place of the sum.
    5. ``'gated'``: a learned sigmoid gate scales each modality, then
       concatenate, project and normalize. No FFN.
    6. ``'attention_pooling'``: self-attend each modality, average over the
       sequence axis, concatenate and project. Returns ``(batch, dim)`` --
       the sequence axis is gone.
    7. ``'bilinear'``: the outer product of exactly two modalities, flattened
       and projected. This is the only real cross-modal product here.
    8. ``'tensor_fusion'``: concatenate, then a wide hidden layer of
       ``num_tensor_projections`` parallel ``Dense(dim)`` units, then one
       linear projection back to ``dim``.

Note on ``'tensor_fusion'``:
    Despite the name, this strategy computes **no outer product and no tensor
    decomposition**. It is concatenation fusion with a wider, non-linear
    hidden layer: concatenate on the feature axis, apply
    ``num_tensor_projections`` parallel ``Dense(dim, activation)`` layers to
    that same concatenation, concatenate their outputs, and project back to
    ``dim``. Only ``'bilinear'`` in this file forms a genuine cross-modal
    product. See ``_call_tensor_fusion`` for the measured cost comparison.

References:
    - Baltrusaitis et al. (2018): Multimodal Machine Learning: A Survey
    - Vaswani et al. (2017): Attention Is All You Need
    - Zadeh et al. (2017): Tensor Fusion Network -- the source of the
      ``'tensor_fusion'`` NAME only. That paper's mechanism (the outer product
      of the modality vectors, optionally low-rank-factorized) is **not**
      implemented here; ``'bilinear'`` is the closest thing in this file.
    - Lu et al. (2019): ViLBERT
"""

import keras
from typing import Optional, Union, List, Dict, Any, Tuple, Literal, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn import create_ffn_layer, FFNType
from dl_techniques.layers.attention import create_attention_layer
from dl_techniques.layers.norms import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------
# Type definitions for fusion strategies
# ---------------------------------------------------------------------

FusionStrategy = Literal[
    # Every modality attends to every other one
    'cross_attention',
    # Concatenate on the feature axis and project
    'concatenation',
    # Element-wise sum, with alignment projections above 2 modalities
    'addition',
    # Element-wise product, same path as 'addition'
    'multiplication',
    # A learned sigmoid gate per modality
    'gated',
    # Self-attend, mean-pool the sequence axis, concatenate and project
    'attention_pooling',
    # Outer product of exactly 2 modalities
    'bilinear',
    # Concatenate, then a wide parallel-Dense hidden layer
    'tensor_fusion'
]

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiModalFusion(keras.layers.Layer):
    """Fuse two or more modalities into one representation.

    Give the layer a list of tensors, one per modality, each shaped
    ``(batch, seq_len, dim)``. It returns the fused features. Which sub-layers
    exist, and what shape comes back, are both decided by ``fusion_strategy``.

    There are eight strategy keys and seven call paths: ``'addition'`` and
    ``'multiplication'`` share one. Three output shapes are possible, so read
    the table below before wiring the output into something downstream.

    ``num_fusion_layers`` above 1 works for ``'cross_attention'`` only. Every
    other strategy raises at construction if you ask for more than one block.
    ``num_tensor_projections`` is read by ``'tensor_fusion'`` only.

    **Architecture Overview:**

    .. code-block:: text

        Modality 1     Modality 2    ...    Modality N
        (B, T1, dim)   (B, T2, dim)         (B, TN, dim)
             │              │                    │
             └──────────────┼────────────────────┘
                            ▼
              ┌──────────────────────────┐
              │  _validate_input_shapes  │  build() only
              │  rank 3, last axis = dim │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │ _require_equal_sequence_ │
              │          lengths         │
              │   skipped for 2 of the   │
              │   8 keys; see below      │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │  dispatch on             │
              │  fusion_strategy         │
              │  8 keys, 7 call paths    │
              └────────────┬─────────────┘
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
      cross_attention  attention_     the other
                        pooling       six keys
             │             │             │
             ▼             ▼             ▼
        tuple of N     (B, dim)     (B, T, dim)
       (B, Ti, dim)

    B is the batch size, Ti the sequence length of modality i, and N the
    number of modalities. The three-way output fork at the bottom is
    ``compute_output_shape``, not a per-strategy detail.

    **Strategies, read from the code:**

    .. code-block:: text

        key                builds                output
        -----------------  --------------------  ------------
        cross_attention    N-1 cross-attn, one   tuple of N
                           norm, one FFN, per    (B, Ti, dim)
                           modality per block
        concatenation      Dense, norm, dropout  (B, T, dim)
        addition           align Dense if N > 2  (B, T, dim)
        multiplication     then norm and FFN     (B, T, dim)
        gated              N sigmoid gates,      (B, T, dim)
                           Dense, norm
        attention_pooling  N self-attn, Dense    (B, dim)
        bilinear           Dense, norm           (B, T, dim)
        tensor_fusion      P Dense + 1 linear    (B, T, dim)

        P = num_tensor_projections. Only 'concatenation' uses
        dropout_rate in its call path. 'gated', 'bilinear',
        'attention_pooling' and 'tensor_fusion' apply no FFN;
        'attention_pooling' and 'tensor_fusion' apply no norm.
        'bilinear' accepts exactly 2 modalities and raises
        otherwise. use_residual is read by 'cross_attention'
        only.

    **The equal-length contract:**

    .. code-block:: text

        call(inputs)
             │
             ▼
        is fusion_strategy in LENGTH_AGNOSTIC_STRATEGIES?
             │
       yes ──┴── no
        │         │
        ▼         ▼
      skip    _require_equal_sequence_lengths(inputs)
                  │
                  ├─ any shape[1] is None ─────► return
                  ├─ all lengths equal ────────► return
                  └─ lengths differ ───────────► ValueError

        Exempt, 2 of 8 keys: 'cross_attention',
        'attention_pooling'. Both go through attention layers
        that accept different query and key lengths.

        Guarded, 6 of 8 keys: 'concatenation', 'addition',
        'multiplication', 'gated', 'bilinear', 'tensor_fusion'.
        They concatenate on the feature axis or combine
        element-wise, so axis 1 must match.

        A None sequence axis is never refused. A symbolic build
        is legal and the check stays out of it rather than
        guessing.

    :param dim: Feature width of the fused representation. Every modality
        must already arrive at this width; ``build`` raises otherwise.
    :type dim: int
    :param fusion_strategy: Which of the eight keys to use.
    :type fusion_strategy: FusionStrategy
    :param num_fusion_layers: How many fusion blocks to stack. Only
        ``'cross_attention'`` accepts a value above 1; anything else raises.
    :type num_fusion_layers: int
    :param attention_config: Extra arguments for ``create_attention_layer``.
        Used by ``'cross_attention'`` and ``'attention_pooling'``. ``dim``
        defaults to this layer's ``dim``.
    :type attention_config: Optional[Dict[str, Any]]
    :param ffn_type: FFN type passed to ``create_ffn_layer``. Read by
        ``'cross_attention'`` and the element-wise strategies.
    :type ffn_type: FFNType
    :param ffn_config: Extra arguments for ``create_ffn_layer``.
        ``hidden_dim`` defaults to ``4 * dim`` and ``output_dim`` is always
        forced to ``dim``.
    :type ffn_config: Optional[Dict[str, Any]]
    :param norm_type: Normalization type passed to
        ``create_normalization_layer``.
    :type norm_type: NormalizationType
    :param norm_config: Extra arguments for ``create_normalization_layer``.
    :type norm_config: Optional[Dict[str, Any]]
    :param num_tensor_projections: Hidden width for ``'tensor_fusion'``, in
        units of ``dim``: the number of parallel ``Dense(dim)`` layers over
        the concatenated modalities, so a hidden layer of width
        ``num_tensor_projections * dim``. Ignored by every other strategy.
    :type num_tensor_projections: int
    :param dropout_rate: Dropout probability. Only ``'concatenation'`` builds
        a Dropout layer, so this is inert for the other strategies.
    :type dropout_rate: float
    :param use_residual: Whether ``'cross_attention'`` adds residual
        connections around the attention and the FFN. No other strategy
        reads it.
    :type use_residual: bool
    :param activation: Activation for the projection layers. The final
        ``'tensor_fusion'`` projection is linear and ignores it.
    :type activation: Union[str, Callable]
    :param kernel_initializer: Initializer for the Dense kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the Dense biases.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the Dense kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Regularizer for the Dense biases.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]

    :ivar dim: The stored feature width.
    :vartype dim: int
    :ivar fusion_strategy: The stored strategy key. Both dispatch dicts are
        looked up with it.
    :vartype fusion_strategy: FusionStrategy
    :ivar num_fusion_layers: The stored block count.
    :vartype num_fusion_layers: int
    :ivar attention_config: The attention arguments, ``{}`` when none were
        given.
    :vartype attention_config: Dict[str, Any]
    :ivar ffn_type: The stored FFN type.
    :vartype ffn_type: FFNType
    :ivar ffn_config: The FFN arguments, ``{}`` when none were given.
    :vartype ffn_config: Dict[str, Any]
    :ivar norm_type: The stored normalization type.
    :vartype norm_type: NormalizationType
    :ivar norm_config: The normalization arguments, ``{}`` when none were
        given.
    :vartype norm_config: Dict[str, Any]
    :ivar num_tensor_projections: The stored hidden-width multiplier.
    :vartype num_tensor_projections: int
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_residual: The stored residual flag.
    :vartype use_residual: bool
    :ivar activation: The resolved activation function.
    :vartype activation: Callable
    :ivar kernel_initializer: The resolved kernel initializer.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar fusion_layers: Attention-bearing sub-layers. One block per fusion
        layer for ``'cross_attention'``; one attention layer per modality for
        ``'attention_pooling'``. Empty for every other strategy.
    :vartype fusion_layers: List[keras.layers.Layer]
    :ivar projection_layers: The Dense projections. For ``'tensor_fusion'``
        this holds ``num_tensor_projections`` hidden layers followed by the
        final linear one, so the last entry is the output projection.
    :vartype projection_layers: List[keras.layers.Dense]
    :ivar norm_layers: The normalization layers, or empty for
        ``'attention_pooling'`` and ``'tensor_fusion'``.
    :vartype norm_layers: List[keras.layers.Layer]
    :ivar ffn_layers: The FFN layers. Only the element-wise strategies fill
        this; ``'cross_attention'`` keeps its FFNs on the block objects.
    :vartype ffn_layers: List[keras.layers.Layer]
    :ivar gate_layers: The sigmoid gates, one per modality. Filled by
        ``'gated'`` only.
    :vartype gate_layers: List[keras.layers.Dense]
    :ivar dropout_layers: The Dropout layers. Filled by ``'concatenation'``
        only.
    :vartype dropout_layers: List[keras.layers.Dropout]
    :ivar LENGTH_AGNOSTIC_STRATEGIES: Class constant. The two keys that may
        fuse modalities of different sequence length.
    :vartype LENGTH_AGNOSTIC_STRATEGIES: frozenset

    :raises ValueError: If ``dim`` or ``num_fusion_layers`` is not positive,
        if ``dropout_rate`` is outside ``[0, 1]``, if
        ``num_tensor_projections`` is not positive under ``'tensor_fusion'``,
        or if ``num_fusion_layers`` exceeds 1 for a non-iterative strategy.
    :raises ValueError: If ``build`` receives fewer than 2 shapes, a shape
        that is not rank 3, or a shape whose last axis is not ``dim``.
    :raises ValueError: If ``call`` receives fewer than 2 tensors, an unknown
        strategy key, or unequal statically-known sequence lengths for a
        length-sensitive strategy.

    Input shape:
        A list or tuple of at least 2 tensors, each of shape
        ``(batch_size, sequence_length, dim)``. ``'bilinear'`` accepts
        exactly 2.

    Output shape:
        ``(batch_size, sequence_length, dim)`` for most strategies;
        ``(batch_size, dim)`` for ``'attention_pooling'``; a tuple of N
        tensors of shape ``(batch_size, sequence_length_i, dim)`` for
        ``'cross_attention'``.

    Example:
        .. code-block:: python

            import keras
            from dl_techniques.layers.fusion.multimodal_fusion import (
                MultiModalFusion,
            )

            vision = keras.random.normal((2, 16, 64))
            text = keras.random.normal((2, 16, 64))

            fusion = MultiModalFusion(dim=64, fusion_strategy='gated')
            fused = fusion([vision, text])
            fused.shape  # (2, 16, 64)
    """

    def __init__(
        self,
        dim: int = 768,
        fusion_strategy: FusionStrategy = 'cross_attention',
        num_fusion_layers: int = 1,
        attention_config: Optional[Dict[str, Any]] = None,
        ffn_type: FFNType = 'mlp',
        ffn_config: Optional[Dict[str, Any]] = None,
        norm_type: NormalizationType = 'layer_norm',
        norm_config: Optional[Dict[str, Any]] = None,
        num_tensor_projections: int = 8,
        dropout_rate: float = 0.1,
        use_residual: bool = True,
        activation: Union[str, Callable] = 'gelu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ) -> None:
        """Store the configuration and prepare the sub-layer containers.

        No sub-layer is created here. Sub-layers need the input shapes, so
        they are all created in :meth:`build`.
        """
        super().__init__(**kwargs)

        # Validate first, so a bad argument fails here and not inside build()
        self._validate_init_params(
            dim, fusion_strategy, num_fusion_layers,
            num_tensor_projections, dropout_rate
        )

        # Store the configuration; build() reads it back
        self.dim = dim
        self.fusion_strategy = fusion_strategy
        self.num_fusion_layers = num_fusion_layers
        self.attention_config = attention_config or {}
        self.ffn_type = ffn_type
        self.ffn_config = ffn_config or {}
        self.norm_type = norm_type
        self.norm_config = norm_config or {}
        self.num_tensor_projections = num_tensor_projections
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

        # Resolve strings to activation, initializer and regularizer objects
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Create the empty containers; build() fills them
        self._init_layer_containers()

    def _validate_init_params(
        self,
        dim: int,
        fusion_strategy: str,
        num_fusion_layers: int,
        num_tensor_projections: int,
        dropout_rate: float
    ) -> None:
        """Check the constructor arguments and raise on the first bad one.

        Two of the four checks are conditional on the strategy.
        ``num_tensor_projections`` is only required to be positive under
        ``'tensor_fusion'``, and ``num_fusion_layers`` above 1 is only
        allowed for ``'cross_attention'``. That second check is what makes
        ``num_tensor_projections`` positive whenever ``_call_tensor_fusion``
        can run.

        :param dim: Feature width. Must be positive.
        :type dim: int
        :param fusion_strategy: The strategy key being configured.
        :type fusion_strategy: str
        :param num_fusion_layers: Number of fusion blocks. Must be positive.
        :type num_fusion_layers: int
        :param num_tensor_projections: Hidden-width multiplier for
            ``'tensor_fusion'``.
        :type num_tensor_projections: int
        :param dropout_rate: Dropout rate. Must be in ``[0, 1]``.
        :type dropout_rate: float

        :raises ValueError: If any of the four checks fails.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        if num_fusion_layers <= 0:
            raise ValueError(f"num_fusion_layers must be positive, got {num_fusion_layers}")

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        if fusion_strategy == 'tensor_fusion' and num_tensor_projections <= 0:
            raise ValueError(
                f"num_tensor_projections must be positive for tensor_fusion, "
                f"got {num_tensor_projections}"
            )

        # Only an iterative strategy can stack more than one fusion block
        iterative_strategies = {'cross_attention'}
        if fusion_strategy not in iterative_strategies and num_fusion_layers > 1:
            raise ValueError(
                f"num_fusion_layers > 1 is only supported for iterative strategies "
                f"{iterative_strategies}, but got strategy '{fusion_strategy}'"
            )

    def _init_layer_containers(self) -> None:
        """Create the six empty sub-layer lists.

        Each strategy fills only the lists it needs, so most of these stay
        empty for any given layer. The per-list contents are documented in
        the class ``:ivar`` block.
        """
        # Attention blocks: 'cross_attention' and 'attention_pooling'
        self.fusion_layers = []
        # Dense projections, used by every strategy except 'cross_attention'
        self.projection_layers = []
        # Normalization layers
        self.norm_layers = []
        # Feed-forward networks: the element-wise strategies only
        self.ffn_layers = []
        # Sigmoid gates: 'gated' only
        self.gate_layers = []
        # Dropout: 'concatenation' only
        self.dropout_layers = []

    def build(self, input_shape: Union[Tuple, List[Tuple]]) -> None:
        """Create the sub-layers for the configured strategy.

        Keras calls this the first time the layer runs. Every sub-layer is
        created here because they all need the input shapes. Only one of the
        eight builders runs, so a layer never carries sub-layers for a
        strategy it does not use.

        :param input_shape: One shape per modality, each
            ``(batch_size, sequence_length, dim)``.
        :type input_shape: Union[Tuple, List[Tuple]]

        :raises ValueError: If the shapes are not a list of at least 2 rank-3
            shapes whose last axis is ``dim``, or if the strategy key is
            unknown.
        """
        if self.built:
            return

        self._validate_input_shapes(input_shape)

        # One builder per strategy key; 'addition' and 'multiplication' share
        strategy_builders = {
            'cross_attention': self._build_cross_attention,
            'concatenation': self._build_concatenation,
            'addition': self._build_elementwise,
            'multiplication': self._build_elementwise,
            'gated': self._build_gated,
            'attention_pooling': self._build_attention_pooling,
            'bilinear': self._build_bilinear,
            'tensor_fusion': self._build_tensor_fusion
        }

        builder = strategy_builders.get(self.fusion_strategy)
        if builder is None:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        builder(input_shape)

        super().build(input_shape)

    def _validate_input_shapes(self, input_shape: Union[Tuple, List]) -> None:
        """Check the modality shapes before any sub-layer is created.

        Four checks: the argument is a list or tuple, it is non-empty and
        holds shapes rather than integers, it holds at least 2 modalities,
        and every shape is rank 3 with a last axis of ``dim``. Sequence
        lengths are not compared here; that is
        :meth:`_require_equal_sequence_lengths`, which runs at call time.

        :param input_shape: The per-modality shapes handed to :meth:`build`.
        :type input_shape: Union[Tuple, List]

        :raises ValueError: If any of the four checks fails.
        """
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError("Expected list or tuple of input shapes")

        if not input_shape or not hasattr(input_shape[0], '__len__'):
            raise ValueError("Expected non-empty list of tensor shapes")

        if len(input_shape) < 2:
            raise ValueError(f"Expected at least 2 modalities, got {len(input_shape)}")

        # Every modality must be rank 3 and already at width dim
        for i, shape in enumerate(input_shape):
            if len(shape) != 3:
                raise ValueError(
                    f"Expected 3D shape (batch, seq, dim) for modality {i}, "
                    f"got shape {shape}"
                )
            if shape[-1] != self.dim:
                raise ValueError(
                    f"Modality {i} dimension {shape[-1]} doesn't match "
                    f"expected dim {self.dim}"
                )

    def _build_cross_attention(self, input_shape: List[Tuple]) -> None:
        """Build the sub-layers for ``'cross_attention'``.

        Creates ``num_fusion_layers`` blocks. Each block holds, per modality
        ``i``: one cross-attention layer for every other modality ``j``, one
        normalization layer, and one FFN. The attention layers are stored on
        the block as ``attention_i``, the rest as ``norm_i`` and ``ffn_i``.

        This is the only strategy that builds more than one block, and the
        only one whose sub-layers do not go into ``self.norm_layers`` and
        ``self.ffn_layers``.

        :param input_shape: One shape per modality.
        :type input_shape: List[Tuple]
        """
        num_modalities = len(input_shape)

        for layer_idx in range(self.num_fusion_layers):
            # A plain Layer used as a named container, so the block's
            # sub-layers are tracked and serialized together
            block = keras.layers.Layer(name=f'fusion_block_{layer_idx}')

            for i in range(num_modalities):
                # One attention layer for each other modality i attends to
                attn_layers = []
                for j in range(num_modalities):
                    # A modality never attends to itself here
                    if i != j:
                        attn_config = self.attention_config.copy()
                        attn_config.setdefault('dim', self.dim)

                        attn_layer = create_attention_layer(
                            attention_type='multi_head_cross',
                            name=f'cross_attn_{layer_idx}_{i}_to_{j}',
                            **attn_config
                        )
                        # Query shape from modality i, key/value from j
                        attn_layer.build([input_shape[i], input_shape[j]])
                        attn_layers.append(attn_layer)

                setattr(block, f'attention_{i}', attn_layers)

                # One normalization layer per modality
                norm_layer = create_normalization_layer(
                    normalization_type=self.norm_type,
                    name=f'norm_{layer_idx}_{i}',
                    **self.norm_config
                )
                norm_layer.build(input_shape[i])
                setattr(block, f'norm_{i}', norm_layer)

                # One FFN per modality; output_dim is forced back to dim
                ffn_config = self.ffn_config.copy()
                ffn_config.setdefault('hidden_dim', self.dim * 4)
                ffn_config['output_dim'] = self.dim

                ffn_layer = create_ffn_layer(
                    ffn_type=self.ffn_type,
                    name=f'ffn_{layer_idx}_{i}',
                    **ffn_config
                )
                ffn_layer.build(input_shape[i])
                setattr(block, f'ffn_{i}', ffn_layer)

            self.fusion_layers.append(block)

    def _build_concatenation(self, input_shape: List[Tuple]) -> None:
        """Build the three sub-layers for ``'concatenation'``.

        One ``Dense(dim, activation)`` over the ``dim * N`` concatenation,
        one normalization layer, one Dropout. This is the only strategy that
        builds a Dropout layer, so ``dropout_rate`` does nothing elsewhere.

        :param input_shape: One shape per modality.
        :type input_shape: List[Tuple]
        """
        num_modalities = len(input_shape)

        # Projection from the concatenated width back down to dim
        proj_layer = keras.layers.Dense(
            units=self.dim,
            activation=self.activation,
            name='concat_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # The concatenation widens the last axis to dim * N
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities
        proj_layer.build(tuple(concat_shape))
        self.projection_layers.append(proj_layer)

        norm_layer = create_normalization_layer(
            normalization_type=self.norm_type,
            name='concat_norm',
            **self.norm_config
        )
        output_shape = proj_layer.compute_output_shape(tuple(concat_shape))
        norm_layer.build(output_shape)
        self.norm_layers.append(norm_layer)

        self.dropout_layers.append(
            keras.layers.Dropout(self.dropout_rate, name='concat_dropout')
        )

    def _build_elementwise(self, input_shape: List[Tuple]) -> None:
        """Build the sub-layers for ``'addition'`` and ``'multiplication'``.

        Both keys share this builder. It always creates one normalization
        layer and one FFN. Above 2 modalities it also creates one alignment
        ``Dense(dim)`` per modality, so ``self.projection_layers`` is empty
        at exactly 2 modalities and length N above that. ``_call_elementwise``
        branches on that emptiness rather than on the count.

        :param input_shape: One shape per modality.
        :type input_shape: List[Tuple]
        """
        num_modalities = len(input_shape)

        # Above 2 modalities, project each into a common space first
        if num_modalities > 2:
            for i in range(num_modalities):
                proj = keras.layers.Dense(
                    self.dim,
                    name=f'align_projection_{i}',
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer
                )
                proj.build(input_shape[i])
                self.projection_layers.append(proj)

        norm_layer = create_normalization_layer(
            normalization_type=self.norm_type,
            name='elementwise_norm',
            **self.norm_config
        )
        norm_layer.build(input_shape[0])
        self.norm_layers.append(norm_layer)

        # The FFN runs after the merge; output_dim is forced back to dim
        ffn_config = self.ffn_config.copy()
        ffn_config.setdefault('hidden_dim', self.dim * 4)
        ffn_config['output_dim'] = self.dim

        ffn_layer = create_ffn_layer(
            ffn_type=self.ffn_type,
            name='elementwise_ffn',
            **ffn_config
        )
        ffn_layer.build(input_shape[0])
        self.ffn_layers.append(ffn_layer)

    def _build_gated(self, input_shape: List[Tuple]) -> None:
        """Build layers for gated fusion strategy.

        Creates gates that learn to weight each modality's contribution.

        :param input_shape: List of input shapes for each modality.
        :type input_shape: List[Tuple]
        """
        num_modalities = len(input_shape)

        # One sigmoid gate per modality, each reading only its own input
        for i in range(num_modalities):
            gate = keras.layers.Dense(
                units=self.dim,
                # Sigmoid keeps the gate in [0, 1]
                activation='sigmoid',
                name=f'gate_{i}',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )
            gate.build(input_shape[i])
            self.gate_layers.append(gate)

        # Projection from the gated dim * N concatenation back to dim
        proj = keras.layers.Dense(
            units=self.dim,
            activation=self.activation,
            name='gated_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities
        proj.build(tuple(concat_shape))
        self.projection_layers.append(proj)

        norm = create_normalization_layer(
            normalization_type=self.norm_type,
            name='gated_norm',
            **self.norm_config
        )
        output_shape = proj.compute_output_shape(tuple(concat_shape))
        norm.build(output_shape)
        self.norm_layers.append(norm)

    def _build_attention_pooling(self, input_shape: List[Tuple]) -> None:
        """Build layers for attention pooling fusion strategy.

        Uses self-attention to pool features before fusion.

        :param input_shape: List of input shapes for each modality.
        :type input_shape: List[Tuple]
        """
        num_modalities = len(input_shape)

        # One attention layer per modality, used as self-attention
        for i in range(num_modalities):
            attn_config = self.attention_config.copy()
            attn_config.setdefault('dim', self.dim)

            attn_layer = create_attention_layer(
                attention_type='multi_head_cross',
                name=f'pool_attention_{i}',
                **attn_config
            )
            # Same shape for query and key/value makes it self-attention
            attn_layer.build([input_shape[i], input_shape[i]])
            self.fusion_layers.append(attn_layer)

        # Projection over the concatenated pooled vectors
        proj = keras.layers.Dense(
            units=self.dim,
            activation=self.activation,
            name='pool_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )
        # Mean-pooling drops the sequence axis, so this shape is rank 2
        pooled_shape = (input_shape[0][0], self.dim * num_modalities)
        proj.build(pooled_shape)
        self.projection_layers.append(proj)

    def _build_bilinear(self, input_shape: List[Tuple]) -> None:
        """Build layers for bilinear pooling fusion strategy.

        Computes outer product between modalities.

        :param input_shape: List of input shapes for each modality.
        :type input_shape: List[Tuple]

        :raises ValueError: If not exactly 2 modalities provided.
        """
        num_modalities = len(input_shape)
        if num_modalities != 2:
            raise ValueError(
                f"Bilinear fusion requires exactly 2 modalities, got {num_modalities}"
            )

        # Projection over the flattened outer product
        proj = keras.layers.Dense(
            units=self.dim,
            activation=self.activation,
            name='bilinear_projection',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        # The outer product is dim x dim, flattened to dim * dim
        batch_size, seq_len, _ = input_shape[0]
        bilinear_flat_shape = (batch_size, seq_len, self.dim * self.dim)
        proj.build(bilinear_flat_shape)
        self.projection_layers.append(proj)

        norm = create_normalization_layer(
            normalization_type=self.norm_type,
            name='bilinear_norm',
            **self.norm_config
        )
        output_shape = proj.compute_output_shape(bilinear_flat_shape)
        norm.build(output_shape)
        self.norm_layers.append(norm)

    def _build_tensor_fusion(self, input_shape: List[Tuple]) -> None:
        """Build layers for the ``'tensor_fusion'`` strategy.

        Creates ``num_tensor_projections`` parallel ``Dense(dim, activation)``
        layers over the feature-axis concatenation of all modalities, plus one
        linear ``Dense(dim)`` over their concatenated outputs. This is the
        hidden and output layer of an MLP -- there is no outer product and no
        decomposition; see ``_call_tensor_fusion``.

        :param input_shape: List of input shapes for each modality.
        :type input_shape: List[Tuple]
        """
        num_modalities = len(input_shape)

        # The concatenation widens the last axis to dim * N
        concat_shape = list(input_shape[0])
        concat_shape[-1] = self.dim * num_modalities

        # Parallel hidden units. Each sees the SAME concatenated input, so
        # this is one wide Dense(dim * num_tensor_projections) written as a
        # list. It is not a decomposition of anything.
        for i in range(self.num_tensor_projections):
            proj = keras.layers.Dense(
                units=self.dim,
                activation=self.activation,
                name=f'tensor_proj_{i}',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer
            )
            proj.build(tuple(concat_shape))
            self.projection_layers.append(proj)

        # The output layer. No activation, unlike the P hidden ones
        final_proj = keras.layers.Dense(
            units=self.dim,
            name='tensor_final_proj',
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer
        )

        final_concat_shape = list(input_shape[0])
        final_concat_shape[-1] = self.dim * self.num_tensor_projections
        final_proj.build(tuple(final_concat_shape))
        self.projection_layers.append(final_proj)

    def call(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]:
        """Fuse the modality tensors with the configured strategy.

        Checks the inputs, looks up the handler, runs the equal-length guard
        unless the strategy is exempt, then delegates. The guard runs here and
        nowhere else, so every length-sensitive strategy gets it.

        :param inputs: One tensor per modality, each
            ``(batch_size, sequence_length, dim)``. At least 2 are required.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Keras training flag, passed down to the sub-layers.
        :type training: Optional[bool]

        :return: The fused features. ``'cross_attention'`` returns a tuple
            with one tensor per modality, ``'attention_pooling'`` returns
            ``(batch_size, dim)``, and every other strategy returns
            ``(batch_size, sequence_length, dim)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, ...]]

        :raises ValueError: If ``inputs`` is not a list or tuple of at least
            2 tensors, if the strategy key is unknown, or if the guard finds
            unequal statically-known sequence lengths.
        """
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("Expected list or tuple of input tensors")
        if len(inputs) < 2:
            raise ValueError(f"Expected at least 2 modalities, got {len(inputs)}")

        # One handler per strategy key; 'addition' and 'multiplication' share
        strategy_handlers = {
            'cross_attention': self._call_cross_attention,
            'concatenation': self._call_concatenation,
            'addition': self._call_elementwise,
            'multiplication': self._call_elementwise,
            'gated': self._call_gated,
            'attention_pooling': self._call_attention_pooling,
            'bilinear': self._call_bilinear,
            'tensor_fusion': self._call_tensor_fusion
        }

        handler = strategy_handlers.get(self.fusion_strategy)
        if handler is None:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        if self.fusion_strategy not in self.LENGTH_AGNOSTIC_STRATEGIES:
            self._require_equal_sequence_lengths(inputs)

        return handler(inputs, training)

    #: The two strategies that can fuse modalities of DIFFERENT sequence
    #: length. Both route through attention layers that accept a query and a
    #: key of different length. Every other strategy either concatenates on
    #: the feature axis or combines element-wise on axis 1, so it needs equal
    #: lengths. `call()` reads this to decide whether to run the guard.
    LENGTH_AGNOSTIC_STRATEGIES = frozenset({'cross_attention', 'attention_pooling'})

    def _require_equal_sequence_lengths(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
    ) -> None:
        """Refuse statically-unequal sequence lengths, naming the strategy.

        Interface contract. One call site, :meth:`call`, guarding the six
        strategy keys outside :attr:`LENGTH_AGNOSTIC_STRATEGIES`. Raises
        ``ValueError`` when every input has a statically-known sequence
        length and they are not all equal. Returns ``None`` otherwise.

        A symbolic build with a ``None`` sequence axis is legal and is NEVER
        refused here. The check stays out of that case instead of guessing.

        **Decision table:**

        .. code-block:: text

            condition                          result
            ---------------------------------  -----------
            any input has rank <= 2            return
            any input has shape[1] is None     return
            all known lengths equal            return
            known lengths differ               ValueError

            The first two rows come from one test: the guard
            collects only statically-known lengths, then bails
            out when it collected fewer than one per input.

        :param inputs: The modality tensors, as passed to :meth:`call`.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]

        :raises ValueError: If the statically-known sequence lengths differ.
        """
        lengths = [
            t.shape[1] for t in inputs
            if len(t.shape) > 2 and t.shape[1] is not None
        ]
        if len(lengths) != len(inputs) or len(set(lengths)) <= 1:
            return

        raise ValueError(
            f"fusion_strategy='{self.fusion_strategy}' requires all modality "
            "inputs to share the same sequence length, because it combines them "
            "on the feature axis (concatenation) or element-wise on the sequence "
            f"axis; got sequence lengths {lengths} for inputs of shapes "
            f"{[tuple(t.shape) for t in inputs]}. Use "
            "fusion_strategy='cross_attention' or 'attention_pooling' for "
            "modalities of different sequence length."
        )

    def _call_cross_attention(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, ...]:
        """Fuse by letting every modality attend to every other one.

        Each modality keeps its own output tensor, so this is the only
        strategy that returns a tuple rather than one fused tensor. It is
        also the only one that stacks more than one block and the only one
        that reads ``use_residual``.

        **Block internals, for one modality i in one block:**

        .. code-block:: text

              outputs[i]  (B, Ti, dim)   outputs[j], j != i
                    │                          │
                    └────────────┬─────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │   cross_attn i to j   │  N-1 of them
                     └───────────┬───────────┘
                                 ▼
                      stack over j, then mean
                       combined  (B, Ti, dim)
                                 │
                  ┌──────────────┴──────────────┐
              use_residual                not use_residual
                  │                              │
                  ▼                              ▼
          outputs[i] + combined               combined
                  └──────────────┬──────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │        norm_i         │
                     └───────────┬───────────┘
                                 ▼  normalized
                     ┌───────────────────────┐
                     │        ffn_i          │
                     └───────────┬───────────┘
                                 ▼  ffn_out
                  ┌──────────────┴──────────────┐
              use_residual                not use_residual
                  │                              │
                  ▼                              ▼
          normalized + ffn_out                ffn_out
                  └──────────────┬──────────────┘
                                 ▼
                    new_outputs[i]  (B, Ti, dim)

            The second residual adds to `normalized`, not to the
            block input. The two forks are written differently in
            the code: the first has no else branch and simply
            skips the addition, the second has an explicit else.
            Both leaves are real.

        :param inputs: One tensor per modality.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Keras training flag.
        :type training: Optional[bool]

        :return: One refined tensor per modality, in input order.
        :rtype: Tuple[keras.KerasTensor, ...]
        """
        outputs = list(inputs)
        num_modalities = len(inputs)

        # Each block reads the previous block's outputs
        for layer_idx in range(self.num_fusion_layers):
            block = self.fusion_layers[layer_idx]
            new_outputs = []

            for i in range(num_modalities):
                # Modality i attends to each of the other N-1 modalities
                attended_features = []
                attention_layers = getattr(block, f'attention_{i}')
                attention_idx = 0

                for j in range(num_modalities):
                    if i != j:
                        attention_layer = attention_layers[attention_idx]
                        attended = attention_layer(
                            query_input=outputs[i],
                            kv_input=outputs[j],
                            training=training
                        )
                        attended_features.append(attended)
                        attention_idx += 1

                # Average the N-1 attended tensors: stack on a new leading
                # axis, then reduce it away
                combined = keras.ops.mean(
                    keras.ops.stack(attended_features, axis=0),
                    axis=0
                )

                # First residual. There is no else: when use_residual is
                # False, `combined` goes on unchanged
                if self.use_residual:
                    combined = outputs[i] + combined

                norm_layer = getattr(block, f'norm_{i}')
                normalized = norm_layer(combined, training=training)

                ffn_layer = getattr(block, f'ffn_{i}')
                ffn_out = ffn_layer(normalized, training=training)

                # Second residual, around the FFN only. This one has an
                # explicit else, unlike the first
                if self.use_residual:
                    output = normalized + ffn_out
                else:
                    output = ffn_out

                new_outputs.append(output)

            outputs = new_outputs

        return tuple(outputs)

    def _call_concatenation(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Fuse by concatenating on the feature axis, then projecting.

        A straight line with no branches. This is the only strategy that
        applies dropout.

        **Block internals:**

        .. code-block:: text

            x_1 ... x_N   each (B, T, dim)
                      │
                      ▼
            concatenate axis=-1   (B, T, dim*N)
                      │
                      ▼
            ┌─────────────────────────────┐
            │   projection_layers[0]      │
            │   Dense(dim, activation)    │
            └──────────────┬──────────────┘
                           ▼  (B, T, dim)
            ┌─────────────────────────────┐
            │   norm_layers[0]            │
            └──────────────┬──────────────┘
                           ▼
            ┌─────────────────────────────┐
            │   dropout_layers[0]         │
            └──────────────┬──────────────┘
                           ▼
                     (B, T, dim)

        :param inputs: One tensor per modality, all the same length.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Keras training flag. Dropout reads it.
        :type training: Optional[bool]

        :return: The fused tensor, ``(B, T, dim)``.
        :rtype: keras.KerasTensor
        """
        # Widen the last axis to dim * N
        concatenated = keras.ops.concatenate(inputs, axis=-1)

        # Project back down to dim
        output = self.projection_layers[0](concatenated, training=training)

        output = self.norm_layers[0](output, training=training)

        output = self.dropout_layers[0](output, training=training)

        return output

    def _call_elementwise(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Fuse by summing or multiplying the modalities element-wise.

        Shared by ``'addition'`` and ``'multiplication'``. Two branches: one
        on whether alignment projections exist, one on which of the two keys
        is configured.

        **Block internals:**

        .. code-block:: text

            x_1 ... x_N   each (B, T, dim)
                        │
              ┌─────────┴──────────┐
        projection_layers      the list is empty
        is non-empty (N > 2)   (exactly 2 modalities)
              │                        │
              ▼                        ▼
        align Dense per modality   inputs unchanged
              └─────────┬──────────────┘
                        ▼
                    aligned    each (B, T, dim)
                        │
              ┌─────────┴──────────┐
         'addition'          'multiplication'
              │                    │
              ▼                    ▼
        sum(stack(aligned))   aligned[0] * a[1] * ...
                              a pairwise loop, not
                              one N-ary op
              └─────────┬──────────┘
                        ▼  (B, T, dim)
              ┌───────────────────┐
              │  norm_layers[0]   │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │  ffn_layers[0]    │
              └─────────┬─────────┘
                        ▼
                  (B, T, dim)

            The first fork tests the list for emptiness, not the
            modality count. The two agree because
            _build_elementwise only creates alignment projections
            above 2 modalities.

        :param inputs: One tensor per modality, all the same length.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Keras training flag.
        :type training: Optional[bool]

        :return: The fused tensor, ``(B, T, dim)``.
        :rtype: keras.KerasTensor
        """
        # Non-empty only above 2 modalities; see _build_elementwise
        if self.projection_layers:
            aligned = [
                proj(inp, training=training)
                for proj, inp in zip(self.projection_layers, inputs)
            ]
        else:
            aligned = list(inputs)

        if self.fusion_strategy == 'addition':
            # Stack on a new leading axis, then sum it away
            output = keras.ops.sum(
                keras.ops.stack(aligned, axis=0),
                axis=0
            )
        # The only other key routed here is 'multiplication'
        else:
            # Pairwise product, folded left to right
            output = aligned[0]
            for inp in aligned[1:]:
                output = keras.ops.multiply(output, inp)

        output = self.norm_layers[0](output, training=training)

        output = self.ffn_layers[0](output, training=training)

        return output

    def _call_gated(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Fuse after scaling each modality by its own learned gate.

        Each gate reads only its own modality, so a gate cannot suppress a
        feature because of what another modality is doing. No branches. No
        FFN and no dropout on this path.

        **Block internals:**

        .. code-block:: text

            per modality i:

              x_i  (B, T, dim)
                 ├──────────────┐
                 │              ▼
                 │      ┌─────────────────┐
                 │      │ gate_layers[i]  │
                 │      │ Dense(sigmoid)  │
                 │      └────────┬────────┘
                 │               ▼  gate in [0, 1]
                 └─────► multiply
                              │  (B, T, dim)
                              ▼
                          gated_i

            then once, over all modalities:

              concatenate gated_1..N   (B, T, dim*N)
                              │
                              ▼
                  ┌────────────────────────┐
                  │  projection_layers[0]  │
                  │  Dense(dim, activation)│
                  └───────────┬────────────┘
                              ▼  (B, T, dim)
                  ┌────────────────────────┐
                  │  norm_layers[0]        │
                  └───────────┬────────────┘
                              ▼
                        (B, T, dim)

        :param inputs: One tensor per modality, all the same length.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Keras training flag.
        :type training: Optional[bool]

        :return: The fused tensor, ``(B, T, dim)``.
        :rtype: keras.KerasTensor
        """
        gated_features = []
        for i, inp in enumerate(inputs):
            # Sigmoid keeps the gate in [0, 1]
            gate_values = self.gate_layers[i](inp, training=training)
            gated = keras.ops.multiply(inp, gate_values)
            gated_features.append(gated)

        # Widen the last axis to dim * N
        concatenated = keras.ops.concatenate(gated_features, axis=-1)

        # Project back down to dim
        output = self.projection_layers[0](concatenated, training=training)

        output = self.norm_layers[0](output, training=training)

        return output

    def _call_attention_pooling(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Self-attend each modality, mean-pool it, then fuse the vectors.

        This is the only strategy whose output has no sequence axis: the mean
        over axis 1 removes it, so the result is ``(B, dim)``. It is also the
        shortest path here, with no normalization, no FFN and no dropout.

        Each modality is pooled on its own before the modalities meet, so
        they may have different sequence lengths. That is why this key is in
        :attr:`LENGTH_AGNOSTIC_STRATEGIES`.

        **Block internals:**

        .. code-block:: text

            per modality i:

              x_i  (B, Ti, dim)
                    │
                    ▼
              ┌──────────────────────────┐
              │  fusion_layers[i]        │
              │  self-attn: query = kv   │
              └────────────┬─────────────┘
                           ▼  (B, Ti, dim)
                    mean over axis 1
                           ▼
                pooled_i  (B, dim)   sequence axis gone

            then once, over all modalities:

              concatenate pooled_1..N   (B, dim*N)
                           │
                           ▼
              ┌──────────────────────────┐
              │  projection_layers[0]    │
              │  Dense(dim, activation)  │
              └────────────┬─────────────┘
                           ▼
                       (B, dim)

        :param inputs: One tensor per modality. Lengths may differ.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Keras training flag.
        :type training: Optional[bool]

        :return: The fused vector, ``(B, dim)``.
        :rtype: keras.KerasTensor
        """
        pooled_features = []

        for i, inp in enumerate(inputs):
            # Same tensor as query and as key/value, so this is self-attention
            attention_layer = self.fusion_layers[i]
            attended_output = attention_layer(
                query_input=inp,
                kv_input=inp,
                training=training
            )

            # Mean over the sequence axis; the result is rank 2
            pooled = keras.ops.mean(attended_output, axis=1)
            pooled_features.append(pooled)

        # Widen the last axis to dim * N
        concatenated = keras.ops.concatenate(pooled_features, axis=-1)

        # Project back down to dim
        output = self.projection_layers[0](concatenated, training=training)

        return output

    def _call_bilinear(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Fuse exactly two modalities through their outer product.

        This is the only genuine outer product in this file. Every pair of
        features gets its own product term, so the projection reads a
        ``dim * dim`` vector per token and the parameter count grows with
        ``dim ** 3``. That is what makes this the most expensive strategy
        here for any large ``dim``.

        Do not confuse this with ``'tensor_fusion'``, which forms no product
        at all.

        **Block internals:**

        .. code-block:: text

            x1  (B, T, dim)           x2  (B, T, dim)
                 │                         │
                 ▼                         ▼
           expand_dims(-1)           expand_dims(-2)
           (B, T, dim, 1)            (B, T, 1, dim)
                 └────────────┬────────────┘
                              ▼
                     multiply, broadcasting
                    = outer product per token
                        (B, T, dim, dim)
                              │
                              ▼
                   reshape (B, T, dim*dim)
                              │
                              ▼
                ┌──────────────────────────┐
                │  projection_layers[0]    │
                │  Dense(dim, activation)  │
                └────────────┬─────────────┘
                             ▼  (B, T, dim)
                ┌──────────────────────────┐
                │  norm_layers[0]          │
                └────────────┬─────────────┘
                             ▼
                        (B, T, dim)

        :param inputs: Exactly 2 tensors, both the same length.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Keras training flag.
        :type training: Optional[bool]

        :return: The fused tensor, ``(B, T, dim)``.
        :rtype: keras.KerasTensor
        """
        x1, x2 = inputs

        # Line up the two feature axes so they broadcast against each other:
        # x1 becomes (..., dim, 1) and x2 becomes (..., 1, dim)
        x1_expanded = keras.ops.expand_dims(x1, axis=-1)
        x2_expanded = keras.ops.expand_dims(x2, axis=-2)

        # The broadcast product is the outer product, (B, T, dim, dim)
        bilinear = keras.ops.multiply(x1_expanded, x2_expanded)

        # Flatten the dim x dim matrix into one dim*dim vector per token
        batch_size = keras.ops.shape(bilinear)[0]
        seq_len = keras.ops.shape(bilinear)[1]
        bilinear_flat = keras.ops.reshape(
            bilinear,
            [batch_size, seq_len, -1]
        )

        # Project back down to dim
        output = self.projection_layers[0](bilinear_flat, training=training)

        output = self.norm_layers[0](output, training=training)

        return output

    def _call_tensor_fusion(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Fuse by concatenating, then running a wide hidden layer.

        **What this actually computes.** Despite the name, there is no outer
        product and no tensor decomposition anywhere in this method. It is a
        one-hidden-layer MLP over the feature-axis concatenation::

            h        = concat([x_1, ..., x_N], axis=-1)     # (B, T, N*dim)
            u_i      = activation(h @ W_i + b_i)            # i = 1..P
            output   = concat([u_1, ..., u_P], -1) @ W_out + b_out

        where ``P = num_tensor_projections``. Every ``u_i`` reads the same
        ``h``, so the parallel list is one wide hidden layer of width
        ``P * dim`` written as ``P`` separate ``Dense`` layers.

        **Block internals:**

        .. code-block:: text

            x_1 ... x_N   each (B, T, dim)
                       │
                       ▼
             concatenate axis=-1   (B, T, dim*N)
                       │
             ┌─────────┼─────────┐   P branches, all
             ▼         ▼         ▼   reading the SAME
        ┌─────────┐┌─────────┐┌─────────┐  concatenation
        │ proj_0  ││ proj_1  ││ proj_P-1│
        │Dense dim││Dense dim││Dense dim│
        │ + activ ││ + activ ││ + activ │
        └────┬────┘└────┬────┘└────┬────┘
             └─────────┬┴──────────┘
                       ▼
             concatenate axis=-1   (B, T, dim*P)
                       │
                       ▼
             ┌───────────────────────────┐
             │  projection_layers[-1]    │
             │  Dense(dim), NO activation│
             └─────────────┬─────────────┘
                           ▼
                     (B, T, dim)

            No normalization, no dropout, no residual anywhere on
            this path.

        Note:
            The ``if projections:`` test in the code below is never False.
            :meth:`_validate_init_params` refuses a non-positive
            ``num_tensor_projections`` under this strategy, so the loop
            always appends at least one entry. The ``else`` branch is dead
            code and is not a path this layer can take.

        **Relation to ``'concatenation'``.** Up to that hidden layer, this is
        the same model. ``'concatenation'`` is ``activation(h @ W) -> norm ->
        dropout``: the same concatenation with a width-``dim`` hidden layer
        and no output projection. So ``'tensor_fusion'`` buys extra hidden
        width and one more linear map, not a different kind of interaction.
        It also applies no normalization and no dropout, unlike
        ``'concatenation'``. For genuine multiplicative cross-modal
        interaction use ``'bilinear'``, which is a real outer product, or
        ``'multiplication'`` / ``'gated'``.

        **Cost.** MEASURED trainable parameters, 2 modalities, ``dim=64``,
        default ``num_tensor_projections=8``: ``concatenation`` 8,384;
        ``tensor_fusion`` 98,880; ``bilinear`` 262,336. The ``tensor_fusion``
        count is EXACT, not asymptotic::

            P * (N*dim*dim + dim)   # the P hidden projections
          + (P*dim*dim + dim)       # the output projection
          = P * (N + 1) * dim^2 + (P + 1) * dim

        i.e. ``24 * dim^2 + 9 * dim`` at ``P=8, N=2``. Verified against
        constructed layers at four settings: ``(dim,N,P) = (64,2,8) -> 98,880``,
        ``(96,2,8) -> 222,048``, ``(48,3,4) -> 37,104``, ``(64,3,8) -> 131,648``,
        all exact. So the growth is quadratic in ``dim`` and LINEAR in both ``N``
        and ``P`` -- like every other strategy here except ``'bilinear'``, which
        is ``dim^3 + 3*dim``. ``tensor_fusion`` is therefore roughly ``12x``
        concatenation but *cheaper* than ``'bilinear'`` for any ``dim > 24``
        (at ``dim=24``: 14,040 vs 13,896; at ``dim=25``: 15,225 vs 15,700); it is
        not the most expensive strategy in this layer.

        :param inputs: One tensor per modality, all the same length.
        :type inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, ...]]
        :param training: Keras training flag.
        :type training: Optional[bool]

        :return: The fused tensor, ``(B, T, dim)``.
        :rtype: keras.KerasTensor

        :raises ValueError: Raised before this method runs, by
            :meth:`_require_equal_sequence_lengths`, if the modalities have
            statically-known but unequal sequence lengths.
        """
        # DECISION plan-2026-08-14T183218-f4c612aa/D-007: unequal sequence
        # lengths must raise a named ValueError. Do NOT fall back on the
        # backend error (197 vision vs 8 text token shapes gave a bare
        # InvalidArgumentError), and do NOT pad or slice to a common length.
        # See decisions.md D-007.

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-024: that check now lives
        # in `_require_equal_sequence_lengths`, called once from `call()`.
        # Do NOT re-inline it here. A per-method copy is how 5 of the 6
        # length-sensitive strategies ended up unguarded.
        # See decisions.md D-024.

        # Widen the last axis to dim * N
        concatenated = keras.ops.concatenate(inputs, axis=-1)

        # One wide hidden layer, written as P parallel Dense layers over the
        # SAME concatenation. Not a decomposition, not an outer product.
        projections = []
        for i in range(self.num_tensor_projections):
            proj_layer = self.projection_layers[i]
            projection = proj_layer(concatenated, training=training)
            projections.append(projection)

        # The else branch is dead: num_tensor_projections is validated
        # positive for this strategy, so `projections` is never empty
        if projections:
            combined = keras.ops.concatenate(projections, axis=-1)
        else:
            combined = concatenated

        # The output layer, linear, back down to dim
        output = self.projection_layers[-1](combined, training=training)

        return output

    def compute_output_shape(
        self,
        input_shape: Union[Tuple, List[Tuple]]
    ) -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
        """Report the output shape, which depends on the strategy.

        Three cases. ``'cross_attention'`` hands the input shapes straight
        back, one per modality. ``'attention_pooling'`` drops the sequence
        axis. Everything else keeps the batch and sequence axes and sets the
        last one to ``dim``.

        :param input_shape: One shape per modality.
        :type input_shape: Union[Tuple, List[Tuple]]

        :return: The per-strategy output shape, as described above.
        :rtype: Union[Tuple[int, ...], List[Tuple[int, ...]]]
        """
        if self.fusion_strategy == 'cross_attention':
            # One tensor per modality, each keeping its own shape
            return input_shape
        elif self.fusion_strategy == 'attention_pooling':
            # Mean-pooling removed the sequence axis
            return (input_shape[0][0], self.dim)
        else:
            # Batch and sequence axes survive; the last axis becomes dim
            return (input_shape[0][0], input_shape[0][1], self.dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments, for serialization.

        Every ``__init__`` parameter is stored. The activation, initializers
        and regularizers are serialized to their dict form, so
        :meth:`from_config` has to deserialize them again.

        :return: The layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'fusion_strategy': self.fusion_strategy,
            'num_fusion_layers': self.num_fusion_layers,
            'attention_config': self.attention_config,
            'ffn_type': self.ffn_type,
            'ffn_config': self.ffn_config,
            'norm_type': self.norm_type,
            'norm_config': self.norm_config,
            'num_tensor_projections': self.num_tensor_projections,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual,
            'activation': keras.activations.serialize(self.activation),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MultiModalFusion':
        """Rebuild the layer from a configuration dict.

        Reverses what :meth:`get_config` serialized. The dict is modified in
        place, so pass a copy if you need to keep the original.

        :param config: A dict as produced by :meth:`get_config`.
        :type config: Dict[str, Any]

        :return: A new layer with that configuration.
        :rtype: MultiModalFusion
        """
        # Turn the serialized dicts back into objects
        config['activation'] = keras.activations.deserialize(config['activation'])
        config['kernel_initializer'] = keras.initializers.deserialize(config['kernel_initializer'])
        config['bias_initializer'] = keras.initializers.deserialize(config['bias_initializer'])
        config['kernel_regularizer'] = keras.regularizers.deserialize(config.get('kernel_regularizer'))
        config['bias_regularizer'] = keras.regularizers.deserialize(config.get('bias_regularizer'))

        return cls(**config)

# ---------------------------------------------------------------------
