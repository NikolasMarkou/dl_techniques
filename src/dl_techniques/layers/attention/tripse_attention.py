"""
TripSE: Triplet Squeeze and Excitation attention, four fusion variants.

Combines Triplet Attention (rotate a plane, pool it, gate it spatially) with
Squeeze-and-Excitation (global channel gating) into one 3D attention block.
The four classes, :class:`TripSE1` through :class:`TripSE4`, share both
primitives and differ only in where the SE block sits relative to the three
triplet branches and how the branches are combined:

.. code-block:: text

    variant  SE position                  fusion   gate input
    -------  ---------------------------  -------  ----------
    TripSE1  after the fusion             sum      spatial
    TripSE2  per branch, before Z-pool    average  spatial
    TripSE3  per branch, parallel to it   average  spatial
    TripSE4  per branch, added to logits  sum      fused 3-D

TripSE4 is the only variant whose gate sees a genuine 3-D tensor: it adds a
channel-logit path (:class:`_SEWeights`, pre-sigmoid) to the spatial logits
before one sigmoid, rather than multiplying two already-gated maps.

None of the five classes here validate constructor arguments: a bad
``reduction_ratio`` or ``kernel_size`` surfaces later as a Keras or Conv2D
error, not at construction time. This behavior is unchanged — see
``plans/plan-2026-07-27T130643-38c5646a/decisions.md``
D-012.

References:
    - Alhazmi, A., & Altahhan, A. (2025). Achieving 3D Attention via Triplet
      Squeeze and Excitation Block. (https://arxiv.org/abs/2505.05943)
    - Misra, D., et al. (2021). Rotate to Attend: Convolutional Triplet
      Attention Module. WACV.
    - Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
      CVPR.
"""

# ---------------------------------------------------------------------

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Tuple, Any, Dict, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.conv_blocks.squeeze_excitation import SqueezeExcitation
from dl_techniques.layers.activations import resolve_activation_layer
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.attention.tripse_attention")
class TripletAttentionBranch(layers.Layer):
    """
    Single branch of the Triplet Attention mechanism.

    Captures cross-dimensional interaction by rotating tensor dimensions,
    applying Z-pooling (concatenation of channel-wise average and max),
    convolution, batch normalization, and sigmoid activation. The resulting
    spatial attention map is broadcast-multiplied onto the permuted input,
    then the inverse permutation restores the original axis order.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │  TripletAttentionBranch — one rotated-plane spatial gate │
        │                                                         │
        │  The one reusable branch. Rotate a plane forward, build  │
        │  a spatial gate from it, apply it, rotate back.          │
        └─────────────────────────────────────────────────────────┘

        Input  [B, H, W, C]
                  ▼
        permute axes, one of 3 patterns  ►  x  [B, D1, D2, D3]
                  ▼
        Z-pool over D3: concat(mean, max)   ►  [B, D1, D2, 2]
                  ▼
        ┌───────────────────────────────────────────────┐
        │ conv  Conv2D(filters=1, kernel_size=k)        │
        │ bn    BatchNormalization                      │
        │ gate  gate_activation (sigmoid by default)    │
        └───────────────────────┬───────────────────────┘
                                ▼
        attention map  [B, D1, D2, 1]
                  ▼
        x * attention map, broadcast over the D3 axis
                  ▼
        inverse permute  ►  Output  [B, H, W, C]

    :param kernel_size: Kernel size for the spatial convolution.
    :type kernel_size: int
    :param permute_pattern: Permutation of ``(H, W, C)`` axes.
        ``(0, 1, 2)`` = H-W plane, ``(0, 2, 1)`` = C-W plane,
        ``(2, 1, 0)`` = H-C plane.
    :type permute_pattern: Tuple[int, int, int]
    :param use_bias: Whether the convolution uses bias.
    :type use_bias: bool
    :param kernel_initializer: Initializer for convolution kernels.
    :type kernel_initializer: str
    :param kernel_regularizer: Regularizer for convolution kernels.
    :type kernel_regularizer: Optional[Any]
    :param gate_activation_type: Activation producing the branch attention map,
        resolved through
        :func:`~dl_techniques.layers.activations.resolve_activation_layer`.
        Defaults to ``'sigmoid'``, which is what bounds the map to ``[0, 1]``.
    :type gate_activation_type: str
    :param gate_activation_args: Optional keyword arguments forwarded to the
        gate activation layer's constructor. Defaults to ``None``.
    :type gate_activation_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: From ``build()``, if the input shape is not 4D.

    .. note::
       This ``__init__`` does not validate its arguments. See the module
       docstring for why that is unchanged rather than fixed.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        permute_pattern: Tuple[int, int, int] = (0, 1, 2),
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        gate_activation_type: str = "sigmoid",
        gate_activation_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Store the branch configuration and create its three sub-layers.

        The convolution, the batch norm and the gate activation are all created
        here; they are given shapes in :meth:`build`, once the permuted spatial
        dims are known. No argument is validated: see the module docstring's
        "Rubric R6" section.

        See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.permute_pattern = permute_pattern
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.gate_activation_type = gate_activation_type
        self.gate_activation_args = gate_activation_args

        # Layers defined in init, built in build
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv"
        )
        self.batch_norm = layers.BatchNormalization(name="bn")
        self.sigmoid = resolve_activation_layer(
            self.gate_activation_type,
            name="gate_activation",
            **(self.gate_activation_args or {}),
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers with shapes derived from the permutation pattern.

        :param input_shape: 4-D shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # input_shape is (B, H, W, C). The permutation is replayed here to get
        # D1 and D2; the conv's channel count is always 2, because Z-pooling
        # concatenates one mean map and one max map. Each sub-layer is still
        # built explicitly, which is what makes .keras round-tripping safe.
        if len(input_shape) != 4:
            raise ValueError(f"Input must be 4D, got {input_shape}")

        batch = input_shape[0]
        permuted_dims = [input_shape[i+1] for i in self.permute_pattern]

        # Conv input: (batch, D1, D2, 2).
        conv_input_shape = (batch, permuted_dims[0], permuted_dims[1], 2)
        self.conv.build(conv_input_shape)

        # BN input: (batch, D1, D2, 1).
        conv_output_shape = (batch, permuted_dims[0], permuted_dims[1], 1)
        self.batch_norm.build(conv_output_shape)

        # Gate activation operates on the BN output (B, D1, D2, 1) — see call().
        # Explicit build is required so a gate activation carrying trainable
        # params (e.g. a parametric activation) round-trips through .keras.
        self.sigmoid.build(conv_output_shape)

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        # 1. Permute
        """Rotate one plane, gate it spatially, and rotate back.

        :param inputs: 4-D input, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. Forwarded explicitly to the batch
            norm and to the gate activation; see the D-015 anchor below for why
            the gate's forward is not redundant.
        :type training: Optional[bool]

        :return: Same shape as ``inputs``, ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        """
        if self.permute_pattern != (0, 1, 2):
            # ops.transpose expects [batch_dim, ...dims]
            # permute_pattern is relative to spatial+channel dims (0,1,2)
            # We map (0,1,2,3) -> (0, p0+1, p1+1, p2+1)
            perm_order = [0] + [p + 1 for p in self.permute_pattern]
            x = ops.transpose(inputs, perm_order)
        else:
            x = inputs

        # 2. Z-Pooling (Concatenate Avg and Max along last dimension)
        # Result shape: (B, D1, D2, 2)
        avg_pool = ops.mean(x, axis=-1, keepdims=True)
        max_pool = ops.max(x, axis=-1, keepdims=True)
        pooled = ops.concatenate([avg_pool, max_pool], axis=-1)

        # 3. Attention Map Generation
        attention = self.conv(pooled)
        attention = self.batch_norm(attention, training=training)
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-015: forward training= to
        # self.sigmoid explicitly -- CallContext.training measured False here under training=True otherwise. See decisions.md.
        attention = self.sigmoid(attention, training=training)

        # 4. Apply Attention
        # x shape: (B, D1, D2, D3), attention shape: (B, D1, D2, 1)
        # Broadcasting handles the multiplication automatically
        scaled = ops.multiply(x, attention)

        # 5. Inverse Permute
        if self.permute_pattern != (0, 1, 2):
            # Calculate inverse permutation
            # Current axes order relative to original: permute_pattern
            # We need to find indices to restore 0,1,2
            inv_pattern = [0, 0, 0]
            for i, p in enumerate(self.permute_pattern):
                inv_pattern[p] = i
            
            # Add batch dim back
            inv_order = [0] + [p + 1 for p in inv_pattern]
            scaled = ops.transpose(scaled, inv_order)

        return scaled

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``input_shape`` unchanged. Attention rescales; it never reshapes.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary holding every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "permute_pattern": self.permute_pattern,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "gate_activation_type": self.gate_activation_type,
            "gate_activation_args": self.gate_activation_args,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.tripse_attention")
class TripSE1(layers.Layer):
    """
    TripSE1: Triplet Attention with Post-Fusion Squeeze-and-Excitation.

    Three parallel Triplet Attention branches (H-W, C-W, H-C planes)
    produce spatial attention maps. Their outputs are summed, and a
    Squeeze-and-Excitation block performs channel-wise recalibration on
    the fused result.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │  TripSE1 — three gated branches, sum, then a single SE   │
        └─────────────────────────────────────────────────────────┘

        Input  [B, H, W, C]
                  │
            ┌─────┴─────┬───────────┐
            ▼           ▼           ▼
        H-W plane   C-W plane   H-C plane
            ▼           ▼           ▼
        ┌─────────────────────────────────────────────────┐
        │ each branch is a TripletAttentionBranch:        │
        │   permute ► Z-pool ► Conv2D ► BN ► gate         │
        │   x * gate ► inverse permute                    │
        │ no SE anywhere inside a branch                  │
        └───────────────────────┬─────────────────────────┘
                                ▼
        element-wise sum of the 3 branch outputs
                  ▼
        ┌─────────────────────────────────────────────────┐
        │ se   SqueezeExcitation, the only SE block, runs │
        │      after the fusion                            │
        └───────────────────────┬─────────────────────────┘
                                ▼
        Output  [B, H, W, C]

    :param reduction_ratio: SE bottleneck reduction ratio.
    :type reduction_ratio: float
    :param kernel_size: Spatial convolution kernel size.
    :type kernel_size: int
    :param use_bias: Whether convolutions use bias.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: str
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[Any]
    :param gate_activation_type: Activation producing each branch's attention
        gate, resolved through
        :func:`~dl_techniques.layers.activations.resolve_activation_layer` and
        shared by all three branches. Defaults to ``'sigmoid'``, which is what
        bounds the gate to ``[0, 1]``.
    :type gate_activation_type: str
    :param gate_activation_args: Optional keyword arguments forwarded to each
        gate activation layer's constructor. Defaults to ``None``.
    :type gate_activation_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    .. note::
       This ``__init__`` does not validate its arguments. See the module
       docstring for why that is unchanged rather than fixed.
    """

    def __init__(
        self,
        reduction_ratio: float = 0.0625,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        gate_activation_type: str = "sigmoid",
        gate_activation_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Store the configuration and create every sub-layer.

        SE placement for this variant: one SqueezeExcitation after the fusion.
        The three branch outputs are summed; the three branches are
        TripletAttentionBranch instances, so this is the only variant that
        reuses that class. See the class docstring for the parameter
        reference.
        """
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.gate_activation_type = gate_activation_type
        self.gate_activation_args = gate_activation_args

        branch_act_kwargs = {
            "gate_activation_type": self.gate_activation_type,
            "gate_activation_args": self.gate_activation_args,
        }

        # Triplet Attention Branches, one per axis pair: H-W, C-W, H-C.
        self.branch_hw = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(0, 1, 2),
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="branch_hw",
            **branch_act_kwargs,
        )
        self.branch_cw = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(0, 2, 1),
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="branch_cw",
            **branch_act_kwargs,
        )
        self.branch_hc = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(2, 1, 0),
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="branch_hc",
            **branch_act_kwargs,
        )

        # SE Block (created here, built in build)
        self.se_block = SqueezeExcitation(
            reduction_ratio=reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the three branches and the post-fusion SE block.

        Sub-layers are built explicitly, in computational order, so every weight
        variable exists before Keras restores a checkpoint into it.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``None``.
        :rtype: None
        """
        if self.built:
            return

        self.branch_hw.build(input_shape)
        self.branch_cw.build(input_shape)
        self.branch_hc.build(input_shape)
        self.se_block.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Run the three branches, sum them, then recalibrate channels.

        :param inputs: 4-D input, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. Forwarded explicitly to every
            sub-layer that takes one.
        :type training: Optional[bool]

        :return: Same shape as ``inputs``, ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        """
        out_hw = self.branch_hw(inputs, training=training)
        out_cw = self.branch_cw(inputs, training=training)
        out_hc = self.branch_hc(inputs, training=training)

        # TripSE1 sums its branches; TripSE2 and TripSE3 divide by 3 instead.
        # The SE block that follows recalibrates channels, so it absorbs the
        # magnitude difference the sum introduces.
        combined = ops.add(ops.add(out_hw, out_cw), out_hc)

        output = self.se_block(combined, training=training)
        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``input_shape`` unchanged. Attention rescales; it never reshapes.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary holding every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "gate_activation_type": self.gate_activation_type,
            "gate_activation_args": self.gate_activation_args,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.tripse_attention")
class TripSE2(layers.Layer):
    """
    TripSE2: Pre-Process Squeeze-and-Excitation.

    Each branch first permutes the input tensor, applies a
    Squeeze-and-Excitation block on the permuted channels, then runs the
    Triplet Attention core (Z-Pool, Conv, BN, Sigmoid) on the SE-refined
    features. Outputs are inverse-permuted and averaged.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │  TripSE2 — per-branch SE first, gate second, average     │
        └─────────────────────────────────────────────────────────┘

        Input  [B, H, W, C]
                  │
            ┌─────┴─────┬───────────┐
            ▼           ▼           ▼
        H-W plane   C-W plane   H-C plane
            ▼           ▼           ▼
        ┌─────────────────────────────────────────────────┐
        │ each branch, SE first then gate:                │
        │   permute  ►  x  [B, D1, D2, D3]                │
        │        ▼                                        │
        │   se_*   SqueezeExcitation  ►  x_se             │
        │          channel recalibration comes first      │
        │        ▼                                        │
        │   Z-pool(x_se) ► conv_* ► bn_* ► gate_*         │
        │          the gate is a spatial map              │
        │        ▼                                        │
        │   x_se * gate  ►  inverse permute               │
        └───────────────────────┬─────────────────────────┘
                                ▼
        average of the 3 branch outputs, sum / 3
                  ▼
        Output  [B, H, W, C]. No SE after the fusion.

    :param reduction_ratio: SE bottleneck reduction ratio.
    :type reduction_ratio: float
    :param kernel_size: Spatial convolution kernel size.
    :type kernel_size: int
    :param use_bias: Whether convolutions use bias.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: str
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[Any]
    :param gate_activation_type: Activation producing each branch's attention
        gate, resolved through
        :func:`~dl_techniques.layers.activations.resolve_activation_layer` and
        shared by all three branches. Defaults to ``'sigmoid'``, which is what
        bounds the gate to ``[0, 1]``.
    :type gate_activation_type: str
    :param gate_activation_args: Optional keyword arguments forwarded to each
        gate activation layer's constructor. Defaults to ``None``.
    :type gate_activation_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    .. note::
       This ``__init__`` does not validate its arguments. See the module
       docstring for why that is unchanged rather than fixed.
    """

    def __init__(
        self,
        reduction_ratio: float = 0.0625,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        gate_activation_type: str = "sigmoid",
        gate_activation_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Store the configuration and create every sub-layer.

        SE placement for this variant: one SqueezeExcitation per branch, before
        the Z-pool. The three branch outputs are combined by averaging. See the
        class docstring for the parameter reference.
        """
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.gate_activation_type = gate_activation_type
        self.gate_activation_args = gate_activation_args

        # Written inline rather than composing TripletAttentionBranch: each
        # variant splices its SE block into a different point of the
        # permute/Z-pool/conv/gate sequence, and the branch class exposes no
        # seam for that. Each branch needs its own SE and Conv blocks too,
        # since the permuted shapes differ per axis pair.
        self._patterns = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
        self._suffixes = ["hw", "cw", "hc"]

        # Containers for sub-layers
        self.se_layers: List[SqueezeExcitation] = []
        self.conv_layers: List[layers.Conv2D] = []
        self.bn_layers: List[layers.BatchNormalization] = []
        self.gate_activations: List[keras.layers.Layer] = []

        for suffix in self._suffixes:
            self.se_layers.append(SqueezeExcitation(
                reduction_ratio=reduction_ratio,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"se_{suffix}"
            ))
            self.conv_layers.append(layers.Conv2D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"conv_{suffix}"
            ))
            self.bn_layers.append(layers.BatchNormalization(name=f"bn_{suffix}"))
            self.gate_activations.append(resolve_activation_layer(
                self.gate_activation_type,
                name=f"gate_activation_{suffix}",
                **(self.gate_activation_args or {}),
            ))

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the per-branch SE, convolution, batch norm and gate layers.

        Sub-layers are built explicitly, in computational order, so every weight
        variable exists before Keras restores a checkpoint into it.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``None``.
        :rtype: None
        """
        if self.built:
            return

        batch = input_shape[0]

        for i, pattern in enumerate(self._patterns):
            # Calculate permuted shape
            # Pattern relative to (H,W,C) -> e.g. (0,2,1) means (H,C,W)
            # Ops.transpose uses (B, H, W, C) indices [0, 1, 2, 3]
            # Pattern indices map to [1, 2, 3]
            perm_indices = [p + 1 for p in pattern]
            permuted_shape = (batch,) + tuple(input_shape[idx] for idx in perm_indices)

            # Build SE on permuted shape
            self.se_layers[i].build(permuted_shape)

            # Conv input is (B, D1, D2, 2)
            d1, d2 = permuted_shape[1], permuted_shape[2]
            self.conv_layers[i].build((batch, d1, d2, 2))
            self.bn_layers[i].build((batch, d1, d2, 1))
            self.gate_activations[i].build((batch, d1, d2, 1))

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Recalibrate channels per branch, then gate spatially, then average.

        :param inputs: 4-D input, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. Forwarded explicitly to every
            sub-layer that takes one.
        :type training: Optional[bool]

        :return: Same shape as ``inputs``, ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        """
        outputs = []
        
        for i, pattern in enumerate(self._patterns):
            # 1. Permute
            if pattern != (0, 1, 2):
                perm_order = [0] + [p + 1 for p in pattern]
                x = ops.transpose(inputs, perm_order)
            else:
                x = inputs
            
            # 2. SE Block
            x_se = self.se_layers[i](x, training=training)
            
            # 3. Triplet Attention Core
            avg_pool = ops.mean(x_se, axis=-1, keepdims=True)
            max_pool = ops.max(x_se, axis=-1, keepdims=True)
            pooled = ops.concatenate([avg_pool, max_pool], axis=-1)
            
            att = self.conv_layers[i](pooled)
            att = self.bn_layers[i](att, training=training)
            att = self.gate_activations[i](att, training=training)

            # 4. Scale
            branch_out = ops.multiply(x_se, att)
            
            # 5. Inverse Permute
            if pattern != (0, 1, 2):
                inv_pattern = [0, 0, 0]
                for idx, p in enumerate(pattern):
                    inv_pattern[p] = idx
                inv_order = [0] + [p + 1 for p in inv_pattern]
                branch_out = ops.transpose(branch_out, inv_order)
            
            outputs.append(branch_out)

        # Average results
        total = ops.add(ops.add(outputs[0], outputs[1]), outputs[2])
        return ops.divide(total, 3.0)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``input_shape`` unchanged. Attention rescales; it never reshapes.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary holding every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "gate_activation_type": self.gate_activation_type,
            "gate_activation_args": self.gate_activation_args,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.tripse_attention")
class TripSE3(layers.Layer):
    """
    TripSE3: Parallel Squeeze-and-Excitation.

    Each branch runs two parallel paths on the permuted input: a spatial
    attention path (Z-Pool, Conv, BN, Sigmoid) and a channel attention
    path (SE block). The SE-scaled features are element-wise multiplied
    by the spatial attention map, producing a joint spatial-channel
    attention. Results are inverse-permuted and averaged.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │  TripSE3 — per-branch SE and gate in parallel, average    │
        └─────────────────────────────────────────────────────────┘

        Input  [B, H, W, C]
                  │
            ┌─────┴─────┬───────────┐
            ▼           ▼           ▼
        H-W plane   C-W plane   H-C plane
            ▼           ▼           ▼
        ┌─────────────────────────────────────────────────┐
        │ each branch, two parallel paths:                │
        │   permute  ►  x  [B, D1, D2, D3]                │
        │        │      both paths read this same x        │
        │        ├───────────────────┐                    │
        │        ▼                   ▼                    │
        │   se_*(x)             Z-pool(x) ► conv_*        │
        │   x_se_scaled         ► bn_* ► gate_*           │
        │   [B, D1, D2, D3]     att_spatial               │
        │                       [B, D1, D2, 1]            │
        │        │                   │                    │
        │        └─────────┬─────────┘                    │
        │                  ▼   element-wise multiply       │
        │   x_se_scaled × att_spatial                     │
        │                  ▼                              │
        │   inverse permute                               │
        └───────────────────────┬─────────────────────────┘
                                ▼
        average of the 3 branch outputs, sum / 3
                  ▼
        Output  [B, H, W, C]. No SE after the fusion.

    :param reduction_ratio: SE bottleneck reduction ratio.
    :type reduction_ratio: float
    :param kernel_size: Spatial convolution kernel size.
    :type kernel_size: int
    :param use_bias: Whether convolutions use bias.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: str
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[Any]
    :param gate_activation_type: Activation producing each branch's attention
        gate, resolved through
        :func:`~dl_techniques.layers.activations.resolve_activation_layer` and
        shared by all three branches. Defaults to ``'sigmoid'``, which is what
        bounds the gate to ``[0, 1]``.
    :type gate_activation_type: str
    :param gate_activation_args: Optional keyword arguments forwarded to each
        gate activation layer's constructor. Defaults to ``None``.
    :type gate_activation_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    .. note::
       This ``__init__`` does not validate its arguments. See the module
       docstring for why that is unchanged rather than fixed.
    """

    def __init__(
        self,
        reduction_ratio: float = 0.0625,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        gate_activation_type: str = "sigmoid",
        gate_activation_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Store the configuration and create every sub-layer.

        SE placement for this variant: one SqueezeExcitation per branch,
        parallel to the spatial path. The two paths read the same permuted
        tensor and their results are multiplied; the three branch outputs are
        then averaged. See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.gate_activation_type = gate_activation_type
        self.gate_activation_args = gate_activation_args

        # Uses the shared SqueezeExcitation rather than the private
        # _SEWeights: multiplication is associative, so SE(x) * att_spatial
        # already equals x * (att_spatial * weights_se).
        self._patterns = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
        self._suffixes = ["hw", "cw", "hc"]

        self.se_layers: List[SqueezeExcitation] = []
        self.conv_layers: List[layers.Conv2D] = []
        self.bn_layers: List[layers.BatchNormalization] = []
        self.gate_activations: List[keras.layers.Layer] = []

        for suffix in self._suffixes:
            self.se_layers.append(SqueezeExcitation(
                reduction_ratio=reduction_ratio,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"se_{suffix}"
            ))
            self.conv_layers.append(layers.Conv2D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"conv_{suffix}"
            ))
            self.bn_layers.append(layers.BatchNormalization(name=f"bn_{suffix}"))
            self.gate_activations.append(resolve_activation_layer(
                self.gate_activation_type,
                name=f"gate_activation_{suffix}",
                **(self.gate_activation_args or {}),
            ))

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the per-branch SE, convolution, batch norm and gate layers.

        Sub-layers are built explicitly, in computational order, so every weight
        variable exists before Keras restores a checkpoint into it.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``None``.
        :rtype: None
        """
        if self.built:
            return

        batch = input_shape[0]
        for i, pattern in enumerate(self._patterns):
            perm_indices = [p + 1 for p in pattern]
            permuted_shape = (batch,) + tuple(input_shape[idx] for idx in perm_indices)

            self.se_layers[i].build(permuted_shape)

            d1, d2 = permuted_shape[1], permuted_shape[2]
            self.conv_layers[i].build((batch, d1, d2, 2))
            self.bn_layers[i].build((batch, d1, d2, 1))
            self.gate_activations[i].build((batch, d1, d2, 1))

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Run the SE and spatial paths in parallel, multiply, then average.

        :param inputs: 4-D input, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. Forwarded explicitly to every
            sub-layer that takes one.
        :type training: Optional[bool]

        :return: Same shape as ``inputs``, ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        """
        outputs = []
        
        for i, pattern in enumerate(self._patterns):
            # 1. Permute
            if pattern != (0, 1, 2):
                perm_order = [0] + [p + 1 for p in pattern]
                x = ops.transpose(inputs, perm_order)
            else:
                x = inputs
            
            # 2. Parallel Path 1: SE Output (X * ChannelWeights)
            x_se_scaled = self.se_layers[i](x, training=training)
            
            # 3. Parallel Path 2: Spatial Attention Map
            avg_pool = ops.mean(x, axis=-1, keepdims=True)
            max_pool = ops.max(x, axis=-1, keepdims=True)
            pooled = ops.concatenate([avg_pool, max_pool], axis=-1)
            
            att_spatial = self.conv_layers[i](pooled)
            att_spatial = self.bn_layers[i](att_spatial, training=training)
            att_spatial = self.gate_activations[i](att_spatial, training=training)
            
            # 4. Combine: SE_Output * Spatial_Map
            # equivalent to X * ChannelWeights * SpatialWeights
            branch_out = ops.multiply(x_se_scaled, att_spatial)
            
            # 5. Inverse Permute
            if pattern != (0, 1, 2):
                inv_pattern = [0, 0, 0]
                for idx, p in enumerate(pattern):
                    inv_pattern[p] = idx
                inv_order = [0] + [p + 1 for p in inv_pattern]
                branch_out = ops.transpose(branch_out, inv_order)
                
            outputs.append(branch_out)

        total = ops.add(ops.add(outputs[0], outputs[1]), outputs[2])
        return ops.divide(total, 3.0)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``input_shape`` unchanged. Attention rescales; it never reshapes.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary holding every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "gate_activation_type": self.gate_activation_type,
            "gate_activation_args": self.gate_activation_args,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.tripse_attention")
class _SEWeights(layers.Layer):
    """
    Internal helper: SE channel logits, pre-sigmoid, with no scaling applied.

    The path is global average pool to ``(B, 1, 1, C)``, then Conv1x1 reduce to
    ``C/r``, then the bottleneck activation, then Conv1x1 restore to ``C``. It
    stops there. There is no sigmoid and no multiply by the input, and that is
    exactly what separates it from a full Squeeze-and-Excitation block. TripSE4
    needs the logits so it can add them to spatial logits in logit space.

    Its bottleneck width differs from the standard SE block's:
    :class:`~dl_techniques.layers.conv_blocks.squeeze_excitation.SqueezeExcitation`
    computes ``max(1, int(round(C * reduction_ratio)))``; this class truncates
    instead, ``max(1, int(C * reduction_ratio))``. They agree except when the
    product lands between integers (measured: ``C=24, reduction_ratio=0.0625``
    gives 1 channel here against 2 there). Left unaligned because matching them
    would change the weight shape of every existing TripSE4 checkpoint.

    Not exported from ``attention/__init__.py`` or registered in
    ``attention/factory.py``, and has no consumer outside :class:`TripSE4`. It
    still carries ``@register_dl_technique(...)`` because it is a sub-layer of
    a serializable layer and must resolve when a TripSE4 ``.keras`` checkpoint
    loads.

    :param reduction_ratio: Bottleneck reduction ratio.
    :type reduction_ratio: float
    :param activation: Activation inside the bottleneck.
    :type activation: str
    :param activation_args: Optional keyword arguments forwarded to the
        bottleneck activation layer's constructor. Defaults to ``None``.
    :type activation_args: Optional[Dict[str, Any]]
    :param use_bias: Whether convolutions use bias.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: str
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[Any]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    """

    def __init__(
        self,
        reduction_ratio: float = 0.25,
        activation: str = 'relu',
        activation_args: Optional[Dict[str, Any]] = None,
        use_bias: bool = False,
        kernel_initializer: str = 'glorot_uniform',
        kernel_regularizer: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """Store the configuration and create the pooling and activation layers.

        ``conv_reduce`` and ``conv_restore`` are left as ``None`` here. Their
        filter counts depend on the build-time channel count, so they are created
        in :meth:`build`. See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.activation = deserialize_activation(activation)
        self.activation_args = activation_args
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.reduction_activation = resolve_activation_layer(
            self.activation,
            name="reduction_activation",
            **(self.activation_args or {}),
        )
        self.conv_reduce = None
        self.conv_restore = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # DECISION plan_2026-06-14_0c5d4a21/D-005: keep the `if self.built` and
        # `is None` guards -- a second build() would otherwise re-create conv_reduce/conv_restore and hit the already-built lock. See decisions.md.
        """Create the two 1x1 convolutions and build every sub-layer.

        The bottleneck width is ``max(1, int(C * reduction_ratio))``, which
        truncates. The shared ``SqueezeExcitation`` layer rounds instead; the
        class docstring records the divergence and why it is not aligned.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``None``.
        :rtype: None
        """
        if self.built:
            return

        input_channels = input_shape[-1]
        bottleneck_channels = max(1, int(input_channels * self.reduction_ratio))

        if self.conv_reduce is None:
            self.conv_reduce = layers.Conv2D(
                filters=bottleneck_channels,
                kernel_size=1,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )
        if self.conv_restore is None:
            self.conv_restore = layers.Conv2D(
                filters=input_channels,
                kernel_size=1,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )

        # Build explicitly
        self.global_pool.build(input_shape)
        # GAP out: (B, 1, 1, C)
        pooled_shape = (input_shape[0], 1, 1, input_channels)
        self.conv_reduce.build(pooled_shape)
        reduced_shape = (input_shape[0], 1, 1, bottleneck_channels)
        self.conv_restore.build(reduced_shape)
        
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        # Squeeze
        """Return per-channel SE logits, without a sigmoid and without scaling.

        :param inputs: 4-D input, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer that
            takes one. The D-015 anchor below explains why the bottleneck
            activation's forward is not redundant.
        :type training: Optional[bool]

        :return: Channel logits of shape ``(B, 1, 1, C)``, pre-sigmoid.
        :rtype: keras.KerasTensor
        """
        x = self.global_pool(inputs)
        # Excitation (MLP)
        x = self.conv_reduce(x, training=training)
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-015 (second site): forward
        # training= to self.reduction_activation explicitly -- measured False under training=True otherwise. See decisions.md.
        x = self.reduction_activation(x, training=training)
        logits = self.conv_restore(x, training=training)
        # Return logits (pre-sigmoid) for addition in TripSE4
        return logits

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the channel-logit shape ``(B, 1, 1, C)``.

        The spatial extent is reduced away here: this sub-layer emits one
        logit per channel, not a feature map.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``(input_shape[0], 1, 1, input_shape[-1])``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], 1, 1, input_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary holding every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "activation": serialize_activation(self.activation),
            "activation_args": self.activation_args,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.tripse_attention")
class TripSE4(layers.Layer):
    """
    TripSE4: Hybrid 3D Attention with Affine Fusion.

    Constructs a true 3D attention tensor per branch by fusing spatial
    logits ``(B, D1, D2, 1)`` and channel logits ``(B, 1, 1, D3)`` via
    broadcasting addition in the logit domain, then applying sigmoid. The
    three branch outputs are summed and refined by a final
    Squeeze-and-Excitation block.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────────────────────────────┐
        │  TripSE4 — logit-space fusion, sum, then a final SE       │
        └─────────────────────────────────────────────────────────┘

        Input  [B, H, W, C]
                  │
            ┌─────┴─────┬───────────┐
            ▼           ▼           ▼
        H-W plane   C-W plane   H-C plane
            ▼           ▼           ▼
        ┌─────────────────────────────────────────────────┐
        │ each branch adds in logit space:                │
        │   permute  ►  x  [B, D1, D2, D3]                │
        │        ├───────────────────┐                    │
        │        ▼                   ▼                    │
        │   Z-pool(x) ► conv_*   se_logits_*(x)           │
        │   ► bn_*               a _SEWeights layer       │
        │   no gate here          no sigmoid               │
        │   logits_spatial       logits_channel           │
        │   [B, D1, D2, 1]       [B, 1, 1, D3]            │
        │        │                   │                    │
        │        └─────────┬─────────┘                    │
        │                  ▼   broadcast add, in logits    │
        │   fused logits  [B, D1, D2, D3]                 │
        │                  ▼                              │
        │   gate_*, one activation  ►  3-D attention       │
        │                  ▼                              │
        │   x * attention_3d  ►  inverse permute          │
        └───────────────────────┬─────────────────────────┘
                                ▼
        element-wise sum of the 3 branch outputs
                  ▼
        ┌─────────────────────────────────────────────────┐
        │ final_se   SqueezeExcitation, after the fusion  │
        └───────────────────────┬─────────────────────────┘
                                ▼
        Output  [B, H, W, C]

    :param reduction_ratio: SE bottleneck reduction ratio.
    :type reduction_ratio: float
    :param kernel_size: Spatial convolution kernel size.
    :type kernel_size: int
    :param use_bias: Whether convolutions use bias.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: str
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[Any]
    :param gate_activation_type: Activation producing each branch's attention
        gate, resolved through
        :func:`~dl_techniques.layers.activations.resolve_activation_layer` and
        shared by all three branches. Defaults to ``'sigmoid'``, which is what
        bounds the gate to ``[0, 1]``.
    :type gate_activation_type: str
    :param gate_activation_args: Optional keyword arguments forwarded to each
        gate activation layer's constructor. Defaults to ``None``.
    :type gate_activation_args: Optional[Dict[str, Any]]
    :param se_reduction_activation_type: Activation inside the :class:`_SEWeights`
        bottleneck MLP of each branch. Defaults to ``'relu'``. This is a
        *pre-sigmoid* path: the SE logits are added to the spatial logits before
        the single gate activation, so this activation must not itself saturate
        the output to ``[0, 1]``.
    :type se_reduction_activation_type: str
    :param se_reduction_activation_args: Optional keyword arguments forwarded to
        the SE bottleneck activation layer's constructor. Defaults to ``None``.
    :type se_reduction_activation_args: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    .. note::
       This ``__init__`` does not validate its arguments. See the module
       docstring for why that is unchanged rather than fixed.
    """

    def __init__(
        self,
        reduction_ratio: float = 0.0625,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        gate_activation_type: str = "sigmoid",
        gate_activation_args: Optional[Dict[str, Any]] = None,
        se_reduction_activation_type: str = "relu",
        se_reduction_activation_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Store the configuration and create every sub-layer.

        SE placement for this variant: a _SEWeights logit path per branch,
        added to the spatial logits, plus one SqueezeExcitation after the
        branches are summed. This is the only variant whose gate activation
        sees a full 3-D tensor. See the class docstring for the parameter
        reference.
        """
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.gate_activation_type = gate_activation_type
        self.gate_activation_args = gate_activation_args
        self.se_reduction_activation_type = se_reduction_activation_type
        self.se_reduction_activation_args = se_reduction_activation_args

        self._patterns = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
        self._suffixes = ["hw", "cw", "hc"]

        # Components
        self.se_logit_layers: List[_SEWeights] = []
        self.conv_layers: List[layers.Conv2D] = []
        self.bn_layers: List[layers.BatchNormalization] = []
        self.gate_activations: List[keras.layers.Layer] = []

        for suffix in self._suffixes:
            # Internal helper to get MLP logits
            self.se_logit_layers.append(_SEWeights(
                reduction_ratio=reduction_ratio,
                activation=self.se_reduction_activation_type,
                activation_args=self.se_reduction_activation_args,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"se_logits_{suffix}"
            ))
            self.conv_layers.append(layers.Conv2D(
                filters=1,
                kernel_size=kernel_size,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"conv_{suffix}"
            ))
            self.bn_layers.append(layers.BatchNormalization(name=f"bn_{suffix}"))
            self.gate_activations.append(resolve_activation_layer(
                self.gate_activation_type,
                name=f"gate_activation_{suffix}",
                **(self.gate_activation_args or {}),
            ))

        self.final_se = SqueezeExcitation(
            reduction_ratio=reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="final_se"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the per-branch logit paths and the final SE block.

        Sub-layers are built explicitly, in computational order, so every weight
        variable exists before Keras restores a checkpoint into it.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``None``.
        :rtype: None
        """
        if self.built:
            return

        batch = input_shape[0]

        for i, pattern in enumerate(self._patterns):
            perm_indices = [p + 1 for p in pattern]
            permuted_shape = (batch,) + tuple(input_shape[idx] for idx in perm_indices)

            self.se_logit_layers[i].build(permuted_shape)

            d1, d2 = permuted_shape[1], permuted_shape[2]
            self.conv_layers[i].build((batch, d1, d2, 2))
            self.bn_layers[i].build((batch, d1, d2, 1))
            # Gate activation operates on the fused 3D logits (B, D1, D2, D3)
            self.gate_activations[i].build(permuted_shape)

        self.final_se.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Add spatial and channel logits, gate once, SUM, then run the final SE.

        :param inputs: 4-D input, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag. Forwarded explicitly to every
            sub-layer that takes one.
        :type training: Optional[bool]

        :return: Same shape as ``inputs``, ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        """
        outputs = []
        
        for i, pattern in enumerate(self._patterns):
            # 1. Permute
            if pattern != (0, 1, 2):
                perm_order = [0] + [p + 1 for p in pattern]
                x = ops.transpose(inputs, perm_order)
            else:
                x = inputs
                
            # 2. Path A: Spatial Logits
            avg_pool = ops.mean(x, axis=-1, keepdims=True)
            max_pool = ops.max(x, axis=-1, keepdims=True)
            pooled = ops.concatenate([avg_pool, max_pool], axis=-1)
            
            logits_spatial = self.conv_layers[i](pooled)
            logits_spatial = self.bn_layers[i](logits_spatial, training=training)
            # Shape: (B, D1, D2, 1)
            
            # 3. Path B: Channel Logits
            logits_channel = self.se_logit_layers[i](x, training=training)
            # Shape: (B, 1, 1, D3)
            
            # 4. Fusion: Broadcast Add
            # (B, D1, D2, 1) + (B, 1, 1, D3) -> (B, D1, D2, D3)
            # This creates a 3D attention tensor
            fused_logits = ops.add(logits_spatial, logits_channel)
            attention_3d = self.gate_activations[i](fused_logits, training=training)
            
            # 5. Apply
            scaled = ops.multiply(x, attention_3d)
            
            # 6. Inverse Permute
            if pattern != (0, 1, 2):
                inv_pattern = [0, 0, 0]
                for idx, p in enumerate(pattern):
                    inv_pattern[p] = idx
                inv_order = [0] + [p + 1 for p in inv_pattern]
                scaled = ops.transpose(scaled, inv_order)
                
            outputs.append(scaled)
            
        # Sum branches
        combined = ops.add(ops.add(outputs[0], outputs[1]), outputs[2])
        
        # Final SE
        output = self.final_se(combined, training=training)
        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: 4-D input shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``input_shape`` unchanged. Attention rescales; it never reshapes.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary holding every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "gate_activation_type": self.gate_activation_type,
            "gate_activation_args": self.gate_activation_args,
            "se_reduction_activation_type": self.se_reduction_activation_type,
            "se_reduction_activation_args": self.se_reduction_activation_args,
        })
        return config

# ---------------------------------------------------------------------

