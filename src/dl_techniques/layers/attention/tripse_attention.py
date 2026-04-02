"""
TripSE: Triplet Squeeze and Excitation Attention Block.

Implementation of "Achieving 3D Attention via Triplet Squeeze and Excitation Block"
(Alhazmi and Altahhan, 2025, arXiv:2505.05943).

Combines Triplet Attention with Squeeze-and-Excitation to create 3D attention
that captures inter-dimensional relationships (from Triplet Attention) and
global channel importance (from SE).

Four variants are provided:

- **TripSE1**: SE block after branch summation (Post-fusion SE).
- **TripSE2**: SE block at beginning of each branch (Pre-process SE).
- **TripSE3**: SE blocks embedded within branches (Parallel SE).
- **TripSE4**: Hybrid with affine combination of spatial and channel
  descriptors (3D Attention).
"""

import keras
from keras import ops, layers, initializers, regularizers
from typing import Optional, Tuple, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..squeeze_excitation import SqueezeExcitation

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TripletAttentionBranch(layers.Layer):
    """
    Single branch of the Triplet Attention mechanism.

    Captures cross-dimensional interaction by rotating tensor dimensions,
    applying Z-pooling (concatenation of channel-wise average and max),
    convolution, batch normalization, and sigmoid activation. The resulting
    spatial attention map is broadcast-multiplied onto the permuted input,
    then the inverse permutation restores the original axis order.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │  Input (B, H, W, C)          │
        └─────────────┬────────────────┘
                      ▼
        ┌──────────────────────────────┐
        │  Permute axes                │
        │  (B, D1, D2, D3)             │
        └─────────────┬────────────────┘
                      ▼
        ┌──────────────────────────────┐
        │  Z-Pool: mean + max on D3    │
        │  ─► (B, D1, D2, 2)           │
        └─────────────┬────────────────┘
                      ▼
        ┌──────────────────────────────┐
        │  Conv2D ─► BN ─► Sigmoid     │
        │  ─► (B, D1, D2, 1)           │
        └─────────────┬────────────────┘
                      ▼
        ┌──────────────────────────────┐
        │  x_permuted * attention_map  │
        └─────────────┬────────────────┘
                      ▼
        ┌──────────────────────────────┐
        │  Inverse Permute             │
        │  Output (B, H, W, C)         │
        └──────────────────────────────┘

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
    """

    def __init__(
        self,
        kernel_size: int = 7,
        permute_pattern: Tuple[int, int, int] = (0, 1, 2),
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.permute_pattern = permute_pattern
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

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
        self.sigmoid = layers.Activation("sigmoid", name="sigmoid")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build sub-layers with shapes derived from the permutation pattern.

        :param input_shape: 4-D shape ``(B, H, W, C)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Determine shape after permutation
        # input_shape is (B, H, W, C)
        # We need to simulate the permutation to know the spatial dims for logging/debug
        # ensuring the conv builds correctly on the pooled channel dim (which is always 2).
        
        # The input to Conv2D is always (B, D1, D2, 2) regardless of permutation
        # because Z-pooling (Mean+Max) reduces the last dim to 2.
        # However, we must ensure explicit build for serialization safety.
        
        if len(input_shape) != 4:
            raise ValueError(f"Input must be 4D, got {input_shape}")

        batch = input_shape[0]
        # Map input dimensions based on pattern to simulate the 'internal' shape
        # Pattern (0,1,2) -> (H, W, C)
        # Pattern (0,2,1) -> (C, W, H) (Example logic, actual depends on implementation)
        # We just need to know the Conv input has 2 channels.
        
        permuted_dims = [input_shape[i+1] for i in self.permute_pattern]
        
        # Conv input: (Batch, D1, D2, 2)
        conv_input_shape = (batch, permuted_dims[0], permuted_dims[1], 2)
        self.conv.build(conv_input_shape)
        
        # BN input: (Batch, D1, D2, 1)
        conv_output_shape = (batch, permuted_dims[0], permuted_dims[1], 1)
        self.batch_norm.build(conv_output_shape)
        
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        # 1. Permute
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
        attention = self.sigmoid(attention)

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
        return input_shape

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "permute_pattern": self.permute_pattern,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TripSE1(layers.Layer):
    """
    TripSE1: Triplet Attention with Post-Fusion Squeeze-and-Excitation.

    Three parallel Triplet Attention branches (H-W, C-W, H-C planes)
    produce spatial attention maps. Their outputs are summed, and a
    Squeeze-and-Excitation block performs channel-wise recalibration on
    the fused result.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────┐
        │  Input (B, H, W, C)               │
        └──────┬─────────┬─────────┬────────┘
               ▼         ▼         ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Branch   │ │ Branch   │ │ Branch   │
        │ H-W      │ │ C-W      │ │ H-C      │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └─────┬──────┴──────┬─────┘
                   ▼             │
        ┌──────────────────────────────────┐
        │  Element-wise Sum                │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Squeeze-and-Excitation          │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Output (B, H, W, C)            │
        └──────────────────────────────────┘

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
    """

    def __init__(
        self,
        reduction_ratio: float = 0.0625,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Triplet Attention Branches
        self.branch_hw = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(0, 1, 2),  # H-W
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="branch_hw"
        )
        self.branch_cw = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(0, 2, 1),  # C-W
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="branch_cw"
        )
        self.branch_hc = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(2, 1, 0),  # H-C
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="branch_hc"
        )
        
        # SE Block (created here, built in build)
        self.se_block = SqueezeExcitation(
            reduction_ratio=reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.branch_hw.build(input_shape)
        self.branch_cw.build(input_shape)
        self.branch_hc.build(input_shape)
        self.se_block.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        out_hw = self.branch_hw(inputs, training=training)
        out_cw = self.branch_cw(inputs, training=training)
        out_hc = self.branch_hc(inputs, training=training)

        combined = ops.add(ops.add(out_hw, out_cw), out_hc)
        # Average before SE? Original paper usually implies Sum or Avg.
        # We'll use sum as per the prompt's implied logic, SE handles magnitude.
        
        output = self.se_block(combined, training=training)
        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TripSE2(layers.Layer):
    """
    TripSE2: Pre-Process Squeeze-and-Excitation.

    Each branch first permutes the input tensor, applies a
    Squeeze-and-Excitation block on the permuted channels, then runs the
    Triplet Attention core (Z-Pool, Conv, BN, Sigmoid) on the SE-refined
    features. Outputs are inverse-permuted and averaged.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────┐
        │  Input (B, H, W, C)               │
        └──────┬─────────┬─────────┬────────┘
               ▼         ▼         ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Permute  │ │ Permute  │ │ Permute  │
        │ + SE     │ │ + SE     │ │ + SE     │
        │ + ZPool  │ │ + ZPool  │ │ + ZPool  │
        │ + Conv   │ │ + Conv   │ │ + Conv   │
        │ + InvPrm │ │ + InvPrm │ │ + InvPrm │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └─────┬──────┴──────┬─────┘
                   ▼
        ┌──────────────────────────────────┐
        │  Average of 3 branches           │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Output (B, H, W, C)            │
        └──────────────────────────────────┘

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
    """

    def __init__(
        self,
        reduction_ratio: float = 0.0625,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # We need distinct SE and Conv blocks for each branch because shapes differ
        self._patterns = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
        self._suffixes = ["hw", "cw", "hc"]
        
        # Containers for sub-layers
        self.se_layers: List[SqueezeExcitation] = []
        self.conv_layers: List[layers.Conv2D] = []
        self.bn_layers: List[layers.BatchNormalization] = []
        
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
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

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
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
            att = ops.sigmoid(att)
            
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

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TripSE3(layers.Layer):
    """
    TripSE3: Parallel Squeeze-and-Excitation.

    Each branch runs two parallel paths on the permuted input: a spatial
    attention path (Z-Pool, Conv, BN, Sigmoid) and a channel attention
    path (SE block). The SE-scaled features are element-wise multiplied
    by the spatial attention map, producing a joint spatial-channel
    attention. Results are inverse-permuted and averaged.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────┐
        │  Input (B, H, W, C)               │
        └──────┬─────────┬─────────┬────────┘
               ▼         ▼         ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Permute  │ │ Permute  │ │ Permute  │
        │ ┌──┬──┐  │ │ ┌──┬──┐  │ │ ┌──┬──┐  │
        │ │SE│Sp│  │ │ │SE│Sp│  │ │ │SE│Sp│  │
        │ └──┴──┘  │ │ └──┴──┘  │ │ └──┴──┘  │
        │ SE*Sp    │ │ SE*Sp    │ │ SE*Sp    │
        │ InvPerm  │ │ InvPerm  │ │ InvPerm  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └─────┬──────┴──────┬─────┘
                   ▼
        ┌──────────────────────────────────┐
        │  Average of 3 branches           │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Output (B, H, W, C)            │
        └──────────────────────────────────┘

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
    """

    def __init__(
        self,
        reduction_ratio: float = 0.0625,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Initialize helper class to get weights instead of scaled input
        # Note: Since the provided SqueezeExcitation returns x*s, and we want s,
        # we will use SqueezeExcitation but need to extract s. 
        # Standard SqueezeExcitation does not expose 's'.
        # However, TripSE3 logic in source was: Output = X * (Att_spatial * Weights_SE).
        # This is equivalent to Output = (X * Weights_SE) * Att_spatial = SE(X) * Att_spatial.
        # So we can use the standard SE block output!
        
        self._patterns = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
        self._suffixes = ["hw", "cw", "hc"]
        
        self.se_layers: List[SqueezeExcitation] = []
        self.conv_layers: List[layers.Conv2D] = []
        self.bn_layers: List[layers.BatchNormalization] = []
        
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        batch = input_shape[0]
        for i, pattern in enumerate(self._patterns):
            perm_indices = [p + 1 for p in pattern]
            permuted_shape = (batch,) + tuple(input_shape[idx] for idx in perm_indices)
            
            self.se_layers[i].build(permuted_shape)
            
            d1, d2 = permuted_shape[1], permuted_shape[2]
            self.conv_layers[i].build((batch, d1, d2, 2))
            self.bn_layers[i].build((batch, d1, d2, 1))
            
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
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
            att_spatial = ops.sigmoid(att_spatial)
            
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

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class _SEWeights(layers.Layer):
    """
    Internal helper that computes SE channel logits without scaling the input.

    Mirrors the Squeeze-and-Excitation path (GAP, bottleneck MLP) but
    returns the pre-sigmoid logits of shape ``(B, 1, 1, C)`` instead of
    the full ``x * sigmoid(logits)`` product. This allows TripSE4 to fuse
    spatial and channel logits in the logit domain before activation.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────┐
        │  Input (B, H, W, C)│
        └────────┬───────────┘
                 ▼
        ┌────────────────────┐
        │  GAP ─► (B,1,1,C) │
        └────────┬───────────┘
                 ▼
        ┌────────────────────┐
        │  Conv1x1 reduce    │
        │  ─► activation     │
        │  ─► Conv1x1 restore│
        └────────┬───────────┘
                 ▼
        ┌────────────────────┐
        │  Logits (B,1,1,C)  │
        └────────────────────┘

    :param reduction_ratio: Bottleneck reduction ratio.
    :type reduction_ratio: float
    :param activation: Activation inside the bottleneck.
    :type activation: str
    :param use_bias: Whether convolutions use bias.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: str
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[Any]
    """
    def __init__(
        self,
        reduction_ratio: float = 0.25,
        activation: str = 'relu',
        use_bias: bool = False,
        kernel_initializer: str = 'glorot_uniform',
        kernel_regularizer: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.reduction_activation = layers.Activation(activation)
        self.conv_reduce = None
        self.conv_restore = None

    def build(self, input_shape):
        input_channels = input_shape[-1]
        bottleneck_channels = max(1, int(input_channels * self.reduction_ratio))
        
        self.conv_reduce = layers.Conv2D(
            filters=bottleneck_channels,
            kernel_size=1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer
        )
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

    def call(self, inputs, training=None):
        # Squeeze
        x = self.global_pool(inputs)
        # Excitation (MLP)
        x = self.conv_reduce(x, training=training)
        x = self.reduction_activation(x)
        logits = self.conv_restore(x, training=training)
        # Return logits (pre-sigmoid) for addition in TripSE4
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class TripSE4(layers.Layer):
    """
    TripSE4: Hybrid 3D Attention with Affine Fusion.

    Constructs a true 3D attention tensor per branch by fusing spatial
    logits ``(B, D1, D2, 1)`` and channel logits ``(B, 1, 1, D3)`` via
    broadcasting addition in the logit domain, then applying sigmoid. The
    three branch outputs are summed and refined by a final
    Squeeze-and-Excitation block.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input (B, H, W, C)                 │
        └──────┬─────────┬─────────┬──────────┘
               ▼         ▼         ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Permute  │ │ Permute  │ │ Permute  │
        │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │
        │ │Spat. │ │ │ │Spat. │ │ │ │Spat. │ │
        │ │logits│ │ │ │logits│ │ │ │logits│ │
        │ └──┬───┘ │ │ └──┬───┘ │ │ └──┬───┘ │
        │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │
        │ │Chan. │ │ │ │Chan. │ │ │ │Chan. │ │
        │ │logits│ │ │ │logits│ │ │ │logits│ │
        │ └──┬───┘ │ │ └──┬───┘ │ │ └──┬───┘ │
        │  Add+Sig │ │  Add+Sig │ │  Add+Sig │
        │  *input  │ │  *input  │ │  *input  │
        │  InvPerm │ │  InvPerm │ │  InvPerm │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └─────┬──────┴──────┬─────┘
                   ▼
        ┌──────────────────────────────────┐
        │  Sum + Final SE Block            │
        └────────────┬─────────────────────┘
                     ▼
        ┌──────────────────────────────────┐
        │  Output (B, H, W, C)            │
        └──────────────────────────────────┘

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
    """

    def __init__(
        self,
        reduction_ratio: float = 0.0625,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self._patterns = [(0, 1, 2), (0, 2, 1), (2, 1, 0)]
        self._suffixes = ["hw", "cw", "hc"]
        
        # Components
        self.se_logit_layers: List[_SEWeights] = []
        self.conv_layers: List[layers.Conv2D] = []
        self.bn_layers: List[layers.BatchNormalization] = []
        
        for suffix in self._suffixes:
            # Internal helper to get MLP logits
            self.se_logit_layers.append(_SEWeights(
                reduction_ratio=reduction_ratio,
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
            
        self.final_se = SqueezeExcitation(
            reduction_ratio=reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="final_se"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        batch = input_shape[0]
        
        for i, pattern in enumerate(self._patterns):
            perm_indices = [p + 1 for p in pattern]
            permuted_shape = (batch,) + tuple(input_shape[idx] for idx in perm_indices)
            
            self.se_logit_layers[i].build(permuted_shape)
            
            d1, d2 = permuted_shape[1], permuted_shape[2]
            self.conv_layers[i].build((batch, d1, d2, 2))
            self.bn_layers[i].build((batch, d1, d2, 1))
        
        self.final_se.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
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
            attention_3d = ops.sigmoid(fused_logits)
            
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

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

