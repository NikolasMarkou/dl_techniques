"""
TripSE: Triplet Squeeze and Excitation Attention Block.

Implementation of "Achieving 3D Attention via Triplet Squeeze and Excitation Block"
(Alhazmi & Altahhan, 2025, arXiv:2505.05943).

Combines Triplet Attention with Squeeze-and-Excitation to create 3D attention that captures:
- Inter-dimensional relationships (from Triplet Attention)
- Global channel importance (from SE)

Four variants are implemented:
- TripSE1: SE block after branch summation (Post-fusion SE).
- TripSE2: SE block at beginning of each branch (Pre-process SE).
- TripSE3: SE blocks embedded within branches (Parallel SE).
- TripSE4: Hybrid with affine combination of spatial and channel descriptors (3D Attention).
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
    Single branch of Triplet Attention mechanism.

    Captures cross-dimensional interaction by rotating tensor dimensions,
    applying Z-pooling (avg + max), convolution, and sigmoid activation.

    Shape Flow:
        Input (B, H, W, C) -> Permute -> (B, D1, D2, D3)
        Z-Pool (D3) -> (B, D1, D2, 2)
        Conv2D -> (B, D1, D2, 1)
        Sigmoid -> Attention Map
        Output -> Input * Attention Map (Broadcasted) -> Inverse Permute

    Parameters:
        kernel_size: Int, kernel size for the convolution. Default 7.
        permute_pattern: Tuple of 3 ints, permutation pattern.
            (0, 1, 2) means no rotation (H-W plane).
            (0, 2, 1) means C-W plane (rotate H-C).
            (2, 1, 0) means H-C plane (rotate W-C).
        use_bias: Bool, whether to use bias in convolution.
        kernel_initializer: Initializer for kernel weights.
        kernel_regularizer: Regularizer for kernel weights.
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
        """Build the branch layers."""
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

    Architecture:
    1. Parallel Branches: H-W (no rot), C-W (rot), H-C (rot).
    2. Summation: Combine outputs of all branches.
    3. SE Block: Apply channel-wise recalibration to the summed features.

    Parameters:
        reduction_ratio: Float, SE reduction ratio.
        kernel_size: Int, convolution kernel size.
        use_bias: Bool, use bias in convolution.
        kernel_initializer: Weights initializer.
        kernel_regularizer: Weights regularizer.
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

    Architecture:
    1. Permute Input for each branch.
    2. Apply SE Block to the permuted input (treating permuted 'channels' as features).
    3. Apply Triplet Attention (Z-Pool -> Conv -> BN -> Sigmoid) to SE output.
    4. Inverse Permute and Average branches.
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

    Architecture:
    1. For each branch, process Input via Triplet Attention (Spatial).
    2. In parallel, process Input via SE Block (Channel).
    3. Multiply Spatial Attention Map by Channel Attention Weights.
    4. Scale Input by combined attention.
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
    Internal helper layer to compute SE logits/weights.
    
    Duplicates logic of SqueezeExcitation but returns the transformed weights 
    (before or after sigmoid) instead of scaling the input.
    Used for TripSE4 where we need to add SE logits to spatial logits.
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

    Architecture:
    1. Per Branch:
       - Path A: Z-Pool -> Conv -> BN (Spatial Logits)
       - Path B: GAP -> MLP (Channel Logits)
       - Fusion: Spatial Logits + Channel Logits (Broadcasting addition)
       - Activation: Sigmoid(Fused Logits) -> 3D Attention Tensor
    2. Fuse branches (Sum).
    3. Final SE Block on output.
    
    This variant constructs a true 3D attention map by fusing spatial and channel
    contexts in the logit domain before activation.
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

