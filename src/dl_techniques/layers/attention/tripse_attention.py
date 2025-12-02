"""
TripSE: Triplet Squeeze and Excitation Attention Block.

Implementation of "Achieving 3D Attention via Triplet Squeeze and Excitation Block"
(Alhazmi & Altahhan, 2025, arXiv:2505.05943).

Combines Triplet Attention with Squeeze-and-Excitation to create 3D attention that captures:
- Inter-dimensional relationships (from Triplet Attention)
- Global channel importance (from SE)

Four variants are implemented:
- TripSE1: SE block after branch summation
- TripSE2: SE block at beginning of each branch
- TripSE3: SE blocks embedded within branches (parallel)
- TripSE4: Hybrid with SE at multiple positions and affine transformation
"""

import keras
from typing import Optional, Tuple, Literal

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable(package="dl_techniques")
class SqueezeExcitation(keras.layers.Layer):
    """
    Squeeze-and-Excitation block.

    Applies global average pooling followed by two fully connected layers
    to recalibrate channel-wise features.

    ASCII Diagram::

        Input (B, H, W, C)
                |
        Global Avg Pool
                |
            (B, 1, 1, C)
                |
          Dense (C/r)
                |
            ReLU
                |
          Dense (C)
                |
            Sigmoid
                |
            (B, 1, 1, C)
                |
                × (element-wise multiply)
                |
        Output (B, H, W, C)

    Parameters
    ----------
    channels : int
        Number of channels to recalibrate.
    reduction_ratio : int, default=16
        Channel reduction ratio for bottleneck.
    activation : str, default="relu"
        Activation function for the bottleneck layer.
    use_bias : bool, default=True
        Whether to use bias in dense layers.
    kernel_initializer : str, default="glorot_uniform"
        Initializer for kernel weights.
    bias_initializer : str, default="zeros"
        Initializer for bias weights.
    kernel_regularizer : Optional, default=None
        Regularizer for kernel weights.
    bias_regularizer : Optional, default=None
        Regularizer for bias weights.

    References
    ----------
    Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        activation: str = "relu",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # Sub-layers created in __init__
        reduced_channels = max(channels // reduction_ratio, 1)

        self.global_avg_pool = keras.layers.GlobalAveragePooling2D(
            keepdims=True,
            name="gap"
        )

        self.dense_1 = keras.layers.Dense(
            reduced_channels,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="fc1"
        )

        self.dense_2 = keras.layers.Dense(
            channels,
            activation="sigmoid",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="fc2"
        )

        logger.debug(
            f"Initialized SqueezeExcitation: channels={channels}, "
            f"reduction_ratio={reduction_ratio}, reduced_channels={reduced_channels}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Parameters
        ----------
        input_shape : Tuple
            Shape of input tensor (batch, height, width, channels).
        """
        # Build sub-layers explicitly
        batch, height, width, channels = input_shape

        # GAP output shape
        gap_output_shape = (batch, 1, 1, channels)
        self.global_avg_pool.build(input_shape)

        # Dense 1 processes flattened GAP output
        self.dense_1.build((batch, 1, 1, channels))

        # Dense 2 output shape
        reduced_channels = max(self.channels // self.reduction_ratio, 1)
        self.dense_2.build((batch, 1, 1, reduced_channels))

        super().build(input_shape)
        logger.debug(f"Built SqueezeExcitation with input_shape={input_shape}")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass of SE block.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch, height, width, channels).
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        keras.KerasTensor
            Recalibrated tensor of same shape as input.
        """
        # Squeeze: Global average pooling
        squeeze = self.global_avg_pool(inputs)

        # Excitation: FC -> activation -> FC -> sigmoid
        excitation = self.dense_1(squeeze)
        excitation = self.dense_2(excitation)

        # Scale input by channel attention weights
        return keras.ops.multiply(inputs, excitation)

    def get_config(self) -> dict:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": keras.saving.serialize_keras_object(self.kernel_regularizer),
            "bias_regularizer": keras.saving.serialize_keras_object(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "SqueezeExcitation":
        """Create layer from config."""
        kernel_regularizer = config.pop("kernel_regularizer", None)
        bias_regularizer = config.pop("bias_regularizer", None)

        if kernel_regularizer is not None:
            config["kernel_regularizer"] = keras.saving.deserialize_keras_object(kernel_regularizer)
        if bias_regularizer is not None:
            config["bias_regularizer"] = keras.saving.deserialize_keras_object(bias_regularizer)

        return cls(**config)


@keras.saving.register_keras_serializable(package="dl_techniques")
class TripletAttentionBranch(keras.layers.Layer):
    """
    Single branch of Triplet Attention mechanism.

    Captures cross-dimensional interaction by rotating tensor dimensions,
    applying Z-pooling (avg + max), convolution, and sigmoid activation.

    ASCII Diagram::

        Input (B, H, W, C)
                |
            Permute (rotation depends on branch)
                |
        (B, D1, D2, D3)  [D1, D2, D3 are permuted dimensions]
                |
            Z-Pooling
        +-------+-------+
        |               |
    Avg Pool        Max Pool
    (along D3)      (along D3)
        |               |
        +-------+-------+
                |
        Concatenate → (B, D1, D2, 2)
                |
            Conv 7x7
                |
        (B, D1, D2, 1)
                |
            BatchNorm
                |
            Sigmoid
                |
        Attention Map (B, D1, D2, 1)
                |
                × (multiply with permuted input)
                |
        Inverse Permute (rotate back)
                |
        Output (B, H, W, C)

    Parameters
    ----------
    kernel_size : int, default=7
        Convolutional kernel size for attention computation.
    permute_pattern : Tuple[int, int, int]
        Permutation pattern for dimension rotation.
        (0,1,2) = no rotation (H-W branch)
        (0,2,1) = rotate C-W (height<->channel)
        (2,1,0) = rotate H-C (width<->channel)
    use_bias : bool, default=False
        Whether to use bias in conv layer.
    kernel_initializer : str, default="glorot_uniform"
        Initializer for kernel weights.
    kernel_regularizer : Optional, default=None
        Regularizer for kernel weights.

    References
    ----------
    Misra et al., "Rotate to Attend: Convolutional Triplet Attention Module," WACV 2021.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        permute_pattern: Tuple[int, int, int] = (0, 1, 2),
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.permute_pattern = permute_pattern
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Sub-layers
        self.conv = keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="conv"
        )

        self.batch_norm = keras.layers.BatchNormalization(name="bn")
        self.sigmoid = keras.layers.Activation("sigmoid", name="sigmoid")

        logger.debug(
            f"Initialized TripletAttentionBranch: kernel_size={kernel_size}, "
            f"permute_pattern={permute_pattern}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Parameters
        ----------
        input_shape : Tuple
            Shape of input tensor (batch, height, width, channels).
        """
        batch, height, width, channels = input_shape

        # After permutation and Z-pooling, we have 2 channels
        # Shape depends on permutation pattern
        if self.permute_pattern == (0, 1, 2):  # H-W branch (no rotation)
            conv_input_shape = (batch, height, width, 2)
        elif self.permute_pattern == (0, 2, 1):  # C-W branch
            conv_input_shape = (batch, channels, width, 2)
        else:  # H-C branch
            conv_input_shape = (batch, height, channels, 2)

        self.conv.build(conv_input_shape)
        self.batch_norm.build((batch, conv_input_shape[1], conv_input_shape[2], 1))

        super().build(input_shape)
        logger.debug(f"Built TripletAttentionBranch with input_shape={input_shape}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of triplet attention branch.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch, height, width, channels).
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        keras.KerasTensor
            Attention-modulated tensor of same shape as input.
        """
        # Step 1: Permute dimensions (rotate tensor)
        if self.permute_pattern != (0, 1, 2):
            x = keras.ops.transpose(inputs, [0] + list(self.permute_pattern))
        else:
            x = inputs

        # Step 2: Z-pooling (concatenate avg and max pooling along last dimension)
        avg_pool = keras.ops.mean(x, axis=-1, keepdims=True)
        max_pool = keras.ops.max(x, axis=-1, keepdims=True)
        pooled = keras.ops.concatenate([avg_pool, max_pool], axis=-1)

        # Step 3: Convolution + BatchNorm + Sigmoid
        attention = self.conv(pooled)
        attention = self.batch_norm(attention, training=training)
        attention = self.sigmoid(attention)

        # Step 4: Scale the permuted input
        scaled = keras.ops.multiply(x, attention)

        # Step 5: Rotate back to original dimensions
        if self.permute_pattern != (0, 1, 2):
            # Inverse permutation
            inv_pattern = [self.permute_pattern.index(i) for i in range(3)]
            scaled = keras.ops.transpose(scaled, [0] + inv_pattern)

        return scaled

    def get_config(self) -> dict:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "permute_pattern": self.permute_pattern,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.saving.serialize_keras_object(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TripletAttentionBranch":
        """Create layer from config."""
        kernel_regularizer = config.pop("kernel_regularizer", None)
        if kernel_regularizer is not None:
            config["kernel_regularizer"] = keras.saving.deserialize_keras_object(kernel_regularizer)
        return cls(**config)


@keras.saving.register_keras_serializable(package="dl_techniques")
class TripSE1(keras.layers.Layer):
    """
    TripSE1: Triplet Attention with SE block after branch summation.

    Architecture:
    1. Three parallel Triplet Attention branches (H-W, C-W, H-C)
    2. Sum branch outputs
    3. Apply single SE block to combined output

    This variant applies global channel weighting after cross-dimensional interaction.

    ASCII Diagram::

                                Input (B, H, W, C)
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
                Branch H-W          Branch C-W          Branch H-C
                (no rotation)       (rotate C-W)        (rotate H-C)
                    |                   |                   |
                Z-Pool              Z-Pool              Z-Pool
            (avg + max pool)    (avg + max pool)    (avg + max pool)
                    |                   |                   |
                Conv 7x7            Conv 7x7            Conv 7x7
                    |                   |                   |
                BatchNorm           BatchNorm           BatchNorm
                    |                   |                   |
                Sigmoid             Sigmoid             Sigmoid
                    |                   |                   |
                Scale Input         Scale Input         Scale Input
                    |                   |                   |
                Rotate Back         Rotate Back         Rotate Back
                    |                   |                   |
                    +-------------------+-------------------+
                                        |
                                      Sum
                                        |
                                    SE Block
                                (GAP → FC → FC)
                                        |
                                Output (B, H, W, C)

    Parameters
    ----------
    reduction_ratio : int, default=16
        Channel reduction ratio for SE block.
    kernel_size : int, default=7
        Convolutional kernel size for triplet attention.
    use_bias : bool, default=False
        Whether to use bias in conv layers.
    kernel_initializer : str, default="glorot_uniform"
        Initializer for kernel weights.
    kernel_regularizer : Optional, default=None
        Regularizer for kernel weights.

    References
    ----------
    Alhazmi & Altahhan, "Achieving 3D Attention via Triplet Squeeze and Excitation Block," 2025.
    """

    def __init__(
        self,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Three triplet attention branches
        self.branch_hw = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(0, 1, 2),  # H-W (no rotation)
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="branch_hw"
        )

        self.branch_cw = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(0, 2, 1),  # C-W (rotate height<->channel)
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="branch_cw"
        )

        self.branch_hc = TripletAttentionBranch(
            kernel_size=kernel_size,
            permute_pattern=(2, 1, 0),  # H-C (rotate width<->channel)
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="branch_hc"
        )

        # SE block will be created in build() when we know channels
        self.se_block = None

        logger.debug(
            f"Initialized TripSE1: reduction_ratio={reduction_ratio}, "
            f"kernel_size={kernel_size}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Parameters
        ----------
        input_shape : Tuple
            Shape of input tensor (batch, height, width, channels).
        """
        batch, height, width, channels = input_shape

        # Build triplet attention branches
        self.branch_hw.build(input_shape)
        self.branch_cw.build(input_shape)
        self.branch_hc.build(input_shape)

        # Create and build SE block
        self.se_block = SqueezeExcitation(
            channels=channels,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se"
        )
        self.se_block.build(input_shape)

        super().build(input_shape)
        logger.debug(f"Built TripSE1 with input_shape={input_shape}, channels={channels}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of TripSE1.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch, height, width, channels).
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        keras.KerasTensor
            Attention-modulated tensor of same shape as input.
        """
        # Three parallel branches
        out_hw = self.branch_hw(inputs, training=training)
        out_cw = self.branch_cw(inputs, training=training)
        out_hc = self.branch_hc(inputs, training=training)

        # Sum branch outputs
        combined = keras.ops.add(keras.ops.add(out_hw, out_cw), out_hc)

        # Apply SE block to combined output
        output = self.se_block(combined, training=training)

        return output

    def get_config(self) -> dict:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.saving.serialize_keras_object(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TripSE1":
        """Create layer from config."""
        kernel_regularizer = config.pop("kernel_regularizer", None)
        if kernel_regularizer is not None:
            config["kernel_regularizer"] = keras.saving.deserialize_keras_object(kernel_regularizer)
        return cls(**config)


@keras.saving.register_keras_serializable(package="dl_techniques")
class TripSE2(keras.layers.Layer):
    """
    TripSE2: Triplet Attention with SE block at beginning of each branch.

    Architecture:
    1. Each of three branches starts with SE block (on permuted input)
    2. SE output goes through Z-pooling, conv, BatchNorm, sigmoid
    3. Average branch outputs

    This variant applies channel weighting before cross-dimensional operations.

    ASCII Diagram::

                                Input (B, H, W, C)
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
                Branch H-W          Branch C-W          Branch H-C
                    |                   |                   |
                (no rotation)       Permute (C,W,H)     Permute (H,C,W)
                    |                   |                   |
                SE Block            SE Block            SE Block
            (GAP → FC → FC)     (GAP → FC → FC)     (GAP → FC → FC)
                    |                   |                   |
                Z-Pool              Z-Pool              Z-Pool
            (avg + max pool)    (avg + max pool)    (avg + max pool)
                    |                   |                   |
                Conv 7x7            Conv 7x7            Conv 7x7
                    |                   |                   |
                BatchNorm           BatchNorm           BatchNorm
                    |                   |                   |
                Sigmoid             Sigmoid             Sigmoid
                    |                   |                   |
                Scale Input         Scale Input         Scale Input
                    |                   |                   |
                (no rotation)       Rotate Back         Rotate Back
                    |                   |                   |
                    +-------------------+-------------------+
                                        |
                                    Average
                                        |
                                Output (B, H, W, C)

    Parameters
    ----------
    reduction_ratio : int, default=16
        Channel reduction ratio for SE blocks.
    kernel_size : int, default=7
        Convolutional kernel size for triplet attention.
    use_bias : bool, default=False
        Whether to use bias in conv layers.
    kernel_initializer : str, default="glorot_uniform"
        Initializer for kernel weights.
    kernel_regularizer : Optional, default=None
        Regularizer for kernel weights.

    References
    ----------
    Alhazmi & Altahhan, "Achieving 3D Attention via Triplet Squeeze and Excitation Block," 2025.
    """

    def __init__(
        self,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # SE blocks and triplet branches will be created in build()
        self.se_hw = None
        self.se_cw = None
        self.se_hc = None

        self.branch_hw = None
        self.branch_cw = None
        self.branch_hc = None

        logger.debug(
            f"Initialized TripSE2: reduction_ratio={reduction_ratio}, "
            f"kernel_size={kernel_size}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Parameters
        ----------
        input_shape : Tuple
            Shape of input tensor (batch, height, width, channels).
        """
        batch, height, width, channels = input_shape

        # Create SE blocks for each branch
        self.se_hw = SqueezeExcitation(
            channels=channels,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_hw"
        )
        self.se_hw.build(input_shape)

        # For C-W branch, after permutation shape is (batch, channels, width, height)
        # But SE still operates on last dimension which is now height (original channels)
        self.se_cw = SqueezeExcitation(
            channels=height,  # After permutation, height becomes channel dim
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_cw"
        )
        cw_shape = (batch, channels, width, height)
        self.se_cw.build(cw_shape)

        # For H-C branch, after permutation shape is (batch, height, channels, width)
        self.se_hc = SqueezeExcitation(
            channels=width,  # After permutation, width becomes channel dim
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_hc"
        )
        hc_shape = (batch, height, channels, width)
        self.se_hc.build(hc_shape)

        # Create triplet attention processing (without SE, since it's applied before)
        # We need custom processing per branch
        # Conv layers for each branch
        self.conv_hw = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_hw"
        )
        self.conv_hw.build((batch, height, width, 2))

        self.conv_cw = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_cw"
        )
        self.conv_cw.build((batch, channels, width, 2))

        self.conv_hc = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_hc"
        )
        self.conv_hc.build((batch, height, channels, 2))

        # BatchNorm layers
        self.bn_hw = keras.layers.BatchNormalization(name="bn_hw")
        self.bn_hw.build((batch, height, width, 1))

        self.bn_cw = keras.layers.BatchNormalization(name="bn_cw")
        self.bn_cw.build((batch, channels, width, 1))

        self.bn_hc = keras.layers.BatchNormalization(name="bn_hc")
        self.bn_hc.build((batch, height, channels, 1))

        super().build(input_shape)
        logger.debug(f"Built TripSE2 with input_shape={input_shape}, channels={channels}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of TripSE2.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch, height, width, channels).
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        keras.KerasTensor
            Attention-modulated tensor of same shape as input.
        """
        # Branch H-W (no rotation)
        x_hw = inputs
        x_hw = self.se_hw(x_hw, training=training)
        # Z-pooling
        avg_hw = keras.ops.mean(x_hw, axis=-1, keepdims=True)
        max_hw = keras.ops.max(x_hw, axis=-1, keepdims=True)
        pooled_hw = keras.ops.concatenate([avg_hw, max_hw], axis=-1)
        # Conv + BN + Sigmoid
        att_hw = self.conv_hw(pooled_hw)
        att_hw = self.bn_hw(att_hw, training=training)
        att_hw = keras.ops.sigmoid(att_hw)
        out_hw = keras.ops.multiply(x_hw, att_hw)

        # Branch C-W (permute to channels, width, height)
        x_cw = keras.ops.transpose(inputs, [0, 3, 2, 1])  # (B, C, W, H)
        x_cw = self.se_cw(x_cw, training=training)
        # Z-pooling
        avg_cw = keras.ops.mean(x_cw, axis=-1, keepdims=True)
        max_cw = keras.ops.max(x_cw, axis=-1, keepdims=True)
        pooled_cw = keras.ops.concatenate([avg_cw, max_cw], axis=-1)
        # Conv + BN + Sigmoid
        att_cw = self.conv_cw(pooled_cw)
        att_cw = self.bn_cw(att_cw, training=training)
        att_cw = keras.ops.sigmoid(att_cw)
        scaled_cw = keras.ops.multiply(x_cw, att_cw)
        # Rotate back
        out_cw = keras.ops.transpose(scaled_cw, [0, 3, 2, 1])  # (B, H, W, C)

        # Branch H-C (permute to height, channels, width)
        x_hc = keras.ops.transpose(inputs, [0, 1, 3, 2])  # (B, H, C, W)
        x_hc = self.se_hc(x_hc, training=training)
        # Z-pooling
        avg_hc = keras.ops.mean(x_hc, axis=-1, keepdims=True)
        max_hc = keras.ops.max(x_hc, axis=-1, keepdims=True)
        pooled_hc = keras.ops.concatenate([avg_hc, max_hc], axis=-1)
        # Conv + BN + Sigmoid
        att_hc = self.conv_hc(pooled_hc)
        att_hc = self.bn_hc(att_hc, training=training)
        att_hc = keras.ops.sigmoid(att_hc)
        scaled_hc = keras.ops.multiply(x_hc, att_hc)
        # Rotate back
        out_hc = keras.ops.transpose(scaled_hc, [0, 1, 3, 2])  # (B, H, W, C)

        # Average the three branches
        output = keras.ops.add(keras.ops.add(out_hw, out_cw), out_hc)
        output = keras.ops.divide(output, 3.0)

        return output

    def get_config(self) -> dict:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.saving.serialize_keras_object(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TripSE2":
        """Create layer from config."""
        kernel_regularizer = config.pop("kernel_regularizer", None)
        if kernel_regularizer is not None:
            config["kernel_regularizer"] = keras.saving.deserialize_keras_object(kernel_regularizer)
        return cls(**config)


@keras.saving.register_keras_serializable(package="dl_techniques")
class TripSE3(keras.layers.Layer):
    """
    TripSE3: Triplet Attention with SE blocks embedded within branches (parallel).

    Architecture:
    1. Each branch processes input through triplet attention (rotation + Z-pool + conv)
    2. In parallel, SE block processes permuted input
    3. SE output scales the branch attention map
    4. Average branch outputs

    This variant applies intermediate channel attention in parallel within each branch.

    ASCII Diagram::

                                Input (B, H, W, C)
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
                Branch H-W          Branch C-W          Branch H-C
                    |                   |                   |
                (no rotation)       Permute (C,W,H)     Permute (H,C,W)
                    |                   |                   |
            +-------+-------+     +-----+-------+     +-----+-------+
            |               |     |             |     |             |
        Z-Pool          SE Block  Z-Pool    SE Block  Z-Pool    SE Block
     (avg+max pool)  (GAP→FC→FC)  |       (GAP→FC→FC)  |       (GAP→FC→FC)
            |               |     |             |     |             |
        Conv 7x7            |     Conv 7x7      |     Conv 7x7      |
            |               |     |             |     |             |
        BatchNorm           |     BatchNorm     |     BatchNorm     |
            |               |     |             |     |             |
        Sigmoid             |     Sigmoid       |     Sigmoid       |
            |               |     |             |     |             |
            +-------×-------+     +------×------+     +------×------+
                    |                   |                   |
              (multiply att        (multiply att      (multiply att
               with SE weights)     with SE weights)   with SE weights)
                    |                   |                   |
                Scale Input         Scale Input         Scale Input
                    |                   |                   |
                (no rotation)       Rotate Back         Rotate Back
                    |                   |                   |
                    +-------------------+-------------------+
                                        |
                                    Average
                                        |
                                Output (B, H, W, C)

    Parameters
    ----------
    reduction_ratio : int, default=16
        Channel reduction ratio for SE blocks.
    kernel_size : int, default=7
        Convolutional kernel size for triplet attention.
    use_bias : bool, default=False
        Whether to use bias in conv layers.
    kernel_initializer : str, default="glorot_uniform"
        Initializer for kernel weights.
    kernel_regularizer : Optional, default=None
        Regularizer for kernel weights.

    References
    ----------
    Alhazmi & Altahhan, "Achieving 3D Attention via Triplet Squeeze and Excitation Block," 2025.
    """

    def __init__(
        self,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Sub-layers will be created in build()
        self.se_hw = None
        self.se_cw = None
        self.se_hc = None

        self.conv_hw = None
        self.conv_cw = None
        self.conv_hc = None

        self.bn_hw = None
        self.bn_cw = None
        self.bn_hc = None

        logger.debug(
            f"Initialized TripSE3: reduction_ratio={reduction_ratio}, "
            f"kernel_size={kernel_size}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Parameters
        ----------
        input_shape : Tuple
            Shape of input tensor (batch, height, width, channels).
        """
        batch, height, width, channels = input_shape

        # SE blocks for each branch (operate on permuted input)
        self.se_hw = SqueezeExcitation(
            channels=channels,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_hw"
        )
        self.se_hw.build(input_shape)

        self.se_cw = SqueezeExcitation(
            channels=height,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_cw"
        )
        self.se_cw.build((batch, channels, width, height))

        self.se_hc = SqueezeExcitation(
            channels=width,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_hc"
        )
        self.se_hc.build((batch, height, channels, width))

        # Conv layers for triplet attention processing
        self.conv_hw = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_hw"
        )
        self.conv_hw.build((batch, height, width, 2))

        self.conv_cw = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_cw"
        )
        self.conv_cw.build((batch, channels, width, 2))

        self.conv_hc = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_hc"
        )
        self.conv_hc.build((batch, height, channels, 2))

        # BatchNorm layers
        self.bn_hw = keras.layers.BatchNormalization(name="bn_hw")
        self.bn_hw.build((batch, height, width, 1))

        self.bn_cw = keras.layers.BatchNormalization(name="bn_cw")
        self.bn_cw.build((batch, channels, width, 1))

        self.bn_hc = keras.layers.BatchNormalization(name="bn_hc")
        self.bn_hc.build((batch, height, channels, 1))

        super().build(input_shape)
        logger.debug(f"Built TripSE3 with input_shape={input_shape}, channels={channels}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of TripSE3.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch, height, width, channels).
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        keras.KerasTensor
            Attention-modulated tensor of same shape as input.
        """
        # Branch H-W (no rotation)
        x_hw = inputs
        # Parallel: Triplet attention processing
        avg_hw = keras.ops.mean(x_hw, axis=-1, keepdims=True)
        max_hw = keras.ops.max(x_hw, axis=-1, keepdims=True)
        pooled_hw = keras.ops.concatenate([avg_hw, max_hw], axis=-1)
        att_hw = self.conv_hw(pooled_hw)
        att_hw = self.bn_hw(att_hw, training=training)
        att_hw = keras.ops.sigmoid(att_hw)
        # Parallel: SE block
        se_weights_hw = self.se_hw(x_hw, training=training)
        # Combine: multiply attention by SE weights, then scale input
        combined_att_hw = keras.ops.multiply(att_hw, se_weights_hw)
        out_hw = keras.ops.multiply(x_hw, combined_att_hw)

        # Branch C-W (permute)
        x_cw = keras.ops.transpose(inputs, [0, 3, 2, 1])  # (B, C, W, H)
        # Triplet attention
        avg_cw = keras.ops.mean(x_cw, axis=-1, keepdims=True)
        max_cw = keras.ops.max(x_cw, axis=-1, keepdims=True)
        pooled_cw = keras.ops.concatenate([avg_cw, max_cw], axis=-1)
        att_cw = self.conv_cw(pooled_cw)
        att_cw = self.bn_cw(att_cw, training=training)
        att_cw = keras.ops.sigmoid(att_cw)
        # SE block
        se_weights_cw = self.se_cw(x_cw, training=training)
        # Combine
        combined_att_cw = keras.ops.multiply(att_cw, se_weights_cw)
        scaled_cw = keras.ops.multiply(x_cw, combined_att_cw)
        out_cw = keras.ops.transpose(scaled_cw, [0, 3, 2, 1])  # Back to (B, H, W, C)

        # Branch H-C (permute)
        x_hc = keras.ops.transpose(inputs, [0, 1, 3, 2])  # (B, H, C, W)
        # Triplet attention
        avg_hc = keras.ops.mean(x_hc, axis=-1, keepdims=True)
        max_hc = keras.ops.max(x_hc, axis=-1, keepdims=True)
        pooled_hc = keras.ops.concatenate([avg_hc, max_hc], axis=-1)
        att_hc = self.conv_hc(pooled_hc)
        att_hc = self.bn_hc(att_hc, training=training)
        att_hc = keras.ops.sigmoid(att_hc)
        # SE block
        se_weights_hc = self.se_hc(x_hc, training=training)
        # Combine
        combined_att_hc = keras.ops.multiply(att_hc, se_weights_hc)
        scaled_hc = keras.ops.multiply(x_hc, combined_att_hc)
        out_hc = keras.ops.transpose(scaled_hc, [0, 1, 3, 2])  # Back to (B, H, W, C)

        # Average the three branches
        output = keras.ops.add(keras.ops.add(out_hw, out_cw), out_hc)
        output = keras.ops.divide(output, 3.0)

        return output

    def get_config(self) -> dict:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.saving.serialize_keras_object(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TripSE3":
        """Create layer from config."""
        kernel_regularizer = config.pop("kernel_regularizer", None)
        if kernel_regularizer is not None:
            config["kernel_regularizer"] = keras.saving.deserialize_keras_object(kernel_regularizer)
        return cls(**config)


@keras.saving.register_keras_serializable(package="dl_techniques")
class TripSE4(keras.layers.Layer):
    """
    TripSE4: Hybrid TripSE with SE at multiple positions and affine transformation.

    Architecture:
    1. Each branch: triplet attention processing in parallel with SE block
    2. SE output is added to branch attention (affine: shift + scale)
    3. Combined through sigmoid to create 3D attention tensor
    4. Sum all branch outputs
    5. Apply final SE block on summed output

    This is the most complex variant with maximum flexibility and best performance.

    ASCII Diagram::

                                Input (B, H, W, C)
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
                Branch H-W          Branch C-W          Branch H-C
                    |                   |                   |
                (no rotation)       Permute (C,W,H)     Permute (H,C,W)
                    |                   |                   |
            +-------+-------+     +-----+-------+     +-----+-------+
            |               |     |             |     |             |
        Z-Pool          SE Block  Z-Pool    SE Block  Z-Pool    SE Block
     (avg+max pool)  (GAP→squeeze)  |     (GAP→squeeze)  |     (GAP→squeeze)
            |               |     |             |     |             |
        Conv 7x7            |     Conv 7x7      |     Conv 7x7      |
            |               |     |             |     |             |
        BatchNorm           |     BatchNorm     |     BatchNorm     |
            |               |     |             |     |             |
            |    Broadcast  |     |    Broadcast|     |    Broadcast|
            |    to 3D      |     |    to 3D    |     |    to 3D    |
            |               |     |             |     |             |
            +-------+-------+     +------+------+     +------+------+
                    |                   |                   |
              (add: affine          (add: affine       (add: affine
               transformation)       transformation)    transformation)
                    |                   |                   |
                Sigmoid             Sigmoid             Sigmoid
              (3D attention)      (3D attention)      (3D attention)
                    |                   |                   |
                Scale Input         Scale Input         Scale Input
                    |                   |                   |
                (no rotation)       Rotate Back         Rotate Back
                    |                   |                   |
                    +-------------------+-------------------+
                                        |
                                      Sum
                                        |
                                Final SE Block
                                (GAP → FC → FC)
                                        |
                                Output (B, H, W, C)

    Note: The key innovation in TripSE4 is the affine transformation that adds SE weights
    to branch attention before sigmoid, creating true 3D attention tensors rather than
    just 2D attention maps.

    Parameters
    ----------
    reduction_ratio : int, default=16
        Channel reduction ratio for SE blocks.
    kernel_size : int, default=7
        Convolutional kernel size for triplet attention.
    use_bias : bool, default=False
        Whether to use bias in conv layers.
    kernel_initializer : str, default="glorot_uniform"
        Initializer for kernel weights.
    kernel_regularizer : Optional, default=None
        Regularizer for kernel weights.

    References
    ----------
    Alhazmi & Altahhan, "Achieving 3D Attention via Triplet Squeeze and Excitation Block," 2025.
    """

    def __init__(
        self,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        use_bias: bool = False,
        kernel_initializer: str = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        # Sub-layers will be created in build()
        self.se_hw = None
        self.se_cw = None
        self.se_hc = None
        self.se_final = None

        self.conv_hw = None
        self.conv_cw = None
        self.conv_hc = None

        self.bn_hw = None
        self.bn_cw = None
        self.bn_hc = None

        logger.debug(
            f"Initialized TripSE4: reduction_ratio={reduction_ratio}, "
            f"kernel_size={kernel_size}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Parameters
        ----------
        input_shape : Tuple
            Shape of input tensor (batch, height, width, channels).
        """
        batch, height, width, channels = input_shape

        # SE blocks for each branch
        self.se_hw = SqueezeExcitation(
            channels=channels,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_hw"
        )
        self.se_hw.build(input_shape)

        self.se_cw = SqueezeExcitation(
            channels=height,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_cw"
        )
        self.se_cw.build((batch, channels, width, height))

        self.se_hc = SqueezeExcitation(
            channels=width,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_hc"
        )
        self.se_hc.build((batch, height, channels, width))

        # Final SE block on summed output
        self.se_final = SqueezeExcitation(
            channels=channels,
            reduction_ratio=self.reduction_ratio,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_final"
        )
        self.se_final.build(input_shape)

        # Conv layers
        self.conv_hw = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_hw"
        )
        self.conv_hw.build((batch, height, width, 2))

        self.conv_cw = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_cw"
        )
        self.conv_cw.build((batch, channels, width, 2))

        self.conv_hc = keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_hc"
        )
        self.conv_hc.build((batch, height, channels, 2))

        # BatchNorm layers
        self.bn_hw = keras.layers.BatchNormalization(name="bn_hw")
        self.bn_hw.build((batch, height, width, 1))

        self.bn_cw = keras.layers.BatchNormalization(name="bn_cw")
        self.bn_cw.build((batch, channels, width, 1))

        self.bn_hc = keras.layers.BatchNormalization(name="bn_hc")
        self.bn_hc.build((batch, height, channels, 1))

        super().build(input_shape)
        logger.debug(f"Built TripSE4 with input_shape={input_shape}, channels={channels}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of TripSE4.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch, height, width, channels).
        training : bool, optional
            Whether in training mode.

        Returns
        -------
        keras.KerasTensor
            Attention-modulated tensor of same shape as input.
        """
        # Branch H-W
        x_hw = inputs
        # Triplet attention processing
        avg_hw = keras.ops.mean(x_hw, axis=-1, keepdims=True)
        max_hw = keras.ops.max(x_hw, axis=-1, keepdims=True)
        pooled_hw = keras.ops.concatenate([avg_hw, max_hw], axis=-1)
        att_hw = self.conv_hw(pooled_hw)
        att_hw = self.bn_hw(att_hw, training=training)
        # SE block (produces channel weights, need to broadcast to 3D)
        # SE output has shape (B, 1, 1, C) after processing
        # We need it to match att_hw shape (B, H, W, 1)
        # Get SE channel attention vector
        se_out_hw = self.se_hw(x_hw, training=training)
        # Extract just the attention weights without multiplying
        # We need to recompute SE to get just the weights
        squeeze_hw = keras.ops.mean(x_hw, axis=[1, 2], keepdims=True)  # (B, 1, 1, C)
        # Add this to the branch attention (affine transformation: shift)
        # att_hw is (B, H, W, 1), squeeze_hw is (B, 1, 1, C)
        # We need to broadcast squeeze_hw to (B, H, W, C)
        se_weights_hw = keras.ops.tile(squeeze_hw, [1, keras.ops.shape(att_hw)[1], keras.ops.shape(att_hw)[2], 1])
        # att_hw needs to be broadcast to (B, H, W, C)
        att_hw_broadcast = keras.ops.tile(att_hw, [1, 1, 1, keras.ops.shape(x_hw)[-1]])
        # Affine: add SE weights (shift) to attention
        combined_hw = keras.ops.add(att_hw_broadcast, se_weights_hw)
        combined_hw = keras.ops.sigmoid(combined_hw)  # 3D attention tensor
        out_hw = keras.ops.multiply(x_hw, combined_hw)

        # Branch C-W
        x_cw = keras.ops.transpose(inputs, [0, 3, 2, 1])  # (B, C, W, H)
        avg_cw = keras.ops.mean(x_cw, axis=-1, keepdims=True)
        max_cw = keras.ops.max(x_cw, axis=-1, keepdims=True)
        pooled_cw = keras.ops.concatenate([avg_cw, max_cw], axis=-1)
        att_cw = self.conv_cw(pooled_cw)
        att_cw = self.bn_cw(att_cw, training=training)
        squeeze_cw = keras.ops.mean(x_cw, axis=[1, 2], keepdims=True)
        se_weights_cw = keras.ops.tile(squeeze_cw, [1, keras.ops.shape(att_cw)[1], keras.ops.shape(att_cw)[2], 1])
        att_cw_broadcast = keras.ops.tile(att_cw, [1, 1, 1, keras.ops.shape(x_cw)[-1]])
        combined_cw = keras.ops.add(att_cw_broadcast, se_weights_cw)
        combined_cw = keras.ops.sigmoid(combined_cw)
        scaled_cw = keras.ops.multiply(x_cw, combined_cw)
        out_cw = keras.ops.transpose(scaled_cw, [0, 3, 2, 1])

        # Branch H-C
        x_hc = keras.ops.transpose(inputs, [0, 1, 3, 2])  # (B, H, C, W)
        avg_hc = keras.ops.mean(x_hc, axis=-1, keepdims=True)
        max_hc = keras.ops.max(x_hc, axis=-1, keepdims=True)
        pooled_hc = keras.ops.concatenate([avg_hc, max_hc], axis=-1)
        att_hc = self.conv_hc(pooled_hc)
        att_hc = self.bn_hc(att_hc, training=training)
        squeeze_hc = keras.ops.mean(x_hc, axis=[1, 2], keepdims=True)
        se_weights_hc = keras.ops.tile(squeeze_hc, [1, keras.ops.shape(att_hc)[1], keras.ops.shape(att_hc)[2], 1])
        att_hc_broadcast = keras.ops.tile(att_hc, [1, 1, 1, keras.ops.shape(x_hc)[-1]])
        combined_hc = keras.ops.add(att_hc_broadcast, se_weights_hc)
        combined_hc = keras.ops.sigmoid(combined_hc)
        scaled_hc = keras.ops.multiply(x_hc, combined_hc)
        out_hc = keras.ops.transpose(scaled_hc, [0, 1, 3, 2])

        # Sum all branches
        combined = keras.ops.add(keras.ops.add(out_hw, out_cw), out_hc)

        # Apply final SE block
        output = self.se_final(combined, training=training)

        return output

    def get_config(self) -> dict:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.saving.serialize_keras_object(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TripSE4":
        """Create layer from config."""
        kernel_regularizer = config.pop("kernel_regularizer", None)
        if kernel_regularizer is not None:
            config["kernel_regularizer"] = keras.saving.deserialize_keras_object(kernel_regularizer)
        return cls(**config)


# Convenience alias for the most commonly used variant
TripSE = TripSE1