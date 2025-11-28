"""
DarkIR: Robust Low-Light Image Restoration Network
==================================================

A Keras 3 implementation of DarkIR, presented at CVPR 2025.

DarkIR is an efficient, all-in-one image restoration network designed to handle
multiple degradations simultaneously: low-light, noise, and blur. Unlike many modern
restoration networks that rely heavily on Vision Transformers, DarkIR utilizes efficient
CNNs augmented with frequency-domain processing and dilated attention mechanisms.

Architectural Highlights:
-------------------------
1. **U-Net Structure**: A hierarchical encoder-decoder architecture with skip connections.
2. **Dilated Branching**: Uses parallel dilated convolutions (rates 1, 4, 9) to capture
   multi-scale context without increasing parameter count significantly.
3. **Frequency MLP (FreMLP)**: Processes features in the Fourier domain to capture global
   characteristics efficiently, replacing heavy self-attention mechanisms.
4. **SimpleGate**: A parameter-free gating mechanism (element-wise multiplication of
   channel splits) used for non-linear activation.
5. **PixelShuffle Upsampling**: Uses sub-pixel convolution for high-quality resolution recovery.
6. **Global Residual**: The network learns the residual image to add to the input.

Key Components:
---------------
- **DarkIREncoderBlock**: Contains Normalization -> Dilated Branches -> SimpleGate -> Frequency MLP.
- **DarkIRDecoderBlock**: Similar to Encoder but replaces FreMLP with inverted FFN structures.
- **DilatedBranch**: A specific branch within blocks handling specific dilation rates.

Usage Example:
--------------
```python
from dl_techniques.models.darkir import create_darkir_model
from dl_techniques.optimization import optimizer_builder

# 1. Create the model (Standard "Medium" Configuration)
model = create_darkir_model(
    img_channels=3,
    width=32,                  # Base feature dimension
    enc_blk_nums=[1, 2, 3],    # Blocks per encoder stage
    dec_blk_nums=[3, 1, 1],    # Blocks per decoder stage
    dilations=[1, 4, 9],       # Dilation rates for parallel branches
    extra_depth_wise=True
)

# 2. Compile for Training
# DarkIR typically uses Charbonnier or L1 Loss.
# We use AdamW with cosine decay.
optimizer = optimizer_builder({
    "type": "adamw",
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "gradient_clipping_by_norm": 1.0
})

model.compile(optimizer=optimizer, loss='mean_absolute_error')

# 3. Inference
# Input: (B, H, W, 3) Scaled [0, 1]
# Output: (B, H, W, 3) Scaled [0, 1]
restored_img = model.predict(low_light_image_batch)
```

Input/Output Shapes:
--------------------
- **Input**: `(Batch, Height, Width, Channels)`
  - Height and Width should ideally be multiples of `2^num_stages` (e.g., multiple of 8 for 3 stages).
  - Data type: Float32, range [0, 1].
- **Output**: `(Batch, Height, Width, Channels)`
  - Same shape as input. Represents the restored image.

Reference:
----------
Feijoo, D., Benito, J. C., Garcia, A., & Conde, M. V. (2025).
"DarkIR: Robust Low-Light Image Restoration". CVPR 2025.
"""

import keras
from keras import layers, ops
from typing import List, Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SimpleGate(keras.layers.Layer):
    """
    SimpleGate: Element-wise multiplicative gating without learnable parameters.

    This layer implements a parameter-free gating mechanism by splitting the input
    channels in half and performing element-wise multiplication. This is an efficient
    alternative to more complex gating mechanisms like GLU or GTU.

    **Intent**: Provide efficient non-linear activation through channel interaction
    without additional parameters, commonly used in efficient vision architectures.

    **Architecture**:
    ```
    Input(shape=[..., 2*C])
           ↓
    Split: x1, x2 (shape=[..., C] each)
           ↓
    Gate: output = x1 ⊙ x2  (element-wise multiply)
           ↓
    Output(shape=[..., C])
    ```

    **Mathematical Operation**:
        Given input x ∈ ℝ^(B×H×W×2C):
        x1, x2 = split(x, axis=-1)  # Each ∈ ℝ^(B×H×W×C)
        output = x1 ⊙ x2            # ∈ ℝ^(B×H×W×C)

    Args:
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch, height, width, channels)`.
        Note: `channels` must be even (divisible by 2).

    Output shape:
        4D tensor with shape: `(batch, height, width, channels // 2)`.

    Example:
        ```python
        # Basic usage
        gate = SimpleGate()
        x = ops.random.normal((2, 32, 32, 64))  # 64 channels
        y = gate(x)  # Output: (2, 32, 32, 32)

        # In a model
        inputs = keras.Input((None, None, 3))
        x = layers.Conv2D(64, 3, padding='same')(inputs)
        x = SimpleGate()(x)  # Reduces to 32 channels
        ```

    Note:
        Input channels must be even. If using SimpleGate, ensure preceding layers
        produce even number of channels (typically achieved by using expansion
        factors that result in even dimensions).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply gating operation by splitting and multiplying.

        Args:
            x: Input tensor of shape (batch, height, width, 2*channels).

        Returns:
            Gated output of shape (batch, height, width, channels).
        """
        # Split along the channel axis (last axis in Keras NHWC format)
        x1, x2 = ops.split(x, 2, axis=-1)
        return x1 * x2

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (channels are halved).

        Args:
            input_shape: Shape tuple of input.

        Returns:
            Shape tuple of output with channels // 2.
        """
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1] // 2 if input_shape[-1] is not None else None
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return super().get_config()


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FreMLP(keras.layers.Layer):
    """
    Frequency MLP: Processes features in the frequency domain for global modeling.

    This layer applies FFT to transform features to frequency domain, processes
    the magnitude spectrum with a simple MLP (keeping phase unchanged), and
    transforms back to spatial domain. This provides an efficient way to capture
    global context without the quadratic complexity of self-attention.

    **Intent**: Enable efficient global feature modeling in CNNs by operating in
    the frequency domain, replacing expensive self-attention mechanisms while
    maintaining similar receptive field properties.

    **Architecture**:
    ```
    Input(shape=[B, H, W, C])
           ↓
    FFT: x_freq = rfft2(x)  # Complex (B, H, W//2+1, C)
           ↓
    Decompose: mag = |x_freq|, phase = ∠x_freq
           ↓
    Process Magnitude:
      mag -> Conv1x1(C → expansion*C) -> LeakyReLU -> Conv1x1(expansion*C → C)
           ↓
    Reconstruct: x_freq' = mag' * exp(i*phase)
           ↓
    IFFT: output = irfft2(x_freq')  # Real (B, H, W, C)
           ↓
    Output(shape=[B, H, W, C])
    ```

    **Mathematical Operations**:
    1. **Forward FFT**: X_freq = FFT2D(x), where X_freq ∈ ℂ^(B×H×W'×C)
    2. **Decomposition**: mag = |X_freq|, phase = arg(X_freq)
    3. **Magnitude Processing**: mag' = MLP(mag)
    4. **Reconstruction**: X'_freq = mag' ⊙ exp(i·phase)
    5. **Inverse FFT**: output = IFFT2D(X'_freq)

    Where:
    - FFT2D/IFFT2D are 2D Fast Fourier Transform operations
    - | · | denotes complex magnitude
    - arg(·) denotes complex phase
    - ⊙ denotes element-wise multiplication

    Args:
        channels: Integer, number of input/output channels. Must be positive.
            The layer maintains channel dimensionality throughout processing.
        expansion: Integer, expansion factor for internal MLP hidden dimension.
            Hidden dimension = channels * expansion. Defaults to 2.
            Larger values provide more capacity but increase computation.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch, height, width, channels)`.
        Shape is preserved; only feature values are transformed.

    Attributes:
        channels: Number of input/output channels.
        expansion: Expansion factor for MLP hidden dimension.
        conv1: First 1x1 convolution (channels → expansion*channels).
        act: LeakyReLU activation with negative slope 0.1.
        conv2: Second 1x1 convolution (expansion*channels → channels).

    Example:
        ```python
        # Basic usage
        freq_mlp = FreMLP(channels=64, expansion=2)
        x = ops.random.normal((2, 32, 32, 64))
        y = freq_mlp(x)  # Output: (2, 32, 32, 64)

        # With different expansion
        freq_mlp = FreMLP(channels=128, expansion=4)  # More capacity

        # In a restoration network
        x = layers.Conv2D(64, 3, padding='same')(inputs)
        x = FreMLP(channels=64)(x)  # Global frequency processing
        ```

    Note:
        - Uses rfft2 (real FFT) for efficiency since input is real-valued
        - Phase information is preserved; only magnitude is processed
        - Computation is O(HW log(HW)) due to FFT, independent of channel count
        - More efficient than self-attention for global modeling (O(H²W²))
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")

        self.channels = channels
        self.expansion = expansion

        # Create sub-layers in __init__
        hidden_dim = int(channels * expansion)
        self.conv1 = layers.Conv2D(
            hidden_dim,
            kernel_size=1,
            strides=1,
            padding='valid',
            name='conv1'
        )
        self.act = layers.LeakyReLU(negative_slope=0.1, name='leaky_relu')
        self.conv2 = layers.Conv2D(
            channels,
            kernel_size=1,
            strides=1,
            padding='valid',
            name='conv2'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build sub-layers for proper serialization.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Build sub-layers explicitly
        self.conv1.build(input_shape)

        # Compute intermediate shape after conv1
        conv1_output_shape = self.conv1.compute_output_shape(input_shape)
        self.act.build(conv1_output_shape)
        self.conv2.build(conv1_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Forward pass: FFT -> Process Magnitude -> IFFT.

        Args:
            x: Input tensor of shape (batch, height, width, channels).

        Returns:
            Output tensor of shape (batch, height, width, channels).
        """
        # Get spatial dimensions for IFFT later
        _, h, w, _ = ops.shape(x)

        # 1. FFT (Fast Fourier Transform) over spatial dimensions (H, W)
        # rfft2: real-to-complex FFT, more efficient for real inputs
        # axes=(1, 2): apply over height and width dimensions
        x_freq = ops.fft.rfft2(x, axes=(1, 2), norm='backward')

        # 2. Extract Magnitude and Phase
        mag = ops.abs(x_freq)
        pha = ops.angle(x_freq)

        # 3. Process Magnitude through MLP
        mag = self.conv1(mag)
        mag = self.act(mag)
        mag = self.conv2(mag)

        # 4. Reconstruct Complex Tensor from processed magnitude and original phase
        real = mag * ops.cos(pha)
        imag = mag * ops.sin(pha)
        x_out_complex = ops.complex(real, imag)

        # 5. Inverse FFT to return to spatial domain
        # s=(h, w): ensure output matches input spatial dimensions
        x_out = ops.fft.irfft2(x_out_complex, axes=(1, 2), s=(h, w), norm='backward')

        return x_out

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input).

        Args:
            input_shape: Shape tuple of input.

        Returns:
            Shape tuple of output (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "expansion": self.expansion
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DilatedBranch(keras.layers.Layer):
    """
    A single branch of dilated depthwise convolution for multi-scale context.

    This layer implements a depthwise convolution with a specific dilation rate,
    used as one branch in parallel multi-scale processing. Depthwise convolutions
    apply separate filters per input channel, reducing parameters while capturing
    spatial patterns at the specified dilation rate.

    **Intent**: Capture spatial context at a specific scale determined by dilation
    rate, to be combined with other branches for multi-scale feature extraction.

    **Architecture**:
    ```
    Input(shape=[B, H, W, C])
           ↓
    Depthwise Conv3x3 (dilation=d, groups=C)
           ↓
    Output(shape=[B, H, W, C*expansion])
    ```

    **Mathematical Operation**:
        For each channel i independently:
        output[:, :, :, i] = conv3x3_dilated(input[:, :, :, i], dilation=d)

    Where dilation rate d controls receptive field size without increasing parameters.

    Args:
        channels: Integer, number of input channels. Used for reference but expansion
            determines output channels. Must be positive.
        expansion: Integer, factor to expand channels by. Output channels will be
            channels * expansion. Defaults to 1 (no expansion). Must be positive.
        dilation: Integer, dilation rate for the convolution. Controls spatial
            receptive field. Common values: 1 (standard), 4, 9 (multi-scale).
            Must be positive. Defaults to 1.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch, height, width, channels * expansion)`.
        Spatial dimensions preserved by padding='same'.

    Attributes:
        channels: Number of input channels.
        expansion: Channel expansion factor.
        dilation: Dilation rate for convolution.
        dw_channels: Computed output channels (channels * expansion).
        conv: Depthwise Conv2D layer with specified dilation.

    Example:
        ```python
        # Single branch with dilation rate 1 (standard convolution)
        branch1 = DilatedBranch(channels=64, expansion=2, dilation=1)

        # Branch with dilation rate 4 (larger receptive field)
        branch2 = DilatedBranch(channels=64, expansion=2, dilation=4)

        # Multi-scale parallel processing
        x = ops.random.normal((2, 32, 32, 64))
        branches = [
            DilatedBranch(channels=64, expansion=2, dilation=d)
            for d in [1, 4, 9]
        ]
        # Combine outputs: sum or concatenate
        outputs = [branch(x) for branch in branches]
        combined = ops.add_n(outputs)  # Element-wise sum
        ```

    Note:
        - Uses groups=dw_channels for true depthwise convolution
        - padding='same' maintains spatial dimensions regardless of dilation
        - Dilation rate increases receptive field: (kernel_size + (kernel_size-1)*(dilation-1))
        - For kernel_size=3: dilation 1→3×3, dilation 4→11×11, dilation 9→19×19
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 1,
        dilation: int = 1,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        self.channels = channels
        self.expansion = expansion
        self.dilation = dilation
        self.dw_channels = int(channels * expansion)

        # Create depthwise convolution in __init__
        self.conv = layers.Conv2D(
            filters=self.dw_channels,
            kernel_size=3,
            padding="same",
            dilation_rate=self.dilation,
            groups=self.dw_channels,  # Depthwise: each filter processes one channel
            use_bias=True,
            name=f"dilated_conv_d{self.dilation}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the convolution layer.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Build sub-layer explicitly
        self.conv.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply dilated depthwise convolution.

        Args:
            x: Input tensor of shape (batch, height, width, channels).

        Returns:
            Output tensor of shape (batch, height, width, channels * expansion).
        """
        return self.conv(x)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Shape tuple of input.

        Returns:
            Shape tuple of output with expanded channels.
        """
        return self.conv.compute_output_shape(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "expansion": self.expansion,
            "dilation": self.dilation
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DarkIREncoderBlock(keras.layers.Layer):
    """
    Encoder Block (EBlock) for DarkIR with parallel dilated branches and FreMLP.

    This block implements the core encoder component of DarkIR, featuring:
    1. Parallel dilated convolution branches for multi-scale context
    2. Channel attention for adaptive feature weighting
    3. SimpleGate activation for efficient non-linearity
    4. Frequency MLP for global feature modeling

    The block uses a dual-residual structure with learnable scaling factors.

    **Intent**: Extract multi-scale features while capturing both local (via dilated
    convolutions) and global (via FreMLP) context for robust low-light restoration.

    **Architecture**:
    ```
    Input(shape=[B, H, W, C])
           ↓
    ┌──────────────────────────────────────────────┐
    │ Path 1: Multi-scale Dilated Processing      │
    ├──────────────────────────────────────────────┤
    LayerNorm → [Optional DW Conv] → 1×1 Conv(→C*dw_expand)
           ↓
    Parallel Dilated Branches [d₁, d₂, ..., dₙ] → Sum
           ↓
    SimpleGate (→C*dw_expand/2)
           ↓
    Channel Attention: GlobalAvgPool → 1×1 Conv → Multiply
           ↓
    1×1 Conv(→C) → Add with Input (scaled by β)
    └──────────────────────────────────────────────┘
           ↓ y
    ┌──────────────────────────────────────────────┐
    │ Path 2: Frequency Domain Processing         │
    ├──────────────────────────────────────────────┤
    LayerNorm → FreMLP → Multiply with y
           ↓
    Add to y (scaled by γ)
    └──────────────────────────────────────────────┘
           ↓
    Output(shape=[B, H, W, C])
    ```

    **Mathematical Operations**:
    1. **Path 1 (Dilated)**: y = x + β · Conv1x1(Attn(SG(∑ᵢ DConvᵢ(Conv1x1(Norm(x))))))
    2. **Path 2 (Frequency)**: out = y + γ · (y ⊙ FreMLP(Norm(y)))

    Where:
    - β, γ: learnable scalar parameters (initialized to 0)
    - DConvᵢ: dilated convolution with rate dᵢ
    - SG: SimpleGate activation
    - Attn: channel attention mechanism
    - ⊙: element-wise multiplication

    Args:
        channels: Integer, number of input/output channels. Must be positive.
            This determines the feature dimension maintained throughout the block.
        dw_expand: Integer, expansion factor for depthwise convolution channels.
            Intermediate channels = channels * dw_expand. Defaults to 2.
            Must be positive and even (for SimpleGate).
        dilations: List of integers, dilation rates for parallel branches.
            Each value creates one branch. Common: [1, 4, 9] for multi-scale.
            Defaults to [1]. All values must be positive.
        extra_depth_wise: Boolean, whether to add extra depthwise conv before branching.
            Adds additional inductive bias. Defaults to False.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch, height, width, channels)`.
        Shape is preserved; residual connections maintain dimensionality.

    Attributes:
        channels: Number of channels.
        dw_expand: Depthwise expansion factor.
        dilations: List of dilation rates.
        extra_depth_wise: Whether extra DW conv is used.
        expanded_channels: Computed intermediate channels.
        norm1, norm2: LayerNorm layers for two paths.
        extra_conv: Optional extra depthwise convolution.
        conv1: 1x1 projection to expanded channels.
        branches: List of DilatedBranch layers.
        sca_avg: Global average pooling for channel attention.
        sca_conv: 1x1 conv for channel attention.
        sg1: SimpleGate activation.
        conv3: 1x1 projection back to original channels.
        freq: FreMLP for frequency domain processing.
        gamma, beta: Learnable residual scaling factors.

    Example:
        ```python
        # Standard encoder block
        block = DarkIREncoderBlock(
            channels=64,
            dw_expand=2,
            dilations=[1, 4, 9]
        )
        x = ops.random.normal((2, 32, 32, 64))
        y = block(x)  # Output: (2, 32, 32, 64)

        # With extra depthwise conv for more inductive bias
        block = DarkIREncoderBlock(
            channels=64,
            dilations=[1, 4, 9],
            extra_depth_wise=True
        )

        # Single-scale (no multi-scale branching)
        block = DarkIREncoderBlock(channels=64, dilations=[1])
        ```

    Note:
        - dw_expand should result in even channels for SimpleGate to work
        - Learnable scales (beta, gamma) start at 0 for stable training
        - FreMLP provides global context without attention's quadratic cost
        - Multiple residual paths enable better gradient flow
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        dilations: List[int] = None,
        extra_depth_wise: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if dw_expand <= 0:
            raise ValueError(f"dw_expand must be positive, got {dw_expand}")
        if dilations is None:
            dilations = [1]
        if not dilations:
            raise ValueError("dilations cannot be empty")
        if any(d <= 0 for d in dilations):
            raise ValueError(f"All dilations must be positive, got {dilations}")

        self.channels = channels
        self.dw_expand = dw_expand
        self.dilations = dilations
        self.extra_depth_wise = extra_depth_wise
        self.expanded_channels = channels * dw_expand

        # Create all sub-layers in __init__
        # Normalization layers
        self.norm1 = create_normalization_layer("layer_norm", axis=-1, epsilon=1e-6)
        self.norm2 = create_normalization_layer("layer_norm", axis=-1, epsilon=1e-6)

        # Extra DW Conv (Optional)
        if self.extra_depth_wise:
            self.extra_conv = layers.Conv2D(
                self.channels,
                kernel_size=3,
                padding="same",
                groups=self.channels,
                use_bias=True,
                name='extra_dw_conv'
            )
        else:
            self.extra_conv = layers.Identity(name='identity_extra')

        # Projection to expanded channels
        self.conv1 = layers.Conv2D(
            self.expanded_channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv1'
        )

        # Parallel Dilated Branches
        self.branches = [
            DilatedBranch(self.channels, self.dw_expand, d, name=f'branch_d{d}')
            for d in self.dilations
        ]

        # Channel Attention / Aggregation
        self.sca_avg = layers.GlobalAveragePooling2D(
            keepdims=True,
            name='channel_attn_pool'
        )
        self.sca_conv = layers.Conv2D(
            self.expanded_channels // 2,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='channel_attn_conv'
        )

        # SimpleGate activation
        self.sg1 = SimpleGate(name='simple_gate')

        # Projection back to original channels
        self.conv3 = layers.Conv2D(
            self.channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv3'
        )

        # Frequency MLP Block
        self.freq = FreMLP(self.channels, expansion=2, name='freq_mlp')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all sub-layers for proper serialization.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Build normalization layers
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # Build first path components
        self.extra_conv.build(input_shape)
        extra_shape = self.extra_conv.compute_output_shape(input_shape)

        self.conv1.build(extra_shape)
        conv1_shape = self.conv1.compute_output_shape(extra_shape)

        # Build all branches
        for branch in self.branches:
            branch.build(conv1_shape)

        # After branches, shape should match conv1_shape
        # SimpleGate halves the channels
        sg_input_shape = conv1_shape
        self.sg1.build(sg_input_shape)
        sg_output_shape = self.sg1.compute_output_shape(sg_input_shape)

        # Build channel attention
        self.sca_avg.build(sg_output_shape)
        sca_avg_shape = self.sca_avg.compute_output_shape(sg_output_shape)
        self.sca_conv.build(sca_avg_shape)

        # Build final projection
        self.conv3.build(sg_output_shape)

        # Build frequency path (operates on input_shape)
        self.freq.build(input_shape)

        # Create learnable scale parameters
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, self.channels),
            initializer="zeros",
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, self.channels),
            initializer="zeros",
            trainable=True
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with dual residual paths.

        Args:
            inputs: Input tensor of shape (batch, height, width, channels).
            training: Boolean, whether in training mode.

        Returns:
            Output tensor of shape (batch, height, width, channels).
        """
        # Store original input for residual
        y = inputs

        # === Path 1: Multi-scale Dilated Processing ===
        x = self.norm1(inputs)
        x = self.extra_conv(x)
        x = self.conv1(x)

        # Sum all parallel branches
        z = ops.add_n([branch(x) for branch in self.branches])

        # SimpleGate activation
        z = self.sg1(z)

        # Channel Attention
        attn = self.sca_avg(z)
        attn = self.sca_conv(attn)
        x = attn * z

        # Project back to original channels
        x = self.conv3(x)

        # First residual with learnable scale
        y = inputs + self.beta * x

        # === Path 2: Frequency Domain Processing ===
        x_step2 = self.norm2(y)
        x_freq = self.freq(x_step2)
        x = y * x_freq  # Element-wise modulation

        # Final residual with learnable scale
        out = y + x * self.gamma

        return out

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input).

        Args:
            input_shape: Shape tuple of input.

        Returns:
            Shape tuple of output (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "dw_expand": self.dw_expand,
            "dilations": self.dilations,
            "extra_depth_wise": self.extra_depth_wise
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DarkIRDecoderBlock(keras.layers.Layer):
    """
    Decoder Block (DBlock) for DarkIR with dual SimpleGate and FFN structure.

    This block implements the decoder component of DarkIR, similar to encoder but
    replacing FreMLP with a gated FFN structure. Features:
    1. Parallel dilated convolution branches for multi-scale context
    2. Channel attention for adaptive feature weighting
    3. Dual SimpleGate activations for efficient non-linearity
    4. Inverted FFN structure for feature refinement

    **Intent**: Refine multi-scale features for image reconstruction while maintaining
    efficient computation through gating mechanisms instead of heavy frequency processing.

    **Architecture**:
    ```
    Input(shape=[B, H, W, C])
           ↓
    ┌──────────────────────────────────────────────┐
    │ Path 1: Multi-scale Dilated Processing      │
    ├──────────────────────────────────────────────┤
    LayerNorm → 1×1 Conv(→C*dw_expand) → [Optional DW Conv]
           ↓
    Parallel Dilated Branches [d₁, d₂, ..., dₙ] → Sum
           ↓
    SimpleGate₁ (→C*dw_expand/2)
           ↓
    Channel Attention: GlobalAvgPool → 1×1 Conv → Multiply
           ↓
    1×1 Conv(→C) → Add with Input (scaled by β)
    └──────────────────────────────────────────────┘
           ↓ y
    ┌──────────────────────────────────────────────┐
    │ Path 2: Gated FFN Processing                │
    ├──────────────────────────────────────────────┤
    LayerNorm → 1×1 Conv(→C*ffn_expand)
           ↓
    SimpleGate₂ (→C*ffn_expand/2)
           ↓
    1×1 Conv(→C) → Add to y (scaled by γ)
    └──────────────────────────────────────────────┘
           ↓
    Output(shape=[B, H, W, C])
    ```

    **Mathematical Operations**:
    1. **Path 1 (Dilated)**: y = x + β · Conv1x1(Attn(SG₁(∑ᵢ DConvᵢ(Conv1x1(Norm(x))))))
    2. **Path 2 (Gated FFN)**: out = y + γ · Conv1x1(SG₂(Conv1x1(Norm(y))))

    Where:
    - β, γ: learnable scalar parameters (initialized to 0)
    - DConvᵢ: dilated convolution with rate dᵢ
    - SG₁, SG₂: SimpleGate activations (separate instances)
    - Attn: channel attention mechanism

    Args:
        channels: Integer, number of input/output channels. Must be positive.
            This determines the feature dimension maintained throughout the block.
        dw_expand: Integer, expansion factor for depthwise convolution channels.
            Intermediate channels for path 1 = channels * dw_expand. Defaults to 2.
            Must be positive and even (for SimpleGate).
        ffn_expand: Integer, expansion factor for FFN path.
            Intermediate channels for path 2 = channels * ffn_expand. Defaults to 2.
            Must be positive and even (for SimpleGate).
        dilations: List of integers, dilation rates for parallel branches.
            Each value creates one branch. Common: [1, 4, 9] for multi-scale.
            Defaults to [1]. All values must be positive.
        extra_depth_wise: Boolean, whether to add extra depthwise conv after projection.
            Note: In decoder, applied AFTER conv1 (opposite of encoder). Defaults to False.
        **kwargs: Additional arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch, height, width, channels)`.

    Output shape:
        4D tensor with shape: `(batch, height, width, channels)`.
        Shape is preserved; residual connections maintain dimensionality.

    Attributes:
        channels: Number of channels.
        dw_expand: Depthwise expansion factor.
        ffn_expand: FFN expansion factor.
        dilations: List of dilation rates.
        extra_depth_wise: Whether extra DW conv is used.
        dw_channels: Computed intermediate channels for path 1.
        ffn_channels: Computed intermediate channels for path 2.
        norm1, norm2: LayerNorm layers for two paths.
        conv1: 1x1 projection to expanded channels.
        extra_conv: Optional extra depthwise convolution.
        branches: List of DilatedBranch layers.
        sca_avg: Global average pooling for channel attention.
        sca_conv: 1x1 conv for channel attention.
        sg1, sg2: Two SimpleGate activations (for separate paths).
        conv3: 1x1 projection back to original channels (path 1).
        conv4: 1x1 expansion for FFN path.
        conv5: 1x1 projection for FFN path.
        gamma, beta: Learnable residual scaling factors.

    Example:
        ```python
        # Standard decoder block
        block = DarkIRDecoderBlock(
            channels=64,
            dw_expand=2,
            ffn_expand=2,
            dilations=[1, 4, 9]
        )
        x = ops.random.normal((2, 32, 32, 64))
        y = block(x)  # Output: (2, 32, 32, 64)

        # With extra depthwise for more capacity
        block = DarkIRDecoderBlock(
            channels=64,
            dilations=[1, 4, 9],
            extra_depth_wise=True
        )

        # Higher FFN expansion for refinement
        block = DarkIRDecoderBlock(
            channels=64,
            ffn_expand=4  # More capacity in FFN path
        )
        ```

    Note:
        - Both dw_expand and ffn_expand should result in even channels for SimpleGate
        - Learnable scales (beta, gamma) start at 0 for stable training
        - FFN path provides feature refinement without global frequency modeling
        - Order differs from encoder: conv1 before extra_conv (if used)
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        dilations: List[int] = None,
        extra_depth_wise: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if dw_expand <= 0:
            raise ValueError(f"dw_expand must be positive, got {dw_expand}")
        if ffn_expand <= 0:
            raise ValueError(f"ffn_expand must be positive, got {ffn_expand}")
        if dilations is None:
            dilations = [1]
        if not dilations:
            raise ValueError("dilations cannot be empty")
        if any(d <= 0 for d in dilations):
            raise ValueError(f"All dilations must be positive, got {dilations}")

        self.channels = channels
        self.dw_expand = dw_expand
        self.ffn_expand = ffn_expand
        self.dilations = dilations
        self.extra_depth_wise = extra_depth_wise
        self.dw_channels = channels * dw_expand
        self.ffn_channels = channels * ffn_expand

        # Create all sub-layers in __init__
        # Normalization layers
        self.norm1 = create_normalization_layer("layer_norm", axis=-1, epsilon=1e-6)
        self.norm2 = create_normalization_layer("layer_norm", axis=-1, epsilon=1e-6)

        # First projection
        self.conv1 = layers.Conv2D(
            self.dw_channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv1'
        )

        # Extra DW Conv (Optional) - NOTE: Applied AFTER conv1 in decoder
        if self.extra_depth_wise:
            self.extra_conv = layers.Conv2D(
                self.dw_channels,
                kernel_size=3,
                padding="same",
                groups=self.channels,  # Groups based on original channels
                use_bias=True,
                name='extra_dw_conv'
            )
        else:
            self.extra_conv = layers.Identity(name='identity_extra')

        # Parallel Dilated Branches
        # Note: In decoder, branches work with dw_channels and expansion=1
        self.branches = [
            DilatedBranch(self.dw_channels, expansion=1, dilation=d, name=f'branch_d{d}')
            for d in self.dilations
        ]

        # Channel Attention
        self.sca_avg = layers.GlobalAveragePooling2D(
            keepdims=True,
            name='channel_attn_pool'
        )
        self.sca_conv = layers.Conv2D(
            self.dw_channels // 2,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='channel_attn_conv'
        )

        # SimpleGate activations (two separate instances)
        self.sg1 = SimpleGate(name='simple_gate_1')
        self.sg2 = SimpleGate(name='simple_gate_2')

        # Projection back to original channels
        self.conv3 = layers.Conv2D(
            self.channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv3'
        )

        # FFN Path projections
        self.conv4 = layers.Conv2D(
            self.ffn_channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv4_ffn_expand'
        )
        self.conv5 = layers.Conv2D(
            self.channels,
            kernel_size=1,
            padding="valid",
            use_bias=True,
            name='conv5_ffn_project'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all sub-layers for proper serialization.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Build normalization layers
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # Build first path components
        self.conv1.build(input_shape)
        conv1_shape = self.conv1.compute_output_shape(input_shape)

        self.extra_conv.build(conv1_shape)
        extra_shape = self.extra_conv.compute_output_shape(conv1_shape)

        # Build all branches
        for branch in self.branches:
            branch.build(extra_shape)

        # After branches, shape should match extra_shape
        # SimpleGate halves the channels
        self.sg1.build(extra_shape)
        sg1_output_shape = self.sg1.compute_output_shape(extra_shape)

        # Build channel attention
        self.sca_avg.build(sg1_output_shape)
        sca_avg_shape = self.sca_avg.compute_output_shape(sg1_output_shape)
        self.sca_conv.build(sca_avg_shape)

        # Build projection back to original channels
        self.conv3.build(sg1_output_shape)

        # Build FFN path (operates on input_shape after norm2)
        self.conv4.build(input_shape)
        conv4_shape = self.conv4.compute_output_shape(input_shape)

        self.sg2.build(conv4_shape)
        sg2_output_shape = self.sg2.compute_output_shape(conv4_shape)

        self.conv5.build(sg2_output_shape)

        # Create learnable scale parameters
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, self.channels),
            initializer="zeros",
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, self.channels),
            initializer="zeros",
            trainable=True
        )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass with dual residual paths.

        Args:
            inputs: Input tensor of shape (batch, height, width, channels).
            training: Boolean, whether in training mode.

        Returns:
            Output tensor of shape (batch, height, width, channels).
        """
        # Store original input for residual
        y = inputs

        # === Path 1: Multi-scale Dilated Processing ===
        x = self.norm1(inputs)

        # Note: Order differs from encoder (conv1 before extra_conv)
        x = self.conv1(x)
        x = self.extra_conv(x)

        # Sum all parallel branches
        z = ops.add_n([branch(x) for branch in self.branches])

        # First SimpleGate activation
        z = self.sg1(z)

        # Channel Attention
        attn = self.sca_avg(z)
        attn = self.sca_conv(attn)
        x = attn * z

        # Project back to original channels
        x = self.conv3(x)

        # First residual with learnable scale
        y = inputs + self.beta * x

        # === Path 2: Gated FFN Processing ===
        x = self.norm2(y)
        x = self.conv4(x)

        # Second SimpleGate activation
        x = self.sg2(x)

        x = self.conv5(x)

        # Final residual with learnable scale
        out = y + x * self.gamma

        return out

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input).

        Args:
            input_shape: Shape tuple of input.

        Returns:
            Shape tuple of output (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "dw_expand": self.dw_expand,
            "ffn_expand": self.ffn_expand,
            "dilations": self.dilations,
            "extra_depth_wise": self.extra_depth_wise
        })
        return config


# ---------------------------------------------------------------------


def create_darkir_model(
    img_channels: int = 3,
    width: int = 32,
    middle_blk_num_enc: int = 2,
    middle_blk_num_dec: int = 2,
    enc_blk_nums: List[int] = None,
    dec_blk_nums: List[int] = None,
    dilations: List[int] = None,
    extra_depth_wise: bool = True,
    use_side_loss: bool = False
) -> keras.Model:
    """
    Create the DarkIR model for low-light image restoration.

    This function constructs a U-Net style architecture with:
    - Multiple encoder stages with downsampling
    - Middle bottleneck with encoder and decoder blocks
    - Multiple decoder stages with upsampling and skip connections
    - Global residual connection from input to output

    **Architecture Overview**:
    ```
    Input (H, W, 3)
         ↓
    Intro Conv (H, W, width)
         ↓
    ┌─────────────────────────────────────┐
    │ Encoder Stage 0                     │
    │  - enc_blk_nums[0] × EncoderBlock   │
    │  - Downsample (H/2, W/2, width*2)   │──→ skip_0
    └─────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────┐
    │ Encoder Stage 1                     │
    │  - enc_blk_nums[1] × EncoderBlock   │
    │  - Downsample (H/4, W/4, width*4)   │──→ skip_1
    └─────────────────────────────────────┘
         ↓
    ... (more encoder stages)
         ↓
    ┌─────────────────────────────────────┐
    │ Middle Section                      │
    │  - middle_blk_num_enc × EncoderBlk  │
    │  - middle_blk_num_dec × DecoderBlk  │
    └─────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────┐
    │ Decoder Stage 0                     │
    │  - Upsample (PixelShuffle)          │
    │  - Add skip_last                    │
    │  - dec_blk_nums[0] × DecoderBlock   │
    └─────────────────────────────────────┘
         ↓
    ... (more decoder stages)
         ↓
    Ending Conv (H, W, 3)
         ↓
    Add Input (Global Residual)
         ↓
    Output (H, W, 3)
    ```

    Args:
        img_channels: Integer, number of input/output image channels.
            Typically 3 for RGB images. Must be positive. Defaults to 3.
        width: Integer, base feature width (initial channel dimension).
            Doubled at each downsampling stage. Must be positive. Defaults to 32.
        middle_blk_num_enc: Integer, number of encoder blocks in middle section.
            Must be non-negative. Defaults to 2.
        middle_blk_num_dec: Integer, number of decoder blocks in middle section.
            Must be non-negative. Defaults to 2.
        enc_blk_nums: List of integers, number of blocks per encoder stage.
            Length determines number of encoder stages (downsampling operations).
            Defaults to [1, 2, 3]. All values must be positive.
        dec_blk_nums: List of integers, number of blocks per decoder stage.
            Must match length of enc_blk_nums. Defaults to [3, 1, 1].
            All values must be positive.
        dilations: List of integers, dilation rates for all blocks.
            Applied to all encoder and decoder blocks. Common: [1, 4, 9].
            Defaults to [1, 4, 9]. All values must be positive.
        extra_depth_wise: Boolean, whether to use extra depthwise convolution
            in all blocks. Adds inductive bias. Defaults to True.
        use_side_loss: Boolean, whether to return intermediate output for
            deep supervision. When True, returns [main_output, side_output].
            Defaults to False.

    Returns:
        keras.Model: The constructed DarkIR model.
            - If use_side_loss=False: Single output of shape (B, H, W, img_channels)
            - If use_side_loss=True: Two outputs [main, side] for multi-task learning

    Input shape:
        4D tensor with shape: `(batch, height, width, img_channels)`.
        Height and width should be multiples of 2^num_stages for clean downsampling.
        Values should be in range [0, 1] (normalized images).

    Output shape:
        4D tensor with shape: `(batch, height, width, img_channels)`.
        Restored image in same range as input [0, 1].

    Example:
        ```python
        # Small model for testing
        model = create_darkir_model(
            img_channels=3,
            width=16,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            dilations=[1]
        )

        # Medium model (paper default)
        model = create_darkir_model(
            img_channels=3,
            width=32,
            enc_blk_nums=[1, 2, 3],
            dec_blk_nums=[3, 1, 1],
            dilations=[1, 4, 9],
            extra_depth_wise=True
        )

        # Large model with deep supervision
        model = create_darkir_model(
            img_channels=3,
            width=48,
            enc_blk_nums=[2, 4, 6],
            dec_blk_nums=[6, 4, 2],
            dilations=[1, 4, 9],
            use_side_loss=True
        )

        # Test forward pass
        x = ops.random.normal((1, 256, 256, 3))
        y = model(x)
        print(y.shape)  # (1, 256, 256, 3)
        ```

    Note:
        - Total downsampling factor: 2^len(enc_blk_nums)
        - For enc_blk_nums=[1,2,3]: 8x downsampling (1/8 resolution at bottleneck)
        - Channel progression: width → 2*width → 4*width → ... → max_channels
        - Skip connections preserve spatial information during upsampling
        - Global residual enables learning only the correction to apply
    """
    # Set defaults
    if enc_blk_nums is None:
        enc_blk_nums = [1, 2, 3]
    if dec_blk_nums is None:
        dec_blk_nums = [3, 1, 1]
    if dilations is None:
        dilations = [1, 4, 9]

    # Validation
    if img_channels <= 0:
        raise ValueError(f"img_channels must be positive, got {img_channels}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if middle_blk_num_enc < 0:
        raise ValueError(f"middle_blk_num_enc must be non-negative, got {middle_blk_num_enc}")
    if middle_blk_num_dec < 0:
        raise ValueError(f"middle_blk_num_dec must be non-negative, got {middle_blk_num_dec}")
    if len(enc_blk_nums) != len(dec_blk_nums):
        raise ValueError(
            f"enc_blk_nums and dec_blk_nums must have same length, "
            f"got {len(enc_blk_nums)} and {len(dec_blk_nums)}"
        )
    if not enc_blk_nums or any(n <= 0 for n in enc_blk_nums):
        raise ValueError(f"All values in enc_blk_nums must be positive, got {enc_blk_nums}")
    if not dec_blk_nums or any(n <= 0 for n in dec_blk_nums):
        raise ValueError(f"All values in dec_blk_nums must be positive, got {dec_blk_nums}")
    if not dilations or any(d <= 0 for d in dilations):
        raise ValueError(f"All dilations must be positive, got {dilations}")

    # === Input ===
    inputs = keras.Input(shape=(None, None, img_channels), name="input_image")

    # === Intro Convolution ===
    x = layers.Conv2D(width, kernel_size=3, padding="same", name="intro")(inputs)

    # === Encoder Path ===
    skips = []
    chan = width

    for i, num_blocks in enumerate(enc_blk_nums):
        # Apply encoder blocks
        for j in range(num_blocks):
            x = DarkIREncoderBlock(
                channels=chan,
                dilations=dilations,
                extra_depth_wise=extra_depth_wise,
                name=f"enc_stage_{i}_block_{j}"
            )(x)

        # Save skip connection
        skips.append(x)

        # Downsample (stride 2 convolution)
        chan = chan * 2
        x = layers.Conv2D(
            chan,
            kernel_size=2,
            strides=2,
            padding="valid",
            name=f"down_{i}"
        )(x)

    # === Middle Section ===
    # Middle Encoder blocks
    for i in range(middle_blk_num_enc):
        x = DarkIREncoderBlock(
            channels=chan,
            dilations=dilations,
            extra_depth_wise=extra_depth_wise,
            name=f"mid_enc_{i}"
        )(x)

    # Store for optional side loss
    x_light = x

    # Middle Decoder blocks
    for i in range(middle_blk_num_dec):
        x = DarkIRDecoderBlock(
            channels=chan,
            dilations=dilations,
            extra_depth_wise=extra_depth_wise,
            name=f"mid_dec_{i}"
        )(x)

    # Residual connection in middle section
    x = layers.Add(name="middle_residual")([x, x_light])

    # === Decoder Path ===
    for i, num_blocks in enumerate(dec_blk_nums):
        # Upsample using PixelShuffle (DepthToSpace in Keras)
        # First expand channels by 4 (2x2 upsampling)
        x = layers.Conv2D(
            chan * 4,
            kernel_size=1,
            use_bias=False,
            name=f"up_conv_{i}"
        )(x)
        x = layers.DepthToSpace(block_size=2, name=f"pixel_shuffle_{i}")(x)

        # Halve channels (due to 2x spatial increase)
        chan = chan // 2

        # Add skip connection
        skip = skips.pop()
        x = layers.Add(name=f"skip_add_{i}")([x, skip])

        # Apply decoder blocks
        for j in range(num_blocks):
            x = DarkIRDecoderBlock(
                channels=chan,
                dilations=dilations,
                extra_depth_wise=extra_depth_wise,
                name=f"dec_stage_{i}_block_{j}"
            )(x)

    # === Ending Convolution ===
    x = layers.Conv2D(
        img_channels,
        kernel_size=3,
        padding="same",
        name="ending"
    )(x)

    # === Global Residual ===
    outputs = layers.Add(name="final_residual")([x, inputs])

    # === Optional Side Loss ===
    if use_side_loss:
        # Create a side output from the middle features
        side_out = layers.Conv2D(
            img_channels,
            kernel_size=3,
            padding="same",
            name="side_out"
        )(x_light)
        return keras.Model(
            inputs=inputs,
            outputs=[outputs, side_out],
            name="DarkIR"
        )

    return keras.Model(inputs=inputs, outputs=outputs, name="DarkIR")


# ---------------------------------------------------------------------