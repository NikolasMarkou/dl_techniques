import keras
from typing import Optional, Union, Tuple, Dict, Any
from keras import layers, initializers, regularizers, activations

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileOneBlock(keras.layers.Layer):
    """
    MobileOne building block with structural reparameterization.

    This layer implements the multi-branched architecture of MobileOne, which
    can be fused into a single, efficient convolutional layer at inference time.
    The design follows the structural reparameterization technique where multiple
    branches during training are mathematically equivalent to a single branch
    during inference.

    **Intent**: Provide an efficient convolutional building block that maintains
    high performance during training through multiple branches while achieving
    optimal inference speed through branch fusion.

    **Architecture (Inference after reparameterization)**:
    ```
    Input → Single Conv2D → Activation → Output
    ```

    **Mathematical Operations**:
    - **Training**: output = activation(Σ(branch_i(x)) + SE(x))
    - **Inference**: output = activation(SE(fused_conv(x)))

    Args:
        out_channels: Integer, number of output channels. Must be positive.
        kernel_size: Integer, size of the main convolution kernel. Must be positive.
        stride: Integer, stride of the convolution. Must be positive. Defaults to 1.
        padding: String, padding mode for convolutions. Either 'same' or 'valid'.
            Defaults to 'same'.
        use_se: Boolean, whether to include Squeeze-and-Excitation block.
            Defaults to False.
        num_conv_branches: Integer, number of Conv-BN branches. Must be positive.
            Defaults to 1.
        activation: String or callable, activation function to use.
            Defaults to 'gelu'.
        kernel_initializer: String or initializer, initializer for conv kernels.
            Defaults to 'he_normal'.
        bias_initializer: String or initializer, initializer for bias terms.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for conv kernels.
        bias_regularizer: Optional regularizer for bias terms.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, new_height, new_width, out_channels)`
        where new_height and new_width depend on stride and padding.

    Attributes:
        conv_branches: List of Conv-BN sequential blocks.
        scale_branch: Optional 1x1 Conv-BN branch.
        skip_branch: Optional skip connection with BatchNormalization.
        se_block: Optional Squeeze-and-Excitation block.
        inference_mode: Boolean indicating whether layer is in inference mode.

    Methods:
        reparameterize(): Fuse training branches into single convolution for inference.
        reset_reparameterization(): Switch back to training mode with multiple branches.

    Example:
        ```python
        # Basic usage
        block = MobileOneBlock(out_channels=64, kernel_size=3, stride=1)
        inputs = keras.Input(shape=(224, 224, 32))
        outputs = block(inputs)  # Shape: (None, 224, 224, 64)

        # With Squeeze-and-Excitation
        block = MobileOneBlock(
            out_channels=128,
            kernel_size=3,
            stride=2,
            use_se=True
        )

        # Multiple conv branches
        block = MobileOneBlock(
            out_channels=256,
            kernel_size=3,
            num_conv_branches=3,
            activation='relu'
        )

        # Reparameterize for inference
        block.reparameterize()  # Fuse branches
        inference_output = block(inputs)  # Uses single conv layer

        # Reset to training mode
        block.reset_reparameterization()
        training_output = block(inputs)  # Uses multiple branches
        ```

    References:
        MobileOne: An Improved One millisecond Mobile Backbone
        https://arxiv.org/abs/2206.04040
    """

    def __init__(
            self,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: str = 'same',
            use_se: bool = False,
            num_conv_branches: int = 1,
            activation: Union[str, callable] = 'gelu',
            kernel_initializer: Union[str, initializers.Initializer] = 'he_normal',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if num_conv_branches <= 0:
            raise ValueError(f"num_conv_branches must be positive, got {num_conv_branches}")
        if padding not in ['same', 'valid']:
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")

        # Store configuration
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # State management
        self.inference_mode = False

        # CREATE all sub-layers in __init__ (unbuilt)
        self.conv_branches = []
        for i in range(self.num_conv_branches):
            conv_branch = keras.Sequential([
                layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    use_bias=False,
                    groups=1,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'conv_branch_{i}_conv'
                ),
                layers.BatchNormalization(name=f'conv_branch_{i}_bn')
            ], name=f'conv_branch_{i}')
            self.conv_branches.append(conv_branch)

        # Scale branch (1x1 conv) if main kernel is larger
        if kernel_size > 1:
            self.scale_branch = keras.Sequential([
                layers.Conv2D(
                    filters=out_channels,
                    kernel_size=1,
                    strides=stride,
                    padding=padding,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name='scale_branch_conv'
                ),
                layers.BatchNormalization(name='scale_branch_bn')
            ], name='scale_branch')
        else:
            self.scale_branch = None

        # Skip branch (will be created in build if applicable)
        self.skip_branch = None

        # SE block if requested - reuse dl_techniques implementation
        if use_se:
            from dl_techniques.layers.squeeze_excitation import SqueezeExcitation
            self.se_block = SqueezeExcitation(
                reduction_ratio=0.25,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='se_block'
            )
        else:
            self.se_block = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create weights and build sub-layers."""
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Input channels dimension must be defined")

        # Create skip branch if input/output channels match and stride is 1
        if input_channels == self.out_channels and self.stride == 1:
            self.skip_branch = layers.BatchNormalization(name='skip_branch_bn')

        # Build all sub-layers explicitly
        for branch in self.conv_branches:
            branch.build(input_shape)

        if self.scale_branch is not None:
            self.scale_branch.build(input_shape)

        if self.skip_branch is not None:
            self.skip_branch.build(input_shape)

        if self.se_block is not None:
            # SE block needs output shape after conv
            conv_output_shape = self.conv_branches[0].compute_output_shape(input_shape)
            self.se_block.build(conv_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the block."""
        x = None

        # Conv branches
        for branch in self.conv_branches:
            branch_out = branch(inputs, training=training)
            x = branch_out if x is None else x + branch_out

        # Scale branch
        if self.scale_branch is not None:
            x = x + self.scale_branch(inputs, training=training)

        # Skip branch
        if self.skip_branch is not None:
            x = x + self.skip_branch(inputs, training=training)

        # Apply activation
        x = self.activation(x)

        # Apply SE block if present
        if self.se_block is not None:
            x = self.se_block(x, training=training)

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        if self.conv_branches:
            return self.conv_branches[0].compute_output_shape(input_shape)

        # Fallback calculation
        if self.padding == 'same':
            height = (input_shape[1] + self.stride - 1) // self.stride if input_shape[1] is not None else None
            width = (input_shape[2] + self.stride - 1) // self.stride if input_shape[2] is not None else None
        else:  # valid padding
            height = (input_shape[1] - self.kernel_size + self.stride) // self.stride if input_shape[
                                                                                             1] is not None else None
            width = (input_shape[2] - self.kernel_size + self.stride) // self.stride if input_shape[
                                                                                            2] is not None else None

        return (input_shape[0], height, width, self.out_channels)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'use_se': self.use_se,
            'num_conv_branches': self.num_conv_branches,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
