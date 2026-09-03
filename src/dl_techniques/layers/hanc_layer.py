"""
Hierarchical Aggregation of Neighborhood Context (HANC) layer, built by the
``HANCLayer`` class.

Self-attention gives every pixel a global view of the feature map but costs
``O(N^2)`` in the number of spatial positions. This layer approximates that
global view with average- and max-pooling instead: at each of ``k-1`` scales
it pools the input, upsamples the pooled map back to full resolution, and
concatenates it onto the original feature map along the channel axis. A
final 1x1 convolution fuses the original features with these multi-scale
summaries. Cost is ``O(N*k)``, linear in the number of scales rather than
quadratic in spatial size.

References:
    - Yan et al., 2023. ACC-UNet: An adaptive context and contrast-aware
      UNet for seismic facies identification.
    - Zhao et al., 2017. Pyramid Scene Parsing Network.
    - Vaswani et al., 2017. Attention Is All You Need.
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any, List, Dict
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.hanc_layer")
class HANCLayer(keras.layers.Layer):
    """Hierarchical Aggregation of Neighborhood Context (HANC) Layer.

    This layer approximates global self-attention by aggregating statistical
    summaries (mean and max) from local neighborhoods at multiple scales.
    It combines these multi-scale context features with the original input
    to create a rich, context-aware representation with linear complexity
    ``O(k)``. For scales ``s in {1, ..., k-1}``, the layer computes
    ``C_avg^(s) = Up(AvgPool_{2^s}(X))``,
    ``C_max^(s) = Up(MaxPool_{2^s}(X))``, concatenates them with the
    original input to form
    ``X_concat = [X, C_avg^(1), C_max^(1), ..., C_avg^(k-1), C_max^(k-1)]``,
    and projects through ``Y = sigma(BN(W * X_concat))``.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │        Input [H, W, C]           │
        └──────┬───────┬──────┬────────────┘
               │       │      │
               │       │      ▼
               │       │   ┌────────────────────────┐
               │       │   │ For each scale s=1..k-1│
               │       │   │  ├─ AvgPool(2^s)       │
               │       │   │  │   → Resize(H, W)    │
               │       │   │  └─ MaxPool(2^s)       │
               │       │   │      → Resize(H, W)    │
               │       │   └──────────┬─────────────┘
               │       │              │
               ▼       ▼              ▼
        ┌──────────────────────────────────┐
        │  Concatenate(axis=-1)            │
        │  [C * (2k - 1) channels]         │
        └───────────────┬──────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────┐
        │  Conv1x1(out_channels) → BN      │
        │  → LeakyReLU                     │
        └───────────────┬──────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────┐
        │     Output [H, W, out_channels]  │
        └──────────────────────────────────┘

    :param in_channels: Number of input channels. Must be positive.
    :type in_channels: int
    :param out_channels: Number of output channels after projection. Must be positive.
    :type out_channels: int
    :param k: Number of hierarchical levels (1-5). k=1: identity only (no pooling),
        k=2: adds 2x2 pooling context, k=3: adds 2x2 and 4x4 pooling context.
    :type k: int
    :param kernel_initializer: Initializer for the 1x1 convolution kernel.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the convolution kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.

    :raises ValueError: If in_channels or out_channels are not positive.
    :raises ValueError: If k is not between 1 and 5.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            k: int = 3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if k < 1 or k > 5:
            raise ValueError(f"k must be between 1 and 5, got {k}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Original channels plus (avg + max) for each of the k-1 scales.
        self.total_concat_channels = in_channels * (1 + 2 * (k - 1))

        self.avg_pooling_layers: List[keras.layers.Layer] = []
        self.max_pooling_layers: List[keras.layers.Layer] = []

        for scale in range(1, self.k):
            pool_size = 2 ** scale

            self.avg_pooling_layers.append(
                keras.layers.AveragePooling2D(
                    pool_size=pool_size,
                    strides=pool_size,
                    padding='same',
                    name=f'avg_pool_{pool_size}x{pool_size}'
                )
            )

            self.max_pooling_layers.append(
                keras.layers.MaxPooling2D(
                    pool_size=pool_size,
                    strides=pool_size,
                    padding='same',
                    name=f'max_pool_{pool_size}x{pool_size}'
                )
            )

        self.concatenate = keras.layers.Concatenate(axis=-1, name='concat_features')

        self.conv = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='fusion_conv'
        )

        self.batch_norm = keras.layers.BatchNormalization(name='fusion_bn')
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers in computational order.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D: {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: expected {self.in_channels}, "
                f"got {input_shape[-1]}"
            )

        for avg_pool, max_pool in zip(self.avg_pooling_layers, self.max_pooling_layers):
            avg_pool.build(input_shape)
            max_pool.build(input_shape)

        concat_shape = tuple(input_shape[:-1]) + (self.total_concat_channels,)
        self.conv.build(concat_shape)
        conv_output_shape = self.conv.compute_output_shape(concat_shape)
        self.batch_norm.build(conv_output_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation.

        :param inputs: Input tensor of shape ``(batch, height, width, in_channels)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode for batch normalization.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch, height, width, out_channels)``.
        :rtype: keras.KerasTensor
        """
        if self.k == 1:
            # k=1 has no pooling scales; still route through conv/bn/activation
            # below so in_channels != out_channels still projects correctly.
            concatenated = inputs
        else:
            features_list = [inputs]
            input_shape = ops.shape(inputs)
            height, width = input_shape[1], input_shape[2]

            for avg_pool, max_pool in zip(self.avg_pooling_layers, self.max_pooling_layers):
                avg_feat = avg_pool(inputs)
                avg_resized = ops.image.resize(
                    avg_feat,
                    size=(height, width),
                    interpolation='nearest'
                )
                features_list.append(avg_resized)

                max_feat = max_pool(inputs)
                max_resized = ops.image.resize(
                    max_feat,
                    size=(height, width),
                    interpolation='nearest'
                )
                features_list.append(max_resized)

            concatenated = self.concatenate(features_list)

        x = self.conv(concatenated)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {len(input_shape)}D")

        return tuple(list(input_shape[:-1]) + [self.out_channels])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'k': self.k,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config