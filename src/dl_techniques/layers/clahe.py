import tensorflow as tf
from keras.api.layers import Layer
from typing import Dict, Any, Optional, Union, Tuple
from keras import initializers, regularizers, constraints

# ---------------------------------------------------------------------


class CLAHE(Layer):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) layer for image enhancement.

    This layer implements CLAHE for single-channel images of arbitrary size. It processes
    the image in tiles and applies histogram equalization with contrast limiting to reduce noise
    amplification while improving local contrast.

    Args:
        clip_limit: float, optional
            The contrast limit for histogram clipping. Higher values give more contrast.
            Defaults to 4.0.
        n_bins: int, optional
            Number of histogram bins. Defaults to 256.
        tile_size: int, optional
            Size of the local region for histogram equalization. Defaults to 16.
        kernel_initializer: Union[str, initializers.Initializer], optional
            Initializer for kernels. Defaults to "glorot_uniform".
        kernel_regularizer: Optional[regularizers.Regularizer], optional
            Regularizer for kernels. Defaults to None.
        kernel_constraint: Optional[constraints.Constraint], optional
            Constraint for kernels. Defaults to None.
        **kwargs: Any
            Additional layer arguments.

    Input shape:
        3D tensor with shape: `(height, width, 1)`
        Input values should be in range [0, 255]

    Output shape:
        Same as input shape
        Output values will be in range [0, 255]
    """

    def __init__(
            self,
            clip_limit: float = 4.0,
            n_bins: int = 256,
            tile_size: int = 16,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            kernel_constraint: Optional[constraints.Constraint] = None,
            **kwargs: Any
    ) -> None:
        """Initialize CLAHE layer."""
        super().__init__(**kwargs)

        # Validate inputs
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {clip_limit}")
        if n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {n_bins}")
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")

        self.clip_limit = float(clip_limit)
        self.n_bins = int(n_bins)
        self.tile_size = int(tile_size)

        # Initialize kernel-related attributes
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer.

        Args:
            input_shape: Tuple of input shape dimensions.

        Raises:
            ValueError: If input shape is invalid.
        """
        if len(input_shape) != 3 or input_shape[-1] != 1:
            raise ValueError(
                f"Expected input shape (height, width, 1), got {input_shape}"
            )

        # Create trainable mapping kernel for fine-tuning
        self.mapping_kernel = self.add_weight(
            name="mapping_kernel",
            shape=(self.n_bins,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )

        super().build(input_shape)

    def _process_tile(self, tile: tf.Tensor) -> tf.Tensor:
        """Process a single image tile using CLAHE.

        Args:
            tile: Input tensor tile of shape (height, width).

        Returns:
            Processed tile tensor with enhanced contrast.
        """
        # Compute histogram
        hist = tf.histogram_fixed_width(
            tile,
            value_range=[0.0, 255.0],
            nbins=self.n_bins
        )
        hist = tf.cast(hist, tf.float32)

        # Apply contrast limiting
        clip_limit = self.clip_limit * tf.reduce_mean(hist)
        hist_clipped = tf.minimum(hist, clip_limit)

        # Redistribute clipped pixels
        excess = tf.reduce_sum(hist - hist_clipped)
        redistribution_value = excess / tf.cast(self.n_bins, tf.float32)
        hist_redistributed = hist_clipped + redistribution_value

        # Compute CDF and normalize
        cdf = tf.cumsum(hist_redistributed)
        cdf_min = tf.reduce_min(cdf)
        denominator = tf.maximum(tf.reduce_max(cdf) - cdf_min, 1e-7)
        cdf_normalized = (cdf - cdf_min) * 255.0 / denominator

        # Apply trainable mapping
        cdf_mapped = cdf_normalized * tf.nn.sigmoid(self.mapping_kernel)

        # Map input values
        indices = tf.cast(tile, tf.int32)
        return tf.gather(cdf_mapped, indices)

    @tf.function
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Apply CLAHE to input tensor.

        Args:
            inputs: Input tensor of shape (height, width, 1).
            training: Boolean indicating if the layer should behave in training mode.
                Currently not used.

        Returns:
            Enhanced tensor of same shape with values in [0, 255].
        """
        x = tf.cast(inputs, tf.float32)
        shape = tf.shape(x)
        height, width = shape[0], shape[1]

        # Calculate tiles
        num_tiles_h = tf.cast(tf.math.ceil(height / self.tile_size), tf.int32)
        num_tiles_w = tf.cast(tf.math.ceil(width / self.tile_size), tf.int32)

        # Process tiles
        tiles = []
        for i in range(num_tiles_h):
            row_tiles = []
            for j in range(num_tiles_w):
                start_h = i * self.tile_size
                start_w = j * self.tile_size
                end_h = tf.minimum(start_h + self.tile_size, height)
                end_w = tf.minimum(start_w + self.tile_size, width)

                tile = x[start_h:end_h, start_w:end_w, 0]
                processed = self._process_tile(tile)

                # Handle edge tiles
                if end_h - start_h < self.tile_size or end_w - start_w < self.tile_size:
                    paddings = [
                        [0, self.tile_size - (end_h - start_h)],
                        [0, self.tile_size - (end_w - start_w)]
                    ]
                    processed = tf.pad(processed, paddings, "SYMMETRIC")

                row_tiles.append(processed)

            # Combine tiles
            row = tf.concat(row_tiles, axis=1)
            row = row[:tf.minimum(self.tile_size, height - i * self.tile_size), :width]
            tiles.append(row)

        # Final combination
        result = tf.concat(tiles, axis=0)
        return tf.reshape(result, [height, width, 1])

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'clip_limit': self.clip_limit,
            'n_bins': self.n_bins,
            'tile_size': self.tile_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CLAHE':
        """Create layer from configuration.

        Args:
            config: Layer configuration dictionary.

        Returns:
            CLAHE layer instance.
        """
        return cls(**config)

# ---------------------------------------------------------------------
