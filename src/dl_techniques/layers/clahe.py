import keras
import tensorflow as tf
from typing import Dict, Any, Optional, Union, Tuple

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CLAHE(keras.layers.Layer):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) layer.

    This layer enhances local contrast in single-channel images. It divides the
    image into small, non-overlapping tiles and applies histogram equalization
    to each tile. To prevent noise amplification, the contrast is limited by
    clipping the histogram at a predefined value before computing the
    cumulative distribution function (CDF).

    **Intent**: Provide a production-ready and serializable Keras layer for
    image preprocessing and enhancement, particularly useful for medical imaging,
    satellite imagery, and other domains where local contrast is critical.

    **Architecture & Stages**:
    ```
    Input Image [H, W, 1]
           ↓
    1. Divide into non-overlapping tiles of `tile_size` x `tile_size`.
           ↓
    2. For each tile:
       a. Compute histogram with `n_bins`.
       b. Clip histogram based on `clip_limit`.
       c. Redistribute clipped values across all bins.
       d. Compute Cumulative Distribution Function (CDF).
       e. Normalize CDF to [0, 255].
       f. Apply a learnable sigmoid-gated mapping.
       g. Map tile pixel values using the final CDF.
           ↓
    3. Stitch processed tiles back together.
           ↓
    Output Enhanced Image [H, W, 1]
    ```

    **Mathematical Operations**:
    For each tile:
    1.  **Histogram Clipping**: `clip_val = clip_limit * mean(hist)`
        `hist_clipped = min(hist, clip_val)`
    2.  **Redistribution**: `excess = sum(hist - hist_clipped)`
        `hist_final = hist_clipped + excess / n_bins`
    3.  **CDF Mapping**: `cdf = cumsum(hist_final)`
        `cdf_norm = (cdf - min(cdf)) * 255 / (max(cdf) - min(cdf))`
    4.  **Trainable Modulation**: `cdf_mapped = cdf_norm * sigmoid(mapping_kernel)`
    5.  **Output**: `output_pixel = cdf_mapped[input_pixel]`

    Args:
        clip_limit (float): The contrast limit for histogram clipping. Higher
            values result in more contrast. Must be positive. Defaults to 4.0.
        n_bins (int): The number of bins to use for the histogram.
            Must be positive. Defaults to 256.
        tile_size (int): The size of the square tiles for local histogram
            equalization. Must be positive. Defaults to 16.
        kernel_initializer (Union[str, keras.initializers.Initializer]):
            Initializer for the `mapping_kernel` weights. Defaults to "glorot_uniform".
        kernel_regularizer (Optional[keras.regularizers.Regularizer]):
            Regularizer function applied to the `mapping_kernel`. Defaults to None.
        kernel_constraint (Optional[keras.constraints.Constraint]):
            Constraint function applied to the `mapping_kernel`. Defaults to None.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.

    Input shape:
        3D tensor with shape: `(height, width, 1)`.
        Input values are expected to be in the range [0, 255].

    Output shape:
        3D tensor with the same shape as the input.
        Output values will be in the range [0, 255].

    Attributes:
        mapping_kernel (keras.Variable): Trainable weight vector of shape
            `(n_bins,)` used to modulate the final CDF mapping, allowing the
            enhancement effect to be fine-tuned during model training.

    Example:
        ```python
        # Standalone usage
        clahe_layer = CLAHE(clip_limit=3.0, tile_size=8)
        # Create a sample grayscale image tensor
        image = tf.random.uniform(shape=(256, 256, 1), maxval=256, dtype=tf.int32)
        image = tf.cast(image, tf.float32)
        enhanced_image = clahe_layer(image)

        # In a Keras model
        inputs = keras.Input(shape=(128, 128, 1))
        x = CLAHE(clip_limit=2.0, name='clahe_enhancement')(inputs)
        # ... subsequent layers ...
        outputs = keras.layers.Conv2D(3, 3, padding='same')(x)
        model = keras.Model(inputs, outputs)
        model.summary()
        ```
    """

    def __init__(
        self,
        clip_limit: float = 4.0,
        n_bins: int = 256,
        tile_size: int = 16,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        kernel_constraint: Optional[keras.constraints.Constraint] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be positive, got {clip_limit}")
        if n_bins <= 1:
            raise ValueError(f"n_bins must be greater than 1, got {n_bins}")
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")

        # Store configuration
        self.clip_limit = float(clip_limit)
        self.n_bins = int(n_bins)
        self.tile_size = int(tile_size)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)

        # Initialize weight attribute
        self.mapping_kernel = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's trainable weights."""
        if len(input_shape) != 3 or input_shape[-1] != 1:
            raise ValueError(
                "Expected input shape (height, width, 1), but "
                f"received {input_shape}"
            )

        self.mapping_kernel = self.add_weight(
            name="mapping_kernel",
            shape=(self.n_bins,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        super().build(input_shape)

    def _process_tile(self, tile: tf.Tensor) -> tf.Tensor:
        """Process a single image tile using CLAHE logic."""
        # Note: tf.histogram_fixed_width is used as there is no Keras Ops equivalent.
        hist = tf.histogram_fixed_width(
            tile, value_range=[0.0, 255.0], nbins=self.n_bins
        )
        hist = keras.ops.cast(hist, self.compute_dtype)

        # Apply contrast limiting
        clip_val = self.clip_limit * keras.ops.mean(hist)
        hist_clipped = keras.ops.minimum(hist, clip_val)

        # Redistribute clipped pixels
        excess = keras.ops.sum(hist - hist_clipped)
        hist_redistributed = hist_clipped + (excess / float(self.n_bins))

        # Compute CDF and normalize
        cdf = keras.ops.cumsum(hist_redistributed)
        cdf_min = cdf[0] # The first value of cumsum is the minimum
        denominator = keras.ops.maximum(cdf[-1] - cdf_min, 1e-7)
        cdf_normalized = (cdf - cdf_min) * 255.0 / denominator

        # Apply trainable mapping
        cdf_mapped = cdf_normalized * keras.ops.sigmoid(self.mapping_kernel)

        # Map input pixel values to the new CDF
        indices = keras.ops.cast(keras.ops.round(tile), "int32")
        return keras.ops.take(cdf_mapped, indices)

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply CLAHE to the input tensor.

        Note: This implementation uses a tf.function-decorated loop for tiling,
        which is specific to the TensorFlow backend's AutoGraph feature. This
        is necessary to handle dynamic input shapes in graph mode.
        """
        x = keras.ops.cast(inputs, self.compute_dtype)
        shape = keras.ops.shape(x)
        height, width = shape[0], shape[1]

        num_tiles_h = keras.ops.cast(keras.ops.ceil(height / self.tile_size), "int32")
        num_tiles_w = keras.ops.cast(keras.ops.ceil(width / self.tile_size), "int32")

        # Process image tile by tile
        processed_rows = []
        for i in range(num_tiles_h):
            processed_cols = []
            for j in range(num_tiles_w):
                start_h = i * self.tile_size
                start_w = j * self.tile_size
                end_h = keras.ops.minimum(start_h + self.tile_size, height)
                end_w = keras.ops.minimum(start_w + self.tile_size, width)

                tile = x[start_h:end_h, start_w:end_w, 0]
                processed_tile = self._process_tile(tile)
                processed_cols.append(processed_tile)

            # Stitch tiles in a row
            processed_rows.append(keras.ops.concatenate(processed_cols, axis=1))

        # Stitch all rows together
        result = keras.ops.concatenate(processed_rows, axis=0)
        # Ensure final output has the original dimensions
        result = result[:height, :width]
        return keras.ops.reshape(result, (height, width, 1))

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            "clip_limit": self.clip_limit,
            "n_bins": self.n_bins,
            "tile_size": self.tile_size,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
        })
        return config

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """The output shape is the same as the input shape."""
        return input_shape

# ---------------------------------------------------------------------

