"""``CLAHE``, a trainable Contrast Limited Adaptive Histogram Equalization layer
for single-channel images.

Standard histogram equalization remaps pixel intensities through one global
cumulative distribution function (CDF), which over-amplifies contrast in some
regions of an image while washing out others. CLAHE instead equalizes each
tile of the image separately and clips each tile's histogram at `clip_limit *
mean(hist)` before building its CDF, redistributing the clipped mass evenly
across the other bins; this bounds the CDF's slope and keeps histogram
equalization from amplifying noise in near-uniform tiles. This layer adds a
learnable sigmoid gate on top of the normalized CDF, `cdf_mapped = cdf_norm *
sigmoid(mapping_kernel)`, so the enhancement strength per intensity level can
be tuned end-to-end with the rest of the model.

This layer runs only on the TensorFlow backend and only eagerly: tile
histograms use `tf.histogram_fixed_width`, which has no `keras.ops`
equivalent, and the tile count depends on the runtime image size, which a
graph `while_loop` cannot trace cleanly. Apply it outside a `@tf.function`,
or per-image with known shapes.

References:
    - Pizer et al., 1987. Adaptive Histogram Equalization and Its Variations.
      (https://doi.org/10.1016/0734-189X(87)90186-X)
"""

import keras
import tensorflow as tf
from typing import Dict, Any, Optional, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.clahe")
class CLAHE(keras.layers.Layer):
    """Trainable Contrast Limited Adaptive Histogram Equalization layer.

    Enhances local contrast in single-channel images by dividing the image into
    non-overlapping tiles and applying histogram equalization per tile. The
    histogram is clipped at ``clip_limit * mean(hist)`` before computing the CDF
    to prevent noise amplification. A learnable sigmoid-gated modulation
    ``cdf_mapped = cdf_norm * sigmoid(mapping_kernel)`` allows end-to-end
    fine-tuning of the enhancement for a downstream task.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input Image (H, W, 1)           │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Divide into tiles (tile_size^2) │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Per-tile processing:            │
        │  histogram ──► clip ──► redist   │
        │  ──► CDF ──► normalize           │
        │  ──► sigmoid(kernel) modulate    │
        │  ──► remap pixel intensities     │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Stitch tiles back together      │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Output Enhanced Image (H,W,1)   │
        └──────────────────────────────────┘

    :param clip_limit: Contrast limit for histogram clipping. Must be positive.
        Defaults to 4.0.
    :type clip_limit: float
    :param n_bins: Number of histogram bins. Must be > 1. Defaults to 256.
    :type n_bins: int
    :param tile_size: Size of square tiles for local equalization. Must be
        positive. Defaults to 16.
    :type tile_size: int
    :param kernel_initializer: Initializer for the ``mapping_kernel`` weights.
        Defaults to ``"glorot_uniform"``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the ``mapping_kernel``.
        Defaults to ``None``.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kernel_constraint: Constraint for the ``mapping_kernel``.
        Defaults to ``None``.
    :type kernel_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional arguments for the ``keras.layers.Layer`` base class.
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
        """Create the layer's trainable weights.

        :param input_shape: Shape tuple ``(height, width, 1)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
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
        """Process a single image tile using CLAHE logic.

        :param tile: 2D tile tensor.
        :type tile: tf.Tensor
        :return: Enhanced tile tensor.
        :rtype: tf.Tensor
        """
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply CLAHE to the input tensor.

        Executes eagerly on the TensorFlow backend: the number of tiles is
        data-dependent (a function of the runtime ``H``/``W``), so the nested
        Python tile loop is run eagerly rather than traced into a graph
        ``while_loop`` (graph tracing of the loop + ``tf.histogram_fixed_width``
        is not supported — see the module-level accepted-exception note).

        :param inputs: Input image tensor of shape ``(H, W, 1)``.
        :type inputs: tf.Tensor
        :return: Enhanced image tensor of same shape.
        :rtype: tf.Tensor
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
        """Return the configuration of the layer for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
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
        """Compute the output shape (same as input).

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output tensor shape (identical to input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

# ---------------------------------------------------------------------
