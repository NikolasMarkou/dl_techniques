"""Canny edge detection, implemented as a Keras layer with fixed
(non-trainable) convolution kernels.

The layer runs the classical five-stage algorithm — Gaussian smoothing,
Sobel gradients, non-maximum suppression, double thresholding, and
hysteresis tracking — as ordinary convolutions and a bounded morphological
loop, so it composes into a Keras graph while contributing no trainable
capacity. Its output is a hard threshold of the gradient field, so it passes
no useful gradient back to its input.

The forward path uses only `keras.ops` and is backend-agnostic. `keras.ops`
has no generic grayscale-dilation primitive, but both dilation stages use
fixed, known structuring elements rather than general ones: the suppression
stage's per-angle line footprint is expressed as a max over three shifted,
``-inf``-padded slices, and the hysteresis stage's flat square footprint is
an ordinary ``keras.ops.max_pool``.

References:
    - Canny, 1986. A Computational Approach to Edge Detection. IEEE TPAMI 8(6).
      (https://doi.org/10.1109/TPAMI.1986.4767851)
    - Sobel and Feldman, 1968. A 3x3 Isotropic Gradient Operator for Image
      Processing. Stanford Artificial Intelligence Project.
    - Serra, 1982. Image Analysis and Mathematical Morphology. Academic Press.
"""

import keras
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.signal_processing.canny")
class Canny(keras.layers.Layer):
    """Multi-stage Canny edge detection layer for single-channel images.

    Applies the classical Canny algorithm as a Keras layer:
    Gaussian smoothing ``I_smooth = I * G(sigma)``, Sobel gradient computation
    ``G = sqrt(Gx^2 + Gy^2)``, non-maximum suppression along the gradient
    direction, double thresholding into strong/weak edges, and iterative
    hysteresis tracking to connect weak edges adjacent to strong ones.
    All convolution kernels are stored as non-trainable weights.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────┐
        │  Input (B, H, W, 1)          │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  1. Gaussian Smoothing       │
        │     I_smooth = I * G(sigma)  │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  2. Sobel Gradient           │
        │     Gx, Gy ──► Mag, Angle    │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  3. Non-Maximum Suppression  │
        │     angle-specific dilation  │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  4. Double Thresholding      │
        │     strong / weak / discard  │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  5. Hysteresis Tracking      │
        │     dilate strong ──► weak   │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Output Edge Map (B,H,W,1)   │
        └──────────────────────────────┘

    :param sigma: Standard deviation for the Gaussian kernel. Must be >= 0.8.
        Defaults to 0.8.
    :type sigma: float
    :param threshold_min: Lower threshold for double thresholding, on the
        un-normalized Sobel gradient magnitude of an image on a 0-255 scale
        (a unit-range image needs thresholds scaled down accordingly).
        Defaults to 50.
    :type threshold_min: int
    :param threshold_max: Upper threshold for double thresholding, same scale
        as ``threshold_min``. Defaults to 80.
    :type threshold_max: int
    :param tracking_connection: Side of the square window used to grow strong
        edges into adjacent weak ones during hysteresis tracking. Must be >= 1;
        use an odd value for a symmetric window. Defaults to 5.
    :type tracking_connection: int
    :param tracking_iterations: Maximum number of hysteresis iterations. Must
        be >= 1. Defaults to 3.
    :type tracking_iterations: int
    :param kwargs: Additional arguments for the ``keras.layers.Layer`` base class.
    """

    def __init__(
            self,
            sigma: float = 0.8,
            threshold_min: int = 50,
            threshold_max: int = 80,
            tracking_connection: int = 5,
            tracking_iterations: int = 3,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if sigma < 0.8:
            raise ValueError(
                "Minimum kernel size needs to be 3, which requires sigma >= 0.8. "
                f"Received sigma={sigma}."
            )
        if threshold_min >= threshold_max:
            raise ValueError(
                f"threshold_min ({threshold_min}) must be less than "
                f"threshold_max ({threshold_max})."
            )
        if tracking_connection < 1:
            raise ValueError(
                f"tracking_connection must be >= 1, got {tracking_connection}."
            )
        if tracking_iterations < 1:
            raise ValueError(
                f"tracking_iterations must be >= 1, got {tracking_iterations}."
            )

        # Store all configuration parameters
        self.sigma = sigma
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.tracking_connection = tracking_connection
        self.tracking_iterations = tracking_iterations

        # Static data for angle calculations, not a weight.
        self.angle_ranges = [
            [157.5, 22.5], [22.5, 67.5], [67.5, 112.5], [112.5, 157.5]
        ]

        # Initialize weight attributes - created in build()
        self.gaussian_kernel = None
        self.sobel_kernel = None

    def _build_gaussian_kernel(self) -> np.ndarray:
        """Create a 2D Gaussian kernel for image smoothing.

        :return: Gaussian kernel array of shape ``(K, K, 1, 1)``.
        :rtype: np.ndarray
        """
        kernel_size = int(((((self.sigma - 0.8) / 0.3) + 1) * 2) + 1)
        kernel_size += 1 if (kernel_size % 2) == 0 else 0

        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)

        normal = 1 / (2.0 * np.pi * (self.sigma ** 2))
        kernel = np.exp(-((xx ** 2) + (yy ** 2)) / (2.0 * (self.sigma ** 2))) * normal
        kernel = kernel / np.sum(kernel)

        return kernel.reshape((kernel_size, kernel_size, 1, 1))

    def _build_sobel_kernel(self) -> np.ndarray:
        """Create Sobel kernels for Gx and Gy gradient computation.

        :return: Stacked Sobel kernels of shape ``(3, 3, 1, 2)``.
        :rtype: np.ndarray
        """
        # Gx kernel: detects vertical edges
        gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # Gy kernel: detects horizontal edges
        gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # Stack to shape (H, W, in_channels, out_channels) -> (3, 3, 1, 2)
        return np.stack([gx_kernel, gy_kernel], axis=-1).reshape((3, 3, 1, 2))

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's non-trainable weights (kernels).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        gaussian_val = self._build_gaussian_kernel()
        self.gaussian_kernel = self.add_weight(
            name="gaussian_kernel", shape=gaussian_val.shape,
            initializer=keras.initializers.Constant(gaussian_val),
            trainable=False,
        )

        sobel_val = self._build_sobel_kernel()
        self.sobel_kernel = self.add_weight(
            name="sobel_kernel", shape=sobel_val.shape,
            initializer=keras.initializers.Constant(sobel_val),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Perform Canny edge detection on the input image tensor.

        :param inputs: Grayscale image tensor of shape ``(B, H, W, 1)``.
        :type inputs: keras.KerasTensor
        :return: Binary edge map of same shape as input.
        :rtype: keras.KerasTensor
        """
        original_dtype = inputs.dtype

        # Stage 1: Noise reduction
        x_smooth = self._conv_symmetric(inputs, self.gaussian_kernel)

        # Stage 2: Gradient calculation
        grad_xy = self._conv_symmetric(x_smooth, self.sobel_kernel)
        grad_x, grad_y = keras.ops.split(grad_xy, 2, axis=-1)

        # Magnitude in float32: a float16 square overflows past |g| = 255.
        # It is deliberately NOT clipped: a clip at 255 turns every strong
        # edge's ridge into a flat plateau, and non-maximum suppression keeps
        # every pixel of a plateau, so edges would come out several pixels wide.
        grad_x = keras.ops.cast(grad_x, "float32")
        grad_y = keras.ops.cast(grad_y, "float32")
        theta = (keras.ops.arctan2(grad_y, grad_x) * (180 / np.pi) + 90) % 180
        grad_mag = keras.ops.sqrt(grad_x ** 2 + grad_y ** 2)

        # Stage 3: Non-maximum suppression
        angle_responses = self._compute_angle_responses(theta, grad_mag)
        # The neighbours are compared by their gradient MAGNITUDE, whatever
        # orientation bin they fall in: reading them from ``angle_responses``
        # would give 0 for a neighbour in another bin, and a much stronger
        # neighbour across the edge could then never suppress the centre.
        max_pool_angle = self._directional_max(
            keras.ops.tile(grad_mag, [1, 1, 1, 4])
        )

        # Stage 4: Double thresholding
        strong_edges, weak_edges = self._apply_double_threshold(
            max_pool_angle, angle_responses, grad_mag
        )

        # Stage 5: Edge tracking by hysteresis
        final_edges = self._track_edges(strong_edges, weak_edges)

        return keras.ops.cast(final_edges, dtype=original_dtype)

    @staticmethod
    def _conv_symmetric(
            inputs: keras.KerasTensor, kernel: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Convolve with ``'same'`` output size and mirrored borders.

        Zero padding would turn every bright image border into a step edge
        against the padding, so the border is mirrored instead.

        :param inputs: Tensor of shape ``(B, H, W, 1)``.
        :type inputs: keras.KerasTensor
        :param kernel: Odd-sized kernel of shape ``(K, K, 1, C_out)``.
        :type kernel: keras.KerasTensor
        :return: Convolved tensor of shape ``(B, H, W, C_out)``.
        :rtype: keras.KerasTensor
        """
        pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
        padded = keras.ops.pad(
            inputs, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]],
            mode="symmetric"
        )
        return keras.ops.conv(padded, kernel, padding="valid")

    def _compute_angle_responses(
            self, theta: keras.KerasTensor, grad_mag: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute angle-specific edge responses for suppression.

        :param theta: Gradient angle tensor.
        :type theta: keras.KerasTensor
        :param grad_mag: Gradient magnitude tensor.
        :type grad_mag: keras.KerasTensor
        :return: Angle-weighted edge response tensor.
        :rtype: keras.KerasTensor
        """
        angle_masks = []
        low, high = self.angle_ranges[0]
        mask = keras.ops.logical_or(
            keras.ops.greater_equal(theta, low),
            keras.ops.less_equal(theta, high)
        )
        angle_masks.append(mask)

        for low, high in self.angle_ranges[1:]:
            mask = keras.ops.logical_and(
                keras.ops.greater_equal(theta, low),
                keras.ops.less(theta, high)
            )
            angle_masks.append(mask)

        stacked_masks = keras.ops.cast(
            keras.ops.concatenate(angle_masks, axis=-1), dtype=grad_mag.dtype
        )
        return stacked_masks * grad_mag

    @staticmethod
    def _directional_max(angle_responses: keras.KerasTensor) -> keras.KerasTensor:
        """Per-angle directional dilation over a fixed 3-pixel line footprint.

        Channel ``c`` holds responses for edges of orientation ``c`` and is
        maxed over the 3 pixels lying on the line through the center that runs
        *across* that edge, i.e. along the gradient direction, which is where
        non-maximum suppression must compare neighbours: a 0° (horizontal)
        edge is compared vertically, 90° horizontally, 45° along the main
        diagonal and 135° along the anti-diagonal. Comparing along the edge
        instead would keep every pixel of a ridge flank and thin nothing.
        ``-inf`` padding means out-of-bounds taps never win a max.

        :param angle_responses: Tensor of shape ``(B, H, W, 4)`` whose channel
            order is ``[0°, 45°, 90°, 135°]`` of the edge orientation. ``call``
            passes the gradient magnitude tiled over the four channels.
        :type angle_responses: keras.KerasTensor
        :return: Directional max tensor of the same shape.
        :rtype: keras.KerasTensor
        """
        padded = keras.ops.pad(
            angle_responses, [[0, 0], [1, 1], [1, 1], [0, 0]],
            constant_values=-np.inf
        )

        c0, c45, c90, c135 = (padded[..., i:i + 1] for i in range(4))

        # 0 degree edge: gradient is vertical (top, mid, bottom row)
        out0 = keras.ops.maximum(
            keras.ops.maximum(c0[:, :-2, 1:-1, :], c0[:, 1:-1, 1:-1, :]),
            c0[:, 2:, 1:-1, :]
        )
        # 45 degree edge (anti-diagonal): gradient runs along the main
        # diagonal (top-left, mid, bottom-right)
        out45 = keras.ops.maximum(
            keras.ops.maximum(c45[:, :-2, :-2, :], c45[:, 1:-1, 1:-1, :]),
            c45[:, 2:, 2:, :]
        )
        # 90 degree edge: gradient is horizontal (left, mid, right column)
        out90 = keras.ops.maximum(
            keras.ops.maximum(c90[:, 1:-1, :-2, :], c90[:, 1:-1, 1:-1, :]),
            c90[:, 1:-1, 2:, :]
        )
        # 135 degree edge (main diagonal): gradient runs along the
        # anti-diagonal (top-right, mid, bottom-left)
        out135 = keras.ops.maximum(
            keras.ops.maximum(c135[:, :-2, 2:, :], c135[:, 1:-1, 1:-1, :]),
            c135[:, 2:, :-2, :]
        )

        return keras.ops.concatenate([out0, out45, out90, out135], axis=-1)

    def _apply_double_threshold(
            self, max_pool_angle: keras.KerasTensor, angle_responses: keras.KerasTensor,
            grad_mag: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Apply double thresholding to find strong and weak edges.

        :param max_pool_angle: Max-pooled angle response tensor.
        :type max_pool_angle: keras.KerasTensor
        :param angle_responses: Angle response tensor.
        :type angle_responses: keras.KerasTensor
        :param grad_mag: Gradient magnitude tensor.
        :type grad_mag: keras.KerasTensor
        :return: Tuple of ``(strong_edges, weak_edges)`` binary masks.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Only a pixel's own-angle channel is non-zero in ``angle_responses``.
        # ``max_pool_angle`` holds the largest magnitude on each channel's
        # across-edge line, so a pixel survives when its own magnitude is that
        # maximum. The ``> 0`` term stops the other, all-zero channels passing
        # on ``0 >= 0``, which would suppress nothing.
        suppressed = keras.ops.where(
            keras.ops.logical_and(
                keras.ops.greater_equal(angle_responses, max_pool_angle),
                keras.ops.greater(angle_responses, 0.0)
            ),
            angle_responses, 0.0
        )
        edge_candidates = keras.ops.expand_dims(
            keras.ops.max(suppressed, axis=-1), axis=-1
        )

        strong = keras.ops.cast(
            keras.ops.greater_equal(edge_candidates, self.threshold_max),
            self.compute_dtype
        )
        weak = keras.ops.cast(
            keras.ops.logical_and(
                keras.ops.greater_equal(edge_candidates, self.threshold_min),
                keras.ops.less(edge_candidates, self.threshold_max)
            ), self.compute_dtype
        )
        return strong, weak

    def _track_edges(
            self, strong_edges: keras.KerasTensor, weak_edges: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Track edges using hysteresis with a keras.ops.while_loop.

        The dilation over the (all-ones) ``tracking_connection`` structuring
        element is mathematically identical to a flat max pool of the same
        window, so it is expressed directly as ``keras.ops.max_pool``.

        :param strong_edges: Binary mask of strong edges.
        :type strong_edges: keras.KerasTensor
        :param weak_edges: Binary mask of weak edges.
        :type weak_edges: keras.KerasTensor
        :return: Final binary edge map.
        :rtype: keras.KerasTensor
        """
        pool_size = (self.tracking_connection, self.tracking_connection)

        def loop_cond(current, has_changed):
            return has_changed

        def loop_body(current, has_changed):
            previous = current
            dilated = keras.ops.max_pool(
                current, pool_size=pool_size, strides=(1, 1), padding="same"
            )
            newly_strong = dilated * weak_edges
            current = keras.ops.clip(strong_edges + newly_strong, 0.0, 1.0)
            has_changed = keras.ops.any(keras.ops.not_equal(current, previous))
            return current, has_changed

        final_edges, _ = keras.ops.while_loop(
            loop_cond, loop_body,
            loop_vars=(strong_edges, keras.ops.convert_to_tensor(True)),
            maximum_iterations=self.tracking_iterations
        )
        return final_edges

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "sigma": self.sigma,
            # ``int()`` would truncate a fractional threshold, so the round trip
            # could reorder or collapse the pair; keep it whole when integral.
            "threshold_min": (
                int(self.threshold_min)
                if self.threshold_min.is_integer() else self.threshold_min
            ),
            "threshold_max": (
                int(self.threshold_max)
                if self.threshold_max.is_integer() else self.threshold_max
            ),
            "tracking_connection": self.tracking_connection,
            "tracking_iterations": self.tracking_iterations,
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
