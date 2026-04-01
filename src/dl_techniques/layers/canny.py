import keras
import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Canny(keras.layers.Layer):
    """Multi-stage Canny edge detection layer for single-channel images.

    Applies the classical Canny algorithm as a differentiable Keras layer:
    Gaussian smoothing ``I_smooth = I * G(sigma)``, Sobel gradient computation
    ``G = sqrt(Gx^2 + Gy^2)``, non-maximum suppression along the gradient
    direction, double thresholding into strong/weak edges, and iterative
    hysteresis tracking to connect weak edges adjacent to strong ones.
    All convolution kernels are stored as non-trainable weights.

    **Architecture Overview:**

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
        │     Gx, Gy ──► Mag, Angle   │
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
    :param threshold_min: Lower threshold for double thresholding.
        Defaults to 50.
    :type threshold_min: int
    :param threshold_max: Upper threshold for double thresholding.
        Defaults to 80.
    :type threshold_max: int
    :param tracking_connection: Connectivity kernel size for hysteresis
        edge tracking. Defaults to 5.
    :type tracking_connection: int
    :param tracking_iterations: Maximum number of hysteresis iterations.
        Defaults to 3.
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
        self.angle_kernel = None
        self.dilation_kernel = None

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

    def _build_angle_kernels(self) -> np.ndarray:
        """Create kernels for angle-specific non-maximum suppression.

        :return: Concatenated angle kernels of shape ``(3, 3, 4)``.
        :rtype: np.ndarray
        """
        inf = np.inf
        # Kernels for 0°, 45°, 90°, 135° detection
        k0 = np.array([[[-inf], [-inf], [-inf]], [[0.0], [0.0], [0.0]], [[-inf], [-inf], [-inf]]])
        k45 = np.array([[[-inf], [-inf], [0.0]], [[-inf], [0.0], [-inf]], [[0.0], [-inf], [-inf]]])
        k90 = np.array([[[-inf], [0.0], [-inf]], [[-inf], [0.0], [-inf]], [[-inf], [0.0], [-inf]]])
        k135 = np.array([[[0.0], [-inf], [-inf]], [[-inf], [0.0], [-inf]], [[-inf], [-inf], [0.0]]])
        # Concatenate on the last axis to create a (3, 3, 4) filter
        # for tf.nn.dilation2d, where depth matches input channels.
        return np.concatenate([k0, k45, k90, k135], axis=-1).astype(np.float32)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's non-trainable weights (kernels).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # 1. Gaussian Kernel
        gaussian_val = self._build_gaussian_kernel()
        self.gaussian_kernel = self.add_weight(
            name="gaussian_kernel", shape=gaussian_val.shape,
            initializer=keras.initializers.Constant(gaussian_val),
            trainable=False,
        )

        # 2. Sobel Kernel
        sobel_val = self._build_sobel_kernel()
        self.sobel_kernel = self.add_weight(
            name="sobel_kernel", shape=sobel_val.shape,
            initializer=keras.initializers.Constant(sobel_val),
            trainable=False,
        )

        # 3. Angle Kernels for NMS
        angle_val = self._build_angle_kernels()
        self.angle_kernel = self.add_weight(
            name="angle_kernel", shape=angle_val.shape,
            initializer=keras.initializers.Constant(angle_val),
            trainable=False,
        )

        # 4. Dilation Kernel for Hysteresis
        self.dilation_kernel = self.add_weight(
            name="dilation_kernel", shape=(self.tracking_connection, self.tracking_connection, 1),
            initializer=keras.initializers.Ones(), trainable=False,
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
        x_smooth = keras.ops.conv(inputs, self.gaussian_kernel, padding="same")

        # Stage 2: Gradient calculation
        grad_xy = keras.ops.conv(x_smooth, self.sobel_kernel, padding="same")
        grad_x, grad_y = keras.ops.split(grad_xy, 2, axis=-1)

        theta = (keras.ops.arctan2(grad_y, grad_x) * (180 / np.pi) + 90) % 180
        grad_mag = keras.ops.clip(
            keras.ops.sqrt(grad_x ** 2 + grad_y ** 2), 0.0, 255.0
        )

        # Stage 3: Non-maximum suppression
        angle_responses = self._compute_angle_responses(theta, grad_mag)
        max_pool_angle = tf.nn.dilation2d(
            angle_responses, self.angle_kernel, strides=(1, 1, 1, 1),
            padding='SAME', data_format='NHWC', dilations=(1, 1, 1, 1)
        )

        # Stage 4: Double thresholding
        strong_edges, weak_edges = self._apply_double_threshold(
            max_pool_angle, angle_responses, grad_mag
        )

        # Stage 5: Edge tracking by hysteresis
        final_edges = self._track_edges(strong_edges, weak_edges)

        return keras.ops.cast(final_edges, dtype=original_dtype)

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

    def _apply_double_threshold(
            self, max_pool_angle: tf.Tensor, angle_responses: tf.Tensor, grad_mag: tf.Tensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Apply double thresholding to find strong and weak edges.

        :param max_pool_angle: Max-pooled angle response tensor.
        :type max_pool_angle: tf.Tensor
        :param angle_responses: Angle response tensor.
        :type angle_responses: tf.Tensor
        :param grad_mag: Gradient magnitude tensor.
        :type grad_mag: tf.Tensor
        :return: Tuple of ``(strong_edges, weak_edges)`` binary masks.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        suppressed = keras.ops.where(
            keras.ops.equal(max_pool_angle, angle_responses), grad_mag, 0.0
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
            self, strong_edges: tf.Tensor, weak_edges: tf.Tensor
    ) -> keras.KerasTensor:
        """Track edges using hysteresis with a tf.while_loop.

        :param strong_edges: Binary mask of strong edges.
        :type strong_edges: tf.Tensor
        :param weak_edges: Binary mask of weak edges.
        :type weak_edges: tf.Tensor
        :return: Final binary edge map.
        :rtype: keras.KerasTensor
        """

        def loop_cond(current, has_changed):
            return has_changed

        def loop_body(current, has_changed):
            previous = tf.identity(current)
            dilated = tf.nn.dilation2d(
                current, self.dilation_kernel, strides=(1, 1, 1, 1),
                padding='SAME', data_format='NHWC', dilations=(1, 1, 1, 1)
            )
            newly_strong = dilated * weak_edges
            current = keras.ops.clip(strong_edges + newly_strong, 0.0, 1.0)
            has_changed = keras.ops.any(keras.ops.not_equal(current, previous))
            return current, has_changed

        final_edges, _ = tf.while_loop(
            loop_cond, loop_body, loop_vars=(strong_edges, tf.constant(True)),
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
            "threshold_min": int(self.threshold_min),
            "threshold_max": int(self.threshold_max),
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
