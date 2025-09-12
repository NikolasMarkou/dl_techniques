import keras
import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Canny(keras.layers.Layer):
    """Keras implementation of the Canny edge detection algorithm.

    This layer performs Canny edge detection, a multi-stage algorithm used to
    detect a wide range of edges in images. The process includes Gaussian
    smoothing, gradient calculation, non-maximum suppression, double
    thresholding, and edge tracking by hysteresis.

    **Intent**: Provide a robust, serializable, and modern Keras 3 implementation
    of the Canny algorithm that can be integrated into neural network models or
    used as a standalone image processing layer.

    **Architecture & Stages**:
    ```
    Input Image [B, H, W, 1]
           ↓
    1. Noise Reduction (Gaussian Filter)
           ↓
    2. Gradient Calculation (Sobel Operator)
           ↓ (Magnitude & Angle)
    3. Non-Maximum Suppression
           ↓
    4. Double Thresholding (Strong & Weak Edges)
           ↓
    5. Hysteresis Edge Tracking
           ↓
    Output Edge Map [B, H, W, 1]
    ```

    **Mathematical Operations**:
    1. **Smoothing**: `I_smooth = I * G(σ)` where `G` is a Gaussian kernel.
    2. **Gradients**: `Gx = I_smooth * S_x`, `Gy = I_smooth * S_y`.
       Magnitude `G = sqrt(Gx² + Gy²)`, Angle `θ = atan2(Gy, Gx)`.
    3. **Suppression**: Discard pixels that are not local maxima in the
       direction of the gradient.
    4. **Thresholding**: Classify edges as strong (`> threshold_max`),
       weak (`> threshold_min`), or non-edges.
    5. **Hysteresis**: Connect weak edges to strong edges iteratively.

    Args:
        sigma (float): Standard deviation for the Gaussian kernel. A larger
            sigma corresponds to more blurring and detection of larger-scale
            edges. Must be >= 0.8. Defaults to 0.8.
        threshold_min (int): The lower threshold for the double thresholding
            stage. Pixels with gradient magnitude below this are discarded.
            Defaults to 50.
        threshold_max (int): The upper threshold for the double thresholding
            stage. Pixels above this are considered strong edges.
            Defaults to 80.
        tracking_connection (int): The connectivity size (kernel size) for the final
            hysteresis edge tracking stage. Determines the neighborhood for
            connecting weak edges to strong ones. Defaults to 5.
        tracking_iterations (int): The maximum number of iterations for the
            hysteresis tracking. Prevents infinite loops in edge connection.
            Defaults to 3.
        **kwargs: Additional arguments for the `keras.layers.Layer` base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, 1)`.
        The input image should be a single-channel (grayscale) tensor.

    Output shape:
        4D tensor with the same shape as the input:
        `(batch_size, height, width, 1)`.
        The output is a binary edge map where `1.0` represents an edge and
        `0.0` represents the background.

    Attributes:
        gaussian_kernel (keras.Variable): Non-trainable weight for Gaussian smoothing.
        sobel_kernel (keras.Variable): Non-trainable weight for Sobel gradient calculation.
        angle_kernel (keras.Variable): Non-trainable weights for non-maximum suppression.
        dilation_kernel (keras.Variable): Non-trainable weight for hysteresis tracking.

    Example:
        ```python
        # Create a Canny edge detection layer
        canny_layer = Canny(sigma=1.0, threshold_min=40, threshold_max=90)

        # Apply to an input image
        # Note: Input tensor should be float type
        input_image = keras.random.uniform(shape=(1, 256, 256, 1)) * 255.0
        edge_map = canny_layer(input_image)

        # Use within a Keras model
        inputs = keras.Input(shape=(256, 256, 1))
        outputs = Canny()(inputs)
        model = keras.Model(inputs, outputs)
        model.summary()
        ```

    Raises:
        ValueError: If `sigma` is less than 0.8 or if `threshold_min` is
            not less than `threshold_max`.
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
        """Creates a 2D Gaussian kernel for image smoothing."""
        kernel_size = int(((((self.sigma - 0.8) / 0.3) + 1) * 2) + 1)
        kernel_size += 1 if (kernel_size % 2) == 0 else 0

        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)

        normal = 1 / (2.0 * np.pi * (self.sigma ** 2))
        kernel = np.exp(-((xx ** 2) + (yy ** 2)) / (2.0 * (self.sigma ** 2))) * normal
        kernel = kernel / np.sum(kernel)

        return kernel.reshape((kernel_size, kernel_size, 1, 1))

    def _build_sobel_kernel(self) -> np.ndarray:
        """Creates Sobel kernels for Gx and Gy gradient computation."""
        # Gx kernel: detects vertical edges
        gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # Gy kernel: detects horizontal edges
        gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # Stack to shape (H, W, in_channels, out_channels) -> (3, 3, 1, 2)
        return np.stack([gx_kernel, gy_kernel], axis=-1).reshape((3, 3, 1, 2))

    def _build_angle_kernels(self) -> np.ndarray:
        """Creates kernels for angle-specific non-maximum suppression."""
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
        """Create the layer's non-trainable weights (kernels)."""
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
        """Perform Canny edge detection on the input image tensor."""
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
        """Compute angle-specific edge responses for suppression."""
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
        """Apply double thresholding to find strong and weak edges."""
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
        """Track edges using hysteresis with a tf.while_loop."""

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
        """Return the configuration of the layer for serialization."""
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
        """The output shape is the same as the input shape."""
        return input_shape

# ---------------------------------------------------------------------
