import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Any

# ---------------------------------------------------------------------

# Constants for mathematical operations
math_ops = tf.math
PI = tf.cast(math_ops.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32)


@dataclass(eq=False, order=False, frozen=True)
class _Node:
    """Internal node structure for kernel management.

    Attributes:
        name: Identifier for the kernel
        kernel: The actual kernel tensor
        carry: Additional data (primarily used for angle ranges in edge detection)
    """
    __slots__ = ('name', 'kernel', 'carry')
    name: str
    kernel: tf.Tensor
    carry: Any

    def __repr__(self) -> str:
        return self.name


class kernels:
    """Manages the kernels required for Canny edge detection.

    This class handles the creation and cycling of various kernels used in the
    edge detection process, including Gaussian smoothing, Sobel operators,
    and specialized angle detection kernels.
    """

    def __init__(self, sigma: float = 0.8, con: int = 5):
        """
        Initialize kernel manager with given parameters.

        Args:
            sigma: Standard deviation for Gaussian kernel (must be >= 0.8)
            con: Connection size for dilation kernel

        Raises:
            ValueError: If sigma < 0.8
        """
        self.items: List[_Node] = []
        if sigma < 0.8:
            raise ValueError('minimum kernel size need to be size of 3 --> sigma > 0.8')
        self.sigma = sigma
        self.con = con
        self.build()

    def __repr__(self) -> str:
        return str(self.items)

    def __next__(self) -> _Node:
        """Cycles through kernels in a rotating fashion."""
        tmp = self.items.pop()
        self.items.insert(0, tmp)
        return tmp

    def build(self):
        """Constructs all required kernels for edge detection."""
        # Calculate Gaussian kernel size
        kernel_size = int(((((self.sigma - 0.8) / 0.3) + 1) * 2) + 1)
        kernel_size += 1 if (kernel_size % 2) == 0 else 0

        # Build Gaussian kernel
        gaussian_kernel = self._build_gaussian_kernel(kernel_size)

        # Build Sobel kernel for gradient detection
        sobel_kernel = self._build_sobel_kernel()

        # Build directional kernels for angle detection
        ang_kernel = self._build_angle_kernels()

        # Build dilation kernel for edge tracking
        dilation_kernel = tf.ones(shape=(self.con, self.con, 1), dtype=tf.float32)

        # Store all kernels with their metadata
        self.items = [
            _Node('gaussian_kernel', gaussian_kernel, None),
            _Node('sobel_kernel', sobel_kernel, None),
            _Node('ang_kernel', ang_kernel,
                  [[157.5, 22.5], [22.5, 67.5], [67.5, 112.5], [112.5, 157.5]]),
            _Node('dilation_kernel', dilation_kernel, None)
        ]
        self.items.reverse()

    def _build_gaussian_kernel(self, kernel_size: int) -> tf.Tensor:
        """Creates 2D Gaussian kernel for image smoothing."""
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        normal = 1 / (2.0 * PI * (self.sigma ** 2))
        kernel = tf.exp(-((xx ** 2) + (yy ** 2)) / (2.0 * (self.sigma ** 2))) * normal
        kernel = kernel / tf.reduce_sum(kernel)
        return tf.reshape(kernel, shape=(kernel_size, kernel_size, 1, 1))

    def _build_sobel_kernel(self) -> tf.Tensor:
        """Creates Sobel kernels for gradient computation."""
        return tf.constant([
            [[[-1, -1]], [[0, -2]], [[1, -1]]],
            [[[-2, 0]], [[0, 0]], [[2, 0]]],
            [[[-1, 1]], [[0, 2]], [[1, 1]]]
        ], dtype=tf.float32)

    def _build_angle_kernels(self) -> tf.Tensor:
        """Creates kernels for angle-specific edge detection."""
        inf = float('inf')
        kernels = [
            # 0° detection
            tf.constant([
                [[-inf], [-inf], [-inf]],
                [[0.0], [0.0], [0.0]],
                [[-inf], [-inf], [-inf]]
            ], dtype=tf.float32),
            # 45° detection
            tf.constant([
                [[-inf], [-inf], [0.0]],
                [[-inf], [0.0], [-inf]],
                [[0.0], [-inf], [-inf]]
            ], dtype=tf.float32),
            # 90° detection
            tf.constant([
                [[-inf], [0.0], [-inf]],
                [[-inf], [0.0], [-inf]],
                [[-inf], [0.0], [-inf]]
            ], dtype=tf.float32),
            # 135° detection
            tf.constant([
                [[0.0], [-inf], [-inf]],
                [[-inf], [0.0], [-inf]],
                [[-inf], [-inf], [0.0]]
            ], dtype=tf.float32)
        ]
        return tf.concat(kernels, axis=-1)

    @staticmethod
    def pad(X: tf.Tensor,
            b: Optional[Union[int, List[int]]] = None,
            h: Optional[Union[int, List[int]]] = None,
            w: Optional[Union[int, List[int]]] = None,
            d: Optional[Union[int, List[int]]] = None,
            **kwargs) -> tf.Tensor:
        """
        Pads the input tensor along specified dimensions.

        Args:
            X: Input tensor of rank 4
            b: Padding for batch dimension
            h: Padding for height dimension
            w: Padding for width dimension
            d: Padding for depth dimension
            **kwargs: Additional arguments for tf.pad

        Returns:
            Padded tensor
        """
        assert len(X.get_shape()) == 4
        if not any([b, h, w, d]):
            return X

        paddings = []
        for arg in [b, h, w, d]:
            if arg is None:
                paddings.append([0, 0])
            else:
                arg = [arg, arg] if isinstance(arg, int) else list(arg)
                paddings.append(arg)

        return tf.pad(X, tf.constant(paddings, dtype=tf.int32), **kwargs)


class Canny(tf.Module):
    """TensorFlow implementation of the Canny edge detection algorithm."""

    def __init__(self,
                 sigma: float = 0.8,
                 threshold_min: int = 50,
                 threshold_max: int = 80,
                 tracking_con: int = 5,
                 tracking_iterations: int = 3,
                 **kwargs):
        """
        Initialize Canny edge detector.

        Args:
            sigma: Standard deviation for Gaussian smoothing
            threshold_min: Lower threshold for edge detection
            threshold_max: Upper threshold for edge detection
            tracking_con: Connection size for edge tracking
            tracking_iterations: Maximum iterations for hysteresis tracking
        """
        super().__init__()
        self.kernels = kernels(sigma, tracking_con)
        self.threshold = (threshold_min, threshold_max)
        self.tracking_iter = tracking_iterations

    @tf.function(autograph=False,
                 reduce_retracing=True,
                 input_signature=[
                     tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32)
                 ])
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        Perform Canny edge detection on input image.

        Args:
            X: Input image tensor [batch, height, width, 1]

        Returns:
            Binary edge map tensor of same shape as input
        """
        d_type = X.dtype
        kernels_ = self.kernels

        # Stage 1: Noise reduction with Gaussian filtering
        with tf.name_scope('noise_reduction'):
            gaussian_kernel = next(kernels_).kernel
            Xg = tf.nn.convolution(X, gaussian_kernel, padding='SAME')

        # Stage 2: Gradient calculation using Sobel operators
        with tf.name_scope('gradient'):
            sobel_kernel = next(kernels_).kernel
            Gxy = tf.nn.convolution(Xg, sobel_kernel, padding='SAME')
            gx, gy = tf.split(Gxy, [1, 1], axis=-1)
            theta = ((math_ops.atan2(gx, gy) * 180 / PI) + 90) % 180
            Gxy = tf.clip_by_value(
                math_ops.sqrt((gx ** 2) + (gy ** 2)),
                0, 255.
            )

        # Stage 3: Non-maximum suppression
        with tf.name_scope('non_maximum_suppression'):
            angle_kernel = next(kernels_)
            angle_X = self._compute_angle_responses(theta, Gxy, angle_kernel)
            max_pool_ang = tf.nn.dilation2d(
                kernels_.pad(angle_X, h=1, w=1, constant_values=0.0),
                angle_kernel.kernel,
                strides=(1, 1, 1, 1),
                padding='VALID',
                data_format='NHWC',
                dilations=(1, 1, 1, 1)
            )

        # Stage 4: Double thresholding
        with tf.name_scope('double_thresholding'):
            edge_candidates = self._apply_double_threshold(
                max_pool_ang, angle_X, Gxy
            )

        # Stage 5: Edge tracking by hysteresis
        with tf.name_scope('dilation_tracking'):
            final_edges = self._track_edges(edge_candidates, kernels_)

        return tf.cast(final_edges, dtype=d_type)

    def _compute_angle_responses(self,
                                 theta: tf.Tensor,
                                 Gxy: tf.Tensor,
                                 angle_kernel: _Node) -> tf.Tensor:
        """Compute angle-specific edge responses."""
        angle_responses = []

        # Handle 0° case specially (wrapping around 180°)
        low, high = angle_kernel.carry[0]
        tmp = math_ops.logical_or(
            math_ops.greater_equal(theta, low),
            math_ops.less_equal(theta, high)
        )
        angle_responses.append(tmp)

        # Handle other angles
        for low, high in angle_kernel.carry[1:]:
            tmp = math_ops.logical_and(
                math_ops.greater_equal(theta, low),
                math_ops.less(theta, high)
            )
            angle_responses.append(tmp)

        return tf.cast(tf.concat(angle_responses, -1), tf.float32) * Gxy

    def _apply_double_threshold(self,
                                max_pool_ang: tf.Tensor,
                                angle_X: tf.Tensor,
                                Gxy: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply double thresholding to identify strong and weak edges."""
        threshold_min, threshold_max = self.threshold

        # Find initial edge candidates
        edge_ = tf.where(
            math_ops.logical_and(
                math_ops.equal(max_pool_ang, angle_X),
                max_pool_ang > threshold_min
            ),
            Gxy, 0.0
        )
        edge_ = tf.expand_dims(tf.reduce_max(edge_, axis=-1), -1)

        # Separate into strong and weak edges
        edge_sure = tf.where(edge_ >= threshold_max, 1.0, 0.0)
        edge_weak = tf.where(
            math_ops.logical_and(
                edge_ >= threshold_min,
                edge_ < threshold_max
            ),
            1.0, 0.0
        )

        return edge_sure, edge_weak

    def _track_edges(self,
                     edge_candidates: Tuple[tf.Tensor, tf.Tensor],
                     kernels_: kernels) -> tf.Tensor:
        """Track edges using hysteresis."""
        edge_sure, edge_weak = edge_candidates
        hysteresis_kernel = next(kernels_).kernel

        def check(curr: tf.Tensor, cond: tf.Tensor) -> tf.Tensor:
            return cond

        def main_(curr: tf.Tensor, cond: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            prev = tf.identity(curr)
            dilation = tf.nn.dilation2d(
                curr,
                hysteresis_kernel,
                strides=(1, 1, 1, 1),
                padding='SAME',
                data_format='NHWC',
                dilations=(1, 1, 1, 1)
            )
            curr = (dilation * edge_weak) + edge_sure - 1
            return curr, math_ops.reduce_max(curr - prev) != 0

        # Iteratively track edges
        edge, _ = tf.while_loop(
            check, main_,
            loop_vars=(edge_sure, True),
            maximum_iterations=self.tracking_iter
        )

        return tf.where(edge + edge_sure > 0, 1.0, 0.0)