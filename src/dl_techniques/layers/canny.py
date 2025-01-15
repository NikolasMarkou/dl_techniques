import tensorflow as tf
from dataclasses import dataclass

# ---------------------------------------------------------------------


math_ops = tf.math
PI = tf.cast(math_ops.angle(tf.constant(-1, dtype=tf.complex64)), tf.float32)


# ---------------------------------------------------------------------


class Canny(tf.Module):
    def __init__(self,
                 sigma: float = 0.8,
                 threshold_min: int = 50,
                 threshold_max: int = 80,
                 tracking_con: int = 5,
                 tracking_iterations: int = 3,
                 **kwargs):
        super().__init__()
        self.kernels = self.kernels(sigma, tracking_con)
        self.threshold = (threshold_min, threshold_max)
        self.tracking_iter = tracking_iterations

    @tf.function(autograph=False,
                 reduce_retracing=True,
                 input_signature=[
                     tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32)
                 ])
    def __call__(self, X):
        d_type = X.dtype

        kernels_ = self.kernels
        with tf.name_scope('noise_reduction'):
            gaussian_kernel = next(kernels_).kernel
            Xg = tf.nn.convolution(X, gaussian_kernel, padding='SAME', name='Xg')

        with tf.name_scope('gradient'):
            sobel_kernel = next(kernels_).kernel
            Gxy = tf.nn.convolution(Xg, sobel_kernel, padding='SAME', name='Gxy')
            gx, gy = tf.split(Gxy, [1, 1], axis=-1)
            theta = ((math_ops.atan2(gx, gy, name='theta') * 180 / PI) + 90) % 180
            Gxy = math_ops.sqrt((gx ** 2) + (gy ** 2), name='Gxy')
            Gxy = tf.clip_by_value(Gxy, 0, 255.)

        with tf.name_scope('non_maximum_suppression'):
            angle_kernel = next(kernels_)
            angle_X = []
            low, high = angle_kernel.carry[0]
            tmp = math_ops.logical_or(math_ops.greater_equal(theta, low), math_ops.less_equal(theta, high))
            angle_X.append(tmp)

            for low, high in angle_kernel.carry[1:]:
                tmp = math_ops.logical_and(math_ops.greater_equal(theta, low), math_ops.less(theta, high))
                angle_X.append(tmp)

            angle_X = tf.cast(tf.concat(angle_X, -1), tf.float32) * Gxy

            max_pool_ang = tf.nn.dilation2d(
                kernels_.pad(angle_X, h=1, w=1, constant_values=0.0),
                angle_kernel.kernel, (1, 1, 1, 1), 'VALID', 'NHWC', (1, 1, 1, 1)
            )

        with tf.name_scope('double_thresholding'):
            threshold_min, threshold_max = self.threshold
            edge_ = tf.where(
                math_ops.logical_and(math_ops.equal(max_pool_ang, angle_X), max_pool_ang > threshold_min), Gxy, 0.0
            )
            edge_ = tf.expand_dims(tf.reduce_max(edge_, axis=-1), -1)
            edge_sure = tf.where(edge_ >= threshold_max, 1.0, 0.0, name='edge_sure')
            edge_week = tf.where(
                math_ops.logical_and(edge_ >= threshold_min, edge_ < threshold_max), 1.0, 0.0, name='edge_week'
            )

        with tf.name_scope('dilation_tracking'):
            hysteresis_kernel = next(kernels_).kernel

            def check(curr, cond):
                return cond

            def main_(curr, cond):
                prev = tf.identity(curr, name='prev_connected')
                dilation = tf.nn.dilation2d(
                    curr, hysteresis_kernel, (1, 1, 1, 1), 'SAME', 'NHWC', (1, 1, 1, 1)
                )
                curr = (dilation * edge_week) + edge_sure - 1
                return curr, math_ops.reduce_max(curr - prev) != 0

            edge, _ = tf.while_loop(check, main_, loop_vars=(edge_sure, True), maximum_iterations=self.tracking_iter)
            edge = tf.where(edge + edge_sure > 0, 1.0, 0)

        return tf.cast(edge, dtype=d_type, name='edge')

    class kernels:
        def __init__(self, sigma: float = 0.8, con: int = 5):
            self.items = []
            if sigma < 0.8: raise ValueError('minimum kernel size need to be size of 3 --> sigma > 0.8')
            self.sigma = sigma
            self.con = con
            self.build()

        def __repr__(self):
            return str(self.items)

        def __next__(self):
            tmp = self.items.pop()
            self.items.insert(0, tmp)
            return tmp

        def build(self):
            sigma = self.sigma
            kernel_size = int(((((sigma - 0.8) / 0.3) + 1) * 2) + 1)
            kernel_size = kernel_size + 1 if (kernel_size % 2) == 0 else kernel_size

            ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
            xx, yy = tf.meshgrid(ax, ax)
            normal = 1 / (2.0 * PI * (sigma ** 2))
            kernel = tf.exp(-((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2))) * normal

            kernel = kernel / tf.reduce_sum(kernel)
            gaussian_kernel = tf.reshape(kernel, shape=(kernel_size, kernel_size, 1, 1), name='gaussian_kernel')

            sobel_kernel = tf.constant(
                [
                    [[[-1, -1]], [[0, -2]], [[1, -1]]],
                    [[[-2, 0]], [[0, 0]], [[2, 0]]],
                    [[[-1, 1]], [[0, 2]], [[1, 1]]]
                ], dtype=tf.float32, name='sobel_kernel'
            )

            ang_0_kernel = tf.constant(
                [
                    [[-float('inf')], [-float('inf')], [-float('inf')]],
                    [[0.0], [0.0], [0.0]],
                    [[-float('inf')], [-float('inf')], [-float('inf')]]
                ], dtype=tf.float32, name='ang_0_kernel'
            )
            ang_45_kernel = tf.constant(
                [
                    [[-float('inf')], [-float('inf')], [0.0]],
                    [[-float('inf')], [0.0], [-float('inf')]],
                    [[0.0], [-float('inf')], [-float('inf')]]
                ], dtype=tf.float32, name='ang_45_kernel'
            )
            ang_90_kernel = tf.constant(
                [
                    [[-float('inf')], [0.0], [-float('inf')]],
                    [[-float('inf')], [0.0], [-float('inf')]],
                    [[-float('inf')], [0.0], [-float('inf')]]
                ], dtype=tf.float32, name='ang_90_kernel'
            )
            ang_135_kernel = tf.constant(
                [
                    [[0.0], [-float('inf')], [-float('inf')]],
                    [[-float('inf')], [0.0], [-float('inf')]],
                    [[-float('inf')], [-float('inf')], [0.0]]
                ], dtype=tf.float32, name='ang_135_kernel'
            )
            ang_kernel = tf.concat([ang_0_kernel, ang_45_kernel, ang_90_kernel, ang_135_kernel], axis=-1)

            dilation_kernel = tf.ones(shape=(self.con, self.con, 1), dtype=tf.float32)

            self.items = [self.__Node('gaussian_kernel', gaussian_kernel, None),
                          self.__Node('sobel_kernel', sobel_kernel, None),
                          self.__Node('ang_kernel', ang_kernel,
                                      [[157.5, 22.5], [22.5, 67.5], [67.5, 112.5], [112.5, 157.5]]),
                          self.__Node('dilation_kernel', dilation_kernel, None)
                          ]
            self.items.reverse()

        @staticmethod
        def pad(X, b=None, h=None, w=None, d=None, **kwargs):
            assert len(X.get_shape()) == 4
            if not b and not h and not w and not d: return X
            paddings = []
            for arg in [b, h, w, d]:
                arg = arg if arg is not None else [0, 0]
                arg = [arg, arg] if issubclass(type(arg), int) else list(arg)
                paddings.append(arg)
            paddings = tf.constant(paddings, dtype=tf.int32)
            padded = tf.pad(X, paddings, **kwargs)
            return padded

        @dataclass(eq=False, order=False, frozen=True)
        class __Node:
            __slots__ = ('name', 'kernel', 'carry')
            name: str
            kernel: tf.Tensor
            carry: object

            def __repr__(self):
                return self.name

# ---------------------------------------------------------------------

