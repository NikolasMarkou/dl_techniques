"""Region-Aware Deformable Convolution (RAD-Conv).

RAD-Conv replaces the fixed-quadrilateral, 4-neighbour bilinear sampling of
deformable convolutions (DCNv1-v4) with, per kernel element and channel
group, four independently predicted non-negative boundary offsets
``(top, bottom, left, right)`` that define an axis-aligned rectangle around
that kernel element's nominal grid position. Each rectangle is exactly
spatially integrated (averaged) via a summed-area-table (Precise RoI
Pooling-style bilinear interpolation of the integral image at the box's four
corners -- exact for a piecewise-constant image, not an approximation),
scaled by an input-dependent per-region weight, mixed by a shared static
per-kernel-element weight matrix, and summed over kernel elements to produce
the output feature map. Decoupling receptive-field geometry (adaptive,
up to the full image) from kernel size (as few as 1x1) lets a single layer
approach the spatial adaptability of attention while keeping the parameter
and compute profile of a convolution.

References:
    - Maleki & Imani, 2025. Region-Aware Deformable Convolutions.
      (https://arxiv.org/abs/2509.15436)
    - Jiang et al., 2018. Acquisition of Localization Confidence for Accurate
      Object Detection (Precise RoI Pooling). (https://arxiv.org/abs/1807.11590)
    - Dai et al., 2017. Deformable Convolutional Networks.
      (https://arxiv.org/abs/1703.06211)
"""

import keras
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Guards the box-area denominator against a degenerate (zero-extent) region.
AREA_EPSILON = 1e-6

# ---------------------------------------------------------------------

def _sat_gather(
    sat: "keras.KerasTensor",
    idx_y: "keras.KerasTensor",
    idx_x: "keras.KerasTensor",
) -> "keras.KerasTensor":
    """Gather ``sat[b, idx_y, idx_x, :]`` for a batched integer index map.

    Follows the repo's linearized-index gather idiom (see
    ``layers/spatial_layer.py::_gather_hw``): flatten the leading three axes
    of ``sat``, form ``(b * Hs + idx_y) * Ws + idx_x``, ``take`` along axis 0,
    reshape back to the query shape plus the channel axis.

    :param sat: ``(B, Hs, Ws, C)`` summed-area-table tensor.
    :type sat: keras.KerasTensor
    :param idx_y: ``(B, L)`` int32 clamped row indices.
    :type idx_y: keras.KerasTensor
    :param idx_x: ``(B, L)`` int32 clamped column indices.
    :type idx_x: keras.KerasTensor
    :return: ``(B, L, C)`` gathered values.
    :rtype: keras.KerasTensor
    """
    sat_shape = keras.ops.shape(sat)
    size_h, size_w, channels = sat_shape[1], sat_shape[2], sat_shape[3]

    lead = keras.ops.shape(idx_y)
    batch_idx = keras.ops.reshape(keras.ops.arange(0, lead[0], dtype="int32"), (-1, 1))
    batch_idx = keras.ops.broadcast_to(batch_idx, lead)

    flat_idx = (batch_idx * size_h + idx_y) * size_w + idx_x
    flat_sat = keras.ops.reshape(sat, (-1, channels))
    gathered = keras.ops.take(flat_sat, keras.ops.reshape(flat_idx, (-1,)), axis=0)
    return keras.ops.reshape(gathered, (lead[0], lead[1], channels))

# ---------------------------------------------------------------------

def _sat_bilinear(
    sat: "keras.KerasTensor",
    y: "keras.KerasTensor",
    x: "keras.KerasTensor",
    max_y: "keras.KerasTensor",
    max_x: "keras.KerasTensor",
) -> "keras.KerasTensor":
    """Bilinearly sample a summed-area table at continuous ``(y, x)``.

    Because the underlying image is piecewise-constant per pixel, the true
    double integral up to a fractional boundary is exactly linear between
    adjacent integer grid points of the SAT along each axis -- so a 4-corner
    bilinear lerp of the (integer-indexed) SAT recovers the exact continuous
    integral value, not merely an interpolated approximation of it.

    :param sat: ``(B, Hs, Ws, C)`` summed-area table, ``Hs = H + 1``, ``Ws = W + 1``.
    :type sat: keras.KerasTensor
    :param y: ``(B, L)`` continuous row coordinates in ``[0, Hs - 1]``.
    :type y: keras.KerasTensor
    :param x: ``(B, L)`` continuous column coordinates in ``[0, Ws - 1]``.
    :type x: keras.KerasTensor
    :param max_y: Scalar int32 tensor, ``Hs - 1``.
    :type max_y: keras.KerasTensor
    :param max_x: Scalar int32 tensor, ``Ws - 1``.
    :type max_x: keras.KerasTensor
    :return: ``(B, L, C)`` interpolated SAT values.
    :rtype: keras.KerasTensor
    """
    y0f = keras.ops.floor(y)
    x0f = keras.ops.floor(x)
    fy = y - y0f
    fx = x - x0f

    y0 = keras.ops.clip(keras.ops.cast(y0f, "int32"), 0, max_y)
    x0 = keras.ops.clip(keras.ops.cast(x0f, "int32"), 0, max_x)
    y1 = keras.ops.clip(y0 + 1, 0, max_y)
    x1 = keras.ops.clip(x0 + 1, 0, max_x)

    v00 = _sat_gather(sat, y0, x0)
    v01 = _sat_gather(sat, y0, x1)
    v10 = _sat_gather(sat, y1, x0)
    v11 = _sat_gather(sat, y1, x1)

    fy = keras.ops.expand_dims(fy, axis=-1)
    fx = keras.ops.expand_dims(fx, axis=-1)

    top = v00 * (1.0 - fx) + v01 * fx
    bot = v10 * (1.0 - fx) + v11 * fx
    return top * (1.0 - fy) + bot * fy

# ---------------------------------------------------------------------

def _box_average(
    sat: "keras.KerasTensor",
    t: "keras.KerasTensor",
    b: "keras.KerasTensor",
    l: "keras.KerasTensor",
    r: "keras.KerasTensor",
) -> "keras.KerasTensor":
    """Exact spatial average of the source image over axis-aligned boxes.

    :param sat: ``(B, H + 1, W + 1, C)`` summed-area table of the source image.
    :type sat: keras.KerasTensor
    :param t: ``(B, L)`` top boundary, continuous, in ``[0, H]``.
    :type t: keras.KerasTensor
    :param b: ``(B, L)`` bottom boundary, continuous, in ``[0, H]``, ``b >= t``.
    :type b: keras.KerasTensor
    :param l: ``(B, L)`` left boundary, continuous, in ``[0, W]``.
    :type l: keras.KerasTensor
    :param r: ``(B, L)`` right boundary, continuous, in ``[0, W]``, ``r >= l``.
    :type r: keras.KerasTensor
    :return: ``(B, L, C)`` per-box spatial average of the source image.
    :rtype: keras.KerasTensor
    """
    sat_shape = keras.ops.shape(sat)
    max_y = sat_shape[1] - 1
    max_x = sat_shape[2] - 1

    val_tl = _sat_bilinear(sat, t, l, max_y, max_x)
    val_tr = _sat_bilinear(sat, t, r, max_y, max_x)
    val_bl = _sat_bilinear(sat, b, l, max_y, max_x)
    val_br = _sat_bilinear(sat, b, r, max_y, max_x)

    box_sum = val_br - val_bl - val_tr + val_tl
    area = (b - t) * (r - l)
    area = keras.ops.expand_dims(keras.ops.maximum(area, AREA_EPSILON), axis=-1)
    return box_sum / area

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.conv_blocks.rad_conv")
class RADConv2D(keras.layers.Layer):
    """Region-Aware Deformable Convolution 2D layer.

    Implements RAD-Conv (Maleki & Imani, 2025). For each output spatial
    position, each of ``K = kernel_size[0] * kernel_size[1]`` kernel elements
    and each of ``groups`` channel groups, the layer predicts four
    non-negative boundary offsets that define an axis-aligned rectangle
    around that kernel element's nominal grid position, exactly integrates
    (spatially averages) the input over that rectangle via a summed-area
    table, scales the result by a predicted per-region dynamic weight, mixes
    the ``K`` per-group regional values with a shared static weight matrix,
    and sums over ``k`` -- decoupling receptive-field extent and aspect
    ratio from the kernel's own size.

    Architecture:

    .. code-block:: text

        Input (B, H, W, C_in)
              │
              ├─────────────────────┬──────────────────────┐
              ▼                     ▼                       ▼
        Conv1x1(4*K*G)        Conv1x1(K*G)         SummedAreaTable(x)
        + softplus            + softmax(axis=K)          │
        (Δt,Δb,Δl,Δr)              m                      │
              │                     │                      │
              ▼                     │                      │
        box corners per (k,g) ──────┼──────────────────────┘
              │                     │
              ▼                     │
        exact box average (B,H,W,G,K,C_in/G)
              │                     │
              └─────────► * m ◄─────┘
                          │
                          ▼
              einsum with static kernel (G,K,C_in/G,C_out/G)
                          │
                          ▼
              concat groups + bias -> (B, H, W, filters)

    :param filters: Number of output channels. Must be a positive int and
        divisible by ``groups``.
    :type filters: int
    :param kernel_size: Number of kernel elements as ``(kh, kw)``, or a single
        int for a square kernel. Defaults to ``3`` (i.e. ``(3, 3)``, ``K=9``).
    :type kernel_size: int or tuple[int, int]
    :param groups: Number of channel groups. Input and output channels must
        each be divisible by ``groups``. Defaults to ``1``.
    :type groups: int
    :param use_bias: Whether to add a learnable per-output-channel bias.
        Defaults to ``True``.
    :type use_bias: bool
    :param offset_bias_init: Initial constant bias for the offset-prediction
        conv (before ``softplus``), controlling the initial region size
        around each kernel element. Defaults to ``1.0``.
    :type offset_bias_init: float
    :param kernel_initializer: Initializer for the static per-``(g, k)``
        mixing weight. Defaults to ``"glorot_uniform"``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for the output bias. Defaults to
        ``"zeros"``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer on the static mixing
        weight. Defaults to ``None``.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.
    :type kwargs: Any
    :raises ValueError: If ``filters``/``groups``/``kernel_size`` are invalid,
        or ``filters`` / the input channel count is not divisible by
        ``groups``.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        groups: int = 1,
        use_bias: bool = True,
        offset_bias_init: float = 1.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        if kernel_size[0] <= 0 or kernel_size[1] <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if filters % groups != 0:
            raise ValueError(
                f"filters ({filters}) must be divisible by groups ({groups})"
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups
        self.use_bias = use_bias
        self.offset_bias_init = float(offset_bias_init)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        kh, kw = self.kernel_size
        self.num_kernel_elements = kh * kw
        # Static (h, w) grid offsets of each kernel element relative to the
        # output position, e.g. for a 3x3 kernel: {-1, 0, 1} x {-1, 0, 1}.
        py = np.arange(kh, dtype="float32") - (kh - 1) / 2.0
        px = np.arange(kw, dtype="float32") - (kw - 1) / 2.0
        grid_y, grid_x = np.meshgrid(py, px, indexing="ij")
        self._base_py = grid_y.reshape(-1)  # (K,)
        self._base_px = grid_x.reshape(-1)  # (K,)

        self.cin_per_group = None
        self.cout_per_group = self.filters // self.groups

        self.offset_conv = None
        self.weight_conv = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the offset/weight predictor convs and the static kernel.

        :param input_shape: ``(B, H, W, C_in)`` input shape. ``C_in`` must be
            statically known.
        :type input_shape: tuple
        :raises ValueError: If the input is not 4D, ``C_in`` is unknown, or
            ``C_in`` is not divisible by ``groups``.
        """
        if len(input_shape) != 4:
            raise ValueError(f"RADConv2D expects a 4D input, got shape {input_shape}")
        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError(
                "RADConv2D requires a statically known channel dimension, "
                f"got input_shape={input_shape}"
            )
        if input_channels % self.groups != 0:
            raise ValueError(
                f"input channels ({input_channels}) must be divisible by "
                f"groups ({self.groups})"
            )
        self.cin_per_group = input_channels // self.groups

        k, g = self.num_kernel_elements, self.groups

        self.offset_conv = keras.layers.Conv2D(
            filters=4 * k * g,
            kernel_size=1,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer=keras.initializers.Constant(self.offset_bias_init),
            name="offset_conv",
            dtype=self.dtype_policy,
        )
        self.offset_conv.build(input_shape)

        self.weight_conv = keras.layers.Conv2D(
            filters=k * g,
            kernel_size=1,
            padding="same",
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="weight_conv",
            dtype=self.dtype_policy,
        )
        self.weight_conv.build(input_shape)

        self.kernel = self.add_weight(
            name="kernel",
            shape=(g, k, self.cin_per_group, self.cout_per_group),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        super().build(input_shape)

    def call(
        self,
        inputs: "keras.KerasTensor",
        training: Optional[bool] = None,
        **kwargs: Any,
    ) -> "keras.KerasTensor":
        """Apply Region-Aware Deformable Convolution.

        :param inputs: ``(B, H, W, C_in)`` input tensor.
        :type inputs: keras.KerasTensor
        :param training: Unused, kept for interface consistency.
        :type training: bool or None
        :return: ``(B, H, W, filters)`` output tensor.
        :rtype: keras.KerasTensor
        """
        shape = keras.ops.shape(inputs)
        batch = shape[0]
        height = shape[1]
        width = shape[2]
        k, g = self.num_kernel_elements, self.groups

        offsets_raw = self.offset_conv(inputs)
        offsets = keras.ops.softplus(keras.ops.cast(offsets_raw, "float32"))
        offsets = keras.ops.reshape(offsets, (batch, height, width, g, k, 4))

        weights_raw = self.weight_conv(inputs)
        weights_raw = keras.ops.reshape(weights_raw, (batch, height, width, g, k))
        if k > 1:
            dyn_weights = keras.ops.softmax(weights_raw, axis=-1)  # (B,H,W,G,K)
        else:
            # softmax over a size-1 axis is always 1 (and Keras warns about
            # it); a single kernel element trivially gets full weight.
            dyn_weights = keras.ops.ones_like(weights_raw)

        # Base (float) output grid coordinates, pixel-corner convention:
        # pixel index i occupies the continuous span [i, i + 1).
        y0 = keras.ops.cast(keras.ops.arange(0, height, dtype="int32"), "float32") + 0.5
        x0 = keras.ops.cast(keras.ops.arange(0, width, dtype="int32"), "float32") + 0.5
        center_y = keras.ops.reshape(y0, (1, height, 1, 1)) + keras.ops.reshape(
            keras.ops.convert_to_tensor(self._base_py), (1, 1, 1, k)
        )
        center_x = keras.ops.reshape(x0, (1, 1, width, 1)) + keras.ops.reshape(
            keras.ops.convert_to_tensor(self._base_px), (1, 1, 1, k)
        )
        # (1, H, 1, K) / (1, 1, W, K) -> broadcast to (1, H, W, K) -> (1,H,W,1,K)
        center_y = keras.ops.expand_dims(
            keras.ops.broadcast_to(center_y, (1, height, width, k)), axis=3
        )
        center_x = keras.ops.expand_dims(
            keras.ops.broadcast_to(center_x, (1, height, width, k)), axis=3
        )

        dt = offsets[..., 0]
        db = offsets[..., 1]
        dl = offsets[..., 2]
        dr = offsets[..., 3]

        height_f = keras.ops.cast(height, "float32")
        width_f = keras.ops.cast(width, "float32")
        t = keras.ops.clip(center_y - dt, 0.0, height_f)
        b = keras.ops.clip(center_y + db, 0.0, height_f)
        l = keras.ops.clip(center_x - dl, 0.0, width_f)
        r = keras.ops.clip(center_x + dr, 0.0, width_f)
        b = keras.ops.maximum(b, t)
        r = keras.ops.maximum(r, l)

        outputs = []
        for group_idx in range(g):
            x_g = inputs[..., group_idx * self.cin_per_group:(group_idx + 1) * self.cin_per_group]
            x_g = keras.ops.cast(x_g, "float32")
            cumulative = keras.ops.cumsum(keras.ops.cumsum(x_g, axis=1), axis=2)
            sat = keras.ops.pad(cumulative, [[0, 0], [1, 0], [1, 0], [0, 0]])

            t_g = keras.ops.reshape(t[:, :, :, group_idx, :], (batch, height * width * k))
            b_g = keras.ops.reshape(b[:, :, :, group_idx, :], (batch, height * width * k))
            l_g = keras.ops.reshape(l[:, :, :, group_idx, :], (batch, height * width * k))
            r_g = keras.ops.reshape(r[:, :, :, group_idx, :], (batch, height * width * k))

            regional = _box_average(sat, t_g, b_g, l_g, r_g)  # (B, H*W*K, Cin_g)
            regional = keras.ops.reshape(
                regional, (batch, height, width, k, self.cin_per_group)
            )

            m_g = keras.ops.expand_dims(dyn_weights[:, :, :, group_idx, :], axis=-1)
            weighted = regional * m_g  # (B,H,W,K,Cin_g)

            out_g = keras.ops.einsum(
                "bhwkc,kco->bhwo", weighted, self.kernel[group_idx]
            )
            outputs.append(out_g)

        output = outputs[0] if g == 1 else keras.ops.concatenate(outputs, axis=-1)
        output = keras.ops.cast(output, self.compute_dtype)
        if self.use_bias:
            output = output + keras.ops.cast(self.bias, self.compute_dtype)
        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: ``(B, H, W, C_in)`` input shape.
        :type input_shape: tuple
        :return: ``(B, H, W, filters)`` output shape.
        :rtype: tuple
        """
        return (input_shape[0], input_shape[1], input_shape[2], self.filters)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Configuration dict with all constructor parameters.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "groups": self.groups,
            "use_bias": self.use_bias,
            "offset_bias_init": self.offset_bias_init,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
