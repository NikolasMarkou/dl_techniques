"""DaSiamRPN, a distractor-aware Siamese region-proposal tracker.

SiamRPN reframes tracking as one-shot local detection: the shared embedding
of the exemplar becomes a set of correlation filters, one per anchor, and
correlating them against the search embedding yields dense classification
logits plus box deltas in a single forward pass, `cls[b,i,j,k], reg[b,i,j,k]
= corr(g(z)_k, h(x))[i,j]`. DaSiamRPN keeps that mechanism and adds the
training and inference policy around it: distractor-aware sampling that
mines semantic negatives across videos and categories, and a long-term arm
that widens the search region on failure and refines the box. Only the
network is ported here; the redetection schedule and the online distractor
template update are tracking-loop policies outside the graph.

The backbone is the DaSiamRPN AlexNet variant (``valid`` everywhere,
11x11/2 + pool 3x3/2, 5x5 + pool 3x3/2, three 3x3 stages, total stride 8)
with a width scale of 1 (VOT/OTB) or 2 (BIG). Each branch then passes a 3x3
adjust convolution; the exemplar side produces ``anchor`` groups of
``feature_out``-channel 4x4 kernels and the search side a ``feature_out``
feature map, correlated per anchor group into ``anchor * 2`` classification
and ``anchor * 4`` regression channels over a 19x19 grid (127 exemplar, 271
search). Three released configurations are pinned in ``MODEL_VARIANTS``.

References:
    - Zhu et al., 2018. Distractor-aware Siamese Networks for Visual Object
      Tracking. ECCV 2018. (https://arxiv.org/abs/1808.06048)
    - Li et al., 2018. High Performance Visual Tracking with Siamese Region
      Proposal Network. CVPR 2018 (SiamRPN).
      (http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)
    - Reference implementation transcribed for widths, anchors and tracking
      hyper-parameters: https://github.com/foolwood/DaSiamRPN
      (``code/net.py`` ``SiamRPN`` / ``SiamRPNBIG`` / ``SiamRPNvot`` /
      ``SiamRPNotb``, ``code/run_SiamRPN.py`` ``generate_anchor`` /
      ``tracker_eval`` / ``TrackerConfig``)
"""

import keras
from keras import layers, ops
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------
# constants and pure helpers (single source of truth)
# ---------------------------------------------------------------------

EXEMPLAR_SIZE = 127
SEARCH_SIZE = 271
TOTAL_STRIDE = 8
ANCHOR_RATIOS = [0.33, 0.5, 1.0, 2.0, 3.0]
ANCHOR_SCALES = [8]
BASE_CHANNELS = [96, 256, 384, 384, 256]


def _valid_out(length: int, kernel: int, stride: int) -> int:
    """Output extent of one ``valid`` stage.

    :param length: Input extent.
    :type length: int
    :param kernel: Kernel extent.
    :type kernel: int
    :param stride: Stride.
    :type stride: int
    :return: Output extent.
    :rtype: int
    """
    return (length - kernel) // stride + 1


def siamrpn_backbone_feature_size(input_size: int) -> int:
    """Backbone feature extent for a square input.

    Chain ``conv11/2, pool3/2, conv5/1, pool3/2, conv3/1, conv3/1, conv3/1``.

    :param input_size: Square input extent.
    :type input_size: int
    :return: Square feature extent.
    :rtype: int
    """
    n = _valid_out(input_size, 11, 2)
    n = _valid_out(n, 3, 2)
    n = _valid_out(n, 5, 1)
    n = _valid_out(n, 3, 2)
    n = _valid_out(n, 3, 1)
    n = _valid_out(n, 3, 1)
    n = _valid_out(n, 3, 1)
    return n


def dasiamrpn_score_size(
    exemplar_size: int = EXEMPLAR_SIZE, search_size: int = SEARCH_SIZE
) -> int:
    """Score-grid extent after the 3x3 adjust convolutions.

    :param exemplar_size: Exemplar extent.
    :type exemplar_size: int
    :param search_size: Search extent.
    :type search_size: int
    :return: Square score extent.
    :rtype: int
    """
    template_kernel = siamrpn_backbone_feature_size(exemplar_size) - 3 + 1
    search_feat = siamrpn_backbone_feature_size(search_size) - 3 + 1
    return search_feat - template_kernel + 1


def generate_dasiamrpn_anchors(
    score_size: int,
    ratios: Optional[List[float]] = None,
    scales: Optional[List[int]] = None,
    total_stride: int = TOTAL_STRIDE,
) -> np.ndarray:
    """Anchor boxes for every score-grid position, transcribed from the reference.

    Bespoke NumPy, not ``layers.AnchorGenerator``: that layer emits center
    *points* on ``(j + 0.5) * stride`` FPN grids, while this layout is
    anchor-major ``(cx, cy, w, h)`` *boxes* over five aspect ratios on a
    ``-(score / 2) * stride`` origin. Adapting one to the other would break
    the bit-identical transcription pinned in the test suite.

    For each ``(ratio, scale)`` pair a zero-centered ``(w, h)`` is formed from
    ``stride * stride`` area units, tiled over the grid whose origin sits at
    ``-(score / 2) * stride``. Output layout is ``(anchor_num * score *
    score, 4)`` in ``(cx, cy, w, h)`` order, anchor-major: row ``a * S * S +
    i * S + j`` is anchor ``a`` at grid position ``(i, j)``, matching the
    reference transcription exactly (verified bit-identical).

    :param score_size: Square score-grid extent.
    :type score_size: int
    :param ratios: Anchor aspect ratios.
    :type ratios: list of float or None
    :param scales: Anchor scales.
    :type scales: list of int or None
    :param total_stride: Network total stride.
    :type total_stride: int
    :return: Anchors of shape ``(score * score * anchor_num, 4)``.
    :rtype: numpy.ndarray
    :raises ValueError: If the score size is not positive.
    """
    if ratios is None:
        ratios = list(ANCHOR_RATIOS)
    if scales is None:
        scales = list(ANCHOR_SCALES)
    if score_size <= 0:
        raise ValueError(f"score_size must be positive, got {score_size}")
    anchor_num = len(ratios) * len(scales)
    size = total_stride * total_stride
    base = np.zeros((anchor_num, 4), dtype=np.float32)
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            base[count, 0] = 0.0
            base[count, 1] = 0.0
            base[count, 2] = float(ws * scale)
            base[count, 3] = float(hs * scale)
            count += 1
    anchors = np.tile(base, score_size * score_size).reshape((-1, 4))
    origin = -(score_size / 2) * total_stride
    shifts = np.array(
        [origin + total_stride * d for d in range(score_size)], dtype=np.float32
    )
    xx, yy = np.meshgrid(shifts, shifts)
    xx = np.tile(xx.flatten(), (anchor_num, 1)).flatten()
    yy = np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchors[:, 0] = xx
    anchors[:, 1] = yy
    return anchors.astype(np.float32)


def create_hann_window(score_size: int, anchor_num: int) -> np.ndarray:
    """Cosine window tiled over anchors, penalizing large displacements.

    :param score_size: Square score-grid extent.
    :type score_size: int
    :param anchor_num: Anchors per position.
    :type anchor_num: int
    :return: Array of shape ``(score * score * anchor_num,)``.
    :rtype: numpy.ndarray
    """
    if anchor_num <= 0:
        raise ValueError(f"anchor_num must be positive, got {anchor_num}")
    window_1d = np.hanning(score_size).astype("float32")
    window_2d = np.outer(window_1d, window_1d).astype("float32")
    return np.tile(window_2d.flatten(), anchor_num).astype("float32")


def decode_dasiamrpn_boxes(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Apply ``(dx, dy, dw, dh)`` deltas to ``(cx, cy, w, h)`` anchors.

    Follows the reference ``tracker_eval`` parameterization exactly:
    centers shift by ``delta * anchor_extent`` and extents scale by
    ``exp(delta)``.

    :param anchors: Array of shape ``(N, 4)``.
    :type anchors: numpy.ndarray
    :param deltas: Array of shape ``(4, N)`` or ``(N, 4)``.
    :type deltas: numpy.ndarray
    :return: Decoded boxes of shape ``(N, 4)`` in ``(cx, cy, w, h)`` order.
    :rtype: numpy.ndarray
    """
    anchors = np.asarray(anchors, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.shape == (anchors.shape[0], 4):
        deltas = deltas.T
    if deltas.shape != (4, anchors.shape[0]):
        raise ValueError(
            f"deltas must be (N, 4) or (4, N), got {deltas.shape} for N={anchors.shape[0]}"
        )
    decoded = np.zeros_like(anchors, dtype=np.float64)
    decoded[:, 0] = deltas[0, :] * anchors[:, 2] + anchors[:, 0]
    decoded[:, 1] = deltas[1, :] * anchors[:, 3] + anchors[:, 1]
    decoded[:, 2] = np.exp(deltas[2, :]) * anchors[:, 2]
    decoded[:, 3] = np.exp(deltas[3, :]) * anchors[:, 3]
    return decoded.astype(np.float32)


def _per_location_channel_correlation(
    search_features: keras.KerasTensor,
    kernels: keras.KerasTensor,
    score_size: int,
    kernel_extent: int,
) -> keras.KerasTensor:
    """Correlate one search map against a batch of multi-channel kernels.

    At each valid offset the search patch ``(Hk, Wk, C)`` is dotted against
    every kernel ``(Hk, Wk, C)`` for that batch element via a single einsum,
    vectorized over the output-channel axis. Loop bounds are static Python
    ints, so tracing stays symbolic.

    :param search_features: Tensor of shape ``(B, Hx, Wx, C)``.
    :type search_features: keras.KerasTensor
    :param kernels: Tensor of shape ``(B, Hk, Wk, Ko, C)``.
    :type kernels: keras.KerasTensor
    :param score_size: Square output extent.
    :type score_size: int
    :param kernel_extent: Square kernel extent.
    :type kernel_extent: int
    :return: Tensor of shape ``(B, score, score, Ko)``.
    :rtype: keras.KerasTensor
    """
    rows: List[keras.KerasTensor] = []
    for i in range(score_size):
        cols: List[keras.KerasTensor] = []
        for j in range(score_size):
            patch = search_features[
                :, i : i + kernel_extent, j : j + kernel_extent, :
            ]
            cols.append(ops.einsum("bhwc,bhwkc->bk", patch, kernels))
        rows.append(ops.stack(cols, axis=1))
    grid = ops.stack(rows, axis=1)
    return grid


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.dasiamrpn.model")
class SiamRPNBackbone(keras.layers.Layer):
    """DaSiamRPN embedding trunk with configurable width scale.

    Same stage geometry as :class:`SiamFCBackbone` but the channel ladder
    scales: ``width_scale=1`` gives ``96, 256, 384, 384, 256`` (VOT/OTB) and
    ``width_scale=2`` doubles every entry (BIG). The original Caffe
    ``groups=2`` on two convolutions is a multi-GPU artifact and is not
    reproduced; dense convolutions learn the same function class.

    :param width_scale: Channel multiplier, 1 or 2.
    :type width_scale: int
    :param use_batch_norm: Whether to keep BatchNorm after the first four convolutions.
    :type use_batch_norm: bool
    :param bn_momentum: BatchNorm momentum in the Keras convention.
    :type bn_momentum: float
    :param bn_epsilon: BatchNorm epsilon.
    :type bn_epsilon: float
    :param kernel_initializer: Initializer for all convolutions.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Passthrough to ``keras.layers.Layer``.
    :raises ValueError: If ``width_scale`` is not 1 or 2.
    """

    def __init__(
        self,
        width_scale: int = 1,
        use_batch_norm: bool = True,
        bn_momentum: float = 0.9,
        bn_epsilon: float = 1e-5,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if width_scale not in (1, 2):
            raise ValueError(f"width_scale must be 1 or 2, got {width_scale}")
        self.width_scale = width_scale
        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.kernel_initializer = kernel_initializer
        channels = [c * width_scale for c in BASE_CHANNELS]
        self.out_channels = channels[-1]

        def _conv(filters: int, kernel: int, stride: int, name: str) -> layers.Conv2D:
            return layers.Conv2D(
                filters=filters,
                kernel_size=kernel,
                strides=stride,
                padding="valid",
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                name=name,
            )

        def _bn(name: str) -> keras.layers.Layer:
            # Routed through the norms factory, which honors an explicitly
            # passed epsilon (its 1e-6 is only the default): the transcribed
            # torch-default epsilon is kept.
            return create_normalization_layer(
                "batch_norm",
                momentum=self.bn_momentum,
                epsilon=self.bn_epsilon,
                name=name,
            )

        self.conv1 = _conv(channels[0], 11, 2, "conv1")
        self.bn1 = _bn("bn1")
        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="pool1")
        self.conv2 = _conv(channels[1], 5, 1, "conv2")
        self.bn2 = _bn("bn2")
        self.pool2 = layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="pool2")
        self.conv3 = _conv(channels[2], 3, 1, "conv3")
        self.bn3 = _bn("bn3")
        self.conv4 = _conv(channels[3], 3, 1, "conv4")
        self.bn4 = _bn("bn4")
        self.conv5 = _conv(channels[4], 3, 1, "conv5")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        shape = input_shape
        self.conv1.build(shape)
        shape = self.conv1.compute_output_shape(shape)
        if self.use_batch_norm:
            self.bn1.build(shape)
        self.pool1.build(shape)
        shape = self.pool1.compute_output_shape(shape)
        self.conv2.build(shape)
        shape = self.conv2.compute_output_shape(shape)
        if self.use_batch_norm:
            self.bn2.build(shape)
        self.pool2.build(shape)
        shape = self.pool2.compute_output_shape(shape)
        self.conv3.build(shape)
        shape = self.conv3.compute_output_shape(shape)
        if self.use_batch_norm:
            self.bn3.build(shape)
        self.conv4.build(shape)
        shape = self.conv4.compute_output_shape(shape)
        if self.use_batch_norm:
            self.bn4.build(shape)
        self.conv5.build(shape)
        super().build(input_shape)

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Embed an image batch.

        :param inputs: Tensor of shape ``(B, H, W, 3)``.
        :type inputs: keras.KerasTensor
        :param training: Training flag forwarded to BatchNorm explicitly.
        :type training: bool or None
        :return: Features of shape ``(B, H', W', out_channels)``.
        :rtype: keras.KerasTensor
        """
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x, training=training)
        x = ops.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x, training=training)
        x = ops.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x, training=training)
        x = ops.relu(x)
        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x, training=training)
        x = ops.relu(x)
        x = self.conv5(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        extent = input_shape[1]
        height_out: Optional[int] = None
        if extent is not None:
            height_out = siamrpn_backbone_feature_size(int(extent))
        return (input_shape[0], height_out, height_out, self.out_channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "width_scale": self.width_scale,
                "use_batch_norm": self.use_batch_norm,
                "bn_momentum": self.bn_momentum,
                "bn_epsilon": self.bn_epsilon,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SiamRPNBackbone":
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        return cls(**config)


@register_dl_technique("dl_techniques.models.dasiamrpn.model")
class DaSiamRPN(keras.Model):
    """Distractor-aware Siamese region-proposal tracker.

    The shared :class:`SiamRPNBackbone` embeds both inputs; 3x3 adjust
    convolutions split each embedding into classification and regression
    streams, the exemplar side is reshaped into per-anchor kernels, and each
    stream is correlated independently. ``call()`` takes a ``(z, x)`` pair
    and returns ``{"cls": (B, S, S, A * 2), "reg": (B, S, S, A * 4)}`` raw
    logits and deltas. No soft-max, box decode, penalty or window lives in
    the graph; :func:`decode_dasiamrpn_boxes` and the anchor factory cover
    post-processing in NumPy.

    :param variant_config: One ``MODEL_VARIANTS`` entry with ``width_scale`` and ``feature_out``.
    :type variant_config: dict or None
    :param anchor_ratios: Anchor aspect ratios.
    :type anchor_ratios: list of float or None
    :param anchor_scales: Anchor scales.
    :type anchor_scales: list of int or None
    :param exemplar_size: Square exemplar extent.
    :type exemplar_size: int
    :param search_size: Square search extent.
    :type search_size: int
    :param use_batch_norm: Forwarded to the backbone.
    :type use_batch_norm: bool
    :param kernel_initializer: Forwarded to backbone and adjust convolutions.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Passthrough to ``keras.Model``.
    :raises ValueError: If the anchor lists are empty or the sizes collapse the grid.
    """

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "big": {"width_scale": 2, "feature_out": 512},
        "vot": {"width_scale": 1, "feature_out": 256},
        "otb": {"width_scale": 1, "feature_out": 256},
    }

    def __init__(
        self,
        variant_config: Optional[Dict[str, Any]] = None,
        anchor_ratios: Optional[List[float]] = None,
        anchor_scales: Optional[List[int]] = None,
        exemplar_size: int = EXEMPLAR_SIZE,
        search_size: int = SEARCH_SIZE,
        use_batch_norm: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if variant_config is None:
            variant_config = dict(self.MODEL_VARIANTS["big"])
        if "width_scale" not in variant_config or "feature_out" not in variant_config:
            raise ValueError("variant_config must carry 'width_scale' and 'feature_out'")
        width_scale = int(variant_config["width_scale"])
        feature_out = int(variant_config["feature_out"])
        if width_scale not in (1, 2):
            raise ValueError(f"width_scale must be 1 or 2, got {width_scale}")
        if feature_out <= 0:
            raise ValueError(f"feature_out must be positive, got {feature_out}")
        self.anchor_ratios = list(anchor_ratios) if anchor_ratios is not None else list(ANCHOR_RATIOS)
        self.anchor_scales = list(anchor_scales) if anchor_scales is not None else list(ANCHOR_SCALES)
        if not self.anchor_ratios or not self.anchor_scales:
            raise ValueError("anchor_ratios and anchor_scales must be non-empty")
        if exemplar_size <= 0 or search_size <= exemplar_size:
            raise ValueError(
                f"need 0 < exemplar ({exemplar_size}) < search ({search_size})"
            )
        self.anchor_num = len(self.anchor_ratios) * len(self.anchor_scales)
        self._score_size = dasiamrpn_score_size(exemplar_size, search_size)
        self._kernel_extent = siamrpn_backbone_feature_size(exemplar_size) - 3 + 1
        if self._score_size <= 0 or self._kernel_extent <= 0:
            raise ValueError(
                f"size pair ({exemplar_size}, {search_size}) collapses the RPN grid "
                f"(score {self._score_size}, kernel {self._kernel_extent})"
            )
        self.variant_config = dict(variant_config)
        self.exemplar_size = exemplar_size
        self.search_size = search_size
        self.use_batch_norm = use_batch_norm
        self.kernel_initializer = kernel_initializer
        self.feature_out = feature_out

        feat_in = BASE_CHANNELS[-1] * width_scale
        self.backbone = SiamRPNBackbone(
            width_scale=width_scale,
            use_batch_norm=use_batch_norm,
            kernel_initializer=kernel_initializer,
            name="backbone",
        )
        self.conv_r1 = layers.Conv2D(
            filters=feature_out * 4 * self.anchor_num,
            kernel_size=3,
            padding="valid",
            kernel_initializer=kernel_initializer,
            name="conv_r1",
        )
        self.conv_r2 = layers.Conv2D(
            filters=feature_out,
            kernel_size=3,
            padding="valid",
            kernel_initializer=kernel_initializer,
            name="conv_r2",
        )
        self.conv_cls1 = layers.Conv2D(
            filters=feature_out * 2 * self.anchor_num,
            kernel_size=3,
            padding="valid",
            kernel_initializer=kernel_initializer,
            name="conv_cls1",
        )
        self.conv_cls2 = layers.Conv2D(
            filters=feature_out,
            kernel_size=3,
            padding="valid",
            kernel_initializer=kernel_initializer,
            name="conv_cls2",
        )
        self.regress_adjust = layers.Conv2D(
            filters=4 * self.anchor_num,
            kernel_size=1,
            padding="valid",
            kernel_initializer=kernel_initializer,
            name="regress_adjust",
        )
        self._feat_in = feat_in

    def build(self, input_shape: Any) -> None:
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self, inputs: Union[Tuple[Any, Any], List[Any]], training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """Run the one-shot detection forward pass.

        :param inputs: Pair ``(z, x)`` with shapes ``(B, Ez, Ez, 3)`` and ``(B, Sx, Sx, 3)``.
        :type inputs: tuple or list of two tensors
        :param training: Training flag forwarded to backbone and adjust convolutions.
        :type training: bool or None
        :return: Dict with ``cls`` logits ``(B, S, S, A * 2)`` and ``reg`` deltas ``(B, S, S, A * 4)``.
        :rtype: dict
        """
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("DaSiamRPN expects a (exemplar, search) pair of two tensors")
        exemplar, search = inputs[0], inputs[1]
        z_feat = self.backbone(exemplar, training=training)
        x_feat = self.backbone(search, training=training)

        r1_raw = self.conv_r1(z_feat, training=training)
        cls1_raw = self.conv_cls1(z_feat, training=training)
        r2_feat = self.conv_r2(x_feat, training=training)
        cls2_feat = self.conv_cls2(x_feat, training=training)

        batch = ops.shape(r1_raw)[0]
        k = self._kernel_extent
        r_kernels = ops.reshape(
            r1_raw, (batch, k, k, self.anchor_num * 4, self.feature_out)
        )
        cls_kernels = ops.reshape(
            cls1_raw, (batch, k, k, self.anchor_num * 2, self.feature_out)
        )
        reg_corr = _per_location_channel_correlation(
            r2_feat, r_kernels, self._score_size, k
        )
        cls_corr = _per_location_channel_correlation(
            cls2_feat, cls_kernels, self._score_size, k
        )
        reg = self.regress_adjust(reg_corr, training=training)
        return {"cls": cls_corr, "reg": reg}

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "variant_config": dict(self.variant_config),
                "anchor_ratios": list(self.anchor_ratios),
                "anchor_scales": list(self.anchor_scales),
                "exemplar_size": self.exemplar_size,
                "search_size": self.search_size,
                "use_batch_norm": self.use_batch_norm,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DaSiamRPN":
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        exemplar_size: int = EXEMPLAR_SIZE,
        search_size: int = SEARCH_SIZE,
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
    ) -> "DaSiamRPN":
        """Create a :class:`DaSiamRPN` from a released configuration name.

        :param variant: One of ``"big"``, ``"vot"``, ``"otb"``.
        :type variant: str
        :param exemplar_size: Square exemplar extent.
        :type exemplar_size: int
        :param search_size: Square search extent.
        :type search_size: int
        :param pretrained: A local checkpoint path, or True to raise since no weights ship here.
        :type pretrained: bool or str
        :param kwargs: Additional constructor arguments.
        :return: A :class:`DaSiamRPN` instance.
        :rtype: DaSiamRPN
        :raises ValueError: If the variant name is unknown.
        :raises NotImplementedError: If ``pretrained`` is True.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: {list(cls.MODEL_VARIANTS.keys())}"
            )
        variant_config = dict(cls.MODEL_VARIANTS[variant])
        model = cls(
            variant_config=variant_config,
            exemplar_size=exemplar_size,
            search_size=search_size,
            **kwargs,
        )
        if pretrained:
            if isinstance(pretrained, str):
                if not model.built:
                    model.build(
                        [
                            (None, exemplar_size, exemplar_size, 3),
                            (None, search_size, search_size, 3),
                        ]
                    )
                report = load_weights_from_checkpoint(
                    target=model, ckpt_path=pretrained, strict=True
                )
                logger.info(report.summary_string())
            else:
                raise NotImplementedError(
                    f"No pretrained DaSiamRPN weights are distributed with dl_techniques "
                    f"(requested variant '{variant}'). Pass a local checkpoint instead: "
                    f"DaSiamRPN.from_variant('{variant}', pretrained='/path/to/weights.keras')."
                )
        return model


# ---------------------------------------------------------------------


def create_dasiamrpn(
    variant: str = "big",
    exemplar_size: int = EXEMPLAR_SIZE,
    search_size: int = SEARCH_SIZE,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> DaSiamRPN:
    """Create a :class:`DaSiamRPN` model. A thin wrapper over ``from_variant``.

    :param variant: One of ``"big"``, ``"vot"``, ``"otb"``.
    :type variant: str
    :param exemplar_size: Square exemplar extent.
    :type exemplar_size: int
    :param search_size: Square search extent.
    :type search_size: int
    :param pretrained: A local checkpoint path, or True to raise.
    :type pretrained: bool or str
    :param kwargs: Additional arguments for :class:`DaSiamRPN`.
    :return: A :class:`DaSiamRPN` instance.
    :rtype: DaSiamRPN
    """
    return DaSiamRPN.from_variant(
        variant,
        exemplar_size=exemplar_size,
        search_size=search_size,
        pretrained=pretrained,
        **kwargs,
    )
