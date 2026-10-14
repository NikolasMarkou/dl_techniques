"""SiamFC fully-convolutional Siamese tracker and the ``create_siamfc`` factory.

:class:`SiamFC` embeds an exemplar patch ``z`` and a search region ``x`` with
one shared AlexNet-style :class:`SiamFCBackbone`, then cross-correlates the
two feature maps into a single-channel score map whose peak marks the target:

    response = corr(f(z), f(x)) + bias

Instead of scoring shifted crops one at a time, one forward pass scores every
offset of the exemplar inside the search region. Input sizes are fixed at
construction: the score-map extent is computed from ``exemplar_size`` and
``search_size``, and ``call`` does not grow it for larger inputs. The output
is logits, 17x17 at the default 127 / 255 sizes. The Hann window helper runs
outside the graph. There is no variant table, and ``pretrained=True`` raises
``NotImplementedError``.

References:
    - Bertinetto et al., 2016. Fully-Convolutional Siamese Networks for
      Object Tracking. (https://arxiv.org/abs/1606.09549)
    - Bertinetto et al., 2017. Learning Feed-Forward One-Shot Learners
      (appendix B center-error protocol, baseline embedding with total
      stride 4). (https://arxiv.org/abs/1606.05233)
    - Reference ports transcribed for hyperparameters:
      https://github.com/bertinetto/siamese-fc (MatConvNet, original),
      https://github.com/rafellerc/Pytorch-SiamFC (``training/models.py``
      ``BaselineEmbeddingNet`` / ``SiameseNet``),
      https://github.com/torrvision/siamfc-tf
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
# constants and shape helpers
# ---------------------------------------------------------------------

EXEMPLAR_SIZE = 127
SEARCH_SIZE = 255
# Reference values for the default sizes; nothing in this file reads them.
TOTAL_STRIDE = 8
SCORE_SIZE = 17
BACKBONE_OUT_CHANNELS = 256

# ---------------------------------------------------------------------

def _valid_out(length: int, kernel: int, stride: int) -> int:
    """Output extent of one ``valid`` convolution/pool stage.

    :param length: Input extent along one axis.
    :type length: int
    :param kernel: Kernel extent.
    :type kernel: int
    :param stride: Stride.
    :type stride: int
    :return: Output extent.
    :rtype: int
    """
    return (length - kernel) // stride + 1

# ---------------------------------------------------------------------

def siamfc_backbone_feature_size(input_size: int) -> int:
    """Return the spatial extent of the SiamFC embedding for a square input.

    Applies the backbone chain ``conv11/2, pool3/2, conv5/1, pool3/2,
    conv3/1, conv3/1, conv3/1`` with ``valid`` padding. :class:`SiamFC` uses
    it to size the score map and :meth:`SiamFCBackbone.compute_output_shape`
    to report shapes. Small inputs give zero or negative values; nothing here
    checks for that.

    :param input_size: Square input extent in pixels.
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

# ---------------------------------------------------------------------

def siamfc_score_size(exemplar_size: int = EXEMPLAR_SIZE, search_size: int = SEARCH_SIZE) -> int:
    """Return the response-map extent for an exemplar/search size pair.

    The defaults give 17.

    :param exemplar_size: Exemplar patch extent.
    :type exemplar_size: int
    :param search_size: Search region extent.
    :type search_size: int
    :return: Square score-map extent.
    :rtype: int
    """
    return (
        siamfc_backbone_feature_size(search_size)
        - siamfc_backbone_feature_size(exemplar_size)
        + 1
    )

# ---------------------------------------------------------------------

def create_hann_window(score_size: int) -> np.ndarray:
    """Build a separable 2D Hann window for inference post-processing.

    NumPy only, outside the graph. The window is returned unblended; the
    caller applies ``response = (1 - w) * response + w * hann``.

    :param score_size: Square score-map extent. Must be positive.
    :type score_size: int
    :return: Array of shape ``(score_size, score_size)`` in ``float32``.
    :rtype: numpy.ndarray
    :raises ValueError: If ``score_size`` is not positive.
    """
    if score_size <= 0:
        raise ValueError(f"score_size must be positive, got {score_size}")
    window_1d = np.hanning(score_size).astype("float32")
    return np.outer(window_1d, window_1d).astype("float32")

# ---------------------------------------------------------------------

def _cross_correlate(
    template_features: keras.KerasTensor,
    search_features: keras.KerasTensor,
    score_size: int,
    template_extent: int,
    bias: keras.KerasTensor,
) -> keras.KerasTensor:
    """Cross-correlate template features over search features, one channel out.

    For each of the ``score_size ** 2`` offsets, dots the template feature
    volume with the aligned search patch, then adds a scalar bias. Loop bounds
    are Python ints from the constructor, so the graph holds one slice and
    sum per offset. Only offsets up to ``score_size`` are scored: a larger
    search map is read from its top-left corner, and a template map that is
    not ``template_extent`` square fails to broadcast.

    :param template_features: Tensor of shape
        ``(B, template_extent, template_extent, C)``.
    :type template_features: keras.KerasTensor
    :param search_features: Tensor of shape ``(B, hx, wx, C)``.
    :type search_features: keras.KerasTensor
    :param score_size: Square response extent ``hx - hz + 1``.
    :type score_size: int
    :param template_extent: Square template feature extent.
    :type template_extent: int
    :param bias: Scalar bias broadcast over the map.
    :type bias: keras.KerasTensor
    :return: Response of shape ``(B, score_size, score_size, 1)``.
    :rtype: keras.KerasTensor
    """
    rows: List[keras.KerasTensor] = []
    for i in range(score_size):
        cols: List[keras.KerasTensor] = []
        for j in range(score_size):
            patch = search_features[
                :, i : i + template_extent, j : j + template_extent, :
            ]
            value = ops.sum(template_features * patch, axis=(1, 2, 3))
            cols.append(value)
        rows.append(ops.stack(cols, axis=1))
    response = ops.stack(rows, axis=1)
    response = ops.expand_dims(response, axis=-1) + bias
    return response


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.siamfc.model")
class SiamFCBackbone(keras.layers.Layer):
    """Embed an image with the SiamFC AlexNet-style trunk, total stride 8.

    Five ``valid`` convolutions and two 3x3/2 max pools. ReLU follows every
    convolution but ``conv5``, which is linear; BatchNorm sits between the
    convolution and the ReLU when ``use_batch_norm`` is set. The 1606.05233
    baseline keeps the second pool at stride 1 (total stride 4); this layer
    does not.

    Block internals:

    .. code-block:: text

        input [B, H, W, 3]
                 │
                 ▼
        ┌─────────────────────┐
        │ conv1  11x11/2, 96  │  bn1*, relu
        └─────────────────────┘
                 │  59 / 123
                 ▼
        ┌─────────────────────┐
        │ pool1  3x3/2        │
        └─────────────────────┘
                 │  29 / 61
                 ▼
        ┌─────────────────────┐
        │ conv2  5x5/1, 256   │  bn2*, relu
        └─────────────────────┘
                 │  25 / 57
                 ▼
        ┌─────────────────────┐
        │ pool2  3x3/2        │
        └─────────────────────┘
                 │  12 / 28
                 ▼
        ┌─────────────────────┐
        │ conv3  3x3/1, 384   │  bn3*, relu
        └─────────────────────┘
                 │  10 / 26
                 ▼
        ┌─────────────────────┐
        │ conv4  3x3/1, 384   │  bn4*, relu
        └─────────────────────┘
                 │  8 / 24
                 ▼
        ┌─────────────────────┐
        │ conv5  3x3/1, 256   │  linear
        └─────────────────────┘
                 │  6 / 22
                 ▼
        output [B, H', W', 256]

    Edges give the extent for a 127 / 255 input. * only when use_batch_norm.

    :param use_batch_norm: Apply BatchNorm after the first four convolutions.
        When off, ``bn1`` to ``bn4`` are still created but never built or
        called.
    :type use_batch_norm: bool
    :param bn_momentum: BatchNorm momentum in the Keras convention (torch 0.1
        maps to 0.9 here). Must be in ``(0, 1)``.
    :type bn_momentum: float
    :param bn_epsilon: BatchNorm epsilon; the default 1e-5 is the torch
        default. Must be positive.
    :type bn_epsilon: float
    :param kernel_initializer: Initializer for all convolutions.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param **kwargs: Passed to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        use_batch_norm: bool = True,
        bn_momentum: float = 0.9,
        bn_epsilon: float = 1e-5,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        """Validate arguments and create the sub-layers.

        Arguments are documented on the class.

        :raises ValueError: If ``bn_momentum`` is outside ``(0, 1)`` or
            ``bn_epsilon`` is not positive.
        """
        super().__init__(**kwargs)
        if not 0.0 < bn_momentum < 1.0:
            raise ValueError(f"bn_momentum must be in (0, 1), got {bn_momentum}")
        if bn_epsilon <= 0.0:
            raise ValueError(f"bn_epsilon must be positive, got {bn_epsilon}")
        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.kernel_initializer = kernel_initializer

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
            # The factory default epsilon is 1e-6; passing bn_epsilon overrides it.
            return create_normalization_layer(
                "batch_norm",
                momentum=self.bn_momentum,
                epsilon=self.bn_epsilon,
                name=name,
            )

        self.conv1 = _conv(96, 11, 2, "conv1")
        self.bn1 = _bn("bn1")
        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="pool1")
        self.conv2 = _conv(256, 5, 1, "conv2")
        self.bn2 = _bn("bn2")
        self.pool2 = layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="pool2")
        self.conv3 = _conv(384, 3, 1, "conv3")
        self.bn3 = _bn("bn3")
        self.conv4 = _conv(384, 3, 1, "conv4")
        self.bn4 = _bn("bn4")
        self.conv5 = _conv(BACKBONE_OUT_CHANNELS, 3, 1, "conv5")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build each sub-layer in order, threading the shape through.

        BatchNorm layers are built only when ``use_batch_norm`` is set.

        :param input_shape: Shape ``(B, H, W, 3)``.
        :type input_shape: tuple
        """
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
        :param training: Forwarded to the BatchNorm layers.
        :type training: bool or None
        :return: Features of shape ``(B, H', W', 256)``.
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
        """Return the feature shape for a square input shape.

        Only ``input_shape[1]`` is read; the output height and width both
        take its feature extent. A ``None`` channel axis stays ``None``.

        :param input_shape: Shape ``(B, H, W, C)`` with ``H == W``.
        :type input_shape: tuple
        :return: Shape ``(B, H', H', 256)``.
        :rtype: tuple
        """
        extent = input_shape[1]
        channels = input_shape[-1]
        if extent is None:
            height_out: Optional[int] = None
        else:
            height_out = siamfc_backbone_feature_size(int(extent))
        return (input_shape[0], height_out, height_out, BACKBONE_OUT_CHANNELS if channels is not None else None)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Base config plus every constructor argument, with
            ``kernel_initializer`` serialized.
        :rtype: dict
        """
        config = super().get_config()
        config.update(
            {
                "use_batch_norm": self.use_batch_norm,
                "bn_momentum": self.bn_momentum,
                "bn_epsilon": self.bn_epsilon,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SiamFCBackbone":
        """Rebuild the layer, deserializing a dict ``kernel_initializer``.

        ``config`` is modified in place.

        :param config: Output of :meth:`get_config`.
        :type config: dict
        :return: A new layer.
        :rtype: SiamFCBackbone
        """
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        return cls(**config)


@register_dl_technique("dl_techniques.models.siamfc.model")
class SiamFC(keras.Model):
    """Score every offset of an exemplar inside a search region.

    One :class:`SiamFCBackbone` embeds both inputs. The exemplar features act
    as the correlation filter and the search features as the signal, giving
    a single-channel map of logits, not probabilities. A learned scalar
    ``response_bias``, zero-initialized, is added to the map; the PyTorch
    port normalizes it with a 1-channel BatchNorm instead. The backbone runs
    with its default ``bn_momentum`` and ``bn_epsilon``; this class does not
    expose them. Inputs must match ``exemplar_size`` and ``search_size``;
    ``call`` does not check this.

    Architecture:

    .. code-block:: text

        z [B, Ez, Ez, 3]         x [B, Sx, Sx, 3]
                │                        │
                ▼                        ▼
        ┌──────────────────┐     ┌──────────────────┐
        │ backbone         │     │ backbone         │
        │ (shared weights) │     │ (shared weights) │
        └──────────────────┘     └──────────────────┘
                │  [B, hz, hz, 256]      │  [B, hx, hx, 256]
                └───────────┬────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ _cross_correlate    │◄── response_bias (scalar)
                 └─────────────────────┘
                            │
                            ▼
               logits [B, score, score, 1]

    Defaults 127 / 255 give hz = 6, hx = 22, score = 17.

    :param exemplar_size: Square exemplar extent in pixels. Must be positive.
    :type exemplar_size: int
    :param search_size: Square search extent in pixels. Must exceed
        ``exemplar_size``.
    :type search_size: int
    :param use_batch_norm: Forwarded to the shared backbone.
    :type use_batch_norm: bool
    :param kernel_initializer: Forwarded to the shared backbone.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param **kwargs: Passed to ``keras.Model``.
    """

    def __init__(
        self,
        exemplar_size: int = EXEMPLAR_SIZE,
        search_size: int = SEARCH_SIZE,
        use_batch_norm: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        """Validate the sizes, derive the extents and create the backbone.

        Arguments are documented on the class.

        :raises ValueError: If ``exemplar_size`` is not positive,
            ``search_size`` does not exceed it, or either feature extent or
            the score extent is not positive.
        """
        super().__init__(**kwargs)
        if exemplar_size <= 0:
            raise ValueError(f"exemplar_size must be positive, got {exemplar_size}")
        if search_size <= exemplar_size:
            raise ValueError(
                f"search_size ({search_size}) must exceed exemplar_size ({exemplar_size})"
            )
        template_extent = siamfc_backbone_feature_size(exemplar_size)
        search_extent = siamfc_backbone_feature_size(search_size)
        score = search_extent - template_extent + 1
        if template_extent <= 0 or search_extent <= 0 or score <= 0:
            raise ValueError(
                f"size pair ({exemplar_size}, {search_size}) collapses the backbone "
                f"(features {template_extent} vs {search_extent}, score {score})"
            )
        self.exemplar_size = exemplar_size
        self.search_size = search_size
        self.use_batch_norm = use_batch_norm
        self.kernel_initializer = kernel_initializer
        self._template_extent = template_extent
        self._search_extent = search_extent
        self._score_size = score

        self.backbone = SiamFCBackbone(
            use_batch_norm=use_batch_norm,
            kernel_initializer=kernel_initializer,
            name="backbone",
        )
        self.response_bias = None

    def build(self, input_shape: Any) -> None:
        """Create ``response_bias`` and build the sub-layers.

        Returns at once if the model is already built.

        :param input_shape: Pair of exemplar and search shapes.
        :type input_shape: list or tuple
        """
        if self.built:
            return
        self.response_bias = self.add_weight(
            name="response_bias", shape=(), initializer="zeros", trainable=True
        )
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self, inputs: Union[Tuple[Any, Any], List[Any]], training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Correlate an exemplar batch against a search batch.

        :param inputs: Pair ``(z, x)`` with shapes ``(B, Ez, Ez, 3)`` and
            ``(B, Sx, Sx, 3)``, where ``Ez == exemplar_size`` and
            ``Sx == search_size``. Not checked here: a wrong ``Ez`` fails in
            the correlation, a larger ``Sx`` is scored from its top-left
            corner only.
        :type inputs: tuple or list of two tensors
        :param training: Forwarded to the backbone.
        :type training: bool or None
        :return: Logit response of shape ``(B, score, score, 1)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``inputs`` is not a tuple or list of two.
        """
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("SiamFC expects a (exemplar, search) pair of two tensors")
        exemplar, search = inputs[0], inputs[1]
        template_features = self.backbone(exemplar, training=training)
        search_features = self.backbone(search, training=training)
        return _cross_correlate(
            template_features,
            search_features,
            self._score_size,
            self._template_extent,
            self.response_bias,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Base config plus every constructor argument, with
            ``kernel_initializer`` serialized.
        :rtype: dict
        """
        config = super().get_config()
        config.update(
            {
                "exemplar_size": self.exemplar_size,
                "search_size": self.search_size,
                "use_batch_norm": self.use_batch_norm,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SiamFC":
        """Rebuild the model, deserializing a dict ``kernel_initializer``.

        ``config`` is modified in place.

        :param config: Output of :meth:`get_config`.
        :type config: dict
        :return: A new model.
        :rtype: SiamFC
        """
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        return cls(**config)


# ---------------------------------------------------------------------


def create_siamfc(
    exemplar_size: int = EXEMPLAR_SIZE,
    search_size: int = SEARCH_SIZE,
    use_batch_norm: bool = True,
    kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> SiamFC:
    """Create a :class:`SiamFC` tracker and optionally load local weights.

    A string ``pretrained`` builds the model on the configured sizes, then
    loads the checkpoint with ``strict=True``; loader errors propagate.

    :param exemplar_size: Square exemplar extent.
    :type exemplar_size: int
    :param search_size: Square search extent.
    :type search_size: int
    :param use_batch_norm: Forwarded to the shared backbone.
    :type use_batch_norm: bool
    :param kernel_initializer: Forwarded to the shared backbone.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param pretrained: Path to a local checkpoint, or ``False`` to skip
        loading. ``True`` raises.
    :type pretrained: bool or str
    :param **kwargs: Additional arguments for :class:`SiamFC`.
    :return: A :class:`SiamFC` instance.
    :rtype: SiamFC
    :raises ValueError: If :class:`SiamFC` rejects the sizes.
    :raises NotImplementedError: If ``pretrained`` is ``True``.
    """
    model = SiamFC(
        exemplar_size=exemplar_size,
        search_size=search_size,
        use_batch_norm=use_batch_norm,
        kernel_initializer=kernel_initializer,
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
            report = load_weights_from_checkpoint(target=model, ckpt_path=pretrained, strict=True)
            logger.info(report.summary_string())
        else:
            raise NotImplementedError(
                "No pretrained SiamFC weights are distributed with dl_techniques "
                f"(requested exemplar {exemplar_size}, search {search_size}). Pass a local "
                "checkpoint instead: create_siamfc(pretrained='/path/to/weights.keras')."
            )
    else:
        logger.info(
            f"Created SiamFC (exemplar={exemplar_size}, search={search_size}, "
            f"score={model._score_size})."
        )
    return model

# ---------------------------------------------------------------------
