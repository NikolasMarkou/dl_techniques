"""SiamFC, a fully-convolutional Siamese tracker, plus the ``create_siamfc`` factory.

Tracking is reframed as similarity learning: one shared embedding network maps
both the small exemplar patch ``z`` and the larger search region ``x`` into a
common feature space, and a cross-correlation ``f(z) * f(x)`` emits a
single-channel score map whose peak marks the target, `response = corr(f(z),
f(x)) + bias`. Because the embedding is fully convolutional with total stride
8 and no padding, a larger search image yields a larger score map with no
architectural change, so translation search is a single forward pass rather
than an enumeration of shifted crops. The backbone is the paper's AlexNet
variant (11x11/2, 5x5, three 3x3 stages, two 3x3/2 pools, ``valid`` everywhere)
with BatchNorm added as in the modern ports; the final convolution is linear.
With the paper sizes (exemplar 127, search 255) the features are 6x6 and
22x22 over 256 channels and the response is 17x17. Training uses centered
pairs with a radius-based logistic label; inference adds a 3-scale pyramid, a
Hann window and a scale penalty outside the graph. This package has no
``MODEL_VARIANTS`` table because the paper ships one architecture; width and
input sizes are constructor arguments instead.

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

# ---------------------------------------------------------------------
# constants and pure shape helpers (single source of truth)
# ---------------------------------------------------------------------

EXEMPLAR_SIZE = 127
SEARCH_SIZE = 255
TOTAL_STRIDE = 8
SCORE_SIZE = 17
BACKBONE_OUT_CHANNELS = 256


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


def siamfc_backbone_feature_size(input_size: int) -> int:
    """Spatial extent of the SiamFC embedding for a square input.

    Applies the backbone reduction chain ``conv11/2, pool3/2, conv5/1,
    pool3/2, conv3/1, conv3/1, conv3/1`` with ``valid`` padding throughout,
    which is the single encoding of this architecture's geometry used by
    ``build()``, ``call()`` and ``compute_output_shape`` alike.

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


def siamfc_score_size(exemplar_size: int = EXEMPLAR_SIZE, search_size: int = SEARCH_SIZE) -> int:
    """Response-map extent for an exemplar/search size pair.

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


def create_hann_window(score_size: int) -> np.ndarray:
    """Separable 2D Hann window penalizing large displacements at inference.

    Pure NumPy post-processing outside the graph, matching the paper's
    ``response = (1 - w) * response + w * hann`` blending.

    :param score_size: Square score-map extent.
    :type score_size: int
    :return: Array of shape ``(score_size, score_size)`` in ``float32``.
    :rtype: numpy.ndarray
    """
    if score_size <= 0:
        raise ValueError(f"score_size must be positive, got {score_size}")
    window_1d = np.hanning(score_size).astype("float32")
    return np.outer(window_1d, window_1d).astype("float32")


def _cross_correlate(
    template_features: keras.KerasTensor,
    search_features: keras.KerasTensor,
    score_size: int,
    template_extent: int,
    bias: keras.KerasTensor,
) -> keras.KerasTensor:
    """Single-channel batched cross-correlation over static extents.

    For each batch element and each valid offset, dots the template feature
    volume against the aligned search patch and adds a scalar bias. The loop
    bounds are Python ints derived from the constructor configuration, so the
    body stays symbolic: slicing with integer bounds and ``ops.sum`` /
    ``ops.stack`` trace cleanly.

    :param template_features: Tensor of shape ``(B, hz, wz, C)``.
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
    """Shared SiamFC embedding trunk with total stride 8.

    The paper's AlexNet variant with ``valid`` padding everywhere: 11x11/2
    (96) + pool 3x3/2, 5x5 (256) + pool 3x3/2, then 384, 384, 256 over 3x3.
    BatchNorm + ReLU follow every convolution except the last, which is
    linear so the correlation reads raw embedding magnitudes. The two
    MaxPool stages give the total stride of 8; the reference baseline that
    keeps the second pool at stride 1 (total stride 4, 33x33 map) is a
    different experiment and is not reproduced here.

    :param use_batch_norm: Whether to keep BatchNorm after the first four convolutions.
    :type use_batch_norm: bool
    :param bn_momentum: BatchNorm momentum in the Keras convention (torch 0.1 maps to 0.9 here).
    :type bn_momentum: float
    :param bn_epsilon: BatchNorm epsilon, matching the torch default rather than the Keras one.
    :type bn_epsilon: float
    :param kernel_initializer: Initializer for all convolutions.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Passthrough to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        use_batch_norm: bool = True,
        bn_momentum: float = 0.9,
        bn_epsilon: float = 1e-5,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
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

        def _bn(name: str) -> layers.BatchNormalization:
            return layers.BatchNormalization(
                momentum=self.bn_momentum, epsilon=self.bn_epsilon, name=name
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
        extent = input_shape[1]
        channels = input_shape[-1]
        if extent is None:
            height_out: Optional[int] = None
        else:
            height_out = siamfc_backbone_feature_size(int(extent))
        return (input_shape[0], height_out, height_out, BACKBONE_OUT_CHANNELS if channels is not None else None)

    def get_config(self) -> Dict[str, Any]:
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
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(config["kernel_initializer"])
        return cls(**config)


@register_dl_technique("dl_techniques.models.siamfc.model")
class SiamFC(keras.Model):
    """Fully-convolutional Siamese tracker over an exemplar/search pair.

    One shared :class:`SiamFCBackbone` embeds both inputs; the exemplar
    becomes the correlation filter and the search region the signal, so the
    output is a single-channel similarity map. ``call()`` takes a ``(z, x)``
    pair of image batches and returns ``(B, score, score, 1)`` logits trained
    with a radius-based logistic label, not probabilities. The response carries
    a learned scalar bias; the PyTorch reference port instead normalizes the
    map with a 1-channel BatchNorm (``match_batchnorm``). The bias form is the
    paper's parameterization and keeps inference independent of batch
    statistics, so it is kept deliberately.

    :param exemplar_size: Square exemplar extent in pixels.
    :type exemplar_size: int
    :param search_size: Square search extent in pixels, must exceed ``exemplar_size``.
    :type search_size: int
    :param use_batch_norm: Forwarded to the shared backbone.
    :type use_batch_norm: bool
    :param kernel_initializer: Forwarded to the shared backbone.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kwargs: Passthrough to ``keras.Model``.
    :raises ValueError: If the sizes are not positive, unordered, or collapse the score map.
    """

    def __init__(
        self,
        exemplar_size: int = EXEMPLAR_SIZE,
        search_size: int = SEARCH_SIZE,
        use_batch_norm: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
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

        :param inputs: Pair ``(z, x)`` with shapes ``(B, Ez, Ez, 3)`` and ``(B, Sx, Sx, 3)``.
        :type inputs: tuple or list of two tensors
        :param training: Training flag forwarded to the backbone explicitly.
        :type training: bool or None
        :return: Logit response of shape ``(B, score, score, 1)``.
        :rtype: keras.KerasTensor
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
    """Create a :class:`SiamFC` tracker with no logic of its own beyond construction.

    :param exemplar_size: Square exemplar extent.
    :type exemplar_size: int
    :param search_size: Square search extent.
    :type search_size: int
    :param use_batch_norm: Forwarded to the shared backbone.
    :type use_batch_norm: bool
    :param kernel_initializer: Forwarded to the shared backbone.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param pretrained: A local checkpoint path to load, or True to raise since no weights ship here.
    :type pretrained: bool or str
    :param kwargs: Additional arguments for :class:`SiamFC`.
    :return: A :class:`SiamFC` instance.
    :rtype: SiamFC
    :raises NotImplementedError: If ``pretrained`` is True.
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
