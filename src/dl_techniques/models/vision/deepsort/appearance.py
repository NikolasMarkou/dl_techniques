"""DeepSORT appearance embedding network, the ``mars-small128`` transcription.

A small residual CNN maps 128x64 person crops to L2-normalized 128-dim
appearance descriptors, `f = normalize(ELU(BN(W . flatten(x))))`, trained so
that cosine distance separates identities. The trunk is two 3x3 convolutions
(32 channels) plus a 2x2 max-pool, followed by three residual stages
(32, 64, 128 channels, the latter two strided), a 128-unit bottleneck with
BatchNorm and ELU, and a final row-wise L2 normalization -- the descriptor
the tracker gallery compares with cosine distance. Every convolution uses
ELU (not ReLU), truncated-normal init (stddev 1e-3), zero biases and L2
1e-8; every BatchNorm uses momentum 0.999 / epsilon 1e-3 (the TF-slim
defaults, not this repo's usual 0.9 / 1e-5). Dropout (rate 0.4) sits after
the first convolution of each residual inner block and before the
bottleneck, active only in training mode.

With ``include_top=True`` a :class:`CosineClassifier` head appends the
training-time cosine-softmax logits, `logits = softplus(s) * F . normalize(W)`,
with a learned scalar scale (L2 1e-1, init 0) and column-normalized class
vectors -- the WACV 2018 cosine-metric-learning parameterization, whose loss
is plain softmax cross-entropy over identities.

Init-scale note: the transcribed truncated-normal 1e-3 init makes
inference-mode activations vanish at random weights (each conv shrinks the
signal ~60x and fresh BatchNorm statistics cannot rescue it), so untrained
features have near-zero norm. In training mode batch statistics normalize
every stage and descriptors are exactly unit-norm; trained weights behave
the same at inference. The unit-norm guard therefore runs in training mode.

References:
    - Wojke et al., 2017. Simple Online and Realtime Tracking with a Deep
      Association Metric. ICIP 2017. (https://arxiv.org/abs/1703.07402)
    - Wojke and Bewley, 2018. Deep Cosine Metric Learning for Person
      Re-identification. WACV 2018.
      (http://openaccess.thecvf.com/content_WACV_2018/html/Wojke_Deep_Cosine_Metric_WACV_2018_paper.html)
    - Reference transcriptions: https://github.com/nwojke/deep_sort
      (``deep_sort_app.py`` tracker) and
      https://github.com/nwojke/cosine_metric_learning
      (``nets/deep_sort/network_definition.py`` + ``residual_net.py``,
      the source of every width, stride, init and epsilon above)
"""

import keras
from keras import layers, ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------
# transcribed constants (single source of truth)
# ---------------------------------------------------------------------

#: Network input: 128 rows, 64 columns, RGB in [0, 1] (ref ``IMAGE_SHAPE``).
INPUT_SHAPE = (128, 64, 3)

#: Descriptor width (ref ``fc1`` units and final conv channels).
EMBEDDING_DIM = 128

#: Transcribed initializers/regularizers (ref truncated-normal 1e-3, L2 1e-8).
TRUNCATED_INIT_STDDEV = 1e-3
KERNEL_L2 = 1e-8

#: Transcribed BatchNorm (TF-slim defaults, NOT the repo 0.9 / 1e-5).
BN_MOMENTUM = 0.999
BN_EPSILON = 1e-3

#: Transcribed dropout (TF keep_prob 0.6).
DROPOUT_RATE = 0.4

#: Transcribed scale-variable L2 on the cosine head.
SCALE_L2 = 1e-1


def _truncated_init() -> keras.initializers.Initializer:
    """Transcribed kernel initializer (truncated normal, stddev 1e-3)."""
    return keras.initializers.TruncatedNormal(stddev=TRUNCATED_INIT_STDDEV)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.deepsort.appearance")
class DeepSortResidualBlock(keras.layers.Layer):
    """Pre-activation residual block of the mars-small128 trunk.

    The inner path runs ``Conv3x3 -> BN -> ELU -> Dropout -> Conv3x3
    (linear)``; a ``use_projection`` block doubles channels at stride 2 with
    a bias-free 1x1 projection shortcut, otherwise the shortcut is identity.
    The ``is_first`` block (conv2_1) skips the pre-activation, matching the
    reference exactly.

    :param filters: Output channels (32, 64 or 128 in the trunk).
    :type filters: int
    :param stride: Stride of the inner first convolution (1 or 2).
    :type stride: int
    :param is_first: Skip the pre-activation (first block of the trunk).
    :type is_first: bool
    :param use_projection: Use a 1x1 strided projection shortcut.
    :type use_projection: bool
    :param dropout_rate: Dropout rate inside the inner path.
    :type dropout_rate: float
    :param kwargs: Passthrough to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        filters: int,
        stride: int = 1,
        is_first: bool = False,
        use_projection: bool = False,
        dropout_rate: float = DROPOUT_RATE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        self.filters = filters
        self.stride = stride
        self.is_first = is_first
        self.use_projection = use_projection
        self.dropout_rate = dropout_rate

        # BatchNorms go through the norms factory, which honors an explicitly
        # passed epsilon (its 1e-6 is only the default): the transcribed
        # TF-slim 1e-3 is kept.
        self.pre_bn = create_normalization_layer(
            "batch_norm", momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="pre_bn"
        )
        self.conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=True,
            kernel_initializer=_truncated_init(),
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.L2(KERNEL_L2),
            name="conv1",
        )
        self.bn1 = create_normalization_layer(
            "batch_norm", momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="bn1"
        )
        self.drop = layers.Dropout(dropout_rate, name="drop")
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            kernel_initializer=_truncated_init(),
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.L2(KERNEL_L2),
            name="conv2",
        )
        # Created always for layout stability; built only when used and
        # pinned by the no-projection-builds-no-projection-weights test.
        self.projection = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=_truncated_init(),
            kernel_regularizer=keras.regularizers.L2(KERNEL_L2),
            name="projection",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        # Linear shape walk mirroring call(): pre-act, conv1, bn1, drop, conv2.
        pre_shape = input_shape
        if not self.is_first:
            self.pre_bn.build(input_shape)
            pre_shape = self.pre_bn.compute_output_shape(input_shape)
        self.conv1.build(pre_shape)
        conv_shape = self.conv1.compute_output_shape(pre_shape)
        self.bn1.build(conv_shape)
        self.drop.build(conv_shape)
        self.conv2.build(conv_shape)
        if self.use_projection:
            self.projection.build(input_shape)
        super().build(input_shape)

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the pre-activation residual block.

        :param inputs: Tensor ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to BatchNorm and Dropout explicitly.
        :type training: bool or None
        :return: Tensor ``(B, H/stride, W/stride, filters)``.
        :rtype: keras.KerasTensor
        """
        h = inputs
        if not self.is_first:
            h = self.pre_bn(h, training=training)
            h = ops.elu(h)
        y = self.conv1(h)
        y = self.bn1(y, training=training)
        y = ops.elu(y)
        y = self.drop(y, training=training)
        y = self.conv2(y)
        shortcut = self.projection(inputs) if self.use_projection else inputs
        return shortcut + y

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        height = None if input_shape[1] is None else (int(input_shape[1]) + self.stride - 1) // self.stride
        width = None if input_shape[2] is None else (int(input_shape[2]) + self.stride - 1) // self.stride
        return (input_shape[0], height, width, self.filters)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "stride": self.stride,
                "is_first": self.is_first,
                "use_projection": self.use_projection,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@register_dl_technique("dl_techniques.models.deepsort.appearance")
class CosineClassifier(keras.layers.Layer):
    """Cosine-softmax training head (the reference ``ball`` layer).

    Holds column-normalized class mean vectors and a learned scalar scale
    (softplus of a zero-init variable, L2 1e-1): ``logits = softplus(s) *
    F . normalize(W)`` over L2-normalized features. Plain softmax
    cross-entropy over these logits is the default training objective.

    :param num_classes: Number of training identities, positive.
    :type num_classes: int
    :param embedding_dim: Feature width the head reads.
    :type embedding_dim: int
    :param kwargs: Passthrough to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = EMBEDDING_DIM,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.class_vectors = None
        self.raw_scale = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if input_shape[-1] is not None and int(input_shape[-1]) != self.embedding_dim:
            raise ValueError(
                f"CosineClassifier expects {self.embedding_dim} features, "
                f"got {input_shape[-1]}"
            )
        self.class_vectors = self.add_weight(
            name="mean_vectors",
            shape=(self.embedding_dim, self.num_classes),
            initializer=keras.initializers.TruncatedNormal(stddev=TRUNCATED_INIT_STDDEV),
            regularizer=None,
            trainable=True,
        )
        self.raw_scale = self.add_weight(
            name="scale",
            shape=(),
            initializer="zeros",
            regularizer=keras.regularizers.L2(SCALE_L2),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Compute scaled cosine logits.

        :param inputs: L2-normalized features ``(B, embedding_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Unused (no stochastic layers); accepted for uniformity.
        :type training: bool or None
        :return: Logits ``(B, num_classes)``.
        :rtype: keras.KerasTensor
        """
        scale = ops.softplus(self.raw_scale)
        normalized = self.class_vectors / ops.maximum(
            ops.sqrt(ops.sum(ops.square(self.class_vectors), axis=0, keepdims=True)),
            keras.config.epsilon(),
        )
        return scale * ops.matmul(inputs, normalized)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return (input_shape[0], self.num_classes)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {"num_classes": self.num_classes, "embedding_dim": self.embedding_dim}
        )
        return config


@register_dl_technique("dl_techniques.models.deepsort.appearance")
class DeepSortAppearanceNet(keras.Model):
    """mars-small128 person appearance embedding network.

    Two 3x3 convolutions (32) plus a 3x3/2 pool feed six residual blocks
    (32, 32, 64/2, 64, 128/2, 128); flatten, dropout, a 128-unit ELU
    bottleneck with BatchNorm and a row-wise L2 normalization emit the
    128-dim descriptor. With ``include_top=True`` the cosine classifier
    appends training logits and the model returns ``(features, logits)``;
    otherwise it returns the features alone. This package has no
    ``MODEL_VARIANTS`` table: the reference ships one network.

    :param include_top: Whether to append the cosine classifier.
    :type include_top: bool
    :param num_classes: Training identities for the head; required when
        ``include_top`` is True.
    :type num_classes: int or None
    :param dropout_rate: Dropout rate (transcribed 0.4).
    :type dropout_rate: float
    :param kwargs: Passthrough to ``keras.Model``.
    """

    def __init__(
        self,
        include_top: bool = False,
        num_classes: Optional[int] = None,
        dropout_rate: float = DROPOUT_RATE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if include_top and (num_classes is None or num_classes <= 0):
            raise ValueError("num_classes must be positive when include_top is True")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        self.include_top = include_top
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        def _conv(filters: int, name: str) -> layers.Conv2D:
            return layers.Conv2D(
                filters=filters,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=True,
                kernel_initializer=_truncated_init(),
                bias_initializer="zeros",
                kernel_regularizer=keras.regularizers.L2(KERNEL_L2),
                name=name,
            )

        def _bn(name: str) -> keras.layers.Layer:
            # Routed through the norms factory, which honors an explicitly
            # passed epsilon (its 1e-6 is only the default): the transcribed
            # TF-slim 1e-3 is kept.
            return create_normalization_layer(
                "batch_norm", momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name=name
            )

        self.conv1_1 = _conv(32, "conv1_1")
        self.bn1_1 = _bn("bn1_1")
        self.conv1_2 = _conv(32, "conv1_2")
        self.bn1_2 = _bn("bn1_2")
        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="pool1")
        self.block2_1 = DeepSortResidualBlock(32, stride=1, is_first=True, name="block2_1")
        self.block2_3 = DeepSortResidualBlock(32, name="block2_3")
        self.block3_1 = DeepSortResidualBlock(64, stride=2, use_projection=True, name="block3_1")
        self.block3_3 = DeepSortResidualBlock(64, name="block3_3")
        self.block4_1 = DeepSortResidualBlock(128, stride=2, use_projection=True, name="block4_1")
        self.block4_3 = DeepSortResidualBlock(128, name="block4_3")
        self.head_drop = layers.Dropout(dropout_rate, name="head_drop")
        self.fc1 = layers.Dense(
            EMBEDDING_DIM,
            kernel_initializer=_truncated_init(),
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.L2(KERNEL_L2),
            name="fc1",
        )
        self.bn_fc1 = _bn("bn_fc1")
        # Created always for layout stability; built only when include_top and
        # pinned by the no-top-builds-no-head-weights test.
        self.classifier = CosineClassifier(
            num_classes if num_classes is not None else 1, name="cosine_head"
        )

    def build(self, input_shape: Any) -> None:
        # Explicit walk (not the symbolic trace): the always-created head
        # must stay unbuilt when include_top is False, and a trace would
        # materialize it. Order mirrors call() exactly.
        if self.built:
            return
        shape = input_shape
        self.conv1_1.build(shape)
        shape = self.conv1_1.compute_output_shape(shape)
        self.bn1_1.build(shape)
        self.conv1_2.build(shape)
        shape = self.conv1_2.compute_output_shape(shape)
        self.bn1_2.build(shape)
        self.pool1.build(shape)
        shape = self.pool1.compute_output_shape(shape)
        for block in (
            self.block2_1, self.block2_3, self.block3_1,
            self.block3_3, self.block4_1, self.block4_3,
        ):
            block.build(shape)
            shape = block.compute_output_shape(shape)
        flat_dim = None if shape[1] is None else int(shape[1]) * int(shape[2]) * int(shape[3])
        flat_shape = (shape[0], flat_dim)
        self.head_drop.build(flat_shape)
        self.fc1.build(flat_shape)
        fc_shape = self.fc1.compute_output_shape(flat_shape)
        self.bn_fc1.build(fc_shape)
        if self.include_top:
            self.classifier.build(fc_shape)
        super().build(input_shape)

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ):
        """Embed a batch of person crops (and optionally classify them).

        :param inputs: Tensor ``(B, 128, 64, 3)`` in [0, 1].
        :type inputs: keras.KerasTensor
        :param training: Forwarded to BatchNorm and Dropout explicitly.
        :type training: bool or None
        :return: Features ``(B, 128)``, or ``(features, logits)`` with
            ``include_top``.
        """
        x = self.conv1_1(inputs)
        x = self.bn1_1(x, training=training)
        x = ops.elu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x, training=training)
        x = ops.elu(x)
        x = self.pool1(x)
        x = self.block2_1(x, training=training)
        x = self.block2_3(x, training=training)
        x = self.block3_1(x, training=training)
        x = self.block3_3(x, training=training)
        x = self.block4_1(x, training=training)
        x = self.block4_3(x, training=training)
        x = ops.reshape(x, (ops.shape(x)[0], -1))
        x = self.head_drop(x, training=training)
        x = self.fc1(x)
        x = self.bn_fc1(x, training=training)
        x = ops.elu(x)
        features = x / ops.maximum(
            ops.sqrt(ops.sum(ops.square(x), axis=1, keepdims=True)),
            keras.config.epsilon(),
        )
        if self.include_top:
            return features, self.classifier(features)
        return features

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "include_top": self.include_top,
                "num_classes": self.num_classes,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------


def create_deepsort_embedding(
    include_top: bool = False,
    num_classes: Optional[int] = None,
    dropout_rate: float = DROPOUT_RATE,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> DeepSortAppearanceNet:
    """Create a :class:`DeepSortAppearanceNet` with no logic beyond construction.

    :param include_top: Whether to append the cosine classifier.
    :type include_top: bool
    :param num_classes: Training identities, required with ``include_top``.
    :type num_classes: int or None
    :param dropout_rate: Dropout rate.
    :type dropout_rate: float
    :param pretrained: A local checkpoint path, or True to raise since no
        weights ship here.
    :type pretrained: bool or str
    :param kwargs: Additional arguments for :class:`DeepSortAppearanceNet`.
    :return: A :class:`DeepSortAppearanceNet` instance.
    :rtype: DeepSortAppearanceNet
    :raises NotImplementedError: If ``pretrained`` is True.
    """
    model = DeepSortAppearanceNet(
        include_top=include_top,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        **kwargs,
    )
    if pretrained:
        if isinstance(pretrained, str):
            if not model.built:
                model.build((None,) + INPUT_SHAPE)
            report = load_weights_from_checkpoint(target=model, ckpt_path=pretrained, strict=True)
            logger.info(report.summary_string())
        else:
            raise NotImplementedError(
                "No pretrained DeepSORT weights are distributed with dl_techniques. "
                "Pass a local checkpoint instead: "
                "create_deepsort_embedding(pretrained='/path/to/weights.keras')."
            )
    else:
        logger.info(
            f"Created DeepSortAppearanceNet (include_top={include_top})."
        )
    return model
