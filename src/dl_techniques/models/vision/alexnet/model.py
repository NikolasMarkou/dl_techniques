"""AlexNet (2012), the original ImageNet winner, with the ``create_alexnet`` factory.

This file holds the ``AlexNet`` Keras model and its factory. The architecture is
five convolutional stages -- each convolution followed by ReLU, with a local
response normalization and a max pool on the first two -- and three fully connected
stages.

Padding
-------

**The paper does not state its padding.** It lists kernel sizes and strides and
omits the parameter entirely, so there is no "faithful padding" to transcribe. This
port uses the configuration of the released Caffe model, which is the only place the
specification survives, and it is the configuration that reproduces the paper's own
Figure 3: ``conv1`` padded 2/stride 4, ``conv2``-``conv5`` padded 1, ``pool1`` and
``pool2`` padded 1, and ``pool3`` padded 0.

That choice is load-bearing and was verified rather than assumed. Measured at input
227, the padded configuration reaches ``6 x 6 x 256`` and gives fc6 the 9216 inputs
the paper states. The two obvious alternatives do not: strict ``padding='valid'``
everywhere reaches only ``2 x 2 x 256`` (fc6 in_features 1024), and all-``same``
convolutions reach ``7 x 7 x 256`` (fc6 in_features 12544). Keras agrees with this
arithmetic. The asymmetry of the released model -- padded pools, unpadded final
pool -- is exactly what produces the paper's figure.

Parameter count
---------------

Measured **60,597,608** at the default ``input_shape=(227, 227, 3)`` and
``num_classes=1000``: 1,891,712 convolutional, 74,752 of non-trainable LRN band
weights, and 58,631,144 fully connected. The "about 62 million" in circulation is
slightly high; this port prints its own measured figure rather than the folklore
one. **97%** of the total sits in the three fully connected layers, which is the
whole point of the architecture and also its main cost -- see the README for what
that means for fine-tuning.

Grouped convolutions
--------------------

``conv2`` through ``conv5`` use ``groups=2``. This is not decoration: the paper
splits each of those layers across two GPUs and only ever connects a channel to
half the preceding feature map. ``groups=2`` is the standard single-device
expression of that, and it is why these layers take 96/256/384/384/256 filters
rather than a doubled width.

References:
    -   Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet
        Classification with Deep Convolutional Neural Networks". NeurIPS 25.
        arXiv:1102.0183. Architecture Section 2, LRN Section 3.4, the
        ``n=5, alpha=1e-4, beta=0.75, k=1`` constants in Section 4.
        https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
    -   The released Caffe model, the source of the padding this port transcribes:
        https://github.com/isl-org/OpenSee
"""

import keras
from keras import layers
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.heads.vision import (
    create_vision_head,
    VisionTaskType,
    HeadConfiguration,
)
from .spatial_guard import (
    STAGES,
    final_feature_extent,
    minimum_spatial_extent,
    validate_spatial_extent,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Minimum spatial extent that keeps the feature map non-degenerate. Derived by
#: :func:`~.spatial_guard.minimum_spatial_extent` from the same stage table the
#: model is built from, so it cannot drift away from the architecture.
MIN_SPATIAL_EXTENT = minimum_spatial_extent()

#: The paper's local-response-normalization hyper-parameters: ``n = 5`` means a
#: depth radius of 2. Consumed by ``create_normalization_layer``, not constructed
#: directly, so the layer is reachable from a config dict.
LRN_HYPERPARAMETERS: Dict[str, Any] = {
    "depth_radius": 2,
    "alpha": 1e-4,
    "beta": 0.75,
    "k": 1.0,
}


@register_dl_technique("dl_techniques.models.alexnet.model")
class AlexNet(keras.Model):
    """The original AlexNet, with its local response normalization intact.

    Five convolutional stages, then three fully connected ones::

        227 x 227 x 3
          -> conv1 11x11/4  pad 2, 96    ->  56 x 56 x 96
          -> LRN, pool1 3x3/2 pad 1      ->  28 x 28 x 96
          -> conv2 5x5/1  pad 1, 256 g2  ->  26 x 26 x 256
          -> LRN, pool2 3x3/2 pad 1      ->  13 x 13 x 256
          -> conv3 3x3/1  pad 1, 384 g2  ->  13 x 13 x 384
          -> conv4 3x3/1  pad 1, 384 g2  ->  13 x 13 x 384
          -> conv5 3x3/1  pad 1, 256 g2  ->  13 x 13 x 256
          -> pool3 3x3/2 pad 0           ->   6 x 6 x 256
          -> flatten 9216
          -> fc6 4096, ReLU, dropout 0.5
          -> fc7 4096, ReLU, dropout 0.5
          -> fc8 1000, softmax

    Every output spatial extent above is measured, not copied from the paper's
    diagram, and each matches it.

    **No pretrained weights ship with this port.** ``pretrained=True`` raises
    ``NotImplementedError`` rather than returning random weights under that name.

    :param num_classes: Width of the final softmax. The paper uses 1000, one per
        ImageNet class.
    :type num_classes: int
    :param input_shape: Spatial shape plus channels. The default is the paper's
        227, which is the crop of a 256-pixel image. It is the paper's own number
        and is kept as the default, but it is not a special one: measured with this
        padding, inputs 224, 225, 226 and 227 all reach the same ``6 x 6 x 256`` and
        the same 9216-wide flatten, because the strides round down to the same
        extent. 256 gives 7 x 7 and 9216 becomes 12544.
    :type input_shape: Tuple[int, int, int]
    :param dropout_rate: Dropout applied after fc6 and fc7. The paper uses 0.5.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for all five convolutions.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for every conv and dense bias.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param include_top: Whether to build the three fully connected layers. With
        ``False`` the model ends at ``pool3`` and returns a ``(batch, h, w, 256)``
        feature map, which is the useful form for dense prediction.
    :type include_top: bool
    :param lrn_hyperparameters: Keyword arguments for the two LRN layers, passed
        through to ``create_normalization_layer('local_response_norm', ...)``. The
        default is the paper's.
    :type lrn_hyperparameters: Optional[Dict[str, Any]]
    :param name: Keras model name.
    :type name: Optional[str]
    :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :raises ValueError: If ``num_classes`` is not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0, 1)``.
    :raises ValueError: If ``input_shape`` is not a 3-tuple with a channels axis.
    :raises ValueError: If any spatial axis is below ``MIN_SPATIAL_EXTENT``
        (measured: 55). The final pool is ``'valid'``, so a smaller input yields an
        all-NaN feature map of the correct shape rather than an error.
    :raises NotImplementedError: Never from ``__init__``; see the class docstring.

    Example:

    .. code-block:: python

        from dl_techniques.models.vision.alexnet import AlexNet

        model = AlexNet(num_classes=1000)
        model.summary()

    .. code-block:: text

        Measured parameter breakdown at input_shape=(227, 227, 3):

        layer          output shape              params
        -------------  ------------------------  -----------
        conv1          (None, 56, 56, 96)            34,944
        lrn1           (None, 56, 56, 96)             9,216
        pool1          (None, 28, 28, 96)                 0
        conv2          (None, 26, 26, 256)          307,456
        lrn2           (None, 26, 26, 256)            65,536
        pool2          (None, 13, 13, 256)                 0
        conv3          (None, 13, 13, 384)          442,752
        conv4          (None, 13, 13, 384)          663,936
        conv5          (None, 13, 13, 256)          442,624
        pool3          (None, 6, 6, 256)                   0
        flatten        (None, 9216)                       0
        fc6            (None, 4096)               37,752,832
        fc7            (None, 4096)               16,781,312
        fc8            (None, 1000)                4,097,000
        ----------------------------------------------------
        Total                                     60,597,608

    ``conv3`` reads 442,752 rather than the 885,120 an ungrouped ``conv3`` would
    need, because it is grouped like the rest: 384 filters x (256/2) input channels
    x 9 + bias. That is the paper's 2-GPU split, not an omission.

    ``lrn1`` and ``lrn2`` are not free. Each LRN holds its neighbourhood band as a
    non-trainable ``C x C`` weight -- 96² for lrn1 and 256² for lrn2 -- because a
    bare tensor built in ``build()`` does not survive being built inside a parent's
    ``scratch_graph``. Those are 74,752 of the total and they are **not**
    learnable; ``count_params()`` counts them, ``trainable_parameters`` does not.
    Dropping them would give 60,522,856.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        input_shape: Tuple[int, int, int] = (227, 227, 3),
        dropout_rate: float = 0.5,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        include_top: bool = True,
        lrn_hyperparameters: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the model and create every sublayer.

        All sublayers are created here, unconditionally, so that a
        ``get_config()`` round trip reproduces the same architecture whatever the
        input rank turns out to be.

        :param num_classes: Width of the final softmax.
        :type num_classes: int
        :param input_shape: ``(height, width, channels)``.
        :type input_shape: Tuple[int, int, int]
        :param dropout_rate: Dropout after fc6 and fc7.
        :type dropout_rate: float
        :param kernel_initializer: Initializer for all convolutions.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param bias_initializer: Initializer for every bias.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param include_top: Whether to build the fully connected layers.
        :type include_top: bool
        :param lrn_hyperparameters: Keyword arguments for both LRN layers. ``None``
            uses the paper's values.
        :type lrn_hyperparameters: Optional[Dict[str, Any]]
        :param name: Keras model name.
        :type name: Optional[str]
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If any constructor argument is out of range.
        """
        super().__init__(name=name, **kwargs)

        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError(f"num_classes must be a positive int, got {num_classes!r}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be a 3-tuple (height, width, channels), got "
                f"{input_shape!r}"
            )

        validate_spatial_extent(
            (input_shape[0], input_shape[1]), model_label=type(self).__name__
        )

        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.include_top = include_top
        self.lrn_hyperparameters = dict(
            LRN_HYPERPARAMETERS if lrn_hyperparameters is None
            else lrn_hyperparameters
        )

        self._build_features()
        # The feature extent is a property of the CONVOLUTION stack, so it is
        # recorded even when include_top=False; otherwise get_feature_extent()
        # would raise on a model that has one perfectly well defined feature map.
        self._feature_extent = final_feature_extent(self.input_shape[0])
        if include_top:
            self._build_classifier()

        logger.info(
            f"AlexNet: input {self.input_shape}, num_classes={self.num_classes}, "
            f"include_top={include_top}, "
            f"min spatial extent {MIN_SPATIAL_EXTENT}"
        )

    def _build_features(self) -> None:
        """Create the five convolutional stages.

        The ``(filters, kernel, stride, groups)`` of every convolution is read from
        the same :data:`STAGES` table the spatial guard validates against, so the
        guard and the architecture cannot disagree about what the model does.
        """
        conv_spec = [
            ("conv1", 96, 11, 4, 1),
            ("conv2", 256, 5, 1, 2),
            ("conv3", 384, 3, 1, 2),
            ("conv4", 384, 3, 1, 2),
            ("conv5", 256, 3, 1, 2),
        ]
        padding_by_stage = {
            stage_name: pad for stage_name, _, _, pad, _ in STAGES
        }

        previous_filters = self.input_shape[2]

        # Which pool follows which conv, read from the guard's own table so the two
        # cannot disagree. conv5's pool is pool3, not pool5 -- stated here because the
        # numbering is the one genuinely confusing thing about this architecture.
        pool_after = {"conv1": "pool1", "conv2": "pool2", "conv5": "pool3"}

        for index, (layer_name, filters, kernel, stride, groups) in enumerate(
            conv_spec, start=1
        ):
            # groups must divide BOTH the input and output channel counts; the
            # paper's 2-way GPU split means conv2 reads half of conv1's 96 filters.
            if groups > 1 and (previous_filters % groups or filters % groups):
                raise ValueError(
                    f"{layer_name}: groups={groups} does not divide the input "
                    f"channels ({previous_filters}) or filters ({filters})"
                )

            # EXPLICIT symmetric padding, not padding='same'. Keras' 'same' splits an
            # odd total pad unevenly and yields ceil(in/stride), which at input 227
            # gives conv1 57 rather than the paper's 56 and propagates to a 7x7 final
            # map instead of 6x6. See spatial_guard.STAGES for the full measurement.
            self._pad(layer_name, padding_by_stage[layer_name])

            setattr(self, layer_name, layers.Conv2D(
                filters=filters,
                kernel_size=kernel,
                strides=stride,
                padding="valid",
                groups=groups,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                name=layer_name,
            ))

            setattr(self, f"relu{index}", layers.ReLU(name=f"relu{index}"))

            # LRN follows conv1 and conv2 only, BEFORE the pool -- the paper's
            # order. It is routed through the norms factory so the layer stays
            # reachable from a config dict.
            if layer_name in ("conv1", "conv2"):
                setattr(self, f"lrn{index}", create_normalization_layer(
                    "local_response_norm",
                    name=f"lrn{index}",
                    **self.lrn_hyperparameters,
                ))

            pool_name = pool_after.get(layer_name)
            if pool_name is not None:
                self._pad(pool_name, padding_by_stage[pool_name])
                setattr(self, pool_name, layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    padding="valid",
                    name=pool_name,
                ))

            previous_filters = filters

    def _pad(self, stage_name: str, pad: int) -> None:
        """Attach an explicit symmetric ``ZeroPadding2D`` to a stage, if it needs one.

        ``pool3`` has ``pad=0``, so no padding layer is created and no pass-through
        sits in the graph -- an identity layer that only adds a name and a node to
        every forward pass is noise.

        :param stage_name: Layer name; the padding layer is ``<stage_name>_pad``.
        :type stage_name: str
        :param pad: Symmetric padding to apply to both spatial axes.
        :type pad: int
        """
        if pad <= 0:
            return
        setattr(self, f"{stage_name}_pad", layers.ZeroPadding2D(
            padding=((pad, pad), (pad, pad)),
            name=f"{stage_name}_pad",
        ))

    def _build_classifier(self) -> None:
        """Create classification head using the vision heads factory."""
        feature_extent = self._feature_extent
        flat_features = 256 * feature_extent * feature_extent
        self._flat_features = flat_features

        head_config = HeadConfiguration.get_default_config(VisionTaskType.CLASSIFICATION)
        head_config.update({
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'normalization_type': 'batch_norm',
            'activation_type': 'relu',
            'use_global_pooling': False,  # AlexNet uses flatten, not pooling
            'use_attention': False,
            'use_ffn': True,
            'ffn_type': 'mlp',
            'ffn_expansion_factor': 1,  # 4096 -> 4096 (no expansion)
            'hidden_dim': 4096,  # fc6/fc7 width
        })

        self.classification_head = create_vision_head(
            VisionTaskType.CLASSIFICATION, **head_config
        )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer, then mark the model built.

        A subclass that creates its sub-layers in ``__init__`` must call this, or
        Keras' base ``build()`` marks the model built while every convolution is still
        unbuilt -- ``summary()`` then prints ``(unbuilt)`` shapes and
        ``count_params()`` returns **0**. Both were observed here before this method
        existed.

        :param input_shape: Shape of the input the model will be called with.
        :type input_shape: Any
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the forward pass.

        :param inputs: Image tensor of shape ``(batch, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag, forwarded to the dropout layers.
        :type training: Optional[bool]
        :return: Class probabilities of shape ``(batch, num_classes)``, or the
            ``(batch, h, w, 256)`` feature map when ``include_top=False``.
        :rtype: keras.KerasTensor
        """
        x = inputs

        x = self.conv1(self.conv1_pad(x))
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(self.pool1_pad(x))

        x = self.conv2(self.conv2_pad(x))
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(self.pool2_pad(x))

        x = self.relu3(self.conv3(self.conv3_pad(x)))
        x = self.relu4(self.conv4(self.conv4_pad(x)))
        x = self.relu5(self.conv5(self.conv5_pad(x)))
        x = self.pool3(x)  # pool3 is unpadded in the reference model

        if not self.include_top:
            return x

        # Factory head returns {'logits': ..., 'probabilities': ...}
        head_output = self.classification_head(x, training=training)
        return head_output['probabilities']

    def get_feature_extent(self) -> int:
        """Per-axis extent of the feature map entering the flatten.

        :return: The measured extent. 6 at the default input shape, and 6 for any
            input from 224 to 227.
        :rtype: int
        """
        return self._feature_extent

    def load_pretrained_weights(self, weights_path: str) -> None:
        """Load weights from a local checkpoint by path.

        :param weights_path: Path to a weights file.
        :type weights_path: str
        :raises NotImplementedError: Always. No pretrained weights ship with this
            port and none are downloaded.
        """
        raise NotImplementedError(
            "AlexNet: no pretrained weights are distributed with this port. Train "
            "from scratch, or convert an external checkpoint and load its weights "
            "by hand; do not expect `pretrained=True` to fetch anything."
        )

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration needed to rebuild this model.

        :return: Dictionary holding every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "include_top": self.include_top,
            "lrn_hyperparameters": dict(self.lrn_hyperparameters),
        })
        return config


# ---------------------------------------------------------------------


def create_alexnet(
        num_classes: int = 1000,
        input_shape: Tuple[int, int, int] = (227, 227, 3),
        dropout_rate: float = 0.5,
        include_top: bool = True,
        pretrained: Union[bool, str] = False,
        lrn_hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
) -> AlexNet:
    """Build an :class:`AlexNet`.

    :param num_classes: Width of the final softmax.
    :type num_classes: int
    :param input_shape: ``(height, width, channels)``. 227 is the paper's own
        extent; 224 through 227 give the identical ``6 x 6 x 256`` result.
    :type input_shape: Tuple[int, int, int]
    :param dropout_rate: Dropout after fc6 and fc7.
    :type dropout_rate: float
    :param include_top: Whether to build the fully connected layers.
    :type include_top: bool
    :param pretrained: Must be ``False``. No pretrained weights ship with this
        port, so ``True`` raises rather than returning random weights under that
        name. A string is treated the same way.
    :type pretrained: Union[bool, str]
    :param lrn_hyperparameters: Keyword arguments for both LRN layers.
    :type lrn_hyperparameters: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for :class:`AlexNet`.
    :type kwargs: Any

    :return: The configured model.
    :rtype: AlexNet

    :raises NotImplementedError: If ``pretrained`` is truthy.

    Example:

    .. code-block:: python

        from dl_techniques.models.vision.alexnet import create_alexnet

        model = create_alexnet(num_classes=10, input_shape=(227, 227, 3))
        logits = model(keras.random.normal((2, 227, 227, 3)))
    """
    if pretrained:
        raise NotImplementedError(
            "create_alexnet(pretrained=True): no pretrained weights are "
            "distributed with this port. Pass pretrained=False and train from "
            "scratch, or convert an external checkpoint and load it by hand."
        )

    return AlexNet(
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        include_top=include_top,
        lrn_hyperparameters=lrn_hyperparameters,
        **kwargs
    )

# ---------------------------------------------------------------------