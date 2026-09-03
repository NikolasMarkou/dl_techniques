"""CapsNetV2, a capsule network with single-step attention routing.

This is the V2 counterpart to :mod:`dl_techniques.models.vision.capsnet.model`. It
replaces the original iterative dynamic-routing loop with a single-step attention
routing capsule, and separates capsule magnitude (a learned sigmoid head) from
capsule orientation so the two no longer share one squash nonlinearity. The stem can
be the legacy two-conv stack or a ResNet backbone from
:mod:`dl_techniques.models.vision.resnet`. The model returns the classification length
tensor directly and trains through standard Keras `compile`/`fit` with margin or
cross-entropy loss, unlike V1's custom `train_step`/`test_step`. Reconstruction, when
enabled, is reached only through the separate :meth:`CapsNetV2.reconstruct` method, so
it never affects the standard training loss. `stem_pretrained=True` on a ResNet stem
raises `NotImplementedError`: no public ResNet weights ship with `dl_techniques`.

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between
      capsules. NeurIPS 30.
    - He, K., et al. (2015). Deep Residual Learning for Image Recognition.
"""

import keras
from typing import Optional, Tuple, Union, Dict, Any, List, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import length
from dl_techniques.losses.capsule_margin_loss import CapsuleMarginLoss
from dl_techniques.layers.capsules import PrimaryCapsule
from dl_techniques.layers.attention.attention_routing_capsule import (
    CapsuleBlockV2,
)
from dl_techniques.optimization import learning_rate_schedule_builder
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.capsnet.model_v2")
class CapsNetV2(keras.Model):
    """Capsule network with a stem, a primary capsule layer, and attention-routed digit capsules.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
          |
          v
        Stem (legacy conv-stack | ResNet backbone)  -> feature map
          |
          v
        PrimaryCapsule                              -> [B, N_p, D_p]
          |
          v
        CapsuleBlockV2 (attention routing)           -> digit_caps [B, num_classes, D_d]
          |
          +--> length(digit_caps) -> class probabilities  (call())
          |
          '--> reconstruct(): mask + Decoder (optional, isolated from the loss path)

    :param num_classes: Number of output classes. Must be positive.
    :type num_classes: int
    :param input_shape: ``(H, W, C)`` shape of the input image, without batch.
    :type input_shape: Tuple[int, int, int]
    :param stem: ``"legacy"`` for the two-conv stack from the original CapsNet paper, or a ResNet variant from `create_resnet` (``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``, ``"resnet152"``).
    :type stem: str
    :param stem_pretrained: Pretrained-weight option for a ResNet stem. False means random init. True raises `NotImplementedError`, since no public ResNet weights ship with `dl_techniques`. A string is a local path to a ``.keras`` weights file.
    :type stem_pretrained: Union[bool, str]
    :param primary_capsules: Number of primary capsules per spatial location, legacy stem only.
    :type primary_capsules: int
    :param primary_capsule_dim: Dimension of each primary capsule.
    :type primary_capsule_dim: int
    :param primary_kernel_size: Conv kernel for the primary-capsule layer, legacy stem only.
    :type primary_kernel_size: Union[int, Tuple[int, int]]
    :param primary_strides: Stride for the primary-capsule conv, legacy stem only.
    :type primary_strides: Union[int, Tuple[int, int]]
    :param digit_capsule_dim: Dimension of each output (class) capsule.
    :type digit_capsule_dim: int
    :param legacy_conv_filters: Filter counts for the legacy stem's two Conv2D layers.
    :type legacy_conv_filters: Optional[List[int]]
    :param loss_type: ``"margin"`` (capsule margin loss, matches V1) or ``"categorical_crossentropy"`` (CCE on softmax(length), supports label smoothing). Read by `create_capsnet_v2` to pick the compile-time loss.
    :type loss_type: str
    :param positive_margin: Positive margin for the margin loss.
    :type positive_margin: float
    :param negative_margin: Negative margin for the margin loss.
    :type negative_margin: float
    :param downweight: Downweight factor for the negative-class term in the margin loss.
    :type downweight: float
    :param reconstruction: Whether to build the decoder used by :meth:`reconstruct`. Reconstruction never participates in the standard loss path.
    :type reconstruction: bool
    :param decoder_architecture: Hidden layer sizes for the decoder, when `reconstruction` is True.
    :type decoder_architecture: Optional[List[int]]
    :param attention_softmax_axis: Forwarded to the attention routing capsule.
    :type attention_softmax_axis: str
    :param attention_top_k: Forwarded to the attention routing capsule.
    :type attention_top_k: Optional[int]
    :param use_load_balancing: Forwarded to the attention routing capsule.
    :type use_load_balancing: bool
    :param load_balancing_weight: Forwarded to the attention routing capsule.
    :type load_balancing_weight: float
    :param block_dropout_rate: Dropout rate inside `CapsuleBlockV2`.
    :type block_dropout_rate: float
    :param block_direction_only_norm: Whether `CapsuleBlockV2` uses length-preserving direction normalization.
    :type block_direction_only_norm: bool
    :param kernel_initializer: Initializer for trainable layers.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for trainable layers.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param name: Model name.
    :type name: Optional[str]

    :note: `call` returns a single tensor of shape ``(batch, num_classes)`` with per-class capsule lengths in ``(0, 1)``. Use :meth:`reconstruct` separately for image reconstructions; it is not part of the forward/loss path.
    """

    LEGACY_STEM = "legacy"
    RESNET_STEMS = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")

    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int],
        stem: Literal[
            "legacy", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "legacy",
        stem_pretrained: Union[bool, str] = False,
        primary_capsules: int = 32,
        primary_capsule_dim: int = 8,
        primary_kernel_size: Union[int, Tuple[int, int]] = 9,
        primary_strides: Union[int, Tuple[int, int]] = 2,
        digit_capsule_dim: int = 16,
        legacy_conv_filters: Optional[List[int]] = None,
        loss_type: Literal["margin", "categorical_crossentropy"] = "margin",
        positive_margin: float = 0.9,
        negative_margin: float = 0.1,
        downweight: float = 0.5,
        reconstruction: bool = False,
        decoder_architecture: Optional[List[int]] = None,
        attention_softmax_axis: Literal["output", "input"] = "output",
        attention_top_k: Optional[int] = None,
        use_load_balancing: bool = False,
        load_balancing_weight: float = 0.01,
        block_dropout_rate: float = 0.0,
        block_direction_only_norm: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        name: Optional[str] = "capsnet_v2",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        # ---- validate ----
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be a 3-tuple (H, W, C), got {input_shape}"
            )
        if stem != self.LEGACY_STEM and stem not in self.RESNET_STEMS:
            raise ValueError(
                f"stem must be 'legacy' or one of {self.RESNET_STEMS}, got {stem!r}"
            )
        if loss_type not in ("margin", "categorical_crossentropy"):
            raise ValueError(
                f"loss_type must be 'margin' or 'categorical_crossentropy', "
                f"got {loss_type!r}"
            )

        # ---- store config ----
        self.num_classes = num_classes
        self._input_shape: Tuple[int, int, int] = tuple(input_shape)  # type: ignore[assignment]
        self.stem = stem
        self.stem_pretrained = stem_pretrained
        self.primary_capsules = primary_capsules
        self.primary_capsule_dim = primary_capsule_dim
        self.primary_kernel_size = primary_kernel_size
        self.primary_strides = primary_strides
        self.digit_capsule_dim = digit_capsule_dim
        self.legacy_conv_filters = list(legacy_conv_filters) if legacy_conv_filters else [256, 256]
        self.loss_type = loss_type
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.downweight = float(downweight)
        self.reconstruction = reconstruction
        self.decoder_architecture = (
            list(decoder_architecture) if decoder_architecture else [512, 1024]
        )
        self.attention_softmax_axis = attention_softmax_axis
        self.attention_top_k = attention_top_k
        self.use_load_balancing = use_load_balancing
        self.load_balancing_weight = float(load_balancing_weight)
        self.block_dropout_rate = float(block_dropout_rate)
        self.block_direction_only_norm = block_direction_only_norm
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # ---- build sub-models ----
        self._build_stem()
        self._build_capsule_head()
        if self.reconstruction:
            self._build_decoder()
        else:
            self.decoder = None

    # ------------------------------------------------------------------
    def _build_stem(self) -> None:
        if self.stem == self.LEGACY_STEM:
            self.stem_layers: List[keras.layers.Layer] = []
            for i, filters in enumerate(self.legacy_conv_filters):
                self.stem_layers.append(
                    keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=9 if i == 0 else 5,
                        strides=1,
                        padding="valid",
                        kernel_initializer=self.kernel_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        name=f"legacy_conv_{i + 1}",
                    )
                )
                self.stem_layers.append(
                    keras.layers.BatchNormalization(name=f"legacy_bn_{i + 1}")
                )
                self.stem_layers.append(keras.layers.ReLU(name=f"legacy_relu_{i + 1}"))
            self.resnet_stem = None
        else:
            # Lazy import — avoid circular dependency when not needed.
            from dl_techniques.models.vision.resnet import create_resnet

            self.resnet_stem = create_resnet(
                variant=self.stem,
                num_classes=0,  # ignored when include_top=False
                input_shape=self._input_shape,
                pretrained=self.stem_pretrained,
                include_top=False,
                kernel_regularizer=self.kernel_regularizer,
            )
            self.stem_layers = []

    def _build_capsule_head(self) -> None:
        # PrimaryCapsule eats a 4-D feature map and produces (B, N, D).
        # Kernel size is config-controlled when stem=legacy; for resnet
        # stems we use a 1×1 to map channel depth to num_caps × dim_caps.
        if self.stem == self.LEGACY_STEM:
            primary_ks = self.primary_kernel_size
            primary_strides = self.primary_strides
        else:
            primary_ks = 1
            primary_strides = 1

        self.primary_caps = PrimaryCapsule(
            num_capsules=self.primary_capsules,
            dim_capsules=self.primary_capsule_dim,
            kernel_size=primary_ks,
            strides=primary_strides,
            padding="valid" if self.stem == self.LEGACY_STEM else "same",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="primary_caps",
        )

        self.digit_caps = CapsuleBlockV2(
            num_capsules=self.num_classes,
            dim_capsules=self.digit_capsule_dim,
            dropout_rate=self.block_dropout_rate,
            direction_only_norm=self.block_direction_only_norm,
            softmax_axis=self.attention_softmax_axis,
            top_k=self.attention_top_k,
            use_load_balancing=self.use_load_balancing,
            load_balancing_weight=self.load_balancing_weight,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="digit_caps",
        )

    def _build_decoder(self) -> None:
        """Optional reconstruction head — used only via :meth:`reconstruct`."""
        decoder_layers: List[keras.layers.Layer] = []
        for i, units in enumerate(self.decoder_architecture):
            decoder_layers.append(
                keras.layers.Dense(
                    units=units,
                    activation="relu",
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"decoder_hidden_{i + 1}",
                )
            )
        flat_size = int(self._input_shape[0] * self._input_shape[1] * self._input_shape[2])
        decoder_layers.append(
            keras.layers.Dense(
                units=flat_size,
                activation="sigmoid",
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="decoder_output",
            )
        )
        decoder_layers.append(
            keras.layers.Reshape(target_shape=self._input_shape, name="decoder_reshape")
        )
        self.decoder = keras.Sequential(decoder_layers, name="reconstruction_decoder")

    # ------------------------------------------------------------------
    def _stem_forward(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        if self.resnet_stem is not None:
            return self.resnet_stem(x, training=training)
        for layer in self.stem_layers:
            x = layer(x, training=training)
        return x

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the forward pass, returning per-class capsule lengths.

        :param inputs: Input images, shape ``[B, H, W, C]``.
        :type inputs: keras.KerasTensor
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: Capsule lengths, shape ``(batch, num_classes)``, values in ``(0, 1)``. Compile with `CapsuleMarginLoss` (default) or `keras.losses.CategoricalCrossentropy` (label smoothing).
        :rtype: keras.KerasTensor
        :raises ValueError: If `inputs` is not 4D.
        """
        if len(inputs.shape) != 4:
            raise ValueError(
                f"Expected 4D input [B, H, W, C], got shape {inputs.shape}"
            )

        features = self._stem_forward(inputs, training=training)
        primary = self.primary_caps(features, training=training)
        digit = self.digit_caps(primary, training=training)
        # ‖digit‖ — per-capsule lengths; this is the prediction.
        return length(digit)

    # ------------------------------------------------------------------
    def get_capsules(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run the forward pass, returning the raw digit capsule pose vectors.

        The pose representation prior to length extraction, useful for :meth:`reconstruct`
        and for downstream pose analysis.

        :param inputs: Input images, shape ``[B, H, W, C]``.
        :type inputs: keras.KerasTensor
        :param training: Whether the call is in training mode.
        :type training: Optional[bool]
        :return: Digit capsule poses, shape ``(batch, num_classes, digit_capsule_dim)``.
        :rtype: keras.KerasTensor
        """
        features = self._stem_forward(inputs, training=training)
        primary = self.primary_caps(features, training=training)
        return self.digit_caps(primary, training=training)

    def reconstruct(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Reconstruct `inputs` through the decoder, when reconstruction is enabled.

        :param inputs: Input images, shape ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param mask: Optional one-hot ``(B, num_classes)`` mask. Falls back to the predicted class (argmax of capsule lengths) when omitted.
        :type mask: Optional[keras.KerasTensor]
        :return: Reconstructed image, shape ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If the model was constructed with `reconstruction=False`, or if `mask`'s last dimension does not equal `num_classes`.
        """
        if self.decoder is None:
            raise ValueError(
                "reconstruct() requires reconstruction=True at construction time."
            )

        digit = self.get_capsules(inputs, training=False)
        lengths = length(digit)
        if mask is None:
            mask = keras.ops.one_hot(keras.ops.argmax(lengths, axis=1), num_classes=self.num_classes)
        else:
            if mask.shape[-1] != self.num_classes:
                raise ValueError(
                    f"mask last-dim must be num_classes={self.num_classes}, "
                    f"got {mask.shape[-1]}"
                )
        masked = digit * keras.ops.expand_dims(mask, -1)
        flat = keras.ops.reshape(masked, (-1, self.num_classes * self.digit_capsule_dim))
        return self.decoder(flat)

    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: Config dict with every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "input_shape": self._input_shape,
                "stem": self.stem,
                "stem_pretrained": (
                    self.stem_pretrained
                    if isinstance(self.stem_pretrained, (bool, str))
                    else False
                ),
                "primary_capsules": self.primary_capsules,
                "primary_capsule_dim": self.primary_capsule_dim,
                "primary_kernel_size": self.primary_kernel_size,
                "primary_strides": self.primary_strides,
                "digit_capsule_dim": self.digit_capsule_dim,
                "legacy_conv_filters": self.legacy_conv_filters,
                "loss_type": self.loss_type,
                "positive_margin": self.positive_margin,
                "negative_margin": self.negative_margin,
                "downweight": self.downweight,
                "reconstruction": self.reconstruction,
                "decoder_architecture": self.decoder_architecture,
                "attention_softmax_axis": self.attention_softmax_axis,
                "attention_top_k": self.attention_top_k,
                "use_load_balancing": self.use_load_balancing,
                "load_balancing_weight": self.load_balancing_weight,
                "block_dropout_rate": self.block_dropout_rate,
                "block_direction_only_norm": self.block_direction_only_norm,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CapsNetV2":
        """Build a model from a config dict, deserializing initializer and regularizer entries.

        :param config: Config dict as returned by `get_config`.
        :type config: Dict[str, Any]
        :return: A new `CapsNetV2` instance.
        :rtype: CapsNetV2
        """
        if "kernel_initializer" in config and isinstance(config["kernel_initializer"], dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if "kernel_regularizer" in config and config["kernel_regularizer"]:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if "input_shape" in config and isinstance(config["input_shape"], list):
            config["input_shape"] = tuple(config["input_shape"])
        # The saved model already contains the stem weights, so deserialization never re-fetches pretrained ones.
        config["stem_pretrained"] = False
        return cls(**config)


# ---------------------------------------------------------------------
#  Factory functions
# ---------------------------------------------------------------------


def _default_recipe(
    learning_rate: float,
    decay_steps: int,
    warmup_steps: Optional[int] = None,
    weight_decay: float = 0.05,
    use_ema: bool = True,
    ema_momentum: float = 0.999,
    global_clipnorm: float = 1.0,
) -> keras.optimizers.Optimizer:
    """Build the modern training recipe: AdamW with a cosine schedule, warmup, and EMA.

    :param learning_rate: Peak learning rate after warmup.
    :type learning_rate: float
    :param decay_steps: Total decay steps for the cosine schedule.
    :type decay_steps: int
    :param warmup_steps: Warmup steps. Defaults to 5% of `decay_steps`.
    :type warmup_steps: Optional[int]
    :param weight_decay: AdamW decoupled weight decay.
    :type weight_decay: float
    :param use_ema: Whether to enable EMA on the weights.
    :type use_ema: bool
    :param ema_momentum: EMA decay.
    :type ema_momentum: float
    :param global_clipnorm: Global-norm gradient clipping.
    :type global_clipnorm: float
    :return: A configured `AdamW` optimizer.
    :rtype: keras.optimizers.Optimizer
    """
    if warmup_steps is None:
        warmup_steps = max(1, int(0.05 * decay_steps))

    schedule = learning_rate_schedule_builder(
        {
            "type": "cosine_decay",
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "warmup_start_lr": 1e-8,
            "decay_steps": decay_steps,
            "alpha": 0.0,
        }
    )

    return keras.optimizers.AdamW(
        learning_rate=schedule,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        global_clipnorm=global_clipnorm,
        use_ema=use_ema,
        ema_momentum=ema_momentum,
        name="AdamW_modern_recipe",
    )


def create_capsnet_v2(
    num_classes: int,
    input_shape: Tuple[int, int, int],
    stem: Literal[
        "legacy", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
    ] = "legacy",
    stem_pretrained: Union[bool, str] = False,
    *,
    learning_rate: float = 1e-3,
    decay_steps: int = 10_000,
    warmup_steps: Optional[int] = None,
    weight_decay: float = 0.05,
    use_ema: bool = True,
    ema_momentum: float = 0.999,
    global_clipnorm: float = 1.0,
    label_smoothing: float = 0.0,
    loss_type: Literal["margin", "categorical_crossentropy"] = "margin",
    positive_margin: float = 0.9,
    negative_margin: float = 0.1,
    downweight: float = 0.5,
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    **model_kwargs: Any,
) -> CapsNetV2:
    """Create and compile a `CapsNetV2` with the modern training recipe.

    Wraps `CapsNetV2` with AdamW, a cosine schedule with linear warmup, EMA, and
    gradient clipping. Compiles with `CapsuleMarginLoss` (default) or
    `keras.losses.CategoricalCrossentropy(label_smoothing=...)`.

    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: ``(H, W, C)`` input shape.
    :type input_shape: Tuple[int, int, int]
    :param stem: Stem variant. See `CapsNetV2`.
    :type stem: str
    :param stem_pretrained: Pretrained-weight option for a ResNet stem.
    :type stem_pretrained: Union[bool, str]
    :param learning_rate: Peak learning rate after warmup.
    :type learning_rate: float
    :param decay_steps: Total decay steps for the cosine schedule.
    :type decay_steps: int
    :param warmup_steps: Warmup steps. Defaults to 5% of `decay_steps`.
    :type warmup_steps: Optional[int]
    :param weight_decay: AdamW decoupled weight decay.
    :type weight_decay: float
    :param use_ema: Whether to enable EMA on the weights.
    :type use_ema: bool
    :param ema_momentum: EMA decay.
    :type ema_momentum: float
    :param global_clipnorm: Global-norm gradient clipping.
    :type global_clipnorm: float
    :param label_smoothing: Used only when `loss_type` is ``"categorical_crossentropy"``.
    :type label_smoothing: float
    :param loss_type: ``"margin"`` (default) or ``"categorical_crossentropy"``.
    :type loss_type: str
    :param positive_margin: Positive margin for the margin loss.
    :type positive_margin: float
    :param negative_margin: Negative margin for the margin loss.
    :type negative_margin: float
    :param downweight: Downweight factor for the margin loss.
    :type downweight: float
    :param optimizer: Supply an optimizer directly, skipping the built-in recipe.
    :type optimizer: Optional[keras.optimizers.Optimizer]
    :param model_kwargs: Forwarded to `CapsNetV2`.
    :return: A compiled `CapsNetV2`.
    :rtype: CapsNetV2
    """
    model = CapsNetV2(
        num_classes=num_classes,
        input_shape=input_shape,
        stem=stem,
        stem_pretrained=stem_pretrained,
        loss_type=loss_type,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
        downweight=downweight,
        **model_kwargs,
    )

    if optimizer is None:
        optimizer = _default_recipe(
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            global_clipnorm=global_clipnorm,
        )

    if loss_type == "margin":
        loss_fn = CapsuleMarginLoss(
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            downweight=downweight,
        )
    else:
        loss_fn = keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing,
            from_logits=False,
        )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    logger.info(
        f"create_capsnet_v2: stem={stem}, num_classes={num_classes}, "
        f"loss={loss_type}, optimizer={optimizer.__class__.__name__}"
    )
    return model


def create_capsnet_v2_pretrained(
    backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] = "resnet18",
    num_classes: int = 10,
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    pretrained: Union[bool, str] = True,
    **kwargs: Any,
) -> CapsNetV2:
    """Build a capsule head on a pretrained ResNet backbone.

    Equivalent to ``create_capsnet_v2(num_classes=num_classes, input_shape=input_shape,
    stem=backbone, stem_pretrained=pretrained, ...)``.

    :param backbone: ResNet variant to use as the stem.
    :type backbone: str
    :param num_classes: Number of output classes.
    :type num_classes: int
    :param input_shape: ``(H, W, C)`` input shape.
    :type input_shape: Tuple[int, int, int]
    :param pretrained: A local ``.keras`` weights path, or True.
    :type pretrained: Union[bool, str]
    :param kwargs: Forwarded to `create_capsnet_v2`.
    :return: A compiled `CapsNetV2`.
    :rtype: CapsNetV2
    :raises ValueError: If `backbone` is not a supported ResNet variant.
    :raises NotImplementedError: If `pretrained` is True rather than a path — no public ResNet weights ship with `dl_techniques`.
    """
    if backbone not in CapsNetV2.RESNET_STEMS:
        raise ValueError(
            f"backbone must be one of {CapsNetV2.RESNET_STEMS}, got {backbone!r}"
        )
    return create_capsnet_v2(
        num_classes=num_classes,
        input_shape=input_shape,
        stem=backbone,
        stem_pretrained=pretrained,
        **kwargs,
    )


# ---------------------------------------------------------------------
