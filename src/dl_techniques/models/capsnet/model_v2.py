"""CapsNet V2 — modernised capsule network with attention routing.

This module is the V2 counterpart to :mod:`dl_techniques.models.capsnet.model`.
It addresses several documented shortcomings of the original 2017 architecture
without breaking the legacy ``CapsNet`` API:

Improvements over V1
--------------------
* **Attention routing.** Replaces the iterative dynamic-routing inner loop with
  a single-step :class:`AttentionRoutingCapsule` (see
  :mod:`dl_techniques.layers.attention.attention_routing_capsule`).
* **Decoupled length & probability.** Capsule magnitude is a learned scalar
  (sigmoid head) rather than a squash side-effect, eliminating saturation at
  zero.
* **Configurable backbone.** Stem can be the legacy two-conv stack or any
  ResNet variant from :mod:`dl_techniques.models.resnet`. Stage-2 pretraining
  flow accepts ``stem_pretrained=True`` (delegates to ResNet's existing
  download fallback) or a local weights path string.
* **Standard ``compile/fit``.** The model returns the classification length
  tensor directly. Margin / cross-entropy loss flows through the standard
  Keras workflow — no custom ``train_step`` / ``test_step`` and no dict outputs.
* **Modern recipe defaults.** :func:`create_capsnet_v2` wires AdamW + cosine
  schedule with linear warmup + EMA + global-norm gradient clipping.
* **Reconstruction is optional and isolated.** When enabled, the decoder is
  exposed via the :meth:`CapsNetV2.reconstruct` helper rather than being baked
  into the loss path. Reconstruction gradients no longer corrupt the
  classification representation in standard training.

References
----------
* Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between
  capsules. NeurIPS 30.
* He, K., et al. (2015). Deep Residual Learning for Image Recognition.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Union, Dict, Any, List, Literal

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import length
from dl_techniques.losses.capsule_margin_loss import CapsuleMarginLoss
from dl_techniques.layers.capsules import PrimaryCapsule
from dl_techniques.layers.attention.attention_routing_capsule import (
    AttentionRoutingCapsule,
    CapsuleBlockV2,
)
from dl_techniques.optimization import learning_rate_schedule_builder

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsNetV2(keras.Model):
    """Modernised Capsule Network with attention routing.

    Architecture::

        Input → Stem (legacy conv | resnet) → PrimaryCapsule
              → CapsuleBlockV2 (AttentionRoutingCapsule) → length
              → class probabilities (via margin loss or CCE)

    Args:
        num_classes: Number of output classes. Must be positive.
        input_shape: ``(H, W, C)`` shape of the input image (without batch).
        stem: Stem variant — ``"legacy"`` (two-conv stack matching the
            original CapsNet paper), or any ResNet variant supported by
            :func:`dl_techniques.models.resnet.create_resnet`
            (``"resnet18"``, ``"resnet34"``, ``"resnet50"``,
            ``"resnet101"``, ``"resnet152"``). Defaults to ``"legacy"``.
        stem_pretrained: Pretrained-weight option for ResNet stems.
            ``False`` (default) means random init. ``True`` attempts to
            download weights via the ResNet's URL config (which contains
            placeholder URLs in this repo — graceful fallback to random
            init on download failure). A string is treated as a local
            path to a ``.keras`` weights file.
        primary_capsules: Number of primary capsules per spatial location
            in the legacy stem. Defaults to ``32``.
        primary_capsule_dim: Dimension of each primary capsule. Defaults
            to ``8``.
        primary_kernel_size: Conv kernel for the primary-capsule layer
            (when stem is legacy). Defaults to ``9``.
        primary_strides: Stride for the primary-capsule conv. Defaults to ``2``.
        digit_capsule_dim: Dimension of each output / class capsule.
            Defaults to ``16``.
        legacy_conv_filters: Filter counts for the legacy stem's two
            Conv2D layers. Defaults to ``[256, 256]``.
        loss_type: Either ``"margin"`` (capsule margin loss; matches V1)
            or ``"categorical_crossentropy"`` (CCE on softmax(length),
            supports label smoothing). Used only by
            :func:`create_capsnet_v2` to pick the compile-time loss.
            Defaults to ``"margin"``.
        positive_margin / negative_margin / downweight: Margin-loss params.
        reconstruction: If ``True``, build a decoder for the
            :meth:`reconstruct` helper. Default ``False`` — reconstruction
            no longer participates in the standard loss path.
        decoder_architecture: Hidden layer sizes for the decoder, when
            ``reconstruction=True``. Defaults to ``[512, 1024]``.
        attention_softmax_axis: Forwarded to
            :class:`AttentionRoutingCapsule`.
        attention_top_k: Forwarded to :class:`AttentionRoutingCapsule`.
        use_load_balancing: Forwarded to :class:`AttentionRoutingCapsule`.
        load_balancing_weight: Forwarded to :class:`AttentionRoutingCapsule`.
        block_dropout_rate: Dropout in :class:`CapsuleBlockV2`.
        block_direction_only_norm: Length-preserving direction LN in
            :class:`CapsuleBlockV2`.
        kernel_initializer / kernel_regularizer: Initialization /
            regularization for trainable layers.
        name: Model name.

    Notes
    -----
    The forward pass returns a single tensor of shape
    ``(batch, num_classes)`` containing per-class capsule lengths in
    ``(0, 1)``. Use :meth:`reconstruct` separately if you need image
    reconstructions; reconstruction is **not** part of the standard
    forward / loss path.
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
            from dl_techniques.models.resnet import create_resnet

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
        """Forward pass returning the per-class capsule lengths.

        Returns a tensor of shape ``(batch, num_classes)`` with values in
        ``(0, 1)``. Compile with :class:`CapsuleMarginLoss` (default) or
        :class:`keras.losses.CategoricalCrossentropy` (label smoothing).
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
        """Forward pass returning the raw digit capsule pose vectors.

        Returns a tensor of shape ``(batch, num_classes, digit_capsule_dim)``
        — the pose representation prior to length extraction. Useful for
        :meth:`reconstruct` and for downstream pose analysis.
        """
        features = self._stem_forward(inputs, training=training)
        primary = self.primary_caps(features, training=training)
        return self.digit_caps(primary, training=training)

    def reconstruct(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Reconstruct ``inputs`` via the decoder (when enabled).

        Args:
            inputs: ``(B, H, W, C)`` image tensor.
            mask: Optional one-hot ``(B, num_classes)`` mask. If ``None``,
                uses the predicted class (argmax of capsule lengths).

        Returns:
            Reconstructed image tensor with shape ``(B, H, W, C)``.

        Raises:
            ValueError: If the model was constructed with
                ``reconstruction=False``.
        """
        if self.decoder is None:
            raise ValueError(
                "reconstruct() requires reconstruction=True at construction time."
            )

        digit = self.get_capsules(inputs, training=False)
        lengths = length(digit)
        if mask is None:
            mask = ops.one_hot(ops.argmax(lengths, axis=1), num_classes=self.num_classes)
        else:
            if mask.shape[-1] != self.num_classes:
                raise ValueError(
                    f"mask last-dim must be num_classes={self.num_classes}, "
                    f"got {mask.shape[-1]}"
                )
        masked = digit * ops.expand_dims(mask, -1)
        flat = ops.reshape(masked, (-1, self.num_classes * self.digit_capsule_dim))
        return self.decoder(flat)

    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
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
        # Don't try to actually pull pretrained weights on deserialization —
        # the saved-model already contains them. Force False here.
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
    """Build the modern training recipe: AdamW + cosine + warmup + EMA."""
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
    """Create and compile a :class:`CapsNetV2` with the modern training recipe.

    Wraps :class:`CapsNetV2` with sensible defaults: AdamW + cosine schedule
    + linear warmup + EMA + gradient clipping. Compiles with
    :class:`CapsuleMarginLoss` (default) or
    :class:`keras.losses.CategoricalCrossentropy(label_smoothing=...)`.

    Args:
        num_classes: Number of output classes.
        input_shape: ``(H, W, C)``.
        stem: Stem variant. See :class:`CapsNetV2`.
        stem_pretrained: Pretrained-weight option for ResNet stems.
        learning_rate: Peak LR after warmup.
        decay_steps: Total decay steps for cosine schedule.
        warmup_steps: Warmup steps. Defaults to ``5 %`` of ``decay_steps``.
        weight_decay: AdamW weight decay (decoupled).
        use_ema: Enable EMA on weights.
        ema_momentum: EMA decay.
        global_clipnorm: Global-norm gradient clipping.
        label_smoothing: For ``loss_type="categorical_crossentropy"`` only.
        loss_type: ``"margin"`` (default) or ``"categorical_crossentropy"``.
        positive_margin / negative_margin / downweight: Margin loss params.
        optimizer: Skip the recipe and supply your own optimizer.
        **model_kwargs: Forwarded to :class:`CapsNetV2`.

    Returns:
        Compiled :class:`CapsNetV2`.
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
    """Convenience wrapper: capsule head on a pretrained ResNet backbone (Stage 2).

    Equivalent to::

        create_capsnet_v2(num_classes=num_classes, input_shape=input_shape,
                          stem=backbone, stem_pretrained=pretrained, ...)

    The repo's :func:`dl_techniques.models.resnet.create_resnet` handles the
    download fallback (logs a warning and continues with random init if the
    URL is unavailable). Pass a local ``.keras`` weights path string to
    ``pretrained`` to skip the download path entirely.
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
