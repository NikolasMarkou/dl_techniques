"""OmniPoint's three per-encoder-feature output heads.

Conceptual Overview:
    All three heads read the SAME encoder spatial feature map (or, for
    :class:`MetricScaleHead`, the same encoder CLS token) -- Step 4 builds no
    conditioning input, so this module's only job is turning one shared
    feature tensor into OmniPoint's per-pixel ray/distance/mask predictions
    and its single per-sample metric scale.

    :class:`RayDistanceHead` and :class:`MaskHead` each wrap exactly one
    :class:`~dl_techniques.models.vision.depth_anything.components.DPTDecoder`
    instance -- the repo's existing per-pixel dense-prediction head, reused
    with head-specific ``output_channels``/``output_activation`` rather than
    reimplemented (``findings/model-house-patterns.md`` #2). Only
    :class:`MetricScaleHead` is new work: `DPTDecoder` is spatial-preserving
    and cannot express a pooled global scalar.

    Output-activation choices are dictated by ``losses/omnipoint_losses.py``,
    not chosen independently here:

    - ``MaskHead`` emits **linear logits**, matching
      ``MaskLoss``'s ``keras.losses.BinaryCrossentropy(from_logits=True)``
      (`omnipoint_losses.py:325`). Passing ``sigmoid`` here would double-apply
      the sigmoid inside that loss's BCE term.
    - ``MetricScaleHead`` emits a **positive scale** (`softplus`), not a
      log-scale value: ``MetricScaleLoss.call`` takes ``log()`` of its
      ``y_pred`` argument directly (`omnipoint_losses.py:302`,
      ``log_s_hat = keras.ops.log(keras.ops.maximum(y_pred, eps))``), so the
      loss owns the log-space transform and the head must hand it a positive,
      non-log value.
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

from dl_techniques.models.vision.depth_anything.components import DPTDecoder
from dl_techniques.layers.geometric.ray_point_combinator import RayPointCombinator
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.omnipoint.heads.ray_distance_head")
class RayDistanceHead(keras.layers.Layer):
    """Per-pixel ray direction + radial distance head.

    Composes one `DPTDecoder` (4 raw channels: 3 for the unnormalized ray, 1
    for the unnormalized distance, both ``output_activation="linear"`` since
    :class:`~dl_techniques.layers.geometric.ray_point_combinator.RayPointCombinator`
    owns the normalize/positive-activation step, not this decoder) with that
    combinator, so the head's own output is already unit-norm/positive by
    construction (Problem Statement invariants 1-2 of
    `models/vision/omnipoint/` plan.md).

    :param dims: Channel dimension per `DPTDecoder` stage.
    :type dims: Optional[list]
    :param upsample_factor: Total bilinear upsampling factor forwarded to
        `DPTDecoder`. Must be a power of 2.
    :type upsample_factor: int
    :param kernel_initializer: Initializer for every `DPTDecoder` conv kernel.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every conv kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param epsilon: Floor forwarded to `RayPointCombinator` (ray-norm and
        distance-activation guard).
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the `Layer` base class.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``, the encoder's
        spatial feature map.

    Output shape:
        3-tuple ``(unit_ray, positive_distance, point)``, each
        ``(batch_size, height * upsample_factor, width * upsample_factor, C)``
        with ``C=3`` for ``unit_ray``/``point`` and ``C=1`` for
        ``positive_distance``.

    Example:
        .. code-block:: python

            head = RayDistanceHead(dims=[256, 128, 64, 32], upsample_factor=1)
            x = keras.random.normal([2, 16, 16, 768])
            ray, distance, point = head(x)
    """

    def __init__(
            self,
            dims: Optional[list] = None,
            upsample_factor: int = 1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            epsilon: float = 1e-8,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dims = list(dims) if dims is not None else [256, 128, 64, 32]
        self.upsample_factor = int(upsample_factor)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.epsilon = epsilon

        self.decoder = DPTDecoder(
            dims=self.dims,
            output_channels=4,
            output_activation="linear",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            upsample_factor=self.upsample_factor,
            name="ray_distance_decoder",
        )
        self.combinator = RayPointCombinator(
            epsilon=self.epsilon, name="ray_point_combinator"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Decode raw ray/distance logits, then normalize/combine them.

        :param inputs: Encoder spatial feature map, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the inner `DPTDecoder`.
        :type training: Optional[bool]
        :return: ``(unit_ray, positive_distance, point)``.
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        raw = self.decoder(inputs, training=training)
        raw_ray, raw_distance = keras.ops.split(raw, [3], axis=-1)
        return self.combinator((raw_ray, raw_distance))

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base `Layer` config plus every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dims": self.dims,
            "upsample_factor": self.upsample_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "epsilon": self.epsilon,
        })
        return config


@register_dl_technique("dl_techniques.models.omnipoint.heads.mask_head")
class MaskHead(keras.layers.Layer):
    """Per-pixel sky/validity mask head: one `DPTDecoder`, linear logits.

    Emits linear (unbounded) logits, matching `MaskLoss`'s
    ``BinaryCrossentropy(from_logits=True)`` -- see this module's docstring.

    :param dims: Channel dimension per `DPTDecoder` stage.
    :type dims: Optional[list]
    :param upsample_factor: Total bilinear upsampling factor. Must be a power
        of 2.
    :type upsample_factor: int
    :param kernel_initializer: Initializer for every conv kernel.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every conv kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the `Layer` base class.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``.

    Output shape:
        4D tensor ``(batch_size, height * upsample_factor,
        width * upsample_factor, 1)``, linear mask logits.
    """

    def __init__(
            self,
            dims: Optional[list] = None,
            upsample_factor: int = 1,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dims = list(dims) if dims is not None else [256, 128, 64, 32]
        self.upsample_factor = int(upsample_factor)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.decoder = DPTDecoder(
            dims=self.dims,
            output_channels=1,
            # DECISION plan-2026-09-11T050223-1b47bcf6/D-010: linear logits, not sigmoid.
            # MaskLoss wraps BinaryCrossentropy(from_logits=True); a sigmoid here would
            # double-apply the squash inside that loss's own BCE term. See decisions.md.
            output_activation="linear",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            upsample_factor=self.upsample_factor,
            name="mask_decoder",
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Decode the mask logits.

        :param inputs: Encoder spatial feature map, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the inner `DPTDecoder`.
        :type training: Optional[bool]
        :return: Mask logits, ``(B, H*upsample_factor, W*upsample_factor, 1)``.
        :rtype: keras.KerasTensor
        """
        return self.decoder(inputs, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base `Layer` config plus every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dims": self.dims,
            "upsample_factor": self.upsample_factor,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config


@register_dl_technique("dl_techniques.models.omnipoint.heads.metric_scale_head")
class MetricScaleHead(keras.layers.Layer):
    """Global per-sample metric scale head: a 2-layer MLP over a pooled token.

    `DPTDecoder` is spatial-preserving and cannot express a pooled scalar
    (`findings/model-house-patterns.md` #2), so this is new work: a small
    ``Dense(hidden, relu) -> Dense(1, softplus)`` MLP over whatever single
    ``(B, D)`` "metric token" the caller supplies -- see
    `model.py`'s module docstring for which token that is and why.

    Output activation is `softplus`, giving a strictly positive scale, NOT a
    log-scale value -- see this module's docstring for why (`MetricScaleLoss`
    takes the log itself).

    :param hidden_dim: Width of the hidden `Dense` layer.
    :type hidden_dim: int
    :param kernel_initializer: Initializer for both `Dense` kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for both `Dense` kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param epsilon: Floor added under the `softplus` output so the scale never
        underflows to exact ``0.0`` in float32 (mirrors
        `RayPointCombinator`'s own distance-activation floor).
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the `Layer` base class.

    Input shape:
        2D tensor ``(batch_size, embed_dim)``, a pooled/CLS token.

    Output shape:
        1D tensor ``(batch_size,)``, strictly positive.
    """

    def __init__(
            self,
            hidden_dim: int = 256,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            epsilon: float = 1e-8,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.epsilon = epsilon

        self.hidden = keras.layers.Dense(
            self.hidden_dim,
            activation="relu",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="metric_scale_hidden",
        )
        # Linear projection to a scalar; softplus + epsilon floor applied
        # explicitly in `call()` (not as the Dense activation) so the floor is
        # visible at the one call site that needs it, mirroring
        # RayPointCombinator's own explicit-floor style.
        self.projection = keras.layers.Dense(
            1,
            activation="linear",
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="metric_scale_projection",
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Compute the positive per-sample metric scale.

        :param inputs: Pooled metric token, ``(B, D)``.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the inner `Dense` layers.
        :type training: Optional[bool]
        :return: Positive scale, ``(B,)``.
        :rtype: keras.KerasTensor
        """
        x = self.hidden(inputs, training=training)
        raw_scale = self.projection(x, training=training)
        positive_scale = keras.ops.maximum(
            keras.ops.softplus(raw_scale), self.epsilon
        )
        return keras.ops.squeeze(positive_scale, axis=-1)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: The base `Layer` config plus every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
