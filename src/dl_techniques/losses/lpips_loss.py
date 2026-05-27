"""
LPIPS-flavored Perceptual Loss
==============================

A clean, dependency-free LPIPS-like perceptual loss using a frozen
ImageNet-pretrained VGG16 backbone. Computes per-channel-normalized
distances between feature maps at selected layers, weighted by per-layer
scalars.

Authored for use with `ConvNeXtPatchVAEV2` (multi-task VAE pretraining)
but generally applicable as a perceptual term in any image-to-image
training (reconstruction, super-resolution, generation, restoration).

Distinct from `dl_techniques.losses.image_restoration_loss.VGGLoss`:
- VGGLoss: MSE on VGG block outputs (un-normalized features).
- LPIPSLoss: L1 / L2 on **L2-channel-normalized** features (LPIPS recipe
  from Zhang et al. 2018, without their learned per-channel weights —
  hence "LPIPS-flavored", not the official LPIPS metric).

Mathematical formulation
------------------------
For a chosen set of VGG layers L with per-layer weight w_l,

    L_lpips(x, x_hat) = Σ_l  w_l · d( normalize(f_l(x)), normalize(f_l(x_hat)) )

where
- f_l(·) is the activation at VGG layer l after ImageNet-mean/std
  preprocessing of inputs in [0, 1],
- normalize(t) = t / (||t||_{channel,2} + eps) is per-channel L2 normalisation
  (channels = last axis),
- d is L1 (default) or L2 mean-squared.

Input range
-----------
The loss expects inputs in [0, 1]. If your data is in another range (e.g.
MSE-standardized CIFAR with per-channel mean/std), denormalize to [0, 1]
in the caller before invoking the loss. Optionally pass `input_range=(a, b)`
and the loss will rescale: ``x' = (x - a) / (b - a)``.

Serialization
-------------
The VGG backbone is built lazily on first call (after deserialization, the
backbone is rebuilt when first invoked). VGG weights are downloaded by
Keras on demand. Layer names and weights are serialized; the VGG model
itself is not — that would balloon the saved archive size.

Example
-------
.. code-block:: python

    from dl_techniques.losses import LPIPSLoss

    loss_fn = LPIPSLoss(
        layer_weights={
            "block1_conv1": 1.0 / 64,
            "block2_conv1": 1.0 / 128,
            "block3_conv1": 1.0 / 256,
            "block4_conv1": 1.0 / 512,
            "block5_conv1": 1.0 / 512,
        },
        distance="l1",
        input_range=(0.0, 1.0),
    )
    model.compile(optimizer="adam", loss=lambda yt, yp: loss_fn(yt, yp))

"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Default block weights — inverse channel count, biases mid-level features
# slightly above low-level edges/colors. Matches VGGLoss's spirit but uses
# the LPIPS normalisation.
_DEFAULT_LAYER_WEIGHTS_VGG16: Dict[str, float] = {
    "block1_conv1": 1.0 / 64.0,
    "block2_conv1": 1.0 / 128.0,
    "block3_conv1": 1.0 / 256.0,
    "block4_conv1": 1.0 / 512.0,
    "block5_conv1": 1.0 / 512.0,
}

# ImageNet preprocessing (standard, used by all keras.applications models).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Channel-normalisation eps (same scale as Zhang 2018 reference impl).
_NORM_EPS = 1.0e-10

# Allowed distances.
_VALID_DISTANCES = ("l1", "l2")

# Allowed VGG variants.
_VALID_VGG_VARIANTS = ("vgg16", "vgg19")


# ---------------------------------------------------------------------------
# LPIPS-flavored loss
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques.losses")
class LPIPSLoss(keras.losses.Loss):
    """LPIPS-flavored perceptual loss using frozen VGG features.

    Args:
        layer_weights: Mapping from VGG layer name → scalar weight. Layers
            must exist in the chosen VGG variant. Default: a 5-block recipe
            keyed by ``block{1..5}_conv1`` with inverse-channel weights.
        vgg_variant: ``"vgg16"`` (default) or ``"vgg19"``.
        input_range: Tuple ``(low, high)`` describing the expected input
            scale. Inputs are linearly mapped to ``[0, 1]`` before VGG
            preprocessing. Default ``(0.0, 1.0)``.
        distance: ``"l1"`` (mean absolute) or ``"l2"`` (mean squared).
            Default ``"l1"``.
        loss_weight: Overall multiplicative weight applied to the final
            loss scalar. Default ``1.0``.
        name: Loss name. Default ``"lpips_loss"``.
        **kwargs: Forwarded to ``keras.losses.Loss``.
    """

    def __init__(
        self,
        layer_weights: Optional[Dict[str, float]] = None,
        vgg_variant: str = "vgg16",
        input_range: Tuple[float, float] = (0.0, 1.0),
        distance: str = "l1",
        loss_weight: float = 1.0,
        name: str = "lpips_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)

        if vgg_variant not in _VALID_VGG_VARIANTS:
            raise ValueError(
                f"vgg_variant must be one of {_VALID_VGG_VARIANTS}, "
                f"got {vgg_variant!r}."
            )
        if distance not in _VALID_DISTANCES:
            raise ValueError(
                f"distance must be one of {_VALID_DISTANCES}, got {distance!r}."
            )
        if input_range[1] <= input_range[0]:
            raise ValueError(
                f"input_range high must exceed low, got {input_range}."
            )

        self.layer_weights = dict(
            layer_weights if layer_weights is not None
            else _DEFAULT_LAYER_WEIGHTS_VGG16
        )
        self.vgg_variant = vgg_variant
        self.input_range = (float(input_range[0]), float(input_range[1]))
        self.distance = distance
        self.loss_weight = float(loss_weight)

        # Lazy-built on first call (see _build_extractor). After
        # deserialization, this is None until first invocation.
        self._extractor: Optional[keras.Model] = None
        # DECISION plan_2026-05-27_4a444b14/D-001: VGG extractor lazily
        # built and held inside the Loss to defer the
        # `keras.applications.VGG{16,19}` import + ImageNet weight download
        # until first call. This keeps deserialization fast and avoids a
        # download-during-config failure when networks are offline. The
        # weights are frozen for the lifetime of this Loss instance.
        self._mean = ops.convert_to_tensor(_IMAGENET_MEAN, dtype="float32")
        self._std = ops.convert_to_tensor(_IMAGENET_STD, dtype="float32")

    # ------------------------------------------------------------------
    # Lazy VGG extractor
    # ------------------------------------------------------------------
    def _build_extractor(self) -> None:
        """Build the frozen VGG feature extractor on first call."""
        if self.vgg_variant == "vgg16":
            backbone = keras.applications.VGG16(
                include_top=False, weights="imagenet"
            )
        else:
            backbone = keras.applications.VGG19(
                include_top=False, weights="imagenet"
            )
        backbone.trainable = False

        # Validate every configured layer exists in the backbone.
        available = {layer.name for layer in backbone.layers}
        missing = [n for n in self.layer_weights if n not in available]
        if missing:
            raise ValueError(
                f"layer_weights references layers not present in "
                f"{self.vgg_variant}: {missing!r}. Available: "
                f"{sorted(available)[:8]}..."
            )

        outputs = [
            backbone.get_layer(name).output
            for name in self.layer_weights
        ]
        self._extractor = keras.Model(
            inputs=backbone.input, outputs=outputs, name=f"{self.vgg_variant}_features"
        )
        self._extractor.trainable = False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _to_unit_range(self, x: keras.KerasTensor) -> keras.KerasTensor:
        lo, hi = self.input_range
        if lo == 0.0 and hi == 1.0:
            return x
        return (x - lo) / (hi - lo)

    def _preprocess(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Map to [0,1] then ImageNet-normalize (per-channel)."""
        x = self._to_unit_range(x)
        x = (x - self._mean) / self._std
        return x

    @staticmethod
    def _channel_normalize(t: keras.KerasTensor) -> keras.KerasTensor:
        """L2-normalize along the channel (last) axis."""
        # Per Zhang 2018 reference: per-spatial-position channel-norm.
        norm = ops.sqrt(
            ops.sum(ops.square(t), axis=-1, keepdims=True) + _NORM_EPS
        )
        return t / norm

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        if self._extractor is None:
            self._build_extractor()

        y_true_n = self._preprocess(y_true)
        y_pred_n = self._preprocess(y_pred)

        feats_true = self._extractor(y_true_n, training=False)
        feats_pred = self._extractor(y_pred_n, training=False)

        # keras.Model returns a list when given multiple outputs.
        if not isinstance(feats_true, (list, tuple)):
            feats_true = [feats_true]
            feats_pred = [feats_pred]

        total = ops.convert_to_tensor(0.0, dtype="float32")
        names = list(self.layer_weights.keys())
        for layer_name, ft, fp in zip(names, feats_true, feats_pred):
            ft_n = self._channel_normalize(ft)
            fp_n = self._channel_normalize(fp)
            diff = ft_n - fp_n
            if self.distance == "l1":
                per_sample = ops.mean(ops.abs(diff), axis=[1, 2, 3])
            else:
                per_sample = ops.mean(ops.square(diff), axis=[1, 2, 3])
            total = total + self.layer_weights[layer_name] * per_sample

        return total * self.loss_weight

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "layer_weights": dict(self.layer_weights),
                "vgg_variant": self.vgg_variant,
                "input_range": list(self.input_range),
                "distance": self.distance,
                "loss_weight": self.loss_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LPIPSLoss":
        # input_range may have round-tripped as a list — coerce back to tuple.
        ir = config.get("input_range")
        if ir is not None:
            config = dict(config)
            config["input_range"] = tuple(ir)
        return cls(**config)
