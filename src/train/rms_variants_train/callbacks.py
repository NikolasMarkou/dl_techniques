"""Probe callbacks for the RMSNorm-variants study.

Each callback targets a specific theoretical claim made by one or more
normalization variants:

- :class:`GradientNormCallback` — global gradient L2 norm trajectory.
  Surfaces gradient-explosion or vanishing-gradient regimes that differ
  between norms.
- :class:`WeightNormTrajectoryCallback` — per-norm-layer weight L2 trajectory.
  Direct test of the **γ-growth-suppression** claim made by
  ``ZeroCenteredRMSNorm``: in vanilla ``RMSNorm`` the per-feature ``scale``
  must grow to compensate for mean drift in the residual stream; the
  zero-centered variant claims to remove this driver.
- :class:`NormLayerActivationCallback` — per-norm-layer activation mean and
  per-sample-RMS std evaluated on a fixed calibration batch.
  Direct test of (a) the **zero-mean output** property of the
  zero-centered variants and (b) the **thick spherical shell** RMS bound
  property of the band variants.
- :class:`NormInternalStatsCallback` — per-norm-layer internal scalar state:
  ``scale`` L2 for the RMSNorm family, ``band_param`` raw value and
  post-sigmoid scale for the band family.

Each callback writes one row per epoch to its own CSV under ``out_dir``.
Lazy-build idioms throughout: no work happens before the first call.

DECISION plan_2026-05-14_3764496e/D-001: 4 callbacks (above ≤3 soft cap) at
the cost of one extra layer of probe complexity, accepted because each
probe tests a distinct variant claim and none is removable.
"""
from __future__ import annotations

import csv
import os
from typing import Iterable, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.adaptive_band_rms import AdaptiveBandRMS
from dl_techniques.layers.norms.band_logit_norm import BandLogitNorm
from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.layers.norms.dynamic_tanh import DynamicTanh
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.norms.zero_centered_band_rms_norm import (
    ZeroCenteredBandRMSNorm,
)
from dl_techniques.layers.norms.zero_centered_rms_norm import ZeroCenteredRMSNorm


NORM_LAYER_CLASSES: Tuple[type, ...] = (
    RMSNorm,
    BandRMS,
    ZeroCenteredRMSNorm,
    ZeroCenteredBandRMSNorm,
    AdaptiveBandRMS,
    BandLogitNorm,
    DynamicTanh,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _walk_layers(root: keras.layers.Layer) -> Iterable[keras.layers.Layer]:
    """Yield ``root`` and every sub-layer recursively (depth-first).

    Keras 3 only exposes ``model.layers`` for top-level children. Sub-blocks
    nested inside custom layers (e.g. ViT transformer blocks) hide their
    own children. We recurse via the ``_layers`` collection that Keras
    populates for every ``Layer`` instance.
    """
    seen: set[int] = set()
    stack: List[keras.layers.Layer] = [root]
    while stack:
        layer = stack.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        yield layer
        # ``_layers`` is the Keras-internal tracking attribute for sub-layers.
        sub = getattr(layer, "_layers", None) or []
        # On some Keras versions sub-layers also live on ``layers`` attribute
        # for Sequential / Functional containers.
        sub = list(sub) + list(getattr(layer, "layers", []) or [])
        for child in sub:
            if isinstance(child, keras.layers.Layer):
                stack.append(child)


def _find_norm_layers(
    model: keras.Model,
    target_classes: Tuple[type, ...] = NORM_LAYER_CLASSES,
) -> List[keras.layers.Layer]:
    """Return every norm-variant instance reachable from ``model``."""
    return [layer for layer in _walk_layers(model) if isinstance(layer, target_classes)]


# ---------------------------------------------------------------------
# 1. GradientNormCallback
# ---------------------------------------------------------------------


class GradientNormCallback(keras.callbacks.Callback):
    """Log global gradient L2 norm per epoch on a fixed probe batch.

    Adapts the pattern at ``src/dl_techniques/optimization/train_vision/
    framework.py:506-519``. Lazily captures a probe batch from
    ``calibration_data`` on the first ``on_epoch_end`` call.

    Output CSV columns: ``epoch, grad_norm_global, grad_norm_max``.

    :param calibration_data: A ``(x, y)`` tuple of NumPy/TF tensors used as
        the probe batch. Should be small (≤32 examples) — this is a
        diagnostic, not a training step.
    :param loss_fn: Loss callable matching the model's compile loss. If
        ``None``, attempts to read ``model.compiled_loss`` / ``model.loss``.
    :param out_dir: Directory where ``grad_norm.csv`` is written.
    """

    def __init__(
        self,
        calibration_data: Tuple[tf.Tensor, tf.Tensor],
        out_dir: str,
        loss_fn: Optional[keras.losses.Loss] = None,
    ) -> None:
        super().__init__()
        self._x, self._y = calibration_data
        self._out_dir = out_dir
        self._loss_fn = loss_fn
        self._csv_path = os.path.join(out_dir, "grad_norm.csv")
        self._initialized = False

    def _init_csv(self) -> None:
        _ensure_dir(self._out_dir)
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "grad_norm_global", "grad_norm_max"])
        self._initialized = True

    def _resolve_loss(self) -> keras.losses.Loss:
        if self._loss_fn is not None:
            return self._loss_fn
        compiled = getattr(self.model, "compiled_loss", None)
        if compiled is not None and getattr(compiled, "_user_loss", None) is not None:
            return compiled._user_loss
        if getattr(self.model, "loss", None) is not None:
            return self.model.loss
        # Fallback for regression-style models with no compile loss:
        return keras.losses.MeanSquaredError()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if not self._initialized:
            self._init_csv()
        loss_fn = self._resolve_loss()
        x = tf.convert_to_tensor(self._x)
        y = tf.convert_to_tensor(self._y)
        with tf.GradientTape() as tape:
            preds = self.model(x, training=False)
            if isinstance(preds, dict):
                # Pick the first output for the gradient probe.
                preds = next(iter(preds.values()))
            loss = loss_fn(y, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)
        finite_grads = [g for g in grads if g is not None]
        if not finite_grads:
            global_norm = float("nan")
            max_norm = float("nan")
        else:
            global_norm = float(tf.linalg.global_norm(finite_grads).numpy())
            max_norm = float(
                tf.reduce_max(
                    tf.stack([tf.norm(g) for g in finite_grads])
                ).numpy()
            )
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, global_norm, max_norm])


# ---------------------------------------------------------------------
# 2. WeightNormTrajectoryCallback
# ---------------------------------------------------------------------


class WeightNormTrajectoryCallback(keras.callbacks.Callback):
    """Log L2 norm of every weight tensor on the targeted norm layers.

    Specifically:
    - ``RMSNorm``/``ZeroCenteredRMSNorm`` with ``use_scale=True``: ``scale``
      L2 norm — the γ-growth probe.
    - ``BandRMS``/``ZeroCenteredBandRMSNorm``: ``band_param`` value (scalar).

    Output CSV columns: ``epoch, layer_name, weight_name, l2``.

    :param target_classes: Tuple of norm-variant classes to track.
    :param out_dir: Directory where ``weight_norm.csv`` is written.
    """

    def __init__(
        self,
        out_dir: str,
        target_classes: Tuple[type, ...] = NORM_LAYER_CLASSES,
    ) -> None:
        super().__init__()
        self._out_dir = out_dir
        self._target_classes = target_classes
        self._csv_path = os.path.join(out_dir, "weight_norm.csv")
        self._initialized = False
        self._target_layers: List[keras.layers.Layer] = []

    def _init(self) -> None:
        _ensure_dir(self._out_dir)
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "layer_name", "weight_name", "l2"])
        self._target_layers = _find_norm_layers(self.model, self._target_classes)
        logger.info(
            f"[WeightNormTrajectoryCallback] tracking "
            f"{len(self._target_layers)} norm layers"
        )
        self._initialized = True

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if not self._initialized:
            self._init()
        rows: List[Tuple[int, str, str, float]] = []
        for layer in self._target_layers:
            for weight in layer.weights:
                val = keras.ops.convert_to_numpy(weight)
                l2 = float(np.sqrt((val.astype(np.float64) ** 2).sum()))
                weight_name = weight.name.split("/")[-1]
                rows.append((epoch, layer.name, weight_name, l2))
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


# ---------------------------------------------------------------------
# 3. NormLayerActivationCallback
# ---------------------------------------------------------------------


class NormLayerActivationCallback(keras.callbacks.Callback):
    """Log per-norm-layer activation statistics on a fixed calibration batch.

    Specifically: ``mean(activation)`` (zero-mean-output probe) and
    ``std(per_sample_RMS(activation))`` (thick-shell band probe).

    Implementation: at first ``on_epoch_end`` build an intermediate
    :class:`keras.Model` mapping the original model's input to every
    targeted norm-layer's output, then run a calibration batch through it
    once per epoch. The intermediate model shares weights with the original
    — no separate training state.

    Output CSV columns: ``epoch, layer_name, mean, per_sample_rms_std,
    per_sample_rms_mean, per_sample_rms_min, per_sample_rms_max``.

    :param calibration_data: A 2D / 3D / 4D tensor of inputs (no labels).
        Used as-is — no augmentation, no shuffling.
    :param target_classes: Tuple of norm-variant classes to track.
    :param out_dir: Output directory.
    """

    def __init__(
        self,
        calibration_data: tf.Tensor,
        out_dir: str,
        target_classes: Tuple[type, ...] = NORM_LAYER_CLASSES,
    ) -> None:
        super().__init__()
        self._x = calibration_data
        self._out_dir = out_dir
        self._target_classes = target_classes
        self._csv_path = os.path.join(out_dir, "activation_stats.csv")
        self._initialized = False
        self._target_layers: List[keras.layers.Layer] = []
        self._probe_model: Optional[keras.Model] = None

    def _init(self) -> None:
        _ensure_dir(self._out_dir)
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "epoch",
                        "layer_name",
                        "mean",
                        "per_sample_rms_std",
                        "per_sample_rms_mean",
                        "per_sample_rms_min",
                        "per_sample_rms_max",
                    ]
                )
        self._target_layers = _find_norm_layers(self.model, self._target_classes)
        logger.info(
            f"[NormLayerActivationCallback] tracking "
            f"{len(self._target_layers)} norm layers"
        )
        # Try Functional intermediate-model construction first (works on
        # Functional models like the E5 microbench). Subclassed models
        # (ViT, ResNet) fall back to a monkey-patched call() tap.
        try:
            outputs = [layer.output for layer in self._target_layers]
            self._probe_model = keras.Model(self.model.input, outputs)
        except (AttributeError, ValueError) as e:
            logger.info(
                f"[NormLayerActivationCallback] Functional probe model "
                f"unavailable ({type(e).__name__}); using call-tap fallback."
            )
            self._probe_model = None
        self._initialized = True

    def _capture_via_call_tap(self) -> List[np.ndarray]:
        """Forward through the full model with each target layer's ``call``
        wrapped to stash its output. Restores original ``call`` on exit.
        """
        originals = []
        captured: List[Optional[np.ndarray]] = [None] * len(self._target_layers)
        for i, layer in enumerate(self._target_layers):
            originals.append((layer, layer.call))

            def _make_wrapper(idx: int, orig: callable):
                def _wrapper(inputs, *args, **kwargs):
                    out = orig(inputs, *args, **kwargs)
                    try:
                        captured[idx] = keras.ops.convert_to_numpy(out)
                    except (TypeError, ValueError):
                        captured[idx] = None
                    return out
                return _wrapper

            layer.call = _make_wrapper(i, layer.call)
        try:
            _ = self.model(self._x, training=False)
        finally:
            for layer, orig in originals:
                layer.call = orig
        # Replace any None entries with a (1,)-shape sentinel.
        return [c if c is not None else np.zeros((1,), dtype=np.float32) for c in captured]

    @staticmethod
    def _activation_stats(act: np.ndarray) -> Tuple[float, float, float, float, float]:
        a = act.astype(np.float64)
        # Reduce over all axes except the leading batch axis to get per-sample RMS.
        axes_to_reduce = tuple(range(1, a.ndim)) if a.ndim > 1 else (0,)
        per_sample_rms = np.sqrt((a ** 2).mean(axis=axes_to_reduce))
        mean = float(a.mean())
        return (
            mean,
            float(per_sample_rms.std()),
            float(per_sample_rms.mean()),
            float(per_sample_rms.min()),
            float(per_sample_rms.max()),
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if not self._initialized:
            self._init()
        if not self._target_layers:
            return
        if self._probe_model is not None:
            outs = self._probe_model(self._x, training=False)
            if not isinstance(outs, list):
                outs = [outs]
            acts = [keras.ops.convert_to_numpy(o) for o in outs]
        else:
            # Subclassed-model fallback: temporarily wrap each target
            # norm-layer's ``call`` to stash its output, run a forward
            # through the full model, then restore.
            acts = self._capture_via_call_tap()
        rows: List[Tuple] = []
        for layer, act in zip(self._target_layers, acts):
            mean, rms_std, rms_mean, rms_min, rms_max = self._activation_stats(act)
            rows.append(
                (epoch, layer.name, mean, rms_std, rms_mean, rms_min, rms_max)
            )
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


# ---------------------------------------------------------------------
# 4. NormInternalStatsCallback
# ---------------------------------------------------------------------


class NormInternalStatsCallback(keras.callbacks.Callback):
    """Log internal state of each norm layer.

    For ``RMSNorm`` / ``ZeroCenteredRMSNorm`` (when ``use_scale=True``):
    L2 of the per-feature ``scale`` parameter.

    For ``BandRMS`` / ``ZeroCenteredBandRMSNorm``: raw ``band_param`` value
    AND the post-sigmoid scale ``(1-α) + α·sigmoid(5·band_param)`` —
    the latter is what actually multiplies the activations.

    Output CSV columns:
    ``epoch, layer_name, kind, scale_l2_or_raw, post_sigmoid_scale``.
    For RMSNorm-family rows, ``post_sigmoid_scale`` is NaN.
    For BandRMS-family rows, ``scale_l2_or_raw`` is the raw band_param.

    :param out_dir: Output directory.
    :param target_classes: Tuple of norm classes to track.
    """

    def __init__(
        self,
        out_dir: str,
        target_classes: Tuple[type, ...] = NORM_LAYER_CLASSES,
    ) -> None:
        super().__init__()
        self._out_dir = out_dir
        self._target_classes = target_classes
        self._csv_path = os.path.join(out_dir, "norm_internal.csv")
        self._initialized = False
        self._target_layers: List[keras.layers.Layer] = []

    def _init(self) -> None:
        _ensure_dir(self._out_dir)
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "epoch",
                        "layer_name",
                        "kind",
                        "scale_l2_or_raw",
                        "post_sigmoid_scale",
                    ]
                )
        self._target_layers = _find_norm_layers(self.model, self._target_classes)
        logger.info(
            f"[NormInternalStatsCallback] tracking "
            f"{len(self._target_layers)} norm layers"
        )
        self._initialized = True

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        if not self._initialized:
            self._init()
        rows: List[Tuple] = []
        for layer in self._target_layers:
            if isinstance(layer, (RMSNorm, ZeroCenteredRMSNorm)):
                if getattr(layer, "use_scale", False) and getattr(layer, "scale", None) is not None:
                    val = keras.ops.convert_to_numpy(layer.scale)
                    l2 = float(np.sqrt((val.astype(np.float64) ** 2).sum()))
                else:
                    l2 = 0.0
                rows.append((epoch, layer.name, "rms_family", l2, float("nan")))
            elif isinstance(layer, (BandRMS, ZeroCenteredBandRMSNorm)):
                if getattr(layer, "band_param", None) is not None:
                    raw = float(keras.ops.convert_to_numpy(layer.band_param))
                    alpha = float(layer.max_band_width)
                    sig = 1.0 / (1.0 + np.exp(-5.0 * raw))
                    post = (1.0 - alpha) + alpha * sig
                else:
                    raw = float("nan")
                    post = float("nan")
                rows.append((epoch, layer.name, "band_family", raw, post))
            elif isinstance(layer, AdaptiveBandRMS):
                # AdaptiveBandRMS: scaling comes from an inner Dense(kernel, bias)
                # projecting per-sample log(RMS) to a per-feature scale logit.
                # Record:
                #   scale_l2_or_raw = L2(kernel)
                #   post_sigmoid_scale = (1-α) + α·σ(5·mean(bias))  — the
                #     scale produced by the layer when log_rms=0 (i.e. for an
                #     input whose aggregate RMS already equals 1 after the
                #     internal normalize step). This is a stable summary
                #     statistic across batches; the per-sample scale itself
                #     is data-dependent and is captured by
                #     NormLayerActivationCallback (act_per_sample_rms_*).
                dense = getattr(layer, "dense_layer", None)
                if (
                    dense is not None
                    and getattr(dense, "kernel", None) is not None
                    and getattr(dense, "bias", None) is not None
                ):
                    kernel = keras.ops.convert_to_numpy(dense.kernel).astype(np.float64)
                    bias = keras.ops.convert_to_numpy(dense.bias).astype(np.float64)
                    kernel_l2 = float(np.sqrt((kernel ** 2).sum()))
                    alpha = float(layer.max_band_width)
                    sig = 1.0 / (1.0 + np.exp(-5.0 * float(bias.mean())))
                    post = (1.0 - alpha) + alpha * sig
                else:
                    kernel_l2 = float("nan")
                    post = float("nan")
                rows.append(
                    (epoch, layer.name, "adaptive_band_family", kernel_l2, post)
                )
            elif isinstance(layer, BandLogitNorm):
                # BandLogitNorm: scaling comes from the inner LayerNormalization's
                # γ/β applied to the per-sample L2 scalar, then `tanh(4·…)`,
                # then mapped to [1-α, 1]. Record:
                #   scale_l2_or_raw = L2(gamma) of the inner LN
                #   post_sigmoid_scale = NaN  — the variant uses tanh, not sigmoid;
                #     the post-tanh saturation profile is data-dependent and is
                #     captured row-wise by NormLayerActivationCallback.
                inner_ln = getattr(layer, "norm", None)
                gamma = getattr(inner_ln, "gamma", None) if inner_ln is not None else None
                if gamma is not None:
                    g = keras.ops.convert_to_numpy(gamma).astype(np.float64)
                    gamma_l2 = float(np.sqrt((g ** 2).sum()))
                else:
                    gamma_l2 = float("nan")
                rows.append(
                    (epoch, layer.name, "band_logit_family", gamma_l2, float("nan"))
                )
            elif isinstance(layer, DynamicTanh):
                # DynamicTanh: output = weight * tanh(alpha * x) + bias.
                # Record:
                #   scale_l2_or_raw = float(alpha)  — the scalar gain on the tanh.
                #   post_sigmoid_scale = NaN  — no sigmoid mechanism.
                # weight/bias trajectories are picked up by
                # WeightNormTrajectoryCallback (which iterates layer.weights).
                if getattr(layer, "alpha", None) is not None:
                    alpha_raw = float(keras.ops.convert_to_numpy(layer.alpha))
                else:
                    alpha_raw = float("nan")
                rows.append(
                    (epoch, layer.name, "dyt_family", alpha_raw, float("nan"))
                )
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


__all__ = [
    "GradientNormCallback",
    "WeightNormTrajectoryCallback",
    "NormLayerActivationCallback",
    "NormInternalStatsCallback",
    "NORM_LAYER_CLASSES",
]
