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
from dl_techniques.layers.norms.zero_centered_adaptive_band_rms_norm import (
    ZeroCenteredAdaptiveBandRMS,
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
    ZeroCenteredAdaptiveBandRMS,
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
            elif isinstance(layer, ZeroCenteredAdaptiveBandRMS):
                # ZeroCenteredAdaptiveBandRMS: mirrors AdaptiveBandRMS — an
                # inner Dense(kernel, bias) projects per-sample log(RMS) to a
                # per-feature scale logit. The zero-centered branch first
                # subtracts the mean, otherwise the wiring is identical.
                # Record:
                #   scale_l2_or_raw = L2(kernel)
                #   post_sigmoid_scale = (1-α) + α·σ(5·mean(bias))
                # ``kind="zc_adaptive_band_family"`` distinguishes this from
                # the non-zero-centered cousin in downstream reports.
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
                    (epoch, layer.name, "zc_adaptive_band_family", kernel_l2, post)
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


# ---------------------------------------------------------------------
# 5. CalibrationCallback + 6. RobustnessProbe
# ---------------------------------------------------------------------
#
# DECISION plan_2026-05-18_63121227/D-003: two end-of-training-only probes
# (CalibrationCallback, RobustnessProbe) at the cost of ~150 LOC in this
# file. Run at ``on_train_end`` only — single inference pass each —
# keeping per-cell wall-clock overhead in the seconds range. Both wrap
# their core in try/except so a probe failure does NOT poison the cell's
# headline metric (LESSONS L74 robustness pattern).


class CalibrationCallback(keras.callbacks.Callback):
    """Compute classification calibration metrics at end of training.

    Two metrics over the validation set:
    - **ECE-15** (Expected Calibration Error, 15 equal-width bins on the
      max-softmax confidence): weighted sum over bins of ``|acc - conf|``.
    - **Brier score** (multi-class): mean over samples of
      ``sum((p_i - y_onehot_i)^2)``.

    Output: one-row CSV ``calibration.csv`` with columns
    ``ece_15, brier_score, n_samples, n_classes, n_bins``.

    Classification only. Gated on ``num_classes >= 2`` and on a label-shape
    rank that can be reduced to integer class IDs (rank-1 sparse, or rank-2
    one-hot/probability).

    :param val_data: ``(x, y)`` validation tuple. ``x`` can be any tensor
        the model accepts; ``y`` is rank-1 sparse OR rank-2 one-hot.
    :param num_classes: Number of classes (>= 2).
    :param out_dir: Output directory.
    :param n_bins: Number of equal-width bins for ECE. Default 15.
    """

    def __init__(
        self,
        val_data: Tuple[tf.Tensor, tf.Tensor],
        num_classes: int,
        out_dir: str,
        n_bins: int = 15,
    ) -> None:
        super().__init__()
        self._x, self._y = val_data
        self._num_classes = int(num_classes)
        self._out_dir = out_dir
        self._n_bins = int(n_bins)
        self._csv_path = os.path.join(out_dir, "calibration.csv")

    @staticmethod
    def _to_class_ids(y: np.ndarray, num_classes: int) -> Optional[np.ndarray]:
        """Reduce ``y`` to a rank-1 integer class-id array, or None on ambiguity."""
        if y.ndim == 1:
            return y.astype(np.int64)
        if y.ndim == 2 and y.shape[1] == num_classes:
            return np.argmax(y, axis=1).astype(np.int64)
        return None

    def _compute(self, probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Return (ECE-15, Brier). ``probs`` is (N, C), ``labels`` is (N,)."""
        n, c = probs.shape
        # ECE
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct = (predictions == labels).astype(np.float64)
        bin_edges = np.linspace(0.0, 1.0, self._n_bins + 1)
        ece = 0.0
        for i in range(self._n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == self._n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        # Brier (multi-class)
        onehot = np.zeros((n, c), dtype=np.float64)
        onehot[np.arange(n), labels] = 1.0
        brier = float(((probs - onehot) ** 2).sum(axis=1).mean())
        return float(ece), brier

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        try:
            if self._num_classes < 2:
                logger.warning(
                    f"[CalibrationCallback] num_classes={self._num_classes} < 2; "
                    f"skipping (not a classification task)."
                )
                return
            _ensure_dir(self._out_dir)
            preds = self.model.predict(self._x, verbose=0)
            if isinstance(preds, dict):
                preds = next(iter(preds.values()))
            probs = keras.ops.convert_to_numpy(preds).astype(np.float64)
            if probs.ndim != 2 or probs.shape[1] != self._num_classes:
                logger.warning(
                    f"[CalibrationCallback] unexpected pred shape {probs.shape}; "
                    f"expected (N, {self._num_classes}). Skipping."
                )
                return
            # If outputs are logits (not in [0, 1] summing to 1), apply softmax.
            row_sums = probs.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                shift = probs - probs.max(axis=1, keepdims=True)
                exp = np.exp(shift)
                probs = exp / exp.sum(axis=1, keepdims=True)
            y_arr = keras.ops.convert_to_numpy(self._y)
            labels = self._to_class_ids(y_arr, self._num_classes)
            if labels is None:
                logger.warning(
                    f"[CalibrationCallback] ambiguous label shape {y_arr.shape}; "
                    f"emitting nan."
                )
                ece, brier = float("nan"), float("nan")
            else:
                ece, brier = self._compute(probs, labels)
            with open(self._csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ece_15", "brier_score", "n_samples", "n_classes", "n_bins"])
                w.writerow([ece, brier, int(probs.shape[0]), self._num_classes, self._n_bins])
        except (RuntimeError, ValueError) as e:
            logger.error(f"[CalibrationCallback] failed: {type(e).__name__}: {e}")


class RobustnessProbe(keras.callbacks.Callback):
    """Measure input-perturbation robustness at end of training.

    For each ``sigma`` in ``sigmas`` (Gaussian standard deviation), evaluate
    the model on ``val_data + Normal(0, sigma)`` noise and record validation
    accuracy. The clean (sigma=0) baseline is included automatically as the
    first row for delta computation downstream.

    Output: ``robustness.csv`` with columns ``sigma, val_acc, n_samples``.

    Classification only — relies on ``model.predict`` returning a (N, C)
    logit/prob tensor and on ``y`` being reducible to integer class IDs.

    :param val_data: ``(x, y)`` validation tuple.
    :param out_dir: Output directory.
    :param sigmas: Tuple of noise std-devs. Default ``(0.01, 0.05, 0.1, 0.2)``.
    """

    def __init__(
        self,
        val_data: Tuple[tf.Tensor, tf.Tensor],
        out_dir: str,
        sigmas: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2),
    ) -> None:
        super().__init__()
        self._x, self._y = val_data
        self._out_dir = out_dir
        self._sigmas = tuple(float(s) for s in sigmas)
        self._csv_path = os.path.join(out_dir, "robustness.csv")

    @staticmethod
    def _accuracy(preds: np.ndarray, y: np.ndarray) -> Optional[float]:
        if preds.ndim != 2:
            return None
        pred_ids = preds.argmax(axis=1)
        if y.ndim == 1:
            labels = y.astype(np.int64)
        elif y.ndim == 2:
            labels = np.argmax(y, axis=1).astype(np.int64)
        else:
            return None
        return float((pred_ids == labels).mean())

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        try:
            _ensure_dir(self._out_dir)
            x_np = keras.ops.convert_to_numpy(self._x).astype(np.float32)
            y_np = keras.ops.convert_to_numpy(self._y)
            rng = np.random.default_rng(seed=0)
            rows: List[Tuple[float, float, int]] = []
            # Include clean baseline at sigma=0.0
            sigmas_full = (0.0,) + self._sigmas
            for sigma in sigmas_full:
                if sigma == 0.0:
                    x_noisy = x_np
                else:
                    noise = rng.normal(loc=0.0, scale=sigma, size=x_np.shape).astype(np.float32)
                    x_noisy = x_np + noise
                preds = self.model.predict(x_noisy, verbose=0)
                if isinstance(preds, dict):
                    preds = next(iter(preds.values()))
                preds_np = keras.ops.convert_to_numpy(preds)
                acc = self._accuracy(preds_np, y_np)
                if acc is None:
                    logger.warning(
                        f"[RobustnessProbe] could not compute accuracy "
                        f"(pred shape {preds_np.shape}, y shape {y_np.shape}); "
                        f"skipping sigma={sigma}."
                    )
                    continue
                rows.append((float(sigma), float(acc), int(x_np.shape[0])))
            with open(self._csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sigma", "val_acc", "n_samples"])
                w.writerows(rows)
        except (RuntimeError, ValueError) as e:
            logger.error(f"[RobustnessProbe] failed: {type(e).__name__}: {e}")


__all__ = [
    "GradientNormCallback",
    "WeightNormTrajectoryCallback",
    "NormLayerActivationCallback",
    "NormInternalStatsCallback",
    "CalibrationCallback",
    "RobustnessProbe",
    "NORM_LAYER_CLASSES",
]
