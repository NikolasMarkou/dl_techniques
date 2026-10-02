"""Training pipeline wrapper for LightGlue: frozen SuperPoint front end plus matcher.

One trainer-local ``keras.Model`` runs the whole stage-1 step inside ``call``, so
stock ``fit`` trains it with no custom ``train_step``:

.. code-block:: text

    image0, image1 ──► frozen SuperPoint (float32, training=False)
                              │
                              ▼
                       decode_superpoint ──► keypoints, descriptors, masks
                              │                      │
        H0to1, image sizes ──►│ homography_matches   │
                              ▼                      ▼
                         labels (-2/-1/j)      LightGlue (static, masked)
                              │                      │
                              └──────► LightGlueLoss.compute ──► add_loss
                                       KeypointMatchMetric  ──► fit logs

The loss is registered with ``add_loss`` and the metrics are updated inside
``call`` (both measured working under stock ``fit`` and ``evaluate``, D-013 and
the step 11 probes in ``decisions.md`` D-014). ``fit`` is called with the
dataset dict as ``x`` and no ``y``.

Checkpointing: the SuperPoint is frozen and large, so the trainer saves the
LightGlue ALONE (``LightGlueCheckpoint``, ``lightglue.keras``) instead of the
wrapper; the wrapper itself still serialises (``get_config`` carries both
sub-model configs).

Importing this module creates no tensor and does not initialise any device.
"""

import os
from typing import Any, Dict, Optional, Tuple

import keras

from dl_techniques.losses.lightglue_loss import LightGlueLoss, pack_matches
from dl_techniques.metrics.keypoint_matching import KeypointMatchMetric
from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue
from dl_techniques.models.vision.keypoints.superpoint.model import SuperPoint
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.keypoint_extraction import decode_superpoint
from dl_techniques.utils.keypoint_matching import homography_matches, label_statistics
from dl_techniques.utils.logger import logger

from train.common.callbacks import resolve_monitor_mode

# ---------------------------------------------------------------------

#: Input keys the wrapper reads (the dict emitted by ``train.lightglue.data``).
INPUT_KEYS: Tuple[str, ...] = ("image0", "image1", "H0to1", "image_size0", "image_size1")


def freeze(model: keras.Model) -> None:
    """Make ``model`` and every sublayer and variable non-trainable.

    Plain ``model.trainable = False`` is NOT enough: Keras 3.8 does not propagate the
    value to sublayers and variables when the layer's own flag is already ``False``,
    which is the state of a SuperPoint deserialised from a config saved frozen (a reloaded
    wrapper): its encoder and every weight stayed trainable and the wrapper's
    ``trainable_weights`` contained them. Toggling through ``True`` forces the propagation.

    :param model: The layer to freeze (idempotent).
    """
    model.trainable = True
    model.trainable = False


def load_superpoint(path: str, image_size: Optional[Tuple[int, int]] = None) -> SuperPoint:
    """Load a trained SuperPoint ``.keras`` checkpoint as a frozen float32 model.

    The checkpoint is loaded with the global dtype policy temporarily forced to
    ``float32`` (restored afterwards), so a mixed policy active in the caller cannot
    leak into layers the archive does not pin: ``SuperPoint`` computes
    ``np.finfo(compute_dtype)`` and fails under ``mixed_bfloat16``.

    :param path: ``.keras`` file written by ``train.superpoint`` (``final_model.keras``).
    :param image_size: Optional ``(height, width)`` the caller will feed. The descriptor
        map is resized to the construction-time size, so a different size is an error.
    :return: The model with ``trainable = False``.
    :raises FileNotFoundError: ``path`` is not a file.
    :raises TypeError: The archive does not hold a ``SuperPoint``.
    :raises ValueError: The model computes in a dtype other than float32, or its input
        size differs from ``image_size``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"SuperPoint checkpoint not found: {path!r}. Train one with "
            "train.superpoint.train_superpoint and pass its final_model.keras."
        )
    previous = keras.config.dtype_policy()
    keras.config.set_dtype_policy("float32")
    try:
        model = keras.saving.load_model(path, compile=False)
    finally:
        keras.config.set_dtype_policy(previous)
    if not isinstance(model, SuperPoint):
        raise TypeError(f"{path!r} holds a {type(model).__name__}, not a SuperPoint")
    if model.compute_dtype != "float32":
        raise ValueError(
            f"SuperPoint at {path!r} computes in {model.compute_dtype}; the frozen front "
            "end must run in float32"
        )
    if image_size is not None and (model.input_height, model.input_width) != tuple(image_size):
        raise ValueError(
            f"image size {tuple(image_size)} does not match the SuperPoint checkpoint's "
            f"input size {(model.input_height, model.input_width)}: its descriptor map is "
            "fixed to the construction-time size"
        )
    freeze(model)
    return model


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.train.lightglue.pipeline")
class LightGlueTrainingModel(keras.Model):
    """Frozen SuperPoint + LightGlue with the loss and metrics inside ``call``.

    **Interface contract**. ``call(inputs)`` takes the dict of
    :func:`train.lightglue.data.make_pair_dataset` (``image0``, ``image1`` ``(B, H, W, 1)``,
    ``H0to1`` ``(B, 3, 3)``, ``image_size0/1`` ``(B, 2)`` as ``(w, h)``), runs the front
    end, labels, matcher and objective, registers the batch-mean objective with
    ``add_loss`` and returns the LightGlue output dict. ``H, W`` must equal the
    SuperPoint's construction-time size (checked statically, ``ValueError``).

    Compile it with ``model.compile(optimizer=..., jit_compile=False)`` (no loss) and
    ``fit(dataset)`` with no ``y``. ``trainable_weights`` are the LightGlue's only.

    :param superpoint: A float32 ``SuperPoint``; frozen here (``trainable = False``) and
        always called with ``training=False``.
    :param lightglue: A ``LightGlue`` whose ``input_dim`` equals the SuperPoint
        ``descriptor_dim``.
    :param max_keypoints: Padded keypoint count per image.
    :param detection_threshold: Minimum heatmap probability of a keypoint.
    :param nms_radius: Non-maximum-suppression radius in pixels.
    :param border: Edge margin in pixels without keypoints.
    :param pos_threshold: Reprojection distance (pixels) of a positive pair; it is also
        the dustbin threshold (glue-factory uses 3 px for both).
    :param kwargs: Forwarded to ``keras.Model``. ``autocast`` is forced off so the
        homography and image sizes keep float32 under a mixed policy.
    :raises ValueError: Non-float32 SuperPoint, descriptor width mismatch, or a
        non-positive count.
    """

    def __init__(
        self,
        superpoint: SuperPoint,
        lightglue: LightGlue,
        max_keypoints: int = 512,
        detection_threshold: float = 0.005,
        nms_radius: int = 4,
        border: int = 4,
        pos_threshold: float = 3.0,
        **kwargs: Any,
    ) -> None:
        kwargs["autocast"] = False
        super().__init__(**kwargs)
        if superpoint.compute_dtype != "float32":
            raise ValueError(
                f"the frozen SuperPoint must compute in float32, got {superpoint.compute_dtype}"
            )
        if lightglue.input_dim != superpoint.descriptor_dim:
            raise ValueError(
                f"LightGlue input_dim ({lightglue.input_dim}) must equal the SuperPoint "
                f"descriptor_dim ({superpoint.descriptor_dim})"
            )
        if max_keypoints < 1 or nms_radius < 0 or border < 0 or pos_threshold <= 0:
            raise ValueError(
                "max_keypoints must be >= 1, nms_radius and border >= 0 and pos_threshold "
                f"> 0, got {max_keypoints}, {nms_radius}, {border}, {pos_threshold}"
            )
        self.superpoint = superpoint
        freeze(self.superpoint)
        self.lightglue = lightglue
        self.max_keypoints = int(max_keypoints)
        self.detection_threshold = float(detection_threshold)
        self.nms_radius = int(nms_radius)
        self.border = int(border)
        self.pos_threshold = float(pos_threshold)

        self.objective = LightGlueLoss()
        threshold = lightglue.filter_threshold
        self.precision_metric = KeypointMatchMetric("precision", threshold, name="precision")
        self.recall_metric = KeypointMatchMetric("recall", threshold, name="recall")
        self.keypoints_metric = keras.metrics.Mean(name="keypoints_per_image")
        self.positive_metric = keras.metrics.Mean(name="positive_fraction")

    # -----------------------------------------------------------------

    def build(self, input_shape: Optional[Dict[str, Tuple[Any, ...]]] = None) -> None:
        """Build both sub-models from the static shapes (no tracing of ``call``).

        :param input_shape: Dict carrying at least ``image0``'s shape ``(B, H, W, C)``.
        """
        if self.built:
            return
        batch = input_shape["image0"][0] if isinstance(input_shape, dict) else None
        if not self.superpoint.built:
            self.superpoint.build(
                (batch, self.superpoint.input_height, self.superpoint.input_width,
                 self.superpoint.input_channels))
        # Freeze AGAIN after the build: variables created by `build` start trainable.
        freeze(self.superpoint)
        n, d = self.max_keypoints, self.superpoint.descriptor_dim
        self.lightglue.build({
            "keypoints0": (batch, n, 2), "keypoints1": (batch, n, 2),
            "descriptors0": (batch, n, d), "descriptors1": (batch, n, d),
            "image_size0": (batch, 2), "image_size1": (batch, 2),
        })
        super().build(input_shape)

    def _extract(self, image0: Any, image1: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Frozen detection and description of both images in one SuperPoint pass."""
        batch = keras.ops.concatenate([image0, image1], axis=0)
        # training=False explicitly: the front end is a fixed feature source.
        features = self.superpoint(batch, training=False)
        decoded = decode_superpoint(
            features, self.max_keypoints, self.detection_threshold,
            self.nms_radius, self.border)
        decoded = {k: keras.ops.stop_gradient(v) for k, v in decoded.items()}
        size = keras.ops.shape(image0)[0]
        first = {k: v[:size] for k, v in decoded.items()}
        second = {k: v[size:] for k, v in decoded.items()}
        return first, second

    def _labelled(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Any, Any, Any, Any]:
        """Detect on both images and label the keypoints from ``H0to1``.

        :param inputs: Dict with the keys of :data:`INPUT_KEYS`.
        :return: ``(det0, det1, mask0, mask1, labels0, labels1)``; masks are float32
            ``(B, N)``, labels int32 ``(B, N)`` (``j`` / -1 dustbin / -2 ignored).
        :raises ValueError: Missing key, or an image size that differs from the
            SuperPoint's construction-time size.
        """
        missing = [key for key in INPUT_KEYS if key not in inputs]
        if missing:
            raise ValueError(f"inputs lack {missing}; expected the keys {INPUT_KEYS}")
        expected = (self.superpoint.input_height, self.superpoint.input_width)
        for key in ("image0", "image1"):
            shape = tuple(inputs[key].shape[1:3])
            if shape != expected:
                raise ValueError(
                    f"{key} has spatial size {shape}, the SuperPoint expects {expected}")

        det0, det1 = self._extract(inputs["image0"], inputs["image1"])
        mask0 = keras.ops.cast(det0["mask"], "float32")
        mask1 = keras.ops.cast(det1["mask"], "float32")
        labels = homography_matches(
            det0["keypoints"], det1["keypoints"], mask0, mask1, inputs["H0to1"],
            inputs["image_size0"], inputs["image_size1"],
            pos_threshold=self.pos_threshold, neg_threshold=self.pos_threshold)
        labels0 = keras.ops.stop_gradient(labels["matches0"])
        labels1 = keras.ops.stop_gradient(labels["matches1"])
        return det0, det1, mask0, mask1, labels0, labels1

    def batch_statistics(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """Label fractions and keypoint count of one batch, for run diagnostics.

        Eager. ``positive``, ``dustbin`` and ``ignored`` are fractions of the real
        keypoints of image 0 (:func:`dl_techniques.utils.keypoint_matching.label_statistics`).

        :param inputs: Dict with the keys of :data:`INPUT_KEYS`.
        :return: Plain floats ``positive``, ``dustbin``, ``ignored`` and
            ``keypoints_per_image`` (mean real keypoints of image 0).
        """
        _, _, mask0, _, labels0, _ = self._labelled(inputs)
        stats = {k: float(v) for k, v in label_statistics(labels0, mask0).items()}
        stats["keypoints_per_image"] = float(keras.ops.mean(keras.ops.sum(mask0, axis=1)))
        return stats

    def call(self, inputs: Dict[str, Any], training: Optional[bool] = None) -> Dict[str, Any]:
        """One pipeline step: detect, label, match, register the loss, update metrics.

        :param inputs: Dict with the keys of :data:`INPUT_KEYS`.
        :param training: Forwarded to the LightGlue (which has no stochastic layer).
        :return: The LightGlue output dict.
        :raises ValueError: See :meth:`_labelled`.
        """
        det0, det1, mask0, mask1, labels0, labels1 = self._labelled(inputs)

        output = self.lightglue({
            "keypoints0": det0["keypoints"], "keypoints1": det1["keypoints"],
            "descriptors0": det0["descriptors"], "descriptors1": det1["descriptors"],
            "image_size0": inputs["image_size0"], "image_size1": inputs["image_size1"],
            "mask0": mask0, "mask1": mask1,
        }, training=training)

        # The masks must reach the confidence term: padded entries are 0 in the log
        # assignments and would otherwise win the argmax target (D-012).
        per_sample = self.objective.compute(
            output["log_assignments"], labels0, labels1,
            output["token_confidences0"], output["token_confidences1"], mask0, mask1)
        self.add_loss(keras.ops.mean(per_sample))

        packed = pack_matches(labels0, labels1)
        for metric in (self.precision_metric, self.recall_metric):
            metric.update_state(packed, output["log_assignments"], mask0=mask0, mask1=mask1)
        self.keypoints_metric.update_state(
            keras.ops.mean(keras.ops.sum(mask0, axis=1)))
        self.positive_metric.update_state(label_statistics(labels0, mask0)["positive"])
        return output

    def compute_output_shape(self, input_shape: Dict[str, Tuple[Any, ...]]) -> Dict[str, Tuple]:
        """Output shapes of the LightGlue dict for a batch of the given images."""
        batch, n = input_shape["image0"][0], self.max_keypoints
        d = self.superpoint.descriptor_dim
        return self.lightglue.compute_output_shape({
            "keypoints0": (batch, n, 2), "keypoints1": (batch, n, 2),
            "descriptors0": (batch, n, d), "descriptors1": (batch, n, d)})

    def get_config(self) -> Dict[str, Any]:
        """Both sub-model configs plus the pipeline hyper-parameters."""
        config = super().get_config()
        config.update({
            "superpoint": keras.saving.serialize_keras_object(self.superpoint),
            "lightglue": keras.saving.serialize_keras_object(self.lightglue),
            "max_keypoints": self.max_keypoints,
            "detection_threshold": self.detection_threshold,
            "nms_radius": self.nms_radius,
            "border": self.border,
            "pos_threshold": self.pos_threshold,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], custom_objects: Any = None) -> "LightGlueTrainingModel":
        """Rebuild both sub-models from their serialised configs."""
        config = dict(config)
        config["superpoint"] = keras.saving.deserialize_keras_object(
            config["superpoint"], custom_objects)
        config["lightglue"] = keras.saving.deserialize_keras_object(
            config["lightglue"], custom_objects)
        return cls(**config)


# ---------------------------------------------------------------------


# DECISION plan-2026-10-02T084508-dd2c07ac/D-014
# The best checkpoint is the LightGlue ALONE, not the wrapper. Do NOT put the stock
# ModelCheckpoint of create_callbacks back "for consistency": it saves the wrapper, which
# serialises (measured) but writes the frozen SuperPoint at every improvement; the
# evaluation script reloads only the LightGlue. Guard: test_lightglue_checkpoint_saves_the_
# lightglue_alone_on_improvement. See decisions.md D-014.
class LightGlueCheckpoint(keras.callbacks.Callback):
    """Save the LightGlue ALONE whenever the monitored value improves.

    The stock ``ModelCheckpoint`` of ``create_callbacks`` would write the wrapper,
    frozen SuperPoint included, at every improvement. This callback writes only the
    ``pipeline.lightglue`` sub-model, which is what the evaluation script reloads.
    It is not serialised with a model (callbacks never are).

    :param filepath: ``.keras`` path (use ``best_checkpoint_path(run_dir)``).
    :param monitor: Log key, e.g. ``val_loss``.
    :param mode: ``min``, ``max`` or None to resolve it from the key name with
        :func:`train.common.callbacks.resolve_monitor_mode`.
    """

    def __init__(self, filepath: str, monitor: str = "val_loss", mode: Optional[str] = None) -> None:
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = resolve_monitor_mode(monitor, mode)
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save ``model.lightglue`` when ``logs[monitor]`` is a finite improvement."""
        value = (logs or {}).get(self.monitor)
        if value is None or value != value:
            logger.warning(f"LightGlueCheckpoint: {self.monitor!r} missing or NaN at epoch {epoch}")
            return
        value = float(value)
        if self._improved(value):
            self.best, self.best_epoch = value, epoch
            self.model.lightglue.save(self.filepath)
            logger.info(
                f"Epoch {epoch + 1}: {self.monitor} improved to {value:.6g}, saved "
                f"{self.filepath}")
