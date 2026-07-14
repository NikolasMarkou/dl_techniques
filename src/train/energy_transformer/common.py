"""Shared plumbing for the two Energy Transformer trainers.

Deliberately SMALL (D-005). It holds only the three things both trainers genuinely share:

1. :func:`build_raw_image_dataset` — a raw-image ``tf.data`` pipeline (imagenette via tfds,
   cifar10 in-memory) with an optional per-sample ``element_map_fn`` hook. The MIM trainer
   passes ``make_masked_patch_map_fn(...)`` through that hook; the classifier passes nothing.
2. :func:`build_optimizer` — the ``learning_rate_schedule_builder`` / ``optimizer_builder``
   block.
3. :class:`EnergyTraceCallback` — the out-of-graph energy-descent probe.

There is NO shared config dataclass and NO shared ``train()`` orchestrator: each trainer owns
its own ``TrainingConfig``, ``parse_arguments()`` and ``config_from_args()`` so that an
argparse flag which never reaches the config is a LOCAL, greppable, testable defect rather
than an inherited one.
"""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

# Verified present offline at $TFDS_DATA_DIR (D-007): 9469 train / 3925 validation, 10 classes.
IMAGENETTE_TFDS_NAME = "imagenette/320px-v2"
IMAGENETTE_NUM_CLASSES = 10

DATASET_NUM_CLASSES: Dict[str, int] = {
    "imagenette": IMAGENETTE_NUM_CLASSES,
    "cifar10": 10,
}

SUPPORTED_DATASETS: Tuple[str, ...] = tuple(DATASET_NUM_CLASSES)

# Element map function: (image, label) -> whatever the trainer wants the batch element to be.
ElementMapFn = Callable[..., Any]


# ---------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------

def _normalization_constants(dataset: str) -> Tuple[List[float], List[float]]:
    """Per-channel mean/std for the [0,1]-scaled images of ``dataset``."""
    if dataset == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    return IMAGENET_MEAN, IMAGENET_STD


def _augment(image: tf.Tensor, seed: Optional[int] = None) -> tf.Tensor:
    """Random flip + reflect-pad-and-crop, on the [0,1] image, BEFORE normalization.

    Augmentation runs pre-normalization on purpose (the ``train_vit`` D-006/D-007 lesson:
    augmenting normalized data and then clipping to [0,1] saturates most pixels and silently
    creates a train/val distribution mismatch). No ``clip_by_value`` here.
    """
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_crop(
        tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT"),
        size=tf.shape(image),
        seed=seed,
    )
    return image


def build_raw_image_dataset(
        dataset: str,
        image_size: int,
        batch_size: int,
        *,
        is_training: bool,
        augment: bool = True,
        element_map_fn: Optional[ElementMapFn] = None,
        shuffle_buffer: int = 4096,
        seed: Optional[int] = None,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch_buffer: int = tf.data.AUTOTUNE,
) -> Tuple[tf.data.Dataset, int, int]:
    """Build a raw-image ``tf.data`` pipeline for imagenette or cifar10.

    The pipeline yields ``(image, label)`` — ``image`` float32, resized to
    ``(image_size, image_size, 3)``, scaled to ``[0, 1]`` and per-channel normalized; ``label``
    an ``int32`` scalar. When ``element_map_fn`` is given it is applied PER SAMPLE (before
    batching) to that pair, which is how the MIM trainer swaps in the masked-patch element
    ``((image, input_mask), target_patches, loss_weight)``.

    Args:
        dataset: ``'imagenette'`` or ``'cifar10'``.
        image_size: Side length the images are resized/cropped to.
        batch_size: Batch size. Training batches use ``drop_remainder=True``.
        is_training: Training pipeline (shuffle + repeat + augment) vs validation
            (``.cache()``, no shuffle, no repeat, no augment).
        augment: Enable train-time augmentation. Ignored when ``is_training`` is False.
        element_map_fn: Optional per-sample transform applied to ``(image, label)``.
        shuffle_buffer: Shuffle buffer for the training pipeline.
        seed: Seed for shuffling and augmentation.
        num_parallel_calls: ``tf.data`` parallelism.
        prefetch_buffer: ``tf.data`` prefetch depth.

    Returns:
        ``(ds, num_examples, num_classes)``. ``num_examples`` is the split's cardinality, from
        which the caller derives ``steps_per_epoch``.

    Raises:
        ValueError: If ``dataset`` is not supported or ``image_size``/``batch_size`` are
            non-positive.
    """
    dataset = dataset.lower()
    if dataset not in DATASET_NUM_CLASSES:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; supported: {sorted(DATASET_NUM_CLASSES)}"
        )
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    mean_vals, std_vals = _normalization_constants(dataset)
    mean = tf.constant(mean_vals, dtype=tf.float32, shape=(1, 1, 3))
    std = tf.constant(std_vals, dtype=tf.float32, shape=(1, 1, 3))

    if dataset == "imagenette":
        # Imported lazily: tfds pulls in a heavy import chain and cifar10 does not need it.
        import tensorflow_datasets as tfds

        split = "train" if is_training else "validation"
        # data_dir=None -> inherits $TFDS_DATA_DIR. The records are PREPARED on disk; no
        # download, no network (D-007). download=False makes an absent record set a LOUD
        # failure instead of a silent multi-GB fetch.
        builder = tfds.builder(IMAGENETTE_TFDS_NAME)
        num_examples = int(builder.info.splits[split].num_examples)
        ds = builder.as_dataset(split=split, shuffle_files=is_training)

        def _decode(element: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
            # Imagenette records are VARIABLE-SIZE (e.g. 320x396x3). The resize is MANDATORY:
            # without it the batch is ragged and tf.data raises at the first batch.
            image = tf.cast(element["image"], tf.float32) / 255.0
            image = tf.image.resize(image, (image_size, image_size), method="bilinear")
            image = tf.ensure_shape(image, (image_size, image_size, 3))
            return image, tf.cast(element["label"], tf.int32)

        ds = ds.map(_decode, num_parallel_calls=num_parallel_calls)
    else:  # cifar10 -- in-memory, mirroring train_vit.create_cifar_dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        images = (x_train if is_training else x_test).astype("float32") / 255.0
        labels = (y_train if is_training else y_test).flatten().astype("int32")
        num_examples = int(images.shape[0])

        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if image_size != 32:
            ds = ds.map(
                lambda img, lbl: (
                    tf.ensure_shape(
                        tf.image.resize(img, (image_size, image_size), method="bilinear"),
                        (image_size, image_size, 3),
                    ),
                    lbl,
                ),
                num_parallel_calls=num_parallel_calls,
            )

    logger.info(
        f"{dataset} [{'train' if is_training else 'validation'}]: {num_examples} examples, "
        f"resized to {image_size}x{image_size}"
    )

    if is_training:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        if augment:
            ds = ds.map(
                lambda img, lbl: (_augment(img, seed=seed), lbl),
                num_parallel_calls=num_parallel_calls,
            )

    def _normalize(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return (image - mean) / std, label

    ds = ds.map(_normalize, num_parallel_calls=num_parallel_calls)

    if not is_training:
        # Validation is small and re-read every epoch off a spinning HDD -- cache it.
        ds = ds.cache()

    if element_map_fn is not None:
        ds = ds.map(element_map_fn, num_parallel_calls=num_parallel_calls)

    if is_training:
        ds = ds.repeat().batch(batch_size, drop_remainder=True)
    else:
        ds = ds.batch(batch_size)

    ds = ds.prefetch(prefetch_buffer)
    return ds, num_examples, DATASET_NUM_CLASSES[dataset]


# ---------------------------------------------------------------------
# optimization
# ---------------------------------------------------------------------

def build_optimizer(config: Any, steps_per_epoch: int) -> keras.optimizers.Optimizer:
    """Build the LR schedule + optimizer from a trainer ``TrainingConfig``.

    Reads ``lr_schedule_type``, ``learning_rate``, ``epochs``, ``warmup_epochs``,
    ``optimizer_type``, ``weight_decay`` and ``gradient_clipping`` off ``config``.

    Double-weight-decay guard (H10 / LESSONS L72): when the optimizer is AdamW, the decay goes
    through ``optimizer_builder`` ONLY. No model layer in this feature sets a
    ``kernel_regularizer=L2(...)`` -- setting both inflates the loss with an L2 penalty AND
    decays the parameters again on the update.

    Args:
        config: The trainer's ``TrainingConfig``.
        steps_per_epoch: Optimizer steps per epoch (drives decay/warmup horizons).

    Returns:
        A configured ``keras.optimizers.Optimizer``.
    """
    lr_schedule = learning_rate_schedule_builder({
        "type": config.lr_schedule_type,
        "learning_rate": config.learning_rate,
        "decay_steps": steps_per_epoch * config.epochs,
        "warmup_steps": steps_per_epoch * config.warmup_epochs,
        "alpha": 0.01,
    })

    opt_config: Dict[str, Any] = {
        "type": config.optimizer_type,
        "gradient_clipping_by_norm": config.gradient_clipping,
    }
    if config.optimizer_type.lower() == "adamw":
        opt_config["weight_decay"] = config.weight_decay

    logger.info(
        f"optimizer={config.optimizer_type}, lr={config.learning_rate}, "
        f"schedule={config.lr_schedule_type}, warmup_steps={steps_per_epoch * config.warmup_epochs}, "
        f"clip_by_norm={config.gradient_clipping}, weight_decay={config.weight_decay}"
    )
    return optimizer_builder(opt_config, lr_schedule)


# ---------------------------------------------------------------------
# energy trace probe
# ---------------------------------------------------------------------

class EnergyTraceCallback(keras.callbacks.Callback):
    """Log the ET block's energy descent trace, OUT OF GRAPH, once per epoch.

    Invariant I5 (H4): the ``(B, T+1)`` energy trace is float32 by design and must NEVER be
    consumed by a graph layer -- under ``mixed_float16`` a default-policy head would autocast
    an O(-1e5) trace down to fp16 and overflow it to inf/nan. The models therefore refuse a
    ``return_energy=True`` backbone outright (D-010), and the trace is read HERE instead: a
    separate PROBE backbone is rebuilt from the live backbone's config with
    ``return_energy=True``, its weights are re-synced from the live model, and it is called on
    one fixed validation batch. The training graph is untouched (``return_energy=True`` costs
    ~1.28x on the graph path; this costs one forward pass per epoch).

    **The weight re-sync is load-bearing and happens EVERY epoch.** A probe whose weights are
    not re-synced logs a stale, plausible, WRONG trace -- a silent lie that looks exactly like
    a real one. The guard is that the epoch-2 trace must DIFFER from the epoch-1 trace.

    Args:
        probe_inputs: One fixed batch of model inputs -- ``(image, input_mask)`` for the MIM
            model, ``image`` for the classifier. Held for the whole run so the traces are
            comparable across epochs.
        csv_path: Where to write the per-epoch trace.
        backbone_attr: Attribute on ``self.model`` holding the live backbone.
    """

    def __init__(
            self,
            probe_inputs: Any,
            csv_path: str,
            backbone_attr: str = "backbone",
    ) -> None:
        super().__init__()
        self.probe_inputs = probe_inputs
        self.csv_path = str(csv_path)
        self.backbone_attr = backbone_attr
        self._header_written = False

    def _build_probe(self, live_backbone: keras.Model) -> keras.Model:
        """Rebuild the backbone from its config with the energy readout enabled."""
        config = dict(live_backbone.get_config())
        config["return_energy"] = True
        probe = live_backbone.__class__.from_config(config)
        probe.build((None,) + tuple(live_backbone.input_shape_config))
        return probe

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        live_backbone = getattr(self.model, self.backbone_attr, None)
        if live_backbone is None:
            raise AttributeError(
                f"EnergyTraceCallback: model {type(self.model).__name__} has no "
                f"'{self.backbone_attr}' attribute; cannot build the energy probe."
            )

        probe = self._build_probe(live_backbone)
        # Re-synced EVERY epoch. Skipping this is the stale-probe failure mode.
        probe.set_weights(live_backbone.get_weights())

        _, energies = probe(self.probe_inputs, training=False)
        # Out of graph, immediately: numpy from here on. Nothing downstream ever sees a tensor.
        trace = np.asarray(keras.ops.convert_to_numpy(energies))
        # The trace's OWN dtype (float32 by the block's design, even under mixed_float16) --
        # logged before any cast, so the log cannot manufacture a dtype the tensor never had.
        trace_dtype = trace.dtype
        per_step = trace.astype(np.float64).mean(axis=0)  # (T+1,)

        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            if not self._header_written:
                writer.writerow(["epoch"] + [f"step_{i}" for i in range(per_step.shape[0])])
                self._header_written = True
            writer.writerow([epoch] + [f"{v:.6f}" for v in per_step.tolist()])

        finite = bool(np.all(np.isfinite(per_step)))
        max_rise = float(np.max(np.diff(per_step))) if per_step.shape[0] > 1 else 0.0
        logger.info(
            f"energy trace (epoch {epoch}, dtype={trace_dtype}): "
            f"E_0={per_step[0]:.4f} -> E_T={per_step[-1]:.4f} "
            f"(delta={per_step[-1] - per_step[0]:+.4f}, finite={finite}, max_rise={max_rise:+.3e})"
        )
        if not finite:
            logger.warning("energy trace contains non-finite values -- the descent has diverged")

# ---------------------------------------------------------------------
