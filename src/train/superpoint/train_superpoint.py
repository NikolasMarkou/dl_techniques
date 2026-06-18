"""SuperPoint stage-4 full joint trainer: detector + descriptor on homography pairs.

This is stage 4 of the SuperPoint training recipe (DeTone et al., CVPRW 2018):
the detector AND descriptor heads are trained jointly on homography image pairs.
For each base image a random homography ``H`` is sampled; the model sees both the
image and its warped copy, the detector head is supervised on the (image's)
65-class grid label, and the descriptor head is trained with the bespoke hinge
correspondence loss between the two coarse descriptor maps under the homography.

Descriptor-loss design (CHOICE = (a), the honest SuperPoint formulation)
-----------------------------------------------------------------------
The plan offered two options. We implement design (a): a minimal custom
:class:`keras.Model` subclass (:class:`SuperPointJointModel`) whose ``train_step``
runs the SuperPoint model on BOTH ``image`` and ``warped_image``, computes the
65-class detector loss on the ``keypoints`` head, and computes the descriptor
loss via :meth:`SuperPointDescriptorLoss.compute(desc1, desc2, correspondence)`
on the two H/8 descriptor maps with a TRUE homography-derived correspondence
(NOT the diagonal-identity convenience path of ``call(y_true, y_pred)``).

Why (a) and not (b): the diagonal-identity ``call()`` path assumes cell ``i`` in
the image corresponds to cell ``i`` in the warped image, which is only true for
an identity homography -- it trains the descriptor head against a wrong target.
Design (a) is the correct objective and fits cleanly here because the
correspondence is precomputed in the DATA pipeline (numpy, per-sample ``H`` is
known there) and passed in as an ``(N, N)`` tensor, so ``train_step`` stays
graph-clean (``keras.ops`` only; no raw ``tf.*``, no homography math in-graph).

Correspondence construction (data pipeline)
-------------------------------------------
The descriptor maps live on the H/8 grid (``N = Hc * Wc`` cells). We take each
cell-center pixel coordinate in the IMAGE frame, warp it forward through ``H``
with :func:`warp_points`, find which warped-image cell it lands in, and mark that
``(i, j)`` pair positive in the ``(N, N)`` correspondence matrix. Cells whose
center warps out of bounds get no positive (an all-zero row) and contribute only
negative-pair pressure -- the standard SuperPoint behaviour.

The descriptor head emits a FULL-resolution ``(B, H, W, 256)`` map (bicubically
upsampled inside the model); we subsample it at the H/8 cell centers to recover
the coarse ``(B, Hc, Wc, 256)`` map the loss expects.

Compilation uses ``jit_compile=False`` (DECISION D-004): the descriptor head's
bicubic upsample (``keras.ops.image.resize`` -> ``ResizeBicubic``) has no
XLA_GPU_JIT OpKernel in TF 2.18.

MagicPoint -> full weight handoff
---------------------------------
If ``magicpoint_checkpoint`` is given, after the model is built we call
:func:`load_weights_from_checkpoint` (name-based, partial -- MagicPoint shares
the encoder + projection + detector head; the descriptor head is new and stays at
its init). Under ``--smoke`` we additionally exercise this code path end-to-end
by saving the freshly-built model to a temp ``.keras`` and loading it back, to
prove the handoff runs without error even when no real MagicPoint checkpoint
exists.

Results are written to the repo-root ``results/`` directory.

Usage::

    MPLBACKEND=Agg python -m train.superpoint.train_superpoint \\
        --variant tiny --input-size 128 --batch-size 8 --epochs 50 \\
        --magicpoint-checkpoint results/magicpoint_.../final_model.keras --gpu 1

    # Fast smoke (seconds on GPU1):
    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg \\
        python -m train.superpoint.train_superpoint --smoke --gpu 1
"""

import time
import json
import keras
import argparse
import tempfile
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple

from train.common import (
    setup_gpu,
    set_seeds,
    create_callbacks as create_common_callbacks,
    save_config_json,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder,
)
from dl_techniques.models.superpoint import create_superpoint
from dl_techniques.losses.superpoint_loss import (
    SuperPointDetectorLoss,
    SuperPointDescriptorLoss,
)
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.utils.homography import sample_homography, warp_image, warp_points
from dl_techniques.datasets.synthetic_shapes import (
    generate_synthetic_sample,
    keypoints_to_grid_labels,
    DEFAULT_CELL,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass
class SuperPointConfig:
    """Configuration for the full joint SuperPoint stage-4 training."""

    # Data
    input_size: int = 128
    channels: int = 1
    cell: int = DEFAULT_CELL  # detector-head cell size (8 -> 65 classes)

    # Model
    variant: str = "tiny"

    # Training
    batch_size: int = 8
    epochs: int = 50
    steps_per_epoch: int = 500

    # Loss weights (paper: detector 1.0, descriptor lambda_d = 250).
    detector_weight: float = 1.0
    descriptor_weight: float = 250.0

    # MagicPoint -> full weight handoff (optional).
    magicpoint_checkpoint: Optional[str] = None

    # Optimization
    learning_rate: float = 1e-3
    optimizer_type: str = "adamw"
    lr_schedule_type: str = "cosine_decay"
    warmup_epochs: int = 5
    gradient_clipping: float = 1.0

    # Monitoring
    early_stopping_patience: int = 15

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results"
    experiment_name: Optional[str] = None

    # Smoke mode: tiny everything, runs in seconds on GPU1.
    smoke: bool = False

    def __post_init__(self):
        if self.smoke:
            self.input_size = 64
            self.batch_size = 2
            self.epochs = 1
            self.steps_per_epoch = 2
            self.variant = "tiny"
            self.early_stopping_patience = 1
            # In smoke we exercise the handoff path via a temp self-save, so a
            # real checkpoint is intentionally NOT required.
            self.magicpoint_checkpoint = None

        if self.input_size <= 0 or self.channels <= 0:
            raise ValueError("Invalid input size or channel configuration")
        if self.input_size % self.cell != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be divisible by cell "
                f"({self.cell}) so the detector grid matches the label map"
            )
        if self.input_size < 64:
            raise ValueError(
                f"input_size ({self.input_size}) must be >= 64 so H/8 >= 8 "
                f"(the 8x8 detector reshape stays meaningful)"
            )
        if self.variant not in ("tiny", "base", "large"):
            raise ValueError(f"Unknown variant: {self.variant}")
        if self.detector_weight < 0.0 or self.descriptor_weight < 0.0:
            raise ValueError("Loss weights must be non-negative")

        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"superpoint_{self.variant}_{timestamp}"


# ---------------------------------------------------------------------
# DATA PIPELINE (homography pairs + correspondence)
# ---------------------------------------------------------------------


def _cell_correspondence(
    homography: np.ndarray, H: int, W: int, cell: int
) -> np.ndarray:
    """Build the H/8-grid correspondence matrix for a homography pair.

    Each coarse cell center (image frame) is warped forward through ``homography``
    via :func:`warp_points`; if it lands inside the warped image its destination
    cell index ``j`` is marked positive against the source cell index ``i``.

    Args:
        homography: ``(3, 3)`` forward homography (image -> warped).
        H: Image height in pixels.
        W: Image width in pixels.
        cell: Detector/descriptor cell size (8).

    Returns:
        ``(N, N)`` ``float32`` correspondence matrix with ``N = (H//cell) *
        (W//cell)``; ``corr[i, j] = 1`` iff source cell ``i`` corresponds to
        warped cell ``j``.
    """
    Hc, Wc = H // cell, W // cell
    n = Hc * Wc

    # Cell-center pixel coords in the image frame, in row-major (i = cy*Wc + cx).
    cy, cx = np.meshgrid(np.arange(Hc), np.arange(Wc), indexing="ij")
    centers = np.stack(
        [(cx.reshape(-1) + 0.5) * cell, (cy.reshape(-1) + 0.5) * cell], axis=1
    ).astype(np.float32)  # (N, 2) as (x, y)

    warped = warp_points(centers, homography)  # (N, 2) as (x, y)

    corr = np.zeros((n, n), dtype=np.float32)
    wx = np.floor(warped[:, 0] / cell).astype(np.int64)
    wy = np.floor(warped[:, 1] / cell).astype(np.int64)
    in_bounds = (wx >= 0) & (wx < Wc) & (wy >= 0) & (wy < Hc)
    src = np.nonzero(in_bounds)[0]
    dst = wy[in_bounds] * Wc + wx[in_bounds]
    corr[src, dst] = 1.0
    return corr


def _pair_generator(
    config: SuperPointConfig,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(image, warped_image, grid_label, correspondence)`` tuples.

    Synthetic-shapes base image -> sample a homography -> warp the image and build
    the H/8 correspondence. The detector label is the IMAGE's 65-class grid label.

    Yields:
        ``image (H, W, 1) f32``, ``warped (H, W, 1) f32``,
        ``grid_label (Hc, Wc) i32``, ``correspondence (N, N) f32``.
    """
    H = W = config.input_size
    rng = np.random.default_rng(config.seed)
    logger.info(
        f"superpoint pair generator: H={H} W={W} cell={config.cell} "
        f"seed={config.seed}"
    )
    while True:
        img, kps = generate_synthetic_sample(H, W, rng)
        label = keypoints_to_grid_labels(kps, H, W, cell=config.cell)
        h_mat = sample_homography((H, W), rng=rng)
        warped = warp_image(img, h_mat).numpy()  # (H, W, 1)
        corr = _cell_correspondence(h_mat, H, W, config.cell)
        yield (
            img.astype(np.float32),
            warped.astype(np.float32),
            label.astype(np.int32),
            corr,
        )


def create_dataset(config: SuperPointConfig) -> tf.data.Dataset:
    """Build an infinite tf.data stream of homography-pair joint-training examples.

    Yields ``(image, {"keypoints": label, "warped_image": warped,
    "correspondence": corr})`` -- the custom ``train_step`` consumes the dict.

    Args:
        config: Training configuration.

    Returns:
        A batched, prefetched ``tf.data.Dataset``.
    """
    H = W = config.input_size
    Hc, Wc = H // config.cell, W // config.cell
    n = Hc * Wc

    output_signature = (
        tf.TensorSpec(shape=(H, W, config.channels), dtype=tf.float32),
        tf.TensorSpec(shape=(H, W, config.channels), dtype=tf.float32),
        tf.TensorSpec(shape=(Hc, Wc), dtype=tf.int32),
        tf.TensorSpec(shape=(n, n), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: _pair_generator(config),
        output_signature=output_signature,
    )

    dataset = dataset.map(
        lambda img, warped, label, corr: (
            img,
            {
                "keypoints": label,
                "warped_image": warped,
                "correspondence": corr,
            },
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# JOINT MODEL (custom train_step -- design (a))
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SuperPointJointModel(keras.Model):
    """Wraps a SuperPoint backbone with a joint detector+descriptor ``train_step``.

    The wrapped ``superpoint`` model is run on BOTH the image and its warped copy.
    The detector loss supervises the image's ``keypoints`` head; the descriptor
    loss is the hinge correspondence loss between the two H/8 descriptor maps under
    the data-supplied homography correspondence (design (a)).
    """

    def __init__(
        self,
        superpoint: keras.Model,
        cell: int = DEFAULT_CELL,
        detector_weight: float = 1.0,
        descriptor_weight: float = 250.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.superpoint = superpoint
        self.cell = cell
        self.detector_weight = detector_weight
        self.descriptor_weight = descriptor_weight

        self.detector_loss_fn = SuperPointDetectorLoss()
        self.descriptor_loss_fn = SuperPointDescriptorLoss()

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.det_tracker = keras.metrics.Mean(name="detector_loss")
        self.desc_tracker = keras.metrics.Mean(name="descriptor_loss")

    def call(self, inputs, training=None):
        """Delegate the plain forward pass to the wrapped SuperPoint model."""
        return self.superpoint(inputs, training=training)

    def _coarse_descriptors(self, desc_full):
        """Subsample the full-res ``(B,H,W,C)`` descriptor map to the H/8 grid.

        The descriptor head upsamples bicubically to full resolution inside the
        model; the correspondence loss operates on the coarse ``(B,Hc,Wc,C)`` map,
        so we sample at cell-center stride. ``keras.ops`` strided slice keeps this
        graph-safe (no raw ``tf.*``).
        """
        half = self.cell // 2
        return desc_full[:, half :: self.cell, half :: self.cell, :]

    @property
    def metrics(self):
        return [self.loss_tracker, self.det_tracker, self.desc_tracker]

    def train_step(self, data):
        x, y = data
        label = y["keypoints"]
        warped = y["warped_image"]
        corr = y["correspondence"]

        with tf.GradientTape() as tape:
            out1 = self.superpoint(x, training=True)
            out2 = self.superpoint(warped, training=True)

            det_loss = self.detector_loss_fn(label, out1["keypoints"])

            desc1 = self._coarse_descriptors(out1["descriptors"])
            desc2 = self._coarse_descriptors(out2["descriptors"])
            desc_loss = self.descriptor_loss_fn.compute(desc1, desc2, corr)

            total = (
                self.detector_weight * det_loss
                + self.descriptor_weight * desc_loss
            )

        grads = tape.gradient(total, self.superpoint.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.superpoint.trainable_variables)
        )

        self.loss_tracker.update_state(total)
        self.det_tracker.update_state(det_loss)
        self.desc_tracker.update_state(desc_loss)
        return {m.name: m.result() for m in self.metrics}


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------


def train_superpoint(config: SuperPointConfig) -> keras.Model:
    """Train SuperPoint jointly (detector + descriptor) on homography pairs."""
    logger.info(f"Starting SuperPoint joint training: {config.experiment_name}")

    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(config, str(output_dir), "config.json")

    # Dataset
    train_dataset = create_dataset(config)

    # Model: full SuperPoint backbone.
    input_shape = (config.input_size, config.input_size, config.channels)
    backbone = create_superpoint(config.variant, input_shape=input_shape)
    backbone.build((None, *input_shape))

    # MagicPoint -> full weight handoff (name-based, partial).
    if config.magicpoint_checkpoint:
        logger.info(
            f"MagicPoint handoff from {config.magicpoint_checkpoint}"
        )
        report = load_weights_from_checkpoint(
            target=backbone,
            ckpt_path=config.magicpoint_checkpoint,
            skip_prefixes=(),  # MagicPoint == same SuperPoint class; load all overlap
        )
        logger.info(f"Handoff report: {report.num_loaded} layers loaded")
    elif config.smoke:
        # Exercise the handoff CODE PATH without a real checkpoint: save the
        # freshly-built model to a temp .keras and load it straight back. This
        # proves load_weights_from_checkpoint runs end-to-end under --smoke.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_ckpt = str(Path(tmp) / "magicpoint_selftest.keras")
            backbone.save(tmp_ckpt)
            report = load_weights_from_checkpoint(
                target=backbone, ckpt_path=tmp_ckpt, skip_prefixes=()
            )
            logger.info(
                f"Smoke handoff self-test OK: {report.num_loaded} layers loaded"
            )

    model = SuperPointJointModel(
        superpoint=backbone,
        cell=config.cell,
        detector_weight=config.detector_weight,
        descriptor_weight=config.descriptor_weight,
    )

    # Optimizer with LR schedule (mirrors the MagicPoint trainer).
    lr_schedule = learning_rate_schedule_builder(
        {
            "type": config.lr_schedule_type,
            "learning_rate": config.learning_rate,
            "decay_steps": config.steps_per_epoch * config.epochs,
            "warmup_steps": config.steps_per_epoch * config.warmup_epochs,
            "alpha": 0.01,
        }
    )
    optimizer = optimizer_builder(
        {
            "type": config.optimizer_type,
            "gradient_clipping_by_norm": config.gradient_clipping,
        },
        lr_schedule,
    )

    # DECISION plan_2026-06-18_e1411ebf/D-004: jit_compile=False is REQUIRED.
    # Do NOT enable XLA here. The descriptor head's bicubic upsample
    # (keras.ops.image.resize -> ResizeBicubic) has no XLA_GPU_JIT OpKernel in
    # TF 2.18, so any XLA-compiled fit step raises a tf2xla conversion error.
    # Losses live inside train_step, so compile() takes no loss= argument.
    # See decisions.md D-004.
    model.compile(optimizer=optimizer, jit_compile=False)
    logger.info(
        f"Model compiled with {backbone.count_params():,} backbone parameters"
    )

    # Callbacks: monitor train loss (no validation split in the smoke stream).
    callbacks, _ = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="superpoint",
        run_dir=str(output_dir),
        monitor="loss",
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_terminate_on_nan=True,
        include_analyzer=False,
    )

    start_time = time.time()
    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    # Save the final SuperPoint backbone (the reusable artifact).
    try:
        model_path = output_dir / "final_model.keras"
        backbone.save(model_path)
        logger.info(f"Final model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    try:
        history_dict = {
            k: [float(v) for v in vals] for k, vals in history.history.items()
        }
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")

    return backbone


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SuperPoint jointly (detector + descriptor) on "
        "homography pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant", choices=["tiny", "base", "large"], default="tiny"
    )
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--detector-weight", type=float, default=1.0)
    parser.add_argument("--descriptor-weight", type=float, default=250.0)
    parser.add_argument(
        "--magicpoint-checkpoint",
        type=str,
        default=None,
        help="Path to a MagicPoint .keras checkpoint for encoder/detector "
        "weight handoff into the full SuperPoint model.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force tiny config (input 64, batch 2, 1 epoch, 2 steps) and "
        "exercise the MagicPoint handoff path via a temp self-save.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_gpu(gpu_id=args.gpu)
    set_seeds(args.seed)

    config = SuperPointConfig(
        input_size=args.input_size,
        variant=args.variant,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        detector_weight=args.detector_weight,
        descriptor_weight=args.descriptor_weight,
        magicpoint_checkpoint=args.magicpoint_checkpoint,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        smoke=args.smoke,
    )

    logger.info(
        f"Config: variant={config.variant}, input={config.input_size}, "
        f"batch={config.batch_size}, epochs={config.epochs}, "
        f"steps/epoch={config.steps_per_epoch}, "
        f"det_w={config.detector_weight}, desc_w={config.descriptor_weight}, "
        f"smoke={config.smoke}"
    )

    try:
        train_superpoint(config)
        logger.info("SuperPoint joint training completed successfully!")
    except Exception as e:
        logger.error(f"SuperPoint joint training failed: {e}")
        raise


if __name__ == "__main__":
    main()
