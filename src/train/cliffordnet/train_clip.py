"""Training script for CliffordCLIP (CLIP with Clifford geometric blocks).

Trains :class:`CliffordCLIP` on an image-caption dataset using a symmetric
contrastive cross-entropy objective. An **optional Stage 0** pretrains the
two towers independently before contrastive training:

- Vision tower on CIFAR-100 classification (recipe from
  ``train.cliffordnet.train_cliffordnet``).
- Text tower on Wikipedia causal LM (recipe from
  ``train.cliffordnet.train_cliffordnet_nlp``).

Stage 0 mutates the existing ``CliffordCLIP`` object in-place via thin
throwaway wrappers — no save/reload between stages. Defaults: 50,000 steps
per sub-stage. Disable with ``--skip-stage0`` or ``--stage0-{vision,lm}-steps 0``.

The training recipe borrows schedule ideas from the Penguin-VL paper
(arXiv:2603.06569):

- AdamW optimiser, cosine LR decay with a 3% warmup ratio
- Two-stage **low-to-high resolution curriculum**: Stage 1 at a smaller
  resolution (e.g., 96x96) followed by Stage 2 at the target resolution
  (e.g., 224x224). The model is the same object across stages; only the
  image preprocessing pipeline and the LR schedule are rebuilt.
- Learnable temperature clipped to ``logit_scale_max`` (OpenCLIP style).

**English-only**: the text tokenizer defaults to tiktoken's ``gpt2``
encoding (50257 BPE tokens, English-trained) rather than ``cl100k_base``
(which is multilingual). COCO captions are already English, so no extra
filtering is required for the default data source. Pass a different
``--tokenizer-encoding`` only if you intentionally want a non-English
tokenizer.

Note that Penguin-VL itself argues against contrastive pretraining as the
ideal initialisation for VLM vision encoders. This script implements
standard CLIP contrastive training because that is what the user asked
for; the distillation reconstruction losses described in the Penguin-VL
paper (amplitude/direction/relation) require a frozen teacher encoder and
are intentionally left out. Adding them is a matter of plugging a teacher
forward pass into the loss in :class:`ContrastiveCliffordCLIP.train_step`.

Dataset: uses the ``coco_captions`` tfds builder by default. Pass
``--synthetic`` for a CPU-friendly smoke test that generates random
pixel/caption pairs on the fly.

Reference training recipes
--------------------------

The defaults in this script are deliberately scaled *down* for a nano
model on a single 4090. For context, here are the two canonical CLIP
recipes. Any divergence from them is intentional (smaller model, smaller
data, single GPU) but worth being aware of when tuning.

**OpenAI CLIP (Radford et al., 2021 — arXiv:2103.00020).**
Trained on 400M web image-text pairs ("WIT") for 32 epochs.

- Architectures: ViT-B/32, ViT-B/16, ViT-L/14 (and ResNet variants).
  Vision patch sizes: 32x32 (B/32), 16x16 (B/16), 14x14 (L/14) on a
  224x224 input → 49, 196, and 256 tokens respectively (plus CLS).
  Text: 12-layer 512-wide 8-head Transformer, 49,408 BPE vocab, context
  length 76 (+ BOS/EOS = 77), max 63M params (tied to ViT-L).
- Optimiser: Adam with decoupled weight decay (`AdamW`-style), applied
  to all weights except biases, gains, and LayerNorm (which is what
  "decoupled weight decay" means here).
- LR: cosine decay, linear warmup over the first 2,000 steps; peak LR
  5e-4 for the ResNet/ViT-B models, 4e-4 for ViT-L.
- Weight decay: 0.2. beta_1=0.9, beta_2=0.98, eps=1e-6.
- Batch size: **32,768** (global, across 256/592 V100 GPUs) — the
  large-batch InfoNCE is central to CLIP's sample efficiency.
- Learnable temperature ``log(1/tau)`` init 0.07, clipped so
  ``exp(logit_scale) <= 100`` to prevent training instability.
- Mixed precision (fp16) with loss-scaling; gradient checkpointing on
  the largest variants; the vision tower runs in mixed precision and
  the text tower in fp32 for ViT-L.
- No weight decay on the temperature.
- Augmentation: only random resized crop from the original image.
- Data: 400M pairs collected from the public web (not released).

**OpenCLIP (Ilharco et al., 2021+; Cherti et al., 2023 —
arXiv:2212.07143).** Reproduces and scales CLIP on LAION-400M and
LAION-2B. Recipes differ slightly per variant but the pattern is:

- Architectures: same B/32, B/16, L/14, plus H/14 (14x14 patches on
  224x224 → 256 tokens) and G/14. Later high-res runs (e.g., L/14-336,
  H/14-378) keep the patch size and grow the input.
- Batch size: 32k–88k (multi-node distributed InfoNCE with
  ``gather-with-grad=True`` — gradients flow through the all-gather so
  negatives from other ranks count toward the loss).
- Peak LR: 5e-4 (B/32, B/16), 4e-4 (L/14), 1e-3 for very small models.
- Warmup: 2,000–10,000 steps; cosine decay to 0 over the full run.
- Weight decay: 0.1–0.2 (typically 0.1 for LAION runs, 0.2 for WIT-like).
- beta_2: 0.98 (kept from OpenAI); eps: 1e-6 or 1e-8.
- Training length: 32 epochs on LAION-400M = 12.8B seen samples
  (B/32), up to 34B samples for the H/14 runs.
- Learnable temperature clipped to ``exp(logit_scale) <= 100`` (same
  as OpenAI).
- Mixed precision (bf16 or fp16) via ``torch.cuda.amp``; activation
  checkpointing for the large models; LayerNorm kept in fp32.
- Random resized crop + horizontal flip; no color jitter in the
  canonical recipes (added in some later runs).
- Text: same 49,408 BPE vocab and context length 77 as OpenAI CLIP.

**This script's defaults vs. the above.** Nano variant (d=256, 12/12),
single 4090, batch 32-64, COCO-2017 (~591k pairs after the all-captions
fix), tiktoken `gpt2` vocab (50,257) with context length 64, cosine
schedule + 3% warmup, weight decay 0.1, `logit_scale_max=100`. Vision
patch size is configurable via ``--vision-patch-size`` — the example CLI
uses 4 on a 96/160-px input (→ 576/1600 tokens) because at those small
resolutions a CLIP-style 14-16 patch would leave too few tokens; full
224-px runs should use 14 or 16 to stay close to the CLIP regime. The
biggest qualitative gap is the batch size: 32 negatives vs. 32k means
contrastive signal is much weaker, which is why nano-on-COCO recall@K
numbers are roughly an order of magnitude below published CLIP results.

Example::

    python -m train.cliffordnet.train_clip \\
        --variant nano --stage1-image-size 96 --stage2-image-size 160 \\
        --stage1-epochs 5 --stage2-epochs 20 --batch-size 64 --gpu 0

    # smoke test
    python -m train.cliffordnet.train_clip --synthetic --stage1-epochs 1 \\
        --stage2-epochs 1 --batch-size 8 --max-train-samples 64
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Early GPU binding — MUST run before TensorFlow is imported.
#
# If ``--gpu N`` is on the command line, set ``CUDA_VISIBLE_DEVICES`` here so
# TensorFlow only sees the requested physical device when it initialises CUDA
# at import time. Setting the env var later (inside ``setup_gpu``) is too
# late: once ``import tensorflow`` has claimed the device list, changing
# ``CUDA_VISIBLE_DEVICES`` has no effect and TF silently allocates on every
# visible GPU — which breaks the single-GPU-job invariant on shared boxes.
#
# We keep the ``--gpu`` flag on the argparse parser as well so the rest of
# the script can still read ``args.gpu`` for logging and memory-growth setup.
# ---------------------------------------------------------------------------
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break
    if _a.startswith("--gpu="):
        os.environ["CUDA_VISIBLE_DEVICES"] = _a.split("=", 1)[1]
        break

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from dl_techniques.losses import CLIPContrastiveLoss
from dl_techniques.models.cliffordnet import CliffordCLIP
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger

from train.common import (
    build_synthetic_image_text_dataset,
    load_cc3m_local_split,
    load_coco2017_local_split,
    make_image_text_tf_dataset,
    setup_gpu,
    validate_model_loading,
)


# =============================================================================
# Config
# =============================================================================


@dataclass
class CliffordCLIPTrainConfig:
    """Hyperparameters for a single training stage."""

    image_size: int
    epochs: int
    batch_size: int
    peak_lr: float
    warmup_ratio: float = 0.03
    weight_decay: float = 0.1
    lr_alpha: float = 1e-2  # cosine decay final fraction of peak_lr
    min_warmup_lr: float = 1e-8


# Image preprocessing, caption tokenisation, split loaders, and the
# tf.data pipeline builder now live in ``train.common.image_text`` so they
# can be reused by any image-text training script. They are imported via
# the re-exports in ``train.common.__init__``.


# (Dataset handling has moved to ``train.common.image_text`` — see the
# import block above for ``build_synthetic_image_text_dataset``,
# ``load_coco2017_local_split``, ``load_cc3m_local_split``, and
# ``make_image_text_tf_dataset``.)


# =============================================================================
# Contrastive training wrapper
# =============================================================================


@keras.saving.register_keras_serializable()
class ContrastiveCliffordCLIP(keras.Model):
    """Thin wrapper that plugs a CLIP contrastive loss into ``fit()``.

    Delegates the forward pass to an inner :class:`CliffordCLIP`. Overrides
    ``train_step``/``test_step`` to apply the shared
    :class:`dl_techniques.losses.CLIPContrastiveLoss` to the similarity
    matrix returned by the inner model. This keeps the base model reusable
    for inference/encoders while letting the training script use the
    standard Keras training loop and the shared ``train.common`` callbacks.

    ``CLIPContrastiveLoss`` is instantiated with ``apply_temperature=False``
    because :class:`CliffordCLIP` already applies the learnable
    ``exp(logit_scale)`` temperature inside its forward pass — scaling
    again in the loss would compound the temperature.
    """

    def __init__(
        self,
        clip_model: Optional[CliffordCLIP] = None,
        label_smoothing: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.clip_model = clip_model
        self.label_smoothing = label_smoothing
        self.loss_fn = CLIPContrastiveLoss(
            apply_temperature=False,
            label_smoothing=label_smoothing,
            name="clip_contrastive_loss",
        )

    def call(
        self,
        inputs,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        return self.clip_model(inputs, training=training)

    def _contrastive_loss(
        self, outputs: Dict[str, keras.KerasTensor]
    ) -> keras.KerasTensor:
        # Call CLIPContrastiveLoss.call() directly and mean the per-sample
        # losses ourselves. This avoids having to synthesise a dummy
        # y_true tensor just to satisfy the base Loss __call__ signature.
        per_sample = self.loss_fn.call(
            None,
            {
                "logits_per_image": outputs["logits_per_image"],
                "logits_per_text": outputs["logits_per_text"],
            },
        )
        return keras.ops.mean(per_sample)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            outputs = self.clip_model(data, training=True)
            loss = self._contrastive_loss(outputs)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            "loss": loss,
            "logit_scale": outputs["logit_scale"],
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        outputs = self.clip_model(data, training=False)
        loss = self._contrastive_loss(outputs)
        return {
            "loss": loss,
            "logit_scale": outputs["logit_scale"],
        }

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["clip_model"] = keras.saving.serialize_keras_object(
            self.clip_model
        )
        config["label_smoothing"] = self.label_smoothing
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ContrastiveCliffordCLIP":
        # Explicit keys only. ``Model.get_config`` also stores standard
        # base-class fields (``name``, ``dtype``, ``trainable``); we accept
        # ``name`` because it round-trips cleanly, and silently drop the rest
        # rather than forwarding arbitrary ``**config`` -- that would mask
        # typos in saved configs and couple this loader to keras internals.
        inner = keras.saving.deserialize_keras_object(config.pop("clip_model"))
        label_smoothing = config.pop("label_smoothing", 0.0)
        name = config.get("name", None)
        return cls(clip_model=inner, label_smoothing=label_smoothing, name=name)


# =============================================================================
# Evaluation: retrieval recall@K
# =============================================================================
#
# Reference numbers — what "good" looks like for CLIP at comparable scale.
# Use these to calibrate CliffordCLIP runs; do not treat them as targets
# for the nano variant.
#
# MSCOCO 5K (Karpathy test split, 5K images / 25K captions):
#   OpenAI CLIP ViT-B/32 (WIT-400M):     i2t R@1 ~50-58, t2i R@1 ~30-38
#   OpenCLIP ViT-B/32 (LAION-2B):        i2t R@5 ~75,    t2i R@5 ~55
#   OpenCLIP H/14, G/14 (LAION-2B):      t2i R@5 ~73-75
#   SOTA VLMs (BLIP, X-VLM, fine-tuned): i2t R@1 ~87,    t2i R@1 ~75
#
# Flickr30K 1K:
#   ViT-B/32 zero-shot:                  i2t R@1 ~75,    t2i R@1 ~55
#   SOTA fine-tuned:                     i2t R@1 ~95+
#
# Zero-shot ImageNet top-1 (the classic CLIP headline):
#   OpenAI CLIP ViT-B/32:                ~63%
#   OpenCLIP ViT-B/32 LAION-2B:          ~66.6%
#   OpenCLIP ViT-G/14 LAION-2B:          ~80% (current frontier)
#
# How to interpret CliffordCLIP numbers
# -------------------------------------
# * R@1 is sensitive, R@5/R@10 smooth it out — always report all three.
#   Random baseline for MSCOCO 5K is 0.02% at R@1; anything above ~1%
#   means the model learned something.
# * Expect 2-5x worse than ViT-B/32 at this scale. ViT-B/32 was trained
#   on 400M pairs; nano on COCO-only (~600K pairs) cannot reach those
#   numbers. A useful internal target for nano/COCO is
#   i2t R@1 >= 10 and i2t R@5 >= 25. Below that, something is wrong.
# * Training loss should fall well below log(batch_size) within a few
#   epochs. log(128) = 4.85, log(512) = 6.24. A loss stuck near
#   log(batch_size) means encoders are not separating pairs.
# * logit_scale should climb from ~14 (init) toward ~30-50. If it
#   saturates at the logit_scale_max clip (default 100), LR is too high
#   or batch is too small. If it stays at 14, there is no gradient
#   signal — check the tokenizer and pad_token_id plumbing first.
# * Comparing losses across batch sizes is meaningless; only compare
#   retrieval metrics. Bigger batch = more in-batch negatives =
#   mechanically higher loss at equal model quality.
#
# Sources:
#   https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md
#   https://github.com/LAION-AI/CLIP_benchmark
#   https://laion.ai/blog/giant-openclip/
#   https://github.com/openai/CLIP/issues/115


def _compute_retrieval_metrics(
    clip_model: CliffordCLIP,
    dataset: tf.data.Dataset,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Compute image->text and text->image recall@K on a dataset.

    Streams the dataset through both encoders, stacks the embeddings, and
    computes the full similarity matrix on the collected features. For a
    tighter memory budget on very large eval sets, truncate ``dataset``.
    """
    image_feats: List[np.ndarray] = []
    text_feats: List[np.ndarray] = []
    for batch in dataset:
        img_f = clip_model.encode_image(batch["image"], training=False)
        txt_f = clip_model.encode_text(batch["text"], training=False)
        image_feats.append(keras.ops.convert_to_numpy(img_f))
        text_feats.append(keras.ops.convert_to_numpy(txt_f))

    image_mat = np.concatenate(image_feats, axis=0)
    text_mat = np.concatenate(text_feats, axis=0)
    sims = image_mat @ text_mat.T  # (N, N)

    n = sims.shape[0]
    labels = np.arange(n)

    def _recall_at_k(scores: np.ndarray, k: int) -> float:
        topk = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[
            :, :k
        ]
        hits = np.any(topk == labels[:, None], axis=1)
        return float(hits.mean())

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"i2t_r@{k}"] = _recall_at_k(sims, k)
        metrics[f"t2i_r@{k}"] = _recall_at_k(sims.T, k)
    metrics["num_pairs"] = float(n)
    return metrics


# =============================================================================
# Step-based callbacks (intermediate-results tracking)
# =============================================================================
#
# Layout is modelled on ``src/train/cliffordnet/train_cliffordnet_nlp.py`` so
# CLIP runs produce the same directory shape as the CliffordNet LM runs:
#
#   results/cliffordclip_<variant>_<timestamp>/
#     checkpoints/
#       step_0000500.keras
#       step_0001000.keras
#       ...
#       final.keras                            <- always-latest state dict
#     retrieval_probes/
#       probes.jsonl                           <- intermediate retrieval metrics
#     training_log.csv                         <- step-level training metrics
#     tensorboard/                             <- TF events for both stages
#     cliffordclip_<variant>.keras             <- final model (convenience)
#     training_summary.txt
#
# A single global step counter persists across stage 1 and stage 2 so the
# step-interval checkpoints and retrieval probes form one coherent timeline
# for the whole training run.


class StepCheckpointCallback(keras.callbacks.Callback):
    """Save checkpoints and step-level metrics at fixed step intervals.

    Mirrors the CliffordNet LM step-checkpoint pattern: maintain a rolling
    window of the ``max_checkpoints`` most recent ``step_NNNNNNN.keras``
    files plus a ``final.keras`` on ``on_train_end``. Also writes per-step
    metrics to ``training_log.csv`` every ``log_every_steps`` steps.

    The callback does not reset its global step counter between stages —
    pass the same instance to every ``fit()`` call so the checkpoint
    timeline is continuous.

    :param save_dir: Run directory. Creates ``checkpoints/`` and
        ``training_log.csv`` inside it.
    :param save_every_steps: Checkpoint interval in training steps.
    :param max_checkpoints: Rolling window size for step-NNN files.
    :param log_every_steps: Step-level CSV logging interval.
    :param initial_step: Starting step count (for resume).
    """

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 500,
        max_checkpoints: int = 3,
        log_every_steps: int = 50,
        initial_step: int = 0,
    ) -> None:
        super().__init__()
        self.save_every_steps = save_every_steps
        self.max_checkpoints = max_checkpoints
        self.log_every_steps = log_every_steps
        self._global_step = initial_step

        self._ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self._ckpt_dir, exist_ok=True)

        self._csv_path = os.path.join(save_dir, "training_log.csv")
        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[csv.DictWriter] = None

        logger.info(
            f"StepCheckpointCallback: save every {save_every_steps:,} steps, "
            f"log every {log_every_steps:,} steps, "
            f"keep max {max_checkpoints} checkpoints"
        )

    @property
    def global_step(self) -> int:
        return self._global_step

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._global_step += 1

        if self._global_step % self.log_every_steps == 0:
            self._log_metrics(logs)

        if self._global_step % self.save_every_steps == 0:
            self._save_checkpoint()

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        # Close the CSV handle and save a ``final.keras`` snapshot. Both
        # stages call this at their end, so ``final.keras`` always reflects
        # the most recent trained weights.
        if self._csv_file is not None:
            self._csv_file.flush()
        path = os.path.join(self._ckpt_dir, "final.keras")
        try:
            self.model.save(path)
            logger.info(f"Final checkpoint saved: {path}")
        except Exception as exc:
            logger.warning(f"Failed to write final.keras: {exc}")

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def _log_metrics(self, logs: Optional[Dict[str, Any]]) -> None:
        if logs is None:
            return
        row = {"step": self._global_step}
        for k, v in logs.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys())
            )
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()
        # Pad any new keys that appeared mid-run with None.
        for k in self._csv_writer.fieldnames:
            row.setdefault(k, None)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _save_checkpoint(self) -> None:
        path = os.path.join(
            self._ckpt_dir, f"step_{self._global_step:07d}.keras"
        )
        try:
            self.model.save(path)
        except Exception as exc:
            logger.warning(f"Checkpoint save failed at step {self._global_step}: {exc}")
            return
        # Release the transient NumPy copies Keras allocates during native
        # .keras serialization. Same motivation as the CliffordNet LM script.
        gc.collect()
        logger.info(
            f"Checkpoint saved: {path} (step {self._global_step:,})"
        )
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        ckpts = sorted(
            glob.glob(os.path.join(self._ckpt_dir, "step_*.keras"))
        )
        while len(ckpts) > self.max_checkpoints:
            old = ckpts.pop(0)
            try:
                os.remove(old)
                logger.info(f"Removed old checkpoint: {old}")
            except OSError as exc:
                logger.warning(f"Could not remove {old}: {exc}")


class RetrievalProbeCallback(keras.callbacks.Callback):
    """Run image<->text retrieval at fixed step intervals.

    This is the CLIP analogue of the ``GenerationProbeCallback`` in the
    CliffordNet LM trainer: at every ``probe_every_steps`` steps, compute
    recall@K on a small held-out slice of val pairs and append the result
    to ``retrieval_probes/probes.jsonl``. The probe dataset is built per
    stage (via :meth:`set_probe_dataset`) so it always uses the current
    training resolution.

    The per-probe JSONL record includes the global step, a ``stage`` tag,
    the current ``logit_scale`` exp-value, and every recall metric — which
    is the timeline readers want for a "how did this training go" view.

    :param clip_model: The underlying :class:`CliffordCLIP` (not the
        training wrapper).
    :param save_dir: Run directory. ``retrieval_probes/probes.jsonl`` is
        created under it.
    :param probe_every_steps: Run a probe every N training steps.
    :param initial_step: Starting step count (for resume).
    """

    def __init__(
        self,
        clip_model: CliffordCLIP,
        save_dir: str,
        probe_every_steps: int = 500,
        initial_step: int = 0,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.probe_every_steps = probe_every_steps
        self._global_step = initial_step
        self._stage_label: str = "unknown"
        self._probe_ds: Optional[tf.data.Dataset] = None

        probe_dir = os.path.join(save_dir, "retrieval_probes")
        os.makedirs(probe_dir, exist_ok=True)
        self._jsonl_path = os.path.join(probe_dir, "probes.jsonl")
        logger.info(
            f"RetrievalProbeCallback: probe every {probe_every_steps:,} "
            f"steps; writing {self._jsonl_path}"
        )

    @property
    def global_step(self) -> int:
        return self._global_step

    def set_probe_dataset(
        self, probe_ds: tf.data.Dataset, stage_label: str
    ) -> None:
        """Install the probe dataset for the current stage.

        Called once per stage from :func:`_run_stage` so the callback
        always evaluates at the active training resolution.
        """
        self._probe_ds = probe_ds
        self._stage_label = stage_label

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._global_step += 1
        if self._probe_ds is None:
            return
        if self._global_step % self.probe_every_steps != 0:
            return
        self._run_probe(logs)

    def _run_probe(self, logs: Optional[Dict[str, Any]]) -> None:
        try:
            metrics = _compute_retrieval_metrics(self.clip_model, self._probe_ds)
        except Exception as exc:
            logger.warning(f"Retrieval probe failed at step {self._global_step}: {exc}")
            return

        # Scrape logit_scale directly from the model so we record its
        # instantaneous value rather than an averaged training metric. Use
        # ``_get_logit_scale`` so the recorded number is the clipped exp
        # that the forward pass actually multiplied into the logits -- not
        # the raw exp which can exceed logit_scale_max mid-training.
        try:
            scale_val = float(
                keras.ops.convert_to_numpy(self.clip_model._get_logit_scale())
            )
        except Exception:
            scale_val = float("nan")

        record = {
            "step": self._global_step,
            "stage": self._stage_label,
            "logit_scale": scale_val,
        }
        if logs is not None:
            for k in ("loss", "val_loss"):
                if k in logs:
                    try:
                        record[k] = float(logs[k])
                    except (TypeError, ValueError):
                        pass
        record.update({k: v for k, v in metrics.items()})

        with open(self._jsonl_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info(
            f"Probe [step {self._global_step:,}] stage={self._stage_label} "
            f"i2t_r@1={metrics.get('i2t_r@1', 0):.4f} "
            f"i2t_r@5={metrics.get('i2t_r@5', 0):.4f} "
            f"t2i_r@1={metrics.get('t2i_r@1', 0):.4f} "
            f"logit_scale={scale_val:.3f}"
        )


# =============================================================================
# Stage 0: independent pretraining (vision on CIFAR-100, text on Wikipedia)
# =============================================================================
#
# The two Stage 0 helpers train the *existing* ``CliffordCLIP`` sub-layers
# in place via a throwaway wrapper that attaches a task-specific head
# (classifier for vision, LM head for text). Weights land directly in
# ``clip_model`` — no save/reload, no cross-model weight transfer.
#
# Why not train standalone ``CliffordNet`` / ``CliffordNetLM`` and transfer?
# Layer names don't match (``clifford_block_N`` vs ``vision_clifford_block_N``
# / ``text_clifford_block_N``), so ``load_weights_from_checkpoint`` would
# need a per-layer name map. Wrapping is simpler and keeps the whole thing
# in one process.


def _iter_clip_sublayers(clip_model: CliffordCLIP):
    """Yield every ``keras.layers.Layer`` tracked by CliffordCLIP we care about."""
    _stem_attrs = (
        "vision_stem_conv", "vision_stem_conv1", "vision_stem_bn1",
        "vision_stem_conv2", "vision_stem_norm",
    )
    _vision_head_attrs = (
        "vision_global_pool", "vision_global_max_pool",
        "vision_head_norm", "vision_head_dropout",
    )
    _text_embed_attrs = (
        "token_embedding", "position_embedding",
        "text_embed_norm", "text_embed_dropout",
    )
    _text_head_attrs = ("text_head_norm", "text_head_dropout")
    _projection_attrs = (
        "vision_head_geo", "text_head_geo",
        "vision_query_pool", "text_query_pool",
        "vision_head_scale", "text_head_scale",
        "vision_projection", "text_projection",
    )
    for name in _stem_attrs:
        layer = getattr(clip_model, name, None)
        if layer is not None:
            yield layer
    for block in clip_model.vision_blocks:
        yield block
    for name in _vision_head_attrs:
        layer = getattr(clip_model, name, None)
        if layer is not None:
            yield layer
    for name in _text_embed_attrs:
        layer = getattr(clip_model, name, None)
        if layer is not None:
            yield layer
    for block in clip_model.text_blocks:
        yield block
    for name in _text_head_attrs:
        layer = getattr(clip_model, name, None)
        if layer is not None:
            yield layer
    for name in _projection_attrs:
        layer = getattr(clip_model, name, None)
        if layer is not None:
            yield layer


# Projection / Clifford-head / cross-modal layers — always frozen during
# Stage 0 (they'd learn against a nonexistent contrastive objective).
_STAGE0_ALWAYS_FROZEN = frozenset({
    "vision_projection", "text_projection",
    "vision_head_geo", "text_head_geo",
    "vision_query_pool", "text_query_pool",
    "vision_head_scale", "text_head_scale",
    "vision_global_max_pool",
})


def _is_vision_backbone(name: str) -> bool:
    return (
        name.startswith("vision_stem")
        or name.startswith("vision_clifford_block_")
        or name in ("vision_global_pool", "vision_head_norm", "vision_head_dropout")
    )


def _is_text_backbone(name: str) -> bool:
    return (
        name in (
            "token_embedding", "position_embedding",
            "text_embed_norm", "text_embed_dropout",
            "text_head_norm", "text_head_dropout",
        )
        or name.startswith("text_clifford_block_")
    )


def _freeze_clip_for_stage0(
    clip_model: CliffordCLIP, *, train_tower: str,
) -> None:
    """Freeze all sub-layers except the requested tower's backbone.

    The learnable ``logit_scale`` scalar is also frozen — it has no role
    before contrastive training and must not drift under weight decay.
    """
    assert train_tower in ("vision", "text"), train_tower
    for layer in _iter_clip_sublayers(clip_model):
        name = layer.name
        if name in _STAGE0_ALWAYS_FROZEN:
            layer.trainable = False
        elif train_tower == "vision":
            layer.trainable = _is_vision_backbone(name)
        else:
            layer.trainable = _is_text_backbone(name)
    if clip_model.logit_scale is not None:
        clip_model.logit_scale.trainable = False


def _unfreeze_clip(clip_model: CliffordCLIP) -> None:
    """Undo Stage 0 freezes so Stage 1 contrastive training trains everything."""
    for layer in _iter_clip_sublayers(clip_model):
        layer.trainable = True
    if clip_model.logit_scale is not None:
        clip_model.logit_scale.trainable = True


class _VisionClassifier(keras.Model):
    """Stage 0 wrapper that reuses CliffordCLIP vision sub-layers.

    Forward path: ``_apply_vision_stem`` → ``vision_blocks`` →
    ``vision_global_pool`` → ``vision_head_norm`` → (optional
    ``vision_head_dropout``) → fresh ``Dense(num_classes)``. The projection
    / Clifford-head sub-layers are intentionally skipped — they have no
    meaning outside contrastive training and should be frozen anyway.
    """

    def __init__(
        self, clip_model: CliffordCLIP, num_classes: int, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.clip_model = clip_model
        self.classifier = keras.layers.Dense(
            num_classes, name="stage0_vision_classifier",
        )

    def call(self, images, training: Optional[bool] = None):
        m = self.clip_model
        x = m._apply_vision_stem(images, training=training)
        for block in m.vision_blocks:
            x = block(x, training=training)
        x = m.vision_global_pool(x)
        x = m.vision_head_norm(x)
        if m.vision_head_dropout is not None:
            x = m.vision_head_dropout(x, training=training)
        return self.classifier(x)


def _run_stage0_vision(
    clip_model: CliffordCLIP,
    args: argparse.Namespace,
    run_dir: str,
) -> None:
    """Pretrain the CliffordCLIP vision tower on CIFAR-100 classification.

    Reuses the augmentation + normalisation pipeline from
    ``train.cliffordnet.train_cliffordnet`` (AutoAugment CIFAR-10 policy,
    random crop, HFlip, per-channel normalise, RandomErasing).
    """
    from train.cliffordnet.train_cliffordnet import (
        build_train_dataset as _cifar_build_train,
        build_eval_dataset as _cifar_build_eval,
    )
    from train.common import load_dataset

    steps_budget = int(args.stage0_vision_steps)
    logger.info("=" * 60)
    logger.info(f"Stage 0 vision: CIFAR-100 pretrain ({steps_budget:,} steps)")
    logger.info("=" * 60)

    (x_train, y_train), (x_test, y_test), _shape, num_classes = load_dataset(
        "cifar100", batch_size=args.stage0_vision_batch_size,
    )
    if num_classes != 100:
        raise RuntimeError(
            f"Stage 0 vision expected CIFAR-100 (100 classes); "
            f"got num_classes={num_classes}"
        )

    train_ds = _cifar_build_train(
        x_train, y_train,
        batch_size=args.stage0_vision_batch_size,
        dataset_name="cifar100",
        random_erasing_prob=0.25,
    )
    val_ds = _cifar_build_eval(
        x_test, y_test,
        batch_size=args.stage0_vision_batch_size,
        dataset_name="cifar100",
    )
    steps_per_epoch = max(1, len(x_train) // args.stage0_vision_batch_size)
    stage0_epochs = max(1, -(-steps_budget // steps_per_epoch))  # ceil
    warmup_steps = max(1, int(0.05 * steps_budget))
    decay_steps = max(1, steps_budget - warmup_steps)

    schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.stage0_vision_lr,
        "decay_steps": decay_steps,
        "alpha": 1e-2,
        "warmup_steps": warmup_steps,
        "warmup_start_lr": 1e-8,
    })
    optimizer = optimizer_builder({
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": args.stage0_vision_wd,
    }, schedule)

    _freeze_clip_for_stage0(clip_model, train_tower="vision")

    wrapper = _VisionClassifier(
        clip_model, num_classes=num_classes, name="stage0_vision_wrapper",
    )
    wrapper.build((None, 32, 32, 3))

    n_train = sum(int(np.prod(v.shape)) for v in wrapper.trainable_variables)
    n_total = sum(int(np.prod(v.shape)) for v in wrapper.variables)
    logger.info(
        f"Stage 0 vision: trainable={n_train:,} / total={n_total:,} params "
        f"| {stage0_epochs} epochs × {steps_per_epoch} steps/epoch "
        f"(budget {steps_budget:,}, warmup {warmup_steps:,}, "
        f"peak_lr={args.stage0_vision_lr}, wd={args.stage0_vision_wd})"
    )

    wrapper.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    tb_dir = os.path.join(run_dir, "tensorboard", "stage0_vision")
    os.makedirs(tb_dir, exist_ok=True)
    wrapper.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage0_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.TensorBoard(
                log_dir=tb_dir, write_graph=False, update_freq="epoch",
            ),
        ],
        verbose=1,
    )

    _unfreeze_clip(clip_model)
    logger.info("Stage 0 vision: complete (CLIP layers unfrozen for Stage 1).")


class _TextLMWrapper(keras.Model):
    """Stage 0 wrapper that reuses CliffordCLIP text sub-layers for CLM.

    Mirrors :meth:`CliffordCLIP.encode_text` up to ``text_head_dropout``,
    then applies a fresh ``Dense(vocab_size)`` for per-token logits.
    Returns a dict ``{"logits": (B, L, V)}`` so it composes with
    ``MaskedCausalLMLoss`` via ``loss={"logits": ...}`` / labels wrapped
    as ``(x, {"logits": y})``.
    """

    def __init__(
        self,
        clip_model: CliffordCLIP,
        vocab_size: int,
        context_length: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.clip_model = clip_model
        self.context_length = context_length
        self.lm_head = keras.layers.Dense(vocab_size, name="stage0_lm_head")

    def call(self, input_ids, training: Optional[bool] = None):
        from keras import ops
        m = self.clip_model
        positions = ops.arange(self.context_length)
        x = m.token_embedding(input_ids) + m.position_embedding(positions)
        x = m.text_embed_norm(x)
        x = m.text_embed_dropout(x, training=training)
        x = ops.expand_dims(x, axis=1)
        for block in m.text_blocks:
            x = block(x, training=training)
        x = ops.squeeze(x, axis=1)
        x = m.text_head_norm(x)
        if m.text_head_dropout is not None:
            x = m.text_head_dropout(x, training=training)
        return {"logits": self.lm_head(x)}


def _run_stage0_lm(
    clip_model: CliffordCLIP,
    args: argparse.Namespace,
    run_dir: str,
) -> None:
    """Pretrain the CliffordCLIP text tower on Wikipedia causal LM.

    Uses the same context length as CliffordCLIP (``args.context_length``)
    so the in-place position_embedding remains shape-compatible with the
    subsequent contrastive stages. Tokenizer is plain tiktoken ``gpt2``
    (vocab 50257, EOT 50256); the 4 extra specials that
    ``train_cliffordnet_nlp.py`` declares for MLM are not emitted in the
    CLM path, so we can pretrain without them.
    """
    from train.common.nlp import preprocess_clm_packed_dataset
    from dl_techniques.datasets.nlp import load_wikipedia_train_val
    from dl_techniques.losses import MaskedCausalLMLoss
    import tiktoken

    steps_budget = int(args.stage0_lm_steps)
    logger.info("=" * 60)
    logger.info(f"Stage 0 LM: Wikipedia CLM pretrain ({steps_budget:,} steps)")
    logger.info("=" * 60)

    # Tokenizer — plain gpt2, no specials; mirrors what train_clip.py uses.
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab
    eot_token_id = int(encoder.eot_token)
    logger.info(
        f"Stage 0 LM tokenizer: gpt2, vocab_size={vocab_size}, "
        f"eot_token_id={eot_token_id}"
    )
    if vocab_size != clip_model.vocab_size:
        raise RuntimeError(
            f"Vocab mismatch: tokenizer n_vocab={vocab_size}, "
            f"clip_model.vocab_size={clip_model.vocab_size}. "
            "Rebuild CliffordCLIP with vocab_size=50257 or change tokenizer."
        )

    # Wikipedia data — fail fast if cache dir is absent.
    if not os.path.isdir(args.stage0_lm_hf_cache):
        raise RuntimeError(
            f"Wikipedia cache dir not found: {args.stage0_lm_hf_cache}. "
            "Point --stage0-lm-hf-cache at a pre-downloaded HF cache or "
            "pass --stage0-lm-steps 0 / --skip-stage0 to bypass LM pretrain."
        )
    train_raw, _val_raw = load_wikipedia_train_val(
        cache_dir=args.stage0_lm_hf_cache,
        min_article_length=500,
        val_fraction=0.02,
        max_val_samples=5000,
        max_train_samples=None,
    )

    chunk_length = args.context_length + 1  # +1 for the causal shift.
    train_ds = preprocess_clm_packed_dataset(
        train_raw,
        encoding_name="gpt2",
        chunk_length=chunk_length,
        batch_size=args.stage0_lm_batch_size,
        eot_token_id=eot_token_id,
        repeat=True,  # guarantee steps_per_epoch never StopIteration.
    )
    train_ds = train_ds.map(
        lambda x, y: (x, {"logits": y}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Schedule: warmup + cosine over the step budget.
    warmup_steps = max(1, int(0.05 * steps_budget))
    decay_steps = max(1, steps_budget - warmup_steps)
    schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.stage0_lm_lr,
        "decay_steps": decay_steps,
        "alpha": 1e-2,
        "warmup_steps": warmup_steps,
        "warmup_start_lr": 1e-8,
    })
    optimizer = optimizer_builder({
        "type": "adamw",
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": args.stage0_lm_wd,
    }, schedule)

    _freeze_clip_for_stage0(clip_model, train_tower="text")

    wrapper = _TextLMWrapper(
        clip_model,
        vocab_size=vocab_size,
        context_length=args.context_length,
        name="stage0_lm_wrapper",
    )
    wrapper.build((None, args.context_length))

    n_train = sum(int(np.prod(v.shape)) for v in wrapper.trainable_variables)
    n_total = sum(int(np.prod(v.shape)) for v in wrapper.variables)
    logger.info(
        f"Stage 0 LM: trainable={n_train:,} / total={n_total:,} params "
        f"| 1 epoch × {steps_budget} steps/epoch "
        f"(warmup {warmup_steps:,}, peak_lr={args.stage0_lm_lr}, "
        f"wd={args.stage0_lm_wd}, context_length={args.context_length})"
    )

    wrapper.compile(
        optimizer=optimizer,
        loss={"logits": MaskedCausalLMLoss()},
        metrics={"logits": ["accuracy"]},
    )

    tb_dir = os.path.join(run_dir, "tensorboard", "stage0_lm")
    os.makedirs(tb_dir, exist_ok=True)
    wrapper.fit(
        train_ds,
        epochs=1,
        steps_per_epoch=steps_budget,
        callbacks=[
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.TensorBoard(
                log_dir=tb_dir, write_graph=False, update_freq="epoch",
            ),
        ],
        verbose=1,
    )

    _unfreeze_clip(clip_model)
    logger.info("Stage 0 LM: complete (CLIP layers unfrozen for Stage 1).")


# =============================================================================
# Stage training
# =============================================================================


def _build_stage_optimizer(
    stage_cfg: CliffordCLIPTrainConfig,
    steps_per_epoch: int,
) -> keras.optimizers.Optimizer:
    """Cosine LR with warmup_ratio warmup + AdamW — Penguin-VL recipe."""
    total_steps = max(1, stage_cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(round(stage_cfg.warmup_ratio * total_steps)))
    decay_steps = max(1, total_steps - warmup_steps)

    schedule = learning_rate_schedule_builder(
        {
            "type": "cosine_decay",
            "learning_rate": stage_cfg.peak_lr,
            "decay_steps": decay_steps,
            "alpha": stage_cfg.lr_alpha,
            "warmup_steps": warmup_steps,
            "warmup_start_lr": stage_cfg.min_warmup_lr,
        }
    )
    optimizer = optimizer_builder(
        {
            "type": "adamw",
            "beta_1": 0.9,
            "beta_2": 0.98,
            "epsilon": 1e-6,
            "weight_decay": stage_cfg.weight_decay,
        },
        schedule,
    )

    # Exclude the CLIP learnable temperature from weight decay. AdamW
    # otherwise pulls ``logit_scale`` toward 0 every step, which *lowers*
    # the effective temperature and collapses the contrastive signal. The
    # original CLIP training recipe also excludes biases and LayerNorm
    # gains; we keep the exclusion minimal (just the temperature) because
    # that is the one we have direct evidence is being harmed by decay.
    # ``exclude_from_weight_decay`` accepts substring patterns matched
    # against ``variable.path``. It must be called before the first
    # apply_gradients (i.e. before the optimizer is built) — we do it here
    # while the optimizer is still fresh.
    if hasattr(optimizer, "exclude_from_weight_decay"):
        try:
            optimizer.exclude_from_weight_decay(var_names=["logit_scale"])
            logger.info(
                "Excluded 'logit_scale' from AdamW weight decay."
            )
        except (ValueError, AttributeError) as exc:
            logger.warning(
                f"Could not exclude logit_scale from weight decay: {exc}"
            )

    logger.info(
        f"Stage optimizer: cosine_decay peak_lr={stage_cfg.peak_lr}, "
        f"warmup_steps={warmup_steps}/{total_steps} "
        f"(ratio={stage_cfg.warmup_ratio}), wd={stage_cfg.weight_decay}"
    )
    return optimizer


def _run_stage(
    stage_name: str,
    wrapper: ContrastiveCliffordCLIP,
    stage_cfg: CliffordCLIPTrainConfig,
    train_ds: tf.data.Dataset,
    val_ds: Optional[tf.data.Dataset],
    probe_ds: Optional[tf.data.Dataset],
    steps_per_epoch: int,
    run_dir: str,
    persistent_callbacks: List[keras.callbacks.Callback],
    retrieval_probe_cb: Optional[RetrievalProbeCallback],
) -> keras.callbacks.History:
    """Run a single curriculum stage via ``model.fit``.

    The caller owns the persistent step-based callbacks
    (:class:`StepCheckpointCallback`, :class:`RetrievalProbeCallback`) so
    their state (global step, open CSV writer, JSONL file) crosses the
    stage boundary. Each stage contributes only a stage-local
    ``TerminateOnNaN`` + ``TensorBoard`` pair on top of that shared list.
    """
    logger.info(
        f"=== {stage_name}: image_size={stage_cfg.image_size}, "
        f"epochs={stage_cfg.epochs}, batch={stage_cfg.batch_size} ==="
    )

    optimizer = _build_stage_optimizer(stage_cfg, steps_per_epoch)
    wrapper.compile(optimizer=optimizer)

    tb_dir = os.path.join(run_dir, "tensorboard", stage_name)
    os.makedirs(tb_dir, exist_ok=True)
    stage_local: List[keras.callbacks.Callback] = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.TensorBoard(
            log_dir=tb_dir,
            write_graph=False,
            update_freq="epoch",
        ),
    ]

    if retrieval_probe_cb is not None and probe_ds is not None:
        retrieval_probe_cb.set_probe_dataset(probe_ds, stage_label=stage_name)

    callbacks = stage_local + list(persistent_callbacks)
    logger.info(
        f"Stage {stage_name}: tensorboard={tb_dir}; "
        f"step-checkpoints + retrieval probes inherited from run-level state"
    )

    history = wrapper.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage_cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# =============================================================================
# Main
# =============================================================================


def _resolve_image_size(args: argparse.Namespace, stage: int) -> int:
    return args.stage1_image_size if stage == 1 else args.stage2_image_size


def _resolve_epochs(args: argparse.Namespace, stage: int) -> int:
    return args.stage1_epochs if stage == 1 else args.stage2_epochs


def _resolve_lr(args: argparse.Namespace, stage: int) -> float:
    return args.stage1_lr if stage == 1 else args.stage2_lr


def _prepare_datasets(
    args: argparse.Namespace,
    encoder,
) -> Tuple[Any, Any, Any, Any]:
    """Load raw image+token arrays/lists for training and evaluation.

    Returns ``(train_images, train_tokens, val_images, val_tokens)``. For
    synthetic mode, both splits are numpy arrays already normalized and
    resized. For COCO mode, both splits are python lists of raw uint8
    images plus numpy int32 token matrices.
    """
    vocab_size = encoder.n_vocab
    eot_token_id = int(encoder.eot_token)
    if args.synthetic:
        train_images, train_tokens = build_synthetic_image_text_dataset(
            num_samples=args.max_train_samples or 256,
            image_size=args.stage2_image_size,
            context_length=args.context_length,
            vocab_size=vocab_size,
            eot_token_id=eot_token_id,
            seed=0,
        )
        val_images, val_tokens = build_synthetic_image_text_dataset(
            num_samples=args.max_val_samples or 64,
            image_size=args.stage2_image_size,
            context_length=args.context_length,
            vocab_size=vocab_size,
            eot_token_id=eot_token_id,
            seed=1,
        )
        return train_images, train_tokens, val_images, val_tokens

    if args.dataset == "coco2017":
        train_images, train_tokens = load_coco2017_local_split(
            split="train",
            coco_root=args.coco_root,
            max_samples=args.max_train_samples,
            encoder=encoder,
            context_length=args.context_length,
        )
        val_images, val_tokens = load_coco2017_local_split(
            split="val",
            coco_root=args.coco_root,
            max_samples=args.max_val_samples,
            encoder=encoder,
            context_length=args.context_length,
        )
    elif args.dataset == "cc3m":
        train_images, train_tokens = load_cc3m_local_split(
            split="train",
            cc3m_root=args.cc3m_root,
            max_samples=args.max_train_samples,
            encoder=encoder,
            context_length=args.context_length,
        )
        val_images, val_tokens = load_cc3m_local_split(
            split="validation",
            cc3m_root=args.cc3m_root,
            max_samples=args.max_val_samples,
            encoder=encoder,
            context_length=args.context_length,
        )
    else:
        raise ValueError(f"Unknown --dataset {args.dataset!r}")
    return train_images, train_tokens, val_images, val_tokens


def train(args: argparse.Namespace) -> None:
    logger.info("Starting CliffordCLIP training")
    setup_gpu(args.gpu)

    # --- Tokenizer ---
    # Default: tiktoken gpt2 encoding (English-trained BPE, 50257 tokens).
    # cl100k_base is intentionally avoided because it is multilingual and
    # this CLIP is English-only. We use the raw tiktoken encoder rather
    # than the shared TiktokenPreprocessor because CLIP does not use the
    # CLS/SEP/PAD/MASK special tokens that the preprocessor requires — we
    # pad with the encoder's native eot_token instead.
    import tiktoken
    encoder = tiktoken.get_encoding(args.tokenizer_encoding)
    vocab_size = encoder.n_vocab
    eot_token_id = int(encoder.eot_token)
    logger.info(
        f"Tokenizer: {args.tokenizer_encoding}, vocab_size={vocab_size}, "
        f"eot_token_id={eot_token_id}"
    )

    # --- Data ---
    train_images, train_tokens, val_images, val_tokens = _prepare_datasets(
        args, encoder
    )

    # --- Model ---
    # Pad sentinel must match the token id used to right-pad the sequences
    # (see train.common.image_text.tokenize_captions /
    # build_synthetic_image_text_dataset).
    clip_model = CliffordCLIP.from_variant(
        variant=args.variant,
        vocab_size=vocab_size,
        image_size=args.stage2_image_size,  # final target resolution
        context_length=args.context_length,
        vision_patch_size=args.vision_patch_size,
        dropout_rate=args.dropout_rate,
        pad_token_id=eot_token_id,
        head_kind=args.head_kind,
        head_cli_mode=args.head_cli_mode,
    )
    # Build at the final resolution so shape checks pass at stage 2 even
    # when we pass stage 1 data through first. Clifford blocks are
    # resolution-agnostic because they are fully convolutional.
    clip_model.build(
        {
            "image": (
                None,
                args.stage2_image_size,
                args.stage2_image_size,
                3,
            ),
            "text": (None, args.context_length),
        }
    )
    clip_model.summary(print_fn=logger.info)

    wrapper = ContrastiveCliffordCLIP(
        clip_model=clip_model,
        label_smoothing=args.label_smoothing,
    )
    wrapper.build(
        {
            "image": (
                None,
                args.stage2_image_size,
                args.stage2_image_size,
                3,
            ),
            "text": (None, args.context_length),
        }
    )

    # --- Results dir ---
    # Single timestamped run dir per launch. Layout mirrors
    # ``train_cliffordnet_nlp.py``:
    #   results/cliffordclip_<variant>_<timestamp>/
    #     checkpoints/step_NNNNNNN.keras + final.keras
    #     retrieval_probes/probes.jsonl  <- intermediate results
    #     tensorboard/stageX_sizeY/
    #     training_log.csv               <- step-level metrics
    #     cliffordclip_<variant>.keras   <- final model (convenience)
    #     training_summary.txt
    from datetime import datetime
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        "results",
        f"cliffordclip_{args.variant}_{run_ts}",
    )
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Run directory: {results_dir}")

    # --- Persistent run-level callbacks (shared across stages) ---
    # Both callbacks update their own global step counters on every
    # ``on_train_batch_end`` — reusing the same instances across both
    # ``fit()`` calls is what gives us a continuous step timeline and
    # one coherent ``checkpoints/`` + ``retrieval_probes/probes.jsonl``.
    step_ckpt_cb = StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=args.save_every_steps,
        max_checkpoints=args.max_checkpoints,
        log_every_steps=args.log_every_steps,
    )
    retrieval_probe_cb = RetrievalProbeCallback(
        clip_model=clip_model,
        save_dir=results_dir,
        probe_every_steps=args.probe_every_steps,
    )
    persistent_callbacks: List[keras.callbacks.Callback] = [
        step_ckpt_cb,
        retrieval_probe_cb,
    ]

    # --- Stage 0: optional independent pretraining ---
    # Resolve skip flag up-front so later checks read the effective values.
    if args.skip_stage0:
        args.stage0_vision_steps = 0
        args.stage0_lm_steps = 0
    if args.synthetic and args.stage0_lm_steps > 0:
        logger.warning(
            "--synthetic + stage0_lm_steps > 0: Wikipedia is not synthetic; "
            "forcing stage0_lm_steps=0 for this run."
        )
        args.stage0_lm_steps = 0

    if args.stage0_vision_steps > 0:
        _run_stage0_vision(clip_model, args, results_dir)
    else:
        logger.info("Stage 0 vision: skipped (stage0_vision_steps=0)")

    if args.stage0_lm_steps > 0:
        _run_stage0_lm(clip_model, args, results_dir)
    else:
        logger.info("Stage 0 LM: skipped (stage0_lm_steps=0)")

    # Sanity: all CLIP layers trainable again so Stage 1 trains the full model.
    for layer in _iter_clip_sublayers(clip_model):
        if not layer.trainable:
            raise RuntimeError(
                f"Layer {layer.name} still frozen after Stage 0. "
                "_unfreeze_clip did not restore trainable flags."
            )
    if clip_model.logit_scale is not None and not clip_model.logit_scale.trainable:
        raise RuntimeError(
            "logit_scale still frozen after Stage 0 — contrastive training "
            "would be broken."
        )

    # --- Curriculum ---
    histories: Dict[str, keras.callbacks.History] = {}
    for stage in (1, 2):
        stage_image_size = _resolve_image_size(args, stage)
        stage_epochs = _resolve_epochs(args, stage)
        if stage_epochs <= 0:
            logger.info(f"Stage {stage} skipped (epochs=0)")
            continue

        # Synthetic mode: images already fixed at stage 2 size. For
        # non-synthetic mode, build the pipeline at the stage-specific size.
        if args.synthetic and stage_image_size != args.stage2_image_size:
            # Resize the pre-generated array via a cheap bilinear pass.
            train_imgs_stage = tf.image.resize(
                train_images, (stage_image_size, stage_image_size),
                method="bilinear",
            ).numpy()
            val_imgs_stage = tf.image.resize(
                val_images, (stage_image_size, stage_image_size),
                method="bilinear",
            ).numpy()
        else:
            train_imgs_stage = train_images
            val_imgs_stage = val_images

        train_ds = make_image_text_tf_dataset(
            train_imgs_stage,
            train_tokens,
            image_size=stage_image_size,
            batch_size=args.batch_size,
            training=True,
            cache_decoded=args.cache_decoded,
        )
        val_ds = make_image_text_tf_dataset(
            val_imgs_stage,
            val_tokens,
            image_size=stage_image_size,
            batch_size=args.batch_size,
            training=False,
            cache_decoded=args.cache_decoded,
        )

        # A smaller slice of val used for in-training retrieval probes.
        # Intentionally capped so each probe is fast; the full-val
        # retrieval is computed once at the very end of training. The
        # probe set is tiny (~512 pairs), so caching it is always fine
        # regardless of the main ``--cache-decoded`` setting.
        probe_n = min(args.probe_num_pairs, len(val_tokens))
        if isinstance(val_imgs_stage, np.ndarray):
            probe_imgs = val_imgs_stage[:probe_n]
        else:
            probe_imgs = list(val_imgs_stage[:probe_n])
        probe_ds = make_image_text_tf_dataset(
            probe_imgs,
            val_tokens[:probe_n],
            image_size=stage_image_size,
            batch_size=args.batch_size,
            training=False,
            cache_decoded=True,
        )

        # steps_per_epoch for the cosine schedule. For list-backed datasets
        # we can't rely on cardinality being known.
        n_train = len(train_tokens)
        steps_per_epoch = max(1, n_train // args.batch_size)

        stage_cfg = CliffordCLIPTrainConfig(
            image_size=stage_image_size,
            epochs=stage_epochs,
            batch_size=args.batch_size,
            peak_lr=_resolve_lr(args, stage),
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
        )

        stage_name = f"stage{stage}_size{stage_image_size}"
        histories[f"stage{stage}"] = _run_stage(
            stage_name=stage_name,
            wrapper=wrapper,
            stage_cfg=stage_cfg,
            train_ds=train_ds,
            val_ds=val_ds,
            probe_ds=probe_ds,
            steps_per_epoch=steps_per_epoch,
            run_dir=results_dir,
            persistent_callbacks=persistent_callbacks,
            retrieval_probe_cb=retrieval_probe_cb,
        )

    # Close the persistent CSV writer cleanly after both stages.
    step_ckpt_cb.close()

    # --- Save final model ---
    final_path = os.path.join(results_dir, f"cliffordclip_{args.variant}.keras")
    try:
        clip_model.save(final_path)
        logger.info(f"Saved final CliffordCLIP to: {final_path}")
        sample = {
            "image": np.zeros(
                (2, args.stage2_image_size, args.stage2_image_size, 3),
                dtype=np.float32,
            ),
            "text": np.zeros((2, args.context_length), dtype=np.int32),
        }
        expected = clip_model(sample, training=False)
        validate_model_loading(
            final_path, sample, expected,
            custom_objects={"CliffordCLIP": CliffordCLIP},
        )
    except Exception as exc:
        logger.warning(f"Failed to save/validate final model: {exc}")

    # --- Retrieval eval on val split at final resolution ---
    logger.info("Computing retrieval recall@K on val split...")
    val_ds_final = make_image_text_tf_dataset(
        val_images,
        val_tokens,
        image_size=args.stage2_image_size,
        batch_size=args.batch_size,
        training=False,
        cache_decoded=False,  # one-shot eval, no reuse benefit
    )
    metrics = _compute_retrieval_metrics(clip_model, val_ds_final)
    logger.info("Retrieval metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # --- Summary ---
    summary_path = os.path.join(results_dir, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CliffordCLIP Training Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Variant: {args.variant}\n")
        f.write(f"Synthetic data: {args.synthetic}\n")
        f.write(f"Vocab size: {vocab_size}\n")
        f.write(f"Context length: {args.context_length}\n")
        f.write(f"Parameters: {clip_model.count_params():,}\n\n")
        f.write(
            f"Stage 0 vision: steps={args.stage0_vision_steps}, "
            f"lr={args.stage0_vision_lr}, wd={args.stage0_vision_wd}, "
            f"batch={args.stage0_vision_batch_size}\n"
        )
        f.write(
            f"Stage 0 LM:     steps={args.stage0_lm_steps}, "
            f"lr={args.stage0_lm_lr}, wd={args.stage0_lm_wd}, "
            f"batch={args.stage0_lm_batch_size}\n"
        )
        f.write(
            f"Stage 1: size={args.stage1_image_size}, "
            f"epochs={args.stage1_epochs}, lr={args.stage1_lr}\n"
        )
        f.write(
            f"Stage 2: size={args.stage2_image_size}, "
            f"epochs={args.stage2_epochs}, lr={args.stage2_lr}\n\n"
        )
        f.write("Retrieval metrics (val split):\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
    logger.info(f"Summary: {summary_path}")
    logger.info("Training complete.")


# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CliffordCLIP with two-stage resolution curriculum"
    )

    # Model
    parser.add_argument(
        "--variant", type=str, default="nano",
        choices=["nano", "mini", "small", "base", "large"],
    )
    parser.add_argument("--vision-patch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument(
        "--head-kind",
        type=str,
        default="learned_query_residual",
        choices=[
            "plain", "mean_max", "learned_query", "learned_query_residual",
        ],
        help=(
            "Projection head variant. 'plain' = single-view GAP/last-token "
            "+ Dense (baseline). 'mean_max' = Clifford-aware head combining "
            "GAP/GMP (vision) and masked-mean/last-token (text) through "
            "SparseRollingGeometricProduct. 'learned_query' = Clifford-aware "
            "head where the second pooling view is a learnable attention "
            "pool. 'learned_query_residual' = plain-CLIP anchor plus a "
            "LayerScale-gated Clifford geometric product of (mean, "
            "attn-pool) as a residual. Defaults to 'learned_query_residual'."
        ),
    )
    parser.add_argument(
        "--head-cli-mode",
        type=str,
        default="full",
        choices=["inner", "wedge", "full"],
        help=(
            "Clifford components used in the Clifford-aware projection head. "
            "'full' uses both the symmetric (inner) and antisymmetric (wedge) "
            "terms; 'wedge' uses only the bivector (structural) term; 'inner' "
            "uses only the scalar (coherence) term. Defaults to 'full'."
        ),
    )
    parser.add_argument(
        "--tokenizer-encoding", type=str, default="gpt2",
        help=(
            "tiktoken encoding name. Default 'gpt2' is English-trained "
            "(50257 tokens). Do NOT change to cl100k_base unless you want "
            "a multilingual model."
        ),
    )

    # Stage 1 (low-res)
    parser.add_argument("--stage1-image-size", type=int, default=96)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)

    # Stage 2 (high-res, target)
    parser.add_argument("--stage2-image-size", type=int, default=160)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--stage2-lr", type=float, default=5e-4)

    # Stage 0 (optional independent pretraining — vision on CIFAR-100, text on Wikipedia)
    parser.add_argument(
        "--stage0-vision-steps", type=int, default=50000,
        help=(
            "Number of training steps for Stage 0 vision pretraining on "
            "CIFAR-100. 0 disables. Default 50000."
        ),
    )
    parser.add_argument(
        "--stage0-lm-steps", type=int, default=50000,
        help=(
            "Number of training steps for Stage 0 LM pretraining on "
            "Wikipedia. 0 disables. Default 50000."
        ),
    )
    parser.add_argument(
        "--skip-stage0", action="store_true",
        help="Shortcut: zero out both Stage 0 step counts.",
    )
    parser.add_argument("--stage0-vision-lr", type=float, default=1e-3)
    parser.add_argument("--stage0-vision-wd", type=float, default=0.1)
    parser.add_argument("--stage0-vision-batch-size", type=int, default=128)
    parser.add_argument("--stage0-lm-lr", type=float, default=3e-4)
    parser.add_argument("--stage0-lm-wd", type=float, default=0.01)
    parser.add_argument("--stage0-lm-batch-size", type=int, default=8)
    parser.add_argument(
        "--stage0-lm-hf-cache", type=str,
        default="/media/arxwn/data0_4tb/datasets/wikipedia",
        help="HuggingFace cache dir for the Wikipedia dataset.",
    )

    # Shared training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)

    # Intermediate-results tracking (step-based checkpoints + retrieval probes)
    parser.add_argument(
        "--save-every-steps", type=int, default=500,
        help="Save a step_NNNNNNN.keras checkpoint every N training steps.",
    )
    parser.add_argument(
        "--log-every-steps", type=int, default=50,
        help="Append a row to training_log.csv every N training steps.",
    )
    parser.add_argument(
        "--max-checkpoints", type=int, default=3,
        help="Rolling window size for step_NNNNNNN.keras files.",
    )
    parser.add_argument(
        "--probe-every-steps", type=int, default=500,
        help=(
            "Compute retrieval recall@K on a small val slice every N "
            "training steps and append to retrieval_probes/probes.jsonl. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--probe-num-pairs", type=int, default=512,
        help=(
            "Number of (image, caption) pairs used per retrieval probe. "
            "Smaller = faster probe, coarser recall. 512 gives meaningful "
            "R@1 resolution without slowing training much."
        ),
    )

    parser.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help=(
            "CLIP contrastive loss label smoothing in [0, 1]. Small values "
            "(~0.1) regularise against overconfident matches on noisy "
            "captions. Set 0.0 to match the original CLIP paper exactly."
        ),
    )

    # Data
    parser.add_argument(
        "--dataset", type=str, default="coco2017",
        choices=["coco2017", "cc3m"],
        help=(
            "Image-caption dataset to train on. 'coco2017' reads the "
            "local MS-COCO 2017 extraction (~118k pairs); 'cc3m' reads "
            "Conceptual Captions 3M extracted by prepare_cc3m.py "
            "(~3.3M pairs)."
        ),
    )
    parser.add_argument(
        "--coco-root", type=str,
        default="/media/arxwn/data0_4tb/datasets/coco_2017",
        help=(
            "Path to an extracted COCO 2017 tree containing train2017/, "
            "val2017/, and annotations/. Symlinks are fine."
        ),
    )
    parser.add_argument(
        "--cc3m-root", type=str,
        default="/media/arxwn/data0_4tb/datasets/cc3m",
        help=(
            "Path to a CC3M extraction (output of prepare_cc3m.py). "
            "Must contain train/, validation/, and the corresponding "
            "*_captions.jsonl files."
        ),
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--cache-decoded", action="store_true",
        help=(
            "Cache JPEG-decoded uint8 tensors in RAM (default: off, stream "
            "from disk every epoch). Caching gives a ~20-40%% per-step "
            "speedup on small datasets but grows RAM linearly with dataset "
            "size — do NOT enable above ~200k samples without watching RAM "
            "usage. The streaming default scales to arbitrary dataset "
            "sizes."
        ),
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use random tensors instead of COCO 2017 (smoke test).",
    )

    # Infra
    parser.add_argument("--gpu", type=int, default=None)

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
