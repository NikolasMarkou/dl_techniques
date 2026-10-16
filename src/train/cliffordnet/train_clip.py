"""Training script for CliffordCLIP (CLIP with Clifford geometric blocks).

Trains :class:`CliffordCLIP` on an image-caption dataset using a symmetric
contrastive cross-entropy objective. An **optional pretraining phase**
pretrains the two towers independently before contrastive training:

- Vision tower on CIFAR-100 classification (recipe from
  ``train.cliffordnet.train_cliffordnet``).
- Text tower on Wikipedia causal LM (recipe from
  ``train.cliffordnet.train_cliffordnet_nlp``).

Pretraining mutates the existing ``CliffordCLIP`` object in-place via thin
throwaway wrappers — no save/reload between phases — trained with stock
``fit``. Defaults: 50,000 steps per pretraining helper. Disable with
``--skip-pretrain`` or ``--pretrain-{vision,lm}-steps 0``. Under
``--synthetic`` every stage runs on in-memory data (random CIFAR-shaped
images with 100 labels, random-word text), so a smoke run reaches both
pretraining stages without CIFAR-100 or Wikipedia on disk.

The training recipe borrows schedule ideas from the Penguin-VL paper
(arXiv:2603.06569):

- AdamW optimiser, cosine LR decay with a 3% warmup ratio
- Single CLIP contrastive stage at a configurable target resolution
  (``--image-size``, default 112). The previous two-stage low-to-high
  resolution curriculum was removed in favour of training directly at
  the final patch size.
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
are intentionally left out. Adding them would mean a teacher forward pass
inside the compiled loss of the stock ``compile``/``fit`` bridge
(``train.common.contrastive.compile_for_contrastive_training``) this script
trains through.

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
        --variant nano --image-size 112 --epochs 10 --peak-lr 5e-4 \\
        --batch-size 64 --gpu 0

    # smoke test, every stage (two pretraining steps each)
    python -m train.cliffordnet.train_clip --synthetic --epochs 1 \\
        --image-size 64 --batch-size 8 --max-train-samples 64 \\
        --pretrain-vision-steps 2 --pretrain-vision-batch-size 8 \\
        --pretrain-lm-steps 2 --pretrain-lm-batch-size 2 \\
        --steps-per-epoch 2 --output-dir /tmp/cliffordclip_smoke

Training is stock ``keras.Model.fit`` on the contrastive bridge of
``train.common.contrastive`` (`compile_for_contrastive_training`), which stacks
the two logit matrices into one output for the compiled loss. Every ``.keras``
the run writes (``checkpoints/step_*.keras``, ``checkpoints/final.keras``,
``cliffordclip_<variant>.keras``) is the bare :class:`CliffordCLIP`, never the
training wrapper.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.models.vision_language.clifford_clip import CliffordCLIP
from dl_techniques.optimization import (
    learning_rate_schedule_builder,
    optimizer_builder,
)
from dl_techniques.utils.logger import logger

from train.common.image_text import IMAGE_MEAN, IMAGE_STD
from train.common import (
    StepCheckpointCallback,
    build_synthetic_image_text_dataset,
    load_cc3m_local_split,
    load_coco2017_local_split,
    make_image_text_tf_dataset,
    set_seeds,
    setup_gpu,
    validate_model_loading,
)
from train.common.contrastive import (
    as_contrastive_fit_batches,
    build_contrastive_loss,
    compile_for_contrastive_training,
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
# Contrastive training: the inner loss and the logged temperature
# =============================================================================
#
# DECISION plan-2026-10-14T062616-81938e81/D-045: train through stock
# `compile`/`fit` on the stacking bridge of `train.common.contrastive`
# (`compile_for_contrastive_training`, D-036): `StackedCLIPLogits` stacks the
# two logit matrices into one output and `StackedCLIPContrastiveLoss` hands
# them to the inner loss `build_contrastive_loss` builds. Do NOT bring back a training wrapper
# with its own `train_step`/`test_step` (the old `ContrastiveCliffordCLIP`,
# whose D-003 said a dict-output CLIP could not use stock fit; the bridge is
# the measured counterexample), and do NOT save the bridge: every `.keras`
# this run writes is the bare CliffordCLIP (`StepCheckpointCallback(
# saved_model=...)`). The step-override census in test_package_api_contract.py
# no longer lists this file. See decisions.md D-045.


class LogitScaleLogger(keras.callbacks.Callback):
    """Write the model's effective temperature into ``logs`` as ``logit_scale``.

    The clipped ``exp(logit_scale)`` the forward pass multiplies into the
    logits (``CliffordCLIP._get_logit_scale``), after every training batch and
    at every epoch end, so the step CSV of `StepCheckpointCallback` and the
    ``fit`` history keep the column the old wrapper's ``train_step`` logged.
    Place it BEFORE the callbacks that read ``logs``: Keras hands every
    callback the same dict, in list order.

    :param clip_model: The :class:`CliffordCLIP` being trained (not the
        training wrapper).
    """

    def __init__(self, clip_model: CliffordCLIP) -> None:
        super().__init__()
        self.clip_model = clip_model

    def _write(self, logs: Optional[Dict[str, Any]]) -> None:
        if logs is not None:
            logs["logit_scale"] = float(
                keras.ops.convert_to_numpy(self.clip_model._get_logit_scale())
            )

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._write(logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._write(logs)


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
    :param probe_every_steps: Run a probe every N training steps. ``0`` (or
        less) disables the probe.
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

        Called once per stage from the training loop so the callback
        always evaluates at the active training resolution.
        """
        self._probe_ds = probe_ds
        self._stage_label = stage_label

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._global_step += 1
        # `--probe-every-steps 0` is the documented "disabled"; without this
        # guard the modulo below raised ZeroDivisionError on the first batch.
        if self._probe_ds is None or self.probe_every_steps <= 0:
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


class GammaProbeCallback(keras.callbacks.Callback):
    """Log the LayerScale gamma magnitude of the projection heads.

    Detects the **E1 wedge-ignition transition**: in the
    ``head_kind="learned_query_residual"`` configuration each tower mixes a
    deterministic feature with a Clifford-geometric branch scaled by a
    learnable per-channel ``LayerScale.gamma`` (init ~1e-5). When training
    pushes the mean of that gamma above ~0.1 the Clifford head has started
    to contribute — i.e. the geometric branch has "ignited". Tracking the
    gamma mean every N steps is the cheap A/B telemetry that tells whether
    the learned-query-residual arm is actually using its extra machinery.

    The ``model`` Keras attaches via ``set_model`` is the training wrapper
    (``train.common.contrastive.StackedCLIPLogits``); the head_scale layers live
    on the inner :class:`CliffordCLIP` (``self.model.clip_model``).

    Heads other than ``learned_query_residual`` (notably ``head_kind="plain"``)
    have NO ``vision_head_scale`` / ``text_head_scale`` (they are ``None``),
    so the lookup is guarded with ``getattr`` and the probe simply skips —
    it never crashes the plain-arm run.

    :param probe_every_steps: Log the gamma means every N training steps.
        Mirrors the other persistent callbacks' step-counter pattern.
    :param initial_step: Starting step count (for resume).
    """

    def __init__(
        self,
        probe_every_steps: int = 500,
        initial_step: int = 0,
    ) -> None:
        super().__init__()
        self.probe_every_steps = probe_every_steps
        self._global_step = initial_step
        logger.info(
            f"GammaProbeCallback: log head-scale gamma mean every "
            f"{probe_every_steps:,} steps (wedge-ignition probe)"
        )

    @property
    def global_step(self) -> int:
        return self._global_step

    def _inner_clip(self) -> Optional[CliffordCLIP]:
        # ``model`` is the StackedCLIPLogits wrapper; the heads live on the
        # inner CliffordCLIP. Bare CliffordCLIP (no wrapper) is tolerated.
        return getattr(self.model, "clip_model", self.model)

    @staticmethod
    def _gamma_mean(inner: Any, attr: str) -> Optional[float]:
        # ``head_kind="plain"`` (and other non-residual heads) leave the
        # head_scale attribute as None -> skip without crashing.
        layer = getattr(inner, attr, None)
        if layer is None:
            return None
        gamma = getattr(layer, "gamma", None)
        if gamma is None:
            return None
        try:
            return float(keras.ops.mean(keras.ops.convert_to_numpy(gamma)))
        except Exception:
            return None

    def on_train_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._global_step += 1
        if self.probe_every_steps <= 0:
            return
        if self._global_step % self.probe_every_steps != 0:
            return
        inner = self._inner_clip()
        if inner is None:
            return
        v_mean = self._gamma_mean(inner, "vision_head_scale")
        t_mean = self._gamma_mean(inner, "text_head_scale")
        if v_mean is None and t_mean is None:
            # No head_scale on either tower (e.g. head_kind="plain"): skip.
            return
        v_str = "n/a" if v_mean is None else f"{v_mean:.5f}"
        t_str = "n/a" if t_mean is None else f"{t_mean:.5f}"
        logger.info(
            f"[gamma-probe] step={self._global_step:,} "
            f"vision_gamma_mean={v_str} text_gamma_mean={t_str}"
        )


# =============================================================================
# Pretraining: independent tower pretraining (vision on CIFAR-100, text on Wikipedia)
# =============================================================================
#
# The two pretraining helpers train the *existing* ``CliffordCLIP`` sub-layers
# in place via a throwaway wrapper that attaches a task-specific head
# (classifier for vision, LM head for text), through stock ``compile``/``fit``.
# Weights land directly in ``clip_model`` — no save/reload, no cross-model
# weight transfer.
#
# Why not train standalone ``CliffordNet`` / ``CliffordNetLM`` and transfer?
# Layer names don't match (``clifford_block_N`` vs ``vision_clifford_block_N``
# / ``text_clifford_block_N``), so ``load_weights_from_checkpoint`` would
# need a per-layer name map. Wrapping is simpler and keeps the whole thing
# in one process.

# DECISION plan-2026-10-14T062616-81938e81/D-069: every direct CliffordCLIP
# sub-layer gets exactly one pretraining role by NAME, from the tables below,
# and an unclassified sub-layer raises. Do NOT go back to hand-kept attribute
# lists read with getattr(model, name, None): four of those names
# (vision_global_pool, vision_global_max_pool, vision_query_pool,
# text_query_pool) had stopped existing, the lookups skipped them in silence,
# the attention context pools were never frozen, and the default vision
# pretraining died on AttributeError.
# test_the_pretraining_freeze_rules_name_real_layers.py checks every name and
# prefix against a built model. See decisions.md D-069.

#: Vision backbone: what the CIFAR-100 classifier runs (stem, staged blocks
#: and their drop-paths, PatchMerging, the stage projections, the mean-pool
#: anchor and the head norm/dropout).
_VISION_BACKBONE_PREFIXES: Tuple[str, ...] = (
    "vision_stem", "vision_clifford_block_", "vision_drop_path_",
    "vision_patch_merge_", "vision_merge_proj_",
)
_VISION_BACKBONE_NAMES: Tuple[str, ...] = (
    "vision_det_pool", "vision_head_norm", "vision_head_dropout",
)

#: Text backbone: what the causal-LM wrapper runs.
_TEXT_BACKBONE_PREFIXES: Tuple[str, ...] = (
    "text_clifford_block_", "text_drop_path_",
)
_TEXT_BACKBONE_NAMES: Tuple[str, ...] = (
    "token_embedding", "position_embedding", "text_embed_norm",
    "text_embed_dropout", "text_head_norm", "text_head_dropout",
)

#: The projections and the Clifford head (geometric products, context pools,
#: the text mean pool, LayerScale gates): always frozen during pretraining,
#: since they would learn against an objective that does not exist yet.
_PRETRAIN_ALWAYS_FROZEN = frozenset({
    "vision_projection", "text_projection",
    "vision_head_geo", "text_head_geo",
    "vision_ctx_pool", "text_ctx_pool", "text_det_pool",
    "vision_head_scale", "text_head_scale",
})


def _iter_clip_sublayers(clip_model: CliffordCLIP) -> List[keras.layers.Layer]:
    """Every direct sub-layer of ``clip_model`` (lists flattened), in tracking order."""
    return list(clip_model.layers)


def _is_vision_backbone(name: str) -> bool:
    return name in _VISION_BACKBONE_NAMES or name.startswith(_VISION_BACKBONE_PREFIXES)


def _is_text_backbone(name: str) -> bool:
    return name in _TEXT_BACKBONE_NAMES or name.startswith(_TEXT_BACKBONE_PREFIXES)


def _freeze_clip_for_pretrain(
    clip_model: CliffordCLIP, *, train_tower: str,
) -> None:
    """Freeze all sub-layers except the requested tower's backbone.

    The learnable ``logit_scale`` scalar is also frozen — it has no role
    before contrastive training and must not drift under weight decay.

    :raises RuntimeError: For a sub-layer with no pretraining role (a layer
        added to CliffordCLIP without a row in the tables above).
    """
    assert train_tower in ("vision", "text"), train_tower
    for layer in _iter_clip_sublayers(clip_model):
        name = layer.name
        if name in _PRETRAIN_ALWAYS_FROZEN:
            layer.trainable = False
        elif _is_vision_backbone(name):
            layer.trainable = train_tower == "vision"
        elif _is_text_backbone(name):
            layer.trainable = train_tower == "text"
        else:
            raise RuntimeError(
                f"CliffordCLIP sub-layer {name!r} has no pretraining role; add "
                "it to the vision/text backbone tables or _PRETRAIN_ALWAYS_FROZEN"
            )
    if clip_model.logit_scale is not None:
        clip_model.logit_scale.trainable = False


def _unfreeze_clip(clip_model: CliffordCLIP) -> None:
    """Undo pretraining freezes so contrastive training trains everything."""
    for layer in _iter_clip_sublayers(clip_model):
        layer.trainable = True
    if clip_model.logit_scale is not None:
        clip_model.logit_scale.trainable = True


class _VisionClassifier(keras.Model):
    """Pretraining wrapper that reuses CliffordCLIP vision sub-layers.

    Forward path: ``_vision_tokens`` (stem + staged Clifford blocks +
    PatchMerging downsamples, flattened to a token sequence) →
    ``vision_det_pool`` (the mean, the model's CLIP anchor) →
    ``vision_head_norm`` → (optional ``vision_head_dropout``) → fresh
    ``Dense(num_classes)``. The projection / Clifford-head sub-layers are
    intentionally skipped — they have no meaning outside contrastive training
    and are frozen anyway.
    """

    def __init__(
        self, clip_model: CliffordCLIP, num_classes: int, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.clip_model = clip_model
        self.classifier = keras.layers.Dense(
            num_classes, name="pretrain_vision_classifier",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the fresh classifier on the pooled width.

        ``clip_model`` is built already. Without this method Keras' default
        ``build`` ran nothing, warned that the classifier had unbuilt state,
        and marked the wrapper built anyway.
        """
        self.classifier.build((input_shape[0], self.clip_model.vision_channels))
        super().build(input_shape)

    def call(self, images, training: Optional[bool] = None):
        m = self.clip_model
        x = m.vision_det_pool(m._vision_tokens(images, training=training))
        x = m.vision_head_norm(x)
        if m.vision_head_dropout is not None:
            x = m.vision_head_dropout(x, training=training)
        return self.classifier(x)


#: Image side of the CIFAR-100 pretraining set, which ``--synthetic`` mimics.
_CIFAR_IMAGE_SHAPE: Tuple[int, int, int] = (32, 32, 3)
_CIFAR100_CLASSES = 100

# DECISION plan-2026-10-14T062616-81938e81/D-090: the vision pretraining
# normalises with the CONTRASTIVE stage's constants (train.common.image_text's
# IMAGE_MEAN/IMAGE_STD, the objects that stage's augment_and_normalize reads),
# not CIFAR-100's own statistics. With CIFAR stats the pretrained tower met a
# shifted input distribution the moment contrastive training began. Do not
# give the pretraining its own constant again.
_PRETRAIN_NORMALIZATION = (IMAGE_MEAN, IMAGE_STD)


def _pretrain_vision_datasets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """The CIFAR-100 pretraining datasets, normalised like the contrastive stage.

    Interface contract (callers: `_run_pretrain_vision`, tests): the
    augmented training and plain evaluation pipelines of
    ``train.cliffordnet.train_cliffordnet`` with ``_PRETRAIN_NORMALIZATION``
    in place of the CIFAR-100 statistics. Images are ``[0, 1]`` floats.

    :return: ``(train_ds, val_ds)``.
    """
    from train.cliffordnet.train_cliffordnet import (
        build_eval_dataset,
        build_train_dataset,
    )

    train_ds = build_train_dataset(
        x_train, y_train,
        batch_size=batch_size,
        dataset_name="cifar100",
        random_erasing_prob=0.25,
        normalization=_PRETRAIN_NORMALIZATION,
    )
    val_ds = build_eval_dataset(
        x_test, y_test,
        batch_size=batch_size,
        dataset_name="cifar100",
        normalization=_PRETRAIN_NORMALIZATION,
    )
    return train_ds, val_ds


def _synthetic_cifar100(num_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random CIFAR-100-shaped images in ``[0, 1]`` and labels in ``[0, 100)``.

    The ``--synthetic`` stand-in for ``load_dataset("cifar100")``: the same
    dtype, range and shapes, so the same augmentation pipeline consumes it.
    """
    rng = np.random.default_rng(seed)
    images = rng.uniform(0.0, 1.0, (num_samples, *_CIFAR_IMAGE_SHAPE)).astype(np.float32)
    labels = rng.integers(0, _CIFAR100_CLASSES, num_samples).astype(np.int64)
    return images, labels


def _run_pretrain_vision(
    clip_model: CliffordCLIP,
    args: argparse.Namespace,
    run_dir: str,
) -> None:
    """Pretrain the CliffordCLIP vision tower on CIFAR-100 classification.

    Reuses the augmentation pipeline from
    ``train.cliffordnet.train_cliffordnet`` (AutoAugment CIFAR-10 policy,
    random crop, HFlip, per-channel normalise, RandomErasing), normalised
    with the contrastive stage's mean/std (`_pretrain_vision_datasets`). Under
    ``--synthetic`` the arrays are random (two training batches and one
    validation batch) and the pipeline is the same.

    The step budget is honoured: an epoch is one pass over the training set
    or the budget, whichever is shorter, and the epoch count is the budget
    over that, rounded up.
    """
    from train.common import load_dataset

    steps_budget = int(args.pretrain_vision_steps)
    batch_size = args.pretrain_vision_batch_size
    logger.info("=" * 60)
    logger.info(f"Pretrain vision: CIFAR-100 pretrain ({steps_budget:,} steps)")
    logger.info("=" * 60)

    if args.synthetic:
        x_train, y_train = _synthetic_cifar100(2 * batch_size, seed=args.seed)
        x_test, y_test = _synthetic_cifar100(batch_size, seed=args.seed + 1)
        num_classes = _CIFAR100_CLASSES
    else:
        (x_train, y_train), (x_test, y_test), _shape, num_classes = load_dataset(
            "cifar100", batch_size=batch_size,
        )
    if num_classes != _CIFAR100_CLASSES:
        raise RuntimeError(
            f"Pretrain vision expected CIFAR-100 (100 classes); "
            f"got num_classes={num_classes}"
        )

    train_ds, val_ds = _pretrain_vision_datasets(
        x_train, y_train, x_test, y_test, batch_size)
    steps_per_epoch = max(1, min(len(x_train) // batch_size, steps_budget))
    pretrain_epochs = max(1, -(-steps_budget // steps_per_epoch))  # ceil
    warmup_steps = max(1, int(0.05 * steps_budget))
    decay_steps = max(1, steps_budget - warmup_steps)

    schedule = learning_rate_schedule_builder({
        "type": "cosine_decay",
        "learning_rate": args.pretrain_vision_lr,
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
        "weight_decay": args.pretrain_vision_wd,
    }, schedule)

    _freeze_clip_for_pretrain(clip_model, train_tower="vision")

    wrapper = _VisionClassifier(
        clip_model, num_classes=num_classes, name="pretrain_vision_wrapper",
    )
    wrapper.build((None, *_CIFAR_IMAGE_SHAPE))

    n_train = sum(int(np.prod(v.shape)) for v in wrapper.trainable_variables)
    n_total = sum(int(np.prod(v.shape)) for v in wrapper.variables)
    logger.info(
        f"Pretrain vision: trainable={n_train:,} / total={n_total:,} params "
        f"| {pretrain_epochs} epochs × {steps_per_epoch} steps/epoch "
        f"(budget {steps_budget:,}, warmup {warmup_steps:,}, "
        f"peak_lr={args.pretrain_vision_lr}, wd={args.pretrain_vision_wd})"
    )

    wrapper.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    tb_dir = os.path.join(run_dir, "tensorboard", "pretrain_vision")
    os.makedirs(tb_dir, exist_ok=True)
    wrapper.fit(
        train_ds.repeat(),
        validation_data=val_ds,
        epochs=pretrain_epochs,
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
    logger.info("Pretrain vision: complete (CLIP layers unfrozen for CLIP training).")


class _TextLMWrapper(keras.Model):
    """Pretraining wrapper that reuses CliffordCLIP text sub-layers for CLM.

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
        self.lm_head = keras.layers.Dense(vocab_size, name="pretrain_lm_head")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the fresh LM head on the text width (see ``_VisionClassifier.build``)."""
        self.lm_head.build(
            (input_shape[0], self.context_length, self.clip_model.text_channels))
        super().build(input_shape)

    def call(self, input_ids, training: Optional[bool] = None):
        m = self.clip_model
        positions = keras.ops.arange(self.context_length)
        x = m.token_embedding(input_ids) + m.position_embedding(positions)
        x = m.text_embed_norm(x)
        x = m.text_embed_dropout(x, training=training)
        # DECISION plan-2026-08-11T141925-eb34352d/D-003
        # Do NOT reintroduce the caller-side expand_dims(axis=1)/squeeze(axis=1)
        # around this loop: the text blocks are sequence-mode and take (B, L, D).
        #
        # The blocks are TRANSFORM-ONLY: the residual add is external, and each
        # block is near-zero at init (LayerScale gamma=1e-5). Applying them as
        # ``x = block(x)`` therefore REPLACES the signal with ~1e-5 of itself per
        # block instead of augmenting it -- measured 2.76 -> 3.05e-12 over 4
        # blocks, underflowing to exactly 0.0 at the default depth. This loop
        # must stay identical to ``CliffordCLIP.encode_text``, reusing that
        # model's own ``text_drop_paths`` so both phases train one function.
        for block, drop_path in zip(m.text_blocks, m.text_drop_paths):
            x = x + drop_path(block(x, training=training), training=training)
        x = m.text_head_norm(x)
        if m.text_head_dropout is not None:
            x = m.text_head_dropout(x, training=training)
        return {"logits": self.lm_head(x)}


def _synthetic_articles(num_articles: int, seed: int) -> List[str]:
    """Random-word articles: the ``--synthetic`` stand-in for Wikipedia.

    Each article is 60 words drawn from a seeded 200-word vocabulary of
    lowercase strings, so the tokenizer and the packing pipeline run exactly
    as on real text.
    """
    rng = np.random.default_rng(seed)
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    words = ["".join(rng.choice(letters, size=int(rng.integers(2, 9))))
             for _ in range(200)]
    return [" ".join(rng.choice(words, size=60)) for _ in range(num_articles)]


def _run_pretrain_lm(
    clip_model: CliffordCLIP,
    args: argparse.Namespace,
    run_dir: str,
) -> None:
    """Pretrain the CliffordCLIP text tower on Wikipedia causal LM.

    Uses the same context length as CliffordCLIP (``args.context_length``)
    so the in-place position_embedding remains shape-compatible with the
    subsequent contrastive training, and the run's own tokenizer
    (``--tokenizer-encoding``, the one the model's vocabulary was sized
    from; default ``gpt2``, vocab 50257, EOT 50256). The 4 extra specials
    that ``train_cliffordnet_nlp.py`` declares for MLM are not emitted in the
    CLM path, so we can pretrain without them. Under ``--synthetic`` the
    articles are random words (:func:`_synthetic_articles`) and the packing
    pipeline is the same.
    """
    from train.common.nlp import preprocess_clm_packed_dataset
    from dl_techniques.losses import MaskedCausalLMLoss
    import tiktoken

    steps_budget = int(args.pretrain_lm_steps)
    logger.info("=" * 60)
    logger.info(f"Pretrain LM: Wikipedia CLM pretrain ({steps_budget:,} steps)")
    logger.info("=" * 60)

    encoder = tiktoken.get_encoding(args.tokenizer_encoding)
    vocab_size = encoder.n_vocab
    eot_token_id = int(encoder.eot_token)
    logger.info(
        f"Pretrain LM tokenizer: {args.tokenizer_encoding}, "
        f"vocab_size={vocab_size}, eot_token_id={eot_token_id}"
    )

    if args.synthetic:
        train_raw = tf.data.Dataset.from_tensor_slices(
            _synthetic_articles(num_articles=16, seed=args.seed))
    else:
        from dl_techniques.datasets.nlp import load_wikipedia_train_val
        # Wikipedia data — fail fast if cache dir is absent.
        if not os.path.isdir(args.pretrain_lm_hf_cache):
            raise RuntimeError(
                f"Wikipedia cache dir not found: {args.pretrain_lm_hf_cache}. "
                "Point --pretrain-lm-hf-cache at a pre-downloaded HF cache or "
                "pass --pretrain-lm-steps 0 / --skip-pretrain to bypass LM pretrain."
            )
        train_raw, _val_raw = load_wikipedia_train_val(
            cache_dir=args.pretrain_lm_hf_cache,
            min_article_length=args.pretrain_lm_min_article_length,
            val_fraction=0.02,
            max_val_samples=5000,
            max_train_samples=None,
            seed=args.seed,
            num_shards=args.pretrain_lm_shuffle_shards,
        )

    chunk_length = args.context_length + 1  # +1 for the causal shift.
    train_ds = preprocess_clm_packed_dataset(
        train_raw,
        encoding_name=args.tokenizer_encoding,
        chunk_length=chunk_length,
        batch_size=args.pretrain_lm_batch_size,
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
        "learning_rate": args.pretrain_lm_lr,
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
        "weight_decay": args.pretrain_lm_wd,
    }, schedule)

    _freeze_clip_for_pretrain(clip_model, train_tower="text")

    wrapper = _TextLMWrapper(
        clip_model,
        vocab_size=vocab_size,
        context_length=args.context_length,
        name="pretrain_lm_wrapper",
    )
    wrapper.build((None, args.context_length))

    n_train = sum(int(np.prod(v.shape)) for v in wrapper.trainable_variables)
    n_total = sum(int(np.prod(v.shape)) for v in wrapper.variables)
    logger.info(
        f"Pretrain LM: trainable={n_train:,} / total={n_total:,} params "
        f"| 1 epoch × {steps_budget} steps/epoch "
        f"(warmup {warmup_steps:,}, peak_lr={args.pretrain_lm_lr}, "
        f"wd={args.pretrain_lm_wd}, context_length={args.context_length})"
    )

    wrapper.compile(
        optimizer=optimizer,
        loss={"logits": MaskedCausalLMLoss()},
        metrics={"logits": ["accuracy"]},
    )

    tb_dir = os.path.join(run_dir, "tensorboard", "pretrain_lm")
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
    logger.info("Pretrain LM: complete (CLIP layers unfrozen for CLIP training).")


# =============================================================================
# Stage training
# =============================================================================


def _build_optimizer(
    train_cfg: CliffordCLIPTrainConfig,
    steps_per_epoch: int,
) -> keras.optimizers.Optimizer:
    """Cosine LR with warmup_ratio warmup + AdamW — Penguin-VL recipe."""
    total_steps = max(1, train_cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(round(train_cfg.warmup_ratio * total_steps)))
    decay_steps = max(1, total_steps - warmup_steps)

    schedule = learning_rate_schedule_builder(
        {
            "type": "cosine_decay",
            "learning_rate": train_cfg.peak_lr,
            "decay_steps": decay_steps,
            "alpha": train_cfg.lr_alpha,
            "warmup_steps": warmup_steps,
            "warmup_start_lr": train_cfg.min_warmup_lr,
        }
    )
    optimizer = optimizer_builder(
        {
            "type": "adamw",
            "beta_1": 0.9,
            "beta_2": 0.98,
            "epsilon": 1e-6,
            "weight_decay": train_cfg.weight_decay,
        },
        schedule,
    )

    # Exclude the CLIP learnable temperature from weight decay. AdamW
    # otherwise pulls ``logit_scale`` toward 0 every step, which *lowers*
    # the effective temperature and collapses the contrastive signal. The
    # original CLIP training recipe also excludes biases and LayerNorm
    # gains; we keep the exclusion minimal (just the temperature) because
    # that is the one we have direct evidence is being harmed by decay.
    # ``exclude_from_weight_decay`` matches each pattern against the
    # variable's leaf ``name`` (``re.search``), not its path, and must be
    # called before the optimizer is built — we do it here while it is still
    # fresh. No fallback: a failure here would let decay pull the temperature
    # down, so it raises (test_the_pretraining_freeze_rules_name_real_layers.py
    # checks the pattern matches exactly ``logit_scale``).
    optimizer.exclude_from_weight_decay(var_names=["logit_scale"])
    logger.info("Excluded 'logit_scale' from AdamW weight decay.")

    logger.info(
        f"CLIP optimizer: cosine_decay peak_lr={train_cfg.peak_lr}, "
        f"warmup_steps={warmup_steps}/{total_steps} "
        f"(ratio={train_cfg.warmup_ratio}), wd={train_cfg.weight_decay}"
    )
    return optimizer


# =============================================================================
# Main
# =============================================================================


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
            image_size=args.image_size,
            context_length=args.context_length,
            vocab_size=vocab_size,
            eot_token_id=eot_token_id,
            seed=0,
        )
        val_images, val_tokens = build_synthetic_image_text_dataset(
            num_samples=args.max_val_samples or 64,
            image_size=args.image_size,
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


def train(args: argparse.Namespace) -> Path:
    """Run one CliffordCLIP training job; return the run directory."""
    logger.info("Starting CliffordCLIP training")
    setup_gpu(args.gpu)
    # `--seed` was read by the LM-pretrain loader only, while its help text
    # promised global tf/keras seeding; this makes the help true.
    set_seeds(args.seed)

    # Set the global mixed-precision policy BEFORE any layer/model is built so
    # it applies to every weight. bf16 needs no LossScaleOptimizer (unlike fp16).
    if args.mixed_bfloat16:
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        logger.info("Mixed precision enabled: global policy = mixed_bfloat16")

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

    # --- Dataset summary ---
    # One bracketed, greppable block so dataset sizes are immediately visible
    # in the training log. Distinct-image counts are reported because COCO
    # carries multiple captions per image (pairs > distinct images).
    if args.synthetic:
        source_name = "synthetic"
        source_root = "(in-memory)"
    elif args.dataset == "coco2017":
        source_name = "coco2017"
        source_root = args.coco_root
    elif args.dataset == "cc3m":
        source_name = "cc3m"
        source_root = args.cc3m_root
    else:
        source_name = args.dataset
        source_root = "(unknown)"

    def _distinct_image_count(images):
        if isinstance(images, np.ndarray):
            return int(images.shape[0])
        # path list (COCO/CC3M): caption-per-image so dedup by path string
        return len(set(images))

    n_train_pairs = len(train_tokens)
    n_val_pairs = len(val_tokens)
    n_train_distinct = _distinct_image_count(train_images)
    n_val_distinct = _distinct_image_count(val_images)
    summary_steps_per_epoch = (
        args.steps_per_epoch or max(1, n_train_pairs // args.batch_size)
    )
    summary_total_steps = summary_steps_per_epoch * args.epochs

    logger.info("=" * 60)
    logger.info("[Dataset summary]")
    logger.info(f"  source           : {source_name}")
    logger.info(f"  source root      : {source_root}")
    logger.info(
        f"  train pairs      : {n_train_pairs:,} "
        f"(distinct images: {n_train_distinct:,})"
    )
    logger.info(
        f"  val pairs        : {n_val_pairs:,} "
        f"(distinct images: {n_val_distinct:,})"
    )
    logger.info(f"  image resolution : {args.image_size}x{args.image_size}")
    logger.info(f"  context length   : {args.context_length}")
    logger.info(f"  vocab size       : {vocab_size}")
    logger.info(f"  batch size       : {args.batch_size}")
    logger.info(f"  steps per epoch  : {summary_steps_per_epoch:,}")
    logger.info(f"  epochs           : {args.epochs}")
    logger.info(f"  total steps      : {summary_total_steps:,}")
    logger.info("=" * 60)

    # --- Model ---
    # Pad sentinel must match the token id used to right-pad the sequences
    # (see train.common.image_text.tokenize_captions /
    # build_synthetic_image_text_dataset).
    clip_model = CliffordCLIP.from_variant(
        variant=args.variant,
        vocab_size=vocab_size,
        image_size=args.image_size,
        context_length=args.context_length,
        vision_patch_size=args.vision_patch_size,
        dropout_rate=args.dropout_rate,
        pad_token_id=eot_token_id,
        head_kind=args.head_kind,
        head_cli_mode=args.head_cli_mode,
        # DECISION plan_2026-05-31_42743977/D-001: thread the CLI flag through
        # (not hardcode text_use_global_context=True, which would silently flip
        # the shipped default; not leave it default-False, which contradicts doc
        # lever G). A flag keeps both A/B arms identical. See decisions.md D-001.
        text_use_global_context=args.text_use_global_context,
    )
    # Build at the configured resolution. Clifford blocks are
    # resolution-agnostic because they are fully convolutional.
    clip_model.build(
        {
            "image": (
                None,
                args.image_size,
                args.image_size,
                3,
            ),
            "text": (None, args.context_length),
        }
    )
    clip_model.summary(print_fn=logger.info)

    # --- Results dir ---
    # Single timestamped run dir per launch. Layout mirrors
    # ``train_cliffordnet_nlp.py``:
    #   results/cliffordclip_<variant>_<timestamp>/
    #     checkpoints/step_NNNNNNN.keras + final.keras
    #     retrieval_probes/probes.jsonl  <- intermediate results
    #     tensorboard/clip/
    #     training_log.csv               <- step-level metrics
    #     cliffordclip_<variant>.keras   <- final model (convenience)
    #     training_summary.txt
    if args.output_dir:
        results_dir = args.output_dir
    else:
        from datetime import datetime
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(
            "results",
            f"cliffordclip_{args.variant}_{run_ts}",
        )
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Run directory: {results_dir}")

    # --- Persistent run-level callbacks ---
    # Both callbacks update their own global step counters on every
    # ``on_train_batch_end``, giving a continuous step timeline and a
    # coherent ``checkpoints/`` + ``retrieval_probes/probes.jsonl``.
    step_ckpt_cb = StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=args.save_every_steps,
        max_checkpoints=args.max_checkpoints,
        log_every_steps=args.log_every_steps,
        analyze_every_steps=0,
        gc_on_save=True,
        # The CliffordCLIP, never the stock-fit bridge that trains it (D-045).
        saved_model=clip_model,
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
    if args.gamma_probe_every_steps > 0:
        persistent_callbacks.append(
            GammaProbeCallback(probe_every_steps=args.gamma_probe_every_steps)
        )

    # --- Pretraining: optional independent tower pretraining ---
    # Resolve skip flag up-front so later checks read the effective values.
    if args.skip_pretrain:
        args.pretrain_vision_steps = 0
        args.pretrain_lm_steps = 0

    if args.pretrain_vision_steps > 0:
        _run_pretrain_vision(clip_model, args, results_dir)
    else:
        logger.info("Pretrain vision: skipped (pretrain_vision_steps=0)")

    if args.pretrain_lm_steps > 0:
        _run_pretrain_lm(clip_model, args, results_dir)
    else:
        logger.info("Pretrain LM: skipped (pretrain_lm_steps=0)")

    # Sanity: all CLIP layers trainable again so contrastive training trains the full model.
    for layer in _iter_clip_sublayers(clip_model):
        if not layer.trainable:
            raise RuntimeError(
                f"Layer {layer.name} still frozen after pretraining. "
                "_unfreeze_clip did not restore trainable flags."
            )
    if clip_model.logit_scale is not None and not clip_model.logit_scale.trainable:
        raise RuntimeError(
            "logit_scale still frozen after pretraining — contrastive training "
            "would be broken."
        )

    # --- CLIP contrastive training ---
    # `({'image', 'text'}, zeros(N))`: stock `fit` needs a `y` to call the
    # loss at all; the bridge's loss and metrics ignore it.
    train_ds = as_contrastive_fit_batches(make_image_text_tf_dataset(
        train_images,
        train_tokens,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=True,
        cache_decoded=args.cache_decoded,
    ))
    val_ds = as_contrastive_fit_batches(make_image_text_tf_dataset(
        val_images,
        val_tokens,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=False,
        cache_decoded=args.cache_decoded,
    ))

    # A smaller slice of val used for in-training retrieval probes.
    # Intentionally capped so each probe is fast; the full-val
    # retrieval is computed once at the very end of training. The
    # probe set is tiny (~512 pairs), so caching it is always fine
    # regardless of the main ``--cache-decoded`` setting.
    probe_n = min(args.probe_num_pairs, len(val_tokens))
    if isinstance(val_images, np.ndarray):
        probe_imgs = val_images[:probe_n]
    else:
        probe_imgs = list(val_images[:probe_n])
    probe_ds = make_image_text_tf_dataset(
        probe_imgs,
        val_tokens[:probe_n],
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=False,
        cache_decoded=True,
    )

    # steps_per_epoch for `fit` and the cosine schedule: `--steps-per-epoch`,
    # or one pass. For list-backed datasets we can't rely on cardinality being
    # known; the training pipeline repeats, so a cap never runs it dry.
    steps_per_epoch = summary_steps_per_epoch

    train_cfg = CliffordCLIPTrainConfig(
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        peak_lr=args.peak_lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    logger.info(
        f"=== CLIP training: image_size={train_cfg.image_size}, "
        f"epochs={train_cfg.epochs}, batch={train_cfg.batch_size}, "
        f"peak_lr={train_cfg.peak_lr} ==="
    )

    optimizer = _build_optimizer(train_cfg, steps_per_epoch)
    wrapper = compile_for_contrastive_training(
        clip_model,
        optimizer,
        inner_loss=build_contrastive_loss(args.loss, args.label_smoothing),
    )

    tb_dir = os.path.join(results_dir, "tensorboard", "clip")
    os.makedirs(tb_dir, exist_ok=True)
    stage_local: List[keras.callbacks.Callback] = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.TensorBoard(
            log_dir=tb_dir,
            write_graph=False,
            update_freq="epoch",
        ),
    ]

    retrieval_probe_cb.set_probe_dataset(probe_ds, stage_label="clip")

    # LogitScaleLogger first: it writes `logit_scale` into the `logs` dict
    # every later callback (TensorBoard, the step CSV, History) reads.
    callbacks = (
        [LogitScaleLogger(clip_model)] + stage_local + list(persistent_callbacks)
    )
    logger.info(
        f"CLIP training: tensorboard={tb_dir}; "
        f"step-checkpoints + retrieval probes inherited from run-level state"
    )

    history = wrapper.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    # Close the persistent CSV writer cleanly.
    step_ckpt_cb.close()

    # DECISION plan-2026-09-16T114148-70556a67/D-008: StepCheckpointCallback's
    # dynamic-schema mode only writes a row every --log-every-steps steps and
    # never flushes at epoch/train end (see decisions.md D-006/D-008) -- this
    # smoke-scale run may finish with zero total steps ever crossing that
    # threshold, leaving training_log.csv missing entirely. Do NOT change the
    # shared step_checkpoint.py callback (out of blast-radius); write a
    # minimal local fallback CSV instead if the callback never created one.
    training_log_path = os.path.join(results_dir, "training_log.csv")
    if not os.path.exists(training_log_path):
        with open(training_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerow([
                len(history.history.get("loss", [])),
                history.history.get("loss", [None])[-1],
                history.history.get("val_loss", [None])[-1],
            ])

    # --- Save final model ---
    # Not inside a broad `except`: the saved model is the run's product, and a
    # failed save logged as a warning let a run end "complete" without it.
    final_path = os.path.join(results_dir, f"cliffordclip_{args.variant}.keras")
    clip_model.save(final_path)
    logger.info(f"Saved final CliffordCLIP to: {final_path}")
    sample = {
        "image": np.zeros(
            (2, args.image_size, args.image_size, 3),
            dtype=np.float32,
        ),
        "text": np.zeros((2, args.context_length), dtype=np.int32),
    }
    # DECISION plan-2026-09-16T114148-70556a67/D-005: use .predict(), not a
    # direct __call__, to build `expected`. validate_model_loading always
    # evaluates the RELOADED model via .predict() (train/common/evaluation.py),
    # and .predict()'s compiled/XLA-fused numerics diverge from a bare eager
    # __call__ by ~1e-4-1e-3 on this 12-layer network -- MEASURED reproducible
    # even with zero save/reload involved (same model, same weights). Mixing
    # __call__ (expected) with .predict() (loaded) therefore fires this
    # warning on every run regardless of whether weights survive the
    # round-trip; do not "fix" this by loosening tolerance or by chasing a
    # weight-materialization bug in CliffordCLIP.build() -- see decisions.md
    # D-005.
    expected = clip_model.predict(sample, verbose=0)
    validate_model_loading(
        final_path, sample, expected,
        custom_objects={"CliffordCLIP": CliffordCLIP},
    )

    # --- Retrieval eval on val split at final resolution ---
    logger.info("Computing retrieval recall@K on val split...")
    val_ds_final = make_image_text_tf_dataset(
        val_images,
        val_tokens,
        image_size=args.image_size,
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
            f"Pretrain vision: steps={args.pretrain_vision_steps}, "
            f"lr={args.pretrain_vision_lr}, wd={args.pretrain_vision_wd}, "
            f"batch={args.pretrain_vision_batch_size}\n"
        )
        f.write(
            f"Pretrain LM:     steps={args.pretrain_lm_steps}, "
            f"lr={args.pretrain_lm_lr}, wd={args.pretrain_lm_wd}, "
            f"batch={args.pretrain_lm_batch_size}\n"
        )
        f.write(
            f"CLIP training: image_size={args.image_size}, "
            f"epochs={args.epochs}, peak_lr={args.peak_lr}\n\n"
        )
        f.write("Retrieval metrics (val split):\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
    logger.info(f"Summary: {summary_path}")
    logger.info("Training complete.")
    return Path(results_dir)


# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CliffordCLIP: optional pretraining + single CLIP stage"
    )

    # Model
    parser.add_argument(
        "--variant", type=str, default="nano",
        choices=sorted(CliffordCLIP.MODEL_VARIANTS),
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

    # CLIP training (single stage at the configured resolution)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--steps-per-epoch", type=int, default=None,
        help=(
            "Cap on CLIP training steps per epoch (default: one pass over the "
            "training pairs). Also sizes the cosine schedule."
        ),
    )
    parser.add_argument("--peak-lr", type=float, default=5e-4)

    # Pretraining (optional independent pretraining — vision on CIFAR-100, text on Wikipedia)
    parser.add_argument(
        "--pretrain-vision-steps", type=int, default=50000,
        help=(
            "Number of training steps for vision pretraining on "
            "CIFAR-100. 0 disables. Default 50000."
        ),
    )
    parser.add_argument(
        "--pretrain-lm-steps", type=int, default=50000,
        help=(
            "Number of training steps for LM pretraining on "
            "Wikipedia. 0 disables. Default 50000."
        ),
    )
    parser.add_argument(
        "--skip-pretrain", action="store_true",
        help="Shortcut: zero out both pretraining step counts.",
    )
    parser.add_argument("--pretrain-vision-lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-vision-wd", type=float, default=0.1)
    parser.add_argument("--pretrain-vision-batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lm-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-lm-wd", type=float, default=0.01)
    parser.add_argument("--pretrain-lm-batch-size", type=int, default=8)
    parser.add_argument(
        "--pretrain-lm-hf-cache", type=str,
        default="/media/arxwn/data0_4tb/datasets/wikipedia",
        help="HuggingFace cache dir for the Wikipedia dataset.",
    )
    parser.add_argument(
        "--pretrain-lm-min-article-length", type=int, default=0,
        help="HF Wikipedia char-length filter for the LM pretrain stage. "
             "0 (default) = no filter (recommended for packed CLM).",
    )
    parser.add_argument(
        "--pretrain-lm-shuffle-shards", type=int, default=4,
        help="Parallel tokenization shards for the LM pretrain Wikipedia "
             "stream (D-002). 1 = single-thread; >1 reshuffles per epoch.",
    )

    # Shared training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument(
        "--loss", choices=["clip", "siglip"], default="clip",
        help="Contrastive loss variant",
    )

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
        "--probe-every-steps", type=int, default=750,
        help=(
            "Compute retrieval recall@K on a small val slice every N "
            "training steps and append to retrieval_probes/probes.jsonl. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--gamma-probe-every-steps", type=int, default=500,
        help=(
            "Log the mean of the projection-head LayerScale gamma "
            "(vision_head_scale / text_head_scale) every N training steps "
            "to track the wedge-ignition transition for the "
            "learned_query_residual head. 0 disables. No effect for "
            "head_kind=plain (no head_scale -> probe skips)."
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
        "--label-smoothing", type=float, default=None,
        help=(
            "CLIP contrastive loss label smoothing in [0, 1] (default 0.1 "
            "with --loss clip). Small values (~0.1) regularise against "
            "overconfident matches on noisy captions. Set 0.0 to match the "
            "original CLIP paper exactly. Refused with --loss siglip, which "
            "has no label smoothing."
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
        help=(
            "Smoke test on in-memory data for every stage: random "
            "image-caption pairs instead of --dataset, random CIFAR-shaped "
            "images with 100 labels instead of CIFAR-100, random-word "
            "articles instead of Wikipedia."
        ),
    )

    # Infra
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Run directory (default: results/cliffordclip_<variant>_<timestamp>).",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index for setup_gpu: counts within CUDA_VISIBLE_DEVICES when "
             "it is set, else names a physical GPU (default: all visible).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global seed for tf/keras + dataset shuffle (D-006).",
    )
    parser.add_argument(
        "--mixed-bfloat16", action="store_true",
        help=(
            "Enable bf16 mixed precision (global policy mixed_bfloat16). "
            "Default off preserves fp32 behavior. No LossScaleOptimizer is "
            "needed for bf16; logit_scale stays float32."
        ),
    )
    parser.add_argument(
        "--text-use-global-context",
        action="store_true",
        help="Enable global-context pooling in the text tower (CliffordCLIP "
             "text_use_global_context=True). Lifts the text receptive-field ceiling "
             "(doc lever G). Vision tower already uses global context; this only "
             "affects the text tower. Default off preserves prior behavior.",
    )

    return parser


#: ``--label-smoothing`` when it is not given, for ``--loss clip``.
_DEFAULT_CLIP_LABEL_SMOOTHING = 0.1


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line; resolve ``--label-smoothing`` per ``--loss``."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.loss == "siglip" and args.label_smoothing is not None:
        parser.error("--label-smoothing applies to --loss clip only")
    if args.label_smoothing is None:
        args.label_smoothing = (
            _DEFAULT_CLIP_LABEL_SMOOTHING if args.loss == "clip" else 0.0
        )
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        parser.error("--steps-per-epoch must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> Optional[Path]:
    """Entry point; returns the run directory. Ctrl-C propagates (non-zero exit)."""
    args = parse_arguments(argv)
    try:
        return train(args)
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
