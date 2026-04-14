"""CliffordUNet NLP Pre-training with Causal Language Modeling.

Combines AU-Net's hierarchical U-Net architecture (arXiv:2506.14761) with
CliffordNet's geometric algebra backbone (arXiv:2601.06793v2) for causal
language modeling.  The model processes token sequences through a
multi-stage contracting path (encoder) with causal window pooling, then
an expanding path (decoder) with multi-linear upsampling and skip
connections.

The CausalCliffordNetBlock operates on 4D tensors ``(B, H, W, D)``; at
each stage the sequence dimension shrinks via pooling (contracting) or
grows via upsampling (expanding), with increasing/decreasing channel
widths respectively.

Usage::

    # Wikipedia (default) on GPU 1
    python -m train.cliffordunet.train_cliffordunet_nlp --gpu 1 --variant nano --epochs 3

    # TFDS dataset with focal loss
    python -m train.cliffordunet.train_cliffordunet_nlp \\
        --dataset-source tfds --dataset-name imdb_reviews --loss-type focal

    # Custom architecture
    python -m train.cliffordunet.train_cliffordunet_nlp \\
        --variant custom --channels 128,192,256 --encoder-depths 4,6,4 \\
        --decoder-depths 4,2 --pool-sizes 4,4 --shifts 1,2,4

    # Resume from checkpoint
    python -m train.cliffordunet.train_cliffordunet_nlp \\
        --resume results/.../checkpoints/step_0050000.keras
"""

import os
import csv
import gc
import json
import glob
import time
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
import tiktoken
from keras import initializers, regularizers

from train.common import setup_gpu
from train.common.evaluation import generate_training_curves
from train.common.nlp import (
    load_text_dataset,
    preprocess_clm_packed_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
)
from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
    GatedGeometricResidual,
)
from dl_techniques.models.cliffordunet.lm import (
    CliffordUNetLM,
    CausalWindowPool,
    MultiLinearUpsample,
    GeometricStridePool,
    GeometricStrideUnpool,
    GGRMerge,
)
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig


_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for CliffordUNet NLP CLM pre-training."""

    # Model — Clifford-native U-Net (constant channel dim D, per-stage shifts)
    variant: str = "nano"
    channels: int = 192
    encoder_depths: List[int] = field(default_factory=lambda: [3, 4, 3])
    decoder_depths: List[int] = field(default_factory=lambda: [3, 3])
    pool_sizes: List[int] = field(default_factory=lambda: [4, 4])
    stage_shifts: List[List[int]] = field(
        default_factory=lambda: [[1, 2], [1, 2, 4], [1, 2, 4, 8]]
    )
    pool_shifts: List[int] = field(default_factory=lambda: [1, 2])
    cli_mode: str = "full"
    ctx_mode: str = "diff"
    use_global_context: bool = False
    dropout_rate: float = 0.0
    stochastic_depth_rate: float = 0.1

    # Tokenizer: raw Tiktoken gpt2 encoding (50,257 tokens including
    # <|endoftext|>=50256, used as BOS/EOS/document separator).
    # No CLS/SEP/MASK/PAD wrapping — the packed preprocessor emits
    # contiguous token chunks straight from the encoder.
    vocab_size: int = 50257
    max_seq_length: int = 512
    encoding_name: str = "gpt2"
    eot_token_id: int = 50256

    # Training
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Loss: "ce" (default) or "focal"
    loss_type: str = "ce"
    focal_gamma: float = 1.0
    label_smoothing: float = 0.0

    # Fix 2 — coarse-anchor loss masking (AU-Net style).
    # When True, only positions that are the LAST fine slot of each coarsest
    # pooling window (i.e. ``t == k * total_pool_factor - 1``) contribute to
    # the loss; all other positions are set to ``ignore_index = -1``. With
    # the strictly causal MultiLinearUpsample (fix 1) this is unnecessary
    # for correctness but is provided as an orthogonal defense-in-depth /
    # ablation knob — turn it on to verify that the model still trains when
    # the gradient signal is restricted to provably-safe positions.
    loss_anchor_only: bool = False

    # Paths
    save_dir: str = "results/cliffordunet_nlp"

    # Data source: "huggingface" or "tfds"
    dataset_source: str = "huggingface"

    # TFDS settings
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # HuggingFace / Wikipedia settings
    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    min_article_length: int = 500
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: Optional[int] = None

    # Checkpointing & analysis (step-based for large datasets)
    checkpoint_every_steps: int = 25000
    analyze_every_steps: int = 50000
    max_checkpoints: int = 3

    # Resume from checkpoint
    resume_from: Optional[str] = None

    # Generation probes
    probe_prompts: List[str] = field(default_factory=lambda: [
        "The United States of America is a",
        "In mathematics, a prime number is",
        "Albert Einstein was born in",
    ])
    probe_max_tokens: int = 100
    probe_temperature: float = 0.85
    probe_top_p: float = 0.92
    probe_repetition_penalty: float = 1.3


# ---------------------------------------------------------------------------
# Step-based Checkpoint & Analysis Callback
# ---------------------------------------------------------------------------


class StepCheckpointCallback(keras.callbacks.Callback):
    """Save model checkpoints and run analysis at fixed step intervals.

    :param save_dir: Root directory for checkpoints and analysis.
    :param save_every_steps: Checkpoint interval in training steps.
    :param analyze_every_steps: Analysis interval. 0 to disable.
    :param max_checkpoints: Keep only the N most recent checkpoints.
    :param model_name: Label for analyzer output.
    :param initial_step: Starting step count (for resume).
    """

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 25000,
        analyze_every_steps: int = 50000,
        max_checkpoints: int = 3,
        model_name: str = "cliffordunet_nlp",
        initial_step: int = 0,
        log_every_steps: int = 100,
    ):
        super().__init__()
        self.save_every_steps = save_every_steps
        self.analyze_every_steps = analyze_every_steps
        self.max_checkpoints = max_checkpoints
        self.model_name = model_name
        self._global_step = initial_step
        self._log_every_steps = log_every_steps

        self._ckpt_dir = os.path.join(save_dir, "checkpoints")
        self._analysis_dir = os.path.join(save_dir, "step_analysis")
        os.makedirs(self._ckpt_dir, exist_ok=True)
        if analyze_every_steps > 0:
            os.makedirs(self._analysis_dir, exist_ok=True)

        self._csv_path = os.path.join(save_dir, "training_log.csv")
        self._csv_file = None
        self._csv_writer = None

        self._analysis_config = AnalysisConfig(
            analyze_weights=True,
            analyze_spectral=True,
            analyze_calibration=False,
            analyze_information_flow=False,
            analyze_training_dynamics=False,
            verbose=False,
        )
        logger.info(
            f"StepCheckpointCallback: save every {save_every_steps} steps, "
            f"analyze every {analyze_every_steps} steps, "
            f"keep max {max_checkpoints} checkpoints, "
            f"log every {log_every_steps} steps"
        )

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1
        if self._global_step % self._log_every_steps == 0:
            self._log_metrics(logs)
        if self._global_step % self.save_every_steps == 0:
            self._save_checkpoint()
        if (
            self.analyze_every_steps > 0
            and self._global_step % self.analyze_every_steps == 0
        ):
            self._run_analysis()

    def on_train_end(self, logs=None):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        path = os.path.join(self._ckpt_dir, "final.keras")
        self.model.save(path)
        logger.info(f"Final checkpoint saved: {path}")

    def _log_metrics(self, logs):
        if logs is None:
            return
        row = {"step": self._global_step, **logs}
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys()),
            )
            if self._csv_file.tell() == 0:
                self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _save_checkpoint(self):
        path = os.path.join(
            self._ckpt_dir, f"step_{self._global_step:07d}.keras"
        )
        self.model.save(path)
        gc.collect()
        logger.info(f"Checkpoint saved: {path} (step {self._global_step:,})")
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        ckpts = sorted(glob.glob(
            os.path.join(self._ckpt_dir, "step_*.keras")
        ))
        while len(ckpts) > self.max_checkpoints:
            old = ckpts.pop(0)
            os.remove(old)
            logger.info(f"Removed old checkpoint: {old}")

    def _run_analysis(self):
        step_dir = os.path.join(
            self._analysis_dir, f"step_{self._global_step:07d}"
        )
        try:
            analyzer = ModelAnalyzer(
                models={self.model_name: self.model},
                config=self._analysis_config,
                output_dir=step_dir,
            )
            analyzer.analyze()
            logger.info(f"Step analysis complete: step {self._global_step:,}")
        except Exception as e:
            logger.error(
                f"Step analysis failed at step {self._global_step}: {e}"
            )


# ---------------------------------------------------------------------------
# Generation Probe Callback
# ---------------------------------------------------------------------------


class GenerationProbeCallback(keras.callbacks.Callback):
    """Generate sample text before each checkpoint to track quality.

    Uses autoregressive decoding with nucleus sampling and repetition
    penalty.  Results are saved to a JSONL file.

    :param probe_every_steps: Run probes every N steps.
    :param prompts: List of prompt strings to generate from.
    :param encoding_name: Tiktoken encoding name.
    :param max_tokens: Maximum tokens to generate per prompt.
    :param temperature: Sampling temperature.
    :param top_p: Nucleus sampling threshold.
    :param repetition_penalty: Penalty for recently generated tokens.
    :param eot_token_id: End-of-text token used for padding context
        windows and blocked from generation so probes produce
        continuous text.
    :param ctx_length: Context window length used for generation
        (typically ``max_seq_length - 1``).
    :param save_dir: Directory to save probe results.
    :param initial_step: Starting step count (for resume).
    """

    def __init__(
        self,
        probe_every_steps: int = 25000,
        prompts: Optional[List[str]] = None,
        encoding_name: str = "gpt2",
        max_tokens: int = 100,
        temperature: float = 0.85,
        top_p: float = 0.92,
        repetition_penalty: float = 1.3,
        eot_token_id: int = 50256,
        ctx_length: int = 511,
        save_dir: Optional[str] = None,
        initial_step: int = 0,
    ):
        super().__init__()
        self.probe_every_steps = probe_every_steps
        self.prompts = prompts or [
            "The United States of America is a",
            "In mathematics, a prime number is",
            "Albert Einstein was born in",
        ]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self._global_step = initial_step

        self._enc = tiktoken.get_encoding(encoding_name)
        self._eot_id = eot_token_id
        self._ctx_len = ctx_length

        self._log_path = None
        if save_dir:
            probe_dir = os.path.join(save_dir, "generation_probes")
            os.makedirs(probe_dir, exist_ok=True)
            self._log_path = os.path.join(probe_dir, "probes.jsonl")

        logger.info(
            f"GenerationProbeCallback: {len(self.prompts)} prompts, "
            f"every {probe_every_steps} steps, "
            f"max_tokens={max_tokens}, temp={temperature}, top_p={top_p}"
        )

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1
        if self._global_step % self.probe_every_steps == 0:
            self._run_probes(logs)

    def _run_probes(self, logs=None):
        step = self._global_step
        train_loss = logs.get("loss", 0.0) if logs else 0.0

        logger.info(f"{'=' * 50}")
        logger.info(
            f"Generation probe @ step {step:,} (train_loss={train_loss:.4f})"
        )
        logger.info(f"{'=' * 50}")

        probe_results = {
            "step": step,
            "train_loss": float(train_loss),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generations": [],
        }

        for prompt in self.prompts:
            t0 = time.time()
            text = self._generate(prompt)
            elapsed = time.time() - t0
            tokens_generated = (
                len(self._enc.encode(text)) - len(self._enc.encode(prompt))
            )

            gen_entry = {
                "prompt": prompt,
                "output": text[:500],
                "tokens": tokens_generated,
                "time_s": round(elapsed, 2),
                "tok_per_s": round(
                    tokens_generated / max(elapsed, 0.01), 1
                ),
            }
            probe_results["generations"].append(gen_entry)

            logger.info(f'Prompt: "{prompt}"')
            logger.info(f"Output: {text[:300]}")
            logger.info(
                f"({tokens_generated} tokens, {elapsed:.1f}s, "
                f"{gen_entry['tok_per_s']} tok/s)"
            )
            logger.info("")

        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(probe_results, ensure_ascii=False) + "\n")

    def _generate(self, prompt: str) -> str:
        """Autoregressive generation with nucleus sampling."""
        ids = self._enc.encode(prompt)
        ctx_len = self._ctx_len
        pad_id = self._eot_id

        for _ in range(self.max_tokens):
            ctx = ids[-ctx_len:]
            real = len(ctx)
            padded = ctx + [pad_id] * (ctx_len - real)
            out = self.model(
                np.array([padded], dtype="int32"), training=False,
            )
            logits = out["logits"][0, real - 1, :].numpy()

            # Block EOT so the probe produces continuous text.
            logits[self._eot_id] = -1e9

            for t in set(ids[-50:]):
                if t == self._eot_id:
                    continue
                if logits[t] >= 0:
                    logits[t] /= self.repetition_penalty
                else:
                    logits[t] *= self.repetition_penalty

            logits /= self.temperature

            sorted_idx = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx]
            probs = np.exp(sorted_logits - sorted_logits[0])
            probs /= probs.sum()
            cutoff = np.searchsorted(np.cumsum(probs), self.top_p) + 1
            top_idx = sorted_idx[:cutoff]
            top_probs = probs[:cutoff]
            top_probs /= top_probs.sum()

            next_token = top_idx[np.random.choice(len(top_idx), p=top_probs)]
            ids.append(int(next_token))

        return self._enc.decode(ids)


# ---------------------------------------------------------------------------
# Model Creation & Resume
# ---------------------------------------------------------------------------


def _extract_step_from_checkpoint(path: str) -> int:
    """Extract training step from checkpoint filename."""
    import re
    basename = os.path.basename(path)
    match = re.search(r"step_(\d+)", basename)
    return int(match.group(1)) if match else 0


def load_model_from_checkpoint(
    path: str,
) -> Tuple[CliffordUNetLM, int]:
    """Load a CliffordUNetLM from a ``.keras`` checkpoint.

    :param path: Path to the checkpoint file.
    :return: ``(model, step)`` tuple.
    """
    logger.info(f"Resuming from checkpoint: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={
            "CliffordUNetLM": CliffordUNetLM,
            "CausalCliffordNetBlock": CausalCliffordNetBlock,
            "GatedGeometricResidual": GatedGeometricResidual,
            # New Clifford-native layers used by the redesigned model:
            "GeometricStridePool": GeometricStridePool,
            "GeometricStrideUnpool": GeometricStrideUnpool,
            "GGRMerge": GGRMerge,
            # Legacy layers retained for back-compat with old checkpoints:
            "CausalWindowPool": CausalWindowPool,
            "MultiLinearUpsample": MultiLinearUpsample,
            "MaskedCausalLMLoss": MaskedCausalLMLoss,
            "FocalCausalLMLoss": FocalCausalLMLoss,
        },
    )
    step = _extract_step_from_checkpoint(path)
    logger.info(
        f"Loaded model: {model.count_params():,} params, "
        f"resumed at step {step:,}"
    )
    return model, step


def create_model(config: TrainingConfig) -> CliffordUNetLM:
    """Create and build a CliffordUNetLM from training configuration."""
    logger.info(f"Creating CliffordUNetLM-{config.variant.upper()}...")

    if config.variant in CliffordUNetLM.MODEL_VARIANTS:
        model = CliffordUNetLM.from_variant(
            config.variant,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            dropout_rate=config.dropout_rate,
        )
    else:
        model = CliffordUNetLM(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            channels=config.channels,
            encoder_depths=config.encoder_depths,
            decoder_depths=config.decoder_depths,
            pool_sizes=config.pool_sizes,
            stage_shifts=config.stage_shifts,
            pool_shifts=config.pool_shifts,
            cli_mode=config.cli_mode,
            ctx_mode=config.ctx_mode,
            use_global_context=config.use_global_context,
            dropout_rate=config.dropout_rate,
            stochastic_depth_rate=config.stochastic_depth_rate,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        )

    # Build with dummy forward pass
    dummy = np.random.randint(
        0, config.vocab_size,
        size=(1, config.max_seq_length - 1),
    ).astype("int32")
    model(dummy, training=False)

    model.summary(print_fn=logger.info)
    return model


# ---------------------------------------------------------------------------
# Loss Construction
# ---------------------------------------------------------------------------


def create_loss_fn(config: TrainingConfig) -> keras.losses.Loss:
    """Create the CLM loss function from configuration."""
    if config.loss_type == "focal":
        loss_fn = FocalCausalLMLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        logger.info(f"Loss: FocalCausalLMLoss(gamma={config.focal_gamma})")
    else:
        loss_fn = MaskedCausalLMLoss(
            label_smoothing=config.label_smoothing,
        )
        logger.info("Loss: MaskedCausalLMLoss")
    return loss_fn


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def _make_anchor_only_label_mapper(
    total_pool_factor: int,
    label_seq_len: int,
    ignore_index: int = -1,
):
    """Build a tf.data map that masks labels at non-anchor positions.

    Anchor positions are the last fine slot of each coarsest pooling
    window: ``t = total_pool_factor - 1, 2*total_pool_factor - 1, ...``.
    Non-anchor positions get their label replaced by ``ignore_index`` so
    :class:`MaskedCausalLMLoss` skips them.

    The mask is built once at graph-construction time as a constant
    boolean vector and broadcast across the batch.
    """
    if total_pool_factor < 1:
        raise ValueError(
            f"total_pool_factor must be >= 1, got {total_pool_factor}"
        )
    keep = [
        (i + 1) % total_pool_factor == 0 for i in range(label_seq_len)
    ]
    keep_const = tf.constant(keep, dtype=tf.bool)  # (label_seq_len,)

    def mask_fn(x, y):
        # y: (batch, label_seq_len) int32 labels (already may contain -1 for PAD)
        ignore = tf.cast(ignore_index, y.dtype)
        masked = tf.where(keep_const, y, ignore)
        return x, masked

    return mask_fn


def load_train_val_datasets(
    config: TrainingConfig,
    total_pool_factor: int = 1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int], Optional[int]]:
    """Load, preprocess, and wrap train/val datasets for dict-output model.

    Uses the concat-and-chunk (GPT-style) packing pipeline: each article
    is tokenized with the raw Tiktoken encoder, separated by an EOT
    token, and split into ``max_seq_length``-long contiguous chunks.
    No CLS/SEP wrapping, no article truncation — every source token is
    trained on exactly once per epoch.

    :param config: Training configuration.
    :param total_pool_factor: Product of model pool sizes — required
        when ``config.loss_anchor_only`` is True so the anchor mask
        aligns with the model's coarsest pooling stride.
    :return: ``(train_ds, val_ds, n_train_articles, n_val_articles)``.
        The article counts are the post-filter Wikipedia counts when
        the HuggingFace source is used, so the caller can compute the
        real ``steps_per_epoch`` instead of relying on a hardcoded
        estimate. For the TFDS source, both counts are ``None``.
    """
    if config.dataset_source == "tfds":
        train_ds, val_ds = _load_tfds_datasets(config)
        n_train_articles: Optional[int] = None
        n_val_articles: Optional[int] = None
    elif config.dataset_source == "huggingface":
        train_ds, val_ds, n_train_articles, n_val_articles = _load_hf_datasets(config)
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}. "
            f"Use 'tfds' or 'huggingface'."
        )

    if config.loss_anchor_only:
        label_seq_len = config.max_seq_length - 1
        anchor_mapper = _make_anchor_only_label_mapper(
            total_pool_factor=total_pool_factor,
            label_seq_len=label_seq_len,
        )
        kept = sum(
            1 for i in range(label_seq_len)
            if (i + 1) % total_pool_factor == 0
        )
        logger.info(
            f"Fix 2 enabled: loss_anchor_only stride={total_pool_factor}, "
            f"keeping {kept}/{label_seq_len} positions per sequence "
            f"(~{100.0 * kept / label_seq_len:.1f}%)"
        )
        train_ds = train_ds.map(
            anchor_mapper, num_parallel_calls=tf.data.AUTOTUNE,
        )
        val_ds = val_ds.map(
            anchor_mapper, num_parallel_calls=tf.data.AUTOTUNE,
        )

    wrap = lambda ds: ds.map(
        lambda x, y: (x, {"logits": y}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return wrap(train_ds), wrap(val_ds), n_train_articles, n_val_articles


def _load_tfds_datasets(
    config: TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_raw = load_text_dataset(
        config.dataset_name, "train", config.max_samples,
    )
    val_raw = load_text_dataset(
        config.dataset_name, "test", config.max_samples,
    )
    train = preprocess_clm_packed_dataset(
        train_raw,
        encoding_name=config.encoding_name,
        chunk_length=config.max_seq_length,
        batch_size=config.batch_size,
        eot_token_id=config.eot_token_id,
    )
    val = preprocess_clm_packed_dataset(
        val_raw,
        encoding_name=config.encoding_name,
        chunk_length=config.max_seq_length,
        batch_size=config.batch_size,
        eot_token_id=config.eot_token_id,
    )
    return train, val


def _load_hf_datasets(
    config: TrainingConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    train_raw, val_raw, n_train, n_val = load_wikipedia_train_val(
        cache_dir=config.hf_cache_dir,
        config_name=config.hf_wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        return_counts=True,
    )
    train = preprocess_clm_packed_dataset(
        train_raw,
        encoding_name=config.encoding_name,
        chunk_length=config.max_seq_length,
        batch_size=config.batch_size,
        eot_token_id=config.eot_token_id,
    )
    val = preprocess_clm_packed_dataset(
        val_raw,
        encoding_name=config.encoding_name,
        chunk_length=config.max_seq_length,
        batch_size=config.batch_size,
        eot_token_id=config.eot_token_id,
    )
    return train, val, n_train, n_val


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compile_model(
    model: CliffordUNetLM,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> None:
    """Compile with AdamW, warmup + cosine decay, and CLM loss."""
    lr_schedule = create_warmup_lr_schedule(
        config.learning_rate,
        config.num_epochs,
        steps_per_epoch,
        config.warmup_ratio,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            clipnorm=1.0,
        ),
        loss={"logits": create_loss_fn(config)},
        metrics={"logits": ["accuracy"]},
    )
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}"
    )


def _estimate_steps_per_epoch(
    config: TrainingConfig,
    n_train_articles: Optional[int] = None,
    avg_tokens_per_article: int = 450,
) -> int:
    """Estimate steps per epoch for the LR schedule.

    With packed preprocessing, one step consumes ``batch_size`` chunks
    of ``max_seq_length`` tokens each, and we emit roughly
    ``avg_tokens_per_article / max_seq_length`` chunks per article.
    The default ``avg_tokens_per_article=450`` is a conservative
    estimate for Wikipedia filtered at ``min_article_length=500`` chars
    — use ``--max-train-samples`` or the real article count (returned
    by :func:`load_train_val_datasets`) for exact numbers.

    :param config: Training configuration.
    :param n_train_articles: Exact post-filter article count. When
        provided, the estimate uses this instead of the hardcoded
        Wikipedia default.
    :param avg_tokens_per_article: Mean token count per retained
        article. Only used for the Wikipedia fallback.
    """
    if config.max_train_samples:
        articles = config.max_train_samples
    elif n_train_articles is not None:
        articles = n_train_articles
    else:
        articles = 4_000_000
    chunks_per_article = max(1, avg_tokens_per_article // config.max_seq_length)
    total_chunks = articles * chunks_per_article
    return max(1, total_chunks // config.batch_size)


def train_cliffordunet_nlp(
    config: TrainingConfig,
) -> Tuple[CliffordUNetLM, keras.callbacks.History]:
    """Run CliffordUNet NLP CLM pre-training.

    :param config: Training configuration.
    :return: Trained model and training history.
    """
    logger.info("=" * 60)
    logger.info("CliffordUNet NLP Causal LM Pre-training")
    logger.info("=" * 60)

    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)

    # Model first — we need its total_pool_factor to build the (optional)
    # fix-2 anchor loss mask in the data pipeline.
    initial_step = 0

    if config.resume_from:
        model, initial_step = load_model_from_checkpoint(config.resume_from)
    else:
        model = create_model(config)

    total_pool_factor = getattr(model, "total_pool_factor", 1)
    if config.loss_anchor_only and total_pool_factor < 2:
        logger.warning(
            "loss_anchor_only=True but total_pool_factor < 2; "
            "no positions will be masked out."
        )

    # Data (must come after model so we know the pool factor). The
    # HuggingFace path returns the exact post-filter article count, so
    # the LR schedule can be sized against the real step count instead
    # of a hardcoded estimate.
    train_dataset, val_dataset, n_train_articles, n_val_articles = (
        load_train_val_datasets(
            config, total_pool_factor=total_pool_factor,
        )
    )

    steps_per_epoch = _estimate_steps_per_epoch(
        config, n_train_articles=n_train_articles,
    )
    validation_steps: Optional[int] = None
    if n_val_articles is not None:
        val_chunks_per_article = max(1, 450 // config.max_seq_length)
        validation_steps = max(
            1, (n_val_articles * val_chunks_per_article) // config.batch_size,
        )

    compile_model(model, config, steps_per_epoch)

    # Callbacks
    variant_label = config.variant
    if config.variant == "custom":
        variant_label = f"c{config.channels}"

    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"CliffordUNetLM-{variant_label}",
        results_dir_prefix="cliffordunet_nlp",
        include_analyzer=False,
    )
    callbacks = [
        cb for cb in callbacks
        if not isinstance(cb, keras.callbacks.CSVLogger)
    ]
    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=config.checkpoint_every_steps,
        analyze_every_steps=config.analyze_every_steps,
        max_checkpoints=config.max_checkpoints,
        model_name=f"CliffordUNetLM-{variant_label}",
        initial_step=initial_step,
    ))

    callbacks.append(GenerationProbeCallback(
        probe_every_steps=config.checkpoint_every_steps,
        prompts=config.probe_prompts,
        encoding_name=config.encoding_name,
        max_tokens=config.probe_max_tokens,
        temperature=config.probe_temperature,
        top_p=config.probe_top_p,
        repetition_penalty=config.probe_repetition_penalty,
        eot_token_id=config.eot_token_id,
        ctx_length=config.max_seq_length - 1,
        save_dir=results_dir,
        initial_step=initial_step,
    ))

    # Train — pass the real steps_per_epoch / validation_steps so the
    # warmup+cosine LR schedule horizon and the fit epoch boundary are
    # driven by the same quantity.
    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch={steps_per_epoch:,}, "
        f"validation_steps={validation_steps}, "
        f"batch_size={config.batch_size}"
    )
    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        verbose=1,
    )
    logger.info("Training completed!")

    generate_training_curves(history, results_dir)

    if "val_loss" in history.history:
        best_epoch = tf.argmin(history.history["val_loss"]).numpy()
        logger.info(
            f"Best epoch: {best_epoch + 1} "
            f"(val_loss: {history.history['val_loss'][best_epoch]:.4f})"
        )

    return model, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CliffordUNet NLP Causal LM Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hardware
    p.add_argument("--gpu", type=int, default=None, help="GPU device index")

    # Model
    p.add_argument(
        "--variant", type=str, default="nano",
        choices=list(CliffordUNetLM.MODEL_VARIANTS.keys()) + ["custom"],
        help="Model variant (nano/mini/base/custom)",
    )
    p.add_argument("--channels", type=int, default=192,
                    help="Constant feature dim D (custom variant)")
    p.add_argument("--encoder-depths", type=str, default="3,4,3",
                    help="Comma-separated encoder blocks per stage (custom)")
    p.add_argument("--decoder-depths", type=str, default="3,3",
                    help="Comma-separated decoder blocks per stage (custom)")
    p.add_argument("--pool-sizes", type=str, default="4,4",
                    help="Comma-separated pool factors between stages (custom)")
    p.add_argument(
        "--stage-shifts", type=str,
        default="1,2;1,2,4;1,2,4,8",
        help=(
            "Per-stage channel shifts, stages separated by ';' and shifts "
            "within a stage by ','. Length must equal encoder_depths length."
        ),
    )
    p.add_argument(
        "--pool-shifts", type=str, default="1,2",
        help="Comma-separated channel shifts inside pool/unpool layers.",
    )
    p.add_argument(
        "--cli-mode", type=str, default="full",
        choices=["inner", "wedge", "full"],
        help="Clifford algebra components",
    )
    p.add_argument(
        "--ctx-mode", type=str, default="diff",
        choices=["diff", "abs"],
        help="Context calculation mode",
    )
    p.add_argument("--use-global-context", action="store_true",
                    help="Enable global context branch")
    p.add_argument("--stochastic-depth-rate", type=float, default=0.1,
                    help="Maximum stochastic depth rate")

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--dropout-rate", type=float, default=0.1)

    # Loss
    p.add_argument(
        "--loss-type", type=str, default="ce",
        choices=["ce", "focal"],
        help="'ce' (MaskedCausalLMLoss) or 'focal' (FocalCausalLMLoss)",
    )
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument(
        "--loss-anchor-only", action="store_true",
        help=(
            "Fix 2: only score loss at the last fine slot of each "
            "coarsest pooling window. Provably-causal positions even "
            "without the strictly causal upsample. Default: off."
        ),
    )

    # Data source
    p.add_argument(
        "--dataset-source", type=str, default="huggingface",
        choices=["tfds", "huggingface"],
    )
    p.add_argument("--dataset-name", type=str, default="imdb_reviews",
                    help="TFDS dataset name")
    p.add_argument("--max-samples", type=int, default=None,
                    help="TFDS max samples")
    p.add_argument("--hf-cache-dir", type=str,
                    default="/media/arxwn/data0_4tb/datasets/wikipedia")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.02)

    # Checkpointing
    p.add_argument("--checkpoint-every-steps", type=int, default=25000)
    p.add_argument("--analyze-every-steps", type=int, default=50000,
                    help="0 to disable")
    p.add_argument("--max-checkpoints", type=int, default=3)

    # Resume
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to .keras checkpoint to resume training from",
    )

    # Output
    p.add_argument("--save-dir", type=str, default="results/cliffordunet_nlp")

    return p


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_int_list_list(s: str) -> List[List[int]]:
    """Parse ``"1,2;1,2,4;1,2,4,8"`` into ``[[1,2],[1,2,4],[1,2,4,8]]``."""
    return [_parse_int_list(part) for part in s.split(";") if part.strip()]


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        variant=args.variant,
        channels=args.channels,
        encoder_depths=_parse_int_list(args.encoder_depths),
        decoder_depths=_parse_int_list(args.decoder_depths),
        pool_sizes=_parse_int_list(args.pool_sizes),
        stage_shifts=_parse_int_list_list(args.stage_shifts),
        pool_shifts=_parse_int_list(args.pool_shifts),
        cli_mode=args.cli_mode,
        ctx_mode=args.ctx_mode,
        use_global_context=args.use_global_context,
        stochastic_depth_rate=args.stochastic_depth_rate,
        dropout_rate=args.dropout_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        loss_anchor_only=args.loss_anchor_only,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        hf_cache_dir=args.hf_cache_dir,
        max_train_samples=args.max_train_samples,
        val_fraction=args.val_fraction,
        checkpoint_every_steps=args.checkpoint_every_steps,
        analyze_every_steps=args.analyze_every_steps,
        max_checkpoints=args.max_checkpoints,
        resume_from=args.resume,
        save_dir=args.save_dir,
    )


def main() -> None:
    """Main entry point for CliffordUNet NLP CLM pre-training."""
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.variant}, "
        f"D={config.channels}, "
        f"enc_depths={config.encoder_depths}, "
        f"dec_depths={config.decoder_depths}, "
        f"pool_sizes={config.pool_sizes}, "
        f"stage_shifts={config.stage_shifts}, "
        f"pool_shifts={config.pool_shifts}, cli_mode={config.cli_mode}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, loss={config.loss_type}, "
        f"source={config.dataset_source}"
    )

    train_cliffordunet_nlp(config)


if __name__ == "__main__":
    main()
