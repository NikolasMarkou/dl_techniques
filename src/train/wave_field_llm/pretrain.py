"""WaveFieldLLM Pre-training Script with Causal Language Modeling.

Pre-trains a WaveFieldLLM decoder on a text dataset using next-token
prediction (causal LM). Mirrors :mod:`train.gpt2.pretrain` so the only
training-side difference between GPT-2 and WaveFieldLLM is the model class
and an optional ``--field-size`` hyperparameter.

Usage::

    # TFDS smoke run on GPU 1
    python -m train.wave_field_llm.pretrain --gpu 1 --variant tiny \
        --dataset-source tfds --dataset-name imdb_reviews --max-samples 64 \
        --epochs 1 --batch-size 2 --max-seq-length 32

    # Wikipedia full pre-training (GPU 0)
    python -m train.wave_field_llm.pretrain --gpu 0 --variant small --epochs 3

    # Resume from checkpoint
    python -m train.wave_field_llm.pretrain --resume results/.../checkpoints/step_0050000.keras
"""

import csv
import os
import json
import glob
import time
import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

import keras
import numpy as np
import tensorflow as tf
import tiktoken

from train.common import setup_gpu
from train.common.evaluation import generate_training_curves
from train.common import plot_step_metrics
from train.common.nlp import (
    create_tokenizer,
    load_text_dataset,
    preprocess_clm_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
    estimate_clm_steps_per_epoch,
    build_clm_metrics,
    augment_probe_results,
)
from dl_techniques.models.wave_field_llm.wave_field_llm import (
    WaveFieldLLM,
    WaveFieldDecoderBlock,
)
from dl_techniques.layers.attention.wave_field_attention import (
    WaveFieldAttention,
    _IdentityPlusNoise,
)
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for WaveFieldLLM CLM pre-training."""

    # Model
    model_variant: str = "small"
    vocab_size: int = 50261
    max_seq_length: int = 512
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    field_size: Optional[int] = None  # None -> 2 * max_seq_length per variant
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    tie_word_embeddings: bool = True

    # Tokenizer (Tiktoken gpt2 encoding — 50,257 base + 4 special)
    encoding_name: str = "gpt2"
    cls_token_id: int = 50257
    sep_token_id: int = 50258
    pad_token_id: int = 50259
    mask_token_id: int = 50260

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

    # Paths
    save_dir: str = "results/wave_field_llm_pretrain"

    # Data source: "huggingface" or "tfds"
    dataset_source: str = "huggingface"

    # TFDS settings
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # HuggingFace / Wikipedia settings
    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    min_article_length: int = 0
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: Optional[int] = None
    shuffle_shards: int = 4

    # Checkpointing & analysis (step-based for large datasets)
    checkpoint_every_steps: int = 25000
    analyze_every_steps: int = 50000
    max_checkpoints: int = 3
    steps_per_epoch: Optional[int] = None

    # Resume from checkpoint
    resume_from: Optional[str] = None
    seed: int = 42

    # Generation probes (run before each checkpoint)
    probe_prompts: List[str] = field(default_factory=lambda: [
        "The United States of America is a",
        "In mathematics, a prime number is",
        "Albert Einstein was born in",
    ])
    probe_max_tokens: int = 100
    probe_temperature: float = 0.85
    probe_top_p: float = 0.92
    probe_repetition_penalty: float = 1.3


# ---------------------------------------------------------------------
# Step-based Checkpoint & Analysis Callback
# ---------------------------------------------------------------------


class StepCheckpointCallback(keras.callbacks.Callback):
    """Save checkpoints + run weight/spectral analysis at fixed step intervals."""

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 25000,
        analyze_every_steps: int = 50000,
        max_checkpoints: int = 3,
        model_name: str = "wave_field_llm",
        initial_step: int = 0,
        log_every_steps: int = 100,
        plot_every_steps: int = 25000,
    ):
        super().__init__()
        self.save_every_steps = save_every_steps
        self.analyze_every_steps = analyze_every_steps
        self.max_checkpoints = max_checkpoints
        self.model_name = model_name
        self._global_step = initial_step
        self._log_every_steps = log_every_steps
        self._plot_every_steps = plot_every_steps
        self._save_dir = save_dir

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
            f"log every {log_every_steps} steps, "
            f"plot every {plot_every_steps} steps"
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
        if (
            self._plot_every_steps > 0
            and self._global_step % self._plot_every_steps == 0
        ):
            self._plot_metrics()

    def on_train_end(self, logs=None):
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        path = os.path.join(self._ckpt_dir, "final.keras")
        self.model.save(path)
        logger.info(f"Final checkpoint saved: {path}")
        self._plot_metrics()

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

    def _plot_metrics(self):
        try:
            plot_step_metrics(self._csv_path, self._save_dir)
        except Exception as e:
            logger.warning(f"Step plot failed at step {self._global_step}: {e}")

    def _save_checkpoint(self):
        path = os.path.join(
            self._ckpt_dir, f"step_{self._global_step:07d}.keras"
        )
        self.model.save(path)
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


# ---------------------------------------------------------------------
# Generation Probe Callback
# ---------------------------------------------------------------------


class GenerationProbeCallback(keras.callbacks.Callback):
    """Generate sample text and log metrics before each checkpoint."""

    def __init__(
        self,
        probe_every_steps: int = 25000,
        prompts: Optional[List[str]] = None,
        encoding_name: str = "gpt2",
        max_tokens: int = 100,
        temperature: float = 0.85,
        top_p: float = 0.92,
        repetition_penalty: float = 1.3,
        eot_token_id: Optional[int] = None,
        save_dir: Optional[str] = None,
        initial_step: int = 0,
        context_window: int = 511,
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
        self._context_window = context_window

        self._enc = tiktoken.get_encoding(encoding_name)
        self._eot_id = int(eot_token_id if eot_token_id is not None else self._enc.eot_token)

        self._probe_log = []
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
            tokens_generated = len(self._enc.encode(text)) - len(self._enc.encode(prompt))

            gen_entry = {
                "prompt": prompt,
                "output": text[:500],
                "tokens": tokens_generated,
                "time_s": round(elapsed, 2),
                "tok_per_s": round(tokens_generated / max(elapsed, 0.01), 1),
            }
            probe_results["generations"].append(gen_entry)

            logger.info(f'Prompt: "{prompt}"')
            logger.info(f"Output: {text[:300]}")
            logger.info(
                f"({tokens_generated} tokens, {elapsed:.1f}s, "
                f"{gen_entry['tok_per_s']} tok/s)"
            )
            logger.info("")

        self._post_generate_hook(probe_results)

        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(probe_results, ensure_ascii=False) + "\n")

    def _generate(self, prompt: str) -> str:
        ids = self._enc.encode(prompt)
        # Block special tokens from sampling: tiktoken's `decode` raises on
        # any id at or above the encoder's `n_vocab` (the base vocab end)
        # because the 4 reserved special-token ids in this codebase
        # (50257..50260) live outside the BPE table. Suppress them at
        # logits time so generation is restricted to decodable tokens.
        special_ids = [
            i for i in range(self._enc.n_vocab, max(self._enc.n_vocab + 1, 50261))
        ]

        for _ in range(self.max_tokens):
            ctx = ids[-self._context_window:]
            out = self.model(
                np.array([ctx], dtype="int32"), training=False,
            )
            logits = out["logits"][0, -1, :].numpy()

            logits[self._eot_id] = -1e9
            for sid in special_ids:
                if sid < logits.shape[0]:
                    logits[sid] = -1e9
            for t in set(ids[-50:]):
                if t == self._eot_id:
                    continue
                logits[t] /= self.repetition_penalty

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

        # `errors="replace"` is a defensive backstop: if a tokenizer surprise
        # (e.g. partial multi-byte BPE chunk at the tail) sneaks through,
        # we want a string back, not a probe-side crash that kills training.
        try:
            return self._enc.decode(ids)
        except (KeyError, UnicodeDecodeError) as e:
            logger.warning(f"Probe decode fell back due to: {e}")
            return self._enc.decode(
                [t for t in ids if t < self._enc.n_vocab],
            )

    def _post_generate_hook(self, results: dict) -> None:
        pass


# ---------------------------------------------------------------------
# Model creation & resume
# ---------------------------------------------------------------------


def _extract_step_from_checkpoint(path: str) -> int:
    import re
    basename = os.path.basename(path)
    match = re.search(r"step_(\d+)", basename)
    if match:
        return int(match.group(1))
    return 0


def load_model_from_checkpoint(path: str) -> Tuple[WaveFieldLLM, int]:
    """Load a WaveFieldLLM model from a ``.keras`` checkpoint."""
    logger.info(f"Resuming from checkpoint: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={
            "MaskedCausalLMLoss": MaskedCausalLMLoss,
            "FocalCausalLMLoss": FocalCausalLMLoss,
            # WaveFieldAttention + _IdentityPlusNoise auto-register via
            # @register_keras_serializable, but listing them defensively
            # protects against import-order surprises.
            "WaveFieldLLM": WaveFieldLLM,
            "WaveFieldDecoderBlock": WaveFieldDecoderBlock,
            "WaveFieldAttention": WaveFieldAttention,
            "_IdentityPlusNoise": _IdentityPlusNoise,
        },
    )
    step = _extract_step_from_checkpoint(path)
    logger.info(
        f"Loaded model: {model.count_params():,} params, "
        f"resumed at step {step:,}"
    )
    return model, step


def create_wave_field_llm_model(config: TrainingConfig) -> WaveFieldLLM:
    """Create and build a WaveFieldLLM model from the training configuration."""
    logger.info(f"Creating WaveFieldLLM-{config.model_variant.upper()}...")

    variant_kwargs = dict(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_length,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        tie_word_embeddings=config.tie_word_embeddings,
    )
    if config.num_layers is not None:
        variant_kwargs["depth"] = config.num_layers
    if config.num_heads is not None:
        variant_kwargs["num_heads"] = config.num_heads
    if config.field_size is not None:
        variant_kwargs["field_size"] = config.field_size

    model = WaveFieldLLM.from_variant(config.model_variant, **variant_kwargs)

    # Build with a dummy forward pass to initialize weights.
    dummy = np.random.randint(
        0, config.vocab_size,
        size=(1, max(1, config.max_seq_length - 1)),
    ).astype("int32")
    model(dummy, training=False)

    logger.info(f"WaveFieldLLM model: {model.count_params():,} parameters")
    return model


# ---------------------------------------------------------------------
# Loss construction
# ---------------------------------------------------------------------


def create_loss_fn(config: TrainingConfig) -> keras.losses.Loss:
    if config.loss_type == "focal":
        loss_fn = FocalCausalLMLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        logger.info(f"Loss: FocalCausalLMLoss(gamma={config.focal_gamma})")
    else:
        loss_fn = MaskedCausalLMLoss(label_smoothing=config.label_smoothing)
        logger.info("Loss: MaskedCausalLMLoss")
    return loss_fn


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_train_val_datasets(
    config: TrainingConfig,
    preprocessor,
    data_seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int]]:
    n_train_articles: Optional[int] = None
    if config.dataset_source == "tfds":
        train_ds, val_ds = _load_tfds_datasets(config, preprocessor)
    elif config.dataset_source == "huggingface":
        train_ds, val_ds, n_train_articles = _load_hf_datasets(
            config, preprocessor, data_seed,
        )
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}. "
            f"Use 'tfds' or 'huggingface'."
        )

    # Wrap labels for dict-output model: (x, y) -> (x, {"logits": y})
    wrap = lambda ds: ds.map(
        lambda x, y: (x, {"logits": y}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return wrap(train_ds), wrap(val_ds), n_train_articles


def _load_tfds_datasets(config, preprocessor):
    train = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    return train, val


def _load_hf_datasets(config, preprocessor, data_seed: int):
    train_raw, val_raw, n_train, _n_val = load_wikipedia_train_val(
        cache_dir=config.hf_cache_dir,
        config_name=config.hf_wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        seed=data_seed,
        num_shards=config.shuffle_shards,
        return_counts=True,
    )
    train = preprocess_clm_dataset(
        train_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        val_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    return train, val, n_train


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def compile_model(
    model: WaveFieldLLM,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> None:
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
        metrics={"logits": build_clm_metrics(config.encoding_name)},
    )
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}"
    )


def _make_steps_per_epoch(
    config: TrainingConfig, n_train_articles: Optional[int],
) -> int:
    if (
        config.dataset_source == "tfds"
        and config.max_samples
        and config.steps_per_epoch is None
    ):
        return max(1, config.max_samples // config.batch_size)
    return estimate_clm_steps_per_epoch(
        num_articles=n_train_articles or config.max_train_samples,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        override=config.steps_per_epoch,
    )


def train_wave_field_llm(
    config: TrainingConfig,
    model_factory: Callable[[TrainingConfig], WaveFieldLLM] = create_wave_field_llm_model,
) -> Tuple[WaveFieldLLM, keras.callbacks.History]:
    """Run WaveFieldLLM CLM pre-training."""
    logger.info("=" * 60)
    logger.info("WaveFieldLLM Causal LM Pre-training")
    logger.info("=" * 60)

    tf.random.set_seed(config.seed)
    keras.utils.set_random_seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    preprocessor = create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    )

    initial_step = (
        _extract_step_from_checkpoint(config.resume_from)
        if config.resume_from else 0
    )
    data_seed = config.seed + initial_step

    train_dataset, val_dataset, n_train_articles = load_train_val_datasets(
        config, preprocessor, data_seed=data_seed,
    )

    steps_per_epoch = _make_steps_per_epoch(config, n_train_articles)

    if config.resume_from:
        model, initial_step = load_model_from_checkpoint(config.resume_from)
    else:
        model = model_factory(config)

    compile_model(model, config, steps_per_epoch)

    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"WaveFieldLLM-{config.model_variant}",
        results_dir_prefix="wave_field_llm_pretrain",
        include_analyzer=False,
    )
    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=config.checkpoint_every_steps,
        analyze_every_steps=config.analyze_every_steps,
        max_checkpoints=config.max_checkpoints,
        model_name=f"WaveFieldLLM-{config.model_variant}",
        initial_step=initial_step,
    ))

    # Generation probe context window: model's max - 1 keeps room for the
    # next token. For the smoke variant (max_seq_len=32) this means 31.
    probe_ctx = max(1, config.max_seq_length - 1)
    probe_cb = GenerationProbeCallback(
        probe_every_steps=config.checkpoint_every_steps,
        prompts=config.probe_prompts,
        encoding_name=config.encoding_name,
        max_tokens=config.probe_max_tokens,
        temperature=config.probe_temperature,
        top_p=config.probe_top_p,
        repetition_penalty=config.probe_repetition_penalty,
        save_dir=results_dir,
        initial_step=initial_step,
        context_window=probe_ctx,
    )
    probe_cb._post_generate_hook = augment_probe_results
    callbacks.append(probe_cb)

    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch~{steps_per_epoch:,}, "
        f"batch_size={config.batch_size}"
    )
    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WaveFieldLLM Causal LM Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hardware
    p.add_argument("--gpu", type=int, default=None, help="GPU device index")

    # Model
    p.add_argument(
        "--variant", type=str, default="small",
        choices=list(WaveFieldLLM.MODEL_VARIANTS.keys()),
        help="WaveFieldLLM model variant",
    )
    p.add_argument("--num-layers", type=int, default=None,
                    help="Override number of decoder blocks")
    p.add_argument("--num-heads", type=int, default=None,
                    help="Override number of attention heads")
    p.add_argument(
        "--field-size", type=int, default=None,
        help="Override wave field grid resolution (default: variant-defined "
             "or 2 * max_seq_length).",
    )
    p.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tie LM head to token embeddings",
    )

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)

    # Loss
    p.add_argument(
        "--loss-type", type=str, default="ce",
        choices=["ce", "focal"],
    )
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # Data source
    p.add_argument(
        "--dataset-source", type=str, default="huggingface",
        choices=["tfds", "huggingface"],
    )
    p.add_argument("--dataset-name", type=str, default="imdb_reviews")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--hf-cache-dir", type=str,
                    default="/media/arxwn/data0_4tb/datasets/wikipedia")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument(
        "--min-article-length", type=int, default=0,
        help="HF Wikipedia char-length filter. 0 = no filter (recommended "
             "for packed CLM).",
    )
    p.add_argument(
        "--shuffle-shards", type=int, default=4,
        help="HF Wikipedia parallel tokenization shards. 1 = single-thread, "
             "deterministic.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global seed. On --resume, data seed is shifted by initial_step.",
    )

    # Checkpointing
    p.add_argument("--checkpoint-every-steps", type=int, default=25000)
    p.add_argument("--analyze-every-steps", type=int, default=50000,
                    help="0 to disable")
    p.add_argument("--max-checkpoints", type=int, default=3)
    p.add_argument(
        "--steps-per-epoch", type=int, default=None,
        help="Override LR-schedule horizon.",
    )

    # Resume
    p.add_argument("--resume", type=str, default=None,
                    help="Path to .keras checkpoint to resume from")

    # Output
    p.add_argument("--save-dir", type=str,
                    default="results/wave_field_llm_pretrain")

    return p


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        model_variant=args.variant,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        field_size=args.field_size,
        tie_word_embeddings=args.tie_word_embeddings,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        hf_cache_dir=args.hf_cache_dir,
        max_train_samples=args.max_train_samples,
        val_fraction=args.val_fraction,
        min_article_length=args.min_article_length,
        shuffle_shards=args.shuffle_shards,
        seed=args.seed,
        steps_per_epoch=args.steps_per_epoch,
        checkpoint_every_steps=args.checkpoint_every_steps,
        analyze_every_steps=args.analyze_every_steps,
        max_checkpoints=args.max_checkpoints,
        resume_from=args.resume,
        save_dir=args.save_dir,
    )


def main() -> None:
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.model_variant}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, loss={config.loss_type}, "
        f"source={config.dataset_source}, "
        f"field_size={config.field_size}"
    )

    train_wave_field_llm(config)


if __name__ == "__main__":
    main()
