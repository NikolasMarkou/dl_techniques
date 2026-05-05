"""CliffordNet NLP Pre-training with Hierarchical Routing Head (CLM).

Sibling of ``train_cliffordnet_nlp.py`` that trains
:class:`CliffordNetLMRouting` — a CliffordNet causal LM whose final vocab
projection is a :class:`RoutingProbabilitiesLayer` (probabilistic
hierarchical tree) instead of a Dense softmax.

Two routing modes are exposed via ``--routing-mode``:

- ``trainable`` (default): learnable affine projection to log2(padded_vocab)
  decisions. ~3000x fewer params than the Dense head at vocab=50K.
- ``deterministic``: parameter-free cosine-basis projection. Useful for
  ablation; expressive ceiling of 16 decisions for ~50K vocab is tight.

Because the routing layer outputs probabilities (not logits), losses are
constructed with ``from_logits=False``.

Usage::

    # Trainable routing on imdb (smoke test)
    MPLBACKEND=Agg python -m train.cliffordnet.train_cliffordnet_nlp_routing \\
        --variant nano --routing-mode trainable \\
        --dataset-source tfds --dataset-name imdb_reviews \\
        --max-samples 100 --epochs 1 --batch-size 4 --gpu 0

    # Deterministic ablation
    MPLBACKEND=Agg python -m train.cliffordnet.train_cliffordnet_nlp_routing \\
        --variant nano --routing-mode deterministic ...

Resume from checkpoint::

    python -m train.cliffordnet.train_cliffordnet_nlp_routing \\
        --resume results/cliffordnet_nlp_routing_*/checkpoints/step_0050000.keras
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
    create_tokenizer,
    load_text_dataset,
    preprocess_clm_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
)
from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
)
from dl_techniques.layers.activations.routing_probabilities import (
    RoutingProbabilitiesLayer,
)
from dl_techniques.models.cliffordnet import CliffordNetLMRouting
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
    """Configuration for CliffordNet routing-head NLP CLM pre-training."""

    # Model
    variant: str = "nano"
    channels: int = 128
    depth: int = 12
    shifts: List[int] = field(default_factory=lambda: [1, 2])
    cli_mode: str = "full"
    ctx_mode: str = "diff"
    use_global_context: bool = False
    dropout_rate: float = 0.1
    stochastic_depth_rate: float = 0.1

    # Routing head
    routing_mode: str = "trainable"

    # Tokenizer (Tiktoken gpt2 encoding -- 50,257 base + 4 special)
    vocab_size: int = 50261
    max_seq_length: int = 512
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
    save_dir: str = "results/cliffordnet_nlp_routing"

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
    """Save model checkpoints and run analysis at fixed step intervals."""

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 25000,
        analyze_every_steps: int = 50000,
        max_checkpoints: int = 3,
        model_name: str = "cliffordnet_nlp_routing",
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

    Note: model output is *probabilities* (sum=1, in [eps, 1-eps]) rather
    than logits, so the generation logic uses log-probs for nucleus
    sampling. Functionally identical to the lm.py probe except for the
    log() conversion.
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
        eot_token_id: Optional[int] = None,
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
        self._eot_id = int(eot_token_id if eot_token_id is not None else self._enc.eot_token)
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

        # Reclaim Python wrappers + tf-eager intermediates accumulated by
        # the 300 autoregressive `model(...)` calls; without this, ~70 MB
        # per probe event leaked into RSS over the run.
        gc.collect()

    def _generate(self, prompt: str) -> str:
        """Autoregressive generation. Model outputs probabilities, so we
        take ``log(p)`` and treat it as logits for the standard pipeline."""
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
            probs = out["logits"][0, real - 1, :].numpy()
            # Convert probabilities to log-probs (logit-equivalent).
            logits = np.log(np.clip(probs, 1e-12, 1.0))

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
            probs_norm = np.exp(sorted_logits - sorted_logits[0])
            probs_norm /= probs_norm.sum()
            cutoff = np.searchsorted(np.cumsum(probs_norm), self.top_p) + 1
            top_idx = sorted_idx[:cutoff]
            top_probs = probs_norm[:cutoff]
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
) -> Tuple[CliffordNetLMRouting, int]:
    """Load a CliffordNetLMRouting from a ``.keras`` checkpoint."""
    logger.info(f"Resuming from checkpoint: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={
            "CliffordNetLMRouting": CliffordNetLMRouting,
            "CausalCliffordNetBlock": CausalCliffordNetBlock,
            "RoutingProbabilitiesLayer": RoutingProbabilitiesLayer,
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


def create_model(config: TrainingConfig) -> CliffordNetLMRouting:
    """Create and build a CliffordNetLMRouting from training configuration."""
    logger.info(
        f"Creating CliffordNetLMRouting-{config.variant.upper()} "
        f"(routing_mode={config.routing_mode})..."
    )

    if config.variant in CliffordNetLMRouting.MODEL_VARIANTS:
        model = CliffordNetLMRouting.from_variant(
            config.variant,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            routing_mode=config.routing_mode,
            dropout_rate=config.dropout_rate,
        )
    else:
        # Custom variant
        model = CliffordNetLMRouting(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            channels=config.channels,
            depth=config.depth,
            shifts=config.shifts,
            cli_mode=config.cli_mode,
            ctx_mode=config.ctx_mode,
            use_global_context=config.use_global_context,
            dropout_rate=config.dropout_rate,
            stochastic_depth_rate=config.stochastic_depth_rate,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
            routing_mode=config.routing_mode,
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
    """Create the CLM loss function from configuration.

    DECISION D-003: ``from_logits=False`` is required because
    :class:`RoutingProbabilitiesLayer` outputs probabilities in
    ``[eps, 1-eps]`` summing to 1, not raw logits. Passing
    ``from_logits=True`` (the default) would softmax already-softmaxed
    values, producing an incorrect loss.
    """
    if config.loss_type == "focal":
        loss_fn = FocalCausalLMLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            from_logits=False,
        )
        logger.info(
            f"Loss: FocalCausalLMLoss(gamma={config.focal_gamma}, "
            f"from_logits=False)"
        )
    else:
        loss_fn = MaskedCausalLMLoss(
            label_smoothing=config.label_smoothing,
            from_logits=False,
        )
        logger.info("Loss: MaskedCausalLMLoss(from_logits=False)")
    return loss_fn


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_train_val_datasets(
    config: TrainingConfig,
    preprocessor,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load, preprocess, and wrap train/val datasets for dict-output model."""
    if config.dataset_source == "tfds":
        train_ds, val_ds = _load_tfds_datasets(config, preprocessor)
    elif config.dataset_source == "huggingface":
        train_ds, val_ds = _load_hf_datasets(config, preprocessor)
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}. "
            f"Use 'tfds' or 'huggingface'."
        )

    # Wrap labels for dict-output model: (x, y) -> (x, {"logits": y})
    # Note: dict key remains "logits" even though values are probabilities;
    # see DECISION D-001 in lm_routing.py.
    wrap = lambda ds: ds.map(
        lambda x, y: (x, {"logits": y}),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return wrap(train_ds), wrap(val_ds)


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


def _load_hf_datasets(config, preprocessor):
    train_raw, val_raw = load_wikipedia_train_val(
        cache_dir=config.hf_cache_dir,
        config_name=config.hf_wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
    )
    train = preprocess_clm_dataset(
        train_raw, preprocessor,
        config.max_seq_length, config.batch_size,
        streaming=True,
    )
    val = preprocess_clm_dataset(
        val_raw, preprocessor,
        config.max_seq_length, config.batch_size,
        streaming=True,
    )
    return train, val


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compile_model(
    model: CliffordNetLMRouting,
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


def _estimate_steps_per_epoch(config: TrainingConfig) -> int:
    """Estimate steps per epoch for LR schedule."""
    if config.dataset_source == "tfds" and config.max_samples:
        return max(1, config.max_samples // config.batch_size)
    if config.max_train_samples:
        return config.max_train_samples // config.batch_size
    return 4_850_000 // config.batch_size


def train_cliffordnet_nlp_routing(
    config: TrainingConfig,
) -> Tuple[CliffordNetLMRouting, keras.callbacks.History]:
    """Run CliffordNet routing-head NLP CLM pre-training."""
    logger.info("=" * 60)
    logger.info("CliffordNet NLP Routing-head Causal LM Pre-training")
    logger.info("=" * 60)

    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)

    preprocessor = create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    )

    train_dataset, val_dataset = load_train_val_datasets(
        config, preprocessor,
    )

    steps_per_epoch = _estimate_steps_per_epoch(config)
    initial_step = 0

    if config.resume_from:
        model, initial_step = load_model_from_checkpoint(config.resume_from)
    else:
        model = create_model(config)

    compile_model(model, config, steps_per_epoch)

    variant_label = config.variant
    if config.variant == "custom":
        variant_label = f"c{config.channels}d{config.depth}"

    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"CliffordNetLMRouting-{variant_label}-{config.routing_mode}",
        results_dir_prefix="cliffordnet_nlp_routing",
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
        model_name=f"CliffordNetLMRouting-{variant_label}-{config.routing_mode}",
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
        ctx_length=config.max_seq_length - 1,
        save_dir=results_dir,
        initial_step=initial_step,
    ))

    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch~={steps_per_epoch:,}, "
        f"batch_size={config.batch_size}, "
        f"routing_mode={config.routing_mode}"
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CliffordNet routing-head NLP Causal LM Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--gpu", type=int, default=None, help="GPU device index")

    p.add_argument(
        "--variant", type=str, default="nano",
        choices=list(CliffordNetLMRouting.MODEL_VARIANTS.keys()) + ["custom"],
        help="Model variant (nano/mini/base/large/xl/custom)",
    )
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--shifts", type=str, default="1,2")
    p.add_argument(
        "--cli-mode", type=str, default="full",
        choices=["inner", "wedge", "full"],
    )
    p.add_argument(
        "--ctx-mode", type=str, default="diff",
        choices=["diff", "abs"],
    )
    p.add_argument("--use-global-context", action="store_true")
    p.add_argument("--stochastic-depth-rate", type=float, default=0.1)

    # Routing head
    p.add_argument(
        "--routing-mode", type=str, default="trainable",
        choices=["trainable", "deterministic"],
        help="Hierarchical routing layer mode",
    )

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

    p.add_argument("--checkpoint-every-steps", type=int, default=25000)
    p.add_argument("--analyze-every-steps", type=int, default=50000)
    p.add_argument("--max-checkpoints", type=int, default=3)

    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to .keras checkpoint to resume training from",
    )

    p.add_argument("--save-dir", type=str,
                    default="results/cliffordnet_nlp_routing")

    return p


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    shifts = [int(s) for s in args.shifts.split(",")]
    return TrainingConfig(
        variant=args.variant,
        channels=args.channels,
        depth=args.depth,
        shifts=shifts,
        cli_mode=args.cli_mode,
        ctx_mode=args.ctx_mode,
        use_global_context=args.use_global_context,
        stochastic_depth_rate=args.stochastic_depth_rate,
        dropout_rate=args.dropout_rate,
        routing_mode=args.routing_mode,
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
        checkpoint_every_steps=args.checkpoint_every_steps,
        analyze_every_steps=args.analyze_every_steps,
        max_checkpoints=args.max_checkpoints,
        resume_from=args.resume,
        save_dir=args.save_dir,
    )


def main() -> None:
    """Main entry point."""
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.variant}, "
        f"routing_mode={config.routing_mode}, "
        f"channels={config.channels}, depth={config.depth}, "
        f"shifts={config.shifts}, cli_mode={config.cli_mode}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, loss={config.loss_type}, "
        f"source={config.dataset_source}"
    )

    train_cliffordnet_nlp_routing(config)


if __name__ == "__main__":
    main()
