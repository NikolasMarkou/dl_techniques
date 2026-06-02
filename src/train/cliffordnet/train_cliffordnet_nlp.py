"""CliffordNet NLP Pre-training with Causal Language Modeling.

Adapts the CliffordNet geometric algebra backbone (arXiv:2601.06793v2) for
causal language modeling. The CausalCliffordNetBlock operates on 4D tensors
``(B, H, W, D)``; sequences are reshaped to ``(B, 1, seq_len, D)`` so the
causal (left-padded) depthwise convolutions act as 1D local context
extractors along the sequence dimension, while the Clifford algebraic
products provide multi-scale channel interaction.

The model (CliffordNetLM) wraps:
  1. Token + learned positional embeddings
  2. L x CausalCliffordNetBlock (operating on ``(B, 1, seq_len, channels)``)
  3. LayerNorm + Dense projection to vocabulary logits

Supports Wikipedia (HuggingFace) and TFDS text datasets, step-based
checkpointing, generation probes, and warmup + cosine decay LR scheduling.

Usage::

    # Wikipedia (default) on GPU 0
    python -m train.cliffordnet.train_cliffordnet_nlp --gpu 0 --variant nano --epochs 3

    # TFDS dataset with focal loss
    python -m train.cliffordnet.train_cliffordnet_nlp \\
        --dataset-source tfds --dataset-name imdb_reviews --loss-type focal

    # Custom architecture
    python -m train.cliffordnet.train_cliffordnet_nlp \\
        --variant custom --channels 256 --depth 12 --shifts 1,2,4,8

    # Resume from checkpoint
    python -m train.cliffordnet.train_cliffordnet_nlp \\
        --resume results/.../checkpoints/step_0050000.keras
"""

import os
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
from train.common import StepCheckpointCallback, GenerationProbeCallback
from train.common.evaluation import generate_training_curves
from train.common.nlp import (
    create_tokenizer,
    load_text_dataset,
    preprocess_clm_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
    estimate_clm_steps_per_epoch,
    build_clm_metrics,
    prepare_dict_keyed_compile,
    augment_probe_results,
)
from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
)
from dl_techniques.models.cliffordnet.lm import CliffordNetLM
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss


_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for CliffordNet NLP CLM pre-training."""

    # Model
    variant: str = "nano"
    channels: int = 128
    depth: int = 12
    shifts: List[int] = field(default_factory=lambda: [1, 2])
    cli_mode: str = "full"
    ctx_mode: str = "diff"
    use_global_context: bool = False
    dropout_rate: float = 0.0
    stochastic_depth_rate: float = 0.1
    tie_word_embeddings: bool = True

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
    save_dir: str = "results/cliffordnet_nlp"

    # Data source: "huggingface" or "tfds"
    dataset_source: str = "huggingface"

    # TFDS settings
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # HuggingFace / Wikipedia settings
    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    # DECISION D-003: 0 → packed CLM uses every token; pass 500+ only for
    # per-doc consumers (MLM, classification).
    min_article_length: int = 0
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: Optional[int] = None
    # DECISION D-002: parallel tokenization shards + per-epoch reshuffle.
    shuffle_shards: int = 4

    # Checkpointing & analysis (step-based for large datasets)
    checkpoint_every_steps: int = 25000
    analyze_every_steps: int = 50000
    max_checkpoints: int = 3
    # Optional override of LR-schedule horizon.
    steps_per_epoch: Optional[int] = None

    # Resume from checkpoint
    resume_from: Optional[str] = None
    # DECISION D-006: end-to-end seed plumbing.
    seed: int = 42

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
# Generation Probe forward-pass closure
# ---------------------------------------------------------------------------


def _make_logits_fn(model, ctx_length: int, pad_id: int):
    """Build the next-token ``logits_fn`` closure for ``GenerationProbeCallback``.

    The model-specific forward pass is lifted VERBATIM from the former local
    ``GenerationProbeCallback._generate`` loop so behaviour is byte-identical:
    every call right-pads the (unpadded) ``ctx`` to a fixed ``ctx_length`` with
    ``pad_id`` (the EOT id), runs the model, and reads the next-token logits
    from the LAST REAL position ``[0, real - 1, :]`` (NOT the last array
    position). The fixed input shape collapses ~100 per-length traces into one
    compiled graph, and reading ``real - 1`` keeps the output semantics
    identical to a variable-length call. Do NOT replace this with a plain
    ``model(ctx)[..., -1, :]`` read: the trailing PAD positions are not valid
    next-token positions for this padded call.
    """

    def logits_fn(ctx: np.ndarray) -> np.ndarray:
        real = ctx.shape[1]
        padded = list(ctx[0]) + [pad_id] * (ctx_length - real)
        out = model(np.array([padded], dtype="int32"), training=False)
        return out["logits"][0, real - 1, :].numpy()

    return logits_fn



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
) -> Tuple[CliffordNetLM, int]:
    """Load a CliffordNetLM from a ``.keras`` checkpoint.

    :param path: Path to the checkpoint file.
    :return: ``(model, step)`` tuple.
    """
    logger.info(f"Resuming from checkpoint: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={
            "CliffordNetLM": CliffordNetLM,
            "CausalCliffordNetBlock": CausalCliffordNetBlock,
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


def create_model(config: TrainingConfig) -> CliffordNetLM:
    """Create and build a CliffordNetLM from training configuration."""
    logger.info(f"Creating CliffordNetLM-{config.variant.upper()}...")

    if config.variant in CliffordNetLM.MODEL_VARIANTS:
        model = CliffordNetLM.from_variant(
            config.variant,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            dropout_rate=config.dropout_rate,
            tie_word_embeddings=config.tie_word_embeddings,
        )
    else:
        # Custom variant: use explicit params from config
        model = CliffordNetLM(
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
            tie_word_embeddings=config.tie_word_embeddings,
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


def load_train_val_datasets(
    config: TrainingConfig,
    preprocessor,
    data_seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int]]:
    """Load, preprocess, and wrap train/val datasets for dict-output model.

    :return: ``(train_ds, val_ds, n_train_articles)``.
    """
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compile_model(
    model: CliffordNetLM,
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
    prepare_dict_keyed_compile(model)
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
    """Resolve steps_per_epoch via the canonical helper (D-001)."""
    if config.dataset_source == "tfds" and config.max_samples and config.steps_per_epoch is None:
        return max(1, config.max_samples // config.batch_size)
    return estimate_clm_steps_per_epoch(
        num_articles=n_train_articles or config.max_train_samples,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        override=config.steps_per_epoch,
    )


def train_cliffordnet_nlp(
    config: TrainingConfig,
) -> Tuple[CliffordNetLM, keras.callbacks.History]:
    """Run CliffordNet NLP CLM pre-training.

    :param config: Training configuration.
    :return: Trained model and training history.
    """
    logger.info("=" * 60)
    logger.info("CliffordNet NLP Causal LM Pre-training")
    logger.info("=" * 60)

    tf.random.set_seed(config.seed)
    keras.utils.set_random_seed(config.seed)

    # Tokenizer
    preprocessor = create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    )

    # DECISION D-006: derive data_seed before building the dataset.
    initial_step = (
        _extract_step_from_checkpoint(config.resume_from)
        if config.resume_from else 0
    )
    data_seed = config.seed + initial_step

    # Data
    train_dataset, val_dataset, n_train_articles = load_train_val_datasets(
        config, preprocessor, data_seed=data_seed,
    )

    # Model -- resume from checkpoint or create fresh
    steps_per_epoch = _make_steps_per_epoch(config, n_train_articles)

    if config.resume_from:
        model, initial_step = load_model_from_checkpoint(config.resume_from)
    else:
        model = create_model(config)

    compile_model(model, config, steps_per_epoch)

    # Callbacks: standard NLP callbacks + step-based checkpointing
    variant_label = config.variant
    if config.variant == "custom":
        variant_label = f"c{config.channels}d{config.depth}"

    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"CliffordNetLM-{variant_label}",
        results_dir_prefix="cliffordnet_nlp",
        include_analyzer=False,
    )
    # Remove epoch-level CSVLogger — StepCheckpointCallback handles
    # step-level CSV logging instead (epochs are too long for Wikipedia).
    callbacks = [
        cb for cb in callbacks
        if not isinstance(cb, keras.callbacks.CSVLogger)
    ]
    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=config.checkpoint_every_steps,
        analyze_every_steps=config.analyze_every_steps,
        max_checkpoints=config.max_checkpoints,
        model_name=f"CliffordNetLM-{variant_label}",
        initial_step=initial_step,
        gc_on_save=True,
    ))

    # Generation probes. The forward pass (right-pad to ctx_length, read the
    # last REAL position) is lifted verbatim into `_make_logits_fn`; the common
    # callback owns the EOT/special suppression, sign-aware repetition penalty,
    # temperature, nucleus sampling, decode, and logging.
    _probe_ctx_length = config.max_seq_length - 1
    _probe_eot_id = tiktoken.get_encoding(config.encoding_name).eot_token
    probe_cb = GenerationProbeCallback(
        logits_fn=_make_logits_fn(model, _probe_ctx_length, _probe_eot_id),
        repetition_penalty_mode="sign_aware",
        probe_every_steps=config.checkpoint_every_steps,
        prompts=config.probe_prompts,
        encoding_name=config.encoding_name,
        max_tokens=config.probe_max_tokens,
        temperature=config.probe_temperature,
        top_p=config.probe_top_p,
        repetition_penalty=config.probe_repetition_penalty,
        eot_token_id=_probe_eot_id,
        ctx_length=_probe_ctx_length,
        save_dir=results_dir,
        initial_step=initial_step,
    )
    probe_cb._post_generate_hook = augment_probe_results
    callbacks.append(probe_cb)

    # Train
    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch~={steps_per_epoch:,}, "
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

    # Plot training curves
    generate_training_curves(history, results_dir)

    # Summary
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
        description="CliffordNet NLP Causal LM Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Hardware
    p.add_argument("--gpu", type=int, default=None, help="GPU device index")

    # Model
    p.add_argument(
        "--variant", type=str, default="nano",
        choices=list(CliffordNetLM.MODEL_VARIANTS.keys()) + ["custom"],
        help="Model variant (nano/mini/base/large/xl/custom)",
    )
    p.add_argument("--channels", type=int, default=128,
                    help="Feature dimension D (custom variant)")
    p.add_argument("--depth", type=int, default=12,
                    help="Number of CliffordNet blocks (custom variant)")
    p.add_argument("--shifts", type=str, default="1,2",
                    help="Comma-separated shift offsets (custom variant)")
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
                    help="Enable global context branch (custom variant)")
    p.add_argument("--stochastic-depth-rate", type=float, default=0.1,
                    help="Maximum stochastic depth rate")
    p.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tie LM head to token embeddings (use --no-tie-word-embeddings to disable)",
    )

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--dropout-rate", type=float, default=0.0)

    # Loss
    p.add_argument(
        "--loss-type", type=str, default="ce",
        choices=["ce", "focal"],
        help="'ce' (MaskedCausalLMLoss) or 'focal' (FocalCausalLMLoss)",
    )
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

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
    p.add_argument(
        "--min-article-length", type=int, default=0,
        help="HF Wikipedia char-length filter. 0 (default) = no filter "
             "(recommended for packed CLM). 500+ for per-doc consumers.",
    )
    p.add_argument(
        "--shuffle-shards", type=int, default=4,
        help="HF Wikipedia parallel tokenization shards (D-002). 1 = "
             "single-thread, deterministic; >1 reshuffles each epoch and "
             "parallelises tokenization.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global seed for tf/keras + dataset shuffle. On --resume, data "
             "seed is shifted by initial_step (D-006).",
    )

    # Checkpointing
    p.add_argument("--checkpoint-every-steps", type=int, default=25000)
    p.add_argument("--analyze-every-steps", type=int, default=50000,
                    help="0 to disable")
    p.add_argument("--max-checkpoints", type=int, default=3)
    p.add_argument(
        "--steps-per-epoch", type=int, default=None,
        help="Override LR-schedule horizon (overrides chunk-aware estimate).",
    )

    # Resume
    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to .keras checkpoint to resume training from",
    )

    # Output
    p.add_argument("--save-dir", type=str, default="results/cliffordnet_nlp")

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
        tie_word_embeddings=args.tie_word_embeddings,
        dropout_rate=args.dropout_rate,
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
    """Main entry point for CliffordNet NLP CLM pre-training."""
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.variant}, "
        f"channels={config.channels}, depth={config.depth}, "
        f"shifts={config.shifts}, cli_mode={config.cli_mode}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, loss={config.loss_type}, "
        f"source={config.dataset_source}"
    )

    train_cliffordnet_nlp(config)


if __name__ == "__main__":
    main()
