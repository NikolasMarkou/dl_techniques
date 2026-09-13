"""GPT-2 Domain Fine-tuning with Causal Language Modeling.

Fine-tunes a pre-trained GPT-2 model on a domain-specific text corpus
using next-token prediction (causal LM). Supports loading from any
HuggingFace text dataset or local text files.

Usage examples::

    # Fine-tune on a HuggingFace dataset
    python -m train.gpt2.finetune \\
        --pretrained results/gpt2_pretrain_v7/gpt2_final_best.keras \\
        --hf-dataset "wikitext" --hf-config "wikitext-103-raw-v1" \\
        --epochs 10 --gpu 1

    # Fine-tune on local text files
    python -m train.gpt2.finetune \\
        --pretrained results/gpt2_pretrain_v7/gpt2_final_best.keras \\
        --text-dir /path/to/domain/texts/ \\
        --epochs 10 --gpu 1

    # Fine-tune with frozen embeddings (only train transformer layers)
    python -m train.gpt2.finetune \\
        --pretrained results/gpt2_pretrain_v7/gpt2_final_best.keras \\
        --hf-dataset "wikitext" --hf-config "wikitext-103-raw-v1" \\
        --freeze-embeddings --epochs 10 --gpu 1
"""

import os
import glob
import keras
import argparse
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    StepPlotCallback,
    set_seeds,
    save_config_json,
    save_training_history_json,
)
from train.common.nlp import (
    create_tokenizer,
    preprocess_clm_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
    estimate_clm_steps_per_epoch,
)
from train.common.clm_pretrain import create_clm_loss_fn

from dl_techniques.datasets.nlp import load_hf_text_dataset
from dl_techniques.models.language.masked_language_model.clm import CausalLanguageModel
from dl_techniques.utils.logger import logger
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


# DECISION plan-2026-08-13T091555-230c101d/D-012: this MUST stay a @dataclass.
# It was a plain class with class-level attribute defaults until step 6 adopted
# `save_config_json(config, results_dir)` below. MEASURED, not assumed: on a
# plain class the helper takes its `hasattr(config, "__dict__")` branch and
# serializes `vars(config)`, which holds only INSTANCE attributes —
# `len(vars(FinetuneConfig()))` was **0**, so the adopted call wrote a
# `config.json` containing `{}` while every default sat on the class and was
# silently omitted (a run whose config.json records nothing looks healthy).
# The `@dataclass` route makes `dataclasses.is_dataclass` true, so the helper
# takes its `dataclasses.asdict` branch and records all 29 fields (MEASURED:
# 0 keys written before, 29 after).
# DO NOT revert this to a plain class, and DO NOT "fix" it inside
# `save_config_json` by teaching it to walk class-level annotations — that
# helper is shared by every trainer in `src/train/` and a class-attribute
# fallback would start recording class defaults an instance never received.
# See decisions.md D-012.
@dataclass
class FinetuneConfig:
    """Configuration for GPT-2 domain fine-tuning."""

    # Pre-trained model
    pretrained_path: str = "results/gpt2_pretrain_v7/gpt2_final_best.keras"

    # Model (must match pre-trained model)
    max_seq_length: int = 512

    # Tokenizer (must match pre-trained model)
    encoding_name: str = "gpt2"
    cls_token_id: int = 50257
    sep_token_id: int = 50258
    pad_token_id: int = 50259
    mask_token_id: int = 50260

    # Training — lower LR than pretraining to avoid catastrophic forgetting
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4  # 5× lower than pretrain
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # Freezing
    freeze_embeddings: bool = False
    freeze_n_layers: int = 0  # Freeze first N transformer layers

    # Loss: "ce" (MaskedCausalLMLoss, default) or "focal" (FocalCausalLMLoss).
    # Applied via create_clm_loss_fn(config), matching pretrain.py's own
    # flags -- fine-tuning is free to select a different loss family than
    # whatever the checkpoint was pre-trained with.
    loss_type: str = "ce"
    focal_gamma: float = 1.0
    label_smoothing: float = 0.0

    # Paths
    save_dir: str = "results/gpt2_finetune"

    # Data source: "huggingface", "text_files", or "tfds"
    data_source: str = "huggingface"

    # HuggingFace dataset settings
    hf_dataset_path: str = "wikitext"
    hf_dataset_name: Optional[str] = "wikitext-103-raw-v1"
    hf_text_field: str = "text"
    hf_cache_dir: Optional[str] = None
    min_text_length: int = 100

    # Local text files settings
    text_dir: Optional[str] = None
    text_glob: str = "*.txt"

    # Train/val split for local files
    val_fraction: float = 0.1

    # Optional override of LR-schedule horizon. None uses
    # the chunk-aware estimator with measured train cardinality.
    steps_per_epoch: Optional[int] = None

    # End-to-end seed plumbing.
    seed: int = 42

    # Analysis
    run_epoch_analysis: bool = True
    analysis_start_epoch: int = 1
    analysis_epoch_frequency: int = 5


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------


def _load_text_files(
    text_dir: str,
    text_glob: str,
    min_length: int,
) -> tf.data.Dataset:
    """Load text from local files as a tf.data.Dataset."""
    pattern = os.path.join(text_dir, text_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {pattern}. Check --text-dir and --text-glob."
        )
    logger.info(f"Found {len(files)} text files in {text_dir}")

    texts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
            if len(text) >= min_length:
                texts.append(text)

    logger.info(f"Loaded {len(texts)} texts (min_length={min_length})")
    return tf.data.Dataset.from_tensor_slices(texts)


def load_finetune_datasets(
    config: FinetuneConfig,
    preprocessor,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load and preprocess train/val datasets for fine-tuning."""

    if config.data_source == "huggingface":
        train_raw, val_raw = _load_hf_train_val(config)
    elif config.data_source == "text_files":
        train_raw, val_raw = _load_text_files_train_val(config)
    else:
        raise ValueError(f"Unknown data_source: {config.data_source!r}")

    # `preprocess_clm_dataset` already yields pre-shifted `(input_ids, labels)`
    # pairs -- exactly the `pre_shifted=True` contract `CausalLanguageModel`
    # expects (see `masked_language_model/clm.py` module docstring). No label
    # wrapping is applied: a `CausalLanguageModel`'s `call()` returns a plain
    # logits tensor (`skip_head=True, output_key="logits"` only names which
    # key to pull OUT of the backbone's own dict output; `CausalLanguageModel`
    # itself never re-wraps that tensor), so a dict-keyed `{"logits": y}`
    # label here would mismatch `_unpack_batch`'s pre_shifted branch, which
    # treats `y` AS the label tensor with no further unwrapping.
    train_ds = preprocess_clm_dataset(
        train_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    val_ds = preprocess_clm_dataset(
        val_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    return train_ds, val_ds


def _load_hf_train_val(
    config: FinetuneConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train/val from HuggingFace dataset."""
    train_raw = load_hf_text_dataset(
        path=config.hf_dataset_path,
        name=config.hf_dataset_name,
        split="train",
        text_field=config.hf_text_field,
        cache_dir=config.hf_cache_dir,
        min_length=config.min_text_length,
        streaming=False,
    )

    # Try standard val split, fall back to test
    for val_split in ("validation", "test"):
        try:
            val_raw = load_hf_text_dataset(
                path=config.hf_dataset_path,
                name=config.hf_dataset_name,
                split=val_split,
                text_field=config.hf_text_field,
                cache_dir=config.hf_cache_dir,
                min_length=config.min_text_length,
                streaming=False,
            )
            logger.info(f"Using '{val_split}' split for validation")
            return train_raw, val_raw
        except (ValueError, KeyError):
            continue

    logger.warning("No val/test split found — using last 10% of train")
    total = tf.data.experimental.cardinality(train_raw).numpy()
    if total > 0:
        val_size = max(int(total * config.val_fraction), 1)
        val_raw = train_raw.skip(total - val_size)
        train_raw = train_raw.take(total - val_size)
    return train_raw, val_raw


def _load_text_files_train_val(
    config: FinetuneConfig,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train/val from local text files."""
    all_data = _load_text_files(
        config.text_dir, config.text_glob, config.min_text_length,
    )
    total = tf.data.experimental.cardinality(all_data).numpy()
    val_size = max(int(total * config.val_fraction), 1)

    all_data = all_data.shuffle(buffer_size=total, seed=42)
    val_raw = all_data.take(val_size)
    train_raw = all_data.skip(val_size)

    logger.info(f"Text files split: {total - val_size} train, {val_size} val")
    return train_raw, val_raw


# ---------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------


def load_pretrained_model(config: FinetuneConfig) -> CausalLanguageModel:
    """Load a pre-trained GPT-2 checkpoint, wrapped in ``CausalLanguageModel``.

    A checkpoint saved by the current (``CausalLanguageModel``-wrapped)
    ``pretrain.py`` deserializes as a ``CausalLanguageModel`` directly. A
    **legacy** checkpoint, saved before that migration, deserializes as a
    bare ``GPT2`` and is wrapped here on the fly so the rest of this module
    (freeze-layer walk, ``compile_model``, ``train_step``) only ever handles
    one shape. This mirrors the disjunction
    ``gpt2/pretrain.py:load_model_from_checkpoint`` already documents for its
    own ``--resume`` path.

    Either way, ``loss_fn`` is (re)set from THIS config's
    ``loss_type``/``focal_gamma``/``label_smoothing`` — fine-tuning is free
    to select a different loss family than whatever the checkpoint was
    pre-trained with, matching ``pretrain.py``'s own CLI-selected loss
    rather than silently keeping the pretrain-time choice frozen.
    """
    logger.info(f"Loading pre-trained model from: {config.pretrained_path}")
    model = keras.models.load_model(
        config.pretrained_path,
        custom_objects={
            "MaskedCausalLMLoss": MaskedCausalLMLoss,
            "FocalCausalLMLoss": FocalCausalLMLoss,
            "CausalLanguageModel": CausalLanguageModel,
        },
    )

    if isinstance(model, CausalLanguageModel):
        clm_model = model
    else:
        logger.info(
            "Loaded a legacy (pre-CausalLanguageModel) bare GPT2 checkpoint; "
            "wrapping it now."
        )
        clm_model = CausalLanguageModel(
            backbone=model,
            vocab_size=model.vocab_size,
            skip_head=True,
            output_key="logits",
            pre_shifted=True,
        )
        # A freshly constructed wrapper is unbuilt until its first forward
        # pass -- `count_params()`/`trainable_weights` below raise on an
        # unbuilt model. `create_gpt2_model` (pretrain.py) forces this the
        # same way for a fresh model; a loaded-then-wrapped one needs it too.
        dummy = np.random.randint(
            0, model.vocab_size, size=(1, config.max_seq_length - 1)
        ).astype("int32")
        clm_model(dummy, training=False)

    clm_model.loss_fn = create_clm_loss_fn(config)

    total_p = clm_model.count_params()
    logger.info(f"Loaded GPT-2: {total_p:,} parameters")

    # Freeze embeddings/transformer layers. GPT2's own top-level `.layers`
    # is a single `[decoder]` entry (its `TextDecoder` sublayer bundles the
    # embeddings and the per-block stack), so neither `clm_model.layers`
    # (`[backbone]`, one opaque layer under the wrapper) nor
    # `clm_model.backbone.layers` (`[decoder]`, one opaque layer under the
    # bare backbone) ever matches "embedding"/"transformer"/"block" --
    # MEASURED: this freeze logic was already a no-op for a bare GPT2 before
    # this fix, independent of the CausalLanguageModel migration. A
    # recursive walk is required to reach the actual named sublayers
    # (`word_embeddings`, `positional_embeddings`, `decoder_layer_<i>`).
    backbone_sublayers = list(
        clm_model.backbone._flatten_layers(include_self=False, recursive=True)
    )
    if config.freeze_embeddings:
        for layer in backbone_sublayers:
            if "embedding" in layer.name.lower():
                layer.trainable = False
                logger.info(f"Froze layer: {layer.name}")

    if config.freeze_n_layers > 0:
        frozen = 0
        for layer in backbone_sublayers:
            name = layer.name.lower()
            if "transformer" in name or "block" in name or "decoder_layer" in name:
                if frozen < config.freeze_n_layers:
                    layer.trainable = False
                    frozen += 1
                    logger.info(f"Froze layer: {layer.name}")

    trainable = sum(
        int(np.prod(w.shape)) for w in clm_model.trainable_weights
    )
    logger.info(f"Trainable parameters: {trainable:,} / {total_p:,}")
    return clm_model


def compile_model(
    model: CausalLanguageModel,
    config: FinetuneConfig,
    steps_per_epoch: int,
) -> None:
    """Compile GPT-2 for fine-tuning with lower LR.

    No ``loss=``/``metrics=`` is passed: ``CausalLanguageModel`` computes and
    reports its own ``loss``/``accuracy``/``perplexity`` via its hand-rolled
    ``train_step``/``test_step`` (``load_pretrained_model`` already set
    ``model.loss_fn`` from this config) -- matching ``pretrain.py``'s own
    ``compile_model``, a compiled ``loss=``/``metrics=`` here would be unused
    dead configuration.
    """
    lr_schedule = create_warmup_lr_schedule(
        config.learning_rate,
        config.num_epochs,
        steps_per_epoch,
        config.warmup_ratio,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer)
    logger.info(
        f"Compiled for fine-tuning: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}"
    )


def finetune_gpt2(
    config: FinetuneConfig,
) -> Tuple[CausalLanguageModel, keras.callbacks.History]:
    """Run GPT-2 domain fine-tuning."""
    logger.info("=" * 60)
    logger.info("GPT-2 Domain Fine-tuning (CLM)")
    logger.info("=" * 60)

    set_seeds(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    # Tokenizer (must match pre-trained model)
    preprocessor = create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    )

    # Data
    train_dataset, val_dataset = load_finetune_datasets(config, preprocessor)

    # Estimate steps_per_epoch via the canonical helper.
    # tf.data.Dataset.cardinality() returns UNKNOWN_CARDINALITY for the
    # generator-backed packed CLM pipeline, so we cannot count chunks
    # directly. Fall back to the helper's default Wikipedia-token estimate
    # (or --steps-per-epoch override). Domain-specific finetune corpora
    # should set --steps-per-epoch explicitly.
    steps_per_epoch = estimate_clm_steps_per_epoch(
        num_articles=None,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        override=config.steps_per_epoch,
    )
    logger.info(f"steps_per_epoch={steps_per_epoch:,}")

    # Model
    model = load_pretrained_model(config)
    compile_model(model, config, steps_per_epoch)

    # Callbacks
    callbacks, results_dir = create_nlp_callbacks(
        model_name="GPT2-finetune",
        results_dir_prefix="gpt2_finetune",
        patience=5,  # less patience than pretraining
        include_analyzer=config.run_epoch_analysis,
        analyzer_epoch_frequency=config.analysis_epoch_frequency,
        analyzer_start_epoch=config.analysis_start_epoch,
    )
    callbacks.append(StepPlotCallback(save_dir=results_dir))
    save_config_json(config, results_dir)

    # Train
    logger.info(
        f"Starting fine-tuning: source={config.data_source}, "
        f"lr={config.learning_rate}, freeze_emb={config.freeze_embeddings}, "
        f"freeze_layers={config.freeze_n_layers}"
    )
    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )
    logger.info("Fine-tuning completed!")
    save_training_history_json(history, results_dir)

    # Save
    final_path = os.path.join(config.save_dir, "gpt2_finetuned.keras")
    model.save(final_path)
    logger.info(f"Model saved to: {final_path}")

    # Summary
    best_epoch = tf.argmin(history.history["val_loss"]).numpy()
    logger.info(
        f"Best epoch: {best_epoch + 1} "
        f"(val_loss: {history.history['val_loss'][best_epoch]:.4f})"
    )
    return model, history


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Main entry point for GPT-2 domain fine-tuning."""
    parser = argparse.ArgumentParser(description="GPT-2 Domain Fine-tuning")
    parser.add_argument(
        "--gpu", type=int, default=None, help="GPU device index"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pre-trained .keras model",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Fine-tuning epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Maximum sequence length (must match pre-trained model)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Peak learning rate (default: 1e-4, lower than pretrain)",
    )

    # Freezing
    parser.add_argument(
        "--freeze-embeddings", action="store_true",
        help="Freeze embedding layers (train only transformer)",
    )
    parser.add_argument(
        "--freeze-n-layers", type=int, default=0,
        help="Freeze first N transformer layers",
    )

    # Loss — matches pretrain.py's flags; fine-tuning may select a different
    # loss family than the checkpoint was pre-trained with.
    parser.add_argument(
        "--loss-type", type=str, default="ce",
        choices=["ce", "focal"],
        help="'ce' (MaskedCausalLMLoss) or 'focal' (FocalCausalLMLoss)",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=1.0,
        help="Focal loss gamma (only if --loss-type focal)",
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--hf-dataset", type=str,
        help="HuggingFace dataset path (e.g. 'wikitext')",
    )
    data_group.add_argument(
        "--text-dir", type=str,
        help="Directory containing .txt files for fine-tuning",
    )

    parser.add_argument(
        "--hf-config", type=str, default=None,
        help="HuggingFace dataset config name",
    )
    parser.add_argument(
        "--hf-cache-dir", type=str, default=None,
        help="Cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--text-glob", type=str, default="*.txt",
        help="Glob pattern for text files (default: '*.txt')",
    )
    parser.add_argument(
        "--save-dir", type=str, default="results/gpt2_finetune",
        help="Output directory",
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=None,
        help="Override LR-schedule horizon (overrides chunk-aware estimate). "
             "Recommended for small finetune corpora to avoid the helper's "
             "Wikipedia-token fallback.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global seed for tf/keras + dataset shuffle (D-006).",
    )

    args = parser.parse_args()

    setup_gpu(gpu_id=args.gpu)

    config = FinetuneConfig()
    config.pretrained_path = args.pretrained
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.max_seq_length = args.max_seq_length
    config.learning_rate = args.learning_rate
    config.freeze_embeddings = args.freeze_embeddings
    config.freeze_n_layers = args.freeze_n_layers
    config.loss_type = args.loss_type
    config.focal_gamma = args.focal_gamma
    config.label_smoothing = args.label_smoothing
    config.save_dir = args.save_dir
    config.steps_per_epoch = args.steps_per_epoch
    config.seed = args.seed

    if args.hf_dataset:
        config.data_source = "huggingface"
        config.hf_dataset_path = args.hf_dataset
        config.hf_dataset_name = args.hf_config
        config.hf_cache_dir = args.hf_cache_dir
    elif args.text_dir:
        config.data_source = "text_files"
        config.text_dir = args.text_dir
        config.text_glob = args.text_glob

    logger.info(
        f"Config: pretrained={config.pretrained_path}, "
        f"epochs={config.num_epochs}, batch_size={config.batch_size}, "
        f"lr={config.learning_rate}, source={config.data_source}"
    )

    model, history = finetune_gpt2(config)
    logger.info(
        f"Fine-tuning complete! Model: {config.save_dir}/gpt2_finetuned.keras"
    )


if __name__ == "__main__":
    main()
