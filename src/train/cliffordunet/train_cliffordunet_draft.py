"""CliffordUNet DFlash-style diffusion draft training.

Trains a :class:`CliffordUNetDraftModel` with a **block-masked denoising**
objective: for each training example, a random contiguous span of tokens is
replaced with a mask token, and the model learns to reconstruct the masked
span given the surrounding context.  This is the DFlash (Ringel et al., 2026)
training signal, adapted to the CliffordUNet backbone and Keras 3.

Unlike the full DFlash training pipeline, this script runs **standalone**:
no verifier model is required.  The ``target_hidden`` cross-conditioning
input of :class:`CliffordUNetDraftModel` is dropped during training and
the model learns a pure self-denoiser that can later be plugged into a
speculative-decoding harness by reading the target hidden states at
inference time.

Pipeline:

1. Load Wikipedia (HuggingFace) or a TFDS text dataset.
2. Tokenize with Tiktoken gpt2 encoding (same tokenizer as
   ``train_cliffordunet_nlp``).
3. For each sequence:

   * Sample a random span of length ``block_size`` (default 16).
   * Replace those tokens with ``mask_token_id``.
   * Use the raw GPT-2 token embedding table as the draft's "target
     hidden" embedding source — the draft projects these through its
     own ``Dense(target_hidden_size)`` so the fact that the embedding
     is not a large verifier's is harmless at training time.
4. Run the draft model's forward pass on the masked embedding to produce
   hidden states, then project to vocabulary logits via a learnable
   ``Dense(vocab_size)`` head.
5. Compute masked CLM cross-entropy on the masked positions only
   (``ignore_index`` outside the span).

Usage::

    python -m train.cliffordunet.train_cliffordunet_draft \\
        --gpu 0 --variant draft_nano --epochs 1 --batch-size 16 \\
        --block-size 16 --max-train-samples 200000
"""

import os
import csv
import gc
import glob
import argparse
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import initializers

from train.common import setup_gpu
from train.common.evaluation import generate_training_curves
from train.common.nlp import (
    create_tokenizer,
    load_text_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
)
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.models.cliffordunet.lm import (
    CausalWindowPool,
    MultiLinearUpsample,
)
from dl_techniques.models.cliffordunet.draft import CliffordUNetDraftModel
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.losses import MaskedCausalLMLoss


_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for CliffordUNet draft model training."""

    # Model
    variant: str = "draft_nano"
    target_hidden_size: int = 1024
    num_target_layers: int = 0  # 0 => disable cross-conditioning during standalone training
    channels: List[int] = field(default_factory=lambda: [256, 384, 512])
    encoder_depths: List[int] = field(default_factory=lambda: [2, 4, 2])
    decoder_depths: List[int] = field(default_factory=lambda: [2, 2])
    pool_sizes: List[int] = field(default_factory=lambda: [4, 4])
    shifts: List[int] = field(default_factory=lambda: [1, 2])
    cli_mode: str = "full"
    ctx_mode: str = "diff"
    use_global_context: bool = False
    dropout_rate: float = 0.0
    stochastic_depth_rate: float = 0.1

    # Tokenizer (Tiktoken gpt2 encoding)
    vocab_size: int = 50261
    max_seq_length: int = 512
    encoding_name: str = "gpt2"
    cls_token_id: int = 50257
    sep_token_id: int = 50258
    pad_token_id: int = 50259
    mask_token_id: int = 50260

    # Block denoising
    block_size: int = 16            # length of masked span per example
    min_block_start: int = 1        # earliest valid start index (avoid masking CLS)

    # Training
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    label_smoothing: float = 0.0

    # Paths
    save_dir: str = "results/cliffordunet_draft"

    # Data source
    dataset_source: str = "huggingface"
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    min_article_length: int = 500
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: Optional[int] = None

    # Checkpointing
    checkpoint_every_steps: int = 10000
    max_checkpoints: int = 3

    resume_from: Optional[str] = None


# ---------------------------------------------------------------------------
# Denoiser head wrapper: draft + vocab projection
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="dl_techniques")
class CliffordUNetDraftLM(keras.Model):
    """CliffordUNetDraftModel wrapped with an embedding table and a vocab head.

    Training harness that turns the draft model into a standalone
    block-masked language model:

    * ``token_embedding`` maps token IDs to draft hidden vectors
      (played by ``target_hidden_size``).
    * ``draft`` is the :class:`CliffordUNetDraftModel` denoiser.
    * ``output_proj`` projects draft hidden states to vocabulary logits.

    At inference time inside a DDTree / DFlash harness, the ``token_embedding``
    is replaced by the verifier model's embedding table and ``output_proj`` is
    replaced by the verifier's ``lm_head`` — this wrapper is for training only.
    """

    def __init__(
        self,
        draft: CliffordUNetDraftModel,
        vocab_size: int,
        mask_token_id: int,
        name: str = "cliffordunet_draft_lm",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.draft = draft
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.target_hidden_size = draft.target_hidden_size

        self.token_embedding = keras.layers.Embedding(
            vocab_size,
            draft.target_hidden_size,
            name="token_embedding",
        )
        self.output_proj = keras.layers.Dense(
            vocab_size,
            use_bias=False,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
            name="lm_head",
        )

    def call(
        self,
        input_ids: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> dict:
        noise_embedding = self.token_embedding(input_ids)
        draft_out = self.draft(
            {"noise_embedding": noise_embedding},
            training=training,
        )
        logits = self.output_proj(draft_out["hidden_states"])
        return {"logits": logits}

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "draft": keras.saving.serialize_keras_object(self.draft),
            "vocab_size": self.vocab_size,
            "mask_token_id": self.mask_token_id,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "CliffordUNetDraftLM":
        draft_cfg = config.pop("draft")
        draft = keras.saving.deserialize_keras_object(
            draft_cfg,
            custom_objects={
                "CliffordUNetDraftModel": CliffordUNetDraftModel,
                "CliffordNetBlock": CliffordNetBlock,
                "CausalWindowPool": CausalWindowPool,
                "MultiLinearUpsample": MultiLinearUpsample,
            },
        )
        return cls(draft=draft, **config)


# ---------------------------------------------------------------------------
# Block-masked data preprocessing
# ---------------------------------------------------------------------------


def make_block_mask_mapper(
    preprocessor,
    max_seq_length: int,
    block_size: int,
    mask_token_id: int,
    pad_token_id: int,
    ignore_index: int = -100,
    min_block_start: int = 1,
):
    """Return a ``tf.data`` map function that emits ``(masked_ids, target_ids)``.

    The mapper tokenizes a text, clips / pads to ``max_seq_length - 1``,
    replaces a random ``block_size``-token contiguous span with
    ``mask_token_id``, and builds a label tensor whose positions outside
    the masked span are set to ``ignore_index`` (skipped by the loss).

    The mask-span start is sampled uniformly in
    ``[min_block_start, clipped_length - block_size]``.
    """
    seq_len = max_seq_length - 1

    def _tf_tokenize(text):
        def _py(t):
            ids = preprocessor.tokenize(t.numpy().decode("utf-8"))
            ids = list(ids[:seq_len])
            ids = ids + [pad_token_id] * (seq_len - len(ids))
            return np.asarray(ids, dtype=np.int32)
        out = tf.py_function(_py, inp=[text], Tout=tf.int32)
        out.set_shape([seq_len])
        return out

    def _mask_block(input_ids):
        input_ids = tf.cast(input_ids, tf.int32)

        length = tf.reduce_sum(
            tf.cast(tf.not_equal(input_ids, pad_token_id), tf.int32)
        )
        effective_len = tf.maximum(length, block_size + min_block_start)
        max_start = tf.maximum(effective_len - block_size, min_block_start + 1)
        start = tf.random.uniform(
            shape=(), minval=min_block_start, maxval=max_start, dtype=tf.int32,
        )

        positions = tf.range(seq_len, dtype=tf.int32)
        is_masked = tf.logical_and(positions >= start, positions < start + block_size)

        masked_ids = tf.where(
            is_masked,
            tf.fill([seq_len], mask_token_id),
            input_ids,
        )
        target_ids = tf.where(
            is_masked,
            input_ids,
            tf.fill([seq_len], ignore_index),
        )
        return masked_ids, target_ids

    def mapper(example):
        if isinstance(example, dict):
            text = example.get("text", example.get("document", ""))
        else:
            text = example
        ids = _tf_tokenize(text)
        masked, target = _mask_block(ids)
        return masked, {"logits": target}

    return mapper


def _build_dataset(
    raw_ds: tf.data.Dataset,
    mapper,
    batch_size: int,
    shuffle: bool,
    streaming: bool = False,
) -> tf.data.Dataset:
    ds = raw_ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle and not streaming:
        ds = ds.shuffle(1024)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_train_val_datasets(
    config: TrainingConfig,
    preprocessor,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    mapper = make_block_mask_mapper(
        preprocessor=preprocessor,
        max_seq_length=config.max_seq_length,
        block_size=config.block_size,
        mask_token_id=config.mask_token_id,
        pad_token_id=config.pad_token_id,
        ignore_index=-100,
        min_block_start=config.min_block_start,
    )

    if config.dataset_source == "tfds":
        train_raw = load_text_dataset(
            config.dataset_name, "train", config.max_samples,
        )
        val_raw = load_text_dataset(
            config.dataset_name, "test", config.max_samples,
        )
        streaming = False
    elif config.dataset_source == "huggingface":
        train_raw, val_raw = load_wikipedia_train_val(
            cache_dir=config.hf_cache_dir,
            config_name=config.hf_wikipedia_config,
            min_article_length=config.min_article_length,
            val_fraction=config.val_fraction,
            max_train_samples=config.max_train_samples,
            max_val_samples=config.max_val_samples,
        )
        streaming = True
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}. "
            f"Use 'tfds' or 'huggingface'."
        )

    train_ds = _build_dataset(
        train_raw, mapper, config.batch_size, shuffle=True, streaming=streaming,
    )
    val_ds = _build_dataset(
        val_raw, mapper, config.batch_size, shuffle=False, streaming=streaming,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def create_model(config: TrainingConfig) -> CliffordUNetDraftLM:
    logger.info(f"Creating CliffordUNetDraftModel-{config.variant.upper()}...")

    if config.variant in CliffordUNetDraftModel.MODEL_VARIANTS:
        draft = CliffordUNetDraftModel.from_variant(
            config.variant,
            target_hidden_size=config.target_hidden_size,
            max_seq_length=config.max_seq_length,
            num_target_layers=config.num_target_layers,
            dropout_rate=config.dropout_rate,
        )
    else:
        draft = CliffordUNetDraftModel(
            target_hidden_size=config.target_hidden_size,
            max_seq_length=config.max_seq_length,
            num_target_layers=config.num_target_layers,
            channels=config.channels,
            encoder_depths=config.encoder_depths,
            decoder_depths=config.decoder_depths,
            pool_sizes=config.pool_sizes,
            shifts=config.shifts,
            cli_mode=config.cli_mode,
            ctx_mode=config.ctx_mode,
            use_global_context=config.use_global_context,
            dropout_rate=config.dropout_rate,
            stochastic_depth_rate=config.stochastic_depth_rate,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        )

    model = CliffordUNetDraftLM(
        draft=draft,
        vocab_size=config.vocab_size,
        mask_token_id=config.mask_token_id,
    )

    dummy = np.random.randint(
        0, config.vocab_size,
        size=(1, config.max_seq_length - 1),
    ).astype("int32")
    model(dummy, training=False)

    model.summary(print_fn=logger.info)
    return model


# ---------------------------------------------------------------------------
# Step-based checkpoint callback (matches train_cliffordunet_nlp.py)
# ---------------------------------------------------------------------------


class StepCheckpointCallback(keras.callbacks.Callback):
    """Save draft checkpoints at fixed step intervals."""

    def __init__(
        self,
        save_dir: str,
        save_every_steps: int = 10000,
        max_checkpoints: int = 3,
        log_every_steps: int = 100,
        initial_step: int = 0,
    ):
        super().__init__()
        self.save_every_steps = save_every_steps
        self.max_checkpoints = max_checkpoints
        self._log_every_steps = log_every_steps
        self._global_step = initial_step

        self._ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self._ckpt_dir, exist_ok=True)

        self._csv_path = os.path.join(save_dir, "training_log.csv")
        self._csv_file = None
        self._csv_writer = None

        logger.info(
            f"StepCheckpointCallback: save every {save_every_steps} steps, "
            f"keep max {max_checkpoints}, log every {log_every_steps}"
        )

    def on_train_batch_end(self, batch, logs=None):
        self._global_step += 1
        if self._global_step % self._log_every_steps == 0:
            self._log_metrics(logs)
        if self._global_step % self.save_every_steps == 0:
            self._save_checkpoint()

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
        ckpts = sorted(glob.glob(
            os.path.join(self._ckpt_dir, "step_*.keras")
        ))
        while len(ckpts) > self.max_checkpoints:
            old = ckpts.pop(0)
            os.remove(old)


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------


def compile_model(
    model: CliffordUNetDraftLM,
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
        loss={
            "logits": MaskedCausalLMLoss(
                ignore_index=-100,
                label_smoothing=config.label_smoothing,
            ),
        },
        metrics={"logits": ["accuracy"]},
    )
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}, ignore_index=-100, "
        f"block_size={config.block_size}"
    )


def _estimate_steps_per_epoch(config: TrainingConfig) -> int:
    if config.max_train_samples:
        return config.max_train_samples // config.batch_size
    return 4_850_000 // config.batch_size


def train_cliffordunet_draft(
    config: TrainingConfig,
) -> Tuple[CliffordUNetDraftLM, keras.callbacks.History]:
    logger.info("=" * 60)
    logger.info("CliffordUNet DFlash Draft Training (block-masked denoising)")
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
        logger.info(f"Resuming from checkpoint: {config.resume_from}")
        model = keras.models.load_model(
            config.resume_from,
            custom_objects={
                "CliffordUNetDraftLM": CliffordUNetDraftLM,
                "CliffordUNetDraftModel": CliffordUNetDraftModel,
                "CliffordNetBlock": CliffordNetBlock,
                "CausalWindowPool": CausalWindowPool,
                "MultiLinearUpsample": MultiLinearUpsample,
                "MaskedCausalLMLoss": MaskedCausalLMLoss,
            },
        )
        import re
        m = re.search(r"step_(\d+)", os.path.basename(config.resume_from))
        initial_step = int(m.group(1)) if m else 0
    else:
        model = create_model(config)

    compile_model(model, config, steps_per_epoch)

    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"CliffordUNetDraft-{config.variant}",
        results_dir_prefix="cliffordunet_draft",
        include_analyzer=False,
    )
    callbacks = [
        cb for cb in callbacks
        if not isinstance(cb, keras.callbacks.CSVLogger)
    ]
    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        save_every_steps=config.checkpoint_every_steps,
        max_checkpoints=config.max_checkpoints,
        initial_step=initial_step,
    ))

    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch~={steps_per_epoch:,}, "
        f"batch_size={config.batch_size}, block_size={config.block_size}"
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
    return model, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CliffordUNet DFlash-style Draft Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument(
        "--variant", type=str, default="draft_nano",
        choices=list(CliffordUNetDraftModel.MODEL_VARIANTS.keys()) + ["custom"],
    )
    p.add_argument("--target-hidden-size", type=int, default=1024,
                   help="Draft operates at this dimension (≈ verifier hidden size)")
    p.add_argument("--num-target-layers", type=int, default=0,
                   help="0 disables cross-conditioning during standalone training")
    p.add_argument("--channels", type=str, default="256,384,512")
    p.add_argument("--encoder-depths", type=str, default="2,4,2")
    p.add_argument("--decoder-depths", type=str, default="2,2")
    p.add_argument("--pool-sizes", type=str, default="4,4")
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

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--dropout-rate", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

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

    p.add_argument("--checkpoint-every-steps", type=int, default=10000)
    p.add_argument("--max-checkpoints", type=int, default=3)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="results/cliffordunet_draft")
    return p


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        variant=args.variant,
        target_hidden_size=args.target_hidden_size,
        num_target_layers=args.num_target_layers,
        channels=_parse_int_list(args.channels),
        encoder_depths=_parse_int_list(args.encoder_depths),
        decoder_depths=_parse_int_list(args.decoder_depths),
        pool_sizes=_parse_int_list(args.pool_sizes),
        shifts=_parse_int_list(args.shifts),
        cli_mode=args.cli_mode,
        ctx_mode=args.ctx_mode,
        use_global_context=args.use_global_context,
        stochastic_depth_rate=args.stochastic_depth_rate,
        dropout_rate=args.dropout_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        hf_cache_dir=args.hf_cache_dir,
        max_train_samples=args.max_train_samples,
        val_fraction=args.val_fraction,
        checkpoint_every_steps=args.checkpoint_every_steps,
        max_checkpoints=args.max_checkpoints,
        resume_from=args.resume,
        save_dir=args.save_dir,
    )


def main() -> None:
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.variant}, D_target={config.target_hidden_size}, "
        f"block_size={config.block_size}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, epochs={config.num_epochs}, "
        f"source={config.dataset_source}"
    )

    train_cliffordunet_draft(config)


if __name__ == "__main__":
    main()
