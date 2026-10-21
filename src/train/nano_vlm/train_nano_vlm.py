"""Train NanoVLM on image/text batches through stock ``compile``/``fit``.

The model is a ``NanoVLM.MODEL_VARIANTS`` row, built by ``create_nanovlm``
with optional size overrides, and compiled with ``NanoVLMLoss``: next-token
cross-entropy over the text positions of the model's ``[vision; text]``
logits (one causal decoder over the projected image tokens and the text),
padding (id 0) ignored. There is no custom training step.

Exactly one data source:

* ``--synthetic``: random images and token sequences generated in memory
  (``BOS``, random content, ``EOS``, right padding), for smoke runs and for
  a memorization check;
* ``--data-json``: a Cauldron-format JSON list of
  ``{'image_path', 'question', 'answer'}`` rows, read by
  ``dl_techniques.datasets.vqa_dataset``.

A run directory (``--output-dir``, default ``results/nanovlm_<variant>_<ts>``)
receives ``config.json``, ``training_log.csv``, ``training_history.json`` and
``final_model.keras`` (rewritten after every epoch).

Example::

    MPLBACKEND=Agg .venv/bin/python -m train.nano_vlm.train_nano_vlm \\
        --synthetic --variant mini --epochs 1 --steps-per-epoch 10
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    setup_gpu,
    setup_mixed_precision,
    set_seeds,
    default_experiment_name,
    save_config_json,
    save_training_history_json,
)
from train.common.callbacks import EpochCSVLogger, EpochLogLine, LearningRateLogger
from dl_techniques.utils.logger import logger
from dl_techniques.losses.nano_vlm_loss import NanoVLMLoss
from dl_techniques.models.vision_language.nano_vlm.model import NanoVLM, create_nanovlm
from dl_techniques.datasets.vqa_dataset import (
    VQADataSequence,
    VQADataProcessor,
    create_vqa_dataset,
    load_cauldron_from_json,
)
from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

# ---------------------------------------------------------------------

#: Token ids shared with ``VQADataProcessor``'s defaults: padding (ignored
#: by the loss), beginning and end of sequence.
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


def build_nanovlm_optimizer(
        learning_rate: float,
        warmup_steps: int,
        decay_steps: int,
        clipnorm: float,
) -> keras.optimizers.Optimizer:
    """AdamW on a linear-warmup cosine schedule with a global-norm clip.

    Under a ``mixed_float16`` policy `keras.Model.compile` wraps the optimizer
    in a ``LossScaleOptimizer``, which unscales the gradients before this
    optimizer clips them, so ``clipnorm`` bounds the true gradient norm.
    """
    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay", "warmup_steps": warmup_steps,
        "warmup_start_lr": 1e-9, "learning_rate": learning_rate,
        "decay_steps": decay_steps, "alpha": 0.0001
    })
    return optimizer_builder({
        "type": "adamw", "beta_1": 0.9, "beta_2": 0.999,
        "gradient_clipping_by_norm": clipnorm
    }, lr_schedule)


def compile_nanovlm(
        model: NanoVLM,
        optimizer: keras.optimizers.Optimizer,
        label_smoothing: float,
) -> NanoVLM:
    """Compile ``model`` for stock ``fit`` with the next-token loss.

    Interface contract (callers: `main`, tests): returns ``model`` itself,
    compiled with ``NanoVLMLoss(ignore_index=PAD_ID, from_logits=True)``. Feed
    its ``fit`` ``(inputs, labels)`` batches where ``inputs`` holds
    ``'images'``, ``'text_tokens'`` and ``'attention_mask'`` and ``labels``
    are the text tokens themselves (the loss shifts them by one).
    """
    model.compile(
        optimizer=optimizer,
        loss=NanoVLMLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing,
                         from_logits=True),
    )
    return model


def make_synthetic_samples(
        num_samples: int,
        image_size: int,
        seq_len: int,
        vocab_size: int,
        seed: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Random image/token-sequence samples, generated in memory.

    Each sequence is ``BOS``, a random-length run of content ids in
    ``[3, vocab_size)``, ``EOS``, then ``PAD`` to ``seq_len``, the layout
    ``VQADataProcessor`` produces. No two samples share a sequence, so the
    task is not degenerate.

    Args:
        num_samples: Number of samples.
        image_size: Square image edge; pixels are uniform in ``[-1, 1)``, the
            range of the processor's default ``'siglip'`` normalization.
        seq_len: Token sequence length, at least 3.
        vocab_size: The model's vocabulary size, at least 4.
        seed: Seed of the generator.

    Returns:
        ``(inputs, labels)``: ``inputs`` holds float32 ``images``
        ``(N, S, S, 3)`` and int32 ``text_tokens`` / ``attention_mask``
        ``(N, L)``; ``labels`` is ``text_tokens``.

    Raises:
        ValueError: If ``seq_len < 3`` or ``vocab_size < 4``.
    """
    if seq_len < 3 or vocab_size < 4:
        raise ValueError(
            f"need seq_len >= 3 and vocab_size >= 4 for BOS, content, EOS; "
            f"got seq_len={seq_len}, vocab_size={vocab_size}"
        )
    rng = np.random.default_rng(seed)
    images = rng.uniform(-1.0, 1.0, (num_samples, image_size, image_size, 3)).astype(np.float32)
    tokens = np.full((num_samples, seq_len), PAD_ID, dtype=np.int32)
    # 1 .. seq_len - 2 content tokens, so BOS and EOS always fit.
    content_lengths = rng.integers(1, seq_len - 1, size=num_samples)
    for i, length in enumerate(content_lengths):
        tokens[i, 0] = BOS_ID
        tokens[i, 1:length + 1] = rng.integers(3, vocab_size, size=length)
        tokens[i, length + 1] = EOS_ID
    inputs = {
        'images': images,
        'text_tokens': tokens,
        'attention_mask': (tokens != PAD_ID).astype(np.int32),
    }
    return inputs, tokens


def sequence_as_dataset(
        sequence: VQADataSequence,
        image_size: int,
        seq_len: int,
) -> tf.data.Dataset:
    """One pass over ``sequence``'s batches as a ``tf.data`` pipeline.

    ``fit`` resets a finite ``PyDataset`` only once it runs dry, exactly as it
    does a finite ``tf.data`` dataset (MEASURED, Keras 3.8: 3 batches at 2
    steps per epoch train ``[2, 1, 2]`` with a "ran out of data" warning), so
    the file path goes through ``tf.data`` too and can be repeated like the
    synthetic one. The batches stay lazy: images are read per batch.
    """
    def batches():
        # DECISION plan-2026-08-18T140459-7991552f/D-009
        # Index by `range(steps_per_epoch)`. Do NOT write
        # `for step, batch in enumerate(train_dataset)`. `create_vqa_dataset`
        # returns a `VQADataSequence(keras.utils.Sequence)`, and in Keras 3
        # `keras.utils.Sequence` IS `PyDataset`, which defines
        # `__getitem__`/`__len__` and no `__iter__`. `enumerate` therefore
        # fell back to Python's legacy `__getitem__` protocol, which walks
        # 0, 1, 2, ... until `IndexError` and IGNORES `__len__` entirely.
        # `VQADataSequence.__getitem__` serves any index and never raises
        # `IndexError`, so epoch 0 never ended: a hang with no failure text.
        # Same mechanism, same repair as the D-034 anchor in
        # `src/train/hrm/train_hrm.py`'s `train_epoch`.
        for index in range(len(sequence)):
            yield sequence[index]
        sequence.on_epoch_end()

    text_spec = tf.TensorSpec((None, seq_len), tf.int32)
    signature = (
        {
            'images': tf.TensorSpec((None, image_size, image_size, 3), tf.float32),
            'text_tokens': text_spec,
            'attention_mask': text_spec,
        },
        text_spec,
    )
    # A generator's cardinality is unknown, and `fit` then warns "ran out of
    # data" at the end of every pass; the sequence knows its length.
    return tf.data.Dataset.from_generator(batches, output_signature=signature).apply(
        tf.data.experimental.assert_cardinality(len(sequence)))


def size_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """The ``vision_config``/``text_config`` keys the size flags set, in the
    one-level merge form ``NanoVLM.from_variant`` takes. Flags left at
    ``None`` keep the row's value."""
    vision: Dict[str, Any] = {}
    text: Dict[str, Any] = {}
    if args.image_size is not None:
        vision['img_size'] = args.image_size
    if args.patch_size is not None:
        vision['patch_size'] = args.patch_size
    if args.embed_dim is not None:
        vision['embed_dim'] = text['embed_dim'] = args.embed_dim
    if args.num_heads is not None:
        vision['num_heads'] = text['num_heads'] = args.num_heads
    if args.vision_depth is not None:
        vision['depth'] = args.vision_depth
    if args.text_depth is not None:
        text['depth'] = args.text_depth
    if args.max_text_length is not None:
        text['max_seq_len'] = args.max_text_length
    return {'vision_config': vision, 'text_config': text}


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the NanoVLM trainer's command line.

    Exactly one data source: ``--synthetic`` or ``--data-json``.
    """
    parser = argparse.ArgumentParser(description="Train NanoVLM")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--variant", type=str, default="base",
                        choices=sorted(NanoVLM.MODEL_VARIANTS),
                        help="NanoVLM.MODEL_VARIANTS row passed to create_nanovlm")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--image-size", type=int, default=None, help="override img_size")
    parser.add_argument("--patch-size", type=int, default=None, help="override patch_size")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="override the vision encoder's and the decoder's embed_dim")
    parser.add_argument("--num-heads", type=int, default=None,
                        help="override the vision encoder's and the decoder's heads")
    parser.add_argument("--vision-depth", type=int, default=None)
    parser.add_argument("--text-depth", type=int, default=None,
                        help="decoder blocks (they run over the vision prefix and the text)")
    parser.add_argument("--max-text-length", type=int, default=None,
                        help="text sequence length (default: the row's max_seq_len)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="cap on training steps per epoch (default: one full pass)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="peak of the warmup-cosine schedule")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--decay-steps", type=int, default=25000)
    parser.add_argument("--clipnorm", type=float, default=0.5,
                        help="global gradient-norm clip")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction,
                        default=True, help="mixed_float16 global policy")
    parser.add_argument("--seed", type=int, default=42)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true",
                        help="train on random image/text samples generated in memory")
    source.add_argument("--data-json", type=str, default=None,
                        help="Cauldron-format JSON list of image_path/question/answer rows")
    parser.add_argument("--synthetic-samples", type=int, default=64,
                        help="number of samples generated by --synthetic")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="run directory (default: results/nanovlm_<variant>_<timestamp>)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """Train NanoVLM end to end; returns the run directory."""
    args = parse_arguments(argv)

    setup_gpu(gpu_id=args.gpu)
    setup_mixed_precision(args.mixed_precision)
    set_seeds(args.seed)

    run_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("results") / default_experiment_name("nanovlm", args.variant)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(vars(args), str(run_dir))

    model = create_nanovlm(
        args.variant, vocab_size=args.vocab_size, **size_overrides(args),
    )
    image_size = model.vision_config['img_size']
    seq_len = model.text_config['max_seq_len']

    if args.synthetic:
        num_samples = args.synthetic_samples
    else:
        samples = load_cauldron_from_json(args.data_json)
        num_samples = len(samples)

    # DECISION plan-2026-10-14T062616-81938e81/D-038: refuse a run with no
    # batch BEFORE building anything. Every batch here is a full one (tf.data
    # with drop_remainder, `VQADataSequence.__len__` floors), so fewer samples
    # than one batch would train ZERO steps and save an untrained model with
    # no error. Do not round the batch count up or shrink the batch silently;
    # the caller chose both numbers and is told. See decisions.md D-038.
    batches_per_pass = num_samples // args.batch_size
    if batches_per_pass == 0:
        raise ValueError(
            f"no full batch: {num_samples} samples // batch_size="
            f"{args.batch_size} == 0, so every epoch would train zero steps. "
            f"Use batch_size <= {num_samples}, or supply more samples."
        )

    # DECISION plan-2026-10-14T062616-81938e81/D-038: build through the
    # model's own build() BEFORE count_params (which raises on an unbuilt
    # subclassed model) and with the data's own shapes. See decisions.md D-038.
    model.build({'images': (None, image_size, image_size, 3),
                 'text_tokens': (None, seq_len)})
    # DECISION plan-2026-08-22T035419-a11304c8/D-008
    # The label is DERIVED, never written down. This line read
    # "Created nanoVLM-222M" while printing the real count beside it; the
    # model `create_nanovlm()` builds measured 305,435,904 parameters at the
    # default `variant="base"` and this trainer's 224px / 512-token build on
    # 2026-08-22, and 280,859,904 once the dead `shared_output_projection`
    # went on 2026-10-15 -- never the "222M" design target carried over from
    # `research/nanoVLM_research.md`. Do not replace this with a hardcoded
    # count: a second literal is the same defect one refactor later, and the
    # count moves with `variant` and the size flags.
    n_params = model.count_params()
    logger.info(f"Created nanoVLM-{n_params / 1e6:.0f}M: {n_params:,} parameters")

    compile_nanovlm(
        model,
        build_nanovlm_optimizer(args.learning_rate, args.warmup_steps,
                                args.decay_steps, args.clipnorm),
        args.label_smoothing,
    )

    if args.synthetic:
        inputs, labels = make_synthetic_samples(
            num_samples, image_size, seq_len, model.vocab_size, args.seed)
        logger.info(f"Synthetic data: {num_samples} image/text samples")
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((inputs, labels))
            .shuffle(num_samples, seed=args.seed)
            .batch(args.batch_size, drop_remainder=True)
        )
    else:
        processor = VQADataProcessor(
            image_size=image_size, max_text_length=seq_len,
            vocab_size=model.vocab_size, pad_token_id=PAD_ID,
            bos_token_id=BOS_ID, eos_token_id=EOS_ID,
        )
        sequence = create_vqa_dataset(samples, processor,
                                      batch_size=args.batch_size, shuffle=True)
        train_dataset = sequence_as_dataset(sequence, image_size, seq_len)
    if args.steps_per_epoch:
        # `fit` with steps_per_epoch keeps ONE iterator across epochs and resets
        # it only once it runs dry, so an epoch that starts mid-pass ends short.
        # Repeated, every epoch has its steps.
        train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    callbacks = [
        # First: `lr` must be in `logs` before CSVLogger writes the row; read
        # at the epoch start, so it is the rate the epoch trained at.
        LearningRateLogger(at_epoch_start=True),
        EpochCSVLogger(str(run_dir / "training_log.csv")),
        # After every epoch, so the file is the last completed epoch even if
        # the run dies later.
        keras.callbacks.ModelCheckpoint(str(run_dir / "final_model.keras")),
        EpochLogLine(("loss",)),
    ]

    logger.info(
        f"Starting NanoVLM training: {args.epochs} epochs, batch_size={args.batch_size}, "
        f"steps/epoch={args.steps_per_epoch or batches_per_pass}"
    )
    history = model.fit(
        train_dataset, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
    )

    save_training_history_json(history, run_dir)
    logger.info(f"Run directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
