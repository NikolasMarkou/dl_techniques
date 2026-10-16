"""
CLIP Training & Inference - stock ``compile``/``fit``.

Implements contrastive language-image pre-training with CLIP-specific metrics
(accuracy, recall@k) and zero-shot retrieval inference.

Entry point: ``python -m train.clip.train_clip`` with either ``--synthetic``
(random image/token pairs generated in memory) or ``--image-dir`` plus
``--captions-file``. A run writes ``config.json``, ``training_log.csv``,
``training_history.json`` and ``final_model.keras`` (the CLIP, rewritten after
every epoch) to ``--output-dir`` (default
``results/clip_<variant>_<timestamp>``). The model is ``create_clip(--variant)``
with the size flags as overrides, trained by stock ``keras.Model.fit`` through
`compile_for_contrastive_training`.

Reference: Radford et al. (2021) "Learning Transferable Visual Models from
Natural Language Supervision" https://arxiv.org/abs/2103.00020
"""

import argparse
import collections
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence
import numpy as np
import keras
import tensorflow as tf

from train.common import (
    setup_gpu,
    set_seeds,
    default_experiment_name,
    save_config_json,
    save_training_history_json,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from train.common.callbacks import EpochLogLine, LearningRateLogger
from train.common.contrastive import (
    as_contrastive_fit_batches,
    compile_for_contrastive_training,
)
from dl_techniques.utils.logger import logger
from dl_techniques.optimization import (
    optimizer_builder,
    learning_rate_schedule_builder
)
from dl_techniques.models.vision_language.clip.model import CLIP, create_clip


# ---------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------

def build_caption_vocabulary(captions: List[str], vocab_size: int) -> Dict[str, int]:
    """Word-level vocabulary over the training captions, for `CLIPDataLoader`.

    Interface contract (callers: `main`, tests): ids 0 and 1 are ``"<pad>"``
    and ``"<unk>"``; pad must be 0 because CLIP pools the last NON-ZERO token.
    The other ``vocab_size - 2`` ids go to the most frequent lower-cased
    whitespace-split words, ties broken alphabetically, so the mapping is
    deterministic. The words are split exactly as
    `CLIPDataLoader.preprocess_text` splits them.

    Args:
        captions: Training captions.
        vocab_size: The model's vocabulary size; every id is below it.

    Returns:
        ``{word: id}`` with at most ``vocab_size`` entries.

    Raises:
        ValueError: If ``vocab_size < 3`` (no room for one word).
    """
    if vocab_size < 3:
        raise ValueError(f"vocab_size must be at least 3, got {vocab_size}")
    counts = collections.Counter(
        word for caption in captions for word in caption.lower().split())
    words = sorted(counts, key=lambda word: (-counts[word], word))
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({word: i + 2 for i, word in enumerate(words[:vocab_size - 2])})
    return vocab


class CLIPDataLoader:
    """Data loader for CLIP training with image preprocessing and text tokenization.

    Args:
        image_size: Size to resize images to (square).
        context_length: Maximum length for text sequences.
        tokenizer: A ``{word: id}`` vocabulary with ``"<pad>"`` at 0 and
            ``"<unk>"`` at 1 (see `build_caption_vocabulary`), or an object
            with an ``encode(text) -> List[int]`` method that pads with 0.
        image_normalization: Normalization strategy ('imagenet', 'clip', or None).
        augmentation: Whether to apply data augmentation during training.
    """

    def __init__(
        self,
        image_size: int = 224,
        context_length: int = 77,
        tokenizer: Optional[Any] = None,
        image_normalization: str = 'clip',
        augmentation: bool = False
    ) -> None:
        self.image_size = image_size
        self.context_length = context_length
        # DECISION plan-2026-10-14T062616-81938e81/D-034: no default tokenizer.
        # The old default was a demo vocabulary of "word_0".."word_999" that
        # matched no real word, so every caption tokenized to the same <unk>
        # run and the text tower saw one input for the whole dataset. Do not
        # bring back a vocabulary that is not built from the captions.
        # See decisions.md D-034.
        if tokenizer is None:
            raise ValueError(
                "CLIPDataLoader needs a tokenizer: build_caption_vocabulary("
                "captions, vocab_size), or an object with encode(text)"
            )
        self.tokenizer = tokenizer
        self.image_normalization = image_normalization
        self.augmentation = augmentation
        logger.info(
            f"CLIPDataLoader: image_size={image_size}, context_length={context_length}, "
            f"normalization={image_normalization}, augmentation={augmentation}"
        )

    def preprocess_image(self, image_path: str, training: bool = True) -> tf.Tensor:
        """Load and preprocess image from file path."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)

        if training and self.augmentation:
            image = self._augment_image(image)

        image = tf.image.resize(
            image, (self.image_size, self.image_size),
            method='bilinear', antialias=True
        )

        if self.image_normalization == 'imagenet':
            # DECISION plan_2026-06-02_35651564/D-002: reference the shared
            # IMAGENET_MEAN/STD constants instead of re-hardcoding the literals.
            # Pure literal swap inside the 'imagenet' branch ONLY — do NOT
            # touch the 'clip' default branch or any other CLIPDataLoader logic
            # (that surface is F13-excluded). Values are bit-equal to the
            # original [0.485,0.456,0.406] / [0.229,0.224,0.225].
            mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
            std = tf.constant(IMAGENET_STD, dtype=tf.float32)
            image = (image / 255.0 - mean) / std
        elif self.image_normalization == 'clip':
            image = image / 255.0

        return image

    def _augment_image(self, image: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation to image."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        if image.shape[-1] == 3:
            image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
            image = tf.image.random_hue(image, max_delta=0.05)
        return tf.clip_by_value(image, 0.0, 255.0)

    def preprocess_text(self, text: str) -> tf.Tensor:
        """Tokenize text into token IDs of shape (context_length,)."""
        if isinstance(self.tokenizer, dict):
            words = text.lower().split()[:self.context_length - 2]
            token_ids = [self.tokenizer.get(word, 1) for word in words]
        else:
            token_ids = self.tokenizer.encode(text)[:self.context_length]

        # Pad or truncate
        token_ids = token_ids[:self.context_length]
        if len(token_ids) < self.context_length:
            token_ids = token_ids + [0] * (self.context_length - len(token_ids))

        return tf.constant(token_ids, dtype=tf.int32)

    def create_dataset(
        self,
        image_paths: List[str],
        captions: List[str],
        batch_size: int = 32,
        shuffle: bool = True,
        buffer_size: Optional[int] = None,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch_size: int = tf.data.AUTOTUNE,
        drop_remainder: bool = True
    ) -> tf.data.Dataset:
        """Create batched tf.data.Dataset for CLIP training."""
        if len(image_paths) != len(captions):
            raise ValueError(
                f"Number of images ({len(image_paths)}) must match captions ({len(captions)})"
            )

        logger.info(f"Creating dataset: {len(image_paths)} samples, batch_size={batch_size}")

        # Tokenized here, eagerly: `preprocess_text` is Python string code, and
        # inside `dataset.map` a caption is a symbolic tf.string tensor
        # (`'SymbolicTensor' object has no attribute 'lower'`).
        token_ids = np.stack([self.preprocess_text(c).numpy() for c in captions])
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, token_ids))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size or len(image_paths))

        def preprocess_fn(img_path, text):
            image = self.preprocess_image(img_path, training=shuffle)
            return image, text

        dataset = dataset.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(prefetch_size)
        return dataset

# ---------------------------------------------------------------------
# Training: stock compile/fit through the shared contrastive bridge
# ---------------------------------------------------------------------
#
# `StackedCLIPLogits`, its loss and metrics, `compile_for_contrastive_training`
# and `as_contrastive_fit_batches` live in `train.common.contrastive` (D-036,
# moved there by D-086 so the CliffordCLIP trainer does not import this one).


def build_clip_optimizer(learning_rate: float, warmup_steps: int) -> keras.optimizers.Optimizer:
    """AdamW on the trainer's linear-warmup cosine schedule, global clip 1.0.

    The clip is the optimizer's own ``global_clipnorm``. Under a
    ``mixed_float16`` policy `keras.Model.compile` wraps the optimizer in a
    ``LossScaleOptimizer``, whose ``apply`` unscales the gradients before the
    inner optimizer clips them, so the bound applies to the true gradient norm.
    """
    lr_schedule = learning_rate_schedule_builder({
        "type": "cosine_decay", "warmup_steps": warmup_steps,
        "warmup_start_lr": 1e-8, "learning_rate": learning_rate,
        "decay_steps": 50000, "alpha": 0.0001
    })
    return optimizer_builder({
        "type": "adamw", "beta_1": 0.9, "beta_2": 0.999,
        "gradient_clipping_by_norm": 1.0
    }, lr_schedule)


def refuse_an_empty_dataset(dataset: tf.data.Dataset) -> None:
    """Raise if ``dataset`` yields no batch.

    With ``drop_remainder`` a dataset smaller than one batch is empty, and the
    old loop logged a loss of exactly 0.0 for it. The cardinality decides when
    tf.data knows it; otherwise one batch is drawn.

    Raises:
        ValueError: If the dataset is empty.
    """
    cardinality = int(dataset.cardinality())
    if cardinality == int(tf.data.UNKNOWN_CARDINALITY):
        empty = next(iter(dataset.take(1)), None) is None
    else:
        empty = cardinality == 0
    if empty:
        raise ValueError(
            "the training dataset yields no batch; with drop_remainder a "
            "dataset smaller than one batch is empty"
        )


# ---------------------------------------------------------------------
# Inference Utilities
# ---------------------------------------------------------------------

class CLIPInference:
    """Inference utilities for trained CLIP models.

    Provides image/text encoding, similarity computation, and zero-shot retrieval.

    Args:
        model: Trained CLIP model instance.
        data_loader: Data loader instance for preprocessing.
        batch_size: Batch size for inference.
    """

    def __init__(self, model: Any, data_loader: CLIPDataLoader, batch_size: int = 64) -> None:
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size

    def encode_images(self, image_paths: List[str], normalize: bool = True, show_progress: bool = True) -> np.ndarray:
        """Encode images to embeddings of shape (num_images, embed_dim)."""
        embeddings_list = []
        num_images = len(image_paths)
        logger.info(f"Encoding {num_images} images...")

        for i in range(0, num_images, self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            for path in batch_paths:
                try:
                    image = self.data_loader.preprocess_image(path, training=False)
                    batch_images.append(image.numpy())
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")
                    batch_images.append(
                        np.zeros((self.data_loader.image_size, self.data_loader.image_size, 3))
                    )

            batch_tensor = keras.ops.convert_to_tensor(np.array(batch_images))
            batch_embeddings = self.model.encode_image(batch_tensor, training=False)
            embeddings_list.append(keras.ops.convert_to_numpy(batch_embeddings))

            if show_progress and (i // self.batch_size) % 10 == 0:
                logger.info(f"  Processed {i + len(batch_paths)}/{num_images} images")

        embeddings = np.vstack(embeddings_list)
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def encode_texts(self, texts: List[str], normalize: bool = True, show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings of shape (num_texts, embed_dim)."""
        embeddings_list = []
        num_texts = len(texts)
        logger.info(f"Encoding {num_texts} texts...")

        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = []
            for text in batch_texts:
                try:
                    tokens = self.data_loader.preprocess_text(text)
                    batch_tokens.append(tokens.numpy())
                except Exception as e:
                    logger.error(f"Error tokenizing text '{text[:50]}...': {e}")
                    batch_tokens.append(np.zeros(self.data_loader.context_length, dtype=np.int32))

            batch_tensor = keras.ops.convert_to_tensor(np.array(batch_tokens))
            batch_embeddings = self.model.encode_text(batch_tensor, training=False)
            embeddings_list.append(keras.ops.convert_to_numpy(batch_embeddings))

            if show_progress and (i // self.batch_size) % 10 == 0:
                logger.info(f"  Processed {i + len(batch_texts)}/{num_texts} texts")

        embeddings = np.vstack(embeddings_list)
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def compute_similarity(self, image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between image and text embeddings."""
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        return np.dot(image_embeddings, text_embeddings.T)

    def retrieve_images(self, query_text: str, image_paths: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k images for a text query."""
        text_embedding = self.encode_texts([query_text], show_progress=False)
        image_embeddings = self.encode_images(image_paths, show_progress=True)
        similarities = self.compute_similarity(image_embeddings, text_embedding)
        scores = similarities[:, 0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(image_paths[i], float(scores[i])) for i in top_indices]
        for rank, (path, score) in enumerate(results, 1):
            logger.info(f"  {rank}. {Path(path).name}: {score:.4f}")
        return results

    def retrieve_texts(self, query_image_path: str, texts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k texts for an image query."""
        image_embedding = self.encode_images([query_image_path], show_progress=False)
        text_embeddings = self.encode_texts(texts, show_progress=True)
        similarities = self.compute_similarity(image_embedding, text_embeddings)
        scores = similarities[0, :]
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(texts[i], float(scores[i])) for i in top_indices]
        for rank, (text, score) in enumerate(results, 1):
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.info(f"  {rank}. '{text_preview}': {score:.4f}")
        return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

#: CLI flags that override the variant's constructor value when given.
_MODEL_SIZE_FLAGS = (
    "vision_layers", "vision_width", "vision_heads", "vision_kv_heads",
    "text_layers", "text_width", "text_heads", "text_kv_heads",
    "embed_dim", "vocab_size", "ffn_multiple_of",
)


def make_synthetic_pairs(
    num_samples: int,
    image_size: int,
    context_length: int,
    vocab_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random image/caption pairs, generated in memory.

    Every caption has its own random length and token ids in
    ``[2, vocab_size)``, right-padded with 0 (CLIP's pad id), so no two pairs
    share a text input and the contrastive task is not degenerate.

    Args:
        num_samples: Number of pairs.
        image_size: Square image edge; pixels are uniform in ``[0, 1)``, the
            range the ``'clip'`` normalization of `CLIPDataLoader` produces.
        context_length: Token sequence length.
        vocab_size: The model's vocabulary size.
        seed: Seed of the generator.

    Returns:
        ``(images, tokens)``: float32 ``(N, S, S, 3)`` and int32 ``(N, L)``.
    """
    rng = np.random.default_rng(seed)
    images = rng.random((num_samples, image_size, image_size, 3), dtype=np.float32)
    tokens = np.zeros((num_samples, context_length), dtype=np.int32)
    lengths = rng.integers(1, context_length + 1, size=num_samples)
    for i, length in enumerate(lengths):
        tokens[i, :length] = rng.integers(2, vocab_size, size=length)
    return images, tokens


def _variant_slug(variant: str) -> str:
    """``"ViT-B/16"`` -> ``"vit_b_16"``, for a run directory name."""
    return variant.lower().replace("/", "_").replace("-", "_")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the CLIP trainer's command line.

    Exactly one data source: ``--synthetic`` or ``--image-dir`` with
    ``--captions-file``.
    """
    parser = argparse.ArgumentParser(description="Train CLIP model")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device index")
    parser.add_argument("--variant", type=str, default="ViT-B/16",
                        choices=sorted(CLIP.MODEL_VARIANTS),
                        help="CLIP.MODEL_VARIANTS row passed to create_clip")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--context-length", type=int, default=77)
    for name in _MODEL_SIZE_FLAGS:
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=None,
            help=f"override the variant's {name}")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="cap on training steps per epoch (default: one full pass)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                        help="linear warmup steps of the cosine schedule")
    parser.add_argument("--seed", type=int, default=42)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true",
                        help="train on random image/caption pairs generated in memory")
    source.add_argument("--image-dir", type=str, default=None,
                        help="Directory with *.jpg / *.png images (sorted by name)")
    parser.add_argument("--captions-file", type=str, default=None,
                        help="Captions file, one per line, in sorted image order")
    parser.add_argument("--synthetic-samples", type=int, default=64,
                        help="number of pairs generated by --synthetic")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="run directory (default: results/clip_<variant>_<timestamp>)")
    args = parser.parse_args(argv)
    if args.image_dir and not args.captions_file:
        parser.error("--image-dir requires --captions-file")
    if args.captions_file and not args.image_dir:
        parser.error("--captions-file requires --image-dir")
    return args


def main(argv: Optional[Sequence[str]] = None) -> Path:
    """Main entry point for CLIP training; returns the run directory."""
    args = parse_arguments(argv)

    setup_gpu(gpu_id=args.gpu)
    set_seeds(args.seed)

    run_dir = (
        Path(args.output_dir) if args.output_dir
        else Path("results") / default_experiment_name("clip", _variant_slug(args.variant))
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config_json(vars(args), str(run_dir))

    overrides = {
        name: getattr(args, name) for name in _MODEL_SIZE_FLAGS
        if getattr(args, name) is not None
    }
    model = create_clip(
        args.variant, image_size=args.image_size,
        context_length=args.context_length, **overrides,
    )
    # Through the model's own build, which materializes every weight (the
    # same population a forward pass creates); count_params raises on an
    # unbuilt model.
    model.build({
        'image': (None, args.image_size, args.image_size, 3),
        'text': (None, args.context_length),
    })
    logger.info(f"Created CLIP {args.variant} model: {model.count_params():,} parameters")

    wrapper = compile_for_contrastive_training(
        model, build_clip_optimizer(args.learning_rate, args.warmup_steps))

    if args.synthetic:
        images, tokens = make_synthetic_pairs(
            num_samples=args.synthetic_samples, image_size=args.image_size,
            context_length=args.context_length, vocab_size=model.vocab_size,
            seed=args.seed,
        )
        logger.info(f"Synthetic data: {len(images)} image/caption pairs")
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((images, tokens))
            .shuffle(len(images), seed=args.seed)
            .batch(args.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        import glob
        image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")) +
                             glob.glob(os.path.join(args.image_dir, "*.png")))
        with open(args.captions_file, 'r') as f:
            captions = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(image_paths)} images and {len(captions)} captions")
        data_loader = CLIPDataLoader(
            image_size=args.image_size, context_length=args.context_length,
            tokenizer=build_caption_vocabulary(captions, model.vocab_size),
        )
        train_dataset = data_loader.create_dataset(
            image_paths, captions, batch_size=args.batch_size,
        )

    refuse_an_empty_dataset(train_dataset)
    train_dataset = as_contrastive_fit_batches(train_dataset)
    if args.steps_per_epoch:
        # `fit` with steps_per_epoch keeps ONE iterator across epochs and resets
        # it only once it runs dry, so an epoch that starts mid-pass ends short
        # (MEASURED, Keras 3.8: 3 batches at 2 steps per epoch train [2, 1, 2]
        # with a "ran out of data" warning). Repeated, every epoch has its steps.
        train_dataset = train_dataset.repeat()

    final_model_path = str(run_dir / "final_model.keras")
    callbacks = [
        # First: `lr` must be in `logs` before CSVLogger writes the row; read
        # at the epoch start, so it is the rate the epoch trained at.
        LearningRateLogger(at_epoch_start=True),
        keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        # The CLIP, not the training wrapper, after every epoch: the file is
        # the last completed epoch even if the run dies later.
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: model.save(final_model_path)),
        EpochLogLine(("loss", "clip_i2t_accuracy", "clip_i2t_recall@5")),
    ]

    logger.info(f"Starting CLIP training: {args.epochs} epochs, batch_size={args.batch_size}")
    history = wrapper.fit(
        train_dataset, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        callbacks=callbacks,
    )

    save_training_history_json(history, run_dir)
    logger.info(f"Run directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
