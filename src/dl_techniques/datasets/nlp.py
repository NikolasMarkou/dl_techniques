"""NLP dataset loaders for text-based training pipelines.

Provides loaders for Wikipedia and other HuggingFace text datasets,
returning ``tf.data.Dataset`` of raw text strings compatible with
the CLM/MLM preprocessing in ``train.common.nlp``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import datasets
import tensorflow as tf

from dl_techniques.utils.logger import logger

# Default cache directory for Wikipedia dataset
DEFAULT_WIKIPEDIA_CACHE_DIR = "/media/arxwn/data0_4tb/datasets/wikipedia"
DEFAULT_WIKIPEDIA_CONFIG = "20231101.en"


def load_wikipedia_train_val(
    cache_dir: str = DEFAULT_WIKIPEDIA_CACHE_DIR,
    config_name: str = DEFAULT_WIKIPEDIA_CONFIG,
    min_article_length: int = 500,
    val_fraction: float = 0.02,
    max_val_samples: int = 5000,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load Wikipedia with a proper random holdout train/val split.

    Uses HuggingFace ``train_test_split()`` which keeps data on disk
    in Arrow format (memory-mapped). No need to load all articles into
    RAM. Generators stream text from disk on demand.

    :param cache_dir: Local directory for Arrow cache files.
    :param config_name: Wikipedia config (e.g. ``'20231101.en'``).
    :param min_article_length: Skip articles shorter than this (chars).
    :param val_fraction: Fraction of articles reserved for validation.
    :param max_val_samples: Max validation articles.
    :param max_train_samples: Limit training articles. ``None`` for all.
    :param seed: Random seed for reproducible splits.
    :return: ``(train_dataset, val_dataset)`` — both yield raw text strings.
    """
    logger.info(
        f"Loading Wikipedia ({config_name}) with holdout split "
        f"(val_fraction={val_fraction}, seed={seed})"
    )

    try:
        hf_dataset = datasets.load_dataset(
            "wikimedia/wikipedia",
            config_name,
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except TypeError:
        hf_dataset = datasets.load_dataset(
            "wikimedia/wikipedia",
            config_name,
            split="train",
            cache_dir=cache_dir,
        )

    total = len(hf_dataset)
    logger.info(f"Wikipedia loaded: {total:,} articles (memory-mapped)")

    # Filter short articles first (stays on disk via Arrow)
    if min_article_length > 0:
        hf_dataset = hf_dataset.filter(
            lambda x: len(x["text"]) >= min_article_length,
            num_proc=4,
        )
        logger.info(
            f"Filtered to {len(hf_dataset):,} articles "
            f"(min_length={min_article_length})"
        )

    # Random holdout split — stays on disk, zero RAM overhead
    split = hf_dataset.train_test_split(
        test_size=val_fraction, seed=seed,
    )
    train_hf = split["train"]
    val_hf = split["test"]

    if max_train_samples is not None:
        train_hf = train_hf.select(range(min(max_train_samples, len(train_hf))))
    if max_val_samples is not None and len(val_hf) > max_val_samples:
        val_hf = val_hf.select(range(max_val_samples))

    logger.info(
        f"Split: {len(train_hf):,} train, {len(val_hf):,} val "
        f"(zero overlap guaranteed)"
    )

    # Build tf.data.Datasets using generators (reads from Arrow on disk)
    train_ds = _hf_to_tf_dataset(train_hf, shuffle=True)
    val_ds = _hf_to_tf_dataset(val_hf, shuffle=False)

    return train_ds, val_ds


def _hf_to_tf_dataset(
    hf_dataset: datasets.Dataset,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Convert HF Arrow dataset to tf.data.Dataset via generator.

    Reads text from memory-mapped Arrow files on demand — no need
    to load all articles into RAM.
    """
    if shuffle:
        hf_dataset = hf_dataset.shuffle()

    def generator():
        for item in hf_dataset:
            yield item["text"]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
    )
    return ds


def load_hf_text_dataset(
    path: str,
    name: Optional[str] = None,
    split: str = "train",
    text_field: str = "text",
    cache_dir: Optional[str] = None,
    min_length: int = 0,
    max_samples: Optional[int] = None,
    streaming: bool = True,
) -> tf.data.Dataset:
    """Load a text dataset from HuggingFace Hub as a ``tf.data.Dataset``.

    Generic loader for any HuggingFace dataset that has a text column.

    :param path: HuggingFace dataset identifier.
    :param name: Dataset configuration name.
    :param split: Dataset split to load.
    :param text_field: Name of the text column.
    :param cache_dir: Local directory for caching.
    :param min_length: Skip texts shorter than this (chars).
    :param max_samples: Limit number of samples.
    :param streaming: If ``True``, stream without downloading everything.
    :return: ``tf.data.Dataset`` yielding raw text strings.
    """
    config_info = f" ({name})" if name else ""
    logger.info(
        f"Loading HF dataset: {path}{config_info}, split={split}, "
        f"streaming={streaming}, cache_dir={cache_dir}"
    )

    try:
        hf_dataset = datasets.load_dataset(
            path, name, split=split, streaming=streaming,
            cache_dir=cache_dir, trust_remote_code=True,
        )
    except TypeError:
        hf_dataset = datasets.load_dataset(
            path, name, split=split, streaming=streaming,
            cache_dir=cache_dir,
        )

    if streaming:
        def generator():
            emitted = 0
            for item in hf_dataset:
                text = item.get(text_field, "")
                if len(text) < min_length:
                    continue
                yield text
                emitted += 1
                if max_samples is not None and emitted >= max_samples:
                    break

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
        )
    else:
        if min_length > 0:
            hf_dataset = hf_dataset.filter(
                lambda x: len(x[text_field]) >= min_length, num_proc=4,
            )
        if max_samples is not None:
            hf_dataset = hf_dataset.select(
                range(min(max_samples, len(hf_dataset)))
            )
        return _hf_to_tf_dataset(hf_dataset)
