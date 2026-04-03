"""NLP dataset loaders for text-based training pipelines.

Provides loaders for Wikipedia and other HuggingFace text datasets,
returning ``tf.data.Dataset`` of raw text strings compatible with
the CLM/MLM preprocessing in ``train.common.nlp``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import datasets
import numpy as np
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
    max_train_samples: Optional[int] = None,
    max_val_samples: int = 5000,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load Wikipedia with a proper random holdout train/val split.

    Downloads Wikipedia to ``cache_dir`` on first call (Arrow format).
    Splits articles randomly by index so train and val are guaranteed
    to have zero overlap regardless of iteration order.

    :param cache_dir: Local directory for Arrow cache files.
    :param config_name: Wikipedia config (e.g. ``'20231101.en'``).
    :param min_article_length: Skip articles shorter than this (chars).
    :param val_fraction: Fraction of articles reserved for validation.
    :param max_train_samples: Limit training articles. ``None`` for all.
    :param max_val_samples: Limit validation articles.
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
    logger.info(f"Wikipedia loaded: {total:,} articles")

    # Random split by index
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total)
    val_size = min(int(total * val_fraction), max_val_samples)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]

    logger.info(
        f"Split: {len(train_indices):,} train, {len(val_indices):,} val "
        f"(zero overlap guaranteed)"
    )

    # Build tf.data.Datasets with shuffled indices
    text_field = "text"

    def make_dataset(idxs, shuffle: bool) -> tf.data.Dataset:
        subset = hf_dataset.select(idxs)
        if min_article_length > 0:
            subset = subset.filter(
                lambda x: len(x[text_field]) >= min_article_length,
                num_proc=4,
            )
        texts = list(subset[text_field])
        ds = tf.data.Dataset.from_tensor_slices(texts)
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(texts), 100000))
        logger.info(
            f"  {'Train' if shuffle else 'Val'} dataset: "
            f"{len(texts):,} articles after filtering"
        )
        return ds

    train_ds = make_dataset(train_indices, shuffle=True)
    val_ds = make_dataset(val_indices, shuffle=False)

    return train_ds, val_ds


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
        texts = hf_dataset[text_field]
        return tf.data.Dataset.from_tensor_slices(texts)
