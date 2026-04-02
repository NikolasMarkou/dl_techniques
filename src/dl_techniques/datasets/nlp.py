"""NLP dataset loaders for text-based training pipelines.

Provides loaders for Wikipedia and other HuggingFace text datasets,
returning ``tf.data.Dataset`` of raw text strings compatible with
the CLM/MLM preprocessing in ``train.common.nlp``.

.. code-block:: text

    ┌──────────────────────┐
    │  HuggingFace Hub /   │
    │  Local Arrow Cache   │
    └─────────┬────────────┘
              │
              ▼
    ┌──────────────────────┐
    │  load_*_dataset()    │──► tf.data.Dataset[str]
    │  filter + extract    │
    └──────────────────────┘
              │
              ▼
    ┌──────────────────────┐
    │  preprocess_clm/mlm  │──► (input_ids, labels) batches
    │  (train.common.nlp)  │
    └──────────────────────┘
"""

from __future__ import annotations

from typing import Optional

import datasets
import tensorflow as tf

from dl_techniques.utils.logger import logger

# Default cache directory for Wikipedia dataset
DEFAULT_WIKIPEDIA_CACHE_DIR = "/media/arxwn/data0_4tb/datasets/wikipedia"
DEFAULT_WIKIPEDIA_CONFIG = "20231101.en"


def load_wikipedia_dataset(
    cache_dir: str = DEFAULT_WIKIPEDIA_CACHE_DIR,
    config_name: str = DEFAULT_WIKIPEDIA_CONFIG,
    split: str = "train",
    min_article_length: int = 100,
    max_samples: Optional[int] = None,
    skip_samples: int = 0,
    streaming: bool = True,
) -> tf.data.Dataset:
    """Load Wikipedia dataset from HuggingFace Hub with local caching.

    Downloads Wikipedia to ``cache_dir`` in Arrow format on first call.
    Subsequent calls load from the local cache instantly.

    :param cache_dir: Local directory for Arrow cache files.
    :param config_name: Wikipedia config (e.g. ``'20231101.en'``).
    :param split: Dataset split (Wikipedia only has ``'train'``).
    :param min_article_length: Skip articles shorter than this (chars).
    :param max_samples: Limit number of articles. ``None`` for all.
    :param skip_samples: Skip the first N articles (after filtering).
        Use to create non-overlapping train/val splits from a single
        streaming source.
    :param streaming: If ``True``, stream from Hub/cache without loading
        all data into memory. If ``False``, download and memory-map.
    :return: ``tf.data.Dataset`` yielding raw text strings.
    """
    return load_hf_text_dataset(
        path="wikimedia/wikipedia",
        name=config_name,
        split=split,
        text_field="text",
        cache_dir=cache_dir,
        min_length=min_article_length,
        max_samples=max_samples,
        skip_samples=skip_samples,
        streaming=streaming,
    )


def load_hf_text_dataset(
    path: str,
    name: Optional[str] = None,
    split: str = "train",
    text_field: str = "text",
    cache_dir: Optional[str] = None,
    min_length: int = 0,
    max_samples: Optional[int] = None,
    skip_samples: int = 0,
    streaming: bool = True,
) -> tf.data.Dataset:
    """Load a text dataset from HuggingFace Hub as a ``tf.data.Dataset``.

    Generic loader for any HuggingFace dataset that has a text column.
    Supports both streaming (memory-efficient) and download modes.

    :param path: HuggingFace dataset identifier (e.g. ``'wikipedia'``,
        ``'openwebtext'``, ``'bookcorpus'``).
    :param name: Dataset configuration name (e.g. ``'20220301.en'``).
    :param split: Dataset split to load.
    :param text_field: Name of the text column in the dataset.
    :param cache_dir: Local directory for caching downloaded data.
        ``None`` uses HuggingFace default cache.
    :param min_length: Skip texts shorter than this (chars).
    :param max_samples: Limit number of samples. ``None`` for all.
    :param skip_samples: Skip the first N samples (after filtering).
        Use to create non-overlapping splits from a single source.
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
            path,
            name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    except TypeError:
        # Newer datasets versions removed trust_remote_code
        hf_dataset = datasets.load_dataset(
            path,
            name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
        )

    if streaming:
        return _streaming_to_tf_dataset(
            hf_dataset,
            text_field=text_field,
            min_length=min_length,
            max_samples=max_samples,
            skip_samples=skip_samples,
        )
    else:
        return _non_streaming_to_tf_dataset(
            hf_dataset,
            text_field=text_field,
            min_length=min_length,
            max_samples=max_samples,
            skip_samples=skip_samples,
        )


def _streaming_to_tf_dataset(
    hf_dataset: datasets.IterableDataset,
    text_field: str,
    min_length: int,
    max_samples: Optional[int],
    skip_samples: int = 0,
) -> tf.data.Dataset:
    """Convert a streaming HF dataset to tf.data.Dataset of strings.

    When ``max_samples`` is ``None`` (training mode), the underlying HF
    iterator persists across ``tf.data`` epoch boundaries so the stream
    advances through the corpus rather than restarting from the first
    article each epoch.  When ``max_samples`` is set (validation mode),
    the iterator resets each call so the same articles are returned.
    """
    # Persistent state shared across generator() invocations.
    # Prevents the training stream from restarting every epoch.
    _state = {"iter": None, "skip_done": False}

    def generator():
        # Persistent mode (training): iterator survives across epochs
        if _state["iter"] is None:
            _state["iter"] = iter(hf_dataset)

        # Skip once on first invocation
        if not _state["skip_done"] and skip_samples > 0:
            skipped = 0
            while skipped < skip_samples:
                try:
                    item = next(_state["iter"])
                    if len(item.get(text_field, "")) >= min_length:
                        skipped += 1
                except StopIteration:
                    break
            _state["skip_done"] = True
            logger.info(f"Skipped {skipped} articles in stream")

        emitted = 0
        for item in _state["iter"]:
            text = item.get(text_field, "")
            if len(text) < min_length:
                continue
            yield text
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                break

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
    )
    logger.info(
        f"Streaming tf.data.Dataset created "
        f"(min_length={min_length}, skip={skip_samples}, "
        f"persistent={max_samples is None})"
    )
    return dataset


def _non_streaming_to_tf_dataset(
    hf_dataset: datasets.Dataset,
    text_field: str,
    min_length: int,
    max_samples: Optional[int],
    skip_samples: int = 0,
) -> tf.data.Dataset:
    """Convert a non-streaming HF dataset to tf.data.Dataset of strings."""
    if min_length > 0:
        hf_dataset = hf_dataset.filter(
            lambda x: len(x[text_field]) >= min_length,
            num_proc=4,
        )
        logger.info(
            f"Filtered to {len(hf_dataset)} articles "
            f"(min_length={min_length})"
        )

    if skip_samples > 0:
        hf_dataset = hf_dataset.select(
            range(min(skip_samples, len(hf_dataset)), len(hf_dataset))
        )
        logger.info(f"Skipped first {skip_samples} samples")

    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
        logger.info(f"Limited to {len(hf_dataset)} samples")

    texts = hf_dataset[text_field]
    dataset = tf.data.Dataset.from_tensor_slices(texts)
    logger.info(f"Non-streaming tf.data.Dataset created: {len(texts)} samples")
    return dataset
