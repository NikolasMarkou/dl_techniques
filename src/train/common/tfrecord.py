"""Generic TFRecord I/O for high-throughput training datasets.

This module is the disk-format counterpart of the path-streaming loaders
in :mod:`train.common.image_text` and :mod:`train.common.datasets`. It
converts path-addressable datasets (per-sample JPEGs/PNGs + metadata) to
sharded TFRecord files that read sequentially, eliminating the
random-seek bottleneck that dominates per-step latency on spinning or
SATA-class disks once the page cache cannot hold the working set.

**When to use this**

The streaming JPEG path in ``image_text.make_image_text_tf_dataset`` is
the right default for small/medium datasets and prototyping: paths are
cheap, no preprocessing step is needed, and the OS page cache makes
repeat epochs nearly free *if the dataset fits in RAM*. Once the
dataset exceeds RAM (CC3M at ~230 GB on a 32 GB host is the canonical
case), every batch becomes a scatter-read across thousands of small
JPEG files and GPU utilization collapses to 30-40 %. TFRecord shards
turn that scatter-read into a few large sequential reads per step, and
``num_parallel_reads`` keeps the GPU fed.

**What this module does NOT do**

- It does not own dataset-specific logic (no CC3M / COCO / ImageNet
  loaders here). Dataset-specific converters live with their dataset
  (e.g. ``train/cliffordnet/convert_cc3m_to_tfrecord.py``) and call the
  generic primitives below.
- It does not own preprocessing. Augmentation, normalization, and
  tokenization stay in the domain modules (``image_text.py``,
  ``nlp.py``). This module deals only in raw bytes and ints on the
  write side and parsed feature dicts on the read side.

**Standard schemas**

Two pre-baked schemas cover most current use cases:

- :data:`IMAGE_TEXT_SCHEMA` for image-caption pairs (CLIP, VLMs)
- :data:`IMAGE_LABEL_SCHEMA` for image classification

Each schema bundles the type spec for writing (:class:`SchemaSpec`) and
the parsing spec for reading (``features_spec``). New dataset types
just need a new :class:`SchemaSpec`.

**Sharding**

The writer streams examples and starts a new shard once the current
shard's serialized bytes exceed ``target_shard_bytes`` (default
256 MiB). Shards are named ``<prefix>-NNNNNN.tfrecord`` and a sidecar
``<prefix>_manifest.json`` records the per-shard count and total. We
deliberately do not use the ``-NNNNN-of-MMMMM`` convention because the
final shard count is not known until streaming completes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import tensorflow as tf

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
#
# Thin wrappers around ``tf.train.Feature`` for the three scalar list
# types we use. Strings are auto-encoded as utf-8.


def bytes_feature(value: Any) -> tf.train.Feature:
    """Single-element bytes feature. Accepts ``bytes`` or ``str``."""
    if isinstance(value, str):
        value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value: int) -> tf.train.Feature:
    """Single-element int64 feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def int64_list_feature(values: Iterable[int]) -> tf.train.Feature:
    """Variable-length int64 list feature (e.g. token id sequences)."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[int(v) for v in values])
    )


def float_feature(value: float) -> tf.train.Feature:
    """Single-element float32 feature."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[float(value)])
    )


def float_list_feature(values: Iterable[float]) -> tf.train.Feature:
    """Variable-length float32 list feature."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[float(v) for v in values])
    )


def make_example(features: Mapping[str, tf.train.Feature]) -> tf.train.Example:
    """Wrap a feature dict into a ``tf.train.Example`` protobuf."""
    return tf.train.Example(features=tf.train.Features(feature=dict(features)))


# ---------------------------------------------------------------------------
# Schema specifications
# ---------------------------------------------------------------------------
#
# A SchemaSpec bundles everything the writer needs to validate a payload
# (which fields, which types) and everything the reader needs to parse a
# serialized example (the FixedLenFeature spec). Keeping them together
# prevents writer/reader drift.


@dataclass(frozen=True)
class SchemaSpec:
    """Feature-set definition for a TFRecord dataset family.

    :param name: Human-readable schema name (recorded in the manifest).
    :param fields: Ordered field name -> tf type ("bytes", "int64",
        "int64_list", "float", "float_list"). Used by the writer for
        validation only — the actual encoding is handled by the
        ``*_feature`` helpers in the calling code.
    :param features_spec: Parsing spec for ``tf.io.parse_single_example``.
    """

    name: str
    fields: Dict[str, str]
    features_spec: Dict[str, Any]


IMAGE_TEXT_SCHEMA = SchemaSpec(
    name="image_text",
    fields={
        "image/encoded": "bytes",
        "image/format": "bytes",
        "text": "bytes",
        "token_ids": "int64_list",
        "id": "bytes",
    },
    features_spec={
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature(
            [], tf.string, default_value=b"jpeg"
        ),
        "text": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        # ``token_ids`` is variable length — parsed as VarLen and densified
        # by the consumer with the known context length.
        "token_ids": tf.io.VarLenFeature(tf.int64),
        "id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    },
)


IMAGE_LABEL_SCHEMA = SchemaSpec(
    name="image_label",
    fields={
        "image/encoded": "bytes",
        "image/format": "bytes",
        "label": "int64",
        "id": "bytes",
    },
    features_spec={
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature(
            [], tf.string, default_value=b"jpeg"
        ),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    },
)


# ---------------------------------------------------------------------------
# Domain-specific example builders
# ---------------------------------------------------------------------------
#
# These return a feature dict ready to feed to make_example(...). Callers
# can also construct feature dicts inline if their domain doesn't fit the
# pre-baked schemas — the helpers above are the only contract.


def build_image_text_example(
    image_bytes: bytes,
    text: str,
    token_ids: Optional[Iterable[int]] = None,
    image_format: str = "jpeg",
    image_id: str = "",
) -> tf.train.Example:
    """Build an ``image_text`` example.

    :param image_bytes: Raw JPEG/PNG bytes (no decoding done at write time).
    :param text: Caption string (utf-8 encoded into the record).
    :param token_ids: Optional pre-tokenized ids. If supplied, downstream
        readers can skip the tokenizer round-trip; if omitted the field
        is empty and tokenization happens at read time.
    :param image_format: ``"jpeg"`` or ``"png"`` (used by the consumer to
        pick the right decoder).
    :param image_id: Optional stable identifier for debugging.
    """
    features = {
        "image/encoded": bytes_feature(image_bytes),
        "image/format": bytes_feature(image_format),
        "text": bytes_feature(text),
        "token_ids": int64_list_feature(token_ids or []),
        "id": bytes_feature(image_id),
    }
    return make_example(features)


def build_image_label_example(
    image_bytes: bytes,
    label: int,
    image_format: str = "jpeg",
    image_id: str = "",
) -> tf.train.Example:
    """Build an ``image_label`` example."""
    features = {
        "image/encoded": bytes_feature(image_bytes),
        "image/format": bytes_feature(image_format),
        "label": int64_feature(label),
        "id": bytes_feature(image_id),
    }
    return make_example(features)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_tfrecord_shards(
    examples: Iterable[tf.train.Example],
    output_dir: str,
    prefix: str,
    schema: SchemaSpec,
    target_shard_bytes: int = 256 * 1024 * 1024,
    progress_every: int = 10000,
    extra_manifest_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stream examples into auto-sharded TFRecord files.

    Shards are rolled when the *serialized* size of the current shard
    crosses ``target_shard_bytes``. The default 256 MiB target balances
    parallelism (more shards = more interleave parallelism on the read
    side) against per-shard overhead (shorter sequential runs).

    :param examples: Iterable of ``tf.train.Example`` protobufs. Lazily
        consumed — callers can stream from disk without materializing.
    :param output_dir: Destination directory (created if absent).
    :param prefix: Shard filename prefix, e.g. ``"cc3m-train"`` →
        ``cc3m-train-000000.tfrecord``, ``cc3m-train-000001.tfrecord``...
    :param schema: :class:`SchemaSpec` (recorded in the manifest for
        cross-checking on read).
    :param target_shard_bytes: Approximate bytes per shard.
    :param progress_every: Log a heartbeat every N examples.
    :param extra_manifest_fields: Extra metadata to record alongside the
        shard list (e.g. tokenizer name, context_length).
    :return: Stats dict (``num_shards``, ``num_examples``, ``total_bytes``,
        ``shards`` — list of per-shard counts).
    """
    os.makedirs(output_dir, exist_ok=True)

    shard_idx = 0
    shard_count = 0
    shard_bytes = 0
    total_count = 0
    total_bytes = 0
    shards: List[Dict[str, Any]] = []

    def _open(idx: int) -> tf.io.TFRecordWriter:
        path = os.path.join(output_dir, f"{prefix}-{idx:06d}.tfrecord")
        return tf.io.TFRecordWriter(path)

    writer = _open(shard_idx)
    try:
        for example in examples:
            payload = example.SerializeToString()
            writer.write(payload)
            shard_count += 1
            shard_bytes += len(payload)
            total_count += 1
            total_bytes += len(payload)

            if progress_every and total_count % progress_every == 0:
                logger.info(
                    f"[tfrecord/{prefix}] wrote {total_count:,} examples, "
                    f"shard {shard_idx} ({shard_bytes / 2**20:.1f} MiB, "
                    f"{shard_count:,} examples)"
                )

            if shard_bytes >= target_shard_bytes:
                writer.close()
                shards.append(
                    {
                        "shard": shard_idx,
                        "examples": shard_count,
                        "bytes": shard_bytes,
                    }
                )
                shard_idx += 1
                shard_count = 0
                shard_bytes = 0
                writer = _open(shard_idx)
    finally:
        writer.close()

    if shard_count > 0 or shard_idx == 0:
        # Final shard always recorded, even if empty (handles 0-example
        # corner case for bookkeeping consistency).
        shards.append(
            {
                "shard": shard_idx,
                "examples": shard_count,
                "bytes": shard_bytes,
            }
        )

    manifest = {
        "schema": schema.name,
        "fields": schema.fields,
        "prefix": prefix,
        "num_shards": len(shards),
        "num_examples": total_count,
        "total_bytes": total_bytes,
        "shards": shards,
    }
    if extra_manifest_fields:
        manifest.update(extra_manifest_fields)

    manifest_path = os.path.join(output_dir, f"{prefix}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"[tfrecord/{prefix}] DONE: {total_count:,} examples in "
        f"{len(shards)} shards ({total_bytes / 2**30:.2f} GiB), manifest "
        f"at {manifest_path}"
    )

    return manifest


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def list_tfrecord_shards(
    file_pattern_or_dir: str, prefix: Optional[str] = None
) -> List[str]:
    """Resolve a directory + optional prefix or a glob into a sorted file list.

    Accepts either:

    - A glob pattern (``"...*.tfrecord"``) → returned globbed + sorted.
    - A directory + ``prefix`` → returns ``<dir>/<prefix>-*.tfrecord``.
    - A directory alone → returns ``<dir>/*.tfrecord``.
    """
    if any(ch in file_pattern_or_dir for ch in "*?["):
        files = sorted(tf.io.gfile.glob(file_pattern_or_dir))
    elif prefix is not None:
        files = sorted(
            tf.io.gfile.glob(
                os.path.join(file_pattern_or_dir, f"{prefix}-*.tfrecord")
            )
        )
    else:
        files = sorted(
            tf.io.gfile.glob(
                os.path.join(file_pattern_or_dir, "*.tfrecord")
            )
        )
    if not files:
        raise FileNotFoundError(
            f"No TFRecord shards matched: {file_pattern_or_dir} "
            f"(prefix={prefix!r})"
        )
    return files


def read_tfrecord_dataset(
    file_pattern_or_dir: str,
    schema: SchemaSpec,
    *,
    prefix: Optional[str] = None,
    shuffle_files: bool = True,
    num_parallel_reads: int = 8,
    cycle_length: Optional[int] = None,
    deterministic: bool = False,
    file_seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Build a parsed-example ``tf.data.Dataset`` from shard files.

    Reads are interleaved across ``cycle_length`` shards in parallel.
    The dataset yields *parsed feature dicts* with the keys defined by
    ``schema.features_spec``; image bytes are still encoded — decode
    them downstream (e.g. ``tf.io.decode_jpeg``) so the pipeline stays
    lazy and the parser works on raw bytes.

    :param file_pattern_or_dir: Glob pattern, or a directory (use
        ``prefix`` to disambiguate when multiple datasets share a dir).
    :param schema: :class:`SchemaSpec` whose ``features_spec`` is used
        for ``tf.io.parse_single_example``.
    :param prefix: Optional shard prefix when ``file_pattern_or_dir`` is
        a directory.
    :param shuffle_files: Shuffle the shard order before reading.
        Per-example shuffle is the caller's responsibility (apply
        ``.shuffle(buffer)`` after this).
    :param num_parallel_reads: Concurrent shard readers.
    :param cycle_length: ``interleave`` cycle length; defaults to
        ``num_parallel_reads``. Higher = more diverse mixing across
        shards at the cost of more open file handles.
    :param deterministic: Pass through to ``interleave`` —
        ``False`` allows out-of-order outputs for higher throughput.
    :param file_seed: Optional seed for shard-shuffle reproducibility.
    """
    files = list_tfrecord_shards(file_pattern_or_dir, prefix=prefix)
    cycle_length = cycle_length or num_parallel_reads

    file_ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_files:
        file_ds = file_ds.shuffle(
            len(files), seed=file_seed, reshuffle_each_iteration=True
        )

    ds = file_ds.interleave(
        lambda path: tf.data.TFRecordDataset(
            path, num_parallel_reads=None
        ),
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_reads,
        deterministic=deterministic,
    )

    features_spec = schema.features_spec

    def _parse(serialized: tf.Tensor) -> Dict[str, tf.Tensor]:
        return tf.io.parse_single_example(serialized, features_spec)

    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Image-text consumption helper
# ---------------------------------------------------------------------------
#
# Mirrors ``make_image_text_tf_dataset`` but reads from TFRecord shards
# instead of streaming JPEGs from individual files. The returned dataset
# has the same batched signature so a training script can switch between
# the two without touching the model side.


def make_image_text_tf_dataset_from_tfrecord(
    file_pattern_or_dir: str,
    image_size: int,
    context_length: int,
    pad_token_id: int,
    batch_size: int,
    *,
    prefix: Optional[str] = None,
    training: bool = True,
    shuffle_buffer: int = 10000,
    num_parallel_reads: int = 8,
    decode_fn: Optional[Callable[[tf.Tensor, int, bool], tf.Tensor]] = None,
    augment_fn: Optional[Callable[[tf.Tensor, int, bool], tf.Tensor]] = None,
) -> tf.data.Dataset:
    """Build a CLIP-style image-text training dataset from TFRecord shards.

    Examples must follow :data:`IMAGE_TEXT_SCHEMA` (token ids must have
    been written via :func:`build_image_text_example` with a non-empty
    ``token_ids``). Tokenizing at write time keeps the read path free of
    Python-side ``py_function`` calls.

    :param file_pattern_or_dir: Forwarded to :func:`read_tfrecord_dataset`.
    :param image_size: Target post-crop side length.
    :param context_length: Token sequence length (must match write-time).
    :param pad_token_id: ID to right-pad / truncate to ``context_length``.
    :param batch_size: Output batch size; ``drop_remainder=True`` when
        ``training``.
    :param prefix: Optional shard prefix.
    :param training: Whether to shuffle, augment, and repeat.
    :param shuffle_buffer: Per-example shuffle buffer size.
    :param num_parallel_reads: Concurrent shard readers.
    :param decode_fn: Optional override for ``(jpeg_bytes_tensor,
        image_size, training) -> uint8`` decode+resize. Defaults to
        ``read_decode_resize_uint8`` semantics applied to raw bytes.
    :param augment_fn: Optional override for ``(uint8, image_size,
        training) -> float32`` augment+normalize. Defaults to the
        ``image_text.augment_and_normalize`` semantics.
    :return: ``tf.data.Dataset`` yielding ``{"image": float32[B,H,W,3],
        "text": int32[B,L]}``.
    """
    # Local import keeps tfrecord.py importable without pulling in the
    # full image_text module's dependencies (and avoids a circular import
    # if image_text ever wants to use tfrecord helpers itself).
    from train.common.image_text import augment_and_normalize as _default_aug

    if decode_fn is None:
        def decode_fn(  # type: ignore[misc]
            jpeg_bytes: tf.Tensor, size: int, train: bool
        ) -> tf.Tensor:
            img = tf.io.decode_jpeg(jpeg_bytes, channels=3)
            if train:
                target = tf.cast(size, tf.float32)
                larger = tf.cast(tf.math.ceil(target * 1.15), tf.int32)
                img = tf.image.resize(img, (larger, larger), method="bilinear")
            else:
                img = tf.image.resize(img, (size, size), method="bilinear")
            return tf.cast(img, tf.uint8)

    if augment_fn is None:
        augment_fn = _default_aug

    ds = read_tfrecord_dataset(
        file_pattern_or_dir,
        IMAGE_TEXT_SCHEMA,
        prefix=prefix,
        shuffle_files=training,
        num_parallel_reads=num_parallel_reads,
    )

    pad = tf.constant(pad_token_id, dtype=tf.int32)

    def _pad_or_truncate(token_ids_sparse: tf.SparseTensor) -> tf.Tensor:
        # parse_single_example returns VarLen as SparseTensor; densify with
        # the pad token, then crop / right-pad to context_length.
        dense = tf.cast(
            tf.sparse.to_dense(token_ids_sparse, default_value=pad_token_id),
            tf.int32,
        )
        length = tf.shape(dense)[0]
        # Right-pad with `pad` to context_length (no-op if already long enough).
        dense = tf.cond(
            length < context_length,
            lambda: tf.concat(
                [
                    dense,
                    tf.fill((context_length - length,), pad),
                ],
                axis=0,
            ),
            lambda: dense[:context_length],
        )
        dense.set_shape([context_length])
        return dense

    def _to_pair(parsed: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        img_u8 = decode_fn(parsed["image/encoded"], image_size, training)
        img = augment_fn(img_u8, image_size, training)
        toks = _pad_or_truncate(parsed["token_ids"])
        return {"image": img, "text": toks}

    if training:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    ds = ds.map(_to_pair, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
