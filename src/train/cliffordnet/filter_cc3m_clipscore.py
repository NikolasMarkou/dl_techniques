"""CC3M per-pair CLIP-score caption filter for CliffordCLIP.

Loads a saved ``CliffordCLIP`` ``.keras`` checkpoint, streams the local CC3M
tree through both encoders, computes the per-pair CLIP-score (row-wise cosine
similarity between the L2-normalized image and text embeddings), and writes a
filtered JSONL manifest containing only the pairs whose score meets a cutoff.
The output manifest uses the SAME schema the source ``<split>_captions.jsonl``
uses (``{"id", "split", "caption"}``), so it is a drop-in replacement that
:func:`train.common.image_text.load_cc3m_local_split` can read directly.

Preprocessing matches training EXACTLY: image paths are reconstructed via the
loader's shard scheme (:func:`train.common.image_text._cc3m_shard_of`), JPEGs
are decoded + resized + ImageNet/CLIP-normalized through the shared
``make_image_text_tf_dataset`` (``training=False``: no crop/flip), and captions
are tokenized with tiktoken ``gpt2`` via the shared ``tokenize_captions`` at the
configured ``context_length``. The score is therefore measured under the same
recipe the model was trained with.

!!! USER-LAUNCHED FULL PASS !!!
The full CC3M filtering pass covers ~2.9M pairs / ~220 GB of JPEGs and takes
multiple hours on a single GPU. It is a USER-LAUNCHED campaign, NOT something
this script runs automatically: invoke without ``--max-samples`` to process the
whole split. Use ``--max-samples N`` for a small SMOKE pass that only verifies
the script + write path. A sub-agent background task dies when the agent exits,
so the maintainer launches the full pass themselves.

Usage (SMOKE)::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python \\
        -m train.cliffordnet.filter_cc3m_clipscore \\
        --checkpoint results/.../checkpoints/final.keras \\
        --out-manifest /tmp/cc3m_filtered.jsonl \\
        --threshold 0.20 --max-samples 1000 --gpu 1

Usage (FULL, user-launched)::

    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python \\
        -m train.cliffordnet.filter_cc3m_clipscore \\
        --checkpoint results/.../checkpoints/final.keras \\
        --out-manifest /media/.../cc3m/train_captions_clip0.20.jsonl \\
        --threshold 0.20 --gpu 1
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import tiktoken

from train.common import setup_gpu
from train.common.image_text import (
    _cc3m_shard_of,
    make_image_text_tf_dataset,
    tokenize_captions,
)
from dl_techniques.models.clip.clifford_clip import CliffordCLIP
from dl_techniques.utils.logger import logger

# Checkpoints saved by ``train_clip.py`` are the BARE ``CliffordCLIP``
# (``ContrastiveCliffordCLIP.clip_model`` is the saved object), so
# ``{"CliffordCLIP": CliffordCLIP}`` is sufficient. The Clifford head/blocks
# auto-register via ``@register_keras_serializable``; a bare checkpoint carries
# no ``CLIPContrastiveLoss``. A future wrapper checkpoint would make
# ``load_model`` raise a clear "Unknown object" naming the class to add here.
_CUSTOM_OBJECTS = {"CliffordCLIP": CliffordCLIP}


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "CC3M CLIP-score caption filter for CliffordCLIP "
            "(full pass is USER-LAUNCHED, ~220 GB / multi-hour)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a saved CliffordCLIP .keras checkpoint",
    )
    p.add_argument(
        "--cc3m-root", type=str,
        default="/media/arxwn/data0_4tb/datasets/cc3m",
        help="CC3M root (contains <split>/ and <split>_captions.jsonl)",
    )
    p.add_argument(
        "--out-manifest", type=str, required=True,
        help="Output JSONL path for the filtered (drop-in) manifest",
    )
    p.add_argument(
        "--threshold", type=float, default=0.20,
        help="CLIP-score (cosine) cutoff; pairs with score >= this are kept",
    )
    p.add_argument(
        "--split", type=str, default="train",
        help="CC3M split to filter (train or validation)",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap on pairs processed (None = full split; small N = SMOKE)",
    )
    p.add_argument(
        "--image-size", type=int, default=112,
        help="Image resolution (MUST match training)",
    )
    p.add_argument(
        "--context-length", type=int, default=64,
        help="Text sequence length (MUST match training)",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Scoring batch size",
    )
    p.add_argument(
        "--gpu", type=int, default=1, help="GPU device index",
    )
    return p


def _read_cc3m_records(
    split: str,
    cc3m_root: str,
    max_samples: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Read CC3M records and reconstruct image paths via the loader's scheme.

    Mirrors :func:`train.common.image_text.load_cc3m_local_split`'s record
    iteration and path reconstruction (reusing :func:`_cc3m_shard_of`) but
    additionally retains the FULL source record per kept pair so the filtered
    manifest can be re-emitted in the identical schema. Records whose caption
    is empty or whose image file is missing are skipped (logged), exactly as
    the loader skips them.

    :param split: CC3M split name (``"train"`` / ``"validation"``).
    :param cc3m_root: CC3M root directory.
    :param max_samples: Optional cap on the number of pairs returned.
    :return: ``(records, image_paths)`` aligned by index. ``records`` are the
        raw JSONL dicts; ``image_paths`` are the resolved absolute JPEG paths.
    :raises FileNotFoundError: If the JSONL or image dir are missing.
    """
    jsonl_path = os.path.join(cc3m_root, f"{split}_captions.jsonl")
    img_root = os.path.join(cc3m_root, split)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"CC3M captions file not found: {jsonl_path}. "
            f"Run train/cliffordnet/prepare_cc3m.py --dst {cc3m_root} first."
        )
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"CC3M image dir not found: {img_root}")

    records: List[Dict[str, Any]] = []
    paths: List[str] = []
    missing = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_id = rec["id"]
            caption = rec.get("caption", "")
            if not caption:
                continue
            candidates = (
                os.path.join(img_root, _cc3m_shard_of(img_id), f"{img_id}.jpg"),
                os.path.join(img_root, f"{img_id}.jpg"),
            )
            img_path = next(
                (p for p in candidates if os.path.exists(p)), None
            )
            if img_path is None:
                missing += 1
                continue
            records.append(rec)
            paths.append(img_path)
            if max_samples is not None and len(paths) >= max_samples:
                break

    if missing:
        logger.warning(
            f"CC3M/{split}: {missing:,} captions skipped (image missing)."
        )
    logger.info(
        f"CC3M/{split}: {len(paths):,} (image, caption) pairs to score."
    )
    return records, paths


def _score_pairs(
    model: CliffordCLIP,
    paths: List[str],
    token_ids: np.ndarray,
    image_size: int,
    batch_size: int,
) -> np.ndarray:
    """Compute the per-pair CLIP-score for every (image, caption) pair.

    The score is the row-wise cosine similarity between the L2-normalized image
    and text embeddings (``encode_image`` / ``encode_text`` both normalize by
    default), i.e. ``sum(img_emb * txt_emb, axis=-1)``. Batches are streamed
    via the shared ``make_image_text_tf_dataset`` (``training=False``) so decode
    + normalization match the eval recipe. Batches whose images fail to decode
    are caught and their pairs assigned ``-inf`` (never kept), so one corrupt
    JPEG cannot abort a multi-hour pass.

    :param model: A loaded ``CliffordCLIP``.
    :param paths: Absolute image paths.
    :param token_ids: ``(N, L)`` int32 tokens aligned with ``paths``.
    :param image_size: Image resolution.
    :param batch_size: Scoring batch size.
    :return: ``(N,)`` float32 array of CLIP-scores aligned with ``paths``.
    """
    ds = make_image_text_tf_dataset(
        images=paths,
        token_ids=token_ids,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
    )
    scores: List[float] = []
    for batch in ds:
        images = batch["image"]
        tokens = batch["text"]
        try:
            img_emb = model.encode_image(images, training=False, normalize=True)
            txt_emb = model.encode_text(tokens, training=False, normalize=True)
            batch_scores = keras.ops.sum(img_emb * txt_emb, axis=-1)
            scores.extend(np.asarray(batch_scores, dtype=np.float32).tolist())
        except Exception as exc:  # noqa: BLE001 - one bad batch must not abort
            n_bad = int(keras.ops.shape(tokens)[0])
            logger.warning(
                f"Scoring batch failed ({exc!r}); dropping {n_bad} pairs."
            )
            scores.extend([float("-inf")] * n_bad)

    out = np.asarray(scores, dtype=np.float32)
    # make_image_text_tf_dataset with training=False does NOT drop the
    # remainder, so the score count must match the input exactly.
    if out.shape[0] != len(paths):
        logger.warning(
            f"Score count {out.shape[0]} != pair count {len(paths)}; "
            f"truncating/padding alignment to the shorter length."
        )
        out = out[: len(paths)]
    return out


def filter_cc3m(args: argparse.Namespace) -> Dict[str, float]:
    """Score CC3M pairs and write the filtered drop-in manifest.

    :param args: Parsed command-line arguments.
    :return: Summary dict with ``total``, ``kept``, ``kept_fraction``, and
        ``mean_score`` keys.
    """
    setup_gpu(gpu_id=args.gpu)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = keras.models.load_model(
        args.checkpoint, compile=False, custom_objects=_CUSTOM_OBJECTS,
    )
    logger.info(f"Loaded CliffordCLIP: {model.count_params():,} params")

    records, paths = _read_cc3m_records(
        split=args.split,
        cc3m_root=args.cc3m_root,
        max_samples=args.max_samples,
    )
    if not paths:
        logger.warning("No CC3M pairs to score; writing empty manifest.")
        open(args.out_manifest, "w").close()
        return {"total": 0, "kept": 0, "kept_fraction": 0.0, "mean_score": 0.0}

    encoder = tiktoken.get_encoding("gpt2")
    captions = [r.get("caption", "") for r in records]
    token_ids = tokenize_captions(captions, encoder, args.context_length)

    logger.info(f"Scoring {len(paths):,} pairs (batch_size={args.batch_size})...")
    scores = _score_pairs(
        model=model,
        paths=paths,
        token_ids=token_ids,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    keep_mask = scores >= args.threshold
    kept = int(keep_mask.sum())
    total = len(paths)
    finite = scores[np.isfinite(scores)]
    mean_score = float(finite.mean()) if finite.size else 0.0

    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)), exist_ok=True)
    with open(args.out_manifest, "w") as out_f:
        for rec, keep in zip(records, keep_mask):
            if keep:
                out_f.write(json.dumps(rec) + "\n")

    kept_fraction = kept / total if total else 0.0
    logger.info(
        f"CC3M filter @ threshold={args.threshold}: "
        f"total={total:,} kept={kept:,} "
        f"kept_fraction={kept_fraction:.4f} mean_score={mean_score:.4f}"
    )
    logger.info(f"Filtered drop-in manifest written -> {args.out_manifest}")
    return {
        "total": float(total),
        "kept": float(kept),
        "kept_fraction": kept_fraction,
        "mean_score": mean_score,
    }


def main() -> None:
    """Main entry point."""
    args = _build_parser().parse_args()
    filter_cc3m(args)


if __name__ == "__main__":
    main()
