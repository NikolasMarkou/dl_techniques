"""Pre-resize CC3M JPEGs onto fast storage (SSD) for I/O-bound CLIP training.

Motivation
----------
CC3M (2.9M small JPEGs, 256 shards) lives on a spinning HDD. Random-access
reads of one 256-image batch per step cost ~256 disk seeks; at ~150 HDD IOPS
that is ~1.7 s/step of pure seek latency, which starves the GPU (observed
4 s/step, GPU util ~0%). The model/geometry are not the bottleneck -- disk
IOPS is. This script makes a one-time, resumable, downscaled copy of CC3M on
the SSD so subsequent training (both A/B arms + the caption-filter pass) reads
from fast storage.

Design
------
- Walk ``<src>/<split>/<shard>/*.jpg`` and MIRROR the relative path to
  ``<dst>/<split>/<shard>/<same>.jpg``. Mirroring the existing relative layout
  means the training loader's ``_cc3m_shard_of`` path reconstruction matches
  with NO loader change -- just point ``--cc3m-root`` at ``<dst>``.
- Each image is resized to ``--size`` (default 144) square JPEG at
  ``--quality`` (default 92). 144 is a small antialias margin above the train
  pipeline's 129 pre-crop resize (``ceil(112 * 1.15)``); the downstream
  random-crop to 112 is unchanged.
- Resumable: an existing destination file is skipped, so a crashed run can be
  re-invoked. Idempotent.
- After image copy, the two split manifests (``<split>_captions.jsonl``) and
  any existing tokenization sidecar caches are symlinked into ``<dst>`` so the
  destination is a drop-in ``--cc3m-root``.

Usage
-----
    MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.resize_cc3m_to_ssd \
        --src-root /media/arxwn/data0_4tb/datasets/cc3m \
        --dst-root /media/arxwn/data_fast/datasets/cc3m_144 \
        --splits train validation --workers 12

The full pass is a multi-hour, HDD-read-bound job; launch it detached.
"""

import argparse
import glob
import os
import time
from multiprocessing import Pool
from typing import List, Tuple

from PIL import Image, ImageFile

from dl_techniques.utils.logger import logger

# Tolerate slightly-truncated JPEGs rather than aborting the whole shard.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Worker-global config (set in __main__ before the Pool forks; Linux fork
# inheritance carries these into every worker without pickling per task).
_SIZE = 144
_QUALITY = 92


def _resize_one(task: Tuple[str, str]) -> str:
    """Resize a single JPEG src->dst. Returns a status token.

    :param task: ``(src_path, dst_path)`` pair.
    :return: ``"ok"`` | ``"skip"`` | ``"err:<msg>"``.
    """
    src, dst = task
    try:
        if os.path.exists(dst):
            return "skip"
        with Image.open(src) as im:
            im = im.convert("RGB").resize((_SIZE, _SIZE), Image.BILINEAR)
            tmp = dst + ".tmp"
            im.save(tmp, format="JPEG", quality=_QUALITY)
        os.replace(tmp, dst)  # atomic publish -> safe under resume
        return "ok"
    except Exception as exc:  # noqa: BLE001 - per-file robustness
        return f"err:{type(exc).__name__}:{str(exc)[:80]}"


def _link_sidecars(src_root: str, dst_root: str, splits: List[str]) -> None:
    """Symlink split manifests + token caches into ``dst_root`` (drop-in root)."""
    for split in splits:
        manifest = f"{split}_captions.jsonl"
        src_m = os.path.join(src_root, manifest)
        dst_m = os.path.join(dst_root, manifest)
        if os.path.exists(src_m) and not os.path.exists(dst_m):
            os.symlink(src_m, dst_m)
            logger.info(f"Linked manifest: {dst_m} -> {src_m}")
        # Reuse any existing tokenization sidecar caches (keyed on manifest
        # mtime+size, which a symlink preserves).
        for cache in glob.glob(src_m + ".tokcache.*"):
            dst_c = os.path.join(dst_root, os.path.basename(cache))
            if not os.path.exists(dst_c):
                os.symlink(cache, dst_c)
                logger.info(f"Linked tokcache: {dst_c}")


def _build_tasks(src_root: str, dst_root: str, split: str) -> List[Tuple[str, str]]:
    """Enumerate (src, dst) jpg pairs for one split, mirroring relative paths."""
    split_src = os.path.join(src_root, split)
    tasks: List[Tuple[str, str]] = []
    for shard in sorted(os.listdir(split_src)):
        shard_dir = os.path.join(split_src, shard)
        if not os.path.isdir(shard_dir):
            continue
        for fn in os.listdir(shard_dir):
            if fn.endswith(".jpg"):
                tasks.append(
                    (
                        os.path.join(shard_dir, fn),
                        os.path.join(dst_root, split, shard, fn),
                    )
                )
    return tasks


def main() -> None:
    """Entry point: parse args, resize all splits with a worker pool."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-root", default="/media/arxwn/data0_4tb/datasets/cc3m")
    p.add_argument("--dst-root", default="/media/arxwn/data_fast/datasets/cc3m_144")
    p.add_argument("--splits", nargs="+", default=["train", "validation"])
    p.add_argument("--size", type=int, default=144, help="Square output size (px)")
    p.add_argument("--quality", type=int, default=92, help="Output JPEG quality")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--log-every", type=int, default=20000)
    args = p.parse_args()

    global _SIZE, _QUALITY
    _SIZE, _QUALITY = args.size, args.quality

    os.makedirs(args.dst_root, exist_ok=True)
    logger.info(
        f"Resize CC3M {args.src_root} -> {args.dst_root} | size={_SIZE} "
        f"q={_QUALITY} workers={args.workers} splits={args.splits}"
    )

    grand_ok = grand_skip = grand_err = 0
    t0 = time.time()
    for split in args.splits:
        tasks = _build_tasks(args.src_root, args.dst_root, split)
        # Pre-create shard dirs once (cheap) so workers never race on mkdir.
        for shard in sorted({os.path.dirname(d) for _, d in tasks}):
            os.makedirs(shard, exist_ok=True)
        logger.info(f"[{split}] {len(tasks):,} images to process")

        ok = skip = err = 0
        done = 0
        with Pool(processes=args.workers) as pool:
            for status in pool.imap_unordered(_resize_one, tasks, chunksize=64):
                done += 1
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    err += 1
                    if err <= 20:
                        logger.warning(f"[{split}] {status}")
                if done % args.log_every == 0:
                    rate = done / max(1e-6, time.time() - t0)
                    logger.info(
                        f"[{split}] {done:,}/{len(tasks):,} "
                        f"(ok={ok:,} skip={skip:,} err={err:,}) {rate:.0f} img/s"
                    )
        logger.info(
            f"[{split}] DONE ok={ok:,} skip={skip:,} err={err:,}"
        )
        grand_ok += ok
        grand_skip += skip
        grand_err += err

    _link_sidecars(args.src_root, args.dst_root, args.splits)
    dt = time.time() - t0
    logger.info(
        f"ALL DONE in {dt/3600:.2f}h | ok={grand_ok:,} skip={grand_skip:,} "
        f"err={grand_err:,} | dst={args.dst_root}"
    )


if __name__ == "__main__":
    main()
