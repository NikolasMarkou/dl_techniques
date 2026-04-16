"""Extract Conceptual Captions 3M (CC3M) to a flat on-disk layout.

Streams the WebDataset tar shards of ``pixparse/cc3m-wds`` from Hugging
Face Hub using only ``huggingface_hub`` + stdlib ``tarfile`` — the HF
``datasets`` library is deliberately bypassed because its auto-inferred
WebDataset schema for ``pixparse/cc3m-wds`` now fails to cast against
the shard contents (parquet has a ``json`` sidecar column not in the
declared feature set).

Output layout::

    <dst_root>/
      train/XX/cc3m_train_NNNNNNNN.jpg         <- 256-way shard hash
      validation/XX/cc3m_validation_NNNNNNNN.jpg
      train_captions.jsonl                     <- {"id","caption","split"}
      validation_captions.jsonl
      _prepare_cc3m_state.json                 <- processed tar names

Each WebDataset tar contains grouped ``{basename}.jpg`` / ``{basename}.txt``
/ ``{basename}.json`` triples. We stream the tar over HTTP (no local tar
file persisted), walk its members in order, pair ``.jpg`` + ``.txt`` by
basename, and flush each completed pair immediately.

**Resumable**: the script stores the set of already-processed tar names
in ``_prepare_cc3m_state.json`` and skips them on restart. Caption
JSONL files are opened in append mode so completed tars keep their
captions across restarts. JPEG files that already exist on disk are
also skipped, giving a second layer of resume safety.

Set ``HF_HOME`` or ``HF_HUB_CACHE`` to a large-disk location *if* you
want ``huggingface_hub`` to cache any downloaded metadata off of ``~``.
The main tar data is streamed, not cached, so normally no large cache
is needed.

Example::

    python -m train.cliffordnet.prepare_cc3m \\
      --dst /media/arxwn/data0_4tb/datasets/cc3m \\
      --splits train validation \\
      --progress-every 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable, List, Optional, Tuple


_HF_REPO_ID = "pixparse/cc3m-wds"
_DEFAULT_SPLITS = ("train", "validation")
_STATE_FILENAME = "_prepare_cc3m_state.json"


# ---------------------------------------------------------------------------
# State (resumable at the tar-shard boundary)
# ---------------------------------------------------------------------------


@dataclass
class _State:
    processed_tars: dict  # {split: [tar_filename, ...]}

    @classmethod
    def load(cls, path: str) -> "_State":
        if os.path.exists(path):
            try:
                with open(path) as f:
                    raw = json.load(f)
                return cls(processed_tars=raw.get("processed_tars", {}))
            except Exception:
                pass
        return cls(processed_tars={})

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"processed_tars": self.processed_tars}, f)
        os.replace(tmp, path)

    def is_done(self, split: str, tar_name: str) -> bool:
        return tar_name in self.processed_tars.get(split, [])

    def mark_done(self, split: str, tar_name: str) -> None:
        self.processed_tars.setdefault(split, []).append(tar_name)


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def _shard_of(img_id: str) -> str:
    """Deterministic 2-char shard for an image id.

    Must stay byte-identical to
    :func:`train.common.image_text._cc3m_shard_of` so the training
    loader can rebuild paths from ids alone.
    """
    h = 0
    for ch in img_id:
        h = (h * 31 + ord(ch)) & 0xFFFF
    return f"{h & 0xFF:02x}"


# ---------------------------------------------------------------------------
# Tar streaming
# ---------------------------------------------------------------------------


def _list_shards(split: str) -> List[str]:
    """Return the list of tar filenames on HF Hub for a given split."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    glob_prefix = f"datasets/{_HF_REPO_ID}"
    all_tars = sorted(fs.glob(f"{glob_prefix}/**/*.tar"))
    token = f"cc3m-{split}-"
    return [
        t.removeprefix(glob_prefix + "/") for t in all_tars if token in t
    ]


def _open_tar_stream(tar_filename: str) -> tarfile.TarFile:
    """Open a tar shard as a streaming ``TarFile`` over HTTPS.

    Uses ``huggingface_hub``'s HTTP session so we inherit auth + retry
    policy. ``mode="r|"`` is the crucial streaming form: members are
    yielded sequentially and the underlying stream is read only once.

    :param tar_filename: Filename relative to the dataset repo root
        (e.g. ``"cc3m-train-0042.tar"``).
    """
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import get_session

    url = hf_hub_url(
        repo_id=_HF_REPO_ID,
        filename=tar_filename,
        repo_type="dataset",
    )
    session = get_session()
    response = session.get(url, stream=True, timeout=120)
    response.raise_for_status()
    # ``mode="r|"`` means sequential-read, streaming from a file-like.
    return tarfile.open(fileobj=response.raw, mode="r|")


def _iter_pairs_in_tar(
    tar: tarfile.TarFile,
) -> Iterable[Tuple[bytes, str]]:
    """Yield ``(jpeg_bytes, caption_str)`` pairs from one streaming tar.

    WebDataset convention: files with the same basename form a sample.
    Files are sorted by basename in the tar, so once we see a new
    basename we can safely flush the previous sample.
    """
    current_base: Optional[str] = None
    current_jpg: Optional[bytes] = None
    current_txt: Optional[str] = None

    def _flush() -> Optional[Tuple[bytes, str]]:
        if current_jpg is not None and current_txt is not None:
            return current_jpg, current_txt
        return None

    for member in tar:
        if not member.isfile():
            continue
        name = member.name
        base, _, ext = name.rpartition(".")
        if not base:
            continue

        if base != current_base:
            pair = _flush()
            if pair is not None:
                yield pair
            current_base = base
            current_jpg = None
            current_txt = None

        if ext == "jpg" or ext == "jpeg":
            f = tar.extractfile(member)
            if f is not None:
                current_jpg = f.read()
        elif ext == "txt":
            f = tar.extractfile(member)
            if f is not None:
                current_txt = f.read().decode("utf-8", errors="ignore")
        # ignore .json / other sidecars

    pair = _flush()
    if pair is not None:
        yield pair


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def _write_pair(
    img_root: str,
    split: str,
    global_idx: int,
    jpg_bytes: bytes,
    caption: str,
    jsonl_fh: Any,
    jpeg_quality: int,
) -> bool:
    """Write one (image, caption) pair to disk. Returns True if written.

    Re-encodes the JPEG to normalise quality / strip metadata. If the
    target path already exists we skip the write but still append the
    caption record so the JSONL stays complete after a resume.
    """
    img_id = f"cc3m_{split}_{global_idx:08d}"
    shard = _shard_of(img_id)
    shard_dir = os.path.join(img_root, shard)
    img_path = os.path.join(shard_dir, f"{img_id}.jpg")

    if not os.path.exists(img_path):
        try:
            from PIL import Image as PILImage
            os.makedirs(shard_dir, exist_ok=True)
            im = PILImage.open(BytesIO(jpg_bytes))
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(img_path, "JPEG", quality=jpeg_quality, optimize=False)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[warn] decode/save failed for {img_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return False

    jsonl_fh.write(
        json.dumps(
            {"id": img_id, "split": split, "caption": caption},
            ensure_ascii=False,
        )
        + "\n"
    )
    return True


# ---------------------------------------------------------------------------
# Per-split driver
# ---------------------------------------------------------------------------


def _extract_split(
    dst_root: str,
    split: str,
    state: _State,
    state_path: str,
    max_samples: Optional[int],
    jpeg_quality: int,
    progress_every: int,
) -> int:
    """Extract one split. Resumable at the tar-shard boundary."""
    img_root = os.path.join(dst_root, split)
    os.makedirs(img_root, exist_ok=True)
    jsonl_path = os.path.join(dst_root, f"{split}_captions.jsonl")

    print(f"[{split}] listing tar shards on HF Hub...", flush=True)
    shards = _list_shards(split)
    print(f"[{split}] {len(shards)} shards total", flush=True)

    start_time = time.time()
    # Track global index across the whole split so filenames are
    # deterministic and the same index always lands on the same shard.
    global_idx = sum(
        # Estimate from already-processed tars: each tar has ~N samples.
        # Actual count is recovered by scanning the jsonl at startup.
        0
        for _ in state.processed_tars.get(split, [])
    )

    # More robust: count existing jsonl lines to seed global_idx.
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            global_idx = sum(1 for _ in f)
        print(
            f"[{split}] resuming from {global_idx:,} already-written pairs",
            flush=True,
        )

    written = 0
    errors = 0
    # Append to the JSONL so resumes preserve previously-written captions.
    with open(jsonl_path, "a") as jsonl_fh:
        for shard_idx, tar_name in enumerate(shards):
            if state.is_done(split, tar_name):
                continue

            try:
                tar = _open_tar_stream(tar_name)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(
                    f"[warn] {tar_name}: HTTP open failed "
                    f"({type(exc).__name__}: {exc}); skipping shard",
                    file=sys.stderr,
                )
                continue

            shard_written = 0
            try:
                for jpg_bytes, caption in _iter_pairs_in_tar(tar):
                    if not caption.strip():
                        continue
                    ok = _write_pair(
                        img_root=img_root,
                        split=split,
                        global_idx=global_idx,
                        jpg_bytes=jpg_bytes,
                        caption=caption,
                        jsonl_fh=jsonl_fh,
                        jpeg_quality=jpeg_quality,
                    )
                    if ok:
                        global_idx += 1
                        written += 1
                        shard_written += 1
                        if progress_every and written % progress_every == 0:
                            elapsed = time.time() - start_time
                            rate = written / max(elapsed, 1e-6)
                            print(
                                f"[{split}] shard {shard_idx + 1}/{len(shards)} "
                                f"{tar_name} — wrote {written:,} this run "
                                f"(total pairs={global_idx:,}), "
                                f"{rate:.1f}/s, errors={errors:,}",
                                flush=True,
                            )
                        if max_samples is not None and written >= max_samples:
                            break
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(
                    f"[warn] {tar_name}: tar read failed "
                    f"({type(exc).__name__}: {exc}); "
                    f"shard partially written ({shard_written} pairs)",
                    file=sys.stderr,
                )
                continue
            finally:
                try:
                    tar.close()
                except Exception:
                    pass
                # Flush the jsonl so the data on disk matches the state
                # file; state + jsonl are committed together.
                jsonl_fh.flush()
                os.fsync(jsonl_fh.fileno())

            state.mark_done(split, tar_name)
            state.save(state_path)

            if max_samples is not None and written >= max_samples:
                break

    elapsed = time.time() - start_time
    print(
        f"[{split}] DONE: wrote {written:,} this run "
        f"(total={global_idx:,}), errors={errors:,}, "
        f"elapsed={elapsed:.0f}s"
    )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream CC3M from Hugging Face (pixparse/cc3m-wds WebDataset "
            "tars) and extract to a flat directory tree compatible with "
            "train.common.image_text.load_cc3m_local_split."
        )
    )
    parser.add_argument(
        "--dst", type=str, required=True,
        help=(
            "Destination root. Images go to <dst>/{train,validation}/, "
            "captions to <dst>/{train,validation}_captions.jsonl."
        ),
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=list(_DEFAULT_SPLITS),
        choices=list(_DEFAULT_SPLITS),
        help="Which splits to extract.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap pairs per split across the whole run (for smoke tests).",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=90,
        help="JPEG quality for re-encoded images.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=5000,
        help="Print progress every N successful writes. 0 disables.",
    )
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    state_path = os.path.join(args.dst, _STATE_FILENAME)
    state = _State.load(state_path)

    print(f"CC3M destination: {args.dst}")
    print(f"HF repo: {_HF_REPO_ID}")
    print(f"State file: {state_path}")
    if state.processed_tars:
        processed_counts = {
            s: len(t) for s, t in state.processed_tars.items()
        }
        print(f"Already-processed tars (resuming): {processed_counts}")

    for split in args.splits:
        _extract_split(
            dst_root=args.dst,
            split=split,
            state=state,
            state_path=state_path,
            max_samples=args.max_samples,
            jpeg_quality=args.jpeg_quality,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
