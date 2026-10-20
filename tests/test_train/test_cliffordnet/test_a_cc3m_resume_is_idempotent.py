"""A resumed CC3M extraction ends with the same captions under the same ids (D-187).

``prepare_cc3m._extract_split`` resumes at the tar-shard boundary. Before D-187 a
shard that failed partway was not marked done, so a rerun extracted it again and
appended its captions a second time under new ids; the shards after it, extracted in
the same run, had already taken the ids that followed the partial shard. Each case
below interrupts an extraction, resumes it in a "new process" (the state is reloaded
from disk), and compares the result with one uninterrupted run: the same
``(id, caption)`` lines in the same order, and every image the one its caption
belongs to. The tar stream and the shard listing are stubbed; everything is written
under ``tmp_path``.
"""

import json
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from train.cliffordnet import prepare_cc3m as module

#: Shard -> pairs, in tar order. ``None`` bytes is a corrupt JPEG; an empty caption
#: is skipped. Both are dropped deterministically, so they shift ids the same way in
#: every run.
_SHARDS = {
    "s0.tar": [(0, "a dog"), (1, "a cat"), (2, "a hat")],
    "s1.tar": [(3, "a car"), (4, ""), (5, "a bus"), (None, "corrupt"), (6, "a cup"),
               (7, "a pen")],
    "s2.tar": [(8, "a map"), (9, "a box")],
}


class _Interrupt(BaseException):
    """Stands in for a kill: not an ``Exception``, so nothing in the extractor catches it."""


def _colour(index):
    return (index * 25, 255 - index * 25, (index * 70) % 256)


def _jpeg(index):
    if index is None:
        return b"not a jpeg"
    buffer = BytesIO()
    Image.new("RGB", (4, 4), _colour(index)).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


class _Tar:
    def __init__(self, name):
        self.name = name

    def close(self):
        pass


def _stub_hub(monkeypatch, fail_at=None):
    """Serve ``_SHARDS``; ``fail_at = (shard, after_n_pairs, exception)`` fails once."""
    pending = [fail_at] if fail_at else []

    def iterate(tar):
        for position, (index, caption) in enumerate(_SHARDS[tar.name]):
            if pending and pending[0][0] == tar.name and pending[0][1] == position:
                _shard, _n, exc = pending.pop()
                raise exc
            yield _jpeg(index), caption

    monkeypatch.setattr(module, "_list_shards", lambda split: list(_SHARDS))
    monkeypatch.setattr(module, "_open_tar_stream", _Tar)
    monkeypatch.setattr(module, "_iter_pairs_in_tar", iterate)


def _extract(root, max_samples=None):
    """One process: load the state from disk, extract the train split."""
    state_path = str(root / module._STATE_FILENAME)
    return module._extract_split(
        dst_root=str(root), split="train", state=module._State.load(state_path),
        state_path=state_path, max_samples=max_samples, jpeg_quality=95,
        progress_every=0,
    )


def _result(root):
    """``[(id, caption)]`` from the JSONL, each image checked against its caption."""
    lines = [json.loads(line) for line in
             (root / "train_captions.jsonl").read_text().splitlines()]
    index_of = {caption: index for pairs in _SHARDS.values()
                for index, caption in pairs if index is not None}
    for line in lines:
        path = root / "train" / module._shard_of(line["id"]) / f"{line['id']}.jpg"
        pixel = np.asarray(Image.open(path).convert("RGB"), dtype=float).mean(axis=(0, 1))
        expected = np.array(_colour(index_of[line["caption"]]), dtype=float)
        assert np.abs(pixel - expected).max() < 6, (line, pixel, expected)
    return [(line["id"], line["caption"]) for line in lines]


@pytest.fixture
def reference(monkeypatch, tmp_path):
    _stub_hub(monkeypatch)
    root = tmp_path / "clean"
    assert _extract(root) == (9, 0)
    result = _result(root)
    assert [caption for _id, caption in result] == [
        "a dog", "a cat", "a hat", "a car", "a bus", "a cup", "a pen", "a map", "a box"]
    assert len({image_id for image_id, _c in result}) == len(result)
    return result


def test_a_shard_that_fails_partway_is_extracted_once_on_resume(
        monkeypatch, tmp_path, reference):
    root = tmp_path / "failed"
    _stub_hub(monkeypatch, fail_at=("s1.tar", 3, ConnectionError("stream reset (test)")))
    _written, errors = _extract(root)
    assert errors == 1
    _stub_hub(monkeypatch)
    _extract(root)
    assert _result(root) == reference


def test_a_run_killed_inside_a_shard_resumes_to_the_same_ids(
        monkeypatch, tmp_path, reference):
    root = tmp_path / "killed"
    _stub_hub(monkeypatch, fail_at=("s1.tar", 4, _Interrupt()))
    with pytest.raises(_Interrupt):
        _extract(root)
    _stub_hub(monkeypatch)
    _extract(root)
    assert _result(root) == reference


def test_a_capped_run_followed_by_a_full_run_matches_one_full_run(
        monkeypatch, tmp_path, reference):
    root = tmp_path / "capped"
    _stub_hub(monkeypatch)
    _extract(root, max_samples=5)
    _extract(root)
    assert _result(root) == reference


def test_a_finished_extraction_rerun_adds_nothing(monkeypatch, tmp_path, reference):
    """Control: a rerun of a complete extraction writes no line."""
    root = tmp_path / "rerun"
    _stub_hub(monkeypatch)
    _extract(root)
    assert _extract(root) == (0, 0)
    assert _result(root) == reference


def test_a_save_killed_partway_leaves_no_file_at_the_image_path(monkeypatch, tmp_path):
    """A resume skips an existing image, so a killed save must not leave one there."""
    import io

    def killed_save(self, fp, *args, **kwargs):
        with open(fp, "wb") as handle:
            handle.write(b"\xff\xd8 partial")
        raise _Interrupt()

    jpeg = _jpeg(0)  # encoded before save is patched
    monkeypatch.setattr(Image.Image, "save", killed_save)
    with pytest.raises(_Interrupt):
        module._write_pair(img_root=str(tmp_path), split="train", global_idx=0,
                           jpg_bytes=jpeg, caption="a dog", jsonl_fh=io.StringIO(),
                           jpeg_quality=95)
    image_id = "cc3m_train_00000000"
    assert not (tmp_path / module._shard_of(image_id) / f"{image_id}.jpg").exists()
