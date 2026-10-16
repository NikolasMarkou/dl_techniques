"""``setup_gpu(gpu_id)`` indexes into an exported ``CUDA_VISIBLE_DEVICES``.

Guard for plan-2026-10-14T062616-81938e81 step 13.6 (a) (D-110). Measured at
e259d9977: ``setup_gpu`` wrote ``CUDA_VISIBLE_DEVICES = str(gpu_id)`` whatever
the environment held, so ``CUDA_VISIBLE_DEVICES=1 ... --gpu 0`` ran on
physical GPU 0. Every Step 13.5 trainer smoke landed on the RTX 4090 that way.
Ruling: when ``CUDA_VISIBLE_DEVICES`` is set, ``gpu_id`` indexes into that
list (out of range raises); only when it is unset does ``gpu_id`` name a
physical device.

Environment-isolated: ``monkeypatch`` owns ``CUDA_VISIBLE_DEVICES`` and the TF
device listing is replaced by a stub, so no test here touches a real device.
"""

import os
import sys

import pytest

import train.common.gpu as gpu_module

VAR = "CUDA_VISIBLE_DEVICES"


@pytest.fixture(autouse=True)
def _no_device_access(monkeypatch):
    """Stub the only device call ``setup_gpu`` makes."""
    monkeypatch.setattr(
        gpu_module.tf.config.experimental, "list_physical_devices",
        lambda *_args, **_kwargs: [],
    )


def _run(monkeypatch, visible, gpu_id):
    if visible is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, visible)
    gpu_module.setup_gpu(gpu_id)
    return os.environ.get(VAR)


@pytest.mark.parametrize("visible, gpu_id, expected", [
    (None, 0, "0"),          # unset: a physical index
    (None, 1, "1"),
    ("1", 0, "1"),           # the Step 13.5 case: GPU 1 exported, --gpu 0
    ("1,0", 0, "1"),
    ("1,0", 1, "0"),
    (" 2 , 3 ", 1, "3"),     # whitespace around entries is not part of an id
    ("GPU-1a2b,GPU-3c4d", 1, "GPU-3c4d"),  # UUID entries pass through
])
def test_gpu_id_selects_from_the_visible_list(monkeypatch, visible, gpu_id, expected):
    assert _run(monkeypatch, visible, gpu_id) == expected


@pytest.mark.parametrize("visible, gpu_id", [
    ("1", 1),     # one GPU visible, index 1 asked for
    ("1,0", 2),
    ("", 0),      # GPUs hidden on purpose
    ("1", -1),
    (None, -1),   # not a device in either reading
])
def test_an_index_outside_the_visible_list_is_refused(monkeypatch, visible, gpu_id):
    if visible is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, visible)
    with pytest.raises(ValueError, match=VAR):
        gpu_module.setup_gpu(gpu_id)
    # A refused selection leaves the environment as it was.
    assert os.environ.get(VAR) == visible


@pytest.mark.parametrize("visible", [None, "1", "0,1"])
def test_no_gpu_id_leaves_the_environment_alone(monkeypatch, visible):
    assert _run(monkeypatch, visible, None) == visible


def test_the_downsampling_trainer_selects_through_setup_gpu(monkeypatch):
    """It wrote ``str(args.gpu)`` itself before calling ``setup_gpu(None)``."""
    import train.cliffordnet.train_downsampling_techniques as trainer

    seen = []
    monkeypatch.setattr(trainer, "setup_gpu", lambda gpu_id=None: seen.append(gpu_id))
    monkeypatch.setattr(trainer, "train_all_variants", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["train_downsampling_techniques", "--gpu", "0"])
    monkeypatch.setenv(VAR, "1")
    trainer.main()
    assert seen == [0]
    assert os.environ[VAR] == "1"


def test_the_clifford_clip_trainer_does_not_preselect_at_import():
    """Its import-time block wrote ``--gpu``'s value before ``setup_gpu`` ran.

    Two selections cannot both index into the list: the first one would have
    already narrowed it to one entry.
    """
    from pathlib import Path

    source = (Path(gpu_module.__file__).resolve().parents[1]
              / "cliffordnet" / "train_clip.py").read_text(encoding="utf-8")
    assert f'os.environ["{VAR}"] =' not in source
    assert "setup_gpu(args.gpu)" in source
