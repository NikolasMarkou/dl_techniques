"""Importing the DINO trainer leaves TensorFlow's devices untouched, so
``setup_gpu`` selects the GPU.

The module docstring said the opposite until plan-2026-10-14T062616-81938e81
step 11.1: that TensorFlow was initialised before ``setup_gpu`` in this
trainer's import order, so ``--gpu`` was "INERT in BOTH halves" and only the
``CUDA_VISIBLE_DEVICES`` prefix selected a device. Measured (F-9 D13 and here):
after ``import train.dino.train_dino`` the eager context is not initialised and
no physical device has been listed, and ``setup_gpu`` is the first statement of
``train_dino``. Each check runs in a fresh process, the only place import order
is observable.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import train.dino.train_dino as train_dino

_SRC = Path(train_dino.__file__).resolve().parents[2]

_PROBE = (
    "import train.dino.train_dino\n"
    "from tensorflow.python.eager import context\n"
    "ctx = context.context()\n"
    "print('STATE', ctx._initialized, ctx._physical_devices is None)\n"
)


def _fresh(code: str, visible: str = None) -> list:
    env = dict(os.environ, MPLBACKEND="Agg")
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    done = subprocess.run([sys.executable, "-c", code], env=env,
                          cwd=str(_SRC.parent), capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return [line for line in done.stdout.splitlines()
            if line.startswith(("STATE", "GPUS"))]


def test_the_import_initialises_no_device():
    assert _fresh(_PROBE) == ["STATE False True"]


def test_the_probe_sees_an_initialised_context():
    """Positive control: one eager op before the read flips both flags."""
    assert _fresh("import tensorflow as tf\ntf.constant(0)\n" + _PROBE) == [
        "STATE True False"]


def test_setup_gpu_after_the_import_selects_one_gpu():
    """Index 0 of the exported list, after the trainer's imports: one GPU and
    no raise (the selection and the memory growth both take effect)."""
    exported = os.environ.get("CUDA_VISIBLE_DEVICES")
    probe = ("import tensorflow as tf\n"
             "print('GPUS', len(tf.config.list_physical_devices('GPU')))\n")
    if _fresh(probe, exported) == ["GPUS 0"]:
        pytest.skip("no GPU visible: nothing to select")
    selected = _fresh(
        "import train.dino.train_dino\n"
        "from train.common import setup_gpu\n"
        "setup_gpu(0)\n" + probe, exported)
    assert selected == ["GPUS 1"]


def test_the_docstring_does_not_call_gpu_inert():
    doc = train_dino.__doc__
    assert "INERT" not in doc and "will not help" not in doc
    assert "CUDA_VISIBLE_DEVICES=N" in doc
