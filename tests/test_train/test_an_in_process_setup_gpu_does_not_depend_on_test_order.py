"""An in-process ``setup_gpu`` passes whatever ran before it in the session.

Since plan-2026-10-14T062616-81938e81 step 11.1 (D-234) ``setup_gpu`` raises
when memory growth cannot be enabled, which TensorFlow refuses once its
devices are initialised. About 70 trainer test files call a ``main`` in
process; each passed only if the first ``setup_gpu`` of its pytest process ran
before any op (130 failed and 78 errors with an op first). ``tests/conftest.py``
enables growth before any test module is imported, as a trainer's first
``setup_gpu`` does, so a later call is TensorFlow's early-return no-op.
"""

import os

import pytest
import tensorflow as tf

from train.common.gpu import setup_gpu

_GPUS = tf.config.list_physical_devices("GPU")
pytestmark = pytest.mark.skipif(not _GPUS, reason="no GPU visible: growth is moot")


def test_memory_growth_is_on_for_every_visible_gpu():
    assert all(tf.config.experimental.get_memory_growth(gpu) for gpu in _GPUS)


def test_setup_gpu_after_an_op_does_not_raise():
    tf.constant(1.0) + 1.0  # the context is initialised now, whatever ran first
    setup_gpu(None)


@pytest.mark.skipif(len(_GPUS) != 1, reason="selection needs a one-GPU export")
def test_selecting_the_exported_gpu_after_an_op_does_not_raise(monkeypatch):
    exported = os.environ.get("CUDA_VISIBLE_DEVICES")
    if exported is None or "," in exported:
        pytest.skip("selection by index 0 needs a one-entry CUDA_VISIBLE_DEVICES")
    tf.constant(1.0) + 1.0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", exported)
    setup_gpu(0)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == exported
