"""Nothing the API does before ``setup_gpu`` may initialise TensorFlow's devices.

Gate 2 found it on the GPU: ``python -m train image_classification ...`` raised
``Could not enable memory growth on /physical_device:GPU:0: Physical devices
cannot be modified after being initialized``. Step 16.1 had added a monitor check
that called ``classification_metrics`` to read the metric names; a Keras metric
allocates variables, the first eager op creates TensorFlow's eager context and
its devices, and ``setup_gpu`` (which comes after the pre-claim checks and the
claim) can no longer set memory growth. A CPU-only suite never sees the error:
with no GPU listed there is no growth to set.

The property is therefore asserted without a GPU. In a child process the real
``train_image_classification`` runs (fake CIFAR-10 arrays installed, ``vit_pico``)
with ``train.common.gpu.setup_gpu`` replaced by a recorder, which reads
``tensorflow.python.eager.context.context()._context_handle`` at the moment of the
call. That handle is ``None`` until the first op, a device listing leaves it
``None``, and a ``keras.ops.zeros`` makes it non-``None`` (the control arm
measures this in the same child, so a TensorFlow release that moves the
attribute turns the control red rather than the test blind). The recorder then
stops the run: only the state at the call matters.

Both ``devices`` paths are covered: the default (``setup_gpu(None)``) and an
explicit index (``setup_gpu(0)``). The tests write only under ``tmp_path``.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: ``argv``: out, devices ("auto" or an index), "control" to run an op first.
DRIVER = textwrap.dedent(
    f"""
    import json
    import sys

    sys.path.insert(0, {str(REPO_ROOT)!r})
    from tensorflow.python.eager import context as eager

    def initialised():
        return eager.context()._context_handle is not None

    import keras
    import train.common.classification_data as classification_data
    import train.common.gpu as gpu
    from tests.classification_fakes import tiny_raw
    from train.api.tasks.image_classification import train_image_classification

    classification_data._load_raw = tiny_raw(n_train=40, n_test=8)
    out, devices, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    seen = {{"at_start": initialised(), "calls": []}}


    class Recorded(Exception):
        pass


    def recorder(gpu_id=None, flag="--gpu"):
        seen["calls"].append({{"gpu_id": gpu_id, "flag": flag, "initialised": initialised()}})
        raise Recorded


    gpu.setup_gpu = recorder
    if mode == "control":
        keras.ops.zeros(1)
        seen["after_op"] = initialised()
    kwargs = {{}} if devices == "auto" else {{"devices": int(devices)}}
    try:
        train_image_classification(
            out=out, model="vit/vit_pico", data={{"dataset": "cifar10"}},
            epochs=1, batch_size=4, **kwargs,
        )
    except Recorded:
        pass
    print(json.dumps(seen))
    """
)


def record(tmp_path: Path, devices: str, mode: str = "real") -> dict:
    """Run the driver in a child and return what the recorder saw."""
    env = {**os.environ, "MPLBACKEND": "Agg", "TF_CPP_MIN_LOG_LEVEL": "3"}
    # The child never reaches the real setup_gpu, so the value only has to name a
    # device the pre-claim check accepts.
    env["CUDA_VISIBLE_DEVICES"] = "0" if devices != "auto" else "-1"
    done = subprocess.run(
        [sys.executable, "-c", DRIVER, str(tmp_path / "run"), devices, mode],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=tmp_path,
        env=env,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_control_an_op_initialises_the_devices_and_the_recorder_sees_it(tmp_path):
    seen = record(tmp_path, "auto", mode="control")
    assert seen["at_start"] is False
    assert seen["after_op"] is True
    assert [call["initialised"] for call in seen["calls"]] == [True]


@pytest.mark.parametrize(
    "devices, gpu_id", [("auto", None), ("0", 0)], ids=["devices_auto", "devices_index"]
)
def test_setup_gpu_is_called_once_before_tensorflow_initialises_its_devices(
    tmp_path, devices, gpu_id
):
    seen = record(tmp_path, devices)
    assert seen["at_start"] is False
    assert seen["calls"] == [{"gpu_id": gpu_id, "flag": "devices", "initialised": False}], seen
