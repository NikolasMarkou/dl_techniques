"""``final_model.keras`` of an API run is never seen half written (review-iter-1 blind spot).

``finalize_last_epoch`` saved the last epoch's model IN PLACE. A kill inside ``model.save`` left a
torn ``final_model.keras`` in a run whose fit had finished, which ``open_run`` refuses to resume, so
the file stayed torn and the only copy of the last epoch's weights was gone. The API path now asks
for ``atomic=True``: the model is saved to ``final_model.partial.keras`` in the same directory,
flushed to disk and moved over the final name with ``os.replace`` (the way
``AtomicModelCheckpoint`` writes the best model, and through the same function). The legacy
default is unchanged (``atomic=False``, in place).

* a REAL ``SIGKILL`` of a child process in the middle of the save (the target is a model of about
  70 MB, so the window is long; the parent kills the recorded PID as soon as the first byte of the
  file exists) leaves no torn ``final_model.keras`` on the atomic path, and DOES leave one on the
  in-place path (the control that proves the kill lands inside the save);
* a completed atomic save leaves the final file readable and no ``.partial`` behind;
* a failed atomic save (an error from ``save``) leaves an earlier whole final file untouched;
* the API task asks for it and the legacy callers do not.

Writes only under ``tmp_path``; the child is killed by its recorded PID.
"""

import os
import signal
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import keras
import numpy as np
import pytest

from train.common import classification_run
from train.common.callbacks import partial_path

CHILD = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    import keras
    import numpy as np
    from types import SimpleNamespace
    from train.common import classification_run

    run_dir, atomic = Path(sys.argv[1]), sys.argv[2] == "atomic"
    inputs = keras.Input((8,))
    model = keras.Model(inputs, keras.layers.Dense(4096 * 4, use_bias=False)(keras.layers.Dense(4096 * 4, use_bias=False)(inputs)))
    last = SimpleNamespace(weights=model.get_weights())
    data = SimpleNamespace(test_ds=None, select_ds=None, x_fig=None)
    print("READY", flush=True)
    classification_run.finalize_last_epoch(
        model, last, data, run_dir, SimpleNamespace(history={}), 1, lambda m, d: {}, **({"atomic": True} if atomic else {}),
    )
    """
)


def readable(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def killed_during_the_save(run_dir: Path, mode: str) -> dict:
    """Run the child, SIGKILL it once a file of the save exists, and report what it left behind."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="3")
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(run_dir), mode], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
        text=True,
    )
    pid = child.pid  # the only process this test ever signals
    try:
        assert child.stdout.readline().strip() == "READY"
        seen = None
        deadline = time.monotonic() + 120
        names = ("final_model.keras", "final_model.partial.keras")
        while time.monotonic() < deadline and child.poll() is None:
            seen = next((name for name in names if (run_dir / name).exists()), None)
            if seen:
                os.kill(pid, signal.SIGKILL)
                break
    finally:
        if child.poll() is None:
            os.kill(pid, signal.SIGKILL)
        child.wait(timeout=60)
    return {
        "seen_while_saving": seen,
        "final_exists": (run_dir / "final_model.keras").exists(),
        "final_readable": readable(run_dir / "final_model.keras"),
        "partial_exists": (run_dir / "final_model.partial.keras").exists(),
    }


def test_a_kill_inside_the_in_place_save_tears_the_final_file_the_control(tmp_path):
    """The legacy path: the control that shows the kill lands while the file is being written."""
    result = killed_during_the_save(tmp_path, "inplace")
    assert result["seen_while_saving"] == "final_model.keras", result
    assert result["final_exists"] and not result["final_readable"], result


def test_a_kill_inside_the_atomic_save_leaves_no_torn_final_file(tmp_path):
    result = killed_during_the_save(tmp_path, "atomic")
    assert result["seen_while_saving"] == "final_model.partial.keras", result
    assert not result["final_exists"], result  # not torn, and not there: the partial is the casualty


def test_a_completed_atomic_save_leaves_a_whole_final_file_and_no_partial(tmp_path):
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(2)])
    data = SimpleNamespace(test_ds=object(), select_ds=None, x_fig=np.zeros((4, 3), np.float32))
    classification_run.finalize_last_epoch(
        model, SimpleNamespace(weights=model.get_weights()), data, tmp_path, SimpleNamespace(history={}), 1,
        lambda m, d: {}, atomic=True,
    )
    assert readable(tmp_path / "final_model.keras")
    assert not (tmp_path / "final_model.partial.keras").exists()
    assert keras.models.load_model(tmp_path / "final_model.keras", compile=False).count_params() == model.count_params()


def test_a_failed_atomic_save_keeps_an_earlier_whole_final_file(tmp_path):
    keras.Sequential([keras.Input((3,)), keras.layers.Dense(2)]).save(tmp_path / "final_model.keras")
    before = (tmp_path / "final_model.keras").read_bytes()

    class Failing:
        def set_weights(self, weights):
            pass

        def save(self, path):
            Path(path).write_bytes(b"half")
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        classification_run.finalize_last_epoch(
            Failing(), SimpleNamespace(weights=[]), SimpleNamespace(test_ds=None), tmp_path, SimpleNamespace(history={}), 1,
            lambda m, d: {}, atomic=True,
        )
    assert (tmp_path / "final_model.keras").read_bytes() == before


def test_the_legacy_default_saves_in_place_and_the_api_task_asks_for_the_atomic_path():
    import inspect

    from train.api.tasks import image_classification

    assert inspect.signature(classification_run.finalize_last_epoch).parameters["atomic"].default is False
    source = inspect.getsource(image_classification.ImageClassificationTrainer.evaluate_and_report)
    assert "atomic=True" in source
    assert partial_path("/x/final_model.keras") == "/x/final_model.partial.keras"
