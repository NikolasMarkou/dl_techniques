"""A stand-in experiment worker: drives the real subprocess path of the cell runner without TensorFlow.

Run as ``python -m tests.test_train.test_mean_shift.fake_worker --worker --cell-dir D --arm A
--config C --seed S [--behavior B]`` from the repository root. The behaviour decides what the
"training" does, so each test can make one cell succeed, fail, hang or lie.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from train.mean_shift import common


def _history(loss):
    return {"loss": loss, "accuracy": [0.1, 0.2][: len(loss)],
            "epoch_seconds": [3.0, 1.0][: len(loss)]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    common.add_worker_arguments(parser)
    parser.add_argument("--behavior", default="ok")
    namespace = parser.parse_args(argv)
    cell_dir = Path(namespace.cell_dir)
    behavior = namespace.behavior
    sys.stderr.write(f"fake worker {namespace.arm} {namespace.config} {namespace.seed} {behavior}\n")

    if behavior == "fail" or (behavior == "fail_first_try" and cell_dir.name.endswith("__try1")):
        sys.stderr.write("boom: the fake training diverged\n")
        return 3
    if behavior == "hang":
        time.sleep(120)
        return 0
    if behavior == "noresult":
        return 0
    loss = [float("nan"), float("nan")] if behavior == "nan" else [2.0, 1.5]
    result = common.build_result(
        _history(loss), params=1234,
        metrics={"accuracy": 0.5 + 0.01 * namespace.seed,
                 "seen_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "seen_mplbackend": os.environ.get("MPLBACKEND"),
                 "manifest_present": (cell_dir.parents[1] / "manifest.json").is_file(),
                 "rows_present": (cell_dir.parents[1] / "rows.jsonl").is_file(),
                 "cwd": os.getcwd()})
    common.write_result(cell_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
