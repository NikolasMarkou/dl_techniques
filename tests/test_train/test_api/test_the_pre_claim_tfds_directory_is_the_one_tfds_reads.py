"""The directory the pre-claim Imagenette check looks in is the directory TFDS reads (review-iter-2 concern 3).

``data_args._tfds_data_dir`` answers, before the run is claimed and without importing TFDS, the
question "where would TFDS look for ``imagenette/320px-v2``?". A wrong answer is a FALSE REFUSAL of
a layout the loader reads, the failure D-034 calls the worst one: ``TFDS_DATA_DIR='~/t'`` was
refused (``~/t`` taken as a relative path) while TFDS read ``$HOME/t`` and its 3,925 validation
records.

TFDS's own answer is ``tensorflow_datasets.core.utils.file_utils.list_data_dirs()``, evaluated from
``$TFDS_DATA_DIR`` when TFDS is imported (``constants.DATA_DIR``), so each case runs in a child with
its environment set before the import and prints both answers. Four environments:

* an absolute path (the control: the two agree);
* ``~/t``: TFDS expands the user directory;
* the EMPTY string: TFDS reads ``.`` (a variable that is set, even to nothing, wins over the
  default; mutant ``da_empty_var`` treated it as unset and looked in ``~/tensorflow_datasets``);
* unset: ``~/tensorflow_datasets``.

The comparison is the answer's text, so a case where the two spell one directory differently
would fail loudly.

Tests write only under ``tmp_path``.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

DRIVER = textwrap.dedent(
    f"""
    import json
    import sys

    sys.path.insert(0, {str(REPO_ROOT)!r})
    from tensorflow_datasets.core.utils import file_utils

    from train.api.data_args import _tfds_data_dir

    print(json.dumps({{"tfds": [str(p) for p in file_utils.list_data_dirs()], "ours": str(_tfds_data_dir())}}))
    """
)


def both_answers(tmp_path, variable):
    """Return ``(TFDS's first data directory, this package's answer)`` for ``$TFDS_DATA_DIR`` = ``variable``."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "TFDS_DATA_DIR"}
    env.update(HOME=str(home), TF_CPP_MIN_LOG_LEVEL="3", CUDA_VISIBLE_DEVICES="-1")
    if variable is not None:
        env["TFDS_DATA_DIR"] = variable
    done = subprocess.run([sys.executable, "-c", DRIVER], capture_output=True, text=True, timeout=600,
                          cwd=tmp_path, env=env)
    assert done.returncode == 0, done.stderr[-3000:]
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    return seen["tfds"][0], seen["ours"]


@pytest.mark.parametrize("variable, expected", [
    ("{tmp}/plain", "{tmp}/plain"),
    ("~/t", "{tmp}/home/t"),
    ("", "."),
    (None, "{tmp}/home/tensorflow_datasets"),
], ids=["absolute", "tilde", "empty", "unset"])
def test_the_package_answers_what_tfds_answers(tmp_path, variable, expected):
    if variable is not None:
        variable = variable.format(tmp=tmp_path)
    tfds_answer, ours = both_answers(tmp_path, variable)
    assert tfds_answer == expected.format(tmp=tmp_path), "TFDS itself changed: the expectation is stale"
    assert ours == tfds_answer
