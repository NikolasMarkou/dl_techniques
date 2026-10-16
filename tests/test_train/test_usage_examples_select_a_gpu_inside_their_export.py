"""A documented command never asks for a GPU outside the list it exports.

Guard for plan-2026-10-14T062616-81938e81 step 14 (D-119). Since D-110,
``--gpu N`` counts within an exported ``CUDA_VISIBLE_DEVICES``, so the usage
line ``CUDA_VISIBLE_DEVICES=1 ... --gpu 1`` raises ``ValueError`` before the run
starts. Measured at 7bd01f8c5: 31 such commands in 27 files under ``src/``
(27 in the docstrings of 25 scripts, the CliffordCLIP evaluation and filter
scripts among them, 3 in ``bfunet/FINDINGS.md`` and the sweep call in
``rms_variants_train/run_n15_chunk1.sh``).

The census joins backslash continuations, so a command spread over several
lines is read as one.
"""

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
_EXPORT = re.compile(r"CUDA_VISIBLE_DEVICES=([0-9][0-9,]*)")
_GPU = re.compile(r"--gpu[ =]+(\d+)")


def _commands_outside_their_export(text):
    """``(first line, exported list, index)`` for every command that overreaches."""
    found, command, start = [], "", None
    for number, raw in enumerate(text.splitlines(), start=1):
        if start is None:
            start = number
        line = raw.rstrip()
        if line.endswith("\\"):
            command += line.rstrip("\\") + " "
            continue
        command += line
        export, gpu = _EXPORT.search(command), _GPU.search(command)
        if export and gpu:
            entries = [entry for entry in export.group(1).split(",") if entry]
            if int(gpu.group(1)) >= len(entries):
                found.append((start, export.group(1), int(gpu.group(1))))
        command, start = "", None
    return found


def test_the_census_reads_continued_commands():
    """Positive control: one-line and continued overreaches, and no fit."""
    text = (
        "    CUDA_VISIBLE_DEVICES=1 python -m train.x --gpu 1\n"      # 1
        "    CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python \\\\\n"     # 2
        "        -m train.x \\\\\n"
        "        --epochs 3 --gpu 1\n"
        "    CUDA_VISIBLE_DEVICES=1 python -m train.x --gpu 0\n"      # fits
        "    CUDA_VISIBLE_DEVICES=0,1 python -m train.x --gpu 1\n"    # fits
        "    python -m train.x --gpu 1\n"                             # no export
    )
    assert [line for line, _, _ in _commands_outside_their_export(text)] == [1, 2]


def test_no_documented_command_asks_for_a_gpu_outside_its_export():
    paths = [path for suffix in ("*.py", "*.md", "*.sh")
             for path in sorted(_SRC.rglob(suffix))]
    assert len(paths) > 1000, "the census would pass vacuously"
    offenders = [
        f"{path.relative_to(_SRC)}:{line}: CUDA_VISIBLE_DEVICES={export} --gpu {index}"
        for path in paths
        for line, export, index in _commands_outside_their_export(
            path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "--gpu counts within the exported list (D-110); drop the flag or use an "
        "index inside the list:\n" + "\n".join(offenders))
