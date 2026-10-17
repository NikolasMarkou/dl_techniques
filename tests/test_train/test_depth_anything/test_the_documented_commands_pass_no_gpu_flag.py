"""The Depth Anything trainer's documented commands select the GPU by export only.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 1.6 (the D-110
rule). ``--gpu N`` indexes into the visible devices: with no
``CUDA_VISIBLE_DEVICES`` export, ``--gpu 0`` is the RTX 4090, and the
iteration-1 smokes ran there by following exactly such a line. At 308a4dfbf
the trainer's README advised ``--gpu 0`` or ``--gpu 1`` and four of its
commands, and its module docstring two, passed ``--gpu 0`` with no export.
"""

import re
from pathlib import Path

from train.depth_anything import train_depth_anything as module

_README = Path(module.__file__).with_name("README.md")


def _commands(text):
    """Every ``train.depth_anything.train_depth_anything`` command, continuations joined."""
    commands, current = [], ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith("\\"):
            current += stripped.rstrip("\\") + " "
            continue
        current += stripped
        if "train.depth_anything.train_depth_anything" in current:
            commands.append(re.sub(r"\s+", " ", current))
        current = ""
    return commands


def test_no_documented_command_passes_gpu_and_each_exports_the_device():
    for source, text in (("README.md", _README.read_text(encoding="utf-8")),
                         ("module docstring", module.__doc__)):
        commands = _commands(text)
        assert commands, f"{source}: no command found, the census would pass vacuously"
        for command in commands:
            assert "--gpu" not in command, f"{source}: {command}"
            assert command.startswith("CUDA_VISIBLE_DEVICES="), f"{source}: {command}"


def test_the_readme_gives_no_gpu_index_advice():
    assert not re.search(r"--gpu[ =]\d", _README.read_text(encoding="utf-8"))
