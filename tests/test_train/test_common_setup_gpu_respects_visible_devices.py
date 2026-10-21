"""``setup_gpu(gpu_id)`` indexes into an exported ``CUDA_VISIBLE_DEVICES``.

Guard for plan-2026-10-14T062616-81938e81 step 13.6 (a) (D-110). Measured at
e259d9977: ``setup_gpu`` wrote ``CUDA_VISIBLE_DEVICES = str(gpu_id)`` whatever
the environment held, so ``CUDA_VISIBLE_DEVICES=1 ... --gpu 0`` ran on
physical GPU 0. Every Step 13.5 trainer smoke landed on the RTX 4090 that way.
Ruling: when ``CUDA_VISIBLE_DEVICES`` is set, ``gpu_id`` indexes into that
list (out of range raises); only when it is unset does ``gpu_id`` name a
physical device.

Step 11.1 (D-234): ``setup_gpu`` logged and swallowed the ``RuntimeError``
``set_memory_growth`` raises once TensorFlow has initialised its devices, and
nothing checked that a ``gpu_id`` selection took effect (a process that had
already listed its devices kept all of them). Both now raise, naming the
import-order cause.

Environment-isolated: ``monkeypatch`` owns ``CUDA_VISIBLE_DEVICES`` and the TF
device calls are replaced by stubs, so no test here touches a real device.
"""

import os
import sys
from types import SimpleNamespace

import pytest

import train.common.gpu as gpu_module

VAR = "CUDA_VISIBLE_DEVICES"


def _devices(count):
    return [SimpleNamespace(name=f"/physical_device:GPU:{i}", device_type="GPU")
            for i in range(count)]


def _stub_devices(monkeypatch, count, growth_error=None):
    """``count`` listed GPUs; ``set_memory_growth`` records, or raises."""
    grown = []

    def set_memory_growth(device, enable):
        if growth_error is not None:
            raise growth_error
        grown.append((device.name, enable))

    monkeypatch.setattr(gpu_module.tf.config.experimental,
                        "list_physical_devices", lambda *_a, **_k: _devices(count))
    monkeypatch.setattr(gpu_module.tf.config.experimental,
                        "set_memory_growth", set_memory_growth)
    return grown


@pytest.fixture(autouse=True)
def _no_device_access(monkeypatch):
    """Stub the two device calls ``setup_gpu`` makes: one GPU listed, as a
    selection that took effect leaves."""
    _stub_devices(monkeypatch, 1)


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


# --- step 11.1: a selection or memory growth that did not take effect raises ---

@pytest.mark.parametrize("listed", [0, 2])
def test_a_selection_that_did_not_take_effect_raises(monkeypatch, listed):
    """With ``gpu_id`` the process must list exactly one GPU afterwards. Two
    listed means TensorFlow read the variable before the write (the devices
    are cached at the first listing); none means the selection names no GPU."""
    _stub_devices(monkeypatch, listed)
    monkeypatch.setenv(VAR, "1")
    with pytest.raises(RuntimeError, match=rf"lists {listed} GPU\(s\), not 1"):
        gpu_module.setup_gpu(0)


@pytest.mark.parametrize("listed", [0, 2])
def test_no_gpu_id_accepts_any_device_count(monkeypatch, listed):
    grown = _stub_devices(monkeypatch, listed)
    monkeypatch.setenv(VAR, "0,1")
    gpu_module.setup_gpu(None)
    assert grown == [(f"/physical_device:GPU:{i}", True) for i in range(listed)]


@pytest.mark.parametrize("gpu_id", [None, 0])
def test_a_memory_growth_failure_raises_and_names_the_cause(monkeypatch, gpu_id):
    """It was logged as "GPU setup error" and the run went on."""
    cause = RuntimeError("Physical devices cannot be modified after being initialized")
    _stub_devices(monkeypatch, 1, growth_error=cause)
    monkeypatch.setenv(VAR, "1")
    with pytest.raises(RuntimeError, match="before setup_gpu ran") as raised:
        gpu_module.setup_gpu(gpu_id)
    assert raised.value.__cause__ is cause


def test_memory_growth_is_enabled_on_the_selected_gpu(monkeypatch):
    grown = _stub_devices(monkeypatch, 1)
    monkeypatch.setenv(VAR, "1")
    gpu_module.setup_gpu(0)
    assert grown == [("/physical_device:GPU:0", True)]


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


# ---------------------------------------------------------------------------
# Census: train/common/gpu.py is the only writer of the variable (step 13.7, D-116)
# ---------------------------------------------------------------------------
#
# Measured at b50fd4e80: after D-110, seventeen statements outside gpu.py still
# wrote the variable -- six ``--gpu`` translations of their own (two sweeps,
# two comparison drivers, two evaluation scripts), the two drivers' own CPU
# pins, a ``setdefault`` in the logic multi-seed sweep and eight CPU pins in
# the time-series export scripts. A second writer is a second rule for one
# flag, which is how D-110's bug survived in six places.

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

_SOURCE_ROOT = Path(gpu_module.__file__).resolve().parents[2]
_THE_ONE_WRITER = Path(gpu_module.__file__).resolve()
_WRITING_METHODS = {"setdefault", "putenv", "unsetenv", "pop", "__setitem__",
                    "__delitem__"}


def _writes_to_the_variable(source):
    """Line numbers of every statement in ``source`` that writes ``VAR``.

    A write is: a subscript store or ``del`` keyed by the name; ``setdefault`` /
    ``pop`` / ``putenv`` / ``unsetenv`` with it as the first argument; a dict
    literal or a call keyword (``env.update(CUDA_VISIBLE_DEVICES=...)``,
    ``dict(os.environ, CUDA_VISIBLE_DEVICES=...)``) carrying it. The key may be
    the literal or a module name bound to it. Reads (``get``, ``in``, a load)
    are not writes.
    """
    tree = ast.parse(source)
    aliases = {
        target.id
        for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant) and node.value.value == VAR
        for target in node.targets if isinstance(target, ast.Name)
    }

    def is_key(node):
        return ((isinstance(node, ast.Constant) and node.value == VAR)
                or (isinstance(node, ast.Name) and node.id in aliases))

    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = (node.targets if isinstance(node, (ast.Assign, ast.Delete))
                       else [node.target])
            if any(isinstance(sub, ast.Subscript) and is_key(sub.slice)
                   for target in targets for sub in ast.walk(target)):
                lines.add(node.lineno)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _WRITING_METHODS and node.args and is_key(node.args[0]):
                lines.add(node.lineno)
            if any(keyword.arg == VAR for keyword in node.keywords):
                lines.add(node.lineno)
        elif isinstance(node, ast.Dict):
            if any(key is not None and is_key(key) for key in node.keys):
                lines.add(node.lineno)
    return sorted(lines)


_EVERY_WRITE_FORM = f'''
import os
NAME = "{VAR}"
os.environ["{VAR}"] = "1"
env["{VAR}"] = str(gpu)
os.environ[NAME] = ""
env.setdefault("{VAR}", "0")
os.environ.pop("{VAR}", None)
del os.environ["{VAR}"]
os.putenv("{VAR}", "1")
env.update({VAR}="1")
child = dict(os.environ, {VAR}="1")
child = {{**os.environ, "{VAR}": "1"}}
'''

_READS_ONLY = f'''
import os
a = os.environ.get("{VAR}")
b = "{VAR}" in os.environ
c = os.environ["{VAR}"]
logger.info("set {VAR}=1 in the shell")
'''


def test_the_census_sees_every_write_form_and_no_read():
    """Positive control: each write line is found, and nothing else."""
    lines = _EVERY_WRITE_FORM.splitlines()
    expected = [i for i, line in enumerate(lines, start=1)
                if line and not line.startswith(("import", "NAME"))]
    assert _writes_to_the_variable(_EVERY_WRITE_FORM) == expected
    assert _writes_to_the_variable(_READS_ONLY) == []


def test_the_census_finds_the_writes_in_gpu_py():
    """Positive control on a real file: the one writer is seen as a writer."""
    assert _writes_to_the_variable(_THE_ONE_WRITER.read_text(encoding="utf-8"))


def test_no_module_outside_gpu_py_writes_the_variable():
    offenders = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        if path.resolve() == _THE_ONE_WRITER:
            continue
        for line in _writes_to_the_variable(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(_SOURCE_ROOT)}:{line}")
    assert offenders == [], (
        f"{VAR} is written outside train/common/gpu.py; select through "
        "visible_device_for / setup_gpu / env_with_visible_devices / hide_gpus "
        "instead:\n" + "\n".join(offenders))
