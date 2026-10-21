"""Child-process launchers and evaluation scripts select a GPU by D-110's rule.

Guard for plan-2026-10-14T062616-81938e81 step 13.7 (a) (D-116). Step 13.6
(D-110) made ``setup_gpu``'s ``--gpu`` count within an exported
``CUDA_VISIBLE_DEVICES``, but six scripts kept their own write. Measured at
b50fd4e80:

- ``rms_variants_train/sweep.py``, ``embeddings_experimental/sweep.py``,
  ``vae/run_sampler_comparison.py`` and ``convnext/run_stochastic_comparison.py``
  gave each child ``CUDA_VISIBLE_DEVICES=str(--gpu)``, so
  ``CUDA_VISIBLE_DEVICES=1 ... --gpu 0`` trained every child on physical GPU 0.
- The VAE launcher also passed its own ``--gpu`` (default 1) to a child whose
  list it had already narrowed to one entry, so the child's ``setup_gpu(1)``
  indexed a one-entry list and raised: the launcher's default invocation could
  not train.
- ``thera/eval_thera.py`` wrote ``str(--gpu)``; ``bfunet/eval_per_pixel_uncertainty.py``
  wrote physical GPU 1 at import even with no flag, over an exported value.

Each case drives the script's real code with the environment patched and the
child launch (or the work after device selection) replaced by a stub, so no
test here starts a child or touches a device.
"""

import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import train.common.gpu as gpu_module

VAR = "CUDA_VISIBLE_DEVICES"


class _Stop(Exception):
    """Raised by a stub where the real script would start real work."""


@pytest.fixture(autouse=True)
def _isolated_visible_devices(monkeypatch):
    """Restore the variable exactly, whatever the code under test wrote.

    ``monkeypatch.delenv`` records nothing when the variable is absent, so a
    write by the code under test would otherwise outlive the test.
    """
    # One GPU listed, growth a no-op: what a selection that took effect leaves
    # (setup_gpu raises otherwise since step 11.1, D-234).
    monkeypatch.setattr(
        gpu_module.tf.config.experimental, "list_physical_devices",
        lambda *_args, **_kwargs: [SimpleNamespace(
            name="/physical_device:GPU:0", device_type="GPU")],
    )
    monkeypatch.setattr(
        gpu_module.tf.config.experimental, "set_memory_growth",
        lambda *_args, **_kwargs: None,
    )
    before = os.environ.get(VAR)
    yield
    if before is None:
        os.environ.pop(VAR, None)
    else:
        os.environ[VAR] = before


def _set_visible(monkeypatch, visible):
    if visible is None:
        monkeypatch.delenv(VAR, raising=False)
    else:
        monkeypatch.setenv(VAR, visible)


def _capture_launch(monkeypatch, module):
    """Replace ``subprocess.run`` in ``module``; record (cmd, env), then stop."""
    seen = []

    def fake_run(cmd, *args, **kwargs):
        seen.append((list(cmd), dict(kwargs["env"])))
        raise _Stop

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return seen


def _reload(monkeypatch, name, argv):
    """(Re)import ``name`` with ``sys.argv`` set, so import-time code runs now."""
    monkeypatch.setattr(sys, "argv", argv)
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    # Executed once: a second execution would read the first one's writes.
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# Sweeps: one resolved device for every cell
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("visible, gpu, expected", [
    ("1", "0", "1"),       # the Step 13.5 case
    ("1,0", "1", "0"),
    (None, "0", "0"),      # unset: --gpu names a physical GPU
])
def test_the_rms_sweep_child_gets_the_resolved_device(
        monkeypatch, tmp_path, visible, gpu, expected):
    from train.rms_variants_train import sweep

    _set_visible(monkeypatch, visible)
    seen = _capture_launch(monkeypatch, sweep)
    monkeypatch.setattr(sys, "argv", [
        "sweep", "--experiments", "e5", "--norms", "rms_norm", "--seeds", "0",
        "--out-dir", str(tmp_path), "--gpu", gpu,
    ])
    with pytest.raises(_Stop):
        sweep.main()
    assert seen[0][1][VAR] == expected
    assert seen[0][1]["MPLBACKEND"] == "Agg"


def test_the_rms_sweep_refuses_an_index_outside_the_visible_list(
        monkeypatch, tmp_path):
    from train.rms_variants_train import sweep

    monkeypatch.setenv(VAR, "1")
    seen = _capture_launch(monkeypatch, sweep)
    monkeypatch.setattr(sys, "argv", [
        "sweep", "--experiments", "e5", "--norms", "rms_norm", "--seeds", "0",
        "--out-dir", str(tmp_path), "--gpu", "1",
    ])
    with pytest.raises(ValueError, match=VAR):
        sweep.main()
    assert seen == [], "a cell launched before the selection was checked"


@pytest.mark.parametrize("visible, gpu, expected", [
    ("1", "0", "1"),
    ("1,0", "1", "0"),
    (None, "0", "0"),
])
def test_the_embeddings_sweep_child_gets_the_resolved_device(
        monkeypatch, tmp_path, visible, gpu, expected):
    from train.embeddings_experimental import sweep

    _set_visible(monkeypatch, visible)
    # A parent backend is replaced, not inherited (the old text pin's intent).
    monkeypatch.setenv("MPLBACKEND", "TkAgg")
    seen = _capture_launch(monkeypatch, sweep)
    with pytest.raises(_Stop):
        sweep.main([
            "--models", "ascii_bert", "--variants", "tiny", "--pooling", "mean",
            "--seeds", "0", "--sweep-root", str(tmp_path), "--gpu", gpu,
        ])
    assert seen[0][1][VAR] == expected
    assert seen[0][1]["MPLBACKEND"] == "Agg"


# ---------------------------------------------------------------------------
# Two-arm comparison drivers: CPU-only themselves, children on the resolved GPU
# ---------------------------------------------------------------------------

def _child_setup_gpu_accepts(cmd, child_visible):
    """The child's own ``setup_gpu(--gpu)`` must accept what it is handed."""
    if "--gpu" not in cmd:
        return
    gpu_module.visible_device_for(int(cmd[cmd.index("--gpu") + 1]), child_visible)


@pytest.mark.parametrize("visible, argv, expected", [
    (None, [], "1"),                 # unset, no flag: the driver's GPU 1 default
    ("1", [], "1"),                  # an export is respected
    ("1", ["--gpu", "0"], "1"),
    ("0,1", ["--gpu", "1"], "1"),
])
def test_the_vae_comparison_child_gets_the_resolved_device(
        monkeypatch, tmp_path, visible, argv, expected):
    _set_visible(monkeypatch, visible)
    monkeypatch.chdir(tmp_path)  # the driver creates ./results
    driver = importlib.import_module("train.vae.run_sampler_comparison")
    seen = _capture_launch(monkeypatch, driver)
    args = driver.build_argument_parser().parse_args(argv)
    with pytest.raises(_Stop):
        # main() hands over what it inherited (proven below, D-118).
        driver.run_comparison(args, visible)
    cmd, env = seen[0]
    assert env[VAR] == expected
    _child_setup_gpu_accepts(cmd, env[VAR])


@pytest.mark.parametrize("visible, gpu, expected", [
    ("1", "0", "1"),
    ("1,0", "1", "0"),
    (None, "0", "0"),
    ("1", "-1", ""),                 # the driver's documented CPU mode
])
def test_the_convnext_comparison_child_gets_the_resolved_device(
        monkeypatch, tmp_path, visible, gpu, expected):
    _set_visible(monkeypatch, visible)
    driver = importlib.import_module("train.convnext.run_stochastic_comparison")
    seen = _capture_launch(monkeypatch, driver)
    args = driver.build_argument_parser().parse_args(
        ["--gpu", gpu, "--output-dir", str(tmp_path)])
    with pytest.raises(_Stop):
        driver.run_comparison(args, visible)
    cmd, env = seen[0]
    assert env[VAR] == expected
    _child_setup_gpu_accepts(cmd, env[VAR])


@pytest.mark.parametrize("visible, expected", [
    (None, "0"),     # unchanged default: physical GPU 0
    ("1", "1"),
    ("", ""),        # GPUs hidden on purpose stay hidden
])
def test_the_logic_multiseed_sweep_child_device(monkeypatch, tmp_path, visible, expected):
    from train.logic import multiseed_sweep

    _set_visible(monkeypatch, visible)
    seen = []

    def fake_run_one(spec, timeout_s, env):
        seen.append(dict(env))
        raise _Stop

    monkeypatch.setattr(multiseed_sweep, "run_one", fake_run_one)
    monkeypatch.setattr(sys, "argv", [
        "multiseed_sweep", "--seeds", "0", "--experiments", "e1",
        "--out-dir", str(tmp_path),
    ])
    with pytest.raises(_Stop):
        multiseed_sweep.main()
    assert seen[0][VAR] == expected


# ---------------------------------------------------------------------------
# Evaluation scripts: their own process
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("visible, argv, expected", [
    ("1", ["--gpu", "0"], "1"),
    (None, ["--gpu", "1"], "1"),
    ("1", [], "1"),                  # no flag leaves an export alone
    ("0,1", [], "0,1"),
])
def test_eval_thera_selects_by_the_rule(monkeypatch, visible, argv, expected):
    from train.thera import eval_thera

    _set_visible(monkeypatch, visible)

    def stop(*_args, **_kwargs):
        raise _Stop

    monkeypatch.setattr(eval_thera.keras.models, "load_model", stop)
    monkeypatch.setattr(sys, "argv", [
        "eval_thera", "--checkpoint", "x.keras", "--data-dir", "d", *argv])
    with pytest.raises(_Stop):
        eval_thera.main()
    assert os.environ.get(VAR) == expected


def test_eval_thera_refuses_an_index_outside_the_visible_list(monkeypatch):
    from train.thera import eval_thera

    monkeypatch.setenv(VAR, "1")
    monkeypatch.setattr(sys, "argv", [
        "eval_thera", "--checkpoint", "x.keras", "--data-dir", "d", "--gpu", "1"])
    with pytest.raises(ValueError, match=VAR):
        eval_thera.main()
    assert os.environ[VAR] == "1"


_PIXEL = "train.bfunet.eval_per_pixel_uncertainty"


@pytest.mark.parametrize("visible, flag, expected", [
    ("0", [], "0"),                  # an export is respected (was overwritten with "1")
    ("0,1", [], "0,1"),
    (None, [], "1"),                 # unset, no flag: the script's GPU 1 default
    ("1", ["--gpu", "0"], "1"),      # the flag counts within the export
    (None, ["--gpu", "0"], "0"),
])
def test_eval_per_pixel_uncertainty_selects_by_the_rule(
        monkeypatch, visible, flag, expected):
    _set_visible(monkeypatch, visible)
    argv = ["eval_per_pixel_uncertainty", "--checkpoint", "x.keras", *flag]
    module = _reload(monkeypatch, _PIXEL, argv)

    def stop(_args):
        raise _Stop

    monkeypatch.setattr(module, "run_evaluation", stop)
    with pytest.raises(_Stop):
        module.main()
    assert os.environ.get(VAR) == expected


# ---------------------------------------------------------------------------
# Step 14 Part 0 (D-118): launchers that forwarded --gpu to a child, and
# default indexes above 0
# ---------------------------------------------------------------------------
#
# Measured at 7bd01f8c5, each in a real process under CUDA_VISIBLE_DEVICES=1:
# - ``mothnet/multiseed_sweep.py`` forwarded its own ``--gpu`` (default 1) to
#   every ``train_mothnet`` child, whose ``setup_gpu(1)`` raised: the sweep's
#   default invocation could not train a single seed.
# - ``rms_variants_train/run_n15_chunk1.sh`` runs the sweep as
#   ``CUDA_VISIBLE_DEVICES=1 ... --gpu 1``: ValueError before the first cell.
# - ``bfunet/run_gabor_depthwise_depth_sweep.sh`` forwarded ``--gpu "$GPU"``
#   (default 1) to ``train_convunext_denoiser``: ValueError at ``setup_gpu``.
# - four scripts (``nam/eval_dfsa.py``, ``nam/test_extreme.py``,
#   ``cliffordnet/eval_clip_retrieval.py``, ``cliffordnet/filter_cc3m_clipscore.py``)
#   default ``--gpu`` to 1, which no one-entry export can satisfy.

import ast  # noqa: E402
import re  # noqa: E402
import runpy  # noqa: E402
import shlex  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

_SRC = Path(gpu_module.__file__).resolve().parents[2]
_TRAIN = _SRC / "train"


def _mothnet_child(monkeypatch, tmp_path, visible, argv):
    """Run the sweep's ``main`` to its first child; return (cmd, env)."""
    from train.mothnet import multiseed_sweep

    _set_visible(monkeypatch, visible)
    seen = []
    real_run_one = multiseed_sweep.run_one

    def first_child_only(spec, timeout_s, env):
        def fake_run(cmd, *args, **kwargs):
            seen.append((list(cmd), dict(kwargs["env"])))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(multiseed_sweep.subprocess, "run", fake_run)
        real_run_one(spec, timeout_s, env)
        raise _Stop  # before any result or summary is written

    monkeypatch.setattr(multiseed_sweep, "run_one", first_child_only)
    with pytest.raises(_Stop):
        multiseed_sweep.main(["--seeds", "1", *argv])
    return seen[0]


@pytest.mark.parametrize("visible, argv, expected", [
    (None, [], "1"),                 # unset, no flag: the sweep's GPU 1 default
    ("1", [], "1"),                  # an export is respected (the child raised)
    ("1", ["--gpu", "0"], "1"),
    ("0,1", ["--gpu", "1"], "1"),
])
def test_the_mothnet_sweep_child_gets_the_resolved_device(
        monkeypatch, tmp_path, visible, argv, expected):
    from train.mothnet import train_mothnet

    cmd, env = _mothnet_child(monkeypatch, tmp_path, visible, argv)
    assert env[VAR] == expected
    # The child selects with its own --gpu (its default when none is passed:
    # None since step 11.1, so the exported list is kept, D-234); that
    # selection must be the device the sweep resolved, not a refusal.
    child_args = train_mothnet.parse_arguments(cmd[cmd.index("-m") + 2:])
    selection = gpu_module.gpu_or_default(child_args.gpu, env[VAR], 0)
    child_device = (env[VAR] if selection is None
                    else gpu_module.visible_device_for(selection, env[VAR]))
    assert child_device == expected


def test_the_mothnet_sweep_refuses_an_index_outside_the_visible_list(
        monkeypatch, tmp_path):
    from train.mothnet import multiseed_sweep

    monkeypatch.setenv(VAR, "1")

    def launched(*_args):
        raise _Stop  # a seed started before the selection was checked

    monkeypatch.setattr(multiseed_sweep, "run_one", launched)
    with pytest.raises(ValueError, match=VAR):
        multiseed_sweep.main(["--seeds", "1", "--gpu", "1"])


# --- a --gpu default is None; any index raises under some export ------------
#
# Step 14 (D-118) found defaults above 0, which no one-entry export holds. Step
# 11.1 (D-234) extends the census to "not None": a default of 0 raises under an
# exported empty CUDA_VISIBLE_DEVICES (CPU on purpose) and overrides a
# multi-entry export. At 6cb295f1d eight scripts still had one: a literal 0
# in the three launchers, mothnet and clevr_hans, a config field in three.

def _gpu_defaults_not_none(source):
    """Line numbers of ``add_argument("--gpu", ..., default=X)`` with X not the
    literal ``None`` (a constant, a name or a config attribute alike: a census
    cannot evaluate the last two, and the repo default is ``None``)."""
    lines = []
    with warnings.catch_warnings():  # invalid escapes in scanned sources
        warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
        tree = ast.parse(source)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--gpu"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and not (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is None):
                lines.append(node.lineno)
    return sorted(lines)


def test_the_default_census_sees_every_default_but_none():
    """Positive control: an index, 0, a config field and a name are found;
    ``None``, no default and ``--gpus`` are not."""
    source = (
        'p.add_argument("--gpu", type=int, default=1)\n'
        'p.add_argument("--gpu", type=int, default=0)\n'
        'p.add_argument("--gpu", type=int, default=None)\n'
        'p.add_argument(\n    "--gpu", type=int, default=2, help="x")\n'
        'p.add_argument("--gpus", type=int, default=1)\n'
        'p.add_argument("--gpu", type=int, default=Config.gpu)\n'
        'p.add_argument("--gpu", type=int, default=DEFAULT_GPU)\n'
        'p.add_argument("--gpu", type=int)\n'
    )
    assert _gpu_defaults_not_none(source) == [1, 2, 4, 7, 8]


def test_every_gpu_option_defaults_to_none():
    offenders = [
        f"{path.relative_to(_SRC)}:{line}"
        for path in sorted(_SRC.rglob("*.py"))
        for line in _gpu_defaults_not_none(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "a --gpu default other than None selects an index under every export "
        "(0 raises under an empty CUDA_VISIBLE_DEVICES); default to None and "
        "pick a preferred GPU with gpu_or_default / child_visible_devices:\n"
        + "\n".join(offenders))


@pytest.mark.parametrize("module_name, argv", [
    ("train.logic.train_e5_clevr_hans", ["train_e5_clevr_hans"]),
    ("train.mothnet.train_mothnet", ["train_mothnet"]),
])
@pytest.mark.parametrize("visible, expected", [
    ("", None),      # GPUs hidden on purpose: setup_gpu(0) raised ValueError
    ("0,1", None),   # an export is used as is
    (None, 0),       # nothing exported: physical GPU 0, the old default
])
def test_a_trainer_without_gpu_flag_keeps_the_export(
        monkeypatch, module_name, argv, visible, expected):
    _set_visible(monkeypatch, visible)
    module = importlib.import_module(module_name)
    seen = []

    def stop(gpu_id=None):
        seen.append(gpu_id)
        raise _Stop

    monkeypatch.setattr(module, "setup_gpu", stop)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_Stop):
        module.main()
    assert seen == [expected]


def test_thera_without_gpu_flag_keeps_the_export(monkeypatch):
    import train.thera.train_thera as thera
    assert thera.TheraConfig.gpu is None
    _set_visible(monkeypatch, "")
    seen = []

    def stop(gpu_id=None):
        seen.append(gpu_id)
        raise _Stop

    monkeypatch.setattr(thera, "setup_gpu", stop)
    with pytest.raises(_Stop):
        thera.train(thera.TheraConfig())
    assert seen == [None]


_NO_FLAG = [("", ""), ("0,1", "0,1"), (None, "0"), ("1", "1")]


@pytest.mark.parametrize("visible, expected", _NO_FLAG)
def test_the_rms_sweep_without_gpu_flag_keeps_the_export(
        monkeypatch, tmp_path, visible, expected):
    from train.rms_variants_train import sweep

    _set_visible(monkeypatch, visible)
    seen = _capture_launch(monkeypatch, sweep)
    monkeypatch.setattr(sys, "argv", [
        "sweep", "--experiments", "e5", "--norms", "rms_norm", "--seeds", "0",
        "--out-dir", str(tmp_path)])
    with pytest.raises(_Stop):
        sweep.main()
    assert seen[0][1][VAR] == expected


@pytest.mark.parametrize("visible, expected", _NO_FLAG)
def test_the_embeddings_sweep_without_gpu_flag_keeps_the_export(
        monkeypatch, tmp_path, visible, expected):
    from train.embeddings_experimental import sweep

    _set_visible(monkeypatch, visible)
    seen = _capture_launch(monkeypatch, sweep)
    with pytest.raises(_Stop):
        sweep.main([
            "--models", "ascii_bert", "--variants", "tiny", "--pooling", "mean",
            "--seeds", "0", "--sweep-root", str(tmp_path)])
    assert seen[0][1][VAR] == expected


@pytest.mark.parametrize("visible, expected", _NO_FLAG)
def test_the_convnext_driver_without_gpu_flag_keeps_the_export(
        monkeypatch, tmp_path, visible, expected):
    _set_visible(monkeypatch, visible)
    driver = importlib.import_module("train.convnext.run_stochastic_comparison")
    seen = _capture_launch(monkeypatch, driver)
    args = driver.build_argument_parser().parse_args(
        ["--output-dir", str(tmp_path)])
    with pytest.raises(_Stop):
        driver.run_comparison(args, visible)
    assert seen[0][1][VAR] == expected


@pytest.mark.parametrize("script, argv", [
    ("nam/eval_dfsa.py", ["eval_dfsa"]),
    ("nam/test_extreme.py", ["test_extreme", "--checkpoint", "x.h5"]),
    ("cliffordnet/eval_clip_retrieval.py",
     ["eval_clip_retrieval", "--checkpoint", "x.keras"]),
    ("cliffordnet/filter_cc3m_clipscore.py",
     ["filter_cc3m_clipscore", "--checkpoint", "x.keras",
      "--out-manifest", "x.jsonl"]),
])
@pytest.mark.parametrize("visible, flag, expected", [
    ("1", [], "1"),                  # an export is respected (was a refusal)
    (None, [], "1"),                 # unset, no flag: the script's GPU 1 default
    ("0,1", ["--gpu", "1"], "1"),
])
def test_a_script_with_a_preferred_gpu_selects_by_the_rule(
        monkeypatch, script, argv, visible, flag, expected):
    _set_visible(monkeypatch, visible)
    real_setup_gpu = gpu_module.setup_gpu

    def stop(gpu_id=None):
        real_setup_gpu(gpu_id)
        raise _Stop  # where the script would load a checkpoint

    # Two of the scripts parse in an ``if __name__ == "__main__"`` block, so
    # each runs as ``__main__`` from its file (one is named ``test_*.py``,
    # which pytest's import hook would claim); every ``setup_gpu`` import sees
    # the stop.
    monkeypatch.setattr(gpu_module, "setup_gpu", stop)
    monkeypatch.setattr(importlib.import_module("train.common"), "setup_gpu", stop)
    monkeypatch.setattr(sys, "argv", [*argv, *flag])
    with pytest.raises(_Stop):
        runpy.run_path(str(_TRAIN / script), run_name="__main__")
    assert os.environ.get(VAR) == expected


# --- shell launchers ----------------------------------------------------------

_EXPORT = re.compile(r"^\s*export\s+" + VAR + r"=(\S*)")
_PREFIX = re.compile(r"(?:^|[;&|]\s*)" + VAR + r"=(\S*)\s+\S")


def _shell_gpu_selections(text):
    """``(line, token, visible)`` for every ``--gpu`` in a shell script.

    ``visible`` is the ``CUDA_VISIBLE_DEVICES`` the script itself gives that
    command (a command prefix, else its latest ``export``), or ``None`` when
    the caller's environment decides. Continuation lines are joined first.
    """
    selections, exported = [], None
    logical, start = "", None
    for number, raw in enumerate(text.splitlines(), start=1):
        if start is None:
            start = number
        if raw.rstrip().endswith("\\"):
            logical += raw.rstrip()[:-1] + " "
            continue
        logical += raw
        line, logical, first = logical, "", start
        start = None
        if line.lstrip().startswith("#"):
            continue
        match = _EXPORT.match(line)
        if match:
            exported = shlex.split(match.group(1))[0] if match.group(1) else ""
            continue
        prefix = _PREFIX.search(line)
        visible = (shlex.split(prefix.group(1))[0] if prefix and prefix.group(1)
                   else "" if prefix else exported)
        for token in re.findall(r"--gpu[= ]+(\S+)", line):
            selections.append((first, token.strip("\"'"), visible))
    return selections


def _shell_offenders(path, text):
    offenders = []
    for line, token, visible in _shell_gpu_selections(text):
        where = f"{path}:{line}"
        if not re.fullmatch(r"-?\d+", token):
            offenders.append(f"{where}: --gpu {token} is not a literal index; "
                             "resolve the device once and export the list")
            continue
        index = int(token)
        if visible is None:
            if index != 0:
                offenders.append(f"{where}: --gpu {index} with no export of its "
                                 "own does not fit CUDA_VISIBLE_DEVICES=1")
            continue
        try:
            gpu_module.visible_device_for(index, visible)
        except ValueError as error:
            offenders.append(f"{where}: {error}")
    return offenders


def test_the_shell_census_reads_exports_prefixes_and_variables():
    """Positive control: every refused form is refused, every fitting one kept."""
    text = (
        "export CUDA_VISIBLE_DEVICES=0\n"
        'run --gpu 0 \\\n  --x 1\n'                              # 2: fits "0"
        "run --gpu 1\n"                                         # 4: outside "0"
        "CUDA_VISIBLE_DEVICES=1 \"$PY\" -m m \\\n  --gpu 1\n"   # 5: outside "1"
        "CUDA_VISIBLE_DEVICES=0,1 run --gpu 1\n"                # 7: fits
        'run --gpu "$GPU"\n'                                     # 8: a variable
        "# run --gpu 7\n"                                        # a comment
    )
    assert [o.split(":")[1] for o in _shell_offenders("s", text)] == [
        "4", "5", "8"]
    unexported = "run --gpu 0\nrun --gpu 1\n"
    assert [o.split(":")[1] for o in _shell_offenders("s", unexported)] == ["2"]


def test_every_shell_launcher_gpu_index_fits_its_own_export():
    scripts = sorted(_TRAIN.rglob("*.sh"))
    assert scripts, "no shell launcher found; the census would pass vacuously"
    offenders = [
        offender for path in scripts
        for offender in _shell_offenders(
            path.relative_to(_SRC), path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], "\n".join(offenders)


_BFUNET_SWEEP = _TRAIN / "bfunet" / "run_gabor_depthwise_depth_sweep.sh"


def _bfunet_selection(visible, argument):
    """Run the sweep script's own selection function in a fresh bash."""
    text = _BFUNET_SWEEP.read_text(encoding="utf-8")
    match = re.search(r"^select_visible_devices\(\) \{$.*?^\}$", text,
                      flags=re.MULTILINE | re.DOTALL)
    assert match, "the sweep script has no select_visible_devices function"
    env = {key: value for key, value in os.environ.items() if key != VAR}
    if visible is not None:
        env[VAR] = visible
    env["PY"] = sys.executable
    return subprocess.run(
        ["bash", "-c", match.group(0) + '\nselect_visible_devices "$1"',
         "bash", argument],
        env=env, cwd=str(_SRC), capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("visible, argument, expected", [
    (None, "", "1"),         # unset, no argument: the script's GPU 1 default
    ("1", "", "1"),          # an export is respected (the child raised)
    ("0,1", "1", "1"),       # an argument counts within the export
])
def test_the_bfunet_depth_sweep_resolves_its_device_once(visible, argument, expected):
    done = _bfunet_selection(visible, argument)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == expected


def test_the_bfunet_depth_sweep_refuses_an_index_outside_the_export():
    done = _bfunet_selection("1", "1")
    assert done.returncode != 0
    assert VAR in done.stderr


# --- the comparison drivers hide GPUs in main, not at import ------------------

_GPU_COUNT = ("import tensorflow as tf\n"
              "print('GPUS', len(tf.config.list_physical_devices('GPU')))\n")


def _gpus_seen_after(prelude):
    env = dict(os.environ, MPLBACKEND="Agg")
    if env.get(VAR) == "":
        # A blank inherited from this pytest process (the very leak under test,
        # when an importing test module was collected first) would make the
        # control see no GPU too, and the comparison pass vacuously.
        del env[VAR]
    done = subprocess.run([sys.executable, "-c", prelude + _GPU_COUNT],
                          env=env, cwd=str(_SRC.parent), capture_output=True,
                          text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return [line for line in done.stdout.splitlines() if line.startswith("GPUS")]


def test_importing_the_comparison_drivers_leaves_the_gpus_visible():
    """Measured at 7bd01f8c5: importing either driver hid every GPU from the
    importing process for good, so a pytest run that collected
    ``test_run_stochastic_comparison.py`` ran every later GPU test on CPU."""
    control = _gpus_seen_after("")
    if control == ["GPUS 0"]:
        pytest.skip("no GPU visible: the probe cannot see one being hidden")
    after = _gpus_seen_after(
        "import train.convnext.run_stochastic_comparison\n"
        "import train.vae.run_sampler_comparison\n"
        "import pytest\n"
        # The convnext driver's own test module, then a device listing in the
        # same process: what a later test in that pytest run would see.
        "pytest.main(['-q', '-p', 'no:cacheprovider', "
        "'tests/test_train/test_convnext/test_run_stochastic_comparison.py'])\n")
    assert after == control


@pytest.mark.parametrize("module_name, visible", [
    ("train.convnext.run_stochastic_comparison", "1"),
    ("train.vae.run_sampler_comparison", "0,1"),
    ("train.vae.run_sampler_comparison", None),
])
def test_a_driver_main_hides_the_gpus_and_keeps_what_it_inherited(
        monkeypatch, module_name, visible):
    _set_visible(monkeypatch, visible)
    driver = importlib.import_module(module_name)
    assert os.environ.get(VAR) == visible, "importing the driver wrote the variable"
    seen = []

    def fake_run_comparison(args, inherited_visible_devices):
        seen.append((inherited_visible_devices, os.environ.get(VAR)))

    monkeypatch.setattr(driver, "run_comparison", fake_run_comparison)
    monkeypatch.setattr(sys, "argv", ["prog"])
    driver.main()
    assert seen == [(visible, "")]


# --- step 11.1 (D-235): the time-series exports hide GPUs in main, not at import

_EXPORTS = [f"train.time_series.{name}.export" for name in (
    "ets", "nbeats", "tirex", "xlstm", "adaptive_ema", "mdn", "prism", "deepar")]


def _module_level_hide_gpus(source):
    """Line numbers of ``hide_gpus()`` calls that run at import: anywhere in
    the module body outside a ``def``/``class`` and outside an
    ``if __name__ == "__main__":`` block."""
    with warnings.catch_warnings():  # invalid escapes in scanned sources
        warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
        tree = ast.parse(source)

    def is_main_guard(node):
        return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__")

    lines, stack = [], list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)) or is_main_guard(node):
            continue
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "hide_gpus":
                lines.append(node.lineno)
        stack.extend(ast.iter_child_nodes(node))
    return sorted(lines)


def test_the_import_time_census_sees_a_module_level_call_only():
    """Positive control: the three import-time forms are found; calls in a
    function, a method and the ``__main__`` block are not."""
    source = (
        "from train.common.gpu import hide_gpus\n"
        "hide_gpus()\n"
        "if True:\n"
        "    gpu.hide_gpus()\n"
        "x = hide_gpus()\n"
        "def main():\n"
        "    hide_gpus()\n"
        "class A:\n"
        "    def f(self):\n"
        "        hide_gpus()\n"
        "if __name__ == '__main__':\n"
        "    hide_gpus()\n"
    )
    assert _module_level_hide_gpus(source) == [2, 4, 5]


def test_no_module_hides_the_gpus_at_import():
    offenders = [
        f"{path.relative_to(_SRC)}:{line}"
        for path in sorted(_SRC.rglob("*.py"))
        for line in _module_level_hide_gpus(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "hide_gpus() at import hides every GPU from the importing process for "
        "good (D-118); call it first thing in main:\n" + "\n".join(offenders))


@pytest.mark.parametrize("module_name", _EXPORTS)
def test_an_export_main_hides_the_gpus_before_anything_else(
        monkeypatch, tmp_path, module_name):
    """``main`` pins the process to the CPU before it reads its model: a
    missing model file exits 1 with the variable already empty."""
    _set_visible(monkeypatch, "1")
    module = importlib.import_module(module_name)
    assert os.environ.get(VAR) == "1", "importing the export wrote the variable"
    monkeypatch.setattr(sys, "argv", [
        "export", "--model_path", str(tmp_path / "missing.keras")])
    with pytest.raises(SystemExit) as exited:
        module.main()
    assert exited.value.code == 1
    assert os.environ.get(VAR) == ""


def test_importing_the_exports_leaves_the_gpus_visible():
    """Fresh process, real devices: the importer keeps its GPUs (at 6cb295f1d
    each export hid them at import, D-118's driver trap in eight programs)."""
    control = _gpus_seen_after("")
    if control == ["GPUS 0"]:
        pytest.skip("no GPU visible: the probe cannot see one being hidden")
    after = _gpus_seen_after("".join(f"import {name}\n" for name in _EXPORTS))
    assert after == control
