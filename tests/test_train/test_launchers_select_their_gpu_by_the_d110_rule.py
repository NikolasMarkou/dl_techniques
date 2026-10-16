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
    monkeypatch.setattr(
        gpu_module.tf.config.experimental, "list_physical_devices",
        lambda *_args, **_kwargs: [],
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
    driver = _reload(monkeypatch, "train.vae.run_sampler_comparison", ["prog"])
    assert os.environ.get(VAR) == "", "the driver process itself is CPU-only"
    seen = _capture_launch(monkeypatch, driver)
    args = driver.build_argument_parser().parse_args(argv)
    with pytest.raises(_Stop):
        driver.run_comparison(args)
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
    driver = _reload(
        monkeypatch, "train.convnext.run_stochastic_comparison", ["prog"])
    assert os.environ.get(VAR) == "", "the driver process itself is CPU-only"
    seen = _capture_launch(monkeypatch, driver)
    args = driver.build_argument_parser().parse_args(
        ["--gpu", gpu, "--output-dir", str(tmp_path)])
    with pytest.raises(_Stop):
        driver.run_comparison(args)
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
