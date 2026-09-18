"""Real end-to-end integration test for ``train.mothnet.train_mothnet``.

Covers:
- A genuine tiny ``train_hebbian()`` run driven through
  ``train_mothnet.main([...])`` IN-PROCESS (never a subprocess) — this is a
  fast smoke-scale proof that the hand-rolled Hebbian loop actually trains,
  distinct from the statistical sweep (plan.md Step 8/9, subprocess-per-seed,
  canonical scale).
- The full expected artifact set lands, redirected under ``tmp_path`` via
  ``--output-dir`` (`test_levjepa/test_train_levjepa.py`'s
  ``TestSmokeTrainingRun`` pattern) — never the real repo-root ``results/``.
- The CHEAP half of plan-level Success Criterion 5 (checkpoint fidelity): a
  reloaded ``best_model.keras`` reproduces the SAME prediction the in-process
  model itself produced (`test_lewm/test_train_lewm.py::
  test_end_to_end_fit_and_reload`'s ``max_diff`` shape). The EXPENSIVE
  canonical-scale half of Criterion 5 is plan.md Step 4, run separately.

``--epochs 1`` is used deliberately, not just as "the tiny end of the allowed
1-or-2 range": with exactly one epoch, ``best_val_accuracy`` starts at
``-1.0`` so the single epoch's ``val_accuracy`` is unconditionally a new
best, meaning ``best_model.keras`` is saved from the SAME (only) trained
state that ``final_model.keras`` captures moments later and that the
captured in-process ``model`` object holds when ``main()`` returns — no
ambiguity about which epoch's weights ``best_model.keras`` reflects.

This test does not mock any MothNet-internal (``train_hebbian``,
``MothNet.call``, ``HebbianReadoutLayer.hebbian_update``, ...) — the only
patched module attributes (``build_model``, ``load_mnist_data``) are
wrap-and-record shims that call straight through to the real function and
return its real result unmodified, purely so the test can observe the
model/data objects ``main()`` itself constructs. A real wiring break in the
hand-rolled loop is expected to surface here, not be swallowed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import train.mothnet.train_mothnet as train_mothnet  # noqa: E402


@pytest.mark.integration
def test_tiny_run_trains_and_checkpoint_round_trips(tmp_path: Path, monkeypatch) -> None:
    """A tiny real ``main()`` run trains via ``train_hebbian`` and round-trips."""
    captured: dict = {}

    _real_build_model = train_mothnet.build_model

    def _capturing_build_model(args, input_dim):
        model = _real_build_model(args, input_dim)
        captured["model"] = model
        # Snapshot BEFORE any train_hebbian call mutates readout_weights, so
        # the anti-stub check below has something genuine to diff against.
        captured["initial_weights"] = [w.copy() for w in model.get_weights()]
        return model

    _real_load_mnist_data = train_mothnet.load_mnist_data

    def _capturing_load_mnist_data(config):
        data = _real_load_mnist_data(config)
        captured["data"] = data
        return data

    monkeypatch.setattr(train_mothnet, "build_model", _capturing_build_model)
    monkeypatch.setattr(train_mothnet, "load_mnist_data", _capturing_load_mnist_data)

    run_dir = tmp_path / "mothnet_smoke_run"
    argv = [
        "--mb-units", "200",
        "--al-units", "64",
        "--epochs", "1",
        "--num-train-samples", "100",
        "--num-val-samples", "50",
        "--batch-size", "32",
        "--viz-freq", "1",
        "--output-dir", str(tmp_path),
        "--experiment-name", "mothnet_smoke_run",
    ]
    exit_code = train_mothnet.main(argv)
    assert exit_code == 0

    # --- (a) full artifact set lands -----------------------------------
    assert (run_dir / "config.json").exists()
    assert (run_dir / "best_model.keras").exists()
    assert (run_dir / "final_model.keras").exists()
    assert (run_dir / "training_log.csv").exists()
    assert (run_dir / "training_history.json").exists()
    assert (run_dir / "visualizations" / "training_dashboard.png").exists()

    # --- genuine training signal, not a stub ----------------------------
    model = captured["model"]
    final_weights = model.get_weights()
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(captured["initial_weights"], final_weights)
    ), "no weight array changed — train_hebbian did not genuinely mutate the model"

    history = json.loads((run_dir / "training_history.json").read_text())
    assert len(history["loss"]) == 1
    assert np.isfinite(history["loss"][0])
    assert 0.0 <= history["val_accuracy"][0] <= 1.0
    assert 0.0 <= history["train_accuracy"][0] <= 1.0

    # --- (b) cheap checkpoint-fidelity half of Criterion 5 --------------
    (_, _), (x_val, _) = captured["data"]
    x_probe = x_val[:8]

    y_in_process = keras.ops.convert_to_numpy(model.extract_features(x_probe))
    reloaded = keras.models.load_model(str(run_dir / "best_model.keras"))
    y_reloaded = keras.ops.convert_to_numpy(reloaded.extract_features(x_probe))

    max_diff = float(np.max(np.abs(y_in_process - y_reloaded)))
    assert max_diff < 1e-4, f"Reload round-trip diff too large: {max_diff:.2e}"


@pytest.mark.integration
def test_final_save_attempted_on_mid_loop_exception(tmp_path: Path, monkeypatch) -> None:
    """Invariant 8 (plan-2026-09-18T060057-c1cfc3d3 Step 7, ``decisions.md`` D-008):
    a forced mid-loop exception must still leave ``final_model.keras`` and
    ``training_history.json`` on disk — best-effort, attempted from the
    ``finally`` clause wrapped around the epoch loop in ``main()``.

    ``model.train_hebbian`` is overridden with an INSTANCE attribute (not a
    class-level monkeypatch) that delegates to the real bound method for the
    first call, then raises on the second — so epoch 1 genuinely trains (one
    real ``train_hebbian`` call, proving this isn't a no-op stub) before the
    forced failure aborts the loop on epoch 2 of a 3-epoch run. The forced
    exception is asserted to be the ONE that propagates out of ``main()``
    unchanged (not masked by a save failure — both real saves succeed here).
    """
    captured: dict = {}

    _real_build_model = train_mothnet.build_model

    def _build_model_with_forced_failure(args, input_dim):
        model = _real_build_model(args, input_dim)
        captured["model"] = model

        real_train_hebbian = model.train_hebbian
        call_count = {"n": 0}

        def _train_hebbian_then_raise(*call_args, **call_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("forced mid-loop failure (test)")
            return real_train_hebbian(*call_args, **call_kwargs)

        model.train_hebbian = _train_hebbian_then_raise
        return model

    monkeypatch.setattr(train_mothnet, "build_model", _build_model_with_forced_failure)

    run_dir = tmp_path / "mothnet_exception_run"
    argv = [
        "--mb-units", "200",
        "--al-units", "64",
        "--epochs", "3",
        "--num-train-samples", "100",
        "--num-val-samples", "50",
        "--batch-size", "32",
        "--viz-freq", "1",
        "--output-dir", str(tmp_path),
        "--experiment-name", "mothnet_exception_run",
    ]

    # --- the ORIGINAL exception is the one that propagates out of main() ---
    with pytest.raises(RuntimeError, match="forced mid-loop failure"):
        train_mothnet.main(argv)

    # --- best-effort final-save net still ran despite the exception --------
    assert (run_dir / "final_model.keras").exists(), (
        "final_model.keras missing after a mid-loop exception — invariant 8's "
        "finally-clause save net did not run"
    )
    assert (run_dir / "training_history.json").exists(), (
        "training_history.json missing after a mid-loop exception — invariant 8's "
        "finally-clause save net did not run"
    )

    # Only epoch 1 completed (real train_hebbian call #1) before the forced
    # raise on call #2 — the partial history is exactly what the finally
    # block had available to save, not fabricated.
    history = json.loads((run_dir / "training_history.json").read_text())
    assert len(history["loss"]) == 1
    assert np.isfinite(history["loss"][0])

    # final_model.keras still round-trips (it captured whatever state `model`
    # held at the moment of the forced failure — after 1 genuine train_hebbian
    # call, not an untrained/unbuilt model).
    reloaded = keras.models.load_model(str(run_dir / "final_model.keras"))
    assert reloaded.built
