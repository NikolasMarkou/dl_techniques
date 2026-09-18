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
import warnings
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


def test_predict_in_batches_matches_unbatched_on_a_non_divisible_fixture() -> None:
    """`_predict_in_batches` (plan-2026-09-18T080513-debe8b11 Step 2) must agree
    with an unbatched `model.extract_features(x)` call to float32 tolerance.

    `batch_size=7` deliberately does NOT divide the fixture's 23 rows, so the
    final chunk (`x[21:23]`, 2 rows) is genuinely partial — exercising the
    exact edge case `_predict_in_batches`'s `range(0, len(x), batch_size)` +
    plain-slice chunking must handle without raising or dropping rows.

    Builds a real, small `MothNet` via `train_mothnet.build_model` (the same
    construction path — parsed args + `build_model` — every other test in this
    module already drives the model through), rather than instantiating
    `MothNet` directly: this file has no existing bare-`MothNet` construction
    pattern to match, and going through `build_model` exercises the trainer's
    own seeding/build sequence instead of inventing a second one.
    """
    args = train_mothnet.parse_arguments([
        "--mb-units", "200",
        "--al-units", "64",
    ])
    x = np.random.default_rng(0).random((23, 64)).astype("float32")
    model = train_mothnet.build_model(args, input_dim=x.shape[1])

    batched = train_mothnet._predict_in_batches(model, x, batch_size=7)
    unbatched = keras.ops.convert_to_numpy(model.extract_features(x))

    assert batched.shape == unbatched.shape == (23, 10)
    np.testing.assert_allclose(batched, unbatched, atol=1e-5, rtol=1e-5)


def _one_hot(class_indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Local one-hot helper — avoids pulling in `keras.utils.to_categorical` for
    a 2-line need in this test module."""
    y = np.zeros((len(class_indices), num_classes), dtype=np.float32)
    y[np.arange(len(class_indices)), class_indices] = 1.0
    return y


def test_render_mb_sparsity_runs_with_all_classes_represented(tmp_path: Path) -> None:
    """`render_mb_sparsity` (plan-2026-09-18T080513-debe8b11 Step 6) must run to
    completion and produce a non-empty PNG on a fixture where every one of the
    model's 10 classes has at least one sample.

    Uses SYNTHETIC labels constructed with an explicit, deterministic
    class assignment (3 samples per class, all 10 classes) rather than a real
    MNIST subsample — sidestepping the question of what class coverage an
    actual MNIST subsample happens to have at a given sample count, since the
    zero-sample-class edge case (the thing that coverage question actually
    matters for) is covered explicitly and deterministically by the next test.
    """
    args = train_mothnet.parse_arguments(["--mb-units", "200", "--al-units", "64"])
    model = train_mothnet.build_model(args, input_dim=64)

    rng = np.random.default_rng(0)
    class_indices = np.repeat(np.arange(10), 3)  # 3 samples/class, all 10 present
    x_sample = rng.random((len(class_indices), 64)).astype("float32")
    y_sample = _one_hot(class_indices, num_classes=10)

    out_path = tmp_path / "epoch_001_mb_sparsity.png"
    train_mothnet.render_mb_sparsity(model, x_sample, y_sample, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_mb_sparsity_handles_a_class_with_zero_samples(tmp_path: Path) -> None:
    """A class entirely absent from `x_sample`/`y_sample` (plausible at a small
    `--num-val-samples`) must not raise and must still produce a well-formed
    image — the all-zero-row/`0.0`-sparsity degrade-gracefully path plan.md's
    Edge Cases section requires.

    Runs with `warnings.simplefilter("error")` so that a regression which skips
    the zero-sample-class guard (computing `.mean(axis=0)` over an EMPTY slice)
    fails LOUDLY here as a raised `RuntimeWarning: Mean of empty slice`, rather
    than silently degrading to a NaN row that still happens to produce a
    non-empty PNG (mere file-existence would not catch that regression — see
    this plan's RED-then-GREEN proof in the executor's report).
    """
    args = train_mothnet.parse_arguments(["--mb-units", "200", "--al-units", "64"])
    model = train_mothnet.build_model(args, input_dim=64)

    rng = np.random.default_rng(1)
    # Classes 0-8 present, class 9 has ZERO rows.
    class_indices = np.repeat(np.arange(9), 2)
    x_sample = rng.random((len(class_indices), 64)).astype("float32")
    y_sample = _one_hot(class_indices, num_classes=10)
    assert not np.any(np.argmax(y_sample, axis=-1) == 9), "test setup: class 9 must be absent"

    out_path = tmp_path / "epoch_002_mb_sparsity.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train_mothnet.render_mb_sparsity(model, x_sample, y_sample, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_mb_sparsity_handles_mb_units_not_divisible_by_bin_count(
    tmp_path: Path,
) -> None:
    """`mb_units=37` does not evenly divide the 200-bin target — MEASURE that
    `np.array_split`-based binning handles this rather than trusting the
    reasoning that it must.
    """
    args = train_mothnet.parse_arguments(["--mb-units", "37", "--al-units", "16"])
    model = train_mothnet.build_model(args, input_dim=32)

    rng = np.random.default_rng(2)
    class_indices = np.repeat(np.arange(10), 2)
    x_sample = rng.random((len(class_indices), 32)).astype("float32")
    y_sample = _one_hot(class_indices, num_classes=10)

    out_path = tmp_path / "epoch_003_mb_sparsity.png"
    train_mothnet.render_mb_sparsity(model, x_sample, y_sample, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_mb_sparsity_no_longer_calls_plt_spy() -> None:
    """Mechanical regression guard for plan.md Success Criterion 6: the old
    binary presence/absence scatter must be gone, not just superseded.
    """
    import inspect

    source = inspect.getsource(train_mothnet.render_mb_sparsity)
    assert "spy" not in source
