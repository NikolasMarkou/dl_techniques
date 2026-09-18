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


# The three NEW periodic-visualization stale-cleanup glob patterns Step 5
# introduced (`decisions.md` D-001), mirrored alongside the pre-existing
# `epoch_*_mb_sparsity.png` pattern that `main()` has cleaned up since before
# this plan (plan-2026-09-18T060057-c1cfc3d3 Step 5 / F-05).
_NEW_PERIODIC_GLOB_PATTERNS = (
    "epoch_*_al_mb_activations_distribution.png",
    "epoch_*_al_mb_activations_heatmap.png",
    "epoch_*_confusion_matrix.png",
)
_ALL_PERIODIC_GLOB_PATTERNS = _NEW_PERIODIC_GLOB_PATTERNS + ("epoch_*_mb_sparsity.png",)


@pytest.mark.integration
def test_periodic_visualization_files_land_at_the_expected_epoch_stamped_names(
    tmp_path: Path,
) -> None:
    """plan.md Step 7 / Success Criterion 5: a tiny real `main()` run with
    `--viz-freq 1` for 2 epochs must produce ALL FIVE periodic/per-epoch
    visualization files, non-empty, at the exact names `main()`'s own loop
    writes them under (confirmed by direct source read, not guessed):
    `training_dashboard.png` (per-epoch, overwritten in place, never
    epoch-stamped) plus the four `epoch_002_*` files from the FINAL
    (`--epochs 2`) epoch — `mb_sparsity`, `al_mb_activations_distribution`,
    `al_mb_activations_heatmap`, `confusion_matrix`.

    Steps 5/6's own executor reports already MEASURED this via a real,
    uncommitted `results/` probe run; this is the permanent, committed,
    automated version of that same proof.
    """
    run_dir = tmp_path / "mothnet_periodic_viz_run"
    argv = [
        "--mb-units", "200",
        "--al-units", "64",
        "--epochs", "2",
        "--num-train-samples", "100",
        "--num-val-samples", "50",
        "--batch-size", "32",
        "--viz-freq", "1",
        "--output-dir", str(tmp_path),
        "--experiment-name", "mothnet_periodic_viz_run",
    ]
    exit_code = train_mothnet.main(argv)
    assert exit_code == 0

    viz_dir = run_dir / "visualizations"
    expected_files = (
        viz_dir / "training_dashboard.png",
        viz_dir / "epoch_002_mb_sparsity.png",
        viz_dir / "epoch_002_al_mb_activations_distribution.png",
        viz_dir / "epoch_002_al_mb_activations_heatmap.png",
        viz_dir / "epoch_002_confusion_matrix.png",
    )
    for expected_file in expected_files:
        assert expected_file.exists(), f"missing periodic visualization file: {expected_file}"
        assert expected_file.stat().st_size > 0, f"empty periodic visualization file: {expected_file}"


@pytest.mark.integration
def test_stale_periodic_visualization_files_are_cleaned_up_on_a_rerun(
    tmp_path: Path,
) -> None:
    """A rerun into the SAME `--experiment-name` must leave exactly the NEW
    run's periodic-visualization files behind, not an accumulation of the
    PRIOR (here, longer) run's files at the same name — mirroring the shape
    of the pre-existing `epoch_*_mb_sparsity.png` stale-cleanup behavior
    (`main()`'s own comment: "must start visualizations/ clean of stale
    periodic PNGs from a PRIOR (possibly longer) run at that name"), but
    exercised here against the THREE NEW glob patterns Step 5 introduced,
    which had no permanent test of this rerun-cleanup behavior until now.

    Runs once at `--epochs 5` (5 periodic files per pattern expected), then
    reruns into the identical `--output-dir`/`--experiment-name` at
    `--epochs 2` (2 files per pattern expected) — proving the SECOND run's
    stale-cleanup glob loops actually deleted the first run's leftover files,
    not merely that a short run alone produces 2 files.
    """
    experiment_name = "mothnet_stale_cleanup_run"
    run_dir = tmp_path / experiment_name
    viz_dir = run_dir / "visualizations"
    base_argv = [
        "--mb-units", "200",
        "--al-units", "64",
        "--num-train-samples", "100",
        "--num-val-samples", "50",
        "--batch-size", "32",
        "--viz-freq", "1",
        "--output-dir", str(tmp_path),
        "--experiment-name", experiment_name,
    ]

    exit_code_first = train_mothnet.main(base_argv + ["--epochs", "5"])
    assert exit_code_first == 0
    for pattern in _ALL_PERIODIC_GLOB_PATTERNS:
        matches = sorted(p.name for p in viz_dir.glob(pattern))
        assert len(matches) == 5, f"{pattern}: expected 5 files after first run, found {matches}"

    exit_code_second = train_mothnet.main(base_argv + ["--epochs", "2"])
    assert exit_code_second == 0
    for pattern in _ALL_PERIODIC_GLOB_PATTERNS:
        matches = sorted(p.name for p in viz_dir.glob(pattern))
        assert len(matches) == 2, (
            f"{pattern}: expected exactly 2 files after rerun (stale files from "
            f"the first, longer run must be cleaned up), found {matches}"
        )


@pytest.mark.integration
def test_al_features_failure_is_independently_fail_soft_from_confusion_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    """Step 5's two NEW periodic render blocks (AL/MB activations, confusion
    matrix) each get their OWN `try/except Exception: logger.warning(...)`
    (plan.md invariant 1 / Step 5.2c) — a failure in ONE must not skip the
    OTHER. Step 5's own executor report proved this manually via a real,
    uncommitted run; this is the permanent, committed, automated version of
    that same proof, and the ONLY thing this step is required to add here
    (the plan's other Step 7 items were already added in Steps 2/4/6).

    `model.extract_al_features` is overridden with an INSTANCE attribute
    (installed via a wrapped `build_model`, mirroring
    `test_final_save_attempted_on_mid_loop_exception`'s own instance-override
    shape above) that raises unconditionally — forcing the AL/MB
    activation-visualization try/except to fail on every periodic epoch.

    Asserts: (a) `main()` still returns 0, not propagating the injected
    exception; (b) `final_model.keras` still exists; (c) the confusion-matrix
    PNG — an INDEPENDENT try/except block that does not call
    `extract_al_features` at all — still gets produced despite the injected
    failure in the OTHER block; and, to prove the injection genuinely fired
    rather than being silently routed around, (d) the AL/MB activation PNGs
    themselves are ABSENT.
    """
    _real_build_model = train_mothnet.build_model

    def _build_model_with_broken_al_features(args, input_dim):
        model = _real_build_model(args, input_dim)

        def _raise_extract_al_features(*_call_args, **_call_kwargs):
            raise RuntimeError("injected extract_al_features failure (test)")

        # Instance-level override — deliberately not a class-level monkeypatch,
        # so only THIS model instance's calls fail (matches
        # `test_final_save_attempted_on_mid_loop_exception`'s own
        # `model.train_hebbian = ...` shape above).
        model.extract_al_features = _raise_extract_al_features
        return model

    monkeypatch.setattr(train_mothnet, "build_model", _build_model_with_broken_al_features)

    run_dir = tmp_path / "mothnet_fail_soft_run"
    argv = [
        "--mb-units", "200",
        "--al-units", "64",
        "--epochs", "2",
        "--num-train-samples", "100",
        "--num-val-samples", "50",
        "--batch-size", "32",
        "--viz-freq", "1",
        "--output-dir", str(tmp_path),
        "--experiment-name", "mothnet_fail_soft_run",
    ]
    exit_code = train_mothnet.main(argv)
    assert exit_code == 0

    assert (run_dir / "final_model.keras").exists()

    viz_dir = run_dir / "visualizations"
    confusion_png = viz_dir / "epoch_002_confusion_matrix.png"
    assert confusion_png.exists(), (
        "confusion-matrix PNG missing — the injected extract_al_features "
        "failure incorrectly took down the INDEPENDENT confusion-matrix "
        "render block too"
    )
    assert confusion_png.stat().st_size > 0

    # The OTHER block's own outputs must be ABSENT — proving the injected
    # failure genuinely prevented ITS OWN render (the mock had a real effect),
    # rather than the confusion-matrix assertion above passing by coincidence.
    assert not (viz_dir / "epoch_002_al_mb_activations_distribution.png").exists()
    assert not (viz_dir / "epoch_002_al_mb_activations_heatmap.png").exists()
