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
import logging
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


def test_render_mb_sparsity_states_fixed_sparsity_once_in_title_and_plain_digit_yticks(
    tmp_path: Path, monkeypatch,
) -> None:
    """Step 6.1 completion-fix (`plan-2026-09-18T080513-debe8b11` D-009,
    adversarial-review finding 5, WARNING): the per-class sparsity annotation
    was ARCHITECTURALLY CONSTANT — every class read the same fixed top-k
    fraction, since `MushroomBodyLayer` enforces a fixed top-k regardless of
    class — so repeating it on all ten y-ticks was misleading (reads like
    broken output). The fixed value must now appear exactly ONCE, in the
    title, and each y-tick label must be a bare class digit with no repeated
    percentage.

    Captures the real ``Axes`` object ``render_mb_sparsity`` builds by
    wrapping ``matplotlib.pyplot.subplots`` (record-then-call-through, the
    same wrap-and-record shape this module's other tests use) rather than
    mocking rendering away — the real heatmap still gets drawn and saved;
    only the returned ``(fig, ax)`` pair is additionally stashed here so the
    test can inspect the title/tick text AFTER the function returns
    (``plt.close(fig)`` inside the function only detaches it from
    pyplot's global figure registry, it does not invalidate the Python
    object this test still holds a reference to).
    """
    args = train_mothnet.parse_arguments(["--mb-units", "200", "--al-units", "64"])
    model = train_mothnet.build_model(args, input_dim=64)

    rng = np.random.default_rng(3)
    class_indices = np.repeat(np.arange(10), 3)
    x_sample = rng.random((len(class_indices), 64)).astype("float32")
    y_sample = _one_hot(class_indices, num_classes=10)

    captured_axes: list = []
    _real_subplots = train_mothnet.plt.subplots

    def _capturing_subplots(*subplot_args, **subplot_kwargs):
        fig, ax = _real_subplots(*subplot_args, **subplot_kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(train_mothnet.plt, "subplots", _capturing_subplots)

    out_path = tmp_path / "epoch_004_mb_sparsity.png"
    train_mothnet.render_mb_sparsity(model, x_sample, y_sample, out_path)

    assert len(captured_axes) == 1
    ax = captured_axes[0]

    title = ax.get_title()
    expected_pct = f"{model.mb_sparsity * 100:.1f}%"
    assert title.count(expected_pct) == 1, (
        "expected the fixed sparsity value to appear exactly once in the "
        f"title, got title={title!r}"
    )

    ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
    assert ytick_labels == [str(c) for c in range(10)], (
        "y-tick labels must be plain class digits with no per-row "
        f"percentage annotation, got {ytick_labels}"
    )
    for label in ytick_labels:
        assert "%" not in label, f"y-tick label still carries a percentage: {label!r}"


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


@pytest.mark.integration
def test_heatmap_activation_call_receives_column_binned_mushroom_body_data(
    tmp_path: Path, monkeypatch
) -> None:
    """Step 5.1 completion-fix (`plan-2026-09-18T080513-debe8b11` D-008,
    adversarial-review finding 1, CRITICAL): the HEATMAP
    ``viz_manager.visualize(...)`` call must receive a ``mushroom_body``
    activations array with <=200 columns, not the raw ``(N, mb_units)``
    array — ``ActivationVisualization``'s heatmap branch (``data_nn.py``)
    does ``ax.imshow(activations[:100])`` with no column-binning of its own,
    so an unbinned ``mb_units=16000`` array renders as a solid black
    rectangle. The DISTRIBUTION call must still receive the UNBINNED array
    (histograms have no column-count legibility problem, so only the
    heatmap-specific call is binned).

    Wraps ``VisualizationManager.visualize`` (record-then-call-through, the
    same wrap-and-record shape this module's other tests use for
    ``build_model``/``load_mnist_data``) rather than mocking it away, so the
    real plugin still runs and a genuinely broken heatmap call would still
    surface as an exception here, not just as a shape mismatch.

    Run at ``--mb-units 16000`` (the shipped default) — not a small
    fixture value — since the CRITICAL this proof targets specifically
    concerns the default-scale render.
    """
    captured_calls: list = []
    _real_visualize = train_mothnet.VisualizationManager.visualize

    def _capturing_visualize(
        self, data, plugin_name=None, save=True, show=False, filename=None, **kwargs
    ):
        captured_calls.append((plugin_name, kwargs.get("plot_type"), data))
        return _real_visualize(
            self, data, plugin_name=plugin_name, save=save, show=show,
            filename=filename, **kwargs,
        )

    monkeypatch.setattr(
        train_mothnet.VisualizationManager, "visualize", _capturing_visualize
    )

    argv = [
        "--mb-units", "16000",
        "--al-units", "64",
        "--epochs", "1",
        "--num-train-samples", "100",
        "--num-val-samples", "50",
        "--batch-size", "32",
        "--viz-freq", "1",
        "--output-dir", str(tmp_path),
        "--experiment-name", "mothnet_heatmap_binning_check",
    ]
    exit_code = train_mothnet.main(argv)
    assert exit_code == 0

    heatmap_calls = [call for call in captured_calls if call[1] == "heatmap"]
    assert len(heatmap_calls) == 1, f"expected exactly 1 heatmap call, got {captured_calls}"
    _, _, heatmap_data = heatmap_calls[0]
    # Step 5.4 completion-fix (D-011, pass-2 WARNING 2): the mushroom_body
    # entry's dict KEY (not just its shape) must self-document the bin size,
    # since `ActivationVisualization` titles each panel by its `activations`
    # dict key and cannot otherwise disclose that these 200 columns are
    # `mb_units // 200`-unit bin means rather than raw neurons — a bare
    # `"mushroom_body"` key is a stale-doc trap, not merely a naming choice.
    mb_heatmap_keys = [
        key for key in heatmap_data.activations if key.startswith("mushroom_body")
    ]
    assert len(mb_heatmap_keys) == 1, (
        f"expected exactly 1 mushroom_body-prefixed key, got {list(heatmap_data.activations)}"
    )
    mb_heatmap_key = mb_heatmap_keys[0]
    assert mb_heatmap_key != "mushroom_body", (
        "the heatmap-only mushroom_body key must carry a bin-size annotation, "
        "not the bare layer name (misleadingly implies raw neurons)"
    )
    assert heatmap_data.activations[mb_heatmap_key].shape[1] <= 200, (
        "heatmap call's mushroom_body activations were not column-binned — "
        f"shape={heatmap_data.activations[mb_heatmap_key].shape}"
    )

    distribution_calls = [call for call in captured_calls if call[1] == "distribution"]
    assert len(distribution_calls) == 1, (
        f"expected exactly 1 distribution call, got {captured_calls}"
    )
    _, _, distribution_data = distribution_calls[0]
    assert distribution_data.activations["mushroom_body"].shape[1] == 16000, (
        "distribution call's mushroom_body activations must stay UNBINNED "
        f"(raw mb_units) — shape={distribution_data.activations['mushroom_body'].shape}"
    )


@pytest.mark.integration
def test_confusion_matrix_class_names_match_the_classes_actually_present(
    tmp_path: Path, monkeypatch,
) -> None:
    """Step 5.2 completion-fix (`plan-2026-09-18T080513-debe8b11` D-007,
    adversarial-review finding 2, WARNING): the confusion-matrix render must
    not silently mislabel its axes when a class is absent from BOTH
    ``y_true`` and ``y_pred`` in the val subsample.

    ``ConfusionMatrixVisualization`` calls
    ``sklearn.metrics.confusion_matrix(y_true, y_pred)`` with no ``labels=``
    argument — sklearn's own default there orders the matrix by the SORTED
    UNION of values appearing at least once in ``y_true`` or ``y_pred``. A
    hardcoded ``class_names=[str(i) for i in range(10)]`` therefore silently
    mismatches the matrix's actual size the moment any class is absent from
    both arrays. This test forces classes 3 and 7 out of BOTH sides:

    - ``load_mnist_data`` is wrapped (real call, then filtered) to drop every
      TRUE class-3/class-7 row from the val split, so ``y_true`` genuinely
      never contains 3 or 7.
    - ``_predict_in_batches`` is wrapped (real call, then masked) to set
      columns 3 and 7 to ``-inf`` before the caller's own ``argmax``, so
      ``y_pred`` genuinely never contains 3 or 7 either — this mirrors the
      review's own exact reproduction (classes absent from BOTH arrays, not
      just one).

    ``VisualizationManager.visualize`` is wrapped (record-then-call-through,
    the same shape ``test_heatmap_activation_call_receives_column_binned_
    mushroom_body_data`` above uses) to capture the ``ClassificationResults``
    object actually passed to the ``confusion_matrix`` plugin. The proof:
    independently recomputing ``sklearn.metrics.confusion_matrix(y_true,
    y_pred)`` (no ``labels=``) on that SAME data must yield a matrix whose
    row/column count equals ``len(classification_results.class_names)`` — a
    fixed, hardcoded-10 ``class_names`` would fail this under the forced
    8-class scenario, since the real matrix collapses to 8x8.
    """
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

    _real_load_mnist_data = train_mothnet.load_mnist_data

    def _load_data_without_classes_3_and_7_in_val(config):
        (x_train, y_train), (x_val, y_val) = _real_load_mnist_data(config)
        val_true_classes = np.argmax(y_val, axis=-1)
        keep_mask = ~np.isin(val_true_classes, [3, 7])
        return (x_train, y_train), (x_val[keep_mask], y_val[keep_mask])

    _real_predict_in_batches = train_mothnet._predict_in_batches

    def _predict_forcing_predictions_away_from_3_and_7(model, x, batch_size):
        logits = _real_predict_in_batches(model, x, batch_size).copy()
        logits[:, [3, 7]] = -np.inf
        return logits

    monkeypatch.setattr(
        train_mothnet, "load_mnist_data", _load_data_without_classes_3_and_7_in_val
    )
    monkeypatch.setattr(
        train_mothnet, "_predict_in_batches",
        _predict_forcing_predictions_away_from_3_and_7,
    )

    captured_calls: list = []
    _real_visualize = train_mothnet.VisualizationManager.visualize

    def _capturing_visualize(
        self, data, plugin_name=None, save=True, show=False, filename=None, **kwargs
    ):
        captured_calls.append((plugin_name, data))
        return _real_visualize(
            self, data, plugin_name=plugin_name, save=save, show=show,
            filename=filename, **kwargs,
        )

    monkeypatch.setattr(
        train_mothnet.VisualizationManager, "visualize", _capturing_visualize
    )

    argv = [
        "--mb-units", "200",
        "--al-units", "64",
        "--epochs", "1",
        "--num-train-samples", "100",
        "--num-val-samples", "200",
        "--batch-size", "32",
        "--viz-freq", "1",
        "--output-dir", str(tmp_path),
        "--experiment-name", "mothnet_cm_absent_class_check",
    ]
    exit_code = train_mothnet.main(argv)
    assert exit_code == 0

    cm_calls = [
        call for call in captured_calls if call[0] == "confusion_matrix"
    ]
    assert len(cm_calls) == 1, f"expected exactly 1 confusion_matrix call, got {captured_calls}"
    _, classification_results = cm_calls[0]

    assert not np.any(np.isin(classification_results.y_true, [3, 7])), (
        "test setup: class 3/7 must be absent from y_true"
    )
    assert not np.any(np.isin(classification_results.y_pred, [3, 7])), (
        "test setup: class 3/7 must be absent from y_pred"
    )

    actual_cm = sklearn_confusion_matrix(
        classification_results.y_true, classification_results.y_pred
    )
    assert actual_cm.shape[0] == len(classification_results.class_names), (
        "class_names length does not match the ACTUAL confusion-matrix size "
        f"sklearn produced — matrix is {actual_cm.shape}, "
        f"class_names={classification_results.class_names!r} "
        "(this is the exact silent axis-mislabeling defect)"
    )
    assert "3" not in classification_results.class_names, (
        "class_names still includes an absent class — not dynamically derived"
    )
    assert "7" not in classification_results.class_names, (
        "class_names still includes an absent class — not dynamically derived"
    )


def test_visualize_or_warn_logs_a_warning_when_visualize_returns_none(caplog) -> None:
    """Step 5.3 completion-fix (`plan-2026-09-18T080513-debe8b11` D-007,
    adversarial-review finding 4, WARNING): `VisualizationManager.visualize()`
    (`dl_techniques/visualization/core.py`) already wraps its own internal
    plugin call in a `try/except Exception: logger.error(...); return None` —
    a plugin-internal failure never raises, it just returns `None` silently.
    `_visualize_or_warn` must notice that and log a `logger.warning`, since
    the trainer's own OUTER try/except blocks (wrapped around the whole
    render call) can never observe a failure that `visualize()` itself
    already swallowed.

    Uses a bare stand-in object with a `.visualize()` method returning `None`
    unconditionally — this test targets `_visualize_or_warn`'s own
    None-handling logic in isolation, not the real `VisualizationManager` or
    any real plugin, so no matplotlib/sklearn call happens here at all.
    """

    class _AlwaysNoneVizManager:
        def visualize(self, data, **kwargs):
            return None

    with caplog.at_level(logging.WARNING, logger="dl"):
        train_mothnet._visualize_or_warn(
            _AlwaysNoneVizManager(), data=object(), epoch=0,
            description="fake visualization", plugin_name="whatever",
        )

    assert "fake visualization" in caplog.text
    assert "None" in caplog.text


def test_visualize_or_warn_does_not_warn_when_visualize_succeeds(caplog) -> None:
    """The complementary GREEN path: a real (non-`None`) return value from
    `.visualize()` must NOT produce a warning — pins the `is None` check
    against a coarser "warn whenever called" regression that would make the
    log noisy on every successful periodic render.
    """

    class _AlwaysSucceedsVizManager:
        def visualize(self, data, **kwargs):
            return "not-none-sentinel"

    with caplog.at_level(logging.WARNING, logger="dl"):
        train_mothnet._visualize_or_warn(
            _AlwaysSucceedsVizManager(), data=object(), epoch=0,
            description="fake visualization", plugin_name="whatever",
        )

    assert caplog.text == ""


@pytest.mark.integration
def test_eval_batch_size_is_actually_forwarded_into_predict_in_batches(
    tmp_path: Path, monkeypatch,
) -> None:
    """Step 7.1 completion-fix (`plan-2026-09-18T080513-debe8b11`,
    adversarial-review finding 6, NOTE): nothing previously proved `main()`
    actually FORWARDS `args.eval_batch_size` into `_predict_in_batches` at
    BOTH of its call sites (val-side and train-side accuracy computation) —
    a hardcoded `5000` at either call site would have left every other test
    in this suite green, since none of them pass a non-default
    `--eval-batch-size` while also inspecting what `_predict_in_batches`
    itself received.

    Wraps `train_mothnet._predict_in_batches` (record-then-call-through, the
    same shape this module's other tests use) and drives a real tiny `main()`
    run with `--eval-batch-size 13` — a value distinct from the default
    (`5000`), from `--batch-size` (`32`), and from every other probe value
    used elsewhere in this test module/`test_cli_contract.py` — asserting
    EVERY recorded call used `batch_size=13`, not the default.
    """
    recorded_batch_sizes: list = []
    _real_predict_in_batches = train_mothnet._predict_in_batches

    def _recording_predict_in_batches(model, x, batch_size):
        recorded_batch_sizes.append(batch_size)
        return _real_predict_in_batches(model, x, batch_size)

    monkeypatch.setattr(
        train_mothnet, "_predict_in_batches", _recording_predict_in_batches
    )

    argv = [
        "--mb-units", "200",
        "--al-units", "64",
        "--epochs", "1",
        "--num-train-samples", "100",
        "--num-val-samples", "50",
        "--batch-size", "32",
        "--eval-batch-size", "13",
        "--viz-freq", "0",
        "--output-dir", str(tmp_path),
        "--experiment-name", "mothnet_eval_batch_size_forwarding_check",
    ]
    exit_code = train_mothnet.main(argv)
    assert exit_code == 0

    assert len(recorded_batch_sizes) == 2, (
        "expected exactly 2 _predict_in_batches calls (val-side + train-side "
        f"accuracy computation) per epoch, got {recorded_batch_sizes}"
    )
    assert recorded_batch_sizes == [13, 13], (
        "--eval-batch-size 13 was not forwarded to BOTH _predict_in_batches "
        f"call sites — recorded batch_size values: {recorded_batch_sizes}"
    )


@pytest.mark.parametrize("bad_value", ["0", "-1", "-100"])
def test_eval_batch_size_below_one_fails_fast_at_parse_time(bad_value, capsys) -> None:
    """Step 2.1 completion-fix (`plan-2026-09-18T080513-debe8b11`,
    adversarial-review finding 7, NOTE): `--eval-batch-size <= 0` used to be
    unvalidated and failed LATE, deep inside `_predict_in_batches`, only
    after a full training epoch had already run — `batch_size=0` raises
    `ValueError: range() arg 3 must not be zero`; a negative value raises
    `ValueError: need at least one array to concatenate` (an empty range).
    `--epochs < 1` already fails fast at parse time via `parser.error`; this
    proves `--eval-batch-size` now follows the exact same pattern: a
    `SystemExit` from `parse_arguments()` itself, before any dataset load,
    GPU setup, or model construction ever runs.
    """
    with pytest.raises(SystemExit):
        train_mothnet.parse_arguments(["--eval-batch-size", bad_value])

    stderr = capsys.readouterr().err
    assert "--eval-batch-size" in stderr
    assert "must be >= 1" in stderr


def test_eval_batch_size_of_one_is_accepted_at_parse_time() -> None:
    """Boundary check complementing the parametrized failure test above —
    `1` is the smallest VALID value and must parse cleanly, not be caught by
    an off-by-one `<= 1` guard.
    """
    args = train_mothnet.parse_arguments(["--eval-batch-size", "1"])
    assert args.eval_batch_size == 1
