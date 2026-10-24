"""The live topography callback: outputs, isolation, tripwire and non-perturbation.

Written RED-first (plan-2026-10-23T190557-972a9ac5, step 1), against the API the
plan fixes and nothing more: ``TopoLMTrainingConfig.viz_every_steps``,
``build_callbacks(config, model, results_dir, initial_step, backbone=...,
preprocessor=...)`` appending the callback LAST, the Keras hooks
(``on_train_begin``, ``on_train_batch_end``, ``on_train_end``), and the output
tree ``<run_dir>/topography_live/{step_NNNNNNN/, latest_*.png, metrics.jsonl,
topography_curve.png}``. The callback is reached through ``build_callbacks`` so
that no test depends on its constructor.

Two instruments are GREEN today and must stay so, because the proof of
non-perturbation rests on them (a guard that cannot fail is not a guard):

- ``TestAnEvalModePassTouchesNothing``: the eval-mode evaluation the callback
  reuses leaves every ``SeedGenerator`` and both global RNGs untouched, measured
  here with a snapshot of my own rather than by reading the code;
- ``TestTheABInstrument``: two plain same-seed runs reproduce each other bit for
  bit IN ONE PROCESS (plan Assumption 4), and a callback that runs a
  ``training=True`` forward makes the comparison fail, i.e. the instrument can
  see the defect it exists to rule out.

Everything that needs the callback is RED until step 6.
"""

import json
import os
import random

import keras
import numpy as np
import pytest

from train.topolm import (
    build_callbacks,
    create_topolm_model,
    evaluate_topography,
    plot_topography,
    SMOKE_STIMULI,
    train_paired,
    train_topolm,
)
from train.topolm import common as C
from dl_techniques.utils.logger import logger

from .test_topolm_trainer import (  # noqa: F401  (the autouse fixture must be in scope)
    LENGTH,
    VOCAB,
    StubTokenizer,
    _config,
    _never_write_to_repo_results,
    _patch_data,
)

#: The tiny head has one ``SeedGenerator`` per tap (4 taps), MEASURED; a Keras
#: that renames ``Layer._seed_generators`` would make the snapshot find none and
#: every "unchanged" assertion pass vacuously, so the count is asserted.
EXPECTED_SEED_GENERATORS = 4


# ---------------------------------------------------------------------
# instruments (independent of the code under test)
# ---------------------------------------------------------------------


def _seed_generators(model):
    """Every ``SeedGenerator`` reachable from ``model``, once each."""
    seen, found = set(), []
    for layer in model._flatten_layers():
        for generator in getattr(layer, "_seed_generators", []):
            if id(generator) not in seen:
                seen.add(id(generator))
                found.append(generator)
    return found


def _rng_snapshot(model):
    """Everything an eval-mode pass must not move."""
    numpy_state = np.random.get_state()
    return {
        "seed_generators": [g.state.numpy().copy() for g in _seed_generators(model)],
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
    }


def _assert_same_snapshot(before, after):
    assert len(before["seed_generators"]) == EXPECTED_SEED_GENERATORS, (
        "the snapshot found the wrong number of SeedGenerators, so 'unchanged' "
        "would be vacuous"
    )
    for index, (left, right) in enumerate(
        zip(before["seed_generators"], after["seed_generators"])
    ):
        assert np.array_equal(left, right), f"SeedGenerator {index} advanced"
    assert before["python"] == after["python"], "Python's random state moved"
    assert before["numpy"][0] == after["numpy"][0]
    assert np.array_equal(before["numpy"][1], after["numpy"][1]), "NumPy's RNG moved"
    assert before["numpy"][2:] == after["numpy"][2:]


def _live(tmp_path, initial_step=0, **overrides):
    """A tiny head and backbone, and the live callback ``build_callbacks`` returns.

    The callback is wired to the head the way ``fit`` would (``set_model``), so a
    test can call its Keras hooks directly, without a ``fit``.
    """
    config = _config(viz_every_steps=2, **overrides)
    head, backbone = create_topolm_model(config)
    callbacks = build_callbacks(
        config, head, str(tmp_path), initial_step,
        backbone=backbone, preprocessor=StubTokenizer(),
    )
    callback = callbacks[-1]
    callback.set_model(head)
    return config, head, backbone, callback


def _spy_on_evaluate(monkeypatch, behaviour):
    """Route the callback's evaluation through ``behaviour(n, real, args, kwargs)``.

    Patched where it is defined and, if the callback module imported the name,
    where it is used, so the test does not care which form the module chose.
    """
    calls = []
    real = C.evaluate_topography

    def wrapper(*args, **kwargs):
        calls.append(len(calls) + 1)
        return behaviour(len(calls), real, args, kwargs)

    monkeypatch.setattr(C, "evaluate_topography", wrapper)
    import train.topolm.live as live

    if hasattr(live, "evaluate_topography"):
        monkeypatch.setattr(live, "evaluate_topography", wrapper)
    return calls


def _raise(error):
    def behaviour(n, real, args, kwargs):
        raise error
    return behaviour


def _live_dir(tmp_path):
    return tmp_path / "topography_live"


def _step_dirs(tmp_path):
    return sorted(
        name for name in os.listdir(_live_dir(tmp_path)) if name.startswith("step_")
    )


# ---------------------------------------------------------------------
# GREEN today: the instruments
# ---------------------------------------------------------------------


class TestAnEvalModePassTouchesNothing:
    def test_evaluating_and_plotting_leaves_every_rng_alone(self, tmp_path):
        """What D-007 rests on, measured: ``evaluate_topography`` (both arms) and
        ``plot_topography`` on the tiny head change no SeedGenerator, no Python
        ``random`` state and no NumPy global state."""
        head, backbone = create_topolm_model(_config())
        before = _rng_snapshot(head)

        report, arrays = evaluate_topography(
            backbone, SMOKE_STIMULI, ("a", "b"), StubTokenizer(),
            readout_fwhm=2.0, min_cluster_size=2,
        )
        plot_topography(report, arrays, str(tmp_path))

        _assert_same_snapshot(before, _rng_snapshot(head))

    def test_the_snapshot_can_see_a_seed_generator_advance(self):
        """The both-ways twin: a ``training=True`` forward moves them."""
        head, backbone = create_topolm_model(_config())
        before = _rng_snapshot(head)
        backbone(np.zeros((2, LENGTH), dtype="int32"), training=True)
        after = _rng_snapshot(head)
        moved = [
            not np.array_equal(left, right)
            for left, right in zip(before["seed_generators"], after["seed_generators"])
        ]
        assert all(moved) and len(moved) == EXPECTED_SEED_GENERATORS

    def test_the_snapshot_can_see_both_global_generators_move(self):
        head, _ = create_topolm_model(_config())
        before = _rng_snapshot(head)
        random.random()
        np.random.rand()
        after = _rng_snapshot(head)
        assert before["python"] != after["python"]
        assert not np.array_equal(before["numpy"][1], after["numpy"][1])


# ---------------------------------------------------------------------
# the A/B instrument (GREEN today) and the callback's side of it (RED)
# ---------------------------------------------------------------------

#: ``(trainable, non_trainable, seed_generators, history)`` of the runs already
#: done in this process, keyed by label, so the controls are paid for once.
_RUNS = {}


class _TrainingModePass(keras.callbacks.Callback):
    """The defect the A/B must be able to see: a ``training=True`` forward every
    second step, which advances every tap's SeedGenerator and overwrites its
    non-trainable ``last_spatial_loss``."""

    def __init__(self, backbone):
        super().__init__()
        self._backbone = backbone
        self._step = 0

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1
        if self._step % 2 == 0:
            self._backbone(np.zeros((2, LENGTH), dtype="int32"), training=True)


def _group(variables):
    return [(v.path, np.array(v.numpy(), copy=True)) for v in variables]


def _train_and_snapshot(monkeypatch, tmp_path, label, extra_callback=False, **overrides):
    """One real ``train_topolm`` on CPU, reduced to what the A/B compares."""
    if label in _RUNS:
        return _RUNS[label]
    _patch_data(monkeypatch)
    monkeypatch.setattr(C, "_run_dir", lambda config: str(tmp_path / label))
    if extra_callback:
        captured = {}
        real_create, real_build = C.create_topolm_model, C.build_callbacks

        def capturing_create(*args, **kwargs):
            pair = real_create(*args, **kwargs)
            captured["backbone"] = pair[1]
            return pair

        monkeypatch.setattr(C, "create_topolm_model", capturing_create)
        monkeypatch.setattr(
            C, "build_callbacks",
            lambda *args, **kwargs: real_build(*args, **kwargs)
            + [_TrainingModePass(captured["backbone"])],
        )
    config = _config(**overrides)
    with keras.device("cpu"):
        result = train_topolm(config, preprocessor=StubTokenizer(), run_topography=False)
    model = result["model"]
    snapshot = {
        "trainable": _group(model.trainable_variables),
        "non_trainable": _group(model.non_trainable_variables),
        "seed_generators": [g.state.numpy().copy() for g in _seed_generators(model)],
        "history": {k: list(v) for k, v in result["history"].history.items()},
        "results_dir": result["results_dir"],
    }
    _RUNS[label] = snapshot
    return snapshot


def _differences(left, right, key):
    """Names of the variables in group ``key`` that are not bit-equal.

    Compared by POSITION, not by name: Keras numbers layers per process
    (``causal_language_model_1/...`` in the second run of a process), so the
    paths of two same-config runs differ in a counter even when the models are
    the same model. The shapes are asserted equal instead.
    """
    assert len(left[key]) == len(right[key])
    assert [a.shape for _, a in left[key]] == [b.shape for _, b in right[key]]
    return [
        name for (name, a), (_, b) in zip(left[key], right[key])
        if not np.array_equal(a, b)
    ]


def _assert_runs_identical(left, right):
    assert _differences(left, right, "trainable") == []
    assert _differences(left, right, "non_trainable") == []
    assert len(left["seed_generators"]) == EXPECTED_SEED_GENERATORS
    for a, b in zip(left["seed_generators"], right["seed_generators"]):
        assert np.array_equal(a, b)
    assert left["history"] == right["history"]


class TestTheABInstrument:
    def test_a_plain_run_is_bit_identical_to_itself_in_one_process(self, monkeypatch, tmp_path):
        """The control the plan insists on: if two plain same-seed runs differ,
        nothing a callback does can be concluded from a comparison."""
        first = _train_and_snapshot(monkeypatch, tmp_path, "plain_a")
        second = _train_and_snapshot(monkeypatch, tmp_path, "plain_b")
        assert len(first["trainable"]) > 10 and len(first["non_trainable"]) > 4
        _assert_runs_identical(first, second)

    def test_a_default_run_writes_no_live_directory(self, monkeypatch, tmp_path):
        run = _train_and_snapshot(monkeypatch, tmp_path, "plain_a")
        assert not os.path.exists(os.path.join(run["results_dir"], "topography_live"))

    def test_negative_control_a_training_mode_pass_makes_the_comparison_fail(self, monkeypatch, tmp_path):
        """NEGATIVE CONTROL: the same run with a ``training=True`` forward every
        second step. Both variable groups must differ, so the A/B can fail on
        either path the defect takes (measured by the explorer: the trainable
        weights move through the advanced SeedGenerators, the non-trainable
        ``last_spatial_loss`` is overwritten even when they are restored)."""
        plain = _train_and_snapshot(monkeypatch, tmp_path, "plain_a")
        perturbed = _train_and_snapshot(
            monkeypatch, tmp_path, "training_pass", extra_callback=True
        )
        assert _differences(plain, perturbed, "trainable"), "the trainable group is blind"
        assert _differences(plain, perturbed, "non_trainable"), (
            "the non-trainable group is blind"
        )

    def test_the_live_pass_leaves_the_run_bit_identical(self, monkeypatch, tmp_path):
        """THE A/B, RED until step 6: same config, same seed, ``viz_every_steps=2``.
        Trainable and non-trainable weights, SeedGenerator states and every
        per-epoch loss equal the plain run's, bit for bit."""
        plain = _train_and_snapshot(monkeypatch, tmp_path, "plain_a")
        live = _train_and_snapshot(
            monkeypatch, tmp_path, "with_live", viz_every_steps=2
        )
        assert os.path.isdir(os.path.join(live["results_dir"], "topography_live"))
        _assert_runs_identical(plain, live)


# ---------------------------------------------------------------------
# RED until step 6: wiring, outputs, cadence
# ---------------------------------------------------------------------


class TestWiring:
    def test_the_callback_is_exported_and_importable(self):
        from train.topolm import TopographyLiveCallback  # noqa: F401
        from train.topolm.live import TopographyLiveCallback as same

        assert same is TopographyLiveCallback

    def test_it_is_the_last_callback_and_the_others_are_unchanged(self, tmp_path):
        config = _config(viz_every_steps=2)
        head, backbone = create_topolm_model(config)
        enabled = build_callbacks(
            config, head, str(tmp_path), 0,
            backbone=backbone, preprocessor=StubTokenizer(),
        )
        baseline = build_callbacks(_config(), head, str(tmp_path), 0)
        assert [type(c) for c in enabled[:-1]] == [type(c) for c in baseline]
        assert type(enabled[-1]).__name__ == "TopographyLiveCallback"

    def test_zero_adds_nothing(self, tmp_path):
        config = _config(viz_every_steps=0)
        head, backbone = create_topolm_model(config)
        callbacks = build_callbacks(
            config, head, str(tmp_path), 0,
            backbone=backbone, preprocessor=StubTokenizer(),
        )
        assert "TopographyLiveCallback" not in [type(c).__name__ for c in callbacks]
        assert len(callbacks) == 4


class TestOutputs:
    def test_a_six_step_run_draws_at_0_2_4_and_6(self, monkeypatch, tmp_path):
        """``viz_every_steps=2`` and 6 optimizer steps: the untrained map, every
        second step, and the end -- which is already a multiple of 2, so the
        ``on_train_end`` draw must not duplicate it."""
        _patch_data(monkeypatch)
        monkeypatch.setattr(C, "_run_dir", lambda config: str(tmp_path / "live_run"))
        result = train_topolm(
            _config(viz_every_steps=2), preprocessor=StubTokenizer(),
            run_topography=False,
        )
        live = tmp_path / "live_run" / "topography_live"
        assert result["results_dir"] == str(tmp_path / "live_run")

        names = sorted(n for n in os.listdir(live) if n.startswith("step_"))
        assert names == ["step_0000000", "step_0000002", "step_0000004", "step_0000006"]
        for name in names:
            assert (live / name / "t_maps_raw.png").stat().st_size > 0
            assert (live / name / "morans_i_raw.png").stat().st_size > 0
            assert not (live / name / "t_maps_readout.png").exists(), "raw arm only"

        for name in ("latest_t_maps_raw.png", "latest_morans_i_raw.png",
                     "topography_curve.png"):
            assert (live / name).stat().st_size > 0, name

        rows = [
            json.loads(line, parse_constant=lambda c: pytest.fail(f"bare {c}"))
            for line in (live / "metrics.jsonl").read_text().splitlines()
        ]
        # The map draws are 0, 2, 4, 6 (the end of training does not duplicate
        # the multiple of 2). The epoch-end validation of step 6 runs AFTER that
        # last draw, so the file closes with one metrics-only row of step 6
        # (step 6.3, TestTheLastValidationIsDrawn pins it); it is not a draw.
        assert [row["step"] for row in rows if not row.get("val_only")] == [0, 2, 4, 6]
        assert [row["step"] for row in rows if row.get("val_only")] == [6]

    def test_the_labels_add_the_resume_offset(self, tmp_path):
        """The batch index restarts every virtual epoch; the label is
        ``initial_step`` plus the callback's own count of optimizer steps."""
        _, _, _, callback = _live(tmp_path, initial_step=100)
        callback.on_train_begin()
        for batch in range(4):
            callback.on_train_batch_end(batch, {})
        callback.on_train_end()
        assert _step_dirs(tmp_path) == ["step_0000100", "step_0000102", "step_0000104"]

    def test_the_end_of_training_draws_a_final_frame_when_it_is_not_a_multiple(
        self, tmp_path
    ):
        _, _, _, callback = _live(tmp_path)
        callback.on_train_begin()
        for batch in range(3):
            callback.on_train_batch_end(batch, {})
        callback.on_train_end()
        assert _step_dirs(tmp_path) == ["step_0000000", "step_0000002", "step_0000003"]

    def test_paired_runs_get_one_live_directory_per_arm(self, monkeypatch):
        _patch_data(monkeypatch)
        results = train_paired(
            _config(viz_every_steps=2, save_dir="paired_live"),
            preprocessor=StubTokenizer(), run_topography=False,
        )
        directories = {
            arm: os.path.join(result["results_dir"], "topography_live")
            for arm, result in results.items()
        }
        assert directories["topographic"] != directories["control"]
        for arm, path in directories.items():
            assert os.path.isdir(path), arm
            assert "step_0000000" in os.listdir(path), arm


# ---------------------------------------------------------------------
# RED until step 6: a visualization failure never kills a run
# ---------------------------------------------------------------------


class TestFailureIsolation:
    def test_three_consecutive_failures_disable_the_callback_quietly(
        self, monkeypatch, tmp_path
    ):
        _, _, _, callback = _live(tmp_path)
        calls = _spy_on_evaluate(monkeypatch, _raise(RuntimeError("boom")))

        callback.on_train_begin()                    # failure 1, step 0
        for batch in range(10):                      # draws due at 2, 4, 6, 8, 10
            callback.on_train_batch_end(batch, {})
        callback.on_train_end()

        assert len(calls) == 3, "it kept drawing after three consecutive failures"

    def test_a_success_resets_the_count(self, monkeypatch, tmp_path):
        """Two failures, a success, two failures, a success, ... never reaches
        three in a row, so the callback keeps going: 8 draws, 8 attempts."""
        _, _, _, callback = _live(tmp_path)
        outcomes = ["raise", "raise", "ok", "raise", "raise", "ok", "raise", "raise"]
        cache = {}

        def behaviour(n, real, args, kwargs):
            if outcomes[n - 1] == "raise":
                raise RuntimeError("boom")
            if "value" not in cache:
                cache["value"] = real(*args, **kwargs)
            return cache["value"]

        calls = _spy_on_evaluate(monkeypatch, behaviour)

        callback.on_train_begin()
        for batch in range(14):
            callback.on_train_batch_end(batch, {})

        assert len(calls) == 8

    def test_a_keyboard_interrupt_is_never_swallowed(self, monkeypatch, tmp_path):
        _, _, _, callback = _live(tmp_path)
        _spy_on_evaluate(monkeypatch, _raise(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            callback.on_train_begin()


# ---------------------------------------------------------------------
# RED until step 6: the tripwire
# ---------------------------------------------------------------------


class TestTheTripwire:
    """The instrument that makes the guard able to fail. A draw that moves
    anything it must not is restored, reported at ERROR level, and the callback
    stops drawing; the run itself goes on."""

    def _drive(self, monkeypatch, tmp_path, perturb):
        config, head, backbone, callback = _live(tmp_path)
        errors = []
        monkeypatch.setattr(
            logger, "error", lambda message, *a, **k: errors.append(str(message))
        )

        def behaviour(n, real, args, kwargs):
            perturb(backbone)
            return real(*args, **kwargs)

        calls = _spy_on_evaluate(monkeypatch, behaviour)
        before = _rng_snapshot(head)
        callback.on_train_begin()
        after = _rng_snapshot(head)
        for batch in range(4):
            callback.on_train_batch_end(batch, {})
        return before, after, errors, calls

    def test_a_seed_generator_advance_is_restored_reported_and_disables(
        self, monkeypatch, tmp_path
    ):
        def advance(backbone):
            keras.random.uniform((1,), seed=backbone.tap_layers[0].seed_generator)

        before, after, errors, calls = self._drive(monkeypatch, tmp_path, advance)
        _assert_same_snapshot(before, after)
        assert errors, "the tripwire fired without an ERROR line"
        assert len(calls) == 1, "the callback kept drawing after the tripwire"

    def test_a_python_rng_draw_is_restored_reported_and_disables(
        self, monkeypatch, tmp_path
    ):
        before, after, errors, calls = self._drive(
            monkeypatch, tmp_path, lambda backbone: random.random()
        )
        _assert_same_snapshot(before, after)
        assert errors and len(calls) == 1

    def test_a_numpy_rng_draw_is_restored_reported_and_disables(
        self, monkeypatch, tmp_path
    ):
        before, after, errors, calls = self._drive(
            monkeypatch, tmp_path, lambda backbone: np.random.rand()
        )
        _assert_same_snapshot(before, after)
        assert errors and len(calls) == 1

    def test_a_clean_draw_fires_nothing(self, monkeypatch, tmp_path):
        before, after, errors, calls = self._drive(
            monkeypatch, tmp_path, lambda backbone: None
        )
        _assert_same_snapshot(before, after)
        assert not errors
        assert len(calls) == 3, "draws at steps 0, 2 and 4 were expected"


# =====================================================================
# step 6 additions (plan-2026-10-23T190557-972a9ac5): the guards the step-1
# file did not cover, each seen RED against a mutation of live.py
# =====================================================================


class TestEvalModeOnly:
    def test_every_forward_in_a_draw_is_training_false_and_never_the_head(
        self, monkeypatch, tmp_path
    ):
        """The rule D-007 anchors, observed rather than read: record every call
        of the backbone and of the head during a draw. Each backbone call
        passes ``training=False`` explicitly; the head (the CLM matmul over the
        vocabulary, tracker state) is not called at all."""
        _, head, backbone, callback = _live(tmp_path)
        calls = []
        for model, label in ((backbone, "backbone"), (head, "head")):
            original = type(model).__call__

            def spy(self, *args, _original=original, **kwargs):
                if self is backbone:
                    calls.append(("backbone", kwargs.get("training", "unset")))
                elif self is head:
                    calls.append(("head", kwargs.get("training", "unset")))
                return _original(self, *args, **kwargs)

            monkeypatch.setattr(type(model), "__call__", spy)

        callback.on_train_begin()

        assert calls, "the draw did not run a forward pass"
        assert {kind for kind, _ in calls} == {"backbone"}, calls
        assert {training for _, training in calls} == {False}, calls


class TestLatestFilesAreAtomic:
    def test_a_latest_file_appears_only_through_os_replace(self, monkeypatch, tmp_path):
        """With ``os.replace`` turned into a recorder that moves nothing, no
        ``latest_*`` file and no curve exists under its final name: they can only
        arrive by the rename. A callback that wrote them in place would show
        them here."""
        _, _, _, callback = _live(tmp_path)
        moves = []
        monkeypatch.setattr(os, "replace", lambda src, dst: moves.append((src, dst)))

        callback.on_train_begin()

        live = _live_dir(tmp_path)
        finals = sorted(os.path.basename(dst) for _, dst in moves)
        assert finals == [
            "latest_morans_i_raw.png", "latest_t_maps_raw.png", "topography_curve.png",
        ]
        for name in finals:
            assert not (live / name).exists(), f"{name} was written in place"
        for source, destination in moves:
            assert os.path.dirname(source) == os.path.dirname(destination), (
                "the temp file must sit beside its target, or the rename is not atomic"
            )

    def test_the_final_names_are_complete_png_files(self, tmp_path):
        _, _, _, callback = _live(tmp_path)
        callback.on_train_begin()
        for name in ("latest_t_maps_raw.png", "latest_morans_i_raw.png",
                     "topography_curve.png"):
            assert (_live_dir(tmp_path) / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert not [n for n in os.listdir(_live_dir(tmp_path)) if n.startswith(".")]


class TestTheMetricsRows:
    def test_a_non_finite_statistic_becomes_null_and_the_row_stays_strict_json(
        self, monkeypatch, tmp_path
    ):
        _, _, _, callback = _live(tmp_path)

        def behaviour(n, real, args, kwargs):
            report, arrays = real(*args, **kwargs)
            arm = report["arms"]["raw"]
            arm["mean_morans_i"] = float("nan")
            first = next(iter(arm["per_tap"].values()))
            first["morans_i"]["standard"] = float("inf")
            return report, arrays

        _spy_on_evaluate(monkeypatch, behaviour)
        callback.on_train_begin()

        text = (_live_dir(tmp_path) / "metrics.jsonl").read_text()
        row = json.loads(text, parse_constant=lambda c: pytest.fail(f"bare {c}"))
        assert row["mean_morans_i"] is None
        assert next(iter(row["per_tap"].values()))["morans_i"] is None

    def test_a_row_carries_the_step_the_validation_loss_was_measured_at(self, tmp_path):
        _, _, _, callback = _live(tmp_path)
        callback.on_train_begin()
        for batch in range(2):
            callback.on_train_batch_end(batch, {})
        callback.on_epoch_end(0, {"val_loss": 3.25})
        callback.on_train_batch_end(2, {})
        callback.on_train_batch_end(3, {})

        rows = [
            json.loads(line)
            for line in (_live_dir(tmp_path) / "metrics.jsonl").read_text().splitlines()
        ]
        assert [r["step"] for r in rows] == [0, 2, 4]
        assert [r["val_loss"] for r in rows] == [None, None, 3.25]
        assert rows[2]["val_loss_step"] == 2

    def test_a_row_holds_one_entry_per_tap_with_the_t_map_summary(self, tmp_path):
        config, head, backbone, callback = _live(tmp_path)
        callback.on_train_begin()
        row = json.loads((_live_dir(tmp_path) / "metrics.jsonl").read_text())
        assert len(row["per_tap"]) == len(backbone.tap_layers)
        for entry in row["per_tap"].values():
            assert set(entry) == {
                "morans_i", "morans_i_islands", "num_significant_units",
                "clusters_a", "clusters_b", "t_min", "t_max", "t_abs_mean",
            }
            assert entry["t_min"] <= entry["t_abs_mean"] <= max(
                abs(entry["t_min"]), abs(entry["t_max"])
            )
        assert row["grid_shape"] == list(backbone.grid_shape)
        assert row["stimuli_are_smoke_set"] is True


class TestBothRanks:
    def test_a_3d_backbone_draws_three_plane_figures_and_finite_rows(self, tmp_path):
        """The callback forwards to the rank-dispatching evaluation and plots, so
        a ``tiny3d`` draw is the 3D figure and a row whose grid is a 3-list."""
        config, head, backbone, callback = _live(
            tmp_path, model_variant="tiny3d", spatial_radius=None
        )
        callback.on_train_begin()

        step_dir = _live_dir(tmp_path) / "step_0000000"
        assert (step_dir / "t_maps_raw.png").stat().st_size > 0
        assert (_live_dir(tmp_path) / "latest_t_maps_raw.png").stat().st_size > 0
        row = json.loads((_live_dir(tmp_path) / "metrics.jsonl").read_text())
        assert row["grid_shape"] == [7, 7, 7]
        assert all(
            entry["t_max"] is not None and entry["t_min"] is not None
            for entry in row["per_tap"].values()
        )


class TestTheFlagIsRefusedNegativeAndNeedsItsInputs:
    def test_a_negative_cadence_is_refused_by_the_field_name(self, tmp_path):
        config = _config(viz_every_steps=-1)
        head, backbone = create_topolm_model(config)
        with pytest.raises(ValueError, match="viz_every_steps"):
            build_callbacks(config, head, str(tmp_path), 0, backbone=backbone,
                            preprocessor=StubTokenizer())

    def test_a_positive_cadence_without_a_backbone_is_refused(self, tmp_path):
        config = _config(viz_every_steps=2)
        head, _ = create_topolm_model(config)
        with pytest.raises(ValueError, match="backbone"):
            build_callbacks(config, head, str(tmp_path), 0)

    def test_the_value_reaches_the_callbacks_cadence(self, tmp_path):
        """D-011: the config-live census does not cover this config, so the field
        is proven live here: it is the callback's cadence, and
        ``contrast_conditions``/``min_cluster_size`` ride along."""
        _, _, _, callback = _live(tmp_path)
        assert callback.viz_every_steps == 2
        assert callback.min_cluster_size == 2
        assert callback.contrast_conditions == ("a", "b")


class TestTheDrawIsCheapAndTheTripwireIsNotBlind:
    def test_a_draw_forwards_in_batches_of_eight(self, monkeypatch, tmp_path):
        """The live pass shares the card with the training step, so it runs at
        batch 8 (not ``extract_tap_activations``' default 32)."""
        _, _, _, callback = _live(tmp_path)
        sizes = []
        real = C.extract_tap_activations

        def spy(backbone, prompts, preprocessor, batch_size=32):
            sizes.append(batch_size)
            return real(backbone, prompts, preprocessor, batch_size=batch_size)

        monkeypatch.setattr(C, "extract_tap_activations", spy)
        callback.on_train_begin()
        assert sizes and set(sizes) == {8}

    def test_a_draw_scores_only_the_two_contrast_conditions_on_the_raw_arm(
        self, monkeypatch, tmp_path
    ):
        _, _, _, callback = _live(tmp_path)
        seen = {}

        def behaviour(n, real, args, kwargs):
            seen["conditions"] = sorted(args[1])
            seen["readout_fwhm"] = kwargs.get("readout_fwhm")
            seen["permutation_p_value"] = kwargs.get("permutation_p_value")
            return real(*args, **kwargs)

        _spy_on_evaluate(monkeypatch, behaviour)
        callback.on_train_begin()
        assert seen == {
            "conditions": ["a", "b"], "readout_fwhm": None,
            "permutation_p_value": False,
        }

    def test_a_blind_snapshot_is_reported_at_the_start_of_training(
        self, monkeypatch, tmp_path
    ):
        """If a Keras release renames ``Layer._seed_generators`` the tripwire finds
        none and every comparison passes vacuously: say so, once, up front."""
        import train.topolm.live as live

        _, _, _, callback = _live(tmp_path)
        warnings = []
        monkeypatch.setattr(live, "seed_generators", lambda model: [])
        monkeypatch.setattr(
            logger, "warning", lambda message, *a, **k: warnings.append(str(message))
        )
        callback.on_train_begin()
        assert any("SeedGenerator" in message for message in warnings)


# =====================================================================
# step 6.2 (plan-2026-10-23T190557-972a9ac5, decision D-018): the final frame
# shows the weights the run ENDS with, and a damaged metrics.jsonl does not
# kill the view. Written RED-first against the 02af58996 callback.
# =====================================================================


def _rows(tmp_path):
    return [
        json.loads(line)
        for line in (_live_dir(tmp_path) / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _force_an_early_stop_at_the_second_evaluation(monkeypatch):
    """Make ``ConsecutiveIncreaseEarlyStopping`` see an increase at every
    evaluation after the first, so with ``early_stop_patience=1`` it stops at
    evaluation 2 and restores the snapshot taken at evaluation 1. The real
    callback, its real ``on_train_end`` restore: only the verdict is forced."""
    from dl_techniques.callbacks import consecutive_increase_early_stopping as stopping

    def forced(self, value):
        if self._previous is None:
            self._previous = value
            return False
        return True

    monkeypatch.setattr(
        stopping.ConsecutiveIncreaseEarlyStopping, "_is_increase", forced
    )


def _stopped_run(monkeypatch, tmp_path, viz_every_steps):
    """A 16-step budget, validation every 4 steps, forced stop at evaluation 2
    (step 8). Returns ``(config, result, rows)`` of a real CPU ``train_topolm``."""
    _patch_data(monkeypatch)
    _force_an_early_stop_at_the_second_evaluation(monkeypatch)
    monkeypatch.setattr(
        C, "_run_dir", lambda config: str(tmp_path / f"stopped_{viz_every_steps}")
    )
    config = _config(
        train_steps=16, eval_every_steps=4, viz_every_steps=viz_every_steps,
        early_stop_patience=1,
    )
    with keras.device("cpu"):
        result = train_topolm(config, preprocessor=StubTokenizer(), run_topography=False)
    live = tmp_path / f"stopped_{viz_every_steps}" / "topography_live"
    rows = [json.loads(l) for l in (live / "metrics.jsonl").read_text().splitlines()]
    return config, result, live, rows


def _post_fit_mean_morans_i(config, result):
    """The mean Moran's I of the model ``fit`` left behind, scored by the same
    public function the callback uses, but called by the test on its own."""
    report, _ = evaluate_topography(
        result["backbone"], SMOKE_STIMULI, ("a", "b"), StubTokenizer(),
        readout_fwhm=None, min_cluster_size=config.min_cluster_size,
        permutation_p_value=False, batch_size=8,
    )
    return report["arms"]["raw"]["mean_morans_i"]


class TestTheFinalFrameShowsTheEndingWeights:
    def test_a_stop_on_a_draw_step_redraws_the_restored_model(self, monkeypatch, tmp_path):
        """``viz_every_steps=4`` divides the stopping step 8: the batch-end draw
        scores the pre-restore weights, the early stopper then restores, and the
        last frame must be the restored model, equal to an independent
        evaluation of the model ``fit`` returned."""
        config, result, live, rows = _stopped_run(monkeypatch, tmp_path, 4)
        assert len(result["history"].history["loss"]) == 2, "the forced stop did not happen"

        assert [row["step"] for row in rows] == [0, 4, 8, 8]
        pre_restore, last = rows[2], rows[3]
        expected = _post_fit_mean_morans_i(config, result)
        assert last["mean_morans_i"] == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert pre_restore["mean_morans_i"] != pytest.approx(expected, rel=1e-6), (
            "the pre-restore frame equals the restored one: the test cannot tell them apart"
        )
        assert last["restored_weights"] is True
        assert all("restored_weights" not in row for row in rows[:3])

        # the frame on disk is the redrawn one
        assert (live / "latest_t_maps_raw.png").read_bytes() == (
            live / "step_0000008" / "t_maps_raw.png"
        ).read_bytes()
        assert sorted(n for n in os.listdir(live) if n.startswith("step_")) == [
            "step_0000000", "step_0000004", "step_0000008",
        ]

    def test_a_stop_off_the_draw_grid_still_ends_on_the_restored_model(
        self, monkeypatch, tmp_path
    ):
        """``viz_every_steps=3``: step 8 was not drawn, so the end-of-training draw
        is the first draw of that step and is a plain row, already restored."""
        config, result, live, rows = _stopped_run(monkeypatch, tmp_path, 3)
        assert [row["step"] for row in rows] == [0, 3, 6, 8]
        assert rows[-1]["mean_morans_i"] == pytest.approx(
            _post_fit_mean_morans_i(config, result), rel=1e-9, abs=1e-12
        )
        assert all("restored_weights" not in row for row in rows)

    def test_weights_that_change_after_the_last_draw_are_drawn_again(self, tmp_path):
        """The mechanism without a ``fit``: a stand-in for the early stopper
        overwrites the weights between the last batch-end draw and
        ``on_train_end``."""
        _, _, backbone, callback = _live(tmp_path)
        callback.on_train_begin()
        for batch in range(2):
            callback.on_train_batch_end(batch, {})
        before = _rows(tmp_path)[-1]["mean_morans_i"]
        for variable in backbone.trainable_variables:
            variable.assign(variable + 0.37)
        callback.on_train_end()

        rows = _rows(tmp_path)
        assert [row["step"] for row in rows] == [0, 2, 2]
        assert rows[-1]["restored_weights"] is True and "restored_weights" not in rows[1]
        assert rows[-1]["mean_morans_i"] != before
        assert _step_dirs(tmp_path) == ["step_0000000", "step_0000002"]
        assert (_live_dir(tmp_path) / "latest_t_maps_raw.png").read_bytes() == (
            _live_dir(tmp_path) / "step_0000002" / "t_maps_raw.png"
        ).read_bytes()

    def test_unchanged_weights_are_not_drawn_twice(self, monkeypatch, tmp_path):
        """No restore, no change: the run ends on a draw step and nothing is
        redrawn, so the steps stay strictly increasing and no row is flagged."""
        _, _, _, callback = _live(tmp_path)
        calls = _spy_on_evaluate(monkeypatch, lambda n, real, a, k: real(*a, **k))
        callback.on_train_begin()
        for batch in range(4):
            callback.on_train_batch_end(batch, {})
        callback.on_train_end()

        assert len(calls) == 3
        rows = _rows(tmp_path)
        steps = [row["step"] for row in rows]
        assert steps == [0, 2, 4] and steps == sorted(set(steps))
        assert all("restored_weights" not in row for row in rows)

    def test_the_fingerprint_sees_a_weight_change_and_nothing_else(self, tmp_path):
        """The both-ways twin of the two tests above, on the fingerprint itself."""
        from train.topolm.live import same_fingerprint

        _, _, backbone, callback = _live(tmp_path)
        first = callback._fingerprint()
        assert same_fingerprint(callback._fingerprint(), first)
        variable = backbone.trainable_variables[0]
        old = np.array(variable.numpy())
        variable.assign(old + 1e-3)
        assert not same_fingerprint(callback._fingerprint(), first)
        variable.assign(old)
        assert same_fingerprint(callback._fingerprint(), first)
        assert not same_fingerprint(first, None)

    def test_a_fingerprint_never_touches_an_rng_or_the_weights(self, tmp_path):
        _, head, backbone, callback = _live(tmp_path)
        before_rng = _rng_snapshot(head)
        before = [np.array(v.numpy()) for v in backbone.weights]
        callback._fingerprint()
        _assert_same_snapshot(before_rng, _rng_snapshot(head))
        for old, variable in zip(before, backbone.weights):
            assert np.array_equal(old, variable.numpy())

    def test_the_curve_uses_the_last_row_of_a_repeated_step(self, monkeypatch):
        """``plot_topography_curve`` draws one point per step: the LAST row of a
        step that was drawn twice (before and after the restore)."""
        import matplotlib.axes
        from train.topolm.plotting import plot_topography_curve

        def row(step, moran, **extra):
            return {
                "step": step, "mean_morans_i": moran, "total_clusters_a": 1,
                "total_clusters_b": 1, "mean_num_significant_units": 3.0,
                "val_loss": None, "val_loss_step": None,
                "per_tap": {"t/a": {"morans_i": moran}}, **extra,
            }

        rows = [row(0, 0.0), row(4, 0.2), row(4, 0.1, restored_weights=True)]
        drawn = []
        real = matplotlib.axes.Axes.plot

        def spy(self, *args, **kwargs):
            drawn.append((kwargs.get("label"), np.asarray(args[0]), np.asarray(args[1])))
            return real(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy)
        plot_topography_curve(rows, None)
        mean_line = [d for d in drawn if d[0] == "mean over taps"]
        assert len(mean_line) == 1
        assert mean_line[0][1].tolist() == [0.0, 4.0]
        assert mean_line[0][2].tolist() == pytest.approx([0.0, 0.1])
        for _, x, _ in drawn:
            assert len(x) == len(set(x.tolist())), "a step was drawn twice"


class TestADamagedMetricsFileDoesNotKillTheView:
    def _write_damaged(self, tmp_path):
        """One good row, then a row cut off mid-write (no newline at the end)."""
        _, _, _, probe = _live(tmp_path / "donor")
        probe.on_train_begin()
        good = (_live_dir(tmp_path / "donor") / "metrics.jsonl").read_text()
        _live_dir(tmp_path).mkdir(parents=True, exist_ok=True)
        (_live_dir(tmp_path) / "metrics.jsonl").write_text(
            good + good[: len(good) // 2]
        )
        return good

    def test_a_truncated_last_line_is_skipped_with_one_warning(
        self, monkeypatch, tmp_path
    ):
        self._write_damaged(tmp_path)
        _, _, _, callback = _live(tmp_path)
        warnings = []
        monkeypatch.setattr(
            logger, "warning", lambda message, *a, **k: warnings.append(str(message))
        )
        callback.on_train_begin()
        for batch in range(2):                      # a second draw, at step 2
            callback.on_train_batch_end(batch, {})

        assert not callback.disabled
        assert callback._consecutive_failures == 0 and callback._draws == 2
        about_the_file = [w for w in warnings if "metrics.jsonl" in w]
        assert len(about_the_file) == 1, about_the_file
        assert "skipped" in about_the_file[0]
        assert (_live_dir(tmp_path) / "topography_curve.png").read_bytes()[:4] == b"\x89PNG"

    def test_a_row_appended_after_a_truncated_line_is_a_whole_row(self, tmp_path):
        """A cut-off row has no newline, so a plain append would glue the next
        row onto it and lose BOTH. The new row must start on its own line."""
        self._write_damaged(tmp_path)
        _, _, _, callback = _live(tmp_path)
        callback.on_train_begin()
        lines = (_live_dir(tmp_path) / "metrics.jsonl").read_text().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0])["step"] == 0
        with pytest.raises(ValueError):
            json.loads(lines[1])
        assert json.loads(lines[2])["step"] == 0

    def test_an_intact_file_warns_about_nothing(self, monkeypatch, tmp_path):
        _, _, _, callback = _live(tmp_path)
        warnings = []
        monkeypatch.setattr(
            logger, "warning", lambda message, *a, **k: warnings.append(str(message))
        )
        callback.on_train_begin()
        assert not [w for w in warnings if "metrics.jsonl" in w]


# =====================================================================
# step 6.3 (plan-2026-10-23T190557-972a9ac5, decision D-019, defect D1): the
# last epoch-end validation is shown. Keras validates AFTER the last batch-end
# draw, so without a closing metrics-only row the curve ends one cadence early.
# Written RED-first against the 244970b78 callback.
# =====================================================================


def _curve_row(step, moran=0.1, **extra):
    """A complete metrics row, as far as the curve reads one."""
    return {
        "step": step, "mean_morans_i": moran, "total_clusters_a": 1,
        "total_clusters_b": 1, "mean_num_significant_units": 3.0,
        "val_loss": None, "val_loss_step": None,
        "per_tap": {"t/a": {"morans_i": moran}}, **extra,
    }


def _val_only(step, val_loss, val_loss_step):
    return {
        "step": step, "val_only": True, "val_loss": val_loss,
        "val_loss_step": val_loss_step,
    }


def _plotted_validation_points(monkeypatch, rows):
    """``[(x list, y list)]`` of every line the curve draws in the val_loss panel
    (the only green one), captured from ``Axes.plot`` itself."""
    import matplotlib.axes
    from train.topolm.plotting import plot_topography_curve

    drawn = []
    real = matplotlib.axes.Axes.plot

    def spy(self, *args, **kwargs):
        if isinstance(kwargs.get("color"), str) and kwargs["color"] == "tab:green":
            drawn.append((np.asarray(args[0]).tolist(), np.asarray(args[1]).tolist()))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy)
    plot_topography_curve(rows, None)
    return drawn


def _drive_to_a_trailing_validation(callback, batches=2, val_loss=3.25):
    """``batches`` optimizer steps (the last one a draw step), then the epoch-end
    validation Keras runs AFTER that batch-end draw, then the end of training."""
    callback.on_train_begin()
    for batch in range(batches):
        callback.on_train_batch_end(batch, {})
    callback.on_epoch_end(0, {"val_loss": val_loss})
    callback.on_train_end()


class TestTheLastValidationIsDrawn:
    def test_a_validation_after_the_last_draw_gets_a_metrics_only_row(self, tmp_path):
        _, _, _, callback = _live(tmp_path)
        _drive_to_a_trailing_validation(callback)

        rows = _rows(tmp_path)
        assert [row["step"] for row in rows] == [0, 2, 2]
        full, closing = rows[1], rows[2]
        assert full["val_loss"] is None and "val_only" not in full
        assert closing["val_only"] is True
        assert closing["val_loss"] == 3.25 and closing["val_loss_step"] == 2
        for name in ("per_tap", "mean_morans_i", "total_clusters_a", "seconds",
                     "grid_shape", "restored_weights"):
            assert name not in closing, name
        # strict JSON, like every other row
        json.loads(
            (_live_dir(tmp_path) / "metrics.jsonl").read_text().splitlines()[-1],
            parse_constant=lambda c: pytest.fail(f"bare {c}"),
        )

    def test_the_closing_row_redraws_only_the_curve(self, monkeypatch, tmp_path):
        """No second map draw: no evaluation, no new step directory, the t-map
        and Moran's I frames untouched byte for byte, the curve rewritten."""
        _, _, _, callback = _live(tmp_path)
        calls = _spy_on_evaluate(monkeypatch, lambda n, real, a, k: real(*a, **k))
        callback.on_train_begin()
        for batch in range(2):
            callback.on_train_batch_end(batch, {})
        callback.on_epoch_end(0, {"val_loss": 3.25})
        live = _live_dir(tmp_path)
        names = ("latest_t_maps_raw.png", "latest_morans_i_raw.png")
        before = {name: (live / name).read_bytes() for name in names}
        curve_before = (live / "topography_curve.png").read_bytes()
        drawn = []
        real_plot = C.plot_topography
        monkeypatch.setattr(
            C, "plot_topography", lambda *a, **k: drawn.append(1) or real_plot(*a, **k)
        )
        callback.on_train_end()

        assert len(calls) == 2 and drawn == [], "a map was drawn for the closing row"
        assert _step_dirs(tmp_path) == ["step_0000000", "step_0000002"]
        for name in names:
            assert (live / name).read_bytes() == before[name], name
        assert (live / "topography_curve.png").read_bytes() != curve_before
        assert not [n for n in os.listdir(live) if n.startswith(".")]

    def test_no_validation_after_the_last_draw_adds_no_row(self, tmp_path):
        """The validation of step 2 is already in the step-4 row, and a run that
        never validated has nothing to add."""
        _, _, _, callback = _live(tmp_path)
        callback.on_train_begin()
        for batch in range(2):
            callback.on_train_batch_end(batch, {})
        callback.on_epoch_end(0, {"val_loss": 3.25})
        for batch in range(2, 4):
            callback.on_train_batch_end(batch, {})
        callback.on_train_end()
        rows = _rows(tmp_path)
        assert [row["step"] for row in rows] == [0, 2, 4]
        assert rows[-1]["val_loss"] == 3.25 and rows[-1]["val_loss_step"] == 2
        assert all("val_only" not in row for row in rows)

        _, _, _, plain = _live(tmp_path / "never_validated")
        plain.on_train_begin()
        for batch in range(2):
            plain.on_train_batch_end(batch, {})
        plain.on_train_end()
        assert all(
            "val_only" not in row for row in _rows(tmp_path / "never_validated")
        )

    def test_a_validation_that_the_end_draw_already_carries_adds_no_row(self, tmp_path):
        """Off the draw grid the on_train_end draw happens AFTER the validation, so
        its own row holds the newest value."""
        _, _, _, callback = _live(tmp_path)
        _drive_to_a_trailing_validation(callback, batches=3)
        rows = _rows(tmp_path)
        assert [row["step"] for row in rows] == [0, 2, 3]
        assert rows[-1]["val_loss"] == 3.25 and rows[-1]["val_loss_step"] == 3
        assert all("val_only" not in row for row in rows)

    def test_a_restored_redraw_carries_the_validation_and_needs_no_closing_row(
        self, tmp_path
    ):
        """Restore (step 6.2) and trailing validation together: the redraw is
        made after the validation, so it is the last word and nothing is added."""
        _, _, backbone, callback = _live(tmp_path)
        callback.on_train_begin()
        for batch in range(2):
            callback.on_train_batch_end(batch, {})
        callback.on_epoch_end(0, {"val_loss": 3.25})
        for variable in backbone.trainable_variables:
            variable.assign(variable + 0.37)
        callback.on_train_end()
        rows = _rows(tmp_path)
        assert [row["step"] for row in rows] == [0, 2, 2]
        assert rows[-1]["restored_weights"] is True
        assert rows[-1]["val_loss"] == 3.25 and rows[-1]["val_loss_step"] == 2
        assert all("val_only" not in row for row in rows)

    def test_a_restore_whose_redraw_failed_still_gets_the_closing_row(
        self, monkeypatch, tmp_path
    ):
        """The redraw of a restored model fails, the earlier row of that step has
        a stale validation: the closing row still brings the newest one."""
        _, _, backbone, callback = _live(tmp_path)
        callback.on_train_begin()
        for batch in range(2):
            callback.on_train_batch_end(batch, {})
        callback.on_epoch_end(0, {"val_loss": 3.25})
        for variable in backbone.trainable_variables:
            variable.assign(variable + 0.37)
        _spy_on_evaluate(monkeypatch, _raise(RuntimeError("redraw failed")))
        callback.on_train_end()
        rows = _rows(tmp_path)
        assert [row["step"] for row in rows] == [0, 2, 2]
        assert rows[-1]["val_only"] is True and rows[-1]["val_loss"] == 3.25

    def test_a_switched_off_callback_writes_no_closing_row(self, tmp_path):
        _, _, _, callback = _live(tmp_path)
        callback.on_train_begin()
        callback.on_epoch_end(0, {"val_loss": 3.25})
        callback.disabled = True
        callback.on_train_end()
        assert [row["step"] for row in _rows(tmp_path)] == [0]

    def test_a_failure_writing_the_closing_row_is_logged_and_never_raised(
        self, monkeypatch, tmp_path
    ):
        import train.topolm.plotting as plotting

        _, _, _, callback = _live(tmp_path)
        warnings = []
        monkeypatch.setattr(
            logger, "warning", lambda message, *a, **k: warnings.append(str(message))
        )
        callback.on_train_begin()
        for batch in range(2):
            callback.on_train_batch_end(batch, {})
        callback.on_epoch_end(0, {"val_loss": 3.25})

        def boom(*args, **kwargs):
            raise RuntimeError("curve broke")

        monkeypatch.setattr(plotting, "plot_topography_curve", boom)
        callback.on_train_end()                     # must not raise
        assert any("curve broke" in w for w in warnings), warnings

    def test_the_curve_of_a_real_run_ends_on_the_last_validation(self, monkeypatch, tmp_path):
        """End to end on CPU: six steps validated every second step. The closing
        row's value is Keras' own last ``val_loss``, and the curve's val_loss
        panel draws it at the step it was measured (the old curve stopped at 4)."""
        _patch_data(monkeypatch)
        monkeypatch.setattr(C, "_run_dir", lambda config: str(tmp_path / "closing"))
        with keras.device("cpu"):
            result = train_topolm(
                _config(viz_every_steps=2), preprocessor=StubTokenizer(),
                run_topography=False,
            )
        history = result["history"].history["val_loss"]
        assert len(history) == 3
        rows = [
            json.loads(line)
            for line in (tmp_path / "closing" / "topography_live" / "metrics.jsonl")
            .read_text().splitlines()
        ]
        assert [row["step"] for row in rows] == [0, 2, 4, 6, 6]
        assert [row.get("val_only", False) for row in rows] == [False] * 4 + [True]
        assert rows[-1]["val_loss"] == pytest.approx(history[-1])
        assert rows[-1]["val_loss_step"] == 6
        assert rows[3]["val_loss_step"] == 4, "the full row is the one before the validation"

        drawn = _plotted_validation_points(monkeypatch, rows)
        assert len(drawn) == 1
        xs, ys = drawn[0]
        assert xs == [2.0, 4.0, 6.0]
        assert ys == pytest.approx(history)


class TestARepeatedStepIsMerged:
    def test_a_metrics_only_row_does_not_shadow_the_map_statistics(self):
        from train.topolm.plotting import last_row_per_step

        rows = [
            _curve_row(0), _curve_row(4, moran=0.4, val_loss=3.5, val_loss_step=2),
            _val_only(4, 3.1, 4),
        ]
        kept = last_row_per_step(rows)
        assert [row["step"] for row in kept] == [0, 4]
        merged = kept[1]
        assert merged["mean_morans_i"] == 0.4 and merged["per_tap"] == {"t/a": {"morans_i": 0.4}}
        assert merged["val_loss"] == 3.1 and merged["val_loss_step"] == 4
        assert "val_only" not in merged, "the merged row is a full row"

    def test_a_later_null_does_not_erase_an_earlier_value(self):
        from train.topolm.plotting import last_row_per_step

        kept = last_row_per_step([
            _curve_row(4, val_loss=3.5, val_loss_step=2),
            {"step": 4, "val_only": True, "val_loss": None, "val_loss_step": None},
        ])
        assert len(kept) == 1
        assert kept[0]["val_loss"] == 3.5 and kept[0]["val_loss_step"] == 2

    def test_a_lone_metrics_only_row_stays_what_it_is(self):
        from train.topolm.plotting import last_row_per_step

        kept = last_row_per_step([_val_only(4, 3.1, 4)])
        assert kept == [_val_only(4, 3.1, 4)]

    def test_the_restored_weights_row_still_replaces_the_row_before_it(self):
        """A map row describes a draw completely, so a non-finite statistic of the
        restored model (null) must not be filled in from the pre-restore draw."""
        from train.topolm.plotting import last_row_per_step

        restored = _curve_row(4, moran=None, restored_weights=True)
        restored["per_tap"] = {"t/a": {"morans_i": None}}
        kept = last_row_per_step([_curve_row(0), _curve_row(4, moran=0.2), restored])
        assert len(kept) == 2
        assert kept[1]["mean_morans_i"] is None and kept[1]["restored_weights"] is True
        assert kept[1]["per_tap"] == {"t/a": {"morans_i": None}}

    def test_a_restore_followed_by_a_closing_row_keeps_both_semantics(self):
        from train.topolm.plotting import last_row_per_step

        rows = [
            _curve_row(4, moran=0.2),
            _curve_row(4, moran=0.1, restored_weights=True),
            _val_only(4, 3.1, 4),
        ]
        kept = last_row_per_step(rows)
        assert len(kept) == 1
        assert kept[0]["mean_morans_i"] == 0.1 and kept[0]["restored_weights"] is True
        assert kept[0]["val_loss"] == 3.1

    def test_rows_of_distinct_steps_and_stepless_rows_are_untouched(self):
        from train.topolm.plotting import last_row_per_step

        stepless = {"note": "x"}
        rows = [_curve_row(0), stepless, _curve_row(2)]
        assert last_row_per_step(rows) == rows

    def test_the_input_rows_are_not_modified(self):
        from train.topolm.plotting import last_row_per_step
        import copy

        rows = [_curve_row(4, val_loss=3.5, val_loss_step=2), _val_only(4, 3.1, 4)]
        snapshot = copy.deepcopy(rows)
        last_row_per_step(rows)
        assert rows == snapshot

    def test_the_curve_keeps_every_validation_even_when_a_row_is_shadowed(
        self, monkeypatch
    ):
        """Cadence of the validation equal to the cadence of the draw: the step-4
        full row is the only one that carries the validation measured at step 2.
        Merging the closing row over it must not delete that point."""
        rows = [
            _curve_row(0), _curve_row(2),
            _curve_row(4, val_loss=3.5, val_loss_step=2),
            _val_only(4, 3.1, 4),
        ]
        drawn = _plotted_validation_points(monkeypatch, rows)
        assert len(drawn) == 1
        assert drawn[0][0] == [2.0, 4.0]
        assert drawn[0][1] == pytest.approx([3.5, 3.1])

    def test_a_curve_with_a_closing_row_draws_the_statistics_of_the_full_row(
        self, monkeypatch
    ):
        import matplotlib.axes
        from train.topolm.plotting import plot_topography_curve

        drawn = []
        real = matplotlib.axes.Axes.plot

        def spy(self, *args, **kwargs):
            drawn.append((kwargs.get("label"), np.asarray(args[0]), np.asarray(args[1])))
            return real(self, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "plot", spy)
        plot_topography_curve([_curve_row(0, 0.0), _curve_row(4, 0.4), _val_only(4, 3.1, 4)], None)
        mean_line = [d for d in drawn if d[0] == "mean over taps"]
        assert len(mean_line) == 1
        assert mean_line[0][1].tolist() == [0.0, 4.0]
        assert mean_line[0][2].tolist() == pytest.approx([0.0, 0.4])
