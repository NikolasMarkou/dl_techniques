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
        assert [row["step"] for row in rows] == [0, 2, 4, 6]

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
