"""Guard: ``WeightEMACallback`` is not a silent no-op, in any of its four ways.

The defect classes this exists to catch
---------------------------------------
1. **A CALLBACK KERAS NEVER INVOKES.** Every arm below drives the callback
   through a REAL ``model.fit()`` over the step-11 pipeline, never by calling
   ``on_train_batch_end`` by hand. A hook-only test cannot see a callback that
   is never wired in -- this repository has measured exactly that failure
   (an appended layer Keras never called).

2. **A SNAPSHOT OF NOTHING.** MEASURED here: Keras 3 calls ``on_train_begin``
   BEFORE the first batch, so a lazily-built model is still unbuilt at that
   point and ``model.trainable_weights`` is ``[]``. An implementation that
   snapshots that empty list averages nothing, forever, and raises nothing.
   ``TestTheSnapshotSurvivesALazilyBuiltModel`` asserts the shadow set is
   non-empty after a fit that never built the model in advance.

3. **A DECAY THAT IS IGNORED.** A two-arm "the shadow changed" test passes
   against an implementation that ignores ``decay`` entirely. The three arms
   here are the ENDPOINTS plus the interior: ``decay=0.0`` must track the live
   weights EXACTLY (``atol=0.0``), ``decay=1.0`` must stay bit-identical to the
   snapshot while the live weights provably move, and ``decay=0.9999`` must have
   moved off the snapshot AND still differ from the live weights -- both halves,
   because either alone is satisfiable by a broken implementation (a shadow
   pinned to the initial weights satisfies the second; a shadow that is just an
   alias of the live weights satisfies the first).

4. **A SWAP WITH NO WAY BACK.** ``apply_to`` must CHANGE the model's outputs
   (otherwise it did nothing) and ``restore`` must return them BIT-identically.

Traps designed out
------------------
* **Every "it moved" arm carries its control.** The ``decay=1.0`` arm asserts
  the live weights moved, or "the shadow did not change" would be trivially
  true against a model that is not training at all.
* **No pasted numbers.** The expectations are derived: exact equality at the
  endpoints (which the two-term update form makes exact), and strict
  inequalities in the interior.
* **The frozen ``pos_embed`` is asserted ABSENT from the shadow set**, together
  with the measurement that motivates the exclusion: the EMA of a constant is
  that constant in exact arithmetic only, and drifts in float32.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Tuple

import numpy as np
import pytest

import keras

from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit.config import DiffusionConfig
from dl_techniques.models.vision_language.dit.model import DiT
from train.dit.ema_callback import (
    DEFAULT_EMA_DECAY,
    SHADOWED_VARIABLE_SET,
    WeightEMACallback,
)
from train.dit.synthetic_data import build_dit_dataset, synthetic_records

SEED = 23
RECORDS = 12
BATCH = 4


# ---------------------------------------------------------------------
# fixtures: a tiny DiT over the step-11 pipeline
# ---------------------------------------------------------------------


@pytest.fixture()
def config() -> DiffusionConfig:
    """8x8x4 latents, 5 classes, a 50-step linear chain."""
    return DiffusionConfig(
        input_size=8,
        in_channels=4,
        num_classes=5,
        num_timesteps=50,
        schedule_name="linear",
    )


def _make_model(config: DiffusionConfig) -> DiT:
    """A DiT small enough to fit on CPU in seconds."""
    keras.utils.set_random_seed(SEED)
    return DiT(
        input_size=config.input_size,
        patch_size=2,
        in_channels=config.in_channels,
        hidden_size=32,
        depth=1,
        num_heads=2,
        num_classes=config.num_classes,
        class_dropout_rate=0.1,
        learn_sigma=True,
    )


def _compiled(config: DiffusionConfig, learning_rate: float = 1e-2) -> DiT:
    model = _make_model(config)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=0.0
        ),
        loss=DDPMHybridLoss(
            schedule_name=config.schedule_name,
            num_timesteps=config.num_timesteps,
            in_channels=config.in_channels,
        ),
    )
    return model


def _dataset(config: DiffusionConfig, steps: int):
    records = synthetic_records(RECORDS, config, seed=SEED)
    return build_dit_dataset(
        records, config, batch_size=BATCH, seed=SEED, steps=steps
    )


def _run(
    config: DiffusionConfig,
    decay: float,
    steps: int,
    prebuild: bool = True,
) -> Tuple[DiT, WeightEMACallback, Dict[str, np.ndarray]]:
    """Fit for ``steps`` batches with the callback attached.

    :return: ``(model, callback, weights_before_fit)`` -- the third element is
        empty when ``prebuild`` is False (the model does not exist yet).
    """
    model = _compiled(config)
    before: Dict[str, np.ndarray] = {}
    if prebuild:
        model.build(_input_shapes(config))
        before = _live(model)
    ema = WeightEMACallback(decay=decay)
    model.fit(
        _dataset(config, steps),
        epochs=1,
        steps_per_epoch=steps,
        verbose=0,
        callbacks=[ema],
    )
    return model, ema, before


def _input_shapes(config: DiffusionConfig) -> List[Tuple]:
    size, channels = config.input_size, config.in_channels
    return [(None, size, size, channels), (None,), (None,)]


def _live(model: keras.Model) -> Dict[str, np.ndarray]:
    """The model's trainable weights as NumPy, keyed by path."""
    return {
        v.path: np.array(keras.ops.convert_to_numpy(v), copy=True)
        for v in model.trainable_weights
    }


def _max_abs_diff(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    assert set(a) == set(b)
    return max(float(np.max(np.abs(a[key] - b[key]))) for key in a)


# ---------------------------------------------------------------------
# it is actually invoked, and it snapshots something
# ---------------------------------------------------------------------


class TestKerasActuallyInvokesIt:
    def test_the_update_count_equals_the_number_of_batches(self, config):
        steps = 5
        _, ema, _ = _run(config, decay=0.9, steps=steps)
        # The model was built before fit(), so on_train_begin snapshotted and
        # every batch end updated: one update per batch, none deferred.
        assert ema.updates == steps
        assert ema.initialized


class TestTheSnapshotSurvivesALazilyBuiltModel:
    def test_on_train_begin_sees_an_unbuilt_model(self, config):
        # MEASURED: this is what Keras hands the callback. The arm exists so
        # the deferral in the implementation is justified by a fact, not a
        # defensive habit.
        model = _compiled(config)
        seen = {}

        class Probe(keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                seen["n"] = len(self.model.trainable_weights)

        model.fit(
            _dataset(config, 1), epochs=1, steps_per_epoch=1, verbose=0,
            callbacks=[Probe()],
        )
        assert seen["n"] == 0
        assert len(model.trainable_weights) > 0

    def test_the_shadow_set_is_not_empty_without_a_prebuild(self, config):
        _, ema, _ = _run(config, decay=0.9, steps=3, prebuild=False)
        assert ema.initialized
        assert len(ema.shadow_values()) > 0
        # One batch was spent taking the deferred snapshot.
        assert ema.updates == 2


# ---------------------------------------------------------------------
# the three decay arms
# ---------------------------------------------------------------------


class TestDecayZeroTracksTheLiveWeightsExactly:
    def test_after_one_batch_the_shadow_equals_the_weights(self, config):
        model, ema, _ = _run(config, decay=0.0, steps=1)
        shadows = ema.shadow_values()
        live = _live(model)
        assert set(shadows) == set(live)
        for key in shadows:
            np.testing.assert_array_equal(shadows[key], live[key])

    def test_it_still_tracks_after_several_batches(self, config):
        model, ema, _ = _run(config, decay=0.0, steps=4)
        assert _max_abs_diff(ema.shadow_values(), _live(model)) == 0.0


class TestDecayOneFreezesTheSnapshot:
    def test_the_shadow_is_bit_unchanged_while_the_weights_move(self, config):
        steps = 5
        model, ema, before = _run(config, decay=1.0, steps=steps)
        live = _live(model)

        # Control first: without this the "unchanged" claim is vacuous against
        # a model that never trained.
        assert _max_abs_diff(before, live) > 0.0

        shadows = ema.shadow_values()
        for key in shadows:
            np.testing.assert_array_equal(shadows[key], before[key])
        assert ema.updates == steps


class TestDecayNineNinesMovesButLags:
    def test_the_shadow_moved_off_its_snapshot_and_differs_from_the_weights(
        self, config
    ):
        steps = 8
        model, ema, before = _run(config, decay=DEFAULT_EMA_DECAY, steps=steps)
        shadows = ema.shadow_values()
        live = _live(model)

        moved = _max_abs_diff(shadows, before)
        lag = _max_abs_diff(shadows, live)
        weight_travel = _max_abs_diff(before, live)

        # Half one: it is not frozen.
        assert moved > 0.0
        # Half two: it is not an alias of the live weights.
        assert lag > 0.0
        # And it lags: at decay 0.9999 the shadow has covered a tiny fraction
        # of the distance the weights travelled. Derived, not pasted --
        # 1 - decay**steps is the total weight a geometric EMA has given to the
        # post-snapshot weights.
        assert moved < weight_travel
        assert moved <= (1.0 - DEFAULT_EMA_DECAY ** steps) * weight_travel * 1.5

    def test_the_default_decay_is_upstreams(self):
        assert DEFAULT_EMA_DECAY == 0.9999
        assert WeightEMACallback().decay == DEFAULT_EMA_DECAY


# ---------------------------------------------------------------------
# the swap, and the way back
# ---------------------------------------------------------------------


def _predict(model: DiT, config: DiffusionConfig) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    size, channels = config.input_size, config.in_channels
    x = rng.normal(size=(2, size, size, channels)).astype("float32")
    t = np.array([3, 11], dtype="int32")
    y = np.array([0, 2], dtype="int32")
    return np.array(
        keras.ops.convert_to_numpy(model([x, t, y], training=False)), copy=True
    )


class TestApplyToAndRestore:
    def test_applying_changes_the_outputs_and_restoring_undoes_it(self, config):
        model, ema, _ = _run(config, decay=0.5, steps=4)
        base = _predict(model, config)
        live = _live(model)

        ema.apply_to(model)
        assert ema.applied
        applied = _predict(model, config)
        # The weights ARE the shadows now.
        assert _max_abs_diff(_live(model), ema.shadow_values()) == 0.0
        assert float(np.max(np.abs(applied - base))) > 0.0

        ema.restore(model)
        assert not ema.applied
        assert _max_abs_diff(_live(model), live) == 0.0
        np.testing.assert_array_equal(_predict(model, config), base)

    def test_the_context_manager_restores_on_an_exception(self, config):
        model, ema, _ = _run(config, decay=0.5, steps=3)
        live = _live(model)

        with contextlib.suppress(RuntimeError):
            with ema.applied_to(model):
                assert _max_abs_diff(_live(model), ema.shadow_values()) == 0.0
                raise RuntimeError("boom")

        assert not ema.applied
        assert _max_abs_diff(_live(model), live) == 0.0

    def test_applying_before_any_fit_raises(self, config):
        model = _compiled(config)
        model.build(_input_shapes(config))
        ema = WeightEMACallback()
        with pytest.raises(RuntimeError, match="no shadows yet"):
            ema.apply_to(model)

    def test_double_apply_raises(self, config):
        model, ema, _ = _run(config, decay=0.5, steps=2)
        ema.apply_to(model)
        try:
            with pytest.raises(RuntimeError, match="already applied"):
                ema.apply_to(model)
        finally:
            ema.restore(model)

    def test_restore_without_apply_raises(self, config):
        model, ema, _ = _run(config, decay=0.5, steps=2)
        with pytest.raises(RuntimeError, match="nothing to restore"):
            ema.restore(model)

    def test_a_differently_shaped_model_is_rejected_not_half_written(
        self, config
    ):
        model, ema, _ = _run(config, decay=0.5, steps=2)
        wider = DiT(
            input_size=config.input_size,
            patch_size=2,
            in_channels=config.in_channels,
            hidden_size=64,
            depth=1,
            num_heads=2,
            num_classes=config.num_classes,
            learn_sigma=True,
        )
        wider.build(_input_shapes(config))
        untouched = _live(wider)
        # The rejection message names WHICH mismatch it found; the claim this
        # arm makes is that the model is rejected WHOLE, not half-written.
        with pytest.raises(
            ValueError, match="absent from this model|shape mismatch"
        ):
            ema.apply_to(wider)
        assert not ema.applied
        assert _max_abs_diff(_live(wider), untouched) == 0.0


# ---------------------------------------------------------------------
# which variables are shadowed (D-023)
# ---------------------------------------------------------------------


class TestTheShadowedSetIsTrainableOnly:
    def test_the_shadow_paths_are_exactly_the_trainable_paths(self, config):
        model, ema, _ = _run(config, decay=0.5, steps=1)
        assert SHADOWED_VARIABLE_SET == "trainable_weights"
        assert set(ema.shadow_values()) == {
            v.path for v in model.trainable_weights
        }

    def test_the_frozen_pos_embed_is_not_shadowed(self, config):
        model, ema, _ = _run(config, decay=0.5, steps=1)
        frozen = [v.path for v in model.non_trainable_weights]
        assert any("pos_embed" in path for path in frozen), frozen
        for path in frozen:
            assert path not in ema.shadow_values()

    def test_the_frozen_table_is_untouched_by_apply_and_restore(self, config):
        model, ema, _ = _run(config, decay=0.5, steps=2)
        frozen_before = {
            v.path: np.array(keras.ops.convert_to_numpy(v), copy=True)
            for v in model.non_trainable_weights
        }
        with ema.applied_to(model):
            for v in model.non_trainable_weights:
                np.testing.assert_array_equal(
                    keras.ops.convert_to_numpy(v), frozen_before[v.path]
                )

    def test_the_ema_of_a_constant_is_the_constant_only_in_exact_arithmetic(
        self,
    ):
        # The MEASUREMENT behind D-023. Widening the shadow set to
        # `model.weights` looks free because `d*c + (1-d)*c == c`; in float32
        # it is not. This arm is executable so nobody re-derives the false
        # claim from the algebra.
        rng = np.random.default_rng(SEED)
        constant = rng.normal(size=(64, 64)).astype("float32")
        shadow = constant.copy()
        for _ in range(200):
            shadow = (
                np.float32(DEFAULT_EMA_DECAY) * shadow
                + np.float32(1.0 - DEFAULT_EMA_DECAY) * constant
            ).astype("float32")
        assert float(np.max(np.abs(shadow - constant))) > 0.0


# ---------------------------------------------------------------------
# construction contract
# ---------------------------------------------------------------------


class TestTheDecayIsValidated:
    @pytest.mark.parametrize("decay", [-0.1, 1.5, float("nan"), float("inf")])
    def test_an_illegal_decay_raises(self, decay):
        with pytest.raises(ValueError, match=r"decay must lie in \[0, 1\]"):
            WeightEMACallback(decay=decay)

    @pytest.mark.parametrize("decay", [0.0, 0.5, 1.0])
    def test_the_endpoints_are_legal(self, decay):
        assert WeightEMACallback(decay=decay).decay == decay

    def test_a_fresh_callback_is_uninitialized(self):
        ema = WeightEMACallback()
        assert not ema.initialized
        assert not ema.applied
        assert ema.updates == 0
        assert ema.shadow_values() == {}
