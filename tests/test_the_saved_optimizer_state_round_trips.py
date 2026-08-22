"""A ``.keras`` archive must not claim an optimizer state it cannot restore.

RD-2 of the R-038 sweep (plan-2026-08-22T035419-a11304c8): 34 node ids across 13
run-groups saved a model whose optimizer had never been built, so
``keras.models.load_model`` warned and silently skipped the optimizer restore.
This module pins BOTH halves of that repair:

* the **treatment** -- ``tests.optimizer_state.build_optimizer_state`` makes the
  saved and reloaded optimizers agree, and the reload is warning-free;
* the **control** -- without it the warning DOES fire. Without this arm the
  treatment assertion is vacuous: a model that Keras happened to reload cleanly
  anyway would pass it.

The control is what makes the 24 call sites added by that step load-bearing
rather than decorative.
"""

import os
import warnings
from typing import List, Tuple

import keras
import numpy as np
import pytest

from tests.optimizer_state import build_optimizer_state

_OPTIMIZER_SKIP_TEXT = "Skipping variable loading for optimizer"


def _tiny_compiled_model() -> keras.Model:
    """A built, compiled, never-fitted model -- the exact RD-2 shape."""
    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=(6,))
    x = keras.layers.Dense(8, activation="relu")(inputs)
    outputs = keras.layers.Dense(2)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def _save_reload(model: keras.Model, path: str) -> Tuple[keras.Model, List[str]]:
    """Save then reload, returning the reloaded model and the warnings raised."""
    model.save(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reloaded = keras.models.load_model(path)
    return reloaded, [str(w.message) for w in caught]


class TestUntrainedCompiledModel:

    def test_the_control_arm_still_warns_without_the_helper(self, tmp_path) -> None:
        """RED proof. Delete ``build_optimizer_state`` from a call site and THIS
        is the shape of what comes back."""
        model = _tiny_compiled_model()
        assert not model.optimizer.built
        assert len(model.optimizer.variables) == 2

        reloaded, messages = _save_reload(model, os.path.join(tmp_path, "control.keras"))

        assert len(reloaded.optimizer.variables) == 2 + 2 * len(model.trainable_variables)
        assert any(_OPTIMIZER_SKIP_TEXT in m for m in messages), messages

    def test_the_helper_makes_the_archive_self_consistent(self, tmp_path) -> None:
        model = _tiny_compiled_model()
        n_after_build = build_optimizer_state(model)

        assert model.optimizer.built
        assert n_after_build == 2 + 2 * len(model.trainable_variables)

        reloaded, messages = _save_reload(model, os.path.join(tmp_path, "treated.keras"))

        assert len(reloaded.optimizer.variables) == n_after_build
        assert not any(_OPTIMIZER_SKIP_TEXT in m for m in messages), messages

    def test_the_helper_is_idempotent(self, tmp_path) -> None:
        model = _tiny_compiled_model()
        first = build_optimizer_state(model)
        second = build_optimizer_state(model)
        assert first == second

    def test_a_model_with_no_optimizer_is_left_alone(self) -> None:
        keras.utils.set_random_seed(0)
        inputs = keras.Input(shape=(6,))
        model = keras.Model(inputs, keras.layers.Dense(2)(inputs))
        assert build_optimizer_state(model) is None


class TestTrainedModelValuesSurviveTheRoundTrip:
    """The counts matching is necessary, not sufficient -- pin the VALUES."""

    def test_optimizer_slot_values_are_restored_exactly(self, tmp_path) -> None:
        model = _tiny_compiled_model()
        x = np.random.RandomState(0).rand(8, 6).astype("float32")
        y = np.random.RandomState(1).rand(8, 2).astype("float32")
        model.train_on_batch(x, y)

        before = [keras.ops.convert_to_numpy(v) for v in model.optimizer.variables]
        # A trained optimizer holds non-zero moments; an all-zero "before" would
        # make the comparison below satisfiable by a fresh optimizer.
        assert max(float(np.max(np.abs(v))) for v in before) > 0.0

        reloaded, messages = _save_reload(model, os.path.join(tmp_path, "trained.keras"))
        after = [keras.ops.convert_to_numpy(v) for v in reloaded.optimizer.variables]

        assert not any(_OPTIMIZER_SKIP_TEXT in m for m in messages), messages
        assert len(after) == len(before)
        for i, (b, a) in enumerate(zip(before, after)):
            assert float(np.max(np.abs(b - a))) == 0.0, i


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "sgd", "rmsprop"])
def test_the_helper_works_for_every_optimizer_the_repo_compiles_with(
    optimizer_name: str, tmp_path
) -> None:
    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=(6,))
    model = keras.Model(inputs, keras.layers.Dense(2)(inputs))
    model.compile(optimizer=optimizer_name, loss="mse")
    expected = build_optimizer_state(model)

    reloaded, messages = _save_reload(model, os.path.join(tmp_path, f"{optimizer_name}.keras"))

    assert len(reloaded.optimizer.variables) == expected
    assert not any(_OPTIMIZER_SKIP_TEXT in m for m in messages), messages
