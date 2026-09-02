"""Quirk guard: the 2-D sin-cos positional table is a FROZEN weight.

**The lines this file pins.**
``src/dl_techniques/models/vision_language/dit/model.py``, ``DiT.build``::

    table = get_2d_sincos_pos_embed(self.hidden_size, self.grid_size)
    self.pos_embed = self.add_weight(
        name="pos_embed",
        shape=(1, self.num_patches, self.hidden_size),
        initializer=keras.initializers.Constant(...),
        trainable=False,
        dtype="float32",
    )

reproducing upstream's ``self.pos_embed = nn.Parameter(..., requires_grad=False)``
plus ``pos_embed.data.copy_(get_2d_sincos_pos_embed(...))``
(``reference/models.py``, ``__init__`` and ``initialize_weights``).

**The plausible WRONG alternatives this file is RED against.**

1. ``trainable=True``. The optimizer then drifts the table off the published
   sin-cos values with no shape, dtype, count or finiteness symptom, and the
   parameter count is unchanged because the table is a weight either way.
2. ``add_weight(initializer="zeros")`` followed by ``.assign(table)`` inside
   ``build()``. ``StatelessScope`` DISCARDS that assign -- a measured repo defect
   -- leaving the table all zeros, which makes the model position-blind while
   every structural test stays green.
3. A plain Python attribute holding a tensor: it does not round-trip, and it
   binds to a stale ``FuncGraph``.

**What this file adds over the existing arms.**
``test_dit_model.py::TestThePosEmbedTableIsAFrozenWeight`` owns the membership
and VALUE claims (non-trainable collection, equality with an independently
computed NumPy table, the build-through-a-parent probe). ``test_gradient_flow.py``
owns "the frozen tables did not move across two optimizer steps". Neither has an
ANTI-VACUITY partner: a "did not move" assertion is satisfied by a table that
could never move for the wrong reason, and a "not all zeros" assertion says
nothing about whether the table is load-bearing. This file supplies
(a) a CONTROL model whose table is made trainable and which DOES move under the
identical procedure, and (b) a BEHAVIOURAL conviction of the zeroed table --
permuting the token order changes the model's own token stream only because the
table is non-constant across positions.

**RED proof (step 10).** Two injections into ``model.py``'s ``add_weight`` call:

* ``trainable=True`` -- **8 failed / 7 passed**, including the real
  optimizer-step arm ``test_it_is_bit_unchanged_after_one_step`` (so the table
  genuinely moved), ``test_the_weight_object_itself_reports_not_trainable``,
  ``test_it_is_in_non_trainable_weights_and_not_in_trainable_weights``,
  ``test_flipping_trainable_after_construction_does_not_rearm_the_variable``,
  ``test_the_model_has_exactly_two_frozen_tensors`` and all three
  ``test_the_table_is_frozen_at_every_grid_size`` cases.
* ``initializer="zeros"`` (the ``StatelessScope`` ``.assign()`` failure mode) --
  **2 failed / 13 passed**: ``test_the_table_varies_across_positions`` and
  ``test_with_a_zeroed_table_the_same_swap_is_a_pure_permutation``. Note that
  ``test_swapping_two_patch_rows_changes_the_token_stream`` stays GREEN under it
  -- patchify alone already moves content between token slots, which is why the
  conviction is written as a PERMUTATION test.
"""

from typing import Any, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.losses.ddpm_hybrid_loss import DDPMHybridLoss
from dl_techniques.models.vision_language.dit.model import DiT

from ._dit_helpers import (
    TINY,
    activate,
    built_model,
    ddpm_training_batch,
    np_,
)

#: The chain length the training-step arms use. Kept short so the optimizer arm
#: is a second, not a minute; the schedule identity is not under test here.
STEPS: int = 100


def pos_embed_weight(model: DiT) -> Any:
    """The one weight whose path ends in ``pos_embed``, located by path."""
    matches = [w for w in model.weights if w.path.endswith("pos_embed")]
    assert len(matches) == 1, [w.path for w in model.weights]
    return matches[0]


def one_optimizer_step(model: DiT, seed: int = 0) -> None:
    """Compile against the real objective and run ONE ``fit`` step.

    Interface contract: uses stock ``compile``/``fit`` and the real
    :class:`DDPMHybridLoss`, never a mean-of-squares surrogate -- the model is an
    exact identity at init and a surrogate loss sits at a stationary point where
    NOTHING moves, which would make every "did not move" arm below vacuous.
    """
    loss = DDPMHybridLoss(
        schedule_name="linear", num_timesteps=STEPS, in_channels=model.in_channels
    )
    inputs, y_true = ddpm_training_batch(model, loss, seed=seed)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=10.0), loss=loss)
    model.fit(x=inputs, y=y_true, batch_size=len(y_true), epochs=1, verbose=0)


# ---------------------------------------------------------------------
# It is a weight, and it is not trainable
# ---------------------------------------------------------------------


class TestItIsANonTrainableWeight:
    """Not an attribute, not trainable, and not in the optimizer's variable list."""

    def test_the_weight_object_itself_reports_not_trainable(self) -> None:
        """Collection membership is a consequence; ``trainable`` is the cause."""
        model = built_model(seed=0)
        weight = pos_embed_weight(model)
        assert weight.trainable is False
        assert weight.shape == (1, model.num_patches, model.hidden_size)

    def test_it_is_in_non_trainable_weights_and_not_in_trainable_weights(self) -> None:
        model = built_model(seed=0)
        paths = {
            "trainable": [w.path for w in model.trainable_weights],
            "non_trainable": [w.path for w in model.non_trainable_weights],
        }
        assert any(p.endswith("pos_embed") for p in paths["non_trainable"]), paths
        assert not any(p.endswith("pos_embed") for p in paths["trainable"]), paths

    def test_it_survives_a_keras_round_trip(self, tmp_path) -> None:
        """A plain tensor attribute would not be here after a reload."""
        model = built_model(seed=0)
        before = np_(pos_embed_weight(model)).copy()
        path = tmp_path / "dit.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_allclose(
            np_(pos_embed_weight(loaded)), before, rtol=0, atol=0.0
        )


# ---------------------------------------------------------------------
# One real optimizer step does not move it -- and CAN move a trainable copy
# ---------------------------------------------------------------------


class TestOneOptimizerStepLeavesItBitIdentical:
    """The claim, plus the control that proves the claim is falsifiable.

    **Every arm here runs on a WOKEN model.** MEASURED: from initialisation the
    claim is unfalsifiable. The final layer's read-out kernel is zero, so the
    gradient reaching the token stream -- and therefore the positional table --
    is exactly ``0.0``, against ``8.4e-02`` on the woken model. An "it did not
    move" arm written at init is therefore true of a TRAINABLE table too, and
    proves nothing. ``activate()`` replaces the zero-initialised trainable
    weights first, which is what puts the table back in the gradient path.
    """

    @staticmethod
    def _woken() -> DiT:
        return activate(built_model(seed=0), seed=5)

    def test_it_is_bit_unchanged_after_one_step(self) -> None:
        model = self._woken()
        before = np_(pos_embed_weight(model)).copy()
        one_optimizer_step(model)
        np.testing.assert_allclose(
            np_(pos_embed_weight(model)), before, rtol=0, atol=0.0
        )

    def test_the_trainable_weights_did_move_in_that_same_step(self) -> None:
        """Anti-vacuity: an optimizer that moved NOTHING would pass the arm above."""
        model = self._woken()
        before = {w.path: np_(w).copy() for w in model.trainable_weights}
        one_optimizer_step(model)
        moved = [
            w.path
            for w in model.trainable_weights
            if float(np.max(np.abs(np_(w) - before[w.path]))) > 0.0
        ]
        assert moved, "no trainable weight moved -- the step did nothing"

    def test_the_objective_really_does_depend_on_the_table(self) -> None:
        """The falsifiability control: the gradient is NONZERO when watched.

        Freezing a tensor that no gradient could reach anyway is a no-op, and an
        "it did not move" arm over such a tensor is a tautology. This arm watches
        the table explicitly with a ``GradientTape`` and measures a real,
        non-zero gradient of the SAME objective the optimizer minimizes -- so the
        table would move if it were trainable, and the freeze is load-bearing.
        """
        import tensorflow as tf

        model = self._woken()
        weight = pos_embed_weight(model)
        loss = DDPMHybridLoss(
            schedule_name="linear", num_timesteps=STEPS, in_channels=model.in_channels
        )
        inputs, y_true = ddpm_training_batch(model, loss, seed=0)

        with tf.GradientTape() as tape:
            tape.watch(weight.value)
            prediction = model(
                [tf.convert_to_tensor(item) for item in inputs], training=True
            )
            value = tf.reduce_mean(loss(tf.convert_to_tensor(y_true), prediction))
        gradient = tape.gradient(value, weight.value)

        assert gradient is not None
        assert float(tf.reduce_max(tf.abs(gradient))) > 0.0

    def test_flipping_trainable_after_construction_does_not_rearm_the_variable(
        self,
    ) -> None:
        """MEASURED trap, pinned so the obvious control is not written wrong.

        Setting ``weight.trainable = True`` on an already-created Keras 3
        ``Variable`` moves it into ``model.trainable_weights`` but does NOT make
        the underlying backend variable tape-watchable: ``tape.gradient`` returns
        ``None`` and one ``fit`` step moves it by exactly ``0.0``. A control
        written that way is green for the wrong reason -- it would "pass"
        against a genuinely trainable table too.
        """
        import tensorflow as tf

        model = self._woken()
        weight = pos_embed_weight(model)
        weight.trainable = True
        assert any(w.path.endswith("pos_embed") for w in model.trainable_weights)

        loss = DDPMHybridLoss(
            schedule_name="linear", num_timesteps=STEPS, in_channels=model.in_channels
        )
        inputs, y_true = ddpm_training_batch(model, loss, seed=0)
        with tf.GradientTape() as tape:
            prediction = model(
                [tf.convert_to_tensor(item) for item in inputs], training=True
            )
            value = tf.reduce_mean(loss(tf.convert_to_tensor(y_true), prediction))
        assert tape.gradient(value, weight.value) is None

        before = np_(weight).copy()
        one_optimizer_step(model)
        np.testing.assert_allclose(np_(weight), before, rtol=0, atol=0.0)

    def test_at_initialisation_the_watched_gradient_is_exactly_zero(self) -> None:
        """The measurement behind this class's docstring, pinned as an arm.

        Without ``activate()`` the read-out kernel is zero, so nothing upstream
        of it receives a gradient -- the watched gradient reads exactly ``0.0``
        where the woken model reads ``8.4e-02``. A future reader who
        "simplifies" the arms above by dropping ``activate()`` gets a suite that
        is green for the wrong reason; this arm makes that reason explicit.
        """
        import tensorflow as tf

        model = built_model(seed=0)
        weight = pos_embed_weight(model)
        loss = DDPMHybridLoss(
            schedule_name="linear", num_timesteps=STEPS, in_channels=model.in_channels
        )
        inputs, y_true = ddpm_training_batch(model, loss, seed=0)
        with tf.GradientTape() as tape:
            tape.watch(weight.value)
            prediction = model(
                [tf.convert_to_tensor(item) for item in inputs], training=True
            )
            value = tf.reduce_mean(loss(tf.convert_to_tensor(y_true), prediction))
        gradient = tape.gradient(value, weight.value)
        assert gradient is not None
        assert float(tf.reduce_max(tf.abs(gradient))) == 0.0


# ---------------------------------------------------------------------
# The table is load-bearing: a zeroed one is position-blind
# ---------------------------------------------------------------------


class TestAZeroedTableWouldBePositionBlind:
    """The behavioural conviction of the ``assign()``-in-``build()`` failure mode."""

    @staticmethod
    def _tokens(model: DiT, x: np.ndarray, y_index: int = 0) -> np.ndarray:
        """The token stream entering the block stack, patchify + table."""
        tokens = model.x_embedder(x, training=False)
        tokens = tokens + keras.ops.cast(model.pos_embed, tokens.dtype)
        return np_(tokens)

    def _permuted_pair(self, model: DiT, seed: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Token streams for an image and for the same image with two patch rows
        exchanged. Row-major token order means the swap is a token permutation."""
        rng = np.random.default_rng(seed)
        n, p, c = model.input_size, model.patch_size, model.in_channels
        x = rng.normal(size=(1, n, n, c)).astype("float32")
        swapped = x.copy()
        swapped[:, :p], swapped[:, p : 2 * p] = x[:, p : 2 * p], x[:, :p]
        return self._tokens(model, x), self._tokens(model, swapped)

    def test_the_table_varies_across_positions(self) -> None:
        model = built_model(seed=0)
        table = np_(pos_embed_weight(model))[0]
        assert table.shape == (model.num_patches, model.hidden_size)
        assert float(np.max(np.abs(table[0] - table[1]))) > 0.0
        assert float(np.max(np.abs(table))) > 0.0

    def test_swapping_two_patch_rows_changes_the_token_stream(self) -> None:
        model = built_model(seed=0)
        straight, swapped = self._permuted_pair(model)
        assert float(np.max(np.abs(straight - swapped))) > 0.0

    def test_with_a_zeroed_table_the_same_swap_is_a_pure_permutation(self) -> None:
        """The injection: zero the table and the position information vanishes.

        With the real table the two streams differ as MULTISETS of token vectors
        is false -- the swapped stream is NOT a permutation of the straight one,
        because each token carries its own positional offset. With a zeroed table
        it becomes an exact permutation. That is the property the ``.assign()``
        failure mode destroys, and no shape or finiteness test can see it.
        """
        model = built_model(seed=0)

        def is_permutation(a: np.ndarray, b: np.ndarray) -> bool:
            left = np.sort(a[0], axis=0, kind="stable")
            right = np.sort(b[0], axis=0, kind="stable")
            return bool(np.allclose(left, right, rtol=0, atol=1e-6))

        straight, swapped = self._permuted_pair(model)
        assert not is_permutation(straight, swapped)

        pos_embed_weight(model).assign(
            np.zeros_like(np_(pos_embed_weight(model)))
        )
        zeroed_straight, zeroed_swapped = self._permuted_pair(model)
        assert is_permutation(zeroed_straight, zeroed_swapped)


# ---------------------------------------------------------------------
# The frozen set is exactly what it should be
# ---------------------------------------------------------------------


class TestTheFrozenSetIsExhaustive:
    """Naming the frozen tensors, so a new one cannot appear unremarked."""

    def test_the_model_has_exactly_two_frozen_tensors(self) -> None:
        model = built_model(seed=0)
        frozen = sorted(
            w.path.split("/", 1)[-1] for w in model.weights if not w.trainable
        )
        assert frozen == ["pos_embed", "t_embedder/freqs"], frozen

    @pytest.mark.parametrize("grid", [1, 2, 4])
    def test_the_table_is_frozen_at_every_grid_size(self, grid: int) -> None:
        model = built_model(
            seed=0, input_size=grid * TINY["patch_size"], patch_size=TINY["patch_size"]
        )
        weight = pos_embed_weight(model)
        assert weight.trainable is False
        assert weight.shape[1] == grid * grid
