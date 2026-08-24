"""Every advisory this tree suppresses on PURPOSE, pinned so it cannot vanish.

Why this file exists
--------------------
``pyproject.toml``'s ``[tool.pytest.ini_options] filterwarnings`` turns every
``UserWarning`` into an error, and the advisories below are then suppressed
**module by module** at the sites that provoke them (a ``pytestmark`` carrying
``pytest.mark.filterwarnings("ignore:...")`` and a comment naming the family).
That arrangement has one failure mode and this file is its only defence: if the
advisory ever stops being emitted -- deleted, its message reworded, its
triggering branch inverted -- every one of those ``ignore`` entries silently
becomes a no-op and NOTHING goes red. A suppression with no paired positive
assertion is a claim nobody checks.

So each advisory gets two arms here:

* a **positive** arm asserting the warning IS raised, with a ``match=`` on the
  exact prefix the ``ignore`` filters key on -- so a reword breaks this file
  before it silently widens the filters; and
* a **control** arm asserting the non-provoking configuration does NOT warn, so
  the positive arm cannot pass by the advisory having become unconditional.

Coverage: **6 of 6** suppressed families, over 30 ``pytestmark`` sites -- zero
suppressions in this tree are unpaired. Two families are emitted by this repo's
own ``src/`` (W-03, W-14). The other four are emitted by Keras and are reached
here through the public API: softmax-over-a-size-1-axis (22 sites),
input-ran-out-of-data (6 sites), model-not-yet-built-on-save (1 site) and
input-structure-mismatch (1 site).

**A new ``ignore:`` mark anywhere under ``tests/`` needs a positive arm here in
the same commit.** That is the whole contract; there is no other instrument.

On not trusting a mark's comment
--------------------------------
Two of the four framework marks were reported as stale ("nothing in the file
provokes them any more") on the strength of stripping the ``pytestmark`` and
seeing a green re-run. That reading was WRONG for both, and the error is
instructive: a suppressed warning leaves no record anywhere, so a green run is
evidence of nothing until you can prove the provoking test actually ran. MEASURED
2026-08-23 with a probe that wraps ``warnings.warn`` itself, ahead of the filter
machinery, so a hit is recorded whether or not a filter would drop it:
``tests/test_train/test_sam/test_train_sam.py`` fires ``ran out of data`` in 5
tests (``TestEveryCLIExpressibleCombinationTrains``) and ``not yet been built``
twice in each of 3 (``TestConfigToConsumptionWiring``). Both marks are load-
bearing; neither was deleted. Confirmed independently by flipping the first
mark's ``ignore:`` to ``error:``, which turned 79 passed into 5 failed.

See ``plans/plan-2026-08-22T035419-a11304c8/decisions.md`` D-252 (R-038 closure)
and ``plans/plan-2026-08-23T091307-9a110062/decisions.md`` D-442.
"""

import warnings

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.initializers.hypersphere_orthogonal_initializer import (
    OrthogonalHypersphereInitializer,
)
from dl_techniques.layers.transformers.transformer import TransformerLayer


# ---------------------------------------------------------------------
# W-03 -- the orthogonality fallback
# `src/dl_techniques/initializers/hypersphere_orthogonal_initializer.py`
# ---------------------------------------------------------------------


class TestTheOrthogonalityFallbackAdvisory:
    """``num_vectors > latent_dim`` is mathematically impossible; we say so."""

    def test_an_infeasible_request_warns(self):
        init = OrthogonalHypersphereInitializer()
        with pytest.warns(
            UserWarning, match=r"Orthogonality constraint violation"
        ):
            out = init(shape=(8, 4))
        # The fallback must still produce the requested geometry -- an advisory
        # that came with a broken tensor would be a defect, not an advisory.
        assert tuple(out.shape) == (8, 4)

    def test_a_feasible_request_does_not_warn(self):
        """The control: without it the positive arm cannot fail."""
        init = OrthogonalHypersphereInitializer()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = init(shape=(4, 8))
        assert tuple(out.shape) == (4, 8)

    def test_the_message_names_both_numbers(self):
        """The ``ignore`` filters key on the prefix; the numbers are the payload."""
        with pytest.warns(UserWarning) as rec:
            OrthogonalHypersphereInitializer()(shape=(9, 3))
        text = str(rec[0].message)
        assert "requesting 9 orthogonal vectors" in text, text
        assert "3-dimensional space" in text, text


# ---------------------------------------------------------------------
# W-14 -- MoE supersedes ffn_type / ffn_args
# `src/dl_techniques/layers/transformers/transformer.py`
# ---------------------------------------------------------------------


def _moe_config():
    return {
        "num_experts": 2,
        "expert_config": {"ffn_config": {"type": "mlp", "hidden_dim": 8}},
    }


class TestTheMoeSupersessionAdvisory:
    """``moe_config`` wins over ``ffn_type``/``ffn_args``, and says so."""

    def test_a_conflicting_ffn_type_warns(self):
        with pytest.warns(UserWarning, match=r"moe_config is provided"):
            TransformerLayer(
                hidden_size=8, num_heads=2, intermediate_size=8,
                ffn_type="swiglu", moe_config=_moe_config(),
            )

    def test_the_default_ffn_type_does_not_warn(self):
        """The control: ``ffn_type='mlp'`` with no ``ffn_args`` is not a conflict."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            TransformerLayer(
                hidden_size=8, num_heads=2, intermediate_size=8,
                moe_config=_moe_config(),
            )

    def test_no_moe_config_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            TransformerLayer(
                hidden_size=8, num_heads=2, intermediate_size=8,
                ffn_type="swiglu",
            )


# ---------------------------------------------------------------------
# The FRAMEWORK advisories, D-301
#
# Everything above is emitted by this repo's own `src/`. Everything below is
# emitted by Keras, and is suppressed at ~30 call sites in `tests/` with a
# per-module `pytestmark`. The failure mode is identical either way -- a
# reworded or deleted framework advisory turns every one of those `ignore:`
# entries into a silent no-op -- but the remedy is not: we cannot pin the
# message at its source, so each arm below provokes the advisory through the
# PUBLIC API the suppressing modules use, and keys `match=` on the exact prefix
# those `ignore:` filters carry.
#
# Each family therefore gets the same two arms as above: a positive one, and a
# control proving the positive one can fail.
# ---------------------------------------------------------------------


class TestTheSoftmaxOverASizeOneAxisAdvisory:
    """`keras/src/ops/nn.py:908`. 22 modules suppress this; all feed a size-1
    axis deliberately (single class, single head, single cluster, minimum
    sequence length), so the advisory describes the test's own input."""

    def test_a_size_one_axis_warns(self):
        with pytest.warns(UserWarning, match=r"You are using a softmax over axis"):
            out = ops.softmax(np.zeros((4, 1), dtype="float32"), axis=-1)
        # The advisory's own claim: the result is identically 1.0. If that ever
        # stops being true the advisory is wrong, not merely reworded.
        assert float(np.max(np.abs(np.asarray(out) - 1.0))) == 0.0

    def test_the_message_names_the_axis_and_the_shape(self):
        """The `ignore:` filters key on the prefix; the axis is the payload."""
        with pytest.warns(UserWarning) as rec:
            ops.softmax(np.zeros((2, 1, 5), dtype="float32"), axis=1)
        text = str(rec[0].message)
        assert "softmax over axis 1" in text, text
        assert "(2, 1, 5)" in text, text

    @pytest.mark.parametrize("axis", [-1, 0, 1])
    def test_a_non_degenerate_axis_does_not_warn(self, axis):
        """The control: the advisory is conditional on `x.shape[axis] == 1`."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = ops.softmax(np.zeros((4, 3), dtype="float32"), axis=axis)
        assert np.all(np.isfinite(np.asarray(out)))


class TestTheInputRanOutOfDataAdvisory:
    """`keras/src/trainers/epoch_iterator.py:81`. Suppressed at 6 sites that run
    a real trainer over a deliberately tiny corpus while `steps_per_epoch` comes
    from the shipped config -- padding the corpus would change what they
    measure."""

    @staticmethod
    def _one_batch_model_and_dataset():
        import tensorflow as tf

        model = keras.Sequential([keras.Input(shape=(2,)), keras.layers.Dense(1)])
        model.compile(optimizer="sgd", loss="mse")
        x = np.zeros((2, 2), dtype="float32")
        y = np.zeros((2, 1), dtype="float32")
        return model, tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

    def test_a_starved_iterator_warns(self):
        model, dataset = self._one_batch_model_and_dataset()
        with pytest.warns(UserWarning, match=r"Your input ran out of data"):
            model.fit(dataset, epochs=1, steps_per_epoch=4, verbose=0)

    def test_the_message_names_the_repeat_remedy(self):
        """The payload the 6 suppressing sites are choosing NOT to apply."""
        model, dataset = self._one_batch_model_and_dataset()
        with pytest.warns(UserWarning) as rec:
            model.fit(dataset, epochs=1, steps_per_epoch=4, verbose=0)
        text = "\n".join(str(w.message) for w in rec)
        assert "`steps_per_epoch * epochs` batches" in text, text
        assert "`.repeat()`" in text, text

    def test_a_repeating_dataset_does_not_warn(self):
        """The control: with `.repeat()` the iterator cannot be exhausted."""
        model, dataset = self._one_batch_model_and_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.fit(dataset.repeat(), epochs=1, steps_per_epoch=4, verbose=0)

    def test_a_dataset_that_covers_the_epoch_does_not_warn(self):
        """The second control: no `steps_per_epoch` override, nothing starves."""
        model, dataset = self._one_batch_model_and_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.fit(dataset, epochs=2, verbose=0)


class TestTheInputStructureMismatchAdvisory:
    """`keras/src/models/functional.py:237`. Suppressed once, in
    `tests/test_models/test_darkir/test_degenerate_middle_and_gate_shape.py`,
    where the probes build throwaway `keras.Model(model.inputs, <intermediate>)`
    views -- `model.inputs` is a one-element LIST -- and call them with a bare
    array. The advisory is about the probe's calling convention."""

    @staticmethod
    def _list_input_model():
        inp = keras.Input(shape=(3,))
        # `[inp]`, not `inp`: this is exactly what `model.inputs` hands back,
        # and it is what makes a bare-array call a structure mismatch.
        return keras.Model([inp], keras.layers.Dense(2)(inp))

    def test_a_bare_array_against_a_list_input_warns(self):
        model = self._list_input_model()
        with pytest.warns(
            UserWarning, match=r"The structure of `inputs` doesn't match"
        ):
            out = model(np.zeros((1, 3), dtype="float32"))
        # Keras warns and then handles it correctly; a broken forward here would
        # make the 1 suppressing site a real defect rather than a convention.
        assert tuple(out.shape) == (1, 2)

    def test_the_message_shows_both_structures(self):
        model = self._list_input_model()
        with pytest.warns(UserWarning) as rec:
            model(np.zeros((1, 3), dtype="float32"))
        text = str(rec[0].message)
        assert "Expected:" in text and "Received:" in text, text

    def test_passing_the_declared_structure_does_not_warn(self):
        """The control: the same model, called with the list it declared."""
        model = self._list_input_model()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = model([np.zeros((1, 3), dtype="float32")])
        assert tuple(out.shape) == (1, 2)

    def test_a_single_tensor_input_model_does_not_warn_on_a_bare_array(self):
        """The second control: the mismatch is the LIST, not the bare array."""
        inp = keras.Input(shape=(3,))
        model = keras.Model(inp, keras.layers.Dense(2)(inp))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            out = model(np.zeros((1, 3), dtype="float32"))
        assert tuple(out.shape) == (1, 2)


class TestTheUnbuiltModelSaveAdvisory:
    """`keras/src/saving/saving_lib.py:103`, guarded by `if not model.built`.
    Suppressed once, in `tests/test_train/test_sam/test_train_sam.py`, whose
    `TestConfigToConsumptionWiring` probes observe the `ModelCheckpoint` CALL
    rather than the archive and so never pay for a full vit_b materialization."""

    @staticmethod
    def _unbuilt_model():
        class Unbuilt(keras.Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(2)

            def call(self, inputs):
                return self.dense(inputs)

        model = Unbuilt()
        assert not model.built, "the probe needs an UNBUILT model to be meaningful"
        return model

    def test_saving_an_unbuilt_model_warns(self, tmp_path):
        model = self._unbuilt_model()
        with pytest.warns(
            UserWarning, match=r"You are saving a model that has not yet been built"
        ):
            model.save(tmp_path / "unbuilt.keras")
        assert (tmp_path / "unbuilt.keras").exists(), (
            "the advisory must not be an error in disguise -- the 1 suppressing "
            "site relies on the save completing."
        )

    def test_the_message_names_the_remedy(self):
        model = self._unbuilt_model()
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            with pytest.warns(UserWarning) as rec:
                model.save(f"{directory}/unbuilt.keras")
        text = "\n".join(str(w.message) for w in rec)
        assert "build" in text.lower(), text

    def test_saving_a_built_model_does_not_warn(self, tmp_path):
        """The control: the advisory is conditional on `not model.built`."""
        model = self._unbuilt_model()
        model(np.zeros((1, 3), dtype="float32"))
        assert model.built
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            model.save(tmp_path / "built.keras")
        assert (tmp_path / "built.keras").exists()
