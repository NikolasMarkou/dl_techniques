"""C-44(a): the compiled loss must agree with the model's output activation.

``create_power_mlp`` defaulted to ``loss="categorical_crossentropy"`` while
``PowerMLP`` defaults to ``output_activation=None`` (linear). A string loss
compiles with ``from_logits=False``, so **following the function's own
documented example** fed unnormalized real-valued outputs to a cross-entropy
that renormalizes by ``output / sum(output)`` and clips. With mixed-sign
activations the denominator can approach zero and negatives clip to
``epsilon``: finite, meaningless, no error. ``test_create_power_mlp`` asserted
only that an object came back.

The pair is what is asserted here. Either half alone is satisfiable by the
defect: a linear output is fine, and a ``from_logits=False`` cross-entropy is
fine -- together they are wrong.

Also covered: ``from_variant`` did ``config.update(kwargs)`` and then
unconditionally overwrote ``config["hidden_units"]``, silently discarding a
caller-supplied value.
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.power_mlp.model import (
    PowerMLP,
    create_power_mlp,
    create_power_mlp_regressor,
    create_power_mlp_binary_classifier,
)


def _emits_probabilities(model: PowerMLP) -> bool:
    return model.output_activation in (
        keras.activations.softmax, keras.activations.sigmoid)


class TestTheDefaultLossMatchesTheDefaultOutput:

    def test_the_documented_example_is_self_consistent(self):
        """The example passes neither `loss` nor `output_activation`."""
        model = create_power_mlp(hidden_units=[784, 128, 64, 10], k=3)

        assert isinstance(model.loss, keras.losses.CategoricalCrossentropy)
        assert model.loss.from_logits is True, (
            "ASSERT-LOSS-MATCHES-OUTPUT: the default output is LINEAR, so the "
            "default loss must read its inputs as logits."
        )
        assert not _emits_probabilities(model)

    def test_a_softmax_output_gets_a_probability_loss(self):
        model = create_power_mlp(
            hidden_units=[20, 16, 5], output_activation="softmax")

        assert _emits_probabilities(model)
        assert model.loss.from_logits is False, (
            "ASSERT-LOSS-MATCHES-OUTPUT: a softmax output must NOT be read as "
            "logits."
        )

    @pytest.mark.parametrize("activation", [None, "softmax"])
    def test_the_pair_agrees_for_every_default_path(self, activation):
        """The single invariant, stated once: loss direction tracks output."""
        kwargs = {} if activation is None else {"output_activation": activation}
        model = create_power_mlp(hidden_units=[20, 16, 5], **kwargs)

        assert model.loss.from_logits is not _emits_probabilities(model), (
            f"ASSERT-PAIR-AGREES: output_activation={activation!r} paired with "
            f"from_logits={model.loss.from_logits}"
        )

    def test_an_explicit_loss_still_wins(self):
        """Anti-vacuity: the override path is untouched."""
        model = create_power_mlp(
            hidden_units=[20, 16, 5], loss="sparse_categorical_crossentropy")
        assert model.loss == "sparse_categorical_crossentropy"

    def test_the_sibling_factories_are_unaffected(self):
        """Both pass `loss=` explicitly, so the new default cannot reach them."""
        regressor = create_power_mlp_regressor(hidden_units=[10, 8, 1])
        assert regressor.loss == "mse"
        assert regressor.output_activation is keras.activations.linear

        binary = create_power_mlp_binary_classifier(hidden_units=[10, 8, 1])
        assert binary.loss == "binary_crossentropy"
        assert binary.output_activation is keras.activations.sigmoid

    def test_the_default_model_actually_trains(self):
        """End to end: the derived loss must be usable, not just correct."""
        model = create_power_mlp(hidden_units=[8, 8, 3])
        x = np.random.normal(size=(16, 8)).astype("float32")
        y = keras.utils.to_categorical(
            np.random.randint(0, 3, size=(16,)), num_classes=3)

        history = model.fit(x, y, epochs=2, verbose=0)
        assert np.all(np.isfinite(history.history["loss"]))


class TestFromVariantDoesNotSilentlyDiscardHiddenUnits:

    def test_a_supplied_hidden_units_is_refused_by_name(self):
        with pytest.raises(ValueError) as excinfo:
            PowerMLP.from_variant(
                "small", num_classes=10, input_dim=784,
                hidden_units=[999, 999],
            )
        assert "hidden_units" in str(excinfo.value), (
            "ASSERT-HIDDEN-UNITS-NOT-DISCARDED: from_variant used to accept "
            "this kwarg and then overwrite it one line later."
        )

    def test_other_kwargs_still_override(self):
        """Anti-vacuity: the `config.update(kwargs)` path still works."""
        model = PowerMLP.from_variant(
            "small", num_classes=10, input_dim=784, k=5)
        assert model.k == 5

    def test_the_variant_architecture_is_what_gets_built(self):
        model = PowerMLP.from_variant("small", num_classes=7, input_dim=13)
        assert model.hidden_units[0] == 13
        assert model.hidden_units[-1] == 7
