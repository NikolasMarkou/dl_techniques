"""The README's deep-supervision and fine-tuning recipes are EXECUTED here.

F-06 / F-07. Three snippets in ``models/vision/resnet/README.md`` taught idioms that
cannot run against this model, because ``ResNet`` is a **subclassed**
``keras.Model``:

* section 7 and section 8 Example 4 read ``len(model.output)`` and keyed
  ``compile(loss={'output_0': ...})`` / ``fit(x, {'output_0': y, ...})`` by
  output NAME. A subclassed model has no ``output``, no ``input`` and no
  ``output_names``, so the dict spec resolves against ``None``.
* section 8 Example 3 called ``model.layers.pop()`` and re-wired from
  ``model.input``.

The list-indexed form the README now documents is the one
``src/train/resnet/train_resnet.py`` already uses (its ``loss``/
``loss_weights``/``metrics`` construction), so this file pins the doc and the
trainer to the same contract.

MEASURED on this repo, resnet18 / ``(32, 32, 3)`` / ``stem_type='cifar'`` /
``enable_deep_supervision=True``:

    len(model.output)          -> AttributeError: The layer res_net has never
                                  been called and thus has no defined output.
                                  (raised even AFTER calling the model)
    model.output_names         -> AttributeError: 'ResNet' object has no
                                  attribute 'output_names'
    dict-keyed compile + fit   -> TypeError: 'NoneType' object is not iterable
    list-indexed compile + fit -> trains; history keys are
                                  ['loss', 'primary_accuracy',
                                   'sparse_categorical_crossentropy_loss', ...]
                                  -- note there is no 'output_0_accuracy', which
                                  is why the README's ModelCheckpoint(
                                  monitor='val_output_0_accuracy') was dropped.

``model.layers.pop()`` deserves its own arm: it is NOT an error. ``layers`` is
a freshly computed property, so ``pop()`` mutates a throwaway list --
``len(model.layers)`` is unchanged and ``model(x)`` is bit-identical
afterwards. A silent no-op is worse than a raise, and only a behavioural
assertion catches it.

RED PROOFS -- four named injections, ACTUAL observed text.

Injection 1, give ``ResNet`` the Functional ``output_names`` the old snippet
assumed it had (``[f"output_{i}" ...]``) -> **3 failed, 3 passed**:

  - test_the_documented_list_indexed_pattern_trains:
    "AssertionError: assert 'primary_accuracy' in {'loss': [5.008768558502197],
     'output_0_loss': [3.4103403091430664],
     'output_0_primary_accuracy': [0.0], 'output_1_loss': [...], ...}"
    -- the metric NAME the README tells a ModelCheckpoint to monitor is
    governed by exactly this attribute.
  - test_the_dict_keyed_pattern_the_readme_used_to_teach_raises:
    "ValueError: For a model with multiple outputs, when providing the
     `metrics` argument as a list, it should have as many entries as the model
     has outputs. Received: metrics=['accuracy'] of length 1 whereas the model
     has 4 outputs." -- it still raises, but NOT the TypeError this arm pins,
    so the arm is specific to the missing-names cause and not to "any error".
  - test_the_functional_attributes_the_old_snippets_read_do_not_exist:
    "Failed: DID NOT RAISE <class 'AttributeError'>"

Injection 2, cache ``layers`` so ``pop()`` persists -> **1 failed, 5 passed**,
only test_layers_pop_is_a_silent_no_op_not_an_error:

  - "AssertionError: layers.pop() appeared to persist; the README's rationale
     for include_top=False assumes `layers` is a recomputed property
     assert 12 == 13"

Injection 3, ``_resolve_outputs`` guesses ``(32, 32, 3)`` instead of raising
when ``input_shape`` is None -> **1 failed, 5 passed**, only
test_the_inference_helper_needs_the_input_shape_the_readme_passes:

  - "Failed: DID NOT RAISE <class 'ValueError'>"

Injection 4, ``self.include_top = True`` unconditionally (Example 3's whole
premise) -> **1 failed, 5 passed**, only
test_the_documented_fine_tuning_wrapper_trains:

  - "ValueError: Input 0 of layer \"global_average_pooling2d\" is incompatible
     with the layer: expected ndim=4, found ndim=2. Full shape received:
     (None, 1000)"

Each injection convicts a different arm, so no arm is carried by another.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.resnet import ResNet
from dl_techniques.utils.deep_supervision import (
    create_inference_model_from_training_model,
    get_model_output_info,
)

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


def _deep_supervised_model() -> ResNet:
    """Build the small deep-supervised model every arm below shares."""
    return ResNet.from_variant(
        "resnet18",
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE,
        stem_type="cifar",
        enable_deep_supervision=True,
    )


def _synthetic_batch(batch: int = 8):
    rng = np.random.default_rng(0)
    x = rng.random((batch, *INPUT_SHAPE)).astype("float32")
    y = rng.integers(0, NUM_CLASSES, (batch,)).astype("int32")
    return x, y


def test_the_documented_list_indexed_pattern_trains() -> None:
    """Section 7 steps 3-7, verbatim: list loss / loss_weights / metrics / labels."""
    model = _deep_supervised_model()

    info = get_model_output_info(model, input_shape=INPUT_SHAPE)
    num_outputs = info["num_outputs"]
    assert num_outputs == 4, f"expected 4 supervision heads, got {num_outputs}"

    losses = [keras.losses.SparseCategoricalCrossentropy(from_logits=True)] * num_outputs
    loss_weights = [1.0, 0.3, 0.2, 0.1]
    metrics = [[keras.metrics.SparseCategoricalAccuracy(name="primary_accuracy")]]
    metrics += [[] for _ in range(num_outputs - 1)]

    model.compile(
        optimizer="adam",
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
    )

    x, y = _synthetic_batch()
    history = model.fit(
        x,
        [y] * num_outputs,
        validation_data=(x, [y] * num_outputs),
        epochs=1,
        verbose=0,
    )

    loss = history.history["loss"][0]
    assert np.isfinite(loss), f"training loss is not finite: {loss}"
    assert loss > 0.0, f"training loss collapsed to {loss}"

    # The metric NAME is the contract the README's ModelCheckpoint depends on.
    assert "primary_accuracy" in history.history
    assert "val_primary_accuracy" in history.history

    # And the name the README used to tell readers to monitor is absent.
    assert "output_0_accuracy" not in history.history
    assert "val_output_0_accuracy" not in history.history


def test_the_dict_keyed_pattern_the_readme_used_to_teach_raises() -> None:
    """RED arm: the removed ``{'output_0': ...}`` form cannot run at all."""
    model = _deep_supervised_model()
    x, y = _synthetic_batch()

    keyed_losses = {f"output_{i}": "sparse_categorical_crossentropy" for i in range(4)}
    keyed_weights = {"output_0": 1.0, "output_1": 0.3, "output_2": 0.2, "output_3": 0.1}
    keyed_labels = {f"output_{i}": y for i in range(4)}

    with pytest.raises(TypeError, match=r"'NoneType' object is not iterable"):
        model.compile(
            optimizer="adam",
            loss=keyed_losses,
            loss_weights=keyed_weights,
            metrics=["accuracy"],
        )
        model.fit(x, keyed_labels, epochs=1, verbose=0)


def test_the_functional_attributes_the_old_snippets_read_do_not_exist() -> None:
    """``.output`` / ``.input`` / ``.output_names`` are why the dict form fails.

    Asserted AFTER a forward pass on purpose: calling the model does not make
    them appear, which is the part that misleads readers.
    """
    model = _deep_supervised_model()
    x, _ = _synthetic_batch(batch=2)
    outputs = model(x, training=False)

    # The model really is multi-output -- the information exists, just not
    # under the Functional attribute names.
    assert isinstance(outputs, (list, tuple))
    assert len(outputs) == 4

    with pytest.raises(AttributeError, match=r"has no defined output"):
        len(model.output)

    with pytest.raises(AttributeError, match=r"has no defined input"):
        _ = model.input

    with pytest.raises(AttributeError, match=r"output_names"):
        _ = model.output_names


def test_layers_pop_is_a_silent_no_op_not_an_error() -> None:
    """Example 3's removed idiom: ``layers.pop()`` changes nothing at all."""
    model = ResNet.from_variant(
        "resnet18",
        num_classes=NUM_CLASSES,
        input_shape=INPUT_SHAPE,
        stem_type="cifar",
    )
    x, _ = _synthetic_batch(batch=2)

    before = np.asarray(model(x, training=False))
    n_before = len(model.layers)

    popped = model.layers.pop()          # raises nothing
    assert popped.name == "classifier"

    assert len(model.layers) == n_before, (
        "layers.pop() appeared to persist; the README's rationale for "
        "include_top=False assumes `layers` is a recomputed property"
    )
    assert model.classifier is not None

    after = np.asarray(model(x, training=False))
    assert float(np.max(np.abs(before - after))) == 0.0, (
        "layers.pop() changed the forward pass; the documented no-op claim "
        "is wrong and Example 3's note must be re-measured"
    )


def test_the_documented_fine_tuning_wrapper_trains() -> None:
    """Example 3 as rewritten: include_top=False -> keras.Input -> keras.Model."""
    base_model = ResNet.from_variant(
        "resnet18",
        include_top=False,
        input_shape=INPUT_SHAPE,
        stem_type="cifar",
    )

    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(100, activation="softmax", name="new_predictions")(x)
    model = keras.Model(inputs, outputs)

    # The load-bearing claim of the README's note: `model.layers` holds the
    # WHOLE backbone as one entry, so `model.layers[:-4]` would slice the
    # wrapper, not the network. Names carry a session-dependent numeric suffix
    # when other models were built in the same process, so match on structure.
    names = [layer.name for layer in model.layers]
    assert len(names) == 4, f"wrapper layout changed: {names}"
    assert names[0].startswith("input_layer"), names
    assert names[1].startswith("res_net"), names
    assert names[2].startswith("global_average_pooling2d"), names
    assert names[3] == "new_predictions", names
    assert len(base_model.layers) > len(model.layers), (
        "the backbone is supposed to be nested inside ONE wrapper layer; "
        f"base_model.layers={len(base_model.layers)} model.layers={len(names)}"
    )

    # Stage 1: early layers frozen.
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    frozen_count = len(model.trainable_weights)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    rng = np.random.default_rng(1)
    x_train = rng.random((4, *INPUT_SHAPE)).astype("float32")
    y_train = keras.utils.to_categorical(rng.integers(0, 100, (4,)), 100)
    stage1 = model.fit(x_train, y_train, epochs=1, verbose=0)
    assert np.isfinite(stage1.history["loss"][0])

    # Stage 2: unfreeze, re-compile, train again.
    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    thawed_count = len(model.trainable_weights)
    assert thawed_count > frozen_count, (
        f"unfreezing did not add trainable tensors: {frozen_count} -> {thawed_count}"
    )

    stage2 = model.fit(x_train, y_train, epochs=1, verbose=0)
    assert np.isfinite(stage2.history["loss"][0])


def test_the_inference_helper_needs_the_input_shape_the_readme_passes() -> None:
    """Section 7 step 8: the helper's ``input_shape`` argument is load-bearing."""
    model = _deep_supervised_model()

    with pytest.raises(ValueError, match=r"input_shape"):
        create_inference_model_from_training_model(model)

    inference_model = create_inference_model_from_training_model(
        model, input_shape=INPUT_SHAPE
    )
    assert inference_model.output.shape == (None, NUM_CLASSES)

    x, _ = _synthetic_batch(batch=4)
    predictions = inference_model(x, training=False)
    assert predictions.shape == (4, NUM_CLASSES)
    np.testing.assert_allclose(
        np.asarray(predictions),
        np.asarray(model(x, training=False)[0]),
        atol=1e-4,
    )
