"""``create_ditxa(variant, pretrained=<path>)`` loads every weight, or raises.

Before 2026-10-15 the advertised path route was dead: ``from_variant`` called
``model.load_weights(path)`` on a model that had never been built, and Keras
raised ``ValueError: You are loading weights into a model that has not yet been
built`` for every checkpoint, matching or not. The repaired route builds the
model from its own ``input_size``/``in_channels`` and copies every layer by name
through ``load_weights_from_checkpoint(..., strict=True, allow_unloaded=())``.

The claims, each on values:

1. a checkpoint of the same variant loads: every weight tensor is
   ``array_equal`` to the saved one, and the output on a fixed input is equal;
2. a checkpoint of another variant (a wider model) raises, and so do a deeper
   and a shallower checkpoint of the same width, which a shape check alone
   would let through (a missing or a surplus block has no shape to mismatch);
3. ``pretrained=True`` raises ``NotImplementedError`` and a missing path raises
   ``FileNotFoundError``.

The saved model is ACTIVATED first (``_ditxa_helpers.activate``): a fresh DiTXA
emits the exact zero tensor, so an output comparison on it would pass for a
model that loaded nothing.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision_language.bit_diffusion.model import (
    DiTXA,
    create_ditxa,
)

from ._ditxa_helpers import activate, batch, np_

#: The smallest shipped variant; the checkpoint under test is one of these.
VARIANT = "tiny"


def _relative_path(variable) -> str:
    """A weight path without the model's auto-numbered name (``di_txa_3/...``)."""
    return variable.path.split("/", 1)[1]


def _saved_activated_model(path, **overrides):
    """Build, activate and save a model of ``VARIANT``; return it and inputs."""
    keras.utils.set_random_seed(0)
    model = DiTXA.from_variant(VARIANT, label_seed=13, **overrides)
    inputs = batch(model, batch_size=2)
    model(inputs, training=False)
    activate(model, seed=5)
    model.save(path)
    return model, inputs


def test_a_saved_variant_loads_every_weight_by_value(tmp_path):
    path = str(tmp_path / "ditxa_tiny.keras")
    source, inputs = _saved_activated_model(path)
    expected = np_(source(inputs, training=False))
    assert np.any(expected != 0.0), "the saved model emits zeros; activate() did not run"

    loaded = create_ditxa(VARIANT, pretrained=path, label_seed=13)

    assert loaded.built
    assert [_relative_path(w) for w in loaded.weights] == [
        _relative_path(w) for w in source.weights
    ]
    unequal = [
        _relative_path(w)
        for w, s in zip(loaded.weights, source.weights)
        if not np.array_equal(np_(w), np_(s))
    ]
    assert not unequal, f"{len(unequal)} weight tensors did not load: {unequal}"
    # Anti-vacuity: the comparison is over trained-looking numbers, not zeros.
    nonzero = sum(1 for w in source.weights if np.any(np_(w) != 0.0))
    assert nonzero >= 40, f"only {nonzero} saved tensors are non-zero"

    np.testing.assert_array_equal(np_(loaded(inputs, training=False)), expected)


def test_a_checkpoint_of_another_variant_raises(tmp_path):
    """``S`` is 384 wide against ``tiny``'s 64: the first layer mismatches."""
    keras.utils.set_random_seed(0)
    other = DiTXA.from_variant("S")
    other(batch(other, batch_size=1), training=False)
    path = str(tmp_path / "ditxa_S.keras")
    other.save(path)
    del other

    with pytest.raises(ValueError, match="Shape mismatch"):
        create_ditxa(VARIANT, pretrained=path)


@pytest.mark.parametrize("depth", [1, 3], ids=["shallower", "deeper"])
def test_a_checkpoint_of_another_depth_raises(tmp_path, depth):
    """No layer mismatches in shape here: a block is missing, or left over.

    ``tiny`` has 2 blocks. A 1-block checkpoint leaves ``block_1`` at its
    initialization; a 3-block checkpoint holds a ``block_2`` with no home.
    Both are a partial load that a shape check alone would report as success.
    """
    path = str(tmp_path / f"ditxa_depth{depth}.keras")
    _saved_activated_model(path, depth=depth)

    with pytest.raises(ValueError, match="block_"):
        create_ditxa(VARIANT, pretrained=path)


def test_pretrained_true_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="DiTXA"):
        create_ditxa(VARIANT, pretrained=True)


def test_a_missing_path_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_ditxa(VARIANT, pretrained=str(tmp_path / "absent.keras"))
