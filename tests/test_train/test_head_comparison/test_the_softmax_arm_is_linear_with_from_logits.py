"""The ``softmax`` arm is a linear Dense trained with ``from_logits=True``.

The module docstring said "Dense(100) + softmax, cross-entropy" (F-9 N17):
the code builds a linear ``Dense(NUM_FINE)`` and compiles
``SparseCategoricalCrossentropy(from_logits=True)``, the softmax living in the
loss. Since plan-2026-10-14T062616-81938e81 step 11.3 the docstring says so,
and the logits rule (which arms emit logits) has one home, ``LOGIT_ARMS``,
read by the loss (``loss_for_head``) and the post-hoc probabilities.
"""

import numpy as np
import pytest

import keras

import train.head_comparison.train_heads as heads


class _Stop(Exception):
    pass


def test_the_arm_is_linear_and_the_loss_reads_logits():
    _, layer, _ = heads.build_model(heads.HeadComparisonConfig(head="softmax"))
    assert isinstance(layer, keras.layers.Dense)
    assert layer.activation is keras.activations.linear
    assert heads.loss_for_head("softmax").from_logits is True


@pytest.mark.parametrize("head", heads.HEAD_ARMS)
def test_each_arm_compiles_the_loss_its_output_needs(head):
    assert heads.loss_for_head(head).from_logits is (head in heads.LOGIT_ARMS)


def test_the_logit_arms_are_the_ones_without_a_probability_layer():
    assert heads.LOGIT_ARMS == {"softmax", "harmonic_logits"}
    with pytest.raises(ValueError, match="unknown head"):
        heads.loss_for_head("softmax_v2")


def test_train_heads_compiles_with_loss_for_head(tmp_path, monkeypatch):
    """``train_heads`` itself, up to ``compile``: the loss it passes is the
    from-logits cross-entropy for the softmax arm."""
    fine = np.arange(200) % heads.NUM_FINE
    images = np.zeros((200, 32, 32, 3), "float32")
    monkeypatch.setattr(heads, "setup_gpu", lambda gpu_id=None: None)
    monkeypatch.setattr(heads, "load_dataset", lambda name: (
        (images, fine), (images, fine), None, None))
    monkeypatch.setattr(keras.datasets.cifar100, "load_data", lambda label_mode: (
        (None, fine // 5), (None, fine // 5)))
    seen = []

    def compile_spy(self, *args, **kwargs):
        seen.append(kwargs["loss"])
        raise _Stop

    monkeypatch.setattr(keras.Model, "compile", compile_spy)
    config = heads.HeadComparisonConfig(head="softmax", epochs=1,
                                        output_dir=str(tmp_path))
    with pytest.raises(_Stop):
        heads.train_heads(config)
    (loss,) = seen
    assert isinstance(loss, keras.losses.SparseCategoricalCrossentropy)
    assert loss.from_logits is True


def test_the_docstring_describes_the_linear_arm():
    doc = heads.__doc__
    assert "Dense(100) + softmax" not in doc
    assert "from_logits=True" in doc.split("``softmax``", 1)[1].split("- ``", 1)[0]
