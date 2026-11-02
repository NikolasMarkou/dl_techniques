"""Each NLP stage is built at the rank ``call`` feeds it.

Measured on 2026-11-02 before the fixes (audit of ``layers/heads``):

* ``normalization_type='batch_norm'`` failed on all six heads. Its ``InputSpec``
  pins the rank at build. Pooled heads built ``norm`` at rank 3 and called it at
  rank 2, ``MultipleChoiceHead`` built it at rank 2 and called it at rank 3, and
  the intermediate block was built at rank 3 and called per row, at rank 2.
* With ``global_response_norm`` and ``dynamic_tanh`` the classification head
  returned ``(1, B, C)``, and with the adaptive band norms ``(B, B, C)``, while
  reporting ``(B, C)``: a norm built at rank 3 kept broadcasting at rank 3.
* ``MultipleChoiceHead`` under ``tf.function`` with unknown batch, choices and
  length raised ``TypeError``: the unfolding reshape lost the static width.
* Task attention hard-coded ``num_heads=8``, so width 12 could not use it.
* ``TokenClassificationHead`` reported ``predictions`` as float32 symbolically
  and returned int32.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.heads.nlp import NLPTaskConfig, create_nlp_head

WIDTH = 16
SEQUENCE = (2, 5, WIDTH)
CHOICES = (2, 3, 5, WIDTH)
HEADS = [
    ("text_classification", {"num_classes": 2}, SEQUENCE),
    ("token_classification", {"num_classes": 2}, SEQUENCE),
    ("question_answering", {}, SEQUENCE),
    ("text_similarity", {}, SEQUENCE),
    ("text_generation", {"vocabulary_size": 9}, SEQUENCE),
    ("multiple_choice", {}, CHOICES),
]
IDS = [row[0] for row in HEADS]


def _zeros(shape):
    return np.zeros(shape, "float32")


def _config(task, fields):
    return NLPTaskConfig(name="x", task_type=task, **fields)


@pytest.mark.parametrize("task,fields,shape", HEADS, ids=IDS)
def test_batch_norm_runs_on_every_head(task, fields, shape) -> None:
    head = create_nlp_head(_config(task, fields), WIDTH,
                           normalization_type="batch_norm")
    assert head(_zeros(shape), training=True)


@pytest.mark.parametrize("task,fields,shape", HEADS, ids=IDS)
def test_batch_norm_runs_without_the_intermediate_block(task, fields, shape) -> None:
    head = create_nlp_head(_config(task, fields), WIDTH,
                           normalization_type="batch_norm", use_intermediate=False)
    assert head(_zeros(shape), training=True)


@pytest.mark.parametrize("task,fields,shape", HEADS, ids=IDS)
def test_a_batch_norm_head_survives_a_keras_round_trip(
        task, fields, shape, tmp_path) -> None:
    inputs = keras.Input(shape[1:])
    head = create_nlp_head(_config(task, fields), WIDTH,
                           normalization_type="batch_norm")
    model = keras.Model(inputs, head(inputs))
    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    data = np.random.default_rng(0).random(shape).astype("float32")
    original, restored = model(data, training=False), reloaded(data, training=False)
    for key in original:
        np.testing.assert_array_equal(np.array(original[key]), np.array(restored[key]))


@pytest.mark.parametrize("norm", [
    "global_response_norm", "dynamic_tanh", "adaptive_band_rms",
    "zero_centered_adaptive_band_rms_norm",
])
def test_a_rank_keeping_norm_returns_the_reported_shape(norm) -> None:
    head = create_nlp_head(_config("text_classification", {"num_classes": 2}), WIDTH,
                           normalization_type=norm)
    outputs = head(_zeros(SEQUENCE))
    assert outputs["logits"].shape == (2, 2)
    assert head.compute_output_shape((2, 5, WIDTH))["logits"] == (2, 2)


@pytest.mark.parametrize("signature", [
    (None, None, None, WIDTH),
    (None, 3, None, WIDTH),
    (None, None, 5, WIDTH),
])
def test_multiple_choice_traces_with_unknown_axes(signature) -> None:
    head = create_nlp_head(_config("multiple_choice", {}), WIDTH)

    @tf.function(input_signature=[tf.TensorSpec(signature, tf.float32)])
    def run(features):
        return head(features, training=False)

    assert run(_zeros(CHOICES))["logits"].shape == (2, 3)


def test_the_task_attention_head_count_is_an_argument() -> None:
    head = create_nlp_head(_config("token_classification", {"num_classes": 2}), 12,
                           use_task_attention=True, num_heads=4)
    assert head.num_heads == 4
    assert head.task_attention.num_heads == 4
    assert head(_zeros((2, 5, 12)))["logits"].shape == (2, 5, 2)


def test_the_task_attention_head_count_defaults_to_eight() -> None:
    head = create_nlp_head(_config("token_classification", {"num_classes": 2}), WIDTH,
                           use_task_attention=True)
    assert head.num_heads == 8
    assert head.task_attention.num_heads == 8


def test_a_width_the_head_count_cannot_divide_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="TokenClassificationHead.*num_heads"):
        create_nlp_head(_config("token_classification", {"num_classes": 2}), 12,
                        use_task_attention=True)


@pytest.mark.parametrize("kwargs", [
    {"num_heads": 4},
    {"num_heads": 4, "use_task_attention": True, "attention_type": "channel"},
])
def test_a_head_count_no_layer_reads_is_refused(kwargs) -> None:
    with pytest.raises(ValueError, match="num_heads"):
        create_nlp_head(_config("token_classification", {"num_classes": 2}), WIDTH,
                        **kwargs)


def test_the_head_count_survives_get_config() -> None:
    head = create_nlp_head(_config("token_classification", {"num_classes": 2}), 12,
                           use_task_attention=True, num_heads=4)
    assert type(head).from_config(head.get_config()).num_heads == 4


def test_predictions_are_reported_with_the_dtype_they_have() -> None:
    head = create_nlp_head(_config("token_classification", {"num_classes": 2}), WIDTH)
    symbolic = head(keras.Input((5, WIDTH)))
    eager = head(_zeros((1, 5, WIDTH)))
    assert symbolic["predictions"].dtype == keras.backend.standardize_dtype(
        eager["predictions"].dtype)
    assert symbolic["logits"].dtype == "float32"
