"""Guards for ``src/train/bert/finetune.py`` and ``src/train/fnet/finetune.py``.

These two scripts are ~60 changed lines apart and are candidates for a shared
scaffold. Nothing in ``tests/`` exercised either of them before this module.

What is pinned, and why each pin is not satisfied-by-construction:

* ``test_sentiment_model_parameter_count`` -- a COUNT, split into encoder and
  head so a head-wiring change is isolated from an encoder change. Weak alone.
* ``test_the_backbone_is_the_pretrained_one`` -- VALUE-carrying: every backbone
  array of the rebuilt model equals the array in the saved encoder, so a model
  that kept its random init (or a load that quietly did nothing) fails.
* ``test_the_head_reads_the_CLS_POSITION`` -- VALUE-carrying, and the one that
  matters: it recomputes the model's own output from the encoder's
  ``last_hidden_state`` at position 0 and compares. It carries its own
  anti-vacuity precondition (position 1 must give a DIFFERENT answer), so it
  cannot pass by the outputs being position-independent.
* ``test_output_shape_is_num_classes`` -- VALUE-carrying.
* ``test_argv_maps_onto_the_config`` -- VALUE-carrying, and the reason this
  module exists at all: it pins ``max_seq_length`` = **256 for bert** and
  **128 for fnet** as a FACT, so a shared scaffold cannot silently harmonize
  them (that drift is deliberate-untouched, see ``src/train/AGENTS.md``).

The encoder is built and saved by this module (tiny variant, small vocab) --
``create_sentiment_model`` rebuilds the model from ``config.pretrained_encoder_path``
with ``include_top=True`` (the head is the model's own ``task_head``), so no
checkpoint from a real run is needed and no dataset is touched.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest

import train.bert.finetune as bert_finetune
import train.fnet.finetune as fnet_finetune

from dl_techniques.models.language.bert import BERT
from dl_techniques.models.language.fnet import FNet

from ._scaffold import assert_layer_names, capture_config_from_argv, effective_config

# Encoder stand-in geometry. Only the ENCODER's own size depends on these; the
# classification head depends on ``hidden_size`` (256, the tiny variant) and on
# ``config.num_classes``, which are the things this module is pinning.
PROBE_VOCAB_SIZE = 512
PROBE_SEQ_LENGTH = 16

BERT_ID = pytest.param("bert", id="bert")
FNET_ID = pytest.param("fnet", id="fnet")

SPECS: Dict[str, Dict[str, Any]] = {
    "bert": {
        "module": bert_finetune,
        "encoder_cls": BERT,
        "encoder_kwargs": {"attention_probs_dropout_rate": 0.1},
        "train_fn": "finetune_sentiment_model",
        # 3,295,232 = 2,244,608 + 1,050,624. The BERT `tiny` variant's
        # intermediate_size was corrected 512 -> 1024 to match the released
        # google/bert_uncased_L-4_H-256_A-4 config it names. The delta is
        # 4 layers x (256*1024 + 1024 + 1024*256 + 256
        #             - 256*512 - 512 - 512*256 - 256) = 1,050,624 exactly.
        # FNet is unaffected: it has no attention FFN of this shape.
        "encoder_params": 3_295_232,
        "head_params": 67_330,
        "layer_names": [
            "embeddings", "encoder_layer_0", "encoder_layer_1",
            "encoder_layer_2", "encoder_layer_3", "task_head",
        ],
        "config_defaults": {
            "pretrained_encoder_path":
                "results/bert_pretrain/pretrained_bert_encoder_best.keras",
            "save_dir": "results/bert_sentiment_finetune",
            "full_analysis_dir": "results/bert_sentiment_finetune/full_analysis",
            "num_classes": 2,
            "encoding_name": "cl100k_base",
            "cls_token_id": 100264,
            "sep_token_id": 100265,
            "pad_token_id": 100266,
            "mask_token_id": 100267,
            "dataset_name": "imdb_reviews",
            "max_samples": None,
            "max_seq_length": 256,  # DRIFT vs fnet's 128 -- pinned, not fixed
            "batch_size": 16,
            "run_two_stage_finetuning": True,
            "stage1_epochs": 5,
            "stage1_learning_rate": 1e-3,
            "stage2_epochs": 10,
            "stage2_learning_rate": 3e-5,
            "weight_decay": 0.01,
            "run_epoch_analysis": True,
            "analysis_start_epoch": 1,
            "analysis_epoch_frequency": 1,
            "run_post_training_analysis": True,
            "analysis_n_samples": 1000,
        },
    },
    "fnet": {
        "module": fnet_finetune,
        "encoder_cls": FNet,
        "encoder_kwargs": {},
        "train_fn": "finetune_sentiment_model",
        "encoder_params": 1_718_272,
        "head_params": 67_330,
        "layer_names": [
            "embeddings", "encoder_layer_0", "encoder_layer_1",
            "encoder_layer_2", "encoder_layer_3", "task_head",
        ],
        "config_defaults": {
            "pretrained_encoder_path":
                "results/fnet_pretrain/pretrained_fnet_encoder_best.keras",
            "save_dir": "results/fnet_sentiment_finetune",
            "full_analysis_dir": "results/fnet_sentiment_finetune/full_analysis",
            "num_classes": 2,
            "encoding_name": "cl100k_base",
            "cls_token_id": 100264,
            "sep_token_id": 100265,
            "pad_token_id": 100266,
            "mask_token_id": 100267,
            "dataset_name": "imdb_reviews",
            "max_samples": None,
            "max_seq_length": 128,  # DRIFT vs bert's 256 -- pinned, not fixed
            "batch_size": 16,
            "run_two_stage_finetuning": True,
            "stage1_epochs": 5,
            "stage1_learning_rate": 1e-3,
            "stage2_epochs": 10,
            "stage2_learning_rate": 3e-5,
            "weight_decay": 0.01,
            "run_epoch_analysis": True,
            "analysis_start_epoch": 1,
            "analysis_epoch_frequency": 1,
            "run_post_training_analysis": True,
            "analysis_n_samples": 1000,
        },
    },
}


def _probe_inputs(batch: int = 2):
    """Token ids that DIFFER per position, so a position-0 vs position-1 read
    of the encoder output is distinguishable."""
    ascending = np.arange(1, PROBE_SEQ_LENGTH + 1, dtype="int32")
    ids = np.stack(
        [ascending if row % 2 == 0 else ascending[::-1] for row in range(batch)]
    )
    return {
        "input_ids": keras.ops.convert_to_tensor(ids),
        "attention_mask": keras.ops.ones((batch, PROBE_SEQ_LENGTH), dtype="int32"),
        "token_type_ids": keras.ops.zeros((batch, PROBE_SEQ_LENGTH), dtype="int32"),
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Save a tiny encoder, then build each script's sentiment model from it.

    ``built(name)`` returns ``(model, encoder)``: the model the script rebuilt
    from the archive, and the headless encoder that archive was saved from.
    """
    cache: Dict[str, Any] = {}
    root = tmp_path_factory.mktemp("encoders")

    def _build(name: str):
        if name not in cache:
            spec = SPECS[name]
            encoder = spec["encoder_cls"].from_variant(
                variant="tiny",
                vocab_size=PROBE_VOCAB_SIZE,
                max_position_embeddings=PROBE_SEQ_LENGTH,
                hidden_dropout_rate=0.1,
                **spec["encoder_kwargs"],
            )
            encoder(_probe_inputs(1), training=False)
            path = str(root / f"{name}_encoder.keras")
            encoder.save(path)

            config = spec["module"].FinetuneConfig()
            config.pretrained_encoder_path = path
            cache[name] = (spec["module"].create_sentiment_model(config), encoder)
        return cache[name]

    return _build


@pytest.mark.parametrize("name", [BERT_ID, FNET_ID])
class TestModelConstruction:
    def test_sentiment_model_parameter_count(self, name, built):
        spec = SPECS[name]
        model, encoder = built(name)
        assert encoder.count_params() == spec["encoder_params"]
        assert model.task_head.count_params() == spec["head_params"]
        assert model.count_params() == spec["encoder_params"] + spec["head_params"]

    def test_sentiment_model_is_the_encoder_class_with_its_own_head(self, name, built):
        spec = SPECS[name]
        model, encoder = built(name)
        assert type(model) is type(encoder) is spec["encoder_cls"]
        assert model.include_top is True and encoder.include_top is False
        assert_layer_names([l.name for l in model.layers], spec["layer_names"])

    def test_output_shape_is_num_classes(self, name, built):
        spec = SPECS[name]
        model, _ = built(name)
        num_classes = spec["config_defaults"]["num_classes"]
        output = model(_probe_inputs(), training=False)
        assert not isinstance(output, dict), list(output)
        assert tuple(output.shape) == (2, num_classes)

    def test_the_backbone_is_the_pretrained_one(self, name, built):
        """Every backbone array equals the saved encoder's, by value.

        Replaces the old ``get_layer(<name>) is encoder`` identity check: the
        encoder is no longer a layer INSIDE the model, it is the model, so the
        claim that matters is that the checkpoint's values arrived.
        """
        model, encoder = built(name)
        loaded = {l.name: l for l in model.layers if l.name != "task_head"}
        saved = {l.name: l for l in encoder.layers}
        assert set(loaded) == set(saved)
        compared = 0
        for layer_name, layer in loaded.items():
            for got, want in zip(layer.weights, saved[layer_name].weights):
                np.testing.assert_array_equal(
                    np.asarray(got), np.asarray(want), err_msg=got.path
                )
                compared += 1
        assert compared == len(encoder.weights) > 0

    def test_the_head_reads_the_CLS_POSITION(self, name, built):
        """The classifier must be fed ``last_hidden_state[:, 0, :]``.

        Recomputed from the model's own head and the saved encoder, then
        compared. The precondition below is what keeps this non-vacuous: if
        position 1 gave the same logits, the assertion could not tell the two
        apart.
        """
        model, encoder = built(name)
        head = model.task_head
        inputs = _probe_inputs()

        hidden = encoder(inputs, training=False)["last_hidden_state"]
        at_cls = np.asarray(
            head({"hidden_states": hidden[:, 0, :]}, training=False)["logits"]
        )
        at_second = np.asarray(
            head({"hidden_states": hidden[:, 1, :]}, training=False)["logits"]
        )
        assert not np.allclose(at_cls, at_second, atol=1e-6), (
            "probe is vacuous: positions 0 and 1 give the same logits"
        )

        actual = np.asarray(model(inputs, training=False))
        np.testing.assert_allclose(actual, at_cls, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", [BERT_ID, FNET_ID])
class TestArgvToConfig:
    def test_argv_maps_onto_the_config_at_its_defaults(self, name, monkeypatch):
        spec = SPECS[name]
        config = capture_config_from_argv(
            monkeypatch, spec["module"], spec["train_fn"], []
        )
        assert effective_config(config) == spec["config_defaults"]

    def test_every_flag_reaches_the_field_it_names(self, name, monkeypatch):
        spec = SPECS[name]
        config = capture_config_from_argv(
            monkeypatch,
            spec["module"],
            spec["train_fn"],
            ["--encoder-path", "/tmp/probe_encoder.keras", "--batch-size", "3",
             "--max-samples", "77", "--no-two-stage", "--skip-analysis"],
        )
        expected = dict(spec["config_defaults"])
        expected.update({
            "pretrained_encoder_path": "/tmp/probe_encoder.keras",
            "batch_size": 3,
            "max_samples": 77,
            "run_two_stage_finetuning": False,
            "run_post_training_analysis": False,
        })
        assert effective_config(config) == expected

    def test_the_two_store_true_flags_are_opt_OUT_switches(self, name, monkeypatch):
        """``--no-two-stage`` and ``--skip-analysis`` are NEGATIONS: omitting
        them must leave both behaviours ON. ``run_post_training_analysis``
        defaulting to True is why the analysis path runs on every default run.
        """
        spec = SPECS[name]
        config = capture_config_from_argv(
            monkeypatch, spec["module"], spec["train_fn"], []
        )
        assert config.run_two_stage_finetuning is True
        assert config.run_post_training_analysis is True
