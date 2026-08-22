"""Guards for ``src/train/bert/pretrain.py`` and ``src/train/fnet/pretrain.py``.

These two scripts are ~55 changed lines apart and are candidates for a shared
scaffold. Nothing in ``tests/`` exercised either of them before this module, so
a merge was unverifiable by construction.

What is pinned, and why each pin is not satisfied-by-construction:

* ``test_mlm_model_parameter_count`` -- a COUNT. Weak on its own (most
  mutations leave it untouched); it is here only to catch a variant/vocab
  change, and it is RED-proven by exactly that.
* ``test_mlm_wrapper_receives_the_config_values`` -- VALUE-carrying. Pins the
  masking hyper-parameters AND the ORDER of ``special_token_ids``
  (cls, sep, pad, mask), which no parameter count can see.
* ``test_encoder_factory_kwargs`` -- VALUE-carrying. Pins the kwargs each
  script hands ``<Model>.from_variant``, read back out of the built encoder's
  own config, including the bert/fnet divergence: bert passes
  ``attention_probs_dropout_rate``, FNet's factory does not take it (Fourier
  token mixing has no attention block).
* ``test_argv_maps_onto_the_config`` -- VALUE-carrying. Pins the whole
  ``argv -> config`` map, i.e. every default a shared scaffold could silently
  harmonize.

Device: CPU-cheap apart from the two model builds (~4s each on the gate
device); both use the scripts' REAL default configs, not a shrunken stand-in.
"""

from typing import Any, Dict

import pytest

import train.bert.pretrain as bert_pretrain
import train.fnet.pretrain as fnet_pretrain

from ._scaffold import assert_layer_names, capture_config_from_argv, effective_config

# ---------------------------------------------------------------------
# Per-script pinned facts. Literal on purpose -- see the module docstring.
# ---------------------------------------------------------------------

BERT = pytest.param("bert", id="bert")
FNET = pytest.param("fnet", id="fnet")

SPECS: Dict[str, Dict[str, Any]] = {
    "bert": {
        "module": bert_pretrain,
        "factory": "create_bert_mlm_model",
        "train_fn": "train_bert_mlm",
        "variant_field": "bert_variant",
        # Built from the script's own default TrainingConfig().
        "total_params": 53_650_613,
        "encoder_params": 27_813_120,
        "model_layer_names": ["bert", "mlm_dense", "mlm_dropout", "mlm_norm", "mlm_output"],
        "encoder_layer_names": [
            "embeddings", "encoder_layer_0", "encoder_layer_1",
            "encoder_layer_2", "encoder_layer_3",
        ],
        "encoder_kwargs": {
            "vocab_size": 100277,
            "max_position_embeddings": 128,
            "hidden_dropout_rate": 0.1,
            "attention_probs_dropout_rate": 0.1,
        },
        "config_defaults": {
            "bert_variant": "tiny",
            "vocab_size": 100277,
            "max_seq_length": 128,
            "encoding_name": "cl100k_base",
            "cls_token_id": 100264,
            "sep_token_id": 100265,
            "pad_token_id": 100266,
            "mask_token_id": 100267,
            "batch_size": 32,
            "num_epochs": 3,
            "learning_rate": 5e-4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mask_ratio": 0.15,
            "random_token_ratio": 0.1,
            "unchanged_ratio": 0.1,
            "save_dir": "results/bert_pretrain",
            "dataset_name": "imdb_reviews",
            "max_samples": 10000,
            "run_epoch_analysis": True,
            "analysis_start_epoch": 1,
            "analysis_epoch_frequency": 5,
        },
    },
    "fnet": {
        "module": fnet_pretrain,
        "factory": "create_fnet_mlm_model",
        "train_fn": "train_fnet_mlm",
        "variant_field": "fnet_variant",
        "total_params": 53_253_301,
        "encoder_params": 27_415_808,
        "model_layer_names": ["f_net", "mlm_dense", "mlm_dropout", "mlm_norm", "mlm_output"],
        "encoder_layer_names": [
            "embeddings", "encoder_layer_0", "encoder_layer_1",
            "encoder_layer_2", "encoder_layer_3",
        ],
        "encoder_kwargs": {
            "vocab_size": 100277,
            "max_position_embeddings": 128,
            "hidden_dropout_rate": 0.1,
        },
        "config_defaults": {
            "fnet_variant": "tiny",
            "vocab_size": 100277,
            "max_seq_length": 128,
            "encoding_name": "cl100k_base",
            "cls_token_id": 100264,
            "sep_token_id": 100265,
            "pad_token_id": 100266,
            "mask_token_id": 100267,
            "batch_size": 32,
            "num_epochs": 3,
            "learning_rate": 5e-4,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "mask_ratio": 0.15,
            "random_token_ratio": 0.1,
            "unchanged_ratio": 0.1,
            "save_dir": "results/fnet_pretrain",
            "dataset_name": "imdb_reviews",
            "max_samples": 10000,
            "run_epoch_analysis": True,
            "analysis_start_epoch": 1,
            "analysis_epoch_frequency": 5,
        },
    },
}

# The masking hyper-parameters every MLM wrapper must receive, and the ORDER
# the special-token list must be built in. Identical in both scripts today --
# stated once, asserted per script.
MLM_WRAPPER_VALUES = {
    "mask_ratio": 0.15,
    "random_token_ratio": 0.1,
    "unchanged_ratio": 0.1,
    "mask_token_id": 100267,
    "vocab_size": 100277,
    "special_token_ids": [100264, 100265, 100266, 100267],  # cls, sep, pad, mask
}


@pytest.fixture(scope="module")
def built(request):
    """Build each MLM model once per module (each build is ~4s)."""
    cache: Dict[str, Any] = {}

    def _build(name: str):
        if name not in cache:
            spec = SPECS[name]
            config = spec["module"].TrainingConfig()
            cache[name] = getattr(spec["module"], spec["factory"])(config)
        return cache[name]

    return _build


@pytest.mark.parametrize("name", [BERT, FNET])
class TestModelConstruction:
    def test_mlm_model_parameter_count(self, name, built):
        spec = SPECS[name]
        model = built(name)
        assert model.count_params() == spec["total_params"]
        assert model.encoder.count_params() == spec["encoder_params"]

    def test_mlm_model_layer_names(self, name, built):
        spec = SPECS[name]
        model = built(name)
        assert_layer_names([l.name for l in model.layers], spec["model_layer_names"])
        assert_layer_names(
            [l.name for l in model.encoder.layers], spec["encoder_layer_names"]
        )

    def test_mlm_wrapper_receives_the_config_values(self, name, built):
        """The masking hyper-parameters and the special-token ORDER."""
        model = built(name)
        observed = {
            "mask_ratio": model.mask_ratio,
            "random_token_ratio": model.random_token_ratio,
            "unchanged_ratio": model.unchanged_ratio,
            "mask_token_id": model.mask_token_id,
            "vocab_size": model.vocab_size,
            "special_token_ids": list(model.special_token_ids),
        }
        assert observed == MLM_WRAPPER_VALUES

    def test_encoder_factory_kwargs(self, name, built):
        """What the script hands ``<Model>.from_variant``, read back out."""
        spec = SPECS[name]
        encoder_config = built(name).encoder.get_config()
        observed = {key: encoder_config[key] for key in spec["encoder_kwargs"]}
        assert observed == spec["encoder_kwargs"]

    def test_only_bert_passes_an_attention_dropout_kwarg(self, name, built):
        """FNet mixes tokens with a Fourier transform -- it has no attention
        block, so its factory must not be handed ``attention_probs_dropout_rate``.
        """
        encoder_config = built(name).encoder.get_config()
        expected = name == "bert"
        assert ("attention_probs_dropout_rate" in encoder_config) is expected


@pytest.mark.parametrize("name", [BERT, FNET])
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
            ["--variant", "small", "--epochs", "7", "--batch-size", "11",
             "--max-samples", "123"],
        )
        expected = dict(spec["config_defaults"])
        expected.update({
            spec["variant_field"]: "small",
            "num_epochs": 7,
            "batch_size": 11,
            "max_samples": 123,
        })
        assert effective_config(config) == expected

    def test_max_samples_zero_is_carried_through_verbatim(self, name, monkeypatch):
        """``main()`` must not coerce or clamp ``--max-samples``: the
        steps-per-epoch fallback in ``train_*_mlm`` branches on its truthiness.
        """
        spec = SPECS[name]
        config = capture_config_from_argv(
            monkeypatch, spec["module"], spec["train_fn"], ["--max-samples", "0"]
        )
        assert effective_config(config)["max_samples"] == 0
