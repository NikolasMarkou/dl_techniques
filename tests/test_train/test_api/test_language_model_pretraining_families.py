"""The ``gpt2_lm`` family: what its adapter offers, refuses and is given, and that every variant builds at the tokenizer's vocabulary.

``train.api.text_models`` (plan-2026-11-24T092348-7806a290 step 4) registers one family for ``language_model_pretraining``:
``gpt2_lm``, the library GPT-2 trained as a causal language model. The library factory (``create_gpt2``) does not take the arguments
the API calls a factory with, and its model returns a dict of ``logits`` and ``last_hidden_state`` that the stock ``fit`` cannot
train, so the adapter builds a FUNCTIONAL model (a float32 id input, an in-graph cast, the backbone named ``gpt2_backbone``, the
logits) and nothing else. Pinned here:

* the census of ``create_gpt2`` and the ``GPT2`` constructor: every parameter is classified (allowed, refused or supplied) and
  every classified key exists, in the pattern of the denoising and pretrain families;
* every variant of the library table builds at the ascii vocabulary (its own width and heads, one block, to keep a 1.5 billion
  parameter variant out of a CPU test), is functional, built, takes float32 ids and returns ``(None, seq_len, vocab_size)``;
* the family belongs to its task alone, and every other family is refused under it;
* the adapter's refusals are named: a vocabulary or a position table that cannot be built, a ``num_classes``, a key the table
  refuses (a task head, the harmonic head, a pad id, a name) or supplies (the vocabulary, the variant, the shape);
* the recipe is complete for the keys the resolver fills, and its notice names every number that no run measured.

Tests write only under ``tmp_path``.
"""

import re

import keras
import pytest

from dl_techniques.models.language.gpt2.gpt2 import GPT2
from train.api import recipes, registry
from train.api import text_models as tm
from train.api.args import OptimArgs
from train.api.errors import ConfigError, ConfigValidationError
from train.api.text_args import GPT2_LM, TASK

from ._text_world import ASCII_VOCAB, SEQ_LEN, TINY_MODEL_ARGS, world  # noqa: F401  (``world`` is a fixture)


def test_the_signature_census_is_empty_in_both_directions():
    census = tm.signature_census(GPT2_LM)
    assert census == {"unclassified": [], "stale": []}, census
    signatures = tm.factory_signatures(GPT2_LM)
    assert set(signatures) == set(tm.FACTORIES[GPT2_LM])
    assert "vocab_size" in signatures[f"{tm.GPT2_MODULE}:GPT2"], "the constructor is part of the census"


def test_the_census_finds_a_parameter_no_table_names_and_a_key_no_factory_has():
    signatures = {"x:f": ["variant", "embed_dim", "a_new_knob"]}
    found = tm.signature_census(GPT2_LM, signatures)
    assert found["unclassified"] == ["a_new_knob"]
    assert "depth" in found["stale"] and "vocab_size" in found["stale"], "keys the tables name that no factory has"


def test_the_tables_partition_the_keys():
    allowed, refused, supplied = (set(tm.ALLOWED[GPT2_LM]), set(tm.REFUSED[GPT2_LM]), set(tm.SUPPLIED[GPT2_LM]))
    assert not (allowed & refused) and not (allowed & supplied) and not (refused & supplied)
    assert {"vocab_size", "variant", "input_shape"} <= supplied
    assert {"include_top", "task_config", "head_config_overrides", "pad_token_id", "head_type", "name", "pretrained"} <= refused
    assert {"embed_dim", "depth", "num_heads", "max_seq_len", "dropout_rate", "tie_word_embeddings"} <= allowed
    for key, why in tm.REFUSED[GPT2_LM].items():
        assert why and not why.endswith("."), key


@pytest.mark.parametrize("variant", sorted(GPT2.MODEL_VARIANTS))
def test_every_variant_builds_at_the_ascii_vocabulary_with_its_own_width(variant):
    keras.backend.clear_session()
    model = tm.create_gpt2_lm(variant=variant, input_shape=(SEQ_LEN,), vocab_size=ASCII_VOCAB, depth=1)
    backbone = model.get_layer(tm.BACKBONE_NAME)
    assert model.built and isinstance(model, keras.Model) and not isinstance(model, GPT2)
    assert model.input_shape == (None, SEQ_LEN) and model.inputs[0].dtype == "float32"
    assert model.output_shape == (None, SEQ_LEN, ASCII_VOCAB)
    table = GPT2.MODEL_VARIANTS[variant]
    assert (backbone.embed_dim, backbone.num_heads, backbone.max_seq_len) == (table["embed_dim"], table["num_heads"], table["max_seq_len"])
    assert backbone.depth == 1 and backbone.vocab_size == ASCII_VOCAB, "the vocabulary is the one given, not the model's 100,277"
    assert model.weights[0].path.startswith(f"{tm.BACKBONE_NAME}/"), "init transfers by the backbone's name"


def test_the_adapter_returns_the_logits_of_ids_cast_in_the_graph():
    keras.backend.clear_session()
    model = tm.create_gpt2_lm(variant="tiny", input_shape=(SEQ_LEN,), vocab_size=ASCII_VOCAB, **TINY_MODEL_ARGS)
    ids = keras.ops.convert_to_tensor([[3.0] * SEQ_LEN, [7.0] * SEQ_LEN])
    logits = keras.ops.convert_to_numpy(model(ids))
    assert logits.shape == (2, SEQ_LEN, ASCII_VOCAB) and logits.dtype.name == "float32"
    same = keras.ops.convert_to_numpy(model(keras.ops.cast(ids, "int32")))
    assert (logits == same).all(), "a float id and the int it names give the same logits (the cast is in the graph)"
    assert not (logits[0] == logits[1]).all(), "the ids reach the embedding"


@pytest.mark.parametrize("kwargs, text", [
    ({"num_classes": 3}, "num_classes"),
    ({"vocab_size": 0}, "vocab_size"),
    ({"vocab_size": True}, "vocab_size"),
    ({"input_shape": (SEQ_LEN, 1)}, "input_shape"),
    ({"input_shape": (None,)}, "input_shape"),
    ({"include_top": True}, "include_top"),
    ({"pad_token_id": 0}, "pad_token_id"),
    ({"head_type": "harmonic"}, "head_type"),
    ({"name": "other"}, "name"),
    ({"pretrained": "x.keras"}, "pretrained"),
    ({"bogus": 1}, "bogus"),
])
def test_the_adapter_refuses_by_name(kwargs, text):
    asked = {"variant": "tiny", "input_shape": (SEQ_LEN,), "vocab_size": ASCII_VOCAB, **TINY_MODEL_ARGS, **kwargs}
    with pytest.raises(ValueError, match=text):
        tm.create_gpt2_lm(**asked)


def test_a_position_table_shorter_than_the_sequence_is_the_librarys_refusal_by_name(world):  # noqa: F811
    spec = registry.MODELS[GPT2_LM]
    with pytest.raises(ConfigError, match="max_seq_len") as caught:
        recipes.build_model(spec, "tiny", None, (513,), {"vocab_size": ASCII_VOCAB, **TINY_MODEL_ARGS})
    assert "gpt2_lm" in str(caught.value)
    built = recipes.build_model(spec, "tiny", None, (512,), {"vocab_size": ASCII_VOCAB, **TINY_MODEL_ARGS})
    assert built.output_shape == (None, 512, ASCII_VOCAB), "the neighbouring length builds"


# ---------------------------------------------------------------------
# The row, its task and the resolver
# ---------------------------------------------------------------------


def test_the_family_trains_this_task_only_and_every_other_family_is_refused_under_it(world):  # noqa: F811
    row = registry.MODELS[GPT2_LM]
    assert row.tasks == (TASK,) and row.transfer is not None
    assert registry.list_models(task=TASK) == [f"{GPT2_LM}/{variant}" for variant in sorted(GPT2.MODEL_VARIANTS)]
    with pytest.raises(ConfigError, match=TASK):
        registry.parse_model_name(f"{GPT2_LM}/tiny", task="image_classification")
    others = sorted(set(registry.MODELS) - {GPT2_LM})
    assert len(others) >= 20, "the control: the other families are really registered"
    for family in others:
        with pytest.raises(ConfigError, match=f"the family '{family}' trains"):
            registry.parse_model_name(f"{family}/anything", task=TASK)


def test_the_resolver_names_every_refused_key_at_once(world):  # noqa: F811
    spec = registry.MODELS[GPT2_LM]
    context = recipes_context(spec)
    with pytest.raises(ConfigValidationError) as caught:
        spec.load_resolver()(context, context_optim(context), {"include_top": True, "vocab_size": 5, "bogus": 1})
    text = str(caught.value)
    for key in ("model_args.include_top", "model_args.vocab_size", "model_args.bogus"):
        assert key in text, key
    assert "set by the task" in text and "not offered by 'gpt2_lm'" in text


def test_an_sgd_decay_is_refused_because_no_regularizer_is_built(world):  # noqa: F811
    spec = registry.MODELS[GPT2_LM]
    context = recipes_context(spec)
    optim = OptimArgs(optimizer="sgd", lr=0.1, weight_decay=0.01, momentum=0.9, beta_1=None, beta_2=None, epsilon=None,
                      schedule="constant", warmup_epochs=0, warmup_start_lr=1e-8, min_lr_ratio=0.01,
                      decay_over="total", clipnorm=None, global_clipnorm=None)
    with pytest.raises(ConfigError, match="weight_decay"):
        spec.load_resolver()(context, optim, {})


def recipes_context(spec):
    from train.api.args import AutoContext

    return AutoContext(family=GPT2_LM, variant="tiny", dataset="synthetic", num_classes=None, image_size=None,
                       defaults=spec.recipe("synthetic", "tiny"))


def context_optim(context):
    return OptimArgs().resolve_auto(context)


def test_the_recipe_fills_every_auto_field_it_can_and_uses_only_known_keys(world):  # noqa: F811
    spec = registry.MODELS[GPT2_LM]
    recipe = spec.recipe("imdb_reviews", "tiny")
    assert set(recipe) <= set(recipes.RECIPE_KEYS), set(recipe) - set(recipes.RECIPE_KEYS)
    resolved = OptimArgs().resolve_auto(recipes_context(spec))
    assert (resolved.optimizer, resolved.lr, resolved.weight_decay, resolved.warmup_epochs, resolved.global_clipnorm) == (
        "adamw", 3e-4, 0.01, 1, 1.0)
    assert resolved.clipnorm is None and resolved.schedule == "cosine_decay" and resolved.decay_over == "total"
    assert (recipe["epochs"], recipe["batch_size"], recipe["checkpoint_args"]) == (10, 16, {"early_stopping_patience": 5})
    assert recipe["metric_args"] == {"watch_metric": "val_loss"}


def test_the_recipe_notice_names_every_number_it_asks_for_and_says_it_is_unmeasured():
    recipe = tm.RECIPES[GPT2_LM]
    optim, patience = recipe["optim_args"], recipe["checkpoint_args"]["early_stopping_patience"]
    (level, text), = tm.recipe_notices_of(GPT2_LM)
    assert level == "info" and "unmeasured" in text and re.search(r"no language-model run", text)
    # each number is the recipe's own, so a changed recipe that forgets its notice fails here
    for words in (f"lr {optim['lr']:g}", f"batch {recipe['batch_size']}", f"weight decay {optim['weight_decay']:g}",
                  f"{recipe['epochs']} epochs", f"{optim['warmup_epochs']} warmup epoch",
                  f"gradient clip of {optim['global_clipnorm']:g}", f"patience {patience}"):
        assert words in text, words
