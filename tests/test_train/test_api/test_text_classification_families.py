"""The classifier families ``bert`` and ``gpt2``: what their adapters offer, refuse and are given, and that padding changes no answer.

``train.api.text_models`` (plan-2026-11-24T092348-7806a290 step 9) holds two families for the ``text_classification`` task: ``bert``
(the library encoder with its sentiment head, read at the ``[CLS]`` position) and ``gpt2`` (the library decoder with the same head,
read at the last real token). The library factories do not take the arguments the API calls a factory with, so each adapter builds
a FUNCTIONAL model (a float32 id input, an in-graph cast, the backbone under a pinned name, the logits) and nothing else. Pinned here:

* the census of ``create_bert`` / ``create_gpt2`` and the two constructors: every parameter is classified (allowed, refused or
  supplied) and every classified key exists;
* every variant of both library tables builds at the ascii vocabulary (its own width and heads, one block), is functional, built,
  takes float32 ids and returns ``(None, num_classes)`` logits, with the head under the backbone;
* PADDING CHANGES NO ANSWER: the same text, right-padded to two lengths, gives equal logits for both families, the mask follows the
  pad id the adapter is given (not a constant), and the two controls that take the mask away (BERT with an all-ones mask, GPT-2
  built without ``pad_token_id``) differ by far more than the tolerance, so the assertion can fail;
* the adapters' refusals are named; the rows are data (task, variants, transfer, sources, observer); the resolver and the recipe;
* the observers read the backbone layer of a built classifier and say ``None`` for a model without one.

Tests write only under ``tmp_path``.
"""

import re

import keras
import numpy as np
import pytest

from dl_techniques.models.language.bert.model import BERT, create_bert
from dl_techniques.models.language.gpt2.gpt2 import GPT2, create_gpt2
from train.api import recipes, registry
from train.api import text_models as tm
from train.api.args import AutoContext, OptimArgs
from train.api.errors import ConfigError, ConfigValidationError
from train.api.registry import CrossTask
from train.api.text_tokenizer import tokenizer_spec

from ._text_world import ASCII_VOCAB, CLASSIFIER_ARGS, SEQ_LEN, world  # noqa: F401  (fixture)

FAMILIES = ("bert", "gpt2")
TASK, LM_TASK, LM_FAMILY = "text_classification", "language_model_pretraining", "gpt2_lm"
ASCII = tokenizer_spec("ascii95")
PAD, CLS = ASCII.pad_id, ASCII.cls_id
CLASSES = 2
#: The most two answers to the same text may differ by when only the padding length differs (float32 noise is 1e-8 here).
TOLERANCE = 1e-6
LIBRARY = {"bert": BERT.MODEL_VARIANTS, "gpt2": GPT2.MODEL_VARIANTS}
#: The table key that names each family's width, depth and the position table the library refuses a longer window by.
WIDTH = {"bert": "hidden_size", "gpt2": "embed_dim"}
DEPTH = {"bert": "num_layers", "gpt2": "depth"}
LIMIT = {"bert": "max_position_embeddings", "gpt2": "max_seq_len"}
BACKBONE = {"bert": "bert_backbone", "gpt2": "gpt2_backbone"}
#: The tokens of the probe text: none is the pad id (0) or the cls id (1), and none is 7, the other pad id the mask test uses.
TEXT = [3, 9, 11, 13, 15]


def rows():
    return {spec.family: spec for spec in tm.builtin_classifier_models()}


def adapter(family):
    return getattr(tm, f"create_{family}_classifier")


def build(family, **changes):
    """A tiny classifier of ``family`` at the ascii vocabulary; ``changes`` replace any argument or ``model_args`` key."""
    asked = {"variant": "tiny", "num_classes": CLASSES, "input_shape": (SEQ_LEN,), "vocab_size": ASCII_VOCAB, "pad_id": PAD,
             "cls_id": CLS, **CLASSIFIER_ARGS[family], **changes}
    keras.backend.clear_session()
    return adapter(family)(**asked)


def row(tokens, length, pad=PAD, cls=CLS):
    """One ``[CLS]`` + ``tokens`` + pad row of ``length`` float32 ids (what ``encode_rows`` makes, as the model takes it)."""
    ids = np.full((1, length), pad, np.float32)
    ids[0, 0] = cls
    ids[0, 1:1 + len(tokens)] = tokens
    return ids


def same_weights(first, second):
    name = BACKBONE["bert" if first.name == "bert" else "gpt2"]
    second.get_layer(name).set_weights(first.get_layer(name).get_weights())
    return second


# ---------------------------------------------------------------------
# The tables and the census
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_the_signature_census_is_empty_in_both_directions(family):
    census = tm.signature_census(family)
    assert census == {"unclassified": [], "stale": []}, census
    signatures = tm.factory_signatures(family)
    assert set(signatures) == set(tm.FACTORIES[family])
    assert "vocab_size" in signatures[next(path for path in signatures if path.endswith((":BERT", ":GPT2")))], \
        "the constructor is part of the census"


@pytest.mark.parametrize("family", FAMILIES)
def test_the_census_finds_a_parameter_no_table_names_and_a_key_no_factory_has(family):
    signatures = {"x:f": ["variant", WIDTH[family], "a_new_knob"]}
    found = tm.signature_census(family, signatures)
    assert found["unclassified"] == ["a_new_knob"]
    assert LIMIT[family] in found["stale"] and "vocab_size" in found["stale"], "keys the tables name that no factory has"
    assert "pad_id" not in found["stale"] and "cls_id" not in found["stale"], "the adapter's own arguments are not stale"


#: A sample of each table, written out here (the code's tables are read against it, not copied from it).
ALLOWED_OF = {
    "bert": {"hidden_size", "num_layers", "num_heads", "intermediate_size", "hidden_dropout_rate", "attention_probs_dropout_rate",
             "max_position_embeddings", "initializer_range", "layer_norm_eps"},
    "gpt2": {"embed_dim", "depth", "num_heads", "max_seq_len", "dropout_rate", "attention_dropout_rate", "initializer_range"},
}
REFUSED_OF = {
    "bert": {"include_top", "task_config", "head_config_overrides", "pretrained", "weights_dataset", "cache_dir", "name"},
    "gpt2": {"include_top", "task_config", "head_config_overrides", "pretrained", "name", "tie_word_embeddings", "head_type",
             "harmonic_exponent", "harmonic_eps"},
}


@pytest.mark.parametrize("family", FAMILIES)
def test_the_tables_partition_the_keys(family):
    allowed, refused, supplied = set(tm.ALLOWED[family]), set(tm.REFUSED[family]), set(tm.SUPPLIED[family])
    assert not (allowed & refused) and not (allowed & supplied) and not (refused & supplied)
    assert {"vocab_size", "variant", "input_shape", "pad_token_id"} <= supplied
    assert REFUSED_OF[family] <= refused and ALLOWED_OF[family] <= allowed
    for key, why in {**tm.REFUSED[family], **tm.SUPPLIED[family]}.items():
        assert why and not why.endswith("."), key


# ---------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------


@pytest.mark.parametrize("family, variant", [(f, v) for f in FAMILIES for v in sorted(LIBRARY[f])])
def test_every_variant_builds_at_the_ascii_vocabulary_with_its_own_width(family, variant):
    keras.backend.clear_session()  # no tiny shrink: the variant's own width and heads, one block to keep the big ones off a CPU
    model = adapter(family)(variant=variant, num_classes=CLASSES, input_shape=(SEQ_LEN,), vocab_size=ASCII_VOCAB, pad_id=PAD,
                            cls_id=CLS, **{DEPTH[family]: 1})
    backbone = model.get_layer(BACKBONE[family])
    assert model.built and isinstance(model, keras.Model) and not isinstance(model, (BERT, GPT2)) and model.name == family
    assert model.input_shape == (None, SEQ_LEN) and model.inputs[0].dtype == "float32"
    assert model.output_shape == (None, CLASSES)
    table = LIBRARY[family][variant]
    assert (getattr(backbone, WIDTH[family]), backbone.num_heads) == (table[WIDTH[family]], table["num_heads"])
    assert getattr(backbone, DEPTH[family]) == 1 and backbone.vocab_size == ASCII_VOCAB, "the vocabulary is the one given"
    assert model.weights[0].path.startswith(f"{BACKBONE[family]}/"), "init transfers by the backbone's name"
    assert any(weight.path.startswith(f"{BACKBONE[family]}/task_head/") for weight in model.weights), "the head is a variable set"


@pytest.mark.parametrize("family", FAMILIES)
def test_the_adapter_returns_the_logits_of_ids_cast_in_the_graph(family):
    model = build(family)
    first = np.array([[CLS] + list(range(3, SEQ_LEN + 2))[:SEQ_LEN - 1]], np.float32)
    second = np.array([[CLS] + list(range(40, 40 + SEQ_LEN - 1))], np.float32)
    logits = keras.ops.convert_to_numpy(model(first))
    assert logits.shape == (1, CLASSES) and logits.dtype.name == "float32"
    same = keras.ops.convert_to_numpy(model(keras.ops.cast(first, "int32")))
    assert (logits == same).all(), "a float id and the int it names give the same logits (the cast is in the graph)"
    assert not (logits == keras.ops.convert_to_numpy(model(second))).all(), "the ids reach the embedding"


@pytest.mark.parametrize("family", FAMILIES)
def test_a_model_saves_and_reloads_with_no_custom_object_and_predicts_the_same(family, tmp_path):
    model = build(family)
    batch = np.concatenate([row(TEXT, SEQ_LEN), row(TEXT[:2], SEQ_LEN)])
    before = model.predict(batch, verbose=0)
    model.save(tmp_path / "m.keras")
    loaded = keras.saving.load_model(tmp_path / "m.keras", compile=False)
    assert (loaded.predict(batch, verbose=0) == before).all()
    assert {weight.path.split("/")[0] for weight in loaded.weights} == {BACKBONE[family]}
    observe = getattr(tm, f"observe_{family}")
    assert observe(loaded) == observe(model) and observe(loaded) is not None, "a loaded file shows the layer its record is checked by"


# ---------------------------------------------------------------------
# Padding changes no answer
# ---------------------------------------------------------------------


def all_ones_bert(seq_len):
    """The control of ``bert``: the same graph with the padding mask taken away (every position attended)."""
    backbone = create_bert("tiny", vocab_size=ASCII_VOCAB, name="bert_backbone", include_top=True, pad_token_id=PAD,
                           task_config=_sentiment(), initializer_range=0.2, **CLASSIFIER_ARGS["bert"])
    inputs = keras.Input(shape=(seq_len,), dtype="float32", name="input_ids")
    ids = keras.ops.cast(inputs, "int32")
    mask = keras.ops.ones_like(ids)
    return keras.Model(inputs, backbone({"input_ids": ids, "attention_mask": mask, "token_type_ids": keras.ops.zeros_like(ids)}),
                       name="bert")


def gpt2_without_pad_id(seq_len):
    """The control of ``gpt2``: the head is not told the pad id, so its mask is all ones and ``last`` reads a pad position."""
    backbone = create_gpt2("tiny", vocab_size=ASCII_VOCAB, name="gpt2_backbone", include_top=True, task_config=_sentiment(),
                           initializer_range=0.2, **CLASSIFIER_ARGS["gpt2"])
    inputs = keras.Input(shape=(seq_len,), dtype="float32", name="input_ids")
    return keras.Model(inputs, backbone(keras.ops.cast(inputs, "int32")), name="gpt2")


def _sentiment():
    from train.common.nlp import sentiment_task_config

    return sentiment_task_config(CLASSES)


CONTROL = {"bert": all_ones_bert, "gpt2": gpt2_without_pad_id}


def gap(make, short=12, long=20, pad=PAD):
    """The largest difference between the logits of one text padded to ``short`` and to ``long`` ids, on one weight set."""
    first, second = make(short), make(long)
    same_weights(first, second)
    return float(np.abs(first(row(TEXT, short, pad=pad)).numpy() - second(row(TEXT, long, pad=pad)).numpy()).max())


@pytest.mark.parametrize("family", FAMILIES)
def test_the_same_text_padded_to_two_lengths_gives_equal_logits_and_the_controls_do_not(family):
    """M20 (the BERT mask not from the pad id) and M21 (the GPT-2 classifier without ``pad_token_id``) fail HERE."""
    def make(length):
        return build(family, input_shape=(length,), initializer_range=0.2)

    assert gap(make) <= TOLERANCE
    keras.backend.clear_session()
    control = gap(CONTROL[family])
    # the controls can fail: far over the tolerance, so an equal answer above is the padding being ignored, not a model that
    # answers alike whatever it reads
    assert control > 100 * TOLERANCE, control


@pytest.mark.parametrize("family", FAMILIES)
def test_the_mask_is_the_pad_id_the_adapter_is_given_not_a_constant(family):
    """A mask hard-wired to id 0 would pass the test above (the ascii pad id is 0) and fail this one (pad id 7)."""
    other_pad = 7
    assert other_pad not in TEXT

    def make(length):
        return build(family, input_shape=(length,), pad_id=other_pad, initializer_range=0.2)

    assert gap(make, pad=other_pad) <= TOLERANCE


# ---------------------------------------------------------------------
# What the adapters refuse, by name
# ---------------------------------------------------------------------

COMMON_REFUSALS = [
    ({"num_classes": 1}, "num_classes"), ({"num_classes": 0}, "num_classes"), ({"num_classes": True}, "num_classes"),
    ({"num_classes": None}, "num_classes"),
    ({"vocab_size": 0}, "vocab_size"), ({"vocab_size": True}, "vocab_size"),
    ({"input_shape": (SEQ_LEN, 1)}, "input_shape"), ({"input_shape": (None,)}, "input_shape"),
    ({"pad_id": -1}, "pad_id"), ({"pad_id": ASCII_VOCAB}, "pad_id"), ({"pad_id": True}, "pad_id"), ({"pad_id": None}, "pad_id"),
    ({"cls_id": ASCII_VOCAB}, "cls_id"), ({"cls_id": PAD}, "pad_id and cls_id must differ"),
    ({"include_top": False}, "include_top"), ({"task_config": {}}, "task_config"),
    ({"head_config_overrides": {"pooling_type": "mean"}}, "head_config_overrides"),
    ({"pad_token_id": 3}, "pad_token_id"), ({"pretrained": "x.keras"}, "pretrained"), ({"name": "other"}, "name"),
    ({"bogus": 1}, "bogus"),
]
OWN_REFUSALS = {
    "bert": [({"weights_dataset": "cased"}, "weights_dataset"), ({"cache_dir": "x"}, "cache_dir"),
             ({"tie_word_embeddings": False}, "tie_word_embeddings")],
    "gpt2": [({"tie_word_embeddings": False}, "tie_word_embeddings"), ({"head_type": "harmonic"}, "head_type"),
             ({"harmonic_eps": 1e-3}, "harmonic_eps"), ({"hidden_size": 64}, "hidden_size")],
}


@pytest.mark.parametrize("family, kwargs, text", [
    (family, kwargs, text) for family in FAMILIES for kwargs, text in COMMON_REFUSALS + OWN_REFUSALS[family]
])
def test_the_adapter_refuses_by_name(family, kwargs, text):
    asked = {"variant": "tiny", "num_classes": CLASSES, "input_shape": (SEQ_LEN,), "vocab_size": ASCII_VOCAB, "pad_id": PAD,
             "cls_id": CLS, **CLASSIFIER_ARGS[family], **kwargs}
    with pytest.raises(ValueError, match=text):
        adapter(family)(**asked)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_position_table_shorter_than_the_sequence_is_the_librarys_refusal_by_name(family):
    kwargs = {**CLASSIFIER_ARGS[family], LIMIT[family]: 16, "vocab_size": ASCII_VOCAB, "pad_id": PAD, "cls_id": CLS}
    keras.backend.clear_session()
    with pytest.raises(ConfigError, match=LIMIT[family]) as caught:
        recipes.build_model(rows()[family], "tiny", CLASSES, (17,), kwargs)
    assert family in str(caught.value)
    keras.backend.clear_session()
    assert recipes.build_model(rows()[family], "tiny", CLASSES, (16,), kwargs).output_shape == (None, CLASSES), \
        "the neighbouring length builds"


# ---------------------------------------------------------------------
# The rows, the resolver, the recipe and the observers
# ---------------------------------------------------------------------


def test_the_head_name_is_the_librarys():
    from dl_techniques.layers.heads.nlp import TASK_HEAD_NAME

    assert tm.HEAD_NAME == TASK_HEAD_NAME == "task_head"


def test_the_names_are_the_ones_the_plan_gives():
    from train.api import text_args

    assert (text_args.CLASSIFICATION_TASK, text_args.BERT, text_args.GPT2_CLASSIFIER) == (TASK, "bert", "gpt2")
    assert (text_args.TASK, text_args.GPT2_LM) == (LM_TASK, LM_FAMILY), "the language task and family keep their names"


@pytest.mark.parametrize("family", FAMILIES)
def test_a_row_is_data_naming_its_task_variants_adapter_resolver_observer_and_transfer(family):
    spec = rows()[family]
    assert spec.family == family and spec.tasks == (TASK,)
    assert spec.load_factory() is adapter(family)
    assert dict(spec.load_variants()) == dict(LIBRARY[family]), "the variants are the library's table, not a copy"
    assert spec.load_resolver() is tm.resolve_text_args
    assert spec.load_observer() is getattr(tm, f"observe_{family}")
    assert spec.defaults == tm.RECIPES[family]
    transfer = spec.transfer
    assert transfer is not None and transfer.head == ("task_head",) and transfer.fixed == ("vocab_size", "tokenizer")
    assert not transfer.resample and not transfer.pinned and transfer.grow is None and not transfer.positional


def test_only_the_gpt2_row_takes_a_source_and_it_is_the_language_family_whose_untied_head_it_drops():
    assert dict(rows()["bert"].transfer.sources) == {}, "a bidirectional encoder cannot be started from a causal decoder"
    assert dict(rows()["gpt2"].transfer.sources) == {(LM_TASK, LM_FAMILY): CrossTask(drop=("lm_head",))}


@pytest.mark.parametrize("family", FAMILIES)
def test_the_family_trains_this_task_only_and_every_other_family_is_refused_under_it(family, world):  # noqa: F811
    assert registry.parse_model_name(f"{family}/tiny", task=TASK) == (family, "tiny")
    for other_task in ("image_classification", LM_TASK):
        with pytest.raises(ConfigError, match=f"the family '{family}' trains {TASK}, not '{other_task}'"):
            registry.parse_model_name(f"{family}/tiny", task=other_task)
    others = sorted(set(registry.MODELS) - set(FAMILIES))
    assert len(others) >= 20 and LM_FAMILY in others, "the control: the other families are really registered"
    for name in others:
        with pytest.raises(ConfigError, match=f"the family '{name}' trains"):
            registry.parse_model_name(f"{name}/anything", task=TASK)


def context_of(spec):
    return AutoContext(family=spec.family, variant="tiny", dataset="imdb_reviews", num_classes=CLASSES, image_size=None,
                       defaults=spec.recipe("imdb_reviews", "tiny"))


@pytest.mark.parametrize("family", FAMILIES)
def test_the_resolver_names_every_refused_key_at_once(family):
    spec = rows()[family]
    context = context_of(spec)
    with pytest.raises(ConfigValidationError) as caught:
        spec.load_resolver()(context, OptimArgs().resolve_auto(context), {"include_top": True, "vocab_size": 5, "bogus": 1})
    text = str(caught.value)
    for key in ("model_args.include_top", "model_args.vocab_size", "model_args.bogus"):
        assert key in text, key
    assert "set by the task" in text and f"not offered by '{family}'" in text


@pytest.mark.parametrize("family", FAMILIES)
def test_a_resolved_model_takes_the_recipes_keys_then_the_callers(family):
    spec = rows()[family]
    context = context_of(spec)
    resolved = spec.load_resolver()(context, OptimArgs().resolve_auto(context), {LIMIT[family]: 128, **CLASSIFIER_ARGS[family]})
    assert resolved == {LIMIT[family]: 128, **CLASSIFIER_ARGS[family]}, "the recipe holds no model key, so the caller's are all"


@pytest.mark.parametrize("family", FAMILIES)
def test_an_sgd_decay_is_refused_because_no_regularizer_is_built(family):
    spec = rows()[family]
    context = context_of(spec)
    optim = OptimArgs(optimizer="sgd", lr=0.1, weight_decay=0.01, momentum=0.9, beta_1=None, beta_2=None, epsilon=None,
                      schedule="constant", warmup_epochs=0, warmup_start_lr=1e-8, min_lr_ratio=0.01,
                      decay_over="total", clipnorm=None, global_clipnorm=None)
    with pytest.raises(ConfigError, match=rf"weight_decay.*the {TASK} task builds none for '{family}'"):
        spec.load_resolver()(context, optim, {})


@pytest.mark.parametrize("family", FAMILIES)
def test_the_recipe_fills_every_auto_field_it_can_and_uses_only_known_keys(family):
    spec = rows()[family]
    recipe = spec.recipe("imdb_reviews", "tiny")
    assert set(recipe) <= set(recipes.RECIPE_KEYS), set(recipe) - set(recipes.RECIPE_KEYS)
    resolved = OptimArgs().resolve_auto(context_of(spec))
    assert (resolved.optimizer, resolved.lr, resolved.weight_decay, resolved.warmup_epochs, resolved.global_clipnorm) == (
        "adamw", 3e-4, 0.01, 1, 1.0)
    assert resolved.clipnorm is None and resolved.schedule == "cosine_decay" and resolved.decay_over == "total"
    assert (recipe["epochs"], recipe["batch_size"], recipe["checkpoint_args"]) == (8, 32, {"early_stopping_patience": 4})
    assert recipe["metric_args"] == {"watch_metric": "val_accuracy"} and recipe["model_args"] == {}


@pytest.mark.parametrize("family", FAMILIES)
def test_the_recipe_notice_names_every_number_it_asks_for_and_says_it_is_unmeasured(family):
    recipe = tm.RECIPES[family]
    optim, patience = recipe["optim_args"], recipe["checkpoint_args"]["early_stopping_patience"]
    (level, text), = tm.recipe_notices_of(family)
    assert level == "info" and "unmeasured" in text and re.search(r"no text-classification run", text) and text.startswith(family)
    for words in (f"lr {optim['lr']:g}", f"batch {recipe['batch_size']}", f"weight decay {optim['weight_decay']:g}",
                  f"{recipe['epochs']} epochs", f"{optim['warmup_epochs']} warmup epoch",
                  f"gradient clip of {optim['global_clipnorm']:g}", f"patience {patience}"):
        assert words in text, words
    assert "majority" in text, "the notice says what the bar is"


def test_the_language_notice_keeps_its_words():
    (level, text), = tm.recipe_notices_of(LM_FAMILY)
    assert level == "info" and text.startswith("gpt2_lm: no language-model run existed before this task")
    assert text.endswith("the bar is the unigram's loss, not a target for the numbers")


@pytest.mark.parametrize("family", FAMILIES)
def test_an_observer_reads_the_backbone_layer_and_says_none_for_a_model_without_one(family):
    model = build(family)
    table_keys = {"bert": ("hidden_size", "num_layers", "num_heads", "intermediate_size"),
                  "gpt2": ("embed_dim", "depth", "num_heads", "max_seq_len")}[family]
    seen = getattr(tm, f"observe_{family}")(model)
    assert seen == {**{key: LIBRARY[family]["tiny"][key] for key in table_keys},
                    **{key: value for key, value in CLASSIFIER_ARGS[family].items() if key in table_keys}}
    assert set(seen) <= set(LIBRARY[family]["tiny"]), "an observed key is a key of the variant table"
    other = {"bert": tm.observe_gpt2, "gpt2": tm.observe_bert}[family]
    assert other(model) is None, "a model of the other family is not this family"
    assert other(keras.Sequential([keras.layers.Dense(2)])) is None
    lm = tm.create_gpt2_lm(variant="tiny", input_shape=(SEQ_LEN,), vocab_size=ASCII_VOCAB, **CLASSIFIER_ARGS["gpt2"])
    assert tm.observe_bert(lm) is None and tm.observe_gpt2(lm) is not None, "the language model is a GPT2 too"
