"""The text model families of ``python -m train``: adapters, refusals, recipes and notices.

One family trains the ``language_model_pretraining`` task (its name is :data:`train.api.text_args.GPT2_LM`) and two train the
``text_classification`` task (:data:`train.api.text_args.BERT` and :data:`train.api.text_args.GPT2_CLASSIFIER`):

``gpt2_lm``
    ``dl_techniques.models.language.gpt2``: the library GPT-2 as a causal language model, trained to predict the next token of a
    window of ``seq_len`` tokens. The library model returns a dict (``logits`` and ``last_hidden_state``), which the stock
    ``fit`` cannot train, and the library's own training wrapper (``CausalLanguageModel``) owns a custom ``train_step``; so the
    adapter returns a FUNCTIONAL ``keras.Model`` over the backbone (a float32 id input, an in-graph cast, the logits) and
    ``compile`` is the stock sparse cross entropy from logits (:func:`language_loss_and_metrics`).
``bert``
    ``dl_techniques.models.language.bert``: the library encoder with its sentiment head (``train.common.nlp.sentiment_task_config``),
    read at the ``[CLS]`` position. The attention mask is derived in the graph from the tokenizer's pad id, so a right-padded row
    answers as its text does.
``gpt2``
    ``dl_techniques.models.language.gpt2``: the library decoder with the same head, read at the LAST real token (the head is
    built with the tokenizer's pad id and skips the padding). Its trunk is ``gpt2_lm``'s, which is what lets a language-model run
    seed it (``init``).

Neither library factory takes the arguments the API calls a factory with (``factory(variant=, num_classes=, input_shape=)``):
the adapters close that gap and nothing else. Importing this module imports no model code and no keras: each adapter imports
its model inside the call, so ``python -m train help`` allocates nothing. ``recipes.register_builtin_models`` calls
:func:`builtin_text_models`, which returns the language row and the two classifier rows of :func:`builtin_classifier_models`.

Interface contract
------------------
``create_gpt2_lm(*, variant, num_classes=None, input_shape, vocab_size, **model_args)``
    :param variant: a key of ``GPT2.MODEL_VARIANTS`` (``tiny``, ``small``, ``medium``, ``large``, ``xl``).
    :param num_classes: must be ``None``: the model predicts tokens, it has no classes.
    :param input_shape: ``(seq_len,)``, a positive int: the length the graph is built at (a longer ``seq_len`` than the position
        table is the library's refusal).
    :param vocab_size: the tokenizer's vocabulary size, an int of at least 2, required: never the model's default of 100,277.
    :param model_args: the keys :data:`ALLOWED` offers (the backbone constructor's arguments).
    :return: a model BUILT for ``(None, seq_len)`` float32 ids whose output is the ``(None, seq_len, vocab_size)`` logits, the
        backbone a layer named :data:`BACKBONE_NAME` (the key a warm start transfers its weights by).
    :raises ValueError: for a non-``None`` ``num_classes``, a bad shape or vocabulary, and any key :data:`REFUSED` or
        :data:`SUPPLIED` holds or no table knows (:func:`train.api.recipes.build_model` turns it into a ``ConfigError``).

``create_bert_classifier(*, variant, num_classes, input_shape, vocab_size, pad_id, cls_id, **model_args)`` / ``create_gpt2_classifier(...)``
    :param variant: a key of ``BERT.MODEL_VARIANTS`` (``tiny``, ``small``, ``base``, ``large``) / of ``GPT2.MODEL_VARIANTS``.
    :param num_classes: an int of at least 2, the width of the sentiment head (1 is the library's regression head).
    :param input_shape: ``(seq_len,)``, a positive int.
    :param vocab_size: the tokenizer's vocabulary size, an int of at least 2, required.
    :param pad_id: the tokenizer's pad id, an id of the vocabulary: the padding mask is every position that holds it (the attention
        mask of ``bert``, the pooling mask of ``gpt2``), so a text padded to any length gives the same logits.
    :param cls_id: the tokenizer's class id, an id of the vocabulary other than ``pad_id``: every row begins with it
        (``train.common.text_data.encode_rows``), and a pad id equal to it would mask that first position.
    :param model_args: the keys :data:`ALLOWED` offers for the family.
    :return: a model BUILT for ``(None, seq_len)`` float32 ids whose output is the ``(None, num_classes)`` logits, the backbone a
        layer named ``BACKBONE_NAMES[family]`` with the head under it (``HEAD_NAME``).
    :raises ValueError: for a bad ``num_classes``, shape, vocabulary or id, and any key :data:`REFUSED` or :data:`SUPPLIED` holds or
        no table knows (:func:`train.api.recipes.build_model` turns it into a ``ConfigError``).

``resolve_text_args(context, optim_args, model_args)``
    The ``resolve_model_args`` of every family. :return: the factory keywords: the recipe's ``model_args``, then the caller's.
    :raises ConfigValidationError: naming EVERY refused key at once (``model_args.<key>`` and why), before the run directory is
        claimed. :raises ConfigError: for an sgd weight decay (no model regularizer is built here).

``problems_of(family, model_args)``
    :return: one line per refusal of ``model_args`` for ``family`` (``[]`` for none). Pure.

``factory_signatures(family)`` / ``signature_census(family, signatures=None)``
    The denoising pattern: the census returns ``{"unclassified": [...], "stale": [...]}``, the parameters of the callables in
    :data:`FACTORIES` (the backbone constructor included, where the overrides land) that :data:`ALLOWED`, :data:`REFUSED` and
    :data:`SUPPLIED` do not name, and the keys those tables name that no callable has (:data:`UNSEEN_KEYS` excepted). A passing
    census is both empty.

``language_loss_and_metrics()`` / ``METRIC_NAMES``
    :return: ``(loss, metrics)`` for ``compile``: ``SparseCategoricalCrossentropy(from_logits=True)`` and the accuracy named
        :data:`ACCURACY` (the token accuracy of the language model, the class accuracy of a classifier: every text model returns
        logits and the same two objects fit both; ``METRIC_NAMES`` is the names alone, for the hook that must build nothing). The
        model returns logits, so the loss takes logits: one module holds both halves of that contract.

``observe_gpt2(model)`` / ``observe_bert(model)``
    The ``ModelSpec.observe`` of the rows: the variant-table keys the backbone layer of a loaded functional model shows
    (``embed_dim``, ``depth``, ``num_heads``, ``max_seq_len``; ``hidden_size``, ``num_layers``, ``num_heads``,
    ``intermediate_size``), ``None`` for a model with no such layer. The model is a functional wrapper, not a ``GPT2`` or a
    ``BERT``, so ``predict`` and ``export`` check a file against its record through this, not through ``isinstance``. The language
    model and the ``gpt2`` classifier both hold a ``GPT2``: what tells them apart is the output, which the record's checks read.

``recipe_notices_of(family)``
    :return: ``[(level, text)]``, the recipe's numbers that no run measured, the task's ``recipe_notices`` hook prints them.

``builtin_text_models()`` / ``builtin_classifier_models()``
    :return: the three text rows, freshly built: the language-model row and the two classifier rows (``bert``, then ``gpt2``);
        the second function alone gives the classifier rows. Each classifier row names the ``text_classification`` task,
        ``task_head`` as its head and the vocabulary and the tokenizer as its fixed keys; ``gpt2`` declares the ``gpt2_lm`` family as
        its one source and ``bert`` none (an encoder cannot start from a causal decoder).

``check_text_window(data, family, variant, kwargs)``
    The pre-claim refusals both text tasks share, from ``TaskTrainer.check_model_args``. :param data: the run's
        :class:`~train.api.text_args.TextData`; ``kwargs``: the factory keywords the resolver returned. :raises ConfigError: a
        tokenizer directory that is missing or holds files of another hash (paths and hashes only), and a ``data.seq_len`` above the
        position table of the model (``max_seq_len`` of GPT-2, ``max_position_embeddings`` of BERT), which the library would refuse
        only when the model is built, after the run directory is claimed.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import recipes
from .args import AutoContext, OptimArgs, is_count
from . import registry
from .config import unknown_key_message
from .denoising_models import named_parameters
from .errors import ConfigError, ConfigValidationError
from .registry import CrossTask, ModelSpec, TransferSpec
from .text_args import BERT, CLASSIFICATION_TASK, CLASSIFIER_METRIC, GPT2_CLASSIFIER, GPT2_LM, TASK, WATCH_METRIC, TextData
from .text_tokenizer import check_tokenizer_dir

# ---------------------------------------------------------------------

_HERE = "train.api.text_models"
GPT2_MODULE = "dl_techniques.models.language.gpt2.gpt2"
BERT_MODULE = "dl_techniques.models.language.bert.model"

#: The task each family trains (the name a refusal says).
TASK_OF = {GPT2_LM: TASK, BERT: CLASSIFICATION_TASK, GPT2_CLASSIFIER: CLASSIFICATION_TASK}

#: The name of the library backbone inside each family's functional model: the layer a warm start transfers its weights by.
BACKBONE_NAMES = {GPT2_LM: "gpt2_backbone", BERT: "bert_backbone", GPT2_CLASSIFIER: "gpt2_backbone"}
BACKBONE_NAME = BACKBONE_NAMES[GPT2_LM]

#: The library's name for a model's task head (``dl_techniques.layers.heads.nlp.TASK_HEAD_NAME``), written out because importing it
#: imports keras; a test holds the two equal. Every weight of the head lives under it, in both classifiers.
HEAD_NAME = "task_head"

#: The keys of ``GPT2.MODEL_VARIANTS`` and ``BERT.MODEL_VARIANTS`` a built backbone reports in its config (the tables'
#: ``description`` is text, not a setting).
TABLE_KEYS = ("embed_dim", "depth", "num_heads", "max_seq_len")
BERT_TABLE_KEYS = ("hidden_size", "num_layers", "num_heads", "intermediate_size")

#: The compiled metric, by name only (``metric_names`` must build nothing); ``language_loss_and_metrics`` builds it under this name.
ACCURACY = "accuracy"
METRIC_NAMES = (ACCURACY,)

# ---------------------------------------------------------------------
# What each family offers, refuses and is given
# ---------------------------------------------------------------------

_BY_VARIANT = f"the variant fixes it: choose model={GPT2_LM}/<variant>"
_FRESH_WEIGHTS = "a model is built with fresh weights; an earlier run's weights come through init"
_NO_LM_HEAD = "a classifier has no language-model head"

#: Keys passed through as the caller gave them: the backbone constructor's arguments the task does not decide.
ALLOWED: Dict[str, Tuple[str, ...]] = {
    GPT2_LM: (
        "embed_dim", "depth", "num_heads", "max_seq_len", "dropout_rate", "attention_dropout_rate", "initializer_range",
        "layer_norm_eps", "attention_type", "ffn_type", "tie_word_embeddings",
    ),
    BERT: (
        "hidden_size", "num_layers", "num_heads", "intermediate_size", "hidden_act", "hidden_dropout_rate",
        "attention_probs_dropout_rate", "max_position_embeddings", "type_vocab_size", "initializer_range", "layer_norm_eps",
        "position_embedding_type", "normalization_type", "normalization_position", "attention_type", "ffn_type",
        "use_stochastic_depth", "stochastic_depth_rate",
    ),
    GPT2_CLASSIFIER: (
        "embed_dim", "depth", "num_heads", "max_seq_len", "dropout_rate", "attention_dropout_rate", "initializer_range",
        "layer_norm_eps", "attention_type", "ffn_type",
    ),
}


def _classifier_refused(family: str, **extra: str) -> Dict[str, str]:
    """The keys every classifier refuses (the head and the name are the task's), then ``extra``: the family's own."""
    return {
        "pretrained": _FRESH_WEIGHTS,
        "include_top": "the classifier is the model with its head, which the task builds",
        "task_config": "the head is the library's sentiment head over the data's classes, which the task builds",
        "head_config_overrides": "the head's pooling is the family's (cls for bert, last for gpt2), not a knob",
        "name": f"the backbone is named {BACKBONE_NAMES[family]!r}, the key a warm start transfers its weights by",
        **extra,
    }


def _supplied(family: str, **extra: str) -> Dict[str, str]:
    """The keywords the task or the API passes to ``family``'s adapter: the variant, the window, the vocabulary and ``extra``."""
    return {
        "variant": f"model={family}/<variant> chooses it",
        "input_shape": "it is (data.seq_len,)",
        "vocab_size": "it is the tokenizer's (data.tokenizer), so it cannot drift from it, never the model's own default",
        **extra,
    }


_PAD_ID = "it is the tokenizer's pad id (data.tokenizer): the padding mask is every position that holds it"

#: Keys of a factory that are refused whatever their value: ``{family: {key: why}}``; ``why`` ends the sentence
#: "model_args.<key> is not offered by <family>: ...".
REFUSED: Dict[str, Dict[str, str]] = {
    GPT2_LM: {
        "pretrained": _FRESH_WEIGHTS,
        "include_top": "a task head replaces the language-model head, and this task trains the language model",
        "task_config": "it is read only with include_top, which this task refuses",
        "head_config_overrides": "it is read only with include_top, which this task refuses",
        "pad_token_id": "only a task head reads it, and a packed window holds no padding",
        "head_type": "the harmonic head scores tokens by distance and needs its own loss, which this task does not train with",
        "harmonic_exponent": "only the harmonic head reads it, which this task refuses",
        "harmonic_eps": "only the harmonic head reads it, which this task refuses",
        "name": f"the backbone is named {BACKBONE_NAME!r}, the key a warm start transfers its weights by",
    },
    BERT: _classifier_refused(
        BERT,
        weights_dataset="it names official weights to download, and none are distributed",
        cache_dir="it names where official weights are cached, and none are distributed",
    ),
    GPT2_CLASSIFIER: _classifier_refused(
        GPT2_CLASSIFIER,
        tie_word_embeddings=f"it ties the language-model head to the embeddings, and {_NO_LM_HEAD}",
        head_type=f"it chooses the language-model head (linear or harmonic), and {_NO_LM_HEAD}",
        harmonic_exponent=f"only the harmonic language-model head reads it, and {_NO_LM_HEAD}",
        harmonic_eps=f"only the harmonic language-model head reads it, and {_NO_LM_HEAD}",
    ),
}

#: Keywords the task or the API passes to the adapter itself: ``{family: {key: why}}``. Naming one in ``model_args`` is refused.
#: The classifiers are also given the tokenizer's pad id (``pad_token_id``, the library's name for it).
SUPPLIED: Dict[str, Dict[str, str]] = {
    GPT2_LM: _supplied(GPT2_LM),
    BERT: _supplied(BERT, pad_token_id=_PAD_ID),
    GPT2_CLASSIFIER: _supplied(GPT2_CLASSIFIER, pad_token_id=_PAD_ID),
}

#: The library callables whose signatures :func:`signature_census` classifies: the factory the adapter calls and the backbone
#: constructor the ``**overrides`` of both land on.
FACTORIES: Dict[str, Tuple[str, ...]] = {
    GPT2_LM: (f"{GPT2_MODULE}:create_gpt2", f"{GPT2_MODULE}:GPT2"),
    BERT: (f"{BERT_MODULE}:create_bert", f"{BERT_MODULE}:BERT"),
    GPT2_CLASSIFIER: (f"{GPT2_MODULE}:create_gpt2", f"{GPT2_MODULE}:GPT2"),
}

#: Keys of the tables that no signature of :data:`FACTORIES` names, so the census cannot call them stale: ``name`` is taken through
#: ``**kwargs`` (``keras.Model``'s) and ``input_shape`` is the adapter's own argument. (``pad_id`` and ``cls_id``, the classifiers'
#: other own arguments, are in no table: a ``model_args`` key of either name is refused as unknown.)
UNSEEN_KEYS: Tuple[str, ...] = ("name", "input_shape")


def problems_of(family: str, model_args: Mapping[str, Any]) -> List[str]:
    """Return one line per refusal of ``model_args`` for ``family``; see the module contract."""
    lines: List[str] = []
    for key, value in model_args.items():
        name = f"model_args.{key}"
        if key in ALLOWED[family]:
            continue
        if key in SUPPLIED[family]:
            lines.append(f"Invalid value for {name!r}: {key} is set by the task ({SUPPLIED[family][key]}), got {value!r}")
        elif key in REFUSED[family]:
            lines.append(f"Invalid value for {name!r}: {key} is not offered by {family!r} ({REFUSED[family][key]}), got {value!r}")
        else:
            lines.append(unknown_key_message("model_args", key, sorted(ALLOWED[family])))
    return lines


def factory_signatures(family: str) -> Dict[str, List[str]]:
    """Return ``{callable path: named parameters}`` of the library callables behind ``family`` (``**kwargs`` left out)."""
    return {path: named_parameters(path) for path in FACTORIES[family]}


def signature_census(
    family: str, signatures: Optional[Mapping[str, Sequence[str]]] = None
) -> Dict[str, List[str]]:
    """Return the callable parameters no table names and the table keys no callable has; see the module contract."""
    if signatures is None:
        signatures = factory_signatures(family)
    offered = {*ALLOWED[family], *REFUSED[family], *SUPPLIED[family]}
    present = {name for names in signatures.values() for name in names}
    return {"unclassified": sorted(present - offered), "stale": sorted(offered - present - set(UNSEEN_KEYS))}


# ---------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------


def _check_window(input_shape: Sequence[Any], vocab_size: Any) -> int:
    """Refuse a window that is not one positive length and a vocabulary of fewer than two ids; return the length."""
    if len(input_shape) != 1 or not is_count(input_shape[0]):
        raise ValueError(f"input_shape must be (seq_len,), one positive int, got {tuple(input_shape)!r}")
    if not is_count(vocab_size) or vocab_size < 2:
        raise ValueError(f"vocab_size must be an int of at least 2 (the tokenizer's vocabulary size), got {vocab_size!r}")
    return int(input_shape[0])


def _id_input(seq_len: int) -> Tuple[Any, Any]:
    """Return ``(inputs, ids)``: the model's float32 input of ``seq_len`` ids and the same ids cast to int32 in the graph.

    The one definition of the input of every text model (D-002, anchored in :func:`create_gpt2_lm`): float32 because the shared
    ``predict`` and ``export`` paths feed float arrays, and exact for every id below 2**24.
    """
    import keras

    inputs = keras.Input(shape=(seq_len,), dtype="float32", name="input_ids")
    return inputs, keras.ops.cast(inputs, "int32")


def create_gpt2_lm(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], vocab_size: int, **model_args: Any
) -> Any:
    """Build the functional GPT-2 language model; see the module contract."""
    if num_classes is not None:
        raise ValueError(f"the {GPT2_LM!r} family predicts tokens and has no classes: num_classes must be None, got {num_classes!r}")
    seq_len = _check_window(input_shape, vocab_size)
    lines = problems_of(GPT2_LM, model_args)
    if lines:
        raise ValueError("; ".join(lines))
    import keras

    from dl_techniques.models.language.gpt2.gpt2 import create_gpt2

    # DECISION plan-2026-11-24T092348-7806a290/D-002
    # A FUNCTIONAL model over the library backbone, trained by the stock `fit`: a float32 id input (ids below 2**24 are exact), an
    # in-graph cast to int32, the backbone's `logits`. Do NOT return the bare GPT2 (its dict output has no single loss target and
    # its metrics would carry the output's name), do NOT wrap it in `CausalLanguageModel` (a custom `train_step`, which this API
    # never uses), and do NOT make the input int32 (the shared export and predict paths feed float arrays). Measured before this
    # was written (`scratch_step4/wrapper_probe.md`): the model builds, trains, saves, reloads and predicts bit for bit.
    backbone = create_gpt2(variant, vocab_size=vocab_size, name=BACKBONE_NAME, **model_args)
    inputs, ids = _id_input(seq_len)
    return keras.Model(inputs, backbone(ids)["logits"], name=GPT2_LM)


def _check_classifier_call(
    family: str, num_classes: Any, input_shape: Sequence[Any], vocab_size: Any, pad_id: Any, cls_id: Any,
    model_args: Mapping[str, Any],
) -> int:
    """Refuse what a classifier adapter does not take (each refusal names its argument) and return ``seq_len``."""
    if not is_count(num_classes) or num_classes < 2:
        raise ValueError(
            f"num_classes must be an int of at least 2, one class per label (1 is the library's regression head), got {num_classes!r}"
        )
    seq_len = _check_window(input_shape, vocab_size)
    for name, value in (("pad_id", pad_id), ("cls_id", cls_id)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab_size:
            raise ValueError(f"{name} must be an id of the vocabulary, an int in [0, {vocab_size}), got {value!r}")
    if pad_id == cls_id:
        raise ValueError(f"pad_id and cls_id must differ, both are {pad_id}: the mask would hide the [CLS] every row begins with")
    lines = problems_of(family, model_args)
    if lines:
        raise ValueError("; ".join(lines))
    return seq_len


def create_bert_classifier(
    *, variant: str, num_classes: int, input_shape: Sequence[Any], vocab_size: int, pad_id: int, cls_id: int, **model_args: Any
) -> Any:
    """Build the functional BERT classifier; see the module contract."""
    seq_len = _check_classifier_call(BERT, num_classes, input_shape, vocab_size, pad_id, cls_id, model_args)
    import keras

    from dl_techniques.models.language.bert.model import create_bert
    from train.common.nlp import sentiment_task_config

    backbone = create_bert(variant, vocab_size=vocab_size, name=BACKBONE_NAMES[BERT], include_top=True,
                           task_config=sentiment_task_config(num_classes), pad_token_id=pad_id, **model_args)
    inputs, ids = _id_input(seq_len)
    # DECISION plan-2026-11-24T092348-7806a290/D-055
    # The padding mask is DERIVED IN THE GRAPH from the tokenizer's pad id (int32, the form of the library's own default mask) and
    # the token types are zeros: a right-padded row must answer as its text does. Do NOT call the backbone with the ids alone
    # (`pad_token_id` is advisory in BERT: padding is attended, and two pad lengths moved the logits by 4.6e-4 where the
    # mask moves them by 5.6e-9, `scratch_step9/pair_probe.md`) and do NOT write the pad id as 0 (the gpt2 tokenizer's is 50259).
    mask = keras.ops.cast(keras.ops.not_equal(ids, pad_id), "int32")
    logits = backbone({"input_ids": ids, "attention_mask": mask, "token_type_ids": keras.ops.zeros_like(ids)})
    return keras.Model(inputs, logits, name=BERT)


def create_gpt2_classifier(
    *, variant: str, num_classes: int, input_shape: Sequence[Any], vocab_size: int, pad_id: int, cls_id: int, **model_args: Any
) -> Any:
    """Build the functional GPT-2 classifier; see the module contract."""
    seq_len = _check_classifier_call(GPT2_CLASSIFIER, num_classes, input_shape, vocab_size, pad_id, cls_id, model_args)
    import keras

    from dl_techniques.models.language.gpt2.gpt2 import create_gpt2
    from train.common.nlp import sentiment_task_config

    # DECISION plan-2026-11-24T092348-7806a290/D-055
    # `pad_token_id` is the tokenizer's: the head pools the LAST real position, and without it the head's mask is all ones, so
    # `last` reads a pad position and two pad lengths moved the logits by 0.049 where this moves them by 0.0
    # (`scratch_step9/pair_probe.md`). The decoder is causal, so right padding changes no earlier hidden state and the decoder
    # needs no mask of its own. Do NOT drop `pad_token_id`, and do NOT ask for another pooling (the head's pooling is not a knob).
    backbone = create_gpt2(variant, vocab_size=vocab_size, name=BACKBONE_NAMES[GPT2_CLASSIFIER], include_top=True,
                           task_config=sentiment_task_config(num_classes), pad_token_id=pad_id, **model_args)
    inputs, ids = _id_input(seq_len)
    return keras.Model(inputs, backbone(ids), name=GPT2_CLASSIFIER)


def _observe(model: Any, layer_class: str, keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Return ``keys`` of the config of the layer of class ``layer_class`` in ``model``, ``None`` for a model without one."""
    # DECISION plan-2026-11-24T092348-7806a290/D-036
    # The loaded model is a functional wrapper, so the family check reads the backbone layer it holds (the segmentation and
    # pretrain rows do the same). Do NOT test ``isinstance(model, GPT2)``: it refuses every honest file; and do NOT compare the whole
    # config (``dropout_rate`` and the rest are free ``model_args``, the record's ``variant_overrides`` hold only the table keys).
    for layer in getattr(model, "layers", None) or []:
        if type(layer).__name__ == layer_class:
            config = layer.get_config()
            return {key: config[key] for key in keys}
    return None


def observe_gpt2(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant keys the GPT-2 layer of ``model`` shows, ``None`` for a model without one; see the module contract."""
    return _observe(model, "GPT2", TABLE_KEYS)


def observe_bert(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant keys the BERT layer of ``model`` shows, ``None`` for a model without one; see the module contract."""
    return _observe(model, "BERT", BERT_TABLE_KEYS)


def vocabulary_of(model: Any) -> Optional[int]:
    """Return the ``vocab_size`` of the GPT-2 or BERT layer ``model`` holds, ``None`` for a model without one."""
    for layer in getattr(model, "layers", None) or []:
        if type(layer).__name__ in ("GPT2", "BERT"):
            return layer.get_config()["vocab_size"]
    return None


def language_loss_and_metrics() -> Tuple[Any, List[Any]]:
    """Return ``(loss, metrics)`` of ``compile``; see the module contract."""
    import keras

    # DECISION plan-2026-11-24T092348-7806a290/D-002
    # The stock loss, from LOGITS: the model's output is the logits, and `from_logits=False` would take them for probabilities and
    # score a hundred times the loss. The library wrapper's `compute_loss` (no mask, no weights) is this same mean over every
    # position; `test_the_stock_compiled_loss_equals_the_library_causal_language_models` holds the two equal. Do NOT pass a loss of
    # our own: the unigram bar is the entropy of this loss.
    return (
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        [keras.metrics.SparseCategoricalAccuracy(name=ACCURACY)],
    )


# ---------------------------------------------------------------------
# The refusals before the claim
# ---------------------------------------------------------------------

#: The key of each family's position table: the row count a window may not pass.
POSITION_KEYS = {GPT2_LM: "max_seq_len", GPT2_CLASSIFIER: "max_seq_len", BERT: "max_position_embeddings"}


def check_text_window(data: TextData, family: str, variant: str, kwargs: Mapping[str, Any]) -> None:
    """Refuse a tokenizer directory and a window the run cannot honour, before the claim; see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-004
    # The tokenizer's files are checked HERE, before the claim, with paths and hashes only (no tensor, no tokenizer library): a
    # directory that is missing or holds a file of another hash must not cost a claimed `out`, and a run made with such a file
    # would match no record made with the pinned copy. Do NOT leave it to `build_data` (the claim is made by then) and do NOT fall
    # back to /tmp or the network when the pinned copy is gone.
    check_tokenizer_dir(data.tokenizer, data.tokenizer_dir)
    key = POSITION_KEYS[family]
    table = registry.MODELS[family].load_variants()[variant]
    if key in kwargs or key in table:
        limit = kwargs.get(key, table.get(key))
    else:  # BERT's table has no position row: the library's own default is the limit (imported here, the check runs after keras)
        from dl_techniques.models.language.bert.model import BERT as BertLayer

        limit = BertLayer.DEFAULT_MAX_POSITION_EMBEDDINGS
    if is_count(limit) and data.seq_len > limit:  # a limit that is no count is the library's refusal, at the build
        raise ConfigError(
            f"Invalid value for 'data.seq_len': {data.seq_len!r} is above the {limit} rows of the position table of "
            f"{family}/{variant} ({key}), so the model cannot be built at it; lower data.seq_len, or raise model_args.{key}"
        )


# ---------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------


def resolve_text_args(context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's; see the module contract."""
    # The adapter passes no kernel regularizer, so sgd's decay would be dropped without a word; the other resolvers refuse it too.
    recipes.refuse_dropped_sgd_decay(optim_args, TASK_OF[context.family], context.family)
    merged = recipes.merge_model_args(context, optim_args, model_args, {})
    lines = problems_of(context.family, merged)
    if lines:
        raise ConfigValidationError(lines)
    return merged


# ---------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------

EPOCHS = 10
BATCH_SIZE = 16
PATIENCE = 5

#: ``gpt2_lm``: common starting values for a small GPT-2, none measured (no language-model run existed before this task): AdamW at
#: 3e-4 with decoupled decay 0.01, one warmup epoch and a cosine decay over all epochs, a global gradient clip of 1.0.
LM_RECIPE: Dict[str, Any] = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "optim_args": recipes.optim_recipe(
        optimizer="adamw", lr=3e-4, weight_decay=0.01, warmup_epochs=1, decay_over="total", global_clipnorm=1.0
    ),
    "transform_args": {},
    "metric_args": {"watch_metric": WATCH_METRIC},
    "checkpoint_args": {"early_stopping_patience": PATIENCE},
    "model_args": {},
    "per_dataset": {},
    "per_variant": {},
}

#: ``bert`` and ``gpt2``: the same starting values for both, trained from scratch (the ``init`` arm of ``gpt2`` runs the same recipe),
#: none measured (no text-classification run existed before this task): AdamW at 3e-4 with decoupled decay 0.01, one warmup epoch
#: and a cosine decay over all epochs, a global gradient clip of 1.0, batch 32, 8 epochs, patience 4. The metric is the accuracy of
#: the selection split: the bar is a statement about it (the majority-class share plus a margin, D-076).
CLASSIFIER_RECIPE: Dict[str, Any] = {
    "epochs": 8,
    "batch_size": 32,
    "optim_args": recipes.optim_recipe(
        optimizer="adamw", lr=3e-4, weight_decay=0.01, warmup_epochs=1, decay_over="total", global_clipnorm=1.0
    ),
    "transform_args": {},
    "metric_args": {"watch_metric": CLASSIFIER_METRIC},
    "checkpoint_args": {"early_stopping_patience": 4},
    "model_args": {},
    "per_dataset": {},
    "per_variant": {},
}

RECIPES: Dict[str, Dict[str, Any]] = {GPT2_LM: LM_RECIPE, BERT: CLASSIFIER_RECIPE, GPT2_CLASSIFIER: CLASSIFIER_RECIPE}

#: What each family's notice says: the kind of run that did not exist before, the model the numbers are common starting values for,
#: and what the bar is (the notice must say that the run's result, not the bar, is what the numbers give).
_CLASSIFIER_BAR = "the majority-class share of the selection split plus three binomial standard errors of it"
_NOTICE_WORDS: Dict[str, Tuple[str, str, str]] = {
    GPT2_LM: ("language-model", "a small GPT-2", "the unigram's loss"),
    BERT: ("text-classification", "a small BERT trained from scratch", _CLASSIFIER_BAR),
    GPT2_CLASSIFIER: ("text-classification", "a small GPT-2 trained from scratch", _CLASSIFIER_BAR),
}


def recipe_notices_of(family: str) -> List[Tuple[str, str]]:
    """Return the ``(level, text)`` lines on the recipe's numbers; see the module contract."""
    recipe = RECIPES[family]
    optim = recipe["optim_args"]
    kind, starting_values_for, bar = _NOTICE_WORDS[family]
    return [(
        "info",
        f"{family}: no {kind} run existed before this task, so every number of its recipe is unmeasured: AdamW, lr "
        f"{optim['lr']:g}, batch {recipe['batch_size']}, weight decay {optim['weight_decay']:g}, {recipe['epochs']} epochs, "
        f"{optim['warmup_epochs']} warmup epoch, a global gradient clip of {optim['global_clipnorm']:g}, patience "
        f"{recipe['checkpoint_args']['early_stopping_patience']}. They are common starting values for {starting_values_for}, not "
        f"a tuned recipe: the run reports what they give, and the bar is {bar}, not a target for the numbers",
    )]


def builtin_text_models() -> Tuple[ModelSpec, ...]:
    """Return the text family specs, freshly built; see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-040
    # `fixed` is the vocabulary and the tokenizer's digest: the embedding table has one row per id and says what each id means, so
    # another vocabulary does not fit and another tokenizer would fit and be wrong; both are refused before the claim
    # (`init_identity` of the task). The window length is NOT one: no weight depends on it (the position table is `max_seq_len`
    # rows; measured: every weight copies from a source of another `seq_len`), so a source of shorter windows may seed longer ones.
    # Do NOT add it (a refusal of a transfer that works), and do NOT add `sources` (the family takes its own family only).
    # DECISION plan-2026-11-24T092348-7806a290/D-053
    # The two classifier rows are registered HERE, in the same commit as the `text_classification` TaskSpec: a family whose task
    # is not in `registry.TASKS` is a family `python -m train` lists and cannot run, and the count guard (`sum(counts) ==
    # len(MODELS)` over the tasks of `BUILTIN_TASKS`) would be red or the documents false. Do NOT remove one without the other.
    return (
        ModelSpec(
            family=GPT2_LM,
            factory=f"{_HERE}:create_gpt2_lm",
            variants_attr=f"{GPT2_MODULE}:GPT2.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=LM_RECIPE,
            resolve_model_args=f"{_HERE}:resolve_text_args",
            observe=f"{_HERE}:observe_gpt2",
            transfer=TransferSpec(fixed=("vocab_size", "tokenizer")),
        ),
        *builtin_classifier_models(),
    )


def builtin_classifier_models() -> Tuple[ModelSpec, ...]:
    """Return the two classifier family specs (``bert``, ``gpt2``), freshly built; see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-054
    # `gpt2` takes the language-model family as its one source, with the untied `lm_head` dropped (an untied source seeds it too; the
    # bare `CrossTask()` refused it, measured: `scratch_step9/pair_probe.md`); `bert` takes none (a bidirectional encoder cannot
    # start from a causal decoder). The head is `task_head`, fresh in both; the vocabulary and the tokenizer's digest are fixed,
    # as the language row's. Do NOT add `seq_len` to `fixed` (no weight depends on it) or a source for `bert`.
    fixed = ("vocab_size", "tokenizer")
    return (
        ModelSpec(
            family=BERT,
            factory=f"{_HERE}:create_bert_classifier",
            variants_attr=f"{BERT_MODULE}:BERT.MODEL_VARIANTS",
            tasks=(CLASSIFICATION_TASK,),
            defaults=CLASSIFIER_RECIPE,
            resolve_model_args=f"{_HERE}:resolve_text_args",
            observe=f"{_HERE}:observe_bert",
            transfer=TransferSpec(head=(HEAD_NAME,), fixed=fixed),
        ),
        ModelSpec(
            family=GPT2_CLASSIFIER,
            factory=f"{_HERE}:create_gpt2_classifier",
            variants_attr=f"{GPT2_MODULE}:GPT2.MODEL_VARIANTS",
            tasks=(CLASSIFICATION_TASK,),
            defaults=CLASSIFIER_RECIPE,
            resolve_model_args=f"{_HERE}:resolve_text_args",
            observe=f"{_HERE}:observe_gpt2",
            transfer=TransferSpec(head=(HEAD_NAME,), fixed=fixed, sources={(TASK, GPT2_LM): CrossTask(drop=("lm_head",))}),
        ),
    )
