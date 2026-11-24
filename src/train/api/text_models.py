"""The text model families of ``python -m train``: adapters, refusals, recipes and notices.

One family trains the ``language_model_pretraining`` task (its name is :data:`train.api.text_args.GPT2_LM`):

``gpt2_lm``
    ``dl_techniques.models.language.gpt2``: the library GPT-2 as a causal language model, trained to predict the next token of a
    window of ``seq_len`` tokens. The library model returns a dict (``logits`` and ``last_hidden_state``), which the stock
    ``fit`` cannot train, and the library's own training wrapper (``CausalLanguageModel``) owns a custom ``train_step``; so the
    adapter returns a FUNCTIONAL ``keras.Model`` over the backbone (a float32 id input, an in-graph cast, the logits) and
    ``compile`` is the stock sparse cross entropy from logits (:func:`language_loss_and_metrics`).

The library factory does not take the arguments the API calls a factory with (``factory(variant=, num_classes=, input_shape=)``):
the adapter closes that gap and nothing else. Importing this module imports no model code and no keras: the adapter imports
its model inside the call, so ``python -m train help`` allocates nothing. ``recipes.register_builtin_models`` calls
:func:`builtin_text_models`.

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

``resolve_text_args(context, optim_args, model_args)``
    The ``resolve_model_args`` of the family. :return: the factory keywords: the recipe's ``model_args``, then the caller's.
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
    :return: ``(loss, metrics)`` for ``compile``: ``SparseCategoricalCrossentropy(from_logits=True)`` and the token accuracy
        named :data:`ACCURACY` (``METRIC_NAMES`` is the names alone, for the hook that must build nothing). The model returns
        logits, so the loss takes logits: one module holds both halves of that contract.

``recipe_notices_of(family)``
    :return: ``[(level, text)]``, the recipe's numbers that no run measured, the task's ``recipe_notices`` hook prints them.

``builtin_text_models()``
    :return: the family rows, freshly built.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import recipes
from .args import AutoContext, OptimArgs, is_count
from .config import unknown_key_message
from .denoising_models import named_parameters
from .errors import ConfigValidationError
from .registry import ModelSpec, TransferSpec
from .text_args import GPT2_LM, TASK, WATCH_METRIC

# ---------------------------------------------------------------------

_HERE = "train.api.text_models"
GPT2_MODULE = "dl_techniques.models.language.gpt2.gpt2"

#: The name of the library backbone inside the functional model: the layer a warm start transfers its weights by.
BACKBONE_NAME = "gpt2_backbone"

#: The compiled metric, by name only (``metric_names`` must build nothing); ``language_loss_and_metrics`` builds it under this name.
ACCURACY = "accuracy"
METRIC_NAMES = (ACCURACY,)

# ---------------------------------------------------------------------
# What each family offers, refuses and is given
# ---------------------------------------------------------------------

_BY_VARIANT = f"the variant fixes it: choose model={GPT2_LM}/<variant>"

#: Keys passed through as the caller gave them: the backbone constructor's arguments the task does not decide.
ALLOWED: Dict[str, Tuple[str, ...]] = {
    GPT2_LM: (
        "embed_dim", "depth", "num_heads", "max_seq_len", "dropout_rate", "attention_dropout_rate", "initializer_range",
        "layer_norm_eps", "attention_type", "ffn_type", "tie_word_embeddings",
    ),
}

#: Keys of a factory that are refused whatever their value: ``{family: {key: why}}``; ``why`` ends the sentence
#: "model_args.<key> is not offered by <family>: ...".
REFUSED: Dict[str, Dict[str, str]] = {
    GPT2_LM: {
        "pretrained": "a model is built with fresh weights; an earlier run's weights come through init",
        "include_top": "a task head replaces the language-model head, and this task trains the language model",
        "task_config": "it is read only with include_top, which this task refuses",
        "head_config_overrides": "it is read only with include_top, which this task refuses",
        "pad_token_id": "only a task head reads it, and a packed window holds no padding",
        "head_type": "the harmonic head scores tokens by distance and needs its own loss, which this task does not train with",
        "harmonic_exponent": "only the harmonic head reads it, which this task refuses",
        "harmonic_eps": "only the harmonic head reads it, which this task refuses",
        "name": f"the backbone is named {BACKBONE_NAME!r}, the key a warm start transfers its weights by",
    },
}

#: Keywords the task or the API passes to the adapter itself: ``{family: {key: why}}``. Naming one in ``model_args`` is refused.
SUPPLIED: Dict[str, Dict[str, str]] = {
    GPT2_LM: {
        "variant": f"model={GPT2_LM}/<variant> chooses it",
        "input_shape": "it is (data.seq_len,)",
        "vocab_size": "it is the tokenizer's (data.tokenizer), so it cannot drift from it, never the model's own default",
    },
}

#: The library callables whose signatures :func:`signature_census` classifies: the factory the adapter calls and the backbone
#: constructor the ``**overrides`` of both land on.
FACTORIES: Dict[str, Tuple[str, ...]] = {
    GPT2_LM: (f"{GPT2_MODULE}:create_gpt2", f"{GPT2_MODULE}:GPT2"),
}

#: Keys of the tables that no signature of :data:`FACTORIES` names, so the census cannot call them stale: ``name`` is taken through
#: ``**kwargs`` (``keras.Model``'s) and ``input_shape`` is the adapter's own argument.
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
# The adapter
# ---------------------------------------------------------------------


def create_gpt2_lm(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], vocab_size: int, **model_args: Any
) -> Any:
    """Build the functional GPT-2 language model; see the module contract."""
    if num_classes is not None:
        raise ValueError(f"the {GPT2_LM!r} family predicts tokens and has no classes: num_classes must be None, got {num_classes!r}")
    if len(input_shape) != 1 or not is_count(input_shape[0]):
        raise ValueError(f"input_shape must be (seq_len,), one positive int, got {tuple(input_shape)!r}")
    if not is_count(vocab_size) or vocab_size < 2:
        raise ValueError(f"vocab_size must be an int of at least 2 (the tokenizer's vocabulary size), got {vocab_size!r}")
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
    inputs = keras.Input(shape=(int(input_shape[0]),), dtype="float32", name="input_ids")
    logits = backbone(keras.ops.cast(inputs, "int32"))["logits"]
    return keras.Model(inputs, logits, name=GPT2_LM)


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
# The resolver
# ---------------------------------------------------------------------


def resolve_text_args(context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's; see the module contract."""
    # The adapter passes no kernel regularizer, so sgd's decay would be dropped without a word; the other resolvers refuse it too.
    recipes.refuse_dropped_sgd_decay(optim_args, TASK, context.family)
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

RECIPES: Dict[str, Dict[str, Any]] = {GPT2_LM: LM_RECIPE}


def recipe_notices_of(family: str) -> List[Tuple[str, str]]:
    """Return the ``(level, text)`` lines on the recipe's numbers; see the module contract."""
    recipe = RECIPES[family]
    optim = recipe["optim_args"]
    return [(
        "info",
        f"{family}: no language-model run existed before this task, so every number of its recipe is unmeasured: AdamW, lr "
        f"{optim['lr']:g}, batch {recipe['batch_size']}, weight decay {optim['weight_decay']:g}, {recipe['epochs']} epochs, "
        f"{optim['warmup_epochs']} warmup epoch, a global gradient clip of {optim['global_clipnorm']:g}, patience "
        f"{recipe['checkpoint_args']['early_stopping_patience']}. They are common starting values for a small GPT-2, not a tuned "
        "recipe: the run reports what they give, and the bar is the unigram's loss, not a target for the numbers",
    )]


def builtin_text_models() -> Tuple[ModelSpec, ...]:
    """Return the text family specs, freshly built; see the module contract."""
    # The empty transfer is the row's declaration (every built-in row has one): the task does not override `init_identity` yet, so
    # a request with `init` is refused before the claim, naming the task.
    return (
        ModelSpec(
            family=GPT2_LM,
            factory=f"{_HERE}:create_gpt2_lm",
            variants_attr=f"{GPT2_MODULE}:GPT2.MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=LM_RECIPE,
            resolve_model_args=f"{_HERE}:resolve_text_args",
            transfer=TransferSpec(),
        ),
    )
