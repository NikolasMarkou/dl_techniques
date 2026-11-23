"""The pretrain model families of ``python -m train``: adapters, refusals, recipes and notices.

Two families train the ``pretrain`` task (their names are :data:`train.api.pretrain_args.ET` and ``BEIT``; the names
they had before, ``energy_transformer`` and ``beit``, are not aliases of them):

``et_mim``
    ``dl_techniques.models.vision.energy_transformer``: the Energy Transformer completing masked patches (method
    ``et_masked``). The model returns the pixels of every patch, the mask reaches the loss as ``sample_weight``.
``beit_mim``
    ``dl_techniques.models.vision.beit``: BEiT predicting, for the masked patches, the code ids a frozen stage-0
    tokenizer gives them (method ``beit_mim``). The model returns logits over the tokenizer's codebook.

Neither library factory takes the arguments the API calls a factory with (``factory(variant=, num_classes=,
input_shape=)``): the adapters close that gap and nothing else. Importing this module imports no model code and no keras:
each adapter imports its model inside the call, so ``python -m train --help`` allocates nothing. Registering the two specs
is the task's (``train.api.tasks.pretrain``); ``recipes.register_builtin_models`` calls :func:`builtin_pretrain_models` then.

Interface contract
------------------
``TASK``
    ``"pretrain"``, the task these families train (:data:`train.api.pretrain_args.TASK`).

``METHOD_OF``
    ``{family: method}``: ``et_masked`` and ``beit_mim``, the two methods of ``train.common.pretrain_data`` (held equal by a
    test).

``create_et / create_beit``
    ``f(*, variant, num_classes=None, input_shape, patch_size, [vocab_size,] **model_args)``
    :param variant: a key of the family's ``MODEL_VARIANTS`` (``et_tiny``, ``beit_base``) or its scale (``tiny``).
    :param num_classes: must be ``None``: the model predicts patches, it has no classes.
    :param input_shape: ``(height, width, channels)`` of ints; a flexible side is refused (the mask has one entry per patch).
    :param patch_size: a positive int dividing both sides. The task passes ``data.patch_size``: the masks and the model share
        one patch. Required.
    :param vocab_size: ``create_beit`` only, a positive int, required: the tokenizer's codebook size, read off it by the task.
    :param model_args: the keys :data:`ALLOWED` offers (the backbone constructor's arguments).
    :return: a model BUILT for the two inputs ``[image (None, H, W, C), mask (None, N)]``, so the sequence's own ``build``
        call is skipped; a call on ``[images, bool_mask]`` returns ``(B, N, P * P * C)`` pixels (ET) or ``(B, N, vocab_size)``
        logits (BEiT).
    :raises ValueError: for a non-``None`` ``num_classes``, a bad shape, patch or vocabulary, and any key :data:`REFUSED`,
        :data:`SUPPLIED` or :data:`TASK_KEYS` hold or no table knows (:func:`train.api.recipes.build_model` turns it into a
        ``ConfigError``).

``resolve_pretrain_args(context, optim_args, model_args)``
    The ``resolve_model_args`` of both families. :return: the factory keywords: the recipe's ``model_args``, then the caller's,
        without the keys the task reads itself (:data:`TASK_KEYS`: ``model_args.tokenizer_checkpoint`` of ``beit_mim``).
    :raises ConfigValidationError: naming EVERY refused key at once (``model_args.<key>`` and why; a key of the other family is
        named with that family), before the run directory is claimed.
    :raises ConfigError: for an sgd weight decay (no model regularizer is built here), a library-supplied key, an adamw
        ``kernel_regularizer``, and for ``et_mim`` a ``global_clipnorm`` that is not a number (the clip is
        load-bearing: a run that nulls it is refused by name, by :func:`train.api.recipes.require_global_clip`, which the
        ``energy_transformer`` classifier calls too).

``problems_of(family, model_args)``
    :return: one line per refusal of ``model_args`` for ``family`` (``[]`` for none). Pure.

``factory_signatures(family)`` / ``signature_census(family, signatures=None)``
    The denoising pattern: the census returns ``{"unclassified": [...], "stale": [...]}``, the parameters of the factories in
    :data:`FACTORIES` (the backbone constructor included, where the overrides land) that :data:`ALLOWED`, :data:`REFUSED` and
    :data:`SUPPLIED` do not name, and the keys those tables name that no factory has. A passing census is both empty.

``recipe_notices_of(family)``
    :return: ``[(level, text)]``, the recipe's doubts the task's ``recipe_notices`` hook prints (D-012).

``builtin_pretrain_models()``
    :return: the two :class:`~train.api.registry.ModelSpec` rows, freshly built, each with its ``init`` transfer.

Recipes
-------
Both families train the legacy trainers' optimizer (``train.energy_transformer.train_masked_completion.TrainingConfig`` and
``train.beit.train_mim.TrainingConfig``, held equal by a test that reads them): AdamW, cosine over all epochs, 100 epochs,
batch 32, weight decay 0.05, patience 15, a GLOBAL clip norm (``gradient_clipping_by_norm``, never a literal ``clipnorm``),
and ET lr 5e-4, 2 warmup epochs, clip 1.0; BEiT lr 1.5e-3, 10 warmup epochs, clip 3.0. The geometry is per dataset
(:data:`train.api.pretrain_args.GEOMETRY`) and the mask keys are the family's (:class:`~train.api.pretrain_args.PretrainData`).

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
from .pretrain_args import BEIT, DECODERS, ET, GEOMETRY, TASK, WATCH_METRIC
from .registry import ModelSpec, TransferSpec

# ---------------------------------------------------------------------

_HERE = "train.api.pretrain_models"
_ET = "dl_techniques.models.vision.energy_transformer.model"
_BEIT = "dl_techniques.models.vision.beit.model"

#: The two methods of ``train.common.pretrain_data`` by family; ``test_pretrain_families`` holds the values equal to its ``METHODS``.
METHOD_OF: Dict[str, str] = {ET: "et_masked", BEIT: "beit_mim"}

# ---------------------------------------------------------------------
# What each family offers, refuses, is given and reads
# ---------------------------------------------------------------------

_BACKBONE_NAME = "the backbone is named {name!r}, the key a warm start into the classifier transfers its weights by"
_BY_VARIANT = "the variant fixes it: choose model={family}/<variant>"

#: Keys passed through as the caller gave them: the backbone constructors' arguments the task does not decide.
ALLOWED: Dict[str, Tuple[str, ...]] = {
    ET: (
        "embed_dim", "num_heads", "head_dim", "hopfield_dim", "num_steps", "step_size", "beta", "attn_self",
        "hopfield_activation", "hopfield_beta", "noise_std", "norm_epsilon", "pos_dropout_rate", "seed",
    ),
    BEIT: (
        "hidden_size", "num_layers", "num_heads", "intermediate_size", "layer_scale_init_value", "layer_norm_eps",
        "drop_path_rate", "hidden_dropout_rate", "attention_probs_dropout_rate", "use_absolute_position_embeddings",
        "use_relative_position_bias", "use_mean_pooling", "initializer_range",
    ),
}

#: Keys of a factory that are refused whatever their value: ``{family: {key: why}}``; ``why`` ends the sentence
#: "model_args.<key> is not offered by <family>: ...".
REFUSED: Dict[str, Dict[str, str]] = {
    ET: {
        "return_energy": "the model needs the prediction head, and the energy trace builds its own probe backbone with it",
        "name": _BACKBONE_NAME.format(name="et_backbone"),
        "scale": _BY_VARIANT.format(family=ET),
    },
    BEIT: {
        "name": _BACKBONE_NAME.format(name="beit_backbone"),
        "scale": _BY_VARIANT.format(family=BEIT),
        # DECISION plan-2026-11-19T081214-5c6cc105/D-031
        # Do NOT move it to ALLOWED: the BEiT constructor raises for it, after the claim (measured; both BEiT rows refuse it).
        "use_shared_relative_position_bias": "the library does not implement it (the BEiT constructor raises)",
    },
}

#: Keywords the task or the API passes to the adapter itself: ``{family: {key: why}}``. Naming one in ``model_args`` is refused.
SUPPLIED: Dict[str, Dict[str, str]] = {
    ET: {
        "variant": f"model={ET}/<variant> chooses it",
        "input_shape": "it is (image_size, image_size, 3), from transform_args.image_size",
        "patch_size": "it is data.patch_size, which the masks and the model share",
    },
    BEIT: {
        "variant": f"model={BEIT}/<variant> chooses it",
        "input_shape": "it is (image_size, image_size, 3), from transform_args.image_size",
        "patch_size": "it is data.patch_size, which the masks and the model share",
        "vocab_size": "it is read off the tokenizer's codebook (model_args.tokenizer_checkpoint), so it cannot drift from it",
    },
}

# DECISION plan-2026-11-12T015747-46e5318f/D-007
# `tokenizer_checkpoint` is a `model_args` key of `beit_mim` that the TASK reads, not the model: it names the frozen stage-0
# tokenizer whose code ids are the targets and whose codebook size is the model's `vocab_size`. The resolver strips it from the
# factory keywords and the adapter refuses it, so no path ever reaches a constructor. It is REQUIRED for a beit_mim run; that
# check (a `.keras` path that exists, named before the claim, with the codebook usage floor) is the task's. Do NOT make it a
# field of `PretrainData`: the et_mim family has no tokenizer, and a key one family refuses by name is a model key.
TASK_KEYS: Dict[str, Tuple[str, ...]] = {ET: (), BEIT: ("tokenizer_checkpoint",)}

#: The library callables whose signatures :func:`signature_census` classifies: the factories the adapter calls and the backbone
#: constructor the ``**overrides`` of both land on.
FACTORIES: Dict[str, Tuple[str, ...]] = {
    ET: (
        f"{_ET}:create_energy_transformer_mim", f"{_ET}:create_energy_transformer_backbone",
        f"{_ET}:EnergyTransformerBackbone",
    ),
    BEIT: (f"{_BEIT}:create_beit_mim", f"{_BEIT}:create_beit_backbone", f"{_BEIT}:BeitModel"),
}

#: The families whose optimizer must clip the GLOBAL gradient norm (the others may run without a clip).
REQUIRES_GLOBAL_CLIP = (ET,)


def _offered(family: str) -> List[str]:
    """The keys of ``family`` a caller may name in ``model_args``: the factory's and the task's."""
    return sorted({*ALLOWED[family], *TASK_KEYS[family]})


def _elsewhere(family: str, key: str) -> Optional[str]:
    """The other family whose tables name ``key`` (a key it takes, supplies, reads or refuses), or ``None``."""
    for other in METHOD_OF:
        if other != family and key in {*_offered(other), *SUPPLIED[other], *REFUSED[other]}:
            return other
    return None


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
        elif key in TASK_KEYS[family]:
            lines.append(f"Invalid value for {name!r}: {key} is read by the task and is not a model keyword, got {value!r}")
        elif _elsewhere(family, key):
            lines.append(
                f"Invalid value for {name!r}: {key} is a key of the {_elsewhere(family, key)!r} family, and {family!r} "
                f"does not take it, got {value!r}"
            )
        else:
            lines.append(unknown_key_message("model_args", key, _offered(family)))
    return lines


def factory_signatures(family: str) -> Dict[str, List[str]]:
    """Return ``{callable path: named parameters}`` of the library callables behind ``family`` (``**kwargs`` left out)."""
    return {path: named_parameters(path) for path in FACTORIES[family]}


def signature_census(
    family: str, signatures: Optional[Mapping[str, Sequence[str]]] = None
) -> Dict[str, List[str]]:
    """Return the factory parameters no table names and the table keys no factory has; see the module contract."""
    if signatures is None:
        signatures = factory_signatures(family)
    offered = {*ALLOWED[family], *REFUSED[family], *SUPPLIED[family]}
    present = {name for names in signatures.values() for name in names}
    return {"unclassified": sorted(present - offered), "stale": sorted(offered - present)}


# ---------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------


def _prepare(
    family: str, num_classes: Optional[int], input_shape: Sequence[Any], patch_size: Any, model_args: Mapping[str, Any]
) -> Tuple[int, int, int]:
    """Refuse what the family does not take and return the input shape; the adapters' common door."""
    if num_classes is not None:
        raise ValueError(
            f"the {family!r} family predicts masked patches and has no classes: num_classes must be None, got {num_classes!r}"
        )
    if len(input_shape) != 3 or not all(is_count(side) for side in input_shape):
        raise ValueError(
            f"input_shape must be (height, width, channels), each a positive int (the mask has one entry per patch, so a "
            f"flexible side is refused), got {tuple(input_shape)!r}"
        )
    height, width, channels = input_shape
    if not is_count(patch_size) or height % patch_size or width % patch_size:
        raise ValueError(
            f"patch_size must be a positive int that divides both sides of input_shape {tuple(input_shape)!r}, got {patch_size!r}"
        )
    lines = problems_of(family, model_args)
    if lines:
        raise ValueError("; ".join(lines))
    return height, width, channels


def _built(model: Any, shape: Tuple[int, int, int], patch_size: int) -> Any:
    """Return ``model`` built for the two inputs ``[image, mask]``, so the run sequence's own ``build`` is skipped."""
    num_patches = (shape[0] // patch_size) * (shape[1] // patch_size)
    model.build([(None, *shape), (None, num_patches)])
    return model


def create_et(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], patch_size: int, **model_args: Any
) -> Any:
    """Build the Energy Transformer masked-completion model; see the module contract."""
    shape = _prepare(ET, num_classes, input_shape, patch_size, model_args)
    from dl_techniques.models.vision.energy_transformer.model import create_energy_transformer_mim

    return _built(create_energy_transformer_mim(variant, shape, patch_size, **model_args), shape, patch_size)


def create_beit(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], patch_size: int, vocab_size: int,
    **model_args: Any
) -> Any:
    """Build the BEiT masked-image-modelling model; see the module contract."""
    shape = _prepare(BEIT, num_classes, input_shape, patch_size, model_args)
    if not is_count(vocab_size) or vocab_size < 2:
        raise ValueError(f"vocab_size must be an int of at least 2 (the tokenizer's codebook size), got {vocab_size!r}")
    from dl_techniques.models.vision.beit.model import create_beit_mim

    return _built(create_beit_mim(variant, shape, patch_size, vocab_size=vocab_size, **model_args), shape, patch_size)


# ---------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------


def resolve_pretrain_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's; see the module contract."""
    # The adapters pass no kernel regularizer, so sgd's decay would be dropped without a word; the segmentation and denoising
    # resolvers refuse it for the same reason.
    recipes.refuse_dropped_sgd_decay(optim_args, TASK, context.family)
    if context.family in REQUIRES_GLOBAL_CLIP:
        recipes.require_global_clip(optim_args, context.family)  # the energy_transformer classifier calls it too
    merged = recipes.merge_model_args(context, optim_args, model_args, {})
    merged = {key: value for key, value in merged.items() if key not in TASK_KEYS[context.family]}
    lines = problems_of(context.family, merged)
    if lines:
        raise ConfigValidationError(lines)
    return merged


# ---------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------

#: The legacy trainers' epochs, batch and patience: both ``TrainingConfig`` s hold 100, 32 and 15.
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15


def _pretrain_recipe(
    optim_args: Dict[str, Any], model_args: Dict[str, Any], data: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a pretrain recipe: the shared shape, the family's optimizer, model keys and mask keys, per-dataset geometry."""
    return {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optim_args": dict(optim_args),
        "transform_args": {},
        "metric_args": {"watch_metric": WATCH_METRIC},
        "checkpoint_args": {"early_stopping_patience": PATIENCE},
        "model_args": dict(model_args),
        "data": dict(data),
        # The one table of sizes, `pretrain_args.GEOMETRY`, is what `pretrain_facts` reads too.
        "per_dataset": {
            name: {"transform_args": {"image_size": size}, "data": {"patch_size": patch}}
            for name, (size, patch) in GEOMETRY.items()
        },
        "per_variant": {},
    }


# train/energy_transformer/train_masked_completion.py TrainingConfig: lr 5e-4, weight_decay 0.05, warmup 2, gradient_clipping 1.0
# (LOAD-BEARING, a GLOBAL norm), `build_optimizer`: cosine over steps_per_epoch * epochs, alpha 0.01 (decay_over total), the
# builder's own betas. num_steps 12, mask_ratio 0.5, mask_token_frac 0.9. The BEiT keys are null here: the family masks by
# blocks, and a null in the recipe is what makes `PretrainData` refuse an explicit value for them.
ET_RECIPE = _pretrain_recipe(
    recipes.optim_recipe(
        optimizer="adamw", lr=5e-4, weight_decay=0.05, warmup_epochs=2, decay_over="total", global_clipnorm=1.0
    ),
    {"num_steps": 12},
    {"mask_ratio": 0.5, "mask_token_frac": 0.9, "num_mask_patches": None, "min_mask_patches_per_block": None},
)

# train/beit/train_mim.py TrainingConfig: lr 1.5e-3, weight_decay 0.05, warmup 10, gradient_clipping 3.0, drop_path_rate 0.1. The
# mask counts are NOT in the recipe: they follow the grid (`pretrain_args.auto_mask_counts`, D-008), the ET keys are null.
BEIT_RECIPE = _pretrain_recipe(
    recipes.optim_recipe(
        optimizer="adamw", lr=1.5e-3, weight_decay=0.05, warmup_epochs=10, decay_over="total", global_clipnorm=3.0
    ),
    {"drop_path_rate": 0.1},
    {"mask_ratio": None, "mask_token_frac": None},
)

RECIPES: Dict[str, Dict[str, Any]] = {ET: ET_RECIPE, BEIT: BEIT_RECIPE}


def recipe_notices_of(family: str) -> List[Tuple[str, str]]:
    """Return the ``(level, text)`` lines on the recipe's learning-rate regime; see the module contract."""
    # DECISION plan-2026-11-12T015747-46e5318f/D-012
    # The recipe is the LEGACY trainers' numbers, and both carry a regime that is doubted and not measured: the learning rate
    # is a paper's value at a far larger batch, used at batch 32 with no scaling. These are NOTICES, not changes: the legacy
    # trainer is not the arbiter (no ET or BEiT run exists to match) and nothing here measured another value. Do NOT
    # "fix" the rate or the batch to the paper's without a run that shows the better one.
    recipe = RECIPES[family]
    optim, batch = recipe["optim_args"], recipe["batch_size"]
    regime = (
        f"{family}: the recipe is the legacy trainer's (AdamW, lr {optim['lr']:g}, batch {batch}, weight decay "
        f"{optim['weight_decay']:g}, {optim['warmup_epochs']} warmup epochs, a global gradient clip of "
        f"{optim['global_clipnorm']:g}, patience {PATIENCE})"
    )
    if family == ET:
        doubt = (
            f"; lr {optim['lr']:g} is the paper's Table 4 value at a batch of 768, 24 times the batch here, and is not scaled "
            "to it: the effect on this run is unmeasured. The global clip is load-bearing (the paper reports the model fails "
            "to train at a learning rate above 1e-4 without it) and cannot be switched off"
        )
    else:
        doubt = (
            f"; lr {optim['lr']:g} is the BEiT recipe's at a global batch of 2048 (128 on 16 GPUs) and is not scaled to the "
            f"batch of {batch}: the effect on this run is unmeasured. val_loss is the mean over examples of the masked-token "
            "cross-entropy, the reference's scale, so it is not the legacy trainer's, which is that mean times the mask fraction"
        )
    return [("info", regime + doubt)]



def _pretrain_transfer(family: str) -> TransferSpec:
    """Return what ``init`` moves between two runs of the pretrain ``family`` (paths relative to the model)."""
    # DECISION plan-2026-11-19T081214-5c6cc105/D-008
    # The family's OWN runs only (no `sources`: U4 rules out a classifier source, and no other family shares a decoder);
    # the decoder is the head, reset when the tokenizer differs (init_from.HEAD_KEYS); size and patch are fixed, so
    # another geometry is refused before the claim. Do NOT add a source or drop a fixed key: no rule resizes the patch
    # kernel or a position table of these models.
    return TransferSpec(head=DECODERS[family], fixed=("channels", "input_size", "patch_size"))


def builtin_pretrain_models() -> Tuple[ModelSpec, ...]:
    """Return the two pretrain family specs, freshly built; see the module contract."""
    return (
        ModelSpec(
            family=ET,
            factory=f"{_HERE}:create_et",
            variants_attr=f"{_ET}:MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=ET_RECIPE,
            resolve_model_args=f"{_HERE}:resolve_pretrain_args",
            transfer=_pretrain_transfer(ET),
        ),
        ModelSpec(
            family=BEIT,
            factory=f"{_HERE}:create_beit",
            variants_attr=f"{_BEIT}:MODEL_VARIANTS",
            tasks=(TASK,),
            defaults=BEIT_RECIPE,
            resolve_model_args=f"{_HERE}:resolve_pretrain_args",
            transfer=_pretrain_transfer(BEIT),
        ),
    )
