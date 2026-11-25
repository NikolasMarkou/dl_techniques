"""The ``beit_tokenizer`` model family of ``python -m train``: the adapter, the refusal tables, the resolver, the recipe and the row.

One family trains the ``beit_tokenizer`` task (its name is :data:`train.api.tokenizer_args.FAMILY`):

``vq_vae_rotation``
    ``dl_techniques.models.vision.vq_vae_rotation``: the library ``VQVAERotationTrick``, the BEiT stage-0 discrete tokenizer. Its auto
    encoder is a stack of stride-2 convolutions (the total stride is ``data.patch_size``), its quantizer the rotation trick, its decoder
    ends in a linear layer, and the class owns the ``train_step`` and ``test_step`` that train and score it (the reconstruction term and
    the quantizer's own). It is the class ``beit_mim`` loads as its frozen tokenizer, so no wrapper stands in for it and the task adds
    no step: the file this family saves is the file ``beit_mim`` reads.

The library has no named scales (its README says so), and the registry's row needs a variants table, so :data:`VARIANTS` is the API's: one
row, ``small``, the legacy trainer's widths. Importing this module imports no model code and no keras: the adapter imports the model inside
the call, so ``python -m train help`` allocates nothing. The row is registered with its task and not before (D-031).

Interface contract
------------------
``create_vq_tokenizer(*, variant, num_classes=None, input_shape, patch_size, **model_args)``
    :param variant: a key of :data:`VARIANTS` (``small``).
    :param num_classes: must be ``None``: the model reconstructs images, it has no classes.
    :param input_shape: ``(height, width, channels)`` of positive ints.
    :param patch_size: the side of the cell one code describes: a power of 2 dividing both sides. The task passes ``data.patch_size``.
        Required.
    :param model_args: the keys :data:`ALLOWED` offers; ``num_embeddings`` (the codebook size) is required (the recipe gives it per
        dataset, :data:`CODEBOOK`). A key of the variant's table (``hidden_channels``, ``num_res_blocks``, ``embedding_dim``) given here
        replaces the table's.
    :return: a model BUILT by a real forward pass at ``input_shape`` (so the sequence's own ``build`` call is skipped), whose
        ``encode_to_indices`` of an ``(N, height, width, channels)`` batch has shape ``(N, height // patch_size, width // patch_size)``:
        that grid is PROVED by running it, never read off the model's own ``downsample_factor``.
    :raises ValueError: for a non-``None`` ``num_classes``, an unknown variant, a bad shape or patch, a missing ``num_embeddings``, any key
        :data:`REFUSED` or :data:`SUPPLIED` holds or no table knows, and a code grid that is not the image over the patch
        (:func:`train.api.recipes.build_model` turns it into a ``ConfigError``).

``resolve_tokenizer_args(context, optim_args, model_args)``
    The ``resolve_model_args`` of the family. :return: the factory keywords: the recipe's ``model_args`` (the codebook size of the
        dataset), then the caller's.
    :raises ConfigValidationError: naming EVERY refused key at once (``model_args.<key>`` and why), before the run directory is claimed.
    :raises ConfigError: for an sgd weight decay (no model regularizer is built here), a library-supplied key, an adamw
        ``kernel_regularizer``.

``problems_of(model_args)``
    :return: one line per refusal of ``model_args`` (``[]`` for none). Pure.

``factory_signatures()`` / ``signature_census(signatures=None)``
    The denoising pattern: the census returns ``{"unclassified": [...], "stale": [...]}``, the parameters of the constructor in
    :data:`FACTORIES` that :data:`ALLOWED`, :data:`REFUSED` and :data:`SUPPLIED` do not name, and the keys those tables name that it has
    not (:data:`UNSEEN_KEYS` excepted). A passing census is both empty.

``precision_problems(precision, prefix="")``
    :return: one line naming ``<prefix>.precision`` when ``precision`` is not ``float32`` (the quantizer's nearest-code search and its
        codebook update were never measured in a mixed precision), ``[]`` for ``float32``. For the task config's ``problems``, which names the field.

``recipe_notices_of(dataset)``
    :return: ``[(level, text)]``, the recipe's numbers no run of this task measured, which the task's ``recipe_notices`` hook prints.

``builtin_tokenizer_models()``
    :return: the one :class:`~train.api.registry.ModelSpec` row, freshly built, with its ``init`` transfer (size, patch and channels are
        fixed, so a source of another geometry is refused before the claim).

Recipe
------
The legacy trainer's (``train.beit.train_tokenizer.TrainingConfig``, held equal by a test that reads it): AdamW, cosine over all epochs,
50 epochs, batch 32, lr 5e-4, 2 warmup epochs, weight decay 0.05, a GLOBAL clip of 1.0 (``global_clipnorm``, never a literal ``clipnorm``),
patience 15. The geometry is per dataset (:data:`train.api.pretrain_args.GEOMETRY`) with the codebook size (:data:`CODEBOOK`).

Import rule
-----------
Module scope imports the standard library and ``train.api`` only.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import recipes
from .args import AutoContext, OptimArgs, is_count
from .config import join_path, unknown_key_message
from .denoising_models import named_parameters
from .errors import ConfigValidationError
from .pretrain_args import GEOMETRY
from .registry import ModelSpec, TransferSpec
from .tokenizer_args import FAMILY, TASK, WATCH_METRIC, geometry_problem

# ---------------------------------------------------------------------

_HERE = "train.api.tokenizer_models"
_MODEL = "dl_techniques.models.vision.vq_vae_rotation.model"

#: The one named scale: the legacy trainer's widths (``TrainingConfig`` defaults), the gate's model. The keys are constructor arguments of
#: the model, so a caller's ``model_args`` replaces them (``small`` is then a name, not a size).
VARIANTS: Dict[str, Dict[str, int]] = {"small": {"hidden_channels": 128, "num_res_blocks": 2, "embedding_dim": 32}}

# ---------------------------------------------------------------------
# What the family offers, refuses and is given
# ---------------------------------------------------------------------

# DECISION plan-2026-11-25T124419-fa94f59c/D-033
# Six keys are offered, the legacy trainer's minus the geometry one (`data.patch_size` is that); every other constructor parameter is
# REFUSED by name before the claim, each with its reason. Do NOT move `kmeans_init`, `dead_code_threshold` or `reconstruction_loss_weight`
# to ALLOWED "for experiments": the first two are the remedies for a collapsed codebook that no run measured through this task (the bar
# reports a collapse and does not repair it), and a weight other than 1 scales `val_reconstruction_loss` while the constant-image
# baseline stays unweighted. A table key is also a claim that the task works with it, and no run says so. The tables are FLAT (one family).
#: Keys passed through as the caller gave them: the constructor's arguments the task does not decide.
ALLOWED: Tuple[str, ...] = (
    "num_embeddings", "embedding_dim", "hidden_channels", "num_res_blocks", "commitment_cost", "use_ema",
)

_UNMEASURED = "no run of this task measured it"
_ONLY_WITH_KMEANS = "it is read only with kmeans_init, which is not offered"

#: Keys of the constructor that are refused whatever their value: ``{key: why}``; ``why`` ends the sentence
#: "model_args.<key> is not offered by <family>: ...".
REFUSED: Dict[str, str] = {
    "encoder": "the task builds the auto encoder, whose total stride is data.patch_size, so the code grid is the one the patches give",
    "decoder": "the task builds the auto decoder, which ends in a linear layer that reaches a standardized image; no run measured another",
    "gradient_mode": f"the recipe is the rotation trick, and {_UNMEASURED} with another gradient estimator",
    "distance_mode": f"the recipe is the euclidean distance, and {_UNMEASURED} with another",
    "ema_decay": f"it is read only with use_ema, and {_UNMEASURED} with another decay (the library's stays)",
    "num_heads": "a multi-head codebook gives several codes per cell, (batch, rows, columns, heads), and beit_mim reads one code per cell",
    "kmeans_init": f"the k-means warm start is a remedy for a collapsed codebook, and {_UNMEASURED}; the bar reports a collapse",
    "kmeans_init_steps": _ONLY_WITH_KMEANS,
    "kmeans_seed": _ONLY_WITH_KMEANS,
    "dead_code_threshold": f"the dead-code reset is a remedy for a collapsed codebook, and {_UNMEASURED}; the bar reports a collapse",
    "diversity_coefficient": f"it adds a loss term, and {_UNMEASURED}",
    "orthogonal_reg_coefficient": f"it adds a loss term, and {_UNMEASURED}",
    "reconstruction_loss_weight": (
        "the bar compares the reconstruction term unweighted: a weight other than 1 scales val_reconstruction_loss, and the "
        "constant-image baseline it is compared with is not scaled"
    ),
    "quantizer_initializer": f"{_UNMEASURED} with another codebook initializer (the library's is the start the recipe was run with)",
    "norm_type": f"{_UNMEASURED} with another normalization (the library's is the start the recipe was run with)",
}

#: Keywords the task or the API passes to the adapter itself: ``{key: why}``. Naming one in ``model_args`` is refused.
SUPPLIED: Dict[str, str] = {
    "variant": f"model={FAMILY}/<variant> chooses it",
    "input_shape": "it is (image_size, image_size, 3), from transform_args.image_size",
    "patch_size": "it is data.patch_size, the side of the cell one code describes",
    "downsample_factor": "it is data.patch_size: the encoder's total stride is the patch the code grid is cut by",
}

#: The library constructor whose signature :func:`signature_census` classifies; the adapter calls the CLASS (the library's
#: ``create_vq_vae_rotation`` has defaults of its own, 512 codes and a 64-wide embedding, which the recipe must not inherit).
FACTORIES: Tuple[str, ...] = (f"{_MODEL}:VQVAERotationTrick",)

#: Keys of the tables that the constructor has no parameter of, so the census cannot call them stale: the adapter's own arguments.
UNSEEN_KEYS: Tuple[str, ...] = ("variant", "patch_size")


def problems_of(model_args: Mapping[str, Any]) -> List[str]:
    """Return one line per refusal of ``model_args``; see the module contract."""
    lines: List[str] = []
    for key, value in model_args.items():
        name = f"model_args.{key}"
        if key in ALLOWED:
            continue
        if key in SUPPLIED:
            lines.append(f"Invalid value for {name!r}: {key} is set by the task ({SUPPLIED[key]}), got {value!r}")
        elif key in REFUSED:
            lines.append(f"Invalid value for {name!r}: {key} is not offered by {FAMILY!r} ({REFUSED[key]}), got {value!r}")
        else:
            lines.append(unknown_key_message("model_args", key, sorted(ALLOWED)))
    return lines


def factory_signatures() -> Dict[str, List[str]]:
    """Return ``{callable path: named parameters}`` of the library constructor behind the family (``**kwargs`` left out)."""
    return {path: named_parameters(path) for path in FACTORIES}


def signature_census(signatures: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, List[str]]:
    """Return the constructor parameters no table names and the table keys it has no parameter of; see the module contract."""
    if signatures is None:
        signatures = factory_signatures()
    offered = {*ALLOWED, *REFUSED, *SUPPLIED}
    present = {name for names in signatures.values() for name in names}
    return {"unclassified": sorted(present - offered), "stale": sorted(offered - present - set(UNSEEN_KEYS))}


def precision_problems(precision: str, prefix: str = "") -> List[str]:
    """Return the line that refuses a ``precision`` other than ``float32``; see the module contract."""
    if precision == "float32":
        return []
    return [
        f"Invalid value for {join_path(prefix, 'precision')!r}: expected float32 for {TASK} (the quantizer's nearest-code search and "
        f"its codebook update were never measured in a mixed precision), got {precision!r}"
    ]


# ---------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------


def create_vq_tokenizer(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], patch_size: int, **model_args: Any
) -> Any:
    """Build the tokenizer at ``input_shape`` and prove its code grid; see the module contract."""
    if num_classes is not None:
        raise ValueError(f"the {FAMILY!r} family reconstructs images and has no classes: num_classes must be None, got {num_classes!r}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} of {FAMILY!r}; the variants are {sorted(VARIANTS)}")
    shape = tuple(input_shape)
    if len(shape) != 3 or not all(is_count(side) for side in shape):
        raise ValueError(f"input_shape must be (height, width, channels), each a positive int, got {shape!r}")
    for side in shape[:2]:
        reason = geometry_problem(patch_size, side)
        if reason is not None:
            raise ValueError(f"patch_size {patch_size!r} {reason} (input_shape {shape!r})")
    lines = problems_of(model_args)
    if lines:
        raise ValueError("; ".join(lines))
    if "num_embeddings" not in model_args:
        raise ValueError("num_embeddings is required: the codebook size has no default here (the recipe gives it per dataset)")
    import keras

    from dl_techniques.models.vision.vq_vae_rotation.model import VQVAERotationTrick

    # DECISION plan-2026-11-25T124419-fa94f59c/D-032
    # The library CLASS is built directly, with the encoder's stride set to the patch, and the code grid is PROVED by running the
    # model: `beit_mim` reads the grid as the patch grid of its own run, and a grid that does not line up trains on targets read from
    # the wrong cells and still reports a finite, plausible loss (the legacy trainer's `build_tokenizer` does the same). Do NOT take
    # the expected grid from the model's own `downsample_factor` (it would agree with itself by construction), do NOT leave the
    # stride at the library's default of 4 (right for cifar10 by luck, wrong at every other patch), and do NOT wrap the model: this
    # is the class `beit_mim` loads, with its own train and test step, so the task adds none.
    model = VQVAERotationTrick(input_shape=shape, downsample_factor=patch_size, **{**VARIANTS[variant], **model_args})
    probe = keras.ops.zeros((1, *shape), dtype="float32")
    model(probe, training=False)
    grid = tuple(int(side) for side in model.encode_to_indices(probe).shape[1:])
    expected = (shape[0] // patch_size, shape[1] // patch_size)
    if grid != expected:
        raise ValueError(
            f"the code grid of the built model is {grid}, but a patch_size of {patch_size} on an image of {shape} gives {expected}: "
            "beit_mim would read every target from the wrong cell"
        )
    return model


# ---------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------


def resolve_tokenizer_args(context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's; see the module contract."""
    # The adapter passes no kernel regularizer, so sgd's decay would be dropped without a word; the other resolvers refuse it too.
    recipes.refuse_dropped_sgd_decay(optim_args, TASK, context.family)
    merged = recipes.merge_model_args(context, optim_args, model_args, {})
    lines = problems_of(merged)
    if lines:
        raise ConfigValidationError(lines)
    return merged


# ---------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------

#: The legacy trainer's epochs, batch and patience (``TrainingConfig``); the gate passes ``epochs=5``, as the 4h tokenizer had.
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 15

# DECISION plan-2026-11-25T124419-fa94f59c/D-034
# The codebook is per dataset, and cifar10's row is NOT the legacy command line's: that gives cifar10 a down-sample of 8 (a 4 x 4 grid,
# which `beit_mim` refuses at its patch of 4) and 8,192 codes, and nobody ran it. The row is the pairing every `beit_mim` gate used
# (patch 4, an 8 x 8 grid, 512 codes); imagenette's is the legacy default at its geometry (224 px, patch 16, 8,192 codes), never run in
# this repository (its notice says so). Do NOT take the legacy command line's numbers for cifar10, and do NOT put the codebook in the
# variant (the model has no scale that changes with the dataset).
#: Codes in the codebook, per dataset: ``num_embeddings`` of the recipe.
CODEBOOK: Dict[str, int] = {"cifar10": 512, "imagenette": 8192}

RECIPE: Dict[str, Any] = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    # train/beit/train_tokenizer.py TrainingConfig: lr 5e-4, weight_decay 0.05, warmup 2, gradient_clipping 1.0 (a GLOBAL norm),
    # `build_optimizer`: cosine over steps_per_epoch * epochs, alpha 0.01 (decay_over total), the builder's own betas.
    "optim_args": recipes.optim_recipe(
        optimizer="adamw", lr=5e-4, weight_decay=0.05, warmup_epochs=2, decay_over="total", global_clipnorm=1.0
    ),
    "transform_args": {},
    "metric_args": {"watch_metric": WATCH_METRIC},
    "checkpoint_args": {"early_stopping_patience": PATIENCE},
    "model_args": {},
    "data": {},
    # The one table of sizes, `pretrain_args.GEOMETRY`, is what `tokenizer_facts` reads too.
    "per_dataset": {
        name: {
            "transform_args": {"image_size": size}, "data": {"patch_size": patch}, "model_args": {"num_embeddings": CODEBOOK[name]},
        }
        for name, (size, patch) in GEOMETRY.items()
    },
    "per_variant": {},
}


def recipe_notices_of(dataset: str) -> List[Tuple[str, str]]:
    """Return the ``(level, text)`` lines on the recipe's numbers for ``dataset``; see the module contract."""
    optim = RECIPE["optim_args"]
    notices = [(
        "info",
        f"{FAMILY}: the recipe is the legacy tokenizer trainer's (AdamW, lr {optim['lr']:g}, batch {BATCH_SIZE}, weight decay "
        f"{optim['weight_decay']:g}, {optim['warmup_epochs']} warmup epochs, a global gradient clip of {optim['global_clipnorm']:g}, "
        f"patience {PATIENCE}, {EPOCHS} epochs). Its effect with this task's decoder, a linear last layer, is unmeasured: whether the "
        "codes reach the perplexity floor is the run's own verdict (the bar), not a promise of these numbers",
    )]
    if dataset == "imagenette":
        size, patch = GEOMETRY[dataset]
        notices.append((
            "info",
            f"{dataset}: {size} px, patch {patch} (a {size // patch} x {size // patch} grid) and {CODEBOOK[dataset]:,} codes are the "
            "legacy trainer's defaults for it, and no run in this repository has trained them: this run is their first measurement "
            "(time, memory and whether the codes reach the floor)",
        ))
    return notices


def builtin_tokenizer_models() -> Tuple[ModelSpec, ...]:
    """Return the one tokenizer family spec, freshly built; see the module contract."""
    # DECISION plan-2026-11-25T124419-fa94f59c/D-031
    # The row is built and tested HERE and registered with its task: `recipes.register_builtin_models` calls this function in the commit
    # that adds the `beit_tokenizer` TaskSpec (step 6). A family whose task is not in `registry.TASKS` is a family `python -m train
    # list_models` lists and no command can run, and the count guard (`sum(counts) == len(MODELS)` over the registered tasks) goes red
    # or the documents false. Do NOT add the call earlier, and do NOT remove one without the other.
    # `fixed` is the geometry a weight depends on: the stem's kernel (channels), the grid the quantizer sees (size and patch). The
    # codebook size is the init engine's geometry key (step 8). The model has no head: the whole network is the thing `init` moves.
    return (
        ModelSpec(
            family=FAMILY,
            factory=f"{_HERE}:create_vq_tokenizer",
            variants_attr=f"{_HERE}:VARIANTS",
            tasks=(TASK,),
            defaults=RECIPE,
            resolve_model_args=f"{_HERE}:resolve_tokenizer_args",
            transfer=TransferSpec(fixed=("channels", "input_size", "patch_size")),
        ),
    )
