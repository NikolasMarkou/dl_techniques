"""The ``dino`` model family of ``python -m train``: the adapter, the refusal tables, the resolver, the recipe and the row.

One family trains the ``dino`` task (its name is :data:`train.api.dino_args.FAMILY`):

``dino_v1``
    ``dl_techniques.models.vision.dino``: the library ``DINOTrainingModel``, a student and an EMA teacher ``DINOv1`` with projection heads. One
    forward pass takes ``(batch, 2 + n_local_crops, H, W, C)`` views and returns the packed ``(batch * n_pairs, 2 * out_dim)`` logits the
    ``DINOLoss`` unpacks; the loss owns the centre, the teacher moves by EMA (a callback of the task), and the model has no ``train_step`` here.

The registry's row needs a variants table, and the library's ``DINOv1.MODEL_VARIANTS`` holds five (base, large and giant included), so
:data:`VARIANTS` is the API's: ``tiny`` and ``small``, with the keys the loaded model shows (a test holds them equal to the library's). Importing
this module imports no model code and no keras: the adapter imports the model inside the call, so ``python -m train help`` allocates nothing. The row is
registered with its task and not before (D-043).

Interface contract
------------------
``create_dino_model(*, variant, num_classes=None, input_shape, patch_size, n_local_crops, dino_out_dim, **model_args)``
    :param variant: a key of :data:`VARIANTS` (``tiny``, ``small``).
    :param num_classes: must be ``None``: a DINO model has no classes.
    :param input_shape: ``(size, size, 3)`` of positive ints: ONE plain image, which every view (the local crops too) is rendered at.
    :param patch_size: a power of 2 dividing the side and leaving at least ``dino_args.MIN_TOKENS`` tokens per view. The task passes ``data.patch_size``.
    :param n_local_crops: local crops per image, at least 1. The task passes ``data.n_local_crops``.
    :param dino_out_dim: the projection head's width, an int of at least 2. Required: the library's 65536 does not fit the recipe's devices.
    :param model_args: nothing else is accepted (:data:`ALLOWED` holds three more keys, which the task reads and the model does not).
    :return: a model BUILT with the rank-5 input ``(None, 2 + n_local_crops, size, size, 3)`` (the sequence's own ``build((None, *input_shape))``
        is refused by this model, so the factory builds it), whose view count and patch grid are PROVED by real forward passes: the packed
        output has ``n_pairs`` rows of ``2 * out_dim`` and the patch layer gives ``(size // patch_size) ** 2`` tokens, each compared with what
        the arguments say, never read off the model's own attributes.
    :raises ValueError: for a non-``None`` ``num_classes``, an unknown variant, a bad shape, patch, crop count or head width, any key
        :data:`REFUSED`, :data:`SUPPLIED` or :data:`TASK_KEYS` holds or no table knows, and a grid or view count the proof does not confirm
        (:func:`train.api.recipes.build_model` turns it into a ``ConfigError``).
    :raises TypeError: for a missing ``patch_size``, ``n_local_crops`` or ``dino_out_dim`` (the task's keywords are required, not defaulted).

``resolve_dino_args(context, optim_args, model_args)`` / ``task_values(context, optim_args, model_args)``
    The ``resolve_model_args`` of the family and its companion. Both lay the recipe's ``model_args`` under the caller's and check the result.
    :return: the factory keywords (``dino_out_dim``) / the task-read values (``teacher_temp``, ``teacher_temp_final``, ``ema_warmup_epochs``).
    :raises ConfigValidationError: naming EVERY refused key at once (``model_args.<key>`` and why), before the run directory is claimed.
    :raises ConfigError: for an sgd weight decay (no model regularizer is built here), a library-supplied key, an adamw ``kernel_regularizer``.

``problems_of(model_args)``
    :return: one line per refusal of ``model_args`` (``[]`` for none): a key that is not offered, a value of an offered key that is not of the
        type the task can use (a bool is no int and no float) or out of range, and a ``teacher_temp_final`` below ``teacher_temp``. Pure.

``factory_signatures()`` / ``signature_census(signatures=None)``
    The denoising pattern: the census returns ``{"unclassified": [...], "stale": [...]}``, the parameters of the callables in :data:`FACTORIES`
    that :data:`ALLOWED`, :data:`REFUSED` and :data:`SUPPLIED` do not name, and the keys those tables name that none has (:data:`UNSEEN_KEYS` excepted).

``precision_problems(precision, prefix="")`` / ``schedule_problems(schedule, prefix="optim_args")`` / ``backup_problems(backup_every_steps, prefix="checkpoint_args")``
    :return: the lines that refuse a ``precision`` other than ``float32``, a ``constant`` schedule, and any ``backup_every_steps``; ``[]`` when
        the value is the accepted one. For the task config's ``problems``, which names the field (``prefix`` is the dotted path of its group).

``warmup_problems(ema_warmup_epochs, epochs, smoke=False)``
    :return: the line that refuses an ``ema_warmup_epochs`` that is not below the run's resolved ``epochs`` (a teacher frozen for the whole run);
        ``[]`` for a smaller warmup and for a smoke run (one epoch, no bar). The task calls it from ``check_model_args``, which sees the resolved epochs.

``ema_warmup_steps(ema_warmup_epochs, steps_per_epoch)``
    :return: the teacher EMA's warmup in optimizer steps, ``round(epochs * steps_per_epoch)``: the recipe is denominated in EPOCHS, so it moves
        with the dataset and the batch (a test holds it equal to the legacy trainer's ``resolve_ema_warmup_steps``).

``recipe_notices_of(dataset)``
    :return: ``[(level, text)]``, what the recipe is and what the gate covers, which the task's ``recipe_notices`` hook prints.

``observe_dino(model)``
    The row's ``ModelSpec.observe``. :return: the variant-table keys (:data:`TABLE_KEYS`) of the loaded model's STUDENT ``get_config()`` when the
        model has a ``student`` (a ``DINOTrainingModel``), ``None`` for a model of any other class.

``builtin_dino_models()``
    :return: the one :class:`~train.api.registry.ModelSpec` row, freshly built, with the empty ``init`` declaration (a dino run is no ``init``
        target) and its ``observe``.

Recipe
------
The legacy trainer's (``train.dino.train_dino.TrainingConfig``, held equal at the shared fields by a test that reads it) with the five
differences the test lists: 60 epochs, patience 70, a 96 px global crop, a 4096-wide head and the probe's bank and query sizes. AdamW, cosine
over all epochs, lr 5e-4, batch 32, 10 warmup epochs, weight decay 0.04, a GLOBAL clip of 3.0, the teacher temperature 0.04 to 0.04 and an EMA
warmup of one epoch.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only.
"""

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import recipes
from .args import AutoContext, OptimArgs, is_count
from .config import join_path, unknown_key_message
from .denoising_models import named_parameters
from .dino_args import FAMILY, GEOMETRY, TASK, WATCH_METRIC, geometry_problem
from .errors import ConfigValidationError
from .registry import ModelSpec, TransferSpec

# ---------------------------------------------------------------------

_HERE = "train.api.dino_models"
_DINO = "dl_techniques.models.vision.dino"

#: The two named scales: the variant-table keys the loaded model shows (a test holds each row equal to ``DINOv1.MODEL_VARIANTS``, which has three more
#: and cannot be imported here). The registry's row reads THIS table, so base, large and giant are refused by absence.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4.0},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0},
}

#: The keys of a variant's table, which the loaded student's ``get_config()`` shows too: what :func:`observe_dino` reads.
TABLE_KEYS: Tuple[str, ...] = tuple(VARIANTS["tiny"])

# ---------------------------------------------------------------------
# What the family offers, refuses and is given
# ---------------------------------------------------------------------

# DECISION plan-2026-11-26T224507-e3381378/D-044
# Four keys are offered, and only because the measurement record names a value for each (`research/2026_dino_ssl_measurements.md`, the rule of D-002):
# the head's width, the two ends of the teacher temperature and the EMA warmup in EPOCHS. Every other constructor parameter and every other flag of the
# legacy trainer is REFUSED by name before the claim, each with its reason. Do NOT move `student_temp`, `center_momentum`, the EMA decays or `ema_warmup_steps`
# to the offered keys "for experiments": the record names no value for the first four, and `ema_warmup_steps` is a constant of one scale (295 steps is one
# epoch at imagenette and batch 32 only, 266 at HEAD). A table key is also a claim that the task works with it, and the gate varies none of them.
# The tables are FLAT (one family) and read from the P9 knob table of the plan (`scratch_step2/p9_knobs.json`, D-031).
#: Keys the factory takes (the model).
FACTORY_KEYS: Tuple[str, ...] = ("dino_out_dim",)

#: Keys of ``model_args`` that the TASK reads (the loss's temperature schedule and the EMA warmup): they are no keyword of any constructor.
TASK_KEYS: Tuple[str, ...] = ("teacher_temp", "teacher_temp_final", "ema_warmup_epochs")

#: Every offered key.
ALLOWED: Tuple[str, ...] = (*FACTORY_KEYS, *TASK_KEYS)

#: Values the task fixes where the record names none (the legacy trainer's defaults) or fixes at the record's own (the probe's estimator):
#: ``{legacy field: value}``. A test holds the first five equal to the legacy ``TrainingConfig``.
FIXED: Dict[str, Any] = {
    "student_temp": 0.1, "center_momentum": 0.9, "ema_decay_start": 0.996, "ema_decay_end": 0.9999, "teacher_temp_warmup_epochs": 30,
    "knn_bank_batches": 64, "knn_query_batches": 32,
}

_NOT_NAMED = "the measurement record names no value for it, and no run of this task varies it"
_BY_VARIANT = "the variant fixes it: choose model=dino_v1/<variant>"
_NO_RUN = "no run measured another, and the gate varies none"
_FIXED_BY_THE_TASK = "the task fixes it"

#: Keys of the constructors and flags of the legacy trainer that are refused whatever their value: ``{key: why}``; ``why`` ends the sentence
#: "model_args.<key> is not offered by <family>: ...".
REFUSED: Dict[str, str] = {
    # The legacy trainer's keys the record names no value for.
    "student_temp": f"{_NOT_NAMED} (the task uses {FIXED['student_temp']})",
    "center_momentum": f"the record names it only in an 8-epoch diagnostic arm (0.0, no effect), so {_NOT_NAMED} (the task uses {FIXED['center_momentum']})",
    "ema_decay_start": f"{_NOT_NAMED} (the task uses {FIXED['ema_decay_start']})",
    "ema_decay_end": f"{_NOT_NAMED} (the task uses {FIXED['ema_decay_end']})",
    "ema_warmup_steps": "it is an absolute step count, a constant of one scale (295 steps is one epoch at imagenette and batch 32 only); use model_args.ema_warmup_epochs",
    "source_image_size": "the record names it only in an 8-epoch diagnostic arm (224 px, no difference); every crop is decoded at transform_args.image_size",
    # The values the task fixes.
    "stateless_augmentation": f"{_FIXED_BY_THE_TASK} on: the reproducible stream is what makes two runs comparable",
    "teacher_temp_warmup_epochs": f"{_FIXED_BY_THE_TASK} at {FIXED['teacher_temp_warmup_epochs']}, and it has no effect while teacher_temp_final equals teacher_temp",
    "knn_eval_every": f"{_FIXED_BY_THE_TASK} to every epoch: a skipped epoch would write NaN, and a NaN monitor marks a run diverged",
    "knn_bank_batches": f"{_FIXED_BY_THE_TASK} at {FIXED['knn_bank_batches']}: the bar's margin was derived at that estimator",
    "knn_query_batches": f"{_FIXED_BY_THE_TASK} at up to {FIXED['knn_query_batches']}: the bar's margin was derived at that estimator",
    "knn_temperature": f"{_FIXED_BY_THE_TASK} at the weighting temperature of the DINO paper, part of the estimator",
    # The constructors' parameters the recipe leaves to the variant and the library.
    "embed_dim": _BY_VARIANT,
    "depth": _BY_VARIANT,
    "num_heads": _BY_VARIANT,
    "mlp_ratio": _BY_VARIANT,
    "in_channels": "the streams decode RGB, so the input has 3 channels",
    "qkv_bias": _NO_RUN,
    "dropout_rate": _NO_RUN,
    "attention_dropout_rate": _NO_RUN,
    "stochastic_depth_rate": _NO_RUN,
    "normalization_type": _NO_RUN,
    "attention_type": _NO_RUN,
    "ffn_type": _NO_RUN,
    "include_top": "a DINO backbone has no classification head here",
    "include_projection_head": "the training model needs the projection head: it is what the loss compares",
    "dino_hidden_dim": _NO_RUN,
    "dino_bottleneck_dim": _NO_RUN,
    "dino_nlayers": _NO_RUN,
    "use_cls_token": "the probe, predict and export read the class token's embedding",
    "name": "the task names the model",
}

#: Keywords the task or the API passes to the adapter itself (or that the data layer decides): ``{key: why}``. Naming one in ``model_args`` is refused.
SUPPLIED: Dict[str, str] = {
    "variant": f"model={FAMILY}/<variant> chooses it",
    "input_shape": "it is (image_size, image_size, 3), from transform_args.image_size",
    "image_size": "it is transform_args.image_size, the size of every crop",
    "patch_size": "it is data.patch_size",
    "n_local_crops": "it is data.n_local_crops",
    "n_local_views": "it is data.n_local_crops",
    "num_classes": "a DINO model has no classes (the backbone is built with none)",
    "local_crop_size": "it is transform_args.image_size: the multi-crop map renders every local crop at the global resolution",
}

#: The callables whose signatures :func:`signature_census` classifies: the training factory, the pair factory it calls and the backbone constructor
#: the ``**kwargs`` of both land on.
FACTORIES: Tuple[str, ...] = (
    f"{_DINO}.training:create_dino_training_model", f"{_DINO}.dino_v1:create_dino_teacher_student_pair", f"{_DINO}.dino_v1:DINOv1",
)

#: Keys of the tables that no constructor has a parameter of, so the census cannot call them stale: the trainer's own flags, the task's keys and the
#: adapter's own arguments.
UNSEEN_KEYS: Tuple[str, ...] = (
    "teacher_temp_final", "ema_warmup_epochs", "ema_warmup_steps", "center_momentum", "ema_decay_start", "ema_decay_end", "source_image_size",
    "stateless_augmentation", "teacher_temp_warmup_epochs", "knn_eval_every", "knn_bank_batches", "knn_query_batches", "knn_temperature",
    "n_local_crops", "local_crop_size",
)


def _real(value: Any) -> bool:
    """Whether ``value`` is a finite real number (a bool is none)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


#: The value rule of each offered key: ``{key: (accepts, what the message says is expected)}``.
VALUE_RULES: Dict[str, Tuple[Callable[[Any], bool], str]] = {
    "dino_out_dim": (
        lambda value: is_count(value) and value >= 2,
        "an int of at least 2 (the teacher's distribution over a head of width 1 is a single point, so the collapse flag would fire at once)"),
    "teacher_temp": (lambda value: _real(value) and value > 0, "a number above 0, not true or false"),
    "teacher_temp_final": (lambda value: _real(value) and value > 0, "a number above 0, not true or false"),
    "ema_warmup_epochs": (lambda value: _real(value) and value >= 0, "a number of at least 0, not true or false"),
}


def problems_of(model_args: Mapping[str, Any]) -> List[str]:
    """Return one line per refusal of ``model_args``; see the module contract."""
    lines: List[str] = []
    for key, value in model_args.items():
        name = f"model_args.{key}"
        if key in ALLOWED:
            accepts, expected = VALUE_RULES[key]
            if not accepts(value):
                lines.append(f"Invalid value for {name!r}: expected {expected}, got {value!r}")
        elif key in SUPPLIED:
            lines.append(f"Invalid value for {name!r}: {key} is set by the task ({SUPPLIED[key]}), got {value!r}")
        elif key in REFUSED:
            lines.append(f"Invalid value for {name!r}: {key} is not offered by {FAMILY!r} ({REFUSED[key]}), got {value!r}")
        else:
            lines.append(unknown_key_message("model_args", key, sorted(ALLOWED)))
    start, end = model_args.get("teacher_temp"), model_args.get("teacher_temp_final")
    if _real(start) and _real(end) and end < start:
        lines.append(
            f"Invalid value for 'model_args.teacher_temp_final': {end!r} is below teacher_temp {start!r}: it is the END of a warmup that starts "
            "at teacher_temp (the record ran 0.04 to 0.04 and the paper 0.04 to 0.07, neither falls), and no run measured a falling schedule"
        )
    return lines


def factory_signatures() -> Dict[str, List[str]]:
    """Return ``{callable path: named parameters}`` of the library callables behind the family (``**kwargs`` left out)."""
    return {path: named_parameters(path) for path in FACTORIES}


def signature_census(signatures: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, List[str]]:
    """Return the constructor parameters no table names and the table keys no constructor has; see the module contract."""
    if signatures is None:
        signatures = factory_signatures()
    offered = {*ALLOWED, *REFUSED, *SUPPLIED}
    present = {name for names in signatures.values() for name in names}
    return {"unclassified": sorted(present - offered), "stale": sorted(offered - present - set(UNSEEN_KEYS))}


# ---------------------------------------------------------------------
# The refusals that are not keys: precision, schedule and backup
# ---------------------------------------------------------------------

# DECISION plan-2026-11-26T224507-e3381378/D-050
# These three are FUNCTIONS the task config calls, not checks of the resolver: the resolver is handed `(context, optim_args, model_args)` and nothing else,
# while the schedule is in `optim_args`, `backup_every_steps` in `checkpoint_args` and `precision` at the top level of the task config, so the one place that sees
# all three before the claim is the task's own `problems` (step 6). Do NOT fold them into `_merged` (it cannot see the checkpoint group), and do NOT drop one because
# "the shared code warns anyway": the constant schedule only WARNS at the end of every epoch (P8), and a mid-epoch copy resumes without error on a centre
# the weights never had.


def precision_problems(precision: str, prefix: str = "") -> List[str]:
    """Return the line that refuses a ``precision`` other than ``float32``; see the module contract."""
    if precision == "float32":
        return []
    return [
        f"Invalid value for {join_path(prefix, 'precision')!r}: expected float32 for {TASK} (the loss's centre update and the teacher's EMA "
        f"were never measured in a mixed precision), got {precision!r}"
    ]


def schedule_problems(schedule: Any, prefix: str = "optim_args") -> List[str]:
    """Return the line that refuses a ``constant`` learning-rate schedule; see the module contract."""
    if schedule != "constant":
        return []
    return [
        f"Invalid value for {join_path(prefix, 'schedule')!r}: {schedule!r} adds a ReduceLROnPlateau(monitor='val_loss') to the run, and a "
        f"{TASK} run has no validation data, so it would warn at the end of every epoch and never act; use 'cosine_decay', the recipe's"
    ]


def warmup_problems(ema_warmup_epochs: Any, epochs: Any, smoke: bool = False) -> List[str]:
    """Return the line that refuses a teacher EMA warmup as long as the run or longer; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-100
    # A warmup of `epochs` or more freezes the teacher for the WHOLE run: `CompiledTeacherEMA` never updates it, the student chases its own
    # initial weights, and the run still reaches the claim, trains to the end and reports a number nobody measured (`ema_warmup_epochs=60` at the
    # recipe's 60 epochs). The check needs the RESOLVED epochs, which the resolver is never handed, so it is a function the task calls from
    # `check_model_args` (before the claim, for a fresh run and a resume alike: both resolve first). A smoke run is exempt: it trains one epoch and
    # applies no bar, so the recipe's own 1-epoch warmup leaves its teacher frozen, and the documents say so. Do NOT make the recipe's warmup
    # shrink with the epochs (a silent change of the recipe), and do NOT refuse the smoke run (the documented plumbing check would need a typed key).
    if smoke or not _real(ema_warmup_epochs) or not _real(epochs) or ema_warmup_epochs < epochs:
        return []
    return [
        f"Invalid value for 'model_args.ema_warmup_epochs': {ema_warmup_epochs!r} is not below the run's {epochs!r} epochs, so the teacher's EMA "
        f"would never start and the teacher would stay at its initial weights for the whole run, a configuration no run measured; use a value "
        f"below {epochs!r} (the recipe's is {RECIPE['model_args']['ema_warmup_epochs']!r}) or train more epochs"
    ]


def backup_problems(backup_every_steps: Any, prefix: str = "checkpoint_args") -> List[str]:
    """Return the line that refuses a ``backup_every_steps``; see the module contract."""
    if backup_every_steps is None:
        return []
    return [
        f"Invalid value for {join_path(prefix, 'backup_every_steps')!r}: {backup_every_steps!r}; a copy in the middle of an epoch would pair the "
        "weights of that step with the loss's centre of the last epoch end (the centre is not in the weights file and moves with every batch), "
        "so a resume from it would train on a centre the weights never had; leave it null: a run is backed up at every epoch end"
    ]


# ---------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------


def create_dino_model(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], patch_size: int, n_local_crops: int, dino_out_dim: int,
    **model_args: Any
) -> Any:
    """Build the DINO training model at ``input_shape`` and prove its views and its grid; see the module contract."""
    if num_classes is not None:
        raise ValueError(f"the {FAMILY!r} family has no classes: num_classes must be None, got {num_classes!r}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r} of {FAMILY!r}; the variants are {sorted(VARIANTS)}")
    shape = tuple(input_shape)
    if len(shape) != 3 or not all(is_count(side) for side in shape) or shape[0] != shape[1] or shape[2] != 3:
        raise ValueError(f"input_shape must be (size, size, 3), one square RGB image, each a positive int, got {shape!r}")
    reason = geometry_problem(patch_size, shape[0])
    if reason is not None:
        raise ValueError(f"patch_size {patch_size!r} {reason} (input_shape {shape!r})")
    if not is_count(n_local_crops):
        raise ValueError(f"n_local_crops must be a positive int (the views are 2 global and n_local_crops local), got {n_local_crops!r}")
    lines = [f"model_args.{key} is read by the task and is not a model keyword, got {value!r}" for key, value in model_args.items() if key in TASK_KEYS]
    lines += problems_of({"dino_out_dim": dino_out_dim, **{key: value for key, value in model_args.items() if key not in TASK_KEYS}})
    if lines:
        raise ValueError("; ".join(lines))
    import keras

    from dl_techniques.models.vision.dino.training import create_dino_training_model

    model = create_dino_training_model(
        variant, image_size=shape[0], patch_size=patch_size, n_local_views=n_local_crops, dino_out_dim=dino_out_dim
    )
    # DECISION plan-2026-11-26T224507-e3381378/D-045
    # The model is built HERE with the rank-5 input, and what it does is PROVED by two real forward passes: the sequence's own
    # `build((None, *input_shape))` is refused by this model (P1: "expects a rank-5 multi-crop input"), and the view count and the patch grid are what the
    # record will state, so they are compared with what the ARGUMENTS say and never with the model's own attributes (`n_views`, `num_patches` agree with
    # themselves by construction). A grid that is not the image over the patch trains on the wrong positions and still reports a finite loss. Do NOT take
    # the expected numbers from the built model, do NOT leave the model unbuilt for the sequence to build, and do NOT wrap it: the file this family
    # saves is the library's training model.
    views = 2 + n_local_crops
    model.build((None, views, *shape))
    packed = model(keras.ops.zeros((1, views, *shape), dtype="float32"), training=False)
    expected_rows, expected_tokens = 2 * views - 2, (shape[0] // patch_size) ** 2
    rows, width = (int(side) for side in packed.shape)
    if (rows, width) != (expected_rows, 2 * dino_out_dim):
        raise ValueError(
            f"the packed output of the built model is {(rows, width)}, but {views} views (2 global and {n_local_crops} local) and a head of "
            f"{dino_out_dim} give {(expected_rows, 2 * dino_out_dim)}: the loss would pair the wrong views"
        )
    tokens = int(model.student.get_layer("patch_embed")(keras.ops.zeros((1, *shape), dtype="float32")).shape[1])
    if tokens != expected_tokens:
        raise ValueError(
            f"the built model cuts an image of {shape} into {tokens} tokens, but a patch_size of {patch_size} gives {expected_tokens}: "
            "the position table would not line up with the patches"
        )
    return model


def observe_dino(model: Any) -> Optional[Dict[str, Any]]:
    """Return the variant-table keys of the student of ``model``, and ``None`` for a model with no student; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-046
    # The row HAS an `observe`: the loaded model is the TRAINING model, which is no `DINOv1` (the student inside it is), so the class path of
    # `check_model_against_record` would refuse every honest file, and without any check a file of one variant would load under a record naming
    # another (the defect the pretrain rows closed, plan 5c6cc105 D-042). P6 measured that the student's own `get_config()` carries the four keys.
    # Do NOT drop `observe` "because the table is module-level", and do NOT read the teacher (it is the same architecture and the same keys).
    student = getattr(model, "student", None)
    if student is None:
        return None
    config = student.get_config()
    return {key: config[key] for key in TABLE_KEYS}


# ---------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------


def _merged(context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the recipe's ``model_args`` under the caller's, checked: the one place both resolvers read."""
    # The adapter passes no kernel regularizer, so sgd's decay would be dropped without a word; the other resolvers refuse it too.
    recipes.refuse_dropped_sgd_decay(optim_args, TASK, context.family)
    merged = recipes.merge_model_args(context, optim_args, model_args, {})
    lines = problems_of(merged)
    if lines:
        raise ConfigValidationError(lines)
    return merged


def resolve_dino_args(context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's, without the task's keys; see the module contract."""
    merged = _merged(context, optim_args, model_args)
    return {key: value for key, value in merged.items() if key not in TASK_KEYS}


def task_values(context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``teacher_temp``, ``teacher_temp_final`` and ``ema_warmup_epochs`` as the request resolves them; see the module contract."""
    merged = _merged(context, optim_args, model_args)
    return {key: merged[key] for key in TASK_KEYS}


def ema_warmup_steps(ema_warmup_epochs: float, steps_per_epoch: int) -> int:
    """Return the teacher EMA's warmup in optimizer steps; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-047
    # The warmup is DENOMINATED IN EPOCHS and resolved here, against the steps the data gives. The record's measured 295 is `num_train // batch_size` of ONE
    # dataset and batch (one epoch there, an arbitrary number anywhere else), and at HEAD that epoch is 266 steps: a typed 295 would freeze the teacher for
    # 1.1 epochs on the gate's own run and 0.2 on cifar10's 1,406. Do NOT type a step count into the recipe or read one off the record, and do NOT offer
    # `ema_warmup_steps` (D-044). `test_the_teacher_ema_warmup_is_denominated_in_epochs...` holds this equal to the legacy trainer's resolution.
    return int(round(ema_warmup_epochs * steps_per_epoch))


# ---------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------

#: The verdict arms' epochs (the legacy trainer's default is 100), the legacy batch, and a patience that lets the 60-epoch horizon run to its end.
EPOCHS = 60
BATCH_SIZE = 32
PATIENCE = 70

RECIPE: Dict[str, Any] = {
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    # train/dino/train_dino.py TrainingConfig: lr 5e-4, weight_decay 0.04, warmup 10, gradient_clipping 3.0 (a GLOBAL norm), `build_optimizer`: cosine
    # over steps_per_epoch * epochs, alpha 0.01 (decay_over total), the builder's own betas.
    "optim_args": recipes.optim_recipe(
        optimizer="adamw", lr=5e-4, weight_decay=0.04, warmup_epochs=10, decay_over="total", global_clipnorm=3.0
    ),
    "transform_args": {},
    "metric_args": {"watch_metric": WATCH_METRIC},
    "checkpoint_args": {"early_stopping_patience": PATIENCE},
    # The improved arm of the record: the teacher temperature held at 0.04 and a one-epoch EMA warmup; the head at the record's 4096.
    "model_args": {"dino_out_dim": 4096, "teacher_temp": 0.04, "teacher_temp_final": 0.04, "ema_warmup_epochs": 1.0},
    "data": {"n_local_crops": 4},
    # The one table of sizes, `dino_args.GEOMETRY`, is what `dino_facts` reads too.
    "per_dataset": {
        name: {"transform_args": {"image_size": size}, "data": {"patch_size": patch}} for name, (size, patch) in GEOMETRY.items()
    },
    "per_variant": {},
}

# DECISION plan-2026-11-26T224507-e3381378/D-049
# The notices name WHERE the gate's numbers are and state none of them, and they say what the gate COVERS, which stays true after any night: a sentence
# that said "unmeasured" went false the night the 4j gate ran, and one that quoted a gate's figures would go stale at the next. Do NOT write a number, a
# count or a date of the gate here, and do NOT say a recipe is unmeasured or "a first measurement": the README block is rewritten by every later gate,
# the pointer and the coverage stay true. (The block itself and the test that it exists are step 9's.)
#: Where the dino gate's numbers live: the block of ``src/train/README.md`` between the ``dino-gate`` markers.
GATE_BLOCK = "the README block 'Measured: the dino gate' (src/train/README.md, between its dino-gate markers)"


def recipe_notices_of(dataset: str) -> List[Tuple[str, str]]:
    """Return the ``(level, text)`` lines on the recipe and what the gate covers for ``dataset``; see the module contract."""
    optim = RECIPE["optim_args"]
    size, patch = GEOMETRY["imagenette"]
    notices = [(
        "info",
        f"{FAMILY}: the recipe is the legacy DINO trainer's optimizer (AdamW, lr {optim['lr']:g}, batch {BATCH_SIZE}, weight decay "
        f"{optim['weight_decay']:g}, {optim['warmup_epochs']} warmup epochs, a global gradient clip of {optim['global_clipnorm']:g}) with "
        f"{EPOCHS} epochs, patience {PATIENCE}, a {RECIPE['model_args']['dino_out_dim']:,}-wide head and {RECIPE['data']['n_local_crops']} local "
        f"crops. The gate trains imagenette ({size} px, patch {patch}) with the tiny variant only: cifar10 and the small variant are offered and "
        f"the gate trained neither. What it measured, for the run it trained, is in {GATE_BLOCK}, and it is no promise for yours",
    )]
    if dataset == "cifar10":
        notices.append((
            "info",
            f"{dataset}: {GEOMETRY[dataset][0]} px and patch {GEOMETRY[dataset][1]} are this recipe's row for it (the library's patch 16 leaves 4 "
            f"tokens), and a DINO run on it was not trained by the gate; the bar's margin for it is above the spread of a random-init model's "
            f"probe, which is all that measures it (see {GATE_BLOCK})",
        ))
    return notices


def builtin_dino_models() -> Tuple[ModelSpec, ...]:
    """Return the one dino family spec, freshly built; see the module contract."""
    # DECISION plan-2026-11-26T224507-e3381378/D-043
    # The row is built and tested HERE and registered with its task: `recipes.register_builtin_models` calls this function in the commit that adds the
    # `dino` TaskSpec (step 6). MEASURED at step 5 (`scratch_step5/measured.md`): registering the family first turned red the three listing tests of
    # `test_every_family_builds_from_its_defaults`, `test_every_row_declares_a_spec` of the init census and the family count sentence of
    # `src/train/AGENTS.md` (32 families whose split over the nine registered tasks sums to 31), and made `python -m train list_models` list `dino_v1/*`
    # with no command to run them. Do NOT add the call without the TaskSpec, nor the TaskSpec without the call (the tokenizer row's rule, plan fa94f59c D-031).
    return (
        ModelSpec(
            family=FAMILY,
            factory=f"{_HERE}:create_dino_model",
            variants_attr=f"{_HERE}:VARIANTS",
            tasks=(TASK,),
            defaults=RECIPE,
            resolve_model_args=f"{_HERE}:resolve_dino_args",
            observe=f"{_HERE}:observe_dino",
            # DECISION plan-2026-11-26T224507-e3381378/D-048
            # EMPTY on purpose, and present because the init census requires a declaration of EVERY row (`undeclared(registry.MODELS) == []`), whatever its
            # task. A dino run is no `init` target (plan D-013: the student and the teacher sit at the same paths, so a copy needs a teacher re-sync that the
            # run does not do, and epoch zero would call `model.evaluate`, which advances the centre): the task refuses before `init_from` reads this row, so
            # a `head` or a `fixed` key here would be a claim nothing reads. Do NOT fill it "to be safe": step 8 adds the `vit` row's `("dino", "dino_v1")`
            # source on the TARGET's row, and the refusal sentence is the task's.
            transfer=TransferSpec(),
        ),
    )
