"""The image-denoising task: a config, a trainer and the keyword-only function.

``train_image_denoising(...)`` is the function a Python caller uses and the function
``python -m train image_denoising key=value ...`` calls, so the two cannot disagree. It validates its keywords into an
:class:`ImageDenoisingConfig` and hands that to :class:`ImageDenoisingTrainer`, a
:class:`~train.api.trainer.TaskTrainer` that supplies the hooks of this task and nothing else (a task overrides hooks,
never stages). The hooks are the denoising assumptions of the run: folders of clean images with the noise added by the
pipeline, the ``val_psnr`` success value and its bar (the PSNR of the noisy input plus a gain), the compile options of
a family that XLA rejects under ``mixed_float16``, the checkpoint selection window that opens when the noise
curriculum reaches its last value, the epoch line, the summary fields and notes, and the reporting after ``fit``.

Interface contract
------------------
``ImageDenoisingConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``, ``model`` and
    ``data`` are required. The groups are those of classification except ``optim_args``
    (:class:`~train.api.denoising_args.DenoiseOptimArgs`: a warmup that follows the epochs), ``transform_args``
    (:class:`~train.api.denoising_args.DenoiseTransformArgs`), ``metric_args``
    (:class:`~train.api.denoising_args.DenoiseMetricArgs`) and ``loss_args``
    (:class:`~train.api.denoising_args.DenoiseLossArgs`); ``data`` is :class:`~train.api.denoising_args.DenoisingData`
    (the folders, the noise and its curriculum: every noise key is under ``data.*``). ``analysis_args.conformal_alpha``
    is refused (a conformal set is a set of classes, and a denoiser has none), and so is a ``data.curriculum_epochs``
    above the epochs.

``ImageDenoisingTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "image_denoising"`` and ``success_metric = "val_psnr"``.

    What a run is. The training noise is stateless and drawn on the fly (``train.common.denoising_data``); the
    validation set is fixed crops with fixed noise keyed on ``(data.val_seed, index)``, so ``val_psnr`` is a function
    of the weights alone (equal across epochs, a resume and the seeds of the training stream). The PSNR of the noisy
    input of that set, ``input_psnr``, is measured before ``fit`` by evaluating the IDENTITY model through the very
    metric objects the run compiles (:func:`train.common.denoising_run.identity_scores`).

    The selection window (``selection_start_epoch``). ``data.curriculum_epochs`` (auto: ``round(0.8 * epochs)``) is the
    epoch the noise upper bound reaches ``data.noise_sigma_max`` at; the checkpoint saves nothing and ``EarlyStopping``
    counts nothing before epoch ``curriculum_epochs - 1`` (0-based), so the selected model has trained on the full
    range it is scored on. ``final_model.keras`` is the deliverable and the summary's ``selection`` block says whether
    the final epoch IS the best (``final_is_best``), the window and its length.

    The verdict. ``success_value`` is the best ``val_psnr`` of the window MINUS ``input_psnr`` (a gain in dB) and
    ``success_threshold`` is ``metric_args.success_gain_db`` (auto 1.0 dB): a denoiser that does not beat its own input
    by the gain has learned nothing, and the run exits 1 with both numbers in the summary.

    The epoch-0 check is the ratio of the untrained model's loss to the loss of the identity on the same set
    (``initial_loss_ratio``); no verdict on the initialization is given (``init_scale_warning`` is null, and
    ``unjudged_note`` says why): the ratio of an untrained denoiser depends on the noise range far more than on the
    scale of its weights, so no threshold on it separates a well-scaled from a mis-scaled model. A non-finite loss is
    refused.

    Compile. ``compile_options`` returns ``{"jit_compile": False}`` for the families XLA rejects under
    ``mixed_float16`` (:func:`train.api.denoising_models.compile_options_for`) and nothing otherwise.

    NOT YET WRITTEN, hooks left at their defaults or returning an explicit empty value (each is a later step of the
    plan, named in the summary's ``notes``): ``task_record`` (no ``task.json``, so ``predict`` and ``export`` refuse
    such a run), ``config_stamps`` (no legacy-reader stamps), ``run_analysis`` (the analyzer is skipped),
    ``figure_callbacks`` (no figure), the held-out cells (``held_out`` says why it is empty) and the homogeneity check
    of the three bias-free families (``homogeneity`` is not written).

``train_image_denoising(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"family/variant"``, e.g. ``"bfconvunext/tiny"``; ``python -m train list_models image_denoising``
        lists them.
    :param data: a mapping (``{"train_dirs": [...], "val_dirs": [...]}``), a
        :class:`~train.api.denoising_args.DenoisingData`, or the path of a YAML file holding one.
    :param epochs: / ``batch_size`` / ``devices`` / ``seed`` / ``precision`` / ``resume_interrupted`` /
        ``checkpoint`` / ``model_args`` and the groups: as in
        :func:`~train.api.tasks.image_classification.train_image_classification`.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see :meth:`~train.api.trainer.TaskTrainer.run`. A
        family that does not train denoising, a ``patch_size`` that is not a multiple of the family's stride, a model
        key a bias-free family refuses, a loss other than ``mse`` for one, a directory with no image, and a
        ``data.curriculum_epochs`` above the epochs are refused before the run directory is claimed.
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012): ``python -m train image_denoising help``
imports this module to print the config's keys and must stay free of TensorFlow. Every ``keras`` and ``train.common``
name is imported inside a method.
"""

import dataclasses
import functools
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .. import recipes, registry
from ..args import (
    AnalysisArgs,
    AutoContext,
    CheckpointArgs,
    DebugArgs,
    LoggerArgs,
)
from ..config import (
    AUTO,
    DEFAULT_SEED,
    PATH_FORM,
    RUN_HELP,
    Auto,
    Precision,
    StrictConfig,
    join_path,
    resume_help,
    run_request,
    validate,
)
from ..data_args import DataFacts
from ..denoising_args import (
    DenoiseLossArgs,
    DenoiseMetricArgs,
    DenoiseOptimArgs,
    DenoisingData,
    DenoiseTransformArgs,
    curriculum_callback,
    denoising_facts,
    load_denoising,
    load_denoising_data_config,
    read_denoising_data_file,
)
from ..denoising_models import TASK, compile_options_for
from ..errors import ConfigError
from ..segmentation_models import check_image_size
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, config_problems

# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ImageDenoisingConfig(StrictConfig):
    """Everything a denoising run is told; see the module contract."""

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={
            "help": "family/variant, e.g. bfconvunext/tiny or scunet/small; python -m train list_models "
            "image_denoising lists them"
        }
    )
    data: DenoisingData = dataclasses.field(
        metadata={
            "help": "the image folders and the noise: data.train_dirs=[a,b] data.val_dirs=[c], with data.noise_* and "
            "data.curriculum_* keys; or the path of a YAML file holding them (data=data.yaml), over which dotted "
            "keys win (data=data.yaml data.max_train_files=100)",
            PATH_FORM: read_denoising_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["epochs"]})
    batch_size: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["batch_size"]})
    seed: int = dataclasses.field(default=DEFAULT_SEED, metadata={"help": RUN_HELP["seed"]})
    devices: Auto[Optional[int]] = dataclasses.field(
        default=AUTO, metadata={"help": RUN_HELP["devices"]}
    )
    precision: Precision = dataclasses.field(
        default="float32", metadata={"help": RUN_HELP["precision"]}
    )
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={"help": resume_help("an epoch of a large set of files can take a long time")},
    )
    checkpoint: Optional[str] = dataclasses.field(
        default=None, metadata={"help": RUN_HELP["checkpoint"]}
    )
    model_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict, metadata={"help": RUN_HELP["model_args"]}
    )
    optim_args: DenoiseOptimArgs = dataclasses.field(default_factory=DenoiseOptimArgs)
    transform_args: DenoiseTransformArgs = dataclasses.field(default_factory=DenoiseTransformArgs)
    metric_args: DenoiseMetricArgs = dataclasses.field(default_factory=DenoiseMetricArgs)
    loss_args: DenoiseLossArgs = dataclasses.field(default_factory=DenoiseLossArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range, a conformal level and a curriculum longer than the run."""
        lines = config_problems(self, prefix)
        alpha = self.analysis_args.conformal_alpha
        if alpha is not None:
            lines.append(
                f"Invalid value for {join_path(join_path(prefix, 'analysis_args'), 'conformal_alpha')!r}: "
                "a conformal prediction set is a set of classes for a whole image, which a denoising run does not "
                f"produce; leave it null, got {alpha!r}"
            )
        curriculum, epochs = self.data.curriculum_epochs, self.epochs
        if isinstance(curriculum, int) and isinstance(epochs, int) and curriculum > epochs:
            lines.append(
                f"Invalid value for {join_path(join_path(prefix, 'data'), 'curriculum_epochs')!r}: {curriculum} is "
                f"above the {epochs} epochs of the run, so the noise would never reach data.noise_sigma_max and the "
                "selection window (which opens at the end of the curriculum) would start after the last epoch; "
                f"expected at most {epochs}"
            )
        return lines


class ImageDenoisingTrainer(TaskTrainer):
    """The hooks of an image-denoising run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    """

    task = TASK
    success_metric = "val_psnr"
    initial_loss_hint = (
        "Lower optim_args.lr, or check the model arguments and the images (an image with a non-finite value)."
    )

    def __init__(self, config: StrictConfig) -> None:
        super().__init__(config)
        #: The scores of the identity model on the fixed validation set (``loss``, ``mae``, ``psnr``, ``ssim``), set
        #: by ``build_data`` (a resume builds the data again, so it is there before the summary or the verdict).
        self._identity: Optional[Dict[str, float]] = None
        #: The 0-based epoch the selection window opens at, set by ``build_data`` from the RESOLVED config.
        self._window_start: Optional[int] = None

    # -- data and model -------------------------------------------------

    def describe_data(self, config: ImageDenoisingConfig) -> DataFacts:
        return denoising_facts(config.data)

    def build_data(self, config: ImageDenoisingConfig, context: AutoContext) -> Any:
        import time

        from dl_techniques.utils.logger import logger
        from train.common import denoising_data, denoising_run

        transform_cfg: DenoiseTransformArgs = config.transform_args
        data = load_denoising(
            config.data,
            patch_size=transform_cfg.patch_size,
            channels=transform_cfg.channels,
            patches_per_image=transform_cfg.patches_per_image,
            batch_size=config.batch_size,
            seed=config.seed,
            augment=denoising_data.build_patch_augment(transform_cfg.augment),
            # The validation split is batched at the training batch (`eval_batch_size=None`), as folder data is for the
            # other tasks: the shared fallback of `analysis_args.eval_batch_size` is 256, and 256 patches of 256 px
            # asked 18 GB of a classifier once. The key still sets the batch of the load check and the analysis.
            eval_batch_size=None,
        )
        started = time.perf_counter()
        identity = denoising_run.identity_scores(data.select_ds, config.loss_args.name)
        if not identity["loss"] > 0:
            # A ValueError: `run_sequence.load_data` turns it into a ConfigError that says `out` is now used.
            raise ValueError(
                f"data.noise_sigma_max: the noisy validation patches equal the clean ones (the identity's "
                f"{config.loss_args.name} loss is {identity['loss']}), so there is no noise to remove and no input "
                "PSNR to beat; raise data.noise_sigma_max above data.noise_sigma_min"
            )
        self._identity = identity
        self._window_start = self.selection_start_epoch(config)
        logger.info(
            f"Identity (the noisy input returned as the denoised image) on the {data.n_select} fixed validation "
            f"patches: PSNR {identity['psnr']:.3f} dB, SSIM {identity['ssim']:.4f}, {config.loss_args.name} loss "
            f"{identity['loss']:.6g} ({time.perf_counter() - started:.1f} s)"
        )
        return data

    def build_model(self, config: ImageDenoisingConfig, context: AutoContext, data: Any) -> Any:
        spec = registry.MODELS[context.family]
        # A flexible input: the model trains on patch_size crops and is saved ready for an image of any size.
        return recipes.build_model(
            spec, context.variant, None, (None, None, data.input_shape[-1]),
            self._model_kwargs(config, context),
        )

    @staticmethod
    def _model_kwargs(config: ImageDenoisingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's."""
        spec = registry.MODELS[context.family]
        return spec.load_resolver()(context, config.optim_args, config.model_args)

    def used_arguments(self, config: ImageDenoisingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def check_model_args(self, config: ImageDenoisingConfig, context: AutoContext) -> None:
        # Before the claim: the adapter's own refusals (a key a bias-free family may not take, a layer that would put
        # a constant into the model), the patch size against the family's stride, and a loss a bias-free family may
        # not be trained with.
        self._model_kwargs(config, context)
        spec = registry.MODELS[context.family]
        transform_cfg: DenoiseTransformArgs = config.transform_args
        if spec.stride_of(context.variant, config.model_args) is not None:
            check_image_size(
                spec, context.variant, config.model_args, transform_cfg.patch_size,
                key="transform_args.patch_size",
            )
        loss_cfg: DenoiseLossArgs = config.loss_args
        if "bias_free" in spec.properties and loss_cfg.name != "mse":
            raise ConfigError(
                f"Invalid value for 'loss_args.name': {loss_cfg.name!r} is refused for {context.family!r}, a "
                "bias-free family: it is trained for the minimum mean square error, whose minimiser is the "
                "conditional mean that its residual-as-score reading (Miyasawa, Tweedie) needs, and an "
                "L1-type loss estimates something else; use loss_args.name=mse, or a family that is not "
                "bias-free (convunext_biased, scunet)"
            )

    def input_check(self, facts: DataFacts, data: Any) -> None:
        if data.num_classes is not None or facts.num_classes is not None:
            raise RuntimeError(
                f"{type(self).__name__} holds no classes, but describe_data said {facts.num_classes!r} and "
                f"build_data loaded {data.num_classes!r}"
            )

    # -- compile --------------------------------------------------------

    def metric_names(self, config: ImageDenoisingConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates variables and the
        # first eager op would initialise TensorFlow's devices before setup_gpu.
        from train.common.denoising_run import DENOISING_METRIC_NAMES

        return list(DENOISING_METRIC_NAMES)

    def build_loss_and_metrics(self, config: ImageDenoisingConfig, data: Any) -> Tuple[Any, Any]:
        from train.common.denoising_run import denoising_loss, denoising_metrics

        loss_cfg: DenoiseLossArgs = config.loss_args
        return denoising_loss(loss_cfg.name), denoising_metrics()

    def compile_options(self, config: ImageDenoisingConfig, spec: registry.ModelSpec) -> Dict[str, Any]:
        return compile_options_for(spec.family, config.precision)

    def initial_loss_check(
        self, config: ImageDenoisingConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the loss over the identity's, and no verdict is given.

        :raises RuntimeError: for a non-finite loss (the model diverges before training).
        """
        import numpy as np

        from dl_techniques.utils.logger import logger
        from train.common import classification_run

        if self._identity is None:
            raise RuntimeError(
                "initial_loss_check compares with the identity's loss, which build_data works out; it was asked "
                "for before the data was built"
            )
        metrics = classification_run.evaluate_dict(model, data.select_ds)
        loss = metrics["loss"]
        if not np.isfinite(loss):
            raise RuntimeError(
                f"Initial validation loss is {loss}: the model diverges before training. {self.initial_loss_hint}"
            )
        ratio = loss / self._identity["loss"]
        logger.info(
            f"Untrained evaluate BEFORE fit: val loss {loss:.6g} on {data.n_select} patches, {ratio:.3g}x the "
            f"identity's {self._identity['loss']:.6g}; val PSNR {metrics['psnr']:.3f} dB against the input's "
            f"{self._identity['psnr']:.3f} dB"
        )
        # DECISION plan-2026-11-10T095307-d483ffab/D-024
        # NO verdict (`None`) on the initialization, whatever the model. The ratio to the identity's loss is the
        # baseline the summary has always carried, but it is not a scale test: an untrained denoiser's output is
        # unrelated to its input, so the ratio is about E[x^2] / E[sigma^2], which is 10 to 300 over the noise ranges
        # this task accepts, for a perfectly scaled model. Do NOT apply `INITIAL_LOSS_WARN_FACTOR` to it (it would warn
        # on every default run) and do NOT run a training-mode pass to get one (it moves the BatchNorm statistics and
        # every random generator, which a diagnostic must not do). A non-finite loss is the one refusal.
        return metrics, ratio, None

    def unjudged_note(self, config: ImageDenoisingConfig) -> str:
        return (
            "`init_scale_warning` is null (not judged): `initial_loss_ratio` is the loss of the untrained model over "
            "the loss of the IDENTITY (the noisy input returned unchanged) on the same fixed validation patches, "
            "kept as the baseline. An untrained denoiser's output is unrelated to its input, so that ratio is large "
            "by design and depends on the noise range, not on the scale of the weights, and no threshold on it "
            "separates a well-scaled model from a mis-scaled one; a non-finite initial loss is refused. For a model "
            "with BatchNorm layers the epoch-0 pass is also in inference mode with the moving statistics unset"
        )

    # -- the selection window and the verdict ---------------------------

    def selection_start_epoch(self, config: ImageDenoisingConfig) -> int:
        # DECISION plan-2026-11-10T095307-d483ffab/D-013
        # The checkpoint and EarlyStopping start at the curriculum's LAST epoch (0-based `curriculum_epochs - 1`, where
        # `NoiseSigmaCurriculumCallback` reaches `noise_sigma_max`): an epoch of the ramp has not trained on the range
        # it is scored on, and a checkpoint taken there is under-exposed to high noise. Do NOT start at
        # `curriculum_epochs` (one epoch late: the full-range epoch the window exists for would be left out), and do
        # NOT leave it at 0 (a ramp epoch with a lucky low-noise `val_psnr` can never be beaten by a model that has
        # seen the whole range; `test_image_denoising_task` plants exactly that curve).
        epochs = config.data.curriculum_epochs
        if not isinstance(epochs, int) or isinstance(epochs, bool):
            raise RuntimeError(
                f"selection_start_epoch needs the resolved config: data.curriculum_epochs is {epochs!r}"
            )
        return epochs - 1

    def extra_callbacks(self, config: ImageDenoisingConfig, data: Any) -> List[Any]:
        # The curriculum assigns the sigma upper bound the pipeline reads, at the start of every epoch (Keras passes
        # the ABSOLUTE epoch, so a resume needs no state of its own).
        return [curriculum_callback(config.data, data.sigma_max_var)]

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the best ``val_psnr`` of the selection window minus the PSNR of the noisy input, in dB."""
        if self._identity is None or self._window_start is None:
            raise RuntimeError(
                "success_value is measured against the noisy input's PSNR, which build_data works out; it was "
                "asked for before the data was built"
            )
        window = hist[self.success_metric][self._window_start:]
        return float(max(window)) - float(self._identity["psnr"])

    def success_threshold(self, config: ImageDenoisingConfig) -> float:
        # DECISION plan-2026-11-10T095307-d483ffab/D-013
        # The bar is a GAIN over the measured input PSNR of this validation set (`success_gain_db`), not an absolute
        # PSNR: the input PSNR is -20 log10 of the typical sigma and moves 20 dB between noise ranges, so one number
        # would pass or fail the same model by the noise it was given. Do NOT hard-code the gain here: it is the
        # config's (`metric_args.success_gain_db`), and a test plants a model that clears 1.0 and not 3.0.
        metric_cfg: DenoiseMetricArgs = config.metric_args
        return float(metric_cfg.success_gain_db)

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: ImageDenoisingConfig) -> Tuple[str, ...]:
        return ("loss", "val_psnr", "val_ssim")

    def summary_fields(self, config: ImageDenoisingConfig, data: Any) -> Dict[str, Any]:
        data_cfg: DenoisingData = config.data
        identity = self._identity or {}
        additive = data_cfg.noise_kind == "additive"
        return {
            "noise": {
                "kind": data_cfg.noise_kind,
                "sigma_min": data_cfg.noise_sigma_min,
                "sigma_max": data_cfg.noise_sigma_max,
                "composite_ratio": data_cfg.noise_composite_ratio if data_cfg.noise_kind == "composite" else None,
                "clip": data_cfg.noise_clip,
                "val_seed": data_cfg.val_seed,
            },
            "curriculum": {
                "schedule": data_cfg.curriculum_schedule,
                "epochs": data_cfg.curriculum_epochs,
                "sigma_max_start": data_cfg.noise_sigma_max_start,
                "sigma_max_end": data_cfg.noise_sigma_max,
                "selection_start_epoch": self._window_start,
            },
            "input_psnr": identity.get("psnr"),
            "score_identity": {
                "holds": bool(additive and not data_cfg.noise_clip),
                "note": (
                    "the residual (noisy minus denoised) is the score of the noisy-image density only for ADDITIVE "
                    "noise that is NOT clipped (Miyasawa, Tweedie): this run's noise is "
                    f"{data_cfg.noise_kind}{', clipped to [0, 1]' if data_cfg.noise_clip else ''}"
                ),
            },
        }

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        notices: List[Tuple[str, str]] = [
            (
                "info",
                f"{family}: the recipe is the legacy denoiser trainers' optimizer (AdamW, cosine, a global clip of "
                "1.0); data.noise_clip is false here (the legacy trainers clip the noisy input to [0, 1], which "
                "breaks the Gaussian noise near 0 and 1), so PSNR values differ from theirs by design, the "
                "checkpoint is selected only from the end of the noise curriculum, and the warmup is 10 percent "
                "of the epochs",
            )
        ]
        if family == "scunet":
            notices.append(
                (
                    "info",
                    "scunet: the library publishes one network; the tiny and small variants are this API's own "
                    "smaller versions of it and no recipe was tuned for them (batch 8, the legacy optimizer)",
                )
            )
        return notices

    def task_notes(
        self,
        config: ImageDenoisingConfig,
        baseline: Dict[str, float],
        ratio: float,
        data: Any,
    ) -> List[str]:
        """Return the denoising lines of ``results_summary.json``'s ``notes``."""
        from train.common import classification_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        metric_cfg: DenoiseMetricArgs = config.metric_args
        data_cfg: DenoisingData = config.data
        identity = self._identity or {}
        start = self._window_start if self._window_start is not None else 0
        notes = [
            f"initial loss {baseline['loss']:.6g} is {ratio:.3g}x the {config.loss_args.name} loss of the identity "
            f"({identity.get('loss', float('nan')):.6g}) on the {data.n_select} fixed validation patches: a plain "
            "evaluate in inference mode; see the `init_scale_warning` note",
            *classification_run.RUN_NOTES[:2],
            "`val_loss` and `val_psnr` are scores of the FIXED validation patches (crops and noise keyed on "
            f"(data.val_seed={data_cfg.val_seed}, index)), so they are a function of the weights alone; `val_psnr` is "
            "the mean over patches of each patch's PSNR at max_val 1 (at most 70 dB per patch), not the PSNR of the "
            "mean squared error",
            f"`success_value` is the best `val_psnr` from epoch {start + 1} on (the end of the noise curriculum, "
            f"`curriculum.selection_start_epoch` {start}) MINUS `input_psnr` "
            f"({identity.get('psnr', float('nan')):.3f} dB, the identity evaluated through the same metric on the same "
            "patches); `success_threshold` is `metric_args.success_gain_db` "
            f"({metric_cfg.success_gain_db:g} dB, an inferred number); the checkpoint and EarlyStopping count only "
            f"epochs from that one on, and `best_epoch` is the best of them (`selection` has the window and "
            "`final_is_best`)",
            "`test_metrics_best` and `test_metrics_final` are null: the data has no test split; the best_model.keras "
            "reload check ran on the validation patches (`best_checkpoint_max_abs_diff`); `held_out` says why no "
            "held-out cell was scored",
            "no model analysis, figure or homogeneity check ran: the denoising task does not write them yet "
            "(`analyzer.status` is 'skipped'), and no task.json was written, so `predict` and `export` refuse this run",
        ]
        if not checkpoint_cfg.save_best:
            notes.append(
                "checkpoint_args.save_best is false: no best_model.keras was written, so the reload check did not run"
            )
        if not checkpoint_cfg.save_last:
            notes.append(
                "checkpoint_args.save_last is false: no final_model.keras was written and "
                "`model_loading_validated` is null"
            )
        if not analysis_cfg.model_analysis:
            notes.append("analysis_args.model_analysis is false: `analyzer.status` is 'skipped'")
        return notes

    # -- the reporting --------------------------------------------------

    def evaluate_and_report(
        self, config: ImageDenoisingConfig, data: Any, fit: FitState
    ) -> Evaluation:
        """Run the denoising reporting sequence; see the module contract."""
        from train.common import classification_run, run_summary

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        metric_cfg: DenoiseMetricArgs = config.metric_args
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        if checkpoint_cfg.save_best:
            # The in-memory model holds the BEST weights here (EarlyStopping restored them). There is no test split
            # (the held-out images are scored by the task from their own directories), so the reload check runs on
            # the validation patches and its metrics are discarded.
            _, load_error, max_diff = classification_run.check_best_checkpoint(
                fit.model,
                fit.run_dir,
                data.select_ds,
                evaluate=classification_run.evaluate_dict,
                split_label="Selection results (reload check only)",
            )
        # The analysis and the figures of a denoiser (a data-free analysis, a triptych, PSNR over sigma) are a later
        # step of the plan: nothing runs here, and the status says so rather than "missing".
        _, analysis = classification_run.report_best_weights(
            fit.model,
            data,
            fit.history,
            fit.run_dir,
            fit.vis_dir,
            fit.experiment_name,
            False,
            run_analysis=functools.partial(self.run_analysis, config),
            name_failed_analyses=True,
            figures=False,
            analysis_off_hint=(
                "the denoising task does not run an analysis yet"
                if analysis_cfg.model_analysis
                else "analysis_args.model_analysis=false"
            ),
            eval_batch_size=analysis_cfg.eval_batch_size,
        )
        # The last epoch's weights replace the best ones here.
        load_detail: Dict[str, Any] = {}
        test_final, load_check = classification_run.finalize_last_epoch(
            fit.model,
            fit.last_weights,
            data,
            fit.run_dir,
            fit.history,
            fit.epochs_run,
            evaluate=classification_run.evaluate_dict,
            save=checkpoint_cfg.save_last,
            atomic=True,
            eval_batch_size=analysis_cfg.eval_batch_size,
            exact_reload=True,
            load_detail=load_detail,
        )
        hist = {key: [float(v) for v in values] for key, values in fit.history.history.items()}
        start = self.selection_start_epoch(config)
        best = run_summary.best_epoch(hist, metric_cfg.watch_metric, metric_cfg.watch_mode, first_epoch=start)
        selection = {
            "start_epoch": start,
            "window_epochs": fit.epochs_run - start,
            "curriculum_epochs": config.data.curriculum_epochs,
            "best_epoch": best,
            "final_is_best": best == fit.epochs_run,
        }
        return Evaluation(
            test_metrics_best=None,
            best_load_error=load_error,
            checkpoint_max_diff=max_diff,
            conformal=None,
            visualizations={"files": [], "failed": [], "skipped": True},
            analysis=analysis,
            test_metrics_final=test_final,
            load_check=load_check,
            load_detail=load_detail or None,
            extras={
                "selection": selection,
                "held_out": {
                    "status": "skipped",
                    "reason": "the denoising task does not score held-out images yet: it has no data.test_dirs key",
                    "sigmas_255": [float(value) for value in metric_cfg.test_sigmas_255],
                },
            },
        )


def train_image_denoising(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[DenoisingData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    checkpoint: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    optim_args: Union[DenoiseOptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[DenoiseTransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[DenoiseMetricArgs, Mapping[str, Any], None] = None,
    loss_args: Union[DenoiseLossArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train an image denoiser; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_denoising_data_config(data)
    given = run_request(
        out=out, model=model, data=data, epochs=epochs, batch_size=batch_size, seed=seed,
        devices=devices, precision=precision, resume_interrupted=resume_interrupted,
        checkpoint=checkpoint, model_args=model_args,
        groups={
            "optim_args": optim_args,
            "transform_args": transform_args,
            "metric_args": metric_args,
            "loss_args": loss_args,
            "checkpoint_args": checkpoint_args,
            "logger_args": logger_args,
            "analysis_args": analysis_args,
            "debug_args": debug_args,
        },
    )
    config = validate(ImageDenoisingConfig, given)
    return ImageDenoisingTrainer(config).run()
