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

    The bias-free contract (``ModelSpec.properties`` holds ``"bias_free"``: ``bfcnn``, ``bfunet``, ``bfconvunext``; no
    other family is ever checked, and its summary says ``homogeneity: {"status": "not_applicable"}``). ``build_model``
    runs :func:`train.common.denoising_homogeneity.check_before_fit` on the model it just built, before ``compile`` and
    ``fit``: a static scan of the layers and a numeric probe of an init-probe CLONE (the model itself is never forwarded
    and no random state moves). Either failing is a ``ValueError`` the sequence reports as a ``ConfigError`` (exit 2)
    carrying the offenders or the numbers; a model that a family's own refusal tables let through but that is not
    homogeneous cannot start training. ``evaluate_and_report`` then probes the BEST and the FINAL weights on the first
    validation patches and returns ``extras["homogeneity"] = {status, static, numeric: {init, best, final}}``; a failed
    best or final reading enters ``Evaluation.failed_checks`` (exit 1, the run and its models are kept).

    The analysis (``run_analysis``, D-014). Data-free: the weights and spectral analyses only. The stock analysis
    reads the output as class probabilities (calibration, information flow, training dynamics: a denoiser's RGB is not
    that), so the task replaces its configuration with
    :func:`train.common.run_summary.data_free_analysis_config` and reads the status back with the data-free reader (the
    stock one calls an analysis that leaves ``model_metrics`` empty "missing"). The summary has no ``ece``; per-pixel
    uncertainty, the Jacobian asymmetry and the PSNR-over-sigma sweep of many models stay the ``train.bfunet`` eval
    scripts.

    The figures (``analysis_args.figures``). ``figure_callbacks`` is a per-epoch triptych
    (:class:`~train.common.denoising_run.DenoisingTriptychCallback`: noisy, denoised, clean and the removed residual of
    a fixed batch); at the end ``psnr_vs_sigma.png`` (one panel per held-out set) and ``residual_whiteness.png`` (the
    autocorrelation of the residual of the best weights) are drawn, and the dashboard of ``fit_callbacks`` has an
    additive PSNR panel. With the switch false none is written and ``visualizations`` says ``skipped``.

    The held-out cells (``data.test_dirs``, ``metric_args.test_sigmas_255``). Every set is listed and decoded once
    before ``fit`` (a leak into train or validation is refused before the claim). After ``fit`` the BEST weights, then
    the FINAL weights, score every whole image of every set at every sigma through
    :mod:`train.common.denoising_run` (reflect pad to the family's size multiple, the run's noise kind and clip, noise
    from a fixed constant seed that is not ``seed``, the same noisy image for both models, an image past the memory rule
    named and skipped): ``held_out[set][sigma] = {input_psnr, best_psnr, final_psnr, n}`` and ``held_out_info``. Without
    ``test_dirs``, ``held_out`` is ``{"status": "skipped", "reason", "sigmas_255"}``. No key averages over sets.

    ``task.json`` (``task_record``, schema in :mod:`train.api.predictor`): the family, the variant, the channels, the
    ``size_multiple`` (the family's stride, 1 for ``bfcnn`` and ``scunet``), ``data_range`` ``"[0,1]"`` and ``output``
    ``images``, plus ``variant_overrides`` when the recipe sets a key of the variant's table to another value. It has
    no input size (the model's input is ``(None, None, channels)``) and no statistics (the pixels are not
    standardized). Written once before ``fit``; a resume leaves it as it is and compares ``channels`` and
    ``data_range`` with the data it loads (D-027: not the held-out sets).

    The provenance stamps (``config_stamps``, D-011). ``config.json`` gets, at its top level beside the config's own
    fields, the keys the pre-API readers look for: ``data_range`` (the gate of ``denoiser_provenance``: without it
    ``eval_psnr_vs_noise``, ``eval_per_pixel_uncertainty``, ``DenoiserPrior`` and ``--init-from`` refuse the
    checkpoint), ``patch_size``, ``channels``, ``val_image_dirs`` (what ``eval_per_pixel_uncertainty`` builds its
    clean batch from; without it that script falls back to ITS OWN default directories without an error),
    ``noise_type``, ``noise_sigma_min``, ``sigma_max_end`` and, for ``bfconvunext`` only, ``convnext_version`` (the
    one bit ``DenoiserPrior`` reads to allow ``resolution="dynamic"``). Each is a copy of a value the config holds,
    written by a fresh run only. Limits, stated because they cannot be checked: a ``.keras`` file copied away from its
    ``config.json`` is refused by those readers (the gate reads the file beside it), and
    ``eval_per_pixel_uncertainty`` clips its noise always, so on a run trained with ``data.noise_clip=false`` its
    numbers describe clipped noise, not the run's.

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
    held_out_sets,
    load_denoising,
    load_denoising_data_config,
    read_denoising_data_file,
)
from ..denoising_models import TASK, compile_options_for
from ..errors import ConfigError
from ..predictor import build_denoising_record, overrides_of_variant, unit_domain_stamp
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
        #: ``{"static", "init"}`` of :func:`train.common.denoising_homogeneity.check_before_fit`, set by ``build_model``
        #: for a bias-free family (a resume builds the model again, so the report has it too), else ``None``.
        self._homogeneity: Optional[Dict[str, Any]] = None
        #: ``{name: [image files]}`` of ``data.test_dirs``, listed and decoded once by ``build_data`` (empty without it).
        self._held_out_sets: Dict[str, List[str]] = {}
        #: The multiple each side of an image is padded to for a whole-image forward pass, set by ``build_model``.
        self._size_multiple: int = 1

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
        # Listed and decoded here, before `fit`, so a held-out file that cannot be read is named at the start of the run
        # and not after hours of training (the leak into train or validation was refused in `describe_data`).
        self._held_out_sets = held_out_sets(config.data, channels=transform_cfg.channels)
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
        model = recipes.build_model(
            spec, context.variant, None, (None, None, data.input_shape[-1]),
            self._model_kwargs(config, context),
        )
        # The multiple a whole image is reflect-padded to for the held-out cells.
        self._size_multiple = self._size_multiple_of(spec, config, context)
        if "bias_free" in spec.properties:
            self._homogeneity = self._check_before_fit(model, data, config.precision)
        return model

    @staticmethod
    def _size_multiple_of(spec: registry.ModelSpec, config: ImageDenoisingConfig, context: AutoContext) -> int:
        """Return the family's stride, the multiple a whole image is padded to (1 for ``bfcnn`` and ``scunet``, which
        have none: ``stride_of`` says ``None``)."""
        return spec.stride_of(context.variant, config.model_args) or 1

    @staticmethod
    def _check_before_fit(model: Any, data: Any, precision: str) -> Dict[str, Any]:
        """Scan and probe a freshly built bias-free model; a refusal is a ``ValueError`` (see the class contract)."""
        from dl_techniques.utils.logger import logger
        from train.common import denoising_homogeneity as homogeneity

        # DECISION plan-2026-11-10T095307-d483ffab/D-025
        # Both checks run HERE, after the claim and setup_gpu, and not before the claim as the plan's step text says.
        # A model cannot be built before the claim: building any Keras model (all five families, also inside a
        # StatelessScope, measured) initialises TensorFlow's devices, and `setup_gpu(devices)` has to come first
        # (invariant 5). A throwaway build in `check_model_args` would break every bias-free run's device choice to
        # catch a defect no key can cause: every key that breaks homogeneity is already refused before the claim by
        # the family's refusal tables (`api/denoising_models.py`, D-015). Do NOT add that build; the cost of this
        # placement is one claimed `out` (the refusal says so) for a factory that ships a non-homogeneous layer.
        # A ValueError becomes the ConfigError of exit 2 in `run_sequence.build_model`. The probe reads a CLONE:
        # do NOT forward the live model here (see `init_probe_clone`).
        # DECISION plan-2026-11-10T095307-d483ffab/D-025
        # The numeric readings are taken only at float32: the tolerance is a float32 number (D-019), and a clean
        # bfcnn or bfunet built under mixed_float16 reads 1.3e-3 and 2.2e-3 on CPU (float16 rounding), so a
        # reading there would refuse every half-precision run D-012 keeps working. Do NOT widen the tolerance for
        # them (no measurement stands behind a wider number) and do NOT probe a float32 copy cloned from the
        # mixed model (tried: it still reads 4e-4 for a clean bfcnn). The static scan runs at every precision.
        checked = homogeneity.check_before_fit(model, homogeneity.probe_batch(data.select_ds), precision)
        if checked["init"]["passed"] is not None:
            logger.info(
                f"Bias-free check before fit: static scan clean, init probe max_rel {checked['init']['max_rel']:.3g} "
                f"(tolerance {checked['init']['tolerance']:g}, {checked['init']['patches']} validation patches)"
            )
        else:
            logger.warning(
                f"Bias-free check before fit: static scan clean; the numeric probe is skipped at precision "
                f"{precision!r} ({checked['init']['reason']})"
            )
        return checked

    @staticmethod
    def _model_kwargs(config: ImageDenoisingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's."""
        spec = registry.MODELS[context.family]
        return spec.load_resolver()(context, config.optim_args, config.model_args)

    def used_arguments(self, config: ImageDenoisingConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def task_record(
        self, config: ImageDenoisingConfig, context: AutoContext, data: Any
    ) -> Dict[str, Any]:
        """Return the denoising ``task.json`` record (schema in :mod:`train.api.predictor`).

        Called before ``build_model`` on a resume, so the size multiple comes from the registry and the config and not
        from ``self._size_multiple``.
        """
        spec = registry.MODELS[context.family]
        return build_denoising_record(
            model=config.model,
            family=context.family,
            variant=context.variant,
            channels=data.input_shape[-1],
            size_multiple=self._size_multiple_of(spec, config, context),
            variant_overrides=overrides_of_variant(
                spec.load_variants()[context.variant], self._model_kwargs(config, context)
            ),
        )

    def config_stamps(self, config: ImageDenoisingConfig, spec: registry.ModelSpec) -> Dict[str, Any]:
        """Return the top-level ``config.json`` keys the pre-API denoiser readers look for (D-011).

        Every value is a copy of one the config holds (``test_denoising_provenance`` holds them equal);
        ``val_image_dirs`` is as given, so a relative directory is read relative to the working directory of the
        script that reads it (a stated limit: use absolute directories for a run an eval script will read). The
        ``convnext_version`` of ``bfconvunext`` is the RESOLVED one: the recipe's model arguments (what
        ``context.defaults`` holds) with the caller's ``model_args`` over them, as the resolver merges them.
        """
        # DECISION plan-2026-11-10T095307-d483ffab/D-011
        # These keys are the contract with four readers that predate the API; each one removed breaks the path it
        # serves, and `val_image_dirs` breaks it SILENTLY (`eval_per_pixel_uncertainty._load_eval_config` falls back
        # to the legacy default directories). Do NOT trim a stamp as "a duplicate of the config", do NOT move the
        # readers to the API's keys (they are not ours to edit, and an old run would stop loading), and do NOT stamp
        # `convnext_version` for another family: `DenoiserPrior` sniffs its PRESENCE to allow the dynamic loader.
        data_cfg: DenoisingData = config.data
        transform_cfg: DenoiseTransformArgs = config.transform_args
        stamps: Dict[str, Any] = {
            "data_range": unit_domain_stamp(),
            "patch_size": transform_cfg.patch_size,
            "channels": transform_cfg.channels,
            "val_image_dirs": list(data_cfg.val_dirs),
            "noise_type": data_cfg.noise_kind,
            "noise_sigma_min": data_cfg.noise_sigma_min,
            "sigma_max_end": data_cfg.noise_sigma_max,
        }
        if "bias_free" in spec.properties:
            variant = registry.parse_model_name(config.model, TASK)[1]
            model_args = {**spec.recipe(data_cfg.format, variant).get("model_args", {}), **config.model_args}
            if "convnext_version" in model_args:  # the bias-free ConvUNext (bfcnn and bfunet hold no such key)
                stamps["convnext_version"] = model_args["convnext_version"]
        return stamps

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

    def figure_callbacks(self, config: ImageDenoisingConfig, data: Any, vis_dir: Path) -> List[Any]:
        """Return the per-epoch triptych of the first fixed validation patches (noisy, denoised, clean, residual)."""
        from train.common import denoising_run

        noisy, clean = denoising_run.first_pairs(data.select_ds, denoising_run.TRIPTYCH_PATCHES)
        return [denoising_run.DenoisingTriptychCallback(noisy, clean, vis_dir, title=vis_dir.parent.name)]

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
            "reload check ran on the validation patches (`best_checkpoint_max_abs_diff`); the held-out images are "
            "scored separately (see `held_out`)",
            "`task.json` and the `config.json` stamps for the pre-API eval scripts are written; the denoising predictor "
            "is not part of this version yet, so `predict` and `export` refuse this run",
        ]
        if data_cfg.test_dirs:
            notes.append(
                "`held_out[set][sigma]` is the mean over the images of that set of each image's PSNR (max_val 1, the "
                "whole image, reflect-padded to the family's size multiple and cropped back) of the noisy input "
                "(`input_psnr`), the best model (`best_psnr`) and the final model (`final_psnr`), `n` images, with the "
                f"run's noise kind ({data_cfg.noise_kind}) and clip ({data_cfg.noise_clip}) and the SAME noisy image "
                "for both models (a fixed seed that is not `seed`: see `held_out_info`); a set is a row of its own, "
                "no key averages sets, and an image past the memory rule is named in `held_out_info.skipped_images`"
            )
        else:
            notes.append("`held_out` is skipped: data.test_dirs is not set, so no whole image was scored")
        if analysis_cfg.model_analysis:
            notes.append(
                "the model analysis is data-free (weights and spectral only; `analyzer.analyzers` lists them): the "
                "stock calibration, information-flow and training-dynamics analyses read class probabilities, which a "
                "denoiser's image output is not, so there is no `ece`"
            )
        if self._homogeneity is not None:
            notes.append(
                "`homogeneity` reads max|f(a x) - a f(x)| / max|a f(x)| at alphas 0.3, 1.3, 3.7, 6.1 (TF32 off, "
                "inference mode) on the first validation patches: `init` on a clone of the untrained model with layer "
                "scales at 1 and random running variances, `best` and `final` on the weights as trained (all three "
                "are `skipped`, with the reason, at a precision other than float32: the tolerance is a float32 "
                "number, and the static scan still ran); a failed "
                "`best` or `final` reading makes `success` false and names `homogeneity_best` or `homogeneity_final` "
                "in `success_blocked_by`. The reading is a property of the network, whatever the noise "
                f"(this run's noise is {data_cfg.noise_kind}: see `score_identity` for what the residual means)"
            )
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

    def run_analysis(
        self,
        config: ImageDenoisingConfig,
        model: Any,
        test_data: Tuple[Any, Any],
        history: Any,
        experiment_name: str,
        run_dir: str,
    ) -> Any:
        """Run the data-free analysis (weights and spectral only); see the module contract."""
        from threadpoolctl import threadpool_limits

        from train.common import evaluation, run_summary

        # DECISION plan-2026-11-10T095307-d483ffab/D-014
        # The analysis of a denoiser is DATA-FREE. The default `run_analysis` picks its configuration by the output
        # rank, and a denoiser's per-pixel output (`outputs_per_pixel(bfcnn)` is true) gets calibration, information
        # flow and training dynamics, which read the RGB image as class probabilities and the clean image as labels:
        # numbers that mean nothing and cost minutes. Do NOT fall back to `evaluation.analysis_config_for` here, and do
        # NOT add calibration to this configuration (`test_image_denoising_task` runs both on one model: the control
        # through `analysis_config_for` produces an ECE, this one leaves `calibration_metrics` empty). Per-pixel
        # uncertainty, the Jacobian asymmetry and the PSNR sweep stay the `train.bfunet` eval scripts. The BLAS limit
        # is `run_data_free_analysis`'s (D-243 of plan 81938e81), the seed the default `run_analysis`'s (D-027).
        analysis_config = dataclasses.replace(run_summary.data_free_analysis_config(), random_state=config.seed)
        with threadpool_limits(limits=run_summary.DATA_FREE_ANALYSIS_BLAS_THREADS):
            results = evaluation.run_model_analysis(
                model, test_data, history, experiment_name, run_dir, config=analysis_config
            )
        if results is not None:
            # The library's own four-panel dashboard reads "No data available" for a data-free analysis.
            run_summary.remove_empty_summary_dashboard(Path(run_dir), results)
        return results

    def evaluate_and_report(
        self, config: ImageDenoisingConfig, data: Any, fit: FitState
    ) -> Evaluation:
        """Run the denoising reporting sequence; see the module contract."""
        import keras
        import numpy as np

        from train.common import classification_run, denoising_run, run_summary
        from train.common import denoising_homogeneity as homogeneity

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        metric_cfg: DenoiseMetricArgs = config.metric_args
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        # The bias-free probes read the same first validation patches the pre-fit check read. The in-memory model
        # holds the BEST weights until `finalize_last_epoch` swaps the last epoch's in, so the best reading comes
        # first and the final one after it.
        bias_free = self._homogeneity is not None
        probe_inputs = homogeneity.probe_batch(data.select_ds) if bias_free else None
        best_reading = homogeneity.read_weights(fit.model, probe_inputs, config.precision) if bias_free else None
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
        # DECISION plan-2026-11-10T095307-d483ffab/D-026
        # Everything that describes the BEST weights is taken HERE, before `finalize_last_epoch` puts the last
        # epoch's weights in the model: the residual of the whiteness figure, the best held-out pass and the analysis.
        # The final held-out pass comes after it, on the same noisy images (`pair_held_out` refuses the pair otherwise).
        # Do NOT move any of them after the swap (the "best" numbers would be the final model's).
        noisy_fig, clean_fig = denoising_run.first_pairs(data.select_ds, denoising_run.WHITENESS_PATCHES)
        residual = None
        if analysis_cfg.figures:
            residual = noisy_fig - np.asarray(keras.ops.convert_to_numpy(fit.model(noisy_fig, training=False)))
        held_spec = self._held_out_spec(config)
        best_scores = self._score_held_out(fit.model, held_spec) if self._held_out_sets else None
        _, analysis = classification_run.report_best_weights(
            fit.model,
            denoising_run.AnalysisSample(noisy_fig, clean_fig),
            fit.history,
            fit.run_dir,
            fit.vis_dir,
            fit.experiment_name,
            analysis_cfg.model_analysis,
            run_analysis=functools.partial(self.run_analysis, config),
            name_failed_analyses=True,
            figures=False,
            analysis_off_hint="analysis_args.model_analysis=false",
            eval_batch_size=analysis_cfg.eval_batch_size,
            read_status=run_summary.read_data_free_analysis_status,
        )
        if analysis.get("failed_analyses") and analysis["status"] == "ok":
            # `report_best_weights` marks only the stock reader's "success" partial; this reader's good status is "ok".
            analysis["status"] = run_summary.PARTIAL_ANALYSIS_STATUS
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
        held_out, held_out_info, cells = self._held_out_block(config, fit, held_spec, best_scores)
        visualizations: Dict[str, Any] = {"files": [], "failed": [], "skipped": True}
        if analysis_cfg.figures:
            visualizations = denoising_run.write_end_figures(
                fit.vis_dir, cells=cells, residual=residual, title=fit.experiment_name
            )
        report, failed = self._homogeneity_report(config, fit, best_reading, probe_inputs)
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
        extras: Dict[str, Any] = {"selection": selection, "homogeneity": report, "held_out": held_out}
        if held_out_info is not None:
            extras["held_out_info"] = held_out_info
        return Evaluation(
            test_metrics_best=None,
            best_load_error=load_error,
            checkpoint_max_diff=max_diff,
            conformal=None,
            visualizations=visualizations,
            analysis=analysis,
            test_metrics_final=test_final,
            load_check=load_check,
            load_detail=load_detail or None,
            extras=extras,
            failed_checks=failed,
        )

    def _held_out_spec(self, config: ImageDenoisingConfig) -> Any:
        """Return the :class:`~train.common.denoising_run.HeldOutSpec` of this run: its noise, channels and multiple."""
        from train.common import denoising_run
        from train.common.denoising_data import DEFAULT_COMPOSITE_RATIO

        data_cfg: DenoisingData = config.data
        ratio = DEFAULT_COMPOSITE_RATIO if data_cfg.noise_composite_ratio is None else data_cfg.noise_composite_ratio
        return denoising_run.HeldOutSpec(
            sigmas_255=tuple(float(value) for value in config.metric_args.test_sigmas_255),
            noise_kind=data_cfg.noise_kind,
            composite_ratio=ratio,
            clip=data_cfg.noise_clip,
            channels=config.transform_args.channels,
            size_multiple=self._size_multiple,
        )

    def _score_held_out(self, model: Any, spec: Any) -> Any:
        """Score ``model`` as it is (whole images, eager forward pass) on every held-out set."""
        from train.common import denoising_run

        return denoising_run.score_held_out(
            lambda noisy: denoising_run.denoise_whole_image(model, noisy, spec.size_multiple, eager=True),
            self._held_out_sets,
            spec,
        )

    def _held_out_block(
        self, config: ImageDenoisingConfig, fit: FitState, spec: Any, best_scores: Any
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Return ``(held_out, held_out_info, cells)``: the summary blocks and the cells for the figure.

        Called after ``finalize_last_epoch``: ``fit.model`` holds the FINAL weights, ``best_scores`` is the pass
        taken on the best ones (``None`` when no set was asked for).
        """
        from dl_techniques.utils.logger import logger
        from train.common import denoising_run

        if best_scores is None:
            return (
                {
                    "status": "skipped",
                    "reason": "data.test_dirs is not set: no held-out image was scored",
                    "sigmas_255": [float(value) for value in spec.sigmas_255],
                },
                None,
                None,
            )
        cells = denoising_run.pair_held_out(best_scores, self._score_held_out(fit.model, spec))
        for name, per_sigma in cells.items():
            for key, cell in per_sigma.items():
                if cell["n"]:
                    logger.info(
                        f"Held-out {name} sigma_255 {key} (n={cell['n']}): input {cell['input_psnr']:.2f} dB, best "
                        f"{cell['best_psnr']:.2f} dB, final {cell['final_psnr']:.2f} dB"
                    )
        info = {
            "sigmas_255": [float(value) for value in spec.sigmas_255],
            "noise": {
                "kind": spec.noise_kind,
                "composite_ratio": spec.composite_ratio if spec.noise_kind == "composite" else None,
                "clip": spec.clip,
            },
            "seed": spec.seed,
            "size_multiple": spec.size_multiple,
            "channels": spec.channels,
            "psnr": "mean over the images of each whole image's PSNR at max_val 1.0 (MSE floored at 1e-12)",
            "memory_rule": {
                "gb_per_megapixel": denoising_run.HELD_OUT_GB_PER_MEGAPIXEL,
                "budget_gb": spec.budget_gb,
            },
            "skipped_images": {name: rows for name, rows in best_scores.skipped.items() if rows},
        }
        return cells, info, cells

    def _homogeneity_report(
        self, config: ImageDenoisingConfig, fit: FitState, best_reading: Optional[Dict[str, Any]], probe_inputs: Any
    ) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
        """Return the summary's ``homogeneity`` block and the failed checks (``()`` for a family never checked).

        Called after ``finalize_last_epoch``: ``fit.model`` holds the FINAL weights, ``best_reading`` is the probe
        taken on the best ones.
        """
        from dl_techniques.utils.logger import logger
        from train.common import denoising_homogeneity as homogeneity

        if self._homogeneity is None:
            return (
                {
                    "status": "not_applicable",
                    "reason": "the family is not bias-free (its ModelSpec has no 'bias_free' property): no "
                    "homogeneity check ran and none was asked for",
                },
                (),
            )
        report = homogeneity.homogeneity_report(
            self._homogeneity["static"],
            self._homogeneity["init"],
            best_reading,
            homogeneity.read_weights(fit.model, probe_inputs, config.precision),
        )
        failed = homogeneity.failed_checks(report)
        readings = ", ".join(
            f"{name} {reading['max_rel']:.3g}" if reading["passed"] is not None else f"{name} skipped"
            for name, reading in report["numeric"].items()
        )
        logger.info(f"Bias-free check after fit: {report['status']} (max_rel {readings})")
        if failed:
            logger.warning(
                f"The trained weights are not degree-1 homogeneous: failed {list(failed)}; the run and its models "
                "are kept and `success` is false"
            )
        return report, failed


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
