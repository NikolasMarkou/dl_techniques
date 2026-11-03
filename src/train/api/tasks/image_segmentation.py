"""The image-segmentation task: a config, a trainer and the keyword-only function.

``train_image_segmentation(...)`` is the function a Python caller uses and the function
``python -m train image_segmentation key=value ...`` calls (D-008), so the two cannot disagree. It
validates its keywords into an :class:`ImageSegmentationConfig` and hands that to
:class:`ImageSegmentationTrainer`, a :class:`~train.api.trainer.TaskTrainer` that supplies the hooks
of this task and nothing else: the sequence is the shared one (D-070: a task overrides hooks, never
stages). The hooks are the segmentation assumptions of the run: pairs of an image and a mask of
class ids, the cross-entropy against ``ln(C)`` at epoch 0, the ``val_miou`` success value and its
bar (the mIoU of predicting the majority class), the epoch line, the summary fields and notes, and
the reporting after ``fit`` (test metrics of the best and the last weights, the model analyzer on
the per-pixel path).

Interface contract
------------------
``ImageSegmentationConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``,
    ``model`` and ``data`` are required. The groups are those of classification except
    ``transform_args`` (:class:`~train.api.segmentation_args.SegTransformArgs`) and ``metric_args``
    (:class:`~train.api.segmentation_args.SegMetricArgs`). ``analysis_args.conformal_alpha`` is
    refused: a conformal set is a set of classes per image, and a segmentation run has none.

``ImageSegmentationTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "image_segmentation"`` and
    ``success_metric = "val_miou"``. ``describe_data`` LISTS the files (a sample of the masks is
    read with PIL, no TensorFlow op); ``build_data`` goes through
    :func:`~train.api.segmentation_args.load_seg_data`; ``build_model`` through the family's
    resolver and :func:`~train.api.recipes.build_model`; the loss and metrics are
    :func:`~train.common.segmentation_run.segmentation_loss` and
    :func:`~train.common.segmentation_run.segmentation_metrics`; ``task_record`` is the version 2
    segmentation ``task.json``.

    What ``loss`` and ``val_loss`` are (D-076, D-077): Keras averages the cross-entropy of a batch over
    its valid pixels and then averages the batches' values weighted by IMAGES, so ``val_loss`` is not
    the cross-entropy pooled over every valid pixel whenever the void share differs between batches.
    ``accuracy`` and ``miou`` ARE pooled over every valid pixel. The best epoch is therefore chosen by
    ``val_miou`` (``metric_args.watch_metric``); the summary says all of this in ``notes``.

    The success bar is ``metric_args.success_threshold`` when given, else the mIoU of predicting the
    majority class of the selection split for every valid pixel (``majority_class_baseline`` in the
    summary): a run that cannot beat a constant has learned no segmentation.

``train_image_segmentation(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"family/variant"``, e.g. ``"convunext/tiny"`` or ``"accunet/base"``.
    :param data: a mapping (``{"dataset": "oxford_pet"}``), a
        :class:`~train.api.segmentation_args.BuiltinSegData`, ``FolderSegData`` or ``CsvSegData`` (a
        mapping's ``format`` says which; ``builtin`` when absent), or the path of a YAML file
        holding one.
    :param epochs: / ``batch_size`` / ``devices`` / ``seed`` / ``precision`` / ``resume_interrupted`` /
        ``checkpoint`` / ``model_args`` and the groups: as in
        :func:`~train.api.tasks.image_classification.train_image_classification`.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see
        :meth:`~train.api.trainer.TaskTrainer.run`. A family that does not train segmentation, an
        ``image_size`` side that is not a multiple of the family's stride, a source with no valid
        pair and a mask value that is neither a class nor ``ignore_index`` are refused before the
        run directory is claimed.
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api`` (D-012):
``python -m train image_segmentation help`` imports this module to print the config's keys and must
stay free of TensorFlow. Every ``keras`` and ``train.common`` name is imported inside a method.
"""

import dataclasses
import functools
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .. import recipes, registry
from ..args import (
    AnalysisArgs,
    AutoContext,
    CheckpointArgs,
    DebugArgs,
    LoggerArgs,
    OptimArgs,
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
from ..errors import ConfigError
from ..predictor import build_segmentation_record, overrides_of_variant
from ..segmentation_args import (
    SegData,
    SegMetricArgs,
    SegTransformArgs,
    load_seg_data,
    load_seg_data_config,
    read_seg_data_file,
    seg_facts,
)
from ..segmentation_models import TASK, check_image_size, sides_of
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, config_problems

#: The per-epoch grid draws this many fixed test pairs, every this many epochs (the legacy trainer's
#: ``--viz-samples`` and ``--viz-freq`` defaults; they are not keys of a run), and the end-of-run grid
#: of the best weights draws the first :data:`FINAL_GRID_ROWS` test pairs.
GRID_SAMPLES = 4
GRID_EVERY = 1
FINAL_GRID_ROWS = 6

# ---------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ImageSegmentationConfig(StrictConfig):
    """Everything a segmentation run is told; see the module contract."""

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={
            "help": "family/variant, e.g. convunext/tiny or accunet/base; python -m train "
            "list_models image_segmentation lists them"
        }
    )
    data: SegData = dataclasses.field(
        metadata={
            "help": "the image and mask pairs and their split options: data.dataset=oxford_pet, or "
            "data.format=folder data.train_images=... data.train_masks=..., or data.format=csv "
            "data.train_csv=...; or the path of a YAML file holding them (data=data.yaml)",
            PATH_FORM: read_seg_data_file,
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
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: SegTransformArgs = dataclasses.field(default_factory=SegTransformArgs)
    metric_args: SegMetricArgs = dataclasses.field(default_factory=SegMetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range, and a conformal level (a classification tool)."""
        lines = config_problems(self, prefix)
        # DECISION plan-2026-10-31T180033-2faf890a/D-077
        # `analysis_args.conformal_alpha` is REFUSED, not ignored: it is a field of the shared group
        # and a quiet no-op would look like a conformal run that found nothing. Do NOT give
        # segmentation its own analysis group to hide the key (four fields and two methods copied,
        # to be kept equal by hand); the refusal names the key and the reason.
        alpha = self.analysis_args.conformal_alpha
        if alpha is not None:
            lines.append(
                f"Invalid value for {join_path(join_path(prefix, 'analysis_args'), 'conformal_alpha')!r}: "
                "a conformal prediction set is a set of classes for a whole image, which a "
                f"segmentation run does not produce; leave it null, got {alpha!r}"
            )
        return lines


class ImageSegmentationTrainer(TaskTrainer):
    """The hooks of an image-segmentation run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    """

    task = TASK
    success_metric = "val_miou"
    initial_loss_hint = (
        "Lower the dropout or drop-path rates in model_args, or check the model and the masks."
    )

    def __init__(self, config: StrictConfig) -> None:
        super().__init__(config)
        #: The majority-class baseline of the selection split, set by ``build_data`` (a resume
        #: builds the data again, so it is always there before the summary or the verdict).
        self._baseline: Optional[Dict[str, Any]] = None

    # -- data and model -------------------------------------------------

    def describe_data(self, config: ImageSegmentationConfig) -> DataFacts:
        transform_cfg: SegTransformArgs = config.transform_args
        size = transform_cfg.image_size
        facts = seg_facts(config.data, None if size == AUTO else sides_of(size))
        return DataFacts(dataset=facts.dataset, num_classes=facts.num_classes, image_size=None)

    def build_data(self, config: ImageSegmentationConfig, context: AutoContext) -> Any:
        import time

        from dl_techniques.utils.logger import logger
        from train.common import segmentation_data, segmentation_run

        transform_cfg: SegTransformArgs = config.transform_args
        size = sides_of(transform_cfg.image_size)
        data = load_seg_data(
            config.data,
            batch_size=config.batch_size,
            seed=config.seed,
            image_size=size,
            augment=segmentation_data.build_segmentation_augment(transform_cfg.augment, size),
            eval_batch_size=config.analysis_args.eval_batch_size,
        )
        started = time.perf_counter()
        counts = segmentation_run.class_pixel_counts(
            data.select_ds, data.num_classes, data.ignore_index
        )
        self._baseline = segmentation_run.majority_class_baseline(counts, data.class_names)
        logger.info(
            f"Majority-class baseline of the {data.n_select} selection pairs: always "
            f"{self._baseline['class_name']!r} gives mIoU {self._baseline['miou']:.4f} and pixel "
            f"accuracy {self._baseline['pixel_accuracy']:.4f} "
            f"({time.perf_counter() - started:.1f} s)"
        )
        return data

    def build_model(
        self, config: ImageSegmentationConfig, context: AutoContext, data: Any
    ) -> Any:
        spec = registry.MODELS[context.family]
        return recipes.build_model(
            spec, context.variant, data.num_classes, data.input_shape,
            self._model_kwargs(config, context),
        )

    @staticmethod
    def _model_kwargs(config: ImageSegmentationConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's."""
        spec = registry.MODELS[context.family]
        return spec.load_resolver()(context, config.optim_args, config.model_args)

    def used_arguments(
        self, config: ImageSegmentationConfig, context: AutoContext
    ) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json`` (D-050): the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def check_model_args(self, config: ImageSegmentationConfig, context: AutoContext) -> None:
        # Before the claim: the adapter's own keys are refused by the resolver, and each side of
        # the image size must be a multiple of the family's stride (D-072), which depends on the
        # variant and on a ``model_args.depth``.
        self._model_kwargs(config, context)
        transform_cfg: SegTransformArgs = config.transform_args
        check_image_size(
            registry.MODELS[context.family], context.variant, config.model_args,
            transform_cfg.image_size,
        )

    def task_record(
        self, config: ImageSegmentationConfig, context: AutoContext, data: Any
    ) -> Dict[str, Any]:
        """Return the segmentation ``task.json`` record (schema in :mod:`train.api.predictor`).

        The inputs lie in [0, 1] and are never standardized, so ``normalize`` is ``unit`` with
        zeros and ones (D-073), and ``output`` is ``masks``: what ``predict`` answers.
        """
        return build_segmentation_record(
            model=config.model,
            family=context.family,
            variant=context.variant,
            class_names=data.class_names,
            input_size=data.input_size,
            channels=data.input_shape[-1],
            normalize="unit",
            mean=data.mean,
            std=data.std,
            ignore_index=data.ignore_index,
            variant_overrides=overrides_of_variant(
                registry.MODELS[context.family].load_variants()[context.variant],
                self._model_kwargs(config, context),
            ),
        )

    def input_check(self, facts: DataFacts, data: Any) -> None:
        if data.num_classes != facts.num_classes:
            raise RuntimeError(
                f"{type(self).__name__}.describe_data said {facts.num_classes} classes, but "
                f"build_data loaded {data.num_classes}"
            )

    # -- compile --------------------------------------------------------

    def metric_names(self, config: ImageSegmentationConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates
        # variables and the first eager op would initialise TensorFlow's devices before setup_gpu.
        from train.common.segmentation_run import SEGMENTATION_METRIC_NAMES

        return list(SEGMENTATION_METRIC_NAMES)

    def build_loss_and_metrics(
        self, config: ImageSegmentationConfig, data: Any
    ) -> Tuple[Any, Any]:
        from train.common.segmentation_run import segmentation_loss, segmentation_metrics

        return (
            segmentation_loss(data.ignore_index),
            segmentation_metrics(data.num_classes, data.ignore_index),
        )

    def initial_loss_check(
        self, config: ImageSegmentationConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the cross-entropy over ``ln(num_classes)``.

        ``ln(C)`` is the loss of a model that predicts every class equally at every valid pixel,
        which holds for the cross-entropy and for no other loss this task will offer (step 35's
        dice and focal losses have their own scale), so a loss other than ``ce`` states its own
        reference here instead of reusing this one.
        """
        from train.common import classification_run

        return classification_run.check_initial_loss(
            model,
            data.select_ds,
            data.n_select,
            data.num_classes,
            evaluate=classification_run.evaluate_dict,
            hint=self.initial_loss_hint,
        )

    # -- the verdict ----------------------------------------------------

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        return max(hist[self.success_metric])

    def success_threshold(self, config: ImageSegmentationConfig) -> float:
        metric_cfg: SegMetricArgs = config.metric_args
        # DECISION plan-2026-10-31T180033-2faf890a/D-077
        # The bar is the mIoU of the majority-class predictor of the SELECTION split (worked out by
        # `build_data`, which a resume runs again), unless the caller states one. Do NOT use
        # `2 / num_classes` (the classification bar: twice chance for a classifier, but a constant
        # segmenter already scores its majority share in pixel accuracy, 0.598 on the Oxford Pet
        # selection split as measured) and do NOT key the verdict on `val_accuracy`: pixel accuracy
        # rewards the dominant class, and the fixtures of `test_image_segmentation_task` have a
        # `val_accuracy` above `val_miou`.
        if metric_cfg.success_threshold is not None:
            return metric_cfg.success_threshold
        if self._baseline is None:
            raise RuntimeError(
                "success_threshold is the majority-class baseline of the selection split, which "
                "build_data works out; it was asked for before the data was built"
            )
        return float(self._baseline["miou"])

    def bar_is_met(self, config: ImageSegmentationConfig, reached: float, threshold: float) -> bool:
        # DECISION plan-2026-10-31T180033-2faf890a/D-087
        # Against the majority-class baseline the best `val_miou` must be STRICTLY above it,
        # compared in float32 with a tie band (`segmentation_run.beats_baseline`): the constant
        # majority model scores the baseline exactly, and `reached >= threshold` called it a
        # success in 8 of 20 measured cases. A bar the caller STATED
        # (`metric_args.success_threshold`) keeps the shared "reached" rule, `>=`.
        if config.metric_args.success_threshold is not None:
            return bool(reached >= threshold)
        from train.common import segmentation_run

        return segmentation_run.beats_baseline(reached, threshold)

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: ImageSegmentationConfig) -> Tuple[str, ...]:
        return ("loss", "accuracy", "miou", "val_loss", "val_accuracy", "val_miou")

    def figure_callbacks(self, config: ImageSegmentationConfig, data: Any, vis_dir: Path) -> List[Any]:
        """Return the per-epoch grid of the first test pairs held in memory; none without a test split.

        The callback wants ``uint8`` images and divides them by 255; ``data.x_fig`` holds the pipeline's own
        bytes divided by 255, so the conversion is exact and the grid shows what the model was given. The
        colour table is the run's (``extended_palette``) and the void value is the data's.
        """
        import numpy as np

        from train.convunext import segmentation_viz as viz

        if len(data.x_fig) == 0:
            return []
        return [
            viz.SegmentationGridCallback(
                np.rint(np.asarray(data.x_fig) * 255.0).astype(np.uint8),
                data.y_fig,
                vis_dir,
                data.class_names,
                GRID_EVERY,
                GRID_SAMPLES,
                config.seed,
                title=vis_dir.parent.name,
                palette=viz.extended_palette(len(data.class_names)),
                ignore_index=data.ignore_index,
                # The grid draws TEST pairs and `val_miou` is scored on the selection split: say so.
                val_population=f"all {data.n_select} selection pairs; the pairs shown are test pairs",
            )
        ]

    def summary_fields(self, config: ImageSegmentationConfig, data: Any) -> Dict[str, Any]:
        return {
            "figure_samples": int(len(data.x_fig)),
            "input_size": list(data.input_size),
            "class_names": list(data.class_names),
            "ignore_index": data.ignore_index,
            "mask_values": list(data.mask_values),
            "majority_class_baseline": self._baseline,
        }

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        notices: List[Tuple[str, str]] = []
        if facts.dataset in registry.FILE_DATASETS:
            notices.append(
                (
                    "info",
                    f"{facts.dataset} data: no legacy trainer reads these files, so the {family} "
                    "recipe is its base recipe with transform_args.augment=flip (D-060) and "
                    "transform_args.image_size is the one you gave",
                )
            )
        if family == "accunet":
            notices.append(
                (
                    "info",
                    "accunet: the legacy trainer (train.accunet) trains the combo loss, which this "
                    "task does not offer yet; it trains the cross-entropy, so its numbers are not "
                    "that trainer's (D-077)",
                )
            )
            notices.append(
                (
                    "info",
                    "accunet: the recipe's batch is 8, the legacy trainer's 16 halved: batch 16 runs "
                    "out of memory at the first step on a 24 GB card (D-088); the learning rate is "
                    "not scaled",
                )
            )
        return notices

    def task_notes(
        self,
        config: ImageSegmentationConfig,
        baseline: Dict[str, float],
        ratio: float,
        data: Any,
    ) -> List[str]:
        """Return the segmentation lines of ``results_summary.json``'s ``notes``."""
        import numpy as np

        from train.common import classification_run, segmentation_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        metric_cfg: SegMetricArgs = config.metric_args
        majority = self._baseline or {}
        notes = [
            f"initial loss {baseline['loss']:.4f} vs ln(C)={np.log(data.num_classes):.4f} "
            f"(ratio {ratio:.2f}): the cross-entropy per valid pixel, a plain evaluate on the "
            "validation split",
            *classification_run.RUN_NOTES,
            "`loss` and `val_loss` are Keras' mean over batches of each batch's cross-entropy "
            "per valid pixel, the batches weighted by images, NOT the cross-entropy pooled over "
            "every valid pixel: with a void label the two differ whenever the void share differs "
            "between batches. `accuracy` and `miou` are pooled over every valid pixel. The best "
            f"epoch is chosen by `{metric_cfg.watch_metric}` (`metric_args.watch_metric`)",
            f"`success_threshold` is {'metric_args.success_threshold' if metric_cfg.success_threshold is not None else 'the mIoU of the majority-class baseline'}"
            f" ({majority.get('class_name')!r} for every valid pixel of the selection split: mIoU "
            f"{majority.get('miou')}, pixel accuracy {majority.get('pixel_accuracy')}); "
            "`majority_class_baseline` holds its numbers",
        ]
        if metric_cfg.success_threshold is None:
            notes.append(
                "`success` needs the best `val_miou` STRICTLY above that baseline: compared in "
                f"float32 and cleared by {segmentation_run.TIE_ULPS} float32 steps of the baseline "
                "(about 1e-6), because a model that predicts the majority class everywhere scores "
                "the baseline and Keras' float32 figure for it differs from the float64 baseline "
                "by rounding"
            )
        if data.ignore_index is not None:
            notes.append(
                f"mask value {data.ignore_index} is void (`ignore_index`): it is left out of the "
                "loss, `accuracy`, `miou`, the baseline and the analyzer's calibration"
            )
        if data.test_ds is None:
            notes.append(
                "the data has no test split: `test_metrics_best`, `test_metrics_final` are null, "
                "no figure was written and the model analysis was skipped; the best_model.keras "
                "reload check ran on the selection split (`best_checkpoint_max_abs_diff`)"
            )
        if not checkpoint_cfg.save_best:
            notes.append(
                "checkpoint_args.save_best is false: no best_model.keras was written, so "
                "`test_metrics_best` is null; the analysis uses the in-memory best weights"
            )
        if not checkpoint_cfg.save_last:
            notes.append(
                "checkpoint_args.save_last is false: no final_model.keras was written and "
                "`model_loading_validated` is null"
            )
        if not analysis_cfg.model_analysis:
            notes.append("analysis_args.model_analysis is false: `analyzer.status` is 'skipped'")
        if not analysis_cfg.figures:
            notes.append("analysis_args.figures is false: no figure was written")
        elif data.test_ds is not None:
            notes.append(
                f"figures are drawn on the first {len(data.x_fig)} test pairs held in memory "
                "(`data.eval_samples`); `per_class_scores.png` and `segmentation_report.json` are the "
                "pooled confusion of the best weights over every pixel of the test split, void left "
                "out, and their mIoU and pixel accuracy equal `test_metrics_best`'s"
            )
        return notes

    # -- the reporting --------------------------------------------------

    def analysis_ignore_label(self, config: ImageSegmentationConfig) -> Optional[int]:
        # Oxford-IIIT Pet has no void label and no such key.
        return getattr(config.data, "ignore_index", None)

    def _figures(self, config: ImageSegmentationConfig, data: Any, fit: FitState) -> Dict[str, Any]:
        """Draw the end-of-run figures of the BEST weights (still in the model) and return the record.

        One prediction pass over the whole test split builds the pooled confusion; the grid of
        predictions is the first :data:`FINAL_GRID_ROWS` test pairs.
        """
        import numpy as np

        from train.common import run_summary, segmentation_figures, segmentation_run

        metric_cfg: SegMetricArgs = config.metric_args
        hist = {key: [float(v) for v in values] for key, values in fit.history.history.items()}
        best_epoch = run_summary.best_epoch(hist, metric_cfg.watch_metric, metric_cfg.watch_mode)
        rows = min(FINAL_GRID_ROWS, len(data.x_fig))
        logits = fit.model(np.asarray(data.x_fig[:rows]), training=False)
        return segmentation_figures.write_segmentation_figures(
            fit.vis_dir,
            data.x_fig[:rows],
            data.y_fig[:rows],
            np.asarray(logits).argmax(axis=-1),
            segmentation_run.confusion_from_dataset(
                fit.model, data.test_ds, data.num_classes, data.ignore_index
            ),
            hist,
            best_epoch,
            list(data.class_names),
            data.ignore_index,
            metric_cfg.watch_metric,
        )

    def evaluate_and_report(
        self, config: ImageSegmentationConfig, data: Any, fit: FitState
    ) -> Evaluation:
        """Run the segmentation reporting sequence; see the module contract."""
        from train.common import classification_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        test_best: Optional[Dict[str, float]] = None
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        # DECISION plan-2026-10-30T132432-a865fa08/D-060: data with no test split skips the test
        # metrics and the analysis, which read test images, and the summary says so. Do NOT
        # substitute the selection split for the test split: it selected the best epoch. The one
        # thing kept is the reload check of ``best_model.keras``, run on the selection split with
        # its metrics discarded.
        has_test = data.test_ds is not None
        if checkpoint_cfg.save_best:
            # The in-memory model holds the BEST weights here (EarlyStopping restored them).
            reloaded, load_error, max_diff = classification_run.check_best_checkpoint(
                fit.model,
                fit.run_dir,
                data.test_ds if has_test else data.select_ds,
                evaluate=classification_run.evaluate_dict,
                split_label="Test results" if has_test else "Selection results (reload check only)",
            )
            test_best = reloaded if has_test else None
        # DECISION plan-2026-10-31T180033-2faf890a/D-079
        # The figures and the confusion are drawn from the BEST weights, so they come BEFORE
        # `finalize_last_epoch`, which puts the last epoch's weights in the model. Do NOT move them
        # after it, and do NOT build the confusion from `x_fig` (200 pairs): the report is the
        # pooled confusion of every test pixel, so its mIoU equals `test_metrics_best`'s.
        figures = (
            self._figures(config, data, fit)
            if analysis_cfg.figures and has_test
            else {"files": [], "failed": [], "skipped": True}
        )
        record, analysis = classification_run.report_best_weights(
            fit.model,
            data,
            fit.history,
            fit.run_dir,
            fit.vis_dir,
            fit.experiment_name,
            analysis_cfg.model_analysis and has_test,
            run_analysis=functools.partial(self.run_analysis, config),
            name_failed_analyses=True,
            figures=False,
            analysis_off_hint=(
                "the data has no test split"
                if analysis_cfg.model_analysis
                else "analysis_args.model_analysis=false"
            ),
            eval_batch_size=analysis_cfg.eval_batch_size,
        )
        # The last epoch's weights replace the best ones here.
        # DECISION plan-2026-10-31T180033-2faf890a/D-088
        # The reload check is the EXACT one here (`exact_reload`): equal weights, and outputs within a
        # tolerance relative to the largest score. Do NOT go back to `validate_model_loading`'s absolute
        # 1e-5: AccUNet's logits reach 220 and a correct file differed by 0.3 under TF32 on the GPU
        # (`model_loading_validated` false for a correct file, no note), review-iter-3 concern 4.
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
        return Evaluation(
            test_metrics_best=test_best,
            best_load_error=load_error,
            checkpoint_max_diff=max_diff,
            conformal=None,
            visualizations=figures,
            analysis=analysis,
            test_metrics_final=test_final,
            load_check=load_check,
            load_detail=load_detail or None,
        )


def train_image_segmentation(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[SegData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    checkpoint: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    optim_args: Union[OptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[SegTransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[SegMetricArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train an image segmenter; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_seg_data_config(data)
    given = run_request(
        out=out, model=model, data=data, epochs=epochs, batch_size=batch_size, seed=seed,
        devices=devices, precision=precision, resume_interrupted=resume_interrupted,
        checkpoint=checkpoint, model_args=model_args,
        groups={
            "optim_args": optim_args,
            "transform_args": transform_args,
            "metric_args": metric_args,
            "checkpoint_args": checkpoint_args,
            "logger_args": logger_args,
            "analysis_args": analysis_args,
            "debug_args": debug_args,
        },
    )
    config = validate(ImageSegmentationConfig, given)
    return ImageSegmentationTrainer(config).run()
