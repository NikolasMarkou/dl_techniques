"""The ``beit_tokenizer`` task: a config, a trainer and the keyword-only function.

``train_beit_tokenizer(...)`` is the function a Python caller uses and the function ``python -m train beit_tokenizer key=value ...``
calls, so the two cannot disagree. It validates its keywords into a :class:`BeitTokenizerConfig` and hands that to
:class:`BeitTokenizerTrainer`, a :class:`~train.api.trainer.TaskTrainer` that supplies the hooks of this task and nothing else (a task
overrides hooks, never stages). One family trains it, ``vq_vae_rotation`` (:mod:`train.api.tokenizer_models`): the library
``VQVAERotationTrick``, the BEiT stage-0 discrete tokenizer, which is the class ``beit_mim`` loads as its frozen tokenizer. It owns its
``train_step`` and ``test_step`` (the reconstruction term and the quantizer's own terms), so the task adds none and compiles with no loss
and no metric: the file this task saves is the file ``beit_mim`` reads (``model_args.tokenizer_checkpoint=<run>/best_model.keras``).

Interface contract
------------------
``BeitTokenizerConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``, ``model`` and ``data`` are required.
    The groups are those of classification except ``data`` (:class:`~train.api.tokenizer_args.TokenizerData`: the dataset, the split and
    the patch one code describes), ``transform_args`` (:class:`~train.api.pretrain_args.PretrainTransformArgs`, the pretrain task's: the
    image size and the augmentation) and ``metric_args`` (:class:`~train.api.tokenizer_args.TokenizerMetricArgs`); there is no loss group.
    ``analysis_args.conformal_alpha`` is refused (a conformal set is a set of classes, and a tokenizer has none) and so is a ``precision``
    other than ``float32``.

``BeitTokenizerTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "beit_tokenizer"`` and ``success_metric = "val_reconstruction_loss"``.

    What a run is. An epoch is ``n_train // batch_size`` steps of an INFINITE image stream (``fit_kwargs`` gives ``steps_per_epoch`` and
    ``validation_steps``); the selection and test streams are finite, fixed by the split seed alone and never augmented, so
    ``val_reconstruction_loss`` is a function of the weights alone: equal across passes, epochs and a resume. A resume replaces the train
    stream by one derived from ``(seed, completed steps)`` (``resume_stream``), as the other image-stream tasks do: a fresh stream, NOT the
    batches the unkilled run reads next (``resume_stream_exact`` stays false), and never a selection row.

    The bar (D-038) has TWO parts and both are required. (1) The reconstruction: the lowest ``val_reconstruction_loss`` of the selection
    window is STRICTLY below ``min(baseline, epoch-zero value)`` beyond the repo's float32 tie band
    (:func:`train.common.forecast_run.beats_bar`). The baseline is the constant-image baseline of the selection images
    (:func:`train.common.tokenizer_data.constant_image_baseline`: the lower of the best per-channel constant and the in-sample mean image,
    in the metric's own reduction), measured in ``build_data``; the epoch-zero value is read from the fit sidecar in
    ``evaluate_and_report``, never from an attribute, because a resume never evaluates the untrained model. (2) The codebook, TWO
    floors on the codes the BEST weights give to every position of every selection image: the POOLED perplexity is at least the floor of
    :func:`train.common.pretrain_run.usage_meets_floor` (the one ``beit_mim`` refuses by), AND the PER-CELL perplexity, exp of the mean over
    the grid's cells of the entropy of each cell's codes (:func:`train.common.tokenizer_data.per_cell_usage`), is at least
    :data:`train.common.tokenizer_data.MIN_PER_CELL_PERPLEXITY`; the pooled floor alone is met by a code that is the cell index and nothing
    else wherever the grid has 64 cells or more (D-067). A constant model, a collapsed codebook, a code that describes position only and a
    tokenizer that meets only one part all miss: the run is kept, exits 1 and its summary says which part. Under ``debug_args.smoke`` no bar
    is applied and both parts are still reported. The perplexity is counted once, after the fit, by one eager pass
    (:func:`train.common.tokenizer_data.code_ids_of`): no per-epoch usage reaches the CSV, and the checkpoint is selected on the
    reconstruction alone, so ``best_model.keras`` is the file the verdict describes.

    The epoch-0 check is the untrained model's reconstruction loss over the baseline's (``initial_loss_ratio``); no verdict on the
    initialization is given (``unjudged_note`` says why); a non-finite loss is refused.

    The record (D-044). ``task_record`` writes a ``beit_tokenizer`` ``task.json`` before ``fit``: the trained side, channels and patch, the word,
    mean and std the streams standardized with (``train.common.image_streams.stream_statistics``), the codebook size, and the variant keys
    the run changed. A resume compares its ``DATA_KEYS`` (the geometry, the scaling and the patch) and leaves the file as it is.
    ``predict`` and ``export`` of the run directory answer the int32 code grid (:mod:`train.api.tokenizer_predictor`).

    ``init`` (D-012, D-050). A run starts from an earlier ``beit_tokenizer`` run (the family's one row declares no head and no source of
    another task). ``init_identity`` gives the channels, the scaling and the RESOLVED size, patch and codebook size this run's record will hold,
    so a source of another channel count, size, patch or codebook size is refused before the claim, naming both values; ``init_overlap`` counts
    the source's seeded train rows among this run's selection rows. Epoch zero is then the LOADED weights' ``val_reconstruction_loss``
    (``initial_loss_ratio`` and ``init_scale_warning`` are null, and the sequence's note says why), so the threshold is ``min(baseline,
    loaded)``, still strictly below; ``Evaluation.loaded`` stays ``None``, since the other tasks' higher-is-better comparison with the loaded
    value would judge a loss backwards. A width, a depth or an embedding size the record does not carry is refused by the engine after the
    claim, complete-or-raise.

    Reporting. ``evaluate_and_report`` follows the pretrain order: the best checkpoint reloaded and scored on the TEST split, the usage of
    the best weights, the data-free analysis (``run_analysis``: the weights and spectral analyses; the stock analysis reads class
    probabilities, which an image is not), the last epoch's weights scored, saved and reloaded. ``Evaluation.extras`` carries ``baseline``,
    ``epoch_zero``, ``selection``, ``reconstruction`` (the first part of the bar with its numbers) and ``codebook_usage`` (the second: the
    pooled ``perplexity`` with ``floor`` and ``meets_floor``, and ``per_cell_perplexity`` with ``per_cell_floor`` and ``per_cell_meets_floor``).

``train_beit_tokenizer(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"vq_vae_rotation/small"``; ``python -m train list_models task=beit_tokenizer`` lists it.
    :param data: a mapping (``{"dataset": "cifar10"}``), a :class:`~train.api.tokenizer_args.TokenizerData`, or the path of a YAML file
        holding one.
    :param epochs: / ``batch_size`` / ``devices`` / ``seed`` / ``precision`` / ``resume_interrupted`` / ``model_args`` and the groups: as in
        :func:`~train.api.tasks.image_classification.train_image_classification`; ``precision`` is ``float32`` only; ``init`` is the
        directory of an earlier ``beit_tokenizer`` run of the same image size, patch and codebook size (see ``init`` above).
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see :meth:`~train.api.trainer.TaskTrainer.run`. A family that does not
        train this task, a model key the family refuses, a patch that does not cut the image into whole code cells and a codebook under the
        floor of the bar are refused before the run directory is claimed; a selection split too small to show the floor is refused by name
        after the data is read, and a split of fewer than :data:`train.common.tokenizer_data.MIN_PER_CELL_IMAGES` images (the per-cell number
        reads low on it) is warned about then and not refused (D-073).
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api``: ``python -m train beit_tokenizer help`` imports this module to print the
config's keys and must stay free of TensorFlow. Every ``keras`` and ``train.common`` name is imported inside a method.
"""

import dataclasses
import functools
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .. import recipes, registry
from ..args import AnalysisArgs, AutoContext, CheckpointArgs, DebugArgs, LoggerArgs, OptimArgs
from ..config import (
    AUTO,
    DEFAULT_SEED,
    PATH_FORM,
    RUN_HELP,
    Auto,
    Precision,
    StrictConfig,
    resume_help,
    run_request,
    validate,
)
from ..data_args import DataFacts
from ..predictor import PRETRAIN_CHANNELS, PRETRAIN_NORMALIZE, build_tokenizer_record, overrides_of_variant
from ..pretrain_args import PretrainTransformArgs
from ..run_dir import CHECKPOINTS_DIR
from ..tokenizer_args import (
    FAMILY,
    TASK,
    WATCH_METRIC,
    TokenizerData,
    TokenizerMetricArgs,
    check_geometry,
    load_tokenizer_data_config,
    read_tokenizer_data_file,
    tokenizer_facts,
)
from ..tokenizer_models import precision_problems, recipe_notices_of
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, check_holds_no_classes, config_problems
from .pretrain import selection_block

# ---------------------------------------------------------------------

#: What the library model's own steps report beside ``loss`` (the total with the quantizer's terms): the reconstruction term the bar is
#: about, and the quantizer's. ``evaluate(return_dict=True)`` and the history hold them, with their ``val_`` twins.
METRIC_NAMES = ("reconstruction_loss", "vq_loss")

_RESUME_ORDER = (
    "the training shuffle and augmentation continue on a fresh stream derived from the seed and the steps completed, never "
    "epoch 0's order and never a selection row, and the selection and test streams stay fixed"
)

_STREAM_RULE = (
    "the selection and test streams are keyed on the split seed and never move; the train shuffle and augmentation on the stream_seed; "
    "a resume derives stream_seed from (seed, completed steps), the first 4 bytes of sha256, and the split never follows it"
)

_CONFORMAL = (
    f"a conformal prediction set is a set of classes, which a {TASK} run does not produce; its output is an image and a grid of code ids"
)


@dataclasses.dataclass(frozen=True)
class BeitTokenizerConfig(StrictConfig):
    """Everything a tokenizer run is told; see the module contract."""

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={"help": f"family/variant, e.g. {FAMILY}/small; python -m train list_models task={TASK} lists them"}
    )
    data: TokenizerData = dataclasses.field(
        metadata={
            "help": "the images and the code cell: data.dataset=cifar10 (or imagenette), with data.patch_size, the side of the cell one "
            "code describes; or the path of a YAML file holding them (data=data.yaml), over which dotted keys win "
            "(data=data.yaml data.max_samples=100)",
            PATH_FORM: read_tokenizer_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["epochs"]})
    batch_size: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["batch_size"]})
    seed: int = dataclasses.field(default=DEFAULT_SEED, metadata={"help": RUN_HELP["seed"]})
    devices: Auto[Optional[int]] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["devices"]})
    precision: Precision = dataclasses.field(
        default="float32",
        metadata={"help": "float32, the only value accepted: the quantizer's nearest-code search and its codebook update were never "
                          "measured in float16 or bfloat16"},
    )
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={"help": resume_help("an epoch of the larger datasets takes minutes", order=_RESUME_ORDER)},
    )
    init: Optional[str] = dataclasses.field(default=None, metadata={"help": RUN_HELP["init"]})
    model_args: Dict[str, Any] = dataclasses.field(default_factory=dict, metadata={"help": RUN_HELP["model_args"]})
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: PretrainTransformArgs = dataclasses.field(default_factory=PretrainTransformArgs)
    metric_args: TokenizerMetricArgs = dataclasses.field(default_factory=TokenizerMetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range, a conformal level and a precision other than float32."""
        return [
            *config_problems(self, prefix, no_conformal_for=TASK, because=_CONFORMAL),
            *precision_problems(self.precision, prefix),
        ]


class BeitTokenizerTrainer(TaskTrainer):
    """The hooks of a tokenizer run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    :cvar resume_order: what a resume does to the training stream, for the summary's resume note (the help of ``resume_interrupted``
        carries the same clause).
    """

    task = TASK
    success_metric = WATCH_METRIC
    resume_order = _RESUME_ORDER
    initial_loss_hint = "Lower optim_args.lr, or check the model arguments and the images."

    def __init__(self, config: StrictConfig) -> None:
        super().__init__(config)
        #: The constant image's reconstruction loss on the select stream, set by ``build_data`` (a resume builds the data again, so it is
        #: there before the summary or the verdict).
        self._baseline: Optional[Any] = None
        #: The batches of one selection pass, set by ``build_data``: the width of the bar's tie band follows it.
        self._select_batches: Optional[int] = None
        #: The epoch-zero ``val_reconstruction_loss`` (the untrained model's), read from the fit sidecar by ``evaluate_and_report`` and by
        #: nothing else.
        self._epoch_zero: Optional[float] = None
        #: The codebook usage of the best weights over every selection position, pooled and per cell, set by ``evaluate_and_report``.
        self._usage: Optional[Any] = None
        self._cell_usage: Optional[Any] = None

    # -- data and model -------------------------------------------------

    def describe_data(self, config: BeitTokenizerConfig) -> DataFacts:
        size = config.transform_args.image_size
        return tokenizer_facts(config.data, None if size == AUTO else size)

    def build_data(self, config: BeitTokenizerConfig, context: AutoContext) -> Any:
        import time

        from dl_techniques.utils.logger import logger
        from train.common import pretrain_run, tokenizer_data

        data_cfg: TokenizerData = config.data
        transform_cfg: PretrainTransformArgs = config.transform_args
        data = tokenizer_data.load_tokenizer_batches(
            dataset=data_cfg.dataset, image_size=transform_cfg.image_size, batch_size=config.batch_size, seed=config.seed,
            stream_seed=None, validation_split=data_cfg.validation_split, max_samples=data_cfg.max_samples,
            augment=transform_cfg.augment)
        side = transform_cfg.image_size // data_cfg.patch_size
        positions, floor = data.n_select * side * side, pretrain_run.MIN_TOKENIZER_PERPLEXITY
        if not config.debug_args.smoke and positions < floor:
            # A ValueError: `run_sequence.load_data` turns it into a ConfigError that says `out` is now used.
            raise ValueError(
                f"the selection split holds {data.n_select} images, {positions} positions of a {side} x {side} code grid, and a "
                f"perplexity counted over {positions} positions cannot reach the floor of {floor:g} effective codes: the bar could never be "
                "met (raise data.max_samples or data.validation_split, or set debug_args.smoke=true for a run with no bar)"
            )
        if not config.debug_args.smoke and data.n_select < tokenizer_data.MIN_PER_CELL_IMAGES:
            # A WARNING and not a refusal (unlike the pooled floor above): a run on a handful of images is still a run worth its files (the
            # resume tests train on six). The per-cell perplexity is at most the number of images and reads low well above that
            # (`MIN_PER_CELL_IMAGES` holds the measurement), so a run on fewer can miss its floor with good codes, and says why now.
            logger.warning(
                f"the selection split holds {data.n_select} images, fewer than the {tokenizer_data.MIN_PER_CELL_IMAGES} the per-cell perplexity "
                f"needs to read true: counted over {data.n_select} images it is at most {data.n_select} and reads low well above that, so this "
                f"run can miss the per-cell floor of {tokenizer_data.MIN_PER_CELL_PERPLEXITY:g} effective codes with good codes (raise "
                "data.max_samples or data.validation_split, or set debug_args.smoke=true for a run with no bar)"
            )
        started = time.perf_counter()
        baseline = tokenizer_data.constant_image_baseline(data.select_ds)
        if not (math.isfinite(baseline.reconstruction_loss) and baseline.reconstruction_loss > 0):
            raise ValueError(
                f"the constant image's reconstruction loss on the {data.n_select} selection images is {baseline.reconstruction_loss}: a "
                "bar cut from it would be vacuous or undefined (check data.max_samples and data.validation_split)"
            )
        self._baseline, self._select_batches = baseline, data.val_steps
        logger.info(
            f"Baseline ({baseline.kind}, the better of the per-channel constant {baseline.channel_mean:.6g} and the mean image "
            f"{baseline.mean_image:.6g}) on the {data.n_select} fixed selection images: reconstruction loss "
            f"{baseline.reconstruction_loss:.6g} ({time.perf_counter() - started:.1f} s)"
        )
        return data

    def input_check(self, facts: DataFacts, data: Any) -> None:
        check_holds_no_classes(self, facts, data)

    def build_model(self, config: BeitTokenizerConfig, context: AutoContext, data: Any) -> Any:
        # The adapter returns a model BUILT at the run's size whose code grid it proved, so the sequence's own `build` call is skipped.
        kwargs = {**self._model_kwargs(config, context), "patch_size": config.data.patch_size}
        return recipes.build_model(registry.MODELS[context.family], context.variant, None, data.input_shape, kwargs)

    @staticmethod
    def _model_kwargs(config: BeitTokenizerConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's (the resolver refuses the rest)."""
        return registry.MODELS[context.family].load_resolver()(context, config.optim_args, config.model_args)

    def task_record(self, config: BeitTokenizerConfig, context: AutoContext, data: Any) -> Dict[str, Any]:
        """Return the ``beit_tokenizer`` ``task.json`` record (schema in :mod:`train.api.predictor`).

        The size and channels are ``data.input_shape`` (``build_model`` builds the model at it), the patch is the data's (the encoder's
        total stride), the statistics are the ones the streams standardize with and the codebook is the resolved ``num_embeddings``. Nothing
        of the model is needed, so a resume asks for it before ``build_model``.
        """
        from train.common.image_streams import stream_statistics

        height, width, channels = data.input_shape
        mean, std = stream_statistics(config.data.dataset)
        kwargs = self._model_kwargs(config, context)
        table = registry.MODELS[context.family].load_variants()
        return build_tokenizer_record(
            model=config.model, family=context.family, variant=context.variant, input_size=(height, width), channels=channels,
            mean=mean, std=std, patch_size=config.data.patch_size, num_embeddings=kwargs["num_embeddings"],
            variant_overrides=overrides_of_variant(table[context.variant], kwargs),
        )

    def init_identity(self, config: BeitTokenizerConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the channels, scaling, size, patch and codebook size this run's record will hold, for an ``init`` source."""
        # DECISION plan-2026-11-25T124419-fa94f59c/D-050
        # The RESOLVED size, patch and codebook, as `task_record` writes them (the size a list, the record's JSON form), never the dataset's
        # table or the recipe's: a typed transform_args.image_size, data.patch_size or model_args.num_embeddings is what the record will hold,
        # and a source of another one would pass this check and be refused only after the claim, by its weights (for the size, never: every
        # weight copies). Do NOT read GEOMETRY[dataset] or CODEBOOK[dataset] here.
        size = config.transform_args.image_size
        return {
            "channels": PRETRAIN_CHANNELS, "normalize": PRETRAIN_NORMALIZE, "input_size": [size, size], "patch_size": config.data.patch_size,
            "num_embeddings": self._model_kwargs(config, context).get("num_embeddings"),
        }

    def init_overlap(self, source: Any, config: BeitTokenizerConfig, data: Any) -> Dict[str, Any]:
        """The source's seeded train cut against this run's selection rows (:func:`train.api.init_from.seeded_rows_split`): the streams cut
        a dataset's rows as classification does, and the dataset's train size backs a capped run."""
        from dl_techniques.datasets.vision.classification import get_spec

        from .. import init_from

        return init_from.seeded_rows_split(source, config, data, get_spec(config.data.dataset).train_size)

    def used_arguments(self, config: BeitTokenizerConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def check_model_args(self, config: BeitTokenizerConfig, context: AutoContext) -> None:
        from train.common import pretrain_run

        # The family's own refusals first (every refused key at once), then the geometry and the codebook against the floor of the bar.
        # The floor is read from its ONE definition here, at the call, because the module that holds it imports TensorFlow (D-035).
        kwargs = self._model_kwargs(config, context)
        check_geometry(
            config.data.patch_size, config.transform_args.image_size, kwargs.get("num_embeddings"),
            floor=pretrain_run.MIN_TOKENIZER_PERPLEXITY, smoke=config.debug_args.smoke)

    # -- compile --------------------------------------------------------

    def metric_names(self, config: BeitTokenizerConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates variables and the first eager op would
        # initialise TensorFlow's devices before setup_gpu.
        return list(METRIC_NAMES)

    def build_loss_and_metrics(self, config: BeitTokenizerConfig, data: Any) -> Tuple[Any, List[Any]]:
        # DECISION plan-2026-11-25T124419-fa94f59c/D-037
        # NO loss and NO metric: the library model's own `train_step` and `test_step` compute the reconstruction term and the quantizer's
        # (probe P1 compiled it three ways and the compiled loss is never read). Do NOT pass an inert `MeanSquaredError` (a loss in the
        # compile config of the saved file that the model does not use, which a reader would take for the objective) and do NOT add a
        # step of the task's own: the file must be the `VQVAERotationTrick` that `beit_mim` loads.
        return None, []

    def initial_loss_check(
        self, config: BeitTokenizerConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the reconstruction loss over the baseline's, and no verdict is given.

        :raises RuntimeError: for a non-finite loss (the model diverges before training).
        """
        from dl_techniques.utils.logger import logger
        from train.common import classification_run, run_summary

        if self._baseline is None:
            raise RuntimeError(
                "initial_loss_check compares with the constant image's loss, which build_data works out; it was asked for before the "
                "data was built"
            )
        metrics = classification_run.evaluate_dict(model, data.select_ds)
        # The total `loss` carries the quantizer's terms and the baseline has none, so the ratio is the reconstruction term over it.
        ratio = run_summary.initial_loss_ratio(
            {"loss": metrics["reconstruction_loss"]}, self._baseline.reconstruction_loss, self.initial_loss_hint)
        logger.info(
            f"Untrained evaluate BEFORE fit: reconstruction loss {metrics['reconstruction_loss']:.6g} (total {metrics['loss']:.6g}) on "
            f"{data.n_select} selection images, {ratio:.3g}x the constant image's {self._baseline.reconstruction_loss:.6g}"
        )
        # No verdict (`None`): the untrained model's loss against a constant image is not a scale test.
        return metrics, ratio, None

    def unjudged_note(self, config: BeitTokenizerConfig) -> str:
        return (
            "`init_scale_warning` is null (not judged): `initial_loss_ratio` is the untrained model's reconstruction loss over the "
            "constant image's on the same fixed selection images, kept as the baseline. A random initialization sits above the "
            "constant image by design (an untrained decoder's output is noise), so no threshold on that ratio separates a well-scaled "
            "model from a mis-scaled one; a non-finite initial loss is refused"
        )

    # -- the two seams of the shared sequence ---------------------------

    def fit_kwargs(self, config: BeitTokenizerConfig, data: Any) -> Dict[str, Any]:
        # The train stream is INFINITE (the shuffle and augmentation never end), so an epoch is `steps_per_epoch` batches by this keyword
        # (the sequence refuses an infinite stream without it); `validation_steps` is the selection pass, stated so that Keras never has
        # to guess a length.
        return {"steps_per_epoch": data.steps_per_epoch, "validation_steps": data.val_steps}

    def resume_stream(self, config: BeitTokenizerConfig, data: Any, completed_steps: int) -> Optional[Any]:
        # DECISION plan-2026-11-25T124419-fa94f59c/D-040
        # A resumed `fit` starts a fresh iterator, so the stream the fresh process built would replay from its first element (epoch 0's
        # order and augmentation). `with_stream` rebuilds the SAME request on a stream seed derived from `(seed, completed_steps)`: the
        # split, the counts and the select and test streams do not move, the train stream does. That is another stream, not the batches
        # the unkilled run reads next, so `resume_stream_exact` stays false and `resume_order` says so. Do NOT return `data` ("nothing to
        # rebuild": that is the replay), do NOT reseed the split, and do NOT set `resume_stream_exact`: no test or note could keep it true.
        return data.with_stream(completed_steps)

    # -- the verdict ----------------------------------------------------

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the best (the lowest) ``val_reconstruction_loss`` of the selection window."""
        start = self.selection_start_epoch(self.config)
        return float(min(hist[self.success_metric][start:]))

    def success_threshold(self, config: BeitTokenizerConfig) -> float:
        """Return ``min(baseline, epoch-zero val_reconstruction_loss)``; ``evaluate_and_report`` has read the second term."""
        if self._baseline is None:
            raise RuntimeError(
                "success_threshold is cut from the baseline (the constant image's reconstruction loss), which build_data works out; it "
                "was asked for before the data was built"
            )
        if self._epoch_zero is None:
            raise RuntimeError(
                "success_threshold needs the epoch-zero val_reconstruction_loss, which evaluate_and_report reads from the fit sidecar; "
                "it was asked for before the reporting"
            )
        return float(min(self._baseline.reconstruction_loss, self._epoch_zero))

    def _beats_bar(self, reached: float, threshold: float) -> bool:
        """The first part of the bar: ``reached`` strictly below ``threshold`` beyond the tie band of the selection batch count."""
        from train.common import forecast_run

        return forecast_run.beats_bar(reached, threshold, self._select_batches or 1)

    def bar_is_met(self, config: BeitTokenizerConfig, reached: float, threshold: float) -> bool:
        from train.common import pretrain_run, tokenizer_data

        # DECISION plan-2026-11-25T124419-fa94f59c/D-038
        # BOTH parts, each by the repo's one definition: the reconstruction STRICTLY below the threshold in float32 and beyond the tie band
        # (`forecast_run.beats_bar`: a constant model scores one or two float32 steps either side of its baseline, MEASURED, so a bare `<`
        # or a `<=` calls it a success), AND the codebook of the best weights at the floor (`pretrain_run.usage_meets_floor`, the same
        # definition `beit_mim` refuses by). Do NOT drop one part: a constant model meets the second and a collapsed codebook the first.
        # Do NOT compare the Python floats, do NOT copy the floor, and do NOT read the usage of the last epoch's weights. The codebook part is
        # itself TWO floors (D-067): the pooled one is met by a code that is the cell index alone at 64 cells or more, so the per-cell one
        # (`tokenizer_data.per_cell_meets_floor`) judges what one cell carries. Do NOT drop it, and do NOT fold it into the pooled floor.
        if self._usage is None or self._cell_usage is None:
            raise RuntimeError(
                "bar_is_met needs the codebook usage of the best weights, pooled and per cell, which evaluate_and_report works out; it was "
                "asked for before the reporting"
            )
        return bool(
            self._beats_bar(reached, threshold) and pretrain_run.usage_meets_floor(self._usage)
            and tokenizer_data.per_cell_meets_floor(self._cell_usage)
        )

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: BeitTokenizerConfig) -> Tuple[str, ...]:
        return ("reconstruction_loss", "val_reconstruction_loss", "vq_loss")

    def summary_fields(self, config: BeitTokenizerConfig, data: Any) -> Dict[str, Any]:
        side = data.input_shape[0] // config.data.patch_size
        return {
            "patch_size": config.data.patch_size,
            "code_grid": [side, side],
            "stream": {"seed": data.seed, "stream_seed": data.stream_seed, "rule": _STREAM_RULE},
        }

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        return recipe_notices_of(facts.dataset)

    def task_notes(self, config: BeitTokenizerConfig, baseline: Dict[str, float], ratio: float, data: Any) -> List[str]:
        """Return the tokenizer lines of ``results_summary.json``'s ``notes``."""
        from train.common import classification_run, pretrain_run, tokenizer_data

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        constant = self._baseline
        zero = "the untrained model's" if config.init is None else "the loaded weights' (under `init`)"
        # DECISION plan-2026-11-19T081214-5c6cc105/D-012
        # Under init the ratio is None (epoch zero is the loaded weights': no ratio is computed), so its line is left out. Do NOT format it
        # unguarded: `{None:.3g}` raises after the run has trained.
        notes = [
            *([f"initial reconstruction loss {baseline['reconstruction_loss']:.6g} is {ratio:.3g}x the constant image's "
               f"({constant.reconstruction_loss:.6g}, {constant.kind}) on the {data.n_select} fixed selection images: a plain evaluate in "
               "inference mode; see the `init_scale_warning` note"] if ratio is not None else []),
            *classification_run.RUN_NOTES[:2],
            "`val_reconstruction_loss` is the mean over batches of the squared error between a standardized image and its reconstruction "
            "through the codebook, without the quantizer's terms (the model's own `test_step`); the selection and test streams are fixed, "
            "so it is a function of the weights alone (equal across passes, epochs and a resume)",
            "the bar has two parts, and both parts are required: (1) `success_value`, the lowest `val_reconstruction_loss` of the selection "
            "window, is strictly below `success_threshold`, min(`baseline.reconstruction_loss`, `epoch_zero.reconstruction_loss`), in "
            f"float32 and beyond `train.common.forecast_run.tie_band` float32 steps (`reconstruction.beats_bar`); the baseline is the "
            f"better of the best per-channel constant and the in-sample mean image of the selection images and {zero} value is read from "
            "the fit sidecar; (2) the codebook usage of the best weights over every position of every selection image has a perplexity "
            f"of at least {pretrain_run.MIN_TOKENIZER_PERPLEXITY:g} effective codes pooled over the positions (`codebook_usage.meets_floor`) "
            f"AND of at least {tokenizer_data.MIN_PER_CELL_PERPLEXITY:g} effective codes per cell, exp of the mean over the code grid's cells of "
            "the entropy of each cell's codes (`codebook_usage.per_cell_meets_floor`: a code that is the cell index alone has the pooled "
            "perplexity of the number of cells and a per-cell one of 1); a constant model, a collapsed codebook and a code that describes "
            "position only each miss one part",
            "`codebook_usage` counts the codes of the best weights (best_model.keras, the file `beit_mim` takes), once, after the fit; "
            "`test_metrics_best` is the reloaded best_model.keras and `test_metrics_final` the last epoch's weights, on the test split, "
            "which never influenced selection",
        ]
        notes.append(
            "the model analysis is data-free (weights and spectral only; `analyzer.analyzers` lists them): the stock analyses read class "
            "probabilities, which an image is not"
            if analysis_cfg.model_analysis else "analysis_args.model_analysis is false: `analyzer.status` is 'skipped'"
        )
        if not checkpoint_cfg.save_best:
            notes.append("checkpoint_args.save_best is false: no best_model.keras was written, so the reload check did not run")
        if not checkpoint_cfg.save_last:
            notes.append(
                "checkpoint_args.save_last is false: no final_model.keras was written and `model_loading_validated` is null"
            )
        return notes

    # -- the reporting --------------------------------------------------

    def run_analysis(
        self,
        config: BeitTokenizerConfig,
        model: Any,
        test_data: Tuple[Any, Any],
        history: Any,
        experiment_name: str,
        run_dir: str,
    ) -> Any:
        """Run the data-free analysis (weights and spectral only); see the module contract."""
        from train.common import run_summary

        # The stock analysis reads the output as class probabilities (calibration, information flow, training dynamics) and the target as
        # labels; the output of a reconstruction is neither (measured, probe P1: the data-free analysis completes, `analyzer.status` ok).
        # Do NOT fall back to the default hook.
        return run_summary.data_free_model_analysis(model, test_data, history, experiment_name, run_dir, config.seed)

    def evaluate_and_report(self, config: BeitTokenizerConfig, data: Any, fit: FitState) -> Evaluation:
        """Run the tokenizer reporting sequence; see the module contract."""
        from types import SimpleNamespace

        from train.common import classification_run, forecast_run, pretrain_data, pretrain_run, run_summary, tokenizer_data
        from train.common.resume_state import FIT_STATE_NAME

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        metric_cfg: TokenizerMetricArgs = config.metric_args
        # The epoch-zero term of the bar, from the sidecar: see `epoch_zero_val_loss` (a resume reads the same number).
        self._epoch_zero = pretrain_run.epoch_zero_val_loss(
            fit.run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME, key="reconstruction_loss")
        baseline = self._baseline
        if baseline is None:
            raise RuntimeError("evaluate_and_report needs the baseline build_data works out; the data was not built")
        test_best: Optional[Dict[str, float]] = None
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        if checkpoint_cfg.save_best:
            # The in-memory model holds the BEST weights here (EarlyStopping restored them): the reloaded file must score them, to the
            # tolerance (the selection and test streams are fixed, so the gap is a number to report and not `0.0` to assert on a GPU).
            test_best, load_error, max_diff = classification_run.check_best_checkpoint(
                fit.model, fit.run_dir, data.test_ds, evaluate=classification_run.evaluate_dict, split_label="Test results")
        # DECISION plan-2026-11-25T124419-fa94f59c/D-039
        # The usage is counted HERE: on the BEST weights (the file `beit_mim` takes and the one the verdict describes), over EVERY position of
        # EVERY selection image (the train rows would be a different, augmented distribution), and BEFORE `report_data_free_run` below swaps
        # the last epoch's weights into the model. Do NOT move the pass after that call, do NOT count the train stream, and do NOT cut it to
        # the first batch.
        codes = tokenizer_data.code_ids_of(fit.model, data.select_ds)
        self._usage = pretrain_data.tokenizer_usage(codes)
        self._cell_usage = tokenizer_data.per_cell_usage(codes)
        image, label = next(iter(data.select_ds))
        analysis, test_final, load_check, load_detail = run_summary.report_data_free_run(
            fit.model,
            SimpleNamespace(x_fig=image.numpy(), y_fig=label.numpy()),
            fit.history,
            fit.run_dir,
            fit.vis_dir,
            fit.experiment_name,
            run_analysis=functools.partial(self.run_analysis, config),
            model_analysis=analysis_cfg.model_analysis,
            eval_batch_size=analysis_cfg.eval_batch_size,
            last_weights=fit.last_weights,
            data=data,
            epochs_run=fit.epochs_run,
            save_last=checkpoint_cfg.save_last,
            load_sample=image.numpy()[: classification_run.LOAD_CHECK_SAMPLES],
        )
        visualizations: Dict[str, Any] = {"files": [], "failed": [], "skipped": not analysis_cfg.figures}
        if analysis_cfg.figures and fit.vis_dir.is_dir():
            visualizations["files"] = sorted(path.name for path in fit.vis_dir.iterdir() if path.is_file())
        hist = {key: [float(v) for v in values] for key, values in fit.history.history.items()}
        start = self.selection_start_epoch(config)
        best = run_summary.best_epoch(hist, metric_cfg.watch_metric, metric_cfg.watch_mode, first_epoch=start)
        reached, threshold = self.success_value(hist), self.success_threshold(config)
        extras: Dict[str, Any] = {
            "baseline": baseline.as_dict(),
            "epoch_zero": {"reconstruction_loss": self._epoch_zero, "binds": bool(self._epoch_zero < baseline.reconstruction_loss)},
            "selection": selection_block(start, fit.epochs_run, best),
            "reconstruction": {
                "success_value": reached, "success_threshold": threshold,
                "tie_band_float32_steps": forecast_run.tie_band(self._select_batches or 1),
                "beats_bar": self._beats_bar(reached, threshold),
            },
            "codebook_usage": {
                "weights": "best_model.keras: the best epoch's weights",
                "positions": self._usage.n_codes, "images": data.n_select, "num_embeddings": int(fit.model.num_embeddings),
                "distinct": self._usage.distinct, "perplexity": self._usage.perplexity, "top1_share": self._usage.top1_share,
                "entropy": self._usage.entropy,
                "floor": pretrain_run.MIN_TOKENIZER_PERPLEXITY, "meets_floor": pretrain_run.usage_meets_floor(self._usage),
                "per_cell_perplexity": self._cell_usage.perplexity, "per_cell_entropy": self._cell_usage.entropy,
                "per_cell_floor": tokenizer_data.MIN_PER_CELL_PERPLEXITY,
                "per_cell_meets_floor": tokenizer_data.per_cell_meets_floor(self._cell_usage),
            },
        }
        # DECISION plan-2026-11-19T081214-5c6cc105/D-012
        # No `loaded=`: `Evaluation.loaded` is higher-is-better (`init_from.beats_loaded`) and would judge this loss backwards; under init the
        # sidecar's epoch zero IS the loaded weights' value, so the threshold is already min(baseline, loaded), strictly below.
        return Evaluation(
            test_metrics_best=test_best,
            best_load_error=load_error,
            checkpoint_max_diff=max_diff,
            conformal=None,
            visualizations=visualizations,
            analysis=analysis,
            test_metrics_final=test_final,
            load_check=load_check,
            load_detail=load_detail,
            extras=extras,
        )


def train_beit_tokenizer(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[TokenizerData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    init: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    optim_args: Union[OptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[PretrainTransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[TokenizerMetricArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train the BEiT stage-0 tokenizer, a ``VQVAERotationTrick``; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_tokenizer_data_config(data)
    given = run_request(
        out=out, model=model, data=data, epochs=epochs, batch_size=batch_size, seed=seed,
        devices=devices, precision=precision, resume_interrupted=resume_interrupted,
        init=init, model_args=model_args,
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
    config = validate(BeitTokenizerConfig, given)
    return BeitTokenizerTrainer(config).run()
