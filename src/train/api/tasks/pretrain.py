"""The pretrain task: a config, a trainer and the keyword-only function.

``train_pretrain(...)`` is the function a Python caller uses and the function ``python -m train pretrain key=value ...``
calls, so the two cannot disagree. It validates its keywords into a :class:`PretrainConfig` and hands that to
:class:`PretrainTrainer`, a :class:`~train.api.trainer.TaskTrainer` that supplies the hooks of this task and nothing else
(a task overrides hooks, never stages). Two families train it, and the FAMILY chooses the method (D-006):
``energy_transformer`` completes the pixels of masked patches (``et_masked``), ``beit`` predicts the code ids a frozen
stage-0 tokenizer gives the masked patches (``beit_mim``). Both take their images from the shared image streams
(``cifar10`` and ``imagenette``); a labelled image is only an image here.

Interface contract
------------------
``PretrainConfig``
    Frozen :class:`~train.api.config.StrictConfig`; one field per keyword of the function. ``out``, ``model`` and ``data``
    are required. The groups are those of classification except ``data`` (:class:`~train.api.pretrain_args.PretrainData`:
    the dataset, the split, the patch and the mask keys), ``transform_args``
    (:class:`~train.api.pretrain_args.PretrainTransformArgs`) and ``metric_args``
    (:class:`~train.api.pretrain_args.PretrainMetricArgs`); there is no loss group (the loss follows the method).
    ``analysis_args.conformal_alpha`` is refused (a conformal set is a set of classes, and a pretrain run has none).

``PretrainTrainer(config)``
    :class:`~train.api.trainer.TaskTrainer` with ``task = "pretrain"`` and ``success_metric = "val_loss"``.

    What a run is. An epoch is ``n_train // batch_size`` steps of an INFINITE masked stream (D-011), so ``fit_kwargs``
    gives ``steps_per_epoch`` and ``validation_steps`` (the shared sequence refuses an infinite stream without them). The
    mask reaches the loss only as ``sample_weight`` (no custom ``train_step``). The select and test masks are keyed on the
    split seed and the row position, so ``val_loss`` is a function of the weights alone: equal across passes, epochs and a
    resume. The train mask is keyed on the stream seed, which a resume replaces (``resume_stream``) by one derived from
    ``(seed, completed steps)``: a resumed fit starts a fresh iterator, and a stream keyed on a counter would otherwise
    replay epoch 0's masks; the split never follows it, so no selection row reaches the train stream.

    The bar (D-009). ``val_loss`` is a loss, so ``bar_is_met`` is STRICTLY below, in float32 (the base class's ``>=`` is the
    wrong direction); ``success_value`` is the minimum of ``val_loss`` over the selection window. The threshold is
    ``min(baseline, epoch-zero val_loss)``: the baseline is the ``val_loss`` of a CONSTANT Keras model (ET: zero patches;
    BEiT: the in-sample unigram log-probabilities of the masked select positions) evaluated under the same compile on the
    same fixed select stream, measured in ``build_data`` and kept on the trainer (a resume rebuilds the data, and the
    number is deterministic); the epoch-zero value is read from the fit sidecar in ``evaluate_and_report``
    (:func:`train.common.pretrain_run.epoch_zero_val_loss`), never from an attribute, because a resume never evaluates the
    untrained model. Under ``debug_args.smoke`` no bar is applied.

    The epoch-0 check is the untrained model's loss over the baseline's (``initial_loss_ratio``); no verdict on the
    initialization is given (``unjudged_note`` says why); a non-finite loss is refused.

    Reporting. ``evaluate_and_report`` follows the denoiser's order: the best checkpoint reloaded and scored on the TEST
    split, the data-free analysis (``run_analysis``: the weights and spectral analyses; the stock analysis reads class
    probabilities, which a masked-patch output is not), the last epoch's weights scored, saved and reloaded through
    ``finalize_last_epoch(sample=[image, mask], exact_reload=True)``. ``Evaluation.extras`` carries ``baseline``,
    ``epoch_zero``, ``selection`` and ``tokenizer``. The energy trace of ET (``figure_callbacks``) is written to
    ``visualizations/energy_trace.csv`` with one header across a resume.

    The task writes no ``task.json`` (D-013): ``task_record`` is ``None``, so ``predict`` and ``export`` of a pretrain run
    directory refuse with the existing ``task.json`` sentence.

    The BEiT tokenizer (D-007). ``model_args.tokenizer_checkpoint`` is required for a ``beit`` run and is read by the TASK,
    never by the model (the resolver strips it). Before the claim ``check_model_args`` refuses a missing key, a path that does
    not exist and one that does not end in ``.keras``; at build time ``train.beit.common.load_frozen_tokenizer`` proves the
    code grid by a forward pass at the run's image size and refuses a file that is not a ``VQVAERotationTrick`` (these are
    found after the claim, and the message says so). ``vocab_size`` is the tokenizer's codebook size. The usage of the codes
    over the masked positions of the selection split (distinct codes, perplexity, top-1 share) is reported in the summary's
    ``tokenizer`` block, and a perplexity below ``train.common.pretrain_run.MIN_TOKENIZER_PERPLEXITY`` (64) is refused.

``train_pretrain(*, out, model, data, ...)``
    :param out: run directory (a ``str`` or a ``Path``), or ``"auto"``.
    :param model: ``"family/variant"``, e.g. ``"energy_transformer/et_tiny"`` or ``"beit/beit_tiny"``;
        ``python -m train list_models task=pretrain`` lists them.
    :param data: a mapping (``{"dataset": "cifar10"}``), a :class:`~train.api.pretrain_args.PretrainData`, or the path of
        a YAML file holding one.
    :param epochs: / ``batch_size`` / ``devices`` / ``seed`` / ``precision`` / ``resume_interrupted`` /
        ``checkpoint`` / ``model_args`` and the groups: as in
        :func:`~train.api.tasks.image_classification.train_image_classification`.
    :return: the :class:`~train.api.trainer.TrainResult`.
    :raises ConfigValidationError: for every bad key, listed together.
    :raises ConfigError: for a request that cannot be honoured; see :meth:`~train.api.trainer.TaskTrainer.run`. A family
        that does not train this task, a model key a family refuses, a mask key of the other family, a patch that does not
        divide the image, a BEiT run with no tokenizer and an ET run with no global gradient clip are refused before the run
        directory is claimed.
    :raises RuntimeError: for a diverged or non-finite run.

Import rule
-----------
Module scope imports only the standard library and ``train.api``: ``python -m train pretrain help`` imports this module to
print the config's keys and must stay free of TensorFlow. Every ``keras`` and ``train.common`` name (and the two trainer
packages ``train.energy_transformer.common`` and ``train.beit.common``) is imported inside a method.
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
    resume_help,
    run_request,
    validate,
)
from ..data_args import DataFacts
from ..errors import ConfigError
from ..pretrain_args import (
    MASK_KEYS,
    TASK,
    PretrainData,
    PretrainMetricArgs,
    PretrainTransformArgs,
    load_pretrain_data_config,
    pretrain_facts,
    read_pretrain_data_file,
)
from ..pretrain_models import BEIT, METHOD_OF, TASK_KEYS, recipe_notices_of
from ..run_dir import CHECKPOINTS_DIR
from ..trainer import Evaluation, FitState, TaskTrainer, TrainResult, check_holds_no_classes, config_problems

# ---------------------------------------------------------------------

#: The ``model_args`` key of ``beit`` that names the frozen stage-0 tokenizer (the task reads it, the model never sees it).
TOKENIZER_KEY = TASK_KEYS[BEIT][0]

_RESUME_ORDER = (
    "the training masks and the shuffle continue on a fresh stream derived from the seed and the steps completed, never "
    "epoch 0's order and never a selection row, and the selection and test masks stay fixed"
)


@dataclasses.dataclass(frozen=True)
class PretrainConfig(StrictConfig):
    """Everything a pretrain run is told; see the module contract."""

    out: str = dataclasses.field(metadata={"help": RUN_HELP["out"]})
    model: str = dataclasses.field(
        metadata={
            "help": "family/variant, e.g. energy_transformer/et_tiny or beit/beit_tiny; python -m train list_models "
            "task=pretrain lists them"
        }
    )
    data: PretrainData = dataclasses.field(
        metadata={
            "help": "the images and the mask: data.dataset=cifar10 (or imagenette), with data.patch_size and the mask "
            "keys of the family (data.mask_ratio and data.mask_token_frac for energy_transformer, "
            "data.num_mask_patches and data.min_mask_patches_per_block for beit); or the path of a YAML file holding "
            "them (data=data.yaml), over which dotted keys win (data=data.yaml data.max_samples=100)",
            PATH_FORM: read_pretrain_data_file,
        }
    )
    epochs: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["epochs"]})
    batch_size: Auto[int] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["batch_size"]})
    seed: int = dataclasses.field(default=DEFAULT_SEED, metadata={"help": RUN_HELP["seed"]})
    devices: Auto[Optional[int]] = dataclasses.field(default=AUTO, metadata={"help": RUN_HELP["devices"]})
    precision: Precision = dataclasses.field(default="float32", metadata={"help": RUN_HELP["precision"]})
    resume_interrupted: bool = dataclasses.field(
        default=False,
        metadata={"help": resume_help("an epoch of the larger datasets takes minutes", order=_RESUME_ORDER)},
    )
    checkpoint: Optional[str] = dataclasses.field(default=None, metadata={"help": RUN_HELP["checkpoint"]})
    model_args: Dict[str, Any] = dataclasses.field(
        default_factory=dict,
        metadata={
            "help": RUN_HELP["model_args"] + "; beit also takes model_args.tokenizer_checkpoint, the path of the "
            "stage-0 tokenizer .keras (python -m train.beit.train_tokenizer), which is required for beit and is "
            "not a model keyword"
        },
    )
    optim_args: OptimArgs = dataclasses.field(default_factory=OptimArgs)
    transform_args: PretrainTransformArgs = dataclasses.field(default_factory=PretrainTransformArgs)
    metric_args: PretrainMetricArgs = dataclasses.field(default_factory=PretrainMetricArgs)
    checkpoint_args: CheckpointArgs = dataclasses.field(default_factory=CheckpointArgs)
    logger_args: LoggerArgs = dataclasses.field(default_factory=LoggerArgs)
    analysis_args: AnalysisArgs = dataclasses.field(default_factory=AnalysisArgs)
    debug_args: DebugArgs = dataclasses.field(default_factory=DebugArgs)

    def problems(self, prefix: str) -> List[str]:
        """Name the top-level fields out of range, and a conformal level (a pretrain run has no classes)."""
        return config_problems(self, prefix, no_conformal_for="pretrain")


class PretrainTrainer(TaskTrainer):
    """The hooks of a pretrain run; see the module contract.

    :cvar initial_loss_hint: the remedy sentence of the non-finite initial loss refusal.
    :cvar resume_order: what a resume does to the training data order, for the summary's resume note (the help of
        ``resume_interrupted`` carries the same clause).
    """

    task = TASK
    success_metric = "val_loss"
    resume_order = _RESUME_ORDER
    initial_loss_hint = "Lower optim_args.lr, or check the model arguments and the images."

    def __init__(self, config: StrictConfig) -> None:
        super().__init__(config)
        #: The constant predictor's ``val_loss`` on the select stream, set by ``build_data`` (a resume builds the data
        #: again, so it is there before the summary or the verdict).
        self._baseline: Optional[Any] = None
        #: The untrained model's ``val_loss``, read from the fit sidecar by ``evaluate_and_report`` and by nothing else.
        self._epoch_zero: Optional[float] = None
        #: The BEiT codebook size, set by ``build_data`` from the tokenizer (``None`` for energy_transformer).
        self._vocab_size: Optional[int] = None
        #: The BEiT tokenizer's usage over the select split's masked positions, set by ``build_data`` (``None`` for ET).
        self._usage: Optional[Any] = None

    # -- data and model -------------------------------------------------

    def describe_data(self, config: PretrainConfig) -> DataFacts:
        size = config.transform_args.image_size
        return pretrain_facts(config.data, None if size == AUTO else size)

    def build_data(self, config: PretrainConfig, context: AutoContext) -> Any:
        import math
        import time

        from dl_techniques.utils.logger import logger
        from train.common import pretrain_data, pretrain_run

        data_cfg: PretrainData = config.data
        transform_cfg: PretrainTransformArgs = config.transform_args
        if data_cfg.format != "builtin":
            # `format` is a Literal of one value today; a second source (files) adds its loader here, not a silent fall
            # through to the built-in streams below.
            raise ValueError(f"data.format {data_cfg.format!r} has no loader in the pretrain task: only 'builtin' does")
        method = METHOD_OF[context.family]
        tokenizer_fn = None
        self._vocab_size = self._usage = None
        if method == pretrain_data.BEIT_MIM:
            tokenizer_fn, self._vocab_size = self._frozen_tokenizer(config)
        data = pretrain_data.load_pretrain_batches(
            method=method,
            dataset=data_cfg.dataset,
            image_size=transform_cfg.image_size,
            patch_size=data_cfg.patch_size,
            batch_size=config.batch_size,
            seed=config.seed,
            stream_seed=None,
            validation_split=data_cfg.validation_split,
            max_samples=data_cfg.max_samples,
            augment=transform_cfg.augment,
            mask_ratio=data_cfg.mask_ratio,
            mask_token_frac=data_cfg.mask_token_frac,
            num_mask_patches=data_cfg.num_mask_patches,
            min_mask_patches_per_block=data_cfg.min_mask_patches_per_block,
            tokenizer_fn=tokenizer_fn,
        )
        code_ids = None
        if method == pretrain_data.BEIT_MIM:
            # One pass of the tokenizer over the selection split serves the usage report, its floor and the baseline.
            code_ids = pretrain_data.masked_code_ids(data.select_ds)
            self._usage = pretrain_run.checked_tokenizer_usage(code_ids, path=str(config.model_args[TOKENIZER_KEY]))
            logger.info(
                f"Tokenizer usage over the {self._usage.n_codes} masked selection positions: {self._usage.distinct} distinct "
                f"codes of {self._vocab_size}, perplexity {self._usage.perplexity:.2f}, top-1 share "
                f"{self._usage.top1_share:.2%}"
            )
        started = time.perf_counter()
        baseline = pretrain_run.constant_baseline(
            method, data.select_ds, vocab_size=self._vocab_size, code_ids=code_ids)
        if not (math.isfinite(baseline.val_loss) and baseline.val_loss > 0):
            # A ValueError: `run_sequence.load_data` turns it into a ConfigError that says `out` is now used.
            raise ValueError(
                f"the constant predictor's val_loss on the {data.n_select} selection images is {baseline.val_loss}: a "
                "bar cut from it would be vacuous or undefined (check data.max_samples and data.validation_split: the "
                "selection split may be empty or masked to nothing)"
            )
        self._baseline = baseline
        logger.info(
            f"Baseline ({baseline.kind}, the constant predictor under the run's loss) on the {data.n_select} fixed "
            f"selection images: val_loss {baseline.val_loss:.6g} ({time.perf_counter() - started:.1f} s)"
        )
        return data

    def _frozen_tokenizer(self, config: PretrainConfig) -> Tuple[Any, int]:
        """Return ``(tokenizer_fn, vocab_size)`` of the BEiT run's frozen tokenizer.

        The legacy loader (``train.beit.common.load_frozen_tokenizer``) proves the code grid by a forward pass at the run's
        image size, so a tokenizer built at another size (or a down-sample that lands on another grid) is refused, naming
        the file, and a file that is not a ``VQVAERotationTrick`` is a named ``TypeError``; ``load_data`` turns both into a
        ``ConfigError`` that says ``out`` is claimed. The path's existence and suffix are checked before the claim
        (:meth:`check_model_args`).
        """
        from train.beit.common import load_frozen_tokenizer

        path = config.model_args.get(TOKENIZER_KEY)
        if path is None:
            raise ConfigError(self._tokenizer_required_message())
        side = config.transform_args.image_size // config.data.patch_size
        shape = (config.transform_args.image_size, config.transform_args.image_size, 3)
        # DECISION plan-2026-11-12T015747-46e5318f/D-007
        # The grid is proved by a REAL forward pass at the run's image size, inside the loader, and the expected grid is the
        # one the PATCHES give, never the tokenizer's own `downsample_factor`: a tokenizer whose grid does not line up with
        # the patch grid trains on targets read from the wrong positions and still reports a finite, plausible loss. Do NOT
        # derive the expected grid from the tokenizer, and do NOT load it with `keras.models.load_model` directly.
        tokenizer = load_frozen_tokenizer(str(path), (side, side), shape)
        return tokenizer, int(tokenizer.model.num_embeddings)

    @staticmethod
    def _tokenizer_required_message() -> str:
        return (
            f"Invalid value for 'model_args.{TOKENIZER_KEY}': a beit run predicts the code ids of a frozen stage-0 "
            "tokenizer, so it needs the path of one (a .keras file); train one with "
            "python -m train.beit.train_tokenizer, then give model_args.tokenizer_checkpoint=<its best_model.keras>"
        )

    def _tokenizer_block(self) -> Optional[Dict[str, Any]]:
        """Return the summary's ``tokenizer`` block (the file, the codebook size, the usage and the floor); ``None`` for ET."""
        from train.common import pretrain_run

        if self._usage is None:
            return None
        return {
            "checkpoint": str(self.config.model_args[TOKENIZER_KEY]),
            "num_embeddings": self._vocab_size,
            "positions": "the masked positions of the selection split, one pass",
            **self._usage.as_dict(),
            "floor": pretrain_run.MIN_TOKENIZER_PERPLEXITY,
        }

    def build_model(self, config: PretrainConfig, context: AutoContext, data: Any) -> Any:
        spec = registry.MODELS[context.family]
        # The patch is the data's: the masks and the model share one patch. The adapter returns a model BUILT for
        # `[image, mask]`, so the sequence's own `build` call is skipped.
        kwargs = {**self._model_kwargs(config, context), "patch_size": config.data.patch_size}
        if context.family == BEIT:
            kwargs["vocab_size"] = self._vocab_size
        return recipes.build_model(spec, context.variant, None, data.input_shape, kwargs)

    @staticmethod
    def _model_kwargs(config: PretrainConfig, context: AutoContext) -> Dict[str, Any]:
        """Return the family adapter's keywords: the recipe's, then the caller's (the task's own keys stripped)."""
        spec = registry.MODELS[context.family]
        return spec.load_resolver()(context, config.optim_args, config.model_args)

    def task_record(self, config: PretrainConfig, context: AutoContext, data: Any) -> None:
        # DECISION plan-2026-11-12T015747-46e5318f/D-013
        # No `task.json`: a pretrain run has no reader yet (`predict` and `export` of its directory refuse with the existing
        # `task.json` sentence), and a record is a schema row in every table of `predictor.TASK_TABLES`, a builder and a
        # predictor, which belong to the plan that gives the task a reader. Do NOT write a partial record "for later": a
        # record that exists makes `load_model` dispatch on a task it has no predictor for. A resume works without one
        # (`check_data_unchanged` skips a `None` record and compares the split sizes with the sidecar).
        return None

    def used_arguments(self, config: PretrainConfig, context: AutoContext) -> Dict[str, Any]:
        """Return what the run was built with, for ``config.json``: the model keywords."""
        return {"model_args": recipes.serialize_model_args(self._model_kwargs(config, context))}

    def check_model_args(self, config: PretrainConfig, context: AutoContext) -> None:
        # Before the claim: the family's own refusals (a key it does not offer, an ET run with no global clip, an sgd decay
        # no regularizer would carry), and a BEiT run with no tokenizer (it has no targets).
        self._model_kwargs(config, context)
        if context.family == BEIT:
            self._check_tokenizer_path(config.model_args.get(TOKENIZER_KEY))

    def _check_tokenizer_path(self, path: Any) -> None:
        """Refuse a missing, non-string, non-``.keras`` or absent tokenizer path before the claim (no file is read)."""
        # DECISION plan-2026-11-12T015747-46e5318f/D-007
        # The path is checked BEFORE the claim, by name, because the loader's own `FileNotFoundError` arrives after the run
        # directory exists and as a traceback. Only existence and the suffix are checked here: the file's content (a
        # `VQVAERotationTrick`, its grid, its usage) is read at build time, which this hook must not do (no tensor before
        # `setup_gpu`). Do NOT load the file here "to refuse earlier".
        if not path:
            raise ConfigError(self._tokenizer_required_message())
        name = f"model_args.{TOKENIZER_KEY}"
        if not isinstance(path, (str, Path)):
            raise ConfigError(f"Invalid value for {name!r}: a path (a string), got {path!r}")
        if not str(path).endswith(".keras"):
            raise ConfigError(
                f"Invalid value for {name!r}: {str(path)!r} does not end in .keras; the tokenizer is the .keras file a "
                "stage-0 run saves (python -m train.beit.train_tokenizer)"
            )
        if not Path(path).is_file():
            raise ConfigError(
                f"Invalid value for {name!r}: {str(path)!r} does not exist (or is not a file); train a tokenizer with "
                "python -m train.beit.train_tokenizer and give the path of its .keras"
            )

    def input_check(self, facts: DataFacts, data: Any) -> None:
        check_holds_no_classes(self, facts, data)

    # -- compile --------------------------------------------------------

    def metric_names(self, config: PretrainConfig, facts: DataFacts) -> List[str]:
        # DECISION plan-2026-10-30T132432-a865fa08/D-055: names only; a Keras metric allocates variables and the first
        # eager op would initialise TensorFlow's devices before setup_gpu. There are none: `loss` and `val_loss` only.
        return []

    def build_loss_and_metrics(self, config: PretrainConfig, data: Any) -> Tuple[Any, List[Any]]:
        from train.common.pretrain_run import pretrain_loss

        return pretrain_loss(data.request["method"]), []

    def initial_loss_check(
        self, config: PretrainConfig, model: Any, data: Any
    ) -> Tuple[Dict[str, float], float, Optional[bool]]:
        """Return the epoch-0 numbers; the ratio is the loss over the baseline's, and no verdict is given.

        :raises RuntimeError: for a non-finite loss (the model diverges before training).
        """
        from dl_techniques.utils.logger import logger
        from train.common import classification_run, run_summary

        if self._baseline is None:
            raise RuntimeError(
                "initial_loss_check compares with the constant predictor's loss, which build_data works out; it was "
                "asked for before the data was built"
            )
        metrics = classification_run.evaluate_dict(model, data.select_ds)
        loss = metrics["loss"]
        ratio = run_summary.initial_loss_ratio(metrics, self._baseline.val_loss, self.initial_loss_hint)
        logger.info(
            f"Untrained evaluate BEFORE fit: val loss {loss:.6g} on {data.n_select} selection images, {ratio:.3g}x the "
            f"constant predictor's {self._baseline.val_loss:.6g}"
        )
        # No verdict (`None`): the untrained model's loss against a constant predictor is not a scale test (a random-init
        # ET sat 2 to 3 times ABOVE the zero predictor in the measured runs, for a model that trains fine).
        return metrics, ratio, None

    def unjudged_note(self, config: PretrainConfig) -> str:
        return (
            "`init_scale_warning` is null (not judged): `initial_loss_ratio` is the loss of the untrained model over the "
            "`val_loss` of a constant predictor on the same fixed selection masks, kept as the baseline. A random "
            "initialization sits above the constant predictor by design (the measured random-init energy transformer "
            "scored 2 to 3 times the zero predictor and then trained), so no threshold on that ratio separates a "
            "well-scaled model from a mis-scaled one; a non-finite initial loss is refused"
        )

    # -- the two seams of the shared sequence ---------------------------

    def fit_kwargs(self, config: PretrainConfig, data: Any) -> Dict[str, Any]:
        # DECISION plan-2026-11-12T015747-46e5318f/D-003
        # The train stream is INFINITE (the masks are keyed on a counter that must keep climbing), so an epoch is
        # `steps_per_epoch` batches by this keyword; the sequence refuses an infinite stream without it. `validation_steps`
        # is the selection stream's one pass, stated so that Keras never has to guess a length. Do NOT drop either "because
        # the defaults work": without `steps_per_epoch` the epoch never ends.
        return {"steps_per_epoch": data.steps_per_epoch, "validation_steps": data.val_steps}

    def resume_stream(self, config: PretrainConfig, data: Any, completed_steps: int) -> Optional[Any]:
        # DECISION plan-2026-11-12T015747-46e5318f/D-004
        # A resumed `fit` starts a fresh iterator, so the stream the fresh process built would replay from its first
        # element (epoch 0's masks, shuffle and augmentation). `with_stream` rebuilds the SAME request on a stream seed
        # derived from `(seed, completed_steps)`: the split, the counts and the select and test masks do not move, the train
        # stream does. Do NOT return `data` ("nothing to rebuild": that is the replay), and do NOT reseed the split.
        return data.with_stream(completed_steps)

    # -- the verdict ----------------------------------------------------

    def success_value(self, hist: Dict[str, List[float]]) -> float:
        """Return the best (the lowest) ``val_loss`` of the selection window."""
        start = self.selection_start_epoch(self.config)
        window = hist[self.success_metric][start:]
        return float(min(window))

    def success_threshold(self, config: PretrainConfig) -> float:
        """Return ``min(baseline, epoch-zero val_loss)``; ``evaluate_and_report`` has read the second term."""
        if self._baseline is None:
            raise RuntimeError(
                "success_threshold is cut from the baseline (the constant predictor's val_loss), which build_data works "
                "out; it was asked for before the data was built"
            )
        if self._epoch_zero is None:
            raise RuntimeError(
                "success_threshold needs the epoch-zero val_loss, which evaluate_and_report reads from the fit sidecar; "
                "it was asked for before the reporting"
            )
        return float(min(self._baseline.val_loss, self._epoch_zero))

    def bar_is_met(self, config: PretrainConfig, reached: float, threshold: float) -> bool:
        import numpy as np

        # DECISION plan-2026-11-12T015747-46e5318f/D-009
        # STRICTLY below, and in float32. The base class's `>=` reads a loss backwards; `<=` would call a model that merely
        # EQUALS the constant predictor a success (a constant model reaches exactly the baseline, and it must never meet the
        # bar); and a float64 `reached` that rounds to the float32 threshold must be judged as the float32 numbers Keras
        # computed them, so equal-in-float32 is a miss. Do NOT compare the Python floats.
        return bool(np.float32(reached) < np.float32(threshold))

    # -- the log and the summary ---------------------------------------

    def epoch_line_metrics(self, config: PretrainConfig) -> Tuple[str, ...]:
        return ("loss", "val_loss")

    def summary_fields(self, config: PretrainConfig, data: Any) -> Dict[str, Any]:
        data_cfg: PretrainData = config.data
        side = data.input_shape[0] // data_cfg.patch_size
        return {
            "method": data.request["method"],
            "mask": {
                "patch_size": data_cfg.patch_size,
                "grid": [side, side],
                "num_patches": side * side,
                **{key: getattr(data_cfg, key) for key in MASK_KEYS},
            },
            "stream": {
                "seed": data.seed,
                "stream_seed": data.stream_seed,
                "rule": (
                    "the selection and test masks are keyed on (seed, split, row position) and never move; the train "
                    "mask and shuffle on (stream_seed, counter); a resume derives stream_seed from (seed, completed "
                    "steps), the first 4 bytes of sha256, and the split never follows it"
                ),
            },
        }

    def recipe_notices(self, facts: DataFacts, family: str) -> List[Tuple[str, str]]:
        return recipe_notices_of(family)

    def task_notes(self, config: PretrainConfig, baseline: Dict[str, float], ratio: float, data: Any) -> List[str]:
        """Return the pretrain lines of ``results_summary.json``'s ``notes``."""
        from train.common import classification_run, pretrain_run

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        constant = self._baseline
        method = data.request["method"]
        loss = (
            "the mean over the loss patches of the patch MSE (the mask is a sample weight N / n_loss)"
            if method == METHOD_OF["energy_transformer"]
            else "the mean over examples of the per-example masked-token cross-entropy (sample weight N / n_masked: the "
            "reference's scale, not the legacy trainer's, which is that mean times the mask fraction)"
        )
        notes = [
            f"initial loss {baseline['loss']:.6g} is {ratio:.3g}x the constant predictor's val_loss "
            f"({constant.val_loss:.6g}, {constant.kind}) on the "
            f"{data.n_select} fixed selection images: a plain evaluate in inference mode; see the `init_scale_warning` note",
            *classification_run.RUN_NOTES[:2],
            f"`val_loss` is {loss}; the selection and test masks are fixed by (seed, split, row position), so it is a "
            "function of the weights alone (equal across passes, epochs and a resume)",
            "`success_value` is the lowest `val_loss` of the selection window; `success_threshold` is "
            "min(`baseline.val_loss`, `epoch_zero.val_loss`): the constant predictor's loss on the same stream and the "
            "untrained model's, read from the fit sidecar; the bar is STRICTLY below, compared in float32",
            "`test_metrics_best` is the reloaded best_model.keras and `test_metrics_final` the last epoch's weights, on the "
            "test split, whose masks are fixed too; the test split never influenced selection",
            "no `task.json` is written: `predict` and `export` of this run directory are refused",
        ]
        if self._usage is not None:
            notes.append(
                f"the tokenizer's usage over the masked selection positions (`tokenizer`: perplexity "
                f"{self._usage.perplexity:.2f}, {self._usage.distinct} distinct codes) passed the floor of "
                f"{pretrain_run.MIN_TOKENIZER_PERPLEXITY:g} effective codes; a lower one is refused, because the bar of a "
                "near-constant target is vacuous"
            )
        if analysis_cfg.model_analysis:
            notes.append(
                "the model analysis is data-free (weights and spectral only; `analyzer.analyzers` lists them): the stock "
                "analyses read class probabilities, which a masked-patch output is not"
            )
        else:
            notes.append("analysis_args.model_analysis is false: `analyzer.status` is 'skipped'")
        if not checkpoint_cfg.save_best:
            notes.append("checkpoint_args.save_best is false: no best_model.keras was written, so the reload check did not run")
        if not checkpoint_cfg.save_last:
            notes.append(
                "checkpoint_args.save_last is false: no final_model.keras was written and `model_loading_validated` is null"
            )
        return notes

    # -- the figures ----------------------------------------------------

    def figure_callbacks(self, config: PretrainConfig, data: Any, vis_dir: Path) -> List[Any]:
        """Return the energy trace of ET (none for BEiT): the energy descent of one fixed selection batch per epoch."""
        if data.request["method"] != METHOD_OF["energy_transformer"]:
            return []
        from train.energy_transformer.common import EnergyTraceCallback

        # DECISION plan-2026-11-12T015747-46e5318f/D-010
        # The trace is attached HERE (it needs the directory, which `extra_callbacks` is not given) and a resume appends
        # to the file the first process began: one header across the whole run. The probe batch is the first SELECTION
        # batch, whose masks are fixed, so the traces of two epochs (and of two processes) are comparable. Do NOT attach it
        # through `extra_callbacks` or the private `_callbacks`, and do NOT write the header on every process.
        probe_inputs = next(iter(data.select_ds))[0]
        return [
            EnergyTraceCallback(
                probe_inputs=probe_inputs,
                csv_path=str(vis_dir / "energy_trace.csv"),
                skip_header_if_file_exists=bool(config.resume_interrupted),
            )
        ]

    # -- the reporting --------------------------------------------------

    def run_analysis(
        self,
        config: PretrainConfig,
        model: Any,
        test_data: Tuple[Any, Any],
        history: Any,
        experiment_name: str,
        run_dir: str,
    ) -> Any:
        """Run the data-free analysis (weights and spectral only); see the module contract."""
        from train.common import run_summary

        # The stock analysis reads the output as class probabilities (calibration, information flow, training dynamics)
        # and the target as labels; the output of a masked-patch model is neither (measured: the data-free analysis
        # completes on both models and both forms of x, `analyzer.status` ok). Do NOT fall back to the default hook.
        return run_summary.data_free_model_analysis(model, test_data, history, experiment_name, run_dir, config.seed)

    def evaluate_and_report(self, config: PretrainConfig, data: Any, fit: FitState) -> Evaluation:
        """Run the pretrain reporting sequence; see the module contract."""
        from types import SimpleNamespace

        from train.common import classification_run, pretrain_run, run_summary
        from train.common.resume_state import FIT_STATE_NAME

        checkpoint_cfg: CheckpointArgs = config.checkpoint_args
        analysis_cfg: AnalysisArgs = config.analysis_args
        metric_cfg: PretrainMetricArgs = config.metric_args
        # The epoch-zero term of the bar, from the sidecar: see `epoch_zero_val_loss` (a resume reads the same number).
        self._epoch_zero = pretrain_run.epoch_zero_val_loss(fit.run_dir / CHECKPOINTS_DIR / FIT_STATE_NAME)
        baseline = self._baseline
        if baseline is None:
            raise RuntimeError("evaluate_and_report needs the baseline build_data works out; the data was not built")
        test_best: Optional[Dict[str, float]] = None
        load_error: Optional[str] = None
        max_diff: Optional[float] = None
        if checkpoint_cfg.save_best:
            # The in-memory model holds the BEST weights here (EarlyStopping restored them). The test masks are fixed, so
            # the reloaded file reproduces them exactly: the reload gap is 0.0, not a tolerance.
            test_best, load_error, max_diff = classification_run.check_best_checkpoint(
                fit.model,
                fit.run_dir,
                data.test_ds,
                evaluate=classification_run.evaluate_dict,
                split_label="Test results",
            )
        # One fixed selection batch: the model's two inputs (image, mask) and its target, the sample of the analysis and
        # of the reload check of `final_model.keras`.
        (image, mask), target, _ = next(iter(data.select_ds))
        inputs = [image.numpy(), mask.numpy()]
        # The best weights are analysed, then the last epoch's replace them (the sequence the denoising task shares).
        analysis, test_final, load_check, load_detail = run_summary.report_data_free_run(
            fit.model,
            SimpleNamespace(x_fig=inputs, y_fig=target.numpy()),
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
            load_sample=[array[: classification_run.LOAD_CHECK_SAMPLES] for array in inputs],
        )
        visualizations: Dict[str, Any] = {"files": [], "failed": [], "skipped": not analysis_cfg.figures}
        if analysis_cfg.figures and fit.vis_dir.is_dir():
            visualizations["files"] = sorted(path.name for path in fit.vis_dir.iterdir() if path.is_file())
        hist = {key: [float(v) for v in values] for key, values in fit.history.history.items()}
        start = self.selection_start_epoch(config)
        best = run_summary.best_epoch(hist, metric_cfg.watch_metric, metric_cfg.watch_mode, first_epoch=start)
        extras: Dict[str, Any] = {
            "baseline": {"kind": baseline.kind, "val_loss": baseline.val_loss},
            "epoch_zero": {"val_loss": self._epoch_zero, "binds": bool(self._epoch_zero < baseline.val_loss)},
            "selection": selection_block(start, fit.epochs_run, best),
            "tokenizer": self._tokenizer_block(),
        }
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


def selection_block(start: int, epochs_run: int, best: int) -> Dict[str, Any]:
    """Return the summary's ``selection`` block.

    :param start: the 0-based epoch the selection window starts at.
    :param epochs_run: the epochs the run trained (the last epoch's 1-based number).
    :param best: the 1-based epoch of the best ``watch_metric`` in the window (``run_summary.best_epoch``).
    :return: the window's start and length, the best epoch, and ``final_is_best``: true exactly when the best epoch is the
        last one run (``best`` is 1-based, so that is ``best == epochs_run``).
    """
    return {
        "start_epoch": start,
        "window_epochs": epochs_run - start,
        "best_epoch": best,
        "final_is_best": best == epochs_run,
    }


def train_pretrain(
    *,
    out: Union[str, Path],
    model: str,
    data: Union[PretrainData, Mapping[str, Any], str, Path],
    epochs: Union[int, str] = AUTO,
    batch_size: Union[int, str] = AUTO,
    seed: int = DEFAULT_SEED,
    devices: Union[int, str, None] = AUTO,
    precision: str = "float32",
    resume_interrupted: bool = False,
    checkpoint: Union[str, Path, None] = None,
    model_args: Optional[Mapping[str, Any]] = None,
    optim_args: Union[OptimArgs, Mapping[str, Any], None] = None,
    transform_args: Union[PretrainTransformArgs, Mapping[str, Any], None] = None,
    metric_args: Union[PretrainMetricArgs, Mapping[str, Any], None] = None,
    checkpoint_args: Union[CheckpointArgs, Mapping[str, Any], None] = None,
    logger_args: Union[LoggerArgs, Mapping[str, Any], None] = None,
    analysis_args: Union[AnalysisArgs, Mapping[str, Any], None] = None,
    debug_args: Union[DebugArgs, Mapping[str, Any], None] = None,
) -> TrainResult:
    """Train a masked-image model; see the module contract."""
    recipes.register_builtin_models()
    if isinstance(data, (str, Path)):
        data = load_pretrain_data_config(data)
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
    config = validate(PretrainConfig, given)
    return PretrainTrainer(config).run()
