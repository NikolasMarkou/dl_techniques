"""
The reusable parts of a labelled-image classification run.

``train.convnext.common.train`` is the reference sequence: fit, divergence check,
best-checkpoint evaluation, conformal sets, figures, analysis, restore of the last
epoch's weights, final evaluation, final save, history, load check, summary. The blocks
below are that sequence cut at its seams, moved here without a change of behaviour so a
second caller (the ``train.api`` image-classification task) runs the same code in the
same order. Only one order can live in a shared helper, and this one is the legacy order.

Call-time seams. Several helpers need a function that tests of the legacy trainer patch
on THEIR module (``_evaluate``, ``run_model_analysis``). A helper that imported and
called those itself would no longer be intercepted, so each is a parameter: the caller
reads the name from its own module globals at the call and passes it in. The helpers
that reach ``train.common.run_summary`` do so through the module object
(``run_summary.load_best_metrics(...)``), so a patch on that module is still seen.

The helpers write no ``results_summary.json`` and own no run directory: the caller keeps
that, and keeps the divergence branch, the conformal call and every ``notes`` line.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf

from dl_techniques.metrics.half_precision_top_k import HalfPrecisionTopKAccuracy
from dl_techniques.utils.logger import logger

from train.common import run_summary
from train.common.callbacks import partial_path, promote_partial
from train.common.classification_data import EVAL_BATCH_SIZE, ClassificationData
from train.common.evaluation import predict_batch_cap, validate_model_loading
from train.common.run_io import save_training_history_json

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Rows fed to the round-trip check of ``final_model.keras``.
LOAD_CHECK_SAMPLES = 64

# The reloaded best checkpoint should reproduce the in-memory best weights exactly.
WEIGHT_MISMATCH_TOLERANCE = 1e-6

# ``check_initial_loss`` warns when loss / ln(num_classes) is STRICTLY above this. A
# hazard floor chosen independently of the PowerMLP trainer's equal value.
INITIAL_LOSS_WARN_FACTOR = 10.0

# ``results_summary.json["status"]``.
STATUS_OK = "ok"
STATUS_DIVERGED = "diverged"

#: The three lines of a finished run's ``notes`` that describe the CSV and the two test scores; they
#: hold for every task that runs this sequence (classification, segmentation), so there is one copy.
RUN_NOTES = (
    "CSV `epoch` is 0-based, `best_epoch` is 1-based (`best_epoch_csv_index` = `best_epoch` - 1)",
    "CSV `lr` is the rate at the START of the epoch (its first step)",
    "`test_metrics_best` is the reloaded best_model.keras, `test_metrics_final` "
    "the last epoch's weights; the test set never influenced selection",
)

# Signature of the ``evaluate`` seam: a model and a batched dataset to ``{metric: float}``.
Evaluate = Callable[[keras.Model, "tf.data.Dataset"], Dict[str, float]]


def evaluate_dict(model: keras.Model, dataset: "tf.data.Dataset") -> Dict[str, float]:
    """``model.evaluate`` over a batched dataset as a plain ``{metric: float}`` dict.

    The default :data:`Evaluate` seam. A caller whose tests patch the seam by name on
    their own module (``train.convnext.common._evaluate``) binds this under that name.

    :param model: A compiled model.
    :param dataset: A batched dataset of ``(x, y)``.
    :return: Every metric ``evaluate`` reports, as floats.
    """
    metrics = model.evaluate(dataset, verbose=0, return_dict=True)
    return {k: float(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------
# Metrics that survive a mixed-precision policy
# ---------------------------------------------------------------------

def classification_metric_names(num_classes: int, top_k: int) -> List[str]:
    """Return the names :func:`classification_metrics` gives its metrics, building none.

    Interface contract: pure, no TensorFlow or Keras call. A Keras metric allocates
    variables, and the first eager op initialises TensorFlow's devices, after which
    ``setup_gpu`` can no longer enable memory growth (D-055). A caller that needs only
    the names before the GPU is set up (the API's monitor check) uses this function.

    :param num_classes: Class count; ``top{k}_accuracy`` exists only above ``top_k``.
    :param top_k: ``k``.
    :return: ``["accuracy"]`` or ``["accuracy", "top{k}_accuracy"]``.
    """
    names = ["accuracy"]
    if num_classes > top_k:
        names.append(f"top{top_k}_accuracy")
    return names


def classification_metrics(
    num_classes: int, top_k: int, half_precision: bool = False
) -> List[keras.metrics.Metric]:
    """``accuracy``, plus ``top{k}_accuracy`` when there are more than ``k`` classes.

    :param num_classes: Class count; top-k of every class would be 1.0 and tells nothing.
    :param top_k: ``k``.
    :param half_precision: ``True`` when the model emits half-precision logits (a
        ``mixed_float16`` or ``mixed_bfloat16`` policy): the top-k metric is then
        :class:`~dl_techniques.metrics.half_precision_top_k.HalfPrecisionTopKAccuracy`.
        ``False`` uses the stock metric, so a float32 model carries no class from this
        repository and reloads with a plain ``load_model``.
    :return: Metric OBJECTS (a string alias such as ``"top_5_accuracy"`` is not resolvable
        by Keras and crashes ``compile``), named by :func:`classification_metric_names` in
        both cases, so a ``watch_metric`` holds whichever precision the run uses.
    """
    names = classification_metric_names(num_classes, top_k)
    metrics: List[keras.metrics.Metric] = [
        keras.metrics.SparseCategoricalAccuracy(name=names[0])
    ]
    if len(names) > 1:
        top_k_class = (
            HalfPrecisionTopKAccuracy if half_precision else keras.metrics.SparseTopKCategoricalAccuracy
        )
        metrics.append(top_k_class(k=top_k, name=names[1]))
    return metrics


# ---------------------------------------------------------------------
# H1: the last epoch's weights
# ---------------------------------------------------------------------

class LastEpochWeights(keras.callbacks.Callback):
    """Keeps a copy of the weights at the end of the most recent completed epoch.

    ``create_callbacks`` builds ``EarlyStopping(restore_best_weights=True)`` and Keras
    3.8 restores the best weights at EVERY train end, so after ``fit`` the in-memory
    model is the best one and the true last-epoch weights are gone unless captured.

    :ivar weights: The ``model.get_weights()`` list of the last finished epoch, or
        ``None`` before the first epoch ends.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weights: Optional[List[np.ndarray]] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self.weights = self.model.get_weights()


# ---------------------------------------------------------------------
# H2: the epoch-0 evaluation
# ---------------------------------------------------------------------

def unset_batch_statistics(model: keras.Model) -> List[str]:
    """Names of the layers whose moving statistics are still the ones they were built with.

    Interface contract. A layer counts when it is trainable and holds a
    ``moving_variance`` (Keras ``BatchNormalization`` and its subclasses) or a
    ``running_var`` (``BiasFreeBatchNorm``), the variance is all ones and, when the layer
    has a ``moving_mean``, the mean is all zeros. A frozen layer (``trainable`` false)
    never uses batch statistics, so an inference-mode pass IS its training behaviour and
    it is not listed. Only variables are read: nothing is assigned, nothing is run, so
    neither a moving statistic nor a random generator moves.

    :param model: A BUILT model (a layer not built yet has no statistics and is not listed).
    :return: The listed layers' names in graph order; empty for a model without
        batch normalization or one whose statistics have moved.
    """
    names = []
    for layer in model._flatten_layers(include_self=False, recursive=True):
        variance = getattr(layer, "moving_variance", None)
        if variance is None:
            variance = getattr(layer, "running_var", None)
        if variance is None or not layer.trainable:
            continue
        mean = getattr(layer, "moving_mean", None)
        at_init = bool(np.all(variance.numpy() == 1.0))
        if mean is not None:
            at_init = at_init and bool(np.all(mean.numpy() == 0.0))
        if at_init:
            names.append(layer.name)
    return names


def check_initial_loss(
        model: keras.Model, dataset: "tf.data.Dataset", n_samples: int, num_classes: int,
        evaluate: Evaluate, hint: str,
) -> Tuple[Dict[str, float], float, Optional[bool]]:
    """Evaluate the untrained model; refuse a non-finite loss, warn on a huge one.

    The single epoch-0 measurement: both the summary's initial-loss ratio and the
    dashboard's baseline marker come from THIS result, so the two cannot disagree. A
    LayerNorm-only model has no training-mode statistics, so a plain inference-mode
    ``evaluate`` is the true loss of the first training step's starting point.

    A model with BatchNorm layers whose moving statistics are still at their initial
    mean 0 and variance 1 is NOT that case: the inference-mode pass applies no
    normalization at all, while the first training step normalizes with batch
    statistics (measured on resnet18, ImageNet shapes, same weights: 120.4 in inference
    mode, 7.15 in training mode). The pass is kept as the epoch-0 baseline (the
    dashboard's marker is what the evaluate measured), but it says nothing about the
    initialization, so no verdict is given: ``warned`` is ``None`` and the log says what
    was measured. No training-mode pass is run to get a verdict: it would move the
    moving statistics and every dropout generator, which a diagnostic must not do
    (:func:`unset_batch_statistics` only reads). A badly scaled BatchNorm model is
    therefore not caught here; its first epochs show it.

    :param model: Built and compiled model.
    :param dataset: The batched validation (selection) split.
    :param n_samples: Its sample count (for the log line).
    :param num_classes: Class count; the uniform-prediction loss is ``ln(num_classes)``.
    :param evaluate: The caller's ``(model, dataset) -> {metric: float}`` seam.
    :param hint: The remedy sentence appended to the refusal, naming the caller's own
        options (a command-line flag, a config key).
    :return: ``(metrics, ratio, warned)``: the evaluate dict, ``loss / ln(num_classes)``
        and whether the ratio is STRICTLY above :data:`INITIAL_LOSS_WARN_FACTOR`;
        ``None`` instead of a bool when the model has unset BatchNorm statistics (not
        judged, see above).
    :raises RuntimeError: If the loss is NaN or infinite (raising, not returning, so no
        empty run directory is left behind as a "finished" run).
    """
    metrics = evaluate(model, dataset)
    unset = unset_batch_statistics(model)  # after the pass: a lazily built model is built now
    loss = metrics["loss"]
    uniform_loss = float(np.log(num_classes))
    logger.info(
        f"Untrained evaluate BEFORE fit: val loss {loss:.4f} on {n_samples} samples "
        f"(uniform-prediction loss would be {uniform_loss:.4f})"
    )
    if not np.isfinite(loss):
        raise RuntimeError(
            f"Initial validation loss is {loss}: the model diverges before training. {hint}"
        )
    ratio = loss / uniform_loss
    # DECISION plan-2026-10-31T180033-2faf890a/D-045: a model with BatchNorm statistics still at
    # their initial values gets NO verdict (``None``), and no training-mode pass is run to get
    # one. Do NOT call the model with ``training=True`` here (it assigns the moving statistics
    # and advances every dropout generator, which a diagnostic must not do), and do NOT go back
    # to ``ratio > INITIAL_LOSS_WARN_FACTOR`` for these models: that told every ResNet its
    # initialization was wrong (17x ln C in inference mode, 1.04x in training mode).
    if unset:
        logger.info(
            f"Initial loss is {ratio:.1f}x ln(num_classes): measured in inference mode with "
            f"the moving statistics of {len(unset)} BatchNorm layer(s) (first: {unset[0]}) "
            "still at mean 0 and variance 1, which normalizes nothing, so it is not the "
            "first training step's starting point (that step uses batch statistics). No "
            "verdict on the initialization is given (init_scale_warning is null)."
        )
        return metrics, ratio, None
    warned = ratio > INITIAL_LOSS_WARN_FACTOR
    if warned:
        logger.warning(
            f"Initial loss {loss:.4f} is {ratio:.1f}x ln(num_classes)={uniform_loss:.4f} "
            f"(warn above {INITIAL_LOSS_WARN_FACTOR:g}x): the initial logit scale is far too "
            "large and the early epochs will be spent recovering."
        )
    return metrics, ratio, warned


# ---------------------------------------------------------------------
# H3: what the fit history says
# ---------------------------------------------------------------------

def summarize_fit(
        history: keras.callbacks.History, monitor: str, constant_lr: bool,
) -> Tuple[Dict[str, List[float]], int, List[str], Optional[List[int]]]:
    """Read a finished ``fit`` history and log the ReduceLROnPlateau reductions.

    A constant schedule adds ReduceLROnPlateau, whose message Keras prints to stdout,
    never to the logger; the rate history is how the run's own log and summary learn
    about it. Under cosine / exponential the rate falls every epoch by design, so the
    reduction list is ``None`` there (not applicable: ReduceLROnPlateau is not
    installed), never an empty list that could be read as "constant and no plateau".

    :param history: The ``History`` returned by ``model.fit``.
    :param monitor: The metric that drives early stopping and the best checkpoint.
    :param constant_lr: Whether the run's schedule is ``constant``.
    :return: ``(hist, epochs_run, non_finite, lr_reduction_epochs)``: the history as
        plain floats, the number of epochs that ran, the names among ``loss`` and
        ``monitor`` holding a non-finite or missing value, and the 1-based epochs whose
        rate fell (``None`` unless ``constant_lr``).
    """
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    epochs_run = len(hist.get(monitor, []))
    non_finite = run_summary.non_finite_metrics(hist, monitor)
    lr_reduction_epochs = (
        run_summary.lr_reduction_epochs(hist.get("lr", [])) if constant_lr else None
    )
    for epoch in lr_reduction_epochs or []:
        logger.info(
            f"ReduceLROnPlateau: learning rate {hist['lr'][epoch - 2]:.3g} -> "
            f"{hist['lr'][epoch - 1]:.3g} from epoch {epoch}"
        )
    return hist, epochs_run, non_finite, lr_reduction_epochs


# ---------------------------------------------------------------------
# H4: the best checkpoint on disk
# ---------------------------------------------------------------------

def check_best_checkpoint(
        model: keras.Model, run_dir: Path, test_ds: "tf.data.Dataset", evaluate: Evaluate,
        split_label: str = "Test results",
) -> Tuple[Optional[Dict[str, float]], Optional[str], Optional[float]]:
    """Test metrics of the reloaded ``best_model.keras`` and its gap to ``model``.

    ``model`` must hold the best weights (true after ``fit``: EarlyStopping restores
    them). The reloaded checkpoint must reproduce them, so a gap above
    :data:`WEIGHT_MISMATCH_TOLERANCE` is logged as a bug signal.

    :param model: The model holding the best weights.
    :param run_dir: The run directory holding ``best_model.keras``.
    :param test_ds: The batched test split (only scored, never used to select).
    :param evaluate: The caller's ``(model, dataset) -> {metric: float}`` seam.
    :param split_label: How the log line names ``test_ds``. A caller whose data has no test
        split passes the selection split to keep the reload check and names it here; the
        metrics it gets back are then NOT test metrics.
    :return: ``(metrics, load_error, max_abs_diff)``; ``metrics`` and ``max_abs_diff``
        are ``None`` when the checkpoint is missing or does not load.
    """
    reloaded, load_error = run_summary.load_best_metrics(
        run_dir, lambda m: evaluate(m, test_ds))
    if reloaded is None:
        return None, load_error, None
    logger.info(f"{split_label} (best_model.keras): {reloaded}")
    in_memory = evaluate(model, test_ds)
    gap = max(abs(in_memory[k] - reloaded[k]) for k in in_memory if k in reloaded)
    if gap > WEIGHT_MISMATCH_TOLERANCE:
        logger.warning(
            f"best_model.keras and the in-memory best weights disagree on the "
            f"{split_label.lower()} data by {gap:.3g}"
        )
    return reloaded, None, gap


# ---------------------------------------------------------------------
# H5: figures and analysis of the best weights
# ---------------------------------------------------------------------

def report_best_weights(
        model: keras.Model, data: ClassificationData, history: keras.callbacks.History,
        run_dir: Path, vis_dir: Path, experiment_name: str, model_analysis: bool,
        run_analysis: Callable[..., Any], figures: bool = True,
        analysis_off_hint: str = "--no-model-analysis", name_failed_analyses: bool = False,
        eval_batch_size: int = EVAL_BATCH_SIZE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Write the figures and run the analyzer for the BEST (selected) weights.

    The model emits LOGITS, so every probability consumer gets a softmax first.

    :param model: The model still holding the best weights.
    :param data: The loaded data; ``x_fig``, ``y_fig``, ``class_names``, ``mean`` and
        ``std`` are read.
    :param history: The ``History`` returned by ``model.fit``.
    :param run_dir: The run directory.
    :param vis_dir: Where the figures go.
    :param experiment_name: The model name the analyzer files its output under.
    :param model_analysis: Whether to run the analyzer.
    :param run_analysis: The caller's ``run_model_analysis`` seam, called as
        ``(model, (x_fig, y_fig), history, experiment_name, str(run_dir))``.
    :param figures: ``False`` writes no figure and does not predict; the record is then
        ``{"files": [], "ece": None, "failed": [], "skipped": True}``. The default
        reproduces the legacy trainers.
    :param analysis_off_hint: How the caller's user switched the analyzer off; named in the
        log line of a skipped analysis. The default is the legacy flag; the API passes its
        own key.
    :param name_failed_analyses: ``True`` adds ``failed_analyses`` (``{name: reason}``) to the
        analyzer status when :func:`train.common.run_summary.failed_analyses` finds one (the
        seam must then return what ``run_model_analysis`` returns), and a status of
        ``success`` becomes ``partial``. Off by default: the legacy trainers' summaries keep
        their keys and values. With nothing failed the key is absent either way.
    :param eval_batch_size: Rows per pass of the figures' ``predict`` (default
        :data:`~train.common.classification_data.EVAL_BATCH_SIZE`, the legacy value). The API
        passes ``analysis_args.eval_batch_size``: 256 images of a 224 px cliffordnet ask for
        18 GB, which is how the reporting of the ImageNet gate run died.
    :return: ``(visualizations, analysis)``: the figure record (it carries ``ece``) and
        the analyzer status read back from disk, or the skipped status.
    """
    if figures:
        logits = model.predict(data.x_fig, batch_size=eval_batch_size, verbose=0)
        probs = keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))
        visualizations = run_summary.write_classification_figures(
            vis_dir, data.x_fig, data.y_fig, probs, data.class_names,
            mean=data.mean, std=data.std,
        )
    else:
        visualizations = {"files": [], "ece": None, "failed": [], "skipped": True}
    if model_analysis:
        results = run_analysis(model, (data.x_fig, data.y_fig), history, experiment_name, str(run_dir))
        analysis = run_summary.read_analysis_status(run_dir, experiment_name)
        if name_failed_analyses:
            failed = run_summary.failed_analyses(results, experiment_name, run_dir)
            if failed:
                analysis = {**analysis, "failed_analyses": failed}
                if analysis["status"] == "success":
                    analysis["status"] = run_summary.PARTIAL_ANALYSIS_STATUS
                logger.warning(f"Analyzer: {sorted(failed)} left no result: {failed}")
        logger.info(f"Analyzer status (read back from disk): {analysis['status']}")
    else:
        analysis = run_summary.skipped_analysis_status()
        logger.info(f"Analyzer skipped ({analysis_off_hint})")
    return visualizations, analysis


# ---------------------------------------------------------------------
# H6: the last epoch's model
# ---------------------------------------------------------------------

def finalize_last_epoch(
        model: keras.Model, last_weights: LastEpochWeights, data: ClassificationData,
        run_dir: Path, history: keras.callbacks.History, epochs_run: int,
        evaluate: Evaluate, save: bool = True, atomic: bool = False,
        eval_batch_size: Optional[int] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[bool]]:
    """Restore the last epoch's weights, score them, save them, write the history, reload.

    The order is the contract: weights, final evaluation, save, history, load check. The
    history is written after the save, so a failed history write raises and cannot cost
    the weights. A load error surfaces; a value mismatch is the recorded ``False``.

    :param model: The model, holding the best weights on entry and the last epoch's on
        return.
    :param last_weights: The :class:`LastEpochWeights` callback that ran during ``fit``.
    :param data: The loaded data; ``test_ds`` and ``x_fig`` are read. A ``test_ds`` of
        ``None`` (no test split) skips the final evaluation, and the load check reads its
        sample from the first batch of ``select_ds``.
    :param run_dir: The run directory; ``final_model.keras`` is written there.
    :param history: The ``History`` returned by ``model.fit``.
    :param epochs_run: The number of epochs that ran (for the log line).
    :param evaluate: The caller's ``(model, dataset) -> {metric: float}`` seam.
    :param save: ``False`` skips the ``final_model.keras`` write and therefore the load
        check, which has nothing to reload; the history is still written.
    :param atomic: ``True`` saves to ``final_model.partial.keras`` and moves it over
        ``final_model.keras`` with :func:`train.common.callbacks.promote_partial`, so a kill
        inside the save never leaves a torn ``final_model.keras``. ``False`` (the default, the
        legacy trainers) saves in place.
    :param eval_batch_size: The run's evaluation batch, or ``None`` (the legacy trainers).
        The load check's two ``predict`` calls run at Keras' 32 unless this is lower
        (:func:`~train.common.evaluation.predict_batch_cap`).
    :return: ``(test_metrics_final, load_check)``; ``load_check`` is ``None`` when
        ``save`` is ``False``, ``test_metrics_final`` when there is no test split.
    """
    # DECISION plan-2026-09-19T040641-db6932ec/D-012: the FINAL model is the last
    # epoch's weights, captured by ``LastEpochWeights``, because Keras 3.8
    # EarlyStopping restores the best weights at every train end. Do NOT save
    # ``model`` as "final" without the ``set_weights`` below: it would silently be
    # the best model and ``test_metrics_final`` would equal ``test_metrics_best`` by
    # construction, hiding any best-versus-final difference.
    model.set_weights(last_weights.weights)
    test_metrics_final = None
    if data.test_ds is not None:
        test_metrics_final = evaluate(model, data.test_ds)
        logger.info(f"Test results (final weights, epoch {epochs_run}): {test_metrics_final}")
    final_path = run_dir / "final_model.keras"
    if save and atomic:
        # DECISION plan-2026-10-31T180033-2faf890a/D-029
        # The API path writes the final model to a partial name and moves it into place. Do NOT
        # save in place here for the API: a kill inside `model.save` tore `final_model.keras` in
        # a run whose fit had finished, which cannot be resumed, so the last epoch's weights
        # were lost (review-iter-1 blind spot). The legacy default stays in place on purpose.
        model.save(partial_path(str(final_path)))
        promote_partial(str(final_path))
    elif save:
        model.save(final_path)
    # After the save: a failed history write raises and must not cost the
    # weights (D-174). The diverged branch of a caller writes it before raising.
    save_training_history_json(history, str(run_dir))
    load_check: Optional[bool] = None
    if save:
        if data.test_ds is None:
            sample = next(iter(data.select_ds.take(1)))[0].numpy()[:LOAD_CHECK_SAMPLES]
        else:
            sample = data.x_fig[:LOAD_CHECK_SAMPLES]
        cap = predict_batch_cap(eval_batch_size)
        # The keyword is passed only when there is one: tests of the legacy trainers patch
        # `validate_model_loading` with the three-argument signature it always had.
        load_check = bool(validate_model_loading(
            str(final_path), sample, model.predict(sample, batch_size=cap, verbose=0),
            **({} if cap is None else {"batch_size": cap}),
        ))
    return test_metrics_final, load_check


# ---------------------------------------------------------------------
# H7: the keys every finished summary carries
# ---------------------------------------------------------------------

def base_summary(
        summary_head: Dict[str, Any], hist: Dict[str, List[float]], epochs_run: int,
        epochs_requested: int, best_epoch: int, lr_reduction_epochs: Optional[List[int]],
        test_metrics_best: Optional[Dict[str, float]],
        test_metrics_final: Optional[Dict[str, float]], best_load_error: Optional[str],
        checkpoint_max_diff: Optional[float], epoch_times: List[float],
        fit_wall_seconds: float, visualizations: Dict[str, Any], load_check: Optional[bool],
        conformal: Optional[Dict[str, Any]], analysis: Dict[str, Any], notes: List[str],
) -> Dict[str, Any]:
    """The summary of a finished (``status: "ok"``) run, as one dict.

    The caller supplies its own ``summary_head`` (configuration keys) and every note;
    the keys below are the ones every classification run reports.

    :param summary_head: Keys the caller's summary opens with (after ``status``).
    :param hist: The history from :func:`summarize_fit`.
    :param epochs_run: Epochs that ran.
    :param epochs_requested: Epochs asked for (``stopped_early`` is ``epochs_run`` below it).
    :param best_epoch: 1-based best epoch of the monitored metric.
    :param lr_reduction_epochs: From :func:`summarize_fit`.
    :param test_metrics_best: From :func:`check_best_checkpoint`.
    :param test_metrics_final: From :func:`finalize_last_epoch`.
    :param best_load_error: From :func:`check_best_checkpoint`.
    :param checkpoint_max_diff: From :func:`check_best_checkpoint`.
    :param epoch_times: Seconds per epoch.
    :param fit_wall_seconds: Wall time of ``fit``.
    :param visualizations: From :func:`report_best_weights`; it carries ``ece``, which the
        summary repeats at top level. A record without the key (another task's) writes no
        ``ece`` key.
    :param load_check: From :func:`finalize_last_epoch`.
    :param conformal: The conformal record, or ``None``.
    :param analysis: From :func:`report_best_weights`.
    :param notes: Plain-language lines, in the caller's order.
    :return: A strict-JSON-ready dict.
    """
    best_i, final_i = best_epoch - 1, epochs_run - 1
    val_keys = [k for k in hist if k.startswith("val_")]
    return {
        "status": STATUS_OK,
        **summary_head,
        "epochs_run": epochs_run,
        "stopped_early": epochs_run < epochs_requested,
        "best_epoch": best_epoch,
        "best_epoch_csv_index": best_i,
        "final_is_best": best_epoch == epochs_run,
        "lr_first_epoch": hist["lr"][0] if hist.get("lr") else None,
        "lr_last_epoch": hist["lr"][-1] if hist.get("lr") else None,
        "lr_reduction_epochs": lr_reduction_epochs,
        "best_val_metrics": {k: hist[k][best_i] for k in val_keys},
        "final_val_metrics": {k: hist[k][final_i] for k in val_keys},
        "test_metrics_best": test_metrics_best,
        "test_metrics_final": test_metrics_final,
        "best_checkpoint_load_error": best_load_error,
        "best_checkpoint_max_abs_diff": checkpoint_max_diff,
        "epoch_times": epoch_times,
        "fit_wall_seconds": fit_wall_seconds,
        # A task whose figure record has no `ece` (not a probability classifier) writes no key.
        **({"ece": visualizations["ece"]} if "ece" in visualizations else {}),
        "visualizations": visualizations,
        "model_loading_validated": load_check,
        "conformal": conformal,
        "analyzer": analysis,
        "notes": notes,
    }
