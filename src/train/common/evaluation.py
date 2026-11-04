"""Common evaluation, visualization, and analysis utilities for training scripts."""

import dataclasses
import os
import keras
import numpy as np
import tensorflow as tf
from typing import Tuple, Any, List, Dict, Optional

from dl_techniques.utils.logger import logger
from dl_techniques.visualization import (
    VisualizationManager,
    PlotConfig,
    PlotStyle,
    ColorScheme,
    TrainingHistory,
    TrainingCurvesVisualization,
    ConfusionMatrixVisualization,
    NetworkArchitectureVisualization,
    ModelComparisonBarChart,
    ROCPRCurves,
    ClassificationResults,
    RegressionResults,
    MultiModelRegression,
    TimeSeriesEvaluationResults,
)
from dl_techniques.analyzer import ModelAnalyzer, AnalysisConfig, DataInput


# ---------------------------------------------------------------------
# The prediction batch of a reporting pass
# ---------------------------------------------------------------------

#: The batch Keras ``predict`` uses when none is passed. The analyzer and the load check of
#: ``final_model.keras`` call ``predict`` without one, so this is the batch they have always
#: run at; :func:`predict_batch_cap` leaves it alone unless the run asks for fewer.
KERAS_PREDICT_BATCH_SIZE = 32


def predict_batch_cap(eval_batch_size: Optional[int]) -> Optional[int]:
    """The ``batch_size`` for a ``predict`` that has always run at Keras' default of 32.

    Interface contract: ``eval_batch_size`` is the run's ``analysis_args.eval_batch_size`` (an
    int >= 1) or ``None`` for "no request". Pure; no TensorFlow call.

    :param eval_batch_size: the evaluation batch the run asks for, or ``None``.
    :return: ``eval_batch_size`` when it is BELOW :data:`KERAS_PREDICT_BATCH_SIZE`, else
        ``None`` (pass nothing: Keras' own 32, so a run that asks for 256 or nothing predicts
        exactly as it did, and its numbers cannot change).
    """
    # DECISION plan-2026-10-31T180033-2faf890a/D-040
    # A CAP, not a copy of the evaluation batch. Do NOT pass ``eval_batch_size`` through as it
    # is: the analyzer and the load check predicted at 32 before the key existed, a default
    # recipe's 256 would make them predict at 256 (a different batch can move the last digits
    # of a probability, and the golden would stop being exact), and 256 images of a 224 px
    # cliffordnet is the 18 GB that broke the reporting of the gate run.
    if eval_batch_size is not None and eval_batch_size < KERAS_PREDICT_BATCH_SIZE:
        return int(eval_batch_size)
    return None


# ---------------------------------------------------------------------
# The test split, once
# ---------------------------------------------------------------------

def evaluate_on_test(model: keras.Model, test_ds: tf.data.Dataset) -> Dict[str, float]:
    """Score ``model`` on the TEST dataset exactly once and return plain floats.

    Called after ``fit`` by the trainers whose selection split is a held-out slice of train
    (the protocol of ``train.common.classification_data`` and ``train.common.image_streams``),
    on the weights EarlyStopping restored. The metrics are the compiled ones (``loss`` and
    any compiled metric), keyed by name.
    """
    metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    result = {name: float(value) for name, value in metrics.items()}
    logger.info(f"Test metrics: {result}")
    return result


# ---------------------------------------------------------------------

#: Relative tolerance of the round-trip check of a model whose scores can be large (a segmentation
#: model's logits reach 220): the allowed difference is ``tolerance + LOAD_RTOL * max|expected|``.
#: Set from the measurements of D-029 (plan 98b866fb fix 9.3), taken after the cause of the 9% to 15% that D-088's
#: TF32 tolerance (0.07) was sized for was found: the in-memory AccUNet predicts WITHOUT XLA (its ``compile`` forces
#: ``jit_compile=False``) and ``load_model(compile=False)`` leaves ``jit_compile`` unset, which Keras resolves to XLA on a
#: GPU. With the reloaded model on the in-memory model's own call path (:func:`_align_predict_path`) a correct file differs
#: by exactly 0.0 on the RTX 4090 in all seven probes (AccUNet with TF32 on and off, ConvUNext, ResNet-seg, ConvNeXt-seg;
#: ``reload_call_path_evidence.json``), and by at most 2.8e-5 of the largest score in the 27 non-AccUNet cells of the iteration 1
#: gate, which were already on one path (``gate_cells`` of the same file: 26 of them 0.0, the one that is not is XLA's own
#: autotuning between two compilations of one model). 3e-4 is 10 times that worst, and 6.7 times under the smallest file with ONE kernel 1% off
#: (0.0020 of the largest score, ConvUNext): never widen it past that. The same files through the old two paths differ by up to
#: 5.5e-2 (AccUNet, TF32 on).
LOAD_RTOL = 3e-4


def tf32_active() -> bool:
    """Whether a GPU is visible and TensorFlow runs its matmuls and convolutions in TF32."""
    return bool(
        tf.config.experimental.tensor_float_32_execution_enabled()
        and tf.config.list_physical_devices("GPU")
    )


@dataclasses.dataclass(frozen=True)
class LoadCheck:
    """What :func:`check_model_loading` measured.

    :param ok: the verdict (outputs within ``tolerance`` and, when given, equal weights).
    :param max_abs_diff: largest absolute output difference, ``None`` for an output structure mismatch.
    :param scale: largest absolute value of the expected output, ``None`` for a structure mismatch.
    :param tolerance: the absolute difference allowed (``atol + rtol * scale``), ``None`` as above.
    :param relative_tolerance: the ``rtol`` used.
    :param weights_equal: ``True`` / ``False`` when reference weights were compared, else ``None``.
    :param argmax_agreement: share of positions whose ``argmax`` over the last axis is equal (not part
        of the verdict), ``None`` for a structure mismatch.
    :param tf32: whether TF32 was in effect (:func:`tf32_active`).
    :param jit_compile: the ``jit_compile`` both models predicted with when a ``reference_model`` was
        given (:func:`_align_predict_path`), else ``None``.
    """

    ok: bool
    max_abs_diff: Optional[float]
    scale: Optional[float]
    tolerance: Optional[float]
    relative_tolerance: float
    weights_equal: Optional[bool]
    argmax_agreement: Optional[float]
    tf32: bool
    jit_compile: Optional[bool] = None

    def as_record(self) -> Dict[str, Any]:
        """The check as the plain dict the summary's ``model_loading_check`` holds."""
        return dataclasses.asdict(self)


def _align_predict_path(loaded_model: keras.Model, reference_model: keras.Model) -> bool:
    """Make ``loaded_model`` predict through the same call path as ``reference_model``; return its ``jit_compile``.

    Interface contract: both are built Keras models; ``loaded_model`` was loaded with ``compile=False``.
    Mutates ``loaded_model`` (its ``jit_compile`` and ``run_eagerly``) and nothing else. Returns the
    ``jit_compile`` the reloaded model will use; it differs from the reference's only when Keras' setter
    refuses XLA for the reloaded model. Cannot fail for a model that predicts.
    """
    # DECISION plan-2026-11-03T221936-98b866fb/D-029
    # The two models of the round-trip check must predict through ONE call path. ``compile()`` is what
    # settles ``jit_compile`` (and ``run_eagerly``), and ``load_model(compile=False)`` does not run it, so
    # the reloaded model resolves ``jit_compile="auto"`` by itself: XLA on a GPU, whatever the model that
    # wrote the file does. AccUNet forces ``jit_compile=False`` in its ``compile`` (XLA cannot compile its
    # backward pass), so its in-memory predict ran without XLA and the reloaded one with it, and under TF32
    # the two differed by 1.7% to 15% of the largest score on a file whose weights were equal (measured, the
    # iteration 1 gate; copying the flag made the difference exactly 0.0). Do NOT answer that with a wider
    # output tolerance (D-088's 0.07 did, and was blind to a wrong file below 7%), and do NOT reload with
    # ``compile=True`` to get the flag (it restores the optimizer and loss, and warns for a model saved
    # before any fit).
    loaded_model.jit_compile = bool(reference_model.jit_compile)
    loaded_model.run_eagerly = bool(reference_model.run_eagerly)
    return bool(loaded_model.jit_compile)  # what the reloaded model will use (Keras' setter refuses XLA for a model that cannot)


def check_model_loading(
        model_path: str,
        test_sample: Any,
        expected_output: np.ndarray,
        custom_objects: Optional[Dict[str, Any]] = None,
        tolerance: float = 1e-5,
        batch_size: Optional[int] = None,
        relative_tolerance: float = 0.0,
        reference_model: Optional[keras.Model] = None,
) -> LoadCheck:
    """
    Reload a saved model, run it on ``test_sample`` and compare with ``expected_output``.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    test_sample : Any
        Input sample for prediction (numpy array or tensor).
    expected_output : np.ndarray
        Expected output from the original model (a dict for a model with named outputs).
    custom_objects : Optional[Dict[str, Any]]
        Custom objects dict for model loading.
    tolerance : float
        Absolute tolerance for output comparison.
    batch_size : Optional[int]
        Rows per ``predict`` pass of the reloaded model; ``None`` is Keras' default
        (:data:`KERAS_PREDICT_BATCH_SIZE`).
    relative_tolerance : float
        Added to the allowed difference as ``relative_tolerance * max|expected|``; ``0.0`` (the
        default) is the absolute check every classification run has always made.
    reference_model : Optional[keras.Model]
        The in-memory model that wrote the file and produced ``expected_output``. When given, (1) the
        reloaded weights must be EQUAL to its ``get_weights()``, array by array, whatever the outputs
        say: a stored file round-trips floats exactly, so the weights are the comparison that has no
        noise and no blind spot; and (2) the reloaded model predicts through the same call path as
        this one (:func:`_align_predict_path`), so that what a device does differently between two
        paths (XLA against none, under TF32) cannot read as a reload difference.

    Returns
    -------
    LoadCheck
        ``ok`` is True if the reloaded outputs match ``expected_output`` within the tolerance (and the
        weights are equal, if compared); False (with a warning) if they do not. A mismatch is a
        verdict the caller records.

    Raises
    ------
    Exception
        Whatever ``keras.models.load_model`` or ``predict`` raises (a missing,
        corrupt or unloadable archive, an input the reloaded model cannot
        run). That is not a verdict about the values, so it is not turned into
        ``False``: an archive that does not load is a defect to surface.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-085: no `except Exception`
    # here. A load error used to return False with a warning, so an archive
    # that does not load read like one whose values drifted. Let it raise.
    # compile=False: the check compares inference outputs; restoring the
    # optimizer of a model saved before any fit only warns about it.
    loaded_model = keras.models.load_model(
        model_path, custom_objects=custom_objects or {}, compile=False
    )

    if isinstance(test_sample, tf.Tensor):
        test_sample = test_sample.numpy()

    jit_compile: Optional[bool] = None
    if reference_model is not None:
        jit_compile = _align_predict_path(loaded_model, reference_model)
    loaded_output = loaded_model.predict(test_sample, batch_size=batch_size, verbose=0)
    tf32 = tf32_active()
    weights_equal: Optional[bool] = None
    if reference_model is not None:
        # DECISION plan-2026-10-31T180033-2faf890a/D-088
        # The weights are compared EXACTLY, beside the outputs. Do NOT drop this for an output tolerance
        # alone: a stored file round-trips floats bit for bit, so this comparison has no noise and no
        # blind spot, while an output tolerance cannot see a change under its own size (one kernel x1.01
        # is 2% of the largest score, and a TF32 device's two paths differ by more than that).
        reference_weights = reference_model.get_weights()
        reloaded = loaded_model.get_weights()
        weights_equal = len(reloaded) == len(reference_weights) and all(
            a.shape == b.shape and np.array_equal(a, b) for a, b in zip(reloaded, reference_weights)
        )

    if isinstance(expected_output, dict) or isinstance(loaded_output, dict):
        if not (isinstance(expected_output, dict) and isinstance(loaded_output, dict)):
            logger.warning(
                "Model output structure mismatch after loading "
                f"(expected={type(expected_output).__name__}, "
                f"loaded={type(loaded_output).__name__})"
            )
            return LoadCheck(False, None, None, None, relative_tolerance, weights_equal, None, tf32, jit_compile)
        if set(expected_output.keys()) != set(loaded_output.keys()):
            logger.warning(
                "Model output keys differ after loading "
                f"(expected={sorted(expected_output.keys())}, "
                f"loaded={sorted(loaded_output.keys())})"
            )
            return LoadCheck(False, None, None, None, relative_tolerance, weights_equal, None, tf32, jit_compile)
        pairs = [(np.asarray(expected_output[k]), np.asarray(loaded_output[k])) for k in expected_output]
    else:
        pairs = [(np.asarray(expected_output), np.asarray(loaded_output))]

    # The absolute check keeps numpy's own `allclose` for a single array (its default rtol of 1e-5 of the
    # reloaded value is part of what every classification run has always been judged by) and a plain
    # max difference for a dict; the relative check is `max_diff <= tolerance + rtol * scale`.
    legacy_allclose = not relative_tolerance and not isinstance(expected_output, dict)
    max_diff = 0.0
    scale = 0.0
    for expected, loaded in pairs:
        if expected.shape != loaded.shape:
            logger.warning(
                f"Model output shape differs after loading (expected={expected.shape}, loaded={loaded.shape})"
            )
            return LoadCheck(False, None, None, None, relative_tolerance, weights_equal, None, tf32, jit_compile)
        diff = np.abs(loaded - expected)
        max_diff = max(max_diff, float(np.max(diff)) if diff.size else 0.0)
        scale = max(scale, float(np.max(np.abs(expected))) if expected.size else 0.0)
    allowed = tolerance + relative_tolerance * scale
    if legacy_allclose:
        outputs_ok = bool(np.allclose(pairs[0][0], pairs[0][1], atol=tolerance))
    else:
        outputs_ok = bool(max_diff <= allowed)
    agreement = None
    first_expected, first_loaded = pairs[0]
    if first_expected.ndim >= 2 and first_expected.shape[-1] > 1:
        agreement = float(np.mean(first_expected.argmax(-1) == first_loaded.argmax(-1)))
    ok = outputs_ok and weights_equal is not False
    if ok:
        logger.info("Model loading validation passed")
    elif not outputs_ok:
        logger.warning(
            f"Model outputs differ after loading (max_diff={max_diff:.6f}, allowed {allowed:.6f}"
            f"{' = atol + rtol * scale' if relative_tolerance else ''}; TF32 {'on' if tf32 else 'off'})"
        )
    else:
        logger.warning("Model weights differ after loading: the reloaded weights are not the in-memory ones")
    return LoadCheck(ok, max_diff, scale, allowed, relative_tolerance, weights_equal, agreement, tf32, jit_compile)


def validate_model_loading(
        model_path: str,
        test_sample: Any,
        expected_output: np.ndarray,
        custom_objects: Optional[Dict[str, Any]] = None,
        tolerance: float = 1e-5,
        batch_size: Optional[int] = None,
) -> bool:
    """
    Validate that a saved model loads correctly and produces expected outputs.

    The absolute check the classification trainers have always made: :func:`check_model_loading`
    with no relative tolerance and no weights comparison, reduced to its verdict. Parameters and
    raises are those of :func:`check_model_loading`.

    Returns
    -------
    bool
        True if the reloaded outputs match ``expected_output`` within ``tolerance``; False (with a
        warning) if they do not.
    """
    return check_model_loading(
        model_path, test_sample, expected_output, custom_objects, tolerance, batch_size
    ).ok


# ---------------------------------------------------------------------

def convert_keras_history_to_training_history(
        keras_history: keras.callbacks.History,
) -> TrainingHistory:
    """Convert Keras training history to visualization framework TrainingHistory."""
    history_dict = keras_history.history
    epochs = list(range(len(history_dict['loss'])))

    train_metrics = {}
    val_metrics = {}

    for key, values in history_dict.items():
        if key.startswith('val_') and key != 'val_loss':
            val_metrics[key.replace('val_', '')] = values
        elif not key.startswith('val_') and key != 'loss':
            train_metrics[key] = values

    return TrainingHistory(
        epochs=epochs,
        train_loss=history_dict['loss'],
        val_loss=history_dict.get('val_loss', []),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )


# ---------------------------------------------------------------------

def _as_prediction_pair(y_true: Any, y_pred: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten two prediction arrays to ``(N,)`` and assert they agree in length.

    Every ``ClassificationResults`` / ``RegressionResults`` consumer indexes
    ``y_true[i]`` against ``y_pred[i]``, so a length mismatch is a silent
    truncation once a boolean mask is applied to one of them. Assert here,
    where the message can name both shapes.
    """
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length; got "
            f"{yt.shape[0]} and {yp.shape[0]}."
        )
    if yt.shape[0] == 0:
        raise ValueError("y_true and y_pred are both empty; nothing to visualize.")
    return yt, yp


def create_classification_results(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        model_name: Optional[str] = None,
) -> ClassificationResults:
    """Create ClassificationResults object for visualization.

    Args:
        y_true: True labels, any shape; flattened to ``(N,)``.
        y_pred: Predicted labels, flattened to ``(N,)``. Must match
            ``y_true`` in length.
        y_prob: Optional class probabilities. **Omitting it is a supported
            mode, not a degraded one** — several classification plugins
            degrade gracefully without it (``PerClassAnalysis`` renders a
            ``'No probability data'`` panel) but ``ROCPRCurves`` does not:
            its per-model loop is ``if results.y_prob is None: continue``,
            so it builds an EMPTY figure and saves a blank PNG without
            raising. Never register ``roc_pr_curves`` for a model built
            without probabilities. Pinned by
            ``tests/test_train/test_common_evaluation_results.py``.
        class_names: One name per class.
        model_name: Display name used in figure titles.
    """
    yt, yp = _as_prediction_pair(y_true, y_pred)
    return ClassificationResults(
        y_true=yt,
        y_pred=yp,
        y_prob=y_prob,
        class_names=class_names,
        model_name=model_name,
    )


def create_regression_results(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
) -> RegressionResults:
    """Create ``RegressionResults``, the input every regression plugin requires.

    ``dl_techniques.visualization.regression`` ships six plugins —
    ``prediction_error``, ``residuals_plot``, ``residual_distribution``,
    ``qq_plot``, ``regression_evaluation_dashboard`` and the
    ``MultiModelRegression`` comparison dashboard — and every one of them
    type-checks its input with ``isinstance(data, RegressionResults)``.
    Nothing in the tree constructed this container, which is why all six
    were unreachable; this is that constructor.

    Args:
        y_true: Ground-truth targets, flattened to ``(N,)``.
        y_pred: Predicted targets, flattened to ``(N,)``. Must match
            ``y_true`` in length.
        model_name: Display name used in figure titles.
        feature_names: Optional per-feature names.

    Returns:
        A ``RegressionResults`` carrying the flattened pair.
    """
    yt, yp = _as_prediction_pair(y_true, y_pred)
    return RegressionResults(
        y_true=yt,
        y_pred=yp,
        model_name=model_name,
        feature_names=feature_names,
    )


def create_multi_model_regression(
        results: Dict[str, Any],
        dataset_name: Optional[str] = None,
) -> MultiModelRegression:
    """Bundle per-model ``RegressionResults`` for the comparison dashboard.

    Args:
        results: ``{model_name: RegressionResults}``, or ``{model_name:
            (y_true, y_pred)}`` which is passed through
            :func:`create_regression_results`. Mixing the two is allowed.
        dataset_name: Optional dataset label for the figure title.
    """
    built = {
        name: (
            value
            if isinstance(value, RegressionResults)
            else create_regression_results(value[0], value[1], model_name=name)
        )
        for name, value in results.items()
    }
    if not built:
        raise ValueError("results must contain at least one model.")
    return MultiModelRegression(results=built, dataset_name=dataset_name)


def create_timeseries_results(
        all_inputs: np.ndarray,
        all_true_forecasts: np.ndarray,
        all_predicted_forecasts: Optional[np.ndarray] = None,
        all_predicted_quantiles: Optional[np.ndarray] = None,
        model_name: Optional[str] = None,
        quantile_levels: Optional[List[float]] = None,
) -> TimeSeriesEvaluationResults:
    """Create ``TimeSeriesEvaluationResults``, the input ``ForecastVisualization`` requires.

    ``ForecastVisualization`` is the only consumer of this container and it
    type-checks with ``isinstance(data, TimeSeriesEvaluationResults)``, so
    without a constructor it was unreachable from the tree as well.

    Shape contract, taken from the container's own annotations:
    ``all_inputs`` is ``(num_samples, input_length)`` and
    ``all_true_forecasts`` is ``(num_samples, forecast_length)``. The two
    must agree on ``num_samples`` — the plugin indexes forecasts by sample
    index — and the forecast axis is kept as-is rather than squeezed,
    because a ``forecast_length`` of 1 must stay two-dimensional.

    A **1-D array on either axis describes ONE window** and is promoted to a
    batch of one, so ``(10,)`` inputs with ``(3,)`` forecasts is the single
    window that predicts three steps ahead. The alternative reading —
    ``(3,)`` meaning three samples each one step ahead — is deliberately NOT
    taken: it would make the promotion rule differ between the two axes, so
    the same call site would mean different things depending on which array
    it touched. Pass ``(N, 1)`` explicitly for N single-step samples.

    Args:
        all_inputs: Windowed history, ``(num_samples, input_length)``.
        all_true_forecasts: Ground-truth futures,
            ``(num_samples, forecast_length)``.
        all_predicted_forecasts: Optional point forecasts, same shape as
            ``all_true_forecasts``.
        all_predicted_quantiles: Optional quantile forecasts,
            ``(num_samples, forecast_length, len(quantile_levels))``.
        model_name: Display name used in figure titles.
        quantile_levels: The levels ``all_predicted_quantiles`` was built
            at. Supplying quantiles without levels is an error rather than
            a silent no-band, because the plugin needs the levels to draw
            the legend.
    """
    inputs = np.asarray(all_inputs)
    true_forecasts = np.asarray(all_true_forecasts)

    # A 1-D array is ONE window on either axis — see the docstring's shape contract.
    if inputs.ndim == 1:
        inputs = inputs[None, :]
    if true_forecasts.ndim == 1:
        true_forecasts = true_forecasts[None, :]

    if inputs.ndim != 2 or true_forecasts.ndim != 2:
        raise ValueError(
            f"all_inputs and all_true_forecasts must be 2-D after promotion; got "
            f"{inputs.ndim}-D and {true_forecasts.ndim}-D."
        )
    if inputs.shape[0] != true_forecasts.shape[0]:
        raise ValueError(
            f"all_inputs and all_true_forecasts must agree on num_samples; got "
            f"{inputs.shape[0]} and {true_forecasts.shape[0]}."
        )
    if inputs.shape[0] == 0:
        raise ValueError("all_inputs is empty; nothing to visualize.")

    predicted = None
    if all_predicted_forecasts is not None:
        predicted = np.asarray(all_predicted_forecasts)
        if predicted.ndim == 1:
            predicted = predicted[None, :]
        if predicted.shape != true_forecasts.shape:
            raise ValueError(
                f"all_predicted_forecasts must match all_true_forecasts "
                f"({true_forecasts.shape}); got {predicted.shape}."
            )

    quantiles = None
    if all_predicted_quantiles is not None:
        quantiles = np.asarray(all_predicted_quantiles)
        if not quantile_levels:
            raise ValueError(
                "quantile_levels is required when all_predicted_quantiles is "
                "given; without them the uncertainty bands cannot be labelled."
            )
        if quantiles.shape[:2] != true_forecasts.shape:
            raise ValueError(
                f"all_predicted_quantiles must lead with all_true_forecasts' "
                f"shape ({true_forecasts.shape}); got {quantiles.shape}."
            )
        if quantiles.shape[2] != len(quantile_levels):
            raise ValueError(
                f"all_predicted_quantiles has {quantiles.shape[2]} quantile "
                f"levels but {len(quantile_levels)} were declared."
            )

    return TimeSeriesEvaluationResults(
        all_inputs=inputs,
        all_true_forecasts=true_forecasts,
        all_predicted_forecasts=predicted,
        all_predicted_quantiles=quantiles,
        model_name=model_name,
        quantile_levels=quantile_levels,
    )


# ---------------------------------------------------------------------

def generate_training_curves(
        history,
        results_dir: str,
        filename: str = "training_curves",
) -> None:
    """
    Generate training curve plots using the visualization framework.

    Accepts either a Keras History object or a raw dict of metric lists
    (as used by in-training callbacks that track metrics manually).

    Parameters
    ----------
    history : keras.callbacks.History or Dict[str, List[float]]
        Keras training history from model.fit(), or a dict mapping
        metric names to lists of per-epoch values. Dict keys follow
        Keras convention: 'loss', 'val_loss', 'metric_name',
        'val_metric_name'.
    results_dir : str
        Directory to save the plot.
    filename : str
        Filename for the saved plot (without extension).
    """
    if isinstance(history, dict):
        history_dict = history
    else:
        history_dict = history.history

    epochs = list(range(len(history_dict['loss'])))
    train_metrics = {}
    val_metrics = {}
    for key, values in history_dict.items():
        if key == 'loss' or key == 'val_loss':
            continue
        if key.startswith('val_'):
            val_metrics[key[4:]] = values
        else:
            train_metrics[key] = values

    training_history = TrainingHistory(
        epochs=epochs,
        train_loss=history_dict['loss'],
        val_loss=history_dict.get('val_loss'),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )

    plot_config = PlotConfig(fig_size=(14, 10), save_dpi=150)
    viz_manager = VisualizationManager(
        experiment_name="",
        output_dir=results_dir,
        config=plot_config,
        timestamp="",
    )
    viz_manager.register_plugin(TrainingCurvesVisualization(
        config=plot_config,
        context=viz_manager.context,
    ))
    viz_manager.visualize(
        data=training_history,
        plugin_name="training_curves",
        show=False,
        filename=filename,
    )


# ---------------------------------------------------------------------

def generate_comprehensive_visualizations(
        viz_manager: VisualizationManager,
        training_history: TrainingHistory,
        classification_results: ClassificationResults,
        model: keras.Model,
        show_plots: bool = False,
        max_roc_classes: int = 20,
) -> None:
    """
    Generate comprehensive visualizations using the VisualizationManager.

    Parameters
    ----------
    viz_manager : VisualizationManager
        Configured visualization manager.
    training_history : TrainingHistory
        Training history data.
    classification_results : ClassificationResults
        Classification results with predictions.
    model : keras.Model
        The trained model.
    show_plots : bool
        Whether to show plots interactively.
    max_roc_classes : int
        Maximum number of classes to generate ROC/PR curves for.
    """
    logger.info("Generating comprehensive visualizations...")

    logger.info("  - Training curves")
    viz_manager.visualize(
        data=training_history,
        plugin_name="training_curves",
        show=show_plots,
        filename="training_curves",
    )

    logger.info("  - Confusion matrix")
    viz_manager.visualize(
        data=classification_results,
        plugin_name="confusion_matrix",
        show=show_plots,
        filename="confusion_matrix",
    )

    if len(classification_results.class_names) <= max_roc_classes:
        logger.info("  - ROC and PR curves")
        viz_manager.visualize(
            data=classification_results,
            plugin_name="roc_pr_curves",
            show=show_plots,
            filename="roc_pr_curves",
        )

    logger.info("  - Network architecture")
    viz_manager.visualize(
        data=model,
        plugin_name="network_architecture",
        show=show_plots,
        filename="architecture",
    )

    logger.info("Visualizations generated successfully")


# ---------------------------------------------------------------------

#: The activation budget, in MB, of the information-flow analysis of a per-pixel model.
#: Measured (plan-2026-10-31T180033-2faf890a step 9) on 200 images of 128 px, 21 classes:
#: the default 2048 MB peaks at 12.3 GB and 103 s, 256 MB at 3.6 GB and 30 s.
PER_PIXEL_MEMORY_LIMIT_MB = 256


def stock_analysis_config(random_state: Optional[int] = None, **overrides: Any) -> AnalysisConfig:
    """The five-analysis configuration every trainer's end-of-run analysis starts from.

    Args:
        random_state: Seed of the analyzer's generator, or ``None`` (unseeded, what every
            legacy trainer uses). The API passes the run's seed so that the bootstrap
            ``pl_pvalue`` of the spectral analysis is reproducible.
        **overrides: Further :class:`AnalysisConfig` fields (a per-pixel model sets its
            budget and void label here).

    Returns:
        An :class:`AnalysisConfig` with weights, calibration, information flow, training
        dynamics and spectral on.
    """
    return AnalysisConfig(
        analyze_weights=True,
        analyze_calibration=True,
        analyze_information_flow=True,
        analyze_training_dynamics=True,
        analyze_spectral=True,
        random_state=random_state,
        **overrides,
    )


def first_sample(test_data: Tuple[Any, Any]) -> Any:
    """One input of ``test_data`` as a batch of one, or ``None`` when it cannot be taken cheaply.

    Args:
        test_data: ``(x, y)`` as :func:`run_model_analysis` takes it: ``x`` an array, a dict
            of arrays or a ``tf.data.Dataset`` of ``(images, labels)`` batches.

    Returns:
        ``x[:1]`` (the dict of such slices), the first image of the first batch of a
        dataset, or ``None`` for anything else.
    """
    x = test_data[0]
    if isinstance(x, tf.data.Dataset):
        for batch_images, _ in x.take(1):
            return batch_images.numpy()[:1]
        return None
    if isinstance(x, dict):
        return {k: v[:1] for k, v in x.items()}
    return x[:1] if hasattr(x, "__getitem__") and len(x) else None


# DECISION plan-2026-10-31T180033-2faf890a/D-027
# The API passes the run's seed as `random_state`, so the spectral bootstrap `pl_pvalue` (and the
# `spectral_recommendations` strings derived from it) repeat between two runs of one seed. A
# seeded classifier therefore gets an explicit config instead of `None`. Do NOT seed the legacy
# call (`run_model_analysis(config=None)`): legacy artifacts stay what they were, unseeded.
def analysis_config_for(
        model: keras.Model,
        ignore_label: Optional[int] = None,
        sample: Optional[Any] = None,
        random_state: Optional[int] = None,
        prediction_batch_size: Optional[int] = None,
) -> Optional[AnalysisConfig]:
    """The analyzer configuration a model needs beyond the default, or ``None``.

    Args:
        model: The model about to be analyzed.
        ignore_label: The label value that marks a void pixel, or ``None``. Read only for
            a per-pixel model.
        sample: One input (see :func:`first_sample`), or ``None``. Read only for a model
            with no static output shape, a subclassed one, whose output rank is then taken
            from one real call.
        random_state: The run's seed, or ``None`` (the legacy, unseeded configuration).
        prediction_batch_size: ``AnalysisConfig.prediction_batch_size`` (see
            :func:`predict_batch_cap`), or ``None`` for the analyzer's own batch. A value
            needs a configuration to live in, so it forces the stock one just as a seed does.
            Read on the rank-2 path only: a per-pixel model predicts in the analyzer's own
            ``PER_PIXEL_BATCH_SIZE`` slices and this value is not passed there.

    Returns:
        With ``random_state`` ``None``: ``None`` for a model that is not per-pixel (output
        rank below 4), which :func:`run_model_analysis` takes as "the default
        configuration", so a classifier is analyzed exactly as before. With a
        ``random_state``: the stock configuration carrying it, since ``None`` has nowhere to
        hold a seed. For a per-pixel model, all five analyses with ``memory_limit_mb`` set
        to :data:`PER_PIXEL_MEMORY_LIMIT_MB`, ``ignore_label`` passed through (so a void
        pixel is dropped before calibration) and the seed.
    """
    if not ModelAnalyzer.outputs_per_pixel(model, sample=sample):
        if random_state is None and prediction_batch_size is None:
            return None
        return stock_analysis_config(random_state, prediction_batch_size=prediction_batch_size)
    return stock_analysis_config(
        random_state, memory_limit_mb=PER_PIXEL_MEMORY_LIMIT_MB, ignore_label=ignore_label)


def run_model_analysis(
        model: keras.Model,
        test_data: Tuple[Any, np.ndarray],
        training_history: keras.callbacks.History,
        model_name: str,
        results_dir: str,
        config: Optional[AnalysisConfig] = None,
) -> Any:
    """
    Run comprehensive model analysis using ModelAnalyzer.

    Parameters
    ----------
    model : keras.Model
        Trained model to analyze.
    test_data : Tuple[Any, np.ndarray]
        Test data as (x_test, y_test). x_test can be numpy array or tf.data.Dataset.
    training_history : keras.callbacks.History
        Keras training history.
    model_name : str
        Name identifier for the model.
    results_dir : str
        Directory to save analysis results.
    config : Optional[AnalysisConfig]
        Analysis configuration. Uses sensible defaults if None.

    Returns
    -------
    Analysis results or None if analysis fails.
    """
    logger.info("Running model analysis...")

    if config is None:
        config = stock_analysis_config()

    try:
        # Extract numpy arrays from tf.data.Dataset if needed
        if isinstance(test_data[0], tf.data.Dataset):
            logger.info("Extracting subset from tf.data.Dataset for analysis...")
            for batch_images, batch_labels in test_data[0].take(1):
                x_test_subset = batch_images.numpy()
                y_test_subset = batch_labels.numpy()
                break
        else:
            x_test_subset = test_data[0][:1000]
            y_test_subset = test_data[1][:1000]

        analyzer = ModelAnalyzer(
            models={model_name: model},
            config=config,
            output_dir=os.path.join(results_dir, "model_analysis"),
            training_history={model_name: training_history.history},
        )

        data = DataInput(x_data=x_test_subset, y_data=y_test_subset)
        analysis_results = analyzer.analyze(data=data)

        logger.info("Model analysis completed successfully")
        return analysis_results

    except Exception as e:
        logger.warning(f"Model analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------

def setup_visualization_manager(
        experiment_name: str,
        results_dir: str,
) -> VisualizationManager:
    """Set up and configure a publication-style visualization manager.

    Creates a ``visualizations`` subdirectory under ``results_dir`` and returns
    a :class:`VisualizationManager` pre-registered with the five standard
    vision-classification templates (training curves, confusion matrix, network
    architecture, model-comparison bars, ROC/PR curves).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment, used to namespace the visualization manager.
    results_dir : str
        Base results directory. Plots are written under
        ``{results_dir}/visualizations``.

    Returns
    -------
    VisualizationManager
        Configured manager with the five standard templates registered.
    """
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    config = PlotConfig(
        style=PlotStyle.PUBLICATION,
        color_scheme=ColorScheme(
            primary="#2E86AB",
            secondary="#A23B72",
            success="#06D6A0",
            warning="#FFD166"
        ),
        title_fontsize=14,
        label_fontsize=12,
        save_format="png",
        dpi=300,
        fig_size=(12, 8)
    )

    viz_manager = VisualizationManager(
        experiment_name=experiment_name,
        output_dir=viz_dir,
        config=config
    )

    viz_manager.register_template("training_curves", TrainingCurvesVisualization)
    viz_manager.register_template("confusion_matrix", ConfusionMatrixVisualization)
    viz_manager.register_template("network_architecture", NetworkArchitectureVisualization)
    viz_manager.register_template("model_comparison_bars", ModelComparisonBarChart)
    viz_manager.register_template("roc_pr_curves", ROCPRCurves)

    logger.info(f"Visualization manager setup complete. Plots will be saved to: {viz_dir}")
    return viz_manager
