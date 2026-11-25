"""The bars of the language tasks (an analytic unigram, a majority share and two strict float32 comparisons) and their reporting.

The bar of ``language_model_pretraining`` is cut from the ``val_loss`` of a CONSTANT model that predicts the unigram
distribution of the selection targets, and the bar of ``text_classification`` from the share of the majority class of the
selection split plus a margin of three binomial standard errors of it. Neither needs a model. A constant Keras model over a
50,261-wide vocabulary at batch 8 and 255 positions is 0.4 GB of logits a forward (3 GB at 100,277), where the unigram is a count
of the targets: O(vocabulary) memory. The module scope imports numpy only (the check against a compiled constant model is a test);
the reporting imports ``train.common`` inside.

Interface contract
------------------
``unigram_val_loss(label_array, vocab_size)``
    :param label_array: the integer target ids of every selection window, any shape; ``vocab_size``: the model's vocabulary.
    :return: ``-sum p log p`` in nats, ``p`` the share of each id among the targets: the ``val_loss`` of a model whose logits
        are ``log p`` at every position under the stock sparse cross-entropy (an in-sample unigram, the strictest constant
        model), computed from a chunked count and never from a ``(rows, vocab_size)`` array.
    :raises ValueError: no target, a non-integer array, a ``vocab_size`` below 1, an id outside ``[0, vocab_size)``.

``majority_share(labels)``
    :return: the share of the most frequent class among ``labels`` (non-negative integer class ids), a float64 ratio whose
        float32 is what Keras' accuracy divides for a model that answers the majority class. :raises ValueError: no label, a
        negative or non-integer one.

``accuracy_margin(share, n_rows)``
    :return: how far above the majority ``share`` of ``n_rows`` selection rows a classifier's bar sits: :data:`BAR_STANDARD_ERRORS`
        binomial standard errors of the share, ``3 * sqrt(share * (1 - share) / n_rows)``, capped at half the way from ``share`` to 1.0
        so that ``share + margin`` stays attainable. On IMDB's 2,498 selection rows it is 0.0300, a bar of 0.5328, which a guesser
        clears in 0.44 percent of 8-epoch runs where the bare share (0.5028) let 97.9 percent through (exact binomial).

``strictly_below(reached, threshold, n_batches)``
    :param reached: a ``val_loss`` (Keras' float32); ``threshold``: the unigram (float64) or the epoch-zero loss; ``n_batches``:
        the batches of the selection pass.
    :return: ``train.common.forecast_run.beats_bar``: whether ``float32(reached)`` is below ``float32(threshold)`` by MORE than
        ``tie_band(n_batches)`` float32 steps (16 up to 16 batches, 96 at 500, 208 at 2,500). A model that merely equals the
        unigram, which Keras scores up to 14 float32 steps either side of the analytic number (measured, 1,250 batches), does not
        meet the bar, and neither does a NaN.

``strictly_above(reached, threshold)``
    :return: whether ``float32(reached)`` is strictly above ``float32(threshold)``. No band: Keras' accuracy of a model that answers
        the majority class is ``float32(k) / float32(n)``, the float32 of the share to the last bit (every count below 2**24).

``report_run(config, data, fit, run_analysis, start)``
    The reporting both text tasks share, the body of their ``evaluate_and_report`` up to the summary's task-specific keys: the best
    checkpoint reloaded and scored on the test split, the data-free analysis (``run_analysis``, the task's bound hook), the last
    epoch's weights scored, saved and reloaded. :param config: the resolved task config (its ``checkpoint_args``, ``analysis_args``
    and ``metric_args``); ``data``: the :class:`~train.common.text_data.TextBatches` (the first :data:`LOAD_CHECK_ROWS` selection rows
    are the sample of the analysis and of the reload check); ``fit``: the ``FitState``; ``start``: the 0-based epoch the selection
    starts at. :return: ``(fields, hist, best)``: the keywords of ``Evaluation`` this reporting fills (everything but ``loaded``,
    ``extras`` and ``failed_checks``), the history as plain floats and the 1-based best epoch of ``metric_args.watch_metric``.
"""

import math
from typing import Any, Callable, Dict, Tuple

import numpy as np

#: Selection rows the reload check of ``final_model.keras`` predicts and the analysis reads: each row of the language model is
#: ``seq_len x vocab_size`` floats of logits (205 MB for four rows at the 50,261-wide vocabulary and 256 tokens), twice.
LOAD_CHECK_ROWS = 4

#: Targets counted per pass: a count never holds more than this many ids as one ``intp`` array (8 MB).
CHUNK_ELEMENTS = 1 << 20

#: Binomial standard errors of the majority share that a classifier's bar sits above it (:func:`accuracy_margin`, D-076).
BAR_STANDARD_ERRORS = 3.0


def unigram_val_loss(label_array: np.ndarray, vocab_size: int) -> float:
    """The entropy of the target ids in nats; see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-006
    # The unigram is COUNTED, not evaluated: do NOT compile a constant Keras model at the real vocabulary to read its loss (the
    # logits tensor is 0.4 GB at 50,261 ids and 3 GB at 100,277, a batch at a time), and do NOT count in a ``(rows, vocab)``
    # array. A test holds this number equal to the `val_loss` of a compiled constant model on an ascii vocabulary, which is what
    # proves the closed form is the loss Keras computes. The count is in-sample (the selection targets themselves), the
    # strictest constant model, as the BEiT bar's.
    ids = np.asarray(label_array)
    if ids.dtype.kind not in "iu":
        raise ValueError(f"the targets must be integer ids, got dtype {ids.dtype}")
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be at least 1, got {vocab_size}")
    if ids.size == 0:
        raise ValueError("no target ids: there is no unigram to count")
    flat = ids.reshape(-1)
    counts = np.zeros(vocab_size, np.int64)
    for start in range(0, flat.size, CHUNK_ELEMENTS):
        part = flat[start:start + CHUNK_ELEMENTS]
        low, high = int(part.min()), int(part.max())
        if low < 0 or high >= vocab_size:
            raise ValueError(
                f"target id {low if low < 0 else high} is outside [0, {vocab_size}): the tokenizer's vocabulary is larger than "
                "the vocab_size given")
        counts += np.bincount(part, minlength=vocab_size)
    share = counts[counts > 0] / flat.size
    return float(-(share * np.log(share)).sum())


def majority_share(labels: np.ndarray) -> float:
    """The share of the most frequent class; see the module contract."""
    ids = np.asarray(labels)
    if ids.size == 0:
        raise ValueError("no labels: there is no majority class")
    if ids.dtype.kind not in "iu" or ids.min() < 0:
        raise ValueError(f"labels must be non-negative integer class ids, got dtype {ids.dtype}")
    return float(np.bincount(ids.reshape(-1)).max() / ids.size)


def accuracy_margin(share: float, n_rows: int) -> float:
    """The margin of a classifier's bar over the majority ``share`` of ``n_rows`` rows; see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-076
    # The bar is the share PLUS this margin, not the bare share: on IMDB's 2,498 selection rows a coin flip clears 0.5028 in 98.3
    # percent of 8-epoch runs (and a constant model with five answers flipped in about half of its epochs), so "above the share"
    # said nothing about learning. Do NOT drop the margin, do NOT make it a key (a knob that lowers it re-opens the hole) and do NOT
    # lift the cap (a bar of 1.0 or more can never be strictly beaten, and a tiny selection split would then fail every run).
    return min(BAR_STANDARD_ERRORS * math.sqrt(share * (1.0 - share) / n_rows), (1.0 - share) / 2.0)


def strictly_below(reached: float, threshold: float, n_batches: int) -> bool:
    """Whether ``reached`` is below ``threshold`` by more than the tie band of ``n_batches`` batches; see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-006
    # DECISION plan-2026-11-24T092348-7806a290/D-020
    # STRICTLY below, in float32, and by MORE than the band of the one loss bar the repo has (`forecast_run.beats_bar`, the depth
    # task's too). Do NOT write `<=` or a bare `<`: the threshold here is a float64 number counted in numpy, the loss a float32
    # mean Keras accumulates batch by batch, and a compiled constant model measured 7 steps below the analytic unigram in 9 of 24
    # cases (14 steps at 1,250 batches), so a bare compare calls a model that is exactly the unigram a success by rounding. Do
    # NOT narrow the band to "be strict": `scratch_step3/ulp_gap*.txt` is the measurement, and the band is 10 times the worst gap.
    from train.common.forecast_run import beats_bar

    return beats_bar(reached, threshold, n_batches)


def strictly_above(reached: float, threshold: float) -> bool:
    """Whether ``reached`` is strictly above ``threshold`` in float32 (an accuracy bar); see the module contract."""
    # DECISION plan-2026-11-24T092348-7806a290/D-006
    # No band, unlike the loss: the share is a float64 ratio of two counts and Keras' accuracy the float32 ratio of the same
    # counts, equal to the last bit for every count below 2**24 (a test sweeps them), so a model that answers the majority class
    # scores exactly the bar and `>` calls it a miss. Do NOT write `>=` (that calls it a success).
    return bool(np.float32(reached) > np.float32(threshold))


def report_run(config: Any, data: Any, fit: Any, run_analysis: Callable[..., Any], start: int) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """The reporting both text tasks share; see the module contract."""
    from types import SimpleNamespace

    from train.common import classification_run, run_summary

    checkpoint_cfg, analysis_cfg, metric_cfg = config.checkpoint_args, config.analysis_args, config.metric_args
    test_best, load_error, max_diff = None, None, None
    if checkpoint_cfg.save_best:
        # The in-memory model holds the BEST weights here (EarlyStopping restored them). The test rows are fixed, so the reloaded
        # file reproduces them exactly.
        test_best, load_error, max_diff = classification_run.check_best_checkpoint(
            fit.model, fit.run_dir, data.test_ds, evaluate=classification_run.evaluate_dict, split_label="Test results")
    # A few fixed selection rows: the sample of the analysis and of the reload check of `final_model.keras`.
    x, y = data.split.select
    sample_x = x[:LOAD_CHECK_ROWS].astype(np.float32)
    # The best weights are analysed, then the last epoch's replace them (the sequence the denoising task shares).
    analysis, test_final, load_check, load_detail = run_summary.report_data_free_run(
        fit.model, SimpleNamespace(x_fig=sample_x, y_fig=y[:LOAD_CHECK_ROWS]), fit.history, fit.run_dir, fit.vis_dir,
        fit.experiment_name, run_analysis=run_analysis, model_analysis=analysis_cfg.model_analysis,
        eval_batch_size=analysis_cfg.eval_batch_size, last_weights=fit.last_weights, data=data, epochs_run=fit.epochs_run,
        save_last=checkpoint_cfg.save_last, load_sample=sample_x, predict_batch_size=data.batch_size)
    visualizations: Dict[str, Any] = {"files": [], "failed": [], "skipped": not analysis_cfg.figures}
    if analysis_cfg.figures and fit.vis_dir.is_dir():
        visualizations["files"] = sorted(path.name for path in fit.vis_dir.iterdir() if path.is_file())
    hist = {key: [float(v) for v in values] for key, values in fit.history.history.items()}
    best = run_summary.best_epoch(hist, metric_cfg.watch_metric, metric_cfg.watch_mode, first_epoch=start)
    fields = dict(test_metrics_best=test_best, best_load_error=load_error, checkpoint_max_diff=max_diff, conformal=None,
                  visualizations=visualizations, analysis=analysis, test_metrics_final=test_final, load_check=load_check,
                  load_detail=load_detail)
    return fields, hist, best
