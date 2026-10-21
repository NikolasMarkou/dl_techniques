"""Every broad handler under ``src/train`` outside a ``main`` is classified.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 4.2 (D-180, D-183).
Step 4's census (``test_a_failed_trainer_run_does_not_exit_zero.py``, whose AST
helpers this file reuses) covers the broad handlers lexically in a ``main``. This
one covers every other broad ``except`` (bare, ``Exception``, ``BaseException``)
under ``src/train``. Each one must be one of:

- **raises** or **exits** non-zero (Step 4's tests);
- **records**: it records the failure and its own function fails later (Step 4's
  ``_fails_later``, the D-095 sweep shape);
- **optional output** (``_OPTIONAL_OUTPUT``): the guarded block only produces an
  optional artifact (a figure, a log line, a best-effort extra). Its shape is
  checked: the handler logs the exception (it names the bound exception in a
  ``logger`` call, passes ``exc_info``, or calls ``logger.exception``), and the
  ``try`` body calls nothing in ``FORBIDDEN_CALLS`` (training, evaluation, model
  saves and loads, the ``train.common.run_io`` writers);
- a named **exemption** (``_EXEMPT``) with a one-line reason: a fallback value, a
  cleanup, an optional dependency, or a failure recorded where the run's output or
  its ``main`` reads it.

Anything else is a swallow: a training, data, evaluation or save failure that the
run hides. At f9541b39b the census found 150 such handlers in 65 files (Step 4's
count of 154 in 66 used a different scan); 26 were swallows, fixed in this step,
and 11 more were made to fail the run where their recorded failure is read.

An entry names a handler by its enclosing function, a text in the handler (its
string constants or its source) and a text in the ``try`` body.
"""

import ast
import warnings

from .test_a_failed_trainer_run_does_not_exit_zero import (
    _SRC,
    _exits_non_zero,
    _fails_later,
    _is_broad,
    _main_broad_handlers,
    _walk_no_defs,
)
from .test_a_history_write_follows_the_model_save import RUN_IO_WRITERS

#: Calls an optional-output ``try`` may not make: a handler around any of them
#: could hide a training, evaluation or save failure.
FORBIDDEN_CALLS = frozenset({
    "fit", "evaluate", "train_on_batch", "test_on_batch",
    "save", "save_weights", "export", "load_model",
}) | RUN_IO_WRITERS


#: Optional output: path -> [(function, text in the handler, text in the try, what it makes)].
_OPTIONAL_OUTPUT = {
    'train/accunet/train_accunet.py': [
        ('on_epoch_end', 'SegmentationGridCallback: predict failed', 'preds = self.model.predict(self.val_x, verbos',
         'segmentation grid figure'),
    ],
    'train/bfunet/common.py': [
        ('build_dashboard_from_dir', 'Could not reconstruct sigma_max:', 'cur = NoiseSigmaCurriculumCallback(sigma_max_',
         "dashboard rebuilt from a finished run's CSV"),
        ('build_dashboard_from_dir', 'Could not reconstruct learning rate:', "spe = cfg.get('steps_per_epoch')",
         "dashboard rebuilt from a finished run's CSV"),
        ('_homogeneity_probe', 'Homogeneity probe skipped (could not syn', 'in_shape = model.input_shape',
         'homogeneity probe log line'),
        ('_homogeneity_probe', 'Homogeneity probe skipped (forward pass ', 'f_ax = np.asarray(_first_output(model(alpha * x',
         'homogeneity probe log line'),
        ('_dc_preservation_probe', 'DC probe skipped (could not synthesize i', 'in_shape = model.input_shape',
         'DC-preservation probe log line'),
        ('_dc_preservation_probe', 'DC probe skipped (forward pass failed):', 'for c in (0.1, 0.25, 0.5, 0.75, 0.9):',
         'DC-preservation probe log line'),
        ('_attempt', 'Visualization', 'draw()',
         'a dashboard figure, recorded in failed'),
    ],
    'train/bfunet/variance_probe.py': [
        ('_plot', 'matplotlib unavailable, skipping plot:', 'import matplotlib',
         "the probe's figure"),
    ],
    'train/blt/train_blt.py': [
        ('evaluate_model', "Generation failed for '", 'generated = self.blt_model.generate(prompt=pr',
         'a logged generation sample'),
    ],
    'train/capsnet/train_capsnet.py': [
        ('on_epoch_end', 'Reconstruction failed epoch', 'predictions = self.model.predict(self.sample_',
         'reconstruction figure'),
    ],
    'train/cliffordnet/train_clip.py': [
        ('_run_probe', 'Retrieval probe failed at step', 'metrics = _compute_retrieval_metrics(self.cli',
         'periodic retrieval probe written to probes.jsonl'),
    ],
    'train/common/callbacks.py': [
        ('_plot', 'EpochMetricsPlotCallback: failed to crea', 'history = self._build_history()',
         'per-epoch metric figures and latest_metrics.json'),
    ],
    'train/common/classification_viz.py': [
        ('_draw', 'Dashboard render failed at epoch', 'render_training_dashboard(self.history, self.',
         'dashboard figure'),
        ('on_epoch_end', 'Dashboard accumulation failed at epoch', 'self.epoch_times.append(time.perf_counter() -',
         'dashboard history'),
    ],
    'train/common/evaluation.py': [
        ('run_model_analysis', 'Model analysis failed:', 'if isinstance(test_data[0], tf.data.Dataset):',
         'the analyzer report'),
    ],
    'train/common/gpu.py': [
        ('log_gpu_peak_memory', 'Could not read GPU memory info: %s', 'info = tf.config.experimental.get_memory_info',
         'GPU memory log line'),
    ],
    'train/common/image_text.py': [
        ('load_cc3m_local_split', ': tokenization cache write failed (', 'np.savez(cache_npz, token_ids=token_ids, src_',
         'the tokenization cache file'),
    ],
    'train/common/keypoint_viz.py': [
        ('on_epoch_end', 'SuperPointVizCallback: failed at epoch', 'from dl_techniques.utils.keypoint_extraction ',
         'keypoint figures'),
        ('on_epoch_end', 'LightGlueVizCallback: failed at epoch', 'batch = {k: keras.ops.convert_to_tensor(v[:se',
         'match figures'),
    ],
    'train/common/run_summary.py': [
        ('run_data_free_analysis', 'Analyzer raised', 'with threadpool_limits(limits=1):',
         'the analyzer report, recorded as status error in the summary'),
        ('_attempt', 'Visualization', 'result = fn()',
         'a figure, recorded in failed'),
        ('write_classification_figures', 'classification_report.json', "with open(vis_dir / 'classification_report.js",
         'classification_report.json, recorded in failed'),
    ],
    'train/common/step_checkpoint.py': [
        ('_plot_metrics', 'Step plot failed at step', 'plot_step_metrics(self._csv_path, self._save_',
         'step metric figures'),
        ('_run_analysis', 'Step analysis failed at step', 'analyzer = ModelAnalyzer(models={self.model_n',
         'step analyzer report'),
    ],
    'train/common/step_plots.py': [
        ('_plot_metrics', 'Step plot failed at step', 'plot_step_metrics(self._csv_path, self._save_',
         'step metric figures'),
    ],
    'train/common/timeseries.py': [
        ('__init__', ': viz-data preparation failed (', 'self.viz_test_data = self._prepare_viz_data()',
         'per-epoch prediction figures'),
    ],
    'train/convunext/common.py': [
        ('attempt', 'Visualization', 'draw()',
         'a figure, recorded in failed'),
    ],
    'train/convunext/segmentation_viz.py': [
        ('_draw', 'Segmentation grid', 'plot_segmentation_grid(self.images, self.mask',
         'segmentation grid figure, recorded in failed'),
    ],
    'train/hkan/train_hkan.py': [
        ('render_figures', "Figure '", 'render()',
         'a figure, recorded in failed'),
    ],
    'train/kan/train_kan.py': [
        ('on_epoch_end', 'Per-epoch dashboard render failed at epo', 'train_history = TrainingHistory(epochs=self._',
         'per-epoch dashboard figure'),
        ('on_epoch_end', 'Periodic function/spline grid render fai', 'render_function_and_spline_grid(self.model, s',
         'function and spline grid figure'),
    ],
    'train/neural_hmm/train_neural_hmm.py': [
        ('render_figures', "Figure '", 'render()',
         'a figure, recorded in failed'),
    ],
    'train/omnipoint/train_omnipoint.py': [
        ('train_omnipoint', 'Failed to save loss curve plot:', 'import matplotlib',
         'loss-curve figure'),
    ],
    'train/rms_variants_train/norm_overhead_bench.py': [
        ('_peak_mem_mb_or_nan', 'peak_mem unavailable (', 'info = tf.config.experimental.get_memory_info',
         'peak-memory telemetry (NaN when unavailable)'),
    ],
    'train/som_nd_soft/train_cifar.py': [
        ('on_epoch_end', 'Failed to create visualizations:', 'reconstructed = self.model.predict(self.x_viz',
         'SOM figures'),
        ('_visualize_som_activations', 'Failed to visualize SOM activations:', 'unique_classes = np.unique(self.y_latent_anal',
         'SOM figure'),
        ('_visualize_grid_interpolations', 'Failed to create grid interpolations:', 'grid_h, grid_w, _ = weights_map.shape',
         'SOM figure'),
        ('_track_sample_trajectories', 'Failed to track trajectories:', 'feature_maps = encoder.predict(self.tracked_s',
         'trajectory figure data'),
        ('_visualize_latent_space', 'Failed to create t-SNE visualization:', 'from sklearn.manifold import TSNE',
         't-SNE figure'),
    ],
    'train/som_nd_soft/train_mnist.py': [
        ('on_epoch_end', 'Failed to create SOM visualizations:', "som_layer = self.model.get_layer('soft_som')",
         'SOM figures'),
        ('_track_sample_trajectories', 'Failed to track trajectories:', "som_layer = self.model.get_layer('soft_som')",
         'trajectory figure data'),
    ],
    'train/superpoint/train_magicpoint.py': [
        ('train_magicpoint', 'MagicPoint viz batch capture failed (viz', 'from train.common.keypoint_viz import SuperPo',
         'keypoint figures'),
    ],
    'train/superpoint/train_superpoint.py': [
        ('train_superpoint', 'SuperPoint viz batch capture failed (viz', 'from train.common.keypoint_viz import SuperPo',
         'keypoint figures'),
    ],
    'train/yolo12/train_coco.py': [
        ('_prepare_visualization_samples', 'Failed to prepare COCO visualization sam', 'sample_batch = next(iter(self.validation_data',
         'detection figures'),
        ('_decode_coco_detections', 'Failed to decode COCO detections:', 'dist_features = 4 * self.reg_max',
         'detection figures'),
        ('on_epoch_end', 'Failed to generate COCO visualizations f', 'predictions = self.model(self.sample_images, ',
         'detection figures'),
        ('_visualize_coco_detections', 'Failed to visualize COCO detections:', 'if isinstance(predictions, dict):',
         'detection figures'),
        ('_visualize_coco_segmentation', 'Failed to visualize COCO segmentation:', 'if isinstance(predictions, dict):',
         'segmentation figures'),
    ],
    'train/yolo12/train_multitask.py': [
        ('save_training_results', 'Failed to save model architecture plot:', 'keras.utils.plot_model(model, to_file=os.path',
         'architecture figure'),
        ('_prepare_visualization_samples', 'Failed to prepare visualization samples:', 'sample_batch = next(iter(self.validation_data',
         'task figures'),
        ('on_epoch_end', 'Failed to generate visualizations for ep', 'predictions = self.model(self.sample_images, ',
         'task figures'),
        ('_visualize_detection', 'Failed to visualize detection:', 'if isinstance(predictions, dict):',
         'detection figure'),
        ('_visualize_segmentation', 'Failed to visualize segmentation:', 'if isinstance(predictions, dict):',
         'segmentation figure'),
        ('_visualize_classification', 'Failed to visualize classification:', 'if isinstance(predictions, dict):',
         'classification figure'),
        ('on_epoch_end', 'Failed to get task weights:', 'task_weights = self.loss_fn.get_task_weights(',
         'task-weight log entries'),
        ('on_epoch_end', 'Failed to get individual losses:', 'individual_losses = self.loss_fn.get_individu',
         'per-task loss log entries'),
    ],
}

#: Named exemptions: path -> [(function, text in the handler, text in the try, reason)].
_EXEMPT = {
    'train/bfunet/common.py': [
        ('_run_test_eval', 'Held-out test evaluation failed', 'block = _evaluate_held_out(config, final_mode',
         "recorded as the test_eval block's status error and logged at ERROR; train writes the summary, then raises on it (D-186)"),
        ('_train_in_run_dir', 'Round-trip pre-save capture skipped:', 'in_shape = model.input_shape',
         "the round-trip check's input; a failed capture records model_loading_validated as null in the summary"),
        ('build_self_iterate_pool', 'continue', 'patch = load_and_preprocess_image(tf.constant',
         'an unreadable image is skipped (attempts bounded); a short pool is warned and an empty pool raises'),
        ('build_fixed_val_batch', 'continue', 'if seed is None:',
         'the fixed visualization batch; an unreadable image is skipped and none at all gives no batch'),
        ('_homogeneity_probe', 'float32', 'compute_dtype = keras.mixed_precision.global_',
         "the global policy only picks the probe's tolerance; float32 when it cannot be read"),
        ('on_train_begin', 'Epoch-0 baseline eval failed:', 'res = self.model.evaluate(self.val_ds, steps=',
         "the epoch-0 point of the dashboard and initial_loss_sanity_eval (null when it fails); the run's own evaluation is separate"),
    ],
    'train/bfunet/eval_psnr_vs_noise.py': [
        ('_load_clean_patch', 'skip unreadable image', 'raw = tf.io.read_file(path)',
         'an unreadable image is skipped and the next is drawn; every result row carries its scored count n'),
        ('_load_full_image', 'skip unreadable image', 'raw = tf.io.read_file(path)',
         'an unreadable image is skipped and the next is drawn; every result row carries its scored count n'),
    ],
    'train/cliffordnet/prepare_cc3m.py': [
        ('_write_pair', '[warn] decode failed for', 'im = PILImage.open(BytesIO(jpg_bytes))',
         'a corrupt JPEG is printed and its pair is not written (caption included)'),
        ('_extract_split', 'HTTP open failed', 'tar = _open_tar_stream(tar_name)',
         'counts the shard as failed and stops the split; the shard stays in progress and a rerun extracts it again under the same ids (D-187); main raises after every split (D-183)'),
        ('_extract_split', 'tar read failed', 'for jpg_bytes, caption in _iter_pairs_in_tar(',
         'counts the shard as failed and stops the split; the shard stays in progress and a rerun extracts it again under the same ids (D-187); main raises after every split (D-183)'),
        ('_extract_split', 'pass', 'tar.close()',
         'closing a stream during cleanup'),
    ],
    'train/cliffordnet/resize_cc3m_to_ssd.py': [
        ('_resize_one', 'err:', 'if os.path.exists(dst):',
         'returns an err status that main counts; main raises after every split (D-183)'),
    ],
    'train/cliffordnet/train_clip.py': [
        ('_run_probe', "scale_val = float('nan')", 'scale_val = float(keras.ops.convert_to_numpy(',
         'a logged diagnostic value (NaN when unreadable)'),
        ('_gamma_mean', 'return None', 'return float(keras.ops.mean(keras.ops.convert',
         'a logged diagnostic value (None when unreadable)'),
    ],
    'train/cliffordnet/train_downsampling_techniques.py': [
        ('train_one_variant', 'BUILD FAILED for', 'model = build_variant(variant_name, num_class',
         'returns None; train_all_variants records the variant as failed and raises after the sweep (D-095)'),
        ('train_one_variant', 'TRAIN FAILED for', 'history = model.fit(train_ds, epochs=epochs, ',
         'returns None; train_all_variants records the variant as failed and raises after the sweep (D-095)'),
    ],
    'train/common/callbacks.py': [
        ('read_current_lr', "return float('nan')", 'optimizer = model.optimizer',
         'the learning rate for a log line or figure; NaN when unreadable'),
    ],
    'train/common/classification_viz.py': [
        ('on_train_begin', 'Dashboard epoch-0 baseline failed:', 'if self.baseline_fn is not None:',
         "the epoch-0 point of the dashboard; the run's own evaluation is separate"),
        ('_current_lr', 'return None', 'return float(keras.ops.convert_to_numpy(self.',
         'the learning rate for the dashboard; None when unreadable'),
    ],
    'train/common/datasets.py': [
        ('get_class_names', 'class_', "info = tfds.builder('imagenet2012').info",
         'label names for figures; class_N when the TFDS metadata is unavailable'),
    ],
    'train/common/image_text.py': [
        ('load_cc3m_local_split', ': tokenization cache read failed (', 'if os.path.exists(cache_npz) and os.path.exis',
         'a tokenization cache; a failed read recomputes the tokens'),
    ],
    'train/common/run_summary.py': [
        ('load_best_metrics', 'Could not evaluate best checkpoint', 'return (evaluate(keras.models.load_model(path',
         "returns the error; the three callers record it as the summary's best_checkpoint_load_error, then fail_on_best_load_error raises (D-186)"),
        ('read_analysis_status', 'Could not read analyzer status from', 'with open(path) as f:',
         "recorded as the analysis block's error in the summary"),
        ('read_data_free_analysis_status', 'unreadable', 'with open(path) as f:',
         "recorded as the analysis block's status unreadable and its error in the summary"),
    ],
    'train/common/step_checkpoint.py': [
        ('_save_checkpoint', 'Checkpoint save failed at step', 'self.checkpoint_model.save(path)',
         'records the failed step; on_train_end raises after the final save (D-183)'),
        ('close', 'pass', 'self._csv_file.close()',
         'closing the CSV handle'),
        ('_log_metrics', 'pass', 'self._csv_file.close()',
         'closing the CSV handle'),
    ],
    'train/common/timeseries.py': [
        ('_track_lr', 'return', 'lr = float(keras.ops.convert_to_numpy(self.mo',
         'the learning rate for the history figure; skipped when unreadable'),
    ],
    'train/doc_res/common.py': [
        ('_image_size', 'cannot read image header for %s: %s', 'with Image.open(path) as handle:',
         'an unreadable header drops the triplet; the dropped count is logged'),
    ],
    'train/doc_res/prepare_doc_res_data.py': [
        ('_http_head', 'return (0, None, f', 'resp = requests.head(url, allow_redirects=Tru',
         'a dead host is a dead-url result; main returns 1 (D-183)'),
        ('_libarchive', 'return None', 'import libarchive',
         'optional dependency import (libarchive-c); its absence is logged at start'),
        ('stage_archive', 'part.unlink()', 'written = _download(archive.url, part, timeou',
         'a failed ArchiveResult; main returns 1 (D-183)'),
        ('stage_archive', 'recorded = {}', 'recorded = json.loads(marker.read_text())',
         'a damaged marker means not staged; the archive is staged again'),
        ('precompute_sidecars', 'prompt sidecar failed for %s: %s', 'with Image.open(image) as handle:',
         'counts the failed sidecar; main returns 1 (D-183)'),
    ],
    'train/doc_scanner/prepare_doc_scanner_data.py': [
        ('_http_head', 'return (0, None, False, f', 'resp = requests.head(url, allow_redirects=Tru',
         'a dead host is reported as status 0; main returns 1 when nothing is staged'),
        ('read_marker', 'return {}', 'loaded = json.loads(marker.read_text())',
         'a damaged marker means not staged'),
        ('stage_uvdoc', 'B are kept in', 'written, on_disk = _download(url, part, timeo',
         'an incomplete StageResult; main returns 1'),
    ],
    'train/logic/multiseed_sweep.py': [
        ('run_one', 'UNEXPECTED error', 'cp = subprocess.run(cmd, check=False, timeout',
         'returns an error status; main raises after the summary (D-183)'),
        ('collect_csvs', '[agg] failed to read', 'df = pd.read_csv(csv_path)',
         'records the unreadable CSV; main raises after the summary (D-183)'),
    ],
    'train/logic/train_benchmark.py': [
        ('train_one', '<to_symbolic failed:', 'symbolic = nc.to_symbolic(top_k=1)',
         'the symbolic-rule column holds the failure text'),
    ],
    'train/logic/train_e4_lowdata_mux.py': [
        ('run_cell', 'params = -1', 'params = int(clf.get_booster().trees_to_dataf',
         'the XGBoost parameter-count column; -1 when the booster dump is unavailable'),
    ],
    'train/logic/train_e4_monks.py': [
        ('run_cell', '<to_symbolic failed:', 'symbolic = nc.to_symbolic(top_k=1)',
         'the symbolic-rule column holds the failure text'),
        ('run_cell', 'params = -1', 'params = int(clf.get_booster().trees_to_dataf',
         'the XGBoost parameter-count column; -1 when the booster dump is unavailable'),
    ],
    'train/logic/train_e5_clevr_hans.py': [
        ('train_and_eval', 'fit raised', 'history = model.fit(train_ds, validation_data',
         'a fit_error row; main raises after the report (D-176)'),
        ('train_and_eval', 'val eval failed', 'val_acc = float(model.evaluate(val_ds, verbos',
         'a None accuracy; main raises after the report (D-176)'),
        ('train_and_eval', 'test eval failed', 'test_acc = float(model.evaluate(test_ds, verb',
         'a None accuracy; main raises after the report (D-176)'),
    ],
    'train/mothnet/multiseed_sweep.py': [
        ('run_one', 'UNEXPECTED error', 'cp = subprocess.run(cmd, check=False, timeout',
         'returns an error status; main raises after the summary (D-183)'),
        ('collect_csvs', '[agg] failed to read', 'run_df = pd.read_csv(csv_path)',
         'a failed status row; main raises after the summary (D-183)'),
    ],
    'train/nam/data_generator.py': [
        ('_safe_eval', 'return (0.0, False)', "result = eval(compile(expression, '<expr>', '",
         'an expression that does not evaluate is an invalid sample, flagged False'),
        ('_parse_single_op', 'left_operand = 0.0', "tree = ast.parse(expression, mode='eval')",
         'an expression that is not one binary operation of two literals has no single operand pair; the multi-op path takes its operands from _parse_multi_op'),
    ],
    'train/ntm/harness.py': [
        ('run_full_suite', "Benchmark '", 'getattr(self, self.BENCHMARK_METHODS[benchmar',
         "recorded in the suite report's errored_benchmarks (02be1d7e/D-014); run_benchmark_suite.run_suite writes the report, then raises (D-186)"),
    ],
    'train/omnipoint/data.py': [
        ('load_megadepth_pair_for_omnipoint', 'Skipping MegaDepth pair (rgb=', "with h5py.File(depth_path, 'r') as f:",
         'one corrupt pair is skipped and logged by name'),
    ],
    'train/omnipoint/kitti_depth.py': [
        ('load_kitti_depth_pair', 'Skipping KITTI pair (rgb=', "rgb_img = Image.open(rgb_path).convert('RGB')",
         'one corrupt pair is skipped and logged by name'),
        ('discover_kitti_depth_pairs', 'Skipping KITTI depth pair (drive=', 'rgb_path = rgb_cam_dir / depth_path.name',
         'a depth frame with no RGB frame is skipped and logged by name'),
    ],
    'train/rms_variants_train/callbacks.py': [
        ('on_train_end', '[DistributionShiftProbe]', "ds = tfds.load(dataset_name, split='test', as",
         'plan_e1f12eab D-002: recorded as a NaN row with status dataset_missing'),
    ],
    'train/rms_variants_train/norm_overhead_bench.py': [
        ('_reset_peak_mem', 'pass', "tf.config.experimental.reset_memory_stats('GP",
         'resetting peak-memory telemetry'),
    ],
    'train/yolo12/train_coco.py': [
        ('calculate_steps_per_epoch', 'Could not determine dataset size:', 'sample_count = 0',
         'a size estimate for steps_per_epoch; a dataset that cannot be read raises in fit'),
    ],
    'train/yolo12/train_multitask.py': [
        ('test_model_compilation', 'Model compilation test failed:', 'sample_batch = next(iter(train_dataset))',
         'a graph-mode probe; a failure switches to eager execution and is logged'),
        ('on_epoch_end', 'pass', 'lr = float(self.model.optimizer.learning_rate',
         'the learning-rate log entry'),
    ],
}


def _parse(source):
    with warnings.catch_warnings():  # invalid escapes in scanned sources
        warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
        return ast.parse(source)


def _handler_text(handler):
    """The handler's source plus its string constants (f-string parts included)."""
    constants = [node.value for node in ast.walk(handler)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    return "\n".join([ast.unparse(handler)] + constants)


def _logs_the_exception(handler):
    for node in _walk_no_defs(handler.body):
        if not isinstance(node, ast.Call):
            continue
        func = ast.unparse(node.func)
        if not func.startswith(("logger.", "self.logger.")):
            continue
        if func.endswith(".exception"):
            return True
        for keyword in node.keywords:
            if keyword.arg == "exc_info" and not (
                    isinstance(keyword.value, ast.Constant) and not keyword.value.value):
                return True
        if handler.name and any(isinstance(sub, ast.Name) and sub.id == handler.name
                                for arg in node.args for sub in ast.walk(arg)):
            return True
    return False


def _forbidden_calls(statements):
    names = []
    for node in _walk_no_defs(statements):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name in FORBIDDEN_CALLS:
                names.append(name)
    return names


def _outside_main_handlers(tree):
    """(function name, function node or None, try, handler) for every broad
    handler that is not lexically in a ``main``."""
    in_main = {id(handler) for _f, _t, handler in _main_broad_handlers(tree)}
    scopes = [("<module>", None, tree.body)]
    scopes += [(node.name, node, node.body) for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for name, func, body in scopes:
        for node in _walk_no_defs(body):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if _is_broad(handler) and id(handler) not in in_main:
                        yield name, func, node, handler


def _matches(entry, name, try_node, handler):
    function, handler_text, try_text, _note = entry
    return (function == name and handler_text in _handler_text(handler)
            and try_text in "\n".join(ast.unparse(s) for s in try_node.body))


def _classify(source, optional=(), exempt=()):
    """Census rows ``(line, function, verdict, entry)``; verdict ``None`` is a swallow.

    ``entry`` is the table entry that matched (``None`` when none did). An optional
    entry whose shape no longer holds is a swallow.
    """
    rows = []
    for name, func, try_node, handler in _outside_main_handlers(_parse(source)):
        entry = None
        if any(isinstance(n, ast.Raise) for n in _walk_no_defs(handler.body)):
            verdict = "raises"
        elif _exits_non_zero(handler.body, False):
            verdict = "exits"
        elif func is not None and _fails_later(func, try_node, handler, False):
            verdict = "records"
        else:
            verdict = None
            for candidate in optional:
                if _matches(candidate, name, try_node, handler):
                    entry = candidate
                    if (_logs_the_exception(handler)
                            and not _forbidden_calls(try_node.body)):
                        verdict = "optional"
            for candidate in exempt:
                if _matches(candidate, name, try_node, handler):
                    entry, verdict = candidate, "exempt"
        rows.append((handler.lineno, name, verdict, entry))
    return sorted(rows, key=lambda row: row[0])


def _census():
    census = {}
    for path in sorted((_SRC / "train").rglob("*.py")):
        relative = path.relative_to(_SRC).as_posix()
        rows = _classify(path.read_text(encoding="utf-8"),
                         _OPTIONAL_OUTPUT.get(relative, ()), _EXEMPT.get(relative, ()))
        if rows:
            census[relative] = rows
    return census


CENSUS = _census()


def test_every_broad_handler_outside_main_is_classified():
    swallows = [f"{path}:{line} ({name})" for path, rows in CENSUS.items()
                for line, name, verdict, _entry in rows if verdict is None]
    assert swallows == [], (
        "these broad handlers outside main() swallow a failure, or are listed as "
        "optional output but do not log the exception or guard a training, "
        "evaluation or save call; re-raise, record the failure and fail the run "
        "(D-095), or classify them in this file:\n" + "\n".join(swallows))


def test_each_table_entry_names_exactly_one_handler():
    """No stale, ambiguous or reasonless entry, and no handler in both tables."""
    for table in (_OPTIONAL_OUTPUT, _EXEMPT):
        for relative, entries in table.items():
            source = (_SRC / relative).read_text(encoding="utf-8")
            handlers = list(_outside_main_handlers(_parse(source)))
            for entry in entries:
                assert entry[3], (relative, entry)
                hits = [handler.lineno for name, _f, try_node, handler in handlers
                        if _matches(entry, name, try_node, handler)]
                assert len(hits) == 1, (relative, entry[:2], hits)
    for relative in set(_OPTIONAL_OUTPUT) & set(_EXEMPT):
        lines = {line for line, _n, _v, entry in CENSUS[relative]
                 if entry in _OPTIONAL_OUTPUT[relative]}
        assert not lines & {line for line, _n, _v, entry in CENSUS[relative]
                            if entry in _EXEMPT[relative]}


def test_the_census_counts_are_pinned():
    """A new handler, or one that changes kind, moves a count and is looked at.

    Optional 60 -> 57 at plan-2026-10-14T062616-81938e81 step 11.1: the SAM,
    SAM 2 and SAM 3 trainers' copies of the GPU-peak loop became calls to
    ``train.common.gpu.log_gpu_peak_memory``, whose handler is listed once."""
    verdicts = [verdict for rows in CENSUS.values() for _l, _n, verdict, _e in rows]
    counts = {kind: verdicts.count(kind)
              for kind in ("raises", "exits", "records", "optional", "exempt", None)}
    assert counts == {"raises": 8, "exits": 4, "records": 4,
                      "optional": 57, "exempt": 60, None: 0}
    assert sum(len(entries) for entries in _OPTIONAL_OUTPUT.values()) == 57
    assert sum(len(entries) for entries in _EXEMPT.values()) == 60


_PLANTED = '''
def plot(model, path):
    try:
        draw(model, path)
    except Exception as e:
        logger.warning(f"plot failed: {e}")


def plot_quietly(path):
    try:
        draw(path)
    except Exception:
        logger.warning("plot failed quietly")


def save_with_the_plot(model, path):
    try:
        draw(model, path)
        model.save(path)
    except Exception as e:
        logger.warning(f"plot or save failed: {e}")


def unlisted(model, x):
    try:
        model.fit(x)
    except Exception as e:
        logger.warning(f"fit failed: {e}")


def read_lr(model):
    try:
        return float(model.optimizer.learning_rate)
    except Exception:
        return float("nan")


def reraises():
    try:
        work()
    except Exception:
        logger.error("work failed")
        raise


def sweep(cells):
    failed = []
    for cell in cells:
        try:
            run(cell)
        except Exception:
            failed.append(cell)
    if failed:
        raise RuntimeError(failed)


def main():
    try:
        run()
    except Exception as e:
        logger.warning(f"left to the main census: {e}")
'''

_PLANTED_OPTIONAL = [
    ("plot", "plot failed:", "draw(model, path)", "a figure"),
    # Listed as optional, but the handler drops the exception.
    ("plot_quietly", "plot failed quietly", "draw(path)", "a figure"),
    # Listed as optional, but the try also saves the model.
    ("save_with_the_plot", "plot or save failed", "draw(model, path)", "a figure"),
]
_PLANTED_EXEMPT = [("read_lr", "nan", "model.optimizer.learning_rate", "a log value")]


def test_the_census_tells_a_swallow_from_a_classified_handler():
    """Positive control over planted forms: four swallows, four classified."""
    verdicts = {name: verdict for _line, name, verdict, _entry
                in _classify(_PLANTED, _PLANTED_OPTIONAL, _PLANTED_EXEMPT)}
    assert verdicts == {
        "plot": "optional",
        "plot_quietly": None,
        "save_with_the_plot": None,
        "unlisted": None,
        "read_lr": "exempt",
        "reraises": "raises",
        "sweep": "records",
    }
    # Unlisted, the optional figure is a swallow too.
    assert dict((name, verdict) for _l, name, verdict, _e in _classify(_PLANTED))["plot"] is None


def test_the_census_sees_the_tree():
    """Anti-vacuity: the scan reaches the handlers it classifies."""
    assert sum(len(rows) for rows in CENSUS.values()) >= 120
    assert len(CENSUS) >= 50
