# Utils

Utility functions and helpers

**72 modules in this category**

## Alignment

### utils.alignment
Platonic Representation Alignment Module for dl_techniques.

*📁 File: `src/dl_techniques/utils/alignment/__init__.py`*

### utils.alignment.alignment
Main alignment API for measuring representation similarity.

**Classes:**
- `Alignment`
- `AlignmentLogger`

**Functions:** `score`, `compute_pairwise_alignment`, `compute_alignment_matrix`, `set_reference_features`, `get_supported_metrics` (and 4 more)

*📁 File: `src/dl_techniques/utils/alignment/alignment.py`*

### utils.alignment.metrics
Alignment metrics for measuring representation similarity.

**Classes:**
- `AlignmentMetrics`

**Functions:** `remove_outliers`, `measure`, `cycle_knn`, `mutual_knn`, `lcs_knn` (and 6 more)

*📁 File: `src/dl_techniques/utils/alignment/metrics.py`*

### utils.alignment.utils
Utility functions for alignment computation and feature processing.

**Functions:** `prepare_features`, `compute_score`, `compute_alignment_matrix`, `normalize_features`, `extract_layer_features` (and 7 more)

*📁 File: `src/dl_techniques/utils/alignment/utils.py`*

## Analyzer_Callback

### callbacks.analyzer_callback
Keras Callback for In-Training Model Analysis

**Classes:**
- `EpochAnalyzerCallback`

**Functions:** `on_epoch_end`

*📁 File: `src/dl_techniques/callbacks/analyzer_callback.py`*

## Arc

### datasets.arc

*📁 File: `src/dl_techniques/datasets/arc/__init__.py`*

### datasets.arc.arc_converters
This module provides a comprehensive suite of utilities for converting, augmenting,

**Classes:**
- `ARCTaskData`
- `AugmentationConfig`
- `ARCFormatConverter`
- `ARCDatasetMerger`
- `ARCDataAugmenter`
- `ARCDatasetSplitter`

**Functions:** `dihedral_transform`, `inverse_dihedral_transform`, `json_to_task_data`, `load_tasks_from_directory`, `grid_to_sequence` (and 5 more)

*📁 File: `src/dl_techniques/datasets/arc/arc_converters.py`*

### datasets.arc.arc_keras
This module provides a comprehensive suite of Keras-native utilities designed to

**Classes:**
- `ARCSequence`

- `ARCGridDecoder` - Keras Layer
  Custom Keras layer for decoding ARC grid sequences.
  ```python
  ARCGridDecoder(max_grid_size: int = 30, pad_token: int = 0, eos_token: int = 1, ...)
  ```

- `ARCGridEncoder` - Keras Layer
  Custom Keras layer for encoding 2D grids to sequences.
  ```python
  ARCGridEncoder(max_grid_size: int = 30, pad_token: int = 0, eos_token: int = 1, ...)
  ```

- `ARCPuzzleEmbedding` - Keras Layer
  Custom embedding layer for ARC puzzle identifiers.
  ```python
  ARCPuzzleEmbedding(num_puzzles: int, embedding_dim: int, mask_zero: bool = True, ...)
  ```
- `ARCAccuracyMetric`

**Functions:** `create_arc_data_generator`, `create_simple_arc_model`, `on_epoch_end`, `build`, `call` (and 15 more)

*📁 File: `src/dl_techniques/datasets/arc/arc_keras.py`*

### datasets.arc.arc_utilities
This module provides a comprehensive, object-oriented toolkit for loading, analyzing,

**Classes:**
- `ARCExample`
- `ARCPuzzle`
- `ARCDatasetStats`
- `ARCDatasetLoader`
- `ARCDatasetAnalyzer`
- `ARCDatasetVisualizer`
- `ARCDatasetValidator`

**Functions:** `load_identifiers_map`, `load_split_metadata`, `load_split_data`, `load_puzzles`, `compute_dataset_statistics` (and 5 more)

*📁 File: `src/dl_techniques/datasets/arc/arc_utilities.py`*

## Bdd100K_Video

### datasets.bdd100k_video
BDD100K video clip dataset for Video-JEPA-Clifford training.

**Functions:** `bdd100k_video_dataset`

*📁 File: `src/dl_techniques/datasets/bdd100k_video.py`*

## Bounding_Box

### utils.bounding_box
Object Detection IoU Utility Module

**Functions:** `bbox_iou`, `bbox_nms`

*📁 File: `src/dl_techniques/utils/bounding_box.py`*

## Coco_Map_Callback

### callbacks.coco_map_callback
COCO mAP evaluation callback for CliffordNet multi-task detection.

**Classes:**
- `COCOMAPCallback`

**Functions:** `on_epoch_end`

*📁 File: `src/dl_techniques/callbacks/coco_map_callback.py`*

## Coco_Multitask_Visualization

### callbacks.coco_multitask_visualization
COCO multi-task visualization callback.

**Classes:**
- `COCOMultiTaskPredictionGridCallback`

**Functions:** `on_epoch_end`

*📁 File: `src/dl_techniques/callbacks/coco_multitask_visualization.py`*

## Conformal_Forecaster

### utils.conformal_forecaster
Conformal Prediction for Time Series Forecasting.

**Classes:**
- `ConformalForecaster`

**Functions:** `calibrate`, `predict`, `update`, `evaluate_coverage`, `get_efficiency_metrics` (and 1 more)

*📁 File: `src/dl_techniques/utils/conformal_forecaster.py`*

## Constants

### utils.constants

*📁 File: `src/dl_techniques/utils/constants.py`*

## Convert

### utils.convert

**Functions:** `convert_numpy_to_python`

*📁 File: `src/dl_techniques/utils/convert.py`*

## Core

### __init__

*📁 File: `src/dl_techniques/__init__.py`*

### callbacks

*📁 File: `src/dl_techniques/callbacks/__init__.py`*

### datasets

*📁 File: `src/dl_techniques/datasets/__init__.py`*

### utils

*📁 File: `src/dl_techniques/utils/__init__.py`*

## Corruption

### utils.corruption
Keras-only implementation of image corruption functions with severity enum.

**Classes:**
- `CorruptionSeverity`
- `CorruptionType`

**Functions:** `apply_gaussian_noise`, `apply_impulse_noise`, `apply_shot_noise`, `apply_gaussian_blur`, `apply_motion_blur` (and 9 more)

*📁 File: `src/dl_techniques/utils/corruption.py`*

## Deep_Supervision

### utils.deep_supervision
Deep-supervision plumbing helpers.

**Functions:** `get_model_output_info`, `create_inference_model_from_training_model`

*📁 File: `src/dl_techniques/utils/deep_supervision.py`*

## Depth_Visualization

### callbacks.depth_visualization
Depth estimation visualization callbacks.

**Classes:**
- `DepthPredictionGridCallback`
- `DepthMetricsCurveCallback`

**Functions:** `on_epoch_end`, `on_epoch_end`

*📁 File: `src/dl_techniques/callbacks/depth_visualization.py`*

## Drop_Path

### utils.drop_path
Drop-path (stochastic depth) rate schedules.

**Functions:** `linear_drop_path_rates`

*📁 File: `src/dl_techniques/utils/drop_path.py`*

## Export

### utils.export

*📁 File: `src/dl_techniques/utils/export/__init__.py`*

### utils.export.onnx
ONNX Export Utility for Keras 3 Models

**Functions:** `export_keras_model_to_onnx`, `export_with_dynamic_batch`, `get_onnx_model_info`, `model_func`

*📁 File: `src/dl_techniques/utils/export/onnx.py`*

### utils.export.tflite

**Functions:** `export_tirex_to_tflite`, `verify_tflite`

*📁 File: `src/dl_techniques/utils/export/tflite.py`*

## Filesystem

### utils.filesystem

**Functions:** `count_available_files`, `image_file_generator`

*📁 File: `src/dl_techniques/utils/filesystem.py`*

## Forecastability_Analyzer

### utils.forecastability_analyzer
Forecastability Assessment Framework.

**Classes:**
- `ForecastabilityAssessor`

**Functions:** `permutation_entropy`, `auto_permutation_entropy`, `calculate_naive_baselines`, `forecast_value_added`, `assess_forecastability`

*📁 File: `src/dl_techniques/utils/forecastability_analyzer.py`*

## Geometry

### utils.geometry

*📁 File: `src/dl_techniques/utils/geometry/__init__.py`*

### utils.geometry.poincare_math
Poincaré Ball Model Geometry Utilities.

**Classes:**
- `PoincareMath`

**Functions:** `safe_norm`, `project`, `exp_map_0`, `log_map_0`, `mobius_add`

*📁 File: `src/dl_techniques/utils/geometry/poincare_math.py`*

## Graphs

### utils.graphs
Graph Data Utilities for sHGCN.

**Functions:** `normalize_adjacency_symmetric`, `normalize_adjacency_row`, `sparse_to_tf_sparse`, `create_random_graph`, `preprocess_features` (and 1 more)

*📁 File: `src/dl_techniques/utils/graphs.py`*

## Inference

### utils.inference
Full Image Inference Utilities for YOLOv12 Multi-task Model.

**Classes:**
- `InferenceConfig`
- `FullImageInference`
- `InferenceProfiler`

**Functions:** `create_inference_engine`, `predict_image`, `predict_batch_images`, `profile_inference`

*📁 File: `src/dl_techniques/utils/inference.py`*

## Jepa_Visualization

### callbacks.jepa_visualization
JEPA latent-masking visualization callbacks.

**Classes:**
- `LatentMaskOverlayCallback`
- `PatchPredictionErrorCallback`

**Functions:** `on_train_begin`, `on_epoch_end`, `on_train_begin`, `on_epoch_end`

*📁 File: `src/dl_techniques/callbacks/jepa_visualization.py`*

## Logger

### utils.logger

*📁 File: `src/dl_techniques/utils/logger.py`*

## Masking

### utils.masking

*📁 File: `src/dl_techniques/utils/masking/__init__.py`*

### utils.masking.factory
Masking utilities for attention and segmentation models.

**Classes:**
- `MaskType`
- `MaskConfig`
- `MaskFactory`

**Functions:** `create_mask`, `apply_mask`, `combine_masks`, `visualize_mask`, `get_mask_info` (and 11 more)

*📁 File: `src/dl_techniques/utils/masking/factory.py`*

### utils.masking.strategies
Advanced Masking and Corruption Strategies

**Functions:** `apply_mlm_masking`

*📁 File: `src/dl_techniques/utils/masking/strategies.py`*

## Nlp

### datasets.nlp
NLP dataset loaders for text-based training pipelines.

**Functions:** `load_wikipedia_train_val`, `load_hf_text_dataset`, `generator`, `generator`

*📁 File: `src/dl_techniques/datasets/nlp.py`*

## Patch_Transforms

### datasets.patch_transforms
Coordinate Transformation Utilities for Patch-based Multi-task Learning.

**Classes:**
- `PatchInfo`
- `DetectionResult`
- `PatchPrediction`
- `CoordinateTransformer`
- `PatchGridGenerator`
- `NonMaximumSuppression`
- `SegmentationStitcher`
- `ClassificationAggregator`
- `ResultAggregator`

**Functions:** `create_patch_grid`, `aggregate_patch_results`, `width`, `height`, `center_x` (and 13 more)

*📁 File: `src/dl_techniques/datasets/patch_transforms.py`*

## Pusht_Hdf5

### datasets.pusht_hdf5
PushT HDF5 dataset loader + synthetic LeWM dataset generator.

**Classes:**
- `PushTHDF5Dataset`

**Functions:** `synthetic_lewm_dataset`, `gen`, `as_tf_dataset`, `gen`

*📁 File: `src/dl_techniques/datasets/pusht_hdf5.py`*

## Random

### utils.random

**Functions:** `rayleigh`, `validate_rayleigh_samples`

*📁 File: `src/dl_techniques/utils/random.py`*

## Scaling

### utils.scaling

**Functions:** `range_from_bits`, `round_clamp`, `sample`, `abs_max`, `abs_mean` (and 2 more)

*📁 File: `src/dl_techniques/utils/scaling.py`*

## Simple_2D

### datasets.simple_2d
2D Classification Dataset Generator

**Classes:**
- `DatasetType`
- `DatasetGenerator`

**Functions:** `generate_dataset`, `generate_moons`, `generate_circles`, `generate_clusters`, `generate_xor` (and 5 more)

*📁 File: `src/dl_techniques/datasets/simple_2d.py`*

## Sut

### datasets.sut
TensorFlow-Native SUT-Crack Dataset Patch-based Loader - Optimized Version.

**Classes:**
- `BoundingBox`
- `ImageAnnotation`
- `TensorFlowNativePatchSampler`
- `OptimizedSUTDataset`

**Functions:** `create_sut_crack_dataset`, `width`, `height`, `center_x`, `center_y` (and 34 more)

*📁 File: `src/dl_techniques/datasets/sut.py`*

## Synthetic_Drone_Video

### datasets.synthetic_drone_video
Synthetic drone-video dataset generator for Video-JEPA-Clifford smoke training.

**Functions:** `synthetic_drone_video_dataset`, `gen`

*📁 File: `src/dl_techniques/datasets/synthetic_drone_video.py`*

## Tabular

### datasets.tabular

**Classes:**
- `TabularDataProcessor`

**Functions:** `fit`, `transform`, `fit_transform`

*📁 File: `src/dl_techniques/datasets/tabular.py`*

## Tensors

### utils.tensors

**Functions:** `reshape_to_2d`, `gram_matrix`, `wt_x_w_normalize`, `power_iteration`, `create_causal_mask` (and 9 more)

*📁 File: `src/dl_techniques/utils/tensors.py`*

## Time_Series

### datasets.time_series
Time Series Dataset Module.

*📁 File: `src/dl_techniques/datasets/time_series/__init__.py`*

### datasets.time_series.base
Time Series Dataset Base Classes.

**Classes:**
- `BaseTimeSeriesDataset`

**Functions:** `download`, `load`, `get_config`, `list_groups`, `split_data` (and 2 more)

*📁 File: `src/dl_techniques/datasets/time_series/base.py`*

### datasets.time_series.config
Time Series Dataset Configuration Module.

**Classes:**
- `TimeSeriesConfig`
- `WindowConfig`
- `DatasetSplits`
- `PipelineConfig`
- `NormalizationConfig`

*📁 File: `src/dl_techniques/datasets/time_series/config.py`*

### datasets.time_series.favorita
Favorita Grocery Sales Forecasting Dataset.

**Classes:**
- `FavoritaDataset`

**Functions:** `load_favorita`, `download`, `load`, `load_raw`

*📁 File: `src/dl_techniques/datasets/time_series/favorita.py`*

### datasets.time_series.generator
Comprehensive Time Series Generator for Deep Learning and Forecasting Experiments

**Classes:**
- `TimeSeriesGeneratorConfig`
- `TimeSeriesGenerator`

**Functions:** `get_task_names`, `get_task_categories`, `get_tasks_by_category`, `generate_task_data`, `generate_all_patterns` (and 3 more)

*📁 File: `src/dl_techniques/datasets/time_series/generator.py`*

### datasets.time_series.long_horizon
Long Horizon Forecasting Benchmark Datasets.

**Classes:**
- `LongHorizonDataset`

**Functions:** `load_ett`, `load_weather`, `load_ecl`, `download`, `load` (and 2 more)

*📁 File: `src/dl_techniques/datasets/time_series/long_horizon.py`*

### datasets.time_series.m4
M4 Competition Dataset Loader.

**Classes:**
- `M4Dataset`

**Functions:** `load_m4`, `get_m4_horizon`, `get_m4_seasonality`, `download`, `load` (and 3 more)

*📁 File: `src/dl_techniques/datasets/time_series/m4.py`*

### datasets.time_series.normalizer
Time Series Normalization Module.

**Classes:**
- `NormalizationMethod`
- `TimeSeriesNormalizer`

**Functions:** `available_methods`, `supports_perfect_inverse`, `fit`, `transform`, `transform_quantile_uniform` (and 4 more)

*📁 File: `src/dl_techniques/datasets/time_series/normalizer.py`*

### datasets.time_series.pipeline
Time Series Data Pipeline Module.

**Functions:** `create_sliding_windows`, `make_tf_dataset`, `make_tf_dataset_from_arrays`, `create_train_val_test_datasets`, `add_time_features` (and 3 more)

*📁 File: `src/dl_techniques/datasets/time_series/pipeline.py`*

### datasets.time_series.utils
Time Series Dataset Utilities.

**Functions:** `extract_file`, `download_file`, `ensure_directory`, `get_cache_path`, `clean_cache` (and 1 more)

*📁 File: `src/dl_techniques/datasets/time_series/utils.py`*

## Tokenizer

### utils.tokenizer
Tiktoken Preprocessor for BERT-style Inputs.

**Classes:**
- `TiktokenPreprocessor`

**Functions:** `get_special_token_ids`, `batch_encode`, `encode`, `vocab_size`, `decode`

*📁 File: `src/dl_techniques/utils/tokenizer.py`*

## Train

### utils.train

**Classes:**
- `TrainingConfig`

**Functions:** `train_model`

*📁 File: `src/dl_techniques/utils/train.py`*

## Training_Curves

### callbacks.training_curves
Training curves callback — save per-epoch metric curves as PNGs.

**Classes:**
- `TrainingCurvesCallback`

**Functions:** `on_epoch_end`

*📁 File: `src/dl_techniques/callbacks/training_curves.py`*

## Universal_Dataset_Loader

### datasets.universal_dataset_loader
Universal Dataset Loader

**Classes:**
- `UniversalDatasetLoader`

**Functions:** `dataset`, `get_generator`, `to_tf_dataset`, `to_tf_dataset_tuple`, `generator_factory` (and 1 more)

*📁 File: `src/dl_techniques/datasets/universal_dataset_loader.py`*

## Vision

### datasets.vision

*📁 File: `src/dl_techniques/datasets/vision/__init__.py`*

### datasets.vision.coco
COCO Dataset Preprocessor for YOLOv12 Pre-training.

**Classes:**
- `AugmentationConfig`
- `DatasetConfig`
- `COCODatasetBuilder`

**Functions:** `create_dummy_coco_dataset`, `create_coco_dataset`, `num_classes`, `from_class_names`, `coco_default` (and 8 more)

*📁 File: `src/dl_techniques/datasets/vision/coco.py`*

### datasets.vision.coco_multitask_local
Local COCO 2017 multi-task loader — image classification + semantic segmentation.

**Classes:**
- `COCOMultiTaskConfig`
- `COCO2017MultiTaskLoader`

**Functions:** `build_coco_multitask_datasets`, `on_epoch_end`, `probe`

*📁 File: `src/dl_techniques/datasets/vision/coco_multitask_local.py`*

### datasets.vision.common
Reusable dataset builders for common vision datasets.

**Classes:**
- `MNISTDatasetBuilder`
- `CIFAR10DatasetBuilder`
- `CIFAR100DatasetBuilder`

**Functions:** `create_dataset_builder`, `get_dataset_info`, `build`, `get_test_data`, `get_class_names` (and 6 more)

*📁 File: `src/dl_techniques/datasets/vision/common.py`*

### datasets.vision.imagenet
ImageNet dataset loader using TensorFlow Datasets.

**Functions:** `load_imagenet`, `preprocess`

*📁 File: `src/dl_techniques/datasets/vision/imagenet.py`*

## Visualization

### utils.visualization
Visualization utilities for comparing confusion matrices across multiple models.

**Functions:** `collage`, `draw_figure_to_buffer`, `plot_confusion_matrices`

*📁 File: `src/dl_techniques/utils/visualization.py`*

## Visualization_Manager

### utils.visualization_manager
Visualization Manager Module

**Classes:**
- `VisualizationConfig`
- `VisualizationManager`

**Functions:** `get_save_path`, `save_figure`, `create_figure`, `plot_matrix`, `plot_history` (and 2 more)

*📁 File: `src/dl_techniques/utils/visualization_manager.py`*

## Vqa_Dataset

### datasets.vqa_dataset
Vision Question Answering (VQA) Data Processor for nanoVLM Training.

**Classes:**
- `BaseTokenizer`
- `SimpleCharTokenizer`
- `VQADataProcessor`
- `VQADataSequence`

**Functions:** `create_vqa_dataset`, `load_cauldron_sample`, `load_cauldron_from_json`, `encode`, `decode` (and 5 more)

*📁 File: `src/dl_techniques/datasets/vqa_dataset.py`*

## Weight_Transfer

### utils.weight_transfer
Generic layer-by-layer weight transfer for Keras 3 models.

**Classes:**
- `TransferReport`

**Functions:** `load_weights_from_checkpoint`, `num_loaded`, `num_shape_mismatch`, `summary_string`

*📁 File: `src/dl_techniques/utils/weight_transfer.py`*

## Yolo_Decode

### utils.yolo_decode
YOLOv12 prediction decoder — standalone utilities for inference / mAP eval.

**Functions:** `make_anchors_np`, `decode_dfl_logits`, `dist_to_xyxy`, `decode_predictions`, `nms_per_class`

*📁 File: `src/dl_techniques/utils/yolo_decode.py`*