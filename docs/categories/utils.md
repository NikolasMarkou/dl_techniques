# Utils

Utility functions and helpers

**59 modules in this category**

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

## Bias_Free_Denoiser

### applications.bias_free_denoiser

*📁 File: `src/dl_techniques/applications/bias_free_denoiser/__init__.py`*

## Bounding_Box

### utils.bounding_box
Object Detection IoU Utility Module

**Functions:** `bbox_iou`, `bbox_nms`

*📁 File: `src/dl_techniques/utils/bounding_box.py`*

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

### applications

*📁 File: `src/dl_techniques/applications/__init__.py`*

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

## Geometric

### applications.geometric

*📁 File: `src/dl_techniques/applications/geometric/__init__.py`*

### applications.geometric.examples
Comprehensive examples of what you can build with the spatial layers.

**Functions:** `build_point_cloud_classifier`, `build_point_cloud_segmentation`, `build_3d_scene_understanding`, `build_vision_language_model`, `build_multi_sensor_fusion` (and 8 more)

*📁 File: `src/dl_techniques/applications/geometric/examples.py`*

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