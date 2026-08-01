# Datasets Package

Data loading, generation, and preprocessing utilities for various domains.

## Structure

### Top-level Modules
- `universal_dataset_loader.py` — Unified Hugging Face Hub streaming loader for multiple dataset types (text, image, audio)
- `simple_2d.py` — Synthetic 2D dataset generators (classification/regression)
- `patch_transforms.py` — Image patch extraction and transformation utilities
- `tabular.py` — Tabular dataset utilities
- `sut.py` — SUT-Crack dataset loader (TF-optimized, vectorized processing)
- `vqa_dataset.py` — VQA dataset processor for nanoVLM training (supports The Cauldron format)
- `nlp.py` — Wikipedia / HF text dataset helpers (`load_wikipedia_train_val`, packed-CLM article counts, shard utilities)
- `bdd100k_video.py` — BDD100K video dataset loader
- `synthetic_drone_video.py` — Synthetic drone-video sequence generator
- `pusht_hdf5.py` — PushT robotics HDF5 dataset loader

### Subpackages
- `arc/` — ARC (Abstraction and Reasoning Corpus) dataset support:
  - `arc_converters.py`, `arc_keras.py`, `arc_utilities.py`
- `vision/` — Computer vision dataset loaders:
  - `coco.py` — COCO dataset, `coco_multitask_local.py` — local multi-task COCO
    variant, `imagenet.py` — ImageNet, `common.py` — shared utilities
  - `masked_patches.py` — `make_masked_patch_map_fn`, the per-sample
    `element_map_fn` for masked-image-modelling objectives
  - `multi_crop.py` — `make_multi_crop_map_fn`, the DINO multi-crop
    (2 global + N local views) `element_map_fn`; local views are rendered at the
    global pixel resolution, so `local_crop_size != global_crop_size` raises
    `NotImplementedError` (positional-embedding interpolation is not implemented)
- `time_series/` — Time series dataset framework:
  - `base.py` — Base dataset class, `config.py` — dataset configuration
  - `generator.py` — Data generators, `pipeline.py` — preprocessing pipelines
  - `normalizer.py` — Normalization strategies, `utils.py` — helpers
  - Domain datasets: `favorita.py`, `m4.py`, `long_horizon.py`

## Conventions

- `__init__.py` is empty — import from submodules directly
- Time series datasets follow a base class pattern with config-driven setup
- Vision datasets provide standard train/val/test splits

## Testing

Tests in `tests/test_datasets/` (if present) or integration tests within model test suites.
