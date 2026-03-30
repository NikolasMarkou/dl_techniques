# Datasets Package

Data loading, generation, and preprocessing utilities for various domains.

## Structure

### Top-level Modules
- `universal_dataset_loader.py` — Unified loader interface for multiple dataset types
- `simple_2d.py` — Synthetic 2D dataset generators (classification/regression)
- `patch_transforms.py` — Image patch extraction and transformation utilities
- `tabular.py` — Tabular dataset utilities
- `sut.py` — SUT-Crack dataset loader (TF-optimized, vectorized processing)
- `vqa_dataset.py` — VQA dataset processor for nanoVLM training (supports The Cauldron format)
- `universal_dataset_loader.py` — Hugging Face Hub streaming loader (text, image, audio)

### Subpackages
- `arc/` — ARC (Abstraction and Reasoning Corpus) dataset support:
  - `arc_converters.py`, `arc_keras.py`, `arc_utilities.py`
- `vision/` — Computer vision dataset loaders:
  - `coco.py` — COCO dataset, `imagenet.py` — ImageNet, `common.py` — shared utilities
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
