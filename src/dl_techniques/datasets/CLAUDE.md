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
- `byte_lm.py` — byte-level packed-CLM primitives for `vocab_size=256` models
  (`text_to_byte_ids`, `byte_ids_to_text`, `pack_byte_windows`,
  `build_byte_clm_dataset`, `estimate_byte_clm_steps_per_epoch`). Consumes the
  raw UTF-8 strings `nlp.py`'s `load_wikipedia_train_val` already returns —
  bytes are derived downstream of text exactly as tokens are. It is the byte
  sibling of `train.common.nlp`'s tiktoken pipeline, which has **no** byte
  path; `estimate_byte_clm_steps_per_epoch` mirrors
  `train.common.nlp.estimate_clm_steps_per_epoch`'s contract in byte units
  rather than extending it, because `dl_techniques` must not import `train`.
  `DEFAULT_AVG_BYTES_PER_ARTICLE = 3054` and
  `DEFAULT_WIKIPEDIA_TOTAL_BYTES = 19_567_594_259` are MEASURED over all 41
  staged Wikipedia Arrow shards, not inherited from the 440-tokens/article
  heuristic.
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
- `document_restoration/` — DocRes document-restoration data utilities:
  - `dtsprompt.py` — the five DTSPrompt generators (`deshadow_prompt`,
    `appearance_prompt`, `deblur_prompt`, `binarization_prompt`,
    `dewarp_prompt`) built on ONE shared `estimate_background` primitive.
    **numpy + scipy only — this module must not import `cv2` or `skimage`**
    (neither is declared in `pyproject.toml`; both merely happen to be
    installed). A suite guard enforces it; the tests may and do import them as
    the OpenCV parity oracle.
  - `tasks.py` — the single `TASKS` table binding each task name to its prompt
    generator, supervised output-channel count, loss name and post-processing
    mode. Nothing else in the tree branches on a DocRes task string.
- `document_rectification/` — DocScanner **dewarping** data utilities. A sibling
  of `document_restoration/`, not part of it: that package owns DocRes's
  photometric tasks (`(image, prompt, restored)`), this one owns geometric
  ground truth (`(image, f_gt, mask)`). The shared piece — `dtsprompt`'s
  `base_coordinate_grid` — is imported, not copied.
  - `synthetic_warp.py` — the synthetic warped-page generator. **numpy + scipy
    only at module scope**, same house rule as `dtsprompt.py`, plus `PIL`:
    Pillow is declared only in the `data` extra, so the single file-touching
    helper `load_rgb` imports it lazily. A suite guard enforces all three.
    Its warp is a composition of **closed-form-invertible** primitives (shear
    couplings, one homography, one fit affine), so the backward map `f_gt`
    (rectified grid -> distorted pixel coords, channel 0 `x`) and the forward
    map `g` are BOTH exact and neither is ever fitted. Read the module
    docstring before changing the warp family: the usual "render the image by
    gathering through the map you supervise on" recipe pins down the OPPOSITE
    direction, and emitting it as `f_gt` is a silent inversion with no shape,
    dtype, unit or range symptom.
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
The `tests/test_datasets/` tree is FLAT for top-level modules — `byte_lm.py` is tested by
`tests/test_datasets/test_byte_lm.py`, mirroring `test_masked_patches.py` and friends.
