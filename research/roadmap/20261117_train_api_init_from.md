# Phase 4g: start a run from an earlier run's weights (`init`), for every task and family

Status: shipped (plan 0fd54b91, branch `train-api/phase-4g`, 2026-11-18), its GPU gate run (numbers in `src/train/README.md`); extended by Phase 4h (plan 5c6cc105, branch `train-api/phase-4h`, 2026-11-23: pretrain as source and target, the `beit` and `energy_transformer` classifiers; code done, its GPU gate pending), whose changes close Section 0. Sections 1 to 11 are the design as written before the plan; Section 0 says what shipped where it differs, each change named by the plan's decision (D-NNN of plan 0fd54b91). The user-facing rules and the shipped refusal messages are in `src/train/README.md` ("Starting from an earlier run: `init`"), the contributor's side in `src/train/AGENTS.md`. Placement: after Phase 4f (`depth_estimation`, branch `train-api/phase-4f`) and before the language tasks. Origin: the user asked whether a trained depth model can be reused to fine-tune at a higher resolution, then asked for a general phase instead of a depth-only patch (4f D-020).

Evidence. The numbers in this design were measured while Phase 4f was planned, by three read-only passes: a shape matrix of 156 rows (each family's smallest variant built at a source setting and at a changed size, head, channel count, lookback, horizon, variant or task, its weights compared path by path and shape by shape; CPU only, about 6 minutes, nothing written under `results/`), a read of the existing warm-start code (the `checkpoint` key, `load_weights_from_checkpoint`, the legacy trainers' flags, what a run directory holds), and a pass over cross-task pairs, split overlap, bars, the optimizer and prior art. Those notes are planning scratch outside the repository, so every fact this design rests on is stated where it is used; a `file:line` reference points into `src/` as it was at writing time.

Everything marked UNMEASURED was to be settled by Step 1 before any code was written; Section 0 says how each was settled.

## 0. As shipped

Decision map (design id, plan decision): 4g-D1 D-002, 4g-D2 D-003, 4g-D3 D-004, 4g-D4 D-005, 4g-D5 D-006, 4g-D6 D-007, 4g-D7
D-008, 4g-D8 D-009, 4g-D9 D-010, 4g-D10 D-011, 4g-D11 D-012, 4g-D12 D-013, 4g-D13 D-014, 4g-D14 D-015, 4g-D15 D-016, 4g-D16
D-020, 4g-D17 D-021, load cost D-022. Kept as designed: 4g-D2, 4g-D5, 4g-D8, 4g-D14, 4g-D16, 4g-D17. Changed:

- **4g-D1 (D-002).** The named refusal of `checkpoint` reaches the command line, a `config=` file and a YAML mapping through the
  one unknown-key message builder. The Python keyword no longer exists, so `train_<task>(checkpoint=...)` is a plain `TypeError`
  (a `**kwargs` catch to name it would be a compatibility path).
- **4g-D3 (D-004).** "Resolves `pretrained` to false" is split in two: under `init` the depth factory gets `pretrained=False` and
  the official-weights check is skipped, while the learning rate and the DPT head's 10x follow the warm start (5e-6). Any explicit
  depth `model_args.pretrained` under `init` is refused; a classifier refuses any `pretrained` other than false.
- **4g-D4 (D-005, D-030, D-031).** Every subclassed model roots its weight paths at its own instance name
  (`vision_transformer_pico` against `vision_transformer_tiny`, `res_net` against the segmenter's `backbone/`), so on raw paths
  the vit grow and both classifier-to-segmenter pairs were refused. Weights are keyed by their path relative to the model: one
  leading component is dropped on each side when it is the model's root (`<model.name>/` wherever a path has it, or the component
  every weight shares: accunet builds under `acc_u_net/`, D-045), and nothing inside is renamed. xlstm's auto-named inner `rnn` and cell
  layers still differ in the production order (target built, then the source loaded beside it), so `TransferSpec.positional`
  pairs weights by order inside the top-level blocks its pattern matches (`[ms]lstm_block_\d+`: every block at any
  `num_layers` and `mlstm_ratio`, D-050), only when the counts, shapes and dtypes all agree.
- **4g-D6 (D-007, D-032, D-033).** `TransferSpec` and `CrossTask` live in `src/train/api/registry.py`, which imports no
  third-party package, so `python -m train help` stays free of numpy. `ModelSpec` does not validate "a spec or a reason" (that
  broke `ModelSpec` constructions in unrelated tests); the census `tests/test_train/test_api/test_init_from_families.py` requires
  it of every row, and a row with neither is refused at run time. The 7 forecasting rows declare it once per `FamilyRow`. 23 rows
  declare a spec and the 2 pretrain rows a reason (4h: all 27 rows declare a spec, `transfer_reason` deleted, plan 5c6cc105 D-011). A seventh field, `positional`, is xlstm's (above).
- **4g-D7 (D-008).** The head identity is compared after the claim, from the two records (a folder target's class names exist
  only once its data is read); input scaling and `fixed` geometry are compared before the claim, against what the task's
  `init_identity(config, context)` hook says this run will record.
- **4g-D9 (D-010, D-018).** The pins reach the build through `AutoContext.pinned`, read only by `recipes.merge_model_args`
  (laid over the computed values; a differing user value refused, an equal one accepted). They are read from the source's
  `config.json` `used.model_args` only. On a resume they come from the run's own `task.json` `init` block.
- **4g-D10 (D-011, D-034).** The source is read first among the pre-claim checks. "Pre-claim code imports no TensorFlow" was
  never true of the existing pre-claim phase (`TaskTrainer.run` imports Keras first); what holds: a pre-claim refusal exits 2,
  never reaches `setup_gpu` and claims nothing, `read_source` loads no TensorFlow, Keras or numpy for an absolute path, and `help`
  stays TensorFlow-free.
- **4g-D11 (D-012, D-037, D-038).** Each task counts its own overlap in a second hook, `init_overlap(source, config, data)`
  (default `unknown`), never in the engine. The `split` block is `relation`, `unit`, `n_select_overlap`, `n_test_overlap`,
  `reason`, `keys`; the units are rows (classification, segmentation), files (denoising), windows (forecasting) and scenes
  (depth). Forecasting rebuilds the source's train segment from its `config.json`: another seed, series length, patterns or
  realizations is `unknown` (D-045), because the two banks can share series (the deterministic patterns barely move with the
  seed, a shorter series is the start of a longer one), and calling it different data would report 0 for them; other fractions
  count the windows that start inside the source's train segment. The bank has no noise key: the generator never read it (D-044).
  Imagenette against ImageNet, and ImageNet itself (held out per class), are `unknown`. The CIFAR-10 count is re-derived by the
  test's own numpy oracle: a seed 42 source trained on 4,531 of a seed 43 run's 5,000 selection rows.
- **4g-D12 (D-013, D-039).** Under `init` the epoch-zero evaluation is one function for every task
  (`init_from.evaluate_loaded`, "Evaluate of the loaded weights before fit"), so the five "Untrained evaluate BEFORE fit" sites
  are untouched and a run without `init` logs as before. The bar term is `init_from.beats_loaded`: strictly above the loaded
  value by more than `forecast_run.tie_band(1)`, 16 float32 steps, carried by `Evaluation.loaded`, which `evaluate_and_report`
  returns (D-050).
- **4g-D13 (D-014, D-040).** Five record kinds, not six (4h: the pretrain record is the sixth, plan 5c6cc105 D-009), and one generic `predictor.OPTIONAL_RECORD_KEYS`; the block is added
  once, at the `task.json` write site. The summary takes it in `TaskTrainer._summary_head` (`init`, `epoch_zero_of`, the three
  flags, also for a diverged run) and adds `loaded_success_value` for classification, segmentation and denoising.
- **4g-D15 (D-016, D-017).** A resume without a backup was already refused, so the re-transfer before the first backup and its
  sha256 refusal row (Section 1.4, last row) are dropped: a resume never reads the source. `init` stays a compared resume key.
- **Not in the design.** The denoising bias-free probe stays on the fresh model before the copy (D-019). A classifier that seeds
  a segmenter is trained with `transform_args.normalize=unit`, because a segmenter never standardizes and the scaling refusal
  stays (D-035).
- **Section 1.4** holds the design's message forms; the shipped wording is quoted in the README.
- **Section 4.** Step 1's measurements were replaced by setup only (D-028): what the code needed was settled by the steps' own
  tests (paths, `SeedGenerator` state, pins, epoch-zero metrics, split rows), the rest by the gate. The gate gained a denoising
  and a forecasting arm (D-023). 18 mutants were planted (M1 to M18), each failing its guard before the code was restored.

**Phase 4h (plan 5c6cc105; D-NNN below are that plan's).** User rulings U1 to U4 of 2026-11-19, as shipped:

- **Pretrain as source and target (4g-D17 lifted).** The pretrain rows are renamed `et_mim` and `beit_mim` with no alias (D-002), and a
  pretrain run writes the sixth record kind (D-009), whose `predict` and `export` answer the backbone's patch-token embeddings (D-003).
  Each pretrain row declares a spec: its decoder is the head, keyed by the tokenizer file's sha256 (`HEAD_KEYS["pretrain"]`), with
  `channels`, `input_size` and `patch_size` fixed and no sources, so it takes its own family only (D-008). The classifiers `beit` and
  `energy_transformer` take the pretrain rows as sources, the decoder dropped (D-010, D-027, D-039): 6 declared cross-task pairs, 27 rows,
  every one a spec, and `transfer_reason` deleted (D-011). A cross-task source pins the keys its record holds (`patch_size`, D-036); the
  identity refusal names every differing `fixed` key (D-038). Under `init` the pretrain bar keeps `min(baseline, epoch zero)` with the
  loaded weights as the epoch-zero term (D-012).
- **Three rules, keys by regex.** `resample` keys full-match the weight path (D-005); `beit_relpos_cubic` is unilm's resize, checked
  against unilm's own lines run verbatim (D-004, D-019; refused below a 3x3 grid, D-020); `et_pos_refused` names both grids and
  resamples nothing (D-006); BEiT's optional absolute table takes `vit_pos_bicubic` (D-007, D-031).
- **Two premises of 4g-D17 were false.** "No local reference to check it against": unilm's `beit/run_class_finetuning.py` is the
  reference, run verbatim into a test fixture (D-004). "Nothing in 4g's engine changes for it": `HEAD_KEYS`, `GEOMETRY_KEYS`,
  `GEOMETRY_NAMES`, the resample lookup and the cross-task pins changed (D-005, D-008, D-036, D-038).

---

## 1. Goal and user-facing behaviour

### 1.1 Goal

One key, `init`, on every task's config, names an earlier `python -m train` run directory. The new run builds its own model from its own config, then copies the source run's weights into it before `compile`, under one rule set shared by all tasks: every weight of the new model is copied, resampled by a rule its family declares, or left fresh because its family declares it a head (or a new block of a deeper variant) and the records say it must be; every weight of the source is used or dropped by a declared rule. Anything else is refused by name, before any weight is copied. The run records where its weights came from, and its success bar certifies that the fine-tune improved on the loaded weights, not that an untrained model learned.

### 1.2 The key

| Key | Type | Default | Meaning |
|---|---|---|---|
| `init` | run directory (str or `Path`; repo-relative paths resolve as `out` does) | `null` | The source run. Its `task.json`, `config.json` and `best_model.keras` are read; its weights seed this run. |

There is no other key. Every choice the old loader exposed (`skip_prefixes`, `strict`, which layers may stay unloaded) is now either a family declaration (Section 2, 4g-D6), a rule derived from the two records (4g-D7), or a refusal. `checkpoint` is removed (4g-D1).

Python: `train_image_classification(out="results/vit_c100_64", model="vit/vit_pico", data={"dataset": "cifar100"}, transform_args={"image_size": 64}, init="results/vit_c10_32")`.

### 1.3 One worked example per kind of reuse

Every example is a full command; counts in parentheses are the measured shape-matrix numbers for the smallest variant, which the implementation re-derives by running and never copies.

1. **Same task at a higher resolution (the depth case).** A depth run trained at 252 px seeds a run at 364 px (a multiple of 14):
   ```bash
   python -m train depth_estimation out=results/depth_vits_364 model=depth_anything/vit_s \
       init=results/<the 252 px depth run> transform_args.image_size=364
   ```
   All weights are copied unchanged (235 of 235 weights equal by path and shape between `vit_s` built at 112 px and at 168, 224, 196x252 and 518 px; a 112 px model's saved weights load unchanged into a 224 px build, while the 112 px model itself fails on a 224 px image): the DINOv2 position table is stored at the official 37x37 grid and resampled inside the model at call time (`DINOv2PositionEmbedding` keeps the (1, 1370, 384) table at every image size; no other family resamples a table). The target is built with `model_args.pretrained` resolved to false, because every encoder weight is overwritten (4g-D3). The 4f head multiplier (`optimizer_options`, head at 10x) applies as it does from the official weights (4g-D14). The bar's epoch-zero term is the source weights scored on 364 px selection crops (4g-D12).

2. **Same task with a new head (classes).** A ViT trained on CIFAR-10 at 32 px seeds a CIFAR-100 run at 64 px; the head is reset (10 to 100 classes: 78 of 80 weights equal, the classifier kernel and bias differ), the position table is resampled from an 8x8 to a 16x16 grid (65 to 257 rows of 192; with the 32 px recipe pinned, 79 of 80 weights are equal and only the table differs), and the patch size stays 4 because the source's architecture is pinned (4g-D9; without the pin the recipe would switch to patch 16 and also lose `patch_embed`, whose kernel goes from 4x4x3x192 to 16x16x3x192: 78 of 80 equal):
   ```bash
   python -m train image_classification out=results/vit_c100_64 model=vit/vit_pico \
       data.dataset=cifar100 transform_args.image_size=64 init=results/vit_c10_32
   ```
   Same task, new head, segmentation: an `oxford_pet` ConvUNext (3 classes) seeds a folder dataset of 21 classes; only `final_output` is reset (156 of 158 copied):
   ```bash
   python -m train image_segmentation out=results/convunext_voc model=convunext/tiny data.format=folder \
       data.train_images=<dir> data.train_masks=<dir> data.num_classes=21 init=results/convunext_pet
   ```
   Same task, new head, forecasting: a horizon change resets the head where the head is the only horizon-sized weight (tirex `quantile_head`, xlstm `point_head`, mdn `mdn_mus`/`mdn_sigmas`, prism `temporal_projector`). A channel change of the input is a stem change, refused except for `timestral` (channel-free: all 76 weights equal from 1 to 3 channels). Denoising has no head change: its output channels equal its input channels, so a channel change also changes the stem and is refused (3 to 1 channels changes the stem and the output kernel in every denoising family, e.g. scunet 146 of 148 equal).

3. **A deeper variant where shapes allow.** `vit_pico` to `vit_tiny` (80 of 80 source weights copied, 72 new weights in 6 new blocks); `resnet18` to `resnet34` (102 of 102, 80 new; segmentation `resnet_seg` 124 of 124):
   ```bash
   python -m train image_classification out=results/vit_tiny_c10 model=vit/vit_tiny data.dataset=cifar10 \
       init=results/vit_c10_32
   ```
   The new blocks start fresh only because the family declares its block pattern (4g-D6 `grow`) and every one of the target's unloaded weights sits in a block the source does not have. A width change (`convnext_v2` atto to femto 1 of 140, `convunext` tiny to small 1 of 158) or a block-type change (`cliffordnet` nano to lite, 12 differing kernels) is refused.

4. **Cross-task backbone: classification into segmentation.** ConvNeXt V1 trained on CIFAR-10 (strides 2, the cifar10 recipe) seeds `convnext_seg` on `oxford_pet`; the classifier's 98 backbone weights land under the segmenter's `backbone/` prefix, `classification_head` is dropped, `segmentation_head` is fresh (98 of 102 classifier weights equal by path and shape once the prefix is dropped):
   ```bash
   python -m train image_segmentation out=results/convnext_seg_pet model=convnext_seg/cifar10 \
       init=results/convnext_c10
   ```
   ResNet: `resnet/resnet18` into `resnet_seg/resnet18` works only from a classifier built with the imagenet stem (100 of 102; the 32 px recipe resolves `stem_type="cifar"`, `recipes.py:552`, which the segmenter never builds): `python -m train image_classification ... model=resnet/resnet18 model_args.stem_type=imagenet` for the source, then `init=` as above.

5. **Segmentation ConvUNext into biased denoising.** `convunext` (segmentation) and `convunext_biased` (denoising) are the same builder with `use_bias=True` (158 of 158 equal at 3 classes and 3 channels; the reverse pair is declared too):
   ```bash
   python -m train image_denoising out=results/convunext_denoise model=convunext_biased/tiny \
       data.train_dirs='["<dir>"]' data.val_dirs='["<dir>"]' init=results/convunext_pet
   ```
   Across tasks the head is always reset (`final_output`): class logits and denoised pixels are not the same quantity even at equal width (4g-D7). The bias-free denoisers (`bfconvunext`, `bfunet`, `bfcnn`) share 0 to 7 weights with either side and are refused (`bfconvunext` shares 7 channel-adjust kernels of 158).

6. **Forecasting at a new lookback, where the family allows.** `tirex`, `xlstm`, `ets` and `timestral` keep every weight across 96, 168 and 336 (all 47, 25, 1 and 76 weights equal from 168 to 96 and to 336):
   ```bash
   python -m train time_series_forecasting out=results/tirex_336 model=tirex/tiny data.format=builtin \
       data.lookback=336 init=results/tirex_168
   ```
   `nbeats` (`dense1`, the basis matrices), `prism` (`temporal_projector`) and `mdn` (`dense_0`) size weights by the lookback (168 to 336 leaves nbeats 30 of 36 weights equal, prism and mdn 27 of 28) and are refused before the run directory is claimed.

### 1.4 Refused by name (the message forms; `<...>` is filled from the records)

| Case | Evidence | Message (ConfigError, exit 2) |
|---|---|---|
| `checkpoint=` given | 4g-D1 | `Unknown key 'checkpoint': it was replaced by 'init', which takes a run directory (python -m train <task> help)` |
| `init` names a file | 4g-D2 | `Invalid value for 'init': <path> is a file; give the run directory that holds it (init reads task.json, config.json and best_model.keras)` |
| no `task.json` (legacy trainer run; a pretrain run before Phase 4h) | a legacy trainer writes no `task.json`, nor did pretrain before 4h (plan 5c6cc105 D-009; its Step 6 reworded the sentence) | `Invalid value for 'init': <run>/task.json does not exist; a source is a python -m train run, which writes it before its first epoch (legacy trainer runs and runs made before their task wrote one have none)` |
| no `best_model.keras` | `task.json` is written before `fit`, the model only at an epoch end | `Invalid value for 'init': <run> holds no best_model.keras (the run ended before its first epoch)` |
| target task is `pretrain` (lifted by Phase 4h, Section 0) | 4g-D17 | `Invalid value for 'init': pretrain does not start from earlier weights: it writes no record, and no family of another task shares the BEiT or Energy Transformer trunk (follow-up: the pretrain record)` |
| family pair not declared | beyond the four declared pairs no registered family shares an architecture with another task's | `Invalid value for 'init': <src family> (<src task>) cannot seed <dst family> (<dst task>); <dst family> takes weights from <its declared sources>` |
| geometry the family cannot change | nbeats `dense1` 168x64 to 336x64, prism `temporal_projector`, mdn `dense_0` | `Invalid value for 'init': nbeats sizes dense1 and its basis matrices by data.lookback; the source trained at 168, this run asks for 336 (families that keep every weight across a lookback change: tirex, xlstm, ets, timestral)` |
| a smaller ViT grid | 4g-D8 | `Invalid value for 'init': vit pos_embed would shrink from a 16x16 to an 8x8 grid; only a larger grid is resampled (bicubic, no antialiasing)` |
| a pinned argument changed | 4g-D9 | `Invalid value for 'model_args.patch_size': the source was built with patch_size=4 and init keeps the source's architecture` |
| cross-task stem mismatch | 96 of 102 equal; the stem and `downsample_0_conv` kernels are 4x4 against 2x2 | `Invalid value for 'init': convnext_seg builds strides 2 (2x2 stem and downsample_0 kernels); the source classifier used strides 4 (the imagenette recipe)` |
| input scaling differs | 4g-D7 | `Invalid value for 'init': the source standardized its input and this run leaves it in [0, 1] (transform_args.normalize)` |
| denoising data range differs | the legacy denoiser trainer refuses a source without `data_range` `[0,1]`; `checkpoint=` has no such gate | `Invalid value for 'init': the source record has data_range <x>, this run trains on [0,1]` |
| `init` with `model_args.pretrained=true` given | 4g-D3 | `Invalid value for 'model_args.pretrained': init replaces every weight, so the official weights would be read and overwritten` |
| `model_args.pretrained=<a path>` in any task | classification `model_args` can forward a path to a factory today, untested | `Invalid value for 'model_args.pretrained': a path is not offered; init takes a run directory` |
| any weight neither copied, resampled, reset, grown nor dropped (after the claim, before any copy) | the layer-name loader skips a whole layer on one differing weight and returns normally; a `pretrained=<resnet50>` load restored 1 layer with 9 mismatches and raised nothing | `Invalid value for 'init': <n> of <m> weights of <dst> have no source weight of their shape and no declared rule: <first 5 paths with both shapes> (+<k> more); nothing was copied` |
| a resume before the first backup whose source changed | 4g-D15 | `Invalid value for 'init': <run>/best_model.keras changed since this run started (sha256 <old> recorded, <new> now); the resume would start from other weights` |

---

## 2. Decisions

Numbered 4g-D1 to 4g-D17 here; the executing plan re-numbers them as its own D-NNN entries.

**4g-D1. Replace `checkpoint` outright with one top-level key `init` (a run directory), no alias.** A single key whose value carries the whole description (the run directory and its records), **at the cost of** breaking every existing `checkpoint=` command and script, and of no per-run transfer options. Evidence: `checkpoint` has one reader (`run_sequence.py:537`), copies same-named top-level layers, refuses only zero loaded layers, records nothing; it was never exercised for segmentation, forecasting or pretrain; a classifier checkpoint loads no layer into a segmenter ("No overlapping layers"). The user's rule is to replace a wrong behaviour outright. A group `init.source` plus options was rejected: every option considered (skip prefixes, a head policy, a strictness level, a leakage override) became a family declaration, a record-derived rule or a refusal, so a group would hold one key. The removed key gets a named refusal (one line in the strict-config unknown-key path), because the closest-key suggestion would otherwise point at `checkpoint_args`; this is a refusal, not a compatibility path.

**4g-D2. The source is a run directory, read through its `task.json` and `config.json` before the claim and its `best_model.keras` through `predictor.load_model` (compile=False) after it, at the cost of refusing bare `.keras` files, legacy trainer runs and pretrain runs.** A record gives the source's task, family, variant, geometry and head identity; `config.json` gives its resolved model arguments and split keys. `predictor.load_model` already imports the record's family, loads with `compile=False`, checks the model against the record and runs a zero forward; reusing it keeps one door for reading a run. The old loader's `compile=True` load raises `TypeError` on 4 of 6 real forecasting runs (tirex, prism quantile, timestral: `DecodedMAE`; mdn: `method`), which `_load_checkpoint` does not catch (it catches `ValueError` only). A bare `.keras` has no split identity and no geometry; a model from outside the repo goes through its family's own `pretrained` argument, not through `init`. The `best_model.keras` of a finished run holds the best weights, as does `final_model.keras`, so no file choice is offered.

**4g-D3. The official-weights door is not folded into `init`; `model_args.pretrained` stays a boolean meaning "the family's official foundation weights" (depth today), a path value is refused in every task, and `init` with an explicit `pretrained=true` is refused, at the cost of two keys that both touch the starting weights.** Folding (`init=official:dinov2-small`) would move 4f's pre-claim weights check and its lr rule (5e-6 with the head at 10x from a pretrained start, 2e-4 from scratch) into `init` right after 4f ships, and the official load fills only the encoder, which is a different completeness rule. A second door is open today: a path value can reach `load_pretrained_checkpoint` through classification `model_args` (untested); refusing a path closes it. With `init`, the depth recipe resolves `pretrained` to false (the source overwrites every encoder weight, and the build no longer reads 88 MB to 1.2 GB of official weights for nothing; 4f D-007 trade-off).

**4g-D4. The transfer unit is the individual weight, keyed by its Keras variable path, with family-declared prefix maps across tasks, at the cost of a new engine beside `load_weights_from_checkpoint` and of exposure to auto-named inner layers (xlstm).** Layer-name matching loses whole monolithic layers on one differing weight: `beit_backbone` (224 weights) on a table change, `depth_head` (31 weights) on an output-channel change, `mdn_core`, `scu_net`, each nbeats stack, prism `temporal_projector`; it sees no nested backbone (resnet to resnet_seg loads 0 of 100; the same helper pointed at the nested model loads 9 of 9 layers). Paths are stable: 24 of 25 families give identical paths on a second build in one process; xlstm's auto-named `rnn`/`m_lstm_cell` drift to `rnn_2`/`m_lstm_cell_1` unless the session is cleared. The run builds the target before it loads the source, so the target gets first-build names; Step 1 measures whether a real xlstm source and target agree in the real order, and if not the xlstm spec declares a positional match inside its named top-level layers (the measured fallback that works: a position-wise assign transferred all 25 xlstm weights), never a name normalization (a recorded lesson: an oracle must not normalize away name disambiguators). `load_weights_from_checkpoint` stays for the legacy trainers; train.api no longer calls it.

**4g-D5. Complete-or-raise in both directions, every check before the first copy, no `skip_mismatch` and no user override, at the cost of refusing partial transfers a user might accept (convnext strides 4 into `convnext_seg`: 96 of 102 equal, 2 stem kernels differ).** The current default returns normally with the trunk random in measured cases: BEiT at 64 px or patch 8 lists `beit_backbone` as a mismatch and returns; the ViT 32 to 64 px, 10 to 100 class transfer leaves `pos_embed` and the head at init without raising; an earlier `pretrained=<resnet50>` load restored 1 layer, 9 mismatched, raised nothing. A recorded lesson of this repo: complete-or-raise both ways, check everything first, a raise copies nothing, only role-identified layers stay fresh and the exemption is logged. Prior art: HF raises on a size mismatch unless `ignore_mismatched_sizes`, which then skips any mismatch, the trunk case included; Keras `skip_mismatch` turns failures into one warning (both documentations read 2026-11-17). Copy the default, not the opt-in.

**4g-D6. A `TransferSpec` on each family's `ModelSpec` row, census-tested against the built models, at the cost of one more declaration per family (23 rows: 6 classification, 4 segmentation, 5 denoising, 7 forecasting, 1 depth) and a census test of about 300 s.** Fields:
- `head`: path prefixes of the final linear (classification `classification_head`, segmentation `final_output` / `output_conv` / `segmentation_head`, denoising `final_output` / `final_conv` / `output_conv`, depth `depth_head/output_conv2_2`, forecasting per family).
- `resample`: path pattern to rule name. One rule exists in 4g: `vit_pos_bicubic` for `pos_embed/pos_embedding` (4g-D8).
- `grow`: a regex of depth-indexed blocks (vit `transformer_layer_\d+`, resnet `stage\d+_block\d+`, resnet_seg the same under `backbone/`), or none.
- `pinned`: factory arguments the target takes from the source (4g-D9).
- `fixed`: geometry keys the family cannot change, refused before the claim (nbeats `lookback`, `horizon`, `channels`; prism `lookback`, `channels`; mdn `lookback`; tirex, xlstm `channels`; the 3-channel stems of every image family under `channels`; each one a measured weight difference on the smallest variant).
- `sources`: cross-task pairs the family accepts, each a `(task, family)` key with a source-to-target prefix map and the source prefixes dropped: `resnet_seg` from classification `resnet` (`""` to `backbone/`, drop `classification_head`), `convnext_seg` from `convnext_v1` (same), `convunext_biased` from segmentation `convunext` (identity map, `final_output` is the head), `convunext` from denoising `convunext_biased`.

A new family declares `transfer=TransferSpec(...)` or `transfer=None` with a reason string; `None` makes `init` refuse that family by name. The census test fails on a row of an `init`-accepting task that declares neither. It also builds each family's smallest variant at the source geometry and at each change its spec names (size, head, lookback, horizon, grow), runs the engine's check only, and requires the verdict to equal the declaration: a `fixed` key must fail at the weight level, an undeclared change must pass. That keeps the pre-claim refusals honest against the real models (the 4f matrix took about 6 minutes on CPU for 156 rows).

**4g-D7. Head reset and input checks are derived from the two records, not chosen by the user, at the cost of resetting a head whose classes are the same set in another order.** The head is copied only when the task's head identity is equal: classification and segmentation `class_names` (ordered list), denoising `channels`, forecasting `horizon`, `head`, `quantile_levels` and `channels`, depth always (one output). Otherwise the target head is fresh and the source head is dropped; across tasks the head is always reset. A same-shape head of other classes is the silent case of today (today a same-shape head is copied). Input checks from the records: classification and segmentation `normalize` mode must be equal (mean and std recorded, not compared: a cross-dataset fine-tune legitimately changes them); denoising `data_range` must be equal (the legacy `[0,1]` gate is not applied to `checkpoint=` today).

**4g-D8. One resample rule: the ViT absolute position table, class row kept, grid resampled bicubic with torch's kernel (`torch_bicubic_resize_matrix`, align_corners false, no antialiasing), to a larger grid only, at the cost of refusing a smaller grid and of no comparison with timm.** Only ViT, Energy Transformer and BEiT carry grid-sized tables nothing resamples; ET and BEiT are pretrain families, out with pretrain (4g-D17; Phase 4h gives BEiT's tables unilm's rule and ET's a named refusal, Section 0); DINOv2 resamples at call and needs no rule. timm resamples with `antialias=True` (`resample_abs_pos_embed`); the in-repo matrix reproduces torch's bicubic without antialiasing and is pinned against a numpy reference (`tests/test_models/test_dino/test_the_dinov2_backbone_is_the_references.py`, `test_the_table_is_resampled_as_torch_does`). Upsampling is where antialiasing has no effect; downsampling without it aliases, so it is refused rather than done differently from timm. Measured while writing this design: neither `torch` nor `timm` is installed in `.venv`, and no ViT or BEiT checkpoint is on disk (only DINOv2 and Timestral weights under `/media/arxwn/data0_4tb/datasets/pretrained/`), so a timm parity check on a known checkpoint is not possible; Step 1 measures the rule's effect instead (a trained source scored at the new size).

**4g-D9. The target is built with the source's architecture: the family's `pinned` factory arguments come from the source's resolved model arguments, and a user value that differs is refused by name, at the cost of no patch-size, stem or stride change under `init`.** The classification recipes derive patch size (4 at 32 px, else 16, `recipes.py:534`), the ResNet stem (cifar at 32 px, `recipes.py:552`) and ConvNeXt strides (from the dataset, `recipes.py:290`) from size and dataset; without a pin, "same model at 64 px" changes `patch_embed` and the stem (measured: the ViT patch kernel 4x4 to 16x16, the ResNet stem 3x3 to 7x7, the ConvNeXt stem and `downsample_0_conv` 2x2 to 4x4). Pinned: vit `patch_size`; resnet `stem_type`; convnext_v1, convnext_v2 `strides`; the segmentation and depth families pin nothing. UNMEASURED: whether the source's `config.json` holds the resolved values (the BEiT run's holds `data.patch_size: 4`, a resolved value), else the `config.json` inside the `.keras` zip (TF-free to read). Step 1 settles it; a source that records neither is refused by name.

**4g-D10. Checks that need no TensorFlow run before the run directory is claimed; the weight-level check runs after the claim and before any copy, at the cost of a claimed directory (config.json and run.log) on a weight-level refusal, as today.** Before the claim: paths and files, both records, the task and family pair, `fixed` geometry, pinned arguments, head identity, input scaling, split identity (4g-D11), the source sha256. After: both models built, the complete-or-raise check, then the copy. Pre-claim code may not load TensorFlow because `setup_gpu` has not run (D-042 order); today a missing checkpoint leaves a claimed directory. The census (4g-D6) makes the pre-claim `fixed` declarations agree with the post-claim engine, so the post-claim refusal is the backstop, not the usual path.

**4g-D11. Split identity: each task declares the keys that decide its rows; a source whose training rows overlap this run's selection or test rows is NOT refused (user, 2026-11-17: "no refusal"): the run counts the overlap and marks its verdict as biased, at the cost of a verdict that can no longer certify generalization when the overlap is not zero.** Measured on CIFAR-10: source seed 42, new seed 43: 4,531 of the 5,000 new selection rows were trained on by the source; same seed, fraction and cap: 0; same seed with a source `max_samples=5000`: 4,500 of 5,000; fraction 0.1 against 0.2: 5,000 of 10,000; the builtin test split stays clean. Identity per task (all TF-free, read from `config.json` and, for folder and csv data, the fit sidecar's `data_fingerprint`):
- classification, segmentation builtin: (dataset, seed, validation_split, max_samples); folder and csv: (data_fingerprint, seed, validation_split, max_samples).
- denoising: (train_dirs, val_dirs, seed, val_seed, max_train_files); the source's training files are compared with this run's validation pool and counted, reusing the path comparison of the existing within-run overlap refusal (`denoising_data.py:236-262`), which itself stays a refusal for this run's own train and validation pools.
- forecasting: builtin (seed, patterns, realizations, series_length, select_fraction, test_fraction); csv (data_sha256, select_fraction, test_fraction). As shipped (D-045): another seed, series_length, patterns or realizations of the builtin bank is `unknown`, because the two banks can share series; another `data.format` is `different_data`; a csv of another sha256 or other value columns is `unknown`.
- depth: (root, train_split); MegaDepth's split is a pure function of these and not seed-keyed.
Relations recorded: `same_split`, `different_data` (other builtin dataset, another forecasting `data.format`; CIFAR-10 and CIFAR-100 are disjoint by their authors' statement, not measured here), `unknown` (folder data under other directories, file data of another fingerprint, a csv of another hash or other value columns, recorded with the reason). On the same data with another identity the run proceeds: where the rows are a pure function of the recorded keys (builtin datasets, csv and folder data through their fingerprint and seed, MegaDepth scenes, denoising file pools) it rebuilds the source's training rows from the source's `config.json` and counts how many of this run's selection rows and test rows are among them; the relation is then `overlap` with both counts, `results_summary.json` carries `selection_biased: true` when the selection count is above 0 and `test_biased: true` when the test count is, the success line printed at the end of the run says "selection-biased" (or "test-biased") next to the verdict, and the log names the counts and the source keys that would give 0 ("use the source's seed, validation_split and max_samples"). Where the source's rows cannot be rebuilt the relation is `unknown`, recorded with the reason, and the verdict is marked `overlap_unknown: true`. Nothing in this rule refuses a run.

**4g-D12. Under `init`, every task's bar gains the epoch-zero term, which is the loaded weights' score, so "met" certifies that the fine-tune improved on its starting weights; labels that say "untrained" are renamed, at the cost of reporting a fine-tune that only holds the source's score as a miss.** Today three bars are absolute and the source meets them at epoch 0 (classification 2x chance, segmentation majority-class mIoU, denoising 1 dB over identity); pretrain, forecasting and depth already include an epoch-zero term that silently turns into "beat the source" (4f D-006). Rule: classification, segmentation and denoising under `init` need both their own bar and a strict improvement over the epoch-zero value of the success metric in the metric's direction (with `forecast_run.tie_band` for float metrics); pretrain is out; forecasting and depth keep their formula and their summary says the epoch-zero term is `init`. Labels: the log line "Untrained evaluate BEFORE fit" becomes "Evaluate of the loaded weights before fit"; `init_scale_warning` and `initial_loss_ratio` are not computed under `init` (they compare the loss with ln(C), a statement about a random start) and the summary gives the reason. UNMEASURED: whether each task's epoch-zero evaluation already includes its success metric (accuracy, mIoU, PSNR gain); Step 1 reads it per task.

**4g-D13. Provenance is an optional `init` block in `task.json` (schema 2 kept, optional like `variant_overrides`) and the same block in `results_summary.json`, at the cost of an edit to `record_problems`' exact key sets for all six record kinds and the record tests.** Today only the path string is kept, in `config.json`. Block: `source` (absolute run directory), `file` (`best_model.keras`), `sha256`, `task`, `family`, `variant`, `geometry` (the source record's geometry keys), `split` (its identity) and `split_relation`, `relation` (`same_task` or `cross_task`), `transfer` (`copied` count; `resampled` with path, rule, from and to shapes; `reset`, `dropped` and `grown` path lists). The summary adds `epoch_zero_of: "init"` beside the epoch-zero value. `task.json` is written before `fit`, so the block exists for a killed run and a resume can check it.

**4g-D14. The optimizer is always fresh; learning rates are the target recipe's, unchanged; the 4f `optimizer_options` seam is used only where a family already returns a table (depth, head at 10x); no freezing, at the cost of from-scratch rates on a warm start (ResNet SGD 0.1 with 5 warm-up epochs, ViT 3e-4) whose effect on transferred features is unmeasured.** No fine-tune rate is measured in this repo except depth's 5e-6 from a pretrained encoder; writing per-family fine-tune defaults would be assumption. A layer-wise rate (`LRMultiplierAdamW`, keyed by variable path) reaches only the families whose recipe returns a table; a user-facing table is the language phase's (its likely second user, 4f D-003). Freezing is out (4g scope): no task supports it, Keras 3.8 ignores a `trainable` change after `compile`, `BatchNormalization.trainable=False` also switches it to inference mode, and a zero gradient is not a freeze under AdamW (measured drift 6.8e-3). The gate (Section 9) runs every arm at the recipe rate; a miss is a STOP, not a silent retune.

**4g-D15. A resume wins: with a backup to restore, the source is not read at all; before the first backup, the transfer runs again and the source's sha256 must equal the recorded one, at the cost of needing the source in place for a resume of a run killed in its first epoch.** Today a resume reloads the checkpoint and then overwrites it with the backup, and a replaced source goes unnoticed. `init` stays a compared key of the resume request (it is not in `RESUME_UNCOMPARED_KEYS`). The epoch-zero numbers on a resume come from the fit sidecar as today.

**4g-D16. Predict and export of a fine-tuned run are those of any run of its task: its record holds its own trained size, and nothing reads the source at predict or export time, at the cost of nothing beyond accepting the optional block.** Classification and segmentation resize inputs to the record's size, denoising is size-free, forecasting refuses another lookback, depth refuses another size at export (4f D-009, D-018). A depth run fine-tuned at 364 px exports at 364 px.

**4g-D17. Pretrain is out of 4g as a source and as a target, refused by name, at the cost of no BEiT or Energy Transformer continuation at a larger grid and no pretrained encoder into a classifier through `python -m train`.** A pretrain run writes no `task.json`; its only consumer, a classifier from the encoder, is a new family row problem (BEiT and ET as classification families), not a loader problem; a continuation at another grid needs a BEiT relative-position-bias rule (12 tables of ((2h-1)(2w-1)+3, heads)) with no local reference to check it against. It goes to the named follow-up "a `task.json` record with `predict` and `export` for pretrain", which then adds the pretrain identity hook and two `TransferSpec` rows; nothing in 4g's engine changes for it. (Phase 4h shipped it; the local reference exists and the engine changed, Section 0.)

**Load cost and memory (part of 4g-D2, settled in Step 1).** The source is deserialized beside the target (two models in host memory), its sha256 is computed before the load and again after the copy (a source rewritten while read is refused), and the source model is released before `compile`. UNMEASURED: time and peak resident memory for `depth_anything/vit_l` (a 2.44 GB archive) and the hash time. STOP IF in Section 7.

---

## 3. Scope

### In
- `init` on classification, segmentation, denoising, forecasting and depth; refused by name on pretrain.
- Same task: any geometry change the family keeps every weight across (image size for the spatial families and depth, lookback for tirex, xlstm, ets, timestral, the static batch for timestral), plus ViT image size through the one resample rule.
- Same task, new head: classes (classification, segmentation), horizon, head kind and quantile levels (forecasting).
- Deeper variant: vit, resnet, resnet_seg.
- Cross-task: resnet to resnet_seg, convnext_v1 to convnext_seg, convunext to and from convunext_biased.
- Pre-claim refusals, the weight-level check, provenance, split identity, the bar's epoch-zero term, resume precedence, the `pretrained` path door closed.
- One GPU gate on the final code HEAD (Section 9).

### Out (each with its reason and where it goes)
| Item | Reason | Goes to |
|---|---|---|
| Pretrain as source or target; BEiT and ET classifier families; BEiT relative-bias and ET table resampling | No record, no consumer family, no reference for the BEiT rule (4g-D17) | Phase 4h, straight after 4g and before the language tasks (user, 2026-11-17); shipped by plan 5c6cc105, where the BEiT reference exists (Section 0) |
| Bare `.keras` sources and legacy trainer runs | No record: no split identity, no geometry (4g-D2) | Not planned; refused by name |
| Folding official weights into `init` | 4g-D3 | Revisit if a second family gets official weights |
| Fine-tune learning-rate defaults per family | Unmeasured (4g-D14) | A decision after the gate, only if an arm misses at the recipe rate |
| User-facing layer-wise rates and freezing | No consumer, Keras traps (4g-D14) | The language tasks phase (second user of `optimizer_options`) |
| A smaller ViT grid | Needs antialiasing (4g-D8) | Not planned; refused by name |
| Input-channel adaptation (timm's `adapt_input_conv`, RGB to grey) | No consumer: segmentation, depth and pretrain fix RGB, a grey denoiser would be the only one | Not planned; refused by name as a stem change |
| Width or block-type variant changes | Under 8 percent of weights fit | Not planned; refused by the weight-level check |
| Cross-family pairs beyond the four declared | No shared architecture in the registry | A new family row declares its `sources` |
| DINO (v1), language tasks, Phase 5 wrappers | Not tasks yet | Their own phases |

---

## 4. Steps (13, riskiest first)

Command conventions, carried over from the 4f plan with the 4f merge commit as `BASE`: `T <files>` runs pytest on the files from the worktree with `CUDA_VISIBLE_DEVICES=""`, `MPLBACKEND=Agg` and `PYTHONPATH=<worktree>/src`, after printing that `dl_techniques` and `train` import from the worktree. Under `tests/test_train/test_api/`: `GUARDS6` = `test_registry.py test_the_api_source_rules.py test_the_api_fields_are_read_in_production.py test_task_json_schema_v2.py test_every_family_builds_from_its_defaults.py test_cli_help_allocates_nothing.py`; `SEAMS` = `test_the_sequence_seams_default_to_nothing.py test_the_stream_contract.py`; `PINS` = `test_the_cli_exit_codes.py test_image_classification_end_to_end.py test_image_segmentation_task.py test_the_run_reaches_what_it_was_told.py test_a_failed_api_run_exits_nonzero.py`; `RECORD` = `test_a_third_task_record_dispatches_by_table.py test_task_json_schema_v2.py test_the_predictor_refuses_a_record_that_is_not_the_models.py`. Under `tests/test_train/`: `CENSUS6` = `test_config_fields_are_live.py test_a_broad_handler_outside_main_is_classified.py test_every_trainer_seeds_through_the_shared_helper.py test_a_history_write_follows_the_model_save.py test_a_failed_trainer_run_does_not_exit_zero.py test_a_dataset_reaching_fit_states_its_length.py`. `DOCS` = `tests/test_train/test_api/test_the_forecasting_docs_state_the_numbers_of_the_code.py tests/test_train/test_api/test_the_knob_docs_name_real_keys.py tests/test_the_guides_cite_only_paths_that_exist.py tests/test_readme_claims.py tests/test_train/test_every_cited_dataset_path_exists.py`. `GOLDEN` = three CPU runs (API classification, legacy convnext, API segmentation) compared with baselines taken at `BASE`, pass when all three are EXACT. Each mutant is one row of the plan's mutant table, planted in the tracked file, run, restored, sha256 equal.

1. **Measure, settle the base (no commit).** [RISK: high] On `BASE` (4f merged). (a) Load cost: `predictor.load_model` on a `depth_anything/vit_l` run or, if none exists, a saved untrained vit_l at 252 px: wall time, peak RSS, sha256 time, both models in memory. (b) A depth source with its custom loss loads with `compile=False` and its weights equal the file's. (c) The data-free model analysis (`run_summary.report_data_free_run`) completes on a `convnext_seg` whose backbone was filled from a classifier. (d) The ViT rule: a trained `vit_pico` CIFAR-10 source (train one for 10 epochs on CPU or reuse a run under `results/`, read-only), resampled to 64 px with patch 4 pinned, scored on CIFAR-10 test images upsampled to 64 px, against the source at 32 px and a fresh model; the resample of one table against `tests/test_models/test_dino/dinov2_numpy_reference.py`'s `torch_bicubic_2d`. (e) Paths: build-then-load order on a real xlstm source and a real `resnet_seg` and `convnext_seg` (the `backbone/` prefix), path equality reported per family. (f) Pinned values: where a source records vit `patch_size`, resnet `stem_type`, convnext `strides` (run `config.json`, else the `.keras` zip's `config.json`), and whether `model_args.stem_type=imagenet` overrides the 32 px recipe. (g) Per task, which metrics the epoch-zero evaluation computes and where they are stored. (h) Whether `model.weights` includes any `SeedGenerator` state (it must not be transferred). (i) GPU 0, 20 steps: depth vit_s at 364 px, batch 8, memory and step time; the selection and test sets fill at 364 px (41.8 percent of files had no qualifying crop at 252, 4f D-014). *Files:* the plan's measurement notes only. *Done when:* every item has a number or a yes/no, and any design change is a new decision before Step 2.

2. **The transfer engine.** [RISK: high] `src/train/api/init_from.py` (new; TF-free at module scope): `plan_transfer(source, target, spec, relation, head_reset) -> TransferPlan` (path match with prefix map, head reset and drop, `grow`, `resample`), `check(plan)` (complete-or-raise both directions, message of Section 1.4), `apply(plan)` (copy after every check), the `vit_pos_bicubic` rule (class row kept, `torch_bicubic_resize_matrix` per axis, larger grid only). RED first with tiny Keras models: exact copy, head reset, cross-task prefix map with a dropped head, grow, resample against `torch_bicubic_2d`, a smaller grid refused, an undeclared mismatch refused naming paths and shapes, a refusal leaves every target weight unchanged. *Tests:* `T tests/test_train/test_api/test_init_from_engine.py`. *Mutants:* M1 the "every source weight used or dropped" check removed; M2 a copy moved before the checks; M3 the resample replaced by nearest. *Done when:* green, M1 to M3 RED.

3. **A `TransferSpec` on every family row, and the census.** [RISK: high] `registry.py` (`TransferSpec`, `ModelSpec.transfer`); one spec per row in `recipes.py`, `segmentation_models.py`, `denoising_models.py`, `forecast_models.py`, `depth_models.py`; `pretrain_models.py` rows declare `None` with the 4g-D17 reason. RED first: the census (every row of an `init` task declares a spec or `None` with a reason; each declared change built at the smallest variant gives the engine verdict the spec predicts; the counts per row are derived by running and compared with the counts this design states, a difference is a STOP). *Tests:* `T tests/test_train/test_api/test_init_from_families.py $GUARDS6`. *Mutants:* M4 nbeats `lookback` removed from `fixed`; M5 the vit `resample` entry removed. *Done when:* green within 300 s, M4 and M5 RED.

4. **`checkpoint` out, `init` in, the source reader and the pre-claim refusals.** [RISK: medium] `config.py` (`RUN_HELP["init"]`, the keyword, the removed-key refusal), `trainer.py` (`CONFIG_FIELDS`, `_load_checkpoint` deleted, the hook `init_identity(config_mapping) -> {"head", "split", "geometry", "input"}` with a default that refuses by name, which pretrain keeps), each `tasks/*.py` (field and hook), `init_from.py` (`read_source`: files, records, sha256, task and family pair, `fixed`, pinned, head identity, input scaling; no TensorFlow). `model_args.pretrained` path values refused in every task; depth's `pretrained` resolves false under `init`, an explicit `true` refused. RED first: each Section 1.4 pre-claim row exits 2 and claims nothing; `help` stays TensorFlow-free. *Tests:* `T tests/test_train/test_api/test_init_from_runs.py -k preclaim tests/test_train/test_api/test_every_cli_key_reaches_the_config.py tests/test_train/test_api/test_a_bad_request_is_refused_before_the_run_is_claimed.py tests/test_train/test_api/test_task_trainer.py $GUARDS6 $CENSUS6` (each file's `checkpoint` references, found by grep at `BASE`, move to `init` or go). *Mutants:* M6 a missing `task.json` reaches the claim. *Done when:* green, M6 RED.

5. **Split identity.** [RISK: medium] `init_identity`'s `split` part per task (Section 2, 4g-D11), the comparison and the relation in `init_from.py`, the denoising count of the source's training files in this run's validation pool. RED first: CIFAR-10 seed 43 against 42 runs and records `overlap` with the selection count equal to an independent oracle of the two index sets (4,531 of 5,000 at the measured setting) and `selection_biased: true`; same seed, fraction and cap records `same_split` with 0 and no bias flag; capped source against uncapped target records its count; a different builtin dataset recorded `different_data`; forecasting another seed `unknown` (as shipped, D-045); depth another `train_split` records the scene overlap; an unrebuildable source records `unknown` and `overlap_unknown: true`; no case exits non-zero. *Tests:* `T tests/test_train/test_api/test_init_from_runs.py -k split`. *Mutants:* M7 the source's rows rebuilt with this run's seed instead of the source's (the count reads 0 and the oracle test fails). *Done when:* green, M7 RED.

6. **Wire the transfer into `build_model`; resume precedence.** [RISK: high] `run_sequence.py` (after build: pinned arguments reach the recipe resolver; the source read through `predictor.load_model`; plan, check, copy; source released; then the denoising homogeneity check where it applies, then `compile`), `recipes.py` (resolvers take pinned values), `resume_decisions.py` or `run_sequence.py` (skip with a backup, sha256 check without). RED first, one tiny CPU source and target per kind: vit 32 to 64 px with 10 to 100 classes, vit_pico to vit_tiny, convnext_v1 to convnext_seg, convunext to convunext_biased, tirex 168 to 336, nbeats 168 to 336 refused, depth (4f's stubbed data) 112 to 168 px; a resume with a backup never opens the source (the source directory renamed before the resume); a resume before the first backup with a changed source refused. *Tests:* `T tests/test_train/test_api/test_init_from_runs.py $SEAMS tests/test_train/test_api/test_resume_hardening.py tests/test_train/test_api/test_backup_every_steps.py`. *Mutants:* M8 the resume path transfers again with a backup present; M9 the pin ignored (the recipe re-derives patch 16). *Done when:* green within 600 s, M8 and M9 RED.

7. **Bars and labels under `init`.** [RISK: medium] `tasks/image_classification.py`, `tasks/image_segmentation.py`, `tasks/image_denoising.py` (threshold and `bar_is_met` with the epoch-zero term under `init` only), `common/classification_run.py` (the log label, `init_scale_warning` and `initial_loss_ratio` not computed under `init`, with the reason), forecasting and depth summaries (`epoch_zero_of`). RED first: a stub source that already meets the absolute bar and does not improve is a miss under `init`; the same numbers without `init` are unchanged. *Tests:* `T tests/test_train/test_api/test_init_from_runs.py -k bar tests/test_train/test_api/test_image_denoising_task.py -k "warm or epoch_zero"`. *Mutants:* M10 the classification threshold without the epoch-zero term. *Done when:* green, M10 RED, `GOLDEN` EXACT on the training numbers.

8. **Record, summary and predictor.** [RISK: medium] `predictor.py` (the optional `init` block in `record_problems` for all six kinds), `trainer.py` (`_summary_head` carries the block), each task's `task_record`. RED first: a fine-tuned run's `task.json` and summary hold the block with derived counts; `load_model`, `predict` and `export` of that run behave as for any run of its task at its own size; a record whose block lacks a key is refused. *Tests:* `T $RECORD tests/test_train/test_api/test_init_from_runs.py -k record`. *Mutants:* M11 the block omitted from `task.json`; M12 the head copied when `class_names` differ at equal count. *Done when:* green, M11 and M12 RED.

9. **Docs.** [RISK: low] `src/train/README.md` (a section "Starting from an earlier run: `init`" with the Section 1 examples and refusals, the shared-key paragraphs at lines 640 and 889 and the line 228 and 794 mentions of `checkpoint`, the follow-up bullet that names Phase 4g), `src/train/AGENTS.md` (the hook, `TransferSpec` and how a new family declares one), `research/roadmap/20261117_train_api_init_from.md` brought to the shipped state. *Tests:* `T $DOCS`. *Mutants:* none (the docs guards' own mutants exist). *Done when:* green; every number derived by script.

10. **Targeted CPU regression.** [RISK: low] `T` on every file touched by Steps 2 to 9 plus `GUARDS6 CENSUS6 SEAMS PINS RECORD DOCS`, `tests/test_utils/test_weight_transfer.py` (unchanged module, still used by the legacy trainers), `tests/test_models/test_dino/` (the matrix's home), and `GOLDEN` once. *Done when:* green within 2 h wall, `GOLDEN` EXACT on training numbers (the config files differ by the renamed key, expected).

11. **Review, at most 2 adversarial passes and 2 fix rounds.** [RISK: low] The reviewer gets the code, not this design's conclusions. *Done when:* the second pass has no open WARNING, or the fix-round cap is hit (STOP and report).

12. **One GPU gate on the final code HEAD** (Section 9). [RISK: medium] *Done when:* every arm's claim holds and every log names GPU 0, the worktree `src` and the HEAD.

13. **Gate numbers into the README.** [RISK: low] A script derives every number from the gate logs into `gate_record.md` and the README; a second script compares them. The commit touches `src/train/README.md` only. *Tests:* `T $DOCS`. *Done when:* 0 mismatches.

---

## 5. Files

**New source (1):** `src/train/api/init_from.py` (reader, plan, check, apply, the resample rule, the split comparison). One module because each part is used only by the others and by `run_sequence.build_model`; the rule set is one concern.

**Modified source:** `src/train/api/config.py`, `src/train/api/trainer.py`, `src/train/api/run_sequence.py`, `src/train/api/registry.py`, `src/train/api/recipes.py`, `src/train/api/segmentation_models.py`, `src/train/api/denoising_models.py`, `src/train/api/forecast_models.py`, `src/train/api/depth_models.py` (from 4f), `src/train/api/pretrain_models.py` (spec `None` with reason), `src/train/api/predictor.py`, `src/train/api/tasks/{image_classification,image_segmentation,image_denoising,time_series_forecasting,pretrain,depth_estimation}.py`, `src/train/common/classification_run.py`, `src/train/api/resume_decisions.py` if Step 1 shows the skip belongs there. Not modified: `src/dl_techniques/utils/weight_transfer.py` (legacy trainers keep it), the model packages (DINOv2's matrix is imported, not changed).

**New tests (3):** `tests/test_train/test_api/test_init_from_engine.py`, `tests/test_train/test_api/test_init_from_families.py`, `tests/test_train/test_api/test_init_from_runs.py`.

**Modified tests (found by grep for `checkpoint` at `BASE`; at writing time):** `test_every_cli_key_reaches_the_config.py`, `test_a_bad_request_is_refused_before_the_run_is_claimed.py`, `test_task_trainer.py`, `test_image_classification_end_to_end.py` (the two checkpoint tests, run with `-k`), `test_image_denoising_task.py`, `test_resume_hardening.py`, `test_backup_every_steps.py`, `test_pretrain_task.py`, `test_pretrain_families.py`, the `RECORD` files, `test_the_api_fields_are_read_in_production.py`, `tests/test_train/test_config_fields_are_live.py` (`REGISTERED`), `test_the_knob_docs_name_real_keys.py` only if a document names an `init` key in a checked form.

**Docs:** `src/train/README.md`, `src/train/AGENTS.md`, `research/roadmap/2026_train_scripts.md` (Section 6 paragraph), `research/roadmap/20261117_train_api_init_from.md` (copy of this design).

---

## 6. Assumptions

1. 4f is merged before 4g starts: `TaskTrainer.optimizer_options`, the depth record (`input_size`), `depth_models.py` and the stubbed depth data fixture exist. Step 1 checks; a difference is a STOP.
2. A 252 px depth run from 4f's gate exists under `results/` and is read-only input to the gate.
3. Keras variable paths of a deserialized source equal those of a fresh build of the same config (measured for fresh builds: 24 of 25 families give identical paths on a second build; Step 1(e) checks deserialized ones).
4. CIFAR-10 and CIFAR-100 share no image (their authors' statement; not measured here).
5. Recipe rates are adequate starting points for a fine-tune (unmeasured; the gate is the test, 4g-D14).
6. The resolved pinned arguments are recoverable without TensorFlow (Step 1(f)).

## 7. Failure modes and pre-mortem

Imagine 4g shipped and failed. Likely causes, each with its STOP IF trigger:
- **Path drift beyond xlstm** made a correct transfer refuse, or a wrong one pass. STOP IF Step 1(e) finds a family other than xlstm whose deserialized source paths differ from the target's.
- **The pin is not recoverable**, so a ViT or ResNet at a new size silently changed its stem. STOP IF Step 1(f) finds a pinned value in neither `config.json` (run or zip).
- **The ViT rule is harmful.** STOP IF Step 1(d)'s resampled source scores at or below the fresh model at 64 px.
- **Load cost.** STOP IF Step 1(a) needs more than 180 s or a peak RSS above three times the source file size for vit_l; the alternative (a weights-only read of the archive's `model.weights.h5` mapped through the config) becomes its own decision.
- **The census disagrees with the matrix.** STOP IF any Step 3 verdict differs from the count this design states for the same change.
- **Depth at 364 px does not fit or the sets do not fill.** STOP IF Step 1(i) runs out of memory or leaves a selection or test slot unfilled.
- **Recipe rates destroy the transferred features.** STOP IF a gate arm misses its bar; record the numbers and ask before choosing a fine-tune rate.
- **No-`init` runs changed.** STOP IF `GOLDEN` is not EXACT on the training numbers.
- **Review churn.** STOP IF a third fix round would be needed.
Other failure modes, guarded rather than stopped on: a source rewritten while read (double sha256), a source moved before a first-epoch resume (refused by name), BatchNorm statistics from another dataset (expected, recorded through `split_relation`), a folder-data leak under other directories (`unknown`, recorded).

## 8. Success criteria (each with how it is verified)

1. Every Section 1.4 pre-claim refusal exits 2 and claims nothing: `test_init_from_runs.py -k preclaim`, M6.
2. Every declared family change gives the predicted verdict on the real model, and counts equal the matrix: `test_init_from_families.py`, M4, M5.
3. A transfer is complete-or-raise in both directions and a refusal copies nothing: `test_init_from_engine.py`, M1, M2.
4. The ViT table equals torch's bicubic (numpy reference) and a smaller grid is refused: `test_init_from_engine.py`, M3.
5. The source's architecture is kept: `test_init_from_runs.py`, M9.
6. Split overlap on the same data is counted, never refused, and marks the verdict: `test_init_from_runs.py -k split`, M7.
7. A resume with a backup never reads the source; before the first backup a changed source is refused: M8.
8. Under `init` a non-improving fine-tune is a miss; without `init` nothing changes: M10, `GOLDEN` EXACT.
9. `task.json` and the summary carry the block, and a fine-tuned run predicts and exports at its own size: `RECORD`, M11, M12.
10. The GPU gate holds every arm's claim on the final HEAD (Section 9), and the README numbers equal the gate record (0 mismatches).

## 9. GPU gate (one, on the final code HEAD, GPU 0, one job at a time)

`CUDA_VISIBLE_DEVICES=0`, never `--gpu`; at least 10 GB free on GPU 0 before each job; outputs by absolute path under `results/` with unique `p4g_gate_*` names; `ls results | sort` before and after (the set only grows); every log names GPU 0, the worktree `src` and the HEAD.

| Arm | Source | Target | Claim |
|---|---|---|---|
| G1 depth, higher resolution | 4f's 252 px `depth_anything/vit_s` gate run (read-only) | same family at 364 px, same root and `train_split`, 10 epochs of 100 steps (or Step 1(i)'s budget) | the transfer copied every weight (count derived), reset 0, resampled 0; `split_relation` `same_split`; the best selection `val_loss` strictly below min(constant, layout, epoch zero of the loaded weights at 364 px); the test block reported |
| G2a, G2b classification to segmentation | G2a: `convnext_v1/cifar10` on cifar10, 10 epochs | G2b: `convnext_seg/cifar10` on `oxford_pet`, the recipe's epochs | 98 backbone weights copied under `backbone/` (count derived), `classification_head` dropped, `segmentation_head` fresh; bar met with the epoch-zero term; the existing 30-epoch from-scratch `convnext_seg` numbers quoted beside it, reported, not graded |
| G3a, G3b same task, new head and a larger grid | G3a: `vit/vit_pico` on cifar10 at 32 px, 10 epochs | G3b: `vit/vit_pico` on cifar100 at 64 px, patch 4 pinned | `pos_embed` resampled (65 to 257 rows), head reset, every other weight copied; bar met with the epoch-zero term |
| G4 refusal and overlap (CPU) | a forecasting `nbeats` gate run and G3a | `nbeats` at lookback 336; `vit_pico` on cifar10 with seed 43 from G3a, 1 epoch | the first exits 2 naming the reason with no directory claimed; the second exits 0 and its summary records `overlap` with the selection count and `selection_biased: true` |

Estimated wall time about 75 minutes (G1 about 25, G2 about 25, G3 about 15, plus loads); the budget is Step 1(i)'s and the recipe epochs', measured, not this estimate.

## 10. Complexity budget

- Source lines: about +900 and -80 (the `checkpoint` field in five configs, `_load_checkpoint`, `RUN_HELP`); tests about +1,100.
- New files: 1 source, 3 test files, 0 children.
- New abstractions: `TransferSpec` (one dataclass), one hook (`init_identity`), one rule (`vit_pos_bicubic`); no new seam in the optimizer path.
- Mutants: 12 (M1 to M12), only on guards that carry a claim.
- Review: at most 2 passes, 2 fix rounds.
- Regression: the touched files and the named sets, `GOLDEN` once after Step 7 and once in Step 10; never the full suite or the whole `tests/test_train/test_api` directory.
- GPU: Step 1(i) about 10 minutes, the gate about 75 minutes, nothing else.

## 11. User answers (2026-11-17)

1. Selection overlap: "no refusal". A source whose training rows overlap this run's selection or test rows is accepted; the run counts the overlap and marks the verdict (4g-D11).
2. Pretrain: "yes". The pretrain record follow-up is Phase 4h, straight after 4g and before the language tasks, so a BEiT or Energy Transformer encoder can seed a classifier through `init` (Section 3, Out).
