# train/cliffordnet

Image classification training for CliffordNet, the attention-free, FFN-free vision backbone whose
blocks fuse a detail stream and a depthwise-conv context stream through
`SparseRollingGeometricProduct` + `GatedGeometricResidual`
(`src/dl_techniques/layers/geometric/clifford_block.py`).

| Script | Models | Datasets |
|---|---|---|
| `train_cliffordnet.py` | `CliffordNet` and `HierarchicalCliffordNet` (`src/dl_techniques/models/vision/cliffordnet/`; the variant table and parameter counts are in that README) | CIFAR-10, CIFAR-100, Imagenette, ImageNet-1k |

This directory holds only that script. The language-model, CLIP, CC3M and downsampling-ablation
scripts that used to live here were removed in `8b94188fd`; the CLIP trainer in `src/train/clip/`
is a different, generic one.

Reference: Brandstetter, J. et al. (2025). *CliffordNet: All You Need is Geometric Algebra*.
arXiv:2601.06793v2.

## Running

Always run from the repository root with the project virtual environment, with matplotlib in
non-interactive mode. Pick the GPU with `CUDA_VISIBLE_DEVICES`; `--gpu` counts within that list.

```bash
# the paper recipe: CIFAR, 200 epochs, batch 128
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar100 --variant nano

# a fixed budget of optimizer steps instead of epochs
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --dataset imagenette --image-size 128 --batch-size 32 --train-steps 1000

# hierarchical and custom architectures
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar100 --variant hier_345 --epoch-analysis-every 10
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar100 --variant custom --channels 192 --depth 16 --shifts 1,2,4,8
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --variant custom_hier --stage-depths 2,3,4 --stage-dims 48,96,192 --shifts 1,2,4
```

## Variants

`--variant` takes a preset of `CliffordNet.MODEL_VARIANTS`: `nano` (default), `lite`, `lite_g`,
`nano_g`, `cn_12_3`, `cn_18_5`, `cn_32_3` (the paper's CliffordNet-32), `cn_32_5` and `cn_64_5`
(the paper's CliffordNet-64); a preset of `HierarchicalCliffordNet.MODEL_VARIANTS`
(`hier_345`: stages of 3, 4 and 5 blocks at widths 32, 64 and 128, the authors' constructor
defaults); or `custom` (isotropic, `--channels`, `--depth`, `--shifts`) or `custom_hier`
(`--stage-depths`, `--stage-dims`, `--shifts`).

A flag that does not apply to the chosen architecture or dataset is refused by name, never
ignored: `--channels`, `--depth` and `--use-global-context` with a hierarchical variant;
`--stage-depths` and `--stage-dims` with anything but `custom_hier`; `--random-erasing-prob`
without `--augment cifar_autoaugment`; `--eval-samples` with CIFAR.

A flag you do not pass takes the preset's value (`--cli-mode`, `--ctx-mode`,
`--use-global-context`), and `config.json` records what the model actually uses. The default
DropPath is 0.3, 0.4 for `cn_64_5` and 0.1 for the hierarchical variants.

## Recipe

The defaults are the recipe of arXiv:2601.06793v2 and its authors' `train.py`
(github.com/ParaMind2025/CAN):

| | Paper v2 / authors | This trainer |
|---|---|---|
| epochs / batch | 200 / 128 | 200 / 128 |
| optimizer | AdamW, lr 1e-3, weight decay 0.1 | same, decay decoupled (no L2 regularizer) |
| warm-up | none | none (`--warmup-epochs 0`) |
| schedule | cosine annealing to 0, per epoch | cosine to 1e-5 (alpha 0.01), per step |
| DropPath | 0.3; 0.4 for `cn_64_5` | same (`--stochastic-depth-rate`) |
| augmentation | AutoAugment CIFAR-10, crop pad 4, flip, random erasing 0.25 | `--augment cifar_autoaugment` (CIFAR default) |
| loss | cross-entropy, no label smoothing | same (`--label-smoothing 0`) |

The optimizer and schedule come from `optimizer_builder` and `learning_rate_schedule_builder`.
The loss is the stock `SparseCategoricalCrossentropy(from_logits=True)`; the model emits logits.

## Datasets and the split

All four go through `train.common.classification_data.load_classification_data`, with one
protocol: a held-out slice of train selects (validation each epoch, early stopping, the best
checkpoint), and the dataset's own test split scores the best checkpoint once.

| Dataset | Selection slice | Test split | Notes |
|---|---|---|---|
| `cifar10`, `cifar100` | seeded `--validation-split` of train (default 0.1) | the 10,000 test images | 32 px. Default augmentation `cifar_autoaugment`. |
| `imagenette` | seeded `--validation-split` of train | the 3,925 labelled validation records | Prepared TFDS records under `$TFDS_DATA_DIR`, never downloaded. Random-resized-crop and flip at `--image-size`. |
| `imagenet` | `--holdout-per-class` images per class (default 10, 10,000 images) | the 50,000 official val images | `--imagenet-root` (default `/media/arxwn/data0_4tb/datasets/imagenet2012`), `--devkit-dir` (default `<root>/devkit`) labels val. |

Imagenette and ImageNet use ImageNet mean/std, the loader's random-resized-crop and flip, and
`--augment none` unless a preset is asked for on top. `--eval-samples` (default 2000, Imagenette
and ImageNet only) is the number of test images kept in memory for the end-of-run figures, the
model analysis and the save/reload check; the final evaluation uses the whole split.
`--max-samples` caps the pools for smoke runs.

**`--image-size` defaults to 224, and the isotropic variants do not fit at that size.** Every
block keeps the full input resolution, so `nano` at 224 px and batch 32 asked XLA for a 34.6 GB
buffer and ran out of memory on the 24 GB RTX 4090. Pass `--image-size 128` for Imagenette and
ImageNet, as the earlier runs did. The CIFAR sets keep 32 px and ignore the flag.

## Length of a run

`--epochs` (default 200) or, exclusively, `--train-steps N`: exactly N optimizer steps, run as
`N // K` rounds of `K = --eval-every-steps` (default 250; K must divide N) over a repeating
train stream. Validation, the best checkpoint, the dashboard and `training_log.csv` work per
round, the cosine schedule spans exactly N steps, and a budget shorter than one epoch is well
defined (CIFAR at batch 32 is 1,406 steps per epoch, ImageNet about 40,000). `--train-steps`
with `--epochs`, `--eval-every-steps` without `--train-steps`, and a K that does not divide N
are parser errors. `config.json` records `train_steps` and `eval_every_steps` and
`training_summary.txt` reports "Steps: 1000 (4 x 250)" instead of "Epochs".

## Flags

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `cifar100` | `cifar10`, `cifar100`, `imagenette`, `imagenet`. |
| `--variant` | `nano` | See Variants. |
| `--epochs` | `200` | Exclusive with `--train-steps`. |
| `--train-steps` / `--eval-every-steps` | none / `250` | See Length of a run. |
| `--batch-size` | `128` | |
| `--learning-rate` | `1e-3` | Peak; cosine decay after the optional warm-up. |
| `--weight-decay` | `0.1` | AdamW, decoupled. |
| `--warmup-epochs` | `0` | Linear warm-up from 1e-8; the paper uses none. |
| `--patience` | `30` | EarlyStopping on `val_accuracy`. |
| `--image-size` | `224` | Imagenette and ImageNet only; use 128. |
| `--validation-split` | `0.1` | CIFAR and Imagenette; refused with ImageNet. |
| `--holdout-per-class` | `10` | ImageNet only. |
| `--imagenet-root` / `--devkit-dir` | see above | ImageNet only. |
| `--eval-samples` | `2000` | ImageNet and Imagenette only. |
| `--max-samples` | none | Cap the pools (smoke runs). |
| `--channels` / `--depth` | `128` / `12` | `custom` only. |
| `--shifts` | the model's | Comma-separated offsets; `custom` and `custom_hier`. |
| `--stage-depths` / `--stage-dims` | model defaults | `custom_hier` only, comma-separated. |
| `--cli-mode` | preset's (`full`; `inner` for `cn_32_5`, `cn_64_5`) | `inner`, `wedge`, `full`. |
| `--ctx-mode` | preset's (`diff`) | `diff` (discrete Laplacian) or `abs`. |
| `--use-global-context` | preset's (on for `lite_g`, `nano_g`) | Adds the gFFN-G branch. |
| `--layer-scale-init` | `1e-5` | |
| `--stochastic-depth-rate` | per variant (see Variants) | Maximum DropPath, ramped linearly over depth. |
| `--dropout-rate` | `0.0` | |
| `--label-smoothing` | `0.0` | In `[0, 1)`. Above 0 the loss is `train.convnext.common.SmoothedSparseCategoricalCrossentropy`, so `loss` and `val_loss` are not comparable with unsmoothed runs. |
| `--augment` | `cifar_autoaugment` on CIFAR, else `none` | A preset of `train.common.image_augment`, applied to the [0, 1] image before standardization. |
| `--random-erasing-prob` | `0.25` | Only with `cifar_autoaugment`. |
| `--conformal-alpha` / `--no-conformal` | `0.1` / off | ImageNet only. After training, `train.common.conformal` fits an APS threshold on the whole val split, checks coverage on random half splits, and writes `conformal.json` and `conformal_model.keras` (outputs `(logits, probs, in_set)`). Pass `--no-conformal` to skip it. |
| `--epoch-analysis-every` | `0` | Data-free weight and spectral analyzer before training (`epoch_000`), after epoch 1, then at N, 2N, ... into `<run>/epoch_analysis/`; 0 is off. |
| `--output-dir` | `results` | A timestamped run directory is created under it. |
| `--seed` | `42` | Python, NumPy and Keras, through `train.common.set_seeds`; also seeds the augmentation. |
| `--gpu` | none | Index within `CUDA_VISIBLE_DEVICES` when it is set. |

`--lr-schedule` and `--show-plots` come from the shared base parser and are not read by this
script.

## Output

`<output-dir>/cliffordnet_<variant>_<dataset>_<timestamp>/` holds:

| File | Content |
|---|---|
| `config.json` | Written before training: the flags above and the resolved model settings. |
| `training_log.csv`, `training_history.json` | Per epoch (per round with `--train-steps`), including `lr`. |
| `best_model.keras` | The checkpoint with the best `val_accuracy` on the selection slice. |
| `cliffordnet_<variant>_<dataset>_final.keras` | The last weights; the save/reload round trip is validated. |
| `visualizations/` | `training_dashboard.png` (redrawn during training), `confusion_matrix.png`, `per_class_metrics.png`, `confidence_calibration.png`, `misclassifications.png`, `classification_report.json`. |
| `model_analysis/` | The full `ModelAnalyzer` pass on the best weights, once after training. |
| `epoch_analysis/` | With `--epoch-analysis-every` only. |
| `conformal.json`, `conformal_model.keras` | ImageNet without `--no-conformal`. |
| `training_summary.txt` | Dataset, variant, parameters, length of the run, test results, best `val_accuracy`. |

A failed step raises and the run exits non-zero; it never finishes with an artifact missing.

## Tests

`tests/test_train/test_cliffordnet/` covers the recipe defaults, the hierarchical variants, label
smoothing, the step budget, the visualizations, the epoch-analysis flag and the Imagenette and
ImageNet loaders. Scope pytest to that directory; the full suite takes about 1.5 hours.
