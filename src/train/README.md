# Training

One directory per runnable pipeline (see `AGENTS.md` for the patterns, `GUIDE.md` for the defect
guards, `REPO_MAP.md` at the repo root for which trainer trains which model). Most are run as
`python -m train.<model>.train_<script>`. One is run as `python -m train`.

## `python -m train` and `train.api`

A key=value command line over strict configs. It has five tasks, `image_classification`
(families `vit`, `resnet`, `convnext_v1`, `convnext_v2`, `cliffordnet`, `cliffordnet_hier`),
`image_segmentation` (families `convunext`, `accunet`, `resnet_seg`, `convnext_seg`), `image_denoising`
(families `bfcnn`, `bfunet`, `bfconvunext`, `convunext_biased`, `scunet`), `pretrain` (families `energy_transformer`, `beit`)
and `time_series_forecasting` (families `nbeats`, `prism`, `tirex`, `xlstm`, `ets`, `timestral`, `mdn`),
and the same code is a Python function. Run it from the repo root with `.venv`, with
`MPLBACKEND=Agg` on a headless machine.

```bash
python -m train help                         # commands and tasks; imports no TensorFlow
python -m train list_models task=image_classification
python -m train image_classification         # every key of the task, its type and default
python -m train image_classification out=auto model=vit/vit_pico data.dataset=cifar10 epochs=10
```

```python
from train.api import main, train_image_classification, load_model, export
result = train_image_classification(out="auto", model="vit/vit_pico", data={"dataset": "cifar10"})
main(["list_tasks"])      # the same as the command line; returns the exit code
```

**Commands**: `help`, `list_tasks`, `list_models`, `export`, `predict`, `convert`, and each registered task.
`-v` anywhere logs at DEBUG. `<command> help` prints that command's keys.

**A family trains one task**: each family lists the task it trains, and `model=` is checked against the task
before the data, the recipe or the run directory: `python -m train image_classification model=<a segmentation
family>/<variant>` exits 2 with the task the family trains and the families of this one, and the same holds the other
way round. The four segmentation families, `convunext`, `accunet`, `resnet_seg` and `convnext_seg`
(`train/api/segmentation_models.py`), train `image_segmentation`. Their `image_size` is an int (a square) or
`[height, width]`, each side a multiple of the family's stride: `2 ** depth` for ConvUNext (8 for `tiny` and `small`,
16 for `base` and `large`, 32 for `xlarge`, following a `depth` in `model_args`), 16 for AccUNet, 32 for `resnet_seg`
(8 with `model_args.stem_type=cifar`) and 16 for `convnext_seg` (4 for its `cifar10` variant), each measured by building
the variant. Measured on CPU: all build at a non-square size and return the input's size. AccUNet itself refuses a
side that is not a multiple of 16; ConvUNext does not (it resizes its decoder to its skips), so for it the rule is the
task's choice, made so that the skips meet the upsampled maps; the backbone families build at a size that is no
multiple and return a map of another size (ResNet18 at 50 px gives 64x64), which the rule refuses before the run.

**`image_segmentation`**: pairs of an image and a mask of class ids.

```bash
python -m train image_segmentation out=auto model=convunext/tiny data.dataset=oxford_pet epochs=30
python -m train image_segmentation out=auto model=accunet/base data.format=folder \
    data.train_images=imgs data.train_masks=masks data.num_classes=2 transform_args.image_size=128
```

* **Data**: `oxford_pet` (the local TFDS cache, never downloaded; three classes `pet`, `background`, `border`; no
  void), a `folder` of two directories joined by file stem, or a `csv` with an `image` and a `mask` column. Masks are
  single-channel PNGs of class ids; `data.ignore_index` names the void value, which is left out of the loss, `accuracy`,
  `miou`, the baseline and the analyzer. File data needs `transform_args.image_size` (files have no native size); Pet
  takes 128 from the recipe. The EXIF orientation is applied to images and masks alike. Pet is read at ONE TFDS version,
  `oxford_iiit_pet/4.0.0` (`segmentation_data.PET_TFDS_VERSION`, held equal to the installed builder's by a test): a
  directory that holds only another version is refused before the run directory is claimed. A 16-bit grey image keeps
  its high byte (`value >> 8`: 0..65535 becomes 0..255, a 12-bit image therefore reaches only 0..15, since the scale is
  fixed and never the image's own range) in training and in `predict` alike, a 16-bit RGB PNG is read by PIL with its
  high byte, and a mode `F` image, or a 32-bit `I` image with a value outside 0..65535, is refused by name: PIL's own
  conversion clipped them to white. **Refused before `fit`:** a 1-bit mask with `data.ignore_index=255` (a 1-bit PNG
  reads as 0 and 255, so every foreground pixel would be void and the run would train on one class; the message names
  the collision), and a declared class that has no pixel in any mask of the fit split (named, with the usual cause:
  `data.class_names`, a void value that is also a class, or a `data.max_samples` cut).
* **Black and white masks** (plan 98b866fb step 4, D-016): a mask stored as exactly 0 and 255 is refused by default, and the
  message points at `data.mask_threshold` (`T`, `0 < T < 255`), which reads a pixel ABOVE `T` (strictly: a pixel equal to `T` is
  background) as class 1 and the rest as class 0, for an 8-bit grey and a 1-bit PNG alike (equal classes, tested). It needs
  exactly 2 classes and no `data.ignore_index`, and is refused by key before the run directory is claimed for a palette mask
  (an index is not an intensity) and a 16-bit mask (its scale is unknown). The one decoder applies it, so the pre-claim sample,
  the checks, the pipelines and the class counts all see the classes; `task.json` records it, the data fingerprint includes it,
  and a resume with another value is refused naming `mask_threshold`. `predict` is unaffected (the key is for masks only).
  Example: `python -m train image_segmentation model=convunext/tiny data.format=folder data.train_images=... data.train_masks=...
  data.num_classes=2 data.mask_threshold=127 transform_args.image_size=128`.
* **Converting ADE20K** (plan 98b866fb step 5, D-017): the ADE20K on disk is the FULL 2021_17_01 release (3,688 classes in
  RGB `_seg.png` files coded `(R // 10) * 256 + G`), not the 150-class SceneParse150 benchmark, so it cannot be read as masks.
  `python -m train convert ade20k root=<release> out=<new dir>` writes, OUTSIDE the source tree, one 8-bit PNG of class
  indices per image (`255` is void), `train.csv` and `val.csv` (columns `image` and `mask`; `image` is the original JPEG,
  never copied), `mapping.json` (every class with its release ids and names, its pixels and images in train and val, the
  coverage) and a YAML data file that `python -m train image_segmentation data=<out>/data.yaml model=convunext/tiny
  transform_args.image_size=256` reads (`val.csv`, the benchmark's 2,000 validation images, is the TEST split; the selection
  split is a slice of train). A capped run is `data=<out>/data.yaml data.max_samples=4000` (the command line wins over the file,
  see Keys above). `mapping=official` (the default) is the MIT SceneParse150 class set
  through the benchmark's own id table (`objectInfo150.txt` and `convertFromADE/mapFromADE.txt` of CSAILVision/sceneparsing,
  held in `dl_techniques.datasets.vision.ade20k.SCENE_PARSE_150`), checked against the release's `objects.txt`: it is refused,
  naming the class, if a class's release names share no synonym with the benchmark's. Class 0 of the benchmark ("other
  objects"), the unlabelled pixels and the 3,538 release classes outside the 150 all become the void value, which is what
  mmsegmentation's `reduce_zero_label` does to the benchmark's own PNGs. `mapping=frequency k=N` instead takes the `N` classes
  with the most TRAIN pixels (a counting pass over every train mask comes first, kept in `pixel_counts.json`). The command is
  resumable and deletes nothing: `manifest.jsonl` records each finished image, a rerun skips a mask whose file exists at the
  recorded size, every file is written to `<name>.tmp` and moved into place, and `root`, `mapping` or `k` changed on a resume
  is refused naming the key. An `out` inside `root`, or around it, is refused; so is a non-empty directory that holds no
  manifest. A pair whose JPEG and mask differ in size, whose JPEG carries an EXIF orientation, or that cannot be decoded is
  excluded and listed in `mapping.json`. **This is not the SceneParse150 benchmark**: the training split here is the
  release's 25,574 images (25,258 `ADE_train_` and 316 `ADE_frame_` video frames; the benchmark's is 20,210) at full size,
  where the benchmark rescaled every image to a short side of at most 512, and the annotations are the release's own; an
  mIoU measured on it is not comparable to published numbers. A class has to have a pixel in the fit split (`data.max_samples` cuts it), or the run is refused before `fit`.
  **Measured, untuned** (plan 98b866fb step 8, D-019; the real conversion took 3 min 11 s with 8 workers for 27,574 images and
  250 MB of masks): `data.max_samples=4000` (fit 3,600, selection 400, test the 2,000 `val.csv` images, which no cap reaches), `convunext/tiny`, 256 by 256 px, batch 16, 10 epochs on one RTX 4090: best `val_miou`
  0.0572 at epoch 10 (the curve flattens: 0.0568 at epoch 9; the majority-class baseline is 0.0012), test mIoU 0.0521, test pixel accuracy 0.534; 55 of the
  150 classes have a test IoU above 0 and 23 above 0.1 (`sky` 0.820, `road` 0.523, `building` 0.506, `grass` 0.472). 9.7 minutes of `fit`. A smoke-scale result of
  a 1.9M-parameter model on 14% of the training images, not a score. **Re-run on the corrected ConvNeXt V2 block** (plan 98b866fb step 10c, D-037; the run
  of step 8 measured 0.0343, 0.0320, 0.465 and 32 classes above 0 on the old block, whose layer scale started at 1e-4 under an L2 and a cap and whose GRN started at ones): same command, same seed. The cap has a floor: class 106 (`canopy`) has no pixel in the
  fit split of `max_samples=1000` (seed 42), which the run refuses before `fit` naming the class; 2000 and above hold all 150.
* **Converting COCO** (plan 98b866fb step 6, D-018): the COCO 2017 on disk has instance annotations only (no stuff, no panoptic file),
  so `python -m train convert coco root=<coco_2017> out=<new dir>` can write semantic masks of `background` (class 0, every
  pixel that is no annotated thing, stuff included), the 80 things (1 to 80, in category-id order) and void 255, with the same
  outputs, manifest, resume and refusals as `convert ade20k`. It needs `pycocotools` (installed in the project environment,
  NOT a declared dependency; a missing one exits 2 naming it). The rules: a crowd region (`iscrowd=1`) is void, as the COCO
  evaluation ignores it; a pixel covered by instances of two DIFFERENT classes is void and one class twice keeps its class
  (the torchvision segmentation reference voids every overlap, and cocostuffapi and panopticapi raise on one); the result does
  not depend on the order of the annotations. Images with no annotation are kept as all-background masks. `val2017` is the
  TEST split of the data file, as in ADE20K. **This is not COCO-Stuff (171 classes) or COCO panoptic**: its mIoU is not
  comparable to published numbers, and most of its pixels are class 0.
  **Measured, untuned** (step 8, D-019; the real conversion took 39 min 1 s with 8 workers for 123,287 images and 427 MB of
  masks): `data.max_samples=4000` (fit 3,600, selection 400, test all 5,000 `val.csv` images: the cap bounds the train pool only, D-022,
  and the run was repeated at the iteration 1 gate for that; below 3,000 class 79, `hair drier`, has no pixel in the fit split and the run is
  refused), `convunext/tiny`, 256 px, batch 16, 5 epochs: best `val_miou` 0.0138 against a majority-class baseline of 0.0091 (on the
  selection split), test mIoU 0.0122, test pixel accuracy 0.727 (predicting background everywhere scores 0.719 and a mIoU of
  0.0089 on the 5,000 test images, 0.723 and 0.0091 on the selection split): the model learned the background and `person` (IoU 0.214)
  and 9 other classes above 0 (`zebra` 0.016, `banana` 0.014). **Re-run on the corrected ConvNeXt V2 block** (step 10c, D-037; the old block gave 0.0118, 0.0106, 0.721,
  `person` 0.123 and 5 other classes above 0): same command, same seed.
* **SUT-Crack** (pavement cracks, `pavement_defect/SUT-Crack/segmentation/{Original Image,Ground Truth}`; plan 98b866fb step 7,
  D-007). **Measured**: 130 pairs joined by the stem (`1` to `130`), JPEG photographs and 1-bit PNG masks of 3024 by 4032 pixels
  holding exactly {0, 255}. 119 JPEGs are stored 4032 by 3024 with EXIF orientation 6 (the loader turns them upright, so
  image and mask are both portrait, 3024 wide by 4032 high) and 11 are stored upright. After `data.mask_threshold=127` the crack
  is 1.32% of all pixels (0.32% to 9.36% per image, no image without a crack), so predicting background everywhere scores a
  mIoU of about 0.49 and a pixel accuracy of about 0.987. There is no test split in the folder, so a `folder` run has the
  selection split only (no test metrics, no figures, no analyzer):
  ```bash
  python -m train image_segmentation out=auto model=convunext/tiny data.format=folder \
      data.train_images="<root>/Original Image" data.train_masks="<root>/Ground Truth" \
      'data.class_names=["background","crack"]' data.mask_threshold=127 'transform_args.image_size=[512,384]' \
      epochs=20 batch_size=4
  ```
  For a held-out test split, list one: **the stems divisible by 5 (every fifth in numeric order, 26 pairs) are the test listing and
  the other 104 are train**, with no random state; `validation_split` 0.1 of train is the selection split (10 pairs, so 94 fit
  pairs), and the test split never selects. The listing is two CSV files, by absolute path:
  ```python
  import csv, pathlib
  src, out = pathlib.Path("<root>"), pathlib.Path("sut_crack")  # <root> holds "Original Image" and "Ground Truth"
  out.mkdir()
  stems = sorted(int(p.stem) for p in (src / "Original Image").glob("*.jpg"))
  for name, keep in (("train", lambda s: s % 5 != 0), ("test", lambda s: s % 5 == 0)):
      with open(out / f"{name}.csv", "w", newline="") as f:
          w = csv.writer(f)
          w.writerow(["image", "mask"])
          w.writerows((src / "Original Image" / f"{s}.jpg", src / "Ground Truth" / f"{s}.png") for s in stems if keep(s))
  ```
  ```bash
  python -m train image_segmentation out=auto model=convunext/tiny data.format=csv \
      data.train_csv=sut_crack/train.csv data.test_csv=sut_crack/test.csv \
      'data.class_names=["background","crack"]' data.mask_threshold=127 'transform_args.image_size=[512,384]' \
      epochs=20 batch_size=4
  ```
  `[512,384]` is `[height, width]`: the portrait 4:3 shape at about 1/8 of the photograph. **Batch 4 is measured** (D-036's rule: 20
  real steps on this data, GPU 0, the recipe's other values): 16 runs out of memory (it asks for 27.6 GB), 8 completes 22 steps
  with a peak of 18,014 MiB but with one allocator garbage collection (so it does not fit by the rule), 4 completes 23 steps at
  9,888 MiB with none (0.27 s per step). **Measured, untuned, one seed** (`convunext/tiny`, 1.94M parameters, 20 epochs
  asked, the recipe's learning rate 1e-3 left alone, RTX 4090): all 20 epochs ran (patience 10 did not fire) and the best
  epoch was 11; best `val_miou` 0.6548 against a majority-class baseline of 0.4903 on the 10 selection pairs (the curve is noisy:
  0.61 to 0.65 from epoch 5 on, ten images), **test mIoU 0.7072** (best weights; 0.7121 final weights) and test pixel accuracy
  0.9910 against 0.9872 for predicting background everywhere. Test IoU per class: background 0.9909, **crack 0.4234** (precision
  0.699, recall 0.518; the prediction follows the long cracks and adds small spots, see `visualizations/predictions_best.png`).
  460 s of `fit`. A one-seed result on 26 test photographs of a 94-pair fit split, not a benchmark: the split rule is this
  repository's own, so it is not comparable to a published SUT-Crack number. **Re-run on the corrected ConvNeXt V2 block** (step 10c, D-037); the step 7 run on the
  old block stopped after 15 epochs (best epoch 5) with `val_miou` 0.6666, test mIoU 0.7155 (0.7071 final), pixel accuracy 0.9907 and crack IoU 0.4403.
* **Recipes** are the legacy trainers' defaults (`train.convunext`, `train.accunet`): AdamW, 30 epochs, batch 16, a
  horizontal flip; ConvUNext lr 1e-3, AccUNet lr 1e-4 with 2 warmup epochs. Three values differ on purpose. The monitor is
  `val_miou`, a maximum; both legacy trainers default to `val_loss`. AccUNet trains the cross-entropy, where its
  legacy trainer trains `combo` (0.5 dice + 0.5 cross-entropy, one-hot targets, no void label); `loss_args.name=ce_dice` is
  the same two terms at weight 1 each, per image, with void left out, and the run log says so. And **AccUNet's batch is 8**: batch
  16 runs out of memory at the first step on a 24 GB card (RTX 4090 with 17.4 GB free) in the legacy trainer and in this
  task alike, batch 8 peaks at 7.6 GB (20 steps, no allocator collection; `tests/test_train/test_api/accunet_memory_evidence.json`).
  The learning rate stays 1e-4, not scaled for the batch: the legacy trainer has no batch rule, and the batch-8
  smoke (3 epochs) learned, `val_miou` 0.31, 0.41, 0.47.
* **Backbone families `resnet_seg` and `convnext_seg`** (plan 98b866fb step 2, D-012): the classifier's own factory with
  `include_top=False` and ONE `SegmentationHead` on its last feature map, in one functional model (`Input -> backbone ->
  segmentation_head`), an FCN-32s-shaped network (Long et al. 2015, arXiv 1411.4038, section 4.1: a single stream, the
  classifier's final map predicts, in-network learned upsampling) with no skip connection. They are the coarse baseline of
  the task, not a competitor of the U-Nets: the paper's own table has FCN-32s at 59.4 mean IU against 62.7 for FCN-8s, and
  DeepLab v3 (arXiv 1706.05587, section 3.1) states that an output stride of 16 gives coarser maps than 8. The variants are
  the classifiers' own (`resnet/resnet18` .. `resnet152`; `cifar10`, `tiny` .. `xlarge`; one table, not a copy), the
  head is the library's `SegmentationHead` with EVERY setting passed (`use_skip_connections=False`, `use_attention=False`,
  `use_ffn=False`, `dropout_rate=0.0`, `normalization_type=layer_norm`, `activation_type=gelu`,
  `hidden_dim=min(256, backbone channels)`: 256 is the head's default and the width of DeepLab v3's ASPP; the head's own
  defaults changed once, `use_ffn=True` and `dropout_rate=0.1`, and a test fails when the head gains a knob). **Stride**,
  measured by building every variant at 128x256: ResNet 32 (8 with `model_args.stem_type=cifar`); ConvNeXt V1
  `strides ** len(depths)`, because the model uses ONE `strides` for the patchify stem and every downsampler (the paper
  uses 4 and 2, a stride of 32): the classifier's `strides=4` gives 256 and a 1x1 map at 128x256, so `convnext_seg` sets
  `strides=2`, which gives 16 (4 for `cifar10`, which has two stages). A `model_args.strides` whose total is no power of two is
  refused by name, as are a bad `stem_type` and the keys the adapter sets (`include_top`, `head_config_overrides`,
  `pretrained`, `name`, and for ResNet `enable_deep_supervision`). `load_model` tells the families and variants apart by the
  nested backbone's own layers (`ResNet`: `blocks_per_stage`, `filters_per_stage`, `block_type`; `ConvNeXtV1`: `depths`,
  `dims`); a ConvNeXt V2 body under a `convnext_seg` record is refused. **Recipes** have no legacy source: the optimizer
  settings are ConvUNext's (AdamW, lr 1e-3, weight decay 1e-4, no warmup, per-variable clipnorm 1, cosine to 1%), 30
  epochs, patience 10, a flip, `val_miou`, Pet at 128 px: the same data, loss and pipeline, an untuned starting point and
  not a result. **`optim_args.optimizer=sgd` with a positive `optim_args.weight_decay` is refused for every segmentation family**
  (exit 2 before `open_run`, the message names both keys): sgd's decay is an L2 penalty on the model's kernels, the
  segmentation task builds none, and the two backbone families end in a head that takes no kernel regularizer, so the number would
  be dropped without a word (it was, before plan 98b866fb D-047); use `optim_args.weight_decay=null` or the AdamW the recipes use.
  Both backbone families keep them, `convnext_seg` included, for every variant. **The `convnext_seg` rate was
  1e-4 (1e-3 for `cifar10`) while the V1 block's layer scale started at 1.0, and is 1e-3 now** (plan 98b866fb step 10b, D-034):
  on that old layer scale `tiny` stayed at the pixel prior at 1e-3 (test mIoU 0.370 and 0.370, seeds 42 and 43), and reached 0.597
  and 0.601 at 1e-4. With the paper's layer scale (start 1e-6, no cap, no penalty) `tiny` trains at 1e-3 and is ahead of 1e-4 on both
  seeds, test mIoU 0.550 against 0.507 (seed 42) and 0.537 against 0.529 (seed 43), and `cifar10` is ahead at 1e-3, 0.673
  against 0.628 (seed 44): modest and uneven margins (0.043, 0.009, 0.045), and every number is lower than on the old layer scale
  at its best rate (`tiny` 0.55 against 0.60, `cifar10` 0.67 against 0.67 at 1e-3, 0.63 against 0.67 at 1e-4). The rows are in
  `tests/test_train/test_api/convnext_seg_lr_evidence.json`, held by a test. `small`, `base`, `large` and `xlarge` have been
  measured at neither rate and take the same 1e-3: a first run of one that stalls at the pixel prior should try a lower rate.
  Every `convnext_seg` number in this section was measured on the paper's layer scale in step 10b. Step 10c changes ConvNeXt V2 only
  and `convnext_seg` is V1: built at the same seed, `convnext_seg/cifar10` and `convnext_seg/tiny` give identical weights and identical
  outputs before and after it (`scratch_step10c/convnext_seg_unchanged_{head,fix}.json`), so those runs were not repeated. The ConvUNext
  numbers below (Pet, SUT-Crack, ADE20K, COCO) are V2 and were re-measured by step 10c.
  **Measured on Oxford-IIIT Pet** (30 epochs, 128 px, batch 16, RTX 4090, the recipes as shipped; majority-class baseline
  0.199): `resnet_seg/resnet18` (12.8M parameters, 8 minutes) best `val_miou` 0.631 at epoch 28, test mIoU 0.633;
  `convnext_seg/tiny` (30.0M, 15 minutes) 0.544 at epoch 28, test 0.550 (seed 42, the paper's layer scale at lr 1e-3; the old
  layer scale at 1e-4 gave 0.597 and 0.597). Both are the coarse baseline, not a U-Net's
  score: `convunext/tiny` through the same task (1.94M parameters, 12 minutes, seed 42) reaches best `val_miou` 0.720 at epoch 23,
  test mIoU 0.725 (the pristine fcf2866d8 code, whose V2 block had the old layer scale and GRN init, gave 0.715 and 0.720 on the
  same command: +0.005 and +0.004, not attributable at the 0.01 rule; step 10c, D-037). **Batch 16**, measured by D-036's procedure (20 real steps on Pet at 128 px on the 24 GB card, with another
  session's two `llama-server` processes holding 15.9 GB of it; `tests/test_train/test_api/backbone_seg_memory_evidence.json`):
  16 fits `resnet18`, `resnet50`, `resnet152`, `convnext_seg/cifar10` and `convnext_seg/tiny` (peaks 1.4 to 2.8 GB);
  `convnext_seg/base` needs batch 8 (16 completes with an allocator collection) and `convnext_seg/xlarge` did not fit at any
  batch from 16 down to 1 on the card as it was shared (it asked 9.0 GiB at batch 16 and 5.1 GiB at 8; from batch 4 down the
  353M parameters, their optimizer state and the workspace had already taken the 8.6 GiB that were free). `checkpoint=` from a
  classification run of the same backbone loads NO layer (measured: the helper compares the top-level layers, `backbone` and
  `segmentation_head` against `stem`, `stage1_block1` ..., "No overlapping layers"); the classes' own `pretrained=<path>`
  does load the backbone (9 layers for `resnet18`, 25 for `convnext_v1/tiny`), which these families refuse as a model
  argument for now.
* **Losses** (`loss_args.name`, plan 98b866fb step 3, D-014): `ce` (the default, Keras' sparse cross-entropy on logits with
  the void value left out), `dice`, `ce_dice` and `focal` (`dl_techniques.losses.sparse_segmentation_loss`, logits and sparse
  targets, a void pixel changes neither value nor gradient). `dice` is the soft Dice of MONAI's `DiceLoss` at its defaults
  (and nnU-Net's formula): per image, per class, the mean over ALL classes (background included, which nnU-Net's trainer
  leaves out with `do_bg=False`), a linear denominator and
  `loss_args.dice_smooth` (default 1e-5, MONAI's, which nnU-Net's trainer also passes; must be positive) in numerator and denominator. `focal` is Lin et al.'s focal loss
  (arXiv 1708.02002, eq. 4) in its softmax form, `-(1 - p_t)^gamma log p_t`, `loss_args.focal_gamma` default 2 (0 is the
  cross-entropy); there is no `alpha`, which balances the two classes of a sigmoid problem and would only rescale a softmax
  pixel. `ce_dice` is the per-image cross-entropy plus `dice`, weights 1 and 1 (nnU-Net's `DC_and_CE_loss`). **One value per
  image**: the three new losses average the valid pixels of each image and Keras averages the images, while `ce` is Keras'
  mean over the BATCH's valid pixels, so with void they weight images against pixels and their values are not comparable
  to `ce`'s (no void: equal). A knob the chosen loss does not read (`dice_smooth` for `focal`, `focal_gamma` for `dice`,
  either for `ce`) exits 2 naming the key. For any loss but `ce` the summary's `init_scale_warning` is `null` and a note
  names the loss: ln(C) is the cross-entropy of a uniform prediction and describes no other loss. The AccUNet recipe stays
  `ce`; to train what its legacy trainer calls `combo` use `loss_args.name=ce_dice`.
* **What `val_loss` is**: Keras averages the cross-entropy of a batch over its valid pixels and then averages the
  batches weighted by images, so `loss` and `val_loss` are NOT the cross-entropy pooled over every valid pixel; with a void
  label the two differ whenever the void share differs between batches. `accuracy` and `miou` ARE pooled over every valid
  pixel. That is why the best epoch is chosen by `val_miou`, and the summary says it in `notes`.
* **Success** is the best `val_miou` strictly above the bar, which is `metric_args.success_threshold` when you give one
  (then reaching it is enough) and else the mIoU of predicting the majority class of the selection split for every valid
  pixel, stated in the summary as `majority_class_baseline`. A run that cannot beat a constant has not learned a
  segmentation, and a model that predicts the majority class everywhere scores exactly that baseline, so "above" is
  compared in `float32` (Keras' dtype) with a band of 16 `float32` steps of the baseline (about 1e-6): a bare `>=` of
  the `float32` metric against the `float64` baseline called such a model a success in 8 of 20 measured cases. The
  summary's notes say so.
* **The reload check** (`model_loading_validated`, and `model_loading_check` for what it measured) reloads
  `final_model.keras` and compares it with the model that wrote it: the weights must be EQUAL array by array, and the
  scores must differ by at most `1e-5 + rtol * (the largest score)`, with `rtol` 3e-4, the reloaded model predicting through
  the SAME call path as the one in memory. The absolute 1e-5 of the classification check is wrong for scores of scale 220
  (AccUNet's correct file differed by 0.3 on the RTX 4090 and was reported false with a true `success` and no note). The
  difference that remained after a relative tolerance was a difference of call paths, not of rounding: `load_model(compile=False)`
  runs no `compile`, so Keras resolves `jit_compile="auto"` to XLA on a GPU, while AccUNet's own `compile` forces XLA off (its
  backward pass cannot compile), and under TF32 the same weights differed by 1.7% to 15% of the largest score between the two
  paths (2 of the 4 AccUNet loss cells of the iteration 1 gate reported `ok_reload_unverified` with equal weights). The check now
  copies the in-memory model's `jit_compile` and `run_eagerly` onto the reloaded one before it predicts. Measured on the RTX 4090 with
  short real fits (`tests/test_train/test_api/reload_call_path_evidence.json`): a correct file differs by exactly 0.0 through one path in
  all seven probes (AccUNet with TF32 on and off, ConvUNext, `resnet_seg`, `convnext_seg`), against up to 5.5e-2 through the old two
  paths; of the 27 cells of the gate that were not AccUNet, and so already on one path, 26 show 0.0 and one 2.8e-5 (XLA's autotuning
  between two compilations). 3e-4 is ten times that worst, and 6.7 times under the smallest wrong file seen, ONE kernel scaled by 1.01
  (0.0020 of the largest score); every planted wrong file (that, noise, a fresh initialisation) is false by its outputs and by its weights, and the
  weights are compared exactly because no output tolerance sees a change under its own size. The summary's `model_loading_check` says the largest difference, the tolerance, whether
  the weights were equal, the argmax agreement, whether TF32 was on and the `jit_compile` both models used. **Whenever the check is false** the status is
  `ok_reload_unverified` (not `ok`) and a note says what was measured; the run's other numbers stand.
* A conformal level (`analysis_args.conformal_alpha`) is refused: a segmentation run has no set of classes per image.
* **Figures** (`analysis_args.figures`, and only with a test split): `visualizations/` gets a grid of four fixed
  test pairs after every epoch (`epoch_NNN_seg_grid.png`, NNN 000 being the untrained model: image, ground truth,
  prediction), the training dashboard, and at the end, for the BEST weights, `predictions_best.png`, `miou_curve.png`,
  `per_class_scores.png` and `segmentation_report.json`, the last two being the pooled confusion over every pixel of
  the test split (its mIoU and pixel accuracy equal `test_metrics_best`'s). Void is drawn grey, listed as `void` in the
  legend and left out of every row's pixel accuracy. More than three classes get a longer palette whose first three
  colours are the Pet trainer's; a legend of more than 24 classes lists the first ones shown and says how many more;
  more than 12 classes draw IoU bars only. The model analysis takes the per-pixel path: void is dropped before the
  calibration, which reads a uniform sample of at most 100,000 pixels.

**`image_denoising`** (plan d483ffab): a blind denoiser trained on folders of clean images; the pipeline adds the noise. The clean
image is the target, so there is no label, no class and no test split in the usual sense. Pixels are in `[0, 1]`, sigma is on that
scale (`sigma_255 = 255 * sigma`) and the model has no sigma input. `out`, `model`, `data.train_dirs` and `data.val_dirs` are
required (directories searched recursively for png, jpg, jpeg and bmp files).

```bash
python -m train list_models task=image_denoising
python -m train image_denoising              # every key of the task, its type and default
python -m train image_denoising out=auto model=bfconvunext/tiny epochs=24 transform_args.patch_size=64 \
    'data.train_dirs=["/media/arxwn/data0_4tb/datasets/coco_2017/train2017","/media/arxwn/data0_4tb/datasets/div2k/train"]' \
    'data.val_dirs=["/media/arxwn/data0_4tb/datasets/div2k/validation"]' \
    'data.test_dirs=["kodak24=/media/arxwn/data0_4tb/datasets/kodak24","cbsd68=/media/arxwn/data0_4tb/datasets/CBSD68/original"]'
```

```python
from train.api import train_image_denoising, load_model
result = train_image_denoising(out="auto", model="bfconvunext/tiny", data={"train_dirs": ["<dir>"], "val_dirs": ["<dir>"]})
denoised = load_model("<run>/final_model.keras").predict("noisy.png")["denoised"][0]   # raw float32, the input's size
```

* **Families** (`train/api/denoising_models.py`; each trains this task only: `python -m train image_denoising model=vit/vit_pico` exits 2 naming
  the families of this task, and the other four tasks refuse these five the same way). `bfcnn`, `bfunet` and `bfconvunext` are bias-free (`ModelSpec.properties` holds
  `"bias_free"`: no additive constant anywhere, so `f(a x) = a f(x)`; the three are checked, see Homogeneity below);
  `convunext_biased` (ConvUNext with `use_bias=true`, its own family because the segmentation family forces
  `output_channels` to the class count) and `scunet` carry constants and are never checked. `scunet` has no variants table in the
  library, which publishes one network: `SCUNET_VARIANTS` is this API's, `base` the published network and `tiny` and `small`
  smaller ones nobody tuned.

| Family | Bias-free | Variants | Each side of an image must be a multiple of | Batch of the recipe |
|---|---|---|---|---|
| `bfcnn` | yes | `tiny` `small` `base` `large` `xlarge` | 1 (no stride) | 16 |
| `bfunet` | yes | `tiny` `small` `base` `large` `xlarge` | `2 ** depth`: 8 for `tiny` and `small`, 16 for `base` and `large`, 32 for `xlarge` | 16, `base` 4 |
| `bfconvunext` | yes | `tiny` `small` `base` `large` `xlarge` | the same as `bfunet` | 16, `base` 4 |
| `convunext_biased` | no | `tiny` `small` `base` `large` `xlarge` | the same as `bfunet` | 16, `base` 4 |
| `scunet` | no | `tiny` `small` `base` | 1 (it pads to 64 inside) | 8 |

`transform_args.patch_size` must be a multiple of the family's number (refused before the run directory is claimed, with the
stride named); `predict` and `export` pad to it themselves. Every recipe is the legacy denoiser's optimizer: AdamW, lr 1e-3,
weight decay 4e-3, a global clip norm of 1.0, cosine over the whole run, 100 epochs. The bias-free recipes build what the legacy
trainers build (batchnorm, LeakyReLU with slope 0.1, the trainable Gabor stem of the two U-Nets, ConvNeXt v1 blocks).
`compile_options` returns `{"jit_compile": False}` for `bfunet`, `bfconvunext` and
`convunext_biased` under `precision=mixed_float16` (XLA rejects the gradient of their bilinear upsampling, measured on an RTX 4090 in step 2, which covered all five families) and
nothing for `bfcnn`, `scunet` or float32. `mixed_bfloat16` was not measured and gets the default.
* **Keys with a default** (the rest are the shared groups; `python -m train image_denoising` lists all):

| Key | Default | What it does |
|---|---|---|
| `data.max_train_files` | `10000` | files used per epoch, split equally over `data.train_dirs` (a directory with fewer files repeats them, so COCO does not drown DIV2K); `null` uses every file once |
| `data.max_val_files` | `500` | validation files, a subset chosen by `data.val_seed` alone |
| `data.test_dirs` | `null` | held-out sets scored on whole images after training, each `<directory>` or `<name>=<directory>` |
| `data.noise_kind` | `"additive"` | `additive`, `multiplicative` or `composite` (formulas below) |
| `data.noise_sigma_min` | `0.0` | lowest sigma of a patch |
| `data.noise_sigma_max` | `0.25` | the last value of the curriculum's upper bound, and the upper bound of the validation sigma |
| `data.noise_sigma_max_start` | `0.025` | the upper bound at epoch 0 |
| `data.noise_composite_ratio` | `null` | composite noise only (`null` reads 0.5, the legacy ratio); refused for another kind |
| `data.noise_clip` | `false` | clip the noisy input to `[0, 1]` |
| `data.curriculum_schedule` | `"linear"` | `linear`, `cosine` or `exp` |
| `data.curriculum_epochs` | `"auto"` | epochs the upper bound takes to reach `data.noise_sigma_max`; auto is `max(1, round(0.8 * epochs))` |
| `data.val_seed` | `0` | seed of the validation crops, sigmas, noise and files; not tied to `seed` |
| `transform_args.patch_size` | `256` | side of the square crops, at least 16 (SSIM's window is 11 pixels) |
| `transform_args.patches_per_image` | `4` | crops per decoded image per epoch (and fixed crops per validation image) |
| `transform_args.augment` | `"auto"` | `none`, `flip` or `dihedral`; auto is the recipe's, a flip for folder data |
| `transform_args.channels` | `3` | 1 (grey) or 3 (RGB): every image is decoded to this many channels |
| `metric_args.watch_metric` | `"auto"` | the metric that picks the best epoch inside the selection window, which starts at the curriculum's last epoch whatever the metric; auto is `val_psnr`; accepted: `val_psnr`, `val_ssim`, `val_mae`, `val_loss` (fixed validation set) and the training `loss`, `psnr`, `mae`, `ssim` (noisier: fresh noise draws each epoch) |
| `metric_args.success_gain_db` | `"auto"` | auto is 1.0 dB, an inferred number and not a literature value |
| `metric_args.test_sigmas_255` | `[15.0, 25.0, 50.0]` | noise levels, on the 0-255 scale, of the held-out cells |
| `optim_args.warmup_epochs` | `"auto"` | `max(1, round(0.1 * epochs))` for a cosine schedule (the legacy rule: 10 at the recipe's 100 epochs), 0 for a constant one |

The loss is `loss_args.name=mse`, with `mae` and `charbonnier` as the other choices (a bias-free family accepts `mse` only).
`analysis_args.conformal_alpha` is refused (a conformal set is a set of classes, and a denoiser has none), and so is a
`data.curriculum_epochs` above `epochs`. The validation split is batched at `batch_size`, so `analysis_args.eval_batch_size`
moves no validation number.
* **Defaults that differ from the legacy trainers** (`train.bfunet.train_convunext_denoiser` and its siblings), each on purpose.
  (1) `data.noise_clip` is false, where the legacy trainers clip the noisy input by default: the clip makes the noise non-Gaussian near 0
  and 1 (on a Kodak crop it changed 0.09, 0.50, 4.48 and 8.85 percent of the pixels at `sigma_255` 15, 25, 50 and 63.75), and the
  residual of a bias-free denoiser is the score of the noisy-image density (Miyasawa, Tweedie) only for additive noise that is not
  clipped. The output is never clipped either, except where a PNG is written. `score_identity` in the summary says whether the
  reading holds for the run. (2) The noise is stateless and the epoch is finite: every random number is a function of a seed,
  every file is decoded once before `fit`, a flat black image leaves the list (logged), and an epoch is exactly
  `files * transform_args.patches_per_image` patches; the legacy pipeline is an endless stream of stateful `tf.random` draws.
  (3) Validation is fixed: crops, sigmas and noise are keyed on `(data.val_seed, index)`, so `val_psnr` is a function of the weights
  alone (equal across epochs, across a resume and across the seeds of the training stream); the legacy validation draws fresh noise
  every epoch from a file set that depends on the seed. `val_psnr` is the mean over patches of each patch's PSNR at `max_val` 1,
  capped at 70 dB per patch, the statistic of the legacy `val_psnr_metric` column. (4) `data.curriculum_epochs` is 0.8 of the
  epochs, where the legacy curriculum spans all of them, and the checkpoint is selected only from the curriculum's last epoch (the
  selection window below). (5) The patience of `checkpoint_args.early_stopping_patience` (the recipe's 100) counts from the window's start, and the
  legacy trainers switch early stopping off; the warmup follows the epochs as the legacy rule does (`max(1, round(0.1 * epochs))`),
  not as the recipe's fixed 10 would. (6) `transform_args.augment` is a
  flip, where the legacy augmentation is dihedral (flips and quarter turns; `transform_args.augment=dihedral` draws the same kind of
  patch, and the two were not compared bit for bit).
* **Noise.** A patch's sigma is uniform in `[data.noise_sigma_min, u]`, where `u` is the curriculum's upper bound for the epoch
  (`data.noise_sigma_max_start` to `data.noise_sigma_max` over `data.curriculum_epochs`, shaped by `data.curriculum_schedule`);
  the validation sigma is uniform in `[data.noise_sigma_min, data.noise_sigma_max]`. With `n`, `n'` standard normal and `s` the patch's sigma:
  `additive` is `y = x + n s`, `multiplicative` is `y = x (1 + n s)` and `composite` is `y = x (1 + n s) + n' r s` with `r` from
  `data.noise_composite_ratio`. Each is a pure function of `(clean, sigma, seed)`, with a test of its variance against the formula.
  Any family may train on any kind; only additive noise keeps the residual a score, and the held-out cells below follow the run's kind.
* **The selection window.** The checkpoint saves nothing and early stopping counts nothing before epoch `data.curriculum_epochs - 1`
  (0-based), so the model that is selected has trained on the full noise range it is scored on. A run of 24 epochs has a
  curriculum of 19, a window of 6 epochs from index 18. Runs of 1, 2 and 3 epochs have windows of 1, 1 and 2 (reported, not refused).
  `final_model.keras` is the deliverable; the summary's `selection` block holds `start_epoch`, `window_epochs`, `curriculum_epochs`,
  `best_epoch` and `final_is_best`, and `best_epoch` is the best of the window, the epoch `best_model.keras` holds. The verdict:
  `success_value` is the best `val_psnr` of the window minus `input_psnr` (the PSNR of the noisy validation set, taken by running an
  identity model through the same metric objects before `fit`), `success_threshold` is `metric_args.success_gain_db`, and a run that
  does not clear it exits 1 with both numbers in the summary. The epoch-0 check is the ratio of the untrained model's loss to the
  identity's (`initial_loss_ratio`) with no verdict on the initialization: an untrained denoiser's ratio depends on the noise range far
  more than on the scale of its weights.
* **Held-out cells** (`data.test_dirs`, `metric_args.test_sigmas_255`). Every set is listed and decoded once before `fit`; a file
  that is also a train or validation file is refused before the run directory is claimed. After `fit` the best weights and then the
  final weights score every WHOLE image of every set at every sigma (reflect pad to the family's multiple, eager forward pass, crop
  back; the noise from a fixed seed that is not `seed`, so both models see the same noisy image, which the code checks every run).
  `held_out[set][sigma]` is `{input_psnr, best_psnr, final_psnr, n}`, with sigma keys `"15"`, `"25"`, `"50"`, and `held_out_info`
  holds the protocol (noise kind, clip, ratio, seed, size multiple, `skipped_images`); no key averages over sets or levels. An image
  past the memory rule (1.6 GB per padded megapixel against 8.0 GB, measured for a base U-Net without XLA; no image of the four
  gate sets is within a factor of five) is named and skipped, never tiled silently. Without `data.test_dirs`, `held_out` says
  `skipped` and why. A set is `<name>=<directory>` or a bare directory named by its last component (`CBSD68/original` would be
  "original"), and may not be called `status`, `reason` or `sigmas_255`.
* **The homogeneity report** (`homogeneity` in the summary; bias-free families only). Refused before the claim, with `out` absent:
  every `model_args` key that can put a constant into the model (per-family tables in `denoising_models.py`, held equal to the
  factories' signatures by a census test: a norm other than batchnorm, an activation outside the positively homogeneous ones, `use_bias`,
  a centered supervision norm, deep supervision), and any `loss_args.name` but `mse`. After the build, before `compile`: a static scan
  of the layers and weights, and a numeric probe on a CLONE of the model with every layer scale at 1 and random running variances
  (`f(a x)` against `a f(x)` at alphas 0.3, 1.3, 3.7 and 6.1, TF32 off, tolerance 1e-3 of the largest `a f(x)`; the clone restores every
  random state, so the probe does not move same-seed weights). A failure there exits 2 with the run directory claimed. At the end the
  probe reads the best and the final weights; a failed reading makes `success` false, names `homogeneity_best` or
  `homogeneity_final` in `success_blocked_by`, exits 1 and keeps the models. `homogeneity.status` is `passed`, `passed_static_only`,
  `failed` or, for `convunext_biased` and `scunet`, `not_applicable` (the check never runs there). The numeric readings are taken at
  `float32` only: the tolerance is a float32 number (a clean `bfcnn` under `mixed_float16` reads 1.3e-3 on CPU), so under any other
  precision the static scan runs, the readings say `skipped` with the reason and the status is `passed_static_only`.
* **`task.json` and the stamps.** The record is `schema_version`, `task`, `model`, `family`, `variant`, `channels`, `size_multiple`,
  `data_range` (`"[0,1]"`) and `output` (`images`), plus `variant_overrides` when the recipe changes a key of the variant's table; it
  has no input size (the model's input is `(None, None, channels)`) and no mean or std. A resume compares `channels` and `data_range`
  with the data it loads, and the held-out sets are not compared (the request check already refuses another `data.test_dirs`). The
  top level of `config.json` also carries the keys the pre-API readers look for: `data_range`, `patch_size`, `channels`,
  `val_image_dirs`, `noise_type`, `noise_sigma_min`, `sigma_max_end` and, for `bfconvunext` only, `convnext_version`. Without
  `data_range` the gate of `dl_techniques.utils.denoiser_provenance` refuses the checkpoint; without `val_image_dirs`
  `eval_per_pixel_uncertainty` falls back to its own default directories with no error. Each stamp is a copy of a config value, held
  equal by a test, and removing one breaks the path it serves.
* **Analysis and figures.** The analysis is data-free (weights and spectral only): the stock calibration, information-flow and
  training-dynamics analyses read class probabilities, which an image is not, so the summary has no `ece`. With `analysis_args.figures`
  true, `visualizations/` gets `epoch_NNN_triptych.png` (noisy, denoised, clean and the removed residual of a fixed batch),
  `psnr_vs_sigma.png` (one panel per held-out set), `residual_whiteness.png` and a `training_dashboard.png` with a PSNR panel.
* **Predict and tiling.** `load_model` returns a `DenoisingPredictor`; `predict(x)` takes a path, a PIL image or an array (or a list)
  and returns `{"denoised": [...]}`, ALWAYS a list, one raw `float32` array per input at that input's size, not clipped, not
  standardized. The image is reflect-padded to the family's multiple, the model runs eagerly with `training=False` (a compiled call
  builds one XLA program per new size, 3 to 44 seconds and up to 17 GB, measured in step 2), and the result is cropped back.
  `DenoisingPredictor.predict(x, tile=128, tile_overlap=32)` cuts the padded image into tiles of one size (both numbers multiples of
  the family's multiple) and blends them with fixed linear ramps over the overlap, so a homogeneous model stays homogeneous; a pass
  that runs out of memory raises a `ConfigError` that names `tile`. **The command line has no `tile` key**: `predict` has exactly
  `model`, `data` and `out`, so `python -m train predict model=<run> data=<image or directory> out=<new directory>` predicts whole
  images and writes one `<stem>.png` per input (the raw output clipped to `[0, 1]` and rounded to 8 bits: the one place the output is
  clipped) and, last, `index.json` (`images` with `path`, `image` and `size`, and a `png` sentence).
* **Export** (`python -m train export model=<run> out=<path> format=...`): the file takes float32 images in `[0, 1]` of a free
  height and width (no mean, no std, no sigma input) and returns the denoised image raw. Measured on untrained weights, 40 cells (five
  families, four formats, a free and a fixed size, CPU, relative to the Keras output): 36 export and match to 2.3e-6 at most; four
  are refused with a stated reason before anything is written. Each refusal has a test that builds the file it prevents and shows
  the reason still true, so a converter that is fixed turns the test red.

| Family | Format | Size | Refused because |
|---|---|---|---|
| `bfunet`, `bfconvunext`, `convunext_biased` | `tflite` | fixed | the interpreter resizes its input without an error and a U-Net then returns numbers that are 1.0, 0.98 and 0.65 (relative) off at another size |
| `scunet` | `onnx` | free | it runs only where both sides are multiples of 64 (a Tile node fails at 96x80, 100x100 and 65x65); a fixed-size onnx of `scunet` works, and so do its `saved_model` and `tflite` |

The command line exports a free size, so the SCUNet onnx is the one refused cell it can reach; a fixed size is the Python
function's. The onnx of a trained bias-free model needed a guard (D-034, found by the gate): `tf2onnx`'s grappler pass aborts the
whole process on a `BiasFreeBatchNorm` whose `gamma` is not 1, which an untrained model never shows, so `train.api.exporting` skips
that pass for a model that holds the layer. The gate re-ran the export of a trained `bfconvunext` and a trained `scunet` in every
format the table allows, and all matched Keras within 1e-3 (the largest relative error 1.97e-6).
* **The eval scripts stay scripts** (they read an API run through the stamps above): `python -m train.bfunet.eval_psnr_vs_noise`
  (the paired PSNR-over-sigma sweep of any number of models, `--model name=path` repeated, `--full-image`, `--no-clip`) and
  `python -m train.bfunet.eval_per_pixel_uncertainty --checkpoint <run>/best_model.keras` (conformal intervals and SURE maps). Both
  load all five families; `--init-from` has a legacy builder to load into for `bfcnn`, `bfunet` and `bfconvunext` only, and
  `DenoiserPrior` loads `bfconvunext` at `dynamic` (it reads the `convnext_version` stamp) and the other four at `fixed256`
  only. The legacy trainers and their README remain; `src/train/bfunet/README.md` has their flags.

**Measured: the denoising gate** (plan d483ffab step 14, RTX 4090, the worktree code). Five families of `tiny`, one recipe, with
`data.train_dirs` COCO train2017 plus DIV2K train at `data.max_train_files=1200`, `data.val_dirs` DIV2K validation at
`data.max_val_files=100`, `epochs=24` (curriculum 19, window 6), batch 16, 64 px crops, 4 per image, seed 42, additive unclipped
noise, float32, four held-out sets (`kodak24`, `cbsd68`, `mcmaster`, `urban100`) at `metric_args.test_sigmas_255` left at its three levels:

```bash
python -m train image_denoising out=results/p4c_s14_gate_bfcnn model=bfcnn/tiny epochs=24 batch_size=16 seed=42 \
    transform_args.patch_size=64 transform_args.patches_per_image=4 data.max_train_files=1200 data.max_val_files=100 \
    'data.train_dirs=["/media/arxwn/data0_4tb/datasets/coco_2017/train2017","/media/arxwn/data0_4tb/datasets/div2k/train"]' \
    'data.val_dirs=["/media/arxwn/data0_4tb/datasets/div2k/validation"]' \
    'data.test_dirs=["kodak24=/media/arxwn/data0_4tb/datasets/kodak24","cbsd68=/media/arxwn/data0_4tb/datasets/CBSD68/original","mcmaster=/media/arxwn/data0_4tb/datasets/mcmaster/color","urban100=/media/arxwn/data0_4tb/datasets/urban100/Urban100_HR"]'
```

`bfcnn`, `bfunet` and `convunext_biased` ran that command line; `bfconvunext` and `scunet` ran the same request through the command
line's trainer class (`ImageDenoisingTrainer`) in a process the kill stage could signal.

The bar was written before the first run (D-032): every one of the 5 families x 4 sets x 3 levels = 60 cells, for the best and for
the final model (120 comparisons), must beat its own noisy input strictly, with no averaging and no tolerance. **All 120 passed**
(`best_model.keras` is the final model in every run: `final_is_best` true), and so did the ten-run smoke of 5 epochs (the five
families, `bfconvunext` under multiplicative and composite noise, `bfunet` with `data.noise_clip=true`, `bfunet` and `bfcnn` under
`mixed_float16`). The families, the bar's worst cell and the cost (parameters by `count_params()`, FLOPs by the TensorFlow v1
profiler on a frozen graph, the method of `research/papers/bfunet/measure_efficiency.py`, the 256 px to 64 px ratio 16.00 in every
row):

| Family | Params | GFLOPs at 64 px | GFLOPs at 256 px | Validation gain over the noisy input (dB) | `kodak24` at `sigma_255` 25, input to final (dB) | Worst cell of the 12, gain (dB) |
|---|---|---|---|---|---|---|
| `bfcnn/tiny` | 39,616 | 0.324 | 5.180 | +8.19 | 20.17 to 29.86 | `urban100` 15, +4.70 |
| `bfunet/tiny` | 1,020,528 | 0.752 | 12.035 | +6.37 | 20.17 to 28.57 | `urban100` 15, +1.92 |
| `bfconvunext/tiny` | 1,930,176 | 1.229 | 19.666 | +10.09 | 20.17 to 29.33 | `urban100` 15, +3.73 |
| `convunext_biased/tiny` | 1,941,315 | 1.215 | 19.446 | +10.05 | 20.17 to 29.94 | `urban100` 15, +5.65 |
| `scunet/tiny` | 1,388,246 | 0.755 | 12.083 | +7.92 | 20.17 to 29.55 | `urban100` 15, +3.85 |

Read it as "every family learns to beat its input on every set and level of this recipe", not as a ranking: 7,200 steps is a
short recipe, the bar is low by construction (D-032 says so), the validation input is 21.47 dB, and nothing here was tuned. The
bias-free runs report `homogeneity.status` `passed`. The kill stage ran for `bfconvunext/tiny` and `scunet/tiny`: a SIGKILL inside the
selection window (epoch index 20, 150 of 300 batches), a resume of the same request, unique and contiguous CSV rows, sigma by
absolute epoch, and the restored `val_psnr` equal to the last finished epoch's row (difference 0 dB and 3.8e-6 dB). Predict at the
images' own sizes (24 Kodak and 68 CBSD68 images, all the sizes right), export and the provenance loads were run on both. The
smoke found that five epochs are not enough for `bfunet` to beat its input (-1.7 dB, in float32, float16 and with clipped noise), and
that a 5-epoch multiplicative run reads -1.0 dB against an input of 30.4 dB; neither is held to the bar.

**Parity with the legacy trainer: a MISS at 3 of 7 gated scalars** (plan d483ffab step 15, rule D-007, with the user's ruling D-036
that the gate runs as written). The two arms: the legacy `train.bfunet.train_convunext_denoiser` (`tiny`, v1, batchnorm; seeds 42,
43, 44 and 42 again) and `python -m train image_denoising model=bfconvunext/tiny` (seeds 42, 43, 44), the same network (1,930,176
parameters; the layers are equal one by one, and the API model adds one weight-free float32 cast layer at its output), 30 epochs of 100 steps from 400 files, 4 crops each, batch 16, 64 px, 50 validation files, on
GPU 0:

    python -m train.bfunet.train_convunext_denoiser --variant tiny --convnext-version v1 --block-normalization batchnorm \
        --patch-size 64 --epochs 30 --steps-per-epoch 100 --validation-steps 20 --max-train-files 400 --max-val-files 50 \
        --batch-size 16 --patches-per-image 4 --viz-freq 10 --seed 42 --experiment-name p4c_s15_legacy_s42 --output-dir results

```bash
python -m train image_denoising out=results/p4c_s15_api_s42 model=bfconvunext/tiny epochs=30 batch_size=16 seed=42 \
    transform_args.patch_size=64 transform_args.patches_per_image=4 data.max_train_files=400 data.max_val_files=50 \
    'data.train_dirs=["/media/arxwn/data0_4tb/datasets/coco_2017/train2017","/media/arxwn/data0_4tb/datasets/div2k/train"]' \
    'data.val_dirs=["/media/arxwn/data0_4tb/datasets/div2k/validation"]' \
    data.noise_clip=true data.curriculum_epochs=30 data.noise_sigma_min=0
```

The API arm sets three keys on purpose (`data.noise_clip=true`, `data.curriculum_epochs=30`, `data.noise_sigma_min=0`) so that it
trains the legacy recipe. The rule: `T = max(range of the three legacy seeds, |L42 - L42 repeat|)` per scalar, written before the
first API run; a scalar passes when `|mean(API) - mean(legacy)| <= T`. The gated scalars are the last `val_psnr` and the final
models' whole-image PSNR on `kodak24` and `cbsd68` at `sigma_255` 15, 25 and 50, all seven models scored by one paired invocation:

    python -m train.bfunet.eval_psnr_vs_noise --model legacy_s42=<run>/final_model.keras --model api_s42=<run>/final_model.keras \
        --dataset kodak24=/media/arxwn/data0_4tb/datasets/kodak24 --dataset cbsd68=/media/arxwn/data0_4tb/datasets/CBSD68/original \
        --sigmas-255 15 25 50 --num-samples 100 --seed 42 --full-image --output-dir results --experiment-name p4c_s15_eval_all

(one `--model` per final model, seven in all; the noisy input is clipped, no `--no-clip`, because both arms trained on clipped noise.)

| Scalar (dB) | Legacy mean | API mean | API minus legacy | T | Verdict |
|---|---|---|---|---|---|
| last `val_psnr` | 29.549 | 29.858 | +0.309 | 0.234 | MISS |
| `kodak24` at 15 | 27.466 | 28.822 | +1.356 | 1.162 | MISS |
| `kodak24` at 25 | 24.573 | 26.272 | +1.699 | 2.474 | PASS |
| `kodak24` at 50 | 17.028 | 19.152 | +2.124 | 4.315 | PASS |
| `cbsd68` at 15 | 27.398 | 28.396 | +0.998 | 0.907 | MISS |
| `cbsd68` at 25 | 24.940 | 26.161 | +1.221 | 1.653 | PASS |
| `cbsd68` at 50 | 18.229 | 20.010 | +1.781 | 3.621 | PASS |

**This is the plan's verdict: not a pass.** The API arm is above the legacy mean on all seven scalars, and three miss their T. Read
the four passes as weak: six of the seven T exceed 0.5 dB (0.9 to 4.3 dB), because the legacy models themselves differ by that much
on whole images between seeds (the final models of seeds 42, 43 and 44 score 19.0, 14.7 and 17.4 dB on `kodak24` at 50), so a port
difference below a few dB cannot be seen there; only the last `val_psnr` (T 0.234) discriminates at a useful level, and it is one of
the three misses. The runs were not repeated and no recipe was tuned (D-036). The verdict stays MISS, and the miss is classified as a
**pipeline effect** (the validation set, plus a small augmentation and stream effect) and not as a defect of the port (D-037). The arms
are not the same pipeline: (1) a stateful endless stream against a finite stateless epoch (which crops of which file are seen, and
when); (2) the legacy dihedral augmentation against the API's flip, which the arm did not override; (3) the validation file set, which
is the legacy's 50 files that depend on the seed (24 to 29 of them are shared with the API's fixed set) against the API's one fixed set, and 320
against 200 patches, so `val_psnr` levels differ between arms by something that is not the weights; (4) the initial weights.

**The matched-set rescore** (a CPU probe after the verdict, not a gated scalar): all seven final models (four legacy, three API) scored
on the API's fixed validation patches (50 DIV2K validation files, 4 crops, clip on, sigma U(0, 0.25)) for `val_seed` 0, 1 and 2. API
minus legacy is +0.135, +0.129 and +0.105 dB, all inside T 0.234 and inside the legacy seed range on that set (0.243); at `val_seed` 0
the legacy models read 29.875, 29.661 and 29.632 (mean 29.723) and the API models 29.930, 29.933 and 29.711 (mean 29.858). The legacy
seed-42 model scores 29.652 on its own seed-dependent set and 29.875 on the API's, so the legacy sets are about 0.2 dB harder. The probe
reproduces the API runs' logged last `val_psnr` (29.930, 29.933, 29.711 against the CSV's 29.928, 29.933, 29.714), so the metric is
computed as logged. Leakage was checked and excluded (train is COCO train2017 plus DIV2K train and validation is DIV2K validation; a
shared resolved path is refused; the validation seeds are disjoint from the training stream; the held-out cells score every model on
identical noisy images). What the rescore does not settle: the residual +0.10 to +0.14 dB agrees with the patch-mode cells below and
with the flip against dihedral difference (2), but an API arm with `transform_args.augment=dihedral` is the one remaining key-level
control, and it was not run.

Supplementary and NOT gated (they do not change the verdict): the same invocation in patch mode (`--patch-size 64`, 100 crops per
set, no `--full-image`) gives a T of at most 0.40 dB on all six cells and the API arm within it on all six (differences from -0.064
to +0.292 dB); and three legacy models score 29.97, 29.76 and 29.63 dB on 6 Kodak images at `sigma_255` 25 when applied to 64-px tiles, against
26.98, 27.80 and 25.35 dB on the same noisy images whole, so the 64-px-trained models are not consistent between tiles and whole
images: a hypothesis for the spread of the whole-image cells, not a finding.

**Limits of the task, as found.**
* The numeric homogeneity probe is skipped under half precision (the tolerance is a float32 number), so a trained offset in a
  `mixed_float16` bias-free run is caught by the static scan only, and `homogeneity.status` says `passed_static_only`.
* A `.keras` file copied away from its `config.json` is refused by the provenance gate and by the readers that use it (the gate
  reads the file beside the model). A `val_image_dirs` stamp written as a relative path stays relative, and the reader is run from
  its own working directory.
* `eval_per_pixel_uncertainty` clips its noise always, so on a run trained with `data.noise_clip=false` its numbers describe clipped
  noise, not the run's. The task's own held-out noise is not byte-equal to `eval_psnr_vs_noise`'s either: stateless per-image
  seeds against a sequential `RandomState` agree in distribution, not in value.
* The command line has no `tile` key (above); `tile` and `tile_overlap` are keyword arguments of `DenoisingPredictor.predict`.
* A fixed-shape `tflite` file of a U-Net family is wrong at another size, and the SCUNet onnx with a free size works only at
  multiples of 64: both are refused (the export table).
* `convunext_biased` and `scunet` have no legacy `--init-from` target, and `DenoiserPrior` `dynamic` is `bfconvunext` only. On an API
  model `fixed256` is a plain load, not locked to 256.
* Multiplicative and composite noise have a variance test and a smoke, and no parity evidence: `eval_psnr_vs_noise` is additive only.
* SCUNet's `tiny` and `small` variants and its recipe are this API's and untuned; the gate shows that `scunet/tiny` trains past its
  input, not how good it is. No denoising number here is a benchmark.
* Three help lines were written for the other tasks and say nothing about this one: `optim_args.warmup_epochs` does not say that auto
  follows the epochs here (above), `analysis_args.conformal_alpha` names only segmentation as refusing a level (denoising refuses
  it too), and `analysis_args.eval_batch_size` describes the splits of the other tasks (the validation split is batched at `batch_size`).
* **The gate bar is low.** "Every held-out cell above its noisy input" cannot tell a learned denoiser from a smoothing filter, and it did
  not discriminate a weak family: `bfunet/tiny` (about 1.0M parameters) sits below `bfcnn/tiny` (about 39k) on validation (+6.37
  against +8.19 dB over the input) and on every held-out cell, and still passes. Only `bfconvunext/tiny` has a reference comparison
  (the parity run above); `bfcnn`, `bfunet`, `convunext_biased` and `scunet` have none, so a defect of a few dB in one of them would pass.
* **Best against final was never exercised on a real run.** In all 8 real runs (the 5 gate runs and the 3 API parity runs) the best
  epoch is the last one (`final_is_best` true; the parity runs have a one-epoch window). The selection window, the early stop and the
  resume of a best epoch before the kill are proven on planted curves and on killed runs of a fixture, not on a real run where they differ.
* **Half precision and the other noise kinds have little evidence.** A `mixed_float16` bias-free run gets the static scan only
  (`passed_static_only`) and was run only for `bfunet` and `bfcnn`, in the 5-epoch smoke. Multiplicative and composite noise ran only in
  that smoke (the multiplicative run ended 1.0 dB below its input); clipped noise ran there and in the additive parity arm, not in the
  24-epoch gate.

**`pretrain`** (plan 46e5318f): masked-image pretraining on CIFAR-10 or Imagenette; the labels are not used. The family chooses the
method (there is no `method` key): `energy_transformer` completes the pixels of masked patches (ET masked completion) and `beit`
predicts the code ids that a frozen stage-0 tokenizer gives the masked patches (BEiT masked image modelling). `out`, `model` and
`data.dataset` are required, and a `beit` run also needs `model_args.tokenizer_checkpoint`. A seeded slice of the train rows selects
the best epoch, as in the other tasks, and the test split is scored once after `fit`.

```bash
python -m train list_models task=pretrain
python -m train pretrain                     # every key of the task, its type and default
python -m train pretrain out=auto model=energy_transformer/et_tiny data.dataset=cifar10 epochs=12
python -m train pretrain out=auto model=beit/beit_tiny data.dataset=cifar10 epochs=12 \
    model_args.tokenizer_checkpoint=/path/to/stage0_run/best_model.keras
```

```python
from train.api import train_pretrain
result = train_pretrain(out="auto", model="energy_transformer/et_tiny", data={"dataset": "cifar10"}, epochs=12)
```

* **Families** (`train/api/pretrain_models.py`; the variants are the keys of each model module's `MODEL_VARIANTS`, and
  `python -m train list_models task=pretrain` prints them). Each family trains this task only: a `vit/vit_pico` model given to `pretrain`
  exits 2 naming the families of this task, and the other four tasks refuse these two the same way.

| Family | Method | Variants | Recipe (the legacy trainer's) |
|---|---|---|---|
| `energy_transformer` | `et_masked` | `et_tiny` `et_small` `et_base` | AdamW, lr 5e-4, 2 warmup epochs, global clip 1.0, `model_args.num_steps` 12 |
| `beit` | `beit_mim` | `beit_tiny` `beit_small` `beit_base` `beit_large` | AdamW, lr 1.5e-3, 10 warmup epochs, global clip 3.0, `model_args.drop_path_rate` 0.1 |

Both recipes also hold 100 epochs, batch 32, weight decay 0.05, cosine decay over the whole run to 0.01 of the peak, and a patience of 15
(`ET_RECIPE` and `BEIT_RECIPE` in `pretrain_models.py`; a test reads the legacy `TrainingConfig` of each trainer and holds them equal).
The geometry follows the dataset: `cifar10` is 32 px at patch 4 (an 8 x 8 grid, N = 64 patches), `imagenette` 224 px at patch 16
(14 x 14, N = 196). The ET masks are `data.mask_ratio` 0.5 of the patches in the loss and `data.mask_token_frac` 0.9 of those hidden in
the input, which is 32 and 29 of 64 patches on CIFAR-10 and 98 and 88 of 196 on Imagenette (counted from
`dl_techniques.datasets.vision.masked_patches`). The BEiT mask is in blocks, 24 patches in blocks of at least 5 on the 8 x 8 grid and
75 in blocks of at least 16 on the 14 x 14 one (the auto rule below).
* **Keys of the task** (`python -m train pretrain` lists all, with their help; the groups `optim_args`, `checkpoint_args`,
  `logger_args`, `analysis_args` and `debug_args`, and `epochs`, `batch_size`, `seed`, `devices`, `precision`, `resume_interrupted`
  and `checkpoint`, are the shared ones and mean what they mean for the other tasks, except where this section says otherwise).
  Required: `out`, `model`, `data.dataset` (`cifar10` or `imagenette`) and, for `beit`, `model_args.tokenizer_checkpoint`.

| Key | Default | Read at |
|---|---|---|
| `data.format` | `"builtin"` | `check_builtin_sources` (through `pretrain_facts` of `api/pretrain_args.py`); `PretrainTrainer.build_data` refuses another value |
| `data.validation_split` | `null` | `PretrainTrainer.build_data`, then `image_streams.load_image_streams`; null is the loader's 0.1 |
| `data.max_samples` | `null` | `PretrainTrainer.build_data`, then `image_streams.load_image_streams` (the test split is never capped); `debug_args.smoke` sets 512 when null |
| `data.patch_size` | `"auto"` | `PretrainData.resolve_auto` (it must divide `transform_args.image_size`), `PretrainTrainer.build_model` (the model's patch), `pretrain_data.load_pretrain_batches` (the masks); auto is 4 for cifar10 and 16 for imagenette |
| `data.mask_ratio` | `"auto"` | `PretrainData.resolve_auto`, `pretrain_data.load_pretrain_batches`, then `make_stateless_masked_patch_map_fn`; energy_transformer only, auto is 0.5 |
| `data.mask_token_frac` | `"auto"` | the same path as `data.mask_ratio`; energy_transformer only, auto is 0.9 |
| `data.num_mask_patches` | `"auto"` | `resolve_mask_counts` of `api/pretrain_args.py` (through `PretrainData.resolve_auto`), then `make_stateless_beit_mim_map_fn`; beit only, an explicit value above N is refused by name |
| `data.min_mask_patches_per_block` | `"auto"` | the same path as `data.num_mask_patches`; beit only, at most `data.num_mask_patches` |
| `transform_args.image_size` | `"auto"` | `PretrainTransformArgs.resolve_auto`, `PretrainTrainer.describe_data`, `PretrainTrainer.build_data`; auto is 32 for cifar10 and 224 for imagenette |
| `metric_args.watch_metric` | `"auto"` | `PretrainMetricArgs.resolve_auto`, `fit_callbacks.build_callbacks`, `PretrainTrainer.evaluate_and_report`; auto is `val_loss`, the only value accepted (the success bar is a statement about the minimum `val_loss`, so the saved checkpoint must be that epoch; `loss`, the train loss, and any other metric are refused by key before the run directory is claimed) |
| `metric_args.watch_mode` | `"auto"` | `PretrainMetricArgs.resolve_auto` (read from the metric's name); auto is `min`, the only value accepted (`val_loss` is a loss, and the bar is its minimum; `max` is refused by key) |

`transform_args.augment` is true for this task (flip and pad-crop of the TRAIN stream only, read by `PretrainTrainer.build_data`
and passed to `load_image_streams`; the classification task's group of that name holds auto, and the knob-docs test reads the first group it finds, so the row
is not in the table above).
`analysis_args.conformal_alpha` is refused (`PretrainConfig.problems`: a conformal set is a set of classes). There is no `loss_args`
group: the loss follows the method (ET: the mean squared error of the patch pixels; BEiT: the sparse categorical cross-entropy of the
code ids on logits). `optim_args.optimizer=sgd` with a positive `optim_args.weight_decay` is refused, because the adapters build no L2
penalty and the decay would be dropped (`pretrain_models.resolve_pretrain_args`; the denoising and segmentation resolvers do the same). A key of the
other family (`data.num_mask_patches` on `energy_transformer`, `model_args.num_steps` on `beit`) is refused by name before the run
directory is claimed, and so is a `model_args` key the model takes from the task (`patch_size`, `input_shape`, `vocab_size`).
`debug_args.smoke` runs one epoch on at most 512 train samples and applies no bar (`success` and `success_threshold` are null).
* **Defaults that differ from the legacy trainers** (`train.energy_transformer.train_masked_completion` and `train.beit.train_mim`),
  each on purpose. (1) The selection and test masks are FIXED (the next bullet): the legacy trainers draw them from a shared stateful
  generator (`tf.random.Generator` for ET, the global `random` module for BEiT), so the same weights score a different `val_loss` on
  every pass (0.37 percent spread over four passes of one set of ET weights, measured on CPU before the port). (2) The global
  gradient clip stays on: both recipes carry it as `optim_args.global_clipnorm` (1.0 and 3.0, the legacy numbers; never a per-variable
  `optim_args.clipnorm`, which leaves the global norm unbounded), and an `energy_transformer` run with `optim_args.global_clipnorm=null`
  is refused by name, because the energy descent is unrolled (`model_args.num_steps` of it, 12 in the recipe) and the paper reports
  that the model fails to train at a learning rate above 1e-4 without the clip. (3) The BEiT loss is the reference's: each example's
  masked tokens weigh `N / n_masked`, so `val_loss` is the mean over examples of the per-example masked-token cross-entropy. The
  legacy weight is the bare mask, which makes `val_loss` that mean times the mask fraction (0.375, that is 24 of 64, in a probe), so a
  legacy BEiT `val_loss` and this one differ by a factor of the mask fraction. The ET loss is the patch MSE weighted `N / n_loss`
  on the loss patches, the legacy scale. (4) The BEiT mask counts follow the grid: the legacy 75 patches cannot be drawn on 64, so the
  legacy default fails at config time on CIFAR-10. The auto rule keeps the recipe's fractions, `num_mask_patches` is
  `max(1, round(75 / 196 * N))` (24 at N = 64) and `min_mask_patches_per_block` is `max(1, round(16 / 196 * N))` (5), the minimum
  clamped to the budget. Block masking can under-fill: 4.50 percent of 2,000 draws at (24, 5) on the 8 x 8 grid (mean shortfall 1.03
  patches, no empty mask), 31.45 percent at (75, 16) on 14 x 14. The weight makes an under-filled mask score on the same scale, and an
  empty mask weighs 0. LIMIT: an empty mask still counts as an example of loss 0 in the batch mean, so it dilutes `val_loss` (a probe
  with empty masks: Keras reported 4.4455 where the mean over the examples that masked something was 6.6683). At the default auto mask
  counts the measured share of empty masks is 0 on the 8 x 8 and 14 x 14 grids, so this bites only on tiny grids; the task does
  not refuse one. (5) An epoch is `n_train // batch_size` steps of an endless stream, the legacy arithmetic (1,406 steps on
  CIFAR-10 and 266 on Imagenette at batch 32); `fit` is given `steps_per_epoch` and `validation_steps` by the task (the seam below), and
  Keras keeps ONE iterator across the epochs of a `fit`, so the counter the train masks are keyed on keeps climbing. (6) The legacy
  trainers have no resume, no run-directory guard and no verdict; this task has all three.
* **The masks are fixed, and keyed.** A mask is a pure function of a key, with no generator state. The selection and test masks are
  keyed on `(seed, split, row position of the finite pass)` and never on the stream seed; the train mask is keyed on
  `(stream seed, "train", counter)`; the three splits use three salts, so one seed gives three different mask sequences. ET draws
  `tf.random.stateless_uniform` on a hash of `(seed, salt)` with the index folded in. BEiT draws a block mask from
  `random.Random(f"{seed}:{salt}:{index}")` inside `tf.numpy_function` (`random` hashes a string seed with sha512, so it does not
  depend on `PYTHONHASHSEED`), through the `rng=` argument that `BeitMaskingGenerator.__call__` takes, so concurrent calls share no
  state. The consequence is that `val_loss` is a function of the weights alone: the epoch-zero pass, every epoch's validation pass,
  the best-checkpoint reload check (the gap in the summary is exactly 0.0, not a tolerance) and a resume all score the same weights
  the same. The regime measured: CPU, `tf.data` with `AUTOTUNE`, two `evaluate` passes equal with `==`; a digest of 500 BEiT masks
  equal in three interpreter processes with other `PYTHONHASHSEED` values; the weights a resume restores scoring exactly the
  `val_loss` the killed run logged for them. Equality on a GPU is not claimed here (the gate below is the first GPU measurement).
  Without the per-call `rng=`, a shared generator whose state is swapped per call corrupted 291 to 363 of 2,000 masks per pass at
  `sys.setswitchinterval(1e-6)` and none at the interpreter's default interval, so the concurrency test sets that interval.
* **The tokenizer** (BEiT only). `model_args.tokenizer_checkpoint` is the `.keras` file of a stage-0 `VQVAERotationTrick`, made by the
  legacy `python -m train.beit.train_tokenizer`; the run reads its code ids as targets and takes `vocab_size` from its codebook. The key
  is required, and a request without it, or with a path that does not exist or does not end in `.keras`, exits 2 before the run
  directory is claimed with a sentence that names that script (no file is read at that point). The grid is proved at build time, after
  the claim, by a real forward pass at the run's `transform_args.image_size`: the expected grid is the patches', never the
  tokenizer's own `downsample_factor`, so a tokenizer built at another size or another down-sample is refused naming the file and the
  size, and a `.keras` file that is not a `VQVAERotationTrick` is refused by name. The grid must be the patches': 32 px at patch 4 needs
  `--image-size 32 --downsample-factor 4`, and 224 px at patch 16 needs `--downsample-factor 16`.

```bash
python -m train.beit.train_tokenizer --dataset cifar10 --image-size 32 --downsample-factor 4 --num-embeddings 512 \
    --epochs 5 --output-dir results --experiment-name my_tokenizer_cifar10
```

  **The usage floor.** Over the masked positions of the selection split (one pass of the tokenizer) the run counts the distinct codes
  and takes the perplexity, `exp` of the entropy of the code histogram (the number of equally used codes it is worth), and the top-1
  share; they go to the summary's `tokenizer` block with the `floor`. A perplexity below 64 effective codes
  (`pretrain_run.MIN_TOKENIZER_PERPLEXITY`, entropy below ln 64 = 4.159 nats by more than `pretrain_run.ENTROPY_ROUNDING_TOLERANCE`, 1e-9 nats; 64 equally used codes pass) is REFUSED after the claim, with the measured numbers in
  the message, because the bar of a near-constant target is vacuous and the legacy tokenizer trainer logs no usage that would show a
  collapse. Where 64 comes from (plan 46e5318f, D-007 and D-019): two stage-0 tokenizers made by the legacy trainer on CIFAR-10 at 32
  px, down-sample 4, 512 codes, measured on the 5,000-image selection split. Over all 320,000 positions the 5-epoch one has 508
  distinct codes, a top-1 share of 3.37 percent, entropy 5.8628 nats and perplexity 351.7, and over the masked positions of the auto
  rule (the ones the floor reads) 507 distinct codes and perplexity 354.8. The 1-epoch one, whose learning rate never left its 2-epoch
  warmup, has 350 distinct codes, a top-1 share of 16.23 percent, entropy 3.1495 nats and perplexity 23.3 over all positions, and 285
  distinct codes and perplexity 24.7 over the masked ones. 64 sits 2.6 times above the weak one and 5.5 times below the strong one at
  the masked positions. It is absolute and not a share of the codebook, because what keeps the bar meaningful is how many distinct targets the
  stream has (10 percent of an 8,192-code book would refuse fine tokenizers). It rests on those two tokenizers only, and it refuses
  the 1-epoch one, which is not collapsed (its unigram bar is 3.2051 nats against ln 512 = 6.2383): a floor of 4 effective codes would
  have admitted it. A one-code tokenizer is refused either way.
  The estimate depends on how many positions are counted, that is on the size of the selection split: a perplexity cannot exceed the
  number of positions, and a small sample reads low for a healthy tokenizer. The 5-epoch tokenizer reads 59.0 over 96 positions
  (`data.max_samples=40`, 4 selection images, 68 distinct codes: REFUSED), 116.1 over 240, 195.6 over 480, 259.5 over 1,223 and 354.5 over
  11,977 (`data.max_samples=5000`). The refusal therefore names the positions and the distinct codes it counted and says the estimate is
  limited by the split size, so a small `data.max_samples` is not blamed on the tokenizer alone.
* **The bar and its baseline** (D-009). `success_metric` is `val_loss`. `success_value` is the lowest `val_loss` of the selection
  window (the whole run, as `selection_start_epoch` is 0), `success_threshold` is `min(baseline, epoch-zero val_loss)` and the bar is
  STRICTLY below, compared in float32 (the base class's `>=` reads a loss backwards, and a model that equals the constant predictor must
  not meet the bar). Met is exit 0, missed is exit 1 with both numbers in `results_summary.json`. The baseline is the `val_loss` of a
  CONSTANT Keras model compiled with the run's own loss and evaluated on the same fixed selection stream, measured in `build_data`
  (a resume measures it again and gets the same number): for ET the zero-patch predictor (`baseline.kind` `zero_patches`; the images are
  standardized, so zero is the dataset mean; 0.9995 on a 512-image CIFAR-10 slice), for BEiT constant logits equal to the log of the
  in-sample frequency of each code over the masked selection positions (`unigram_codes`, the best constant predictor on that stream; its
  `val_loss` is the entropy of the masked codes up to the per-example weighting, 5.8716 nats for the 5-epoch tokenizer against ln 512 = 6.2383). Comparing through the same loss and the
  same batch aggregation means the bar and `val_loss` cannot differ by a reduction. The epoch-zero term is the `val_loss` of the untrained
  model, read from `checkpoints/fit_state.json` (`initial.metrics.loss`) by `evaluate_and_report` and never from memory, because a resume
  never evaluates the untrained model; `epoch_zero.binds` in the summary says whether it was the smaller term. It did not bind in the
  measured runs: a random-init ET scored 2.28 to 3.01 over five initialization seeds against 0.9995, and a random-init `beit_tiny`
  6.26 to 6.28 over three against 5.8716, so there the threshold is the baseline and the term is a guard. Because a random initialization sits above the constant predictor,
  `initial_loss_ratio` was above 1 in those runs, and no verdict is given on the initialization (`init_scale_warning` is null with a note). The
  summary also carries `method`, `mask` (patch size, grid and the four mask keys), `stream`, `selection`
  and `tokenizer` (null for ET). The run directory is the shared contract without `task.json`: `config.json`, `config_requested.json`,
  `run.log`, `training_log.csv`, `training_history.json`, `best_model.keras`, `final_model.keras`, `results_summary.json`,
  `visualizations/` (`training_dashboard.png`, and for ET `energy_trace.csv`, the energy descent of one fixed selection batch per
  epoch with one header across a resume; both only while `analysis_args.figures` is true), `model_analysis/` (while `analysis_args.model_analysis` is true; data-free: weights and
  spectral analyses only, since the stock ones read class probabilities), `checkpoints/` and `run.lock`.
* **The stream seed on a resume, and what it costs.** The train stream is endless and a resumed `fit` starts a fresh iterator, which
  would replay epoch 0 (the order, the masks and the augmentation), as the loaders of the three image tasks do. This task rebuilds the stream:
  after the backup pair is settled, the shared sequence calls the task's `resume_stream` with `completed_steps = start *
  steps_per_epoch + (batch + 1 if batch else 0)` (the formula the resume already checks against the backup's optimizer steps), and the
  task rebuilds the same request on a stream seed of `int.from_bytes(sha256(f"{seed}:{completed_steps}")[:4], "big")`. A fresh run's
  stream seed is `seed` itself (`stream.stream_seed` in the summary). Only the stream seed moves: the split keeps `seed`, so no
  selection row ever reaches the train stream, and the selection and test masks stay fixed. A run killed twice reads a third distinct
  stream. The cost: a resumed run is NOT the unkilled run's continuation, its data order is a function of the completed-step count and
  not of the stream the killed run would have read, so the replay of the unkilled stream is not reproduced; the resumed process loads the
  data twice (`build_data` builds it and measures the baseline again, then `resume_stream` rebuilds the full request on the new stream
  seed, `PretrainBatches.with_stream`), so a kill plus its resume reads the data three times in total; and a killed epoch is still lost and trained again from its first batch unless
  `checkpoint_args.backup_every_steps` bounds the loss (then only the batches the last copy had not taken run). **Augmentation and
  reproducibility.** With `transform_args.augment` true (the default) the augmented train PIXELS are not reproduced by one seed. On
  CIFAR-10 the cause is the stateful ops: its augmentation is the plain hook `flip_pad_crop(seed=stream)`
  (`train/common/image_streams.py`), which calls `tf.image.random_flip_left_right(seed=)` and `tf.image.random_crop(seed=)`
  (`dl_techniques/datasets/vision/crops.py`) inside a map with `num_parallel_calls=AUTOTUNE` (`augment_hook.with_hook`), so the scheduling
  order of the parallel calls decides which op draws which number: a same-seed rerun of an augmented CIFAR-10 run in a fresh process is
  NOT bit-reproducible (the review measured one hook and one seed in three fresh processes: three different digests at AUTOTUNE, three
  equal ones at `num_parallel_calls=1`). On Imagenette the augmentation is the seeded hook, whose per-element seeds come from
  `tf.data.Dataset.random` with `rerandomize_each_iteration` (`dl_techniques/datasets/vision/augment_hook.py`), so two iterators of one stream seed differ in their
  pixels there, by design. What stays deterministic on both: the masks, the shuffle and the stream-seed derivation, and the resume tests
  and the exactness claims here run unaugmented. The help of
  `resume_interrupted` says it in one clause: "the training masks and the shuffle continue on a fresh stream derived from the seed and
  the steps completed, never epoch 0's order and never a selection row, and the selection and test masks stay fixed" (the other tasks
  say "the data order replays epoch 0's"), and the summary's resume note carries the same clause (`TaskTrainer.resume_order`).
* **`predict` and `export` are not available for this task.** A pretrain run writes no `task.json` (`task_record` returns `None`;
  a resume works without one and compares the split sizes with the sidecar), so both commands exit 2 and write nothing. For a run
  directory the sentence is `<run>/task.json does not exist; a model is loaded from a run directory, or from a .keras file whose
  directory holds task.json`; for the `.keras` file inside it, `<file> has no record beside it; a .keras file is read with the first
  of best_model.keras.task.json, best_model.task.json, task.json that exists in <run> (export writes best_model.keras.task.json)`. It
  is the existing refusal and it names no pretrain-specific reason; a pretrain-specific sentence needs a change in `predictor.py`.
* **Limits of the task.**
  * No DINO: it has no `validation_data` (its centering is state that an evaluate pass mutates), a k-NN probe as its validation and EMA
    callbacks, so it needs a sequence variant of its own. Imagenette and CIFAR-10 only (`data.format` has the one value `builtin`).
  * No record: no `predict`, `export` or `load_model` of a pretrain run, and no warm start of a classifier from a pretrained encoder
    through `python -m train`. The backbones keep the layer names `et_backbone` and `beit_backbone`, which such a warm start would
    transfer by; the shared `checkpoint` key was not exercised for this task.
  * The tokenizer is made by hand with the legacy script, and the floor rests on two tokenizers.
  * The recipe's learning rate and batch are doubted, not measured, and `run.log` says so (a recipe notice): ET lr 5e-4 is the paper's
    Table 4 value at a batch of 768, 24 times the batch here, and BEiT lr 1.5e-3 is the recipe's at a global batch of 2,048 (128 on 16
    GPUs), neither scaled to batch 32. The legacy trainers are not a parity arbiter (no ET or BEiT run existed to compare with), and
    no parity is claimed.
  * The BEiT stream is input-bound on the CPU: the tokenizer runs inside `tf.data` at batch 1, 1,894 images per second with
    augmentation, so one CIFAR-10 epoch of 1,406 steps of 32 images takes 23.8 seconds of input (1,406 x 32 / 1,894), plus the 3.2 seconds of the
    5,000-image selection pass, at least 27 seconds before any model compute (12 epochs: 5.4 minutes; the GPU gate's G3 epochs after
    the first took 89.8 seconds on average in the final gate, model compute included, with the host's CPU loaded by other processes; 53.5 seconds in the earlier one);
    ET's stream costs 2.5 seconds per epoch. Measured on the CPU with the GPU hidden.
  * The exact-equality claims above were measured on the CPU only.
* **Gate numbers** (plan 46e5318f, Step 17.11: the GPU measurement on the code that ships, commit 87083cf81 of the branch). Five runs on GPU 0, an RTX 4090, one job at a time (the two G4 rows are each a killed run and its resume), each
  `python -m train pretrain` with `out`, `model`, `data.dataset` and `epochs` as the table gives them, `CUDA_VISIBLE_DEVICES=0` and no `--gpu`; the log of
  every run names the device and the worktree `src` as the origin of `dl_techniques` and `train`. The bar (D-017) was written before the first run: it is the
  bar of the section above (the best `val_loss` strictly below `min(baseline, epoch-zero)`), judged from each run's own fixed selection stream, and for the
  5-epoch Imagenette run it gates on "learns something" only (every `val_loss` finite, the best below the epoch-zero value, one energy-trace header,
  `optim_args.global_clipnorm` still 1.0) and reports the zero-predictor bar. Every number is read from the run's `results_summary.json` and
  `training_log.csv` by script. A step time is the median of Keras's progress-bar ms per step, then the mean seconds of the epochs after the first
  (the first holds the XLA compile and the first pass: 41.4 s, 48.2 s and 190.9 s on G1, G2 and G3; the two kill-and-resume rows give the ms only, their epochs are cut by the kill); memory is the
  peak of the run's own process in `nvidia-smi`, sampled every second. An earlier gate on commit ba4e32e49 (21 files of `src` older than the code these numbers come from) read the same bar and also met it in all five runs; parameters, baselines and thresholds are the same floats in both, the best `val_loss` differed (the cause was not isolated: GPU nondeterminism or the code changes) (G1 0.1743, G2 1.0225, G3 3.7712, G4 ET 0.2728, G4 BEiT 4.3424; here G1 0.1742, G2 1.0096, G3 3.7632, G4 ET 0.2793, G4 BEiT 4.3458, G1 -0.1 percent, G2 -1.3 percent, G3 -0.2 percent, G4 ET +2.4 percent, G4 BEiT +0.1 percent), the untrained `val_loss` of G2 and of the ET kill run moved in the sixth or seventh digit (G2 2.0297486782073975 before, 2.029759407043457 now), and the epochs of G3 and of the BEiT kill-and-resume run took 89.8 s and 86.8 s here against 53.5 s and 53.8 s there, with processes of other projects loading the host's CPU during these runs (load average 23 to 31), so the step and epoch times of the BEiT rows are not a clean measurement of the model.

| Arm | Task | Epochs | Params | Step, epoch | Baseline | Epoch zero | Threshold | Best `val_loss` | Verdict | Final is best | Peak GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | `energy_transformer/et_tiny`, cifar10 | 12 | 252,913 | 13 ms, 18.0 s | 1.0041 (zero_patches) | 2.3379 | 1.0041 | 0.1742 | exit 0, success true | true | 2.4 GiB |
| G2 | `energy_transformer/et_tiny`, imagenette (224 px, patch 16) | 5 | 555,457 | 24 ms, 7.3 s | 1.6403 (zero_patches) | 2.0298 | 1.6403 | 1.0096 | exit 0, success true | true | 1.9 GiB |
| G3 | `beit/beit_tiny`, cifar10, the 5-epoch tokenizer | 12 | 5,457,872 | 63 ms, 89.8 s | 5.8716 (unigram_codes) | 6.2661 | 5.8716 | 3.7632 | exit 0, success true | true | 2.7 GiB |
| G4 ET | `energy_transformer/et_tiny`, cifar10, killed in epoch 2, resumed | 4 | 252,913 | 11 ms | 1.0041 (zero_patches) | 2.3379 | 1.0041 | 0.2793 | exit 0, success true | true | 2.4 GiB |
| G4 BEiT | `beit/beit_tiny`, cifar10, killed in epoch 2, resumed | 4 | 5,457,872 | 53 ms | 5.8716 (unigram_codes) | 6.2661 | 5.8716 | 4.3458 | exit 0, success true | true | 2.7 GiB |

  No arm missed its bar. G1: 0.1742 against a zero predictor of 1.0041 (the untrained model scores 2.3379); its test loss on the reloaded best checkpoint is 0.1743. G2: 1.0096 against 1.6403 and an epoch-zero 2.0298 (the zero-predictor bar is met at 5 epochs, reported only), the energy trace has 1 header and 5 rows, the recorded global clip is 1.0. G3: 3.7632 against a unigram bar of 5.8716 (ln 512 = 6.2383), usage floor passed with 507 distinct codes, perplexity 354.8. In every run the epoch-zero term did not bind (the threshold is the baseline). The best epoch is the last one in all five runs, so none of them had stopped improving when its budget ended; `val_loss` rose between consecutive epochs 0 times in G1, 0 in G2 and 0 in G3.
  The exactness claims above, now on the GPU: the best checkpoint's reload gap is exactly 0.0 in all five runs; the zero-predictor baseline is the same float in G1 and in the ET kill run (1.0040831565856934 and 1.0040831565856934); the untrained `val_loss` is not the same float across two processes with one seed in either family: ET 2.3379111289978027 in G1 and 2.3379108905792236 in the ET kill run (relative difference 1.0e-07), BEiT 6.2661356925964355 in G3 and 6.266140460968018 in the BEiT kill run (7.6e-07), so equality across processes on the GPU is measured for the baseline and not for the untrained model's `val_loss` (the earlier gate's ET pair was equal, its BEiT pair was not).
  **Legacy rescore (Step 18, report only, no parity claimed, D-012).** The legacy trainer (`train.energy_transformer.train_masked_completion`, `et_tiny`, cifar10, 12 epochs, its own defaults and stateful masks, 252,913 parameters like G1) was run once and its `best_model.keras` scored on the API's fixed selection stream under the API's loss: 0.17493 against 0.17435 for the earlier gate's G1 best checkpoint scored the same way (the control reproduces that run's logged 0.17435 to 3e-8; the final gate's G1 logged 0.17422 and was not rescored); on the test stream 0.17504 against 0.17450. The two runs differ in their train masks and streams, so the numbers are reported and not graded.
  **Kill and resume (G4).** Each run was started with `checkpoint_args.backup_every_steps=300`, its PID recorded, and sent SIGKILL (that PID only) when the backup's metadata first read epoch 2 batch 299, that is after 2 CSV rows and 300 batches of the third epoch (the process exit status was 137); the same command with `resume_interrupted=true` then finished with exit 0 in ET and exit 0 in BEiT. The copy held 3112 optimizer steps (2 x 1,406 + 300), the final copy 5624 (4 x 1,406): no step lost or run twice after the copy. The CSV has one row per epoch (0 to 3) and the two rows written before the kill are unchanged. The learning rate the CSV records at each epoch start equals the run's own schedule evaluated at 0, 1,406, 3,112 and 4,218 steps, exactly, in both families: `optimizer.iterations == epoch * steps_per_epoch` at the starts of epochs 0, 1 and 3, and the epoch the kill landed in starts from the copy, at 2 x 1,406 + 300 (the documented behaviour of `backup_every_steps`), so for that one epoch the equation reads plus 300. The first three train batches of the resumed process were not read back from the real fit (it was not traced); the stream its stream seed gives (`with_stream(3112)`), digested the way the CPU resume test does on the real CIFAR-10 unaugmented, differs from epoch 0's stream at all three batches in both families.
* **Follow-up plans**, named and not started: the language tasks (the next user of `fit_kwargs` and `resume_stream`, after
  `time_series_forecasting` below), depth, DINO, the BEiT tokenizer as a method of its own (`beit_tokenizer`), a `task.json` record with `predict`
  and `export` for pretrain, and the Phase 5 wrappers (the repo map entry among them).

**`time_series_forecasting`** (plan 3a8319ef): trains a forecaster on windows of a series, either a synthetic bank of the library's
generator patterns or one csv of your own. `out`, `model` and `data` are required (every key of `data` has a default, but the group `data` itself must be named: `data.format=builtin` is the synthetic bank); `data.lookback` and `data.horizon` come from the family's recipe. The family chooses the model. Every series is cut BY TIME into a training slice, a slice that selects the best epoch and a
slice scored once after `fit`, and the verdict is a bar against two baselines and the untrained model (below). The run is
reproducible, resumable, and `predict` and `export` work on its run directory.

```bash
python -m train list_models task=time_series_forecasting
python -m train time_series_forecasting     # every key of the task, its type and default
python -m train time_series_forecasting out=auto model=nbeats/base data.format=builtin epochs=10
python -m train time_series_forecasting out=auto model=prism/tiny data.format=builtin 'model_args.quantile_levels=[0.1,0.5,0.9]'
python -m train time_series_forecasting out=auto model=nbeats/small data.format=csv data.path=series.csv \
    'data.value_cols=["load"]' data.time_col=ts
```

```python
from train.api import train_time_series_forecasting
result = train_time_series_forecasting(out="auto", model="nbeats/base", data={"lookback": 168, "horizon": 24})
```

* **Families** (`train/api/forecast_models.py`; the variants are the keys of each family's variants table, `python -m train list_models
  task=time_series_forecasting` prints them). Each family trains this task only; `model=vit/vit_pico` exits 2 naming the families of
  this task, and the other four tasks refuse these seven the same way. Every row is held to its library factory's signature by
  `signature_census`: a key the family offers is passed through (and may override a variant's value), a key it refuses or forces is
  named in the refusal, and a key the adapter takes from the data (`lookback`, `horizon`, the channels) is refused as `model_args`.

| Family | Variants | Head and output | Channels | Loss |
|---|---|---|---|---|
| `nbeats` | `tiny` `small` `base` | point; the forecast branch only (the model returns forecast and residual) | any | MAE |
| `prism` | `tiny` `small` `base` `large` | point, or quantile with `model_args.quantile_levels` | any | MAE, or the pinball loss |
| `tirex` | `tiny` `small` `medium` `large` | quantile, 9 levels (0.1 to 0.9) | 1 | the pinball loss |
| `xlstm` | `tiny` `small` | point, or quantile with `model_args.quantile_levels` | any for a point head, 1 for a quantile one | MAE, or the pinball loss |
| `ets` | `ANN` `AAN` `AAA` | point; one smoothing parameter per component | 1 | MAE |
| `timestral` | `5M` `28M` `123M` `209M` | quantile, 21 levels; built at a static batch (the data drops the remainder) | any | the model's own objective |
| `mdn` | `tiny` `small` `base` | a mixture over the horizon; the point is its mean | 1 | the mixture's negative log-likelihood |

  A family with a fixed loss refuses `loss_args.name` by name; a point head takes `mae`, `mse` or `mase` (the per-batch
  approximation `MASELoss`, whose value moves with batch composition). `timestral` forecasts one patch of 32 points at a time, so a
  `data.horizon` above its patch size (32 unless `model_args.patch_size` says otherwise) is refused: its `call` is the forecaster's first block only up to one patch, and a longer horizon would
  grade a frame the model was never trained in (measured: horizon 48 differs from the library's multi-block forecaster by 2.54 on a
  scale of 16, steps 1 to 32 by 0.0). The model's own normalization is OFF for `nbeats`, `tirex` and `xlstm` (`use_normalization` is
  forced, and a user value is refused as a double normalization); `prism` and `mdn` have none and need the data layer; `ets` and
  `timestral` are equivariant themselves. A model regularizer (`kernel_regularizer` and its kin) is refused for the families that
  have one (the optimizer's decay is the only decay), and so is `use_quantile_head` (the head follows `model_args.quantile_levels`).
  **Refused by name, with the measured reason** (raised before the claim, one sentence each, a test per family): `deepar`, because its
  trained graph is teacher-forced, so `call` takes a window that CONTAINS the horizon (changing the horizon values moved `mu` at the
  horizon positions by up to 0.659), an exported `serve` function would take the answer as its input, and its forecast path is an eager
  Monte-Carlo loop no export format carries; `adaptive_ema`, a regime and slope filter whose outputs have no horizon axis, so a
  forecast and the bar "beats naive" are undefined for it.
* **Recipes** (`RECIPES`, one per family in `train/api/forecast_models.py`, derived by script from the specs): AdamW, 10
  epochs, batch 128, `data.lookback` 168 and `data.horizon` 24, weight decay 0.004, one warmup epoch, cosine over the whole run to
  0.01 of the peak, a GLOBAL gradient clip of 1.0 (never a per-variable `optim_args.clipnorm`), early stopping after 10 epochs without a
  better `val_mae`, learning rate 1e-4 except `ets` (5e-2, no warmup: one scalar parameter did not move at 1e-4, 20 steps at 1e-3
  moved its `mae` from 1.0069 to 1.0066) and `mdn` (5e-4). An epoch is `data.steps_per_epoch` steps, 500 at the defaults, so a default
  run is 5,000 steps. The numbers are the legacy trainers' where those agree with the measured ones and the measured ones where they
  do not; the next bullet names each difference.
* **Keys of the task** (`python -m train time_series_forecasting` lists all, with their help; the groups `optim_args`,
  `checkpoint_args`, `logger_args`, `analysis_args` and `debug_args`, and `epochs`, `batch_size`, `seed`, `devices`, `precision`,
  `resume_interrupted` and `checkpoint`, are the shared ones and mean what they mean for the other tasks, except where this section says
  otherwise). Required: `out`, `model` and `data`; the group `data` itself must be named (`data.format=builtin` is the synthetic bank), and a csv run also needs `data.format` and `data.path`.

| Key | Default | Read at |
|---|---|---|
| `data.format` | `"builtin"` | `check_forecast_data` (through `TimeSeriesData.resolve_auto`, before the claim), `TimeSeriesForecastingTrainer.check_model_args` (a csv gives the channel count), `forecast_data.load_forecast_batches` |
| `data.path` | `null` | `forecast_series.read_csv_series`, from `check_forecast_data`, `TimeSeriesForecastingTrainer.check_model_args` and `forecast_data.load_forecast_batches`; csv only, refused for `builtin` |
| `data.value_cols` | `null` | `forecast_series.read_csv_series`; csv only; null is every column except the time column and an unnamed index column |
| `data.time_col` | `null` | `forecast_series.read_csv_series`, again by `ForecastPredictor` for a csv given to `predict` (the record's `time_col`); csv only; null means row order is time |
| `data.lookback` | `"auto"` | `TimeSeriesData.resolve_auto` (the recipe's 168), `check_forecast_data`, `geometry_problems`, `window_problems` (the family's measured minimum, before the claim), the adapters through `TimeSeriesForecastingTrainer.build_model`, `forecast_data.load_forecast_batches` |
| `data.horizon` | `"auto"` | the same path as `data.lookback` (the recipe's 24); a `timestral` run is refused above its patch size and an `nbeats` run below 2 (`window_problems`) |
| `data.realizations` | `4` | `forecast_data.build_bank`; builtin only |
| `data.series_length` | `10000` | `forecast_data.build_bank`, `check_forecast_data` (a synthetic series is split like a csv of that many rows); builtin only |
| `data.noise_level` | `0.1` | `forecast_data.build_bank`; builtin only |
| `data.patterns` | `null` | `forecast_data.load_forecast_batches` (an unknown name is refused); builtin only; null is every pattern |
| `data.select_fraction` | `0.15` | `forecast_series.split_bounds`, `check_forecast_data`; with `data.test_fraction` it must leave a train slice |
| `data.test_fraction` | `0.15` | `forecast_series.split_bounds`, `check_forecast_data` |
| `data.n_select` | `null` | `forecast_series.check_split_sizes` (before the claim) and `forecast_data.load_forecast_batches`, rounded DOWN to whole batches; fewer than one batch is refused; null is 8 windows per series (builtin) or every window of the segment whose targets do not overlap (csv) |
| `data.n_test` | `null` | the same path as `data.n_select` |
| `data.steps_per_epoch` | `null` | `forecast_data.load_forecast_batches`, then `TimeSeriesForecastingTrainer.fit_kwargs`; null is the number of train windows over the batch size, at most 500 |
| `data.max_samples` | `null` | `forecast_data.load_forecast_batches`; caps the train windows an epoch is sized from when `data.steps_per_epoch` is null, never the selection or test windows |
| `transform_args.scale_floor` | `1e-06` | `TimeSeriesForecastingTrainer.build_data`, then `forecast_data.context_stats_np`; recorded in `task.json` and read again by `predict` and `export`; the flatness threshold of a context and of the flat-window check of a csv (`forecast_series.check_csv_level`) |
| `metric_args.watch_metric` | `"auto"` | `ForecastMetricArgs.resolve_auto`, `fit_callbacks.build_callbacks`; auto is `val_mae`, the only value accepted (the bar is about the minimum `val_mae`, so the saved checkpoint must be that epoch) |
| `metric_args.watch_mode` | `"auto"` | the same path; auto is `min`, the only value accepted |

  Two rows are not in the table, because the knob-docs test reads the first group that has the field and `data.dataset` is the
  classification task's there and `loss_args.name` the denoising task's: `data.dataset` is `"synthetic"`, the only value (read by
  `TimeSeriesData.problems` and `forecast_facts`; a csv is named by `data.path`), and `loss_args.name` is `null`, the family's own
  loss (read by `TimeSeriesForecastingTrainer.build_loss_and_metrics` and, through `loss_problem`, by
  `TimeSeriesForecastingTrainer.check_model_args`).
  `jit_compile` is a key of the task itself (default `false`, read by `TimeSeriesForecastingTrainer.compile_options`, which passes
  it to the ONE `compile` call, and by `TimeSeriesForecastingTrainer.check_model_args`). `analysis_args.conformal_alpha` is refused (a conformal set is a set of
  classes; the interval forecast of this task is a quantile head). `debug_args.smoke` runs one epoch on at most 512 train samples
  and applies no bar. `precision` must be `float32`: `mixed_float16` and `mixed_bfloat16` are REFUSED by name before the run directory
  is claimed (D-041). Measured on `nbeats/tiny`, a run under either policy trains, writes its `task.json` and passes its own reload
  check (the training process has the mixed policy), and then `load_model` (so `predict` and `export`) refuses the run's own
  `best_model.keras` in a float32 process: a float16 against float32 clash in the layers (`InvalidArgumentError`, a half tensor met a
  float tensor), which the refusal used to report as a wrong lookback. The same file runs under a mixed global policy, so the cause is
  the process and not the file; a forecaster that cannot be predicted from is not offered.
* **Defaults that differ from the legacy trainers** (`train.time_series.*`), each on purpose. (1) The reconstruction term is NOT
  ported: the legacy N-BEATS trainer adds a term whose target is the INPUT, so minimizing it pushes the backcasts toward zero and it is a
  defect, not a regularizer (test MAE of 400 steps on one seed, with the term 0.7575 and without 0.7704, inside one seed's noise); the
  task trains the stock `NBeatsNet` on its forecast branch alone (`compile(loss=[loss, None])`, the metrics a list of lists), and
  `check_model_loading` compares every output of a list or tuple model, not the first only. Keras warns that two kernels of `nbeats`
  receive no gradient: expected, named in `run.log`. (2) Standardization is by the CONTEXT only. The legacy processor standardizes the
  whole window with the horizon included, so the future leaks into the statistics (on its windows a zero predictor scored a mean
  squared error of 1.1678 against naive-last's 1.4737, which only the leaked statistics explain) and N-BEATS then normalized a second time inside
  `call`. Here each window is divided by the mean and the population standard deviation of its context per channel, and its target by
  the same numbers, in ONE place (the data layer, `forecast_data.standardize_tf`, with a numpy twin `context_stats_np` that `predict`
  calls and a keras-ops twin `context_stats_ops` that the export calls; tests hold each twin equal to the tf function). (3) The context is centred on its FIRST value before its mean and std are taken, and the scale is the
  population std of the context (D-040, which replaced the floor `max(std, 1e-3 * mean|context|)` of the first version). The
  differences of nearby numbers are exact, so the float32 function the export runs and the float64 numpy function `predict` runs
  standardize a window to the same values at ANY level (the first version centred on the mean of the raw float32 values: a file
  forecast 0.25 of a window std away from `predict` at a level of 1e5 over a std of 8, and 5.8 at 1e6), and the scale does not depend on the level. A context is FLAT when
  `std <= transform_args.scale_floor * mean|context|` (the default is 1e-6, about eight float32 steps of relative resolution): its scale
  is then 1.0, the series' own unit, so a flat zero context (the absolute floor of 1e-8 that the first version used turned one
  `sparse_events` window into a standardized target of 1.08e9) and a flat context at any level give the raw deviation of the target.
  What this gives, and where it stops: shift invariance holds while a window's std is above `transform_args.scale_floor` (1e-06) times its mean |x|, that is while the level over the std is below 1e6. Inside the bound a window shifted by a constant is standardized to the same numbers and forecast shifted by it (measured through `predict` on float64 windows of `nbeats/tiny` and `prism/tiny`, window std about 5.8: the forecast of the window shifted by 0, 50, 1e3, 1e5 and 1e6 is the shifted forecast to 2.2e-11 of a window std at most, the last at a level over the std of 1.7e5; at 1e7 and 1e9 the window is flat and the same probe gives 0.064 (`nbeats`) and 0.38 (`prism`) window stds, the same at both levels because a flat window is forecast from its raw deviations, and the review's other heads at 1e9 gave 0.03 to 3.6). Beyond the bound the window is FLAT: a flat window is trained and scored in the series' raw units, not standardized, and `predict` returns `flat`, one bool per window, so the caller sees which windows were forecast that way. Measured at a std of about 8: the three twins agree with a float128 oracle to 6e-8 (numpy, tf) and 1.8e-7
  (float32 function) of a standardized value at levels 0, 50, 1e3, 1e5 and 1e6; a float32 `keras` or `saved_model` file (nbeats and prism quantile measured) agrees with `predict` on the same windows
  to 1e-4 of the window std plus one float32 step of the forecast itself; at 1e6 that step is 0.0625, 0.8 percent of this std, so a float32 window
  at 1e6 carries its own rounding and no float32 function can forecast it better (inside the bound `predict` of float64 windows holds the shift to 1e-10 of a std). `lookback` must be at least 2 (one context value has no spread: refused by name before the claim), and a csv is refused before
  the claim when more than 10% of the windows of a segment (train, select or test, at stride `data.horizon` and at the resolved `data.lookback` and
  `data.horizon`) have a flat context (a series near 1e12 moving by 8, the review's csv shifted by 1e7, a long constant stretch), with a sentence naming the
  segment, the flat count and the two causes (a level large against the spread: remove it; constant stretches: fill or cut them); the check, the
  training function, `predict` and the summary read one rule, `forecast_series.context_spread` (a column that never moves is left out of the count: its unit
  scale loses nothing). A column whose nonzero max |x| is outside [1e-12, 1e15] is refused before the claim too, with a sentence naming the
  float32 range of the exported function (it squares deviations in float32: a 1e40 column trained and its keras file returned NaN, and 1e200 overflowed the float64
  std). The numpy twin is `context_stats_np` (returns the anchor, the offset from it and the scale), the tf twin `standardize_tf`
  and the keras-ops twin `context_stats_ops`; tests hold each equal to the numpy one.
  (4) No stacked L2: the legacy kernel regularizers and the AdamW decay both decayed the weights; here the optimizer's decay is the
  only one and `sgd` with a positive decay is refused (the adapters build no L2 penalty). (5) Warmup is in epochs (`optim_args.warmup_epochs`,
  as in every task), the legacy one in steps (1,000 on the command line, 5,000 in the dataclass): the recipe's one epoch is 500 steps. (6) `jit_compile` is `false`
  for every family unless you ask: Keras 3.8 resolves `auto` to XLA whenever a GPU is present, so a GPU run and a CPU run would build
  different graphs, and a GPU-only defect hides under CPU tests. Measured on GPU 0, batch 128: `false` costs 3.6x (`timestral`) to 11.4x
  (`xlstm`, 484 ms against 42.5 ms a step) per step, `tirex` resolves `auto` to off anyway, and `ets` FAILS under XLA
  (`InvalidArgumentError` in the gradient of its loop), so `jit_compile=true` is refused for `ets` and is an opt-in for the rest.
  (7) `tirex`, `nbeats` and `xlstm` normalization is forced off (see Families). (8) Ten epochs of 500 steps (the legacy epoch counts
  are not used): 400 steps already beat the bar by 6.9 to 8.7 percent for three seeds and 100 and 200 steps do not, so no default is
  shorter. (9) The validation windows are fixed, not a fresh noise draw, and the whole stream is stateless (the next two bullets);
  the legacy path was data-bound (274 ms a batch of 128 against about 60 ms of model compute).
* **The data, and what validation does NOT prove.** The bank is `data.realizations` independent draws of every pattern of
  `TimeSeriesGenerator` (69 patterns, so 276 series of 10,000 points at the defaults), each a pure function of
  `(seed, pattern, realization)` (a fresh generator per series, a sha256-derived seed: the same bank in any process and any
  `PYTHONHASHSEED`; a non-finite series is dropped and counted in the summary's `series_dropped`, an all-dropped bank is a named
  error). Every series is split by time, rows 0 to 7,000 train, 7,000 to 8,500 select, 8,500 to 10,000 test at the defaults, and every
  window, context and target, lies inside ONE segment, so no validation window overlaps a training one. A train window is a
  stride-1 window of the train segment, drawn by `tf.random.stateless_uniform` keyed on `(stream seed, counter)` (a `Dataset.range`
  counter stream; no `from_generator`). The selection windows are `data.n_select` fixed windows, 8 per series: 2,176 at the defaults
  (276 x 8 = 2,208 rounded down to 17 batches of 128), a function of `(seed, split, index)` and of `batch_size` (the windows are a prefix of one
  sequence cut to whole batches, so the count follows the batch: 2,176, 2,208, 2,200 and 2,048 at batch 128, 32, 100 and 512, and the baselines and the bar
  are computed on the windows of the run, so `val_mae` is comparable between runs of ONE batch size), so `val_mae` is a function of the
  weights alone. **What this does NOT prove.** Train, selection and test series come from the same 69 archetypes and the same noise
  draws' neighbours (two realizations of a seasonal pattern correlate at 0.987 to 0.995), and validation is temporal within each series (a validation window is the FUTURE of a
  series the model trained on), and a periodic pattern (`energy_consumption`: 5 of the 2,176 selection windows at seeds 42 and 7) can hold a selection window with an exact
  twin among the train windows of another realization: beating the baselines is a check that the wiring and the learning work, not a claim about unseen real
  series or unseen patterns. Four realizations were chosen by measurement (the standard error of the baseline mean on the selection
  windows 0.0107 against 0.0179 at two; no train-versus-select gap at two, four or eight); do not lower it to two (the bar then flips
  on seed noise) or raise it without measuring the bank build (about 3 s at four).
* **A csv.** `data.format=csv` reads ONE wide table, a row per time step: `data.path`; `data.value_cols` (each column numeric and finite,
  they become the channels; null is every column except `data.time_col` and an unnamed index column); `data.time_col` (optional; when
  given it must be strictly increasing, else the run is refused, and when absent the row order IS time). The rows are cut into train,
  select and test segments by `data.select_fraction` and `data.test_fraction`, every window lies inside one segment, the selection
  and test windows have non-overlapping targets (stride `data.horizon`) and the train windows stride 1. **Refused by name before the
  claim and before TensorFlow loads** (`train/api/forecast_series.py`, pandas and numpy only): a missing file or column; a column that
  is not numeric (a column with one text cell is refused, not dropped, which a numeric selection would do silently); a non-finite
  cell (`inf` included); an unsorted, repeated or unparseable time column; a segment shorter than `data.lookback + data.horizon`; a
  selection or test segment that holds fewer than one batch of non-overlapping windows, with the row count it needs in the message
  (21,600 rows at the defaults: 168 + 24 x 128 = 3,240 rows per segment over 0.15); a request of more selection or test windows than a
  csv segment holds; a csv whose select segment is constant in every value column (every window is flat, naive-last and seasonal-naive
  score exactly 0 and the bar could be neither met nor judged); a csv of which more than 10% of the windows of a segment are flat, and a value column outside the float32 range
  (see the standardization above); more channels than the family takes. Not offered:
  a long format with an id column, gap handling, resampling. A resume refuses a changed file (the fingerprint holds the sha256 of its
  bytes and every window key, and `task.json` records the same sha256).
* **What is refused before the claim, and what is not** (D-041, D-042). Before `out` is claimed and before TensorFlow loads
  (`TimeSeriesForecastingTrainer.check_model_args` and `TimeSeriesForecastingConfig.problems`): `precision` other than `float32`; a
  `data.lookback` below the family's measured minimum, a number of the variant (`forecast_models.window_problems`, the
  `FamilyRow.min_lookback` rows, measured by building each variant at decreasing lookbacks: `prism` tiny 8, small 31, base 31, large 61;
  `ets/AAA` needs `model_args.seasonal_period` plus 1; an `ets/AAA` run without `model_args.seasonal_period` (an int of at least 2) is refused by name; every other measured variant builds at 2, the floor of every family); a
  `data.horizon` below 2 for `nbeats` (its seasonality stack has `horizon // 2` harmonics); `data.n_select` or `data.n_test` of fewer
  than one batch, and a `data.patterns` subset whose 8 windows per series are fewer than one batch; for a csv, a segment of fewer than
  one batch of windows, more windows asked than a segment holds and a constant select segment (`forecast_series.check_split_sizes`, the
  functions the loader calls). `tirex` large and `timestral` 28M, 123M and 209M were not built when the minimums were measured and take
  the floor; the minimum of a prism moves with `model_args` that change its `tree_depth` or `num_wavelet_levels`, which the table does not
  know, so such a run is refused by the library at the build. **Still refused after the claim** (exit 2 with `out` in use), because the
  refusal needs the generator or a standardized window, which load TensorFlow: an unknown `data.patterns` name; the default window count
  (8 per series) of `data.patterns` null against the batch size, because the generator names the patterns; a bank series that is
  non-finite (it is dropped, and the run is refused only when none is left); and a select set that is flat in SOME windows only, where
  the baselines (numbers of standardized windows) come out 0.
* **The bar** (D-008, D-037, D-038 of plan 3a8319ef). `success_metric` is `val_mae`, the mean absolute error of the point forecast
  (the median of a quantile head) in context-standardized units, so it is scale-free across series. `success_value` is the lowest
  `val_mae` of the selection window, `success_threshold` is `min(naive-last, seasonal-naive, epoch-zero)` and the bar is met only when
  `success_value` is more than `TIE_ULPS * max(1, ceil(sqrt(n_batches / 16)))` float32 steps
  (16 up to 16 select batches, 32 at the 17 of the defaults, 192 at 2,048; `forecast_run.tie_band`, the width `TIE_ULPS` imported from
  `segmentation_run`, `n_batches` the batch count of the selection windows) BELOW the threshold, both read as float32: equal or within the band is a miss (D-043: the float32 accumulation of the
  compiled metric drifts with the number of selection windows, measured at +20 steps at 262,144 windows, outside a fixed 16), so a model that outputs the naive-last forecast cannot pass by rounding (it scored one float32 step under its
  float64 baseline on 2 of 8 seeds before the band). Met is exit 0, missed is exit 1 with the numbers in `results_summary.json`. The
  three terms are measured on the SAME fixed selection windows the checkpoint is selected on, never on the test windows: naive-last
  repeats the last context value; seasonal-naive picks, per window, the period from 1 to `min(lookback // 2, lookback - horizon)` with the lowest backtest MAE
  over the context's last `horizon` points and repeats it (`forecast_run.seasonal_naive`, moved from the legacy `timestral` trainer
  unchanged, which now imports it); both are numpy and know no model, computed in `build_data` (a resume computes them again and gets
  the same numbers); the epoch-zero value is the untrained model's `val_mae`, read from `checkpoints/fit_state.json` by
  `evaluate_and_report` and never from memory, because a resume never evaluates the untrained model. `epoch_zero.binds` in the summary
  says whether it was the smallest term. For `ets` it can be: its untrained forecast (ANN, 0.837) sat below naive-last (0.9218), so the
  bar there means "beat the untrained model", and a miss is reported, not softened. There is no `success_threshold` field (the
  pretrain task has none either): a bar a caller can weaken says less. `nbeats/base` on the default bank, CPU, one seed (step 8 of
  the plan, before the gate): `val_mae` 0.5792 after 10 epochs, 30.2 percent below the better baseline. The summary also carries `baselines` (`naive_last`, `seasonal_naive`,
  `context_mean`, which is reported and not in the bar, and `n_windows`), `epoch_zero`, `canonical_metrics` for the selection and
  the test windows (MAE, RMSE, sMAPE, rMAE against naive, MASE scaled by the context's own one-step differences, and for a quantile
  head coverage and sharpness), `lookback`, `horizon`, `channels`, `n_series`, `series_dropped`, `segments`, `flat_windows` (the flat select and test windows, and the share `train_fraction` of the train segment's windows at stride `data.horizon` that are flat: a flat window is scored in the series' raw units, so a non-zero count says part of `val_mae` is not in standardized units) and `stream`; the run
  directory is the shared one plus `task.json` and `visualizations/forecast_test.png` (one forecast figure of the test windows).
* **The stream seed on a resume.** The train stream is endless and a resumed `fit` starts a fresh iterator, which would replay epoch
  0's windows. The task rebuilds the stream (`TimeSeriesForecastingTrainer.resume_stream`): the shared sequence passes
  `completed_steps = start * steps_per_epoch + (batch + 1 if batch else 0)` and the task returns the data on a stream seed of
  `int.from_bytes(sha256(f"{seed}:{completed_steps}")[:4], "big")` (`derive_stream_seed`, the pretrain task's function). Only the
  stream seed moves: the bank, the split, the counts and the selection and test windows keep `seed`, so no selection window ever
  reaches the train stream and the bar stays comparable across a kill. The cost: a resumed run is NOT the unkilled run's
  continuation (its data order is a function of the completed-step count), and a killed epoch is trained again from its first batch
  unless `checkpoint_args.backup_every_steps` bounds the loss. `run.log` carries `stream seed S, first train batch sha256 D`, computed
  from the pure map at counters 0 to 2, so a resumed run is seen to differ from epoch 0's stream; the summary's `stream` block holds
  both. The help of `resume_interrupted` says it in one clause (`TaskTrainer.resume_order`).
* **`predict`.** `python -m train predict model=<run dir or .keras> data=<csv, .npy or .npz> out=forecast.json` writes ONE new JSON
  file (`out` must not exist): `task`, `model`, `data`, `lookback`, `horizon`, `channels`, `quantile_levels`, `point`, `quantiles` and `flat`.
  A csv gives the LAST `lookback` rows of the record's `value_cols`, after the loader's own check that the record's `time_col` is strictly increasing (a newest-first export is refused, not forecast from its oldest rows; a run that named no time column skips the check); a `.npy` or the one array of a `.npz` is `(lookback, channels)` or
  `(N, lookback, channels)`; a directory is refused by name ("a directory, not a file"), exit 2. From Python,
  `load_model(...)` returns a `ForecastPredictor` and `ForecastPredictor.predict(x)` takes an array, a list of windows (each an array or
  a csv path) or a csv path and returns `{"point", "quantiles", "quantile_levels", "flat"}`: `point` is `(N, horizon, channels)` float64 in the
  series' OWN units (the median for a quantile head), `quantiles` is `(N, horizon, channels, Q)` and the levels are the record's, and
  both are `None` for a point head, and `flat` is a bool `(N,)`, True for a window whose context is flat in a channel (forecast in the series' raw units). A window is forecast from its own context statistics only (the numpy twin), so a window scaled by a constant gives the forecast
  scaled by it and a window shifted by a constant gives the forecast shifted by it while its `flat` stays False, that is while its std is above `scale_floor` times its mean |x| (see the
  standardization above for the measured bound). A window of another length or
  channel count, a non-finite cell, an empty input and a csv with a missing column are named errors; a model whose output disagrees
  with the record (decode row, head, levels, median, horizon, channels) is refused at load.
* **`export`.** The file takes RAW float32 windows `(batch, lookback, channels)` in the series' own units and returns the forecast
  `(batch, horizon, channels)` in the same units; the standardization and its inverse, and the decode of the family's output, are
  inside it (`ExportRule.served`, `ServedForecaster`, one registered layer that the `keras` archive also holds). **The `keras` file
  needs the train package importable** (the source tree on the path, or an editable install): `ServedForecaster` is a class of the
  train package, which the wheel does not ship (`packages.find` takes `dl_techniques*` only), so `keras.saving.load_model` of the file in
  an interpreter without `train` stops with Keras's own `Could not deserialize class 'ServedForecaster' because its parent module
  train.common.forecast_run cannot be imported` (a test loads all nine files in a fresh interpreter that imports
  `train.common.forecast_run` and `keras`, and pins that sentence for an interpreter without `train`). `saved_model`, `tflite` and `onnx` carry
  no such class. The library function `train.api.export(format="keras", input_shape=...)` holds `input_shape` to the record's window
  like the other formats (the command line has no such key). A quantile head returns `{"point", "quantiles"}` from `saved_model` and `keras` and the point from `tflite` and
  `onnx`; a `timestral` file takes ONE window per call. The matrix, derived from `check_export` (36 cells, 31 export and 5 are refused by
  name, before anything is written):

| Decode row | `keras` | `saved_model` | `tflite` | `onnx` |
|---|---|---|---|---|
| `nbeats`, `prism_point`, `prism_quantile`, `tirex`, `mdn` | yes | yes | yes | yes |
| `xlstm_point`, `xlstm_quantile` | yes | yes | yes | REFUSED |
| `ets` | yes | yes | yes | REFUSED |
| `timestral` | yes | yes | REFUSED | REFUSED |

  The five refusals and their measured reasons: `tflite` of `timestral` (the converter stops at `Cumsum` with built-in operators and at
  `Asinh` with select TF operators); `onnx` of `timestral` (the file is written and onnxruntime cannot initialize it); `onnx` of both
  `xlstm` rows (onnxruntime `InvalidGraph` at a `Pad` in an mLSTM block's causal convolution); `onnx` of `ets` (the converter cannot
  find a `while` output). `tflite` of `tirex`, `xlstm` and `ets` uses select TF operators, so the file needs the Flex delegate. Every
  cell that exports matched Keras on the CPU: the worst gap, as a share of the window's standard
  deviation, was 3.4e-6 for `saved_model`, 5.3e-5 for `tflite` and 7.1e-5 for `onnx` (measured in plan 3a8319ef, tiny models of every
  family at `(None, 64, 1)`).
* **Limits of the task.** No `deepar` and no `adaptive_ema` (above); `timestral` only up to one patch of horizon; a long-format csv,
  gaps and resampling are not offered; `analysis_args.conformal_alpha` is refused; the model analysis is data-free (weights and
  spectral only, since the stock analyses read class probabilities) and for `ets`, with one parameter, the analyzer finds nothing and the
  summary says `skipped`; mixed precision is refused (see the keys above); the numbers are in context-standardized units, so a `val_mae` is not in the
  series' units (`predict` is). Beating the baselines on the synthetic bank is a learning check (see "What this does NOT prove"), and
  the recipe's learning rate and batch were measured on `nbeats` only; the other families run the same recipe unless the table says
  otherwise and carry no tuning claim.
* **Gate numbers** (plan 3a8319ef: the GPU measurement of this task, run on HEAD `1904f07a8` of the branch `train-api/phase-4e`; a later gate re-runs the same bar on the HEAD it names and replaces these numbers if they move). Every run on GPU 0, an RTX 4090, one job at a time, `CUDA_VISIBLE_DEVICES=0` and no `--gpu`; the log of each names the device and the worktree `src` as the origin of `dl_techniques` and `train`.
  The bar is the one above (best `val_mae` more than the tie band of float32 steps under the smallest of naive-last, seasonal-naive and epoch-zero, all on the same fixed selection windows; the band was 16 steps when these gates ran and is 32 at their 17 select batches under the rule above) and was written before the first run: G1 and G2 are gated and G4 reports its verdict; G3 records the verdict of each family and gates none (300 steps are too few to expect every family to meet it). A miss is reported as it is, with no re-run. Every number is read from the run's `results_summary.json`, `training_log.csv` and log by script. A step time is the median over epochs of Keras's ms per step, then the mean seconds of the epochs after the first (which include the validation pass); the host was loaded (a foreign process held about 2.5 of 12 cores), so these are not the 35 ms the earlier measurement gave for `nbeats/base`. Memory is nvidia-smi's figure for the run's own process, sampled every second.

| Arm | Task | Epochs | Params | Step, epoch | Baselines | Epoch zero | Threshold | Best `val_mae` | Verdict | Final is best | Peak GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | `nbeats/base`, synthetic bank | 20 of 500 steps | 2,326,272 | 69 ms, 34.3 s | naive-last 0.8857, seasonal-naive 0.8303 | 4.9937 | 0.8303 (seasonal-naive) | 0.5574 | met, exit 0 | true | 526 MiB |
| G2 | `nbeats/base`, killed in epoch 3 and resumed | 6 of 500 steps (the 4 epochs of the resumed process timed) | 2,326,272 | 80 ms, 42.1 s | naive-last 0.8857, seasonal-naive 0.8303 | 4.9937 | 0.8303 (seasonal-naive) | 0.5984 | met, exit 0 | true | 526 MiB |
| G3 | one run of each other family, 3 epochs of 100 steps (verdict recorded, not gated) | 3 of 100 steps (each) | prism point 7,168; prism quantile 5,090; tirex 347,288; xlstm point 235,232; xlstm quantile 238,352; ets 1; timestral 5,208,992; mdn 142,197 | prism point 65 ms, 5.8 s; prism quantile 61 ms, 5.4 s; tirex 63 ms, 5.6 s; xlstm point 603 ms, 59.3 s; xlstm quantile 586 ms, 57.7 s; ets 180 ms, 17.8 s; timestral 99 ms, 9.0 s; mdn 30 ms, 2.4 s | naive-last 0.8857, seasonal-naive 0.8303 (the windows of G1, all eight) | prism point 1.0076; prism quantile 1.7379; tirex 2.7795; xlstm point 1.1869; xlstm quantile 1.3164; ets 0.8019; timestral 1.2227; mdn 0.9971 | prism point 0.8303; prism quantile 0.8303; tirex 0.8303; xlstm point 0.8303; xlstm quantile 0.8303; ets 0.8019; timestral 0.8303; mdn 0.8303 | prism point 0.9915; prism quantile 1.0367; tirex 0.9030; xlstm point 0.9459; xlstm quantile 0.9764; ets 0.7885; timestral 0.6112; mdn 0.6615 | prism point missed; prism quantile missed; tirex missed; xlstm point missed; xlstm quantile missed; ets met; timestral met; mdn met | prism point true; prism quantile true; tirex true; xlstm point true; xlstm quantile true; ets false; timestral true; mdn true | prism point 530 MiB; prism quantile 530 MiB; tirex 702 MiB; xlstm point 5010 MiB; xlstm quantile 5010 MiB; ets 452 MiB; timestral 1426 MiB; mdn 462 MiB |
| G4 | `nbeats/tiny`, a csv written from the bank | 20 of 63 steps | 73,024 | 48 ms, 2.2 s | naive-last 1.2432, seasonal-naive 0.3822 | 1.5340 | 0.3822 (seasonal-naive) | 0.1545 | met, exit 0 | false | 404 MiB |

  G1 met the bar by 32.9 percent under the better baseline (best 0.5574 against 0.8303; the best epoch is the last, so it had not stopped improving; the test windows score 0.5692 MAE on the best weights, rMAE 0.6178). G4 met it by 59.6 percent (its best epoch was epoch 17 of 20, not the last). In G3, 3 of 8 families met the bar at 300 steps (ets, timestral, mdn) and 5 missed it (prism point, prism quantile, tirex, xlstm point, xlstm quantile); the `ets` bar is its own untrained value (0.8019, below naive-last), so it means beating the untrained model. The best checkpoint's reload gap is exactly 0.0 in all 12 runs.
  **jit_compile (G3).** `xlstm/tiny` with `jit_compile=true` ran at 62 ms per step against 603 ms at the default, 9.7 times faster, with nearly the same untrained value (1.1866 and 1.1869); `ets/ANN` with `jit_compile=true` exited 2 in 5 s with the named refusal and created no run directory.
  **Kill and resume (G2).** `checkpoint_args.backup_every_steps=100`; the recorded PID was sent SIGKILL (that PID only) when the CSV held 2 rows and the backup's metadata first read epoch 2 batch 99, inside the third epoch; the same command with `resume_interrupted=true` finished with exit 0. The copy held 1100 optimizer steps (2 x 500 + 100) and the final copy 3000 (6 x 500): no step lost or run twice after the copy. The CSV has one row per epoch (0 to 5), the two rows written before the kill are unchanged, and the learning rate it records at each epoch start equals the run's own schedule at the step count the process had settled at (epoch 2, the killed epoch, starts at 1100 rather than 1000, as `backup_every_steps` documents). The first-batch digest of the killed process is `551e739ed829253c...` (stream seed 42) and that of the resumed process `34c39c6781bc477c...` (stream seed 1850174256, `derive_stream_seed(42, 1100)`): they differ. The final `val_mae` 0.5984 is under the bar of 0.8303. Not shown: that the real resumed `fit` read exactly the logged windows (the CPU test pins that on fakes), and GPU training is not bit-reproducible across processes (the untrained `val_mae` is the same float in G1 and G2, the epoch-0 trained one is not).
  **Predict and export of G1 and G4 (CPU).** `python -m train predict` of the G1 `best_model.keras` and run directory and of the G4 run directory (a csv and an array) exited 0 and equals `ForecastPredictor` to 4e-16 (float64 JSON). All eight export cells (keras, saved_model, tflite, onnx for each run) exited 0 and matched `ForecastPredictor` on raw windows: the worst gap, as a share of the window's standard deviation, was 2.3e-06 for `keras`, 2.3e-06 for `saved_model`, 3.1e-06 for `tflite` and 2.0e-06 for `onnx` (tolerances 3e-5, 3e-5, 5e-4, 5e-4), and the same windows scaled by 1000 and shifted by 50 gave gaps of the same size (2.0e-06 or less), so the files do the standardization themselves. No `nbeats` cell is refused, so no named refusal was exercised there.
  The G4 csv is 3,000 rows of `trend_daily_seasonal` and `multi_seasonal` from the bank with a timestamp column; at the recipe's batch of 128 a file that size is refused by name (the selection segment must hold one batch of non-overlapping windows), so G4 ran at batch 32 with lookback 48 and horizon 12.

**Keys**: `key=value` tokens; a dot reaches a group (`optim_args.lr=3e-4`, `data.dataset=cifar10`);
`config=run.yaml` loads a file first and the command line wins over it. A value is read by
the field's type (`true`/`false`, `3e-4`, `null`, JSON for a list or mapping; a string is verbatim). A list is JSON with every text
quoted, and the whole token goes in single quotes for the shell (the list `["pet","background"]` for `data.class_names`);
`[pet,background]` is refused with that sentence in the message, for every list and mapping key of every task. An unknown key, a wrong type or a bad choice is refused with the list of all
problems and a "did you mean". `auto` asks for the family's recipe default.
`data.format` is `builtin` (the library's datasets), `folder` (`data.train_dir`, one directory of
images per class, optional `data.test_dir`) or `csv` (`data.train_csv`, columns `path` and `label`).
**A data file and dotted keys are two layers of one group** (plan 98b866fb 7.1, D-021): `data=<file>` is the path of a YAML
file holding the group, and `data.<key>=` tokens put over it, key by key, so a capped or tweaked run needs no copy of the file.
The same rule holds for all five tasks and for the `data: <file>` of a `config=` file; the lowest layer is the `config=` file, then
the group's file, then the dotted keys, and the file's `format` reads dotted keys that state none (`data.path_column=` after a
csv file); a `data.format=` that differs from the file's is refused naming both formats (give a file of that format, or every key on the command line with no file). A key given twice on the command line, a file twice, or a file together with a JSON object for the group is still
refused as a repeated key. `config.json` records the resolved values, as for every key, and not which layer a value came from.
```bash
python -m train image_classification out=auto model=vit/vit_pico data=cifar_100.yaml data.max_samples=2000
python -m train image_segmentation out=auto model=convunext/tiny data=<out>/data.yaml data.max_samples=4000 'transform_args.image_size=[256,256]'
```

**The run directory**: `out=auto` makes `results/<timestamp>...`; an existing `out` is refused,
nothing is overwritten or deleted. A run writes `config_requested.json` (what was asked),
`config.json` (resolved), `run.log`, `training_log.csv`, `results_summary.json`, `task.json`
(class names, input shape, mean and std: what a model needs to be used without its config; a segmentation, denoising or forecasting run writes the keys of its own record, a `pretrain` run writes none),
`best_model.keras`, `final_model.keras`, the figures and analysis, and `checkpoints/`
(`backup/`, `fit_state.json`) and `run.lock` for resume. `run.lock` is a kernel `flock` held for the
life of the run and released when the process ends, a SIGKILL included, so two resumes cannot both run
and a reused pid blocks nobody; the pid in the file is a label for the refusal message.

**Resume**: `resume_interrupted=true` with the same `out` and the same request continues a killed
run. It is refused for a finished run (one with a `results_summary.json`), another request, other data, a live
process holding the run, or a torn backup; `devices` and `analysis_args.eval_batch_size` may differ. The training data
order replays epoch 0's, except for `pretrain` and `time_series_forecasting`, which train a resumed run on a fresh stream (their sections above), and a half-done epoch is trained again (the key's help lists the rest).
A run killed **before** a code change that alters training (step 10b of plan 98b866fb: the ConvNeXt V1 layer scale starts at 1e-6 and is
no longer clamped or decayed; step 10c: ConvNeXt V2 blocks have no layer scale by default and a zero GRN gamma; step 10d: the CIFAR-10
AutoAugment policy; step 10e: the ResNet decay and clip) and resumed **after** it continues under the new rules: its early epochs trained
under the old ones and its later epochs train under the new ones, and nothing in the run records which epoch changed. Start such a run again.
A run whose **training finished and whose reporting died** (the sidecar says `fit_finished`, there is no summary: the
figures, the analyzer, the conformal sets or the final evaluation raised, for example out of memory), or whose
**process died between the last epoch and `fit` returning** (every epoch ended, or early stopping had waited out its
patience; no `fit_finished` yet: the window holds the dashboard's last draw and the restore of the best weights), is
not trained again: the same command with `resume_interrupted=true` skips `fit` and runs only the reporting, from
`best_model.keras` (the best weights), `checkpoints/backup/latest.weights.h5` (the last epoch's weights, which were
only in memory) and `training_log.csv` (read, never written: its sha256 is the same before and after). The last
epoch's weights are used only when that backup is the epoch-end copy of the epochs that ran (metadata epoch and
batch 0, optimizer step count epochs times steps); a missing, torn, mid-epoch or older copy is refused with exit 2
naming `latest.weights.h5`, nothing is changed, and the previous `.bkp` copy is never put in its place, because a
final model one epoch old would look like the final model. An existing `final_model.keras` is written over, and
it holds what a whole run's holds: the last epoch's weights and the last epoch's optimizer state (moments and step
count, restored from the same backup copy). The summary says `reporting_resumed: true`, `epoch_times` and `fit_wall_seconds` are null (no epoch ran in that segment),
`best_checkpoint_max_abs_diff` compares `best_model.keras` with itself, and a second resume is refused. The reason a
reporting died is an `ERROR` line of `run.log` (exception type and message) that ends with a sentence saying
whether `resume_interrupted=true` can run the reporting again, worded from the checks the resume applies (a run
with `checkpoint_args.save_best=false` has no best checkpoint, so the line says the resume would be refused and
why); an exception of `fit` is logged the same way (`Training failed: ...`), a `SystemExit` or `KeyboardInterrupt`
is not logged as a failure, and the command exits 1. If the analyzer of a reporting-only resume writes nothing, the
summary says that `analyzer.path` names the earlier segment's file. For folder and CSV data the sidecar holds a fingerprint of the
files: `data.fingerprint=listing` (the default) hashes every file's relative path, class and size, in listing
order, and opens no file (a CSV whose rows were reordered splits differently, so it is another listing), so two
same-sized files swapped, or a pixel edited, go unseen; `data.fingerprint=content` also hashes every
file's bytes, streamed, which reads all the data once (a pass over the disk tree, minutes on a large one on a
spinning disk; the seconds go to `run.log`). A resume refuses a changed mode or digest, and a sidecar written before the rows were hashed in listing order
(no `data_fingerprint_scheme`).

**Figures, analysis and TensorBoard**: five keys decide what is written besides the models and the
summary. The region between the markers below is read by
`tests/test_train/test_api/test_the_knob_docs_name_real_keys.py`, which checks the config keys (and any `=value`), the
defaults in the table, the hook names and signatures, the file paths and the output file names it mentions against the
code, and the cost table against its evidence file, so a stale one of those fails the test. It does not check the
wording of a description (what a knob writes or needs) or any number that is not a cost in that table; the test's
docstring lists what is unchecked.

<!-- knobs:begin -->
| Key | Default | What it does |
|---|---|---|
| `analysis_args.figures` | `true` | Writes `visualizations/`: `training_dashboard.png` (redrawn after epoch 1, about 20 times per run, and at the end), then after the best weights are loaded `confusion_matrix.png`, `per_class_metrics.png`, `confidence_calibration.png`, `misclassifications.png` and `classification_report.json` (these five need a test split). `false` writes none of them and `ece` in the summary is null |
| `analysis_args.model_analysis` | `true` | Runs the full model analyzer once, after training, on the best weights, into `model_analysis/` (`analysis_results.json` and seven PNGs: weights, spectral, information flow, calibration, training dynamics and the summary dashboard). Needs a test split. `false` leaves `analyzer.status` at `skipped` |
| `analysis_args.epoch_analysis_every` | `0` | `N > 0` runs the data-free epoch analyzer (weights and spectral only) before training into `epoch_analysis/epoch_000/` (the untrained model), after epoch 1 into `epoch_001/`, then at epochs N, 2N, ... into `epoch_NNN/`. `0` is off. A failure on a built model stops the fit, so no snapshot is silently missing. A resumed run does not redo `epoch_000` and `epoch_001` |
| `analysis_args.conformal_alpha` | `null` | A level strictly between 0 and 1 (0.1 aims at 90% coverage) writes `conformal.json` and `conformal_model.keras` (outputs logits, probabilities and set membership), calibrated on the test split. `null` skips it and the summary's `conformal` is null. A level with no test split is skipped with a note |
| `analysis_args.eval_batch_size` | `"auto"` | Images per pass of the predictions made after training: the figures' predictions (every dataset, folder and csv included), the model analyzer's predictions (it caps its own 32 at this value when this is lower; its information-flow pass sizes its own batch from the analyzer's memory budget and does not read it) and the final-model load check (the same cap), and the validation and test splits of the in-memory datasets (CIFAR, MNIST). The validation and test splits of ImageNet, Imagenette, folder and csv data are batched at `batch_size` and ignore it. `auto` is the recipe's value, and 256 where the recipe says nothing, so a default run evaluates exactly as before. The cliffordnet recipes on ImageNet and Imagenette set the batch they train at (8 for `cliffordnet`, 16 for `cliffordnet_hier`), because 256 images of 224 px ask 18 GB. Those recipe values are keyed to the dataset: folder and csv data take 256 whatever the family or the image size (a run on files of 224 px or more with a family whose ImageNet recipe sets a batch logs a warning naming `batch_size` and this key). It changes no weight, and it may differ when a run is resumed |
| `checkpoint_args.backup_every_steps` | `null` | `N >= 2` (and below the steps of an epoch) also writes the resume backup after every N-th batch inside an epoch, so a kill loses at most N batches of training; `null` backs up at each epoch end only. A resume continues the killed epoch on the weights of the last copy and runs only the batches that copy had not taken, from a fresh pass of the data (some examples of that epoch repeat and some are missed); the optimizer's step count and the learning rate of every step are the unkilled run's, and that epoch's row holds the train metrics of its remaining batches only. Each copy writes the weights and optimizer state twice (the previous copy is kept); the cost per save is in the backup table of the ImageNet section |
| `logger_args.tensorboard` | `{}` | `{}` writes TensorBoard event files into `tensorboard/` (`train/` and `validation/`); `null` turns them off; a dict with keys is refused, because no TensorBoard option is wired |

Measured cost, one run each, so a difference of a few seconds is inside the noise. Setup: `vit/vit_pico` on 240 fake
CIFAR-10 images (240 test images), 3 epochs, batch 16, CPU with one thread. Each row turns on the one setting, against a
base run with figures, model analysis and epoch analysis off and TensorBoard on (the `base` row, wall seconds of the whole
call); the TensorBoard row is the base run against one with TensorBoard off. The cost grows with the number of layers and
test images. The durations of the knobs are stated only in this table, and `test_the_knob_docs_name_real_keys.py` recomputes every row from
`tests/test_train/test_api/knob_cost_evidence.json` (a copy of the measurement in the untracked
`plans/plan-2026-10-31T180033-2faf890a/scratch_step13/measure_out.txt`; the test does not re-measure it).

<!-- costs:begin -->
| Setting | Extra wall seconds | Output, MB |
|---|---|---|
| `base` | 68.0 | - |
| `analysis_args.epoch_analysis_every=1` | 29.3 | 6.4 |
| `analysis_args.model_analysis=true` | 23.4 | 3.1 |
| `analysis_args.figures=true` | 6.9 | 0.4 |
| `analysis_args.conformal_alpha=0.1` | 4.2 | 32.6 |
| `logger_args.tensorboard={}` | 3.0 | 0.2 |
<!-- costs:end -->

`epoch_analysis_every=1` writes four snapshots (`epoch_000` to `epoch_003`). The `conformal_model.keras` of the conformal
row is a full copy of the model. The `base` row is the whole call, not an extra, and the TensorBoard row is inside the
noise of one run each.

**A failed analysis is named, with its reason.** The analyzer isolates a failing analysis and carries on, so
a run can write a normal-looking `model_analysis/` after one of its parts failed. For this task the summary's
`analyzer` block gains `summary.analyzer.failed_analyses`, an object that maps each name to its reason, only when
it is not empty (a clean run writes the same keys as before), and `run.log` carries a WARNING. A name is an
analysis that was attempted and left no result (`weights`, `calibration`, `information_flow`,
`training_dynamics`, `spectral`), a figure set that raised (`visualizer:calibration`, `visualizer:summary`, and so
on), `analysis` (a failure before any analyzer ran) or `model_analysis` (the whole call raised and returned
nothing). The reason is `ExcType: message` of the swallowed exception, or a sentence for an analysis that ran and
produced nothing (spectral on a model whose layers all have fewer than `spectral_min_evals` = 10 eigenvalues
says so). When any analysis is named, `summary.analyzer.status` reads `partial` instead of `success`; `error`,
`missing` and the other statuses are unchanged. The analyzer writes `model_analysis/analysis_results.json` before
it draws any figure, so a figure that crashed loses no computed number; the file gains an `analysis_failures`
object only when something failed. In the analyzer's summary dashboard a missing ECE or Brier prints `n/a`, not 0.0.

**For a task author**: the trainer calls two optional hooks and one required one for this output. They are
documented at their definitions in `api/trainer.py`: `TaskTrainer.figure_callbacks(config, data, vis_dir)`
returns per-epoch figure callbacks (called only while `analysis_args.figures` is true, placed after the
epoch log line and before the dashboard); `TaskTrainer.run_analysis(config, model, test_data, history, experiment_name, run_dir)`
returns what the model analyzer returns, `None` when it raised; `TaskTrainer.evaluate_and_report(config, data, fit)`
is the required end-of-run sequence that calls them. `src/train/AGENTS.md` has the rules.
<!-- knobs:end -->

**Predict**: `python -m train predict model=<run dir or .keras> data=<image or directory> out=pred.json`
writes a new JSON list of `{"path", "labels", "scores"}`, every class best first. A directory is
searched recursively. `out` must not exist. A segmentation run and a denoising run write a new directory of PNG
files and an `index.json` instead (the segmentation paragraph below and `image_denoising` above), and a forecasting run takes a csv or a `.npy` or `.npz` of windows and writes one JSON file of forecasts (`time_series_forecasting` above). A `pretrain` run directory is
refused (it has no `task.json`; `pretrain` above has the sentence), and so is its `export`. **On ImageNet the `labels` are WordNet ids** such as `n01806143`, not
words, and so are the `class_names` of `task.json`. The loader sorts the 1,000 ids and takes the position as the class
index; the devkit's `meta.mat` maps an id to its words:

```python
from scipy.io import loadmat
rows = [r for r in loadmat("<imagenet_root>/devkit/meta.mat", squeeze_me=True)["synsets"] if int(r["num_children"]) == 0]
words = {str(r["WNID"]): str(r["words"]) for r in rows}      # "n02012849" -> "crane"
```

The names stay ids because the words repeat, and the predictor refuses a record whose class names repeat (`'class_names'
repeats ...`). Measured on the real devkit (`meta.mat`, 1,000 leaf synsets): 998 distinct first words; `crane` is two classes
(`n02012849`, the bird, and `n03126707`, the machine; their whole `words` strings are equal too) and `maillot` is two
(`n03710637` and `n03710721`, whose strings differ: `maillot` and `maillot, tank suit`). Use the id as the key. Through the Python `Predictor`, a colour array or PIL image given to a
grey model (MNIST) is made grey on its 8-bit pixels with the loader's own rule, `(9797 R + 19234 G + 3737 B) >> 15`
(libpng's weights as `tf.io.decode_image(channels=1)` uses them), so a PNG gives the loader's pixels exactly (checked
within 1e-5, measured 0). **Measured limit:** a JPEG given as a path is read by the loader's decoder and is exact, but
a JPEG that was already decoded to an array or a PIL image has lost the Y plane the decoder takes its grey from, and
PIL's and TensorFlow's JPEG decoders differ, so its grey is off the loader's by up to 14 levels of 255 on random
colour noise at quality 75 (mean 0.56; 13 and 1.15 at quality 95) and by 1 to 2 levels (mean 0.5 to 0.6, about
0.002) on smooth images: above 1e-5, and not removable from the decoded pixels.
A 16-bit grey PIL image (`I;16` and its byte-order variants) is converted as the decoder reads the same file, each
sample's high byte, and is equal to the loader on it; PIL's `I` and `F` (32-bit integers and floats) are refused with a
`ConfigError` naming the mode, because the loader reads no file of that depth. A 16-bit RGB or RGBA PNG opens in PIL as
8-bit, which cannot be told from an 8-bit file, so for a grey model it is within 1 level of 255 of the loader (measured; a
path is exact). `load_model` refuses, at load, a `task.json` whose `task` is not one this reader knows or that its family
does not train, whose family or variant is not the model's (a classification family by its class and `get_config()`, a segmentation
family by its layers; see the segmentation paragraph), or whose `input_shape` (`input_size` for a segmentation run) the model does not take, whose `mean` is outside 0 to 1 or whose `std` is
outside 1e-6 to 1 (the images are standardized from the range 0 to 1; a `mean` of 1e30 used to predict NaN without an error),
and a `.keras` zip with no `config.json` or no weights, each with a `ConfigError` naming the problem. **Limit:** the class
names are checked for their count and for repeats, never for their order, so a `task.json` whose names are a permutation of
the right ones loads and reports the wrong name for the swapped classes; nothing in the file or the model can tell. A run whose `model_args` change a key of its variant's table
(`depth` of `cliffordnet/nano`, `blocks_per_stage` of `resnet/resnet18`, `depths` of a convnext) records those keys in an
optional `variant_overrides` object of `task.json`, written only when there is one; `predict` and `export` then compare
the model with the variant plus those changes (and `export` copies the object into `<out>.task.json`). A ViT's `scale`
cannot be overridden: choose the variant (`model=vit/vit_tiny`); the request is refused before the run directory is made.

**Predict for a segmentation run** (D-080, `train.api.segmentation_predictor`): `load_model` returns a
`SegmentationPredictor` when the record's `task` is `image_segmentation`, and `predict model=<run dir or .keras>
data=<image or directory> out=<new directory>` writes `<stem>.png` for each image, the class id of every pixel
(8 bit; 16 bit above 255 classes), and, LAST, `index.json` (`{"class_names", "masks": [{"path", "mask", "size"}]}`). A
mask has the size of its image after the EXIF orientation is applied, not the model's: the decoder is the training
pipeline's (`segmentation_data.image_to_rgb`, one function for a path and a PIL image), the image is resized to
`input_size` bilinearly as in training, and the model's logits are resized BILINEARLY to the image's size before the
argmax (a nearest-neighbour resize of the argmax is blocky and drops the margin between classes; a hand fixture whose
two answers differ pins it). The resize runs in pieces of classes when `height * width * classes` passes 2^26 floats and
the pieces give the answer of one piece. From Python, `SegmentationPredictor.predict(x)` takes a path, a PIL image or an
array (RGBA loses its alpha, grey is repeated, a floating array must lie in [0, 1] and is rounded to its byte) or a list
of them, and returns `{"masks": [...], "class_names": [...]}` ALWAYS as a list (a mask is already 2-D). A run that fails
half way leaves the PNGs it wrote and no `index.json`; two images with one stem (`a/cat.png`, `b/cat.jpg`) and a record
whose `output` is `logits` are refused before the model is loaded. **What `load_model` checks of a segmentation file (D-088, which replaces the limit D-080 stated):** the two
families are functional models with no class and no table keys in their config, but they share no layer class, so
`ModelSpec.observe` reads the layers. A file that holds no layer of the family the record names (a `convunext/tiny`
file under an `accunet/base` record, or the reverse) is refused naming the layers it does hold, and the variant keys
the layers show must equal the record's table plus its recorded `variant_overrides`: ConvUNext's `depth` (the count of
`DownsampleAndSkip`), `initial_filters` (the width of `ConvStem`), `blocks_per_level` and `convnext_version`; AccUNet's
`base_filters` and `mlfc_iterations`. It also checks that the variant is in the table, that each override is a table
key, and that the model takes `input_size` and gives one score per class name per pixel at that size. What it cannot see:
ConvUNext's `drop_path_rate` (no layer shows it) and a `model_args` key outside the table (`kernel_size`), because
rebuilding the variant to compare weight shapes was rejected (such a key changes the shapes and the record does not carry
it) and so was comparing `model.name` (`model_args.model_name` renames it); two variants that differ only there are not
told apart, and neither are two runs of one family and variant with the same class count.

**`task.json` is schema version 2** (D-073, `train.api.predictor`): a classification record has the keys and the order
of version 1 and only `schema_version` changed; a segmentation record has `input_size` as `[height, width]` (not
necessarily square), `channels`, `ignore_index`, `mask_threshold` and `output` where a classification record has
`input_shape` and `eval_transform`; a denoising record has `channels`, `size_multiple`, `data_range` and `output` and no class names,
statistics or input size (see `image_denoising`); a forecasting record has `lookback`, `horizon`, `channels`, `value_cols`, `time_col`, `scale_floor`, `head`, `quantile_levels`, `point_index`, `decode`, `data_format` and `data_sha256` and no class names, statistics or input size (see `time_series_forecasting`). `load_model` holds a forecasting record to its model: `lookback` and `channels` against the input the model was BUILT for (every family, a mismatch is a named error that names the field), and the decode row, head, levels, median, horizon and channels against what the model outputs; `time_col` is the time column the run checked, and `predict` of a csv checks the order with the loader's own function (null: the rows are the time order). `scale_floor` is not checked against the model: the model does not carry it, so a hand-edited value cannot be detected, and it changes every forecast (it is the flatness threshold of the standardization). A record of version 1 is refused, in one sentence that names the file and says the
run has to be trained again, by `predict`, by `export` and by `resume_interrupted=true` (the last before the data is
loaded). There is no converter and no reader: a run directory made before this change cannot be predicted from, exported
or resumed (its files are not touched, and `best_model.keras` still loads in Keras).

**Export**: `python -m train export model=<run dir or .keras> out=<path> format=keras|saved_model|tflite|onnx`
writes one new file (a directory for `saved_model`) and `<out>.task.json` beside it, and prints
only the path on stdout. Input contract: float32 `(batch, height, width, channels)` pixels scaled
to `[0, 1]` then standardized, `(x - mean) / std`, with the mean and std in `<out>.task.json`;
the output is raw logits, one per class name in that file for a classifier, `(batch, classes)`. **A forecasting run** exports raw windows `(batch, lookback, channels)` and returns the forecast in the series' own units, and refuses five of its 36 cells by name (`time_series_forecasting` above). **A segmentation run**
(D-081) exports the same four formats and the output is one score per class per pixel, `(batch, height, width, classes)`,
at the `input_size` of `<out>.task.json` (a version 2 record, the run's own, byte for byte): height and width are fixed
to it and may differ, only the batch is free, and the mask is the argmax over the last axis after the caller resizes the
scores to the picture (`python -m train predict` does that). Measured on CPU, ConvUNext `tiny` and AccUNet at 32x64 in
all four formats: 2e-6 at most against Keras on scores of scale 1, two calls of an artifact equal, `tflite` with built-in
operators only (no Flex); ConvUNext `tiny` with the real Oxford-IIIT Pet weights of the step-28 run on 16 real test
images at 128x128 and, the same weights in a copy built at 128x256, at 128x256: 5e-6 at most on scores of scale 7, every
argmax equal in all four formats. Serving on a GPU is again a different computation: Keras on the GPU against the CPU files, same weights and images, differs by up to 8.2e-3 on scores of scale 7.7 and 1 pixel in 9,400 has another argmax (agreement 0.999893); no format changes that. A `mixed_float16` run returns half-precision scores (Keras does), and a model that
computes in `bfloat16` is refused as `tflite` by name, as a classifier is. Other pixel ranges give numbers and
no error. A `tflite` file of a CliffordNet needs the Flex delegate (a WARNING says so). The check `export` makes of such a
file runs the Keras model on the CPU, as the interpreter does, so a GPU machine compares CPU with CPU (the tolerance is
unchanged: 1e-4 of the logits' scale, and a file that really differs is still refused). **Serving on a GPU is a different
computation from the CPU's**: it differs from the CPU's by about 1e-3 in the logits. Measured on the ImageNet gate's `cliffordnet/nano`, the same weights run on the GPU and on the CPU: up to 1.1e-3 on logits of scale 3.6 for 16 real validation images (5.8e-4 on scale 3.0 for 16 noise images), argmaxes equal on all 16 of each. No export format changes that.

**A trained AccUNet's `saved_model` against Keras depends on the device the comparison runs on** (plan 98b866fb step 1, `scratch_step1/accunet_export_regimes.json`; the iteration 3 gate had read 1.7% of the scale, 2.29 on 135, with the GPU visible). One `saved_model` of the real Pet weights of `train_api_review3_accunet_pet_b8` (scores up to 220.5), exported by a CPU process and again by a GPU process, run on 16 real test images at 128x128, against Keras in the same regime: on the CPU 9.2e-4 (4.2e-6 of the scale, argmax agreement 1.0, so inside the 1e-4 `RTOL`); on the GPU with TF32 on (the RTX 4090's default) 9.3 to 9.4 (4.2% to 4.3% of the scale, agreement 0.9957 to 0.9963); on the GPU with TF32 off 2.5e-3 (1.1e-5, agreement 1.0). Against Keras on the CPU as the reference, the TF32 file differs by 2.6 to 2.8 (1.2% to 1.3%). The export is not at fault, the device's arithmetic is: a comparison of an export with Keras belongs on the CPU, as the `tflite` check already does (D-042), and the exported file is the same bytes whichever process wrote it. ConvUNext's GPU gap is a hundredth of that (8.2e-3 on scores of scale 7.7, 1.1e-3 of the scale, step 31).

**EXIF orientation and a segmentation prediction** (plan 98b866fb step 1, `scratch_step1/exif_stability.json`, 16 real Pet test photographs, CPU, ConvUNext `tiny` and AccUNet `base`, 3 epochs each, plus the 20-epoch ConvUNext and a second AccUNet): a picture and the same decoded array stored rotated with EXIF orientation 6 (a PNG, lossless) decode to equal arrays and give equal logits and equal masks, for all four runs; a test pins it. A JPEG that was rotated and encoded again is a different array (mean difference 1.4 of 255) and the prediction moves with the array, not with the orientation: against the upright file, ConvUNext agrees on 99.8% of the pixels (99.9% for a re-encode with no rotation at all), AccUNet on 92.7% to 92.9% (93.1% to 94.3% with no rotation), so about 40 times as many pixels change for AccUNet. This is a property of those AccUNet weights under a second JPEG generation, not of the EXIF handling.

**Exit codes**: 0 success; 2 a request that cannot be honoured (bad key, value, path, data,
resume), found before the run directory is claimed whenever it can be seen without loading the data (the few that need
the loaded data, such as a `checkpoint_args.backup_every_steps` not below the steps of an epoch, come after the claim and
say so), and before TensorFlow is imported only for the checks the command line makes itself (an unknown key, a value out
of range, a missing path of `predict` or `export`: 0.04 s). Everything that needs the model name or the data comes after
the import, because naming a model imports its family's module and that imports TensorFlow (measured, one run each, no
GPU: a missing `data.imagenet_root` exits 2 after 3.2 s, a bad `metric_args.watch_metric` after 4.0 s, a resume of a missing
run after 3.9 s). Checking the model name without importing the family was judged not worth the registry change. 1 a run
that finished without reaching its
success threshold, or any unexpected failure (a traceback). `data.dataset=imagenet` and `imagenette` look at
the disk before the run directory is made, with path tests only: a missing `data.imagenet_root`, a root
without `train/` or `val/`, a devkit (`data.devkit_dir`, or `devkit/` under the root) without `meta.mat` or
`ILSVRC2012_validation_ground_truth.txt`, and no `imagenette/320px-v2` records under `$TFDS_DATA_DIR` (or
`~/tensorflow_datasets`) each exit 2 with a message that names the key and the path looked at.

**Recipes on ImageNet and Imagenette**: each of the six families has an entry for `imagenet` and for
`imagenette` in its recipe, so the notice `No recipe overlay for <family> on <dataset>` is printed only for a
dataset a family has no entry for (`mnist` for `vit`, `resnet` and both cliffordnet families). What a run on one of
these pairs resolves to, with every value `auto`:

| Family | Dataset | Epochs | Batch | Optimizer | Peak lr | Weight decay | Warmup epochs | Augment | Smoothing | Image size | Patience | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vit | imagenet | 50 | 128 | adamw | 3e-4 | 0.05 | 5 | none | 0 | 224 | 15 | legacy: `vit/train_vit.py:67,83-89,98,290` |
| vit | imagenette | 200 | 128 | adamw | 3e-4 | 0.05 | 20 | none | 0 | 224 | 15 | new: rules E, W below |
| resnet | imagenet | 100 | 256 | sgd | 0.1 | 1e-4 as `L2(5e-5)` | 5 | none, loader colour jitter | 0 | 224 | 15 | legacy: `resnet/train_resnet.py:61,74-80,87,345` |
| resnet | imagenette | 200 | 128 | sgd | 0.05 | 1e-4 as `L2(5e-5)` | 10 | none | 0 | 224 | 15 | new: rules E, B, L, W |
| convnext_v1, convnext_v2 | imagenet | 100 | 64 | adamw | 1e-3 | 1e-4 | 0 | none | 0 | 224 | 50 | legacy: `convnext/common.py:155,160-165,258-265` |
| convnext_v1, convnext_v2 | imagenette | 100 | 64 | adamw | 1e-3 | 1e-4 | 0 | none | 0 | 224 | 50 | new (the legacy trainer refuses Imagenette): rules E, M |
| cliffordnet | imagenet | 200 | 8 | adamw | 1e-3 | 0.1 | 0 | none | 0 | 224 | 30 | legacy except the batch (legacy 128): `cliffordnet/train_cliffordnet.py:241-246,611,652-664,681`, `common/args.py:345`; batch measured, see Memory fit |
| cliffordnet | imagenette | 200 | 8 | adamw | 1e-3 | 0.1 | 0 | none | 0 | 224 | 30 | legacy except the batch (legacy 128): `cliffordnet/train_cliffordnet.py:247-251,611,652-664,681`; batch measured, see Memory fit |
| cliffordnet_hier | imagenet | 200 | 16 | adamw | 1e-3 | 0.1 | 0 | none | 0 | 224 | 30 | legacy except the batch (legacy 128): same lines as `cliffordnet`; batch measured, see Memory fit |
| cliffordnet_hier | imagenette | 200 | 16 | adamw | 1e-3 | 0.1 | 0 | none | 0 | 224 | 30 | legacy except the batch (legacy 128): same lines as `cliffordnet`; batch measured, see Memory fit |

"Augment none" adds nothing to the loader's own random-resized-crop and flip, which every ImageNet and Imagenette
run has (`resnet` on ImageNet adds the loader's colour jitter through `imagenet_augment_color`, which the loader
refuses for Imagenette). "Smoothing" is the label smoothing of the loss, 0 everywhere: the task builds a plain sparse
cross-entropy, `vit` has no smoothing (`vit/train_vit.py:290`; its `label_smoothing` field, which only refused a nonzero value, is deleted with its flag), the `label_smoothing` field of `resnet/train_resnet.py`
(default 0.1, never read by its loss) is deleted with its flag, and the other two trainers default to 0. "Image size" is the dataset's own 224 (`transform_args.image_size`
can change it). Other values of the six recipes: ViT uses patch 16 at 224 with dropout 0.1 and attention dropout 0.1
(`vit/train_vit.py:79-80,426-428`); the sgd weight decay of `resnet` is an L2 penalty on the model, of strength `wd / 2`
because Keras' `L2(l)` has the gradient `2 * l * w` and He et al.'s SGD weight decay `wd` adds `wd * w` to it (before step 10e of plan 98b866fb
the penalty was `L2(wd)`, twice the paper's decay), and it reaches convolution and dense kernels only, where torchvision's reference
trainer decays every parameter by default (BatchNorm scale and shift and the dense bias, 0.09 percent of `resnet18`'s parameters, get none: Goyal et al. 2017, arXiv 1706.02677 section 4.2, apply no decay to the BatchNorm coefficients, and the "no bias decay" rule of Bag of Tricks, He et al. 2019, arXiv 1812.01187, decays only the weights of convolution and fully connected layers, which is this rule);
`resnet` has no gradient clip (the legacy trainer's global clip of 1.0 is deleted from both arms: neither He et al. 2015 nor torchvision's
reference trainer clips); adamw decays in the optimizer (the legacy `resnet` trainer under `--optimizer adamw` builds no penalty and hands the optimizer the typed `--weight-decay`, since step 13.2 of plan 98b866fb; before it built the `wd / 2` penalty as well and left the optimizer its own default 0.004); `convnext_v1` and `convnext_v2` use strides 4, drop path 0.1 and dropout 0 on both datasets; stochastic
depth is 0.3 for `cliffordnet` (0.4 for `cn_64_5`) and 0.1 for `cliffordnet_hier`. The cliffordnet batches are measured, not
legacy (Memory fit, below). The table is pinned, value by value, by
`tests/test_train/test_api/test_the_imagenet_and_imagenette_recipes.py`, which also reads the legacy trainers'
own defaults and compares.

**What the `resnet` decay and clip changes did to a run** (step 10e of plan 98b866fb; `resnet/resnet18`, Imagenette at 128 px,
`transform_args.image_size=128 epochs=30`, the recipe otherwise, RTX 4090, 30 epochs, a selection split of 946 images so one image is
0.11 points; before = the recipe with `L2(1e-4)` and a global clip of 1.0, after = `L2(5e-5)` and no clip). Best selection accuracy,
seeds 42 and 43: before 0.8150 and 0.8224, after 0.8478 and 0.8404 (+3.3 and +1.8 points); test accuracy at the best epoch 0.7997 and
0.8196 before, 0.8270 and 0.8316 after. The same tree with one old setting put back through `optim_args` shows where it came from: the old
clip alone (`optim_args.global_clipnorm=1.0`) costs 3.4 and 1.4 points against the new recipe, the doubled decay alone
(`optim_args.weight_decay=2e-4`) changes it by -1.3 and +0.3 points (not attributable), and both put back reproduce the old run (seed 42:
0.8150 against 0.8150). So at this length the doubled decay cost nothing measurable, and the clip cost about two points; a 30-epoch run
of a small model is not a statement about the 100-epoch ImageNet recipe.

**The `resnet` legacy trainer and the API on ImageNet** (step 11 of plan 98b866fb, no parity claim). `resnet/train_resnet.py` reads ImageNet
only: `TrainingConfig.imagenet_root` is checked by `require_imagenet_root` and `train_resnet_imagenet` loads the dataset `"imagenet"`, with no
`--dataset` flag, so only the API can train a `resnet` on Imagenette or CIFAR-10. Both arms apply the decay as `L2(wd / 2)` on the kernels
(`train.common.optimizer.sgd_weight_decay_regularizer`) and neither clips the gradient. On one subset (`resnet18`, 224 px, batch 256, seed 42,
12,800 training images, 5,000 held-out selection images, 50 steps, `warmup_epochs=0`, RTX 4090) the two paths agree on the parameter count
(11,699,112), on the sha256 of the images and labels of the first four batches, and on the untrained inference-mode loss (104.9541, 15.2 times ln 1000,
both with the BatchNorm moving statistics still at their start values); the running training loss of the 50 steps differs by at most 0.008
(a repeat of the legacy arm differs from the first by 0.009), and after the epoch the training loss is 7.6176 and 7.6150 (legacy, twice) against
7.6148 (API). The selection loss is 7.577 and 7.651 (legacy, twice) against 7.827 (API), selection accuracy 0.0026 and 0.0020 against 0.0026:
the API value is 0.213 above the mean of the two legacy runs (7.614) and 0.176 above the higher one, 2.4 times their spread of 0.074, so it
lies outside both, and the cause is not established (only the first four of the 50 training batches were hashed, so the equality of the data
covers those and the 5,000 selection images). The model is at chance, so the gap says nothing about which path trains better. One epoch of 50 steps cannot rank the paths.

**The four Imagenette pairs that have no legacy trainer** (`vit`, `resnet`, `convnext_v1`, `convnext_v2`) get values
from these rules, each a derivation from a legacy value or a cited recipe, not from a run: no run of these four pairs
has been made, so read them as starting points. Only `cliffordnet` has an Imagenette parity run (the nano variant at 128 px and batch 32, see Parity below).

| Rule | Value | Basis |
|---|---|---|
| E, epochs | 200 for `vit` and `resnet` at batch 128, 100 for `convnext` at batch 64 | The fit split is 8,523 images (9,469 train records less 946 held out), 66 steps an epoch at batch 128 and 133 at batch 64, so 200 and 100 epochs are 13,200 and 13,300 steps: the step count of the one legacy Imagenette recipe, `cliffordnet` at 200 epochs (`cliffordnet/train_cliffordnet.py:664`). The ImageNet epochs (50, 100) would be 3,300 and 6,600 steps |
| B, batch | `vit` 128 and `convnext` 64 unchanged; `resnet` 128 | `resnet`'s 256 is 33 steps an epoch on this split (`resnet/train_resnet.py:61`) |
| L, learning rate | `resnet` 0.05; the others unchanged | linear scaling of the SGD rate with the batch, 0.1 x 128 / 256 (Goyal et al. 2017, "Accurate, Large Minibatch SGD"); the adamw rates are the legacy ones at the legacy batch |
| W, warmup | `vit` 20 epochs, `resnet` 10, `convnext` 0 | the legacy share of the schedule: `vit` 5 of 50 epochs (`vit/train_vit.py:83,87`), `resnet` 5 of 100 (`resnet/train_resnet.py:74,78`), `convnext` none (`convnext/common.py:264`) |
| Weight decay, patience, dropout | the legacy values (`resnet`: `wd` 1e-4 applied as `L2(wd / 2)`, the paper's strength) | no cited recipe changes them with the size of the data |
| M, `convnext` model arguments | strides 4, drop path 0.1, dropout 0 | the legacy table's only 224 px row (`convnext/common.py:155,160-165`), the ConvNeXt-T values; without an entry the factory would apply drop path 0 |

**Memory fit** (measured, GPU 0): for the smallest variant of each of the six families on both datasets, for
`cliffordnet/lite` (the factory's default variant) and `cliffordnet/cn_64_5` (its largest), and for `resnet/resnet50`
(the legacy ImageNet trainer's default variant), 20 real train steps ran through `python -m train image_classification`
on an RTX 4090 (24,564 MiB; 717 MiB in use when idle), with XLA as Keras sets it (`jit_compile` auto; the logs show the
XLA cluster compiled), the recipe's own values except `batch_size`, and `epochs=1` on a `data.max_samples` that gives 20
steps. "Allocator peak" is the run's own `GPU peak on` log line (`tf.config.experimental.get_memory_info`, read right
after `fit`, so the validation pass is in it); "Device" is the highest `nvidia-smi` `memory.used` sampled every half
second. The device figure is always above the peak plus the idle use (the allocator holds a pool larger than its peak, and
the CUDA context and kernels sit outside it) and, once a model has a big temporary, the pool grows to about 22,600 MiB,
92% of the card, so Device is a ceiling and not a headroom measure. **The rule:** a batch fits when it completes 20 steps
with the allocator peak at or under 20,000 MiB, the device at or under 23,000 MiB, and no allocator garbage-collection or
out-of-memory line in the log (TensorFlow degrades silently otherwise: `Garbage collection: deallocate free memory
regions` means the allocator reached the card's limit and recovered, which is a run with no margin). The batch is halved
from the recipe's until one fits. An OOM was observed in each failed row below, not inferred: the process died on
`Out of memory while trying to allocate N bytes` or on an allocator `ran out of memory` line. On ImageNet the rows near 100
images per second are limited by the disk (the first read of those files, below); the rows that repeat a
`data.max_samples` read the same files from the page cache.

<!-- memory:begin -->
| Model | Dataset | Image | Batch | Allocator peak, MiB | Device, MiB | Steps | Seconds per step | Images per second | Result |
|---|---|---|---|---|---|---|---|---|---|
| `vit/vit_pico` | imagenet | 224 | 128 | 3,235 | 6,293 | 20 | 1.330 | 96.3 | fits |
| `vit/vit_pico` | imagenette | 224 | 128 | 3,230 | 6,277 | 20 | 0.045 | 2,843.7 | fits |
| `resnet/resnet18` | imagenet | 224 | 256 | 5,326 | 18,165 | 20 | 2.624 | 97.5 | fits |
| `resnet/resnet18` | imagenette | 224 | 128 | 2,630 | 9,330 | 20 | 0.061 | 2,092.5 | fits |
| `resnet/resnet50` | imagenet | 224 | 256 | 18,239 | 22,591 | 20 | 0.326 | 785.3 | completed, allocator garbage collection, no margin |
| `resnet/resnet50` | imagenet | 224 | 128 | 10,579 | 22,589 | 20 | 0.164 | 780.4 | fits |
| `resnet/resnet50` | imagenette | 224 | 128 | 10,577 | 22,589 | 20 | 0.166 | 768.9 | fits |
| `convnext_v1/tiny` | imagenet | 224 | 64 | 4,184 | 9,337 | 20 | 0.598 | 106.9 | fits |
| `convnext_v1/tiny` | imagenette | 224 | 64 | 4,162 | 9,336 | 20 | 0.056 | 1,141.6 | fits |
| `convnext_v2/atto` | imagenet | 224 | 64 | 1,238 | 5,269 | 20 | 0.034 | 1,899.2 | fits |
| `convnext_v2/atto` | imagenette | 224 | 64 | 1,232 | 5,270 | 20 | 0.030 | 2,125.0 | fits |
| `cliffordnet/nano` | imagenet | 224 | 128 | - | 17,657 | - | - | - | OOM, asks 147.6 GiB |
| `cliffordnet/nano` | imagenet | 224 | 64 | - | 17,591 | - | - | - | OOM, asks 55.8 GiB |
| `cliffordnet/nano` | imagenet | 224 | 32 | - | 9,367 | - | - | - | OOM, asks 32.2 GiB |
| `cliffordnet/nano` | imagenet | 224 | 16 | 18,391 | 22,575 | 20 | 0.260 | 61.5 | completed, allocator garbage collection, no margin |
| `cliffordnet/nano` | imagenet | 224 | 8 | 10,804 | 19,587 | 20 | 0.133 | 60.2 | fits |
| `cliffordnet/nano` | imagenette | 224 | 128 | - | 17,654 | - | - | - | OOM, asks 147.6 GiB |
| `cliffordnet/nano` | imagenette | 224 | 64 | - | 17,590 | - | - | - | OOM, asks 55.8 GiB |
| `cliffordnet/nano` | imagenette | 224 | 32 | - | 9,366 | - | - | - | OOM, asks 32.2 GiB |
| `cliffordnet/nano` | imagenette | 224 | 16 | 18,388 | 22,574 | 20 | 0.260 | 61.6 | completed, allocator garbage collection, no margin |
| `cliffordnet/nano` | imagenette | 224 | 8 | 10,869 | 19,586 | 20 | 0.134 | 59.6 | fits |
| `cliffordnet/nano` | imagenette | 128 | 64 | 18,758 | 22,577 | 20 | 0.373 | 171.5 | completed, allocator garbage collection, no margin |
| `cliffordnet/nano` | imagenette | 128 | 32 | 14,315 | 19,589 | 20 | 0.183 | 174.4 | fits |
| `cliffordnet/lite` | imagenet | 224 | 128 | - | 22,565 | - | - | - | OOM, asks 112.3 GiB |
| `cliffordnet/lite` | imagenet | 224 | 64 | - | 17,591 | - | - | - | OOM, asks 60.0 GiB |
| `cliffordnet/lite` | imagenet | 224 | 32 | - | 14,487 | - | - | - | OOM, asks 38.3 GiB |
| `cliffordnet/lite` | imagenet | 224 | 16 | - | 9,351 | - | - | - | OOM, asks 24.9 GiB |
| `cliffordnet/lite` | imagenet | 224 | 8 | 16,458 | 22,575 | 20 | 0.200 | 40.0 | fits |
| `cliffordnet/lite` | imagenette | 224 | 128 | - | 17,645 | - | - | - | OOM |
| `cliffordnet/lite` | imagenette | 224 | 64 | - | 22,559 | - | - | - | OOM, asks 60.0 GiB |
| `cliffordnet/lite` | imagenette | 224 | 32 | - | 14,487 | - | - | - | OOM, asks 38.3 GiB |
| `cliffordnet/lite` | imagenette | 224 | 16 | - | 7,815 | - | - | - | OOM, asks 24.9 GiB |
| `cliffordnet/lite` | imagenette | 224 | 8 | 16,456 | 22,575 | 20 | 0.201 | 39.8 | fits |
| `cliffordnet/lite` | imagenette | 128 | 64 | - | 9,351 | - | - | - | OOM, asks 29.9 GiB |
| `cliffordnet/lite` | imagenette | 128 | 32 | 19,897 | 22,577 | 20 | 0.278 | 114.9 | completed, allocator garbage collection, no margin |
| `cliffordnet/lite` | imagenette | 128 | 16 | 15,148 | 19,859 | 20 | 0.146 | 109.8 | fits |
| `cliffordnet/cn_64_5` | imagenette | 224 | 8 | - | 3,249 | - | - | - | OOM, asks 32.9 GiB |
| `cliffordnet/cn_64_5` | imagenette | 224 | 4 | - | 2,225 | - | - | - | OOM, asks 21.9 GiB |
| `cliffordnet/cn_64_5` | imagenette | 224 | 2 | 18,166 | 22,619 | 21 | 0.207 | 9.7 | fits |
| `cliffordnet/cn_64_5` | imagenette | 224 | 1 | 6,405 | 9,613 | 22 | 0.129 | 7.8 | fits |
| `cliffordnet_hier/hier_345` | imagenet | 224 | 128 | - | 19,695 | - | - | - | OOM, asks 70.8 GiB |
| `cliffordnet_hier/hier_345` | imagenet | 224 | 64 | - | 17,589 | - | - | - | OOM, asks 33.4 GiB |
| `cliffordnet_hier/hier_345` | imagenet | 224 | 32 | 18,647 | 22,589 | 20 | 0.282 | 113.6 | completed, allocator garbage collection, no margin |
| `cliffordnet_hier/hier_345` | imagenet | 224 | 16 | 11,582 | 21,653 | 20 | 0.140 | 114.7 | fits |
| `cliffordnet_hier/hier_345` | imagenette | 224 | 128 | - | 17,652 | - | - | - | OOM, asks 70.8 GiB |
| `cliffordnet_hier/hier_345` | imagenette | 224 | 64 | - | 17,591 | - | - | - | OOM, asks 26.8 GiB |
| `cliffordnet_hier/hier_345` | imagenette | 224 | 32 | 18,442 | 22,585 | 20 | 0.289 | 110.7 | completed, allocator garbage collection, no margin |
| `cliffordnet_hier/hier_345` | imagenette | 224 | 16 | 11,601 | 21,657 | 20 | 0.142 | 112.7 | fits |
<!-- memory:end -->

What the table settles, in the recipes:

* `vit/vit_pico`, `resnet/resnet18`, `convnext_v1/tiny` and `convnext_v2/atto` fit at their recipes' batches on both
  datasets (allocator peaks 1,232 to 5,326 MiB). Their recipes did not change.
* `cliffordnet/nano` at 224 px does not fit at batch 128, 64 or 32 (the 32 asks 32.2 GiB, the legacy docstring's 34.6 GB),
  completes at 16 with a garbage collection (no margin) and fits at **8** (allocator peak 10,869 MiB). `cliffordnet/lite`
  also needs **8** (peak 16,456 MiB). `cliffordnet_hier/hier_345` fits at **16** (peak 11,601 MiB; 32 completes with a
  garbage collection). So the `cliffordnet` recipe on ImageNet and Imagenette is batch 8, the `cliffordnet_hier` recipe
  is 16, and both were 128. The CIFAR recipes keep 128 (32 px).
* **The learning rate is not scaled with the batch.** `cliffordnet/train_cliffordnet.py` has no batch rule (its learning rate
  is 1e-3 at batch 128 in the paper recipe and at batch 32 in its own README), so none is invented, and no run has
  checked that 1e-3 suits a batch of 8 or 16. (`resnet` is different: its Imagenette lr 0.05 is the linear rule of
  Goyal et al. 2017, as the rules above say.)
* **The image size stays the data's own 224.** The legacy README asks for `--image-size 128` on the isotropic variants,
  but in this API the data decides the size before the recipe is read, so a recipe cannot set it. The legacy README's
  form `--image-size 128 --batch-size 32` is `transform_args.image_size=128 batch_size=32` here, and it was measured: `cliffordnet/nano` fits at 32 there
  (peak 14,315 MiB, 174 images per second against 60 at 224 px), and `cliffordnet/lite` fits at 16. A parity run against the
  legacy trainer sets both keys itself.
* **The evaluation batch is a recipe key too.** The training batches above leave the passes AFTER `fit` at the module constant
  256 images, and 256 images of 224 px asked 18 GB of the same card: the gate's `cliffordnet/nano` trained and then died in the
  figures' `predict`. `analysis_args.eval_batch_size` is now the batch of those passes; the `cliffordnet` and
  `cliffordnet_hier` ImageNet and Imagenette recipes set it to the batch they train at (8 and 16). Measured with that run's
  weights, 512 images, allocator peak: 924 MiB at batch 8, 1,468 at 16, 2,555 at 32, 5,043 at 64, 17,775 at 256; the real
  reporting pass of that run (20,000 test images, figures, analyzer, final evaluation, load check) at batch 8 peaked at
  4,681.5 MiB. `cliffordnet` on CIFAR at 224 px is not reachable: `transform_args.image_size` is refused for in-memory data.
  Folder and csv data at 224 px are NOT covered: the overlay is keyed to the dataset names, so they resolve to the base recipe's
  batch of 128 and to 256 here. The review ran `cliffordnet/nano` at 256 on 224 px files: the allocator logged an out-of-memory
  request of 12.27 GiB for the figures and the run recovered and finished; no death was reproduced, so the answer is a warning at
  resolution (both keys named), not a changed default.

Not covered by a recipe, and measured: `resnet/resnet50` on ImageNet completes at the recipe's batch 256 but with a garbage
collection and the device at 22,591 MiB (no margin), and a bounded whole run of it (`data.max_samples=2560`, one epoch)
also reported to exit 0 with the evaluation batch left at 256 (the figures' `predict` alone peaks at 4,487 MiB); batch 128 fits (peak 10,579 MiB), with `optim_args.lr=0.05` by the same
linear rule. `cliffordnet/cn_64_5` does not fit at 8 or 4 on Imagenette and fits at 2 (peak 18,166 MiB); it was not run on
ImageNet (the tensors are the same but for the 1,000-way head). A recipe is per family and `per_variant` applies to every
dataset, so changing the batch for these two would change their CIFAR recipes; they are left, and
`tests/test_train/test_api/test_the_recipe_batches_are_the_measured_memory_fit.py` names them. The larger variants of the other families (`vit_tiny`
and up, `resnet101`, `convnext` above `tiny`) were not measured. A run of 20 steps is not a run of 200 epochs: nothing here
says the margin holds for the whole run. The table is pinned to `tests/test_train/test_api/memory_fit_evidence.json` by the
test, row by row.

**ImageNet on a spinning disk.** The ImageNet train images are on a hard disk (a 4 TB WD40EFAX, `ROTA=1`) on the development
machine. In the runs above the first read of files cost **96.3, 97.5 and 106.9 images per second** (`vit/vit_pico`,
`resnet/resnet18`, `convnext_v1/tiny`; cold random reads of single files measured 44 to 115), against about 1,150 images
per second of CPU decode, so the disk is the limit, not the GPU. One epoch is 1,271,167 images; that is 3.7 to 3.3 hours,
and a run of E epochs takes (validation not counted):

<!-- imagenet-time:begin -->
| Epochs | Days at 96.3 images per second | Days at 106.9 images per second |
|---|---|---|
| 50 | 7.6 | 6.9 |
| 100 | 15.3 | 13.8 |
| 200 | 30.6 | 27.5 |
<!-- imagenet-time:end -->

The one longer measurement on the whole train set (step 12 of plan 98b866fb, `resnet/resnet18`, batch 256, 150 steps through the API path,
12 minutes, with another process holding 11 GB of the card) read **52 images per second over steps 1 to 50 and 79 and 76 over steps 50 to 100
and 100 to 150**, so a steady rate near 77 images per second puts one epoch at 4.6 hours of training, and about 5 hours with its
evaluation, figures and analysis; at 77 images per second the table above is 9.6 days for 50 epochs and 19.1 for 100.

**One full ImageNet epoch, measured** (step 12 of plan 98b866fb, D-061, D-064, D-066): `python -m train image_classification
model=resnet/resnet18 data.dataset=imagenet epochs=1 optim_args.warmup_epochs=0 checkpoint_args.backup_every_steps=500 seed=42`
(the corrected recipe: `L2(wd/2)` on the kernels, no clip), the whole pool of 1,271,167 fit images, 4,965 steps at batch 256. The run
was killed by a logout at step 4,223 and continued with `resume_interrupted=true` from the copy taken after step 4,000, so its last 965
steps came from a fresh pass of the data (some images seen twice, some not at all); the reporting ran in a second resume. After the
epoch: selection split (10,000 held-out train images) `val_accuracy` **0.2597**, top-5 0.5008, `val_loss` 4.047; official
validation set (50,000 images, the test split) accuracy **0.2418**, top-5 **0.4803**, loss 4.130. The learning rate was a cosine
decay from 0.1 to 0.001 over the 4,965 steps with no warmup (`run.log`: `CosineDecay`, `decay_steps` 4965, `alpha` 0.01, 0 warmup
steps). The 0.0099 in `training_log.csv`'s `lr` column and in `results_summary.json`'s `lr_first_epoch` and `lr_last_epoch` is its
value at step 4,000, where the resumed pass started (the optimizer's step count was restored, not reset): those fields record the
rate at the first step of the segment that wrote them, not a constant rate. One epoch is a check that the path learns, not a point
on the 100-epoch curve.
Both passes shared the host with CPU test jobs and ran slower than the probe: 4.7 s per step (55 images per second) over the first
pass's 4,223 steps, 5.6 s per step (46 images per second) over the resumed 965 steps and the selection evaluation, about 6.7 hours of
training for the epoch; the 77 images per second above was measured with no such jobs.

So the 50-epoch ImageNet recipe (`vit`) takes about a week, and the 100-epoch ones (`resnet`, `convnext`) about two. A
killed epoch is lost: the run goes back to the start of the epoch it was in (`resume_interrupted=true` trains it again), and
at these rates an epoch is hours; `checkpoint_args.backup_every_steps` bounds the loss to that many batches (next section). A shorter trial is
`data.max_samples` (it caps the train pool and, since plan 98b866fb 7.2, NOT the test split; it does NOT cap the selection set, which is
`data.holdout_per_class` images per class, 10 by default, so 10,000 images of the 1,000 classes whatever the cap: the bounded
gate with `data.max_samples=20000` (run BEFORE 7.2, when the cap also cut the test split) trained on 20,000 images, selected on 10,000 and tested on 20,000 official validation images
(`fit_state.json` `data_counts`: `n_train` 20000, `n_select` 10000, `n_test` 20000); set `data.holdout_per_class` to shrink it).

**Backing up inside an epoch.** `checkpoint_args.backup_every_steps` set to N (an integer from 2 to one below the steps of an epoch;
`null`, the default, backs up at each epoch end only, as before) also writes the resume backup (`checkpoints/backup/`, the
previous copy kept beside it) after every N-th batch of an epoch, so a kill loses at most N batches of training instead of the
epoch. A value that is not below the steps of an epoch of the loaded data is refused (exit 2, before the model is built). What a
resume from such a copy is, measured with real `SIGKILL`s of a child process
(`tests/test_train/test_api/test_backup_every_steps.py`, which counts every batch each process ran):

* the epoch that was killed, after S batches in all and `done` of them taken by the last copy, continues on the weights, Adam
  moments, loss scale and optimizer step count of that copy and runs only the R = S - `done` batches the copy had not taken;
  the epochs after it run in full. The review reproduction: killed after 15 of 18 batches with a copy every 5, the resume ran
  3 + 18 = 21 batches (36 without the key), and the run executed 54 updates, the counter's 54 (69 without the key). The updates a
  run executes are always `epochs * steps_per_epoch` plus the batches each kill lost since the copy it resumed from (13 in the
  test's five-kill chain, 3 for a kill in the epoch-end write, 5 for a torn newest copy);
* the examples are the epoch's first `done` batches of the killed process and the first R batches of a **fresh** pass of the data:
  the iterator of a resumed run is new and its position is not stored (measured: a fresh process with the same seed starts at
  epoch 0's order). So in that one epoch some examples are seen twice and some not at all. With two independent orders the
  expected share of the epoch's examples seen twice is R(S - R)/S^2, and the share seen not at all is the same; it is at most 25% (R = S/2). Worked
  example, the `vit/vit_pico` ImageNet run at batch 128: after 7,200 of the 9,930 steps of an epoch the last copy is after step 7,000, the resume
  runs the 2,930 steps that copy had not taken (not 9,930), and 20.8% of that epoch's examples are seen twice and the same share not at all. Measured exactly in the test setup (a script over the test child's
  data, not kept in the repository: 72 examples, 18 steps of 4): copy after 15 steps, 10 of its 72 examples twice and 10 not at all (13.9%, the formula's 0.139); after 10 steps, 21 and 21
  (29.2%, the formula's 24.7%). Every other epoch sees every example once. `skip` is not used to fast-forward: on the real Imagenette
  pipeline `skip(200).take(20)` decoded 7,055 images (220 batches) and took 2.2 s, `take(20)` of a fresh pass decoded 664 and
  took 0.28 s (a one-off measurement, not kept in the repository);
* the optimizer's step count is not touched: it continues from the copy's `epoch * steps_per_epoch + done` and the run ends on exactly
  `epochs * steps_per_epoch`, and every step uses the learning rate the unkilled run used at that step (compared step by step in each of
  the test's 12 other runs). Under `mixed_float16` the counter is the loss-scale optimizer's inner one, which lags the batches by the steps skipped
  for non-finite gradients; the resume keeps that lag;
* that epoch's row in `training_log.csv`, `training_history.json`, TensorBoard and the dashboard is written once, by the resumed
  process, and its **train** metrics (loss, accuracy, ...), its `lr` (the rate of the first remaining step) and its `epoch_times`
  entry cover the R remaining batches only, while its validation metrics are the whole validation set's. A `metric_args.watch_metric` that is a train metric
  (not `val_*`) is therefore judged on R batches in that epoch; the summary's `notes` say so. The CSV, the sidecar and the best checkpoint
  hold finished epochs only before it; a kill inside the **first** epoch resumes too, from its first copy;
* a kill between the weights and the metadata of a copy, or a torn newest copy, is answered by the previous copy (N batches older): the
  pair is checked against the optimizer's step count before `fit`, and a resume that cannot verify a pair refuses it by name. A kill in
  the shortened epoch, just after its process began, in the write of its epoch-end copy, in the epoch after it, and as that epoch began,
  each resumes to the same 54 steps (`RAN` in the test);
* a resumed run is a valid run of the same recipe, not the unkilled run's trajectory: in the test setup (`vit/vit_pico`, 3 epochs of 18
  batches, a copy every 5) the final weights differ from the unkilled run's by 1.1% to 2.0% of their norm (1.34% for the review
  reproduction; max absolute difference 0.0031), against 70.6% for another seed, and Adam's moments differ by more than their own norm
  (they follow the data order). An unkilled run with the key set is weight for weight the run without it. The run with the key unset
  is unchanged.

Time, measured (`tests/test_train/test_api/backup_resume_evidence.json`, recomputed by the test). A resume saves the batches it does
not run and pays for the copies it writes. CPU, `vit/vit_pico`, 225 steps an epoch, 2 epochs, a copy every 75 batches, killed after 160
batches of epoch 1 (last copy after batch 150), real `SIGKILL`, two repetitions: with the key the resume ran 75 batches (`fit` 33.0 and 32.1 s,
the whole process 46.3 and 44.6 s) against 225 without it (57.1 and 57.4 s; 69.4 and 69.8 s): 24.7 s of `fit` saved, 43% of it, and the run
executed 460 updates against 610 for the 450 asked. GPU (RTX 4090, Imagenette, batch 64, 133 steps an epoch, 3 epochs, the recorded PID killed
when the trace showed the batches below, 399 updates asked): `resnet/resnet50` with a copy every 40 (last copy after batch 120) resumed with
13 + 133 = 146 batches against 266 without the key, `fit` took 76.8 s against 85.6 s (8.8 s saved) and the run executed 404 updates against 524;
`resnet/resnet18` with a copy every 30 (last copy after batch 90) resumed with 176 batches against 266 and `fit` took 38.7 s against 37.7 s (no
saving). A resnet18 epoch on this pipeline takes 8 s and a copy 0.8 s: the key's five extra copies cost about 4 s of the 5.6 s the 90 batches
saved, and a new process spends 20 to 50 s compiling in its first epoch whatever it trains (the two first-epoch times differ by 2.5 s the other
way), so that difference is inside the noise. The key pays where an epoch is long against a copy (ImageNet: hours of batches against the
seconds of the table below), not on a toy.

A save writes `latest.weights.h5` (the weights and the optimizer state) after copying the previous pair to `*.bkp`, so the disk
holds two files of the size below. Measured (CPU, no data, the ImageNet recipe's shapes and optimizers, the disk of `results/`
which is an SSD; the cost is the write into the page cache, nothing is flushed):

<!-- backup-cost:begin -->
| Model | Optimizer | Parameters | File, MB | Seconds per save |
|---|---|---|---|---|
| `vit/vit_pico` | adamw | 3,048,232 | 35.2 | 0.38 |
| `resnet/resnet18` | sgd | 11,699,112 | 89.4 | 0.67 |
| `resnet/resnet50` | sgd | 25,610,152 | 195.7 | 1.54 |
| `convnext_v1/tiny` | adamw | 33,233,992 | 380.9 | 2.61 |
| `convnext_v2/atto` | adamw | 4,516,640 | 52.1 | 0.66 |
| `cliffordnet/nano` | adamw | 1,552,616 | 18.2 | 0.39 |
| `cliffordnet_hier/hier_345` | adamw | 883,576 | 10.6 | 0.35 |
<!-- backup-cost:end -->

At `backup_every_steps=500`, `vit/vit_pico` at batch 128 (9,930 steps an epoch) takes 19 copies of 0.38 s inside an epoch (7.2 s)
and a kill loses at most 10.0 to 11.1 minutes of training; `resnet/resnet18` at batch 256 (4,965 steps) takes 9 copies of 0.67 s
(6.0 s) and loses at most 20.0 to 22.2 minutes (the restart itself (process start, data load, compilation) comes on top and is not
included). The epoch-end copy is written in either case. The numbers come from the rates
above and the table; `test_backup_every_steps.py` recomputes them.

The exact commands for a full 50-epoch and a full 100-epoch run, from the repo root with the project virtual environment.
**They were NOT run**, because each takes a week or more of the only working GPU; the test
`tests/test_train/test_api/test_the_help_and_the_commands_state_the_imagenet_cost.py` only checks that each line is a valid request
(every key exists, every value has its type), not that it trains.

<!-- imagenet-commands:begin -->
```bash
# NOT RUN. 50 epochs of the vit recipe on the full ImageNet: about a week.
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train image_classification \
    out=results/imagenet_vit_pico_50ep model=vit/vit_pico data.dataset=imagenet epochs=50 \
    checkpoint_args.backup_every_steps=500

# NOT RUN. 100 epochs of the resnet recipe on the full ImageNet: about two weeks.
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train image_classification \
    out=results/imagenet_resnet18_100ep model=resnet/resnet18 data.dataset=imagenet epochs=100 \
    checkpoint_args.backup_every_steps=500

# NOT RUN. After a kill, the same line with resume_interrupted=true continues it (the killed epoch runs only the batches the last copy had not taken).
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train image_classification \
    out=results/imagenet_resnet18_100ep model=resnet/resnet18 data.dataset=imagenet epochs=100 \
    checkpoint_args.backup_every_steps=500 resume_interrupted=true
```
<!-- imagenet-commands:end -->

**Parity**: the API run and the legacy trainer were compared on `vit_pico` on CIFAR-10 (10 epochs, three
seeds each) and on `cliffordnet/nano` on Imagenette at batch 32 and 128 px (20 epochs, three seeds each); in both
the means differ by less than a tolerance measured from the legacy trainer's own run-to-run spread (on Imagenette the API
was lower on both statistics, by 0.0106 and 0.0088 against tolerances of 0.0137 and 0.0116, with a selection split of 946
images: "no difference larger than the machine's own repeat noise", not "equal"). **Parity does not cover the shipped
`cliffordnet` recipe**: batch 8 (16 for `cliffordnet_hier`) at 224 px with a learning rate of 1e-3 that is not scaled for
that batch (Memory fit, above), which nobody has trained to an accuracy, so whether it learns on Imagenette or ImageNet is
unknown. The other families' recipes are pinned by tests that build them, not by a run, **except the five below**. The legacy `train.<model>` scripts remain.

**Parity on CIFAR-10 for `convnext_v1`, `convnext_v2`, `cliffordnet` (`nano`) and `cliffordnet_hier` (`hier_345`), and on Oxford-IIIT Pet for
`convunext` (`tiny`), against the legacy trainers as corrected by plan 98b866fb** (step 10, D-050 to D-056, commit 197c16d5d, RTX 4090). The
protocol is `scratch_parity_classification/protocol.md` of that plan, sha256
`17593391dee83f658788dfbca84e1eb7d0c2b208a421a0fb325a13bb00d96ffc`, with addenda 1 to 5a (the Pet pair is addendum 5). Every arm of a family
ran 20 epochs (the first epoch of each CIFAR family projected 30 epochs above the protocol's 0.40 h per run; Pet ran 20 epochs and patience
20 on both arms, the parent plan's gate, so early stopping cannot fire), with the settings the protocol's checklist resolved through each
trainer's own parser: 45,000 images to fit and a 5,000-image selection split for CIFAR-10, 3,312 to fit and 368 selection for Pet. The
rule is the parent plan's D-005: legacy seeds 42, 43, 44 and 42 again, API seeds 42, 43, 44; the scalars are the best and the last epoch's
`val_accuracy` (`val_miou` for Pet) on the selection split; `T = max(range of the three legacy seeds, |L42 - L42r2|)` per scalar, written
before the family's first API run; PASS iff the means differ by at most `T` on both. Legacy seeds 42 and 43 are the fix steps' "after" runs,
because a CPU same-seed one-epoch fit at the parity commit gave exactly the weights it gave at each fix commit (102, 112, 190 and 208 arrays);
seed 44 and the repeat of 42 ran at the parity commit.

| family (20 epochs) | `T` best / last | mean legacy best / last | mean API best / last | API minus legacy | verdict |
|---|---|---|---|---|---|
| `convnext_v1/cifar10` | 0.0120 / 0.0162 | 0.8123 / 0.8107 | 0.8098 / 0.8093 | -0.0025 / -0.0015 | PASS |
| `convnext_v2/cifar10` | 0.0128 / 0.0116 | 0.8423 / 0.8417 | 0.8483 / 0.8477 | +0.0061 / +0.0059 | PASS |
| `cliffordnet/nano` | 0.0170 / 0.0156 | 0.8749 / 0.8734 | 0.8716 / 0.8714 | -0.0033 / -0.0020 | PASS |
| `cliffordnet_hier/hier_345` | 0.0074 / 0.0076 | 0.8815 / 0.8814 | 0.8819 / 0.8818 | +0.0004 / +0.0004 | PASS |
| `convunext/tiny`, Pet (`val_miou`) | 0.0102 / 0.0098 | 0.7072 / 0.7067 | 0.7041 / 0.7035 | -0.0031 / -0.0032 | PASS |

Read each PASS as "no difference larger than the legacy trainer's own spread over three seeds and one repeat", not as "equal": the CIFAR-10
selection split is 5,000 images (one image is 0.02 points), the Pet selection split 368, and `T` from three seeds is a noisy range (it was
0.007 to 0.017 here, never near the 0.03 at which the protocol refuses to judge). Test accuracy, reported and not in the rule, differs
between the arms' means by -0.0025, +0.0039, +0.0016 and -0.0024 (CIFAR-10, in the table's order) and by -0.0019 test mIoU on Pet. The
numbers are of the corrected legacy: the fixes of this plan moved the CIFAR-10 mean best `val_accuracy` of the legacy `convnext_v1` by about
-3.7 points (step 10b, the layer scale; both seeds) and of `cliffordnet_hier` by about -0.8 points (step 10d, the AutoAugment table; not
attributable by that plan's rule), so these means are not comparable with any run made before them. The earlier verdicts stand for the code they measured: `vit_pico` on CIFAR-10 is not on the path of any fix of this plan (it trains with
`cifar_light`, not the AutoAugment policy, and the API classification golden, a `vit_pico` fit on CPU, compares exactly with its baseline at
every fix commit, final weights included); `cliffordnet/nano` on Imagenette is not on the policy of the AutoAugment fix either (the
Imagenette recipe uses no augmentation preset) and the `StochasticDepth` fix leaves a `fit` exactly unchanged. The Pet pair replaces the
parent plan's PASS for `convunext`, which was a verdict on the old ConvNeXt V2 block. The segmentation families other than `convunext`,
the other classification families, and the shipped `cliffordnet` recipe on Imagenette have no parity run. The denoising families other than `bfconvunext/tiny` have none either, and that one is a MISS at 3 of 7 gated scalars (the `image_denoising` section above).
