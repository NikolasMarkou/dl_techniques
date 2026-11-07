# Training

One directory per runnable pipeline (see `AGENTS.md` for the patterns, `GUIDE.md` for the defect
guards, `REPO_MAP.md` at the repo root for which trainer trains which model). Most are run as
`python -m train.<model>.train_<script>`. One is run as `python -m train`.

## `python -m train` and `train.api`

A key=value command line over strict configs. It has two tasks, `image_classification`
(families `vit`, `resnet`, `convnext_v1`, `convnext_v2`, `cliffordnet`, `cliffordnet_hier`) and
`image_segmentation` (families `convunext`, `accunet`, `resnet_seg`, `convnext_seg`), and the same code is a Python function. Run it from the repo root with `.venv`, with
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
The same rule holds for both tasks and for the `data: <file>` of a `config=` file; the lowest layer is the `config=` file, then
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
(class names, input shape, mean and std: what a model needs to be used without its config),
`best_model.keras`, `final_model.keras`, the figures and analysis, and `checkpoints/`
(`backup/`, `fit_state.json`) and `run.lock` for resume. `run.lock` is a kernel `flock` held for the
life of the run and released when the process ends, a SIGKILL included, so two resumes cannot both run
and a reused pid blocks nobody; the pid in the file is a label for the refusal message.

**Resume**: `resume_interrupted=true` with the same `out` and the same request continues a killed
run. It is refused for a finished run (one with a `results_summary.json`), another request, other data, a live
process holding the run, or a torn backup; `devices` and `analysis_args.eval_batch_size` may differ. The training data
order replays epoch 0's and a half-done epoch is trained again (the key's help lists the rest).
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
searched recursively. `out` must not exist. **On ImageNet the `labels` are WordNet ids** such as `n01806143`, not
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
`input_shape` and `eval_transform`. A record of version 1 is refused, in one sentence that names the file and says the
run has to be trained again, by `predict`, by `export` and by `resume_interrupted=true` (the last before the data is
loaded). There is no converter and no reader: a run directory made before this change cannot be predicted from, exported
or resumed (its files are not touched, and `best_model.keras` still loads in Keras).

**Export**: `python -m train export model=<run dir or .keras> out=<path> format=keras|saved_model|tflite|onnx`
writes one new file (a directory for `saved_model`) and `<out>.task.json` beside it, and prints
only the path on stdout. Input contract: float32 `(batch, height, width, channels)` pixels scaled
to `[0, 1]` then standardized, `(x - mean) / std`, with the mean and std in `<out>.task.json`;
the output is raw logits, one per class name in that file for a classifier, `(batch, classes)`. **A segmentation run**
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
| vit | imagenet | 50 | 128 | adamw | 3e-4 | 0.05 | 5 | none | 0 | 224 | 15 | legacy: `vit/train_vit.py:67,83-89,99,103` |
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
cross-entropy, `vit` refuses a nonzero value (`vit/train_vit.py:125`), the `label_smoothing` field of `resnet/train_resnet.py`
(default 0.1, never read by its loss) is deleted with its flag, and the other two trainers default to 0. "Image size" is the dataset's own 224 (`transform_args.image_size`
can change it). Other values of the six recipes: ViT uses patch 16 at 224 with dropout 0.1 and attention dropout 0.1
(`vit/train_vit.py:79-80,451-453`); the sgd weight decay of `resnet` is an L2 penalty on the model, of strength `wd / 2`
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
7.6148 (API). The selection loss is 7.577 and 7.651 (legacy, twice) against 7.827 (API), selection accuracy 0.0026 and 0.0020 against 0.0026: a
gap two runs of the legacy arm cannot place inside or outside its own spread, at an accuracy that is chance. One epoch of 50 steps cannot rank the paths.

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
evaluation, figures and analysis. The one full ImageNet epoch was not run (it exceeded the 4.5 GPU hours this plan allowed for the family); at
77 images per second the table above is 9.6 days for 50 epochs and 19.1 for 100.

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
the other classification families, and the shipped `cliffordnet` recipe on Imagenette have no parity run.
