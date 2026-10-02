# LightGlue trainer

Trains `LightGlue` (`src/dl_techniques/models/vision/keypoints/lightglue/`), the keypoint
matcher of Lindenberger, Sarlin and Pollefeys (arXiv:2306.13643), on homography pairs made
from ordinary photographs, with a **frozen, already trained SuperPoint**
(`src/dl_techniques/models/vision/keypoints/superpoint/`) as the keypoint and descriptor
source. The model package README explains the model; this file covers the trainer and the
evaluation script.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.lightglue.train_lightglue --help
MPLBACKEND=Agg .venv/bin/python -m train.lightglue.eval_homography --help
```

This is **stage 1 only** (synthetic homography pairs). There is no MegaDepth stage 2 and no
pretrained weight is distributed; see section 9.

## 1. Files

| File | Contents |
|---|---|
| `train_lightglue.py` | `LightGlueTrainConfig`, the parse-first CLI, optimizer, callbacks, the run directory |
| `pipeline.py` | `LightGlueTrainingModel` (frozen SuperPoint + decode + labels + LightGlue + loss), `LightGlueCheckpoint`, `load_superpoint`, `freeze` |
| `data.py` | `make_pair_dataset`: COCO photographs to `{image0, image1, H0to1, image_size0, image_size1}` |
| `eval_homography.py` | corner-error AUC of LightGlue against a mutual-nearest-neighbour baseline |

The library pieces the trainer uses are `dl_techniques.utils.keypoint_extraction`
(batched SuperPoint decode), `dl_techniques.utils.keypoint_matching` (homography ground-truth
labels), `dl_techniques.losses.lightglue_loss.LightGlueLoss` and
`dl_techniques.metrics.keypoint_matching.KeypointMatchMetric`.

## 2. Pipeline design

One step of `fit` does the whole of the following inside the graph:

```
image0, image1 (grayscale, size fixed by the SuperPoint checkpoint), H0to1
   |  frozen SuperPoint, float32, training=False
   v
decode_superpoint: softmax, depth-to-space, NMS, threshold, top-k, padding mask,
                   descriptor sampling         -> keypoints, descriptors, masks (both images)
   |
   |  homography_matches(keypoints, H0to1, pos_threshold)  -> labels -2 / -1 / j
   v
LightGlue (static masked path, mask0 / mask1)  -> per-layer log assignments, confidences
   |
   +--> LightGlueLoss.compute  -> add_loss        (stock fit minimizes it)
   +--> KeypointMatchMetric + Mean trackers       (precision, recall, label statistics)
```

- **Frozen SuperPoint.** The checkpoint fixes the image size (its descriptor map is resized to
  the size it was built with) and the LightGlue input width. `--image-size` is optional and is
  only checked against the checkpoint. The SuperPoint must be float32; `load_superpoint`
  refuses a missing file, a size mismatch and a non-float32 model. There is no mixed-precision
  flag. `freeze` toggles `trainable` through `True` first, because Keras 3.8 treats
  `trainable = False` on a model that is already `False` (a deserialised frozen one) as a
  no-op that leaves its variables trainable.
- **Labels come from the homography.** Same rule as glue-factory's
  `gt_matches_from_homography`, checked by exact equality against a frozen output of that
  function. Keypoint `i` of image 0 is a positive when it is the mutual nearest neighbour of a
  keypoint of image 1 within `--pos-threshold` pixels after reprojection (the pair distance is
  the larger of the forward and the backward reprojection error). Dustbin is decided per
  image from ONE direction: image 0 is dustbin when no candidate has a forward error within
  the threshold, image 1 when no candidate has a backward error within it. A keypoint that
  warps outside the other image is also dustbin (an extension, not in glue-factory). Everything
  else that is not positive, a real keypoint with a one-sided candidate in range that did not
  win the pair contest, is ignored (label -2, no loss, no metric). Padded slots are ignored.
  The same threshold is used for positives and dustbin (glue-factory's 3 px). Rule and
  history: `decisions.md` D-011 (first reading, two-sided max) superseded by D-016 of the plan
  that added this trainer.
- **Loss under stock `fit`, no custom `train_step`.** `LightGlueLoss.compute` (the glue-factory
  form: balanced negative log-likelihood per layer, layer weights `gamma ** (L - 1 - i)`,
  plus a token-confidence binary cross-entropy against the detached final layer) is
  registered with `add_loss` inside the wrapper's `call`, and the trainer compiles without a
  loss and with `jit_compile=False`. `fit` receives the dataset dict as `x` and no `y`. The
  confidence term needs the real-keypoint masks under padding, so the loss is computed in
  `call`, where the masks are known (D-012, D-014).
- **Checkpointing (D-014).** The wrapper serialises (a test pins it) but is not what the run
  keeps: `create_callbacks` supplies early stopping, the CSV logger and
  `TerminateOnNaN`, its stock `ModelCheckpoint` is removed, and `LightGlueCheckpoint` saves
  the **LightGlue alone** whenever the monitored value improves. A base SuperPoint is tens of
  megabytes of weights that never change, and `eval_homography` only needs the matcher.
  `predict` mutates the wrapper's metric attributes (D-013); the trainer never calls it.
- **Data.** `make_pair_dataset` decodes a photograph to grayscale, centre-crops the longer
  side to the view aspect ratio and resizes it to a source frame `source_scale`
  (default 1.5, a `make_pair_dataset` argument, not a trainer flag) times the view size (a smaller photograph is resized up).
  Both images are then independent warped patches of that source, with NO pixel reading
  outside the source frame, so there is no black border (glue-factory's pair is border-free
  too, and a zero-filled wedge would be a learnable dustbin shortcut). Per view a homography
  is drawn (rotation 20 degrees, scale 0.8 to 1.2, perspective 0.001, translation 0.08,
  per view, so the relative homography spans about twice that) and the patch is shrunk about
  the source centre until its corners fit. Photometric jitter is applied to image 1 after the
  warp. Remaining differences to glue-factory (quad shrunk to fit instead of corners sampled
  with a convexity floor, no photometric `dark` mode, no right-only mode) are listed in the
  `train/lightglue/data.py` docstring. The seed of each pair is `[seed, stream index]`, so a
  run is reproducible and epochs differ. `H0to1` is the forward map between the two views: a
  point `p` of image 0 appears at `H0to1 @ p` in image 1. A corrupt file is skipped
  (`ignore_errors`) in training.
- **Optimizer.** AdamW with linear warmup then cosine decay, decoupled weight decay that
  excludes biases and norm parameters, global-norm clipping, built by
  `dl_techniques.optimization`.

## 3. Flags (`train_lightglue`)

| Flag | Default | Meaning |
|---|---|---|
| `--superpoint-checkpoint` | required | trained SuperPoint `.keras` file; fixes the image size and the LightGlue input width |
| `--coco-dir` | `/media/arxwn/data0_4tb/datasets/coco_2017/train2017` | folder of training photographs (jpg or png, not recursive) |
| `--val-images` | 256 | images held out for validation (fixed split from `--seed`); 0 monitors the training loss |
| `--image-size` | the checkpoint's | square size, checked against the checkpoint |
| `--max-keypoints` | 512 | padded keypoints per image |
| `--nms-radius` | 4 | detection NMS radius in pixels |
| `--detection-threshold` | 0.005 | minimum heatmap probability of a keypoint |
| `--border` | 4 | pixels at the image edge without keypoints |
| `--pos-threshold` | 3.0 | reprojection error of a positive pair in pixels; also the dustbin threshold |
| `--batch-size` | 16 | image pairs per step |
| `--epochs` | 10 | epochs requested |
| `--steps-per-epoch` | `train images // batch size` | steps per epoch |
| `--validation-steps` | the whole held-out set | validation batches |
| `--learning-rate` | 1e-4 | peak learning rate |
| `--weight-decay` | 0.01 | decoupled AdamW weight decay |
| `--warmup-steps` | 500 | linear warmup steps |
| `--clip-norm` | 1.0 | global gradient-norm clip; 0 disables |
| `--patience` | 5 | early-stopping patience in epochs |
| `--num-layers` | 9 | LightGlue layers (paper: 9) |
| `--descriptor-dim` | 256 | LightGlue internal width (paper: 256) |
| `--num-heads` | 4 | LightGlue attention heads (paper: 4) |
| `--seed` | 42 | seed of the weights, the pair sampling and the validation split |
| `--gpu` | none | GPU index; see section 8 |
| `--output-dir` | `<repo>/results` | base directory of the run |
| `--experiment-name` | `lightglue_<timestamp>` | run directory name |

There is no preset and no `--smoke` flag: a quick plumbing run passes explicit small values
for the size flags (section 7). Invalid values and a reused `--experiment-name` are refused
before anything is written.

## 4. Run directory

```
results/<experiment-name>/
  config.json               the parsed arguments
  run.log                   the run's log
  training_log.csv          per-epoch loss, positive_fraction, precision, recall, keypoints_per_image, val_*
  training_history.json     the same history as JSON
  best_model.keras          the LightGlue alone at the best monitored epoch
  lightglue.keras           the LightGlue at the end of training (reloaded and checked after saving)
  results_summary.json      strict JSON
```

`results_summary.json` keys: `model`, `seed`, `image_size`, `epochs_requested`, `epochs_run`,
`steps_per_epoch`, `validation_steps`, `batch_size`, `train_images`, `val_images`, `params`
(`lightglue_trainable`, `superpoint_frozen`), `devices`, `monitor`, `final` (last-epoch log
values), `best_checkpoint` (`monitor`, `value`, `epoch`, `path`), `label_statistics` (positive,
dustbin and ignored fraction of one training batch and keypoints per image),
`lightglue_model`, `lightglue_reload_verified`, `fit_seconds`.

`eval_homography` writes `config.json`, `run.log` and `results_summary.json` in its own run
directory. Nothing in an existing run directory is overwritten or deleted.

## 5. Producing a SuperPoint checkpoint

`src/train/superpoint/` has no README; the flags below are verbatim from
`python -m train.superpoint.<script> --help`. Every trainer there writes `final_model.keras`
into its run directory, and that file is what `--superpoint-checkpoint` takes.

The path that was run for the plumbing smoke test (section 7) is the shortest one, the joint
SuperPoint stage on synthetic shapes with no MagicPoint stage:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.superpoint.train_superpoint \
    --variant tiny --input-size 128 --data-mode synthetic --batch-size 4 --epochs 2 \
    --steps-per-epoch 30 --experiment-name smoke_superpoint
```

The SuperPoint trainer's own usage text describes a longer chain: `train_magicpoint` on
synthetic shapes, then `homographic_adaptation` to pseudo-label real photographs, then
`train_superpoint --data-mode pseudo --pseudo-labels-dir <the adaptation run>` with
`--magicpoint-checkpoint` for the weight handoff. Only the synthetic-shapes command above has
been run in this work; the chain is described in the scripts' docstrings and was not run
here. Note that `homographic_adaptation` defaults to `/media/arxwn/data0_4tb/datasets/COCO/train2017`
(capital letters) while this trainer defaults to `.../coco_2017/train2017`; pass
`--image-dirs` explicitly.

## 6. Full training: long GPU jobs, NOT RUN HERE

Nothing in this section was run. The three commands are for you to run, one after the other
(never in parallel on one GPU). Every size below is a **suggestion, unmeasured**: the only
settings measured in this work are the tiny ones of section 7. Assumptions: a 12 GB GPU,
square 240 pixel input (a multiple of 8, as SuperPoint requires), 512 keypoints. Attention
memory grows with the square of the keypoint count, so a 480 pixel input with 1024 keypoints
is a larger setting that needs a smaller `--batch-size`; lower it on an
out-of-memory error.

```bash
# 1. SuperPoint, joint stage on synthetic shapes (a quality-limited detector; the
#    pseudo-label chain of section 5 gives a stronger one).
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.superpoint.train_superpoint \
    --variant tiny --input-size 240 --data-mode synthetic --batch-size 16 \
    --epochs 50 --steps-per-epoch 500 --experiment-name superpoint_240

# 2. LightGlue on COCO train2017 homography pairs with that SuperPoint frozen.
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lightglue.train_lightglue \
    --superpoint-checkpoint results/superpoint_240/final_model.keras \
    --max-keypoints 512 --batch-size 8 --epochs 40 --steps-per-epoch 5000 \
    --validation-steps 100 --val-images 512 --learning-rate 1e-4 --warmup-steps 1000 \
    --experiment-name lightglue_240

# 3. Corner-error AUC on fixed-seed val2017 pairs, LightGlue against mutual nearest neighbours.
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lightglue.eval_homography \
    --lightglue results/lightglue_240/lightglue.keras \
    --superpoint-checkpoint results/superpoint_240/final_model.keras \
    --max-keypoints 512 --num-pairs 500 --experiment-name lightglue_240_eval
```

Do not pass `--gpu` together with `CUDA_VISIBLE_DEVICES` (section 8). With
`--steps-per-epoch` left at its default an epoch is `train images // batch size` steps (the
smoke run reported 118,271 training images with 16 held out). Step 3 refuses to start without `opencv-python` (it uses
`cv2.findHomography`).

## 7. Smoke run evidence (plumbing only, NOT a quality result)

Run 2026-10-02 on one GPU shared with other work, with tiny settings, to show that the pieces
run end to end. A SuperPoint (tiny, 128 pixels, 2 x 30 steps on synthetic shapes) was frozen
under a full-size LightGlue (9 layers, 256 wide, 4 heads; 11,851,601 trainable parameters;
the frozen SuperPoint has 12,559,393):

```bash
MPLBACKEND=Agg .venv/bin/python -m train.lightglue.train_lightglue \
    --superpoint-checkpoint results/smoke_superpoint_20261002/final_model.keras \
    --image-size 128 --max-keypoints 128 --batch-size 4 --epochs 2 --steps-per-epoch 30 \
    --validation-steps 4 --val-images 16 --warmup-steps 10 --experiment-name smoke_lightglue_20261002
```

The command above omits the device selection; the run itself passed `--gpu 0` (section 8).

- Training loss, epoch means: 8.18 then 4.53; validation loss 4.51 then 4.05. Finite, not
  constant, decreasing over 60 steps. That shows the loss is wired, and nothing more.
- Label statistics of one training batch: 21% of the keypoints are positives, 79% dustbin,
  none ignored, 128 keypoints per image (measured under the superseded two-sided dustbin rule of D-011; the one-sided rule of D-016 makes a few percent ignored, so these fractions were not re-measured).
- Last-epoch training precision 0.04, recall 0.005; validation 0.08 and 0.015. These are
  near zero after 60 steps, as expected.
- `lightglue.keras` reloaded in a fresh process as a `LightGlue` with the same parameter
  count (`lightglue_reload_verified: true`).
- `eval_homography` on 8 pairs ran and wrote a strict-JSON summary: for 6 of 8 pairs the
  matcher produced fewer than 4 matches, so the homography failed and was counted at
  `--max-error`. The mutual-nearest-neighbour baseline had no failures.

**These numbers describe a matcher that has seen 60 steps and a detector trained for 60
steps on synthetic shapes. They say nothing about matching quality**, and no comparison
between the matcher and the baseline can be drawn from them.

## 8. GPU note

`--gpu N` calls `train.common.setup_gpu`, which **overwrites** `CUDA_VISIBLE_DEVICES` with `N`.
Setting `CUDA_VISIBLE_DEVICES=1` and also passing `--gpu 0` therefore runs on physical GPU 0.
This was observed in the smoke runs: the SuperPoint and LightGlue runs executed on GPU 0
because `--gpu 0` was passed, while the evaluation run, which omitted `--gpu`, ran on GPU 1.
Select the device one way only: `CUDA_VISIBLE_DEVICES=1` alone, or `--gpu 1` alone.

## 9. Evaluation metrics (`eval_homography`)

Pairs are built from a folder of photographs (default COCO `val2017`; the first `--num-pairs`
sorted images, one pair each) with the trainer's data pipeline: a fixed seed, no photometric
jitter, the same wide homography ranges. The frozen SuperPoint supplies keypoints and
descriptors, matches become a homography with `cv2.findHomography` (RANSAC, threshold
`--ransac-threshold`), and the homography is scored against the sampled one. Definitions are
transcribed from glue-factory.

| Metric | Definition |
|---|---|
| corner error | mean over the four image corners `(0,0), (W,0), (W,H), (0,H)` of the distance between the corner projected by the estimated and by the ground-truth homography, in pixels |
| failed estimate | fewer than 4 matches, or a degenerate or non-finite homography; counted at `--max-error` (default 1000 px), never skipped. glue-factory records `inf`; a finite value keeps the mean finite. Median and every AUC below `--max-error` are the same as with `inf` |
| `auc@t` | area under the recall-versus-error curve up to `t` pixels, divided by `t`, for `t` in 1, 3, 5, 10 (errors sorted, recall `(i + 1) / n`, a `(0, 0)` point prepended, trapezoid rule) |
| `mean_error`, `median_error` | mean and median corner error over pairs |
| `failures`, `mean_matches` | failed estimates and matches per pair |
| `mean_precision`, `mean_recall` | predicted pairs equal to the ground-truth label over predicted pairs; positive labels that were predicted over all positive labels (labels at `--pos-threshold`; pairs with an empty denominator are left out of the mean) |
| `mean_stop_layer` | LightGlue only: mean 1-based layer whose assignment was used by `match()` (early exit) |

The two methods are `lightglue` (the adaptive eager `LightGlue.match`, with
`--depth-confidence`, `--width-confidence` and `--filter-threshold` overriding the
checkpoint's values) and `mnn` (mutual nearest neighbours by cosine similarity on the same
keypoints, with an optional `--mnn-ratio` test).

## 10. Limitations

- **Stage 1 only.** Synthetic homography pairs from COCO. There is no MegaDepth stage 2: the
  MegaDepth data available locally has no camera poses, so pose supervision is not possible.
  The model is therefore trained on planar, synthetic geometry and its behaviour on real
  wide-baseline pairs is untested.
- **The detector is fixed to the in-repo SuperPoint**, whose image size is baked into its
  descriptor map; changing the image size means a SuperPoint trained or built for that size.
  The model itself is detector-agnostic.
- **No pretrained weights** are distributed, for SuperPoint or for LightGlue, and nothing in
  this work was trained to a quality worth reporting. A converted torch checkpoint is
  possible by the transpose rule in the model README but was not done here.
- **Evaluation is synthetic.** The homography AUC on COCO pairs is not the paper's
  HPatches or MegaDepth benchmark and must not be compared to its numbers.
- `eval_homography` needs `opencv-python`.

## 11. Tests

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_train/test_lightglue \
    tests/test_train/test_config_fields_are_live.py -q
```
