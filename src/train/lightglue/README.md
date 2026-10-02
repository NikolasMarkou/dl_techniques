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
pretrained weight is distributed; see section 10.

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
  The same threshold is used for positives and dustbin (glue-factory's 3 px). The rule, its
  history (a first reading took the dustbin from the two-sided max distance and mislabelled 3.3%
  of real keypoints that glue-factory ignores) and the guard tests are in the module docstring
  and the decision anchor of `dl_techniques/utils/keypoint_matching.py`; the frozen
  glue-factory output it is checked against is `tests/test_utils/data/glue_factory_homography_labels.npz`.
- **Loss under stock `fit`, no custom `train_step`.** `LightGlueLoss.compute` (the glue-factory
  form: balanced negative log-likelihood per layer, layer weights `gamma ** (L - 1 - i)`,
  plus a token-confidence binary cross-entropy against the detached final layer) is
  computed from the pre-sigmoid **logits** (`token_logits0/1`, D-018) in the stable form
  `max(z, 0) - z * t + log1p(exp(-|z|))`, so its gradient is `sigmoid(z) - t` at every logit.
  The loss functions (`lightglue_confidence_loss`, `LightGlueLoss.compute`) take logits; a
  probability passed by mistake is a silent error. The objective is registered with `add_loss` inside the wrapper's `call`, and the trainer compiles without a
  loss and with `jit_compile=False`. `fit` receives the dataset dict as `x` and no `y`. The
  confidence term needs the real-keypoint masks under padding, so the loss is computed in
  `call`, where the masks are known (D-012, D-014).
- **Checkpointing (D-014).** The wrapper serialises (a test pins it) but is not what the run
  keeps: `create_callbacks` supplies early stopping, the CSV logger and
  `TerminateOnNaN`, its stock `ModelCheckpoint` is removed, and `LightGlueCheckpoint` saves
  the **LightGlue alone** whenever the monitored value improves. A base SuperPoint is tens of
  megabytes of weights that never change, and `eval_homography` only needs the matcher.
  `predict` mutates the wrapper's metric attributes (D-013); the trainer never calls it.
- **Data.** `make_pair_dataset` decodes a photograph to grayscale, centre-crops the **longer**
  side to the view aspect ratio (the shorter side is kept whole) and resizes it to a source
  frame `source_scale` times the view size (default 1.5, a `make_pair_dataset` argument, not a
  trainer flag; a smaller photograph is resized up). Both images are then independent warped
  patches of that source (D-017): per view `sample_homography_tf` perturbs the view rectangle
  (rotation 20 degrees, scale 0.8 to 1.2, perspective 0.001, translation 0.08, per view, so the
  relative homography spans about twice that), the quad is centred in the source and shrunk
  about the centre by the largest factor that keeps all four corners one pixel inside the
  frame (capped at source resolution). No output pixel reads outside the source frame, so
  there is no black border: glue-factory's pair is border-free too, and a zero-filled wedge was
  a learnable dustbin shortcut in the first version of this pipeline. Photometric jitter is
  applied to image 1 after the warp. `H0to1` is the forward map between the views (a point `p`
  of image 0 appears at `H0to1 @ p` in image 1). Each view's randomness is stateless per pair,
  `[seed, i]` for view 1 and `[seed + 16, i]` for view 0, so a run is reproducible and epochs
  differ. A corrupt file is skipped (`ignore_errors`) in training. Remaining differences to
  glue-factory's sampler are in section 11.
- **Optimizer.** AdamW with linear warmup then cosine decay, decoupled weight decay that
  excludes biases, LayerNorm gamma/beta (name patterns) and the positional-encoding frequency
  `posenc.kernel` (by variable, because its name is plain `kernel` like every Dense kernel; the
  Dense and attention kernels still decay; see section 11), global-norm clipping, built by
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

Run 2026-10-02 with tiny settings, after the label, data and loss fixes of sections 2 and 11,
to show that the pieces run end to end. A SuperPoint (tiny, 128 pixels, 2 x 30 steps on
synthetic shapes) was frozen under a full-size LightGlue (9 layers, 256 wide, 4 heads;
11,851,601 trainable parameters; the frozen SuperPoint has 12,559,393). Device selection was
`CUDA_VISIBLE_DEVICES=1` alone (section 8), run summaries report GPU index 1.

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.lightglue.train_lightglue \
    --superpoint-checkpoint results/smoke_superpoint_20261002/final_model.keras \
    --image-size 128 --max-keypoints 128 --batch-size 4 --epochs 6 --steps-per-epoch 100 \
    --validation-steps 4 --val-images 16 --warmup-steps 10 --experiment-name smoke_lightglue_v3_20261002
```

- Training loss, epoch means over 6 x 100 steps: 6.50 falling to 4.02; validation loss 4.34
  falling to 3.95. Finite and decreasing. (The 2 x 30 step run of the same flags: loss 10.23 then
  5.34.) That shows the loss is wired, and nothing more.
- Label statistics of one training batch: positive 0.27, dustbin 0.72, ignored 0.012, about 127
  keypoints per image. The ignored class is non-zero, as expected under the one-sided rule.
- Last-epoch training precision 0.18, recall 0.011; validation 0.21 and 0.012.
- `lightglue.keras` reloaded in a fresh process as a `LightGlue` with the same parameter
  count (`lightglue_reload_verified: true`).
- `eval_homography` on 64 pairs (`--border` left at the trainer default 4):

  | method | mean matches | failed estimates | mean corner error | precision | recall |
  |---|---|---|---|---|---|
  | lightglue | 1.8 | 54 of 64 | 862 px | 0.107 | 0.013 |
  | mnn | 96.2 | 0 | 29.7 px | 0.039 | 0.097 |

  (failed estimates are counted at `--max-error` 1000.)

**LightGlue still loses to the mutual-nearest-neighbour baseline here**: after 600 steps on a
near-random tiny SuperPoint (the baseline's own precision is 0.04) it emits one or two matches
per pair. The mean stop layer is 2.0, but a diagnostic eval with adaptive depth and pruning off
(`--depth-confidence -1 --width-confidence -1`, stop layer 9.0) still gave about 3 matches per
pair and 52 of 64 failures, so early exit is not the cause. The pipeline does overfit one fixed
batch to precision and recall 1.0 in 200 steps (independent reviewer probe), so the wiring
works; whether a properly trained detector and a long run recover matches is **untested**.
These numbers prove plumbing only and support no comparison and no quality claim.

## 8. GPU note

`--gpu N` calls `train.common.setup_gpu`, which **overwrites** `CUDA_VISIBLE_DEVICES` with `N`.
`CUDA_VISIBLE_DEVICES=1` together with `--gpu 0` therefore runs on physical GPU 0 (this
happened in the first smoke run and was found by reading the run summaries). Select the device
one way only: `CUDA_VISIBLE_DEVICES=1` alone, or `--gpu 1` alone. Every command in this
README uses the first form.

## 9. Evaluation metrics (`eval_homography`)

Pairs are built from a folder of photographs (default COCO `val2017`; the first `--num-pairs`
sorted images, one pair each) with the trainer's data pipeline: a fixed seed, no photometric
jitter, the same per-view homography ranges as training (border-free views, `--border` as in training). The frozen SuperPoint supplies keypoints and
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
| `mean_precision`, `mean_precision_strict`, `mean_recall` | `mean_precision` is the training metric's definition: correct pairs over predicted pairs, where a pair whose keypoint is labelled -2 (ignored or padded) in either image is no prediction and leaves numerator and denominator. `mean_precision_strict` keeps those pairs as false positives (never higher). Recall is positive labels that were predicted over all positive labels (labels at `--pos-threshold`; pairs with an empty denominator are left out of the mean) |
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

## 11. Differences from glue-factory training

Reference: `gluefactory/configs/superpoint+lightglue_homography.yaml` and `train.py` defaults of
cvg/glue-factory `main`, read 2026-10-02. "Intentional" means chosen for this trainer's
circumstances; "unmeasured" means the effect on accuracy was not tested.

| Aspect | glue-factory | here | Intentional |
|---|---|---|---|
| Detector | `gluefactory_nonfree.superpoint` (official SuperPoint weights) | in-repo SuperPoint, frozen, you supply the checkpoint | yes (user decision; no official weights) |
| `nms_radius` | 3 | 4 (`--nms-radius`) | yes, the trainer's own default; unmeasured |
| `detection_threshold`, `force_num_keypoints` | 0.0 with `force_num_keypoints: True` (always 512 real keypoints, no padding) | 0.005, top-k to `--max-keypoints`, padded and masked | yes, padding is handled by masks; fewer real keypoints on weak images |
| Keypoint border | the extractor's default | `--border 4` | yes |
| Optimizer | Adam, no weight decay | AdamW, weight decay 0.01 on the Dense and attention kernels; biases, LayerNorm gamma/beta and the positional-encoding `kernel` are excluded (the encoding therefore matches glue-factory) | decay of the other kernels is a deliberate AdamW choice, unmeasured against no decay; guard `test_optimizer.py` |
| Gradient clipping | none (`clip_grad: None`) | global norm 1.0 (`--clip-norm`) | yes |
| LR schedule | constant 1e-4, then exponential decay from epoch 20 (10 epochs per factor 10) | peak 1e-4, 500 linear warmup steps, cosine decay to the end | yes |
| Epochs, batch | 40 epochs, batch 128, `train_size` 150000 pairs | `--epochs` 10, batch 16, an epoch is one pass over the images; the suggested full run in section 6 is 40 x 5000 steps of batch 8, which is 1.6 million pairs | yes (12 GB GPU) |
| Dataset | revisitop1m (Oxford-Paris distractors) | COCO train2017, `--val-images` held out | yes (local data) |
| Image size | patches of 640 x 480 (dataset default `patch_shape`) | square, fixed by the SuperPoint checkpoint (240 suggested) | yes |
| Views | image 0 is a warped patch too (`right_only` off), corners sampled by `sample_homography_corners` (difficulty 0.7, `max_angle` 45, convexity floor) | both views are warped patches, quad = perturbed rectangle shrunk to fit (D-017), per-view ranges: rotation 20 degrees, scale 0.8 to 1.2, perspective 0.001, translation 0.08 | yes, a smaller port; large perturbations cost zoom instead of being re-drawn; unmeasured |
| Augmentation | `photometric: lg` on both views | brightness, contrast, gamma and noise jitter on image 1 only, no flag | simpler on purpose; the `lg` augmentation is not ported, unmeasured |
| Labels | `gt_matches_from_homography`, thresholds 3 and 3 | same rule, equal on the frozen fixture; plus an extension that a keypoint warping outside the other image is dustbin | the extension is intentional |
| Loss | NLL (`gamma` 1, `nll_balancing` 0.5) plus confidence BCE with logits | same, from logits (D-018); padded keypoints masked | yes (masks) |
| Model | `flash: false`, `checkpointed: true` | no activation checkpointing, no flash path | yes |
| Stage 2 | MegaDepth | none (no poses available) | yes |

The model itself was checked against the official torch code, not only against a numpy
transcription; see "Pretrained weights and conversion" in the model README.

## 12. Tests

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_train/test_lightglue \
    tests/test_train/test_config_fields_are_live.py -q
```
