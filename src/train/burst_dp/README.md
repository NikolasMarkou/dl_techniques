# BurstDP training

Train [`dl_techniques.models.burst_dp.BurstDP`](../../dl_techniques/models/burst_dp/)
on COCO 2017 with on-the-fly synthetic auxiliary-view generation.

## Quick start

```bash
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --preset burst_dp_small \
    --image-size 256 \
    --patch-size 16 \
    --n-max 5 --n-min 1 \
    --batch-size 4 \
    --epochs 40 \
    --coco-root /media/arxwn/data0_4tb/datasets/coco_2017 \
    --out-dir src/results/burst_dp/run01 \
    --gpu 0 \
    --mixed-precision
```

## Smoke run (a few images, fast)

```bash
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --preset burst_dp_pico \
    --image-size 128 \
    --patch-size 16 \
    --n-max 3 --n-min 1 \
    --batch-size 2 \
    --epochs 1 \
    --max-train-images 64 --max-val-images 16 \
    --coco-root /media/arxwn/data0_4tb/datasets/coco_2017 \
    --out-dir src/results/burst_dp/smoke \
    --gpu 0
```

## Datasets

`--dataset {coco,div2k,vggface2}` selects the training source. Default is
`coco` and is bit-for-bit back-compatible with the previous behavior.

| Dataset    | Mode         | Source layout                                                 | Heads trained         |
|------------|--------------|---------------------------------------------------------------|-----------------------|
| `coco`     | multi-task   | `<coco-root>/{train2017,val2017,annotations}/...`             | `recon` + `segmentation` |
| `div2k`    | fidelity-only| `<div2k-root>/{train,validation}/*.png` (800 / 100 images)    | `recon` (seg head zero-weighted) |
| `vggface2` | fidelity-only| `<vggface2-root>/{train_list.txt,test_list.txt,train/,test/}` | `recon` (seg head zero-weighted) |

**Fidelity-only mode**: DIV2K and VGG-Face2 lack segmentation labels.
The `BurstDP` model is hardcoded dual-head, so we keep the segmentation
head in the graph but zero its loss weight and drop its metrics. The
seg head still runs forward + backward but contributes nothing to the
total loss — a ~5-15% compute overhead in exchange for zero blast
radius on the model / config / saved-checkpoint surface. If you set
`--loss-seg` to a non-zero value with a fidelity-only dataset, the
trainer logs a warning and forces it to `0.0`.

Example invocations:

```bash
# DIV2K, single GPU, full 800 train images
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --dataset div2k \
    --div2k-root /media/arxwn/data0_4tb/datasets/div2k \
    --preset burst_dp_small --image-size 256 --patch-size 16 \
    --n-max 5 --n-min 1 --batch-size 4 --epochs 40 \
    --out-dir src/results/burst_dp/div2k_run01 --gpu 0 --mixed-precision
```

```bash
# VGG-Face2 — always cap train/val with --max-*-images
# (3.1M training JPGs otherwise)
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --dataset vggface2 \
    --vggface2-root /media/arxwn/data0_4tb/datasets/VGG-Face2/data \
    --preset burst_dp_small --image-size 256 --patch-size 16 \
    --n-max 5 --n-min 1 --batch-size 4 --epochs 10 \
    --max-train-images 100000 --max-val-images 5000 \
    --out-dir src/results/burst_dp/vggface2_run01 --gpu 0 --mixed-precision
```

## Visualizations

The trainer saves a `(num_samples x 6)` recon + segmentation comparison
grid (ref / aux[0] / recon pred / recon target / seg pred / seg target)
under `out_dir/viz/` on the configured cadence:

- `--viz-every-steps N` (default `500`) — save every N optimizer steps,
  files `viz/step_NNNNNNN.png`. Pass `0` to disable.
- `--viz-every-epochs M` (default `1`) — save every M epochs, files
  `viz/epoch_NNNN.png`. Pass `0` to disable.
- `--viz-num-samples K` (default `4`) — rows per grid, capped to the
  val batch size.

Both triggers can fire independently; setting both to `0` skips the
callback entirely.

## Outputs

- `burst_dp_best.keras` — best validation checkpoint
- `burst_dp_final.keras` — final epoch
- `tb/` — TensorBoard logs (per-head losses)
- `viz/` — periodic recon + segmentation comparison PNGs
- `training_curves/` — per-epoch loss / segmentation / other curve PNGs
- `history.csv` — epoch metrics
- `run_config.json` — args + model config + distortion spec for reproduction

> **`--out-dir` gotcha**: `--out-dir` is resolved relative to the *current
> working directory*. Always launch from the repo root so results land in
> `results/burst_dp/...`, not `src/results/burst_dp/...`.

---

# Reconstruction-Head Underfitting Investigation

This section documents the investigation into why the BurstDP **reconstruction
head underfits** — a multi-iteration debugging effort tracked under
`plans/plan_2026-05-20_b8f8df89/`. It is recorded here so the next person does
not re-walk the dead ends.

## Status: OPEN (paused 2026-05-22)

The catastrophic DIV2K failure is **fixed**. The COCO reconstruction head
**still underfits** (val PSNR ~22.3 dB, ~0.3 dB above the identity-noise
floor). The loss-reweight fix was tried and **failed its gate**. The next
lever — a pretrained encoder — has not yet been attempted.

## Iteration 1-2 — DIV2K catastrophic failure (FIXED)

**Symptom**: on DIV2K fidelity-only training, recon val PSNR collapsed to
~11.7 dB — far below a plain identity copy (~22 dB). The model could not even
overfit 4 images; gradient global-norm collapsed 17.1 → 0.006 in 50 steps.

**False start (iter-1)**: the failure was first attributed to a missing
residual path. Fixes applied (and *kept* — they are correct improvements):
- Residual reconstruction parameterization: the recon head emits a signed
  `delta`; the model computes `recon = clip(ref + delta, 0, 1)`. Commits
  `433c543c`, `95d22deb`. The DPTDecoder uses `output_activation="linear"`
  and a small-scale (`stddev=0.05`, **not** zero) `residual_proj` init — an
  exact-zero ControlNet-style init deadlocks because the decoder is trained
  from scratch. See `models/burst_dp/heads.py` and decisions D-001/D-002.
- LR warmup fix (`3eac0576`): `CosineDecay(initial_learning_rate=0.0, ...)` —
  the previous code set it equal to `warmup_target`, making the warmup a flat
  plateau with no ramp.

**True root cause (iter-2)**: the ViT encoder was built with
`normalization_position="post"` (the ViT class default). Post-norm has no
clean residual highway, so self-attention rank-collapse won and patch-token
diversity decayed exponentially (layer-by-layer: 0.22 → 0.0002 by layer 11) —
the encoder emitted spatially-constant features. **Fix**: construct the
encoder with `normalization_position="pre"` (commit `4815876b`).

**Outcome**: DIV2K val PSNR recovered 11.7 → 21.9 dB. The catastrophic failure
is resolved.

**Dead end**: LayerScale was added (`6b69273f`) as a deferred lever to tame
residual explosion — it had **no effect** (+0.04 dB) and is not the
constraint. It remains in the model but is inert; do not pursue it.

## Iteration 3 — COCO multitask underfitting (OPEN)

On the COCO dual-head run the segmentation head learns normally but the recon
head plateaus just above the identity floor.

### run01 — `results/burst_dp/coco_small_run01/` (baseline)

Full COCO (118k train), 10 epochs, `bs=8`, equal loss weights (1:1).

| metric | epoch 0 | epoch 9 |
|--------|---------|---------|
| recon_loss (Charbonnier) | 0.0674 | 0.0645 |
| val recon PSNR | 20.6 | 22.03 |
| val recon SSIM | 0.44 | 0.455 |
| val seg mIoU | 0.016 | 0.110 |

Recon is near-flat; segmentation learns fine.

### Initial diagnosis (findings F6/F7/F8)

- **F6** — with equal weights, `seg_loss` (~1.5) is ~12-22x larger than
  `recon_loss` (~0.067); the shared encoder gradient is segmentation-dominated.
- **F7** — the recon decoder is single-scale: `DPTDecoder` upsamples 16x from
  one 16×16 ViT layer with no multi-scale skips.
- **F8** — COCO corruption is mild; the model sits near the identity floor.

### Epistemic audit + correction

A formal audit (`analyses/analysis_2026-05-21_65e767ae/`) challenged the
diagnosis. It flagged two real issues but **itself made an arithmetic error**:
it used `0.0559` as the identity Charbonnier floor — a **noise-only** figure
that ignored the full corruption stack (brightness, contrast, blur,
motion-blur, occlusion). The proposed gate/criteria "fixes" (B1/B2) were built
on that wrong number and were reverted (decisions D-005 → D-006).

### Diagnostic ladder — `findings/diagnostic_ladder.py`

Run on the trained run01 checkpoint + a real COCO batch. Three measurements:

| Rung | Question | Result |
|------|----------|--------|
| A | blurry-mean collapse, or benign? | `std(recon) ≈ std(clean)`; `MAE(recon,clean) 0.0785 < MAE(ref,clean) 0.0805`. **NOT collapse** — recon = ref + a small helpful correction. |
| B | true encoder gradient ratio? | `‖grad_seg‖/‖grad_recon‖ = 17.6x` (measured at the shared encoder). recon **is** gradient-starved. |
| C | did the iter-2 pre-norm fix hold on COCO? | patch-token diversity 0.51-0.59 across all 12 layers. **Encoder healthy.** |

True identity floor (measured): Charbonnier **~0.080** (MAE ~0.0805), PSNR
~21.8 dB. run01 at 0.065 is therefore ~20% *below* the floor — the recon head
*is* denoising, just weakly.

### run02 — `results/burst_dp/coco_small_run02/` (loss reweight — FAILED)

Subset (8k train), 20 epochs, `bs=8`, `--loss-recon 18 --loss-seg 1` (18 ≈ the
measured 17.6x gradient ratio).

| metric | run01 best | run02 best | verdict |
|--------|-----------|-----------|---------|
| recon_loss | 0.0645 | 0.0629 | −2% only |
| val recon PSNR | 22.03 | 22.37 | +0.34 dB; **never reached the 22.4 dB gate** |
| val recon SSIM | 0.455 | 0.472 | marginal |
| val seg mIoU | 0.110 | **0.018** | **collapsed** |

**Gate FAILED.** Both pre-mortem signals fired: recon barely moved
(signal A — reweight insufficient) and segmentation collapsed (signal B —
reweight too aggressive).

### Conclusion

**Loss balance is not the binding constraint.** Equalizing the encoder
gradient (the measured 17.6x → 18:1 weight) produced only ~0.3 dB on recon
while destroying segmentation. F6 is real but not load-bearing — the recon
ceiling is **representational**, not a gradient-budget problem. The encoder is
healthy (diverse tokens) yet still cannot supply denoising-grade features.

### Open levers (next steps, untried)

1. **Pretrained ViT encoder** (recommended). The encoder is trained from
   scratch; a pretrained init would give it real low-level priors. This
   matches the lesson from depth estimation (`train_depth_estimation.py`
   already supports `--init-from` backbone transfer via
   `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`).
2. **Multi-scale decoder** (plan v3 step 4, low confidence). Add ViT
   `return_hidden_states` + a `MultiScaleReconstructionHead` tapping 4 layers.
   The audit rates this low-leverage: denoising is a *local* task a 16×16
   decoder can already express.
3. **Recon-only warmup**. Train a few epochs with `--loss-seg 0` (already
   supported via the fidelity-only path) so the encoder/decoder learn
   denoising without segmentation competition, then enable seg.

### Reproduction

```bash
# baseline (run01-equivalent), launched from repo root
MPLBACKEND=Agg .venv/bin/python -m train.burst_dp.train_burst_dp \
    --dataset coco --batch-size 8 --epochs 10 --gpu 0 \
    --out-dir results/burst_dp/coco_small_run01

# diagnostic ladder (fast, no training)
.venv/bin/python plans/plan_2026-05-20_b8f8df89/findings/diagnostic_ladder.py
```

Full trail: `plans/plan_2026-05-20_b8f8df89/` (decisions.md, findings/) and
`analyses/analysis_2026-05-21_65e767ae/summary.md` (the audit, including its
own corrected error).
