# MatchChannels in the bfunet ConvUNeXt denoiser — findings

**Question this answers:** when you turn on `--zero-pad-channels` and/or
`--extra-zero-output-channels` in `train_convunext_denoiser.py`, do the weightless
`MatchChannels` layers actually *do* anything — and specifically, do they deliver the
training stability the design was meant to give?

**Short answer:**
- On **final quality** they are neutral: the trained denoiser is just as good as the
  default (learned 1×1-conv) version, with ~6% fewer parameters.
- On **training stability** — the actual reason the design exists — they help: in a
  production-representative setup (Laplacian pyramid + trainable Gabor, run to
  convergence), zero-pad channel matching gave **~36% lower run-to-run variance** and
  **~27% smoother loss curves**, at equal-or-slightly-better quality.
- The scary-looking behavior of `--extra-zero-output-channels` (the model output is
  *dead at initialization*) is **harmless**: it self-heals during training.

Everything below is reproducible with `src/train/bfunet/variance_probe.py`
(see "How to reproduce").

---

## 1. Background: what MatchChannels is and where it lives

A U-Net constantly changes the number of channels (feature maps) between stages. The
**default** way to change channel count is a learned `1×1` convolution (`Conv2D`,
`use_bias=False` here because the model is bias-free). `MatchChannels`
(`src/dl_techniques/layers/match_channels.py`) is a **weightless** alternative:

- growing channels → **zero-pad** (append all-zero channels),
- shrinking channels → **slice** (keep the first / last N channels),
- equal channels → passthrough (identity).

It has **no trainable weights**. Two trainer flags switch it on, both **OFF by default**:

| flag | factory param | what it replaces | call sites in `bfconvunext.py` |
|------|---------------|------------------|-------------------------------|
| `--zero-pad-channels` | `zero_pad_channels` | the per-level / bottleneck / decoder 1×1 channel-adjust convs | `:791` encoder, `:844` bottleneck, `:920` decoder skip-merge |
| `--extra-zero-output-channels` | `extra_zero_output_channels` | the learned final 1×1 output projection | `:949` output zero-pad, `:1036` final tail-slice |

**Default build:** with both flags off (the default), **zero `MatchChannels` layers are
even constructed** — every call site is behind an `if zero_pad_channels:` /
`if extra_zero_output_channels:` gate. So for a normal training run the layers cost
nothing and change nothing. The rest of this document is about what happens when you
**enable** them.

The original design intent (from the author): the default 1×1-conv projections were
causing gradient-flow issues and **high run-to-run variance** in metrics — the final
result was good, but the training *process* was noisy. The redesign aimed to (a) improve
gradient flow and (b) give the network **zero-initialized "scratch pad" channels** it can
learn to write into. That intent is what the empirical work below actually tests.

---

## 2. Mechanism: what the layers do when enabled

The network is **bias-free** (`use_bias=False` everywhere) and uses **ConvNeXt V2 style
blocks** with **zero-initialized residual scaling** (layerscale-γ / GRN start near zero,
so a fresh block's contribution starts near zero). Two facts drive everything:

1. A zero channel passed through a bias-free depthwise conv **stays zero** — no bias to
   lift it off zero. But the LayerNorm's cross-channel mean-subtraction and the block's
   pointwise (`1×1`) mixing **repopulate** a zero channel over training. So a
   zero-padded channel is not permanently dead — it is **scratch space the network
   fills in** as it trains (confirming the design's "scratch pad" idea).

2. `--extra-zero-output-channels` routes the **entire model output** through a freshly
   zero-padded channel subspace *and* deletes the learned final projection. At
   initialization this makes the whole output ≈ 0 (measured L2 ≈ 2e-4 vs ≈ 180 for the
   default). This looks broken, but it is just an extreme case of the zero-init residual
   pattern the architecture already uses everywhere — over training the output subspace
   **revives to full magnitude** and reaches normal quality.

**Per-flag verdict:**

| flag / sites | behavior at init | behavior after training | net |
|--------------|------------------|-------------------------|-----|
| `zero_pad_channels` (encoder / bottleneck / decoder) | healthy output; padded channels start at 0 | padded channels become active features; equal quality; **converges slower** | benign, quality-neutral, fewer params |
| `extra_zero_output_channels` (output path) | output ≈ 0 (dead) | self-heals to equal quality | not detrimental (self-healing zero-init residual) |

---

## 3. Empirical results

All experiments build the **real** `create_convunext_denoiser` model and (for the
variance runs) stream the **real** COCO+DIV2K patch pipeline. Three model variants are
compared throughout:

- **B (baseline)** — both flags OFF (the default learned 1×1-conv path)
- **V (variant)** — the tested flag ON (weightless `MatchChannels`)
- **C** — `zero_pad_channels` only (used to isolate which flag causes what)

### 3.1 Initialization + short-horizon probe (single seed)

- **Parameter count:** enabling the flags removes trainable parameters (the 1×1 convs
  they replace). Measured deltas ranged from ~6k (tiny toy model) to ~242k (small@128) —
  the `MatchChannels` layers themselves are confirmed **weightless (0 params)**.
- **Dead output at init:** `extra_zero_output_channels` (model A) output L2 ≈ 9.5e-5 vs
  baseline ≈ 180. Isolated to the output-path flag by the model-C control (zero-pad-only
  output was healthy, L2 ≈ 1298).
- **Self-healing:** trained 2000 steps at production config, model A's output revived
  (2e-4 → ~30) and reached loss parity with baseline (0.00406 vs 0.00429). **The
  "detrimental" first impression was wrong** — it was an init-time artifact.
- **Gotcha #1 — never audit at init:** a static / init-time / few-step probe *mispredicts
  the trained model* for this architecture. The output looks dead but trains fine.
- **Gotcha #2 — Adam fakes "learning":** in a gradient-starved subspace, Adam's
  per-parameter normalization makes weight norms grow (looks like learning) while an SGD
  control shows **zero real movement**. Do not use "weights are changing" as proof of
  learning without a gradient-magnitude or SGD control.

### 3.2 Variance test — first pass (small@128, no Laplacian, 1500 steps, 5 seeds)

This was the first real multi-seed test. It was **inconclusive by design flaw**: 1500
steps is not enough for the zero-pad variant to converge, so the comparison was
apples-to-oranges.

| metric | baseline | variant (zero-pad) | variant/baseline |
|--------|----------|--------------------|------------------|
| final PSNR (dB) | 29.42 | 23.33 | variant still climbing |
| across-seed MSE std | 8.7e-5 | 9.6e-4 | 11.0× (variant higher — but unconverged) |
| trajectory roughness | 0.606 | 0.496 | 0.82× (variant smoother) |
| gradient-norm CV | 5.43 | 3.48 | **0.64× (variant steadier)** |

Read: at a **cut-short** horizon the variant looks worse on variance, purely because it
had not finished converging (a descending run has naturally more spread). The gradient
flow was already **steadier** for zero-pad, and the signal was **stronger than on a tiny
toy** (0.64× vs 0.90×) — i.e. the effect grows with scale, so it must be measured at
scale and at convergence.

### 3.3 Variance test — production-representative (small@128, Laplacian + free Gabor, 10 epochs = 10k steps, 5 seeds)

This is the fair test: **both** conditions use the Laplacian pyramid and a **trainable
("free") Gabor stem**; only `zero_pad_channels` is toggled; each run trained to
convergence (10 epochs × 1000 steps).

| metric | baseline (1×1 conv) | variant (zero-pad) | variant/baseline |
|--------|---------------------|--------------------|------------------|
| final PSNR mean (dB) | 30.496 | **30.801** | +0.3 dB |
| final MSE mean | 0.000893 | 0.000832 | 0.93× (better) |
| **across-seed MSE std** | 3.2e-5 | **2.0e-5** | **0.64× (36% less variance)** |
| **across-seed MSE CV** | 3.55% | **2.44%** | **0.69×** |
| **trajectory roughness** | 0.368 | **0.267** | **0.73× (27% smoother)** |
| gradient-norm CV | 4.38 | 5.57 | 1.27× (higher — see caveat) |
| params | 4,282,560 | 4,040,640 | −241,920 |

Per-seed consistency is visibly tighter for the variant: variant PSNR spans 30.6–30.9
(0.3 dB) and is higher on **all 5 seeds**; baseline spans 30.3–30.7 (0.44 dB).

**This is the design goal, achieved:** lower run-to-run variance, smoother training,
equal-or-better quality, fewer parameters.

---

## 4. Interpretation

- **Final quality:** the flags are **neutral** — same converged quality as the default,
  ~6% fewer params. (At short horizons the zero-pad variant is *behind* because it
  **converges slower**; the zero-init scratch channels take time to fill. Give it the
  full budget and it reaches parity, even marginally ahead.)

- **Training stability (the real goal):** in the production-representative setup, zero-pad
  channel matching **reduced run-to-run variance by ~36%** and **smoothed the loss curve
  by ~27%**. The author's original motivation — 1×1 convs causing noisy training — is
  **borne out** once the test is run correctly (converged, real architecture, multiple
  seeds).

- **`extra_zero_output_channels`:** not detrimental. Its dead-at-init output is a
  self-healing zero-init residual. It also deletes the learned final projection at no
  measured quality cost.

- **Why earlier looks contradicted this:** two traps. (1) Measuring at init / a few steps
  mispredicts the trained model. (2) Measuring variance before the slower-converging
  variant has converged inflates its apparent variance. Both are resolved by running to
  convergence at realistic scale.

---

## 5. Caveats (read before over-claiming)

1. **Gradient-norm steadiness is not a robust win.** Zero-pad's gradient-norm CV was
   *lower* (0.64×, steadier) in the no-Laplacian run but *higher* (1.27×) in the
   Laplacian + free-Gabor run. "Steadier gradients" is architecture-dependent. The
   *outcome* the author cared about — low-variance, consistent final metrics — is what
   held up, not this intermediate signal.
2. **Sample size.** The headline variance numbers are from **5 seeds**. Treat ratios like
   0.64× as a solid *directional* estimate, not a p-value. The direction is consistent
   (variant better on all 5 seeds), which is what gives confidence.
3. **Scale.** Confirmed at `small`/128px on a fixed noise sigma (curriculum frozen to
   isolate architecture variance). A `base`/256px production-config confirmation run is
   the natural next step for the highest-confidence version.
4. **Task.** Denoising with additive Gaussian noise on COCO+DIV2K patches. Conclusions are
   about *this* denoiser family; do not blindly transfer to other tasks/architectures.

---

## 6. How to reproduce

Tool: `src/train/bfunet/variance_probe.py`. It builds the real model + real data
pipeline, runs baseline (flag OFF, 1×1-conv) vs variant (flag ON, weightless) across N
seeds at a fixed noise sigma, and reports the three metrics that matter: across-seed
variance, trajectory roughness, gradient-norm CV. Outputs a JSON, a markdown report, and
a per-seed trajectory plot into `results/<output-dir>/`.

```bash
# The production-representative run behind Section 3.3 (Laplacian + free Gabor, converged):
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.variance_probe \
    --variant small --patch-size 128 --batch-size 8 --patches-per-image 4 \
    --seeds 5 --epochs 10 --steps-per-epoch 1000 --log-every 50 \
    --max-train-files 2000 --sigma 0.1 --gpu 1 \
    --laplacian-pyramid --trainable-gabor \
    --output-dir results/variance_probe_zeropad_small_laplacian_freegabor

# Quick sanity check (tiny, ~1 min):
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.variance_probe \
    --variant tiny --patch-size 64 --batch-size 4 --seeds 2 --steps 30 --gpu 1

# Highest-confidence, production scale (heavier, fewer seeds if GPU-bound):
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m train.bfunet.variance_probe \
    --variant base --patch-size 256 --batch-size 4 --seeds 3 --epochs 10 --gpu 1 \
    --laplacian-pyramid --trainable-gabor
```

Useful flags: `--compare {zero_pad_channels,extra_zero_output_channels}` (which flag to
toggle for the variant), `--laplacian-pyramid` and `--trainable-gabor` (applied to BOTH
conditions so only the compared flag differs), `--epochs` / `--steps-per-epoch` (or
`--steps` directly), `--sigma` (fixed noise level).

**Reading the report:** ratios are `variant / baseline`; **< 1 means the weightless
variant is more stable / steadier** on that axis. The variance-reduction goal is supported
when `final_mse_std_ratio < 1` and `final_mse_cv_ratio < 1` — which is what the converged
production-representative run shows (0.64× and 0.69×).

---

## 7. One-paragraph summary

By default the `MatchChannels` flags do nothing (no layers are built). When enabled they
are weightless replacements for the learned 1×1 channel-projection convs. They do not
change final denoising quality (parity, ~6% fewer params) and the alarming "dead output"
of `--extra-zero-output-channels` self-heals during training. Their real value is in
**training stability**: in a production-representative setup run to convergence, zero-pad
channel matching cut run-to-run variance by ~36% and smoothed the loss curve by ~27% —
delivering exactly the low-variance training the design was created for. The one honest
caveat is that intermediate "gradient-norm steadiness" is architecture-dependent, and the
headline numbers come from 5 seeds at `small`/128px, so a `base`/256px run is the natural
confirmation.

---

## 8. Eval-interpretation caveats — ConvUNext base checkpoint (20260707) + app wiring

Findings from evaluating `results/convunext_denoiser_base_20260707_122133/best_model.keras`
with the two eval tools and wiring it into `src/applications/bias_free_denoiser/`
(plan_2026-07-10_77fb9b17). Numbers recorded verbatim; these are interpretation caveats,
not defects.

**PSNR-vs-noise sweep** (`eval_psnr_vs_noise.py`, 100 DIV2K-val patches):
σ255 → PSNR(dB): 5→40.68, 10→37.37, 15→35.51, 25→33.26, 35→31.79, 50→30.27, 65→29.12
(monotone, +6.4…+15.8 dB gain over the noisy input). Input-PSNR arithmetic checks out:
σ255=5 ⇒ σ_norm=0.0196 ⇒ `20·log10(1/0.0196)=34.16 dB` vs measured input 34.28; the small
positive bias is the expected effect of clipping the noisy input to `[-0.5,0.5]` (reduces
effective noise energy).

**The two tools' PSNRs are on DIFFERENT samples — do not read them as mutual
corroboration.** `eval_psnr_vs_noise` uses 100 random patches; `eval_per_pixel_uncertainty`
uses `--n-test 32`. Both use the IDENTICAL PSNR definition (`common._mean_psnr`,
`10·log10(1/MSE)`, `max_val=1.0` — the uncertainty tool imports `_mean_psnr`/`add_awgn`
from the psnr tool), so there is NO definitional discrepancy. But matched by noise level
they differ ~1.5 dB (33.26 dB @ σ255=25 vs the uncertainty tool's 31.73 dB @ σ_norm=0.10 ≈
σ255=25.5). That gap is sampling scope (32 vs 100 patches + crop strategy), not a
computation error — treat them as two independent measurements.

**Conformal coverage undercoverage is systematic, and it is NOT finite-sample noise.**
`eval_per_pixel_uncertainty.py` per-σ empirical coverage (target 0.90): 0.877 / 0.879 /
0.878 / 0.878 / 0.877 across σ_norm 0.05–0.25. With `--n-calib 32` images = 6.29M pooled
pixels the conformal quantile is near-exact, so the ~2.3-pt shortfall is NOT calibration-size
noise. It is the expected undercoverage of a single per-σ (Mondrian-on-σ-only) quantile
evaluated on HELD-OUT IMAGES with heterogeneous per-image residual scales: pixel
exchangeability holds only if the images are exchangeable. Image-conditional or
variance-scaled conformal would tighten it. (Prior precedent 0.887–0.894 was on a different
convunext checkpoint.)

**`variance_probe.py` is not applicable here** — it trains fresh models to compare
`zero_pad_channels` variants (§6); it has no `--checkpoint` flag and cannot evaluate a saved
checkpoint.

**App graph-relax loader is bit-identical (verified).** `DenoiserPrior.from_pretrained`
loads a ConvUNext checkpoint by loading the saved fixed-256 graph and relaxing its
`InputLayer` to `(None,None,C)` (`_relax_to_flexible_input`, mirrors this dir's
`eval_psnr_vs_noise.py::_to_flexible_input`). Verified on CPU: 133/133 weight arrays
byte-equal and forward-pass `max|Δ|=0.000e+00` at 256×256 vs the original graph; the
original rejects 320×448 while the relaxed model runs it. Depth-2 ⇒ inputs must be
divisible by 4 for exact pool/upsample reconstruction.
