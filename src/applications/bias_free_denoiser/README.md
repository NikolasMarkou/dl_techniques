# Bias-Free Denoiser: Inverse Problems via the Implicit Prior

One unified interface that solves all of Kadkhodaie & Simoncelli's inverse problems
using the prior implicit in a bias-free denoiser. Every problem runs through the
**same** stochastic coarse-to-fine ascent loop; only the *measurement operator*
changes.

Reference: A. Kadkhodaie & E. P. Simoncelli, *"Stochastic Solutions for Linear
Inverse Problems using the Prior Implicit in a Denoiser"* (NeurIPS 2021). The
denoiser is a bias-free **ConvUNext** (bias-free ConvNeXt U-Net, BF-CNN-style:
strictly bias-free so it is input-scale equivariant / degree-1 homogeneous, trained
blind on additive Gaussian noise with an MSE objective). `DenoiserPrior.from_pretrained`
also still loads a bias-free CliffordUNet checkpoint — the architecture is auto-detected
from the checkpoint's `config.json`.

Shipped checkpoint: `results/convunext_denoiser_base_20260707_122133/` (ConvUNext base,
3.4M params, blind additive curriculum σ up to 0.5). Measured blind denoising PSNR on
DIV2K-validation (256px patches, 100 images): 40.7 dB @ σ255=5, 35.5 @ σ255=15, 33.3 @
σ255=25, 30.3 @ σ255=50, 29.1 @ σ255=65 — a single blind model across the whole range.

## Theory (one paragraph)

Miyasawa's / Tweedie's identity states that for an image corrupted by additive
Gaussian noise, the MMSE denoiser `D(y)` satisfies `D(y) = y + sigma^2 * grad_y log p(y)`.
The **residual** `f(y) = D(y) - y` is therefore proportional to the score of the
(noise-blurred) image prior. A bias-free denoiser gives this score for free at every
noise level, so we can walk uphill on `log p` without ever writing the prior down.
`UniversalInverseSolver` runs the paper's Algorithm 2: from a coarse noisy init it
takes annealed gradient steps `y <- y + h_t * d_t + gamma_t * z_t`, where the ascent
direction combines the prior score with a data-consistency term
`d_t = (I - MM^T) f(y) + M(m - M^T y)`. For the empty operator this degenerates
exactly to Algorithm 1 (unconstrained prior sampling, `d_t = f(y)`).

## The homogeneity bridge: this BLIND denoiser can emulate a NOISE-CONDITIONAL one

The checkpoint is **degree-1 positively homogeneous to float32 precision**:

```
D(a*y) = a*D(y)    measured rel. err 2.5e-05, FLAT across a in [0.12, 9.9] (an 80x range)
                   (a control that breaks bias-freeness fires at 8.3e-01 -- the probe works)
```

A flat error across an 80x range is float32 rounding noise, **not** a violation (a real violation
*grows* with `a` — a LayerNorm-block sibling checkpoint shows 81-98%). Root cause: `BiasFreeBatchNorm`
(no beta) + `use_bias=False` everywhere + `leaky_relu` (positively homogeneous).

> **Footgun:** the ConvUNext factory default activation is `gelu`, which would make exact homogeneity
> **mathematically impossible**. Any new denoiser checkpoint intended for this app must keep a
> positively-homogeneous activation, and homogeneity must be **probed numerically** — `verify_bias_free()`
> only inspects flags and will happily pass a non-homogeneous model.

**Why this matters.** Homogeneity makes the following an *exact identity*:

```
D_sigma(y) := sigma * D(y / sigma)        # a noise-conditional denoiser at level sigma
```

Every post-2021 inverse-problem solver — **DDRM** (2201.11793), **DDNM** (2212.00490), **DPS**
(2209.14687), **PiGDM**, **DCDP** — requires a *noise-conditional* network and is normally unreachable
from a single blind denoiser. **The literature contains no published bridge.** This checkpoint has one,
exactly, with **zero retraining** — which is what `ddnm.py` in this package exploits.

## Guardrails (hard constraints on any product claim)

- **No calibrated uncertainty. Ever.** The Jacobian is non-conservative (asymmetry 0.14, ~800x a
  box-blur baseline), so **no global energy / log-density exists**. Point estimates and samples only.
- **RED / PnP / MRED convergence guarantees do NOT transfer.** Beyond non-conservativeness, the
  denoiser is **not passive**: measured `||J||_2 = 1.22-1.36` on clean inputs. MRED (2202.04961)
  requires passivity, so it does not rescue them either.
- **The six inverse problems have NO task-specific SOTA baseline.** Only plain AWGN denoising is
  benchmarked (`src/train/bfunet/FINDINGS.md` §8.1). Every "it works" here is a **capability** claim,
  not a **competitiveness** claim.

## Domain requirement: `[-0.5, +0.5]`

The checkpoint was trained on pixels in **`[-0.5, +0.5]`** (center `c0 = 0.0`). The
`residual = score` identity is only valid in that domain, so every signal fed to the
prior/solver must live there. Use the helpers:

- `DenoiserPrior.ingest(x)` — normalize `uint8 [0,255]` / float `[0,1]` to `[-0.5, +0.5]`.
- `DenoiserPrior.denorm(x)` — map `[-0.5, +0.5]` back to `[0, 1]` for display/export.

## Public API

```python
from applications.bias_free_denoiser import (
    DenoiserPrior,             # frozen denoiser wrapped as an implicit prior
    UniversalInverseSolver,    # the unified Algorithm-1/2 loop
    MeasurementOperator,       # abstract base for the six problems
    NullOperator,              # empty M -> prior sampling (Algorithm 1)
    InpaintingOperator,        # centered missing block
    RandomPixelsOperator,      # random missing pixels
    SuperResolutionOperator,   # kxk block-average downsample
    SpectralDeblurOperator,    # unitary-DFT low-pass
    CompressiveSensingOperator # structured Rademacher-DCT-subsample
)
```

- **`DenoiserPrior`** — loads the frozen denoiser (registrar-first, `compile=False`).
  The default `from_pretrained(...)` returns a fully-convolutional graph at
  `(None, None, 3)` so it runs any sufficiently-divisible `H, W` in a single pass.
  Architecture is auto-detected: a ConvUNext checkpoint loads the saved graph and relaxes
  its size-locked input in place (bit-identical weights); a CliffordUNet checkpoint rebuilds
  via its factory + weight-transfer. `resolution="fixed256"` loads the saved 256x256 graph
  for either. Exposes `residual(y) = D(y) - y` (the score estimate) plus
  `ingest` / `denorm` / `tile`.
- **`UniversalInverseSolver`** — `solve(operator, measurements=..., shape=..., seed=...)`
  runs the annealed ascent and returns `(best_y, info)` where `info` holds
  `sigma_values`, `iterations`, and (when measurements are given) `constraint_errors`.
- **`MeasurementOperator`** — the single load-bearing abstraction. Each subclass
  implements `measure` / `adjoint` / `project` / `init_mean` with orthonormal columns
  (`M^T M ~= I`) and **no dense `N x N` matrix** (everything is O(N) or O(N log N)).

### Prior sampling (Algorithm 1)

```python
prior = DenoiserPrior.from_pretrained("results/convunext_denoiser_base_20260707_122133/best_model.keras")
solver = UniversalInverseSolver(prior, max_iterations=500)
sample, info = solver.solve(NullOperator(), measurements=None, shape=(1, 256, 256, 3), seed=0)
```

### An inverse problem (inpainting)

```python
target = DenoiserPrior.ingest(clean_uint8_image)[None, ...]   # [1, H, W, 3] in [-0.5, +0.5]
op = InpaintingOperator(target.shape[1:], block_size=64)
measurements = op.measure(target)
recon, info = solver.solve(op, measurements=measurements, seed=0)
restored = DenoiserPrior.denorm(recon[0])                     # back to [0, 1]
```

Swap `InpaintingOperator` for `RandomPixelsOperator`, `SuperResolutionOperator`,
`SpectralDeblurOperator`, or `CompressiveSensingOperator` — the solve call is
identical.

## Supported problems

| id | Operator | Problem |
|----|----------|---------|
| `denoise` | single forward pass `D(y)` | Blind Miyasawa/Tweedie MMSE denoise: no measurement operator, no iteration (the natural least-squares estimate at any noise level) |
| `prior` | `NullOperator` | Unconstrained sampling from the implicit prior (Algorithm 1) |
| `inpaint` | `InpaintingOperator` | Fill a centered missing block |
| `random_pixels` | `RandomPixelsOperator` | Fill randomly missing pixels |
| `super_resolution` | `SuperResolutionOperator` | Upsample a `kxk` block-averaged image |
| `deblur` | `SpectralDeblurOperator` | Recover high frequencies from a DFT low-pass |
| `compressive_sensing` | `CompressiveSensingOperator` | Reconstruct from structured random measurements |

## CLI

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python \
    -m applications.bias_free_denoiser.main --problem all
```

Runs the selected problem(s) on a synthetic in-domain target (or `--image PATH` for a
real image, resized to `--size`) and writes a grid PNG — target, measured/degraded
view, reconstruction, and the `sigma_t` convergence curve per problem — to
repo-root `results/`. It is fully headless (`MPLBACKEND=Agg`). `--problem` accepts any
of the seven ids (`denoise`, `prior`, `inpaint`, `random_pixels`, `super_resolution`,
`deblur`, `compressive_sensing`) or `all`.

**General flags:** `--checkpoint`, `--iterations` (default **500**), `--sigma0`,
`--seed`, `--size`, and the per-problem knobs `--block`, `--keep-ratio`, `--sr-factor`,
`--keep-fraction`, `--measurement-ratio`.

**Solver-regime flags** (the annealed-ascent schedule from Algorithm 2). The defaults
below are the paper-quality regime — an A/B (measured on the previous CliffordUNet
checkpoint; pending re-measurement on the ConvUNext default) showed **+13.4 dB mean PSNR**
on the inverse tasks versus the old capped/short-budget defaults, because the old cap left
the ascent stuck at `sigma_t ~= 0.25-0.33`, never reaching `sigma_l`:

| flag | default | meaning |
|------|---------|---------|
| `--h-max` | `none` | Step-size cap. `none` = uncapped paper schedule (best measured). Pass a float (e.g. `0.1`) to re-impose a cap for extra stability. |
| `--sigma-l` | `0.01` | Effective-noise stop threshold (paper-exact). The loop runs until `sigma_t` falls below this. |
| `--h0` | `0.01` | Step-size schedule parameter `h_t = h0*t/(1+h0*(t-1))` (paper-exact). |
| `--patience` | `0` | No-improvement iterations before early stop; `0` = disabled (a full-budget run never early-stops). |
| `--beta` | `0.01` | Injected-noise fraction (paper-exact); `1.0` = no injected noise. |

**Optional quality levers** (all OFF by default — the default path is byte-identical to
without them):

| flag | default | meaning |
|------|---------|---------|
| `--final-projection` | off | Impose exact hard data consistency once at solve return (`measure(recon) == measurements`). Helps mask-structured tasks (inpaint `+0.29 dB`, CS `+0.22 dB`); ~no effect on SR/deblur; no-op for `prior`. Cheapest lever (one post-loop op). |
| `--num-samples N` | `1` | "Ours-avg": average `N` independent solve trajectories (different seeds). Measured **+1.4 to +3.1 dB** across tasks at `N=4`, at `N`x runtime. `N=1` is a single solve. Note: this is **Monte-Carlo variance reduction on an under-converged chain**, not a prior improvement — it improves PSNR *and* perceptual quality together (measured), so it is not a metric artifact, but it is also not evidence of a better prior. |
| `null_space_noise` | off | **Confine the injected noise `z_t` to the null space of `M`.** Constructor-only (not yet on the CLI). **STRONGLY task-dependent — read the table below before enabling.** |

### `null_space_noise`: ship it PER-OPERATOR-CLASS, never globally

Measured on the frozen `convunext_denoiser_base_20260710_220452` checkpoint (500 iters, fixed
budget, 4 DIV2K-val images):

| task | operator class | baseline | `null_space_noise` | delta |
|------|----------------|---------:|-------------------:|------:|
| **super_resolution x4** | transform | 27.56 | **30.17** | **+2.61 dB** |
| **deblur** | transform | 24.34 | **26.63** | **+2.29 dB** |
| compressive_sensing | transform | 32.12 | 31.99 | −0.13 dB |
| inpaint | **mask** | 24.45 | 22.95 | **−1.50 dB** |
| random_pixels | **mask** | 28.99 | 14.78 | **−14.21 dB** |

**Why the sign flips** (mechanism, not curve-fitting): projecting `z_t` into `null(M)` changes the
*spatial statistics* of the noise the denoiser sees. For **mask** operators, `diag(I − MMᵀ)` is
bimodal — the noise becomes spatially **patchy** (coefficient of variation **3.88** inpaint,
**0.66** random_pixels), which is out-of-distribution for a net trained on spatially-uniform
additive Gaussian noise, so it *breaks* the denoiser. For **transform** operators the diagonal is
near-constant (CV ≈ **0.058**), the noise stays uniform and in-distribution, and confining it to
the null space is a pure win. Harm scales with null-space coverage (random_pixels 70% ≫ inpaint 6.25%).

> **Standing falsifiable prediction:** Bayer demosaicing is a *mask* operator with `p = 2/3`, giving
> CV `sqrt(p(1−p))/p` = **0.707** — adjacent to random_pixels. The rule therefore predicts
> `null_space_noise` **loses ~10–14 dB on demosaicing**. One solver run falsifies the rule that
> currently justifies shipping this flag. It has not been run.

### What does NOT work (measured, so you don't spend a month on it)

**Retuning the annealing schedule is worth +0.00 dB.** It is tempting to read the schedule as buggy:
`sigma_t` is computed from the *combined* `d_t = prior_term + data_term`, so range-space error looks
like it leaks noise into the null space via the isotropic injection. That reading was tested and is
**wrong**: `sigma_t² = sigma_prior² + sigma_data²` to machine precision (rel. err 1.5e-8) — the
orthogonal subspaces add in **quadrature**, so `sigma_t` is the *exact total error scale*, which is
precisely what a blind denoiser's implicit noise level must match. The isotropic injection is the
annealed-Langevin **thermostat** and is load-bearing (a `beta` sweep shows removing it costs PSNR
monotonically, paper setting optimal). Three independent interventions on all three of `sigma_t`'s
roles yield ≤ ±0.2 dB. See `solver.py`'s module docstring.

**The real structural upgrade is to replace the solver family** — see below.

The `denoise` task takes a single forward pass through the denoiser (no operator, no
solver). Its `--noise-sigma` flag (default `0.1`) synthesizes in-domain Gaussian test
noise that is added to the target — and lightly clipped back to `[-0.5, +0.5]` — before
the single denoise pass; `--noise-sigma 0` denoises the input as-is. This std is only a
test-harness knob: it is NEVER fed to the model, because the denoiser is bias-free and
noise-level-blind.

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python \
    -m applications.bias_free_denoiser.main --problem denoise --noise-sigma 0.1
```

## Convergence / iteration budget

Reconstruction quality tracks the iteration budget: the ascent must run long enough for
the effective noise `sigma_t` to anneal down to `sigma_l` (`0.01`). The default regime
(`--iterations 500`, uncapped `--h-max none`) reaches that convergence floor on the inverse
tasks; a much shorter budget (e.g. `--iterations 40-200`) stops the ascent well above
`sigma_l` and produces a coarse, unconverged result. The paper reports 300-650 iterations
to convergence, so `500` is a sensible default — raise it for the hardest low-measurement
regimes, or lower it for a fast preview.

Two independent knobs previously starved this loop and are now surfaced with better
defaults (see the Solver-regime flags above): the `--h-max` step-size cap and the
`--patience` early stop. If you want the old fast-but-coarse behaviour back, pass
`--h-max 0.1 --iterations 200`.

For a further quality bump at extra runtime, add `--num-samples 4` (Ours-avg) and/or
`--final-projection` (see Optional quality levers).

## GUI

An interactive Streamlit app (upload -> pick problem -> reconstruct) is provided in
`streamlit_app.py`. The sidebar exposes the same controls as the CLI — iterations,
`sigma0`, `beta`, and the solver-regime knobs (an "uncap step size" toggle for `h_max`,
plus `sigma_l` / `h0` / `patience`) and the optional quality levers (final-projection
checkbox, Ours-avg `num_samples`):

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/streamlit run \
    src/applications/bias_free_denoiser/streamlit_app.py \
    --server.address 127.0.0.1 --server.port 8501
```

For the `denoise` task the app also offers **live-webcam streaming denoise** via
[`streamlit-webrtc`](https://github.com/whitphx/streamlit-webrtc): pick the `denoise`
problem and the `Webcam (live)` source, and each incoming webcam frame is passed through
a single forward pass `D(y)` in real time (raise the sidebar `Noise sigma` to inject
synthetic noise and watch it removed). Live streaming is available for `denoise` ONLY —
the six iterative inverse problems each need hundreds of forward passes per image, which
is unsuitable for live video, so they remain snapshot/upload-only.

The core (`denoiser_prior.py`, `operators.py`, `solver.py`) is GUI-free and fully
usable headless / programmatically; only `streamlit_app.py` imports streamlit.
