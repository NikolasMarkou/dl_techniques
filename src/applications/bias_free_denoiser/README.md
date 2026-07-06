# Bias-Free Denoiser: Inverse Problems via the Implicit Prior

One unified interface that solves all of Kadkhodaie & Simoncelli's inverse problems
using the prior implicit in a bias-free denoiser. Every problem runs through the
**same** stochastic coarse-to-fine ascent loop; only the *measurement operator*
changes.

Reference: A. Kadkhodaie & E. P. Simoncelli, *"Stochastic Solutions for Linear
Inverse Problems using the Prior Implicit in a Denoiser"* (NeurIPS 2021). The
denoiser is a bias-free CliffordUNet (BF-CNN-style: strictly bias-free so it is
input-scale equivariant / degree-1 homogeneous, trained blind on additive Gaussian
noise with an MSE objective).

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
  The default `from_pretrained(...)` rebuilds the fully-convolutional graph at
  `(None, None, 3)` and transfers weights, so it runs any `H, W` divisible by 8 in a
  single pass; `resolution="fixed256"` loads the saved 256x256 graph. Exposes
  `residual(y) = D(y) - y` (the score estimate) plus `ingest` / `denorm` / `tile`.
- **`UniversalInverseSolver`** — `solve(operator, measurements=..., shape=..., seed=...)`
  runs the annealed ascent and returns `(best_y, info)` where `info` holds
  `sigma_values`, `iterations`, and (when measurements are given) `constraint_errors`.
- **`MeasurementOperator`** — the single load-bearing abstraction. Each subclass
  implements `measure` / `adjoint` / `project` / `init_mean` with orthonormal columns
  (`M^T M ~= I`) and **no dense `N x N` matrix** (everything is O(N) or O(N log N)).

### Prior sampling (Algorithm 1)

```python
prior = DenoiserPrior.from_pretrained("results/cliffordunet_denoiser_base_20260705_004751/best_model.keras")
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
    -m applications.bias_free_denoiser.main --problem all --iterations 200
```

Runs the selected problem(s) on a synthetic in-domain target (or `--image PATH` for a
real image, resized to `--size`) and writes a grid PNG — target, measured/degraded
view, reconstruction, and the `sigma_t` convergence curve per problem — to
repo-root `results/`. It is fully headless (`MPLBACKEND=Agg`). `--problem` accepts any
of the seven ids (`denoise`, `prior`, `inpaint`, `random_pixels`, `super_resolution`,
`deblur`, `compressive_sensing`) or `all`. Key flags:
`--checkpoint`, `--iterations`, `--sigma0`, `--beta`, `--seed`, `--size`, and the
per-problem knobs `--block`, `--keep-ratio`, `--sr-factor`, `--keep-fraction`,
`--measurement-ratio`.

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

The reconstruction quality tracks the iteration budget. A modest budget
(`--iterations 40-200`) runs in a few minutes and produces a coarse but sensible
result; the paper-quality reconstruction needs **hundreds to ~1000 iterations**
(early stopping halts once the effective noise `sigma_t` stops improving). Crank
`--iterations` for better quality at the cost of runtime.

## GUI

An interactive Streamlit app (upload -> pick problem -> reconstruct) is provided in
`streamlit_app.py`:

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
