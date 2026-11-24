# Multi-Image Super-Resolution from Unaligned Inputs with Per-Pixel Uncertainty

## Problem formulation

- **Input:** N low-resolution (LR) frames of the same scene, with unknown sub-pixel to multi-pixel shifts, rotations, and possibly local motion. One frame is designated as the **reference (base) frame**.
- **Output:**
  - A high-resolution (HR) image on the reference frame's pixel grid.
  - A per-pixel uncertainty map of the same size, either a variance σ² or a Laplace scale b.
- **Why it works:** Burst super-resolution reconstructs high-resolution images with richer detail by fusing the sub-pixel information spread across multiple low-resolution frames. Because each frame is shifted by a different sub-pixel amount, each one carries different aliasing artifacts, and those artifacts are complementary information about the underlying high-resolution image.

## Generic pipeline

1. **Reference selection.** Most networks treat frame 0 as the base. To super-resolve a chosen image, reorder the burst so that image comes first. Selection criteria include sharpness (variance of the Laplacian), lowest noise, or a user choice.
2. **Alignment** of every frame to the reference. Options:
   - Global homography.
   - Optical flow (PWC-Net style).
   - Deformable convolutions in feature space.
   - Implicit alignment through attention or state-space querying.
3. **Fusion.** Attention or affinity weighting across frames, which down-weights misaligned or occluded pixels.
4. **Reconstruction and upsampling** to the target scale (x4 or x8).
5. **Uncertainty head.** Extra output channels predicting log σ² (Gaussian) or log b (Laplace), trained with a negative log-likelihood (NLL) loss.
   - This captures **aleatoric** uncertainty (noise and ambiguity in the data).
   - Ensembles, MC dropout, or diffusion sampling add **epistemic** uncertainty (what the model itself does not know).

## Best models found

### Deterministic burst SR backbones (no native uncertainty)

| Model | Venue | Alignment / fusion | Notes |
|---|---|---|---|
| **QMambaBSR** | CVPR 2025 | Query state-space model | Its Query State Space Model handles both inter-frame querying and intra-frame scanning to fuse useful sub-pixels more efficiently. Reported new SOTA on four synthetic and real benchmarks. |
| **BIPNet (extended)** | TPAMI 2025 | Edge-enhanced alignment, pseudo-burst fusion | Combines edge-enhanced alignment, pseudo-burst feature fusion, and adaptive group upsampling, with SOTA results on SR, low-light enhancement, and denoising, plus lightweight mobile variants. |
| **Burstormer** | CVPR 2023 | Enhanced deformable alignment | Reaches SOTA across burst SR, low-light enhancement, and denoising, often with fewer parameters and faster runtime. |
| **FBAnet** | ICCV 2023 | Homography plus affinity fusion | Uses simple homography alignment instead of pixel-wise alignment, then a Federated Affinity Fusion strategy to aggregate complementary information across frames. Built for real misaligned bursts. Introduces the RealBSR dataset. |
| **GenMFSR** | arXiv 2026 | Diffusion latent | Presented as the first burst RAW-to-RGB diffusion model, reporting SOTA perceptual quality with realistic textures. Repeated sampling yields a variance map. |
| **DBSR / DeepRep** | CVPR / ICCV 2021 | Flow plus attention fusion | Foundational baselines: one learns alignment of noisy RAW frames in latent space via optical flow and fuses them with attention; the other reparameterizes the reconstruction objective in a learned latent space. |

### Models with native per-pixel uncertainty

| Model | Type | Uncertainty mechanism |
|---|---|---|
| **PIUnet** (Valsesia & Magli) | Supervised, multi-temporal | Estimates per-pixel aleatoric uncertainty for multi-temporal SR using a Laplacian NLL loss. Official PyTorch code is available, and it is benchmarked on PROBA-V. |
| **Self-Supervised Uncertainty SR** (Zheng et al.) | Self-supervised, no HR ground truth | A self-supervised Gaussian NLL loss jointly estimates pixel-wise posterior means and variances. |
| **SuperF** | Test-time optimization (implicit neural representation), no training data | Shares one implicit neural representation across frames and jointly optimizes each frame's affine alignment. Its uncertainty is a band-specific estimate per LR frame, so it flags unreliable input pixels rather than giving output-pixel uncertainty directly. It handles shifts of several pixels without a pre-alignment step. |
| **Deep ensembles / MC dropout** | Wrapper for any backbone | For ESRGAN, an ensemble of 5 generators worked best, and its per-pixel standard deviation served as a usable proxy for per-pixel error. |

## Recommended architecture

- **Backbone:** QMambaBSR or Burstormer. For large real-world misalignment, add a homography pre-alignment step as in FBAnet.
- **Reference:** reorder the burst so the selected image is frame 0.
- **Uncertainty head:** add channels for μ and log σ² (or log b), trained with Gaussian or Laplacian NLL.
- **Epistemic term:** a 3 to 5 member ensemble. Total variance = mean of the predicted σ² + variance of the ensemble means μ.
- **No HR ground truth (e.g. satellite):** use the Zheng et al. self-supervised NLL scheme, or SuperF with the Gaussian NLL loss.
- **Perceptual quality over fidelity:** GenMFSR, with uncertainty taken as the variance across multiple diffusion samples.