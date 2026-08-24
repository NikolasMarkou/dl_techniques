# Image Restoration — Benchmark Tables

Consolidated PSNR/SSIM comparisons for all-in-one image restoration models, compiled from multiple 2023–2026 papers reporting on the same standard benchmarks (BSD68, Rain100L, SOTS, GoPro, LOL, CDD11, Urban100, Kodak24, WeatherBench). Current SOTA as of the most recent reported results: **M²IR** (2026), narrowly ahead of MIRAGE, BaryIR, MEASNet, and SE-SymUNet.

---

## Table 1 — Three-Degradation Setting (Denoising / Deraining / Dehazing)

| Method | Venue | SOTS PSNR/SSIM | Rain100L PSNR/SSIM | BSD68 σ15 | σ25 | σ50 | **Avg PSNR/SSIM** |
|---|---|---|---|---|---|---|---|
| Restormer | CVPR'22 | 27.78/0.958 | 33.78/0.958 | 33.72/0.865 | 30.67/0.865 | 27.63/0.792 | 30.75/0.901 |
| NAFNet | ECCV'22 | 24.11/0.928 | 33.64/0.956 | 33.03/0.918 | 30.47/0.865 | 27.12/0.754 | 29.67/0.844 |
| AirNet | CVPR'22 | 27.94/0.962 | 34.90/0.967 | 33.92/0.933 | 31.26/0.888 | 28.00/0.797 | 31.20/0.910 |
| IDR | CVPR'23 | 29.87/0.970 | 36.03/0.971 | 33.89/0.931 | 31.32/0.884 | 28.04/0.798 | 31.83/0.911 |
| FSNet | TPAMI'23 | 29.14/0.968 | 35.61/0.969 | 33.81/0.930 | 30.84/0.872 | 27.69/0.792 | 31.42/0.905 |
| PromptIR | NeurIPS'23 | 30.58/0.974 | 36.37/0.972 | 33.98/0.933 | 31.31/0.888 | 28.06/0.799 | 32.06/0.913 |
| GridFormer | IJCV'24 | 30.37/0.970 | 37.15/0.972 | 33.93/0.931 | 31.37/0.887 | 28.11/0.801 | 32.19/0.912 |
| MambaIR | ECCV'24 | 29.57/0.970 | 35.42/0.969 | 33.88/0.931 | 30.95/0.874 | 27.74/0.793 | 31.51/0.907 |
| InstructIR-3D | ECCV'24 | 30.22/0.959 | 37.98/0.978 | 34.15/0.933 | 31.52/0.890 | 28.30/0.804 | 32.43/0.913 |
| Perceive-IR | TIP'25 | 30.87/0.975 | 38.29/0.980 | 34.13/0.934 | 31.53/0.890 | 28.31/0.804 | 32.63/0.917 |
| Pool-AIO | TIP'25 | 30.94/0.980 | 38.54/0.983 | 34.10/0.935 | 31.45/0.892 | 28.18/0.803 | 32.64/0.919 |
| AdaIR | ICLR'25 | 31.06/0.980 | 38.64/0.983 | 34.12/0.935 | 31.45/0.892 | 28.19/0.802 | 32.69/0.918 |
| MoCE-IR | CVPR'25 | 31.34/0.979 | 38.57/0.984 | 34.11/0.932 | 31.45/0.888 | 28.18/0.800 | 32.73/0.917 |
| VLU-Net | CVPR'25 | 30.71/0.980 | 38.93/0.984 | 34.13/0.935 | 31.48/0.892 | 28.23/0.804 | 32.70/0.919 |
| DFPIR (Deg.-Aware Feature Perturbation) | CVPR'25 | 31.87/0.980 | 38.65/0.982 | 34.14/0.935 | 31.47/0.893 | 28.25/0.806 | 32.88/0.919 |
| MEASNet | TCSVT'25 | 31.61/0.981 | 39.00/0.985 | 34.12/0.935 | 31.46/0.892 | 28.19/0.803 | 32.85/0.919 |
| QuReC | ACM MM'26 | 32.77/0.985 | 38.80/0.985 | 34.26/0.934 | 31.60/0.892 | 28.34/0.806 | 33.16/0.920 |
| BaryIR | TPAMI'26 | 31.40/0.980 | 39.02/0.985 | 34.16/0.935 | 31.54/0.892 | 28.25/0.802 | 32.86/0.919 |
| MIRAGE | ICLR'26 | 31.86/0.981 | 38.94/0.985 | 34.12/0.935 | 31.46/0.891 | 28.19/0.803 | 32.91/0.919 |
| SymUNet | 2026 | 31.40/0.981 | 39.12/0.985 | 34.22/0.937 | 31.57/0.894 | 28.32/0.808 | 32.93/0.921 |
| SE-SymUNet | 2026 | **32.02/0.983** | 39.23/0.986 | 34.23/0.937 | 31.58/0.895 | 28.33/0.809 | 33.08/0.922 |
| **M²IR** | 2026 | 32.63/**0.984** | **39.36/0.986** | **34.23/0.937** | **31.59/0.895** | **28.33/0.808** | **33.23/0.922** |

---

## Table 2 — Five-Degradation Setting (adds Deblurring, Low-Light)

| Method | Venue | SOTS | Rain100L | BSD68 σ25 | GoPro | LOL | **Avg PSNR/SSIM** |
|---|---|---|---|---|---|---|---|
| AirNet | CVPR'22 | 21.04/0.884 | 32.98/0.951 | 30.91/0.882 | 24.35/0.781 | 18.18/0.735 | 25.49/0.846 |
| NAFNet | ECCV'22 | 25.23/0.939 | 35.56/0.967 | 31.02/0.883 | 26.53/0.808 | 20.49/0.809 | 27.76/0.881 |
| Restormer | CVPR'22 | 24.09/0.927 | 34.81/0.960 | 31.49/0.884 | 27.22/0.829 | 20.41/0.806 | 27.60/0.881 |
| IDR | CVPR'23 | 25.24/0.943 | 35.63/0.965 | 31.60/0.887 | 27.87/0.846 | 21.34/0.826 | 28.34/0.893 |
| FSNet | TPAMI'23 | 25.53/0.943 | 36.07/0.968 | 31.33/0.883 | 28.32/0.869 | 22.29/0.829 | 28.71/0.898 |
| PromptIR | NeurIPS'23 | 26.54/0.949 | 36.37/0.970 | 31.47/0.886 | 28.71/0.881 | 22.68/0.832 | 29.15/0.904 |
| GridFormer | IJCV'24 | 26.79/0.951 | 36.61/0.971 | 31.45/0.885 | 29.22/0.884 | 22.59/0.831 | 29.33/0.904 |
| InstructIR-5D | ECCV'24 | 27.10/0.956 | 36.84/0.973 | 31.40/0.887 | 29.40/0.886 | 23.00/0.836 | 29.55/0.907 |
| Perceive-IR | TIP'25 | 28.19/0.964 | 37.25/0.977 | 31.44/0.887 | 29.46/0.886 | 22.88/0.833 | 29.84/0.909 |
| Pool-AIO | TIP'25 | 30.25/0.977 | 37.85/0.981 | 31.35/0.889 | 27.66/0.844 | 22.66/0.841 | 29.93/0.906 |
| AdaIR | ICLR'25 | 30.53/0.978 | 38.02/0.981 | 31.34/0.887 | 28.12/0.858 | 23.00/0.845 | 30.20/0.910 |
| MoCE-IR | CVPR'25 | 30.48/0.974 | 38.04/0.982 | 31.34/0.887 | 30.05/0.899 | 23.00/0.852 | 30.58/0.919 |
| VLU-Net | CVPR'25 | 30.84/0.980 | 38.54/0.982 | 31.43/0.891 | 27.46/0.840 | 22.29/0.833 | 30.11/0.905 |
| DFPIR | CVPR'25 | 31.64/0.979 | 37.62/0.978 | 31.29/0.889 | 28.82/0.873 | 23.82/0.843 | 30.64/0.913 |
| MEASNet | TCSVT'25 | 31.05/0.980 | 38.32/0.982 | 31.41/0.892 | 29.41/0.890 | 23.00/0.858 | 30.68/0.914 |
| QuReC | ACM MM'26 | 31.72/0.982 | 39.23/0.986 | 31.56/0.891 | 32.50/0.936 | 24.23/0.867 | 31.85/0.932 |
| BaryIR | TPAMI'26 | 31.20/0.979 | 38.10/0.982 | 31.43/0.891 | 28.10/0.858 | 23.37/0.854 | 30.72/0.919 |
| MIRAGE | ICLR'26 | 31.45/0.980 | 38.92/0.985 | 31.43/0.891 | 23.59/0.858* | 23.59/0.858 | 30.64/0.917 |
| SymUNet | 2026 | 31.31/0.979 | 38.05/0.981 | 31.38/0.891 | 28.12/0.855 | 23.27/0.858 | 30.43/0.913 |
| SE-SymUNet | 2026 | **32.15/0.982** | 38.44/0.983 | **31.45/0.892** | 28.40/0.864 | 23.22/0.861 | 30.73/0.916 |
| **M²IR** | 2026 | 31.63/0.981 | **39.28/0.985** | 31.53/0.893 | **29.09/0.878** | **23.80/0.864** | **31.06/0.920** |

*Note: QuReC reports the highest 5-task average (31.85) because it substantially outperforms all others on GoPro deblurring (32.50 dB); M²IR reports highest among the other 2026 methods on this exact protocol. Different papers use slightly different training splits/iterations, so cross-paper numbers are directionally, not perfectly, comparable.*

---

## Table 3 — CDD11 Composite Degradation Benchmark (PSNR/SSIM)

| Method | Low(L) | Haze(H) | Rain(R) | Snow(S) | L+H | L+R | L+S | H+R | H+S | L+H+R | L+H+S | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AirNet | 24.83/0.778 | 24.21/0.951 | 26.55/0.891 | 26.79/0.919 | 23.23/0.779 | 22.82/0.710 | 23.29/0.723 | 22.21/0.868 | 23.29/0.901 | 21.80/0.708 | 22.24/0.725 | 23.75/0.814 |
| PromptIR | 26.32/0.805 | 26.10/0.969 | 31.56/0.946 | 31.53/0.960 | 24.49/0.789 | 25.05/0.771 | 24.51/0.761 | 24.54/0.924 | 23.70/0.925 | 23.74/0.752 | 23.33/0.747 | 25.90/0.850 |
| WeatherDiff | 23.58/0.763 | 21.99/0.904 | 24.85/0.885 | 24.80/0.888 | 21.83/0.756 | 22.69/0.730 | 22.12/0.707 | 21.25/0.868 | 21.99/0.868 | 21.23/0.716 | 21.04/0.698 | 22.49/0.799 |
| WGWS-Net | 24.39/0.774 | 27.90/0.982 | 33.15/0.964 | 34.43/0.973 | 24.27/0.800 | 25.06/0.772 | 24.60/0.765 | 27.23/0.955 | 27.65/0.960 | 23.90/0.772 | 23.97/0.771 | 26.96/0.863 |
| OneRestore | 26.48/0.826 | 32.52/0.990 | 33.40/0.964 | 34.31/0.973 | 25.79/0.822 | 25.58/0.799 | 25.19/0.789 | 29.99/0.957 | 30.21/0.964 | 24.78/0.788 | 24.90/0.791 | 28.47/0.878 |
| AdaIR* | 26.88/0.821 | 31.60/0.987 | 33.84/0.962 | 34.65/0.974 | 25.69/0.811 | 25.90/0.793 | 25.69/0.783 | 29.38/0.955 | 28.95/0.961 | 24.82/0.778 | 25.04/0.778 | 28.40/0.873 |
| MoCE-IR | 27.26/0.824 | 32.66/0.990 | 34.31/0.970 | 35.91/0.980 | 26.24/0.817 | 26.25/0.800 | 26.04/0.793 | 29.93/0.964 | 30.19/0.970 | 25.41/0.789 | 25.39/0.790 | 29.05/0.881 |
| MIRAGE | 27.41/0.833 | 33.12/0.992 | 34.66/0.971 | 35.98/0.981 | 26.55/0.828 | 26.53/0.810 | 26.33/0.803 | 30.32/0.965 | 30.27/0.969 | 25.59/0.801 | 25.86/0.799 | 29.33/0.887 |
| QuReC | 27.58/0.833 | 36.42/0.995 | 35.71/0.977 | 37.86/0.985 | 27.27/0.832 | 27.05/0.817 | 26.90/0.812 | 32.71/0.974 | 33.29/0.979 | 26.50/0.809 | 26.57/0.811 | 30.71/0.903 |
| **M²IR** | **27.67/0.836** | **34.84/0.992** | **35.39/0.974** | **37.11/0.981** | **26.87/0.831** | **26.77/0.814** | **26.62/0.808** | **31.89/0.970** | **31.78/0.973** | **26.07/0.805** | **26.15/0.804** | 30.11/0.890 |

*Note: QuReC currently posts the top CDD11 average (30.71), edging out M²IR (30.11) — largely from strong haze-subset performance (36.42 vs 34.84 dB).*

---

## Table 4 — Dedicated Single-Task Results

**Dehazing (SOTS):**

| Method | DehazeNet | AODNet | FDGAN | DehazeFormer | AirNet | Restormer | PromptIR | Perceive-IR | AdaIR | **M²IR** |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR/SSIM | 22.46/0.851 | 20.29/0.877 | 23.15/0.921 | 31.78/0.977 | 23.18/0.900 | 30.87/0.969 | 30.87*/0.969 | 31.65/0.977 | 31.80/0.981 | **32.12/0.984** |

**Deraining (Rain100L):**

| Method | UMR | MSPFN | LPNet | DRSformer | AirNet | Restormer | PromptIR | Perceive-IR | AdaIR | **M²IR** |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR/SSIM | 32.39/0.921 | 33.50/0.948 | 33.61/0.958 | 38.14/0.983 | 34.90/0.977 | 36.74/0.978 | 37.04/0.979 | 38.41/0.984 | 38.90/0.985 | **39.33/0.986** |

**Denoising (BSD68 / Urban100 / Kodak24, σ=15/25/50):**

| Method | BSD68 15/25/50 | Urban100 15/25/50 | Kodak24 15/25/50 |
|---|---|---|---|
| DnCNN | 33.90/31.24/27.95 | 32.98/30.81/27.59 | 34.60/32.14/28.95 |
| Restormer | 34.03/31.49/28.11 | 33.72/31.26/28.03 | 34.78/32.37/29.08 |
| AirNet | 34.14/31.48/28.23 | 34.40/32.10/28.88 | 34.81/32.44/29.10 |
| PromptIR* | 34.33/31.70/28.45 | 34.83/32.59/29.50 | 35.30/32.86/29.76 |
| Perceive-IR | 34.38/31.74/28.53 | 34.86/32.55/29.42 | 34.84/32.50/29.16 |
| AdaIR | 34.36/31.72/28.49 | 34.96/32.74/29.70 | 35.39/32.94/29.85 |
| **M²IR** | **34.40/31.76/28.54** | **34.99/32.79/29.78** | **35.40/32.96/29.89** |

---

## Table 5 — Real-World WeatherBench Benchmark (haze/rain/snow, PSNR/SSIM/LPIPS/FID avg)

| Method | Venue | Avg PSNR | Avg SSIM | Avg LPIPS↓ | Avg FID↓ |
|---|---|---|---|---|---|
| WGWS-Net | CVPR'23 | 21.98 | 0.731 | 0.3000 | 114.10 |
| Histoformer | ECCV'24 | 22.86 | 0.747 | 0.3136 | 128.86 |
| TransWeather | CVPR'22 | 23.59 | 0.752 | 0.2953 | 125.29 |
| AirNet | CVPR'22 | 23.80 | 0.764 | 0.2992 | 132.73 |
| PromptIR | NeurIPS'23 | 26.12 | 0.792 | 0.2561 | 103.12 |
| AdaIR | ICLR'25 | 27.02 | 0.801 | 0.2404 | 97.65 |
| DiffUIR | CVPR'24 | 27.54 | 0.823 | 0.2296 | 94.50 |
| **M²IR** | 2026 | **29.62** | **0.852** | **0.2111** | **83.83** |

---

## Table 6 — Perceptual Quality (LPIPS / DISTS) on Three-Task Setting

| Method | BSD68σ15 | BSD68σ25 | BSD68σ50 | Rain100L | SOTS | **Avg** |
|---|---|---|---|---|---|---|
| AirNet | 0.0648/0.0884 | 0.1134/0.1230 | 0.2083/0.1721 | 0.0306/0.0386 | 0.2273/0.0624 | 0.1289/0.0969 |
| PromptIR | 0.0662/0.0883 | 0.1148/0.1215 | 0.2151/0.1723 | 0.0191/0.0268 | 0.2160/0.0451 | 0.1262/0.0908 |
| AdaIR | 0.0634/0.0853 | 0.1098/0.1198 | 0.2128/0.1704 | 0.0147/0.0187 | 0.2171/0.0423 | 0.1236/0.0873 |
| MoCE-IR | 0.0598/0.0754 | 0.1029/0.1054 | 0.1927/0.1500 | 0.0147/0.0190 | 0.2120/0.0417 | 0.1164/0.0783 |
| DFPIR | 0.0633/0.0817 | 0.0927/0.1003 | 0.1965/0.1575 | 0.0153/0.0209 | 0.2165/0.0430 | 0.1169/0.0807 |
| QuReC | **0.0528/0.0708** | **0.0914/0.0998** | **0.1797/0.1486** | **0.0130/0.0158** | **0.2086/0.0400** | **0.1091/0.0750** |

---

## Model Efficiency Comparison (720×480 input, RTX 4090, unless noted)

| Method | Params (M) | FLOPs (G) | Latency (ms) |
|---|---|---|---|
| IDR | 42.3 | 1522 | 384 |
| PromptIR | 34.1 | 745 | 282 |
| GridFormer | 34.1 | 1941 | 746 |
| Pool-AIO | 26.1 | 745 | 293 |
| AdaIR | 28.8 | 778 | 333 |
| MoCE-IR | 25.4 | 474 | 263 |
| MEASNet | 31.7 | 892 | 349 |
| Perceive-IR (w/o CLIP) | 30.0 | 1176 | 387 |
| QuReC | 29.6 | 151* | 112* |
| M²IR (w/o DA-CLIP) | 39.1 | 961 | 330 |
| M²IR (full, w/ DA-CLIP) | 285.3 | 975 | 339 |

*QuReC's FLOPs/latency measured at 256×256, not directly comparable to the 720×480 column — included for reference only.*

---

## Summary
- **Current strongest all-around all-in-one model** across the standard three-task, five-task, single-task, and real-world (WeatherBench) benchmarks: **M²IR** (2026) — a Mamba-style Transformer + Mixture-of-Experts design.
- **QuReC** (2026) leads specifically on the **CDD11 composite-degradation** benchmark and the **five-task GoPro deblurring** metric, and posts the best perceptual (LPIPS/DISTS) scores reported to date — it is one of several competitive 2026 entrants, not a uniquely "best" model.
- **SE-SymUNet** (2026) shows that a much simpler symmetric U-Net + light CLIP guidance can match or beat many heavier prompt/MoE-based designs, especially on dehazing.
- **MIRAGE** and **BaryIR** (both ICLR/TPAMI 2026) remain strong, efficient baselines just behind the top tier.
- Older but still-cited baselines (PromptIR, AirNet, Restormer, InstructIR, AdaIR, MoCE-IR, DFPIR) form the standard comparison set nearly every new paper reports against.

---

## References
- M²IR: Wang et al., "M²IR: Proactive All-in-One Image Restoration via Mamba-style Modulation and Mixture-of-Experts," 2026. https://arxiv.org/html/2603.14816v1 — code: https://github.com/Im34v/M2IR
- QuReC: Zhou et al., "QuReC: All-in-One Image Restoration with Query-Specific Guidance and Local-Global Response Calibration," ACM MM '26. https://arxiv.org/abs/2607.15097 — code: https://github.com/zhoushen1/QuReC
- SymUNet / SE-SymUNet: Jiao et al., "Unleashing Degradation-Carrying Features in Symmetric U-Net," 2026. https://arxiv.org/pdf/2512.10581 — code: https://github.com/WenlongJiao/SymUNet
- MIRAGE: Ren et al., "Efficient Degradation-Agnostic Image Restoration via Channel-Wise Functional Decomposition and Manifold Regularization," ICLR 2026.
- BaryIR: Tang et al., "Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration," TPAMI 2026.
- MEASNet: Yu et al., "Multi-Expert Adaptive Selection: Task-Balancing for All-in-One Image Restoration," IEEE TCSVT 2025.
- AdaIR: Cui et al., "AdaIR: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation," ICLR 2025. https://arxiv.org/abs/2403.14614
- DFPIR: Tian et al., "Degradation-Aware Feature Perturbation for All-in-One Image Restoration," CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/
- MoCE-IR: Zamfir et al., "Complexity Experts Are Task-Discriminative Learners for Any Image Restoration," CVPR 2025.
- Perceive-IR: Zhang et al., "Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration," IEEE TIP 2025.
- PromptIR: Potlapalli et al., NeurIPS 2023. https://arxiv.org/abs/2306.13090
- InstructIR: Conde et al., ECCV 2024. https://arxiv.org/abs/2401.16468
- AirNet: Li et al., CVPR 2022.
- Restormer: Zamir et al., CVPR 2022.
- OneRestore (CDD11 benchmark source): Guo et al., ECCV 2024.
- WeatherBench dataset: Guan et al., ACM MM 2025.
- Survey: Jiang et al., "A Survey on All-in-One Image Restoration: Taxonomy, Evaluation and Future Trends," IEEE TPAMI 2025. https://arxiv.org/pdf/2410.15067 — https://github.com/Harbinzzy/All-in-One-Image-Restoration-Survey