# State-of-the-Art Loss Functions for Deep Neural Networks (CV + NLP)

**Scope:** Proven / production-grade and current-frontier (2024–2026) training objectives. Each entry gives a high-level idea, a formula sketch, target tasks, the metrics it moves, maturity, and a citation for later implementation.

**Maturity legend:** 🟢 battle-tested default · 🟡 strong, widely adopted · 🔵 frontier / active research (2025–2026)

**v2 (rev.):** arXiv IDs for every entry below were verified against source. Added keypoint-regression and classification losses, GRPO→R1 lineage, and the EIoU arXiv ID.

---

## Part I — Computer Vision

### 1. Object Detection

Modern detectors use a **compound loss** = classification term + localization (bbox) term + (optionally) a distribution term. The 2024–2026 movement is almost entirely in the localization (IoU-family) and quality-aware classification terms.

#### 1.1 Classification / quality terms

| Loss | Idea (high level) | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **Focal Loss** | Down-weights easy negatives so rare/hard objects dominate the gradient. | `FL = -α(1-p_t)^γ log(p_t)` | Dense detection (RetinaNet, YOLO cls) | AP, AP_small | 🟢 | Lin et al. 2017, arXiv:1708.02002 |
| **Varifocal Loss (VFL)** | Asymmetric focal that weights positives by their IoU-aware classification score (IACS), aligning cls confidence with localization quality. | `VFL = -q(q logp + (1-q)log(1-p))` for pos; focal for neg | Dense detection (VarifocalNet, YOLOv8+) | AP, AP@0.75 | 🟢 | Zhang et al. 2020, arXiv:2008.13367 |
| **Quality/Generalized Focal (GFL) + Distribution Focal Loss (DFL)** | Merges cls+quality into one continuous label; DFL models each box edge as a discrete distribution instead of a Dirac, learning localization uncertainty. | `DFL = -((y_{i+1}-y)log S_i + (y-y_i)log S_{i+1})` | YOLOv8/v11/v12, RTMDet regression head | AP, boundary precision | 🟢 | Li et al. 2020, arXiv:2006.04388 |
| **Poly Loss** | Reframes CE/Focal as a Taylor series of polynomial bases; tuning the leading coefficient (ε) beats fixed Focal with one hyperparameter. | `CE + ε(1-p_t)` | Detection cls, classification | AP, top-1 | 🟡 | Leng et al. 2022, arXiv:2204.12511 |

#### 1.2 Bounding-box regression (IoU family) — the most active 2023–2026 area

| Loss | What it adds over plain IoU | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|
| **GIoU** | Penalty from smallest enclosing box → non-zero gradient when boxes don't overlap. | General detection | AP | 🟢 | Rezatofighi et al. 2019, arXiv:1902.09630 |
| **DIoU / CIoU** | DIoU adds normalized center-distance; CIoU adds aspect-ratio consistency (YOLOv4–v8 default). | General detection | AP, convergence speed | 🟢 | Zheng et al. 2020, arXiv:1911.08287 |
| **EIoU / Focal-EIoU** | Splits CIoU's aspect term into explicit width/height penalties; focal weighting up-weights high-quality (high-IoU) examples. | General detection | AP@0.75 | 🟡 | Zhang et al. 2021, arXiv:2101.08158 (Neurocomputing 2022) |
| **SIoU** | Adds an **angle cost** so the box moves along the correct direction first (angle+distance+shape+IoU). | Real-time detectors | AP, convergence | 🟡 | Gevorgyan 2022, arXiv:2205.12740 |
| **WIoU (Wise-IoU)** | Dynamic **non-monotonic** focusing (v3): uses each box's outlier degree vs. the batch to down-weight low-quality/noisy labels; builds on Focal-EIoU. | Noisy-label / real-world detection | AP robustness | 🟡 | Tong et al. 2023, arXiv:2301.10051 |
| **MPDIoU** | Minimum-point-distance metric; fixes the case where pred and GT share aspect ratio but differ in size. | Detection + instance seg | AP | 🟡 | Ma et al. 2023, arXiv:2307.07662 |
| **Alpha-IoU** | Power-parameterized family (`IoU^α`) to sharpen high-IoU gradients / tune convergence. | General detection | AP@0.75 | 🟡 | He et al. 2021, arXiv:2110.13675 |
| **Focaler-IoU** | Reweights IoU across the easy/hard regime via linear-interval mapping; composable with any IoU loss. | Imbalanced-difficulty detection | AP | 🔵 (2024) | Zhang 2024 |
| **Shape-IoU / FPDIoU / DAPIoU / MoEIoU** | 2024–2026 refinements: shape/scale-adaptive weighting, four-point-distance penalties, angle-precision terms, and a mixture-of-experts blend of IoU losses reporting best mAP50:95 on YOLOv12/YOLO26. | Small-object, defect, general | mAP50:95 | 🔵 | MoEIoU 2026, arXiv:2606.00844; DAPIoU 2025 (IEEE Access 10.1109/ACCESS.2025.3567767) |

> **Practical note:** YOLOv8→v12 default = **DFL + CIoU**; swapping the IoU term for SIoU/WIoU/MPDIoU is the standard cheap accuracy lever. Tiny-object work (remote sensing) uses **Joint Optimization Loss** dynamically balancing cls vs. reg per sample (Shi et al. 2025, *Remote Sensing* 17(8):1476).

#### 1.3 Set-prediction (DETR family)

| Loss | Idea | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|
| **Hungarian / bipartite-matching loss** | One-to-one optimal assignment between queries and GT → removes NMS; per-match = cls (focal) + L1 + GIoU. | DETR, Deformable-DETR, DINO, RT-DETR, Co-DETR, RF-DETR | AP (COCO 63%+ AP for DINO/Co-DETR) | 🟢 | Carion et al. 2020, arXiv:2005.12872; DINO 2022, arXiv:2203.03605 |
| **Contrastive denoising (CDN)** | Adds noised GT queries as auxiliary denoising task to stabilize matching and speed convergence. | DINO / RT-DETR | AP, convergence | 🟡 | DINO 2022, arXiv:2203.03605 |

#### 1.4 Keypoint / landmark / pose regression

| Loss | Idea | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|
| **Wing Loss** | Amplifies gradients on small/medium localization errors (log-region near zero, L1 far out) → better on fine landmark offsets than L2/L1. | Facial landmarks, pose | NME, PCK | 🟡 | Feng et al. 2018, arXiv:1711.06753 |
| **Adaptive Wing Loss** | Makes the Wing curvature adapt to pixel type; standard for heatmap-based landmark regression. | Heatmap landmark/pose | NME, AUC | 🟡 | Wang et al. 2019, arXiv:1904.07399 |
| **OKS loss** | Optimizes Object Keypoint Similarity directly (the COCO pose metric) instead of per-joint L2. | 2D human pose (YOLO-pose, RTMPose) | AP (OKS) | 🟡 | Maji et al. 2022 (YOLO-Pose), arXiv:2204.06806 |

---

### 2. Semantic / Instance / Medical Segmentation

Segmentation losses split into **distribution-based** (CE-derived), **region-based** (overlap), **boundary-based**, and **compound**. 2024–2026 practice heavily favors compound (region + boundary) and adaptive variants.

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **Cross-Entropy / Weighted-CE** | Per-pixel classification; class weights for imbalance. | `-Σ w_c y_c log p_c` | All segmentation | mIoU, pixel-acc | 🟢 | — |
| **Dice Loss** | Directly optimizes overlap (F1 of masks); robust to fg/bg imbalance. | `1 - 2|A∩B|/(|A|+|B|)` | Medical, general | Dice/DSC, IoU | 🟢 | Milletari et al. 2016 (V-Net), arXiv:1606.04797 |
| **Generalized Dice (GDL)** | Per-class inverse-frequency weighting for multi-class imbalance. | weighted Dice | Multi-organ | mDice | 🟢 | Sudre et al. 2017, arXiv:1707.03237 |
| **Tversky Loss** | Asymmetric FP vs FN control via α,β → tune precision/recall (e.g. penalize missed lesions). | `TP/(TP+αFP+βFN)` | Lesion/vessel/tumor seg | recall, Dice | 🟢 | Salehi et al. 2017, arXiv:1706.05721 |
| **Focal Tversky** | Tversky + focusing exponent γ to emphasize hard, small ROIs. | `(1-TI)^γ` | Small-structure medical seg | Dice on small classes | 🟡 | Abraham & Khan 2018, arXiv:1810.07842 |
| **Lovász-Softmax** | Differentiable surrogate that directly optimizes the IoU (Jaccard) via the Lovász extension. | convex Lovász hull of Jaccard | Cityscapes-style seg | mIoU | 🟡 | Berman et al. 2018, arXiv:1705.08790 |
| **Boundary Loss** | Distance-map-weighted term integrating error over region contours; complements region losses on thin/irregular boundaries. | integral over boundary w/ signed dist. map | Highly-imbalanced medical seg | Hausdorff, boundary-F | 🟡 | Kervadec et al. 2018, arXiv:1812.07032 |
| **clDice (centerline Dice)** | Topology-preserving loss enforcing connectivity of tubular structures. | Dice on morphological skeletons | Vessels, roads, neurons | clDice, Betti error | 🟡 | Shit et al. 2021, arXiv:2003.07311 |
| **Unified Focal Loss** | Generalizes CE, Focal, Dice, Focal-Tversky under shared hyperparameters; shrinks the tuning space. | grouped focal+region | Class-imbalanced seg | Dice, mIoU | 🟡 | Yeung et al. 2021, arXiv:2102.04525 |
| **Combo Loss** | Weighted CE + Dice — the workhorse default for most medical pipelines. | `λ·CE + (1-λ)·Dice` | Medical, general | Dice | 🟢 | Taghanaki et al. 2019 |
| **Adaptive TverskyCE (2025)** | Learns α,β (and CE weight) during training rather than fixing them; +Dice on pancreas/UNet-3D over static Tversky. | dynamic α,β | 3D medical seg | Dice | 🔵 | Zhang et al. 2025 |

> **Practical note:** 2025 surveys converge on **region+boundary compound losses** (e.g. Dice/Tversky + boundary or clDice) as the safe SOTA default; Focal-Tversky is the recommended single pick for imbalanced/complex medical cases (arXiv:2312.05391, "Loss Functions in the Era of Semantic Segmentation," 2025).

---

### 3. Face Recognition / Deep Metric Learning

All are **margin-based softmax** variants on L2-normalized features and class weights (angular embedding).

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **CosFace (LMCL)** | Additive cosine margin. | `cos θ_y - m` | Face verification/ID | TAR@FAR, LFW/IJB | 🟢 | Wang et al. 2018, arXiv:1801.09414 |
| **ArcFace** | Additive **angular** margin — cleaner geometric interpretation, still the reference baseline. | `cos(θ_y + m)` | Face rec, retrieval, re-ID | LFW, IJB-C, MegaFace | 🟢 | Deng et al. 2019, arXiv:1801.07698 |
| **MagFace** | Magnitude of the feature encodes quality; margin scales with feature norm → clusters by reliability. | norm-adaptive margin | Face rec + quality est. | IJB-B/C, quality AUC | 🟡 | Meng et al. 2021, arXiv:2103.06627 |
| **ElasticFace** | Draws the margin from a distribution (not a fixed scalar) for more flexible class boundaries. | `cos(θ_y + m), m∼N(μ,σ)` | Face rec | IJB-C, LFW | 🟡 | Boutros et al. 2021, arXiv:2109.09416 |
| **AdaFace** | Quality-**adaptive** margin: uses feature norm as an image-quality proxy; emphasizes hard samples only when quality is high, ignores unidentifiable ones. | margin g(‖z‖) | Low-quality / surveillance face rec | IJB-B/C, TinyFace | 🟡 (current SOTA default) | Kim et al. 2022, arXiv:2204.00964 |

---

### 4. Self-Supervised & Vision-Language Pretraining

The frontier: sigmoid contrastive + self-distillation + masked prediction, combined into unified recipes (SigLIP 2, DINOv3, TIPSv2).

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **InfoNCE / NT-Xent (contrastive)** | Softmax over one positive vs many negatives in a batch; foundation of CLIP/SimCLR. | `-log( e^{s+/τ} / Σ e^{s/τ} )` | VL pretraining, retrieval, SSL | zero-shot acc, R@k | 🟢 | Oord et al. 2018, arXiv:1807.03748; CLIP: Radford et al. 2021, arXiv:2103.00020 |
| **Sigmoid loss (SigLIP)** | Pairwise binary cross-entropy on every image–text pair; **no global softmax normalization** → memory-friendly, works at small batch, better alignment. | `Σ log σ(z_{ij}·label)` | VL pretraining | zero-shot ImageNet, retrieval | 🟢 | Zhai et al. 2023, arXiv:2303.15343 |
| **SigLIP 2 recipe** | Sigmoid loss **+** decoder captioning (LocCa) **+** self-distillation **+** masked prediction **+** online data curation; multilingual (109 langs). | multi-term sum | zero-shot cls, retrieval, dense/localization, VLM backbone | ImageNet ZS, ADE20k, retrieval | 🔵 (Feb 2025) | Tschannen et al. 2025, arXiv:2502.14786 |
| **DINO self-distillation** | Student matches EMA-teacher soft output over learnable prototypes via CE across multi-crop views; **no negatives, no labels**. | `-Σ P_t log P_s` | SSL vision backbone | k-NN / linear probe | 🟢 | Caron et al. 2021, arXiv:2104.14294; DINOv2 2023, arXiv:2304.07193 |
| **iBOT masked-image loss** | Adds masked-patch prediction (BEiT-style) on top of DINO for dense features. | patch-level CE to teacher | Dense prediction pretraining | seg mIoU, depth | 🟡 | Zhou et al. 2021, arXiv:2111.07832 |
| **KoLeo regularizer** | Differential-entropy term spreading embeddings uniformly on the hypersphere (used in DINOv2/v3). | nearest-neighbor entropy | SSL feature quality | retrieval, uniformity | 🟡 | Sablayrolles et al. 2019 |
| **DINOv3 Gram anchoring** | New regularizer aligning the student's patch-feature **Gram matrix** to a teacher's, preventing dense-feature collapse when scaling to 7B params. | Gram-matrix consistency | Scaled SSL (7B backbone) | ADE20k 55.9 mIoU, DAVIS 83.3 J&F | 🔵 (Aug 2025) | Siméoni et al. 2025, arXiv:2508.10104 |

> **Practical note:** For frozen-backbone dense tasks in 2025–2026, DINOv3 and SigLIP 2 are the reference encoders; text-alignment (zero-shot) is added post-hoc with a CLIP/LiT-style contrastive (InfoNCE) head on a frozen vision tower.

---

### 5. Generative Models & Image Restoration

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **Denoising / noise-prediction (DDPM ε-loss)** | Regress the added noise (or v-parameterization) at each timestep. | `E‖ε - ε_θ(x_t,t)‖²` | Diffusion image/audio gen | FID, IS | 🟢 | Ho et al. 2020, arXiv:2006.11239 |
| **Flow Matching (conditional FM)** | Regress a velocity field transporting noise→data along probability paths; simulation-free, stabler than score matching. | `E‖v_θ(x_t,t) - u_t‖²` | Modern generative (images, audio, molecules) | FID, sample steps | 🔵→🟢 | Lipman et al. 2022, arXiv:2210.02747 |
| **Rectified Flow** | Straight-line probability paths → few-step / one-step sampling via "reflow"; backbone of SD3, FLUX. | linear-path FM + reflow | Text-to-image/audio/video SOTA | FID, GenEval, NFE | 🔵→🟢 | Liu et al. 2022, arXiv:2209.03003; SD3: Esser et al. 2024, arXiv:2403.03206 |
| **LPIPS (perceptual)** | L2 in a pretrained deep-feature space; correlates with human perception far better than PSNR/SSIM. | `Σ_l ‖w_l⊙(φ_l(x)-φ_l(y))‖²` | Super-res, restoration, gen | LPIPS, DISTS | 🟢 (still the perceptual default 2025) | Zhang et al. 2018, arXiv:1801.03924 |
| **DISTS** | Combines structure + texture similarity; robust to texture resampling. | struct+texture SSIM in deep feats | Texture-heavy SR/restoration | DISTS | 🟡 | Ding et al. 2020, arXiv:2004.07728 |
| **GAN / adversarial (relativistic, hinge)** | Discriminator supplies a perceptual-realism gradient; core of ESRGAN/Real-ESRGAN, still SOTA for perceptual SR. | min-max / hinge | Perceptual SR, deblur | NIQE, CLIP-IQA, FID | 🟢 | ESRGAN 2018; NTIRE/AIM 2025 benchmarks confirm |

> **Practical note (SR/restoration, NTIRE/AIM 2025):** typical winning recipe = pixel loss (L1/MSE) + **LPIPS** + adversarial, with staged training; PSNR track uses L1/MSE only, perceptual track adds LPIPS+GAN. Perception–distortion tradeoff means you cannot maximize PSNR and LPIPS simultaneously.

---

## Part II — NLP / LLM

### 1. Core token-level objectives

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **Cross-Entropy (next-token)** | Standard autoregressive MLE. | `-Σ log p(x_t|x_<t)` | Pretraining, SFT | perplexity | 🟢 | — |
| **Label smoothing** | Softens one-hot targets → calibration, less overconfidence. | `(1-ε)y + ε/K` | Classification, MT | acc, ECE, BLEU | 🟢 | Müller et al. 2019, arXiv:1906.02629 |
| **Z-loss** | Regularizes the softmax log-partition (logit norm) for training stability at scale. | `λ (log Z)²` | Large-scale pretraining | stability | 🟡 | PaLM (Chowdhery 2022) |

### 2. Preference Optimization / Alignment (offline)

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **DPO** | Reparameterizes RLHF's reward via the policy itself; single logistic loss on chosen>rejected, no reward model, no sampling. | `-log σ(β[Δlog π_θ - Δlog π_ref])` | Preference alignment | win-rate (AlpacaEval, Arena) | 🟢 (dominant default) | Rafailov et al. 2023, arXiv:2305.18290 |
| **IPO** | Fixes DPO's overfitting to deterministic preferences with a squared-error regularizer around a target margin. | `(Δ - 1/2β)²` | Alignment (noisy prefs) | win-rate, robustness | 🟡 | Azar et al. 2023, arXiv:2310.12036 |
| **KTO** | Learns from **unpaired** thumbs-up/down using a Kahneman-Tversky utility; no preference pairs needed. | prospect-theory value fn | Alignment from binary feedback | win-rate | 🟡 | Ethayarajh et al. 2024, arXiv:2402.01306 |
| **ORPO** | Reference-free; folds preference (odds-ratio penalty) into the SFT loss → single-stage alignment (Gemma 2). | `CE + λ·log-odds-ratio` | Combined SFT+align | win-rate | 🟡 | Hong et al. 2024, arXiv:2403.07691 |
| **SimPO** | Reference-free; length-normalized implicit reward + target margin γ → matches/beats DPO cheaper. | `-log σ(β/|y|·Δlogp - γ)` | Alignment | win-rate, length bias | 🟡 (used in Qwen2.5) | Meng et al. 2024, arXiv:2405.14734 |
| **CPO** | Sequence-likelihood reward + BC (NLL) regularizer; drops the reference model. | `DPO-bound + NLL` | MT, alignment | COMET, win-rate | 🟡 | Xu et al. 2024, arXiv:2401.08417 |

> **Practical note:** Head-to-head benchmarks (2025, e.g. arXiv:2410.04203, 2406.16061) find **no variant dominates DPO across the board**; the margin/length-normalization term is the common ingredient of the improvements. Real 2025 stacks mix them (DPO + SimPO pass; KTO for unpaired telemetry).

### 3. RL for Reasoning (online, verifiable rewards)

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **PPO** | Clipped policy-gradient with a learned value/critic + KL-to-reference; the classic RLHF final-pass workhorse. | clipped surrogate + KL | RLHF, reasoning | reward, win-rate | 🟢 | Schulman et al. 2017, arXiv:1707.06347 |
| **GRPO** | **Critic-free**: normalize rewards within a group of sampled completions to get the advantage → cheap, powers DeepSeek-R1. | group-normalized advantage × clipped ratio + KL | Math/code reasoning (RLVR) | pass@1, GSM8K/AIME | 🔵→🟢 (2024–2026 default) | Shao et al. 2024 (DeepSeekMath), arXiv:2402.03300; R1: arXiv:2501.12948 |
| **Dr. GRPO** | Removes GRPO's length and std normalization biases → unbiased token-level gradient, better token efficiency. | debiased GRPO | Reasoning | pass@1 per token | 🔵 | Liu et al. 2025, arXiv:2503.20783 |
| **DAPO** | Decoupled ("clip-higher") clipping + dynamic sampling + token-level loss + overlong reward shaping; stabilizes long-CoT at scale. | modified clipped PG | Long-CoT reasoning | AIME, stability | 🔵 (2025) | Yu et al. 2025, arXiv:2503.14476 |
| **GSPO** | Sequence-level importance ratio + sequence-level clipping → lower variance, stable especially for MoE. | seq-likelihood ratio | Large-scale / MoE RL | stability, pass@1 | 🔵 (2025) | Zheng et al. 2025, arXiv:2507.18071 |

### 4. Text Embeddings / Retrieval

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **Contrastive InfoNCE + hard negatives** | In-batch + mined hard negatives with temperature; the standard for dense retrievers (E5, BGE, GTE). | `-log e^{s+/τ}/Σe^{s/τ}` | Retrieval, STS, RAG embeddings | MTEB, nDCG, R@k | 🟢 | Wang (E5) 2022, arXiv:2212.03533; SimCSE: Gao et al. 2021, arXiv:2104.08821 |
| **Multiple-negatives ranking / matryoshka** | Ranking loss over many negatives; matryoshka nests multiple truncation dims in one embedding. | ranking + nested-dim CE | Scalable embeddings | MTEB | 🟡 | Kusupati et al. 2022 (MRL), arXiv:2205.13147 |

### 5. Knowledge Distillation (LLM compression)

| Loss | Idea | Formula sketch | Tasks | Metrics | Maturity | Cite |
|---|---|---|---|---|---|---|
| **Forward-KL (classic KD)** | Match teacher's full distribution; mass-covering → can over-smooth / hallucinate in generation. | `KL(p_T ‖ p_S)` | Model compression | task acc, ppl | 🟢 | Hinton et al. 2015, arXiv:1503.02531 |
| **Reverse-KL** | Mode-seeking: student concentrates on teacher's dominant modes → preferred for generative LLMs. | `KL(p_S ‖ p_T)` | LLM distillation | win-rate, acc | 🟡 | MiniLLM: Gu et al. 2023, arXiv:2306.08543 |
| **GKD (generalized/on-policy KD)** | Train on **student-generated** sequences with JSD/RKL → fixes train/inference mismatch. | divergence on student rollouts | Reasoning/instruction distillation | pass@1, win-rate | 🔵 | Agarwal et al. 2024, arXiv:2306.13649 |
| **On-Policy Distillation (reverse-KL on rollouts)** | Dense per-token teacher scoring of the student's own outputs; strong for continual learning without forgetting. | per-token RKL, teacher as regularizer | Specialist / continual distillation | task acc, retention | 🔵 (2025) | Thinking Machines Lab 2025 (thinkingmachines.ai/blog/on-policy-distillation) |
| **DistiLLM-2 (contrastive KD)** | Skew-KL contrastive objective: ↑ likelihood of teacher outputs while ↓ student outputs; blends off- + on-policy data. | contrastive skew-KL | LLM distillation | win-rate | 🔵 (2025) | Ko et al. 2025, arXiv:2503.07067 |

---

## Quick selection cheat-sheet

- **Detection, one lever:** keep DFL + swap CIoU → **SIoU / WIoU / MPDIoU** (noisy data → WIoU).
- **Keypoint / pose:** **Adaptive Wing** (heatmaps) or **OKS loss** (direct metric).
- **Medical/imbalanced seg:** **Dice/Tversky + boundary (or clDice)**; single pick → **Focal-Tversky**.
- **Face / retrieval embeddings:** **AdaFace** (varying quality) or **ArcFace** (clean).
- **VL / SSL pretraining:** **SigLIP 2** (contrastive) or **DINOv3** (label-free dense).
- **Generative images:** **rectified-flow / flow-matching**; add **LPIPS + GAN** for perceptual SR.
- **LLM alignment (offline):** **DPO** baseline; **SimPO/ORPO** for reference-free/cheaper; **KTO** for unpaired feedback.
- **LLM reasoning (online):** **GRPO**, upgrade to **DAPO/GSPO** at scale, **Dr. GRPO** to debias.
- **LLM compression:** **reverse-KL / on-policy distillation**; **DistiLLM-2** for the strongest 2025 recipe.

---

## References (arXiv / venue)

Focal Loss 1708.02002 · VarifocalNet 2008.13367 · Generalized Focal Loss (GFL/DFL) 2006.04388 · Poly Loss 2204.12511 · GIoU 1902.09630 · DIoU/CIoU 1911.08287 · Focal-EIoU 2101.08158 · SIoU 2205.12740 · WIoU 2301.10051 · MPDIoU 2307.07662 · Alpha-IoU 2110.13675 · Focaler-IoU (Zhang 2024) · MoEIoU 2606.00844 · DAPIoU (IEEE Access 2025, 10.1109/ACCESS.2025.3567767) · JOL tiny-object (Remote Sensing 17(8):1476, 2025) · DETR 2005.12872 · DINO-det 2203.03605 · Wing 1711.06753 · Adaptive Wing 1904.07399 · YOLO-Pose/OKS 2204.06806 · Dice/V-Net 1606.04797 · Generalized Dice 1707.03237 · Tversky 1706.05721 · Focal-Tversky 1810.07842 · Lovász-Softmax 1705.08790 · Boundary Loss 1812.07032 · clDice 2003.07311 · Unified Focal 2102.04525 · Seg-loss survey 2312.05391 · CosFace 1801.09414 · ArcFace 1801.07698 · MagFace 2103.06627 · ElasticFace 2109.09416 · AdaFace 2204.00964 · InfoNCE/CPC 1807.03748 · CLIP 2103.00020 · SigLIP 2303.15343 · SigLIP 2 2502.14786 · DINO 2104.14294 · DINOv2 2304.07193 · DINOv3 2508.10104 · iBOT 2111.07832 · DDPM 2006.11239 · Flow Matching 2210.02747 · Rectified Flow 2209.03003 · SD3 2403.03206 · LPIPS 1801.03924 · DISTS 2004.07728 · Label smoothing 1906.02629 · DPO 2305.18290 · IPO 2310.12036 · KTO 2402.01306 · ORPO 2403.07691 · SimPO 2405.14734 · CPO 2401.08417 · PPO 1707.06347 · GRPO/DeepSeekMath 2402.03300 · DeepSeek-R1 2501.12948 · Dr.GRPO 2503.20783 · DAPO 2503.14476 · GSPO 2507.18071 · E5 2212.03533 · SimCSE 2104.08821 · Matryoshka 2205.13147 · KD 1503.02531 · MiniLLM 2306.08543 · GKD 2306.13649 · DistiLLM-2 2503.07067 · On-Policy Distillation (Thinking Machines Lab, 2025)

*Compiled July 2026. arXiv IDs verified against source at compile time. IDs numbered ≥2506 (e.g. MoEIoU) are recent-cycle preprints; re-confirm before pinning a dependency. Non-arXiv items are cited by venue/DOI.*