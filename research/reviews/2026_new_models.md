# dl_techniques: Model Candidates

## Index

| # | Model | Family | Complexity | Priority | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | VGGT | `vision/vggt` | High | P1 | Proposed |
| 2 | CoAtNet | `vision/coatnet` | Medium | P0 | Proposed |
| 3 | MaxViT | `vision/maxvit` | Medium | P1 | Proposed |
| 4 | FastViT | `vision/fastvit` | n/a | n/a | Exists |
| 5 | EfficientFormer (V1, V2) | `vision/efficientformer` | Medium | P1 | Proposed |
| 6 | Next-ViT | `vision/next_vit` | Medium | P2 | Proposed |
| 7 | CPVT (PEG) | `layers/embedding` | Low | P0 | Proposed |
| 8 | PVT (V1, V2) | `vision/pvt` + `layers/attention` | Medium | P1 | Proposed |
| 9 | Perceiver (+ IO) | `general_purpose/perceiver` | Medium | P2 | Partial (layers exist) |
| 10 | MBConv / EfficientNet | `layers/conv_blocks` (+ `vision/efficientnet`) | Low | P0 | Partial |
| 11 | ConvNeXt | `vision/convnext` | n/a | n/a | Exists |
| 12 | HierarchicalCliffordNet | `vision/cliffordnet` | Medium | P0 | Partial (isotropic exists) |
| 13 | NVIDIA Cosmos (Predict 2.5, Tokenizer, Cosmos 3) | `vision_language/cosmos` | High | P1 | Proposed |
| 14 | LFM2.5-VL (+ LFM2 backbone) | `language/lfm2` + `vision_language/lfm2_vl` | Medium | P1 | Proposed |
| 15 | NVIDIA Lyra 2.0 | `vision_language/lyra` | High | P2 | Proposed |
| 16 | MoGe v1, v2, v3 | `vision/moge` | Medium (v1, v2), High (v3) | P1 (v1, v2), P2 (v3) | Proposed |

Context for 2 to 12: hybrid conv-early / attention-late design study with a Clifford stage, and the baselines for its ablation ladder.

---

## 1. VGGT: Visual Geometry Grounded Transformer

| Field | Content |
| --- | --- |
| Name | VGGT (VGGT-1B, VGGT-1B-Commercial); follow-up: VGGT-Omega |
| Paper | Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny. *VGGT: Visual Geometry Grounded Transformer*. CVPR 2025 (Best Paper). [arXiv:2503.11651](https://arxiv.org/abs/2503.11651) |
| Project | [https://vgg-t.github.io/](https://vgg-t.github.io/) |
| Reference impl | [https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt) (PyTorch) |
| License | Code: commercial use permitted, military excluded. Weights: `VGGT-1B` non-commercial; `VGGT-1B-Commercial` gated, commercial |
| Family | `src/dl_techniques/models/vision/vggt/` (alt: `point_cloud/`) |
| Complexity | High |
| Priority | P1 |

### Summary

Feed-forward multi-view 3D reconstruction. From 1 to hundreds of images, predicts in one pass: camera extrinsics + intrinsics, depth maps, point maps, and dense features for point tracking. No test-time global optimization (bundle adjustment optional).

### Architecture

| Component | Details |
| --- | --- |
| Patch embed | DINOv2 ViT-L/14 patchifier, 518 px input |
| Per-frame tokens | Patch tokens + 1 camera token + 4 register tokens |
| Reference frame | Frame 1 uses separate learnable camera/register tokens; all outputs expressed in frame 1 coordinates |
| Backbone ("Aggregator") | 24 blocks of Alternating Attention: frame-wise self-attn (within each image) then global self-attn (across all frames). dim 1024, 16 heads |
| Block details | 2D RoPE, QK-norm, LayerScale (init 0.01), no cross-attention |
| Camera head | 4 self-attn layers on camera tokens, predicts 9-D pose encoding: translation (3), quaternion (4), FoV (2) |
| DPT head | Multi-scale features from intermediate blocks (frame + global outputs concatenated, 2048-D); outputs depth + confidence, point map + confidence |
| Track head | CoTracker2-style tracker on DPT feature maps, query points from frame 1 |
| Params | \~1.2B |

### Training

| Item | Details |
| --- | --- |
| Losses | Camera: Huber on pose encoding. Depth / point map: aleatoric-uncertainty weighted L1 + gradient term, minus log-confidence regularizer. Tracking: L1 on tracks + BCE visibility |
| Normalization | Ground truth only: scene scaled by mean point distance to origin in frame-1 coordinates; predictions not normalized |
| Data | Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic Environments, Aria Digital Twin |
| Compute | 64 A100, \~9 days, 160K iterations, 2 to 24 frames per scene |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| `models/vision/dino` (DINOv2) | Patch embed + weight loader (`safetensors` reader) |
| `models/vision/depth_anything` | DPT head |
| `layers/embedding` (RoPE registry) | 2D RoPE |
| `layers/norms` | QK-norm, LayerNorm |
| `models/vision/keypoints` (SuperPoint, LightGlue) | Track head comparisons / eval baselines |
| `models/point_cloud/latent_gmm_registration` | Point map post-processing, registration eval |

### New components needed

| Component | Target |
| --- | --- |
| `AlternatingAttentionBlock` (frame / global) | `layers/transformers/` |
| Camera + register token handling with first-frame distinct tokens | `models/vision/vggt/` |
| Camera head + 9-D pose encoding \<-> extrinsics/intrinsics utils | `models/vision/vggt/` + `utils/geometry` |
| Multi-output DPT head (depth, point map, confidence) | extend `depth_anything` DPT or `layers/heads/vision` |
| Confidence-weighted regression loss (aleatoric + gradient) | `losses/` |
| Pose Huber loss, track loss | `losses/` |
| Track head (CoTracker2-style correlation + iterative refinement) | `models/vision/vggt/track_head.py` (phase 2) |
| PyTorch checkpoint converter + numerical parity oracle (`reference.py`) | `models/vision/vggt/` |
| Metrics: AUC@30 pose, Chamfer / accuracy / completeness | `metrics/` |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | Aggregator + camera head + depth DPT head, configurable small variants for tests |
| 2 | Point map head, confidence losses, weight conversion from `VGGT-1B-Commercial`, parity check vs PyTorch |
| 3 | Track head, trainer `src/train/vggt/`, Co3D eval |
| 4 (optional) | VGGT-Omega variant |

### Risks / notes

| Item | Note |
| --- | --- |
| Memory | Global attention over all frames' tokens is O((N x P)^2); needs chunking or flash attention path for many frames |
| Training cost | Full reproduction infeasible; value is in faithful port + pretrained weights + small-scale trainers |
| Weights | Non-commercial `VGGT-1B` vs gated commercial checkpoint; document clearly |
| VGGT-Omega | Released 2026; authors flagged possible benchmark contamination in an ancestor checkpoint of the released 1B model; weights under FAIR Noncommercial Research License |

---

## 2. CoAtNet

| Field | Content |
| --- | --- |
| Paper | Dai, Liu, Le, Tan. *CoAtNet: Marrying Convolution and Attention for All Data Sizes*. NeurIPS 2021. [arXiv:2106.04803](https://arxiv.org/abs/2106.04803) |
| Reference impl | No standalone official repo; timm `coatnet.py` / `maxxvit.py` (Apache-2.0) |
| Family | `src/dl_techniques/models/vision/coatnet/` |
| Complexity | Medium |
| Priority | P0 |

**Architecture.** 5 stages S0 to S4: conv stem, then C-C-T-T. S1/S2 use MBConv (expansion 4, SE), S3/S4 use pre-norm relative-attention Transformer blocks. Downsampling: stride-2 in MBConv, max-pool shortcut + strided conv in Transformer stages. Relative attention adds a translation-invariant 2D bias `w[i-j]` to logits before softmax (pre-normalization). Variants CoAtNet-0 to 7.

| Existing | Reuse |
| --- | --- |
| `mobilenet_v2`, `conv_blocks/universal_inverted_bottleneck.py`, `conv_blocks/squeeze_excitation.py` | MBConv stages |
| `attention/window_attention.py` (Swin relative bias) | Relative position bias table logic |
| `layers/transformers/` | Pre-norm block |

| New | Target |
| --- | --- |
| `RelativeAttention2D` (global, input-independent bias, pre-softmax) | `layers/attention/` + registry |
| Transformer-stage downsample (pool shortcut) | `layers/pooling/` |
| Stage-type config string (`"CCTT"`) for ablations | `models/vision/coatnet/` |

**Notes.** Best single template for the ablation ladder: stage-type string lets C/T/Clifford be swapped per stage.

---

## 3. MaxViT

| Field | Content |
| --- | --- |
| Paper | Tu, Talebi, Zhang, Yang, Milanfar, Bovik, Li. *MaxViT: Multi-Axis Vision Transformer*. ECCV 2022. [arXiv:2204.01697](https://arxiv.org/abs/2204.01697) |
| Reference impl | google-research/maxvit (TF/JAX); timm `maxxvit.py` |
| Family | `src/dl_techniques/models/vision/maxvit/` |
| Complexity | Medium |
| Priority | P1 |

**Architecture.** Conv stem, 4 stages of MaxViT blocks. Each block: MBConv (SE) -> block attention (non-overlapping P x P windows, local) -> grid attention (G x G grid, sparse dilated global). Each attention followed by FFN, relative position bias. Linear complexity. Variants T/S/B/L/XL.

| Existing | Reuse |
| --- | --- |
| `attention/window_attention.py` | Block attention |
| MBConv sources (see 10) | MBConv |

| New | Target |
| --- | --- |
| `grid_partition` / `grid_reverse` ops | `layers/attention/` utils |
| `MultiAxisAttention` (block + grid) | `layers/attention/` + registry |
| `MaxViTBlock` | `layers/transformers/` |

**Notes.** Grid attention = cheapest global mixing at fixed cost; strong comparison point against a Clifford global-context stage. Shares relative bias with CoAtNet (MaxViT authors are partly the same group).

---

## 4. FastViT

**Status: exists.** `models/vision/fastvit/`, `layers/fastvit/` (RepMixer, RepCPE, reparam large-kernel conv, committed `reference.py`), `conv_blocks/repmixer_block.py`. Used by MobileCLIP v2.

| Gap check | Action |
| --- | --- |
| Inference-time reparameterization export | Verify present and tested |
| Variant coverage (T8, T12, S12, SA12/24/36, MA36) | Verify |
| Reuse RepCPE as CPE option for hybrid study | Register in embedding factory (see 7) |

---

## 5. EfficientFormer (V1, V2)

| Field | Content |
| --- | --- |
| Paper | V1: Li, Yuan, Wen, Hu, Evangelidis, Tulyakov, Wang, Ren. *EfficientFormer: Vision Transformers at MobileNet Speed*. NeurIPS 2022. [arXiv:2206.01191](https://arxiv.org/abs/2206.01191). V2: *Rethinking Vision Transformers for MobileNet Size and Speed*. ICCV 2023. [arXiv:2212.08059](https://arxiv.org/abs/2212.08059) |
| Reference impl | snap-research/EfficientFormer (verify license) |
| Family | `src/dl_techniques/models/vision/efficientformer/` |
| Complexity | Medium |
| Priority | P1 |

**Architecture.** V1: dimension-consistent design. Stages 1 to 3 use Meta4D blocks (4D tensors, pooling token mixer + conv MLP with BN), last stage switches to Meta3D (flattened tokens, MHSA + linear MLP, LN). Latency-driven slimming search. V2: unified FFN with depthwise conv, local-window / stride attention in stage 3, attention with talking heads, dual-path downsampling.

| Existing | Reuse |
| --- | --- |
| `models/vision/metaformer/` | README only, no model code; EfficientFormer is a natural first MetaFormer instance |
| `layers/pooling/` | Pooling token mixer base |

| New | Target |
| --- | --- |
| `PoolMixer` (avgpool(x) - x) | `layers/` token mixers |
| `Meta4DBlock`, `Meta3DBlock`, 4D->3D transition | `layers/transformers/` |
| V2 attention with talking heads + stride attention | `layers/attention/` |

**Notes.** Doubles as the MetaFormer / PoolFormer implementation; fills the empty `metaformer` package.

---

## 6. Next-ViT

| Field | Content |
| --- | --- |
| Paper | Li, Xia, Li, Xiao, Wang, Wu, Wang, Zheng, Pan. *Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios*. 2022. [arXiv:2207.05501](https://arxiv.org/abs/2207.05501) |
| Reference impl | bytedance/Next-ViT (verify license) |
| Family | `src/dl_techniques/models/vision/next_vit/` |
| Complexity | Medium |
| Priority | P2 |

**Architecture.** Next Convolution Block (NCB): Multi-Head Convolutional Attention (MHCA, grouped 3x3 conv) + MLP. Next Transformer Block (NTB): Efficient MHSA (spatial-reduced K/V) and MHCA branches, outputs concatenated for high/low-frequency fusion. Next Hybrid Strategy (NHS): each stage is (NCB x N + NTB x 1) x L. Variants S/B/L.

| Existing | Reuse |
| --- | --- |
| SRA from PVT (see 8) | E-MHSA |

| New | Target |
| --- | --- |
| `MultiHeadConvAttention` | `layers/attention/` |
| `NextConvBlock`, `NextTransformerBlock` | `layers/transformers/` |

**Notes.** Low priority once CoAtNet + MaxViT + PVT exist; main value is the per-stage interleaving pattern (NHS) as an ablation variant.

---

## 7. CPVT (Conditional Positional Encoding)

| Field | Content |
| --- | --- |
| Paper | Chu, Tian, Zhang, Wang, Shen. *Conditional Positional Encodings for Vision Transformers*. ICLR 2023. [arXiv:2102.10882](https://arxiv.org/abs/2102.10882) |
| Reference impl | Meituan-AutoML/CPVT (verify license); timm `PosConv` |
| Family | `src/dl_techniques/layers/embedding/` (layer, not a model package) |
| Complexity | Low |
| Priority | P0 |

**Architecture.** Positional Encoding Generator (PEG): reshape tokens to 2D, depthwise 3x3 conv with zero padding, add residual, flatten. Inserted after first encoder block. Zero padding leaks absolute position; supports variable input resolution. CPVT-GAP replaces class token with global average pooling.

| Existing | Reuse |
| --- | --- |
| `layers/fastvit/rep_conditional_pos_enc.py` (RepCPE) | Reparameterizable variant already present |

| New | Target |
| --- | --- |
| `PositionalEncodingGenerator` | `layers/embedding/` + `EMBEDDING_REGISTRY` |
| Optional `cpvt` variant in `models/vision/vit` | Config flag |

**Notes.** Needed to test the claim that a Clifford stage (local shifted interactions with padding) supplies implicit position, removing explicit position embeddings.

---

## 8. PVT (V1, V2)

| Field | Content |
| --- | --- |
| Paper | V1: Wang, Xie, Li, Fan, Song, Liang, Lu, Luo, Shao. *Pyramid Vision Transformer*. ICCV 2021. [arXiv:2102.12122](https://arxiv.org/abs/2102.12122). V2: *PVT v2: Improved Baselines with Pyramid Vision Transformer*. CVMJ 2022. [arXiv:2106.13797](https://arxiv.org/abs/2106.13797) |
| Reference impl | whai362/PVT (verify license) |
| Family | `src/dl_techniques/models/vision/pvt/` |
| Complexity | Medium |
| Priority | P1 |

**Architecture.** 4-stage pyramid (strides 4/8/16/32) of plain transformer blocks. Spatial-Reduction Attention (SRA): K and V downsampled by ratio R_i (strided conv + LN) before attention. V2 adds overlapping patch embedding, conv FFN (depthwise 3x3 in MLP, implicit position), and linear SRA (adaptive avg-pool K/V to fixed 7x7).

| New | Target |
| --- | --- |
| `SpatialReductionAttention` (conv and linear modes) | `layers/attention/` + registry |
| `OverlapPatchEmbed` | `layers/embedding/` or `layers/` stems |
| `ConvFFN` (depthwise conv inside MLP) | `layers/ffn/` + `FFN_REGISTRY` |

**Notes.** SRA is reused by Next-ViT and is the cheap-global-attention baseline for late stages.

---

## 9. Perceiver (+ Perceiver IO)

| Field | Content |
| --- | --- |
| Paper | Jaegle, Gimeno, Brock, Zisserman, Vinyals, Carreira. *Perceiver: General Perception with Iterative Attention*. ICML 2021. [arXiv:2103.03206](https://arxiv.org/abs/2103.03206). IO: ICLR 2022. [arXiv:2107.14795](https://arxiv.org/abs/2107.14795) |
| Reference impl | google-deepmind/deepmind-research `perceiver/` (JAX, Apache-2.0) |
| Family | `src/dl_techniques/models/general_purpose/perceiver/` |
| Complexity | Medium |
| Priority | P2 |

**Architecture.** Learned latent array (N \<\< M) cross-attends to flattened input with Fourier position features, followed by latent self-attention tower; repeated with optional weight sharing. Cost O(MN) + O(N^2). IO adds query-based cross-attention decoder for arbitrary outputs.

| Existing | Reuse |
| --- | --- |
| `layers/attention/perceiver_attention.py`, `layers/transformers/perceiver_transformer.py` | Cross-attention + block |

| New | Target |
| --- | --- |
| Fourier position feature encoder | `layers/embedding/` |
| Full Perceiver model with weight-shared repeats | `models/general_purpose/perceiver/` |
| IO decoder (output queries) | same package |

**Notes.** For the hybrid study: a Perceiver-style latent bottleneck is an alternative late stage to full attention on the final feature map.

---

## 10. MBConv / EfficientNet

| Field | Content |
| --- | --- |
| Paper | Sandler, Howard, Zhu, Zhmoginov, Chen. *MobileNetV2*. CVPR 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381). Tan, Le. *EfficientNet*. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) |
| Family | `src/dl_techniques/layers/conv_blocks/` (+ optional `models/vision/efficientnet/`) |
| Complexity | Low |
| Priority | P0 |
| Ablation role | Model C early stage |

**Status: partial.** Inverted residual exists inside `mobilenet_v2`; `universal_inverted_bottleneck.py` (MobileNetV4 UIB) and `squeeze_excitation.py` exist.

| New | Target |
| --- | --- |
| Standalone `MBConvBlock`: expand 1x1 -> DW kxk -> SE -> project 1x1, stochastic depth, configurable norm/activation | `layers/conv_blocks/` |
| Fused-MBConv (EfficientNetV2) | same |
| EfficientNet B0 to B7 compound scaling (optional) | `models/vision/efficientnet/` |

**Notes.** One shared block consumed by CoAtNet, MaxViT and the ablation ladder; verify UIB can express MBConv before writing a new class.

---

## 11. ConvNeXt

**Status: exists.** `models/vision/convnext/` (V1, V2), `conv_blocks/convnext_v1_block.py`, `convnext_v2_block.py`, trainers in `src/train/convnext/`.

| Gap check | Action |
| --- | --- |
| Stage builder usable standalone (to plug ConvNeXt stages into a hybrid) | Expose `build_stage` helper if missing |
| Ablation role | Alternative early stage for model C |

---

## 12. HierarchicalCliffordNet

| Field | Content |
| --- | --- |
| Paper | Ji. *CliffordNet: All You Need is Geometric Algebra*. 2026. [arXiv:2601.06793](https://arxiv.org/abs/2601.06793) |
| Reference impl | ParaMind2025/CAN (MIT) |
| Family | `src/dl_techniques/models/vision/cliffordnet/` |
| Complexity | Medium |
| Priority | P0 |
| Ablation role | Model B: hierarchy without attention |

**Status: partial.** `models/vision/cliffordnet/model.py` is isotropic (constant width, no stages); `layers/geometric/clifford_block.py` exists.

| New | Target |
| --- | --- |
| `HierarchicalCliffordNet`: 4 stages, per-stage width/depth, downsampling between stages | `models/vision/cliffordnet/hierarchical.py` |
| Downsample layer options (strided conv, patch merging, blur pool) | reuse `layers/pooling/patch_merging.py`, `blur_pool.py` |
| Per-stage `shifts` scaling with resolution | config |
| Multi-scale feature outputs for dense heads | model API |

**Notes.** Confirm the reference repo's hierarchical config (stage widths, shift sets, downsample op) before porting; paper's main results are isotropic.

---

## Ablation ladder (hybrid study)

| Model | Early stages | Late stages | Source entries |
| --- | --- | --- | --- |
| A | CliffordNet isotropic | CliffordNet isotropic | existing |
| B | HierarchicalCliffordNet | HierarchicalCliffordNet | 12 |
| C | MBConv or ConvNeXt | Transformer | 10, 11, 2 |
| D | Clifford | Transformer (full / SRA / grid / Perceiver latent) | 12, 2, 8, 3, 9 |
| E | Clifford, no explicit pos-emb vs PEG | Transformer | 12, 7 |

## Shared components build order

| Order | Component | Unblocks |
| --- | --- | --- |
| 1 | `MBConvBlock` | CoAtNet, MaxViT, model C |
| 2 | `PositionalEncodingGenerator` | CPVT, ablation E |
| 3 | `RelativeAttention2D` | CoAtNet, MaxViT |
| 4 | `SpatialReductionAttention`, `ConvFFN`, `OverlapPatchEmbed` | PVT, Next-ViT |
| 5 | Grid partition + `MultiAxisAttention` | MaxViT |
| 6 | `PoolMixer`, Meta4D/Meta3D | EfficientFormer, MetaFormer |

---

## 13. NVIDIA Cosmos World Foundation Models

| Field | Content |
| --- | --- |
| Name | Cosmos family: Tokenizer, Predict1, Predict2.5 (2B, 14B), Transfer2.5, Cosmos 3 (Edge 4B, Nano 16B, Super 64B) |
| Papers | *Cosmos World Foundation Model Platform for Physical AI*. 2025. [arXiv:2501.03575](https://arxiv.org/abs/2501.03575). *World Simulation with Video Foundation Models for Physical AI* (Predict2.5 / Transfer2.5). 2025. [arXiv:2511.00062](https://arxiv.org/abs/2511.00062). Cosmos 3 technical report (June 2026): [PDF](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) |
| Reference impl | [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5), [NVIDIA/cosmos](https://github.com/nvidia/Cosmos) (Cosmos 3), Diffusers `Cosmos3OmniPipeline` |
| License | Predict2.5 / Transfer2.5: NVIDIA Open Model License. Cosmos 3: Linux Foundation OpenMDW-1.1 |
| Family | `src/dl_techniques/models/vision_language/cosmos/` (next to `sd3_mmdit`, `dit`) |
| Complexity | High |
| Priority | P1 |

### Summary

Video world models for physical AI (robotics, driving, smart spaces): predict future world states as video from text, image, video, camera or action inputs. Main value for the repo: a clean latent video DiT with flow matching, action conditioning, and (Cosmos 3) a two-tower Mixture-of-Transformers unifying AR reasoning and diffusion generation.

### Generations

| Version | Core design |
| --- | --- |
| Tokenizer (2025) | Causal video tokenizer, continuous (CV) and discrete (DV, FSQ) variants, 2-level Haar wavelet input, factorized spatio-temporal causal conv + attention; compression 4x8x8 / 8x8x8 / 8x16x16 (verify against paper) |
| Predict1 | Two tracks: diffusion DiT (EDM, 7B/14B, T5 text encoder) and autoregressive (4B/12B on discrete tokens) |
| Predict2.5 | Single flow-matching DiT unifying Text2World, Image2World, Video2World; Cosmos-Reason1 as text encoder; Wan2.1 VAE as tokenizer |
| Transfer2.5 | ControlNet-style branch on Predict2.5 (edge, blur, segmentation, depth), 4 control blocks, one after every 7 main blocks |
| Cosmos 3 | Omnimodal two-tower Mixture-of-Transformers: AR reasoner tower + diffusion generator tower, separate per-layer parameters, joint attention; text, image, video, audio, action in and out |

### Predict2.5 architecture (primary port target)

| Component | Details |
| --- | --- |
| Tokenizer | Wan2.1 causal VAE, 4x8x8 (T, H, W) compression; 93 frames -> 24 latent frames at 16 fps |
| Patchify | 1x2x2 on latents |
| Block | Self-attn -> cross-attn (text) -> MLP, each modulated by AdaLN (scale, shift, gate) from timestep |
| AdaLN | AdaLN-LoRA, rank 256 |
| Position | 3D-factorized RoPE only; absolute position embedding removed for resolution / length generalization |
| Config 2B | 32 layers, dim 2048, FFN 8192, 16 heads, head dim 128, GELU |
| Config 14B | 36 layers, dim 5120, FFN 20480, 40 heads, head dim 128, GELU |
| Text encoder | Cosmos-Reason1 activations concatenated across multiple blocks, projected to 1024-D, fed via cross-attention |
| Conditioning frames | Frame replacement: GT latent frames concatenated with noisy frames along time, plus binary mask channel; loss only on generated frames |
| Action conditioning | Action-embedder MLP added to timestep embedding (beats cross-attn and channel concat in their ablation) |
| Camera conditioning | Plucker raymaps, patchified, projected and added to video tokens before self-attn |
| Multi-view | Views concatenated on latent time axis, per-view learned embedding in channels, RoPE built per view |

### Training (Predict2.5)

| Item | Details |
| --- | --- |
| Objective | Flow matching, velocity target v = eps - x, MSE |
| Timesteps | Shifted logit-normal, shift 1 at 256p rising to 5 at 720p; 5% of samples from top 2% noise |
| Curriculum | Text2Image 256p -> + Video2World -> 480p -> 720p -> + Text2World (0/1/2 cond frames at 0.5/0.25/0.25) |
| Optimizer | AdamW (0.9, 0.999), LR 3e-5 (2B) / 1.3e-5 (14B), WD 1e-3, 2000-step warmup, linear decay |
| Post-training | Domain SFT, model merging (model soup), GRPO-style RL with VideoAlign reward, rCM 4-step distillation |
| Data | 200M curated clips |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| `models/vision/dit`, `models/vision_language/sd3_mmdit` | DiT skeleton, cross-attn conditioning |
| `layers/transformers/adaln_zero.py`, `sd3_adaln.py` | AdaLN modulation base |
| `losses/flow_matching_velocity_loss.py` | FM velocity loss |
| `layers/embedding/video_rope.py`, `multi_axis_rope.py` | 3D-factorized RoPE |
| `layers/signal_processing/haar_wavelet_decomposition.py` | Tokenizer wavelet stem |
| `layers/adapters/lora.py` | LoRA post-training |
| `callbacks/ema_shadow_callback.py` | EMA weights |
| `models/vision/vae`, `vq_vae` | VAE / quantizer base |
| `models/world_models/video_jepa`, `levjepa`, `lewm` | Latent-space world model counterparts for comparison |
| `models/language/qwen` | Small decoder-only stand-in for Cosmos-Reason1 text encoder |

### New components needed

| Component | Target |
| --- | --- |
| `CausalConv3D` + causal 3D VAE (Wan2.1-style) | `layers/` + `models/vision/` tokenizer package |
| FSQ quantizer (for Cosmos DV tokenizer) | `layers/` quantizers |
| `AdaLNLoRA` modulation | `layers/transformers/` |
| `CosmosDiTBlock` (self-attn, cross-attn, MLP, AdaLN-LoRA, QK-norm) | `layers/transformers/` |
| Multi-layer text-feature aggregation projector | `models/vision_language/cosmos/` |
| Frame-replacement conditioning + mask channel + masked FM loss | model + `losses/` |
| Shifted logit-normal timestep sampler | `models/vision_language/cosmos/` or shared diffusion utils |
| Action embedder (added to timestep embedding) | `models/vision_language/cosmos/` |
| Plucker raymap encoder | `utils/geometry` + `layers/embedding/` |
| ControlNet branch (Transfer2.5) | phase 3 |
| Mixture-of-Transformers layer (per-modality params, joint attention) | `layers/transformers/` (Cosmos 3, phase 4) |
| Metrics: FVD, Sampson error (multi-view consistency) | `metrics/` |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | Causal 3D VAE tokenizer + `CosmosDiT` at small scale; flow matching Text/Image/Video2World on a small video dataset (`bdd100k_video`, `synthetic_drone_video` loaders exist) |
| 2 | Action-conditioned variant (Bridge-style 7-D actions), camera / Plucker conditioning |
| 3 | Transfer2.5 ControlNet branch; optional weight import from Predict2.5-2B with parity check |
| 4 | Cosmos 3 MoT block as standalone layer + toy AR+DM model |

### Risks / notes

| Item | Note |
| --- | --- |
| Scale | 2B to 64B params, 200M clips; full reproduction infeasible. Port architecture with small configs; pretrained import only for 2B |
| Text encoder | Cosmos-Reason1 is a large VLM; use a small encoder stand-in for in-repo training |
| Sequence length | 720p x 93 frames gives very long token sequences; needs windowed / sparse attention or low-res configs |
| Dependencies | Predict2.5 depends on external Wan2.1 VAE; implementing it in Keras is a prerequisite sub-project |
| Licensing | Two different licenses across generations; document per checkpoint |
| Relation to VGGT (#1) | Complementary: VGGT reconstructs geometry from views, Cosmos generates views; Plucker raymap and camera utils shareable |

---

## 14. LFM2.5-VL (Liquid AI)

| Field | Content |
| --- | --- |
| Name | LFM2.5-VL-450M, LFM2.5-VL-1.6B, LFM2.5-VL-3B; predecessor LFM2-VL; sibling LFM2.5-Audio-1.5B |
| Paper | Liquid AI. *LFM2 Technical Report*. 2025. [arXiv:2511.23404](https://arxiv.org/abs/2511.23404). Blogs: [LFM2-VL](https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models), [LFM2.5-VL-3B](https://www.liquid.ai/blog/lfm2-5-vl-3b) |
| Reference impl | Hugging Face `transformers` (`Lfm2`, `Lfm2Vl`), weights at [huggingface.co/LiquidAI](https://huggingface.co/LiquidAI) |
| License | LFM Open License v1.0 (Apache-2.0 based): free for research and for commercial use under $10M revenue; above that requires a commercial license |
| Family | Backbone: `src/dl_techniques/models/language/lfm2/`. VLM: `src/dl_techniques/models/vision_language/lfm2_vl/` |
| Complexity | Medium |
| Priority | P1 |

### Summary

Edge-oriented VLM family. Three parts: LFM2.5 hybrid language backbone (gated short convolutions + a few GQA layers), SigLIP2 NaFlex vision encoder at native resolution, and a pixel-unshuffle 2-layer MLP projector. Image token budget is user-tunable at inference without retraining.

### Variants

| Model | LM backbone | Vision encoder |
| --- | --- | --- |
| LFM2.5-VL-450M | LFM2.5-350M | SigLIP2 NaFlex base, 86M |
| LFM2.5-VL-1.6B | LFM2.5-1.2B-Base | SigLIP2 NaFlex shape-optimized, 400M |
| LFM2.5-VL-3B | LFM2.5-2.6B | SigLIP2 NaFlex shape-optimized, 400M |

### LFM2 / LFM2.5 language backbone

| Component | Details |
| --- | --- |
| Layout (1.2B) | 16 layers: 10 double-gated short-conv (LIV) blocks + 6 GQA blocks |
| Short-conv block | Linear in-proj to (B, C, x); `y = C * causal_dwconv1d(B * x)`, kernel 3; out-proj. Input-dependent gates, tiny KV-free state cache |
| Attention block | GQA, RoPE, QK-norm (RMSNorm on q and k) |
| FFN / norm | SwiGLU, pre-RMSNorm |
| Context / vocab | 32K context, 65,536 vocab (1.2B) |
| Training (LFM2.5-1.2B) | 28T tokens pretraining (up from 10T in LFM2), scaled RL post-training |
| Design origin | Hardware-in-the-loop architecture search targeting CPU/NPU latency and memory |

### Vision path

| Component | Details |
| --- | --- |
| Encoder | SigLIP2 NaFlex: native resolution and aspect ratio up to 512x512, no upscaling |
| Large images | Split into non-overlapping 512x512 tiles + downscaled thumbnail for global context (1.6B and up); special tokens mark tile positions and thumbnail start |
| Projector | Pixel unshuffle (token reduction) + 2-layer MLP |
| Token budget | Example (LFM2-VL): 256x384 -> 96 tokens, 384x680 -> 240, 1000x3000 -> 1,020; max image tokens and tile count tunable at inference |
| Training (LFM2-VL) | Joint mid-training with text:image ratio annealed 95% -> 30%, then joint SFT; \~100B multimodal tokens |
| LFM2.5-VL-3B additions | Grounding, screen understanding, function calling, multi-image; non-reasoning model |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| `layers/attention/group_query_attention.py` | GQA blocks |
| `layers/embedding/rotary_position_embedding.py` | RoPE |
| `layers/norms` (RMSNorm), `layers/ffn` (SwiGLU) | Block internals |
| `layers/pooling/pixel_unshuffle.py` | Projector token reduction |
| `models/vision_language/nano_vlm`, `fastvlm` | VLM assembly, image-token splicing, `nano_vlm_loss` |
| `models/vision/vit_siglip`, `losses/siglip_contrastive_loss.py` | Encoder base (note: `vit_siglip` is not SigLIP2) |
| `models/language/qwen` (Qwen3Next), `zamba2`, `mamba` | Hybrid-backbone comparisons |
| `layers/adapters/lora.py` | Fine-tuning (vendor recommends task fine-tuning) |

### New components needed

| Component | Target |
| --- | --- |
| `GatedShortConv` (double-gated causal depthwise conv1d, kernel 3, with step-wise state cache) | `layers/` sequence mixers; register alongside attention |
| `LFM2Block` with per-layer type schedule (conv / attn) | `layers/transformers/` |
| `LFM2` causal LM + `create_lfm2(variant)` | `models/language/lfm2/` |
| SigLIP2 NaFlex encoder (variable-resolution patching, aspect-preserving resize, padding mask) | `models/vision/siglip2/` |
| Tiling + thumbnail image processor with special tokens | `models/vision_language/lfm2_vl/` |
| `PixelUnshuffleMLPProjector` | `layers/heads/vlm` or package-local |
| HF checkpoint converter + parity oracle | both packages |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | `GatedShortConv` + `LFM2` text backbone (350M config), HF weight import, logits parity |
| 2 | SigLIP2 NaFlex encoder (86M), weight import, parity |
| 3 | `LFM2VL` assembly: tiling, thumbnail, projector, chat template; 450M end-to-end parity |
| 4 | 1.6B / 3B configs; optional LFM2.5-Audio as separate entry |

### Risks / notes

| Item | Note |
| --- | --- |
| Licensing | Revenue-capped commercial terms; document per checkpoint |
| Version details | LFM2.5-2.6B layer split and LFM2.5 vocab size changes: verify against HF configs before porting |
| NaFlex | Variable-length patch sequences need masking support in Keras attention path |
| Value beyond the VLM | `GatedShortConv` is a cheap local mixer usable in the hybrid study (entries 2 to 12) as a 1D analogue of conv-early / attention-late |

---

## 15. NVIDIA Lyra 2.0: Explorable Generative 3D Worlds

| Field | Content |
| --- | --- |
| Name | Lyra 2.0 (full, 35-step) and DMD-distilled (4-step); predecessor Lyra 1.0 (ICLR 2026) |
| Paper | Shen, Bahmani, He, et al. (NVIDIA Spatial Intelligence Lab). *Lyra 2.0: Explorable Generative 3D Worlds*. April 2026. [arXiv:2604.13036](https://arxiv.org/abs/2604.13036) |
| Project | [https://research.nvidia.com/labs/sil/lyra2/](https://research.nvidia.com/labs/sil/lyra2/) |
| Reference impl | GitHub release + [nvidia/Lyra-2.0](https://huggingface.co/nvidia/Lyra-2.0) on Hugging Face |
| License | Code: Apache-2.0 ([nv-tlabs/lyra](https://github.com/nv-tlabs/lyra)). Weights: separate NVIDIA model license per model card; mirrored card text restricts production use. Check `nvidia/Lyra-2.0` card before use |
| Family | `src/dl_techniques/models/vision_language/lyra/` (or a new `world_model/` family grouping 13 and 15) |
| Complexity | High |
| Priority | P2 |

### Summary

Single image -> long camera-controlled walkthrough video -> feed-forward 3D Gaussians + mesh. Addresses two failure modes of long-horizon autoregressive video generation: spatial forgetting (revisited regions hallucinated) and temporal drifting (error accumulation). Key idea: use noisy per-frame 3D geometry only for information routing (which history frames to retrieve, where they correspond), leave appearance to the video prior.

### Pipeline

| Stage | Details |
| --- | --- |
| Loop | Retrieve -> generate -> update, per user-defined camera trajectory segment + optional text |
| Backbone | Wan 2.1-14B DiT, flow matching, Wan 2.1 VAE (8x spatial, 4x temporal, C=16), 832x480 |
| 3D cache | Per-frame full-res depth + camera, plus downsampled world-space point cloud; never fused into one global cloud |
| Retrieval | Project cached point clouds into target view, z-test visibility, greedy max-coverage selection of N_s = 5 frames (training: sample proportional to visibility) |
| Context layout | Anchor `f1k1` + spatial slots `f4k2 f1k1` + FramePack temporal slots `f16k4 f2k2 f1k1` + generate `g20` (fn km: n frames at spatial subsampling m) |
| Correspondence | Canonical coordinate map (u, v, slot id) per retrieved frame, forward-warped to target with depth, + warped depth = 4 channels; sinusoidal PE -> pixel shuffle -> linear; added to Q and K (not V) in every block |
| Camera control | Depth-warped latest frame concatenated on channels + Plucker rays (6-D) added to Q and K |
| Anti-drift | Self-augmentation: with p_aug, noise history latent at t \~ U(0, 0.5), one-step denoise with the model, condition on that; target still from clean history. One extra forward pass |
| Distillation | DMD, 35 -> 4 steps, CFG distilled in, \~13x faster per step |
| 3D lift | Depth Anything V3 feed-forward 3DGS, Gaussian DPT head downsampled by k x k, fine-tuned on generated scenes |
| Mesh | Median-depth oriented points -> SDF on hierarchical sparse grid (OpenVDB), marching cubes |
| Data | DL3DV (10K clips); poses via ViPE, depth via DAv3, captions via Qwen3-VL-8B |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| Entry 13 (Cosmos) plan: causal 3D VAE, latent video DiT, flow matching, Plucker raymap encoder | Same backbone stack (Wan 2.1 VAE is shared by both) |
| Entry 1 (VGGT) | Camera utils, depth / point-map heads; alternative geometry estimator for the 3D cache |
| `models/vision/depth_anything` | Base for the DAv3-style Gaussian DPT head |
| `layers/pooling/pixel_shuffle.py`, `pixel_unshuffle.py` | Conditioning projections |
| `losses/flow_matching_velocity_loss.py` | Training loss |
| `models/point_cloud/latent_gmm_registration` | Point-cloud ops |

### New components needed

| Component | Target |
| --- | --- |
| FramePack context compression (variable-kernel patchify per slot) | `layers/embedding/` or package-local |
| Per-frame 3D cache + visibility-based greedy retrieval | `models/vision_language/lyra/memory.py` |
| Differentiable-free forward warp (splatting with z-buffer) | `utils/geometry` |
| Canonical-coordinate correspondence encoder + Q/K-only injection | `layers/attention/` (attention with additive Q/K bias tokens) |
| Self-augmentation training step | trainer `src/train/lyra/` |
| Gaussian DPT head (downsampled) + 3DGS rasterizer | `layers/heads/vision` + rasterizer (likely out of scope in pure Keras; export `.ply` only) |
| DMD distillation | optional |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | Depends on Cosmos phase 1 (VAE + DiT). Add Plucker + depth-warp camera conditioning on a small DiT |
| 2 | 3D cache, retrieval, canonical-coordinate Q/K injection, FramePack slots |
| 3 | Self-augmentation training on a small multi-view video set; ablation vs global point cloud |
| 4 | Feed-forward Gaussian head and `.ply` export; mesh extraction left to external tools |

### Risks / notes

| Item | Note |
| --- | --- |
| Scale | 14B backbone; port the mechanisms on a small DiT, not the weights |
| External dependencies | DAv3, ViPE, OpenVDB, 3DGS rasterizer are outside Keras scope |
| Reusable ideas | Q/K-only conditioning injection, geometry-routed retrieval memory, and one-step self-augmentation are portable to any autoregressive video model in the repo (`video_jepa`, `lewm`) |
| Limitations (paper) | Static scenes only; inherits exposure inconsistencies from DL3DV |

---

## 16. MoGe: Monocular Geometry Estimation (v1, v2, v3)

| Field | Content |
| --- | --- |
| Name | MoGe-1, MoGe-2, MoGe-3 (Microsoft Research, with USTC, Tsinghua) |
| Papers | v1: Wang, Xu, Dai, Xiang, Deng, Tong, Yang. *MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision*. CVPR 2025 (Oral). [arXiv:2410.19115](https://arxiv.org/abs/2410.19115). v2: *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details*. 2025. [arXiv:2507.02546](https://arxiv.org/abs/2507.02546). v3: Kong, Li, Wang, Xu, Yao, Xiang, Yang. *MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement*. July 2026. [arXiv:2607.17967](https://arxiv.org/abs/2607.17967) |
| Reference impl | [microsoft/MoGe](https://github.com/microsoft/MoGe) (PyTorch, ONNX export) |
| License | Code MIT; bundled DINOv2 code Apache-2.0 |
| Family | `src/dl_techniques/models/vision/moge/` (one package, `v1` / `v2` / `v3` modules) |
| Complexity | Medium (v1, v2), High (v3) |
| Priority | P1 (v1, v2), P2 (v3) |

### Summary

Single image -> per-pixel 3D point map (camera frame), depth, valid mask, camera FoV / intrinsics, and (v2) metric scale and normals, in one forward pass. Contribution is mainly supervision design (v1), metric scale + data refinement (v2), and 3D-space refinement of fine structure (v3).

### Checkpoints

| Version | Checkpoint | Metric scale | Normals | Params |
| --- | --- | --- | --- | --- |
| v1 | `Ruicheng/moge-vitl` | No (affine-invariant) | No | 314M |
| v2 | `moge-2-vitl` | Yes | No | 326M |
| v2 | `moge-2-vitl-normal` / `vitb-normal` / `vits-normal` | Yes | Yes | 331M / 104M / 35M |
| v3 | `moge-3-vitl` / `moge-3-vitg` | Yes | Yes | 370M / 1.25B |

### Architecture and method by version

| Version | Details |
| --- | --- |
| v1 encoder | DINOv2 ViT, variable token budget (suggested 1200 to 2500 tokens), aspect ratios 2:1 to 1:2 |
| v1 head | Lightweight conv decoder from multi-level ViT features -> affine-invariant point map + mask |
| v1 supervision | Robust, optimal, efficient (ROE) global alignment solver: closed-form search for scale and z-shift between prediction and GT under truncated L1; multi-scale local geometry loss on sampled spherical neighborhoods, each independently aligned; mask loss for sky / infinite regions |
| v1 camera | Focal length and z-shift recovered from the predicted point map by a small optimization; optional GT FoV input |
| v2 additions | Metric scale predicted separately from relative geometry (global scale head on ViT global token, decoupled from the affine point map); optional normal head; real-data refinement pipeline using synthetic-trained model to clean and fill noisy real labels for sharper detail (verify head placement against code) |
| v3 additions | Frozen MoGe-2 ViT-L base; coarse point map lifted onto a sparse voxel shell; Self-Guided Sparse 3D Refiner (SSR) applies sparse 3D convolutions over 3D neighborhoods instead of image-plane neighborhoods; iterative steps (3 by default) |
| v3 cost | ViT-L, 700^2, A100 FP16: 121 ms (3 steps) vs 39 ms for MoGe-2 |
| v3 gain | Local point-map threshold accuracy 55.9 vs 46.6 (MoGe-2), zero-shot |
| Extras | Panorama inference by splitting equirectangular images into perspective views and merging |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| `models/vision/dino` (DINOv2 + official checkpoint reader) | Encoder and weight loading, directly |
| `models/vision/depth_anything` | Multi-level ViT feature extraction pattern, depth metrics |
| `metrics/` (depth metrics) | Evaluation |
| Entry 1 (VGGT) | Same point-map / camera output convention; MoGe is the monocular counterpart and a geometry source for entry 15's 3D cache |
| `models/point_cloud/` | Point-map post-processing |

### New components needed

| Component | Target |
| --- | --- |
| MoGe conv decoder (point map, mask, normal heads) | `models/vision/moge/` |
| Global metric-scale head | `models/vision/moge/` |
| ROE alignment solver (scale + shift, truncated L1) | `losses/` helper, usable at train and eval time |
| Affine-invariant global point-map loss + multi-scale local geometry loss | `losses/` |
| Normal loss (angular) | `losses/` |
| Focal / shift recovery from point map | `utils/geometry` |
| Variable-token resize logic (num_tokens -> resolution) | `models/vision/moge/` |
| Sparse voxel shell + sparse 3D conv refiner (v3) | No native sparse conv in Keras; options: dense 3D conv on a cropped occupancy grid, or gather-based neighborhood conv on voxel coords |
| PyTorch -> Keras converter + parity oracle | `models/vision/moge/reference.py` pattern |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | v1 on the existing DINOv2 encoder: decoder, ROE solver, global + local losses, focal recovery; weight import and parity |
| 2 | v2: scale head, normal head, ViT-S / B / L variants, weight import |
| 3 | Small trainer on synthetic depth data (`src/train/moge/`) |
| 4 | v3 SSR with a gather-based sparse 3D conv layer; parity only if sparse ops can be matched |

### Risks / notes

| Item | Note |
| --- | --- |
| Value | Best effort-to-value ratio among geometry entries: small, MIT-licensed, reuses DINOv2 already in the repo |
| v3 sparse ops | Sparse 3D convolution is the main porting risk; treat as research port |
| v3 weights | README in mid-2026 listed v3 weights as coming soon; later listings show `moge-3-vitl` (370M) and `moge-3-vitg` (1.25B); confirm availability on Hugging Face |
| Synergy | Shared losses and geometry utils with VGGT (1); depth / point-map provider for Lyra (15) |

---

## Official references (papers + GitHub)

Verification: "R" = repository confirmed during this review, "P" = paper / arXiv ID confirmed during this review. Missing letter = known official source, not re-checked here; confirm before porting.

| # | Model | Paper | Official repository | Notes | Verified |
| --- | --- | --- | --- | --- | --- |
| 1 | VGGT | Wang et al. *VGGT: Visual Geometry Grounded Transformer*. CVPR 2025. [arXiv:2503.11651](https://arxiv.org/abs/2503.11651) | [https://github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt) | Follow-up: [https://github.com/facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) | R, P |
| 2 | CoAtNet | Dai, Liu, Le, Tan. *CoAtNet: Marrying Convolution and Attention for All Data Sizes*. NeurIPS 2021. [arXiv:2106.04803](https://arxiv.org/abs/2106.04803) | None released by the authors | Closest references: timm `maxxvit.py`, [https://github.com/google-research/maxvit](https://github.com/google-research/maxvit) | R (no official repo), P |
| 3 | MaxViT | Tu et al. *MaxViT: Multi-Axis Vision Transformer*. ECCV 2022. [arXiv:2204.01697](https://arxiv.org/abs/2204.01697) | [https://github.com/google-research/maxvit](https://github.com/google-research/maxvit) | TF / JAX |  |
| 4 | FastViT | Vasu et al. *FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization*. ICCV 2023. [arXiv:2303.14189](https://arxiv.org/abs/2303.14189) | [https://github.com/apple/ml-fastvit](https://github.com/apple/ml-fastvit) | Already ported in repo |  |
| 5a | EfficientFormer | Li et al. *EfficientFormer: Vision Transformers at MobileNet Speed*. NeurIPS 2022. [arXiv:2206.01191](https://arxiv.org/abs/2206.01191) | [https://github.com/snap-research/EfficientFormer](https://github.com/snap-research/EfficientFormer) |  |  |
| 5b | EfficientFormerV2 | Li et al. *Rethinking Vision Transformers for MobileNet Size and Speed*. ICCV 2023. [arXiv:2212.08059](https://arxiv.org/abs/2212.08059) | [https://github.com/snap-research/EfficientFormer](https://github.com/snap-research/EfficientFormer) | Same repo |  |
| 6 | Next-ViT | Li et al. *Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios*. 2022. [arXiv:2207.05501](https://arxiv.org/abs/2207.05501) | [https://github.com/bytedance/Next-ViT](https://github.com/bytedance/Next-ViT) |  |  |
| 7 | CPVT | Chu, Tian, Zhang, Wang, Shen. *Conditional Positional Encodings for Vision Transformers*. ICLR 2023. [arXiv:2102.10882](https://arxiv.org/abs/2102.10882) | [https://github.com/Meituan-AutoML/CPVT](https://github.com/Meituan-AutoML/CPVT) | PEG also used in [https://github.com/Meituan-AutoML/Twins](https://github.com/Meituan-AutoML/Twins) | R, P |
| 8a | PVT | Wang et al. *Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions*. ICCV 2021. [arXiv:2102.12122](https://arxiv.org/abs/2102.12122) | [https://github.com/whai362/PVT](https://github.com/whai362/PVT) |  |  |
| 8b | PVTv2 | Wang et al. *PVT v2: Improved Baselines with Pyramid Vision Transformer*. CVMJ 2022. [arXiv:2106.13797](https://arxiv.org/abs/2106.13797) | [https://github.com/whai362/PVT](https://github.com/whai362/PVT) | Same repo |  |
| 9a | Perceiver | Jaegle et al. *Perceiver: General Perception with Iterative Attention*. ICML 2021. [arXiv:2103.03206](https://arxiv.org/abs/2103.03206) | [https://github.com/google-deepmind/deepmind-research/tree/master/perceiver](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver) | JAX / Haiku |  |
| 9b | Perceiver IO | Jaegle et al. *Perceiver IO: A General Architecture for Structured Inputs & Outputs*. ICLR 2022. [arXiv:2107.14795](https://arxiv.org/abs/2107.14795) | Same as 9a |  |  |
| 10a | MobileNetV2 (MBConv) | Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381) | [https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) |  |  |
| 10b | EfficientNet | Tan, Le. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) | [https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) |  |  |
| 11a | ConvNeXt | Liu et al. *A ConvNet for the 2020s*. CVPR 2022. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) | [https://github.com/facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt) | Already ported in repo |  |
| 11b | ConvNeXt V2 | Woo et al. *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*. CVPR 2023. [arXiv:2301.00808](https://arxiv.org/abs/2301.00808) | [https://github.com/facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2) | Already ported in repo |  |
| 12 | CliffordNet (CAN) | Ji. *CliffordNet: All You Need is Geometric Algebra*. 2026. [arXiv:2601.06793](https://arxiv.org/abs/2601.06793) | [https://github.com/ParaMind2025/CAN](https://github.com/ParaMind2025/CAN) | MIT | R, P |
| 13a | Cosmos-Predict2.5 / Transfer2.5 | NVIDIA. *World Simulation with Video Foundation Models for Physical AI*. 2025. [arXiv:2511.00062](https://arxiv.org/abs/2511.00062) | [https://github.com/nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5), [https://github.com/nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) |  | R, P |
| 13b | Cosmos platform, Predict1, Tokenizer | NVIDIA. *Cosmos World Foundation Model Platform for Physical AI*. 2025. [arXiv:2501.03575](https://arxiv.org/abs/2501.03575) | [https://github.com/nvidia-cosmos/cosmos-predict1](https://github.com/nvidia-cosmos/cosmos-predict1), [https://github.com/NVIDIA/Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) | Tokenizer benchmark: [https://github.com/NVlabs/TokenBench](https://github.com/NVlabs/TokenBench) | R (Tokenizer), P |
| 13c | Cosmos-Transfer1 | NVIDIA. *Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control*. 2025. [arXiv:2503.14492](https://arxiv.org/abs/2503.14492) | [https://github.com/nvidia-cosmos/cosmos-transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1) | Predecessor of Transfer2.5 | P |
| 13d | Cosmos-Reason1 | NVIDIA. *Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning*. 2025. [arXiv:2503.15558](https://arxiv.org/abs/2503.15558) | [https://github.com/nvidia-cosmos/cosmos-reason1](https://github.com/nvidia-cosmos/cosmos-reason1) | Text encoder of Predict2.5 | P |
| 13e | Cosmos 3 | NVIDIA. *Cosmos 3 Technical Report*. June 2026. [PDF](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf) | [https://github.com/NVIDIA/Cosmos](https://github.com/NVIDIA/Cosmos), [https://github.com/NVIDIA/Cosmos-Framework](https://github.com/NVIDIA/Cosmos-Framework) | Diffusers `Cosmos3OmniPipeline` | R, P |
| 14 | LFM2 / LFM2.5-VL | Liquid AI. *LFM2 Technical Report*. 2025. [arXiv:2511.23404](https://arxiv.org/abs/2511.23404) | No standalone model-code repo; reference implementation in [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers) (`models/lfm2`, `models/lfm2_vl`) | Org: [https://github.com/Liquid4All](https://github.com/Liquid4All) (`cookbook`, `onnx-export`); weights: [https://huggingface.co/LiquidAI](https://huggingface.co/LiquidAI) | R, P |
| 15a | Lyra 2.0 | Shen, Bahmani, et al. *Lyra 2.0: Explorable Generative 3D Worlds*. 2026. [arXiv:2604.13036](https://arxiv.org/abs/2604.13036) | [https://github.com/nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) (under `Lyra-2/`) | GUI and training code released 2026-07-20 | R, P |
| 15b | Lyra 1.0 | Bahmani, Shen, et al. *Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation*. ICLR 2026 | [https://github.com/nv-tlabs/lyra](https://github.com/nv-tlabs/lyra) | arXiv ID not confirmed here | R, P (title / venue) |
| 16a | MoGe | Wang et al. *MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision*. CVPR 2025 (Oral). [arXiv:2410.19115](https://arxiv.org/abs/2410.19115) | [https://github.com/microsoft/MoGe](https://github.com/microsoft/MoGe) | All versions in one repo | R, P |
| 16b | MoGe-2 | Wang et al. *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details*. 2025. [arXiv:2507.02546](https://arxiv.org/abs/2507.02546) | [https://github.com/microsoft/MoGe](https://github.com/microsoft/MoGe) |  | R, P |
| 16c | MoGe-3 | Kong et al. *MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement*. 2026. [arXiv:2607.17967](https://arxiv.org/abs/2607.17967) | [https://github.com/microsoft/MoGe](https://github.com/microsoft/MoGe) |  | R, P |

### Shared dependencies

| Dependency | Used by | Paper | Repository | Verified |
| --- | --- | --- | --- | --- |
| DINOv2 | 1, 16 | Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |  |
| Wan 2.1 (VAE + DiT) | 13, 15 | Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models*. 2025. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314) | [https://github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | P |
| SigLIP 2 (NaFlex) | 14 | Tschannen et al. *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. 2025. [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) | [https://github.com/google-research/big_vision](https://github.com/google-research/big_vision) |  |
| FramePack | 15 | Zhang, Agrawala. *Packing Input Frame Context in Next-Frame Prediction Models for Video Generation*. 2025. [arXiv:2504.12626](https://arxiv.org/abs/2504.12626) | [https://github.com/lllyasviel/FramePack](https://github.com/lllyasviel/FramePack) |  |
| Flow matching | 13, 15 | Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | n/a | P |
| timm | 2, 3, 7, 8 | n/a | [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |  |