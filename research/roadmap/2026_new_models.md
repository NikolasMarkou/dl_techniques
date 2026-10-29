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
| 14 | LFM2.5-VL (+ LFM2 backbone) | `language/lfm2` + `vision_language/lfm2_vl` | Medium | P2 | Superseded by 33 (LFM2 kept as a config of the LFM3 code) |
| 15 | NVIDIA Lyra 2.0 | `vision_language/lyra` | High | P2 | Proposed |
| 16 | MoGe v1, v2, v3 | `vision/moge` | Medium (v1, v2), High (v3) | P1 (v1, v2), P2 (v3) | Proposed |
| 17 | MiniCPM-o 4.5 | `vision_language/minicpm_o` | High | P2 | Proposed |
| 18 | EdgeCrafter (ECViT, ECDet, ECInsSeg, ECPose) | `vision/edgecrafter` | Medium | P1 | Proposed |
| 19 | SigLIP 2 (FixRes + NaFlex) | `vision_language/siglip2` | Medium | P0 | Partial (`vit_siglip`, SigLIP loss exist) |
| 20 | Perception Encoder (PE-core / lang / spatial) | `vision_language/perception_encoder` | Medium | P1 | Proposed |
| 21 | MetaCLIP 2 | `vision_language/clip` (config + data recipe) | Low | P2 | Partial (`clip` exists) |
| 22 | AIMv2 | `vision/aimv2` | Medium | P1 | Proposed |
| 23 | DINOv2 / DINOv3 | `vision/dino` | n/a | n/a | Exists (dedup) |
| 24 | Qwen3-VL-Embedding / Reranker | `vision_language/qwen3_vl_embedding` | High | P1 | Proposed |
| 25 | WeMM-Embedding | recipe on 24 | Medium | P2 | Proposed |
| 26 | Omni embedders: e5-omni, Omni-Embed-Nemotron, LCO-Embedding-Omni | `losses/` + recipe on 24 | Medium | P1 (e5-omni losses), P2 (models) | Proposed |
| 27 | VIRTUE | `vision_language/virtue` | Medium | P2 | Proposed |
| 28 | VLM embedder lineage: E5-V, VLM2Vec / V2, GME, ColPali | `vision_language/` | Low to Medium | P1 (ColPali, VLM2Vec), P2 (others) | Partial (`colbert` exists) |
| 29 | Emu3 | `vision_language/emu3` | High | P2 | Proposed |
| 30 | BAGEL | `vision_language/bagel` | High | P2 | Proposed |
| 31 | Qwen3-Omni | `vision_language/qwen3_omni` | High | P2 | Proposed |
| 32 | Nemotron 3 Nano Omni | `vision_language/nemotron_omni` | High | P2 | Proposed |
| 33 | LFM3 (1.7B backbone, VL-2B, d1-2B) | `language/lfm3` + `vision_language/lfm3_vl` | Medium | P1 | Proposed |
| 34 | LFM3-Omni-2B (+ LFM3-Audio-2B, d1-omni-2B) | `vision_language/lfm3_omni` | High | P1 | Proposed |

Context for 20 to 32: multimodality beyond CLIP (Oct 2026). SigLIP 2 deduplicated into entry 19; DINOv2 / DINOv3 already in repo.

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

**Status: exists.** `models/vision/fastvit/` (RepMixer, attention block, stage and the reference constants in `components.py`), `layers/conv_blocks/reparam_large_kernel_conv.py`, `layers/embedding/rep_conditional_pos_enc.py` (RepCPE), `conv_blocks/repmixer_block.py`. Used by MobileCLIP v2.

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
| `layers/embedding/rep_conditional_pos_enc.py` (RepCPE) | Reparameterizable variant already present |

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
| `models/vision/video_jepa`, `levjepa`, `lewm` | Latent-space world model counterparts for comparison |
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

## 17. MiniCPM-o 4.5 (OpenBMB)

| Field | Content |
| --- | --- |
| Name | MiniCPM-o 4.5 (9B, end-to-end omni-modal, full-duplex); siblings MiniCPM-V 4.5 / 4.7 (vision only) |
| Paper | Cui, Xu, Wang, Yu, et al. (MiniCPM-o Team, OpenBMB). *MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction*. April 2026. [arXiv:2604.27393](https://arxiv.org/abs/2604.27393) |
| Reference impl | [https://github.com/OpenBMB/MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) (cited in paper; main code also at [https://github.com/OpenBMB/MiniCPM-V)](https://github.com/OpenBMB/MiniCPM-V%29), demo [https://github.com/OpenBMB/MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo), C++ runtime [https://github.com/tc-mb/llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni) |
| Weights | [https://huggingface.co/openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) |
| License | Not checked; MiniCPM weights historically ship under a separate model license. Verify before use |
| Family | `src/dl_techniques/models/vision_language/minicpm_o/` (or a new `omni/` family) |
| Complexity | High |
| Priority | P2 |

### Summary

Open 9B omni model that sees, listens and speaks simultaneously. Text-only LLM output plus lightweight speech decoders, all connected through hidden states and trained end to end. Key technique: Omni-Flow, which serializes time-aligned input and output streams into one causal sequence so a standard decoder LM can run full duplex and act proactively. Runs on edge devices under 12 GB RAM (INT4).

### Architecture

| Component | Details |
| --- | --- |
| Vision encoder | SigLIP ViT (0.4B), LLaVA-UHD slicing for any aspect ratio; max 448x448 in full-duplex mode, 2240x2240 otherwise |
| Vision compression | Each slice -> 1024 ViT tokens -> resampler (cross-attention, learned queries) -> 64 tokens (16x) |
| Audio encoder | Whisper Medium encoder (0.3B), chunk-based streaming, 50 tokens/s |
| Audio projector | 2-layer MLP with 5x temporal compression -> 10 tokens/s |
| LLM backbone | Qwen3-8B; emits text only (3 to 4 decode steps/s at speech pace) |
| Speech token decoder | Llama-style \~0.3B; input per text token = text embedding + MLP-projected LLM hidden state; outputs S3 (CosyVoice) speech tokens |
| Waveform | Streaming flow-matching decoder (CosyVoice2-style), conditioned on reference audio from the multimodal system prompt (voice cloning) |
| Total | \~9B parameters, all learnable parts differentiably connected |

### Omni-Flow (full-duplex framework)

| Item | Details |
| --- | --- |
| Streams | env-visual, env-audio (includes user speech), out-stream (text + speech) |
| Serialization | Per time chunk k: group g_k = \[visual tokens; audio tokens; output tokens\]; groups concatenated into one causal sequence. No output -> single `[listen]` token |
| Chunk size | 1.0 s best; 0.2 s and 0.1 s degrade sharply |
| Boundaries | Explicit boundary tokens between groups beat implicit |
| Control | Listen-Speak (binary control token, then content) beats Listen-Text (shared output space) |
| TAIL | Time-Aligned Interleaving: per chunk, number of text tokens adapts to accumulated speech playback so speech stays aligned with the timeline; bounded look-ahead defers the last few tokens' speech to the next chunk |

### Training

| Stage | Details |
| --- | --- |
| Init | MiniCPM-V 4.5 pretraining checkpoint + pretrained Whisper; new audio projector, LLM-to-speech projector, speech decoder |
| 1. Speech pretraining | Freeze pretrained parts, train new speech modules only |
| 2. Joint pretraining | Unfreeze all; balanced VL / speech / omni mix, fixed modality ratio per step via data-parallel rank assignment; next-token objective incl. full-duplex timeline data |
| 3. Joint SFT | Large-scale instruction tuning, then human-annotated tuning; random frame resolution 0.2 to 0.4 MP, 1 to 5 FPS |
| 4. RL | GRPO with accuracy + format rewards and a smooth length reward (Kimi-K1.5 adapted); RLAIF-V for hallucination |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| `models/language/qwen` (`qwen3.py`) | LLM backbone (Qwen3-8B config) |
| `models/vision/vit_siglip`, `losses/siglip_contrastive_loss.py` | Vision encoder base |
| `layers/attention/perceiver_attention.py` | Resampler (learned queries cross-attending ViT tokens) |
| `models/vision_language/nano_vlm`, `fastvlm` | VLM assembly and token splicing |
| `losses/flow_matching_velocity_loss.py` | Waveform flow-matching decoder |
| Entry 14 (LFM2.5-VL) | Shares tiling / slicing image processor logic |

### New components needed

| Component | Target |
| --- | --- |
| Audio front end (log-mel, chunked) | `layers/` signal processing (no audio stack exists in repo) |
| Whisper encoder (streaming, chunk-based) | `models/audio/whisper/` (new family) |
| `MiniCPMResampler` (2D pos-aware learned-query cross-attn) | `layers/` or package-local |
| LLaVA-UHD slicing processor | `models/vision_language/minicpm_o/` |
| Speech token decoder (small Llama) with LLM hidden-state injection | package-local |
| S3 speech tokenizer + streaming flow-matching vocoder (CosyVoice2) | `models/audio/` (phase 4 or treat as external) |
| Omni-Flow serializer: time-chunked group builder, `[listen]` / boundary / control tokens | `models/vision_language/minicpm_o/omniflow.py` |
| TAIL interleaving scheduler for training targets | same |
| Smooth length reward for GRPO | `losses/` or RL utils |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | Vision path only (MiniCPM-V style): SigLIP + resampler + Qwen3, slicing; weight import and parity on image tasks |
| 2 | Audio input: Whisper encoder + 5x MLP projector; ASR / audio QA parity |
| 3 | Omni-Flow serialization + Listen-Speak control on a small Qwen3 config with toy streaming data |
| 4 | Speech output: speech token decoder + flow-matching vocoder, TAIL |

### Risks / notes

| Item | Note |
| --- | --- |
| Audio gap | Repo has no audio encoder, mel front end, or vocoder; phases 2 and 4 are a new sub-family |
| Scale | 9B; full parity needs large-memory inference; port with small configs first |
| Reusable ideas | Omni-Flow time-chunk serialization and the Listen-Speak control token are model-agnostic and testable on any small decoder LM in the repo |
| Paper numbers | TAIL trades some TTS accuracy (EN WER 3.93 vs 2.38 fixed interleave) for timeline alignment |

---

## 18. EdgeCrafter: Compact ViTs for Edge Dense Prediction

| Field | Content |
| --- | --- |
| Name | EdgeCrafter: ECViT backbones (T, T+, S, S+), ECDet, ECInsSeg, ECPose (S / M / L / X) |
| Paper | Liu, Hou, Li, Wang, Sha, Yu, Wang, Ru, Yu, Shen (Intellindust AI Lab). *EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation*. TMLR 2026. [arXiv:2603.18739](https://arxiv.org/abs/2603.18739) |
| Project | [https://intellindust-ai-lab.github.io/projects/EdgeCrafter/](https://intellindust-ai-lab.github.io/projects/EdgeCrafter/) |
| Reference impl | [https://github.com/Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) (PyTorch); weights on Hugging Face (`Intellindust/ECDet_*`) and GitHub releases |
| License | Not checked; verify repo license and weight terms |
| Family | `src/dl_techniques/models/vision/edgecrafter/` |
| Complexity | Medium |
| Priority | P1 |

### Summary

Claim: compact ViTs lose to YOLO-class CNNs on edge dense prediction because of weak task-specific representation learning, not architecture. Fix: distill a detection-adapted DINOv3 teacher into a small plain ViT, add a conv stem and a trivial multi-scale neck, and plug into an RT-DETR / D-FINE style detector. Same backbone + encoder transfers to instance segmentation and pose with small heads. COCO-only supervision.

### Pipeline

| Stage | Details |
| --- | --- |
| 1. Teacher | DINOv3 adapted to COCO detection inside the ECDet formulation: ECTeacher-S (DINOv3-S in ECDet-L), ECTeacher-B (DINOv3-B in ECDet-X) |
| 2. Distillation | Student ECViT final-block tokens -> single token-wise linear adapter -> L2 to teacher's last two blocks (one-to-many). IN-1K + COCO images, 224 px, 50 epochs, LARS, 1 register token |
| 3. Downstream | Distilled backbone -> ECDet; backbone + encoder reused for ECInsSeg and ECPose |

### Architecture

| Component | Details |
| --- | --- |
| Stem | 4 stacked 3x3 stride-2 convs replacing 16x16 patch embed (stride 16); dilation > 1 on the last conv hurts |
| Backbone | Plain ViT, 1 register token. T: dim 192, 3 heads, FFN 4. T+: 256, 4, 4. S: 384, 6, 4. S+: 384, 6, 6 |
| Multi-scale neck | Average last two block outputs -> stride-16 map; bilinear resize to strides 8 / 16 / 32 + 1x1 conv per level (no FPN) |
| Encoder | RT-DETR: AIFI (self-attn on stride-32 map) + CCFF (CNN cross-scale fusion) |
| Decoder | DETR set prediction, self-attn + deformable cross-attn, 4 layers, 300 queries |
| Det loss | VFL-style classification (DEIM), L1, GIoU, D-FINE DDF + FGL distribution losses; Hungarian matching |
| ECInsSeg head | F(8) resized -> depthwise conv + MLP pixel embeddings; query MLP embeddings; dot-product mask logits; BCE + Dice |
| ECPose head | DETRPose-style: per person 1 instance token + K keypoint tokens; varifocal (OKS target) + visible-joint L1 + OKS loss |

### Results (COCO val2017, T4 TensorRT FP16)

| Model | Params | GFLOPs | Latency | AP |
| --- | --- | --- | --- | --- |
| ECDet-S / M / L / X | 10M / 18M / 31M / 49M | 26 / 53 / 101 / 151 | 5.41 / 7.98 / 10.49 / 12.70 ms | 51.7 / 54.3 / 57.0 / 57.9 box |
| ECInsSeg-S / X | 10.3M / 49.9M | 33.1 / 168.1 | 6.96 / 14.96 ms | 43.0 / 48.4 mask |
| ECPose-S / X | 9.9M / 50.6M | 30.4 / 172.2 | 5.54 / 14.31 ms | 68.9 / 74.8 keypoint |

### Key ablations

| Finding | Detail |
| --- | --- |
| Teacher scale | DINOv3-B best; DINOv3-L worse (capacity gap) |
| Teacher adaptation | COCO-adapting teacher before distillation: 53.5 -> 54.3 AP |
| Supervised IN-21K init | Worse than no pretraining for tiny ViTs |
| Conv stem | +0.8 AP vs vanilla patch embed, mostly small objects |
| Optimizer | LARS > AdamW for distillation |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| `models/vision/detr` | DETR decoder / set-prediction base (check if Hungarian matcher exists) |
| `layers/embedding/register_tokens.py` | Register token |
| `models/vision/dino` (`dino_v2.py`, `dino_v3.py`, `official_weights.py`) | DINOv3 teacher available in repo |
| `models/vision/yolo12`, `losses/yolo12_multitask_loss.py` | CNN baselines on the same benchmark |
| `datasets/vision/coco.py`, `callbacks/coco_map_callback.py` | Data and mAP evaluation |
| `models/vision/vit` | Plain ViT body |

### New components needed

| Component | Target |
| --- | --- |
| `ConvStem` (4x 3x3 s2) | `layers/` stems; also usable in entries 2 to 12 |
| `SimpleMultiScaleNeck` (avg last-k blocks, resize, 1x1) | `layers/` necks |
| RT-DETR encoder (AIFI, CCFF) | `layers/` detection |
| Multi-scale deformable attention | `layers/attention/` (none in repo) |
| Hungarian matcher, GIoU, varifocal, D-FINE DDF / FGL losses | `losses/` |
| Feature-alignment distillation trainer (one-to-many, linear adapter) | `src/train/edgecrafter/` |
| LARS optimizer | `optimization/` |
| Mask head (dot-product), Dice loss | `layers/heads/vision`, `losses/` |
| DETRPose-style pose decoder, OKS loss | phase 3 |
| Detection adaptation of existing DINOv3 (teacher) | trainer |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | ECViT (stem, registers, neck) + weight import of distilled backbones; feature parity |
| 2 | RT-DETR encoder + deformable decoder + losses -> ECDet-S; weight import and COCO mAP parity |
| 3 | ECInsSeg mask head; ECPose decoder |
| 4 | Distillation trainer with DINOv2 / DINOv3 teacher on a small subset |

### Risks / notes

| Item | Note |
| --- | --- |
| Deformable attention | Main porting cost; also needed by any future RT-DETR / D-FINE / DEIM work |
| Teacher | DINOv3 exists in repo (`dino_v3.py`); verify official weight loading for S and B before distillation |
| Latency | ViT latency on T4 is higher than YOLO26 at equal AP; edge advantage is params and accuracy, not raw speed |
| Reusable ideas | Detection-specialized teacher distillation and the conv stem + simple neck are model-agnostic; directly relevant to the hybrid study (entries 2 to 12) |

---

## 19. SigLIP 2

| Field | Content |
| --- | --- |
| Name | SigLIP 2: FixRes (B/32, B/16, L/16, So400m/14, So400m/16, g/16) and NaFlex (B/16, So400m/16) |
| Paper | Tschannen, Gritsenko, Wang, et al. (Google DeepMind). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. 2025. [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) |
| Reference impl | [https://github.com/google-research/big_vision](https://github.com/google-research/big_vision) (JAX, official); Hugging Face `transformers` (`Siglip2Model`, NaFlex) |
| License | Apache-2.0 (big_vision code and released checkpoints; verify per checkpoint) |
| Family | `src/dl_techniques/models/vision_language/siglip2/` (vision tower also exported for `vision/`) |
| Complexity | Medium |
| Priority | P0 |

### Summary

Drop-in successor to SigLIP: same architecture (backward compatible), better training recipe. Combines sigmoid image-text loss with captioning-based pretraining (LocCa), late-stage self-distillation and masked prediction (SILC / TIPS) for dense features, and active data curation for small models. Multilingual text tower. NaFlex variant handles native aspect ratio and variable sequence length in one model.

### Why P0

| Dependent entry | Use |
| --- | --- |
| 14 LFM2.5-VL | Vision encoder (SigLIP2 NaFlex B/16 86M and So400m 400M) |
| 17 MiniCPM-o 4.5 | SigLIP (v1) So400m-class encoder; SigLIP 2 is a backward-compatible upgrade |
| 18 EdgeCrafter, 16 MoGe, 1 VGGT | Alternative dense-feature backbone / teacher to DINOv2 / v3 |
| Existing `nano_vlm`, `fastvlm`, CLIP work | Stronger vision tower |

### Architecture

| Component | Details |
| --- | --- |
| Image tower | Plain ViT, learned position embeddings, MAP head (attention pooling) for the global embedding |
| Text tower | Transformer encoder, multilingual Gemma tokenizer (256k vocab) |
| Training-only heads | LocCa transformer decoder with cross-attention to un-pooled image tokens; EMA teacher for self-distillation |
| FixRes | Fixed square input (e.g. 224 / 256 / 384 / 512), same layout as SigLIP v1 |
| NaFlex | Resize so H and W are patch multiples, aspect ratio kept, patch count \<= target sequence length; position embedding grid bilinearly resized to the patch grid; padding + attention mask. Combines FlexiViT (multiple seq lengths) and NaViT (native aspect) ideas. HF default `max_num_patches=256` |

### Training recipe

| Component | Weight / timing | Details |
| --- | --- | --- |
| Sigmoid loss | 100% of training | Pairwise binary classification over image-text pairs (SigLIP v1) |
| LocCa loss | 100% of training, equal weight with sigmoid | Decoder on un-pooled features: captioning, referring expression prediction, grounded captioning |
| Self-distillation (SILC) | Last 20% | Local-to-global correspondence vs EMA teacher on augmented views |
| Masked prediction (TIPS) | Last 20% | 50% of student patch embeddings replaced by mask tokens; match teacher features at masked positions |
| Active data curation (ACID) | Small models (B/16, B/32) | Teacher-guided online example selection, acts as implicit distillation |
| Data | WebLI, multilingual, de-biasing filters | Training-scale numbers: verify against paper |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| `models/vision/vit_siglip` | ViT body; check whether it matches SigLIP v1 layout (MAP head, LN placement) |
| `losses/siglip_contrastive_loss.py` | Sigmoid loss |
| `layers/attention/perceiver_attention.py` | MAP head (single learned query cross-attention) |
| `models/vision/dino` | EMA teacher and self-distillation pattern (DINO line) |
| `models/language/gemma` (if present) or `layers/tokenizers` | Gemma tokenizer for the text tower |
| Entry 14 LFM2.5-VL plan | Was going to build `models/vision/siglip2/` there; consolidate here |

### New components needed

| Component | Target |
| --- | --- |
| `MAPHead` (multihead attention pooling) | `layers/pooling/` if not already present |
| NaFlex patchifier: aspect-preserving resize, variable patch count, padding mask, resized position embeddings | `layers/embedding/` |
| Masked attention path for padded patch sequences | `layers/attention/` |
| SigLIP 2 text tower + Gemma tokenizer wiring | `models/vision_language/siglip2/` |
| big_vision / HF checkpoint converter + parity oracle (`reference.py`) | same |
| LocCa decoder, SILC / TIPS losses (training only) | phase 3, `losses/` |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | FixRes image + text towers, MAP head; import B/16 and So400m/14 weights; zero-shot parity on a small ImageNet subset |
| 2 | NaFlex patchifier and masking; import NaFlex B/16 and So400m/16; parity across several aspect ratios and patch budgets |
| 3 | Training recipe: sigmoid + LocCa decoder; SILC / TIPS late-stage losses; small-scale run |
| 4 | Export vision tower as a registered backbone for entries 14, 17 and `nano_vlm` |

### Risks / notes

| Item | Note |
| --- | --- |
| Existing `vit_siglip` | Name suggests SigLIP but may not match the official layout; decide whether to fix it or add a new package |
| NaFlex masking | Variable-length batches need consistent masking in attention, MAP head and position embedding resize |
| Weight import | So400m uses non-standard width/depth (shape-optimized); configs must match exactly |
| Training scale | Full recipe infeasible; value is in faithful architecture, weights and the NaFlex machinery |

---

# Part 2: Multimodality beyond CLIP (as of Oct 2026)

Grouping: contrastive dual encoders (19 to 22), vision-only SSL (23), MLLM-based universal embedders (24 to 28), unified understanding + generation (29, 30), omni any-to-any (31, 32).

---

## 20. Perception Encoder (Meta)

| Field | Content |
| --- | --- |
| Paper | Bolya et al. *Perception Encoder: The best visual embeddings are not at the output of the network*. 2025. [arXiv:2504.13181](https://arxiv.org/abs/2504.13181) |
| Reference impl | [https://github.com/facebookresearch/perception_models](https://github.com/facebookresearch/perception_models) |
| Variants | PE-core (B, L, G \~2B): CLIP-style image / video encoder. PE-lang: language-aligned for MLLMs. PE-spatial: dense-aligned |
| Key idea | Contrastive-only training with a robust image recipe + video data engine; best general features sit in intermediate layers. Language alignment (MLLM tuning) and spatial alignment (self-distillation from own intermediate layers + SAM 2 mask cues) surface them |
| Overlap | `models/vision_language/clip`, `losses/clip_contrastive_loss.py`, `models/vision/dino` (self-distillation), `vision_language/sam` |
| New | Attention-pool head, PE-specific ViT tweaks (2D RoPE, layer scale), intermediate-layer alignment heads, weight converter |
| Priority | P1: one family covers CLIP-replacement, MLLM tower and dense backbone |

---

## 21. MetaCLIP 2 (Meta)

| Field | Content |
| --- | --- |
| Paper | Chuang et al. *MetaCLIP 2: A Worldwide Scaling Recipe*. 2025. [arXiv:2507.22062](https://arxiv.org/abs/2507.22062) |
| Reference impl | [https://github.com/facebookresearch/MetaCLIP](https://github.com/facebookresearch/MetaCLIP) |
| Key idea | Same CLIP architecture (ViT-H/14 scale); contribution is worldwide (300+ language) metadata curation and a training recipe that removes the English-vs-multilingual trade-off |
| Overlap | `models/vision_language/clip` covers the model |
| New | Multilingual tokenizer wiring, metadata-balanced curation utility, weight converter |
| Priority | P2: mostly a data recipe; port weights into existing `clip` |

---

## 22. AIMv2 (Apple)

| Field | Content |
| --- | --- |
| Paper | Fini et al. *Multimodal Autoregressive Pre-training of Large Vision Encoders*. CVPR 2025. [arXiv:2411.14402](https://arxiv.org/abs/2411.14402) |
| Reference impl | [https://github.com/apple/ml-aim](https://github.com/apple/ml-aim) |
| Key idea | ViT encoder (prefix attention) + causal multimodal decoder that autoregressively regresses image patches, then generates caption tokens. Dense supervision from every patch and token |
| Overlap | `models/vision/vit`, `layers/transformers/` (causal decoder), `models/vision/mae` style pixel targets if present |
| New | Prefix-attention mask in encoder, joint patch-regression + text decoder, weight converter (300M to 2.7B) |
| Priority | P1: clean generative alternative to contrastive pretraining; small enough to train in-repo |

---

## 23. DINOv2 / DINOv3

**Status: exists.** `models/vision/dino/` (`dino_v1.py`, `dino_v2.py`, `dino_v3.py`, `official_weights.py`, `training.py`). No action; used as teacher / backbone by entries 1, 16, 18, 20.

---

## 24. Qwen3-VL-Embedding / Qwen3-VL-Reranker (Alibaba)

| Field | Content |
| --- | --- |
| Paper | Li et al. *Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking*. 2026. [arXiv:2601.04720](https://arxiv.org/abs/2601.04720) |
| Reference impl | [https://github.com/QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) |
| Variants | Embedding 2B (28 layers, 2048-D) and 8B (36 layers, 4096-D); Reranker 2B / 8B; 32K context; 30+ languages |
| Embedder | Qwen3-VL backbone, last hidden state as embedding; text, image, document image, video in one space; Matryoshka dims; instruction-aware; multi-stage training from contrastive pretraining to reranker distillation |
| Reranker | Single-tower cross-encoder; relevance = generation probability of `yes` / `no` tokens |
| Fine-tuning | LoRA r=32, alpha=32 on q/k/v/up/down/gate |
| Result | 8B: 77.8 on MMEB-V2 at release |
| Overlap | `models/language/qwen` (Qwen3 text), `losses/infonce_loss.py`, `layers/adapters/lora.py`, entry 19 / 14 vision assembly |
| New | Qwen3-VL vision tower + merger (not in repo), last-token pooling, Matryoshka loss, yes/no reranker head, MMEB eval harness |
| Priority | P1: reference recipe that 25, 26, 27 build on |

---

## 25. WeMM-Embedding (Tencent WeChat)

| Field | Content |
| --- | --- |
| Paper | Zhou, Mei, Li, Wang, Rao, Lyu. *WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report*. 2026. [arXiv:2608.24053](https://arxiv.org/abs/2608.24053) |
| Reference impl | [https://github.com/Tencent/WeMM-Embedding](https://github.com/Tencent/WeMM-Embedding) (Apache-2.0 per secondary source) |
| Variants | 2B, 4B, 9B on native multimodal Qwen3.5 backbones |
| Architecture | Interleaved segments + appended `<embedding>` token; last-token pooling + L2 norm; multiple `<embedding>` tokens per sequence give several embeddings in one pass; Matryoshka truncation |
| Training | Stage 1 broad alignment on several hundred million pairs; stage 2 curated hard negatives, reranker-derived relevance, cross-scale knowledge transfer; semantic-ID (3-level residual k-means) resampling to de-duplicate dense regions |
| Results | MMEB-V2: 2B 77.9, 4B 79.2, 9B 80.6 (rank 1 at release vs Qwen3-VL-Embedding-8B 77.8); also reports MMEB-V3 |
| Overlap | Same stack as 24 |
| New | Multi-`<embedding>`-token pooling, semantic-ID resampler, cross-scale distillation loss |
| Priority | P2: implement as recipe variant of 24 |

---

## 26. Omni embedders (audio included) + MMEB-V3

| Model | Paper | Backbone / key idea | Weights / code |
| --- | --- | --- | --- |
| e5-omni | Chen, Gao, Timofte, Sakai, Dou. *e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings*. Findings of ACL 2026. [arXiv:2601.03666](https://arxiv.org/abs/2601.03666) | VLM bi-encoder (3B, 7B). Three plug-in modules: modality-aware temperature calibration, negative curriculum with debiasing, batch whitening + CORAL covariance alignment. Backbone-agnostic | [https://huggingface.co/collections/Haon-Chen/e5-omni](https://huggingface.co/collections/Haon-Chen/e5-omni) |
| Omni-Embed-Nemotron | Xu et al. *Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video*. 2025 | Qwen-Omni backbone, 3B | Hugging Face `nvidia/omni-embed-nemotron-3b` (not verified) |
| LCO-Embedding-Omni | Xiao et al. Language-centric omni-modal embedding. 2025 / 2026 | Language-centric alignment, learned compression; 3B, 7B | Not verified |
| MMEB-V3 (benchmark) | Huang et al. *MMEB-V3: Measuring the Performance Gaps of Omni-Modality Embedding Models*. 2026. [arXiv:2604.23321](https://arxiv.org/abs/2604.23321) | 190 tasks: MMEB-V2 (78) + text (53) + agent (47) + audio (11) + MCMR; OmniSET diagnostics show query-modality bias and cross-modal asymmetry |  |

| Item | Content |
| --- | --- |
| MMEB-V3 snapshot (V3-All) | e5-omni 7B 47.1, e5-omni 3B 44.6, Omni-Embed-Nemotron 3B 43.5, LCO-Embedding-Omni 7B 40.6 (from WeMM repo table) |
| Overlap | Entry 24 stack; entry 17 audio sub-family (Whisper) |
| New | e5-omni losses (`ModalityTemperatureContrastive`, `DebiasedCurriculumNegatives`, `WhiteningCovarianceAlignment`) in `losses/`; audio encoder from entry 17 |
| Priority | P1 for e5-omni losses (cheap, backbone-agnostic, usable with existing `clip` / `siglip` models); P2 for full omni models |

---

## 27. VIRTUE (Sony)

| Field | Content |
| --- | --- |
| Paper | Wang, Tateishi, Wu, Takahashi, Mitsufuji. *VIRTUE: Visual-Interactive Text-Image Universal Embedder*. ICLR 2026. [arXiv:2510.00523](https://arxiv.org/abs/2510.00523) |
| Reference impl | [https://github.com/sony/virtue](https://github.com/sony/virtue) |
| Key idea | SAM 2 prompt encoder processes visual prompts (point, box, mask; uniform points when none) -> segmentation-language connector compresses to entity embeddings -> VLM fuses with global image + text embeddings -> contrastive training. Region-aware retrieval |
| Benchmark | SCaR: 1M-sample segmentation-and-scene caption retrieval |
| Results | +3.1 to 8.5% on 36 MMEB tasks; +15.2 to 20.3% on SCaR |
| Overlap | `models/vision_language/sam`, `losses/sam2_video_loss.py`, entry 24 stack |
| New | Segmentation-language connector, visual-prompt batching |
| Priority | P2 |

---

## 28. VLM embedder lineage

| Model | Paper | Key idea | Repo | Priority |
| --- | --- | --- | --- | --- |
| E5-V | Jiang et al. *E5-V: Universal Embeddings with Multimodal Large Language Models*. 2024. [arXiv:2407.12580](https://arxiv.org/abs/2407.12580) | Prompt-based single-word summarization; text-only contrastive training transfers to images | [https://github.com/kongds/E5-V](https://github.com/kongds/E5-V) | P2 |
| VLM2Vec / VLM2Vec-V2 | Jiang et al. *VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks*. ICLR 2025. [arXiv:2410.05160](https://arxiv.org/abs/2410.05160). Meng et al. *VLM2Vec-V2*. 2025. [arXiv:2507.04590](https://arxiv.org/abs/2507.04590) | Instruction-conditioned VLM embedder; introduced MMEB and MMEB-V2 (78 tasks: image, video, visual docs) | [https://github.com/TIGER-AI-Lab/VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) | P1 (MMEB harness) |
| GME | Zhang et al. *GME: Improving Universal Multimodal Retrieval by Multimodal LLMs*. CVPR 2025. [arXiv:2412.16855](https://arxiv.org/abs/2412.16855) | Qwen2-VL embedder, synthetic fused-modal training data | Hugging Face `Alibaba-NLP/gme-Qwen2-VL-*` | P2 |
| ColPali | Faysse et al. *ColPali: Efficient Document Retrieval with Vision Language Models*. ICLR 2025. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) | PaliGemma patch embeddings + ColBERT late interaction over document page images; ViDoRe benchmark | [https://github.com/illuin-tech/colpali](https://github.com/illuin-tech/colpali) | P1 |

| Item | Content |
| --- | --- |
| Overlap | `models/language/colbert` + `losses/colbert_loss.py` (late interaction for ColPali); `losses/infonce_loss.py` |
| New | Multi-vector page encoder (ColPali), MMEB / ViDoRe eval harness |

---

## 29. Emu3 (BAAI)

| Field | Content |
| --- | --- |
| Paper | Wang et al. *Emu3: Next-Token Prediction is All You Need*. 2024. [arXiv:2409.18869](https://arxiv.org/abs/2409.18869). Journal version: Nature 2026 (per brief; not verified) |
| Reference impl | [https://github.com/baaivision/Emu3](https://github.com/baaivision/Emu3) |
| Key idea | Single decoder-only Transformer trained from scratch with pure next-token prediction over discrete tokens for text, images and video; generation and understanding in one model |
| Components | VQ visual tokenizer (spatial + temporal compression), Llama-style 8B decoder, DPO for generation |
| Overlap | `models/vision/vq_vae`, `models/language/qwen` style decoder |
| New | Video-capable VQ tokenizer, interleaved token vocabulary, CFG for AR image sampling |
| Priority | P2 |

---

## 30. BAGEL (ByteDance Seed)

| Field | Content |
| --- | --- |
| Paper | Deng et al. *Emerging Properties in Unified Multimodal Pretraining*. 2025. [arXiv:2505.14683](https://arxiv.org/abs/2505.14683) |
| Reference impl | [https://github.com/ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) |
| Key idea | Mixture-of-Transformers: understanding expert + generation expert sharing self-attention; 7B active / 14B total; trained on large interleaved text-image-video data |
| Encoders | Two: ViT (SigLIP-family) for understanding, VAE latents for generation; image generation via rectified flow / flow matching |
| Result | 88% GenEval with LLM rewriter (vs FLUX.1-dev 82%) |
| Overlap | `losses/flow_matching_velocity_loss.py`, `models/vision/vae`, entry 13 Cosmos 3 MoT layer, entry 19 SigLIP 2 |
| New | MoT layer (shared with Cosmos 3 phase 4), mixed AR + flow training loop |
| Priority | P2; build MoT once for 13 and 30 |

---

## 31. Qwen3-Omni (Alibaba)

| Field | Content |
| --- | --- |
| Paper | Qwen Team. *Qwen3-Omni Technical Report*. 2025. [arXiv:2509.17765](https://arxiv.org/abs/2509.17765) |
| Reference impl | [https://github.com/QwenLM/Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) |
| Key idea | Thinker-Talker architecture, both MoE: Thinker handles text, image, audio, video understanding; Talker streams speech from Thinker states via multi-codebook codec tokens; low first-packet latency |
| Claim | Open-source SOTA on 32 of 36 audio and audio-visual benchmarks |
| Overlap | `layers/moe`, `models/language/qwen`; entry 17 (MiniCPM-o) audio sub-family and speech decoder |
| New | Audio encoder (AuT), multi-codebook speech tokenizer and code-to-wave decoder, TM-RoPE |
| Priority | P2: share audio stack with 17 |

---

## 32. Nemotron 3 Nano Omni (NVIDIA)

| Field | Content |
| --- | --- |
| Release | 2026-04-28; model card `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` (also FP8, NVFP4) |
| Paper | No standalone paper found; model card is the reference |
| License | NVIDIA Open Model Agreement, commercial use allowed |
| Backbone | Nemotron 3 Nano 30B-A3B: Mamba2-Transformer hybrid MoE, 31B total / \~3B active; 23 Mamba2 + 23 MoE + 6 GQA layers; 128 experts, top-6 (layer counts from secondary sources) |
| Encoders | C-RADIO v4-H (image, video frames), Parakeet (speech) |
| I/O | Video, audio, image, text in; text out; 256K context; reasoning toggle |
| Overlap | `layers/ssm/mamba2.py`, `layers/moe/`, `layers/attention/group_query_attention.py`, `models/language/mamba` |
| New | Hybrid layer schedule, RADIO encoder, Parakeet (FastConformer) encoder |
| Priority | P2; but backbone pieces already in repo make a small hybrid-MoE text config cheap |

---

## 33. LFM3 (Liquid AI)

| Field | Content |
| --- | --- |
| Release | 2026-10-27. Blog: *Introducing LFM3: Every Modality for Specialized Intelligence* ([https://www.liquid.ai/blog/lfm3](https://www.liquid.ai/blog/lfm3), HF mirror [https://huggingface.co/blog/LiquidAI/lfm3)](https://huggingface.co/blog/LiquidAI/lfm3%29). No paper |
| Models | LFM3-1.7B-Base, LFM3-1.7B (text), LFM3-VL-2B (text, image, video), d1-2B (decision); omni / audio variants in entry 34 |
| Reference impl | Hugging Face `transformers` (`trust_remote_code`), weights in the [LFM3 collection](https://huggingface.co/collections/LiquidAI/lfm3); fine-tuning notebooks in [https://github.com/Liquid4All/cookbook](https://github.com/Liquid4All/cookbook) |
| License | LFM Open License v1.0 (revenue-capped commercial use, as LFM2) |
| Family | `src/dl_techniques/models/language/lfm3/` + `src/dl_techniques/models/vision_language/lfm3_vl/` |
| Complexity | Medium |
| Priority | P1; supersedes entry 14 for new work |

### Backbone (LFM3-1.7B-Base)

| Component | Details |
| --- | --- |
| Params / depth | 1.73B, 24 layers |
| Tri-hybrid layer mix | 13 gated short convolutions (kernel 7, up from 3 in LFM2) + 5 sliding-window attention (256-token window) + 6 global attention |
| Head-wise gated attention | Sigmoid gate scales each head's output; a head can switch off; helps suppress activation outliers for quantization |
| Partial RoPE | Global layers rotate only the highest-frequency half of each head (other half is position-free content matching); SWA layers keep full RoPE |
| Sandwich norm | Normalize both block input and block output |
| Vocab / context | 128K vocabulary (shared with LFM2.5-8B-A1B); 256K context |
| Pretraining | 22T tokens; final 3.5T as mid-training on curated agentic / tool-call data, also extends context to 256K |
| Design origin | Hardware-in-the-loop search over 14 device / backend combinations (phones, Raspberry Pi 5, laptops, cloud CPU / GPU, Qualcomm Hexagon NPU) |
| Claims | Beats LFM2.5-2.6B with 35% fewer params; \~25% faster than LFM2-1.2B per parameter at 8K context; \~1 GB at 4-bit |

### LFM3-VL-2B

| Component | Details |
| --- | --- |
| Params | 1.92B = 1.73B LM + 164M vision encoder + 25M projector |
| Vision encoder | In-house conv + attention hybrid; \~3x less compute than the SigLIP2 encoder of LFM2.5-VL; matches Qwen3.5-2B vision encoder at 2.2x fewer FLOPs |
| Video | Each pair of sampled frames embedded into one temporal slice (halves temporal length) |
| Capabilities | Reasoning before answering, grounding / pointing, spatial reasoning, tool calling; 128K context |
| Claim | 62-benchmark vision average 71.4 vs 57.6 for LFM2.5-VL-3B; on par with Gemma 4 12B |
| Robotics demo | Fine-tuned into orchestrator, navigator and VLA (96.2% LIBERO avg, \<50 ms action chunks on Jetson AGX Orin) |

### d1-2B (decision model)

| Item | Details |
| --- | --- |
| Built on | LFM3-VL-2B |
| Function | Answers many structured questions (yes/no, choice, multi-choice, score, point, read, estimate) about one input in a single forward pass, each with a calibrated probability |
| Note | Interesting non-generative head design; worth a separate small study |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| Entry 14 plan: `GatedShortConv`, `LFM2Block`, layer-type schedule | Same mixer with kernel 7; LFM2 becomes a config |
| `layers/attention/group_query_attention.py`, `layers/attention/window_attention.py` | Global and sliding-window attention |
| `layers/embedding/rotary_position_embedding.py` | Needs partial-RoPE option |
| `layers/norms` (RMSNorm), `layers/ffn` (SwiGLU) | Block internals |
| `models/vision_language/nano_vlm`, `fastvlm` | VLM assembly |

### New components needed

| Component | Target |
| --- | --- |
| Sliding-window causal attention with KV cache window | `layers/attention/` |
| Head-wise sigmoid output gate | `layers/attention/` option |
| Partial RoPE (high-frequency half only) | `layers/embedding/` option |
| Sandwich-norm block wrapper | `layers/transformers/` |
| Tri-hybrid layer schedule config (`conv`, `swa`, `global`) | `models/language/lfm3/` |
| In-house conv + attention vision encoder, frame-pair temporal embedding | `models/vision_language/lfm3_vl/` |
| d1 multi-question calibrated head | phase 3 |
| HF converter + logits parity | both packages |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | Backbone: tri-hybrid blocks, gated attention, partial RoPE, sandwich norm; import LFM3-1.7B-Base weights; logits parity |
| 2 | LFM3-VL-2B: vision encoder + projector + video frame pairing; parity |
| 3 | d1-2B decision head |

### Risks / notes

| Item | Note |
| --- | --- |
| Freshness | Released one day before this entry; layer order, encoder internals and configs must be read from HF configs / remote code |
| No paper | Blog and model cards are the only references |
| Hybrid study | Tri-hybrid (conv / SWA / global) is a direct 1D analogue of the conv-early, attention-late study (entries 2 to 12) |

---

## 34. LFM3-Omni-2B (Liquid AI)

| Field | Content |
| --- | --- |
| Release | 2026-10-27, same blog as entry 33 |
| Models | LFM3-Omni-2B (text, image, video, audio in; text and speech out), LFM3-Audio-2B (text, audio in; text and speech out), d1-omni-2B (decision, audio-capable) |
| Reference impl | Hugging Face `transformers` (`AutoModelForMultimodalLM`, `trust_remote_code`); weights [https://huggingface.co/LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B), [https://huggingface.co/LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) |
| License | LFM Open License v1.0 |
| Family | `src/dl_techniques/models/vision_language/lfm3_omni/` (audio parts shared with a new `models/audio/` family) |
| Complexity | High |
| Priority | P1: smallest open omni model with speech output in this list (\~2.2B), most feasible omni port |

### Architecture

| Component | Details |
| --- | --- |
| Backbone | LFM3 tri-hybrid LM (entry 33) |
| Vision | LFM3-VL encoder + projector (entry 33) |
| Audio encoder | In-house, fully causal, sliding-window attention + short convolutions; enables prefill while the user speaks, time to first token as low as 50 ms |
| Audio decoder | Generates audio codes alongside text (interleaved text + audio mode via \`\< |
| LiquidCodec | New audio tokenizer / decoder, 24 kHz output, streaming, \~87x real time on one CPU thread (Apple M4 Max) |
| Generation modes | Text only, speech only, interleaved text + speech; separate audio sampling params (temperature, top-k, repetition window) |
| Context | 32K (secondary sources) |

### Claims

| Item | Details |
| --- | --- |
| Audio-visual video | Beats models up to 10.7B (e.g. Qwen2.5-Omni-7B) on Daily-Omni, WorldSense, AV-SpeakerBench |
| Retained skills | VQA, ASR, TTS, instruction following, tool calls from spoken requests |
| LFM3-Audio-2B | 18 languages; UTMOSv2 3.83 vs 3.22 for LFM2.5-Audio-1.5B; adds audio understanding, reasoning, tool calling |
| d1-omni-2B | DecisionBench 83.4 by voice vs 83.6 by text |

### Overlap with existing repo

| Existing | Reuse |
| --- | --- |
| Entry 33 backbone + vision | Shared |
| `layers/attention/window_attention.py`, entry 14 / 33 `GatedShortConv` | Audio encoder mixers |
| Entry 17 (MiniCPM-o) plan: mel front end, streaming audio encoder, speech token decoder | Shared audio sub-family |
| Entry 31 (Qwen3-Omni) | Comparison target; shared codec / streaming design questions |

### New components needed

| Component | Target |
| --- | --- |
| Audio front end (log-mel or learned) | `layers/` signal processing |
| Causal SWA + short-conv audio encoder with streaming state | `models/audio/lfm3_audio_encoder/` |
| Audio-code decoder head + interleaved text / audio generation loop | `models/vision_language/lfm3_omni/` |
| LiquidCodec (streaming neural codec decoder) | `models/audio/liquidcodec/` |
| Mixed-modality chat template and special tokens | package-local |

### Implementation plan

| Phase | Scope |
| --- | --- |
| 1 | Depends on entry 33 phase 1 and 2 |
| 2 | Audio encoder + projector; ASR / audio QA parity using LFM3-Audio-2B weights |
| 3 | Audio decoder + LiquidCodec; TTS and interleaved speech output parity |
| 4 | Full LFM3-Omni-2B assembly; d1-omni head |

### Risks / notes

| Item | Note |
| --- | --- |
| Freshness | Released one day before this entry; codec and decoder internals only available via remote code |
| Audio gap | Same gap as entries 17 and 31; build the audio sub-family once |
| Value | Single 2B model that exercises every new stack in this list (hybrid backbone, vision, audio in, speech out) |

---

## Key benchmarks to track

| Area | Benchmark | Reference |
| --- | --- | --- |
| Universal embeddings | MMEB-V2 (78 tasks), MMEB-V3 (190 tasks) | VLM2Vec-V2 [arXiv:2507.04590](https://arxiv.org/abs/2507.04590); MMEB-V3 [arXiv:2604.23321](https://arxiv.org/abs/2604.23321) |
| Zero-shot / retrieval | ImageNet zero-shot, COCO / Flickr30K retrieval | standard |
| Document retrieval | ViDoRe | ColPali paper |
| Region-aware retrieval | SCaR | VIRTUE paper |
| MLLM understanding | MMMU, video QA suites | MMMU: Yue et al. 2024 [arXiv:2311.16502](https://arxiv.org/abs/2311.16502) |
| Unified generation | GenEval | Ghosh et al. 2023 [arXiv:2310.11513](https://arxiv.org/abs/2310.11513) |
| Omni / audio-visual | Omni-Bench, audio-visual QA, MAEB, MVEB | per paper |

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
| 17 | MiniCPM-o 4.5 | Cui et al. *MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction*. 2026. [arXiv:2604.27393](https://arxiv.org/abs/2604.27393) | [https://github.com/OpenBMB/MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) (also [https://github.com/OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)) | Demo: [https://github.com/OpenBMB/MiniCPM-o-Demo;](https://github.com/OpenBMB/MiniCPM-o-Demo;) C++: [https://github.com/tc-mb/llama.cpp-omni;](https://github.com/tc-mb/llama.cpp-omni;) weights: [https://huggingface.co/openbmb/MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5) | R, P |
| 18 | EdgeCrafter | Liu et al. *EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation*. TMLR 2026. [arXiv:2603.18739](https://arxiv.org/abs/2603.18739) | [https://github.com/Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Project: [https://intellindust-ai-lab.github.io/projects/EdgeCrafter/;](https://intellindust-ai-lab.github.io/projects/EdgeCrafter/;) sibling DEIMv2 from same lab | R, P |
| 19 | SigLIP 2 | Tschannen et al. *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features*. 2025. [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) | [https://github.com/google-research/big_vision](https://github.com/google-research/big_vision) | Also Hugging Face `transformers` `Siglip2Model` | P |
| 20 | Perception Encoder | Bolya et al. *Perception Encoder: The best visual embeddings are not at the output of the network*. 2025. [arXiv:2504.13181](https://arxiv.org/abs/2504.13181) | [https://github.com/facebookresearch/perception_models](https://github.com/facebookresearch/perception_models) |  |  |
| 21 | MetaCLIP 2 | Chuang et al. *MetaCLIP 2: A Worldwide Scaling Recipe*. 2025. [arXiv:2507.22062](https://arxiv.org/abs/2507.22062) | [https://github.com/facebookresearch/MetaCLIP](https://github.com/facebookresearch/MetaCLIP) |  |  |
| 22 | AIMv2 | Fini et al. *Multimodal Autoregressive Pre-training of Large Vision Encoders*. CVPR 2025. [arXiv:2411.14402](https://arxiv.org/abs/2411.14402) | [https://github.com/apple/ml-aim](https://github.com/apple/ml-aim) |  |  |
| 23 | DINOv2 / DINOv3 | See shared dependencies | [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | Already in repo |  |
| 24 | Qwen3-VL-Embedding / Reranker | Li et al. *Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking*. 2026. [arXiv:2601.04720](https://arxiv.org/abs/2601.04720) | [https://github.com/QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) |  | R, P |
| 25 | WeMM-Embedding | Zhou et al. *WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report*. 2026. [arXiv:2608.24053](https://arxiv.org/abs/2608.24053) | [https://github.com/Tencent/WeMM-Embedding](https://github.com/Tencent/WeMM-Embedding) |  | R, P |
| 26a | e5-omni | Chen et al. *e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings*. Findings of ACL 2026. [arXiv:2601.03666](https://arxiv.org/abs/2601.03666) | Weights only: [https://huggingface.co/collections/Haon-Chen/e5-omni](https://huggingface.co/collections/Haon-Chen/e5-omni) |  | R (weights), P |
| 26b | Omni-Embed-Nemotron | Xu et al. *Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video*. 2025 | Hugging Face `nvidia/omni-embed-nemotron-3b` | arXiv ID not confirmed | P (title / year) |
| 26c | LCO-Embedding-Omni | Xiao et al. 2025 / 2026 | Not confirmed |  |  |
| 26d | MMEB-V3 | Huang et al. *MMEB-V3: Measuring the Performance Gaps of Omni-Modality Embedding Models*. 2026. [arXiv:2604.23321](https://arxiv.org/abs/2604.23321) | Eval code also in Tencent/WeMM-Embedding `mmeb_v3_eval/` |  | P |
| 27 | VIRTUE | Wang et al. *VIRTUE: Visual-Interactive Text-Image Universal Embedder*. ICLR 2026. [arXiv:2510.00523](https://arxiv.org/abs/2510.00523) | [https://github.com/sony/virtue](https://github.com/sony/virtue) |  | R, P |
| 28a | E5-V | Jiang et al. *E5-V: Universal Embeddings with Multimodal Large Language Models*. 2024. [arXiv:2407.12580](https://arxiv.org/abs/2407.12580) | [https://github.com/kongds/E5-V](https://github.com/kongds/E5-V) |  |  |
| 28b | VLM2Vec / VLM2Vec-V2 | Jiang et al. ICLR 2025. [arXiv:2410.05160](https://arxiv.org/abs/2410.05160); Meng et al. 2025. [arXiv:2507.04590](https://arxiv.org/abs/2507.04590) | [https://github.com/TIGER-AI-Lab/VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) |  |  |
| 28c | GME | Zhang et al. *GME: Improving Universal Multimodal Retrieval by Multimodal LLMs*. CVPR 2025. [arXiv:2412.16855](https://arxiv.org/abs/2412.16855) | Hugging Face `Alibaba-NLP/gme-Qwen2-VL-*` |  |  |
| 28d | ColPali | Faysse et al. *ColPali: Efficient Document Retrieval with Vision Language Models*. ICLR 2025. [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) | [https://github.com/illuin-tech/colpali](https://github.com/illuin-tech/colpali) |  |  |
| 29 | Emu3 | Wang et al. *Emu3: Next-Token Prediction is All You Need*. 2024. [arXiv:2409.18869](https://arxiv.org/abs/2409.18869) | [https://github.com/baaivision/Emu3](https://github.com/baaivision/Emu3) | Nature 2026 version not verified |  |
| 30 | BAGEL | Deng et al. *Emerging Properties in Unified Multimodal Pretraining*. 2025. [arXiv:2505.14683](https://arxiv.org/abs/2505.14683) | [https://github.com/ByteDance-Seed/Bagel](https://github.com/ByteDance-Seed/Bagel) |  |  |
| 31 | Qwen3-Omni | Qwen Team. *Qwen3-Omni Technical Report*. 2025. [arXiv:2509.17765](https://arxiv.org/abs/2509.17765) | [https://github.com/QwenLM/Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) |  |  |
| 32 | Nemotron 3 Nano Omni | Model card (no paper found) | [https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) |  | R (model card) |
| 33 | LFM3 (1.7B, VL-2B, d1-2B) | Liquid AI. *Introducing LFM3: Every Modality for Specialized Intelligence*. Blog, 2026-10-27 (no paper) | No model-code repo; HF `transformers` remote code, [https://huggingface.co/collections/LiquidAI/lfm3;](https://huggingface.co/collections/LiquidAI/lfm3;) notebooks [https://github.com/Liquid4All/cookbook](https://github.com/Liquid4All/cookbook) | HF blog: [https://huggingface.co/blog/LiquidAI/lfm3](https://huggingface.co/blog/LiquidAI/lfm3) | R, P (blog) |
| 34 | LFM3-Omni-2B / LFM3-Audio-2B | Same blog as 33 (no paper) | [https://huggingface.co/LiquidAI/LFM3-Omni-2B](https://huggingface.co/LiquidAI/LFM3-Omni-2B), [https://huggingface.co/LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) |  | R, P (blog) |

### Shared dependencies

| Dependency | Used by | Paper | Repository | Verified |
| --- | --- | --- | --- | --- |
| DINOv2 | 1, 16 | Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision*. TMLR 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |  |
| Wan 2.1 (VAE + DiT) | 13, 15 | Wan Team. *Wan: Open and Advanced Large-Scale Video Generative Models*. 2025. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314) | [https://github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | P |
| SigLIP 2 (NaFlex) | 14, 17 | See entry 19 | [https://github.com/google-research/big_vision](https://github.com/google-research/big_vision) | P |
| FramePack | 15 | Zhang, Agrawala. *Packing Input Frame Context in Next-Frame Prediction Models for Video Generation*. 2025. [arXiv:2504.12626](https://arxiv.org/abs/2504.12626) | [https://github.com/lllyasviel/FramePack](https://github.com/lllyasviel/FramePack) |  |
| Flow matching | 13, 15 | Lipman et al. *Flow Matching for Generative Modeling*. ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) | n/a | P |
| timm | 2, 3, 7, 8 | n/a | [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |  |
| MiniCPM-V | 17 | Yao et al. *MiniCPM-V: A GPT-4V Level MLLM on Your Phone*. 2024. [arXiv:2408.01800](https://arxiv.org/abs/2408.01800). Follow-up: *MiniCPM-V 4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe*. 2025 | [https://github.com/OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) | R, P |
| LLaVA-UHD | 17 | Guo et al. *LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images*. ECCV 2024. [arXiv:2403.11703](https://arxiv.org/abs/2403.11703) | [https://github.com/thunlp/LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | P (title / venue) |
| Whisper | 17 | Radford et al. *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML 2023. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) | [https://github.com/openai/whisper](https://github.com/openai/whisper) |  |
| Qwen3 | 17 | Qwen Team. *Qwen3 Technical Report*. 2025. [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) | [https://github.com/QwenLM/Qwen3](https://github.com/QwenLM/Qwen3) |  |
| CosyVoice / CosyVoice 2 | 17 | Du et al. *CosyVoice*. 2024. [arXiv:2407.05407](https://arxiv.org/abs/2407.05407). *CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models*. 2024. [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | [https://github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | P |
| SigLIP | 14, 17 | Zhai et al. *Sigmoid Loss for Language Image Pre-Training*. ICCV 2023. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | [https://github.com/google-research/big_vision](https://github.com/google-research/big_vision) |  |
| DINOv3 | 18 (teacher) | Simeoni et al. *DINOv3*. 2025. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) | [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | P (title / year) |
| RT-DETR / RT-DETRv2 | 18 (encoder) | Zhao et al. *DETRs Beat YOLOs on Real-time Object Detection*. CVPR 2024. [arXiv:2304.08069](https://arxiv.org/abs/2304.08069). Lv et al. *RT-DETRv2*. 2024. [arXiv:2407.17140](https://arxiv.org/abs/2407.17140) | [https://github.com/lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | P (titles / venues) |
| D-FINE | 18 (losses) | Peng et al. *D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement*. ICLR 2025. [arXiv:2410.13842](https://arxiv.org/abs/2410.13842) | [https://github.com/Peterande/D-FINE](https://github.com/Peterande/D-FINE) | P (title / venue) |
| DEIM / DEIMv2 | 18 (cls loss, sibling) | Huang et al. *DEIM: DETR with Improved Matching for Fast Convergence*. CVPR 2025. [arXiv:2412.04234](https://arxiv.org/abs/2412.04234). Huang et al. *Real-Time Object Detection Meets DINOv3*. 2025. [arXiv:2509.20787](https://arxiv.org/abs/2509.20787) | [https://github.com/Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM), [https://github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | P (titles / venues) |
| Deformable DETR | 18 (decoder) | Zhu et al. *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR 2021. [arXiv:2010.04159](https://arxiv.org/abs/2010.04159) | [https://github.com/fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) | P (title / venue) |
| Conv stem | 18 | Xiao et al. *Early Convolutions Help Transformers See Better*. NeurIPS 2021. [arXiv:2106.14881](https://arxiv.org/abs/2106.14881) | n/a | P (title / venue) |
| Registers | 18 | Darcet et al. *Vision Transformers Need Registers*. ICLR 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) | n/a |  |
| LARS | 18 (distillation optimizer) | You, Gitman, Ginsburg. *Large Batch Training of Convolutional Networks*. 2017. [arXiv:1708.03888](https://arxiv.org/abs/1708.03888) | n/a | P (title / year) |