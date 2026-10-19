# CliffordNet: Receptive Field Options

Oct 16, 2026 · @Nikolas Markou

CliffordNet's spatial receptive field (RF) comes only from its 3x3 depthwise context and depth. MetaFormer and PoolFormer evidence points to three levers that matter most: content-dependent global mixing in late layers (+3.8 points in PoolFormer, CAFormer's ImageNet record), a larger-RF stem (+7.2 points for isotropic IdentityFormer), and learned rather than wider fixed local kernels. The same evidence warns that the token mixer is second-order to the block structure, so expect modest gains on CIFAR's 16x16 token grid and test the best options at higher resolution.

## Baseline: where CliffordNet's RF comes from

Spatial RF in CliffordNet ([arXiv 2601.06793](https://arxiv.org/abs/2601.06793)) grows only through the stem, the local context operator and depth; channel rolling adds none.

- **Stem.** Patch embedding is a conv with kernel = stride = P (P = 2 on CIFAR), so each token sees a non-overlapping 2x2 pixel patch. A 32x32 image becomes a 16x16 token grid that stays fixed through all layers (isotropic design).
- **Local context.** `C_loc` stacks two 3x3 depthwise convs, a 5x5 footprint per block. Theoretical RF after L blocks is 1 + 4L tokens, which covers the 16x16 grid after 4 blocks. The paper calls this "e.g., 7x7", which two 3x3 kernels do not give; Algorithm 1 shows a single DWConv.
- **Channel rolling.** `T_s` shifts along the channel axis D, not H or W. It pairs channels at the same spatial position, so the paper's claim that rolling captures long-range dependencies does not hold spatially.
- **gFFN-G.** GlobalAvgPool gives a global RF, but as one mean vector with no spatial layout. It lifts Nano from 77.82% to 78.22% at 1.43M to 2.22M params.
- **Effective vs theoretical RF.** With stacked small kernels, effective RF grows roughly with sqrt(L), well below 1 + 4L.

## MetaFormer and PoolFormer: premises and metrics

Both papers argue that the block structure, not the token mixer, carries most of the performance, and both quantify how much spatial mixing and RF add on top.

- **MetaFormer is Actually What You Need for Vision (PoolFormer, CVPR 2022).** A MetaFormer block is Norm, token mixer, residual, then Norm, channel MLP, residual; the token mixer is left unspecified. PoolFormer sets it to parameter-free 3x3 average pooling minus the input (stride 1) inside a 4-stage hierarchy (H/4 to H/32 tokens) and still beats DeiT and ResMLP.
- **MetaFormer Baselines for Vision (TPAMI).** MetaFormer has a solid lower bound (identity mixer, IdentityFormer, above 80%), works with arbitrary mixers (frozen random mixing, RandFormer, above 81%), and reaches state of the art with ordinary mixers: ConvFormer (inverted separable conv, 7x7 depthwise) beats ConvNeXt, and CAFormer (conv in stages 1 to 2, vanilla attention in stages 3 to 4) reaches 85.5% at 224 on ImageNet-1K.

All results are ImageNet-1K top-1 at 224 unless noted.

| Experiment | Result | RF reading |
| --- | --- | --- |
| PoolFormer-M36 vs DeiT-B and ResMLP-B24 | 82.1% at 56M vs 81.8% at 86M and 81.0% at 116M | A 3x3 fixed mixer suffices when the hierarchy supplies RF |
| PoolFormer S12 / S24 / S36 / M36 / M48 | 77.2 / 80.3 / 81.4 / 82.1 / 82.5% at 12 / 21 / 31 / 56 / 73M | Scaling depth and width, not mixer RF, drives gains |
| S12 mixer swap: identity / frozen global random / pooling / depthwise conv | 74.3 / 75.8 / 77.2 / 78.1% | Any spatial mixing helps; learned local beats fixed local beats random global |
| S12 pooling size 3 / 5 / 7 / 9 | 77.2 / 77.2 / 77.1 / 76.8% | Wider fixed averaging adds nothing and hurts at 9 |
| S12 global mixer in late stages: \[P,P,P,A\] / \[P,P,A,A\] / \[P,P,P,FC\] / \[P,P,FC,FC\] | 78.3% (14.0M) / 81.0% (16.5M) / 77.5% / 77.9% | Late global mixing pays most when content-dependent (attention +3.8 vs spatial FC +0.7) |
| S12 component removal: channel MLP / residual / norm | 5.7% / 0.1% / 46.1% | Block structure dominates any mixer change |
| M48: IdentityFormer / RandFormer / PoolFormerV2 | 80.4% / 81.4% / 82.6% | Downsampling alone gives 80%+; random late global adds 1.0; local pooling adds 1.2 more |
| Isotropic IdentityFormer-S, patchify stem vs 4-layer conv stem (DeiT-S: 79.8%) | 68.2% vs 75.4% | Stem RF is a major factor (+7.2); the authors say so directly |
| ConvFormer vs CAFormer (attention in last 2 stages), S18 / B36 | 83.0 vs 83.6% / 84.8 vs 85.5% | Late attention over a 7x7 conv base adds 0.6 to 0.7 |
| ADE20K UperNet mIoU: ConvNeXt-T / ConvFormer-S18 / CAFormer-S18 | 46.7 / 48.6 / 48.9 | On dense prediction, late global attention adds little over a strong local mixer |

## Implications for CliffordNet

Read through MetaFormer, CliffordNet is a MetaFormer with a fused mixer, and RF changes touch only its token-mixing half.

- **It is a MetaFormer, not FFN-free channel mixing.** `C_loc` is the token mixer. `Linear_det`, the rolled dot/wedge product, `Linear_proj` and the gate do the channel mixing that PoolFormer shows is indispensable (S12 falls to 5.7% without it). "No FFN" means no separate MLP block, not no channel MLP.
- **Differential mode is a learned PoolFormer mixer.** `C = C_loc(H) - H` mirrors PoolFormer's `AvgPool(X) - X`; both are discrete Laplacians. Learning the kernel helps in both: depthwise conv beats pooling by 0.9 in PoolFormer, and diff beats abs mode by about 1.4 in CliffordNet.
- **Expect modest returns from RF alone.** Mixer swaps move PoolFormer-S12 within a 3.8-point band (74.3 to 78.1%), while removing block components collapses it. On a 16x16 grid that 1 + 4L already covers after 4 blocks, gains will come mostly from effective RF and early globality.
- **Wider windows need learned weights.** Fixed pooling is flat to negative from 3 to 9; ConvFormer's gains come from learned 7x7 kernels.
- **Global mixing belongs late and should depend on content.** Attention in the last two stages adds 3.8 over pooling; a spatial FC adds 0.7. In an isotropic net, "late stages" means the last third of blocks. gFFN-G sits in between: a layout-free mean, made content-dependent only through the product.
- **The stem is an RF lever CliffordNet ignores.** A conv stem lifts isotropic IdentityFormer by 7.2 points; CliffordNet's stem is a non-overlapping 2x2 patchify.
- **Transfer caveat.** MetaFormer numbers are ImageNet-1K at 224, mostly hierarchical; CliffordNet is isotropic on CIFAR-100. Treat them as direction, not predicted deltas.

## Methods

Nineteen ways to grow RF, ordered from stem to global; each plugs in as the context `C(H)`, around it, or at the stem. Cost is in N = h x w tokens and D channels. External evidence is from the two MetaFormer papers unless noted.

| Category | Method | Cost | Evidence and notes |
| --- | --- | --- | --- |
| Stem | Overlapping conv stem (e.g., stacked 3x3 convs) before or instead of the 2x2 patchify | One-off, O(N k^2 D) | +7.2 for isotropic IdentityFormer; cheapest strong lever |
| Kernel size | Learned large DW kernels (7x7 to 31x31) in `C_loc` | O(N k^2 D) | ConvFormer uses 7x7; RepLKNet-style parallel small kernel and reparam aid training. Fixed pooling 3 to 9 does not help (77.2 to 76.8) |
| Kernel size | Deeper factorized stack (3 or 4 DW 3x3) | O(N 9 D) per conv | Same idea as the paper, more stages |
| Dilation | Dilated DW 3x3 in `C_loc`, rates scheduled across layers (1, 2, 4, 8) | Same as 3x3 | RF grows exponentially with depth; mix rates (HDC style) to avoid gridding; mirrors the exponential shift set, in space |
| Subsampling | Compute `C` on a pooled or strided map, upsample before the product | Below baseline | Keeps the isotropic `H` stream intact; only the context is coarse |
| Subsampling | Hierarchical stages or larger patch size | Below baseline | Hierarchy alone gives IdentityFormer-M48 80.4% with no token mixing, but it breaks the isotropic design and fixed-phase-space flow; larger patches lose detail on 32x32 CIFAR |
| Axial | Strip or axial convs (1xk + kx1), or row/column mean pooling | O(N D) | Full row and column reach per block |
| Frequency | FFT global filter as `C` (GFNet style) | O(N D log N) | True global RF, keeps 2D topology, no serialization |
| Frequency | Damped-cosine global kernel as `C` via FFT | O(N D log N) | About 3 params per channel group; content-independent on its own (see next section) |
| Placement | Global context only in the last third of blocks | Below all-layer global | PoolFormer \[P,P,A,A\] +3.8; CAFormer +0.6 to 0.7; applies to any global option below |
| Global, content-dependent | Linear attention as `C` | O(N D) | Closest linear analogue of the attention that wins late; CliffordNet lists it but never tests it |
| Global, content-dependent | Full softmax attention in late blocks | O(N^2 D), N = 256 on CIFAR | Breaks strict linearity but cheap at 16x16; direct CAFormer analogue |
| Global | Learned global tokens or registers that `H` interacts with | O(N k D) | Richer than a single average vector |
| Global, control | Frozen random global mixing (RandFormer) in late blocks | O(N^2 D), frozen | Lower bound for any global option: +1.0 to +2.0 over identity |
| Iteration | K weight-shared Laplacian steps per block (multiple Euler substeps) | No extra params | Fits the paper's ODE framing; no external evidence |
| Spatial-channel product | Wedge/dot between `H_i` and `C_(i+offset)` (spatial roll inside the product) | O(N D \|S\|) | Extends the shift idea to space; sparse offsets edge toward dilation |
| Data-dependent | Deformable conv as `C(H)` | O(N k^2 D) | Matches the paper's "dynamic metric" future work |
| Non-local | kNN graph in feature space (ViG style) as `C` | O(N k D) + kNN search | Semantic, not geometric, neighbors |
| Depth | More layers | O(N D) per layer | Theoretical RF grows linearly; effective RF only about sqrt(L) |

## Candidate global context: damped-cosine kernel

A 2D radial damped-cosine kernel, applied by FFT, gives a global, spatially resolved context for about 3 parameters per channel group; the geometric product supplies the content dependence the kernel lacks.

```latex
k(r) = e^{-\alpha r} \cos(\omega r + \varphi), \quad r = \lVert (\Delta x, \Delta y) \rVert, \quad \mathcal{C}_{wave} = k * H
```

- **Origin.** Adapted from Wave Field attention, a 1D language model whose per-head kernel is one S4D mode (RetNet decay-with-rotation without Q·K). On its own it is linear time-invariant and fails recall-heavy tasks (Lambada 1.1%).
- **Why the pairing works.** CliffordNet's dot (H ⊙ T\_s(C)) and wedge are multiplicative and content-dependent, the same remedy Hyena applied to long convolutions. This matches MetaFormer's finding that late global mixing pays most when content-dependent.
- **Where it plugs in.** As a third term in the paper's superposition (Eq. 7), next to `C_loc` and `C_glo`, in the last third of blocks.
- **Versus gFFN-G and GFNet.** Unlike gFFN-G it keeps spatial layout at near-zero parameter cost. Unlike a free GFNet filter it is continuous in r, so it is resolution-independent.
- **Caveats.** A radial kernel has no orientation (use separate x and y frequencies if needed). FFT wraps at borders; zero-pad to 2h x 2w if edges matter. Images are not causal, so no masking is needed.

## Recommendations and ablation plan

Try the stem first, then late content-dependent global context, then learned wider kernels; all keep the isotropic stream and native 2D topology.

1. **Overlapping conv stem.** Strongest external evidence (+7.2 on isotropic IdentityFormer) at a one-off cost.
2. **Content-dependent global context in the last third of blocks.** Linear attention keeps O(N); an FFT filter or damped-cosine kernel through the geometric product keeps O(N log N) with few parameters. Evidence: PoolFormer \[P,P,A,A\] +3.8, CAFormer +0.6 to 0.7.
3. **Learned 7x7 depthwise kernel in `C_loc`.** ConvFormer's mixer; fixed wider averaging does not help.
4. **Iterated Laplacian substeps.** Zero new parameters and native to the ODE framing, but no external evidence; lower priority.

Ablation plan on CIFAR-100, Nano(diff), same 200-epoch recipe:

- **Controls.** `C_loc` replaced by identity (IdentityFormer analogue: how much spatial mixing matters at all), and frozen random global mixing in the last third of blocks (RandFormer analogue: the floor any global option must beat).
- **Arms.** Conv stem; learned 7x7 DW; late global context (linear attention, FFT filter, damped-cosine kernel) vs gFFN-G; dilation schedule.
- **Report.** Top-1, params, training time, and effective-RF maps from input gradients.
- **Scale check.** Rerun the best two arms at higher resolution (e.g., ImageNet-100 at 224), where RF limits bite and MetaFormer numbers are comparable.

## Sources

- [CliffordNet: All You Need is Geometric Algebra (arXiv 2601.06793v2)](https://arxiv.org/html/2601.06793v2)
- [MetaFormer Is Actually What You Need for Vision (arXiv 2111.11418)](https://ar5iv.labs.arxiv.org/html/2111.11418)
- [MetaFormer Baselines for Vision (arXiv 2210.13452v4)](https://arxiv.org/html/2210.13452v4)
- [Wave Field LLM, Hugging Face forum post](https://discuss.huggingface.co/t/wave-field-llm-o-n-log-n-attention-via-wave-equation-dynamics-within-5-of-standard-transformer/173625) and [BENCHMARKS.md](https://github.com/badaramoni/wave-field-llm/blob/main/BENCHMARKS.md)

The S4D, RetNet, Hyena, GFNet, RepLKNet and HDC comparisons are from memory.
