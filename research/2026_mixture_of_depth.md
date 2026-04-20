# Mixture-of-Depths Attention (MoDA): A Technical Report

**Report date:** 2026-04-20
**Primary sources:**
- Zhu et al., *Mixture-of-Depths Attention*, arXiv:2603.15619v1 (2026-03-16). HUST + ByteDance Seed.
  GitHub: https://github.com/hustvl/MoDA
- Lianghui Zhu, *The Second Half of Model Architecture* (companion essay, 2026-04-11).

**Scope:** This report focuses on MoDA — the depth-attention architecture proposed by Zhu et al. — and contextualizes it against prior inter-layer communication mechanisms. It is not about the 2024 DeepMind "Mixture-of-Depths" work of Raposo et al., which uses the same phrase to describe per-token *compute routing* (skipping layers for some tokens). Both lines share the intuition that depth is a resource, but they act on different axes: Raposo et al. skip compute; Zhu et al. enrich communication. See §8 for a disambiguation.

---

## 1. Motivation: the depth-communication bottleneck

The field spent a decade scaling four axes of Transformer training — parameters, data, sequence length, and depth. Three of them scaled in quality as well as quantity. Depth did not.

The residual primitive `x_{l+1} = x_l + F(x_l)` introduced by ResNet (2015) is still the dominant inter-layer interface. Everything added since — norm placement (Pre-LN, Post-LN, DeepNorm), scaled residuals, DenseNet-style concat, DenseFormer / LIMe learnable weights, ByteDance Hyper-Connections, DeepSeek mHC, MUDDFormer — either (i) blends layer outputs with better coefficients, (ii) widens the residual pipe to multiple channels, or (iii) generates blending weights dynamically from the current hidden state.

Zhu et al. characterize all of these as *accumulation-frame* methods:

> Layer `l` decides how much of each prior layer's contribution to absorb into a single running hidden state, based only on its own state. The sources themselves have no voice in the decision.

The alternative is a *retrieval frame*: layer `l` emits a query, every earlier layer exposes keys and values describing what it holds, and content-based matching (dot-product attention) selects what to read. The query encodes "what is needed"; the keys encode "what is available." This is the same category shift that sequence attention brought to token communication in 2017 — and the argument is that the depth dimension never received it.

### 1.1 Why accumulation loses information

Two concrete failure modes motivate the retrieval frame:

1. **Information dilution in the residual stream.** After `L` layers, `x_L = x_0 + Σ F_i(x_i)`. Each addition competes for a finite-capacity channel. Layers that would write something useful often learn to stay silent to avoid overwriting prior information; the model becomes "nominally deep, effectively shallow." ELMo (2018) already showed different layers encode qualitatively different information (shallow = syntax, deep = semantics), yet downstream layers must read them all through a single summed trajectory.
2. **Attention-sink as a symptom.** The well-documented attention-sink phenomenon — models dumping probability mass onto a handful of fixed tokens (often BOS) — is reinterpreted here as the sequence mechanism absorbing pressure that the depth channel cannot handle. MoDA's ablations show the sink diminishes once depth retrieval is available.

### 1.2 The design-space framing

Zhu et al. structure the design space around three operations each inter-layer mechanism must perform: **read** (how does layer `l` access earlier information?), **operate** (what transformation is applied?), **write** (how is the output exposed to later layers?). Under this lens:

| Mechanism | Read | Operate | Write | Data-dependent? |
|-----------|------|---------|-------|-----------------|
| Residual | sum previous hidden | `F(x)` | overwrite `x` | no |
| DenseNet | concat all prior | `F([x_0..x_l])` | append | no |
| DenseFormer / LIMe | weighted sum | learned static weights | overwrite | no (fixed after training) |
| Hyper-Connections / mHC | `n`-channel mix | mixing matrix | append channels | no (structure fixed) |
| MUDDFormer | dynamic blend | weights from `x_l` | overwrite | partially (query side only) |
| Depth Attention | attend to layer KVs | dot-product softmax | append KV | yes (both sides) |
| **MoDA (proposed)** | **joint attend to seq + depth KV** | **unified softmax** | **append KV** | **yes (both sides)** |

The key phrase: *both sides get a voice*.

---

## 2. Prior art on depth attention

Several groups converged on "attention across depth" independently in 2024-2025, which Zhu et al. treat as corroboration that the concept was right:

- **Google DCA** (Depth-wise Cross-Attention)
- **Huawei MRLA** (Multi-layer Retrieval Attention)
- **Hessian.AI Dreamer**
- **Kimi AttnRes**

All four apply dot-product attention over a depth-KV cache. None shipped at scale. The blocker was not the idea; it was the kernel. A naive PyTorch implementation of the forward-backward pass measured **44,924 ms** on a representative workload — roughly 1400× slower than the FlashAttention-2 baseline it needed to replace. The field faced a trilemma:

- Simplify depth attention for speed → lose selective retrieval (collapses back toward accumulation).
- Keep full expressivity → impractical to train.
- Engineer a hardware-aware kernel that preserves expressivity → open problem.

MoDA is the third path: an algorithmic-plus-systems contribution, where the algorithmic payload (unified sequence+depth attention under a single softmax) is co-designed with a three-stage kernel optimization (Flash Depth Attention).

---

## 3. The MoDA operator

### 3.1 Notation

- `T`: sequence length (query tokens)
- `L`: layer depth at which MoDA is applied (number of preceding layers contributing depth KV)
- `D`: hidden dimension
- `H_q`, `H_k`: query / key-value head counts
- `G = H_q / H_k`: GQA group size
- `d = D / H_q`: per-head dimension
- `C`: query-chunk size in the kernel

### 3.2 Baseline sequence attention (GQA)

For reference, standard grouped-query attention at layer `l`:

```
Q = X W_Q,   K = X W_K,   V = X W_V
head_h   = softmax( Q_h K_{φ(h)}^T / √d + M_seq ) V_{φ(h)}
output   = Concat_h(head_h) W_O
```

with `φ(h) = ⌊h / G⌋` the head-grouping map and `M_seq` the causal mask.

### 3.3 Depth-KV cache

Each preceding layer `l' < l` exposes a `(K^{(l')}, V^{(l')})` pair per token. Aggregated across layers, the depth-KV cache at layer `l` has shape `(T, L, H_k, d)` for keys and values. MoDA reuses the sequence-attention KV projections by default (no new parameters on the attention side); a separate projection generates additional depth KV from the FFN input (see §4.2).

### 3.4 Joint softmax over sequence and depth KV

For query `q_i` at token position `i` and head `h`, let:

- `K^seq_i = { K^{(l)}_j : j ≤ i }` — the `O(i)` current-layer keys visible under the causal mask.
- `K^depth_i = { K^{(l')}_{i,l'} : l' < l }` — the `L` depth keys at *this same token position* from preceding layers. (Depth retrieval is per-token: layer `l` retrieves from layer `l' < l`'s *own* KV for token `i`, not across tokens × layers.)

MoDA concatenates them and applies a single softmax:

```
logits_i = [ q_i K^seq_i^T , q_i K^depth_i^T ] / √d  + M_MoDA
α_i      = softmax(logits_i)
o_i      = α_i · [ V^seq_i ; V^depth_i ]
```

The single softmax is load-bearing. Under one normalizer, the model *freely trades* between "look back in the sequence" and "look back in depth" — probability mass is reallocated based on which source is more relevant, rather than each source being normalized independently and then blended.

### 3.5 Masking

MoDA uses a *grouped causal* mask for the sequence portion:

- Sequence mask: `⌊i_q / G⌋ ≥ i_k` (standard GQA-compatible causal).
- Depth mask: allow attention to depth slot `(i, l')` only when `⌊i_q / G⌋ = ⌊j_d / L⌋` — i.e., a query only looks at the depth stack belonging to its own base-time index. This ties each query to a specific depth column rather than the full `T × L` grid, and is the key to §4's efficiency gains.

### 3.6 Write step

- The current layer's `(K, V)` are appended to the depth cache for future layers.
- FFN input is *also* projected into depth KV by a lightweight per-layer projection (this is where MoDA spends most of its extra parameters).

### 3.7 Complexity summary (Table 1 of the paper)

| Quantity | Depth Dense | Plain Depth Attn | MoDA |
|---|---|---|---|
| Params (extra) | `O(L² D²)` | `O(L D²)` | `O(L D² / G)` |
| Decoding KV cache | `O(L D)` | `O(L D / G)` | `O(L D / G)` |
| Prefill FLOPs | `O(T L² D²)` | `O(T L² D)` | `O(T L² D)` |
| Data-dependent? | ✗ | ✓ | ✓ |
| Unified softmax? | ✗ | ✗ | ✓ |

Depth dense is quadratic in both depth and width and is impractical. Plain depth attention matches MoDA asymptotically but lacks the unified softmax. MoDA's `1/G` factor on the parameter count comes from reusing the GQA-grouped KV projections.

---

## 4. Flash Depth Attention (FDA): the kernel

MoDA would not be trainable without the kernel. FDA delivers **1458× total speedup** over naive PyTorch (2128.9 ms → 1.46 ms on the reference setup) and reaches **97.3% of FlashAttention-2 efficiency at 64K sequence length**. Three layered optimizations:

### 4.1 Flash-compatible depth-KV layout

**Problem:** A naive implementation stores depth KV as a 4-D tensor `(T, L, H_k, d)` and issues a scatter/gather per query token, killing memory bandwidth.

**Fix:** Flatten the depth cache along a single `T × L` axis so that for each query chunk the relevant depth KV is a *contiguous* block on device memory. Standard Flash-style tiling (load into SRAM, accumulate online softmax) then applies directly.

Impact: **2128.9 ms → 13.1 ms (162.5× speedup)** from this layout alone.

### 4.2 Chunk-aware depth-KV access

**Problem:** A query chunk of size `C` does not need the full `T × L` depth span. Under the mask in §3.5 each query looks at a *depth column of height `L` at its own token index*, so a chunk of `C` queries needs only `C × L` depth entries.

**Fix:** Index the depth cache by query chunk. Utilization improves from `1/T` to `1/C`.

Impact: **13.1 ms → 6.3 ms (52% further reduction)**.

### 4.3 Group-aware indexing

**Problem:** Under GQA, `G` adjacent query rows share a base-time index `⌊i_q / G⌋`. They therefore attend to *the same* depth column.

**Fix:** Share the depth-KV load across the `G` queries in a group. The effective depth span per chunk shrinks from `C × L` to `(C/G) × L`.

Impact: **6.3 ms → 1.46 ms**. Combined speedup vs naive PyTorch is 1458×.

### 4.4 Forward kernel sketch (Algorithm 1)

Per query chunk (size `C`, loaded once into SRAM):

1. Initialize online-softmax state `(m, acc, o)` with `m = -∞`, `acc = 0`, `o = 0`.
2. Loop over fully-visible sequence blocks (no mask) — stream KV in, accumulate.
3. Loop over boundary sequence blocks — apply grouped causal mask.
4. Loop over depth blocks — apply the depth mask (§3.5), accumulate into the same `(m, acc, o)` state.
5. Final normalization: `o ← o / acc`.
6. Write unnormalized `O ∈ ℝ^{T_q × (H_k · d)}`.

The unified softmax is implemented as the *same* online-softmax state being updated across both the sequence passes and the depth pass, which is what makes the "single normalization" property of MoDA a real property of the kernel and not just the math.

### 4.5 Empirical efficiency (Table 2)

All measurements on A100, bf16, forward+backward, relative to FlashAttention-2.

**Sequence length scaling (`G=8, L=64, C=64`):**

| `T` | Extra time vs FA2 |
|---|---|
| 4K | +25.86% |
| 16K | +8.59% |
| 65K | +2.73% |

Overhead is amortized at long context — MoDA's penalty goes away exactly where long-context training matters.

**GQA group size scaling (`T=16K, L=64`):**

| `G` | Extra time | Depth utilization |
|---|---|---|
| 2 | +27.07% | 3.12% |
| 32 | +2.84% | 50.00% |

Larger groups help (higher utilization → less wasted bandwidth), motivating MoDA deployments that use large `G`.

**Depth scaling (`T=16K, G=8`):**

| `L` | Extra time |
|---|---|
| 64 | +8.59% |
| 256 | +30.52% |

Roughly linear in `L`, matching the `O(T L² D)` analysis (the second `L` is amortized across queries).

---

## 5. Training and evaluation

### 5.1 Setup

- **Models:** 700M and 1.5B decoder-only Transformers built on the OLMo2 recipe.
- **Data:** 400B-token OLMo2 corpus subset. Ablations use FineWeb-Edu.
- **Precision:** bf16.
- **Batch size / seq len:** 1024 × 4096.
- **Optimizer:** AdamW.
- **LR schedule:** warmup to 3e-4 over 2k steps, cosine decay to 3e-5.

### 5.2 Downstream benchmarks

PiQA, HellaSwag, WinoGrande, OpenBookQA, BoolQA, SciQ, COPA, MMLU, ARC-Easy, ARC-Challenge.

### 5.3 700M ablations (Table 3)

| Variant | Train PPL | C4 val PPL | DS avg | Params (M) | FLOPs (T) |
|---|---|---|---|---|---|
| OLMo2 baseline (36 layers) | 14.49 | 18.59 | 56.93 | 669.0 | 8.01 |
| + Depth KV only (reuse attn proj) | 14.08 | 18.48 | 58.10 | 669.0 | 8.02 |
| + Depth KV + FFN proj | 13.90 | 18.21 | 58.87 | 705.7 | 8.33 |
| + Extra attn proj | 13.83 | 18.17 | 58.97 | 742.4 | 8.63 |

**Read:**

- Just reusing the existing KV projections as depth KV (no parameter cost) already buys **+1.17 DS avg** and 0.41 train-PPL.
- Adding an FFN-side depth KV projection is the best accuracy-per-parameter step (+0.77 DS avg, +36.7M params).
- Adding attention-side depth projection saturates (+0.10 DS avg for +36.7M params). The recommended default is **row 3: reuse attention KV + FFN projection**.

### 5.4 1.5B scale (Table 4, downstream)

| Model | HellaSwag | WinoGrande | ARC-C | MMLU | Avg |
|---|---|---|---|---|---|
| OLMo2 1.5B | 65.86 | 63.22 | 42.47 | 27.73 | 62.28 |
| MoDA 1.5B | 66.24 | 65.59 | 46.82 | 29.59 | 64.39 |
| Δ | +0.38 | +2.37 | **+4.35** | +1.86 | **+2.11** |

The improvement on ARC-C (reasoning over scientific text) and WinoGrande (pronoun resolution requiring long-distance coreference) is outsized. Both are tasks where "reach back to an earlier layer's specific representation" is a plausible failure mode for residual-only models.

### 5.5 1.5B validation PPL (Table 5, selected)

| Domain | OLMo2 | MoDA | Δ |
|---|---|---|---|
| C4 | 16.16 | 15.97 | -0.19 |
| Reddit | 21.21 | 20.85 | -0.36 |
| Wiki-text | 10.41 | 10.16 | -0.25 |
| **10-domain average** | 13.67 | 13.47 | -0.20 |

### 5.6 Layer count and norm placement (Table 6, FineWeb-Edu, 48 layers)

| Variant | Train loss |
|---|---|
| Baseline (pre-norm) | 3.3800 |
| + Depth KV (pre-norm) | 3.3759 (-0.0041) |
| + Depth KV (post-norm) | 3.3653 (-0.0147) |
| + FFN projection (post-norm) | 3.3484 (-0.0316 vs baseline) |

Post-norm benefits from MoDA substantially more than pre-norm, which is consistent with the hypothesis that MoDA addresses signal dilution that is worse in post-norm stacks. This is an actionable finding: at 48+ layers, post-norm + MoDA is the recommended combination.

### 5.7 Attention-mass analysis (Figure 5)

Visualizations split attention probability between the sequence region (left) and depth region (right). Findings:

- Substantial and consistent mass on depth KV across heads and layers — it is not a vestigial channel.
- Heads with sharp diagonal sequence patterns *still* allocate probability to depth slots, i.e., the two are complementary, not redundant.
- The BOS-style attention sink is visibly reduced: probability mass that would otherwise dump onto position 0 is absorbed by meaningful depth entries.

### 5.8 Kernel ablation (Table 7)

| Kernel | Time |
|---|---|
| Naive PyTorch | 2128.9 ms |
| + Flash-compatible layout | 13.1 ms (162.5×) |
| + Chunk-aware | 6.3 ms (total 338×) |
| + Group-aware (final FDA) | 1.46 ms (total 1458×) |

---

## 6. What MoDA actually changes at layer level

The conventional Transformer block, viewed in detail, is:

```
residual → sequence-attention → residual → FFN
```

Once depth retrieval is cheap, each layer's *effective* pipeline becomes:

```
depth-attn → sequence-attn → depth-attn → FFN
```

i.e., three attention operations over different KV sets sharing almost the same query. The natural move is **fusion**: collapse the three into one unified sequence+depth attention, which is exactly MoDA. The FFN still gets its own depth-KV projection on the input side (see §5.3), but the attention sub-layer is now a single op.

This matters for a subtle reason. Doing depth attention and sequence attention *separately* and then summing would reintroduce an accumulation step — you'd be adding two independently-normalized outputs, with all of §1.1's problems in miniature. The unified softmax is what prevents MoDA from collapsing back into the accumulation frame at the fused-op level.

---

## 7. Implementation considerations

For a Keras 3 / dl_techniques port, the following points from the paper are load-bearing:

1. **Head layout.** GQA with large `G` is both a compute win (kernel utilization) and a parameter win (MoDA's `1/G` factor). Prefer `G ≥ 8`.
2. **Depth-KV source.** Reuse the sequence attention's `K, V` projections as the default depth-KV source. Add a dedicated FFN-input projection only if the parameter budget allows — it's the next 0.8 DS-avg point.
3. **Depth span `L`.** The paper uses `L = 64` as the standard. `L = 256` is still viable with a +30% kernel cost; below `L = 16` the retrieval mechanism starves.
4. **Norm placement.** Use post-norm for stacks of 48+ layers when running MoDA; the layer-count ablation is unambiguous.
5. **Mask.** Grouped causal for sequence, per-token column for depth. Do not let queries attend across depth columns of other tokens — it breaks the chunk-aware kernel and has no modeling justification.
6. **Softmax.** A *single* online-softmax state across both sequence and depth passes is required. Two independently-softmaxed outputs that are summed are *not* MoDA.
7. **Decoding cache.** At inference, the depth cache grows with layer count, not with sequence length beyond what GQA already stores. Per-token decoding cost is `O(L D / G)`.
8. **Memory at scale.** The authors explicitly flag that for very deep / very large models, the full depth-KV cache becomes the dominant memory term. Mitigations they suggest but do not yet ship: bounded `S ≪ L` depth slot buffer, dynamic slot selection, sliding-window policies, hybrid recency + utility schemes. Treat these as open problems for a production port.

---

## 8. Disambiguation: MoD (2024) vs MoDA (2026)

| | Raposo et al. 2024 — *Mixture-of-Depths* (DeepMind) | Zhu et al. 2026 — *Mixture-of-Depths Attention* (HUST + ByteDance) |
|---|---|---|
| What varies | Which *tokens* get processed at each layer | Which *layers* a token's attention reaches back to |
| Mechanism | Top-k router per layer decides which tokens pass through the block; the rest skip | Single-softmax attention over current-layer sequence KV + preceding-layers depth KV |
| Goal | Save FLOPs by skipping unimportant tokens | Improve expressivity by replacing residual accumulation with retrieval |
| Axis of action | Sequence axis (token selection) | Depth axis (layer selection) |
| Residual connection | Unchanged | Complemented by a parallel depth-KV retrieval path |
| Cost profile | Lower FLOPs per layer | Slightly higher FLOPs per layer, amortized at long context |

The two ideas are orthogonal and in principle composable: a MoD router could decide *whether* a token traverses a block, while MoDA decides *how* that block reads prior layers. The Zhu et al. paper does not discuss composition, and this report is not aware of published results combining them.

---

## 9. Open questions and limitations (per §6 of the paper and independent reading)

1. **Trillion-parameter regime.** The current FDA kernel is not tuned for the memory scheduling, pipelining, and NVLink traffic patterns of trillion-param training. The authors explicitly call this out.
2. **Depth-cache memory.** Growing linearly with `L`, the depth cache dominates memory at very deep networks. Bounded-slot variants are unimplemented.
3. **Static `L`.** The current design attends to a fixed `L` preceding layers. Whether learned per-token `L` (a MUDDFormer-style dynamic depth budget) helps on top of MoDA is untested.
4. **Encoder-only and MoE.** All results are dense decoder-only LMs. Behavior under MoE routing is unstudied.
5. **Very short sequences.** Overhead at `T = 4K` is +25.86% — not obviously worth the modeling gain for short-context workloads. MoDA's value is tied to long-context regimes.
6. **Attention sink reduction.** Quantitative claims in §5.7 are primarily visualization-based; a rigorous quantification (e.g., a reduction in BOS mass as a fraction across heads/layers, matched against a pre-trained attention-sink benchmark) is not in the paper and would strengthen the interpretability argument.
7. **Independent reproduction.** As of this report, the paper is ≤1 month old (v1 on 2026-03-16); the code is public but no independent replication is on file. Results on OLMo2 700M/1.5B are consistent internally but have not yet been checked by a third party.

---

## 10. Bottom line

- **Problem:** Depth in Transformers scaled in quantity but not in quality, because the inter-layer channel remained the 2015 residual sum.
- **Reframe:** Inter-layer communication should be *retrieval* (content-matched), not *accumulation* (coefficient-blended). This is a category correction that prior work (DenseNet → MUDDFormer) approached but never crossed.
- **Method:** MoDA fuses sequence attention and depth attention under a single softmax per layer, with each query jointly attending to current-layer sequence KV and per-token depth KV from all preceding layers.
- **Kernel:** Flash Depth Attention — flash-compatible layout + chunk-aware indexing + group-aware sharing — closes the naive → production gap (1458× speedup), reaching 97.3% of FlashAttention-2 throughput at 64K context.
- **Results:** +2.11 absolute DS-avg points at 1.5B scale, -0.20 validation PPL averaged across 10 domains, with the gains concentrated in reasoning-heavy tasks (ARC-C +4.35, WinoGrande +2.37). Overhead amortizes at long context.
- **Recommended default:** GQA with `G ≥ 8`; reuse attention KV as depth KV plus a dedicated FFN-input depth-KV projection; `L ≈ 64`; post-norm at ≥48 layers; single online-softmax state across the fused sequence+depth pass.

The broader claim — that "the second half of architecture research is about scaling communication between components, not computation inside them" — is a generalization, not established. MoDA is its first concrete instance on the depth axis. Whether the same retrieval-replaces-accumulation move pays off between modalities, between time steps, or across other static data-independent channels is open.

---

## References

- Zhu, L., Fang, Y., Liao, B., Wang, S., Cheng, T., Huang, Z., Chen, C., Wei, L., Zeng, Y., Wang, Y., Lin, Y., Li, Y., & Wang, X. (2026). *Mixture-of-Depths Attention*. arXiv:2603.15619v1. https://arxiv.org/html/2603.15619v1
- Zhu, L. (2026, April 11). *The Second Half of Model Architecture* (companion essay).
- Raposo, D. et al. (2024). *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models*. (Orthogonal work sharing the name; see §8.)
- He, K. et al. (2015). *Deep Residual Learning for Image Recognition*. (The primitive MoDA proposes to replace.)
- Huang, G. et al. (2017). *Densely Connected Convolutional Networks* (DenseNet).
- Peters, M. et al. (2018). *Deep contextualized word representations* (ELMo) — cited for the layer-specialization observation motivating the retrieval frame.
- Dao, T. et al. (2022). *FlashAttention* — the kernel design pattern FDA inherits and extends to the depth dimension.
- Follow-up / related depth-attention lines: Google DCA, Huawei MRLA, Hessian.AI Dreamer, Kimi AttnRes, DenseFormer, LIMe, ByteDance Hyper-Connections, DeepSeek mHC, ColorfulClouds MUDDFormer.
