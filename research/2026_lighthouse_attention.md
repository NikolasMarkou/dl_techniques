# Lighthouse Attention — Long-Context Pretraining via Symmetric Hierarchical Selection

**Paper**: *Long Context Pre-Training with Lighthouse Attention* (Peng, Ghosh, Quesnelle — Nous Research, arXiv:2605.06554v1)
**Code**: https://github.com/ighoshsubho/lighthouse-attention (patch on top of `pytorch/torchtitan @ 61c25f8d`)
**Status**: training-only attention substitute, used for pretraining only; the model converts back to dense SDPA via a short resume phase before inference.

---

## 1. TL;DR

Lighthouse Attention is a **training-only**, **selection-based**, **hierarchical** drop-in replacement for full softmax attention. It is designed for **long-context pretraining** (≥100K tokens) where standard SDPA's Θ(N²) cost dominates wall-clock.

The construction is unusually clean:

1. **Symmetric pyramid pool**: Average-pool Q, K *and* V by factor `p` across `L` levels. Unlike NSA / HISA / InfLLM-v2 / DSA / MoBA, queries are pooled too. This makes the resulting attention call Θ(S²d) instead of Θ(NSd).
2. **Parameter-free scoring**: Each pyramid entry gets two scalars: the per-head ℓ₂ norm of its pooled Q projection (`s^QK`) and of its pooled K projection (`s^KQ`). No learned scorer, no auxiliary loss, no Gumbel-softmax, no straight-through estimator.
3. **Outside-the-kernel selection**: A fused chunked-bitonic top-K kernel picks `k` entries jointly across all levels; rejected coarse entries are retained as causal-boundary filler so the gathered sub-sequence is *contiguous and hole-free*. The actual attention call is **stock FlashAttention** on this dense sub-sequence.
4. **Two-stage training**: Stage 1 trains for the bulk of the budget with Lighthouse active; Stage 2 (SDPA-resume) disables selection and finetunes briefly under dense SDPA. The resumed checkpoint **matches or beats dense-from-scratch** at the same token budget — the load-bearing correctness claim of the paper.

**Headline numbers** (530M Llama-3, 16k optimizer steps, ~50B tokens on C4):

- **17× faster** fwd+bwd at 512K context, single B200 (single-layer microbenchmark)
- **1.4–1.7× end-to-end** pretraining speedup at 98K context vs dense SDPA baseline
- Final loss **0.6825–0.7102** (Lighthouse → SDPA) vs **0.7237** (dense SDPA from scratch) — *better* loss at *lower* B200-hours
- Scales to **1M-token training** across 32 B200s with context parallelism (CP=8, DP=1, 4 nodes)

---

## 2. Why this works (the two prior-art mistakes Lighthouse avoids)

Prior selection-based attention (NSA, HISA, InfLLM-v2, DSA, MoBA) makes two design decisions that are fine for *inference* but quietly bad for *training*:

1. **Asymmetry**: only K and V are pooled; Q stays at full resolution. The hierarchy becomes a compressed addressable memory rather than a multi-scale representation. Fine for autoregressive decoding (one query at a time); wasteful when all queries co-occur in a training forward.
2. **Architectural entanglement**: selection lives *inside* the attention kernel as a sparse matmul / per-query gather. This locks out FlashAttention's bit-exact dense kernels and requires a bespoke sparse forward+backward per GPU architecture.

There is also a third concern specific to training: an inference-time sparse method is by construction at most as good as its dense backbone. A training-time sparse method has a harder bar — *will the resulting weights still be a competent dense-attention model?* If not, it has trained a specialist of its own approximation.

Lighthouse's design (symmetric pool + non-differentiable top-K + stock FlashAttention + SDPA-resume) is structured precisely to satisfy that third criterion.

---

## 3. The Four Stages

Let `H_t ∈ R^{N×d_model}` be the layer input. Standard SDPA:

```
Q, K, V = H_t @ W_Q, H_t @ W_K, H_t @ W_V          # (N, d) each per head
O = softmax( Q Kᵀ / √d + M ) V                     # Θ(N²d)
```

Lighthouse replaces the softmax call with a 4-stage pipeline; gradients flow only along the trunk (scatter → FA → gather → pool → projections), the green selector branch is non-differentiable.

### Stage (i) — Pyramid Pool

Average-pool Q, K, V symmetrically into `L` levels with pooling factor `p`:

```
W_i^{(l)}   = [ i·p^l, (i+1)·p^l − 1 ]              # window at level l
Q_i^{(l)}   = mean({ Q_j : j ∈ W_i^{(l)} })         # symmetric: same for K, V
```

- Level 0 is the original full-resolution sequence (`p^0 = 1`).
- Level `l` has `N/p^l` entries, each summarising `p^l` base positions.
- Requirement: `p^{L-1} | N`.
- Cost: Θ(Nd) (a view + mean, fused via `torch.compile`).
- Total pyramid entries: Σ N/p^l ≤ N · p/(p−1).

Pooled queries and pooled keys at level `l` live in the **same representation space**, so summary–summary interactions are well-defined. This is what makes the dense kernel call Θ(S²d) rather than Θ(NSd).

### Stage (ii) — Scoring and Top-K

Two scalar scores per entry, one as query, one as key:

```
At level 0:   s_{0,i}^QK = ‖Q_i‖₂,   s_{0,i}^KQ = ‖K_i‖₂
At level l:   s_{l,i}^QK = max_{0≤j<p^l} s_{0, ip^l + j}^QK    # max-pool from level 0
              s_{l,i}^KQ = max_{0≤j<p^l} s_{0, ip^l + j}^KQ
```

**Max-pooling** the coarser-level scores (rather than recomputing norms on pooled projections) lets a coarse span inherit the importance of its strongest constituent token.

Selection runs jointly over the concatenated QK and KQ streams across all levels:

```
I = TopK( { s_{l,i}^QK, s_{l,i}^KQ : (l,i) ∈ P },  k )
```

An entry chosen via its KQ score still enters the gather as its own `(Q_i^{(l)}, K_i^{(l)}, V_i^{(l)})` triple — the KQ stream just gives keys a *second chance* to surface tokens that are important as receivers but not as senders.

**Coarsest level is always retained in full**: it is cheap (`N/p^{L-1}` entries) and guarantees at least one contributor at every base position, so the gathered sub-sequence has no holes.

**Critical: rejected coarse entries are kept as causal-boundary filler.** If we threw away every rejected coarse entry, the level-1 survivors would no longer tile the base sequence — there would be gaps over positions whose coarse summary didn't make the cut and whose finer descendants weren't selected either. Those gaps would force a sparse-aware causal mask. Lighthouse instead keeps rejected coarse entries (`pk` per level, where the `p` factor is exactly this causal-boundary bookkeeping). After sorting by base-sequence position, the result is **topologically causal with no holes** — the standard lower-triangular causal mask Just Works, and the attention kernel never sees a sparse layout.

### Stage (iii) — Gathered-Sequence Attention

```
Q̃_m = Q_{i_m}^{(l_m)},  K̃_m = K_{i_m}^{(l_m)},  Ṽ_m = V_{i_m}^{(l_m)},   (l_m, i_m) ∈ I
S   = N/p^{L-1} + (L−1)·p·k                    # gathered sequence length
Õ   = FlashAttention(Q̃, K̃, Ṽ ; M̃)             # ordinary S×S causal mask
```

At N=1M, L=4, p=4, k=4096:  S ≈ 6.5×10⁴ ≪ N. Choosing `L = log_p(N/k)` balances the two terms and gives `S = Θ(k log_p(N/k))`.

This is the only Θ(S²d) call in the layer. Everything else is Θ(Nd) or Θ(N log k).

### Stage (iv) — Scatter-Back

Each output entry is written to the `p^l` base positions it represents, with a causal shift:

```
R(l, i) = [ i·p^l + p^l − 1,  i·p^l + 2·p^l − 2 ]      # range of fan-out
O_j     = Σ_{ m : j ∈ R(l_m, i_m) } Õ_m                # accumulate
```

The shift of `p^l − 1` starts the write at the *last* summarised token — so a coarse summary of positions `[ip^l, (i+1)p^l − 1]` writes to `[(i+1)p^l − 1, (i+2)p^l − 2]`. This preserves causality: base position `j` never receives a summary that contains its own future.

Within a level, consecutive windows write to disjoint adjacent ranges; across levels, contributions are summed. Per-position fan-in is bounded by `L` regardless of `k`.

The accumulation is implemented as a custom Triton kernel with two variants:
- **Default**: fp atomic-add (non-deterministic, fast).
- **Reproducibility**: integer atomic (deterministic, 1.2–2× slower).

---

## 4. Algorithm (one layer, one head)

```
Input:  X ∈ R^{N×d_model}, projections W_Q, W_K, W_V, pyramid (L, p), budget k
Output: O ∈ R^{N×d}

1.  Q, K, V ← X W_Q, X W_K, X W_V                        # fused GEMM
2.  (Q^{(l)}, K^{(l)}, V^{(l)})_{l=0..L-1} ← Pool_μ(Q,K,V)   # view+mean
3.  s^QK_{l,i} ← Pool_max ‖Q_j‖₂,  s^KQ_{l,i} ← Pool_max ‖K_j‖₂
4.  I ← ChunkedBitonicTopK(s^QK, s^KQ, k)                # custom Triton/CUDA kernel
5.  Q̃, K̃, Ṽ ← gather(Q^{(·)}, K^{(·)}, V^{(·)}; I)        # torch.gather
6.  Õ ← FlashAttention(Q̃, K̃, Ṽ)                         # stock FA-3 / cuDNN-SDPA
7.  O ← ScatterBack(Õ, I)                                # custom Triton kernel
return O
```

---

## 5. Complexity

| Stage                       | Primitive               | Cost              |
|-----------------------------|-------------------------|-------------------|
| Projections Q,K,V           | GEMM                    | Θ(N · d_model · d)|
| Pyramid pool                | view+mean               | Θ(N d)            |
| Scoring (norms, max-pool)   | norm+max                | Θ(N d)            |
| Top-K selection             | chunked bitonic         | Θ(N log k)        |
| Gather                      | `torch.gather`          | Θ(S d)            |
| **Dense sub-seq attention** | FlashAttention          | **Θ(S² d)**       |
| Scatter-back                | custom atomic           | Θ(N d)            |

With `S = N/p^{L−1} + (L−1)pk` and balanced `L = log_p(N/k)`:
`S = Θ(k log_p(N/k))`, so attention cost = Θ(k² log²N · d) — polylogarithmic at fixed k.

Per-layer total at bounded k: **Θ(N · d)** — linear in N (up to a log k factor for top-K).

Comparison:

| Method                          | Per-layer compute |
|---------------------------------|-------------------|
| Dense softmax                   | Θ(N² d)           |
| Log-Linear Attention            | Θ(N log N · d)    |
| **Lighthouse (bounded k)**      | **Θ(N · d)**      |
| Linear attention / SSMs         | Θ(N · d)          |

---

## 6. Training Recipe (load-bearing)

The trained weights must remain a competent dense-attention model after sparse training. Recipe is two-stage:

- **Stage 1 (Lighthouse)**: train the majority of the budget with selection enabled.
- **Stage 2 (SDPA-resume)**: resume the Stage-1 checkpoint with selection disabled — same optimizer state, same dataloader continuation, ordinary dense SDPA. A brief tail (4k–6k of 16k total) is enough.

At each resume the loss **transiently spikes** (1.12–1.57 nats) as the model is first asked to use attention it was not trained against, then recovers within ~1–1.5k SDPA steps and crosses below the dense baseline.

Across three split points (10k+6k / 11k+5k / 12k+4k of 16k total) **every recovered run matches or beats** dense-from-scratch (final loss 0.6980–0.7102 vs 0.7237). Recovery is robust — the recipe doesn't pivot on a precise schedule.

---

## 7. Hyperparameters and Ablations (530M Llama-3, ctx=98K, 16k steps, ~50.3B tokens, 8×B200)

| Configuration                              | Scorer  | B200-h ↓ | Tok/s/GPU (k) ↑ | Final Loss ↓ |
|--------------------------------------------|---------|---------:|-----------------:|-------------:|
| **SDPA baseline (dense-from-scratch)**     | —       |    303.2 |             45.6 |       0.7237 |
| LH→SDPA (12k+4k) L=3,p=2,k=6144            | Dilated |    214.7 |             74.7 |       0.7102 |
| LH→SDPA (11k+5k) L=3,p=2,k=6144            | Dilated |    219.6 |             75.4 |       0.7001 |
| LH→SDPA (10k+6k) L=3,p=2,k=6144            | Dilated |    228.0 |             75.0 |       0.6980 |
| L=3, p=2, k=1536                           | Dilated |    203.9 |             93.9 |   **0.6825** |
| L=3, p=4, k=1536                           | Dilated |    197.2 |             99.5 |       0.6881 |
| L=3, p=4, k=1536                           | Norm    |    179.6 |        **126.0** |       0.6946 |
| L=3, p=8, k=1536                           | Dilated |    206.2 |             92.1 |       0.6828 |
| L=4, p=2, k=1536                           | Dilated |    200.2 |             96.4 |       0.6978 |
| L=5, p=2, k=1536                           | Dilated |    201.5 |             96.3 |       0.6991 |
| L=3, p=2, k=2048                           | Dilated |    208.1 |             90.9 |       0.6880 |
| L=3, p=2, k=4096                           | Dilated |    215.7 |             83.5 |       0.6951 |
| **CP: k=4096, ctx=256K, CP=8, DP=1**       | Norm    |   1300.3 |             48.9 |   **0.6721** |

Findings (full grid in §A of paper):

- **Every** Lighthouse configuration matches or beats the dense baseline — recoverability is not specific to any one hyperparameter setting.
- **Scorer**: projection-norm is within ~0.01 nats of dilated softmax in either direction, but is parameter-free and ~9% cheaper in B200-hours. Default: `norm`.
- **Pool factor p**: ∈ {2, 4, 8} land within ~0.02 nats. Effect is dominated by throughput/memory tradeoffs.
- **Levels L**: L=3 best; L=4, L=5 monotonically worse (deeper pyramid spreads the budget across more coarse levels, leaving fewer entries at the finest level).
- **Top-K k**: counter-intuitively, *smaller* k gives lower loss (0.6825 → 0.6880 → 0.6890 → 0.6951 over k ∈ {1536, 2048, 3072, 4096}, dips back at 6144). Plausibly hierarchical selection regularises at this token budget; whether it reverses at much larger budgets is an open question.

**Stage-1 throughput**: 84–126k tok/s/GPU vs ~46k for dense SDPA — roughly **2× per-step**. Stage-2 (SDPA-resume) matches the dense baseline; all the saving comes from Stage 1.

**End-to-end 10k+6k recipe**: 22.5–27.0h wall-clock vs 37.9h for dense baseline → **1.40×–1.69× speedup** at matched or lower final loss.

### Long-Context Retrieval (NIAH, simplified single-digit passkey)

At 530M params and 16k steps, full-prose NIAH scores near zero (model too small). The paper uses a single-digit variant: hide one of `{0..9}` in alphanumeric filler at depths `{0, 15, 30, 50, 70, 85, 100}%` across contexts `{4, 8, 16, 32, 64, 96}K`; argmax over the 10 digit tokens at the last position; mean over 10 trials.

Mean retrieval rate over the grid:

| Config (L=3, p=4)              | Mean |
|-------------------------------|-----:|
| k=2048 dilated → SDPA          | 0.76 |
| k=1536 dilated → SDPA          | 0.73 |
| k=2048 norm → SDPA             | 0.72 |
| **Dense SDPA from scratch**    | 0.72 |
| k=1536 norm → SDPA             | 0.65 |

Two patterns: (i) larger k is the dominant axis; (ii) `norm` hurts retrieval more than it hurts loss — at fixed k, switching dilated→norm costs 0.04–0.08 retrieval. **Default choice depends on the downstream task**: `norm` for loss/throughput, `dilated` for retrieval.

---

## 8. Scaling vs Dense Attention

Single-layer microbench on a single B200 (bf16, B=1, H=8, d=128, L=3, p=4, sparsity ≈ 1:64):

- SDPA scales Θ(N²d); Lighthouse scales Θ(S²d) with S ≪ N.
- At short contexts the two curves track each other (selector overhead dominates).
- At N=512K: **21× faster forward**, **17.3× faster fwd+bwd**.
- Equivalently, dense SDPA needs ~113K (fwd) or ~122K (fwd+bwd) of context to match Lighthouse's runtime at 512K.

The Lighthouse curve already includes pyramid pool + scoring + top-K + scatter overhead — it is not just the FlashAttention call.

---

## 9. Context Parallelism (CP)

Beyond ~100K context, the 530M model OOMs on a single B200 regardless of attention method (activations + grads + optimizer state). Lighthouse extends cleanly to CP **with no sparse-aware collectives** because its pre-attention primitives are local:

1. The coarsest pool window `p^{L−1}` (e.g. 64) is orders of magnitude smaller than the shard size (`N/W = 128K` at N=1M, W=8) → pooling and scoring need no inter-rank communication.
2. Each rank runs the chunked-bitonic top-K on its own pyramid, producing `I_r ⊆ P_r` from tokens it already owns.
3. The gathered sub-sequence is **dense**, so FlashAttention runs under standard **ring attention** — KV shards rotate through the ring as in a fully dense long-context run. Each rank's queries see the cross-shard context selected by every other rank's Lighthouse pipeline.

This last property is only possible because Lighthouse's selection output is a *contiguous tensor*. Sparse-selection kernels cannot express ring rotation without engineering specific to the sparse layout.

CP cost: ~10% per-rank throughput overhead vs single-device extrapolation. Validated configs: CP=2/DP=4 at 98K, **CP=8/DP=1 at 256K**, and **1M-token training on 32 B200s (4 nodes, CP=8)**.

The CP path **requires the `norm` scorer** — the `dilated` and `gla` scorers are refused at construction time:

```
ValueError: lighthouse_scorer='dilated' is not supported under context parallelism.
The CP path was validated only for 'norm'.
```

CP also requires a load-balancing knob to be disabled in the underlying PyTorch CP API (`_cp_options.enable_load_balance = True/False`); the patch threads `enable_load_balance=is_lighthouse_cp` through `create_context_parallel_ctx`.

---

## 10. Scorer Variants

The reference implementation ships three:

### 10a. `norm` (default, CP-compatible)

```python
scores_qk = xq.norm(dim=-1)   # (B, S, H)
scores_kq = xk.norm(dim=-1)   # (B, S, H)
# coarser levels: max-pool the level-0 norms
```

Parameter-free, Θ(N), no Q–K interaction. Strictly weaker than a learned QK scorer. Any positive result is a *lower bound* on what selection-based training can deliver.

### 10b. `dilated`

Run softmax attention over the pyramid with dilation factor δ at O(N²d/δ). Sub-quadratic but still super-linear; an order of magnitude more expensive than `norm` at long context.

```python
# Reshape to (B, S/δ, δ·H, D); call inner attention twice (QKV and KQV);
# take ℓ₂ norm of each output as the score.
```

Provides Q–K interaction in the score. Slightly better retrieval; ~9% more wall-clock.

### 10c. `gla` (gated linear attention; requires `flash-linear-attention`)

Uses `chunk_simple_gla` from the `fla` package, with a per-token gate projection `wg : R^{d_model} → R^H` and `gk = log_sigmoid(wg(x)) / 16`. Most expressive scorer; only path that adds learnable parameters (the gate `wg`).

The patch's `_build_lighthouse_scorer` dispatches on `lighthouse_scorer ∈ {norm, dilated, gla}`; selecting `gla` flags `needs_gate=True` so the `Attention` module instantiates `self.wg` and threads `gk` through the forward.

---

## 11. Architecture / Reference Setup

530M Llama-3-style decoder:
- `d_model = 1024`, `n_layers = 30`, `n_heads = 8`, `n_kv_heads = 8`, head dim 128, FFN hidden 1536
- `rope_theta = 10000`
- byte-level tokenizer
- **Layers {0, 1, 28, 29} retain dense SDPA**; the other 26 use Lighthouse
- Inner attention call uses cuDNN-SDPA (PyTorch 2.11.0+cu128, cuDNN 9.19.0, CUDA 12.8)

Training:
- C4 dataset, seq_len **98,304**, global batch **32**
- AdamW: lr `2e-3`, β₁=0.9, β₂=0.95, weight_decay 0.1
- linear warmup 2k steps, gradient-norm clip 1.0
- bfloat16, FSDP only (no TP, no PP for the main ablation grid)
- 16,000 optimizer steps total (~50.3B tokens)

For 1M-token training: same architecture, `lighthouse_full_attn_layers=[]` (no dense layers), CP=8, DP=1, L=4, p=4, k=4096.

---

## 12. The Two Critical Kernels

The implementation lives in two new files plus ~600 lines of edits on `torchtitan`. Two stages get custom kernels:

### 12a. Chunked-Bitonic Top-K (`lighthouse_selection_cuda.py`)

A textbook bitonic top-K sorts the entire score stream in shared memory or registers — fails at k=4096 over a pyramid of ~2N entries. Lighthouse uses a **chunked** design:

- Partition the score stream into fixed-size chunks (TOPK_PER_TILE = 128).
- Each chunk maintains a running top-m buffer (TOPK_HALF = 64 each for QK and KQ streams) via **in-register bitonic merge** using `__shfl_xor_sync` warp shuffles for k≤32 and shared-memory swaps for k>32.
- `N/N_chunk` chunks dispatch as independent CTAs (one CTA per `(batch, kv_head)` × `(seq_chunk)`).
- No thread block ever holds more than `m` scores; the work is fully parallel.

**Coarse-to-fine cascade inside the kernel**: starts at level `num_levels` (coarsest), keeps top-K parents, descends into the `p` children of each parent, runs top-K again, and so on. At level 1 (base) all surviving entries are kept. Rejected entries from coarser levels are emitted to the output buffer alongside selected ones (the causal-boundary filler from §3 stage (ii)).

**Output layout — packed uint64 keys**: each output slot stores `(level_ordered_key : uint32) << 32 | (actual_index : uint32)`. The level-ordered key is a float (base position with sub-level shift) encoded so its bit pattern sorts the same way as the float value would. A single global `torch.sort` along the last dim then produces the topologically-causal contiguous sub-sequence in O(N log N) time. The lower 32 bits are masked off to recover the actual indices.

The kernel is compiled at runtime via `torch.cuda._compile_kernel(..., nvcc_options=["--use_fast_math", "--std=c++17"])`. There is also a pure-Triton variant (`LighthouseSelectionQkKqChunkedTopk`) in `lighthouse_selection.py` for portability.

**Asserts** (from `LighthouseSelectionQkKqChunkedTopkNvrtc.forward`):
- `topk % 128 == 0`
- `pooling_factor` is a power of 2
- `topk // 2 ≥ pooling_factor`
- `N_CHUNK_COARSEST ≥ TOPK_PER_TILE` (i.e. the coarsest level divided across tiles must leave each tile with ≥128 entries)
- `scores_qk.dtype == float32` (selection scores are always upcast to fp32)

### 12b. Scatter-Back (`_ScatterToBaseAutograd` in `lighthouse_selection.py`)

A custom `torch.autograd.Function` with a Triton forward and Triton backward.

**Forward** `_scatter_fanout_fwd_kernel`:
- For each gathered output position `s ∈ [0, S_sel)`, the index `idx` encodes which pyramid level it came from (via cumulative level sizes `N_shard, N_shard/p, N_shard/p², …`).
- Compute `base_pos = global_off + local_pos · p^l` and `fan_out = p^l`.
- Atomic-add the output vector `O_s` to all `p^l` positions `[base_pos + p^l − 1, base_pos + 2·p^l − 2]` in the base sequence.
- Uses `tl.atomic_add(..., sem="relaxed", scope="gpu")` for the fp non-deterministic path.

**Backward** `_scatter_fanout_bwd_kernel`:
- Reads `p^l` positions from `grad_output` and sums them into `grad_attn[s]`.
- No atomic needed (each output is read by exactly one input).

The two key shape parameters baked into the kernel:
- `MAX_PF = pooling_factor ^ (num_levels − 1)` — the maximum fan-out (coarsest level).
- `BLOCK_D = next_power_of_2(D)` — head dimension rounded up for vectorised loads.

The forward is non-deterministic by default (fp atomic ordering); the deterministic variant uses integer atomics and is 1.2–2× slower (intended only for reproducing results).

### 12c. Helper: CP all-to-all primitives

`lighthouse_selection.py` also implements custom `torch.autograd.Function`s for the CP all-to-alls:

- `UnstripeFunction` / `_unstripe_impl`: convert a striped layout (`N_shard` strided across `W` ranks) to a contiguous slice.
- `StripeFunction` / `_stripe_impl`: inverse, used in the backward.
- `SeqToHeadParallel` / `HeadToSeqParallel`: reshape between sequence-parallel and head-parallel layouts for ring attention. Requires `H % world_size == 0`.

The `LighthouseCP` module splits the sequence into a head half and a tail half (`s // 2` each), runs scoring + top-K per half (`topk_per_load = topk // (world_size * 2)` for global-topk-across-load-balanced halves), gathers, calls the inner attention, and scatters each half back independently.

---

## 13. Code Layout — File-by-File

### 13a. `src/lighthouse_selection.py` (~600 lines)

Top-level classes:

| Class                                   | Role                                                                                 |
|-----------------------------------------|--------------------------------------------------------------------------------------|
| `DialatedLighthouseNormScorer`          | `norm` scorer — non-CP. Returns `scores_qk_cat, scores_kq_cat, xq_cat, xk_cat, xv_cat`. |
| `DialatedLighthouseNormScorerCP`        | `norm` scorer — CP-aware (used per shard).                                            |
| `DilatedLighthouseScorer`               | `dilated` scorer — runs softmax attention over a dilated pyramid for Q→K and K→Q.    |
| `DilatedLighthouseGLAScorer`            | `gla` scorer — uses `chunk_simple_gla` from `fla`. Requires gate `gk`.               |
| `LighthouseLocal`                       | Non-CP wrapper: scorer → top-K → gather → attention → scatter.                       |
| `LighthouseCP`                          | CP wrapper: head/tail split + per-shard scoring + ring-attention gather.             |
| `LighthouseSelectionQkKqChunkedTopk`    | Pure-Triton top-K kernel (autotuned, `@torch.compiler.disable`).                     |
| `_ScatterToBaseAutograd`                | Triton fwd+bwd scatter-back as a custom autograd Function.                           |

Triton kernels:

- `_compare_and_swap`, `_bitonic_merge`, `argsort`, `merge_topk_gather` — bitonic top-K primitives.
- `_encode_indices`, `_encode_indices_selected` — uint64 key packing (level-order ⨁ actual-index).
- `lighthouse_selection_scores_qk_kq_chunked_topk_kernel` — main top-K kernel.
- `_scatter_fanout_fwd_kernel`, `_scatter_fanout_bwd_kernel` — scatter-back.

### 13b. `src/lighthouse_selection_cuda.py` (~300 lines)

NVRTC-compiled CUDA variant of the top-K kernel, dispatched via `torch.cuda._compile_kernel`. Lower latency than Triton at the cost of having to recompile per `(pooling_factor, topk_per_tile)`. The class `LighthouseSelectionQkKqChunkedTopkNvrtc` is what the reference `LighthouseLocal` / `LighthouseCP` actually call.

### 13c. The torchtitan patch (`lighthouse-attention.patch`)

Touches 8 files; cumulative diff ~3700 lines (most of it is config registration).

| File                                            | Purpose                                                                              |
|-------------------------------------------------|--------------------------------------------------------------------------------------|
| `torchtitan/models/llama3/model/args.py`        | Add `dilation`, `hidden_dim`, `use_selection_lighthouse`, `use_lighthouse_cp`, `lighthouse_num_levels`, `lighthouse_pooling_factor`, `lighthouse_topk`, `lighthouse_scorer`, `lighthouse_full_attn_layers` to `TransformerModelArgs`. |
| `torchtitan/models/llama3/model/model.py`       | `_build_lighthouse_scorer(model_args, head_dim)` dispatches on scorer name; wires `wg` for GLA; FFN honors explicit `hidden_dim`. `Attention.forward` switches between SDPA and selection paths; `set_use_lighthouse(False)` disables for resume / dense layers; `Attention.set_cp_info(rank, world_size, cp_group)` propagates CP state. `Transformer.__init__` applies `set_use_lighthouse(False)` to the layer ids in `lighthouse_full_attn_layers`. |
| `torchtitan/models/llama3/__init__.py`          | Registers ~26 ablation flavors (`ablation_270m_lighthouse_topk{K}_pool{P}_lvl{L}[_cp[_1m]]` and dim-matched dense `_sdpa` flavors for resume). |
| `torchtitan/models/llama3/infra/parallelize.py` | `apply_compile` uses `compile_config.fullgraph` so transformer blocks compile with graph breaks (required by `@torch.compiler.disable` on scorers). |
| `torchtitan/distributed/utils.py`               | `create_context_parallel_ctx(..., enable_load_balance=True)` knob (toggles `_cp_options.enable_load_balance` on the experimental PyTorch CP API). |
| `torchtitan/config/job_config.py`               | Adds `fullgraph: bool = False` to `Compile` dataclass.                               |
| `torchtitan/hf_datasets/text_datasets.py`       | Registers a `c4_local` dataset entry for an on-disk C4 mirror at `/home/c4`.         |
| `torchtitan/train.py`                           | When CP is enabled and the model has Lighthouse-CP modules: calls `module.set_cp_info(cp_rank, cp_world_size, cp_group)` once on first `forward_backward_step`; threads `enable_load_balance=is_lighthouse_cp` through the CP context. |

Two new files (`lighthouse_selection.py`, `lighthouse_selection_cuda.py`) are copied into `torchtitan/models/llama3/model/` *separately* from the patch.

---

## 14. Reproduction Instructions

### 14a. Environment

- **Python 3.13**, **CUDA 12.8** toolkit on the host
- **PyTorch 2.11.0+cu128**, cuDNN 9.19.0
- GPU: NVIDIA B200 (sm_100) — Hopper (H100) will also work but is not what the paper benchmarks
- `triton==3.6.0`, `einops==0.8.2`, `datasets==4.8.2`, `transformers==5.5.4`, `wandb==0.25.1`, `flash-linear-attention==0.4.2` (only if using `gla` scorer)
- Optional: FA4 from source (`pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=cute`) — the default cuDNN-SDPA path works fine.

### 14b. Apply the patch

```bash
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout 61c25f8d                                                    # tested SHA

cp /path/to/lighthouse-attention/src/lighthouse_selection.py      torchtitan/models/llama3/model/
cp /path/to/lighthouse-attention/src/lighthouse_selection_cuda.py torchtitan/models/llama3/model/

git apply /path/to/lighthouse-attention/lighthouse-attention.patch

python3.13 -m venv .venv && source .venv/bin/activate
pip install -r /path/to/lighthouse-attention/requirements.txt
pip install -e . --no-deps
```

### 14c. Launch Stage 1 (Lighthouse)

```bash
# Substitute placeholder paths in the toml:
sed -e 's|<DUMP_FOLDER>|/scratch/runs/topk1536|' \
    -e 's|<HF_ASSETS_PATH>|/scratch/tokenizer/bytes|' \
    -e 's|<CHECKPOINT_FOLDER>|/scratch/ckpts/topk1536|' \
    /path/to/lighthouse-attention/configs/topk/topk1536.toml > /tmp/stage1.toml

torchrun --nproc-per-node 8 ./torchtitan/train.py --job.config_file /tmp/stage1.toml
```

This trains 10,000 steps with `flavor = ablation_270m_lighthouse_topk1536_pool4_lvl3`.

### 14d. Launch Stage 2 (SDPA resume)

Create a second toml pointing at the same `[checkpoint] folder` but with:
- `[training] steps = 16000` (continues to 16k total)
- `[model] flavor = ablation_270m_topk1536_pool4_lvl3_sdpa` (dim-matched dense variant — same dim, hidden_dim, n_heads, n_kv_heads, no `use_selection_lighthouse`)

Run with the same `torchrun` command.

### 14e. Context-parallel (CP=2/DP=4 at 98K)

```bash
torchrun --nnodes 1 --nproc-per-node 8 ./torchtitan/train.py \
    --job.config_file /path/to/lighthouse-attention/configs/cp/norm_cp2_dp4.toml
```

The toml sets `context_parallel_degree = 2` and uses `flavor = ablation_270m_lighthouse_topk1536_pool4_lvl3` (which has `use_lighthouse_cp=True` and `lighthouse_scorer="norm"`). The patch wires `set_cp_info(...)` on first step automatically.

For 1M-token: `ablation_270m_lighthouse_topk4096_pool4_lvl4_cp_1m` (L=4, p=4, k=4096, `lighthouse_full_attn_layers=[]`), 4 nodes × 8 GPUs, CP=8, DP=1.

---

## 15. Important Implementation Details / Gotchas

1. **Selection runs at `torch.no_grad()`**. Top-K is non-differentiable; indices are int32; gradients carry no information through the selector branch.
2. **`@torch.compiler.disable` on the top-K and GLA scorer**. The `inductor` backend cannot trace through NVRTC-compiled kernels or `chunk_simple_gla`. This forces the transformer block to compile with `fullgraph=False` (graph breaks allowed) — the patch adds the `compile_config.fullgraph` knob to enable this without breaking other models.
3. **Scores upcast to fp32** before the top-K kernel. bf16 maxima around the score range can collide and produce non-deterministic selection; the kernel asserts fp32.
4. **FFN hidden dim**. Lighthouse flavors set `hidden_dim` directly on `TransformerModelArgs`. The patch falls back to upstream's `4 * dim` when `hidden_dim is None`. Ablation flavors use `hidden_dim=1536` (3× smaller than upstream's default 4096), which is a deliberate part of the 530M parameter count.
5. **Full-attention layers**. The ablation flavors keep layers `[0, 1, 28, 29]` (first two and last two) on dense SDPA. This consistently improves recoverability; the 1M-token config sets `lighthouse_full_attn_layers=[]` because at that scale even four dense layers are too expensive.
6. **GLA scorer adds a parameter**. `wg : Linear(d_model, n_heads, bias=True)` and `gate_logit_normalizer = 16`. The patch initializes `wg.weight ~ trunc_normal(0, 0.02)`, `wg.bias = 0`.
7. **CP path requires `norm` scorer**. `_build_lighthouse_scorer` raises `ValueError` if `use_lighthouse_cp and scorer != "norm"`. The CP path has only been validated for that variant.
8. **CP load balancing**. PyTorch's experimental CP API does ring-attention load balancing by default (cuts each shard in half, pairs head+tail across ranks). Lighthouse CP needs this *on* (`enable_load_balance=True`), which is why `LighthouseCP.forward` always processes `xq[:, :half]` and `xq[:, half:]` separately and uses `topk_per_load = topk // (world_size * 2)`.
9. **Scatter-back non-determinism**. The default fp-atomic scatter produces different bit-exact output across runs at fixed seed; this is acceptable for training (the accumulated gradient is statistically identical). For reproducibility, swap to the integer-atomic variant at 1.2–2× wall-clock cost.
10. **Dataset is on-disk C4**. The patch hardcodes `c4_local` to `path="/home/c4"`. Override before running outside the paper's exact environment.

---

## 16. Limitations

- **Symmetric Q/K/V pooling presumes all queries co-occur in one forward pass.** Autoregressive *decoding* violates this; Lighthouse is **not an inference-time substitute**. The dense-SDPA resume is what converts the checkpoint into a serveable model. Every downstream evaluation in the paper runs after the resume.
- **Inner attention is Θ(S²d) on the gathered sub-sequence.** Sub-quadratic in N at fixed k, but *not strictly linear*. Regimes where k must scale with N to maintain recall are not characterised in the paper.
- **All experiments are at 530M params, 16k steps, 50B tokens.** No evidence at frontier scale. The smaller-k-helps trend is plausibly a regularisation artefact at this token budget.
- **NIAH evaluation is a simplified single-digit passkey**, not the standard prose variant — necessary because of the model size, but limits direct comparison to published NIAH numbers.

---

## 17. Open Questions / Future Directions

1. **Asymmetric sparse resumption**. Swap the dense-SDPA resume for an inference-oriented asymmetric sparse target (DSA, NSA, HISA, MoBA) so the converted checkpoint is natively serveable.
2. **Adaptive selection budget**. Per-layer or per-head `k` allocation instead of one fixed `k` across the whole model.
3. **Beyond text**. Vision, audio, and video have natural multi-scale structure that fits the pyramid.
4. **Scaling**. Does the smaller-k-helps trend reverse at frontier-scale token budgets?
5. **Serving integration**. Continuous batching, speculative decoding, KV-cache management for the resumed model — needed to translate the training speedup into deployment speedup.

---

## 18. Build-It Checklist (Keras 3 / TF 2.18 port path for `dl_techniques`)

If we wanted to add a Lighthouse layer to this repo, the minimum viable port:

1. **`PyramidPool` layer**: symmetric average-pool Q, K, V into L levels with factor p; output flat-concatenated tensors of shape `(B, S_pyr, H, D)` with `S_pyr = Σ N/p^l`.
2. **`NormScorer`**: per-head ℓ₂ norms at level 0; `tf.reduce_max` along window dim for coarser levels. Output `(B, S_pyr, H)`.
3. **`ChunkedBitonicTopK`**: this is the hard one. Options:
   - Wrap the existing CUDA NVRTC kernel via `tf.load_op_library` (requires writing a thin TF custom op).
   - Reimplement the chunked-bitonic logic in `tf.raw_ops.TopKV2` over fixed-size chunks, then merge — slower but pure TF.
   - Approximate via `tf.math.top_k` over the flat pyramid plus an explicit causal-boundary pass to insert rejected coarse entries. Loses the stratified selection guarantee.
4. **`scatter_to_base_sequence`**: a `tf.scatter_nd` with explicit per-level fan-out. Forward is straightforward; backward is `tf.gather` of `grad_output`. Both can be expressed as `tf.function` ops.
5. **`LighthouseAttention` Keras layer**:
   - `__init__`: `n_heads, head_dim, num_levels=3, pooling_factor=2, topk=1536, scorer='norm', full_attention=False`.
   - `build`: standard `wq, wk, wv, wo` Linear layers; optional `wg` for GLA.
   - `call`: switch on `full_attention` to choose the dense path (Keras `MultiHeadAttention` or `keras.ops.dot_product_attention`) vs the Lighthouse path.
   - Serialisable via `get_config()` per repo conventions.
6. **Two-stage training**: the cleanest expression in this repo is a callback that flips `full_attention=True` on all Lighthouse layers at a configurable step boundary; learning rate, optimizer state, and dataloader continue unchanged.

The challenges in order of difficulty: (1) the custom top-K kernel — Triton/CUDA path doesn't survive in Keras+JAX/TF; (2) keeping the scatter-back differentiable; (3) replicating the exact CP path under TF distribution strategies.

---

## 19. References

- Peng, Ghosh, Quesnelle. *Long Context Pre-Training with Lighthouse Attention*. arXiv:2605.06554v1.
- Reference implementation: `github.com/ighoshsubho/lighthouse-attention`.
- Underlying torchtitan commit: `pytorch/torchtitan @ 61c25f8d`.
- Key prior art: FlashAttention (Dao 2022, 2024; Shah 2024), NSA (Yuan 2025), DSA (DeepSeek 2025), HISA (Zhao 2026), MoBA (Lu 2025), InfLLM-v2 (Zhao 2026), Ring Attention (Liu 2023).

---

## 20. Verbatim Kernel & Module Reference

This appendix exists so the document is genuinely standalone. Everything below is either the literal reference source (Apache-2.0 / MIT-style upstream) or pseudocode dense enough to reimplement without consulting external files.

### 20a. The float→ordered-uint encoding trick

The chunked-bitonic top-K outputs `(key : uint32) << 32 | (actual_index : uint32)` and a single global `torch.sort` over the resulting uint64 stream produces the topologically-causal layout. The key field is a float (the level-ordered base position) encoded so its **bit pattern** has the same total order as the float's numeric value, across both positive and negative floats. Standard IEEE-754 ordering doesn't have this property (negative floats sort *reversed* by bit pattern); the fix is a one-bit flip for non-negatives and a bitwise NOT for negatives:

```cuda
__device__ __forceinline__ unsigned int float_to_ordered_uint(float f) {
  unsigned int u = __float_as_uint(f);
  return (f >= 0.0f) ? (u ^ 0x80000000u) : (~u);
}
```

The corresponding inverse (not used in the forward path, included for completeness):

```cuda
__device__ __forceinline__ float ordered_uint_to_float(unsigned int u) {
  unsigned int raw = (u & 0x80000000u) ? (u ^ 0x80000000u) : (~u);
  return __uint_as_float(raw);
}
```

After global sort, the lower 32 bits are masked off to recover the actual base-sequence indices:

```python
packed_sorted, _ = torch.sort(indices_ptr, descending=False)
actual_indices = (packed_sorted & 0xFFFFFFFF).to(torch.int32)
```

The level-order key encodes both base position **and** sub-level position so siblings of the same parent sort stably:

```cuda
// Rejected coarse entry: key = c_l * level_absolute_idx     (sorts by base position)
unsigned int encode_key(int level_local_idx, int c_l) {
  return float_to_ordered_uint(__int2float_rn(c_l * level_local_idx));
}

// Selected entry: key = c_l * (idx + 1) - 2^(1 - level)
//                                          ^^^^^^^^^^^^ tiny sub-level shift
unsigned int encode_key_selected(int level_local_idx, int c_l, int level) {
  float ordered = __int2float_rn(c_l * (level_local_idx + 1)) - exp2f(1.0f - (float)level);
  return float_to_ordered_uint(ordered);
}
```

The `exp2f(1 - level)` shift is what places level-`l` summaries *just before* the last base position they summarise, implementing the causal-shift `p^l − 1` from §3 stage (iv) at the *index* level.

### 20b. In-register bitonic sort (CUDA, descending, SORT_SIZE elements per CTA)

This is the kernel-internal sort used after each batch merge of the chunked-bitonic top-K. Phase 1 is fully intra-warp (uses `__shfl_xor_sync`); phase 2 mixes shared memory for k>32 with intra-warp shuffles for the inner stages.

```cuda
__device__ __forceinline__ void bitonic_sort_descending(
    float* s_scores, int* s_indices) {
  const int tid = threadIdx.x;

  // Phase 1: intra-warp (k ≤ 32) via warp shuffles.
  { float my_score = s_scores[tid]; int my_index = s_indices[tid];
    #pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
      #pragma unroll
      for (int j = k >> 1; j > 0; j >>= 1) {
        int  ixj = tid ^ j;
        bool desc = ((tid & k) == 0);
        float o_score = __shfl_xor_sync(0xffffffff, my_score, j);
        int   o_index = __shfl_xor_sync(0xffffffff, my_index, j);
        bool swap = (tid < ixj)
            ? (desc ? (my_score < o_score) : (my_score > o_score))
            : (desc ? (my_score > o_score) : (my_score < o_score));
        if (swap) { my_score = o_score; my_index = o_index; }
      }
    }
    s_scores[tid] = my_score; s_indices[tid] = my_index;
    __syncthreads();
  }

  // Phase 2: cross-warp (k > 32). Outer stages use shared memory, inner ≤ 16 use shuffles.
  #pragma unroll
  for (int k = 64; k <= SORT_SIZE; k <<= 1) {
    #pragma unroll
    for (int j = k >> 1; j > 16; j >>= 1) {
      int ixj = tid ^ j;
      if (ixj > tid) {
        bool desc = ((tid & k) == 0);
        float a = s_scores[tid], b = s_scores[ixj];
        bool swap = desc ? (a < b) : (a > b);
        if (swap) {
          s_scores[tid] = b; s_scores[ixj] = a;
          int ti = s_indices[tid];
          s_indices[tid] = s_indices[ixj]; s_indices[ixj] = ti;
        }
      }
      __syncthreads();
    }
    { float my_score = s_scores[tid]; int my_index = s_indices[tid];
      #pragma unroll
      for (int j = 16; j > 0; j >>= 1) {
        int  ixj = tid ^ j;
        bool desc = ((tid & k) == 0);
        float o_score = __shfl_xor_sync(0xffffffff, my_score, j);
        int   o_index = __shfl_xor_sync(0xffffffff, my_index, j);
        bool swap = (tid < ixj)
            ? (desc ? (my_score < o_score) : (my_score > o_score))
            : (desc ? (my_score > o_score) : (my_score < o_score));
        if (swap) { my_score = o_score; my_index = o_index; }
      }
      s_scores[tid] = my_score; s_indices[tid] = my_index;
      __syncthreads();
    }
  }
}
```

### 20c. Chunked-bitonic top-K main loop (CUDA, level-cascade)

The kernel iterates levels coarse-to-fine. Per level, it either consumes the tile span (coarsest) or expands the previous level's surviving parents into `p` children each. At each batch within a level: load `BATCH = TOPK_HALF` new candidates into the upper half of the sort buffer (lower half = running top-m); sort descending; the upper half after sort = rejects (emitted to output for causal boundary), the lower half = new top-m. The QK and KQ streams are merged on the same buffer in alternation.

```cuda
constexpr int POOLING_FACTOR = <P>;
constexpr int TOPK = <TOPK_PER_TILE = 128>;
constexpr int TOPK_HALF = TOPK / 2;
constexpr int BATCH = TOPK_HALF;
constexpr int SORT_SIZE = TOPK;
constexpr int WARMUP_BATCHES = TOPK / BATCH;       // = 2
constexpr int SHIFT = log2(P);                     // bit shift for parent index

extern "C" __global__ void lighthouse_selection_chunked_topk_kernel(
    const float* scores_qk, const float* scores_kq,
    unsigned int* keys_out, int* values_out,
    /* strides */, int num_levels, int pooling_factor,
    int KVH, int N, int N_CHUNK_COARSEST, int NUM_SEL_IDX_CHUNK)
{
  const int tid       = threadIdx.x;
  const long bh       = blockIdx.x;             // batch * KVH
  const long seq      = blockIdx.y;             // tile index along N/p^(L-1)

  /* shared-memory layout */
  extern __shared__ char smem[];
  float* s_topk_scores_qk  = (float*) smem;
  int*   s_topk_indices_qk = (int*)  (s_topk_scores_qk  + TOPK_HALF);
  float* s_topk_scores_kq  = (float*)(s_topk_indices_qk + TOPK_HALF);
  int*   s_topk_indices_kq = (int*)  (s_topk_scores_kq  + TOPK_HALF);
  int*   s_parent          = (int*)  (s_topk_indices_kq + TOPK_HALF);
  float* s_sort_scores     = (float*)(s_parent          + TOPK);
  int*   s_sort_indices    = (int*)  (s_sort_scores     + SORT_SIZE);

  if (tid < TOPK) s_parent[tid] = -1;
  __syncthreads();

  for (int level = num_levels; level >= 1; --level) {
    int c_l = pow(pooling_factor, level - 1);
    int K_end = N / c_l - 1;
    int level_offset = (level <= 1) ? 0
        : (int)((float)N * (pooling_factor - powf(pooling_factor, 2 - level))
                            / (pooling_factor - 1.0f));
    int  num_K_indices;
    bool use_parents;
    if (level == num_levels) { num_K_indices = N_CHUNK_COARSEST; use_parents = false; }
    else                     { num_K_indices = POOLING_FACTOR * TOPK; use_parents = true; }
    int K_end_abs = (K_end >= 0) ? K_end + level_offset : -1;

    if (level > 1) {
      // init running top-m buffers
      if (tid < TOPK_HALF) {
        s_topk_scores_qk[tid]  = -1e6f; s_topk_indices_qk[tid] = -1;
        s_topk_scores_kq[tid]  = -1e6f; s_topk_indices_kq[tid] = -1;
      } __syncthreads();

      int total_batches = (num_K_indices + BATCH - 1) / BATCH;
      for (int batch = 0; batch < total_batches; ++batch) {
        int kv_start = batch * BATCH;

        /* === QK stream merge === */
        // lower half = running top; upper half = new candidates
        if (tid < TOPK_HALF) {
          s_sort_scores[tid]  = s_topk_scores_qk[tid];
          s_sort_indices[tid] = s_topk_indices_qk[tid];
        } else {
          int bi = tid - TOPK_HALF;
          int abs_col = kv_start + bi;
          int K_idx = -1;
          if (abs_col < num_K_indices) {
            if (use_parents) {
              int pidx = abs_col >> SHIFT;
              int coff = abs_col & (POOLING_FACTOR - 1);
              int pval = (pidx < TOPK) ? s_parent[pidx] : -1;
              if (pval >= 0) {
                K_idx = pval * POOLING_FACTOR + coff;
                if (K_idx > K_end || K_idx < 0) K_idx = -1;
              }
            } else {
              K_idx = (seq * N_CHUNK_COARSEST) + abs_col;
            }
          }
          int K_abs = (K_idx >= 0) ? K_idx + level_offset : -1;
          bool valid = (K_abs >= 0) && (K_abs <= K_end_abs) && (K_end >= 0);
          s_sort_scores[tid]  = valid ? __ldg(&scores_qk[K_abs * stride_sqk_N]) : -1e6f;
          s_sort_indices[tid] = K_idx;
        }
        __syncthreads();
        bitonic_sort_descending(s_sort_scores, s_sort_indices);
        if (tid < TOPK_HALF) {
          s_topk_scores_qk[tid]  = s_sort_scores[tid];
          s_topk_indices_qk[tid] = s_sort_indices[tid];
        }
        int rej_qk_idx = (tid >= TOPK_HALF) ? s_sort_indices[tid] : -1;
        __syncthreads();

        /* === KQ stream merge over QK rejects === */
        if (tid < TOPK_HALF) {
          s_sort_scores[tid]  = s_topk_scores_kq[tid];
          s_sort_indices[tid] = s_topk_indices_kq[tid];
        } else {
          int rej_abs = (rej_qk_idx >= 0) ? rej_qk_idx + level_offset : -1;
          bool kq_valid = (rej_abs >= 0) && (rej_abs <= K_end_abs) && (K_end >= 0);
          s_sort_scores[tid]  = kq_valid ? __ldg(&scores_kq[rej_abs * stride_skq_N]) : -1e6f;
          s_sort_indices[tid] = rej_qk_idx;
        }
        __syncthreads();
        bitonic_sort_descending(s_sort_scores, s_sort_indices);
        if (tid < TOPK_HALF) {
          s_topk_scores_kq[tid]  = s_sort_scores[tid];
          s_topk_indices_kq[tid] = s_sort_indices[tid];
        }

        /* === emit causal-boundary rejects after WARMUP_BATCHES === */
        if (batch >= WARMUP_BATCHES && tid >= TOPK_HALF) {
          int dr_idx = s_sort_indices[tid];
          int dr_abs = (dr_idx >= 0) ? dr_idx + level_offset : -1;
          bool dr_valid = (dr_abs >= 0) && (dr_abs <= K_end_abs) && (K_end >= 0);
          if (dr_valid) write_encoded(keys_out, values_out,
              out_off + (tid - TOPK_HALF), encode_key(dr_idx, c_l), dr_abs);
        }
        if (batch >= WARMUP_BATCHES) out_off += BATCH;

        /* === on last batch, emit the level's selected top-K === */
        if (batch == total_batches - 1) {
          __syncthreads();
          int topk_idx = (tid < TOPK_HALF) ? s_topk_indices_qk[tid]
                                           : s_topk_indices_kq[tid - TOPK_HALF];
          int topk_abs = (topk_idx >= 0) ? topk_idx + level_offset : -1;
          if (topk_idx >= 0) write_encoded(keys_out, values_out,
              out_off + tid, encode_key_selected(topk_idx, c_l, level), topk_abs);
          out_off += TOPK;
        }
        __syncthreads();
      }
    } else {
      // Level 1 (base): every surviving child is kept; no selection.
      int total_batches_l1 = (num_K_indices + TOPK - 1) / TOPK;
      for (int batch = 0; batch < total_batches_l1; ++batch) {
        int abs_col = batch * TOPK + tid;
        int K_idx = -1;
        if (abs_col < num_K_indices) {
          int pidx = abs_col >> SHIFT;
          int coff = abs_col & (POOLING_FACTOR - 1);
          int pval = (pidx < TOPK) ? s_parent[pidx] : -1;
          if (pval >= 0) {
            K_idx = pval * POOLING_FACTOR + coff;
            if (K_idx > K_end || K_idx < 0) K_idx = -1;
          }
        }
        int K_abs = (K_idx >= 0) ? K_idx + level_offset : -1;
        bool valid = (K_abs >= 0) && (K_abs <= K_end_abs) && (K_end >= 0);
        if (valid) write_encoded(keys_out, values_out,
            out_off + tid, encode_key(K_idx, c_l), K_abs);
        out_off += min(TOPK, num_K_indices - batch * TOPK);
        __syncthreads();
      }
    }

    // promote this level's survivors to next iteration's parents
    if (tid < TOPK_HALF)        s_parent[tid] = s_topk_indices_qk[tid];
    else if (tid < TOPK)        s_parent[tid] = s_topk_indices_kq[tid - TOPK_HALF];
    __syncthreads();
  }
}
```

### 20d. Selection-index output buffer size

```python
def compute_output_indices_selection(N, num_levels, pooling_factor, topk):
    total = 0
    for level in range(num_levels, 0, -1):
        c_l = pooling_factor ** (level - 1)
        if level == num_levels:
            total += N // c_l                            # coarsest level retained whole
            assert (N // c_l) % pooling_factor == 0
        else:
            total += pooling_factor * topk               # K selected + (p-1)·K causal-boundary rejects
    return total
```

Result tensor: `(B, KVH, sel_idx)` int32 (after the lower-32-bit mask of the int64 packed result).

### 20e. Triton scatter-back forward (fp atomic, default)

```python
@triton.jit
def _scatter_fanout_fwd_kernel(
    attn_out_ptr, indices_ptr, rank_ids_ptr, output_ptr,
    N_shard, N_total, S_sel, D,
    num_levels: tl.constexpr, pooling_factor: tl.constexpr,
    MAX_PF: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bh     = tl.program_id(0)
    s      = tl.program_id(1)
    flat_s = bh * S_sel + s
    valid  = s < S_sel

    idx     = tl.load(indices_ptr + flat_s, mask=valid, other=-1)
    valid   = valid & (idx >= 0)
    rank_id = tl.load(rank_ids_ptr + flat_s, mask=valid, other=0)
    global_off = rank_id * N_shard

    # Decode (level, local_pos) from the cumulative level layout:
    # level l contributes N_shard/p^l entries, packed in order l=0,1,...,L-1.
    base_pos = 0; fan_out = 1; cum_size = 0; pf_l = 1
    for _l in tl.static_range(num_levels):
        level_size  = N_shard // pf_l
        level_start = cum_size
        level_end   = cum_size + level_size
        in_level    = (idx >= level_start) & (idx < level_end)
        local_pos   = idx - level_start
        bp          = global_off + local_pos * pf_l
        base_pos    = tl.where(in_level, bp,   base_pos)
        fan_out     = tl.where(in_level, pf_l, fan_out)
        cum_size    = level_end
        pf_l        = pf_l * pooling_factor

    # Output range: [base_pos + (fan_out - 1), base_pos + (2*fan_out - 2)]
    c_offs        = tl.arange(0, MAX_PF)                        # [MAX_PF]
    d_offs        = tl.arange(0, BLOCK_D)                       # [BLOCK_D]
    out_positions = base_pos + (fan_out - 1) + c_offs           # [MAX_PF]
    c_mask        = valid & (c_offs < fan_out) & \
                    (out_positions >= 0) & (out_positions < N_total)
    mask_2d       = c_mask[:, None] & (d_offs[None, :] < D)

    ao_val = tl.load(attn_out_ptr + flat_s * D + d_offs,
                     mask=valid & (d_offs < D), other=0.0)      # [BLOCK_D]
    ao_2d  = tl.broadcast_to(ao_val[None, :], [MAX_PF, BLOCK_D])

    ptrs = output_ptr + bh * N_total * D \
           + out_positions[:, None] * D + d_offs[None, :]
    tl.atomic_add(ptrs, ao_2d, mask=mask_2d, sem="relaxed", scope="gpu")
```

Backward is the dual — read from `grad_output` at the same `[base+fan-1, base+2fan-2]` range, sum across the fan-out dim, store at `grad_attn[s]`. No atomic needed.

### 20f. `LighthouseLocal.forward` (canonical non-CP module, verbatim)

```python
class LighthouseLocal(nn.Module):
    def __init__(self, attention, scorer, num_levels=3, pooling_factor=4, topk=1024):
        super().__init__()
        self.scorer = scorer
        self.num_levels    = num_levels
        self.pooling_factor = pooling_factor
        self.topk          = topk
        self.attention     = attention

    def forward(self, xq, xk, xv):                          # all (B, S, H, D)
        _, s, _, d = xq.shape

        # (1) pyramid + scoring — returns concatenated multi-level tensors.
        scores_qk_cat, scores_kq_cat, xq_cat, xk_cat, xv_cat = self.scorer(xq, xk, xv)
        # scores_*_cat : (B, H, S_pyr) ; x*_cat : (B, H, S_pyr, D)

        # (2) chunked-bitonic top-K — non-differentiable.
        with torch.no_grad():
            indices = LighthouseSelectionQkKqChunkedTopkNvrtc.forward(
                scores_qk_cat.to(torch.float32),
                scores_kq_cat.to(torch.float32),
                self.num_levels, self.pooling_factor, self.topk, s,
            )                                                # (B, H, S_sel) int32

        # (3) gather the selected (Q, K, V) triples.
        idx4 = indices.unsqueeze(-1).expand(-1, -1, -1, d)
        selected_xq = torch.gather(xq_cat, dim=2, index=idx4)
        selected_xk = torch.gather(xk_cat, dim=2, index=idx4)
        selected_xv = torch.gather(xv_cat, dim=2, index=idx4)

        # (4) stock FlashAttention on the contiguous dense sub-sequence.
        attn_out = self.attention(selected_xq, selected_xk, selected_xv)

        # (5) scatter-back to the N-token base sequence; rank_ids=0 for non-CP.
        rank_ids = torch.zeros_like(indices)
        output   = scatter_to_base_sequence(
            attn_out, indices, rank_ids,
            s, s,                                            # N_shard = N_total = s
            self.num_levels, self.pooling_factor,
        )
        return output                                         # (B, H, N, D)
```

### 20g. `LighthouseCP.forward` and the head/tail split

PyTorch's experimental CP API does **causal load-balancing**: each rank holds two non-contiguous chunks of the sequence — the first `s/(2W)` tokens of one half and the last `s/(2W)` tokens of the other — so that all ranks see roughly the same amount of work despite the causal mask. Lighthouse mirrors that split exactly:

```python
class LighthouseCP(nn.Module):
    def __init__(self, attention, scorer, num_levels=3, pooling_factor=4, topk=1024):
        super().__init__()
        self.scorer, self.attention = scorer, attention
        self.num_levels, self.pooling_factor, self.topk = num_levels, pooling_factor, topk
        self._rank = 0; self._world_size = 1; self._cp_group = None

    def set_cp_info(self, rank, world_size, cp_group):
        self._rank, self._world_size, self._cp_group = rank, world_size, cp_group

    def forward(self, xq, xk, xv):
        _, s, _, d = xq.shape
        half = s // 2
        # Each shard sees a half-sequence. The *2 below funds the head/tail
        # load-balance split; the *world_size aggregates back to a global topk.
        topk_per_load = self.topk // (self._world_size * 2)

        # Two independent scorer+topk passes — head and tail halves.
        sqk_h, skq_h, xq_h, xk_h, xv_h = self.scorer(xq[:,:half], xk[:,:half], xv[:,:half])
        sqk_t, skq_t, xq_t, xk_t, xv_t = self.scorer(xq[:,half:], xk[:,half:], xv[:,half:])

        with torch.no_grad():
            idx_h = LighthouseSelectionQkKqChunkedTopkNvrtc.forward(
                sqk_h.float(), skq_h.float(),
                self.num_levels, self.pooling_factor, topk_per_load, half)
            idx_t = LighthouseSelectionQkKqChunkedTopkNvrtc.forward(
                sqk_t.float(), skq_t.float(),
                self.num_levels, self.pooling_factor, topk_per_load, half)

        gather = lambda x_h, x_t, ih, it: torch.cat([
            torch.gather(x_h, 2, ih.unsqueeze(-1).expand(-1,-1,-1,d)),
            torch.gather(x_t, 2, it.unsqueeze(-1).expand(-1,-1,-1,d)),
        ], dim=2)
        sel_xq = gather(xq_h, xq_t, idx_h, idx_t)
        sel_xk = gather(xk_h, xk_t, idx_h, idx_t)
        sel_xv = gather(xv_h, xv_t, idx_h, idx_t)

        # Stock FlashAttention; under CP this runs through ring attention because
        # the surrounding context_parallel(...) ctx-mgr is active.
        attn_out = self.attention(sel_xq, sel_xk, sel_xv)
        attn_h, attn_t = attn_out.chunk(2, dim=2)

        # Scatter each half back independently into its half-sized output.
        out_h = scatter_to_base_sequence(attn_h, idx_h, torch.zeros_like(idx_h),
                                         half, half, self.num_levels, self.pooling_factor)
        out_t = scatter_to_base_sequence(attn_t, idx_t, torch.zeros_like(idx_t),
                                         half, half, self.num_levels, self.pooling_factor)
        return torch.cat([out_h, out_t], dim=2)
```

The `Attention.forward` only calls `selection_attn(xq, xk, xv)` — the `context_parallel(...)` ctx-mgr around the train step is what gets the underlying `self.attention(...)` call to use ring attention. Lighthouse adds no ring-aware code path.

### 20h. The two scorers, verbatim

`norm` (default, CP-compatible):

```python
class DialatedLighthouseNormScorer(torch.nn.Module):
    def __init__(self, pooling_factor=4, levels=3):
        super().__init__()
        self.pooling_factor, self.levels = pooling_factor, levels

    def forward(self, xq, xk, xv):                          # (B, S, H, D)
        b, s, h, d = xq.shape
        scores_qk = xq.norm(dim=-1)                          # (B, S, H)
        scores_kq = xk.norm(dim=-1)
        xq_l, xk_l, xv_l = [xq], [xk], [xv]
        sqk_l, skq_l     = [scores_qk], [scores_kq]
        for level in range(1, self.levels):
            pf = self.pooling_factor ** level
            ll = s // pf
            xq_l.append(xq[:,:ll*pf].view(b, ll, pf, h, d).mean(2))
            xk_l.append(xk[:,:ll*pf].view(b, ll, pf, h, d).mean(2))
            xv_l.append(xv[:,:ll*pf].view(b, ll, pf, h, d).mean(2))
            sqk_l.append(scores_qk[:,:ll*pf].view(b, ll, pf, h).max(2).values)
            skq_l.append(scores_kq[:,:ll*pf].view(b, ll, pf, h).max(2).values)
        # Output layout: heads first, then concatenated pyramid along S.
        return (torch.cat(sqk_l, dim=1).transpose(1, 2),     # (B, H, S_pyr)
                torch.cat(skq_l, dim=1).transpose(1, 2),
                torch.cat(xq_l,  dim=1).transpose(1, 2),     # (B, H, S_pyr, D)
                torch.cat(xk_l,  dim=1).transpose(1, 2),
                torch.cat(xv_l,  dim=1).transpose(1, 2))
```

`dilated` (Q–K interaction, non-CP):

```python
class DilatedLighthouseScorer(torch.nn.Module):
    def __init__(self, attention, dilation=4, pooling_factor=4, levels=3):
        super().__init__()
        self.dilation, self.pooling_factor, self.levels = dilation, pooling_factor, levels
        self.attention = attention                            # same SDPA wrapper as outer

    def forward(self, xq, xk, xv):
        b, s, h, d = xq.shape
        # Reshape (B, S, H, D) → (B, S/δ, δ·H, D): groups δ tokens per "wide head".
        xq2 = xq.view(b, s // self.dilation, self.dilation * h, d)
        xk2 = xk.view(b, s // self.dilation, self.dilation * h, d)
        xv2 = xv.view(b, s // self.dilation, self.dilation * h, d)

        # Two attention passes: Q→K and K→Q. ‖output‖₂ is the score.
        out_qk = self.attention(xq2.transpose(1,2), xk2.transpose(1,2), xv2.transpose(1,2))
        out_kq = self.attention(xk2.transpose(1,2), xq2.transpose(1,2), xv2.transpose(1,2))
        out_qk = out_qk.transpose(1,2).view(b, s, h, d)
        out_kq = out_kq.transpose(1,2).view(b, s, h, d)
        scores_qk = out_qk.norm(dim=-1).transpose(1, 2)       # (B, H, S)
        scores_kq = out_kq.norm(dim=-1).transpose(1, 2)

        # Rebuild the full multi-level (Q, K, V) and score tensors as before.
        # (Pyramid pool branch identical to the norm scorer; omitted here.)
        ...
        return scores_qk_cat, scores_kq_cat, xq_cat, xk_cat, xv_cat
```

GLA scorer adds the per-token gate `wg(x)` and uses `fla.ops.simple_gla.chunk_simple_gla` instead of softmax attention.

### 20i. Consolidated SDPA-Resume Procedure

The two stages are wired together by file-level state, not by code paths. Step-by-step:

1. **Train Stage 1** with the Lighthouse flavor (e.g. `ablation_270m_lighthouse_topk1536_pool4_lvl3`) for 10,000 steps. The flavor has `use_selection_lighthouse=True` and `lighthouse_full_attn_layers=[0, 1, 28, 29]`.
2. **Checkpoint** at step 10,000. Both model weights *and* optimizer state must be saved (`last_save_model_only = false`).
3. **Create the Stage 2 toml**: copy the Stage 1 toml and change:
   - `[model] flavor = ablation_270m_topk1536_pool4_lvl3_sdpa` (the dim-matched dense variant — same `dim`, `hidden_dim`, `n_heads`, `n_kv_heads`, `rope_theta`, but no `use_selection_lighthouse` and no Lighthouse-specific keys).
   - `[training] steps = 16000` (total step count continues — the trainer counts from the checkpointed step, not from zero).
   - `[checkpoint] folder` points at the *same* directory as Stage 1.
   - Optionally lower `[lr_scheduler]` so the resume tail anneals (paper uses the original linear schedule continuation).
4. **Launch Stage 2** with the same `torchrun` command. The trainer detects the existing checkpoint, restores weights + optimizer state + dataloader position, and continues. Because the new flavor has no `use_selection_lighthouse`, `Attention.forward` falls into the `else` branch and runs ordinary SDPA on all 30 layers.
5. **Loss spike at resume** (1.12–1.57 nats) recovers within ~1–1.5k SDPA steps. Final loss at step 16,000 matches or beats dense-from-scratch.

**Why this works**: the Lighthouse and dense flavors share the same `wq, wk, wv, wo, w1, w2, w3, norm, output, tok_embeddings` parameter names and shapes. The Lighthouse-specific module (`self.selection_attn`) has no learnable parameters in the `norm` scorer case; in the GLA case the `wg` projection is simply ignored on resume and is *not* loaded into the dense flavor (HF checkpoint loader skips missing keys). The model is genuinely a dense Transformer after the toggle.

**Alternative in-process toggle** (not used by the paper but supported by the patch): keep the Lighthouse flavor and call `model.set_use_lighthouse(False)` on every `Attention` module mid-training. This skips the checkpoint round-trip but requires retraining-loop instrumentation; the file-level approach is what the configs ship.

### 20j. Minimum tensor-shape contract for a reimplementation

| Tensor                 | Shape                          | Dtype         | Notes                                  |
|------------------------|--------------------------------|---------------|----------------------------------------|
| `xq, xk, xv` (input)   | `(B, S, H, D)`                 | bf16          | per-head, post-RoPE                     |
| `scores_qk_cat`        | `(B, H, S_pyr)`                | bf16 → fp32   | `S_pyr = Σ_{l=0..L-1} N/p^l`            |
| `xq_cat, xk_cat, xv_cat` | `(B, H, S_pyr, D)`            | bf16          | levels concatenated along seq dim      |
| `indices` (after kernel) | `(B, H, S_sel)`              | int32         | `S_sel = N/p^{L-1} + (L-1)·p·k`         |
| `selected_xq/xk/xv`    | `(B, H, S_sel, D)`             | bf16          | output of `torch.gather`                |
| `attn_out`             | `(B, H, S_sel, D)`             | bf16          | FlashAttention output, causal mask     |
| `output` (final)       | `(B, H, N, D)`                 | bf16          | scatter-back result; contiguous & dense |

Constraints:
- `p^{L-1} | N` (coarsest level must tile the sequence).
- `topk % 128 == 0` and `topk // 2 ≥ p` (kernel asserts).
- `p` is a power of 2 (kernel uses bit shifts for parent-index decoding).
- `H % world_size == 0` when CP is enabled (head-parallel all-to-all).

### 20k. The five non-obvious correctness invariants

If a port doesn't preserve these, the recoverability claim falls apart:

1. **Coarsest level is retained in full** — every base position has at least one contributor in the gathered sub-sequence. Without this, scatter-back leaves zeros at positions whose coarse summary was rejected, training is destabilised, and the SDPA-resume diverges.
2. **Rejected coarse entries are kept as causal-boundary filler** — emit them to the output buffer at `WARMUP_BATCHES` onward, alongside selected ones. Removing this would leave the gathered sub-sequence with holes at level-`l` boundaries.
3. **Topological sort by base position** after the kernel — the `torch.sort` over uint64 keys is what makes the standard lower-triangular causal mask correct. Skipping it would force a sparse-aware mask.
4. **Scatter shift of `p^l − 1`** — coarse summaries write to `[base + p^l − 1, base + 2p^l − 2]`, *not* `[base, base + p^l − 1]`. The shift is what stops a base token from receiving a summary that contains its own future.
5. **Top-K is `torch.no_grad()`** — indices must carry no gradient. Adding a straight-through estimator or Gumbel-softmax breaks the empirical result (the authors specifically call this out as a design choice).
