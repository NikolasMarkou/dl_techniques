# AU-Net: Autoregressive U-Net for Language Modeling

**Paper:** *From Bytes to Ideas: Language Modeling with Autoregressive U-Nets* (arXiv:2506.14761, June 17 2025)
**Authors:** Mathurin Videau, Badr Youbi Idrissi, Marc Schoenauer, Olivier Teytaud, David Lopez-Paz, et al.
**Affiliations:** Meta FAIR; TAU, INRIA & LISN, CNRS & Université Paris-Saclay; INSA Rouen Normandy, LITIS.
**Code:** `github.com/facebookresearch/lingua/tree/main/apps/aunet` (Meta Lingua framework).

---

## 1. Motivation

BPE-style tokenizers impose a **fixed, irrevocable granularity** before training. Consequences:

1. Static vocabulary freezes the unit of prediction (next token, always).
2. Embedding table treats morphologically related tokens as unrelated atoms (`run`, `runs`, `running`).
3. Poor cross-lingual / low-resource coverage; brittle on character-level tasks (spelling, edits, anagrams).
4. Artifacts (SolidGoldMagikarp-style glitch tokens, digit splitting, whitespace oddities).

**AU-Net's thesis:** move tokenization *inside the model* as a learned, hierarchical, multi-scale operation over raw bytes. Train once; the model chooses its own semantic granularity at each depth.

---

## 2. Architecture

### 2.1 Overall shape

A U-Net applied to a 1D byte sequence with strict causal masking:

```
Stage 1 (bytes)      ──────────────────────────────────────────────▶ Stage 1'
                         │                                     ▲
                         ▼ pool (at word boundaries)           │ upsample + skip
                     Stage 2 (words)     ───────────────▶  Stage 2'
                         │                              ▲
                         ▼ pool (every 2 words)         │
                     Stage 3 (word-pairs) ──▶ Stage 3'
                         │                  ▲
                         ▼ pool (every 4 words)
                     Stage 4 (4-word chunks, deepest)
```

- **Contracting path:** successive pooling ops shorten the sequence, each stage operating at a coarser linguistic unit.
- **Expanding path:** Multi-Linear Upsampling restores length; skip connections from contracting path inject fine-grained info.
- **Monolithic, not a stack of local models:** attention is global at every stage (windowed only at byte-Stage 1 for compute reasons). This is the key difference vs. MegaByte / BLT / SpaceByte, which process chunks locally.
- Each stage is itself a transformer block stack (attention + FFN, RoPE, RMSNorm, standard LLaMA-style components — AU-Net is a *scheduling* architecture wrapping transformers, not a replacement for attention).

### 2.2 Splitting function (the critical piece)

The splitting function defines *where* pooling happens. It must be **causal**: future bytes cannot alter past split decisions, or autoregressive generation breaks.

Rule used in the paper (Latin scripts, regex-based, inspired by GPT-4o pre-tokenization):

| Stage | Unit | Split rule |
|-------|------|-----------|
| 1 | byte | every byte (no pool) |
| 2 | word | at whitespace / word boundary regex |
| 3 | 2-word | at end of every 2nd word, or sentence end |
| 4 | 4-word | at end of every 4th word, or sentence end |

The splitting function is **user-defined** and modular; alternatives (fixed windows, entropy-based like BLT, learned) are compatible. Non-space-delimited languages (CJK, etc.) are flagged as future work.

### 2.3 Pooling

"Pooling" is literally **index selection**: at split positions, take that vector, project to the next stage's hidden dim via a linear layer. No attention pooling, no averaging. Simplicity is intentional — keeps the contraction factor exact and causally clean.

### 2.4 Multi-Linear Upsampling

Coarse → fine expansion must generate multiple fine positions from one coarse vector **without leaking future info**.

For a coarse vector `c` that must expand into `m` fine positions:

```
fine[i] = W_i · c + b_i        for i = 0 .. m-1
```

- `W_i`, `b_i` are **position-specific** linear maps, *shared across segments* but *distinct within a segment*.
- Output is merged with the skip connection from the matching contracting stage via residual addition.
- Effect: each fine slot gets a different "view" of the same coarse concept — implicit multi-token prediction without auxiliary heads or losses.

### 2.5 Causal masking, end to end

The non-trivial invariant: pooling + upsampling + skip must not let byte position `t` see anything derived from bytes `> t`.

- Splitting is defined on past bytes only.
- At each stage, attention uses a causal mask on the coarser sequence.
- The "latest vector at each stage must be cached" point is the inference-time corollary: when a new byte arrives, only the currently-open segment at each stage is recomputed; finalised segments are frozen.

### 2.6 Training vs. inference

- **Training:** full sequence processed in parallel; all stages active simultaneously; single next-byte cross-entropy loss at the output (plus the implicit multi-scale prediction baked into deeper stages seeing larger future horizons via the expanding path).
- **Inference:** autoregressive byte-by-byte, but most stages only advance when a split boundary is crossed. Deeper stages fire rarely ⇒ amortised cost per byte is low.

### 2.7 Complexity

- FLOPs/byte for AU-Net = Σ over stages of (stage FLOPs) / (cumulative contraction factor `k_i`).
- Deeper stages run on short sequences ⇒ their quadratic attention cost is cheap.
- **Overall scaling is linear in sequence length** (windowed attention at byte stage; short sequences at deeper stages), vs. quadratic for flat byte transformers.

---

## 3. Training setup

- **Corpus:** DCLM (DataComp-LM), ~4T tokens in the full scaling regime; controlled ablations at 60B and 370B LLaMA-3 tokens.
- **Compute-matched comparison:** because BPE ≈ 4.56 bytes/token on DCLM (LLaMA 3 tokenizer), a byte model trained on 273B bytes is compared to a BPE model on 60B tokens ⇒ identical FLOP budget and identical data-seen.
- **Scaling-law methodology (Bi et al. 2024):** instead of parameter count, model size is measured in FLOPs per input unit. Hyperparameters (batch size `BSZ(C) = A·C^α`, learning rate `LR(C) = B·C^β`) are fit from small-scale sweeps (25M–500M) and extrapolated. Byte-level training required *new* BSZ/LR formulas vs. token-level — non-trivial finding on its own.
- **Data-to-model ratio `γ`:** expressed in LLaMA-3 token equivalents (`γ_token`) so byte and BPE runs are on the same axis; paper reports a ~2 tokens-per-param regime for the main sweep.

---

## 4. Results

### 4.1 Headline numbers

| Config | HellaSwag | ARC-Easy | TriviaQA | MMLU | GSM8K |
|---|---|---|---|---|---|
| BPE transformer (matched) | baseline | baseline | baseline | baseline | baseline |
| AU-Net-2 (2 stages) | ≈ | ≈ | ≈ | ↓ | ↓ |
| AU-Net-3 (3 stages) | ≈ / ↑ | ≈ / ↑ | ≈ / ↑ | ↓ | ↓ |
| AU-Net-4 (4 stages) | best in family | best in family | best in family | still lags | still lags |

Shallow hierarchies **tie** strong BPE baselines; deeper ones **trend upward**. AU-Net-3 is the sweet spot under the 2-tokens-per-param ratio.

### 4.2 Specific claims from Section 2.3 of the prompt

- **HellaSwag, ARC-Easy, TriviaQA:** AU-Net-2 / AU-Net-3 track the BPE baseline closely.
- **LLaMA 3.1 8B comparison:** AU-Net and the paper's own BPE baseline land "remarkably close" to LLaMA 3.1 8B on several evals at ~100× less compute. Not controlled (LLaMA 3.1 used different data mix, instruction-tuning pipeline, etc.) — directional only.
- **MMLU and GSM8K:** AU-Net underperforms. The interpretation offered: these capabilities emerge **later on the compute curve** for hierarchical byte models than for flat BPE. Math especially suffers because BPE's digit-level tokenization accidentally encodes structure that AU-Net must learn.
- **Diminishing returns past 3 stages** under γ ≈ 2 tokens/param. More stages help only if more data is given to match the extra capacity.

### 4.3 Character-level / multilingual

- **No BPE = no BPE artifacts.** Spelling, character manipulation, anagram-style tasks: AU-Net significantly outperforms BPE baselines.
- **Multilingual / low-resource:** byte input gives uniform coverage across scripts; no tokenizer-vocab bias toward English. Reported gains on cross-lingual transfer are a consistent theme.

### 4.4 What the paper does *not* claim

- AU-Net does not beat frontier BPE LLMs at equal compute on MMLU / math.
- No controlled ablation vs. LLaMA 3.1 at matched data — the compute-ratio comparison is a vibe check, not a head-to-head.
- Splitting rule is handcrafted regex for Latin scripts; not yet proven on Chinese/Japanese/Arabic at scale.

---

## 5. Positioning vs. other byte / hierarchical models

| Model | Byte-native | Hierarchy | Attention scope | Splitting |
|---|---|---|---|---|
| **MegaByte** (Yu 2023) | ✅ | 2-stage | **local** within fixed block | fixed window |
| **SpaceByte** (Slagle 2024) | ✅ | 2-stage | local | space-based |
| **BLT** (Pagnoni 2024) | ✅ | 2-stage | local | entropy-based |
| **HAT** (Neitemeier 2025) | ✅ | byte + word | local | boundary-based |
| **Hourglass / Nawrot 2022** | ❌ (tokens) | U-Net-like | global | fixed pooling |
| **AU-Net** | ✅ | **2–4 stage**, extensible | **global** at every stage | user-defined (regex in paper) |

Distinguishing AU-Net features: (a) global attention at every depth, (b) input-adaptive pooling, (c) multiple stages (not just 2), (d) monolithic rather than a cascade of local encoders.

---

## 6. Reference-level implementation sketch (PyTorch-ish pseudocode)

```python
class AUNetStage(nn.Module):
    """One transformer stack at a given hierarchy level."""
    def __init__(self, d_model, n_layers, n_heads, window=None):
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, window)
                                      for _ in range(n_layers)])
    def forward(self, x, causal_mask):
        for b in self.blocks:
            x = b(x, causal_mask)
        return x

class MultiLinearUpsample(nn.Module):
    """Expand 1 coarse vector → m fine vectors with position-specific linears."""
    def __init__(self, d_in, d_out, max_segment_len):
        self.W = nn.Parameter(torch.randn(max_segment_len, d_in, d_out))
        self.b = nn.Parameter(torch.zeros(max_segment_len, d_out))
    def forward(self, coarse, segment_lengths):
        # for each coarse vec c_j producing fine slots 0..L_j-1:
        #   fine[j, i] = c_j @ self.W[i] + self.b[i]
        ...

def splitting_function_stage2(byte_seq):
    # regex: boundaries at whitespace / punctuation
    return boundary_indices  # strictly causal: depends only on past bytes

class AUNet(nn.Module):
    def __init__(self, n_stages=3, d=[512, 1024, 2048], n_layers=[6, 12, 12]):
        self.contract = nn.ModuleList([AUNetStage(d[i], n_layers[i]) for i in range(n_stages)])
        self.expand   = nn.ModuleList([AUNetStage(d[i], n_layers[i]) for i in range(n_stages-1)])
        self.pools    = nn.ModuleList([nn.Linear(d[i], d[i+1]) for i in range(n_stages-1)])
        self.upsamples= nn.ModuleList([MultiLinearUpsample(d[i+1], d[i], max_seg_len)
                                        for i in range(n_stages-1)])

    def forward(self, bytes):
        # contracting path
        h = embed(bytes)
        skips = []
        for i, stage in enumerate(self.contract):
            h = stage(h, causal_mask_for(h))
            skips.append(h)
            if i < len(self.contract) - 1:
                idx = splitting_function(i+1, bytes)   # causal
                h = self.pools[i](h[idx])
        # expanding path
        for i in reversed(range(len(self.expand))):
            h = self.upsamples[i](h, segment_lengths(i))
            h = h + skips[i]                            # residual skip
            h = self.expand[i](h, causal_mask_for(h))
        return lm_head(h)                               # next-byte logits
```

Key invariants an implementer must preserve:

1. `splitting_function` never peeks at future bytes.
2. Causal mask recomputed at each stage for the *post-pooling* sequence length.
3. At inference, cache the latest open-segment vector at each stage; only flush when its boundary fires.
4. Skip connections are residual adds, not concatenations (paper spec).

---

## 7. Takeaways

- **What AU-Net proves:** byte-native LLMs with learned hierarchy can match BPE transformers at equal compute on general benchmarks, and dominate them on character-level and multilingual tasks, *without* a tokenizer.
- **What it does not (yet) prove:** frontier-parity on knowledge-dense (MMLU) and symbolic-reasoning (GSM8K) evals; robustness of handcrafted splitting rules outside Latin scripts.
- **Most transferable artifact:** the scaling-law methodology for byte-level models (new BSZ/LR formulas, FLOPs-per-input-unit-based model-size definition) will likely outlast the specific architecture.
- **Open research hooks:** learned splitting functions, non-Latin scripts, scaling past 4 stages with matched data, combining AU-Net's hierarchy with MoE / SSM inner blocks.
