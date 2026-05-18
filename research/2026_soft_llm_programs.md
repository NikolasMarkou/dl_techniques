# SoftProg: Typed Predicate-Argument Memory for LLMs

**Working title**: SoftProg
**Honest title**: Typed Structured Retrieval with Predicate-Argument Codes for LLM Augmentation
**Status**: Specification (post-adversarial-review). The "soft program" framing is *aspirational* and earns its name only if the compose / substitute / inspect gates (§9) pass. Otherwise the system is honest typed retrieval, which is still a contribution.
**Date**: 2026-05-18
**Author**: Nikolas Markou (review and counter-spec: Claude Opus 4.7)
**Prior work this builds on / displaces**: MEMORY-VQ (NAACL 2024), COCONUT (Meta 2024), RETRO (DeepMind 2021), kNN-LM (Khandelwal 2020), latent-token VQ-VAE (Meta/Berkeley 2025), ROME / MEMIT / AnyEdit knowledge editing.
**Adversarial review parent**: `research/2026_llm_tree_grammar.md`.

---

## 0. One-page summary

**Problem.** Existing retrieval-augmented LMs (RAG, kNN-LM, RETRO, MEMORY-VQ) retrieve over *surface text* or *raw token-state vectors*. Two consequences: (i) paraphrase redundancy — semantically equivalent sentences with different surface forms occupy distinct memory; (ii) opaque retrieval — there is no inspectable structural query language.

**Proposal.** Memory entries are **typed predicate-argument tuples** encoded as discrete codes via a hierarchical-residual VQ over parse subtrees. Codes are trained to be **paraphrase-invariant** (contrastive objective from step 1, not "open problem"). Retrieval uses **two indices in parallel**: dense FAISS (semantic similarity) and a **type-signature inverted index** (precise structural lookup). A **Perceiver-IO** cross-attention block at LLM mid-layer fuses retrieved entries. The LLM backbone (Gemma-3-8B or Qwen-3-8B) is frozen with optional LoRA on the post-fusion layers.

**Operational tests that earn the "program" framing** (§9):
- **G_COMPOSE**: two retrieved entries can be combined to answer a multi-hop query neither covers alone.
- **G_SUBSTITUTE**: replacing one typed slot in an entry changes the LLM's downstream prediction in the expected direction.
- **G_INSPECT**: retrieved entries render to human-readable form (e.g. `caused(raise_rates[Bank_of_England], inflation_target_2pct)`).

If any of G_COMPOSE / G_SUBSTITUTE / G_INSPECT fails, the spec retains its technical contribution (typed retrieval beating MEMORY-VQ at matched index size) but the "soft program" name is dropped publicly. This is non-negotiable: rhetoric without operational content is forbidden.

**Repo components used** (all in `dl_techniques`):
- `models/tree_transformer/` (Wang-soft-constituency encoder)
- `layers/vector_quantizer_rotation_trick.py` + `layers/embedding/hierarchical_codebook_embedding.py` (RVQ stack)
- `layers/attention/perceiver_attention.py` (fusion block)
- `models/gemma/` or `models/qwen/` (LLM backbone)
- `models/memory_bank/`, `layers/memory/ntm_interface.py` (memory addressing patterns, optional)
- `metrics/perplexity_metric.py`, `metrics/llm_metrics.py`

**Greenfield components** (must be written):
- Biaffine relation-classifier head over Wang-induced spans (recover deprel / arc labels).
- Predicate-argument decoder (replaces surface-token decoder of vanilla VQ-VAE).
- Type-signature inverted index (RocksDB-backed).
- Memory schema / serialization layer.
- Wikipedia memory pipeline (parse → encode → dedupe → index).
- Update API (single-shot fact insertion).

---

## 0.5. Build Path: MV-SoftProg → Full SoftProg

**Honesty about complexity.** The full architecture in §2-§15 is the *destination*, not the *starting point*. It has ~10 greenfield components, 8 gates, a 5-term loss, a 4-level RVQ stack, two indices, multi-source SRL data, and an 8B-parameter backbone with LoRA. Joint pass rate at P(each gate)=0.75 is ~10%. For a solo researcher on a 4090+4070, building the full system from day 1 is a high-variance bet.

**The core hypothesis is H1** (codes can be made paraphrase-invariant via contrastive + predicate-argument decoding). Everything else is downstream consumption of H1. If H1 fails, none of the architecture matters; if H1 succeeds, complexity is added one component at a time, each justified by a measurement.

### 0.5.1 MV-SoftProg (2 weeks, single 4090, tests H1 only)

| Layer | MV-SoftProg | Full spec (§2-§15) |
|---|---|---|
| Parse | spaCy out-of-the-box | Tree Transformer + biaffine relation head + dual-head NER |
| Subtree composer | mean-pool over subtree node embeddings | Label-aware TreeGRU |
| VQ | single 4096-code rotation-trick VQ | 4-level Hierarchical-Residual VQ (4×1024) |
| Decoder | predict 3 lemma slots: `(predicate_lemma, ARG0_lemma, ARG1_lemma)` from VOCAB, no class clustering | 5-head typed-class decoder (predicate type, 3 argument classes, modifier set) |
| Training data | PropBank gold (~50k) + 1M ParaNMT pairs | + AMR-3.0 + UD + UniversalPropositions + silver-LLaMA-AMR Wikipedia (3M) + Europarl (200k×3 langs) |
| Loss | `L_PA_recon + 1.0·L_paraphrase_contrastive + 0.25·L_commit` | 6 terms (adds crosslingual contrastive, structural probe, entropy regularizer) |
| Memory index | FAISS HNSW dense only | FAISS HNSW-PQ + RocksDB type-signature inverted |
| Fusion | **none** — measure code properties on the encoder side only | Perceiver-IO + LoRA injected at LLM layer N/2 |
| LLM backbone | **none** for the core gate; frozen Gemma-3-2B for one optional downstream sanity check | Gemma-3-8B or Qwen-3-8B + LoRA rank 8 on N/2..N |
| Gates | **G2.1 only** (paraphrase code-tuple overlap ≥ 3× random on held-out paraphrase generator) | G2.1, G2.2, G2.3, G2.4, G_PPL, G_FAITH, G_UPDATE, G_LATENCY, G_COMPOSE, G_SUBSTITUTE, G_INSPECT |
| Timeline | 2 weeks | 8 weeks |
| Compute | single 4090 | 4090 + 4070, sequential GPU per repo convention |

### 0.5.2 MV-SoftProg pass/fail decision tree

- **G2.1 passes (overlap ≥ 3× random)** → unlock the next addition. Pick the next gate that addresses the next-highest-risk hypothesis (see §16 hypothesis table). Add components incrementally, never in bulk.
- **G2.1 marginal (1.5×–3×)** → tune `λ_paraphrase` up, add hard-negative mining, retry. One week of tuning before declaring marginal-fail.
- **G2.1 fails (≤ 1.5×)** → the bottleneck did not learn an abstract language. The whole research program is dead. Two weeks spent. Total saving vs. full spec: 6 weeks + reputation.

### 0.5.3 Incremental expansion rule

After MV passes G2.1, add components in this order, each justified by the gate it addresses:

1. **Add G2.2** (deprel-from-code probe ≥ 80%). Adds: `L_structural_probe`. No new training data. ~3 days.
2. **Add G2.4** (cross-lingual overlap ≥ 2× random). Adds: Europarl pairs + `L_crosslingual_contrastive`. ~1 week.
3. **Add the type-signature inverted index** and `G_INSPECT`. Replaces lemma-slot decoder with typed-class decoder (filler classes). Earns the right to write `caused(raise_rates[Bank_of_England], inflation_target)` rendering. ~1 week.
4. **Add fusion + frozen Gemma-3-2B** for `G_PPL` against a MEMORY-VQ baseline on a 1M-sentence Simple Wikipedia memory. Smaller than 8B, smaller than 80M. ~1 week.
5. **Add `G_UPDATE`** vs ROME on Simple Wikipedia facts. ~3 days.
6. **Scale to Gemma-3-8B + 80M sentences + RVQ + RocksDB + chunked retrieval** only after steps 1-5 succeed. This is what the full spec describes.

### 0.5.4 What this discipline buys

- **Bounded loss on failure**: each step is killable independently. The 2-week MV pilot risks 2 weeks, not 8.
- **Measurement-driven complexity**: every added component points at a passing/failing gate, not at an aesthetic.
- **Compound risk drops sharply**: P(MV passes) × P(step 1 | MV passes) × … is a chain of conditional probabilities you actively maintain, not a flat product.
- **Honest deliverable at any stop point**: failing at step 4 still gives you a paper on "paraphrase-invariant predicate-argument codes" (steps 1-3), which is a contribution in its own right.

The rest of this spec (§1-§16) describes the destination. Treat it as architecture, not as a checklist for week 1.

---

## 1. Design Principles (binding)

1. **The objective specifies the abstraction.** No "hope it emerges." Paraphrase-invariance is a contrastive *loss term* from step 1, not a probing diagnostic.
2. **Predicate-argument structure, not surface tokens.** The VQ decoder reconstructs typed tuples, not subwords.
3. **Operational tests gate the framing.** G_COMPOSE, G_SUBSTITUTE, G_INSPECT are required for the "program" claim; perplexity gates are not.
4. **Frontload the cheapest informative experiment.** Week 1 is paraphrase code overlap + cross-lingual overlap on synthetic data. If they fail at 100k-pair scale, the whole pipeline is dead and we have saved 7 weeks.
5. **Honest baselining.** MEMORY-VQ (16x compression, KILT-parity) is the principal baseline. kNN-LM is too old.
6. **Honest naming.** If a component reduces to a known technique (e.g. typed retrieval = kNN over a learned typed embedding), say so.
7. **Latency is a gate, not a footnote.** End-to-end p50 ≤ 25ms / token at k=10, p99 ≤ 60ms / token. If breached, re-architect.
8. **Updatability is a gate.** Insert a fact via memory write; the LLM must use it correctly within one inference call. ROME/MEMIT comparison is the deliverable.

---

## 2. Architecture

```
                    raw text input
                          |
                          v
              +-----------------------+
              | Tree Transformer      |  (Wang soft constituency, repo)
              |   + Biaffine Relation |  (NEW; recovers (head, deprel) per span)
              +-----------------------+
                          |
              spans + arcs + entity tags
                          |
                          v
              +-----------------------+
              | Subtree composer      |  TreeGRU, label-aware
              | (head-attended,       |  composition over (child_h, deprel_emb)
              |  arc-label conditioned)
              +-----------------------+
                          |
                node embeddings z_node (d=512)
                          |
                          v
              +-----------------------+
              | Hierarchical-Residual |  (rotation-trick VQ, 4 levels x 1024)
              | VQ stack              |  Level1: predicate type
              |                       |  Level2: argument-type pattern
              |                       |  Level3: filler class
              |                       |  Level4: residual
              +-----------------------+
                          |
              code tuple per subtree node: (c1, c2, c3, c4)
                          |
                          v
              +-----------------------+
              | Predicate-Argument    |  (NEW decoder)
              | Decoder               |  Outputs: typed tuple
              |                       |  (predicate_lemma_cluster,
              |                       |   ARG0_filler_class,
              |                       |   ARG1_filler_class,
              |                       |   modifier_set)
              +-----------------------+
                          |
                          v
            ============================
            Memory entries (one per sentence-root subtree + sub-entries per main predicate)
            ============================
                          |
              +-----------+-----------+
              |                       |
              v                       v
        +----------+         +---------------------+
        |  FAISS   |         |   RocksDB inverted  |
        |  HNSW    |         |   type-signature    |
        |  dense   |         |   index             |
        +----------+         +---------------------+
              |                       |
              +-----------+-----------+
                          |
                          v
              +-----------------------+
              | Perceiver-IO fusion   |  (repo: perceiver_attention)
              | block, M=16 latents,  |
              | retrieves k=10        |
              +-----------------------+
                          |
                          v
              +-----------------------+
              |  LLM backbone         |  (Gemma-3-8B or Qwen-3-8B)
              |  frozen + LoRA rank 8 |
              |  on layers N/2..N     |
              +-----------------------+
                          |
                          v
                    next-token output
```

---

## 3. Stage 1 — Span + Arc Encoder

### 3.1 Architecture

Use `models/tree_transformer/` (Wang et al. unsupervised soft constituency) as the span inducer. Add a **biaffine relation head** that, given two spans (parent candidate, child candidate), predicts the dependency relation label or NULL.

```python
encoder = TreeTransformer.from_variant("base")    # 6 layers, d=512, repo
# Outputs: token embeddings + soft constituency probabilities (group attention)

span_extractor = HardSpanExtractor(             # NEW; thresholded group attention
    group_attention_threshold=0.5,
    max_spans_per_sentence=64,
)

biaffine_relation = BiaffineRelation(           # NEW; small head
    d_model=512,
    mlp_arc=300,
    mlp_rel=100,
    num_rels=37,                                # UD v2 relation set
)

ner_head = SpanNERHead(                         # NEW; dual-track
    primary="span_constrained",
    fallback="token_window_topk",               # breaks parse-subtree recall ceiling
    width_buckets=[1, 2, 3, 5, 8, 15, 30],     # log-width buckets
    num_entity_types=18,                       # OntoNotes 5
)
```

### 3.2 Training

| Field | Value |
|---|---|
| Datasets | UD English-EWT + OntoNotes 5 (joint) + WikiAnn gold (NER held-out) + WikiNER (silver, downweighted 0.3) |
| Encoder init | mDeBERTa-v3-base (NOT frozen BERT; finetune all). Falls back to XLM-R-base if mDeBERTa licensing blocks. |
| Loss | `L = L_constituency_LM + 0.5*L_arc + 0.3*L_rel + 0.3*L_NER`. Constituency LM = Wang's MLM loss over span boundaries. |
| Optimizer | AdamW, lr=2e-4 encoder, 1e-3 heads, warmup 4000 |
| Batch | 32 sentences |
| Epochs | up to 30, early stop on UAS+LAS+F1 average, patience 5 |
| Calibration | Add temperature scaling layer for span-confidence; calibrate on dev set. Required for memory-pipeline confidence filtering. |

### 3.3 Targets

| Metric | Target | Rationale |
|---|---|---|
| UAS (UD English dev) | ≥ 91 | Achievable with mDeBERTa + Wang induction. Below SOTA biaffine-on-finetuned-encoder (~94); acceptable trade for unified architecture. |
| LAS | ≥ 88 | Same caveat. |
| Span F1 (NER, OntoNotes test) | ≥ 86 | Dual-head with fallback breaks the parse-subtree recall ceiling. |
| Confidence calibration ECE | ≤ 0.05 | So that the 0.85 filter at memory time means something. |

### 3.4 Stage 1 gates

| Gate | Threshold | Action if fail |
|---|---|---|
| G1.1 | LAS ≥ 88 on UD English dev | Switch to fully-supervised biaffine (Dozat) — abandon the unsupervised-span fast path. Cost: +1 week training. |
| G1.2 | NER F1 ≥ 85 dual-head | Inspect span-coverage ratio; if parse-spans cover < 80% of gold entities, the fallback head must dominate. |
| G1.3 | ECE ≤ 0.05 | Recalibrate; without this, downstream filter is noise. |

---

## 4. Stage 2 — Hierarchical-Residual VQ over Subtrees

### 4.1 Architecture

```python
subtree_composer = LabelAwareTreeGRU(
    d_model=512,
    deprel_embed_dim=64,     # arc-label conditioning
    composition="head_attended_weighted_sum",
    max_depth=10,
)

rvq_stack = HierarchicalResidualVQ(             # repo: vector_quantizer_rotation_trick
    levels=[                                    #       + hierarchical_codebook_embedding
        {"size": 1024, "d": 512, "semantic_role": "predicate_type"},
        {"size": 1024, "d": 512, "semantic_role": "argument_type_pattern"},
        {"size": 1024, "d": 512, "semantic_role": "filler_class"},
        {"size": 1024, "d": 512, "semantic_role": "residual"},
    ],
    update_rule="rotation_trick",               # Fifty et al. 2024
    commitment_beta=0.25,
)

predicate_argument_decoder = PADecoder(         # NEW; replaces surface decoder
    output_heads=[
        ("predicate_lemma_class", num_classes=500, loss="cross_entropy"),
        ("arg0_filler_class",     num_classes=64,  loss="cross_entropy"),
        ("arg1_filler_class",     num_classes=64,  loss="cross_entropy"),
        ("arg2_filler_class",     num_classes=64,  loss="cross_entropy"),
        ("modifier_set",          num_classes=32,  loss="bce_multilabel"),
    ],
    cross_attend_to="rvq_code_embeddings",
)
```

### 4.2 Training data

| Source | Role | Size |
|---|---|---|
| OntoNotes SRL gold (PropBank-style) | Predicate-argument supervision | ~50k sentences |
| AMR-3.0 | Richer typed graphs; convert to PA tuples | ~60k |
| UniversalPropositions (7 langs) | Cross-lingual SRL | ~70k total |
| **Silver Wikipedia SRL** | Bootstrap. Use finetuned-LLaMA-3-AMR (per arxiv 2508.05028, SMATCH 0.804) to label 5M Wikipedia sentences. **Confidence filter ≥ 0.85.** | ~3M usable |
| **ParaNMT-50M** (or SynCSE-generated, modern alternative) | Paraphrase pairs for contrastive loss | 5M pairs |
| **Europarl EN-FR / EN-DE / EN-ZH** parallel sentences | Cross-lingual structure pairs | 200k each |

### 4.3 Loss (binding form — every term has a job)

```
L_VQ = L_PA_recon
     + λ_para * L_paraphrase_contrastive
     + λ_xling * L_crosslingual_contrastive
     + λ_struct * L_structural_probe
     + β * L_commit
     + γ * L_codebook_entropy

L_PA_recon
    = Σ cross-entropy over typed-tuple slots
      (predicate_lemma_class, arg_filler_classes, modifier_set)
    -- This is the abstraction. Decoding predicate-argument tuples,
       NOT surface tokens, is what forces codes to be structural.

L_paraphrase_contrastive
    = InfoNCE over (anchor_codes, paraphrase_codes; negatives = batch + hard mined)
    -- Hard negatives: sentences with high lexical overlap but different
       predicate-argument structure (mined via inverse PA-tuple-hash).
    λ_para = 1.0   (co-equal with reconstruction)

L_crosslingual_contrastive
    = InfoNCE over (en_codes, parallel_lang_codes; negatives = batch)
    λ_xling = 0.5  (lower weight; only active when parallel data is in batch)

L_structural_probe
    = cross-entropy of (POS, deprel, entity_type) predicted from RVQ codes alone
    -- Gate 2 promoted from diagnostic to objective.
    λ_struct = 0.3

L_commit (rotation-trick standard form)
    β = 0.25

L_codebook_entropy
    = (H_target - H(codebook_usage))^2          # CORRECTED SIGN
    -- Drives entropy toward H_target (typically log(K) - 1).
    -- Previous "-H" form rewarded collapse; this form penalizes deviation.
    γ = 0.1
```

### 4.4 Codebook collapse — multiple independent regularizers

1. RVQ structure (4 levels): each level only needs ~512 of 1024 codes active to function. Redundancy across levels.
2. Rotation-trick VQ update (Fifty et al. 2024): subsumes EMA + dead-code-restart.
3. Entropy regularizer (corrected sign).
4. Hard-negative mining in contrastive batches: prevents code-cluster collapse onto frequent surface patterns.

### 4.5 Stage 2 gates (the make-or-break gates)

| Gate | Threshold | Action if fail |
|---|---|---|
| **G2.1 (paraphrase invariance)** | Paraphrase pair code-tuple overlap ≥ 3× random baseline on held-out paraphrase generator unseen at train time. | Tune λ_para upward; if still failing at λ_para=2.0, the bottleneck is the decoder — add stronger structural supervision. |
| **G2.2 (structural probe)** | Probe accuracy on **deprel from code-tuple alone** ≥ 80% (POS ≥ 90%). | Increase λ_struct; if still failing, the codes are not learning structure. Stop and re-design. |
| **G2.3 (codebook utilization)** | ≥ 80% codes used on each RVQ level at convergence. | Lower codebook size or increase γ. |
| **G2.4 (cross-lingual code overlap)** | EN-FR parallel sentence code overlap ≥ 2× random. | Cross-lingual claim is dead. Mark as English-only. |

### 4.6 Frontloaded Week-1 pilot

A 100k-sentence synthetic pilot **before** investing 4 weeks in Stage 1. Use:
- spaCy parses (good enough for pilot)
- PropBank gold for the decoder
- 50k ParaNMT pairs
- 5k Europarl EN-FR pairs

Run for 5 days on a single 4090. If G2.1 or G2.4 fail at this scale, the whole spec is dead — abort, save the remaining 9 weeks.

---

## 5. Memory Schema and Indexing

### 5.1 Entry schema

```json
{
  "entry_id": "wiki_en_0000001_root",
  "code_tuple": [412, 891, 23, 1204],
  "code_embedding": [...],                       // 512d, sum of RVQ level embeddings
  "type_signature": {
    "predicate_type_code": 412,
    "argument_type_pattern": [23, 89],
    "filler_class_codes": [1024, 877]
  },
  "predicate_argument": {                        // decoded form for inspection
    "predicate": "raise",
    "ARG0": {"text": "Bank of England", "type": "ORG"},
    "ARG1": {"text": "interest rates", "type": "MEASURE"},
    "modifiers": []
  },
  "surface": "The Bank of England raised interest rates.",
  "surface_variants": [                          // collected at dedup time
    "Interest rates were raised by the Bank of England."
  ],
  "provenance": {
    "doc_id": "wiki:monetary_policy",
    "section": "History",
    "sentence_index": 17,
    "parse_confidence": 0.94,
    "srl_confidence": 0.88,
    "indexed_at": "2026-05-18T..."
  }
}
```

**Storage**: Parquet on disk for entries; RocksDB key-value for inverted index; FAISS HNSW-PQ for dense.

### 5.2 Indices

**Dense index** (FAISS):
- HNSW(M=32, efConstruction=200) over PQ-compressed 512d → ~32 bytes/entry.
- 80M entries × 32 bytes ≈ 2.6 GB index. Manageable.

**Type-signature inverted index** (RocksDB):
- Key: `(predicate_type_code, sorted_arg_pattern_codes)` packed as 8 bytes.
- Value: list of `(entry_id, code_embedding_pq, surface_text_offset)`.
- Average bucket size: 80M / (1024 × 64²) buckets ≈ 19 entries / bucket. Mostly small; long-tail rare.

### 5.3 Dedup

Group by **typed-PA-tuple hash** (NOT code-hash; codes already encode this).
Hash key: `(predicate_lemma_class, sorted_ARG_filler_classes, modifier_set_hash)`.
Keep entry with highest `parse_confidence + srl_confidence`. Store other surfaces as variants.

**Expected dedup** on full Wikipedia: 8-15% (corrected estimate; original spec's 15-25% was optimistic — Wikipedia is paraphrase-sparse).

### 5.4 Retrieval at inference

```python
def retrieve(input_parse, llm_hidden, k=10):
    # Dense path
    q_dense = dense_projector(llm_hidden)        # learned projection, trained in Stage 4
    dense_hits = faiss.search(q_dense, k)

    # Structural path (uses INPUT parse, NOT mid-layer hidden state)
    # This fixes a documented bug in StructMem: hidden state doesn't expose parse cleanly.
    type_sig = build_type_signature(input_parse)
    struct_hits = rocksdb.query(type_sig, k)     # exact + 1-arg-substitution neighbors

    # Per-token gating (not single scalar)
    gate = gating_mlp(llm_hidden)                # 2 scalars, no softmax constraint
    return weighted_merge(dense_hits, struct_hits, gate)
```

Note: structural query uses the **input parse** (computed once per request), not a parse of the LLM hidden state. This eliminates StructMem's hand-waved `structural_probe(h)`.

### 5.5 Chunked retrieval at generation time

RETRO-style. Refresh memory context only at chunk boundaries (every K=64 generated tokens **or** at sentence-end punctuation). Within a chunk, the cached Perceiver-IO latents serve all tokens.

Latency budget per token:
- Within chunk: just LLM forward + Perceiver-IO over cached latents → ~5-8ms.
- At chunk boundary: + parse update (≈ 30ms on GPU) + retrieval (≈ 2ms) + Perceiver-IO refresh (≈ 2ms) → ~40ms one-off, amortized over K=64 tokens = 0.6ms/token.

Effective amortized cost: ≤ 10ms/token at k=10. **Within G_LATENCY budget.**

---

## 6. Stage 3 — Perceiver-IO Fusion + Frozen LLM

### 6.1 Fusion block

```python
fusion = PerceiverIOFusion(                      # repo: perceiver_attention
    num_latents=16,
    latent_dim=512,
    cross_attend_heads=8,
    process_layers=2,                            # latent self-attention depth
)
# Q latents (learned) attend over k=10 retrieved entries' code_embeddings.
# Output: 16 latent vectors injected as cross-attention into LLM at layer N/2.
```

Perceiver-IO advantages over StructMem's vanilla cross-attention:
- Cost constant in k (the proposal's k=10 is small; matters more when k scales).
- Bottleneck regularization built in.
- No `learned_alpha` scalar — the latent processor merges dense+structural results internally.

### 6.2 LLM backbone

| Choice | Rationale |
|---|---|
| Gemma-3-8B (in `models/gemma/`) | Repo-available; permissive licensing; mid-layer-hook friendly. |
| Qwen-3-8B (in `models/qwen/`) | Fallback if Gemma-3 hooks break. Strong multilingual base — helps G_CROSSLINGUAL. |

Frozen backbone, LoRA rank 8 on layers N/2 .. N. Empirically (per ROME and LoRA papers), late-half LoRA absorbs the residual from fusion without disturbing earlier representations.

### 6.3 Stage-3 training

| Field | Value |
|---|---|
| Objective | Standard next-token CE on held-out Wikipedia + held-out OpenWebText slice (NOT in memory). |
| Trainable | Fusion latents, fusion Q/K/V projections, LoRA adapters, dense_projector (Stage 5.4). |
| Data | 500M tokens held-out from memory corpus. |
| Optimizer | AdamW, lr 1e-4 fusion, 5e-5 LoRA, 1e-4 projector. |
| Batch | 16 sequences × 2k tokens. |
| Epochs | up to 3, early stop on held-out perplexity. |
| Warmup | 1000 steps. |

---

## 7. Training Schedule (8 weeks, frontloaded validation)

| Week | Activity | Gates checked |
|---|---|---|
| **1** | **Pilot.** 100k synthetic paraphrase pairs + Europarl. Train just composer + RVQ + PA decoder + contrastive losses. spaCy parses. PropBank gold for decoder. | **G2.1, G2.4 only.** If either fails, abort. |
| 2-3 | Stage 1: Tree Transformer + biaffine relation head + dual-head NER on UD + OntoNotes + WikiAnn. mDeBERTa-v3 init, finetuned. | G1.1, G1.2, G1.3 |
| 4 | Stage 2 (full): wire stage-1 outputs into composer + RVQ + PA decoder. Add `L_structural_probe`. | G2.1, G2.2, G2.3 |
| 5 | Stage 4 pilot: 5M Simple Wikipedia → memory index. Build RocksDB inverted + FAISS HNSW-PQ. Measure dedup rate, lookup p99. | G_LATENCY pilot |
| 6 | Stage 3 fusion: train Perceiver-IO + LoRA on Gemma-3-8B with pilot index. | G_PPL, G_FAITH, G_UPDATE |
| 7 | **Operational tests** for "program" framing: G_COMPOSE, G_SUBSTITUTE, G_INSPECT. **If any fail**, drop "program" framing publicly. | G_COMPOSE, G_SUBSTITUTE, G_INSPECT |
| 8 | Full Wikipedia indexing (80M sentences) + final eval on TriviaQA + DocRED + cross-lingual probe. | All gates re-checked. |

Total budget: 8 weeks (2 less than StructMem original). Achievable because Stage-1 supervised parser is reduced to a small relation-head over already-induced spans.

---

## 8. The Eight Gates (binding pass/fail)

| Gate | Metric | Threshold | Comparison |
|---|---|---|---|
| G2.1 | Paraphrase code-overlap (held-out generator) | ≥ 3× random | — |
| G2.2 | Deprel-from-code probe accuracy | ≥ 80% | — |
| G2.4 | EN-FR parallel code overlap | ≥ 2× random | — |
| G_PPL | Held-out perplexity (entity-dense Wikipedia) | Beat MEMORY-VQ at matched 512-byte/entry index size | **MEMORY-VQ**, not kNN-LM |
| G_FAITH | TriviaQA EM with retrieval / FActScore on Wikipedia bios | ≥ MEMORY-VQ + 2 points OR equal with smaller index | Faithfulness, not just PPL |
| G_UPDATE | Single-fact insertion: index 1 new sentence, query within 5 inference calls | ≥ 95% correct vs ROME (~70% at single-edit, degrades with #edits) | ROME / MEMIT |
| G_LATENCY | End-to-end token p50 / p99 | ≤ 25ms / ≤ 60ms at k=10 | — |
| G_COMPOSE | Multi-hop question accuracy where two retrieved entries each cover one hop | ≥ 60% vs ~40% for MEMORY-VQ at same k | Earns the "program" framing |
| G_SUBSTITUTE | Substituting one ARG slot in retrieved entry changes LLM output in expected direction | ≥ 70% directional agreement | Earns the framing |
| G_INSPECT | Human evaluator can correctly identify retrieved entry's intent from rendered form | ≥ 85% agreement on a 200-item evaluation | Earns the framing |

**Honesty rule.** If G_COMPOSE / G_SUBSTITUTE / G_INSPECT do not all pass, the paper drops the "soft program" framing and is presented as "Typed Structured Retrieval with Predicate-Argument Codes." The technical contribution (G_PPL, G_FAITH, G_UPDATE vs MEMORY-VQ / ROME) is still publishable.

---

## 9. The Operational "Program" Tests (Detail)

These three tests are what distinguishes SoftProg from typed retrieval. Each must be designed with care.

### 9.1 G_COMPOSE — multi-hop retrieval composition

**Setup**. Build a held-out QA set of 500 two-hop questions of the form:
> "What language is spoken in the country where the Eiffel Tower is located?"

Each question's gold answer requires combining facts from two distinct Wikipedia sentences (e.g. `located_in(Eiffel_Tower, Paris)` + `capital_of(Paris, France)` + `language_spoken_in(France, French)`).

**Metric**. Top-1 EM. Compose-vs-no-compose ablation: at retrieval, force k=2 entries (one per hop) versus k=10 (model picks). If the structured retrieval enables composition that flat retrieval cannot, EM goes up at k=2.

**Pass**: ≥ 60% EM with structured retrieval; ≥ 15 points absolute improvement over MEMORY-VQ at matched k.

### 9.2 G_SUBSTITUTE — counterfactual slot replacement

**Setup**. 200 sentences with clear PA structure. For each: identify the ARG1 slot, replace with a typed-compatible alternative (e.g. "The Bank of England raised **interest rates**" → "raised **inflation expectations**"). Re-encode and re-inject.

**Metric**. LLM's downstream completion changes in the expected direction (human-rated agreement). Compare against MEMORY-VQ (which retrieves surface text; substitution would require text editing — likely brittle).

**Pass**: ≥ 70% directional agreement.

### 9.3 G_INSPECT — human-readable rendering

**Setup**. Rendering function `render(entry) → string` produces forms like:
```
raise(ARG0: Bank_of_England[ORG], ARG1: interest_rates[MEASURE])
  modifiers: []
  surface_canonical: "The Bank of England raised interest rates."
```

**Metric**. 200 entries shown to 5 annotators (no context). Annotators state the entry's "intent" or "claim." Compared against the gold sentence's intent.

**Pass**: ≥ 85% agreement.

---

## 10. Known Risks and Mitigations

| # | Risk | Likely | Severity | Mitigation |
|---|---|---|---|---|
| R1 | Paraphrase-contrastive loss collapses codes onto a small set | low (multiple regularizers) | high | Hard-negative mining; entropy regularizer with target H = log(K)-1; codebook-utilization gate G2.3. |
| R2 | PA decoder cross-entropy plateaus at unusable levels (poor structural signal) | medium | high | Multi-source SRL: PropBank + AMR + UD + LLaMA-distilled silver. Confidence-filtered to 0.85. |
| R3 | Wang-induced spans miss key constituents → biaffine relation head has nothing to label | low | medium | Augmentation: at low Wang span confidence, fall back to a small dedicated dependency parser (DiaParser) for supervision. |
| R4 | Tokenization mismatch parser-side vs LLM-side breaks fusion | low (Perceiver projection handles basis change) | medium | Train fusion with parser tokens projected to LLM-token space first; measure cross-tokenizer alignment as a Stage-3 diagnostic. |
| R5 | MEMORY-VQ already beats SoftProg at matched size | medium | high | Build the MEMORY-VQ baseline ourselves at week 5 on the same corpus; if it beats SoftProg, the typed-retrieval claim is dead. |
| R6 | "Soft program" framing fails compose/substitute/inspect | medium | medium (only framing, not science) | Drop framing publicly. Re-name. Spec retains technical contribution. |
| R7 | Update gate (G_UPDATE) — newly indexed fact doesn't surface in retrieval | medium | high | Force-inject the new entry into the dense + inverted indices with a high-recall query path. Validate with the ROME benchmark. |
| R8 | Wikipedia parse failure rate at 0.85 conf threshold > 50% | medium | medium | Lower threshold to 0.75 with confidence as a feature; or fall back to LLaMA-AMR silver labels for low-conf parses. |
| R9 | Latency budget breached on a 4090 / 4070 setup | low (chunked) | medium | Chunked retrieval (K=64). If still breached, scale down to k=5 or precompute memory context per document. |
| R10 | Cross-lingual claim (G2.4) fails | medium | low (we can still ship English-only) | Reframe as "multilingual-parallel-trainable" rather than "language-agnostic memory." |
| R11 | Codebook size 1024×4 levels insufficient for 80M sentences | low | high | Scale Level-3 (filler-class) to 4096; coarsen Level-1 (predicate-type) to 512. RVQ structure absorbs the rescaling. |

---

## 11. Differentiation vs. Prior Art (the table that goes in the paper)

| Work | Year | Memory unit | Retrieval signal | Compression | Cross-lingual | Updatable | Inspectable | Composable |
|---|---|---|---|---|---|---|---|---|
| kNN-LM | 2020 | token-state vector | dense similarity | 1× | no | retrain | no | no |
| RETRO | 2021 | text chunks | dense + chunked attn | 1× | partial | retrain | partial | no |
| MEMORY-VQ (LUMEN-VQ) | 2024 | VQ-compressed token states | dense | 16× | no | retrain | no | no |
| ROME / MEMIT | 2023 | LLM FFN mid-layer | n/a (in-weight) | n/a | no | yes (degrades) | no | no |
| COCONUT | 2024 | continuous LLM-internal latent | n/a | n/a | no | retrain | no | partial (planning) |
| Latent-token VQ-VAE | 2025 | VQ-compressed reasoning step | n/a (in-context) | ~4× | no | retrain | no | partial |
| **SoftProg** | 2026 | **typed PA-tuple via RVQ** | **dense + type-signature** | **≥ 16× target** | **yes (gated G2.4)** | **yes (indexable)** | **yes (gated G_INSPECT)** | **yes (gated G_COMPOSE)** |

The contribution is the **combination** of (typed retrieval + structural decoder + paraphrase-contrastive objective + updatability via index write). Each individual element exists; the integrated system + the operational tests do not.

---

## 12. What This Spec Drops From the Original Proposal

- **Supervised biaffine dependency parsing as a Stage** → fused into the encoder via small relation head over Wang-induced spans.
- **Hand-waved `structural_probe(hidden_state)`** → eliminated. Structural query uses the input parse.
- **Single learned scalar α for index merge** → replaced with per-token gating MLP.
- **VQ-VAE codebook collapse mitigation matrix (EMA / Gumbel / STE / restart)** → collapsed to rotation-trick + entropy reg.
- **Llama-3-8B** → replaced with Gemma-3-8B or Qwen-3-8B (repo-available).
- **Reconstruction over surface tokens** → replaced by reconstruction over typed PA tuples.
- **"Paraphrase contrastive loss may be necessary" as Open Problem** → moved to the center as a Day-1 binding loss term.

## 13. What This Spec Adds vs. the Counter-Proposal Sketch in `2026_llm_tree_grammar.md` §8

- Concrete RVQ stack (4 levels × 1024) with explicit semantic-role assignment per level.
- Concrete training-data list including silver-SRL bootstrap from finetuned-LLaMA-AMR.
- Concrete dual-index implementation (FAISS HNSW-PQ + RocksDB inverted).
- Chunked retrieval at generation time (RETRO-borrowed); explicit latency budget.
- Eight named gates with thresholds.
- Three **operational tests** (G_COMPOSE / G_SUBSTITUTE / G_INSPECT) that make the "program" claim earn its name or get retired.
- Explicit MEMORY-VQ baseline as the principal comparison.
- ROME / MEMIT baseline for the update gate.
- Eight risks (R1-R11) with mitigations, none deferred to "Open Problems."

---

## 14. Open Problems (Honest, Bounded)

1. **PA decoder coverage on Wikipedia.** PropBank + UniversalPropositions cover ~50% of Wikipedia frame types. Silver-AMR labels close ~30% more. Remaining ~20%: low-frequency frames; train a `null_predicate` fallback that just preserves entity-type signature.

2. **Recursive composition.** G_COMPOSE tests two-hop. Three-hop and beyond require iterative retrieval (retrieve, fuse, retrieve again). Not in scope of this spec; deferred to a follow-up after G_COMPOSE passes.

3. **Memory staleness at the typed-signature level.** When the world changes ("CEO of X is now Y"), the old typed entry for the old CEO stays in the index. Need a soft-deletion + recency-prior mechanism. Sketch: per-entry `validity_score` decays unless refreshed; retrieval down-weights stale entries.

4. **Adversarial paraphrases.** ParaNMT / SynCSE-generated paraphrases over-represent NMT artifacts. Real-world paraphrase distributions (informal text, dialect, code-switching) may not be covered by G2.1. Mitigation deferred.

5. **Beyond Wikipedia.** Memory built on Wikipedia covers entity-relation knowledge but not procedural / opinion / dialogue content. Cross-domain memory mixing is a follow-up.

---

## 15. Repo Component Manifest

For implementers. All paths under `src/dl_techniques/`.

**Reused as-is**:
- `models/tree_transformer/{model.py, components.py}` — Wang encoder.
- `layers/vector_quantizer_rotation_trick.py` — Fifty-et-al VQ.
- `layers/embedding/hierarchical_codebook_embedding.py` — multi-level codebook.
- `layers/attention/perceiver_attention.py` — fusion block.
- `models/gemma/` or `models/qwen/` — backbone.
- `metrics/{perplexity_metric.py, llm_metrics.py, sequence_metrics.py}` — eval.
- `losses/dino_loss.py` (template for InfoNCE-style contrastive) and `losses/siglip_contrastive_loss.py` — reference patterns for contrastive impl.
- `losses/utilization_loss.py` — codebook entropy regularizer.

**New components to author**:
- `models/softprog/biaffine_relation_head.py` — biaffine over Wang spans.
- `models/softprog/dual_head_ner.py` — span-constrained + fallback NER.
- `models/softprog/label_aware_treegru.py` — arc-label-conditioned subtree composer.
- `models/softprog/predicate_argument_decoder.py` — PA-tuple decoder replacing surface VQ-VAE decoder.
- `models/softprog/rvq_stack.py` — hierarchical residual VQ wrapping rotation-trick VQ.
- `models/softprog/memory_index.py` — FAISS + RocksDB unified API.
- `models/softprog/perceiver_fusion.py` — wires perceiver_attention into LLM at layer N/2.
- `models/softprog/program_renderer.py` — human-readable rendering for G_INSPECT.
- `train/softprog/{stage1_encoder.py, stage2_vq.py, stage3_fusion.py, stage4_index.py}` — training scripts.
- `applications/softprog_explorer/` — TUI/Web for inspecting retrieved entries (supports G_INSPECT).

---

## 16. Hypothesis Summary at Spec-Finalization

(From the epistemic-deconstructor session that produced this spec; full session at `analyses/analysis_2026-05-18_fbaa6a5c/`.)

| Hypothesis | Posterior | Verdict |
|---|---:|---|
| H1: contrastive + PA decoder achieves ≥3× paraphrase code overlap | 0.66 | Likely. Spec frontloads this as G2.1, Week 1 pilot. |
| H2: type-signature index + dense beats MEMORY-VQ on entity-dense PPL | 0.43 | Uncertain. G_PPL must validate. Build MEMORY-VQ baseline ourselves. |
| H3: cross-lingual transfer ≥ 50% of monolingual gain | 0.36 | Uncertain. Cheap Week-1 test. |
| H4 (adversarial): "soft program" framing is rhetoric without operational content | 0.40 | **Live.** G_COMPOSE / G_SUBSTITUTE / G_INSPECT must all pass; otherwise framing dropped honestly. |
| H5 (adversarial): PA-decoder training data is the bottleneck | 0.25 | Mitigated by silver-AMR bootstrap + multi-source PropBank+AMR+UD. |
| H_S_prime: scope drivers exist outside initial S | 0.60 | Three folded in (tokenization, eval methodology, latency); spec absorbs them. |
| H_SCOPE_tokenization | 0.20 | Weakened by Perceiver projection design. |
| H_SCOPE_eval | 0.25 | Folded into G_FAITH + G_UPDATE gates. |
| H_SCOPE_latency | 0.21 | Folded into G_LATENCY gate; chunked retrieval mitigation. |

**Final disposition**: spec is buildable. The most live risk is H4 (framing) — addressed by the honesty rule in §8. The most consequential gate is G_PPL vs MEMORY-VQ; if SoftProg cannot beat MEMORY-VQ at matched index size on entity-dense text, the technical contribution shrinks to "interpretable retrieval at the cost of perplexity" — still publishable but a weaker story.

---

## Sources

- [MEMORY-VQ: Compression for Tractable Internet-Scale Memory (Zemlyanskiy et al., NAACL 2024)](https://arxiv.org/abs/2308.14903)
- [COCONUT — Training LLMs to Reason in a Continuous Latent Space (Hao et al., Meta, Dec 2024)](https://arxiv.org/abs/2412.06769)
- [RETRO: Improving LMs by Retrieving from Trillions of Tokens (Borgeaud et al., DeepMind 2021)](https://arxiv.org/abs/2112.04426)
- [Latent Token via VQ-VAE (Meta/Berkeley, Mar 2025)](https://www.marktechpost.com/2025/03/19/this-ai-paper-introduces-a-latent-token-approach-enhancing-llm-reasoning-efficiency-with-vq-vae-compression/)
- [Finetuned LLMs on AMR Parsing — SMATCH 0.804 (2025)](https://arxiv.org/html/2508.05028)
- [UniversalPropositions cross-lingual SRL](https://github.com/mynlp/UniversalPropositions)
- [Semantic Role Labeling: A Systematical Survey (2025)](https://arxiv.org/html/2502.08660v1)
- [Rotation-trick VQ (Fifty et al. 2024) — reference for rotation_trick update](https://arxiv.org/abs/2410.06424)
- [Residual Vector Quantization — SoundStream / EnCodec / RVQ-GAN](https://arxiv.org/abs/2306.06546)
- [ROME — Rank-One Model Editing (Meng et al.)](https://arxiv.org/abs/2202.05262)
- [MEMIT — Mass-Editing Memory in a Transformer (Meng et al. 2023)](https://memit.baulab.info/)
- [AnyEdit (2025)](https://arxiv.org/html/2502.05628v1)
- [Paraphrase-based Contrastive Learning (NAACL 2025)](https://aclanthology.org/2025.naacl-srw.39/)
- [Wang et al. — Tree Transformer (unsupervised constituency in self-attention)](https://arxiv.org/abs/1909.06639)
- Parent adversarial review: `research/2026_llm_tree_grammar.md` (in this repo).
- Epistemic session: `analyses/analysis_2026-05-18_fbaa6a5c/`.

---

*End of specification.*
