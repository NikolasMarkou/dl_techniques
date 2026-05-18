# StructMem: A Deep Adversarial Review

**Reviewer**: Claude (Opus 4.7, 1M context)
**Date**: 2026-05-18
**Scope**: The "StructMem" proposal (tree transformer + VQ-VAE abstract tokenizer + structured Wikipedia memory + LLM cross-attention fusion).
**Posture**: Adversarial. Find issues, gaps, bugs, opportunities. Cross-check against `dl_techniques` building blocks. Cite recent prior art. Push back. Then sketch a new paradigm.

---

## TL;DR

The proposal is internally coherent and the staged training schedule is professional. But three load-bearing claims do not survive scrutiny:

1. **The novelty claim is false.** §12 asserts "the combination of end-to-end learned abstract language, tree-grounded compression, and structured memory retrieval for LLM augmentation is not present in existing literature." This is wrong. **MEMORY-VQ (NAACL 2024)** already compresses retrieval memory with a VQ-VAE at 16x with KILT-parity. **Meta's COCONUT (Dec 2024)** and the Meta/Berkeley "Latent Token" line (Mar 2025) already use VQ-VAE compression in the LLM reasoning loop. The proposal needs to position against these, not pretend they don't exist.
2. **The bottleneck objective is wrong by the author's own admission.** §13 concedes that reconstruction loss drives the tokenizer toward surface preservation, not abstract structure, and that Gate 2 success "emerges from the inductive bias" rather than from an explicit objective. That is a polite way of saying: the proposal hopes the right thing happens. Hope is not an objective.
3. **The repo's `tree_transformer` is not what the proposal calls a "tree transformer".** The existing `src/dl_techniques/models/tree_transformer/` implements Wang et al.'s *unsupervised soft constituency* via Group Attention. The proposal specifies *supervised biaffine dependency parsing* (Dozat & Manning). These are different architectures with different training data, different outputs, and different downstream consumers. Reusing the repo name without porting a biaffine head is a recipe for silent failure.

The bones are good. The flesh on the bones is partly imaginary. Below: a stage-by-stage adversarial review, then a counter-proposal.

---

## 1. What Already Exists in `dl_techniques` (and What Doesn't)

A reality check before architecting on top of this codebase.

| Component the proposal needs | Status in repo | Notes |
|---|---|---|
| Tree Transformer encoder | **Mismatch.** `models/tree_transformer/` is Wang et al. *unsupervised soft constituency* via Group Attention, not Dozat biaffine dependency. | Reuse the name and you will mislead yourself. They are different artifacts. |
| Biaffine dependency parser head | **Absent.** No `BiaffineParser`, no UD loader, no MST decoder (Chu-Liu-Edmonds). | Must be built. |
| Span-based NER head | **Absent.** `layers/nlp_heads/` has factory/task types but no span-NER. | Must be built. |
| VQ-VAE | **Present.** `layers/vector_quantizer.py`, `layers/vector_quantizer_rotation_trick.py` (Fifty et al. rotation trick — better at codebook collapse than vanilla VQ). | Use the rotation-trick variant. The proposal's "EMA + dead-code restart" is yesterday's mitigation; rotation trick subsumes both. |
| Hierarchical codebook | **Partial.** `layers/embedding/hierarchical_codebook_embedding.py`. | Reusable for multi-resolution abstract codes. |
| Tree-structured composition (TreeLSTM / TreeGRU) | **Absent.** | The proposal's `BottomUpComposer(cell="LSTM"/"TreeGRU")` does not exist in the repo. Must be built. |
| Cross-attention fusion at LLM mid-layer | **Present and overlapping options.** `layers/attention/multi_head_cross_attention.py`, `shared_weights_cross_attention.py`, `perceiver_attention.py`. | The proposal picks vanilla cross-attention. Perceiver-style latent-bottleneck cross-attention is a strictly better fit for "10 retrieved entries → fused context vector" — it caps cost and gives you a free bottleneck regularizer. |
| Memory / NTM-style external memory | **Present.** `layers/memory/` (NTM, MANN, SOM). | Not currently wired for retrieval, but the addressing patterns transfer. |
| FAISS plumbing / HNSW indices | **Absent.** | External dep. Plan for it. |
| LLM backbone with mid-layer hook | **Present.** `models/gemma/`, `models/gpt2/`, `models/qwen/`. | The proposal names Llama-3-8B but there is no Llama in the repo. Use Gemma 3 or Qwen 3 instead. |
| Byte Latent Transformer (BLT) | **Present.** `models/byte_latent_transformer/`. | This is the most underappreciated building block here — see §10. |
| Codebook utilization / perplexity metrics | **Present.** `metrics/perplexity_metric.py`, `metrics/llm_metrics.py`. | Codebook utilization metric is not standardized; add one. |

**Implication.** The proposal's architecture is *partly* available off the shelf in this repo, but the load-bearing novel components — biaffine parser, tree-constrained span NER, hierarchical VQ over parse subtrees, structural inverted index — are 100% greenfield. The "Weeks 1-2" budget for Stage 1 assumes mature parser infrastructure. It is not there.

---

## 2. Prior Art the Proposal Misses or Underplays

§12 is the weakest section of the proposal. Critical omissions:

### 2.1 MEMORY-VQ (Zemlyanskiy et al., NAACL 2024, arXiv 2308.14903)

The closest prior work, missing entirely from §12. MEMORY-VQ:
- Applies a VQ-VAE to retrieval-augmented memory (LUMEN backbone).
- Achieves **16x compression with parity on KILT**.
- Replaces token-level memory vectors with integer codes + decompression.

The proposal's E5.3 hypothesis ("4x smaller index at kNN-LM parity") is a *less ambitious* version of an already-published result. If the only novelty is "now do it with parse-aware codes instead of token codes," that is a delta paper, not a "new paradigm." The proposal needs to either:
- Show **>16x** compression at parity (the structural prior must pay off relative to a non-structural baseline), or
- Show qualitative advantages MEMORY-VQ cannot reach (cross-lingual retrieval, paraphrase invariance, structural query language).

### 2.2 COCONUT (Hao et al., Meta, Dec 2024, arXiv 2412.06769)

Continuous latent reasoning inside the LLM, alternating language and latent mode. This is the dominant paradigm for "compress LLM-internal computation." The proposal is the *external memory* dual of COCONUT. Position explicitly.

### 2.3 Meta/Berkeley Latent Token via VQ-VAE (Mar 2025)

VQ-VAE compression of reasoning traces — replaces early reasoning steps with discrete latent tokens, retains later steps as text, randomly mixes both. This is the exact codebook regime the proposal targets (4096 codes, ~4-8 codes per chunk), just on reasoning rather than memory. Cite. Differentiate. Or steal the random-mixing curriculum — it is a stronger learning signal than reconstruction alone.

### 2.4 RETRO degradation past k=40

The Borgeaud et al. (DeepMind) RETRO paper shows perplexity *degrades* past 40 retrieved neighbors. The proposal picks k=10, which is fine, but the structural-retrieval branch will return noisier neighbors (a "(ORG nsubj VERB)" pattern matches thousands of Wikipedia sentences, most irrelevant). Without explicit re-ranking the structural index will dilute the dense index, not complement it.

### 2.5 Decoder-only LLMs already do AMR at SMATCH 0.804

The proposal positions itself against hand-engineered AMR. But finetuned decoder-only LLMs (Phi, Gemma, LLaMA, DeepSeek-R1) now hit competitive AMR parsing without bottlenecks. If the LLM can already extract structured meaning, why does the retrieval index need to do it too? The proposal needs a sharper answer to: **what does an abstract codebook give us that the LLM cannot do internally with a few latent tokens?** The answer is probably "inspection, update, and cross-corpus invariance" — but say so.

---

## 3. Stage-by-Stage Critique

### Stage 1 — Grammar (the LAS target is tighter than advertised)

**LAS >90 on UD English with frozen BERT is not "easy."** Current SOTA on PTB/EWT is in the LAS 95-96 range using *finetuned* XLM-R/DeBERTa-v3 with biaffine heads. Freezing BERT trades 4-5 LAS points for stability. The hyperparameter table claims this halves data requirements (true, plausibly) — but the LAS target is then tight, not generous. Plan to **finetune the top 4 BERT layers** at minimum, or move to DeBERTa-v3 / mDeBERTa-v3, or accept LAS ~88-89.

**E1.1 (tree attention > flat) is a weak experimental design.** Tree attention with arc-constrained masks needs the *predicted* parse to apply the mask. At init the predicted parse is garbage. Either (a) train flat first then add tree attention as a second pass, or (b) use gold arcs during training and predicted arcs only at inference (oracle/non-oracle ablation). Otherwise E1.1 measures a chicken-and-egg failure, not tree-attention's actual benefit.

**Multilingual UD 2.13 is uneven.** 250+ treebanks, but ~50 dominate; transfer to low-resource languages depends on byte-/character-level robustness more than on architecture. The "English LAS +0.5 from multilingual training" hypothesis (E1.4) is plausibly true *at this scale*, but for English specifically, a monolingual model with finetuned encoder will beat it.

**Missing**: parse confidence calibration. §13 introduces a 0.85 confidence filter for the memory pipeline. Biaffine parsers' softmax probabilities are notoriously *over-confident*. You need temperature scaling or MC-dropout to make 0.85 mean something. Add to Stage 1 evaluation.

### Stage 2 — NER (the parse-subtree constraint has a hidden recall ceiling)

**Span candidates constrained to parse subtrees imposes a hard recall ceiling at LAS.** If LAS = 90, ~10% of arcs are wrong, and any entity whose head is in a misattached subtree is *unreachable*. Span-based NER F1 will plateau at the geometric mean of independent span recall and parse correctness on the span. On long entities in long sentences this gets ugly fast.

**Fix**: dual-head — (i) tree-constrained primary, (ii) unconstrained fallback re-ranker over the top-K spans from the constrained head plus K spans by token-window heuristics. Standard NER (FLERT-style) hits F1 ~93-94 on CoNLL; the proposal targets >88 which is below modern baselines unless you assume the constraint is a feature.

**E2.4 (parse-NER correlation diagnostic) is good but the threshold is arbitrary.** r > 0.5 across sentence bins is a low bar. r should be *increasing in sentence length* — short sentences have high LAS and high NER regardless. Use partial correlation controlling for length, or stratify.

**WikiNER silver labels.** WikiNER's silver labels are noisy (Wikipedia-link based). Training on silver and reporting "recall on Wikipedia held-out" is partly a recall-of-noise measurement. Hold out a *manually-curated* Wikipedia NER slice (or use the WikiAnn gold annotation slice) for the E2.3 evaluation.

### Stage 3 — Abstract Tokenizer (the core proposal, the weakest stage)

This is where the proposal needs the most work. The author has already noticed (§13) that the objective is misaligned, but the proposed contrastive loss is bolted on as an "Open Problem" rather than placed at the center.

**Bug 1: Reconstruction loss optimizes for surface form.** Cross-entropy over decoded surface tokens means: codes that preserve word-order and lemmas win. Paraphrase-invariance is anti-correlated with reconstruction quality. The two Bank of England sentences in §2 will NOT produce the same code sequence under L_reconstruct unless the decoder is also surface-form-blind (e.g. predicts predicate-argument tuples). The proposal contradicts itself between §2 and §6.

**Bug 2: The compression target is double-counted.** "32 tokens → 4-8 abstract tokens" sounds like 4-8x. But modern subword tokenizers already give you ~1.3 tokens per word, and a 32-token sentence is ~24 words ≈ 6 phrases. 4-8 abstract tokens *is* one-per-phrase. That is not compression of *structure*; that is just constituent count. The real compression target should be: **codebook entropy per sentence**, not abstract-token count. A 4096-codebook with 6 codes/sentence gives ~72 bits/sentence; raw text is ~120 bits at typical entropy. So actual compression is ~1.7x at best.

**Bug 3: γ * L_diversity has the wrong sign in the equation.** The equation reads `L_abstract = L_reconstruct + 0.25 * L_commit - 0.1 * H(codebook)`. Subtracting H *encourages* low entropy, i.e. *causes* codebook collapse. The text says "penalize codebook collapse." Sign error — should be `+ γ * (-H)` if γ > 0 means penalty for low H, but then the variable name `L_diversity = -H` is misleading. Cleaner: `L = L_recon + β L_commit - γ H` with γ > 0 (reward high H). The current writeup is contradictory.

**Bug 4: EMA + Gumbel + straight-through is not a 3-way ablation.** They are different parts of the pipeline. EMA is a *codebook update rule* (vs. backprop). Gumbel/STE are *gradient estimators*. You can stack EMA codebook updates with STE encoder gradients. E3.3 conflates two orthogonal axes. Re-decompose.

**Better objective (use day 1, not as "Open Problem"):**
```
L = L_predicate_argument_recon          # decode (subj, verb, obj, modifiers)
  + λ_para * L_paraphrase_contrastive   # codes invariant across paraphrase pairs
  + λ_struct * L_structural_supervise   # codes predict deprel/POS (Gate 2 as objective, not test)
  + β L_commit  - γ H(codebook)
```
- `L_predicate_argument_recon`: don't reconstruct surface tokens. Reconstruct a canonical predicate-argument structure (head lemma + dependents' lemmas + relations). This *is* the abstraction.
- `L_paraphrase_contrastive`: ParaNMT-50M, MS-COCO captions, NLI-derived paraphrase pairs. Force same codes.
- `L_structural_supervise`: lightweight probing head trained jointly. If you want POS/deprel predictable from codes, supervise it. Don't hope it emerges.

**Bug 5: Codebook of 4096 across 80M sentences.** Mean 100k sentences per code if uniform. Either (a) the codebook captures only shallow patterns ("noun phrase head," "PP attachment"), in which case it is just a clustering of dependency labels and doesn't merit a learned VQ, or (b) the codebook is highly non-uniform with a long tail of dead codes, in which case rotation-trick or EMA do not fully save you. Plan for **hierarchical/residual VQ**: coarse 256-code top level + fine 4096-code residual. This is standard for audio (SoundStream, EnCodec) and worth porting.

**Bug 6: Subtree pooling with LSTM cell ignores arc labels.** The composition function uses children's hidden states but the proposal does not say how the dependency *label* (nsubj vs obj vs amod) enters the composition. If labels are not in the composition, the abstract codes cannot encode "X did Y to Z" vs "X was done Y by Z" differently. Either concatenate the arc-label embedding to each child input, or gate the composition by label (label-aware tree attention).

### Stage 4 — Wikipedia Memory (engineering risk underrated)

**Deduplication rate estimate of 15-25% is optimistic.** Wikipedia sentences are paraphrase-sparse *within* the corpus (each article is independently authored). Structural-hash collisions will dominate on (a) lead-sentence templates ("X is a Y born in Z"), (b) infobox-derived sentences. Expect <10% dedup on full-prose Wikipedia. If your sales pitch is "4x smaller memory than kNN-LM," you cannot afford to lose half of that to dedup miss.

**80M sentences in 24h pilot budget.** Stage 4 Week 6 says "build pilot index <24h" on 5M sentences. But the *full* table (§7 phased rollout) targets 80M sentences for Phase 2. With parsing + NER + VQ encoding at typical GPU rates (~1k sent/sec on a 4090, less with biaffine), 80M sentences = ~22 hours single-GPU just for forward passes. Add I/O, FAISS HNSW build (~10h for 80M vectors), and the 24h figure is for the *pilot*, not the full run. The Week 9 line item ("English Wikipedia indexing") gets one week. Optimistic. Plan for 3-5 days minimum on a single 4090, less on the 4070.

**Bug: structural_hash double-counts.** `hash(abstract_codes + entity_types + dep_relations)` includes both abstract codes (which already encode structure) and dep_relations (which are part of what the codes are supposed to encode). The hash collapses two sentences only when both their codes *and* their explicit deprels match. If codes are paraphrase-invariant but deprels differ between active/passive, the hash will not collapse them — defeating the point. Either drop deprels from the hash (trust the codes) or define a canonicalization (active-voice normalization) before hashing.

**Memory schema is heavy.** ~800 bytes per entry × 80M = ~64 GB just for JSON. Plus 512-d float32 embedding = 160 GB. Plus FAISS HNSW overhead. Plan for ~300 GB on disk for full Wikipedia. Practical for one workstation; not free.

### Stage 5 — LLM Fusion (the structural probe is hand-waving)

**Bug: "q_struct = parse(h)" is undefined and likely impossible.** The retrieval pseudocode calls `structural_probe(hidden_state)` to extract (entity_type, dep_relation, head_pos) from the LLM's mid-layer hidden state. The LLM is frozen. Its hidden state at layer N/2 has *not* been trained to expose dependency relations. There are three options, none stated:
1. Train the probe on labeled data (UD + NER) to extract structural features from h. Then the LLM hidden state must already encode parse info linearly probeable. This is *partially true* (Hewitt & Manning structural probe) but UAS ~85 at best from probing, far below the parser-grade signal needed.
2. Send the input through the *parser* in parallel and use *its* features for the structural query — bypassing the LLM hidden state entirely.
3. Train the LLM to expose structural features (LoRA or unfreezing) — kills the "frozen LLM" premise.

Pick one. The current writeup implies (1) but ignores its severe accuracy ceiling.

**Bug: injection at layer N/2 plus residual `h' = h + context`.** The downstream layers (N/2..N) were never trained to consume a memory-residual signal. With LoRA off, the model treats `context` as noise. The "1-point perplexity improvement" target requires LoRA to work; without it the gain is likely <0.3 nats. The frozen-LLM scenario will fail Gate 3.

**E5.3 (parity at 4x smaller index vs kNN-LM) is the right benchmark but the wrong baseline.** kNN-LM is from 2020. The right baseline is MEMORY-VQ at 16x compression. If you can't beat MEMORY-VQ at its own game, the structural angle adds engineering without scientific payoff.

**E5.4 (learned α) is a single scalar.** A single global α cannot adapt per query. Either make α a per-token learned function of h (gating network) or use product-of-experts (dense and structural always contribute, no merge weight needed).

### Stage 6 (missing) — Inference cost

The proposal never discusses inference latency. With k=10 retrieved entries, two index lookups (dense FAISS HNSW + structural inverted) per inference step, plus a cross-attention pass over 10 entries' embeddings, you add ~5-15ms per token on a 4090. For a 100-token generation: 0.5-1.5s overhead vs a no-retrieval Gemma-3-8B. Acceptable. But say so. And design the structural index for sublinear lookup (the proposed "(ORG, raised, rates)" key has high cardinality on Wikipedia — millions of (entity_type, lemma, dep_role) triples; need an LSM-tree or RocksDB layer, not a Python dict).

---

## 4. Methodological / Procedural Concerns

**Gate ordering inverts the risk.** Gate 1 (tree attention helps long sentences) is the *easiest* gate to pass and the *least informative*. Gate 2 (paraphrase code overlap) is the *hardest* to pass and the *most informative*. The schedule front-loads weeks 1-2 on Gate 1 and reaches Gate 2 only in week 5. If Gate 2 fails, weeks 1-4 of parser/NER training are not wasted (they're useful artifacts), but the *thesis* of the paper is dead. Frontload Gate 2 with a synthetic-data pilot in Week 1.

**No baseline for the "structured memory" claim before Stage 5.** The proposal compares to kNN-LM in E5.3 *after* building a full Wikipedia index. The cheaper comparison is: take a 1M-sentence slice, build (a) MEMORY-VQ on raw token states and (b) StructMem on parse codes, both at matched index size. Compare perplexity *before* committing to the full pipeline.

**The "Wikipedia is clean" assumption.** Wikipedia has: lists, math, code blocks, IPA, foreign-language quotations, citation markers, infobox residue (even after wikiextractor). Parse confidence on these will be miserable. The 0.85 threshold may filter 30-50% of "sentences." The effective corpus size is far smaller than 80M.

**Cross-lingual hypothesis is untested before commitment.** "Train on multilingual UD so codes are language-invariant" is asserted, never validated. Add an experiment: train tokenizer on English+French parses of the same sentence (Europarl), test code overlap. If overlap is <2x random, the cross-lingual claim is dead and English-only training is strictly better.

**No mention of model size.** The tree transformer is 6 layers / d=512 / heads=8. That is ~40M params. With biaffine head, plus VQ-VAE encoder/decoder (each ~30M), the *front-end* is ~100M params. Trainable on a 4090, yes, but the proposal does not place this in the parameter budget vs LLM (~8B). The front-end is 1.25% of the backbone. Reasonable.

---

## 5. Specific Bugs in the Pseudocode

1. **Diversity loss sign** (Stage 3): `- γ * H(codebook)` rewards collapse. Fix to `+ γ * (entropy_target - H)^2` or similar.
2. **Span representation** (Stage 2): `width_emb(min(end - start, 30))` — cap at 30 leaves entity spans like long org names ("Federal Reserve Bank of New York") clipped at width 30. Use log-width buckets instead.
3. **`mask_type = "arc_constrained"`** (Stage 1): not defined. Assumed = "attend only along predicted arcs." But at training step 0, arcs are random. Specify curriculum: warm-up with full attention for first 1000 steps, anneal mask in.
4. **VectorQuantizer EMA + commitment loss**: with EMA codebook update, commitment loss only affects the *encoder*, not the codebook. State this. Otherwise readers expect β to also update codebook.
5. **Memory schema `abstract_embedding`**: "mean-pooled over abstract codes" — but the dense FAISS index queries against LLM hidden states (different space). Where is the projection trained? Add a contrastive alignment objective in Stage 5 (CLIP-style) or this index returns garbage.
6. **`learned_alpha` in retrieval**: a scalar in [0,1] cannot encode "which index do I trust." Use a gating MLP with two scalars (no constraint to sum to 1), or normalize at the end.

---

## 6. Opportunities the Proposal Misses

### 6.1 Use the existing Wang Tree Transformer for free unsupervised structure

`models/tree_transformer/` already learns soft constituency unsupervised. Stage 1 can be **skipped** for tasks that don't need explicit UD relation labels: use Group Attention's induced spans as subtree boundaries directly. Saves 2 weeks. Risk: NER (Stage 2) wants typed arcs (nsubj for entity-as-subject filtering); soft constituency only gives you spans. Compromise: dual-track — Wang Tree Transformer for span induction + a small biaffine *relation classifier* head over predicted spans (lighter than full parsing).

### 6.2 Byte Latent Transformer (BLT) is the right Stage 3 architecture

`models/byte_latent_transformer/`. BLT is *exactly* a learned discrete tokenizer with dynamic compression rates. Replace VQ-VAE with BLT's entropy-driven patching: high-entropy regions get more codes, low-entropy regions get fewer. The "compression ratio" stops being a hyperparameter and becomes input-adaptive. This is a much stronger paradigm than "fixed 4-8x compression."

### 6.3 Perceiver-style fusion beats vanilla cross-attention

`layers/attention/perceiver_attention.py` already exists. Replace the Stage 5 cross-attention with a Perceiver-IO block: learned latent queries attend over k=10 retrieved entries. Pros: constant cost regardless of k, built-in bottleneck regularization, no scalar α needed. The proposal's vanilla cross-attention is a 2017 design when 2024 designs are available.

### 6.4 Hierarchical / Residual VQ via `hierarchical_codebook_embedding.py`

Already in the repo. Use it. Two-level codebook with coarse (256) over phrase types + fine (4096) over predicate-argument fillers gives semantic factoring that flat VQ cannot.

### 6.5 The vector quantizer rotation trick

`vector_quantizer_rotation_trick.py` (Fifty et al. 2024) makes most of the proposed "codebook collapse prevention" tricks redundant. Use it from day 1; drop EMA / dead-code-restart from the experimental matrix.

### 6.6 Memory bank + NTM as the structural index backbone

`models/memory_bank/`, `layers/memory/ntm_interface.py`. The proposal builds a custom dual-index (FAISS + inverted Python dict). NTM gives differentiable read/write addressing over a typed memory, which lets you train the *index policy* jointly with the LLM fusion layer. Likely better Gate 3 result. Trade-off: NTM at 80M entries is impractical — keep FAISS for the dense layer and use NTM-style addressing only over a *summary index* (e.g., codebook-N-gram embeddings, ~10k entries).

### 6.7 SOM as a structural map

`layers/memory/som_nd_soft_layer.py`. A self-organizing map over the abstract code space gives you a topologically-organized memory: nearby map cells = nearby structures. Query by location, not by hash. Inspectable. Updatable. Interesting research direction for §13's "memory staleness" problem.

---

## 7. The Cognitive Trap: Bottleneck Romanticism

The proposal exhibits a recognizable failure pattern: **the belief that adding a bottleneck imposes the right abstraction**. It does not. A bottleneck only forces you to be lossy. *What* you lose is determined by the objective, and the objective here is reconstruction, which preserves the wrong thing.

Variants of this trap in the deep learning literature: vanilla autoencoders learning to copy via skip-connection-like shortcuts; InfoBottleneck papers finding that the bottleneck doesn't actually align with the task; VQ-VAEs collapsing onto unigram statistics. The mitigation is always the same: **specify the abstraction in the objective.** The proposal's Open Problem #1 says this implicitly. Move it to the center.

---

## 8. A Counter-Proposal: "Soft-Programmatic Memory"

Same problem statement, different solution. Drop the "learn an abstract language and hope it generalizes" framing. Replace with:

> Memory entries are **soft programs** in a typed predicate-argument calculus. The LLM doesn't retrieve them — it **executes** them as a thought-step.

### 8.1 Core change of frame

| Proposal (StructMem) | Counter-proposal (SoftProg) |
|---|---|
| Abstract codes are *compressed representations* of sentences | Abstract codes are *predicate-argument programs* (a small, learned typed lambda calculus) |
| Retrieved via similarity over hidden states | Retrieved by *unification* (Prolog-style) on the LLM's working-memory program state |
| Goal: lower perplexity by adding context | Goal: lower perplexity by adding *reusable computation* |
| Bottleneck via reconstruction | Bottleneck via *program-equivalence* (paraphrases = same program) |

### 8.2 What this gives you

1. **Paraphrase invariance becomes the loss, not an emergent property.** Two paraphrases are two surface realizations of the same program. Train with paraphrase-pair contrastive loss from step 1.
2. **Inspection is free.** A retrieved entry is a typed expression like `caused(raise_rates(Bank_of_England), inflation_target)`. No more "what does code 412 mean?"
3. **Cross-lingual transfer is structural.** Programs are language-agnostic. The English-French overlap experiment becomes the centerpiece, not a footnote.
4. **The structural index becomes a real index.** Inverted index keyed by program *type signature*, not hash. Sub-millisecond lookups, sublinear scaling, easy updates.

### 8.3 Architecture (uses repo components)

```
Text
 │
 ▼
[Wang Tree Transformer]                 # models/tree_transformer (unsupervised soft constituency)
 │ + small biaffine relation head        # ~10M params, predicts deprel given spans
 ▼
[Span-typed embeddings]
 │
 ▼
[Hierarchical-Residual VQ]              # layers/vector_quantizer_rotation_trick + 
 │ coarse: predicate types               #   hierarchical_codebook_embedding
 │ fine:   argument fillers
 ▼
[Program assembly]                       # NTM-style differentiable: 
 │ - head predicate slot                 #   models/memory_bank + layers/memory/
 │ - typed argument slots                #
 ▼
[Memory index]                           # FAISS dense + RocksDB inverted on type-sigs
 │
 ▼
[Perceiver fusion @ LLM mid-layer]       # layers/attention/perceiver_attention
 │   (Gemma-3 / Qwen-3 backbone)         # models/gemma or models/qwen
 ▼
Next token
```

### 8.4 Training schedule (8 weeks, not 10)

```
Wk 1:    Synthetic-data pilot. 100k toy paraphrase pairs (back-translation EN-FR-EN). 
         Train Hierarchical-Residual VQ with paraphrase-contrastive loss.
         GATE: Code overlap on paraphrases > 3x random. (This is the proposal's Gate 2 -- 
         done first because it is the most informative.)

Wk 2-3:  Wang Tree Transformer + biaffine relation head, joint training on UD English + 
         OntoNotes. NO frozen BERT -- use mDeBERTa-v3-base, finetune all.
         GATE: LAS >= 92 on UD English dev.

Wk 4:    Span-NER + Stage 3 finalization, with parse output now feeding VQ.

Wk 5:    Memory population pilot on Simple Wikipedia (5M sentences).
         GATE: dedup >= 15%, structural index lookup p99 <= 5ms.

Wk 6-7:  Perceiver fusion with frozen Gemma-3-8B + rank-16 LoRA on N/2..N layers.
         GATE: perplexity beats MEMORY-VQ baseline at matched index size on 
         entity-dense Wikipedia held-out.

Wk 8:    Full Wikipedia indexing if Wk 6-7 passes. Final evaluation.
         GATE: Cross-lingual retrieval -- French query, English index, perplexity 
         improvement >= 50% of monolingual gain.
```

### 8.5 What this gives up

- No supervised dependency parser (Wang Tree Transformer instead) → 3-5 LAS points on hard sentences.
- Smaller initial corpus (paraphrase pairs ~50M tokens, not full UD).
- Higher complexity in Stage 3 (predicate-argument decoder is harder than surface reconstruction).

### 8.6 What it doesn't give up

The thesis of the original proposal: *externalize structural language understanding into a retrievable, inspectable, updatable memory*. That stays. The pipeline gets simpler and the gates get sharper.

---

## 9. Falsification Tests (Sharpened)

The proposal's Gates 1-3 are reasonable but easy. Sharper versions:

| Gate | Original | Sharper |
|---|---|---|
| **G1** | Tree attention > flat by 2 LAS on long sentences | Tree attention > flat by 2 LAS *with matched parameter budget* and *matched compute*. The current G1 lets you spend more compute and call it victory. |
| **G2** | Probing >70% + paraphrase overlap > 2x random | Paraphrase overlap >= 3x random on **held-out paraphrase pair sources** (i.e., the back-translation system used at test time was never seen at train time). Plus: probing accuracy of *deprel from code alone* >= 80% (deprel is the structural signal that matters; POS is weaker). |
| **G3** | Beat no-retrieval by >1 perplexity point + match kNN-LM at <=50% size | Beat **MEMORY-VQ** at matched index size on **entity-dense** held-out. The kNN-LM baseline is 6 years old. |
| **G4 (new)** | — | Cross-lingual retrieval: French input through same parser, query English index, get >= 50% of monolingual perplexity gain. Tests the cross-lingual claim explicitly. |
| **G5 (new)** | — | **Update test**: insert a new fact into memory via a single sentence parse-and-index call, verify the LLM uses it on a downstream question within one inference call. Tests the "updatable" claim. |

If G5 fails, the proposal does not deliver an *updatable* knowledge store — it delivers a *static* retrieval-augmented LM. Most of the motivation in §1 evaporates.

---

## 10. Recommendations (Prioritized)

1. **Fix the §6 objective.** Day-1 contrastive loss over paraphrases. Decode predicate-argument tuples, not surface tokens. This is non-negotiable; the current objective is wrong.
2. **Frontload Gate 2** (Week 1, synthetic-data pilot). Cheapest informative experiment in the schedule. Do not invest 4 weeks in parser/NER before validating the bottleneck.
3. **Replace `EMA + dead-code restart` with rotation-trick VQ** (already in repo). Remove E3.3 conflations.
4. **Position against MEMORY-VQ explicitly** in §12. Either show >16x compression or qualitative advantages.
5. **Replace cross-attention fusion with Perceiver-IO** (already in repo).
6. **Add a dual-head NER fallback** to break the parse-subtree recall ceiling.
7. **Fix the structural_hash double-counting** — hash either codes or deprels, not both.
8. **Define `structural_probe(h)` rigorously** or replace with parallel parse of input (bypass LLM hidden state for structural query).
9. **Plan for ~300 GB on disk** and ~3-5 days indexing for full Wikipedia. Update Week 9.
10. **Add the cross-lingual gate (G4) and update gate (G5).** These test the actual motivation.
11. **Use Gemma-3-8B or Qwen-3-8B, not Llama-3-8B.** The repo has the former two.
12. **Consider BLT-style entropy-driven patching** as an alternative to fixed-rate VQ for Stage 3.

---

## 11. Final Assessment

The proposal is **competent but conventional**, with three identifiable flaws (novelty overclaim, surface-form objective, repo-component mismatch). It will likely produce a working pipeline that delivers a 0.3-0.8 perplexity-point improvement on Wikipedia-domain held-out text and a publishable workshop paper. It will **not** in its current form produce a paradigm shift, because:

- The objective preserves surface form (Bug 1, Stage 3).
- The structural index will dilute, not complement, the dense index (RETRO-k-saturation, Stage 5 §3).
- The structural probe is hand-waved (Stage 5).
- Prior work (MEMORY-VQ, COCONUT) already covers the compression and discrete-latent territory.

The **SoftProg** counter-proposal (§8) is a candidate paradigm shift: it reframes memory as soft programs rather than compressed sentences, makes paraphrase-invariance an objective rather than an emergent property, and uses repo components more aggressively. It is also riskier — the predicate-argument decoder is harder to train than surface reconstruction.

If the goal is **a publishable extension of MEMORY-VQ**, the original proposal with the fixes in §10 is sufficient. If the goal is **a new paradigm**, SoftProg or something like it is required.

---

## Sources

- [MEMORY-VQ: Compression for Tractable Internet-Scale Memory (Zemlyanskiy et al., NAACL 2024)](https://arxiv.org/abs/2308.14903)
- [Training Large Language Models to Reason in a Continuous Latent Space — COCONUT (Hao et al., Meta, Dec 2024)](https://arxiv.org/abs/2412.06769)
- [Latent Token / VQ-VAE Compression of Reasoning (Meta/Berkeley, Mar 2025)](https://www.marktechpost.com/2025/03/19/this-ai-paper-introduces-a-latent-token-approach-enhancing-llm-reasoning-efficiency-with-vq-vae-compression/)
- [Improving language models by retrieving from trillions of tokens — RETRO (Borgeaud et al., DeepMind)](https://arxiv.org/abs/2112.04426)
- [Deep Biaffine Attention for Neural Dependency Parsing (Dozat & Manning, 2017)](https://arxiv.org/abs/1611.01734)
- [Evaluation of Finetuned LLMs in AMR Parsing (2025)](https://arxiv.org/html/2508.05028)
- [Dependency Parsing is More Parameter-Efficient with Normalization (2025)](https://arxiv.org/html/2505.20215v2)
- [DiaParser — Direct Attentive Dependency Parser](https://github.com/Unipisa/diaparser)
- Repository components cited: `src/dl_techniques/models/tree_transformer/`, `models/vq_vae/`, `models/byte_latent_transformer/`, `models/gemma/`, `models/memory_bank/`, `layers/vector_quantizer.py`, `layers/vector_quantizer_rotation_trick.py`, `layers/embedding/hierarchical_codebook_embedding.py`, `layers/attention/perceiver_attention.py`, `layers/attention/shared_weights_cross_attention.py`, `layers/memory/ntm_interface.py`, `layers/memory/som_nd_soft_layer.py`.
