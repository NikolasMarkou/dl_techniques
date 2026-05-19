# SoftProg

**Author**: Nikolas Markou
**Hardware**: single GPU 1 (RTX 4070, 12 GB)
**Precheck artifacts**: `research/softprog_precheck/`

---

## 1. Hypothesis

**H1 (primary).** Predicate-argument structure of a sentence is a *more invariant* representation than its surface form. Two sentences with the same meaning but different word order, voice, or lexical choices share PA structure where they do not share words.

**H2 (operational).** A retrieval-augmented LM that indexes its memory by PA-structure keys (in addition to or instead of dense embeddings and BM25) will retrieve *better evidence* for queries whose surface form has drifted from the indexed text — measured as exact-match on multi-hop QA over a fixed corpus.

**H3 (adversarial, must survive).** The PA-structure signal, even if real, may be:
- (a) too weak to translate to retrieval gains (signal exists, utility doesn't), or
- (b) subsumed by dense embeddings (which approximate the same invariance implicitly), or
- (c) recoverable by BM25 at lower cost (the cheap baseline wins).

**Scope.** English text, Wikipedia-style prose, single-GPU budget, single researcher.

**Out of scope.** Cross-lingual transfer, full Wikipedia scale, learned codes, fusion modules. All deferred until H2 is decided.

---

## 2. Tests

### 2.1 Test T1 — Premise check (run, see §3.1)

Does PA-tuple representation paraphrase-align on text where surface signal fails?

- **Data.** PAWS `labeled_final` dev split. 2000 random pairs. PAWS positives and negatives have matched lexical content by construction; the dataset is *built* to break bag-of-words methods.
- **Extractor.** spaCy `en_core_web_sm`. For each VERB token: `(lemma, nsubj_lemma, dobj_lemma)`. Sentence signature = set of these tuples.
- **Comparators.**
  - PA-tuple exact-match rate on paraphrase pairs vs non-paraphrase pairs.
  - PA-tuple Jaccard on paraphrase vs non-paraphrase vs random pairs.
  - Content-lemma Jaccard (NOUN/VERB/PROPN/ADJ, stop-filtered) on the same splits.
- **Pass criterion.** PA-tuple paraphrase/non-paraphrase exact-match ratio ≥ 1.30 with content-lemma ratio ≤ 1.10 (i.e. PA discriminates where lemmas don't).
- **Kill criterion.** PA-tuple ratio < 1.10 on paraphrase vs lexically-matched non-paraphrase → H1 dead.

### 2.2 Test T2 — Retrieval utility (planned, see §4)

Does the T1 signal translate to retrieval-augmented LM exact-match on multi-hop QA?

- **Data.** 100k Simple Wikipedia sentences as memory corpus. 200 hand-written two-hop questions.
- **Retrievers.** Each retrieves top-5 evidence sentences:
  - **R_PA**: SQLite inverted index keyed on `(pred_lemma)`, `(pred_lemma, subj_lemma)`, `(pred_lemma, dobj_lemma)`. Query: spaCy-parse the question, build the same keys, look them up.
  - **R_BM25**: `rank_bm25` over the 100k sentences.
  - **R_dense**: FAISS-Flat over `sentence-transformers/all-MiniLM-L6-v2` embeddings.
  - **R_ensemble**: union of top-5 from each, deduplicated, scored by Reciprocal Rank Fusion (k=60).
- **Reader.** Frozen Gemma instance (smallest variant that fits in 12 GB; 4-bit if needed). Prompt: `"Context:\n{evidence}\n\nQuestion: {q}\nAnswer:"`.
- **Metric.** Exact match. Bootstrap 95% CI on 200 questions (n=10000 resamples).
- **Pass criterion (claim viability).** EM(R_PA) ≥ EM(R_BM25) + 3 AND EM(R_PA) ≥ EM(R_dense) + 3 *or* EM(R_ensemble) ≥ EM(best single) + 3.
- **Tie criterion (signal-but-no-utility).** |EM(R_PA) − EM(best baseline)| ≤ 2.
- **Kill criterion (signal-doesn't-translate).** EM(R_PA) < EM(R_BM25) − 2.

### 2.3 Test T3 — Coverage sanity (cheap, run before T2)

Will the parser cover real Wikipedia prose, or did PAWS flatter spaCy?

- Parse 10k random Simple Wikipedia sentences with spaCy.
- Measure: fraction with ≥ 1 PA-tuple, mean tuples per sentence, fraction with subject elision.
- **Pass.** ≥ 60% coverage, ≥ 1.5 tuples/sentence on average.
- **Fail.** < 60% coverage → upgrade to `en_core_web_trf` or `_md` before T2; re-measure.

### 2.4 Test T4 — Latency budget (cheap, run during T2 build)

Can the v0 stack serve a query end-to-end in < 200 ms p50, < 500 ms p99 on GPU 1?

- Measure: parse + index lookup + Gemma forward over a 200-question batch. Report p50/p99.
- **Pass.** Within budget.
- **Fail.** Drop dense ANN to PQ; pre-parse queries asynchronously; document overage.

---

## 3. Findings

### 3.1 T1 — Premise check (n=2000 PAWS dev pairs)

| Signal | Paraphrase (label=1, n=877) | Non-paraphrase, lexically matched (label=0, n=1123) | Random pair | P / NP ratio | P / Random ratio |
|---|---|---|---|---|---|
| **PA-tuple exact match** | **40.5%** | 27.2% | ~0% | **1.49** | — |
| **PA-tuple Jaccard** | 0.650 | 0.527 | 0.030 | **1.23** | **21.4** |
| **Content-lemma Jaccard** | 0.893 | 0.892 | 0.004 | **1.00** | 229 |
| **Root-only predicate match** | 61.7% | 55.7% | 3.5% | 1.11 | — |

**Verdict on T1**: **PASS**. PA-tuple exact-match ratio P/NP = 1.49 (≥ 1.30 threshold), content-lemma ratio = 1.00 (≤ 1.10 threshold). The structural signal exists exactly where the surface signal fails. PAWS coverage 82.3% (2569 / 3122 unique sentences yield ≥ 1 PA tuple).

**Failure modes from sample inspection** (`research/softprog_precheck/samples.jsonl`):
- **F1**: subject elision. spaCy drops subjects in passive constructions and detached relative clauses (~25% of cases). Example: "Mrs. Glad had worked … and worked as inspector" → second clause's subject is empty.
- **F2**: no coreference. "Vijay lives" vs "He lives" produce different tuples for what is semantically the same predicate-argument frame.
- **F3**: no entity-type discrimination. "Robert Goldsborough born" and "Vincent Goldsborough born" produce identical tuples, which makes the PA index unable to disambiguate two distinct people sharing a surname.
- **F4**: modifier semantics ignored. "received advice from his uncle, founder of the *first* museum" vs "*same* museum" produce identical PA tuples but mean different things.

**Implication.** The 1.49 ratio is a *floor* with a noisy off-the-shelf extractor. F1-F4 are addressable in v1 with NER + coref + modifier features, each adding signal a learned ablation could quantify.

### 3.2 T1.5 — Repository premise verification (cheap reads)

- `src/dl_techniques/models/tree_transformer/` — confirmed Wang unsupervised soft constituency (Group Attention), not Dozat biaffine dependency. Documented as such in `components.py` (`"learn soft, hierarchical constituency trees directly from text"`).
- `src/dl_techniques/layers/vector_quantizer_rotation_trick.py` — Fifty et al. ICLR 2025 rotation-trick VQ, production-grade, multi-head + EMA + dead-code-reinit + k-means init + diversity/orthogonal regularizers. Strict superset of vanilla VQ.

### 3.3 T2, T3, T4 — Not yet run

The gap between "signal exists" (T1 PASS) and "useful for an LLM" (T2) is exactly what the v0 build measures.

---

## 4. Plan

### 4.1 v0 build — 5 working days, single GPU 1

The smallest thing that uses the T1 signal in a retrieval-augmented LM and compares against the right baselines.

| Day | Task | Output |
|---|---|---|
| 1 | T3 first. Parse 10k Simple Wikipedia random sample with spaCy. Verify coverage ≥ 60%. Then parse the full 100k corpus; cache PA tuples + MiniLM embeddings to Parquet. | `corpus_features.parquet`, T3 verdict. |
| 2 | Build three retrievers: SQLite PA index, `rank_bm25` BM25, FAISS-Flat MiniLM dense. Implement Reciprocal Rank Fusion ensemble. | `retrievers/` module, smoke tests. |
| 3 | Hand-write 200 two-hop questions over the 100k corpus. Pattern: `Q: <fact1 about X> AND <fact2 about X>; A: <answer using both>`. Tedious; unavoidable. | `qa_gold.jsonl`. |
| 4 | Wire frozen Gemma + retrieve-prompt-answer loop. Four configurations: R_PA, R_BM25, R_dense, R_ensemble. Measure T4 latency during this step. | `runner.py`, `eval_results.json`. |
| 5 | Compute EM + bootstrap CIs per retriever. Pick a branch (§4.2). Document. | Decision logged here. |

### 4.2 Decision branches after T2

```
T2 result: EM(R_PA) vs EM(best baseline)
│
├── R_PA wins by ≥ 3 EM (T2 PASS)
│   → v0 is the paper. Write it up. Document T1+T2 jointly.
│   → Optional v1 experiments (each independently justified):
│        v1a NER-enriched keys     (addresses F3, ~2 days)
│        v1b coref on queries      (addresses F2, ~3 days)
│        v1c learned PA-code (RVQ) (only if v1a/b saturate, ~1 week)
│        v1d scale to 1M-10M sentences (~3 days compute on GPU 1)
│
├── R_PA ties baselines (|Δ| ≤ 2)  (T2 TIE)
│   → signal exists, utility doesn't. Try ONE addition, re-measure:
│        first F3 fix (NER buckets in PA keys) — cheapest, biggest sample-failure-rate
│        if still tie: F2 fix (coref) — heavier
│        if still tie after both: stop. Negative result paper.
│
└── R_PA loses by > 2 EM  (T2 FAIL)
    → 1.49 doesn't translate. Publish negative result.
    → Document corpus + queries + retriever code so the next person
      doesn't repeat the test.
```

### 4.3 Hardware plan

- All training/inference uses GPU 1 (RTX 4070, 12 GB). Verified free at 11.8 GB.
- No parallel GPU jobs (per memory `feedback_no_parallel_gpu`).
- spaCy parse is CPU-bound; runs in parallel with GPU work.
- Gemma at 4-bit (~2 GB for 2B variant, ~5 GB for 8B) leaves headroom for retrieval embeddings in GPU memory.

### 4.4 Risk register (post-measurement)

| ID | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | Multi-hop QA at 100k scale is too hard for small Gemma; all retrievers tie at low EM | medium | medium | Fall back to single-hop fact verification (TriviaQA subset) where any retrieval helps. |
| R2 | Simple Wikipedia is unrepresentative of real prose | low-medium | high | T3 coverage check before T2; spot-check on a 1k Wikipedia-prose slice. |
| R3 | 200 hand-written questions is a noisy estimator | high | medium | Bootstrap CIs; if T2 result is within CI of a tie, expand to public benchmark (HotpotQA, 2WikiMultiHopQA). |
| R4 | spaCy `sm` is the wrong size | low | low | Re-run T1 with `en_core_web_trf` (~30 min). |
| R5 | 4-bit Gemma loses retrieval-following ability | medium | medium | Try `_2b-it` first; fall back to Qwen-1.5B. |
| R6 | T1 result on PAWS doesn't transfer to Wikipedia prose (paraphrase distribution differs) | medium | medium | Note in T2 design: also evaluate on a 1k held-out Simple-Wiki-self-paraphrase pair set generated by back-translation. |

### 4.5 What this plan refuses to do (preemptive)

- Build VQ codes before T2 measures whether unlearned spaCy PA tuples are enough.
- Build fusion / LoRA / Perceiver-IO before T2 measures whether in-context retrieval works.
- Claim cross-lingual transfer before T2 measures English.
- Claim a "soft program" paradigm before the compose/substitute/inspect tests are defined and run.
- Index full Wikipedia before T2 says 100k works.

---

## 5. Open questions (deferred to post-T2)

1. Is the 1.49 ratio specific to PAWS adversarial pairs, or does it generalize to natural paraphrases? Test on a back-translation pair set or PAWS-X.
2. Does the signal compose? If two retrieved entries share one PA argument, does the LLM successfully chain them in a two-hop answer? T2 measures this implicitly via EM.
3. Is RVQ over learned span embeddings *strictly better* than a hash of spaCy lemmas? Would need a v1 ablation.

---

## 6. Sources

- Precheck code + results: `research/softprog_precheck/{run_precheck.py, results.json, samples.jsonl}`.
- PAWS dataset: [Zhang, Baldridge, He (NAACL 2019)](https://github.com/google-research-datasets/paws).
- MEMORY-VQ (v1 baseline if learned codes return): [Zemlyanskiy et al., NAACL 2024](https://arxiv.org/abs/2308.14903).
- Rotation-trick VQ (in repo): `src/dl_techniques/layers/vector_quantizer_rotation_trick.py`, Fifty et al. ICLR 2025.
- Wang Tree Transformer (in repo): `src/dl_techniques/models/tree_transformer/`.
