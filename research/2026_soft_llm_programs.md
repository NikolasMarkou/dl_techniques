# SoftProg (measurement-grounded)

**Working title**: SoftProg.
**What it is**: predicate-argument structure as a retrieval key for an LLM.
**Status**: premise checked on real data (PAWS, n=2000, 2026-05-18). Signal exists. Build the bones, measure again, then decide.
**Hardware constraint**: single GPU 1 (RTX 4070, 12 GB).
**Author**: Nikolas Markou.

---

## 0. The precheck (already run)

Question: do predicate-argument tuples paraphrase-align where surface text does not?

Script: `research/softprog_precheck/run_precheck.py`. Data: PAWS labeled-final dev. Extractor: spaCy `en_core_web_sm`, `(root verb lemma, nsubj lemma, dobj lemma)` per verb. Comparator: Jaccard + exact-match over the tuple set.

| Signal | Paraphrase (label=1) | Non-paraphrase (label=0, lexically matched) | Random pair | P / NP | P / Random |
|---|---|---|---|---|---|
| PAS-tuple exact match | **40.5%** | 27.2% | ~0% | **1.49×** | — |
| PAS-tuple Jaccard | 0.650 | 0.527 | 0.030 | **1.23×** | **21.4×** |
| Content-lemma Jaccard | 0.893 | 0.892 | 0.004 | **1.00×** | 229× |

**Reading**. The content-lemma baseline cannot tell paraphrases from non-paraphrases on PAWS (1.00×). PAS-tuples can (1.49×). The structural signal lives in exactly the place where surface signal fails. This is a real, replicable, single-laptop result.

**Caveats from inspecting samples**:
- 18% of PAWS sentences yield zero PAS tuples (fragments, lists).
- spaCy drops subjects in passives or detached relative clauses (~25% of cases).
- No coreference: "Vijay lives" and "He lives" parse to different subjects.
- No entity-type discrimination: "Robert Goldsborough born" vs "Vincent Goldsborough born" produce identical tuples and confuse paraphrase detection on the negative side.

A learned, coref-resolved, NER-enriched extractor would push the 1.49× higher. But the floor is already 1.49× without any training.

---

## 1. What we build (bones)

A retrieval-augmented inference path that uses PAS tuples as an inverted-index key, *in addition to* dense embeddings. No training in v0.

| Component | Choice | Justification |
|---|---|---|
| Parser / PA extractor | spaCy `en_core_web_sm` | Already installed; gives the 1.49× signal measured. |
| Memory corpus | 100k Simple Wikipedia sentences | Fits on a 4070; cheaper than full Wikipedia by 800×. |
| PA index | Python dict → SQLite. Keys: `(predicate_lemma)`, `(predicate_lemma, subj_lemma)`, `(predicate_lemma, dobj_lemma)`. | No FAISS, no RocksDB. SQLite trivially supports the lookup pattern. |
| Dense baseline (must beat) | `sentence-transformers/all-MiniLM-L6-v2` (~22M params, runs on CPU) | The honest comparator for retrieval. |
| Sparse baseline (must beat) | BM25 over the same 100k corpus (`rank_bm25` package) | The cheap-and-strong baseline that gets skipped in too many papers. |
| Reader LLM | Frozen Gemma-3 (smallest variant that fits in 12 GB at 4-bit, or repo's existing Gemma) | No LoRA, no fusion module. Plain in-context retrieval. |
| Eval | Multi-hop QA over the 100k corpus (200 hand-written two-hop questions) + held-out perplexity vs no-retrieval | Three numbers, comparable to baselines. |

Everything in this v0 fits on one GPU and one researcher's week. No RVQ, no Perceiver-IO, no learned codes, no LoRA. Those are v1 if v0 wins.

---

## 2. Decision tree (when to escalate)

```
v0 measured against BM25 + dense on 200-question multi-hop QA
│
├── PA-index beats BM25 + dense by ≥ 3 EM points
│     → v0 is the paper. Write it. Stop.
│
├── PA-index ties BM25 + dense (±2 EM points)
│     → the signal exists but spaCy/PAWS-tuple is too coarse.
│       Add ONE of the following, measure again:
│         (a) NER on top of PAS (entity-type buckets in the key)
│         (b) coreference resolution (use AllenNLP or a small model)
│         (c) ensemble PA-retrieval with BM25 score
│     → if any addition lifts to ≥ +3 EM, ship that version.
│
└── PA-index loses to BM25 + dense by > 2 EM
      → the 1.49× signal does not translate to retrieval utility.
        Document the negative result. Stop.
```

Three outcomes. All three are publishable (negative result is honest). At most ~6 weeks total. No 8-week-grand-plan.

---

## 3. Concrete day-by-day

| Day | Work | Output |
|---|---|---|
| 1 | Build the Simple Wikipedia 100k corpus. Parse with spaCy. Cache PAS tuples + content lemmas + sentence embeddings (MiniLM) to disk. | `~5 GB` of cached features. |
| 2 | Build the three indices: SQLite PA index, BM25, FAISS-Flat dense (100k is tiny, no PQ needed). | 3 indices on disk. |
| 3 | Hand-write 200 two-hop questions over the corpus. Pattern: `Q: <fact1 about X> and <fact2 about X>; A: <answer requiring both>`. Tedious but unavoidable. | `qa_gold.jsonl`. |
| 4 | Wire frozen Gemma into a `retrieve → prompt → answer` loop. Three configurations: PA-only, BM25, dense, PA+BM25 ensemble. | `eval_results.json`. |
| 5 | Run the eval. Look at the table. Pick a branch from §2. | Decision. |

---

## 4. What v0 deliberately leaves out

Everything from the prior spec rev that didn't measure:

- VQ codes (rotation-trick or otherwise). Not needed for v0; spaCy gives the signal.
- Tree Transformer encoder. Not needed; spaCy parses good enough.
- Biaffine relation head. Not needed; UD relations come free with spaCy.
- Predicate-argument decoder. Not needed; we use the extractor directly, not learned codes.
- Perceiver-IO fusion. Not needed; in-context retrieval works.
- LoRA. Not needed; frozen LLM.
- Cross-lingual claim. English-only. Test it if v0 wins.
- Eight gates. Three numbers (EM on multi-hop, PPL, latency).
- "Soft program" framing. The retriever is structured. That's it.

Each can come back in v1, justified by a v0 measurement.

---

## 5. v1 candidates (only if v0 wins by ≥ 3 EM)

In order of cheapest-with-clearest-upside:

1. **Entity-type-enriched PA keys**. Sample failure mode: "Robert Goldsborough born" = "Vincent Goldsborough born" under spaCy. Add NER type and lemma to the key. ~2 days.
2. **Coreference resolution on the input query** so "He lives in X" matches indexed "Vijay lives in X". ~3 days with a small pretrained coref model.
3. **Learned PA-code via rotation-trick VQ** over spaCy embeddings. Only worthwhile if v0 + entity-type + coref still has a measurable gap to fill. Uses `src/dl_techniques/layers/vector_quantizer_rotation_trick.py` (already production-grade in this repo). ~1 week.
4. **LoRA fusion** at LLM mid-layer if in-context retrieval saturates context budget. ~1 week.
5. **Scale to full English Wikipedia** if everything works on 100k. ~3 days of compute.

Each step adds **only if the previous measurement justifies it.**

---

## 6. Risks (real, after measurement)

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Multi-hop QA is too hard for Gemma at this scale and all three retrievers tie at low EM | medium | medium | Switch from multi-hop to single-hop fact-verification (TriviaQA subset) where any retrieval helps. |
| Simple Wikipedia is too clean — PAS-tuple signal is artifact of clean parsing, won't transfer to messy text | low-medium | high | Spot-check parser coverage on a 10k Wikipedia-prose slice. If coverage drops below 60%, escalate to a stronger parser before v0. |
| spaCy `en_core_web_sm` is the wrong size — `_md` or `_trf` may shift the result | low | low | Re-run precheck with `en_core_web_trf`. ~30 min. |
| 200 hand-written questions is a noisy estimator | high | medium | Bootstrap CIs on EM; use a public multi-hop benchmark (HotpotQA, 2WikiMultiHopQA) if budgets allow. |
| Frozen Gemma in 12 GB requires 4-bit quant which degrades retrieval-following | medium | medium | Try Gemma-2B-it at 4-bit; fall back to Qwen-1.5B if memory tight. |

---

## 7. What's measured, what isn't

Already measured (precheck):
- ✅ PAS-tuple paraphrase signal exists at 1.49× on adversarial PAWS, 21× vs random.
- ✅ Content-lemma baseline cannot match the discriminator (1.00× on PAWS).
- ✅ Coverage 82% on PAWS-style English Wikipedia prose.
- ✅ Failure modes documented from samples.

Not yet measured (would require v0 build):
- Whether the 1.49× signal translates to retrieval-augmented LM EM gains.
- Whether PA-index beats BM25.
- Whether PA-index beats dense MiniLM.
- Whether the ensemble dominates either.
- Latency end-to-end on GPU 1 with frozen Gemma.

The gap between "signal exists" and "useful for an LLM" is exactly what v0 measures.

---

## 8. Honest assessment

- The premise was unverified for two committed spec revisions; it is now measured.
- The result is real but modest: 1.49× exact-match on adversarial paraphrases is not a paradigm shift; it is enough signal to be worth one week of v0.
- The right next action is to **build the smallest thing that uses the signal and measure whether it beats the right baselines**, not to spec a more elaborate architecture.
- If v0 loses to BM25, the honest paper is the negative result. That is still a contribution.

---

## Sources

- Precheck script and results: `research/softprog_precheck/run_precheck.py`, `results.json`, `samples.jsonl`.
- PAWS: [Zhang, Baldridge, He, NAACL 2019](https://github.com/google-research-datasets/paws).
- MEMORY-VQ (baseline to position against if v1 happens): [Zemlyanskiy et al., NAACL 2024](https://arxiv.org/abs/2308.14903).
- Wang Tree Transformer (already in repo, optional v1 component): `src/dl_techniques/models/tree_transformer/`.
- Rotation-trick VQ (already in repo, optional v1 component): `src/dl_techniques/layers/vector_quantizer_rotation_trick.py`, Fifty et al. ICLR 2025.
