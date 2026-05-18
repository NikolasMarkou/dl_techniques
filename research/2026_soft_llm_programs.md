# SoftProg (bones)

**Working title**: SoftProg
**What it actually is**: predicate-argument structure as a retrieval key for an LLM.
**Status**: 200 lines of Python, not a paradigm. If it works, then complexity is earned.
**Date**: 2026-05-18

---

## The whole idea

Surface text is one realization of an underlying predicate-argument structure. Paraphrases share that structure. Dense embeddings approximate it; BM25 misses it; predicate-argument tuples *are* it.

So: build a retrieval index keyed on predicate-argument tuples extracted from text by an off-the-shelf parser. At inference, parse the query, look up entries by tuple overlap, hand the retrieved sentences to a frozen LLM as plain context. Measure.

No training. No VQ. No fusion module. No new layers.

---

## Components

| Piece | What | How |
|---|---|---|
| **Parser** | Extract `(predicate_lemma, [arg_lemmas])` per sentence | `amrlib` (off-the-shelf AMR) or Claude/GPT-4 via API for the pilot |
| **Corpus** | Source of memory entries | 100k Simple Wikipedia sentences (not 80M; not Wikipedia full) |
| **Index** | Map from tuple parts to entries | Python `dict[predicate_lemma → list[entry_id]]` + `dict[arg_lemma → list[entry_id]]`. SQLite if it outgrows RAM. |
| **Retrieval** | Score entries by predicate match + arg overlap | k=5 entries by `(pred_match * 2 + arg_jaccard)` |
| **LLM** | Read retrieved entries as text | Frozen Gemma-3-2B (repo) or any local model. No LoRA. No fusion. Plain in-context. |
| **Evaluation** | What we measure | Three numbers (§ Gates) |

That's the whole architecture.

---

## Pre-check (1 day, kill switch)

Before writing the retriever:

1. Take 1k paraphrase pairs (PAWS) + 1k random non-paraphrase pairs from the same corpus.
2. Extract `(predicate, args)` tuples for all 4k sentences.
3. Compute tuple match rate on paraphrases vs random.

**Pass**: paraphrase tuple match ≥ 3× random tuple match.
**Fail**: < 2×. Project dies. One day spent.

If this fails, no architecture saves the thesis. Paraphrase-invariance of predicate-argument structure was an assumption; this measures whether it's a fact.

---

## Build (1 week if pre-check passes)

| Day | Task |
|---|---|
| 1-2 | Parse 100k Simple Wikipedia sentences. Cache the JSON. |
| 3 | Build the two dicts. Implement retrieval scorer. |
| 4 | Wire frozen Gemma-3-2B + prompt template `"Context:\n{retrieved}\n\nQuestion: {q}\nAnswer:"`. |
| 5-6 | Run the three evaluations below. |
| 7 | Write up the result. |

---

## Gates (only three)

1. **Pre-check tuple match**: paraphrase / random ≥ 3×.
2. **Multi-hop QA**: 200 hand-written two-hop questions over the 100k corpus. SoftProg vs (a) no-retrieval Gemma-3-2B, (b) BM25 retrieval over the same 100k. SoftProg must beat BM25 by ≥ 5 points EM.
3. **Inspectability**: print the retrieved tuple. A human reading it understands what the LLM saw. No metric — just a screenshot in the paper.

Three gates. If all three pass, *then* think about VQ codes, learned fusion, full Wikipedia. Not before.

---

## What this drops from the prior spec

- Tree Transformer
- Biaffine relation head
- Hierarchical-Residual VQ (4 levels × 1024)
- Predicate-argument decoder
- Paraphrase-contrastive loss
- Cross-lingual contrastive loss
- Type-signature inverted index (RocksDB)
- FAISS HNSW-PQ dense index
- Perceiver-IO fusion
- LoRA on Gemma-3-8B
- Wikipedia full scale (80M sentences)
- Eight gates
- The "soft program" framing as a research claim

All of those become *follow-up work earned by a passing first result*, not the first result itself.

---

## What survives

A measurement and a 200-line retriever. If the measurement says the thesis is alive, you have a paper. If not, you spent a week and learned something true.

---

## Honest assessment

This is RAG with a different retrieval key. The novelty is "structured key derived from a parser, not a dense embedding." That has been tried; it has worked sometimes (knowledge-graph-augmented RAG) and failed others. Whether *this particular keying* beats BM25 + dense on multi-hop QA is the empirical question. It is *one* empirical question, answerable in a week, by one person.

If you want to invent a paradigm, invent it after the measurement passes.
