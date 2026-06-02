# Embedding Model Benchmarks

Reference targets for the dl_techniques causal-LM + MRL embedding experiments.
Numbers are from public MTEB leaderboard snapshots and original papers; cite the date you pulled them.

> **Snapshot date**: 2026-05-13

## What MTEB Measures

The Massive Text Embedding Benchmark (MTEB; Muennighoff et al., 2022, arXiv:2210.07316) aggregates **56 English datasets** across **8 task families**: retrieval (15 datasets, derived from BEIR), reranking (4), clustering (11), classification (12), pair classification (3), STS (10), summarization (1), and bitext mining. Each task family uses the metric appropriate for it (nDCG@10 for retrieval, MAP for reranking, V-measure for clustering, accuracy for classification, Spearman for STS), then the "MTEB Average" reported on the leaderboard is the unweighted mean over all 56 datasets. Higher is better, no caps.

MTEB has since been extended to **MMTEB** (multilingual, 250+ languages), **MTEB-v2**, and several language-specific tracks (C-MTEB, MTEB-fr, MTEB-pl). Unless noted, all numbers in the tables below are MTEB English v1, which is what the public leaderboard shows by default.

## How to Read These Numbers

- **MTEB Average** is convenient but coarse. It mixes very different tasks; a model that crushes classification but is mediocre at retrieval can post the same average as the inverse. Always check the per-family columns for your use case.
- **Retrieval Avg** is the column that matters for RAG / dense search. STS Avg is what matters for paraphrase, deduplication, and clustering quality.
- **Classification Avg** on MTEB is dominated by classifier-on-frozen-embeddings probes; high scores reward embeddings that linearly separate classes, not embeddings that are good for retrieval.
- **Contamination**: any model whose contrastive training mix included MTEB train splits (or near-duplicates) inflates its score. The official leaderboard pins a "training data overlap" discussion; treat 7B-tier scores skeptically when the model was trained on synthetic data generated from GPT-4 against MTEB-style queries.
- **Self-reported vs leaderboard-evaluated**: the leaderboard ingests numbers from `mteb run`; the model card sometimes lists higher self-reported numbers. Where they disagree, the leaderboard wins for cross-comparison.

## Benchmark Tables

### Tiny (<=50M params)

| Model | Params | Dim | Ctx | MTEB Avg | Retrieval | STS | Classif | MRL | Instruct | Year | License |
|-------|--------|-----|-----|----------|-----------|-----|---------|-----|----------|------|---------|
| paraphrase-MiniLM-L3-v2 | 17M | 384 | 128 | ? | ? | ? | ? | No | No | 2021 | Apache-2.0 |
| all-MiniLM-L6-v2 | 23M | 384 | 256 | 56.26 | 41.95 | 78.90 | 63.05 | No | No | 2021 | Apache-2.0 |
| all-MiniLM-L12-v2 | 33M | 384 | 256 | 56.53 | 42.69 | 79.32 | 63.21 | No | No | 2021 | Apache-2.0 |
| bge-small-en-v1.5 | 33M | 384 | 512 | 62.17 | 51.68 | 81.59 | 74.14 | No | Yes (query prefix) | 2023 | MIT |
| e5-small-v2 | 33M | 384 | 512 | 59.93 | 49.04 | 80.39 | 72.94 | No | Yes (query/passage prefix) | 2022 | MIT |

### Small (50-150M)

| Model | Params | Dim | Ctx | MTEB Avg | Retrieval | STS | Classif | MRL | Instruct | Year | License |
|-------|--------|-----|-----|----------|-----------|-----|---------|-----|----------|------|---------|
| all-mpnet-base-v2 | 110M | 768 | 384 | 57.78 | 43.81 | 80.28 | 65.07 | No | No | 2021 | Apache-2.0 |
| bge-base-en-v1.5 | 109M | 768 | 512 | 63.55 | 53.25 | 82.40 | 75.53 | No | Yes (query prefix) | 2023 | MIT |
| e5-base-v2 | 110M | 768 | 512 | 61.50 | 50.29 | 81.06 | 73.84 | No | Yes (query/passage prefix) | 2022 | MIT |
| nomic-embed-text-v1 | 137M | 768 | 8K | 62.39 | 53.01 | 81.94 | 74.12 | No | Yes (task prefix) | 2024 | Apache-2.0 |
| nomic-embed-text-v1.5 | 137M | 768 (-> 64) | 8K | 62.28 | 53.01 | 81.94 | 73.55 | Yes (768/512/256/128/64) | Yes (task prefix) | 2024 | Apache-2.0 |
| jina-embeddings-v2-base-en | 137M | 768 | 8K | 60.38 | 47.87 | 80.70 | 73.45 | No | No | 2023 | Apache-2.0 |
| snowflake-arctic-embed-m | 110M | 768 | 512 | 64.20 | 54.90 | 82.06 | 75.99 | No | Yes (query prefix) | 2024 | Apache-2.0 |

### Medium (150M-500M)

| Model | Params | Dim | Ctx | MTEB Avg | Retrieval | STS | Classif | MRL | Instruct | Year | License |
|-------|--------|-----|-----|----------|-----------|-----|---------|-----|----------|------|---------|
| bge-large-en-v1.5 | 335M | 1024 | 512 | 64.23 | 54.29 | 83.11 | 75.97 | No | Yes (query prefix) | 2023 | MIT |
| e5-large-v2 | 335M | 1024 | 512 | 62.25 | 50.56 | 81.78 | 75.24 | No | Yes (query/passage prefix) | 2022 | MIT |
| bge-m3 (dense) | 568M | 1024 | 8K | 59.84 | 48.82 | 78.71 | 74.06 | No | No | 2024 | MIT |
| gte-large-en-v1.5 | 434M | 1024 | 8K | 65.39 | 57.91 | 81.43 | 77.75 | No | No | 2024 | Apache-2.0 |
| mxbai-embed-large-v1 | 335M | 1024 (-> 512) | 512 | 64.68 | 54.39 | 85.00 | 75.64 | Yes (MRL + binary) | Yes (query prefix) | 2024 | Apache-2.0 |
| snowflake-arctic-embed-l | 335M | 1024 | 512 | 64.18 | 55.98 | 82.13 | 75.83 | No | Yes (query prefix) | 2024 | Apache-2.0 |
| jina-embeddings-v3 | 570M | 1024 (-> 32) | 8K | 65.52 | 53.88 | 85.80 | 82.58 | Yes (1024 down to 32) | Yes (task LoRA) | 2024 | CC-BY-NC-4.0 |
| multilingual-e5-large-instruct | 560M | 1024 | 512 | 64.41 | 57.12 | 80.97 | 77.56 | No | Yes (instruction) | 2024 | MIT |

### Large (500M-2B)

| Model | Params | Dim | Ctx | MTEB Avg | Retrieval | STS | Classif | MRL | Instruct | Year | License |
|-------|--------|-----|-----|----------|-----------|-----|---------|-----|----------|------|---------|
| gte-Qwen2-1.5B-instruct | 1.5B | 1536 | 32K | 67.16 | 58.29 | 82.78 | 79.65 | No | Yes (instruction) | 2024 | Apache-2.0 |
| NV-Embed-v1 | 7.85B | 4096 | 4K | 69.32 | 59.36 | 82.84 | 87.35 | No | Yes (instruction) | 2024 | CC-BY-NC-4.0 |

### XL (>=7B, decoder-LLM-based)

| Model | Params | Dim | Ctx | MTEB Avg | Retrieval | STS | Classif | MRL | Instruct | Year | License |
|-------|--------|-----|-----|----------|-----------|-----|---------|-----|----------|------|---------|
| e5-mistral-7b-instruct | 7.11B | 4096 | 4K | 66.63 | 56.89 | 84.63 | 78.47 | No | Yes (instruction) | 2024 | MIT |
| SFR-Embedding-Mistral | 7.11B | 4096 | 4K | 67.56 | 59.00 | 85.05 | 78.33 | No | Yes (instruction) | 2024 | CC-BY-NC-4.0 |
| Linq-Embed-Mistral | 7.11B | 4096 | 4K | 68.20 | 60.20 | 84.69 | 80.20 | No | Yes (instruction) | 2024 | CC-BY-NC-4.0 |
| gte-Qwen2-7B-instruct | 7.61B | 3584 | 32K | 70.24 | 60.25 | 83.39 | 86.58 | No | Yes (instruction) | 2024 | Apache-2.0 |
| NV-Embed-v2 | 7.85B | 4096 | 32K | 72.31 | 62.65 | 84.31 | 90.37 | No | Yes (instruction) | 2024 | CC-BY-NC-4.0 |

### Proprietary (API)

| Model | Params | Dim | Ctx | MTEB Avg | Retrieval | STS | Classif | MRL | Instruct | Year | License |
|-------|--------|-----|-----|----------|-----------|-----|---------|-----|----------|------|---------|
| OpenAI text-embedding-3-small | n/a | 1536 (-> 512) | 8191 | 62.26 | 51.08 | 81.34 | 73.16 | Yes (MRL) | No | 2024 | Proprietary |
| OpenAI text-embedding-3-large | n/a | 3072 (-> 256) | 8191 | 64.60 | 55.44 | 81.73 | 75.45 | Yes (MRL) | No | 2024 | Proprietary |
| Cohere embed-english-v3.0 | n/a | 1024 | 512 | 64.47 | 55.00 | 81.42 | 76.49 | No (int8/binary quant) | Yes (input_type) | 2023 | Proprietary |
| Cohere embed-multilingual-v3.0 | n/a | 1024 | 512 | 64.01 | 53.84 | 80.36 | 76.01 | No (int8/binary quant) | Yes (input_type) | 2023 | Proprietary |
| Voyage voyage-3 | n/a | 1024 | 32K | 65.40 | ? | ? | ? | No | No | 2024 | Proprietary |
| Voyage voyage-3-large | n/a | 2048 (-> 256) | 32K | 67.10 | 60.50 | ? | ? | Yes (MRL + int8/binary) | No | 2025 | Proprietary |

## 2026 SOTA Themes Captured in the Numbers

- **Decoder-LLM foundations dominate above 1B params.** NV-Embed-v2 (72.31), gte-Qwen2-7B-instruct (70.24), and Linq-Embed-Mistral (68.20) all use 7B+ decoder bases fine-tuned with instruction prompts and synthetic GPT-4 data. The encoder-only BERT-family caps out around 65 even at 335M-570M params.
- **MRL is now standard in newer releases.** OpenAI v3 (small + large), jina-embeddings-v3, nomic-embed-text-v1.5, mxbai-embed-large-v1, and voyage-3-large all ship Matryoshka heads. The "old" generation (bge-*, e5-*, all-mpnet) does not.
- **Multilingual + multi-vector** (BGE-M3) trades ~3-4 MTEB-en points vs comparably-sized English-only models for 100+ language coverage and a built-in sparse + ColBERT-style multi-vector output.
- **Instruction-tuning is universal at the top.** Every model above 65 MTEB Avg uses some form of task/instruction prefix at inference time (E5 query/passage, BGE query prefix, NV-Embed task instruction, jina-v3 task LoRA). Unconditioned encoders are no longer competitive at the top.
- **Open SOTA at small/medium scale has shifted.** 2021: `*-MiniLM`, `mpnet`. 2023: `bge-*` and `e5-*`. 2024: `gte-*v1.5`, `snowflake-arctic-embed-*`, `mxbai-*`, `jina-v3`. Each generation gained roughly 5-7 MTEB Avg points at fixed parameter count, mostly from better contrastive data and longer context.
- **API models are no longer SOTA.** OpenAI text-embedding-3-large (64.6) sits below open `gte-large-en-v1.5` (65.39) and the 7B open models clear it by 5-8 points. Cohere embed-english-v3 (64.47) is similar. The proprietary pitch is now context/latency/multilingual/quantization rather than raw MTEB.

## Sources

- [MTEB Leaderboard (HuggingFace Space)](https://huggingface.co/spaces/mteb/leaderboard) - pulled 2026-05-13
- [Awesome Agents - Embedding Model Leaderboard MTEB March 2026](https://awesomeagents.ai/leaderboards/embedding-model-leaderboard-mteb-march-2026/)
- [pecollective - Text Embedding Models Compared 2026](https://pecollective.com/tools/text-embedding-models-compared/)
- Muennighoff et al., *MTEB: Massive Text Embedding Benchmark*, arXiv:2210.07316 (2022)
- Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5), arXiv:2212.03533 (2022)
- Wang et al., *Improving Text Embeddings with Large Language Models* (E5-Mistral), arXiv:2401.00368 (2024)
- Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding* (BGE), arXiv:2309.07597 (2023)
- Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity*, arXiv:2402.03216 (2024)
- Li et al., *Towards General Text Embeddings with Multi-stage Contrastive Learning* (GTE), arXiv:2308.03281 (2023)
- Sturua et al., *jina-embeddings-v3: Multilingual Embeddings With Task LoRA*, arXiv:2409.10173 (2024)
- Nussbaum et al., *Nomic Embed: Training a Reproducible Long Context Text Embedder*, arXiv:2402.01613 (2024)
- Lee et al., *NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models*, arXiv:2405.17428 (2024)
- Merrick et al., *Arctic-Embed: Scalable, Efficient, and Accurate Text Embedding Models*, arXiv:2405.05374 (2024)
- [Salesforce Research - SFR-Embedding-Mistral blog](https://www.salesforce.com/blog/sfr-embedding/)
- [Voyage AI - voyage-3-large announcement](https://blog.voyageai.com/2025/01/07/voyage-3-large/)
- [Cohere - Embed v3 blog](https://cohere.com/blog/introducing-embed-v3)
- [OpenAI - New embedding models and API updates](https://openai.com/index/new-embedding-models-and-api-updates/)
- Individual HuggingFace model cards for all listed models, accessed 2026-05-13
