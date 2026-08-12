# Sparse Attention & Transformer Architectures — Survey

Self-contained reference of distinct interpretations/implementations of "sparse attention" across the literature. Grouped by what the sparsity actually targets: (A) fixed/structured sparse patterns (pre-defined, trained-in), (B) learned content-based routing (the pattern is discovered, not fixed), (C) training-free inference-time KV-cache sparsity (post-hoc, no retraining), (D) hardware-aligned natively-trainable sparsity, (E) low-rank/kernel approximations often grouped with sparse attention in benchmarks. Each entry includes **measured results / impact**.

---

## A. Fixed/Structured Sparse Patterns (the pattern is hand-designed, baked in at training time)

### 1. Sparse Transformer (Child, Gray, Radford, Sutskever — OpenAI, 2019)
- **Published:** Apr 23, 2019, arXiv:1904.10509
- **Summary:** Introduces factorized "strided" and "fixed" sparse attention patterns that decompose full O(n²) attention into two sparser sub-patterns (a local/strided one and a column/fixed one) whose combination still lets information propagate across the whole sequence in O(n√n).
- **Distinction:** The foundational entry in this whole space — the first to show a hand-designed, non-learned sparsity pattern could replace full attention outright (not just approximate it) while still modeling very long sequences, across text, image, and audio domains with one architecture.
- **Results/impact:** <cite index="27-1">Reduces the complexity of attention from quadratic to O(n√n) and can model sequences tens of thousands of timesteps long using hundreds of layers</cite>, and set a new state of the art for density modeling on Enwik8, CIFAR-10, and ImageNet-64 using the same architecture across text, image, and audio. OpenAI publicly framed the result as roughly a 30x increase in the practically modelable sequence length. Ancestor of nearly every later "structured sparse pattern" method (Longformer, BigBird, MInference's static patterns).

### 2. Longformer (Beltagy, Peters, Cohan, 2020)
- **Published:** Apr 2020, arXiv:2004.05150
- **Summary:** Combines a fixed sliding-window (local) attention pattern with a small number of task-specific "global" tokens (e.g. `[CLS]`) that attend to and are attended by everything, giving linear-in-sequence-length complexity.
- **Distinction:** Simplest and most widely adopted pattern in the family — pure window + a handful of global anchors, no clustering, hashing, or randomness. Optimized for long documents rather than autoregressive generation.
- **Results/impact:** <cite index="42-1">Reports 94.8% accuracy on Hyperpartisan News Detection at 4096-token context length</cite>, and became a standard long-document baseline; on the Long Range Arena (LRA) benchmark it remains a strong, memory-efficient reference point though BigBird edges it out on average score.

### 3. BigBird (Zaheer, Guruganesh, Dubey et al., NeurIPS 2020)
- **Published:** Jul 2020, arXiv:2007.14062
- **Summary:** Extends Longformer's local+global pattern with an added **random** attention component (each query also attends to a small random set of tokens), and proves theoretically that this combination is a universal approximator of full attention and Turing-complete.
- **Distinction:** Adds the theoretical guarantee missing from Longformer (random edges give the sparse graph the expander-graph properties needed to approximate dense attention), at the cost of a slightly more complex three-part pattern.
- **Results/impact:** <cite index="41-1">Across the Long Range Arena benchmark, BigBird achieves the best qualitative performance integrated across all five tasks, with consistently good — if rarely best-in-class — performance on any single task</cite>, while being far more memory-efficient than vanilla Transformers at 3–4K token lengths.

### 4. Routing Transformer (Roy, Saffar, Vaswani, Grangier, 2020)
- **Published:** Mar 2020, arXiv:2003.05997 (TACL)
- **Summary:** Learns sparsity via **online k-means clustering** of queries and keys — tokens in the same content cluster attend to one another, rather than attending based on fixed position (window/stride) or random edges.
- **Distinction:** The first entry in this survey where the sparsity pattern is *learned from content* rather than fixed by position — a conceptual bridge to the later "content-based routing" family (SeerAttention, MoSA below), but using classical clustering instead of a learned gate.
- **Results/impact:** A later adaptive-sparse-attention comparison reports its own learnable/adaptive method <cite index="37-1">achieving an average 0.6% accuracy improvement and about 18% lower inference time than Routing Transformer</cite>, implying Routing Transformer itself was a strong but beatable clustering-based baseline; it remained a standard comparison point for content-adaptive sparsity for several years.

### 5. Sinkhorn Transformer / Synthesizer (Tay et al., 2020)
- **Published:** 2020, referenced together in the Long Range Arena benchmark (arXiv:2011.04006)
- **Summary:** Sinkhorn Transformer sorts blocks of tokens via a differentiable sorting network (Sinkhorn normalization) so that attention is computed only within reordered, locally-coherent blocks; Synthesizer replaces content-based attention scores with learned or randomly-initialized synthetic attention weights.
- **Distinction:** Sinkhorn Transformer is a *sorting/permutation*-based sparsity method (yet another way of deciding which tokens matter to which), distinct from windowing, hashing, and clustering. Synthesizer is more radical: it questions whether query-key dot products are even necessary for a useful "sparse-like" attention weighting.
- **Results/impact:** <cite index="41-1">Evaluated head-to-head against Sparse Transformer, Longformer, Linformer, Reformer, BigBird, Linear Transformers, and Performers on the Long Range Arena, the ten models represent a diverse cross-section of the efficient-attention design space</cite>, with Sinkhorn Transformer landing in the middle of the pack on both accuracy and memory efficiency rather than leading on any axis — a useful "less proven at scale" data point in the family.

---

## B. Locality-Sensitive Hashing / Similarity-Based Sparsity

### 6. Reformer (Kitaev, Kaiser, Levskaya, 2020)
- **Published:** Jan 2020, arXiv:2001.04451 (ICLR 2020)
- **Summary:** Uses **Locality-Sensitive Hashing (LSH)** to bucket queries and keys by approximate cosine similarity, so each token only attends within its hash bucket; combined with reversible residual layers to cut activation-memory cost during backprop.
- **Distinction:** The only entry here that pairs sparsity with a *memory-architecture* change (reversible layers) rather than sparsity alone — its main selling point was training 64K-length sequences on a single accelerator, not raw accuracy gains.
- **Results/impact:** <cite index="45-1">Combined with reversible residual layers, Reformer was the first architecture in this lineage to support 64K-length sequence training</cite>. However, independent replications are candid about a real weakness: <cite index="44-1">Reformer processes longer sequences faster and more memory-efficiently, but no accuracy improvement over full attention is observed when using it</cite> — one of the few "efficiency-only, no free accuracy" results in the survey.

### 7. SMYRF (Daras, Kitaev, Odena, Dimakis, 2020)
- **Published:** Oct 2020, arXiv:2010.05315
- **Summary:** Replaces Reformer's angular LSH with an **asymmetric clustering transform** designed specifically to maximize query–key inner products (rather than minimize Euclidean/angular distance), so clusters better match what attention actually needs.
- **Distinction:** A direct, drop-in refinement of the LSH-attention idea — same "bucket and attend locally" skeleton as Reformer, but a geometry better matched to dot-product attention rather than distance-based hashing.
- **Results/impact:** <cite index="42-1">SMYRF's asymmetric clustering transformation was shown to largely outperform both the E2LSH and Reformer's cross-polytope LSH schemes when substituted into an otherwise identical model on the IMDB benchmark</cite> — a clean ablation showing the *clustering geometry*, not just "sparsity" in the abstract, is what drives quality in this family.

---

## C. Learned, Content-Based Sparse Routing (pattern discovered per input, not fixed, not clustered offline)

### 8. SeerAttention (Gao et al., Microsoft, 2024)
- **Published:** Oct 17, 2024, arXiv:2410.13276 (NeurIPS 2025)
- **Summary:** Adds a small learnable "gate" module — inspired by Mixture-of-Experts routing — that pools queries/keys per block and predicts which blocks of the attention matrix are worth computing; trained via lightweight self-distillation from a pretrained dense model rather than from scratch.
- **Distinction:** Unlike MInference/MoA below (which *calibrate* a static pattern offline), SeerAttention *learns* a small neural gate that adapts sparsity per input at block granularity — a middle ground between fully-fixed patterns and fully-dynamic per-token routing.
- **Results/impact:** Its own extension, SeerAttention-R, <cite index="69-1">trained on just 0.4B tokens, maintains near-lossless reasoning accuracy with a 4K-token budget on the AIME benchmark even at large 64/128-token sparse block sizes</cite>, and later work composing it with orthogonal token-level sparsity <cite index="72-1">pushed SeerAttention's attention speedup from 2.19x to 2.47x on RULER with negligible accuracy change</cite> — evidence the learned-gate approach composes well with other sparsity axes.

### 9. Mixture of Sparse Attention — MoSA (Piękos, Csordás, Schmidhuber, 2025)
- **Published:** May 1, 2025, arXiv:2505.00315
- **Summary:** Borrows expert-choice routing from Mixture-of-Experts and applies it to *tokens within each attention head*: each head dynamically selects k tokens to attend to (from a sequence of length T), cutting a single head's cost from O(T²) to O(k²+T), which frees compute budget to run more, more-specialized heads.
- **Distinction:** The clearest "positive outlier" in the sparse-attention literature — nearly every other sparse method trades some accuracy for speed; MoSA is presented as the exception.
- **Results/impact:** <cite index="73-1">In an IsoFLOP setting across four scales (28M–516M parameters), MoSA is the only sparse attention method among those tested that improved perplexity over the dense baseline, by up to 27% at matched compute</cite>, and <cite index="73-1">in a perplexity-matched setting a pure PyTorch implementation (no custom CUDA kernel) still improved wall-clock time and memory simultaneously while shrinking the KV cache</cite> — a rare case where sparsity is a net win on quality, not just efficiency.

### 10. MInference 1.0 (Jiang, Li, Zhang et al., Microsoft, NeurIPS 2024)
- **Published:** Jul 2, 2024, arXiv:2407.02490
- **Summary:** Identifies three recurring sparse-attention "shapes" across long-context LLM heads — A-shape, Vertical-Slash, and Block-Sparse — then uses an offline, kernel-aware search to assign the best pattern+parameters to each head, and reconstructs the actual sparse mask dynamically per input at inference time.
- **Distinction:** A hybrid of "fixed pattern family" (group A) and "dynamic per-input" sparsity: the *shape* is chosen once per head offline, but the specific mask within that shape is estimated fresh for every prompt — aimed squarely at accelerating the prefill stage of very long contexts, not decoding.
- **Results/impact:** <cite index="62-1">Reduces attention FLOPs by 95% and achieves up to 10x speedup for 1M-token contexts on a single A100 GPU, cutting latency from 30 minutes to 3 minutes per prompt</cite>, while matching or surpassing dense-attention baselines on Needle-in-a-Haystack, RULER, and InfiniteBench — one of the largest absolute wall-clock wins in this survey.

### 11. Quest (Tang, Zhao, Zhu et al., MIT/SJTU, 2024)
- **Published:** Jun 16, 2024, arXiv:2406.10774
- **Summary:** Observes that which tokens are "critical" depends on the *current query*, not just the key content — so instead of a fixed important-token set, Quest summarizes KV-cache pages by their per-channel min/max key values and estimates each page's relevance freshly for every incoming query before loading only the top-K pages.
- **Distinction:** Directly rebuts the query-agnostic assumption behind static/offline-calibrated methods (e.g. H2O, MInference's per-head patterns): Quest's sparsity decision is re-computed at every decoding step, per query, not fixed per head or per prompt.
- **Results/impact:** <cite index="57-1">Achieves up to 2.23x self-attention speedup, translating into a 7.03x reduction in end-to-end inference latency, with negligible accuracy loss on long-dependency tasks</cite> — a case where query-level (not just head- or prompt-level) dynamism pays off disproportionately in latency.

---

## D. Training-Free, Inference-Time KV-Cache Sparsity (no retraining; sparsify a frozen model's cache)

### 12. StreamingLLM / Attention Sinks (Xiao, Tian, Chen, Han, Lewis, 2023)
- **Published:** Sep 29, 2023, arXiv:2309.17453 (ICLR 2024)
- **Summary:** Observes that the first few tokens of a sequence receive disproportionate attention regardless of content ("attention sinks") and keeps only those sink tokens plus a sliding window of recent tokens in the KV cache, discarding everything else — enabling theoretically unbounded streaming generation.
- **Distinction:** Unlike H2O/Quest below, importance here is *positional*, not content- or query-dependent — sink status is a structural property of early-sequence positions, discovered rather than designed, and the policy needs no scoring computation at all.
- **Results/impact:** <cite index="52-1">Preserving the first few "sink" tokens alongside a recency window prevents the severe accuracy collapse seen when those tokens are dropped, while StreamingLLM remains fast and hardware-friendly for arbitrarily long streams</cite> — establishes attention sinks as a load-bearing structural fact about trained transformers, not just an implementation trick.

### 13. H2O — Heavy-Hitter Oracle (Zhang, Sheng, Zhou et al., NeurIPS 2023)
- **Published:** Jun 24, 2023, arXiv:2306.14048
- **Summary:** Finds that a small subset of tokens ("Heavy Hitters") accumulate the majority of attention score mass across generation steps, and formulates KV-cache eviction as a dynamic submodular optimization problem, keeping only Heavy Hitters plus a small recency window.
- **Distinction:** Where StreamingLLM's importance signal is purely positional, H2O's is *accumulated attention score* — a content/history-based signal recomputed as generation proceeds, giving a data-driven rather than structural eviction rule.
- **Results/impact:** <cite index="51-1">LLM attention matrices are observed to be over 95% sparse at inference time across a wide range of pretrained models, meaning roughly 5% of the KV cache suffices to reproduce the same output token at each step</cite>, and <cite index="53-1">retaining just 20% of the KV cache as Heavy Hitters increases throughput by up to 29x over Hugging Face Accelerate and reduces latency by up to 1.9x</cite> on OPT-6.7B/30B — among the largest throughput multipliers of any method in this survey, achieved purely at inference time with no retraining.

### 14. SnapKV (Li et al., 2024)
- **Published:** 2024, arXiv:2404.14469
- **Summary:** Compresses the KV cache by clustering important tokens identified from attention patterns observed in a local "observation window" near the end of the prompt, then pooling to select a compact retained set — aimed specifically at the prefill-to-decode transition.
- **Distinction:** Whereas H2O accumulates importance scores continuously throughout generation, SnapKV makes its compression decision once, from a bounded observation window at the *end* of the prompt, which is cheaper and better suited to one-shot long-prompt compression before decoding begins.
- **Results/impact:** Positioned as a standard modern baseline alongside H2O and StreamingLLM in later KV-cache-compression papers; <cite index="50-1">unlike H2O, SnapKV captures attention signals from a localized window and applies a more nuanced clustering algorithm including a pooling step</cite>, and is consistently reported to match or exceed H2O's accuracy-at-budget tradeoff on needle-in-a-haystack-style long-context retrieval tests.

### 15. SparQ Attention (Ribar, Chelombiev, Hudlass-Galley et al., Graphcore, ICML 2024)
- **Published:** 2024 (ICML 2024)
- **Summary:** Reduces the memory-*bandwidth* cost of attention (rather than raw FLOPs) by approximating which keys matter using only a subset of each key vector's components, fetching the full vectors only for the top-scoring subset, then correcting the softmax normalization for the tokens it dropped.
- **Distinction:** Targets a different bottleneck than most of this list — bandwidth to/from KV-cache memory during decoding, not attention FLOPs or KV-cache footprint — making it complementary to (not competing with) eviction-based methods like H2O/SnapKV.
- **Results/impact:** Reported in follow-on surveys as a standard training-free baseline alongside Quest for bandwidth-efficient long-context decoding; its central claim — that attention can be approximated well from partial key vectors plus a normalization correction — has been adopted as a building block in several subsequent training-free methods.

---

## E. Hardware-Aligned, Natively-Trainable Sparse Attention (sparsity designed in from pretraining, not bolted on after)

### 16. Native Sparse Attention — NSA (Yuan, Gao, Dai et al., DeepSeek, 2025)
- **Published:** Feb 16, 2025, arXiv:2502.11089 (ACL 2025 Best Paper)
- **Summary:** A **dynamic hierarchical** sparse mechanism combining three parallel branches per query — coarse-grained compressed-token summaries for global context, fine-grained selected tokens for local precision, and a sliding window — with an arithmetic-intensity-balanced kernel design so the theoretical FLOP reduction actually converts into hardware speedup, and trained end-to-end from scratch rather than applied post-hoc.
- **Distinction:** Explicitly targets the failure mode that undermines many earlier "sparse at inference only" methods: <cite index="19-1">many prior sparse-attention approaches fall short of delivering proportional speedups on modern hardware because irregular memory access patterns and token-selection overhead erase the theoretical gains, and most approaches only speed up inference while still relying on fully dense attention during training</cite>, meaning models never actually learn to operate under sparsity. NSA is trained sparse from the start, closing that gap.
- **Results/impact:** <cite index="21-1">Experiments show NSA pretrained from scratch maintains or exceeds Full Attention model performance across general benchmarks, long-context tasks, and instruction-based reasoning</cite>, while <cite index="17-1">achieving up to 9.0x forward-pass and 6.0x backward-pass speedup at 64K context length during training, and up to 11.6x decoding speedup at 64K context, driven mainly by reduced memory access</cite> — one of the very few methods in this survey reporting simultaneous speed *and* quality gains (echoing MoSA's finding) at a production-relevant lab scale, and now the direct architectural basis for DeepSeek's V3.2-Exp Sparse Attention (DSA).

### 17. Mixture-of-Block Attention — MoBA (Lu et al., Moonshot AI/Kimi, 2025)
- **Published:** 2025, arXiv:2502.13189
- **Summary:** Partitions the sequence into blocks and applies MoE-style gating so each query dynamically routes to (attends only within) a subset of blocks, letting the model learn — rather than hand-specify — the block-sparsity boundary between long-range and local attention.
- **Distinction:** Sits between NSA's fixed three-branch hierarchy and SeerAttention's single learnable gate: MoBA's routing granularity is the *block*, chosen per query via expert-choice-style routing, giving it flexibility closer to full learned sparsity while retaining block-sparse kernels' hardware efficiency.
- **Results/impact:** Cited in contemporaneous surveys as part of the "Mixture of Attention" family <cite index="71-1">alongside MoA, SwitchHead, and MoH — a shift toward dynamic, head- and block-level specialization that brings MoE-style routing into the self-attention mechanism itself</cite>, and has since been adopted as a production long-context mechanism at Moonshot AI.

---

## Cross-Cutting Comparison

| # | Name | Year | Sparsity decided by... | Trained-in or post-hoc | Headline result |
|---|------|------|------------------------|--------------------------|------------------|
| 1 | Sparse Transformer | 2019 | Fixed strided/local pattern | Trained-in | O(n√n) complexity; new SOTA density modeling on Enwik8/CIFAR-10/ImageNet-64 |
| 2 | Longformer | 2020 | Fixed window + global tokens | Trained-in (or fine-tuned) | 94.8% acc. Hyperpartisan @ 4096 tokens; strong LRA baseline |
| 3 | BigBird | 2020 | Window + global + random | Trained-in | Best average score across Long Range Arena's 5 tasks |
| 4 | Routing Transformer | 2020 | Online k-means clustering | Trained-in | Strong clustering baseline; later beaten by ~0.6% acc / 18% latency |
| 5 | Sinkhorn Transformer | 2020 | Learned block sorting | Trained-in | Mid-pack on LRA; less dominant than BigBird/Longformer |
| 6 | Reformer | 2020 | LSH bucketing | Trained-in | 64K-length training; faster/leaner but no accuracy gain |
| 7 | SMYRF | 2020 | Asymmetric clustering (LSH successor) | Drop-in / fine-tune | Outperforms Reformer's LSH schemes head-to-head on IMDB |
| 8 | SeerAttention | 2024 | Learned MoE-style block gate | Self-distilled onto pretrained model | Near-lossless AIME @ 4K budget; composable to 2.47x speedup |
| 9 | MoSA | 2025 | Expert-choice token routing per head | Trained-in | Only method to *beat* dense perplexity, up to 27% at matched FLOPs |
| 10 | MInference 1.0 | 2024 | Offline per-head shape + online mask | Post-hoc, training-free | 10x prefill speedup, 30min→3min per 1M-token prompt |
| 11 | Quest | 2024 | Per-query page relevance | Post-hoc, training-free | 2.23x attention / 7.03x end-to-end latency speedup |
| 12 | StreamingLLM | 2023 | Positional "attention sinks" | Post-hoc, training-free | Unbounded streaming generation, no accuracy collapse |
| 13 | H2O | 2023 | Accumulated attention score | Post-hoc, training-free | Up to 29x throughput, 1.9x lower latency at 20% cache |
| 14 | SnapKV | 2024 | Observation-window clustering | Post-hoc, training-free | Matches/exceeds H2O accuracy-at-budget on long-context QA |
| 15 | SparQ Attention | 2024 | Partial-key-vector bandwidth approx. | Post-hoc, training-free | Reduces KV memory-bandwidth cost, not just FLOPs/footprint |
| 16 | Native Sparse Attention (NSA) | 2025 | Hierarchical compress+select+window | Natively trained from scratch | 9–11.6x train/decode speedup at 64K, matches/beats dense quality |
| 17 | MoBA | 2025 | MoE-style block routing per query | Natively trained | Production long-context mechanism at Moonshot AI/Kimi |

## Overall Impact Assessment
- **Largest, most reproduced production impact:** NSA (ACL 2025 Best Paper, now underlying DeepSeek's DSA) and MInference 1.0 (10x prefill speedup adopted widely for million-token contexts) — both combine large measured speedups with maintained or improved quality at frontier-lab scale.
- **Most surprising/counterintuitive result:** MoSA and NSA are the two methods in this survey that report sparsity *improving* quality (up to 27% perplexity and matching/exceeding dense benchmarks respectively) rather than trading accuracy for speed — evidence that forcing specialization or hierarchy can act as a useful inductive bias, not just an approximation.
- **Largest raw efficiency multiplier:** H2O's up-to-29x throughput improvement over other serving systems at only 20% KV-cache budget, though this is a systems-throughput comparison against other serving stacks rather than an apples-to-apples attention-FLOPs count like MInference's or NSA's.
- **Weakest/most caveated result:** Reformer — genuine memory and speed gains from LSH + reversible layers, but explicitly no accuracy improvement over full attention, positioning it as a pure efficiency trade rather than a "free lunch."
- **Clearest fault line in the field:** training-free, post-hoc KV-cache methods (StreamingLLM, H2O, SnapKV, Quest, SparQ) that sparsify an already-trained dense model, versus natively-trainable methods (NSA, MoBA, MoSA, Sparse Transformer) that bake sparsity into pretraining itself — the latter group is where "sparsity that improves rather than merely preserves quality" has so far been found.

## Key Takeaway
"Sparse attention" spans at least five distinct engineering answers to the same question — *which query-key pairs are actually worth computing?* — (1) fixed, hand-designed positional patterns decided once at architecture-design time (Sparse Transformer, Longformer, BigBird), (2) similarity/clustering-based grouping that reorganizes tokens before attending (Reformer, SMYRF, Routing Transformer, Sinkhorn Transformer), (3) learned, content-based routing that a small trained gate or expert-choice mechanism decides per input (SeerAttention, MoSA, MoBA), (4) training-free eviction or bandwidth tricks applied to an already-trained model's KV cache at inference time (StreamingLLM, H2O, SnapKV, Quest, SparQ), and (5) hardware-co-designed sparsity trained end-to-end from scratch so theoretical FLOP savings survive contact with real GPUs (NSA). The field's trajectory runs roughly from (1)→(4)→(5): early work fixed the pattern by hand, the 2023–2024 wave made sparsity a training-free retrofit for serving efficiency, and the 2025 wave (NSA, MoSA, MoBA) pushes sparsity back into pretraining itself, which is also where the rare "sparsity that beats dense" results have started to appear.

## References
- Child, R., Gray, S., Radford, A., Sutskever, I. "Generating Long Sequences with Sparse Transformers." arXiv:1904.10509 (2019).
- Beltagy, I., Peters, M. E., Cohan, A. "Longformer: The Long-Document Transformer." arXiv:2004.05150 (2020).
- Zaheer, M., Guruganesh, G., Dubey, A. et al. "Big Bird: Transformers for Longer Sequences." NeurIPS 2020, arXiv:2007.14062.
- Roy, A., Saffar, M., Vaswani, A., Grangier, D. "Efficient Content-Based Sparse Attention with Routing Transformers." TACL 2021, arXiv:2003.05997.
- Tay, Y. et al. "Sparse Sinkhorn Attention." ICML 2020 / "Synthesizer: Rethinking Self-Attention for Transformer Models." referenced via Long Range Arena, arXiv:2011.04006.
- Kitaev, N., Kaiser, Ł., Levskaya, A. "Reformer: The Efficient Transformer." ICLR 2020, arXiv:2001.04451.
- Daras, G., Kitaev, N., Odena, A., Dimakis, A. G. "SMYRF: Efficient Attention using Asymmetric Clustering." arXiv:2010.05315 (2020).
- Gao, Y. et al. "SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs." arXiv:2410.13276 (2024); "SeerAttention-R: Sparse Attention Adaptation for Long Reasoning" (2025).
- Piękos, P., Csordás, R., Schmidhuber, J. "Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing." arXiv:2505.00315 (2025).
- Jiang, H., Li, Y., Zhang, C. et al. "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention." NeurIPS 2024, arXiv:2407.02490.
- Tang, J., Zhao, Y., Zhu, K. et al. "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference." arXiv:2406.10774 (2024).
- Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. "Efficient Streaming Language Models with Attention Sinks." ICLR 2024, arXiv:2309.17453.
- Zhang, Z., Sheng, Y., Zhou, T. et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." NeurIPS 2023, arXiv:2306.14048.
- Li, Y. et al. "SnapKV: LLM Knows What You Are Looking for Before Generation." arXiv:2404.14469 (2024).
- Ribar, L., Chelombiev, I., Hudlass-Galley, L. et al. "SparQ Attention: Bandwidth-Efficient LLM Inference." ICML 2024.
- Yuan, J., Gao, H., Dai, D. et al. "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention." ACL 2025 (Best Paper), arXiv:2502.11089.
- Lu, E. et al. "MoBA: Mixture of Block Attention for Long-Context LLMs." arXiv:2502.13189 (2025).