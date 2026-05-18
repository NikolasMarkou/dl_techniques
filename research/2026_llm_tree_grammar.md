# StructMem: short review

The proposal (tree transformer → VQ-VAE → Wikipedia memory → LLM cross-attention) has four killing problems. Listed in order of severity.

## 1. The novelty claim is false

**MEMORY-VQ** (Zemlyanskiy et al., NAACL 2024, arXiv 2308.14903) already compresses retrieval memory with VQ-VAE at **16× with KILT-parity**. The proposal's headline result (E5.3: "4× smaller than kNN-LM at parity") is a weaker version of an already-published result. Comparing against kNN-LM (2020) instead of MEMORY-VQ is the wrong baseline.

## 2. The objective optimizes for the wrong thing

Reconstruction CE over surface tokens makes codes preserve word-order and lemmas. That is anti-correlated with paraphrase-invariance. The proposal admits this in §13 as an "Open Problem" — but the rest of the architecture assumes the bottleneck does what reconstruction loss won't make it do.

If you want abstract codes, the objective must be abstract. Decode predicate-argument tuples, not surface text. Or add a paraphrase contrastive loss as a binding term, not as a footnote.

## 3. The repo's `tree_transformer` is not what the proposal calls a "tree transformer"

`src/dl_techniques/models/tree_transformer/` implements Wang et al.'s **unsupervised soft constituency** (Group Attention). The proposal specifies **supervised biaffine dependency parsing** (Dozat & Manning). Different architecture, different training data, different outputs. Reusing the name will mislead the implementer.

## 4. The structural probe is hand-waved

§8 Stage 5: `q_struct = parse(h)` — parse the LLM's mid-layer hidden state to extract (entity_type, dep_relation, head_pos). The LLM is frozen and was never trained to expose dependency structure linearly. Probing UAS tops out around 85 with strong probes. This is not a parser-grade signal, and the gate (G3) depends on it.

Fix: run the input through the parser in parallel; use *its* features for the structural query. Don't probe the LLM.

## Bugs in the pseudocode

- **§6 diversity loss sign**: `L = … - γ H(codebook)` *rewards* collapse. Should be `+ γ (H_target − H)²` or similar.
- **§7 structural_hash**: hashes codes *and* deprels. Codes are supposed to encode deprels. The hash will only collide when both match — defeating dedup of true paraphrases.
- **§4 arc-constrained attention** is undefined for step 0 (no parse yet). Needs a warm-up curriculum.
- **§7 dedup rate 15-25%**: optimistic. Wikipedia is paraphrase-sparse within itself. Expect 8-12% on full prose.
- **§5 span-NER constrained to parse subtrees**: hard recall ceiling at LAS. If LAS=90, ~10% of entities in misattached subtrees are unreachable. Needs a fallback head.

## What survives if you fix the four killers

A retrieval-augmented LM with parse-aware codes. That is a sensible delta paper *against MEMORY-VQ*. Not a paradigm shift. Numbers to beat:
- **G_PPL**: beat MEMORY-VQ at matched bytes/entry on entity-dense Wikipedia.
- **G_UPDATE**: beat ROME on single-fact insertion.
- **G_INSPECT**: a human can read a retrieved entry and say what it means.

If you can hit those three, you have a paper. If not, you have eight weeks of training scripts.

## Sources

- [MEMORY-VQ (NAACL 2024)](https://arxiv.org/abs/2308.14903)
- [COCONUT (Meta, Dec 2024)](https://arxiv.org/abs/2412.06769)
- [RETRO (DeepMind 2021)](https://arxiv.org/abs/2112.04426)
- [Rotation-trick VQ (Fifty et al. 2024)](https://arxiv.org/abs/2410.06424)
- [ROME](https://arxiv.org/abs/2202.05262), [MEMIT](https://memit.baulab.info/)
- Repo Tree Transformer: `src/dl_techniques/models/tree_transformer/` (Wang-soft-constituency, NOT biaffine).
