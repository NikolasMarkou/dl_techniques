# Transformer Output Token Handling Techniques (2020-2025)

## Introduction

### What is Token Handling?

Transformers process sequences through self-attention mechanisms, operating on token embeddings that represent input data (text, images, video). While attention layers excel at modeling relationships between tokens, a critical challenge remains: **how do we extract meaningful task-specific outputs from variable-length token sequences?**

**Token handling** encompasses the strategies and mechanisms transformers employ to:
1. **Aggregate** distributed information across tokens into fixed-size representations
2. **Select** or learn specialized tokens that capture task-relevant features  
3. **Generate** new tokens for sequence-to-sequence tasks
4. **Structure** outputs for complex predictions (bounding boxes, masks, multiple objects)

### The Problem Token Handling Solves

Original transformer architectures process all input tokens uniformly, but downstream tasks require different output formats:

- **Classification tasks** need a single vector representing the entire input
- **Object detection** requires multiple structured outputs (boxes, classes, scores)
- **Generation tasks** must produce variable-length sequences efficiently
- **Embedding tasks** need representations optimized for similarity comparison
- **Dense prediction** tasks (segmentation, depth) require pixel-level outputs

Without effective token handling, transformers face several issues:
- **Representation collapse**: All tokens converge to similar values, losing discriminative power
- **Computational inefficiency**: Processing hundreds of tokens when only aggregate features are needed
- **Information bottlenecks**: Single [CLS] token may not capture all relevant information
- **Anisotropic embeddings**: Sentence embeddings cluster in narrow cones, hindering similarity tasks

### Key Innovation Categories (2020-2025)

The 2020-2025 period saw transformative advances in token handling:

**1. Learnable Output Queries** (2020-2023)
DETR introduced learnable object queries for detection, spawning a family of query-based methods. These specialized tokens attend to input features via cross-attention, each learning to detect specific objects or predict structured outputs.

**2. Register Tokens** (2023-2024)  
Vision Transformers were found to develop "artifact" tokens with anomalously high norms. Register tokens—extra learnable tokens that absorb global information—eliminate these artifacts and improve dense prediction performance by 2-4%.

**3. Parameter-Efficient Soft Prompts** (2021-2022)
Prompt tuning and prefix tuning prepend learnable continuous vectors to inputs, achieving fine-tuning-comparable performance with <0.1% tuned parameters, democratizing model adaptation.

**4. Contrastive Pooling Methods** (2021-2023)
SimCSE and related methods apply contrastive learning to pooled representations, improving sentence embedding quality by 4-8% on semantic similarity benchmarks by addressing anisotropy.

**5. Multimodal Bridge Tokens** (2021-2023)
Q-Former and Perceiver Resampler use learned queries to compress visual features into fixed-size representations aligned with language model embeddings, enabling efficient vision-language fusion.

**6. Efficient Generation Mechanisms** (2021-2023)
Speculative decoding and non-autoregressive transformers introduce token-level parallelism, achieving 2-5× speedups in generation tasks.

### Impact and Importance

Token handling innovations have enabled:
- **Higher accuracy**: DETR family achieves 42-60+ mAP on COCO; SimCSE reaches 81.6% on STS benchmarks
- **Parameter efficiency**: Prompt tuning matches full fine-tuning with 0.01-3% parameters
- **Computational efficiency**: Speculative decoding provides 2-3× speedup; register tokens improve performance without added cost
- **Broader applicability**: Query-based methods unify detection, segmentation, and multimodal tasks in single frameworks

### Organization

This reference catalogs **67 distinct techniques** with full citations, years, and performance metrics organized by:
- **Method type**: Special tokens, pooling, prompts, generation, multimodal
- **Architecture**: Encoder-only, decoder-only, encoder-decoder, cross-modal
- **Task category**: Classification, detection, segmentation, generation, embeddings

---

## Special Output Tokens and Learned Queries

### CLS Token and Variants

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **Register Tokens** | 2023 | Additional learnable tokens (typically 4) appended to ViT input to absorb global information, eliminating high-norm artifact tokens in background areas | +2-4% OOD accuracy; improved dense prediction tasks | Darcet, Oquab, Mairal, Bojanowski. "Vision Transformers Need Registers." ICLR 2024 | Encoder-only (ViT) |
| **Jumbo CLS Token** | 2025 | Wider CLS token split to match patch width before attention, processed with dedicated wider FFN shared across layers | Faster inference; maintained accuracy | Fuller, Yassin, Kyrollos, Shelhamer, Green. "Simpler Fast Vision Transformers with a Jumbo CLS Token." arXiv 2025 | Encoder-only (ViT) |
| **Class Attention (CaiT)** | 2021 | Two-stage learning: self-attention on patches only, then multi-head class attention stage where CLS aggregates patch information | Improved ImageNet accuracy | Touvron et al. "Going Deeper with Image Transformers." ICCV 2021 | Encoder-only (ViT) |
| **Separate CLS Normalization** | 2023 | Batch normalization applied specifically to CLS token instead of layer normalization, improving embedding uniformity | Improved self-supervised learning | Chen et al. "On Separate Normalization in Self-supervised Transformers." arXiv 2023 | Encoder-only (ViT) |

### Cross-Attention Query Tokens

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **DETR Object Queries** | 2020 | Fixed set of 100 learned embeddings serving as object slots; each query predicts one object via Hungarian matching | 42 AP on COCO (ResNet-50 backbone) | Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko. "End-to-End Object Detection with Transformers." ECCV 2020 | Encoder-decoder |
| **Deformable DETR Reference Points** | 2021 | Object queries attend to sparse sampling points around learnable reference points; multi-scale deformable attention | 46.3 AP on COCO; 10× faster convergence | Zhu, Su, Lu, Li, Wang, Dai. "Deformable DETR: Deformable Transformers for End-to-End Object Detection." ICLR 2021 | Encoder-decoder |
| **DAB-DETR Dynamic Anchor Boxes** | 2022 | 4D anchor box coordinates (x,y,w,h) as queries with layer-by-layer refinement providing position and scale priors | Improved convergence speed | Liu et al. "DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR." ICLR 2022 | Encoder-decoder |
| **DQ-DETR Dynamic Queries** | 2024 | Dynamically adjusts query count based on image content using counting module; position-enhanced queries from density maps | Better tiny object detection | "DQ-DETR: DETR with Dynamic Query for Tiny Object Detection." arXiv 2024 | Encoder-decoder |
| **Query2Label** | 2021 | Label embeddings as queries in transformer decoder to pool class-related features via cross-attention for multi-label classification | Improved multi-label classification | Liu, Zhang, Yang, Su, Zhu. "Query2Label: A Simple Transformer Way to Multi-Label Classification." BMVC 2021 | Encoder-decoder |
| **Perceiver Latent Queries** | 2021 | Fixed-size latent array (256-512) produces queries for cross-attention with high-dimensional inputs; O(MN) complexity for any modality | Handles arbitrary modalities efficiently | Jaegle et al. "Perceiver: General Perception with Iterative Attention." ICML 2021 | Cross-attention encoder |
| **Perceiver IO Decode Queries** | 2022 | Task-specific output queries of arbitrary shape attend to latent representations for structured outputs | Flexible output structures | Jaegle et al. "Perceiver IO: A General Architecture for Structured Inputs & Outputs." ICLR 2022 | Encoder-decoder |
| **Q-Former (BLIP-2)** | 2023 | 32 learnable query embeddings extract visual features from frozen image encoder via cross-attention, project to LLM input space | Efficient vision-LLM alignment | Li, Li, Savarese, Hoi. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023 | Vision-LLM bridge |
| **Perceiver Resampler (Flamingo)** | 2022 | Learned latent queries resample variable-length visual features into fixed-number embeddings for vision-language fusion | Few-shot vision-language learning | Alayrac et al. "Flamingo: a Visual Language Model for Few-Shot Learning." NeurIPS 2022 | Encoder-decoder |

### Segmentation Output Tokens

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **Mask2Former Queries** | 2022 | Learnable queries predict masks via masked cross-attention constrained to predicted foreground regions; unified instance/semantic/panoptic segmentation | Universal segmentation framework | Cheng, Misra, Schwing, Kirillov, Girdhar. "Masked-attention Mask Transformer for Universal Image Segmentation." CVPR 2022 | Encoder-decoder |
| **SAM Output Tokens** | 2023 | Output token embeddings concatenated with prompt tokens pass through bidirectional cross-attention; each token generates mask via MLP + dot product | Promptable segmentation | Kirillov et al. "Segment Anything." ICCV 2023 | Encoder-decoder |
| **HQ-SAM High-Quality Token** | 2023 | Additional learnable token in SAM's mask decoder for high-quality masks; fused with early and final ViT features | Higher quality masks | Ke, Ye, Danelljan et al. "Segment Anything in High Quality." NeurIPS 2023 | SAM extension |
| **SAM 2 Object Pointers** | 2024 | Lightweight vectors for high-level semantic representation derived from mask decoder output tokens; memory bank stores per-frame memories | Video segmentation and tracking | "SAM 2: Segment Anything in Images and Videos." Meta 2024 | Streaming memory transformer |

---

## Soft Prompts and Prefix Tuning

### Prefix-Based Techniques

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **Prefix-Tuning** | 2021 | Continuous task-specific vectors prepended to inputs at every layer; freezes LM parameters while optimizing ~0.1% prefix parameters | Matches fine-tuning with 0.1% parameters | Li, Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation." ACL 2021 | Decoder-only & encoder-decoder |
| **Prompt Tuning** | 2021 | Soft prompt embeddings prepended only to input layer with frozen LM; matches fine-tuning at 10B+ parameters with <0.01% tuned parameters | Matches fine-tuning at scale with <0.01% params | Lester, Al-Rfou, Constant. "The Power of Scale for Parameter-Efficient Prompt Tuning." EMNLP 2021 | Encoder-decoder (T5) |
| **P-Tuning** | 2021 | Trainable continuous embeddings interleaved with discrete prompts; LSTM/MLP encoders stabilize training | Improves GPT understanding tasks | Liu et al. "GPT Understands, Too." arXiv 2021 | Decoder-only & encoder-only |
| **P-Tuning v2** | 2022 | Deep prompt tuning at every transformer layer; achieves fine-tuning-comparable performance with 0.1%-3% tuned parameters across all scales | Matches fine-tuning with 0.1-3% params universally | Liu et al. "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally." ACL 2022 | All architectures |
| **PTR (Prompt Tuning with Rules)** | 2022 | Encodes task prior knowledge into rules; designs sub-prompts for each rule for many-class classification | Improved few-shot classification | Han, Zhao, Ding, Liu, Sun. "PTR: Prompt Tuning with Rules for Text Classification." AI Open 2022 | Encoder-only |
| **Knowledgeable Prompt-tuning (KPT)** | 2022 | Incorporates external knowledge into prompt verbalizers by expanding label words using knowledge bases | Knowledge-enhanced classification | Hu et al. "Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification." ACL 2022 | Encoder-only |
| **T5 Encoder-Decoder Soft Prompts** | 2022 | Soft prompts at both encoder AND decoder levels in T5 for improved controlled text generation | Better controllable generation | Senadeera, Soysa. "Controlled Text Generation using T5 based Encoder-Decoder Soft Prompt Tuning." arXiv 2022 | Encoder-decoder |

### Memory and Adapter Tokens

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **Memory Transformer** | 2020 | Additional trainable memory tokens appended to input interact through self-attention; precursor to register tokens | Improved translation performance | Burtsev et al. "Memory Transformer." arXiv 2020 | Encoder-decoder |
| **Adapter Tuning** | 2019-2021 | Small bottleneck modules inserted between transformer layers with task-specific parameters | Parameter-efficient transfer learning | Houlsby et al. "Parameter-Efficient Transfer Learning for NLP." ICML 2019; Pfeiffer et al. AdapterHub 2020-2021 | All architectures |
| **Mix-and-Match Adapters** | 2022 | Combines adapters with prefix tuning in unified configurations | Unified parameter-efficient methods | He et al. "Towards a Unified View of Parameter-Efficient Transfer Learning." ICLR 2022 | All architectures |

---

## Pooling and Aggregation Techniques

### Mean and Max Pooling

| Technique | Year | Description | Performance Improvement | Citation | Tasks |
|-----------|------|-------------|------------------------|----------|-------|
| **Sentence-BERT Mean Pooling** | 2019 | Averages all token embeddings from last layer for fixed-size sentence vectors | Enables efficient similarity search | Reimers, Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019 | Similarity, retrieval |
| **E5 Average Pooling** | 2022 | Masked attention-weighted average pooling with query/passage prefixes for task adaptation | SOTA embedding performance | Wang et al. "Text Embeddings by Weakly-Supervised Contrastive Pre-training." arXiv 2022 | Retrieval, similarity |
| **First-Last Average Pooling** | 2020 | Averages embeddings from both first and last transformer layers to capture diverse features | Better feature diversity | Li et al. "On the Sentence Embeddings from Pre-trained Language Models." EMNLP 2020 | STS |
| **Token Pooling (ViT)** | 2023 | Max pooling to downsample visual tokens, reducing computation while preserving features | Reduced computation | Marin et al. "Token Pooling in Vision Transformers for Image Classification." WACV 2023 | Image classification |
| **Mean-Max Concatenate** | 2020+ | Concatenates mean and max pooling outputs for both average and salient features | Complementary features | Various implementations; sentence-transformers | Classification, regression |
| **GAP vs CLS (ViT)** | 2022 | Global Average Pooling of patch tokens outperforms CLS by ~1.8% for image classification | +1.8% over CLS token | Beyer, Zhai, Kolesnikov. "Better plain ViT baselines for ImageNet-1k." arXiv 2022 | Image classification |

### Attention-Weighted Pooling

| Technique | Year | Description | Performance Improvement | Citation | Tasks |
|-----------|------|-------------|------------------------|----------|-------|
| **Pooling by Multihead Attention (PMA)** | 2019 | Aggregates features using learned seed vectors with multihead attention; permutation-invariant | Permutation-invariant aggregation | Lee et al. "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks." ICML 2019 | Set problems, clustering |
| **Attentive Statistics Pooling** | 2018-2020 | Learnable attention weights dynamically weight tokens before computing mean and std | Improved speaker embeddings | Okabe et al. "Attentive Statistics Pooling for Deep Speaker Embedding." 2018, extended 2020+ | Speaker verification |
| **Multi-Query Multi-Head Attention (MQMHA)** | 2021 | Attention over whole feature + attention over split parts for diversified information | Diversified speaker features | Zhao et al. "Multi-query multi-head attention pooling and Inter-topK penalty for speaker verification." arXiv 2021 | Speaker verification |

### Hierarchical Pooling

| Technique | Year | Description | Performance Improvement | Citation | Tasks |
|-----------|------|-------------|------------------------|----------|-------|
| **Hi-Transformer** | 2021 | Models documents hierarchically; sentence representations then document representations with hierarchical attentive pooling | Efficient long document modeling | Wu et al. "Hi-Transformer: Hierarchical Interactive Transformer for Efficient and Effective Long Document Modeling." ACL 2021 | Document classification |
| **Hierarchical Visual Transformer (HVT)** | 2021 | Progressive pooling of visual tokens across stages using max/average pooling layers | Scalable vision transformers | Pan et al. "Scalable Vision Transformers with Hierarchical Pooling." ICCV 2021 | Image classification |
| **Hourglass Hierarchical Transformer** | 2022 | U-Net-like shortening (average/linear/attention pooling) and upsampling for long sequences | More efficient language modeling | Nawrot et al. "Hierarchical Transformers Are More Efficient Language Models." ACL 2022 | Language modeling |
| **PoolingFormer** | 2021 | Two-level attention: sliding window + pooled global attention for long documents | Long document understanding | Zhang et al. "Poolingformer: Long Document Modeling with Pooling Attention." ICML 2021 | QA, document understanding |
| **BoM-Pooling (Bag-of-Mer)** | 2025 | Locality-aware hierarchical pooling with windowed average + attention pooling for protein sequences | Enhanced protein prediction | "Locality-aware pooling enhances protein language model performance." Bioinformatics 2025 | Protein prediction |

### Graph Pooling with Transformers

| Technique | Year | Description | Performance Improvement | Citation | Tasks |
|-----------|------|-------------|------------------------|----------|-------|
| **GMAPS** | 2022 | Multihead attention clusters nodes and constructs coarsened graph with self-supervised learning | Improved graph classification | "Graph Multihead Attention Pooling with Self-Supervised Learning." Entropy 2022 | Graph classification |
| **Gapformer** | 2023 | Graph pooling with transformer; computes attention with pooling nodes to reduce quadratic complexity | Efficient node classification | "Gapformer: Graph Transformer with Graph Pooling for Node Classification." IJCAI 2023 | Node classification |

---

## Contrastive Learning Embedding Methods

| Technique | Year | Description | Performance Improvement | Citation | Tasks |
|-----------|------|-------------|------------------------|----------|-------|
| **SimCSE** | 2021 | CLS token pooling with contrastive learning using dropout as data augmentation; also supports mean pooling | Unsup: 76.3% STS (+4.2%); Sup: 81.6% (+2.2%) | Gao, Yao, Chen. "SimCSE: Simple Contrastive Learning of Sentence Embeddings." EMNLP 2021 | STS, retrieval |
| **DiffCSE** | 2022 | Extends SimCSE with equivariant contrastive learning; learns embeddings sensitive to MLM-based edits | Improved discriminative representations | Chuang et al. "DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings." NAACL 2022 | STS |
| **ConSERT** | 2021 | Average pooling of last two layers with contrastive learning and data augmentation (token deletion, shuffling) | Robust transfer learning | Yan et al. "ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer." ACL 2021 | STS, transfer |
| **SimCSE++** | 2023 | Improves SimCSE with off-dropout sampling and dimension-wise contrastive learning to address rank bottleneck | Further STS improvements | "SimCSE++: Improving Contrastive Learning for Sentence Embeddings." EMNLP 2023 | STS |
| **BERT-Flow** | 2020 | Normalizing flows transform anisotropic BERT embeddings to isotropic Gaussian distribution | Addresses anisotropy | Li et al. "On the Sentence Embeddings from Pre-trained Language Models." EMNLP 2020 | STS |
| **BERT-Whitening** | 2021 | Whitening transformation (mean=0, covariance=identity) reduces dimensionality and improves isotropy | Faster retrieval; better semantics | Su et al. "Whitening Sentence Representations for Better Semantics and Faster Retrieval." arXiv 2021 | Similarity, retrieval |

---

## Generation Task Techniques

### Control Tokens and Instruction Tuning

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **CTRL Control Codes** | 2019-2020 | Domain, style, date, entity control codes prepended to input for controllable generation and source attribution | Controllable generation | Keskar et al. "CTRL: A Conditional Transformer Language Model for Controllable Generation." arXiv 2019, adopted 2020+ | Decoder-only |
| **FLAN Instruction Tuning** | 2022 | Fine-tuning on 60+ tasks via instruction templates; special EOS tokens separate inputs from targets | Zero-shot task generalization | Wei et al. "Finetuned Language Models Are Zero-Shot Learners." ICLR 2022 | Decoder-only |
| **FLAN-T5 / Flan Collection** | 2022-2023 | Scales to 1800+ tasks with mixed prompt settings (zero-shot, few-shot, chain-of-thought) | Improved instruction following | Chung et al. "Scaling Instruction-Finetuned Language Models." arXiv 2022; Longpre et al. "The Flan Collection." arXiv 2023 | Encoder-decoder |
| **T5 Text-to-Text** | 2020 | Task-specific prefix tokens ("translate:", "summarize:"); sentinel tokens for span corruption pre-training | Unified text-to-text framework | Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR 2020 | Encoder-decoder |

### Non-Autoregressive Generation

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **Non-Autoregressive Transformer (NAT)** | 2018-2020 | Generates all output tokens simultaneously; fertility-guided copying with length prediction tokens | Parallel generation | Gu et al. "Non-Autoregressive Neural Machine Translation." ICLR 2018, extended 2020+ | Encoder-decoder |
| **Glancing Transformer (GLAT)** | 2021 | Glancing Language Model samples reference tokens as hints during training for single-pass parallel generation | Improved NAT quality | Qian et al. "Glancing Transformer for Non-Autoregressive Neural Machine Translation." ACL 2021 | Encoder-decoder |
| **Levenshtein Transformer** | 2019-2020 | Insertion and deletion operations as atomic generation; generates placeholder tokens then fills them | Flexible parallel generation | Gu, Wang. "Levenshtein Transformer." NeurIPS 2019, extended 2020+ | Encoder-decoder |

### Speculative Decoding

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **Speculative Decoding (Google)** | 2023 | Smaller drafter model generates token prefixes verified in parallel by larger model; 2-3× speedup with exact distribution | 2-3× inference speedup | Leviathan, Kalman, Matias. "Fast Inference from Transformers via Speculative Decoding." ICML 2023 | All architectures |
| **Speculative Sampling (DeepMind)** | 2023 | Speculative generation with rejection sampling guaranteeing identical outputs to standard decoding | Faster decoding; exact outputs | Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv 2023 | Decoder-only |
| **SpecDec** | 2023 | Optimized drafter model and verification achieving ~5× speedup for seq2seq tasks | ~5× speedup for seq2seq | Xia et al. "Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation." EMNLP Findings 2023 | Encoder-decoder |

### Advanced Decoding

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **Contrastive Decoding** | 2023 | Contrastive objective between expert and amateur LM; selects tokens maximizing likelihood difference | Higher quality generation | Li et al. "Contrastive Decoding: Open-ended Text Generation as Optimization." ACL 2023 | Decoder-only |

---

## Multimodal Output Techniques

| Technique | Year | Description | Performance Improvement | Citation | Modality |
|-----------|------|-------------|------------------------|----------|----------|
| **CLIP Dual-Encoder** | 2021 | Separate image/text encoders produce vectors in shared space; image uses attention pooling on GAP representation; text uses [EOS] token | Zero-shot image classification | Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021 | Vision-language |
| **BLIP MED Architecture** | 2022 | Unified model with [CLS], [Encode], [Decode] tokens for different functionalities (understanding vs generation) | Multi-task vision-language | Li, Li, Xiong, Hoi. "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation." ICML 2022 | Vision-language |
| **RA-CLIP** | 2023 | Retrieval augmentation module fuses retrieved knowledge with query embeddings during training and inference | Enhanced retrieval-based learning | Xie et al. "RA-CLIP: Retrieval Augmented Contrastive Language-Image Pre-training." CVPR 2023 | Vision-language |

---

## State Space Model Output Techniques

| Technique | Year | Description | Performance Improvement | Citation | Architecture |
|-----------|------|-------------|------------------------|----------|--------------|
| **Mamba Selective State Space** | 2023 | Input-dependent SSM parameters (B, C, Δ) enable content-based selective propagation; hidden state as compressed context; linear complexity | Linear-time sequence modeling | Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv 2023 | S6 blocks |
| **Mamba-2 SSD** | 2024 | Scalar-identity restriction on A matrix enables mathematical duality with attention; hardware-efficient matrix multiplication | Hardware-efficient SSM | Dao, Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." 2024 | SSD layer |

---

## Quick Reference by Architecture Type

### Encoder-only
Register Tokens, Jumbo CLS, CaiT Class Attention, P-Tuning, P-Tuning v2, SimCSE, DiffCSE, ConSERT, BERT-Flow, BERT-Whitening, Mean/Max Pooling variants, Sentence-BERT

### Decoder-only
Prefix-Tuning, P-Tuning, CTRL, FLAN, Speculative Decoding, Speculative Sampling, Contrastive Decoding, Mamba/Mamba-2

### Encoder-decoder
DETR variants, Mask2Former, SAM, Perceiver/Perceiver IO, Q-Former, T5, FLAN-T5, NAT, GLAT, Levenshtein Transformer, SpecDec, Adapter Tuning

### Cross-modal/Multimodal
CLIP, BLIP, BLIP-2, Flamingo Perceiver Resampler, RA-CLIP

---

## Quick Reference by Task Type

### Classification
CLS tokens, Register Tokens, Query2Label, P-Tuning variants, Prompt Tuning, SimCSE, pooling methods (mean, max, attention-weighted)

### Object detection
DETR Object Queries (42 AP), Deformable DETR (46.3 AP), DAB-DETR, DQ-DETR, RT-DETR (53+ AP), RF-DETR (60+ AP)

### Segmentation
Mask2Former Queries, SAM Output Tokens, HQ-SAM Token, SAM 2 Object Pointers

### Text generation
Prefix-Tuning, Prompt Tuning, CTRL, T5 text-to-text, FLAN instruction tuning, Speculative Decoding (2-5× speedup), Contrastive Decoding

### Sentence/document embeddings
Sentence-BERT pooling, E5, SimCSE (76.3-81.6% on STS), DiffCSE, ConSERT, BERT-Flow, BERT-Whitening, Hi-Transformer, PoolingFormer

### Vision-language
CLIP, BLIP, BLIP-2 Q-Former, Flamingo Perceiver Resampler, RA-CLIP

---

## Performance Benchmarks Summary

**Object Detection (COCO mAP):**
- DETR (2020): 42 AP
- Deformable DETR (2021): 46.3 AP
- RT-DETR (2024): 53+ AP  
- RF-DETR (2025): 60+ AP

**Sentence Embeddings (STS Spearman correlation):**
- Baseline BERT: ~72%
- SimCSE Unsupervised (2021): 76.3% (+4.2%)
- SimCSE Supervised (2021): 81.6% (+2.2%)

**Parameter Efficiency:**
- Prompt Tuning: <0.01% parameters matches fine-tuning at 10B+ scale
- P-Tuning v2: 0.1-3% parameters matches fine-tuning universally
- Prefix-Tuning: ~0.1% parameters for generation tasks

**Speed Improvements:**
- Speculative Decoding: 2-3× speedup
- SpecDec: ~5× speedup for seq2seq
- Register Tokens: No added computational cost, improved performance

**Vision Improvements:**
- Register Tokens: +2-4% OOD accuracy
- GAP vs CLS: +1.8% ImageNet accuracy