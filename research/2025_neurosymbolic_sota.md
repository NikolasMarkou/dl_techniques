# RESEARCH REPORT: SYMBOLS IN NEUROSYMBOLIC AI - 2024-2025 UPDATE

## EXECUTIVE SUMMARY

This comprehensive research report synthesizes cutting-edge developments (2024-2025) across 15 key areas of neurosymbolic AI. The research reveals three transformative breakthroughs: (1) **Sparse Autoencoders scaling to 34M features** in production models, enabling direct symbol extraction from neural networks; (2) **Codebook collapse problem definitively solved** through FSQ, SimVQ, and MGVQ approaches; (3) **Object-centric learning scaling to real-world data** via self-supervised feature reconstruction. These advances position neurosymbolic AI as a practical alternative to pure scaling laws, achieving 100× smaller models with comparable performance.

---

## SECTION-BY-SECTION UPDATE RECOMMENDATIONS

### 1. VECTOR QUANTIZATION METHODS - MAJOR UPDATES

**Add New Methods Section:**
- **MGVQ (2025)**: First discrete tokenizer matching continuous VAEs (ImageNet rFID 0.49 vs SD-VAE 0.91). Use nested sub-codebooks with 256^8 capacity.
- **FSQ (ICLR 2024)**: Eliminates all VQ complexity—no commitment losses, codebook reseeding, or entropy penalties. Achieves 100% utilization by design through implicit codebook [8,5,5,5] product quantization.
- **SimVQ (2024)**: Reparameterizes codebook via linear transformation over latent basis, achieving near 100% utilization at any scale.

**Update Codebook Collapse Solutions:**
- Add FSQ as simplest solution (eliminate problem by design)
- Add SimVQ linear space optimization (theoretical solution)
- Add Rotation Trick (ICLR 2025) for gradient propagation
- Update benchmarks: MGVQ now SOTA across metrics

**Practical Implementation:**
- Recommend FSQ for new projects (simplicity)
- Recommend MGVQ for best quality
- Update code repositories: vector-quantize-pytorch, FSQ implementations

**Key Citation:** Jia et al. (2025). "MGVQ." arXiv:2507.07997; Mentzer et al. (2024). "FSQ." ICLR 2024

---

### 2. OBJECT-CENTRIC LEARNING - REAL-WORLD SCALING

**Major Update: Add Real-World Methods Section**

**DINOSAUR (ICLR 2023, updated 2024):**
- First to scale to real-world datasets (COCO 68.6 FG-ARI vs SLATE 43.6)
- Key innovation: Reconstruct DINO/DINOv2 features instead of pixels
- Architecture: Frozen DINO ViT → Slot Attention → MLP decoder
- GitHub: amazon-science/object-centric-learning-framework

**VideoSAUR/VideoSAURv2 (NeurIPS 2023/2024):**
- First object-centric video on unconstrained data (YouTube-VIS)
- SlotMixer decoder achieves linear complexity
- Temporal feature similarity loss for motion-based grouping
- MOVi-E: 77.1 video FG-ARI with DINOv2

**Update Slot Attention Variants:**
- **Adaptive Slot Attention (CVPR 2024)**: Dynamically determines slot count via Gumbel-Softmax
- **Slot Mixture Module (ICLR 2024)**: Uses GMM instead of k-means clustering
- **Guided Slot Attention (CVPR 2024)**: Query-based guidance for better foreground-background separation

**Add Performance Benchmarks Table:**
- MOVi-E: VideoSAURv2 (77.1), STEVE (50.6)
- COCO: SlotDiffusion (40.0 FG-ARI), DINOSAUR (39.7)

**Implementation Guidance:**
- Use OCLF framework (Amazon Science) for production
- Pre-trained models available
- DINOv2 features crucial for real-world performance

**Key Citation:** Seitzer et al. (2023). "DINOSAUR." ICLR 2023; Zadaianchuk et al. (2023). "VideoSAUR." NeurIPS 2023

---

### 3. CONCEPT LEARNING & INTERPRETABILITY - CBM EVOLUTION

**Add Major CBM Variants Section (2024-2025):**

**Graph CBM (ICLR 2025):**
- Constructs latent concept graphs capturing correlations
- More effective interventions using concept relationships
- Addresses independence assumption limitation

**Information Bottleneck CBM (ICLR 2025):**
- Limits mutual information to prevent concept leakage
- Enhanced accuracy without compromising target prediction

**Concept Bottleneck Generative Models (ICLR 2024):**
- First extension to GANs/VAEs/diffusion models
- Steering ~10× more effective than baselines
- Post-hoc training: 4-15× faster, +25% steerability

**Hybrid CBM (CVPR 2025):**
- Static (LLM-derived) + dynamic (learnable) concept banks
- Adapts to concept incompleteness

**Editable CBM (ICLR 2025):**
- Remove/insert data or concepts without retraining
- Closed-form approximations via influence functions

**Update Prototype Learning:**
- **R3-ProtoPNet (ICML 2024)**: RLHF-inspired with human feedback (1-5 scale)
- **ProtoPFormer (IJCAI 2024)**: Solves ViT "distraction problem"
- **ProtoSolo (2025)**: Single prototype per classification

**Add Mechanistic Interpretability Connection:**
- **Sparse Autoencoders**: Anthropic's 34M features in Claude 3 Sonnet
- **OpenAI**: 16M features in GPT-4 with k-sparse autoencoders
- Features: monosemantic units corresponding to human-understandable concepts
- Applications: Golden Gate Bridge (multimodal), brain sciences, code errors, safety-relevant features

**Critical Addition - Symbol Extraction Framework:**
1. Train SAE on neural activations
2. Extract interpretable features (monosemantic units)
3. Binarize/threshold activations
4. Apply logic programming (FOLD-SE-M)
5. Generate human-readable symbolic rules

**Key Citation:** Templeton et al. (2024). "Scaling Monosemanticity." Anthropic; Gao et al. (2024). "Scaling SAEs." OpenAI, arXiv:2406.04093

---

### 4. PROGRAM SYNTHESIS - ARC BREAKTHROUGH

**Update ARC Challenge Progress:**
- 2023 Baseline: 33% → 2024 Best: 55.5%
- OpenAI o3: 88% (high-compute), 76% (low-compute)
- Key methods: Test-Time Training (6× improvement), LLM-guided synthesis

**Add LILO as Major DreamCoder Alternative (ICLR 2024):**
- Dual-system synthesis: LLM-guided + enumerative
- Stitch compression: 1000-10000× faster than DreamCoder
- Auto-documentation: Natural language function names/docstrings
- Solves more complex tasks, discovers "vowel" concept
- GitHub: gabegrand/lilo

**Add Stitch (POPL 2023):**
- 1000-10000× faster, 100× less memory than DreamCoder
- Top-down corpus-guided synthesis
- Comparable/better library quality

**Update Recent Methods:**
- AbstractBeam (2024): Library learning for LambdaBeam
- PeARL (2024): Enables DreamCoder on ARC tasks
- ARC-AGI-2 (2025): New harder benchmark, $1M+ prize

**Key Citation:** Grand et al. (2024). "LILO." ICLR 2024; Bowers et al. (2023). "Stitch." POPL 2023

---

### 5. CAUSAL DISCOVERY & REPRESENTATION - LLM INTEGRATION

**Add LLM-Enhanced Causal Discovery Section (2024 Trend):**
- LLM-derived priors integrated into scoring functions
- High-level variable proposal for unstructured data
- Error detection/correction of LLM knowledge
- Applications: medicine, finance, genetics

**Add Novel 2024 Algorithms:**
- **CLOC (NeurIPS 2024)**: Cluster-level causal learning (exponentially fewer tests)
- **Agentic Stream of Thought (2025)**: Multiple LLMs with hierarchical decomposition

**Add Identifiable CRL Breakthroughs:**
- **von Kügelgen (2024 Cambridge Thesis)**: Three settings for identifiability
  - Unsupervised: IMA constraint
  - Multi-view: Invariant latent blocks
  - Multi-environment: Single-node interventions
  
- **Causal to Concept-Based (NeurIPS 2024)**: Relaxes to geometric concepts as linear subspaces, requires only n+2 environments

- **Morioka & Hyvarinen (ICML 2024)**: Identifiability via grouping variables (no temporal structure/interventions needed)

**Update Disentanglement:**
- ICM-VAE (2024): Flow-based diffeomorphic functions
- CausCell (Nature Comms 2025): Single-cell with causal DAG

**Key Citation:** von Kügelgen (2024). "Identifiable CRL." Cambridge Thesis; Rajendran et al. (2024). "Causal to Concept." NeurIPS 2024

---

### 6. SCENE GRAPHS & RELATIONAL LEARNING - ONE-STAGE REVOLUTION

**Add CVPR 2024 Breakthrough Methods:**

**EGTR (Best Paper Candidate):**
- Lightweight one-stage extracting graphs from DETR attention
- Eliminates separate triplet detectors
- SOTA on Visual Genome and Open Image V6
- GitHub: naver-ai/egtr

**HiKER-SGG (CMU):**
- Hierarchical knowledge graphs from external sources
- Strong zero-shot on corrupted images

**Open-Vocabulary Methods:**
- **OvSGTR (ECCV 2024 Best Paper Candidate)**: Fully open-vocabulary with MegaSG dataset
- **Pixels to Graphs (CVPR 2024)**: Image-to-sequence problem formulation

**Add HyperGLM (2024-2025):**
- HyperGraph integrating entity + procedural graphs
- Beyond pairwise: complex multi-object interactions
- VSGR dataset: 1.9M frames, 5 tasks

**Update GNN Architectures:**
- **G-Retriever (NeurIPS 2024)**: Integrates GNNs, LLMs, RAG; PyG 2.6 support
- **GNAN (NeurIPS 2024)**: Interpretable-by-design with global/local explanations

**Add Visual Relationship Detection:**
- Scene-Graph ViT (ECCV 2024): Decoder-free, real-time
- UniVRD: 38.07 mAP on HICO-DET (+14.26 improvement)

**Key Citation:** Im et al. (2024). "EGTR." CVPR 2024; He et al. (2024). "G-Retriever." NeurIPS 2024

---

### 7. SPARSE CODING & DICTIONARY LEARNING - THE BREAKTHROUGH

**CRITICAL UPDATE: This is THE major advance for neurosymbolic AI**

**Add Anthropic's "Scaling Monosemanticity" (May 2024) as Featured Section:**
- **34 million features** extracted from Claude 3 Sonnet (production model)
- Scaling laws: Loss decreases as power law with compute
- Highly abstract, multimodal features:
  - Golden Gate Bridge (activates on text AND images)
  - Brain sciences (neuroscience, cognitive science, psychology)
  - Code errors (bugs across multiple languages)
  - Safety-relevant: security vulnerabilities, bias, deception, power-seeking
- <300 active features/token, 65%+ variance explained
- Visualizer: transformer-circuits.pub/2024/scaling-monosemanticity/

**Add OpenAI's Work (June 2024):**
- 16M feature autoencoder on GPT-4
- k-sparse autoencoders using TopK activation (directly controls sparsity)
- New metrics: downstream loss, explainability, ablation sparsity
- GitHub: openai/sparse_autoencoder

**Add Technical Architecture Section:**
```
Input: Neural activation x (768-dim)
Encoder: W_enc @ x + b_enc → ReLU/TopK → f (16M-dim)
Decoder: W_dec @ f + b_dec → x_reconstructed
Loss: MSE + λ·L1(f) [L1] or MSE with TopK(f,k) [TopK]
```

**CRITICAL: Add Neurosymbolic AI Applications Section:**

**Vision Transformers (Padalkar et al., 2025 - Cambridge Core TPLP):**
- Sparse concept layer inspired by SAEs
- FOLD-SE-M algorithm generates logic program rules from binarized activations
- **First successful bridge from neural to symbolic via interpretable sparse features**

**Symbol Extraction Framework:**
1. Train SAE on neural activations
2. Extract interpretable features (monosemantic)
3. Binarize/threshold feature activations
4. Apply symbolic reasoning (logic programming)
5. Generate human-readable symbolic rules

**Model Steering via Features:**
- Clamping features controls behavior
- Examples: Golden Gate Bridge feature, code error feature, secrecy feature

**Add Feature Properties:**
- Feature splitting: Finer concepts as dictionary size increases
- Universality: Same features across models (0.72 correlation)
- Compositionality: Features combine linearly

**Implementation Resources:**
- OpenAI sparse_autoencoder (GPT-2): github.com/openai/sparse_autoencoder
- AI Safety Foundation: github.com/ai-safety-foundation/sparse_autoencoder
- TransformerLens integration

**Key Citation:** Templeton et al. (2024). "Scaling Monosemanticity." Anthropic; Padalkar et al. (2025). "Symbolic Rule Extraction from ViTs." arXiv:2505.06745

---

### 8. ENERGY-BASED MODELS - MODERN HOPFIELD NETWORKS

**Add Nobel Prize Recognition (2024):**
- John J. Hopfield and Geoffrey E. Hinton awarded Nobel Prize in Physics

**Update Core Framework:**
- Modern Hopfield update rule **mathematically equivalent** to transformer attention
- Exponential storage capacity (vs. linear in classical)
- One-step retrieval with exponentially small errors

**Add 2024-2025 Theoretical Advances:**

**Sparse and Structured Hopfield Networks (2024):**
- Connection to Fenchel-Young losses
- Sparse transformations enable exact memory retrieval
- First framework with exact convergence guarantees

**Hopfield-Fenchel-Young Networks (2024):**
- Unified framework for classical, modern, sparse variants
- Energy as difference of two Fenchel-Young losses

**Continuous-Time Memories (2025):**
- Compresses discrete memories into continuous-time
- Maintains performance with reduced computational cost

**Add Transformer Connection Section:**
- Temperature β governs phase transition (low: global averaging, high: pattern-specific)
- In-context learning as associative memory (2025 finding)
- Attention layer = gradient descent on energy landscape

**Add Neurosymbolic Applications:**
- Vector Symbolic Architectures with attention-based recall
- Chain-of-Thought reasoning as energy minimization
- Sparse Quantized Hopfield Network (Nature Comms 2024)

**Implementation:**
- PyTorch Hopfield layers: ml-jku/hopfield-layers
- Three layer types: Hopfield, HopfieldPooling, HopfieldLayer

**Key Citation:** Ramsauer et al. (2021). "Hopfield Networks is All You Need." ICLR 2021; Santos et al. (2024). "Sparse Hopfield Networks." arXiv:2402.13725

---

### 9. DISCRETE LATENT VARIABLE MODELS - TRAINING SOLVED

**Add Major VAE Variants (2024-2025):**

**VAEVQ (2025):**
- Soft categorical distribution with learnable codebook
- Fully differentiable from reconstruction loss
- Significant token length reduction

**DisCo-Diff (ICML 2024):**
- Discrete-continuous hybrid for diffusion models
- Only 10 discrete latents needed (codebook 100)
- ImageNet-64 FID: 2.36→1.22

**Add Gumbel-Softmax Improvements:**

**Decoupled ST-GS (2024):**
- Decouples temperature for forward (τ_fwd) and backward (τ_bwd) passes
- Independent control of discreteness vs. gradient smoothness
- Optimal bias-variance trade-off

**GST (ICML 2022):**
- 55% higher maximum returns in MADDPG

**Add VQ-STE++ (ICML 2023) - Major Training Advance:**

**Solutions to Core Problems:**
- **Affine re-parameterization**: Matches embedding moments
- **Alternating optimization**: Coordinate descent style
- **Synchronized commitment loss**: Updates codebook in task loss direction

**Add Codebook Collapse Solutions:**
- **CVQ-VAE (ICCV 2023)**: 100% utilization via dynamic initialization
- **Soft Convex Quantization (2024)**: Convex optimization, lower MSE
- **EdVAE (2024)**: Evidential deep learning prevents collapse

**Add Neurosymbolic Applications:**
- **Neurosymbolic Diffusion Models (2025)**: First diffusion over NeSy concepts, SOTA accuracy

**Key Citation:** Xu et al. (2024). "DisCo-Diff." ICML 2024; Huh et al. (2023). "VQ-STE++." ICML 2023

---

### 10. TOPOLOGICAL METHODS - TDA APPLICATIONS

**Add Recent TDA Applications (2024-2025):**

**Vector Stitching (November 2024):**
- Combines raw images with TDA-derived topological information
- Enhanced CNN performance on limited datasets

**OOD Detection (January 2025):**
- TDA characterizes OOD examples via latent embeddings
- Identifies topological "landmarks"

**Molecular Graphs (2025):**
- FTPG: Fuzzy neural network with TDA
- Outperforms SOTA GNN baselines

**Add Persistent Homology Advances:**
- Atom-Specific Persistent Homology (2024): 55% MAE reduction for defect predictions
- TopP-S (2024): Most accurate aqueous solubility predictions

**Add Symbol Extraction Methods:**
- **Topology of Transformations (JMLR 2024)**: Networks reduce complexity layer-by-layer
- **TOPF (2024-2025)**: Point-level topological features
- **ShapeDiscover (2025)**: Cover learning without filter functions

**Add Practical Tools:**
- **Scikit-TDA**: `pip install scikit-tda`
- **GUDHI**: C++ with Python, TensorFlow integration
- **Giotto-TDA**: Most comprehensive (JMLR 2021)
- **TopoX Suite (JMLR 2024)**: TopoNetX, TopoEmbedX, TopoModelX (≥95% coverage)

**Key Citation:** Ballester et al. (2024). "TDA for Neural Networks." arXiv:2312.05840; Hajij et al. (2024). "TopoX." JMLR 2024

---

### 11. SELF-SUPERVISED & CONTRASTIVE LEARNING - SYMBOL EMERGENCE

**Add DINOv3 (August 2025) - SOTA:**

**Gram Anchoring Innovation:**
- Maintains dense feature quality during extended training
- Operates on Gram matrices (pairwise patch products)
- 7B parameters, 1.7B images

**Performance:**
- ADE20k: 55.9 mIoU (linear probe)
- Superior dense features across benchmarks
- Stable at 4K+ resolutions

**Discrete Representation Capability:**
- Highly structured patch-level features
- Clear semantic clustering
- Emergent object-level representations

**Add Vector Quantized Methods:**

**BEiT v2 (ICLR 2024):**
- Vector-Quantized Knowledge Distillation
- Codebook learns explicit semantics
- 84.4% ImageNet, 49.2% ADE20k

**Add SwAV and Extensions:**
- Swapping Assignments Between Views
- No pairwise comparisons or memory bank needed
- 75.3% ImageNet with ResNet-50
- Multi-crop crucial (2 global + 4-6 local)

**CRITICAL: Add Connections to Symbol Learning Section:**

**Extracting Symbolic Sequences (March 2025) - BREAKTHROUGH:**
- **Martinez Pozos & Meza Ruiz**, arXiv:2503.04900
- Extends DINO to handle visual AND symbolic information
- Generates discrete, structured symbolic sequences from visual data
- Decoder transformer uses cross-attention for interpretable symbols
- **First direct method for SSL-to-symbol extraction**

**Symbolic Autoencoding (2024):**
- Discrete bottleneck connecting generative models
- >98% token accuracy on SCAN, CFQ, COGS

**Emergent Interpretable Symbols:**
- V3 framework: Discrete content + continuous style
- One-to-one alignment with human knowledge

**Key Citation:** Oquab et al. (2025). "DINOv3." arXiv:2508.10104; Martinez Pozos & Meza Ruiz (2025). "Extracting Symbolic Sequences." arXiv:2503.04900

---

### 12. HYBRID NEURAL-SYMBOLIC ARCHITECTURES - SYSTEMATIC VIEW

**Add Systematic Review (2025):**
- **Colelough & Regli**, arXiv:2501.05435
- 167 papers from 1,428 candidates (2020-2024)
- Learning/inference: 63%, Logic/reasoning: 35%
- Knowledge representation: 44%, Explainability: 28%
- Meta-cognition: 5% (least explored)

**Update Kautz's Taxonomy:**
1. Symbolic[Neural]: Symbolic invokes neural (AlphaGo)
2. Neural[Symbolic]: Neural calls symbolic engines
3. Neural|Symbolic: Neural interprets perception as symbols
4. Neural:Symbolic→Neural: Symbolic generates training data
5. NeuralSymbolic: Neural networks from symbolic rules

**Add Recent Innovations:**
- **Neuro→Symbolic←Neuro (2025)**: Outperforms other architectures across metrics
- Integration approaches: Sequential, Parallel, Integrated

**Add Production Applications:**
- Amazon Vulcan & Rufus (2025): Warehouse robots, shopping assistant
- Google DeepMind AlphaProof/AlphaGeometry 2 (2024): IMO silver-medalist

**Add Efficiency Evidence:**
- 100× smaller models (symbolic distillation)
- 96.9% parameter reduction (PhysORD)
- 1-10% of data required vs. pure neural

**Key Citation:** Colelough & Regli (2025). "NeSy AI in 2024." arXiv:2501.05435; Velasquez et al. (2025). "NeSy as Antithesis to Scaling." PNAS Nexus

---

### 13. DIFFERENTIABLE LOGIC SYSTEMS - NANOSECOND INFERENCE

**Add Convolutional Differentiable Logic Gate Networks (NeurIPS 2024 Oral) - MAJOR BREAKTHROUGH:**

**Citation:** Petersen et al., NeurIPS 2024

**Performance:**
- 86.29% accuracy on CIFAR-10 using only 61 million logic gates
- **29× smaller** than state-of-the-art
- Inference in **4 nanoseconds**
- Scales logic gate networks by 10×+

**Technical Details:**
- Uses NAND, OR, XOR gates
- Differentiable relaxation enables gradient optimization
- Hardware-friendly: CPUs, GPUs, FPGAs, ASICs

**Add Extensions:**
- **Recurrent DLGNs (2025)**: Sequential logic elements for machine translation

**Add Differentiable Logic Machines (TMLR 2024):**
- Solves ILP and RL problems
- First-order logic with weights on predicates
- **3.5× higher success rate** on ILP vs SOTA

**Key Citation:** Petersen et al. (2024). "Convolutional DLGNs." NeurIPS 2024 Oral

---

### 14. LOGIC TENSOR NETWORKS - UPDATE STATUS

**Reaffirm Core Framework (2022, active 2024):**
- Fully differentiable first-order logic
- Fuzzy logic semantics
- TensorFlow 2 implementation
- GitHub: logictensornetworks/logictensornetworks

**Capabilities:**
- Multi-label classification, relational learning, clustering
- Semi-supervised learning, embedding learning, query answering

**Grounding:**
- Constants → ℝⁿ vectors
- Functions → neural networks
- Predicates → fuzzy relations
- Quantifiers → aggregation operators

**Key Citation:** Badreddine et al. (2022). "Logic Tensor Networks." Artificial Intelligence 303:103649

---

### 15. NEURAL THEOREM PROVING - APPROACHING HUMAN LEVEL

**Add Major Systems (2024):**

**LEGO-Prover (ICLR 2024):**
- 57.0% on miniF2F-valid (up from 48.0%)
- 50.0% on miniF2F-test (up from 45.5%)
- Generated 20,000+ new skills
- Modular proof construction

**DeepSeek-Prover-V1.5 (ICLR 2025):**
- Proof assistant feedback for RL and MCTS
- Large-scale synthetic data

**Add PutnamBench (NeurIPS 2024):**
- 1,692 formalizations of 640 theorems
- Multilingual: Lean 4, Isabelle, Coq
- Based on Putnam Mathematical Competition

**Add Technical Approaches:**
- **3SIL**: 70.2% on AIM (vs Waldmeister 65.5%), combined 90%
- **Neural Rewriting**: 8.3% more theorems proved

**Key Citation:** Wang et al. (2024). "LEGO-Prover." ICLR 2024

---

### 16. RULE EXTRACTION AND SYMBOLIC DISTILLATION - CRITICAL ADVANCE

**Add Symbolic Knowledge Distillation of LLMs (2024):**

**Framework** (arXiv:2408.10210):
1. Craft customized prompts
2. Apply NLP techniques (NER, POS, dependency parsing)
3. Transform to structured formats
4. Train critic model to filter

**GPT-3 Distillation:**
- **100× smaller models** with superior performance
- ATOMIC-10X knowledge graph
- COMET-DISTIL for commonsense

**CRITICAL: Add Rule Extraction from Vision Transformers (2025) - FIRST SUCCESSFUL:**

**Citation:** Padalkar & Gupta, arXiv:2505.06745, Cambridge Core TPLP 2025

**Major Innovation:**
- **First successful extraction of executable logic programs from ViTs**
- Sparse concept layer inspired by Sparse Autoencoders
- FOLD-SE-M algorithm for rule generation
- **5.14% better accuracy** than standard ViT
- Rules semantically meaningful and concise

**Add CNN Rule Extraction:**
- NeSyFOLD (AAAI 2024): Maps kernel activations to logic programs
- Medical imaging (IEEE 2024): Validated against radiomics features

**Add Symbolic Distillation Techniques:**
- **Ctrl-G (2024)**: 175B→7B+2B, 30%+ better on constrained generation

**Key Citation:** Padalkar & Gupta (2025). "Rule Extraction from ViTs." arXiv:2505.06745; Survey (2024). "Symbolic Distillation." arXiv:2408.10210

---

### 17. DIFFERENTIABLE ILP - ADVANCES

**Add Recent Frameworks:**

**DFORL (AI Journal 2024):**
- Generates first-order Horn clauses
- Novel propositionalization method
- Precise, robust, scalable

**δILP Extensions (2024):**
- Large-scale predicate invention via high-dimensional gradient descent
- No need to specify precise structure

**αILP (Machine Learning 2023):**
- Object-centric perception with ILP
- Handles complex visual scenes
- **First 4th-generation neurosymbolic for visual reasoning**

**GLIDR (2024):**
- Graph-like syntax (branches and cycles)
- Differentiable message passing
- Extractable explicit logic rules

**Key Citation:** Gao et al. (2024). "DFORL." Artificial Intelligence; Purgał et al. (2024). "δILP in High-Dimensional Space." arXiv:2208.06652

---

### 18. CLUSTERING FOR SYMBOL EXTRACTION - PRACTICAL METHODS

**Add DKM (Apple ML Research 2024):**
- Differentiable k-means as attention problem
- Joint optimization of parameters and centroids
- Superior compression-accuracy on ImageNet1k and GLUE

**Add CNNI (2024):**
- First parametric clustering for non-convex data
- Clustering index as loss function
- MMJ-SC index for non-flat geometry

**Add Neuron Clustering (MDPI 2024):**
- 54% reduction in visual clutter
- 88.7% reduction in connections
- Healthcare explainability

**Add NEON Framework:**
- Explains cluster assignments via neural networks
- Layer-wise Relevance Propagation
- Pixel-wise contributions for images

**Key Citation:** Apple ML Research (2024). "DKM." arXiv

---

### 19. TRANSFORMER-BASED METHODS - MECHANISTIC INSIGHTS

**Add Mechanistic Analysis (ACL 2024):**

**Backward Chaining Circuits (Brinkmann et al.):**
- Depth-bounded recurrent mechanisms in parallel
- Stores intermediate results in "register tokens"
- Implements backward chaining interpretably
- Inductive bias toward parallelized search

**Add Symbolic Framework (NAACL 2024):**
- Fine-tuned BERT rivals GPT-4 in-distribution
- Perturbations reduce performance by up to 80 F1
- Reveals brittleness to symbolic transformations

**Add Reasoning Step Limits:**
- L-layer Transformer: O(L) to O(L²) reasoning steps
- Buffer mechanism stores diverse information

**Add Symbolic Integration (NeurIPS MATH-AI 2024):**
- 30% accuracy gain, 70% precision gain over heuristics
- Layer Integrated Gradients for interpretability

**Key Citation:** Brinkmann et al. (2024). "Mechanistic Analysis." ACL 2024; Meadows et al. (2024). "Symbolic Framework." NAACL 2024

---

### 20. LLM EMERGENT SYMBOLIC CAPABILITIES - UNDERSTANDING EMERGENCE

**Add Characterization:**
- Ability not present in smaller models but present in larger
- Cannot be predicted by extrapolating smaller model performance
- Genuine discontinuities at ~13B-175B parameter thresholds

**Add Documented Capabilities:**

**In-Context Learning:**
- Learn novel input-label associations in real-time
- Semantically Unrelated Label ICL

**Theory of Mind:**
- False-belief tasks
- Age 4-5 human cognitive skill approximated

**Mathematical Reasoning:**
- **OpenAI o1**: 83.3% on AIME 2024 (vs GPT-4o 13.4%)
- 89.0% on Codeforces (vs GPT-4o 11.0%)

**Add Theoretical Perspectives:**
- Phase transitions in semantic space
- Energy landscapes reorganize before emergence
- Adjacent Possible Theory: path-dependent capabilities

**Add Critiques:**
- No clear coarse-grained variables for most behaviors
- Out-of-distribution failure (GSM-Symbolic brittleness)
- Limited evidence of true understanding

**Key Citation:** Wei et al. (2022). "Emergent Abilities." NeurIPS 2022; OpenAI (2024). "o1 System Card."

---

## CROSS-CUTTING THEMES TO EMPHASIZE

### 1. The Interpretability Breakthrough

**2024-2025 MILESTONE: Sparse Autoencoders Scale to Production**
- Anthropic: 34M features in Claude 3 Sonnet
- OpenAI: 16M features in GPT-4
- **Single most important advance for neurosymbolic AI**
- Direct extraction of monosemantic symbols from frontier models

### 2. Codebook Collapse: SOLVED

**Three Independent Solutions:**
- FSQ: Eliminate by design (100% utilization)
- SimVQ: Linear space optimization (near 100%)
- MGVQ: Multiple small codebooks (100%)

**Impact:** Vector quantization now practical at scale

### 3. Real-World Scaling Success

**Object-Centric Learning:**
- DINOSAUR/VideoSAUR work on COCO, YouTube-VIS
- Key: Self-supervised feature reconstruction

**Scene Graphs:**
- EGTR/OvSGTR achieve SOTA on Visual Genome
- Open-vocabulary methods scale to unconstrained data

### 4. Neurosymbolic Efficiency vs. Pure Scaling

**Quantitative Evidence:**
- 100× smaller models (symbolic distillation)
- 96.9% parameter reduction (PhysORD)
- 29× compression (logic gates)
- 1-10% of data required

### 5. Production Deployment Success

**2024-2025 Applications:**
- Amazon (Vulcan, Rufus)
- Google DeepMind (AlphaProof, AlphaGeometry 2)
- Anthropic (Claude 3 Sonnet with 34M interpretable features)
- OpenAI (o1 with 83.3% on AIME)

---

## CRITICAL RESEARCH GAPS

### 1. Direct Neural-to-Symbolic Translation (HIGH PRIORITY)

**Current State:** Only Martinez Pozos et al. (2025) directly extracts symbolic sequences from visual SSL
**Gap:** Limited to single domain; need generalizable framework
**Recommendation:** Major focus area for guide expansion

### 2. Meta-Cognition (5% of Research)

**Gap:** Systems reasoning about their own reasoning
**Needed:** Self-aware neurosymbolic architectures
**Applications:** Self-correction, uncertainty quantification

### 3. Multimodal Neurosymbolic Systems

**Gap:** Most work single-modality
**Needed:** Unified vision-language-action frameworks
**Applications:** Robotics, embodied AI

### 4. Hardware Co-Design

**Gap:** Hardware optimized for neural networks, not hybrids
**Opportunity:** Logic gates run in nanoseconds on specialized hardware

---

## IMPLEMENTATION RECOMMENDATIONS

### Quick Start by Goal:

**Extract Symbols from Models:**
→ Sparse Autoencoders (OpenAI sparse_autoencoder library)

**Learn Discrete Representations:**
→ Finite Scalar Quantization (vector-quantize-pytorch)

**Object-Centric Visual Understanding:**
→ DINOSAUR (OCLF framework)

**Symbolic Rule Extraction:**
→ Padalkar et al. (2025) ViT method + FOLD-SE-M

**Neurosymbolic Reasoning:**
→ Logic Tensor Networks (TensorFlow 2)

### Essential Tools:

**Interpretability:**
- TransformerLens, sparse_autoencoder, Circuitsvis

**Discrete Representations:**
- vector-quantize-pytorch, FSQ, OCLF

**Neurosymbolic:**
- Logic Tensor Networks, Scallop, DeepProbLog

**Topological:**
- Scikit-TDA, GUDHI, TopoX

---

## TOP 50 CRITICAL CITATIONS (2024-2025)

### Interpretability & Sparse Coding (TOP PRIORITY)
1. Templeton et al. (2024). "Scaling Monosemanticity." Anthropic
2. Gao et al. (2024). "Scaling and Evaluating SAEs." arXiv:2406.04093
3. Padalkar & Gupta (2025). "Rule Extraction from ViTs." arXiv:2505.06745

### Vector Quantization
4. Jia et al. (2025). "MGVQ." arXiv:2507.07997
5. Mentzer et al. (2024). "FSQ." ICLR 2024
6. Zhu et al. (2024). "SimVQ." arXiv:2411.02038

### Object-Centric Learning
7. Seitzer et al. (2023). "DINOSAUR." ICLR 2023
8. Zadaianchuk et al. (2023). "VideoSAUR." NeurIPS 2023
9. Fan et al. (2024). "Adaptive Slot Attention." CVPR 2024

### Concept Learning
10. OpenReview (2025). "Graph CBM." ICLR 2025 qPH7lAyQgV
11. OpenReview (2025). "Information Bottleneck CBM." ICLR 2025 2xRTdzmQ6C
12. OpenReview (2024). "CB-GM." ICLR 2024 L9U5MJJleF

### Program Synthesis
13. Grand et al. (2024). "LILO." ICLR 2024
14. Bowers et al. (2023). "Stitch." POPL 2023
15. ARC Prize (2024). "ARC Prize Technical Report." arXiv:2412.04604

### Causal Discovery
16. von Kügelgen (2024). "Identifiable CRL." Cambridge PhD Thesis
17. Rajendran et al. (2024). "Causal to Concept." NeurIPS 2024
18. Morioka & Hyvarinen (2024). "CRL via Grouping." ICML 2024

### Scene Graphs & GNNs
19. Im et al. (2024). "EGTR." CVPR 2024
20. He et al. (2024). "G-Retriever." NeurIPS 2024
21. Chen et al. (2024). "OvSGTR." ECCV 2024

### Energy-Based Models
22. Ramsauer et al. (2021). "Hopfield Networks is All You Need." ICLR 2021
23. Santos et al. (2024). "Sparse Hopfield Networks." arXiv:2402.13725
24. Santos et al. (2024). "Hopfield-Fenchel-Young." arXiv:2411.08590

### Discrete Latent Variables
25. Xu et al. (2024). "DisCo-Diff." ICML 2024
26. Huh et al. (2023). "VQ-STE++." ICML 2023
27. Yang et al. (2025). "VAEVQ." arXiv:2511.06863

### Topological Methods
28. Ballester et al. (2024). "TDA for Neural Networks." arXiv:2312.05840
29. Hajij et al. (2024). "TopoX." JMLR 2024 Article 374
30. Pham et al. (2025). "FTPG Molecular Graphs." Molecular Informatics

### Self-Supervised Learning
31. Oquab et al. (2025). "DINOv3." arXiv:2508.10104
32. Martinez Pozos & Meza Ruiz (2025). "Extracting Symbolic Sequences." arXiv:2503.04900
33. Peng et al. (2024). "BEiT v2." ICLR 2024

### Neurosymbolic Architectures
34. Colelough & Regli (2025). "NeSy AI in 2024." arXiv:2501.05435
35. Velasquez et al. (2025). "NeSy as Antithesis." PNAS Nexus pgaf117
36. arXiv (2025). "Neuro→Symbolic←Neuro." arXiv:2502.11269

### Differentiable Logic
37. Petersen et al. (2024). "Convolutional DLGNs." NeurIPS 2024 Oral
38. Miotti et al. (2025). "Recurrent DLGNs." arXiv:2508.06097
39. TMLR (2024). "Differentiable Logic Machines."

### Logic Tensor Networks
40. Badreddine et al. (2022). "Logic Tensor Networks." AI 303:103649

### Neural Theorem Proving
41. Wang et al. (2024). "LEGO-Prover." ICLR 2024
42. NeurIPS (2024). "PutnamBench."

### Rule Extraction & Distillation
43. Survey (2024). "Symbolic Knowledge Distillation." arXiv:2408.10210
44. AAAI (2024). "NeSyFOLD."

### Differentiable ILP
45. Gao et al. (2024). "DFORL." Artificial Intelligence
46. Purgał et al. (2024). "δILP in High-D Space." arXiv:2208.06652

### Transformer Reasoning
47. Brinkmann et al. (2024). "Backward Chaining Circuits." ACL 2024
48. Meadows et al. (2024). "Symbolic Framework." NAACL 2024

### LLM Emergence
49. Wei et al. (2022). "Emergent Abilities." NeurIPS 2022
50. OpenAI (2024). "o1 System Card."

---

## FINAL RECOMMENDATIONS FOR GUIDE UPDATE

### Priority 1: Add Sparse Autoencoders as Featured Method
- This is THE breakthrough for symbol extraction
- Create dedicated section with Anthropic/OpenAI work
- Include implementation guide and visualization tools

### Priority 2: Update Vector Quantization Completely
- FSQ, MGVQ, SimVQ solve codebook collapse
- Replace old methods with 2024-2025 approaches
- Add performance benchmarks table

### Priority 3: Add Real-World Object-Centric Learning
- DINOSAUR and VideoSAUR scale to real data
- Critical for practical neurosymbolic applications
- Include OCLF framework usage

### Priority 4: Expand Concept Learning with CBM Variants
- 5+ major new variants (Graph, IB, Generative, Hybrid, Editable)
- Update prototype learning with R3, ProtoPFormer, ProtoSolo
- Add mechanistic interpretability connection

### Priority 5: Update Each Section with 2024-2025 Methods
- Every section has significant new developments
- Add performance benchmarks where available
- Include GitHub repositories and implementation resources

### Priority 6: Add Cross-Cutting Themes Section
- Convergence on sparse representations
- Neurosymbolic efficiency vs. pure scaling
- Production deployment success stories

### Priority 7: Highlight Research Gaps
- Direct neural-to-symbolic translation (highest priority)
- Meta-cognition (least explored)
- Multimodal integration
- Hardware co-design

---

## CONCLUSION

The 2024-2025 period represents a watershed moment for neurosymbolic AI, with three transformative breakthroughs:

1. **Sparse Autoencoders scaling to 34M features** enable direct symbol extraction from production models
2. **Codebook collapse definitively solved** through FSQ, SimVQ, and MGVQ approaches
3. **Real-world scaling achieved** for object-centric learning via self-supervised feature reconstruction

These advances position neurosymbolic AI as a practical, efficient alternative to pure scaling laws, achieving 100× smaller models with comparable or superior performance while maintaining interpretability and symbolic reasoning capabilities.

The guide should emphasize that the field has matured from theoretical exploration to practical deployment, with production systems at Amazon, Google DeepMind, Anthropic, and OpenAI demonstrating real-world impact. The convergence on sparse, discrete representations across multiple independent research areas provides a unified foundation for future neurosymbolic systems.

**Most Critical Update**: The sparse autoencoder breakthrough (Anthropic's 34M features, OpenAI's 16M features) represents the single most important advance for extracting interpretable symbols from neural networks and should be featured prominently in any updated guide on symbols in neurosymbolic AI.