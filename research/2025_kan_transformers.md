# Kolmogorov-Arnold Networks: Architecture advances and transformer integration (2024-2025)

Kolmogorov-Arnold Networks (KANs) have emerged as a significant neural architecture innovation, with the foundational paper published in **April 2024** sparking rapid development of variants and transformer integrations. KANs place **learnable activation functions on edges** rather than fixed activations on nodes, achieving substantial improvements in symbolic regression and scientific computing—though MLPs retain advantages for general classification tasks. The most impactful development has been the **Kolmogorov-Arnold Transformer (KAT)**, which achieves **+3.1% accuracy gains** over Vision Transformers on ImageNet through rational function bases and group parameter sharing.

This report synthesizes the technical landscape of KAN research from 2024-2025, covering core architectural innovations, transformer integration methods, performance benchmarks, and theoretical foundations.

---

## The foundational KAN architecture redefines neural network design

The original KAN paper (arXiv:2404.19756, **April 30, 2024**) by Ziming Liu, Yixuan Wang, and colleagues at MIT, Caltech, and Harvard introduced a fundamentally different approach to neural network construction. Rather than employing fixed activation functions on neurons with learnable linear weights (as in MLPs), KANs use **learnable univariate functions parametrized as B-splines on every edge**.

The mathematical foundation rests on the Kolmogorov-Arnold representation theorem, which states any continuous multivariate function can be decomposed as:

**f(x₁, ..., xₙ) = Σ Φq(Σ φq,p(xp))**

where φ and Φ are continuous univariate functions. This theorem, proven by Kolmogorov and Arnold in 1956-1957 in response to Hilbert's 13th Problem, demonstrates that **addition is the only truly multivariate operation**—all other continuous functions decompose into univariate components and sums.

The KAN layer applies learnable activation functions φ(x) = Σᵢ cᵢBᵢ(x), where Bᵢ are B-spline basis functions (default: cubic, k=3) and cᵢ are trainable coefficients. The default configuration uses **G=3 grid intervals** and **k=3 spline order**, yielding G+k parameters per edge. A key innovation is **grid extension**—training begins on coarse grids and progressively refines to finer grids, enabling learning of both low-frequency and high-frequency function components.

**KAN 2.0** (arXiv:2408.10205, **August 19, 2024**) expanded the framework with MultKAN (multiplication nodes), a KAN compiler for symbolic formula conversion, and enhanced scientific discovery pipelines. The pykan library now supports bidirectional "Science↔KAN" workflows.

---

## Basis function variants dramatically improved computational efficiency

The original B-spline implementation faced severe computational bottlenecks—**10-200x slower than MLPs**—prompting extensive research into alternative basis functions throughout 2024-2025:

| Variant | ArXiv ID | Date | Basis Function | Speed vs Original |
|---------|----------|------|----------------|-------------------|
| **FastKAN** | 2405.06721 | May 10, 2024 | Gaussian RBF | **3.33x faster** |
| **FasterKAN** | — | May 2024 | RSWAF | 1.5x faster than FastKAN |
| **ChebyKAN** | 2405.07200 | May 12, 2024 | Chebyshev polynomials | **Fastest overall** |
| **Wav-KAN** | 2405.12832 | May 21, 2024 | Wavelets | Multi-scale capability |
| **FourierKAN** | — | 2024 | Fourier series | Periodic patterns |
| **TKAN** | 2405.07344 | May 12, 2024 | B-splines + temporal | Time series |
| **fKAN** | 2406.07456 | June 2024 | Fractional Jacobi | Scientific computing |
| **BSRBF-KAN** | 2406.11173 | June 17, 2024 | B-splines + RBF combined | 97.55% MNIST accuracy |

**ChebyKAN** emerged as the speed-accuracy frontrunner, replacing B-splines with Chebyshev polynomials computed via efficient recursive formulas or trigonometric definitions. Benchmarks show ChebyKAN achieving **1.03ms forward pass** versus **4.76ms for efficient-kan** on equivalent architectures—though still **2x slower than baseline MLPs** (0.47ms).

**FastKAN** demonstrated that KANs are mathematically equivalent to certain **Radial Basis Function networks**, using Gaussian RBFs: bᵢ(u) = exp(-(u - uᵢ)²/h²). This insight simplified implementation and eliminated grid adjustment requirements through LayerNorm input scaling.

For scientific computing, **Wav-KAN** introduced wavelet bases that capture both high-frequency and low-frequency components simultaneously—particularly valuable for physics-informed neural networks where solutions exhibit multi-scale structure.

---

## Transformer integration follows two primary architectural patterns

Research on KAN-transformer hybrids has coalesced around two integration strategies: **FFN/MLP replacement** (mature and well-validated) and **attention mechanism replacement** (emerging research frontier).

### FFN replacement: the dominant integration approach

The **Kolmogorov-Arnold Transformer (KAT)** (arXiv:2409.10594, **September 2024**, accepted to **ICLR 2025**) by Xingyi Yang and Xinchao Wang at National University of Singapore represents the most successful KAN-transformer integration. KAT replaces MLP layers with Group-Rational KAN (GR-KAN) while preserving standard attention mechanisms:

**Standard ViT:** x_ℓ = Attn(Norm(x_{ℓ-1})) + MLP(Norm(x))  
**KAT:** x_ℓ = Attn(Norm(x_{ℓ-1})) + **GR-KAN**(Norm(x))

GR-KAN addresses three critical challenges that prevented naive KAN integration:

1. **Base function**: Replaces B-splines with **rational functions** (CUDA-optimized), avoiding spline computation overhead
2. **Efficiency**: Implements **Group KAN**—sharing activation weights across neuron groups rather than per-edge
3. **Initialization**: Uses **variance-preserving initialization** for training stability in deep networks

Notably, vanilla KAN integration into ViT-B/L fails to converge (producing NaN errors), while ViT-T/S with vanilla KAN achieves only ~63% accuracy versus **82.3% for KAT-B**. The group sharing mechanism proves essential for scalability.

Other FFN replacement architectures include:

- **S-KANformer** (NeurIPS ML4PS 2024): Uses **SineKAN** in the final decoder block only, achieving orders-of-magnitude speedup while outperforming standard transformers on symbolic regression
- **SCKansformer** (arXiv:2406.09931, IEEE JBHI 2024): Combines KAN encoder with spatial/channel reconstruction for medical cell classification
- **ViKANformer** (arXiv:2503.01124, March 2025): Modular "plug-and-play" design testing multiple KAN variants (SineKAN, Fast-KAN, FourierKAN) as dimension-wise FFN replacements
- **Kanformer** (arXiv:2510.06706, October 2025): Integrates ChebyKAN into Conformer architecture for speech detection, achieving **60.55% relative improvement** on ASVspoof2021

### Attention mechanism replacement: the emerging frontier

**KArAt** (Kolmogorov-Arnold Attention, arXiv:2503.10632, **March 2025**) represents the first systematic attempt to replace softmax attention with learnable KAN-based functions:

**Traditional attention:** A^{i,j} = softmax(qᵢ · kⱼ / √d)  
**KArAt:** A^{i,j} = **Φ^{i,j}**(qᵢ · kⱼ / √d) where Φ is learnable

KArAt supports multiple bases (Fourier, Wavelets, Splines, Rational Functions) but faces a critical challenge: learnable per-pair activations cause **O(N²) memory explosion**. The **Modular KArAt** variant addresses this through low-rank approximation of the operator matrix, with **Fourier-KArAt** providing the best computational efficiency.

---

## Medical imaging and scientific computing show strongest KAN advantages

Performance benchmarks reveal KANs excel in **domain-specific applications** rather than general classification:

### Vision transformer benchmarks (KAT)

| Model | ImageNet-1K Top-1 | Improvement |
|-------|-------------------|-------------|
| ViT-B baseline | 79.2% | — |
| **KAT-B** | **82.3%** | **+3.1%** |
| DeiT-S baseline | 78.8% | — |
| **KAT-S** | **81.2%** | **+2.4%** |
| KAT-B (pretrained init) | 82.7% | +3.5% |

KAT also achieves **+3.0 AP improvement** on MS-COCO object detection and **2.4% improvement** on ADE20K semantic segmentation using UperNet.

However, **training speed remains problematic**: FlashKAT (arXiv:2505.13813, May 2025) documented that KAT trains **123x slower** than standard transformers before optimization, reducible to ~20% slower with memory stall mitigations.

### Medical image segmentation (U-KAN, AAAI 2025)

**U-KAN** (arXiv:2406.02918, June 2024) demonstrates KAN advantages in medical imaging:

- Achieves **highest IoU and F1 scores** across BUSI, GLAS, and CVC-ClinicDB benchmarks
- Outperforms U-Net, U-Net++, U-Mamba, and U-NeXt
- Delivers higher accuracy with **lower computational cost**

**UKAST** (arXiv:2511.04084, November 2025) further integrates GR-KAN into Swin Transformer encoders for data-efficient medical segmentation, achieving SOTA on four diverse benchmarks.

### Scientific computing and symbolic regression

The original KAN paper demonstrated:

- 2-layer width-10 KAN achieves **100x more accuracy** than 4-layer width-100 MLP on function fitting (10⁻⁷ vs 10⁻⁵ MSE)
- **100x more parameter efficient** (10² vs 10⁴ parameters)
- 2D Poisson equation error reduced from 24% to **0.05%** with grid extension

Comprehensive benchmarks ("KAN or MLP: A Fairer Comparison") controlling for parameters/FLOPs found:

- **KAN significantly outperforms** on symbolic formula representation
- **MLP outperforms** on MNIST, CIFAR-10, general CV, NLP, and audio tasks
- KAN's advantage stems primarily from **learnable B-spline activations**, not architectural novelty—MLPs with B-spline activations match KAN performance

---

## Approximation theory provides dimension-independent error bounds

KAN's theoretical appeal rests on favorable approximation properties. **Theorem 2.1** (Liu et al., 2024) establishes:

**‖f - KAN_G(x)‖ ≤ C · G^{-(k+1-m)}**

where G is grid size, k is spline order, and m is derivative order. Critically, the bound is **independent of input dimension n**, suggesting KANs can potentially overcome the curse of dimensionality for functions admitting smooth KAN representations.

**Expressiveness comparisons** (Wang et al., arXiv:2410.01803, October 2024) proved:

1. **MLP ⊆ KAN**: Any MLP can be represented using comparable-size KANs
2. **KAN → MLP**: Conversion increases parameters by factor G (grid size)
3. **Spectral bias**: KANs exhibit less low-frequency bias than MLPs; grid extension improves high-frequency learning

**Free-Knot KAN** (arXiv:2501.09283, January 2025) extends theoretical flexibility by making spline knot positions trainable alongside coefficients, addressing fixed-grid limitations.

Robustness analysis (Dong et al., arXiv:2408.07314) showed KANs exhibit **lower Lipschitz constants** than MLPs, providing adversarial robustness advantages—output is primarily determined by the base component rather than B-spline variations.

---

## Implementation ecosystem supports multiple variants and applications

### Primary repositories

| Repository | Stars | Description |
|------------|-------|-------------|
| **pykan** (Official) | 15.9k+ | MIT implementation by Ziming Liu |
| **efficient-kan** | Popular | Fast pure-PyTorch implementation |
| **fast-kan** | — | RBF-based, 3.33x faster |
| **KAT** | 833+ | Kolmogorov-Arnold Transformer (ICLR 2025) |
| **awesome-kan** | 3.1k+ | Comprehensive resource collection |

### Specialized implementations

- **ChebyKAN**: github.com/SynodicMonth/ChebyKAN
- **Wav-KAN**: github.com/zavareh1/Wav-KAN
- **FourierKAN**: github.com/GistNoesis/FourierKAN
- **U-KAN**: github.com/CUHK-AIM-Group/U-KAN
- **KAN-GPT**: github.com/AdityaNG/kan-gpt
- **SCKansformer**: github.com/JustlfC03/SCKansformer
- **UKAST**: github.com/nsapkota417/UKAST

### Computational benchmarks (V100 GPU, [28×28, 256, 10] network)

| Implementation | Forward Pass | Memory |
|----------------|--------------|--------|
| **MLP baseline** | **0.47 ms** | 0.10 GB |
| ChebyKAN | 1.03 ms | 0.14 GB |
| efficient-kan | 4.76 ms | 0.13 GB |
| FourierKAN | 17.93 ms | 1.96 GB |

---

## Conclusion

KAN architectures have transitioned from theoretical curiosity to practical tools for **specific domains**—particularly scientific computing, symbolic regression, and medical imaging—where their interpretability and function approximation properties provide genuine advantages. The **Kolmogorov-Arnold Transformer** represents the most mature integration with transformers, achieving meaningful accuracy improvements (+2-3%) on vision tasks through rational function bases and group parameter sharing.

However, practitioners should recognize clear limitations: KANs remain **4-100x slower** than MLPs, underperform on general classification tasks, and require careful architectural choices (basis function selection, grid configuration, group sharing) to scale effectively. The emerging work on attention mechanism replacement (KArAt) opens new research directions but faces memory efficiency challenges.

For researchers evaluating KAN adoption, the key insight from comprehensive benchmarking is that **KAN's advantages stem primarily from learnable activation functions**—suggesting hybrid approaches combining MLP-style efficiency with KAN-inspired learnable nonlinearities may ultimately prove most practical for production systems.