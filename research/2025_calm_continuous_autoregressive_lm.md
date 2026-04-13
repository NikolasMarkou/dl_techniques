# CALM: Continuous Autoregressive Language Models

**Paper:** *Continuous Autoregressive Language Models* (arXiv:2510.27688v1, 31 Oct 2025)
**Authors:** Chenze Shao, Darren Li, Fandong Meng, Jie Zhou
**Affiliations:** WeChat AI, Tencent Inc.; Qiuzhen College, Tsinghua University
**Code:** `github.com/shaochenze/calm` · **Project:** `shaochenze.github.io/blog/2025/CALM`

---

## 1. Core thesis

LLM scaling has three known axes: parameters, data, and compute. CALM proposes a **fourth axis: semantic bandwidth per generative step**.

- A discrete token carries only ~15–18 bits (log₂ of a 32k–256k vocab). Raising this requires exponential vocab growth → softmax becomes intractable.
- **Solution:** replace "predict next token" with "predict next **continuous vector**", where each vector is an autoencoder-compressed chunk of K tokens.
- Sequence length drops by factor K → training and inference FLOPs drop proportionally.
- Reconstruction fidelity: >99.9% token-level accuracy even under noisy latents.

The catch: moving to ℝˡ kills softmax → no tractable likelihood. CALM therefore builds a **fully likelihood-free stack** (training loss, evaluation metric, temperature sampling).

---

## 2. Component 1 — The Autoencoder

Maps `K` tokens ↔ one vector `z ∈ ℝˡ`. **Context-free** (chunks processed independently) for simplicity; context-aware version left as future work.

### 2.1 Architecture

Encoder `f_enc: 𝒱^K → ℝˡ`:
```
K tokens → K embeddings → position-wise FFN (per-token)
        → flatten → Linear(Kd → d) → FFN → Linear(d → l) → z
```

Decoder `g_dec: ℝˡ → 𝒱^K` mirrors this:
```
z → Linear(l → d) → FFN → Linear(d → Kd) → reshape to K hidden states
  → per-position FFN → tied-embedding logits → argmax → K tokens
```

Lightweight: `d=512`, ~75M params total, negligible vs. the LM.

### 2.2 Why a plain AE fails

Pure reconstruction produces a **brittle** latent manifold. Tiny perturbations from the generative head decode to totally unrelated tokens. Four remedies are stacked:

**(a) Variational regularization.** Encoder outputs `(μ, σ)`; `z ~ 𝒩(μ, σ²I)`. Total loss:

```
ℒ_total = ℒ_ae + β · ℒ_KL           (β = 0.001)
ℒ_KL = -½ Σᵢ (1 + log σᵢ² − σᵢ² − μᵢ²)
```

Smooths the manifold. Encoder converges to σᵢ ≈ 0.3 → decoder must tolerate substantial noise.

**(b) KL clipping (Kingma et al. 2016)** prevents posterior collapse:

```
ℒ_KL^clip = Σᵢ max(λ_KL, ℒ_KL,ᵢ)    (λ_KL = 0.5)
```

Without this, ~71/128 dims collapsed to the prior in their experiments → catastrophic downstream.

**(c) Latent dropout** (p=0.15 on z) forces redundant encoding.

**(d) Input token dropout** (p=0.15, CBOW-style) forces semantic rather than index-level compression.

Both dropouts active **only during AE training**; disabled when training the LM.

### 2.3 Hyperparameters that matter

| K (chunk size) | latent dim `l` | token-level reconstruction |
|---|---|---|
| 4 | 10 (reconstruction-only) | >99.9% |
| 4 | **128** (full recipe) | >99.9% under σ≈0.3 noise |

Latent `l=128` is the sweet spot: l=32 too brittle, l=256 encodes noise that burdens the downstream LM. **Scaling the AE (more layers, larger d, 100B train tokens) did not help** — the task is inherently easy.

---

## 3. Component 2 — The Language Model (Next-Vector Prediction)

Sequence reframed:
```
X = (x₁,…,x_T) → Z = (z₁,…,z_L),   L = T/K
p(Z) = Πᵢ p(zᵢ | z<ᵢ)
```

But `p(zᵢ | z<ᵢ)` is intractable (uncountable support). Cannot use cross-entropy; cannot use perplexity.

### 3.1 Overall architecture

```
prev K tokens → embed → 2-layer MLP compressor → input vector (dim d)
             → Transformer backbone (LLaMA-style: RMSNorm, SwiGLU, RoPE)
             → hidden state h_{i-1}
             → Energy-based generative head → sampled z_i
             → frozen AE decoder → next K tokens → feedback
```

**Input is discrete, not continuous.** They tested feeding `z_{i-1}` directly — performance collapsed (Table 5: BrierLM 3.25 vs. 4.70 for discrete input). The compact latent is too brittle to serve as Transformer input; grounding the autoregressive loop in tokens is critical.

### 3.2 The Energy-based Generative Head

Purpose: single-step sample `z_i ~ p(· | h_{i-1})`. Diffusion / Flow Matching would work but require 10–100 iterative steps per vector — they'd erase the K-fold speedup.

**Inputs:** hidden state `h`, noise `ε ∈ ℝ^{d_noise}` with each dim `~ 𝒰[-0.5, 0.5]`.

**Body:** stack of `L` residual MLP blocks (~L/4 of Transformer depth, so head ≈ 10% of total params):

```
ε₀ = ε (projected)
For each block:
    ε_{l+1} = ε_l + SwiGLU( Linear_a(ε_l) + Linear_b(h) )
z = Linear(ε_L)         # projects to latent dim l
```

Each block ≈ 6d² params.

### 3.3 Training — The Energy Loss

Built from the **Energy Score** (Székely 2003), a strictly proper scoring rule:

```
S(P, y) = 𝔼_{x',x''~P} ‖x' − x''‖^α  − 2·𝔼_{x~P} ‖x − y‖^α      α ∈ (0, 2)
```

First term encourages **diversity** (penalizes mode collapse). Second term encourages **fidelity** (samples near ground truth). Strictly proper → maximized uniquely at P = Q.

Monte Carlo estimator used as loss. With `N` samples from the head and `M` targets from the AE posterior (not a single z — posterior is Gaussian, so drawing M reduces variance cheaply):

```
ℒ_energy = Σᵢ [ (2/(NM)) ΣₙΣₘ ‖z_{i,m} − z̃_{i,n}‖
              − (1/(N(N−1))) Σ_{n≠k} ‖z̃_{i,n} − z̃_{i,k}‖ ]
```

Defaults: **N=8** (model samples, expensive), **M=100** (target samples, nearly free), **α=1**. Ablations: α<1 fails (gradient explosion); α=2 collapses (only proper, not strictly proper).

Why likelihood-free matters: the loss only needs a **sampling interface** to the head. This is what lets the head be arbitrary (diffusion, flow, energy) with no architectural constraints.

---

## 4. Component 3 — BrierLM (Evaluation)

Perplexity requires explicit likelihoods → dead. Needed: a strictly proper, sample-based LM metric.

**Brier score** (Brier 1950):
```
Brier(P, y) = 2 P(y) − Σ_x P(x)²
```

Decomposes as `−Σ(P(x)−Q(x))² + const` → uniquely maximized at P=Q. Strictly proper.

**Likelihood-free estimator** using only two samples `x₁, x₂ ~ P`:
- `Σ_x P(x)²` = collision probability → estimated by `𝕀{x₁ = x₂}`
- `P(y)` → estimated by `𝕀{x = y}`

```
Brier(P, y) ≈ 𝕀{x₁ = y} + 𝕀{x₂ = y} − 𝕀{x₁ = x₂}
```

**Brier-n** extends to n-grams (treating the n-gram as the atomic outcome). Final composite, mirroring BLEU:

```
BrierLM = 100 · (Π_{n=1..4} Brier-n)^{0.25}
```

**Validation:** against cross-entropy across training checkpoints, Pearson r = −0.966, Spearman ρ = −0.991. Monotonically aligned with perplexity → trustworthy drop-in replacement.

**Bonus:** BrierLM works for any implicit generative model (discrete diffusion LMs, etc.) — no more fragile ELBO bounds.

---

## 5. Component 4 — Likelihood-Free Temperature Sampling

Standard temperature (rescale logits) needs explicit probabilities → dead. Goal: sample from `P_T(x) ∝ P(x)^{1/T}` given only a black-box sampler for P.

### 5.1 Exact algorithm (Algorithm 1)

Decompose `1/T = n + α`, `n = ⌊1/T⌋`, `α ∈ [0,1)`.

**Stage 1 (integer part):** draw `n` samples. If **all identical**, accept the value as candidate `x*`; else restart. Accept probability ∝ `P(x*)^n`.

**Stage 2 (fractional part — Bernoulli Factory, Mendo 2019):** simulate a coin of bias `P(x*)^α` given only a P-sampler:
```
i ← 1
loop:
    draw x ~ P
    if x = x*:        return x*              # accept
    else:
        draw u ~ 𝒰(0,1)
        if u < α/i:    restart whole process # reject
        else:          i ← i+1
```

**Theorem:** the acceptance probability is exactly `P(x)^n · P(x)^α = P(x)^{1/T}`, so accepted samples follow `P_T`. Proof uses the generalized binomial theorem:

```
p_stage2 = p · Σ_{k≥0} (1−p)^k Π_{j=1..k}(1 − α/j)
         = p · (p−1+1)^{α−1} = p^α
```

**Expected sampler calls:**
```
𝔼[N_total] = [ n + 𝕀(α>0) · Σ_x P(x)^{1/T − 1} ] / Z_T,   Z_T = Σ_x P(x)^{1/T}
```

Bounded by `1 + n/Z_T` for `T ≤ 0.5`, and by `1 + |𝒳|^{2−1/T}/Z_T` for `0.5 < T < 1`. → **Algorithm is only practical for low T**, and T→1 can blow up to the size of sample space `|𝒳| = |𝒱|^K`.

### 5.2 Practical approximation (Algorithm 2, low-T regime)

Drawing `n` identical samples in a row has vanishing probability → prohibitive rejection rate. Replace with **batch combinatorial search** at `T = 1/n`:

```
Draw batch ℬ of size N (N >> n)
Count c_x for each unique x
For m = n, n-1, …, 1:
    candidates = { x : c_x ≥ m }
    weights    = { (c_x choose m) }
    if candidates non-empty: break (fallback to smaller m)
Sample output from candidates weighted by (c_x choose m)
```

Biased at finite N, but **asymptotically unbiased**: `lim_{N→∞} P_alg(x; N) = P_T(x)`.

Proof sketch: `c_x/N →ᵖ P(x)` by WLLN; `W_x/N^n = (1/n!) Π_{j=0..n-1}(c_x−j)/N →ᵖ P(x)^n / n!`; ratios of these converge to `P(x)^n / Σ P(z)^n = P_T(x)`. Bounded Convergence Theorem turns convergence-in-probability into convergence-of-expectation.

### 5.3 Empirical behavior

Two knobs govern the accuracy-diversity trade-off:
- **Batch size N** (stronger effect): larger N sharpens distribution (higher accuracy, higher collision rate)
- **Temperature T** (weaker effect, capped by batch info content)

Matching a Transformer's T ≈ 0.6 requires N ≈ 100; T ≈ 0.5 requires N ≈ 200. The CALM N-curve traces the same accuracy-diversity frontier as a Transformer's T-curve.

---

## 6. Experimental Results

**Setup:** Pile uncopyrighted (~230B tokens, LLaMA 3 tokenizer), eval on WikiText-103. Four scales S/M/L/XL. Context length = 2048 **steps** (so 2048·K tokens for CALM). 250k steps, 2M token batches, AdamW, lr=3e-4.

### 6.1 Main numbers (K=4)

| Model | Params | Train FLOPs (×10²⁰) | Infer FLOPs/tok (×10⁸) | BrierLM |
|---|---|---|---|---|
| Transformer-S | 281M | 6.6 | 4.4 | 6.05 |
| Transformer-M | 465M | 11.9 | 7.9 | 7.07 |
| Transformer-L | 849M | 22.5 | 15.0 | 8.98 |
| **CALM-M (K=4)** | 371M | **3.7** | **2.9** | 5.72 |
| **CALM-L (K=4)** | 735M | **7.7** | **4.6** | 6.58 |
| **CALM-XL (K=4)** | **1.82B** | 19.5 | 9.4 | 8.53 |

CALM-M ≈ Transformer-S at **44% fewer train FLOPs, 34% fewer infer FLOPs**. CALM-XL approaches Transformer-L at less compute. All CALM numbers *include* AE overhead (75M params + encode/decode FLOPs).

### 6.2 Effect of K

- `K=1`: CALM **underperforms** a discrete Transformer at matched compute → the continuous prediction task is intrinsically harder at the same sequence length. This is the headline caveat.
- `K=2`: ≈ halves cost, marginal BrierLM drop.
- **`K=4`: surpasses the Transformer performance–compute frontier.** Sweet spot.
- `K=8`: performance drops — likely a capacity ceiling; hypothesis is that bigger models are needed to exploit higher bandwidth.

### 6.3 Generative head comparison

| Head | Inference steps | BrierLM |
|---|---|---|
| Diffusion (100 steps) | many | lowest |
| Flow Matching midpoint | 4 ≈ optimal | middle |
| **Energy (single-step)** | **1** | **highest ceiling** |

### 6.4 Training dynamics

CALM-XL shows a slower initial curve (continuous prediction is hard to bootstrap) but overtakes the plateau that flat Transformers hit — its large parameter count eventually pays off where a traditional LM saturates.

---

## 7. Positioning

| Approach | Chunks? | Continuous? | Single-step gen? | Parallel/AR |
|---|---|---|---|---|
| BPE Transformer | ✗ | ✗ | ✓ | AR token-by-token |
| MegaByte | ✓ (bytes) | ✗ | ✗ (inner AR loop) | AR + AR |
| Large Concept Models | ✓ (sentences) | ✓ (SONAR) | ✗ (diffusion) | AR |
| GIVT (Tschannen 2023) | ✓ | ✓ (GMM) | ✓ | AR |
| Li 2024 (diffusion head) | ✓ | ✓ | ✗ | AR + diffusion |
| **CALM** | ✓ (K tokens) | ✓ | ✓ (energy) | AR vector-by-vector |

Nearest neighbors: Large Concept Models (uses SONAR + diffusion; heavy + iterative), Li 2024 (per-vector diffusion head; iterative). CALM distinguishes by combining **lightweight AE + robust latent + single-step energy head + discrete input feedback**.

---

## 8. Reference Implementation Sketch

```python
# ----- Autoencoder (train alone first) -----
class AE(nn.Module):
    def __init__(self, K=4, d=512, l=128, vocab=128256):
        self.emb = nn.Embedding(vocab, d)
        self.enc_ffn1 = PositionWiseFFN(d)
        self.enc_comp = nn.Linear(K*d, d)
        self.enc_ffn2 = FFN(d)
        self.enc_mu   = nn.Linear(d, l)
        self.enc_logv = nn.Linear(d, l)
        self.dec_in   = nn.Linear(l, d)
        self.dec_ffn1 = FFN(d)
        self.dec_exp  = nn.Linear(d, K*d)
        self.dec_ffn2 = PositionWiseFFN(d)
        self.K = K

    def encode(self, x):                    # x: (B, K)
        h = self.enc_ffn1(self.emb(x))      # (B, K, d)
        h = self.enc_comp(h.flatten(1))     # (B, d)
        h = self.enc_ffn2(h)
        return self.enc_mu(h), self.enc_logv(h)

    def decode(self, z):
        h = self.dec_ffn1(self.dec_in(z))
        h = self.dec_exp(h).view(-1, self.K, d)
        h = self.dec_ffn2(h)
        return h @ self.emb.weight.T        # tied logits

    def forward(self, x):
        mu, logv = self.encode(x)
        std = (0.5*logv).exp()
        z   = mu + std * torch.randn_like(std)   # reparam
        # optional z-dropout (train only)
        return self.decode(z), mu, logv, z

# loss: CE(reconstruction) + β·KL_clipped

# ----- Energy generative head -----
class EnergyHead(nn.Module):
    def __init__(self, d, l, n_blocks, d_noise):
        self.in_eps = nn.Linear(d_noise, d)
        self.blocks = nn.ModuleList([EnergyBlock(d) for _ in range(n_blocks)])
        self.out    = nn.Linear(d, l)

    def sample(self, h, n_samples=1):
        B = h.size(0)
        h_rep = h.unsqueeze(1).expand(-1, n_samples, -1).reshape(B*n_samples, -1)
        eps   = (torch.rand(B*n_samples, d_noise) - 0.5)
        e = self.in_eps(eps)
        for blk in self.blocks:
            e = blk(e, h_rep)                # residual MLP fusing h
        return self.out(e).view(B, n_samples, -1)

# ----- CALM -----
class CALM(nn.Module):
    def __init__(self, ae, K=4, d=1024, ...):
        self.ae = ae                         # frozen after AE pretraining
        self.in_compress = MLP2(K*d, d)
        self.backbone    = LLaMATransformer(d, n_layers, ...)
        self.head        = EnergyHead(d, l=ae.l, n_blocks=n_layers//4, d_noise=d)

    def step(self, prev_tokens):             # prev_tokens: (B, K)
        emb = self.ae.emb(prev_tokens).flatten(1)
        x   = self.in_compress(emb)
        h   = self.backbone(x)               # causal
        return h                             # pass h to head for sample/loss

    def energy_loss(self, h_seq, z_target_samples):
        # h_seq: (B, L, d); z_target_samples: (B, L, M, l) from AE posterior
        N, M = 8, 100
        z_hat = self.head.sample(h_seq.flatten(0,1), n_samples=N) \
                         .view(B, L, N, -1)
        # pairwise distances: ||z_m - z̃_n|| and ||z̃_n - z̃_k||
        term_fid = (z_target_samples.unsqueeze(3) - z_hat.unsqueeze(2)) \
                   .norm(dim=-1).mean(dim=(2,3))
        term_div = pairwise_dist(z_hat).mean(dim=-1)   # excluding diagonal
        return (2*term_fid - term_div).sum()
```

### Invariants an implementer must respect

1. AE trained first, then frozen. No gradients from LM loss to AE.
2. Reparameterization trick only during AE training; at CALM training time, use `μ` or re-sample from the posterior as targets (they sample M targets per step).
3. Energy-loss gradients flow through `z̃_n` (model samples) only; the `z_target` samples are detached.
4. Feedback loop at inference: `z_i → g_dec → argmax → K tokens → next input`. Discrete path is non-differentiable but inference-only.
5. Noise ε for the head is drawn independently per sample, not shared.

---

## 9. Takeaways & Open Problems

**What CALM establishes:**
- A fourth scaling axis (semantic bandwidth K) exists and is empirically exploitable.
- A fully likelihood-free stack (training + eval + sampling) is feasible and produces results consistent with likelihood-based evaluation (r = −0.97 with cross-entropy).
- Energy-score training + Bernoulli-factory-based sampling give principled alternatives where softmax can't reach.

**What it doesn't:**
- K=1 underperforms the baseline → the continuous objective has a learnability tax that hasn't been fully amortized.
- Only tested at ≤1.82B params; scaling laws as a function of (N, D, K) remain to be mapped.
- AE is context-free. Context-aware AEs could raise both fidelity and semantic coherence.
- Reinforcement learning (needs log-probs) and KL-distillation (needs PMFs) need to be rewritten for sample-based regimes.
- Temperature algorithm is expensive near T=1 (rejection scales with |𝒱|^K).

**Most transferable artifacts:**
1. **BrierLM** — usable today for any implicit LM (discrete diffusion, energy-based, etc.).
2. **Likelihood-free temperature sampling** — works for any black-box discrete sampler.
3. **Robust VAE recipe** (β=0.001, KL clip λ=0.5, dual dropout) — replicable for any latent-vector LM.
