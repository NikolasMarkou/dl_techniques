# Orthonormal Regularization *with* Centering Normalization and feature eselection

> **TL;DR**   We revisit orthonormal‐regularised layers and show that **mean‑centering (without variance scaling)** is the natural companion to a Stiefel‑constrained weight matrix.  Logit (L2) Normalization projects the centred features onto the unit hypersphere, while per‑channel scales in \([0,1]\) provide sparse, interpretable attention.

---

## 1  Introduction & Background

### 1.1 Why normalise?
Modern deep nets rely on normalisation—BatchNorm [1], LayerNorm [2], GroupNorm [3]—to tame internal covariate shift, speed convergence, and stabilise gradients.  Most techniques enforce **both** zero mean and unit variance; yet several works (e.g. Weight Standardisation [4]) note that *mean removal alone* can already confer large benefits.

### 1.2 Why orthonormal weights?
Enforcing \(W^{\top}W≈I\) keeps the singular spectrum tight, prevents exploding/vanishing signals, and preserves Euclidean geometry.  Orthogonal regularisers or Stiefel parametrisations have improved GAN training [5], metric learning [6], and robust classifiers [7].  However, when paired with full variance‑normalisation, the two constraints can fight—**both try to govern scale**.

### 1.3 Our premise
If weights are already well‑conditioned, then only **mean** needs normalising downstream; variance can be left to the network to exploit until the final L2 projection.  We therefore propose a block:

```text
Dense (Ortho) → Centering Norm → Logit Norm → Scale [0,1]
```

and provide a geometric + gradient‑level analysis alongside implementation tips and an empirical testbed.

---

## 2  Revised Architecture

```text
Dense → Soft Orthonormal Reg. → Centering Norm → Logit (L2) Norm → Scale [0,1]
```

| Stage | Purpose | Key Constraint |
|-------|---------|----------------|
| **Dense + Ortho** | Well‑conditioned linear map | \(W^\top W≈I\) |
| **Centering Norm** | Zero‑mean activations | \(\mu=0\) |
| **Logit Norm** | Unit‑norm activations | \(\|z'\|_2=1\) |
| **Scale [0,1]** | Sparse, learnable attention | \(0≤s_c≤1\) |

---

## 3  Component Walk‑through

### 3.1 Dense Layer with Soft Orthonormal Regularization

\[
\mathcal L_{\text{ortho}}\;=\;\lambda(t)\,\big\|W^{\top}W-I\big\|_{\mathrm F}^2
\]

*Moves weights towards the **Stiefel manifold** while allowing gradual relaxation via a schedule \(\lambda(t)\).*  

```
┌ Weight Space ┐      ┌ Stiefel ┐
│  · ·         │  →   │  · ·   │
│  · ·         │      │  · ·   │
└──────────────┘      └────────┘
```

### 3.2 Centering Normalization (mean‑only)

\[
\text{CN}(\mathbf x)=\mathbf x-\boldsymbol\mu,\qquad \boldsymbol\mu=\frac1n\sum_{i=1}^n\mathbf x_i
\]

Preserves *variance* yet aligns the cloud with the origin.

### 3.3 Logit (L2) Normalization

\[
\text{LN}_{\text{logit}}(\mathbf x)=\frac{\mathbf x}{\|\mathbf x\|_2+\varepsilon}
\]

Projects centred vectors onto the **unit hypersphere**.

### 3.4 Per‑channel Scale \([0,1]\)

*Two common parameterisations*

| Method | Forward | Gradient near limits |
|--------|---------|----------------------|
| **Hard clip** | \(y=s_{\text{clip}}\,x\) | Zero at boundaries (may stall) |
| **Sigmoid** | \(y=\sigma(u)\,x\) | Smooth; vanishes only asymptotically |

Produces soft attention and potential sparsity.

---

## 4  End‑to‑end Signal Path

```text
x ──Dense+Ortho──▶ z   ─Centering──▶ z'  ─L2──▶ z" ─Scale──▶ y
      W^⊤W≈I          μ=0             ‖·‖₂=1          0≤s≤1
```

---

## 5  Gradient Highlights

* **Scale gate** &nbsp;\(∂L/∂z''_c=∂L/∂y_c·s_c\)
* **Hypersphere coupling** &nbsp;Competitive Jacobian  
  \(I/\|z'\|_2-\frac{z'z'^\top}{\|z'\|_2^3}\).
* **Centering Jacobian** &nbsp;Rank‑1 adjustment ensuring gradient sums to zero.
* **Ortho term** &nbsp;\(4\,\lambda(t)\,W(W^\top W-I)\) keeps weights well‑conditioned.

```
∂L/∂y → ∂L/∂z" → ∂L/∂z' → ∂L/∂z → ∂L/∂W → ∂L/∂x
        scale‑gate   sphere‑comp   rank‑1       + ortho
```

---

## 6  Layer Norm vs Centering Norm (Quick Contrast)

| Aspect | **Layer Norm** | **Center‑only (ours)** |
|--------|----------------|------------------------|
| Mean | Forced 0 | Forced 0 |
| Variance | Forced 1 | *Preserved* |
| Redundancy with Ortho | High | Low |
| Gradient Coupling | Full covariance | Rank‑1 (milder) |

---

## 7  Dynamic Orthogonality Schedule

Cosine anneal example:

\[
\lambda(t)=\lambda_{\max}\;\frac{1+\cos\big(\pi\,\mathrm{min}(t/T_{\text r},1)\big)}{2}
\]

Keeps conditioning early, relaxes constraints late.

---

## 8  Variance Drift & Mitigations

* **Monitor** per‑channel variance & CoV across blocks.
* **Mitigate** via occasional full Layer Norm, residual links, or variance‑based conditional norm.

---

## 9  Implementation Notes on Scale \(s\)

* *Sigmoid* preferred for continuous gradients.
* Hard‑clip can yield true zeros → genuine sparsity (but risk of dead channels).

---

## 10  Suggested Empirical Protocol

1. **Ablations**: Baseline → +Ortho → +Center → +L2 → +Scale.  
2. **Metrics**: accuracy, convergence speed, gradient norms, \(\|W^\top W-I\|_F\), scale histograms.
3. **Plots**: variance drift, orthogonality curve, scale evolution.

---

## 11  Conclusion

*Mean‑centering plus Stiefel‑regularised weights yields the **minimum‑necessary** normalisation for geometrically stable, directional representations—avoiding the redundancy of full Layer Norm while preserving variance information.*

> **Outcome:** A compact, interpretable block for tasks that value directionality, sparsity, and transfer‑friendly embeddings—awaiting empirical confirmation.

---

## Appendix A  Symbol & Shape Reference

| Symbol | Meaning | Shape (sample) | Shape (batch) |
|--------|---------|----------------|---------------|
| \(x\) | input | \([F]\) | \(\mathbf x\,[B×F]\) |
| \(W\) | weight matrix | \([F_{\text{in}}×F_{\text{out}}]\) | same |
| \(z,z',z''\) | intermediate activations | \([F_{\text{out}}]\) | \(\mathbf z\,[B×F_{\text{out}}]\) |
| \(y\) | output | \([F_{\text{out}}]\) | \(\mathbf y\,[B×F_{\text{out}}]\) |
| \(s\) | scale (after clip/σ) | \([F_{\text{out}}]\) | shared |
| \(\mu\) | mean vector | \([F_{\text{out}}]\) | \(\boldsymbol\mu\,[B×F_{\text{out}}]\) |
| \(\lambda(t)\) | ortho penalty | scalar | scalar |
| \(I\) | identity | \([F_{\text{out}}×F_{\text{out}}]\) | — |

---

### References
[1] Ioffe & Szegedy, *Batch Normalization*, 2015.  
[2] Ba, Kiros, Hinton, *Layer Normalization*, 2016.  
[3] Wu & He, *Group Normalization*, 2018.  
[4] Qiao et al., *Weight Standardization*, 2019.  
[5] Brock et al., *Large‑Scale GANs*, 2021.  
[6] Wang et al., *Orthogonal Constraints in Metric Learning*, 2017.  
[7] Huang et al., *Orthogonal Training for Robustness*, 2021.

