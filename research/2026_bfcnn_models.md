# What an Optimal Bias-Free Image Prior Unlocks

*A comprehensive review of capabilities and applications enabled by a near-optimal, bias-free image-score model built on Miyasawa's principle.*

**Date:** 2026-06-28
**Scope:** Foundations, capability primitives, an application catalogue (established → speculative), honest failure modes, and a concrete build roadmap for this repository.

---

## Bottom line

Suppose we have trained the best possible *bias-free* image denoiser. Three facts then hold simultaneously:

1. **It is a denoiser.** Given a noisy image at any noise level, it returns the cleanest possible estimate.
2. **It is a score model.** By Miyasawa's 1961 identity (the same relation modern work calls Tweedie's formula), the denoiser's *residual* — what it removes — is, up to a scale factor, the gradient of the log-probability of images: it points "uphill" toward more probable images. This gradient field is called the **score**, and it is the practically usable form of `P(x)`.
3. **It is a generative prior.** Because we can read the score at every point and every noise level, we can walk uphill on the probability landscape. That single ability lets us *sample* new images, *restore* degraded ones, *score* how natural an image is, and *detect* what does not belong — all from one model, with no task-specific retraining.

The honest catch: we never get the *normalized* probability `P(x)` itself (the partition function is intractable). We get its gradient. Almost everything useful needs only the gradient, but a few tempting applications (notably naive likelihood-based outlier detection) are exactly the ones the gradient does not cleanly support — and the literature shows they fail. This review separates the two.

---

## Part 1 — The foundation

### 1.1 What "bias-free" means and why it matters

A standard convolutional network computes, at each layer, `activation(W·x + b)` and, in batch-norm layers, an affine shift `(x − μ)/σ · γ + β`. A **bias-free** network removes every additive constant: no `b` term in convolutions, no `β` shift in batch-norm (the multiplicative scale `γ` stays).

The consequence is a clean mathematical property. With ReLU activations and no additive constants, the whole network is **homogeneous of degree one**:

```
f(α·y) = α·f(y)   for any scalar α ≥ 0
```

Locally, this means the denoiser behaves like a linear map through the origin:

```
f(y) = A(y)·y
```

where `A(y)` is an input-dependent matrix (the network's Jacobian). This single property buys a remarkable empirical result, reported by Mohan, Kadkhodaie, Simoncelli & Fernandez-Granda (ICLR 2020, *Robust and Interpretable Blind Image Denoising via Bias-Free CNNs*, arXiv:1906.05478):

- Train a conventional denoiser on a narrow noise range (e.g. standard deviation σ ∈ [0, 10]) and test it at σ = 90, and it falls apart — PSNR collapses, the image turns to artefacts. The reason is that the accumulated bias terms grow without bound once you leave the training regime.
- Train the **bias-free version of the same architecture** on the same narrow range, and it holds state-of-the-art quality across the *entire* noise range it never saw. Because there are no additive constants, scaling the input simply scales the output; there is no fixed noise assumption baked in.

This is what makes the model a *universal* prior rather than a noise-level-specific filter. It is the precondition for everything below.

### 1.2 Miyasawa's identity: the residual IS the score

Consider an image `x` corrupted by additive white Gaussian noise:

```
y = x + n,   n ~ Normal(0, σ²·I)
```

The optimal (minimum mean-squared-error) denoiser returns the posterior mean `E[x | y]`. Miyasawa (1961) — anticipated by Robbins (1956) and popularized in modern statistics as Tweedie's formula (Efron 2011) — proved that this posterior mean can be written purely in terms of the gradient of the log-density of the *noisy* images:

```
E[x | y] = y + σ²·∇_y log p(y)
```

Rearranged, this is the workhorse identity of the entire field:

```
∇_y log p(y)  =  ( f(y) − y ) / σ²
                 └────┬────┘
              the denoiser's residual
```

In words: **what the denoiser subtracts from the image, divided by the noise variance, is the score** — the direction of steepest increase in log-probability. The denoiser never computes a probability; it computes a gradient field over image space. That gradient field is the implicit prior.

Two clarifications that prevent confusion:

- **We get the score, not the density.** There is no normalized `P(x)` number you can read off. You get `∇ log p`, a vector at each pixel. Computing the actual probability would require the intractable normalizing constant.
- **It is a family of priors, one per noise level.** The model knows `∇ log p_σ` for every σ — the log-density of images *blurred by Gaussian noise of width σ*. Small σ gives a sharp, high-detail prior; large σ gives a smooth, coarse prior. This ladder of noise levels is exactly what makes annealed sampling work (Section 2.3).

This residual=score identity is also precisely the training target of **denoising score matching** (Vincent 2011) and the core of every modern **diffusion model** (Ho, Jain & Abbeel 2020; Song & Ermon 2019; Song et al. 2021). A diffusion model *is* a denoiser trained across noise levels; the "epsilon prediction" it learns is the score in disguise (`ε_θ(x_t,t) = −√(1−ᾱ_t)·∇log p_t(x_t)`). So our hypothetical model sits at the exact mathematical centre of contemporary generative AI.

### 1.3 What the model "knows" — geometry, not pixels

Three deeper findings characterize what an optimal version of this model actually represents:

- **Adaptive manifold projection.** Because the bias-free denoiser is locally `A(y)·y`, its Jacobian can be inspected directly. It turns out to act as a *projection onto a low-dimensional subspace* — the local tangent plane of the "manifold" of natural images. The effective dimension of that subspace shrinks as noise grows (roughly `d ≈ α/σ`): at high noise the model commits only to coarse structure; at low noise it preserves fine detail. This is why one network spans all noise levels.

- **Geometry-adaptive harmonic bases.** Kadkhodaie, Guth, Simoncelli & Mallat (ICLR 2024 Outstanding Paper, *Generalization in diffusion models arises from geometry-adaptive harmonic representations*, arXiv:2310.02557) showed the optimal denoiser performs shrinkage in bases that oscillate along image contours and in smooth regions — close to the mathematically optimal bases (bandlets/curvelets) for natural images. The model has effectively *discovered* the right representation for image geometry.

- **It generalizes, it does not memorize.** The same paper demonstrated a sharp threshold: train two networks on *non-overlapping* halves of a large dataset (≥ ~100,000 images) and they converge to nearly the *same* denoising function and score field (cosine similarity > 0.99), producing near-identical samples. Below ~100 images they memorize; above ~10⁵ they learn a *shared, true* prior. So "the best possible P(x)" is a well-defined target: at scale, all sufficiently good denoisers converge to the same underlying image prior. This is what justifies treating the model as an estimate of a real, objective distribution rather than an artefact of one training run.

---

## Part 2 — Capability primitives

Everything the model can do reduces to a small set of primitive operations. The applications in Part 3 are combinations of these.

### 2.1 Read the score (gradient toward more-probable images)

`g(y, σ) = (f(y) − y)/σ²`. A vector field over image space. Tells you, from any image, which way to nudge each pixel to make the image more "natural" at scale σ. This is the atomic operation; the rest are built on it.

### 2.2 Denoise / posterior-mean estimate

`f(y)` itself: the best guess of the clean image behind a noisy one, at any noise level, blind. Useful directly, and as the inner loop of everything else.

### 2.3 Sample new high-probability images (annealed gradient ascent)

Start from pure noise; repeatedly step uphill along the score while slowly lowering the noise level σ from large to small. This is **annealed Langevin dynamics** (Song & Ermon 2019) or, equivalently, the reverse diffusion process (Song et al. 2021):

```
y_t = y_{t-1} + h_t·f(y_{t-1}) + γ_t·z_t,    z_t ~ Normal(0, I)
σ_t² = (1 − β·h_t)²·σ_{t-1}²          (noise annealed downward)
```

The coarse-to-fine schedule is what makes it work: at high σ the landscape is smooth and easy to climb; as σ drops, detail is filled in. Output is a fresh, plausible image drawn from the learned prior.

### 2.4 Solve any linear inverse problem — without retraining

This is the headline result of Kadkhodaie & Simoncelli (NeurIPS 2021, *Stochastic Solutions for Linear Inverse Problems using the Prior Implicit in a Denoiser*, arXiv:2007.13640). Given measurements `y_c = M·x` (M is any linear operator: blur, downsampling, a sampling mask, a sensing matrix), run the same uphill-sampling loop but split the gradient into two parts — the prior pulling toward natural images, and a data term pinning the result to the measurements:

```
σ²·∇ log p(y | x_c) = (I − MᵀM)·f(y)  +  M·(x_c − Mᵀy)
                       └ prior in the ─┘    └ data fidelity ─┘
                          null space            in measured space
```

**One trained denoiser solves deblurring, super-resolution, inpainting, compressed sensing, randomly-missing-pixel recovery, and more — all by changing M, with zero task-specific training.** Reported results include 4× super-resolution (Set5 PSNR 29.47/SSIM 0.894), spectral super-resolution from 5% of frequencies (Set5 PSNR 29.22), and compressed sensing from 10% random measurements (PSNR 25.47, beating ISTA-Net) — and roughly two orders of magnitude faster than the older Deep Image Prior approach (~9 s vs ~1,200 s). This is the same idea as **Plug-and-Play Priors** (Venkatakrishnan, Bouman & Wohlberg 2013) and **Regularization by Denoising / RED** (Romano, Elad & Milanfar 2017), which drop a denoiser into ADMM / proximal solvers as an implicit regularizer.

### 2.5 Project onto the natural-image manifold

A single denoise (or a short uphill walk at small σ) snaps an arbitrary image toward the nearest plausible natural image. The *residual* of that projection — how far the image had to move, and where — is itself an information-rich signal (used for detection, Section 3.3).

### 2.6 Measure "naturalness" via score magnitude

`‖∇ log p(x)‖` is small where an image is typical and large where it is atypical (the gradient is steep on the slopes leading down to improbable regions). This gives a *reference-free* naturalness signal — with important caveats (Section 4).

### 2.7 Inspect the local linear filter (Jacobian)

Because the model is locally `A(y)·y`, you can extract `A(y)` and read off, for any image, exactly which neighbouring pixels were averaged to produce each output pixel — an interpretable, structure-adaptive filter. The bias-free property is what makes this analysis exact rather than approximate. Useful for science (what features does the model rely on?) and for debugging/trust.

### 2.8 Define a perceptual distance / metric on image space

The score field induces a Riemannian geometry: directions off the manifold are "expensive," directions along it are "cheap." This yields manifold-aware distances and geodesics between images — a principled alternative to pixel-wise Euclidean distance for interpolation, retrieval, and morphing.

---

## Part 3 — Application catalogue

Organized by maturity: **A) Established** (demonstrated in the literature), **B) Emerging** (early results, active research), **C) Speculative-but-plausible** (novel combinations inferred from the primitives; flagged honestly).

### 3.A Established applications

#### A1. Universal image restoration
Deblurring, denoising at any level, super-resolution, inpainting, JPEG-artefact removal, old-photo restoration, missing-pixel/scratch repair. All are instances of Section 2.4 — pick the operator M, run the sampler. The selling point over task-specific restoration networks: **one model, every task, no retraining**, plus a *distribution* of plausible reconstructions (you can sample several and show uncertainty) rather than one blurry average.

#### A2. Medical image reconstruction
- **Accelerated MRI** from undersampled k-space — the operator M is a Fourier transform with a sampling mask. Recent work (*Noisy MRI Reconstruction via MAP Estimation with an Implicit Deep-Denoiser Prior*, arXiv:2511.11963) uses exactly the Miyasawa residual as the prior gradient and reports strong behaviour at 8× acceleration, preserving anatomy while resisting scanner-noise amplification.
- **Sparse-view / low-dose CT**, **PET with limited counts**, **photon-limited microscopy** — same template, different M. A prior trained once on anatomy serves all acceleration factors and sampling patterns, which is operationally valuable because protocols change constantly.

#### A3. Scientific & computational imaging
- **Astronomy:** deconvolving the telescope point-spread function while preserving faint structure (e.g. *PI-AstroDeconv*, arXiv:2403.01692).
- **Computational microscopy / optical sectioning, phase retrieval, fluorescence** under photon starvation.
- Generally: any field with a known forward physics operator and expensive/scarce measurements is a fit.

#### A4. Unconditional and conditional image generation
The model is, by construction, a diffusion-model generator (Section 2.3). It can synthesize novel images, and with the inverse-problem machinery it does class- or measurement-conditioned generation. This is the mainstream of generative AI; our model is its mathematical core, with the bias-free property as a robustness bonus.

#### A5. Adversarial purification
**DiffPure** (Nie et al., ICML 2022) adds a little noise to an adversarially-perturbed input and runs the reverse process to "wash out" the attack before classification, with no need to retrain or even know the classifier. Robustness comparable to adversarial training. Our bias-free model is a drop-in purifier. (Caveats: high latency; weaker on colour-space attacks — Section 4.)

### 3.B Emerging applications

#### B1. Anomaly / defect detection (the *right* way)
Train the prior on "normal" data only (healthy scans, defect-free product images). At test time, the **projection residual** (Section 2.5) — how much the image had to be changed to look normal — localizes anomalies pixel-by-pixel. Demonstrated in industrial inspection (AUROC ~99.7% with composite scoring) and medical anomaly localization (Zhou et al., MICCAI 2022). Note this uses *reconstruction error*, not raw likelihood — which is why it works where likelihood fails (Section 4).

#### B2. Reference-free image-quality assessment
The score-magnitude naturalness signal (Section 2.6) generalizes the classic NIQE natural-scene-statistics metric (Mittal et al. 2013) — but with a far richer, learned prior instead of a hand-fit Gaussian model. Potential: a no-reference quality score that flags compression banding, ringing, or over-sharpening as regions of high score magnitude. Promising but not yet a validated standalone metric.

#### B3. AI-generated / deepfake image detection
**DIRE** (Wang et al., ICCV 2023) exploits a subtle asymmetry: diffusion-generated images are reconstructed by a diffusion model with *lower* error than real images (they already live on the model's manifold). The reconstruction-error map is a near-perfect detector (99.9% accuracy on diffusion-generated images, generalizing across generators). Frequency-guided variants (FIRE, CVPR 2025) add robustness. Our model gives the cleanest possible reconstruction operator for this test. **Caveat:** confounded by JPEG compression and defeated by attackers who know the detector.

#### B4. Tamper / forgery localization
Splices and edits leave regions that sit slightly off the natural-image manifold; the local projection residual highlights them. Multi-scale forensic frameworks report ~87% accuracy across mixed manipulation types with pixel-level localization (F1 ~0.76).

#### B5. Generative / perceptual compression
A learned prior *is* an entropy model: the more probable an image, the fewer bits to code it (the Ballé–Minnen hyperprior lineage, NeurIPS 2018). Score-based decoders (Hoogeboom et al., arXiv:2305.18231) beat prior state-of-the-art (HiFiC, PO-ELIC) on perceptual quality (FID) at a given bitrate by *generating* plausible detail the bitstream cannot afford to store. Our model can serve both as the entropy model and as the generative decoder. **Caveat:** diffusion decoding is slow; one-step-distillation research (OneDC) is closing the gap.

### 3.C Speculative-but-plausible applications

These are novel combinations of the primitives. Each is grounded in the mechanics above, but none is a solved result — they are research bets, flagged as such with the specific risk.

#### C1. Prior-guided active sensing ("smart capture")
Combine the prior with optimal-experiment-design: at each step, choose the *next* measurement (next MRI k-space line, next microscope scan position, next camera exposure) that most reduces the prior's uncertainty about the reconstruction. The prior tells you what you already "know" about likely images, so you spend your measurement budget only where the image is genuinely unpredictable. Compressed-sensing theory predicts large gains (≈ log n SNR improvement for sparse signals; Axelsson & Wakin 2014). *Risk:* adaptive measurement adds per-step compute and assumes you can query the sensor sequentially.

#### C2. A universal "naturalness regularizer" for other models
Use the score as a plug-in loss term to keep *any* other system's output on the natural-image manifold: stabilize GAN training, regularize neural-radiance-field / 3D reconstructions, constrain video-frame interpolation, or post-process the output of a cheap fast model to remove its characteristic artefacts. The prior becomes a reusable "is this a real image?" gradient that any pipeline can call.

#### C3. Physically-grounded data augmentation and simulation
Sample *conditioned* images (Section 2.4) to generate realistic training data for downstream tasks where real data is scarce or privacy-constrained (rare medical findings, edge-case driving scenes). Because the prior is calibrated to real-image statistics, the synthetic data sits on the true manifold rather than in a GAN's mode-collapsed corner.

#### C4. Image "typicality fingerprinting" for dataset curation
Run a whole dataset through the score-magnitude signal to rank images by how typical they are. Surface mislabeled, corrupted, duplicated, or out-of-domain examples at the tails. A cheap, unsupervised data-quality auditor. *Risk:* score magnitude conflates "rare but valid" with "corrupt" — needs human review at the tails.

#### C5. Perceptual interpolation, morphing, and editing via manifold geodesics
Use the induced metric (Section 2.8) to interpolate between two images *along the manifold* — every intermediate frame is a plausible image, not a ghosted cross-fade. Enables semantically smooth morphing, attribute editing (walk along a learned direction), and controllable image blending without a separately trained editing model.

#### C6. Steganalysis and watermark/forensic analysis
Hidden payloads and some watermarks perturb an image slightly off the natural manifold in structured ways; the projection residual and score field are a natural substrate for detecting or localizing them, and for distinguishing benign compression noise from deliberate embedding. Largely unexplored with bias-free score priors.

#### C7. Scientific discovery via the learned representation
The geometry-adaptive bases the model discovers (Section 1.3) are themselves an object of study: inspecting the Jacobian's eigenvectors across image classes could reveal what statistical regularities define a domain (faces vs. galaxies vs. tissue), feeding back into compression codecs, perceptual models of human vision, and theory of natural-image statistics.

---

## Part 4 — Limits and failure modes (read before believing the hype)

An honest review must mark the boundaries. These are not minor caveats; they determine which applications are real.

### 4.1 You do not get a usable likelihood number
The model gives `∇ log p`, not `p`. Computing the normalized probability needs the intractable partition function. This directly kills the most naive application:

**Likelihood-based outlier detection does not work.** Nalisnick et al. (ICLR 2019, *Do Deep Generative Models Know What They Don't Know?*) showed deep generative models assign *higher* likelihood to clearly out-of-distribution data than to their own training data (the classic example: a model trained on CIFAR-10 rates SVHN images as *more* probable). So you cannot just threshold a likelihood to catch anomalies. The workarounds that *do* work are **typicality tests** (is this in the model's typical set, not just high-likelihood?) and **reconstruction/projection residuals** (B1, B3, B4) — which is why those applications succeed while raw-likelihood OOD fails. Score *magnitude* is a better signal than likelihood but still conflates "rare-but-real" with "corrupt."

### 4.2 Additive Gaussian noise only
The entire residual=score identity is derived for **additive white Gaussian noise**. It has no clean analogue for multiplicative noise, Poisson (shot) noise, or structured corruption. (This repository's own notes confirm: per-pixel multiplicative noise has no clean linear-domain residual=score identity; the bias-free construction is additive-only.) Inverse problems with non-Gaussian noise models need extra care and may break the clean theory.

### 4.3 Sampling is slow and the schedule is fiddly
Annealed Langevin / reverse-diffusion sampling and the inverse-problem solver take many iterations (tens to thousands of network evaluations). Real-time use (adversarial purification in a live system, interactive editing) is currently impractical without distillation/acceleration. Convergence on the curved image manifold lacks the clean guarantees of the Euclidean case.

### 4.4 Convergence and optimality are not guaranteed
Plug-and-play / RED solvers converge cleanly only under denoiser conditions (non-expansiveness, local homogeneity — the latter is exactly what bias-free buys, which is a point in its favour). A non-ideal denoiser can make the inverse-problem iteration diverge or settle on artefacts.

### 4.5 Domain match is everything
A prior trained on natural photos does not transfer to medical, satellite, or infrared imagery — it will confidently "restore" the wrong statistics. Every application implicitly assumes the prior was trained on the right distribution. Cross-domain transfer is largely unproven.

### 4.6 Detection methods are attackable
DIRE/forensic detectors (B3, B4) are confounded by ordinary lossy compression and can be defeated by an adversary who knows the detector. They are useful signals, not tamper-proof guarantees.

### 4.7 "Best possible" is an idealization
The convergence-to-a-shared-prior result (Section 1.3) needs ~10⁵+ images and large models; the "optimal" denoiser is an asymptotic target, not something you reach with a small dataset. Below that scale the model memorizes, and its "prior" is its training set.

---

## Part 5 — Why *bias-free* specifically matters here

It is worth isolating what the bias-free constraint contributes, since it is the premise of the question:

- **Noise-level universality.** One model is a valid prior at every σ. This is mandatory for the annealed sampler and the inverse-problem solver, which sweep σ from large to small. A noise-specific denoiser cannot do this.
- **Exact local linearity → clean score and clean Jacobian.** The residual=score reading and the interpretable filter analysis (Section 2.7) are *exact* under homogeneity, not approximate.
- **Local homogeneity is the convergence condition** that plug-and-play/RED solvers want (Section 4.4). Bias-free hands it to you by construction.
- **Cleaner residuals for detection.** Without bias artefacts contaminating the residual, projection-error signals (anomaly, forgery, deepfake detection) are less noisy and more discriminative.

In short, bias-free is not a minor architectural tweak — it is the property that turns "a good denoiser" into "a reusable, universal, analyzable prior."

---

## Part 6 — Concrete build roadmap for this repository

This repository already has the core asset: a **bias-free ConvNeXt/ConvUNext denoiser with a frozen Gabor stem, trained under a noise-σ curriculum** (trainer at `train/bfunet/train_convunext_denoiser.py`). That curriculum — spanning many noise levels — is precisely what produces the universal, all-σ prior this review assumes. The following is a pragmatic, increasing-ambition path to harvest the capabilities above.

**Stage 0 — Expose the score primitive.** Wrap the trained denoiser in a thin `score(y, sigma) = (f(y) − y) / sigma**2` API. This is the foundation for everything else. Validate it by checking that a single denoise reduces score magnitude.

**Stage 1 — Unconditional sampler.** Implement the annealed-Langevin loop (Section 2.3). Sanity-check that samples look like the training domain. This proves the prior is generative and exercises the σ-schedule code you will reuse everywhere.

**Stage 2 — Universal inverse-problem solver.** Implement the constrained sampler (Section 2.4) parameterized by an arbitrary linear operator M. Demonstrate deblur + super-resolution + inpainting from the *same* checkpoint. This is the highest value-per-effort capability and directly reproduces the Kadkhodaie–Simoncelli result on your own prior. (Reference implementation: `github.com/LabForComputationalVision/universal_inverse_problem`.)

**Stage 3 — Detection toolkit.** Build the projection-residual and score-magnitude signals (Sections 2.5, 2.6) into an anomaly/quality/forensics utility. Cheap to build on top of Stage 0; opens B1, B2, B3, B4. Remember to use *reconstruction residual*, not raw likelihood (Section 4.1).

**Stage 4 — Research bets.** Pick from Part 3.C — prior-guided active sensing (C1) and the universal naturalness regularizer (C2) are the most differentiated and have the clearest payoff for downstream models in this library.

Throughout, respect the limits: additive-Gaussian only (4.2), domain-matched prior (4.5), and slow sampling (4.3) — budget for distillation if any path needs to be interactive.

---

## References

**Foundations — bias-free denoising and the score identity**
1. Mohan, Kadkhodaie, Simoncelli & Fernandez-Granda (2020). *Robust and Interpretable Blind Image Denoising via Bias-Free Convolutional Neural Networks.* ICLR 2020. arXiv:1906.05478. https://arxiv.org/abs/1906.05478
2. Robbins (1956). *An Empirical Bayes Approach to Statistics.* Proc. 3rd Berkeley Symposium, Vol. I, 157–163.
3. Miyasawa (1961). *An Empirical Bayes Estimator of the Mean of a Normal Population.* Bull. International Statistical Institute 38, 181–188.
4. Efron (2011). *Tweedie's Formula and Selection Bias.* JASA 106(496), 1602–1614. https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf
5. Hyvärinen (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR 6, 695–709.
6. Vincent (2011). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation 23(7), 1661–1674.

**Inverse problems via implicit priors**
7. Kadkhodaie & Simoncelli (2021). *Stochastic Solutions for Linear Inverse Problems using the Prior Implicit in a Denoiser.* NeurIPS 2021. arXiv:2007.13640. https://arxiv.org/abs/2007.13640
8. Venkatakrishnan, Bouman & Wohlberg (2013). *Plug-and-Play Priors for Model Based Reconstruction.* IEEE GlobalSIP. DOI:10.1109/GlobalSIP.2013.6737048
9. Romano, Elad & Milanfar (2017). *The Little Engine That Could: Regularization by Denoising (RED).* SIAM J. Imaging Sci. 10(4), 1804–1844. arXiv:1611.02862
10. Milanfar & Delbracio (2024). *Denoising: A Powerful Building-Block for Imaging, Inverse Problems, and Machine Learning.* Phil. Trans. R. Soc. A. arXiv:2409.06219
11. Kadkhodaie & Simoncelli MRI application (2025). *Noisy MRI Reconstruction via MAP Estimation with an Implicit Deep-Denoiser Prior.* arXiv:2511.11963
12. *PI-AstroDeconv: A Physics-Informed Unsupervised Learning Method for Astronomical Image Deconvolution.* arXiv:2403.01692

**Score-based generative models, geometry, generalization**
13. Song & Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS 2019. arXiv:1907.05600
14. Ho, Jain & Abbeel (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239
15. Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. arXiv:2011.13456
16. Kadkhodaie, Guth, Simoncelli & Mallat (2024). *Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations.* ICLR 2024 (Outstanding Paper). arXiv:2310.02557
17. Simoncelli & Olshausen (2001). *Natural Image Statistics and Neural Representation.* Annual Review of Neuroscience 24, 1193–1216.

**Detection, quality, purification, compression, sensing**
18. Nalisnick, Matsukawa, Teh, Görür & Lakshminarayanan (2019). *Do Deep Generative Models Know What They Don't Know?* ICLR 2019. arXiv:1810.09136
19. Nalisnick et al. (2019). *Detecting Out-of-Distribution Inputs to Deep Generative Models Using Typicality.* arXiv:1906.02994
20. Mittal, Soundararajan & Bovik (2013). *Making a "Completely Blind" Image Quality Analyzer (NIQE).* IEEE Signal Processing Letters 22(3), 209–212.
21. Nie, Guo, Huang, Xiao, Vahdat & Anandkumar (2022). *Diffusion Models for Adversarial Purification (DiffPure).* ICML 2022. https://diffpure.github.io/
22. Wang et al. (2023). *DIRE for Diffusion-Generated Image Detection.* ICCV 2023. https://github.com/ZhendongWang6/DIRE
23. Chu et al. (2025). *FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error.* CVPR 2025. arXiv:2412.07140
24. Zhou et al. (2022). *Diffusion Models for Medical Anomaly Detection.* MICCAI 2022.
25. Minnen, Ballé & Toderici (2018). *Joint Autoregressive and Hierarchical Priors for Learned Image Compression.* NeurIPS 2018. arXiv:1809.02736
26. Hoogeboom, Agustsson, Mentzer, Versari, Toderici & Theis (2023). *High-Fidelity Image Compression with Score-Based Generative Models.* arXiv:2305.18231
27. Axelsson & Wakin (2014). *Compressed Sensing with Prior Information: Optimal Strategies, Geometry, and Bounds.* arXiv:1408.5250

**Reference implementation**
- Lab for Computational Vision, universal inverse problem solver: https://github.com/LabForComputationalVision/universal_inverse_problem
- Bias-free denoising: https://labforcomputationalvision.github.io/bias_free_denoising/
