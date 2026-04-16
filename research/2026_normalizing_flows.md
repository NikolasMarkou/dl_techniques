# Normalizing Flows for Financial Time Series — A Calibrated Survey

**Author**: Compiled by the `dl_techniques` research track
**Date**: 2026-04-16
**Scope**: Foundational theory, finance-specific applications, temporal modeling, heavy-tail treatment, implementation landscape, and a calibrated critique.
**Method**: Claims below were extracted from ~25 papers and are tagged with posterior confidence from a Bayesian hypothesis tracker (see `analyses/analysis_2026-04-16_6d622783/`). Every cited source was fetched and verified (HTTP 200) before inclusion. Absent / unverifiable references are surfaced explicitly, not quietly dropped.

---

## Executive Summary

Two statements about this field are now **confirmed** at posterior ≥ 0.95:

1. **[H1, 0.97]** — For heavy-tailed targets like financial returns, a normalizing flow with a Gaussian base and a Lipschitz coupling/spline architecture is **structurally incapable** of producing heavy-tailed outputs. The fix is either a heavy-tailed base with learnable tail parameters (Tail-Adaptive Flows, Student-t / Variance Gamma / NIG bases) or a non-Lipschitz architecture (Neural Autoregressive Flows).
2. **[H3, 0.96]** — The NF-for-finance literature is systematically **publication-biased**. Across the papers surveyed, none includes the canonical classical baselines (GARCH-GPD with Kupiec/Christoffersen/ES-bootstrap; t-copula-GARCH on multivariate VaR; Heston calibration at the 1–3 bp bar; EnbPI/SPCI conformal intervals). Reported headline improvements are against Gaussian-base NFs or unnamed "neural baselines" — rarely against the 20-year-deployed quantitative-finance baselines.

Four further statements stand with medium posterior:

- **[H4, 0.72]** — Mixture-of-flow architectures can enforce arbitrage-free option pricing *by construction*, not just in expectation. Yang & Hospedales (UAI 2023) is the primary worked example.
- **[H6, 0.27 — weakened]** — "GARCH + NF beats pure flows" is not empirically supported by any quantitative paper in the corpus; the canonical blog-post construction (Sarem Seitz) is pedagogical and qualitative.
- **[H2, 0.53]** — Autoregressive conditioning is the right pattern for multivariate time series, but AR-plus-*flow* specifically has been superseded by AR-plus-*diffusion* on the standard probabilistic-forecasting benchmarks (Rasul et al. TimeGrad, ICML 2021).
- **[H8, 0.67]** — Material drivers of deployability live *outside* the scope "NF architecture + financial data + training loss": Extreme Value Theory, copulas, stochastic-volatility-plus-jumps, regime-switching, and conformal prediction are the accomplice literatures that determine what a flow must beat.

If you are choosing whether to bet on normalizing flows for a finance problem, the single most useful question is: **what classical baseline already solves my problem, and what does the NF add that my baseline cannot?** If the answer is "the flow adds flexibility I don't need," stop.

---

## 1. Mathematical Foundations

### 1.1 The change-of-variables identity

A normalizing flow is an invertible map $f: \mathbb{R}^d \to \mathbb{R}^d$ that pulls a simple base density $p_Z(z)$ back to a target density $p_X(x)$:

$$ \log p_X(x) = \log p_Z(f^{-1}(x)) + \log\left|\det J_{f^{-1}}(x)\right|. $$

Three requirements bind every design choice:

| Requirement | Tension |
|:------------|:--------|
| $f$ must be invertible (bijective, in practice a diffeomorphism). | Rules out arbitrary neural networks. |
| The Jacobian determinant must be cheap. | Forces triangular, diagonal, or structured Jacobians. |
| Both $f$ and $f^{-1}$ should be fast to compute. | Tension between fast sampling and fast density evaluation. |

### 1.2 Taxonomy of canonical architectures

Papamakarios, Nalisnick, Rezende, Mohamed & Lakshminarayanan 2021 (*Normalizing Flows for Probabilistic Modeling and Inference*, JMLR 22:1–64) is the authoritative survey. Their taxonomy splits flows into **finite compositions** (autoregressive, coupling, linear, residual) and **continuous-time** (Neural ODE, FFJORD) families. For finance applications, only the finite compositions matter at present.

| Paper | Year | Family | Log-det Jacobian | Density eval | Sampling |
|:------|:-----|:-------|:-----------------|:-------------|:---------|
| **NICE** (Dinh et al.) | 2014 | additive coupling + diag rescale | unit-triangular + diag | $O(d)$ | $O(d)$ |
| **RealNVP** (Dinh et al.) | 2016 | affine coupling | triangular | $O(d)$ | $O(d)$ |
| **MAF** (Papamakarios et al.) | 2017 | masked autoregressive, affine | triangular | $O(1)$ parallel | $O(d)$ sequential |
| **IAF** (Kingma et al.) | 2016 | inverse autoregressive, affine | triangular | $O(d)$ sequential | $O(1)$ parallel |
| **Glow** (Kingma-Dhariwal) | 2018 | 1×1 conv + actnorm + affine coupling | diag + full + triangular | $O(d)$ | $O(d)$ |
| **NAF** (Huang et al.) | 2018 | monotonic MLP per dim | triangular | $O(1)$ parallel | **no analytic inverse** — numerical bisection |
| **NSF** (Durkan et al.) | 2019 | rational-quadratic spline | diagonal / triangular | $O(d)$ | $O(d)$ analytic |

**Universal approximation status.** Among the seven above, only NAF (Huang, Krueger, Lacoste & Courville, ICML 2018) has a *proved* universal-approximation theorem: the Deep Sigmoidal Flow is a universal approximator for continuous probability distributions. The JMLR 2021 survey flags the universality of affine autoregressive flows with multiple layers as an **open problem**, and the minimum coupling depth required for universality is likewise unresolved.

**MAF/IAF duality.** Masked autoregressive flows evaluate density in a single parallel pass but sample sequentially in $O(d)$ steps; the inverse autoregressive flow flips this asymmetry. Coupling flows (RealNVP, Glow, NSF) are symmetric, which is why they dominate image modeling but cost more per layer. For finance applications, density evaluation is the primary operation (training by maximum likelihood, VaR/ES via density integration), so MAF is a natural default.

### 1.3 The tail problem (the single most load-bearing result for finance)

Financial log-returns are leptokurtic at daily and intraday scales. Empirical excess kurtosis for equity indices is in the 3–20 range; individual equities can exceed 100. Gaussian-tailed models are systematically mis-specified on the left tail (the risk tail).

**Theorem (Jaini, Kobyzev, Yu & Brubaker, ICML 2020, "Tails of Lipschitz Triangular Flows", arXiv:1907.04481).** Let $T: \mathbb{R}^d \to \mathbb{R}^d$ be an increasing triangular map that is Lipschitz-continuous, and let $p$ be a light-tailed source density. Then the push-forward $T_\# p$ is also light-tailed. More generally, Lipschitz triangular maps **preserve the tail class** of their source.

The mechanism is the density-quantile ratio $T'(z) = f_{Q_p}(u) / f_{Q_q}(u)$: to map a light-tailed $p$ to a heavy-tailed $q$, $T'$ must be unbounded, which contradicts Lipschitz continuity.

**Which architectures does this rule out?** From the paper's own statement: *"affine flow models like Real-NVP, NICE, MAF, Glow etc. cannot capture heavier tails than the source density"* when the source is $\mathcal{N}(0,I)$. NSF is explicitly in this class as well — the paper states that outside its transformation window $[-B, B]$, the map is the identity, *"resulting in linear tails"*, which is Lipschitz.

**What survives?** Two escape routes:

1. **Heavy-tailed base** — replace $\mathcal{N}(0,I)$ with Student-t, Variance Gamma, Normal-Inverse Gaussian, or a copula with heavy-tailed marginals. **Tail-Adaptive Flows (TAF)** (Jaini et al.'s proposed fix) use Student-t with **learnable degrees of freedom $\nu$**, jointly optimized with the flow parameters by maximum likelihood. Setting $\nu \to \infty$ recovers Gaussian.
2. **Non-Lipschitz flow** — Neural Autoregressive Flows (Huang 2018) have unbounded slopes in their monotonic MLP transformers, breaking the Lipschitz assumption. They pay for this by having no analytic inverse (numerical bisection required for sampling).

**Empirical corroboration (Jaini et al.).** On a synthetic Gaussian-to-Student-$t_2$ mapping task (true tail coefficient $\gamma \approx 0.81$): standard RealNVP recovers $\gamma \approx 0.15$ (matching the *source*, failing to capture the target's tails). TAF-RealNVP recovers $\gamma \approx 0.80$, essentially matching truth.

**Extension (Laszkiewicz, Lederer & Fischer).** Copula-Based Normalizing Flows (arXiv:2107.07352, ICML 2021 INNF+ workshop) and Marginal Tail-Adaptive Normalizing Flows (arXiv:2206.10311, ICML 2022) generalize the fix to per-dimension tail indices and arbitrary heavy-tailed marginals — relevant when different assets have different tail heaviness.

**Consequence for financial applications.** A naive RealNVP/MAF/NSF with Gaussian base on raw daily returns is **mathematically** (not just empirically) mis-specified. Any serious NF-for-finance construction must either (a) use a heavy-tail base with learnable tail parameter, (b) use a non-Lipschitz flow, or (c) operate on GARCH-filtered residuals which are closer to iid.

---

## 2. The Financial Data Problem

Before surveying NF-for-finance work, it is worth stating clearly what makes financial returns hard:

| Feature | Implication for NF design |
|:--------|:--------------------------|
| Leptokurtic marginals (excess kurtosis 3–20+) | Need heavy-tail base or non-Lipschitz flow (§1.3). |
| Volatility clustering (autocorrelated squared returns) | Need temporal conditioning — either GARCH-filter preprocessing or AR/state-space flow. |
| Tail dependence (joint extremes across assets) | Need multivariate model that captures co-crashes (copula with tail parameter, or multivariate heavy-tail base). |
| Non-stationarity / regime change | Need regime-switching structure or online adaptation; naive NF density estimation blurs regimes. |
| Leverage effect (asymmetric vol response to returns) | Asymmetric architectures or GJR-style baselines. |
| Sparse extreme events (years between true crashes) | Evaluation cannot rely on in-sample NLL; rolling-window backtests are mandatory. |

Any paper that does not address at least 3 of these features for the relevant task is probably not deployable, however good the reported numbers.

---

## 3. Finance-Specific NF Research

### 3.1 Lévy-Flow Models: Heavy-Tail-Aware NFs for Financial Risk (Drissi 2026)

**Citation**: Drissi, R. (2026). *Lévy-Flow Models: Heavy-Tail-Aware Normalizing Flows for Financial Risk Management*. arXiv:2604.00195.
**Status**: Single-author 2026 preprint. Not yet peer-reviewed.

**Method.** Replaces the Gaussian base of a Neural Spline Flow with **Variance Gamma** or **Normal-Inverse Gaussian** Lévy-process bases. Introduces an "identity-tail preserving" construction so that the flow acts as the identity outside a compact transformation region, guaranteeing that base-distribution tails propagate through unchanged.

**Data.** S&P 500 daily returns plus "additional assets" (unspecified).

**Claim.** *"VG-based flows reduce test negative log-likelihood by 69% relative to Gaussian flows"*; *"exact 95% VaR calibration"* for VG flows; NIG flows give most accurate Expected Shortfall.

**What is missing.** No walk-forward / rolling-window evaluation. No absolute NLL numbers (only relative). No public code. No comparison against GARCH-GPD or t-copula-GARCH. No Kupiec/Christoffersen/ES-bootstrap. No breakdown across market regimes. Single author.

**Calibrated reading.** The *mechanism* is sound (heavy-tail Lévy base is exactly the fix Jaini et al. prescribes). The *magnitude* reported ("69%") is uninformative without the absolute baseline NLL — on S&P daily returns, a Gaussian-base NF is catastrophically mis-specified, so any heavy-tail alternative will look dramatic without necessarily being deployable. The "exact 95% VaR" claim is trivially achievable on a single sample path; the hard test is conditional coverage across rolling windows and stress periods. Treat this paper as strong *mechanistic* evidence for H1 and a textbook example of the H3 publication-bias pattern simultaneously.

### 3.2 Mixture of Normalizing Flows for European Option Pricing (Yang & Hospedales, UAI 2023)

**Citation**: Yang, Y. & Hospedales, T. M. (2023). *Mixture of Normalizing Flows for European Option Pricing*. UAI 2023, PMLR 216:2390–2399.
**Code**: https://github.com/qmfin/MoNF

**Method.** Models the risk-neutral density directly via a mixture of normalizing flows. Translates no-static-arbitrage constraints on European call prices (convex in strike, monotone decreasing, $C(0)=S$, $C(\infty)=0$) into architectural constraints on the mixture density. The authors state: *"our solution meets those constraints exactly by design"* rather than via penalty terms.

**Calibrated reading [H4, 0.72].** The arbitrage-free-by-construction claim is architecturally grounded — the mixture parameterization yields densities that push through the call-price integrals to satisfy the required convexity/monotonicity. The paper is peer-reviewed (UAI is archival) and releases public code. Caveat: the "by construction" claim rests on the mixture parameterization actually yielding non-negative butterfly spreads across *all* strikes and maturities — a property that a user replicating the code must verify for their specific parameterization, not just assume.

**What the paper does NOT do.** Report IV-surface RMSE at the 1–3 bp bar that modern Heston / rough-Heston / SABR calibrators achieve on SPX. This is the industry-relevant comparison. Without it, "outperforms other methods" is insufficient for deployment judgement.

### 3.3 Learning Sentimental and Financial Signals with NFs (Tai et al., IEEE SPL 2021)

**Citation**: Tai, W., Zhong, T., Mo, Y. & Zhou, F. (2021). *Learning Sentimental and Financial Signals with Normalizing Flows for Stock Movement Prediction*. IEEE Signal Processing Letters 29, 414–418. DOI 10.1109/LSP.2021.3135793.

**Note on attribution.** The original reading list supplied the authorship as "Tang et al."; the correct first author is **Tai**.

**Method.** A VAE-style latent that fuses tweet embeddings with historical price sequences; a normalizing-flow posterior replaces the Gaussian posterior.

**Status.** Paywalled; full tables were not extractable for this survey. Likely data: the StockNet benchmark (Xu & Cohen 2018, 88 S&P stocks, 2014-01-01 to 2016-01-01).

**Calibrated reading.** StockNet is a narrow benchmark — 88 large-caps over two years. Accuracy deltas at IEEE-SPL's typical scale (1–2 points over VAE baselines) are within seed variance. The specific ascription of gains to the NF posterior (versus the fusion architecture changes) requires ablation. Treat as illustrative of the pattern, not definitive evidence.

### 3.4 Generative ML for Multivariate Equity Returns (Tepelyan & Gopal, ICAIF 2023)

**Citation**: Tepelyan, R. & Gopal, A. (2023). *Generative Machine Learning for Multivariate Equity Returns*. ICAIF 2023, arXiv:2311.14735.

**Method.** Conditional normalizing flows and conditional IWAEs on 500-dim S&P 500 daily returns; downstream use for synthetic data generation, VaR, correlation estimation, portfolio optimization.

**Status.** Peer-reviewed at ICAIF. The closest available "multivariate equity returns via NF" reference in the corpus. Absolute numbers for NLL, Sharpe, VaR coverage live in the paper body; abstract-level claims do not expose them.

**Note on reading-list error.** The original reading list suggested a "Lopez-Lira & Tang 2023" paper as an NF-based financial forecasting reference. That paper (Lopez-Lira & Tang 2023, arXiv:2304.07619) is *Can ChatGPT Forecast Stock Price Movements?* and uses GPT, not NFs. Tepelyan-Gopal is the closest real substitute.

### 3.5 Copula-Based NFs (Laszkiewicz, Lederer & Fischer, ICML 2021 workshop)

**Citation**: Laszkiewicz, M., Lederer, J. & Fischer, A. (2021). *Copula-Based Normalizing Flows*. ICML 2021 INNF+ workshop. arXiv:2107.07352.
**Code**: https://github.com/MikeLasz/Copula-Based-Normalizing-Flows

**Method.** Replace the Gaussian base with a copula with heavy-tailed marginals, separating multivariate dependence structure from tail behavior.

**Finance evaluation.** None. Evaluated on synthetic heavy-tailed benchmarks. This is a mechanism paper, not a finance-applied one.

**Follow-up.** *Marginal Tail-Adaptive Normalizing Flows*, Laszkiewicz et al., ICML 2022, arXiv:2206.10311 — more rigorous per-dimension tail index treatment.

### 3.6 Gap: Claimed citations that do not exist

One paper the original reading list implicitly assumed exists does not: a "Chen-Bhamra-Lokka-Fang-Gao NF-GARCH" paper cannot be located in arXiv, Google Scholar, Semantic Scholar, or Bhamra's official publication page (https://www.harjoatbhamra.com/research). Bhamra's research is asset pricing / monetary policy / corporate default, not ML. If you find such a paper, check the authorship carefully — the claim surfaced during this survey's literature search and could not be verified.

---

## 4. Temporal Modeling

NFs on their own are static density estimators. For time series, some temporal structure must be added.

### 4.1 Conditioned NFs (Rasul et al., ICLR 2021)

**Citation**: Rasul, K., Sheikh, A., Schuster, I., Bergmann, U. & Vollgraf, R. (2021). *Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows*. ICLR 2021. arXiv:2002.06103.
**Code**: https://github.com/zalandoresearch/pytorch-ts

**Method.** An RNN or Transformer encoder produces a hidden state $h_t$; a normalizing flow models $p(x_t \mid h_t)$ with weights shared across time. Tested with RealNVP and MAF heads; MAF generally performs better. Gaussian base.

**Datasets.** Exchange-rate (FX), Solar, Electricity, Traffic, Taxi, Wikipedia. **No equities, no raw returns, no options, no order book.** The single finance-like dataset (`exchange-rate`) is the FX benchmark that is historically easy.

**Results (CRPS-Sum, smaller is better).** Transformer-MAF vs GP-Copula: Traffic 0.056 vs 0.078 (−28%); Taxi 0.179 vs 0.208; Wiki 0.063 vs 0.086; Exchange 0.005 vs 0.007.

**Limitations (stated).** Missing data not handled; regime-change not addressed beyond training-window mean-scaling.

**Calibrated reading [H2, 0.53].** Rasul established AR+NF as a viable *pattern* for multivariate probabilistic forecasting — this is important and real. But the paper's experimental scope does not include financial returns in the risk-management sense, and the naive AR+MAF construction inherits the Jaini tail limitation (Gaussian base + affine MAF on raw series).

### 4.2 TimeGrad — AR + Diffusion (same authors, ICML 2021)

**Citation**: Rasul, K., Seward, C., Schuster, I. & Vollgraf, R. (2021). *Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting*. ICML 2021. arXiv:2101.12072.

**Why it matters here.** Same benchmarks as Rasul et al. 2021; same LSTM encoder; the flow head is replaced by a denoising diffusion head. Result: AR+diffusion **beats** Transformer-MAF on 5/6 benchmarks — Traffic 0.044 vs 0.056 (−21%); Taxi 0.114 vs 0.179 (−36%); Wiki 0.049 vs 0.063 (−23%). AR+flow is superseded on standard benchmarks by AR+diffusion.

**Implication.** When the downstream task is probabilistic forecasting with no structural requirement of exact likelihood, diffusion may be the better head. NFs retain their edge when (a) exact likelihood is required (density-based risk measures), (b) exact sampling is cheap and useful (Monte-Carlo VaR), or (c) invertibility is load-bearing (arbitrage-free pricing constraints).

### 4.3 Normalizing Kalman Filters (de Bézenac et al., NeurIPS 2020)

**Citation**: de Bézenac, E., Rangapuram, S. S., Benidis, K. et al. (2020). *Normalizing Kalman Filters for Multivariate Time Series Analysis*. NeurIPS 2020.

**Method.** A linear Gaussian state-space model on the latent state $l_t$; a RealNVP-based flow $f_t$ on the *observation* space: $y_t = f_t(A_t l_t + \varepsilon_t)$. The flow direction is inverted from standard NF usage — data pulled back to Gaussian, not Gaussian pushed to data.

**Reconciliation with classical Kalman theory.** Propositions 1 and 2 of the paper show that filtered and smoothed latent distributions remain Gaussian with closed-form Kalman-like recursive updates, and the likelihood is tractable. Setting $f_t = \text{id}$ recovers DeepState (Rangapuram et al. 2018) exactly.

**What the flow adds.** A learned invertible warp of the observation manifold that simultaneously breaks (i) Gaussian marginals and (ii) linear inter-series dependence — neither of which a learned linear transition $F_t$ can achieve alone.

**Missing-data handling.** Native: Kalman equations marginalize over missing values in closed form, and a per-series (local) flow preserves this property. Tested at up to 90% missingness on electricity.

**Limitation (author-stated).** *"We no longer have identifiability w.r.t. the state space parameters"* — flow and SSM parameters are entangled; constraining the flow expressivity is suggested as future work.

**Results (CRPS-Sum).** Wins on 4/5 standard benchmarks; loses traffic to DeepAR/GP-Copula. Largest gain on wiki: 0.071 vs GP-Copula 0.092 (−23%) and DeepAR 0.127 (−44%).

**Calibrated reading.** NKF is the cleanest example of a *hybrid* — classical structure (Kalman tractable inference) combined with a learned invertible component (flow) that expands expressiveness while preserving closed-form inference. This is the compositional approach the field should emulate more broadly.

### 4.4 Temporal-Conditioned NFs for Anomaly Detection (Baumgartner et al., 2026)

**Citation**: Baumgartner, D., Langseth, H., Engø-Monsen, K. & Ramampiaro, H. (2026). *Temporal-Conditioned Normalizing Flows for Multivariate Time Series Anomaly Detection*. arXiv:2603.09490.
**Code**: https://github.com/2er0/HistCondFlow

**Method.** AR+flow for anomaly detection. Historical window $w_t = x_{t-k:t-1}$ passed to a conditioner $\Theta(\cdot)$; flow models $p(x_t \mid w_t)$. Five variants: passthrough, MLP, CNN, stateless LSTM, stateful LSTM.

**Results.** Beats plain RealNVP by +0.06 to +0.19 AUC on FSB / SRB / GHL. Loses to MTGFlow on SMD; loses to IF-LOF on SRB in external comparison.

**Calibrated reading.** Modest empirical improvement over a naive RealNVP baseline. Key deep-AD baselines (GANF, USAD, OmniAnomaly, MAD-GAN, MTAD-GAT) are not in the comparison. Not finance-specific; anomaly detection is tangentially relevant to rare-event market detection but needs an entirely different dataset to count as a finance result.

### 4.5 Other relevant temporal-flow work

- **ProFITi** (Yalavarthi et al., ICML 2024, arXiv:2402.06293) — invertible attention (SITA) for irregular multivariate time series. Applied to medical data (MIMIC), not finance. Reports ~4× higher average likelihood vs baselines. Architecturally novel; code at https://github.com/yalavarthivk/ProFITi.
- **IN-Flow** (Fan et al., KDD 2025, arXiv:2401.16777) — flow as *preprocessing* via bi-level optimization. Removes distribution shift before a downstream point forecaster. Up to 54% MSE reduction on Weather/CAISO. A flow used for stationarization rather than density modeling.
- **Marino et al. 2020** (AABI, arXiv:2010.03172) — affine autoregressive flow applied across time: $x_t = \mu_\theta(x_{<t}) + \sigma_\theta(x_{<t}) \cdot y_t$. The flow is a "moving reference frame" subtracting predictable structure. Applied to video, not finance.

---

## 5. GARCH + NF: Theory vs Practice

### 5.1 The pedagogical construction (Sarem Seitz blog)

Sarem Seitz, *Let's make GARCH more flexible with Normalizing Flows* (https://sarem-seitz.com/posts/lets-make-garch-more-flexible-with-normalizing-flows.html).

**Idea.** Treat GARCH innovations $\varepsilon_t \sim \mathcal{N}(0,1)$ as a **base distribution**; compose a monotone planar flow on top. Because the composed map is monotone, quantile scaling under varying $\sigma_t$ is preserved, so volatility clustering survives the transform while the conditional shape becomes non-Gaussian.

**Pseudocode sketch.**

```
struct PF_GARCH: sigma0, gamma, alpha, beta, flow   // flow = Chain(Planar, Planar, Planar)

def logpdf(m, y):
    y_tilde = flow.inverse(y)                       # pull back to GARCH latent
    sigma2_1 = sigma0 ** 2
    for t in 2..T:
        sigma2_t = gamma + alpha * sigma2_{t-1} + beta * eps2_{t-1}
        eps_t    = y_tilde_{t-1} / sigma_t
    for t in 1..T:
        base_t  = Normal(0, sigma_t)
        flow_t  = TransformedDist(base_t, flow)
        ell_t   = logpdf(flow_t, y_t)
    return mean(ell)
```

**What the blog post does.** Fits on AAPL log-returns, stacks three planar layers, trains with ADAM for 1000 iterations, plots quantile bands and a 60-day / 75k Monte-Carlo forecast.

**What the blog post does NOT do.** Report log-likelihood vs vanilla GARCH. Backtest VaR or ES. Compare against GARCH-Student-t, GARCH-GPD, t-copula, EnbPI, or any other baseline. The author explicitly flags that the forecast interval *"appears a little too small and might actually under-estimate potential risk"*.

**Calibrated reading [H6, 0.27].** The construction is elegant and the code is a useful starting template — but there is no quantitative evidence in the reference blog that GARCH+NF beats GARCH-Student-t. H6 is currently **not supported** by the canonical public GARCH+NF reference.

### 5.2 Adjacent literature

- **Di Persio et al. 2023** (arXiv:2310.01063) — GRU + {GARCH, EGARCH, GJR, APARCH} on Garman-Klass range-based volatility (S&P, gold, Bitcoin). Explicit quote: *"hybrid solutions produce more accurate point volatility forecasts"* but *"this does not necessarily translate into superior VaR and ES forecasts."* Important caveat: point-vol improvement does **not** imply tail-risk improvement.
- **Chen, Wang & Gerlach 2023** (arXiv:2302.08002) — LSTM + Realized GARCH with SMC Bayesian inference across 31 equity indices including COVID. This is the right *structure* for a rigorous NF-GARCH paper to emulate: 31 assets, multi-axis evaluation (likelihood, volatility, VaR/ES, option pricing).
- **arXiv:2311.00580** — Tail-modeling NF with a Flexible Tail Transformation (FTT) bijector; VaR/ES on daily returns (stocks + crypto); baselines include standard NFs and GPD. Not a GARCH+NF paper; GARCH is a baseline.

### 5.3 The synthesis nobody has published yet

Putting Jaini 2020 + Huang 2018 + McNeil-Frey 2000 together gives a straightforward construction that no paper in the corpus has evaluated at the backtest bar:

1. Fit univariate GARCH(1,1)-like model to each series; extract standardized residuals $\eta_t = r_t / \sigma_t$.
2. Model $p(\eta_t \mid h_t)$ as a Neural Autoregressive Flow (non-Lipschitz, universal) with a Student-t base whose degrees of freedom are learnable (per-dim).
3. Condition on a recurrent summary of past standardized residuals and optionally cross-asset state.
4. Evaluate on Kupiec + Christoffersen + McNeil-Frey ES bootstrap against AR-GARCH-GPD (McNeil-Frey 2000) and t-copula-GJR-GARCH.

This is the minimal construction justified by theory (tail-preservation + universal-approximation + volatility-clustering prior) that has not been tested. H14 (0.17) and H15 (0.11) in the tracker carry these as open conjectures — their low posteriors reflect absence of evidence, not evidence of absence.

---

## 6. Competing Baselines (What NFs Must Beat)

The single strongest finding of this survey is that **almost no NF-for-finance paper evaluates against the classical baselines that have been deployed and backtested for decades**. Any practitioner choosing a model must know these.

### 6.1 EVT / Peaks-over-Threshold (Univariate tail risk)

**Reference**: McNeil, Frey, Embrechts (2015), *Quantitative Risk Management* (Princeton, 2nd ed). Seminal paper: McNeil & Frey (2000), *J. Empirical Finance* 7, 271–300.

**Method.** Two-stage: (1) AR-GARCH filter the return series; (2) fit a Generalized Pareto Distribution to residual exceedances over a threshold $u$; (3) recover VaR and ES in closed form: $\text{ES}_\alpha = (\text{VaR}_\alpha + \beta - \xi u)/(1-\xi)$ for $\xi < 1$.

**Backtesting infrastructure.** Kupiec test (unconditional coverage), Christoffersen test (conditional coverage), McNeil-Frey bootstrap (ES backtesting). These are regulatory-standard tests.

**Bar for NF-based tail models.** You must match or beat AR-GARCH-GPD on these three backtests at 99% VaR and 97.5% ES across multiple rolling windows and multiple asset classes. If you haven't done this comparison, you are not yet competing with the industry baseline.

### 6.2 t-Copula GARCH (Multivariate tail risk)

**Reference**: McNeil-Frey-Embrechts 2015 chapters 7–8; comparative studies in *Frontiers Appl. Math. Stat.* (2025) and *Comput. Econ.* (2022).

**Method.** Univariate GARCH (often GJR or t-GARCH) fits each series. Standardized residuals are PIT-transformed and linked via a Student-t copula (symmetric tail dependence via a single parameter $\nu$). Vine copulas extend to asymmetric tail dependence.

**Empirical benchmark.** Student-t copula with GJR-GARCH marginals produces VaR violation rates within the 95% Kupiec acceptance region for portfolios of up to 10 assets, and passes Christoffersen conditional-coverage tests in many settings. Gaussian copula (DCC) does not.

### 6.3 Stochastic Volatility + Jumps (Option Pricing)

**References**: Heston (1993); Bates (1996); Duffie, Pan, Singleton (2000); Bauer (TU Wien thesis on fast Heston calibration); Rosenbaum et al. 2021 (rough Heston, arXiv:2107.01611).

**Calibration quality (SPX).** Classical Heston via Fourier / COS methods achieves ~3–4 bp weighted RMSE on liquid SPX surfaces; quadratic rough Heston achieves sub-3 bp on a wider surface; neural rough Heston reaches sub-0.3% RMSE.

**Bar for NF-based option pricing.** 1–3 bp RMSE on liquid SPX strikes and maturities. Anything greater than 5 bp is not competitive. Yang-Hospedales 2023 (§3.2) does not report bp-level RMSE; any NF-pricing paper that skips this number should be read skeptically.

### 6.4 Conformal Prediction for Time Series (Intervals)

**References**: Xu & Xie 2021 (EnbPI), ICML PMLR 139:11559–11569, arXiv:2010.09107. Follow-up SPCI, arXiv:2212.03463.

**Guarantee.** Under stationarity + strong mixing + consistency of the base predictor, marginal coverage $\geq 1-\alpha$ *asymptotically*. Not finite-sample, not conditional.

**Empirical.** Coverage within ~±2% of nominal at 90% and 95% levels on solar, traffic, stock benchmarks.

**Implication for NFs.** If the downstream task is *prediction-interval coverage* (and not full density calibration), EnbPI/SPCI is a cheap, distribution-free competitor. NF papers claiming better calibration must show head-to-head comparison.

### 6.5 Regime-Switching GARCH

**References**: Hamilton 1989; Haas, Mittnik & Paolella 2004.

**Why it matters.** Financial data exhibits discrete regimes (risk-on/off, crisis/normal). Markov-switching GARCH captures this with interpretable parameters. NF unconditional density estimation blurs regimes; NF+AR must *re-learn* what Markov-switching provides by construction. Every multivariate NF paper claiming "captures regime-like behavior" should be challenged to compare against a Markov-switching baseline.

---

## 7. Implementation Landscape

### 7.1 Libraries

| Library | Backend | Maintained | NSF | NAF | Status |
|:--------|:--------|:-----------|:----|:----|:-------|
| **nflows** (bayesiains) | PyTorch | ❌ (v0.14 Nov 2020) | ✓ | ✗ | Widely cited but stale |
| **TFP** (tensorflow-probability) | TF / Keras 3 | ✓ | ✗ (custom) | ✗ | Native Keras path; RealNVP + MAF shipped |
| **normflows** (Stimper) | PyTorch | ✓ | ✓ | ✗ | JOSS 2023 |
| **zuko** (probabilists) | PyTorch | ✓ | ✓ | ✓ | Modern API, supports NAF + UMNN |

**Finance-specialized NF library.** None exists. Confirmed against GitHub search and the `janosh/awesome-normalizing-flows` curated list. A Keras-native, finance-specialized NF package is a deployable niche.

### 7.2 The `janosh/awesome-normalizing-flows` reality

The curated list has sections: Publications (60), Applications (8), Packages (14), Repos (18), Blog Posts (5). **There is no dedicated Finance or Financial-Risk section.** Time series is represented by roughly two entries (Rasul 2021, NKF 2020) under Applications / Publications. The original reading list's framing that this list contains "sections on Time Series and Financial Risk" is inaccurate and should not be propagated.

### 7.3 Recommended implementation path for `dl_techniques` (Keras 3 / TF 2.18)

1. **Base bijectors** — Use `tfp.bijectors.RealNVP`, `MaskedAutoregressiveFlow`, `Chain`, `Invert`; wrap in `tfp.layers.AutoregressiveTransform` / `DistributionLambda` for Keras integration.
2. **Rational-quadratic splines** — Not in TFP core. Implement as a custom `tfp.bijectors.Bijector` with `_forward`, `_inverse`, `_forward_log_det_jacobian` — the NSF paper's bin-lookup + quadratic-solve is straightforward to port.
3. **Student-t base with learnable $\nu$** — `tfp.distributions.StudentT` with a trainable variable for `df`; add to the model as a `tfp.layers.IndependentStudentT` or build inside a custom Keras layer.
4. **GARCH preprocessing** — Keep it classical: `arch` library or a hand-rolled GARCH(1,1) fit per series, feed standardized residuals to the flow. Don't try to learn volatility clustering from scratch with a flow when GARCH does it with 3 parameters.
5. **Backtest harness** — Implement Kupiec, Christoffersen, McNeil-Frey ES bootstrap in a self-contained module. Any NF-for-finance contribution from this codebase should pass these before being reported.

---

## 8. Calibrated Critique

Synthesizing the tracker's posterior distribution over claims:

**What holds up.**

- **[H1, 0.97]** Heavy-tailed base + tail-aware architecture is genuinely required for heavy-tailed targets. The theoretical argument (Jaini 2020) is tight; the empirical demonstration (TAF-RealNVP vs standard RealNVP on Student-t targets) is clean.
- **[H4, 0.72]** Arbitrage-free-by-construction option pricing via mixture of NFs is architecturally grounded (Yang-Hospedales UAI 2023). Code is public; claim is verifiable.

**What is overclaimed.**

- **[H3, 0.96]** Systematic publication bias through baseline omission. The corpus pattern is too consistent to be accidental: papers headline against Gaussian-base NFs or unnamed "neural baselines," not against the quantitative-finance baselines that the NF would need to beat for deployment.
- **[H6, 0.27]** GARCH+NF as a volatility approach outperforming pure flows is unsupported by quantitative evidence in the surveyed corpus. The pedagogical construction exists; the backtested result does not.
- **[H2, 0.53]** AR+flow is a useful pattern, but AR+diffusion supersedes it on the standard probabilistic forecasting benchmarks. NF's residual relevance lives in (a) exact-likelihood risk measures, (b) arbitrage-free pricing constraints, (c) invertible preprocessing.

**What is missing.**

- No paper in the corpus reports Kupiec + Christoffersen + McNeil-Frey ES backtest results at 99% VaR / 97.5% ES against AR-GARCH-GPD across multiple asset classes and rolling windows. This is the minimum bar for a deployable NF-based risk model.
- No paper reports IV-surface RMSE at the 1–3 bp bar against classical Heston / rough Heston on SPX. This is the minimum bar for a deployable NF-based pricing model.
- No paper compares against EnbPI/SPCI when the task is prediction-interval coverage.
- No paper uses explicit regime-switching structure on returns data.

**What is theoretically ruled out.**

- RealNVP / Glow / MAF / NSF with Gaussian base and Lipschitz nets cannot produce heavier-than-Gaussian output tails. Any claim that "vanilla RealNVP models fat-tailed returns well" is false by Jaini's theorem; the papers that appear to do this on finite samples are fitting the *body* of the distribution well and masking the tail failure with in-sample metrics.

---

## 9. The Unevaluated Construction (Open Problem)

Theoretically motivated, empirically untested composition:

- **Base $B$**: Student-t with per-dimension learnable $\nu$ (TAF principle).
- **Architecture $A$**: NAF (non-Lipschitz, universal) — or a mix of spline coupling with NAF endpoints if NAF's numerical-inverse cost is prohibitive.
- **Temporal structure $T$**: AR conditioning on a recurrent summary of past standardized residuals.
- **Financial prior $I$**: GARCH preprocessing on each univariate series; optionally a Markov-switching latent for regime handling; optionally a copula overlay for multivariate tail dependence.

**Why this has not been published**: probably a combination of (i) nobody has joined the four literatures at once; (ii) NAF's numerical inversion cost is irritating in practice; (iii) the proper evaluation requires infrastructure (Kupiec/Christoffersen/ES backtest harness, clean multi-asset rolling data) that is not standard in ML research settings.

**Why it might be worth doing in `dl_techniques`**: (a) the theoretical case is clean; (b) the comparison bar (classical GARCH-GPD + t-copula-GJR-GARCH) is well-specified; (c) a public Keras implementation fills a real gap in the tooling landscape (§7.1).

---

## 10. Reading Order (If You Are Starting From Zero)

1. **Papamakarios et al. 2021** (JMLR) — https://jmlr.org/papers/v22/19-1028.html — the survey. §2–§4 for architectures.
2. **Dinh, Sohl-Dickstein & Bengio 2016** (RealNVP) — arXiv:1605.08803 — the canonical coupling flow.
3. **Papamakarios et al. 2017** (MAF) — arXiv:1705.07057 — autoregressive construction + MAF/IAF duality.
4. **Durkan et al. 2019** (NSF) — arXiv:1906.04032 — splines; note the explicit linear-tail remark.
5. **Huang et al. 2018** (NAF) — arXiv:1804.00779 — the universal-approximation theorem.
6. **Jaini et al. 2020** (Tails of Lipschitz Triangular Flows) — arXiv:1907.04481 — **read this before any finance-application paper**.
7. **McNeil, Frey & Embrechts 2015** — *Quantitative Risk Management* — chapters on EVT, GARCH, copulas. The baseline you are competing against.
8. **Rasul et al. 2021** (AR+NF TS) — arXiv:2002.06103 — the temporal-NF pattern.
9. **de Bézenac et al. 2020** (NKF) — NeurIPS — the cleanest hybrid.
10. **Yang & Hospedales 2023** (MoNF option pricing) — PMLR v216 — arbitrage-free by construction.
11. **Drissi 2026** (Lévy-Flow) — arXiv:2604.00195 — read with §3.1's caveats.

---

## 11. Posterior Tags — Claim Index

Every claim in this survey can be traced back to a hypothesis in the Bayesian tracker. The final posteriors:

| ID | Hypothesis | Posterior | Status |
|:---|:-----------|----------:|:-------|
| H1 | Heavy-tailed base materially improves tail fit vs Gaussian base | **0.97** | CONFIRMED |
| H2 | AR+flow dominant pattern for multivariate TS forecasting | 0.53 | ACTIVE (mitigated by AR+diffusion) |
| H3 | NF-for-finance claims are systematically publication-biased | **0.96** | CONFIRMED |
| H4 | Mixture flows enforce arbitrage-free pricing by construction | 0.72 | ACTIVE |
| H5 | A cited arXiv ID is hallucinated | 0.06 | WEAKENED (IDs 2604/2603.* verified real) |
| H6 | GARCH+NF outperforms pure flows on volatility | 0.27 | WEAKENED (no quantitative evidence) |
| H7 | $[H_S]$ Drivers within scope S | 0.44 | ACTIVE |
| H8 | $[H_{S'}]$ Drivers outside S (EVT, copula, SVJ, regime, conformal) | 0.67 | ACTIVE |
| H9 | EVT baseline remains competitive with NF on univariate tail risk | 0.62 | ACTIVE |
| H10 | t-copula GARCH remains competitive with NF on multivariate risk | 0.55 | ACTIVE |
| H11 | Heston/Bates/Merton remain competitive on option pricing | 0.55 | ACTIVE |
| H12 | Conformal (EnbPI/SPCI) competitive for intervals | 0.41 | ACTIVE |
| H13 | Regime-switching models capture structure NFs miss | 0.46 | ACTIVE |
| H14 | GARCH-residual + tail-adaptive NF composition wins (conjecture) | 0.17 | WEAKENED (untested) |
| H15 | NAF + Student-t + GARCH composition (conjecture) | 0.11 | WEAKENED (untested) |

---

## 12. References

- Baumgartner, D., Langseth, H., Engø-Monsen, K. & Ramampiaro, H. (2026). *Temporal-Conditioned Normalizing Flows for Multivariate Time Series Anomaly Detection*. arXiv:2603.09490.
- de Bézenac, E. et al. (2020). *Normalizing Kalman Filters for Multivariate Time Series Analysis*. NeurIPS 2020.
- Dinh, L., Krueger, D. & Bengio, Y. (2014). *NICE: Non-linear Independent Components Estimation*. arXiv:1410.8516.
- Dinh, L., Sohl-Dickstein, J. & Bengio, S. (2016). *Density Estimation using Real NVP*. arXiv:1605.08803.
- Di Persio, L. et al. (2023). *Combining Deep Learning and GARCH for Financial Volatility and Risk Forecasting*. arXiv:2310.01063.
- Drissi, R. (2026). *Lévy-Flow Models: Heavy-Tail-Aware Normalizing Flows for Financial Risk Management*. arXiv:2604.00195.
- Durkan, C., Bekasov, A., Murray, I. & Papamakarios, G. (2019). *Neural Spline Flows*. NeurIPS 2019. arXiv:1906.04032.
- Fan, W. et al. (2025). *Addressing Distribution Shift in Time Series Forecasting with Instance Normalization Flows*. KDD 2025. arXiv:2401.16777.
- Huang, C.-W., Krueger, D., Lacoste, A. & Courville, A. (2018). *Neural Autoregressive Flows*. ICML 2018. arXiv:1804.00779.
- Jaini, P., Kobyzev, I., Yu, Y. & Brubaker, M. A. (2020). *Tails of Lipschitz Triangular Flows*. ICML 2020. arXiv:1907.04481.
- Kingma, D. P. & Dhariwal, P. (2018). *Glow: Generative Flow with Invertible 1×1 Convolutions*. NeurIPS 2018. arXiv:1807.03039.
- Laszkiewicz, M., Lederer, J. & Fischer, A. (2021). *Copula-Based Normalizing Flows*. ICML 2021 INNF+. arXiv:2107.07352.
- Laszkiewicz, M. et al. (2022). *Marginal Tail-Adaptive Normalizing Flows*. ICML 2022. arXiv:2206.10311.
- Marino, J., Chen, L., He, J. & Mandt, S. (2020). *Improving Sequential LVMs with Autoregressive Flows*. AABI 2020. arXiv:2010.03172.
- McNeil, A. J. & Frey, R. (2000). *Estimation of Tail-Related Risk Measures for Heteroscedastic Financial Time Series*. J. Empirical Finance 7, 271–300.
- McNeil, A. J., Frey, R. & Embrechts, P. (2015). *Quantitative Risk Management: Concepts, Techniques, and Tools* (2nd ed). Princeton.
- Papamakarios, G., Pavlakou, T. & Murray, I. (2017). *Masked Autoregressive Flow for Density Estimation*. NeurIPS 2017. arXiv:1705.07057.
- Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S. & Lakshminarayanan, B. (2021). *Normalizing Flows for Probabilistic Modeling and Inference*. JMLR 22(57):1–64. https://jmlr.org/papers/v22/19-1028.html.
- Rasul, K., Sheikh, A., Schuster, I., Bergmann, U. & Vollgraf, R. (2021). *Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows*. ICLR 2021. arXiv:2002.06103.
- Rasul, K., Seward, C., Schuster, I. & Vollgraf, R. (2021). *Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting*. ICML 2021. arXiv:2101.12072.
- Rezende, D. J. & Mohamed, S. (2015). *Variational Inference with Normalizing Flows*. ICML 2015. arXiv:1505.05770.
- Rosenbaum, M. et al. (2021). *Deep Learning Volatility: Fast Calibration of Rough Heston*. arXiv:2107.01611.
- Sarem Seitz. (2022). *Let's make GARCH more flexible with Normalizing Flows*. https://sarem-seitz.com/posts/lets-make-garch-more-flexible-with-normalizing-flows.html.
- Tai, W., Zhong, T., Mo, Y. & Zhou, F. (2021). *Learning Sentimental and Financial Signals with NFs for Stock Movement Prediction*. IEEE SPL 29, 414–418.
- Tepelyan, R. & Gopal, A. (2023). *Generative Machine Learning for Multivariate Equity Returns*. ICAIF 2023. arXiv:2311.14735.
- Xu, C. & Xie, Y. (2021). *Conformal Prediction Interval for Dynamic Time-Series*. ICML 2021. arXiv:2010.09107.
- Yalavarthi, V. K., Scholz, R., Born, S. & Schmidt-Thieme, L. (2024). *Probabilistic Forecasting of Irregular Time Series via Conditional Flows (ProFITi)*. ICML 2024. arXiv:2402.06293.
- Yang, Y. & Hospedales, T. M. (2023). *Mixture of Normalizing Flows for European Option Pricing*. UAI 2023. PMLR 216:2390–2399.

---

## Appendix: Session Artifacts

This survey was produced under the `epistemic-deconstructor` protocol at STANDARD tier. The full session artifacts live at:

```
analyses/analysis_2026-04-16_6d622783/
├── analysis_plan.md
├── scope_audit.md
├── domain_glossary.md
├── domain_metrics.json
├── domain_sources.md
├── observations.md
├── observations/obs_001 … obs_008.md
├── hypotheses.json
├── decisions.md
├── phase_outputs/phase_0…phase_5.md
├── state.md, progress.md
└── summary.md
```
