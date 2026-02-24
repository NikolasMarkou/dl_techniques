# Inductive Biases in Time Series Deep Learning: A Complete Practitioner's Guide

---

## Table of Contents

1. [Introduction](#introduction)
   - 1.1 [What is an inductive bias?](#11-what-is-an-inductive-bias)
   - 1.2 [Why inductive biases matter more for time series than for other domains](#12-why-inductive-biases-matter-more-for-time-series-than-for-other-domains)
   - 1.3 [The explicit vs. implicit distinction](#13-the-explicit-vs-implicit-distinction)
   - 1.4 [The bias-variance lens: why "more flexible" is not always better](#14-the-bias-variance-lens-why-more-flexible-is-not-always-better)
   - 1.5 [How this guide is organized](#15-how-this-guide-is-organized)
2. [The Complete Catalog of Inductive Biases for Time Series](#2-the-complete-catalog-of-inductive-biases-for-time-series)
   - 2.1 [Temporal locality](#21-temporal-locality)
   - 2.2 [Stationarity](#22-stationarity)
   - 2.3 [Periodicity and seasonality](#23-periodicity-and-seasonality)
   - 2.4 [Smoothness and continuity](#24-smoothness-and-continuity)
   - 2.5 [Causality](#25-causality)
   - 2.6 [Scale invariance](#26-scale-invariance)
   - 2.7 [Sparsity](#27-sparsity)
   - 2.8 [Compositionality](#28-compositionality)
   - 2.9 [Reversibility](#29-reversibility)
   - 2.10 [Multi-resolution structure](#210-multi-resolution-structure)
   - 2.11 [Autoregressive structure](#211-autoregressive-structure)
   - 2.12 [Conditional independence / Markov property](#212-conditional-independence--markov-property)
   - 2.13 [Trend-cycle-seasonal decomposition](#213-trend-cycle-seasonal-decomposition)
   - 2.14 [Distributional assumptions](#214-distributional-assumptions)
   - 2.15 [Symmetry and equivariance](#215-symmetry-and-equivariance)
   - 2.16 [Channel independence vs. channel mixing](#216-channel-independence-versus-channel-mixing)
   - 2.17 [Patching / tokenization structure](#217-patching--tokenization-structure)
   - 2.18 [Low-rank structure](#218-low-rank-structure)
   - 2.19 [Positional structure](#219-positional-structure)
   - 2.20-2.22 [Hierarchical processing, weight sharing, spectral bias](#220-222-additional-biases)
3. [How Architectures Encode Biases: A Comparative Analysis](#3-how-architectures-encode-biases-a-comparative-analysis)
   - 3.1 [RNNs (LSTM, GRU)](#31-rnns-lstm-gru)
   - 3.2 [Temporal Convolutional Networks (TCN/WaveNet)](#32-temporal-convolutional-networks-tcnwavenet)
   - 3.3 [Vanilla Transformers](#33-vanilla-transformers)
   - 3.4 [PatchTST](#34-patchtst)
   - 3.5 [iTransformer](#35-itransformer)
   - 3.6 [TimesNet](#36-timesnet)
   - 3.7 [N-BEATS and N-HiTS](#37-n-beats-and-n-hits)
   - 3.8 [DLinear and NLinear](#38-dlinear-and-nlinear)
   - 3.9 [State Space Models (S4 and Mamba)](#39-state-space-models-s4-and-mamba)
   - 3.10 [Comparative summary table](#310-comparative-summary)
4. [What Recent Research (2023-2025) Reveals About Inductive Biases](#4-what-recent-research-2023-2025-reveals-about-inductive-biases)
   - 4.1 [The DLinear shock and its aftermath](#41-the-dlinear-shock-and-its-aftermath)
   - 4.2 [Patching as the dominant tokenization strategy](#42-patching-as-the-dominant-tokenization-strategy)
   - 4.3 [The channel independence paradox](#43-the-channel-independence-paradox)
   - 4.4 [Foundation models and their design choices](#44-foundation-models-and-their-design-choices)
   - 4.5 [Scaling laws reveal architecture matters more than size](#45-scaling-laws-reveal-architecture-matters-more-than-size)
   - 4.6 [SSMs emerge as efficient alternatives](#46-ssms-emerge-as-efficient-alternatives)
5. [Matching Biases to Tasks and Data Characteristics](#5-matching-biases-to-tasks-and-data-characteristics)
   - 5.1 [Which biases matter most for which tasks](#51-which-biases-matter-most-for-which-tasks)
   - 5.2 [The flexibility-bias tradeoff](#52-the-flexibility-bias-tradeoff)
   - 5.3 [Dataset characteristics decision matrix](#53-dataset-characteristics-decision-matrix)
6. [Combining Multiple Inductive Biases Effectively](#6-combining-multiple-inductive-biases-effectively)
   - 6.1 [Compatible combinations](#61-compatible-combinations)
   - 6.2 [Potentially conflicting combinations](#62-potentially-conflicting-combinations)
   - 6.3 [Effective combination strategies](#63-effective-combination-strategies)
7. [Conclusion: Principles for Principled Model Design](#7-conclusion-principles-for-principled-model-design)

---

## 1. Introduction

### 1.1 What is an inductive bias?

An inductive bias is any assumption a learning algorithm makes *before* observing any training data that constrains the space of hypotheses it can learn. In the words of Tom Mitchell: without bias, a learner cannot generalize beyond the training examples it has already seen. Every machine learning model carries inductive biases whether its designer is conscious of them or not. A linear regression assumes the target is a linear combination of inputs. A decision tree assumes axis-aligned rectangular decision boundaries. A convolutional neural network assumes local connectivity, weight sharing, and translation equivariance. These are not deficiencies -- they are the mechanism through which prior knowledge about problem structure is injected into learning, enabling generalization from finite data.

Battaglia et al. (2018) formalized the concept for deep learning in their landmark paper "Relational inductive biases, deep learning, and graph networks", distinguishing biases along several dimensions: what entities are assumed to exist, what relationships hold between them, what rules govern their interaction, and how information is aggregated across entities. A fully-connected layer assumes all input dimensions interact but imposes no structure on how; a convolutional layer assumes locality and weight sharing; a recurrent layer assumes sequential processing with shared dynamics; an attention layer assumes pairwise interactions with data-dependent weighting. Each architectural primitive is a different bet about the structure of the problem.

For time series, the relevant entities are time steps, channels/variates, frequency components, and temporal scales. The relationships include temporal ordering, adjacency, periodicity, cross-channel correlation, and causal dependency. The rules govern how information flows, combines, and transforms across these entities and relationships. Understanding these dimensions is the key to making principled architectural decisions rather than blindly importing designs from NLP or vision.

The critical point is that **inductive bias is not optional**. A model with no inductive bias -- a truly unconstrained function approximator -- would require exponentially more data to learn patterns that a properly biased model can capture from modest samples. The No Free Lunch theorem guarantees that no single set of biases works best for all problems. The practitioner's task is therefore not to eliminate bias but to select biases that match the structure of their specific data and task. Getting this selection right has a larger effect on model performance than hyperparameter tuning, training tricks, or scaling model size.

### 1.2 Why inductive biases matter more for time series than for other domains

Time series modeling sits at a unique intersection of properties that amplifies the importance of inductive bias selection compared to vision or NLP. Understanding *why* this is the case is essential to appreciating the stakes of the design decisions cataloged in this guide.

**Data scarcity is structural, not incidental.** A single time series is one realization of a stochastic process. Unlike images -- where millions of independent samples are readily available -- a univariate time series of 10,000 steps provides at most a few thousand overlapping training windows, all drawn from the same trajectory and therefore highly correlated with one another. The effective sample size is often orders of magnitude smaller than the nominal dataset size. Strong inductive biases compensate for this scarcity by restricting the hypothesis space to functions consistent with known temporal structure, trading off hypothesis space volume for sample efficiency.

**The input lacks the semantic density of natural language.** A single word token in NLP carries rich, discrete semantic content; a single time step in a time series is just a scalar (or low-dimensional vector) with minimal standalone meaning. This asymmetry has profound architectural implications. When Zeng et al. (AAAI 2023) demonstrated that Transformers applied point-wise to time series barely outperform -- or underperform -- simple linear models, they exposed the failure of directly importing NLP's tokenization assumptions. The solution (patching, as pioneered by PatchTST) works precisely because it imposes a locality bias that groups adjacent time steps into semantically meaningful units, restoring the token-level information density that Transformers require to function effectively. This is not a minor implementation detail: it is a foundational insight about what makes a "good token" for self-attention.

**Non-stationarity is the default, not the exception.** Images don't change their statistical properties over time. Text corpora are approximately stationary at the corpus level. Time series, however, exhibit distribution shift as a matter of course: trends evolve, seasonal patterns drift, volatility clusters, and regime changes occur abruptly or gradually. Models that assume stationarity fail catastrophically on non-stationary data unless explicit mechanisms -- RevIN (Reversible Instance Normalization), adaptive normalization, differencing, decomposition -- are incorporated. This makes stationarity-handling biases not merely helpful but essential for production deployment. The community's systematic adoption of RevIN-like preprocessing between 2022-2025 is a direct response to this reality.

**Domain knowledge is rich and well-formalized.** Decades of statistical time series analysis -- Box-Jenkins methodology, exponential smoothing state-space models, structural time series models, spectral analysis, Granger causality -- provide a treasure trove of formalized knowledge about temporal structure. Classical statistics tells us about autoregressive dependencies, seasonal patterns at known periods, trend-seasonal-residual decomposition, impulse response functions, unit roots, and cointegration. Deep learning models that ignore this accumulated knowledge and attempt to learn everything from scratch are leaving information on the table. The most successful modern architectures -- N-BEATS with its polynomial/Fourier basis stacks, Autoformer with its moving-average decomposition, FEDformer with its frequency-domain attention -- are precisely those that translate classical statistical insights into neural architectural constraints.

**The temporal dimension creates unique evaluation pitfalls.** In time series, the temporal ordering of data creates strict constraints on what information is available at prediction time. Violating causality -- using future information to predict the past -- produces optimistic results that collapse in deployment. Even subtle leakage through normalization statistics computed on the full dataset (rather than only past data) can inflate metrics. The causality inductive bias is therefore not a modeling preference but a correctness requirement for any forecasting system intended for real-world use.

**The space of possible temporal patterns is vast but structured.** A time series of length 1000 with 64-bit float precision spans a space of $\mathbb{R}^{1000}$ -- combinatorially enormous. Yet real-world time series occupy a tiny, highly structured submanifold of this space: they are smooth, periodic, autoregressive, composed of trends and cycles, and exhibit cross-channel correlations that reflect physical or economic coupling. Inductive biases are the tool through which we tell the model "don't search all of $\mathbb{R}^{1000}$; search this structured submanifold instead."

### 1.3 The explicit vs. implicit distinction

This guide systematically categorizes each inductive bias along two dimensions of encoding. Understanding this distinction is critical for deciding how aggressively to impose a given assumption.

**Explicit (architectural) encoding** hard-codes the bias into the model's computational structure. The model *cannot* violate the assumption regardless of what data it sees, how long it trains, or how it is optimized. Examples include:

- Causal masking in self-attention: the model is structurally prevented from seeing future data
- Convolutional kernels: the model can only combine inputs within a local receptive field
- Fourier basis constraints (N-BEATS seasonality stack): the output is forced to lie in the span of sinusoidal functions
- Channel-independent processing: the model has no pathway through which cross-channel information can flow

Explicit biases are powerful when the assumption is known to be correct -- they eliminate entire regions of hypothesis space that would otherwise require data to rule out. But they create blind spots when the assumption is violated: a causal mask prevents the bidirectional context access needed for imputation; a fixed-period Fourier basis cannot represent aperiodic patterns; channel independence discards genuinely useful cross-variate correlations.

**Implicit (procedural) encoding** nudges the model toward solutions consistent with the bias through training procedures, regularization, data augmentation, or loss function design, without hard architectural constraints. The model *can* violate the assumption if the data strongly contradicts it, but doing so incurs a penalty or requires overcoming an optimization barrier. Examples include:

- Smoothness regularization: penalizes but doesn't prevent sharp transitions in predictions
- Temporal augmentation (jittering, time warping): encourages but doesn't guarantee shift invariance
- Curriculum learning (short sequences first): biases toward local pattern learning early but allows global patterns later
- Weight decay: implicitly favors simpler (lower-rank, smoother) functions but doesn't eliminate complex ones

Implicit biases are more flexible and forgiving than explicit ones, but they are weaker -- they require sufficient training dynamics and data pressure to take effect, and their influence is modulated by hyperparameters (regularization strength, augmentation probability) that must themselves be tuned.

The practical design principle is: **use explicit encoding when you have high confidence in the assumption, and implicit encoding when you want to encourage a property without ruling out alternatives.** Many successful architectures combine both: PatchTST explicitly enforces channel independence and locality (via patching) while implicitly encouraging temporal invariance through weight sharing across channels and positions.

### 1.4 The bias-variance lens: why "more flexible" is not always better

The classical bias-variance tradeoff provides the theoretical grounding for why inductive biases help. A model's expected prediction error decomposes into:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

**Bias** measures systematic error from simplifying assumptions -- a linear model applied to a nonlinear problem has high bias. **Variance** measures sensitivity to the particular training sample -- a large Transformer with few training examples has high variance. Adding inductive biases increases bias (by restricting the hypothesis space) but decreases variance (by preventing overfitting to noise). The optimal balance depends on the ratio of data quantity to problem complexity.

For time series, this tradeoff is especially acute because:

- Effective sample sizes are small (high variance risk), favoring stronger biases
- Temporal correlations between training windows reduce the effective information content further
- Non-stationarity means the target function itself changes, punishing models that memorize specific patterns rather than learning generalizable structure
- The signal-to-noise ratio in many domains (finance, IoT, climate) is low, making variance reduction critical

This explains the DLinear result: a single linear layer has *extremely* high bias (it can only represent linear functions) but near-zero variance. On datasets where the temporal structure is approximately linear after decomposition, this tradeoff is optimal. Complex Transformers have low bias but high variance, and with the modest effective sample sizes of standard time series benchmarks (ETTh1 has only ~17K hours), the variance cost overwhelms the flexibility benefit.

The actionable lesson: **when in doubt, bias harder.** Start with strong assumptions, validate each through ablation (remove the bias, measure performance change), and relax only those assumptions that demonstrably hurt performance. This is the opposite of the "scale and hope" approach that works in NLP but fails for most time series problems.

### 1.5 How this guide is organized

**Section 2** catalogs 22 distinct inductive biases relevant to time series, each with its formal definition, explicit encoding methods (architectural), and implicit encoding methods (training/augmentation/regularization). This is the reference section -- meant to be consulted when designing a new architecture or understanding an existing one.

**Section 3** analyzes how 9 major architecture families (RNNs, TCNs, vanilla Transformers, PatchTST, iTransformer, TimesNet, N-BEATS/N-HiTS, DLinear, S4/Mamba) encode these biases differently, with a comparative summary table. This section answers "which architecture embodies which assumptions?"

**Section 4** synthesizes key findings from the 2023-2025 research wave, including the DLinear revolution, patching's dominance, the channel independence paradox, foundation model design choices, scaling law discoveries, and SSM emergence. This is the "state of the field" section.

**Section 5** provides practical matching guidance: which biases matter for which tasks (forecasting, classification, anomaly detection, imputation), the flexibility-bias tradeoff in practice, and a dataset-characteristics decision matrix for selecting the right biases given your data's properties.

**Section 6** covers combining multiple biases, including compatible pairings, conflicting combinations to avoid, and effective composition strategies (encoder-decoder, stacked typed blocks, parallel branches, coarse-to-fine).

**Section 7** provides ready-to-use Keras 3 / TensorFlow 2.18 implementation patterns for the most important biases: causal convolutions, causal attention, series decomposition, RevIN, smoothness-aware loss, low-rank layers, and quantile loss.

**Section 8** distills the three core principles that emerge from the entire analysis.

---

## 2. The Complete Catalog of Inductive Biases for Time Series

The following catalog draws on Battaglia et al.'s (2018) foundational framework on relational inductive biases and synthesizes findings from 30+ papers published at top venues between 2020 and 2025. Each bias is defined, then paired with concrete explicit (architectural) and implicit (training procedure) encoding methods.

### 2.1 Temporal locality

Nearby time steps carry more predictive information than distant ones; statistical dependence decays with temporal distance.

**Explicit encoding:**
- **1D convolutions** with kernel size *k* restrict each output to *k* contiguous time steps. Weight sharing across positions provides translation equivariance.
- **Local attention windows** mask the attention matrix so position *i* attends only within a window of *w* neighbors: `M[i,j] = -inf if |i-j| > w/2`.
- **Power-law attention decay** (Powerformer, Hegazy & Erichson, 2025) reweights attention scores: `A'[i,j] = A[i,j] * f(|i-j|)` where *f* is a heavy-tailed decay function.
- **ALiBi** (Attention with Linear Biases) subtracts a linear penalty proportional to distance: `A[i,j] -= m * |i-j|`.
- **RNN hidden state** creates a Markovian bottleneck `h_t = f(h_{t-1}, x_t)` that inherently favors recent information due to gradient decay.

**Implicit induction:**
- Curriculum learning: train on short sequences first, gradually increase length
- Random cropping: sample shorter contiguous sub-windows during training
- Exponential decay weighting in loss: weight recent prediction errors more heavily
- Higher dropout rate on distant attention connections

### 2.2 Stationarity

Statistical properties (mean, variance) don't change over time -- or can be transformed to achieve this.

**Explicit encoding:**
- **RevIN** (Kim et al., ICLR 2022) applies symmetric normalize-then-denormalize: `x_hat = (x - mu_x) / sigma_x` at input, `y_hat = y_norm * sigma_x + mu_x` at output. Learnable affine parameters adapt the transformation.
- **Differencing layers** compute `Delta_x_t = x_t - x_{t-1}`, mirroring ARIMA's I(d) component.
- **Adaptive normalization variants** (SAN, FAN, WDAN) handle slice-level or frequency-adaptive normalization for complex non-stationarity.

**Implicit induction:**
- Instance-level z-score normalization per window (standard in PatchTST, iTransformer)
- Training on rolling windows from different time periods forces generalization across distribution shifts
- Magnitude warping augmentation: scale series by random smooth functions

### 2.3 Periodicity and seasonality

Regular cyclical patterns at fixed intervals (daily, weekly, yearly) dominate many real-world time series.

**Explicit encoding:**
- **N-BEATS interpretable seasonality stack** constrains output to Fourier basis: `y_s = Sum(a_k*cos(2*pi*k*t/P) + b_k*sin(2*pi*k*t/P))`. The network outputs only coefficients; basis functions are hardcoded.
- **FEDformer** (Zhou et al., ICML 2022) operates attention in the Fourier/wavelet domain, applying attention on randomly selected frequency modes.
- **Autoformer auto-correlation** replaces point-wise attention with period-based dependencies: `R(tau) = Sum(x_t * x_{t+tau})`, aggregating sub-series at dominant periods.
- **TimesNet 2D reshaping** transforms 1D series into 2D tensors based on FFT-detected periods, then applies 2D inception convolutions to model intra-period and inter-period patterns.
- **Periodic positional encodings** at multiple frequencies encode periodicity expectations.

**Implicit induction:**
- Calendar feature injection (day-of-week, hour-of-day, month) as exogenous covariates
- Phase-shift augmentation: shift series by `dt in {0, P/4, P/2, 3P/4}` for known period *P*
- Fourier auxiliary loss: `||FFT(y_hat) - FFT(y)||^2`

### 2.4 Smoothness and continuity

Adjacent predictions should be close in value; abrupt jumps are unlikely for most physical or economic processes.

**Explicit encoding:**
- **Polynomial basis** in N-BEATS trend stack: `y_t = Sum(theta_p * t^p)` for *p = 0,...,P* (typically P <= 3). Low-degree polynomials are inherently smooth.
- **Moving-average decomposition** (Autoformer): `Trend_t = (1/k)*Sum(x_{t+j})` acts as a low-pass filter.
- **Neural ODEs**: `dx/dt = f_theta(x,t)` guarantees continuity by construction since ODE solutions are always continuous.
- **Spline-based decoders** predict control points, then interpolate with cubic splines.

**Implicit induction:**
- Total variation regularization: `lambda * Sum|y_{t+1} - y_t|` penalizes abrupt changes
- Spectral regularization: penalize high-frequency components of predictions
- Weight decay (L2) biases toward smoother function approximations
- Jittering augmentation: `x' = x + epsilon, epsilon ~ N(0, sigma^2)`

### 2.5 Causality

Predictions at time *t* depend only on information from times <= *t*. The temporal arrow-of-time constraint.

**Explicit encoding:**
- **Causal masking in self-attention**: set `M[i,j] = -inf` for *j > i* before softmax, creating a lower-triangular mask. In Keras 3: `attn_layer(query=x, value=x, key=x, use_causal_mask=True)`.
- **Causal dilated convolutions** (WaveNet/TCN): left-only padding by `(kernel_size-1)*dilation` on the left, zero on the right. In Keras 3: `Conv1D(filters=64, kernel_size=3, padding='causal', dilation_rate=d)`.
- **Unidirectional RNNs**: `h_t = f(h_{t-1}, x_t)` processes left-to-right only.
- **DAG-based attention masking** (CAIFormer, 2025) uses a pre-estimated causal graph to restrict which variables can attend to which.

**Implicit induction:**
- Teacher forcing with autoregressive training
- Chronological train/val/test splits prevent future information leakage
- Scheduled sampling: gradually replace ground-truth with model predictions

### 2.6 Scale invariance

Similar patterns appear at different temporal and amplitude scales.

**Explicit encoding:**
- **Multi-scale dilated convolutions**: parallel convolution blocks with dilation rates *d in {1, 2, 4, 8, 16, ...}*, each capturing a different temporal scale simultaneously.
- **N-HiTS multi-scale patching**: different patch sizes in parallel stacks, with hierarchical interpolation. Each stack specializes in a different frequency band.
- **SCINet**: recursive splitting of series into even/odd subsequences at progressively coarser scales.
- **Wavelet architectures** (DeSpaWN): embed DWT into CNN layers with learnable wavelet filters.

**Implicit induction:**
- Multi-scale training with different input lengths (96, 192, 336, 720)
- Magnitude warping: multiply by random smooth curves
- Window slicing: training on subsequences of varying lengths
- Time warping: non-linear resampling of the time axis

### 2.7 Sparsity

Only a few features, time steps, or frequency components are relevant at any given time.

**Explicit encoding:**
- **ProbSparse attention** (Informer) selects top-*u* queries based on KL-divergence from uniform distribution, setting the rest to zero.
- **Top-k frequency selection** (FEDformer) randomly selects *s << d* Fourier modes.
- **Sparsemax/entmax attention** replacing softmax produces exactly-zero attention weights.
- **Sparse mixture-of-experts**: route each input through only *k* of *N* experts.

**Implicit induction:**
- L1 regularization: `lambda * ||W||_1` encourages weight sparsity
- Dropout: randomly zeroing activations at rate *p*
- Group Lasso: `lambda * Sum||W_g||_2` zeros entire channel groups
- Iterative magnitude pruning post-training

### 2.8 Compositionality

Complex patterns are structured combinations of simpler sub-patterns.

**Explicit encoding:**
- **N-BEATS doubly residual stacking**: block *k* operates on residual `r_k = r_{k-1} - backcast_k`. Final forecast = `Sum(forecast_k)`. Explicitly decomposes signal into additive components.
- **N-BEATS interpretable typed stacks**: trend stack (polynomial basis) + seasonality stack (Fourier basis), each constrained to its designated function class.
- **Autoformer/FEDformer MOEDecomp blocks** inserted between every attention layer: `Trend = AvgPool(x), Seasonal = x - Trend`. Progressive refinement through network depth.

**Implicit induction:**
- Residual connections: `y = F(x) + x` decomposes transformation into incremental additive steps
- Progressive training: train trend component first, freeze, then train seasonality on residual
- Multi-task learning encourages reusable compositional features

### 2.9 Reversibility

Patterns are consistent regardless of temporal direction; the model can leverage both past and future context.

**Explicit encoding:**
- **Bidirectional RNNs** (BiLSTM): forward and backward hidden states concatenated at each position.
- **Non-causal self-attention** (PatchTST, iTransformer): standard Transformer encoder attention without causal mask allows every position to attend to all others.
- **RevIN symmetric structure**: normalization/denormalization symmetry embodies invertibility.
- **Normalizing flows**: coupling layers (RealNVP, NICE) that are bijective by construction.

**Implicit induction:**
- Time-reversal augmentation: with probability *p*, reverse the series during training
- Bidirectional masked pre-training (BERT-style): mask random patches, reconstruct from surrounding context
- Symmetric loss functions (MSE, symmetric MAPE)

### 2.10 Multi-resolution structure

Important features exist at multiple temporal scales simultaneously.

**Explicit encoding:**
- **WaveNet stacked dilated convolutions**: layers with dilations *d = 1, 2, 4, 8, 16...* provide receptive fields at exponentially increasing scales.
- **N-HiTS hierarchical interpolation**: different expression lengths per stack level, each focusing on a specific frequency band.
- **WPMixer**: multi-resolution wavelet decomposition + patching + MLP mixing.
- **Feature Pyramid Networks for 1D**: multi-scale feature maps from different CNN levels combined via lateral connections.

**Implicit induction:**
- Training at multiple sampling rates simultaneously (hourly, daily, weekly)
- Multi-scale loss: compute loss at original + 2x, 4x, 8x downsampled versions
- Progressive growing of receptive field during training

### 2.11 Autoregressive structure

Future values depend directly on past values: `x_t = f(x_{t-1}, ..., x_{t-p}) + epsilon`.

**Explicit encoding:**
- **Autoregressive decoders**: generate one step at a time, feeding predictions back. GPT-style causal Transformer decoders.
- **DeepAR**: LSTM outputs distribution parameters per step `(mu_t, sigma_t) = f_theta(h_{t-1}, x_{t-1})`.
- **Explicit AR component** (LSTNet): parallel linear autoregressive module `y_t = Sum(w_i * x_{t-i})` summed with neural network output.

**Implicit induction:**
- Next-step prediction loss: `L = ||x_{t+1} - f(x_{1:t})||^2`
- Teacher forcing during training
- Scheduled sampling: gradual transition from ground truth to own predictions

### 2.12 Conditional independence / Markov property

Given the current state, the future is independent of the distant past. Only finite memory matters.

**Explicit encoding:**
- **Fixed context window**: `input_size = L` constrains the model to the last *L* steps.
- **State space models** (S4, Mamba): fixed-dimensional hidden state `h_t = A*h_{t-1} + B*x_t` implements a linear Markov chain.
- **LSTM/GRU forget gate**: `f_t = sigma(W_f*[h_{t-1}, x_t])` controls memory retention, implementing adaptive memory truncation.
- **Sliding window attention**: attention computed only within a window of size *w*.

**Implicit induction:**
- Training on short subsequences
- Truncated backpropagation through time (TBPTT)
- Exponential decay regularization on attention weights by distance

### 2.13 Trend-cycle-seasonal decomposition

`y_t = T_t + S_t + R_t` (additive) or `y_t = T_t * S_t * R_t` (multiplicative).

**Explicit encoding:**
- **N-BEATS interpretable**: trend stack with polynomial basis (degree <= 3) + seasonality stack with Fourier basis (*K* harmonics).
- **Autoformer decomposition block**: between every layer, `Trend = AvgPool_k(x)`, `Seasonal = x - Trend`.
- **STL preprocessing**: fixed STL decomposition before neural network, with three sub-series fed to separate encoder branches.
- **LaST**: VAE-based disentangled seasonal-trend representations learned from a probabilistic perspective.

**Implicit induction:**
- Multi-head architecture with separate smooth (trend) and periodic (seasonal) losses
- Augmenting decomposition components independently then recombining

### 2.14 Distributional assumptions

Assumptions about the probability distribution of values and forecast errors.

**Explicit encoding:**
- **Parametric output heads** (DeepAR): output `(mu, sigma)` for Gaussian; `(mu, sigma, nu)` for Student-*t*; `(alpha, beta)` for Negative Binomial. Loss = `-log p(y | params)`.
- **Quantile regression**: output multiple quantiles with pinball loss `rho_tau(y - q_tau)`. Non-parametric.
- **Normalizing flow output layers**: invertible transformations mapping base distribution to complex target.
- **Conformal prediction wrappers**: post-hoc calibrated prediction intervals without distributional assumptions.

**Implicit induction:**
- MSE loss implicitly assumes Gaussian noise (equivalent to MLE under `N(y_hat, sigma^2)`)
- MAE loss implies Laplace distribution
- Huber loss implies Gaussian center + Laplace tails
- Log-transform preprocessing assumes log-normal distribution

### 2.15 Symmetry and equivariance

Model output transforms predictably under input transformations.

**Explicit encoding:**
- **1D convolutions = translation equivariance**: weight sharing ensures `Conv1d(shift(x)) = shift(Conv1d(x))`.
- **Global average pooling = translation invariance**: averaging over time yields position-invariant output.
- **Channel-shared networks** (PatchTST CI) = permutation equivariance across channels: same weights applied to each channel independently.

**Implicit induction:**
- Temporal shift augmentation: `x'_t = x_{t+delta}` for random *delta*
- Channel permutation augmentation during training

### 2.16 Channel independence versus channel mixing

Whether multivariate channels should be modeled separately or with cross-channel interactions.

**Explicit encoding:**
- **Channel-independent (CI) design** (PatchTST): each channel processed as a separate univariate series through shared Transformer. No cross-channel attention. Acts as strong regularization.
- **iTransformer (inverted attention)**: treats each variable as a token, applies self-attention across channels (not time). Explicitly models cross-channel relationships.
- **Crossformer two-stage attention**: Stage 1: cross-time within each channel. Stage 2: cross-channel at each time step.
- **Graph Neural Networks** (STGNNs): model cross-channel dependencies through a graph structure.

**Implicit induction:**
- Channel dropout: randomly zero entire channels during training
- CI pre-training then channel-mixing fine-tuning
- Higher weight decay on cross-channel attention weights

### 2.17 Patching / tokenization structure

How raw values are segmented into meaningful tokens for processing.

**Explicit encoding:**
- **PatchTST patching**: non-overlapping patches of length *P* with stride *S*. Each patch linearly embedded to *d*-dim vector. Reduces sequence from *L* to *L/P* tokens while retaining local semantic information.
- **Convolutional tokenization**: `Conv1D` with stride *S* and kernel *K* creates learned tokens adaptively.
- **Chronos quantization tokenization**: scales time series by absolute mean, then quantizes into 4096 uniform categorical bins with cross-entropy loss -- treating forecasting as classification.

**Implicit induction:**
- Patch size hyperparameter tuning via cross-validation implicitly selects appropriate granularity
- Overlapping patches with varying strides as augmentation

### 2.18 Low-rank structure

Weight matrices or attention patterns have effective dimensionality much lower than nominal size.

**Explicit encoding:**
- **Factorized weight matrices**: replace `W in R^{m x n}` with `W = UV` where `U in R^{m x r}, V in R^{r x n}` with *r << min(m,n)*.
- **LoRA-style adaptation**: low-rank updates `W' = W + Delta_W` where `Delta_W = AB`. Applied to fine-tuning time series foundation models.
- **Bottleneck layers**: dimension reduction then expansion (e.g., *d -> d/4 -> d*) forces low-rank representation.

**Implicit induction:**
- Nuclear norm regularization: `lambda * ||W||_*` penalizes sum of singular values
- Weight decay implicitly biases toward low-rank solutions
- Dropout on hidden dimensions reduces effective rank

### 2.19 Positional structure

The model must understand where in time each observation occurs.

**Explicit encoding:**
- **Sinusoidal positional encoding** (Vaswani et al.): `PE(pos, 2i) = sin(pos / 10000^{2i/d})`. Fixed frequencies; generalizes to unseen lengths.
- **Learnable positional embeddings**: `pe = Parameter(zeros(1, max_len, d_model))`.
- **Rotary Position Embedding (RoPE)**: encodes position by rotating query/key vectors in 2D subspaces. Naturally handles relative positions.
- **Time2Vec**: `t2v(t) = [w0*t + phi0, sin(w1*t + phi1), ..., sin(wk*t + phik)]`.
- **Calendar embeddings**: learned embeddings for hour, day, month concatenated with temporal embeddings.
- **ALiBi**: linear distance penalty `A[i,j] -= m*|i-j|` combining positional awareness with locality.

**Implicit induction:**
- Relying on convolutions or recurrence alone for positional information (no explicit encoding)
- Temporal feature engineering (time-of-day, day-of-week as input features)

### 2.20-2.22 Additional biases

**Hierarchical processing**: stacking layers creates increasing abstraction. Early layers capture local patterns; deeper layers capture long-range features. This is inherent to deep architectures.

**Weight sharing / parameter tying**: CNNs share kernel weights across positions; RNNs share across time steps; PatchTST shares encoder weights across channels. Reduces parameters and assumes the same function applies everywhere.

**Spectral bias**: neural networks learn low-frequency components first (the "F-Principle"). Standard networks have a well-documented spectral bias. Frequency-domain architectures (FEDformer, FreTS) explicitly control which frequencies are modeled.

---

## 3. How Architectures Encode Biases: A Comparative Analysis

Understanding which architectures encode which biases is essential for matching model to data.

### 3.1 RNNs (LSTM, GRU)

RNNs enforce the strongest **sequential ordering** bias through `h_t = f(h_{t-1}, x_t)`. The fixed-size hidden state creates a **recency bias** -- recent observations influence predictions more due to gradient decay. Gating mechanisms (LSTM forget gate, GRU update gate) provide **adaptive memory timescales**. The same recurrence function at every step encodes **time-translation invariance**. However, RNNs lack explicit multi-scale decomposition, cannot parallelize during training, and compress all history through a bottleneck that limits very long-range dependencies.

### 3.2 Temporal Convolutional Networks (TCN/WaveNet)

TCNs combine **temporal locality** (kernel restricts receptive field), **translation equivariance** (weight sharing), **hierarchical multi-scale** processing (dilated convolutions with rates *1, 2, 4, 8, ...*), and **causality** (left-only padding). The receptive field grows exponentially with depth: for dilations `[1,2,4,8,16,32]` and kernel size 3, `RF = 1 + 2*2*63 = 253` steps. They lack global attention and explicit periodicity handling.

### 3.3 Vanilla Transformers

Self-attention computes **global pairwise relationships** between all time points -- a minimal locality bias. Without positional encoding, attention is **permutation-invariant**, fundamentally misaligned with temporal ordering. Zeng et al. (AAAI 2023) demonstrated this dramatically: **shuffling input sequences barely degraded Transformer forecasting performance**, proving they don't properly capture temporal order. Individual time points have low semantic content compared to NLP tokens, making point-wise attention maps potentially meaningless. Quadratic `O(L^2)` complexity limits scalability.

### 3.4 PatchTST

Patching segments time series into subseries-level patches analogous to ViT patches, encoding **local semantic grouping**. This solves the vanilla Transformer's "sparse semantic density" problem by ensuring each token contains meaningful local structure. **Channel independence** processes each variate through shared weights, acting as strong regularization that prevents overfitting to spurious cross-channel correlations. Patching reduces tokens from *L* to *L/P*, enabling attention over **much longer lookback windows** within the same compute budget. PatchTST achieved **21% MSE reduction** over prior Transformer models.

### 3.5 iTransformer

The key insight is **axis inversion**: instead of temporal tokens with multi-variate embedding, iTransformer treats each variable's entire time series as a variate token. Self-attention operates **across variables** (capturing multivariate correlations), while FFN operates **on each variate's temporal embedding** (learning temporal representations). This resolves the architectural misalignment that plagued conventional Transformers. Improvements are most pronounced on high-dimensional datasets where cross-variate dependencies matter (Traffic, PEMS).

### 3.6 TimesNet

TimesNet introduces a unique **2D periodicity-aware bias**. It reshapes 1D time series into 2D tensors based on FFT-discovered periods, where columns represent **intra-period variation** (within a cycle) and rows represent **inter-period variation** (across cycles). Standard 2D inception convolutions then capture both variation types simultaneously. This leverages well-established vision backbones for temporal processing. The architecture achieved state-of-the-art across **five mainstream tasks** (long/short-term forecasting, imputation, classification, anomaly detection).

### 3.7 N-BEATS and N-HiTS

N-BEATS encodes **basis expansion** and **compositionality** through doubly residual stacking: each block outputs a backcast (removes what it modeled from the signal) and a forecast (its contribution to the prediction). The interpretable version constrains trend stacks to polynomial bases and seasonality stacks to Fourier bases. N-HiTS extends this with **multi-resolution** bias: MaxPool with different kernel sizes across stacks creates frequency-band specialization. Blocks with few output coefficients capture low-frequency trends; blocks with many capture high-frequency details. N-HiTS reduces computational complexity dramatically for long horizons.

### 3.8 DLinear and NLinear

These encode the strongest possible bias: **linearity**. DLinear separates trend (moving average) from seasonal (residual), applies separate linear layers, and sums. NLinear subtracts the last value, applies a linear layer, and adds it back. Their success proved that many benchmarks have predominantly linear temporal patterns and that **excessive complexity hurts when the underlying signal is simple**.

### 3.9 State Space Models (S4 and Mamba)

S4 encodes a **continuous-time dynamical systems** prior through `dx/dt = Ax + Bu`, making it naturally robust to sampling rate changes (**resolution invariance**). The HiPPO initialization provides mathematically optimal polynomial projection of input history. **Mamba** extends S4 with **input-dependent dynamics** (selectivity): `Delta, B, C` become functions of the input, allowing content-dependent information routing. This creates a middle ground between Transformers (global attention, no sequential bias) and RNNs (strict sequential processing). Both achieve **O(L)** complexity.

### 3.10 Comparative summary

| Architecture | Primary biases | Temporal modeling | Cross-variate | Complexity |
|---|---|---|---|---|
| RNN/LSTM | Sequential, recency, adaptive memory | Hidden state recurrence | Requires explicit design | O(L) sequential |
| TCN/WaveNet | Locality, hierarchy, causality | Dilated causal convolutions | None (standard) | O(L) parallel |
| Vanilla Transformer | Global pairwise (permutation-invariant) | Attention + positional encoding | Via token design | O(L^2) |
| PatchTST | Local semantics, channel independence | Patch-level attention | None (by design) | O((L/P)^2) |
| iTransformer | Cross-variate correlations | FFN per variate | Attention across variates | O(V^2) |
| TimesNet | Multi-periodicity, 2D spatial | 2D conv on reshaped tensors | Limited | O(L) |
| N-BEATS/N-HiTS | Basis expansion, decomposition, multi-scale | FC stacks with residual | None | O(L) |
| DLinear | Linearity, decomposition | Direct linear mapping | None | O(1) |
| S4/Mamba | Continuous-time dynamics, selective memory | Structured SSM | Channel independent | O(L) |

---

## 4. What Recent Research (2023-2025) Reveals About Inductive Biases

### 4.1 The DLinear shock and its aftermath

Zeng et al.'s 2023 finding that a single linear layer outperforms all Transformer-based forecasting models catalyzed a fundamental rethinking of the field. The key insight was that self-attention's **permutation invariance** is misaligned with time series' temporal ordering. The community response diverged: PatchTST showed Transformers work when patching provides locality bias; iTransformer showed the problem was which dimension attention operates on, not attention itself; and foundation model builders (TimesFM, Chronos, MOMENT) demonstrated that scale + appropriate tokenization can overcome architectural limitations.

### 4.2 Patching as the dominant tokenization strategy

**Patching has become near-universal** across top-performing models. It provides the locality bias point-wise tokenization lacks while enabling efficient long-context processing. Every major foundation model uses it: MOMENT (fixed-length disjoint patches with masked prediction), TimesFM (patch-based with residual blocks), Timer (patch-level generation), MOIRAI (patch-based with masked encoder). Typical patch lengths range from **16 to 64** for most applications.

### 4.3 The channel independence paradox

Han et al. (TKDE 2024) provided theoretical analysis showing channel-independent (CI) models consistently outperform channel-dependent (CD) models despite discarding cross-channel correlations. **CI trades capacity for robustness**: it prevents overfitting to non-stationary cross-channel correlations that don't generalize. The emerging resolution is selective channel mixing: LIFT (ICLR 2024) identifies locally stationary lead-lag relationships between variates; C-LoRA (CIKM 2024) adds channel-aware low-rank adaptation as a plug-in.

### 4.4 Foundation models and their design choices

The 2024-2025 foundation model wave reveals interesting inductive bias choices:

- **Decoder-only** (TimesFM, Timer, Sundial): GPT-style next-token prediction with autoregressive bias. Best for forecasting.
- **Encoder-only** (MOMENT): BERT-style masked prediction. Better for general analysis tasks (classification, anomaly detection, imputation).
- **Encoder-decoder** (Chronos): T5-based with quantized tokenization. Chronos's minimalist approach -- 4096 categorical bins with cross-entropy loss, no time-series-specific design -- achieves surprising zero-shot results, suggesting LLM architectures possess transferable sequential reasoning.
- **Sundial** (ICML 2025 Oral) scales to **1 trillion time points** of pre-training data, establishing that data scale matters but architecture-dependent scaling exponents vary significantly.

### 4.5 Scaling laws reveal architecture matters more than size

Yao et al. (ICLR 2025) showed that log-likelihood loss follows power-law scaling for time series foundation models, but the scaling exponent **differs significantly by architecture**. Simply making models bigger doesn't guarantee proportional improvement -- unlike NLP. Data quality and architectural fit matter as much as parameter count.

### 4.6 SSMs emerge as efficient alternatives

Mamba-based time series models (S-Mamba, MambaTS, Mamba4Cast) offer **O(L)** complexity with selective state-space mechanisms. MambaTS (ICLR 2025) argues causal convolution is unnecessary for forecasting and introduces Variable-Aware Scan (VAST) for discovering optimal cross-variate scan orders. Mamba4Cast demonstrates zero-shot forecasting trained solely on synthetic data, suggesting SSMs' continuous-time priors generalize well.

---

## 5. Matching Biases to Tasks and Data Characteristics

### 5.1 Which biases matter most for which tasks

**Forecasting (short-horizon)** demands locality, causality, and autoregressive structure. Recent values are most predictive. Conv1D with small kernels or shallow LSTMs suffice. Smoothness regularization prevents erratic predictions.

**Forecasting (long-horizon)** requires multi-scale hierarchy, decomposition, and stationarity handling. Dilated convolutions or patch-based Transformers capture patterns at multiple scales. RevIN or differencing handles distribution shift. The **direct multi-output strategy** avoids error accumulation of recursive methods.

**Classification** benefits most from translation invariance, locality, and multi-scale feature extraction. Research on fMRI classification (2025) found **CNNs achieved the highest AUC-ROC** because discriminative information is carried by local patterns. Transformers performed near chance with limited data (~180 subjects) while CNNs excelled.

**Anomaly detection** relies on reconstruction bias (autoencoders assume normal data compresses well) or prediction bias (models trained on normal data produce high error on anomalies). Hybrid Conv-BiLSTM autoencoders perform best for multivariate industrial time series. Bottleneck dimensionality is the critical tuning parameter.

**Imputation** needs bidirectional context (non-causal attention), smoothness, and cross-channel information. Unlike forecasting, imputation benefits from full attention where every time step attends to all observed values. Graph-based methods (GRIN) condition on cross-series correlations via message passing.

### 5.2 The flexibility-bias tradeoff

**Strong biases help** when data is limited (CNNs outperform Transformers in small-sample settings), domain knowledge is reliable (known seasonality makes Fourier bases powerful), and tasks have clear structure (causality for forecasting). **Strong biases hurt** when assumptions are violated (causal masking prevents bidirectional access needed for imputation), data is rich enough for the model to learn structure (Transformers catch up with sufficient data), and distributions shift (hardcoded periodicity fails when period length changes).

The practical test: **large train-validation gap** indicates high variance (model too flexible, needs more bias). **High error on both** indicates high bias (model too constrained, needs more flexibility). Ablation studies -- removing each bias component and measuring impact -- are the gold standard for validating bias appropriateness.

### 5.3 Dataset characteristics decision matrix

| Dataset property | Recommended biases | Go-to architectures |
|---|---|---|
| Short series (< 100 steps) | Strong locality, pooling, simplicity | Conv1D stack, shallow LSTM, MLP |
| Long series (> 1000 steps) | Hierarchical, patch-based, decomposition | PatchTST, N-HiTS, S4/Mamba |
| Non-stationary | Instance normalization, decomposition, differencing | RevIN + any backbone |
| Strong periodicity | Fourier basis, FFT-based reshaping | N-BEATS interpretable, TimesNet |
| Many correlated variables | Graph-based relational bias, variate-token attention | STGNNs, iTransformer |
| Irregular sampling | Continuous-time dynamics, time-aware attention | Neural ODEs, GRU-D, S4 |
| Data-poor (< 1K samples) | Maximum architectural bias, transfer learning | CNNs + pretrained foundation models |
| Data-rich (> 100K samples) | Minimal bias, maximum flexibility | Large Transformers, Mamba |

---

## 6. Combining Multiple Inductive Biases Effectively

### 6.1 Compatible combinations

**Causality + locality** combine naturally in causal dilated convolutions (TCN). **Decomposition + periodicity** separate trend/seasonality, then apply polynomial bases to trend and Fourier bases to seasonality. **Hierarchical processing + residual connections** (N-BEATS) stack blocks where each removes its captured signal. **Graph structure + temporal sequence** (STGNNs) combine GNNs for relational processing with RNNs/TCNs for temporal processing.

### 6.2 Potentially conflicting combinations

**Strong locality + global attention** may compete if applied naively. The solution is hierarchical combination: local processing first (CNN), then global attention (Transformer). **Causal + bidirectional** are fundamentally incompatible for the same operation but can coexist: causal in the primary prediction path, bidirectional in auxiliary pre-training tasks.

### 6.3 Effective combination strategies

- **Encoder-decoder**: encode with one bias type (CNN for local features), decode with another (LSTM for sequential generation)
- **Stacked typed blocks** (N-BEATS): trend stack + seasonality stack + generic stack, each with different constraints
- **Parallel branches** (Inception-style): multiple Conv1D branches with different kernel sizes, outputs concatenated
- **Coarse-to-fine**: predict at low resolution first (strong smoothness bias), then refine at high resolution
- **Knowledge distillation**: transfer a CNN teacher's locality bias into a student MLP without architectural constraints

---

## 7. Conclusion: Principles for Principled Model Design

The 2023-2025 revolution in time series deep learning teaches three fundamental lessons.

First, **inductive bias selection dominates architecture choice**: DLinear's linear layers with decomposition bias outperformed complex Transformers; PatchTST's locality bias rescued Transformers; iTransformer's axis inversion unlocked cross-variate modeling. The architecture is the medium through which biases are expressed, not the end in itself.

Second, **four biases are near-universally beneficial**: patching (local semantic grouping), instance normalization (stationarity handling), decomposition (trend-seasonal separation), and residual connections (compositional refinement). These should be default components in any time series model regardless of the backbone architecture.

Third, **the optimal bias configuration is data-dependent, not task-dependent**. A highly periodic, stationary, data-poor time series demands strong Fourier bases and simple architectures regardless of whether the task is forecasting or classification. A high-dimensional, non-stationary, data-rich multivariate dataset benefits from flexible architectures with selective channel mixing regardless of the downstream application. The practical workflow is: characterize your data (stationarity tests, autocorrelation analysis, cross-channel correlation structure, sample size), match biases to characteristics using the decision matrix above, implement them in Keras 3 using the provided patterns, and validate each bias through ablation studies measuring its individual contribution.