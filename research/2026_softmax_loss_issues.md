# The Role of Quantization in Loss Spikes: Consolidated Thread and Paper Brief

Nov 4, 2026 · @Nikolas Markou

## Overview

Late, unrecoverable loss spikes in LLM pretraining ("time-bomb spikes") are caused by BF16 quantization error in the LM head, which grows as Adam pushes the mean output logit to extreme values; training the LM head with SGD with momentum (SGD-M) removes the drift and the spikes.

**Source material consolidated here**

- Paper: Zaiqian Chen (led the project; tagged in the thread as @tingtang222), Aditya Somasundaram, Sophie Li, Sean McLeish, John Kirchenbauer, Tom Goldstein, Micah Goldblum. *The Role of Quantization in Loss Spikes*. arXiv:2610.38277v2 \[cs.LG\], 2 Nov 2026. [arxiv.org/abs/2610.38277](https://arxiv.org/abs/2610.38277). Code: [github.com/tingtang2/quantization-loss-spikes](https://github.com/tingtang2/quantization-loss-spikes). Corresponding author: zc2666@columbia.edu.
- Affiliations: Columbia University (Chen, Li, Goldblum), Mohamed bin Zayed University of Artificial Intelligence (Somasundaram), University of Maryland (McLeish, Goldstein), Vector Institute (Kirchenbauer).
- X thread: 13-post thread by Micah Goldblum (@micahgoldblum, Professor at Columbia University), plus the public replies beneath it.
- Companion paper: Barron, Gnanapragasam, Tschannen (Google DeepMind). *A Fix for Softmax and Cross-Entropy*. [arXiv:2610.31642](https://arxiv.org/abs/2610.31642) v3, 29 Oct 2026. Tested as a mitigation in the main paper (App. C.4); covered in its own section below.

**Core argument in five steps**

1. Cross entropy gradients for the LM head sum to zero across the vocabulary.
2. Adam's per-coordinate preconditioning breaks that zero sum, so the mean output embedding and the mean output logit drift.
3. Adding a constant to all logits is harmless in exact arithmetic: softmax, loss and gradients are unchanged.
4. In BF16, the spacing between representable numbers grows with magnitude. A large mean logit erases the small differences between logits that set the softmax probabilities.
5. Quantization error rises by orders of magnitude, then the loss diverges and never recovers.

The companion paper by Barron et al. describes a related but distinct failure in the same family; see [Consolidated findings across both papers](#consolidated-findings-across-both-papers).

**Key findings**

| Finding | Evidence |
| --- | --- |
| Fast, transient spikes are largely suppressed by learnable RMSNorm gains | 5 matched seed pairs, 162M models (Fig. 2, 13) |
| Time-bomb spikes emerge late in overtraining and do not recover | 404M models, 96B tokens, about 12x Chinchilla (Fig. 3, 16) |
| Lower LM head precision brings the failure earlier | BF16 true fails first, BF16 mixed later, FP32 stable (Fig. 1, 6, 19) |
| LM head SGD-M (momentum 0.99) prevents time bombs and stays competitive in loss | 404M LR sweep, 28M precision runs, 1.4B run (Fig. 4, 5, 15) |
| z-loss, mu-centering and LM head SGD-M all keep quantization error bounded | 28M models, 5 seeds, BF16 mixed and true (Fig. 7, 20) |

## Consolidated findings across both papers

The two papers describe the same family of failure, not the same bug: in both, finite precision breaks the zero-sum structure of softmax gradients and Adam amplifies the small error into a loss spike, but the error arises at different points, from different growing quantities, and needs different fixes.

### The shared mechanism

Both failures follow one template:

1. **Softmax depends only on differences; floating point resolves only relative to magnitude.** Any computation that forms a large number before taking a small difference loses the difference. Barron et al.: 1 + e^(-m) rounds to 1. Chen et al.: mean logit + small per-token deviation rounds onto a coarse BF16 grid.
2. **The zero-sum gradient is a structural invariant.** Softmax cross-entropy gradients sum to zero over classes. Barron et al. break it by rounding the target entry to 0; Chen et al. show Adam's per-row rescaling breaks it in the LM head update.
3. **Adam turns a small systematic error into a large step.** Normalizing each coordinate by its own history means tiny but consistent gradients get full-size updates. Barron et al.: v decays during the gradient blackout, then a restored gradient divided by a tiny sqrt(v) + eps spikes the loss; in distillation a 1e-8 residual gradient becomes real drift. Chen et al.: rescaled zero-sum gradients produce persistent mean drift.
4. **Lower precision moves the threshold earlier; higher precision postpones but does not abolish it.** Barron et al.: cliffs at margin 5.5 (BF16), 16.6 (FP32), 36.7 (FP64), and FP64 standard cross-entropy still fails at eps = 1e-16. Chen et al.: BF16 true fails before BF16 mixed, FP32 is stable only over the horizons studied.
5. **Weight decay and Adam's eps set how far logits can grow.** Barron et al.: with weight decay, margins settle at log(1/eps). Chen et al.: weight decay delays time bombs (their time-bomb runs use none).

### Where they differ

| Aspect | Softmax Collapse (Barron et al.; Liu et al.; Prieto et al.) | Time-bomb spike (Chen et al.) |
| --- | --- | --- |
| Growing quantity | Logit margin m: the target pulls away from the other logits | Mean logit: all logits shift together; r = abs(mu\_z) / sigma\_z grows |
| Does the growth change the predicted distribution? | Yes, the model becomes more confident | No, a common shift is invisible in exact arithmetic |
| Where precision is lost | Inside log-softmax / cross-entropy (1 + e^(-m)) | In the LM head matmul output, before softmax runs |
| What breaks first | Loss-gradient zero-sum (target entry rounds to 0) | Update zero-sum (Adam rescaling), then logit resolution |
| Root cause | Rounding in the loss implementation | Optimizer geometry (Adam), exposed by rounding |
| Spike shape | Recurring, recoverable (slingshot) | Late, monotonic, unrecoverable |
| Setting studied | Grokking on modular arithmetic, small models, near-perfect confidence | LM pretraining, 28M to 1.4B, eval loss around 3 to 4 nats |
| Fix | Rewrite softmax / cross-entropy in log space | Stop the drift (LM head SGD-M, mu-centering), bound logits (z-loss), or raise LM head precision |

The distillation bug, Softmax Cancellation, is a third member of the family: two nearly equal probabilities subtracted after rounding, with Adam amplifying the residual.

### Evidence that the fixes are not interchangeable

| Fix | Softmax Collapse / slingshot | Time-bomb spike | Softmax Cancellation |
| --- | --- | --- | --- |
| Fixed cross-entropy (Barron et al.) | Eliminated, all precisions | BF16 mixed: no time bomb within 16B tokens, but logit norm and error keep growing. BF16 true: delayed, all 5 seeds still fail | Eliminated |
| Zero-sum projection (Liu et al.) | Spikes removed, but margins stall near 18 and accuracy drops below 50% at eps ≤ 1e-12 | Not tested | Not tested |
| StableMax (Prieto et al.) | Mitigated; margins blow up to about 1e4, accuracy lags | Not tested | Not tested |
| Higher precision | Postpones the cliff (FP64 fails at eps = 1e-16) | FP32 LM head stable over the horizons studied, at high cost | FP64 residual too small to matter |
| LM head SGD-M | Not tested; Liu reports (thread reply) that switching to SGD does not remove slingshot spikes in their setting | Prevented | Not tested |
| mu-centering | Not tested | Prevented | Not tested |
| z-loss | Not tested | Prevented | Not tested |
| Weight decay / larger eps | eps ≥ 1e-7 keeps the FP32 equilibrium below the cliff; weight decay caps margins at log(1/eps) | Weight decay delays onset | Not tested |

The pattern is a rough double dissociation: the loss-level fix does not stop LM head drift, and the optimizer-level fix reportedly does not stop collapse. The evidence comes from different setups, so this is suggestive rather than a controlled head-to-head test.

### Unified picture

A softmax output layer trained with Adam in low precision has two independent ways to push logits into a region where rounding erases the differences that matter:

- **Confidence growth (margin):** legitimate learning drives the target far above the rest. Breaks the loss computation. Fixed by computing the loss in log space.
- **Drift growth (mean):** Adam moves every logit together with no effect on predictions. Breaks the logit computation itself. Fixed by preserving or restoring the zero mean, bounding logit size, or raising precision.

Each fix removes one path and leaves the other open, which explains why Barron et al.'s fix only delays time bombs and why Chen et al.'s fixes say nothing about slingshots.

### Combined practical recipe for BF16 training

This recipe is a synthesis of the two papers; neither tested all of it together.

1. Use a numerically fixed log-softmax and cross-entropy. Barron et al. report no meaningful slowdown, so it is low-risk insurance against loss-level rounding.
2. Control LM head mean drift with LM head SGD-M (momentum 0.99, tuned head LR), mu-centering, or z-loss. SGD-M also saves one optimizer buffer.
3. Check Adam's eps against the precision cliff where the loss is computed: log(1/eps) above c\_d means margins can reach the cliff before equilibrium.
4. Keep weight decay unless there is a reason not to; both papers find it limits logit growth.
5. Monitor, from early in the run: output logit norm and mean, r = abs(mu\_z) / sigma\_z, RMS(z\_bf16 - z\_fp32) on a sample of hidden states, and the share of tokens whose margin exceeds the cliff.
6. Validate stability at the full intended horizon; time bombs can stay hidden in short pilot runs.

### Open questions the combination raises

- How often Softmax Collapse occurs in LM pretraining. Barron et al. call such extreme margins uncommon in large-scale training, and Liu et al. saw logit growth but no slingshot spikes in GPT training; near-deterministic tokens may still cross the BF16 cliff of 5.5 (an inference, untested).
- Whether fixed cross-entropy plus LM head SGD-M together beat either alone at long horizons.
- Whether Barron et al.'s weight-decay and eps equilibrium also bounds Chen et al.'s mean drift, which would explain why weight decay delays time bombs.
- Whether both effects worsen in FP8, as both lines of work imply.

## Background concepts

**The LM head.** The language modeling (LM) head is the final layer mapping the last hidden state to one score (logit) per vocabulary token. With vocabulary size V, hidden state h in R^d and output embedding matrix E in R^(V x d) (one row e\_i per token), the logits are z = Eh. "Untied" embeddings, used in the paper, mean E is a separate matrix from the input embedding table.

**Softmax and cross entropy.** With one-hot label y (exactly one entry equals 1):

```latex
p_i = \mathrm{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^{V} \exp(z_j)}, \qquad \mathcal{L}_{CE} = -\sum_{k=1}^{V} y_k \log p_k
```

Softmax depends only on differences between logits. Adding the same constant c to every logit leaves p, the loss and all gradients unchanged in exact arithmetic.

**Number formats.**

| Format | Significand precision | Exponent bits | Gap between neighbors near 1 | Gap near 1024 |
| --- | --- | --- | --- | --- |
| FP32 | 24 bits (23 stored + implicit) | 8 | about 1.2e-7 | about 1.2e-4 |
| BF16 | 8 bits (7 stored + implicit) | 8 | about 0.008 | 8 |

BF16 keeps FP32's exponent range but only 8 bits of significand. For a value in \[2^e, 2^(e+1)), the spacing between consecutive representable values (one ULP, unit in the last place) is 2^(e-7). Relative precision is fixed, so absolute precision worsens as numbers get larger.

**Precision regimes used in the paper (applied to the LM head; all other layers use BF16 mixed).**

| Regime | LM head weights and optimizer state | LM head matmul (GEMM) | Flax / PyTorch analogue |
| --- | --- | --- | --- |
| FP32 | FP32 | FP32 | full precision |
| BF16 mixed | FP32 master copy | BF16 (weights and activations cast at runtime) | `param_dtype=float32`, `dtype=bfloat16`; `torch.autocast` |
| BF16 true | BF16 (no FP32 master copy) | BF16 | pure BF16 training |

BF16 mixed is the standard setting for large-scale pretraining: faster matmuls and lower activation memory with full-precision weight updates.

**Adam / AdamW.** Adam keeps a running mean of gradients (first moment m, coefficient beta1) and of squared gradients (second moment v, coefficient beta2). Each parameter's update is roughly m / (sqrt(v) + eps), so every coordinate gets its own effective step size. AdamW decouples weight decay from the gradient update. In the LM head, rare tokens have small v, so Adam scales their updates up. This adaptivity is a key reason Adam beats plain gradient descent on language models (Kunstner et al., 2024).

**SGD with momentum (SGD-M).** Update = learning rate x running average of gradients. No per-coordinate rescaling, so any linear structure in the gradient (such as summing to zero) carries over to the update. In steady state, momentum beta amplifies a constant gradient by 1/(1 - beta): 10x at 0.9, 100x at 0.99.

**Chinchilla-optimal budget.** About 20 training tokens per parameter (Hoffmann et al., 2022). Training well beyond it is called overtraining; most modern models are overtrained to cut inference cost.

## Two flavors of loss spikes

The paper separates LM head instabilities into two types; only the second survives modern architectures.

| Property | Fast spikes | Time-bomb spikes |
| --- | --- | --- |
| Shape | Abrupt loss jump, then recovery | Slow, sustained, monotonic rise in eval loss |
| When | Any time, at high learning rates | Late, after prolonged (over)training |
| Recovery | Yes | No (terminal) |
| Logit signature | Sharp swings in the mean output logit | Accelerating growth of output logit norm and mean |
| Triggered by | Specific tokens in specific contexts, on top of an unstable parameter/optimizer state | Accumulated Adam-induced drift plus BF16 resolution loss |
| Suppressed by learnable RMSNorm gains | Largely yes | No |
| Made worse by | Parameter-less RMSNorm | Higher learning rate, longer horizon, smaller batch, no weight decay, lower precision |
| Cost | Noise and wasted steps | A run that looks healthy until most of the compute is spent, then fails |

Why time bombs matter more: they end up "wasting compute and infuriating the researchers" (thread, post 3). Because they grow with training horizon, they become more relevant as models are trained far beyond the compute-optimal budget (e.g. Qwen3; Everett and Qiu, 2026).

Learning rate also selects the type of instability: Fig. 8 shows 1.4B runs (10B tokens) at peak LR 0.01, 0.07 and 0.1 behaving differently. The authors use LRs near 0.07 to elicit fast spikes for study.

## Experimental setup

Spikes are reproduced in 28M to 1.4B parameter models by training at elevated learning rates, the standard small-scale proxy for large-scale instability (Wortsman et al., 2024).

**Shared configuration**

| Component | Setting |
| --- | --- |
| Architecture | Modernized GPT-2: rotary positional embeddings (RoPE), GELU, RMSNorm, untied embeddings, no biases |
| Attention stabilizer | QK-Norm in every run, removing attention-logit spikes so output-logit instability is isolated |
| Data / tokenizer | FineWeb / GPT-2 tokenizer (V = 50,260 in the analysis) |
| Loss | Standard cross entropy unless noted; z-loss runs use lambda = 1e-4 |
| Optimizer | AdamW, eps = 1e-8, beta1 = 0.9, beta2 = 0.95 |
| Schedule | Linear warmup, cosine decay, global gradient-norm clipping at 1 |
| Weight decay / batch | 1e-4 / 262,144 tokens (defaults) |
| Framework | JAX + Flax |
| Precision | BF16 mixed by default |
| Hardware | Up to 32 Google Cloud TPU v6e or v4; 4x NVIDIA B200 for 404M time-bomb runs; 8x H100 and 8x A100 also used for tuning runs |

**Experiment-specific settings**

| Experiment | Model size | Tokens | Tokens / param | Norm type | Batch (tokens) | Weight decay |
| --- | --- | --- | --- | --- | --- | --- |
| Fast spikes | 162M, 404M, 1.4B | About Chinchilla-optimal (162M); 10B (1.4B) | about 20 | Parameter-less RMSNorm | 262,144 | 1e-4 |
| Time-bomb LR sweep | 404M | 96B | about 240 (12x Chinchilla) | Learnable RMSNorm | 131,072 | 0 |
| Precision and mitigation | 28M | 16B | about 570 (28x Chinchilla) | Learnable RMSNorm | 131,072 | 0 |

Time-bomb runs use a smaller batch and no weight decay on purpose: preliminary experiments showed smaller batches bring the onset earlier and weight decay delays it (also noted by Wortsman et al., 2024).

Learning-rate grid (from Wortsman et al., 2024): 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1.

**Measurements**

- Output logit norm and output logit mean, tracked over training.
- Logit quantization error: at each logging step, recompute the LM head logits in FP32 (FP32 GEMM) from the same hidden states, then take RMS(z\_bf16 - z\_fp32). Analogous to reconstruction error in quantization work such as OPTQ (Frantar et al., 2023).
- Predictive-distribution quantization error: KL(p\_fp32 || p\_bf16) between the two softmax outputs, following Lotfi et al. (2026).
- Spike score (following OLMo 2): percentage of steps whose loss exceeds 7 standard deviations above a trailing 1000-step window.
- Time-bomb definition: three consecutive evaluations after the training midpoint, each at least 0.01 above the running-best eval loss.

## Fast spikes in detail

Fast spikes come from an unstable parameter and optimizer state, not from bad data: removing the triggering tokens or batches hides or postpones them but never cures them, while learnable RMSNorm gains largely remove them.

**1. A few outlier tokens carry the spike (App. A.2, Fig. 9).** During a spike, a small subset of tokens has cross entropy above 100 nats and accounts for most of the loss jump. Outliers are flagged with a robust z-score over a rolling window of the preceding 256 steps, because token-loss distributions are heavy-tailed (an adaptation of Karpathy's 2024 outlier-skipping idea). Masking as little as 0.2% of tokens removes the visible spikes in a 162M model at peak LR 0.15, but consistently hurts training-loss convergence.

**2. Spike events are extremely sensitive to context (App. A.3, Table 1, Fig. 10).** The highest-loss prediction at a spike peak (162M run) had context "...the commencement of the 50th Commemoration of that War.The Command", target token "ery" (completing "Commandery").

|  | Original context (spiking) | Final token "Command" replaced by a comma |
| --- | --- | --- |
| Top-5 predictions (logit) | ible (782.05), ments (669.86), ured (640.67), ive (603.56), ivism (595.26) | the (95.01), in (94.12), however (93.80), until (92.99), and (92.96) |
| Loss on target "ery" | 610.00 nats | 12.73 nats |
| Logit mean / std over 50,260 tokens | -57.455 / 121.866 | 77.459 / 6.874 |

The swap shrinks the logit standard deviation 17x, back to the range seen at non-spiking steps. Similar recoveries occur when swapping other positions or using a single space or empty token as the replacement. The model places nearly all its probability on one wrong token, and tiny context changes undo it. The perturbations needed are non-systematic, so they do not yield a practical train-time strategy.

**3. Fast spikes coincide with swings in the mean output logit (App. A.4, Fig. 11).** A spiking 162M run shows sharp swings in mean output logit at spike onset; a matched run differing only in seed shows none.

**4. Skipping triggering batches only delays spikes (Fig. 12).** Following Karpathy's procedure (skip any batch with loss above 7 standard deviations of a rolling 1000-step window and revert the optimizer state), reverting to either the previous or the second-previous optimizer state still leads to new mean-logit swings and spikes. This extends PaLM's observation that replaying the same batches from other checkpoints does not reproduce spikes: the data triggers, but the parameters and optimizer state are the cause.

**5. Learnable RMSNorm gains largely suppress them (Sec. 2.4, Fig. 2, 13).** Parameter-less RMSNorm (no learnable gain, used in OLMo and some other suites) is the default in the fast-spike experiments. Re-running a 162M spiking configuration with learnable gains, same seeds and hyperparameters, removes the fast spikes, the mean-logit swings and the large logit-norm growth in all 5 seed pairs. The 5 parameterized runs nearly overlap, showing low seed variance.

**6. Mitigations also work in this regime (App. A.6).** See the Mitigations section for the learning-rate sweep, spike scores and 1.4B results.

## Time-bomb spikes: how they emerge

Time bombs persist with learnable RMSNorm gains and appear only after long training: two 404M runs track each other for most of 96B tokens, then the higher-LR run stalls and its eval loss climbs while its output logit norm accelerates (Fig. 3; thread post 4).

**Learning rate (Fig. 16).** Across the 404M sweep (3e-4 to 3e-1, 96B tokens), all runs follow similar loss curves for much of training. Higher LRs produce much larger late growth in output logit norm (up to about 1e7) and output logit mean (up to about 200), and several of them time-bomb.

**Training horizon (Fig. 17).** 28M models at peak LR 0.1 trained for 4B, 8B, 12B and 16B tokens: short runs finish before anything goes wrong; longer runs show progressively larger logit norms (up to about 1e8), mean-logit drift (up to about 1000), and late eval-loss deterioration. A time bomb can stay hidden in a short pilot run and only detonate when training is extended.

**Other knobs.** Smaller batches bring the onset earlier; weight decay delays it.

**Interpretation.** Each Adam step adds a small non-zero contribution to the mean output embedding (next section). A larger LR makes each contribution bigger, a longer horizon gives more steps to accumulate. That matches both trends.

## Mechanism, part 1: Adam drifts the mean logit

The cross entropy gradient of the LM head sums to zero over the vocabulary, but Adam's per-coordinate rescaling breaks that, so the mean output embedding and mean logit drift persistently (Bis et al., 2021; Stollenwerk and Stollenwerk, 2025; thread post 5).

**The zero-sum property.** With z = Eh:

```latex
\frac{\partial \mathcal{L}_{CE}}{\partial E} = (p - y)\,h^{\top}, \qquad \sum_{i=1}^{V} \frac{\partial \mathcal{L}_{CE}}{\partial E_{i,j}} = h_j \underbrace{\sum_{i=1}^{V} (p_i - y_i)}_{=0} = 0
```

The inner sum is zero because p and y each sum to one. So for every hidden dimension j, the gradient column sums to zero across tokens.

**Consequence.** Any update that preserves this zero sum leaves the mean output embedding unchanged:

```latex
\mu = \frac{1}{V}\sum_{i=1}^{V} e_i
```

**How Adam breaks it.** Adam's first moment (an average of zero-sum gradients) is still zero-sum. Dividing each coordinate by its own sqrt(second moment) reweights vocabulary rows unequally. Second moments track token frequency strongly: rare tokens get small v and therefore amplified updates. The weighted update no longer sums to zero, so mu moves every step. Since the mean logit is mu^T h, it drifts too, toward extreme values.

**How SGD-M preserves it.** SGD-M's update is a linear combination of past gradients with one shared scalar per step. A sum of zero-sum vectors is zero-sum, so mu stays fixed (up to its initial value and weight decay).

On its own, this explains why logits grow. It does not explain why growth breaks training; see part 2.

## Mechanism, part 2: why the drift breaks BF16

The drift is invisible in exact arithmetic but, in BF16, it inflates logit magnitudes until rounding wipes out the small logit differences that define the softmax (thread posts 6 and 7).

**The puzzle: centering changes nothing in exact arithmetic (Sec. 3.4).** Let E' = E - 1 mu^T (subtract the mean embedding from every row). For a fixed h, every logit shifts by the same scalar mu^T h, so p is unchanged, and so is the LM head gradient. The gradient sent back into the hidden state is also unchanged:

```latex
\frac{\partial \mathcal{L}_{CE}}{\partial h} = E'^{\top}(p - y) = E^{\top}(p - y) - \mu \sum_{i=1}^{V}(p_i - y_i) = E^{\top}(p - y)
```

So without weight decay and with exact computation, mean drift should affect neither predictions nor the optimization path, and mu-centering should not stabilize anything. In practice, drift does destabilize training and mu-centering does help.

**The resolution: fixed relative precision (App. C.1).** BF16 spacing is ULP(x) = 2^(e-7) for x in \[2^e, 2^(e+1)): about 0.008 between 1 and 2, but 8 between 1024 and 2048. The mean logit inflates every logit's magnitude without changing the differences, so the differences fall below the rounding grid.

Worked example from the paper:

| Logits | Exact probabilities | After BF16 rounding |
| --- | --- | --- |
| (0, 1) | (0.27, 0.73) | (0.27, 0.73) |
| (-1025, -1024), same shift of -1025 | (0.27, 0.73) | both round to -1024, giving (0.5, 0.5) |

Numerically stable softmax (subtracting the max logit) cannot help: the information was lost in the matmul that produced the logits, before softmax runs.

**The quantity that matters: relative logit mean.**

```latex
r = \frac{|\mu_z|}{\sigma_z}
```

where mu\_z and sigma\_z are the mean and standard deviation of the logits. When r is much less than 1, logit magnitudes are of order sigma\_z and rounding is small relative to their spread. When r is much greater than 1, magnitudes are set by |mu\_z|, and rounding error becomes large relative to the differences that matter. Fig. 18 confirms it: normalized error RMS(z\_bf16 - z\_fp32) / std(z\_fp32) rises steadily with r (plotted over r from about 1e-4 to 1e2) across BF16 true and BF16 mixed runs at every learning rate.

**Full causal chain**

1. Adam's preconditioner gives the LM head update a non-zero vocabulary mean.
2. The mean output embedding drifts each step, faster at high LR, more over long horizons.
3. The mean logit grows; r = |mu\_z| / sigma\_z rises.
4. BF16 spacing at that magnitude exceeds the logit differences; logit and softmax quantization error rise by orders of magnitude.
5. Predictions and gradients become corrupted; eval loss rises and does not recover.

## Precision experiments

Precision sets the timing of the time bomb: with only the LM head precision changed, BF16 true fails first, BF16 mixed later, and FP32 never fails within 16B tokens, even though FP32 logits also grow (Fig. 1, 6; thread posts 8 and 9).

**Setup.** 28M models, 16B tokens (about 570 tokens per parameter, 28x Chinchilla), peak LR 0.1, 5 seeds per setting, learnable RMSNorm, AdamW. Only the LM head precision varies; the rest of the network stays in BF16 mixed.

**Results at peak LR 0.1 (Fig. 1, 6)**

| LM head precision | Eval loss | Logit quantization error RMS(z\_bf16 - z\_fp32) | KL(p\_fp32 ‖ p\_bf16) | Outcome |
| --- | --- | --- | --- | --- |
| BF16 true | Diverges first | Grows by orders of magnitude, earliest | Grows by orders of magnitude | Time bomb, earliest |
| BF16 mixed | Diverges later | Grows by orders of magnitude, later | Grows by orders of magnitude | Time bomb, later |
| FP32 | Stable throughout | Not applicable (reference) | Not applicable | Stable, despite logit norm growth |

The ordering in loss matches the ordering in both error metrics, and in each reduced-precision run the error surges before the loss fails.

**Results across learning rates (App. C.2, Fig. 19).** Fraction of seeds that time-bomb, LRs 1e-3 to 1e-1:

- FP32: 0/5 at peak LR 1e-1, the largest tested, where every BF16 run fails.
- BF16 mixed: 0/5 at 1e-3, 3e-3 and 1e-2; 2/5 at 3e-2; 5/5 at 1e-1.
- BF16 true: 4/5 or 5/5 in every reported setting, starting at smaller LRs than BF16 mixed.

Lower precision lowers the learning rate at which time bombs appear. (The figure is a grid; the cell positions for BF16 true are hard to read from the extracted text, so check Fig. 19 for exact LR alignment.)

**Why FP32 survives.** FP32 has 24 bits of significand, about 65,000 times finer spacing than BF16 at the same magnitude, so the same mean-logit growth does not erase the logit differences over the horizons studied. The paper does not claim FP32 is immune at arbitrarily long horizons.

**Takeaway (paper's boxed summary).** Mean output logit drift is harmless in exact arithmetic but amplifies numerical error in finite precision. Lower LM head precision accelerates time bombs; higher precision improves stability.

## Mitigations compared

LM head SGD-M with momentum 0.99 is the only intervention that stayed stable at every learning rate and scale tested, while matching or beating the alternatives on final loss in most settings (thread posts 10 and 11).

**The methods**

| Method | Acts on | What it does | Source |
| --- | --- | --- | --- |
| z-loss | Loss | Adds lambda x log^2 Z, with Z = sum\_j exp(z\_j), pushing the log-partition toward zero; lambda = 1e-4 | Chowdhery et al., 2023 (PaLM) |
| mu-centering | Output embeddings | After each step, subtracts the mean output embedding mu from every row, so the rows sum to zero and the mean logit stays 0 regardless of hidden-state size | Stollenwerk et al., 2026 |
| Row-centering | Per-row update | Subtracts each token row's mean across the hidden dimension from its update; local to each row, so it does not restore the zero sum across the vocabulary | Evaluated in this paper (App. A.6) |
| LM head SGD-M (proposed) | Optimizer | SGD with momentum on the LM head only; AdamW for everything else; separately tuned head LR; momentum 0.99; no weight decay on the head in fast-spike runs | This paper |
| Logit soft-capping | Logits | Scaled tanh bounds logits before softmax | Gemma 2 (discussed, not tested) |
| Coupled Adam | Optimizer | Modifies Adam's second moment so the LM head drift does not arise | Stollenwerk and Stollenwerk, 2025 (discussed) |
| Fixed softmax | Cross entropy implementation | Preserves the zero-sum gradient at large logit margins | Barron et al., 2026 (tested in App. C.4) |

**Why momentum 0.99.** Removing Adam's rescaling cuts the effective step size for most vocabulary rows, especially rare tokens. Raising momentum from 0.9 to 0.99 raises steady-state amplification 1/(1 - beta) from 10x to 100x. beta = 0.99 beats 0.9 at every LR and both model sizes in Table 2. This may explain why Stollenwerk and Stollenwerk (2025), who tested LM head SGD-M at 0.9, found it slightly worse than coupled Adam. For the head LR, the highest value in the grid was best at every trunk LR; going higher gave no further gain.

### Time-bomb regime (404M, 96B tokens)

**Untuned comparison (Fig. 4).** z-loss, mu-centering and LM head SGD-M all improve stability over AdamW and sharply reduce logit growth. mu-centering and SGD-M suppress logit-norm growth most strongly at high LRs. The AdamW baseline shows the largest late growth and the time bombs. SGD-M stays competitive in final eval loss across the sweep.

**Fair tuning budget for AdamW (Fig. 5, App. B.3).** The AdamW-based methods received the same head-specific tuning budget as SGD-M:

- Tuning only the AdamW head LR: modest gains.
- Raising the head's beta1 from 0.9 to 0.99: closes much of the loss gap to SGD-M (consistent with Everett and Qiu, 2026, on retuning optimizer memory along the overtraining axis).
- Combining beta1 = 0.99 heads with mu-centering or z-loss: better at some LRs.
- But every AdamW variant diverges or time-bombs at the largest trunk LR (3e-1). SGD-M stays stable.

### Fast-spike regime (App. A.6)

**Learning-rate sweep, final eval loss (Table 2).** Parameter-less RMSNorm, LR 0.1 to 0.3. † = loss spiked and did not recover.

| Model | Method | LR 0.10 | 0.12 | 0.145 | 0.175 | 0.21 | 0.25 | 0.30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 162M | AdamW | 3.642 | 3.679 | 3.733 | 3.733 | 12.552† | 23.223† | 10.845† |
| 162M | mu-centering | 3.637 | 3.656 | 3.674 | 3.701 | 4.094† | 6.063† | 5.950† |
| 162M | z-loss | 3.636 | 3.655 | 3.690 | 3.718 | 5.594† | 5.964† | 7.495† |
| 162M | Row-centering | 3.617 | 3.631 | 3.633 | 3.718 | 3.697 | 3.793 | 3.863 |
| 162M | SGD-M 0.9 | 3.926 | 3.891 | 3.869 | 3.849 | 3.821 | 3.803 | 3.789 |
| 162M | SGD-M 0.99 | 3.683 | 3.674 | 3.659 | 3.649 | 3.642 | 3.666 | 3.736 |
| 404M | AdamW | 3.295 | 3.323 | 3.248 | 11.125† | 11.006† | 16.731† | 28.241† |
| 404M | mu-centering | 3.172 | 3.197 | 3.221 | 3.231 | 5.144† | 6.088† | 7.115† |
| 404M | z-loss | 3.161 | 3.183 | 3.213 | 3.249 | 3.289 | 7.266† | 6.588† |
| 404M | Row-centering | 3.257 | 3.166 | 3.550 | 4.930† | 12.106† | 3.310 | 3.452 |
| 404M | SGD-M 0.9 | 3.335 | 3.319 | 3.304 | 3.280 | 3.268 | 3.255 | 3.244 |
| 404M | SGD-M 0.99 | 3.186 | 3.183 | 3.180 | 3.173 | 3.169 | 3.167 | 3.167 |

mu-centering and z-loss break at the high end at both sizes; row-centering is stable at 162M but erratic at 404M; SGD-M 0.99 never diverges and is best or near-best from about LR 0.175 up.

**Across 5 seeds at LR 0.15 (Table 3).** Mean ± standard error; spike score = % of steps above 7 sigma of a trailing 1000-step window.

| Method | 162M eval loss | 162M spike score | 404M eval loss | 404M spike score |
| --- | --- | --- | --- | --- |
| AdamW baseline | 3.702 ± 0.014 | 0.088 ± 0.029 | 3.916 ± 0.278 | 0.085 ± 0.029 |
| z-loss | 3.644 ± 0.006 | 0.064 ± 0.027 | 3.214 ± 0.008 | 0.106 ± 0.040 |
| mu-centering | 3.646 ± 0.009 | 0.066 ± 0.043 | 3.208 ± 0.004 | 0.064 ± 0.002 |
| Row-centering | **3.621 ± 0.016** | 0.038 ± 0.007 | 4.858 ± 1.438 | 0.056 ± 0.011 |
| SGD-M 0.9 | 3.855 ± 0.003 | 0.038 ± 0.004 | 3.301 ± 0.003 | 0.060 ± 0.002 |
| SGD-M 0.99 | 3.644 ± 0.005 | 0.038 ± 0.004 | **3.178 ± 0.002** | 0.069 ± 0.003 |

**1.4B scale (Fig. 15).** Peak LR 0.07, single seed. LM head SGD-M (0.99) reaches lower training loss than AdamW, mu-centering and z-loss. A Muon backbone (Jordan et al., 2024) plus LM head SGD-M reaches the lowest loss of all configurations shown.

### Fixed softmax of Barron et al. (App. C.4, Fig. 21, 22)

Concurrent work found that standard softmax cross entropy can round the target logit's gradient to zero at high confidence ("softmax collapse"), breaking the zero-sum gradient. The authors swapped in Barron et al.'s corrected cross entropy (28M, LR 0.1, 5 seeds):

| Precision | Fixed softmax outcome | Logit norm and quantization error |
| --- | --- | --- |
| BF16 mixed | No time bomb within 16B tokens | Keep growing, far above z-loss runs |
| BF16 true | Failure delayed; all 5 seeds still time-bomb | Track the baseline until late, then diverge |

The fix repairs errors in the loss and its gradients, but not the Adam-induced mean drift or the logit differences lost in the BF16 forward matmul. It helps, but is not sufficient under BF16 true.

**Boxed summary from the paper.** Replacing AdamW with SGD-M on the LM head suppresses logit growth and improves long-horizon stability. Raising AdamW's beta1 recovers much of the loss gap but does not consistently prevent time bombs.

## Unifying principle: every working fix bounds quantization error

z-loss, mu-centering and LM head SGD-M intervene on the loss, the embeddings and the optimizer respectively, yet all three keep LM head quantization error bounded, while the unmitigated baseline's error grows by orders of magnitude (Fig. 7, 20; thread post 12).

**Evidence (28M, peak LR 0.1, 5 seeds each)**

| Method | BF16 mixed (Fig. 7) | BF16 true (Fig. 20) |
| --- | --- | --- |
| AdamW baseline | Time bomb; logit error and KL rise by orders of magnitude | Catastrophic failure after error grows several orders of magnitude |
| z-loss | No time bomb; both error metrics bounded | No time bomb; bounded |
| mu-centering | No time bomb; bounded | No time bomb; bounded |
| LM head SGD-M | No time bomb; bounded | No time bomb; bounded, but noticeably higher loss from early in training (slower optimization, not instability) |

The BF16 true SGD-M gap is the one weak spot for the proposed method. A reply on the thread (@letsunifyai) suggests a cause: without an FP32 master copy, any SGD-M step smaller than half the BF16 gap to the next value rounds away, and SGD-M's steps on rare-token rows are exactly the small ones Adam would have scaled up. Stochastic rounding on the head is their suggested fix. This is a hypothesis from the discussion, not tested in the paper.

**Interpretation.** These are not competing theories of instability. They are three ways of keeping the LM head in a regime where BF16 is precise enough. Conversely, an FP32 LM head does not need any of them over the horizons studied.

**Practical cost comparison (Sec. 3.4)**

| Fix | Throughput cost | Memory cost | Notes |
| --- | --- | --- | --- |
| FP32 LM head | High: FP32 GEMMs run at a fraction of BF16 speed, and the LM head is among the largest matmuls (output dimension = vocabulary size) | Doubles the logits tensor, often the largest activation; grows with batch, sequence length and vocabulary | Most direct, most expensive |
| LM head SGD-M | Negligible | Saves memory: one fewer optimizer moment buffer for the output embeddings | Needs its own head LR and momentum 0.99 |
| z-loss | Small: one auxiliary loss term | Negligible | Standard in many recipes (PaLM, SmolLM playbook, Marin) |
| mu-centering | Small: one reduction across vocabulary rows per step | Negligible | Structural; exact zero-mean guarantee |

**Thread summary of the principle:** keep the LM head in a regime where BF16 remains stable.

## Companion paper: A Fix for Softmax and Cross-Entropy (Barron et al., 2026)

Barron et al. show that standard softmax, log-softmax and cross-entropy implementations contain two rounding bugs, give drop-in log-space fixes, and use them to eliminate slingshot spikes; the time-bomb paper tested this fix and found it delays but does not prevent time bombs, because time-bomb error arises earlier, in the LM head matmul.

**Paper.** Jonathan T. Barron, Dan Gnanapragasam, Michael Tschannen (Google DeepMind). *A Fix for Softmax and Cross-Entropy*. [arXiv:2610.31642](https://arxiv.org/abs/2610.31642) v3 \[cs.LG\], 29 Oct 2026. License CC BY 4.0. Code below is reproduced from the paper under that license.

**Abstract in brief.** Bug 1, Softmax Collapse, rounds the target logit's gradient to zero when that logit dominates, so gradients no longer sum to zero. Bug 2, Softmax Cancellation, leaves non-zero gradients in distillation even when student and teacher logits are equal. The fixes remove slingshot spikes, speed up grokking with weight decay, prevent accuracy collapse without weight decay, stop distilled students drifting from their teacher, and reveal that post-memorization logit growth settles where weight decay balances Adam's eps.

### Why standard implementations round badly

All major frameworks (JAX, PyTorch, TensorFlow) subtract the max logit before exponentiating, to avoid overflow (exp overflows above about 88 in FP32):

```latex
\tilde z_k = z_k - z_{\max}, \qquad \mathrm{logsoftmax}(z)_k = \tilde z_k - \log\sum_j e^{\tilde z_j}, \qquad q_k = \frac{e^{\tilde z_k}}{\sum_j e^{\tilde z_j}}
```

Cross-entropy against a target distribution p has exact gradient g = q - p. Max-subtraction prevents overflow but leaves two precision bugs. The key fact: floating point is far finer near 0 than near 1. In FP32, numbers near 0 can be as small as about 1e-38, but anything below about 1e-7 added to 1.0 is lost.

### Bug 1: Softmax Collapse (identified by Prieto et al., 2025)

**Logit margin**, how far the target logit exceeds the log-sum-exp of all others:

```latex
m = z_t - \log\sum_{j \neq t} e^{z_j}
```

After max-subtraction the target term becomes e^0 = 1 and the others sum to e^(-m), so the denominator is 1 + e^(-m). Exact loss and gradients:

```latex
\mathcal{L} = \log(1 + e^{-m}), \qquad g_t = \underbrace{\frac{1}{1+e^{-m}} - 1}_{\text{standard}} = \underbrace{\frac{-e^{-m}}{1+e^{-m}}}_{\text{fixed}}, \qquad g_j = \frac{e^{z_j - z_t}}{1+e^{-m}} \;(j \neq t)
```

**Precision cliff c\_d.** Once m exceeds c\_d, 1 + e^(-m) rounds to exactly 1.0. Then the loss and log-probability round to 0, q\_t rounds to 1, and the standard target gradient q\_t - 1 becomes 0. The non-target gradients are computed without adding 1, so they stay positive until exponent underflow (m about 88). Result: the gradient sum becomes e^(-m)/(1+e^(-m)) > 0 (Liu et al., 2026).

| Format | Precision cliff c\_d (margin where the target gradient vanishes) |
| --- | --- |
| BF16 | about 5.5 |
| FP32 | about 16.6 |
| FP64 | about 36.7 |

Standalone softmax has the same bug: its backward pass zeroes the gradient of the largest entry (Qiu and Yao, 2026).

**The fix.** Never add 1 to e^(-m). Set the max (or target) slot to an exact constant 0, compute the margin from the other entries, and combine with a stable softplus(x) = log(1 + e^x):

```latex
\tilde z_k = \mathbf{1}[k \neq i_{\max}](z_k - z_{i_{\max}}), \quad m = -\mathrm{LSE}_{j \neq i_{\max}} \tilde z_j, \quad \mathrm{logsoftmax}(z)_k = \tilde z_k - \mathrm{softplus}(-m)
```

For hard-target cross-entropy, use the target t directly, with no argmax: L = softplus(-m), m = -LSE over j ≠ t of (z\_j - z\_t). Because z\_t then enters only through the K - 1 differences, autodiff produces g\_t as the exact negative sum of the non-target gradients, so the gradients sum to zero and g\_t stays near -e^(-m) down to underflow.

```python
import jax, jax.numpy as jnp, optax

def log_softmax(z):
  i_max = jnp.argmax(z, axis=-1)
  z_max = jnp.take_along_axis(z, i_max[..., None], axis=-1)  # Max logit.
  is_max = jax.nn.one_hot(i_max, z.shape[-1], dtype=bool)    # Max mask.
  shifted = jnp.where(is_max, 0.0, z - z_max)                # Exact 0 at argmax.
  exp_others = jnp.where(is_max, 0.0, jnp.exp(shifted))      # Non-max exp.
  s = jnp.sum(exp_others, axis=-1, keepdims=True)            # Sum non-max exp.
  return shifted - jnp.log1p(s)                              # log_softmax(z).

softmax = lambda z: jnp.exp(log_softmax(z))

def cross_entropy(z, t):
  z_t = jnp.take_along_axis(z, t[..., None], axis=-1)        # Target logit.
  is_t = jax.nn.one_hot(t, z.shape[-1], dtype=bool)          # Target mask.
  shifted = jnp.where(is_t, 0.0, z - z_t)                    # Exact 0 at target.
  b = jnp.max(shifted, axis=-1, keepdims=True)               # Max shift >= 0.
  s = jnp.sum(jnp.where(is_t, 0.0, jnp.exp(shifted - b)), axis=-1)
  return b[..., 0] + jnp.log1p(s + jnp.expm1(-b[..., 0]))    # -log_softmax(z)_t.
```

**Worked example**, z = \[9, -9, -9\] in FP32, target t = 0 (m = 18, beyond the FP32 cliff):

| Quantity | Standard (jax.nn / optax) | Fixed |
| --- | --- | --- |
| log\_softmax(z) | \[0.0, -18.0, -18.0\] | \[-3.046e-8, -18.0, -18.0\] |
| Gradient of softmax(z)\[0\] | \[0.0, -1.523e-8, -1.523e-8\] | \[3.046e-8, -1.523e-8, -1.523e-8\] |
| Cross-entropy loss | 0.0 | 3.046e-8 |
| Cross-entropy gradient | \[0.0, 1.523e-8, 1.523e-8\] (sum > 0) | \[-3.046e-8, 1.523e-8, 1.523e-8\] (sum = 0) |

**Errata (v3).** An earlier version of this code produced NaN gradients when all but one logit is effectively -infinity (as with causal-attention masking or extreme logits). It wrote softplus(-m) as softplus(log s), which makes autodiff differentiate through log 0 when s = 0; log1p(s) avoids this. The authors attribute the slip to an AI agent that simplified their internal code (which was correct and produced all results). Found by Maaz (2026), [arXiv:2610.36427](https://arxiv.org/abs/2610.36427).

### Bug 2: Softmax Cancellation (distillation and KL regularization)

With teacher logits z\_T and student logits z\_S, standard soft-target cross-entropy computes p = softmax(z\_T) and q = softmax(z\_S) separately, then g = q - p. As z\_S approaches z\_T, subtracting two nearly equal probabilities loses the small logit difference. Worse, the forward pass uses softmax while the backward pass uses exp(log\_softmax), so p and q differ slightly even when z\_S = z\_T, leaving a non-zero gradient that never reaches zero.

**The fix.** Pass z\_T into the loss and compute q - p in log space from delta = z\_S - z\_T, using expm1(x) = e^x - 1 and log1p(x) = log(1 + x):

```latex
d_k = \delta_k - \sum_i p_i \delta_i, \qquad s_k = d_k - \mathrm{log1p}\Big(\sum_i p_i\, \mathrm{expm1}(d_i)\Big), \qquad u_k = p_k\, \mathrm{expm1}(s_k)
```

Here s\_k = log(q\_k / p\_k) and u\_k = q\_k - p\_k. The log1p input is always non-negative. When |s\_k| ≥ 1 it falls back to q\_k - p\_k to avoid 0 x infinity, then sets g\_k = u\_k - q\_k x sum(u) so the gradient sums to zero. The gradient is exact as z\_S approaches z\_T and exactly 0 at equality.

```python
def soft_target_cross_entropy(z_S, z_T):
  log_q = log_softmax(z_S)
  q, p = jnp.exp(log_q), softmax(z_T)
  delta = z_S - z_T                                                    # Logit difference.
  d = delta - jnp.sum(p * delta, axis=-1, keepdims=True)               # Center under p.
  s = d - jnp.log1p(jnp.sum(p * jnp.expm1(d), axis=-1, keepdims=True)) # log(q / p).
  u = jnp.where(jnp.abs(s) < 1, p * jnp.expm1(s), q - p)               # Difference q - p.
  g = u - q * jnp.sum(u, axis=-1, keepdims=True)                       # Ensure sum(g) = 0.
  sg = jax.lax.stop_gradient
  loss = -jnp.sum(p * sg(log_q), axis=-1)                              # Forward loss.
  return loss + jnp.sum(sg(g) * (z_S - sg(z_S)), axis=-1)              # Custom gradient g.
```

**Worked example**, z\_T = \[2, 1, 0\], z\_S = z\_T + \[-1e-7, 1e-7, 0\] in FP32:

| Implementation | Gradient |
| --- | --- |
| Standard | \[-5.960e-8, 1.452e-8, -1.123e-9\] (2nd entry nearly 3x too small, 3rd has wrong sign, sum ≠ 0) |
| Fixed | \[-4.596e-8, 4.144e-8, 4.513e-9\] |
| Exact (FP64) | \[-4.596e-8, 4.144e-8, 4.513e-9\] |

### How Softmax Collapse produces slingshot spikes

In grokking (Power et al., 2022), a network memorizes the training set quickly and generalizes much later. Thilak et al. (2022) observed slingshot spikes after memorization: flat near-zero loss plateaus broken by sudden spikes that coincide with jumps in test accuracy. The mechanism under AdamW:

1. After memorization, margins m grow as target probabilities approach 1.
2. Once an example's margin exceeds c\_d, its target gradient rounds to 0 while non-target gradients stay positive.
3. When every example in a batch is above c\_d, parameter gradients nearly vanish and Adam's second moment v decays exponentially toward zero; weight decay slowly shrinks the weights.
4. Weight decay or the residual non-target updates pull a few examples back below c\_d, and their target gradients (about -e^(-m)) switch back on.
5. Adam divides these restored gradients by the tiny sqrt(v) + eps, producing a huge step: the loss spikes, v resets, and the cycle repeats.

With the fixed loss, g\_t never rounds to zero, so Adam's moments decay smoothly together and the spikes disappear.

**Whether spikes seem to help depends on weight-norm control.**

| Setting | Effect of slingshot spikes | Example result |
| --- | --- | --- |
| LayerNorm before output, no weight decay (Thilak et al.; Prieto et al.) | Each spike inflates weight norm, shrinking the effective step size: an accidental learning-rate decay that looks helpful | n/a |
| Weight decay controls norms (Varma et al., 2023) | Spikes at c\_32 slow and hurt grokking | a+b mod 31: standard 94 ± 5% vs fixed 100 ± 0% test accuracy |
| Neither LayerNorm nor weight decay (Doshi et al., 2024) | One spike is destructive | a²+b² mod 97 (lambda = 0, eps = 1e-16): standard falls from 61% to 23% test accuracy near step 1250; fixed continues to 85%. a³+b mod 97: 60% to 21% |

### Margin equilibria: why margins stop near log(1/eps)

With weight decay, fixed cross-entropy margins still level off near 18 in FP32, while without weight decay they keep climbing. At a stationary point of AdamW, the moments converge to m̂ = g and sqrt(v̂) = |g|, and the update vanishes:

```latex
\frac{g_i}{|g_i| + \epsilon} + \lambda\,\theta_i = 0 \quad \forall i
```

Learning rate, beta1 and beta2 cancel, so RMSProp and LaProp share this stationary point. After memorization |g| shrinks like e^(-m). With eps = 0, the normalized step g/|g| stays at magnitude 1 however small g gets, so margins grow until gradients underflow. With eps > 0, the step shrinks like g/eps once |g| falls below eps. Balancing it against weight decay needs |g| about eps x lambda x |theta|, which acts like an L2 regularizer of weight eps x lambda, so the margin stops at:

```latex
m^{\star}(\epsilon) \approx \log(1/\epsilon)
```

**Why default eps causes slingshots in FP32.** With eps = 1e-8, the equilibrium log(1/eps) ≈ 18.4 sits just above the FP32 cliff of 16.6, so standard cross-entropy hits the cliff before reaching equilibrium. With eps ≥ 1e-7, log(1/eps) ≤ 16.1 < 16.6 and standard cross-entropy settles without spiking.

**Validation.** Fixed cross-entropy with LaProp on a+b mod 31 in FP32 (beta1 = 0.9, beta2 = 0.98, 3 seeds per setting), updating v directly without squaring g (squaring underflows below |g| = 2^-63, near m = 38). With lambda = 1, each eps > 0 run follows the eps = 0 trajectory then levels off at log(1/eps); eps = 0 grows until FP32 gradients underflow near m = 81.2. With lambda = 0, nothing balances the step and margins overshoot log(1/eps).

### Grokking results across precisions

Setup (following Varma et al., 2023 and Lyle et al., 2025): 1-layer, 4-head, width-64 ReLU Transformer, no LayerNorm, a+b mod 31, 50% training split, 10,000 epochs of full-batch AdamW (lr 1e-3, beta1 = 0.9, beta2 = 0.98, lambda = 1). Varied: loss, precision of logits at the loss, and eps.

- eps = 1e-8: standard cross-entropy stalls at the BF16 cliff (5.5) and repeatedly spikes at the FP32 cliff (16.6). Fixed crosses both smoothly to about 18.4 and reaches 100% in all three precisions.
- eps = 1e-16: the equilibrium rises to about 36.8, just past the FP64 cliff (36.7), so even FP64 standard cross-entropy breaks; fixed converges in every precision.

**Table 1 of Barron et al.** a+b mod 31, lambda = 1, 3 to 5 seeds. Predicted equilibrium margins: 18.4, 27.6, 36.8.

| Precision | Variant | Test acc. eps 1e-8 (%) | eps 1e-12 (%) | eps 1e-16 (%) | Final margin eps 1e-8 | eps 1e-12 | eps 1e-16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP32 | Fixed CE | 100.0 ± 0.0 | 99.6 ± 0.7 | 99.0 ± 2.3 | 17.9 | 27.4 | 36.8 |
| FP32 | Standard CE | 94.3 ± 5.4 | 96.4 ± 5.1 | 96.5 ± 4.3 | 13.0 | 15.1 | 12.7 |
| BF16 | Fixed CE | 100.0 ± 0.1 | 99.8 ± 0.2 | 98.9 ± 2.4 | 17.9 | 27.4 | 36.8 |
| BF16 | Standard CE | 93.3 ± 7.4 | 94.9 ± 4.9 | 94.2 ± 8.3 | 5.1 | 5.6 | 5.8 |
| FP64 | Fixed CE | 100.0 ± 0.0 | 99.5 ± 1.1 | 99.0 ± 2.0 | 17.9 | 27.4 | 36.8 |
| FP64 | Standard CE | 100.0 ± 0.0 | 100.0 ± 0.1 | 97.2 ± 3.2 | 17.9 | 27.4 | 30.1 |
| FP32 | Zero-sum projection (Liu et al., 2026) | 97.9 ± 3.2 | 48.9 ± 2.6 | 49.3 ± 2.5 | 17.0 | 18.9 | 18.9 |
| FP32 | StableMax (Prieto et al., 2025) | 97.3 ± 4.7 | 95.6 ± 2.0 | 97.3 ± 4.3 | 3200 | 5661 | 6313 |
| FP64 | StableMax (Prieto et al., 2025) | 97.4 ± 4.2 | 96.7 ± 5.3 | 95.6 ± 2.7 | 3170 | 21252 | 2322 |

Why the alternatives fall short: Liu et al.'s zero-sum projection subtracts the mean gradient so gradients sum to zero, which removes spikes, but once g\_t is already 0 it gives the target only -1/K of the non-target total, stalling margins near 18 and dropping accuracy below 50% for eps ≤ 1e-12. Prieto et al.'s StableMax replaces exp with a piecewise rational function (x + 1 for x ≥ 0, 1/(1 - x) for x < 0); growing linearly, it lets margins blow up to about 1e4 and lags fixed cross-entropy in accuracy.

**Four benchmarks at eps = 1e-12:** 1-layer Transformers on a+b mod 113 (Nanda et al., 2023) and a+b mod 31 (Varma et al., 2023) with lambda = 1; 2-layer MLPs on a²+b² mod 97 and a³+b mod 97 (Doshi et al., 2024) with lambda = 0.2. FP32 standard cross-entropy spikes at the FP32 cliff on all four; FP32 fixed matches FP64 standard and reaches 100% test accuracy.

### Distillation results

A 2-layer ReLU MLP student is initialized with the FP32 weights of its teacher (100% accuracy on a+b mod 53) and trained 2,000 full-batch Adam steps (lr 2e-3, beta1 = 0.9, beta2 = 0.98, lambda = 0, eps = 1e-8, 50% split, temperature T = 4 so m/T stays below the BF16 cliff). The true gradient is zero, so the student should not move. Adam amplifies the standard loss's residual gradient:

| Precision of forward pass and loss | Residual gradient (standard) | Standard outcome | Fixed outcome |
| --- | --- | --- | --- |
| BF16 | about 1e-3 | Student drifts away from teacher; test accuracy degrades | KL = 0, 100% accuracy |
| FP32 | about 1e-8 | Initial KL spike, then a KL floor near 1e-6 | KL = 0, 100% accuracy |
| FP64 | about 1e-17 | Update too small to change FP32 weights | KL = 0, 100% accuracy |

### Authors' conclusion and scope

Rewriting these functions in log space with softplus, expm1 and log1p restores accurate, zero-sum gradients in BF16, FP32 and FP64 without casting to higher precision and without meaningful slowdown. The authors note that the extreme margins and near-zero student-teacher gaps they probe are uncommon in standard large-scale training, but position the functions as drop-in replacements for high-confidence prediction and close-teacher distillation.

How this paper relates to the time-bomb paper, and the combined evidence on fixes, is in [Consolidated findings across both papers](#consolidated-findings-across-both-papers).

### Key references from Barron et al.

| Reference | Role in Barron et al. |
| --- | --- |
| Prieto, Barsbey, Mediano and Birdal, 2025. Grokking at the edge of numerical stability. ICLR 2025 | Named Softmax Collapse; showed it stalls grokking; proposed StableMax |
| Liu, Cao, Li and Zhou, 2026. Grokking or glitching? [arXiv:2605.06152](https://arxiv.org/abs/2605.06152) | Linked collapse to slingshot spikes; proposed zero-sum gradient projection |
| Thilak et al., 2022. The Slingshot mechanism. [arXiv:2206.04817](https://arxiv.org/abs/2206.04817) | Original description of slingshot spikes |
| Qiu and Yao, 2026. Why low-precision Transformer training fails: An analysis on Flash Attention. ICLR 2026 | Same rounding bug in standalone softmax |
| Blanchard, Higham and Higham, 2021. Accurately computing the log-sum-exp and softmax functions. IMA J. Numer. Anal. | Excluding the max from the log-sum-exp; numerical basis of the fix |
| Higham, 2002. Accuracy and stability of numerical algorithms. SIAM | Max-subtraction standard practice |
| Maaz, 2026. Property-based testing for robust softmax and cross-entropy. [arXiv:2610.36427](https://arxiv.org/abs/2610.36427) | Found the NaN-gradient bug fixed in v3 |
| Hinton, Vinyals and Dean, 2015. Distilling the knowledge in a neural network. arXiv:1503.02531 | Soft-target distillation with temperature |
| Power et al., 2022. Grokking. arXiv:2201.02177 | Defines grokking |
| Nanda et al., 2023. Progress measures for grokking via mechanistic interpretability. ICLR 2023 | a+b mod 113 benchmark; used FP64 casting |
| Varma et al., 2023. Explaining grokking through circuit efficiency. arXiv:2309.02390 | a+b mod 31 setup with weight decay; used FP64 casting |
| Doshi, He, Das and Gromov, 2024. Grokking modular polynomials. arXiv:2406.03495 | a²+b² and a³+b mod 97 MLP benchmarks |
| Lyle, Sokar, György and Pascanu, 2025. What can grokking teach us about learning under non-stationarity? CoLLAs 2025 | Grokking training setup |
| Kingma and Ba, 2015 (Adam); Loshchilov and Hutter, 2019 (AdamW) | Optimizers whose normalization amplifies the bug |
| Tieleman and Hinton, 2012 (RMSProp); Ziyin, Wang and Ueda, 2020 (LaProp, arXiv:2002.04839) | Share the same stationary point; LaProp used to test the equilibrium |
| Wortsman et al., 2024; Molybog et al., 2023 (A theory on Adam instability, arXiv:2304.09871); Everett et al., 2024 (Scaling exponents across parameterizations and optimizers, ICML); Kosson, Messmer and Jaggi, 2024 (Rotational equilibrium, ICML) | Prior analyses of eps and weight-decay equilibria in Adam |
| Rosset, Zhu and Hastie, 2004 (Margin maximizing loss functions, NeurIPS); Soudry et al., 2018 (The implicit bias of gradient descent on separable data, JMLR) | Margin growth and implicit L2-like regularization |

## Discussion from the X thread

Replies raised five open questions the paper does not answer: inference-time effects, Pythia/FP16 history, early warning signals, FP32-only logits, and other quantization hotspots. Micah Goldblum answered three replies; the rest were unanswered at capture time.

**Thread map (posts 3 to 13 by @micahgoldblum)**

| Post | Content | Paper section |
| --- | --- | --- |
| 3 | Fast spikes vs time bombs; learnable RMSNorm gains suppress fast spikes | 2.4, 3.1 |
| 4 | Two LRs track each other, then the higher one diverges as LM head logits grow | 3.1, Fig. 3 |
| 5 | Zero-sum gradient broken by Adam, causing mean embedding and mean logit drift | 3.2 |
| 6 | Puzzle: a constant shift changes neither softmax nor gradients in exact arithmetic | 3.4 |
| 7 | BF16 spacing grows with magnitude, eroding logit differences | 3.4, C.1 |
| 8 | BF16 true fails first, BF16 mixed later, FP32 stable | 3.4, Fig. 6 |
| 9 | FP32 logits still grow without divergence; BF16 error grows by orders of magnitude | Fig. 1, 6 |
| 10 | Fix: SGD-M on the LM head only, AdamW elsewhere | 3.3 |
| 11 | SGD-M keeps logit growth and quantization error nearly flat, competitive final loss | Fig. 1, 4 |
| 12 | z-loss, mu-centering and SGD-M all bound quantization error | 3.4, Fig. 7 |
| 13 | Paper link and credits: @tingtang222 (lead), @aditsom7, Sophie Li, @jwkirchenbauer, @SeanMcleish, @tomgoldsteincs | n/a |

**Questions with author replies**

| Who | Question or point | Author reply |
| --- | --- | --- |
| Nathan Godey (@nthngdy) | Did you look at loss collapse in Pythia models, which used FP16 and have public checkpoints? | Not studied. |
| @bouonkofun | Post 7 should apply at inference too: does SGD-M or z-loss shrink BF16 vs FP32 gaps at inference? | Not studied; a fair test would compare models with the same FP32 train loss. |
| Crio Songo (@shuizhuyu) | Does the optimizer change hold across scales and data mixes? | Yes in their experiments; wants broader validation. |

**Related findings and pointers raised**

- **Neel Rajani:** MiniMax-M1 found its LM head needed FP32 to avoid a training/inference probability mismatch in RL ([arXiv:2506.13585](https://arxiv.org/abs/2506.13585)). Suggests the LM head's precision sensitivity is underexplored generally.
- **Liu Hanqing (@LhqPhy), author of Liu et al. 2026:** In their balanced classification setting, Adam's asymmetric updates are largely irrelevant; the key mechanism is softmax collapse, and switching to SGD does not remove the spikes. Paper: [Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes](https://arxiv.org/abs/2605.06152). The time-bomb paper treats this as a different phenomenon (see Related work).
- **Hudson Gouge:** Reports independently finding LM head quantization error as the cause of the same failure.
- **Jialong Sun:** Suggests Hyperball (norm-preserving optimization, used in the Marin recipe) might remove LM head quantization error, at the cost of larger gradients.

**Open questions without replies**

- **@letsunifyai:** Is SGD-M's higher loss under BF16 true (Fig. 20) caused by small updates rounding away without an FP32 master copy? Would stochastic rounding on the head fix it?
- **W. Cheng and Vanraj Narula:** Does the fix still matter if only the final logit computation runs in FP32? Partly answered by the paper: an FP32 LM head (weights and GEMM) stays stable, at a high compute and memory cost.
- **@yvbbrjdr:** Why train with BF16 logits at all? Answer from the paper: FP32 LM head GEMMs are slow and FP32 logits double the largest activation tensor.
- **Omkar Ray:** Is there an early signal? The paper's figures suggest yes: logit norm, mean logit, r = |mu\_z| / sigma\_z, and RMS(z\_bf16 - z\_fp32) all rise well before the loss fails. Dheiver Santos makes the same monitoring recommendation.
- **@aapakari:** Optimizer-only fix, with no precision change? Yes: SGD-M works with the LM head left in BF16 mixed.
- **@R24864627:** What about attention scores in intermediate layers? Out of scope: all runs use QK-Norm to remove attention-logit growth. Chen et al. (2026) study a related BF16 attention-gradient failure.
- **Alano:** Long-run stability tests are essential to validate the fix.

## Limitations and future work

The main caveat is external validity: spikes were induced with unusually high learning rates in models of 1.4B parameters or fewer, so results may not fully carry over to spikes that arise naturally in frontier-scale runs.

**Limitations stated or visible in the paper**

- Spikes are elicited with higher-than-normal LRs, a common practice (Zhao et al., 2025; Stollenwerk et al., 2026; Su et al., 2025; Li et al., 2025) but not proof of transfer.
- Largest model is 1.4B, and that comparison is a single seed. The time-bomb and precision studies use 404M and 28M models.
- Whether LM head SGD-M beats standard optimizers in natural large-scale settings is open.
- FP32 stability is shown only over the horizons studied.
- Under BF16 true, SGD-M converges to a noticeably higher loss than z-loss or mu-centering.
- Evidence that the mitigations work through quantization-error control is correlational: all of them bound the error, and an FP32 head needs none of them.
- The exact-arithmetic invariance argument assumes no weight decay; time-bomb experiments use weight decay 0.

**Future work proposed by the authors**

- Specialized optimizers designed for the LM head.
- Evaluating these mitigations in lower-precision formats such as FP8.

**Further directions from the thread:** inference-time BF16 vs FP32 gaps, Pythia FP16 checkpoints, stochastic rounding for BF16 true heads, and broader validation across scales and data mixes.

## Related work and expanded references

The paper sits between three literatures: documented pretraining spikes, Adam's effect on the LM head, and finite-precision softmax failures; its contribution is linking Adam-induced drift to BF16 forward-pass error. Descriptions of 2026 works below come from how this paper characterizes them.

### How it differs from the closest concurrent work

| Work | Failure studied | Where the numerical error is | Relation to time bombs |
| --- | --- | --- | --- |
| Chen et al., 2026 | Late-training blow-ups in BF16 FlashAttention | Backward pass of fused attention; BF16 rounding breaks the zero-sum of attention-softmax gradients | Analogous mechanism in attention, not the LM head |
| Liu et al., 2026 | Slingshot spikes (abrupt, recoverable), mostly in grokking | Softmax cross entropy gradients (softmax collapse) | Also see abnormal logit growth in GPT training but no Slingshot spikes there |
| Barron et al., 2026 | Slingshot spikes on synthetic modular arithmetic | Softmax cross entropy gradient at high confidence | Their fix delays but does not prevent time bombs under BF16 true |
| This paper | Slow, unrecoverable LM head time bombs | Forward LM head matmul, driven by Adam mean drift | n/a |

### Loss spikes and large-scale training reports

| Reference | What it is | Role here |
| --- | --- | --- |
| Chowdhery et al., 2023. PaLM: Scaling language modeling with pathways. JMLR 24 | Google's 540B model report | First identified output-logit divergence; introduced z-loss; found replaying batches from other checkpoints does not reproduce spikes; used checkpoint rewinds and batch skipping |
| Zhang et al., 2022. OPT: Open pre-trained transformer language models. [arXiv:2205.01068](https://arxiv.org/abs/2205.01068) | Meta's 175B open model | Documented spikes and restarts from checkpoints |
| Zeng et al., 2023. GLM-130B. ICLR 2023 | Bilingual 130B model | Spike reports; embedding-layer stabilization |
| BigScience Workshop, 2022. BLOOM. arXiv:2211.05100 | 176B multilingual open model | Spike reports; embedding layer norm |
| Almazrouei et al., 2023. The Falcon series of open language models. [arXiv:2311.16867](https://arxiv.org/abs/2311.16867) | Falcon model report | Heuristic spike recovery |
| OLMo et al., 2025. 2 OLMo 2 Furious. [arXiv:2501.00656](https://arxiv.org/abs/2501.00656) | AI2 open model report | Source of the spike-score metric |
| Groeneveld et al., 2024. OLMo: Accelerating the science of language models. ACL 2024 | First OLMo report | Uses parameter-less norms |
| DeepSeek-AI, 2026. DeepSeek-V4 | Model report | Documents spikes |
| Kimi Team, 2026. Kimi K2: Open agentic intelligence. [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) | Model report | Documents spikes and architectural stabilizers |
| Wortsman et al., 2024. Small-scale proxies for large-scale transformer training instabilities. ICLR 2024 | Reproduces large-model instabilities in small models via high LR | Basis of the method: LR grid, slow divergences, QK-Norm recommendation, weight decay delaying failure |
| Karpathy, 2024. X post on skipping updates above 7 sigma ([link](https://x.com/karpathy/status/1812917107379872145)) | Outlier-skip heuristic in llm.c | Basis of the batch-skipping and token-masking experiments |
| Huang et al., 2025. SPAM: Spike-aware Adam with momentum reset. ICLR 2025 | Optimizer-state reset on spikes | Example of optimizer-level spike fixes |
| Pascanu et al., 2013. On the difficulty of training recurrent neural networks. [arXiv:1211.5063](https://arxiv.org/abs/1211.5063) | Origin of gradient clipping | Example of gradient-level fixes |
| Dettmers et al., 2021. 8-bit optimizers via block-wise quantization. arXiv:2110.02861 | Low-memory optimizers; stable embedding layer | Example of embedding stabilizers |

### LM head optimization and drift

| Reference | What it is | Role here |
| --- | --- | --- |
| Bis et al., 2021. Too much in common: Shifting of embeddings in transformer language models. NAACL 2021 | Shows embeddings share a common shift | Early evidence of the zero-sum gradient structure and mean drift |
| Stollenwerk and Stollenwerk, 2025. Better embeddings with coupled Adam. ACL 2025 | Links mean embedding drift to Adam's second moment; proposes coupled Adam | Core mechanism for logit growth; tested LM head SGD-M at momentum 0.9 and found it slightly worse |
| Stollenwerk, Lokrantz and Hertzberg, 2026. Output embedding centering for stable LLM pretraining. [arXiv:2601.02031](https://arxiv.org/abs/2601.02031) | Proposes mu-centering | Main baseline; source of the shift-invariance argument |
| Kunstner et al., 2024. Heavy-tailed class imbalance and why Adam outperforms gradient descent on language models. NeurIPS 37 | Shows Adam helps rare classes | Explains why removing Adam from the head needs high momentum |
| Zhao et al., 2025. Deconstructing what makes a good optimizer for autoregressive language models. ICLR 2025 | Compares optimizers; highlights adaptivity on the last layer | Supports LM head-specific optimizer choices; high-LR methodology |
| Everett and Qiu, 2026. Optimizer memory schedules for outscaling the overtraining axis. [arXiv:2609.04577](https://arxiv.org/abs/2609.04577) | Optimizer tuning along the overtraining axis | Explains why higher beta1 helps at long horizons; overtraining trend |
| Jordan et al., 2024. Muon: An optimizer for hidden layers ([post](https://kellerjordan.github.io/posts/muon/)) | Orthogonalized-momentum optimizer | Used as trunk optimizer in a 1.4B run with LM head SGD-M |
| Loshchilov and Hutter, 2019. Decoupled weight decay regularization. ICLR 2019 | AdamW | Baseline optimizer |

### Finite precision and quantization

| Reference | What it is | Role here |
| --- | --- | --- |
| Chen et al., 2026. Broken symmetry in BF16 attention: Why FlashAttention gradients blow up late in training. [arXiv:2609.34272](https://arxiv.org/abs/2609.34272) | BF16 rounding in fused attention backward pass | Concurrent, analogous mechanism in attention |
| Liu, Cao, Li and Zhou, 2026. Grokking or glitching? How low-precision drives Slingshot loss spikes. [arXiv:2605.06152](https://arxiv.org/abs/2605.06152) | Softmax collapse explains Slingshot spikes | Concurrent; different failure (also cited by its author in the thread) |
| Barron, Gnanapragasam and Tschannen, 2026. A fix for softmax and cross-entropy. [arXiv:2610.31642](https://arxiv.org/abs/2610.31642) | Numerically corrected softmax cross entropy | Tested as a mitigation in App. C.4; full summary in the companion-paper section |
| Thilak et al., 2022. The Slingshot mechanism. [arXiv:2206.04817](https://arxiv.org/abs/2206.04817) | Periodic spikes with adaptive optimizers in grokking | Defines the Slingshot phenomenon |
| Kalamkar et al., 2019. A study of bfloat16 for deep learning training. [arXiv:1905.12322](https://arxiv.org/abs/1905.12322) | BF16 format study | Source for BF16's 8-bit significand |
| Frantar et al., 2023. OPTQ (GPTQ). ICLR 2023 | Post-training weight quantization | Template for the reconstruction-error metric |
| Lotfi et al., 2026. Quantized reasoning models think they need to think longer, but they do not. [arXiv:2606.00206](https://arxiv.org/abs/2606.00206) | Quantization effects on reasoning models | Template for the KL distribution-error metric |
| Su et al., 2025. Characterization and mitigation of training instabilities in microscaling formats. [arXiv:2506.20752](https://arxiv.org/abs/2506.20752) | MX low-precision training instabilities | High-LR proxy methodology |

### Stabilizing architectures and logit control

| Reference | What it is | Role here |
| --- | --- | --- |
| Dehghani et al., 2023. Scaling vision transformers to 22 billion parameters. ICML 2023 | ViT-22B | Popularized QK-Norm, used in all runs |
| Henry et al., 2020. Query-key normalization for transformers. Findings of EMNLP 2020 | Original QK-Norm | Attention stabilizer |
| Gemma Team, 2024. Gemma 2. [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) | Google open model | Logit soft-capping via tanh |
| Allal et al., 2025. The Smol training playbook | Hugging Face training guide | z-loss as a standard safeguard |
| Marin Community, 2026a, b. GitHub issues [#8435](https://github.com/marin-community/marin/issues/8435) (535B-A23B hero run, 18T tokens) and [#8227](https://github.com/marin-community/marin/issues/8227) (feature ablations) | Open frontier-style recipe | Uses z-loss with hyperball optimization; ablations show large degradation without z-loss |
| Zhang and Sennrich, 2019. Root Mean Square Layer Normalization. NeurIPS 2019 | RMSNorm | Norm layer; gains on vs off controls fast spikes |
| Yang et al., 2025. Qwen3 technical report. arXiv:2505.09388 | Qwen3 | Example of learnable norm gains and heavy overtraining |
| Shoeybi et al., 2019. Megatron-LM. arXiv:1909.08053 | Model-parallel training | Example of learnable norm gains |
| Wang et al., 2026. HRM-Text. [arXiv:2605.20613](https://arxiv.org/abs/2605.20613) | Model suite | Uses parameter-less norms |
| Geiping et al., 2025. Scaling up test-time compute with latent reasoning: A recurrent depth approach. NeurIPS 2025 | Recurrent-depth LM | Reports instabilities with parameter-less norms |
| Filatov et al., 2026. Optimal scaling needs optimal norm. [arXiv:2510.03871](https://arxiv.org/abs/2510.03871) | Norm-based scaling study | Reports instabilities with parameter-less norms |
| Li et al., 2025. Model merging in pre-training of large language models. NeurIPS 2025 | Merging during pretraining | High-LR methodology precedent |

### Models, data and tools

| Reference | Role here |
| --- | --- |
| Radford et al., 2019. GPT-2 | Base architecture and tokenizer |
| Su et al., 2024. RoFormer (RoPE). Neurocomputing 568 | Rotary positional embeddings |
| Hendrycks and Gimpel, 2023. GELU. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | Activation function |
| Penedo et al., 2024. The FineWeb datasets. NeurIPS Datasets and Benchmarks | Training data |
| Bradbury et al., 2018. JAX ([github](http://github.com/jax-ml/jax)); Heek et al., 2024. Flax ([github](http://github.com/google/flax)) | Training framework |
| Bloom et al., 2025. Empire AI. PEARC '25 | Compute provider (with Google TPU Research Cloud) |

### Referenced only in the X thread

| Reference | Context |
| --- | --- |
| MiniMax, 2025. MiniMax-M1. [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) | Raised by Neel Rajani: LM head needed FP32 to avoid train/inference probability mismatch in RL |
| Pythia model suite (EleutherAI) | Raised by Nathan Godey: FP16-era training with public checkpoints and loss collapse, a cheap test bed |
| Hoffmann et al., 2022. Training compute-optimal large language models (Chinchilla) | Not cited in the paper by name; source of the about-20-tokens-per-parameter budget used in its ratios |

## Glossary

| Term | Meaning |
| --- | --- |
| BF16 mixed | FP32 master weights and optimizer states; BF16 casts for matmuls and activations |
| BF16 true | Weights and optimizer states also stored in BF16 (applied to the LM head here) |
| Chinchilla-optimal | About 20 training tokens per parameter |
| Fast spike | Abrupt loss jump that recovers |
| GEMM | General matrix multiply; the LM head's main computation |
| Grokking | Generalization that arrives long after a network has memorized its training set |
| KL(p\_fp32 ‖ p\_bf16) | Distribution-level quantization error between FP32 and BF16 softmax outputs |
| LM head | Final layer mapping hidden state to vocabulary logits, z = Eh |
| Logit margin (m) | Target logit minus the log-sum-exp of all other logits (Barron et al.) |
| Logit quantization error | RMS(z\_bf16 - z\_fp32) on the same hidden states |
| Margin equilibrium | Under AdamW with weight decay, margins settle near log(1/eps) |
| Mean output embedding (mu) | Average of the LM head rows; sets the mean logit mu^T h |
| mu-centering | Subtract mu from every LM head row after each step |
| Overtraining | Training beyond the compute-optimal token budget |
| Precision cliff (c\_d) | Margin above which 1 + e^(-m) rounds to 1: about 5.5 (BF16), 16.6 (FP32), 36.7 (FP64) |
| QK-Norm | Normalizing queries and keys to prevent attention-logit growth |
| r = abs(mu\_z) / sigma\_z | Relative logit mean; the predictor of BF16 logit error |
| Row-centering | Subtract each row's mean over the hidden dimension from its update |
| SGD-M | Stochastic gradient descent with momentum |
| Slingshot | Periodic, recoverable spikes in grokking-style training |
| Softmax Cancellation | Precision loss when subtracting nearly equal student and teacher probabilities in distillation |
| Softmax collapse | Rounding drives a confident target's gradient to zero, breaking the zero-sum gradient |
| Spike score | % of steps with loss above 7 sigma of a trailing 1000-step window |
| StableMax | Prieto et al.'s softmax variant replacing exp with a piecewise rational function |
| Time-bomb spike | Late, slow, unrecoverable divergence tied to LM head logit growth |
| ULP | Unit in the last place: gap between adjacent representable numbers; 2^(e-7) in BF16 |
| Zero-sum property | Softmax cross entropy gradient sums to zero over the vocabulary |
| Zero-sum projection | Liu et al.'s fix subtracting the mean logit gradient so gradients sum to zero |
| z-loss | Auxiliary penalty lambda x log^2 Z on the softmax normalizer |