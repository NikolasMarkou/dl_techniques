# Layer Ordering: Dense, Normalization, Activation (v2)

A practical, source-backed reference on where to place the normalization layer relative to the linear (Dense/Conv) layer and the activation function. Updated with current (2025-2026) practice.

## TL;DR

| Situation | Recommended order | Confidence |
|---|---|---|
| Default CNN / MLP, unsure | Dense -> Norm -> Activation | High (safe default) |
| ReLU / GELU CNNs and MLPs | Dense -> Norm -> Activation, try Norm-after as a tweak | Medium (difference is usually small) |
| Sigmoid / Tanh (bounded) | Classic: Norm before. But recent evidence is mixed | Medium (contested) |
| Transformers / LLMs | Pre-Norm: Norm inside the residual branch, before each sub-block | High (current standard) |

There is no single universally correct order. Option 1 is the best default for feedforward/conv nets. The picture genuinely changes for bounded activations and for transformers.

---

## Option 1: Dense -> Norm -> Activation

```
x -> Dense(Wx + b) -> Norm -> Activation -> out
```

### Why use it

This is the original formulation from the BatchNorm paper (Ioffe and Szegedy, 2015), which normalized the pre-activation. It is the generally accepted default and the order assumed by most frameworks and tutorials.

### Reasoning

1. **Protects saturating activations.** Normalizing to roughly zero mean and unit variance before the nonlinearity keeps inputs in the responsive region of sigmoid/tanh, mitigating vanishing gradients. This was the original motivation.
2. **Controlled input distribution to the nonlinearity.** The activation always receives a stable distribution regardless of weight scale, smoothing the loss surface and allowing higher learning rates.
3. **Helps avoid dying ReLUs.** Properly scaled pre-activations reduce the chance of ReLU units getting stuck at zero.
4. **Best supported.** Most reference implementations and pretrained weights assume this order.

### Drawbacks

- After ReLU, roughly half the values are zeroed, so the input to the next layer no longer has zero mean. Part of the normalization guarantee is consumed by the activation.
- Mild redundancy between BN's learnable scale/shift and what the activation then distorts.

---

## Option 2: Dense -> Activation -> Norm

```
x -> Dense(Wx + b) -> Activation -> Norm -> out
```

### Why use it

An empirical alternative. For ReLU, results are typically close to Option 1, so it is a reasonable thing to try. For some tasks and some architectures it wins.

### Reasoning

1. **Clean input to the next layer.** Normalizing after the activation means the next Dense/Conv layer receives a controlled distribution even though ReLU has already removed the negatives. You normalize what the next layer actually sees.
2. **ReLU does not saturate on the positive side**, so the original "protect the activation" argument for Option 1 is weaker with ReLU.
3. **Negatives excluded from statistics.** With ReLU, putting Norm after means the discarded negative values do not pollute the minibatch mean and variance.

### Important nuances from recent work

- For **ReLU**, the swapped order is generally not much different from the conventional order. Treat it as a hyperparameter, not a fix.
- For **bounded activations like Tanh**, the classic theory says normalize before, but this is now contested. At least one study reports that placing BatchNorm after a bounded activation achieves considerably better results across several benchmarks and architectures. Other work in different settings (e.g. PDE solvers) still finds Norm-before-tanh trains more smoothly. So for bounded activations, test both rather than assuming.
- Initialization theory (2025) shows the two placements induce qualitatively different initial states: Norm-before-ReLU gives a "weakly prejudiced" but depth-stable init, while Norm-after-ReLU promotes a "neutral" init.

### Drawbacks

- Less standard, so fewer pretrained models and references follow it.
- Not safe to assume it helps; on many tasks it is a wash.

---

## Transformers and LLMs: Pre-Norm is the standard

Modern large language models do not use either simple ordering above. They use **Pre-Norm**, where the normalization is applied to the input of each sub-block (attention or feed-forward) inside the residual branch.

```
Pre-Norm  (modern):  y = x + Module(Norm(x))
Post-Norm (original): y = Norm(x + Module(x))
```

The original Transformer (Vaswani et al., 2017) used Post-Norm. Most current models switched to Pre-Norm.

### Reasoning

1. **Training stability at depth.** Pre-Norm keeps a clean identity path through the residual connections, improving gradient flow and preventing the loss spikes and divergence that plague deep Post-Norm models.
2. **Little or no warmup needed.** Pre-Norm tolerates larger learning rates and reduces or removes the need for learning-rate warmup schedules.
3. **Tradeoff.** Pre-Norm can underperform Post-Norm in final quality and tends to grow activation variance with depth, while Post-Norm keeps variance roughly constant but degrades gradient flow. This tension drives the newer hybrids below.

### Normalization choice: RMSNorm

Most modern LLMs replaced LayerNorm with **RMSNorm**, which drops the mean-centering step and only rescales by the root mean square. It is cheaper and works because transformer activations are approximately centered already. The LLaMA recipe (Pre-Norm + RMSNorm + SwiGLU + RoPE) became the de facto open-source template, adopted by Mistral, Qwen, DeepSeek, Gemma, Phi, and others. Falcon is a notable exception that kept LayerNorm after ablations, showing the "standard" choices are not universally optimal.

### What is new (2024-2026)

The field is moving beyond plain Pre-Norm:

- **Double Norm / sandwich norm.** Apply normalization both before and after each sub-block. Grok and Gemma 2 use LayerNorm before and after each block.
- **OLMo 2** places normalization only after each block (a post-norm-style variant inside the residual structure).
- **Peri-LN** and **HybridNorm** are recent proposals that mix placements to get linear (rather than exponential) variance growth and better stability than either pure Pre-Norm or pure Post-Norm.
- **Normalization-free directions** such as Dynamic Tanh (DyT) attempt to replace the norm layer entirely.

Note this is a different design axis from the Dense/Norm/Activation question: it is about where Norm sits relative to the residual connection, not just relative to the activation.

---

## Decision rules

1. **CNN / MLP with ReLU, GELU, Leaky ReLU:** start with Dense -> Norm -> Activation. If validation stalls, try Norm-after as a hyperparameter; expect a small effect.
2. **Bounded activations (sigmoid, tanh):** classic default is Norm before, but test both, since recent evidence shows Norm-after can win for bounded activations.
3. **Transformers / LLMs:** use Pre-Norm with RMSNorm before each attention and FFN sub-block, plus a final norm on the network output. Consider double-norm/sandwich variants for very deep or unstable training.
4. **Match the norm type to the order.** The before/after debate is mainly a BatchNorm concern. LayerNorm/RMSNorm normalize per sample and are less sensitive to it, but the saturation logic still applies to bounded activations.

---

## Common misconceptions corrected

- **"ResNet proves Option 1 is the only correct order."** Only ResNet v1. The improved pre-activation ResNet (He et al., 2016) reorders to Norm -> ReLU -> Conv and trains better. Even ResNet does not endorse a single ordering.
- **"You must always normalize before the activation."** Overstated. For ReLU the difference is usually small; for bounded activations the answer is genuinely contested.
- **"Softmax needs pre-normalization like sigmoid."** Softmax is an output layer; you do not normalize hidden pre-activations into it the way you would for hidden sigmoid/tanh units.
- **"Pre-Norm is just Option 1 applied to transformers."** No. Pre-Norm refers to placement relative to the residual connection, not relative to the activation inside the FFN.

---

## References

- Ioffe and Szegedy, "Batch Normalization", 2015 (original Norm-before-activation formulation).
- He et al., "Identity Mappings in Deep Residual Networks", 2016 (pre-activation ResNet).
- Vaswani et al., "Attention Is All You Need", 2017 (original Post-Norm Transformer).
- Xiong et al., "On Layer Normalization in the Transformer Architecture", 2020 (Pre-Norm vs Post-Norm analysis).
- Zhang and Sennrich, "Root Mean Square Layer Normalization" (RMSNorm), 2019.
- "Batch Normalization and Bounded Activation Functions", OpenReview (Norm-after wins for tanh).
- "Where You Place the Norm Matters", arXiv 2505.11312, 2025 (initialization effects of BN placement).
- "Peri-LN", arXiv 2502.02732, 2025; "HybridNorm", arXiv 2503.04598, 2025 (modern hybrid placements).
- Touvron et al., LLaMA, 2023 (Pre-Norm + RMSNorm + SwiGLU + RoPE recipe).