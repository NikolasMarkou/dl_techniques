# CliffordNet Training & Inference

Training pipelines and inference tools for the CliffordNet model family -- geometric-algebra neural networks that replace both attention and FFN with Clifford geometric products.

All models share the same algebraic core (`SparseRollingGeometricProduct` + `GatedGeometricResidual`) from `dl_techniques/layers/geometric/clifford_block.py`. The family spans three domains: vision classification, causal language modeling, and image denoising.

**Reference**: Brandstetter, J. et al. (2025). *CliffordNet: All You Need is Geometric Algebra*. arXiv:2601.06793v2.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Vision Classification](#2-vision-classification)
3. [NLP Pre-training](#3-nlp-pre-training)
4. [Image Denoising](#4-image-denoising)
5. [Power Sampling Inference](#5-power-sampling-inference)

---

## 1. Overview

| Script | Model | Domain | Description |
|:-------|:------|:-------|:------------|
| `train_cliffordnet.py` | `CliffordNet` | Vision | CIFAR-10/100 classification with AutoAugment |
| `train_cliffordnet_nlp.py` | `CliffordNetLM` | NLP | Causal language model pre-training on Wikipedia |
| `train_denoiser.py` | `CliffordNetDenoiser` | Vision | Bias-free image denoiser (Miyasawa's theorem) |
| `infer_cliffordnet_nlp.py` | `CliffordNetLM` | NLP | Inference with standard and power sampling |

### Shared design

All three architectures are **attention-free and FFN-free**. Each block follows the same pattern:

```
Input
  +--> Detail stream (1x1 Dense)
  +--> Context stream (depthwise conv, optionally causal)
  |
  v
SparseRollingGeometricProduct(detail, context)   -- replaces attention + FFN
  |
  v
GatedGeometricResidual(input, product)           -- LayerScale + DropPath
  |
  v
Output = Input + residual
```

The context stream extracts local spatial/temporal patterns; the geometric product fuses them with the detail stream across multiple channel shifts, approximating the Clifford algebra product `AB = A . B + A ^ B`.

---

## 2. Vision Classification

**Script**: `train_cliffordnet.py`
**Model**: `CliffordNet` (`dl_techniques/models/cliffordnet/model.py`)

Trains the CliffordNet vision backbone on CIFAR-10 or CIFAR-100 following the protocol from the original paper: AdamW with cosine decay + linear warmup, AutoAugment (CIFAR-10 policy), random flip/crop, per-channel normalization, and random erasing.

### Architecture

```
Input (B, H, W, 3)
  --> GeometricStem (Conv2D + BN)
  --> L x CliffordNetBlock (DWConv context, geometric product, GGR)
  --> GlobalAvgPool --> LayerNorm --> Dense
  --> Logits (B, num_classes)
```

### Variants

| Variant | Channels | Depth | Shifts | Global Context |
|:--------|:--------:|:-----:|:-------|:--------------:|
| `nano` | 128 | 12 | [1, 2] | No |
| `lite` | 128 | 12 | [1, 2, 4, 8, 16] | No |
| `lite_g` | 128 | 12 | [1, 2, 4, 8, 16] | Yes |

### Usage

```bash
# CIFAR-10, nano variant
python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar10 --variant nano --epochs 200 --gpu 0

# CIFAR-100, lite_g (with global context)
python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar100 --variant lite_g --epochs 300

# Custom architecture
python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar10 --variant custom \
    --channels 192 --depth 16 --shifts 1,2,4,8 \
    --cli-mode full --ctx-mode diff
```

### Training protocol

| Setting | Value |
|:--------|:------|
| Optimizer | AdamW (beta1=0.9, beta2=0.999) |
| LR schedule | Cosine decay with linear warmup |
| Default LR | 1e-3 |
| Warmup | 10 epochs |
| Weight decay | 0.05 (decoupled, no L2 regularizer) |
| Augmentation | AutoAugment (CIFAR-10 policy) + random flip + pad-4/crop-32 + random erasing |
| Normalization | Per-channel (dataset mean/std) |

### Key parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `--dataset` | `cifar10` | Dataset (`cifar10`, `cifar100`) |
| `--variant` | `nano` | Model variant (`nano`, `lite`, `lite_g`, `custom`) |
| `--epochs` | 200 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--learning-rate` | 1e-3 | Peak learning rate |
| `--weight-decay` | 0.05 | AdamW weight decay |
| `--warmup-epochs` | 10 | Linear warmup epochs |
| `--dropout-rate` | 0.0 | Dropout rate |
| `--stochastic-depth-rate` | 0.1 | Max stochastic depth rate |
| `--cli-mode` | `full` | Clifford components (`inner`, `wedge`, `full`) |
| `--ctx-mode` | `diff` | Context mode (`diff` = discrete Laplacian, `abs`) |

### Post-training

After training, the script automatically:
- Saves the final model and best checkpoint
- Validates serialization round-trip
- Runs `ModelAnalyzer` (weight/spectral analysis)
- Generates training curves and evaluation metrics

---

## 3. NLP Pre-training

**Script**: `train_cliffordnet_nlp.py`
**Model**: `CliffordNetLM` (`dl_techniques/models/cliffordnet/lm.py`)

Pre-trains a causal language model on English Wikipedia (HuggingFace) or TFDS text datasets. The `CausalCliffordNetBlock` operates on 4D tensors `(B, 1, seq_len, D)` where left-only padded depthwise convolutions enforce strict autoregressive causality -- no attention mask needed.

### Architecture

```
Token IDs (B, seq_len)
  --> Token Embedding + Learned Positional Embedding
  --> LayerNorm --> Dropout
  --> Reshape to (B, 1, seq_len, D)
  --> L x CausalCliffordNetBlock    (left-padded DWConv, causal cumulative mean)
  --> Squeeze to (B, seq_len, D)
  --> LayerNorm --> Dropout --> Dense(vocab_size)
  --> {"logits": (B, seq_len, vocab_size)}
```

### Variants

| Variant | Channels | Depth | Shifts | Stoch. Depth |
|:--------|:--------:|:-----:|:-------|:------------:|
| `nano` | 128 | 12 | [1, 2] | 0.05 |
| `mini` | 192 | 12 | [1, 2, 4] | 0.10 |
| `base` | 384 | 18 | [1, 2, 4, 8, 16] | 0.15 |
| `large` | 512 | 20 | [1, 2, 4, 8, 16] | 0.20 |
| `xl` | 768 | 28 | [1, 2, 4, 8, 16] | 0.25 |

### Tokenizer

Uses `tiktoken` with the GPT-2 BPE encoding (50,257 base tokens) plus 4 special tokens:

| Token | ID | Purpose |
|:------|:---|:--------|
| CLS | 50257 | Sequence start marker |
| SEP | 50258 | Separator |
| PAD | 50259 | Padding |
| MASK | 50260 | Masking (for future MLM) |

Total vocabulary size: 50,261.

### Usage

```bash
# Wikipedia (default), nano variant
python -m train.cliffordnet.train_cliffordnet_nlp \
    --gpu 0 --variant nano --epochs 3

# TFDS dataset with focal loss
python -m train.cliffordnet.train_cliffordnet_nlp \
    --dataset-source tfds --dataset-name imdb_reviews --loss-type focal

# Custom architecture
python -m train.cliffordnet.train_cliffordnet_nlp \
    --variant custom --channels 256 --depth 12 --shifts 1,2,4,8

# Resume from checkpoint
python -m train.cliffordnet.train_cliffordnet_nlp \
    --resume results/.../checkpoints/step_0050000.keras
```

### Training protocol

| Setting | Value |
|:--------|:------|
| Optimizer | AdamW (clipnorm=1.0) |
| LR schedule | Warmup (10% of steps) + cosine decay |
| Default LR | 3e-4 |
| Weight decay | 0.01 |
| Loss | `MaskedCausalLMLoss` or `FocalCausalLMLoss` |
| Data | Wikipedia (streaming, no cache to avoid OOM) |

### Features

- **Step-based checkpointing**: saves every N steps (default 25,000) instead of per-epoch, since Wikipedia epochs are very long
- **Generation probes**: autoregressive text generation at each checkpoint to track quality over training (nucleus sampling with repetition penalty)
- **Step-level CSV logging**: loss/accuracy logged every 100 steps
- **Configurable loss**: standard cross-entropy (`MaskedCausalLMLoss`) or focal loss (`FocalCausalLMLoss`) for handling token frequency imbalance

### Key parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `--variant` | `nano` | Model variant |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 8 | Batch size |
| `--max-seq-length` | 512 | Maximum sequence length |
| `--learning-rate` | 3e-4 | Peak learning rate |
| `--loss-type` | `ce` | Loss function (`ce` or `focal`) |
| `--dataset-source` | `huggingface` | Data source (`huggingface` or `tfds`) |
| `--checkpoint-every-steps` | 25000 | Checkpoint interval |
| `--analyze-every-steps` | 50000 | Weight/spectral analysis interval |

---

## 4. Image Denoising

**Script**: `train_denoiser.py`
**Model**: `CliffordNetDenoiser` (`dl_techniques/models/cliffordnet/denoiser.py`)

Trains a **bias-free** image denoiser built on CliffordNet blocks. The bias-free constraint satisfies Miyasawa's theorem (1961): a least-squares denoiser must have zero mean output, which implies no additive bias anywhere in the network. This makes the denoiser an implicit score function estimator, suitable for diffusion models.

### Bias-free design

Every component in the network obeys:
- **No additive bias** in any Conv2D or Dense layer (`use_bias=False`)
- **Normalization without centering**: `BatchNorm(center=False)`, `LayerNorm(center=False)` -- scale only, no shift
- **Linear final output**: no activation on the output projection
- **Residual learning**: `output = input + f(input)` so the network learns the noise residual
- **Zero-centered inputs**: data normalized to `[-1, +1]`

### Architecture

```
Input (B, H, W, C) in [-1, 1]
  --> BatchNorm(center=False) --> Conv2D(bias=False)
  --> L x BiasFreeClifordNetBlock
  --> LayerNorm(center=False)
  --> Conv2D(C, 1x1, bias=False, linear) --> residual
  --> Output = Input + residual
```

### Variants

| Variant | Channels | Depth | Shifts | Global Context |
|:--------|:--------:|:-----:|:-------|:--------------:|
| `tiny` | 64 | 6 | [1, 2] | No |
| `small` | 96 | 8 | [1, 2, 4] | No |
| `base` | 128 | 12 | [1, 2, 4, 8] | No |
| `large` | 128 | 16 | [1, 2, 4, 8, 16] | Yes |

### Usage

```bash
# Train tiny denoiser on grayscale patches
python -m train.cliffordnet.train_denoiser \
    --model-variant tiny \
    --epochs 100 --batch-size 32 \
    --patch-size 128 --channels 1 \
    --train-dirs /data/train --val-dirs /data/val \
    --gpu 0

# RGB denoiser, base variant, custom noise range
python -m train.cliffordnet.train_denoiser \
    --model-variant base \
    --channels 3 --patch-size 256 \
    --noise-sigma-min 0.0 --noise-sigma-max 0.3 \
    --train-dirs /data/train --val-dirs /data/val
```

### Training protocol

| Setting | Value |
|:--------|:------|
| Optimizer | AdamW |
| LR schedule | Cosine decay with warmup |
| Default LR | 1e-3 |
| Warmup | 5 epochs |
| Weight decay | 1e-5 |
| Gradient clipping | 1.0 |
| Loss | MSE (pixel-level reconstruction) |
| Metric | PSNR |
| Noise | Gaussian, uniform or log-uniform sigma in [sigma_min, sigma_max] |

### Key parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `--model-variant` | `tiny` | Denoiser variant |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--patch-size` | 128 | Random crop patch size |
| `--channels` | 1 | Image channels (1=grayscale, 3=RGB) |
| `--noise-sigma-min` | 0.0 | Minimum noise std |
| `--noise-sigma-max` | 0.5 | Maximum noise std |
| `--train-dirs` | (required) | Training image directories |
| `--val-dirs` | (required) | Validation image directories |

---

## 5. Power Sampling Inference

**Script**: `infer_cliffordnet_nlp.py`
**Module**: `dl_techniques/models/cliffordnet/power_sampling.py`

After training `CliffordNetLM`, generate text using **power sampling** -- an inference-time method that improves reasoning and coherence without any additional training or reward models.

### The problem with standard sampling

Standard autoregressive decoding samples one token at a time from the model's distribution *p*. Low-temperature sampling sharpens the distribution but only at the *local* (per-token) level. This amplifies shortcuts:

- Guessing instead of planning
- Premature answers
- Locally plausible but globally poor trajectories

Low temperature asks *"How likely is this token?"* -- but reasoning is fundamentally about *"If I pick this token, how good are the futures it leads to?"*

### Power sampling: trajectory-level optimization

Power sampling targets the *power distribution* **p^alpha** (where alpha = 1/temperature > 1). Instead of sharpening individual token probabilities, it reweights entire trajectories to favor globally more coherent sequences.

The key insight from the literature: **reasoning capabilities attributed to RL post-training (GRPO, RLHF) may already exist in the base model's distribution**. RL doesn't inject new ideas into the model -- it reshapes probability mass, making certain trajectories more likely. Power sampling achieves the same effect at inference time by amplifying high-probability trajectories through MCMC refinement.

### Algorithm

Three generation methods are available:

| Method | Sampling from | Speed | Use case |
|:-------|:-------------|:------|:---------|
| `standard` | *p* (nucleus sampling) | Fast (1x) | Baseline generation |
| `power` | *p^alpha* (MCMC) | ~160x forward passes | Best trajectory coherence |
| `max_swap` | *p^infinity* (deterministic) | ~160x forward passes | Highest probability trajectories |

**MCMC power sampling** (`power` method):

1. Split generation into `block_num` blocks (default 16)
2. For each block, generate `jump_size` tokens using low-temperature proposal sampling
3. Run `mcmc_steps` (default 10) Metropolis-Hastings refinement steps:
   - Pick a random position *idx* in the generated sequence
   - Re-generate from *idx* to the end as a proposal
   - Compute the MH acceptance ratio:
     ```
     log r = sum(target_log_prob_proposal) + sum(proposal_log_prob_current)
           - sum(target_log_prob_current) - sum(proposal_log_prob_proposal)
     ```
     where `target_log_prob = (1/temp) * log p(token)` is the power distribution
   - Accept with probability `min(1, exp(log r))`
4. The `max_swap` variant accepts deterministically whenever `log r > 0` (greedy trajectory optimization, approximating *p^infinity*)

### Usage

```bash
# Standard nucleus sampling (baseline)
python -m train.cliffordnet.infer_cliffordnet_nlp \
    --checkpoint results/.../checkpoints/final.keras \
    --prompt "The capital of France is" \
    --method standard --gpu 0

# MCMC power sampling (alpha=4)
python -m train.cliffordnet.infer_cliffordnet_nlp \
    --checkpoint results/.../checkpoints/final.keras \
    --prompt "In mathematics, a prime number is" \
    --method power --temperature 0.25 --mcmc-steps 10

# Max-swap (deterministic trajectory optimization)
python -m train.cliffordnet.infer_cliffordnet_nlp \
    --checkpoint results/.../checkpoints/final.keras \
    --prompt "Albert Einstein was born in" \
    --method max_swap

# Compare all three methods side-by-side
python -m train.cliffordnet.infer_cliffordnet_nlp \
    --checkpoint results/.../checkpoints/final.keras \
    --prompt "The theory of relativity states that" \
    --compare

# Batch prompts from file, save results as JSON
python -m train.cliffordnet.infer_cliffordnet_nlp \
    --checkpoint results/.../checkpoints/final.keras \
    --prompts-file prompts.txt --method power \
    --output-json results.json
```

### Parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `--temperature` | 0.25 | Sampling temperature; alpha = 1/temp (0.25 -> alpha=4) |
| `--mcmc-steps` | 10 | MH refinement steps per block |
| `--block-num` | 8 | Number of generation blocks |
| `--max-tokens` | 100 | Maximum tokens to generate |
| `--top-p` | 0.92 | Nucleus sampling threshold |
| `--repetition-penalty` | 1.3 | Sign-aware repetition penalty |

### Programmatic API

```python
import tiktoken
from dl_techniques.models.cliffordnet.power_sampling import (
    PowerSampler,
    PowerSamplingConfig,
)

# Load trained model
model = load_model("checkpoints/final.keras")
enc = tiktoken.get_encoding("gpt2")

# Configure sampler
config = PowerSamplingConfig(
    temperature=0.25,     # alpha = 4
    mcmc_steps=10,
    block_num=8,
    max_tokens=200,
)
sampler = PowerSampler(model, enc, config=config)

# Generate with power sampling
text, info = sampler.generate_text(
    "The theory of relativity states that",
    method="power",
)
print(text)
print(f"Acceptance ratio: {info['acceptance_ratio']:.1%}")
print(f"Time: {info['elapsed_s']:.1f}s")

# Compare methods
for method in ["standard", "power", "max_swap"]:
    text, info = sampler.generate_text("Once upon a time", method=method)
    print(f"\n--- {method} ---")
    print(text[:200])
```

### Implementation notes

- All sampling logic uses NumPy (not TF ops) to avoid Keras graph retraces
- Fixed-shape input padding (511 tokens) ensures a single compiled graph for all forward passes
- No KV-cache: each forward pass recomputes all positions. Power sampling is designed for offline/batch inference, not real-time generation
- The MCMC acceptance criterion is the standard Metropolis-Hastings ratio adapted for autoregressive models

### References

1. Karan, A. & Du, Y. (2025). **Reasoning with Sampling: Your Base Model is Smarter Than You Think**. arXiv:2510.14901. [Paper](https://arxiv.org/abs/2510.14901) | [Code](https://github.com/aakaran/reasoning-with-sampling)
2. Bou Ammar, H. et al. (2026). **Scalable Power Sampling for LLM Reasoning**. arXiv:2601.21590. [Paper](https://arxiv.org/abs/2601.21590)
3. Brandstetter, J. et al. (2025). **CliffordNet: All You Need is Geometric Algebra**. arXiv:2601.06793v2.
