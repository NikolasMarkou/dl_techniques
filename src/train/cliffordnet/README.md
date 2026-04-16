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
6. [CLIP Contrastive Pretraining](#6-clip-contrastive-pretraining)

---

## 1. Overview

| Script | Model | Domain | Description |
|:-------|:------|:-------|:------------|
| `train_cliffordnet.py` | `CliffordNet` | Vision | CIFAR-10/100 classification with AutoAugment |
| `train_cliffordnet_nlp.py` | `CliffordNetLM` | NLP | Causal language model pre-training on Wikipedia |
| `train_denoiser.py` | `CliffordNetDenoiser` | Vision | Bias-free image denoiser (Miyasawa's theorem) |
| `infer_cliffordnet_nlp.py` | `CliffordNetLM` | NLP | Inference with standard and power sampling |
| `train_clip.py` | `CliffordCLIP` | Vision-Language | CLIP-style dual encoder (both towers Clifford) |
| `prepare_cc3m.py` | -- | Data prep | Resumable CC3M extractor from HF Hub tar shards |

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

---

## 6. CLIP Contrastive Pretraining

**Script**: `train_clip.py`
**Model**: `CliffordCLIP` (`dl_techniques/models/cliffordnet/clip.py`)
**Data prep**: `prepare_cc3m.py`
**Reference**: Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). arXiv:2103.00020.

A CLIP-style dual encoder where **both towers are built from Clifford blocks** -- no attention anywhere in either the vision or the text path. The vision tower uses bidirectional `CliffordNetBlock`s over 2-D feature maps; the text tower uses `CausalCliffordNetBlock`s over a `(B, 1, seq_len, D)` layout so left-only depthwise context enforces autoregressive compatibility at the feature-extraction stage.

### Architecture

```
Image (B, H, W, 3)                           Tokens (B, L)
     |                                            |
  Conv2D patch stem + BN                    Token emb + positional emb + LN
     |                                            |
  L x CliffordNetBlock                      expand to (B, 1, L, D)
  (bidirectional 2-D DWConv context)              |
     |                                      L x CausalCliffordNetBlock
  GlobalAvgPool                             (left-padded DWConv context)
     |                                            |
  LayerNorm                                 squeeze to (B, L, D)
     |                                            |
  Dense(embed_dim)                          gather last token --> Dense(embed_dim)
     |                                            |
  L2-normalize                              L2-normalize
            \                            /
             cosine similarity @ exp(logit_scale)
                        |
             Symmetric InfoNCE loss
             (label-smoothed, both directions)
```

### Variants

| Variant | Vision channels | Vision depth | Text channels | Text depth | Embed dim | Params |
|:--------|:---------------:|:------------:|:-------------:|:----------:|:---------:|:------:|
| `nano` | 96 | 8 | 128 | 6 | 256 | ~8.5M |
| `mini` | 128 | 10 | 192 | 8 | 384 | ~18M |
| `base` | 192 | 12 | 256 | 10 | 512 | ~45M |
| `large` | 256 | 16 | 384 | 12 | 768 | ~120M |

### Tokenizer

Uses `tiktoken` with the GPT-2 BPE encoding (50,257 tokens, English-trained). The encoder's native `eot_token` (ID 50256) doubles as both end-of-sequence and the pad sentinel. This avoids collisions with real caption tokens and matches the CLIP convention -- no extra special tokens are added.

### Datasets

| Dataset | Pairs | Loader | Layout |
|:--------|:------|:-------|:-------|
| Synthetic | any | `build_synthetic_image_text_dataset` | in-memory random tensors (smoke tests) |
| MS-COCO 2017 | ~118 k | `load_coco2017_local_split` | `train2017/*.jpg`, `annotations/captions_*.json` |
| CC3M | ~2.9 M | `load_cc3m_local_split` | extracted by `prepare_cc3m.py` (see below) |

All loaders return a `(list[str], np.ndarray int32[N, L])` tuple -- image paths plus right-padded token IDs -- so the training script dispatches between datasets via a single `--dataset` flag. See `train.common.image_text` for the shared surface.

### CC3M extractor: `prepare_cc3m.py`

Streams `pixparse/cc3m-wds` tar shards from the Hugging Face Hub directly via `huggingface_hub` + stdlib `tarfile`. The `datasets.load_dataset` path is deliberately avoided because `pixparse/cc3m-wds`'s JSON sidecar column fails HF's feature-schema cast.

Usage:

```bash
PYTHONPATH=src python -m train.cliffordnet.prepare_cc3m \
    --dst /path/to/cc3m \
    --splits validation train \
    --progress-every 10000
```

Output layout (matching `load_cc3m_local_split`):

```
<dst>/
  train/XX/cc3m_train_NNNNNNNN.jpg          # 256-way hash-shard
  validation/XX/cc3m_validation_NNNNNNNN.jpg
  train_captions.jsonl                       # one {id, caption, split} per line
  validation_captions.jsonl
  _prepare_cc3m_state.json                   # completed tar shard list
```

**Resumable**: two layers of safety. The state file records completed tar shards so re-running skips them. Individual JPEGs are additionally skipped if the target path already exists, so even a re-processed tar produces no duplicates. Network errors are caught per-shard and reported in the log (`errors=N`) without aborting the run -- re-run the same command to pick up any partially-failed shards.

**Observed throughput**: ~120 pairs/sec on a typical home uplink. Full 2.9M extraction takes ~7 hours end-to-end, yielding ~230 GiB on disk.

### Training script: `train_clip.py`

Usage:

```bash
# Smoke run: 100k CC3M subset, ~12.5k steps, ~1h on RTX 4070
PYTHONPATH=src python -m train.cliffordnet.train_clip \
    --variant mini --dataset cc3m \
    --batch-size 32 --context-length 64 \
    --stage1-image-size 112 --stage1-epochs 4 --stage1-lr 5e-4 \
    --stage2-epochs 0 \
    --warmup-ratio 0.03 --weight-decay 0.1 --dropout-rate 0.1 \
    --label-smoothing 0.1 \
    --max-train-samples 100000 --max-val-samples 5000 \
    --save-every-steps 500 --log-every-steps 50 \
    --probe-every-steps 500 --probe-num-pairs 512 \
    --max-checkpoints 3 --gpu 1

# Full run: 950k CC3M, ~30k steps, ~5h on RTX 4090
PYTHONPATH=src python -m train.cliffordnet.train_clip \
    --variant mini --dataset cc3m \
    --batch-size 32 --context-length 64 \
    --stage1-image-size 112 --stage1-epochs 1 --stage1-lr 5e-4 \
    --stage2-epochs 0 \
    --warmup-ratio 0.03 --weight-decay 0.1 --dropout-rate 0.1 \
    --label-smoothing 0.1 \
    --max-train-samples 950000 --max-val-samples 5000 \
    --save-every-steps 2000 --log-every-steps 100 \
    --probe-every-steps 2000 --probe-num-pairs 512 \
    --max-checkpoints 3 --gpu 0
```

### Training protocol

| Setting | Value |
|:--------|:------|
| Optimizer | AdamW, `logit_scale` **excluded** from weight decay |
| LR schedule | Cosine decay with linear warmup (3% of steps) |
| Default LR | 5e-4 |
| Weight decay | 0.1 |
| Label smoothing | 0.1 |
| Loss | `CLIPContrastiveLoss` (symmetric InfoNCE), `apply_temperature=False` |
| Temperature | Learnable `logit_scale`, init 2.6592 (=> `exp(...) = 14.3`), capped at 100 |
| Augmentation | Random crop + horizontal flip + OpenAI/CLIP normalization |
| Image normalization | `[0.4814, 0.4578, 0.4082]` mean / `[0.2686, 0.2613, 0.2758]` std |

### Key parameters

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `--variant` | `mini` | Model variant (`nano`, `mini`, `base`, `large`) |
| `--dataset` | `synthetic` | `synthetic`, `coco`, `cc3m` |
| `--batch-size` | 32 | Proven to fit mini on 12 GB VRAM at 112² |
| `--context-length` | 64 | Text sequence length |
| `--stage1-image-size` | 112 | Training resolution (single-stage proven) |
| `--stage1-epochs` | 10 | Stage-1 epochs |
| `--stage2-epochs` | 0 | Stage-2 (higher resolution) -- set 0 for single-stage |
| `--stage1-lr` | 5e-4 | Peak LR |
| `--warmup-ratio` | 0.03 | Linear warmup fraction |
| `--label-smoothing` | 0.1 | Contrastive loss label smoothing |
| `--save-every-steps` | 2000 | Checkpoint cadence |
| `--probe-every-steps` | 2000 | Retrieval probe cadence |
| `--probe-num-pairs` | 512 | Probe-set size (quick signal, not full val) |
| `--max-checkpoints` | 3 | Rolling checkpoint window |
| `--max-train-samples` | `None` | Optional cap for budget-constrained runs |

### Features

- **Step-based checkpointing** (not per-epoch) -- CC3M epochs are long, a rolling `step_NNNNNNN.keras` window + `final.keras` gives fine-grained recovery
- **Retrieval probes every N steps**: R@1/R@5/R@10 on a small held-out slice (512 pairs by default), appended to `retrieval_probes/probes.jsonl` -- curve diagnostics without burning minutes on full 5k-pair eval
- **Early GPU binding**: parses `--gpu` from `sys.argv` **before** `import tensorflow` so `CUDA_VISIBLE_DEVICES` is set before TF claims all visible devices -- the "GPU setup error" logged late is benign and does not indicate a real failure
- **`logit_scale` excluded from weight decay** via `optimizer.exclude_from_weight_decay(var_names=["logit_scale"])` -- without this AdamW decays the learnable temperature toward zero and kills the contrastive signal. Verify the startup log line `Excluded 'logit_scale' from AdamW weight decay.` on every run

### Reference results

All runs share the same mini config (18M params, batch 32, 112², label smoothing 0.1, single-stage). Retrieval R@K is computed on the respective val split at the end of training.

| Run | Steps | Data | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5 | i2t R@10 |
|:----|------:|:-----|--------:|--------:|--------:|--------:|--------:|
| COCO-mini-v4 (baseline) | 10,000 | 32 k COCO | 0.40% | 0.48% | 1.92% | 2.40% | 3.86% |
| CC3M smoke (plain head, old code) | 12,500 | 100 k CC3M (4 ep) | 0.46% | 0.38% | 1.58% | 1.38% | 2.64% |
| **CC3M full (plain head)** | 29,687 | 950 k CC3M (1 ep) | **1.08%** | **1.22%** | **3.82%** | **5.14%** | **6.70%** |

On a 5 k-pair val pool, random R@1 is 0.02% and random R@5 is 0.10%. The CC3M-full run's i2t R@1 is **~54× random** and R@5 is **~38× random**. Comparable per-step behavior to the COCO baseline, with CC3M's larger and more diverse pool pushing absolute numbers up meaningfully once the full budget is spent.

### Projection-head A/B sweep (CC3M-smoke, 12,500 steps, 5 k val pairs)

Same config as CC3M-smoke above; only ``--head-kind`` and ``--head-cli-mode`` vary. Runs performed on RTX 4090 (current hardware) for same-day apples-to-apples comparison.

| Head | cli_mode | Params | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5 | i2t R@10 | t2i R@10 |
|:----|:--------|-------:|--------:|--------:|--------:|--------:|---------:|---------:|
| `plain` (fresh baseline) | —    | 18.04 M | 0.32%  | 0.38%  | 1.46%  | 1.94%  | 3.02% | 3.32% |
| `mean_max` | full | 18.34 M | 0.30%  | 0.16%  | 1.42%  | 1.20%  | 2.50% | 2.42% |
| `learned_query` | full | 18.34 M | 0.24%  | 0.44%  | 1.62%  | 1.88%  | 2.86% | 3.08% |
| **`learned_query_residual`** *(default)* | full | 18.34 M | **0.46%** | **0.40%** | 1.50%  | 1.90%  | 2.94% | **3.38%** |
| `learned_query_residual` | wedge | 18.19 M | 0.36%  | 0.38%  | 1.50%  | 1.40%  | 2.34% | 2.92% |

**Takeaways:**

- `plain` and `learned_query_residual full` are the top two heads. The residual variant matches or beats plain on 5/6 metrics, with the largest win (+0.14 pp, +44% relative) on the hardest metric (i2t R@1). It also matches the old-hardware README baseline on i2t R@1 and beats it on t2i R@5, i2t R@10.
- `mean_max` (mean + max-pool via geometric product) loses on every metric — max-pool is the wrong second view for Clifford mixing.
- `learned_query` (non-residual) ties plain on average but is noisy per-metric. Using the geometric product as the *sole* embedding signal underperforms using it as a residual additive term.
- `wedge`-only cli_mode underperforms `full`; the symmetric inner-product component still contributes.

The default `head_kind` is set to `learned_query_residual` with `cli_mode=full`. The LayerScale gamma (init 1e-5) means the head starts behaving like plain CLIP and lets gradients pull in Clifford content only where it measurably helps.

**Healthy training diagnostics** (from the CC3M-full run):

- Loss crossed `log(batch_size) = log(32) ≈ 3.47` by step 1,000 (3.4% of budget)
- Loss descended monotonically from 3.48 to ~2.0 by step 12,000, then plateaued
- `logit_scale` (= `exp(variable)`) bottomed at ~9.62 around step 18-20 k, then climbed back to ~9.77 by the end -- the canonical CLIP **bottom-and-climb-back** signal indicating the model has crossed from "random discrimination" to "better-than-random discrimination" (see note below)
- Probe R@5 monotonically increased from 1.0% to 17% over the run

**Note on `logit_scale` dynamics**: the gradient of the symmetric InfoNCE loss w.r.t. `logit_scale` is proportional to `(positive-pair similarity) - (mean similarity over all pairs)`. Early in training, random embeddings put this near zero or negative -- the optimizer softens the temperature to minimize loss without learning discriminative features. Once embeddings actually separate positives from negatives, the gradient flips sign and the optimizer sharpens the temperature to exploit the newfound discrimination. A `logit_scale` that falls monotonically forever is a smoking gun that the model is not learning useful structure -- it's just going maximally uncertain. The bottom-and-climb-back pattern is therefore a training-time diagnostic, not an outcome.

### Operational lessons

Paid for these in real training incidents -- documented here so future runs avoid them:

1. **XLA fragmentation at stage transitions**. When a curriculum increases resolution (e.g. 112² -> 128²), stage-1's compiled kernels stay resident while stage-2 compiles its own, spiking VRAM during the transition window. This OOM'd three times on a 12 GB card between 160² / 128² / 112². **Fix**: set `--stage2-epochs 0` for single-stage training on 12 GB cards. The curriculum is nice-to-have, not load-bearing. The save-model -> `clear_session()` -> load-model trick would work for a true curriculum but is a non-trivial refactor.

2. **`logit_scale` in weight decay is a silent killer**. AdamW's decoupled weight decay will quietly drive the learnable temperature to zero, flattening the softmax and killing the contrastive signal regardless of how good the embeddings are. Verify the startup log line on every run.

3. **`.cache()` on the dataset is a RAM bomb at scale**. The default `cache_decoded=False` streams from disk. Enable `--cache-decoded` only for subsets under ~200 k samples (roughly 10-40 GB post-decode cache).

4. **I/O, not compute, bounds CC3M training**. At 32 random JPEG reads per step from a SATA-class disk with a dataset that doesn't fit in page cache, both RTX 4070 (12 GB) and RTX 4090 (24 GB) bottom out at ~540 ms/step with 30-40% GPU utilization. The 4090's extra FLOPs are unreachable through this pipeline. For repeat training runs on the same dataset, the high-throughput path is to convert JPEGs to TFRecord shards (see `train.common.tfrecord`) -- sequential reads from ~256 MiB shards typically recover 3-5× speedup. One-shot runs can live with the I/O cost.

### Serialization

`CliffordCLIP` has full `get_config` / `from_config` / `from_variant` round-trip serialization and is exported from `dl_techniques.models.cliffordnet`. The training script additionally writes:

```
results/cliffordclip_<variant>_<timestamp>/
  checkpoints/step_NNNNNNN.keras      # rolling window + final.keras
  retrieval_probes/probes.jsonl        # one line per probe
  tensorboard/<stage>/{train,validation}/
  training_log.csv                     # step-level loss, lr, logit_scale
  cliffordclip_<variant>.keras         # final model at run root
  training_summary.txt                 # final val R@K + hyperparameters
```

### Related utility: `train.common.tfrecord`

Generic TFRecord I/O for converting path-addressable datasets (per-sample JPEGs + metadata) to sharded TFRecord files. Not specific to CLIP -- any image-text or image-label dataset can use it. Pre-baked schemas (`IMAGE_TEXT_SCHEMA`, `IMAGE_LABEL_SCHEMA`), example builders, auto-sharded writer with sidecar manifest, and a companion `make_image_text_tf_dataset_from_tfrecord` that returns the same `{"image", "text"}` batches as the path-streaming loader so training code does not need to change to consume the faster format.

Use case: once a dataset exceeds page-cache size (roughly, raw bytes > available RAM), the random-access JPEG streaming pipeline becomes I/O bound and GPU utilization collapses. TFRecord shards convert that scatter-read into a small number of sequential reads per step.
