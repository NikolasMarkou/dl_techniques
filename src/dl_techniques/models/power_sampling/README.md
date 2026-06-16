# Power Sampling: Inference-Time Reasoning for Any Causal LLM/VLM

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-pure--inference-blue.svg)](https://numpy.org/)

**Power sampling** draws trajectories from the *power distribution* `p^alpha` (with `alpha = 1/temperature`) instead of the base distribution `p`, using an autoregressive **Metropolis-Hastings (MCMC) refinement loop** to improve *global* trajectory quality rather than just *local* per-token confidence. It is an inference-time method: no extra training, no reward model, no weights of its own. This package is fully **model-agnostic** (any callable LM/VLM behind an injected `logits_fn`) and **tokenizer-agnostic** (anything satisfying `TokenizerProtocol`).

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Problem: Local vs Global Sampling Quality](#2-the-problem-local-vs-global-sampling-quality)
3. [How It Works](#3-how-it-works)
4. [Algorithm Deep Dive](#4-algorithm-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration Reference](#7-configuration-reference)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Notes](#10-performance-notes)
11. [Method Selection (standard vs power vs max_swap)](#11-method-selection-standard-vs-power-vs-max_swap)
12. [Serialization & Statelessness](#12-serialization--statelessness)
13. [Troubleshooting & FAQs](#13-troubleshooting--faqs)
14. [Technical Details](#14-technical-details)
15. [Testing & Validation](#15-testing--validation)
16. [Citation](#16-citation)

---

## 1. Overview

### What is Power Sampling?

**Power sampling** is an inference-time decoding strategy that targets the *power distribution* `p^alpha` rather than the base model distribution `p`. Where low-temperature sampling sharpens probability mass at the *individual token* level, power sampling reweights *entire trajectories* to favor globally coherent sequences. The key insight from the literature is that reasoning capabilities usually attributed to RL post-training (GRPO, RLHF) may already exist in the base model's distribution — RL does not inject new ideas, it reshapes probability mass. Power sampling achieves a similar effect at inference time by amplifying high-probability trajectories through MCMC refinement.

This package is a refactor and generalization of the original CliffordNet implementation into a standalone, model- and tokenizer-agnostic engine.

### Key Features

1. **Model-agnostic** via an injected `logits_fn` closure (`make_logits_fn`): any callable LM/VLM returning logits works. CliffordNetLM + tiktoken is just one example.
2. **Tokenizer-agnostic** via `TokenizerProtocol`: anything exposing `encode`/`decode` drives the sampler. No `import tiktoken` at module load.
3. **Batched-parallel MCMC**: all MCMC proposals within a block run through the model in a single batched forward pass for high GPU utilization.
4. **VLM-aware adapter** (`VLMForwardAdapter`): binds a fixed image and a `text_slice_start` offset so the engine can drive the text suffix while the image prefix stays fixed.
5. **Three generation methods**: `standard` (nucleus sampling baseline), `power` (MCMC over `p^alpha`), and `max_swap` (deterministic greedy trajectory optimization, approximating `p^infinity`).
6. **Pure-NumPy post-logit pipeline**: forward passes run on the model's device; sampling math uses NumPy on CPU. No `tf.gather_nd`, no Keras graph retraces.

### Power Sampling vs Temperature Sampling

| Aspect | Low-temperature sampling | Power sampling (`p^alpha`) |
|:---|:---|:---|
| Sharpens at | Local (per-token) confidence | Global (whole-trajectory) probability |
| Question answered | "How likely is this token?" | "How good are the futures this token leads to?" |
| Mechanism | Scale logits by `1/T` once | MCMC refinement over trajectories drawn at `alpha = 1/T` |
| Diversity cost | Collapses to greedy at low `T` | Preserves diversity via stochastic acceptance |
| Cost | 1x forward passes | ~`block_num * (1 + mcmc_steps)` proposal generations |

---

## 2. The Problem: Local vs Global Sampling Quality

Standard autoregressive decoding samples one token at a time from `p`. Low-temperature sampling sharpens the distribution, but only at the *local* (per-token) level. This amplifies shortcuts: guessing instead of planning, premature answers, and locally plausible but globally poor trajectories.

```
┌─────────────────────────────────────────────────────────────┐
│  Low-temperature sampling (per-token sharpening)            │
│                                                             │
│  p_T(x_t | x_<t)  ∝  p(x_t | x_<t)^(1/T)                    │
│                                                             │
│  Each token is chosen to be locally most likely.           │
│  Asks:  "How likely is THIS token, right now?"             │
│                                                             │
│  Failure mode: a confident-but-wrong early token greedily   │
│  commits the model to a globally poor trajectory.           │
└─────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────┐
│  Power sampling (trajectory-level sharpening)               │
│                                                             │
│  p_alpha(x_1..x_n)  ∝  p(x_1..x_n)^alpha,  alpha = 1/T      │
│                                                             │
│  WHOLE sequences are reweighted; an MCMC loop swaps in      │
│  re-generated suffixes that raise trajectory probability.   │
│  Asks:  "If I pick this token, how GOOD are the futures?"  │
│                                                             │
│  Diversity is preserved because acceptance is stochastic    │
│  (Metropolis-Hastings), not greedy.                         │
└─────────────────────────────────────────────────────────────┘
```

Pushing temperature toward 0 to chase quality collapses generation toward greedy decoding, destroying diversity and exploration. Power sampling instead keeps a proposal temperature in a usable range and lets the **MCMC acceptance rule** do the trajectory-level sharpening, so high `alpha` (low `T`) improves coherence without zeroing out diversity.

---

## 3. How It Works

The engine generates one block of tokens at a time. After each block it samples `mcmc_steps` proposals — each a re-generation from a random cut point to the end — runs them all through the model in a single batched forward pass, and accepts or rejects each via Metropolis-Hastings.

```
┌────────────────────────────────────────────────────────────────────┐
│                    Power Sampling Generation Loop                   │
│                                                                    │
│  prompt ──► tokenize ──► [optional CLS prepend] ──► gen = prompt    │
│                                                       │            │
│   ┌───────────────────────────────────────────────────┘            │
│   ▼                                                                 │
│  for block_idx in range(block_num):                                 │
│                                                                    │
│   ┌──────────────────────────┐                                     │
│   │ naive_temp_generate      │  append jump_size tokens at temp T   │
│   │  (proposal distribution) │  record proposal & target log-probs  │
│   └────────────┬─────────────┘                                     │
│                ▼                                                     │
│   ┌──────────────────────────┐                                     │
│   │ sample K=mcmc_steps cut   │  idx_k = randint(c, t-1)            │
│   │ points; re-generate each  │  prefixes = gen[:idx_k]             │
│   │ to the end IN PARALLEL    │  ── ONE batched forward per step ── │
│   │  (_batched_generate)      │                                     │
│   └────────────┬─────────────┘                                     │
│                ▼                                                     │
│   ┌──────────────────────────┐    accept w.p. min(1, e^{log r})    │
│   │ Metropolis-Hastings       │    (max_swap: accept iff log r > 0) │
│   │ accept / reject each       │ ─► on accept: gen = proposal,      │
│   │ proposal                  │    splice in its log-probs          │
│   └──────────────────────────┘                                     │
│                                                                    │
│  return gen[strip:]  (strip the CLS prefix only if one was added)  │
└────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Tokenize** the prompt via `tokenizer.encode(prompt)`. If `cls_token_id is not None`, prepend it and set `strip = 1`; otherwise `strip = 0`. The context boundary `c = len(prompt_ids)`.
2. **Per block**, call `naive_temp_generate` to append `jump_size = max_tokens // block_num` tokens at the proposal temperature, recording each token's proposal log-prob and target (power) log-prob.
3. **Sample `mcmc_steps` cut points** uniformly in `[c, t-1]`; build the prefixes `gen[:idx]` and per-proposal token counts `t - idx`.
4. **Re-generate all proposals in parallel** via `_batched_generate` — one batched forward pass per generation step across the whole proposal batch.
5. **Metropolis-Hastings acceptance** for each proposal; on accept, replace `gen` with the proposal and splice in the new log-probs.
6. **Repeat** for every block, then return the generated tokens with the CLS prefix stripped only when it was prepended.

---

## 4. Algorithm Deep Dive

### 4.1 The Power Distribution

Power sampling targets the power distribution, the base distribution raised to the exponent `alpha`:

$$p_\alpha(x) \propto p(x)^{\alpha}, \quad \alpha = \frac{1}{T}$$

With `T < 1` (so `alpha > 1`) the distribution is sharpened at the *trajectory* level. The "target" log-probability of a generated token under this distribution is `(1/T) * log p(token)`, which the engine accumulates as `log_probs_unnorm`. The "proposal" log-probability is the temperature-scaled + nucleus distribution actually sampled from, accumulated as `log_probs_norm`.

### 4.2 Metropolis-Hastings Acceptance (`mcmc_power_sample`)

For each proposal, with `target` log-probs from `p^alpha` and `proposal` log-probs from the actual sampling distribution, the log-acceptance ratio over the re-generated suffix is:

$$\log r = \sum \text{target}_{\text{prop}} + \sum \text{proposal}_{\text{cur}} - \sum \text{target}_{\text{cur}} - \sum \text{proposal}_{\text{prop}}$$

The proposal is accepted with probability `min(1, e^{log r})`:

$$P(\text{accept}) = \min\!\left(1,\ e^{\log r}\right)$$

Concretely the engine draws `u ~ Uniform(0,1)` and accepts iff `u < exp(min(log r, 0))`. The `target` terms are the power-distribution log-probs `(1/T) log p`, the `proposal` terms are the temperature+nucleus log-probs; the cross-cancellation of forward/backward proposal densities is exactly the standard MH correction adapted for autoregressive re-generation.

### 4.3 Max-Swap (`max_swap`, approximating `p^infinity`)

The `max_swap` variant is the deterministic, greedy limit. It accepts a proposal **iff the trajectory log-probability strictly improves**:

$$\log r = \sum \text{target}_{\text{prop}} - \sum \text{target}_{\text{cur}}, \quad \text{accept} \iff \log r > 0$$

This is hill-climbing over trajectory probability and approximates sampling from `p^infinity` (always pick the most probable trajectory among the proposals seen). It maximizes coherence at the cost of diversity.

### 4.4 Sign-Aware Repetition Penalty + Nucleus Proposal

Every token is drawn through `_sample_token`, which builds the proposal distribution in this order:

1. **Special-token masking**: for each `sid in special_token_ids` (guarded by `sid < vocab_size`), set the logit to `-1e9` so those tokens are never generated.
2. **Sign-aware repetition penalty**: for each recently used token within `repetition_window`, divide the logit by `repetition_penalty` when it is `>= 0`, and *multiply* by `repetition_penalty` when it is `< 0`. This sign-awareness ensures the penalty always reduces a token's probability regardless of its logit sign (a naive divide would *increase* a negative logit).
3. **Temperature scaling**: divide by the temperature.
4. **Nucleus (top-p) sampling**: keep the smallest set of tokens whose cumulative probability reaches `top_p`, renormalize, and sample.

The proposal log-prob is read from the scaled (post-temperature) log-softmax; the target log-prob is the base-model log-softmax divided by temperature.

---

## 5. Quick Start Guide

### Installation

Power sampling ships as part of `dl_techniques` — no separate install. It depends only on NumPy at module load (no top-level TensorFlow or `tiktoken` import). `tiktoken` is optional and only needed if you choose it as your tokenizer.

### Your First PowerSampler (30 seconds)

Build a sampler over any callable model + any tokenizer and call `generate_text`.

```python
from dl_techniques.models.power_sampling import (
    PowerSampler, PowerSamplingConfig,
)

model = build_my_causal_lm()      # any callable returning {"logits": float32[B,T,V]}
tokenizer = get_my_tokenizer()    # any object with encode(str)->List[int], decode(List[int])->str

config = PowerSamplingConfig(
    temperature=0.25,             # alpha = 1/0.25 = 4
    mcmc_steps=10,
    block_num=8,
    max_tokens=100,
)
sampler = PowerSampler(model, tokenizer, config)

# String-in / string-out, MCMC power sampling
text, info = sampler.generate_text("The theory of relativity states that", method="power")
print(text)
print(f"acceptance ratio: {info['acceptance_ratio']:.1%}  |  time: {info['elapsed_s']:.1f}s")
```

---

## 6. Component Reference

### 6.1 `PowerSampler`

The main engine. Forward passes run on the model's device; post-logit sampling runs in NumPy on CPU.

**Constructor**:

```python
PowerSampler(
    model_or_logits_fn,                    # a callable model OR a pre-built LogitsFn
    tokenizer,                             # any TokenizerProtocol (encode/decode)
    config=None,                           # PowerSamplingConfig (defaults if None)
    *,
    logits_fn=None,                        # explicit LogitsFn override (unambiguous path)
)
```

If `logits_fn` is omitted, `model_or_logits_fn` is wrapped automatically into single- and batched-logits closures via `make_logits_fn` / `make_batch_logits_fn` using `config.ctx_len` and `config.pad_token_id`. To inject a pre-built closure unambiguously (e.g. from a VLM adapter), pass it via the `logits_fn=` keyword.

**Generation methods**:

| Method | Signature (key args) | Returns |
|:---|:---|:---|
| `generate_standard(prompt, temperature=0.85, top_p=0.92, max_tokens=100, repetition_penalty=1.3)` | baseline nucleus sampling | `(token_ids: List[int], info: Dict)` |
| `mcmc_power_sample(prompt, temperature=None, mcmc_steps=None, max_tokens=None, block_num=None)` | MCMC over `p^alpha` | `(token_ids: List[int], info: Dict)` |
| `max_swap(prompt, temperature=None, mcmc_steps=None, max_tokens=None, block_num=None)` | deterministic trajectory optimization | `(token_ids: List[int], info: Dict)` |
| `generate_text(prompt, method="power", **kwargs)` | dispatch + decode | `(text: str, info: Dict)` |

`info` always contains timing; the power/max_swap methods additionally report `acceptance_ratio` (always in `[0, 1]`), `total_steps`, `acceptances`. `generate_text` accepts `method` in `{"standard", "power", "max_swap"}` and raises `ValueError` on any other value.

### 6.2 `PowerSamplingConfig`

A plain `@dataclass` (not a Keras component — no serialization machinery). Fields:

- **Algorithm hyperparameters**: `temperature`, `mcmc_steps`, `block_num`, `max_tokens`, `top_p`, `repetition_penalty`, `repetition_window`.
- **Tokenizer/model identity (generalized, no GPT-2 defaults)**: `special_token_ids` (empty set), `cls_token_id` (`None`), `pad_token_id` (`None`), `ctx_len` (`None`).

See [Configuration Reference](#7-configuration-reference) for defaults and meanings.

### 6.3 `make_logits_fn` / `make_batch_logits_fn`

Factories that turn any callable model into a `LogitsFn` closure (single position) or a batched closure.

```python
make_logits_fn(
    model,
    ctx_len=None,             # None => variable-length forward; int => right-pad to ctx_len
    pad_id=None,              # REQUIRED when ctx_len is set
    logits_key="logits",      # dict key for the logits tensor; None => model returns a bare tensor
    position=-1,              # -1 => last REAL token; else a fixed position
    text_slice_start=0,       # offset added to the gather index (skip vision-token prefix for VLMs)
    extra_inputs=None,        # dict of extra model inputs, e.g. {"images": img}; triggers dict-input call
    token_key="text_tokens",  # dict key for the token array when extra_inputs is set
) -> LogitsFn                 # maps token ids -> float32[V]

make_batch_logits_fn(...)     # same kwargs; maps List[List[int]] -> float32[B, V]
```

Key behaviors:
- `logits_key=None` makes the closure treat the model output as a bare logits tensor (plain-tensor models).
- `ctx_len` set => sequences are right-padded to a fixed length with `pad_id` (fixed-shape models such as CliffordNetLM); `ctx_len=None` => variable-length forward, no padding.
- `text_slice_start` shifts the gather index by the vision-token prefix length for VLMs.
- The last-token gather is **pure NumPy** indexing on the host-side array — there is no `tf.gather_nd` and no top-level `import tensorflow` anywhere in the package.

### 6.4 `VLMForwardAdapter`

Bridges a dict-input vision-language model to the `LogitsFn` interface by binding a fixed image and a `text_slice_start` offset.

```python
VLMForwardAdapter(
    model, image,
    *,
    image_key="images",
    token_key="text_tokens",
    text_slice_start,          # vision sequence length — CALLER'S responsibility
    logits_key="logits",
    ctx_len=None,
    pad_id=None,
)
# .as_logits_fn()        -> single-sequence LogitsFn bound to the fixed image
# .as_batch_logits_fn()  -> batched closure bound to the fixed image
```

`text_slice_start` (== vision sequence length) is supplied by the caller; repo VLMs (e.g. `nano_vlm`) do not expose it as a property, and auto-deriving it is out of scope.

### 6.5 `TokenizerProtocol`

A `runtime_checkable` `typing.Protocol` capturing the only two methods the sampler ever calls:

```python
class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...
```

Vocabulary metadata (`vocab_size`, special-token ids) is intentionally **not** part of this contract — it lives in `PowerSamplingConfig` so tokenizers lacking those attributes still satisfy the protocol. `tiktoken.Encoding`, an HF `AutoTokenizer`, a SentencePiece wrapper, or a hand-rolled char tokenizer all qualify with no inheritance.

---

## 7. Configuration Reference

| Field | Default | Meaning |
|:---|:---|:---|
| `temperature` | `0.25` | Proposal temperature; the power exponent is `alpha = 1/temperature` (0.25 → alpha=4). |
| `mcmc_steps` | `10` | Number of Metropolis-Hastings proposals generated (and evaluated) per block. |
| `block_num` | `16` | Number of generation blocks; MCMC refinement runs after each block. |
| `max_tokens` | `512` | Maximum tokens to generate (split into `block_num` chunks of `max_tokens // block_num`). |
| `top_p` | `0.92` | Nucleus (top-p) cumulative-probability threshold for the proposal distribution. |
| `repetition_penalty` | `1.3` | Sign-aware penalty applied to recently generated tokens. |
| `repetition_window` | `50` | Number of recent tokens considered for the repetition penalty. |
| `special_token_ids` | `set()` | Token ids masked to `-1e9` (never generated). Empty by default — supply your model's ids. |
| `cls_token_id` | `None` | Token prepended to every prompt. `None` => no prepend and nothing stripped from the output. |
| `pad_token_id` | `None` | Right-padding token id. Required when `ctx_len` is set; otherwise unused. |
| `ctx_len` | `None` | Fixed context length for fixed-shape models. `None` => variable-length forward, no padding. |

---

## 8. Comprehensive Usage Examples

### Example 1: GPT-2 via `tiktoken`

A GPT-2-like model returning `{"logits": float32[B, T, V]}`, driven by the `tiktoken` GPT-2 encoding. GPT-2 needs no CLS prepend; supply `pad_token_id` / `ctx_len` only if your model is fixed-shape.

```python
import tiktoken
from dl_techniques.models.power_sampling import PowerSampler, PowerSamplingConfig

enc = tiktoken.get_encoding("gpt2")     # satisfies TokenizerProtocol (encode/decode)
model = load_gpt2_like_model()          # callable returning {"logits": float32[B,T,V]}

config = PowerSamplingConfig(
    temperature=0.25,
    mcmc_steps=10,
    block_num=8,
    max_tokens=128,
    pad_token_id=enc.eot_token,         # only needed for fixed-shape forward
    ctx_len=1024,                       # fixed context window; omit for variable-length
)
sampler = PowerSampler(model, enc, config)

text, info = sampler.generate_text("In mathematics, a prime number is", method="power")
print(text)
```

### Example 2: Generic tokenizer via `TokenizerProtocol`

Any object exposing `encode`/`decode` works — here a thin wrapper around a Hugging Face `AutoTokenizer` (or SentencePiece). No inheritance, no registration.

```python
from transformers import AutoTokenizer
from dl_techniques.models.power_sampling import PowerSampler, PowerSamplingConfig

class HFTokenizerAdapter:
    """Wrap an HF tokenizer to satisfy TokenizerProtocol."""
    def __init__(self, name: str):
        self._tok = AutoTokenizer.from_pretrained(name)

    def encode(self, text: str):
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=True)

tokenizer = HFTokenizerAdapter("gpt2")
model = load_my_lm()                    # callable returning {"logits": ...}

config = PowerSamplingConfig(temperature=0.3, mcmc_steps=8, block_num=8, max_tokens=120)
sampler = PowerSampler(model, tokenizer, config)

text, info = sampler.generate_text("Once upon a time", method="max_swap")
print(text)
```

### Example 3: VLM via `VLMForwardAdapter`

For a dict-input vision-language model, bind a fixed image and the vision-token offset. The text suffix is sampled while the image prefix stays fixed. Pass the resulting closure via the `logits_fn=` keyword.

```python
import numpy as np
from dl_techniques.models.power_sampling import (
    PowerSampler, PowerSamplingConfig, VLMForwardAdapter,
)

vlm = load_my_vlm()                     # callable on {"images": img, "text_tokens": ids}
image = load_image_tensor()             # fixed image, e.g. float32[1, H, W, 3]
vision_seq_len = 196                    # CALLER-supplied vision-token count

adapter = VLMForwardAdapter(
    vlm, image,
    image_key="images",
    token_key="text_tokens",
    text_slice_start=vision_seq_len,    # gather index is offset past the vision prefix
    logits_key="logits",
)
logits_fn = adapter.as_logits_fn()

tokenizer = get_my_tokenizer()
config = PowerSamplingConfig(temperature=0.25, mcmc_steps=6, block_num=6, max_tokens=64)
sampler = PowerSampler(vlm, tokenizer, config, logits_fn=logits_fn)

text, info = sampler.generate_text("Describe the image:", method="power")
print(text)
```

> Caveat: `text_slice_start` (the vision sequence length) is the caller's responsibility. A wrong value gathers logits from inside the vision prefix and produces garbage — see [Troubleshooting](#13-troubleshooting--faqs).

### Example 4: CliffordNet (the original use case)

CliffordNetLM is a fixed-shape model with GPT-2-style special tokens. Behavior is preserved by passing the CliffordNet ids explicitly (the package no longer hard-codes them).

```python
import tiktoken
from dl_techniques.models.cliffordnet.model import CliffordNetLM
from dl_techniques.models.power_sampling import PowerSampler, PowerSamplingConfig

model = CliffordNetLM.from_variant("base")     # or keras.models.load_model(...)
enc = tiktoken.get_encoding("gpt2")

config = PowerSamplingConfig(
    temperature=0.25,
    mcmc_steps=10,
    block_num=8,
    max_tokens=200,
    cls_token_id=50257,
    pad_token_id=50260,
    special_token_ids={50257, 50258, 50259, 50260},
    ctx_len=511,                               # fixed-shape: pad to 511
)
sampler = PowerSampler(model, enc, config=config)

for method in ["standard", "power", "max_swap"]:
    text, info = sampler.generate_text("The theory of relativity states that", method=method)
    print(f"\n--- {method} ---\n{text[:200]}")
```

---

## 9. Advanced Usage Patterns

### Plain-tensor-output models (`logits_key=None`)

If your model returns a bare logits tensor instead of a dict, build the closure with `logits_key=None` and inject it explicitly:

```python
from dl_techniques.models.power_sampling import (
    PowerSampler, PowerSamplingConfig, make_logits_fn,
)

logits_fn = make_logits_fn(model, logits_key=None)   # out IS the logits tensor
sampler = PowerSampler(model, tokenizer, PowerSamplingConfig(), logits_fn=logits_fn)
```

### Variable-length forward (`ctx_len=None`)

Leaving `ctx_len=None` (the default) runs an unpadded variable-length forward pass — appropriate for models that accept dynamic sequence lengths. No `pad_token_id` is required in this mode.

### Supplying a pre-built `logits_fn`

The `logits_fn=` keyword is the unambiguous path for any custom closure (VLM adapter, custom batching, a remote model proxy). When supplied, the batched path falls back to looping the single-position closure.

### Tuning `mcmc_steps` / `block_num`

More `block_num` => more frequent refinement (finer granularity, more forward passes). More `mcmc_steps` => more proposals per block (higher acceptance opportunity, more compute). Start from `block_num=8, mcmc_steps=10` and scale down for speed.

---

## 10. Performance Notes

- **Batched MCMC proposals**: all `mcmc_steps` proposals in a block are re-generated in parallel via `_batched_generate`, which issues a single batched forward pass per generation step. A single active sequence short-circuits to the unbatched path to avoid batch overhead.
- **GPU forward / CPU NumPy split**: the model forward runs on whatever device the model uses; everything after the logits (`_log_softmax`, special-token masking, repetition penalty, nucleus sampling, the MH acceptance test) is pure NumPy on CPU. This keeps the sampling math out of the compiled graph and avoids retraces.
- **No KV cache**: each forward pass recomputes all positions, so power sampling is designed for offline/batch inference, not real-time streaming.
- **Cost model**: roughly `block_num * (jump_size + sum of mcmc_steps proposal lengths)` token-forwards. The `power` and `max_swap` methods cost on the order of ~`(1 + mcmc_steps)`x the token-forwards of the `standard` baseline.

---

## 11. Method Selection (standard vs power vs max_swap)

| Method | Samples from | Relative cost | When to use |
|:---|:---|:---|:---|
| `standard` | `p` (nucleus sampling) | 1x | Fast baseline; A/B comparison; when latency matters more than coherence. |
| `power` | `p^alpha` (MCMC) | ~`(1 + mcmc_steps)`x | Best general-purpose trajectory coherence while preserving diversity. |
| `max_swap` | `p^infinity` (deterministic) | ~`(1 + mcmc_steps)`x | Maximum trajectory probability (greedy hill-climb); use when you want the single most coherent continuation and accept lower diversity. |

Rule of thumb: default to `power`; drop to `standard` for speed; reach for `max_swap` when you want the most-probable trajectory and diversity is not a concern.

---

## 12. Serialization & Statelessness

There is nothing to serialize. `PowerSampler` is a pure-Python inference engine: it holds references to a model, a tokenizer, and a `@dataclass` config, but it owns **no weights** and is **not** a `keras.Model` — hence no `@keras.saving.register_keras_serializable`, no `build`, no `get_config`. The model and tokenizer are serialized through their own mechanisms; the sampler is reconstructed by simply re-instantiating it. Critically, `generate_standard` does **not** mutate `self.config`: per-call `top_p`/`repetition_penalty` overrides are applied through a transient `dataclasses.replace` copy that is restored in a `finally` block, so config fields are identical before and after a call (exception-safe).

---

## 13. Troubleshooting & FAQs

**Q: My decoded text has a leading space (or odd spacing).**

A: This is a tokenizer artifact, not a sampling bug. Many BPE tokenizers (GPT-2/HF) encode a leading space into the first token, so `decode` may return a leading space. The sampler only calls `encode`/`decode` and never alters the byte stream. Strip/trim on your side if needed, or use a tokenizer wrapper that handles it.

**Q: The acceptance ratio is near 0 — nothing is being accepted.**

A: The proposal and target distributions are too far apart. A very low `temperature` (very high `alpha`) makes the target distribution extremely peaked, so almost no stochastic proposal clears the MH test. Raise `temperature` (lower `alpha`), reduce `mcmc_steps`, or use `max_swap` if you specifically want deterministic acceptance.

**Q: `ValueError: logits_key='logits' not found in model output`.**

A: Your model does not return a dict keyed `"logits"`. If it returns a bare tensor, build the closure with `logits_key=None` (see [Advanced Usage](#9-advanced-usage-patterns)); if it uses a different key, pass that key to `make_logits_fn(..., logits_key="your_key")`. The error message lists the available keys.

**Q: My VLM output is garbage.**

A: Almost always a wrong `text_slice_start`. It must equal the vision-token sequence length so the gather index lands on the first text position, not inside the image prefix. The package cannot auto-derive it for arbitrary VLMs — verify it against your model's vision encoder output length.

---

## 14. Technical Details

- **Pure-NumPy post-logit pipeline.** Forward passes return `.numpy()`-able eager tensors; `_to_numpy` coerces them and all downstream math is NumPy. The last-token gather is NumPy fancy indexing (`logits[np.arange(B), idx]`), deliberately replacing the original `tf.gather_nd` so the package carries no top-level TensorFlow import.
- **Generalized identity.** The config defaults carry no GPT-2/CliffordNet ids (`special_token_ids=set()`, `cls_token_id=None`, `pad_token_id=None`, `ctx_len=None`). Model-specific behavior is opt-in at the call site — restoring the old hard-coded ids would silently mis-mask any other model.
- **CLS handling.** A CLS token is prepended only when `cls_token_id is not None`, and stripped from the returned token ids only when it was actually prepended. The context boundary `c` is set accordingly so MCMC cut points never re-sample the prompt.
- **No `self.config` mutation.** Per-call overrides in `generate_standard` use a transient `dataclasses.replace` copy restored in `finally` (exception-safe), so concurrent or repeated calls never observe a mutated config.

---

## 15. Testing & Validation

The engine is exercised by a keras-free mock suite (a mock model returning `{"logits": ...}` plus a char-level mock tokenizer) covering: config defaults; `_log_softmax`/`_nucleus_sample` numerics; `make_logits_fn` for dict / plain-tensor / VLM-offset outputs; `generate_standard`; `mcmc_power_sample` (acceptance ratio in `[0, 1]`, output length); `max_swap`; `generate_text` dispatch + `ValueError` on a bad method; CLS-prepend on/off; and the no-config-mutation invariant.

Run the scoped suite (no GPU required):

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 \
    .venv/bin/python -m pytest tests/test_models/test_power_sampling/ -vvv
```

---

## 16. Citation

This package adapts the autoregressive MCMC power-sampling approach from:

```bibtex
@article{karan2025reasoning,
  title   = {Reasoning with Sampling: Your Base Model is Smarter Than You Think},
  author  = {Karan, Aayush and Du, Yilun},
  journal = {arXiv preprint arXiv:2510.14901},
  year    = {2025}
}

@article{bouammar2026scalable,
  title   = {Scalable Power Sampling for LLM Reasoning},
  author  = {Bou Ammar, Haitham and others},
  journal = {arXiv preprint arXiv:2601.21590},
  year    = {2026}
}
```

- Karan, A. & Du, Y. (2025). *Reasoning with Sampling: Your Base Model is Smarter Than You Think*. arXiv:2510.14901.
- Bou Ammar, H. et al. (2026). *Scalable Power Sampling for LLM Reasoning*. arXiv:2601.21590.

---
