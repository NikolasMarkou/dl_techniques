# Power Sampling: Inference-Time Reasoning for Any Causal LLM/VLM

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-pure--inference-blue.svg)](https://numpy.org/)

**Power sampling** draws trajectories from the *power distribution* `p^alpha` (`alpha = 1/temperature`) instead of the base distribution `p`, using an autoregressive **Metropolis-Hastings refinement loop** to improve *global* trajectory quality rather than *local* per-token confidence.

It is an inference-time method: no extra training, no reward model, no weights of its own. **There is no model class in this package and no `create_*` factory** — nothing here subclasses `keras.Model`. It is model-agnostic (any callable LM/VLM behind an injected `logits_fn`) and tokenizer-agnostic (anything satisfying `TokenizerProtocol`).

---

## 1. Overview

Where low-temperature sampling sharpens probability mass at the *individual token* level, power sampling reweights *entire trajectories* to favor globally coherent sequences. The insight from the literature is that reasoning capabilities usually attributed to RL post-training may already exist in the base model's distribution — RL does not inject new ideas, it reshapes probability mass. Power sampling achieves a similar effect at inference time by amplifying high-probability trajectories through MCMC refinement.

| Aspect | Low-temperature sampling | Power sampling (`p^alpha`) |
|:---|:---|:---|
| Sharpens at | Local (per-token) confidence | Global (whole-trajectory) probability |
| Question answered | "How likely is this token?" | "How good are the futures this token leads to?" |
| Mechanism | Scale logits by `1/T` once | MCMC refinement over trajectories drawn at `alpha = 1/T` |
| Diversity cost | Collapses to greedy at low `T` | Preserved via stochastic acceptance |
| Cost | 1x forward passes | ~`(1 + mcmc_steps)`x |

### Key features

1. **Model-agnostic** via an injected `logits_fn` closure (`make_logits_fn`): any callable LM/VLM returning logits works.
2. **Tokenizer-agnostic** via `TokenizerProtocol` — `encode`/`decode` and nothing else. No `import tiktoken` at module load.
3. **Batched-parallel MCMC without breaking the chain** (§3).
4. **VLM-aware adapter** (`VLMForwardAdapter`): binds a fixed image and a `text_slice_start` offset so the engine drives the text suffix while the image prefix stays fixed.
5. **Three generation methods**: `standard`, `power`, `max_swap`.
6. **Pure-NumPy post-logit pipeline**: forward passes run on the model's device; all sampling math is NumPy on CPU. No `tf.gather_nd`, no Keras graph retraces, no top-level TensorFlow import.

---

## 2. The Problem: Local vs Global Sampling Quality

Standard autoregressive decoding samples one token at a time from `p`. Low-temperature sampling sharpens the distribution, but only *locally*:

```
p_T(x_t | x_<t)  ∝  p(x_t | x_<t)^(1/T)

Asks: "How likely is THIS token, right now?"
Failure: a confident-but-wrong early token greedily commits the model
         to a globally poor trajectory.
```

Pushing `T` toward 0 to chase quality collapses generation toward greedy decoding, destroying diversity and exploration. Power sampling instead reweights whole sequences:

```
p_alpha(x_1..x_n)  ∝  p(x_1..x_n)^alpha,  alpha = 1/T

Asks: "If I pick this token, how GOOD are the futures?"
```

It keeps the proposal temperature in a usable range and lets the **MH acceptance rule** do the trajectory-level sharpening, so high `alpha` improves coherence without zeroing out diversity.

---

## 3. How It Works

The engine generates one block of tokens at a time. After each block it samples `mcmc_steps` cut points, re-generates from each cut point to the end, and accepts or rejects each via Metropolis-Hastings.

```
prompt ──► tokenize ──► [optional CLS prepend] ──► gen = prompt

for each block:
   naive_temp_generate  ──► append this block's tokens at temperature T,
                            recording proposal and target log-probs
   sample K = mcmc_steps cut points, ONCE per block:  idx_k ~ U[c, t-1]
   batch the proposals STILL OWED, all cut from the CURRENT gen
                            ── one batched forward per generation step ──
   Metropolis-Hastings, in order:
       reject ──► state unchanged; consume the next queued proposal
       accept ──► gen = proposal, splice its log-probs,
                  DISCARD the rest of the batch and re-generate

return gen[strip:]     (strip the CLS prefix only if one was added)
```

**Why the batch is discarded on acceptance.** Batching spans a run of *rejections* only: a rejection leaves the chain state unchanged, so proposals queued behind it are still valid draws from `q(.|x)`. An *acceptance* moves the state, so everything queued behind it would have been drawn from a state the chain never occupied. Expected batched forward rounds per block: `1 + (number of acceptances)` — one at 0% acceptance, `mcmc_steps` at 100%.

Cut points are pre-drawn because `idx ~ Uniform[c, t-1]` does not depend on the chain state. The *continuations* do, and are not pre-drawn across an acceptance.

Per-block token counts come from `_block_sizes(max_tokens, block_num)`, which distributes the remainder so every block gets at least one token and they sum exactly to `max_tokens` — `_block_sizes(10, 4)` is `[3, 3, 2, 2]`, not four blocks of 2. `block_num > max_tokens` is clamped down with a warning.

---

## 4. Algorithm Deep Dive

### 4.1 The power distribution

$$p_\alpha(x) \propto p(x)^{\alpha}, \quad \alpha = \frac{1}{T}$$

With `T < 1` the distribution is sharpened at the *trajectory* level. The **target** log-probability of a generated token is `(1/T) * log p(token)`, accumulated as `log_probs_unnorm`. The **proposal** log-probability is the temperature-scaled + nucleus distribution actually sampled from, accumulated as `log_probs_norm`.

### 4.2 Metropolis-Hastings acceptance (`mcmc_power_sample`)

Over the re-generated suffix:

$$\log r = \sum \text{target}_{\text{prop}} + \sum \text{proposal}_{\text{cur}} - \sum \text{target}_{\text{cur}} - \sum \text{proposal}_{\text{prop}}$$

The engine draws `u ~ Uniform(0,1)` and accepts iff `u < exp(min(log r, 0))`, i.e. with probability `min(1, e^{log r})`. The cross-cancellation of forward and backward proposal densities is exactly the standard MH correction, adapted for autoregressive re-generation.

### 4.3 Max-swap (`max_swap`, approximating `p^infinity`)

The deterministic, greedy limit. Accepts a proposal **iff the trajectory log-probability strictly improves**:

$$\log r = \sum \text{target}_{\text{prop}} - \sum \text{target}_{\text{cur}}, \quad \text{accept} \iff \log r > 0$$

Hill-climbing over trajectory probability. Maximum coherence, minimum diversity.

### 4.4 The proposal distribution

Every token is drawn through `_sample_token`, which builds the proposal in this order:

1. **Special-token masking**: for each `sid in special_token_ids` (guarded by `sid < vocab_size`), set the logit to `-1e9`.
2. **Sign-aware repetition penalty**: for each recently used token within `repetition_window`, *divide* the logit by `repetition_penalty` when it is `>= 0` and *multiply* when it is `< 0`. The sign-awareness matters: a naive divide would *increase* a negative logit.
3. **Temperature scaling**: divide by the temperature.
4. **Nucleus (top-p)**: keep the smallest set whose cumulative probability reaches `top_p`, renormalize, sample.

**The proposal log-prob is returned by the nucleus draw itself** — the log probability of the drawn token under the **truncated, renormalized** distribution of step 4, the distribution the token actually came from. It is deliberately *not* the post-temperature full-vocabulary log-softmax: those agree only at `top_p = 1.0`, and this is the `q(x|x')/q(x'|x)` factor of the MH ratio, so reading the untruncated one biases acceptance whenever top-p excludes real mass (measured at `top_p = 0.5` over a linear logit ramp: **0.602 nats apart**).

The **target** log-prob is the base-model log-softmax divided by temperature — base, i.e. before masking, penalty and truncation, because it must describe `p^alpha` and not the proposal.

---

## 5. Quick Start Guide

Ships as part of `dl_techniques` — no separate install, and only NumPy is imported at module load.

```python
from dl_techniques.models.common.power_sampling import (
    PowerSampler, PowerSamplingConfig,
)

model = build_my_causal_lm()      # any callable returning {"logits": float32[B,T,V]}
tokenizer = get_my_tokenizer()    # any object with encode(str)->List[int], decode(List[int])->str

config = PowerSamplingConfig(
    temperature=0.25,             # alpha = 4
    mcmc_steps=10,
    block_num=8,
    max_tokens=100,
    pad_token_id=0,               # REQUIRED at mcmc_steps >= 2 — see §7
)
sampler = PowerSampler(model, tokenizer, config)

text, info = sampler.generate_text("The theory of relativity states that", method="power")
print(text)
print(f"acceptance: {info['acceptance_ratio']:.1%} | alpha: {info['alpha']} | {info['elapsed_s']:.1f}s")
```

---

## 6. Component Reference

### 6.1 `PowerSampler`

```python
PowerSampler(
    model_or_logits_fn,   # a callable model OR a pre-built LogitsFn
    tokenizer,            # any TokenizerProtocol
    config=None,          # PowerSamplingConfig (defaults if None)
    *,
    logits_fn=None,       # explicit LogitsFn override — the unambiguous path
)
```

With `logits_fn` omitted, `model_or_logits_fn` is wrapped automatically into single- and batched-logits closures via `make_logits_fn` / `make_batch_logits_fn` using `config.ctx_len` and `config.pad_token_id`. To inject a pre-built closure (e.g. from a VLM adapter), pass it as `logits_fn=`; the batched path then falls back to looping the single-position closure.

| Method | Returns | `info` keys |
|:---|:---|:---|
| `generate_standard(prompt, temperature=0.85, top_p=0.92, max_tokens=100, repetition_penalty=1.3)` | `(List[int], Dict)` | `elapsed_s`, `tokens_generated`, `tok_per_s` |
| `mcmc_power_sample(prompt, temperature=None, mcmc_steps=None, max_tokens=None, block_num=None)` | `(List[int], Dict)` | `acceptance_ratio`, `acceptances`, `total_steps`, `elapsed_s`, `alpha` |
| `max_swap(prompt, ... same overrides ...)` | `(List[int], Dict)` | same, minus `alpha` |
| `generate_text(prompt, method="power", **kwargs)` | `(str, Dict)` | dispatch + decode |

`generate_text` accepts `method` in `{"standard", "power", "max_swap"}` and raises `ValueError` on anything else. `None` overrides fall back to the config value.

### 6.2 `make_logits_fn` / `make_batch_logits_fn`

```python
make_logits_fn(
    model,
    ctx_len=None,             # None => variable-length forward; int => right-pad to ctx_len
    pad_id=None,              # REQUIRED when ctx_len is set
    logits_key="logits",      # dict key; None => the model returns a bare tensor
    position=-1,              # -1 => last REAL token; else a fixed position
    text_slice_start=0,       # offset added to the gather index (skip a vision prefix)
    extra_inputs=None,        # e.g. {"images": img}; triggers the dict-input call
    token_key="text_tokens",  # dict key for the token array when extra_inputs is set
) -> LogitsFn                 # List[int] -> float32[V]

make_batch_logits_fn(...)     # same kwargs; List[List[int]] -> float32[B, V]
```

The last-token gather is pure NumPy fancy indexing (`logits[np.arange(B), idx]`) on the host-side array.

### 6.3 `VLMForwardAdapter`

```python
VLMForwardAdapter(
    model, image, *,
    image_key="images", token_key="text_tokens",
    text_slice_start,          # vision sequence length — CALLER'S responsibility
    logits_key="logits", ctx_len=None, pad_id=None,
)
# .as_logits_fn()        -> single-sequence LogitsFn bound to the fixed image
# .as_batch_logits_fn()  -> batched closure bound to the fixed image
```

`text_slice_start` (== vision sequence length) is supplied by the caller; repo VLMs such as `nano_vlm` do not expose it as a property, and auto-deriving it is out of scope. A wrong value gathers logits from *inside* the vision prefix and produces garbage.

### 6.4 `TokenizerProtocol`

```python
class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> str: ...
```

A `runtime_checkable` `typing.Protocol` capturing the only two methods the sampler ever calls. Vocabulary metadata is intentionally **not** part of the contract — it lives in `PowerSamplingConfig`, so tokenizers lacking those attributes still qualify. A `tiktoken.Encoding`, an HF `AutoTokenizer` wrapper, a SentencePiece wrapper or a hand-rolled char tokenizer all satisfy it with no inheritance.

### 6.5 `_log_softmax` / `_nucleus_sample`

Exported despite the leading underscore, because the test suite and the sampler's callers pin their exact numerics. `_nucleus_sample` returns `(token_id, log_prob)` where the log probability is taken over the truncated, renormalized nucleus — the density the token was drawn from, which the caller cannot recover from a full-vocabulary log-softmax.

---

## 7. Configuration Reference

`PowerSamplingConfig` is a plain `@dataclass` — not a Keras component, no registration, no `get_config`.

| Field | Default | Meaning |
|:---|:---|:---|
| `temperature` | `0.25` | Proposal temperature; power exponent is `alpha = 1/temperature`. |
| `mcmc_steps` | `10` | MH proposals generated and evaluated per block. |
| `block_num` | `16` | Generation blocks; refinement runs after each. |
| `max_tokens` | `512` | Total tokens, split across blocks by `_block_sizes` (§3). |
| `top_p` | `0.92` | Nucleus threshold for the proposal distribution. |
| `repetition_penalty` | `1.3` | Sign-aware penalty on recently generated tokens. |
| `repetition_window` | `50` | How many recent tokens it considers. |
| `special_token_ids` | `set()` | Ids masked to `-1e9`. Empty by default — supply your model's. |
| `cls_token_id` | `None` | Token prepended to every prompt. `None` => no prepend, nothing stripped. |
| `pad_token_id` | `None` | Right-padding id. See below. |
| `ctx_len` | `None` | Fixed context length for fixed-shape models. `None` => variable-length, unpadded. |

**The identity fields carry no GPT-2 / CliffordNet defaults.** They are empty/`None` so the engine drives any model; restoring hard-coded ids would silently mis-mask every other model.

**`pad_token_id` is required in two cases**, both refused eagerly by `PowerSampler.__init__` rather than surfacing mid-run:

- `ctx_len` is set (fixed-shape forward passes), **and**
- `mcmc_steps >= 2` on the wrapped-model path — MCMC proposals are re-generated from random cut points, so the proposal batch holds prefixes of unequal length and must be right-padded to the batch maximum.

It is unused only for single-proposal or injected-closure sampling.

---

## 8. Comprehensive Usage Examples

### Example 1: GPT-2 via `tiktoken`

```python
import tiktoken
from dl_techniques.models.common.power_sampling import PowerSampler, PowerSamplingConfig

enc = tiktoken.get_encoding("gpt2")     # satisfies TokenizerProtocol
model = load_gpt2_like_model()          # callable returning {"logits": float32[B,T,V]}

config = PowerSamplingConfig(
    temperature=0.25, mcmc_steps=10, block_num=8, max_tokens=128,
    pad_token_id=enc.eot_token,
    ctx_len=1024,                       # omit for a variable-length model
)
sampler = PowerSampler(model, enc, config)
text, info = sampler.generate_text("In mathematics, a prime number is", method="power")
```

### Example 2: Any tokenizer, via the protocol

```python
from transformers import AutoTokenizer

class HFTokenizerAdapter:
    def __init__(self, name): self._tok = AutoTokenizer.from_pretrained(name)
    def encode(self, text): return self._tok.encode(text, add_special_tokens=False)
    def decode(self, ids):  return self._tok.decode(ids, skip_special_tokens=True)

config = PowerSamplingConfig(temperature=0.3, mcmc_steps=8, block_num=8,
                             max_tokens=120, pad_token_id=0)
sampler = PowerSampler(load_my_lm(), HFTokenizerAdapter("gpt2"), config)
text, info = sampler.generate_text("Once upon a time", method="max_swap")
```

No inheritance, no registration.

### Example 3: A VLM

```python
from dl_techniques.models.common.power_sampling import (
    PowerSampler, PowerSamplingConfig, VLMForwardAdapter,
)

adapter = VLMForwardAdapter(
    vlm, image,
    image_key="images", token_key="text_tokens",
    text_slice_start=196,               # CALLER-supplied vision-token count
    logits_key="logits",
)
config = PowerSamplingConfig(temperature=0.25, mcmc_steps=6, block_num=6,
                             max_tokens=64, pad_token_id=0)
sampler = PowerSampler(vlm, tokenizer, config, logits_fn=adapter.as_logits_fn())
text, info = sampler.generate_text("Describe the image:", method="power")
```

### Example 4: CliffordNet, the original use case

A fixed-shape model with GPT-2-style special tokens. Behavior is preserved by passing the ids explicitly.

```python
import tiktoken
from dl_techniques.models.vision.cliffordnet.lm import CliffordNetLM
from dl_techniques.models.common.power_sampling import PowerSampler, PowerSamplingConfig

enc = tiktoken.get_encoding("gpt2")
model = CliffordNetLM.from_variant("base", vocab_size=50261)   # vocab_size is REQUIRED

config = PowerSamplingConfig(
    temperature=0.25, mcmc_steps=10, block_num=8, max_tokens=200,
    cls_token_id=50257,
    pad_token_id=50260,
    special_token_ids={50257, 50258, 50259, 50260},
    ctx_len=511,
)
sampler = PowerSampler(model, enc, config=config)

for method in ["standard", "power", "max_swap"]:
    text, info = sampler.generate_text("The theory of relativity states that", method=method)
    print(f"\n--- {method} ---\n{text[:200]}")
```

---

## 9. Advanced Usage Patterns

**Plain-tensor models.** If the model returns a bare logits tensor rather than a dict, build the closure with `logits_key=None` and inject it:

```python
logits_fn = make_logits_fn(model, logits_key=None)
sampler = PowerSampler(model, tokenizer, PowerSamplingConfig(), logits_fn=logits_fn)
```

**Variable-length forward (`ctx_len=None`).** Runs an unpadded pass, appropriate for models accepting dynamic lengths. No `pad_token_id` is needed for single-sequence generation (`generate_standard`, or MCMC at `mcmc_steps=1`); `mcmc_steps >= 2` still needs one (§7).

**Tuning.** More `block_num` => finer-grained refinement, more forward passes. More `mcmc_steps` => more proposals per block. Start from `block_num=8, mcmc_steps=10` and scale down for speed.

---

## 10. Performance Notes

- **GPU forward / CPU NumPy split.** The model forward runs on whatever device the model uses; everything after the logits — `_log_softmax`, masking, repetition penalty, nucleus sampling, the MH test — is NumPy on CPU. This keeps the sampling math out of the compiled graph and avoids retraces.
- **No KV cache.** Every forward pass recomputes all positions. Designed for offline/batch inference, not real-time streaming.
- **Cost model.** Roughly `(1 + mcmc_steps)`x the `standard` baseline at 0% acceptance. Acceptances add speculative waste: proposals queued behind an accepted one are thrown away, so a block generates between `mcmc_steps` and `mcmc_steps * (mcmc_steps + 1) / 2` proposals. At `mcmc_steps=10` and a ~30% acceptance rate that is roughly 2.5–3x the minimum proposal FLOPs, bought in exchange for `1 + acceptances` batched call-rounds instead of `mcmc_steps` sequential ones.
- A single active sequence short-circuits to the unbatched path to avoid batch overhead.

### Method selection

| Method | Samples from | Cost | When |
|:---|:---|:---:|:---|
| `standard` | `p` (nucleus) | 1x | Fast baseline; A/B comparison; latency over coherence. |
| `power` | `p^alpha` (MCMC) | ~`(1+mcmc_steps)`x | Default. Trajectory coherence with diversity preserved. |
| `max_swap` | `p^infinity` | ~`(1+mcmc_steps)`x | The single most coherent continuation; diversity is not a concern. |

---

## 11. Statelessness

There is nothing to serialize. `PowerSampler` holds references to a model, a tokenizer and a dataclass config; it owns no weights and is not a `keras.Model`, hence no registration decorator, no `build`, no `get_config`. Reconstruct it by re-instantiating.

`generate_standard` does **not** mutate `self.config`: per-call `top_p` / `repetition_penalty` overrides go through a transient `dataclasses.replace` copy restored in a `finally` block, so config fields are identical before and after — including when the loop raises.

---

## 12. Troubleshooting & FAQs

**The acceptance ratio is near 0.** The proposal and target distributions are too far apart. A very low `temperature` makes the target extremely peaked, so almost no stochastic proposal clears the MH test. Raise `temperature`, reduce `mcmc_steps`, or use `max_swap` for deterministic acceptance.

**`ValueError: PowerSamplingConfig.pad_token_id is required when mcmc_steps >= 2`.** Working as intended (§7). Supply a pad id, or drop to `mcmc_steps=1`.

**`ValueError: logits_key='logits' not found in model output`.** Your model does not return a dict keyed `"logits"`. Bare tensor: `logits_key=None`. Different key: pass it. The error message lists the available keys.

**My VLM output is garbage.** Almost always a wrong `text_slice_start` — it must equal the vision-token sequence length so the gather lands on the first *text* position (§6.3).

**Decoded text has a leading space.** A tokenizer artifact. Many BPE tokenizers encode a leading space into the first token; the sampler only calls `encode`/`decode` and never alters the byte stream.

---

## 13. Testing & Validation

A keras-free mock suite (a mock model returning `{"logits": ...}` plus a char-level mock tokenizer) covers: config defaults; `_log_softmax` / `_nucleus_sample` numerics, including that the nucleus draw reports the renormalized density it sampled from, with the degenerate `top_p = 1.0` twin as the anti-vacuity control; the MH chain proposing from the state it just accepted; the eager `pad_token_id` precondition; `make_logits_fn` for dict / plain-tensor / VLM-offset outputs; `generate_standard`; `mcmc_power_sample` (acceptance ratio in `[0, 1]`, output length); `max_swap`; `generate_text` dispatch and its `ValueError`; CLS-prepend on and off; and the no-config-mutation invariant.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 \
    .venv/bin/python -m pytest tests/test_models/test_power_sampling/ -vvv
```

---

## 14. Citation

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
