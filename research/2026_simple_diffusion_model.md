# Simplex Diffusion Models (SDM) — Integration Research

**Paper:** Deschenaux et al., *Simplex Diffusion Models*, arXiv:2609.35553v1 [cs.LG] 28 Sep 2026 (CC BY 4.0).
**Status:** research-only, no code written. Target: new model family properly integrated into `dl_techniques`.
**Date:** 2026-10-14. **Author:** integration investigation (no implementation).

> House rules that govern everything below: `research/2026_keras_custom_models_instructions_v2.md` (mandatory authoring guide),
> `src/dl_techniques/models/AGENTS.md` (authoring contract), `src/dl_techniques/models/GUIDE.md` (measured defects),
> `REPO_MAP.md` (model/trainer/test triangle), per-package `AGENTS.md` files.

---

## 1. What SDM is in one page

SDM lifts discrete diffusion from token space `X = {1..N}` to the probability simplex `Δ_N`.
Intermediate states are **distributions over tokens**, not tokens. This carries denoiser uncertainty
across steps and mitigates *information collapse* (discrete samplers discard the categorical
distribution at each step and must bolt on Self-Conditioning or predictor-corrector to recover it).

Three claims structure the paper:

1. **Dirichlet forward with decoupled mean/concentration + closed-form reverse bridge.**
   Forward `p_{t|0}(P_t|P_0) = Dir(P_t, c_t(α_t P_0 + (1-α_t)π))` with schedule `α_t ↓ (1→0)`
   and concentration `c_t > 0`, prior `π` (uniform in practice, full support required).
   Reverse is a DDIM-style bridge with a **churn** `κ ∈ [0,1]` controlling stochasticity.
   Training is plain **cross-entropy** (Eq. 16), no ODE integration (unlike Dirichlet Flow Matching).
2. **Unification.** Sampling `x_t ~ Cat(P_t)` recovers discrete diffusion marginals (Prop 3.1).
   `ε → 0` (high temperature) collapses to vertices = discrete diffusion; `ε → ∞` gives
   deterministic interpolation + Gaussian fluctuations (Props 4.1, E.2–E.4).
3. **Empirics.** Competitive without Self-Conditioning (SC): TinyGSM 49.0% @T=0.1 vs 45.8% best
   discrete baseline; 8-step distilled SDM 32.1% GSM8K > 128-step distilled discrete 21.4%;
   Sudoku 99.1% w/ SC (≈ best 99.3%); OWT 17.0 GenPPL / 5.46 H1 in 64 steps.

Scope for us: **phase 1 = forward + reverse + CE + DiT denoiser + sampler**. Defer distillation
(App. G), auto-guidance (K.4), CFG (C.2), logit-shaping stack (K.5) to phase 2.

---

## 2. Forward process (what to implement verbatim)

### 2.1 Probability path

```
P_0 = e_{x0}                                    # one-hot, Eq. 4
β_t(P_0,π) = c_t (α_t P_0 + (1-α_t) π)           # Eq. 6
p_{t|0}(P_t|P_0) = Dir(P_t, β_t)                 # concentration C = c_t
E[P_t|P_0] = α_t P_0 + (1-α_t) π := P̄_t
Cov = (diag(P̄_t) − P̄_t P̄_tᵀ) / (c_t + 1)
p_1 = Dir(·, c_1 π)  (α_1 = 0, independent of P_0)
```

Multi-token (C.1): product over positions `∏_ℓ Dir(P_t^ℓ, β_t(P_0^ℓ,π))`, shape `(B, L, N)`.
Denoiser is joint `P̂^θ : (Δ_N)^L × [0,1] → (Δ_N)^L`; forward/reverse factorize per position,
prediction sees full sequence.

### 2.2 Interpolation representation (Prop 3.2 — the sampler's friend)

```
W_t ~ Beta(c_t α_t, c_t(1−α_t)),  V_t ~ Dir(c_t(1−α_t) π), independent
P_t = W_t P_0 + (1−W_t) V_t     # Eq. 8, exact via Prop A.1
```

Degenerate convention (A.1): zero Dirichlet coords = support on face; zero Beta param =
degenerate at 0/1; `t=0` formulas are limits.

### 2.3 Concentration schedule via normalized variance (§4.2)

Do **not** expose raw `c_t ∈ (0,∞)` as the user knob. For uniform `π = 1/N`:

```
TVar(t) = Tr[Cov] = (N−1)/N · (1−α_t²)/(c_t+1)     # Eq. 20
ν_t := 1/(c_t+1)  ⇔  c_t = 1/ν_t − 1,  ν_t ∈ (0,1)
```

Two shipped forms (Eq. 21):

```
ν^cst_t = ν_0
ν^cst-lin_t = ν_0 (t<ℓ) else ν_0 + (t−ℓ)/(1−ℓ)·(ν_1−ν_0)
```

Paper sweeps: Sudoku best `ν^cst-lin(0.4,0.75,0.8)` and `(0.2,0.75,0.2)`;
TinyGSM expectation-input best `(0.2,0.5,0.2)`; also `Constant(0.5/0.25/0.1)`.
At `t=1`, `c_1 = N` ≈ uniform-on-simplex; larger concentrates on `π`.

### 2.4 Temperature parameterization (theory + Dirichlet sampling)

Analysis uses `c_t = ε/(1−α_t)` (`ε` = inverse temperature). Then `b_t = c_t(1−α_t) = ε`
constant, `r_{s,t} = 1`, bridge simplifies to `P_s = W P_t + (1−W) e_{x0}` at `κ=0`.
Figure 6/7: `ε→0` = vertices/discrete; `ε→∞` = deterministic `P̄_t`; intermediate =
usable SDM. Gaussian fluctuation (E.7): `√ε(P_t−P̄_t) → N(0,(1−α_t)²V_t)`,
`V_t = V_π + α_t(P_0−π)(P_0−π)ᵀ`, projected spectrum in `[min π_i, 2]` — stable.

**Dirichlet sampling (App. D):** Gamma-construction `Y_i ~ Gamma(α_i,1)`,
`P = Y/ΣY` (Eq. 26/44). Paper uses Marsaglia–Tsang (Alg. 3) with boosting for `α<1`,
rejects JAX-native vmap key-splitting (memory overhead) for counter-based vectorized PRNG
(3.8× speedup note). For us: **check `keras`/`tf.random.gamma` first**; only hand-roll if
measured slow/biased. `keras.random.gamma` exists but verify dtype + graph-mode + XLA
before committing (see §10 hazards).

---

## 3. Reverse bridge (the core new sampler)

### 3.1 Compatibility requirement

Need `p_{s|0,t}` with `p_{s|0}(·|P_0) = ∫ p_{s|0,t}(·|P_0,P_t) p_{t|0}(dP_t|P_0)` (Eq. 9).

Write `β_t = a_t P_0 + b_t π`, `a_t = c_t α_t`, `b_t = c_t(1−α_t)`,
`r_{s,t} = min(1, b_s/b_t)`, `ρ^κ_{s,t} = (1−κ) r_{s,t}`.

### 3.2 One reverse step (Prop 3.3, Alg. 2)

Given `P_t` at time `t`, predicted clean `P_0 = e_{x̃0}`, `x̃0 ~ Cat(P̂^θ(t,P_t))`:

```
for i=1..N:  B_i ~ Beta(ρ·β_t,i, (1−ρ)·β_t,i)          # thinning, Prop A.2
P^κ_{s,t,i} = B_i·P_{t,i} / Σ_j B_j·P_{t,j}            # stays on simplex
W ~ Beta(ρ·c_t, c_s − ρ·c_t)                            # mixing weight
V ~ Dir(β_s − ρ·β_t)                                    # innovation, ≥0 by r_{s,t}
P_s = W·P^κ_{s,t} + (1−W)·V                            # Eq. 12
```

Boundary: `ρ=1` ⇒ `P^κ = P_t`, skip `B_i`. `κ=1` ⇒ `p_{s|0,t} = p_{s|0}` (independent
resample, star-shaped/Okhotin et al. 2023). `κ=0` keeps max compatible mass (Eq. 13).

Per-position loop for sequences (Eq. 34–39); final step `x^ℓ ~ Cat(P̂^ℓ(t_1,P_{t1}))`.

Eager Python loop over `M` steps (64–512 typical; paper up to 8k NFE for curves).
Same tracing rule as `GaussianDiffusion`: **loops stay eager, per-step ops traceable**
(see `models/vision/dit/diffusion.py` module docstring + `test_dit_diffusion.py` call-count test).

### 3.3 DDIM connection

At `c_t = ε/(1−α_t)`, `κ=0`, `ε→∞`: `P_s = (1−α_s)/(1−α_t)·P_t + (α_s−α_t)/(1−α_t)·e_{x0}`
(Eq. 14) — exact deterministic DDIM rule. Paper's sampler is thus a stochastic DDIM on the
simplex; `κ` is the `η`-analogue (DiT's `ddim_sample(eta)` in `dit/diffusion.py:912-987`).

### 3.4 Exact belief-memory view (E.4–E.5, useful for tests)

Under exact posterior + `κ=0`, `P_{ti} = Σ_{j≥i} L_{i,j} e_{x0^j} + L_{i,M} P_{tM}`,
`L_i ~ Dir(c_{ti}−c_{ti+1},…,c_{tM})`. Mean profile independent of `ε`;
`ε→∞` → deterministic mix, `ε→0` → categorical jump (Cor. E.1). Good oracle for a
stick-breaking unit test: mean + variance of `L_i`, not just shape.

---

## 4. Training and inference contracts

### 4.1 Training (Alg. 1)

```
x0 ~ p_data;  t ~ w(t)          # uniform default; adaptive §J.1 phase 2
P_0 = one_hot(x0);  β_t = c_t(α_t P_0 + (1−α_t)π)
P_t ~ Dir(β_t)
P̂_0 = P̂^θ(t, P_t)               # softmax output
L = −log P̂_{0,x0}               # CrossEntropy, Eq. 16
```

`w(t)` uniform over inference grid ⇒ negative ELBO (Prop F.1, factor `1/M` unbiased via
uniform `k` + `P_k ~ Dir(β_{tk})`). Sequence loss = sum over positions (C.1); Sudoku/TinyGSM
corrupt target half only, keep prompt clean; OWT corrupts all 1024.

**Loss placement:** plain CE needs no packed-`y_true` trick. If we later need `t`-dependent
weighting, do it at the **sampling level** (cf. `FlowMatchingVelocityLoss` docstring:
time-weighting deliberately not in loss). Keep `call()` returning `(B,)` per-sample
(losses/AGENTS.md shape rule + `test_the_premature_scalar_family_is_pinned.py`).

### 4.2 Inference (Alg. 2)

```
P_1 ~ Dir(c_1 π)
for k=M..2:  t=t_k, s=t_{k−1}
  x̃0 ~ Cat(P̂^θ(t,P_t));  P_0 = e_{x̃0}
  r, ρ, aW=ρc_t, bW=c_s−ρc_t, βV=β_s−ρβ_t   # §3.2
  W~Beta(aW,bW); V~Dir(βV); B_i thinnings; P^κ; P_{k−1}=W·P^κ+(1−W)·V
x ~ Cat(P̂^θ(t_1,P_{t1})); return x
```

Temperature `T`: `softmax(ζ/T)` replaces `P̂^θ` (I.1, Eq. 148). Paper: `T∈{1.0, 0.1}`,
greedy AR still best on TinyGSM (62.6%) — report both.

### 4.3 Denoiser inputs (J.2 — affects FLOPs + arch)

```
Expectation: e(P_t)_ℓ = P_t^{ℓ⊤} E = Σ_i P_{t,i}^ℓ E_i    # Eq. 163, dense 2Nd/token
Argmax:      e(P_t)_ℓ = E[argmax_i P_{t,i}^ℓ]              # Eq. 164, lookup
+ SC: backbone input e(P_t)+LN(h_prev), h=f_θ(·,t), P̂=g(h) # I.3, Eqs. 154-155
```

Expectation uses full belief; argmax keeps sampler continuous while denoiser sees discrete
tokens. TinyGSM N≈49k,d=768: expectation +≈29% forward FLOPs (paper). SC training cost
3.9 vs 3.0 forward-equivalents; SC runs train 77% steps to match FLOPs (I.3). SC init:
LN scale+shift = 0.

### 4.4 Schedules: training sampler vs inference grid (J.1)

- **Time sampler:** uniform (†) vs adaptive (‡, `q(t) ∝ dL_t/dt`, 50-bin piecewise-linear
  + EMA 0.9, refit every 50 steps after 1k warmup, inverse-CDF via 1k-point lookup).
- **Inference grid:** linear `i/n`, cosine `cos(π/2·(1−i/n))`, adaptive `t_i = F̂^{−1}(i/n)`.
Adaptive needs adaptive-trained checkpoint. Paper: adaptive helps expectation-input most
(+9–11 pts @512 steps, +18–19 @64); can hurt argmax + UDM+PC; MDM+SC benefits.

### 4.5 Extensions (phase 2, do not build now)

- CFG (C.2): dropout condition `p_drop`, guided `P̃ ∝ (P̂^u)^{1−w}(P̂^c)^w` (Eq. 43).
- DMD distillation (App. G, Alg. 4): generator `G_η` + auxiliary `D_φ`, surrogate
  `at·log(P_t)ᵀ·sg(D_φ−D_teach)`, differentiable Gamma path. Single gradient path through
  `F_t`, unlike IDLM's discarded indirect term.
- Auto-guidance (K.4): `ζ̃ = ζ_final + w(ζ_final − ζ_K)`, early-mid `K∈[20k,50k]` best.
- Logit shaping (K.5): nucleus top-p → power-law anneal `T(t)=T_start+(T_end−T_start)t^1.5`
  → sequence freq penalty `ζ−λlog(1+c)` → local freq penalty (SDM-only gain).

---

## 5. Empirical anchors (what "working" means)

| Task | Setup | SDM | Baseline |
|---|---|---|---|
| Sudoku 180 steps | DiT 8L/512d/8h 28.6M, 50k steps | 88.4% (κ=1, no SC) / **99.1%** (+SC) | MDM+SC 99.3, UDM+PC 95.8, DFM 76.7 |
| TinyGSM 512 steps | DiT 12L/768d/12h 167.9M, 250k×512 | T=1 **45.8%** / T=0.1 **49.0%** (expectation, no SC) | UDM+PC 36.8 / MDM+SC 45.8; Argmax+SC 57.0% @8k NFE |
| TinyGSM 64 steps | same, adaptive grid | 32.7% T=1 / 39.5% T=0.1 | UDM+PC 23.7 / 38.3 |
| GSM8K distilled 8 NFE | SDM-DMD | **32.1%** | IDLM 21.4% @128 NFE |
| OWT 64 steps | DiT 12L/768d, 1M×512, L=1024 | 17.0/5.46 ≈ val (15.2/5.46) w/ 4-stage shaping | MDM/UDM same frontier; AR better than data (benchmark warning) |
| Molecules | BERT 12L/768, SAFE 1880 vocab, L=256 | quality≈87% + diversity 0.857 (conf. sampling) | MDM 92.8 / UDM 92.0 (higher quality, lower diversity) |

Invariants: `κ=1` best everywhere without SC (73.9→88.4 Sudoku, 12.6→45.8 TinyGSM T=1);
`κ=0.2` sometimes < `κ=0` for argmax @64 steps (non-monotone — sweep `{0,0.2,1.0}` min).

---

## 6. Codebase reuse inventory (verified paths)

### 6.1 Model template — copy `vision/dit/`, do not re-derive

| Asset | Path | Reuse |
|---|---|---|
| House model shape | `src/dl_techniques/models/vision/dit/model.py` (`DiT`, `create_dit`, `from_variant`, `MODEL_VARIANTS`, `get_config`, `pretrained=True→NotImplementedError`) | Template for `SimplexDiT` class + factory; BERT (`models/language/bert/model.py`) is normative Sphinx exemplar |
| Block pattern | `models/vision/dit/blocks.py` (`DiTBlock`, `DiTFinalLayer`, explicit names, build-only-what-call-runs) | Reuse, not fork: SDM DiT needs bidirectional (non-causal) blocks; DiT already non-causal |
| Sampler pattern | `models/vision/dit/diffusion.py` (`GaussianDiffusion`: `q_sample`, `p_mean_variance`, `p_sample`/`ddim_sample` + loops, respacing, explicit `seed`, eager loops + traceable steps, nonzero-mask + `eta=0` pins) | Mirror as `SimplexDiffusion` sampler: `dirichlet_sample`, `bridge_step`, `sample_loop_progressive` + `sample_loop`; keep explicit-seed + call-count test |
| Schedule value-object | `src/dl_techniques/utils/ddpm_schedule.py` (`DDPMSchedule` frozen dataclass float64 + `space_timesteps` + `respaced`) | Precedent for `SimplexSchedule` (α/ν/c tables float64, NumPy-owned, cast at use); lives in `utils/`, **not** in model pkg (D-030: three consumers across packages) |
| Config | `models/vision/dit/config.py` (`DIT_VARIANTS`, `DiffusionConfig`) | Add `SIMPLEX_VARIANTS` (Sudoku-28M, TinyGSM/OWT-168M rows) + `SimplexConfig(α, ν, π, κ, grid)` |
| Loss pattern | `src/dl_techniques/losses/ddpm_hybrid_loss.py` (packed-`y_true` contract, stop-grad, `(B,)` return, float64 tables→cast-in-call) | SDM CE is simpler (no packing); copy `(B,)` + dtype + validation discipline |
| Bridge precedent | `models/vision_language/bit_diffusion/{sde.py,bridge_process.py,blocks.py,config.py}` (`BridgeSDE`, `bridge_math_dtype` never-narrow, `_expand_like` broadcast, `TIME_EPS` clamping, score targets + DSM weights, golden float64 pins) | Closest math cousin; copy never-narrow dtype + expand + eps-clamp habits; **do not** import from it (models→models import noted as hazard in `layers/embedding/sincos_pos_embed_2d.py`) |
| Flow baseline | `models/vision_language/{ideogram4/scheduler.py,sd3_mmdit/}` + `losses/flow_matching_velocity_loss.py` | Scheduler + velocity-loss conventions if we later add DFM baseline |

### 6.2 Backbone pieces — factory-first (layers/AGENTS.md § Layer Reuse Policy)

| Domain | Entry point | Keys to use |
|---|---|---|
| Embeddings | `layers/embedding/factory.py::create_embedding_layer` (`EMBEDDING_REGISTRY`, 15 types) | `timestep` (DiT/GLIDE cos-first, `hidden_size` req.), `scalar_sinusoidal` (Ideogram4 `[0,1]` scalar, NOT interchangeable with `timestep`), `bert_embeddings` / `modern_bert_embeddings` for token tables, `class_label` for CFG null row |
| Norms | `layers/norms/factory.py::create_normalization_layer` | Route all norms here; factory eps `1e-6` vs Keras default `1e-3` (1000× hazard, GUIDE §5). Pass `epsilon=` explicitly with citation |
| Attention/FFN | `layers/attention/factory.py`, `layers/ffn/factory.py` (`gelu_tanh` = DiT's tanh-GELU) | `TransformerLayer` (`layers/transformers/transformer.py`, direct import, no factory) covers most; `text_encoder.py`/`vision_encoder.py` preset builders |
| AdaLN | `layers/transformers/sd3_adaln.py` (`AdaLayerNormZero` 6-way, `AdaLayerNormContinuous` 2-way, `modulate`) | Direct reuse for simplex DiT conditioning `c = t_emb (+ y_emb)`; DiT `blocks.py` shows composition |
| Masking | `utils/masking/strategies.py`, `dl_techniques.utils.masking` (canonical mask factory) + `layers/attention/common.py::apply_attention_mask` | Use for padding/conditional-prefix handling; **never** hand-roll additive `−1e9` mask (fp16 `0·−inf=NaN`, GUIDE §4) |
| Heads | `layers/heads/factory.py::create_head(domain,…)` + `layers/heads/nlp/` | Denoiser output head = logits over vocab; follow `common/masked_language_model` head conventions (untied by default for model-agnostic wrapper) |
| Activations | `layers/activations/common.py` trio + `utils/activation_serialization.py` | Canonical `(de)serialization`; Layer-rejecting contract |

### 6.3 Training wrappers — do not invent a new one

| Asset | Path | Note |
|---|---|---|
| MLM/CLM wrappers | `models/common/masked_language_model/{mlm.py,clm.py,utils.py}` + `models/language/AGENTS.md` | `MaskedLanguageModel` (bidirectional, dynamic masking in `train_step`, TF-only) / `CausalLanguageModel` (`skip_head`, `pre_shifted`, `loss_fn`, `aggregate_backbone_losses`, `output_key`, causality probe). SDM denoiser is **bidirectional non-causal** → MLM-shaped wrapper is the closer cousin, but SDM needs simplex-state input + Dirichlet corruption, so **new wrapper**, same head discipline (`task_head/` prefix, byte-identical headless output) |
| NLP pipeline | `src/train/common/nlp.py` (`create_tokenizer`, `preprocess_{mlm,clm}_dataset`, `estimate_clm_steps_per_epoch`, warmup schedule, `build_sentiment_model_from_encoder`) + `src/train/AGENTS.md` Pattern 3 | Reuse tokenization/dataset/steps helpers; byte path uses `datasets/byte_lm.py` (vocab 256 cannot use tiktoken pipeline) |
| Trainer scaffold | `src/train/dit/{train_dit.py,synthetic_data.py,ema_callback.py}` | Copy CLI→config map (`CLI_TO_CONFIG` + `test_cli_contract`), explicit `model.build()` before `fit` (EMA correctness), stock `fit` assertion (`train_step is keras.Model.train_step`), timestamped run dirs, `create_callbacks` + `WeightEMACallback(0.9999)` |
| Common trainer lib | `src/train/common/{callbacks.py,args.py,run_io.py,stats.py,optimizer.py}` | `create_callbacks`, `resolve_monitor_mode`, `set_seeds`, `setup_gpu`, `optimizer_builder` (note renamed clipping keys + hardcoded `AdamW` name) |
| Losses on hand | `losses/{masked_causal_lm_loss.py (MaskedCausalLMLoss, (B,) token-mean-preserving), flow_matching_velocity_loss.py, any_loss.py}` | SDM-CE should mirror `MaskedCausalLMLoss` `(B,)` semantics; register via `@register_dl_technique("dl_techniques.losses.<module>")`, export from `losses/__init__.py` |
| Utils | `utils/{logger.py, keras_registration.py, weight_transfer.py, sequence_length_guard.py, tensors.py, dtype_policy.py}` | Central logger (no print, no logging in `call`); stripped registration key `dl_techniques.models.<name>.<module>`; `load_backbone_from_checkpoint` for future warm-starts; `RecordedBuildShapeMixin` + int64 tables for XLA/GPU |
| Datasets | `datasets/{nlp.py, byte_lm.py}` | Wikipedia/HF conventions (`min_article_length`, `num_shards`, `return_counts`); same corpus, different tokenizer for byte vs BPE |

### 6.4 Tests to mirror

`tests/test_models/test_dit/` (surface, config, diffusion incl. nonzero-mask/`eta=0`/respacing-map,
blocks attribution, identity-at-init, CFG 3-channel, variance-logit, null-row, frozen pos-embed,
unpatchify orientation, build-parity, gradient flow) + `tests/test_models/test_masked_language_model/`
+ `tests/test_losses/test_ddpm_hybrid_loss.py` + `test_the_premature_scalar_family_is_pinned.py`
+ `test_package_api_contract.py` + `test_the_legacy_alias_namespace_has_no_collisions.py`
+ tree-wide `pytest tests/test_models/ -q --collect-only` (~3 s) after public-surface changes.
Scoped runs only (full suite ≈1.5 h, no parallel pytest, `CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg`).

---

## 7. Proposed integration (no code yet)

**Location:** `src/dl_techniques/models/language/simplex_diffusion/` (file by input modality:
token sequences; cf. `models/README.md`). Family `__init__` stays docstring-only; leaf binds
`create_simplex_dit` + `__all__` + `README.md`. Registration strips family:
`@register_dl_technique("dl_techniques.models.simplex_diffusion.model")`.

```
models/language/simplex_diffusion/
  __init__.py       # SimplexDiT, create_simplex_dit, SimplexDiffusion, SimplexSchedule bits
  model.py          # denoiser backbone (DiT-shaped, bidirectional, adaLN, Sphinx docstring per bert/model.py)
  forward.py        # β_t, ν→c, dirichlet_sample (Gamma path), Prop 3.1/3.2 oracles
  reverse.py        # thin (B_i) + W/V + P_s, κ, multi-token loop, adaptive/linear/cosine grids
  denoiser_inputs.py# expectation vs argmax + SC (LN-zero-init)
  schedules.py      # α_t (1−t default, cosine opt), ν^cst/cst-lin, adaptive sampler state (phase 2)
  config.py         # SIMPLEX_VARIANTS + SimplexConfig
  README.md         # usage + measured variant table + limits (no unearned claims per test_docs guard)
```

**Do NOT bury** a general sampler/schedule in the model dir if two packages will import it:
schedule value-object → `utils/simplex_schedule.py` (DDPMSchedule precedent D-030);
CE loss → `losses/simplex_crossentropy.py` (or reuse stock CE + masking — decide at build).

**Trainer:** `src/train/simplex_diffusion/train_simplex.py` (`train_<model>.py` naming — never
`train.py`), Pattern 3 + DiT stock-`fit` discipline; Sudoku (180-token, target-half corruption),
TinyGSM (SmolLM 49k, L=512), OWT (GPT-2 50k, L=1024) dataset switches; `--smoke` wiring preset.

**Config liveness:** every `TrainingConfig` field read on run path + `CLI_TO_CONFIG` both-directions
pin (`tests/test_train/test_config_fields_are_live.py` row).

---

## 8. Keras hazards to design around (from GUIDE + v2 guide)

1. Create in `__init__` (named sublayers), build tree in `build()` **exactly what `call()` runs**;
   `compute_output_shape` from stored config; `get_config` round-trips all args; no `.assign()` in
   `build()` (StatelessScope zeros), no `convert_to_tensor` in `build()` (FuncGraph bind), constant
   tables via initializer/NumPy + convert in `call()`.
2. No `keras.ops.tril/triu` (trace-time TypeError); no NumPy fancy indexing (use gather / flat-row
   gather for XLA `take_along_axis` retrace issue); no `if` on tensor values; `training=` forwarded
   explicitly; no logging in `call()`; `keras.ops` over raw TF; `import keras` qualified.
3. Dirichlet/Beta/Gamma draws: explicit `seed`/`SeedGenerator` (global seed does not reseed existing
   generators — DiT pin); never-narrow dtype (`max(input,float32)`, keep float64 path); int tables
   int64 (`INDEX_WEIGHT_DTYPE`); fp16 sentinel via `ops.where`, not `+ (1−mask)·(−1e9)`;
   assert `isfinite` in tests, not just shape; non-square fixtures to catch transposes.
4. Subclassed model (not functional) to keep head-swap weight paths; flat `List[Layer]`, never
   `List[List[Layer]]` under a Layer (98% weight loss precedent); `List` needs per-index names.
5. New `__init__` validation must not break old archives (`from_config` substitute+warn path);
   literal `==` check on registration key (round-trip is self-consistent); `custom_objects` keyed
   by `get_registered_name`; never bare `register_keras_serializable`; no `__init__` name matching
   own subpackage.
6. Docs: no `.keras`-unverifiable perf claims (`test_docs_make_no_unearned_performance_claims.py`);
   re-derive counts with commands beside them; match file docstring style, never wholesale convert.

---

## 9. Phased build plan

**Phase 0 (this doc):** agree leaf name (`language/simplex_diffusion`), phase-1 scope
(forward+reverse+CE+argmax/expectation, linear/cosine grids, `κ∈{0,0.2,1}`), deferred list (adaptive
sampler, DMD, auto-guidance, CFG, shaping stack, molecules).

**Phase 1:** `SimplexSchedule` + Gamma→Dirichlet + bridge step (float64 goldens vs paper Eqs. 6/10–12);
`SimplexDiT` tiny (BERT-tiny scale) + CE; eager sampler loop; Sudoku-30/81 smoke + tiny synthetic
vocab overfit; guard suite: Prop 3.1 marginal, bridge compatibility (Eq. 9 backward induction),
stick-breaking moments, `ε→0` vertex / `ε→∞` mean limits, nonzero-mask/`κ=1`-independence/`η=0`
analogues, build-parity, value round-trip, knob-sensitivity, contract oracle — each proven RED first.

**Phase 2:** adaptive sampler/grid, SC, CFG, TinyGSM/OWT trainers, DMD, shaping, full tables.

## 10. Open decisions for reviewer

1. Leaf path: `language/simplex_diffusion/` (proposed) vs `general_purpose/` (modality-agnostic claim)?
2. Share `TimestepEmbedding`/`AdaLayerNormZero` as-is, or wrap for `(B,L)` per-position times?
3. Gamma sampler: `tf.random.gamma`/`keras.random.gamma` sufficient, or port Marsaglia–Tsang + Greaves one-liner?
4. New `SimplexCrossEntropy` loss vs stock `SparseCategoricalCrossentropy` + `MaskedCausalLMLoss` weighting?
5. Sudoku first (180 steps, 14 vocab, fastest signal) vs TinyGSM first (paper's main claim)?

---

## 11. Implementation appendix — operations checklist (added 2026-10-14 gap-fill)

This section makes §2–§4 implementation-ready. Nothing here changes the plan; it writes out
the tensor shapes, boundary cases, and phase-2 equations that §4.5 had only summarized.

### 11.1 Shapes and defaults

- Single token: `P_t ∈ Δ_N`, `β_t ∈ ℝ_+^N`. Batch+sequence: `P_t : (B, L, N)`, all draws
  factorize over `(B, L)`, never across `L` except through the denoiser.
- `α_t`: non-increasing, `α_0 = 1`, `α_1 = 0`. Default `α_t = 1 − t` (paper Fig. 2,
  distillation teacher). Cosine grid is a *sampling grid*, not a change of `α_t` itself.
- `π`: full support required (else `P_t` reveals `x0` a.s. for `t < 1`). Default uniform `1/N`.
- `a_t = c_t α_t`, `b_t = c_t(1−α_t)`, `β_t = a_t·e_{x0} + b_t·π` per position.
- `r_{s,t} = min(1, b_s / b_t)` (`b_t > 0` for `t < 1`; `t = 1` is the `Dir(c_1 π)` endpoint).
  `ρ^κ_{s,t} = (1−κ)·r_{s,t}`. Clamp `t ∈ [TIME_EPS, 1−TIME_EPS]`-style epsilons at both ends
  (cf. `bit_diffusion` `TIME_EPS`): at `t = 0`, `β_0` degenerates to a vertex; treat only via limit.
- Denoiser I/O: `P̂^θ(t, P_t) : (B, L, N)` softmax rows; `t : (B,)` or scalar broadcast.

### 11.2 Dirichlet / Beta / Gamma primitives (App. A + D)

- Densities: `Beta(x;a,b) = Γ(a+b)/(Γ(a)Γ(b))·x^{a−1}(1−x)^{b−1}`;
  `Dir(P;β) = Γ(Σβ_i)/∏Γ(β_i)·∏p_i^{β_i−1}`; mean `β/C`, cov `(diag(P̄)−P̄P̄ᵀ)/(C+1)`, `C = Σβ_i`.
- Gamma construction: `X_i ~ Gamma(α_i,1)` indep. ⇒ `X/ΣX ~ Dir(α)`.
- Prop A.1 (sum): `X ~ Dir(α)`, `V ~ Dir(γ=β−α)`, `W ~ Beta(Σα, Σγ)` indep. ⇒ `WX+(1−W)V ~ Dir(β)`.
- Prop A.2 (thinning): `X ~ Dir(α)`, `B_i ~ Beta(ρа_i,(1−ρ)α_i)` indep. ⇒
  `Y_i = B_i X_i/Σ_j B_j X_j ~ Dir(ρα)` (restrict to support face `{i:α_i>0}`).
- Marsaglia–Tsang (Alg. 3) per Gamma draw: if `α<1` boost `α'=α+1` else `α'=α`;
  `d=α'−1/3`, `c=1/√(9d)`; repeat `z~N(0,1)`, `u~U(0,1)`, `v=(1+cz)^3`, `x̂=d·v`
  until `u<1−0.0331·z⁴` (squeeze) or `log u<0.5z²+d(1−v+log v)`; if boosted,
  `x=x̂·u_boost^{1/α}`. Vectorize over `(B·L·N)` with one key split, not vmap-per-element.

### 11.3 Expanded training step (Alg. 1, single position; loop over L)

```
x0 ~ p_data  [(B,L) int] ;  t ~ w(t)  [(B,) float in (0,1]]
P0 = one_hot(x0, N)                        # (B,L,N)
β  = c(t)[α(t)·P0 + (1−α(t))·π]            # broadcast π (N,) → (B,L,N)
Pt = dirichlet_sample(β)                   # (B,L,N), strictly positive a.s.
logits = backbone(t, embed(Pt))            # embed = expectation (163) or argmax (164)
P̂ = softmax(logits, −1)
loss_{b,ℓ} = −log P̂[b,ℓ,x0[b,ℓ]]           # (B,L); reduce: mean over ℓ then (B,) per-sample
```

### 11.4 Expanded reverse step (Alg. 2, per position ℓ, drop ℓ for clarity)

Inputs `Pt (N,)`, `t > s`, `κ`, tables `c, α`:

```
x̃0 ~ Cat(P̂^θ(t,Pt));  P0 = one_hot(x̃0)
bt = c_t(1−α_t);  bs = c_s(1−α_s);  r = min(1, bs/bt);  ρ = (1−κ)·r
aW = ρ·c_t;  bW = c_s − ρ·c_t                      # both ≥ 0; if bW == 0, W ≡ 1
βt = a_t·P0 + bt·π;  βs = a_s·P0 + bs·π;  βV = βs − ρ·βt   # ≥ 0 coord-wise
if ρ == 1:  Pthin = Pt
else:       B_i ~ Beta(ρ·βt_i,(1−ρ)·βt_i) ∀i;  Pthin = B⊙Pt / Σ(B⊙Pt)
W ~ Beta(aW,bW) (degenerate at 0/1 if aW or bW == 0);  V ~ Dir(βV)
Ps = W·Pthin + (1−W)·V
```

`κ = 1` ⇒ `ρ = 0` ⇒ `W ≡ 0`, `V ~ Dir(βs)`, `Ps` independent of `Pt` (star-shaped).
Final: after `t_1` step, `x_ℓ ~ Cat(P̂^θ(t_1,P_{t1})_ℓ)` independently.

### 11.5 Discrete-diffusion reference (App. I, baselines only)

Bridge `p_{s|0,t}(xs|x0,xt) = Cat(xs; [α_{t|s}e_{xt}+(1−α_{t|s})(e_{xt}ᵀπ)1]⊙[α_s e_{x0}+(1−α_s)π]
/ (α_t(e_{xt}ᵀe_{x0}) + (1−α_t)(e_{xt}ᵀπ)))`, `α_{t|s} = α_t/α_s`.
Masked: unmasked stays, masked → `Cat((α_s−α_t)/(1−α_t)·e_{x0} + (1−α_s)/(1−α_t)·m)`.
Predictor-corrector baseline: predict `x̂0 ~ Cat(x̂θ)`, re-noise `x_{s} ~ Cat(α_s e_{x̂0}+(1−α_s)π)`.
Temperature: `softmax(ζ/T)` in place of `P̂^θ`.

### 11.6 Adaptive time sampler + grids (J.1, phase 2 exact steps)

1. Warmup 1k steps `t ~ U(0,1)`, push `(t, L_t)` to ring buffer.
2. Every 50 steps: 50-bin means of `L_t`, linear interp at bin centers, `q̂(t) = max(0, dL/dt)`
   via finite differences; EMA `q̂_EMA ← 0.9·q̂_EMA + 0.1·q̂`; CDF `F̂` on 1k grid.
3. Train: `u ~ U(0,1)`, `t = F̂^{−1}(u)` (linear interp). Sample: `t_i = F̂^{−1}(i/n)`.
   Linear `t_i = i/n`; cosine `t_i = cos(π/2·(1−i/n))`.

### 11.7 CFG, ELBO, distillation surrogate (phase 2 equations)

- CFG (C.2): drop condition w.p. `p_drop` → `∅`; guided
  `P̃_{0,i} = (P̂^u_i)^{1−w}(P̂^c_i)^w / Σ_j (P̂^u_j)^{1−w}(P̂^c_j)^w`; sample `P_0 ~ P̃_0`
  in place of `P̂^θ` in §11.4.
- ELBO (F.1): `R^k_j = p_{tk|0,tk+1}(·|e_j,P_{k+1})`, `K^θ_k = Σ_j P̂^θ_j·R^k_j`,
  bridge law `Q_{x0} = Dir(P_M;c_1π)∏R^{x0}_k`; `log p_θ(x0) ≥ Σ_{k=1}^M E_{Dir(β_{tk})}[log P̂^θ(t_k,P_k)_{x0}]`.
- DMD (G.1, Alg. 4): forward map `Pt = F_t(P0^s,V)` via Gamma path, differentiable in `P0^s = G_η`;
  Tweedie-for-Dirichlet `∇log p_Y(y) = {(E[α_i(X)|Y]−1)/y_i}`;
  generator gradient flows only through `F_t`:
  `∇ℒ^ = ∫ a_t·DG_ηᵀ·DF_tᵀ·sg(D_φ−D_teach)/Pt`;
  surrogate `ℒ̂ = ∫ a_t·log(Pt)ᵀ·sg(D_φ−D_teach)`; auxiliary `ℒ_φ = −Σ_k log D_φ(sg(Pt))_k·sg(P0^s)_k`.
  IDLM difference: IDLM drops the indirect (`DF_t`) term heuristically; SDM keeps exactly it.

### 11.8 Logit-shaping stack (K.5, OWT phase 2)

Per position with logits `{ζ_i}`: `TOP_p` (sorted nucleus mask to `−∞`);
anneal `T(t) = T_start+(T_end−T_start)·t^{1.5}`, `T_start = 0.8·T_end`;
sequence penalty `ζ_i − λ·log(1+c_{t,i})`, `c` = argmax-histogram (185) for SDM;
local penalty `ζ_i − γ·ĉ_{t,j,i}`, `ĉ = log(1+α_t N/(1−α_t))·δ(argmax)` (188–189).
Sweep `T_end ∈ §K.5 list`, `p ∈ {0.92,0.96}`, `λ ∈ {1.5,3,4.5,6,8}`, `γ ∈ {2.0,2.5}` (SDM only).

---

## References

- Deschenaux et al. 2026 (SDM paper + supplement: Algs. 1–4, Props. 3.1–3.3/4.1/A.1–A.2/E.1–E.4/F.1/G.1–G.2, §C–K).
- Peebles & Xie 2022 (DiT); Ho et al. 2020 (DDPM); Song et al. 2021 (DDIM); Stark et al. 2024 (DFM);
  Sahoo et al. 2024 / Shi et al. 2024 (MDM); Austin et al. 2021 / Campbell et al. 2022 (DDM).
- In-tree: `models/vision/dit/{README,PORT_NOTES,model,blocks,diffusion,config}.py`,
  `models/vision_language/bit_diffusion/{sde,bridge_process}.py`,
  `models/language/bert/model.py`, `models/common/masked_language_model/README.md`,
  `losses/{ddpm_hybrid_loss,flow_matching_velocity_loss,masked_causal_lm_loss}.py`,
  `utils/ddpm_schedule.py`, `layers/{embedding/factory,transformers/sd3_adaln}.py`,
  `src/train/{dit/train_dit.py,AGENTS.md}`, `models/{AGENTS,GUIDE,README}.md`.
