# Roadmap: a Composable LLM Building-Block API for `dl_techniques`

Target location in the repo: `research/roadmap/2026_llm_building_blocks.md`
Scope of this roadmap: **embeddings** and **language models** first, every other domain as a reserved extension point.

"Embeddings" is read in both senses this repo uses: (a) the **input embedding stage** of any sequence model (token, scale, absolute position, embedding norm) and (b) **text embedding models** (backbone + pooling + Matryoshka + L2), as in `research/2025_embedding_models_research.md` and `models/language/qwen/qwen3_embeddings.py`.

---

## 0. Problem statement

**One sentence:** swapping one component of a language model in `dl_techniques` (GQA to MLA, dense FFN to MoE, attention to Mamba-2 / Gated DeltaNet, standard residual to hyper-connections) today means editing or forking a model class, so ablations cost days instead of a config diff.

Measured leaks at HEAD (`e4317c1`, 2026-11-12):

| Leak | Evidence |
|---|---|
| Every language model owns its own stack loop | 20 leaf packages under `models/language/`; only 7 compose `TransformerLayer` / `TextDecoder` / `TextEncoder` at all |
| Token embedding hand-rolled per model | `keras.layers.Embedding(` constructed directly at **14** sites under `models/language/`; `EMBEDDING_REGISTRY` (16 keys) has **no** plain token key |
| No mixer abstraction beyond softmax attention | `TransformerLayer` takes `attention_type` only; SSM (`SSM_REGISTRY`, 4 keys), KDA, `GatedLinearAttentionBlock` live in separate call contracts; hybrids ship bespoke blocks (`Qwen3NextBlock`, `Gemma3TransformerBlock`, `Mamba2ResidualBlock`, `Zamba2SharedAttentionBlock`) |
| Per-attention-type parameter translation is a hand ladder | `TransformerLayer._get_attention_params` has **11** `elif self.attention_type` branches |
| No residual-stream abstraction | zero files implement hyper-connections, mHC, AttnRes, value residual |
| No decode-time state | zero KV-cache / recurrent-state implementations in `layers/` |
| Placeholders waiting for a cheaper way | `models/language/gemma/gemma1.py`, `gemma2.py`, `gemma4.py` are **0 lines** |
| Ablation axes from `research/2026_llm_architecture_mega_ablation.md` not expressible | of ~40 axes in its section 5, roughly half need new code and none is a one-line change today |

## 1. Success criteria (measurable)

| ID | Criterion | Measured by |
|---|---|---|
| S1 | Any single ablation axis in the coverage matrix (Appendix A) is a diff confined to one spec field, at most 5 lines | review of the sweep YAMLs in `src/train/composable_ablation/` |
| S2 | Existing models re-expressed as presets reproduce the original model's logits after weight transfer: max abs diff `<= 1e-5` (float32, TF32 off) | parity tests per preset (Phase 3) |
| S3 | Zero regressions in existing suites touched (`test_transformers/`, `test_attention/`, `test_moe/`, `test_ssm/`, `test_embedding/`, `test_norms/`, `test_models/test_qwen*`, `test_gpt2`, `test_gemma*`, `test_mamba`, `test_modern_bert`) | scoped pytest per `AGENTS.md` |
| S4 | Step time overhead of a spec-built model vs its hand-written counterpart `<= 3%` (target, not estimate) under `jit_compile=True`, `mixed_float16` | benchmark harness (Phase 6) |
| S5 | `.keras` save/load round trip and weight-path stability for every preset and for a randomized spec sample | existing `roundtrip_instrument_oracle`, `test_a_language_model_weight_paths_are_stable.py` |
| S6 | A spec-built LM trains through `CausalLanguageModel` and `python -m train` with no new trainer code; an embedder trains through a `text_embedding` task | Phase 5 smoke runs |
| S7 | Capability claims (causal-safe, padding-mask honored, fp16-safe) are **measured** and pinned by tests, never declared | Phase 0 capability table + guard tests |

---

## 2. Reuse inventory (what already exists, what the API needs, the gap)

This table is the backbone of the plan: every API concept maps to an existing asset first.

| API concept | Existing asset (path) | Reuse mode | Gap |
|---|---|---|---|
| Attention mixers | `ATTENTION_REGISTRY` (37 keys), `create_attention_layer` (`layers/attention/factory.py`) | wrap via mixer adapter | no sliding-window GQA, no KV conv, no sinks, no value-residual hook |
| SSM mixers | `SSM_REGISTRY` (4 keys: `selective_ssm`, `context_mamba`, `mamba`, `mamba2`), `create_ssm_layer` (`layers/ssm/factory.py`) | wrap via mixer adapter | `SSM_REGISTRY` missing from `REPO_MAP.md` registry table |
| Linear-attention mixers | `kimi_delta` in `ATTENTION_REGISTRY` (`kimi_delta_attention.py`), `GatedLinearAttentionBlock` (`layers/transformers/gated_linear_attention_block.py`) | wrap | Gated DeltaNet proper (scalar per-head decay delta rule) absent; Qwen3-Next uses GLA, not GDN |
| Dense FFN | `FFN_REGISTRY` (22 keys: `swiglu`, `geglu`, `squared_relu`, `mlp`, `gelu_tanh`, `glu`, `reglu`, ...), `create_ffn_from_config`, `assemble_ffn_config`, `build_transformer_ffn_config` | direct | none |
| MoE | `layers/moe/` (`MoEConfig`, `ExpertConfig`, `GatingConfig`, `MixtureOfExperts`; gating `linear` / `cosine` / `softmoe`; sparse kernel gated against dense oracle; aux + z loss via `add_loss`) | direct | sigmoid routing + aux-loss-free bias, shared experts, LatentMoE, null experts |
| Norms | `create_normalization_layer` (18 types, `_TYPE_TO_CLASS`, signature-derived validation) | direct | placement semantics (pre/post/sandwich) live only inside `TransformerLayer` |
| QK-norm | `create_qk_norms` in `layers/attention/common.py`; `qk_norm_type` on most attention keys | direct | none |
| RoPE | `RotaryPositionEmbedding` (partial via `rope_percentage`), `dual_rope`, `continuous_rope`, `multi_axis_rope`, `mrope_ideogram4`; GQA/MLA/gated take `rope_theta`, `rope_percentage` | translate `PositionSpec` to these kwargs | YaRN absent; ALiBi absent; NoPE path unclear (GQA docstring example uses `rope_percentage=0.0`, `RotaryPositionEmbedding` rejects 0); default divergence: `RotaryPositionEmbedding` defaults `rope_percentage=0.5`, GQA defaults `1.0` |
| Token embedding | `albert_factorized`, `bert_embeddings`, `modern_bert_embeddings` in `EMBEDDING_REGISTRY`; `hierarchical_codebook_embedding.py`; `TextDecoder` (`learned` / `factorized`) | register token keys, compose | plain token key; `sqrt(dim)` scale exists only inside `gemma3.py` |
| Absolute positions | `positional_learned`, `PositionalEmbedding`, sincos in `TextDecoder` | direct | none |
| Residual modifiers | `StochasticDepth`, `LayerScale` (`layers/regularization/`), `utils/drop_path.linear_drop_path_rates`, `residual_output_kernel_initializer` on `TransformerLayer` | direct | stream topologies (HC, mHC, AttnRes, value residual, gated residual) |
| Homogeneous block | `TransformerLayer` (`layers/transformers/transformer.py`), `TRANSFORMER_BLOCK_REGISTRY` (12) | parity reference, later delegate | attention-only mixer slot |
| Causal stack skeleton | `layers/transformers/causal_stack.py` (`CausalTransformerStackMixin`) | pattern reference | homogeneous, `LayerNormalization` final norm at `1e-3` |
| Masks | `utils/masking/` (`create_causal_attend_mask`), `apply_attention_mask` (`attention/common.py`, fp16-safe) | direct | sliding-window and prefix-LM mask builders to confirm in `utils/masking/` |
| LM head, tying | `utils/tied_embeddings.tied_embedding_logits`; `losses/harmonic_loss.harmonic_logits` (used by GPT2) | direct | logit soft-cap |
| Task heads | `layers/heads/nlp/` (`build_model_task_head`, `validate_task_head_knobs`, `finalize_head_outputs`, `TextGenerationHead`, `text_similarity`) | direct | none |
| Sequence pooling | `SEQUENCE_POOLING_REGISTRY` (`attention`, `mode`, `weighted`, `sequence`); `sequence` strategies include `mean`, `cls`/`first`, `last`, `max`, `mean_max`, `top_k_mean`, `attention` | direct | audit `last` under left padding |
| Model plumbing | `utils/model_build` (`materialize_sublayers`, `concretize_axes`, `refuse_unknown_model_kwargs`), `utils/sequence_length_guard.RecordedBuildShapeMixin`, `utils/weight_transfer` (`load_weights_or_raise`, `TransferReport`, `load_pretrained_checkpoint`, `refuse_pretrained`), `utils/keras_registration.register_dl_technique` | direct | none |
| CLM training | `models/common/lm_pretraining/clm.py` `CausalLanguageModel` (backbone contract: `hidden_size` attribute + mapping with `last_hidden_state`; future-leak probe; `aggregate_backbone_losses`) | ComposableLM satisfies the contract | none |
| MLM training | `models/common/lm_pretraining/mlm.py` | encoder embedders pretrain | none |
| Trainer library | `train/common/clm_pretrain.py`, `train/common/nlp.py` (`preprocess_clm_packed_dataset`, `build_clm_metrics`, ...), `generation_probe.py`, `compare_runs.py`, `step_checkpoint.py` | direct | none |
| CLI | `train/api` (lazy registry, `StrictConfig`, `register_model`, `register_task`, `exporting`) | register families + tasks | tasks are image-only today (`image_classification`, `image_denoising`, `image_segmentation`) |
| Ablation harness patterns | `train/rms_variants_train/` (`sweep.py`, `hypotheses.py`, `stats.py`, `seed_utils.py`, `norm_overhead_bench.py`, `report.py`), `train/mean_shift/` (gated subprocess per cell, `rows.jsonl`) | copy the shape | none |
| Metrics | `metrics/perplexity_metric.py`, `metrics/llm_metrics.py` (`BitsPerToken`, `BitsPerCharacter`, `self_bleu`, `distinct_n`) | direct | none |
| Losses | `MaskedCausalLMLoss`, `FocalCausalLMLoss`, `infonce_loss.py`, `clip_contrastive_loss.py`, `siglip_contrastive_loss.py` | direct | Matryoshka loss wrapper |
| Test oracles | `tests/test_models/{smoke_contract,lazy_build_contract,roundtrip_instrument,gradient_flow,knob_sensitivity,initializer_replay,precision_arm}_oracle.py`, the `test_a_language_*` family, `test_layers/test_factory_registry_drift.py`, fixtures `golden_reference_device`, `dtype_policy`, `tf32_disabled` | enroll new subjects | none |

---

## 3. Architecture

### 3.1 Layering

```
L7  User facade          dl_techniques.layers.composable.nn      (nn.LM, nn.Embedder, nn.Block, nn.GQA, nn.MoE, ...)
L6  Models               models/language/composable_lm/          ComposableLM       (+ presets = MODEL_VARIANTS)
                         models/language/composable_embedder/    ComposableEmbedder (+ presets)
L5  Stages               EmbeddingStage, LMHeadStage, EmbeddingHeadStage
L4  Stack                BlockStack: schedule, sharing, mask cache, residual stream driver, final norm
L3  Block                ComposableBlock: [pre_mixer_conv] mixer [pre_ffn_conv] ffn | norm placement | residual
L2  Registries/adapters  MIXER_REGISTRY + MixerAdapter, RESIDUAL_REGISTRY, PositionSpec resolver, capability table
L1  Specs (pure data)    frozen dataclasses, JSON-serializable, schema_version, path-qualified validation
L0  Existing primitives  ATTENTION / SSM / FFN / MoE / norms / EMBEDDING / SEQUENCE_POOLING factories, utils
```

Rule: L2 to L7 contain **no math** that already exists in L0. New math (GDN, YaRN, HC, LatentMoE, short conv) goes into the proper L0 subpackage and its registry, per `models/AGENTS.md` "Layer Reuse Policy".

### 3.2 Target user API (the blog idea, in this repo's vocabulary)

```python
from dl_techniques.layers.composable import nn

block = nn.Block(
    mixer=nn.GQA(num_heads=16, num_kv_heads=4,                 # or nn.MLA, nn.KDA, nn.GatedDeltaNet, nn.Mamba2
                 position=nn.RoPE(theta=1e6), qk_norm="rms_norm"),
    ffn=nn.MoE(expert=nn.SwiGLU(hidden_dim=4096),              # or nn.SwiGLU(...) dense
               num_experts=64, top_k=2),
    norm=nn.RMSNorm(epsilon=1e-6, placement="pre"),            # or nn.LayerNorm, placement="sandwich"
    residual=nn.Residual(),                                    # or nn.HyperConnections(n=4), nn.MHC(n=4), nn.AttnRes()
)

model = nn.LM(
    vocab_size=128_000, dim=2048, max_seq_len=8192,
    embedding=nn.TokenEmbedding(scale=None),
    layers=nn.repeat(block, 24),
    head=nn.LMHead(tie=True),
)                                                              # -> ComposableLM (keras.Model)

# Hybrid 3:1, NoPE on global layers (2026 hybrid pattern)
local  = block.replace(mixer=nn.KDA(num_heads=16))
glob   = block.replace(mixer=nn.GQA(num_heads=16, num_kv_heads=4, position=nn.NoPE()))
hybrid = nn.LM(vocab_size=128_000, dim=2048, layers=nn.pattern([local, local, local, glob], repeats=6))

# Text embedder on the same blocks
emb = nn.Embedder(
    vocab_size=151_936, dim=1024,
    layers=nn.repeat(block.replace(ffn=nn.SwiGLU(hidden_dim=2816)), 12),
    attention_mode="bidirectional",
    head=nn.EmbeddingHead(pooling="last", matryoshka_dims=(64, 128, 256, 512, 1024), normalize=True),
)

spec = model.spec            # ModelSpec, pure data; spec.to_config() is what get_config() stores
nn.accounting(spec)          # params, active params, FLOPs/token, KV bytes/token, recurrent state bytes
```

Vocabulary rule: facade argument names are the **canonical registry names** (`num_heads`, `num_kv_heads`, `hidden_dim`, `num_experts`, `top_k`). No third vocabulary of aliases. Model-level context (`dim`, `max_seq_len`, init policy, dtype policy, layer index) is injected at resolution, never repeated per block.

### 3.3 Specs (L1)

```python
@dataclass(frozen=True)
class PositionSpec:            # translated per mixer by positions.py
    type: Literal["rope", "nope", "yarn", "alibi", "mrope"] = "rope"
    theta: float = 10_000.0
    fraction: float = 1.0      # -> rope_percentage
    scaling: Optional[Mapping[str, Any]] = None   # YaRN factor, original_max_seq_len, beta_fast/slow

@dataclass(frozen=True)
class MixerSpec:
    type: str                  # MIXER_REGISTRY key
    params: Mapping[str, Any]
    position: Optional[PositionSpec] = None
    attention_mode: Literal["causal", "bidirectional", "sliding", "prefix"] = "causal"
    window: Optional[int] = None

@dataclass(frozen=True)
class FFNSpec:   type: str; params: Mapping[str, Any]                       # FFN_REGISTRY key
@dataclass(frozen=True)
class MoESpec:   expert: FFNSpec; num_experts: int; top_k: int
                 gating: str = "linear"; shared_experts: int = 0; null_experts: int = 0
                 latent_dim: Optional[int] = None; aux_loss_weight: float = 0.01; z_loss_weight: float = 1e-3
@dataclass(frozen=True)
class NormSpec:  type: str; epsilon: float; placement: Literal["pre", "post", "sandwich", "none"] = "pre"
@dataclass(frozen=True)
class ResidualSpec: type: str = "standard"; params: Mapping[str, Any] = {}
                    layer_scale: Optional[float] = None; drop_path: Optional[float] = None
@dataclass(frozen=True)
class BlockSpec: mixer: MixerSpec; ffn: Optional[Union[FFNSpec, MoESpec]]; norm: NormSpec
                 residual: ResidualSpec = ResidualSpec()
                 pre_mixer_conv: Optional[Mapping] = None    # Canon-A
                 pre_ffn_conv: Optional[Mapping] = None      # Canon-C
                 share_key: Optional[str] = None             # weight sharing (Zamba2)
                 cross_mixer: None = None                    # RESERVED v1 (encoder-decoder)
                 conditioning: None = None                   # RESERVED v1 (AdaLN / FiLM)
@dataclass(frozen=True)
class EmbeddingSpec: token: str = "learned"; token_params: Mapping = {}
                     scale: Union[None, Literal["sqrt_dim"], float] = None
                     absolute_position: Optional[Literal["learned", "sincos"]] = None
                     norm: Optional[NormSpec] = None; dropout: float = 0.0
@dataclass(frozen=True)
class LMHeadSpec: tie: bool = True; bias: bool = False; softcap: Optional[float] = None
                  logits: Literal["linear", "harmonic"] = "linear"
@dataclass(frozen=True)
class EmbeddingHeadSpec: pooling: str = "mean"; pooling_params: Mapping = {}
                         projection_dim: Optional[int] = None
                         matryoshka_dims: Optional[Tuple[int, ...]] = None; normalize: bool = True
@dataclass(frozen=True)
class InitSpec: std: float = 0.02; residual_scaling: Literal["none", "depth"] = "none"; seed: Optional[int] = None
@dataclass(frozen=True)
class ModelSpec: schema_version: int; vocab_size: int; dim: int; max_seq_len: int
                 embedding: EmbeddingSpec; layers: Tuple[BlockSpec, ...]; final_norm: NormSpec
                 head: Union[LMHeadSpec, EmbeddingHeadSpec]; init: InitSpec
```

Spec rules:

1. Frozen, hashable, `to_config()` / `from_config()` producing plain JSON (no Keras objects). A spec hash names a run directory.
2. **Every numerically meaningful default is explicit in presets** (epsilon, rope fraction, init std). The repo has three documented default traps: the norm factory's imposed `1e-6` epsilon, `causal_stack.py`'s `1e-3` final norm, and the `0.5` vs `1.0` `rope_percentage` defaults. `NormSpec.epsilon` therefore has no default.
3. Validation derives accepted kwargs from the **target class constructor** (reuse the `_constructor_keywords` pattern in `layers/transformers/factory.py` and `_accepted_params` in `layers/norms/factory.py`), never from curated lists (the norm factory docstring records that curated lists were wrong for 19 of 19 types).
4. Errors are path-qualified: `layers[7].mixer.params.num_kv_heads: 5 does not divide num_heads=16`.
5. `schema_version` is stored; reserved fields raise `NotImplementedError("schema v1")` when set, so filling them later is not a breaking change.

### 3.4 Mixer contract (L2)

```python
class MixerAdapter(keras.layers.Layer):
    """Uniform call: (x[B,S,D], attention_mask, training) -> y[B,S,D]."""
```

`MIXER_REGISTRY` entries point at **existing registry keys**, not classes:

```python
MIXER_REGISTRY = {
  "mha":            {"source": "attention", "key": "multi_head",        "param_map": {...}},
  "gqa":            {"source": "attention", "key": "group_query",       ...},
  "mqa":            {"source": "attention", "key": "group_query",       "fixed": {"num_kv_heads": 1}},
  "mla":            {"source": "attention", "key": "multi_head_latent", ...},
  "gated":          {"source": "attention", "key": "gated",             ...},
  "differential":   {"source": "attention", "key": "differential",      ...},
  "kda":            {"source": "attention", "key": "kimi_delta",        ...},
  "gated_deltanet": {"source": "attention", "key": "gated_deltanet",    ...},   # Phase 4
  "gla":            {"source": "class",     "key": "GatedLinearAttentionBlock", ...},
  "mamba":          {"source": "ssm",       "key": "mamba",             ...},
  "mamba2":         {"source": "ssm",       "key": "mamba2",            ...},
  "short_conv":     {"source": "ssm",       "key": "short_conv",        ...},   # Phase 4 (LFM2)
  "linear":         {"source": "attention", "key": "linear",            ...},
  "performer":      {"source": "attention", "key": "performer",         ...},
  "lighthouse":     {"source": "attention", "key": "lighthouse",        ...},
  "wave_field":     {"source": "attention", "key": "wave_field",        ...},
  "fnet":           {"source": "attention", "key": "fnet",              ...},   # bidirectional only
}
```

Each entry carries a **measured** `MixerCapabilities` row (generated in Phase 0, pinned by a test, rendered into the module docstring the way `ATTENTION_REGISTRY`'s table is generated rather than transcribed):

| Field | Values | How measured |
|---|---|---|
| `causal` | `mask` / `internal` / `unsupported` | future-leak probe (extracted from `CausalLanguageModel._verify_backbone_causality`) |
| `bidirectional` | bool | probe with a full mask |
| `padding_mask` | `honored` / `ignored` / `n/a` | right- and left-padding equivalence test (see `2026_attention_open_design_questions.md` Q1 to Q4: `mobile_mqa`, `shared_weights_cross`, `anchor` ignore or mishandle masks) |
| `positional` | `rope_kwargs` / `implicit` / `none` | introspection |
| `state` | `kv` / `recurrent` / `conv` / `none` | declared, then verified in Phase 6 |
| `fp16` | `ok` / `refuses` | `dtype_policy` fixture (`rpc` cannot run under `mixed_float16`, Q5) |
| `static_seq_len` | bool | GLA has static vs symbolic dispatch |
| `domains` | `{"text"}`, `{"vision"}`, ... | vision-only keys (`cbam`, `channel`, `spatial`, `window`, `beit`, `area`, `tripse*`, ...) are excluded from the text facade, kept for provisions |

The adapter replaces the 11-branch ladder in `TransformerLayer._get_attention_params` with one data-driven `param_map`. Migration of `TransformerLayer` onto it happens only in Phase 7, behind parity gates.

Mask handling: `BlockStack` builds each distinct mask once per forward (causal, bidirectional, sliding window `w`, prefix) through `utils/masking/` and hands each adapter the one its `attention_mode` needs. Recurrent mixers that are causal by construction receive the padding mask; the adapter zeroes padded positions before the scan so left padding cannot leak into state.

### 3.5 Residual stream protocol (L2)

The stack owns the stream; each block owns its per-layer residual parameters.

```python
class ResidualStream(keras.layers.Layer):
    def init_stream(self, h):           ...   # (B,S,D) -> stream
    def read(self, stream, branch):     ...   # stream -> branch input (B,S,D); branch in {"mixer","ffn"}
    def write(self, stream, y, branch): ...   # merge branch output
    def finalize(self, stream):         ...   # stream -> (B,S,D)
```

| Type | Stream | read | write | Identity test |
|---|---|---|---|---|
| `standard` | `h` | `h` | `h + y` | reference |
| `hyper_connections` (Zhu et al. 2024) | `(B,S,n,D)` | learned width mix | learned depth mix + `B`-weighted write | `n=1`, identity init equals `standard` bitwise |
| `mhc` (manifold-constrained HC) | `(B,S,n,D)` | as HC | mixing matrix projected to doubly stochastic (Sinkhorn) | Sinkhorn output rows and cols sum to 1; `n=1` equals `standard` |
| `attn_res` (Attention Residuals) | list of prior layer outputs | softmax-weighted depth attention | append | uniform weights equal running mean |
| `value_residual` (ResFormer) | `h` + first-layer `V` | needs mixer hook `exposes_values` | `standard` | mixing weight 0 equals `standard` |
| `gated` | `h` | `h` | `h + g * y` | `g=1` equals `standard` |

Norm placement composes with the stream: `pre` = norm on `read`; `post` = norm after `write` (valid only for single-tensor streams, validator rejects `post` + HC); `sandwich` = norm on `read` and on `y`. `layer_scale` and `drop_path` reuse `LayerScale` and `StochasticDepth` on `y`. The ablation study's note that an mHC init bug collapsed four streams into one becomes a permanent guard test: after init, stream-to-stream correlation `< 1`.

### 3.6 Embedding stage and output stages (L5)

`EmbeddingStage` = token lookup (registry) -> optional scale (`sqrt_dim` covers Gemma) -> optional absolute position -> optional norm -> dropout. It exposes `embedding_weights` so `tied_embedding_logits` and `CausalLanguageModel._locate_embedding_weights` find the table without model-specific code.

New `EMBEDDING_REGISTRY` keys (the key set is pinned public API: update the pin tests in the same commit):

| Key | Backing |
|---|---|
| `token_learned` | `keras.layers.Embedding` |
| `token_factorized` | `albert_factorized` |
| `token_hierarchical_codebook` | `hierarchical_codebook_embedding.py` |
| `token_superposition` | provision; promote from `train/common/token_superposition.py` after Phase 5 |

`LMHeadStage`: tied (`tied_embedding_logits`) or untied dense; optional bias; optional soft-cap `c * tanh(z / c)` computed in float32; optional `harmonic_logits`; task heads via `build_model_task_head` when `include_top` / `task_config` are given (same knobs and `validate_task_head_knobs` as `Qwen3`).

`EmbeddingHeadStage`: `create_sequence_pooling_layer` -> optional projection -> Matryoshka truncation set -> L2 normalize. Returns a mapping `{"embedding": ..., "matryoshka": {d: ...}}` so a Matryoshka loss can read all prefixes in one pass.

### 3.7 Stack and schedules (L4)

`BlockStack(layers: Tuple[BlockSpec, ...])` with helpers in `schedules.py`:

| Helper | Meaning | Precedent |
|---|---|---|
| `nn.repeat(block, n)` | homogeneous stack | `TransformerLayer` loops |
| `nn.pattern(unit, repeats)` | hybrid ratios (3:1 KDA:GQA, 5:1 SWA:global) | `Gemma3`, `Qwen3Next` |
| `nn.share(block, key)` | same weights at several depths | `Zamba2` shared attention + LoRA (`LoRAAdapter`) |
| `nn.override(i, **fields)` | per-layer edits (MoE on some layers) | `Qwen3(moe_layers=...)` |
| `drop_path="linear"` | depth-scaled drop path | `utils/drop_path.linear_drop_path_rates` |

Validation: the residual type is uniform across the stack (stream shape must not change mid-stack); every `causal` LM layer has a mixer whose measured `causal` capability is not `unsupported`.

### 3.8 Inference state (provision in v1, built in Phase 6)

```python
class MixerState:  # per layer, a nested structure of tensors
    kind: Literal["kv", "recurrent", "conv", "none"]
MixerAdapter.step(x_t, state, position) -> (y_t, state)     # decode
MixerAdapter.prefill(x, mask) -> (y, state)                 # prompt
```

`kv`: preallocated `(B, kv_heads, S_max, head_dim)` with a write index (sliding layers keep a ring buffer of `window`). `recurrent`: KDA / GDN / Mamba-2 / GLA matrix states plus their short-conv tail. `conv`: last `k-1` inputs. Training path stays single-tensor so functional build and serialization are unaffected (ADR-9).

---

## 4. Phased plan

Estimates are ranges in engineer-weeks for one engineer familiar with the repo; they are estimates, not commitments. Each phase lists reuse first.

### Phase 0: Measure and decide (1 to 2 weeks)

| ID | Task | Output |
|---|---|---|
| T0.1 | Open a plan id; record decisions as `# DECISION <plan-id>/D-NNN` anchors registered in `plans/ANCHORS.md` | plan dir |
| T0.2 | Capability audit of every text-eligible `ATTENTION_REGISTRY` / `SSM_REGISTRY` key plus `GatedLinearAttentionBlock`: causal probe, padding (left and right), fp16, static length | `capabilities.py` generated table |
| T0.3 | Call-signature census: mask kwarg name, `training` position, return type (Mamba2 residual block returns `(out, residual)`) | `param_map` / `call_map` per key |
| T0.4 | Default census: epsilon, `rope_percentage`, `rope_theta`, init per class vs factory | defaults table pinned into presets |
| T0.5 | Verify NoPE path for GQA / gated / MLA (`rope_percentage=0.0`) | fix or add explicit `use_rope` flag |
| T0.6 | Confirm sliding-window and prefix mask builders in `utils/masking/`; list what to add | gap list |
| T0.7 | Write ADRs (Section 6) and confirm package names | ADR file in plan dir |
| T0.8 | Pick parity targets and their reference checkpoints (local `results/` only; nothing distributed) | preset list |

Exit: capability table reviewed; ADRs accepted; no code outside `plans/`.

### Phase 1: Specs, registries, adapters (2 to 4 weeks)

New: `layers/composable/{spec,context,validation,mixers,capabilities,positions,residuals}.py`, residual `standard` only.

| ID | Task | Reuse |
|---|---|---|
| T1.1 | Spec dataclasses + `to_config` / `from_config` + `schema_version` | `utils/activation_serialization`, `initializer_spec` convention |
| T1.2 | Constructor-introspection validator with path-qualified errors | `transformers/factory._constructor_keywords`, `norms/factory._accepted_params` |
| T1.3 | `MIXER_REGISTRY` + `MixerAdapter` (training path) | `create_attention_layer`, `create_ssm_layer` |
| T1.4 | `positions.py`: `PositionSpec` -> per-mixer kwargs; refuse on mixers with `positional="none"` unless `nope` | attention kwargs `rope_theta`, `rope_percentage` |
| T1.5 | `residuals.py` protocol + `standard`; `LayerScale`, `StochasticDepth` modifiers | `layers/regularization/` |
| T1.6 | Extract the future-leak probe from `CausalLanguageModel` into `utils/` as `assert_causal(layer_or_model, ...)`; keep the model calling it | `models/common/lm_pretraining/clm.py` |
| T1.7 | Extend `tests/test_layers/test_factory_registry_drift.py` to `MIXER_REGISTRY` and `RESIDUAL_REGISTRY` | existing drift test |

Tests: spec round trip (property-based over a random spec sampler); every adapter key: build, `compute_output_shape` equals call shape, causal probe, padding equivalence, `dtype_policy` arms; refusal tests for bad combos.

Exit: every Phase-1 mixer key builds through the adapter and passes its measured capability guards.

### Phase 2: Block, stack, stages, models, facade (3 to 5 weeks)

New: `layers/composable/{block,stack,schedules,embedding_stage,output_stage,accounting,nn}.py`; leaf packages `models/language/composable_lm/` and `models/language/composable_embedder/`.

| ID | Task | Reuse |
|---|---|---|
| T2.1 | `ComposableBlock`: optional pre-mixer conv, mixer, optional pre-FFN conv, FFN (`create_ffn_from_config`) or MoE (`MixtureOfExperts` from `MoEConfig`), norm placement, residual protocol; sub-layers created unconditionally in `__init__`, built in `build` (golden rules, `research/2026_keras_custom_models_instructions_v2.md` sections 1.2, 1.3) | `TransformerLayer` internals as reference |
| T2.2 | `BlockStack`: schedules, sharing, mask cache, final norm | `causal_stack.py` pattern, `utils/masking` |
| T2.3 | `EmbeddingStage` + new token keys in `EMBEDDING_REGISTRY` | `albert_factorized`, `hierarchical_codebook_embedding`, `PositionalEmbedding` |
| T2.4 | `LMHeadStage`, `EmbeddingHeadStage` | `tied_embedding_logits`, `harmonic_logits`, `heads/nlp/model_head.py`, `create_sequence_pooling_layer` |
| T2.5 | `ComposableLM(keras.Model)` following the house model shape: `@register_dl_technique("dl_techniques.models.composable_lm.model")`, `RecordedBuildShapeMixin`, `materialize_sublayers`, `refuse_unknown_model_kwargs`, `include_top` / `task_config` / `head_config_overrides`, returns `{"logits", "last_hidden_state"}`, exposes `hidden_size` (satisfies `CausalLanguageModel`), `pretrained=True` raises via `refuse_pretrained` | `qwen3.py` as the closest template |
| T2.6 | `ComposableEmbedder(keras.Model)` with `attention_mode` override at stack level | `qwen3_embeddings.py` |
| T2.7 | `accounting.py`: total and active params, FLOPs/token, KV bytes/token/layer, recurrent state bytes; cross-checked against `count_params()` in tests | ablation doc section 3.10 "equal serving cost" |
| T2.8 | `nn.py` facade; `nn.LM` / `nn.Embedder` return models, sub-constructors return specs | none |
| T2.9 | `create_composable_lm()` / `create_composable_embedder()` factories, curated `__all__`, `README.md` per leaf | house shape |

Tests: enroll both models in `smoke_contract`, `lazy_build_contract`, `roundtrip_instrument`, `gradient_flow`, `knob_sensitivity` (every spec field changes output or is declared inert), `initializer_replay`, `precision_arm` oracles and the `test_a_language_*` family; pairwise covering array over {mixer x ffn x norm placement x residual x position} instead of the full cross product.

Exit: M1 (see Section 8) and both models round-trip through `.keras`.

### Phase 3: Parity presets (4 to 7 weeks)

Presets live in `models/language/composable_lm/presets.py` as `MODEL_VARIANTS`; weight maps in `parity.py` produce a `TransferReport` via `utils/weight_transfer`. Original models are **not** modified.

| Order | Target | Exercises | Notes |
|---|---|---|---|
| 1 | `qwen3` dense | GQA, RMSNorm pre, SwiGLU, RoPE, untied head | built on `TransformerLayer`: smallest delta |
| 2 | `qwen3` MoE (`moe_layers`) | `nn.override`, MoE, aux losses | `CausalLanguageModel(aggregate_backbone_losses=True)` |
| 3 | `gpt2` | MHA, learned positions, tied head, `harmonic_logits` option | via `TextDecoder` |
| 4 | `qwen3_embeddings` | embedder, bidirectional, pooling, truncation | first embedder parity |
| 5 | `modern_bert` | window/global alternation (`window_band`), encoder | embedder backbone |
| 6 | `gemma3` | sliding/global pattern, `sqrt_dim` scale | ends Gemma placeholders path |
| 7 | `mamba2` | SSM mixer, fused add-norm stream | `Mamba2ResidualBlock` returns `(out, residual)`: equals `standard` stream with pre-norm, prove it |
| 8 | `qwen3_next` | GLA + gated attention + MoE hybrid | |
| 9 | `zamba2` | shared block + LoRA (`nn.share`) | last: needs sharing |

Parity test template: build original, build preset, transfer weights by map, forward the same ids on `golden_reference_device` with TF32 off (`tests/numerics.true_float32_matmul()`), assert `max|dlogits| <= 1e-5`; also `mixed_float16` at a looser bound.

Then fill the empty Gemma files as presets (`gemma1`, `gemma2` with soft-cap, `gemma4`) instead of new classes, if the user wants those models at all.

Exit: M2; parity green for orders 1 to 7 at minimum.

### Phase 4: Missing components for the ablation axes (6 to 10 weeks, prioritized by expected effect in the mega ablation)

Priority follows the study's "best individual changes" table (fine-grained MoE and linear-attention hybrids first, cosmetic changes last).

| Prio | Component | Home (L0) | Reuse | Equivalence test |
|---|---|---|---|---|
| P1 | MoE sigmoid routing + aux-loss-free bias (DeepSeek-V3) | `layers/moe/gating.py` new `sigmoid` gating; bias-update callback beside it | `BaseGating`, `create_gating` | top-k indices equal `linear` gating when bias is 0 and scores monotone |
| P1 | Shared experts | `layers/moe/layer.py` + `ExpertConfig` | `create_ffn_from_config` | `num_experts=1, top_k=1, shared=0` equals dense FFN |
| P1 | Fine-grained expert sizing helper | `accounting.py` | | equal active FLOPs vs dense |
| P1 | Gated DeltaNet | generalize `kimi_delta_attention.py` decay granularity (`channel` vs `head`) and register `gated_deltanet` | KDA scan, chunking, short conv | `decay="head"` with tied channels equals KDA with uniform `alpha` |
| P1 | Sliding-window GQA | mask builder in `utils/masking/` + `attention_mode="sliding"` | `apply_attention_mask` | `window >= seq_len` equals full causal |
| P2 | LatentMoE | `layers/moe/layer.py` down/up projection around experts | `MixtureOfExperts` | `latent_dim=dim`, identity projections equals plain MoE |
| P2 | Null experts | gating option | | `null_experts=0` equals current |
| P2 | YaRN | `layers/embedding/rotary_position_embedding.py` `scaling` | RoPE | `factor=1` equals RoPE |
| P2 | NoPE per layer | fix from T0.5 | | |
| P2 | Hyper-connections, mHC | `layers/composable/residuals.py` | | Section 3.5 identities |
| P2 | AttnRes | `residuals.py` | | uniform weights equal mean |
| P2 | Shared causal depthwise short conv primitive | extract from KDA / Mamba-2 into one layer (location confirmed in T0.7; candidate `layers/conv_blocks/`) | existing conv code | outputs equal the inline versions |
| P2 | KV conv (Inkling) | GQA option `kv_conv_size` using the primitive | | size 1 identity kernel equals plain GQA |
| P3 | Canon-A / Canon-C | `BlockSpec.pre_mixer_conv` / `pre_ffn_conv` | primitive | zero-init equals no conv |
| P3 | LFM2 gated short conv mixer | `layers/ssm/` + `SSM_REGISTRY` `short_conv` | primitive | |
| P3 | Value residual | GQA hook `exposes_values`, `residuals.py` | | weight 0 equals `standard` |
| P3 | Logit soft-cap (attention and output) | attention option + `LMHeadStage` | | `cap -> inf` equals uncapped |
| P3 | Attention sinks | GQA option | | |
| P4 | RWKV-7, GDN-2, ALiBi | new mixers / position type | | |

Every new L0 component follows `research/2026_keras_custom_models_instructions_v2.md`, gets `compute_output_shape`, `get_config`, a registry entry, a capability row, and the drift test update in the same commit.

Exit: M3; Appendix A has no `missing` cell in priorities P1 and P2.

### Phase 5: Training and ablation harness (2 to 4 weeks)

| ID | Task | Reuse |
|---|---|---|
| T5.1 | `train/api/tasks/causal_lm.py`: wraps `ComposableLM` in `CausalLanguageModel`, sets `aggregate_backbone_losses` from `model.has_auxiliary_losses`, reports **split** loss (task vs aux) as TopoLM's Pattern 8 requires | `train/common/clm_pretrain.py`, `train/common/nlp.py`, `build_clm_metrics`, `BitsPerToken` |
| T5.2 | `train/api/tasks/text_embedding.py`: contrastive (InfoNCE / SigLIP) + Matryoshka loss over `EmbeddingHeadStage` outputs | `infonce_loss.py`, `siglip_contrastive_loss.py` |
| T5.3 | `register_model` for `composable_lm/*` and `composable_embedder/*` with presets as variants; spec overrides via `key=value` | `train/api/registry.py` |
| T5.4 | `src/train/composable_ablation/`: YAML of base spec + axis diffs; one subprocess per cell (`mean_shift` pattern), seeds via `seed_utils`, `rows.jsonl`, `report.py`, `compare_runs.py` | `rms_variants_train/`, `mean_shift/` |
| T5.5 | Equal-cost comparison mode: normalize cells by `accounting.py` (equal active params, equal FLOPs/token, or equal KV+state bytes) | ablation doc 3.10 |
| T5.6 | Register config classes in `tests/test_train/test_config_fields_are_live.py` | existing guard |

Exit: M4.

### Phase 6: Efficiency, inference, production trust (4 to 8 weeks)

| ID | Task | Notes |
|---|---|---|
| T6.1 | Benchmark harness: tokens/s, step time, peak memory, spec model vs hand-written original at equal config, `jit_compile=True`, `mixed_float16` | copy `norm_overhead_bench.py` shape; S4 gate |
| T6.2 | One attention kernel seam: route softmax mixers through `keras.ops.dot_product_attention` where numerics match | no file uses it today; on the TF backend it brings no flash kernel, the value is a single seam for a future JAX backend |
| T6.3 | `MixerState`, `prefill`, `step` for `kv`, `recurrent`, `conv` | Section 3.8 |
| T6.4 | `ComposableLM.generate()`; consolidate decode-time sampling now spread across `train/common/generation_probe.py` and `models/vision_language/nano_vlm/model.py` | |
| T6.5 | Decode parity: `generate` with cache equals full recompute logits per step | guard |
| T6.6 | Export through `train/api/exporting` | |
| T6.7 | Chunked-scan audit for KDA / GDN / Mamba-2 under fp16 | `GatedLinearAttentionBlock` static vs symbolic paths |

Exit: M5; S4 met or the gap documented with a measured cause.

### Phase 7: Migration and consolidation (3 to 6 weeks, optional, only behind parity gates)

| ID | Task | Gate |
|---|---|---|
| T7.1 | `TransformerLayer._get_attention_params` delegates to `MixerAdapter.param_map` | all `test_transformers/` green, weight paths stable |
| T7.2 | `TextDecoder` / `TextEncoder` become thin presets over `BlockStack` | `nano_vlm` and `gpt2` checkpoints load unchanged |
| T7.3 | Models whose blocks are expressible (`qwen3`, `gemma3`, `mamba2`) keep their public class and checkpoint layout; internals may delegate | `test_a_language_model_weight_paths_are_stable.py`, `test_a_rebuilt_model_has_identical_weight_paths.py` |
| T7.4 | Docs: `REPO_MAP.md` registry table (+ `MIXER_REGISTRY`, `RESIDUAL_REGISTRY`, and the already-missing `SSM_REGISTRY`), `models/AGENTS.md` reuse-policy table, `layers/AGENTS.md`, `layers/transformers/README.md` | same commit as code |

Models with genuinely different topology stay bespoke: `hnet`, `blt`, `hrm`, `trm`, `ptrm`, `tree_transformer`, `wave_field`, `topolm` (its regularizer needs a branch tensor the shared blocks discard; `REPO_MAP.md` records this rule). The residual-stream protocol may later expose such tensors via a hook; not in v1.

### Phase 8+: Provisions for the rest (designed now, built later)

| Domain | Extension point reserved in v1 | Existing assets to plug in |
|---|---|---|
| Vision encoders | `EmbeddingSpec.token` types `patch_2d` / `patch_1d`; `MixerCapabilities.domains`; `attention_mode="bidirectional"` | `patch_embedding.py`, `positional_sine_2d`, `axial_rope_2d`, `VisionEncoder`, window/area/beit attention keys |
| VLM | prefix inputs to `EmbeddingStage`; `PositionSpec.type="mrope"`; prefix-LM mask | `ModalityProjection`, `multi_axis_rope`, `mrope_ideogram4`, `nano_vlm` (uses `TextDecoder`, gets hybrids after T7.2) |
| Encoder-decoder | `BlockSpec.cross_mixer` | `multi_head_cross`, `TransformerDecoderLayer` |
| Diffusion / conditioned blocks | `BlockSpec.conditioning` | `adaln_zero`, `sd3_adaln`, `FiLM` |
| Time series | patch embedding + `heads/time_series` | `create_time_series_head` |
| Tokenizer-free | `EmbeddingSpec.token="byte"` + chunking stage | `layers/blt/`, `layers/dynamic_chunking/` |
| Depth recurrence / MoD | `StackSpec.recurrence`, router per block | `hrm`, `trm`, `research/2026_mixture_of_depth.md` |
| Multi-token prediction | `LMHeadSpec.num_future_tokens` | none |

Rule for provisions: a reserved field exists in the schema with a `None` default and a `NotImplementedError` on use, so implementing it later changes no stored config.

---

## 5. Testing strategy

| Layer | Tests | Reuse |
|---|---|---|
| Specs | round trip, hash stability, path-qualified errors, reserved fields refuse | property sampler |
| Registries | drift guard, key-set pin, capability table pinned and regenerated by script | `test_factory_registry_drift.py` |
| Mixers | causal probe, left/right padding equivalence, fp16 arm, output shape | `dtype_policy`, `tf32_disabled`, `apply_attention_mask` |
| Degenerate-case identities | MoE(1 expert) = dense; HC(n=1) = standard; mHC(n=1) = standard; GDN(tied) = KDA; MQA = GQA(kv=1); SWA(w>=S) = causal; YaRN(1) = RoPE; zero-init Canon = none; soft-cap(inf) = none; LatentMoE(identity) = MoE | cheapest strong guards |
| Blocks / stacks | pairwise covering array, gradient flow, knob sensitivity | `gradient_flow_oracle`, `knob_sensitivity_oracle` |
| Models | smoke, lazy build, round trip, initializer replay, precision arms, language contract family | all `tests/test_models/*_oracle.py`, `test_a_language_*` |
| Parity | per preset logits equality after weight transfer | `golden_reference_device`, `utils/weight_transfer` |
| Decode | cache vs recompute | Phase 6 |
| Trainer | config fields live, tiny end-to-end CLM and embedding runs | `test_config_fields_are_live.py` |

Repo rules that apply: a guard must be proven red (inject the defect, watch it fail, restore); tests for `composable_lm` live flat in `tests/test_models/test_composable_lm/`; scope pytest to touched modules, never the 1.5-hour full suite as a routine check; never delete anything under `results/`.

---

## 6. Architecture decisions (ADRs, with alternatives considered)

| ID | Decision | Alternatives rejected and why |
|---|---|---|
| ADR-1 | User API composes **specs (data)**, models are built from specs; `get_config` stores `spec.to_config()` | Passing layer instances (PyTorch style): breaks config-as-data, nests Keras objects in configs, makes ablations un-diffable |
| ADR-2 | `MIXER_REGISTRY` **references** existing registry keys; no parallel implementations | New mixer classes per API: duplicates 37 attention + 4 SSM implementations and their hard-won fixes |
| ADR-3 | Residual stream owned by the stack through a 4-method protocol | Residual inside block only: cannot express HC (n streams) or AttnRes (depth history) |
| ADR-4 | Norm placement is a `NormSpec` field | Separate block types per placement: combinatorial explosion |
| ADR-5 | RoPE stays inside attention layers; `PositionSpec` is translated into their kwargs | External RoPE pre-transform: needs q/k, which only exist inside the mixer |
| ADR-6 | Code lives in `layers/composable/` plus two leaf packages under `models/language/`; no 14th top-level subpackage | top-level `dl_techniques/llm/`: splits the factory surface `REPO_MAP.md` routes through |
| ADR-7 | Existing models untouched until parity presets pass; migration optional (Phase 7) | Rewrite-first: risks checkpoints under `results/` that cannot be regenerated |
| ADR-8 | Stay on TF backend with `keras.ops`; isolate the attention kernel behind one seam | JAX migration now: out of scope, blocks everything else |
| ADR-9 | Training call returns one tensor; decode state through separate `prefill` / `step` | State threaded through `call`: complicates functional build, shape inference, serialization |
| ADR-10 | Capabilities are measured and pinned, never declared | Hand-written flags: the attention open-questions brief documents masks silently ignored by three types |
| ADR-11 | Canonical parameter names are the registry names; no alias layer | Short aliases (`heads`, `kv_heads`): a third vocabulary to keep in sync |

---

## 7. Non-functional requirements

| Area | Requirement |
|---|---|
| Serialization | 100% of presets and sampled specs round-trip through `.keras`; `schema_version` stored |
| Determinism | same spec + seed gives identical initial weights (`initializer_replay_oracle`) |
| Precision | every text mixer either passes the `mixed_float16` arm or refuses at build with a named reason |
| Performance | spec overhead `<= 3%` step time vs hand-written (target); zero Python work inside `call` beyond the block loop |
| Errors | invalid specs fail at construction with a path-qualified message, never at first forward |
| Logging | `dl_techniques.utils.logger` only; no logging inside `call()` |
| Docs | `README.md` + `AGENTS.md` for `layers/composable/`; `README.md` per leaf; registry tables generated from code |
| Test budget | new scoped suites under 10 minutes on one GPU |

---

## 8. Milestones, sequencing, estimates

| Milestone | Definition of done |
|---|---|
| M1 | `nn.LM` builds a Qwen3-shaped dense model, trains 100 steps through `CausalLanguageModel`, round-trips |
| M2 | Qwen3 dense + MoE, GPT-2, Qwen3-Embedding parity green |
| M3 | 3:1 KDA:GQA and 3:1 GDN:GQA hybrids with DeepSeek-style MoE train through `python -m train` |
| M4 | One YAML runs at least 10 axes from Appendix A at equal active parameters, with a generated report |
| M5 | KV-cache / recurrent-state generation matches recompute; S4 benchmark published |

Critical path: Phase 0 -> 1 -> 2 -> 3 (orders 1 to 4) -> 5. Phase 4 P1 items can start after Phase 1 in parallel (they are L0 work with their own tests). Phase 6 starts after Phase 3 order 6. Phase 7 is optional.

| Phase | Estimate (eng-weeks) |
|---|---|
| 0 Measure and decide | 1 to 2 |
| 1 Specs, registries, adapters | 2 to 4 |
| 2 Block, stack, stages, models | 3 to 5 |
| 3 Parity presets | 4 to 7 |
| 4 Missing components | 6 to 10 |
| 5 Training and ablation | 2 to 4 |
| 6 Efficiency and inference | 4 to 8 |
| 7 Migration (optional) | 3 to 6 |
| **Total** | **25 to 46**; about 12 to 20 calendar weeks with two engineers splitting Phase 4 and Phase 6 |

Biggest estimate risk: Phase 3 weight maps for bespoke blocks (`qwen3_next`, `zamba2`) and Phase 6 recurrent-state decode for chunked scans.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Heterogeneous mask semantics across attention keys | high | capability audit T0.2; text facade only exposes keys whose padding is `honored` |
| Default drift (epsilon `1e-6` vs `1e-3`, `rope_percentage` `0.5` vs `1.0`) silently changes numerics | high | no-default `epsilon`; presets pin every numeric default; parity tests |
| MoE aux losses dropped in training | medium | `has_auxiliary_losses` drives `aggregate_backbone_losses`; split-loss reporting (TopoLM measured 17.67 aggregated vs 7.68 not, same batches) |
| Adapter indirection slows training | low | adapters resolve at build only; S4 benchmark gate |
| Recurrent mixers leak left padding into state | medium | adapter zeroes padded inputs; left-padding equivalence test |
| Scope creep toward all 37 attention keys | high | `domains` capability; phase-1 list in Section 3.4 is the whole text surface |
| Long context memory on TF backend (no flash kernel) | medium | prefer sliding / linear mixers for long-context ablations; kernel seam T6.2 for a future backend |
| Adoption chicken-and-egg (the blog's own point) | medium | presets reproduce models people already use here; the ablation harness is the reason to adopt; migrate internal callers in Phase 7 |
| Serialization of nested specs grows configs | low | plain JSON; schema version; size check in round-trip test |

---

## 10. Out of scope for this roadmap

Distributed training (data/tensor/pipeline parallel, FSDP), custom CUDA kernels, FP8, JAX/PyTorch backend migration, post-training (SFT, RLHF, RL), tokenizer training, speculative decoding, vision/VLM/time-series composition beyond the reserved fields in Phase 8, rewriting `hnet`, `blt`, `hrm`, `trm`, `ptrm`, `tree_transformer`, `wave_field`, `topolm`.

---

## Appendix A: Ablation-axis coverage matrix

Axes from `research/2026_llm_architecture_mega_ablation.md` section 5. Status: **exists** (L0 has it, spec field only), **adapter** (exists, needs translation or mask work), **new** (Phase 4).

| Axis | Spec field | Status | Phase 4 prio |
|---|---|---|---|
| Learned absolute positions | `embedding.absolute_position="learned"` | exists | |
| RoPE, base, partial | `mixer.position=RoPE(theta, fraction)` | exists | |
| NoPE everywhere / global only | `mixer.position=NoPE()` per layer | adapter (T0.5) | P2 |
| YaRN | `RoPE(scaling=...)` | new | P2 |
| RMSNorm vs LayerNorm | `norm.type` | exists | |
| Bias removal | mixer / ffn `use_bias` | exists | |
| Sandwich norm | `norm.placement="sandwich"` | adapter | |
| Post-norm (+ QK-norm, OLMo 2) | `norm.placement="post"` | exists | |
| QK-norm (learned scales) | `mixer.params.qk_norm_type` | exists | |
| DyT | `norm.type="dynamic_tanh"` | exists | |
| Embedding `sqrt(d)` scale | `embedding.scale="sqrt_dim"` | adapter | |
| Logit soft-cap | `head.softcap`, mixer option | new | P3 |
| GELU MLP / SwiGLU / GeGLU / Squared ReLU | `ffn.type` | exists | |
| GQA / MQA | `mixer.type="gqa"/"mqa"` | exists | |
| MLA | `mixer.type="mla"` | exists | |
| Gated attention | `mixer.type="gated"` | exists | |
| KV conv | `mixer.params.kv_conv_size` | new | P2 |
| Attention sinks | `mixer.params.num_sinks` | new | P3 |
| Sliding windows (3 of 4, 1 global in 8) | `attention_mode="sliding"`, `nn.pattern` | new (mask) | P1 |
| KDA hybrid | `mixer.type="kda"`, `nn.pattern` | exists | |
| Gated DeltaNet hybrid | `mixer.type="gated_deltanet"` | new | P1 |
| Mamba-2 hybrid | `mixer.type="mamba2"` | exists | |
| RWKV-7 | `mixer.type="rwkv7"` | new | P4 |
| LFM2 short conv | `mixer.type="short_conv"` | new | P3 |
| Fine-grained MoE (128 experts, top-k) | `ffn=MoE(num_experts, top_k)` | exists | |
| Sigmoid routing + aux-free bias | `MoE(gating="sigmoid")` | new | P1 |
| Shared expert | `MoE(shared_experts=1)` | new | P1 |
| LatentMoE | `MoE(latent_dim=dim//2)` | new | P2 |
| Null experts | `MoE(null_experts=k)` | new | P2 |
| Mixtral style (8, top-2) | `MoE(8, 2)` | exists | |
| Squared-ReLU experts | `MoE(expert=FFN("squared_relu"))` | exists | |
| Canon-A / Canon-C | `pre_mixer_conv` / `pre_ffn_conv` | new | P3 |
| Hyper-connections | `residual=HyperConnections(n)` | new | P2 |
| mHC | `residual=MHC(n)` | new | P2 |
| AttnRes | `residual=AttnRes()` | new | P2 |
| Value residual | `residual=ValueResidual()` | new | P3 |
| Sparse attention (Lighthouse) | `mixer.type="lighthouse"` | adapter | |
| Layer scale, drop path | `residual.layer_scale`, `residual.drop_path` | exists | |

## Appendix B: New and touched files

```
src/dl_techniques/layers/composable/
    __init__.py  AGENTS.md  README.md  GUIDE.md
    nn.py  spec.py  context.py  validation.py
    mixers.py  capabilities.py  positions.py  residuals.py
    block.py  stack.py  schedules.py
    embedding_stage.py  output_stage.py
    accounting.py  state.py                          # state.py in Phase 6
src/dl_techniques/models/language/composable_lm/
    __init__.py  README.md  model.py  presets.py  parity.py
src/dl_techniques/models/language/composable_embedder/
    __init__.py  README.md  model.py  presets.py  parity.py
src/dl_techniques/utils/causality_probe.py            # extracted from clm.py (T1.6)
src/dl_techniques/utils/masking/                      # sliding-window, prefix builders (T0.6)
src/dl_techniques/layers/embedding/factory.py         # token_* keys
src/dl_techniques/layers/embedding/rotary_position_embedding.py   # YaRN
src/dl_techniques/layers/attention/kimi_delta_attention.py        # decay granularity -> gated_deltanet
src/dl_techniques/layers/attention/group_query_attention.py       # NoPE fix, kv_conv, sinks, exposes_values
src/dl_techniques/layers/attention/factory.py                     # gated_deltanet key
src/dl_techniques/layers/moe/{gating,layer,config,experts}.py     # sigmoid, shared, latent, null
src/dl_techniques/layers/ssm/{factory,short_conv}.py              # LFM2 mixer
src/dl_techniques/layers/conv_blocks/                             # shared causal depthwise conv (location per T0.7)
src/train/api/tasks/{causal_lm,text_embedding}.py
src/train/composable_ablation/{__init__,config,sweep,report,README}.py/.md
tests/test_layers/test_composable/
tests/test_models/test_composable_lm/
tests/test_models/test_composable_embedder/
tests/test_train/test_composable_ablation/
REPO_MAP.md  src/dl_techniques/models/AGENTS.md  src/dl_techniques/layers/AGENTS.md  src/train/AGENTS.md
research/roadmap/2026_llm_building_blocks.md          # this document
```