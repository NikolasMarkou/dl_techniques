# H-Net: hierarchical dynamic chunking for byte-level language modelling

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 port of **H-Net** (Hwang et al., 2025, [arXiv:2507.07955](https://arxiv.org/abs/2507.07955),
reference implementation [goombalab/hnet](https://github.com/goombalab/hnet)). H-Net is a
byte-level causal language model with **no tokenizer**: it learns where to cut the byte stream,
runs a deeper network on the shorter chunked sequence, and scatters the result back to full
resolution.

> **No pretrained weights are distributed with this repository.**
> `HNet.from_variant(..., pretrained=True)` and `create_hnet(..., pretrained=True)` both raise
> `NotImplementedError`, naming the variant. They never warn-and-return-random-weights.
> The reference checkpoints are **not loadable here** either — see § 5, where the two structural
> reasons are stated. To get a model, build it and train it.

> **Nothing here has been trained.** This README carries no benchmark numbers, no perplexity and
> no quality claim, because none has been measured. What is documented is the architecture, the
> public surface, and the places this port deliberately differs from the reference.

---

## 1. The architecture

```text
input_ids (B, L) int32                     raw UTF-8 bytes, vocab_size = 256
      |
Embedding(256, d_model[0])                 stddev = 1.0  (NOT 0.02 -- see model.py)
      |
HNetStage(stage_idx=0) ------------------------------------------------------+
      |                                                                      |
      |   encoder isotropic stack        d_model[0]                          |
      |         |                                                            |
      |   RoutingModule                  boundary_prob / boundary_mask       |
      |         |                        boundary <=> p > 0.5                |
      |   ChunkLayer                     (B, L, D) -> (B, max_chunks, D)     |
      |         |                                                            |
      |   HNetStage(stage_idx=1)         the SAME layer, one level down      |
      |         |                        (recursion bottoms out at the       |
      |         |                         innermost "main" stack)            |
      |   DeChunkLayer                   EMA over the inner sequence,        |
      |         |                        scattered back to (B, L, D)         |
      |   gated residual + residual_proj                                     |
      |         |                                                            |
      |   decoder isotropic stack                                            |
      +----------------------------------------------------------------------+
      |
lm_head                                    Dense(256, no bias, stddev = 0.02)
      |                                    or, tied, hidden @ embeddings.T
      v
logits (B, L, 256)
```

Each stage's encoder / decoder / innermost stack is an **isotropic** stack of pre-norm blocks
described by a small layout language: `m4` is four Mamba-2 blocks, `T22` is twenty-two
attention-plus-SwiGLU blocks, and an uppercase letter is what adds the MLP. The nesting is the
hierarchy: `["m4", ["T1m4", ["T26"], "m4T1"], "m4"]` is three stages and two chunking levels.

Module map:

| File | Holds |
|---|---|
| `config.py` | the `arch_layout` parser, the frozen spec dataclasses, `MODEL_VARIANTS`, `n_residuals` / `n_residuals_by_stage` |
| `components.py` | `build_mixer` / `build_mlp` / `build_causal_keep_mask`, `HNetBlock`, `HNetIsotropic` |
| `stage.py` | `HNetStage` — the recursive encoder / route / chunk / inner / dechunk / residual / decoder layer |
| `losses.py` | `ratio_loss`, `total_ratio_loss`, `DEFAULT_TARGET_RATIO` |
| `model.py` | `HNet`, `create_hnet`, `default_max_chunks`, the init constants |

The three chunking layers are **not** in this package. They live in
`src/dl_techniques/layers/dynamic_chunking/` (`RoutingModule`, `ChunkLayer`, `DeChunkLayer`) and
have their own README-level docstring there; this package is architecture assembly only.

## 2. Usage

```python
from dl_techniques.models.language.hnet import create_hnet, HNet, MODEL_VARIANTS

model = create_hnet("hnet_1stage_L", max_seq_len=1024)      # random init
sorted(MODEL_VARIANTS)                                       # the six names below
```

A small model, built from a config rather than a variant name:

```python
import keras
from dl_techniques.models.language.hnet import HNet, HNetArchConfig
from dl_techniques.models.language.hnet.config import AttnSpec

cfg = HNetArchConfig(
    arch_layout=["m1", ["T1"], "m1"],
    d_model=[16, 16],
    d_intermediate=[0, 0],
    attn_cfg=AttnSpec(num_heads=(2, 2), rotary_emb_dim=(4, 4), window_size=(-1, -1)),
)
model = HNet(cfg, max_chunks=(8,), max_seq_len=32, headdim=8)
model(keras.ops.zeros((2, 16), dtype="int32")).shape       # (2, 16, 256)
```

The **ratio (load-balancing) loss** is contributed through `self.add_loss` with coefficient
`RATIO_LOSS_ALPHA = 0.03`, so stock `model.fit()` picks it up. There is no custom `train_step`
anywhere in this package, deliberately: a hand-written one silently discards Keras' own
`scale_loss` under `mixed_float16`.

## 3. The six shipped variants

Every row is transcribed from the correspondingly named JSON in the reference repository's
`configs/` directory, with `ssm_cfg.chunk_size` dropped (§ 5.4). `ssm_cfg` is
`d_conv=4, expand=2, d_state=128` in all six, so the table does not repeat it.

| Variant | `configs/` citation | `arch_layout` | `d_model` | `d_intermediate` | `rotary_emb_dim` | `window_size` |
|---|---|---|---|---|---|---|
| `hnet_1stage_L` | `configs/hnet_1stage_L.json` | `["m4", ["T22"], "m4"]` | 1024, 1536 | 0, 4096 | 32, 48 | 1023, -1 |
| `hnet_1stage_XL` | `configs/hnet_1stage_XL.json` | `["m4", ["T24"], "m4"]` | 1024, 2048 | 0, 5504 | 32, 64 | 1023, -1 |
| `hnet_2stage_L` | `configs/hnet_2stage_L.json` | `["m4", ["T1m4", ["T26"], "m4T1"], "m4"]` | 1024, 1024, 1536 | 0, 2816, 4096 | 32, 32, 48 | 1023, 1023, -1 |
| `hnet_2stage_XL` | `configs/hnet_2stage_XL.json` | `["m4", ["T1m4", ["T27"], "m4T1"], "m4"]` | 1024, 1536, 2048 | 0, 4096, 5504 | 32, 48, 64 | 1023, 1023, -1 |
| `hnet_2stage_XL_chinese` | `configs/hnet_2stage_XL_chinese.json` | `["m4", ["T1m4", ["T30"], "m4T1"], "m4"]` | 1024, 1536, 2048 | 0, 4096, 5504 | 32, 48, 64 | 1023, 1023, -1 |
| `hnet_2stage_XL_code` | `configs/hnet_2stage_XL_code.json` | `["m4", ["T1m4", ["T28"], "m4T1"], "m4"]` | 1024, 1536, 2048 | 0, 4096, 5504 | 32, 48, 64 | 1023, 1023, -1 |

`num_heads` is 16 at every stage of every variant. Naming: a "1stage" variant has **one chunking
level** and therefore **two** stages; a "2stage" variant has two chunking levels and three stages.

The depth-scaled initializer denominator per variant, derived by
`n_residuals_by_stage(parse_arch_layout(cfg.arch_layout))` — outermost first:

| Variant | per-stage denominator | hierarchy total |
|---|---|---|
| `hnet_1stage_L` | (8, 52) | 52 |
| `hnet_1stage_XL` | (8, 56) | 56 |
| `hnet_2stage_L` | (8, 20, 72) | 72 |
| `hnet_2stage_XL` | (8, 20, 74) | 74 |
| `hnet_2stage_XL_chinese` | (8, 20, 80) | 80 |
| `hnet_2stage_XL_code` | (8, 20, 76) | 76 |

Residual-writing projections (`out_proj`, `w_o`, `down_proj`) get `0.02 / sqrt(n_k)` at stage `k`;
every other Dense keeps the flat `0.02`. This per-stage form is the reference's
(`hnet.py:121-147` threads the count inward); the single hierarchy-wide number coincides with it
only at the innermost stage.

## 4. Weights

No H-Net checkpoint ships with this repository, and none can be converted from the reference
without extra work (§ 5.1 and § 5.2 each independently prevent a straight load). Concretely:

```python
create_hnet("hnet_1stage_L", pretrained=True)     # NotImplementedError, naming the variant
```

`keras.models.load_model("your_own.keras")` is the supported warm-start path; importing this
package registers `HNet`, `HNetStage`, `HNetBlock` and `HNetIsotropic` (and, transitively, the
three chunking layers) for deserialization, so no `custom_objects` argument is needed.

## 5. Recorded divergences from the reference

Eight. Each is a deliberate choice with a consequence, not an approximation, and each is anchored in
the source and recorded in `plans/plan-2026-09-09T042752-6d66ac56/decisions.md`. The count is not
copied between documents: re-derive it with

```bash
grep -c '^### 5\.' src/dl_techniques/models/language/hnet/README.md
```

### 5.1 RoPE pairing is interleaved, not the reference's split-half

The reference applies GPT-NeoX **split-half** rotary embedding (`rotary.py:396-404`,
`interleaved=False`). This port builds its attention with the existing
`create_attention_layer("group_query", ...)` factory entry, and that layer's rotary sub-layer is
the **interleaved** (GPT-J / RoFormer) pairing. Partial rotary is preserved exactly, as
`rope_percentage = rotary_emb_dim / head_dim`, and the anchor is
`# DECISION plan-2026-09-09T042752-6d66ac56/D-005` in `components.py`.

**What it forecloses**: a reference H-Net checkpoint cannot be loaded into this port as-is. The
two conventions are related by a fixed permutation of the **rows of `q_proj` and `k_proj`**, so a
converter would have to permute those rows; copying the weights straight across produces a model
that runs and is wrong. From random initialisation the two pairings are training-equivalent, which
is why the divergence was accepted (D-005).

### 5.2 Padded / masked layout only — the packed `cu_seqlens` path is not ported

Every module upstream branches on a packed, ragged layout (`cu_seqlens`, `(total, D)`) versus a
padded one (`(B, L, D)` plus a boolean mask). Only the **padded** path is ported. The packed path
exists for the CUDA kernel APIs (`mamba` wants a leading batch dim of 1, `flash_attn_varlen` wants
none), and neither kernel is used here (D-007, ghost constraint G3).

**Consequence**: compute is spent on padded positions, and there is no true sequence packing. It
is a throughput cost, not a correctness one.

### 5.3 A fixed `max_chunks`, not the reference's batch-dependent width

Upstream, the chunked width is `max(boundary_mask.sum(-1))` (`dc.py:186-199`) — data-dependent and
**batch-dependent**. This port takes a fixed `max_chunks` per chunking level as a constructor
argument and truncates in **position order** (the first `max_chunks` boundaries), replicating the
stable-partition `argsort` gather exactly for everything within the cap. Anchor:
`# DECISION plan-2026-09-09T042752-6d66ac56/D-007` in
`src/dl_techniques/layers/dynamic_chunking/chunk_layer.py`.

**Consequence, named plainly**: a row whose real boundary count exceeds `max_chunks` **loses its
tail chunks**. Those bytes still carry their own encoder output into the decoder through the
residual branch (the residual is read per-position, before chunking), but they are unrepresented in
the inner stage, and on the way back `DeChunkLayer` clips their scatter index to the last retained
chunk — so they receive that chunk's value rather than one of their own. This is a failure mode the
reference does not have. It
must be measured on the corpus you train on, not assumed rare; `default_max_chunks` only supplies a
starting point (`max_seq_len // 2`, halving per level, a 3x margin over the 6.0 target ratio) and
says so at its definition site.

**The other direction — a sequence SHORTER than the cap — is supported and was not.** `L <
max_chunks[0]` is the ordinary case for a short input: `default_max_chunks(2, max_seq_len=2048)` is
`(1024,)`, so any prompt below 1024 bytes has `L < M`. `ChunkLayer` right-pads its permutation with
index `0` for that case; `DeChunkLayer` did not, so **every** H-Net raised
`InvalidArgumentError: slice index <L> of dimension 1 out of bounds` on such an input — at its own
constructor defaults, including `create_hnet("hnet_1stage_L")`. Both layers now share
`layers/dynamic_chunking/indexing.py::pad_permutation_to_width`, so `M > L`, `M == L` and `M < L`
are handled identically on the two sides by construction. Anchor:
`# DECISION plan-2026-09-09T042752-6d66ac56/D-029`. The surplus inner columns are inert — a row's
valid chunk count never exceeds `L`, so `plug_back_idx` never reads them.

Truncation is by position and not by boundary probability on purpose: magnitude-order truncation
lets a later, higher-scoring byte displace an earlier boundary, which is a causality defect
`layers/blt/blt_blocks.py` already measured and guards against.

### 5.4 No `chunk_size` knob

The reference's `SSMConfig` declares `chunk_size=256` for Mamba-2's chunked scan. This repository's
`Mamba2Layer` implements a **sequential** scan and has no chunked path, so the field is absent from
`SSMSpec` rather than present and ignored — a declared-but-unread config field is the exact defect
`tests/test_train/test_config_fields_are_live.py` exists to reject. Do not add it back without a
consumer.

### 5.5 `residual_proj` ships a zero bias

`hnet.py:102-107` zeroes the residual projection's **weight** and flags it `_no_reinit`; it does not
touch the bias, which `torch.nn.Linear` leaves uniformly random in `+/- 1/sqrt(D)`. This port
zero-initialises **both**, so the residual branch is exactly a no-op at step 0 and the claim
"starts at exactly zero" is literally checkable. Anchor:
`# DECISION plan-2026-09-09T042752-6d66ac56/D-019` in `stage.py`.

**Consequence**: at initialisation, stage 0's residual term differs from the reference by a
per-channel random constant of magnitude ~`1/sqrt(1024)` ~ 0.031. Since no reference weights are
ever loaded, nothing downstream depends on reproducing that draw.

### 5.6 `DEFAULT_TARGET_RATIO = 6.0` is this port's choice, not a transcription

The target compression ratio `N` of the ratio loss appears **nowhere in the reference repository** —
its training script is not part of the released code. `6.0` is the compression the paper reports
targeting for its single-stage byte models, adopted here as a documented default and labelled as
such at its definition site in `losses.py`. Do not later cite it as reference-faithful (D-021).

Every training entry point is expected to set `target_ratios` per chunking level explicitly.

### 5.7 `RATIO_LOSS_ALPHA = 0.03` is this port's choice too

The same situation as § 5.6, one line up in `model.py`, and it was not disclosed until the
adversarial review of 2026-09-09 pointed out that the constant sits between two CITED ones
(`EMBEDDING_INIT_STDDEV` cites `mixer_seq.py:60`, `INITIALIZER_RANGE` cites `mixer_seq.py:53`) and
therefore reads as transcribed. It is not: `grep -rn "0[.]03"` over the whole reference repository
returns **zero** hits, because the coefficient lives in the training script the reference does not
release. `0.03` is the weight the paper reports for the load-balancing term, adopted here as a
documented default and now labelled as such at its definition site.

**Consequence**: it is the auxiliary-loss weight of every run launched from `src/train/hnet/`
(`HNetTrainingConfig.ratio_loss_alpha` defaults to it), so a training campaign should treat it as a
hyper-parameter to sweep rather than as a fidelity constraint. Pinned by
`test_model.py::TestRatioLossWiring` in three ways — value, provenance, and observable effect on
`model.losses` — because a 10x change to it previously survived all 665 tests of this port.

### 5.8 `d_intermediate = 0` is a SENTINEL here, and an explicit width upstream

The reference's `HNetConfig.d_intermediate` is a per-stage list whose "unset" value is `None`
(`hnet/modules/mlp.py:21-24` reads `if d_intermediate is None: derive`, and any other value is used
verbatim). This port types the field as a tuple of `int`, which cannot carry `None`, so **`0` is
this port's spelling of that sentinel**: `build_mlp` derives `round_up(8 * d_model / 3, 128)` from
it. Upstream, a literal `0` is not a sentinel at all — it is an explicit width, and
`nn.Linear(d_model, 0)` is what upstream would build.

**Why it is a divergence and not an implementation detail**: the two readings disagree for any
caller that pairs `d_intermediate = 0` with an UPPERCASE layout letter, i.e. a stage that really has
an MLP. Upstream gives that stage a zero-width MLP; this port gives it the derived width. That
combination is reachable, and this plan's own fixtures reach it (`test_model.py`,
`test_stage.py` and `test_causality.py` all pair `d_intermediate=[0, 0]` with a `T` stack).

**Why no shipped model moves**: all six `MODEL_VARIANTS` rows carry their `0` on an all-lowercase
stage, which has no MLP for either reading to apply to; every positive shipped value already equals
the width the derivation produces (1024→2816, 1536→4096, 2048→5504). All 16 `(variant, stage)`
widths were measured across the change and the diff is empty (D-031).

**Consequence**: a config transcribed literally from an upstream JSON is safe, because upstream's
own configs use `0` only where this port also has no MLP; a HAND-WRITTEN config that means "give
this uppercase stage no MLP" cannot be expressed, and will silently get the derived width instead.
Use a lowercase layout letter for that.

Anchored at `components.py::build_mlp` and in `config.py`'s `HNetArchConfig` docstring; pinned by
`test_components.py::TestGuardSixIntermediateWidth` and by the hand-written
`SHIPPED_SWIGLU_WIDTHS` table. Recorded here after review pass 2 found that D-031 claimed this
divergence was registered when it was not.

## 6. Tests

| Suite | Covers |
|---|---|
| `tests/test_models/test_hnet/test_arch_layout.py` | the layout parser, per-stage indexing, `n_residuals` |
| `tests/test_models/test_hnet/test_components.py` | mixer dispatch, pre-norm block arithmetic, the causal keep mask, RoPE liveness |
| `tests/test_models/test_hnet/test_stage.py` | the recursion, the gated residual, symbolic build, serialization |
| `tests/test_models/test_hnet/test_losses.py` | the ratio loss against a transcribed reference |
| `tests/test_models/test_hnet/test_model.py` | the assembled model, the depth-scaled init, the six-variant sweep |
| `tests/test_layers/test_dynamic_chunking/` | the three chunking layers against a float64 NumPy oracle transcribed from `dc.py` |

Run them scoped, never as part of the full suite:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_hnet/ -q
```

## References

- Hwang, B., Wang, S., Gu, A., 2025. *Dynamic Chunking for End-to-End Hierarchical Sequence
  Modeling.* [arXiv:2507.07955](https://arxiv.org/abs/2507.07955)
- Reference implementation: [github.com/goombalab/hnet](https://github.com/goombalab/hnet) —
  `hnet/models/hnet.py`, `hnet/models/mixer_seq.py`, `hnet/modules/dc.py`,
  `hnet/modules/isotropic.py`, `hnet/models/config_hnet.py`
- Dao, T., Gu, A., 2024. *Transformers are SSMs: Generalized Models and Efficient Algorithms
  Through Structured State Space Duality (Mamba-2).*
  [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
