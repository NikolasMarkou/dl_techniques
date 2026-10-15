# Clifford CLIP audit + Clifford small-VLM plan (single document)

Date: 2026-10-15. All executable claims verified CPU-only (`CUDA_VISIBLE_DEVICES=""`).
Repo: `dl_techniques` (Keras 3 / TF 2.18). Plan mode: no source files modified.

---

## 1. Audit (a): CliffordCLIP parts — VERIFIED

Reference oracle: vanilla `models/vision_language/clip/model.py:63 CLIP`.
Subject: `models/vision_language/clifford_clip/model.py:116 CliffordCLIP`,
`create_clifford_clip` (`model.py:1467`), variants `nano,nano_g,mini,small,base,large` (`model.py:330`).

| # | Part | CliffordCLIP | Vanilla CLIP | Status |
|---|------|--------------|--------------|--------|
| 1 | Vision tower | Conv stem + staged `CliffordNetBlock`s + `PatchMerging` + optional `Dense`, external residual + `StochasticDepth` (`model.py:632,1107`) | `patch_conv` + CLS + isotropic `TransformerLayer`s (`clip/model.py:518`) | confirmed, tiny 1-stage forwards `(2,32,32,3)->(2,32)` |
| 2 | Text tower | Learned token+position `Embedding`, `CausalCliffordNetBlock`s, external residual (`model.py:818,1199`) | Token embed, GQA+RoPE `TransformerLayer`s, causal mask (`clip/model.py:560,772`) | confirmed; position = learned table, NOT RoPE |
| 3 | Pooling | Vision `SequencePooling(mean)` anchor; text `last_non_pad_token(ids,pad=0)`, right-pad assumption (`model.py:1147,1210`; `clip_utils.py:78`) | Vision CLS-0 (`clip/model.py:733`); text same last-non-pad (`clip/model.py:779`) | confirmed; vision anchor is the intentional delta |
| 4 | Head | `apply_clifford_head` (`clip_utils.py:28`): `plain→anchor`, `mean_max/learned_query→geo`, `learned_query_residual→anchor+LayerScale(geo)`, gamma init 1e-5 | N/A (bias-free Dense only) | confirmed all 4 `head_kind`s; `plain` builds no `*_head_geo` |
| 5 | Projection | `LN → Dropout? → Dense(embed,use_bias=True) → fp32 L2+1e-8` (`model.py:1162,1237`) | `Dense(embed,use_bias=False) → L2` (`clip/model.py:553,599,741`) | confirmed; norms measured `1.0`; bias flag is a real divergence |
| 6 | Temperature | `exp(logit_scale)` float32, clipped `≤logit_scale_max=100`, init 2.6592 (~14.28) (`model.py:1251`) | bare `exp`, unclamped, same init (`clip/model.py:846`) | confirmed both measure `14.2848` |
| 7 | `call` contract | **Requires both** `inputs["image"],inputs["text"]` (`model.py:1277`) — single-modality raises `KeyError: 'text'` (verified) | Partial: image-only / text-only omit logits keys (`clip/model.py:793,859`, verified) | **MISMATCH — fix or document before retrieval use** |
| 8 | Loss | `CLIPContrastiveLoss(y_dummy,{logits_per_image,logits_per_text}\|tuple)→(B,)`, `apply_temperature=False` (`losses/clip_contrastive_loss.py:419`); SigLIP alt `temperature=1.0,learnable=False` (`train/cliffordnet/train_clip.py:268`) | same loss class | confirmed `1.0379` tiny batch, dict≡tuple; stock `fit` needs `train/clip/train_clip.py:StackedCLIPLogits` bridge (import verified) |
| 9 | Factory/serial | `from_variant(variant,vocab_size,image_size,context_length,pretrained=False)` — data args REQUIRED, no 224/77 defaults (D-047); `refuse_pretrained`; key `dl_techniques.models.clifford_clip.model` | `from_variant(ViT-B/32…)` | confirmed `nano` builds at 64/16; `.keras` round-trip max-diff `0.0` |
| 10 | Tests/trainer | `tests/test_models/test_clifford_clip/` + `test_clip/` = **77 collected** (`--collect-only`) | — | confirmed |

**Trainer staleness (real issue):** `train/cliffordnet/train_clip.py:650,707,789`
references `vision_global_pool`, `vision_query_pool/text_query_pool`,
`vision_global_max_pool` — `grep` over `clifford_clip/` returns **no matches**
(current names: `vision_det_pool/vision_ctx_pool/text_det_pool/text_ctx_pool`).
`_VisionClassifier.call` will `AttributeError` if vision pretraining runs.
Contrastive path (uses `encode_*` + bridge) is unaffected. Fix names or delete
the wrapper before use.

---

## 2. Small VLM (b): B1 — Clifford vision + stock `TextDecoder`

Reuse `NanoVLM` shell (`nano_vlm/model.py:267,490`); swap only `_vision_tokens`.
Reference layout: `VisionEncoder(none) [B,1+P,Dv] → ModalityProjection → [B,Sv,Dt]`
+ `TextDecoder.embed_tokens → [B,T,Dt] → decode([vision;text], one causal mask)`
→ tied `hidden@word.T` or untied `Dense(V)` (`nano_vlm/model.py:95,537`).
Loss `NanoVLMLoss` scores trailing-T logits, shifts, ignores pad (`losses/nano_vlm_loss.py:132`);
`generate()` re-runs decoder per token, no KV-cache (`nano_vlm/model.py:564`).

### 2.1 Adapter (the only new vision code, ~30 lines, in the new package)

Clifford vision emits `(B,H',W',Dv_last)`, **no CLS**, where
`H'=(image_size//patch_size)//2^(n_stages-1)` (`clifford_clip/model.py:1077`).
`PixelShuffle/ModalityProjection` assume `[CLS+spatial]`, require `H'%s==0`
(`pooling/pixel_shuffle.py:29`, `modality_projection.py:36`). Direct plug
misreads token 0 as CLS and fails at `H'=7,s=2` (stock 224/p4).

Design: `reshape(B,H'*W',Dv) → prepend learned CLS [1,Dv] → ModalityProjection`.
`input_dim=Dv_last` (nano 256 / mini 384), `output_dim=Dt`; `projection_config`
must not set them (`nano_vlm/model.py:373` raises). Default `scale_factor=1`
(lossless, no divisibility trap); `s=2` only when `H'%2==0`.
`Sv = H'*W'+1` (s=1). Examples: 128px/p4/4-stage → `H'=4` → **Sv=17**;
224px → `H'=7` → Sv=50 (s=1 only). `num_vision_tokens` stays derived
(`nano_vlm/model.py:310`); `Sv=(grid/s)^2+1` formula (`README:35`) is replaced
by the Clifford one above.

### 2.2 Default proposal (LOCKED)

`128px / p4 / 4-stage (H'=4, Sv=17)`, `Dt=384, depth=6, heads=6 (head_dim 64)`,
tied (`learned` table, `V×384 ≈ 12.3M @32k`), `max_seq_len=128` text
(joint 145). Vision `cli_mode=full, ctx_mode=diff`, `LayerScale 1e-5`,
`vision_use_global_context=False`, external residual mandatory (GUIDE §5:
`x=block(x)` collapses ~1e-5/block).

### 2.3 Decoder config — GQA (LOCKED), with one code prerequisite

SOTA in-tree: CLIP `group_query+RMSNorm(pre)+SwiGLU+RoPE` (`clip/model.py:101`);
`NanoVLM modern` `differential+RMSNorm+SwiGLU+pre`, decoder
`factorized+RMSNorm+SwiGLU+pre+learned`, untied (`nano_vlm/README.md:56`).
`TextDecoder` defaults are legacy GPT-2 (`multi_head/layer_norm/post/mlp/gelu`,
`text_decoder.py:160`).

Pinned B1 block:

```python
text_config = dict(vocab_size=V, embed_dim=384, depth=6, num_heads=6,
    max_seq_len=128, embedding_type='learned', positional_type='learned',
    attention_type='group_query', normalization_type='rms_norm',
    normalization_position='pre', ffn_type='swiglu',
    dropout_rate=0.1, attention_dropout_rate=0.1, layer_norm_eps=1e-6,
    n_kv_head=2, attention_args=dict(num_kv_heads=2, rope_percentage=0.0,
        qk_norm_type='rms_norm', dropout_rate=0.1),
    ffn_args=dict(ffn_expansion_factor=4, ffn_multiple_of=64))
```

Prerequisite (verified by execution 2026-10-15): `TextDecoder.__init__` has
**no** `attention_args/n_kv_head/ffn_args` (all `False`), while
`TransformerLayer` accepts all three; GQA registry defaults are
`max_seq_len=2048, rope%=1.0` with `num_kv_heads` required. So
`attention_type='group_query'` via `text_config` today degrades to MHA-width
KV **and** double-counts position (learned table + GQA RoPE). The ~10-line
`TextDecoder` passthrough extension (`rope_percentage=0.0` load-bearing) is
required for true 6Q/2KV; fallback without it is `multi_head` + same norm/FFN
(mini-parity, zero code change).

### 2.4 Local + global attention — 5:1, deferred

Precedents: Gemma3 `layer_types` 5×sliding+1×full ×3, window 512
(`gemma3.py:224`; small 2:1; README: locals extract, globals route every 4–6);
ModernBERT `window_band` local + global every `interval=3` (`modern_bert/model.py:22,236`;
local band is dense N×N+mask — bias, not FLOPs, `model.py:243`; locals inherit
position via global RoPE residual). `TextDecoder` is single-type only — mixture
needs a Gemma-style custom stack. **At N≈145 global GQA is ~21k scores/layer:
run all-global.** Adopt 5:1 (`5×sliding(w=256)+1×full`, ×2 at depth 12, full at
6,12; 4:1 equivalent) only with the 1024-context upgrade below.

### 2.5 Context lengths

* **128 text (joint 145) — SHIP.** COCO/CC3M-length captions + short
  instructions; table `(Sv+T)×384` ≈ 56k params. (Agreed: COCO/CC3M stay
  CLIP-only; VLM data is §3.)
* 256 — multi-turn instruction; still all-global.
* **1024 (joint 1041) — needs:** depth-12 mixed 5:1 custom stack (depth-6 gives a
  single global router — too few); `max_seq_len=1024` upfront
  (`SequenceLengthError` guard); packed long-form data (tail positions train
  only on long samples; `sincos` extrapolates better than learned);
  batch 8–16 + accumulation + bf16. Attention 1.08M scores/layer (~50×).

---

## 3. VLM training data (HF open, Apache-2.0 SmolVLM recipe)

SmolVLM (`blog/smolvlm`): `HuggingFaceM4/the_cauldron` (1.88M multi-task) +
`HuggingFaceM4/Docmatix` (2.4M docs / 9.5M QA); eval MMMU/MathVista/MMStar/
DocVQA/TextVQA. Oumi VL-SFT table confirms IDs below.

* **Stage 1 alignment (short):** `allenai/pixmo-cap`, `nlphuji/flickr30k`.
* **Stage 2 instruction (128–256, packed):** `liuhaotian/LLaVA-Instruct-150K`,
  `HuggingFaceH4/llava-instruct-mix-vsft`, `HuggingFaceM4/the_cauldron`
  (sampled 50–100k first).
* **Stage 3 text-in-image:** `facebook/textvqa` (45k/28k OpenImages — works at
  128px + ctx 128; PICK for tiny-res). `HuggingFaceM4/Docmatix` deferred to the
  224px+/1024 variant (doc text unreadable at 128px). `hiyouga/geometry3k`
  (MathVista proxy). Eval: `merve/vqav2-small`, TextVQA-val, DocVQA-test.
* **Preference (later):** `HuggingFaceH4/rlaif-v_formatted` (83k DPO).

---

## 4. Synthetic puzzles (5–10% mix, seeded numpy+PIL-lazy, no cv2/skimage)

All Q/A short (`NanoVLMLoss`-compatible, <128 tokens); hold out unseen
palette/texture/size bins; report plain-vs-styled gap as the robustness metric.

1. **Sudoku** (solver-backed, unique): `valid?` → `digit(r,c)?` → row/col/box
   → full 81-char. Axes: mono → cell/box colors → glyph+texture grain →
   rotated/missing gridlines.
2. **Mazes — rectangular / circular / 3D-projected+distorted** (user-locked):
   rectangular DFS 9→15→21 `(r,c)`; circular polar rings×wedges `(ring,wedge)`,
   same action vocab; 3D = world-plane homography (random quad tilt/roll,
   horizon out) + radial `r(1+k1r²+k2r⁴)` via numpy bilinear (precedents:
   `transform_sequences._affine_warp`, `synthetic_warp`; house numpy+PIL rule).
   Answers in topology coords (viewpoint invariance). Curriculum flat →
   circular → tilted → distorted+texture; eval on held-out (tilt,k1,k2).
3. **Color Nonograms** 8–12², clues rendered in-image: `cell?` → `row?` →
   full; mono → multi-color layers → font/texture-clue noise.
4. **ARC rule grids** 5–10² (reuse `datasets/arc/`): recolor-by-count,
   symmetry completion, motif translate; `apply rule?` → `which rule?` →
   `where?`; 2-color → 4–6 palette → textured fills → distractors.
5. **Odd-one-out + tile swap** 4–6²: one tile differs in rotation/mirror/hue/
   grain/scale; `which (r,c)?` → `how?` → `swap to fix?`; flat → JND gaps →
   busy low-contrast backgrounds.
6. **Spot-the-difference montages** (single-image constraint → A|B panels +
   divider in one 128px frame, ~56px/panel; fine-grain gated to 224px):
   base scene × N edits from {remove,add,recolor,retexture,move,rotate,scale};
   `how many?` → `missing/added in B?` → `where?` → `describe (N≤3)`;
   1 coarse aligned → 2–3 styled → +panel jitter (±2px/0.95× kills pixel-diff).

Tiny-model note: 6-layer nets saturate on validity/single-cell/next-move;
full-solve/full-path are stretch goals — gate promotion on early-task saturation.

---

## 5. New package + build order

`models/vision_language/clifford_nanovlm/{__init__.py,model.py,README.md}`:
`CliffordNanoVLM` = Clifford staged vision (`stage_channels/depths/shifts`,
`patch_size`, `image_size`) + CLS + `ModalityProjection` + stock/extended
`TextDecoder`; `create_clifford_nanovlm(variant,vocab_size,image_size,context_length)`;
registration `dl_techniques.models.clifford_nanovlm.model` (family stripped);
variants `tiny/small` (e.g. Dv=192,Dt=384,Sv=17).

1. Adapter + shape tests (Sv, s=1/2, `H'%s` refusal, CLS handling).
2. Forward/generate parity vs `NanoVLM` tiny on CPU; causality probe
   (`test_the_text_logits_are_causal.py` template).
3. `get_config` round-trip (value-compare, target `0.0` like §1);
   `tests/test_models/test_clifford_nanovlm/`; tree-wide `--collect-only`.
4. Fix §1 trainer staleness (rename or delete `_VisionClassifier` path).
5. `TextDecoder` GQA passthrough extension (or accept MHA fallback).
6. Trainers: CLIP contrastive (existing bridge) → VLM next-token
   (Pattern-3 + `NanoVLMLoss`, packed long-form only when 1024).

## 6. Open decisions for user

* GQA fallback vs `TextDecoder` extension (recommend extension).
* 128-ship vs 1024-upfront (recommend 128 now; adapter unaffected by later bump).
* Puzzle sample counts per level (proposal: 20k validity / 10k single-cell /
  5k full-solve equivalents per family).
* Docmatix inclusion gated on 224px+1024 variant (recommend deferred).
