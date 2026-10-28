# MambaLCT

Long-term context single-object tracker after Li et al., 2024
([paper](https://arxiv.org/abs/2412.13615)): a shared hierarchical
appearance encoder plus a unidirectional selective-SSM scanner that rolls
target-change cues from the first frame into context tokens conditioning
each new frame.

## Architecture

```
template (B,Ht,Wt,3) ──╲
                        shared UcaEncoder ── t_tokens (B,Nt,512)
search clip (B,T,Hs,Ws,3) ──╱  (per-frame) ── s_flow (B,T,Ns,512)
                                                 │ + context (B,1,512)
                                                 ▼
                                    ContextMambaLayer ── enhanced (B,T,Ns,512)
                                                         updated_context (B,1,512)
                                                 │ cross-attn template
                                                 ▼
                                    per-frame head ── scores (B,T,1), boxes (B,T,4)
```

`UcaEncoder` is a 3-stage hierarchical ViT (4×4 stem + 2×2 downsamplers, total
stride 16, CPE + `TransformerLayer` blocks) built only from the library
factories. Its layers are named `stem`, `stage{S}_cpe`, `stage{S}_block{B}`
(S and B counted from 1), `downsample_{i}_conv` and `downsample_{i}_norm` (i
from 0, between two stages) and `final_norm`. `ContextMambaLayer` (`layers/ssm`) prepends the context tokens to
the temporal scan (causal carry into every frame) and appends one learnable
bridge token whose output becomes the history-aggregated context update.
Fusion uses the `multi_head_cross` attention factory key, with an initializer of its own for each of its
three projections (`q`, `kv`, `proj`: see Archives, below). The tracking head
is architecture-specific (per-frame pooled box + score) and lives here —
`heads/vision/` holds generic detection heads, not template-conditioned
tracker heads.

**Attention types of the encoder.** `UcaEncoder(attention_type=...)` accepts `"multi_head"` (the default and the
only one `MambaLCT` builds), `"group_query"` and `"multi_head_latent"`, the three whose projections each start
from their own random draw (the replay oracle finds 0 pairs of kernels carrying one draw in each; `multi_head`
needs the output initializer the encoder gives it, the other two clone one per projection themselves). Any other
value raises `ValueError` naming that set. The others were refused because they cannot be made to draw
independently from this side, or cannot run on the encoder's tokens (measured on a two-stage encoder with two
blocks per stage): `differential` replays 60 of 630 kernel pairs, `anchor` 40 of 496, `window` 4 of 276 and
`window_band` 4 of 190 (their layers hand one initializer instance to several projections and declare no
argument that separates them; changing those layers would move the seeded weights of every model built on
them), `window_zigzag`, `fnet` and `lighthouse` need a statically known sequence length, `beit` fails inside its
attention at the encoder's token shape, and `multi_head_cross` is not an attention type of `TransformerLayer`.
`test_a_refused_attention_type_would_have_to_run_and_draw_independently` holds those nine as strict expected
failures: when a layer is fixed the row passes and the run goes red, which is the cue to widen the set.

## Variants

| Variant | Template | Search | Backbone |
|---|---|---|---|
| `mambalct-256` | 128 | 256 | stages [128, 256, 512], depths [2, 2, 6] |
| `mambalct-384` | 192 | 384 | same backbone, larger inputs |

Widths/depths are this repository's own (paper: HiViT-Base, 72M); no paper
numbers are claimed for this code. The v1 head predicts one global box per
frame; a dense OSTrack-style score-map head is deferred. Training samples
clips (`train/mambalct`, clip length 2); inference carries
`updated_context` frame to frame.

## Usage

```python
from dl_techniques.models.vision.mambalct import create_mambalct

model = create_mambalct("mambalct-256")
out = model([template, search_clip, None])
boxes = out["boxes"]  # (B, T, 4) normalized (cx, cy, w, h)
```

## Archives

**Archive note (2026-10-28).** `UcaEncoder`'s layers were renamed (`stage_{s}_block_{b}` to `stage{S}_block{B}`,
`merge_{s}` to `downsample_{s-1}_conv`, `merge_norm_{s}` to `downsample_{s-1}_norm`, `cpe_{s}` to
`stage{s+1}_cpe`), and the attention output projection of every block now draws independently of its
`qkv` projection. Measured by saving a tiny `MambaLCT` (`template_size=32`, `search_size=64`, stages
`[64, 128, 512]` of depth 1, 86 tensors) in a separate process from the old revision and restoring it
with the current code (`git archive`, `python -I`), inference and `training=True` outputs compared at
`atol=0`:

- From `be92a74eb` (the stem already one `ConvStem`): `keras.models.load_model` and `model.load_weights`
  both restore all 86 tensors and the outputs are equal in both modes, although 47 `w.path` strings
  changed.
- From `20b48aba6^` (the stem a `Conv2D` named `stem` and a `stem_norm`, two tracked layers): neither
  restores, `Layer 'stem_conv1' expected 2 variables, but received 0 variables during loading`. Transplanting
  by `w.path` (`stem/{kernel,bias}` to `stem/stem_conv1/*`, `stem_norm/*` to `stem/stem_norm1/*`, and the
  four renames above) assigns all 86 tensors (51 of the old paths renamed) and the outputs are equal at
  `atol=0`.
- The renames are free only because the encoder's list attributes keep their names. A `.keras` archive
  stores a layer under its attribute path (`encoder/merges/...`, `encoder/merge_norms/...`,
  `encoder/cpe_convs/...`, `encoder/blocks/...`), not under its layer name: a first version of this change
  renamed the attributes `merges` and `merge_norms` to match the layers, and every archive from `be92a74eb`
  then refused to load (`Layer 'downsample_0_conv' expected 2 variables, but received 0`). So the two
  attributes stay `merges` and `merge_norms`; `test_a_saved_mambalct_stores_the_encoder_under_the_attribute_names_old_archives_use`
  pins the stored names.
- Seeded starting weights change (measured at seed 1234 against the `be92a74eb` build of the same tiny model,
  `scratch_iter2/step16_1/mlct_pseed_rows.txt`): **25 of the 86 tensors differ and 61 are equal**. The equal ones
  are every tensor created before the first new initializer (the detection head, 12 of 12; the encoder stem,
  4 of 4; `stage1_cpe`; the first block's `qkv` kernel and bias) and every constant (all biases, all norm scales
  and offsets: `final_norm`, `fusion_norm`, the block norms). The 25 are random kernels created after that point,
  by group: encoder 15 of 53 (the `proj` kernel of each of the 3 blocks, the `fc1` and `fc2` kernels of each, the
  `qkv` kernel of blocks 2 and 3, the two downsample convolutions, `stage2_cpe` and `stage3_cpe`), `context_mamba`
  7 of 13 (`input_proj`, `in_proj`, `conv1d`, `x_proj`, `dt_proj` kernel and `dt_proj` bias, `out_proj`) and
  `fusion_attn` 3 of 6 (`q`, `kv` and `proj` kernels). 10 of the 25 are outside the encoder. They differ for one
  reason: Keras draws the seed of every random initializer, when the initializer is constructed, from one stream
  (`random.seed(1234)` then `randint`), so a build with five more initializer objects than before (one attention
  output initializer per encoder block, 3, and the fusion attention's `kv` and `proj` initializers, 2) moves every
  later initializer to a later position of the stream. `scratch_iter2/step16_1/mlct_seed_positions.json` lists the
  stream position of each kernel's seed in both builds: 0 for the 7 kernels before the first change, then 1, 2
  and 3 positions as the encoder's blocks insert their initializers, and 3 to 5 for the context SSM and the
  fusion attention; no tensor differs for a reason outside those positions. The trained function is the same; the replay oracle finds no pair of kernels carrying one draw in
  the encoder (3 of 120 before) nor in the whole tracker (3 of 496 before: the fusion attention's `q`, `kv` and
  `proj`, which one initializer instance had initialized alike).
- Transcripts: `plans/plan-2026-10-26T125959-7af851e5/scratch_iter2/step16/archive_summary.txt`,
  `scratch_iter2/step16/archive_mambalct_values.txt` and `scratch_iter2/archive/transcripts.txt`.
