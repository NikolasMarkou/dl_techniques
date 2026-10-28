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
Fusion uses the `multi_head_cross` attention factory key. The tracking head
is architecture-specific (per-frame pooled box + score) and lives here —
`heads/vision/` holds generic detection heads, not template-conditioned
tracker heads.

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
- Seeded starting weights change: a model built with a given seed keeps its stem, its first positional
  convolution and the first block's `qkv` kernel, and draws every other random kernel afresh (13 of the 48
  tensors of the two-stage roster encoder), because each block's attention output projection now takes its own
  initializer. The trained function is the same; the replay oracle finds no pair of kernels carrying one draw
  (it found 3 of 120 before).
- Transcripts: `plans/plan-2026-10-26T125959-7af851e5/scratch_iter2/step16/archive_summary.txt`,
  `scratch_iter2/step16/archive_mambalct_values.txt` and `scratch_iter2/archive/transcripts.txt`.
