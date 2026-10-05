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

`UcaEncoder` is a 3-stage hierarchical ViT (4×4 stem + 2×2 merges, total
stride 16, CPE + `TransformerLayer` blocks) built only from the library
factories. `ContextMambaLayer` (`layers/ssm`) prepends the context tokens to
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
