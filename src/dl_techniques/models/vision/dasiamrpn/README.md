# DaSiamRPN: Distractor-Aware Siamese Region Proposal Tracking

A Keras 3 implementation of **DaSiamRPN** — SiamRPN one-shot detection with
distractor-aware training policy and long-term tracking extensions.

## 1. Overview

**DaSiamRPN** (Zhu et al., ECCV 2018) extends SiamRPN: the exemplar embedding
becomes per-anchor correlation kernels, correlated against the search
embedding to yield dense classification logits plus box deltas. The paper's
contributions around the network are a distractor-aware sampling strategy
and a local-to-global redetection arm; this package ports the network
(backbone + RPN adjust convolutions + per-anchor correlation) and provides
anchors, box decoding and windowing as NumPy post-processing.

Key properties:

1. **Shared backbone**: one `SiamRPNBackbone` (width scale 1 or 2).
2. **Per-anchor correlation**: exemplar kernels `(K, K, C, A*2 / A*4)`
   against search features, 19x19 grid at 127/271.
3. **Raw outputs**: `{"cls": (B, S, S, A*2), "reg": (B, S, S, A*4)}`,
   no soft-max or decode in the graph.
4. **Variants**: `big` (width 2, feat 512), `vot` / `otb` (width 1,
   feat 256) — the released configurations.

The paper reports winning the VOT-2018 real-time challenge and strong
OTB/UAV results; no such claim is made for this code, which has not been
trained or benchmarked here.

## 2. Usage

```python
import numpy as np
from dl_techniques.models.vision.dasiamrpn import create_dasiamrpn

model = create_dasiamrpn("big")
z = np.zeros((1, 127, 127, 3), dtype="float32")
x = np.zeros((1, 271, 271, 3), dtype="float32")
out = model((z, x), training=False)
print(out["cls"].shape, out["reg"].shape)  # (1, 19, 19, 10) (1, 19, 19, 20)
```

Anchors and decoding:

```python
from dl_techniques.models.vision.dasiamrpn import generate_dasiamrpn_anchors, decode_dasiamrpn_boxes
anchors = generate_dasiamrpn_anchors(19)  # (19*19*5, 4) in (cx, cy, w, h)
```

`pretrained=True` raises `NotImplementedError`; pass a local checkpoint path
instead.

## 3. Components

| Name | Purpose |
| :--- | :--- |
| **`DaSiamRPN`** | Model over `(z, x)` returning `{"cls", "reg"}`. |
| **`SiamRPNBackbone`** | Width-scaled embedding trunk (scale 1 or 2). |
| **`create_dasiamrpn` / `from_variant`** | Named-configuration construction. |
| **`generate_dasiamrpn_anchors`** | Reference anchor factory. |
| **`decode_dasiamrpn_boxes`** | Reference delta parameterization. |
| **`create_hann_window`** | Cosine window tiled over anchors. |

Deliberate non-goals: the global-search redetection schedule and the online
distractor template update are tracking-loop policies, not network layers,
and are not implemented here.

## 4. Citation

```bibtex
@inproceedings{zhu2018distractor,
  title={Distractor-aware siamese networks for visual object tracking},
  author={Zhu, Zheng and Wang, Qiang and Li, Bo and Wu, Wei and Yan, Junjie and Hu, Weiming},
  booktitle={ECCV},
  year={2018}
}
```
