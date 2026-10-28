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
| **`DaSiamRPN`** | Model over `(z, x)` returning `{"cls", "reg"}`. Takes no `input_shape`: the sizes are `exemplar_size` and `search_size`, and passing `input_shape` or `input_dim` raises `ValueError`. |
| **`SiamRPNBackbone`** | Width-scaled embedding trunk (scale 1 or 2). Stage 1 (11x11/2 convolution, BatchNorm, ReLU, 3x3/2 pool) is one `ConvStem` named `stem`; stages 2 to 5 keep their own layers. |
| **`create_dasiamrpn` / `from_variant`** | Named-configuration construction. |
| **`generate_dasiamrpn_anchors`** | Reference anchor factory. |
| **`decode_dasiamrpn_boxes`** | Reference delta parameterization. |
| **`create_hann_window`** | Cosine window tiled over anchors. |

Deliberate non-goals: the global-search redetection schedule and the online
distractor template update are tracking-loop policies, not network layers,
and are not implemented here.

`kernel_initializer` is a name, a serialized dict or an `Initializer` instance. The model stores a name or dict (an instance is serialized once) and all ten convolutions (five in the backbone, five adjust convolutions) draw from their own copy, so an instance does not hand one sample to ten kernels, and `get_config()` is JSON-safe.

### Archives written before the `ConvStem` stage 1

The first convolution, its BatchNorm and the first pool used to be loose layers of the backbone (`conv1`, `bn1`, `pool1`); stage 1 is now one tracked layer, `stem`. Archives of `DaSiamRPN` written by the earlier code (measured with `be92a74eb` and with `20b48aba6^`, which give the same results, for the `otb` and `big` variants, each saved in a separate process and loaded by the current code) do not load by either route. `keras.models.load_model` and `model.load_weights` into a current model both fail with "A total of 2 objects could not be loaded ... Layer 'stem_conv1' expected 2 variables, but received 0".

The weights themselves are intact and move by path. Copying the 36 weights of an earlier model (both variants) by these renames into a current model gives outputs equal at atol 0, and a model built under the same seed holds the same tensors as the earlier one:

| Earlier path | Current path |
| :--- | :--- |
| `da_siam_rpn/backbone/conv1/{kernel,bias}` | `da_siam_rpn/backbone/stem/stem_conv1/{kernel,bias}` |
| `da_siam_rpn/backbone/bn1/*` | `da_siam_rpn/backbone/stem/stem_norm1/*` |

The backbone has 26 weights (18 trainable) and 3,751,680 parameters at width scale 1 and 14,925,312 at width scale 2. The `get_config()` of `DaSiamRPN` and `SiamRPNBackbone` now writes the stored initializer name or dict where it used to write the serialized instance; configs written by the earlier code still build.

## 4. Citation

```bibtex
@inproceedings{zhu2018distractor,
  title={Distractor-aware siamese networks for visual object tracking},
  author={Zhu, Zheng and Wang, Qiang and Li, Bo and Wu, Wei and Yan, Junjie and Hu, Weiming},
  booktitle={ECCV},
  year={2018}
}
```
