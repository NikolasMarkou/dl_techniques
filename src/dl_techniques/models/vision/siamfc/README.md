# SiamFC: Fully-Convolutional Siamese Tracking

A Keras 3 implementation of **SiamFC** — tracking as offline similarity
learning with a shared fully-convolutional embedding and cross-correlation.

## 1. Overview

**SiamFC** (Bertinetto et al., ECCV 2016) trains one embedding network on
pairs of crops and tracks by correlating the exemplar features against the
search-region features. The peak of the resulting score map is the target.

Key properties:

1. **Shared weights**: the same `SiamFCBackbone` embeds both inputs.
2. **Fully-convolutional search**: total stride 8, `valid` padding
   everywhere, so a larger search image yields a larger score map.
3. **Single-channel response**: `response = corr(f(z), f(x)) + bias`,
   trained with a radius-based logistic label.
4. **Paper sizes**: exemplar 127, search 255, features 6x6 / 22x22 over 256
   channels, response 17x17.

The paper reports real-time frame rates and state-of-the-art results on its
benchmarks at publication time; no such claim is made for this code, which
has not been trained or benchmarked here.

## 2. Usage

```python
import numpy as np
from dl_techniques.models.vision.siamfc import create_siamfc

model = create_siamfc()
z = np.zeros((1, 127, 127, 3), dtype="float32")
x = np.zeros((1, 255, 255, 3), dtype="float32")
response = model((z, x), training=False)
print(response.shape)  # (1, 17, 17, 1)
```

Custom sizes (must keep `search > exemplar` with a positive score map):

```python
model = create_siamfc(exemplar_size=127, search_size=255, use_batch_norm=True)
```

Serialization round-trips through `.keras` with no `custom_objects`:

```python
model.save("siamfc.keras")
loaded = __import__("keras").models.load_model("siamfc.keras")
```

`pretrained=True` raises `NotImplementedError`; pass a local checkpoint path
instead: `create_siamfc(pretrained="/path/to/weights.keras")`.

## 3. Components

| Name | Purpose |
| :--- | :--- |
| **`SiamFC`** | Model over `(z, x)` returning `(B, S, S, 1)` logits. |
| **`SiamFCBackbone`** | Shared 5-stage AlexNet-variant embedding, stride 8. |
| **`create_siamfc`** | Thin factory over `SiamFC`. |
| **`siamfc_score_size`** | Pure score-extent helper, single source of truth. |
| **`create_hann_window`** | NumPy Hann window for inference blending. |

There is no `MODEL_VARIANTS` table: the paper ships one architecture, so
width and input sizes are constructor arguments. This is deliberate and
documented in `model.py`.

## 4. Training notes

- Compile with binary cross-entropy over the radius-based label
  (positive radius / negative radius in search pixels, divided by stride 8).
- Inference (outside the graph): 3-scale pyramid, Hann window via
  `create_hann_window(17)`, scale penalty.

## 5. Citation

```bibtex
@inproceedings{bertinetto2016fully,
  title={Fully-convolutional siamese networks for object tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip H S},
  booktitle={ECCV},
  year={2016}
}
```
