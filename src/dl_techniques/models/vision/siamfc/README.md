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
| **`SiamFC`** | Model over `(z, x)` returning `(B, S, S, 1)` logits. Takes no `input_shape`: the sizes are `exemplar_size` and `search_size`, and passing `input_shape` or `input_dim` raises `ValueError`. |
| **`SiamFCBackbone`** | Shared 5-stage AlexNet-variant embedding, stride 8. Stage 1 (11x11/2 convolution, BatchNorm, ReLU, 3x3/2 pool) is one `ConvStem` named `stem`; stages 2 to 5 keep their own layers. |
| **`create_siamfc`** | Thin factory over `SiamFC`. |
| **`siamfc_score_size`** | Pure score-extent helper, single source of truth. |
| **`create_hann_window`** | NumPy Hann window for inference blending. |

There is no `MODEL_VARIANTS` table: the paper ships one architecture, so
width and input sizes are constructor arguments. This is deliberate and
documented in `model.py`.

`kernel_initializer` is a name, a serialized dict or an `Initializer` instance. The model stores a name or dict (an instance is serialized once) and every convolution draws from its own copy, so an instance does not hand one sample to five kernels, and `get_config()` is JSON-safe.

## 4. Training notes

- Compile with binary cross-entropy over the radius-based label
  (positive radius / negative radius in search pixels, divided by stride 8).
- Inference (outside the graph): 3-scale pyramid, Hann window via
  `create_hann_window(17)`, scale penalty.

### Archives written before the `ConvStem` stage 1

The first convolution, its BatchNorm and the first pool used to be loose layers of the backbone (`conv1`, `bn1`, `pool1`); stage 1 is now one tracked layer, `stem`. Archives of `SiamFC` written by the earlier code (measured with `be92a74eb` and with `20b48aba6^`, which give the same results, each saved in a separate process and loaded by the current code) do not load by either route. `keras.models.load_model` and `model.load_weights` into a current model both fail with "A total of 2 objects could not be loaded ... Layer 'stem_conv1' expected 2 variables, but received 0" (1 object without BatchNorm).

The weights themselves are intact and move by path. Copying the 27 weights of an earlier model (11 without BatchNorm) by these renames into a current model gives outputs equal at atol 0, and a model built under the same seed holds the same tensors as the earlier one:

| Earlier path | Current path |
| :--- | :--- |
| `siam_fc/backbone/conv1/{kernel,bias}` | `siam_fc/backbone/stem/stem_conv1/{kernel,bias}` |
| `siam_fc/backbone/bn1/*` | `siam_fc/backbone/stem/stem_norm1/*` |

The shared backbone has 26 weights (18 trainable) and 3,751,680 parameters, 10 weights and 3,747,200 parameters without BatchNorm; `SiamFC` adds the scalar `response_bias`. The `get_config()` of `SiamFC` and `SiamFCBackbone` now writes the stored initializer name or dict where it used to write the serialized instance; configs written by the earlier code still build.

## 5. Citation

```bibtex
@inproceedings{bertinetto2016fully,
  title={Fully-convolutional siamese networks for object tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip H S},
  booktitle={ECCV},
  year={2016}
}
```
