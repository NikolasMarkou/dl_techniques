# ViT with a hierarchical MLP stem (`vit_hmlp`)

`ViTHMLP` is a Vision Transformer whose patch embedding is the hierarchical MLP (hMLP) stem of
Touvron et al., *Three things everyone should know about Vision Transformers* (2022). Each patch
runs through a hierarchy of linear projections, normalization and non-linearity on its own, so no
operation crosses a patch boundary and masking before or after the stem gives the same result.
Everything after the stem is a standard transformer encoder. `stem_norm_layer` picks `'batch'`
(the default) or `'layer'` normalization inside the stem.

## Output and head

With `include_top=True` the model returns **logits** of shape `(batch, num_classes)`: compile
with `SparseCategoricalCrossentropy(from_logits=True)` and apply `keras.ops.softmax` to read
probabilities. The head is the shared vision `ClassificationHead` configured as the ViT
fine-tuning head (Dosovitskiy et al. 2020, Sec. 3.1, "a single linear layer at fine-tuning
time"): the CLS token of the final norm, `Dropout(dropout_rate)`, one `Dense(num_classes)`. The
model's `kernel_initializer`, `bias_initializer`, `kernel_regularizer` and `bias_regularizer`
reach that `Dense`. `head_config_overrides` reshapes the head (for example
`{'block_dims': (512,)}` adds a hidden block); `name`, `num_classes` and `task_type` in it raise
`ValueError`.

With `include_top=False`, `pooling` selects the output: `'cls'`, `'mean'` or `'max'` return
`(batch, embed_dim)`, and `None` returns the full normalized sequence. Unlike `vit`, `mean` and
`max` pool over the CLS token too; that difference is deliberate and pinned by
`tests/test_models/test_vit_hmlp/test_cls_slice_identity.py`.

The `return_dict` and `head_config_preset` arguments were removed on 2026-10-14. Archives saved
before that date (a dict output, or the two-`DenseBlock` head) do not load and are not supported.

## Quick start

```python
import keras
from dl_techniques.models.vision.vit_hmlp import create_vit_hmlp

model = create_vit_hmlp(input_shape=(64, 64, 3), num_classes=10, scale="tiny", patch_size=16)
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

`scale` is one of `'tiny'`, `'small'`, `'base'`, `'large'`, `'huge'` (`ViTHMLP.MODEL_VARIANTS`).
