# CLIP: Contrastive Language-Image Pre-Training

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **CLIP (Contrastive Language-Image Pre-training)**: two towers map images and captions into one embedding space, trained so that a matched pair scores higher than every mismatched pair in the batch.

This is a **modernized** CLIP, not a weight-compatible port. Both towers are built from this repo's `TransformerLayer` with grouped-query attention, RMSNorm, SwiGLU and rotary position embeddings, and neither tower carries a learned positional table. No pretrained weights are distributed.

The package also ships `CliffordCLIP` (`clifford_clip.py`), a Clifford-algebra variant that shares the contrastive objective but not the tower internals. It is not documented here.

---

## 1. Overview: What is CLIP and Why It Matters

CLIP addresses a supervision problem, not an architectural one. Labelled image datasets are small and their label sets are closed, so a classifier can only ever name categories somebody enumerated in advance. Image-caption pairs are abundant and open-ended, but a caption is not a label: predicting it token by token is expensive, and two captions of the same picture rarely agree word for word.

The contrastive formulation extracts a usable signal anyway by asking a much weaker question — *which caption in this batch belongs to which image*. Both modalities are mapped into one space, each feature is L2-normalized so an inner product is a cosine, and a batch of `N` pairs yields an `N x N` similarity matrix whose diagonal must dominate both its row and its column:

```
S = tau * f_I(I) @ f_T(T)^T
```

Two things follow. It scales, because the batch supplies its own negatives — the `N^2 - N` mismatched pairs cost nothing to construct. And it yields zero-shot classification for free: any set of class names can be encoded as text and used directly as a classifier weight matrix.

**The loss is not implemented here.** This module produces the two logits matrices and the temperature; the contrastive loss lives in `dl_techniques.losses`.

---

## 2. The Problem CLIP Solves

| | Supervised ImageNet-style training | CLIP |
|---|---|---|
| Supervision | Manual labels from a fixed set | Web alt-text, open-ended |
| Cost per example | Human annotation | Free, already on the page |
| New class | Collect data, retrain | Write a prompt |
| What is learned | Which of `K` bins | Which caption describes this image |

The consequence is that a single pre-trained CLIP serves an unbounded number of classification tasks: describe the classes in words, encode them once, and compare. That property is why CLIP became a component of text-to-image models, retrieval indexes and larger VLMs rather than just a classifier.

---

## 3. How CLIP Works: Core Concepts

```
Image batch (B, 224, 224, 3) ──► Vision tower ──► project ──► L2 norm ──► (B, D)
                                                                            │
Text batch  (B, 77) ────────────► Text tower ───► project ──► L2 norm ──► (B, D)
                                                                            │
                     logits = clip_scale * image_features @ text_features^T
                     ──► logits_per_image (B, B), logits_per_text (B, B)
```

Training maximizes the diagonal of that matrix under a symmetric cross-entropy over its rows and its columns. Both towers end in a **bias-free** `Dense(embed_dim)` out of their native width (768 vision / 512 text at base scale), followed by L2 normalization. Keeping the projection last and the normalization after it is what makes the dot product a cosine and the temperature the only scale in the logits.

---

## 4. Architecture Deep Dive

### 4.1 Vision tower

Strided-convolution patch embedding, a learnable CLS token prepended to the patch sequence, `vision_layers` transformer blocks, and a read of position 0.

The CLS token is a single `(1, 1, vision_width)` weight broadcast across the batch — every image starts from the same query vector, and the attention blocks are what make its final state image-specific. It is created in `build()` alongside the temperature, not in `__init__`.

The tower is **bidirectional** and stays so: there is no ordering over image patches to respect.

### 4.2 Text tower — and its causal mask

Token embedding, `text_layers` transformer blocks, and a read of the last non-padding position.

**The tower is causal.** `encode_text` builds a lower-triangular mask and passes it to every text block, so the pooled last real token is the only position that has read the whole sentence — which is what makes last-token pooling meaningful. Two details matter if you touch this code:

- The mask is built in the masking factory's **block** semantics (`True` = mask out) and inverted once to the **attend** semantics the attention layers want.
- It is broadcast to **rank 3** `(B, L, L)` on purpose. A rank-2 mask is read by `GroupedQueryAttention` as a `(batch, seq)` *padding* mask, not as a `(seq, seq)` *score* mask.

Pooling uses `last_non_pad_token`, which counts non-pad tokens and reads index `count - 1`. That **assumes right padding and pad id `0`** — the id is hard-coded at the call site. A left-padded batch, or a tokenizer whose pad id is not zero, pools the wrong position silently rather than raising. This also differs from OpenAI CLIP, which locates the EOT token as the argmax of the token ids.

### 4.3 Position encoding

Neither tower has a learned positional embedding. Position enters only as **RoPE** inside the grouped-query attention, rotating queries and keys by an angle proportional to index. On the image side that means patches are positioned along the flattened raster order with CLS at index 0 — a real departure from CLIP's learned positional embedding, and worth knowing before comparing numbers.

### 4.4 Temperature

`logit_scale` is an unconstrained scalar weight holding the **logarithm** of the temperature; `exp` is applied on every use, which keeps the multiplier positive under ordinary gradient descent without a constraint object. The default init `2.6592` is `ln(1 / 0.07)`, the paper's starting temperature of roughly 14.3.

**This class applies no upper clamp.** Unlike the MobileCLIP models in this repo, a diverging temperature here produces `inf` logits and a `nan` loss with no other symptom. A trainer that expects OpenCLIP's clamp must supply it.

### 4.5 `call` is partial by design

Passing only `image` or only `text` returns just that tower's features and **omits the logits keys entirely**, so encoding a caption bank for retrieval does not require fabricating a dummy image batch. The output dict's shape is therefore input-dependent, and any consumer indexing it must account for that.

---

## 5. Quick Start Guide

```python
import keras
import numpy as np
from dl_techniques.models.vision_language.clip.model import CLIP

model = CLIP.from_variant("ViT-B/32")
model.build({'image': (None, 224, 224, 3), 'text': (None, 77)})
print(f"{model.count_params():,} parameters")   # 147,929,857

images = np.random.rand(4, 224, 224, 3).astype("float32")
tokens = np.random.randint(1, 49408, (4, 77)).astype("int32")

outputs = model({'image': images, 'text': tokens}, training=False)
print(sorted(outputs))
# ['image_features', 'logit_scale', 'logits_per_image',
#  'logits_per_text', 'text_features']
print(outputs['image_features'].shape)    # (4, 512)
print(outputs['logits_per_image'].shape)  # (4, 4)

# Single modality: only that tower's features come back.
print(sorted(model({'text': tokens}, training=False)))   # ['text_features']
```

**No tokenizer ships here.** The model expects CLIP's byte-pair encoding with a 49,408-token vocabulary, right-padded with id `0`. Pretrained tokenizers are available in libraries such as Hugging Face `transformers` (`openai/clip-vit-base-patch32`).

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`CLIP`** | `...clip.model.CLIP` | The dual-encoder `keras.Model`. |
| **`CLIP.from_variant`** | `...clip.model.CLIP.from_variant` | Build from a `MODEL_VARIANTS` key; kwargs override the row. |
| **`create_clip_variant`** | `...clip.model.create_clip_variant` | Thin wrapper around `from_variant`. |
| **`create_clip_model`** | `...clip.model.create_clip_model` | Thin wrapper around the constructor, for custom configs. |
| **`CliffordCLIP`** | `...clip.clifford_clip.CliffordCLIP` | Clifford-algebra variant; separate architecture. |
| **`TransformerLayer`** | `...layers.transformers.TransformerLayer` | The block both towers are built from. |

Model methods beyond the Keras surface: `encode_image(image, training=None)` and `encode_text(text, training=None)`, both returning L2-normalized features.

---

## 7. Configuration & Model Variants

`CLIP.MODEL_VARIANTS`:

| Variant | `embed_dim` | Patch | Vision layers / width / heads / kv | Text layers / width / heads / kv |
|:---|:---:|:---:|:---:|:---:|
| **`ViT-B/32`** | 512 | 32 | 12 / 768 / 12 / 4 | 12 / 512 / 8 / 8 |
| **`ViT-B/16`** | 512 | 16 | 12 / 768 / 12 / 4 | 12 / 512 / 8 / 8 |
| **`ViT-L/14`** | 768 | 14 | 24 / 1024 / 16 / 4 | 12 / 768 / 12 / 12 |
| **`ViT-H/14`** | 1024 | 14 | 32 / 1280 / 16 / 4 | **24** / 1024 / 16 / 16 |

**`ViT-H/14` has 24 text layers, not 12.** It is the one scale where the text tower deepens; B/32, B/16 and L/14 all legitimately have 12. Copying the previous row is exactly how that gets missed — do not "restore consistency".

The `*_kv_heads` columns are **not** from any released CLIP: no CLIP checkpoint uses grouped-query attention. They are this implementation's declared modernization.

Other constructor arguments (with defaults): `image_size=224`, `vocab_size=49408`, `context_length=77`, `ffn_expansion_factor=4`, `ffn_multiple_of=256`, `dropout_rate=0.0`, `attention_dropout_rate=0.0`, `logit_scale_init=2.6592`. The constructor validates divisibility eagerly: `image_size % patch_size`, each width by its head count, each head count by its kv-head count.

---

## 8. Comprehensive Usage Examples

### Example 1: Zero-shot classification

```python
import keras
import numpy as np
from dl_techniques.models.vision_language.clip.model import create_clip_variant

model = create_clip_variant("ViT-B/32")
model.build({'image': (None, 224, 224, 3), 'text': (None, 77)})

image = np.random.rand(1, 224, 224, 3).astype("float32")
classes = ["a photo of a dog", "a photo of a cat", "a drawing of a car"]
tokens = tokenize(classes)          # your CLIP BPE tokenizer, right-padded with 0

image_features = model.encode_image(image)     # (1, 512), already L2-normalized
text_features = model.encode_text(tokens)      # (3, 512)

similarities = keras.ops.matmul(image_features, keras.ops.transpose(text_features))
probs = keras.ops.softmax(similarities, axis=-1)
print(classes[int(keras.ops.argmax(probs, axis=-1)[0])])
```

Both `encode_*` methods normalize, so the matmul is already a cosine. For a calibrated score, multiply by `keras.ops.exp(model.logit_scale)` first.

### Example 2: Caching a prompt bank

Text embeddings for a fixed prompt set are constant. Compute them once and keep only `encode_image` in the loop — the partial `call` contract exists precisely so this needs no dummy image batch.

```python
text_features = model.encode_text(tokens)          # once
for batch in stream:
    image_features = model.encode_image(batch)     # per frame
```

---

## 9. Advanced Usage Patterns

### Fine-tuning with a contrastive loss

```python
import keras

def contrastive_loss(logits_per_image):
    batch_size = keras.ops.shape(logits_per_image)[0]
    labels = keras.ops.arange(batch_size)
    loss_img = keras.losses.sparse_categorical_crossentropy(
        labels, logits_per_image, from_logits=True)
    loss_txt = keras.losses.sparse_categorical_crossentropy(
        labels, keras.ops.transpose(logits_per_image), from_logits=True)
    return (loss_img + loss_txt) / 2.0
```

`dl_techniques.losses` carries a ready-made CLIP contrastive loss; prefer it over a hand-rolled training loop.

### Clamping the temperature

This class does not clamp, so a trainer that wants OpenCLIP's behaviour must add it after each step:

```python
model.logit_scale.assign(keras.ops.minimum(model.logit_scale, keras.ops.log(100.0)))
```

### Mixed precision

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model = create_clip_variant("ViT-B/16")
```

`model.fit()` inserts the loss-scale optimizer automatically.

---

## 10. Training and Best Practices

- **Optimizer**: AdamW. The weight decay matters for both towers.
- **Schedule**: cosine decay with a short linear warmup.
- **Batch size is the main lever.** Contrastive learning scales with negatives per positive; the original CLIP trained above 32,000. On one GPU, gradient accumulation is how you approximate it.
- **Padding**: right-pad with id `0`. Pooling silently reads the wrong position otherwise (§4.2).
- **Temperature**: watch `exp(logit_scale)`. There is no clamp here.

---

## 11. Serialization & Deployment

`CLIP` and its layers register through `@register_dl_technique` and carry a complete `get_config`, so the standard round trip works:

```python
model.save('clip_vit_b32.keras')
restored = keras.models.load_model('clip_vit_b32.keras')
```

`compute_output_shape` mirrors `call`'s partial contract: it returns only the keys `call` would emit for the given input spec.

---

## 12. Testing & Validation

```python
import numpy as np
from dl_techniques.models.vision_language.clip.model import CLIP

def test_creation_all_variants():
    for variant in CLIP.MODEL_VARIANTS:
        assert CLIP.from_variant(variant) is not None

def test_forward_pass_shapes():
    model = CLIP.from_variant("ViT-B/32")
    images = np.random.rand(4, 224, 224, 3).astype("float32")
    texts = np.random.randint(1, 49408, (4, 77)).astype("int32")
    out = model({'image': images, 'text': texts}, training=False)
    assert out['image_features'].shape == (4, 512)
    assert out['logits_per_image'].shape == (4, 4)
    assert model.encode_text(texts).shape == (4, 512)
```

The package suite lives at `tests/test_models/test_clip/`.

---

## 13. Troubleshooting

**Text features look wrong / retrieval is poor.** Check padding. Pooling assumes right padding and pad id `0`; a left-padded batch pools a pad position and never raises.

**Loss becomes `nan`.** Either the learning rate (use warmup, peak around `1e-5`–`1e-4` for fine-tuning) or a diverging `logit_scale`. There is no clamp in this class — add one (§9).

**`KeyError: 'logits_per_image'`.** You passed one modality. `call` omits the logits keys unless both are present (§4.5).

**Comparing against published CLIP numbers.** Do not. This is a modernized architecture (GQA, RMSNorm, SwiGLU, RoPE instead of learned positions) with no ported weights.

---

## 14. Citation

```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya
          and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda
          and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year={2021}
}
```

The modernizations this implementation adds: Grouped-Query Attention (Ainslie et al., 2023, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)), RoPE (Su et al., 2021, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)), RMSNorm (Zhang and Sennrich, 2019, [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)) and SwiGLU (Shazeer, 2020, [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)).
