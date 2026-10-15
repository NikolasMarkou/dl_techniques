# NanoVLM

A small vision-language model: a ViT vision tower, a text tower (causal decoder or bidirectional encoder) and a `MultiModalFusion` layer, under one vocabulary head. Its output is unnormalized **logits** over the vocabulary at every position of the fused sequence. No pretrained weights are distributed.

`research/nanoVLM_research.md` is an unimplemented design proposal, not a description of this package.

---

## 1. Architecture

```
images [B,H,W,C]             text_tokens [B,T]
      │                            │
      ▼                            ▼
VisionEncoder               TextDecoder (causal) or TextEncoder
output_mode='none'          (optional attention_mask)
      │ [B,Sv,D]                   │ [B,T,D]
      └─────────────┬──────────────┘
                    ▼
            MultiModalFusion          8 strategies, default 'cross_attention'
                    │
   'cross_attention': concat(vision, text) on the sequence axis -> [B, Sv+T, D]
   six equal-length strategies: one fused tensor              -> [B, T, D] (Sv == T)
   'attention_pooling': pooled                                -> [B, D]
                    │
                    ▼
     dropout, then the vocabulary projection
     tied:   features @ word_embeddings.T   (no output Dense is built)
     untied: output_projection, a bias-free Dense(vocab_size)
                    │
                    ▼
     logits [B, Sv+T, vocab] / [B, T, vocab] / [B, vocab]
```

- `Sv` is `(img_size / patch_size)^2 + 1` (the CLS token): 197 at 224px, 577 at 384px.
- Only `'cross_attention'` and `'attention_pooling'` fuse streams of different length. The other six raise a `ValueError` naming the requirement unless `T == Sv`.
- `compute_output_shape` asks the fusion layer for the fused length, so it agrees with the eager call and with `model(keras.Input(...))` for all eight strategies (`tests/test_models/test_nano_vlm/test_fusion_contract.py`).
- **Tying.** `use_shared_embedding=True` (the default) with a decoder ties the output to the word embedding table at call time and builds no output `Dense`. An encoder model is always untied. A decoder with `embedding_type='factorized'` has no full table to tie to and refuses `use_shared_embedding=True`.
- `generate()` accepts only `'cross_attention'`. It is a plain top-k sampling loop with no key-value cache, meant for evaluating a trained model.

## 2. Variants and factory

`NanoVLM.MODEL_VARIANTS` holds the four rows. `vocab_size` is a caller argument; every row's `fusion_strategy` default is `'cross_attention'`.

| Variant | Image | Width | Depth (vision / text) | Heads | Fusion blocks | Tied | Parameters |
|---|---|---|---|---|---|---|---|
| `mini` | 224 | 384 | 6 / 6 | 6 | 3 | yes | 44,793,216 |
| `base` | 224 | 768 | 12 / 12 | 12 | 6 | yes | 280,859,904 |
| `large` | 384 | 1024 | 24 / 24 | 16 | 8 | yes | 841,325,568 |
| `modern` | 224 | 768 | 12 / 12 | 12 | 1 | no | 232,551,436 |

Parameters at `vocab_size=32000`, measured with `create_nanovlm(v).build(...)` then `count_params()` on 2026-10-15. `modern` uses differential attention, RMSNorm, SwiGLU and pre-normalization in the vision tower; RoPE, factorized word embeddings, RMSNorm and SwiGLU in the text tower; SwiGLU and RMSNorm in the fusion layer. Its factorized table cannot be tied, so it builds `output_projection`.

```python
from dl_techniques.models.vision_language.nano_vlm import NanoVLM, create_nanovlm

model = create_nanovlm("base", vocab_size=32000)          # delegates to from_variant
model = NanoVLM.from_variant("mini", vocab_size=1000)

# The three config dicts merge ONE level deep into the row; other keywords
# replace the row's value or go to the constructor.
small = NanoVLM.from_variant(
    "modern", vocab_size=64,
    vision_config={"embed_dim": 64, "num_heads": 1},
    text_config={"embed_dim": 64, "num_heads": 1},
    fusion_config={"attention_config": {"num_heads": 1}},
)
```

- `pretrained` must be falsy. `True` or a path raises `NotImplementedError` before anything is built: no weights ship, and the model has no backbone/head split for a partial load. Load a local archive with `keras.models.load_model(path)`.
- An unknown keyword raises `ValueError`.
- `create_modern_nanovlm` was removed on 2026-10-15; its configuration is the `'modern'` row.

## 3. Training

The output is logits, so a loss takes `from_logits=True`. `dl_techniques.losses.NanoVLMLoss` is the next-token cross-entropy for this model: it scores the trailing `T` positions of the logits (the text positions; under `'cross_attention'` the vision prefix is dropped), shifts by one and ignores padding id 0. It works under a stock `compile`:

```python
from dl_techniques.losses import NanoVLMLoss

model.compile(optimizer="adamw", loss=NanoVLMLoss(ignore_index=0))
model.fit({"images": images, "text_tokens": tokens, "attention_mask": mask}, tokens)
```

The trainer is `src/train/nano_vlm/train_nano_vlm.py` (stock `compile`/`fit`, no custom training step). Exactly one data source, `--synthetic` (in-memory samples) or `--data-json` (a Cauldron-format list of `image_path`/`question`/`answer` rows); `--output-dir` defaults to `results/nanovlm_<variant>_<timestamp>`:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.nano_vlm.train_nano_vlm \
    --synthetic --variant mini --epochs 1 --steps-per-epoch 10 --output-dir /tmp/nanovlm
```

## 4. Known limitations

Both are measured and pinned by strict `xfail` tests, so a fix shows up as an XPASS:

- **The default fusion is not causal beyond one block.** `'cross_attention'` lets every modality attend to every other one in every block. From the second block on, a text position attends to vision tokens that already attended to the whole text sequence, so a later token changes earlier text logits (`test_the_text_logits_are_causal.py`). Every row except `modern` uses 3 to 8 blocks. Next-token training on such a model can copy the answer instead of predicting it.
- **Under `NanoVLMLoss` the last fusion block's vision side does not train.** Its cross-attention, norm and FFN feed only the vision positions the loss drops, so their gradient is identically zero (`test_every_weight_trains_in_one_fit_step.py`).

## 5. Tests

`tests/test_models/test_nano_vlm/` (model, factory, fusion contract, gradient flow, sampling) and `tests/test_train/test_nano_vlm/` (`main(argv)` end to end on synthetic data and on image files).
