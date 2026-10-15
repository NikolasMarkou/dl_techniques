# NanoVLM

A small vision-language model in the nanoVLM layout: a ViT vision tower, a `ModalityProjection` into the text width, and ONE causal `TextDecoder` run over `[projected image tokens; text token embeddings]`, under a vocabulary head tied to the word embeddings. Its output is unnormalized **logits** over the vocabulary at every position of the joint sequence. No pretrained weights are distributed.

`research/nanoVLM_research.md` is a design proposal, not a description of this package; its status block says where the two differ.

---

## 1. Architecture

```
images [B,H,W,C]                         text_tokens [B,T]
      │                                         │
      ▼                                         ▼
VisionEncoder (output_mode='none')       TextDecoder.embed_tokens
      │ [B, 1+P, Dv]                     (word table only)
      ▼                                         │
ModalityProjection: pixel shuffle s,            │
then Dense to the text width Dt                 │
      │ [B, Sv, Dt]                             │ [B, T, Dt]
      └──────────────────┬──────────────────────┘
                         ▼
          concat on the sequence axis [B, Sv+T, Dt]
                         ▼
          TextDecoder.decode_embeddings: positions 0..Sv+T-1,
          ONE causal mask over vision and text, blocks, final norm
                         ▼
          dropout, then the vocabulary projection
          tied:   hidden @ word_embeddings.T   (no output Dense is built)
          untied: output_projection, a bias-free Dense(vocab_size)
                         ▼
          logits [B, Sv+T, vocab]
```

- `Sv` is `NanoVLM.num_vision_tokens`: `(grid / s)^2 + 1` for a `grid x grid` patch grid, pixel-shuffle factor `s` (default 2) and the CLS token. It is 50 at 224px and 145 at 384px, with 16px patches.
- **Causal everywhere.** Every position attends to itself and earlier positions only, vision tokens included. This is the reference nanoVLM's behaviour: its language model is a causal decoder over the concatenation, not a prefix-LM with a bidirectional image block. A text position therefore reads the whole image and the text before it, and nothing after it (`test_the_text_logits_are_causal.py`, at 3 blocks and on the `mini` row).
- The text logits are the trailing `T` positions. An optional `attention_mask` `(B, T)` covers the text; the vision prefix is always attended.
- `text_config['max_seq_len']` is the **text** length limit. The decoder is built with `num_vision_tokens + max_seq_len` positions, so the vision prefix never uses up text positions. A longer static text length is refused at build.
- The vision and text widths may differ; the projection maps one to the other. `projection_config` passes `scale_factor`, `use_gelu`, `use_layer_norm` and the initializers/regularizers to `ModalityProjection`. Its defaults are a 2x2 pixel shuffle, GELU and layer norm, the projector of the research doc's design.
- **Tying.** `use_shared_embedding=True` (the default) ties the output to the word embedding table at call time and builds no output `Dense`. A decoder with `embedding_type='factorized'` has no full table to tie to and refuses `use_shared_embedding=True`.
- `generate()` encodes the image once and re-runs the decoder over `[vision; text so far]` for each new token. It is a plain top-k sampling loop with no key-value cache, meant for evaluating a trained model. A prompt plus `max_length` beyond `max_seq_len` is refused.

Until 2026-10-15 a separate text tower fed a `MultiModalFusion` layer (8 strategies, default `'cross_attention'`). That design was removed, with no alias, for two measured defects. First, the symmetric cross-attention let a later text token change earlier text logits from the second fusion block on. Second, the last fusion block's vision side never trained. The removed surface: `fusion_config`, `fusion_strategy`, `text_component_type` (and the bidirectional encoder mode) and the trainer flags `--fusion-strategy` and `--num-fusion-layers`. Archives saved before 2026-10-15 do not load.

## 2. Variants and factory

`NanoVLM.MODEL_VARIANTS` holds the four rows. `vocab_size` is a caller argument.

| Variant | Image | Vision tokens | Width | Depth (vision / decoder) | Heads | Tied | Parameters |
|---|---|---|---|---|---|---|---|
| `mini` | 224 | 50 | 384 | 6 / 6 | 6 | yes | 34,761,216 |
| `base` | 224 | 50 | 768 | 12 / 12 | 12 | yes | 198,223,872 |
| `large` | 384 | 145 | 1024 | 24 / 24 | 16 | yes | 644,164,608 |
| `modern` | 224 | 50 | 768 | 12 / 12 | 12 | no | 216,855,820 |

Parameters at `vocab_size=32000` and 512 text tokens, measured with `create_nanovlm(v).build(...)` then `count_params()` on 2026-10-15. `modern` uses differential attention, RMSNorm, SwiGLU and pre-normalization in the vision tower. Its decoder uses factorized word embeddings, RMSNorm, SwiGLU, pre-normalization and learned positions. The row's former `positional_type='rope'` built no positional encoding at all and is now refused by `TextDecoder`. Its factorized table cannot be tied, so it builds `output_projection`.

```python
from dl_techniques.models.vision_language.nano_vlm import NanoVLM, create_nanovlm

model = create_nanovlm("base", vocab_size=32000)          # delegates to from_variant
model = NanoVLM.from_variant("mini", vocab_size=1000)

# vision_config / text_config / projection_config merge ONE level deep into
# the row; other keywords replace the row's value or go to the constructor.
small = NanoVLM.from_variant(
    "modern", vocab_size=64,
    vision_config={"embed_dim": 64, "num_heads": 1},
    text_config={"embed_dim": 64, "num_heads": 1},
)
```

- `pretrained` must be falsy. `True` or a path raises `NotImplementedError` before anything is built: no weights ship, and the model has no backbone/head split for a partial load. Load a local archive with `keras.models.load_model(path)`.
- An unknown keyword raises `ValueError`.
- `create_modern_nanovlm` was removed on 2026-10-15; its configuration is the `'modern'` row.

## 3. Training

The output is logits, so a loss takes `from_logits=True`. `dl_techniques.losses.NanoVLMLoss` is the next-token cross-entropy for this model: it scores the trailing `T` positions of the logits (the text positions; the vision prefix is dropped), shifts by one and ignores padding id 0. It works under a stock `compile`, and every trainable variable receives a gradient under it (`test_every_weight_trains_in_one_fit_step.py`):

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

## 4. Tests

`tests/test_models/test_nano_vlm/` covers:

- the model and the factory;
- the vision-prefix layout and shape contract;
- causality;
- gradient flow and one-step training;
- sampling.

`tests/test_train/test_nano_vlm/` runs `main(argv)` end to end on synthetic data and on image files. `tests/test_layers/test_transformers/test_the_decoder_runs_on_precomputed_embeddings.py` pins the `TextDecoder` split the model runs on.
