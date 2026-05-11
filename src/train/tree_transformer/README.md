# Tree Transformer Training

Pretrain a TreeTransformer encoder via MLM, then fine-tune it on a downstream
classification task. Mirrors `src/train/bert/{pretrain.py, finetune.py}`
(Pattern 3 NLP).

## Prerequisites

- `.venv` virtualenv with `tensorflow==2.18`, `keras>=3.8`, `tiktoken`,
  `tensorflow-datasets`.
- IMDB reviews is auto-downloaded by `tfds` to `~/tensorflow_datasets/` on
  first use (~80MB).
- Always set `MPLBACKEND=Agg` on headless/remote machines to avoid X11
  crashes.

## Recipe 1: Pretrain smoke (CPU-runnable, <60s)

Quickest end-to-end check — `tiny` variant, 64 samples, 1 epoch.

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python -m train.tree_transformer.pretrain \
    --variant tiny --epochs 1 --batch-size 8 --max-samples 64
```

Outputs (under `results/tree_transformer_pretrain_*/`):

- `tree_transformer_mlm_final_best.keras` — full MLM model (encoder + head).
- `pretrained_tree_transformer_encoder_best.keras` — encoder-only, the
  artifact you feed into `finetune.py`.

## Recipe 2: Pretrain at scale (GPU 0)

```bash
MPLBACKEND=Agg \
    .venv/bin/python -m train.tree_transformer.pretrain \
    --variant base --epochs 3 --batch-size 32 --max-samples 50000 \
    --max-seq-length 128 --gpu 0
```

`pad_token_id` is automatically set to `100266` (tiktoken `cl100k_base` pad)
inside the trainer — do NOT pass it on the command line.

## Recipe 3: Finetune from a pretrained encoder

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python -m train.tree_transformer.finetune \
    --pretrained-encoder-path results/tree_transformer_pretrain_<TS>/pretrained_tree_transformer_encoder_best.keras \
    --epochs 1 --batch-size 8 --max-samples 64
```

Two-stage training (frozen encoder → full fine-tune) is on by default. Pass
`--no-two-stage` to skip Stage 1.

## Flags

### `pretrain.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--variant` | `tiny` | TreeTransformer variant (`tiny`/`small`/`base`/`large`) |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--max-samples` | `10000` | Max training samples (None → full) |
| `--max-seq-length` | `128` | Max sequence length |
| `--learning-rate` | `5e-4` | Peak learning rate (warmup + cosine) |
| `--gpu` | `None` | GPU device index |

### `finetune.py`

| Flag | Default | Meaning |
|------|---------|---------|
| `--pretrained-encoder-path` | (config default) | Path to a `.keras` encoder |
| `--epochs` | `None` | Total epochs (overrides stage1+stage2) |
| `--batch-size` | `16` | Batch size |
| `--max-samples` | `None` | Max training samples |
| `--max-seq-length` | `128` | Max sequence length |
| `--no-two-stage` | off | Disable two-stage training |
| `--gpu` | `None` | GPU device index |

## Notes

- Mixed precision (`mixed_float16`) is safe on the model side — `GroupAttention`
  uses a dtype-aware mask sentinel. To enable, call
  `keras.mixed_precision.set_global_policy('mixed_float16')` before creating
  the model.
- `clipnorm=1.0` is set on both `pretrain.py` and `finetune.py` to keep the
  `log`/`exp` ops in `GroupAttention` numerically stable (see README §13).
- The `pretrain.py` trainer disables the per-epoch ModelAnalyzer by default
  (`run_epoch_analysis=False`); flip the config flag to enable it on long
  runs.
