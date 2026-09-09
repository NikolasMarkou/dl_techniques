# `train/hnet` — H-Net byte-level pretraining

H-Net is a byte-level (`vocab_size = 256`) causal language model: no tokenizer anywhere in the
pipeline. Raw UTF-8 is packed into causal windows by `dl_techniques.datasets.byte_lm`, and the
model learns its own chunk boundaries from the cosine similarity of adjacent hidden states. This
directory holds the whole non-model half of the port.

| Module | What it does |
|---|---|
| `prepare_hnet_data.py` | The corpus manifest: reports the already-staged Wikipedia dump and creates the **gated** FineWeb-Edu `sample-10BT` slot. Downloads nothing without an explicit `--download`. |
| `common.py` | The config, the CLI flags, the tf.data byte pipeline, the optimizer, model construction and `fit()`. |
| `train_hnet.py` | The training entry point. |

**No H-Net has been trained in this repository beyond a 200-step dev-scale smoke run.** There are
no pretrained weights and no trained reference variant. What HAS run, once, is the smoke run of
2026-09-09: the `dev` layout (`["m1", ["T1"], "m1"]`, 170572 parameters — four orders below
`hnet_1stage_L`) on the staged Wikipedia dump, `--seq-len 512 --batch-size 8`, 10 epochs x 20 steps
= **200 optimizer steps in 168 s** on GPU 0, logged to
`results/hnet_dev_20260909_161050/training_log.csv`: train loss 5.4403 -> 3.5595 (ratio **0.6543**),
`val_loss` 5.1516 -> 3.4630 monotonically, and a `best_model.keras` that reloads and reproduces its
logits exactly. That is a proof the pipeline runs and the loss falls; it is **not** a trained model
and carries no quality claim. Greedy decoding from that checkpoint emits 64 consecutive spaces —
what a 170k-parameter model at ~4.99 bits/byte looks like. No accuracy, perplexity or throughput
number for any reference variant exists. `HNet.from_variant(..., pretrained=True)` raises
`NotImplementedError` naming the variant, for exactly that reason.

## Running it

```bash
# what the CLI offers -- allocates nothing, touches no GPU
MPLBACKEND=Agg .venv/bin/python -m train.hnet.train_hnet --help

# the dev-scale layout on the staged Wikipedia dump
MPLBACKEND=Agg .venv/bin/python -m train.hnet.train_hnet \
    --arch-variant dev --seq-len 512 --batch-size 8 \
    --epochs 1 --steps-per-epoch 200 --gpu 0

# corpus status; writes only the gated slot's README, transfers nothing
MPLBACKEND=Agg .venv/bin/python -m train.hnet.prepare_hnet_data --dry-run
```

Artifacts land in a timestamped `hnet_*` directory under repo-root `results/` (`--output-dir`
moves the root). Checkpoint selection is on `val_loss`, whose direction is resolved by
`train.common.resolve_monitor_mode` and never hand-written beside the monitor name.

`--arch-variant` accepts `dev` plus the six variants transcribed from the reference repository's
`configs/*.json` (`hnet_1stage_L`, `hnet_1stage_XL`, `hnet_2stage_L`, `hnet_2stage_XL`,
`hnet_2stage_XL_chinese`, `hnet_2stage_XL_code`). All seven are asserted to CONSTRUCT in
`tests/test_train/test_hnet/test_cli_contract.py`; only `dev` has ever been run. `dev` is a tiny,
non-reference layout invented here — `["m1", ["T1"], "m1"]` at `d_model = 64`, i.e. one Mamba-2
encoder layer, one chunking level, one attention layer inside, one Mamba-2 decoder layer. It is
deliberately absent from the model package's cited `MODEL_VARIANTS` table.

Everything else is `--help`. Every flag maps to one `HNetTrainingConfig` field, and that mapping
is pinned flag by flag by the CLI contract guard; `--gpu` is the single exception, consumed by
`setup_gpu` and never a config field.

## The corpus

| Corpus | Status |
|---|---|
| Wikipedia `20231101.en` | **Staged.** 41/41 shards, ~20 GB of HF `datasets` Arrow cache under `/media/arxwn/data0_4tb/datasets/wikipedia`, read through `dl_techniques.datasets.nlp.load_wikipedia_train_val`. This is the default `--dataset-root`, and it is what the run commands above train on. Measured over all 41 shards, an article averages 3053.71 bytes. |
| FineWeb-Edu `sample/10BT` | **Gated, empty.** ~28.5 GB, ODC-BY. The slot exists at `/media/arxwn/data0_4tb/datasets/fineweb_edu/sample-10BT/` with a README stating the reason and no `.ok` marker. This is not a dead host or a technical failure: the transfer is user-gated and was deliberately not run. Pass `--download` to `prepare_hnet_data.py` to fill it. |

## Scale: what this hardware can and cannot do

The Mamba-2 mixer in this repository runs a **sequential scan** — one `while_loop` iteration per
timestep — so cost grows with sequence length faster than linearly. Measured directly, on a bare
`Mamba2Layer(d_model=256, d_state=128, d_conv=4, expand=2, headdim=64)` inside a minimal model, at
`batch = 8`, timing a real forward + backward step (5 timed steps after 2 warm-up), on an RTX 4090:

| L | median s/step | ratio vs previous L |
|---|---|---|
| 512 | 0.9886 | — |
| 1024 | 1.9473 | 1.970 |
| 2048 | 4.6578 | 2.392 |
| 4096 | 12.6327 | 2.712 |
| 8192 | 38.7089 | 3.064 |

Read the last column: the per-doubling cost is itself rising (1.97 → 2.39 → 2.71 → 3.06), so the
short-sequence numbers do not extrapolate. **One block** at `L = 8192` costs 38.7 s per step; a
model with eight of them would cost roughly five minutes per step, which puts byte-level context
lengths in the reference's regime out of reach on this hardware regardless of which card is used.
The same probe on the RTX 4070 (12 GB) is 1.03–1.30x slower and OOMs in the backward pass at
`L = 4096`.

That is why `dev` exists and why the shipped defaults are `--seq-len 512 --batch-size 8`. It is
also why no reference variant has been trained here: at `d_model = 1024` with 22+ inner layers,
none of them is a run this machine can finish.

## Design notes worth knowing before editing

- **No custom `train_step`.** The auxiliary boundary-ratio loss (`alpha = 0.03`, target
  downsampling ratio 6.0) reaches the optimizer through the model's own `add_loss`, which stock
  `fit()` already sums into the compiled loss. Adding it at compile time as well would
  double-count it.
- **No per-stage learning-rate multipliers.** The reference groups parameters into per-multiplier
  optimizer groups; Keras 3's optimizer takes one learning rate for all variables. The knob is
  therefore absent from the config rather than declared and unread — see the module docstring of
  `common.py`.
- **Weight decay is the optimizer's alone.** AdamW applies it decoupled; no `kernel_regularizer`
  is ever attached, and the excluded-variable patterns (`bias`, `gamma`, `beta`) match the
  reference's own exclusion set under Keras' spelling.
- **The cosine decay horizon starts at the END of warmup**, not at step 0: `WarmupSchedule` hands
  the primary schedule `step - warmup_steps`. Passing the full step budget as `decay_steps`
  silently stretches the cosine past the end of the run.

## Where the rest lives

- Model package: `src/dl_techniques/models/language/hnet/` (architecture, variants, and the
  recorded divergences from the reference — do not restate the count here, it drifted from four to
  six once already; re-derive it with
  `grep -c '^### 5\.' src/dl_techniques/models/language/hnet/README.md`, which reads **7** today).
- Chunking layers: `src/dl_techniques/layers/dynamic_chunking/`.
- Byte pipeline primitives: `src/dl_techniques/datasets/byte_lm.py`.
- Tests: `tests/test_train/test_hnet/`, `tests/test_models/test_hnet/`,
  `tests/test_layers/test_dynamic_chunking/`, `tests/test_datasets/test_byte_lm.py`.
