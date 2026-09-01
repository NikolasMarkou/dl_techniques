# train/cliffordnet — CliffordNet family

Training and inference for CliffordNet: attention-free, FFN-free networks whose
blocks fuse a detail stream and a depthwise-conv context stream through
`SparseRollingGeometricProduct` + `GatedGeometricResidual`
(`src/dl_techniques/layers/geometric/clifford_block.py`). Three domains are
trainable here: CIFAR classification, causal language modeling, and CLIP-style
image-text contrastive learning.

| Script | Model | What it does |
|---|---|---|
| `train_cliffordnet.py` | `CliffordNet` (`models/vision/cliffordnet/model.py`) | CIFAR-10/100 classification. |
| `train_downsampling_techniques.py` | hand-composed `CliffordNetBlock` stacks | CIFAR-100 ablation of 6 downsampling compositions. |
| `train_cliffordnet_nlp.py` | `CliffordNetLM` (`models/vision/cliffordnet/lm.py`) | Causal LM pretraining on Wikipedia or TFDS text. |
| `train_clip.py` | `CliffordCLIP` (`models/vision_language/clip/clifford_clip.py`) | Dual-encoder CLIP, both towers Clifford. |
| `infer_cliffordnet_nlp.py` | `CliffordNetLM` | Generation: nucleus, MCMC power sampling, max-swap. |
| `eval_clip_retrieval.py` | `CliffordCLIP` | COCO 2017 zero-shot retrieval, R@1/5/10 both directions. |
| `filter_cc3m_clipscore.py` | `CliffordCLIP` | CLIP-score caption filter; writes a drop-in CC3M manifest. |
| `prepare_cc3m.py` | — | Resumable CC3M extractor from HF Hub tar shards. |
| `resize_cc3m_to_ssd.py` | — | One-time resumable downscaled CC3M copy onto SSD. |

Companion note: `VARIATIONS_COMPARISON.md` (architecture-variation results).

Reference: Brandstetter, J. et al. (2025). *CliffordNet: All You Need is
Geometric Algebra*. arXiv:2601.06793v2.

---

## 1. Vision classification

```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar10 --variant nano --epochs 200 --batch-size 128 --gpu 0

# custom architecture
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar100 --variant custom \
    --channels 192 --depth 16 --shifts 1,2,4,8 --cli-mode full --ctx-mode diff
```

| Variant | Channels | Depth | Shifts | Global context |
|---|---|---|---|---|
| `nano` | 128 | 12 | 1,2 | no |
| `lite` | 128 | 12 | 1,2,4,8,16 | no |
| `lite_g` | 128 | 12 | 1,2,4,8,16 | yes |

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `cifar100` | `cifar10` or `cifar100`. |
| `--variant` | `nano` | `nano`, `lite`, `lite_g`, `custom`. |
| `--epochs` | `200` | |
| `--batch-size` | `128` | |
| `--learning-rate` | `1e-3` | Peak LR; cosine decay + linear warmup. |
| `--weight-decay` | `0.1` | AdamW decoupled; no L2 regularizer is added. |
| `--warmup-epochs` | `5` | |
| `--patience` | `30` | EarlyStopping on `val_accuracy`. |
| `--channels` / `--depth` / `--shifts` | `128` / `12` / none | `custom` variant only. `--shifts` is comma-separated. |
| `--cli-mode` | `full` | `inner`, `wedge`, `full`. |
| `--ctx-mode` | `diff` | `diff` (discrete Laplacian) or `abs`. |
| `--use-global-context` | off | |
| `--layer-scale-init` | `1e-5` | |
| `--stochastic-depth-rate` | `0.3` | |
| `--dropout-rate` | `0.0` | |
| `--random-erasing-prob` | `0.25` | |
| `--gpu` | none | |

`--image-size`, `--lr-schedule` and `--show-plots` come from the base parser
and **are not read by this script**.

Augmentation is AutoAugment (CIFAR-10 policy) + random flip + pad-4/crop-32 +
random erasing, with per-channel normalization. After training the script saves
the final and best models, validates the serialization round trip, runs
`ModelAnalyzer`, and writes curves.

Output: `results/cliffordnet_<variant>_<dataset>_<timestamp>/` with `best_model.keras`,
`cliffordnet_<variant>_<dataset>_final.keras`, `training_summary.txt`, the
config/history JSON, and the CSV log.

### Downsampling ablation

`CliffordNetBlock` is dim-preserving, so a hierarchical CliffordNet must be
composed by hand from a stem, several stages, and an inter-stage downsampler.

| Variant key | Shape |
|---|---|
| `V0_baseline_isotropic` | patch-2 stem, 12 isotropic blocks @ C=128, no downsampling |
| `V1_3stage_strided_conv` | 64-128-256, strided 3x3 Conv2D |
| `V2_3stage_avgpool` | 64-128-256, AvgPool2D(2) + 1x1 projection |
| `V3_3stage_patch_merging` | 64-128-256, Swin-style `PatchMerging` |
| `V4_4stage_aggressive` | 64-128-256-512, `PatchMerging` at each transition |
| `V5_2stage_aggressive_stem` | patch-4 stem, 128-256, depthwise-separable strided |

```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \
    --variant all --epochs 100 --gpu 0

MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_downsampling_techniques \
    --variant V3_3stage_patch_merging --smoke-test --gpu 0
```

Flags: `--variant` (`all` by default), `--epochs` 100, `--batch-size` 128,
`--learning-rate` 1e-3, `--weight-decay` 0.1, `--patience` 30,
`--warmup-epochs` 5, `--stochastic-depth-rate` 0.1, `--layer-scale-init` 1e-5,
`--dropout-rate` 0.0, `--smoke-test` (3 epochs, batch 32, no augmentation),
`--skip-save`, `--gpu`.

---

## 2. NLP pretraining

Causal LM on English Wikipedia (HuggingFace) or a TFDS text dataset.
`CausalCliffordNetBlock` consumes `(B, seq_len, D)` directly; left-padded
depthwise convolutions enforce causality, so there is no attention mask.
Tokenizer is tiktoken GPT-2 BPE (50,257) plus CLS/SEP/PAD/MASK at 50257-50260,
total vocabulary 50,261.

| Variant | Channels | Depth | Shifts | Stoch. depth |
|---|---|---|---|---|
| `nano` | 128 | 12 | 1,2 | 0.05 |
| `mini` | 192 | 12 | 1,2,4 | 0.10 |
| `base` | 384 | 18 | 1,2,4,8,16 | 0.15 |
| `large` | 512 | 20 | 1,2,4,8,16 | 0.20 |
| `xl` | 768 | 28 | 1,2,4,8,16 | 0.25 |

```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet_nlp \
    --gpu 0 --variant nano --epochs 3

MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet_nlp \
    --resume results/cliffordnet_nlp/.../checkpoints/step_0050000.keras
```

| Flag | Default | Notes |
|---|---|---|
| `--variant` | `nano` | plus `custom` with `--channels`/`--depth`/`--shifts`. |
| `--epochs` | `3` | |
| `--batch-size` | `8` | |
| `--max-seq-length` | `512` | |
| `--learning-rate` | `3e-4` | Warmup (10% of steps) + cosine decay, AdamW clipnorm 1.0. |
| `--dropout-rate` | `0.0` | |
| `--stochastic-depth-rate` | `0.1` | |
| `--tie-word-embeddings` / `--no-tie-word-embeddings` | tied | |
| `--cli-mode` / `--ctx-mode` | `full` / `diff` | |
| `--use-global-context` | off | |
| `--loss-type` | `ce` | `ce` = `MaskedCausalLMLoss`, `focal` = `FocalCausalLMLoss`. |
| `--focal-gamma` | `1.0` | |
| `--label-smoothing` | `0.0` | |
| `--dataset-source` | `huggingface` | or `tfds`. |
| `--dataset-name` | `imdb_reviews` | TFDS only. |
| `--max-samples` | none | TFDS cap. |
| `--max-train-samples` | none | HF cap. |
| `--val-fraction` | `0.02` | |
| `--min-article-length` | `0` | 0 = no filter, correct for packed CLM. |
| `--shuffle-shards` | `4` | 1 = deterministic single-thread. |
| `--hf-cache-dir` | none | |
| `--seed` | `42` | On `--resume` the data seed is shifted by the initial step. |
| `--steps-per-epoch` | none | Overrides the chunk-aware estimate. |
| `--checkpoint-every-steps` | `25000` | Wikipedia epochs are long; checkpointing is step-based. |
| `--analyze-every-steps` | `50000` | 0 disables. |
| `--max-checkpoints` | `3` | Rolling window. |
| `--resume` | none | Path to a `.keras` checkpoint. |
| `--save-dir` | `results/cliffordnet_nlp` | **Dead flag.** It reaches `TrainingConfig.save_dir` and nothing reads it. The run directory is always `results/cliffordnet_nlp_CliffordNetLM-<variant>_<timestamp>` from `create_nlp_callbacks`. |
| `--gpu` | none | |

Each checkpoint also runs a generation probe (nucleus sampling with repetition
penalty), and step-level loss/accuracy goes to CSV.

### Inference

```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.infer_cliffordnet_nlp \
    --checkpoint results/.../step_0050000.keras \
    --prompt "The history of" --method power --compare
```

Power sampling optimizes the whole trajectory rather than each token greedily:
it samples a block, then runs MCMC refinement steps that propose token swaps and
accept them by the sequence-level score at `alpha = 1/temperature`. `max_swap`
is the deterministic variant.

| Flag | Default | Notes |
|---|---|---|
| `--checkpoint` | required | |
| `--prompt` / `--prompts-file` | none | Mutually exclusive. |
| `--method` | `power` | `standard`, `power`, `max_swap`. |
| `--compare` | off | Run all three side by side. |
| `--temperature` | `0.25` | `alpha = 1/temperature`. |
| `--mcmc-steps` | `10` | Refinement steps per block. |
| `--max-tokens` | `100` | |
| `--block-num` | `8` | |
| `--top-p` | `0.92` | |
| `--repetition-penalty` | `1.3` | |
| `--output-json` | none | |
| `--gpu` | none | |

---

## 3. CLIP contrastive pretraining

```bash
# CC3M smoke: ~12.5k steps. --skip-pretrain goes straight to the CLIP stage.
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_clip \
    --variant mini --dataset cc3m --cc3m-root /path/to/cc3m \
    --skip-pretrain --max-train-samples 100000 \
    --batch-size 32 --image-size 112 --epochs 4 --gpu 0

# smoke with no data at all
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_clip --synthetic --gpu 0
```

| Flag | Default | Notes |
|---|---|---|
| `--variant` | `nano` | `nano`, `mini`, `small`, `base`, `large`. |
| `--dataset` | `coco2017` | or `cc3m`; `--synthetic` overrides both with random tensors. |
| `--coco-root` | `/media/arxwn/data0_4tb/datasets/coco_2017` | needs `train2017/`, `val2017/`, `annotations/`. |
| `--cc3m-root` | `/media/arxwn/data0_4tb/datasets/cc3m` | output of `prepare_cc3m.py`; needs `train/`, `validation/`, `*_captions.jsonl`. |
| `--batch-size` | `128` | 32 is the size proven to fit `mini` on 12 GB at 112^2. |
| `--image-size` | `112` | Single stage; there is no resolution curriculum. |
| `--context-length` | `64` | |
| `--epochs` | `10` | |
| `--peak-lr` | `5e-4` | Cosine decay + linear warmup. |
| `--warmup-ratio` | `0.03` | |
| `--weight-decay` | `0.1` | `logit_scale` is excluded from it. |
| `--loss` | `clip` | `clip` (symmetric InfoNCE) or `siglip`. |
| `--label-smoothing` | `0.1` | Set 0.0 to match the CLIP paper exactly. |
| `--head-kind` | `learned_query_residual` | `plain`, `mean_max`, `learned_query`, `learned_query_residual`. |
| `--head-cli-mode` | `full` | Clifford components in the projection head. |
| `--vision-patch-size` | `4` | |
| `--dropout-rate` | `0.1` | |
| `--tokenizer-encoding` | `gpt2` | English-trained, 50257 tokens. |
| `--text-use-global-context` | off | Global-context pooling in the text tower only. |
| `--skip-pretrain` | off | Zeros both pretrain step counts. |
| `--pretrain-vision-steps` | `50000` | CIFAR-100 vision pretraining; 0 disables. |
| `--pretrain-lm-steps` | `50000` | Wikipedia LM pretraining; 0 disables. |
| `--pretrain-vision-lr` / `-wd` / `-batch-size` | `1e-3` / `0.1` / `128` | |
| `--pretrain-lm-lr` / `-wd` / `-batch-size` | `3e-4` / `0.01` / `8` | |
| `--pretrain-lm-hf-cache` | none | For offline runs. |
| `--pretrain-lm-min-article-length` | `0` | |
| `--pretrain-lm-shuffle-shards` | `4` | |
| `--save-every-steps` | `500` | |
| `--log-every-steps` | `50` | |
| `--max-checkpoints` | `3` | Rolling `step_NNNNNNN.keras` window. |
| `--probe-every-steps` | `750` | Retrieval probe cadence; 0 disables. |
| `--probe-num-pairs` | `512` | |
| `--gamma-probe-every-steps` | `500` | Logs projection-head LayerScale gamma. No effect for `--head-kind plain`. |
| `--max-train-samples` / `--max-val-samples` | none | |
| `--cache-decoded` | off | See gotchas. |
| `--mixed-bfloat16` | off | No LossScaleOptimizer needed; `logit_scale` stays fp32. |
| `--seed` | `42` | |
| `--gpu` | none | Parsed from `sys.argv` before `import tensorflow`, so the late "GPU setup error" log line is benign. |

Output:

```
results/cliffordclip_<variant>_<timestamp>/
  checkpoints/step_NNNNNNN.keras + final.keras
  retrieval_probes/probes.jsonl
  tensorboard/{train,validation}/
  training_log.csv
  cliffordclip_<variant>.keras
  training_summary.txt
```

`CliffordCLIP` is **not** exported from
`dl_techniques.models.vision.cliffordnet` (that package exports `CliffordNet`
and `create_cliffordnet` only). Import it from
`dl_techniques.models.vision_language.clip.clifford_clip`.

### Companion tools

```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.eval_clip_retrieval \
    --checkpoint results/.../checkpoints/final.keras --max-samples 1000 --gpu 1

MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.filter_cc3m_clipscore \
    --checkpoint results/.../checkpoints/final.keras \
    --out-manifest /tmp/cc3m_filtered.jsonl --threshold 0.20 --gpu 1

MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.resize_cc3m_to_ssd \
    --src-root /media/arxwn/data0_4tb/datasets/cc3m \
    --dst-root /media/arxwn/data_fast/datasets/cc3m_144 \
    --splits train validation --workers 12
```

`eval_clip_retrieval.py`'s `--image-size` (112) and `--context-length` (64)
**must match what the checkpoint was trained with**.

### Gotchas

1. **`logit_scale` in weight decay is a silent killer.** AdamW's decoupled decay
   drives the learnable temperature to zero and flattens the softmax regardless
   of embedding quality. Confirm the startup line
   `Excluded 'logit_scale' from AdamW weight decay.` on every run.
2. **`--cache-decoded` is a RAM bomb at scale.** It caches decoded uint8 tensors
   in RAM; roughly 10-40 GB for 200k samples. Default off streams from disk and
   scales to any dataset size.
3. **I/O, not compute, bounds CC3M training.** At 32 random JPEG reads per step
   from a SATA-class disk with a dataset larger than page cache, both a 12 GB
   4070 and a 24 GB 4090 bottom out around 540 ms/step at 30-40% GPU
   utilization. For repeat runs on the same data, convert to TFRecord shards
   with `train.common.tfrecord` — sequential reads from ~256 MiB shards recover
   several times the throughput. `resize_cc3m_to_ssd.py` is the cheaper fix.
4. **There is no multi-resolution curriculum, on purpose.** An earlier version
   raised resolution mid-run; the lower-res compiled kernels stayed resident
   while the higher-res ones compiled, spiking VRAM and OOMing repeatedly on a
   12 GB card. Use one fixed `--image-size`.

> **Removed 2026-08-10.** The three denoising pipelines, the routing-LM trainer,
> the COCO multi-task trainer, the depth-estimation trainer, the embedding and
> LM U-Net trainers and the Wikipedia pretraining helper were deleted along with
> the model modules and strided `*BlockDSv2` blocks they depended on. Recover
> them from git history if needed.
