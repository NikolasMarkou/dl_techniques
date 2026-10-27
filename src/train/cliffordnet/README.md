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
| `train_cliffordnet_nlp.py` | `CliffordNetLM` (`models/language/cliffordlm/model.py`, see `models/language/cliffordlm/README.md`) | Causal LM pretraining on Wikipedia or TFDS text. |
| `train_clip.py` | `CliffordCLIP` (`models/vision_language/clifford_clip/model.py`) | Dual-encoder CLIP, both towers Clifford. |
| `infer_cliffordnet_nlp.py` | `CliffordNetLM` | Generation: nucleus, MCMC power sampling, max-swap. |
| `eval_clip_retrieval.py` | `CliffordCLIP` | COCO 2017 zero-shot retrieval, R@1/5/10 both directions. |
| `filter_cc3m_clipscore.py` | `CliffordCLIP` | CLIP-score caption filter; writes a drop-in CC3M manifest. |
| `prepare_cc3m.py` | — | Resumable CC3M extractor from HF Hub tar shards. `--dedupe` reports the shards an extraction from before 2026-10-20 holds twice (reads only); `--dedupe --apply` backs up the JSONL next to the dataset and removes them (D-196). |
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

Variants are the presets of `CliffordNet.MODEL_VARIANTS` (table with counts in
`src/dl_techniques/models/vision/cliffordnet/README.md`): `nano`, `lite`, `lite_g`,
`nano_g`, `cn_12_3`, `cn_18_5`, `cn_32_3` (the paper's CliffordNet-32), `cn_32_5`
and `cn_64_5` (the paper's CliffordNet-64), plus `custom`. The hierarchical model
(`HierarchicalCliffordNet`, see the model README) adds `hier_345` (the authors'
constructor defaults: stages of 3, 4, 5 blocks at widths 32, 64, 128) and
`custom_hier` (`--stage-depths`, `--stage-dims`, `--shifts`). Its default DropPath is
0.1, the authors' hierarchical default; no accuracy of it is published.

A flag that does not apply to the chosen architecture is refused by name, never
ignored: `--channels`, `--depth` and `--use-global-context` with a hierarchical
variant, and `--stage-depths`/`--stage-dims` with anything but `custom_hier`.
```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --dataset cifar100 --variant hier_345 --epoch-analysis-every 10 --output-dir results
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet \
    --variant custom_hier --stage-depths 2,3,4 --stage-dims 48,96,192 --shifts 1,2,4
```

**Paper v2 recipe.** The defaults are the recipe of arXiv:2601.06793v2 and its
authors' `train.py` (github.com/ParaMind2025/CAN, read through a fetch summary):

| | Paper v2 / authors | This trainer |
|---|---|---|
| epochs / batch | 200 / 128 | 200 / 128 |
| optimizer | AdamW, lr 1e-3, weight decay 0.1 | same |
| warm-up | none ("without using warm-up training epochs") | none (`--warmup-epochs 0`) |
| schedule | cosine annealing to 0, per epoch | cosine to 1e-5 (alpha 0.01), per step |
| DropPath | 0.3; 0.4 for `cn_64_5` | 0.3; 0.4 for `cn_64_5` (`--stochastic-depth-rate`) |
| augmentation | AutoAugment CIFAR-10, crop pad 4, flip, random erasing 0.25 | same |
| loss | cross-entropy, no label smoothing | same |

Earlier runs of this trainer used a 5-epoch warm-up (the old default) and, for
`--variant lite_g`, trained WITHOUT the global branch: the parser always passed
`--cli-mode full` and no global context as overrides of the preset. A flag you do
not pass now takes the preset's value, and `config.json` records what the model
actually uses.

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `cifar100` | `cifar10`, `cifar100` or `imagenet` (the extracted ImageNet-1k, via `train.common.imagenet`). |
| `--imagenet-root` / `--devkit-dir` | `/media/arxwn/data0_4tb/datasets/imagenet2012` / `<root>/devkit` | `--dataset imagenet` only. The devkit labels val. |
| `--image-size` | `224` | Used by `--dataset imagenet`; the CIFAR sets keep 32. |
| `--eval-samples` | `2000` | `--dataset imagenet` only (refused otherwise): the first N val images feed the end-of-run figures, the model analysis and the save/reload check. Validation during training and the final evaluation use all 50,000. |
| `--variant` | `nano` | A preset (list above) or `custom`. |
| `--epochs` | `200` | |
| `--batch-size` | `128` | |
| `--learning-rate` | `1e-3` | Peak LR; cosine decay, after the optional warm-up. |
| `--weight-decay` | `0.1` | AdamW decoupled; no L2 regularizer is added. |
| `--warmup-epochs` | `0` | Linear warm-up from 1e-8; the paper uses none. |
| `--patience` | `30` | EarlyStopping on `val_accuracy`. |
| `--channels` / `--depth` / `--shifts` | `128` / `12` / none | `custom` variant only (`--shifts` also for `custom_hier`). `--shifts` is comma-separated. |
| `--stage-depths` / `--stage-dims` | model defaults (3,4,5 / 32 doubling) | `custom_hier` only, comma-separated. |
| `--cli-mode` | preset's (`full`; `inner` for `cn_32_5`, `cn_64_5`) | `inner`, `wedge`, `full`. |
| `--ctx-mode` | preset's (`diff`) | `diff` (discrete Laplacian) or `abs`. |
| `--use-global-context` | preset's (on for `lite_g`, `nano_g`) | Adds the gFFN-G branch; pass it to add the branch to another preset. |
| `--layer-scale-init` | `1e-5` | |
| `--stochastic-depth-rate` | `0.3` (`0.4` for `cn_64_5`) | Maximum DropPath, linearly ramped over depth. |
| `--dropout-rate` | `0.0` | |
| `--label-smoothing` | `0.0` | Opt-in, `a` in `[0, 1)`, anything else is a parser error. `0.0` keeps the stock `SparseCategoricalCrossentropy(from_logits=True)`; above 0 the loss is `train.convnext.common.SmoothedSparseCategoricalCrossentropy`, so `loss` and `val_loss` include the smoothing and are not comparable with unsmoothed runs (accuracy is unchanged in meaning). |
| `--random-erasing-prob` | `0.25` | CIFAR only; refused with `--dataset imagenet`, which uses random-resized-crop and flip, no AutoAugment and no random erasing. |
| `--epoch-analysis-every` | `0` | Standard analysis schedule, 0 = off: the data-free weight/spectral analyzer runs before training (`epoch_000`), after epoch 1, then at N, 2N, ... into `<run>/epoch_analysis/epoch_NNN/`. |
| `--gpu` | none | |

`--image-size`, `--lr-schedule` and `--show-plots` come from the base parser
and **are not read by this script**.

Augmentation is AutoAugment (CIFAR-10 policy) + random flip + pad-4/crop-32 +
random erasing, with per-channel normalization. After training the script saves
the final and best models, validates the serialization round trip, runs
`ModelAnalyzer`, and writes curves.

Output: `results/cliffordnet_<variant>_<dataset>_<timestamp>/` with `best_model.keras`,
`cliffordnet_<variant>_<dataset>_final.keras`, `training_summary.txt`, the
config/history JSON, and the CSV log. Also `visualizations/` (`training_dashboard.png`, redrawn
during training; `confusion_matrix.png`, `per_class_metrics.png`, `confidence_calibration.png`,
`misclassifications.png`, `classification_report.json`), `model_analysis/` (once, after training)
and, with `--epoch-analysis-every`, `epoch_analysis/`. These follow the image classification
pattern in `src/train/AGENTS.md`. The validation set is the test split, so the best
checkpoint and the reported accuracy are selected on test data.

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
| `--steps-per-epoch` | none | One epoch is `steps_per_epoch` steps over the repeated stream: the TFDS stream's counted batch count, or the chunk-aware Wikipedia estimate; this flag overrides both (D-265 of plan-2026-10-14T062616-81938e81). |
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

# smoke with no data at all: every stage, two pretraining steps each
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_clip --synthetic \
    --image-size 64 --batch-size 8 --max-train-samples 64 --epochs 1 --steps-per-epoch 2 \
    --pretrain-vision-steps 2 --pretrain-vision-batch-size 8 \
    --pretrain-lm-steps 2 --pretrain-lm-batch-size 2 \
    --output-dir /tmp/cliffordclip_smoke --gpu 0
```

Pretraining is on by default (50000 steps per tower) and runs before the CLIP stage on the
same `CliffordCLIP`, through stock `fit`: the vision tower as a CIFAR-100 classifier (the
mean-pooled anchor, the head norm, a fresh `Dense`), the text tower as a causal LM. Each stage
freezes everything but its tower's backbone, chosen by sub-layer name from the tables in
`train_clip.py`; a sub-layer no table names raises. `--synthetic` covers both stages with
in-memory data, so the smoke above reaches them with no CIFAR-100 or Wikipedia on disk. A
pretraining budget shorter than one CIFAR-100 epoch is honoured (it used to train a full
epoch).

| Flag | Default | Notes |
|---|---|---|
| `--variant` | `nano` | `CliffordCLIP.MODEL_VARIANTS`: `base`, `large`, `mini`, `nano`, `nano_g`, `small`. |
| `--dataset` | `coco2017` | or `cc3m`; `--synthetic` replaces it, CIFAR-100 and Wikipedia with in-memory data. |
| `--coco-root` | `/media/arxwn/data0_4tb/datasets/coco_2017` | needs `train2017/`, `val2017/`, `annotations/`. |
| `--cc3m-root` | `/media/arxwn/data0_4tb/datasets/cc3m` | output of `prepare_cc3m.py`; needs `train/`, `validation/`, `*_captions.jsonl`. |
| `--batch-size` | `128` | 32 is the size proven to fit `mini` on 12 GB at 112^2. |
| `--image-size` | `112` | Single stage; there is no resolution curriculum. |
| `--context-length` | `64` | |
| `--epochs` | `10` | |
| `--steps-per-epoch` | one pass | Cap on CLIP steps per epoch; also sizes the cosine schedule. |
| `--peak-lr` | `5e-4` | Cosine decay + linear warmup. |
| `--warmup-ratio` | `0.03` | |
| `--weight-decay` | `0.1` | `logit_scale` is excluded from it. |
| `--loss` | `clip` | `clip` (symmetric InfoNCE) or `siglip`. |
| `--label-smoothing` | `0.1` with `clip` | Set 0.0 to match the CLIP paper exactly. Refused with `--loss siglip` (no smoothing there). |
| `--head-kind` | `learned_query_residual` | `plain`, `mean_max`, `learned_query`, `learned_query_residual`. |
| `--head-cli-mode` | `full` | Clifford components in the projection head. |
| `--vision-patch-size` | `4` | |
| `--dropout-rate` | `0.1` | |
| `--tokenizer-encoding` | `gpt2` | English-trained, 50257 tokens. LM pretraining uses it too. |
| `--text-use-global-context` | off | Global-context pooling in the text tower only. |
| `--skip-pretrain` | off | Zeros both pretrain step counts. |
| `--pretrain-vision-steps` | `50000` | CIFAR-100 vision pretraining; 0 disables. An epoch is one pass or the budget, whichever is shorter. |
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
| `--seed` | `42` | Seeds Python, numpy and Keras/TF (`set_seeds`) and the LM-pretrain loader. |
| `--output-dir` | `results/cliffordclip_<variant>_<timestamp>` | Run directory. |
| `--gpu` | none | Index within an exported `CUDA_VISIBLE_DEVICES` (D-110), selected by `setup_gpu` at the start of the run; `setup_gpu` raises if the selection or the memory growth does not take effect. |

Training is stock `compile`/`fit` on the contrastive bridge of `src/train/clip/train_clip.py`
(`compile_for_contrastive_training` with the `--loss` loss as its inner loss). Output:

```
results/cliffordclip_<variant>_<timestamp>/      (or --output-dir)
  checkpoints/step_NNNNNNN.keras + final.keras
  retrieval_probes/probes.jsonl
  tensorboard/clip/   (and tensorboard/pretrain_{vision,lm}/ when those stages run)
  training_log.csv    step rows: loss, clip_i2t_accuracy, clip_i2t_recall@5, logit_scale
  cliffordclip_<variant>.keras
  training_summary.txt
```

Every `.keras` above is the bare `CliffordCLIP`, never the training wrapper, so
`keras.models.load_model(path)` returns the model directly.

`CliffordCLIP` is **not** exported from
`dl_techniques.models.vision.cliffordnet` (that package exports `CliffordNet`
and `create_cliffordnet` only). Import it from
`dl_techniques.models.vision_language.clifford_clip.model`.

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
4. **Do not raise resolution mid-run.** A multi-resolution curriculum keeps the
   lower-res compiled kernels resident while the higher-res ones compile,
   spiking VRAM and OOMing on a 12 GB card. Use one fixed `--image-size`.
5. **The first CLIP step takes about 3.5 minutes.** Measured on the RTX 4090
   (plan-2026-10-14T062616-81938e81 D-280), `--variant nano --synthetic
   --image-size 112 --epochs 1 --steps-per-epoch 12 --skip-pretrain --seed 42`,
   batch 128, one run per row, all in one session. First step = wall time of
   training batch 0; steady = median of batches 1 to 11; ratios are against the
   A0 median (246.1 s first, 0.151 s steady):

   | Arm | First step s | x A0 | Steady s | x A0 | Epoch train loss |
   |---|---|---|---|---|---|
   | A0: stock AdamW, run 1 / run 2 | 246.2 / 246.0 | 1.00 | 0.150 / 0.152 | 1.00 | 2.16589 / 2.16589 |
   | A: `HoistedAdamW`, the schedule read once for the decay (**the default now**) | 203.7 | 0.83 | 0.155 | 1.03 | 2.16590 |
   | B: A plus Adam's scalars (`iterations + 1`, both powers, `alpha`) once per step | 190.0 | 0.77 | 0.152 | 1.01 | 2.16591 |
   | C: `keras.optimizers.SGD`, no decay (bounds the optimizer's share; not shippable) | 149.3 | 0.61 | 0.138 | 0.91 | 5.46 |
   | D: A0 with `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_llvm_module_compilation_parallelism=true"` | 205.6 | 0.84 | 0.261 | 1.73 | 2.16590 |
   | E cold: A0 with `TF_XLA_FLAGS=--tf_xla_persistent_cache_directory=<absolute dir>` | 249.1 | 1.01 | 0.153 | 1.01 | 2.16590 |
   | E warm: the same command again, cache filled | 125.1 | 0.51 | 0.149 | 0.99 | 2.16589 |

   Stock AdamW reads the learning-rate schedule once per decayed variable, so
   the first step traced one `WarmupSchedule` per variable; `_build_optimizer`
   now builds `HoistedAdamW` (`dl_techniques.optimization`), which reads it once
   per step and equals AdamW within 1e-6. The rule fixed before the runs (D-274
   (e)): first step at most 0.85x A0, the cut above twice the A0 spread (0.16 s
   here), steady step at most 1.3x A0. A and B both met it; the simpler, A,
   ships, and B's further 14 s is not taken. D cuts as much as A by skipping
   GPU autotuning but runs every later step 1.73x slower: not recommended. E is
   the cheapest relaunch: with a persistent XLA cache the second launch of the
   same configuration skips the compile (about 121 s), at an unchanged steady
   step; set `TF_XLA_FLAGS=--tf_xla_persistent_cache_directory=/absolute/dir`
   when you rerun one configuration (the cache is keyed on the graph, so a new
   variant, image size or batch compiles afresh; 4 entries, 49 MB here). What
   remains of the first step is Python tracing of the forward, backward and
   optimizer graph. History on the RTX 4070 (D-244): first step 264 s, steady
   0.25 s; without XLA 135 s and 0.66 s per later step, so XLA (the default,
   `jit_compile="auto"`) pays for itself after about 320 steps. Each
   pretraining stage compiles its own graph once more, with stock AdamW.
