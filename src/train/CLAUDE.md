# Training Scripts

Production-grade training pipelines for models in `dl_techniques/models/`. Each subdirectory corresponds to a model architecture.

## Structure

```
src/train/
├── common/              # Shared utilities (GPU, datasets, callbacks, evaluation)
│   ├── gpu.py           # setup_gpu(gpu_id), log_gpu_peak_memory(), setup_mixed_precision()
│   ├── args.py          # create_base_argument_parser()
│   ├── datasets.py      # load_dataset(), load_imagenet_dataset(), get_class_names()
│   ├── callbacks.py     # create_callbacks(), create_learning_rate_schedule()
│   ├── evaluation.py    # validate_model_loading(), run_model_analysis(), visualizations
│   ├── run_io.py        # prepare_run_dir(), save_training_history_json(), default_experiment_name()
│   ├── stats.py         # mean_std(), bootstrap_ci(), paired_permutation_test(), format_mean_std()
│   ├── nlp.py           # NLP tokenization, text datasets, warmup LR, NLP callbacks
│   ├── image_text.py    # Image-text dataset loading (COCO, CC3M)
│   ├── megadepth.py     # MegaDepth RGB+depth dataset pipeline
│   ├── tfrecord.py      # TFRecord read/write utilities
│   └── __init__.py      # Re-exports all public functions
├── convnext/            # ConvNeXt V1, V2, V2+MAE
├── cliffordnet/         # CliffordNet classification, causal-LM pretraining + inference, CLIP
├── time_series/         # Time-series trainers (grouped)
│   ├── mdn/             # Mixture Density Network forecasting
│   ├── nbeats/          # N-BEATS
│   ├── prism/           # PRISM probabilistic forecaster
│   ├── tirex/           # TiRex patch transformer
│   └── adaptive_ema/    # Adaptive EMA slope filter
├── bert/                # BERT pretrain/finetune
├── ...
└── CLAUDE.md
```

> External SOTA benchmark reference docs have moved to `research/benchmarks/` (see Reference Documents below).

## Reference Documents

The `research/benchmarks/` directory (repo root) holds external SOTA snapshots. Consult them **before** kicking off a training run at the relevant scale so you set realistic targets instead of celebrating a number that's already 10 points below open weights. Each file is dated at the top; re-pull when leaderboards move.

- **`research/benchmarks/EMBEDDINGS_BENCHMARKS.md`** — MTEB leaderboard across 33+ models grouped by parameter count (Tiny ≤50M → XL ≥7B → Proprietary API). Per-task columns (Retrieval / STS / Classif), MRL/instruct flags, license. Use for any embedding / dense-retrieval / MRL training run.
- **`research/benchmarks/LLM_BENCHMARKS.md`** — Causal-LM reference targets across 6 tiers (Tiny ≤2B → XL/MoE ≥100B → Proprietary). Columns: MMLU, GPQA Diamond, HumanEval, MATH, Arena ELO, context length. Includes a separate SWE-bench Verified table for agentic eval. Use for pretraining, instruction tuning, and reasoning-model runs.
- **`research/benchmarks/VISION_BENCHMARKS.md`** — Pure-vision targets: ImageNet-1K classification (5 param tiers), COCO detection (mAP@50:95), ADE20K / Cityscapes semantic seg (mIoU), COCO instance seg (mask AP), monocular depth (NYU / KITTI AbsRel), and video understanding (Kinetics-400/600, SSv2). Use for classification, detection, seg, depth, and pure-video runs.
- **`research/benchmarks/VLM_BENCHMARKS.md`** — Vision-language model targets across 5 size tiers (Tiny ≤4B → XL ≥40B → Proprietary). Capability families: general reasoning (MMMU / MMStar / MMBench / MathVista), document & OCR (DocVQA / ChartQA / OCRBench), hallucination (POPE / HallusionBench), grounding (RefCOCO), video VLM (Video-MME / MVBench / EgoSchema / LongVideoBench), agentic/GUI (ScreenSpot-Pro / OSWorld), and a separate CLIP-style dual-encoder table (zero-shot IN-1K, COCO/Flickr retrieval). Use for any LLaVA/Qwen-VL/InternVL-class instruction-tuned VLM or CLIP-family pretraining run.
- **`research/benchmarks/METRICS.md`** — Metric *definition* reference complementing the four leaderboard files above. 16 task families (classification, detection, semantic/instance/panoptic seg, depth, regression, time series, IR/ranking, embeddings, NLP generation, code, speech, image generation/restoration, calibration, anomaly/OOD, RL), ~122 metrics total. Each entry: 2-6 sentence description, plain-text formula, edge cases / pitfalls, typical reporting convention, plus `In dl_techniques` pointers where the repo already implements it. Consult **before** writing a custom Keras metric or interpreting a number from any leaderboard.

## File Naming

Name scripts `train_<model>.py`, **not** `train.py`. Files named `train.py` shadow the `train` package and break `from train.common import ...`.

## Imports from `train.common`

Always use the shared utilities instead of writing local versions:

```python
from train.common import (
    setup_gpu,                                  # GPU memory growth + device selection
    create_base_argument_parser,                # standard argparse with common training args
    create_callbacks,                           # EarlyStopping, ModelCheckpoint, CSVLogger, etc.
    create_learning_rate_schedule,              # cosine, exponential, constant
    load_dataset,                               # mnist, cifar10, cifar100, imagenet
    get_class_names,                            # human-readable class labels
    validate_model_loading,                     # round-trip serialization check
    convert_keras_history_to_training_history,   # for visualization framework
    create_classification_results,              # for confusion matrix, ROC/PR
    generate_comprehensive_visualizations,       # training curves, confusion matrix, etc.
    run_model_analysis,                         # full ModelAnalyzer pipeline
)
```

## Training Script Patterns

There are 4 patterns depending on the domain. Pick the one closest to your model.

---

### Pattern 1: Vision Classification (MNIST/CIFAR/ImageNet)

**Used by:** ConvNeXt, CapsNet, CoshKan, CoshNet, CliffordNet, PowerMLP, KAN, ViT, SOM, MobileNet (V1-V4)

This is the most common pattern. Uses `load_dataset()`, `create_base_argument_parser()`, and the full evaluation pipeline.

```python
from train.common import (
    setup_gpu, create_base_argument_parser, create_callbacks,
    create_learning_rate_schedule, load_dataset, get_class_names,
    validate_model_loading, run_model_analysis,
)

def train_model(args):
    setup_gpu(args.gpu)

    # Data via common
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(args.dataset)
    class_names = get_class_names(args.dataset, num_classes)

    # Model (local)
    model = create_my_model(variant=args.variant, input_shape=input_shape, num_classes=num_classes)

    # LR schedule via common
    lr = create_learning_rate_schedule(args.learning_rate, args.lr_schedule, args.epochs)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks via common
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.variant}",
        results_dir_prefix="my_model",
        monitor="val_accuracy",          # classification → val_accuracy
        patience=args.patience,
        use_lr_schedule=True,            # we handle LR externally
    )

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)

    # Post-training analysis via common
    run_model_analysis(model, (x_test, y_test), history, "my_model", results_dir)

def main():
    parser = create_base_argument_parser("Train MyModel", default_dataset="cifar10")
    parser.add_argument('--variant', type=str, default='tiny')  # model-specific args
    args = parser.parse_args()
    train_model(args)
```

**`create_base_argument_parser()`** provides: `--dataset`, `--image-size`, `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--lr-schedule`, `--patience`, `--gpu`, `--show-plots`. Add model-specific args on top.

> **Warning: Double Weight Decay** — When using `AdamW` (which applies decoupled weight decay internally), do **not** also pass `kernel_regularizer=L2(weight_decay)` to the model. This causes double weight decay: the L2 penalty inflates the loss and the optimizer applies weight decay again on the parameter update. Use either `AdamW(weight_decay=...)` alone (preferred, matches most paper recipes) or `Adam` + `kernel_regularizer=L2(...)` — never both.

---

### Pattern 2: Time-Series / Probabilistic (N-BEATS, PRISM, TiRex, MDN)

**Used by:** N-BEATS, PRISM, TiRex, MDN

These scripts use synthetic data generators (not `load_dataset()`), monitor `val_loss`, and need `TerminateOnNaN`. The analyzer is conditional on a `--deep-analysis` flag. They keep a local argparse because the base parser's `--dataset` choices don't apply.

```python
from train.common import setup_gpu, create_callbacks as create_common_callbacks
from dl_techniques.analyzer import AnalysisConfig

class MyTrainer:
    def _train_model(self, data_pipeline, exp_dir):
        # Callbacks via common — note the extended parameters
        callbacks, _ = create_common_callbacks(
            model_name="MyModel",
            results_dir_prefix=exp_dir,
            monitor="val_loss",                    # time-series → val_loss
            patience=25,
            use_lr_schedule=self.config.use_warmup, # ReduceLR only when no warmup
            include_terminate_on_nan=True,           # essential for TS/probabilistic
            include_analyzer=self.config.perform_deep_analysis,  # conditional
            analyzer_config=AnalysisConfig(          # lightweight config
                analyze_weights=True, analyze_spectral=True,
                analyze_calibration=False, analyze_information_flow=False,
                analyze_training_dynamics=False, verbose=False),
            analyzer_start_epoch=self.config.analysis_start_epoch,
            analyzer_epoch_frequency=self.config.analysis_frequency,
        )
        # Domain-specific callback (keep local)
        callbacks.append(MyPerformanceCallback(self.config, viz_dir))

        history = self.model.fit(data_pipeline['train_ds'], validation_data=data_pipeline['val_ds'],
                                 epochs=self.config.epochs, callbacks=callbacks)

def main():
    # Local argparse — base parser's --dataset doesn't apply
    parser = argparse.ArgumentParser(description="Train MyModel")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=None)
    # ... model-specific args ...
    args = parser.parse_args()
    setup_gpu(args.gpu)
```

---

### Pattern 3: NLP Pretrain/Finetune (BERT, FNet)

**Used by:** BERT pretrain/finetune, FNet pretrain/finetune

These scripts use shared NLP utilities from `train.common.nlp` for tokenization, text dataset loading, preprocessing, warmup LR schedules, and NLP-specific callbacks. Model creation and training logic remain local.

```python
from train.common import setup_gpu
from train.common.nlp import (
    create_tokenizer,              # TiktokenPreprocessor factory
    load_text_dataset,             # TFDS text datasets (e.g., imdb_reviews)
    preprocess_mlm_dataset,        # tokenize + batch for MLM pretraining
    preprocess_classification_dataset,  # tokenize + batch for classification
    create_warmup_lr_schedule,     # warmup + cosine decay
    create_nlp_callbacks,          # common callbacks with NLP defaults
)

def train_model(config):
    setup_gpu(args.gpu)

    preprocessor = create_tokenizer(config.encoding_name, config.max_seq_length, ...)
    train_ds = preprocess_mlm_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )

    model = create_my_encoder_mlm_model(config)  # local — model-specific
    lr = create_warmup_lr_schedule(config.learning_rate, config.num_epochs, steps, config.warmup_ratio)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr, ...))

    callbacks, results_dir = create_nlp_callbacks(
        model_name="BERT-tiny",
        results_dir_prefix="bert_pretrain",
        monitor="val_loss",          # NLP pretrain → val_loss
    )
    model.fit(train_ds, callbacks=callbacks, ...)

def main():
    parser = argparse.ArgumentParser(description="Pretrain BERT")
    parser.add_argument("--gpu", type=int, default=None)
    # ... NLP-specific args (variant, max-samples, etc.)
    args = parser.parse_args()
    setup_gpu(args.gpu)
```

**`train.common.nlp` API:**
| Function | Purpose |
|----------|---------|
| `create_tokenizer(encoding, max_len, ...)` | TiktokenPreprocessor with cl100k_base defaults |
| `decode_text(text)` | TF tensor → Python string |
| `load_text_dataset(name, split, max_samples, as_supervised)` | TFDS text dataset loading |
| `preprocess_mlm_dataset(ds, preprocessor, seq_len, batch)` | Tokenize + batch for MLM (no `.cache()` — D-005) |
| `preprocess_clm_dataset(ds, preprocessor, seq_len, batch)` | Concat-and-chunk packed CLM. Signature has **no** `streaming` parameter (D-004). |
| `preprocess_clm_packed_dataset(ds, encoding_name, chunk_length, batch_size, eot_token_id, ...)` | Lower-level packed CLM with `repeat=True` for explicit step-budget loops |
| `preprocess_classification_dataset(ds, preprocessor, seq_len, batch)` | Tokenize + batch with labels |
| `estimate_clm_steps_per_epoch(num_articles, max_seq_length, batch_size, override=None, avg_tokens_per_article=440)` | **Canonical** chunk-aware steps-per-epoch helper (D-001). Use this everywhere — never roll a local `_estimate_steps_per_epoch`. |
| `create_warmup_lr_schedule(lr, epochs, steps, warmup_ratio)` | Warmup + cosine decay. Now defined in `dl_techniques.optimization.schedule` and re-exported here. |
| `create_nlp_callbacks(name, prefix, ...)` | Common callbacks with TensorBoard enabled |
| `evaluate_mlm_model(mlm_model, preprocessor, test_texts=None)` | Qualitative MLM probe + `visualize_mlm_predictions`. Defaults to `DEFAULT_MLM_PROBE_TEXTS`. Shared by bert/fnet `pretrain.py`, each of which binds `evaluate_model` to it as a plain alias. |
| `run_finetune_post_training_analysis(config, model_name, create_initial_model, results_dir)` | The full `ModelAnalyzer` comparison over Initial/Best/Final snapshots. Shared by bert/fnet `finetune.py`. `create_initial_model` is a zero-arg factory, called after the analysis dir exists. `results_dir` is REQUIRED (the run dir `create_nlp_callbacks` returned) and has no default on purpose — see "FIXED: `best_sentiment_model.keras`" below. |
| `sentiment_final_model_filename(model_name)` | `f"{model_name}_sentiment_final_best.keras"`. Used at BOTH the save site and the analysis read site so the two cannot drift. |
| `best_checkpoint_path(results_dir)` (in `train/common/callbacks.py`, not `nlp.py`) | `<results_dir>/best_model.keras` — the ONE producer of the best-checkpoint path, used both by `create_callbacks`' `ModelCheckpoint` and by every reader. |

**Wikipedia/HF data path conventions** (`dl_techniques.datasets.nlp.load_wikipedia_train_val`):

- **Default `min_article_length=0`** (D-003). Packed CLM uses every token; pass 500+ only if a downstream consumer treats one document as one training example.
- **`num_shards`** (D-002). Train pipeline supports parallel tokenization shards with per-epoch reshuffle. `num_shards=1` keeps today's deterministic single-thread behaviour. Default in CLM consumers is 4.
- **`return_counts=True`** returns post-filter article counts; pass them to `estimate_clm_steps_per_epoch` for an accurate LR-schedule horizon.

**CLM consumer CLI flag conventions** — every CLM training script must expose the same four flags so users can switch scripts without relearning: `--steps-per-epoch`, `--seed`, `--min-article-length`, `--shuffle-shards`. Resume seeding (D-006): when `--resume <ckpt>` is set, derive `data_seed = config.seed + initial_step` so resumed runs see new article ordering instead of replaying the first N chunks.

#### `tree_transformer` is a deliberate NON-ADOPTER of the bert/fnet scaffold

`src/train/tree_transformer/` looks like a third Pattern-3 package and its own README says it "mirrors" bert. **It is not folded into the shared bert/fnet scaffold, on purpose.** Do not "finish the job" by merging it without re-reading this section.

Measured (`diff -u0 A B | grep -c '^[+-][^+-]'`, at the post-consolidation tree; re-derive before quoting):

| Pair | `finetune.py` | `pretrain.py` |
|---|---|---|
| bert vs fnet | 56 | 55 |
| tree vs bert | 209 | 168 |
| tree vs fnet | 207 | 167 |

tree_transformer is ~4x further from bert than fnet is, and **equidistant from both** — it is its own branch, not a copy-of-bert with the model swapped. The distance is whole features, not naming:

- `tree_transformer/finetune.py` has **no** `post_training_analysis`, no `prepare_data_for_analyzer`, and zero references to `ModelAnalyzer` / `AnalysisConfig` / `DataInput` (grep: 0 hits in the file).
- `tree_transformer/pretrain.py` has **no** `evaluate_model` and never calls `visualize_mlm_predictions`.
- Its `FinetuneConfig` declares 18 annotated fields to bert's 24; the **6** it drops are exactly the analysis block's: `full_analysis_dir`, `run_epoch_analysis`, `analysis_start_epoch`, `analysis_epoch_frequency`, `run_post_training_analysis`, `analysis_n_samples`. It adds none.

Folding it in would therefore mean inventing config toggles for code that **does not exist** — an all-or-nothing code-PRESENCE difference cannot be expressed as a flag without first writing the missing features, against a package that had zero test coverage. (bert and fnet are now covered by `tests/test_train/test_bert_fnet/`; tree_transformer still is not.)

#### Drifts between bert / fnet / tree_transformer that were deliberately NOT harmonized

These are real divergences, recorded rather than fixed, because the consolidation that found them was behaviour-preserving. Each is pinned or documented so it cannot change by accident:

| Drift | bert | fnet | tree_transformer |
|---|---|---|---|
| `finetune.py` `max_seq_length` default (line 58 in all three) | **256** | **128** | 128 |
| `finetune.py` `stage1_epochs` / `stage2_epochs` | 5 / 10 | 5 / 10 | **2 / 3** |
| `finetune.py` optimizer `clipnorm` | absent | absent | **1.0** (`finetune.py:133`) |
| MLM `steps_per_epoch` fallback | `max_samples // batch_size if max_samples else 1000` | same | **`max(1, (max_samples or 10000) // batch_size)`** |
| Seeding in `pretrain.py` | `set_seeds(42)` | `set_seeds(42)` | `set_seeds(42)` — **drift RESOLVED 2026-08-13**; was inline `tf.random.set_seed(42)` + `keras.utils.set_random_seed(42)` in `train_tree_transformer_mlm()` |

The `max_seq_length` 256-vs-128 split is the one a shared scaffold is most likely to "tidy away": it doubles or halves bert's fine-tuning input length with no visible justification in either file. It is pinned as a FACT by `tests/test_train/test_bert_fnet/test_finetune_scripts.py::TestArgvToConfig` — harmonizing either value turns that guard RED.

**CORRECTED (2026-08-13): the seeding row is no longer a drift, and the reason it was recorded as one was wrong.** This section used to claim "tree_transformer's inline seeding is weaker than it looks: `set_seeds` also sets `PYTHONHASHSEED`, `random` and `numpy`, which the inline pair does not — so tree_transformer runs are less reproducible than bert/fnet's despite looking equivalent." That claim is **REFUTED by measurement**, and it is kept here rather than deleted so a reader who remembers it knows it was tested:

- `keras.utils.set_random_seed` *already* seeds Python `random`, NumPy and TF — it is literally `random.seed(s); np.random.seed(s); tf.random.set_seed(s)` (installed `keras/src/utils/rng_utils.py`). The explicit `random`/`numpy` calls inside `set_seeds` are redundant with its own last line; `src/train/common/seed.py:7-10` says so in its docstring.
- The only real difference, `os.environ.setdefault("PYTHONHASHSEED", ...)`, is a **documented no-op after process start** — `set_seeds`'s own docstring says so (`src/train/common/seed.py`, cited by symbol: "PYTHONHASHSEED first (best-effort, no-op once Python is up)"). It buys no reproducibility at all.
- Measured A/B, two fresh subprocesses per site, drawing `random.random()`, `np.random.rand(3)`, `tf.random.uniform((3,))` and `keras.random.normal((3,))`: the two routes are **RNG-stream identical** at every one of the 7 migrated sites (`0.6394267984578837` / `0.3745401188473625` / `0.6645621061325073` / `0.08454562723636627`). The comparator was RED-proven — a different seed on one side fires `ASSERT-DRAWS-BIT-IDENTICAL` at all 7.
- **"RNG-stream identical", not "no behaviour change" — two non-draw deltas exist and are worth naming.** `set_seeds` emits a `logger.info("[seed] all RNGs seeded with seed=…")` line the inline pairs did not (at the two `logic/` sweep sites that is once per seed, inside `run_cell()`), and it runs `os.environ.setdefault("PYTHONHASHSEED", str(seed))` in seven more processes — which at the loop sites binds the env var to the FIRST iteration's seed and leaves it set. Neither changes a draw; both are process state, and the claim is about draws.

So the two routes were always equally reproducible; the divergence was cosmetic, not behavioural. It is now gone anyway: `pretrain.py` calls `set_seeds(42)` inside `train_tree_transformer_mlm()` (cited by enclosing function, not by line number — the previous citation, `pretrain.py:170-171`, had drifted to `:221-222` before anyone noticed). `finetune.py` already used `set_seeds(42)`.

Six sibling files were migrated in the same change for the same measured reason: `cliffordnet/train_cliffordnet_nlp.py`, `logic/train_e4_monks.py`, `logic/train_e4_lowdata_mux.py`, `wave_field/train_memory.py` (since DELETED — see § (g)), `wave_field/pretrain.py`, `vae/evaluate_samplers.py`. The two `logic/` calls stay **inside** `run_cell()`, which the `for seed in seeds:` sweep calls once per seed — hoisting them to `main()` is a behaviour change, not a tidy-up, and the probe's second control confirms it (hoisting changes the per-iteration draw sequence, firing `ASSERT-LOOP-PER-ITERATION-SEQUENCE-UNCHANGED`).

#### FIXED: `best_sentiment_model.keras` was a filename nothing wrote

Recorded here because the fix established a rule worth keeping. **The defect:** `post_training_analysis` loaded `<config.save_dir>/best_sentiment_model.keras`, a name with several READ sites and **zero WRITE sites anywhere in `src/`**, while `ModelCheckpoint` wrote `<results_dir>/best_model.keras` (`train/common/callbacks.py`) — a different DIRECTORY *and* a different FILENAME, `results_dir` being the timestamped run dir `create_nlp_callbacks` returns. `run_post_training_analysis` defaults to `True`, so `train.bert.finetune` and `train.fnet.finetune` raised `ValueError: File not found` at the end of **every** default run, after training had finished and the final model had already been saved. `train/bert/deploy.py` carried a third spelling of the same dead name (with an extra `checkpoints/` component), so `python -m train.bert.deploy` with no arguments could only ever report the model missing.

**The rule that replaced it:** a checkpoint path has exactly ONE producer, and both ends of the contract call it.

- `train/common/callbacks.py` owns `BEST_CHECKPOINT_FILENAME` + `best_checkpoint_path(results_dir)`. `create_callbacks` configures `ModelCheckpoint` with it; `run_finetune_post_training_analysis` reads through it. They cannot disagree.
- `run_finetune_post_training_analysis` takes `results_dir` as a **required** parameter — deliberately with no `config.save_dir` fallback, since a convenience default is what would make the silent-miss reachable again.
- `finetune_sentiment_model` returns `(model, history, results_dir)`; `main()` threads the run dir into the analysis call.
- `deploy.py`'s `--model_path` default is `os.path.join(FinetuneConfig.save_dir, sentiment_final_model_filename("bert"))` — both halves imported from the fine-tuning script's own save site, never re-typed. It points at the FINAL model; the best-val snapshot lives in a timestamped dir `deploy` cannot know, so pass `--model_path` for that.
- Guarded by `tests/test_train/test_bert_fnet/test_analysis_reads_what_training_writes.py`, which compares the two PRODUCERS (the `ModelCheckpoint`'s configured `filepath` vs the path the analysis hands `keras.models.load_model`) rather than pinning a path literal — a literal would re-create the lockstep invariant that caused this.

Same shape, same reason as `sentiment_final_model_filename(model_name)`: if a filename must be known in two places, it is a function, not a string typed twice.

#### FIXED: tree_transformer's pretrain -> finetune hand-off

**The defect (F-24, measured 2026-08-12 on a real 1-step run).** `pretrain.py` wrote the encoder ONLY to `os.path.join(results_dir, ...)`, where `results_dir` is the TIMESTAMPED directory `create_nlp_callbacks` returns, while `finetune.py` defaulted `pretrained_encoder_path` to the STATIC `results/tree_transformer_pretrain/...` — a directory `pretrain.py`'s own `os.makedirs(config.save_dir)` created and then left EMPTY. So a plain `pretrain` followed by a plain `finetune` could not work without `--pretrained-encoder-path`. This was the INVERSE of the folklore that tree_transformer had fixed a run-directory bug bert still has: bert's `pretrain` writes its encoder to the static `config.save_dir` precisely so `finetune` can name it, which is why bert's hand-off works.

**The fix.** Same rule as `best_checkpoint_path` above — one producer, both ends call it:

- `pretrain.PRETRAINED_ENCODER_FILENAME` + `pretrained_encoder_path(root)` are the sole producer of the name.
- `pretrain.save_pretrained_encoder(encoder, results_dir, save_dir)` writes BOTH copies: the timestamped one (that run's own evidence) and the static one (the hand-off `finetune` reads). Keep both — dropping the run copy loses provenance, dropping the static one re-breaks the default.
- `finetune.FinetuneConfig.pretrained_encoder_path` is DERIVED from `pretrained_encoder_path(TrainingConfig.save_dir)`, not typed as a literal.
- `os.makedirs(config.save_dir)` in `pretrain.py` is no longer vestigial — it prepares the hand-off directory it always looked like it was preparing.

Guarded by `tests/test_train/test_tree_transformer/test_encoder_handoff.py` (5 tests — the first tests of any kind for `src/train/tree_transformer/`), which EXECUTES the save through a recording stand-in rather than grepping the source. That distinction is load-bearing: the first draft of this guard checked the source for `pretrained_encoder_path(config.save_dir)` and passed against a mutation that computed the path and never wrote to it — the exact shape of the original defect.

#### FIXED: `tree_transformer/finetune.py`'s dead `save_dir`

`FinetuneConfig.save_dir` was declared and read exactly once — by an `os.makedirs(config.save_dir, exist_ok=True)` that created `results/tree_transformer_sentiment_finetune/` and never wrote a byte into it. Every artefact goes to the timestamped `results_dir` instead. Setting `save_dir` therefore made an empty directory appear and changed nothing else — a knob that silently does nothing, the same class as the CLI args that silently no-op when `main()` forgets to forward them.

Both the `makedirs` and the field are removed. (`pretrain.py`'s identically-spelled `os.makedirs(config.save_dir)` is *not* dead — it prepares the encoder hand-off directory; see above.)

Guarded generally, not just for this field, by `test_encoder_handoff.py::test_finetune_declares_no_config_field_it_never_reads`: every annotated `FinetuneConfig` field must be read somewhere in the module. **The guard's read-set is scoped to accesses on the config object** (`config.X`, `FinetuneConfig.X`, `self.X`) rather than every `Attribute` node in the module — measured necessity, not caution: the first draft counted module-wide attribute names, so the unrelated `_PRETRAIN_SAVE_DIR = _PretrainConfig.save_dir` at the top of the file put `save_dir` in the read-set and the guard passed against the very mutation it exists to catch.

#### FIXED: dead config fields across `src/train/` (27 removed, 0 pinned — `KNOWN_DEAD` is empty)

**The defect class.** A dataclass field is declared, sometimes documented, sometimes written into `config.json` — and NOTHING ever reads it. A user who sets it sees nothing happen. It is the same class as a CLI arg that silently no-ops because `main()` forgets to forward it, and the same class as `FinetuneConfig.save_dir` above. Twenty instances were removed in `360f3addf` (five) and `774084d48` (fifteen), and the **seven** the guard itself found were removed afterwards in `cf0b22d91`, `80727c128` and `508de5edb` — twenty-seven in total. Nothing read any of them, so no runtime behaviour changed; the only observable difference is that `config.json` no longer records the removed keys.

Two sub-classes are worth naming because each has its own tell:

- **The unimplemented promise.** The field's own docstring/comment describes a CAPABILITY the code never implements — `CopyTaskConfig.min_sequence_length` ("for variable-length tasks"), `AssociativeRecallConfig.min_items`/`max_items` ("for variable difficulty" / "for capacity testing"), `PretrainConfig.load_from_disk` ("If True, looks for local arrow files" — loading is unconditionally `streaming=True`), `TrainingConfig.mixup_alpha` (no mixup exists). Deleting the field deletes the promise, including its `:param:` line. Inventing the feature to justify the field would be the wrong repair.
- **Recorded but inert.** The field reaches a whole-config dump (`save_config_json` / `asdict` / `prepare_run_dir`) and nothing else. **Serialization is NOT consumption.** The worst of these were resnet's and vit's `save_model_checkpoints`: checkpoints ARE written, unconditionally, by `train.common.create_callbacks`, so setting the field `False` wrote `false` into `config.json` while checkpointing carried on. `save_best_only` had the identical shape — `common/callbacks.py` hard-codes `save_best_only=True` — and was deleted from both classes in `80727c128` for the same reason. **DELETE, not WIRE**: wiring it means adding a parameter to `create_callbacks`, which has 22 call sites across 21 files, for a knob no reachable invocation sets (no `--save-best-only` flag, no `TrainingConfig(save_best_only=...)` construction in `src/` or `tests/`).

**The guard: `tests/test_train/test_config_fields_are_live.py`** (12 collected: 9 `REGISTERED` classes + 2 scoping tests + 1 empty-parameter placeholder). For every config class in its `REGISTERED` list, every annotated field must have a consumption site. Add a row when you fix a config; never delete one.

Three design points, each forced by a measured failure:

1. **The read-set is scoped to the config object** (`self.X`, `config.X`, `cfg.X`, `*_config.X`, `<ClassName>.X`, or through one `.config`/`.cfg` hop) — never a bare `<anything>.X`. Same lesson as the `save_dir` guard above, asserted directly by `test_the_read_set_rejects_an_unscoped_receiver` so the scoping cannot be loosened silently. **The consumption test is an AST walk, and the first draft's was not — which is why that scoping was fiction for a day.** That draft matched four regex alternatives against raw lines, and the one meant for a kwarg at a construction site, `\b<field>\s*=[^=]`, carried no receiver at all. Measured under it: `unrelated_object.save_best_only = 5` counted as consumption, a bare local `save_best_only = 5` counted, and a prose line merely CONTAINING `--save-best-only` counted (the CLI-flag alternative was an unanchored substring match). Any of those three anywhere in an importer would have kept a genuinely dead field green. The four routes are now AST shapes — a scoped `ast.Attribute`, an `ast.keyword` named for the field, an `ast.Constant` equal to the field name, an `ast.Constant` equal to `--field-name` — and `test_an_unscoped_assignment_is_not_consumption` pins the closed hole. A Store-context scoped attribute (`config.total_steps = args.total_steps`) still counts on purpose: that is how the CLI wiring consumes a field.
   **A construction-site kwarg counts as consumption even when the field is never read.** That is deliberate (a field wired by any route is live), but know what it buys you: for a frozen spec dataclass built in one place, like `RunSpec`, EVERY field passed at construction is credited regardless of receiver naming. The receiver rule only becomes load-bearing for a field that is read off the object but never passed.
2. **The search is importer-scoped, not repo-wide.** It covers the declaring module plus every file that actually IMPORTS the class from it — module-local would produce false positives (the NTM task configs are consumed by `harness.py`), while a repo-wide name grep produces false NEGATIVES. That is not hypothetical: the scratch detector that motivated this work reported `CIFARSOMConfig.perceptual_weight` alive because `losses/image_restoration_loss.py` has an unrelated `self.perceptual_weight`, and `CIFARSOMConfig.checkpoint_frequency` alive because `blt/train_blt.py` has an unrelated `self.config.checkpoint_frequency`. Both are dead. Matching the module (not just the class name) is also what keeps resnet's and vit's two distinct `TrainingConfig` classes apart.
3. **`KNOWN_DEAD` pinned rather than ignored, and is now EMPTY.** Seven further dead fields found by the guard itself were outside the removal sweep's declared scope: `CIFARSOMConfig.perceptual_weight`, `CIFARSOMConfig.checkpoint_frequency`, `CopyTaskConfig.max_sequence_length` (with its `:param:` line), `ExperimentConfig.csv_filename`, `RunSpec.csv_filename` (`rms_variants_train/sweep.py` — found later, and NOT the live twin the `ExperimentConfig` exemption comment claimed it was; it was dead by the same test) and `save_best_only` on both `TrainingConfig`s. All seven are now DELETED, all six classes that held them are in `REGISTERED`, and `KNOWN_DEAD == {}` — the documented goal state, reached, not an exemption list quietly discarded. `test_known_dead_fields_are_still_dead` survives as one skipped empty-parameter placeholder; **that skip is not evidence of anything** (an all-skip test reads as green). The evidence that those six classes are still covered is a re-add mutation per class, below. Do not add a row here to make a newly-dead field go green: wire it or delete it.

RED-proven with mutations, each restored from `cp` backups verified by `sha256sum -c` (never `git checkout --`).

When the guard landed: M1-M8 re-add one removed field per touched config class, M9 adds an unrelated `unrelated_unused_knob` (isolating: the guard is not keyed to one field name), and all nine fire `test_config_declares_no_field_it_never_consumes[<class>]` naming the field. M10 WIRED an exempted field (`_ = config.save_best_only`) and fired the *other* assertion, `test_known_dead_fields_are_still_dead[train.vit.train_vit.py-TrainingConfig]`.

**M10 is not reproducible any more and is kept only as history**: it mutated `save_best_only`, a field that no longer exists, and `KNOWN_DEAD` is empty, so `test_known_dead_fields_are_still_dead` has no parameters left to fire. Do not re-run it as written.

When `KNOWN_DEAD` was emptied: one re-add mutation per formerly-exempt class — `CIFARSOMConfig.perceptual_weight`, `CopyTaskConfig.max_sequence_length`, `ExperimentConfig.csv_filename`, resnet's and vit's `TrainingConfig.save_best_only`, and `RunSpec.csv_filename` — each firing `test_config_declares_no_field_it_never_consumes[<class>]` naming that field and only that one (`1 failed, 10 passed, 1 skipped` every time). Two further mutations prove the AST scoping: dropping the receiver check fires BOTH `test_the_read_set_rejects_an_unscoped_receiver` and `test_an_unscoped_assignment_is_not_consumption`; re-adding an unscoped bare-`ast.Name` route (the old regex's kwarg alternative) fires the latter on `save_best_only = 5`. One mutation deliberately did NOT fire and is recorded rather than hidden: renaming `sweep.py`'s `run_cfg` receiver back to a bare `spec` leaves the guard green, because `RunSpec`'s construction-site kwargs already credit every field — see design point 1.

---

### Pattern 4: Denoising / Detection (BFCNN, BFUNet, YOLO12, ResNet-ImageNet)

**Used by:** BFCNN, BFUNet, YOLO12-COCO, ResNet, DarkIR

These scripts use file-based datasets (not `load_dataset()`), monitor `val_loss` or domain metrics (`val_psnr`), and have domain-specific callbacks (visualization, deep supervision scheduling). They wrap `create_callbacks()` and append domain callbacks.

> **BFCNN note:** the flat bias-free ResNet (BFCNN) denoiser is now trained via `train/bfunet/train_bfcnn_denoiser.py` — a thin `common.py` consumer (alongside the ConvUNeXt and plain U-Net denoisers), NOT a separate `train/bfcnn/` package. That directory was removed; the callbacks/dashboard/curriculum come from the shared `common.train()` substrate rather than the local wrap-`create_callbacks` shape shown below. The example remains a valid illustration of that shape for the other Pattern-4 users.

```python
from train.common import setup_gpu, create_callbacks as create_common_callbacks

def create_callbacks(config, val_directories, num_outputs):
    """Common callbacks + domain-specific denoising/detection callbacks."""
    callbacks, results_dir = create_common_callbacks(
        model_name=config.experiment_name,
        results_dir_prefix="bfcnn",
        monitor="val_loss",                  # denoising → val_loss
        patience=config.early_stopping_patience,
        use_lr_schedule=True,
        include_tensorboard=True,            # optional
        include_analyzer=False,              # disable for sub-stages if needed
    )
    # Domain-specific callbacks (keep local)
    if config.enable_deep_supervision and num_outputs > 1:
        callbacks.append(DeepSupervisionWeightScheduler(config, num_outputs))
    callbacks.append(MetricsVisualizationCallback(config))
    callbacks.append(StreamingResultMonitor(config, val_directories))
    return callbacks, results_dir
```

---

### Pattern 5: Depth Estimation (MegaDepth)

**Used by:** `src/train/depth_anything/train_depth_anything.py` — the only surviving
Pattern-5 trainer. The original exemplar was `train.cliffordnet.train_depth_estimation`,
which the depth_anything trainer was derived from 1:1; that trainer and the
`CliffordNetUNet` / `create_cliffordnet_depth` model behind it were deleted on
2026-08-10. Read `train_depth_anything.py` for the live version of everything below.

These scripts use `train.common.megadepth` for the MegaDepth RGB+depth dataset pipeline, depth-specific metrics from `dl_techniques.metrics.depth_metrics`, and visualization callbacks from `dl_techniques.callbacks.depth_visualization`. They monitor `val_loss` and use `optimizer_builder` / `learning_rate_schedule_builder` from `dl_techniques.optimization`.

```python
from train.common import setup_gpu, create_callbacks as create_common_callbacks
from train.common.megadepth import (
    discover_megadepth_pairs,
    load_and_process_pair,
    MegaDepthDataset,
)
from dl_techniques.models.depth_anything.model import create_depth_anything
from dl_techniques.metrics.depth_metrics import AbsRelMetric, DeltaThresholdMetric
from dl_techniques.callbacks.depth_visualization import (
    DepthPredictionGridCallback,
    DepthMetricsCurveCallback,
)

def train_model(config):
    setup_gpu(args.gpu)

    # Data via common
    rgb_paths, depth_paths = discover_megadepth_pairs(config.megadepth_root)
    train_ds = MegaDepthDataset(
        train_rgb, train_depth,
        batch_size=config.batch_size,
        patch_size=config.patch_size,
    )

    # Model
    model = create_depth_anything(
        encoder_type=config.encoder_type,
        image_shape=(config.patch_size, config.patch_size, 3),
    )

    # Compile with depth-specific loss + metrics from dl_techniques
    model.compile(
        optimizer=optimizer,
        loss=DepthEstimationLoss(...),       # local — domain-specific
        metrics=[AbsRelMetric(), DeltaThresholdMetric(1.25)],
    )

    # Callbacks: common + depth visualization from dl_techniques
    callbacks, results_dir = create_common_callbacks(monitor="val_loss", ...)
    callbacks.append(DepthMetricsCurveCallback(output_dir=viz_dir))
    callbacks.append(DepthPredictionGridCallback(
        val_rgb=..., val_depth=..., val_mask=..., output_dir=viz_dir,
    ))

    model.fit(train_ds, callbacks=callbacks, ...)
```

**`train.common.megadepth` API:**
| Function/Class | Purpose |
|----------------|---------|
| `discover_megadepth_pairs(root, max_files)` | Scan MegaDepth directory for matched RGB+HDF5 depth pairs |
| `load_and_process_pair(rgb_path, depth_path, patch_size, ...)` | Load, crop, normalize, augment one RGB+depth pair |
| `MegaDepthDataset(rgb_paths, depth_paths, batch_size, patch_size, ...)` | `keras.utils.PyDataset` with multiprocessing for batched loading |

#### Pretrained backbone init (`--init-from`)

`train_depth_anything.py` supports initializing from a saved `.keras` model — typically a self-supervised pretraining checkpoint:

```bash
MPLBACKEND=Agg python -m train.depth_anything.train_depth_anything \
    --encoder-type vit_b --epochs 100 --batch-size 16 --patch-size 384 \
    --init-from results/depth_anything_pretrain_*/model_inference.keras \
    --seed 42 \
    --gpu 0
```

Under the hood this calls `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint` which, after `model.build()` and before the probe forward pass, loads weights layer-by-layer (skipping any layer whose name starts with the model's head prefix — `dpt_decoder` for Depth Anything).

**Gotcha — Keras 3.8 `.keras` + `by_name` is broken.** `model.load_weights(path.keras, by_name=True, skip_mismatch=True)` raises `ValueError("Invalid keyword arguments: {'by_name': True}")` in Keras 3.8+ — the `by_name` path is only supported for legacy `.h5`/`.hdf5` files. Use `load_weights_from_checkpoint` (full-model load + layer-by-layer `set_weights`) for name-based transfer from `.keras` checkpoints. Re-measured 2026-08-10 (`grep -rn "by_name=by_name" src/dl_techniques/`): **six** `load_pretrained_weights` helpers still forward the flag straight into `self.load_weights` and carry this latent bug — `models/convnext/convnext_v1.py`, `models/resnet/model.py`, `models/distilbert/model.py`, `models/modern_bert/model.py`, `models/bert/bert.py`, `models/fnet/model.py`. The three previously listed here (`cliffordnet/model.py`, `bias_free_denoisers/bfunet.py`, `convnext/convnext_v2.py`) have since been migrated to `load_weights_from_checkpoint` and now ignore the argument — do not use them as evidence of the bug.

**Reproducibility.** `--seed <int>` (default 42) seeds Python/NumPy/TF/Keras at startup so two runs with the same seed have bitwise-identical initialization. `MegaDepthDataset` does not currently expose a seed, so dataset shuffle ordering is not reproducible — acceptable for baseline-vs-pretrained-init comparison (init differences dwarf shuffle differences at realistic run lengths).

---

## `create_callbacks()` Full API Reference

```python
create_callbacks(
    model_name: str,                          # used in directory naming + analyzer
    results_dir_prefix: str = "model",        # results/{prefix}_{name}_{timestamp}/
    monitor: str = 'val_accuracy',            # metric for EarlyStopping + ModelCheckpoint
    patience: int = 15,                       # EarlyStopping patience
    use_lr_schedule: bool = True,             # True = skip ReduceLROnPlateau
    analyzer_epoch_frequency: int = 1,        # EpochAnalyzerCallback frequency
    include_tensorboard: bool = False,        # add TensorBoard callback
    include_terminate_on_nan: bool = False,    # add TerminateOnNaN (first in list)
    include_analyzer: bool = True,            # add EpochAnalyzerCallback
    analyzer_config: Optional[AnalysisConfig] = None,  # custom analyzer settings
    analyzer_start_epoch: int = 1,            # delay analyzer start
) -> Tuple[List[Callback], str]              # (callbacks, results_dir)
```

**Always included:** EarlyStopping, ModelCheckpoint, CSVLogger.

**Optional (via parameters):**
| Parameter | Callback | When to use |
|-----------|----------|-------------|
| `include_tensorboard=True` | TensorBoard | NLP, denoising, detection scripts |
| `include_terminate_on_nan=True` | TerminateOnNaN | Time-series, probabilistic models |
| `include_analyzer=True` (default) | EpochAnalyzerCallback | Most scripts. Set `False` for sub-stages of multi-stage training |
| `use_lr_schedule=False` | ReduceLROnPlateau | When NOT using an external LR schedule |

**Monitor values by domain:**
| Domain | monitor | mode |
|--------|---------|------|
| Classification | `val_accuracy` | max |
| Denoising / Segmentation | `val_loss` | min |
| Time-series / NLP pretrain | `val_loss` | min |
| Detection | `val_loss` | min |
| Custom metric | `val_psnr`, `val_f1`, etc. | auto (`max` if 'accuracy' in name, else `min`) |

## What Lives in `train.common` vs. Locally

**Use from `train.common`:**
- `setup_gpu(gpu_id)` — GPU memory growth + device selection. Always pass `args.gpu`.
- `create_callbacks(...)` — standard callbacks. See API reference above.
- `create_base_argument_parser(description, default_dataset)` — standard argparse. Only for vision/classification scripts that use `load_dataset()`.
- `create_learning_rate_schedule(lr, type, epochs, steps_per_epoch)` — cosine, exponential, constant. **Now defined in `dl_techniques.optimization.schedule`** and re-exported from `train.common`; both import paths work and resolve to the same object.
- `load_dataset(name, batch_size, image_size)` — MNIST, CIFAR-10/100, ImageNet only.
- `get_class_names(dataset, num_classes)` — human-readable labels.
- `validate_model_loading(path, sample, expected, custom_objects)` — round-trip serialization check.
- `run_model_analysis(model, test_data, history, name, results_dir, config)` — full ModelAnalyzer pipeline.
- `discover_megadepth_pairs(root)`, `MegaDepthDataset(...)` — MegaDepth RGB+depth dataset pipeline.
- `compare_runs(run_a_dir, run_b_dir, labels, output_dir)` — side-by-side two-run comparison. CLI: `python -m train.common.compare_runs A B [--labels A B]`. Emits `comparison.md` + PNG loss/metric curves.
- `StepCheckpointCallback(save_dir, save_every_steps, analyze_every_steps=0, max_checkpoints, model_name, initial_step, log_every_steps, plot_every_steps, step_counter=None, gc_on_save=False, csv_fields=None)` — step-indexed CSV logging + rolling `.keras` checkpoint window + optional periodic ModelAnalyzer (`analyze_every_steps=0` disables it) + step-loss plots. Pass an external `step_counter` for resume/shared-counter setups (else an internal counter is used); `gc_on_save=True` runs `gc.collect()` after each save; `csv_fields=None` uses a dynamic schema, a tuple pins a fixed schema. Use instead of a per-trainer step-checkpoint class.
- `set_seeds(seed)` — canonical reproducible seeding (sets `PYTHONHASHSEED` + `random` + `numpy` + `keras.utils.set_random_seed`). Use instead of an inline RNG-seeding block.
- `save_config_json(config, results_dir, filename="config.json")` — dump a dataclass / object / dict config to JSON (dataclass-aware, numpy-safe). Returns the written path.
- `prepare_run_dir(config, output_dir=None)` — create `output_dir/experiment_name`, `mkdir(parents=True)`, and write `config.json` into it; returns the `Path`. Pass `output_dir=` when the trainer resolves the path itself (e.g. the SAM trainers' `resolved_output_dir(config)`). Use instead of the three-line preamble.
- `save_training_history_json(history, output_dir)` — dump `history.history` (or a raw dict) as `{metric: [floats]}`. **Best-effort**: warns and returns `None` on failure rather than raising, because it runs after the weights are already saved. Use instead of an inline `try/json.dump` block.
- `default_experiment_name(*parts)` — underscore-join the parts and append the run timestamp. Use in `__post_init__` instead of an inline `strftime("%Y%m%d_%H%M%S")`. Empty/`None` parts are dropped. **Careful**: if a prefix already ends in `_`, concatenate it with the next fragment yourself (`f"{prefix}{variant}"`) — passing them separately inserts a second underscore.
- `log_gpu_peak_memory()` — log peak/current memory for every visible GPU (reporting only; never raises).
- `setup_mixed_precision(enabled, policy="mixed_float16") -> bool` — set the global dtype policy (and explicitly reset to `float32` when disabled, since the policy is process-wide). Returns whether it was enabled. Wrap the optimizer in `LossScaleOptimizer` at the call site for `mixed_float16`; `mixed_bfloat16` needs no loss scaling.
- `mean_std`, `bootstrap_ci`, `paired_permutation_test`, `format_mean_std` (`train.common.stats`) — NaN-tolerant, degenerate-safe sweep statistics. Pass an explicit `rng=np.random.default_rng(SEED)`.
- `json_numpy_default` — pass as `json.dump(..., default=json_numpy_default)` to serialize numpy scalars / arrays (native numeric, not strings).
- `CIFAR10_MEAN`, `CIFAR10_STD` — CIFAR-10 per-channel mean/std for normalization. Distinct from the OpenAI-CLIP `IMAGE_MEAN`/`IMAGE_STD` in `image_text.py` — never conflate the two.

**Use from `dl_techniques` (library-level components):**
- `dl_techniques.metrics.depth_metrics` — AbsRelMetric, DeltaThresholdMetric, SqRelMetric, RMSEMetric, RMSELogMetric.
- `dl_techniques.callbacks.depth_visualization` — DepthPredictionGridCallback, DepthMetricsCurveCallback.
- `dl_techniques.optimization` — `optimizer_builder()`, `learning_rate_schedule_builder()`, `create_learning_rate_schedule()`, `create_warmup_lr_schedule()`, `WarmupSchedule`. Build optimizers through `optimizer_builder()` rather than calling `keras.optimizers.AdamW(...)` directly: it handles gradient clipping and weight-decay exclusions in the constructor, where Keras requires them. Never set `optimizer.clipnorm` after construction.
- `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint(target, ckpt_path, skip_prefixes, strict)` — layer-by-layer weight transfer from a saved `.keras` model. Use this (not `model.load_weights(by_name=True)` which is broken in Keras 3.8 for `.keras` files).

**Keep local to each script:**
- Model creation and compilation — architecture-specific.
- Domain-specific losses (e.g. `DepthEstimationLoss` with masked L1 + gradient matching).
- Custom argparse when `create_base_argument_parser()` doesn't fit (NLP, time-series, detection, depth).
- Training summary writing — model-specific fields.

## Data Loading

`load_dataset()` handles mnist, cifar10, cifar100, and imagenet. Returns:
- Numpy datasets: `(x_train, y_train), (x_test, y_test), input_shape, num_classes`
- ImageNet (tf.data): `train_ds, val_ds, input_shape, num_classes`

For NLP text datasets, use `train.common.nlp`: `load_text_dataset()`, `preprocess_mlm_dataset()`, `preprocess_classification_dataset()`.

For MegaDepth RGB+depth pairs, use `train.common.megadepth`: `discover_megadepth_pairs()`, `MegaDepthDataset()`. Produces `(rgb, y_true)` where `y_true = [depth, mask]` concatenated on the last axis.

For time-series, file-based images, or other non-standard data, write local data loading. Do NOT try to force it through `load_dataset()`.

## GPU Selection

Every script must support `--gpu`:

```python
setup_gpu(args.gpu)  # pass gpu_id from argparse
```

This sets `CUDA_VISIBLE_DEVICES` when a specific GPU is requested, or enables memory growth on all GPUs when `None`.

## Scripts That Don't Use `train.common` Callbacks (and Why)

These scripts have legitimate reasons for local callback management:

| Script | Reason |
|--------|--------|
| bert/wikipedia/* | MirroredStrategy distributed training, BackupAndRestore for fault tolerance |
| blt | Multi-stage pipeline (entropy pretraining + main training), class-based trainer |
| yolo12/train_multitask | Per-task callbacks, per-task loss tracking, per-task visualization |
| tabm | Custom TabMTrainer class, not standard Keras fit() |

**This table is about CALLBACKS and nothing else.** It is not a licence to skip the CLI. Every entry point still owes a `parse_arguments()`/`main(argv)` whose FIRST statement parses argv, so `--help` exits 0 with a `usage:` line and allocates nothing. `bert/wikipedia/*` had no parser at all, so `--help` ran `main()` for real (reaching `MirroredStrategy`, and for `finetune.py` a full TFDS dataset build) before crashing; all three are fixed. A repo-wide sweep of all `src/train/` entry points (125 under the filter given in "Consolidation candidates measured and REFUSED (2026-08-13)" below; this sentence used to say 115, a figure no filter reproduces) found exactly one further offender, `train.tabm.train_tabm`, whose failure mode was worse than a crash: it ignored `--help`, ran all five example pipelines to completion and exited **0** with no `usage:` line, so an exit-code-only sweep read it as healthy. Also fixed, and both packages now have their first tests (`tests/test_train/test_bert_wikipedia/`, `tests/test_train/test_tabm/`).

Two traps that sweep exposed, worth knowing before adding a parser:
- **`argparse` is not enough on its own** — and this one is now FIXED at the root, but the lesson is the point. `train.common`'s package `__init__` imports `image_text.py`, whose `IMAGE_MEAN`/`IMAGE_STD` used to be built with `tf.constant(...)` at MODULE scope. A module-level eager op initializes TF's eager context and creates a GPU device, so `from train.common import setup_gpu` at module scope made `--help` allocate a GPU no matter where you parsed — on all 125 entry points — and once produced a false 12-error test "regression" that was really `cudaSetDevice()` self-contention between concurrent suites. **The rule that survives the fix: never run an eager TF op at module scope anywhere under `train/common/`.** The package `__init__` re-exports it, so the cost is paid by every importer of every submodule, not just yours.
  Both constants are now plain Python lists, matching the sibling convention of `CIFAR10_MEAN`/`IMAGENET_MEAN` in `datasets.py`; the sole consumer, `augment_and_normalize`, needed no change, because TF promotes the RHS to the LHS's float32 at the subtraction site (`datasets.py`'s ImageNet normalization has always subtracted a list this way). Measured: `"Created device"` lines on `CUDA_VISIBLE_DEVICES=1 python -c "import train.common"` went **1 → 0**. Guarded by `tests/test_train/test_common_image_text.py`, which pairs the absence check with a positive liveness arm (a subprocess that deliberately DOES allocate) — an absence assertion with no liveness arm passes on a CPU-only box and proves nothing. **Behaviour change to know about:** device allocation now happens at first real TF use rather than at import, so the first line of a run's GPU log moves later.
  **That guard is GPU-CONDITIONAL — a green run is only evidence when it reports `0 skipped`.** `_subprocess_env` (`tests/test_train/test_common_image_text.py:54-63`) does `env.setdefault("CUDA_VISIBLE_DEVICES", "1")`, so it INHERITS whatever the parent shell exported; on this repo's layout GPU 0 is routinely full with the user's jobs, and a session run under `CUDA_VISIBLE_DEVICES=0` can leave the liveness subprocess unable to allocate. When that happens every test in the module **SKIPS rather than fails** (correct — a vacuous pass would be worse) and the absence assertion has proven nothing. Quote `passed` WITH `skipped`, never `passed` alone.
  **Two facts measured on 2026-08-13 by the iteration's adversarial reviewer, recorded because the plan itself never produced them:** (1) driving 4 synthetic JPEGs through the REAL file-path branch of `make_image_text_tf_dataset(paths, tokens, image_size=32, batch_size=2, training=False)` and A/B-ing the shipped plain-list constants against monkeypatched `tf.constant` ones gives **max|delta| = 0.0 exactly** (shapes `(2,32,32,3)` / `(2,8)`, all finite) — the eval path is genuinely inert end-to-end, not just at `augment_and_normalize`; (2) all **21** `train/common/*.py` submodules, imported individually under `CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=0`, emit **0** `"Created device"` lines — the fix is comprehensive, even though the shipped guard only parametrizes 3 module paths, so a future module-scope eager op in e.g. `tfrecord.py` would NOT be caught.
  **The honest limit: the `training=True` branch of that pipeline is not comparable by this method and remains unverified by anything.** The same A/B on the train split gave max|delta| **2.98** — but the null control (list-vs-list, same seed) gave **3.27**. Shuffle plus `random_crop` nondeterminism swamps any constants effect, so on that branch neither a pass nor a fail is readable. Do not cite a train-split A/B as evidence either way without first killing the nondeterminism.
  The four entry points that carry a local deferred-import workaround for this (`bert/wikipedia/{pretrain,pretrain_english,finetune}.py` import `setup_gpu` inside `main()`; `tabm/train_tabm.py` likewise) still work and are no longer NECESSARY. They are deliberately left in place — removing them is out of scope of the fix and belongs to its own change with its own `--help` gate. All four in-code `# DECISION plan-2026-08-12T201216-50fc0975/D-006` anchors were given a SUPERSEDED line on 2026-08-13 so the call-site comment no longer asserts a defect that has been fixed.
- **Exit 0 is not a passing `--help`.** A script with no parser ignores `--help` entirely and exits 0 after doing its whole job. Assert a `usage:` line, not just the exit code.

When writing a new script that genuinely can't use `create_callbacks()`, document the reason in a comment at the top of the callbacks section.

A different kind of non-adoption is documented under Pattern 3: `tree_transformer` *does* use `create_nlp_callbacks`, but is deliberately NOT folded into the shared bert/fnet finetune/pretrain scaffold — see "`tree_transformer` is a deliberate NON-ADOPTER of the bert/fnet scaffold" for the measured reasons, the drifts left unharmonized, and three known open defects.

## Consolidation candidates measured and REFUSED (2026-08-13)

Five `src/train/` consolidation candidates were measured on 2026-08-13. Two were acted on (the `image_text.py` import-time GPU allocation, and the seven inline RNG seeders — both documented above). The rest were **measured and refused**, and the measurement is recorded here so the next sweep reads it instead of re-deriving it.

**This section is a DIFFERENT axis from the "Scripts That Don't Use `train.common` Callbacks" table above.** That table is about adoption of the generic `create_callbacks()` scaffold; none of its four packages (`bert/wikipedia`, `blt`, `yolo12/train_multitask`, `tabm`) appears anywhere in the seven-class schedule family below. Do not merge the two lists.

Every path below was resolved with `test -e` and every count re-derived with the command shown, at the commit that added this section. Where a symbol is unambiguous, it is cited *by symbol* — line numbers in this file have drifted twice before.

### (a) The schedule-callback family — no shared `_interp` helper

**Seven** classes interpolate a scalar over training progress and assign it somewhere. Not twelve: the "12" figure is not reconstructible under any definition tried. The base count is whatever an AST walk over `ClassDef.bases` across `src/dl_techniques/` + `src/train/` returns — **68 classes in 51 files as of 2026-08-13**; re-derive it rather than quoting it, since it moves whenever any package is added or removed. There were **11** `.py` files under `src/dl_techniques/callbacks/` (10 plus `__init__.py`) on the same date. Use the AST walk, **not** a name grep: `grep -rn "^class .*Callback" src/dl_techniques/ src/train/ --include="*.py" | wc -l` happened to return 68 too on that date, but it is a **different** 68 — it counts at least one class whose name ends in `Callback` while having no base at all, and it misses indented definitions such as `BenchmarkCallback` at `src/train/ntm/harness.py:754`. The agreement was a coincidence of two offsetting errors, which is exactly why the grep must not be used as the measurement. (This paragraph used to say the grep gives 67; it did not, under any scope tried — corrected 2026-08-13.)

| Class | Path |
|---|---|
| `NoiseSigmaCurriculumCallback` | `src/dl_techniques/callbacks/noise_sigma_curriculum.py` |
| `TemperatureAnnealingCallback` | `src/dl_techniques/callbacks/temperature_annealing.py` |
| `TeacherEMACallback` | `src/dl_techniques/models/depth_anything/teacher_ema.py` |
| `KLWarmupCallback` | `src/train/vae/train_vae.py` |
| `DeepSupervisionWeightScheduler` | `src/train/resnet/train_resnet.py` |
| `WWPGDProjectionCallback` | `src/dl_techniques/optimization/ww_pgd_optimizer.py` |
| `PhaseScheduler` | `src/dl_techniques/models/memory_bank/phase_scheduler.py` — a **discrete phase lookup**, not an interpolator; listed for completeness |

Three of the seven live OUTSIDE `dl_techniques/callbacks/` (under `models/` and `optimization/`), which is why enumerating only that directory undercounts. `ScheduledDropout` (`src/dl_techniques/layers/scheduled_dropout.py`) is **a `Layer`, not a `Callback`** — it reuses `keras.optimizers.schedules.LearningRateSchedule` as its decay engine and belongs to neither list.

**The measured ceiling: ~18 executable lines.** Only two of the seven share identical math — `NoiseSigmaCurriculumCallback._interp` and `TemperatureAnnealingCallback._temperature_at`, both **11** AST-counted executable statements (docstring stripped), both the same `clamp(epoch)/total` fraction fed through linear / cosine / exp-in-log-space branches. Collapsing just those two into a shared helper leaves ~2 lines at each call site, so it saves **~18 executable lines net** — below the ~20-line threshold at which a consolidation is worth its indirection. The rest have nothing to give: `TeacherEMACallback` is already factored into 3-line standalone schedule functions, `KLWarmupCallback`'s math is 2 lines and linear-only, and `WWPGDProjectionCallback._compute_hardness` uses a structurally different two-stage warmup+ramp offset that a shared `(start, end, frac)` signature would change rather than refactor.

**Why the ceiling is so low: the bulk of each class is plumbing, not math.** `keras.Variable`-vs-plain-attribute resolution, per-class `get_config`/`from_config` contracts that all round-trip differently, fail-soft guards, and embedded `# DECISION` anchors recording load-bearing history. `noise_sigma_curriculum.py` is 159 lines around an 11-line kernel; `ww_pgd_optimizer.py` is 644 lines around an 8-line hardness ramp plus a ~400-line SVD/Cayley pipeline that has nothing to do with progress interpolation.

**The primitive already exists — point at it instead of building a second one.** `schedule_builder` in `src/dl_techniques/optimization/deep_supervision.py` (`schedule_builder(config, no_outputs, invert_order=False) -> Callable[[float], np.ndarray]`) is already a generalized progress→value factory with **10** registered shapes (`ScheduleType`: `constant_equal`, `constant_low_to_high`, `constant_high_to_low`, `linear_low_to_high`, `non_linear_low_to_high`, `custom_sigmoid_low_to_high`, `scale_by_scale_low_to_high`, `cosine_annealing`, `curriculum`, `step_wise`). `DeepSupervisionWeightScheduler` already consumes it. Its return contract is an `np.ndarray` of per-output weights, not a scalar, so it does not fit the scalar consumers without adaptation — which is the work any future consolidation would actually be signing up for.

### (b) `plot_training_history` — 3 definitions, deliberately not merged

`grep -rn "def plot_training_history" src/ --include="*.py"` → exactly 3, one caller each, each caller in the same file as its definition (no cross-file callers).

| Definition | Length | Shape |
|---|---|---|
| `capsnet/train_capsnet.py:215-283` | 69 lines | hand-rolled 2x2 Matplotlib grid, direct `plt.savefig` |
| `power_mlp/train_power_mlp.py:162-224` | 63 lines | hand-rolled 2x2 grid **plus** a `pandas.Series.rolling` moving-average stability panel |
| `vae/train_vae.py:240-257` | 18 lines | shim: aliases `total_loss`→`loss` / `val_total_loss`→`val_loss`, then delegates to the shared `generate_training_curves` (`train/common/evaluation.py`) |

A merge would have to **invent 3-4 toggles for code that does not exist as a toggle today**:
1. moving-average panel on/off — power_mlp only, and it drags in a `pandas` dependency the other two do not have;
2. a per-model conditional metric key — `capsule_accuracy` / `reconstruction_loss` / `margin_loss` vs plain `accuracy` vs none;
3. "Not Available" placeholder text — capsnet only (3 sites);
4. vae's metric-key aliasing, or else leaving vae routed through `generate_training_curves` and not merging it at all, which reduces "3 definitions" to "2 to merge, 1 legitimately separate".

That is the same anti-pattern this document already forbids in the tree_transformer non-adoption section under Pattern 3 ("Folding it in would therefore mean inventing config toggles for code that **does not exist**"). These are three diagnostic contracts, not three styles of one function.

**The narrower win that IS available**, if someone wants one: capsnet and power_mlp end with byte-identical `plt.tight_layout()` / `plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')` / `plt.close()` boilerplate. That is safely mergeable without touching any diagnostic content.

### (c) Corrected counts, each with the command that MEASURES it

| Claim | Command | Value |
|---|---|---|
| Schedule interpolators | classification of the 68 `*Callback`-based classes above — see (a) | **7**, not 12 |
| Canonical `set_seeds` users | `grep -rl "set_seeds(" src/train/ \| grep -v "common/seed.py\|common/__init__.py\|README.md\|CLAUDE.md" \| wc -l` | **72** (was **65** before the 7 inline seeders were migrated on 2026-08-13; never 70) |
| `src/train/` entry points | `grep -rl '__name__ == .__main__.' src/train/ --include="*.py" \| grep -v "/common/\|/__init__\.py$\|/test_"` | **125**. The often-quoted **115 is not reproducible under any filter tried** — the raw guard grep gives 127, and 125 after excluding `common/`, `__init__.py` and `test_*`. State your filter whenever you quote this. |
| Direct `keras.optimizers.*` calls | `grep -rn -E "keras\.optimizers\.(Adam\|AdamW\|SGD\|RMSprop)\(" --include="*.py" src/train/` | **63 across 49 files** — CONFIRMED exactly. The `--include="*.py"` filter is **required**: without it you get 66, and the three extra hits are prose mentions inside this very file. |

### (d) The optimizer ruling — `optimizer_builder()` adoption REFUSED for the AdamW-pretrain cluster

Adoption of `optimizer_builder()` for the 9-file AdamW-pretrain cluster was measured against a **rule pre-registered before the probe ran**: ADOPT only if (i) all 9 sites are config-equivalent under a comparator that also checks post-construction weight-decay exclusions, AND (ii) the AST-stripped executable-statement delta over the 9 files is **≤ −9**.

**Condition (i) is what decided this, and it decided it on CORRECTNESS, not on line count.** Two independent failures, either one sufficient:

1. `optimizer_builder` **renames** the clipping keys (see the first trap below), so the naive migration — copying `clipnorm=1.0` straight into the config dict — **silently drops gradient clipping at all 9 sites**, with no error and no warning. Measured directly, not reasoned about: the control writing the literal key fired the comparator's config check with `config[clipnorm]: actual=1.0 builder=None`.
2. **2 of the 9 sites did not construct at all** on the as-written side (`bert/wikipedia/{pretrain,pretrain_english}.py` — `jit_compile=True`, see the bug note below; FIXED 2026-08-13, after this measurement was taken), so the comparator's `CHK-CONSTRUCT` check reported MATERIAL there and no equivalence claim was even available.

**The +7 delta CORROBORATES the ruling; it did not decide it — and it could not have.** Measured delta: **+7**. Composition: **+0 body** — every site is exactly one statement before and one statement after (5 are an `optimizer = …` assignment; 4 are an inline `optimizer=` kwarg inside an existing `model.compile(…)` call, which is 0 statements of its own on both sides) — and **+7 imports** (7 of 9 files need a new `from dl_techniques.optimization import optimizer_builder`; the 2 `bert/wikipedia` files already have such a line to absorb the name). Note honestly that condition (ii)'s `≤ −9` threshold was **structurally incapable of returning ADOPT** for a one-statement-to-one-statement replacement: with a +0 body delta by construction, the only movement available is the import line, which is non-negative. Read (ii) as a falsified intuition rather than a discriminator — it refutes the "replaces N lines with one call" framing, since raw physical lines move **+3** (58 → 61) and the apparent multi-line saving is line-wrapping. **Ruling: DOCUMENT.** No `src/` edit was made.

Three sites are excluded on their own merits, independent of the arithmetic:

- `coshnet/train_coshnet.py:99` — `epsilon=1e-8`, a non-default. `optimizer_builder` defaults to `1e-7` and would change it **silently**.
- `thera/train_thera.py:451` — `global_clipnorm=None` set **deliberately**, because clipping happens manually inside a custom `train_step` (D-012). Reflexively filling clipping in would double-clip.
- `vae/train_vae.py:336` — a `keras.optimizers.get()` fallback for arbitrary optimizer names. `optimizer_builder` raises on an unknown `type`, so migrating this site would REMOVE functionality, not refactor it.

Separately, **16 bare one-liners** are excluded as a class: 15 sites pass only `learning_rate=` (AST-verified) plus `convunext/train_convunext.py:229`, which passes the LR positionally. Routing any of these through the builder replaces one line with a dict-literal plus a call — it is net line-**positive**. "Consolidate all 63 optimizer sites" is not a supportable framing.

> **Scope note for a future sweep:** the 3-kwarg `AdamW(learning_rate=…, weight_decay=…, clipnorm=…)` signature actually occurs at **17** sites, not 9 (AST scan over `keras.optimizers.AdamW` calls whose kwargs are a superset of those three). The nine measured above are the NLP-pretrain subset. Do not assume 9 bounds the surface.

**Two traps the probe surfaced — these are more valuable than the ruling itself.**

- **`optimizer_builder` silently ignores a literal `"clipnorm"` config key.** Its keys are RENAMED: `gradient_clipping_by_value` → `clipvalue`, `gradient_clipping_by_norm_local` → `clipnorm`, `gradient_clipping_by_norm` → `global_clipnorm` (`src/dl_techniques/optimization/optimizer.py:140-152`). All 9 cluster sites pass `clipnorm=1.0`. A naive migration that copies that key straight into the config dict **drops gradient clipping on all 9 with no error and no warning**. This is a live trap for anyone adopting the builder today, not a historical note. Measured directly: a control writing the literal key fired the comparator's config check with `config[clipnorm]: actual=1.0 builder=None`.
- **`optimizer_builder` hard-codes `"name": "AdamW"`** (`src/dl_techniques/optimization/optimizer.py:305`) against Keras' own `"adamw"` default, so every AdamW built through it differs from a directly-constructed one on the `name` key. Cosmetic for numerics, but the name is the optimizer's **variable scope** — a checkpoint-compatibility consideration that already applies to the **23** files under `src/train/` that use `build_optimizer` today (`grep -rl "build_optimizer" src/train/ --include="*.py" | wc -l`). The `--include="*.py"` is **required**, for the same reason as in the optimizer-call row of table (c): without it the answer is 24, because this very paragraph is the 24th hit.

**A REAL BUG — found by the consolidation sweep, FIXED 2026-08-13 in its own change:**

`src/train/bert/wikipedia/pretrain.py` and `pretrain_english.py` passed **`jit_compile=True` to a Keras 3 optimizer CONSTRUCTOR**. Keras 3 optimizers have no such parameter; the call raises `ValueError: Argument(s) not recognized: {'jit_compile': True}`. It sat on an unconditional path inside `with strategy.scope():` immediately before `model.compile(...)`, so **both scripts crashed the moment they reached model compilation, on every run, for their whole existence**. Invisible to the `--help` gate (a train-time raise, and all 125 entry points exit 0) and invisible to the test suite, which never constructed the optimizer.

**The obvious repair was WRONG, and this doc previously prescribed it.** An earlier revision of this section said "the correct repair routes XLA to `model.compile(jit_compile=True)`". Measured 2026-08-13 (TF 2.18 / Keras 3.8, one GPU, tiny BERT + the real `MaskedLanguageModel`), that repair merely relocates the crash from compile time to step 1:

| Arm | Result |
|---|---|
| `AdamW(..., jit_compile=True)` — as written | `ValueError: Argument(s) not recognized` |
| `compile(jit_compile=True)`, fp16, **no strategy** | trains, loss 5.5958 |
| `compile(jit_compile=True)`, fp16, **MirroredStrategy** | **`InvalidArgumentError`** — see below |
| `compile(jit_compile=True)`, float32, **MirroredStrategy** | **`InvalidArgumentError`** — same op |
| no `jit_compile`, fp16, MirroredStrategy | trains, loss 5.6153 — **the shipped configuration** |

Two independent, structural blockers under a distribution strategy:

1. `MaskedLanguageModel.train_step` calls `optimizer.apply_gradients`, which under any strategy emits **`CollectiveGatherV2`** — an op with **no XLA_GPU_JIT kernel**, so tf2xla conversion fails hard. Reproduced in *both* float32 and mixed_float16, so it is architectural, not a precision issue.
2. Under `mixed_float16` only, `LossScaleOptimizer`'s finite-gradient `Cond` additionally trips `merge_call called while defining a new graph or a tf.function`.

A **single-replica** `MirroredStrategy` fails identically, and both scripts build `MirroredStrategy()` unconditionally (falling back only if construction throws). So on any GPU host the XLA path is **unreachable by construction**. The fix therefore *removes* the XLA request and records why; a `--jit-compile` flag was considered and rejected because it could only ever refuse on the hardware these scripts target — the dead-knob class `tests/test_train/test_config_fields_are_live.py` exists to prevent. Enabling XLA here requires removing the collective from the training step, not re-adding a keyword.

Guarded by `tests/test_train/test_bert_wikipedia/test_optimizer_construction.py` (8 tests). The durable one is an **AST sweep over all of `src/train/`** asserting that no `keras.optimizers.*` constructor receives a `compile()`-only kwarg (`jit_compile`, `run_eagerly`, `steps_per_execution`) — it catches the defect *class* anywhere in the tree, not just the two sites that had it. It ships with a liveness arm that plants the defect in a temp file and requires the detector to flag it, plus a negative arm proving `jit_compile=` on `model.compile(...)` is *not* flagged. Two further tests execute construct → compile → `fit` under `mixed_float16`, and one pins the XLA-under-strategy failure itself so that if a future TF makes it work, the test fails and this note gets revisited rather than silently trusted. RED-proven by re-injecting the original line: `ASSERT-NO-COMPILE-KWARG-ON-OPTIMIZER` and `ASSERT-NO-LIVE-JIT-COMPILE-IN-pretrain.py` both fired, while the DECISION comment quoting the defective line was correctly *not* flagged.

### (e) The 3 numpy/`random`-only seeders are deliberately NOT migrated

- `src/train/tabm/train_tabm.py:518` — `np.random.seed(42)`
- `src/train/nam/eval_dfsa.py:18-19` — `random.seed(42)` + `np.random.seed(42)`
- `src/train/nam/train_dfsa_ste.py:267` — `random.seed(42)`

Routing these through `set_seeds` would **ADD** TF and Keras seeding they do not have today, changing the model-init RNG stream. That is a behaviour change, not a refactor, and belongs to a plan that can measure it. The contrast is the whole point: the 7 files that WERE migrated on 2026-08-13 were provably bit-identical before and after (`random` / NumPy / TF / Keras draws all matched, comparator RED-proven with a different-seed control), which is exactly why they were safe and these three are not.

### (f) Known test-coverage gaps — recorded, not fixed

- **`DeepSupervisionWeightScheduler`** (`src/train/resnet/train_resnet.py:134`) has **zero** test coverage: `grep -rn "DeepSupervisionWeightScheduler" tests/` → 0 hits. It is the only one of the seven schedule callbacks in (a) that is untested; any consolidation touching it is unguarded and should add a test as a precondition, not a side-effect.
- **Of the 7 files whose seeding was migrated on 2026-08-13, only `tree_transformer/pretrain.py` was exercised by any test *at the time of that migration*.** No test referenced `logic/train_e4_monks.py`, `logic/train_e4_lowdata_mux.py`, `wave_field/train_memory.py`, `wave_field/pretrain.py`, `vae/evaluate_samplers.py` or `cliffordnet/train_cliffordnet_nlp.py`. **Two of those seven no longer read that way, both changed later on 2026-08-13:** `wave_field/pretrain.py` now IS covered (`tests/test_train/test_wave_field/test_run_artifacts.py`, 5 tests, which execute the trainer body through a stub harness), and `wave_field/train_memory.py` was DELETED on user instruction (§ (g)), taking its 5 companion tests with it. `tests/test_train/test_vae/` still does not exist. `tests/test_train/test_logic/` and `tests/test_train/test_cliffordnet/` DO exist but cover other modules — `test_logic/` covers `train_benchmark`, `train_boolean_circuit`, `train_e1_image` and `train_e3_faithfulness` but neither `e4` script, and `test_cliffordnet/` contains only `test_train_clip_textlm.py`. A directory bearing a package's name is not coverage of that package.

### (g) The CLM `TrainingConfig` ruling — base class ADOPTED for two trainers, REFUSED for `train_memory.py` (which was then DELETED)

> **Status update, later the same day (2026-08-13): `src/train/wave_field/train_memory.py` was DELETED on explicit user instruction** ("delete train_memory, i dont care about it"). The ADOPT half below is unaffected and still describes the shipped code; the REFUSE half is now history. It is kept in full rather than deleted because the measurement is what produced the general rule — *a config that LACKS fields the shared base declares is a feature fork, not a subclass* — and that rule is live wherever the next candidate turns up. **The model package it trained, `src/dl_techniques/models/memory_bank/`, was NOT deleted** and keeps its 83 tests (`tests/test_models/test_memory_bank/`); it simply ships with no trainer now. `src/train/wave_field/` is a one-trainer package (`pretrain.py`), and the five `TestTrainMemoryLlmWritesRunArtifacts` tests went with the file.

Measured 2026-08-13 against a rule **pre-registered before the probe ran**: ADOPT a shared `ClmPretrainConfig` base only if (i) for every affected config class the `{field name → default, default_factory}` map is EQUAL before and after, AND (ii) the AST-stripped executable-statement delta over the touched files is net-negative. Conjunct (i) is the one doing the work — a silently-changed default changes real runs with no error and no test in this tree would see it.

**ADOPTED for `train/gpt2/pretrain.py` + `train/wave_field/pretrain.py`.** `ClmPretrainConfig` now lives in `src/train/common/clm_pretrain.py` (43 fields, re-exported from `train/common/__init__.py`) and both subclass it, overriding only `save_dir` — plus `field_size` for wave_field. Before the extraction the two classes were measured to diverge on **exactly one** default (`save_dir`), zero `default_factory` values, zero type annotations and one additive field. Rule (i): EQUAL for all four classes checked (`gpt2:TrainingConfig` 43 fields, `gpt2:SOTrainingConfig` 48, `wave_field:TrainingConfig` 44, `train_memory:TrainingConfig` 49), default-constructed instance dicts included. Rule (ii): **−39** executable statements (450 → 411 over five files).

**REFUSED for `train/wave_field/train_memory.py`, and rule (i) made the refusal mechanical rather than a judgement call.** (Past tense from here: the file was deleted hours later — see the status note at the top of this section. The command below is no longer runnable at HEAD; it was measured at commit `9f3208319`, the last commit containing the file.)

```
grep -n "num_layers\|num_heads" src/train/wave_field/train_memory.py   # → 0 hits (at 9f3208319)
```

The base carries `num_layers` and `num_heads`; that config did not have them and nothing in the file read them. Inheriting would have **ADDED two fields no code reads and no CLI flag sets** — the dead-knob class `tests/test_train/test_config_fields_are_live.py` exists to catch, and the "invent config toggles for code that does not exist" anti-pattern sections (b) and the tree_transformer non-adoption already refuse. It also overrode `learning_rate` 3e-4 → 1e-5 and added eight fields of its own (`field_size`, `top_k`, `memory_lr`, `phase1_steps`, `phase2_steps`, `phase3_steps`, `warmup_num_batches`, `init_from`). It was the feature fork §2 of that plan's findings already classed it as, and the config measurement agreed independently. **The durable rule: do not re-litigate a fork-vs-subclass refusal by "just adding the missing fields".**

**One accepted contract change, recorded so it is not mistaken for drift:** dataclass inheritance appends subclass-only fields AFTER inherited ones, so `wave_field`'s `field_size` moved from declaration position 6 to last. Every construction of that class in the repo is by KEYWORD (`_config_from_args`, and `SOTrainingConfig(**vars(base), ...)` for the gpt2 sibling), so no caller is order-sensitive. Restoring the position by putting `field_size` in the shared base would plant a dead WaveField-only field in every GPT-2 config — see the `# DECISION` anchors on `ClmPretrainConfig` and on `train/wave_field/pretrain.py`'s `TrainingConfig`.

**Gates held across both changes:** all five trainers' `--help` stdout stayed **byte-identical** (`COLUMNS=80`, stdout only — `2>&1` is not byte-stable): `train.gpt2.pretrain` 3796, `train.gpt2.pretrain_so` 4743, `train.gpt2.finetune` 2105, `train.wave_field.pretrain` 3333, `train.wave_field.train_memory` 3556, all exit 0. **Four of those five baselines are still live and still hold; the fifth (3556) retired with the deleted file** — re-verify against the four, not the five. That gate was proven non-vacuous before it was trusted: a one-character help-string edit moved 3796 → 3797. `pyflakes src/train/` stayed 248 and `pyflakes src/` 578.

The same change renamed `gpt2_variant` → `model_variant` in `train/gpt2/pretrain.py` (7 sites) and `pretrain_so.py` (1), which is what unblocked the shared base. `model_variant` is the repo-majority spelling (`vit`, `resnet`, `wave_field`, `rms_variants_train`). No `--gpt2-variant` CLI flag ever existed and the old field name is now gone from the tree — `git grep -- "gpt2-variant" -- "*.py"` and `git grep -- "gpt2_variant" -- "*.py"` both → **0 hits**. The `-- "*.py"` pathspec is **required**, for the same reason as in table (c): without it each grep returns 1, and that hit is this very paragraph. None of the five trainers uses argparse `dest=` either, which is why a config-field rename could not move `--help` output.

### (h) The MODEL-side refusals for the `gpt2` / `wave_field` cluster (2026-08-13)

Section (g) above rules on the *trainer* configs. This subsection records the **model-package** refusals measured by the same plan, so the next factory-adoption sweep over `src/dl_techniques/models/{gpt2,wave_field}/` reads them instead of re-deriving them. Each refusal below cost a measurement; none of them is a style opinion. The model-side copies live in `src/dl_techniques/models/gpt2/README.md` § 15 and `src/dl_techniques/models/wave_field/README.md` § 15.

**1. The FFN swap in `WaveFieldDecoderBlock` — REFUSED on dropout placement.** `create_ffn_layer('mlp')` builds an `MLPBlock` whose `call` applies dropout **between** `fc1`+activation and `fc2` (`src/dl_techniques/layers/ffn/mlp.py:296-307`), while the block applies it **after** the output projection, on the block output before the residual add. Measured at `dropout_rate=0.1` under an **identical Bernoulli mask** on unit-variance activations: **max |Δ| 0.3953, ≈38% of signal RMS**. That is a behaviour change at any non-zero dropout, not a refactor. A `training=False` numeric probe is BLIND to it — the semantic read of both `call()` bodies is the deciding instrument, not a delta. (`MLPBlock` also names its sub-layers `fc1`/`fc2`, not `ffn_dense_1`/`ffn_dense_2`, so a `by_name` load would silently fail to bind.) An earlier audit reached the same verdict independently (`research/2026_models_layer_reuse_audit.md`, cleanup C3).

**2. The four `LayerNormalization` sites — REFUSED because the factory returns the identical class.** `create_normalization_layer('layer_norm', ...)` resolves to `keras.layers.LayerNormalization` itself (`src/dl_techniques/layers/norms/factory.py`). Routing `attn_norm`, `ffn_norm`, `embed_norm` and `final_norm` through it constructs the same object, achieves **zero line reduction** (the explicit `epsilon` override is still required, since the factory's own default differs from the sites' `1e-5`), and fixes no defect. "Uses the factory" is not the goal; reaching a *different, better* implementation is, and here there is none.

**3. `build_optimizer` / `optimizer_builder` — REFUSED, already ruled in § (d).** Do not re-derive it for these five trainers. The deciding fact is the field-name contract mismatch: the builder RENAMES the clipping keys (`gradient_clipping_by_norm_local` → `clipnorm`), so copying these trainers' literal `clipnorm=1.0` into a config dict **silently drops gradient clipping** with no error and no warning. All five spell their fields `learning_rate` / `warmup_ratio`, not the builder's `lr_schedule_type` / `warmup_epochs`. `prepare_run_dir` is refused separately and structurally: it derives the run dir as `Path(config.output_dir) / config.experiment_name`, while all of these trainers receive their run dir FROM `create_nlp_callbacks`, which owns the `results/<prefix>_<timestamp>/` naming.

**4. `models/gpt2/` has ZERO adoptable factory sites — and the obvious grep is a GHOST.** `gpt2.py` builds exactly two things: a `TextDecoder` (which is itself already composed of `TransformerLayer`, `create_normalization_layer`, and `dl_techniques.utils.masking`'s mask builders) and, only when `tie_word_embeddings=False`, a bare `Dense(vocab_size, use_bias=False)` LM head. No factory covers a vocabulary projection — it is not an attention / FFN / normalization / embedding / activation primitive. **`grep -rn "create_.*_layer" src/dl_techniques/models/gpt2/ --include="*.py"` returns 0 hits, and that measures the wrong thing**: the model reaches the factories correctly THROUGH `TextDecoder`, one level down, which is the right depth of reuse per `src/dl_techniques/models/CLAUDE.md` § Layer Reuse Policy. Do not "fix" a zero here.

**What WAS adopted on the model side**, for contrast, so the pattern is legible: `wave_field`'s positional path now goes through `create_embedding_layer('positional_learned', ..., dropout_rate=0.0)` with `embed_dropout` kept as a separate post-`embed_norm` layer — the same split (and the same `dropout_rate=0.0` idiom, with the same reason) that `src/dl_techniques/layers/transformers/text_decoder.py` already uses. Equivalence was MEASURED at fixed transplanted weights: max abs delta **exactly 0.0** on both outputs, CPU, `training=False`.

**A grep-scoping trap, for the third time in this file** (see table (c) and the closing note of § (g)): any `src/train/`-wide grep quoted in prose must carry `--include="*.py"` or a `-- "*.py"` pathspec, or this document's own text becomes the extra hit. Item 4's grep hit it live: the moment `src/dl_techniques/models/gpt2/README.md` was written, an unscoped `grep -rn "create_.*_layer" src/dl_techniques/models/gpt2/` returned **3** hits — all of them prose in the new README explaining that the answer is zero. With `--include="*.py"` it is 0, as claimed.
