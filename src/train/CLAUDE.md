# Training Scripts

Production-grade training pipelines for models in `dl_techniques/models/`. Each subdirectory corresponds to a model architecture.

## Structure

```
src/train/
├── common/              # Shared utilities (GPU, datasets, callbacks, evaluation)
│   ├── gpu.py           # setup_gpu(gpu_id)
│   ├── args.py          # create_base_argument_parser()
│   ├── datasets.py      # load_dataset(), load_imagenet_dataset(), get_class_names()
│   ├── callbacks.py     # create_callbacks(), create_learning_rate_schedule()
│   ├── evaluation.py    # validate_model_loading(), run_model_analysis(), visualizations
│   └── __init__.py      # Re-exports all public functions
├── convnext/            # ConvNeXt V1, V2, V2+MAE
├── nbeats/              # N-BEATS time-series
├── bert/                # BERT pretrain/finetune
├── ...
└── CLAUDE.md
```

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

These scripts use their own tokenized datasets (not `load_dataset()`), need TensorBoard, and have custom EpochAnalyzerCallback settings. They wrap `create_callbacks()` in a local function that adds domain-specific behavior.

```python
from train.common import setup_gpu, create_callbacks as create_common_callbacks

def create_callbacks(config):
    """Wrap common callbacks with NLP-specific settings."""
    callbacks, results_dir = create_common_callbacks(
        model_name=config.model_name,
        results_dir_prefix="bert",
        monitor="val_loss",                  # NLP pretrain → val_loss
        patience=config.patience,
        use_lr_schedule=True,
        include_tensorboard=True,            # NLP scripts use TensorBoard
        analyzer_start_epoch=config.analysis_start_epoch,
        analyzer_epoch_frequency=config.analysis_epoch_frequency,
    )
    return callbacks, results_dir

def main():
    # Local argparse — NLP-specific args (vocab, tokenizer, etc.)
    parser = argparse.ArgumentParser(description="Pretrain BERT")
    parser.add_argument("--gpu", type=int, default=None)
    # ...
    args = parser.parse_args()
    setup_gpu(args.gpu)
```

---

### Pattern 4: Denoising / Detection (BFCNN, BFUNet, YOLO12, ResNet-ImageNet)

**Used by:** BFCNN, BFUNet, YOLO12-COCO, ResNet, DarkIR

These scripts use file-based datasets (not `load_dataset()`), monitor `val_loss` or domain metrics (`val_psnr`), and have domain-specific callbacks (visualization, deep supervision scheduling). They wrap `create_callbacks()` and append domain callbacks.

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
- `create_learning_rate_schedule(lr, type, epochs, steps_per_epoch)` — cosine, exponential, constant.
- `load_dataset(name, batch_size, image_size)` — MNIST, CIFAR-10/100, ImageNet only.
- `get_class_names(dataset, num_classes)` — human-readable labels.
- `validate_model_loading(path, sample, expected, custom_objects)` — round-trip serialization check.
- `run_model_analysis(model, test_data, history, name, results_dir, config)` — full ModelAnalyzer pipeline.

**Keep local to each script:**
- Model creation and compilation — architecture-specific.
- Domain-specific callbacks (visualization, deep supervision scheduling, performance monitoring).
- Custom data loading/generation (NLP tokenization, time-series generators, file-based image loading).
- Custom argparse when `create_base_argument_parser()` doesn't fit (NLP, time-series, detection).
- Training summary writing — model-specific fields.

## Data Loading

`load_dataset()` handles mnist, cifar10, cifar100, and imagenet. Returns:
- Numpy datasets: `(x_train, y_train), (x_test, y_test), input_shape, num_classes`
- ImageNet (tf.data): `train_ds, val_ds, input_shape, num_classes`

For non-standard data (NLP, time-series, file-based images), write local data loading. Do NOT try to force it through `load_dataset()`.

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
| ntm/train_multitask | 6 algorithmic task generators with task-specific evaluation |
| bfunet/train_conditional* | TrainingPipeline inheritance with `_create_callbacks()` override |

When writing a new script that genuinely can't use `create_callbacks()`, document the reason in a comment at the top of the callbacks section.
