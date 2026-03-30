# Training Scripts

Production-grade training pipelines for models in `dl_techniques/models/`. Each subdirectory corresponds to a model architecture.

## Structure

```
src/train/
├── common/              # Shared utilities (GPU, datasets, callbacks, evaluation)
│   ├── gpu.py           # setup_gpu()
│   ├── datasets.py      # load_dataset(), load_imagenet_dataset(), get_class_names()
│   ├── callbacks.py     # create_callbacks(), create_learning_rate_schedule()
│   ├── evaluation.py    # validate_model_loading(), run_model_analysis(), visualizations
│   └── __init__.py      # Re-exports all public functions
├── convnext/            # ConvNeXt V1, V2, V2+MAE
├── power_mlp/           # PowerMLP
├── mobilenet/           # MobileNet variants
├── ...
└── CLAUDE.md
```

## Writing a Vision Training Script

### 1. File Naming

Name scripts `train_<model>.py`, **not** `train.py`. Files named `train.py` shadow the `train` package and break `from train.common import ...`.

### 2. Imports from `train.common`

Always use the shared utilities instead of writing local versions:

```python
from train.common import (
    setup_gpu,
    load_dataset,
    get_class_names,
    create_callbacks,
    create_learning_rate_schedule,
    validate_model_loading,
    convert_keras_history_to_training_history,
    create_classification_results,
    generate_comprehensive_visualizations,
    run_model_analysis,
)
```

### 3. Script Structure

A standard vision training script follows this pattern:

```python
# 1. Imports: model-specific + train.common
from dl_techniques.models.<model> import MyModel, create_my_model
from train.common import setup_gpu, load_dataset, get_class_names, ...

# 2. Model-specific config (keep local — likely to diverge per model)
def create_model_config(dataset, variant, ...) -> Dict[str, Any]: ...

# 3. Optional: model-specific visualization setup (if using custom PlotConfig)
def setup_visualization_manager(...) -> VisualizationManager: ...

# 4. Main training function
def train_model(args):
    setup_gpu()

    # Load data via common loader
    train_data, test_data, input_shape, num_classes = load_dataset(args.dataset)
    class_names = get_class_names(args.dataset, num_classes)

    # Create model (model-specific)
    model = create_my_model(variant=args.variant, input_shape=input_shape, ...)

    # LR schedule via common
    lr = create_learning_rate_schedule(args.learning_rate, args.lr_schedule, args.epochs)

    # Compile
    model.compile(optimizer=..., loss=..., metrics=[...])

    # Callbacks via common (includes EpochAnalyzerCallback)
    callbacks, results_dir = create_callbacks(
        model_name=f"{args.dataset}_{args.variant}",
        results_dir_prefix="my_model",    # directory prefix in results/
        patience=args.patience,
        use_lr_schedule=(args.lr_schedule != 'constant'),
    )

    # Train
    history = model.fit(...)

    # Validate serialization via common
    custom_objects = {"MyModel": MyModel, ...}
    validate_model_loading(model_path, test_sample, expected_output, custom_objects)

    # Post-training analysis via common
    run_model_analysis(model, test_data, history, model_name, results_dir)

    # Save summary

# 5. argparse main()
def main():
    parser = argparse.ArgumentParser(...)
    ...
```

### 4. What Lives in `train.common` vs. Locally

**Use from `train.common`:**
- `setup_gpu()` — GPU memory growth configuration
- `load_dataset(name, batch_size, image_size)` — loads mnist, cifar10, cifar100, imagenet
- `get_class_names(dataset, num_classes)` — human-readable class labels
- `create_callbacks(model_name, results_dir_prefix, ...)` — EarlyStopping, ModelCheckpoint, CSVLogger, EpochAnalyzerCallback, optional ReduceLROnPlateau
- `create_learning_rate_schedule(lr, type, epochs, steps_per_epoch)` — cosine, exponential, constant
- `validate_model_loading(path, sample, expected, custom_objects)` — round-trip serialization check
- `convert_keras_history_to_training_history(history)` — converts to visualization TrainingHistory
- `create_classification_results(y_true, y_pred, y_prob, class_names, model_name)` — for visualization
- `generate_comprehensive_visualizations(viz_manager, ...)` — training curves, confusion matrix, ROC/PR, architecture
- `run_model_analysis(model, test_data, history, model_name, results_dir, config)` — full ModelAnalyzer pipeline (weights, spectral, calibration, training dynamics)

**Keep local to each script:**
- `create_model_config()` — model-specific hyperparameter defaults per dataset/variant
- Model creation and compilation — architecture-specific
- Custom callbacks (e.g., NaNStoppingCallback for unstable models)
- Custom data preparation if the model needs non-standard input (e.g., flattening for MLPs, MAE masking)
- Custom visualization manager setup (if using custom PlotConfig/templates)
- Training summary writing — model-specific fields

### 5. Data Loading

`load_dataset()` handles mnist, cifar10, cifar100, and imagenet. It returns:
- For numpy datasets: `(x_train, y_train), (x_test, y_test), input_shape, num_classes`
- For ImageNet (tf.data): `train_ds, val_ds, input_shape, num_classes`

If your model needs non-standard preprocessing (e.g., flattening for MLPs, custom augmentation), call `load_dataset()` first then transform locally:

```python
(x_train, y_train), (x_test, y_test), input_shape, num_classes = load_dataset(args.dataset)
# MLP-specific: flatten
x_train = x_train.reshape(x_train.shape[0], -1)
```

### 6. Callbacks and Analysis

`create_callbacks()` returns a list that includes `EpochAnalyzerCallback` by default — this runs weight and spectral analysis at each epoch end. The `results_dir_prefix` parameter controls the output directory naming: `results/{prefix}_{model_name}_{timestamp}/`.

`run_model_analysis()` runs the full `ModelAnalyzer` pipeline post-training. It handles both numpy arrays and `tf.data.Dataset` inputs, saves plots and JSON results to `{results_dir}/model_analysis/`.

### 7. Model Serialization Validation

Always validate model save/load round-trips before evaluation:

```python
custom_objects = {"MyModel": MyModel, "MyBlock": MyBlock}
validate_model_loading(model_path, test_sample, expected_output, custom_objects)
```

Pass `custom_objects` explicitly — the common function doesn't know about model-specific classes.
