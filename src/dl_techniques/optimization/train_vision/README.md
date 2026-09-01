# `train_vision` — vision training pipeline

An end-to-end Keras 3 image-classification pipeline: config dataclass → LR schedule and
optimizer built from `dl_techniques.optimization` → `model.fit` with checkpointing, TensorBoard,
CSV logging and periodic plots → post-training classification visualizations and a full
`ModelAnalyzer` run.

You supply two things: a `DatasetBuilder` subclass and a `model_builder` function. The pipeline
owns everything else.

**Classification only.** The loss is hard-coded to
`keras.losses.SparseCategoricalCrossentropy(from_logits=config.from_logits)` and the metrics to
sparse accuracy + sparse top-5 accuracy. For anything else, build the optimizer yourself with
`optimizer_builder` and write your own loop.

## Quick start

```python
import keras
import tensorflow as tf
from dl_techniques.optimization.train_vision import (
    TrainingConfig, TrainingPipeline, DatasetBuilder,
)

class Cifar10Builder(DatasetBuilder):
    def build(self):
        (xt, yt), (xv, yv) = keras.datasets.cifar10.load_data()
        xt, xv = xt.astype("float32") / 255.0, xv.astype("float32") / 255.0
        bs = self.config.batch_size
        train = tf.data.Dataset.from_tensor_slices((xt, yt)).shuffle(10_000).batch(bs).repeat()
        val = tf.data.Dataset.from_tensor_slices((xv, yv)).batch(bs).repeat()
        return train, val, len(xt) // bs, len(xv) // bs

def build_model(config: TrainingConfig) -> keras.Model:
    return keras.Sequential([
        keras.layers.Input(shape=config.input_shape),
        keras.layers.Conv2D(32, 3, activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(config.num_classes, activation="softmax"),
    ])

config = TrainingConfig(
    input_shape=(32, 32, 3),
    num_classes=10,
    epochs=50,
    batch_size=128,
    learning_rate=1e-3,
    optimizer_type="adamw",
    lr_schedule_type="cosine_decay",
    warmup_steps=1000,
    output_dir="results",
    experiment_name="cifar10_demo",
)

pipeline = TrainingPipeline(config)
model, history = pipeline.run(build_model, Cifar10Builder(config))
```

## API

| Name | What it is |
|---|---|
| `TrainingConfig` | dataclass holding every knob; `.save(path)` / `.load(path)` as JSON |
| `TrainingPipeline(config)` | `.run(model_builder, dataset_builder, custom_callbacks=None) -> (model, History)` |
| `DatasetBuilder` | ABC; implement `build()`, optionally `get_test_data()` and `get_class_names()` |
| `ModelBuilder` | type alias for `Callable[[TrainingConfig], keras.Model]` — not a class |
| `DataInput` | re-export of `dl_techniques.analyzer.DataInput`; what `get_test_data()` returns |
| `create_argument_parser()` | `argparse.ArgumentParser` for the flags below |
| `config_from_args(args)` | `Namespace` → `TrainingConfig` |

`DatasetBuilder.build()` returns `(train_ds, val_ds, steps_per_epoch, validation_steps)`. The
datasets must repeat — the pipeline passes `steps_per_epoch` to `fit`. `steps_per_epoch` may be
`None` only if `config.steps_per_epoch` is set; otherwise `run()` raises `ValueError`.

`get_test_data()` returning `None` means the classification visualizations and the calibration /
information-flow parts of the analysis are skipped.

## `TrainingConfig`

| Field | Default | Notes |
|---|---|---|
| `input_shape` | `(224, 224, 3)` | |
| `num_classes` | `1000` | |
| `epochs` / `batch_size` | `100` / `64` | |
| `lr_schedule_type` | `'cosine_decay'` | `'cosine_decay'`, `'exponential_decay'`, `'cosine_decay_restarts'`, `'constant'` |
| `learning_rate` | `1e-3` | |
| `decay_steps` | `None` | defaults to total training steps |
| `decay_rate` | `0.9` | `exponential_decay` only |
| `alpha` | `1e-4` | cosine schedules only |
| `t_mul` / `m_mul` | `2.0` / `0.9` | `cosine_decay_restarts` only |
| `warmup_steps` | `1000` | in optimizer steps |
| `warmup_start_lr` | `1e-8` | |
| `optimizer_type` | `'adamw'` | any type `optimizer_builder` accepts |
| `weight_decay` | `1e-4` | **only forwarded for `adamw`** — see gotchas |
| `beta_1` / `beta_2` / `epsilon` / `amsgrad` | Adam family | |
| `rho` / `momentum` / `centered` / `nesterov` | RMSprop / Adadelta / SGD | |
| `gradient_clipping_value` | `None` | → `clipvalue` |
| `gradient_clipping_norm_local` | `None` | → `clipnorm` |
| `gradient_clipping_norm_global` | `1.0` | → `global_clipnorm` (**on by default**) |
| `from_logits` | `False` | must match your model's output activation |
| `steps_per_epoch` / `validation_steps` | `None` | override the `DatasetBuilder` values |
| `early_stopping_patience` | `25` | |
| `monitor_metric` / `monitor_mode` | `'val_accuracy'` / `'max'` | drives checkpointing and early stopping |
| `output_dir` / `experiment_name` | `'results'` / auto | auto name is `f"{model_args['variant'] or 'model'}_{timestamp}"` |
| `enable_visualization` | `True` | |
| `enable_analysis` | `True` | post-training `ModelAnalyzer` run |
| `visualization_frequency` | `10` | plots every N epochs |
| `enable_convergence_analysis` / `enable_overfitting_analysis` | `True` | |
| `enable_gradient_tracking` | `False` | costs an extra forward/backward per plotted epoch |
| `enable_gradient_topology_viz` | `False` | |
| `enable_classification_viz` | `True` | needs `get_test_data()` |
| `create_final_dashboard` | `True` | |
| `model_args` | `{}` | free-form dict passed through to your `model_builder` |

`to_schedule_config(total_steps)` and `to_optimizer_config()` produce the dicts handed to
`learning_rate_schedule_builder` and `optimizer_builder`; call them yourself if you want to
inspect what the pipeline will build.

## CLI

`create_argument_parser()` defines exactly these flags — nothing else:

| Flag | Default |
|---|---|
| `--input-shape` (3 ints) | `224 224 3` |
| `--num-classes` | `1000` |
| `--epochs` | `100` |
| `--batch-size` | `64` |
| `--lr-schedule` | `cosine_decay` |
| `--learning-rate` | `0.001` |
| `--warmup-steps` | `1000` |
| `--alpha` | `0.0001` |
| `--optimizer` | `adamw` |
| `--weight-decay` | `0.0001` |
| `--gradient-clip` | `1.0` |
| `--from-logits` | off |
| `--output-dir` | `results` |
| `--experiment-name` | auto |
| `--no-visualization`, `--no-analysis`, `--no-convergence-analysis`, `--no-overfitting-analysis`, `--no-classification-viz`, `--no-final-dashboard` | off (feature enabled) |
| `--enable-gradient-tracking` | off |
| `--config` | load a saved `TrainingConfig` JSON |

There is no `--model`, `--dataset`, `--decay-steps` or `--visualization-frequency` flag; set
those fields on the `TrainingConfig` directly. Wire it up yourself:

```python
from dl_techniques.optimization.train_vision import create_argument_parser, config_from_args

args = create_argument_parser().parse_args()
config = config_from_args(args)
```

## Output layout

```
<output_dir>/<experiment_name>/
├── config.json                 # written at startup
├── best_model.keras            # ModelCheckpoint on monitor_metric
├── final_model.keras           # saved after fit()
├── training_log.csv            # CSVLogger
├── tensorboard_logs/
├── visualizations/<experiment_name>/<timestamp>/
│   ├── training_curves.png     lr_schedule.png
│   ├── convergence_analysis.png  overfitting_analysis.png
│   ├── gradient_topology.png   (enable_gradient_topology_viz)
│   ├── network_architecture.png  class_balance.png
│   ├── confusion_matrix.png    roc_pr_curves.png
│   ├── classification_report.png  per_class_analysis.png  error_analysis.png
│   └── dashboard.png           (create_final_dashboard)
└── analysis/                   # ModelAnalyzer output
    ├── analysis_results.json   summary_dashboard.png
    ├── training_dynamics.png   weight_learning_journey.png
    ├── confidence_calibration_analysis.png
    ├── information_flow_analysis.png
    └── spectral_summary.png
```

The extra `<experiment_name>/<timestamp>/` nesting under `visualizations/` comes from
`VisualizationContext.get_save_path`, not from this package.

## Gotchas

- **`weight_decay` is dropped for every optimizer except `adamw`.** `to_optimizer_config()` only
  adds the key on the `adamw` branch, so `optimizer_type='sgd', weight_decay=1e-4` trains with no
  decay and no warning.
- **Gradient clipping is on by default** (`gradient_clipping_norm_global=1.0`). Set it to `None`
  if you want unclipped gradients. Set at most one of the three clipping fields — Keras rejects
  more than one.
- `lr_schedule_type='constant'` bypasses `schedule_builder` entirely and passes the bare
  `learning_rate` float. `learning_rate_schedule_builder` itself has no `'constant'` type.
- `monitor_metric` defaults to `'val_accuracy'`; the pipeline names its metric `accuracy`, so
  validation must actually run or checkpointing and early stopping never fire.
- `from_logits=False` is the default. A model ending in a `Dense` without softmax will train
  against the wrong loss silently.
- `run()` calls `model.summary(expand_nested=True)` and, with analysis on, a full
  `ModelAnalyzer.analyze()`. On a large model that is minutes, not seconds — disable with
  `enable_analysis=False` for quick experiments.
- Analysis and visualization failures are caught and logged, not raised. A missing plot means
  reading the log, not a crash.

## See also

- `../README.md` — the builders this pipeline drives, including the `optimizer_builder` key-
  renaming trap.
- `../CLAUDE.md` — package map.
- `dl_techniques/analyzer/README.md`, `dl_techniques/visualization/README.md`.
