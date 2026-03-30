# Optimization

Training optimization utilities

**10 modules in this category**

## Constants

### optimization.constants
Default Configuration Constants for Optimization Module.

*📁 File: `src/dl_techniques/optimization/constants.py`*

## Core

### optimization
Optimization Module for Deep Learning Techniques.

*📁 File: `src/dl_techniques/optimization/__init__.py`*

## Deep_Supervision

### optimization.deep_supervision
Deep Supervision Weight Scheduling Module for Deep Learning Techniques.

**Classes:**
- `ScheduleType`

**Functions:** `schedule_builder`, `constant_equal_schedule`, `constant_low_to_high_schedule`, `constant_high_to_low_schedule`, `linear_low_to_high_schedule` (and 6 more)

*📁 File: `src/dl_techniques/optimization/deep_supervision.py`*

## Muon_Optimizer

### optimization.muon_optimizer
Muon: MomentUm Orthogonalized by Newton-schulz Optimizer.

**Classes:**
- `Muon`

**Functions:** `build`, `update_step`, `get_config`, `from_config`, `process_wide` (and 2 more)

*📁 File: `src/dl_techniques/optimization/muon_optimizer.py`*

## Optimizer

### optimization.optimizer
Optimizer and Learning Rate Schedule Builder Module for Deep Learning Techniques.

**Classes:**
- `ScheduleType`
- `OptimizerType`

**Functions:** `learning_rate_schedule_builder`, `optimizer_builder`

*📁 File: `src/dl_techniques/optimization/optimizer.py`*

## Schedule

### optimization.schedule
Learning Rate Schedule Builder Module for Deep Learning Techniques.

**Classes:**
- `ScheduleType`

**Functions:** `schedule_builder`

*📁 File: `src/dl_techniques/optimization/schedule.py`*

## Sled_Supervision

### optimization.sled_supervision
Self Logits Evolution Decoding (SLED) Module for Keras 3.

**Classes:**
- `SledEvolutionType`
- `SledLogitsProcessor`

**Functions:** `sled_builder`

*📁 File: `src/dl_techniques/optimization/sled_supervision.py`*

## Train_Vision

### optimization.train_vision

*📁 File: `src/dl_techniques/optimization/train_vision/__init__.py`*

### optimization.train_vision.framework
Model-agnostic training framework for vision tasks with integrated

**Classes:**
- `TrainingConfig`
- `DatasetBuilder`
- `EnhancedVisualizationCallback`
- `TrainingPipeline`

**Functions:** `create_argument_parser`, `config_from_args`, `to_schedule_config`, `to_optimizer_config`, `save` (and 8 more)

*📁 File: `src/dl_techniques/optimization/train_vision/framework.py`*

## Warmup_Schedule

### optimization.warmup_schedule
Warmup Learning Rate Schedule Implementation for Deep Learning Techniques.

**Classes:**
- `WarmupSchedule`

**Functions:** `get_config`, `from_config`, `warmup_fn`, `primary_fn`

*📁 File: `src/dl_techniques/optimization/warmup_schedule.py`*