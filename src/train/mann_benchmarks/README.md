# Memory-Augmented Neural Network Benchmarks

A comprehensive benchmark suite for evaluating memory-augmented neural networks (MANNs) based on evaluation metrics and tasks from the NTM/DNC literature through modern Transformer-memory architectures.

## Overview

This package provides:

1. **Data Generators** for classic and modern MANN benchmark tasks
2. **Evaluation Metrics** as Keras metrics and standalone functions
3. **Benchmark Harness** for running standardized evaluations
4. **Compositional Generalization** tests (SCAN, COGS-style)
5. **Algorithmic Reasoning** tasks (CLRS-style)

## Installation

```python
# Add to your project
from mann_benchmarks import BenchmarkHarness, CopyTaskConfig
```

## Quick Start

```python
from mann_benchmarks import (
    BenchmarkHarness,
    BenchmarkSuiteConfig,
    CopyTaskConfig,
)

# Create harness with default configuration
harness = BenchmarkHarness()

# Run individual benchmark
results = harness.run_copy_task_benchmark(model)
print(f"Sequence Accuracy: {results.metrics['sequence_accuracy'].value:.4f}")
print(f"Bit Error Rate: {results.metrics['bit_error_rate'].value:.6f}")

# Run full benchmark suite
report = harness.run_full_suite(model, model_name="MyMANN")
harness.save_report("benchmark_results.json", report)
```

## Benchmark Tasks

### Core NTM/DNC Tasks

| Task | Description | Key Metric |
|------|-------------|------------|
| **Copy** | Store and reproduce binary sequences | Sequence Accuracy |
| **Associative Recall** | Store key-value pairs, retrieve by query | Recall Accuracy |
| **Repeat Copy** | Reproduce sequence N times | Sequence Accuracy |
| **Priority Access** | Output items sorted by priority | Sequence Accuracy |
| **Graph Traversal** | BFS/DFS on adjacency matrix | Reachability Accuracy |

### Compositional Generalization

| Benchmark | Focus | Splits |
|-----------|-------|--------|
| **SCAN** | Navigation commands | Simple, Length, Add Primitive |
| **COGS** | Semantic parsing | Lexical, Structural |
| **CFQ** | Freebase queries | Compound Divergence |

### Algorithmic Reasoning (CLRS-style)

- Sorting: insertion_sort, bubble_sort
- Searching: binary_search, linear_search
- Graph: bfs, dfs, dijkstra
- Basic: minimum, maximum, reverse

### bAbI Tasks

Subset of 20 QA tasks testing:
- Single/Two/Three Supporting Facts
- Yes/No Questions
- Counting
- Lists/Sets
- Coreference
- Deduction
- Positional Reasoning
- Path Finding

## Evaluation Metrics

### Keras Metrics (for training)

```python
from mann_benchmarks import (
    SequenceAccuracy,
    PerStepAccuracy,
    BitErrorRate,
    ExactMatchAccuracy,
)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        SequenceAccuracy(threshold=0.5),
        PerStepAccuracy(),
        BitErrorRate(),
    ]
)
```

### Evaluation Functions

```python
from mann_benchmarks import (
    evaluate_copy_task,
    evaluate_associative_recall,
    evaluate_babi_task,
    compute_length_generalization_score,
    compute_capacity_degradation_curve,
)

# Copy task evaluation
results = evaluate_copy_task(model, inputs, targets, masks)
# Returns: sequence_accuracy, per_step_accuracy, bit_error_rate

# Length generalization
score = compute_length_generalization_score(
    model,
    test_data_by_length,
    train_lengths=[5, 10, 15, 20]
)
# Returns: generalization ratio (OOD/in-dist performance)

# Memory capacity
curve = compute_capacity_degradation_curve(
    model,
    test_data_by_capacity
)
# Returns: AUC of accuracy vs memory load curve
```

## Data Generators

### Copy Task

```python
from mann_benchmarks import CopyTaskGenerator, CopyTaskConfig

config = CopyTaskConfig(
    sequence_length=10,
    vector_size=8,
    delay_length=1,
    num_samples=1000
)
generator = CopyTaskGenerator(config)
data = generator.generate()
# data.inputs: (batch, seq_len, features)
# data.targets: (batch, seq_len, features)
# data.masks: (batch, seq_len)
```

### SCAN Compositional

```python
from mann_benchmarks import ScanGenerator, ScanTaskConfig

config = ScanTaskConfig(split_type='length')
generator = ScanGenerator(config)
train_samples, test_samples = generator.generate_split()

# Encode for model
inputs, targets, input_lens, target_lens = generator.encode_samples(test_samples)
```

### Algorithmic Tasks

```python
from mann_benchmarks import AlgorithmicTaskGenerator, AlgorithmicTaskConfig

config = AlgorithmicTaskConfig(
    task_name='insertion_sort',
    train_size=16,
    test_size=64
)
generator = AlgorithmicTaskGenerator(config)

# In-distribution data
train_data = generator.generate(problem_size=16)

# Out-of-distribution data for generalization testing
test_data = generator.generate(problem_size=64)
```

## Benchmark Harness

### Running Individual Benchmarks

```python
harness = BenchmarkHarness()

# Copy task
copy_results = harness.run_copy_task_benchmark(model)

# Length generalization
len_gen = harness.run_length_generalization_benchmark(model)

# Memory capacity
capacity = harness.run_capacity_benchmark(model, item_counts=[2,4,8,16,32])

# SCAN compositional
scan_results = harness.run_scan_benchmark(model)

# Algorithmic
algo_results = harness.run_algorithmic_benchmark(model)
```

### Full Suite Evaluation

```python
harness = BenchmarkHarness(BenchmarkSuiteConfig(verbose=True))

report = harness.run_full_suite(
    model,
    model_name="NTM_v1",
    benchmarks=[
        "copy_task",
        "associative_recall",
        "length_generalization",
        "memory_capacity",
        "scan"
    ]
)

# Save results
harness.save_report("results/ntm_benchmark.json", report)

# Access summary
print(f"Total benchmarks: {report.summary['total_benchmarks']}")
print(f"Passed: {report.summary['benchmarks_passed']}")
print(f"Avg error rate: {report.summary['average_error_rate']:.4f}")
```

### Training Callbacks

```python
from mann_benchmarks import create_benchmark_callbacks

callbacks = create_benchmark_callbacks(
    harness,
    validation_data=(val_inputs, val_targets),
    benchmark_interval=5  # Evaluate every 5 epochs
)

model.fit(
    train_inputs, train_targets,
    epochs=100,
    callbacks=callbacks
)
```

## Success Criteria

Based on literature standards:

| Task | Pass Criterion |
|------|----------------|
| bAbI | < 5% error per task |
| Copy | > 99% sequence accuracy |
| Associative Recall | > 95% recall accuracy |
| SCAN (simple) | > 99% exact match |
| SCAN (length) | > 20% exact match |
| Length Generalization | > 0.8 generalization ratio |

## Configuration

### BenchmarkSuiteConfig

```python
from mann_benchmarks import BenchmarkSuiteConfig

config = BenchmarkSuiteConfig(
    copy_config=CopyTaskConfig(sequence_length=20),
    associative_recall_config=AssociativeRecallConfig(num_items=8),
    length_gen_config=LengthGeneralizationConfig(
        train_lengths=[5, 10, 15],
        test_lengths=[20, 30, 50, 100]
    ),
    output_dir="./results",
    verbose=True
)

harness = BenchmarkHarness(config)
```

## References

Key papers informing this benchmark suite:

1. **Neural Turing Machines** (Graves et al., 2014)
2. **Differentiable Neural Computer** (Graves et al., 2016)
3. **SCAN** (Lake & Baroni, 2018) - Compositional generalization
4. **COGS** (Kim & Linzen, 2020) - Semantic parsing generalization
5. **CLRS Benchmark** (Veličković et al., 2022) - Algorithmic reasoning
6. **bAbI Tasks** (Weston et al., 2015) - QA reasoning
7. **Titans** (Behrouz et al., 2024) - Modern memory architectures
