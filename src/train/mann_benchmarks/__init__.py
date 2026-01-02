"""
Memory-Augmented Neural Network Benchmarks.

A comprehensive benchmark suite for evaluating memory-augmented neural networks
based on evaluation metrics from the literature including NTM, DNC, and modern
Transformer-memory architectures.

Benchmark Categories
--------------------
1. **Accuracy Metrics**: Sequence accuracy, per-step accuracy, exact match, BER
2. **Generalization Metrics**: Length generalization, compositional generalization
3. **Memory-Specific Metrics**: Memory utilization, capacity degradation
4. **Efficiency Metrics**: Training time, inference throughput

Supported Tasks
---------------
- Copy Task (classic NTM benchmark)
- Associative Recall
- Repeat Copy
- Priority Access
- Graph Traversal
- Dynamic N-Gram
- bAbI QA Tasks (subset)
- SCAN Compositional Generalization
- COGS Semantic Parsing
- CLRS Algorithmic Reasoning

Example Usage
-------------
>>> from mann_benchmarks import BenchmarkHarness, CopyTaskConfig
>>> 
>>> # Create harness with default config
>>> harness = BenchmarkHarness()
>>> 
>>> # Run individual benchmark
>>> results = harness.run_copy_task_benchmark(model)
>>> print(f"Sequence Accuracy: {results.metrics['sequence_accuracy'].value}")
>>> 
>>> # Run full suite
>>> report = harness.run_full_suite(model, model_name="MyMANN")
>>> harness.save_report("results.json", report)
>>> 
>>> # Use Keras metrics in training
>>> metrics = harness.get_keras_metrics()
>>> model.compile(optimizer='adam', loss='bce', metrics=metrics)
"""
from .config import (
    AlgorithmicTaskConfig,
    AssociativeRecallConfig,
    BabiTaskConfig,
    BenchmarkSuiteConfig,
    CopyTaskConfig,
    LengthGeneralizationConfig,
    ScanTaskConfig,
)
from .data_generators import (
    AlgorithmicTaskGenerator,
    AssociativeRecallGenerator,
    CopyTaskGenerator,
    DynamicNGramGenerator,
    PriorityAccessGenerator,
    RepeatCopyGenerator,
    TaskData,
    TraversalGenerator,
)
from .compositional_generators import (
    CFQGenerator,
    CogsExample,
    CogsGenerator,
    ScanGenerator,
    ScanSample,
    ScanSplit,
)
from .babi_generator import (
    BabiGenerator,
    BabiSample,
)
from .metrics import (
    BenchmarkResults,
    BitErrorRate,
    EvaluationResult,
    ExactMatchAccuracy,
    MemoryUtilizationMetric,
    PerStepAccuracy,
    SequenceAccuracy,
    compute_capacity_degradation_curve,
    compute_length_generalization_score,
    evaluate_associative_recall,
    evaluate_babi_task,
    evaluate_copy_task,
)
from .harness import (
    BenchmarkHarness,
    BenchmarkRun,
    SuiteReport,
    create_benchmark_callbacks,
)

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "AlgorithmicTaskConfig",
    "AssociativeRecallConfig",
    "BabiTaskConfig",
    "BenchmarkSuiteConfig",
    "CopyTaskConfig",
    "LengthGeneralizationConfig",
    "ScanTaskConfig",
    # Data generators
    "AlgorithmicTaskGenerator",
    "AssociativeRecallGenerator",
    "CopyTaskGenerator",
    "DynamicNGramGenerator",
    "PriorityAccessGenerator",
    "RepeatCopyGenerator",
    "TaskData",
    "TraversalGenerator",
    # Compositional generators
    "CFQGenerator",
    "CogsExample",
    "CogsGenerator",
    "ScanGenerator",
    "ScanSample",
    "ScanSplit",
    # bAbI generator
    "BabiGenerator",
    "BabiSample",
    # Metrics
    "BenchmarkResults",
    "BitErrorRate",
    "EvaluationResult",
    "ExactMatchAccuracy",
    "MemoryUtilizationMetric",
    "PerStepAccuracy",
    "SequenceAccuracy",
    "compute_capacity_degradation_curve",
    "compute_length_generalization_score",
    "evaluate_associative_recall",
    "evaluate_babi_task",
    "evaluate_copy_task",
    # Harness
    "BenchmarkHarness",
    "BenchmarkRun",
    "SuiteReport",
    "create_benchmark_callbacks",
]
