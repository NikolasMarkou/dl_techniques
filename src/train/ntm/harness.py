"""
Benchmark Harness for Memory-Augmented Neural Networks.

Provides a unified interface for running benchmark suites, collecting
results, and generating reports for MANN evaluation.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
import numpy as np

from .babi_generator import BabiGenerator, BabiSample
from .compositional_generators import (
    CFQGenerator,
    CogsGenerator,
    ScanGenerator,
)
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


@dataclass
class BenchmarkRun:
    """Record of a single benchmark run.
    
    :param benchmark_name: Name of the benchmark.
    :param model_name: Name of the model being evaluated.
    :param results: BenchmarkResults from the evaluation.
    :param runtime_seconds: Time taken for the run.
    :param timestamp: When the run was executed.
    :param config: Configuration used for the benchmark.
    """
    benchmark_name: str
    model_name: str
    results: BenchmarkResults
    runtime_seconds: float
    timestamp: str
    config: Optional[Dict[str, Any]] = None


@dataclass
class SuiteReport:
    """Complete report from a benchmark suite run.
    
    :param suite_name: Name of the benchmark suite.
    :param model_name: Name of the evaluated model.
    :param runs: List of individual benchmark runs.
    :param summary: Aggregated summary statistics.
    :param total_runtime: Total time for all benchmarks.
    """
    suite_name: str
    model_name: str
    runs: List[BenchmarkRun]
    summary: Dict[str, Any]
    total_runtime: float


class BenchmarkHarness:
    """Harness for running MANN benchmarks.
    
    Provides a unified interface for executing benchmarks, collecting
    metrics, and generating reports.
    
    :param config: Configuration for the benchmark suite.
    
    Example::
    
        harness = BenchmarkHarness()
        results = harness.run_copy_task_benchmark(model)
        harness.run_full_suite(model, model_name="MyMANN")
        harness.save_report("results.json")
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkSuiteConfig] = None
    ) -> None:
        """Initialize the benchmark harness.
        
        :param config: Benchmark suite configuration. Uses defaults if None.
        """
        self.config = config or BenchmarkSuiteConfig()
        self._runs: List[BenchmarkRun] = []
        self._start_time: Optional[float] = None
    
    def run_copy_task_benchmark(
        self,
        model: keras.Model,
        config: Optional[CopyTaskConfig] = None
    ) -> BenchmarkResults:
        """Run the copy task benchmark.
        
        :param model: Keras model to evaluate.
        :param config: Copy task configuration.
        :return: BenchmarkResults with copy task metrics.
        """
        config = config or self.config.copy_config
        generator = CopyTaskGenerator(config)
        
        # Generate test data
        data = generator.generate()
        
        start_time = time.time()
        results = evaluate_copy_task(
            model,
            data.inputs,
            data.targets,
            data.masks
        )
        runtime = time.time() - start_time
        
        self._record_run(
            "copy_task",
            model.name if hasattr(model, 'name') else "unknown",
            results,
            runtime,
            asdict(config)
        )
        
        return results
    
    def run_associative_recall_benchmark(
        self,
        model: keras.Model,
        config: Optional[AssociativeRecallConfig] = None
    ) -> BenchmarkResults:
        """Run the associative recall benchmark.
        
        :param model: Keras model to evaluate.
        :param config: Associative recall configuration.
        :return: BenchmarkResults with recall metrics.
        """
        config = config or self.config.associative_recall_config
        generator = AssociativeRecallGenerator(config)
        
        data = generator.generate()
        
        start_time = time.time()
        results = evaluate_associative_recall(
            model,
            data.inputs,
            data.targets
        )
        runtime = time.time() - start_time
        
        self._record_run(
            "associative_recall",
            model.name if hasattr(model, 'name') else "unknown",
            results,
            runtime,
            asdict(config)
        )
        
        return results
    
    def run_length_generalization_benchmark(
        self,
        model: keras.Model,
        config: Optional[LengthGeneralizationConfig] = None
    ) -> EvaluationResult:
        """Run length generalization evaluation.
        
        :param model: Keras model to evaluate.
        :param config: Length generalization configuration.
        :return: EvaluationResult with generalization score.
        """
        config = config or self.config.length_gen_config
        copy_config = CopyTaskConfig(random_seed=config.random_seed)
        generator = CopyTaskGenerator(copy_config)
        
        # Generate datasets for each length
        datasets = generator.generate_length_generalization_set(config)
        
        # Convert to format expected by metric function
        test_data_by_length = {}
        for length, data in datasets.items():
            test_data_by_length[length] = (data.inputs, data.targets)
        
        start_time = time.time()
        result = compute_length_generalization_score(
            model,
            test_data_by_length,
            config.train_lengths
        )
        runtime = time.time() - start_time
        
        # Create benchmark results wrapper
        bench_results = BenchmarkResults(
            task_name="length_generalization",
            metrics={"generalization_score": result}
        )
        
        self._record_run(
            "length_generalization",
            model.name if hasattr(model, 'name') else "unknown",
            bench_results,
            runtime,
            asdict(config)
        )
        
        return result
    
    def run_capacity_benchmark(
        self,
        model: keras.Model,
        item_counts: Optional[List[int]] = None
    ) -> EvaluationResult:
        """Run memory capacity evaluation.
        
        :param model: Keras model to evaluate.
        :param item_counts: List of item counts to test.
        :return: EvaluationResult with capacity metrics.
        """
        if item_counts is None:
            item_counts = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
        
        config = self.config.associative_recall_config
        generator = AssociativeRecallGenerator(config)
        
        # Generate capacity test data
        datasets = generator.generate_capacity_test(item_counts)
        
        test_data_by_capacity = {}
        for count, data in datasets.items():
            test_data_by_capacity[count] = (data.inputs, data.targets)
        
        start_time = time.time()
        result = compute_capacity_degradation_curve(
            model,
            test_data_by_capacity
        )
        runtime = time.time() - start_time
        
        bench_results = BenchmarkResults(
            task_name="memory_capacity",
            metrics={"capacity_curve": result}
        )
        
        self._record_run(
            "memory_capacity",
            model.name if hasattr(model, 'name') else "unknown",
            bench_results,
            runtime,
            {"item_counts": item_counts}
        )
        
        return result
    
    def run_babi_benchmark(
        self,
        model: keras.Model,
        config: Optional[BabiTaskConfig] = None
    ) -> Dict[int, BenchmarkResults]:
        """Run bAbI task benchmark suite.
        
        :param model: Keras model to evaluate.
        :param config: bAbI task configuration.
        :return: Dictionary mapping task ID to BenchmarkResults.
        """
        config = config or self.config.babi_config
        generator = BabiGenerator(config)
        
        all_results = {}
        
        for task_id in config.task_ids:
            try:
                samples = generator.generate(task_id)
                stories, questions, answers = generator.encode_batch(samples)
                
                start_time = time.time()
                results = evaluate_babi_task(
                    model, stories, questions, answers, task_id
                )
                runtime = time.time() - start_time
                
                all_results[task_id] = results
                
                self._record_run(
                    f"bAbI_task_{task_id}",
                    model.name if hasattr(model, 'name') else "unknown",
                    results,
                    runtime,
                    {"task_id": task_id}
                )
                
            except ValueError as e:
                if self.config.verbose:
                    print(f"Skipping task {task_id}: {e}")
        
        return all_results
    
    def run_scan_benchmark(
        self,
        model: keras.Model,
        config: Optional[ScanTaskConfig] = None
    ) -> BenchmarkResults:
        """Run SCAN compositional generalization benchmark.
        
        :param model: Keras model to evaluate.
        :param config: SCAN task configuration.
        :return: BenchmarkResults with compositional metrics.
        """
        config = config or self.config.scan_config
        generator = ScanGenerator(config)
        
        train_samples, test_samples = generator.generate_split()
        
        # Encode test samples
        test_inputs, test_targets, _, _ = generator.encode_samples(test_samples)
        
        start_time = time.time()
        predictions = model.predict(test_inputs, verbose=0)
        runtime = time.time() - start_time
        
        # Compute exact match accuracy
        if len(predictions.shape) > 2:
            pred_tokens = np.argmax(predictions, axis=-1)
        else:
            pred_tokens = np.round(predictions).astype(int)
        
        # Mask padding
        mask = test_targets != 0
        matches = (pred_tokens == test_targets) | ~mask
        seq_accuracy = np.mean(np.all(matches, axis=-1))
        
        results = BenchmarkResults(
            task_name=f"SCAN_{config.split_type}",
            metrics={
                "sequence_accuracy": EvaluationResult(
                    metric_name="sequence_accuracy",
                    value=float(seq_accuracy)
                )
            },
            error_rate=float(1.0 - seq_accuracy)
        )
        
        self._record_run(
            f"SCAN_{config.split_type}",
            model.name if hasattr(model, 'name') else "unknown",
            results,
            runtime,
            asdict(config)
        )
        
        return results
    
    def run_algorithmic_benchmark(
        self,
        model: keras.Model,
        config: Optional[AlgorithmicTaskConfig] = None
    ) -> BenchmarkResults:
        """Run CLRS-style algorithmic reasoning benchmark.
        
        :param model: Keras model to evaluate.
        :param config: Algorithmic task configuration.
        :return: BenchmarkResults with algorithmic task metrics.
        """
        config = config or self.config.algorithmic_config
        generator = AlgorithmicTaskGenerator(config)
        
        # Generate in-distribution test data
        in_dist_data = generator.generate(problem_size=config.train_size)
        
        # Generate out-of-distribution test data
        ood_data = generator.generate(problem_size=config.test_size)
        
        start_time = time.time()
        
        # Evaluate in-distribution
        in_dist_pred = model.predict(in_dist_data.inputs, verbose=0)
        in_dist_acc = self._compute_task_accuracy(
            in_dist_data.targets, in_dist_pred, config.task_name
        )
        
        # Evaluate out-of-distribution
        ood_pred = model.predict(ood_data.inputs, verbose=0)
        ood_acc = self._compute_task_accuracy(
            ood_data.targets, ood_pred, config.task_name
        )
        
        runtime = time.time() - start_time
        
        results = BenchmarkResults(
            task_name=f"algorithmic_{config.task_name}",
            metrics={
                "in_distribution_accuracy": EvaluationResult(
                    metric_name="in_distribution_accuracy",
                    value=float(in_dist_acc)
                ),
                "ood_accuracy": EvaluationResult(
                    metric_name="ood_accuracy",
                    value=float(ood_acc)
                ),
                "generalization_ratio": EvaluationResult(
                    metric_name="generalization_ratio",
                    value=float(ood_acc / (in_dist_acc + 1e-8))
                )
            },
            error_rate=float(1.0 - ood_acc)
        )
        
        self._record_run(
            f"algorithmic_{config.task_name}",
            model.name if hasattr(model, 'name') else "unknown",
            results,
            runtime,
            asdict(config)
        )
        
        return results
    
    def _compute_task_accuracy(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        task_name: str
    ) -> float:
        """Compute task-specific accuracy.
        
        :param targets: Ground truth values.
        :param predictions: Model predictions.
        :param task_name: Name of the task.
        :return: Accuracy value.
        """
        if task_name in ["insertion_sort", "bubble_sort", "reverse"]:
            # Sequence matching with tolerance
            tolerance = 0.1
            matches = np.abs(predictions - targets) < tolerance
            return float(np.mean(np.all(matches, axis=(1, 2))))
        
        elif task_name in ["binary_search", "linear_search", "minimum", "maximum"]:
            # Argmax matching
            pred_idx = np.argmax(predictions, axis=-1)
            true_idx = np.argmax(targets, axis=-1)
            return float(np.mean(pred_idx == true_idx))
        
        elif task_name in ["bfs", "dfs"]:
            # Ranking correlation
            from scipy.stats import spearmanr
            correlations = []
            for i in range(len(targets)):
                corr, _ = spearmanr(targets[i], predictions[i].flatten())
                correlations.append(corr if not np.isnan(corr) else 0.0)
            return float(np.mean(correlations))
        
        else:
            # Default: MSE-based accuracy
            mse = np.mean((predictions - targets) ** 2)
            return float(np.exp(-mse))
    
    def run_full_suite(
        self,
        model: keras.Model,
        model_name: str,
        benchmarks: Optional[List[str]] = None
    ) -> SuiteReport:
        """Run the complete benchmark suite.
        
        :param model: Keras model to evaluate.
        :param model_name: Name for reporting.
        :param benchmarks: List of benchmarks to run. Runs all if None.
        :return: SuiteReport with all results.
        """
        self._runs = []
        self._start_time = time.time()
        
        available_benchmarks = {
            "copy_task": self.run_copy_task_benchmark,
            "associative_recall": self.run_associative_recall_benchmark,
            "length_generalization": self.run_length_generalization_benchmark,
            "memory_capacity": self.run_capacity_benchmark,
            "scan": self.run_scan_benchmark,
        }
        
        if benchmarks is None:
            benchmarks = list(available_benchmarks.keys())
        
        for benchmark_name in benchmarks:
            if benchmark_name in available_benchmarks:
                if self.config.verbose:
                    print(f"Running {benchmark_name}...")
                try:
                    available_benchmarks[benchmark_name](model)
                except Exception as e:
                    if self.config.verbose:
                        print(f"Error in {benchmark_name}: {e}")
        
        total_runtime = time.time() - self._start_time
        
        # Compute summary
        summary = self._compute_summary()
        
        return SuiteReport(
            suite_name="MANN_Benchmark_Suite",
            model_name=model_name,
            runs=self._runs.copy(),
            summary=summary,
            total_runtime=total_runtime
        )
    
    def _record_run(
        self,
        benchmark_name: str,
        model_name: str,
        results: BenchmarkResults,
        runtime: float,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a benchmark run.
        
        :param benchmark_name: Name of the benchmark.
        :param model_name: Name of the model.
        :param results: Results from the benchmark.
        :param runtime: Runtime in seconds.
        :param config: Configuration used.
        """
        run = BenchmarkRun(
            benchmark_name=benchmark_name,
            model_name=model_name,
            results=results,
            runtime_seconds=runtime,
            timestamp=datetime.now().isoformat(),
            config=config
        )
        self._runs.append(run)
        
        if self.config.verbose:
            self._print_run_summary(run)
    
    def _print_run_summary(self, run: BenchmarkRun) -> None:
        """Print summary of a benchmark run.
        
        :param run: BenchmarkRun to summarize.
        """
        print(f"\n{'='*50}")
        print(f"Benchmark: {run.benchmark_name}")
        print(f"Runtime: {run.runtime_seconds:.2f}s")
        
        for metric_name, result in run.results.metrics.items():
            print(f"  {metric_name}: {result.value:.4f}")
        
        if run.results.passed is not None:
            status = "PASSED" if run.results.passed else "FAILED"
            print(f"  Status: {status}")
        print(f"{'='*50}\n")
    
    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics from all runs.
        
        :return: Summary dictionary.
        """
        summary = {
            "total_benchmarks": len(self._runs),
            "total_runtime": sum(r.runtime_seconds for r in self._runs),
            "benchmarks_passed": sum(
                1 for r in self._runs 
                if r.results.passed is True
            ),
            "benchmarks_failed": sum(
                1 for r in self._runs 
                if r.results.passed is False
            ),
            "average_error_rate": np.mean([
                r.results.error_rate for r in self._runs 
                if r.results.error_rate is not None
            ]) if self._runs else 0.0
        }
        
        # Per-benchmark summary
        summary["per_benchmark"] = {}
        for run in self._runs:
            summary["per_benchmark"][run.benchmark_name] = {
                "runtime": run.runtime_seconds,
                "error_rate": run.results.error_rate,
                "passed": run.results.passed
            }
        
        return summary
    
    def save_report(
        self,
        filepath: str,
        report: Optional[SuiteReport] = None
    ) -> None:
        """Save benchmark report to JSON file.
        
        :param filepath: Path to save the report.
        :param report: Report to save. Uses latest run if None.
        """
        if report is None:
            report = SuiteReport(
                suite_name="MANN_Benchmark_Suite",
                model_name="unknown",
                runs=self._runs,
                summary=self._compute_summary(),
                total_runtime=sum(r.runtime_seconds for r in self._runs)
            )
        
        # Convert to serializable format
        def make_serializable(obj):
            if isinstance(obj, (BenchmarkRun, SuiteReport, BenchmarkResults, EvaluationResult)):
                return {k: make_serializable(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            else:
                return obj
        
        serializable_report = make_serializable(report)
        
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        if self.config.verbose:
            print(f"Report saved to {filepath}")
    
    def get_keras_metrics(self) -> List[keras.metrics.Metric]:
        """Get Keras metric objects for training callbacks.
        
        :return: List of Keras metrics.
        """
        return [
            SequenceAccuracy(name="seq_acc"),
            PerStepAccuracy(name="step_acc"),
            BitErrorRate(name="ber"),
            ExactMatchAccuracy(name="exact_match")
        ]


def create_benchmark_callbacks(
    harness: BenchmarkHarness,
    validation_data: Tuple[np.ndarray, np.ndarray],
    benchmark_interval: int = 5
) -> List[keras.callbacks.Callback]:
    """Create Keras callbacks for benchmark evaluation during training.
    
    :param harness: BenchmarkHarness instance.
    :param validation_data: Tuple of (inputs, targets) for validation.
    :param benchmark_interval: Epochs between benchmark runs.
    :return: List of Keras callbacks.
    """
    class BenchmarkCallback(keras.callbacks.Callback):
        """Callback for running benchmarks during training."""
        
        def __init__(
            self,
            harness: BenchmarkHarness,
            val_data: Tuple[np.ndarray, np.ndarray],
            interval: int
        ) -> None:
            super().__init__()
            self.harness = harness
            self.val_inputs, self.val_targets = val_data
            self.interval = interval
            self.history: List[Dict[str, float]] = []
        
        def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
            if (epoch + 1) % self.interval == 0:
                results = evaluate_copy_task(
                    self.model,
                    self.val_inputs,
                    self.val_targets
                )
                
                metrics = {
                    f"benchmark_{k}": v.value 
                    for k, v in results.metrics.items()
                }
                self.history.append({"epoch": epoch + 1, **metrics})
                
                if logs is not None:
                    logs.update(metrics)
    
    return [BenchmarkCallback(harness, validation_data, benchmark_interval)]
