"""
Evaluation Metrics for Memory-Augmented Neural Networks.

This module provides metrics for evaluating MANN performance including
accuracy metrics, generalization metrics, memory-specific metrics, and
efficiency metrics.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import keras
import numpy as np


@dataclass
class EvaluationResult:
    """Container for evaluation results.
    
    :param metric_name: Name of the metric.
    :param value: Computed metric value.
    :param details: Optional detailed breakdown.
    :param metadata: Optional additional information.
    """
    metric_name: str
    value: float
    details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResults:
    """Aggregated results from a benchmark evaluation.
    
    :param task_name: Name of the benchmark task.
    :param metrics: Dictionary of metric name to EvaluationResult.
    :param passed: Whether the benchmark was passed (if applicable).
    :param error_rate: Overall error rate.
    """
    task_name: str
    metrics: Dict[str, EvaluationResult] = field(default_factory=dict)
    passed: Optional[bool] = None
    error_rate: Optional[float] = None


class SequenceAccuracy(keras.metrics.Metric):
    """Sequence-level accuracy metric.
    
    Measures the fraction of complete sequences that are exactly correct.
    A sequence is considered correct only if ALL elements match.
    
    :param threshold: Threshold for binary predictions.
    :param name: Name of the metric.
    
    Example::
    
        metric = SequenceAccuracy()
        metric.update_state(y_true, y_pred)
        accuracy = metric.result()
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        name: str = "sequence_accuracy",
        **kwargs
    ) -> None:
        """Initialize sequence accuracy metric.
        
        :param threshold: Threshold for converting predictions to binary.
        :param name: Metric name.
        :param kwargs: Additional arguments for base class.
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")
    
    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """Update metric state with new predictions.
        
        :param y_true: Ground truth sequences.
        :param y_pred: Predicted sequences.
        :param sample_weight: Optional sample weights.
        """
        # Binarize predictions
        y_pred_binary = keras.ops.cast(y_pred > self.threshold, y_true.dtype)
        
        # Check if entire sequence matches
        # Shape: (batch, seq_len, features) -> (batch,)
        matches = keras.ops.all(
            keras.ops.equal(y_true, y_pred_binary),
            axis=tuple(range(1, len(keras.ops.shape(y_true))))
        )
        
        if sample_weight is not None:
            matches = matches * keras.ops.cast(sample_weight, matches.dtype)
            self.correct.assign_add(keras.ops.sum(matches))
            self.total.assign_add(keras.ops.sum(sample_weight))
        else:
            self.correct.assign_add(
                keras.ops.cast(keras.ops.sum(matches), self.correct.dtype)
            )
            self.total.assign_add(
                keras.ops.cast(keras.ops.shape(y_true)[0], self.total.dtype)
            )
    
    def result(self) -> keras.KerasTensor:
        """Compute final accuracy value.
        
        :return: Sequence accuracy as a scalar.
        """
        return self.correct / (self.total + keras.backend.epsilon())
    
    def reset_state(self) -> None:
        """Reset metric state."""
        self.correct.assign(0.0)
        self.total.assign(0.0)
    
    def get_config(self) -> Dict[str, Any]:
        """Get metric configuration.
        
        :return: Configuration dictionary.
        """
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


class PerStepAccuracy(keras.metrics.Metric):
    """Per-step accuracy metric.
    
    Measures the fraction of individual timesteps that are correct,
    regardless of the sequence-level correctness.
    
    :param threshold: Threshold for binary predictions.
    :param name: Name of the metric.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        name: str = "per_step_accuracy",
        **kwargs
    ) -> None:
        """Initialize per-step accuracy metric.
        
        :param threshold: Threshold for converting predictions to binary.
        :param name: Metric name.
        :param kwargs: Additional arguments for base class.
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")
    
    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """Update metric state.
        
        :param y_true: Ground truth values.
        :param y_pred: Predicted values.
        :param sample_weight: Optional sample weights.
        """
        y_pred_binary = keras.ops.cast(y_pred > self.threshold, y_true.dtype)
        matches = keras.ops.cast(
            keras.ops.equal(y_true, y_pred_binary),
            "float32"
        )
        
        if sample_weight is not None:
            # Broadcast sample weight
            weight_shape = keras.ops.shape(y_true)
            sw = keras.ops.reshape(sample_weight, (-1,) + (1,) * (len(weight_shape) - 1))
            matches = matches * keras.ops.cast(sw, matches.dtype)
        
        self.correct.assign_add(keras.ops.sum(matches))
        self.total.assign_add(
            keras.ops.cast(keras.ops.size(y_true), self.total.dtype)
        )
    
    def result(self) -> keras.KerasTensor:
        """Compute final accuracy.
        
        :return: Per-step accuracy as a scalar.
        """
        return self.correct / (self.total + keras.backend.epsilon())
    
    def reset_state(self) -> None:
        """Reset metric state."""
        self.correct.assign(0.0)
        self.total.assign(0.0)


class BitErrorRate(keras.metrics.Metric):
    """Bit Error Rate metric for binary vector outputs.
    
    Measures the fraction of bits that differ between prediction
    and ground truth in binary sequences.
    
    :param threshold: Threshold for binary predictions.
    :param name: Name of the metric.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        name: str = "bit_error_rate",
        **kwargs
    ) -> None:
        """Initialize bit error rate metric.
        
        :param threshold: Threshold for binarization.
        :param name: Metric name.
        :param kwargs: Additional arguments.
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.errors = self.add_weight(name="errors", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")
    
    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """Update metric state.
        
        :param y_true: Ground truth binary values.
        :param y_pred: Predicted values.
        :param sample_weight: Optional sample weights (applied per-sample).
        """
        y_pred_binary = keras.ops.cast(y_pred > self.threshold, y_true.dtype)
        errors = keras.ops.cast(
            keras.ops.not_equal(y_true, y_pred_binary),
            "float32"
        )
        
        self.errors.assign_add(keras.ops.sum(errors))
        self.total.assign_add(
            keras.ops.cast(keras.ops.size(y_true), self.total.dtype)
        )
    
    def result(self) -> keras.KerasTensor:
        """Compute bit error rate.
        
        :return: BER as a scalar.
        """
        return self.errors / (self.total + keras.backend.epsilon())
    
    def reset_state(self) -> None:
        """Reset metric state."""
        self.errors.assign(0.0)
        self.total.assign(0.0)


class ExactMatchAccuracy(keras.metrics.Metric):
    """Exact match accuracy for sequence-to-sequence tasks.
    
    For categorical outputs where sequences must match exactly.
    
    :param pad_token_id: ID of padding token to ignore.
    :param name: Name of the metric.
    """
    
    def __init__(
        self,
        pad_token_id: int = 0,
        name: str = "exact_match_accuracy",
        **kwargs
    ) -> None:
        """Initialize exact match metric.
        
        :param pad_token_id: Padding token to ignore in comparison.
        :param name: Metric name.
        :param kwargs: Additional arguments.
        """
        super().__init__(name=name, **kwargs)
        self.pad_token_id = pad_token_id
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")
    
    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None
    ) -> None:
        """Update metric state.
        
        :param y_true: Ground truth token IDs.
        :param y_pred: Predicted logits or token IDs.
        :param sample_weight: Optional sample weights.
        """
        # If predictions are logits, convert to token IDs
        if len(keras.ops.shape(y_pred)) > len(keras.ops.shape(y_true)):
            y_pred = keras.ops.argmax(y_pred, axis=-1)
        
        y_pred = keras.ops.cast(y_pred, y_true.dtype)
        
        # Create mask for non-padding tokens
        mask = keras.ops.cast(
            keras.ops.not_equal(y_true, self.pad_token_id),
            "float32"
        )
        
        # Check element-wise equality
        equal = keras.ops.cast(keras.ops.equal(y_true, y_pred), "float32")
        equal = equal * mask
        
        # Sequence is correct if all non-padding tokens match
        seq_correct = keras.ops.all(
            keras.ops.equal(equal, mask),
            axis=tuple(range(1, len(keras.ops.shape(y_true))))
        )
        
        self.correct.assign_add(
            keras.ops.cast(keras.ops.sum(seq_correct), self.correct.dtype)
        )
        self.total.assign_add(
            keras.ops.cast(keras.ops.shape(y_true)[0], self.total.dtype)
        )
    
    def result(self) -> keras.KerasTensor:
        """Compute exact match accuracy.
        
        :return: Accuracy as a scalar.
        """
        return self.correct / (self.total + keras.backend.epsilon())
    
    def reset_state(self) -> None:
        """Reset metric state."""
        self.correct.assign(0.0)
        self.total.assign(0.0)


def compute_length_generalization_score(
    model: keras.Model,
    test_data_by_length: Dict[int, Tuple[np.ndarray, np.ndarray]],
    train_lengths: List[int],
    metric_fn: Optional[Callable] = None
) -> EvaluationResult:
    """Compute length generalization score.
    
    Measures how well model performance degrades as sequence length
    increases beyond training distribution.
    
    :param model: Trained Keras model.
    :param test_data_by_length: Dictionary mapping length to (inputs, targets).
    :param train_lengths: List of lengths seen during training.
    :param metric_fn: Function to compute per-length metric. Defaults to accuracy.
    :return: EvaluationResult with generalization score and breakdown.
    """
    if metric_fn is None:
        metric_fn = lambda y_true, y_pred: np.mean(
            np.all(np.round(y_pred) == y_true, axis=tuple(range(1, y_true.ndim)))
        )
    
    results_by_length = {}
    
    for length, (inputs, targets) in test_data_by_length.items():
        predictions = model.predict(inputs, verbose=0)
        score = metric_fn(targets, predictions)
        results_by_length[length] = float(score)
    
    # Compute generalization score
    # Ratio of OOD performance to in-distribution performance
    train_scores = [results_by_length[l] for l in train_lengths if l in results_by_length]
    ood_lengths = [l for l in results_by_length if l not in train_lengths]
    ood_scores = [results_by_length[l] for l in ood_lengths]
    
    if train_scores and ood_scores:
        avg_train = np.mean(train_scores)
        avg_ood = np.mean(ood_scores)
        gen_score = avg_ood / (avg_train + 1e-8) if avg_train > 0 else 0.0
    else:
        gen_score = 0.0
    
    return EvaluationResult(
        metric_name="length_generalization_score",
        value=gen_score,
        details={
            "by_length": results_by_length,
            "train_avg": float(np.mean(train_scores)) if train_scores else 0.0,
            "ood_avg": float(np.mean(ood_scores)) if ood_scores else 0.0
        },
        metadata={
            "train_lengths": train_lengths,
            "ood_lengths": ood_lengths
        }
    )


def compute_capacity_degradation_curve(
    model: keras.Model,
    test_data_by_capacity: Dict[int, Tuple[np.ndarray, np.ndarray]],
    metric_fn: Optional[Callable] = None
) -> EvaluationResult:
    """Compute memory capacity degradation curve.
    
    Measures how accuracy degrades as memory load increases.
    
    :param model: Trained Keras model.
    :param test_data_by_capacity: Dictionary mapping item count to data.
    :param metric_fn: Function to compute metric per capacity level.
    :return: EvaluationResult with degradation curve.
    """
    if metric_fn is None:
        metric_fn = lambda y_true, y_pred: np.mean(
            np.all(np.round(y_pred) == y_true, axis=tuple(range(1, y_true.ndim)))
        )
    
    results_by_capacity = {}
    
    for capacity, (inputs, targets) in sorted(test_data_by_capacity.items()):
        predictions = model.predict(inputs, verbose=0)
        score = metric_fn(targets, predictions)
        results_by_capacity[capacity] = float(score)
    
    # Find capacity at which performance drops below threshold
    threshold = 0.9
    max_capacity = 0
    for cap, score in sorted(results_by_capacity.items()):
        if score >= threshold:
            max_capacity = cap
        else:
            break
    
    # Compute area under the curve (normalized)
    capacities = sorted(results_by_capacity.keys())
    scores = [results_by_capacity[c] for c in capacities]
    auc = float(np.trapz(scores, capacities) / (capacities[-1] - capacities[0] + 1e-8))
    
    return EvaluationResult(
        metric_name="capacity_degradation",
        value=auc,
        details={
            "by_capacity": results_by_capacity,
            "max_capacity_at_90pct": max_capacity
        }
    )


class MemoryUtilizationMetric:
    """Metric for analyzing memory utilization patterns.
    
    Tracks how effectively the model uses its memory slots.
    
    :param memory_size: Number of memory slots.
    """
    
    def __init__(self, memory_size: int) -> None:
        """Initialize memory utilization metric.
        
        :param memory_size: Number of memory slots to track.
        """
        self.memory_size = memory_size
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.write_counts = np.zeros(self.memory_size)
        self.read_counts = np.zeros(self.memory_size)
        self.total_writes = 0
        self.total_reads = 0
        self.entropy_sum = 0.0
        self.num_steps = 0
    
    def update(
        self,
        write_weights: np.ndarray,
        read_weights: np.ndarray
    ) -> None:
        """Update with attention weights from a forward pass.
        
        :param write_weights: Write attention weights (batch, memory_size).
        :param read_weights: Read attention weights (batch, memory_size).
        """
        # Accumulate normalized weights
        write_sum = np.sum(write_weights, axis=0)
        read_sum = np.sum(read_weights, axis=0)
        
        self.write_counts += write_sum
        self.read_counts += read_sum
        self.total_writes += np.sum(write_sum)
        self.total_reads += np.sum(read_sum)
        
        # Compute entropy of attention distribution
        for weights in [write_weights, read_weights]:
            # Add small epsilon for numerical stability
            probs = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-8)
            probs = np.clip(probs, 1e-10, 1.0)
            entropy = -np.sum(probs * np.log(probs), axis=-1)
            self.entropy_sum += np.sum(entropy)
            self.num_steps += len(weights)
    
    def compute(self) -> EvaluationResult:
        """Compute final utilization metrics.
        
        :return: EvaluationResult with utilization statistics.
        """
        # Normalize counts
        write_dist = self.write_counts / (self.total_writes + 1e-8)
        read_dist = self.read_counts / (self.total_reads + 1e-8)
        
        # Compute utilization rate (fraction of slots used above threshold)
        threshold = 1.0 / (self.memory_size * 10)
        write_utilized = np.sum(write_dist > threshold) / self.memory_size
        read_utilized = np.sum(read_dist > threshold) / self.memory_size
        
        # Average entropy (higher = more uniform usage)
        avg_entropy = self.entropy_sum / (self.num_steps + 1e-8)
        max_entropy = np.log(self.memory_size)
        normalized_entropy = avg_entropy / max_entropy
        
        return EvaluationResult(
            metric_name="memory_utilization",
            value=float((write_utilized + read_utilized) / 2),
            details={
                "write_utilization": float(write_utilized),
                "read_utilization": float(read_utilized),
                "normalized_entropy": float(normalized_entropy),
                "write_distribution": write_dist.tolist(),
                "read_distribution": read_dist.tolist()
            }
        )


def evaluate_babi_task(
    model: keras.Model,
    stories: np.ndarray,
    questions: np.ndarray,
    answers: np.ndarray,
    task_id: int
) -> BenchmarkResults:
    """Evaluate model on a bAbI task.
    
    Standard bAbI success criterion: <5% error per task.
    
    :param model: Trained model.
    :param stories: Encoded story arrays.
    :param questions: Encoded question arrays.
    :param answers: Ground truth answer indices.
    :param task_id: bAbI task ID for reporting.
    :return: BenchmarkResults with pass/fail status.
    """
    # Assume model takes (stories, questions) as input
    predictions = model.predict([stories, questions], verbose=0)
    
    # Get predicted answer indices
    if len(predictions.shape) > 1:
        pred_indices = np.argmax(predictions, axis=-1)
    else:
        pred_indices = np.round(predictions).astype(int)
    
    # Flatten if needed
    true_indices = answers.flatten()
    pred_indices = pred_indices.flatten()
    
    # Compute accuracy and error rate
    correct = np.sum(pred_indices == true_indices)
    total = len(true_indices)
    accuracy = correct / total
    error_rate = 1.0 - accuracy
    
    # bAbI pass criterion: <5% error
    passed = error_rate < 0.05
    
    return BenchmarkResults(
        task_name=f"bAbI_task_{task_id}",
        metrics={
            "accuracy": EvaluationResult(
                metric_name="accuracy",
                value=accuracy
            ),
            "error_rate": EvaluationResult(
                metric_name="error_rate",
                value=error_rate
            )
        },
        passed=passed,
        error_rate=error_rate
    )


def evaluate_copy_task(
    model: keras.Model,
    inputs: np.ndarray,
    targets: np.ndarray,
    masks: Optional[np.ndarray] = None
) -> BenchmarkResults:
    """Evaluate model on copy task.
    
    :param model: Trained model.
    :param inputs: Input sequences.
    :param targets: Target sequences.
    :param masks: Optional output masks.
    :return: BenchmarkResults with multiple metrics.
    """
    predictions = model.predict(inputs, verbose=0)
    
    # Apply mask if provided
    if masks is not None:
        mask_expanded = np.expand_dims(masks, -1)
        predictions = predictions * mask_expanded
        targets = targets * mask_expanded
    
    # Binarize predictions
    pred_binary = (predictions > 0.5).astype(np.float32)
    
    # Sequence accuracy
    seq_matches = np.all(pred_binary == targets, axis=(1, 2))
    seq_accuracy = np.mean(seq_matches)
    
    # Per-step accuracy
    step_accuracy = np.mean(pred_binary == targets)
    
    # Bit error rate
    bit_errors = np.mean(pred_binary != targets)
    
    return BenchmarkResults(
        task_name="copy_task",
        metrics={
            "sequence_accuracy": EvaluationResult(
                metric_name="sequence_accuracy",
                value=float(seq_accuracy)
            ),
            "per_step_accuracy": EvaluationResult(
                metric_name="per_step_accuracy",
                value=float(step_accuracy)
            ),
            "bit_error_rate": EvaluationResult(
                metric_name="bit_error_rate",
                value=float(bit_errors)
            )
        },
        error_rate=float(1.0 - seq_accuracy)
    )


def evaluate_associative_recall(
    model: keras.Model,
    inputs: np.ndarray,
    targets: np.ndarray,
    tolerance: float = 0.1
) -> BenchmarkResults:
    """Evaluate model on associative recall task.
    
    :param model: Trained model.
    :param inputs: Input sequences (key-value pairs + query).
    :param targets: Target values.
    :param tolerance: Tolerance for continuous value comparison.
    :return: BenchmarkResults with recall metrics.
    """
    predictions = model.predict(inputs, verbose=0)
    
    # Ensure correct shape
    if len(predictions.shape) > len(targets.shape):
        predictions = predictions[:, -1, :]  # Take last timestep
    
    # Compute accuracy with tolerance
    within_tolerance = np.abs(predictions - targets) < tolerance
    recall_accuracy = np.mean(np.all(within_tolerance, axis=-1))
    
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # Cosine similarity
    pred_norm = predictions / (np.linalg.norm(predictions, axis=-1, keepdims=True) + 1e-8)
    target_norm = targets / (np.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)
    cosine_sim = np.mean(np.sum(pred_norm * target_norm, axis=-1))
    
    return BenchmarkResults(
        task_name="associative_recall",
        metrics={
            "recall_accuracy": EvaluationResult(
                metric_name="recall_accuracy",
                value=float(recall_accuracy)
            ),
            "mse": EvaluationResult(
                metric_name="mse",
                value=float(mse)
            ),
            "cosine_similarity": EvaluationResult(
                metric_name="cosine_similarity",
                value=float(cosine_sim)
            )
        },
        error_rate=float(1.0 - recall_accuracy)
    )
