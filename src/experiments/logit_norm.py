import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    regular_predictions: np.ndarray
    coupled_predictions: np.ndarray
    metrics: Dict[str, float]


class ExperimentRunner:
    """Runner for LogitNorm experiments."""

    def __init__(
            self,
            num_classes: int = 10,
            coupling_strength: float = 1.0,
            constant: float = 1.0
    ):
        self.regular_head = tf.keras.layers.Dense(num_classes, activation='sigmoid')
        self.coupled_head = CoupledMultiLabelHead(
            constant=constant,
            coupling_strength=coupling_strength
        )

    def run_experiment_1_mutually_exclusive_labels(
            self,
            num_samples: int = 1000
    ) -> ExperimentResults:
        """
        Experiment 1: Mutually Exclusive Labels

        Tests how well the model handles cases where labels should be mutually exclusive.
        Example: Medical diagnosis where certain conditions cannot co-occur.
        """
        # Generate synthetic logits that should produce mutually exclusive predictions
        logits = tf.random.normal((num_samples, 3)) * 5  # Scaled for stronger signals

        # Get predictions
        regular_preds = self.regular_head(logits)
        coupled_preds = self.coupled_head(logits)

        # Calculate metrics
        metrics = {
            'regular_multi_active': np.mean(np.sum(regular_preds > 0.5, axis=1) > 1),
            'coupled_multi_active': np.mean(np.sum(coupled_preds > 0.5, axis=1) > 1),
            'regular_mean_conf': np.mean(regular_preds),
            'coupled_mean_conf': np.mean(coupled_preds)
        }

        return ExperimentResults(regular_preds, coupled_preds, metrics)

    def run_experiment_2_hierarchical_labels(
            self,
            num_samples: int = 1000
    ) -> ExperimentResults:
        """
        Experiment 2: Hierarchical Labels

        Tests handling of hierarchical label relationships.
        Example: Image classification where "vehicle" → "car" → "sports car"
        """
        # Generate hierarchical logits (parent should influence children)
        parent_logits = tf.random.normal((num_samples, 1)) * 2
        child1_logits = parent_logits + tf.random.normal((num_samples, 1))
        child2_logits = parent_logits + tf.random.normal((num_samples, 1))

        logits = tf.concat([parent_logits, child1_logits, child2_logits], axis=1)

        # Get predictions
        regular_preds = self.regular_head(logits)
        coupled_preds = self.coupled_head(logits)

        # Calculate metrics
        metrics = {
            'regular_hierarchy_violation': np.mean(
                (regular_preds[:, 0] < regular_preds[:, 1]) |
                (regular_preds[:, 0] < regular_preds[:, 2])
            ),
            'coupled_hierarchy_violation': np.mean(
                (coupled_preds[:, 0] < coupled_preds[:, 1]) |
                (coupled_preds[:, 0] < coupled_preds[:, 2])
            )
        }

        return ExperimentResults(regular_preds, coupled_preds, metrics)

    def run_experiment_3_confidence_calibration(
            self,
            num_samples: int = 1000
    ) -> ExperimentResults:
        """
        Experiment 3: Confidence Calibration

        Tests how well the model calibrates confidence across multiple labels.
        Example: When high confidence in one prediction should reduce confidence in others.
        """
        # Generate logits with varying confidence levels
        base_logits = tf.random.normal((num_samples, 4))
        high_conf_idx = tf.random.uniform((num_samples,), 0, 4, dtype=tf.int32)

        # Create one high confidence prediction per sample
        confidence_boost = tf.zeros_like(base_logits)
        confidence_boost = tf.tensor_scatter_nd_update(
            confidence_boost,
            tf.stack([tf.range(num_samples), high_conf_idx], axis=1),
            tf.ones(num_samples) * 5.0
        )

        logits = base_logits + confidence_boost

        # Get predictions
        regular_preds = self.regular_head(logits)
        coupled_preds = self.coupled_head(logits)

        # Calculate metrics
        metrics = {
            'regular_mean_others': np.mean(regular_preds[
                                               np.arange(num_samples)[:, None] != high_conf_idx[:, None]
                                               ]),
            'coupled_mean_others': np.mean(coupled_preds[
                                               np.arange(num_samples)[:, None] != high_conf_idx[:, None]
                                               ]),
            'regular_multi_high_conf': np.mean(np.sum(regular_preds > 0.9, axis=1)),
            'coupled_multi_high_conf': np.mean(np.sum(coupled_preds > 0.9, axis=1))
        }

        return ExperimentResults(regular_preds, coupled_preds, metrics)


def run_all_experiments() -> None:
    """Run and display results for all experiments."""
    runner = ExperimentRunner(coupling_strength=1.5)

    # Experiment 1: Mutually Exclusive Labels
    print("\nExperiment 1: Mutually Exclusive Labels")
    print("-" * 50)
    results1 = runner.run_experiment_1_mutually_exclusive_labels()
    print(f"Regular % samples with multiple active: {results1.metrics['regular_multi_active'] * 100:.2f}%")
    print(f"Coupled % samples with multiple active: {results1.metrics['coupled_multi_active'] * 100:.2f}%")
    print(f"Regular mean confidence: {results1.metrics['regular_mean_conf']:.3f}")
    print(f"Coupled mean confidence: {results1.metrics['coupled_mean_conf']:.3f}")

    # Experiment 2: Hierarchical Labels
    print("\nExperiment 2: Hierarchical Labels")
    print("-" * 50)
    results2 = runner.run_experiment_2_hierarchical_labels()
    print(f"Regular hierarchy violations: {results2.metrics['regular_hierarchy_violation'] * 100:.2f}%")
    print(f"Coupled hierarchy violations: {results2.metrics['coupled_hierarchy_violation'] * 100:.2f}%")

    # Experiment 3: Confidence Calibration
    print("\nExperiment 3: Confidence Calibration")
    print("-" * 50)
    results3 = runner.run_experiment_3_confidence_calibration()
    print(f"Regular mean confidence in other labels: {results3.metrics['regular_mean_others']:.3f}")
    print(f"Coupled mean confidence in other labels: {results3.metrics['coupled_mean_others']:.3f}")
    print(f"Regular avg number of high confidence preds: {results3.metrics['regular_multi_high_conf']:.2f}")
    print(f"Coupled avg number of high confidence preds: {results3.metrics['coupled_multi_high_conf']:.2f}")


if __name__ == "__main__":
    run_all_experiments()