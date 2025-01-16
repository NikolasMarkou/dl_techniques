"""
# CoupledLogitNorm Experiments

## Overview
This experimental framework evaluates a novel approach to multi-label classification using coupled logit normalization. The core innovation lies in creating deliberate interdependencies between label predictions through a normalization-based coupling mechanism. This approach introduces a form of "confidence budget" across labels, potentially improving prediction quality in scenarios with complex label relationships.

## Technical Implementation
The implementation consists of two main components:

1. `CoupledLogitNorm`: A custom layer that implements coupled logit normalization
2. `CoupledMultiLabelHead`: A classification head that combines `CoupledLogitNorm` with sigmoid activation

The coupling mechanism is controlled by two key parameters:
- `constant`: Scaling factor for normalization (higher values reduce coupling)
- `coupling_strength`: Additional factor to control coupling intensity (1.0 = standard LogitNorm)

## Experimental Design

The framework includes three experiments designed to evaluate different aspects of the coupled normalization approach:

### Experiment 1: Mutually Exclusive Labels
Tests the model's ability to handle mutually exclusive label scenarios, similar to medical diagnosis where certain conditions cannot co-occur.

**Methodology:**
- Generates synthetic logits with strong signals
- Compares regular vs. coupled predictions
- Metrics focus on multi-label activation rates and confidence levels

**Expected Benefits:**
- Reduced simultaneous activation of mutually exclusive labels
- More appropriate confidence distribution across labels

### Experiment 2: Hierarchical Labels
Evaluates the handling of hierarchical label relationships, similar to nested classification scenarios (e.g., "vehicle" → "car" → "sports car").

**Methodology:**
- Generates hierarchically structured logits
- Parent logits influence child logits
- Tracks hierarchy violation rates

**Expected Benefits:**
- Better preservation of hierarchical relationships
- Reduced likelihood of logical contradictions in predictions

### Experiment 3: Confidence Calibration
Assesses the model's ability to properly calibrate confidence across multiple labels when strong evidence exists for specific predictions.

**Methodology:**
- Creates scenarios with varying confidence levels
- Introduces high-confidence signals for specific labels
- Measures impact on related predictions

**Expected Benefits:**
- Better distribution of confidence across labels
- Reduced occurrence of multiple high-confidence predictions
- More realistic confidence assignments

## Metrics and Evaluation

Each experiment tracks specific metrics:

1. **Mutual Exclusivity Metrics:**
   - Percentage of samples with multiple active labels
   - Mean confidence levels

2. **Hierarchical Metrics:**
   - Rate of hierarchy violations
   - Parent-child relationship consistency

3. **Calibration Metrics:**
   - Mean confidence in secondary predictions
   - Frequency of multiple high-confidence predictions

## Implementation Notes

- Uses TensorFlow 2.18.0 and Keras 3.8.0
- Implements custom Keras layers with proper serialization
- Ensures numerical stability through epsilon values
- Maintains compatibility with training and inference modes

## Potential Applications

1. **Medical Diagnosis:**
   - Handling mutually exclusive conditions
   - Representing diagnostic hierarchies
   - Confidence calibration in predictions

2. **Computer Vision:**
   - Hierarchical object classification
   - Multi-label scene understanding
   - Attribute prediction with dependencies

3. **Natural Language Processing:**
   - Topic classification with hierarchies
   - Multi-label text classification
   - Sentiment analysis with multiple aspects

## Limitations and Considerations

- Coupling strength needs careful tuning per application
- May require additional computational resources
- Performance impact on very large label spaces
- Potential interaction with other regularization techniques

## Future Research Directions

1. **Adaptive Coupling:**
   - Dynamic adjustment of coupling strength
   - Learning optimal coupling parameters
   - Context-dependent coupling mechanisms

2. **Scalability:**
   - Efficient implementation for large label spaces
   - Sparse computation strategies
   - Batch-wise coupling approaches

3. **Integration:**
   - Combination with other normalization techniques
   - Integration with attention mechanisms
   - Extension to other neural architectures
"""

import numpy as np
import tensorflow as tf
from keras.api import layers
from dataclasses import dataclass
from typing import Tuple, List, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.logit_norm import CoupledMultiLabelHead

# ---------------------------------------------------------------------


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    regular_predictions: np.ndarray
    coupled_predictions: np.ndarray
    metrics: Dict[str, float]

# ---------------------------------------------------------------------


class ExperimentRunner:
    """Runner for LogitNorm experiments."""

    def __init__(
            self,
            coupling_strength: float = 1.0,
            constant: float = 1.0
    ):
        """
        Initialize the experiment runner.

        Args:
            coupling_strength (float): Strength of coupling between labels
            constant (float): Scaling factor for normalization
        """
        self.coupling_strength: float = coupling_strength
        self.constant: float = constant

    def run_experiment_1_mutually_exclusive_labels(
            self,
            num_samples: int = 10000
    ) -> ExperimentResults:
        """
        Experiment 1: Mutually Exclusive Labels

        Tests how well the model handles cases where labels should be mutually exclusive.
        Example: Medical diagnosis where certain conditions cannot co-occur.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            ExperimentResults: Container with predictions and metrics
        """
        num_classes = 3  # Fixed for this experiment
        # Generate synthetic logits that should produce mutually exclusive predictions
        logits = tf.random.normal((num_samples, num_classes)) * 5  # Scaled for stronger signals

        # Get predictions
        regular_head = layers.Dense(num_classes, activation='sigmoid')
        coupled_head = CoupledMultiLabelHead(
            constant=self.constant,
            coupling_strength=self.coupling_strength
        )
        regular_preds = regular_head(logits)
        coupled_preds = coupled_head(logits)

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
            num_samples: int = 10000
    ) -> ExperimentResults:
        """
        Experiment 2: Hierarchical Labels

        Tests handling of hierarchical label relationships.
        Example: Image classification where "vehicle" → "car" → "sports car"

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            ExperimentResults: Container with predictions and metrics
        """
        num_classes = 3  # Fixed for this experiment
        # Generate hierarchical logits (parent should influence children)
        parent_logits = tf.random.normal((num_samples, 1)) * 2
        child1_logits = parent_logits + tf.random.normal((num_samples, 1))
        child2_logits = parent_logits + tf.random.normal((num_samples, 1))

        logits = tf.concat([parent_logits, child1_logits, child2_logits], axis=1)

        # Get predictions
        regular_head = layers.Dense(num_classes, activation='sigmoid')
        coupled_head = CoupledMultiLabelHead(
            constant=self.constant,
            coupling_strength=self.coupling_strength
        )
        regular_preds = regular_head(logits)
        coupled_preds = coupled_head(logits)

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
            num_samples: int = 10000
    ) -> ExperimentResults:
        """
        Experiment 3: Confidence Calibration

        Tests how well the model calibrates confidence across multiple labels.
        Example: When high confidence in one prediction should reduce confidence in others.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            ExperimentResults: Container with predictions and metrics
        """
        num_classes = 4  # Fixed for this experiment
        # Generate logits with varying confidence levels
        base_logits = tf.random.normal((num_samples, num_classes))
        high_conf_idx = tf.random.uniform((num_samples,), 0, num_classes, dtype=tf.int32)

        # Create one high confidence prediction per sample
        confidence_boost = tf.zeros_like(base_logits)
        confidence_boost = tf.tensor_scatter_nd_update(
            confidence_boost,
            tf.stack([tf.range(num_samples), high_conf_idx], axis=1),
            tf.ones(num_samples) * 5.0
        )

        logits = base_logits + confidence_boost

        # Get predictions
        regular_head = layers.Dense(num_classes, activation='sigmoid')
        coupled_head = CoupledMultiLabelHead(
            constant=self.constant,
            coupling_strength=self.coupling_strength
        )
        regular_preds = regular_head(logits)
        coupled_preds = coupled_head(logits)

        # Create boolean mask for non-high-confidence predictions
        idx_mask = tf.cast(tf.ones_like(regular_preds), dtype=tf.bool)
        idx_range = tf.range(num_samples)
        idx_mask = tf.tensor_scatter_nd_update(
            idx_mask,
            tf.stack([idx_range, high_conf_idx], axis=1),
            tf.zeros(num_samples, dtype=tf.bool)
        )

        # Calculate metrics
        metrics = {
            'regular_mean_others': tf.reduce_mean(tf.boolean_mask(regular_preds, idx_mask)).numpy(),
            'coupled_mean_others': tf.reduce_mean(tf.boolean_mask(coupled_preds, idx_mask)).numpy(),
            'regular_multi_high_conf': np.mean(np.sum(regular_preds > 0.9, axis=1)),
            'coupled_multi_high_conf': np.mean(np.sum(coupled_preds > 0.9, axis=1))
        }

        return ExperimentResults(regular_preds, coupled_preds, metrics)

def run_all_experiments() -> None:
    """Run and display results for all experiments."""
    runner = ExperimentRunner(coupling_strength=1.5)

    # Experiment 1: Mutually Exclusive Labels
    logger.info("==================================================================================")
    logger.info("Experiment 1: Mutually Exclusive Labels")
    logger.info("-" * 50)
    results1 = runner.run_experiment_1_mutually_exclusive_labels()
    logger.info(f"Regular % samples with multiple active: {results1.metrics['regular_multi_active'] * 100:.2f}%")
    logger.info(f"Coupled % samples with multiple active: {results1.metrics['coupled_multi_active'] * 100:.2f}%")
    logger.info(f"Regular mean confidence: {results1.metrics['regular_mean_conf']:.3f}")
    logger.info(f"Coupled mean confidence: {results1.metrics['coupled_mean_conf']:.3f}")

    # Experiment 2: Hierarchical Labels
    logger.info("==================================================================================")
    logger.info("Experiment 2: Hierarchical Labels")
    logger.info("-" * 50)
    results2 = runner.run_experiment_2_hierarchical_labels()
    logger.info(f"Regular hierarchy violations: {results2.metrics['regular_hierarchy_violation'] * 100:.2f}%")
    logger.info(f"Coupled hierarchy violations: {results2.metrics['coupled_hierarchy_violation'] * 100:.2f}%")

    # Experiment 3: Confidence Calibration
    logger.info("==================================================================================")
    logger.info("Experiment 3: Confidence Calibration")
    logger.info("-" * 50)
    results3 = runner.run_experiment_3_confidence_calibration()
    logger.info(f"Regular mean confidence in other labels: {results3.metrics['regular_mean_others']:.3f}")
    logger.info(f"Coupled mean confidence in other labels: {results3.metrics['coupled_mean_others']:.3f}")
    logger.info(f"Regular avg number of high confidence preds: {results3.metrics['regular_multi_high_conf']:.2f}")
    logger.info(f"Coupled avg number of high confidence preds: {results3.metrics['coupled_multi_high_conf']:.2f}")


if __name__ == "__main__":
    run_all_experiments()
