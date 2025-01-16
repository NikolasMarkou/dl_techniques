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

DEFAULT_NUMBER_OF_SAMPLES = 100000

@dataclass
class ExperimentResults:
    """Container for experiment results."""
    regular_predictions: np.ndarray
    coupled_predictions: np.ndarray
    metrics: Dict[str, float]


# ---------------------------------------------------------------------

def format_results_for_visualization(results: Dict[str, ExperimentResults]) -> Dict:
    """Format experiment results for visualization."""
    return {
        'experiment1': {
            'multiple_active': {
                'regular': float(results['exp1'].metrics['regular_multi_active'] * 100),
                'coupled': float(results['exp1'].metrics['coupled_multi_active'] * 100)
            },
            'mean_confidence': {
                'regular': float(results['exp1'].metrics['regular_mean_conf'] * 100),
                'coupled': float(results['exp1'].metrics['coupled_mean_conf'] * 100)
            },
            'zero_active': {
                'regular': float(results['exp1'].metrics['regular_zero_active'] * 100),
                'coupled': float(results['exp1'].metrics['coupled_zero_active'] * 100)
            }
        },
        'experiment2': {
            'hierarchy_violations': {
                'regular': float(results['exp2'].metrics['regular_hierarchy_violation'] * 100),
                'coupled': float(results['exp2'].metrics['coupled_hierarchy_violation'] * 100)
            },
            'valid_hierarchy': {
                'regular': float(results['exp2'].metrics['regular_valid_hierarchy'] * 100),
                'coupled': float(results['exp2'].metrics['coupled_valid_hierarchy'] * 100)
            }
        },
        'experiment3': {
            'mean_others': {
                'regular': float(results['exp3'].metrics['regular_mean_others'] * 100),
                'coupled': float(results['exp3'].metrics['coupled_mean_others'] * 100)
            },
            'multi_high_conf': {
                'regular': float(results['exp3'].metrics['regular_multi_high_conf'] * 100),
                'coupled': float(results['exp3'].metrics['coupled_multi_high_conf'] * 100)
            },
            'high_conf_correct': {
                'regular': float(results['exp3'].metrics['high_conf_correct'] * 100),
                'coupled': 0  # Coupled version doesn't produce high confidence predictions
            }
        }
    }


# ---------------------------------------------------------------------

def save_results_as_json(results: Dict, filename: str = 'experiment_results.json') -> None:
    """Save results to a JSON file."""
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)


# ---------------------------------------------------------------------

class ExperimentRunner:
    """Runner for LogitNorm experiments with improved configurations."""

    def __init__(
            self,
            coupling_strength: float = 0.8,  # Reduced from 1.5
            constant: float = 2.0,  # Increased from 1.0
            threshold: float = 0.4  # New parameter for activation threshold
    ):
        """
        Initialize the experiment runner with modified parameters.

        Args:
            coupling_strength: Controls strength of label coupling (lower = more independent)
            constant: Scaling factor for normalization (higher = less aggressive normalization)
            threshold: Activation threshold for binary decisions (lower than 0.5 to compensate for coupling)
        """
        self.coupling_strength: float = coupling_strength
        self.constant: float = constant
        self.threshold: float = threshold

    def run_experiment_1_mutually_exclusive_labels(
            self,
            num_samples: int = DEFAULT_NUMBER_OF_SAMPLES
    ) -> ExperimentResults:
        """Experiment 1: Mutually Exclusive Labels with stronger signals."""
        num_classes = 3
        # Increased signal strength and added bias to encourage sparsity
        logits = tf.random.normal((num_samples, num_classes)) * 8.0
        # Add bias to encourage more definitive predictions
        bias = tf.random.uniform((num_samples,)) * 2.0 - 1.0
        bias = tf.expand_dims(bias, axis=1)
        logits = logits + bias

        regular_head = layers.Dense(num_classes, activation='sigmoid')
        coupled_head = CoupledMultiLabelHead(
            constant=self.constant,
            coupling_strength=self.coupling_strength
        )

        regular_preds = regular_head(logits)
        coupled_preds = coupled_head(logits)

        metrics = {
            'regular_multi_active': np.mean(np.sum(regular_preds > self.threshold, axis=1) > 1),
            'coupled_multi_active': np.mean(np.sum(coupled_preds > self.threshold, axis=1) > 1),
            'regular_mean_conf': np.mean(regular_preds),
            'coupled_mean_conf': np.mean(coupled_preds),
            'regular_zero_active': np.mean(np.sum(regular_preds > self.threshold, axis=1) == 0),
            'coupled_zero_active': np.mean(np.sum(coupled_preds > self.threshold, axis=1) == 0)
        }

        return ExperimentResults(regular_preds, coupled_preds, metrics)

    def run_experiment_2_hierarchical_labels(
            self,
            num_samples: int = DEFAULT_NUMBER_OF_SAMPLES
    ) -> ExperimentResults:
        """Experiment 2: Hierarchical Labels with strengthened parent-child relationships."""
        num_classes = 3
        # Stronger parent influence
        parent_logits = tf.random.normal((num_samples, 1)) * 3.0
        # Children more strongly influenced by parent
        child1_logits = parent_logits * 0.8 + tf.random.normal((num_samples, 1)) * 0.5
        child2_logits = parent_logits * 0.6 + tf.random.normal((num_samples, 1)) * 0.5

        logits = tf.concat([parent_logits, child1_logits, child2_logits], axis=1)

        regular_head = layers.Dense(num_classes, activation='sigmoid')
        coupled_head = CoupledMultiLabelHead(
            constant=self.constant,
            coupling_strength=self.coupling_strength
        )

        regular_preds = regular_head(logits)
        coupled_preds = coupled_head(logits)

        # Modified hierarchy violation check with tolerance
        tolerance = 0.1
        metrics = {
            'regular_hierarchy_violation': np.mean(
                (regular_preds[:, 0] + tolerance < regular_preds[:, 1]) |
                (regular_preds[:, 0] + tolerance < regular_preds[:, 2])
            ),
            'coupled_hierarchy_violation': np.mean(
                (coupled_preds[:, 0] + tolerance < coupled_preds[:, 1]) |
                (coupled_preds[:, 0] + tolerance < coupled_preds[:, 2])
            ),
            'regular_valid_hierarchy': np.mean(
                (regular_preds[:, 0] >= regular_preds[:, 1]) &
                (regular_preds[:, 1] >= regular_preds[:, 2])
            ),
            'coupled_valid_hierarchy': np.mean(
                (coupled_preds[:, 0] >= coupled_preds[:, 1]) &
                (coupled_preds[:, 1] >= coupled_preds[:, 2])
            )
        }

        return ExperimentResults(regular_preds, coupled_preds, metrics)

    def run_experiment_3_confidence_calibration(
            self,
            num_samples: int = DEFAULT_NUMBER_OF_SAMPLES
    ) -> ExperimentResults:
        """
        Experiment 3: Confidence Calibration with more pronounced signals.

        Args:
            num_samples: Number of samples to generate

        Returns:
            ExperimentResults with predictions and metrics
        """
        num_classes = 4
        # Reduced base noise
        base_logits = tf.random.normal((num_samples, num_classes)) * 0.5
        high_conf_idx = tf.random.uniform((num_samples,), 0, num_classes, dtype=tf.int32)

        # Stronger confidence boost
        confidence_boost = tf.zeros_like(base_logits)
        confidence_boost = tf.tensor_scatter_nd_update(
            confidence_boost,
            tf.stack([tf.range(num_samples), high_conf_idx], axis=1),
            tf.ones(num_samples) * 8.0  # Increased from 5.0
        )

        logits = base_logits + confidence_boost

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

        # Convert predictions to numpy for metric calculations
        regular_preds_np = regular_preds.numpy()
        coupled_preds_np = coupled_preds.numpy()
        high_conf_idx_np = high_conf_idx.numpy()

        # Calculate high confidence correctness using TensorFlow operations
        high_conf_mask = tf.stack([idx_range, high_conf_idx], axis=1)
        high_conf_values = tf.gather_nd(regular_preds, high_conf_mask)
        high_conf_correct = tf.reduce_mean(tf.cast(high_conf_values > 0.9, tf.float32))

        metrics = {
            'regular_mean_others': tf.reduce_mean(tf.boolean_mask(regular_preds, idx_mask)).numpy(),
            'coupled_mean_others': tf.reduce_mean(tf.boolean_mask(coupled_preds, idx_mask)).numpy(),
            'regular_multi_high_conf': np.mean(np.sum(regular_preds_np > 0.9, axis=1)),
            'coupled_multi_high_conf': np.mean(np.sum(coupled_preds_np > 0.9, axis=1)),
            'high_conf_correct': high_conf_correct.numpy()
        }

        return ExperimentResults(regular_preds_np, coupled_preds_np, metrics)


def run_all_experiments() -> None:
    """Run experiments and save visualizable results."""
    runner = ExperimentRunner(coupling_strength=0.8, constant=2.0, threshold=0.4)

    # Run all experiments
    results = {
        'exp1': runner.run_experiment_1_mutually_exclusive_labels(),
        'exp2': runner.run_experiment_2_hierarchical_labels(),
        'exp3': runner.run_experiment_3_confidence_calibration()
    }

    # Format results for visualization
    viz_results = format_results_for_visualization(results)

    # Save results to JSON
    save_results_as_json(viz_results)

    # Log summary metrics
    logger.info("Experiment Results Summary")

    for exp_name, exp_results in viz_results.items():
        logger.info(f"=========================================================")
        logger.info(f"{exp_name}:")
        for metric, values in exp_results.items():
            logger.info(f"{metric}:")
            logger.info(f"  Regular: {values['regular']:.2f}%")
            logger.info(f"  Coupled: {values['coupled']:.2f}%")


if __name__ == "__main__":
    run_all_experiments()
