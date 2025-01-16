"""
CoupledLogitNorm Multi-Label Classification Experiments
====================================================

Overview
--------
This experimental framework evaluates a novel approach to multi-label classification
using coupled logit normalization. The core innovation introduces deliberate coupling
between label predictions through a normalization mechanism, creating a form of
"confidence budget" across labels. This aims to improve prediction quality in scenarios
with complex label relationships.

Architecture Components
----------------------
1. CoupledLogitNorm Layer
   - Custom activation layer implementing coupled normalization
   - Introduces interdependencies between label predictions
   - Parameters:
     * coupling_strength: Controls interaction strength (0.8 default)
     * constant: Scaling factor for normalization (2.0 default)

2. Neural Network
   - Three-layer architecture (256 → 128 → 64)
   - ReLU activation in hidden layers
   - Batch normalization after each dense layer
   - Dropout (0.3) for regularization
   - CoupledLogitNorm activation in output layer

Experimental Design
------------------
Three major experiments testing different aspects of the coupling mechanism:

1. Mutual Exclusivity Test
   Data Generation:
   - 10,000 samples, 64 features, 3 classes
   - High noise level (0.5)
   - Significant feature overlap (0.7)
   - Close class centers (1.5)
   - Added complexity:
     * Non-linear feature transformations
     * Structured noise from other classes
     * Mixed random and structured noise
     * Weak activation leakage
   Purpose:
   - Test handling of mutually exclusive labels
   - Evaluate coupling effect on label competition
   - Measure confidence distribution

2. Hierarchical Labels Test
   Data Generation:
   - 10,000 samples, 64 features
   - Three-level hierarchy [2, 4, 8]
   - Parent-child relationships
   - Hierarchical noise structure
   Purpose:
   - Test preservation of hierarchical relationships
   - Evaluate parent-child prediction consistency
   - Measure hierarchy violation rates

3. Confidence Calibration
   Evaluation:
   - High confidence prediction analysis
   - Multi-label activation patterns
   - Zero-label case handling
   Purpose:
   - Assess confidence distribution
   - Evaluate prediction sparsity
   - Measure calibration quality

Evaluation Metrics
-----------------
1. Basic Classification Metrics:
   - Average precision
   - Micro/Macro AUC
   - Label cardinality
   - Label density

2. Multi-Label Specific:
   - Multi-label samples rate
   - Zero-label samples rate
   - Label correlation analysis

3. Hierarchical Metrics:
   - Hierarchy violation rate
   - Hierarchy compliance
   - Hierarchical accuracy
   - Level-wise performance

4. Data Complexity Metrics:
   - Linear separability
   - Feature correlations
   - Class distances
   - Decision boundary characteristics

Training Configuration
---------------------
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 30
- Validation split: 0.2
- Binary cross-entropy loss
- Early stopping patience: 5

Baselines
---------
1. Standard Neural Network:
   - Identical architecture
   - Regular sigmoid activation
   - No coupling mechanism

2. Performance Comparisons:
   - Mutual exclusivity handling
   - Hierarchy preservation
   - Confidence calibration
   - Prediction sparsity

Implementation Notes
-------------------
- TensorFlow 2.18.0
- Keras 3.8.0
- Custom layer implementation
- Type hints and documentation
- Reproducible random seeds
- Comprehensive logging
- Modular architecture

Future Extensions
----------------
1. Architecture Variations:
   - Dynamic coupling strength
   - Attention mechanisms
   - Residual connections
   - Feature disentanglement

2. Data Complexity:
   - Adversarial examples
   - Temporal dependencies
   - Multi-task scenarios
   - More complex hierarchies

3. Evaluation Extensions:
   - Robustness testing
   - Decision boundary visualization
   - Feature attribution
   - Confidence analysis
"""

import keras
import numpy as np
from keras import layers
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from sklearn.metrics import average_precision_score, roc_auc_score


from dl_techniques.layers.logit_norm import CoupledMultiLabelHead

@dataclass
class SyntheticData:
    """Container for synthetic dataset."""
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    label_names: List[str]
    hierarchy_matrix: Optional[np.ndarray] = None


def validate_data_difficulty(data: SyntheticData) -> Dict[str, float]:
    """
    Validate that the generated data is appropriately challenging.

    Args:
        data: Generated synthetic data

    Returns:
        Dictionary of difficulty metrics
    """
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score

    # Try linear classification
    clf = LinearSVC(dual="auto")
    clf.fit(data.x_train, np.argmax(data.y_train, axis=1))
    linear_pred = clf.predict(data.x_test)
    linear_acc = accuracy_score(np.argmax(data.y_test, axis=1), linear_pred)

    # Compute feature correlations
    feature_corr = np.corrcoef(data.x_train.T)
    avg_feature_corr = np.mean(np.abs(feature_corr - np.eye(feature_corr.shape[0])))

    # Compute class separability
    class_means = []
    for i in range(data.y_train.shape[1]):
        class_samples = data.x_train[data.y_train[:, i] == 1]
        class_means.append(np.mean(class_samples, axis=0))

    class_distances = []
    for i in range(len(class_means)):
        for j in range(i + 1, len(class_means)):
            dist = np.linalg.norm(class_means[i] - class_means[j])
            class_distances.append(dist)

    avg_class_distance = np.mean(class_distances)

    return {
        'linear_accuracy': linear_acc,
        'avg_feature_correlation': avg_feature_corr,
        'avg_class_distance': avg_class_distance
    }


class DataGenerator:
    """Generates synthetic data for experiments."""

    def generate_mutual_exclusive_data(
            self,
            num_samples: int = 10000,
            num_features: int = 64,
            num_classes: int = 3,
            noise_level: float = 0.5,  # Increased from 0.1
            feature_overlap: float = 0.7,  # New parameter for controlling separability
            center_proximity: float = 1.5,  # New parameter for controlling class overlap
    ) -> SyntheticData:
        """
        Generate challenging data for mutually exclusive labels with:
        - Higher noise levels
        - Reduced feature separability
        - Overlapping class distributions

        Args:
            num_samples: Number of samples to generate
            num_features: Dimension of input features
            num_classes: Number of classes
            noise_level: Amount of noise to add (higher = noisier)
            feature_overlap: Degree of feature sharing between classes (0-1)
            center_proximity: Distance between class centers (lower = more overlap)

        Returns:
            SyntheticData object containing train/test splits
        """
        # Generate class centers closer to each other
        base_centers = np.random.normal(0, center_proximity, (num_classes, num_features))

        # Create overlapping features between classes
        shared_features = np.random.normal(0, 1, (int(num_features * feature_overlap),))

        # Mix shared and unique features for each class
        centers = np.zeros((num_classes, num_features))
        for i in range(num_classes):
            # Determine which features are shared vs unique
            num_shared = int(num_features * feature_overlap)
            num_unique = num_features - num_shared

            # Mix shared and unique features
            centers[i, :num_shared] = shared_features + np.random.normal(0, 0.3, (num_shared,))
            centers[i, num_shared:] = base_centers[i, num_shared:]

            # Add random rotation to make classes less linearly separable
            rotation = np.random.randn(num_features, num_features)
            rotation = np.linalg.qr(rotation)[0]  # Orthogonal rotation matrix
            centers[i] = np.dot(rotation, centers[i])

        # Generate samples with increased complexity
        x = np.zeros((num_samples, num_features))
        y = np.zeros((num_samples, num_classes))

        for i in range(num_samples):
            # Select random class
            class_idx = np.random.randint(num_classes)

            # Generate base sample
            x[i] = centers[class_idx]

            # Add structured noise
            structured_noise = np.zeros(num_features)

            # Add influence from other classes
            for other_class in range(num_classes):
                if other_class != class_idx:
                    influence = np.random.normal(0, 0.3)
                    structured_noise += influence * centers[other_class]

            # Add random noise
            random_noise = np.random.normal(0, noise_level, num_features)

            # Combine noise components
            total_noise = 0.7 * structured_noise + 0.3 * random_noise
            x[i] += total_noise

            # Add label
            y[i, class_idx] = 1

            # Occasionally add weak activation for other classes
            if np.random.random() < 0.1:  # 10% chance
                other_class = np.random.choice([c for c in range(num_classes) if c != class_idx])
                x[i] += 0.3 * centers[other_class]

        # Add non-linear transformations to subset of features
        nonlinear_features = np.random.choice(num_features, size=int(num_features * 0.3), replace=False)
        for feat in nonlinear_features:
            x[:, feat] = np.sin(x[:, feat])

        # Normalize features
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

        # Split into train/test
        split_idx = int(0.8 * num_samples)
        return SyntheticData(
            x_train=x[:split_idx],
            y_train=y[:split_idx],
            x_test=x[split_idx:],
            y_test=y[split_idx:],
            label_names=[f"Class_{i}" for i in range(num_classes)]
        )

    def generate_hierarchical_data(
            self,
            num_samples: int = 10000,
            num_features: int = 64,
            hierarchy_levels: List[int] = [2, 3, 4],
            noise_level: float = 0.1
    ) -> SyntheticData:
        """
        Generate hierarchical label data.
        Labels follow a tree structure where parent activation implies child activation.

        Args:
            num_samples: Number of samples to generate
            num_features: Dimension of input features
            hierarchy_levels: Number of classes at each level
            noise_level: Amount of noise to add

        Returns:
            SyntheticData object containing train/test splits
        """
        total_classes = sum(hierarchy_levels)
        hierarchy_matrix = np.zeros((total_classes, total_classes))

        # Build hierarchy matrix
        current_idx = 0
        for level, num_classes in enumerate(hierarchy_levels[:-1]):
            next_level_start = sum(hierarchy_levels[:level + 1])
            classes_next_level = hierarchy_levels[level + 1]

            for i in range(num_classes):
                children_per_parent = classes_next_level // num_classes
                for j in range(children_per_parent):
                    child_idx = next_level_start + i * children_per_parent + j
                    hierarchy_matrix[current_idx + i, child_idx] = 1

            current_idx += num_classes

        # Generate features and labels
        x = np.zeros((num_samples, num_features))
        y = np.zeros((num_samples, total_classes))

        for i in range(num_samples):
            # Select random path through hierarchy
            current_level_idx = 0
            active_nodes = []

            for level, num_classes in enumerate(hierarchy_levels):
                if level == 0:
                    # Select root node
                    node_idx = np.random.randint(num_classes)
                else:
                    # Select child based on parent
                    parent_idx = active_nodes[-1]
                    possible_children = np.where(hierarchy_matrix[parent_idx] == 1)[0]
                    node_idx = np.random.choice(possible_children)

                active_nodes.append(node_idx)
                y[i, node_idx] = 1

            # Generate features
            x[i] = np.random.normal(0, 1, num_features)
            # Add signal for active nodes
            for node_idx in active_nodes:
                x[i] += np.random.normal(2, noise_level, num_features)

        # Split into train/test
        split_idx = int(0.8 * num_samples)
        return SyntheticData(
            x_train=x[:split_idx],
            y_train=y[:split_idx],
            x_test=x[split_idx:],
            y_test=y[split_idx:],
            label_names=[f"Level{i}_Class{j}" for i, n in enumerate(hierarchy_levels)
                         for j in range(n)],
            hierarchy_matrix=hierarchy_matrix
        )


def build_coupled_network(
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.2,
        coupling_strength: float = 0.8,
        constant: float = 2.0
) -> keras.Model:
    """
    Build a neural network with CoupledLogitNorm activation using Keras Functional API.

    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        coupling_strength: Coupling strength for CoupledLogitNorm
        constant: Scaling constant for CoupledLogitNorm

    Returns:
        Compiled Keras model
    """
    # Input layer
    inputs = keras.Input(shape=(input_dim,))
    x = inputs

    # Hidden layers
    for dim in hidden_dims:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    # Pre-activation dense layer
    logits = layers.Dense(num_classes, activation='linear')(x)

    # Coupled activation output
    outputs = CoupledMultiLabelHead(
        constant=constant,
        coupling_strength=coupling_strength
    )(logits)

    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='coupled_network')

    return model


class ExperimentMetrics:
    """Metrics for evaluating multi-label classification experiments."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        hierarchy_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute comprehensive metrics for evaluation."""
        metrics = {}

        # Basic metrics
        metrics['average_precision'] = average_precision_score(y_true, y_pred)

        # Handle potential single-class case for ROC AUC
        try:
            metrics['micro_auc'] = roc_auc_score(y_true, y_pred, average='micro')
            metrics['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')
        except ValueError:
            metrics['micro_auc'] = np.nan
            metrics['macro_auc'] = np.nan

        # Multi-label specific metrics
        metrics['label_cardinality'] = np.mean(np.sum(y_pred > 0.5, axis=1))
        metrics['label_density'] = metrics['label_cardinality'] / y_pred.shape[1]

        # Mutual exclusivity metrics
        metrics['multi_label_samples'] = np.mean(np.sum(y_pred > 0.5, axis=1) > 1)
        metrics['zero_label_samples'] = np.mean(np.sum(y_pred > 0.5, axis=1) == 0)

        # Hierarchical metrics
        if hierarchy_matrix is not None:
            total_violations = 0
            total_relations = 0

            # For each parent node
            for parent_idx in range(hierarchy_matrix.shape[0]):
                # Get indices of child nodes
                child_indices = np.where(hierarchy_matrix[parent_idx] == 1)[0]

                if len(child_indices) > 0:
                    parent_preds = y_pred[:, parent_idx][:, np.newaxis]  # Add dimension for broadcasting
                    child_preds = y_pred[:, child_indices]  # Shape: (n_samples, n_children)

                    # Count violations: parent score should be >= child scores
                    violations = np.sum(parent_preds < child_preds)
                    total_violations += violations
                    total_relations += len(child_indices) * y_pred.shape[0]

            if total_relations > 0:
                metrics['hierarchy_violation_rate'] = total_violations / total_relations
                metrics['hierarchy_compliance'] = 1.0 - metrics['hierarchy_violation_rate']

            # Add hierarchical accuracy
            correct_hierarchies = 0
            total_hierarchies = 0

            for i in range(y_true.shape[0]):
                for parent_idx in range(hierarchy_matrix.shape[0]):
                    child_indices = np.where(hierarchy_matrix[parent_idx] == 1)[0]
                    if len(child_indices) > 0:
                        total_hierarchies += 1
                        if all(y_true[i, parent_idx] >= y_true[i, child_idx] for child_idx in child_indices):
                            if all(y_pred[i, parent_idx] >= y_pred[i, child_idx] for child_idx in child_indices):
                                correct_hierarchies += 1

            if total_hierarchies > 0:
                metrics['hierarchical_accuracy'] = correct_hierarchies / total_hierarchies

        return metrics


def analyze_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                        hierarchy_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Detailed analysis of predictions including confusion matrices per level."""
    analysis = {}

    # Basic prediction analysis
    analysis['threshold'] = 0.5
    pred_labels = y_pred > analysis['threshold']

    # Per-class metrics
    analysis['per_class'] = {
        'precision': [],
        'recall': [],
        'f1': []
    }

    for i in range(y_true.shape[1]):
        true_pos = np.sum((y_true[:, i] == 1) & (pred_labels[:, i] == 1))
        false_pos = np.sum((y_true[:, i] == 0) & (pred_labels[:, i] == 1))
        false_neg = np.sum((y_true[:, i] == 1) & (pred_labels[:, i] == 0))

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        analysis['per_class']['precision'].append(precision)
        analysis['per_class']['recall'].append(recall)
        analysis['per_class']['f1'].append(f1)

    # Hierarchical analysis
    if hierarchy_matrix is not None:
        analysis['hierarchy'] = {
            'level_metrics': {},
            'parent_child_correlations': []
        }

        # Calculate level-wise metrics
        current_level = 0
        nodes_in_level = [i for i in range(hierarchy_matrix.shape[0])
                          if not any(hierarchy_matrix[:, i])]  # Find root nodes

        while nodes_in_level:
            level_true = y_true[:, nodes_in_level]
            level_pred = y_pred[:, nodes_in_level]

            analysis['hierarchy']['level_metrics'][f'level_{current_level}'] = {
                'accuracy': np.mean((level_pred > 0.5) == level_true),
                'mean_confidence': np.mean(level_pred),
                'num_nodes': len(nodes_in_level)
            }

            # Find nodes in next level
            next_level = []
            for node in nodes_in_level:
                children = np.where(hierarchy_matrix[node] == 1)[0]
                next_level.extend(children)

            nodes_in_level = next_level
            current_level += 1

            # Calculate parent-child prediction correlations
            for parent_idx in range(hierarchy_matrix.shape[0]):
                child_indices = np.where(hierarchy_matrix[parent_idx] == 1)[0]
                if len(child_indices) > 0:
                    parent_preds = y_pred[:, parent_idx]
                    for child_idx in child_indices:
                        child_preds = y_pred[:, child_idx]
                        correlation = np.corrcoef(parent_preds, child_preds)[0, 1]
                        analysis['hierarchy']['parent_child_correlations'].append({
                            'parent': parent_idx,
                            'child': child_idx,
                            'correlation': correlation
                        })

    return analysis


def train_and_evaluate(
        data: SyntheticData,
        model_params: Dict,
        training_params: Dict
) -> Tuple[keras.Model, Dict[str, float]]:
    """Train and evaluate the coupled network."""

    model = \
        build_coupled_network(
            input_dim=data.x_train.shape[1],
            num_classes=data.y_train.shape[1],
            **model_params
        )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=training_params.get('learning_rate', 0.001)),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.AUC()
        ]
    )

    # Train model
    history = model.fit(
        data.x_train, data.y_train,
        validation_data=(data.x_test, data.y_test),
        **training_params
    )

    # Evaluate
    y_pred = model.predict(data.x_test)
    metrics = ExperimentMetrics.compute_metrics(
        data.y_test, y_pred, data.hierarchy_matrix
    )

    return model, metrics


def run_training_experiments():
    """Run training experiments with synthetic data."""
    # Initialize data generator
    data_gen = DataGenerator()

    # Experiment 1: Mutual Exclusivity
    print("\nExperiment 1: Mutual Exclusivity")
    print("=" * 50)

    data_mutual = data_gen.generate_mutual_exclusive_data(
        num_samples=10000,
        num_features=64,
        num_classes=3,
        noise_level=0.1
    )

    model_params = {
        'hidden_dims': [128, 64],
        'dropout_rate': 0.2,
        'coupling_strength': 0.8,
        'constant': 2.0
    }

    training_params = {
        'batch_size': 64,
        'epochs': 20,
        'validation_split': 0.2,
        'verbose': 1
    }

    model_mutual, metrics_mutual = train_and_evaluate(
        data_mutual, model_params, training_params
    )

    print("\nMutual Exclusivity Results:")
    for metric, value in metrics_mutual.items():
        print(f"{metric}: {value:.4f}")

    # Experiment 2: Hierarchical Labels
    print("\nExperiment 2: Hierarchical Labels")
    print("=" * 50)

    data_hierarch = data_gen.generate_hierarchical_data(
        num_samples=10000,
        num_features=64,
        hierarchy_levels=[2, 4, 8],
        noise_level=0.1
    )

    model_params['coupling_strength'] = 1.0  # Adjusted for hierarchical

    model_hierarch, metrics_hierarch = train_and_evaluate(
        data_hierarch, model_params, training_params
    )

    print("\nHierarchical Results:")
    for metric, value in metrics_hierarch.items():
        print(f"{metric}: {value:.4f}")

    # Compare with baseline (uncoupled) models
    model_params['coupling_strength'] = 0.0

    print("\nBaseline (Uncoupled) Results:")
    _, metrics_baseline_mutual = train_and_evaluate(
        data_mutual, model_params, training_params
    )
    _, metrics_baseline_hierarch = train_and_evaluate(
        data_hierarch, model_params, training_params
    )

    print("\nMutual Exclusivity Baseline:")
    for metric, value in metrics_baseline_mutual.items():
        print(f"{metric}: {value:.4f}")

    print("\nHierarchical Baseline:")
    for metric, value in metrics_baseline_hierarch.items():
        print(f"{metric}: {value:.4f}")

    return {
        'mutual': {
            'coupled': metrics_mutual,
            'baseline': metrics_baseline_mutual
        },
        'hierarchical': {
            'coupled': metrics_hierarch,
            'baseline': metrics_baseline_hierarch
        }
    }


if __name__ == "__main__":
    results = run_training_experiments()