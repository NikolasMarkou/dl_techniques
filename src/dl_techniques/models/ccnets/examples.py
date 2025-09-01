"""
CCNets Usage Example - Complete Guide

This example demonstrates how to use the refined CCNets implementation
for explainable cooperative learning on a classification task.
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Tuple
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .models import create_ccnets_model, CCNetsModel

# ---------------------------------------------------------------------

def create_synthetic_dataset(
    n_samples: int = 1000,
    input_dim: int = 20,
    n_classes: int = 5,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for CCNets demonstration.

    Args:
        n_samples: Number of samples to generate
        input_dim: Dimensionality of input features
        n_classes: Number of classes
        noise_level: Amount of noise to add

    Returns:
        Tuple of (X, y) where X is input data and y is one-hot labels
    """
    np.random.seed(42)

    # Create class centers
    centers = np.random.randn(n_classes, input_dim) * 2

    # Generate samples
    X = []
    y = []

    for i in range(n_samples):
        # Choose random class
        class_idx = np.random.randint(n_classes)

        # Generate sample near class center with noise
        sample = centers[class_idx] + np.random.randn(input_dim) * noise_level

        X.append(sample)
        y.append(class_idx)

    X = np.array(X)
    y = keras.utils.to_categorical(y, n_classes)

    return X.astype(np.float32), y.astype(np.float32)


def demonstrate_basic_usage():
    """Demonstrate basic CCNets usage."""
    print("=== CCNets Basic Usage Example ===\n")

    # 1. Create synthetic dataset
    print("1. Creating synthetic dataset...")
    X_train, y_train = create_synthetic_dataset(
        n_samples=1000,
        input_dim=20,
        n_classes=5
    )
    X_test, y_test = create_synthetic_dataset(
        n_samples=200,
        input_dim=20,
        n_classes=5
    )

    print(f"Training data: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")

    # 2. Create CCNets model using factory function
    print("\n2. Creating CCNets model...")
    model = create_ccnets_model(
        input_dim=20,           # Input feature dimension
        explanation_dim=8,      # Compressed explanation dimension
        output_dim=5,           # Number of classes
        explainer_kwargs={
            'hidden_dims': [64, 32],
            'dropout_rate': 0.2,
            'activation': 'relu'
        },
        reasoner_kwargs={
            'hidden_dims': [64, 32],
            'dropout_rate': 0.2,
            'fusion_dim': 256
        },
        producer_kwargs={
            'hidden_dims': [32, 64],
            'dropout_rate': 0.2,
            'output_activation': 'sigmoid'
        },
        loss_weights=[1.0, 1.0, 1.0]  # Equal weighting
    )

    # 3. Compile model
    print("3. Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        run_eagerly=True  # For debugging - remove for performance
    )

    # 4. Train model
    print("4. Training model...")

    # Custom training loop to show cooperative learning
    batch_size = 32
    epochs = 10

    # Convert to tf.data.Dataset for efficient training
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size).shuffle(1000)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    # Training loop with monitoring
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training
        epoch_losses = []
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            # Prepare data in CCNets format: ([observations, labels], targets)
            data = ((x_batch, y_batch), None)

            # Training step
            metrics = model.train_step(data)
            epoch_losses.append(metrics)

            if step % 10 == 0:
                print(f"  Step {step}: "
                      f"total_loss={metrics['loss']:.4f}, "
                      f"inf_loss={metrics['inference_loss']:.4f}, "
                      f"gen_loss={metrics['generation_loss']:.4f}, "
                      f"rec_loss={metrics['reconstruction_loss']:.4f}")

        # Evaluation
        eval_losses = []
        for x_batch, y_batch in test_dataset:
            data = ((x_batch, y_batch), None)
            eval_metrics = model.test_step(data)
            eval_losses.append(eval_metrics)

        # Average metrics
        avg_eval_loss = np.mean([m['loss'] for m in eval_losses])
        print(f"  Validation loss: {avg_eval_loss:.4f}")

    return model, X_test, y_test


def demonstrate_prediction_and_explanation(
    model: CCNetsModel,
    X_test: np.ndarray,
    y_test: np.ndarray
):
    """Demonstrate prediction and explanation capabilities."""
    print("\n=== Prediction and Explanation ===\n")

    # 1. Make predictions
    print("1. Making predictions...")
    predictions = model.predict_step(X_test[:10])

    predicted_classes = np.argmax(predictions['predictions'], axis=1)
    true_classes = np.argmax(y_test[:10], axis=1)
    explanations = predictions['explanations']
    reconstructions = predictions['reconstructions']

    print(f"Predictions shape: {predictions['predictions'].shape}")
    print(f"Explanations shape: {explanations.shape}")
    print(f"Reconstructions shape: {reconstructions.shape}")

    # 2. Analyze predictions
    print("\n2. Prediction Analysis:")
    for i in range(5):
        print(f"  Sample {i}: True={true_classes[i]}, "
              f"Predicted={predicted_classes[i]}, "
              f"Explanation norm={np.linalg.norm(explanations[i]):.3f}")

    # 3. Reconstruction quality
    print("\n3. Reconstruction Quality:")
    reconstruction_errors = np.mean(np.abs(X_test[:10] - reconstructions), axis=1)
    print(f"Average reconstruction error: {np.mean(reconstruction_errors):.4f}")
    print(f"Std reconstruction error: {np.std(reconstruction_errors):.4f}")


def demonstrate_cooperative_analysis(
    model: CCNetsModel,
    X_sample: np.ndarray,
    y_sample: np.ndarray
):
    """Demonstrate analysis of cooperative behavior."""
    print("\n=== Cooperative Behavior Analysis ===\n")

    # 1. Forward pass to get all outputs
    sample_batch = X_sample[:5]
    label_batch = y_sample[:5]

    outputs = model([sample_batch, label_batch], training=False)
    losses = model.compute_losses(sample_batch, label_batch, outputs)

    print("1. Loss Components:")
    print(f"  Inference Loss: {losses['inference_loss']:.4f}")
    print(f"  Generation Loss: {losses['generation_loss']:.4f}")
    print(f"  Reconstruction Loss: {losses['reconstruction_loss']:.4f}")

    print("\n2. Network Errors (Cooperative Objectives):")
    print(f"  Explainer Error: {losses['explainer_error']:.4f}")
    print(f"  Reasoner Error: {losses['reasoner_error']:.4f}")
    print(f"  Producer Error: {losses['producer_error']:.4f}")

    # 2. Explanation analysis
    explanations = outputs['explanation_vector']
    print(f"\n3. Explanation Analysis:")
    print(f"  Explanation statistics:")
    print(f"    Mean: {np.mean(explanations):.4f}")
    print(f"    Std: {np.std(explanations):.4f}")
    print(f"    Min: {np.min(explanations):.4f}")
    print(f"    Max: {np.max(explanations):.4f}")

    # 3. Network cooperation assessment
    print(f"\n4. Cooperation Assessment:")

    # Lower values indicate better cooperation
    cooperation_score = (losses['inference_loss'] +
                        losses['generation_loss'] +
                        losses['reconstruction_loss']) / 3

    print(f"  Cooperation Score (lower is better): {cooperation_score:.4f}")

    # Balance between network errors
    error_balance = np.std([
        losses['explainer_error'],
        losses['reasoner_error'],
        losses['producer_error']
    ])
    print(f"  Error Balance (lower is more balanced): {error_balance:.4f}")


def visualize_explanations(
    model: CCNetsModel,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    save_path: str = "ccnets_explanations.png"
):
    """Visualize explanation vectors and their relationships."""
    print(f"\n=== Visualizing Explanations ===\n")

    # Get explanations for sample data
    outputs = model([X_sample, y_sample], training=False)
    explanations = outputs['explanation_vector']
    predictions = outputs['inferred_label']
    reconstructions = outputs['reconstructed_observation']

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Explanation vector heatmap
    axes[0, 0].imshow(explanations[:20], cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Explanation Vectors (First 20 samples)')
    axes[0, 0].set_xlabel('Explanation Dimension')
    axes[0, 0].set_ylabel('Sample Index')

    # 2. Prediction confidence
    pred_confidence = np.max(predictions, axis=1)
    axes[0, 1].hist(pred_confidence, bins=20, alpha=0.7)
    axes[0, 1].set_title('Prediction Confidence Distribution')
    axes[0, 1].set_xlabel('Max Probability')
    axes[0, 1].set_ylabel('Frequency')

    # 3. Reconstruction error vs explanation norm
    explanation_norms = np.linalg.norm(explanations, axis=1)
    reconstruction_errors = np.mean(np.abs(X_sample - reconstructions), axis=1)

    axes[1, 0].scatter(explanation_norms, reconstruction_errors, alpha=0.6)
    axes[1, 0].set_title('Explanation Norm vs Reconstruction Error')
    axes[1, 0].set_xlabel('Explanation L2 Norm')
    axes[1, 0].set_ylabel('Reconstruction Error')

    # 4. Class-wise explanation patterns
    true_classes = np.argmax(y_sample, axis=1)
    for class_idx in range(min(5, np.max(true_classes) + 1)):
        class_mask = true_classes == class_idx
        if np.any(class_mask):
            class_explanations = explanations[class_mask]
            mean_explanation = np.mean(class_explanations, axis=0)
            axes[1, 1].plot(mean_explanation, label=f'Class {class_idx}', alpha=0.8)

    axes[1, 1].set_title('Average Explanation Patterns by Class')
    axes[1, 1].set_xlabel('Explanation Dimension')
    axes[1, 1].set_ylabel('Average Value')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Visualization saved to {save_path}")


def main():
    """Main function demonstrating complete CCNets usage."""
    print("CCNets - Causal Cooperative Networks Demonstration")
    print("=" * 60)

    # 1. Basic usage
    model, X_test, y_test = demonstrate_basic_usage()

    # 2. Prediction and explanation
    demonstrate_prediction_and_explanation(model, X_test, y_test)

    # 3. Cooperative behavior analysis
    demonstrate_cooperative_analysis(model, X_test, y_test)

    # 4. Visualizations
    visualize_explanations(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("CCNets demonstration completed successfully!")
    print("\nKey takeaways:")
    print("- CCNets provides built-in explainability through explanation vectors")
    print("- Cooperative learning balances three network objectives")
    print("- The model can both predict and generate/reconstruct data")
    print("- Explanation vectors provide interpretable latent representations")


if __name__ == "__main__":
    main()