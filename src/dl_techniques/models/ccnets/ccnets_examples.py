"""
CCNets Examples and Usage

This module provides complete examples of how to use the CCNets framework
for different types of problems, including classification, regression,
and data simulation scenarios.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import ops
from dl_techniques.utils.logger import logger


def create_synthetic_dataset(
    n_samples: int = 1000,
    input_dim: int = 20,
    n_classes: int = 10,
    noise_level: float = 0.1,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic dataset for testing CCNets.

    Args:
        n_samples: Number of samples to generate
        input_dim: Dimensionality of input features
        n_classes: Number of classes
        noise_level: Amount of noise to add
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (observations, labels)
    """
    np.random.seed(random_seed)

    # Generate class centers
    centers = np.random.randn(n_classes, input_dim) * 2

    # Generate samples
    observations = []
    labels = []

    for i in range(n_samples):
        # Random class
        class_idx = np.random.randint(n_classes)

        # Generate sample around class center
        sample = centers[class_idx] + np.random.randn(input_dim) * noise_level

        # One-hot encode label
        label = np.zeros(n_classes)
        label[class_idx] = 1.0

        observations.append(sample)
        labels.append(label)

    observations = np.array(observations)
    labels = np.array(labels)

    logger.info(f"Generated synthetic dataset: {n_samples} samples, {input_dim}D input, {n_classes} classes")

    return observations, labels


def create_loan_approval_dataset(
    n_samples: int = 1000,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic loan approval dataset for demonstrating CCNets.

    This creates realistic financial data where CCNets can learn to:
    1. Predict loan approval/rejection
    2. Generate approvable profiles from rejected applications
    3. Provide explanations for decisions

    Args:
        n_samples: Number of loan applications to generate
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (applications, approvals) where approvals are binary
    """
    np.random.seed(random_seed)

    # Feature definitions
    # [income, credit_score, debt_ratio, employment_years, age, loan_amount]
    applications = []
    approvals = []

    for _ in range(n_samples):
        # Generate realistic financial features
        income = np.random.lognormal(10.5, 0.5)  # Log-normal income distribution
        credit_score = np.random.normal(650, 100)  # Credit score
        credit_score = np.clip(credit_score, 300, 850)

        debt_ratio = np.random.beta(2, 5)  # Debt-to-income ratio
        employment_years = np.random.exponential(5)  # Years of employment
        age = np.random.normal(40, 12)  # Age
        age = np.clip(age, 18, 80)

        loan_amount = np.random.lognormal(10, 0.8)  # Loan amount requested

        # Normalize features
        application = np.array([
            income / 100000,  # Normalize income
            (credit_score - 300) / 550,  # Normalize credit score
            debt_ratio,  # Already 0-1
            min(employment_years / 20, 1),  # Normalize employment years
            (age - 18) / 62,  # Normalize age
            loan_amount / 500000  # Normalize loan amount
        ])

        # Simple approval logic with some noise
        approval_score = (
            0.3 * application[0] +  # Income weight
            0.4 * application[1] +  # Credit score weight
            -0.3 * application[2] +  # Debt ratio weight (negative)
            0.2 * application[3] +  # Employment weight
            0.1 * application[4] +  # Age weight
            -0.2 * application[5]   # Loan amount weight (negative)
        )

        # Add noise and threshold
        approval_score += np.random.normal(0, 0.1)
        is_approved = 1.0 if approval_score > 0.3 else 0.0

        applications.append(application)
        approvals.append([is_approved, 1.0 - is_approved])  # One-hot encode

    applications = np.array(applications)
    approvals = np.array(approvals)

    logger.info(f"Generated loan approval dataset: {n_samples} applications")
    logger.info(f"Approval rate: {np.mean(approvals[:, 0]):.2%}")

    return applications, approvals


def train_ccnets_classifier_example():
    """
    Complete example of training a CCNets model for classification.

    This example demonstrates:
    1. Data preparation
    2. Model creation and configuration
    3. Training with monitoring
    4. Evaluation and analysis
    """
    from .ccnets_model import create_ccnets_model
    from .ccnets_training import (
        prepare_ccnets_data,
        train_ccnets_model,
        evaluate_ccnets_model,
        ccnets_inference
    )

    logger.info("=== CCNets Classification Example ===")

    # 1. Create dataset
    observations, labels = create_synthetic_dataset(
        n_samples=2000,
        input_dim=20,
        n_classes=5,
        noise_level=0.15
    )

    # 2. Prepare data
    train_gen, val_gen = prepare_ccnets_data(
        observations, labels,
        validation_split=0.2,
        batch_size=32
    )

    # 3. Create model
    model = create_ccnets_model(
        input_dim=20,
        explanation_dim=10,
        output_dim=5,
        explainer_kwargs={'hidden_dims': [64, 32], 'dropout_rate': 0.2},
        reasoner_kwargs={'hidden_dims': [64, 32], 'dropout_rate': 0.2},
        producer_kwargs={'hidden_dims': [64, 32], 'dropout_rate': 0.2},
        loss_weights=[1.0, 1.0, 1.0]
    )

    # 4. Train model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    history = train_ccnets_model(
        model=model,
        train_data=train_gen,
        val_data=val_gen,
        epochs=50,
        optimizer=optimizer,
        verbose=1
    )

    # 5. Evaluate model
    test_observations, test_labels = create_synthetic_dataset(
        n_samples=500,
        input_dim=20,
        n_classes=5,
        noise_level=0.15,
        random_seed=123
    )

    test_gen, _ = prepare_ccnets_data(
        test_observations, test_labels,
        validation_split=0.0,
        batch_size=32
    )

    metrics = evaluate_ccnets_model(model, test_gen, return_outputs=True)

    # 6. Print results
    logger.info("=== Training Results ===")
    logger.info(f"Final Cooperation Score: {metrics['cooperation_score']:.6f}")
    logger.info(f"Cross-verification Accuracy: {metrics.get('cross_verification_accuracy', 'N/A'):.4f}")
    logger.info(f"Mean Inference Loss: {metrics['mean_inference_loss']:.6f}")
    logger.info(f"Mean Generation Loss: {metrics['mean_generation_loss']:.6f}")
    logger.info(f"Mean Reconstruction Loss: {metrics['mean_reconstruction_loss']:.6f}")

    # 7. Test inference
    sample_obs = test_observations[:10]
    results = ccnets_inference(model, sample_obs, return_explanations=True)

    logger.info("=== Inference Test ===")
    logger.info(f"Predictions shape: {results['predictions'].shape}")
    logger.info(f"Explanations shape: {results['explanations'].shape}")

    return model, history, metrics


def train_ccnets_loan_approval_example():
    """
    Complete example using CCNets for loan approval with data simulation.

    This demonstrates the practical application described in the guide:
    generating approvable loan profiles from rejected applications.
    """
    from .ccnets_model import create_ccnets_model
    from .ccnets_training import (
        prepare_ccnets_data,
        train_ccnets_model,
        evaluate_ccnets_model,
        simulate_approval_scenario
    )

    logger.info("=== CCNets Loan Approval Example ===")

    # 1. Create loan dataset
    applications, approvals = create_loan_approval_dataset(n_samples=2000)

    # 2. Prepare data
    train_gen, val_gen = prepare_ccnets_data(
        applications, approvals,
        validation_split=0.2,
        batch_size=32
    )

    # 3. Create model optimized for loan data
    model = create_ccnets_model(
        input_dim=6,  # 6 financial features
        explanation_dim=8,  # Compact explanations
        output_dim=2,  # Binary classification
        explainer_kwargs={'hidden_dims': [32, 16], 'dropout_rate': 0.3},
        reasoner_kwargs={'hidden_dims': [32, 16], 'dropout_rate': 0.3},
        producer_kwargs={'hidden_dims': [32, 16], 'dropout_rate': 0.3},
        loss_weights=[1.2, 1.0, 1.0]  # Emphasize inference for consistency
    )

    # 4. Train with financial-specific optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)

    history = train_ccnets_model(
        model=model,
        train_data=train_gen,
        val_data=val_gen,
        epochs=100,
        optimizer=optimizer,
        verbose=1
    )

    # 5. Test data simulation capability
    test_applications, test_approvals = create_loan_approval_dataset(
        n_samples=100, random_seed=456
    )

    # Find rejected applications
    rejected_indices = np.where(test_approvals[:, 0] == 0)[0]

    if len(rejected_indices) > 0:
        logger.info("=== Data Simulation Test ===")

        # Take first rejected application
        rejected_app = test_applications[rejected_indices[0:1]]

        # Get explanation vector
        explanation = model.explainer_network(rejected_app, training=False)

        # Create approval label
        approval_label = np.array([[1.0, 0.0]])  # Approved

        # Simulate approvable application
        simulation_result = simulate_approval_scenario(
            model, rejected_app, explanation, approval_label
        )

        logger.info(f"Original rejected application: {rejected_app[0]}")
        logger.info(f"Generated approvable application: {simulation_result['approvable_application'][0]}")
        logger.info(f"Validation error: {simulation_result['validation_error']:.6f}")

        # Analyze the changes
        changes = simulation_result['approvable_application'][0] - rejected_app[0]
        feature_names = ['Income', 'Credit Score', 'Debt Ratio', 'Employment Years', 'Age', 'Loan Amount']

        logger.info("=== Recommended Changes for Approval ===")
        for i, (name, change) in enumerate(zip(feature_names, changes)):
            if abs(change) > 0.01:  # Only show significant changes
                direction = "increase" if change > 0 else "decrease"
                logger.info(f"  {name}: {direction} by {abs(change):.3f}")

    # 6. Evaluate overall performance
    test_gen, _ = prepare_ccnets_data(
        test_applications, test_approvals,
        validation_split=0.0,
        batch_size=32
    )

    metrics = evaluate_ccnets_model(model, test_gen, return_outputs=True)

    logger.info("=== Final Results ===")
    logger.info(f"Cooperation Score: {metrics['cooperation_score']:.6f}")
    logger.info(f"Cross-verification Accuracy: {metrics.get('cross_verification_accuracy', 'N/A'):.4f}")

    return model, history, metrics


def visualize_ccnets_cooperation(history: keras.callbacks.History):
    """
    Visualize the cooperation dynamics during CCNets training.

    Args:
        history: Training history from CCNets model
    """
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CCNets Training Dynamics', fontsize=16)

        # Plot individual losses
        ax1 = axes[0, 0]
        if 'inference_loss' in history.history:
            ax1.plot(history.history['inference_loss'], label='Inference Loss', color='red')
        if 'generation_loss' in history.history:
            ax1.plot(history.history['generation_loss'], label='Generation Loss', color='blue')
        if 'reconstruction_loss' in history.history:
            ax1.plot(history.history['reconstruction_loss'], label='Reconstruction Loss', color='green')
        ax1.set_title('Individual Loss Components')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot network errors
        ax2 = axes[0, 1]
        if 'explainer_error' in history.history:
            ax2.plot(history.history['explainer_error'], label='Explainer Error', color='purple')
        if 'reasoner_error' in history.history:
            ax2.plot(history.history['reasoner_error'], label='Reasoner Error', color='orange')
        if 'producer_error' in history.history:
            ax2.plot(history.history['producer_error'], label='Producer Error', color='brown')
        ax2.set_title('Network-Specific Errors')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error')
        ax2.legend()
        ax2.grid(True)

        # Plot total loss
        ax3 = axes[1, 0]
        if 'loss' in history.history:
            ax3.plot(history.history['loss'], label='Training Loss', color='black')
        if 'val_loss' in history.history:
            ax3.plot(history.history['val_loss'], label='Validation Loss', color='gray', linestyle='--')
        ax3.set_title('Total Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)

        # Plot cooperation score if available
        ax4 = axes[1, 1]
        if 'cooperation_score' in history.history:
            ax4.plot(history.history['cooperation_score'], label='Cooperation Score', color='darkgreen')
            ax4.set_title('Cooperation Quality')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Cooperation Score')
            ax4.legend()
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, 'Cooperation Score\nNot Available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cooperation Quality')

        plt.tight_layout()
        plt.show()

    except ImportError:
        logger.warning("Matplotlib not available for visualization")


def run_complete_ccnets_demo():
    """
    Run a complete demonstration of CCNets capabilities.

    This function executes both classification and loan approval examples,
    showcasing the full range of CCNets functionality.
    """
    logger.info("Starting Complete CCNets Demonstration")

    try:
        # Run classification example
        logger.info("\n" + "="*50)
        logger.info("PART 1: Classification Example")
        logger.info("="*50)

        class_model, class_history, class_metrics = train_ccnets_classifier_example()

        # Run loan approval example
        logger.info("\n" + "="*50)
        logger.info("PART 2: Loan Approval Example")
        logger.info("="*50)

        loan_model, loan_history, loan_metrics = train_ccnets_loan_approval_example()

        # Visualize results
        logger.info("\n" + "="*50)
        logger.info("PART 3: Results Visualization")
        logger.info("="*50)

        logger.info("Classification Results:")
        visualize_ccnets_cooperation(class_history)

        logger.info("Loan Approval Results:")
        visualize_ccnets_cooperation(loan_history)

        # Summary
        logger.info("\n" + "="*50)
        logger.info("DEMO SUMMARY")
        logger.info("="*50)
        logger.info(f"Classification Cooperation Score: {class_metrics['cooperation_score']:.6f}")
        logger.info(f"Loan Approval Cooperation Score: {loan_metrics['cooperation_score']:.6f}")
        logger.info("CCNets demonstration completed successfully!")

        return {
            'classification': {'model': class_model, 'history': class_history, 'metrics': class_metrics},
            'loan_approval': {'model': loan_model, 'history': loan_history, 'metrics': loan_metrics}
        }

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    # Run the complete demo when this file is executed
    results = run_complete_ccnets_demo()