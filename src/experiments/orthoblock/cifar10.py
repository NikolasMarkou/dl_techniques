"""
OrthoCenterBlock Experiment

This script conducts a comparative experiment between:
1. A baseline CNN model with standard dense layers
2. The same CNN architecture with OrthoCenterBlock layers on the CIFAR-10 dataset

OrthoCenterBlock combines:
- Orthonormal-regularized weights (enforcing W^T * W ≈ I)
- Centering normalization (mean subtraction)
- Logit normalization (projection to unit hypersphere)
- Constrained scale parameters [0,1] for interpretable feature attention

The experiment examines:
- Classification accuracy comparison between models
- Training/validation loss and accuracy curves
- Orthogonality measure (||W^T W - I||_F) over training
- Distribution of scale parameters throughout training
- Convergence speed analysis

Both models share identical initialization and architecture except for
the replaced dense layer. Results are saved as visualizations and model files
in the experiment directory.
"""

import os
import keras
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.api.datasets import cifar10
from keras.api.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from dl_techniques.utils.logger import logger
from dl_techniques.layers.experimental.orthoblock import OrthoBlock

# ====================== CONFIGURATION ======================

CONFIG = {
    # Experiment setup
    "EXPERIMENT_DIR": "orthocenter_experiment",
    "RANDOM_SEED": 42,

    # Training parameters
    "EPOCHS": 10,  # Set to lower value for quick testing
    "BATCH_SIZE": 128,
    "INITIAL_LR": 0.001,
    "LR_SCHEDULE_THRESHOLDS": [50, 75],
    "LR_VALUES": [0.001, 0.0001, 0.00001],
    "EARLY_STOPPING_PATIENCE": 20,

    # Dataset configuration
    "VALIDATION_SIZE": 5000,

    # Activation functions
    "ACTIVATION": "mish",  # Default activation used throughout the network

    # Model architecture
    "CONV_BLOCKS": [
        {"FILTERS": 32, "KERNEL_SIZE": 3, "DROPOUT": 0.2},
        {"FILTERS": 64, "KERNEL_SIZE": 3, "DROPOUT": 0.3},
        {"FILTERS": 128, "KERNEL_SIZE": 3, "DROPOUT": 0.4},
    ],
    "DENSE_UNITS": 256,
    "FINAL_DROPOUT": 0.5,
    "NUM_CLASSES": 10,
    "OUTPUT_ACTIVATION": "softmax",

    # OrthoCenterBlock parameters
    "ORTHO_REG_FACTOR": 0.1,
    "SCALE_INITIAL_VALUE": 0.5,
    "ORTHO_REG_SCHEDULER": {
        "MAX_FACTOR": 0.1,
        "MIN_FACTOR": 0.001
    }
}

# Set random seed for reproducibility
np.random.seed(CONFIG["RANDOM_SEED"])
keras.utils.set_random_seed(CONFIG["RANDOM_SEED"])

# Create experiment directory
EXPERIMENT_DIR = CONFIG["EXPERIMENT_DIR"]
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

# Configure GPU memory growth if using TensorFlow backend
try:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info("GPU memory growth enabled")
except:
    logger.info("Not using TensorFlow or no GPU available")


class OrthoRegScheduler(keras.callbacks.Callback):
    """Callback to schedule orthogonal regularization strength."""

    def __init__(self, max_factor=CONFIG["ORTHO_REG_SCHEDULER"]["MAX_FACTOR"],
                 min_factor=CONFIG["ORTHO_REG_SCHEDULER"]["MIN_FACTOR"],
                 total_epochs=CONFIG["EPOCHS"]):
        super().__init__()
        self.max_factor = max_factor
        self.min_factor = min_factor
        self.total_epochs = total_epochs
        self.ortho_layers = []

    def on_train_begin(self, logs=None):
        # Find all OrthoCenterBlock layers
        for layer in self.model.layers:
            if isinstance(layer, OrthoBlock):
                self.ortho_layers.append(layer)

        logger.info(f"Found {len(self.ortho_layers)} OrthoBlock layers for scheduling")

    def on_epoch_begin(self, epoch, logs=None):
        # Cosine annealing schedule
        progress = min(epoch / self.total_epochs, 1.0)
        factor = self.max_factor * (0.5 * (1.0 + np.cos(progress * np.pi))) + self.min_factor

        # Update the regularization factor for each layer
        for layer in self.ortho_layers:
            # Access the dense sublayer's kernel regularizer
            if hasattr(layer.dense, 'kernel_regularizer'):
                reg = layer.dense.kernel_regularizer

                # Check if we're using the ortho regularizer directly or a combined one
                if hasattr(reg, 'factor'):
                    # Direct OrthonomalRegularizer
                    reg.factor = factor
                else:
                    # This could be a combined regularizer - we'll need to patch it
                    # This is more complex and implementation-dependent
                    pass


class MetricsTracker(keras.callbacks.Callback):
    """Callback to track additional metrics during training."""

    def __init__(self, validation_data=None, model_type="unknown"):
        super().__init__()
        self.validation_data = validation_data
        self.model_type = model_type
        self.batch_metrics = {'train_loss': [], 'train_acc': []}
        self.epoch_metrics = {'val_loss': [], 'val_acc': [], 'epoch_times': [],
                              'ortho_metrics': [], 'scale_histograms': []}
        self.ortho_layers = []
        self.start_time = None

    def on_train_begin(self, logs=None):
        # Find OrthoCenterBlock layers
        for layer in self.model.layers:
            if isinstance(layer, OrthoBlock):
                self.ortho_layers.append(layer)

        self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        # Track batch-level metrics
        if logs:
            self.batch_metrics['train_loss'].append(logs.get('loss'))
            self.batch_metrics['train_acc'].append(logs.get('accuracy'))

    def calculate_orthogonality(self, layer):
        """Calculate orthogonality measure ||W^T W - I||_F for a layer."""
        if not hasattr(layer, 'dense'):
            return None

        weights = layer.dense.kernel.numpy()
        wt_w = np.matmul(weights.T, weights)
        identity = np.eye(wt_w.shape[0], wt_w.shape[1])
        diff = wt_w - identity
        return np.sqrt(np.sum(np.square(diff)))

    def get_scale_histogram(self, layer):
        """Get histogram data of constrained scale parameters."""
        if not hasattr(layer, 'constrained_scale'):
            return None

        # Get the scales - this depends on how LearnableMultiplier is implemented
        # Assuming it has a weight attribute we can access
        if hasattr(layer.constrained_scale, 'multiplier'):
            scales = layer.constrained_scale.multiplier.numpy()
            return scales
        return None

    def on_epoch_end(self, epoch, logs=None):
        # Track epoch time
        epoch_time = time.time() - self.start_time
        self.epoch_metrics['epoch_times'].append(epoch_time)
        self.start_time = time.time()

        # Track validation metrics
        if logs:
            self.epoch_metrics['val_loss'].append(logs.get('val_loss'))
            self.epoch_metrics['val_acc'].append(logs.get('val_accuracy'))

        # Track orthogonality metrics for each OrthoCenterBlock
        ortho_values = {}
        for i, layer in enumerate(self.ortho_layers):
            ortho_value = self.calculate_orthogonality(layer)
            if ortho_value is not None:
                ortho_values[f'layer_{i}'] = ortho_value

        self.epoch_metrics['ortho_metrics'].append(ortho_values)

        # Track scale parameter histograms
        scale_histograms = {}
        for i, layer in enumerate(self.ortho_layers):
            scales = self.get_scale_histogram(layer)
            if scales is not None:
                scale_histograms[f'layer_{i}'] = scales

        self.epoch_metrics['scale_histograms'].append(scale_histograms)


def load_and_preprocess_cifar10():
    """Load and preprocess the CIFAR-10 dataset."""
    # Load dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert to float and normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, CONFIG["NUM_CLASSES"])
    y_test = keras.utils.to_categorical(y_test, CONFIG["NUM_CLASSES"])

    # Create a validation split
    val_size = CONFIG["VALIDATION_SIZE"]
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    logger.info(f"Training set: {x_train.shape}, {y_train.shape}")
    logger.info(f"Validation set: {x_val.shape}, {y_val.shape}")
    logger.info(f"Test set: {x_test.shape}, {y_test.shape}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_model_template(input_shape=(32, 32, 3), num_classes=CONFIG["NUM_CLASSES"]):
    """Create a template model with placeholders for the final dense layer.
       This function will be used to ensure identical initialization for both models.
    """
    # Store initial random state to ensure same initialization
    random_state = np.random.get_state()

    inputs = keras.Input(shape=input_shape)

    # Create convolutional blocks based on configuration
    x = inputs
    for i, block_config in enumerate(CONFIG["CONV_BLOCKS"]):
        # First conv layer in block
        x = keras.layers.Conv2D(
            block_config["FILTERS"],
            (block_config["KERNEL_SIZE"], block_config["KERNEL_SIZE"]),
            padding='same'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(CONFIG["ACTIVATION"])(x)

        # Second conv layer in block
        x = keras.layers.Conv2D(
            block_config["FILTERS"],
            (block_config["KERNEL_SIZE"], block_config["KERNEL_SIZE"]),
            padding='same'
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(CONFIG["ACTIVATION"])(x)

        # Pooling and dropout
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = keras.layers.Dropout(block_config["DROPOUT"])(x)

    # Flatten layer before final dense/orthocenter block
    flattened = keras.layers.Flatten()(x)

    # This gives us a template that we'll use to create both models
    template_model = keras.Model(inputs=inputs, outputs=flattened, name="template_model")

    # Restore random state
    np.random.set_state(random_state)

    return template_model


def create_models(input_shape=(32, 32, 3), num_classes=CONFIG["NUM_CLASSES"]):
    """Create both baseline and orthocenter models with identical initialization."""

    # Create the shared template
    template_model = create_model_template(input_shape, num_classes)

    # Create baseline model using template
    baseline_inputs = keras.Input(shape=input_shape)
    x_baseline = template_model(baseline_inputs)

    # For baseline: Regular Dense layer -> Dropout -> Output
    x_baseline = keras.layers.Dense(CONFIG["DENSE_UNITS"])(x_baseline)
    x_baseline = keras.layers.BatchNormalization()(x_baseline)
    x_baseline = keras.layers.Activation(CONFIG["ACTIVATION"])(x_baseline)
    x_baseline = keras.layers.Dropout(CONFIG["FINAL_DROPOUT"])(x_baseline)
    baseline_outputs = keras.layers.Dense(num_classes, activation=CONFIG["OUTPUT_ACTIVATION"])(x_baseline)

    baseline_model = keras.Model(inputs=baseline_inputs, outputs=baseline_outputs, name="baseline_cnn")

    # Create orthocenter model using the same template
    orthocenter_inputs = keras.Input(shape=input_shape)
    x_orthocenter = template_model(orthocenter_inputs)

    # For orthocenter: OrthoCenterBlock -> Dropout -> Output
    x_orthocenter = OrthoBlock(
        units=CONFIG["DENSE_UNITS"],
        activation=CONFIG["ACTIVATION"],
        ortho_reg_factor=CONFIG["ORTHO_REG_FACTOR"],
        scale_initial_value=CONFIG["SCALE_INITIAL_VALUE"]
    )(x_orthocenter)
    x_orthocenter = keras.layers.Dropout(CONFIG["FINAL_DROPOUT"])(x_orthocenter)
    orthocenter_outputs = keras.layers.Dense(num_classes, activation=CONFIG["OUTPUT_ACTIVATION"])(x_orthocenter)

    orthocenter_model = keras.Model(inputs=orthocenter_inputs, outputs=orthocenter_outputs, name="orthocenter_cnn")

    return baseline_model, orthocenter_model


def lr_schedule(epoch, lr):
    """Learning rate schedule using CONFIG parameters."""
    thresholds = CONFIG["LR_SCHEDULE_THRESHOLDS"]
    lr_values = CONFIG["LR_VALUES"]

    if epoch > thresholds[1]:
        return lr_values[2]
    elif epoch > thresholds[0]:
        return lr_values[1]
    else:
        return lr_values[0]


def train_and_evaluate(model, data, model_name, epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"]):
    """Train and evaluate a model."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["INITIAL_LR"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create callbacks
    checkpoint_path = os.path.join(EXPERIMENT_DIR, f"{model_name}_best.keras")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=CONFIG["EARLY_STOPPING_PATIENCE"],
        restore_best_weights=True
    )

    lr_scheduler = LearningRateScheduler(lr_schedule)

    metrics_tracker = MetricsTracker(validation_data=(x_val, y_val), model_type=model_name)

    callbacks = [checkpoint, early_stopping, lr_scheduler, metrics_tracker]

    # Add OrthoRegScheduler for OrthoCenterBlock model
    if model_name == "orthocenter_cnn":
        ortho_scheduler = OrthoRegScheduler(
            max_factor=CONFIG["ORTHO_REG_SCHEDULER"]["MAX_FACTOR"],
            min_factor=CONFIG["ORTHO_REG_SCHEDULER"]["MIN_FACTOR"],
            total_epochs=epochs
        )
        callbacks.append(ortho_scheduler)

    # Train model
    logger.info(f"Training {model_name}...")
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    logger.info(f"Evaluating {model_name} on test set...")
    test_results = model.evaluate(x_test, y_test, verbose=1)
    logger.info(f"{model_name} Test Loss: {test_results[0]:.4f}")
    logger.info(f"{model_name} Test Accuracy: {test_results[1]:.4f}")

    # Save final model
    final_path = os.path.join(EXPERIMENT_DIR, f"{model_name}_final.keras")
    model.save(final_path)

    # Save training history
    history_df = pd.DataFrame({
        'train_loss': history.history['loss'],
        'train_accuracy': history.history['accuracy'],
        'val_loss': history.history['val_loss'],
        'val_accuracy': history.history['val_accuracy']
    })
    history_df.to_csv(os.path.join(EXPERIMENT_DIR, f"{model_name}_history.csv"), index=False)

    # Return results dictionary
    return {
        'test_loss': test_results[0],
        'test_accuracy': test_results[1],
        'history': history.history,
        'metrics_tracker': metrics_tracker,
        'model': model
    }


def visualize_results(baseline_results, orthocenter_results):
    """Visualize and compare results between the two models."""

    # 1. Training and Validation Accuracy Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(baseline_results['history']['accuracy'], label='Baseline Train')
    plt.plot(baseline_results['history']['val_accuracy'], label='Baseline Val')
    plt.plot(orthocenter_results['history']['accuracy'], label='OrthoCenterBlock Train')
    plt.plot(orthocenter_results['history']['val_accuracy'], label='OrthoCenterBlock Val')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Training and Validation Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(baseline_results['history']['loss'], label='Baseline Train')
    plt.plot(baseline_results['history']['val_loss'], label='Baseline Val')
    plt.plot(orthocenter_results['history']['loss'], label='OrthoCenterBlock Train')
    plt.plot(orthocenter_results['history']['val_loss'], label='OrthoCenterBlock Val')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_DIR, 'accuracy_loss_comparison.png'), dpi=300)

    # 3. Orthogonality Metrics Over Time (for OrthoCenterBlock model only)
    ortho_metrics = orthocenter_results['metrics_tracker'].epoch_metrics['ortho_metrics']
    if ortho_metrics and len(ortho_metrics) > 0 and 'layer_0' in ortho_metrics[0]:
        plt.figure(figsize=(10, 5))
        ortho_values = [metrics.get('layer_0', np.nan) for metrics in ortho_metrics]
        plt.plot(ortho_values, 'o-', label='||W^T W - I||_F')
        plt.title('Orthogonality Measure Over Training')
        plt.ylabel('Frobenius Norm')
        plt.xlabel('Epoch')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EXPERIMENT_DIR, 'orthogonality_metric.png'), dpi=300)

    # 4. Scale Parameter Distributions
    scale_histograms = orthocenter_results['metrics_tracker'].epoch_metrics['scale_histograms']
    if scale_histograms and len(scale_histograms) > 0:
        epochs_to_plot = [0, len(scale_histograms) // 3, 2 * len(scale_histograms) // 3, -1]
        epochs_to_plot = [ep if ep >= 0 else len(scale_histograms) + ep for ep in epochs_to_plot]

        plt.figure(figsize=(12, 8))
        for i, epoch in enumerate(epochs_to_plot):
            if epoch < len(scale_histograms) and 'layer_0' in scale_histograms[epoch]:
                plt.subplot(2, 2, i + 1)
                scales = scale_histograms[epoch]['layer_0']
                plt.hist(scales, bins=20, alpha=0.7)
                plt.title(f'Scale Distribution (Epoch {epoch})')
                plt.xlabel('Scale Value')
                plt.ylabel('Count')
                plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(EXPERIMENT_DIR, 'scale_distributions.png'), dpi=300)

    # 5. Test Performance Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['Baseline', 'OrthoCenterBlock'],
            [baseline_results['test_accuracy'], orthocenter_results['test_accuracy']],
            color=['skyblue', 'orange'])
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(EXPERIMENT_DIR, 'test_accuracy_comparison.png'), dpi=300)

    # 6. Summarize results in text file
    with open(os.path.join(EXPERIMENT_DIR, 'experiment_summary.txt'), 'w') as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=================\n\n")

        f.write("Baseline Model:\n")
        f.write(f"  - Test Loss: {baseline_results['test_loss']:.4f}\n")
        f.write(f"  - Test Accuracy: {baseline_results['test_accuracy']:.4f}\n\n")

        f.write("OrthoCenterBlock Model:\n")
        f.write(f"  - Test Loss: {orthocenter_results['test_loss']:.4f}\n")
        f.write(f"  - Test Accuracy: {orthocenter_results['test_accuracy']:.4f}\n\n")

        improvement = (orthocenter_results['test_accuracy'] - baseline_results['test_accuracy']) * 100
        f.write(f"Improvement: {improvement:.2f}% accuracy\n\n")

        # Calculate convergence speed (epochs to 90% of max val accuracy)
        baseline_val_acc = baseline_results['history']['val_accuracy']
        ortho_val_acc = orthocenter_results['history']['val_accuracy']

        baseline_max = max(baseline_val_acc)
        ortho_max = max(ortho_val_acc)

        baseline_epochs_to_90 = next((i for i, x in enumerate(baseline_val_acc) if x >= 0.9 * baseline_max),
                                     len(baseline_val_acc))
        ortho_epochs_to_90 = next((i for i, x in enumerate(ortho_val_acc) if x >= 0.9 * ortho_max), len(ortho_val_acc))

        f.write(f"Convergence Speed:\n")
        f.write(f"  - Baseline: {baseline_epochs_to_90} epochs to reach 90% of max val accuracy\n")
        f.write(f"  - OrthoCenterBlock: {ortho_epochs_to_90} epochs to reach 90% of max val accuracy\n")


def main():
    """Main function to run the experiment."""
    logger.info("Loading CIFAR-10 dataset...")
    data = load_and_preprocess_cifar10()

    # Create both models with identical initialization
    logger.info("Creating models with identical initialization...")
    baseline_model, orthocenter_model = create_models()

    # Print model summaries to verify structures
    logger.info("Baseline Model Summary:")
    baseline_model.summary(print_fn=logger.info)

    logger.info("OrthoCenterBlock Model Summary:")
    orthocenter_model.summary(print_fn=logger.info)

    # Train and evaluate both models
    test_epochs = 10  # For quick testing. Set to CONFIG["EPOCHS"] for full run
    baseline_results = train_and_evaluate(baseline_model, data, "baseline_cnn", epochs=test_epochs)
    orthocenter_results = train_and_evaluate(orthocenter_model, data, "orthocenter_cnn", epochs=test_epochs)

    # Visualize and compare results
    logger.info("Generating visualizations and comparison...")
    visualize_results(baseline_results, orthocenter_results)

    logger.info(f"Experiment completed. Results saved to: {EXPERIMENT_DIR}")


if __name__ == "__main__":
    main()