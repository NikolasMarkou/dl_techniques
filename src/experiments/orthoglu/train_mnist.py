import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse


from dl_techniques.layers.ffn.orthoglu_ffn import OrthoGLUFFN


# ---------------------------------------------------------------------
# 1. Data Loading and Preparation for MNIST
# ---------------------------------------------------------------------
def get_mnist_dataset():
    """Loads and preprocesses the MNIST dataset for CNNs."""
    print("\n--- Loading and preparing MNIST dataset ---")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print(f"x_train shape: {x_train.shape}, {x_train.shape[0]} train samples")
    print(f"x_test shape: {x_test.shape}, {x_test.shape[0]} test samples")
    return (x_train, y_train), (x_test, y_test)


# ---------------------------------------------------------------------
# 2. Model Architectures with CNN Frontend
# ---------------------------------------------------------------------
class BaselineGLUFFN(keras.layers.Layer):
    """A standard GLU FeedForward network using Dense layers."""

    def __init__(self, hidden_dim, output_dim, activation="gelu", dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_proj = keras.layers.Dense(hidden_dim * 2, name="input_proj_dense")
        self.output_proj = keras.layers.Dense(output_dim, name="output_proj_dense")
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.activation = keras.activations.get(activation)

    def call(self, inputs, training=None):
        gate, value = tf.split(self.input_proj(inputs), num_or_size_splits=2, axis=-1)
        gated_value = self.activation(gate) * value
        return self.output_proj(self.dropout(gated_value, training=training))


def build_baseline_model(input_shape, num_classes, num_blocks, hidden_dim=256, dropout_rate=0.1):
    """Builds a CNN model with a configurable standard GLU FFN backend."""
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2,2), activation="relu")(inputs)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.GlobalMaxPool2D()(x)
    x = keras.layers.Flatten()(x)
    for i in range(num_blocks):
        x = BaselineGLUFFN(hidden_dim, hidden_dim, dropout_rate=dropout_rate, name=f"glu_block_{i + 1}")(x)
        x = keras.layers.LayerNormalization(name=f"ln_{i + 1}")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier", use_bias=False)(x)
    return keras.Model(inputs, outputs, name=f"Baseline_CNN_GLU_{num_blocks}_Blocks")


def build_ortho_model(input_shape, num_classes, num_blocks, hidden_dim=256, dropout_rate=0.1, ortho_reg_factor=0.01):
    """Builds a CNN model with a configurable OrthoGLU FFN backend."""
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2,2),activation="relu")(inputs)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = keras.layers.GlobalMaxPool2D()(x)
    x = keras.layers.Flatten()(x)
    for i in range(num_blocks):
        x = OrthoGLUFFN(hidden_dim, hidden_dim, dropout_rate=dropout_rate, ortho_reg_factor=ortho_reg_factor,
                        name=f"ortho_glu_block_{i + 1}")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier", use_bias=False)(x)
    return keras.Model(inputs, outputs, name=f"Ortho_CNN_GLU_{num_blocks}_Blocks")


# ---------------------------------------------------------------------
# 3. Training and Evaluation Utilities
# ---------------------------------------------------------------------
def calculate_ece(y_true, y_pred_probs, n_bins=15):
    """Calculates the Expected Calibration Error (ECE)."""
    confidences, predictions = np.max(y_pred_probs, axis=1), np.argmax(y_pred_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    ece, bin_boundaries = 0.0, np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.mean(in_bin) > 0:
            ece += np.abs(np.mean(confidences[in_bin]) - np.mean(accuracies[in_bin])) * np.mean(in_bin)
    return ece


def calculate_fit_scores(history):
    """Calculates overfitting and underfitting scores from training history."""
    final_train_acc, final_val_acc = history.history['accuracy'][-1], history.history['val_accuracy'][-1]
    return {"overfit_score": (final_train_acc - final_val_acc) * 100, "underfit_score": (1.0 - final_val_acc) * 100}


def plot_history(histories, title="Model Training Comparison"):
    """Plots training and validation accuracy/loss for given histories."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for name, history in histories.items():
        ax1.plot(history.history['val_accuracy'], label=f'{name} Val Acc', linestyle='--')
        ax2.plot(history.history['val_loss'], label=f'{name} Val Loss', linestyle='--')
    ax1.set_title('Validation Accuracy');
    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Accuracy');
    ax1.legend()
    ax2.set_title('Validation Loss');
    ax2.set_xlabel('Epoch');
    ax2.set_ylabel('Loss');
    ax2.legend()
    fig.suptitle(title, fontsize=16);
    plt.show()


def inspect_feature_gates(ortho_model):
    """Inspects the learned scaling factors from the first FFN OrthoBlock."""
    print("\n--- Inspecting Learned Feature Gates from First OrthoGLUFFN Block ---")
    try:
        first_ortho_glu_layer = ortho_model.get_layer("ortho_glu_block_1")
        gate_weights = first_ortho_glu_layer.input_proj_ortho.constrained_scale.get_weights()[0]
        print(
            f"Shape: {gate_weights.shape}, Min: {np.min(gate_weights):.4f}, Max: {np.max(gate_weights):.4f}, Mean: {np.mean(gate_weights):.4f}")
        print(f"Gates effectively closed (< 0.1): {np.sum(gate_weights < 0.1)} / {len(gate_weights)}")
        plt.figure(figsize=(10, 5));
        plt.hist(gate_weights, bins=50, color='skyblue', edgecolor='black')
        plt.title("Distribution of Gate Values in First OrthoBlock");
        plt.xlabel("Gate Value");
        plt.ylabel("Frequency");
        plt.grid(True);
        plt.show()
    except Exception as e:
        print(f"Could not inspect weights. Error: {e}")


def visualize_model_predictions(model_name, x_data, y_true, y_pred_probs, num_examples=5):
    """Visualizes model predictions, focusing on high/low confidence cases."""
    predictions = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1)

    correct_indices = np.where(predictions == y_true)[0]
    incorrect_indices = np.where(predictions != y_true)[0]

    # Identify specific cases
    high_conf_correct = correct_indices[np.argsort(confidences[correct_indices])[-num_examples:]]
    high_conf_incorrect = incorrect_indices[np.argsort(confidences[incorrect_indices])[-num_examples:]]
    low_conf = np.argsort(confidences)[:num_examples]

    fig, axes = plt.subplots(3, num_examples, figsize=(15, 9))
    fig.suptitle(f'Prediction Analysis for: {model_name}', fontsize=20)

    for i in range(num_examples):
        # High Confidence, Correct
        idx = high_conf_correct[i]
        axes[0, i].imshow(x_data[idx].squeeze(), cmap='gray')
        axes[0, i].set_title(f"True: {y_true[idx]}\nPred: {predictions[idx]} ({confidences[idx]:.2f})", color='green')
        axes[0, i].axis('off')

        # High Confidence, Incorrect
        idx = high_conf_incorrect[i]
        axes[1, i].imshow(x_data[idx].squeeze(), cmap='gray')
        axes[1, i].set_title(f"True: {y_true[idx]}\nPred: {predictions[idx]} ({confidences[idx]:.2f})", color='red')
        axes[1, i].axis('off')

        # Low Confidence / Ambiguous
        idx = low_conf[i]
        axes[2, i].imshow(x_data[idx].squeeze(), cmap='gray')
        axes[2, i].set_title(f"True: {y_true[idx]}\nPred: {predictions[idx]} ({confidences[idx]:.2f})", color='orange')
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel("Confident Correct", fontsize=12, rotation=90, labelpad=20)
    axes[1, 0].set_ylabel("Confident Incorrect\n(Miscalibrated)", fontsize=12, rotation=90, labelpad=20)
    axes[2, 0].set_ylabel("Low Confidence\n(Ambiguous)", fontsize=12, rotation=90, labelpad=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare CNN-FFN models on MNIST.")
    parser.add_argument("--blocks", type=int, default=1, help="Number of GLU/OrthoGLU blocks after the CNN base.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate for FFN blocks.")
    parser.add_argument("--ortho_reg_factor", type=float, default=0.1, help="Orthogonal regularization factor.")
    args = parser.parse_args()

    # System Constants
    NUM_CLASSES, INPUT_SHAPE = 10, (28, 28, 1)

    # 1. Load Data
    (X_train, y_train), (X_val, y_val) = get_mnist_dataset()

    # 2. Build Models
    baseline_model = build_baseline_model(INPUT_SHAPE, NUM_CLASSES, args.blocks, dropout_rate=args.dropout_rate)
    ortho_model = build_ortho_model(INPUT_SHAPE, NUM_CLASSES, args.blocks, dropout_rate=args.dropout_rate, ortho_reg_factor=args.ortho_reg_factor)

    optimizer = keras.optimizers.Adam(1e-3)
    loss_fn = "sparse_categorical_crossentropy"
    baseline_model.compile(optimizer, loss=loss_fn, metrics=["accuracy"])
    ortho_model.compile(keras.optimizers.Adam(1e-3), loss=loss_fn, metrics=["accuracy"])

    print("\n--- Baseline Model Summary ---");
    baseline_model.summary()
    print("\n--- OrthoGLUFFN Model Summary ---");
    ortho_model.summary()

    # 3. Train Models
    print(f"\n--- Training Models for {args.epochs} epochs with {args.blocks} FFN blocks ---")
    history_baseline = baseline_model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                                          validation_data=(X_val, y_val), verbose=2)
    history_ortho = ortho_model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
                                    validation_data=(X_val, y_val), verbose=2)

    # 4. Final Performance Analysis
    print("\n--- Generating Final Performance Metrics ---")
    baseline_preds, ortho_preds = baseline_model.predict(X_val), ortho_model.predict(X_val)
    ece_baseline, ece_ortho = calculate_ece(y_val, baseline_preds), calculate_ece(y_val, ortho_preds)
    fit_scores_baseline, fit_scores_ortho = calculate_fit_scores(history_baseline), calculate_fit_scores(history_ortho)

    print("\n" + "=" * 50);
    print(" " * 15 + "PERFORMANCE REPORT");
    print("=" * 50)
    print(f"\n[Baseline Model: {baseline_model.name}]")
    print(f"  - Final Val Acc: {history_baseline.history['val_accuracy'][-1]:.4f}, ECE: {ece_baseline:.4f}")
    print(
        f"  - Overfit Score: {fit_scores_baseline['overfit_score']:.2f}%, Underfit Score: {fit_scores_baseline['underfit_score']:.2f}%")
    print(f"\n[OrthoGLUFFN Model: {ortho_model.name}]")
    print(f"  - Final Val Acc: {history_ortho.history['val_accuracy'][-1]:.4f}, ECE: {ece_ortho:.4f}")
    print(
        f"  - Overfit Score: {fit_scores_ortho['overfit_score']:.2f}%, Underfit Score: {fit_scores_ortho['underfit_score']:.2f}%")
    print("=" * 50)

    # 5. Visualizations
    plot_history({"Baseline CNN+GLU": history_baseline, "Ortho CNN+GLU": history_ortho},
                 f"Model Comparison on MNIST ({args.blocks} FFN Blocks)")
    inspect_feature_gates(ortho_model)
    visualize_model_predictions(baseline_model.name, X_val, y_val, baseline_preds)
    visualize_model_predictions(ortho_model.name, X_val, y_val, ortho_preds)