import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import argparse

from dl_techniques.layers.ffn.orthoglu_ffn import OrthoGLUFFN
from dl_techniques.layers.stochastic_gradient import StochasticGradient


# ---------------------------------------------------------------------
# 1. Data Generation and Preparation
# ---------------------------------------------------------------------

def get_dataset(n_samples=50000, n_features=128, n_classes=10):
    """
    Generates a challenging synthetic classification dataset with the specified signature.
    The composition of features (informative, redundant, noise) is derived internally.
    """
    # 1. Internal feature composition is defined here.
    n_informative = 20
    n_redundant = 60
    n_base_features = n_informative + n_redundant

    # 2. Calculate the number of noise features based on the total n_features.
    n_noise_features = n_features - n_base_features

    # 3. Input validation: Ensure the total features can accommodate the base features.
    if n_noise_features < 0:
        raise ValueError(
            f"n_features ({n_features}) must be greater than or equal to the sum "
            f"of informative and redundant features ({n_base_features})."
        )

    print(f"\n--- Generating dataset with {n_features} total features ({n_classes} classes) ---")
    print(f"    - Informative features: {n_informative}")
    print(f"    - Redundant (correlated) features: {n_redundant}")
    print(f"    - Derived noise features: {n_noise_features}")

    # 4. Generate only the structured (non-noise) features.
    X_base, y = make_classification(
        n_samples=n_samples,
        n_features=n_base_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.00,
        random_state=42,
    )

    # 5. Generate and concatenate the noise features if required.
    if n_noise_features > 0:
        noise = np.random.randn(n_samples, n_noise_features)
        X_combined = np.concatenate([X_base, noise], axis=1)

        # CRITICAL: Shuffle the feature columns to mix noise with structured features.
        feature_indices = np.arange(X_combined.shape[1])
        np.random.seed(42)  # For reproducible shuffle
        np.random.shuffle(feature_indices)
        X_final = X_combined[:, feature_indices]
    else:
        # If no noise features are requested, use the base data directly.
        X_final = X_base

    # 6. Split the final dataset.
    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    return (X_train, y_train), (X_val, y_val)


# ---------------------------------------------------------------------
# 2. Model Architectures
# ---------------------------------------------------------------------
class BaselineGLUFFN(keras.layers.Layer):
    """A standard GLU FeedForward network using Dense layers."""

    def __init__(self, hidden_dim, output_dim, activation="gelu", dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.input_proj = keras.layers.Dense(hidden_dim * 2, name="input_proj_dense")
        self.output_proj = keras.layers.Dense(output_dim, name="output_proj_dense")
        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        gate_and_value = self.input_proj(inputs)
        gate, value = tf.split(gate_and_value, num_or_size_splits=2, axis=-1)
        gated_value = self.activation(gate) * value
        gated_value = self.dropout(gated_value, training=training)
        return self.output_proj(gated_value)


def build_baseline_model(input_shape, num_classes, num_blocks, hidden_dim=256, dropout_rate=0.1):
    """Builds a model using a configurable number of stacked standard GLU FFNs."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i in range(num_blocks):
        x = BaselineGLUFFN(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            name=f"glu_block_{i + 1}"
        )(x)
        x = keras.layers.LayerNormalization(name=f"ln_{i + 1}")(x)
        x = StochasticGradient(drop_path_rate=0.25)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name=f"Baseline_GLU_{num_blocks}_Blocks")
    return model


def build_ortho_model(input_shape, num_classes, num_blocks, hidden_dim=256, dropout_rate=0.1, ortho_reg_factor=0.01):
    """Builds a model using a configurable number of stacked OrthoGLUFFNs."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i in range(num_blocks):
        x = OrthoGLUFFN(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout_rate=dropout_rate,
            ortho_reg_factor=ortho_reg_factor,
            name=f"ortho_glu_block_{i + 1}"
        )(x)
        x = StochasticGradient(drop_path_rate=0.25)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name=f"OrthoGLUFFN_{num_blocks}_Blocks")
    return model


# ---------------------------------------------------------------------
# 3. Training and Evaluation Utilities (Augmented)
# ---------------------------------------------------------------------
def calculate_ece(y_true, y_pred_probs, n_bins=15):
    """Calculates the Expected Calibration Error (ECE)."""
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calculate_fit_scores(history):
    """Calculates overfitting and underfitting scores from training history."""
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    # Overfit score: The gap between training and validation accuracy. Higher means more overfitting.
    overfit_score = (final_train_acc - final_val_acc) * 100

    # Underfit score: The model's final error rate on validation data. Higher means more underfitting.
    underfit_score = (1.0 - final_val_acc) * 100

    return {"overfit_score": overfit_score, "underfit_score": underfit_score}


def plot_history(histories, title="Model Training Comparison"):
    """Plots training and validation accuracy/loss for given histories."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for name, history in histories.items():
        ax1.plot(history.history['val_accuracy'], label=f'{name} Val Acc', linestyle='--')
        ax2.plot(history.history['val_loss'], label=f'{name} Val Loss', linestyle='--')
    ax1.set_title('Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    fig.suptitle(title, fontsize=16)
    plt.show()


def inspect_all_feature_gates(ortho_model, num_blocks):
    """
    Inspects and visualizes the learned scaling factors (feature gates)
    from ALL OrthoGLUFFN blocks in the model.
    """
    print("\n--- Inspecting Learned Feature Gates from All OrthoGLUFFN Blocks ---")

    # Create a figure with subplots for each block
    if num_blocks == 0:
        print("No OrthoGLUFFN blocks to inspect.")
        return

    fig, axes = plt.subplots(
        nrows=num_blocks,
        ncols=1,
        figsize=(10, 4 * num_blocks),
        squeeze=False  # Always return a 2D array for axes
    )
    axes = axes.flatten()  # Flatten to 1D for easy iteration

    all_gates_found = True
    for i in range(num_blocks):
        block_name = f"ortho_glu_block_{i + 1}"
        ax = axes[i]
        try:
            # Systematically access each block by its name
            ortho_glu_layer = ortho_model.get_layer(block_name)
            gate_weights = ortho_glu_layer.input_proj_ortho.constrained_scale.get_weights()[0]

            print(f"\n[Block: {block_name}]")
            print(f"  - Shape of gate weights: {gate_weights.shape}")
            print(
                f"  - Min: {np.min(gate_weights):.4f}, Max: {np.max(gate_weights):.4f}, Mean: {np.mean(gate_weights):.4f}")
            closed_gates = np.sum(gate_weights < 0.1)
            print(f"  - Gates effectively closed (< 0.1): {closed_gates} / {len(gate_weights)}")

            # Plot the histogram on the corresponding subplot
            ax.hist(gate_weights, bins=50, color='skyblue', edgecolor='black', range=(0, 1))
            ax.set_title(f"Distribution of Gate Values in {block_name}, (Constrained to [0, 1])")
            ax.set_ylabel("Frequency")
            ax.grid(True)

        except Exception as e:
            print(f"Could not inspect weights for {block_name}. Error: {e}")
            ax.set_title(f"Could not retrieve weights for {block_name}")
            ax.axis('off')
            all_gates_found = False

    if all_gates_found:
        fig.suptitle("Feature Gate Distributions Across All OrthoGLUFFN Blocks", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare GLU FFN models.")
    parser.add_argument("--blocks", type=int, default=4, help="Number of GLU/OrthoGLU blocks to stack.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples in the dataset.")
    parser.add_argument("--n_features", type=int, default=128, help="Number of features in the dataset.")
    parser.add_argument("--n_classes", type=int, default=40, help="Number of classes for classification.")
    parser.add_argument("--dropout_rate", type=float, default=0.25, help="Dropout rate.")
    parser.add_argument("--ortho_reg_factor", type=float, default=1.0, help="Orthogonal regularization factor.")
    args = parser.parse_args()

    # 1. Load Data
    (X_train, y_train), (X_val, y_val) = get_dataset(
        n_samples=args.n_samples, n_features=args.n_features, n_classes=args.n_classes
    )
    input_shape = (X_train.shape[1],)

    # 2. Build Models
    baseline_model = build_baseline_model(input_shape, args.n_classes, num_blocks=args.blocks,
                                          dropout_rate=args.dropout_rate)
    ortho_model = build_ortho_model(input_shape, args.n_classes, num_blocks=args.blocks,
                                    ortho_reg_factor=args.ortho_reg_factor, dropout_rate=args.dropout_rate)

    loss_fn = "sparse_categorical_crossentropy"

    baseline_model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3), loss=loss_fn, metrics=["accuracy"])
    ortho_model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-3), loss=loss_fn, metrics=["accuracy"])

    print("\n--- Baseline Model Summary ---");
    baseline_model.summary()
    print("\n--- OrthoGLUFFN Model Summary ---");
    ortho_model.summary()

    # 3. Train Models
    print(f"\n--- Training Models for {args.epochs} epochs with {args.blocks} blocks ---")

    print(f"\n--- Training ortho FFN model")
    history_ortho = ortho_model.fit(
        X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, y_val), verbose=2
    )

    print(f"\n--- Training baseline FFN model")
    history_baseline = baseline_model.fit(
        X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_val, y_val), verbose=2
    )

    # 4. Final Performance Analysis
    print("\n--- Generating Final Performance Metrics ---")
    baseline_preds = baseline_model.predict(X_val, batch_size=args.batch_size)
    ortho_preds = ortho_model.predict(X_val, batch_size=args.batch_size)

    ece_baseline = calculate_ece(y_val, baseline_preds)
    ece_ortho = calculate_ece(y_val, ortho_preds)

    fit_scores_baseline = calculate_fit_scores(history_baseline)
    fit_scores_ortho = calculate_fit_scores(history_ortho)

    print("\n" + "=" * 50)
    print(" " * 15 + "PERFORMANCE REPORT")
    print("=" * 50)
    print(f"\n[Baseline Model: {baseline_model.name}]")
    print(f"  - Final Validation Accuracy: {history_baseline.history['val_accuracy'][-1]:.4f}")
    print(f"  - ECE Score:                 {ece_baseline:.4f} (Lower is better)")
    print(f"  - Overfit Score:             {fit_scores_baseline['overfit_score']:.2f}% (Train > Val Acc Gap)")
    print(f"  - Underfit Score (Val Error):{fit_scores_baseline['underfit_score']:.2f}% (100% - Val Acc)")

    print(f"\n[OrthoGLUFFN Model: {ortho_model.name}]")
    print(f"  - Final Validation Accuracy: {history_ortho.history['val_accuracy'][-1]:.4f}")
    print(f"  - ECE Score:                 {ece_ortho:.4f} (Lower is better)")
    print(f"  - Overfit Score:             {fit_scores_ortho['overfit_score']:.2f}% (Train > Val Acc Gap)")
    print(f"  - Underfit Score (Val Error):{fit_scores_ortho['underfit_score']:.2f}% (100% - Val Acc)")
    print("=" * 50)

    # 5. Visualizations
    plot_history(
        {"Baseline GLU": history_baseline, "OrthoGLUFFN": history_ortho},
        title=f"Model Comparison ({args.blocks} Blocks) on High-Redundancy Data"
    )
    # MODIFIED: Call the new function to inspect all blocks.
    inspect_all_feature_gates(ortho_model, args.blocks)
