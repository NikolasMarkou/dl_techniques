import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

from dl_techniques.layers.layer_scale import LearnableMultiplier, LayerScale
from dl_techniques.regularizers.binary_preference import BinaryPreferenceRegularizer

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic dataset with informative and redundant features
# 2 informative features, 18 noise features, 4 classes
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    n_classes=4,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize the data
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


# Build model with LayerScale
def build_model_with_layerscale():
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        # LayerScale with very small initial values to suppress most features
        LayerScale(init_values=0.01, projection_dim=20),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Build baseline model without LayerScale
def build_baseline_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(20,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Train the models
layerscale_model = build_model_with_layerscale()
history_layerscale = layerscale_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

baseline_model = build_baseline_model()
history_baseline = baseline_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Get the learned scaling factors from LayerScale
layerscale_layer = layerscale_model.layers[0]
gamma_weights = layerscale_layer.gamma.numpy()

# Evaluate models
layerscale_preds = np.argmax(layerscale_model.predict(X_test), axis=1)
baseline_preds = np.argmax(baseline_model.predict(X_test), axis=1)

layerscale_acc = accuracy_score(y_test, layerscale_preds)
baseline_acc = accuracy_score(y_test, baseline_preds)

print(f"LayerScale Model Accuracy: {layerscale_acc:.4f}")
print(f"Baseline Model Accuracy: {baseline_acc:.4f}")

# Visualize the learned gamma weights
feature_indices = np.arange(20)
feature_types = ['Informative' if i < 2 else 'Redundant' if i < 4 else 'Noise' for i in range(20)]
feature_colors = ['green' if t == 'Informative' else 'orange' if t == 'Redundant' else 'red' for t in feature_types]

plt.figure(figsize=(14, 6))
bars = plt.bar(feature_indices, gamma_weights, color=feature_colors)
plt.xlabel('Feature Index')
plt.ylabel('Learned Scaling Factor (γ)')
plt.title('LayerScale Learned Feature Importance')
plt.xticks(feature_indices)

# Add feature type annotations
for i, (bar, ftype) in enumerate(zip(bars, feature_types)):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{ftype[0]}', ha='center', va='bottom', fontweight='bold')
    plt.text(bar.get_x() + bar.get_width() / 2., height / 2,
             f'{height:.3f}', ha='center', va='center', color='white', fontweight='bold')

plt.axhline(y=0.01, color='black', linestyle='--', label='Initial Value')
plt.legend()

# Top 5 most important features according to LayerScale
top_features = np.argsort(gamma_weights)[::-1][:5]
print(f"Top 5 most important features: {top_features}")
print(f"Their scaling factors: {gamma_weights[top_features]}")

# Create a 2D visualization of the most important features
if len(top_features) >= 2:
    top_two = top_features[:2]
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        X_test[:, top_two[0]],
        X_test[:, top_two[1]],
        c=y_test,
        cmap='viridis',
        s=50,
        alpha=0.7,
        edgecolors='k'
    )

    plt.xlabel(f'Feature {top_two[0]} (γ={gamma_weights[top_two[0]]:.3f})')
    plt.ylabel(f'Feature {top_two[1]} (γ={gamma_weights[top_two[1]]:.3f})')
    plt.title('2D Visualization of Top Features Selected by LayerScale')
    plt.colorbar(scatter, label='Class')
    plt.grid(alpha=0.3)