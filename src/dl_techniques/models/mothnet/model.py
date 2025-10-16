"""
MothNet: Bio-Mimetic Feature Generation for Few-Shot Learning.

This module provides an implementation of MothNet, a computational
model of the insect olfactory network designed to excel at machine learning
tasks with limited training data. Its primary function is to serve as a
powerful, automatic feature generator that can be prepended to any standard
ML classifier, creating a hybrid "insect cyborg" model with significantly
enhanced performance.

Core Concept: The "Insect Cyborg"
----------------------------------
The central idea is to address the "limited data" problem by leveraging the
remarkable data efficiency of biological neural networks. Moths can learn to
identify new odors from just a few exposures. MothNet mimics the key
architectural principles that enable this rapid learning, extracting rich,
"orthogonal" class-relevant information that conventional ML methods often miss.

When these bio-mimetic features are concatenated with the original data and
fed into a standard classifier (like an SVM or a simple Neural Net), the
resulting "cyborg" model demonstrates a dramatic reduction in test error
(20-60% reported in original research) and a significant decrease in data
requirements (e.g., matching 100-sample performance with only 30 samples).
"""

import keras
import numpy as np
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.mothnet_blocks import (
    AntennalLobeLayer,
    MushroomBodyLayer,
    HebbianReadoutLayer
)

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="MothNet")
class MothNet(keras.Model):
    """
    Complete MothNet architecture combining AL, MB, and Hebbian readout layers.

    This model implements a bio-mimetic feature generator inspired by the moth
    olfactory network. It transforms input features through three stages that
    mimic insect brain processing: contrast enhancement, high-dimensional sparse
    coding, and associative learning. The model can be used as a standalone
    classifier or as a feature extractor to augment other ML models in the
    "insect cyborg" paradigm.

    **Intent**: Provide a complete, production-ready implementation of the moth
    olfactory network for few-shot learning applications. The model extracts
    features that standard ML methods struggle to find, particularly effective
    when training data is scarce (1-100 samples per class).

    **Architecture**:
    ```
    Input Features (shape=[batch, input_dim])
           ↓
    ┌─────────────────────────────────────┐
    │  Antennal Lobe (AL)                 │
    │  - Competitive inhibition           │
    │  - Contrast enhancement             │
    │  - Shape: [batch, al_units]         │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │  Mushroom Body (MB)                 │
    │  - 20-50x dimensional expansion     │
    │  - Sparse random projection (10%)   │
    │  - Winner-take-all (5-15% active)   │
    │  - Shape: [batch, mb_units]         │
    └─────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────┐
    │  Hebbian Readout                    │
    │  - Correlation-based learning       │
    │  - No backpropagation               │
    │  - Shape: [batch, num_classes]      │
    └─────────────────────────────────────┘
           ↓
    Class Logits or Features for "Cyborg" Augmentation
    ```

    **Usage Paradigms**:

    1. **Standalone Classifier** (Hebbian training):
       ```python
       mothnet = MothNet(num_classes=10, mb_units=4000)
       mothnet.train_hebbian(x_train, y_train_onehot, epochs=5)
       predictions = mothnet.predict(x_test)
       ```

    2. **Feature Extractor for "Cyborg" Models**:
       ```python
       # Train MothNet
       mothnet.train_hebbian(x_train, y_train_onehot, epochs=5)

       # Extract features
       features = mothnet.extract_features(x_data)

       # Augment original features
       x_cyborg = np.concatenate([x_data, features], axis=1)

       # Train conventional ML model on augmented features
       svm = SVC()
       svm.fit(x_cyborg_train, y_train)
       # Typically achieves 20-60% error reduction!
       ```

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    al_units : Optional[int], default=None
        Number of AL units. If None, uses input dimension (no compression).
    mb_units : int, default=2000
        Number of MB units. Typical range: 2000-4000 for 20-50x expansion.
    mb_sparsity : float, default=0.1
        Fraction of MB neurons that fire for any input (5-15% typical).
    connection_sparsity : float, default=0.1
        Sparsity of AL→MB projection (10% non-zero typical).
    hebbian_learning_rate : float, default=0.01
        Learning rate for Hebbian updates (0.001-0.1 typical).
    inhibition_strength : float, default=0.5
        Competitive inhibition strength in AL (0-1 range).
    al_activation : str, default='relu'
        Activation function for AL layer.
    mb_activation : str, default='relu'
        Activation function for MB layer.
    **kwargs
        Additional keyword arguments for the base Model class.

    Input shape
        (batch_size, input_dim): 2D tensor of feature vectors.
        Example: (32, 784) for vectorized MNIST.

    Output shape
        (batch_size, num_classes): Class logits (before softmax).

    Attributes
    ----------
    antennal_lobe : AntennalLobeLayer
        Competitive inhibition layer for contrast enhancement.
    mushroom_body : MushroomBodyLayer
        High-dimensional sparse projection layer.
    readout : HebbianReadoutLayer
        Hebbian learning readout layer.

    Example
    -------
    >>> # Create MothNet for MNIST-like task (10 classes, 784 input features)
    >>> model = MothNet(
    ...     num_classes=10,
    ...     al_units=85,  # Compress to 85 features
    ...     mb_units=4000,  # 47x expansion
    ...     mb_sparsity=0.1,  # 400 active neurons per sample
    ...     hebbian_learning_rate=0.01
    ... )
    >>>
    >>> # Build model
    >>> model.build((None, 784))
    >>> print(f"Parameters: {model.count_params():,}")
    >>>
    >>> # Train with Hebbian learning (requires one-hot labels)
    >>> y_train_onehot = keras.utils.to_categorical(y_train, 10)
    >>> history = model.train_hebbian(
    ...     x_train, y_train_onehot,
    ...     epochs=5, batch_size=32
    ... )
    >>>
    >>> # Use as feature extractor for "cyborg" models
    >>> train_features = model.extract_features(x_train)
    >>> x_cyborg = np.concatenate([x_train, train_features], axis=1)
    >>>
    >>> # Train SVM on augmented features
    >>> from sklearn.svm import SVC
    >>> svm = SVC(kernel='rbf')
    >>> svm.fit(x_cyborg, y_train)

    Notes
    -----
    **Key Performance Insights from Original Research**:

    1. **Error Reduction**: "Cyborg" models (ML + MothNet features) achieve
       20-60% relative error reduction compared to baseline ML models.

    2. **Data Efficiency**: A cyborg trained on 30 samples/class can match
       the accuracy of a baseline trained on 100 samples/class (>3x reduction
       in data requirements).

    3. **Orthogonal Information**: MothNet extracts features that are
       complementary to what norm-based ML methods find, explaining why
       simple concatenation is so effective.

    4. **Superiority Over Alternatives**: MothNet features outperform PCA,
       PLS, and standard neural network features for augmentation.

    **When to Use MothNet**:

    - ✓ Limited training data (1-100 samples per class)
    - ✓ Need for rapid learning from few examples
    - ✓ High-dimensional input (images, spectra, omics data)
    - ✓ Want to boost existing ML models without retraining from scratch
    - ✗ Abundant training data (deep learning may be better)
    - ✗ Need for online/incremental learning (Hebbian learning is batch-based)
    """

    def __init__(
        self,
        num_classes: int,
        al_units: Optional[int] = None,
        mb_units: int = 2000,
        mb_sparsity: float = 0.1,
        connection_sparsity: float = 0.1,
        hebbian_learning_rate: float = 0.01,
        inhibition_strength: float = 0.5,
        al_activation: str = 'relu',
        mb_activation: str = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.al_units = al_units
        self.mb_units = mb_units
        self.mb_sparsity = mb_sparsity
        self.connection_sparsity = connection_sparsity
        self.hebbian_learning_rate = hebbian_learning_rate
        self.inhibition_strength = inhibition_strength
        self.al_activation = al_activation
        self.mb_activation = mb_activation

        # Sub-layers will be initialized in build()
        self.antennal_lobe = None
        self.mushroom_body = None
        self.readout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create and build all sub-layers.

        CRITICAL: Explicitly builds each sub-layer for robust serialization.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        al_units = self.al_units if self.al_units is not None else input_dim

        # Create sub-layers
        self.antennal_lobe = AntennalLobeLayer(
            units=al_units,
            inhibition_strength=self.inhibition_strength,
            activation=self.al_activation,
            name='antennal_lobe'
        )

        self.mushroom_body = MushroomBodyLayer(
            units=self.mb_units,
            sparsity=self.mb_sparsity,
            connection_sparsity=self.connection_sparsity,
            activation=self.mb_activation,
            trainable_projection=False,
            name='mushroom_body'
        )

        self.readout = HebbianReadoutLayer(
            units=self.num_classes,
            learning_rate=self.hebbian_learning_rate,
            name='hebbian_readout'
        )

        # Explicitly build sub-layers in computational order
        self.antennal_lobe.build(input_shape)

        al_output_shape = self.antennal_lobe.compute_output_shape(input_shape)
        self.mushroom_body.build(al_output_shape)

        mb_output_shape = self.mushroom_body.compute_output_shape(al_output_shape)
        self.readout.build(mb_output_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through MothNet (AL → MB → Readout).

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Whether the model is in training mode.

        Returns
        -------
        keras.KerasTensor
            Class logits of shape (batch_size, num_classes).
        """
        # Stage 1: Competitive inhibition (contrast enhancement)
        al_output = self.antennal_lobe(inputs, training=training)

        # Stage 2: High-dimensional sparse projection
        mb_output = self.mushroom_body(al_output, training=training)

        # Stage 3: Hebbian readout
        output = self.readout(mb_output, training=training)

        return output

    def extract_features(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Extract MothNet features (readout activations) for "cyborg" augmentation.

        This method is used to generate features that can be concatenated with
        original inputs to create augmented feature vectors for conventional ML.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        keras.KerasTensor
            Feature tensor of shape (batch_size, num_classes).
        """
        return self(inputs, training=False)

    def extract_mb_features(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Extract Mushroom Body activations (sparse codes before readout).

        Useful for analysis or using MB representations directly as features.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        keras.KerasTensor
            MB feature tensor of shape (batch_size, mb_units) with sparse activations.
        """
        al_output = self.antennal_lobe(inputs, training=False)
        mb_output = self.mushroom_body(al_output, training=False)
        return mb_output

    def train_hebbian(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, list]:
        """
        Train the model using Hebbian learning.

        This is the primary training method for MothNet. Unlike standard Keras
        training, this uses the Hebbian update rule for the readout layer instead
        of backpropagation.

        Parameters
        ----------
        x : np.ndarray
            Training data of shape (num_samples, input_dim).
        y : np.ndarray
            Training labels of shape (num_samples, num_classes).
            MUST be one-hot encoded. Use keras.utils.to_categorical(labels, num_classes).
        epochs : int, default=1
            Number of training epochs. In the original research, 1-5 epochs
            were typically sufficient.
        batch_size : int, default=32
            Batch size for training. Affects update frequency and memory usage.
        verbose : int, default=1
            Verbosity level:
            - 0: Silent
            - 1: Progress bar with loss per epoch
            - 2: One line per epoch

        Returns
        -------
        dict
            Training history with 'loss' key containing list of epoch losses.

        Example
        -------
        >>> # Prepare one-hot labels
        >>> y_train_onehot = keras.utils.to_categorical(y_train, num_classes=10)
        >>>
        >>> # Train MothNet
        >>> history = model.train_hebbian(
        ...     x_train, y_train_onehot,
        ...     epochs=5, batch_size=32, verbose=1
        ... )
        >>>
        >>> print(f"Final loss: {history['loss'][-1]:.4f}")

        Notes
        -----
        The loss computed here (cross-entropy) is for monitoring only. Weight
        updates are performed via the Hebbian rule, not gradient descent on
        this loss. The loss should generally decrease as the Hebbian associations
        strengthen, but it's not being directly optimized.
        """
        num_samples = x.shape[0]
        history = {'loss': []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Shuffle data each epoch
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            # Mini-batch Hebbian training
            for i in range(0, num_samples, batch_size):
                batch_x = x_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]

                # Convert to tensors
                batch_x_tensor = keras.ops.convert_to_tensor(batch_x)
                batch_y_tensor = keras.ops.convert_to_tensor(batch_y)

                # Forward pass through AL and MB to get pre-synaptic activations
                al_output = self.antennal_lobe(batch_x_tensor, training=True)
                mb_output = self.mushroom_body(al_output, training=True)

                # Apply Hebbian update to readout weights
                self.readout.hebbian_update(mb_output, batch_y_tensor)

                # Compute loss for monitoring (cross-entropy)
                logits = self.readout(mb_output, training=True)
                loss = keras.ops.mean(
                    keras.losses.categorical_crossentropy(
                        batch_y_tensor, logits, from_logits=True
                    )
                )
                epoch_loss += keras.ops.convert_to_numpy(loss)
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)

            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

        return history

    def get_config(self) -> Dict[str, Any]:
        """
        Return model configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'al_units': self.al_units,
            'mb_units': self.mb_units,
            'mb_sparsity': self.mb_sparsity,
            'connection_sparsity': self.connection_sparsity,
            'hebbian_learning_rate': self.hebbian_learning_rate,
            'inhibition_strength': self.inhibition_strength,
            'al_activation': self.al_activation,
            'mb_activation': self.mb_activation,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MothNet':
        """
        Create model from configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary from get_config().

        Returns
        -------
        MothNet
            New model instance with the specified configuration.
        """
        return cls(**config)


# Utility Functions

def create_cyborg_features(
    mothnet: MothNet,
    x_data: np.ndarray
) -> np.ndarray:
    """
    Create augmented 'cyborg' features by concatenating original features with MothNet outputs.

    This is the key operation for creating "insect cyborg" models that combine
    conventional ML with bio-mimetic feature extraction.

    Parameters
    ----------
    mothnet : MothNet
        Trained MothNet model (must have been trained via train_hebbian).
    x_data : np.ndarray
        Original feature data of shape (num_samples, input_dim).

    Returns
    -------
    np.ndarray
        Augmented features of shape (num_samples, input_dim + num_classes).
        First input_dim columns are original features, last num_classes columns
        are MothNet readout activations.

    Example
    -------
    >>> # Train MothNet
    >>> mothnet.train_hebbian(x_train, y_train_onehot, epochs=5)
    >>>
    >>> # Create cyborg features for train and test sets
    >>> x_train_cyborg = create_cyborg_features(mothnet, x_train)
    >>> x_test_cyborg = create_cyborg_features(mothnet, x_test)
    >>>
    >>> # Train conventional ML on augmented features
    >>> from sklearn.svm import SVC
    >>> svm = SVC()
    >>> svm.fit(x_train_cyborg, y_train)
    >>> accuracy = svm.score(x_test_cyborg, y_test)
    >>> print(f"Cyborg accuracy: {accuracy:.3f}")

    Notes
    -----
    The original research showed that simple concatenation of MothNet features
    with original features consistently improves accuracy across multiple ML
    methods (SVM, kNN, Neural Networks). This suggests MothNet extracts
    complementary information not captured by standard ML feature spaces.
    """
    # Extract MothNet features
    mothnet_features = keras.ops.convert_to_numpy(
        mothnet.extract_features(x_data)
    )

    # Concatenate with original features
    cyborg_features = np.concatenate([x_data, mothnet_features], axis=1)

    return cyborg_features


