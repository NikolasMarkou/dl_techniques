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

Key Architectural Components
----------------------------
The MothNet architecture is a feed-forward cascade of three specialized layers,
each inspired by a specific region of the insect brain:

1.  **AntennalLobeLayer (AL)**: Contrast Enhancement
    -   **Biological Analogy**: The Antennal Lobe, the first olfactory
        processing center.
    -   **Mechanism**: Implements competitive inhibition, where neurons suppress
        each other's activity. This creates a "winner-take-more" dynamic that
        sharpens the input signal, enhances contrast, and suppresses noise.

2.  **MushroomBodyLayer (MB)**: High-Dimensional Sparse Coding
    -   **Biological Analogy**: The Mushroom Body, a center for associative
        learning and memory.
    -   **Mechanism**: Projects the sharpened signal from the AL into a much
        higher-dimensional space using sparse, random connections. It then
        enforces sparse firing via a top-k winner-take-all mechanism. This
        transformation untangles complex patterns and creates unique, robust,
        and highly discriminative combinatorial codes for each input.

3.  **HebbianReadoutLayer**: Associative Learning
    -   **Biological Analogy**: Synaptic plasticity in readout neurons.
    -   **Mechanism**: Uses a local, correlation-based Hebbian learning rule
        ("fire together, wire together") instead of backpropagation. Weights
        are strengthened based on the co-occurrence of pre-synaptic (MB) and
        post-synaptic (target class) activity. This forms direct associations
        between the sparse MB codes and their corresponding classes.

Usage Paradigms
---------------

1.  **As a Feature Extractor for "Cyborg" Models (Primary Use Case)**:
    This is the most powerful way to use MothNet. The model is first trained
    using its built-in Hebbian learning, then used to generate features for an
    external, conventional ML model.

    ```python
    from sklearn.svm import SVC
    import numpy as np

    # 1. Initialize and train MothNet
    mothnet = MothNet(num_classes=10, mb_units=4000)
    # Note: y_train must be one-hot encoded for Hebbian training
    mothnet.train_hebbian(x_train, y_train_onehot, epochs=5)

    # 2. Create augmented "cyborg" features
    x_train_cyborg = create_cyborg_features(mothnet, x_train)
    x_test_cyborg = create_cyborg_features(mothnet, x_test)

    # 3. Train a conventional ML model on the augmented data
    svm = SVC()
    svm.fit(x_train_cyborg, y_train)
    accuracy = svm.score(x_test_cyborg, y_test)
    print(f"Cyborg SVM Accuracy: {accuracy:.4f}")
    ```

2.  **As a Standalone Classifier**:
    The model can also be used directly for classification after Hebbian training.
    Its performance is strong, but the "cyborg" approach is typically superior as
    it combines the strengths of both biological and statistical learning.

    ```python
    # After Hebbian training (as above)...
    logits = mothnet.predict(x_test)
    predictions = np.argmax(logits, axis=1)
    ```

When to Use This Module
-----------------------
-   **Primary Target**: Classification tasks with **limited training data**
    (e.g., 1 to 100 samples per class).
-   **Problem Domain**: Ideal for high-dimensional, unstructured data like
    vectorized images, sensor readings, or scientific measurements where
    feature engineering is challenging.
-   **Goal**: To significantly boost the performance of existing ML pipelines
    or to enable effective learning where it was previously impossible due
    to data scarcity.
"""

import keras
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="MothNet")
class AntennalLobeLayer(keras.layers.Layer):
    """
    Antennal Lobe layer implementing competitive inhibition for contrast enhancement.

    This layer enhances contrast by having each neuron excite itself while inhibiting
    all other neurons through global inhibition. This creates a "winner-take-more"
    dynamic that sharpens the input signal and suppresses noise, mimicking the
    competitive dynamics observed in the insect antennal lobe.

    **Intent**: Model the first stage of olfactory processing in moths, where
    receptor signals are sharpened through competitive inhibition, creating a
    cleaner and more robust representation for downstream processing.

    **Architecture**:
    ```
    Input Features (shape=[batch, features])
           ↓
    Linear Transform: h = W·x + b
           ↓
    Compute Global Mean: μ = mean(h)
           ↓
    Lateral Inhibition: h_inhibited = h - α·μ
           ↓
    Activation: output = activation(h_inhibited)
           ↓
    Output (shape=[batch, units])
    ```

    **Mathematical Operations**:
    1. **Excitation**: h = W·x + b (learnable linear transformation)
    2. **Global Inhibition**: μ = (1/N)·∑h_i (mean activity across neurons)
    3. **Competitive Response**: y = activation(h - α·μ)

    Where α controls inhibition strength (higher values = stronger competition).

    Parameters
    ----------
    units : int
        Number of output units (typically same as input dimension).
    inhibition_strength : float, default=0.5
        Strength of lateral inhibition (α). Higher values create stronger
        competitive dynamics. Range: [0, 1], where 0 = no inhibition,
        1 = maximum competition.
    activation : str, default='relu'
        Activation function to apply after inhibition. Common choices:
        'relu', 'gelu', 'swish'.
    kernel_initializer : str or keras.initializers.Initializer, default='glorot_uniform'
        Initializer for the excitatory weight matrix.
    kernel_regularizer : Optional[keras.regularizers.Regularizer], default=None
        Regularizer function applied to excitatory weights.
    use_bias : bool, default=True
        Whether to include bias terms in the transformation.
    **kwargs
        Additional keyword arguments for the base Layer class.

    Input shape
        (batch_size, input_dim): 2D tensor of feature vectors.

    Output shape
        (batch_size, units): 2D tensor after competitive inhibition.

    Attributes
    ----------
    excitatory_weights : keras.Variable
        Weight matrix for excitatory connections, shape (input_dim, units).
    bias : keras.Variable, optional
        Bias vector, shape (units,), present if use_bias=True.

    Example
    -------
    >>> # Create AL layer for 85-dimensional input
    >>> al_layer = AntennalLobeLayer(
    ...     units=85,
    ...     inhibition_strength=0.5,
    ...     activation='relu'
    ... )
    >>> x = keras.ops.ones((32, 85))  # Batch of 32 samples
    >>> output = al_layer(x)
    >>> print(output.shape)  # (32, 85)

    Notes
    -----
    The competitive inhibition mechanism serves two purposes:
    1. **Contrast Enhancement**: Strong features become relatively stronger
    2. **Noise Suppression**: Weak, noisy features are suppressed below threshold

    This is analogous to contrast enhancement in image processing, but applied
    to abstract feature spaces.
    """

    def __init__(
        self,
        units: int,
        inhibition_strength: float = 0.5,
        activation: str = 'relu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.inhibition_strength = inhibition_strength
        self.activation_fn = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create layer weights for excitatory connections.

        This is called automatically when the layer first processes input.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Create excitatory connection weights
        self.excitatory_weights = self.add_weight(
            name='excitatory_weights',
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply competitive inhibition to inputs.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Whether the layer is in training mode (unused, for API compatibility).

        Returns
        -------
        keras.KerasTensor
            Output tensor of shape (batch_size, units) after competitive inhibition.
        """
        # Excitatory response: linear transformation
        excitation = keras.ops.matmul(inputs, self.excitatory_weights)

        if self.use_bias:
            excitation = excitation + self.bias

        # Global inhibition: subtract scaled mean activity
        mean_activity = keras.ops.mean(excitation, axis=-1, keepdims=True)
        inhibition = mean_activity * self.inhibition_strength

        # Combined competitive response
        output = excitation - inhibition

        # Apply nonlinearity
        output = self.activation_fn(output)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape for this layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Output shape (batch_size, units).
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'inhibition_strength': self.inhibition_strength,
            'activation': keras.activations.serialize(self.activation_fn),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'use_bias': self.use_bias,
        })
        return config


@keras.saving.register_keras_serializable(package="MothNet")
class MushroomBodyLayer(keras.layers.Layer):
    """
    Mushroom Body layer implementing high-dimensional sparse random projection.

    This layer projects inputs into a much higher-dimensional space using sparse,
    random connections (mimicking AL→MB connections in insects), then enforces
    sparse firing through top-k winner-take-all selection. This creates a
    combinatorial code that is both robust to noise and highly discriminative,
    similar to the sparse coding observed in insect mushroom bodies.

    **Intent**: Implement the critical expansion and sparsification stage of the
    moth olfactory network, where low-dimensional odor representations are
    transformed into high-dimensional, sparse codes that enable rapid learning
    with minimal examples.

    **Architecture**:
    ```
    Input from AL (shape=[batch, input_dim])
           ↓
    Sparse Random Projection: h = W_sparse·x + b
           │ (connection_sparsity = 10% non-zero)
           │ (expansion: input_dim → units, typically 20-50x)
           ↓
    Activation: h_active = activation(h)
           ↓
    Top-K Selection (WTA): select k = sparsity·units largest values
           ↓
    Sparse Output: only k neurons fire, rest = 0
           ↓
    Output to Readout (shape=[batch, units], ~90% zeros)
    ```

    **Mathematical Operations**:
    1. **Sparse Projection**: h = W_sparse·x + b
       - W_sparse has only 10% non-zero entries (fixed random pattern)
       - Dimensionality expansion: units >> input_dim

    2. **Sparse Coding**: output = TopK(activation(h), k)
       - k = ⌊sparsity × units⌋ (typically 5-15% of neurons)
       - Creates combinatorial representation: C(units, k) possible patterns

    3. **Information Capacity**: With 4000 neurons and 10% sparsity:
       - C(4000, 400) ≈ 10^459 possible unique patterns
       - Far exceeds typical classification needs

    Parameters
    ----------
    units : int
        Number of output units (typically 20-50x input dimension).
        Example: For 85 AL neurons, use 2000-4000 MB neurons.
    sparsity : float, default=0.1
        Fraction of neurons that fire for any input (k/units).
        Range: (0, 1), typical values: 0.05-0.15.
    connection_sparsity : float, default=0.1
        Sparsity of the random projection matrix (fraction of non-zero weights).
        Range: (0, 1), typical value: 0.1.
    activation : str, default='relu'
        Activation function before top-k selection.
    kernel_initializer : str or keras.initializers.Initializer, default='glorot_uniform'
        Initializer for the non-zero projection weights.
    trainable_projection : bool, default=False
        Whether projection matrix is trainable. In biological systems, these
        connections are random and fixed.
    use_bias : bool, default=True
        Whether to include bias terms.
    **kwargs
        Additional keyword arguments for the base Layer class.

    Input shape
        (batch_size, input_dim): 2D tensor from Antennal Lobe.

    Output shape
        (batch_size, units): Sparse 2D tensor with only k active neurons per sample.

    Attributes
    ----------
    projection_weights : keras.Variable
        Sparse random projection matrix, shape (input_dim, units).
    projection_mask : np.ndarray
        Binary mask defining which connections exist (fixed at initialization).
    k : int
        Number of neurons kept active per sample (k = ⌊sparsity × units⌋).
    bias : keras.Variable, optional
        Bias vector, shape (units,), present if use_bias=True.

    Example
    -------
    >>> # Create MB layer with 50x expansion and 10% sparsity
    >>> mb_layer = MushroomBodyLayer(
    ...     units=4000,  # 50x expansion from 80 AL neurons
    ...     sparsity=0.1,  # 400 active neurons per input
    ...     connection_sparsity=0.1  # Sparse random connections
    ... )
    >>> x = keras.ops.ones((32, 80))  # Batch from AL
    >>> output = mb_layer(x)
    >>> print(output.shape)  # (32, 4000)
    >>> # Count non-zeros per sample
    >>> active = keras.ops.sum(keras.ops.cast(output > 0, 'int32'), axis=1)
    >>> print(active)  # Each sample has ~400 active neurons

    Notes
    -----
    **Why High-Dimensional Sparse Coding Works**:

    1. **Separation**: High dimensions untangle complex, non-linearly separable
       patterns (similar to kernel trick in SVMs).

    2. **Sparsity**: Enforces selectivity - each active neuron becomes a
       specialized detector for specific feature combinations.

    3. **Combinatorial Power**: With n units and k active, there are C(n,k)
       possible patterns, providing massive representational capacity.

    4. **Noise Robustness**: Sparse codes are resistant to corruption since
       pattern identity depends on which neurons fire, not exact values.

    The combination is particularly powerful for few-shot learning because
    each example gets a unique, discriminative representation that standard
    ML methods can easily separate.
    """

    def __init__(
        self,
        units: int,
        sparsity: float = 0.1,
        connection_sparsity: float = 0.1,
        activation: str = 'relu',
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        trainable_projection: bool = False,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.sparsity = sparsity
        self.connection_sparsity = connection_sparsity
        self.activation_fn = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.trainable_projection = trainable_projection
        self.use_bias = use_bias
        self.k = int(units * sparsity)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create sparse random projection matrix.

        This is called automatically when the layer first processes input.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        # Initialize full weight matrix
        weights = keras.ops.convert_to_numpy(
            self.kernel_initializer(shape=(input_dim, self.units))
        )

        # Create sparse connection mask (fixed random pattern)
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        self.projection_mask = rng.rand(input_dim, self.units) < self.connection_sparsity

        # Apply sparsity mask
        weights = weights * self.projection_mask

        self.projection_weights = self.add_weight(
            name='projection_weights',
            shape=(input_dim, self.units),
            initializer=keras.initializers.Constant(weights),
            trainable=self.trainable_projection,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply sparse high-dimensional projection with winner-take-all.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim).
        training : bool, optional
            Whether the layer is in training mode (unused, for API compatibility).

        Returns
        -------
        keras.KerasTensor
            Sparse output tensor of shape (batch_size, units) with only top-k
            neurons active per sample.
        """
        # Apply sparse random projection
        output = keras.ops.matmul(inputs, self.projection_weights)

        if self.use_bias:
            output = output + self.bias

        # Apply activation
        output = self.activation_fn(output)

        # Enforce sparse firing via top-k winner-take-all
        top_k_values, top_k_indices = keras.ops.top_k(output, k=self.k)

        # Create sparse output (only top-k neurons fire, rest are zero)
        sparse_output = keras.ops.zeros_like(output)

        # Scatter top-k values back to their positions
        batch_size = keras.ops.shape(output)[0]
        batch_indices = keras.ops.repeat(
            keras.ops.arange(batch_size)[:, None],
            repeats=self.k,
            axis=1
        )

        # Stack batch and feature indices for scatter operation
        indices = keras.ops.stack([
            keras.ops.reshape(batch_indices, [-1]),
            keras.ops.reshape(top_k_indices, [-1])
        ], axis=1)

        # Scatter top-k values into sparse tensor
        sparse_output = keras.ops.scatter_nd_update(
            sparse_output,
            indices,
            keras.ops.reshape(top_k_values, [-1])
        )

        return sparse_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape for this layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Output shape (batch_size, units).
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'sparsity': self.sparsity,
            'connection_sparsity': self.connection_sparsity,
            'activation': keras.activations.serialize(self.activation_fn),
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'trainable_projection': self.trainable_projection,
            'use_bias': self.use_bias,
        })
        return config


@keras.saving.register_keras_serializable(package="MothNet")
class HebbianReadoutLayer(keras.layers.Layer):
    """
    Hebbian readout layer implementing local correlation-based learning.

    This layer uses Hebbian plasticity ("neurons that fire together, wire together")
    rather than backpropagation for weight updates. Connections strengthen when
    pre-synaptic (MB) and post-synaptic (class label) neurons are simultaneously
    active, implementing a biologically plausible local learning rule.

    **Intent**: Provide an alternative to global error minimization (backpropagation)
    by using a simple, local correlation-based learning rule. This demonstrates that
    effective learning doesn't require error signals to propagate backward through
    the network.

    **Architecture**:
    ```
    MB Input (shape=[batch, mb_units])
           ↓
    Linear Transform: logits = W·x + b
           │ W updated via Hebbian rule, not backprop
           ↓
    Class Logits (shape=[batch, num_classes])

    Hebbian Update (during training):
    ΔW = α · (x^T · y) / batch_size

    Where:
    - x: MB activations (pre-synaptic)
    - y: Target labels one-hot (post-synaptic)
    - α: Learning rate
    ```

    **Mathematical Operations**:
    1. **Forward Pass**: logits = W·x + b (standard linear transformation)

    2. **Hebbian Update**:
       - Compute correlation: C = (1/N)·∑(x_i ⊗ y_i) over batch
       - Update weights: W_new = W_old + α·C
       - Where x_i ⊗ y_i is the outer product

    3. **Biological Interpretation**:
       - High MB activity + High target → Strengthen connection
       - High MB activity + Low target → No change (asymmetric rule)
       - Low MB activity → No change (requires pre-synaptic activity)

    Parameters
    ----------
    units : int
        Number of output units (number of classes).
    learning_rate : float, default=0.01
        Hebbian learning rate (α in ΔW = α·x^T·y).
        Typical range: [0.001, 0.1].
    kernel_initializer : str or keras.initializers.Initializer, default='glorot_uniform'
        Initializer for the readout weight matrix.
    kernel_regularizer : Optional[keras.regularizers.Regularizer], default=None
        Regularizer for the readout weights (applied during Hebbian updates).
    use_bias : bool, default=True
        Whether to include bias terms.
    **kwargs
        Additional keyword arguments for the base Layer class.

    Input shape
        (batch_size, input_dim): 2D tensor from Mushroom Body (sparse MB codes).

    Output shape
        (batch_size, units): Class logits before softmax.

    Attributes
    ----------
    readout_weights : keras.Variable
        Weight matrix for readout connections, shape (input_dim, units).
        Updated via Hebbian rule, not backpropagation.
    bias : keras.Variable, optional
        Bias vector, shape (units,), present if use_bias=True.
        Can be trained via backprop or manually updated.

    Example
    -------
    >>> # Create Hebbian readout for 10-class classification
    >>> readout = HebbianReadoutLayer(
    ...     units=10,
    ...     learning_rate=0.01
    ... )
    >>>
    >>> # Forward pass
    >>> mb_codes = keras.ops.ones((32, 4000))  # Batch of MB codes
    >>> logits = readout(mb_codes)
    >>> print(logits.shape)  # (32, 10)
    >>>
    >>> # Hebbian training (in training loop)
    >>> y_true = keras.ops.one_hot(labels, num_classes=10)
    >>> readout.hebbian_update(mb_codes, y_true)

    Notes
    -----
    **Advantages of Hebbian Learning**:

    1. **Local Computation**: Weight updates only require local information
       (pre- and post-synaptic activity), not global error signals.

    2. **Biological Plausibility**: This rule is observed in real neural systems
       and may explain aspects of biological learning.

    3. **Complementary to Backprop**: Hebbian learning excels at associative
       learning and discovers different patterns than gradient descent.

    **Why It Works with Sparse MB Codes**:

    - Sparsity is CRITICAL: With dense codes, Hebbian learning strengthens
      all connections indiscriminately, learning nothing discriminative.

    - With sparse codes: Only the specific handful of MB neurons active for
      each class strengthen their connections to that class's output.

    - This creates selective associations: MB pattern A → Class 1,
      MB pattern B → Class 2, with minimal interference.

    **Custom Training Required**:

    This layer cannot be trained with standard `model.fit()`. Use the
    `train_hebbian()` method of the MothNet model, or implement a custom
    training loop that calls `hebbian_update()`.
    """

    def __init__(
        self,
        units: int,
        learning_rate: float = 0.01,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.learning_rate = learning_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create readout weights.

        This is called automatically when the layer first processes input.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        self.readout_weights = self.add_weight(
            name='readout_weights',
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=False,  # Updated via Hebbian rule, not backprop
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True,  # Bias can be trained normally or via Hebbian updates
            )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute readout activations (forward pass).

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of shape (batch_size, input_dim) from Mushroom Body.
        training : bool, optional
            Whether the layer is in training mode (unused, for API compatibility).

        Returns
        -------
        keras.KerasTensor
            Class logits of shape (batch_size, units).
        """
        output = keras.ops.matmul(inputs, self.readout_weights)

        if self.use_bias:
            output = output + self.bias

        return output

    def hebbian_update(
        self,
        pre_synaptic: keras.KerasTensor,
        post_synaptic: keras.KerasTensor
    ) -> None:
        """
        Apply Hebbian weight update rule: ΔW = α · (x^T · y) / batch_size.

        This method should be called during training to update weights based on
        correlations between MB activations and target labels.

        Parameters
        ----------
        pre_synaptic : keras.KerasTensor
            Pre-synaptic activations (MB layer output) of shape (batch_size, input_dim).
            These are the sparse MB codes.
        post_synaptic : keras.KerasTensor
            Post-synaptic activations (target labels, one-hot encoded) of shape
            (batch_size, units). Represents desired output neuron activity.

        Notes
        -----
        The update is averaged over the batch to make learning rate independent
        of batch size. The outer product x^T·y computes correlation between
        each MB neuron and each class.
        """
        # Compute batch-averaged outer product: (1/N) · x^T · y
        batch_size = keras.ops.shape(pre_synaptic)[0]
        weight_update = keras.ops.matmul(
            keras.ops.transpose(pre_synaptic),
            post_synaptic
        ) / keras.ops.cast(batch_size, dtype=pre_synaptic.dtype)

        # Apply Hebbian update: W_new = W_old + α·ΔW
        new_weights = self.readout_weights + self.learning_rate * weight_update
        self.readout_weights.assign(new_weights)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape for this layer.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input tensor.

        Returns
        -------
        tuple
            Output shape (batch_size, units).
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'learning_rate': self.learning_rate,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'use_bias': self.use_bias,
        })
        return config

# ---------------------------------------------------------------------

