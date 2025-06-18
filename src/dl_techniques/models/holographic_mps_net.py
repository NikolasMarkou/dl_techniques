"""
Holographic Encoder-Decoder with Entropy-Guided Architecture

This file implements an experimental encoder-decoder model. Its architecture is
inspired by analogies from quantum information theory and the holographic principle.
The primary goal is to structure the flow of information through the network in a
deliberate way, rather than relying solely on standard layers and gradient descent.

This is achieved by using custom `MPSLayer` layers and a unique `EntropyRegularizer`
that guides how different parts of the network learn.

Key Concepts & Mechanisms Explained
===================================

This model does NOT simulate quantum physics. It uses concepts from that field as
inspiration for its architecture. Here is how those concepts are translated into code:

1. Singular Value Entropy & Information Flow
------------------------------------------
Instead of the abstract "Entanglement Entropy," this model operationalizes the idea
using the Shannon entropy of a layer's singular values.

- A layer's weight matrix can be decomposed (via SVD) into singular values, which
  describe the "strength" of its primary transformation axes.
- **Low Entropy (Low-Rank):** If a few singular values are large and the rest are near zero,
  the layer performs a simple, low-rank transformation. It can only capture dominant,
  global, or coarse-grained features.
- **High Entropy (High-Rank):** If many singular values are of similar magnitude, the
  layer performs a complex, high-rank transformation, allowing it to capture fine-grained,
  local details.
- The `EntropyRegularizer` adds a penalty to the loss function, pushing the singular
  value entropy of a layer's weight matrix towards a specified target.

2. Multi-Scale Holographic Decoding
----------------------------------
This is the model's implementation of the "holographic principle." The decoder is
composed of multiple, parallel branches. The single latent vector is fed into all of them.

- Each branch is assigned a different `target_entropy` by its regularizer, ranging
  from low to high.
- **Low-Entropy Branches:** Forced to learn global, coarse features.
- **High-Entropy Branches:** Forced to learn local, detailed features.
- The final output is created by combining the representations from all these "scales."

This creates a "holographic" representation where the complete information is distributed
across specialized feature sets.

3. MPS-Based Layers (`MPSLayer`)
--------------------------------
The encoder and decoder branches use `MPSLayer`, inspired by Matrix Product States from
physics. In a machine learning context, an MPS layer is:
- A parameter-efficient way to represent a large linear transformation.
- Well-suited for 1D sequence data, as it is designed to capture local and
  long-range correlations along a chain.
- Its complexity is controlled by the `bond_dim` hyperparameter.


Architectural Summary & Data Flow
=================================
1.  **Input Flattening:** The input (e.g., an image) is flattened into a 1D vector.
    (Note: This discards crucial spatial information).
2.  **Encoding:** The `MPSLayer` encoder compresses this vector into a latent representation.
3.  **Parallel Decoding:** The latent vector is processed by multiple decoder branches
    simultaneously, each regularized to a different entropy target.
4.  **Combination:** The outputs of all branches are concatenated.
5.  **Projection:** A final dense layer maps the combined features to the desired output shape.


Potential Strengths & Weaknesses
================================

Strengths:
----------
- **Structured Inductive Bias:** Forces the model to learn a separation of features
  (global vs. local), which could improve generalization.
- **Potential for Interpretability:** Analyzing the output of individual branches could
  reveal what kind of features the model is learning at different scales.
- **Alternative to Attention:** The `MPSLayer` provides a computationally different
  (O(N)) approach for modeling long-range dependencies in sequences compared to
  attention (O(N^2)).

Weaknesses:
-----------
- **Loss of Spatial Structure:** Flattening the input is a major drawback for data like
  images, where spatial proximity is key. This model is likely to underperform
  CNNs on such tasks.
- **Computational Cost:** The `EntropyRegularizer` requires computing an SVD at each
  training step, which can significantly slow down training.
- **Experimental Nature:** As a non-standard architecture, it lacks the extensive
  community support, pre-trained models, and best practices of models like Transformers or ResNets.
"""
import keras
import numpy as np
from keras import ops
from typing import Tuple, Optional, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


from dl_techniques.utils.logger import logger
from dl_techniques.layers.mps_layer import MPSLayer
from dl_techniques.regularizers.entropy_regularizer import EntropyRegularizer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HolographicEncoderDecoder(keras.Model):
    """
    Holographic Encoder-Decoder inspired by quantum information theory.

    This model attempts to capture the idea of holographic encoding where information
    about the whole system is encoded in a way that respects entropy scaling principles.

    Parameters
    ----------
    input_shape : Tuple[int, ...]
        Shape of the input data.
    latent_dim : int, default=64
        Dimension of the latent space.
    output_shape : Optional[Tuple[int, ...]], default=None
        Shape of the output data. If None, uses input_shape.
    bond_dim : int, default=16
        Bond dimension for MPS layers.
    num_branches : int, default=3
        Number of decoder branches.
    regularization_strength : float, default=0.01
        Strength of the entropy regularization.
    use_bias : bool, default=True
        Whether to use bias in layers.
    kernel_initializer : Union[str, keras.initializers.Initializer], default="he_normal"
        Initializer for kernels.
    kernel_regularizer : Optional[keras.regularizers.Regularizer], default=None
        Additional regularizer for kernels.

    Raises
    ------
    ValueError
        If input dimensions are invalid or configuration parameters are out of range.

    Examples
    --------
    >>> model = HolographicEncoderDecoder(
    ...     input_shape=(28, 28, 1),
    ...     latent_dim=64,
    ...     bond_dim=16,
    ...     num_branches=3
    ... )
    >>> x = keras.random.normal((32, 28, 28, 1))
    >>> y = model(x)
    >>> print(y.shape)
    (32, 28, 28, 1)
    """

    def __init__(
            self,
            input_shape: Tuple[int, ...],
            latent_dim: int = 64,
            output_shape: Optional[Tuple[int, ...]] = None,
            bond_dim: int = 16,
            num_branches: int = 3,
            regularization_strength: float = 0.01,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ) -> None:
        """
        Initialize the HolographicEncoderDecoder model.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of the input data.
        latent_dim : int, default=64
            Dimension of the latent space.
        output_shape : Optional[Tuple[int, ...]], default=None
            Shape of the output data. If None, uses input_shape.
        bond_dim : int, default=16
            Bond dimension for MPS layers.
        num_branches : int, default=3
            Number of decoder branches.
        regularization_strength : float, default=0.01
            Strength of the entropy regularization.
        use_bias : bool, default=True
            Whether to use bias in layers.
        kernel_initializer : Union[str, keras.initializers.Initializer], default="he_normal"
            Initializer for kernels.
        kernel_regularizer : Optional[keras.regularizers.Regularizer], default=None
            Additional regularizer for kernels.
        **kwargs
            Additional keyword arguments for the Model base class.

        Raises
        ------
        ValueError
            If input dimensions are invalid or configuration parameters are out of range.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not input_shape or any(dim <= 0 for dim in input_shape):
            raise ValueError("input_shape must contain positive integers")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if bond_dim <= 0:
            raise ValueError("bond_dim must be positive")
        if num_branches <= 0:
            raise ValueError("num_branches must be positive")
        if not 0.0 <= regularization_strength <= 1.0:
            raise ValueError("regularization_strength must be between 0 and 1")

        # Store configuration
        self._input_shape = input_shape
        self.input_dim = int(np.prod(input_shape))
        self.latent_dim = latent_dim
        self._output_shape = output_shape if output_shape is not None else input_shape
        self.output_dim = int(np.prod(self._output_shape))
        self.bond_dim = bond_dim
        self.num_branches = num_branches
        self.regularization_strength = regularization_strength
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Initialize sublayer attributes - will be created in build()
        self.encoder_mps = None
        self.decoder_branches = []
        self.output_projection = None
        self._build_input_shape = None

        logger.info(
            f"Initialized HolographicEncoderDecoder with input_shape={input_shape}, "
            f"latent_dim={latent_dim}, num_branches={num_branches}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model components.

        This method creates all the sublayers of the model including:
        - Encoder MPS layer for latent encoding
        - Multiple decoder branches with different entropy targets
        - Output projection layer

        Parameters
        ----------
        input_shape : Tuple[Optional[int], ...]
            Shape of the input tensor including batch dimension.
        """
        # Store for serialization
        self._build_input_shape = input_shape

        logger.info(f"Building HolographicEncoderDecoder with input_shape={input_shape}")

        # Create entropy regularizer
        entropy_reg = EntropyRegularizer(strength=self.regularization_strength)

        # Combine regularizers if provided
        if self.kernel_regularizer is not None:
            # Create a simple combination approach
            combined_regularizer = self.kernel_regularizer
        else:
            combined_regularizer = keras.regularizers.L2(1e-4)

        # Encoder layer
        self.encoder_mps = MPSLayer(
            output_dim=self.latent_dim,
            bond_dim=self.bond_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=combined_regularizer,
            name="encoder_mps"
        )

        # Create multiple decoder branches with different entropy profiles
        # This creates a holographic-like property where information is distributed
        # across different entropy scales
        self.decoder_branches = []
        for i in range(self.num_branches):
            # Each branch has a different entropy target
            # Linear interpolation from low to high entropy
            target_entropy = 0.3 + (0.6 * i / max(1, self.num_branches - 1))

            branch_reg = EntropyRegularizer(
                strength=self.regularization_strength,
                target_entropy=target_entropy
            )

            # Calculate branch output dimension
            branch_output_dim = max(1, self.output_dim // self.num_branches)

            branch = MPSLayer(
                output_dim=branch_output_dim,
                bond_dim=self.bond_dim,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=branch_reg,
                name=f"decoder_branch_{i}"
            )
            self.decoder_branches.append(branch)

        # Final output projection
        self.output_projection = keras.layers.Dense(
            self.output_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=combined_regularizer,
            name="output_projection"
        )

        super().build(input_shape)
        logger.info("HolographicEncoderDecoder build completed")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the holographic encoder-decoder model.

        This method implements the holographic encoding-decoding process:

        1. Flatten input data to 2D tensor (batch_size, input_dim)
        2. Encode data through MPS layer to latent representation
        3. Decode through multiple branches with different entropy targets
        4. Combine multi-scale features and project to output space

        The multi-branch decoding is key to the holographic principle implementation:

        - Each branch has a different entropy target (from low to high)
        - Low-entropy branches capture global, coarse-grained features
        - High-entropy branches encode local, fine-grained features
        - Together they form a holographic representation where information is
          distributed across different "scales" or "layers"

        This multi-scale approach resembles holographic principles where information
        about a system is encoded at different "scales" or "regions" of the representation,
        with the complete information recoverable from the entire set.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor of original shape.
        training : Optional[bool], default=None
            Whether the model is in training mode.

        Returns
        -------
        keras.KerasTensor
            Output tensor of shape [batch_size, *output_shape].
        """
        # Flatten input to 2D tensor using keras.ops
        batch_size = ops.shape(inputs)[0]
        x = ops.reshape(inputs, [batch_size, -1])  # Shape: [batch_size, input_dim]

        # Encode using MPS layer to latent representation
        latent = self.encoder_mps(x, training=training)  # Shape: [batch_size, latent_dim]

        # Decode through multiple branches with different entropy targets
        # Each branch has a different entropy regularization target
        # This creates a holographic-like encoding across different "scales"
        branch_outputs = []
        for branch in self.decoder_branches:
            branch_output = branch(latent, training=training)
            branch_outputs.append(branch_output)

        # Concatenate branch outputs to combine information from different scales
        multi_scale_features = ops.concatenate(branch_outputs, axis=-1)

        # Final projection to output dimension
        output = self.output_projection(multi_scale_features, training=training)

        # Reshape to original output shape using keras.ops
        output_shape_list = [batch_size] + list(self._output_shape)
        output = ops.reshape(output, output_shape_list)

        return output

    def get_config(self) -> Dict[str, Any]:
        """
        Get the model configuration for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            'input_shape': self._input_shape,
            'latent_dim': self.latent_dim,
            'output_shape': self._output_shape,
            'bond_dim': self.bond_dim,
            'num_branches': self.num_branches,
            'regularization_strength': self.regularization_strength,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get the build configuration for proper serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the model from a configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @property
    def input_shape_property(self) -> Tuple[int, ...]:
        """
        Get the input shape.

        Returns
        -------
        Tuple[int, ...]
            Input shape tuple.
        """
        return self._input_shape

    @property
    def output_shape_property(self) -> Tuple[int, ...]:
        """
        Get the output shape.

        Returns
        -------
        Tuple[int, ...]
            Output shape tuple.
        """
        return self._output_shape

# ---------------------------------------------------------------------
