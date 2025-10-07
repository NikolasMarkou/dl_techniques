"""
This module provides a comprehensive suite of Keras-native utilities designed to
facilitate the development and training of deep learning models on the
Abelian Reasoning Corpus (ARC) dataset.

The ARC dataset poses a unique challenge, representing abstract reasoning tasks as
transformations of 2D colored grids. Working with this data requires specialized
pre-processing, data loading, and model components. This module encapsulates these
requirements into a cohesive set of reusable tools that integrate seamlessly with the
Keras API.

The module is organized into several key components:

1.  **`ARCSequence` (Data Generator):**
    -   A highly efficient data loader built on `keras.utils.Sequence`. It handles
        the loading of ARC's `.npy` data files, provides on-the-fly batching, and
        supports shuffling between epochs, which is essential for robust training.
    -   It also includes preprocessing steps, such as normalizing input values to a
        `[-1, 1]` range, to prepare the data for neural network consumption.

2.  **`ARCGridEncoder` and `ARCGridDecoder` (Data Formatting Layers):**
    -   ARC tasks are defined on 2D grids, but Transformer models require 1D sequences.
        These layers handle the crucial conversion between these two formats.
    -   `ARCGridEncoder`: Takes a 2D grid and flattens it into a 1D sequence, adding
        the necessary padding and token offsets required by the ARC data format.
    -   `ARCGridDecoder`: Performs the reverse operation, taking a model's 1D output
        sequence and reshaping it back into a 2D grid for visualization or evaluation.

3.  **`ARCPuzzleEmbedding` (Task Conditioning Layer):**
    -   A specialized embedding layer that creates a unique, learnable vector for each
        puzzle ID in the ARC dataset.
    -   This allows a model to be "conditioned" on the specific puzzle it is solving,
        enabling it to learn puzzle-specific strategies or patterns. It is a simple
        but powerful way to provide task-specific context.

4.  **`ARCAccuracyMetric` (Custom Metric):**
    -   A custom `keras.metrics.Metric` designed specifically for evaluating ARC task
        performance.
    -   It correctly handles the special padding (`PAD`) and end-of-sequence (`EOS`)
        tokens, ensuring they are ignored during accuracy calculation.
    -   It can compute both token-level accuracy (how many individual grid cells are
        correct) and the more stringent sequence-level accuracy (is the entire output
        grid perfect?).

Together, these utilities provide the essential building blocks for creating, training,
and evaluating Keras models on the ARC dataset, abstracting away much of the
domain-specific data handling complexity.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import keras
from keras import ops
from keras import layers
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

class ARCSequence(keras.utils.Sequence):
    """
    Keras Sequence for loading ARC data efficiently.

    This class provides batched access to ARC datasets with proper
    shuffling and preprocessing for training.
    """

    def __init__(self,
                 dataset_path: str,
                 split: str = "train",
                 subset: str = "all",
                 batch_size: int = 32,
                 shuffle: bool = True,
                 max_grid_size: int = 30,
                 normalize_inputs: bool = True):
        """
        Initialize the ARC sequence.

        Args:
            dataset_path: Path to the ARC dataset directory
            split: Dataset split ('train' or 'test')
            subset: Dataset subset (default: 'all')
            batch_size: Batch size for training
            shuffle: Whether to shuffle data between epochs
            max_grid_size: Maximum grid size (should match dataset)
            normalize_inputs: Whether to normalize input values to [-1, 1]
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_grid_size = max_grid_size
        self.normalize_inputs = normalize_inputs

        # Load data
        self._load_data()

        # Initialize indices
        self.indices = np.arange(len(self.inputs))
        self.on_epoch_end()

    def _load_data(self) -> None:
        """Load the dataset from disk."""
        split_dir = self.dataset_path / self.split

        # Load arrays
        self.inputs = np.load(split_dir / f"{self.subset}__inputs.npy")
        self.labels = np.load(split_dir / f"{self.subset}__labels.npy")
        self.puzzle_identifiers = np.load(split_dir / f"{self.subset}__puzzle_identifiers.npy")

        # Load metadata
        with open(split_dir / "dataset.json", 'r') as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded ARC {self.split} dataset: {len(self.inputs)} examples")

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return (len(self.inputs) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batch of data.

        Args:
            index: Batch index

        Returns:
            Tuple of (inputs, labels) for the batch
        """
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.inputs))
        batch_indices = self.indices[start_idx:end_idx]

        # Get batch data
        batch_inputs = self.inputs[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Preprocess
        batch_inputs = self._preprocess_inputs(batch_inputs)
        batch_labels = self._preprocess_labels(batch_labels)

        return batch_inputs, batch_labels

    def _preprocess_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Preprocess input data.

        Args:
            inputs: Raw input data

        Returns:
            Preprocessed inputs
        """
        # Convert to float32
        inputs = inputs.astype(np.float32)

        # Normalize to [-1, 1] range if requested
        if self.normalize_inputs:
            # ARC tokens: PAD=0, EOS=1, colors=2-11
            # Map to [-1, 1] range
            vocab_size = self.metadata.get("vocab_size", 12)
            inputs = (inputs / (vocab_size - 1)) * 2.0 - 1.0

        return inputs

    def _preprocess_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Preprocess label data.

        Args:
            labels: Raw label data

        Returns:
            Preprocessed labels
        """
        # Keep labels as integers for classification
        return labels.astype(np.int32)

    def on_epoch_end(self) -> None:
        """Shuffle indices at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class ARCGridDecoder(layers.Layer):
    """
    Custom Keras layer for decoding ARC grid sequences.

    This layer converts flattened sequences back to 2D grids by removing
    padding and EOS tokens, then reshaping to the original grid format.
    """

    def __init__(self,
                 max_grid_size: int = 30,
                 pad_token: int = 0,
                 eos_token: int = 1,
                 color_offset: int = 2,
                 **kwargs):
        """
        Initialize the grid decoder layer.

        Args:
            max_grid_size: Maximum grid size for reshaping
            pad_token: Padding token value
            eos_token: End-of-sequence token value
            color_offset: Offset added to original colors
            **kwargs: Additional keyword arguments for Layer
        """
        super().__init__(**kwargs)
        self.max_grid_size = max_grid_size
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.color_offset = color_offset

        # Support configs for serialization
        self.supports_masking = False

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Args:
            input_shape: Input shape tuple
        """
        super().build(input_shape)

        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected 2D input, got {len(input_shape)}D")

        expected_seq_len = self.max_grid_size * self.max_grid_size
        if input_shape[1] != expected_seq_len:
            raise ValueError(f"Expected sequence length {expected_seq_len}, got {input_shape[1]}")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len)
            training: Whether in training mode

        Returns:
            Decoded grid tensor of shape (batch_size, max_grid_size, max_grid_size)
        """
        # Reshape to 2D grids
        batch_size = ops.shape(inputs)[0]
        grids = ops.reshape(inputs, (batch_size, self.max_grid_size, self.max_grid_size))

        # Remove color offset (clamp to avoid negative values)
        grids = ops.maximum(grids - self.color_offset, 0)

        return grids

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int, int]:
        """
        Compute output shape.

        Args:
            input_shape: Input shape tuple

        Returns:
            Output shape tuple
        """
        return (input_shape[0], self.max_grid_size, self.max_grid_size)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "max_grid_size": self.max_grid_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "color_offset": self.color_offset
        })
        return config


class ARCGridEncoder(layers.Layer):
    """
    Custom Keras layer for encoding 2D grids to sequences.

    This layer converts 2D grids to flattened sequences with proper
    padding and EOS token placement for ARC dataset format.
    """

    def __init__(self,
                 max_grid_size: int = 30,
                 pad_token: int = 0,
                 eos_token: int = 1,
                 color_offset: int = 2,
                 **kwargs):
        """
        Initialize the grid encoder layer.

        Args:
            max_grid_size: Maximum grid size
            pad_token: Padding token value
            eos_token: End-of-sequence token value
            color_offset: Offset to add to original colors
            **kwargs: Additional keyword arguments for Layer
        """
        super().__init__(**kwargs)
        self.max_grid_size = max_grid_size
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.color_offset = color_offset

        self.supports_masking = False

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Args:
            input_shape: Input shape tuple
        """
        super().build(input_shape)

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, height, width), got {len(input_shape)}D")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        Args:
            inputs: Input tensor of shape (batch_size, height, width)
            training: Whether in training mode

        Returns:
            Encoded sequence tensor of shape (batch_size, max_grid_size^2)
        """
        batch_size = ops.shape(inputs)[0]

        # Add color offset
        encoded_grids = inputs + self.color_offset

        # Pad to max grid size if necessary
        current_height = ops.shape(inputs)[1]
        current_width = ops.shape(inputs)[2]

        # Calculate padding needed
        pad_height = self.max_grid_size - current_height
        pad_width = self.max_grid_size - current_width

        # Apply padding if needed
        if pad_height > 0 or pad_width > 0:
            paddings = [[0, 0], [0, pad_height], [0, pad_width]]
            encoded_grids = ops.pad(encoded_grids, paddings, constant_values=self.pad_token)

        # Flatten to sequences
        sequences = ops.reshape(encoded_grids, (batch_size, -1))

        return sequences

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int]:
        """
        Compute output shape.

        Args:
            input_shape: Input shape tuple

        Returns:
            Output shape tuple
        """
        return (input_shape[0], self.max_grid_size * self.max_grid_size)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "max_grid_size": self.max_grid_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "color_offset": self.color_offset
        })
        return config


class ARCPuzzleEmbedding(layers.Layer):
    """
    Custom embedding layer for ARC puzzle identifiers.

    This layer creates learnable embeddings for different puzzles,
    which can be used to condition the model on puzzle-specific information.
    """

    def __init__(self,
                 num_puzzles: int,
                 embedding_dim: int,
                 mask_zero: bool = True,
                 embeddings_initializer: str = "uniform",
                 embeddings_regularizer: Optional[keras.regularizers.Regularizer] = None,
                 **kwargs):
        """
        Initialize the puzzle embedding layer.

        Args:
            num_puzzles: Number of unique puzzles
            embedding_dim: Dimension of embedding vectors
            mask_zero: Whether to mask zero values
            embeddings_initializer: Initializer for embeddings
            embeddings_regularizer: Regularizer for embeddings
            **kwargs: Additional keyword arguments for Layer
        """
        super().__init__(**kwargs)
        self.num_puzzles = num_puzzles
        self.embedding_dim = embedding_dim
        self.mask_zero = mask_zero
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)

        self.supports_masking = mask_zero

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        Args:
            input_shape: Input shape tuple
        """
        super().build(input_shape)

        # Create embedding weights
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_puzzles, self.embedding_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            trainable=True
        )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        Args:
            inputs: Input tensor with puzzle IDs
            training: Whether in training mode

        Returns:
            Embedded puzzle representations
        """
        # Cast inputs to int32
        inputs = ops.cast(inputs, "int32")

        # Gather embeddings
        embeddings = ops.take(self.embeddings, inputs, axis=0)

        return embeddings

    def compute_mask(self, inputs: keras.KerasTensor,
                     mask: Optional[keras.KerasTensor] = None) -> Optional[keras.KerasTensor]:
        """
        Compute mask for the layer.

        Args:
            inputs: Input tensor
            mask: Input mask

        Returns:
            Output mask
        """
        if not self.mask_zero:
            return None

        return ops.not_equal(inputs, 0)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Input shape tuple

        Returns:
            Output shape tuple
        """
        return input_shape + (self.embedding_dim,)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "num_puzzles": self.num_puzzles,
            "embedding_dim": self.embedding_dim,
            "mask_zero": self.mask_zero,
            "embeddings_initializer": keras.initializers.serialize(self.embeddings_initializer),
            "embeddings_regularizer": keras.regularizers.serialize(self.embeddings_regularizer)
        })
        return config


class ARCAccuracyMetric(keras.metrics.Metric):
    """
    Custom accuracy metric for ARC datasets.

    This metric computes accuracy while properly handling padding tokens
    and providing both token-level and sequence-level accuracy.
    """

    def __init__(self,
                 ignore_tokens: List[int] = [0, 1],  # PAD and EOS tokens
                 sequence_level: bool = False,
                 name: str = "arc_accuracy",
                 **kwargs):
        """
        Initialize the accuracy metric.

        Args:
            ignore_tokens: List of token IDs to ignore in accuracy computation
            sequence_level: Whether to compute sequence-level accuracy
            name: Name of the metric
            **kwargs: Additional keyword arguments for Metric
        """
        super().__init__(name=name, **kwargs)
        self.ignore_tokens = ignore_tokens
        self.sequence_level = sequence_level

        # State variables
        self.total_correct = self.add_weight(name="total_correct", initializer="zeros")
        self.total_count = self.add_weight(name="total_count", initializer="zeros")

    def update_state(self,
                     y_true: keras.KerasTensor,
                     y_pred: keras.KerasTensor,
                     sample_weight: Optional[keras.KerasTensor] = None) -> None:
        """
        Update metric state.

        Args:
            y_true: True labels
            y_pred: Predicted labels (logits)
            sample_weight: Optional sample weights
        """
        # Get predictions
        predictions = ops.argmax(y_pred, axis=-1)

        # Create mask for valid tokens
        mask = ops.ones_like(y_true, dtype="bool")
        for ignore_token in self.ignore_tokens:
            mask = mask & ops.not_equal(y_true, ignore_token)

        # Compute accuracy
        matches = ops.equal(y_true, predictions)
        valid_matches = matches & mask

        if self.sequence_level:
            # Sequence-level accuracy: all tokens in sequence must be correct
            valid_counts = ops.sum(ops.cast(mask, "float32"), axis=-1)
            correct_counts = ops.sum(ops.cast(valid_matches, "float32"), axis=-1)
            sequence_correct = ops.equal(valid_counts, correct_counts)

            batch_correct = ops.sum(ops.cast(sequence_correct, "float32"))
            batch_total = ops.cast(ops.shape(y_true)[0], "float32")
        else:
            # Token-level accuracy
            batch_correct = ops.sum(ops.cast(valid_matches, "float32"))
            batch_total = ops.sum(ops.cast(mask, "float32"))

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, "float32")
            batch_correct = batch_correct * sample_weight
            batch_total = batch_total * sample_weight

        # Update state
        self.total_correct.assign_add(batch_correct)
        self.total_count.assign_add(batch_total)

    def result(self) -> keras.KerasTensor:
        """
        Compute the final metric result.

        Returns:
            Accuracy value
        """
        return ops.divide_no_nan(self.total_correct, self.total_count)

    def reset_state(self) -> None:
        """Reset metric state."""
        self.total_correct.assign(0.)
        self.total_count.assign(0.)

    def get_config(self) -> Dict[str, Any]:
        """
        Get metric configuration.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "ignore_tokens": self.ignore_tokens,
            "sequence_level": self.sequence_level
        })
        return config

# ---------------------------------------------------------------------

def create_arc_data_generator(dataset_path: str,
                              split: str = "train",
                              batch_size: int = 32,
                              shuffle: bool = True,
                              normalize_inputs: bool = True) -> ARCSequence:
    """
    Create an ARC data generator for Keras training.

    Args:
        dataset_path: Path to the ARC dataset
        split: Dataset split ('train' or 'test')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        normalize_inputs: Whether to normalize inputs

    Returns:
        ARCSequence data generator
    """
    return ARCSequence(
        dataset_path=dataset_path,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        normalize_inputs=normalize_inputs
    )

# ---------------------------------------------------------------------

def create_simple_arc_model(vocab_size: int = 12,
                            seq_len: int = 900,
                            embed_dim: int = 256,
                            num_layers: int = 4,
                            num_heads: int = 8) -> keras.Model:
    """
    Create a simple transformer model for ARC tasks.

    Args:
        vocab_size: Size of vocabulary
        seq_len: Sequence length
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads

    Returns:
        Keras Model instance
    """
    # Input layers
    inputs = layers.Input(shape=(seq_len,), name="inputs")
    puzzle_ids = layers.Input(shape=(), name="puzzle_ids")

    # Token embeddings
    token_embeddings = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)

    # Positional embeddings
    pos_embeddings = layers.Embedding(seq_len, embed_dim)(
        ops.arange(seq_len)[None, :]
    )

    # Puzzle embeddings
    puzzle_embeddings = ARCPuzzleEmbedding(1000, embed_dim)(puzzle_ids)
    puzzle_embeddings = layers.RepeatVector(seq_len)(puzzle_embeddings)

    # Combine embeddings
    x = token_embeddings + pos_embeddings + puzzle_embeddings

    # Transformer layers
    for i in range(num_layers):
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            name=f"attention_{i}"
        )(x, x)

        # Add & norm
        x = layers.LayerNormalization(name=f"norm1_{i}")(x + attention_output)

        # Feed forward
        ff_output = layers.Dense(embed_dim * 4, activation="relu", name=f"ff1_{i}")(x)
        ff_output = layers.Dense(embed_dim, name=f"ff2_{i}")(ff_output)

        # Add & norm
        x = layers.LayerNormalization(name=f"norm2_{i}")(x + ff_output)

    # Output head
    outputs = layers.Dense(vocab_size, name="output_logits")(x)

    # Create model
    model = keras.Model(
        inputs=[inputs, puzzle_ids],
        outputs=outputs,
        name="simple_arc_model"
    )

    return model

# ---------------------------------------------------------------------

