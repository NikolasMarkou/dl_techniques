"""
Neural Turing Machine (NTM) Model Wrapper.

This module provides a high-level `NTMModel` class that wraps the low-level
`NTMCell` into a standard Keras Model, complete with variant presets (tiny, base, large)
and optional output projection.

It serves as the primary entry point for instantiating trainable NTMs in
application code.

Classes:
    NTMModel: Configurable NTM model with presets.

Functions:
    create_ntm_variant: Factory function for easy instantiation.
"""

import keras
from keras import layers
import dataclasses
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.ntm import NTMCell, NTMConfig

# ---------------------------------------------------------------------
# NTM Model
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="DLTechniques")
class NTMModel(keras.Model):
    """
    Neural Turing Machine Model.

    A wrapper around the `NTMCell` that creates a fully unrolled Recurrent Neural Network.
    This model provides a sequence-to-sequence or sequence-to-vector interface
    compatible with standard Keras workflows.

    **Architecture**:
    ```
    Input(shape=[batch, seq_len, input_dim])
           ↓
    RNN(NTMCell) -> Unrolls over sequence
           ↓
    (Optional) Dense(output_dim)
           ↓
    Output
    ```

    **Presets**:
    - **tiny**: Small memory (32x16), simple controller, good for unit tests.
    - **base**: Standard NTM (128x32), LSTM controller, robust baseline.
    - **large**: Large memory (256x64), deep controller, for complex tasks.

    Args:
        input_shape: Tuple (seq_len, input_dim). seq_len can be None.
        output_dim: Dimension of the final output.
        config: NTMConfig object or dict defining NTM hyperparameters.
        return_sequences: Whether to return the full sequence or just the last output.
        use_projection: Whether to apply a dense projection to the NTM output.
        **kwargs: Additional arguments for Model base class.
    """

    NTM_VARIANTS = {
        'tiny': {
            'memory_size': 32,
            'memory_dim': 16,
            'controller_dim': 64,
            'num_read_heads': 1,
            'num_write_heads': 1,
            'shift_range': 3,
            'controller_type': 'lstm'
        },
        'base': {
            'memory_size': 128,
            'memory_dim': 32,
            'controller_dim': 256,
            'num_read_heads': 1,
            'num_write_heads': 1,
            'shift_range': 3,
            'controller_type': 'lstm'
        },
        'large': {
            'memory_size': 256,
            'memory_dim': 64,
            'controller_dim': 512,
            'num_read_heads': 2,
            'num_write_heads': 2,
            'shift_range': 3,
            'controller_type': 'lstm'
        }
    }

    def __init__(
            self,
            input_shape: Tuple[Optional[int], int],
            output_dim: int,
            config: Union[NTMConfig, Dict[str, Any]],
            return_sequences: bool = True,
            use_projection: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.input_shape_config = input_shape
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.use_projection = use_projection

        # Configuration handling
        if isinstance(config, dict):
            self.config_obj = NTMConfig.from_dict(config)
            self.config_dict = config
        else:
            self.config_obj = config
            self.config_dict = config.to_dict()

        # Create Layers (Golden Rule: Create all layers in __init__)

        # 1. NTM Cell
        self.cell = NTMCell(self.config_obj, name="ntm_cell")

        # 2. RNN Wrapper
        # We wrap the cell in an RNN layer to handle unrolling
        # return_state=False because we usually don't need raw NTM internal states
        # in the high-level model output
        self.rnn = layers.RNN(
            self.cell,
            return_sequences=return_sequences,
            return_state=False,
            name="ntm_rnn"
        )

        # 3. Output Projection
        if self.use_projection:
            self.projection = layers.Dense(
                output_dim,
                name="output_projection"
            )
        else:
            self.projection = None

        # Build the model if input shape is provided (optional but good for summary())
        # Note: Keras 3 models often defer build, but we can hint it here.
        # We generally avoid calling self.build() in __init__ to allow flexibility,
        # but we can set the input spec.

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model layers with explicit shapes.

        Args:
            input_shape: Shape of the input tensor (batch, seq_len, dim).
        """
        # 1. Build RNN
        self.rnn.build(input_shape)

        # 2. Build Projection
        # The cell output size is defined in NTMCell.output_size
        rnn_output_dim = self.cell.output_size

        if self.use_projection:
            # If return_sequences: (batch, seq_len, rnn_out)
            # Else: (batch, rnn_out)
            # Dense builds on the last dimension
            self.projection.build((None, rnn_output_dim))

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass.

        Args:
            inputs: Input tensor (batch, seq_len, input_dim).
            training: Whether to run in training mode (affects Dropout/RNN).

        Returns:
            Output tensor.
        """
        x = self.rnn(inputs, training=training)

        if self.use_projection:
            x = self.projection(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on configuration."""
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        last_dim = self.output_dim if self.use_projection else self.cell.output_size

        if self.return_sequences:
            return (batch_size, seq_len, last_dim)
        else:
            return (batch_size, last_dim)

    def get_config(self) -> Dict[str, Any]:
        """Serialize configuration."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape_config,
            'output_dim': self.output_dim,
            'config': self.config_dict,
            'return_sequences': self.return_sequences,
            'use_projection': self.use_projection,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NTMModel':
        """Deserialize configuration."""
        # NTMConfig reconstruction happens in __init__ via dict check
        # We copy to avoid modifying the original config dict
        config = config.copy()
        return cls(**config)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_shape: Tuple[Optional[int], int],
            output_dim: int,
            return_sequences: bool = True,
            **kwargs: Any
    ) -> 'NTMModel':
        """
        Create NTM model from a predefined variant.

        Args:
            variant: One of 'tiny', 'base', 'large'.
            input_shape: Input sequence shape (seq_len, dim).
            output_dim: Output dimension.
            return_sequences: Whether to output full sequence.
            **kwargs: Overrides for specific config parameters (e.g., controller_type).

        Returns:
            Configured NTMModel instance.
        """
        if variant not in cls.NTM_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(cls.NTM_VARIANTS.keys())}")

        variant_config = cls.NTM_VARIANTS[variant].copy()

        # Separate model arguments from NTMConfig arguments
        ntm_field_names = {f.name for f in dataclasses.fields(NTMConfig)}

        config_overrides = {k: v for k, v in kwargs.items() if k in ntm_field_names}
        model_kwargs = {k: v for k, v in kwargs.items() if k not in ntm_field_names}

        # Update variant defaults with overrides
        variant_config.update(config_overrides)

        ntm_config = NTMConfig(**variant_config)

        return cls(
            input_shape=input_shape,
            output_dim=output_dim,
            config=ntm_config,
            return_sequences=return_sequences,
            **model_kwargs
        )


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_ntm_variant(
        variant: str,
        input_shape: Tuple[Optional[int], int],
        output_dim: int,
        return_sequences: bool = True,
        **kwargs: Any
) -> NTMModel:
    """
    Factory function to create an NTM model variant.

    Args:
        variant: One of 'tiny', 'base', 'large'.
        input_shape: Tuple (seq_len, input_dim).
        output_dim: Size of output vector.
        return_sequences: If True, returns (batch, seq, out). Else (batch, out).
        **kwargs: Additional overrides for NTM configuration.

    Returns:
        An uncompiled NTMModel instance.
    """
    return NTMModel.from_variant(
        variant=variant,
        input_shape=input_shape,
        output_dim=output_dim,
        return_sequences=return_sequences,
        **kwargs
    )