import keras
from keras import ops
from typing import Any, Dict, List, Tuple, Union

from dl_techniques.layers.ntm.ntm_interface import NTMConfig
from dl_techniques.layers.ntm.baseline_ntm import NeuralTuringMachine


@keras.saving.register_keras_serializable()
class NTMMultiTask(keras.Model):
    """
    A Neural Turing Machine wrapper for Multi-Task Learning.

    This model wraps a standard NTM and handles the fusion of sequence data
    with task-specific conditioning vectors. It broadcasts the one-hot task ID
    across the temporal dimension and concatenates it with the input sequence,
    allowing the NTM controller to condition its operations on the specific task context.

    **Architecture**:
    ```
    Inputs: [Sequence(batch, seq, dim), TaskID(batch, tasks)]
             |
             v
    Broadcast TaskID -> (batch, seq, tasks)
             |
             v
    Concatenate [Sequence, TaskID] -> (batch, seq, dim + tasks)
             |
             v
    NeuralTuringMachine
             |
             v
    Output (batch, seq, output_dim)
    ```

    Args:
        ntm_config: Configuration object for the internal NTM.
        output_dim: Dimensionality of the output.
        num_tasks: Number of distinct tasks (size of one-hot vector).
        **kwargs: Additional arguments for the Model base class.
    """

    def __init__(
            self,
            ntm_config: Union[NTMConfig, Dict[str, Any]],
            output_dim: int,
            num_tasks: int,
            **kwargs: Any
    ):
        super().__init__(**kwargs)

        # Handle configuration serialization/deserialization
        if isinstance(ntm_config, dict):
            self.ntm_config = NTMConfig.from_dict(ntm_config)
        else:
            self.ntm_config = ntm_config

        self.output_dim = output_dim
        self.num_tasks = num_tasks

        # Create sub-layers in __init__ (Golden Rule)
        self.ntm_layer = NeuralTuringMachine(
            config=self.ntm_config,
            output_dim=output_dim,
            return_sequences=True,
            return_state=False,
            name="core_ntm"
        )

    def build(self, input_shape: Union[List[Tuple], Tuple]) -> None:
        """
        Build the model and its sub-layers.

        Calculates the effective input shape for the internal NTM (feature dim + task dim)
        and explicitly builds it.

        Args:
            input_shape: List of shapes [(batch, seq_len, feat_dim), (batch, num_tasks)]
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(f"Expected input_shape to be a list of length 2, got {input_shape}")

        sequence_shape, task_shape = input_shape

        # Calculate combined dimension: feature_dim + num_tasks
        # sequence_shape is (batch, seq_len, feature_dim)
        if sequence_shape[-1] is None:
            raise ValueError("Last dimension of input sequence must be defined.")

        combined_feature_dim = sequence_shape[-1] + self.num_tasks

        # The NTM layer expects (batch, seq_len, combined_dim)
        ntm_input_shape = (sequence_shape[0], sequence_shape[1], combined_feature_dim)

        self.ntm_layer.build(ntm_input_shape)

        super().build(input_shape)

    def call(self, inputs: List[keras.KerasTensor]) -> keras.KerasTensor:
        """
        Forward pass.

        Args:
            inputs: List containing:
                - sequence_input: Tensor of shape (Batch, Seq_Len, Dim)
                - task_id_input: Tensor of shape (Batch, Num_Tasks)

        Returns:
            Output tensor of shape (Batch, Seq_Len, Output_Dim)
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("NTMMultiTask expects inputs=[sequence, task_id]")

        x, task_one_hot = inputs

        # Get dynamic dimensions using ops for graph safety
        input_shape = ops.shape(x)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        # 1. Expand task_one_hot to (Batch, 1, Num_Tasks)
        task_expanded = ops.expand_dims(task_one_hot, axis=1)

        # 2. Broadcast across sequence length: (Batch, Seq_Len, Num_Tasks)
        # We explicitly cast shapes to ensure compatibility with ops.broadcast_to
        target_shape = (batch_size, seq_len, self.num_tasks)
        task_broadcasted = ops.broadcast_to(task_expanded, target_shape)

        # 3. Concatenate: (Batch, Seq_Len, Dim + Num_Tasks)
        ntm_input = ops.concatenate([x, task_broadcasted], axis=-1)

        # 4. Pass to NTM
        return self.ntm_layer(ntm_input)

    def compute_output_shape(self, input_shape: List[Tuple]) -> Tuple[int, int, int]:
        """
        Compute output shape based on input shapes.

        Args:
            input_shape: List of [(batch, seq, dim), (batch, tasks)]

        Returns:
            Output shape tuple (batch, seq, output_dim)
        """
        sequence_shape = input_shape[0]
        # Return (batch, seq_len, output_dim)
        return (sequence_shape[0], sequence_shape[1], self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.
        """
        config = super().get_config()
        config.update({
            "ntm_config": self.ntm_config.to_dict(),
            "output_dim": self.output_dim,
            "num_tasks": self.num_tasks,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NTMMultiTask":
        """
        Create model from configuration dictionary.
        """
        # Ensure ntm_config is reconstructed properly
        if "ntm_config" in config and isinstance(config["ntm_config"], dict):
            config["ntm_config"] = NTMConfig.from_dict(config["ntm_config"])
        return cls(**config)