"""
Neural Turing Machine conditioned on a task identity for multi-task sequence
learning.

The NTM's algorithmic tasks — copy, repeat-copy, associative recall, priority sort
— share an input format but demand different memory disciplines, and a single NTM
trained on their union has no way to know which discipline the current sequence
calls for. Task conditioning supplies that missing bit. A one-hot task vector is
broadcast across the temporal axis and concatenated onto every timestep's features,
so the controller sees the task identity at the same moment it sees the first
symbol, and the head parameters it emits — key, sharpness, interpolation gate,
shift, erase and add vectors — can be functions of the task from step one.

Broadcasting rather than prepending a task token is the deliberate choice here. A
prepended token would make the task identity a memory the controller has to hold
across the whole sequence, competing for exactly the recurrent capacity that
external memory exists to relieve; carrying it on every timestep instead makes it a
free-standing input the controller can consult without remembering. The cost is
`num_tasks` extra input features at every step, which is negligible against the
controller width.

The wrapper is otherwise thin: it computes the fused feature dimension
(`feature_dim + num_tasks`) in `build` and hands it to the inner
`NeuralTuringMachine`, which is the fully unrolled sequence-level NTM rather than
the RNN cell. Output is always a sequence — these tasks are supervised at every
timestep, not at the end — so `return_sequences=True` is fixed rather than exposed,
and `return_state=False` keeps the memory matrix out of the model's outputs. The
broadcast target shape is built from `ops.shape(x)` rather than static dimensions
so a variable sequence length survives graph tracing.

References:
    - Graves et al., 2014. Neural Turing Machines.
      (https://arxiv.org/abs/1410.5401)
    - Caruana, 1997. Multitask Learning. Machine Learning 28, 41-75.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.memory.ntm_interface import NTMConfig
from dl_techniques.layers.memory.baseline_ntm import NeuralTuringMachine

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NTMMultiTask(keras.Model):
    """
    A Neural Turing Machine wrapped in a task-conditioning fusion step.

    Wraps a sequence-level ``NeuralTuringMachine`` and handles the fusion of the
    input sequence with a one-hot task vector: the task id is broadcast across
    the temporal axis and concatenated onto every timestep's features, so the
    controller can condition its head parameters on the task identity from the
    first symbol rather than having to remember a prepended token. Everything
    else is delegated to the inner NTM.

    The inner NTM is fixed at ``return_sequences=True`` and
    ``return_state=False``: these tasks are supervised at every timestep, and the
    memory matrix is internal state rather than a model output, so neither is
    exposed as a constructor knob.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐   ┌──────────────────────────┐
        │ sequence [B, T, feature_dim] │   │ task_id [B, num_tasks]   │
        └──────────────┬───────────────┘   └────────────┬─────────────┘
                       │                                ▼
                       │                  ┌──────────────────────────┐
                       │                  │ expand_dims axis=1       │
                       │                  │   → [B, 1, num_tasks]    │
                       │                  └────────────┬─────────────┘
                       │                                ▼
                       │                  ┌──────────────────────────┐
                       │                  │ broadcast_to             │
                       │                  │   → [B, T, num_tasks]    │
                       │                  │ (target built from       │
                       │                  │  ops.shape, so a dynamic │
                       │                  │  T survives tracing)     │
                       │                  └────────────┬─────────────┘
                       └───────────────►(concat, −1)◄──┘
                                             │
                                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  core_ntm: NeuralTuringMachine                               │
        │    input width = feature_dim + num_tasks                     │
        │    controller → head params (key, sharpness, gate, shift,    │
        │                 erase, add) → addressing → memory read/write │
        │    return_sequences=True, return_state=False                 │
        └──────────────────────────────┬───────────────────────────────┘
                                       ▼
        ┌──────────────────────────────────────┐
        │  Output [B, T, output_dim]           │
        └──────────────────────────────────────┘

    :param ntm_config: Configuration for the internal NTM — an :class:`NTMConfig`,
        or the dict it serializes to (the ``from_config`` path).
    :type ntm_config: Union[NTMConfig, Dict[str, Any]]
    :param output_dim: Dimensionality of the per-timestep output.
    :type output_dim: int
    :param num_tasks: Number of distinct tasks, i.e. the width of the one-hot
        task vector. This is exactly how many features the fusion step adds to
        every timestep.
    :type num_tasks: int
    :param kwargs: Additional keyword arguments for the ``keras.Model`` base class.

    :raises ValueError: From :meth:`build` if ``input_shape`` is not a pair, or if
        the sequence's feature dimension is undefined; from :meth:`call` if
        ``inputs`` is not a ``[sequence, task_id]`` pair.

    Input shape:
        A list of two tensors:

        - sequence: ``(batch_size, seq_len, feature_dim)``.
        - task id: ``(batch_size, num_tasks)``, one-hot.

    Output shape:
        3D tensor ``(batch_size, seq_len, output_dim)``. There is one output mode
        only — the model always returns a sequence and never the memory state.

    Example:
        >>> config = NTMConfig(memory_size=128, memory_dim=20, controller_units=100)
        >>> model = NTMMultiTask(ntm_config=config, output_dim=8, num_tasks=4)
        >>> model.build([(None, 20, 8), (None, 4)])
        >>>
        >>> sequence = keras.random.normal((2, 20, 8))
        >>> task_id = keras.ops.one_hot(keras.ops.array([0, 2]), 4)
        >>> outputs = model([sequence, task_id], training=False)   # (2, 20, 8)

    Note:
        The task vector is BROADCAST, not prepended. A prepended task token would
        force the controller to carry the task identity in recurrent state across
        the whole sequence — competing for the capacity that external memory
        exists to relieve — whereas broadcasting makes it a free-standing input at
        every step, at a cost of ``num_tasks`` extra features per timestep.

    Attributes:
        ntm_config: The resolved :class:`NTMConfig`.
        ntm_layer: The inner ``NeuralTuringMachine``, named ``core_ntm``.
    """

    def __init__(
            self,
            ntm_config: Union[NTMConfig, Dict[str, Any]],
            output_dim: int,
            num_tasks: int,
            **kwargs: Any
    ):
        """Resolve the NTM configuration and create the inner NTM.

        The config is accepted either live or as its serialized dict, so
        ``from_config`` and direct construction take the same path. See the class
        docstring for the parameter reference.
        """
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
        """Build the inner NTM at the FUSED input width.

        The NTM never sees the raw sequence: its input width is
        ``feature_dim + num_tasks``, computed here from the sequence shape, which
        is why the feature dimension must be statically known even though the
        sequence length need not be.

        :param input_shape: A pair of shapes,
            ``[(batch, seq_len, feature_dim), (batch, num_tasks)]``.
        :type input_shape: Union[List[Tuple], Tuple]
        :raises ValueError: If ``input_shape`` is not a length-2 sequence, or if
            the sequence's last dimension is ``None``.
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

    def call(
            self,
            inputs: List[keras.KerasTensor],
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Broadcast the task vector across time, fuse it onto the sequence, run the NTM.

        The broadcast target is built from ``keras.ops.shape(x)`` rather than the
        static shape, so a variable sequence length survives graph tracing.

        :param inputs: A ``[sequence, task_id]`` pair — a
            ``(batch, seq_len, feature_dim)`` tensor and a
            ``(batch, num_tasks)`` one-hot tensor.
        :type inputs: List[keras.KerasTensor]
        :param training: Training-mode flag, forwarded explicitly to the NTM layer
            rather than left to Keras' ambient ``CallContext``. Nothing in the NTM
            stack is training-sensitive today, so this currently changes no
            output; see the ``D-059`` anchor above.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch, seq_len, output_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``inputs`` is not a ``[sequence, task_id]`` pair.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("NTMMultiTask expects inputs=[sequence, task_id]")

        x, task_one_hot = inputs

        # Get dynamic dimensions using ops for graph safety
        input_shape = keras.ops.shape(x)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        # 1. Expand task_one_hot to (Batch, 1, Num_Tasks)
        task_expanded = keras.ops.expand_dims(task_one_hot, axis=1)

        # 2. Broadcast across sequence length: (Batch, Seq_Len, Num_Tasks)
        # We explicitly cast shapes to ensure compatibility with ops.broadcast_to
        target_shape = (batch_size, seq_len, self.num_tasks)
        task_broadcasted = keras.ops.broadcast_to(task_expanded, target_shape)

        # 3. Concatenate: (Batch, Seq_Len, Dim + Num_Tasks)
        ntm_input = keras.ops.concatenate([x, task_broadcasted], axis=-1)

        # 4. Pass to NTM
        return self.ntm_layer(ntm_input, training=training)

    def compute_output_shape(self, input_shape: List[Tuple]) -> Tuple[int, int, int]:
        """Return the output shape, taking batch and length from the sequence input.

        The task shape contributes nothing: fusion widens the features, and the
        NTM's ``output_dim`` replaces that width entirely.

        :param input_shape: A pair of shapes,
            ``[(batch, seq_len, feature_dim), (batch, num_tasks)]``.
        :type input_shape: List[Tuple]
        :return: ``(batch, seq_len, output_dim)``.
        :rtype: Tuple[int, int, int]
        """
        sequence_shape = input_shape[0]
        # Return (batch, seq_len, output_dim)
        return (sequence_shape[0], sequence_shape[1], self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        The NTM configuration is stored as its dict form, so the inner NTM is
        reconstructed from config rather than serialized as a sub-layer.

        :return: The configuration dictionary.
        :rtype: Dict[str, Any]
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
        """Create a model from configuration, rebuilding the :class:`NTMConfig`.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: An ``NTMMultiTask`` instance.
        :rtype: NTMMultiTask
        """
        # Ensure ntm_config is reconstructed properly
        if "ntm_config" in config and isinstance(config["ntm_config"], dict):
            config["ntm_config"] = NTMConfig.from_dict(config["ntm_config"])
        return cls(**config)

# ---------------------------------------------------------------------
