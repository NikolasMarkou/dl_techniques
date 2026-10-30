"""
Transition layers of the Neural HMM: the initial-state distribution ``log pi`` and the
transition matrices ``log A``, on the uniform batched shapes the inference module reads.

Both layers return ``(log_pi, log_A)`` with ``log_pi`` of shape ``(B, K)`` and ``log_A`` of
shape ``(B, T, K, K)``. ``log_A[:, t, i, j]`` is the log-probability of moving from state
``i`` at step ``t - 1`` INTO state ``j`` at step ``t``: rows are the source, columns the
destination, every row of ``exp(log_A)`` sums to 1, and the ``t = 0`` slot is never read.

- :class:`StaticTransition` is a plain HMM: one learned matrix, broadcast over the batch and
  time axes of whatever ``(B, T, ...)`` tensor it is called on.
- :class:`InputDependentTransition` adds a per-step offset projected from a ``(B, T, H)``
  context to the static logits: Tran et al.'s context transitions when the context
  summarizes the observations, an input-output HMM (Bengio and Frasconi, 1995) when it
  summarizes an exogenous input. The projection is zero-initialized, so training starts at
  the static solution.

The topology says which transitions exist at all, independently of the layer type:

- ``ergodic``: every transition;
- ``left_to_right``: only to the same or a later state, and the chain starts in state 0;
- ``banded``: like ``left_to_right``, skipping at most ``band_width`` states forward.

A forbidden transition (or start state) has probability exactly 0. It is masked by
selection (``keras.ops.where`` with :func:`dl_techniques.utils.dtype_policy.mask_sentinel`),
never by adding a large negative number, which overflows float16.

References:
    - Rabiner, 1989. A Tutorial on Hidden Markov Models and Selected Applications in
      Speech Recognition. Proceedings of the IEEE 77(2), pp. 257-286.
    - Bengio and Frasconi, 1995. An Input Output HMM Architecture. Advances in Neural
      Information Processing Systems 7, pp. 427-434.
"""

from typing import Any, Optional, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.dtype_policy import accumulation_dtype, mask_sentinel
from dl_techniques.utils.keras_registration import register_dl_technique

from .inference import LogSpaceOutputSpec

# ---------------------------------------------------------------------

#: Topologies the transition layers accept.
TOPOLOGIES: Tuple[str, ...] = ("ergodic", "left_to_right", "banded")

# ---------------------------------------------------------------------


def topology_masks(n_states: int, topology: str, band_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return which start states and which transitions a topology allows.

    :param n_states: Number of hidden states ``K``.
    :param topology: One of :data:`TOPOLOGIES`.
    :param band_width: Largest forward skip allowed by ``banded``; ignored otherwise.
    :return: ``(start_keep, transition_keep)``: boolean numpy arrays of shape ``(K,)``
        and ``(K, K)``, True where allowed. Rows of ``transition_keep`` are the source.
    :raises ValueError: If ``topology`` is unknown.
    """
    ones = np.ones((n_states, n_states), dtype=bool)
    if topology == "ergodic":
        return np.ones(n_states, dtype=bool), ones
    if topology == "left_to_right":
        transition_keep = np.triu(ones)
    elif topology == "banded":
        transition_keep = np.triu(ones) & ~np.triu(ones, k=band_width + 1)
    else:
        raise ValueError(f"Unknown topology '{topology}'. Available: {list(TOPOLOGIES)}.")
    start_keep = np.zeros(n_states, dtype=bool)
    start_keep[0] = True
    return start_keep, transition_keep


def _batch_time_zeros(inputs: keras.KerasTensor, dtype: str) -> keras.KerasTensor:
    """Return zeros of shape ``(B, T)`` for a ``(B, T, ...)`` tensor, read from its shape only.

    :param inputs: Any tensor of rank 2 or more.
    :param dtype: Dtype of the zeros.
    :return: ``(B, T)`` zeros; works with unknown ``B`` and ``T``.
    """
    leading = inputs[(slice(None), slice(None)) + (0,) * (len(inputs.shape) - 2)]
    return keras.ops.zeros_like(leading, dtype=dtype)


@register_dl_technique("dl_techniques.models.neural_hmm.transitions")
class StaticTransition(LogSpaceOutputSpec, keras.layers.Layer):
    """A time-invariant transition matrix and initial distribution, with a topology.

    ``log_pi = log_softmax(where(start_keep, pi_logits, sentinel))`` and
    ``log_A = log_softmax(where(transition_keep, transition_logits, sentinel), axis=-1)``,
    broadcast to ``(B, K)`` and ``(B, T, K, K)``. Both outputs are in the accumulation dtype
    (float32 under a float16 policy).

    :param n_states: Number of hidden states ``K``; positive.
    :param topology: One of ``ergodic``, ``left_to_right``, ``banded``.
    :param band_width: Largest forward skip under ``banded``; non-negative.
    :param self_loop_init: Initial diagonal of the transition logits. A positive value
        makes states sticky at the start of training, which keeps early decoding from
        flickering between states.
    :param kwargs: Base ``keras.layers.Layer`` arguments.
    :raises ValueError: If an argument is out of range or ``topology`` is unknown.
    """

    def __init__(
        self,
        n_states: int,
        topology: str = "ergodic",
        band_width: int = 1,
        self_loop_init: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if n_states <= 0:
            raise ValueError(f"n_states must be positive, got {n_states}.")
        if band_width < 0:
            raise ValueError(f"band_width must be non-negative, got {band_width}.")
        self.n_states = n_states
        self.topology = topology
        self.band_width = band_width
        self.self_loop_init = self_loop_init
        # Config-derived constants: held as numpy, converted in call(), never weights.
        self._start_keep, self._transition_keep = topology_masks(n_states, topology, band_width)
        self.pi_logits = None
        self.transition_logits = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the initial-state and transition logits.

        :param input_shape: Shape of the tensor the layer is called on, ``(B, T, ...)``.
        :raises ValueError: If the input has rank below 2.
        """
        if len(input_shape) < 2:
            raise ValueError(f"Expected an input of shape (batch, time, ...), got {tuple(input_shape)}.")
        self.pi_logits = self.add_weight(name="pi_logits", shape=(self.n_states,), initializer="zeros")
        self.transition_logits = self.add_weight(
            name="transition_logits",
            shape=(self.n_states, self.n_states),
            initializer=keras.initializers.Identity(gain=self.self_loop_init),
        )
        super().build(input_shape)

    def _normalize(self, pi_logits: keras.KerasTensor, transition_logits: keras.KerasTensor) -> Tuple:
        """Mask forbidden entries, then log-normalize in the accumulation dtype.

        :param pi_logits: ``(K,)`` logits in the compute dtype.
        :param transition_logits: ``(..., K, K)`` logits in the compute dtype.
        :return: ``(log_pi (K,), log_A (..., K, K))`` in the accumulation dtype.
        """
        sentinel = mask_sentinel(self.compute_dtype)
        start_keep = keras.ops.convert_to_tensor(self._start_keep)
        transition_keep = keras.ops.convert_to_tensor(self._transition_keep)
        pi_logits = keras.ops.where(start_keep, pi_logits, keras.ops.full_like(pi_logits, sentinel))
        transition_logits = keras.ops.where(
            transition_keep, transition_logits, keras.ops.full_like(transition_logits, sentinel)
        )
        # DECISION plan-2026-10-13T192257-1bb3bf1f/D-010: mask in the compute dtype with
        # mask_sentinel (finite there), THEN cast to the accumulation dtype and normalize.
        # Do NOT write ``logits + (1 - keep) * -1e9``: under mixed_float16 the -1e9 is -inf
        # and 0 * -inf turns every allowed entry into NaN. Do NOT log_softmax in float16:
        # the log-probabilities feed a log-space recursion that accumulates.
        accum = accumulation_dtype(self.compute_dtype)
        log_pi = keras.ops.log_softmax(keras.ops.cast(pi_logits, accum), axis=-1)
        log_A = keras.ops.log_softmax(keras.ops.cast(transition_logits, accum), axis=-1)
        return log_pi, log_A

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> Tuple:
        """Return the initial and transition log-probabilities, batched.

        :param inputs: Any ``(B, T, ...)`` tensor; only its batch and time axes are read.
        :param training: Unused; accepted for a uniform call signature.
        :return: ``(log_pi (B, K), log_A (B, T, K, K))`` in the accumulation dtype.
        """
        log_pi, log_A = self._normalize(self.pi_logits, self.transition_logits)
        zeros = _batch_time_zeros(inputs, log_A.dtype)
        return zeros[:, 0, None] + log_pi, zeros[:, :, None, None] + log_A

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple:
        """Return ``((B, K), (B, T, K, K))`` from the stored config.

        :param input_shape: ``(B, T, ...)``.
        :return: The two output shapes.
        """
        batch, steps = input_shape[0], input_shape[1]
        return (batch, self.n_states), (batch, steps, self.n_states, self.n_states)

    def get_config(self) -> dict:
        """Return every constructor argument.

        :return: The layer config.
        """
        config = super().get_config()
        config.update(
            {
                "n_states": self.n_states,
                "topology": self.topology,
                "band_width": self.band_width,
                "self_loop_init": self.self_loop_init,
            }
        )
        return config


@register_dl_technique("dl_techniques.models.neural_hmm.transitions")
class InputDependentTransition(StaticTransition):
    """Per-step transition matrices driven by a context.

    ``log_A[:, t] = log_softmax(where(keep, transition_logits + W h_t, sentinel))``, where
    ``W`` is a zero-initialized, bias-free projection of the ``(B, T, H)`` context to
    ``K * K`` offsets (the static logits already carry the constant term). At
    initialization every step equals the static matrix; the context only matters once the
    gradient moves ``W``. ``log_pi`` does not depend on the context.

    :param n_states: Number of hidden states ``K``; positive.
    :param topology: One of ``ergodic``, ``left_to_right``, ``banded``.
    :param band_width: Largest forward skip under ``banded``; non-negative.
    :param self_loop_init: Initial diagonal of the static transition logits.
    :param kwargs: Base ``keras.layers.Layer`` arguments.
    :raises ValueError: If an argument is out of range or ``topology`` is unknown.
    """

    def __init__(
        self,
        n_states: int,
        topology: str = "ergodic",
        band_width: int = 1,
        self_loop_init: float = 2.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_states, topology, band_width, self_loop_init, **kwargs)
        self.context_projection = keras.layers.Dense(
            n_states * n_states, use_bias=False, kernel_initializer="zeros", name="context_projection"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the static logits and build the context projection.

        :param input_shape: Context shape ``(B, T, H)``.
        :raises ValueError: If the context is not rank 3.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Expected a context of shape (batch, time, features), got {tuple(input_shape)}.")
        self.context_projection.build(input_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> Tuple:
        """Return the initial and the per-step transition log-probabilities.

        :param inputs: Context ``(B, T, H)``.
        :param training: Forwarded to the context projection.
        :return: ``(log_pi (B, K), log_A (B, T, K, K))`` in the accumulation dtype.
        """
        offset = self.context_projection(inputs, training=training)
        shape = keras.ops.shape(offset)
        offset = keras.ops.reshape(offset, (shape[0], shape[1], self.n_states, self.n_states))
        log_pi, log_A = self._normalize(self.pi_logits, self.transition_logits + offset)
        zeros = _batch_time_zeros(inputs, log_pi.dtype)
        return zeros[:, 0, None] + log_pi, log_A
