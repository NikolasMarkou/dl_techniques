"""
Self Logits Evolution Decoding (SLED) Module for Keras 3.

This module provides a backend-agnostic Keras 3 implementation of the SLED
algorithm, as described in the paper "SLED: Self Logits Evolution Decoding for
Improving Factuality in Large Language Models" (arXiv:2411.02433). This code
is fully portable and will run on any Keras backend (TensorFlow, PyTorch, JAX).

SLED is an inference-time decoding framework that enhances the factual accuracy
of Large Language Models (LLMs) without requiring fine-tuning or external
knowledge bases. It operates by leveraging the "latent knowledge" from the
model's earlier layers to correct and refine the output logits of the final layer.

Key Features:
- Implements the three-phase SLED process using only the Keras 3 API.
- Fully backend-agnostic (TensorFlow, PyTorch, JAX compatible).
- Optimized for performance by focusing computations on the top-k tokens.
- Follows a factory pattern (`sled_builder`) for easy instantiation from a
  configuration dictionary.
- Uses `keras.ops.take_along_axis` for efficient, backend-agnostic gathering.
- Replicates complex indexed scatter operations using a combination of
  `keras.ops.one_hot` and `keras.ops.where` for maximum portability.

Usage Example:
    >>> import keras
    >>> import numpy as np
    >>>
    >>> # Configuration for the SLED processor
    >>> config = {
    ...     "type": "sled_v1",
    ...     "config": {
    ...         "evolution_rate": 0.5,
    ...         "evolution_scale": 10,
    ...         "temperature": 1.0,
    ...         "use_tau_in_update": True,
    ...         "inactive_logit_value": -1e9
    ...     }
    ... }
    >>>
    >>> # Build the processor
    >>> sled_processor = sled_builder(config)
    >>>
    >>> # --- In a model's generate loop ---
    >>> # The input can be a NumPy array, tf.Tensor, torch.Tensor, etc.
    >>> B, V, N_LAYERS = 2, 50257, 32
    >>> all_logits_for_step = [
    ...     np.random.randn(B, V).astype("float32") for _ in range(N_LAYERS)
    ... ]
    >>>
    >>> # Apply SLED to get the corrected logits (returns a KerasTensor)
    >>> evolved_logits = sled_processor(all_logits_for_step)
    >>>
    >>> # The evolved_logits can now be used for token sampling or greedy search
    >>> next_token = keras.ops.argmax(evolved_logits, axis=-1)
    >>> print(evolved_logits.shape)
    (2, 50257)
"""

import keras
from enum import Enum
from keras import KerasTensor
from typing import Dict, List, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.constants import TYPE_STR, CONFIG_STR

# ---------------------------------------------------------------------

class SledEvolutionType(str, Enum):
    """Enumeration of available SLED algorithm versions."""
    SLED_V1 = "sled_v1"

# ---------------------------------------------------------------------

def _cosine_similarity(
    x1: KerasTensor, x2: KerasTensor, axis: int = -1, epsilon: float = 1e-8
) -> KerasTensor:
    """
    Computes cosine similarity between two tensors along a specified axis.
    This is a robust, backend-agnostic implementation using keras.ops.

    Args:
        x1: The first tensor.
        x2: The second tensor. Must be broadcastable to the shape of x1.
        axis: The axis along which to compute similarity.
        epsilon: A small value to avoid division by zero for zero-vectors.

    Returns:
        A tensor containing the cosine similarity.
    """
    dot_product = keras.ops.sum(x1 * x2, axis=axis)
    norm_x1 = keras.ops.sqrt(
        keras.ops.maximum(keras.ops.sum(keras.ops.square(x1), axis=axis), epsilon)
    )
    norm_x2 = keras.ops.sqrt(
        keras.ops.maximum(keras.ops.sum(keras.ops.square(x2), axis=axis), epsilon)
    )
    return dot_product / (norm_x1 * norm_x2)


class SledLogitsProcessor:
    """
    A callable class that applies the SLED algorithm to a set of logits using the Keras API.
    """
    def __init__(
        self,
        evolution_rate: float,
        evolution_scale: int,
        temperature: float,
        use_tau_in_update: bool,
        inactive_logit_value: float
    ):
        """
        Initializes the SLED Logits Processor with configured hyperparameters.

        Args:
            evolution_rate (float): The 'alpha' parameter. Controls the magnitude
                of the logit update.
            evolution_scale (int): The 'k' parameter. The number of top tokens
                to consider for the evolution process.
            temperature (float): The 'tau' parameter. Used for softening the
                softmax distributions. Must be > 0.
            use_tau_in_update (bool): If True, divides the logit update term by
                the temperature, following Eq. 5 in the paper.
            inactive_logit_value (float): The logit value to assign to tokens
                not in the top-k set. Should be a large negative number.
        """
        if not evolution_rate > 0:
            raise ValueError("evolution_rate (alpha) must be positive.")
        if not evolution_scale > 0:
            raise ValueError("evolution_scale (k) must be a positive integer.")
        if not temperature > 0:
            raise ValueError("temperature (tau) must be positive.")

        self.alpha = evolution_rate
        self.k = evolution_scale
        self.tau = temperature
        self.use_tau_in_update = use_tau_in_update
        self.eta = inactive_logit_value
        logger.info(f"SLED Processor (Keras) initialized with: alpha={self.alpha}, k={self.k}, tau={self.tau}")

    def __call__(self, all_logits: List[KerasTensor]) -> KerasTensor:
        """
        Applies the SLED algorithm to modify the final layer's logits.

        Args:
            all_logits (List[KerasTensor]): A list of logit tensors (e.g., NumPy array,
                tf.Tensor, torch.Tensor), one for each layer of the model. Tensors
                should correspond to the next token prediction logits, with shape
                [batch_size, vocab_size]. The list must be ordered from the
                earliest layer (index 0) to the final layer (index -1).

        Returns:
            KerasTensor: The modified ("self-evolved") logits from the final layer,
                with shape [batch_size, vocab_size].
        """
        if not isinstance(all_logits, list) or len(all_logits) < 2:
            raise ValueError("all_logits must be a list of at least two tensors.")

        logits_n_final = keras.ops.convert_to_tensor(all_logits[-1])
        early_logits_list = [keras.ops.convert_to_tensor(logits) for logits in all_logits[:-1]]
        dtype = logits_n_final.dtype

        # --- Setup and Pre-computation ---
        batch_size, vocab_size = keras.ops.shape(logits_n_final)
        self.k = min(self.k, vocab_size)

        p_n_final = keras.ops.softmax(logits_n_final / self.tau, axis=-1)
        early_p_list = [keras.ops.softmax(logits / self.tau, axis=-1) for logits in early_logits_list]
        _, top_k_indices = keras.ops.top_k(logits_n_final, self.k, sorted=False)

        # --- Phases 1 & 2: Estimate and Ensemble P_latent ---
        m_numerators = keras.ops.zeros((batch_size, self.k), dtype=dtype)
        m_denominator = keras.ops.zeros((), dtype=dtype)
        pe_k = keras.ops.one_hot(top_k_indices, vocab_size, dtype=dtype)

        for logits_n, p_n in zip(early_logits_list, early_p_list):
            diff_vec_expanded = keras.ops.expand_dims(logits_n - logits_n_final, axis=1)
            p_n_expanded = keras.ops.expand_dims(p_n, axis=1)
            grads_k = p_n_expanded - pe_k

            cossims_k = _cosine_similarity(diff_vec_expanded, grads_k, axis=-1)
            m_k_n = keras.ops.square(keras.ops.maximum(cossims_k, 0.0))

            m_numerators += m_k_n
            m_denominator += keras.ops.sum(m_k_n)

        if m_denominator == 0:
            logger.warning("SLED denominator is zero. All layer contrasts were misaligned. Returning original logits.")
            return logits_n_final

        m_k = m_numerators / m_denominator

        # --- Phase 3: Self-Evolution ---
        # Use keras.ops.take_along_axis for backend-agnostic indexed gathering.
        p_n_final_k = keras.ops.take_along_axis(p_n_final, top_k_indices, axis=1)

        update_term = self.alpha * (p_n_final_k - m_k)
        if self.use_tau_in_update:
            update_term /= self.tau

        original_top_k_logits = keras.ops.take_along_axis(logits_n_final, top_k_indices, axis=1)
        updated_top_k_logits = original_top_k_logits - update_term

        # Replicate scatter update using one-hot masking and keras.ops.where
        # This is a highly portable, backend-agnostic way to perform an indexed update.
        final_mask = keras.ops.sum(pe_k, axis=1)
        updates_expanded = keras.ops.expand_dims(updated_top_k_logits, axis=-1)
        scattered_updates = keras.ops.sum(pe_k * updates_expanded, axis=1)

        evolved_logits = keras.ops.where(
            keras.ops.cast(final_mask, "bool"),
            scattered_updates,
            self.eta
        )
        return evolved_logits

# ---------------------------------------------------------------------

def sled_builder(
    config: Dict[str, Union[str, Dict[str, Union[float, str, bool, int]]]]
) -> SledLogitsProcessor:
    """
    Builds a SledLogitsProcessor from a configuration dictionary.

    This factory function parses a configuration object to instantiate and
    return a callable SLED processor, ready for use in any Keras 3 environment.

    Args:
        config (Dict): A configuration dictionary. Expected format:
            {
                "type": "sled_v1",
                "config": {
                    "evolution_rate": float,
                    "evolution_scale": int,
                    "temperature": float,
                    "use_tau_in_update": bool (optional),
                    "inactive_logit_value": float (optional)
                }
            }

    Returns:
        SledLogitsProcessor: An initialized, callable SLED processor instance.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    sled_type = config.get(TYPE_STR)
    if sled_type is None:
        raise ValueError("SLED type cannot be None - must specify 'type' in config")
    if not isinstance(sled_type, str):
        raise TypeError("SLED type must be a string")

    sled_type = sled_type.strip().lower()

    sled_params = config.get(CONFIG_STR, {})
    if not isinstance(sled_params, dict):
        raise TypeError("'config' must be a dictionary containing SLED parameters")

    logger.info(f"Building SLED processor (Keras): type=[{sled_type}], params=[{sled_params}]")

    if sled_type == SledEvolutionType.SLED_V1:
        try:
            return SledLogitsProcessor(
                evolution_rate=float(sled_params.get("evolution_rate", 0.5)),
                evolution_scale=int(sled_params.get("evolution_scale", 10)),
                temperature=float(sled_params.get("temperature", 1.0)),
                use_tau_in_update=bool(sled_params.get("use_tau_in_update", True)),
                inactive_logit_value=float(sled_params.get("inactive_logit_value", -1e9))
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid SLED parameters in config: {e}") from e
    else:
        raise ValueError(
            f"Unknown SLED type: [{sled_type}]. "
            f"Supported types: {[t.value for t in SledEvolutionType]}"
        )

# ---------------------------------------------------------------------
