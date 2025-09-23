"""
Self Logits Evolution Decoding (SLED) Module for Keras 3 / TensorFlow.

This module provides a Keras 3 implementation of the SLED algorithm, as
described in the paper "SLED: Self Logits Evolution Decoding for Improving
Factuality in Large Language Models" (arXiv:2411.02433). This version is
specifically designed for use with a TensorFlow backend (v2.18+).

SLED is an inference-time decoding framework that enhances the factual accuracy
of Large Language Models (LLMs) without requiring fine-tuning or external
knowledge bases. It operates by leveraging the "latent knowledge" from the
model's earlier layers to correct and refine the output logits of the final layer.

This implementation is structured as a general-purpose logits processor, making
it adaptable to any Keras-based model that can provide access to the logits
from multiple layers during generation.

Key Features:
- Implements the three-phase SLED process for Keras/TensorFlow.
- Optimized for performance by focusing computations on the top-k tokens.
- Fully configurable hyperparameters (evolution_rate, scale, temperature).
- Follows a factory pattern (`sled_builder`) for easy instantiation from a
  configuration dictionary.
- Prioritizes backend-agnostic `keras.ops` for compatibility, using `tf`
  operations where necessary for advanced indexing.

Mathematical Behavior:
The processor takes a list of logit tensors (tf.Tensor) from all model layers.
It computes a corrective update for the final layer's logits based on the
divergence between early-layer and final-layer logits, steering the output
distribution toward a more factually consistent result.

1.  **Phase 1 (Estimate):** For each early layer, a "latent" probability
    distribution is estimated by computing the cosine similarity between the
    logit evolution vector (early_logits - final_logits) and the ideal gradient
    for each potential token.
2.  **Phase 2 (Ensemble):** The latent distributions from all early layers are
    ensembled into a single `P_latent` distribution via a weighted average.
3.  **Phase 3 (Self-Evolution):** The final logits are updated by taking a
    gradient-like step that minimizes the KL divergence between the model's
    output distribution and the estimated `P_latent`.

Usage Example:
    >>> import tensorflow as tf
    >>> import keras
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
    >>> # Build the processor using the factory
    >>> sled_processor = sled_builder(config)
    >>>
    >>> # --- In a model's generate loop ---
    >>> # Assume we get logits from all layers for the last token
    >>> B, V, N_LAYERS = 2, 50257, 32  # Batch, Vocab, Num Layers
    >>> all_logits_for_step = [
    ...     tf.random.normal((B, V)) for _ in range(N_LAYERS)
    ... ]
    >>>
    >>> # Apply SLED to get the corrected logits
    >>> evolved_logits = sled_processor(all_logits_for_step)
    >>>
    >>> # The evolved_logits can now be used for token sampling or greedy search
    >>> next_token = tf.argmax(evolved_logits, axis=-1)
    >>> print(evolved_logits.shape)
    (2, 50257)
"""

import keras
from enum import Enum
import tensorflow as tf
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

def _cosine_similarity(x1: tf.Tensor, x2: tf.Tensor, axis: int = -1, epsilon: float = 1e-8) -> tf.Tensor:
    """
    Computes cosine similarity between two tensors along a specified axis.

    Args:
        x1: The first tensor.
        x2: The second tensor. Must be broadcastable to the shape of x1.
        axis: The axis along which to compute similarity.
        epsilon: A small value to avoid division by zero.

    Returns:
        A tensor containing the cosine similarity.
    """
    dot_product = keras.ops.sum(x1 * x2, axis=axis)
    norm_x1 = keras.ops.sqrt(keras.ops.maximum(keras.ops.sum(keras.ops.square(x1), axis=axis), epsilon))
    norm_x2 = keras.ops.sqrt(keras.ops.maximum(keras.ops.sum(keras.ops.square(x2), axis=axis), epsilon))
    return dot_product / (norm_x1 * norm_x2)


class SledLogitsProcessor:
    """
    A callable class that applies the SLED algorithm to a set of logits using Keras/TensorFlow.
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
        logger.info(f"SLED Processor (Keras/TF) initialized with: alpha={self.alpha}, k={self.k}, tau={self.tau}")

    def __call__(self, all_logits: List[tf.Tensor]) -> tf.Tensor:
        """
        Applies the SLED algorithm to modify the final layer's logits.

        Args:
            all_logits (List[tf.Tensor]): A list of logit tensors, one for each
                layer of the model. Tensors should correspond to the next token
                prediction logits, with shape [batch_size, vocab_size]. The list
                must be ordered from the earliest layer (index 0) to the final
                layer (index -1).

        Returns:
            tf.Tensor: The modified ("self-evolved") logits from the final
                layer, with shape [batch_size, vocab_size].
        """
        if not isinstance(all_logits, list) or len(all_logits) < 2:
            raise ValueError("all_logits must be a list of at least two tensors (one early layer, one final layer).")

        logits_n_final = all_logits[-1]
        early_logits_list = all_logits[:-1]
        dtype = logits_n_final.dtype

        # --- Setup and Pre-computation ---
        batch_size, vocab_size = keras.ops.shape(logits_n_final)
        self.k = min(self.k, vocab_size)

        p_n_final = keras.ops.softmax(logits_n_final / self.tau, axis=-1)
        early_p_list = [keras.ops.softmax(logits / self.tau, axis=-1) for logits in early_logits_list]
        _, top_k_indices = keras.ops.top_k(logits_n_final, self.k, sorted=False)

        # --- Phases 1 & 2: Estimate and Ensemble P_latent ---
        m_numerators = keras.ops.zeros((batch_size, self.k), dtype=dtype)
        m_denominator = tf.constant(0.0, dtype=dtype)
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
        # We need tf.gather_nd and tf.tensor_scatter_nd_update for indexed operations
        batch_indices = tf.range(batch_size, dtype=top_k_indices.dtype)
        batch_indices = keras.ops.expand_dims(batch_indices, axis=-1)
        # Create indices of shape [batch_size, k, 2] for gather_nd/scatter_nd
        gather_indices = keras.ops.concatenate([
            keras.ops.tile(batch_indices, [1, self.k]),
            keras.ops.expand_dims(top_k_indices, axis=-1)
        ], axis=-1)

        p_n_final_k = tf.gather_nd(p_n_final, gather_indices)

        update_term = self.alpha * (p_n_final_k - m_k)
        if self.use_tau_in_update:
            update_term /= self.tau

        original_top_k_logits = tf.gather_nd(logits_n_final, gather_indices)
        updated_top_k_logits = original_top_k_logits - update_term

        # Create base tensor and scatter the updated values
        base_logits = keras.ops.full(keras.ops.shape(logits_n_final), self.eta, dtype=dtype)
        evolved_logits = tf.tensor_scatter_nd_update(base_logits, gather_indices, updated_top_k_logits)

        return evolved_logits


def sled_builder(
    config: Dict[str, Union[str, Dict[str, Union[float, str, bool, int]]]]
) -> SledLogitsProcessor:
    """
    Builds a SledLogitsProcessor from a configuration dictionary for Keras/TF.

    This factory function parses a configuration object to instantiate and
    return a callable SLED processor, ready for use in an LLM's decoding loop.

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

    Raises:
        ValueError: If the config is invalid or the type is not supported.
        TypeError: If config or its components have incorrect types.
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

    logger.info(f"Building SLED processor (Keras/TF): type=[{sled_type}], params=[{sled_params}]")

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