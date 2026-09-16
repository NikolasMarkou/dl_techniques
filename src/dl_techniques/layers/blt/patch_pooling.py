"""PatchPooling: pool byte hidden states within each patch into one patch
vector, for the Byte Latent Transformer (BLT) local encoder.

Supports three pooling strategies (``'max'``, ``'mean'``, ``'attention'``),
each producing a fixed ``max_patches`` number of output slots regardless of
how many patches the input actually contains, and each ending with a Dense
projection to ``output_dim``.
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.blt.patch_pooling")
class PatchPooling(keras.layers.Layer):
    """Pool byte hidden states within each patch into one patch vector.

    The output always has ``max_patches`` slots, built by looping over patch
    indices in Python, so the layer emits that many sub-graphs whatever the
    sequence contains. Every method ends with a Dense projection to
    ``output_dim``.

    Architecture (attention pooling):

    .. code-block:: text

        byte hiddens [B, S, H]        patch_ids [B, S]
              │                             │
              ▼                             ▼
        ┌──────────────────────────────────────┐
        │ for p in range(max_patches):         │
        │   keep bytes with patch_ids == p     │
        │   zero the rest                      │
        │   learnable queries cross-attend     │
        │   mean over the queries              │
        └──────────────────────────────────────┘
              │ stack over p, [B, P, H]
              ▼
        ┌──────────────────────────────────────┐
        │ dense to output_dim                  │
        └──────────────────────────────────────┘
              │
              ▼
        patch representations [B, P, output_dim]

    Attention pooling passes no attention mask, so the zeroed out-of-patch
    positions still take part as keys and values.

    Methods and empty patches:

    .. code-block:: text

        method     non-empty patch          empty patch
        max        per-patch maximum        zero vector
        mean       per-patch mean           zero vector
        attention  queries attend, then     attends over an
                   mean over queries        all-zero sequence

    The ``max`` path masks with an internal ``-1e9`` sentinel and rescues empty
    patches to zero rather than leaving the sentinel in place.

    :param pooling_method: One of ``'max'``, ``'mean'``, ``'attention'``. An
        unknown value raises ``ValueError`` from ``call``, not from the
        constructor.
    :type pooling_method: str
    :param output_dim: Output dimension of the patch representations.
    :type output_dim: int
    :param num_queries: Number of query vectors for attention pooling. Unused
        by the other two methods.
    :type num_queries: int
    :param max_patches: Number of patch slots emitted, used as the static
        patch count in ``call``.
    :type max_patches: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
            self,
            pooling_method: str = 'attention',
            output_dim: int = 768,
            num_queries: int = 4,
            max_patches: int = 64,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.pooling_method = pooling_method
        self.output_dim = output_dim
        self.num_queries = num_queries
        self.max_patches = max_patches

        # Both depend on the input width, so build() creates them.
        self.attention_layer = None
        self.output_projection = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the pooling and projection layers.

        The attention head count and key width are derived from the input
        width, not from ``output_dim``.

        :param input_shape: Byte hidden shape ``(batch, seq_len, hidden_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        input_dim = input_shape[-1]

        if self.pooling_method == 'attention':
            num_heads = min(8, input_dim)
            key_dim = max(input_dim // num_heads, 1)
            self.attention_layer = keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                name='patch_attention'
            )

            self.query_embeddings = self.add_weight(
                shape=(self.num_queries, input_dim),
                initializer='glorot_uniform',
                trainable=True,
                name='query_embeddings'
            )

            # Explicit build, because a lazy first-call build leaves the
            # attention weights unloadable on a .keras reload.
            query_shape = (input_shape[0], self.num_queries, input_dim)
            kv_shape = (input_shape[0], None, input_dim)
            self.attention_layer.build(query_shape, kv_shape, kv_shape)

        self.output_projection = keras.layers.Dense(
            self.output_dim,
            name='output_projection'
        )
        projection_input_shape = (input_shape[0], None, input_dim)
        self.output_projection.build(projection_input_shape)

        super().build(input_shape)

    def call(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Pool byte hidden states into patch representations.

        :param byte_hiddens: Byte hidden states, shape ``(batch_size,
            seq_len, hidden_dim)``.
        :type byte_hiddens: keras.KerasTensor
        :param patch_ids: Patch ids, shape ``(batch_size, seq_len)``.
        :type patch_ids: keras.KerasTensor
        :param training: Whether in training mode. Reaches the attention
            sub-layer only.
        :type training: Optional[bool]
        :return: Patch representations, shape ``(batch_size, max_patches,
            output_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``pooling_method`` is not one of the three names.
        """
        batch_size = ops.shape(byte_hiddens)[0]
        seq_len = ops.shape(byte_hiddens)[1]
        hidden_dim = ops.shape(byte_hiddens)[2]

        # A static count keeps the loop and the output shape graph-safe.
        num_patches = self.max_patches

        if self.pooling_method == 'max':
            return self._max_pooling(byte_hiddens, patch_ids, num_patches)
        elif self.pooling_method == 'mean':
            return self._mean_pooling(byte_hiddens, patch_ids, num_patches)
        elif self.pooling_method == 'attention':
            return self._attention_pooling(byte_hiddens, patch_ids, num_patches, training)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def _max_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int
    ) -> keras.KerasTensor:
        """Take the per-channel maximum over each patch's bytes."""
        batch_size = ops.shape(byte_hiddens)[0]
        hidden_dim = ops.shape(byte_hiddens)[2]

        patch_reps = []

        for p in range(num_patches):
            mask = ops.equal(patch_ids, p)
            mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)

            masked_hiddens = ops.where(mask_expanded, byte_hiddens, -1e9)
            patch_max = ops.max(masked_hiddens, axis=1)

            # DECISION plan-2026-08-18T140459-7991552f/D-039: rescue empty patches to
            # zero; the -1e9 sentinel would dominate the downstream LayerNorm.
            # See decisions.md.
            has_any = ops.any(mask, axis=1, keepdims=True)
            patch_max = ops.where(has_any, patch_max, ops.zeros_like(patch_max))

            patch_reps.append(patch_max)

        result = ops.stack(patch_reps, axis=1)

        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def _mean_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int
    ) -> keras.KerasTensor:
        """Take the mean over each patch's bytes, empty patches giving zero."""
        batch_size = ops.shape(byte_hiddens)[0]

        patch_reps = []

        for p in range(num_patches):
            mask = ops.equal(patch_ids, p)
            mask_expanded = ops.expand_dims(ops.cast(mask, byte_hiddens.dtype), axis=-1)

            masked_hiddens = byte_hiddens * mask_expanded
            patch_sum = ops.sum(masked_hiddens, axis=1)
            # The floor of 1 turns an empty patch into a zero vector.
            patch_count = ops.sum(ops.cast(mask, byte_hiddens.dtype), axis=1, keepdims=True)
            patch_mean = patch_sum / ops.maximum(patch_count, 1.0)

            patch_reps.append(patch_mean)

        result = ops.stack(patch_reps, axis=1)

        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def _attention_pooling(
            self,
            byte_hiddens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            num_patches: int,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Let learnable queries cross-attend to each patch's zeroed sequence."""
        batch_size = ops.shape(byte_hiddens)[0]

        patch_reps = []

        for p in range(num_patches):
            mask = ops.equal(patch_ids, p)

            # Out-of-patch positions are zeroed rather than masked, so they
            # remain in the key and value sequence.
            mask_expanded = ops.expand_dims(mask, axis=-1)
            patch_hiddens = ops.where(
                mask_expanded,
                byte_hiddens,
                ops.zeros_like(byte_hiddens)
            )

            queries = ops.expand_dims(self.query_embeddings, axis=0)
            queries = ops.tile(queries, [batch_size, 1, 1])

            attended = self.attention_layer(
                query=queries,
                value=patch_hiddens,
                key=patch_hiddens,
                training=training
            )

            patch_rep = ops.mean(attended, axis=1)

            patch_reps.append(patch_rep)

        result = ops.stack(patch_reps, axis=1)

        if self.output_projection is not None:
            result = self.output_projection(result)

        return result

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Byte hidden shape ``(batch, seq_len, hidden_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, None, output_dim)``. The patch axis is reported
            as dynamic even though ``call`` always emits ``max_patches``;
            ``LocalEncoder.compute_output_shape`` reports the static count.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        return (batch_size, None, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'pooling_method': self.pooling_method,
            'output_dim': self.output_dim,
            'num_queries': self.num_queries,
            'max_patches': self.max_patches
        })
        return config

# ---------------------------------------------------------------------
