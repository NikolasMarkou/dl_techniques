"""Memory write controller — projects pre-block hidden state into M_WM.

Wraps a :class:`WorkingMemoryBank` and right-pads the resulting
``(K_wm, V_wm)`` along the time axis to ``max_seq_len`` so that the
read controller's ``ops.one_hot(num_classes=M_static)`` call has a
static shape (``M_static = S_lt + max_seq_len``).

The controller is stateless across batches: it returns
``(K_wm, V_wm, padding_mask)`` and the parent model wires the dataflow
to the read controller. Storing per-batch state on the controller would
break ``keras.Model.save`` round-trip (per LESSONS — frozen state must
be ``add_weight(trainable=False)`` not a plain attribute set in
``call``).
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

from dl_techniques.models.memory_bank.memory_banks import WorkingMemoryBank


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MemoryWriteController(keras.layers.Layer):
    """Project ``X_W (B, T, D)`` into right-padded ``(K_wm, V_wm)``.

    :param d_k: Key dimensionality (forwarded to :class:`WorkingMemoryBank`).
    :param d_v: Value dimensionality.
    :param embed_dim: Hidden-state dimensionality of ``X_W``.
    :param max_seq_len: Static maximum sequence length. Output keys/values
        are right-padded to this length along axis=1 with zeros, and the
        returned padding mask carries 1.0 on real positions and 0.0 on
        padded positions.
    :param initializer_range: Stddev for projection weight init.
    :param kwargs: Forwarded to :class:`keras.layers.Layer`.
    """

    def __init__(
        self,
        d_k: int,
        d_v: int,
        embed_dim: int,
        max_seq_len: int,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {max_seq_len}"
            )

        self.d_k = d_k
        self.d_v = d_v
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.initializer_range = initializer_range

        self.wm_bank = WorkingMemoryBank(
            d_k=d_k,
            d_v=d_v,
            embed_dim=embed_dim,
            initializer_range=initializer_range,
            name="memory_wm_bank",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.wm_bank.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        x_w: Any,
        training: Optional[bool] = None,
    ) -> Tuple[Any, Any, Any]:
        """Project and right-pad to ``max_seq_len``.

        :param x_w: Hidden state ``(B, T, D)`` with ``T <= max_seq_len``.
        :param training: Forwarded for parity.
        :returns: ``(K_wm_padded (B, max_seq_len, d_k),
                     V_wm_padded (B, max_seq_len, d_v),
                     padding_mask (B, max_seq_len))``.
            ``padding_mask`` is 1.0 for real WM positions, 0.0 for padded
            positions.
        """
        k_wm, v_wm = self.wm_bank(x_w, training=training)

        # Right-pad along axis=1 to max_seq_len via concatenation with
        # zeros — this avoids ops.pad's dynamic-paddings-tensor failure
        # mode under tf.function tracing (some Keras 3 / TF 2.18 backends
        # call `len(paddings)` which fails on symbolic tensors).
        b = ops.shape(k_wm)[0]
        t = ops.shape(k_wm)[1]
        pad_len = self.max_seq_len - t

        zeros_k = ops.zeros((b, pad_len, self.d_k), dtype=k_wm.dtype)
        zeros_v = ops.zeros((b, pad_len, self.d_v), dtype=v_wm.dtype)
        k_wm_padded = ops.concatenate([k_wm, zeros_k], axis=1)
        v_wm_padded = ops.concatenate([v_wm, zeros_v], axis=1)

        # Reshape to pin the static axis-1 size so downstream ops.one_hot
        # has a known M_static.
        k_wm_padded = ops.reshape(
            k_wm_padded, (b, self.max_seq_len, self.d_k),
        )
        v_wm_padded = ops.reshape(
            v_wm_padded, (b, self.max_seq_len, self.d_v),
        )

        # Padding mask: 1.0 on positions [0, T), 0.0 on [T, max_seq_len).
        positions = ops.arange(self.max_seq_len, dtype="int32")
        positions = ops.expand_dims(positions, axis=0)  # (1, max_seq_len)
        positions = ops.broadcast_to(positions, (b, self.max_seq_len))
        t_b = ops.broadcast_to(
            ops.expand_dims(t, axis=0), (b,),
        )
        t_b = ops.expand_dims(t_b, axis=-1)  # (B, 1)
        padding_mask = ops.cast(positions < t_b, "float32")

        return k_wm_padded, v_wm_padded, padding_mask

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        b = input_shape[0]
        return (
            (b, self.max_seq_len, self.d_k),
            (b, self.max_seq_len, self.d_v),
            (b, self.max_seq_len),
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_k": self.d_k,
            "d_v": self.d_v,
            "embed_dim": self.embed_dim,
            "max_seq_len": self.max_seq_len,
            "initializer_range": self.initializer_range,
        })
        return config
