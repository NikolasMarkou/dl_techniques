"""Attention as mean shift on tied query, key and value tokens, the :class:`MeanShiftAttention` layer.

Mean shift moves a point to the kernel-weighted mean of a set of points. With the
Gaussian kernel the weights are a softmax over the keys,
``softmax_j(q.k_j / h^2 - ||k_j||^2 / (2 h^2))``, so one step with bandwidth
``h = head_dim ** 0.25`` is scaled dot-product attention minus a key-norm bias.
This layer projects the tokens once (queries, keys and values are the same
projected tokens), then runs ``num_steps`` damped mean-shift steps per head:
``z <- (1 - eta) z + eta * softmax(logits) @ keys``. In the blurring variant the
keys move together with the queries; in the non-blurring variant they stay at the
projected tokens. The bandwidth is learnable (one value, one per head, or one per
token) and is kept inside ``[min_bandwidth, max_bandwidth]``. The von Mises-Fisher
kernel does the same on L2-normalised tokens. Every logit and softmax runs in at
least float32, also under ``mixed_float16``.

References:
    - Cheng, 1995. Mean shift, mode seeking, and clustering.
    - Comaniciu and Meer, 2002. Mean shift: a robust approach toward feature
      space analysis.
    - Carreira-Perpinan, 2015. A review of mean-shift algorithms for clustering.
      (https://arxiv.org/abs/1503.00687)
    - Vaswani et al., 2017. Attention is all you need.
      (https://arxiv.org/abs/1706.03762)
    - Ramsauer et al., 2021. Hopfield networks is all you need.
      (https://arxiv.org/abs/2008.02217)
"""

import math
import numbers
import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import clone_initializer
from dl_techniques.layers.mixtures.mean_shift_ops import kernel_logits
from dl_techniques.utils.dtype_policy import accumulation_dtype, stability_floor
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tensors import l2_normalize
from dl_techniques.utils.validation import positive_float, positive_int

from .common import apply_attention_mask, validate_head_divisibility

# ---------------------------------------------------------------------

_KERNELS = ("gaussian", "vmf")

_BANDWIDTH_MODES = ("global", "per_head", "per_token")

_NORMALIZE_EPS = 1e-12

# ---------------------------------------------------------------------

def softplus_inverse(y: float) -> float:
    """Return ``x`` with ``softplus(x) = y`` for a positive ``y``.

    Computed as ``y + log(-expm1(-y))``, which is exact for every ``y > 0`` in
    double precision: ``y = 1e-6`` gives about ``-13.8`` and ``y = 100`` gives
    ``100 - 3.7e-44``. The result is a Python float.

    :param y: target value of the softplus, strictly positive.
    :type y: float
    :return: the pre-image of ``y`` under the softplus.
    :rtype: float
    :raises ValueError: if ``y`` is not strictly positive and finite.
    """
    if not math.isfinite(y) or y <= 0.0:
        raise ValueError(f"softplus_inverse needs a finite y > 0, got {y!r}")
    # DECISION plan-2026-10-29T104646-27f045c5/D-011: the bandwidth initialiser inverts the softplus with y + log(-expm1(-y)).
    # Do not clamp y to a floor or use log(expm1(y)); a hard-floor inverse collapsed draws elsewhere. See decisions.md.
    return y + math.log(-math.expm1(-y))

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.mean_shift_attention")
class MeanShiftAttention(keras.layers.Layer):
    """
    Attention whose forward pass is ``num_steps`` mean-shift steps on tied tokens.

    One shared projection maps the tokens to ``num_heads`` heads of ``head_dim``
    features; queries, keys and values are all these projected tokens. Each step
    moves every query to the kernel-weighted mean of the keys, damped by the step
    size ``eta``: ``z <- (1 - eta) z + eta * softmax(kernel_logits(z, keys, h)) @ keys``.
    Blurring keeps ``keys = z`` (the key set moves with the queries); non-blurring
    keeps ``keys`` at the projected tokens. With ``num_steps=1``, the Gaussian
    kernel and the default bandwidth ``head_dim ** 0.25`` the weights equal the
    scaled dot-product weights minus a key-norm bias.

    Architecture:

    .. code-block:: text

        inputs [B, N, dim]            (cross form: [query, key_value], no blurring)
                │
                ▼
        ┌─────────────────────────────┐
        │ projection: Dense(H * d)    │  tied Q = K = V
        └──────────────┬──────────────┘
                       ▼
        split heads [B, H, N, d]; vMF: L2-normalise per head
                       │
                       ▼   z = queries, keys0 = projected key tokens
        ┌─ repeat num_steps times ─────────────────────────────┐
        │  keys = z (blurring)  or  keys0 (non-blurring)        │
        │          │                                            │
        │          ▼                                            │
        │  logits = kernel_logits(z, keys, h)   [B, H, N, Nk]   │
        │          │       h: global / per_head / per_token     │
        │          ▼                                            │
        │  shift to kept max, mask (1 = keep), softmax, dropout   │
        │          │                                            │
        │          ▼                                            │
        │  z = (1 - eta) z + eta * weights @ keys               │
        │  vMF: z = normalise(z)                                │
        └──────────────────────┬────────────────────────────────┘
                               ▼
        zero the rows with no kept key; merge heads [B, N, H * d]
                               │
                               ▼
        ┌─────────────────────────────┐
        │ output_projection: Dense(dim)│
        └──────────────┬──────────────┘
                       ▼
        output [B, N, dim]   (+ last-step weights [B, H, N, Nk])

    The mask is a keep mask (1 or True keeps a key): rank 2 ``(B, Nk)``, rank 3
    ``(B, Nq, Nk)`` or rank 4 ``(B, 1 or H, Nq, Nk)``. A masked key has exactly
    zero weight at every step, at any bandwidth: the logits of each row are shifted
    so that its best kept logit is 0 before the mask is applied, so a kept logit far
    below the mask constant (a small bandwidth, widely spread tokens) cannot lose to
    a masked key. A query row with no kept key (a pairwise mask that forbids every
    key, or a key set that is entirely padding) has attention weights of zeros at
    every step, does not move, and gets an output of zeros; under blurring it
    therefore never hands a forbidden key to another row. A padded query under a
    rank-2 padding mask still has kept keys, so its output is computed and left to
    the caller to ignore. The Gaussian centre is the mean of the keys that at least
    one query row keeps, for every mask rank (per head under a rank-4 mask), so a
    value stored at a key that no row keeps cannot set it. Under blurring a
    pairwise mask that is not transitive still lets a row read the moved value of a
    key, and that value has read other keys, so a read can span two hops.

    Under a causal mask the centre still includes every key (each key is kept by
    some row), so a later token can change an earlier row through the centre at
    float32 rounding level, and the change grows with the token magnitude relative
    to the output scale. With the default Gaussian bandwidth, 8 tokens and an output
    scale of 4, setting a future token to 1e1, 1e2, 1e3 and 1e4 changes the earlier
    rows by 6e-7, 2e-5, 3.3e-3 and 0.49 (without blurring). The change is exactly 0
    at ``bandwidth_init=0.3`` and for the vMF kernel at 0.5, and a value at a padded
    key changes no kept row under a rank-2 padding mask, alone or combined with a
    causal mask. For strictly causal use keep the token scale moderate (normalise
    the inputs) until a causal-safe centre is added.

    Input shape:
        ``(batch_size, seq_len, dim)``, or a list ``[query, key_value]`` of two such
        tensors (only with ``blurring=False``). Rank 4 raises ``ValueError``.

    Output shape:
        ``(batch_size, seq_len_query, dim)``.

    Example:

    .. code-block:: python

        layer = MeanShiftAttention(dim=64, num_heads=4, num_steps=2)
        outputs = layer(keras.random.normal((2, 10, 64)))

    :param dim: Token width. Positive integer.
    :type dim: int
    :param num_heads: Number of heads. Positive integer.
    :type num_heads: int
    :param head_dim: Features per head. ``None`` means ``dim // num_heads`` and then
        ``dim`` must be divisible by ``num_heads``.
    :type head_dim: Optional[int]
    :param num_steps: Number of mean-shift steps, a Python int >= 1.
    :type num_steps: int
    :param blurring: Move the keys together with the queries. Must be False for the
        ``[query, key_value]`` input form.
    :type blurring: bool
    :param kernel: ``"gaussian"`` or ``"vmf"`` (L2-normalised tokens).
    :type kernel: str
    :param bandwidth_mode: ``"global"`` (one learnable scalar), ``"per_head"`` (one
        per head) or ``"per_token"`` (one per head and query token, from a linear
        map of the projected tokens).
    :type bandwidth_mode: str
    :param bandwidth_init: Initial bandwidth in ``(min_bandwidth, max_bandwidth]``.
        ``None`` means ``head_dim ** 0.25``. For ``kernel="vmf"`` that default gives
        ``kappa = 1 / sqrt(head_dim)`` on unit vectors, a nearly uniform kernel
        (measured at dim 64, 8 heads, 32 tokens: the largest weight is 0.047 against
        a uniform 0.031, while the Gaussian default reaches 0.72), so a smaller
        value such as 0.3 is the usual choice for vMF.
    :type bandwidth_init: Optional[float]
    :param min_bandwidth: Lower end of the bandwidth range, positive.
    :type min_bandwidth: float
    :param max_bandwidth: Upper end of the bandwidth range, larger than
        ``min_bandwidth``.
    :type max_bandwidth: float
    :param step_size: Damping ``eta`` in the open interval (0, 2). With
        ``learnable_step_size`` it is the initial value of a per-head
        ``2 * sigmoid(raw)``.
    :type step_size: float
    :param learnable_step_size: Learn one step size per head.
    :type learnable_step_size: bool
    :param dropout_rate: Dropout on the attention weights, in ``[0, 1)``.
    :type dropout_rate: float
    :param use_bias: Add a bias to the two projections.
    :type use_bias: bool
    :param kernel_initializer: Initializer of the two projection kernels; each
        projection gets its own independent clone.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer of the two projection kernels.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Passed to :class:`keras.layers.Layer`.

    :raises ValueError: if an argument is out of range or of the wrong type, if the
        input is not rank 3 with last axis ``dim``, if the list input does not have
        two entries or is used with ``blurring=True``.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            head_dim: Optional[int] = None,
            num_steps: int = 1,
            blurring: bool = True,
            kernel: str = "gaussian",
            bandwidth_mode: str = "per_head",
            bandwidth_init: Optional[float] = None,
            min_bandwidth: float = 1e-2,
            max_bandwidth: float = 1e2,
            step_size: float = 1.0,
            learnable_step_size: bool = False,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        """Initialise the layer and create its sub-layers."""
        super().__init__(**kwargs)

        owner = "MeanShiftAttention"
        self.dim = positive_int(dim, "dim", owner)
        self.num_heads = positive_int(num_heads, "num_heads", owner)
        if head_dim is None:
            validate_head_divisibility(self.dim, self.num_heads)
            resolved_head_dim = self.dim // self.num_heads
        else:
            resolved_head_dim = positive_int(head_dim, "head_dim", owner)
        self._head_dim_arg = head_dim
        self.head_dim = resolved_head_dim
        self.num_steps = positive_int(num_steps, "num_steps", owner)

        for name, value in (
                ("blurring", blurring),
                ("learnable_step_size", learnable_step_size),
                ("use_bias", use_bias),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{owner}: {name} must be a bool, got {value!r}")
        self.blurring = blurring
        self.learnable_step_size = learnable_step_size
        self.use_bias = use_bias

        if kernel not in _KERNELS:
            raise ValueError(
                f"{owner}: kernel must be one of {list(_KERNELS)}, got {kernel!r}"
            )
        self.kernel = kernel
        if bandwidth_mode not in _BANDWIDTH_MODES:
            raise ValueError(
                f"{owner}: bandwidth_mode must be one of {list(_BANDWIDTH_MODES)}, "
                f"got {bandwidth_mode!r}"
            )
        self.bandwidth_mode = bandwidth_mode

        self.min_bandwidth = positive_float(min_bandwidth, "min_bandwidth", owner)
        self.max_bandwidth = positive_float(max_bandwidth, "max_bandwidth", owner)
        if self.min_bandwidth >= self.max_bandwidth:
            raise ValueError(
                f"{owner}: min_bandwidth must be below max_bandwidth, got "
                f"min_bandwidth={min_bandwidth!r} and max_bandwidth={max_bandwidth!r}"
            )
        self._bandwidth_init_arg = bandwidth_init
        if bandwidth_init is None:
            resolved_init = float(self.head_dim) ** 0.25
        else:
            resolved_init = positive_float(bandwidth_init, "bandwidth_init", owner)
        if not self.min_bandwidth < resolved_init <= self.max_bandwidth:
            raise ValueError(
                f"{owner}: bandwidth_init must lie in (min_bandwidth, max_bandwidth] "
                f"= ({self.min_bandwidth}, {self.max_bandwidth}], got "
                f"{bandwidth_init!r} (resolved to {resolved_init})"
            )
        self.bandwidth_init = resolved_init

        if (
                isinstance(step_size, bool)
                or not isinstance(step_size, numbers.Real)
                or not 0.0 < step_size < 2.0
        ):
            raise ValueError(
                f"{owner}: step_size must be a number in (0, 2), got {step_size!r}"
            )
        self.step_size = float(step_size)
        if (
                isinstance(dropout_rate, bool)
                or not isinstance(dropout_rate, numbers.Real)
                or not 0.0 <= dropout_rate < 1.0
        ):
            raise ValueError(
                f"{owner}: dropout_rate must be a number in [0, 1), got {dropout_rate!r}"
            )
        self.dropout_rate = float(dropout_rate)

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        width = self.num_heads * self.head_dim
        # DECISION plan-2026-08-22T035419-a11304c8/D-200: each projection gets its
        # own clone_initializer(...), never a shared instance, or the two kernels replay one draw. See decisions.md.
        self.projection = keras.layers.Dense(
            width,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="projection",
        )
        self.output_projection = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self.kernel_regularizer,
            name="output_projection",
        )

        # The dropout runs on float32 weights, so it computes in the accumulation dtype.
        self.attention_dropout = keras.layers.Dropout(
            self.dropout_rate,
            dtype=accumulation_dtype(self.compute_dtype),
            name="attention_dropout",
        )

        self._raw_bandwidth_init = softplus_inverse(
            self.bandwidth_init - self.min_bandwidth
        )
        self.bandwidth_projection = None
        if self.bandwidth_mode == "per_token":
            self.bandwidth_projection = keras.layers.Dense(
                self.num_heads,
                kernel_initializer="zeros",
                bias_initializer=keras.initializers.Constant(self._raw_bandwidth_init),
                name="bandwidth_projection",
            )
        self.raw_bandwidth = None
        self.raw_step = None

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]],
    ) -> None:
        """Build the sub-layers and create the bandwidth and step-size weights.

        :param input_shape: shape ``(B, N, dim)`` or a list of two such shapes.
        :type input_shape: Union[tuple, list]
        :raises ValueError: if a shape is not rank 3 with last axis ``dim``, or if
            the list form has a length other than 2 or is used with blurring.
        """
        if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(
                input_shape[0], (list, tuple)
        ):
            shapes = [tuple(shape) for shape in input_shape]
            self._check_cross_form(len(shapes))
        else:
            shapes = [tuple(input_shape)]
        for shape in shapes:
            if len(shape) != 3:
                raise ValueError(
                    f"MeanShiftAttention expects inputs of shape (batch, seq_len, "
                    f"{self.dim}), got rank {len(shape)}: {shape}"
                )
            if shape[-1] != self.dim:
                raise ValueError(
                    f"MeanShiftAttention expects last axis {self.dim}, got {shape[-1]}"
                )

        width = self.num_heads * self.head_dim
        self.projection.build((None, None, self.dim))
        self.output_projection.build((None, None, width))
        if self.bandwidth_projection is not None:
            self.bandwidth_projection.build((None, None, width))
        elif self.bandwidth_mode == "per_head":
            self.raw_bandwidth = self.add_weight(
                name="raw_bandwidth",
                shape=(self.num_heads,),
                initializer=keras.initializers.Constant(self._raw_bandwidth_init),
                trainable=True,
                autocast=False,
            )
        else:
            self.raw_bandwidth = self.add_weight(
                name="raw_bandwidth",
                shape=(),
                initializer=keras.initializers.Constant(self._raw_bandwidth_init),
                trainable=True,
                autocast=False,
            )
        if self.learnable_step_size:
            self.raw_step = self.add_weight(
                name="raw_step",
                shape=(self.num_heads,),
                initializer=keras.initializers.Constant(
                    math.log(self.step_size / (2.0 - self.step_size))
                ),
                trainable=True,
                autocast=False,
            )
        super().build(input_shape)

    def _check_cross_form(self, count: int) -> None:
        """Raise unless a list input has two entries and the keys do not blur.

        :param count: number of entries in the list input.
        :type count: int
        :raises ValueError: if ``count`` is not 2 or ``blurring`` is True.
        """
        if count != 2:
            raise ValueError(
                f"MeanShiftAttention: a list input must be [query, key_value], "
                f"got {count} entries"
            )
        if self.blurring:
            raise ValueError(
                "MeanShiftAttention: the [query, key_value] form needs blurring=False "
                "because blurring moves the key set with the queries"
            )

    def _normalize(self, x: keras.KerasTensor, acc: str) -> keras.KerasTensor:
        """L2-normalise the last axis; a zero row stays zero.

        :param x: tensor ``(..., d)`` in the accumulation dtype.
        :type x: keras.KerasTensor
        :param acc: accumulation dtype name.
        :type acc: str
        :return: unit-norm rows.
        :rtype: keras.KerasTensor
        """
        return l2_normalize(x, axis=-1, eps=stability_floor(acc, _NORMALIZE_EPS))

    def _split_heads(self, x: keras.KerasTensor, acc: str) -> keras.KerasTensor:
        """Reshape projected tokens ``(B, N, H*d)`` to ``(B, H, N, d)`` in ``acc``.

        :param x: projected tokens.
        :type x: keras.KerasTensor
        :param acc: accumulation dtype name.
        :type acc: str
        :return: per-head tokens.
        :rtype: keras.KerasTensor
        """
        shape = keras.ops.shape(x)
        x = keras.ops.reshape(x, (shape[0], shape[1], self.num_heads, self.head_dim))
        return keras.ops.cast(keras.ops.transpose(x, (0, 2, 1, 3)), acc)

    def _bandwidth(self, tokens: keras.KerasTensor, acc: str) -> keras.KerasTensor:
        """Return the bandwidth ``h`` in ``[min_bandwidth, max_bandwidth]``.

        :param tokens: projected query tokens ``(B, N, H*d)``.
        :type tokens: keras.KerasTensor
        :param acc: accumulation dtype name.
        :type acc: str
        :return: ``h`` shaped ``()`` (global), ``(1, H, 1, 1)`` (per head) or
            ``(B, H, N, 1)`` (per token), in ``acc``.
        :rtype: keras.KerasTensor
        """
        if self.bandwidth_mode == "per_token":
            raw = self.bandwidth_projection(tokens)
            raw = keras.ops.expand_dims(keras.ops.transpose(raw, (0, 2, 1)), axis=-1)
        elif self.bandwidth_mode == "per_head":
            raw = keras.ops.reshape(self.raw_bandwidth, (1, self.num_heads, 1, 1))
        else:
            raw = self.raw_bandwidth
        raw = keras.ops.cast(raw, acc)
        h = self.min_bandwidth + keras.ops.softplus(raw)
        return keras.ops.clip(h, self.min_bandwidth, self.max_bandwidth)

    def _step_size(self, acc: str) -> Union[float, keras.KerasTensor]:
        """Return the damping ``eta``: a Python float, or ``(1, H, 1, 1)`` if learnable.

        :param acc: accumulation dtype name.
        :type acc: str
        :return: the step size.
        :rtype: Union[float, keras.KerasTensor]
        """
        if not self.learnable_step_size:
            return self.step_size
        eta = 2.0 * keras.ops.sigmoid(keras.ops.cast(self.raw_step, acc))
        return keras.ops.reshape(eta, (1, self.num_heads, 1, 1))

    @staticmethod
    def _keep_mask(attention_mask: keras.KerasTensor) -> keras.KerasTensor:
        """Turn a keep mask of rank 2, 3 or 4 into a boolean ``(B, 1 or H, Nq or 1, Nk)``.

        :param attention_mask: keep mask, 1 or True keeps a key.
        :type attention_mask: keras.KerasTensor
        :return: boolean mask of rank 4.
        :rtype: keras.KerasTensor
        :raises ValueError: if the rank is not 2, 3 or 4.
        """
        rank = len(attention_mask.shape)
        keep = keras.ops.cast(attention_mask, "bool")
        if rank == 2:
            return keras.ops.expand_dims(keras.ops.expand_dims(keep, 1), 1)
        if rank == 3:
            return keras.ops.expand_dims(keep, 1)
        if rank == 4:
            return keep
        raise ValueError(
            f"MeanShiftAttention: attention_mask must have rank 2, 3 or 4, got rank {rank}"
        )

    @staticmethod
    def _rows_with_a_kept_key(keep: keras.KerasTensor) -> keras.KerasTensor:
        """Mark the query rows that keep at least one key.

        :param keep: boolean keep mask ``(B, 1 or H, Nq or 1, Nk)``.
        :type keep: keras.KerasTensor
        :return: boolean ``(B, 1 or H, Nq or 1, 1)``, False where a row keeps nothing.
        :rtype: keras.KerasTensor
        """
        return keras.ops.any(keep, axis=-1, keepdims=True)

    @staticmethod
    def _shift_to_kept_maximum(
            logits: keras.KerasTensor, keep: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Shift every row so its best kept logit is 0 and park the masked logits at 0.

        The additive mask bias is a fixed constant, so a kept logit far below it (a
        small bandwidth, widely spread tokens) would lose to a masked key. After this
        shift the best kept logit is 0 and every masked entry is 0 before the bias, so
        the bias puts it strictly below the best kept key and its weight is exactly 0.
        A row with no kept key shifts by 0 and has all entries 0. The row maximum
        carries no gradient and the softmax is unchanged by the shift.

        :param logits: logits ``(B, H, Nq, Nk)`` in the accumulation dtype.
        :type logits: keras.KerasTensor
        :param keep: boolean keep mask ``(B, 1 or H, Nq or 1, Nk)``.
        :type keep: keras.KerasTensor
        :return: shifted logits, shaped like ``logits``, finite wherever ``logits`` is.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-10-29T104646-27f045c5/D-027: shift by the kept maximum and park masked logits at 0 before the bias.
        # Do not rely on the fixed mask constant alone: at a small bandwidth a kept logit is far below it. See decisions.md.
        kept_max = keras.ops.stop_gradient(
            keras.ops.max(
                keras.ops.where(keep, logits, keras.ops.full_like(logits, -math.inf)),
                axis=-1,
                keepdims=True,
            )
        )
        kept_max = keras.ops.where(
            keras.ops.any(keep, axis=-1, keepdims=True),
            kept_max,
            keras.ops.zeros_like(kept_max),
        )
        return keras.ops.where(
            keep, logits - kept_max, keras.ops.zeros_like(logits)
        )

    @staticmethod
    def _hold_still(
            live: keras.KerasTensor, moved: keras.KerasTensor, z: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Keep the current ``z`` where a query row has no kept key.

        :param live: boolean ``(B, 1 or H, Nq or 1, 1)``, False for a row with no kept key.
        :type live: keras.KerasTensor
        :param moved: the shifted queries ``(B, H, Nq, d)``.
        :type moved: keras.KerasTensor
        :param z: the current queries ``(B, H, Nq, d)``.
        :type z: keras.KerasTensor
        :return: ``moved`` in live rows, ``z`` in the others.
        :rtype: keras.KerasTensor
        """
        return keras.ops.where(live, moved, z)

    def call(
            self,
            inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
            attention_mask: Optional[keras.KerasTensor] = None,
            return_attention_scores: bool = False,
            training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Run the mean-shift steps.

        :param inputs: tokens ``(B, N, dim)``, or ``[query, key_value]`` (only with
            ``blurring=False``).
        :type inputs: Union[keras.KerasTensor, list]
        :param attention_mask: optional keep mask of rank 2 ``(B, Nk)``, rank 3
            ``(B, Nq, Nk)`` or rank 4 ``(B, 1 or H, Nq, Nk)``; 1 or True keeps.
        :type attention_mask: Optional[keras.KerasTensor]
        :param return_attention_scores: also return the weights of the last step.
        :type return_attention_scores: bool
        :param training: whether dropout is active.
        :type training: Optional[bool]
        :return: output ``(B, Nq, dim)``, or ``(output, weights)`` with ``weights``
            of shape ``(B, H, Nq, Nk)``; both in the compute dtype.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        :raises ValueError: for a list input of the wrong length or with blurring,
            or a mask of rank other than 2, 3 or 4.
        """
        compute = self.compute_dtype
        acc = accumulation_dtype(compute)

        if isinstance(inputs, (list, tuple)):
            self._check_cross_form(len(inputs))
            query_tokens, key_tokens = inputs
        else:
            query_tokens = key_tokens = inputs

        projected = self.projection(query_tokens)
        z = self._split_heads(projected, acc)
        if key_tokens is query_tokens:
            keys0 = z
        else:
            keys0 = self._split_heads(self.projection(key_tokens), acc)
        if self.kernel == "vmf":
            z = self._normalize(z, acc)
            keys0 = self._normalize(keys0, acc)

        h = self._bandwidth(projected, acc)
        eta = self._step_size(acc)
        keep = None if attention_mask is None else self._keep_mask(attention_mask)
        live = None
        centre_mask = None
        if keep is not None:
            live = self._rows_with_a_kept_key(keep)
            # The keys some query row keeps set the Gaussian centre, so a key no row keeps cannot.
            centre_mask = keras.ops.any(keep, axis=2)

        weights = None
        for _ in range(self.num_steps):
            keys = z if self.blurring else keys0
            logits = kernel_logits(z, keys, h, self.kernel, key_mask=centre_mask)
            if keep is not None:
                logits = self._shift_to_kept_maximum(logits, keep)
                logits = apply_attention_mask(
                    logits, keep, out_dtype=acc, rescue_axis=-1
                )
            # DECISION plan-2026-10-29T104646-27f045c5/D-021: exp(logits - logsumexp) replaces keras.ops.softmax.
            # Do not restore softmax: a size-1 key axis warns and the warnings-as-errors guard fails. See decisions.md.
            weights = keras.ops.exp(
                logits - keras.ops.logsumexp(logits, axis=-1, keepdims=True)
            )
            if live is not None:
                weights = weights * keras.ops.cast(live, acc)
            if self.dropout_rate > 0.0:
                weights = self.attention_dropout(weights, training=training)
            moved = keras.ops.matmul(weights, keys)
            if live is not None:
                # DECISION plan-2026-10-29T104646-27f045c5/D-004: a query row with no kept key stays put at every step and is zeroed after the loop.
                # Do not let the rescued keep-all row move: as a blurring key it would hand forbidden keys to other rows. See decisions.md.
                moved = self._hold_still(live, moved, z)
            z = (1.0 - eta) * z + eta * moved
            if self.kernel == "vmf":
                z = self._normalize(z, acc)

        if live is not None:
            z = z * keras.ops.cast(live, acc)

        shape = keras.ops.shape(z)
        merged = keras.ops.reshape(
            keras.ops.transpose(z, (0, 2, 1, 3)),
            (shape[0], shape[2], self.num_heads * self.head_dim),
        )
        output = self.output_projection(keras.ops.cast(merged, compute))
        output = keras.ops.cast(output, compute)
        if return_attention_scores:
            return output, keras.ops.cast(weights, compute)
        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]],
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, the query shape ``(B, Nq, dim)``.

        :param input_shape: a shape tuple or a list of two shape tuples.
        :type input_shape: Union[tuple, list]
        :return: ``(B, Nq, dim)``.
        :rtype: tuple
        """
        if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(
                input_shape[0], (list, tuple)
        ):
            query_shape = tuple(input_shape[0])
        else:
            query_shape = tuple(input_shape)
        return query_shape[:-1] + (self.dim,)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments, ``None`` kept as given.

        :return: configuration dictionary.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "head_dim": self._head_dim_arg,
            "num_steps": self.num_steps,
            "blurring": self.blurring,
            "kernel": self.kernel,
            "bandwidth_mode": self.bandwidth_mode,
            "bandwidth_init": self._bandwidth_init_arg,
            "min_bandwidth": self.min_bandwidth,
            "max_bandwidth": self.max_bandwidth,
            "step_size": self.step_size,
            "learnable_step_size": self.learnable_step_size,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------
