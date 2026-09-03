"""The Energy Transformer block, its Hopfield memory module, and the optional weighted-
adjacency projector, built by :class:`EnergyTransformer`, :class:`HopfieldNetwork`, and
:class:`WeightedAdjacencyProjector`.

The block replaces the usual attention-then-FFN stack with explicit gradient descent on
one scalar energy. Each step normalizes the token state, evaluates the negative gradient
of `E(g) = E_ATT(g) + E_HN(g)` with respect to that normalized state, and adds it back to
the token state:

.. code-block:: text

    for t in 1..T:
        g      = EnergyLayerNorm(x)
        update = attn.update(g) + hopfield.update(g)   # == -dE/dg
        x      = x + alpha * update

Every gradient is hand-coded in closed form with `keras.ops` (no autodiff in this
package). `HopfieldNetwork` replaces the usual feed-forward block: one tied `(K, D)`
memory matrix applied per token, with no bias and no independent up/down projections.
`WeightedAdjacencyProjector` is an optional trainable reweighting of a graph adjacency,
consumed by `EnergyAttention`.

A caller who wants the energy trace back (`return_energy=True`) gets it in float32 even
under a mixed-precision policy, because a realistic trace magnitude overflows float16; a
head consuming that trace must itself be built with `dtype='float32'`.

References:
    - Hoover et al., "Energy Transformer", NeurIPS 2023. (https://arxiv.org/abs/2302.07253)
    - Ramsauer et al., "Hopfield Networks is All You Need", ICLR 2021.
"""

import math
import keras
from keras import ops, initializers
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# DECISION plan_2026-07-13_57c9833e/D-004: direct imports of the concrete norm/attention
# classes, not factory calls -- this block calls their energy()/update() duck-typed pair,
# which a factory-returned generic Layer does not guarantee. See decisions.md.
from dl_techniques.layers.norms.energy_layer_norm import EnergyLayerNorm
from dl_techniques.layers.attention.energy_attention import EnergyAttention

# Imported, not re-implemented, so the "reduce in at least float32" rule has one definition.
from dl_techniques.layers.attention.energy_attention import _mask_dtype, _token_keep
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

# The only supported Hopfield activations. See the D-005 anchor in HopfieldNetwork.__init__.
_VALID_ACTIVATIONS = ('relu', 'softmax')


@register_dl_technique("dl_techniques.layers.transformers.energy_transformer")
class HopfieldNetwork(keras.layers.Layer):
    """Energy Transformer Hopfield / associative-memory module: tied weights, no bias.

    Exposes a scalar per-token energy `E_HN(g)` together with its exact closed-form
    negative gradient, so an `EnergyTransformer` block can perform gradient descent on
    `E_ATT + E_HN`. This is the paper's analog of the feed-forward block, but it is not an
    MLP: one tied `(K, D)` matrix `xi` is used in both directions (up-project, then
    down-project by its transpose), there is no bias, and the activation is the
    derivative of the energy's integrand, not a pointwise nonlinearity between two layers.
    It is not registered in the FFN factory for the same reason.

    # DECISION plan_2026-07-13_57c9833e/D-002: not registered in the FFN factory --
    # registering an associative memory as an FFN type would invite swapping it for a
    # real FFN and silently destroying the descent guarantee. See decisions.md.

    Mathematics (`B`=batch, `N`=tokens, `D`=`dim`, `K`=`hopfield_dim`):

    .. code-block:: text

        h_{n k} = sum_d xi_{k d} g_{n d}                      # (B, N, K), per token
        E_HN    = - sum_n sum_k G(h_{n k})       where  G' = r
        -dE_HN/dg_{n d} = sum_k xi_{k d} r(h_{n k})           # (B, N, D)

    Each activation carries a matched energy and gradient factor `r`. Pairing one
    activation's energy with the other's gradient factor is a silent break: the layer
    still runs and still trains, but the energy stops descending.

    .. code-block:: text

        activation   r(h)                       E_HN
        ----------   ------------------------   -----------------------------------------
        'relu'       relu(h)                    -0.5 * sum_{n,k} relu(h_{n k})^2
        'softmax'    softmax_k(beta * h)_{n k}  -(1/beta) * sum_n logsumexp_k(beta*h_{n k})

    In the `'softmax'` case both the energy's `logsumexp` and the gradient's softmax run
    over the memory axis `k`, never the token axis `n`.

    Every token is processed independently, so this layer takes no `attention_mask`: a
    token cannot influence any other token, and only `EnergyAttention` mixes tokens.

    Note:
        Standalone use outside `EnergyTransformer` needs manual masking. `call()` takes no
        `mask` keyword and `supports_masking` is `False`, so a Keras-propagated mask (for
        example from an upstream `Embedding(mask_zero=True)`) is dropped with a warning,
        not honoured. A dropped PAD token is not silently harmless here: `mask_zero=True`
        emits a real, non-zero embedding row for id 0 and marks it as metadata only, so an
        unmasked PAD token gets a real update of the same order of magnitude as a real
        token. Call `update(g, mask=keep)` / `energy(g, mask=keep)` with an explicit
        rank-2 `(B, N)` keep, or use it inside `EnergyTransformer`, which already does.

    :meth:`update` returns `-dE/dg`, the descent direction, not the gradient — a consumer
    adds `step_size * update`. This layer and `EnergyAttention` both expose the same
    duck-typed trio, `energy(g) -> (B,)` / `update(g) -> (B, N, D)` / `call(...) ->
    update(...)`, by convention rather than a shared base class.

    :param dim: Token embedding dimension ``D``.
    :type dim: int
    :param hopfield_dim: Number of stored memories ``K`` (the rows of ``xi``).
    :type hopfield_dim: int
    :param activation: Energy/gradient pair — ``'relu'`` (default, the paper's config for
        both its headline models) or ``'softmax'`` (the modern Hopfield energy).
    :type activation: str
    :param hopfield_beta: Inverse temperature of the ``'softmax'`` branch, ignored by
        ``'relu'``. Defaults to ``1.0``.
    :type hopfield_beta: float
    :param kernel_initializer: Initializer for ``xi``. Defaults to
        ``TruncatedNormal(stddev=0.02)`` (the paper's ``N(0, 0.02)``).
    :type kernel_initializer: Union[str, initializers.Initializer]

    :raises ValueError: If ``dim <= 0``, ``hopfield_dim <= 0``, ``hopfield_beta <= 0``, or
        ``activation`` is not one of ``{'relu', 'softmax'}``.

    Input shape:
        3D tensor ``(batch, num_tokens, dim)``.

    Output shape:
        Identical to the input shape — ``(batch, num_tokens, dim)``.

    :ivar xi: The tied, bias-free memory matrix, shape ``(hopfield_dim, dim)``.
    :vartype xi: keras.Variable

    Example:
        >>> layer = HopfieldNetwork(dim=64, hopfield_dim=256)
        >>> g = keras.random.normal((2, 16, 64))
        >>> layer.energy(g).shape       # scalar energy per batch element
        (2,)
        >>> layer(g).shape              # == layer.update(g) == -dE_HN/dg
        (2, 16, 64)

    References:
        - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253, eq. (5), (9).
        - Ramsauer et al., "Hopfield Networks is All You Need", ICLR 2021 (``'softmax'``).
    """

    def __init__(
        self,
        dim: int,
        hopfield_dim: int,
        activation: str = 'relu',
        hopfield_beta: float = 1.0,
        kernel_initializer: Union[str, initializers.Initializer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ----- validation -----
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(hopfield_dim, int) or hopfield_dim <= 0:
            raise ValueError(
                f"hopfield_dim must be a positive integer, got {hopfield_dim}"
            )

        # DECISION plan_2026-07-13_57c9833e/D-005: 'power' has no call sites and no gradient-
        # oracle coverage; both paper headline configs use 'relu'. See decisions.md.
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {set(_VALID_ACTIVATIONS)}, got "
                f"{activation!r}"
            )

        if not isinstance(hopfield_beta, (int, float)) or hopfield_beta <= 0:
            raise ValueError(
                f"hopfield_beta must be a positive number, got {hopfield_beta}"
            )

        # ----- store ALL configuration -----
        self.dim = int(dim)
        self.hopfield_dim = int(hopfield_dim)
        self.activation = str(activation)
        self.hopfield_beta = float(hopfield_beta)
        self.kernel_initializer = (
            initializers.TruncatedNormal(stddev=0.02)
            if kernel_initializer is None
            else initializers.get(kernel_initializer)
        )

        # ----- weights are created in build() -----
        self.xi: Optional[keras.Variable] = None

        logger.debug(
            f"Initialized HopfieldNetwork with dim={self.dim}, "
            f"hopfield_dim={self.hopfield_dim}, activation={self.activation}, "
            f"hopfield_beta={self.hopfield_beta}"
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the single tied, bias-free ``(K, D)`` memory matrix ``xi``.

        :param input_shape: Input shape ``(batch, num_tokens, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the last axis of ``input_shape`` is not ``dim``.
        """
        if self.built:
            return

        feature_dim = input_shape[-1]
        if feature_dim is not None and int(feature_dim) != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} does not match dim={self.dim}"
            )

        # One matrix, used in both directions (up-project via xi, down-project via its
        # transpose). No bias: the closed-form gradient below has no term for one.
        self.xi = self.add_weight(
            name="xi",
            shape=(self.hopfield_dim, self.dim),   # (K, D)
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    # -----------------------------------------------------------------
    # Public API: energy / update / call
    # -----------------------------------------------------------------

    def energy(
        self,
        g: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Scalar Hopfield energy ``E_HN`` per batch element (paper eq. 5).

        .. code-block:: text

            h_{n k} = sum_d xi_{k d} g_{n d}

            'relu'    : E_HN = -0.5 * sum_n keep_n * sum_k relu(h_{n k})^2
            'softmax' : E_HN = -(1/beta) * sum_n keep_n * logsumexp_k( beta * h_{n k} )

        The ``'softmax'`` ``logsumexp`` runs over the memory axis ``k`` (axis ``-1``), not
        the token axis ``n``.

        :meth:`update` must match these formulas exactly — they are the spec the gradient
        oracle checks against.

        :param g: Token state ``(B, N, D)``, typically the output of ``EnergyLayerNorm``.
        :type g: keras.KerasTensor
        :param mask: Optional rank-2 ``(B, N)`` per-token validity mask. A masked-out (PAD)
            token contributes zero to the energy — see the D-005 anchor below.
        :type mask: Optional[keras.KerasTensor]

        :return: Energy of shape ``(B,)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        # A public method callable outside __call__, where Keras has not opened an autocast
        # scope, so a float32 g meeting float16 xi would otherwise raise.
        g = ops.cast(g, self.compute_dtype)

        h = ops.einsum('kd,bnd->bnk', self.xi, g)        # (B, N, K)

        # Reduces in at least float32: an fp16 accumulator overflows well before the layer
        # itself misbehaves, summing over N tokens x K memories.
        reduce_dtype = _mask_dtype(self.compute_dtype)
        h = ops.cast(h, reduce_dtype)

        # Per-token energy first, then the reduction over tokens, so the mask below can
        # gate the token sum.
        if self.activation == 'relu':
            r = ops.relu(h)
            per_token = -0.5 * ops.sum(ops.square(r), axis=-1)           # (B, N)
        else:
            # Softmax over the memory axis, not the token axis: G is not separable per k.
            lse = ops.logsumexp(self.hopfield_beta * h, axis=-1)         # (B, N)
            per_token = -(1.0 / self.hopfield_beta) * lse                # (B, N)

        # DECISION plan_2026-07-13_ca4f71a2/D-005: a masked-out token contributes zero to
        # E_HN; masking only here and not in update() would break update() == -dE/dg. See decisions.md.
        if mask is not None:
            per_token = per_token * _token_keep(mask, reduce_dtype)      # (B, N)

        energy = ops.sum(per_token, axis=1)                              # (B,)

        # DECISION plan_2026-07-13_ca4f71a2/D-005: returned in the reduce dtype, not cast
        # back to compute dtype -- E_HN overflows fp16 at N=1024 otherwise. See decisions.md.
        return energy                                                     # (B,), >= float32

    # -----------------------------------------------------------------

    def update(
        self,
        g: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Return ``-dE_HN/dg``, the descent direction, not the gradient.

        This is the negative gradient. The consumer adds ``step_size * update`` to the
        token state; flipping the sign at the call site inverts the dynamics into energy
        ascent, which still runs and still produces finite outputs.

        .. code-block:: text

            -dE_HN/dg_{n d} = sum_k xi_{k d} r(h_{n k})

            'relu'    : r(h) = relu(h)                    (pairs with -0.5*sum relu(h)^2)
            'softmax' : r(h) = softmax_k(beta * h)        (pairs with -(1/beta)*logsumexp_k)

        The ``r`` branch here and the energy branch in :meth:`energy` are a matched pair
        per activation. Crossing them (relu energy with softmax ``r``, or vice versa)
        still runs and still trains, but ``update`` stops being the gradient of the
        reported energy. Only ``test_gradient_oracle`` (parametrized over both
        activations) proves the pairing.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor
        :param mask: Optional rank-2 ``(B, N)`` per-token validity mask — the same mask
            :meth:`energy` is given. A masked row's update is exactly ``0``, because the
            masked energy does not depend on that token at all.
        :type mask: Optional[keras.KerasTensor]

        :return: ``-dE_HN/dg`` of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        g = ops.cast(g, self.compute_dtype)

        h = ops.einsum('kd,bnd->bnk', self.xi, g)        # (B, N, K)

        if self.activation == 'relu':
            r = ops.relu(h)                                              # (B, N, K)
        else:
            # Softmax over the memory axis, not the token axis, to keep this the gradient
            # of the reported energy and not introduce token mixing.
            r = ops.softmax(self.hopfield_beta * h, axis=-1)             # (B, N, K)

        # Down-project with the same matrix, transposed (tied weights).
        update = ops.einsum('kd,bnk->bnd', self.xi, r)   # == -dE_HN/dg, (B, N, D)

        # DECISION plan_2026-07-13_ca4f71a2/D-005: zero the masked rows -- the derivative
        # of the masked energy, not a safety hack; must match energy()'s mask. See decisions.md.
        if mask is not None:
            keep = _token_keep(mask, self.compute_dtype)                 # (B, N)
            update = update * ops.expand_dims(keep, axis=-1)             # (B, N, 1)

        return update

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Return the energy descent direction ``-dE_HN/dg`` for ``inputs``.

        Unlike an FFN, this does NOT return an up-project / activate / down-project stack
        with independent weights; it returns :meth:`update`, the exact negative gradient of
        :meth:`energy`. There is no ``attention_mask`` argument: the layer is strictly
        per-token, so a token cannot influence any other token and masking is meaningless.

        :param inputs: Token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param training: Unused; the layer is deterministic.
        :type training: Optional[bool]

        :return: Tensor of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        return self.update(inputs)

    # -----------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape (identity — the update lives in the input's space).

        Uses only the passed shape and stored config, never a weight shape, so it is valid
        on an unbuilt layer.

        :param input_shape: Input shape ``(batch, num_tokens, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'hopfield_dim': self.hopfield_dim,
            'activation': self.activation,
            'hopfield_beta': self.hopfield_beta,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        })
        return config

    # -----------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HopfieldNetwork":
        """Reconstruct the layer from its serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]

        :return: A new ``HopfieldNetwork`` instance.
        :rtype: HopfieldNetwork
        """
        config = dict(config)
        if 'kernel_initializer' in config:
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )
        return cls(**config)

# ---------------------------------------------------------------------


# DECISION plan-2026-07-15T053724-78001af1/D-002: own Layer class, not an inline Conv2D
# in the block's call() -- a lazily created trainable sub-layer drops its weights on a
# .keras round-trip, and X⊗X vs A' pairing order needs isolated test coverage. See decisions.md.
@register_dl_technique("dl_techniques.layers.transformers.energy_transformer")
class WeightedAdjacencyProjector(keras.layers.Layer):
    """Trainable per-head weighted adjacency ``Ŵ`` from tokens plus a binary adjacency.

    Realizes the Energy Transformer's graph score reweighting (paper App D.1):
    ``Â = Conv2D(X⊗X) ⊙ A'``. For every ordered token pair it forms the outer product of
    the two token embeddings, runs a small Conv2D over the resulting ``N x N`` grid to
    produce ``H`` per-head edge weights, and zeros the weight on every non-edge. The
    result is a finite ``(B, H, N, N)`` tensor that :class:`EnergyTransformer` hoists once
    per ``call()`` and threads into ``EnergyAttention`` as ``adjacency_weight``.

    Architecture:

    .. code-block:: text

        x [B, N, D]         adjacency [B, N, N]
          │
          ▼ (proj_dim only)
        ┌──────────┐
        │  Dense   │  D -> P
        └────┬─────┘
             ▼
        outer product, per pair            [B, N, N, P, P]
             │
             ▼ reshape
        ┌──────────┐
        │  Conv2D  │  P^2 -> H channels
        └────┬─────┘
             ▼
        transpose, then multiply by adjacency
             │
             ▼
        Ŵ [B, H, N, N]  (finite)

    Mathematics (``B``=batch, ``N``=tokens, ``D``=``embed_dim``, ``P``=``proj_dim`` or
    ``D``, ``H``=``num_heads``):

    .. code-block:: text

        X'          = X  (or  Dense_P(X)  when proj_dim is set)      # (B, N, P)
        outer_{b n m p q} = X'_{b n p} * X'_{b m q}                  # (B, N, N, P, P)
        outer       = reshape -> (B, N, N, P^2)                      # P^2 conv channels
        raw_{b n m h} = Conv2D(H, kernel_size, 'same')(outer)        # (B, N, N, H)
        Ŵ_{b h n m}  = transpose(raw) * A'_{b n m}                   # (B, H, N, N), finite

    The Conv2D sees ``P^2`` input channels, so at the paper's graph width (``D=128``) that
    is 16384 channels through the conv. Setting ``proj_dim = P < D`` projects the tokens
    down first so the conv sees only ``P^2`` channels; ``None`` (default) keeps the full
    ``D^2``, which is fine for a small ``D`` or small ``N``.

    Zeroing ``Ŵ`` on non-edges matches the paper's ``⊙ A'``, but the actual non-edge
    suppression is the existing keep-bias already applied in
    :meth:`EnergyAttention._project` from the same binary adjacency. ``Ŵ`` therefore
    stays finite; it never carries ``-inf``.

    :param num_heads: Number of attention heads ``H`` (the Conv2D output channels). Must
        match the consuming :class:`EnergyAttention`.
    :type num_heads: int
    :param embed_dim: Token embedding dimension ``D``.
    :type embed_dim: int
    :param kernel_size: Conv2D spatial kernel size over the ``N × N`` grid. ``1`` (default,
        the paper's per-pair ``1 × 1`` conv). Larger values mix neighbouring pairs.
    :type kernel_size: int
    :param proj_dim: If set, first project tokens ``D -> proj_dim`` (a ``Dense``) so the conv
        sees ``proj_dim^2`` channels instead of ``D^2`` — the OOM escape hatch. ``None``
        (default) keeps the full ``D^2``.
    :type proj_dim: Optional[int]

    :raises ValueError: If ``num_heads <= 0``, ``embed_dim <= 0``, ``kernel_size <= 0`` or an
        explicit ``proj_dim <= 0``.

    Input shape:
        A 2-tuple ``((B, N, D), (B, N, N))`` — the token state and the binary adjacency.

    Output shape:
        ``(B, H, N, N)`` — the per-head finite weighted adjacency ``Ŵ``.
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        kernel_size: int = 1,
        proj_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive integer, got {embed_dim}")
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive integer, got {kernel_size}")
        if proj_dim is not None and (not isinstance(proj_dim, int) or proj_dim <= 0):
            raise ValueError(f"proj_dim must be a positive integer or None, got {proj_dim}")

        self.num_heads = int(num_heads)
        self.embed_dim = int(embed_dim)
        self.kernel_size = int(kernel_size)
        self.proj_dim = int(proj_dim) if proj_dim is not None else None

        # The channel width the outer product feeds into the conv: P^2.
        self._pair_dim = self.proj_dim if self.proj_dim is not None else self.embed_dim

        # Sub-layers created in __init__, built explicitly in build().
        self.token_proj: Optional[keras.layers.Dense] = (
            keras.layers.Dense(
                self.proj_dim, name="token_proj", dtype=self.dtype_policy
            )
            if self.proj_dim is not None
            else None
        )
        self.conv = keras.layers.Conv2D(
            filters=self.num_heads,
            kernel_size=self.kernel_size,
            padding="same",
            name="pair_conv",
            dtype=self.dtype_policy,
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Tuple[Optional[int], ...], ...]) -> None:
        """Explicitly build the (optional) token projection and the pair Conv2D.

        :param input_shape: 2-tuple ``((B, N, D), (B, N, N))``.
        :type input_shape: Tuple[Tuple[Optional[int], ...], ...]
        """
        if self.built:
            return

        x_shape = input_shape[0]
        n = x_shape[1]

        if self.token_proj is not None:
            self.token_proj.build(x_shape)

        # The conv sees the (B, N, N, P^2) image of pairwise outer products.
        conv_input_shape = (x_shape[0], n, n, self._pair_dim * self._pair_dim)
        self.conv.build(conv_input_shape)

        super().build(input_shape)

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Return the finite per-head weighted adjacency ``Ŵ`` of shape ``(B, H, N, N)``.

        :param inputs: 2-tuple ``(x, adjacency)`` — ``x`` the token state ``(B, N, D)`` and
            ``adjacency`` the binary ``(B, N, N)`` keep (``1`` = edge, ``0`` = non-edge).
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor]

        :return: ``Ŵ`` of shape ``(B, H, N, N)``, finite.
        :rtype: keras.KerasTensor
        """
        x, adjacency = inputs

        # The block hands us its compute-dtype x; the mask-dtype upcast happens at the
        # consumption site in _project.
        x = ops.cast(x, self.compute_dtype)

        x_p = self.token_proj(x) if self.token_proj is not None else x   # (B, N, P)

        # Per-pair outer product, then flatten the (p, q) axes into P^2 conv channels.
        shp = ops.shape(x_p)
        b, n = shp[0], shp[1]
        outer = ops.einsum("bnp,bmq->bnmpq", x_p, x_p)                   # (B, N, N, P, P)
        outer = ops.reshape(outer, (b, n, n, self._pair_dim * self._pair_dim))

        raw = self.conv(outer)                                           # (B, N, N, H)
        w_hat = ops.transpose(raw, (0, 3, 1, 2))                        # (B, H, N, N)

        # Zero Ŵ on non-edges (spec fidelity) -- the -inf non-edge suppression itself is
        # the existing keep-bias in _project, not this multiply.
        adjacency = ops.cast(adjacency, self.compute_dtype)
        w_hat = w_hat * ops.expand_dims(adjacency, axis=1)              # (B, H, N, N)

        return w_hat

    # -----------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[Tuple[Optional[int], ...], ...],
    ) -> Tuple[Optional[int], ...]:
        """Return the ``(B, H, N, N)`` output shape, valid on an unbuilt layer.

        :param input_shape: 2-tuple ``((B, N, D), (B, N, N))``.
        :type input_shape: Tuple[Tuple[Optional[int], ...], ...]

        :return: ``(B, num_heads, N, N)``.
        :rtype: Tuple[Optional[int], ...]
        """
        x_shape = input_shape[0]
        return (x_shape[0], self.num_heads, x_shape[1], x_shape[1])

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim,
            'kernel_size': self.kernel_size,
            'proj_dim': self.proj_dim,
        })
        return config

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.transformers.energy_transformer")
class EnergyTransformer(keras.layers.Layer):
    """Energy Transformer block: ``T`` steps of gradient descent on one scalar energy.

    Replaces the usual attention-then-FFN residual stream with an explicit optimization.
    There is no FFN and no value matrix. The block defines one scalar energy
    ``E(g) = E_ATT(g) + E_HN(g)`` and repeatedly steps the token state downhill on it
    (paper eq. 6, alg. 1; eq. 27 for the optional noise).

    .. code-block:: text

        for t in 1..T:
            g      = EnergyLayerNorm(x)                  # the "activation function"
            update = attn.update(g) + hopfield.update(g) # == -dE/dg
            x      = x + alpha * update                  # == x - alpha * dE/dg
            [+ sqrt(alpha) * noise_std * N(0, 1)   if training and noise_std > 0]

    Every ``update()`` in this feature returns ``-dE/dg``, the descent direction, so
    :meth:`call` adds it rather than subtracting. The gradient is taken with respect to
    ``g`` and applied to ``x`` — paper eq. 6, since ``EnergyLayerNorm`` is the map
    ``g = dL/dx`` of a Lagrangian whose Hessian is positive semidefinite for
    ``gamma > 0``, which is what makes ``dE/dt <= 0`` hold.

    Descent is guaranteed only for ``noise_std == 0``, ``gamma > 0`` and a sufficiently
    small ``step_size``. With ``noise_std > 0`` this is Langevin sampling (eq. 27) instead
    of descent, and the energy is expected to fluctuate upward.

    The three sub-layers are created in ``__init__`` and explicitly built in :meth:`build`,
    never lazily in :meth:`build` or :meth:`call`, so their weights survive a ``.keras``
    round trip.

    Note:
        Cost is mode-dependent; an op-count ratio is not a cost ratio. Figures are a
        median of 20 runs on an idle RTX 4070 at ``D=768, H=12, head_dim=64, K=3072,
        N=256, B=8, T=12``:

        .. code-block:: text

            return_energy=True, relative to the default False
                forward        eager 1.68x  | graph 1.44x     | jit 0.9-1.1x
                fwd+bwd step   eager 1.44x  | graph 1.28x     | jit 1.02x

            ET vs one vanilla TransformerLayer at matched width
                forward        eager 7.5x   | graph 8.1-8.7x  | jit 6.0-7.6x
                fwd+bwd step   eager 5.5x   | graph 8.8x      | jit 10.7x
                params         3.54M  vs  7.09M   (ET has half the parameters)

        ``return_energy=True`` makes ``2T + 1`` internal projection calls instead of
        ``T``, but under ``jit_compile=True`` XLA eliminates the shared computation
        ``energy()`` and ``update()`` do at each step, so the trace is close to free on
        the compiled path even though it costs ~1.7x in eager mode. A step is cheaper
        than a full transformer block because the block has no value matrix and no
        separate FFN projections, so ``T=12`` steps land at roughly 6-11x one block
        rather than 12x. Backward memory is linear in ``T``: the loop is unrolled and
        every step's activations are held for the backward pass.

        The per-step keep-mask rebuild in ``EnergyAttention._build_keep_mask`` looks
        costly in eager profiling (~27% of the forward pass) but is not worth hoisting:
        the compiled graph either constant-folds it away or common-subexpression-
        eliminates it down to one build, so the measured end-to-end hoist delta is
        roughly zero. See ``plans/plan_2026-07-14_e5955791/decisions.md`` D-005.

    :param embed_dim: Token embedding dimension ``D``.
    :type embed_dim: int
    :param num_heads: Number of attention heads ``H``.
    :type num_heads: int
    :param head_dim: Per-head key/query dimension ``Y``.
    :type head_dim: int
    :param hopfield_dim: Number of Hopfield memories ``K``.
    :type hopfield_dim: int
    :param num_steps: Number of descent steps ``T``. Defaults to ``12``.
    :type num_steps: int
    :param step_size: Descent step ``alpha``. Defaults to ``0.1``.
    :type step_size: float
    :param beta: Attention inverse temperature. ``None`` resolves to ``1/sqrt(head_dim)``
        inside ``EnergyAttention`` itself.
    :type beta: Optional[float]
    :param attn_self: If ``False`` (default, the paper's ET-Full), a token does not attend
        to itself.
    :type attn_self: bool
    :param hopfield_activation: ``'relu'`` (default) or ``'softmax'``.
    :type hopfield_activation: str
    :param noise_std: Std-dev of the eq. 27 noise, injected as
        ``sqrt(step_size) * noise_std * N(0, 1)`` only when ``training`` is truthy.
        ``0.0`` (default) disables it.
    :type noise_std: float
    :param return_energy: If ``True``, :meth:`call` returns ``(x, energies)`` with
        ``energies`` of shape ``(B, num_steps + 1)``, per sample rather than batch-reduced
        so a per-sample descent violation stays visible. The energy trace is ``float32``
        even under ``mixed_float16``, since its magnitude overflows float16 at a realistic
        sequence length; a head consuming it under a mixed-precision policy must itself be
        built ``dtype='float32'`` (see :meth:`compute_output_spec`).
    :type return_energy: bool
    :param hopfield_beta: Inverse temperature of the ``'softmax'`` Hopfield branch. See the
        D-007 anchor in ``__init__``.
    :type hopfield_beta: float
    :param norm_epsilon: ``epsilon`` of the inner ``EnergyLayerNorm``.
    :type norm_epsilon: float
    :param seed: Seed for the ``noise_std`` RNG. See the D-007 anchor in ``__init__``.
    :type seed: Optional[int]

    :raises ValueError: If ``embed_dim <= 0``, ``num_heads <= 0``, ``head_dim <= 0``,
        ``hopfield_dim <= 0``, ``num_steps < 1``, ``step_size <= 0`` or ``noise_std < 0``.

    Input shape:
        3D tensor ``(batch, num_tokens, embed_dim)``.

    Output shape:
        ``(batch, num_tokens, embed_dim)``; or, with ``return_energy=True``, the pair
        ``((batch, num_tokens, embed_dim), (batch, num_steps + 1))``.

    :ivar norm: The inner ``EnergyLayerNorm`` (scalar gamma, vector delta).
    :vartype norm: EnergyLayerNorm
    :ivar attention: The inner ``EnergyAttention`` — token mixing, the only masked module.
    :vartype attention: EnergyAttention
    :ivar hopfield: The inner :class:`HopfieldNetwork`, applied per token.
    :vartype hopfield: HopfieldNetwork

    Example:
        >>> block = EnergyTransformer(
        ...     embed_dim=64, num_heads=4, head_dim=16, hopfield_dim=128,
        ...     num_steps=8, step_size=0.05, return_energy=True,
        ... )
        >>> x = keras.random.normal((2, 16, 64))
        >>> y, energies = block(x, training=False)
        >>> y.shape, energies.shape
        ((2, 16, 64), (2, 9))

    References:
        - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253, eq. (6),
          alg. 1, eq. (27).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        hopfield_dim: int,
        num_steps: int = 12,
        step_size: float = 0.1,
        beta: Optional[float] = None,
        attn_self: bool = False,
        hopfield_activation: str = 'relu',
        noise_std: float = 0.0,
        return_energy: bool = False,
        # DECISION plan_2026-07-13_57c9833e/D-007: hopfield_beta is a second, independent
        # temperature (not beta, which is meaningless over the hopfield_dim memory axis).
        # norm_epsilon and seed are both forced, not optional. See decisions.md.
        hopfield_beta: float = 1.0,
        norm_epsilon: float = 1e-5,
        seed: Optional[int] = None,
        # DECISION plan-2026-07-15T053724-78001af1/D-002: weighted-adjacency flag, additive
        # and default-off. A flag-on model has a different, non-weight-compatible weight
        # set, so the projector is created only when the flag is on. See decisions.md.
        use_weighted_adjacency: bool = False,
        adjacency_kernel_size: int = 1,
        adjacency_proj_dim: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # ----- validation -----
        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive integer, got {embed_dim}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be a positive integer, got {head_dim}")
        if not isinstance(hopfield_dim, int) or hopfield_dim <= 0:
            raise ValueError(
                f"hopfield_dim must be a positive integer, got {hopfield_dim}"
            )
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError(f"num_steps must be an integer >= 1, got {num_steps}")
        if not isinstance(step_size, (int, float)) or step_size <= 0:
            raise ValueError(f"step_size must be a positive number, got {step_size}")
        if not isinstance(noise_std, (int, float)) or noise_std < 0:
            raise ValueError(f"noise_std must be a non-negative number, got {noise_std}")
        if not isinstance(adjacency_kernel_size, int) or adjacency_kernel_size <= 0:
            raise ValueError(
                f"adjacency_kernel_size must be a positive integer, got "
                f"{adjacency_kernel_size}"
            )
        if adjacency_proj_dim is not None and (
            not isinstance(adjacency_proj_dim, int) or adjacency_proj_dim <= 0
        ):
            raise ValueError(
                f"adjacency_proj_dim must be a positive integer or None, got "
                f"{adjacency_proj_dim}"
            )

        # `beta`, `hopfield_activation` and `hopfield_beta` are validated by the sub-layers
        # they belong to (EnergyAttention / HopfieldNetwork) — not re-validated here, so
        # there is exactly ONE source of truth per rule.

        # ----- store ALL configuration -----
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.hopfield_dim = int(hopfield_dim)
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.beta = beta
        self.attn_self = bool(attn_self)
        self.hopfield_activation = str(hopfield_activation)
        self.noise_std = float(noise_std)
        self.return_energy = bool(return_energy)
        self.hopfield_beta = float(hopfield_beta)
        self.norm_epsilon = float(norm_epsilon)
        self.seed = seed
        self.use_weighted_adjacency = bool(use_weighted_adjacency)
        self.adjacency_kernel_size = int(adjacency_kernel_size)
        self.adjacency_proj_dim = (
            int(adjacency_proj_dim) if adjacency_proj_dim is not None else None
        )

        # sqrt(alpha), the eq. 27 noise scaling. Precomputed: `step_size` is a Python float.
        self._sqrt_step_size = math.sqrt(self.step_size)

        # Sub-layers are created here, not lazily in build()/call(), so their weights are
        # tracked at serialization time and survive a .keras round trip.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-001: pass self.dtype_policy (the policy
        # object), not self.dtype -- self.dtype is the variable dtype only and breaks
        # mixed_float16 sub-layers under a global mixed policy. See decisions.md.
        self.norm = EnergyLayerNorm(
            epsilon=self.norm_epsilon,
            name="energy_layer_norm",
            dtype=self.dtype_policy,
        )
        self.attention = EnergyAttention(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            beta=self.beta,           # None -> EnergyAttention resolves 1/sqrt(head_dim)
            attn_self=self.attn_self,
            name="energy_attention",
            dtype=self.dtype_policy,
        )
        self.hopfield = HopfieldNetwork(
            dim=self.embed_dim,
            hopfield_dim=self.hopfield_dim,
            activation=self.hopfield_activation,
            hopfield_beta=self.hopfield_beta,
            name="hopfield_network",
            dtype=self.dtype_policy,
        )

        # DECISION plan-2026-07-15T053724-78001af1/D-002: weighted-adjacency projector
        # created only when the flag is on, keeping every default checkpoint free of dead
        # weights. Its own weights train via the outer fit() backward pass. See decisions.md.
        self.adjacency_projector: Optional[WeightedAdjacencyProjector] = (
            WeightedAdjacencyProjector(
                num_heads=self.num_heads,
                embed_dim=self.embed_dim,
                kernel_size=self.adjacency_kernel_size,
                proj_dim=self.adjacency_proj_dim,
                name="weighted_adjacency_projector",
                dtype=self.dtype_policy,
            )
            if self.use_weighted_adjacency
            else None
        )

        # Only needed when noise is on, but created unconditionally so that `seed` round-
        # trips and the layer's structure does not depend on a numeric value.
        self.seed_generator = keras.random.SeedGenerator(seed=self.seed)

        self.supports_masking = True

        logger.debug(
            f"Initialized EnergyTransformer with embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"hopfield_dim={self.hopfield_dim}, num_steps={self.num_steps}, "
            f"step_size={self.step_size}, attn_self={self.attn_self}, "
            f"hopfield_activation={self.hopfield_activation}, "
            f"noise_std={self.noise_std}, return_energy={self.return_energy}"
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Explicitly build every sub-layer, then call ``super().build()`` last.

        All three sub-layers see the same shape: the block is shape-preserving at every
        stage (``g``, both updates, and ``x`` are all ``(B, N, D)``).

        :param input_shape: Input shape ``(batch, num_tokens, embed_dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the last axis of ``input_shape`` is not ``embed_dim``.
        """
        if self.built:
            return

        feature_dim = input_shape[-1]
        if feature_dim is not None and int(feature_dim) != self.embed_dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} does not match "
                f"embed_dim={self.embed_dim}"
            )

        # Explicit sub-layer builds. NOT optional: without them the sub-layers are built
        # lazily on the first `call()`, which is exactly the path that loses weights on a
        # `.keras` round-trip.
        self.norm.build(input_shape)
        self.attention.build(input_shape)
        self.hopfield.build(input_shape)

        if self.adjacency_projector is not None:
            # The projector consumes (x, adjacency): the token state (B, N, D) and a binary
            # (B, N, N) adjacency. Built explicitly here (golden pattern) so its trainable
            # Conv2D/Dense survive a `.keras` round-trip.
            num_tokens = input_shape[1]
            adjacency_shape = (input_shape[0], num_tokens, num_tokens)
            self.adjacency_projector.build((tuple(input_shape), adjacency_shape))

        super().build(input_shape)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def _hopfield_token_mask(
        self,
        attention_mask: Optional[keras.KerasTensor],
        mask: Optional[keras.KerasTensor],
    ) -> Optional[keras.KerasTensor]:
        """The per-token ``(B, N)`` keep the HopfieldNetwork must see, from both masks.

        # DECISION plan_2026-07-13_ca4f71a2/D-006: a rank-2 attention_mask and the Keras
        # mask are the same object (a per-token validity mask); forwarding only one of
        # them drifted the Hopfield energy +27.3% on the attention_mask-only path. See decisions.md.

        A rank-3 ``(B, N, N)`` or rank-4 ``(B, H, N, N)`` ``attention_mask`` is a different
        object — a pair-level key-times-query mask with no per-token reading — and does not
        reach the Hopfield energy. Masking the energy without masking its update with the
        same keep would make ``update() != -dE/dg``, since ``E_HN`` is a sum of uncoupled
        per-token terms and its gradient is exactly ``keep_n * de_n/dg_n``.

        :param attention_mask: Optional keep mask of rank 2 / 3 / 4. Only rank 2 has a
            per-token reading.
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` validity mask.
        :type mask: Optional[keras.KerasTensor]

        :return: A ``(B, N)`` 0/1 keep tensor, the logical AND of whichever of the two
            carry a per-token reading, or ``None`` when neither does (the Hopfield energy
            then sums over every token, which is correct: nothing declared any invalid).
        :rtype: Optional[keras.KerasTensor]
        """
        keep_dtype = _mask_dtype(self.compute_dtype)
        token_keep: Optional[keras.KerasTensor] = None

        if attention_mask is not None and len(attention_mask.shape) == 2:
            token_keep = ops.cast(attention_mask, keep_dtype)             # (B, N)

        if mask is not None:
            keras_keep = _token_keep(mask, keep_dtype)                    # (B, N), rank-checked
            # Multiplication is the logical AND -- neither mask resurrects a token the
            # other hid.
            token_keep = (
                keras_keep if token_keep is None else token_keep * keras_keep
            )

        return token_keep

    # -----------------------------------------------------------------

    def _binary_adjacency(
        self,
        attention_mask: Optional[keras.KerasTensor],
    ) -> Optional[keras.KerasTensor]:
        """The binary ``(B, N, N)`` adjacency ``A'`` the projector needs, or ``None``.

        The graph model passes its adjacency as the pair-level ``attention_mask`` (D-006:
        rank-3 ``(B, N, N)`` is a key x query keep). Only a pair-level mask carries an
        edge/non-edge reading:

        * rank 3 ``(B, N, N)`` — already ``A'``, returned as-is.
        * rank 4 ``(B, H, N, N)`` — reduced to ``(B, N, N)`` by a keep-in-ANY-head
          (``max`` over the head axis): an edge exists if the pair is kept for any head.
        * rank 2 ``(B, N)`` / ``None`` — a per-token validity mask carries NO pair-level
          edge structure, so there is no ``A'`` to build ``Ŵ`` from -> ``None`` (the
          weighted path is inert, exactly as with the flag off).

        :param attention_mask: Optional KEEP mask of rank 2 / 3 / 4.
        :type attention_mask: Optional[keras.KerasTensor]

        :return: ``(B, N, N)`` binary adjacency, or ``None``.
        :rtype: Optional[keras.KerasTensor]
        """
        if attention_mask is None:
            return None
        rank = len(attention_mask.shape)
        if rank == 3:
            return attention_mask
        if rank == 4:
            return ops.max(attention_mask, axis=1)
        return None

    # -----------------------------------------------------------------

    def _weighted_adjacency(
        self,
        x: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor],
    ) -> Optional[keras.KerasTensor]:
        """Compute the per-block-constant ``Ŵ`` from the block input ``x``.

        # DECISION plan-2026-07-15T053724-78001af1/D-002: called once before the T-step
        # loop, never from the evolving g -- Ŵ must stay constant for EnergyAttention's
        # closed-form gradient (dŴ/dg == 0) to hold. See decisions.md.

        :param x: The block input token state ``(B, N, D)`` (not the evolving state).
        :type x: keras.KerasTensor
        :param attention_mask: Optional KEEP mask — a rank-3/4 one carries the adjacency.
        :type attention_mask: Optional[keras.KerasTensor]

        :return: ``Ŵ`` of shape ``(B, H, N, N)``, or ``None`` when the flag is off or no
            pair-level adjacency is present.
        :rtype: Optional[keras.KerasTensor]
        """
        if not self.use_weighted_adjacency:
            return None
        adjacency = self._binary_adjacency(attention_mask)
        if adjacency is None:
            return None
        return self.adjacency_projector((x, adjacency))

    # -----------------------------------------------------------------

    def energy(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        mask: Optional[keras.KerasTensor] = None,
        adjacency_weight: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """The block's total scalar energy ``E(g) = E_ATT(g) + E_HN(g)``, shape ``(B,)``.

        This is the Lyapunov function the block descends, the sum of exactly two terms.
        ``EnergyLayerNorm``'s Lagrangian is not a third term here — it is the potential
        whose Hessian makes the descent positive semidefinite, not a quantity in ``E``
        itself. See decisions.md ``plan_2026-07-13_57c9833e/D-005``.

        :param g: Normalized token state ``(B, N, D)``, i.e. ``self.norm(x)``, not ``x``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional keep mask. A rank-2 ``(B, N)`` one is a per-token
            validity mask and reaches both terms (equivalent to ``mask``). A rank-3/rank-4
            one is a key x query pair mask with no per-token reading, so it reaches
            ``EnergyAttention`` only — the Hopfield net has no pairs.
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` boolean validity mask.
            ANDed with ``attention_mask`` inside ``EnergyAttention`` (decisions.md D-003), and
            ANDed again into the ``HopfieldNetwork`` token keep, whose energy is a SUM over
            tokens and so must exclude the PAD tokens (decisions.md D-005, D-006).
        :type mask: Optional[keras.KerasTensor]
        :param adjacency_weight: Optional finite per-head weighted adjacency ``Ŵ``
            (paper eq. 25), forwarded unchanged to ``EnergyAttention.energy``. Must be the
            same tensor the matching ``self.attention.update`` receives in :meth:`call`,
            hoisted once by :meth:`_weighted_adjacency` rather than recomputed here.
        :type adjacency_weight: Optional[keras.KerasTensor]

        :return: Energy of shape ``(B,)``, in the reduce dtype (``>= float32``), never the
            compute dtype.
        :rtype: keras.KerasTensor
        """
        # DECISION plan_2026-07-13_ca4f71a2/D-005: both terms return in the reduce dtype,
        # never cast down -- an O(-8e4) energy is -inf under mixed_float16. See decisions.md.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-002: mask must reach self.attention here and
        # at the matching update() call in call() -- a merge landing in only one path breaks
        # update() == -dE/dg. See decisions.md.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-005: mask also reaches self.hopfield, since
        # its energy is a reduction over tokens (unlike its update, which needs no mask). See decisions.md.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-006: the Hopfield mask comes from BOTH masks
        # via _hopfield_token_mask, not from mask alone -- the rank-2 attention_mask path
        # drifted +27.3% otherwise. See decisions.md.
        hopfield_mask = self._hopfield_token_mask(attention_mask, mask)
        return (
            self.attention.energy(
                g, attention_mask=attention_mask, mask=mask,
                adjacency_weight=adjacency_weight,
            )
            + self.hopfield.energy(g, mask=hopfield_mask)
        )

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Run ``num_steps`` of energy descent on the token state (paper eq. 6, alg. 1).

        ``supports_masking = True`` and this signature declares ``mask``, which is what
        makes Keras inject a propagated mask (for example from an upstream
        ``Embedding(mask_zero=True)``); declaring it is necessary but not sufficient, since
        the block must also forward it to ``self.attention``. When both ``mask`` and
        ``attention_mask`` are supplied the rule is a logical AND — a token is attended
        only if valid under both, and neither mask can un-mask what the other hid.

        A rank-2 ``attention_mask`` is the same object as a Keras ``mask``: both are
        per-token validity masks, and using either drops the token from the attention keep
        mask and from the Hopfield energy's token sum, so its row passes through unchanged
        at ``noise_std=0``. A rank-3 ``(B, N, N)`` or rank-4 ``(B, H, N, N)``
        ``attention_mask`` is a different object, a pair-level key-times-query keep mask
        with no per-token reading, and it does not mask the Hopfield energy.

        :param inputs: Token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional keep mask (``1`` = attend). Rank-2 ``(B, N)`` is a
            per-token validity mask, equivalent to ``mask``. Rank-3 ``(B, N, N)`` / rank-4
            ``(B, H, N, N)`` is a pair-level key x query mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: When truthy and ``noise_std > 0``, the eq. 27 noise is injected.
            Accepts a Python ``bool``/``None`` or a symbolic scalar bool tensor from a
            traced graph function; a tensor is gated without ever calling ``bool()`` on it.
        :type training: Optional[bool]
        :param mask: Keras-propagated rank-2 ``(B, N)`` boolean per-token validity mask.
            Normally injected by Keras, not passed by hand.
        :type mask: Optional[keras.KerasTensor]

        :return: ``x`` of shape ``(B, N, D)``; or ``(x, energies)`` with ``energies`` of
            shape ``(B, num_steps + 1)`` when ``return_energy=True``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        x = inputs
        energies: List[keras.KerasTensor] = []

        # DECISION plan_2026-07-14_e5955791/D-003: noise gating is split into two stages
        # because `self.noise_std > 0.0 and training` calls Tensor.__bool__() the moment
        # training is a traced symbolic tensor, raising OperatorNotAllowedInGraphError. See decisions.md.
        add_noise = False                                    # unconditional (Python) noise
        noise_gate: Optional[keras.KerasTensor] = None       # tensor-gated noise
        if self.noise_std > 0.0:
            if training is None or isinstance(training, bool):
                add_noise = bool(training)
            else:
                noise_gate = ops.cast(training, "bool")

        # The per-token keep the Hopfield module sees, hoisted out of the loop since it
        # does not depend on the descent step. Must match what self.energy() uses.
        hopfield_mask = self._hopfield_token_mask(attention_mask, mask)

        # DECISION plan-2026-07-15T053724-78001af1/D-002: Ŵ is computed once from the
        # block input, held constant across all T steps, and must reach both energy() and
        # update() as the same tensor. See decisions.md.
        adjacency_weight = self._weighted_adjacency(inputs, attention_mask)

        for _ in range(self.num_steps):
            g = self.norm(x)

            if self.return_energy:
                energies.append(
                    self.energy(
                        g, attention_mask=attention_mask, mask=mask,
                        adjacency_weight=adjacency_weight,
                    )
                )

            # update == -dE/dg: both sub-layers return the descent direction, never the
            # gradient. See the class docstring.
            #
            # DECISION plan_2026-07-13_ca4f71a2/D-005: mask reaches both sub-layers' updates
            # here, matching what self.energy() passed their energy() above, or
            # update() != -dE/dg. See decisions.md.
            update = (
                self.attention.update(
                    g, attention_mask=attention_mask, mask=mask,
                    adjacency_weight=adjacency_weight,
                )
                + self.hopfield.update(g, mask=hopfield_mask)
            )

            # Adding alpha * update is x = x - alpha * dE/dg -- paper eq. 6, gradient taken
            # w.r.t. g and applied to x. See the class docstring.
            x = x + self.step_size * update

            if add_noise or noise_gate is not None:
                # eq. 27 (Langevin). ops.shape(x), never a Python int, so this works under
                # a symbolic/variable batch size.
                noisy = x + self._sqrt_step_size * self.noise_std * keras.random.normal(
                    shape=ops.shape(x),
                    dtype=self.compute_dtype,
                    seed=self.seed_generator,
                )
                # DECISION plan_2026-07-14_e5955791/D-003: the tensor branch draws noise
                # and discards it when the gate is False -- the accepted price of a
                # graph-safe training flag. See decisions.md.
                x = noisy if noise_gate is None else ops.where(noise_gate, noisy, x)

        if self.return_energy:
            # The (T+1)-th reading: the energy after the last step, so the caller sees the
            # effect of the final update too.
            g = self.norm(x)
            energies.append(
                self.energy(
                    g, attention_mask=attention_mask, mask=mask,
                    adjacency_weight=adjacency_weight,
                )
            )
            return x, ops.stack(energies, axis=-1)            # (B, N, D), (B, T + 1)

        return x

    # -----------------------------------------------------------------

    def compute_mask(
        self,
        inputs: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> Union[
        Optional[keras.KerasTensor],
        List[Optional[keras.KerasTensor]],
    ]:
        """Propagate the token mask, but never onto the energy tensor.

        # DECISION plan_2026-07-13_ca4f71a2/D-002: with return_energy=True this layer
        # emits (x, energies); the (B, N) token mask must not attach to the (B, T+1)
        # energy tensor, which has no token axis. See decisions.md.

        :param inputs: Token state ``(B, N, D)`` (unused; the layer is shape-preserving).
        :type inputs: keras.KerasTensor
        :param mask: Incoming ``(B, N)`` token mask, or ``None``.
        :type mask: Optional[keras.KerasTensor]

        :return: ``mask``; or ``[mask, None]`` when ``return_energy=True`` (one entry per
            output tensor).
        :rtype: Union[Optional[keras.KerasTensor], List[Optional[keras.KerasTensor]]]
        """
        if self.return_energy:
            return [mask, None]
        return mask

    # -----------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]:
        """Return the output shape, valid on an unbuilt layer.

        Uses only the passed shape and stored config ints (``num_steps``), never a weight
        shape, so it works before ``build()``.

        :param input_shape: Input shape ``(batch, num_tokens, embed_dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: ``input_shape``; or, with ``return_energy=True``, the two-tuple
            ``(input_shape, (batch, num_steps + 1))``.
        :rtype: Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]
        """
        input_shape = tuple(input_shape)
        if self.return_energy:
            batch = input_shape[0]
            return input_shape, (batch, self.num_steps + 1)
        return input_shape

    # -----------------------------------------------------------------

    def compute_output_spec(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Symbolic output spec: the energy is ``>= float32``, the state is compute dtype.

        # DECISION plan_2026-07-13_ca4f71a2/D-006: exists to tell the truth about the
        # energy's dtype -- Keras' default compute_output_spec stamps every output with
        # self.compute_dtype, which under mixed_float16 claims float16 for a tensor that
        # actually runs float32. See decisions.md.

        # DECISION plan_2026-07-13_ca4f71a2/D-007: a downstream layer autocasts inputs to
        # its own compute dtype regardless of the upstream symbolic dtype, so a head
        # consuming the energy trace must itself be built dtype='float32'. See decisions.md.

        :param inputs: Symbolic token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Unused (does not affect shape or dtype).
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Unused (does not affect shape or dtype).
        :type training: Optional[bool]
        :param mask: Unused (does not affect shape or dtype).
        :type mask: Optional[keras.KerasTensor]

        :return: ``KerasTensor(B, N, D)`` in the compute dtype; or that plus
            ``KerasTensor(B, num_steps + 1)`` in the reduce dtype (``>= float32``) when
            ``return_energy=True``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        shape = self.compute_output_shape(inputs.shape)   # the one shape authority

        if not self.return_energy:
            return keras.KerasTensor(shape, dtype=self.compute_dtype)

        state_shape, energy_shape = shape
        return (
            keras.KerasTensor(state_shape, dtype=self.compute_dtype),
            keras.KerasTensor(energy_shape, dtype=_mask_dtype(self.compute_dtype)),
        )

    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'hopfield_dim': self.hopfield_dim,
            'num_steps': self.num_steps,
            'step_size': self.step_size,
            'beta': self.beta,
            'attn_self': self.attn_self,
            'hopfield_activation': self.hopfield_activation,
            'noise_std': self.noise_std,
            'return_energy': self.return_energy,
            'hopfield_beta': self.hopfield_beta,
            'norm_epsilon': self.norm_epsilon,
            'seed': self.seed,
            'use_weighted_adjacency': self.use_weighted_adjacency,
            'adjacency_kernel_size': self.adjacency_kernel_size,
            'adjacency_proj_dim': self.adjacency_proj_dim,
        })
        return config

# ---------------------------------------------------------------------
