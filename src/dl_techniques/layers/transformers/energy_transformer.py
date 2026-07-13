"""
Energy Transformer (ET) — Hopfield associative memory module (and, soon, the ET block).

Implements the modules of the Energy Transformer, Hoover, Liang, Pham, Panda, Strobelt,
Zaki, Chau, Krotov, "Energy Transformer", NeurIPS 2023 (https://arxiv.org/abs/2302.07253).

This module currently holds:

- :class:`HopfieldNetwork` — the ET Hopfield / associative-memory module (eq. 5, 9). A
  single tied ``(K, D)`` memory matrix, applied strictly per token.
- ``EnergyTransformer`` — the ``T``-step energy-descent block (eq. 6, alg. 1). *Lands in a
  follow-up step; it composes* :class:`HopfieldNetwork` *with* ``EnergyLayerNorm``
  (``layers/norms/energy_layer_norm.py``) *and* ``EnergyAttention``
  (``layers/attention/energy_attention.py``).

Architecture (the ET block; the Hopfield module is the right-hand branch)::

    x  (B, N, D)
    |
    +--> for t in 1..T:
    |        g = EnergyLayerNorm(x)                     # (B, N, D)
    |
    |        E(g) = E_ATT(g)          +   E_HN(g)       # one scalar per sample
    |                  |                     |
    |          EnergyAttention          HopfieldNetwork
    |        (token MIXING; eq. 3-4)   (per-token; eq. 5)
    |                  |                     |
    |              -dE_ATT/dg           -dE_HN/dg
    |                  \\_______   _______/
    |                          \\ /
    |                        update  == -dE/dg   (the DESCENT direction)
    |                          |
    |        x = x + alpha * update                     # eq. 6: tau dx/dt = -dE/dg
    |
    v
    x  (B, N, D)

**Duck-typed convention (NOT an ABC).** :class:`HopfieldNetwork` and ``EnergyAttention``
both expose the same trio — ``energy(g, ...) -> (B,)`` / ``update(g, ...) -> (B, N, D)`` /
``call(...) -> update(...)`` — and the block consumes exactly that. Two implementors and
one consumer earn the *convention*, not an inheritance hierarchy.

**No autodiff in src/.** Every gradient here is hand-coded in closed form with
``keras.ops`` only (``keras.ops.grad`` does not exist in keras 3.8, and a backend-specific
autodiff tape is forbidden in ``src/`` — decisions.md D-001). Correctness rests entirely
on the autodiff oracle test ``test_gradient_oracle`` in
``tests/test_layers/test_transformers/test_energy_transformer.py``.

References:
    - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253, eq. (5), (9).
    - Ramsauer et al., "Hopfield Networks is All You Need", ICLR 2021 (the ``'softmax'``
      / modern-Hopfield energy). NOTE: this is NOT ``layers/attention/hopfield_attention.py``,
      which implements Ramsauer's *attention* (separate Q/K/V Dense projections) and is a
      completely different mechanism.
"""

import keras
from keras import ops, initializers
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# The ONLY supported Hopfield activations. See the D-005 anchor in `__init__` for why
# `'power'` is deliberately absent.
_VALID_ACTIVATIONS = ('relu', 'softmax')


@keras.saving.register_keras_serializable()
class HopfieldNetwork(keras.layers.Layer):
    """Energy Transformer Hopfield / associative-memory module (tied weights, bias-free).

    **Intent**: expose a scalar per-token energy ``E_HN(g)`` together with its exact
    closed-form negative gradient, so that an ``EnergyTransformer`` block can perform
    *provable gradient descent* on ``E_ATT + E_HN``. This is the paper's analog of the
    feed-forward MLP, but it is **not an MLP**.

    # DECISION plan_2026-07-13_57c9833e/D-002
    This layer is deliberately **NOT registered in the FFN factory** (``ffn/factory.py``),
    and must not be "helpfully" added to ``create_ffn_layer``. It is not FFN-shaped: there
    is ONE tied ``(K, D)`` matrix ``xi`` used in BOTH directions (up-project, then
    down-project by its transpose) — not independent up/down projections — there is **no
    bias**, and its ``activation`` is not a pointwise nonlinearity applied *between* two
    layers: it is ``r = G'``, the DERIVATIVE of the energy's integrand ``G``. Registering
    it as an FFN type would misrepresent an associative memory as an MLP and would invite
    callers to swap it for a real FFN, silently destroying the descent guarantee (an MLP
    is not the gradient of any energy this block reports). See decisions.md D-002.

    **Mathematics** (notation: ``B``=batch, ``N``=tokens, ``D``=``dim``, ``K``=``hopfield_dim``):

    .. code-block:: text

        h_{n k} = sum_d xi_{k d} g_{n d}                      # (B, N, K), per token
        E_HN    = - sum_n sum_k G(h_{n k})       where  G' = r
        -dE_HN/dg_{n d} = sum_k xi_{k d} r(h_{n k})           # (B, N, D)

    Each activation carries **both** an energy and its matching gradient factor ``r``.
    They are a consistent pair by construction; pairing one activation's energy with the
    other's gradient is a SILENT correctness break (the layer still runs, still trains,
    still emits finite outputs — and the energy simply stops descending):

    .. code-block:: text

        activation   r(h)                       E_HN
        ----------   ------------------------   -----------------------------------------
        'relu'       relu(h)                    -0.5 * sum_{n,k} relu(h_{n k})^2
        'softmax'    softmax_k(beta * h)_{n k}  -(1/beta) * sum_n logsumexp_k(beta*h_{n k})

    Check: ``d/dh [-0.5 relu(h)^2] = -relu(h)``, so ``-dE/dh = relu(h) = r``.
    Check: ``d/dh [-(1/b) logsumexp_k(b h)] = -softmax_k(b h)``, so ``-dE/dh = r``.

    In the ``'softmax'`` case ``G`` is **not separable per-**``k``: its energy is a
    ``logsumexp`` over the **MEMORY** axis ``k``, not a sum of per-memory terms. The
    softmax is likewise over the MEMORY axis ``k``, **never** over the token axis ``n``.

    **NO TOKEN MIXING.** Every token is processed independently (``h`` depends only on
    ``g[:, n, :]``). Consequently this layer takes **no** ``attention_mask``: masking is
    meaningless here, since a token cannot influence any other token. Only
    ``EnergyAttention`` mixes tokens, and only it accepts a mask.

    **SIGN DISCIPLINE.** :meth:`update` returns ``-dE/dg`` — the **descent direction**,
    NOT the gradient. A consumer therefore *adds* ``step_size * update``. Do not "fix"
    this sign: flipping it silently turns the block's dynamics into energy *ascent*, which
    still runs and still produces finite outputs.

    **Duck-typed convention (NOT an ABC).** This layer and ``EnergyAttention`` both expose
    the trio ``energy(g) -> (B,)`` / ``update(g) -> (B, N, D)`` / ``call(...) -> update(...)``.
    Two implementors and one consumer earn the *convention*, not a base class or a
    ``Protocol``.

    :param dim: Token embedding dimension ``D``.
    :type dim: int
    :param hopfield_dim: Number of stored memories ``K`` (the rows of ``xi``).
    :type hopfield_dim: int
    :param activation: Energy/gradient pair to use — ``'relu'`` (default; the paper's
        config for BOTH its image and graph headline models) or ``'softmax'`` (the modern
        Hopfield energy).
    :type activation: str
    :param hopfield_beta: Inverse temperature of the ``'softmax'`` branch. **Read only by
        that branch**; ignored by ``'relu'``. Defaults to ``1.0``.
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

    Attributes:
        xi: The tied, bias-free memory matrix, shape ``(hopfield_dim, dim)`` == ``(K, D)``.

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

        # DECISION plan_2026-07-13_57c9833e/D-005
        # `'power'` (r = relu(h)^(p-1), G = relu(h)^p / p) is deliberately NOT implemented
        # and must NOT be added speculatively. It has ZERO call sites, and BOTH of the
        # paper's headline configurations (image ET-Full, graph ET) use `'relu'`. It is a
        # ~4-line additive extension (one more `r` branch, one more `E` branch, one more
        # `power` ctor arg) if a caller ever genuinely needs it — at which point it also
        # earns a gradient-oracle parametrization. Adding an unexercised third
        # energy/gradient PAIR now would ship an untested descent guarantee.
        # See decisions.md D-005.
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

        # ONE matrix, used in BOTH directions (up-project via `xi`, down-project via its
        # transpose). NO BIAS, by construction: the paper's energy E_HN is defined without
        # one, and a bias would not be expressible in the closed-form gradient below.
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

    def energy(self, g: keras.KerasTensor) -> keras.KerasTensor:
        """Scalar Hopfield energy ``E_HN`` per batch element (paper eq. 5).

        .. code-block:: text

            h_{n k} = sum_d xi_{k d} g_{n d}

            'relu'    : E_HN = -0.5 * sum_{n,k} relu(h_{n k})^2
            'softmax' : E_HN = -(1/beta) * sum_n logsumexp_k( beta * h_{n k} )

        The ``'softmax'`` ``logsumexp`` is over the **MEMORY** axis ``k`` (axis ``-1``),
        NOT the token axis ``n``.

        **These formulas are the SPEC.** :meth:`update` must match *these*; never edit
        this method to make the gradient oracle pass (plan STOP-IF 1).

        :param g: Token state ``(B, N, D)``, typically the output of ``EnergyLayerNorm``.
        :type g: keras.KerasTensor

        :return: Energy of shape ``(B,)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        h = ops.einsum('kd,bnd->bnk', self.xi, g)        # (B, N, K)

        if self.activation == 'relu':
            r = ops.relu(h)
            return -0.5 * ops.sum(ops.square(r), axis=(1, 2))            # (B,)

        # 'softmax' — G is NOT separable per memory k: logsumexp over the MEMORY axis.
        lse = ops.logsumexp(self.hopfield_beta * h, axis=-1)             # (B, N)
        return -(1.0 / self.hopfield_beta) * ops.sum(lse, axis=1)        # (B,)

    # -----------------------------------------------------------------

    def update(self, g: keras.KerasTensor) -> keras.KerasTensor:
        """Return ``-dE_HN/dg`` — the DESCENT DIRECTION, **not** the gradient.

        **SIGN DISCIPLINE**: this is the *negative* gradient. The consumer *adds*
        ``step_size * update`` to the token state. A reader who assumes this returns
        ``+dE/dg`` and "fixes" the sign at the call site silently inverts the dynamics into
        energy ASCENT — which still runs, still trains, and still produces finite outputs.

        .. code-block:: text

            -dE_HN/dg_{n d} = sum_k xi_{k d} r(h_{n k})

            'relu'    : r(h) = relu(h)                    (pairs with -0.5*sum relu(h)^2)
            'softmax' : r(h) = softmax_k(beta * h)        (pairs with -(1/beta)*logsumexp_k)

        The ``r`` branch below and the energy branch in :meth:`energy` are a matched PAIR
        per activation. Do NOT cross them (e.g. relu energy + softmax ``r``): the layer
        would still run, still train and still emit finite output, while ``update`` would
        no longer be the gradient of the energy this same layer reports and the block's
        descent guarantee would silently evaporate. It is not verifiable by inspection;
        the ONLY thing proving this pairing is ``test_gradient_oracle`` (S6b), which is
        parametrized over BOTH activations for exactly this reason.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor

        :return: ``-dE_HN/dg`` of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        h = ops.einsum('kd,bnd->bnk', self.xi, g)        # (B, N, K)

        if self.activation == 'relu':
            r = ops.relu(h)                                              # (B, N, K)
        else:
            # softmax over the MEMORY axis k (axis=-1), NOT the token axis n. Softmaxing
            # over tokens would (a) introduce token mixing into a strictly per-token layer
            # and (b) stop being the gradient of the reported energy.
            r = ops.softmax(self.hopfield_beta * h, axis=-1)             # (B, N, K)

        # Down-project with the SAME matrix, transposed (tied weights).
        return ops.einsum('kd,bnk->bnd', self.xi, r)     # == -dE_HN/dg, (B, N, D)

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
        on an UNBUILT layer.

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
