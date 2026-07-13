"""
Energy Transformer (ET) — Hopfield associative memory module + the ET block.

Implements the modules of the Energy Transformer, Hoover, Liang, Pham, Panda, Strobelt,
Zaki, Chau, Krotov, "Energy Transformer", NeurIPS 2023 (https://arxiv.org/abs/2302.07253).

This module holds:

- :class:`HopfieldNetwork` — the ET Hopfield / associative-memory module (eq. 5, 9). A
  single tied ``(K, D)`` memory matrix, applied strictly per token.
- :class:`EnergyTransformer` — the ``T``-step energy-descent block (eq. 6, alg. 1; eq. 27
  for the optional noise). It composes :class:`HopfieldNetwork` with ``EnergyLayerNorm``
  (``layers/norms/energy_layer_norm.py``) and ``EnergyAttention``
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

import math
import keras
from keras import ops, initializers
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# DECISION plan_2026-07-13_57c9833e/D-004
# These are DIRECT imports of the three CONCRETE classes, deliberately NOT
# `create_normalization_layer(...)` / `create_attention_layer(...)` factory calls. Do NOT
# "improve" this into factory composition. `EnergyTransformer` does not consume these as
# generic `keras.layers.Layer`s — it calls their `energy(g, ...)` / `update(g, ...)` pair,
# which is a duck-typed convention private to this feature and is NOT part of any Keras
# layer contract and NOT guaranteed by any factory type. A factory returns a generic
# layer; calling `.energy()` on it is an unchecked duck-type that would raise
# `AttributeError` at runtime for any of the other 30 registered attention types (or 16
# norm types). Precedent for reuse-by-direct-import in this package:
# `ideogram4_block.py:36-46` direct-imports RMSNorm / SwiGLUFFN / Ideogram4Attention.
# The factory registrations of `EnergyLayerNorm` ('energy_layer_norm') and
# `EnergyAttention` ('energy') still exist and are still correct — they are there for
# THIRD-PARTY reuse, not for this block's own composition. See decisions.md D-004.
from dl_techniques.layers.norms.energy_layer_norm import EnergyLayerNorm
from dl_techniques.layers.attention.energy_attention import EnergyAttention

# The "reduce in AT LEAST float32" rule (D-009). IMPORTED, not re-implemented: the dtype
# rule must have exactly ONE definition, or the two ET modules drift apart the first time
# one of them is fixed. See decisions.md D-009.
from dl_techniques.layers.attention.energy_attention import _mask_dtype

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

        # C-2: this is a PUBLIC method, callable OUTSIDE `__call__` — where Keras has NOT
        # opened an autocast scope. Without this cast a float32 `g` meets float16 `xi` under
        # `mixed_float16` and the einsum raises InvalidArgumentError. The duck-typed
        # energy()/update() convention is the layer's ADVERTISED surface, so it must be safe
        # to call standalone.
        g = ops.cast(g, self.compute_dtype)

        h = ops.einsum('kd,bnd->bnk', self.xi, g)        # (B, N, K)

        # The reduction runs in AT LEAST float32: E_HN sums over N tokens x K memories, and
        # an fp16 accumulator overflows (max 65504) long before the layer itself misbehaves.
        # Same reasoning as the D-009 anchor in `energy_attention.py`.
        reduce_dtype = _mask_dtype(self.compute_dtype)
        h = ops.cast(h, reduce_dtype)

        if self.activation == 'relu':
            r = ops.relu(h)
            energy = -0.5 * ops.sum(ops.square(r), axis=(1, 2))          # (B,)
        else:
            # 'softmax' — G is NOT separable per memory k: logsumexp over the MEMORY axis.
            lse = ops.logsumexp(self.hopfield_beta * h, axis=-1)         # (B, N)
            energy = -(1.0 / self.hopfield_beta) * ops.sum(lse, axis=1)  # (B,)

        return ops.cast(energy, self.compute_dtype)

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

        # C-2: cast at the head of the public method — see the note in `energy()`.
        g = ops.cast(g, self.compute_dtype)

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


@keras.saving.register_keras_serializable()
class EnergyTransformer(keras.layers.Layer):
    """Energy Transformer block — ``T`` steps of gradient descent on ONE scalar energy.

    **Intent**: replace the standard ``attn -> FFN`` residual stream with an explicit,
    provable optimization. There is no FFN and no value matrix. The block defines a single
    scalar energy ``E(g) = E_ATT(g) + E_HN(g)`` and repeatedly steps the token state
    DOWNHILL on it (paper eq. 6, alg. 1; eq. 27 for the optional noise).

    .. code-block:: text

        for t in 1..T:
            g      = EnergyLayerNorm(x)                  # the "activation function"
            update = attn.update(g) + hopfield.update(g) # == -dE/dg
            x      = x + alpha * update                  # == x - alpha * dE/dg
            [+ sqrt(alpha) * noise_std * N(0, 1)   if training and noise_std > 0]

    **TWO THINGS A FUTURE READER WILL TRY TO "FIX". BOTH ARE CORRECT AS WRITTEN.**

    1. **SIGN DISCIPLINE — the block ADDS.** Every ``update()`` in this feature
       (:class:`HopfieldNetwork`, ``EnergyAttention``) returns ``-dE/dg``: the **DESCENT
       DIRECTION**, *not* the gradient. So :meth:`call` performs ``x = x + alpha * update``,
       which IS ``x = x - alpha * dE/dg``. A reader who assumes ``update()`` returns
       ``+dE/dg`` will "fix" this to a subtraction and silently invert the dynamics into
       energy **ASCENT** — which still runs, still trains, and still produces finite,
       plausible-looking outputs. The only thing that catches it is
       ``test_energy_is_non_increasing`` (S7).
    2. **THE GRADIENT IS TAKEN W.R.T.** ``g`` **AND APPLIED TO** ``x``. This is paper
       eq. 6 (``tau dx/dt = -dE/dg``), **not a typo**. ``EnergyLayerNorm`` is the
       "activation function" ``g = dL/dx`` of a Lagrangian ``L`` whose Hessian ``dg/dx`` is
       PSD (for ``gamma > 0``); that is *exactly* what makes the descent provable:
       ``dE/dt = -(dE/dg)^T (dg/dx) (dE/dg) <= 0``. Do NOT "correct" it to ``-dE/dx``.

    **Descent is claimed only for** ``noise_std == 0``, ``gamma > 0`` and a sufficiently
    small ``step_size``. With ``noise_std > 0`` this is Langevin sampling (eq. 27), not
    descent, and the energy is expected to fluctuate upward.

    **Composition (Keras 3 golden pattern).** The three sub-layers are created in
    ``__init__`` and EXPLICITLY built in :meth:`build`. They are never created in
    :meth:`build` or :meth:`call` — a lazily-created sub-layer silently DROPS ITS WEIGHTS
    on a ``.keras`` round-trip.

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
    :param beta: Attention inverse temperature. ``None`` -> ``1 / sqrt(head_dim)``, resolved
        by ``EnergyAttention`` itself (this block does NOT duplicate that default).
    :type beta: Optional[float]
    :param attn_self: If ``False`` (default, the paper's ET-Full), a token does not attend
        to itself.
    :type attn_self: bool
    :param hopfield_activation: ``'relu'`` (default) or ``'softmax'``.
    :type hopfield_activation: str
    :param noise_std: Std-dev of the eq. 27 noise, injected as
        ``sqrt(step_size) * noise_std * N(0, 1)`` and **only when ``training`` is truthy**.
        ``0.0`` (default) disables it.
    :type noise_std: float
    :param return_energy: If ``True``, :meth:`call` returns ``(x, energies)`` with
        ``energies`` of shape ``(B, num_steps + 1)`` — **PER-SAMPLE, not batch-reduced**: a
        batch mean would hide a per-sample descent violation.
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

    Attributes:
        norm: The inner ``EnergyLayerNorm`` (scalar gamma, vector delta).
        attention: The inner ``EnergyAttention`` (token mixing; the ONLY masked module).
        hopfield: The inner :class:`HopfieldNetwork` (per token).

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
        # DECISION plan_2026-07-13_57c9833e/D-007
        # These THREE kwargs are ADDITIVE to the user-approved signature and each one is
        # FORCED, not speculative. Do NOT drop them as "unused parameters", and do NOT
        # collapse `hopfield_beta` into `beta`:
        #   * `hopfield_beta` — a SECOND, INDEPENDENT temperature. `beta` defaults to
        #     `1/sqrt(head_dim)`, which is the softmax temperature over N TOKENS; that is a
        #     MEANINGLESS temperature for a softmax over K = `hopfield_dim` MEMORIES.
        #     Reusing one `beta` for both would silently couple two unrelated temperatures
        #     and make the Hopfield energy's sharpness a function of `head_dim`.
        #   * `norm_epsilon` — a pass-through to `EnergyLayerNorm.epsilon`. Without it the
        #     block cannot configure its own norm, and the norms factory (which forwards an
        #     `epsilon` via `setdefault`) would be able to express a configuration this
        #     block cannot.
        #   * `seed` — `noise_std > 0` makes this layer STOCHASTIC, and an unseeded
        #     stochastic layer FLAKES the save/load round-trip test (LESSONS [I:4]).
        # All three default to the approved behavior, so the approved surface is preserved
        # byte-for-byte. See decisions.md D-007.
        hopfield_beta: float = 1.0,
        norm_epsilon: float = 1e-5,
        seed: Optional[int] = None,
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

        # sqrt(alpha), the eq. 27 noise scaling. Precomputed: `step_size` is a Python float.
        self._sqrt_step_size = math.sqrt(self.step_size)

        # ----- sub-layers: CREATED IN __init__, built in build() -----
        # The Keras 3 golden pattern. A sub-layer created lazily (in `build()` or `call()`)
        # is not tracked at serialization time and SILENTLY DROPS ITS WEIGHTS on a `.keras`
        # round-trip (MEMORY: reference_subclassed_model_lazy_build_serialization).
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-001: pass `self.dtype_policy` (the POLICY
        # OBJECT) — NOT `self.dtype`, and NOT nothing.
        #   * Nothing (the bug this replaces): a sub-layer with no `dtype=` reads the
        #     GLOBAL policy, so `EnergyTransformer(dtype='float64')` built float32
        #     sub-layers and died at `x = x + self.step_size * update` below with
        #     `InvalidArgumentError: cannot compute AddV2 ... expected double ... is float`.
        #   * `self.dtype` (the sibling convention at `free_transformer.py:412`) is the
        #     VARIABLE dtype, which is 'float32' under ANY mixed policy. It is a PARTIAL
        #     fix that goes green on float64 while leaving `dtype='mixed_float16'` crashing
        #     — and, measured in the step-1 probe matrix, it also BREAKS the currently-green
        #     GLOBAL-mixed_float16 path (it would pin the sub-layers to float32 while the
        #     block computes in float16). Do NOT "simplify" this to `self.dtype` to match
        #     the sibling.
        # Only the policy object (or its `.name`) carries compute AND variable dtype.
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
        """EXPLICITLY build every sub-layer, then ``super().build()`` LAST.

        All three sub-layers see the SAME shape: the block is shape-preserving at every
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

        super().build(input_shape)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def energy(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """The block's total scalar energy ``E(g) = E_ATT(g) + E_HN(g)``, shape ``(B,)``.

        This is the Lyapunov function the block descends. It is the sum of exactly TWO
        terms. The ``EnergyLayerNorm`` Lagrangian is deliberately NOT a third term — it is
        the *potential whose Hessian* ``dg/dx`` makes the descent PSD, not a term in ``E``.
        Adding it here would make :meth:`call`'s reported energy a quantity the update does
        not descend, and would silently invalidate S7. See decisions.md D-005.

        :param g: NORMALIZED token state ``(B, N, D)`` — i.e. ``self.norm(x)``, not ``x``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional KEEP mask, forwarded to ``EnergyAttention`` only
            (``HopfieldNetwork`` is strictly per-token and takes no mask).
        :type attention_mask: Optional[keras.KerasTensor]

        :return: Energy of shape ``(B,)``.
        :rtype: keras.KerasTensor
        """
        return (
            self.attention.energy(g, attention_mask=attention_mask)
            + self.hopfield.energy(g)
        )

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Run ``num_steps`` of energy descent on the token state (paper eq. 6, alg. 1).

        :param inputs: Token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional KEEP mask (``1`` = attend). A rank-2 ``(B, N)`` mask
            is a SYMMETRIC per-token validity mask (key AND query axes) — see decisions.md
            D-008.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: When truthy AND ``noise_std > 0``, the eq. 27 noise is injected.
        :type training: Optional[bool]

        :return: ``x`` of shape ``(B, N, D)``; or ``(x, energies)`` with ``energies`` of
            shape ``(B, num_steps + 1)`` when ``return_energy=True``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        x = inputs
        add_noise = self.noise_std > 0.0 and training
        energies: List[keras.KerasTensor] = []

        for _ in range(self.num_steps):
            g = self.norm(x)

            if self.return_energy:
                energies.append(self.energy(g, attention_mask=attention_mask))

            # `update` == -dE/dg (BOTH sub-layers return the DESCENT DIRECTION, never the
            # gradient). See the class docstring, point 1.
            update = (
                self.attention.update(g, attention_mask=attention_mask)
                + self.hopfield.update(g)
            )

            # ADDING alpha * update IS `x = x - alpha * dE/dg`. The gradient is w.r.t. `g`
            # and applied to `x` — paper eq. 6, NOT a typo. See the class docstring,
            # point 2. Flipping this sign gives energy ASCENT, which still runs.
            x = x + self.step_size * update

            if add_noise:
                # eq. 27 (Langevin). `ops.shape(x)` — NEVER a Python int off the batch or
                # token axis, or the layer breaks under a symbolic/variable batch size.
                x = x + self._sqrt_step_size * self.noise_std * keras.random.normal(
                    shape=ops.shape(x),
                    dtype=self.compute_dtype,
                    seed=self.seed_generator,
                )

        if self.return_energy:
            # The (T+1)-th reading: the energy AFTER the last step. Without it the caller
            # cannot see the effect of the final update, and `diff(energies)` would only
            # cover T-1 of the T steps.
            g = self.norm(x)
            energies.append(self.energy(g, attention_mask=attention_mask))
            return x, ops.stack(energies, axis=-1)            # (B, N, D), (B, T + 1)

        return x

    # -----------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]:
        """Return the output shape — valid on an UNBUILT layer.

        Uses ONLY the passed shape and stored config ints (``num_steps``), never a weight
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
        })
        return config

# ---------------------------------------------------------------------
