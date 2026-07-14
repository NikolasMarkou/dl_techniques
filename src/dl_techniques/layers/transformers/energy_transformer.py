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
from dl_techniques.layers.attention.energy_attention import _mask_dtype, _token_keep

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

    **STANDALONE MASKING TRAP — read this before using the layer outside a block.**
    ``call()`` takes **no** ``mask`` keyword and ``supports_masking`` is **False** (the Keras
    default; deliberately not raised). ``energy()`` and ``update()`` DO take a ``mask``, but
    Keras only injects masks into ``call()``. Consequences:

    * **Inside** ``EnergyTransformer``: **SAFE.** The block never relies on Keras injection —
      it computes the per-token keep itself (``_hopfield_token_mask``) and passes it
      EXPLICITLY into ``self.hopfield.energy(..., mask=...)`` and ``.update(..., mask=...)``.
    * **Standalone, e.g. ``Embedding(mask_zero=True) -> HopfieldNetwork``: a TRAP.** Keras
      emits a ``UserWarning`` (*"was passed an input with a mask attached"*) and **drops the
      mask**. It is worth being precise about why that MATTERS, because the intuition "PAD
      tokens are zeros, so they contribute nothing" is FALSE here (all measured):
      ``mask_zero=True`` does **not** zero the PAD vector — it emits the id-0 embedding row
      like any other row (measured ``|x_pad| = 4.7e-2``, not 0) and attaches the mask as
      **metadata only**. With the metadata dropped, a PAD token is just a token: it gets a
      real, non-zero update of the SAME order as a real one (measured ``|update| = 7.8e-3`` on
      PAD rows vs ``7.8e-3`` on real rows). Nothing errors; the output is finite and silently
      polluted. Either **mask by hand** (call ``update(g, mask=keep)`` / ``energy(g,
      mask=keep)`` yourself with an explicit rank-2 ``(B, N)`` keep), or use the
      ``EnergyTransformer`` block, which already does. Do NOT "fix" this by flipping
      ``supports_masking = True`` without ALSO adding a ``mask`` parameter to ``call()`` —
      that combination silences the warning while still dropping the mask, which is strictly
      worse than today (F-02).

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

        The ``'softmax'`` ``logsumexp`` is over the **MEMORY** axis ``k`` (axis ``-1``),
        NOT the token axis ``n``.

        **These formulas are the SPEC.** :meth:`update` must match *these*; never edit
        this method to make the gradient oracle pass (plan STOP-IF 1).

        :param g: Token state ``(B, N, D)``, typically the output of ``EnergyLayerNorm``.
        :type g: keras.KerasTensor
        :param mask: Optional rank-2 ``(B, N)`` per-token validity mask. A masked-out (PAD)
            token contributes EXACTLY ZERO to the energy — see the D-005 anchor below.
        :type mask: Optional[keras.KerasTensor]

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

        # PER-TOKEN energy first, then the reduction over tokens — so the mask below can gate
        # the token sum. Mathematically identical to reducing in one shot.
        if self.activation == 'relu':
            r = ops.relu(h)
            per_token = -0.5 * ops.sum(ops.square(r), axis=-1)           # (B, N)
        else:
            # 'softmax' — G is NOT separable per memory k: logsumexp over the MEMORY axis.
            lse = ops.logsumexp(self.hopfield_beta * h, axis=-1)         # (B, N)
            per_token = -(1.0 / self.hopfield_beta) * lse                # (B, N)

        # DECISION plan_2026-07-13_ca4f71a2/D-005
        # A masked-out (PAD) token contributes EXACTLY ZERO to E_HN. WHAT NOT TO DO:
        #   * Do NOT restore the unmasked `ops.sum(..., axis=(1, 2))`. Plan assumption A6
        #     ("HopfieldNetwork is strictly per-token, so it needs no mask") is TRUE for the
        #     state UPDATE — no token mixing, so a PAD cannot leak into a real token — and
        #     FALSE for this REDUCTION over tokens: every PAD token added its own energy to the
        #     public trace, drifting it +74.9% (3 pads) to +224.6% (9 pads) and making the
        #     energies of a variable-length batch incomparable across rows. EVERY reduction
        #     over the token axis needs the mask; only the per-token maps do not.
        #   * Do NOT mask HERE and leave `update()` unmasked. `update()` is `-dE/dg` of THIS
        #     energy: `dE/dg_n = keep_n * de_n/dg_n`, so the gradient at a masked row is
        #     exactly zero and `update()` must be zero there too. The pair is masked TOGETHER
        #     or not at all (`test_gradient_oracle[*-masked]` proves it tensor-wide, PAD rows
        #     included). The energy is the SPEC and the update follows it — never the reverse.
        # See decisions.md D-005.
        if mask is not None:
            per_token = per_token * _token_keep(mask, reduce_dtype)      # (B, N)

        energy = ops.sum(per_token, axis=1)                              # (B,)

        # DECISION plan_2026-07-13_ca4f71a2/D-005
        # Returned in the REDUCE dtype (>= float32) — NOT cast back to the compute dtype.
        # WHAT NOT TO DO: do NOT re-add `ops.cast(energy, self.compute_dtype)` here. It is the
        # exact bug fixed in `EnergyAttention.energy()` (same anchor): casting in float32 to
        # protect the accumulator and then casting the result back into fp16 reintroduces the
        # overflow on the last op — `E_HN` sums over N tokens x K memories and clears fp16's
        # 65504 max at N=1024 (measured: `-inf` under `mixed_float16`, both activations).
        # Safe because `energy()` is a REPORTED DIAGNOSTIC only: the block's state update comes
        # from `update()`, and nothing in the compute path contracts the energy against fp16
        # weights. See decisions.md D-005.
        return energy                                                     # (B,), >= float32

    # -----------------------------------------------------------------

    def update(
        self,
        g: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
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
        :param mask: Optional rank-2 ``(B, N)`` per-token validity mask — the SAME mask
            :meth:`energy` is given. A masked row's update is exactly ``0``, because the
            masked energy does not depend on that token at all (D-005 anchor below).
        :type mask: Optional[keras.KerasTensor]

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
        update = ops.einsum('kd,bnk->bnd', self.xi, r)   # == -dE_HN/dg, (B, N, D)

        # DECISION plan_2026-07-13_ca4f71a2/D-005
        # ZERO the masked rows. This is NOT a safety hack and NOT "masking the update because
        # the energy is masked" — it is the DERIVATIVE of the masked energy: E_HN is a SUM of
        # per-token terms with NO coupling, so `dE/dg_n = keep_n * de_n/dg_n`, which is exactly
        # zero wherever `keep_n == 0`. WHAT NOT TO DO:
        #   * Do NOT drop this multiply while `energy()` keeps its mask. The layer would still
        #     run and still train, `update()` would silently stop being `-dE/dg` at the PAD
        #     rows, and the block would descend a function it does not report.
        #   * Do NOT "fix" a masked-oracle failure by un-masking `energy()` instead: the energy
        #     is the SPEC (invariant I1), the update follows it.
        # No mask => byte-identical to the old path.
        #
        # SCOPE OF THE "a masked row comes out exactly as it went in" CLAIM: it holds for the
        # DESCENT (`x = x + alpha * update`), i.e. at `noise_std == 0` OR `training=False`. It
        # does NOT hold when the eq.-27 Langevin noise is on: `EnergyTransformer.call` adds
        # that noise to the WHOLE state tensor, PAD rows included (measured `max|y_pad -
        # x_pad| = 0.127` at `noise_std=0.1, training=True`). That is harmless — a PAD row
        # cannot leak into a real token (the attention keep mask zeroes every pair it
        # participates in, and this layer does not mix tokens at all), and the real tokens'
        # outputs are unchanged — but the passthrough guarantee is a claim about the UPDATE,
        # not about the noise. Do NOT "fix" a noisy PAD row by masking the noise unless a
        # caller actually needs bit-exact PAD passthrough during training; it would add a mask
        # multiply to the hot loop for a value nothing reads. See decisions.md D-005.
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

    **COST — measured, not extrapolated.** ``T`` descent steps do NOT cost ``T`` transformer
    blocks. Against a vanilla ``TransformerLayer`` at matched width, ET-Full (``T = 12``):

    .. code-block:: text

        wall clock   : 7.6x a vanilla TransformerLayer   (NOT the naive 12x)
        per ET step  : 0.63x one vanilla block
        params       : 3.54 M  vs  7.09 M  (ET has HALF the parameters)

    A step is cheaper than a block because the block is *missing pieces*: **no value matrix,
    no separate FFN up/down projection, no output projection** — the Hopfield's ``xi`` is ONE
    tied ``(K, D)`` matrix used in both directions, and attention has only ``w_key`` /
    ``w_query``. So ``T`` steps of a 0.63x block land at ~7.6x, not 12x. Two more measured
    facts a caller should budget for:

    * ``return_energy=True`` costs **1.60x** wall clock — it makes ``2T + 1`` ``_project()``
      calls instead of ``T`` (an energy reading before every step, plus the final one).
    * **Backward memory is LINEAR in** ``T``. The loop is fully unrolled and nothing is
      recomputed, so every step's activations are held for the backward pass. ``T`` is a
      memory dial, not just a compute one.

    **DO NOT "OPTIMIZE" THE PER-STEP KEEP-MASK REBUILD — it was measured, and it is free.**
    ``EnergyAttention._build_keep_mask`` is loop-invariant (it reads ``ops.shape(g)[1]``, the
    masks and ``attn_self`` — never ``g``'s *values*) yet ``_project()`` rebuilds it on every
    one of the ``T`` steps. In EAGER profiling this looks like ~27% of the forward pass, which
    is exactly why it keeps getting re-proposed. **It is not worth hoisting.** In the OPTIMIZED
    compiled graph the mask is not merely CSE-d, it is **constant-folded out of existence**
    (verified: the mask's op count is FLAT at ``T = 4/12/24`` in optimized HLO, while positive
    controls scale with ``T``; grappler independently collapses it too). Measured end-to-end
    fwd+bwd hoist delta on an idle GPU: **-0.09% (jit), -0.06% (graph), +0.25% (dynamic-N)** —
    i.e. nothing, or worse. The only real win is **+12.9% on ``run_eagerly=True``**, a DEBUG
    mode that is itself 4.2x slower than jit, so nobody trains on it. Hoisting it would widen
    three PUBLIC duck-typed signatures (``_project``/``energy``/``update``) and add a fresh way
    to break ``update() == -dE/dg`` (a half-applied mask has ALREADY shipped once here) to buy
    zero. See ``plans/plan_2026-07-14_e5955791/decisions.md`` D-005.

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
        batch mean would hide a per-sample descent violation. The energy trace is
        **``float32`` BY DESIGN** even under ``mixed_float16`` (the state output stays
        ``float16``): the trace is O(-1e5) at a realistic sequence length, which is ``-inf``
        in fp16. Both the runtime tensor (D-005) and the SYMBOLIC KerasTensor
        (:meth:`compute_output_spec`, D-006) are ``>= float32``.
        **Consuming the trace under a mixed_float16 policy: build the head ``float32``** —
        ``keras.layers.Dense(1, dtype='float32')(energies)``. A default-policy layer
        autocasts its inputs to its OWN fp16 compute dtype and an O(-1e5) energy overflows to
        ``nan``/``inf`` there. That is a property of the CONSUMER's policy, not of this
        layer's output (see the D-006 anchor at :meth:`compute_output_spec`).
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

    def _hopfield_token_mask(
        self,
        attention_mask: Optional[keras.KerasTensor],
        mask: Optional[keras.KerasTensor],
    ) -> Optional[keras.KerasTensor]:
        """The per-token ``(B, N)`` keep the HopfieldNetwork must see — from BOTH masks.

        # DECISION plan_2026-07-13_ca4f71a2/D-006
        A **rank-2** ``attention_mask`` and the Keras-propagated ``mask`` are THE SAME OBJECT
        — a per-token validity mask — and this method is what makes that documented
        equivalence (D-008, and the D-002 anchor in ``EnergyAttention._build_keep_mask``)
        actually TRUE. Iter-2 forwarded ONLY the Keras ``mask`` to ``self.hopfield``, so the
        two paths silently forked: measured on the same weights (8 tokens, 3 PAD,
        ``num_steps=3``), the Keras-mask energy was pad-invariant (0.0000% drift) while the
        rank-2-``attention_mask`` energy drifted **+27.3%** and the PAD rows kept moving
        (passthrough 4.73 vs 0.0) — the R2 bug, verbatim, on the path a caller with no
        ``Embedding(mask_zero=True)`` (e.g. an MIM model) is FORCED to use.

        WHAT NOT TO DO:
          * Do NOT derive a token keep from a rank-3 ``(B, N, N)`` or rank-4
            ``(B, H, N, N)`` ``attention_mask``. Those are PAIR-level (KEY x QUERY) masks:
            they say "token m may not attend to token n", NOT "position n is not a token".
            There is no per-token reading (a row can be half-masked), so any reduction —
            ``any``/``all`` over an axis — would INVENT a semantics the caller never
            specified, and would silently freeze rows the caller only meant to exclude from
            attention. They correctly leave the Hopfield energy alone. That is NOT the fork
            this fixes; it is a genuinely different object, and it is documented as such.
          * Do NOT mask the Hopfield ENERGY here without masking its UPDATE with the SAME
            keep. ``E_HN`` is a sum of UNCOUPLED per-token terms, so
            ``dE/dg_n = keep_n * de_n/dg_n``: masking BOTH IS the derivative of the masked
            energy (see the D-005 anchors in ``HopfieldNetwork``). Masking one alone makes
            ``update() != -dE/dg`` and the block descends a function it does not report —
            ``test_gradient_oracle`` is what catches it.
          * Do NOT re-implement the rank-2 contract: ``_token_keep`` owns it (D-005).
        See decisions.md D-006.

        :param attention_mask: Optional KEEP mask of rank 2 / 3 / 4. Only rank 2 has a
            per-token reading.
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` validity mask.
        :type mask: Optional[keras.KerasTensor]

        :return: A ``(B, N)`` 0/1 keep tensor — the LOGICAL AND of whichever of the two
            carry a per-token reading (D-003) — or ``None`` when neither does (then the
            Hopfield energy sums over every token, which is correct: nothing declared any
            token invalid).
        :rtype: Optional[keras.KerasTensor]
        """
        keep_dtype = _mask_dtype(self.compute_dtype)
        token_keep: Optional[keras.KerasTensor] = None

        if attention_mask is not None and len(attention_mask.shape) == 2:
            token_keep = ops.cast(attention_mask, keep_dtype)             # (B, N)

        if mask is not None:
            keras_keep = _token_keep(mask, keep_dtype)                    # (B, N), rank-checked
            # Multiplication IS the logical AND (D-003) — neither mask can resurrect a token
            # the other hid.
            token_keep = (
                keras_keep if token_keep is None else token_keep * keras_keep
            )

        return token_keep

    # -----------------------------------------------------------------

    def energy(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """The block's total scalar energy ``E(g) = E_ATT(g) + E_HN(g)``, shape ``(B,)``.

        This is the Lyapunov function the block descends. It is the sum of exactly TWO
        terms. The ``EnergyLayerNorm`` Lagrangian is deliberately NOT a third term — it is
        the *potential whose Hessian* ``dg/dx`` makes the descent PSD, not a term in ``E``.
        Adding it here would make :meth:`call`'s reported energy a quantity the update does
        not descend, and would silently invalidate S7. See decisions.md
        ``plan_2026-07-13_57c9833e/D-005`` (a DIFFERENT decision from this plan's D-005,
        which governs the energy's dtype and its mask — the anchors below).

        :param g: NORMALIZED token state ``(B, N, D)`` — i.e. ``self.norm(x)``, not ``x``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional KEEP mask. A **rank-2** ``(B, N)`` one is a per-token
            validity mask and reaches BOTH terms — it is equivalent to ``mask`` (D-006). A
            rank-3/rank-4 one is a KEY x QUERY PAIR mask with no per-token reading, so it
            reaches ``EnergyAttention`` only (the Hopfield net has no pairs).
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` boolean validity mask.
            ANDed with ``attention_mask`` inside ``EnergyAttention`` (decisions.md D-003), and
            ANDed again into the ``HopfieldNetwork`` token keep, whose energy is a SUM over
            tokens and so must exclude the PAD tokens (decisions.md D-005, D-006).
        :type mask: Optional[keras.KerasTensor]

        :return: Energy of shape ``(B,)``, in the REDUCE dtype (>= ``float32``) — NOT the
            compute dtype. Under ``mixed_float16`` the trace stays ``float32``; see the
            D-005 anchors in ``EnergyAttention.energy`` / :meth:`HopfieldNetwork.energy`.
        :rtype: keras.KerasTensor
        """
        # DECISION plan_2026-07-13_ca4f71a2/D-005
        # BOTH terms are returned in the reduce dtype (>= float32), so this sum type-checks
        # under EVERY policy — including `mixed_float16`, where the compute dtype is float16
        # and an O(-8e4) energy is `-inf`. Do NOT cast either term (or this sum) down to
        # `self.compute_dtype`: the energy is a reported diagnostic, consumed ONLY by the
        # `return_energy=True` trace in `call()`, never by the state update. See decisions.md
        # D-005.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-002
        # `mask` MUST be forwarded to `self.attention` HERE and at the `self.attention.update`
        # site in `call()` below — the SAME mask, through the SAME kwarg. WHAT NOT TO DO:
        #   * Do NOT forward it to only one of them. `update()` is `-dE/dg` of THIS `energy()`
        #     ONLY IF both see the same masks; a merge landing in one path leaves the layer
        #     running, training, and silently NOT descending (plan STOP-IF 2). The autodiff
        #     oracle `test_gradient_oracle` (run WITH a Keras mask) is what catches that.
        #   * Do NOT re-implement the merge here. `EnergyAttention._build_keep_mask` ANDs
        #     `mask` with `attention_mask` (D-003); a second merge site is a second thing to
        #     get wrong. See decisions.md D-002.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-005
        # `mask` ALSO goes to `self.hopfield` — and it must, for the ENERGY. This CORRECTS the
        # earlier D-002 bullet (and plan Assumption A6) that said the Hopfield net "needs no
        # mask because it is strictly per-token": that is true of the state UPDATE (no token
        # mixing => a PAD cannot leak into a real token) and FALSE of the ENERGY, which is a
        # REDUCTION over the token axis — so every PAD token was adding its own energy to the
        # public trace (+74.9% with 3 pads, +224.6% with 9). WHAT NOT TO DO: do not "restore"
        # `self.hopfield.energy(g)` without the mask; the E_ATT term is already pad-invariant,
        # so an unmasked E_HN makes exactly HALF the reported energy mask-aware — the worst of
        # both worlds, and invisible unless you compare padded vs unpadded rows.
        # See decisions.md D-005.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-006
        # The Hopfield mask comes from `_hopfield_token_mask(attention_mask, mask)` — BOTH
        # masks — NOT from `mask` alone. Forwarding only the Keras `mask` (iter-2) left the
        # rank-2 `attention_mask` path summing E_HN over PAD tokens (+27.3% drift), i.e. HALF
        # the reported energy mask-aware, on the very path the class docstring declares
        # IDENTICAL to the Keras mask. Do NOT "simplify" this back to `mask=mask`.
        # See decisions.md D-006.
        hopfield_mask = self._hopfield_token_mask(attention_mask, mask)
        return (
            self.attention.energy(g, attention_mask=attention_mask, mask=mask)
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

        **Masking.** ``supports_masking = True``, and this signature DECLARES ``mask`` —
        which is what makes Keras inject a propagated mask (e.g. from an upstream
        ``Embedding(mask_zero=True)``). Do NOT remove the parameter: ``supports_masking``
        alone only suppresses Keras' "layer does not support masking" error; without a
        ``mask`` parameter here the mask is silently DROPPED and PAD tokens influence every
        real token (F-02). Declaring it is necessary but NOT sufficient — the block must also
        FORWARD it to ``self.attention`` (see the D-002 anchor in :meth:`energy`). When BOTH
        ``mask`` and ``attention_mask`` are supplied the rule is a **logical AND** — a token
        is attended only if valid under both, and neither mask can un-mask what the other hid
        (decisions.md D-003).

        **A rank-2** ``attention_mask`` **IS a Keras** ``mask`` (D-006). Both are per-token
        VALIDITY masks and they now produce the IDENTICAL result: the token is dropped from
        the attention keep mask (symmetrically, key AND query — D-008) AND from the Hopfield
        energy's token SUM, so its energy contribution is zero, its gradient is zero, and its
        row passes through unchanged (at ``noise_std=0``; see the ``HopfieldNetwork.update``
        D-005 anchor for the eq.-27 noise caveat). Use whichever your pipeline produces.

        **A rank-3** ``(B, N, N)`` **or rank-4** ``(B, H, N, N)`` ``attention_mask`` is a
        different object: a PAIR-level (KEY x QUERY) keep mask. It says "``m`` may not attend
        to ``n``", not "``n`` is not a token" — a row can be half-masked — so it has NO
        per-token reading and it does NOT mask the Hopfield energy. That is a genuine
        semantic difference, not a fork.

        :param inputs: Token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional KEEP mask (``1`` = attend). Rank-2 ``(B, N)`` = a
            per-token validity mask, equivalent to ``mask`` (D-006, D-008). Rank-3
            ``(B, N, N)`` / rank-4 ``(B, H, N, N)`` = a pair-level key x query mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: When truthy AND ``noise_std > 0``, the eq. 27 noise is injected.
            May be a Python ``bool``/``None`` **or a symbolic scalar bool tensor** (a traced
            graph function passes the latter): both are honoured, and a tensor is gated
            without ever calling ``bool()`` on it (decisions.md D-003).
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

        # DECISION plan_2026-07-14_e5955791/D-003: the eq. 27 noise is gated in TWO
        # stages, and the split is load-bearing. Do NOT collapse it back to
        # `add_noise = self.noise_std > 0.0 and training`: Python `and` calls
        # `Tensor.__bool__()`, so the moment a caller wraps this layer in a traced graph
        # function and `training` arrives as a SYMBOLIC bool tensor, that line raises
        # `OperatorNotAllowedInGraphError`. (`fit`/`predict`/`jit_compile` resolve
        # `training` to a Python bool, which is why 247 tests never saw it.)
        #
        #   1. `self.noise_std > 0.0` reads a CONFIG FLOAT (the ctor validates it as a
        #      Python number), so it is a PYTHON bool at trace time and STAYS a Python
        #      `if`. That is what keeps the DEFAULT (`noise_std == 0.0`) path
        #      trace-time-eliminated: it never constructs the RNG op, and therefore never
        #      a cond over it. Do NOT "simplify" this to one uniform `ops.cond` — that
        #      would put a tensor cond on the path ~100% of users run, for a feature
        #      almost nobody enables.
        #   2. Only `training` may be a tensor, and only INSIDE that branch is it
        #      resolved: a Python bool/`None` short-circuits exactly as before; a tensor
        #      is gated with `ops.where` below. Do NOT coerce a traced `training` to a
        #      Python bool via a Keras helper — no stable Keras 3 helper is CORRECT for a
        #      genuinely-traced flag, and coercion would make a `False` tensor ADD noise
        #      (or a `True` tensor skip it): a loud crash traded for a silent wrong answer.
        add_noise = False                                    # unconditional (Python) noise
        noise_gate: Optional[keras.KerasTensor] = None       # tensor-gated noise
        if self.noise_std > 0.0:
            if training is None or isinstance(training, bool):
                add_noise = bool(training)
            else:
                noise_gate = ops.cast(training, "bool")

        # The per-token keep the Hopfield module sees: the AND of the Keras `mask` and a
        # RANK-2 `attention_mask` (D-006). Hoisted out of the loop — it does not depend on the
        # descent step. It must be the SAME tensor `self.energy()` uses, or `update()` stops
        # being `-dE/dg` (the D-006 anchor at `_hopfield_token_mask`).
        hopfield_mask = self._hopfield_token_mask(attention_mask, mask)

        for _ in range(self.num_steps):
            g = self.norm(x)

            if self.return_energy:
                energies.append(
                    self.energy(g, attention_mask=attention_mask, mask=mask)
                )

            # `update` == -dE/dg (BOTH sub-layers return the DESCENT DIRECTION, never the
            # gradient). See the class docstring, point 1.
            #
            # DECISION plan_2026-07-13_ca4f71a2/D-005: `mask` goes to BOTH sub-layers' updates
            # — and it must be the SAME mask the `self.energy()` call above passes to their
            # `energy()`, or `update() != -dE/dg` (plan STOP-IF 2). The Hopfield update is
            # masked because the masked ENERGY has zero gradient at a PAD row, not for safety;
            # see the D-005 anchor in `HopfieldNetwork.update`. (This supersedes the original
            # D-002 bullet "do NOT pass `mask` to `self.hopfield`", written when Assumption A6
            # was believed to cover the energy too.) Since D-006 the Hopfield's mask is
            # `hopfield_mask` (Keras `mask` AND a rank-2 `attention_mask`) — exactly what
            # `self.energy()` passes it, which is the property that keeps the pair a
            # (energy, -gradient) pair.
            update = (
                self.attention.update(g, attention_mask=attention_mask, mask=mask)
                + self.hopfield.update(g, mask=hopfield_mask)
            )

            # ADDING alpha * update IS `x = x - alpha * dE/dg`. The gradient is w.r.t. `g`
            # and applied to `x` — paper eq. 6, NOT a typo. See the class docstring,
            # point 2. Flipping this sign gives energy ASCENT, which still runs.
            x = x + self.step_size * update

            if add_noise or noise_gate is not None:
                # eq. 27 (Langevin). `ops.shape(x)` — NEVER a Python int off the batch or
                # token axis, or the layer breaks under a symbolic/variable batch size.
                noisy = x + self._sqrt_step_size * self.noise_std * keras.random.normal(
                    shape=ops.shape(x),
                    dtype=self.compute_dtype,
                    seed=self.seed_generator,
                )
                # DECISION plan_2026-07-14_e5955791/D-003: the tensor branch draws the
                # noise and then DISCARDS it when the gate is False. That wasted draw is
                # the accepted price of a graph-safe `training`, and it is paid ONLY when
                # `noise_std > 0` AND `training` is a traced tensor — never on the default
                # path (see the two-stage gate above).
                x = noisy if noise_gate is None else ops.where(noise_gate, noisy, x)

        if self.return_energy:
            # The (T+1)-th reading: the energy AFTER the last step. Without it the caller
            # cannot see the effect of the final update, and `diff(energies)` would only
            # cover T-1 of the T steps.
            g = self.norm(x)
            energies.append(self.energy(g, attention_mask=attention_mask, mask=mask))
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
        """Propagate the token mask — but NEVER onto the energy tensor.

        # DECISION plan_2026-07-13_ca4f71a2/D-002
        With ``return_energy=True`` this layer emits a TUPLE ``(x, energies)`` of shapes
        ``(B, N, D)`` and ``(B, T + 1)``. The incoming mask is ``(B, N)`` — a PER-TOKEN
        validity mask. Do NOT inherit ``Layer.compute_mask``'s default (return the mask
        unchanged): Keras would then attach the ``(B, N)`` token mask to the ``(B, T + 1)``
        energy tensor as well, and any downstream mask-consuming layer would receive a
        mask whose shape does not match its input. The energy is a scalar-per-step
        REDUCTION over tokens — no token axis survives, so it carries no token mask.
        See decisions.md D-002.

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

    def compute_output_spec(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Symbolic output spec — the ENERGY is ``>= float32``, the state is compute dtype.

        # DECISION plan_2026-07-13_ca4f71a2/D-006
        This method exists ONLY to tell the truth about the energy's DTYPE, and it is
        LOAD-BEARING — do NOT delete it as "redundant with ``compute_output_shape``".
        Keras' default ``compute_output_spec`` derives the shapes from
        :meth:`compute_output_shape` and then stamps EVERY output with
        ``self.compute_dtype``. Under ``mixed_float16`` that made the symbolic ``energies``
        KerasTensor claim ``float16`` while the tensor actually produced at runtime is
        ``float32`` (D-005: the energy is returned in the REDUCE dtype so an O(-1e5) trace
        does not overflow fp16's 65504 limit). ``model.outputs`` therefore reported
        ``['float16', 'float16']`` for a model whose ``predict`` returns
        ``['float16', 'float32']`` — the PUBLIC dtype contract, which is what an exporter,
        a downstream shape/dtype inference, or a reader of ``model.summary()`` believes.

        WHAT NOT TO DO:
          * Do NOT let the energy be stamped ``self.compute_dtype``. The graph would again
            advertise fp16 for a float32 tensor.
          * Do NOT "fix" the dtype mismatch by casting the energy DOWN in :meth:`energy`
            instead — that IS the D-005 bug (`-inf` at N >= 512; see the anchors in
            ``EnergyAttention.energy`` / :meth:`HopfieldNetwork.energy`).
          * Do NOT drop :meth:`compute_output_shape`: it is still the SHAPE authority, and
            the shapes here are taken FROM it, never re-derived.

        # DECISION plan_2026-07-13_ca4f71a2/D-007
        **MEASURED FACT, do not mis-attribute it to this method** (it is why the energy head
        below must be float32): under a GLOBAL ``mixed_float16`` policy, a downstream layer
        autocasts its float inputs to its OWN ``compute_dtype`` — ``Layer.__call__`` does
        this from the layer's policy, NOT from the upstream symbolic dtype. So
        ``keras.layers.Dense(1)(energies)`` overflows to ``nan``/``inf`` at ``N=1024``
        (E ~ -8e4) BOTH before and after this method existed; control-proven with a plain
        float32 ``keras.Input``, no ET involved. This method cannot prevent that, and it does
        not claim to. **A head consuming the energy trace must be built ``dtype='float32'``**
        (see the ``return_energy`` docstring). What this method fixes is the LIE — the graph
        now says float32, so a dtype-aware consumer, an exporter, and
        ``model.outputs[1].dtype`` all see the tensor that actually flows.
        See decisions.md D-006.

        :param inputs: Symbolic token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Unused (does not affect shape or dtype).
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Unused (does not affect shape or dtype).
        :type training: Optional[bool]
        :param mask: Unused (does not affect shape or dtype).
        :type mask: Optional[keras.KerasTensor]

        :return: ``KerasTensor(B, N, D)`` in the compute dtype; or that plus
            ``KerasTensor(B, num_steps + 1)`` in the REDUCE dtype (>= ``float32``) when
            ``return_energy=True``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        shape = self.compute_output_shape(inputs.shape)   # ONE shape authority

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
        })
        return config

# ---------------------------------------------------------------------
