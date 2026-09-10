"""Float64 NumPy transcription of the H-Net dynamic-chunking reference.

This module is an **oracle**, not an implementation. Every expression below was
transcribed line by line from the vendored PyTorch sources in `_reference/`
*before* any Keras layer of this port existed, and the `file:line` of each
transcribed expression is quoted in the docstring immediately above it. Read it
with `_reference/dc.py` open beside it; the line numbers cited are that file's.

Why the module is shaped this way
---------------------------------

An oracle written by the same hand, in the same session, from the same mental
model as the code under test is not an oracle -- it is a second copy of the same
misunderstanding, and it agrees with a defect as happily as with a fix. Three
things are done about that, and all three are load-bearing:

1. **Transcription, not recall.** Nothing here is written from what the formula
   is remembered to be. Each line is a transliteration of a specific PyTorch
   line, cited.
2. **Ordering.** This module is written before `RoutingModule`, `ChunkLayer` and
   `DeChunkLayer` exist, so it cannot be shaped to agree with them.
3. **Hand-computed pins.** `test_hnet_reference_numpy.py` pins one value per
   function to arithmetic carried out on paper from a 4-token, 2-channel
   example. A number produced by running this module can never validate this
   module.

Deliberate scope limits
-----------------------

- **Padded/masked path only.** The reference branches everywhere on packed mode
  (`cu_seqlens`, ragged `(T, D)`); that branch is an artifact of the CUDA kernel
  APIs and is not ported (plan decision D-007). The packed lines are cited where
  they are skipped, never silently dropped.
- **No `inference_params` / `step()` path.** The single exception is
  `dc.py:333`, the explicit non-kernel EMA recurrence, which is used as the
  ground truth for `dechunk_reference` -- see that function's docstring.
- **float64 throughout.** Everything is cast to float64 on entry. This module is
  the reference the float32 layers are graded against, so it must not itself
  carry float32 noise.

Naming: no `test_` prefix, so pytest never collects this file.
"""

from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants transcribed from the reference
# ---------------------------------------------------------------------------

#: `F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)` -- `dc.py:95-96`.
PAD_PROB = 1.0

#: `torch.clamp(..., min=1e-4, max=1 - (1e-4))` -- `dc.py:256`.
P_CLAMP_MIN = 1e-4
P_CLAMP_MAX = 1.0 - 1e-4

#: `torch.nn.functional.normalize` default `eps` (PyTorch 2.x): the denominator
#: is `max(||x||_2, eps)`, NOT `||x||_2 + eps`. Used by `dc.py:88-89`.
_NORMALIZE_EPS = 1e-12


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """`torch.nn.functional.normalize(x, dim=-1)` -- called at `dc.py:88-89`.

    PyTorch computes ``x / clamp_min(||x||_2, eps)`` with ``eps = 1e-12``. The
    clamp (rather than an added epsilon) matters for the all-zero rows a padding
    mask produces: it makes them normalize to exactly zero instead of to NaN.

    :param x: Array to normalize.
    :type x: numpy.ndarray
    :param axis: Axis to normalize along.
    :type axis: int
    :return: float64 array, same shape as ``x``.
    :rtype: numpy.ndarray
    """
    x = np.asarray(x, dtype=np.float64)
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    return x / np.maximum(norm, _NORMALIZE_EPS)


def _stable_partition_indices(boundary_mask: np.ndarray) -> np.ndarray:
    """The padded-mode stable partition -- `dc.py:188-191` (and `dc.py:265-269`).

    Transcribed verbatim::

        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )                                                        # dc.py:188-190
        seq_sorted_indices = torch.argsort(token_idx, dim=1)      # dc.py:191

    Kept positions take keys ``0..L-1`` and dropped positions ``L..2L-1``, so
    every key is distinct by construction: boundary positions sort to the front
    in original order, non-boundary positions follow in original order, and
    ``argsort`` stability is never actually required. (Measured: zero duplicate
    keys in the worst case over 200 random draws -- plan decision D-010(a).)

    :param boundary_mask: ``(B, L)`` boolean.
    :type boundary_mask: numpy.ndarray
    :return: ``(B, L)`` int64 permutation, one row per batch element.
    :rtype: numpy.ndarray
    """
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    _, seq_len = boundary_mask.shape
    token_idx = np.arange(seq_len)[None, :] + (~boundary_mask).astype(np.int64) * seq_len
    return np.argsort(token_idx, axis=1, kind="stable")


def routing_reference(
    hidden_states: np.ndarray,
    mask: Optional[np.ndarray] = None,
    q_weight: Optional[np.ndarray] = None,
    k_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`RoutingModule.forward` -- `dc.py:69-138`, padded/masked branch only.

    Transcription, expression by expression:

    - ``q_proj_layer`` / ``k_proj_layer`` are ``nn.Linear(d, d, bias=False)``
      whose weights are copied from ``torch.eye(d_model)`` and flagged
      ``_no_reinit`` (`dc.py:53-59`), so the DEFAULT here (``q_weight is None``)
      is the identity and this function then computes raw adjacent cosine
      similarity. ``nn.Linear`` computes ``x @ W.T``; that is transcribed
      literally below so a caller passing a non-symmetric weight gets the
      reference's orientation, not the transpose of it.
    - Cosine similarity (`dc.py:86-90`)::

          cos_sim = torch.einsum(
              "b l d, b l d -> b l",
              F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
              F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
          )

      Note the shift direction: entry ``t`` of ``cos_sim`` pairs ``h_t`` (through
      ``q``) with ``h_{t+1}`` (through ``k``), and after the left-pad below it
      becomes the boundary probability of position ``t+1``.
    - ``boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)``
      (`dc.py:92`), described by its own comment as a no-op absent precision
      issues.
    - ``boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)``
      (`dc.py:95-96`) -- position 0 of every row is forced to probability 1.0.
    - `dc.py:98-100` (`boundary_prob[cu_seqlens[:-1]] = PAD_PROB`) is the packed
      branch and is NOT transcribed: D-007 ports the padded path only.
    - ``boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)``
      (`dc.py:102`) -- a 2-class distribution ``[1 - p, p]``.
    - ``selected_idx = torch.argmax(boundary_prob, dim=-1)`` (`dc.py:104`) and
      ``boundary_mask = selected_idx == 1`` (`dc.py:106`).

      **The tie matters and is transcribed, not decided.** At exactly ``p = 0.5``
      the two classes are equal and ``argmax`` returns the FIRST maximal index,
      i.e. 0, i.e. NOT a boundary. So the realised predicate is ``p > 0.5``, not
      ``p >= 0.5``. ``np.argmax`` has the same first-maximum rule as
      ``torch.argmax`` on CPU, so the transliteration is exact. Prose that
      describes this as a "hard threshold at ``p >= 0.5``" is off by the tie
      case, and ``p = 0.5`` is reachable exactly (any two orthogonal adjacent
      hidden states give ``cos_sim = 0``). `test_hnet_reference_numpy.py` pins
      the tie.
    - ``boundary_mask = boundary_mask & mask`` (`dc.py:107-109`) -- no invalid
      token may be selected.
    - `dc.py:111-128` is the `inference_params` cache update; not transcribed.
    - ``selected_probs = boundary_prob.gather(dim=-1,
      index=selected_idx.unsqueeze(-1))`` (`dc.py:130-132`) -- the probability of
      whichever class won, i.e. ``p`` at a boundary and ``1 - p`` elsewhere.
      Note this is gathered from the UNMASKED ``selected_idx``, so a padded
      position still reports its own winning class even though its
      ``boundary_mask`` entry was just forced to False.

    :param hidden_states: ``(B, L, D)`` hidden states.
    :type hidden_states: numpy.ndarray
    :param mask: ``(B, L)`` boolean validity mask, or ``None``.
    :type mask: numpy.ndarray or None
    :param q_weight: ``(D, D)`` query projection weight in ``nn.Linear`` layout
        (applied as ``x @ W.T``); ``None`` means the reference's identity init.
    :type q_weight: numpy.ndarray or None
    :param k_weight: ``(D, D)`` key projection weight, same layout.
    :type k_weight: numpy.ndarray or None
    :return: ``(boundary_prob (B, L, 2), boundary_mask (B, L) bool,
        selected_probs (B, L, 1))`` -- the three fields of
        ``RoutingModuleOutput`` (`dc.py:14-18`, `dc.py:134-138`).
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    _, _, d_model = hidden_states.shape

    q_weight = np.eye(d_model, dtype=np.float64) if q_weight is None else np.asarray(
        q_weight, dtype=np.float64
    )
    k_weight = np.eye(d_model, dtype=np.float64) if k_weight is None else np.asarray(
        k_weight, dtype=np.float64
    )

    # dc.py:86-90 -- q sees h[:, :-1], k sees h[:, 1:]; nn.Linear is x @ W.T.
    q_proj = _normalize(hidden_states[:, :-1] @ q_weight.T)
    k_proj = _normalize(hidden_states[:, 1:] @ k_weight.T)
    cos_sim = np.einsum("bld,bld->bl", q_proj, k_proj)

    # dc.py:92
    boundary_prob = np.clip((1.0 - cos_sim) / 2.0, 0.0, 1.0)

    # dc.py:95-96 -- left-pad position 0 with PAD_PROB.
    boundary_prob = np.pad(
        boundary_prob, ((0, 0), (1, 0)), mode="constant", constant_values=PAD_PROB
    )

    # dc.py:102
    boundary_prob = np.stack((1.0 - boundary_prob, boundary_prob), axis=-1)

    # dc.py:104-106
    selected_idx = np.argmax(boundary_prob, axis=-1)
    boundary_mask = selected_idx == 1
    if mask is not None:
        # dc.py:107-109
        boundary_mask = boundary_mask & np.asarray(mask, dtype=bool)

    # dc.py:130-132
    selected_probs = np.take_along_axis(
        boundary_prob, selected_idx[..., None], axis=-1
    )

    return boundary_prob, boundary_mask, selected_probs


def chunk_reference(
    hidden_states: np.ndarray,
    boundary_mask: np.ndarray,
    max_chunks: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """`ChunkLayer.forward` -- `dc.py:169-207`, padded/masked branch only.

    Transcribed verbatim from the ``else`` branch (`dc.py:181-205`)::

        num_tokens = boundary_mask.sum(dim=-1)                    # dc.py:183
        next_max_seqlen = int(num_tokens.max())                   # dc.py:184
        token_idx = arange(L)[None, :] + (~boundary_mask).long() * L
        seq_sorted_indices = torch.argsort(token_idx, dim=1)      # dc.py:188-191
        next_hidden_states = torch.gather(
            hidden_states, dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                -1, -1, hidden_states.shape[-1]),
        )                                                         # dc.py:193-199
        next_mask = arange(next_max_seqlen)[None, :] < num_tokens[:, None]
                                                                  # dc.py:201-204

    The packed branch (`dc.py:174-180`, boolean-mask gather + recomputed
    ``cu_seqlens``) is not transcribed (D-007).

    **The one deliberate divergence, and it is a parameter, not a rewrite.** The
    reference's output width is ``next_max_seqlen = max(boundary_mask.sum(-1))``
    -- data-dependent AND batch-dependent, so row ``i``'s padding is a function
    of row ``j``'s bytes. This port fixes the width with a ``max_chunks``
    constructor argument (D-007) and keeps the FIRST ``max_chunks`` boundaries in
    POSITION order. Passing ``max_chunks=None`` here reproduces the reference's
    own width exactly, so the divergence can be switched off and the two widths
    compared. Everything inside the width is `dc.py:186-204` unchanged.

    Two consequences of taking `dc.py:201-204` verbatim under a fixed width,
    both intended:

    - Columns beyond a row's own boundary count hold whatever the stable
      partition placed there -- NON-boundary hidden states, in position order.
      They are garbage marked invalid by ``next_mask``, exactly as upstream.
    - When a row has MORE boundaries than ``max_chunks``, ``arange(max_chunks) <
      num_tokens`` is all-True, which is right: every column of that row is a
      real chunk, and the row's tail boundaries are simply lost. That loss is
      D-007's named new failure mode.

    :param hidden_states: ``(B, L, D)`` hidden states.
    :type hidden_states: numpy.ndarray
    :param boundary_mask: ``(B, L)`` boolean.
    :type boundary_mask: numpy.ndarray
    :param max_chunks: Fixed output width; ``None`` reproduces the reference's
        data-dependent ``max(boundary_mask.sum(-1))``.
    :type max_chunks: int or None
    :return: ``(next_hidden_states (B, W, D), next_mask (B, W) bool)`` with
        ``W = max_chunks`` (or the reference width when ``max_chunks is None``).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)

    # dc.py:183
    num_tokens = boundary_mask.sum(axis=-1)
    # dc.py:184, or the D-007 fixed cap.
    width = int(num_tokens.max()) if max_chunks is None else int(max_chunks)

    # dc.py:188-191
    seq_sorted_indices = _stable_partition_indices(boundary_mask)

    # dc.py:193-199
    gather_idx = seq_sorted_indices[:, :width, None]
    gather_idx = np.broadcast_to(
        gather_idx, (gather_idx.shape[0], gather_idx.shape[1], hidden_states.shape[-1])
    )
    next_hidden_states = np.take_along_axis(hidden_states, gather_idx, axis=1)

    # dc.py:201-204
    next_mask = np.arange(width)[None, :] < num_tokens[:, None]

    return next_hidden_states, next_mask


def dechunk_reference(
    inner_hidden_states: np.ndarray,
    boundary_prob: np.ndarray,
    boundary_mask: np.ndarray,
) -> np.ndarray:
    """`DeChunkLayer.forward` -- `dc.py:239-313`, padded branch, EMA per `dc.py:333`.

    Transcribed:

    - ``p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - 1e-4)``
      (`dc.py:256`). The clamp is on the FULL-resolution probabilities, before
      the gather.
    - The padded branch (`dc.py:262-273`) reuses ChunkLayer's stable partition to
      pull ``p`` into boundary order and keeps the first
      ``hidden_states.shape[1]`` columns -- i.e. the INNER (compressed) width::

          token_idx = arange(L)[None, :] + (~boundary_mask).long() * L
          seq_sorted_indices = torch.argsort(token_idx, dim=1)   # dc.py:265-269
          p = torch.gather(p, dim=1,
                           index=seq_sorted_indices[:, :hidden_states.shape[1]])
                                                                 # dc.py:271-273

    - **The recurrence.** `dc.py:275-294` computes the EMA by re-expressing it as
      a diagonal SSM and handing it to ``mamba_chunk_scan_combined``
      (``dt = log(1/(1-p))``, ``x = h/dt``, ``A = -1``, ``B = p``, ``C = 1``, so
      the discretized update is ``h_t = exp(-dt_t) h_{t-1} + dt_t B_t x_t =
      (1 - p_t) h_{t-1} + p_t h_t``). That is a throughput reformulation, not the
      definition. The unambiguous ground truth is the reference's own kernel-free
      ``step()`` (`dc.py:333`)::

          result = p * current_hidden_states + (1 - p) * inference_params.last_value

      transcribed here as ``h_t = p_t * x_t + (1 - p_t) * h_{t-1}`` with
      ``h_{-1} = 0`` (`dc.py:232-237`: the allocated ``DeChunkState.last_value``
      is ``torch.zeros``).

      The closed form ``A_t = prod(1-p_s)``, ``h_t = A_t * cumsum(p_s x_s / A_s)``
      is NOT used and must not be substituted for the loop: it returns ``nan``
      from ``L >= 64`` in the ``p = 1 - 1e-4`` regime that `dc.py:256`'s own
      clamp produces, in float64 as well as float32, because ``A_t = 1e-4t``
      underflows and ``1/A_s`` then overflows (measured, plan decision D-010(c)).
      This is the same formulation the Keras `DeChunkLayer` will implement, so
      the two agree by construction of the reference, not by coincidence.

    - Scatter back to full resolution (`dc.py:302-308`)::

          plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1
          out = torch.gather(out, dim=1,
                             index=plug_back_idx.unsqueeze(-1).expand(-1, -1, D))

      Every non-boundary position therefore repeats the most recent chunk value.

    Not transcribed: the packed branch (`dc.py:258-260`, `dc.py:296-301`), the
    bf16 cast (`dc.py:275-284`, `dc.py:313` -- this module is float64 by charter)
    and the ``inference_params`` write-back (`dc.py:310-311`).

    :param inner_hidden_states: ``(B, M, D)`` chunk-resolution hidden states.
    :type inner_hidden_states: numpy.ndarray
    :param boundary_prob: ``(B, L, 2)`` routing distribution, or ``(B, L)``
        holding ``p`` directly; ``[..., -1]`` is taken either way, matching
        `dc.py:256`.
    :type boundary_prob: numpy.ndarray
    :param boundary_mask: ``(B, L)`` boolean.
    :type boundary_mask: numpy.ndarray
    :return: ``(B, L, D)`` full-resolution output.
    :rtype: numpy.ndarray
    """
    inner_hidden_states = np.asarray(inner_hidden_states, dtype=np.float64)
    boundary_prob = np.asarray(boundary_prob, dtype=np.float64)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)

    # dc.py:256 -- boundary_prob[..., -1] works for both (B, L, 2) and (B, L).
    p_full = boundary_prob[..., -1] if boundary_prob.ndim == 3 else boundary_prob
    p_full = np.clip(p_full, P_CLAMP_MIN, P_CLAMP_MAX)

    inner_len = inner_hidden_states.shape[1]

    # dc.py:265-273
    seq_sorted_indices = _stable_partition_indices(boundary_mask)
    p = np.take_along_axis(p_full, seq_sorted_indices[:, :inner_len], axis=1)

    # dc.py:333 -- the explicit, kernel-free recurrence, h_{-1} = 0 (dc.py:232-237).
    out = np.zeros_like(inner_hidden_states)
    carry = np.zeros(
        (inner_hidden_states.shape[0], inner_hidden_states.shape[2]), dtype=np.float64
    )
    for t in range(inner_len):
        p_t = p[:, t, None]
        carry = p_t * inner_hidden_states[:, t, :] + (1.0 - p_t) * carry
        out[:, t, :] = carry

    # dc.py:302-308
    plug_back_idx = np.cumsum(boundary_mask.astype(np.int64), axis=1) - 1
    gather_idx = np.broadcast_to(
        plug_back_idx[..., None],
        (plug_back_idx.shape[0], plug_back_idx.shape[1], out.shape[-1]),
    )
    return np.take_along_axis(out, gather_idx, axis=1)


def ratio_loss_reference(
    boundary_prob: np.ndarray,
    boundary_mask: np.ndarray,
    target_ratio: float,
) -> float:
    """`load_balancing_loss` -- `train.py:13-40`, transcribed verbatim.

    ::

        boundary_prob = router_output.boundary_prob
        tokenized_prob = boundary_prob[..., -1]                   # train.py:30-31
        boundary_mask = router_output.boundary_mask               # train.py:32

        true_ratio = boundary_mask.float().mean()                 # train.py:34
        average_prob = tokenized_prob.float().mean()              # train.py:35

        return (
            (1 - true_ratio) * (1 - average_prob) +
            (true_ratio) * (average_prob) * (N-1)
        ) * N / (N-1)                                             # train.py:37-40

    Three properties of the transcription worth naming, because each is a way a
    re-derivation from the paper would differ from the code:

    - Both means are taken over the WHOLE tensor -- batch and sequence axes
      together, one scalar per call (`train.py:20-21` says so explicitly: the
      loss is computed per minibatch and then averaged over minibatches, not
      computed per example).
    - No padding mask is applied to either mean. Upstream passes the router
      output straight in; padded positions are counted.
    - ``N`` is the target downsampling factor and ``N > 1`` is required
      (`train.py:25`); ``N = 1`` divides by zero, and that is upstream's
      documented precondition, not an omission here.

    :param boundary_prob: ``(B, L, 2)`` routing distribution (``[..., -1]`` is
        ``p``), or ``(B, L)`` holding ``p`` directly.
    :type boundary_prob: numpy.ndarray
    :param boundary_mask: ``(B, L)`` boolean.
    :type boundary_mask: numpy.ndarray
    :param target_ratio: ``N``, the target downsampling factor; must be ``> 1``.
    :type target_ratio: float
    :return: The scalar load-balancing (ratio) loss.
    :rtype: float
    """
    boundary_prob = np.asarray(boundary_prob, dtype=np.float64)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    n = float(target_ratio)

    # train.py:30-31
    tokenized_prob = boundary_prob[..., -1] if boundary_prob.ndim == 3 else boundary_prob

    # train.py:34-35
    true_ratio = boundary_mask.astype(np.float64).mean()
    average_prob = tokenized_prob.mean()

    # train.py:37-40
    return float(
        ((1.0 - true_ratio) * (1.0 - average_prob) + true_ratio * average_prob * (n - 1.0))
        * n
        / (n - 1.0)
    )


# ---------------------------------------------------------------------------
# Tolerance derivation for float64 comparisons against hand arithmetic
# ---------------------------------------------------------------------------

_F64_EPS = float(np.finfo(np.float64).eps)   # 2.220446049250313e-16
_F64_U = _F64_EPS / 2.0                      # unit roundoff
_TAIL_FACTOR = 8.0                           # 8-sigma tail on the random-walk model


def hand_pin_atol(num_rounded_ops: int, scale: float) -> float:
    """Bound on ``|float64 evaluation - exact hand arithmetic|``.

    Interface contract: pure function, no state, never raises for
    ``num_rounded_ops >= 0``; returns a strictly positive float. Callers MUST
    pass ``rtol=0`` to ``np.testing.assert_allclose`` -- its default
    ``rtol=1e-7`` would contribute ~1e-7 of silent tolerance and make any bound
    derived here decorative.

    Derivation (so this is a bound, not a pasted magic number):

    1. Each rounded float64 operation contributes an error of order
       ``u * |partial result|`` with ``u = eps/2 = 1.11e-16``.
    2. Rounding errors are not adversarially aligned; over a chain of ``M``
       rounded operations they accumulate as a random walk, giving a relative
       error of order ``sqrt(M) * u``. Only ONE side of the comparison is
       charged here, because the other side is exact rational arithmetic done on
       paper -- that is the whole point of a hand pin, and it is what makes this
       different from `tests/numerics.reassociation_atol`, which charges both
       sides of a float32-vs-float32 comparison.
    3. Take an 8-sigma tail factor and scale by the output magnitude, since the
       error is relative::

           atol = 8 * sqrt(M) * u * max(1, |output|)

    Why this is not `tests/numerics.reassociation_atol`: that helper hardcodes
    the float32 unit roundoff (``_F32_U``), which is 8.4e+08 times larger than
    float64's. Reusing it here would produce a bound of order 1e-6 on values of
    order 1 -- twelve orders of magnitude above this path's actual noise, i.e. a
    tolerance that cannot fail. The two helpers answer different questions in
    different dtypes; this one is deliberately NOT a copy, and it stays local to
    the dynamic-chunking suite until a second consumer outside it appears.

    :param num_rounded_ops: Number of rounded float64 operations on the longest
        dependency chain of the compared quantity.
    :type num_rounded_ops: int
    :param scale: Magnitude of the compared output, ``max|expected|``.
    :type scale: float
    :return: Absolute tolerance.
    :rtype: float
    """
    return _TAIL_FACTOR * np.sqrt(float(num_rounded_ops)) * _F64_U * max(1.0, float(scale))
