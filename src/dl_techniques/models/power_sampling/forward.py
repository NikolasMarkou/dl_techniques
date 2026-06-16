"""Model-call indirection: turn ANY causal LM/VLM into a :data:`LogitsFn`.

The power sampler never calls a concrete model directly. Instead it is driven by
a closure that maps a token-id sequence to a single ``float32[V]`` logit vector
for the target position. This module builds those closures, generalizing the
two hardcoded model call-sites in the original CliffordNet implementation
(``cliffordnet/power_sampling.py`` ``_forward`` / ``_forward_batch``), which
assumed:

- the model output is a dict keyed ``"logits"``,
- a fixed ``ctx_len`` with right-padding by a known ``pad_id``,
- last-token gather via ``tf.gather_nd``.

Here all three are parameterized:

- ``logits_key`` selects the output tensor (or ``None`` for a plain-tensor model);
- ``ctx_len``/``pad_id`` are optional — ``ctx_len is None`` means a
  variable-length forward pass with no padding (G1);
- the last-token gather is pure numpy (constraint C3 — **no** ``tf.gather_nd``,
  **no** ``import tensorflow``).

For vision-language models, :class:`VLMForwardAdapter` binds a fixed image (or
other extra inputs) and a ``text_slice_start`` offset so the sampler can drive
the text suffix while the image prefix stays fixed.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from dl_techniques.models.power_sampling.protocols import LogitsFn
from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Small coercion / extraction helpers
# ---------------------------------------------------------------------------
def _to_numpy(x: Any) -> np.ndarray:
    """Coerce a model output tensor to a numpy array.

    The model is expected to return ``.numpy()``-able eager tensors (Keras 3
    eager execution). A plain numpy array or list is also accepted.

    :param x: A tensor-like object (eager tensor, numpy array, or sequence).
    :return: The value as a :class:`numpy.ndarray`.
    """
    numpy_method = getattr(x, "numpy", None)
    if callable(numpy_method):
        return numpy_method()
    return np.asarray(x)


def _extract_logits(out: Any, logits_key: Optional[str]) -> np.ndarray:
    """Pull the rank-3 ``(B, T, V)`` logits tensor out of a model output.

    :param out: The raw model output. When ``logits_key`` is set and ``out`` is
        a mapping, ``out[logits_key]`` is used; otherwise ``out`` is treated as
        the logits tensor directly (plain-tensor models, constraint C2).
    :param logits_key: Key into a dict-like output, or ``None`` for a bare tensor.
    :return: The logits as a ``(B, T, V)`` :class:`numpy.ndarray`.
    :raises ValueError: If ``logits_key`` is set but missing from a mapping
        output (named, with the available keys), or if the resulting tensor is
        not rank 3.
    """
    is_mapping = isinstance(out, dict) or (
        hasattr(out, "__getitem__") and hasattr(out, "keys")
    )

    if logits_key is not None and is_mapping:
        try:
            available = list(out.keys())
        except Exception:  # pragma: no cover - defensive
            available = None
        if logits_key not in (available if available is not None else out):
            raise ValueError(
                f"logits_key={logits_key!r} not found in model output; "
                f"available keys: {available}"
            )
        logits = _to_numpy(out[logits_key])
    else:
        # logits_key is None, or out is not a mapping -> treat out as the tensor.
        logits = _to_numpy(out)

    if logits.ndim != 3:
        raise ValueError(
            "Expected model logits of rank 3 (B, T, V), got shape "
            f"{logits.shape} (ndim={logits.ndim}). Check logits_key="
            f"{logits_key!r} and the model output contract."
        )
    return logits


# ---------------------------------------------------------------------------
# Single-sequence forward closure
# ---------------------------------------------------------------------------
def make_logits_fn(
    model: Any,
    ctx_len: Optional[int] = None,
    pad_id: Optional[int] = None,
    logits_key: Optional[str] = "logits",
    position: int = -1,
    text_slice_start: int = 0,
    extra_inputs: Optional[Dict[str, Any]] = None,
    token_key: str = "text_tokens",
) -> LogitsFn:
    """Build a single-sequence :data:`LogitsFn` closure for ``model``.

    The returned closure maps one token-id sequence (``List[int]`` or
    ``int32[T]`` / ``int32[1, T]`` array) to a ``float32[V]`` logit vector for a
    single target position. It generalizes the source ``_forward``
    (``cliffordnet/power_sampling.py:128-145``).

    :param model: Any callable accepting either an ``int32[1, T]`` array (when
        ``extra_inputs is None``) or a dict input (when ``extra_inputs`` is set,
        for VLMs) and returning logits.
    :param ctx_len: Fixed context length. If ``None`` (default), the sequence is
        passed through unpadded (variable-length forward, G1). If set, the last
        ``ctx_len`` tokens are kept and right-padded with ``pad_id``.
    :param pad_id: Padding token id. **Required** when ``ctx_len`` is not ``None``.
    :param logits_key: Key into a dict-like model output, or ``None`` to treat
        the output as a bare logits tensor (constraint C2).
    :param position: Target position. ``-1`` (default) selects the last *real*
        token; any other value selects that fixed position. Combined with
        ``text_slice_start`` for the final gather index.
    :param text_slice_start: Offset added to the gather index, used by VLMs to
        skip the vision-token prefix (the text suffix starts at this index).
    :param extra_inputs: Optional dict of extra model inputs (e.g.
        ``{"images": <array>}`` for a VLM). When provided, the model is called
        with ``{**extra_inputs, token_key: arr}``; otherwise with the bare array.
    :param token_key: Dict key under which the token array is placed when
        ``extra_inputs`` is provided. Defaults to ``"text_tokens"`` (nano_vlm).
    :return: A :data:`LogitsFn` closure.
    :raises ValueError: If ``ctx_len`` is set but ``pad_id`` is ``None``.
    """
    if ctx_len is not None and pad_id is None:
        raise ValueError(
            "pad_id is required when ctx_len is not None "
            f"(ctx_len={ctx_len}); cannot right-pad without a pad token id."
        )

    def fn(token_ids: Sequence[int]) -> np.ndarray:
        ids = [int(t) for t in np.asarray(token_ids).reshape(-1)]

        if ctx_len is not None:
            ctx = ids[-ctx_len:]
            real = len(ctx)
            padded = ctx + [pad_id] * (ctx_len - real)
        else:
            padded = ids
            real = len(ids)

        arr = np.array([padded], dtype="int32")  # (1, T_text)

        if extra_inputs is None:
            out = model(arr, training=False)
        else:
            out = model({**extra_inputs, token_key: arr}, training=False)

        logits = _extract_logits(out, logits_key)  # (1, T, V)
        # T is the FULL output sequence length (vision prefix + text); for a
        # VLM it exceeds the text token-array length by text_slice_start.
        T = logits.shape[1]

        if position == -1:
            gather_idx = text_slice_start + (real - 1)
        else:
            gather_idx = text_slice_start + position

        # DECISION plan_2026-06-16_535b4f02/D-001: last-token gather is pure
        # numpy indexing on the host-side array (constraint C3). Do NOT
        # reintroduce tf.gather_nd or a top-level `import tensorflow` — the
        # model returns .numpy()-able eager tensors, so we gather after
        # _to_numpy. See decisions.md D-001.
        if not (0 <= gather_idx < T):
            raise ValueError(
                f"gather_idx={gather_idx} out of range for output sequence "
                f"length T={T} (text_slice_start={text_slice_start}, "
                f"real={real}, position={position}). For VLMs check "
                "text_slice_start (vision-token offset)."
            )

        return logits[0, gather_idx].astype("float32")  # (V,)

    return fn


# ---------------------------------------------------------------------------
# Batched forward closure
# ---------------------------------------------------------------------------
def make_batch_logits_fn(
    model: Any,
    ctx_len: Optional[int] = None,
    pad_id: Optional[int] = None,
    logits_key: Optional[str] = "logits",
    position: int = -1,
    text_slice_start: int = 0,
    extra_inputs: Optional[Dict[str, Any]] = None,
    token_key: str = "text_tokens",
) -> Callable[[List[List[int]]], np.ndarray]:
    """Build a batched logits closure mirroring source ``_forward_batch``.

    The returned closure maps a batch of token-id sequences to a
    ``float32[B, V]`` array, running the whole batch through the model in a
    single call (high GPU utilization). It generalizes the source
    ``_forward_batch`` (``cliffordnet/power_sampling.py:147-177``), replacing
    ``tf.gather_nd`` with numpy fancy indexing.

    :param model: Any callable; see :func:`make_logits_fn`.
    :param ctx_len: Fixed context length, or ``None`` for variable-length.
        When ``None`` and the batch contains prefixes of differing lengths,
        sequences are right-padded to the batch maximum with ``pad_id`` and each
        sequence is gathered at its own real length.
    :param pad_id: Padding token id. **Required** when ``ctx_len`` is set, and
        also required for a variable-length batch with unequal prefix lengths.
    :param logits_key: Key into a dict-like model output, or ``None``.
    :param position: Target position; ``-1`` selects each sequence's last real
        token, else a fixed position.
    :param text_slice_start: Offset added to every gather index (VLM prefix).
    :param extra_inputs: Optional extra model inputs (VLM); see
        :func:`make_logits_fn`.
    :param token_key: Dict key for the token batch when ``extra_inputs`` is set.
    :return: A closure ``fn(batch_token_ids) -> float32[B, V]``.
    :raises ValueError: If padding is needed but ``pad_id`` is ``None``.
    """
    if ctx_len is not None and pad_id is None:
        raise ValueError(
            "pad_id is required when ctx_len is not None "
            f"(ctx_len={ctx_len}); cannot right-pad without a pad token id."
        )

    def fn(batch_token_ids: List[List[int]]) -> np.ndarray:
        B = len(batch_token_ids)
        seqs = [
            [int(t) for t in np.asarray(ids).reshape(-1)]
            for ids in batch_token_ids
        ]

        real_lens: List[int] = []
        if ctx_len is not None:
            batch_ctx = []
            for ids in seqs:
                ctx = ids[-ctx_len:]
                real = len(ctx)
                batch_ctx.append(ctx + [pad_id] * (ctx_len - real))
                real_lens.append(real)
        else:
            real_lens = [len(ids) for ids in seqs]
            max_len = max(real_lens) if real_lens else 0
            needs_pad = any(r != max_len for r in real_lens)
            if needs_pad and pad_id is None:
                raise ValueError(
                    "pad_id is required for a variable-length batch with "
                    f"unequal prefix lengths (lengths={real_lens})."
                )
            batch_ctx = [ids + [pad_id] * (max_len - len(ids)) for ids in seqs]

        batch_input = np.array(batch_ctx, dtype="int32")  # (B, T_text)

        if extra_inputs is None:
            out = model(batch_input, training=False)
        else:
            out = model({**extra_inputs, token_key: batch_input}, training=False)

        logits = _extract_logits(out, logits_key)  # (B, T, V)
        # T is the FULL output sequence length (vision prefix + text).
        T = logits.shape[1]

        # DECISION plan_2026-06-16_535b4f02/D-001: numpy fancy indexing replaces
        # tf.gather_nd (source:177) — constraint C3. Do NOT reintroduce
        # tf.gather_nd / tensorflow. See decisions.md D-001.
        idx = np.array(
            [
                text_slice_start + (real_lens[i] - 1 if position == -1 else position)
                for i in range(B)
            ]
        )
        if np.any(idx < 0) or np.any(idx >= T):
            raise ValueError(
                f"batch gather indices {idx.tolist()} out of range for "
                f"sequence length T={T} (text_slice_start={text_slice_start}, "
                f"real_lens={real_lens}, position={position})."
            )

        return logits[np.arange(B), idx].astype("float32")  # (B, V)

    return fn


# ---------------------------------------------------------------------------
# VLM adapter
# ---------------------------------------------------------------------------
class VLMForwardAdapter:
    """Bridges a dict-input VLM to the :data:`LogitsFn` interface.

    Holds a VLM model, a fixed image/extra-inputs tensor, and the vision-token
    offset (``text_slice_start``) so power sampling can drive the
    text-generation suffix while the image prefix stays fixed.

    NOTE: ``text_slice_start`` (== vision sequence length) is the caller's
    responsibility; repo VLMs (e.g. nano_vlm) do not expose it as a property.
    (user decision 2 / plan Failure Modes "VLM adapter".)
    """

    def __init__(
        self,
        model: Any,
        image: Any,
        *,
        image_key: str = "images",
        token_key: str = "text_tokens",
        text_slice_start: int,
        logits_key: Optional[str] = "logits",
        ctx_len: Optional[int] = None,
        pad_id: Optional[int] = None,
    ) -> None:
        """Construct a VLM adapter.

        :param model: The VLM, called with a dict input
            ``{image_key: image, token_key: <int32[B,T]>}``.
        :param image: The fixed image (or extra-inputs) tensor bound for every
            forward pass.
        :param image_key: Dict key for ``image`` in the model input.
        :param token_key: Dict key for the token array in the model input.
        :param text_slice_start: Vision-token offset; the text suffix begins at
            this index. Caller's responsibility (user decision 2).
        :param logits_key: Key into the model's dict output, or ``None``.
        :param ctx_len: Optional fixed text context length.
        :param pad_id: Padding token id (required when ``ctx_len`` is set).
        """
        self.model = model
        self.image = image
        self.image_key = image_key
        self.token_key = token_key
        self.text_slice_start = text_slice_start
        self.logits_key = logits_key
        self.ctx_len = ctx_len
        self.pad_id = pad_id

    def as_logits_fn(self) -> LogitsFn:
        """Return a single-sequence :data:`LogitsFn` bound to the fixed image."""
        return make_logits_fn(
            self.model,
            ctx_len=self.ctx_len,
            pad_id=self.pad_id,
            logits_key=self.logits_key,
            text_slice_start=self.text_slice_start,
            extra_inputs={self.image_key: self.image},
            token_key=self.token_key,
        )

    def as_batch_logits_fn(self) -> Callable[[List[List[int]]], np.ndarray]:
        """Return a batched logits closure bound to the fixed image."""
        return make_batch_logits_fn(
            self.model,
            ctx_len=self.ctx_len,
            pad_id=self.pad_id,
            logits_key=self.logits_key,
            text_slice_start=self.text_slice_start,
            extra_inputs={self.image_key: self.image},
            token_key=self.token_key,
        )


__all__ = ["make_logits_fn", "make_batch_logits_fn", "VLMForwardAdapter"]
