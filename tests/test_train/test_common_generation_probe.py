"""Token-identity contract test for the common ``GenerationProbeCallback``.

This test PINS THE CONTRACT for the (not-yet-existing) unified
``GenerationProbeCallback`` that will live in ``src/train/common/generation_probe.py``.
It is intentionally written TEST-FIRST: until step 2 of the plan creates the
class, the import below raises ``ModuleNotFoundError`` and this whole module is
RED at collection time. That is the correct/expected state for step 1.

The contract this test encodes (the three load-bearing design points):

1. SEEDED RNG (reproducibility, deliberate improvement over copies A-D which
   used the global ``np.random.choice``). The common class takes ``seed`` and
   uses ``rng = np.random.default_rng(seed)`` for ``rng.choice(...)``. The
   reference loop below uses the SAME seeded generator so identity is exact.

2. PADDING-AGNOSTIC COMMON CLASS. The common class passes the UNPADDED ``ctx``
   slice (``ids[-ctx_length:]``) to the ``logits_fn`` closure, and consumes the
   returned ``float32[vocab]`` next-position vector directly. It never indexes a
   sequence position itself; the closure already returns the single
   next-position vector. (D-002)

3. THREE REPETITION-PENALTY MODES. ``repetition_penalty_mode`` selects:
     - ``"divide"``     : ``logits[t] /= penalty``                 (copies A/B)
     - ``"sign_aware"`` : ``logits[t] = l/p if l > 0 else l*p``    (copies C/D)
     - ``"multiply"``   : ``logits[t] *= penalty``  (log-prob domain, copy E)
   This test exercises ``"divide"`` (the canonical default) and ``"multiply"``
   (the routing log-prob adapter). ``"sign_aware"`` is covered by the reference
   implementation and asserted at the algorithm level via ``_apply_rep_penalty``.

All tests are CPU-only, < 2s, and require no tiktoken download, no GPU, and no
training. A stub identity encoder is injected via the ``encoder=`` constructor
kwarg (the testability contract step 2 must implement).
"""

from typing import Callable, List, Optional

import numpy as np

# RED until step 2 creates this module. ModuleNotFoundError here is the
# intended/correct failure for step 1 (test-first).
from train.common.generation_probe import GenerationProbeCallback


# --------------------------------------------------------------------------- #
# Test fixtures: tiny vocab, deterministic stub model, identity stub encoder.
# --------------------------------------------------------------------------- #

VOCAB = 64
CTX_LENGTH = 8
MAX_TOKENS = 6
EOT_ID = 0  # default end-of-text id for the canonical tests
# special_ids range mirrors the ground-truth algorithm:
#   range(vocab_base, max(vocab_base + 1, 50261))
# With VOCAB=64 < 50261, this range starts past the vocab so the
# `if sid < vocab` guard drops every special id -> no-op for the tiny vocab.
SPECIAL_IDS_START = VOCAB


class _StubEncoder:
    """Identity int-passthrough encoder.

    ``encode`` / ``decode`` are pass-throughs so no tiktoken model download is
    needed. Carries ``n_vocab`` and ``eot_token`` attributes exactly like a
    tiktoken ``Encoding`` so the common class can read them the same way it
    reads a real encoder.
    """

    def __init__(self, n_vocab: int = VOCAB, eot_token: int = EOT_ID):
        self.n_vocab = n_vocab
        self.eot_token = eot_token

    def encode(self, s: str) -> List[int]:
        if not s:
            return []
        return [int(tok) for tok in s.split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join(str(int(t)) for t in ids)


def _make_stub_model(value_matrix: np.ndarray) -> Callable:
    """Return a callable mimicking ``model(ctx, training=False)``.

    The returned object always emits the SAME per-position rows regardless of
    which context ids are passed (sliced to the input length), so the ONLY
    source of variation across runs is the seeded sampling RNG. This makes the
    identity assertions exact.

    ``value_matrix`` shape: ``[CTX_LENGTH, VOCAB]``. The closure reads
    ``[0, -1, :]`` -> the last row of the returned (sliced) block.
    """

    class _Stub:
        def __call__(self, ctx, training: bool = False):  # noqa: D401
            seq = int(np.asarray(ctx).shape[1])
            block = value_matrix[np.newaxis, :seq, :]
            return {"logits": block.astype("float32")}

    return _Stub()


# --------------------------------------------------------------------------- #
# Independent reference oracle. Implements the canonical generation algorithm
# from scratch (extracted from wave_field_llm/pretrain.py:245-295). It must NOT
# call any production code -- it is the ground truth the callback is asserted
# against.
# --------------------------------------------------------------------------- #


def _apply_rep_penalty(value: float, penalty: float, mode: str) -> float:
    """Repetition-penalty mode dispatch (the three documented modes)."""
    if mode == "divide":
        return value / penalty
    if mode == "sign_aware":
        return value / penalty if value > 0 else value * penalty
    if mode == "multiply":
        return value * penalty
    raise ValueError(f"unknown repetition_penalty_mode: {mode!r}")


def _reference_generate(
    prompt_ids: List[int],
    logits_provider: Callable[[np.ndarray], np.ndarray],
    *,
    vocab: int,
    ctx_length: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    mode: str,
    eot_id: int,
    special_ids_start: int,
    seed: int,
    stop_on_eot: bool,
) -> List[int]:
    """Oracle: reproduce the canonical per-token generation loop exactly."""
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    special_ids = list(range(special_ids_start, max(special_ids_start + 1, 50261)))

    generated = 0
    for _ in range(max_tokens):
        ctx = ids[-ctx_length:]
        logits = np.array(
            logits_provider(np.array([ctx], dtype="int32")), dtype="float32"
        ).copy()

        if not stop_on_eot:
            logits[eot_id] = -1e9
        for sid in special_ids:
            if sid < logits.shape[0]:
                logits[sid] = -1e9
        for t in set(ids[-50:]):
            if t == eot_id:
                continue
            logits[t] = _apply_rep_penalty(float(logits[t]), repetition_penalty, mode)

        logits = logits / temperature

        sorted_idx = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_idx]
        probs = np.exp(sorted_logits - sorted_logits[0])
        probs /= probs.sum()
        cutoff = int(np.searchsorted(np.cumsum(probs), top_p)) + 1
        top_idx = sorted_idx[:cutoff]
        top_probs = probs[:cutoff]
        top_probs /= top_probs.sum()

        next_token = int(top_idx[rng.choice(len(top_idx), p=top_probs)])
        ids.append(next_token)
        generated += 1

        if stop_on_eot and next_token == eot_id:
            break

    # Return only the freshly generated ids (after the prompt).
    return ids[len(prompt_ids):]


def _build_callback(
    logits_fn: Callable,
    *,
    seed: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.3,
    mode: str = "divide",
    stop_on_eot: bool = False,
    eot_token_id: int = EOT_ID,
    max_tokens: int = MAX_TOKENS,
    encoder: Optional[_StubEncoder] = None,
) -> GenerationProbeCallback:
    """Construct the common callback against the contract step 2 must honor.

    Constructor contract assumed here (step 2 MUST accept this signature):
      - ``logits_fn``               : Callable[[np.ndarray], np.ndarray] -> float32[vocab]
      - ``encoder=<obj>``           : pre-built encoder used INSTEAD of
                                      ``tiktoken.get_encoding(encoding_name)``.
                                      When provided, ``encoding_name`` is ignored.
      - ``ctx_length``              : unpadded slice length (``ids[-ctx_length:]``)
      - ``max_tokens``              : per-prompt generation budget
      - ``temperature`` / ``top_p`` : nucleus sampling controls
      - ``repetition_penalty``      : penalty magnitude
      - ``repetition_penalty_mode`` : one of "divide" | "sign_aware" | "multiply"
      - ``eot_token_id``            : end-of-text id
      - ``stop_on_eot``             : break on EOT vs suppress EOT (-1e9) + run full
      - ``seed``                    : seeds ``np.random.default_rng(seed)``
    """
    return GenerationProbeCallback(
        logits_fn=logits_fn,
        encoder=encoder if encoder is not None else _StubEncoder(),
        encoding_name=None,
        ctx_length=CTX_LENGTH,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        repetition_penalty_mode=mode,
        eot_token_id=eot_token_id,
        stop_on_eot=stop_on_eot,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_generation_matches_reference_divide_unpadded():
    """Canonical path: seed=42, divide-mode rep penalty, unpadded ctx.

    Asserts the common callback's raw generated token-id list is byte-identical
    to the independent reference loop under the SAME seeded RNG. Any
    position-index, sampling, or RNG divergence breaks this exact equality.
    """
    rng = np.random.default_rng(0)
    logits_matrix = rng.standard_normal((CTX_LENGTH, VOCAB)).astype("float32")
    model = _make_stub_model(logits_matrix)

    def logits_fn(ctx):
        return model(ctx, training=False)["logits"][0, -1, :].astype("float32")

    cb = _build_callback(
        logits_fn,
        seed=42,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        mode="divide",
    )

    prompt = [1, 2, 3]
    tokens_cb = cb._generate_token_ids(prompt)
    tokens_ref = _reference_generate(
        prompt,
        logits_fn,
        vocab=VOCAB,
        ctx_length=CTX_LENGTH,
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        mode="divide",
        eot_id=EOT_ID,
        special_ids_start=SPECIAL_IDS_START,
        seed=42,
        stop_on_eot=False,
    )

    assert tokens_cb == tokens_ref, (
        f"Token mismatch (divide/unpadded): callback={tokens_cb}, "
        f"reference={tokens_ref}. Indicates a position-index, sampling, or "
        "seeded-RNG divergence."
    )


def test_routing_prob_adapter_multiply():
    """Routing path: prob->log adapter + multiply-mode rep penalty.

    The stub model returns a PROBABILITY simplex per row. The caller closure
    applies ``np.log(np.clip(probs, 1e-12, 1.0))`` BEFORE the callback samples
    (the log-prob domain of copy E). The reference applies the SAME adapter to
    the SAME prob matrix, then runs the multiply-mode loop. Token-id equality
    locks the prob->log adapter + multiply mode.
    """
    rng = np.random.default_rng(1)
    probs_matrix = rng.dirichlet(np.ones(VOCAB), size=CTX_LENGTH).astype("float32")
    model = _make_stub_model(probs_matrix)

    def logits_fn(ctx):
        probs = model(ctx, training=False)["logits"][0, -1, :]
        return np.log(np.clip(probs, 1e-12, 1.0)).astype("float32")

    cb = _build_callback(
        logits_fn,
        seed=7,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        mode="multiply",
    )

    prompt = [5, 6]
    tokens_cb = cb._generate_token_ids(prompt)
    tokens_ref = _reference_generate(
        prompt,
        logits_fn,
        vocab=VOCAB,
        ctx_length=CTX_LENGTH,
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        mode="multiply",
        eot_id=EOT_ID,
        special_ids_start=SPECIAL_IDS_START,
        seed=7,
        stop_on_eot=False,
    )

    assert tokens_cb == tokens_ref, (
        f"Token mismatch (multiply/routing): callback={tokens_cb}, "
        f"reference={tokens_ref}. Indicates the prob->log adapter or the "
        "multiply rep-penalty mode diverged."
    )


def test_stop_on_eot():
    """``stop_on_eot=True`` terminates early when EOT is sampled.

    Make EOT's logit dominate so it is selected on the first token, then assert:
      - with ``stop_on_eot=True``: generation stops EARLY (len < max_tokens)
        and EOT appears in the output.
      - with ``stop_on_eot=False``: EOT is suppressed (-1e9) and the loop runs
        the full ``max_tokens`` budget (EOT never appears).
    """
    eot = 3
    big_max_tokens = 20

    logits_matrix = np.full((CTX_LENGTH, VOCAB), -100.0, dtype="float32")
    logits_matrix[:, eot] = 100.0
    model = _make_stub_model(logits_matrix)

    def logits_fn(ctx):
        return model(ctx, training=False)["logits"][0, -1, :].astype("float32")

    # stop_on_eot=True -> should break early with EOT present.
    cb_stop = _build_callback(
        logits_fn,
        seed=0,
        mode="divide",
        stop_on_eot=True,
        eot_token_id=eot,
        max_tokens=big_max_tokens,
    )
    tokens_stop = cb_stop._generate_token_ids([1])
    assert eot in tokens_stop, "EOT should be sampled when its logit dominates."
    assert len(tokens_stop) < big_max_tokens, (
        f"stop_on_eot=True must terminate before max_tokens; "
        f"got {len(tokens_stop)} tokens."
    )

    # stop_on_eot=False -> EOT suppressed, loop runs full budget.
    cb_full = _build_callback(
        logits_fn,
        seed=0,
        mode="divide",
        stop_on_eot=False,
        eot_token_id=eot,
        max_tokens=big_max_tokens,
    )
    tokens_full = cb_full._generate_token_ids([1])
    assert len(tokens_full) == big_max_tokens, (
        f"stop_on_eot=False must run the full max_tokens budget; "
        f"got {len(tokens_full)} tokens."
    )
    assert eot not in tokens_full, (
        "EOT must be suppressed (-1e9) when stop_on_eot=False."
    )
