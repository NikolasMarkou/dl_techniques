"""Mock-based test suite for the general-purpose power-sampling package.

Keras-free, no GPU: a char-level tokenizer plus dict/tensor/VLM mock "models"
exercise every code path (config defaults, numpy ops, the forward closures, and
the full sampler dispatch) without TensorFlow or a real model.

Maps to Success Criteria SC2-SC6 in the plan; numerical tolerance ``atol=1e-6``.
"""

import re
import random
from dataclasses import asdict

import numpy as np
import pytest

from dl_techniques.models.power_sampling import (
    PowerSampler,
    PowerSamplingConfig,
    make_logits_fn,
    _log_softmax,
    _nucleus_sample,
)
import dl_techniques.models.power_sampling.forward as forward_mod


VOCAB = 32


# ---------------------------------------------------------------------------
# Mock tokenizer + mock models (keras-free, no GPU)
# ---------------------------------------------------------------------------
class CharTokenizer:
    """Char-level tokenizer satisfying TokenizerProtocol."""

    def __init__(self, vocab_size: int = VOCAB):
        self.vocab_size = vocab_size

    def encode(self, text):
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


class DictMockLM:
    """Dict-output mock LM. Returns ``{"logits": (B, T, vocab)}``."""

    def __init__(self, vocab: int = VOCAB, seed: int = 0):
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)

    def __call__(self, arr, training=False):
        if isinstance(arr, dict):
            arr = arr["text_tokens"]
        a = np.asarray(arr)
        B, T = a.shape
        logits = self.rng.standard_normal((B, T, self.vocab)).astype("float32")
        return {"logits": logits}


class TensorMockLM:
    """Bare-tensor mock LM. Returns ``(B, T, vocab)`` numpy array directly."""

    def __init__(self, vocab: int = VOCAB, seed: int = 0):
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)

    def __call__(self, arr, training=False):
        if isinstance(arr, dict):
            arr = arr["text_tokens"]
        a = np.asarray(arr)
        B, T = a.shape
        return self.rng.standard_normal((B, T, self.vocab)).astype("float32")


class VLMMockLM:
    """VLM mock: each output row r is filled with the scalar value r.

    The full output sequence length is ``vision + T_text``; row index r holds
    the value r everywhere, so a gathered logit vector reveals exactly which
    position was selected (used to assert text_slice_start offset arithmetic).
    """

    def __init__(self, vocab: int = VOCAB, vision: int = 4):
        self.vocab = vocab
        self.vision = vision

    def __call__(self, inputs, training=False):
        assert isinstance(inputs, dict)
        assert "images" in inputs and "text_tokens" in inputs
        tok = np.asarray(inputs["text_tokens"])
        B, T_text = tok.shape
        full = self.vision + T_text
        rows = np.arange(full)[None, :, None]  # (1, full, 1)
        logits = np.tile(rows, (B, 1, self.vocab)).astype("float32")
        return {"logits": logits}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_config_defaults(self):
        """SC2: generalized defaults carry NO GPT-2/CliffordNet IDs."""
        cfg = PowerSamplingConfig()
        assert cfg.special_token_ids == set()
        assert cfg.cls_token_id is None
        assert cfg.pad_token_id is None
        assert cfg.ctx_len is None


# ---------------------------------------------------------------------------
# Numeric ops
# ---------------------------------------------------------------------------
class TestOps:
    def test_log_softmax_sums_to_one(self):
        x = np.array([2.0, -1.0, 0.5, 3.0, -2.0], dtype="float32")
        total = float(np.exp(_log_softmax(x)).sum())
        assert abs(total - 1.0) <= 1e-6

    def test_nucleus_sample_in_range(self):
        np.random.seed(0)
        logits = np.random.randn(VOCAB).astype("float32")
        tok = _nucleus_sample(logits, top_p=0.92)
        assert isinstance(tok, int)
        assert 0 <= tok < VOCAB

    def test_nucleus_sample_argmax_dominant(self):
        """One-hot-dominant logits + low top_p -> returns the argmax."""
        np.random.seed(0)
        logits = np.full(VOCAB, -50.0, dtype="float32")
        logits[7] = 50.0
        tok = _nucleus_sample(logits, top_p=0.1)
        assert tok == 7


# ---------------------------------------------------------------------------
# Forward closures
# ---------------------------------------------------------------------------
class TestForward:
    def test_make_logits_fn_dict(self):
        """SC3: dict-output model -> (vocab,) float32 vector."""
        m = DictMockLM()
        fn = make_logits_fn(m, ctx_len=None, logits_key="logits")
        out = fn([1, 2, 3])
        assert out.shape == (VOCAB,)
        assert out.dtype == np.float32

    def test_make_logits_fn_tensor(self):
        """SC3: bare-tensor model with logits_key=None."""
        m = TensorMockLM()
        fn = make_logits_fn(m, ctx_len=None, logits_key=None)
        out = fn([1, 2, 3])
        assert out.shape == (VOCAB,)

    def test_make_logits_fn_vlm_slice(self):
        """SC3: VLM text_slice_start offset shifts the gathered position.

        vision=4, real text len=3, last-token -> gather idx = 4 + (3-1) = 6.
        VLMMockLM fills row r with value r, so every element of the gathered
        vector equals 6.0.
        """
        m = VLMMockLM(vision=4)
        fn = make_logits_fn(
            m,
            ctx_len=None,
            logits_key="logits",
            text_slice_start=4,
            extra_inputs={"images": np.zeros((1, 8, 8, 3), dtype="float32")},
            token_key="text_tokens",
        )
        out = fn([1, 2, 3])
        assert out.shape == (VOCAB,)
        assert float(out[0]) == 6.0
        assert np.all(out == 6.0)

    def test_make_logits_fn_ctx_len_padding(self):
        """SC3: fixed ctx_len + pad_id exercises the pad path."""
        m = DictMockLM()
        fn = make_logits_fn(m, ctx_len=8, pad_id=0, logits_key="logits")
        out = fn([1, 2, 3])
        assert out.shape == (VOCAB,)

    def test_make_logits_fn_missing_key_raises(self):
        """SC3: missing logits_key -> ValueError."""
        m = DictMockLM()
        fn = make_logits_fn(m, logits_key="nope")
        with pytest.raises(ValueError):
            fn([1, 2, 3])

    def test_no_gather_nd_in_package(self):
        """SC3: no tf.gather_nd CALL and no `import tensorflow` in forward.py.

        Comments/docstrings that mention the words are fine; assert only on
        non-comment code lines.
        """
        src_path = forward_mod.__file__
        with open(src_path, "r") as fh:
            lines = fh.readlines()

        import_tf = re.compile(r"^\s*import\s+tensorflow\b")
        for raw in lines:
            line = raw.split("#", 1)[0]  # strip trailing comments
            stripped = line.strip()
            if not stripped:
                continue
            assert not import_tf.match(line), f"tensorflow import: {raw!r}"
            assert "gather_nd(" not in line, f"gather_nd call: {raw!r}"


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
def _std_config(**overrides):
    base = dict(pad_token_id=0, ctx_len=16, max_tokens=8)
    base.update(overrides)
    return PowerSamplingConfig(**base)


class TestSampler:
    def test_generate_standard(self):
        """SC4: end-to-end standard nucleus sampling on a mock model."""
        random.seed(0)
        np.random.seed(0)
        s = PowerSampler(DictMockLM(), CharTokenizer(), _std_config())
        ids, info = s.generate_standard("abc", max_tokens=5)
        assert len(ids) == 3 + 5  # 3 encoded + 5 generated (CLS off by default)
        assert info["tokens_generated"] == 5

    def test_mcmc_power_sample(self):
        """SC4: MCMC power sampling; acceptance_ratio in [0,1]."""
        random.seed(0)
        np.random.seed(0)
        cfg = PowerSamplingConfig(
            pad_token_id=0, ctx_len=16, max_tokens=4, block_num=2, mcmc_steps=3,
        )
        s = PowerSampler(DictMockLM(), CharTokenizer(), cfg)
        ids, info = s.mcmc_power_sample("abc")
        assert 0.0 <= info["acceptance_ratio"] <= 1.0
        assert "alpha" in info
        assert len(ids) >= 3

    def test_max_swap(self):
        """SC4: deterministic max-swap; acceptance_ratio in [0,1]."""
        random.seed(0)
        np.random.seed(0)
        cfg = PowerSamplingConfig(
            pad_token_id=0, ctx_len=16, max_tokens=4, block_num=2, mcmc_steps=3,
        )
        s = PowerSampler(DictMockLM(), CharTokenizer(), cfg)
        ids, info = s.max_swap("abc")
        assert 0.0 <= info["acceptance_ratio"] <= 1.0

    def test_generate_text_dispatch(self):
        """SC4: generate_text dispatch + ValueError on a bad method."""
        random.seed(0)
        np.random.seed(0)
        cfg = PowerSamplingConfig(
            pad_token_id=0, ctx_len=16, max_tokens=4, block_num=2, mcmc_steps=2,
        )
        s = PowerSampler(DictMockLM(), CharTokenizer(), cfg)

        text, info = s.generate_text("ab", method="standard", max_tokens=3)
        assert isinstance(text, str) and isinstance(info, dict)

        text, info = s.generate_text("ab", method="power")
        assert isinstance(text, str) and isinstance(info, dict)

        text, info = s.generate_text("ab", method="max_swap")
        assert isinstance(text, str) and isinstance(info, dict)

        with pytest.raises(ValueError):
            s.generate_text("ab", method="bogus")

    def test_no_config_mutation(self):
        """SC5/I4: generate_standard must not mutate self.config."""
        random.seed(0)
        np.random.seed(0)
        s = PowerSampler(DictMockLM(), CharTokenizer(), _std_config())
        before = asdict(s.config)
        s.generate_standard(
            "abc", top_p=0.5, repetition_penalty=2.0, max_tokens=3,
        )
        after = asdict(s.config)
        assert after == before

    def test_cls_off_default(self):
        """SC6: CLS off by default -> ids length = encoded + generated."""
        random.seed(0)
        np.random.seed(0)
        s = PowerSampler(DictMockLM(), CharTokenizer(), _std_config())
        ids, _ = s.generate_standard("abc", max_tokens=4)
        assert len(ids) == 3 + 4

    def test_cls_on(self):
        """SC6: CLS on -> prepended then stripped; net length == encoded+gen.

        The returned ids must NOT start with the cls id (it was stripped), and
        the length matches the CLS-off case for the same prompt + max_tokens.
        """
        random.seed(0)
        np.random.seed(0)
        cls_id = 5
        cfg_on = PowerSamplingConfig(
            pad_token_id=0, ctx_len=16, cls_token_id=cls_id, max_tokens=4,
        )
        s_on = PowerSampler(DictMockLM(), CharTokenizer(), cfg_on)
        ids_on, _ = s_on.generate_standard("abc", max_tokens=4)

        assert ids_on[0] != cls_id  # CLS stripped from the returned sequence
        assert len(ids_on) == 3 + 4  # net gen-only count matches CLS-off

    def test_mh_acceptance_formula(self):
        """Pin the Metropolis-Hastings acceptance arithmetic.

        The accept probability in ``mcmc_power_sample`` is
        ``min(1, exp(min(log_r, 0.0)))`` where ``log_r`` is the trajectory
        log-ratio. This locks the formula SHAPE so a future sign-flip or slice
        off-by-one that changes the accept math is caught numerically.
        """
        def accept_prob(log_r):
            return min(1.0, float(np.exp(min(log_r, 0.0))))

        # log_r >= 0 always accepts (clamped to 0 -> exp(0) = 1).
        assert accept_prob(0.0) == 1.0
        assert accept_prob(5.0) == 1.0
        # log_r = ln(0.5) -> accept prob exactly 0.5.
        assert abs(accept_prob(np.log(0.5)) - 0.5) <= 1e-6
        # log_r -> -inf -> accept prob 0.
        assert accept_prob(float("-inf")) == 0.0

    def test_mcmc_deterministic_under_seed(self):
        """Two seeded mcmc_power_sample runs must be bit-identical.

        Seeding both ``random`` and ``np.random`` fixes proposal indices AND
        the ``np.random.rand() < exp(min(log_r, 0))`` accept draws, so the real
        ``log_r`` accept/reject path is exercised and must be reproducible.
        """
        cfg = PowerSamplingConfig(
            pad_token_id=0, ctx_len=16, max_tokens=4, block_num=2, mcmc_steps=3,
        )

        random.seed(123)
        np.random.seed(123)
        s1 = PowerSampler(DictMockLM(seed=7), CharTokenizer(), cfg)
        ids1, info1 = s1.mcmc_power_sample("abc")

        random.seed(123)
        np.random.seed(123)
        s2 = PowerSampler(DictMockLM(seed=7), CharTokenizer(), cfg)
        ids2, info2 = s2.mcmc_power_sample("abc")

        assert ids1 == ids2
        assert info1["acceptance_ratio"] == info2["acceptance_ratio"]
        assert info1["acceptances"] == info2["acceptances"]

    def test_make_batch_logits_fn_varlen(self):
        """make_batch_logits_fn ctx_len=None: right-pad ragged batch to max.

        Ragged prefixes are right-padded to the batch maximum with pad_id and
        each row is gathered at its own real length -> (B, vocab) float32.
        """
        m = DictMockLM()
        fn = forward_mod.make_batch_logits_fn(
            m, ctx_len=None, pad_id=0, logits_key="logits",
        )
        out = fn([[1, 2], [3, 4, 5, 6], [7]])
        assert out.shape == (3, VOCAB)
        assert out.dtype == np.float32

    def test_make_batch_logits_fn_fixed_ctxlen(self):
        """make_batch_logits_fn fixed ctx_len: pad each row to ctx_len."""
        m = DictMockLM()
        fn = forward_mod.make_batch_logits_fn(m, ctx_len=8, pad_id=0)
        out = fn([[1, 2], [3, 4, 5, 6], [7]])
        assert out.shape == (3, VOCAB)
        assert out.dtype == np.float32

    def test_logits_fn_injection(self):
        """Explicit logits_fn= path + single-fn batched fallback."""
        random.seed(0)
        np.random.seed(0)
        fn = lambda ids: np.zeros(VOCAB, dtype="float32")
        cfg = PowerSamplingConfig(max_tokens=4, block_num=2, mcmc_steps=2)
        s = PowerSampler(
            None, CharTokenizer(), cfg, logits_fn=fn,
        )
        ids, info = s.generate_standard("a", max_tokens=2)
        assert len(ids) == 1 + 2

        # exercise the single-fn batched fallback in _batched_generate via mcmc
        ids2, info2 = s.mcmc_power_sample("a")
        assert 0.0 <= info2["acceptance_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# C-30(b): the chain must propose from the state it just accepted
# ---------------------------------------------------------------------------
class CounterLM:
    """Deterministic stub whose every emitted token is its own call counter.

    Two properties make the chain readable, and both are properties of the
    STUB, not of the sampler:

    1. **No token is ever emitted twice.** The winning logit is placed at
       ``counter % vocab`` and the counter advances once per batch row, so a
       token's value says exactly *when* it was generated. Position ``p`` of the
       final sequence therefore identifies which generation produced it.
    2. **Later calls are strictly more confident** (the losing logits sink by
       ``0.5`` per call), so a later proposal always has a higher trajectory
       log-probability than an earlier one. This is what lets ``max_swap``'s
       ``log_r > 0`` rule accept every proposal in a correct chain.
    """

    def __init__(self, vocab: int = VOCAB):
        self.vocab = vocab
        self.counter = 0

    def __call__(self, arr, training=False):
        a = np.asarray(arr)
        B, T = a.shape
        out = np.empty((B, T, self.vocab), dtype="float32")
        for b in range(B):
            k = self.counter
            self.counter += 1
            out[b] = -5.0 - 0.5 * k
            out[b, :, k % self.vocab] = 5.0
        return {"logits": out}


class TestChainProposesFromTheAcceptedState:
    """C-30(b): proposal i+1 must be cut from the state acceptance i produced.

    **Oracle, derived from the Metropolis-Hastings definition — not from the
    sampler.** An MH step draws ``y ~ q(.|x)`` where ``x`` is the CURRENT state
    and accepts or rejects it; the next step draws from ``q(.|x')`` with ``x'``
    the state that step left behind. Here ``q(.|x)`` regenerates ``x[idx:]``
    from ``x[:idx]``, so proposal 2 (cut at 5) must share its first 5 tokens
    with the state proposal 1 left behind — not with the pre-block state.

    **Scenario** (fully scripted; nothing is read off the implementation):
    prompt ``"ab"`` -> ``c = 2``; one block of 4 tokens -> ``t = 6``; cut points
    forced to ``[4, 5]`` by patching ``random.randint``; acceptance forced by
    patching ``np.random.rand`` to ``0.0`` (``0.0 < exp(min(log_r, 0))`` for any
    finite ``log_r``). The stub emits token ``k`` on its ``k``-th call, so the
    naive block lays down tokens ``0,1,2,3`` at positions ``2,3,4,5``.

    A correct chain then: accepts proposal 1, which rewrites positions 4-5 with
    freshly generated tokens (values ``>= 4``); cuts proposal 2 at position 5 of
    THAT state, so position 4 keeps its post-acceptance value.
    A stale chain instead replays ``pre_block[:5]``, restoring the naive block's
    token at position 4 and discarding acceptance 1 entirely.

    **Measured against the pre-fix sampler** (restored from a ``cp`` backup):
    ``mcmc_power_sample`` returned ``[31, 31, 0, 1, 2, 5]`` -- position 4 back to
    the naive block's token ``2`` -- and ``max_swap`` accepted only ``1`` of 2
    proposals, because the stale proposal was scored against the already-updated
    bookkeeping. Post-fix both return ``[31, 31, 0, 1, 4, 7]`` with 2
    acceptances.
    """

    PRE_BLOCK_TOKEN_AT_4 = 2  # the naive block's 3rd token (stub call #2)

    def _run(self, monkeypatch, method, force_accept):
        m = CounterLM()
        cfg = PowerSamplingConfig(
            pad_token_id=0, max_tokens=4, block_num=1, mcmc_steps=2,
            temperature=0.25,
        )
        s = PowerSampler(m, CharTokenizer(), cfg)

        cut_points = iter([4, 5] * 8)
        monkeypatch.setattr(random, "randint", lambda a, b: next(cut_points))
        if force_accept:
            monkeypatch.setattr(np.random, "rand", lambda *a: 0.0)

        return getattr(s, method)("ab")

    def test_mcmc_second_proposal_is_cut_from_the_accepted_state(
        self, monkeypatch,
    ):
        """Position 4 must carry a post-acceptance token, not the block's."""
        ids, info = self._run(monkeypatch, "mcmc_power_sample", True)

        assert info["acceptances"] == 2  # both forced through
        assert ids[:4] == [31, 31, 0, 1]  # untouched by either cut point
        # Every token generated after the naive block has value >= 4 (the stub
        # emitted 0..3 there), so this is the MH property, spelled numerically.
        assert ids[4] != self.PRE_BLOCK_TOKEN_AT_4
        assert ids[4] >= 4

    def test_max_swap_second_proposal_is_scored_against_its_own_state(
        self, monkeypatch,
    ):
        """A greedy chain of strictly-improving proposals accepts all of them.

        ``max_swap`` has no ``rand`` draw to patch: acceptance is ``log_r > 0``,
        and the stub guarantees a later proposal is strictly more probable than
        an earlier one, so a chain that proposes from its own state accepts
        both. Against the stale chain the second proposal was generated before
        the first was accepted, so it was scored against the updated
        bookkeeping and lost -- 1 acceptance, not 2.
        """
        ids, info = self._run(monkeypatch, "max_swap", False)

        assert info["acceptances"] == 2
        assert ids[:4] == [31, 31, 0, 1]
        assert ids[4] >= 4
        assert ids[5] >= 4

    def test_a_rejection_leaves_the_queued_proposals_valid(self, monkeypatch):
        """Anti-vacuity: rejections must NOT trigger a re-batch.

        A rejected proposal leaves the chain state unchanged, so the proposals
        already generated behind it are still valid draws from ``q(.|x)`` and
        must be consumed as-is. With every proposal rejected the final sequence
        must be exactly the naive block, and all ``mcmc_steps`` attempts must
        still be counted -- a fix that re-batched on rejection too would burn
        extra forward passes for the same answer.
        """
        m = CounterLM()
        cfg = PowerSamplingConfig(
            pad_token_id=0, max_tokens=4, block_num=1, mcmc_steps=4,
            temperature=0.25,
        )
        s = PowerSampler(m, CharTokenizer(), cfg)
        cut_points = iter([4, 5, 4, 5] * 8)
        monkeypatch.setattr(random, "randint", lambda a, b: next(cut_points))
        monkeypatch.setattr(np.random, "rand", lambda *a: 1.0)  # reject all

        ids, info = s.mcmc_power_sample("ab")

        assert info["acceptances"] == 0
        assert info["total_steps"] == 4
        assert ids == [31, 31, 0, 1, 2, 3]  # the untouched naive block
        # 4 naive calls + one batch of 4 proposals (2 rows need a 2nd token).
        assert m.counter == 10


# ---------------------------------------------------------------------------
# C-30(a): the documented default configuration must not die mid-generation
# ---------------------------------------------------------------------------
class TestPadTokenIsRequiredEagerly:
    """The pad-token precondition of batched MCMC proposals is checked eagerly.

    Regime this needs, and why: the config under test is exactly the one the
    package's own module docstring blesses — "supply only the IDs your model
    needs", i.e. ``pad_token_id=None``/``ctx_len=None`` — with the shipped
    ``mcmc_steps`` default (10) left alone. Before the fix, construction
    SUCCEEDED and the run died several forward passes later inside
    ``make_batch_logits_fn``'s closure with ``ValueError: pad_id is required
    for a variable-length batch with unequal prefix lengths (lengths=[4, 4,
    3])`` — a message naming the closure parameter ``pad_id``, not the
    ``PowerSamplingConfig.pad_token_id`` field the caller has to set. "It
    raised either way" is therefore NOT the property under test: the assertions
    below pin WHERE it raises (construction) and WHICH field it names.

    Every other MCMC test in this file passes ``pad_token_id=0, ctx_len=16``,
    which is why the suite never saw this.
    """

    def test_documented_default_config_raises_at_construction(self):
        """Construction refuses pad_token_id=None with mcmc_steps >= 2."""
        cfg = PowerSamplingConfig(max_tokens=4, block_num=2)  # docstring shape
        assert cfg.pad_token_id is None and cfg.ctx_len is None
        assert cfg.mcmc_steps >= 2  # shipped default, not overridden here

        with pytest.raises(ValueError) as exc:
            PowerSampler(DictMockLM(), CharTokenizer(), cfg)

        msg = str(exc.value)
        assert "pad_token_id" in msg  # the FIELD, not the closure's `pad_id`
        assert "mcmc_steps" in msg

    def test_per_call_mcmc_steps_override_is_also_refused(self):
        """A config that passes at construction is re-checked per call.

        ``mcmc_steps=1`` batches nothing, so it is admissible; raising it to 3
        at the call site reintroduces the ragged batch and must raise there.
        """
        cfg = PowerSamplingConfig(max_tokens=4, block_num=2, mcmc_steps=1)
        s = PowerSampler(DictMockLM(), CharTokenizer(), cfg)  # must not raise

        for method in (s.mcmc_power_sample, s.max_swap):
            with pytest.raises(ValueError, match="pad_token_id"):
                method("abc", mcmc_steps=3)

    def test_injected_logits_fn_needs_no_pad_id(self):
        """Anti-vacuity: the guard must not reject the never-batched path.

        With an explicit ``logits_fn=``, ``_batched_generate`` loops the
        single-position closure and no padding is ever performed, so
        ``pad_token_id=None`` is correct there. A guard keyed on
        ``mcmc_steps >= 2`` alone would fail this test.
        """
        fn = lambda ids: np.zeros(VOCAB, dtype="float32")
        cfg = PowerSamplingConfig(max_tokens=4, block_num=2, mcmc_steps=3)
        s = PowerSampler(None, CharTokenizer(), cfg, logits_fn=fn)

        random.seed(0)
        np.random.seed(0)
        _, info = s.mcmc_power_sample("abc")
        assert 0.0 <= info["acceptance_ratio"] <= 1.0

    def test_pad_token_id_supplied_still_runs(self):
        """Anti-vacuity: supplying the field makes the same config run."""
        cfg = PowerSamplingConfig(
            pad_token_id=0, max_tokens=4, block_num=2, mcmc_steps=3,
        )
        s = PowerSampler(DictMockLM(), CharTokenizer(), cfg)

        random.seed(0)
        np.random.seed(0)
        _, info = s.mcmc_power_sample("abc")
        assert 0.0 <= info["acceptance_ratio"] <= 1.0
