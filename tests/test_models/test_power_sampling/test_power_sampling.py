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
