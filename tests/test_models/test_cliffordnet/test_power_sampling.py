"""Tests for CliffordNet power sampling inference.

Covers PowerSampler initialization, standard generation, MCMC power
sampling, max-swap, log probability computation, and utility functions.
"""

import pytest
import numpy as np
import keras

from dl_techniques.models.cliffordnet.lm import CliffordNetLM
from dl_techniques.models.cliffordnet.power_sampling import (
    PowerSampler,
    PowerSamplingConfig,
    _log_softmax,
    _nucleus_sample,
)


def _random_ids(shape, vocab_size):
    """Generate random integer token IDs."""
    return np.random.randint(0, vocab_size, shape).astype(np.int32)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


# Small vocab + short sequences for fast tests
_VOCAB_SIZE = 256
_MAX_SEQ = 32
_CTX_LEN = _MAX_SEQ - 1  # 31


@pytest.fixture
def tiny_config():
    """Minimal CliffordNetLM config for fast testing."""
    return {
        "vocab_size": _VOCAB_SIZE,
        "max_seq_length": _MAX_SEQ,
        "channels": 64,
        "depth": 2,
        "shifts": [1, 2],
        "dropout_rate": 0.0,
        "stochastic_depth_rate": 0.0,
    }


@pytest.fixture
def tiny_model(tiny_config):
    """Pre-built tiny CliffordNetLM."""
    model = CliffordNetLM(**tiny_config)
    dummy = _random_ids((1, _CTX_LEN), _VOCAB_SIZE)
    model(dummy, training=False)
    return model


class _FakeEncoding:
    """Minimal tiktoken-compatible encoding for tests without tiktoken."""

    def encode(self, text: str):
        # Map each character to its ordinal mod vocab_size
        return [ord(c) % _VOCAB_SIZE for c in text]

    def decode(self, ids):
        return "".join(chr(i % 128) for i in ids if 32 <= (i % 128) < 127)


@pytest.fixture
def fake_encoding():
    return _FakeEncoding()


@pytest.fixture
def sampling_config():
    """Power sampling config tuned for tiny model."""
    return PowerSamplingConfig(
        temperature=0.5,
        mcmc_steps=2,
        block_num=2,
        max_tokens=8,
        top_p=0.95,
        repetition_penalty=1.0,  # disable for predictability
        repetition_window=10,
        special_token_ids=set(),  # no special tokens in tiny vocab
        cls_token_id=0,
        pad_token_id=1,
        ctx_len=_CTX_LEN,
    )


@pytest.fixture
def sampler(tiny_model, fake_encoding, sampling_config):
    """PowerSampler wrapping the tiny model."""
    return PowerSampler(
        tiny_model, fake_encoding, _MAX_SEQ, sampling_config,
    )


# ---------------------------------------------------------------------
# Utility Function Tests
# ---------------------------------------------------------------------


class TestUtilityFunctions:

    def test_log_softmax_sums_to_one(self):
        logits = np.random.randn(100).astype(np.float32)
        log_probs = _log_softmax(logits)
        # exp(log_probs) should sum to 1
        probs = np.exp(log_probs)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_log_softmax_preserves_order(self):
        logits = np.array([3.0, 1.0, 2.0])
        log_probs = _log_softmax(logits)
        assert log_probs[0] > log_probs[2] > log_probs[1]

    def test_log_softmax_numerical_stability(self):
        # Large logits should not produce inf/nan
        logits = np.array([1000.0, 999.0, 998.0])
        log_probs = _log_softmax(logits)
        assert np.all(np.isfinite(log_probs))

    def test_nucleus_sample_returns_valid_index(self):
        logits = np.random.randn(50).astype(np.float32)
        for _ in range(20):
            token = _nucleus_sample(logits, top_p=0.9)
            assert 0 <= token < 50

    def test_nucleus_sample_respects_top_p(self):
        # With top_p=0.01, should almost always pick the top token
        logits = np.zeros(100, dtype=np.float32)
        logits[42] = 10.0  # Make token 42 dominant
        tokens = [_nucleus_sample(logits, top_p=0.01) for _ in range(50)]
        assert tokens.count(42) > 40  # should be almost all 42


# ---------------------------------------------------------------------
# PowerSampler Tests
# ---------------------------------------------------------------------


class TestPowerSamplerInit:

    def test_initialization(self, sampler):
        assert sampler.model is not None
        assert sampler.config.temperature == 0.5
        assert sampler.config.mcmc_steps == 2

    def test_default_config(self, tiny_model, fake_encoding):
        s = PowerSampler(tiny_model, fake_encoding)
        assert s.config.temperature == 0.25
        assert s.config.mcmc_steps == 10


class TestForwardPass:

    def test_forward_returns_logits(self, sampler):
        ids = [0, 10, 20, 30]
        logits, real = sampler._forward(ids)
        assert logits.shape == (_CTX_LEN, _VOCAB_SIZE)
        assert real == 4

    def test_forward_with_long_input(self, sampler):
        # Longer than ctx_len — should truncate
        ids = list(range(_CTX_LEN + 10))
        logits, real = sampler._forward(ids)
        assert logits.shape == (_CTX_LEN, _VOCAB_SIZE)
        assert real == _CTX_LEN


class TestSampleToken:

    def test_returns_valid_token(self, sampler):
        logits = np.random.randn(_VOCAB_SIZE).astype(np.float32)
        token, lp_norm, lp_unnorm = sampler._sample_token(
            logits, temperature=0.5,
        )
        assert 0 <= token < _VOCAB_SIZE
        assert np.isfinite(lp_norm)
        assert np.isfinite(lp_unnorm)

    def test_log_probs_are_negative(self, sampler):
        logits = np.random.randn(_VOCAB_SIZE).astype(np.float32)
        _, lp_norm, lp_unnorm = sampler._sample_token(
            logits, temperature=0.5,
        )
        assert lp_norm <= 0.0
        assert lp_unnorm <= 0.0


class TestNaiveTempGenerate:

    def test_generates_correct_length(self, sampler):
        context = [0, 10, 20]
        ids, lp_norm, lp_unnorm = sampler.naive_temp_generate(
            context, temperature=0.5, num_tokens=5,
        )
        assert len(ids) == 3 + 5  # context + generated
        assert len(lp_norm) == 5
        assert len(lp_unnorm) == 5

    def test_context_preserved(self, sampler):
        context = [0, 10, 20]
        ids, _, _ = sampler.naive_temp_generate(
            context, temperature=0.5, num_tokens=3,
        )
        assert ids[:3] == context

    def test_tokens_in_valid_range(self, sampler):
        context = [0]
        ids, _, _ = sampler.naive_temp_generate(
            context, temperature=0.5, num_tokens=10,
        )
        for t in ids:
            assert 0 <= t < _VOCAB_SIZE


class TestMCMCPowerSample:

    def test_produces_output(self, sampler):
        ids, info = sampler.mcmc_power_sample("hello")
        assert len(ids) > 0
        assert "acceptance_ratio" in info
        assert "elapsed_s" in info

    def test_acceptance_ratio_valid(self, sampler):
        _, info = sampler.mcmc_power_sample("test")
        assert 0.0 <= info["acceptance_ratio"] <= 1.0

    def test_total_steps_correct(self, sampler):
        _, info = sampler.mcmc_power_sample(
            "test", mcmc_steps=3, block_num=2,
        )
        # 2 blocks × 3 steps = 6 total attempts
        assert info["total_steps"] == 6

    def test_override_parameters(self, sampler):
        ids, info = sampler.mcmc_power_sample(
            "test",
            temperature=0.5,
            mcmc_steps=1,
            max_tokens=4,
            block_num=2,
        )
        # Should generate ~4 tokens (2 blocks × 2 tokens each)
        assert info["total_steps"] == 2  # 2 blocks × 1 step


class TestMaxSwap:

    def test_produces_output(self, sampler):
        ids, info = sampler.max_swap("hello")
        assert len(ids) > 0
        assert "acceptance_ratio" in info

    def test_acceptance_ratio_valid(self, sampler):
        _, info = sampler.max_swap("test")
        assert 0.0 <= info["acceptance_ratio"] <= 1.0


class TestGenerateStandard:

    def test_produces_output(self, sampler):
        ids, info = sampler.generate_standard("hello", max_tokens=5)
        assert len(ids) > 0
        assert "elapsed_s" in info
        assert "tok_per_s" in info

    def test_correct_token_count(self, sampler):
        ids, info = sampler.generate_standard(
            "hello", max_tokens=10,
        )
        # Should have prompt tokens + 10 generated
        assert info["tokens_generated"] == 10


class TestGenerateText:

    def test_standard_method(self, sampler):
        text, info = sampler.generate_text(
            "hello", method="standard", max_tokens=5,
        )
        assert isinstance(text, str)

    def test_power_method(self, sampler):
        text, info = sampler.generate_text(
            "hello", method="power",
            max_tokens=4, block_num=2, mcmc_steps=1,
        )
        assert isinstance(text, str)
        assert "acceptance_ratio" in info

    def test_max_swap_method(self, sampler):
        text, info = sampler.generate_text(
            "hello", method="max_swap",
            max_tokens=4, block_num=2, mcmc_steps=1,
        )
        assert isinstance(text, str)

    def test_invalid_method_raises(self, sampler):
        with pytest.raises(ValueError, match="Unknown method"):
            sampler.generate_text("hello", method="invalid")
