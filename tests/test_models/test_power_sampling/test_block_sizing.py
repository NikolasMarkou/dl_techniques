"""F-56: `PowerSampler` silently truncated `max_tokens`, and crashed below
`block_num`.

Both MCMC entry points sized their blocks with ``jump_size = max_tok // blocks``
under a comment reading "Adjust block size to divide evenly". Two defects fell
out of that one expression:

1. **Silent truncation.** The remainder was discarded, so with the shipped
   ``block_num=16`` a caller asking for ``max_tokens=50`` got 48 tokens back and
   nothing said so.
2. **An opaque stdlib crash.** With ``max_tokens < block_num`` the quotient is
   ``0``: every block appended nothing, so ``t == len(gen) == c`` and the
   cut-point draw ``random.randint(c, t - 1)`` became ``randint(c, c - 1)`` ->
   ``ValueError: empty range``, raised from inside ``random``, with nothing in
   the traceback naming ``max_tokens`` or ``block_num``.

Neither behaviour was tested; the existing suite only ever ran
``max_tokens=4, block_num=2``, which divides evenly. See decisions.md
plan-2026-08-18T140459-7991552f/D-037.
"""

import random

import numpy as np
import pytest

from dl_techniques.models.power_sampling import PowerSampler, PowerSamplingConfig

from .test_power_sampling import CharTokenizer, DictMockLM


def _sampler(**overrides):
    base = dict(pad_token_id=0, ctx_len=64, max_tokens=8, mcmc_steps=1)
    base.update(overrides)
    return PowerSampler(DictMockLM(), CharTokenizer(), PowerSamplingConfig(**base))


PROMPT = "abc"
_PROMPT_LEN = len(CharTokenizer().encode(PROMPT))


class TestBlockSizesIsTheOnlyDivision:
    """The helper both entry points now share."""

    def test_sizes_sum_to_max_tokens_exactly(self):
        for max_tokens in range(1, 60):
            for block_num in (1, 2, 3, 7, 16):
                sizes = PowerSampler._block_sizes(max_tokens, block_num)
                assert sum(sizes) == max_tokens, (max_tokens, block_num, sizes)

    def test_no_block_is_empty(self):
        """The precondition the cut-point draw actually needs."""
        for max_tokens in range(1, 60):
            for block_num in (1, 2, 3, 7, 16):
                sizes = PowerSampler._block_sizes(max_tokens, block_num)
                assert min(sizes) >= 1, (max_tokens, block_num, sizes)

    def test_the_remainder_goes_to_the_leading_blocks(self):
        assert PowerSampler._block_sizes(50, 16) == [4] * 2 + [3] * 14
        assert PowerSampler._block_sizes(8, 3) == [3, 3, 2]

    def test_block_count_is_clamped_below_max_tokens(self):
        assert PowerSampler._block_sizes(3, 16) == [1, 1, 1]

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_arguments_are_refused_by_name(self, bad):
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            PowerSampler._block_sizes(bad, 4)
        with pytest.raises(ValueError, match="block_num must be positive"):
            PowerSampler._block_sizes(8, bad)


class TestNoSilentTruncation:
    """`max_tokens=50` with the shipped `block_num=16` used to give 48."""

    @pytest.mark.parametrize("method", ["mcmc_power_sample", "max_swap"])
    def test_the_requested_number_of_tokens_is_generated(self, method):
        random.seed(0)
        np.random.seed(0)
        s = _sampler(max_tokens=50, block_num=16)
        ids, _ = getattr(s, method)(PROMPT)
        assert len(ids) - _PROMPT_LEN == 50, (
            f"{method} generated {len(ids) - _PROMPT_LEN} tokens for a "
            f"requested max_tokens=50; 50 // 16 == 3 discards the remainder "
            f"(F-56)"
        )

    def test_an_evenly_dividing_request_is_unchanged(self):
        """Control: the regime the pre-existing suite covered still holds, so
        the fix is a repair of the remainder case and not a re-sizing."""
        random.seed(0)
        np.random.seed(0)
        s = _sampler(max_tokens=48, block_num=16)
        ids, _ = s.mcmc_power_sample(PROMPT)
        assert len(ids) - _PROMPT_LEN == 48


class TestShortRequestsDoNotCrash:
    """`max_tokens < block_num` used to raise `ValueError: empty range` from
    inside stdlib `random`."""

    @pytest.mark.parametrize("method", ["mcmc_power_sample", "max_swap"])
    def test_max_tokens_below_block_num_still_generates(self, method):
        random.seed(0)
        np.random.seed(0)
        s = _sampler(max_tokens=3, block_num=16)
        ids, info = getattr(s, method)(PROMPT)
        assert len(ids) - _PROMPT_LEN == 3
        assert info["total_steps"] > 0, (
            "the chain made zero proposals -- the run did not crash, but it "
            "did not sample either"
        )

    @pytest.mark.parametrize("method", ["mcmc_power_sample", "max_swap"])
    def test_a_single_token_request_still_generates(self, method):
        random.seed(0)
        np.random.seed(0)
        s = _sampler(max_tokens=1, block_num=16)
        ids, _ = getattr(s, method)(PROMPT)
        assert len(ids) - _PROMPT_LEN == 1
