"""Power sampling for CliffordNet causal language models.

Implements scalable power sampling for inference-time reasoning improvement,
adapting the autoregressive MCMC approach from:

    - Karan, A. & Du, Y. (2025). *Reasoning with Sampling: Your Base Model
      is Smarter Than You Think*. arXiv:2510.14901.
    - Bou Ammar, H. et al. (2026). *Scalable Power Sampling for LLM
      Reasoning*. arXiv:2601.21590.

**Core idea**: Instead of sampling from the base model distribution *p*,
sample from the power distribution *p^alpha* (alpha = 1/temperature).  Low
temperature sharpens *local* token confidence; power sampling sharpens
*global trajectory* quality.  The MCMC loop refines generated text by
proposing alternative continuations and accepting those with higher
trajectory-level probability under *p^alpha*.

Usage::

    import tiktoken
    from dl_techniques.models.cliffordnet.lm import CliffordNetLM
    from dl_techniques.models.cliffordnet.power_sampling import PowerSampler

    model = CliffordNetLM.from_variant("nano", vocab_size=50261)
    model(np.zeros((1, 511), dtype="int32"), training=False)  # build

    enc = tiktoken.get_encoding("gpt2")
    sampler = PowerSampler(model, enc, max_seq_length=512)

    # Standard nucleus sampling (baseline)
    ids = sampler.generate_standard("The capital of France is", max_tokens=50)
    print(enc.decode(ids))

    # MCMC power sampling (improved reasoning)
    ids = sampler.mcmc_power_sample("The capital of France is", max_tokens=50)
    print(enc.decode(ids))
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import tiktoken

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PowerSamplingConfig:
    """Configuration for power sampling inference.

    :param temperature: Controls the power distribution exponent.
        ``alpha = 1 / temperature``.  Lower temperature → sharper
        distribution → higher-quality but less diverse trajectories.
    :param mcmc_steps: Number of Metropolis-Hastings refinement steps
        per generation block.
    :param block_num: Number of generation blocks.  Total tokens are split
        into ``block_num`` chunks; MCMC refinement runs after each chunk.
    :param max_tokens: Maximum number of tokens to generate.
    :param top_p: Nucleus sampling threshold for the proposal distribution.
    :param repetition_penalty: Sign-aware penalty applied to recently
        generated tokens.
    :param repetition_window: Number of recent tokens to penalize.
    :param special_token_ids: Token IDs to never generate.
    :param cls_token_id: Token ID prepended to every prompt.
    :param pad_token_id: Token ID used for right-padding to fixed length.
    :param ctx_len: Context window length for fixed-shape forward passes.
    """

    temperature: float = 0.25
    mcmc_steps: int = 10
    block_num: int = 16
    max_tokens: int = 512
    top_p: float = 0.92
    repetition_penalty: float = 1.3
    repetition_window: int = 50
    special_token_ids: Set[int] = field(
        default_factory=lambda: {50257, 50258, 50259, 50260},
    )
    cls_token_id: int = 50257
    pad_token_id: int = 50260
    ctx_len: int = 511


# ---------------------------------------------------------------------------
# Power Sampler
# ---------------------------------------------------------------------------


class PowerSampler:
    """Wraps a :class:`CliffordNetLM` for power-distribution sampling.

    All sampling logic uses NumPy (not TF ops) to avoid graph retraces.
    The model forward pass is the only TF operation.

    :param model: A built :class:`CliffordNetLM` instance.
    :param encoding: A ``tiktoken.Encoding`` for tokenization/decoding.
    :param max_seq_length: Model's maximum sequence length (for positional
        embeddings).  Context window is ``max_seq_length - 1``.
    :param config: Sampling configuration.  Uses defaults if ``None``.
    """

    def __init__(
        self,
        model,
        encoding: tiktoken.Encoding,
        max_seq_length: int = 512,
        config: Optional[PowerSamplingConfig] = None,
    ):
        self.model = model
        self.encoding = encoding
        self.max_seq_length = max_seq_length
        self.config = config or PowerSamplingConfig()

    # -----------------------------------------------------------------
    # Low-level helpers
    # -----------------------------------------------------------------

    def _forward(self, token_ids: List[int]) -> np.ndarray:
        """Run a single fixed-shape forward pass.

        Pads ``token_ids`` to ``config.ctx_len`` and returns the full
        logits array for every position.

        :param token_ids: Token IDs (at most ``config.ctx_len``).
        :return: Logits array of shape ``(ctx_len, vocab_size)``.
        """
        cfg = self.config
        ctx = token_ids[-cfg.ctx_len:]
        real = len(ctx)
        padded = ctx + [cfg.pad_token_id] * (cfg.ctx_len - real)
        out = self.model(
            np.array([padded], dtype="int32"), training=False,
        )
        return out["logits"][0].numpy(), real

    def _sample_token(
        self,
        logits: np.ndarray,
        temperature: float,
        recent_tokens: Optional[List[int]] = None,
    ) -> Tuple[int, float, float]:
        """Sample a single token from logits with temperature and nucleus.

        Returns the sampled token and both log probabilities needed for
        the MCMC acceptance criterion.

        :param logits: Raw logits for a single position, shape ``(V,)``.
        :param temperature: Sampling temperature.
        :param recent_tokens: Recent token IDs for repetition penalty.
        :return: ``(token_id, log_prob_norm, log_prob_unnorm)`` where
            ``log_prob_norm`` is the log probability under the proposal
            (temperature-scaled + nucleus) and ``log_prob_unnorm`` is
            ``(1/temperature) * log p(token)`` under the base model.
        """
        cfg = self.config

        # Base model log probabilities (before temperature)
        base_log_probs = _log_softmax(logits)

        # Working copy for sampling modifications
        working_logits = logits.copy()

        # Mask special tokens
        for sid in cfg.special_token_ids:
            if sid < len(working_logits):
                working_logits[sid] = -1e9

        # Repetition penalty (sign-aware)
        if recent_tokens:
            window = recent_tokens[-cfg.repetition_window:]
            for t in set(window):
                if t not in cfg.special_token_ids and t < len(working_logits):
                    if working_logits[t] >= 0:
                        working_logits[t] /= cfg.repetition_penalty
                    else:
                        working_logits[t] *= cfg.repetition_penalty

        # Temperature scaling
        scaled_logits = working_logits / temperature

        # Nucleus (top-p) sampling
        token_id = _nucleus_sample(scaled_logits, cfg.top_p)

        # Log probabilities for MCMC
        scaled_log_probs = _log_softmax(scaled_logits)
        log_prob_norm = scaled_log_probs[token_id]
        log_prob_unnorm = base_log_probs[token_id] / temperature

        return int(token_id), float(log_prob_norm), float(log_prob_unnorm)

    # -----------------------------------------------------------------
    # Proposal distribution: low-temperature autoregressive generation
    # -----------------------------------------------------------------

    def naive_temp_generate(
        self,
        context: List[int],
        temperature: float,
        num_tokens: int,
    ) -> Tuple[List[int], List[float], List[float]]:
        """Generate tokens autoregressively at the given temperature.

        This serves as the proposal distribution for MCMC power sampling.

        :param context: Prefix token IDs.
        :param temperature: Sampling temperature.
        :param num_tokens: Number of tokens to generate.
        :return: ``(ids, log_probs_norm, log_probs_unnorm)`` where
            ``ids`` is the full sequence (context + generated tokens).
        """
        cfg = self.config
        ids = list(context)
        log_probs_norm: List[float] = []
        log_probs_unnorm: List[float] = []

        for _ in range(num_tokens):
            logits_full, real = self._forward(ids)
            logits = logits_full[real - 1]

            token_id, lp_norm, lp_unnorm = self._sample_token(
                logits, temperature, recent_tokens=ids,
            )
            ids.append(token_id)
            log_probs_norm.append(lp_norm)
            log_probs_unnorm.append(lp_unnorm)

        return ids, log_probs_norm, log_probs_unnorm

    # -----------------------------------------------------------------
    # MCMC Power Sampling
    # -----------------------------------------------------------------

    def mcmc_power_sample(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        mcmc_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,
        block_num: Optional[int] = None,
    ) -> Tuple[List[int], Dict]:
        """Generate text using MCMC power sampling.

        Samples from *p^alpha* where ``alpha = 1 / temperature``.  The
        generation is split into ``block_num`` blocks; after each block,
        ``mcmc_steps`` Metropolis-Hastings refinement steps are applied.

        :param prompt: Text prompt to continue.
        :param temperature: Override config temperature.
        :param mcmc_steps: Override config MCMC steps.
        :param max_tokens: Override config max tokens.
        :param block_num: Override config block count.
        :return: ``(token_ids, info)`` where ``token_ids`` are the
            generated tokens (without CLS prefix) and ``info`` contains
            ``acceptance_ratio``, ``total_steps``, ``elapsed_s``.
        """
        cfg = self.config
        temp = temperature if temperature is not None else cfg.temperature
        steps = mcmc_steps if mcmc_steps is not None else cfg.mcmc_steps
        max_tok = max_tokens if max_tokens is not None else cfg.max_tokens
        blocks = block_num if block_num is not None else cfg.block_num

        alpha = 1.0 / temp
        logger.info(
            f"MCMC power sampling: alpha={alpha:.1f}, "
            f"temp={temp}, mcmc_steps={steps}, "
            f"max_tokens={max_tok}, blocks={blocks}"
        )

        # Tokenize prompt
        prompt_ids = [cfg.cls_token_id] + self.encoding.encode(prompt)
        c = len(prompt_ids)  # context boundary

        # Adjust block size to divide evenly
        jump_size = max_tok // blocks

        gen = list(prompt_ids)
        log_probs_norm: List[float] = []
        log_probs_unnorm: List[float] = []
        attempts = 0
        acceptances = 0

        t0 = time.time()

        for block_idx in range(blocks):
            # Generate one block of tokens with naive temperature sampling
            target_len = len(gen) + jump_size
            gen, lp_norm, lp_unnorm = self.naive_temp_generate(
                gen, temp, num_tokens=jump_size,
            )
            log_probs_norm.extend(lp_norm)
            log_probs_unnorm.extend(lp_unnorm)

            # MCMC refinement
            for _ in range(steps):
                attempts += 1
                t = len(gen)
                idx = random.randint(c, t - 1)

                # Propose: re-generate from idx to end
                prop, lp_prop, target_lp_prop = self.naive_temp_generate(
                    gen[:idx], temp, num_tokens=t - idx,
                )

                s = len(prop)
                lp_cur = log_probs_norm[idx - c: s - c]
                target_lp_cur = log_probs_unnorm[idx - c: s - c]

                # Metropolis-Hastings acceptance criterion
                log_r = (
                    sum(target_lp_prop) + sum(lp_cur)
                    - sum(target_lp_cur) - sum(lp_prop)
                )

                if np.random.rand() < np.exp(min(log_r, 0.0)):
                    acceptances += 1
                    gen = list(prop)
                    log_probs_norm[idx - c:] = list(lp_prop)
                    log_probs_unnorm[idx - c:] = list(target_lp_prop)

        elapsed = time.time() - t0
        acceptance_ratio = acceptances / max(attempts, 1)

        logger.info(
            f"Power sampling complete: {len(gen) - c} tokens, "
            f"acceptance={acceptance_ratio:.2%}, "
            f"{elapsed:.1f}s"
        )

        info = {
            "acceptance_ratio": acceptance_ratio,
            "total_steps": attempts,
            "acceptances": acceptances,
            "elapsed_s": elapsed,
            "alpha": alpha,
        }
        return gen[1:], info  # strip CLS

    def max_swap(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        mcmc_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,
        block_num: Optional[int] = None,
    ) -> Tuple[List[int], Dict]:
        """Generate text using deterministic max-swap power sampling.

        Like :meth:`mcmc_power_sample` but always accepts proposals that
        improve the trajectory probability (greedy at the trajectory
        level).  Approximates sampling from *p^infinity*.

        :param prompt: Text prompt to continue.
        :param temperature: Override config temperature.
        :param mcmc_steps: Override config MCMC steps.
        :param max_tokens: Override config max tokens.
        :param block_num: Override config block count.
        :return: ``(token_ids, info)`` — same format as
            :meth:`mcmc_power_sample`.
        """
        cfg = self.config
        temp = temperature if temperature is not None else cfg.temperature
        steps = mcmc_steps if mcmc_steps is not None else cfg.mcmc_steps
        max_tok = max_tokens if max_tokens is not None else cfg.max_tokens
        blocks = block_num if block_num is not None else cfg.block_num

        logger.info(
            f"Max-swap power sampling: temp={temp}, "
            f"mcmc_steps={steps}, max_tokens={max_tok}, blocks={blocks}"
        )

        prompt_ids = [cfg.cls_token_id] + self.encoding.encode(prompt)
        c = len(prompt_ids)

        jump_size = max_tok // blocks

        gen = list(prompt_ids)
        log_probs_norm: List[float] = []
        log_probs_unnorm: List[float] = []
        attempts = 0
        acceptances = 0

        t0 = time.time()

        for block_idx in range(blocks):
            gen, lp_norm, lp_unnorm = self.naive_temp_generate(
                gen, temp, num_tokens=jump_size,
            )
            log_probs_norm.extend(lp_norm)
            log_probs_unnorm.extend(lp_unnorm)

            for _ in range(steps):
                attempts += 1
                t = len(gen)
                idx = random.randint(c, t - 1)

                prop, lp_prop, target_lp_prop = self.naive_temp_generate(
                    gen[:idx], temp, num_tokens=t - idx,
                )

                s = len(prop)
                target_lp_cur = log_probs_unnorm[idx - c: s - c]

                # Deterministic: accept if trajectory probability improves
                log_r = sum(target_lp_prop) - sum(target_lp_cur)

                if log_r > 0:
                    acceptances += 1
                    gen = list(prop)
                    log_probs_norm[idx - c:] = list(lp_prop)
                    log_probs_unnorm[idx - c:] = list(target_lp_prop)

        elapsed = time.time() - t0
        acceptance_ratio = acceptances / max(attempts, 1)

        logger.info(
            f"Max-swap complete: {len(gen) - c} tokens, "
            f"acceptance={acceptance_ratio:.2%}, "
            f"{elapsed:.1f}s"
        )

        info = {
            "acceptance_ratio": acceptance_ratio,
            "total_steps": attempts,
            "acceptances": acceptances,
            "elapsed_s": elapsed,
        }
        return gen[1:], info  # strip CLS

    def generate_standard(
        self,
        prompt: str,
        temperature: float = 0.85,
        top_p: float = 0.92,
        max_tokens: int = 100,
        repetition_penalty: float = 1.3,
    ) -> Tuple[List[int], Dict]:
        """Standard nucleus sampling (baseline for comparison).

        Uses the same sampling pipeline as the training probe callback.

        :param prompt: Text prompt to continue.
        :param temperature: Sampling temperature.
        :param top_p: Nucleus sampling threshold.
        :param max_tokens: Maximum tokens to generate.
        :param repetition_penalty: Repetition penalty factor.
        :return: ``(token_ids, info)`` where ``info`` contains timing.
        """
        cfg = self.config
        ids = [cfg.cls_token_id] + self.encoding.encode(prompt)

        t0 = time.time()

        # Temporarily override config for this generation
        old_penalty = cfg.repetition_penalty
        old_top_p = cfg.top_p
        cfg.repetition_penalty = repetition_penalty
        cfg.top_p = top_p

        for _ in range(max_tokens):
            logits_full, real = self._forward(ids)
            logits = logits_full[real - 1]
            token_id, _, _ = self._sample_token(
                logits, temperature, recent_tokens=ids,
            )
            ids.append(token_id)

        # Restore config
        cfg.repetition_penalty = old_penalty
        cfg.top_p = old_top_p

        elapsed = time.time() - t0
        info = {
            "elapsed_s": elapsed,
            "tokens_generated": max_tokens,
            "tok_per_s": max_tokens / max(elapsed, 0.01),
        }
        return ids[1:], info  # strip CLS

    # -----------------------------------------------------------------
    # Convenience: string-in / string-out
    # -----------------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        method: str = "power",
        **kwargs,
    ) -> Tuple[str, Dict]:
        """Generate text with the specified method.

        :param prompt: Input text prompt.
        :param method: ``"standard"``, ``"power"``, or ``"max_swap"``.
        :param kwargs: Passed to the chosen generation method.
        :return: ``(text, info)`` tuple.
        """
        if method == "standard":
            ids, info = self.generate_standard(prompt, **kwargs)
        elif method == "power":
            ids, info = self.mcmc_power_sample(prompt, **kwargs)
        elif method == "max_swap":
            ids, info = self.max_swap(prompt, **kwargs)
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"Use 'standard', 'power', or 'max_swap'."
            )
        text = self.encoding.decode(ids)
        return text, info


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    shifted = logits - logits.max()
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    return shifted - log_sum_exp


def _nucleus_sample(logits: np.ndarray, top_p: float) -> int:
    """Sample a token using nucleus (top-p) sampling.

    :param logits: Logits for a single position (already temperature-scaled).
    :param top_p: Cumulative probability threshold.
    :return: Sampled token ID.
    """
    sorted_idx = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]

    # Numerically stable softmax
    probs = np.exp(sorted_logits - sorted_logits[0])
    probs /= probs.sum()

    # Find nucleus cutoff
    cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
    top_idx = sorted_idx[:cutoff]
    top_probs = probs[:cutoff]
    top_probs /= top_probs.sum()

    return int(top_idx[np.random.choice(len(top_idx), p=top_probs)])
