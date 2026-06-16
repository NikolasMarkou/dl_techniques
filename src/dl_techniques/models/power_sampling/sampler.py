"""General-purpose power sampling for any causal LLM/VLM + any tokenizer.

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

This engine is fully decoupled from any concrete model or tokenizer: it is
driven by an injected :data:`~dl_techniques.models.power_sampling.protocols.LogitsFn`
closure (built automatically from any callable Keras model via
:func:`~dl_techniques.models.power_sampling.forward.make_logits_fn`) and any
object satisfying
:class:`~dl_techniques.models.power_sampling.protocols.TokenizerProtocol`
(``encode``/``decode``). CliffordNetLM + tiktoken is just ONE example — the
same sampler drives a GPT-2, a generic HF model, or a VLM (via
:class:`~dl_techniques.models.power_sampling.forward.VLMForwardAdapter`).

Usage::

    import numpy as np
    from dl_techniques.models.power_sampling import (
        PowerSampler, PowerSamplingConfig,
    )

    # Any callable model returning {"logits": float32[B, T, V]} works.
    # CliffordNetLM is one example; a GPT-2 or generic LM works the same way.
    model = build_my_causal_lm()          # any callable Keras model
    tokenizer = get_my_tokenizer()        # any object with encode/decode

    # Generalized config: supply only the IDs your model needs. Defaults carry
    # NO GPT-2/CliffordNet IDs (cls/pad/special are None/empty, ctx_len=None).
    config = PowerSamplingConfig(
        cls_token_id=50257,               # optional; None => no CLS prepend
        pad_token_id=50260,               # required only for fixed ctx_len
        special_token_ids={50257, 50258, 50259, 50260},
        ctx_len=511,                      # None => variable-length forward
    )
    sampler = PowerSampler(model, tokenizer, config)

    # Standard nucleus sampling (baseline)
    ids = sampler.generate_standard("The capital of France is", max_tokens=50)
    print(tokenizer.decode(ids[0]))

    # MCMC power sampling (improved reasoning)
    ids, info = sampler.mcmc_power_sample("The capital of France is", max_tokens=50)
    print(tokenizer.decode(ids))

A pre-built ``LogitsFn`` closure (e.g. from a VLM adapter) can be injected
directly via the ``logits_fn=`` keyword.
"""

import time
import random
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.models.power_sampling.config import PowerSamplingConfig
from dl_techniques.models.power_sampling.protocols import TokenizerProtocol, LogitsFn
from dl_techniques.models.power_sampling.ops import _log_softmax, _nucleus_sample
from dl_techniques.models.power_sampling.forward import (
    make_logits_fn,
    make_batch_logits_fn,
)


# ---------------------------------------------------------------------------
# Power Sampler
# ---------------------------------------------------------------------------


class PowerSampler:
    """Power-distribution sampling for any causal LM/VLM + any tokenizer.

    Forward passes run on whatever device the injected model uses; post-logit
    sampling uses NumPy on CPU. MCMC proposals within each block are generated
    in parallel via batched forward passes for high GPU utilization.

    The sampler is fully decoupled from concrete model/tokenizer types: it is
    driven by a :data:`LogitsFn` closure (single-position) plus an optional
    batched closure, and a :class:`TokenizerProtocol` object.

    :param model_or_logits_fn: EITHER a callable Keras model (wrapped
        automatically via :func:`make_logits_fn` using ``config.ctx_len`` /
        ``config.pad_token_id``) OR a pre-built :data:`LogitsFn` closure. To
        pass a ``LogitsFn`` unambiguously, use the ``logits_fn=`` kwarg.
    :param tokenizer: Any object satisfying :class:`TokenizerProtocol`
        (``encode``/``decode``).
    :param config: :class:`PowerSamplingConfig`; defaults to
        ``PowerSamplingConfig()`` if ``None``.
    :param logits_fn: Optional explicit single-position :data:`LogitsFn`
        override (the unambiguous path for injecting a closure).
    """

    def __init__(
        self,
        model_or_logits_fn,
        tokenizer: TokenizerProtocol,
        config: Optional[PowerSamplingConfig] = None,
        *,
        logits_fn: Optional[LogitsFn] = None,
    ):
        self.config = config or PowerSamplingConfig()
        self.tokenizer = tokenizer
        self.model = model_or_logits_fn  # kept for reference/back-compat
        cfg = self.config
        if logits_fn is not None:
            self._logits_fn = logits_fn
            # caller-supplied: batched path falls back to looping single fn
            self._batch_logits_fn = None
        else:
            # wrap the model into single + batch logits fns using config
            self._logits_fn = make_logits_fn(
                model_or_logits_fn,
                ctx_len=cfg.ctx_len,
                pad_id=cfg.pad_token_id,
                logits_key="logits",
            )
            self._batch_logits_fn = make_batch_logits_fn(
                model_or_logits_fn,
                ctx_len=cfg.ctx_len,
                pad_id=cfg.pad_token_id,
                logits_key="logits",
            )

    # -----------------------------------------------------------------
    # Low-level helpers
    # -----------------------------------------------------------------

    def _forward(self, token_ids: List[int]) -> np.ndarray:
        """Run a single forward pass via the injected logits closure.

        :param token_ids: Token IDs for the sequence.
        :return: Logits array of shape ``(vocab_size,)``.
        """
        return self._logits_fn(token_ids)  # (V,)

    def _forward_batch(
        self, batch_token_ids: List[List[int]],
    ) -> np.ndarray:
        """Batched forward pass via the injected batch logits closure.

        Falls back to looping the single-position closure when only a
        ``logits_fn`` was supplied (no batch closure available).

        :param batch_token_ids: List of B token ID sequences.
        :return: Logits array of shape ``(B, vocab_size)``.
        """
        if self._batch_logits_fn is not None:
            return self._batch_logits_fn(batch_token_ids)  # (B, V)
        # caller supplied only a single-position logits_fn: loop
        return np.stack(
            [self._logits_fn(ids) for ids in batch_token_ids], axis=0,
        )

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
    # Single-sequence autoregressive generation
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
        ids = list(context)
        log_probs_norm: List[float] = []
        log_probs_unnorm: List[float] = []

        for _ in range(num_tokens):
            logits = self._forward(ids)

            token_id, lp_norm, lp_unnorm = self._sample_token(
                logits, temperature, recent_tokens=ids,
            )
            ids.append(token_id)
            log_probs_norm.append(lp_norm)
            log_probs_unnorm.append(lp_unnorm)

        return ids, log_probs_norm, log_probs_unnorm

    # -----------------------------------------------------------------
    # Batched autoregressive generation (for parallel MCMC proposals)
    # -----------------------------------------------------------------

    def _batched_generate(
        self,
        prefixes: List[List[int]],
        num_tokens_list: List[int],
        temperature: float,
    ) -> Tuple[List[List[int]], List[List[float]], List[List[float]]]:
        """Generate tokens for multiple sequences in parallel.

        Uses batched forward passes so all sequences share a single GPU
        call per generation step.  Sequences that finish early are
        removed from the batch to save compute.

        :param prefixes: List of B prefix token ID sequences.
        :param num_tokens_list: Number of tokens to generate per sequence.
        :param temperature: Sampling temperature.
        :return: ``(seqs, log_probs_norm, log_probs_unnorm)`` where each
            is a list of B items (one per sequence).
        """
        B = len(prefixes)
        if B == 0:
            return [], [], []

        max_gen = max(num_tokens_list)

        seqs = [list(p) for p in prefixes]
        log_probs_norm: List[List[float]] = [[] for _ in range(B)]
        log_probs_unnorm: List[List[float]] = [[] for _ in range(B)]

        for step in range(max_gen):
            # Find sequences that still need tokens
            active = [i for i in range(B) if step < num_tokens_list[i]]
            if not active:
                break

            # Batched forward pass for all active sequences
            if len(active) == 1:
                # Single sequence: use unbatched path to avoid overhead
                i = active[0]
                logits = self._forward(seqs[i])
                token_id, lp_n, lp_u = self._sample_token(
                    logits, temperature, recent_tokens=seqs[i],
                )
                seqs[i].append(token_id)
                log_probs_norm[i].append(lp_n)
                log_probs_unnorm[i].append(lp_u)
            else:
                batch_ids = [seqs[i] for i in active]
                logits_batch = self._forward_batch(batch_ids)

                for j, i in enumerate(active):
                    token_id, lp_n, lp_u = self._sample_token(
                        logits_batch[j], temperature, recent_tokens=seqs[i],
                    )
                    seqs[i].append(token_id)
                    log_probs_norm[i].append(lp_n)
                    log_probs_unnorm[i].append(lp_u)

        return seqs, log_probs_norm, log_probs_unnorm

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
        ``mcmc_steps`` proposals are generated **in parallel** via batched
        forward passes and evaluated with Metropolis-Hastings acceptance.

        :param prompt: Text prompt to continue.
        :param temperature: Override config temperature.
        :param mcmc_steps: Override config MCMC steps.
        :param max_tokens: Override config max tokens.
        :param block_num: Override config block count.
        :return: ``(token_ids, info)`` where ``token_ids`` are the
            generated tokens (without CLS prefix when one was prepended) and
            ``info`` contains ``acceptance_ratio``, ``total_steps``,
            ``elapsed_s``.
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

        # Tokenize prompt (CLS prepend only when configured; G2/I2)
        encoded = self.tokenizer.encode(prompt)
        if cfg.cls_token_id is not None:
            prompt_ids = [cfg.cls_token_id] + list(encoded)
            strip = 1
        else:
            prompt_ids = list(encoded)
            strip = 0
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
            gen, lp_norm, lp_unnorm = self.naive_temp_generate(
                gen, temp, num_tokens=jump_size,
            )
            log_probs_norm.extend(lp_norm)
            log_probs_unnorm.extend(lp_unnorm)

            # Generate all MCMC proposals in parallel (batched)
            t = len(gen)
            indices = [random.randint(c, t - 1) for _ in range(steps)]
            prefixes = [list(gen[:idx]) for idx in indices]
            num_tokens = [t - idx for idx in indices]

            props, lp_props_list, target_lp_props_list = (
                self._batched_generate(prefixes, num_tokens, temp)
            )

            # Evaluate acceptance for each proposal
            for i in range(steps):
                attempts += 1
                idx = indices[i]
                s = len(props[i])

                lp_cur = log_probs_norm[idx - c: s - c]
                target_lp_cur = log_probs_unnorm[idx - c: s - c]

                # Metropolis-Hastings acceptance criterion
                log_r = (
                    sum(target_lp_props_list[i]) + sum(lp_cur)
                    - sum(target_lp_cur) - sum(lp_props_list[i])
                )

                if np.random.rand() < np.exp(min(log_r, 0.0)):
                    acceptances += 1
                    gen = list(props[i])
                    log_probs_norm[idx - c:] = list(lp_props_list[i])
                    log_probs_unnorm[idx - c:] = list(
                        target_lp_props_list[i],
                    )

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
        return gen[strip:], info  # strip CLS only when prepended

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
        level).  Approximates sampling from *p^infinity*.  Proposals are
        generated in parallel via batched forward passes.

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

        # Tokenize prompt (CLS prepend only when configured; G2/I2)
        encoded = self.tokenizer.encode(prompt)
        if cfg.cls_token_id is not None:
            prompt_ids = [cfg.cls_token_id] + list(encoded)
            strip = 1
        else:
            prompt_ids = list(encoded)
            strip = 0
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

            # Generate all proposals in parallel (batched)
            t = len(gen)
            indices = [random.randint(c, t - 1) for _ in range(steps)]
            prefixes = [list(gen[:idx]) for idx in indices]
            num_tokens = [t - idx for idx in indices]

            props, lp_props_list, target_lp_props_list = (
                self._batched_generate(prefixes, num_tokens, temp)
            )

            # Evaluate acceptance for each proposal
            for i in range(steps):
                attempts += 1
                idx = indices[i]
                s = len(props[i])

                target_lp_cur = log_probs_unnorm[idx - c: s - c]

                # Deterministic: accept if trajectory probability improves
                log_r = sum(target_lp_props_list[i]) - sum(target_lp_cur)

                if log_r > 0:
                    acceptances += 1
                    gen = list(props[i])
                    log_probs_norm[idx - c:] = list(lp_props_list[i])
                    log_probs_unnorm[idx - c:] = list(
                        target_lp_props_list[i],
                    )

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
        return gen[strip:], info  # strip CLS only when prepended

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

        # Tokenize prompt (CLS prepend only when configured; G2/I2)
        encoded = self.tokenizer.encode(prompt)
        if cfg.cls_token_id is not None:
            ids = [cfg.cls_token_id] + list(encoded)
            strip = 1
        else:
            ids = list(encoded)
            strip = 0

        t0 = time.time()

        # DECISION plan_2026-06-16_535b4f02/D-001: C8/I4 no-config-mutation.
        # Per-call top_p/repetition_penalty overrides are applied via a
        # dataclasses.replace COPY bound transiently, then the original is
        # rebound in `finally`. Do NOT restore the source's in-place
        # cfg.field = X mutate-then-restore (source:573-587): it is not
        # exception-safe and leaves observable self.config altered if the
        # loop raises. The swap-and-restore keeps config fields identical
        # before==after (SC5/I4). See decisions.md D-001.
        original = self.config
        self.config = replace(
            original, repetition_penalty=repetition_penalty, top_p=top_p,
        )
        try:
            for _ in range(max_tokens):
                logits = self._forward(ids)
                token_id, _, _ = self._sample_token(
                    logits, temperature, recent_tokens=ids,
                )
                ids.append(token_id)
        finally:
            self.config = original  # guaranteed restore -> before==after

        elapsed = time.time() - t0
        info = {
            "elapsed_s": elapsed,
            "tokens_generated": max_tokens,
            "tok_per_s": max_tokens / max(elapsed, 0.01),
        }
        return ids[strip:], info  # strip CLS only when prepended

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
        text = self.tokenizer.decode(ids)
        return text, info


__all__ = ["PowerSampler"]
