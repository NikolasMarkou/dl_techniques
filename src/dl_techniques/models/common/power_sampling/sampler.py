"""General-purpose power sampling for any causal LLM/VLM, with any tokenizer.

:class:`PowerSampler` samples from the power distribution p^alpha (alpha =
1 / temperature) instead of the base model distribution p. Low temperature
sharpens local token confidence; power sampling instead sharpens global
trajectory quality, by proposing alternative continuations in an MCMC loop
and accepting those with higher trajectory-level probability under p^alpha.

The sampler is decoupled from any concrete model or tokenizer. It is driven
by an injected :data:`~dl_techniques.models.common.power_sampling.protocols.LogitsFn`
closure (built automatically from any callable Keras model via
:func:`~dl_techniques.models.common.power_sampling.forward.make_logits_fn`)
and any object satisfying
:class:`~dl_techniques.models.common.power_sampling.protocols.TokenizerProtocol`.
The same sampler drives CliffordNetLM, a GPT-2, a generic HF model, or a VLM
(via :class:`~dl_techniques.models.common.power_sampling.forward.VLMForwardAdapter`).

References:
    - Karan, A. & Du, Y., 2025. Reasoning with Sampling: Your Base Model is
      Smarter Than You Think. (https://arxiv.org/abs/2510.14901)
    - Bou Ammar, H. et al., 2026. Scalable Power Sampling for LLM Reasoning.
      (https://arxiv.org/abs/2601.21590)
"""

import time
import random
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from dl_techniques.utils.logger import logger
from dl_techniques.models.common.power_sampling.config import PowerSamplingConfig
from dl_techniques.models.common.power_sampling.protocols import TokenizerProtocol, LogitsFn
from dl_techniques.models.common.power_sampling.ops import _log_softmax, _nucleus_sample
from dl_techniques.models.common.power_sampling.forward import (
    make_logits_fn,
    make_batch_logits_fn,
)


# Power Sampler


class PowerSampler:
    """Power-distribution sampling for any causal LM/VLM + any tokenizer.

    Forward passes run on whatever device the injected model uses; post-logit
    sampling uses NumPy on CPU. MCMC proposals are generated in parallel via
    batched forward passes for high GPU utilization, but only across a run of
    rejections: an acceptance moves the chain state, so the proposals queued
    behind it are discarded and re-generated from the new state.

    The sampler is decoupled from concrete model/tokenizer types: it is
    driven by a :data:`LogitsFn` closure (single-position) plus an optional
    batched closure, and a :class:`TokenizerProtocol` object.

    MCMC block loop:

    .. code-block:: text

        gen (chain state) ──► naive_temp_generate ──► gen + block tokens
                                                             │
                                     cut points idx ~ U[c,t) │
                                                             ▼
                                   ┌──── _batch_proposals ────┐
                                   │ regenerate gen[idx:] per │
                                   │   cut, batched forward   │
                                   └────────────┬─────────────┘
                                                 ▼
                                  MH accept/reject, one at a time
                                  accept ──► gen updated, rest of
                                  batch discarded, re-drawn from
                                  the new state

    :param model_or_logits_fn: Either a callable Keras model (wrapped
        automatically via :func:`make_logits_fn` using ``config.ctx_len`` /
        ``config.pad_token_id``) or a pre-built :data:`LogitsFn` closure. To
        pass a ``LogitsFn`` unambiguously, use the ``logits_fn=`` kwarg.
    :param tokenizer: Any object satisfying :class:`TokenizerProtocol`
        (``encode``/``decode``).
    :param config: :class:`PowerSamplingConfig`; defaults to
        ``PowerSamplingConfig()`` if ``None``.
    :param logits_fn: Optional explicit single-position :data:`LogitsFn`
        override (the unambiguous path for injecting a closure).
    :raises ValueError: If ``config.mcmc_steps >= 2`` while a wrapped model is
        driven with ``pad_token_id=None`` and ``ctx_len=None`` — the proposal
        batch is ragged and cannot be padded. See
        :meth:`_require_pad_id_for_batched_proposals`.

    Example::

        model = build_my_causal_lm()    # any callable Keras model
        tokenizer = get_my_tokenizer()  # any object with encode/decode

        # Supply only the IDs the model needs; defaults carry no
        # GPT-2/CliffordNet IDs (cls/pad/special are None/empty, ctx_len=None).
        config = PowerSamplingConfig(
            cls_token_id=50257,          # None => no CLS prepend
            pad_token_id=50260,          # required for fixed ctx_len and for
                                          # mcmc_steps >= 2
            special_token_ids={50257, 50258, 50259, 50260},
            ctx_len=511,                 # None => variable-length forward
        )
        sampler = PowerSampler(model, tokenizer, config)

        ids = sampler.generate_standard("The capital of France is", max_tokens=50)
        print(tokenizer.decode(ids[0]))

        ids, info = sampler.mcmc_power_sample("The capital of France is", max_tokens=50)
        print(tokenizer.decode(ids))

    A pre-built ``LogitsFn`` closure (e.g. from a VLM adapter) can be injected
    directly via the ``logits_fn=`` keyword.
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
        # Kept for reference/back-compat.
        self.model = model_or_logits_fn
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

        self._require_pad_id_for_batched_proposals(cfg.mcmc_steps)

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-017: check the pad-token
    # precondition here, eagerly, not inside `make_batch_logits_fn`'s `fn` —
    # that raise fires mid-generation, several blocks in, and names the wrong
    # field. See decisions.md D-017.
    def _require_pad_id_for_batched_proposals(self, steps: int) -> None:
        """Refuse a config whose MCMC proposals cannot be batched.

        :param steps: Number of MCMC proposals per block (config value or a
            per-call override).
        :raises ValueError: If ``steps >= 2`` while the sampler drives a
            wrapped model through the batched closure with
            ``pad_token_id=None`` and ``ctx_len=None``.
        """
        if steps < 2:
            # One proposal per block: batch of 1, never ragged.
            return
        if self._batch_logits_fn is None:
            # Injected logits_fn: the batched path loops the single fn.
            return
        cfg = self.config
        if cfg.ctx_len is not None:
            # Fixed-shape path; pad_id already validated by the closure.
            return
        if cfg.pad_token_id is not None:
            return
        raise ValueError(
            "PowerSamplingConfig.pad_token_id is required when mcmc_steps >= 2 "
            f"(got mcmc_steps={steps}, pad_token_id=None, ctx_len=None). MCMC "
            "proposals are re-generated from random cut points, so the batch "
            "holds prefixes of unequal length and must be right-padded. Fix: "
            "pass pad_token_id=<your model's pad id> (any id works if the "
            "model ignores padded positions), or set mcmc_steps=1, or inject a "
            "pre-built closure via logits_fn= (that path is never batched)."
        )

    # -----------------------------------------------------------------
    # Low-level helpers
    # -----------------------------------------------------------------

    def _forward(self, token_ids: List[int]) -> np.ndarray:
        """Run a single forward pass via the injected logits closure.

        :param token_ids: Token IDs for the sequence.
        :return: Logits array of shape ``(vocab_size,)``.
        """
        return self._logits_fn(token_ids)

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
            return self._batch_logits_fn(batch_token_ids)
        # Caller supplied only a single-position logits_fn: loop it.
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
            ``log_prob_norm`` is the log probability under the proposal —
            special-token masking, repetition penalty, temperature scaling
            and the renormalized top-p nucleus, i.e. the exact distribution
            the token was drawn from — and ``log_prob_unnorm`` is
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

        # Nucleus (top-p) sampling. The proposal log-probability comes back
        # from the draw itself: it is the density over the truncated,
        # renormalized nucleus, which _log_softmax(scaled_logits) is not.
        token_id, log_prob_norm = _nucleus_sample(scaled_logits, cfg.top_p)

        # Target (power) log-probability: alpha * log p(token) under the base
        # model, before masking, repetition penalty and truncation.
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

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-018: batch proposals only
    # across a run of rejections, not once per block — a pre-loop batch
    # compared proposal i+1 against a state an earlier acceptance had already
    # moved past, silently overwriting the accepted improvement. See
    # decisions.md D-018.
    def _batch_proposals(
        self,
        gen: List[int],
        indices: List[int],
        t: int,
        temperature: float,
    ) -> Tuple[List[List[int]], List[List[float]], List[List[float]]]:
        """Generate one batch of proposals, all cut from the CURRENT state.

        :param gen: The current chain state (token ids).
        :param indices: Cut points still owed in this block; every proposal
            regenerates ``gen[idx:]`` from ``gen[:idx]``.
        :param t: Block-invariant sequence length (a proposal replaces the
            suffix, so ``len(gen)`` is unchanged by an acceptance).
        :param temperature: Proposal temperature.
        :return: ``(proposals, log_probs_norm, log_probs_unnorm)``, one entry
            per index, in the order given.
        """
        prefixes = [list(gen[:idx]) for idx in indices]
        num_tokens = [t - idx for idx in indices]
        return self._batched_generate(prefixes, num_tokens, temperature)

    # -----------------------------------------------------------------
    # MCMC Power Sampling
    # -----------------------------------------------------------------

    # DECISION plan-2026-08-18T140459-7991552f/D-037: spread the remainder
    # over the leading blocks and clamp block_num to max_tokens. The plain
    # `max_tok // blocks` dropped the remainder (50 tokens requested, 48
    # generated) and could yield an empty block, which crashed the cut-point
    # draw. See decisions.md.
    @staticmethod
    def _block_sizes(max_tokens: int, block_num: int) -> List[int]:
        """Split ``max_tokens`` into per-block generation counts.

        :param max_tokens: Total number of tokens to generate. Must be
            positive.
        :param block_num: Requested number of MCMC blocks. Must be positive;
            clamped down to ``max_tokens`` (with a warning) so that no block is
            empty.
        :return: A list of per-block token counts, each ``>= 1``, summing
            exactly to ``max_tokens``.
        :raises ValueError: If either argument is not positive.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if block_num <= 0:
            raise ValueError(f"block_num must be positive, got {block_num}")

        n_blocks = min(block_num, max_tokens)
        if n_blocks != block_num:
            logger.warning(
                f"block_num={block_num} exceeds max_tokens={max_tokens}; "
                f"using {n_blocks} block(s) of 1 token so that no MCMC block "
                f"is empty."
            )
        base, remainder = divmod(max_tokens, n_blocks)
        return [base + 1 if i < remainder else base for i in range(n_blocks)]

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
        ``mcmc_steps`` proposals are evaluated with Metropolis-Hastings
        acceptance.  Each proposal regenerates the suffix of the current
        chain state from its own cut point; proposals are batched across a run
        of rejections (which leave the state unchanged) and re-generated after
        every acceptance, so the chain is a valid sequential MH chain rather
        than a set of proposals scored against a state they never saw.

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

        self._require_pad_id_for_batched_proposals(steps)

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
        # Context boundary.
        c = len(prompt_ids)

        # Per-block token counts sum to exactly `max_tok`; none is zero.
        block_sizes = self._block_sizes(max_tok, blocks)

        gen = list(prompt_ids)
        log_probs_norm: List[float] = []
        log_probs_unnorm: List[float] = []
        attempts = 0
        acceptances = 0

        t0 = time.time()

        for block_idx, jump_size in enumerate(block_sizes):
            # Generate one block of tokens with naive temperature sampling
            gen, lp_norm, lp_unnorm = self.naive_temp_generate(
                gen, temp, num_tokens=jump_size,
            )
            log_probs_norm.extend(lp_norm)
            log_probs_unnorm.extend(lp_unnorm)

            # Cut points idx ~ Uniform[c, t-1] are drawn once per block; they
            # do not depend on the chain state. The continuations they
            # generate do.
            t = len(gen)
            indices = [random.randint(c, t - 1) for _ in range(steps)]

            i = 0
            while i < steps:
                props, lp_props_list, target_lp_props_list = (
                    self._batch_proposals(gen, indices[i:], t, temp)
                )

                for k, idx in enumerate(indices[i:]):
                    attempts += 1
                    i += 1
                    s = len(props[k])

                    lp_cur = log_probs_norm[idx - c: s - c]
                    target_lp_cur = log_probs_unnorm[idx - c: s - c]

                    # Metropolis-Hastings acceptance criterion
                    log_r = (
                        sum(target_lp_props_list[k]) + sum(lp_cur)
                        - sum(target_lp_cur) - sum(lp_props_list[k])
                    )

                    if np.random.rand() < np.exp(min(log_r, 0.0)):
                        acceptances += 1
                        gen = list(props[k])
                        log_probs_norm[idx - c:] = list(lp_props_list[k])
                        log_probs_unnorm[idx - c:] = list(
                            target_lp_props_list[k],
                        )
                        # The rest of this batch is stale; re-batch.
                        break

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
        # Strip the CLS token only when one was prepended.
        return gen[strip:], info

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
        level).  Approximates sampling from *p^infinity*.  Proposals follow the
        same chain discipline: batched across rejections, re-generated from the
        state left behind by every accepted swap.

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

        self._require_pad_id_for_batched_proposals(steps)

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

        block_sizes = self._block_sizes(max_tok, blocks)

        gen = list(prompt_ids)
        log_probs_norm: List[float] = []
        log_probs_unnorm: List[float] = []
        attempts = 0
        acceptances = 0

        t0 = time.time()

        for block_idx, jump_size in enumerate(block_sizes):
            gen, lp_norm, lp_unnorm = self.naive_temp_generate(
                gen, temp, num_tokens=jump_size,
            )
            log_probs_norm.extend(lp_norm)
            log_probs_unnorm.extend(lp_unnorm)

            # Same chain discipline as mcmc_power_sample: cut points once per
            # block, continuations re-derived from the state after each swap.
            t = len(gen)
            indices = [random.randint(c, t - 1) for _ in range(steps)]

            i = 0
            while i < steps:
                props, lp_props_list, target_lp_props_list = (
                    self._batch_proposals(gen, indices[i:], t, temp)
                )

                for k, idx in enumerate(indices[i:]):
                    attempts += 1
                    i += 1
                    s = len(props[k])

                    target_lp_cur = log_probs_unnorm[idx - c: s - c]

                    # Deterministic: accept if trajectory probability improves
                    log_r = sum(target_lp_props_list[k]) - sum(target_lp_cur)

                    if log_r > 0:
                        acceptances += 1
                        gen = list(props[k])
                        log_probs_norm[idx - c:] = list(lp_props_list[k])
                        log_probs_unnorm[idx - c:] = list(
                            target_lp_props_list[k],
                        )
                        # The rest of this batch is stale; re-batch.
                        break

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
        # Strip the CLS token only when one was prepended.
        return gen[strip:], info

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

        # DECISION plan_2026-06-16_535b4f02/D-001: swap in a `dataclasses.
        # replace` copy and restore in `finally`, not an in-place mutate-then
        # -restore — the in-place form is not exception-safe. See decisions.md D-001.
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
            # Restore unconditionally so config is unchanged even if the
            # loop raised.
            self.config = original

        elapsed = time.time() - t0
        info = {
            "elapsed_s": elapsed,
            "tokens_generated": max_tokens,
            "tok_per_s": max_tokens / max(elapsed, 0.01),
        }
        # Strip the CLS token only when one was prepended.
        return ids[strip:], info

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
