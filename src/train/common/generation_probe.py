"""Unified ``GenerationProbeCallback`` for autoregressive CLM training probes.

This is the single home for the generation-probe callback that previously
existed as five near-identical copies under ``src/train/`` (gpt2,
wave_field_llm, and three cliffordnet NLP trainers). It periodically generates
sample text from the model during training, logs the result (and tok/s), and
optionally appends a JSONL probe record.

Design contract (pinned by ``tests/test_train/test_common_generation_probe.py``):

1. SEEDED RNG. Sampling uses ``np.random.default_rng(seed)`` (``seed=None`` ->
   a fresh non-deterministic generator). This is a deliberate improvement over
   the legacy copies A-D which called the global ``np.random.choice``.

2. PADDING-AGNOSTIC. The class NEVER reads a model-specific output key
   (``["logits"]``) and NEVER indexes a sequence position itself. It passes the
   UNPADDED ``ctx`` slice (``ids[-ctx_length:]``) to a caller-supplied
   ``logits_fn`` closure and consumes the returned ``float32[vocab]``
   next-position vector directly. Position handling and any model-output
   adaptation (e.g. probabilities -> log-probs) live in the caller closure.
   (D-002 / D-003)

3. THREE REPETITION-PENALTY MODES selected by ``repetition_penalty_mode``:
     - ``"divide"``     : ``x / penalty``                (copies A/B)
     - ``"sign_aware"`` : ``x / p if x > 0 else x * p``  (copies C/D)
     - ``"multiply"``   : ``x * penalty``                (copy E, log-prob domain)

All five legacy copies' divergence axes are parameterised here: external
``step_counter`` (E), ``stop_on_eot`` (E), ``seed`` (E), ``pad_token_id``,
``repetition_penalty_mode`` (C/D/E), ``trigger_requires_positive_step`` (E's
``step > 0`` resume guard), and ``gc_on_probe`` (E's ``gc.collect()``). The
``_post_generate_hook`` method stays overridable / monkeypatchable.
"""

import os
import gc
import json
import time
from typing import Callable, List, Optional

import keras
import numpy as np

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Repetition-penalty mode dispatch
# ---------------------------------------------------------------------


def _apply_rep_penalty(value: float, penalty: float, mode: str) -> float:
    """Apply one of the three documented repetition-penalty modes.

    Args:
        value: the current logit (or log-prob) for a repeated token.
        penalty: the repetition-penalty magnitude.
        mode: one of ``"divide"`` | ``"sign_aware"`` | ``"multiply"``.

    Returns:
        The penalised value.
    """
    if mode == "divide":
        return value / penalty
    if mode == "sign_aware":
        return value / penalty if value > 0 else value * penalty
    if mode == "multiply":
        return value * penalty
    raise ValueError(f"unknown repetition_penalty_mode: {mode!r}")


# ---------------------------------------------------------------------
# Generation Probe Callback
# ---------------------------------------------------------------------


class GenerationProbeCallback(keras.callbacks.Callback):
    """Generate sample text periodically during training to track quality."""

    def __init__(
        self,
        logits_fn: Callable[[np.ndarray], np.ndarray],
        probe_every_steps: int = 25000,
        prompts: Optional[List[str]] = None,
        encoding_name: Optional[str] = "gpt2",
        encoder: Optional[object] = None,
        max_tokens: int = 100,
        temperature: float = 0.85,
        top_p: float = 0.92,
        repetition_penalty: float = 1.3,
        repetition_penalty_mode: str = "divide",
        eot_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        ctx_length: int = 511,
        stop_on_eot: bool = False,
        save_dir: Optional[str] = None,
        initial_step: int = 0,
        step_counter: Optional[object] = None,
        seed: Optional[int] = None,
        trigger_requires_positive_step: bool = False,
        gc_on_probe: bool = False,
    ):
        """Construct the unified generation-probe callback.

        Args:
            logits_fn: closure mapping ``int32[1, seq]`` -> ``float32[vocab]``;
                returns the next-position vector. Owns all model-output
                adaptation (position indexing, padding, prob->log). The class
                never reads ``self.model[...]`` directly.
            probe_every_steps: run probes every N optimizer steps.
            prompts: prompt strings; defaults to a 3-prompt set.
            encoding_name: tiktoken encoding name; only used when ``encoder``
                is None. Pass None when ``encoder`` is supplied.
            encoder: pre-built encoder (tiktoken ``Encoding`` or compatible).
                When given, used INSTEAD of ``tiktoken.get_encoding``.
            max_tokens: per-prompt generation budget.
            temperature: sampling temperature.
            top_p: nucleus-sampling cumulative-probability cutoff.
            repetition_penalty: penalty magnitude for recently-seen tokens.
            repetition_penalty_mode: ``"divide"`` | ``"sign_aware"`` |
                ``"multiply"``.
            eot_token_id: end-of-text id; defaults to ``encoder.eot_token``.
            pad_token_id: padding id; defaults to ``eot_token_id``.
            ctx_length: unpadded context slice length (``ids[-ctx_length:]``).
            stop_on_eot: when True, EOT is sampleable and generation breaks on
                it; when False, EOT is suppressed (-1e9) and the full budget
                runs.
            save_dir: when set, append JSONL probe records under
                ``save_dir/generation_probes/probes.jsonl``.
            initial_step: starting value for the internal step counter.
            step_counter: external counter exposing ``.value``; when None an
                internal counter (incremented per batch) is used.
            seed: seed for ``np.random.default_rng``; None -> fresh generator.
            trigger_requires_positive_step: when True, suppress the probe at
                step 0 (resume guard); trigger requires ``step > 0``.
            gc_on_probe: when True, run ``gc.collect()`` after each probe round.
        """
        super().__init__()
        self.logits_fn = logits_fn
        self.probe_every_steps = probe_every_steps
        self.prompts = prompts or [
            "The United States of America is a",
            "In mathematics, a prime number is",
            "Albert Einstein was born in",
        ]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_mode = repetition_penalty_mode
        self.stop_on_eot = stop_on_eot
        self.ctx_length = ctx_length
        self.trigger_requires_positive_step = trigger_requires_positive_step
        self.gc_on_probe = gc_on_probe

        # Seeded RNG (seed may be None -> fresh non-deterministic generator).
        self._rng = np.random.default_rng(seed)

        # External step counter (copy E) vs internal counter (copies A-D).
        self._counter = step_counter
        self._global_step = initial_step

        # Encoder: pre-built (test / caller) takes precedence over tiktoken.
        if encoder is not None:
            self._enc = encoder
        else:
            import tiktoken
            self._enc = tiktoken.get_encoding(encoding_name)

        self._eot_id = int(
            eot_token_id if eot_token_id is not None else self._enc.eot_token
        )
        # Distinct from EOT so the model sees a true PAD where context is
        # short; defaults to EOT for back-compat.
        self._pad_id = int(
            pad_token_id if pad_token_id is not None else self._eot_id
        )

        self._log_path = None
        if save_dir:
            probe_dir = os.path.join(save_dir, "generation_probes")
            os.makedirs(probe_dir, exist_ok=True)
            self._log_path = os.path.join(probe_dir, "probes.jsonl")

        logger.info(
            f"GenerationProbeCallback: {len(self.prompts)} prompts, "
            f"every {probe_every_steps} steps, "
            f"max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, "
            f"mode={repetition_penalty_mode}, pad_id={self._pad_id}, "
            f"eot_id={self._eot_id}, stop_on_eot={stop_on_eot}"
        )

    # ------------------------------------------------------------------
    # Step triggering
    # ------------------------------------------------------------------

    def _current_step(self) -> int:
        if self._counter is not None:
            return int(self._counter.value)
        return self._global_step

    def on_train_batch_end(self, batch, logs=None):
        if self._counter is None:
            self._global_step += 1
        step = self._current_step()
        triggers = (not self.trigger_requires_positive_step or step > 0)
        if triggers and step % self.probe_every_steps == 0:
            self._run_probes(logs)

    # ------------------------------------------------------------------
    # Core generation (testable, model-output-agnostic)
    # ------------------------------------------------------------------

    def _generate_token_ids(self, prompt_ids: List[int]) -> List[int]:
        """Autoregressively sample token ids from ``logits_fn``.

        Returns ONLY the freshly-generated ids (the prompt is excluded). This
        is the testable core asserted byte-for-byte against the reference loop.
        """
        ids = list(prompt_ids)
        # Block special tokens from sampling: tiktoken's `decode` raises on any
        # id at or above the encoder's `n_vocab` (the base vocab end) because
        # the reserved special-token ids live outside the BPE table. Suppress
        # them at logits time so generation is restricted to decodable tokens.
        vocab_base = self._enc.n_vocab
        special_ids = list(range(vocab_base, max(vocab_base + 1, 50261)))

        for _ in range(self.max_tokens):
            ctx = ids[-self.ctx_length:]
            # DECISION plan_2026-06-02_cc4d4e14/D-002: pass the UNPADDED ctx to
            # the closure, which returns the next-position [vocab] vector. The
            # common class never indexes a sequence position (no `[0,-1,:]`,
            # no `[0,real-1,:]` here) and never reads a model output key — that
            # is the caller closure's job (see decisions.md D-002/D-003).
            logits = np.asarray(
                self.logits_fn(np.array([ctx], dtype="int32")),
                dtype="float32",
            ).copy()

            # Ordering MUST match the test oracle (_reference_generate):
            # EOT/special suppression -> rep penalty -> temperature -> nucleus.
            if not self.stop_on_eot:
                logits[self._eot_id] = -1e9
            for sid in special_ids:
                if sid < logits.shape[0]:
                    logits[sid] = -1e9

            # DECISION plan_2026-06-02_cc4d4e14/D-003: rep-penalty MODE is
            # parameterised (divide/sign_aware/multiply); the prob->log adapter
            # for copy E lives in the caller closure, NOT here. Do not hardcode
            # a single mode (that was the C/D/E divergence) — see decisions.md.
            for t in set(ids[-50:]):
                if t == self._eot_id:
                    continue
                logits[t] = _apply_rep_penalty(
                    float(logits[t]),
                    self.repetition_penalty,
                    self.repetition_penalty_mode,
                )

            logits = logits / self.temperature

            sorted_idx = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx]
            probs = np.exp(sorted_logits - sorted_logits[0])
            probs /= probs.sum()
            cutoff = int(np.searchsorted(np.cumsum(probs), self.top_p)) + 1
            top_idx = sorted_idx[:cutoff]
            top_probs = probs[:cutoff]
            top_probs /= top_probs.sum()

            next_token = int(top_idx[self._rng.choice(len(top_idx), p=top_probs)])
            ids.append(next_token)

            if self.stop_on_eot and next_token == self._eot_id:
                break

        return ids[len(prompt_ids):]

    # ------------------------------------------------------------------
    # Runtime probe path (decode + log)
    # ------------------------------------------------------------------

    def _generate(self, prompt: str) -> str:
        """Encode a prompt, generate, and decode to text (with fallback)."""
        prompt_ids = self._enc.encode(prompt)
        generated = self._generate_token_ids(prompt_ids)
        ids = list(prompt_ids) + generated

        # `errors="replace"`-style defensive backstop: if a tokenizer surprise
        # (e.g. a partial multi-byte BPE chunk at the tail) sneaks through, we
        # want a string back, not a probe-side crash that kills training.
        try:
            return self._enc.decode(ids)
        except (KeyError, UnicodeDecodeError) as e:
            logger.warning(f"Probe decode fell back due to: {e}")
            return self._enc.decode([t for t in ids if t < self._enc.n_vocab])

    def _run_probes(self, logs=None):
        step = self._current_step()
        train_loss = logs.get("loss", 0.0) if logs else 0.0

        logger.info(f"{'=' * 50}")
        logger.info(
            f"Generation probe @ step {step:,} (train_loss={train_loss:.4f})"
        )
        logger.info(f"{'=' * 50}")

        probe_results = {
            "step": step,
            "train_loss": float(train_loss),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generations": [],
        }

        for prompt in self.prompts:
            t0 = time.time()
            text = self._generate(prompt)
            elapsed = time.time() - t0
            tokens_generated = (
                len(self._enc.encode(text)) - len(self._enc.encode(prompt))
            )

            gen_entry = {
                "prompt": prompt,
                "output": text[:500],
                "tokens": tokens_generated,
                "time_s": round(elapsed, 2),
                "tok_per_s": round(tokens_generated / max(elapsed, 0.01), 1),
            }
            probe_results["generations"].append(gen_entry)

            logger.info(f'Prompt: "{prompt}"')
            logger.info(f"Output: {text[:300]}")
            logger.info(
                f"({tokens_generated} tokens, {elapsed:.1f}s, "
                f"{gen_entry['tok_per_s']} tok/s)"
            )
            logger.info("")

        # Extension point for probe-time aggregate metrics (Self-BLEU,
        # distinct-2, mean tok/s). Default is a no-op; trainers bind a concrete
        # hook on the probe instance (copy E monkeypatches it).
        self._post_generate_hook(probe_results)

        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(probe_results, ensure_ascii=False) + "\n")

        # Reclaim Python wrappers + tf-eager intermediates accumulated by the
        # autoregressive `logits_fn(...)` calls (copy E's leak mitigation).
        if self.gc_on_probe:
            gc.collect()

    def _post_generate_hook(self, results: dict) -> None:
        """Override or rebind on the instance for custom probe-time analysis.

        Default: no-op.
        """
        return None
