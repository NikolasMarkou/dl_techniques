"""Configuration for the general-purpose power-sampling inference engine.

:class:`PowerSamplingConfig` is a plain ``@dataclass`` (it is *not* a Keras
component — no ``@keras.saving.register_keras_serializable``, no ``build``/
``get_config``). It carries the sampling hyperparameters plus the tokenizer/model
identity fields the engine needs but cannot read off a generic tokenizer.
"""

from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class PowerSamplingConfig:
    """Configuration for power-sampling inference.

    The algorithm hyperparameters are kept verbatim from the original
    CliffordNet implementation. The tokenizer/model identity fields are
    **generalized**: they default to empty/``None`` so the engine drives any
    LLM/VLM with any tokenizer, instead of hard-coding GPT-2/CliffordNet IDs.

    # DECISION plan_2026-06-16_535b4f02/D-001: identity fields are generalized
    # to empty-set / None defaults (C7/G1/G2). Do NOT restore the GPT-2 defaults
    # {50257..50260}/cls=50257/pad=50260/ctx_len=511 — that would re-couple this
    # general package to CliffordNet and silently mis-mask any other model.
    # CliffordNet behavior is preserved by the consumer passing those IDs
    # explicitly at the call site (see decisions.md D-001 / infer_cliffordnet_nlp.py).

    :param temperature: Controls the power-distribution exponent.
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
    :param special_token_ids: Token IDs to never generate. Empty by default;
        supply the model's special-token IDs to mask them out.
    :param cls_token_id: Token ID prepended to every prompt. If ``None`` (the
        default), no CLS token is prepended and none is stripped from the output.
    :param pad_token_id: Token ID used for right-padding to a fixed length.
        Required when ``ctx_len`` is set (fixed-shape models); otherwise unused.
    :param ctx_len: Context window length for fixed-shape forward passes. If
        ``None`` (the default), the model receives the variable-length sequence
        unpadded; if set, sequences are right-padded to this length with
        ``pad_token_id`` for fixed-shape models such as CliffordNetLM.
    """

    # Algorithm hyperparameters (kept verbatim from source).
    temperature: float = 0.25
    mcmc_steps: int = 10
    block_num: int = 16
    max_tokens: int = 512
    top_p: float = 0.92
    repetition_penalty: float = 1.3
    repetition_window: int = 50

    # Tokenizer/model identity — generalized (no GPT-2 defaults).
    special_token_ids: Set[int] = field(default_factory=set)  # C7: empty default
    cls_token_id: Optional[int] = None  # G2: None => no CLS prepend
    pad_token_id: Optional[int] = None  # C7: required for fixed-shape padding
    ctx_len: Optional[int] = None  # G1: None => variable-length forward


__all__ = ["PowerSamplingConfig"]
