"""CliffordNet NLP Pre-training with Hierarchical Routing Head (CLM).

Sibling of ``train_cliffordnet_nlp.py`` that trains
:class:`CliffordNetLMRouting` — a CliffordNet causal LM whose final vocab
projection is a :class:`RoutingProbabilitiesLayer` (probabilistic
hierarchical tree) instead of a Dense softmax.

Two routing modes are exposed via ``--routing-mode``:

- ``trainable`` (default): learnable affine projection to log2(padded_vocab)
  decisions. ~3000x fewer params than the Dense head at vocab=50K.
- ``deterministic``: parameter-free cosine-basis projection. Useful for
  ablation; expressive ceiling of 16 decisions for ~50K vocab is tight.

Because the routing layer outputs probabilities (not logits), losses are
constructed with ``from_logits=False``.

Usage::

    # Trainable routing on imdb (smoke test)
    MPLBACKEND=Agg python -m train.cliffordnet.train_cliffordnet_nlp_routing \\
        --variant nano --routing-mode trainable \\
        --dataset-source tfds --dataset-name imdb_reviews \\
        --max-samples 100 --epochs 1 --batch-size 4 --gpu 0

    # Deterministic ablation
    MPLBACKEND=Agg python -m train.cliffordnet.train_cliffordnet_nlp_routing \\
        --variant nano --routing-mode deterministic ...

Resume from checkpoint::

    python -m train.cliffordnet.train_cliffordnet_nlp_routing \\
        --resume results/cliffordnet_nlp_routing_*/checkpoints/step_0050000.keras

================================================================================
KNOWN MODELING LIMITATIONS (NOT BUGS — INHERENT TO THE ROUTING-HEAD DESIGN)
================================================================================

These are deliberate trade-offs, not defects. They are documented here so
readers don't mistake low absolute perplexity for a training issue.

1. Routing-head expressive ceiling (16-bit channel for 50K vocab)
-----------------------------------------------------------------
``RoutingProbabilitiesLayer`` makes ``d = ceil(log2(N))`` binary decisions
to discriminate ``N`` classes. For vocab=50,261 -> padded to 65,536 ->
``d = 16`` decisions.

- Output manifold dimensionality: 16 (vs. ~768 effective dims for a Dense
  head). The 65,536-dim probability vector lies on a 16-dim sigmoid
  manifold, *not* on a free 65,535-dim simplex.
- Parameter count: ~12K at D=768 (vs. ~38.6M for a Dense head). 3,000x
  compression — and the lost capacity is *real* capacity, not redundancy.
- Sibling entanglement: two leaves sharing ancestor bits have probabilities
  that co-move under any change to those decisions. The model cannot raise
  P(token_a) without also moving P of every token sharing ancestor bits
  with token_a, regardless of context.
- Trainable mode does NOT lift this ceiling: it learns *better* 16
  directions in feature space, but the output manifold is still 16-dim.
- Expected impact: a ~0.3-0.6 nats CE gap above a tied-embedding Dense
  baseline at convergence, growing with vocabulary size.
- Per-token CE leakage from sigmoid clipping (eps=1e-7, d=16):
  ~1.6e-6 nats — negligible at the per-token level.

2. Arbitrary token-id-to-leaf mapping (the "leaf-arrangement penalty")
----------------------------------------------------------------------
Tokens map to leaf positions by integer ID. Tiktoken gpt2 IDs are assigned
in BPE merge-frequency order — essentially incidental from a semantic
standpoint. ID 1234 is not semantically near ID 1235.

- Two leaves' probability coupling is determined by their lowest-common-
  ancestor depth. Random ID-to-leaf assignment means semantically-related
  tokens (e.g. "mat", "floor", "chair", "bed") sit at unrelated leaves
  with shallow LCAs, forcing the model to express common context-
  conditional distributions through tree branches that don't align with
  semantics. Every prediction pays a KL cost.
- BPE merge order is *worse* than uniformly random in practice: it
  accidentally clusters surface-form variants (" the"/"the"/"The") but
  not semantic roles.
- Fixable cheaply: a static permutation derived from a Huffman tree over
  unigram frequencies (~0.15 nats recovered) or recursive spectral
  clustering of token co-occurrence (~0.4 nats recovered) plugs in as a
  fixed lookup table at the routing-layer boundary, with zero runtime
  overhead. Currently neither is implemented — the trivial ID->leaf map
  is the worst end of the design space.
- A future change should add a ``--vocab-permutation {none,huffman,
  spectral}`` flag and precompute the permutation from the training
  corpus before the first epoch.

3. Routing-tree gradient asymmetry (minor)
------------------------------------------
For a target leaf ``i``, the gradient w.r.t. decision logit ``z_k`` has
magnitude that scales with the number of leaves in the sub-tree rooted at
that decision. Root decisions see gradient aggregated over all 65K
leaves; near-leaf decisions see gradient from only 2 leaves. This
produces a systematic *gradient-magnitude imbalance by tree depth* that
slows updates to deep decisions. Layer-wise LR scaling (deeper = higher
LR) would help; not currently implemented.

================================================================================
"""

import os
import gc
import json
import glob
import time
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import tensorflow as tf
import tiktoken
from keras import initializers

from train.common import setup_gpu
from train.common import StepCheckpointCallback
from train.common.evaluation import generate_training_curves
from train.common.nlp import (
    create_tokenizer,
    load_text_dataset,
    preprocess_clm_dataset,
    create_warmup_lr_schedule,
    create_nlp_callbacks,
    estimate_clm_steps_per_epoch,
    build_clm_metrics,
    prepare_dict_keyed_compile,
    augment_probe_results,
)
from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
)
from dl_techniques.layers.activations.routing_probabilities import (
    RoutingProbabilitiesLayer,
)
from dl_techniques.layers.embedding.hierarchical_codebook_embedding import (
    HierarchicalCodebookEmbedding,
)
from dl_techniques.layers.embedding.albert_factorized_embedding import (
    AlbertFactorizedEmbedding,
)
from dl_techniques.models.cliffordnet import CliffordNetLMRouting
from dl_techniques.utils.logger import logger
from dl_techniques.datasets.nlp import load_wikipedia_train_val
from dl_techniques.losses import MaskedCausalLMLoss, FocalCausalLMLoss


_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Configuration for CliffordNet routing-head NLP CLM pre-training."""

    # Model
    variant: str = "nano"
    channels: int = 128
    depth: int = 12
    shifts: List[int] = field(default_factory=lambda: [1, 2])
    cli_mode: str = "full"
    ctx_mode: str = "diff"
    use_global_context: bool = False
    dropout_rate: float = 0.1
    stochastic_depth_rate: float = 0.1

    # Routing head
    routing_mode: str = "trainable"

    # Token (input) embedding strategy: "hce" (default, parameter-efficient
    # additive codebook), "albert" (factorized inner-dim projection),
    # "dense" (legacy keras.layers.Embedding). See lm_routing.py for the
    # full design rationale and trade-offs.
    input_embedding: str = "hce"
    embedding_bottleneck_dim: Optional[int] = None  # ALBERT: k
    hce_num_chunks: int = 2                          # HCE: K
    hce_chunk_bits: Optional[int] = None             # HCE: log2(M); auto if None

    # Tokenizer (Tiktoken gpt2 encoding -- 50,257 base + 4 special)
    vocab_size: int = 50261
    max_seq_length: int = 512
    encoding_name: str = "gpt2"
    cls_token_id: int = 50257
    sep_token_id: int = 50258
    pad_token_id: int = 50259
    mask_token_id: int = 50260

    # Training
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Loss: "ce" (default) or "focal"
    loss_type: str = "ce"
    focal_gamma: float = 1.0
    label_smoothing: float = 0.0

    # Paths
    save_dir: str = "results/cliffordnet_nlp_routing"

    # Data source: "huggingface" or "tfds"
    dataset_source: str = "huggingface"

    # TFDS settings
    dataset_name: str = "imdb_reviews"
    max_samples: Optional[int] = 10000

    # HuggingFace / Wikipedia settings
    hf_cache_dir: str = "/media/arxwn/data0_4tb/datasets/wikipedia"
    hf_wikipedia_config: str = "20231101.en"
    # DECISION D-003: 0 → packed CLM uses every token; stub filtering only
    # loses tokens. Pass 500+ for per-doc consumers (MLM, classification).
    min_article_length: int = 0
    val_fraction: float = 0.02
    max_val_samples: int = 5000
    max_train_samples: Optional[int] = None
    # DECISION D-002: parallel tokenization shards + per-epoch reshuffle.
    shuffle_shards: int = 4

    # DECISION D-006: end-to-end seed plumbing. On --resume the train loop
    # derives data_seed = seed + initial_step so resumed runs see new
    # article ordering instead of replaying the first N chunks.
    seed: int = 42

    # Checkpointing & analysis (step-based for large datasets).
    # analyze_every_steps default 0 = OFF: spectral analysis blocks the
    # training thread for multiple seconds. Enable explicitly if needed.
    checkpoint_every_steps: int = 25000
    analyze_every_steps: int = 0
    max_checkpoints: int = 3

    # Optional override of steps_per_epoch (for streaming HF data this is
    # the only way to get a correct LR schedule horizon).
    steps_per_epoch: Optional[int] = None

    # Resume from checkpoint
    resume_from: Optional[str] = None

    # Architecture overrides for named variants. Only keys present here are
    # forwarded to `from_variant(...)`; unset keys leave the variant default.
    arch_overrides: Dict[str, Any] = field(default_factory=dict)

    # Probe RNG seed
    probe_seed: int = 42

    # Generation probes
    probe_prompts: List[str] = field(default_factory=lambda: [
        "The United States of America is a",
        "In mathematics, a prime number is",
        "Albert Einstein was born in",
    ])
    probe_max_tokens: int = 100
    probe_temperature: float = 0.85
    probe_top_p: float = 0.92
    probe_repetition_penalty: float = 1.3


# ---------------------------------------------------------------------------
# Shared step counter
# ---------------------------------------------------------------------------


class StepCounter(keras.callbacks.Callback):
    """Owns the global training-step counter; other callbacks read from it.

    Registered first so its on_train_batch_end runs before any consumer.
    Without a single owner, multiple callbacks each maintain their own
    counter and silently drift if one misses a batch.
    """

    def __init__(self, initial_step: int = 0):
        super().__init__()
        self.value: int = initial_step

    def on_train_batch_end(self, batch, logs=None):
        self.value += 1


# ---------------------------------------------------------------------------
# Generation Probe Callback
# ---------------------------------------------------------------------------


class GenerationProbeCallback(keras.callbacks.Callback):
    """Generate sample text periodically to track quality.

    Model output is *probabilities* (sum=1, in [eps, 1-eps]) rather
    than logits, so the generation logic uses log-probs for nucleus
    sampling.
    """

    def __init__(
        self,
        step_counter: StepCounter,
        probe_every_steps: int = 25000,
        prompts: Optional[List[str]] = None,
        encoding_name: str = "gpt2",
        max_tokens: int = 100,
        temperature: float = 0.85,
        top_p: float = 0.92,
        repetition_penalty: float = 1.3,
        eot_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        ctx_length: int = 511,
        save_dir: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()
        self._counter = step_counter
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
        self._rng = np.random.default_rng(seed)

        self._enc = tiktoken.get_encoding(encoding_name)
        self._eot_id = int(
            eot_token_id if eot_token_id is not None else self._enc.eot_token
        )
        # Distinct from EOT so the model sees a true PAD where context is
        # short. Defaults to EOT for back-compat.
        self._pad_id = int(pad_token_id if pad_token_id is not None
                           else self._eot_id)
        self._ctx_len = ctx_length

        self._log_path = None
        if save_dir:
            probe_dir = os.path.join(save_dir, "generation_probes")
            os.makedirs(probe_dir, exist_ok=True)
            self._log_path = os.path.join(probe_dir, "probes.jsonl")

        logger.info(
            f"GenerationProbeCallback: {len(self.prompts)} prompts, "
            f"every {probe_every_steps} steps, "
            f"max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, "
            f"pad_id={self._pad_id}, eot_id={self._eot_id}"
        )

    def on_train_batch_end(self, batch, logs=None):
        step = self._counter.value
        if step > 0 and step % self.probe_every_steps == 0:
            self._run_probes(logs)

    def _run_probes(self, logs=None):
        step = self._counter.value
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
            text, tokens_generated = self._generate(prompt)
            elapsed = time.time() - t0

            gen_entry = {
                "prompt": prompt,
                "output": text[:500],
                "tokens": tokens_generated,
                "time_s": round(elapsed, 2),
                "tok_per_s": round(
                    tokens_generated / max(elapsed, 0.01), 1
                ),
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
        # distinct-2, mean tok/s). Default is a no-op; trainers bind a
        # concrete hook (e.g. ``augment_probe_results``) on the probe
        # instance.
        self._post_generate_hook(probe_results)

        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(probe_results, ensure_ascii=False) + "\n")

        # Reclaim Python wrappers + tf-eager intermediates accumulated by
        # the 300 autoregressive `model(...)` calls; without this, ~70 MB
        # per probe event leaked into RSS over the run.
        gc.collect()

    def _post_generate_hook(self, results: dict) -> None:
        """Override or rebind on the instance for custom probe-time
        analysis. Default: no-op.
        """
        return None

    def _generate(self, prompt: str) -> Tuple[str, int]:
        """Autoregressive generation. Model outputs probabilities, so we
        take ``log(p)`` and treat it as logits for the standard pipeline.

        Returns ``(decoded_text, tokens_generated)``. Generation stops on
        EOT. Context is trimmed to its actual length when shorter than
        ``ctx_len`` so we don't waste compute on pad positions.
        """
        ids = self._enc.encode(prompt)
        # Block special tokens from sampling: tiktoken's `decode` raises on
        # any id at or above the encoder's `n_vocab` (the base vocab end)
        # because the 4 reserved special-token ids in this codebase
        # (50257..50260) live outside the BPE table. Suppress them at
        # logits time so generation is restricted to decodable tokens.
        special_ids = [
            i for i in range(self._enc.n_vocab, max(self._enc.n_vocab + 1, 50261))
        ]
        len_initial = len(ids)
        ctx_len = self._ctx_len

        for _ in range(self.max_tokens):
            ctx = ids[-ctx_len:]
            real = len(ctx)
            # Trim to actual length — model.call uses ops.shape(input_ids)[1]
            # for positional embeddings, so it accepts variable seq length.
            # When real == ctx_len, no padding; when real < ctx_len, pad
            # with PAD (not EOT) so the model sees true padding.
            if real < ctx_len:
                padded = ctx + [self._pad_id] * (ctx_len - real)
                in_ids = np.asarray([padded], dtype="int32")
            else:
                in_ids = np.asarray([ctx], dtype="int32")
            out = self.model(in_ids, training=False)
            probs = out["logits"][0, real - 1, :].numpy()
            # Convert probabilities to log-probs (logit-equivalent).
            # log(p) is always <= 0 after clip — repetition penalty has only
            # one branch.
            logits = np.log(np.clip(probs, 1e-12, 1.0))
            for sid in special_ids:
                if sid < logits.shape[0]:
                    logits[sid] = -1e9

            for t in set(ids[-50:]):
                # Penalize repeats by pushing log-probs further negative.
                logits[t] *= self.repetition_penalty

            logits /= self.temperature

            sorted_idx = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx]
            probs_norm = np.exp(sorted_logits - sorted_logits[0])
            probs_norm /= probs_norm.sum()
            cutoff = np.searchsorted(np.cumsum(probs_norm), self.top_p) + 1
            top_idx = sorted_idx[:cutoff]
            top_probs = probs_norm[:cutoff]
            top_probs /= top_probs.sum()

            next_token = int(top_idx[
                self._rng.choice(len(top_idx), p=top_probs)
            ])
            ids.append(next_token)
            if next_token == self._eot_id:
                break

        # `errors="replace"` is a defensive backstop: if a tokenizer surprise
        # (e.g. partial multi-byte BPE chunk at the tail) sneaks through,
        # we want a string back, not a probe-side crash that kills training.
        try:
            return self._enc.decode(ids), len(ids) - len_initial
        except (KeyError, UnicodeDecodeError) as e:
            logger.warning(f"Probe decode fell back due to: {e}")
            return self._enc.decode(
                [t for t in ids if t < self._enc.n_vocab],
            ), len(ids) - len_initial


# ---------------------------------------------------------------------------
# Model Creation & Resume
# ---------------------------------------------------------------------------


def _extract_step_from_checkpoint(path: str) -> int:
    """Extract training step from checkpoint filename."""
    import re
    basename = os.path.basename(path)
    match = re.search(r"step_(\d+)", basename)
    return int(match.group(1)) if match else 0


def load_model_from_checkpoint(
    path: str,
) -> Tuple[CliffordNetLMRouting, int]:
    """Load a CliffordNetLMRouting from a ``.keras`` checkpoint."""
    logger.info(f"Resuming from checkpoint: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={
            "CliffordNetLMRouting": CliffordNetLMRouting,
            "CausalCliffordNetBlock": CausalCliffordNetBlock,
            "RoutingProbabilitiesLayer": RoutingProbabilitiesLayer,
            "HierarchicalCodebookEmbedding": HierarchicalCodebookEmbedding,
            "AlbertFactorizedEmbedding": AlbertFactorizedEmbedding,
            "MaskedCausalLMLoss": MaskedCausalLMLoss,
            "FocalCausalLMLoss": FocalCausalLMLoss,
        },
    )
    step = _extract_step_from_checkpoint(path)
    logger.info(
        f"Loaded model: {model.count_params():,} params, "
        f"resumed at step {step:,}"
    )
    return model, step


def create_model(config: TrainingConfig) -> CliffordNetLMRouting:
    """Create and build a CliffordNetLMRouting from training configuration."""
    logger.info(
        f"Creating CliffordNetLMRouting-{config.variant.upper()} "
        f"(routing_mode={config.routing_mode})..."
    )

    embedding_kwargs: Dict[str, Any] = {
        "input_embedding": config.input_embedding,
        "embedding_bottleneck_dim": config.embedding_bottleneck_dim,
        "hce_num_chunks": config.hce_num_chunks,
        "hce_chunk_bits": config.hce_chunk_bits,
    }

    if config.variant in CliffordNetLMRouting.MODEL_VARIANTS:
        # Forward any user-specified architecture overrides (channels,
        # depth, shifts, cli_mode, ctx_mode, use_global_context,
        # stochastic_depth_rate). dropout_rate is always forwarded.
        from_variant_kwargs: Dict[str, Any] = {
            "vocab_size": config.vocab_size,
            "max_seq_length": config.max_seq_length,
            "routing_mode": config.routing_mode,
            "dropout_rate": config.dropout_rate,
            **embedding_kwargs,
        }
        from_variant_kwargs.update(config.arch_overrides)
        if config.arch_overrides:
            logger.info(
                f"Variant '{config.variant}' with CLI overrides: "
                f"{config.arch_overrides}"
            )
        model = CliffordNetLMRouting.from_variant(
            config.variant, **from_variant_kwargs,
        )
    else:
        # Custom variant
        model = CliffordNetLMRouting(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            channels=config.channels,
            depth=config.depth,
            shifts=config.shifts,
            cli_mode=config.cli_mode,
            ctx_mode=config.ctx_mode,
            use_global_context=config.use_global_context,
            dropout_rate=config.dropout_rate,
            stochastic_depth_rate=config.stochastic_depth_rate,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
            routing_mode=config.routing_mode,
            **embedding_kwargs,
        )

    # Build with dummy forward pass
    dummy = np.random.randint(
        0, config.vocab_size,
        size=(1, config.max_seq_length - 1),
    ).astype("int32")
    model(dummy, training=False)

    model.summary(print_fn=logger.info)
    return model


# ---------------------------------------------------------------------------
# Loss Construction
# ---------------------------------------------------------------------------


def create_loss_fn(config: TrainingConfig) -> keras.losses.Loss:
    """Create the CLM loss function from configuration.

    DECISION D-003: ``from_logits=False`` is required because
    :class:`RoutingProbabilitiesLayer` outputs probabilities in
    ``[eps, 1-eps]`` summing to 1, not raw logits. Passing
    ``from_logits=True`` (the default) would softmax already-softmaxed
    values, producing an incorrect loss.
    """
    if config.loss_type == "focal":
        loss_fn = FocalCausalLMLoss(
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            from_logits=False,
        )
        logger.info(
            f"Loss: FocalCausalLMLoss(gamma={config.focal_gamma}, "
            f"from_logits=False)"
        )
    else:
        loss_fn = MaskedCausalLMLoss(
            label_smoothing=config.label_smoothing,
            from_logits=False,
        )
        logger.info("Loss: MaskedCausalLMLoss(from_logits=False)")
    return loss_fn


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_train_val_datasets(
    config: TrainingConfig,
    preprocessor,
    data_seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Optional[int]]:
    """Load, preprocess, and wrap train/val datasets for dict-output model.

    The training dataset is `.repeat()`'d so multi-epoch runs and explicit
    ``steps_per_epoch`` arithmetic work — required for streaming HF data
    and consistent for TFDS. Validation dataset is NOT repeated; it must
    terminate so val metrics are computed once per epoch.

    :return: ``(train_ds, val_ds, n_train_articles)``. ``n_train_articles``
        is the post-filter article count for the HF Wikipedia path (used by
        ``estimate_clm_steps_per_epoch``); ``None`` for the TFDS path.
    """
    n_train_articles: Optional[int] = None
    if config.dataset_source == "tfds":
        train_ds, val_ds = _load_tfds_datasets(config, preprocessor)
    elif config.dataset_source == "huggingface":
        train_ds, val_ds, n_train_articles = _load_hf_datasets(
            config, preprocessor, data_seed,
        )
    else:
        raise ValueError(
            f"Unknown dataset_source: {config.dataset_source!r}. "
            f"Use 'tfds' or 'huggingface'."
        )

    def _wrap_with_dict_label(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.map(
            lambda x, y: (x, {"logits": y}),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    train_ds = _wrap_with_dict_label(train_ds).repeat()
    val_ds = _wrap_with_dict_label(val_ds)
    return train_ds, val_ds, n_train_articles


def _load_tfds_datasets(config, preprocessor):
    train = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "train", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        load_text_dataset(config.dataset_name, "test", config.max_samples),
        preprocessor, config.max_seq_length, config.batch_size,
    )
    return train, val


def _load_hf_datasets(config, preprocessor, data_seed: int):
    train_raw, val_raw, n_train, _n_val = load_wikipedia_train_val(
        cache_dir=config.hf_cache_dir,
        config_name=config.hf_wikipedia_config,
        min_article_length=config.min_article_length,
        val_fraction=config.val_fraction,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        seed=data_seed,
        num_shards=config.shuffle_shards,
        return_counts=True,
    )
    train = preprocess_clm_dataset(
        train_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    val = preprocess_clm_dataset(
        val_raw, preprocessor,
        config.max_seq_length, config.batch_size,
    )
    return train, val, n_train


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def compile_model(
    model: CliffordNetLMRouting,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> None:
    """Compile with AdamW, warmup + cosine decay, and CLM loss."""
    lr_schedule = create_warmup_lr_schedule(
        config.learning_rate,
        config.num_epochs,
        steps_per_epoch,
        config.warmup_ratio,
    )
    prepare_dict_keyed_compile(model)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            clipnorm=1.0,
        ),
        loss={"logits": create_loss_fn(config)},
        metrics={"logits": build_clm_metrics(config.encoding_name)},
    )
    logger.info(
        f"Compiled: AdamW, peak_lr={config.learning_rate}, "
        f"wd={config.weight_decay}"
    )


def _make_steps_per_epoch(
    config: TrainingConfig, n_train_articles: Optional[int],
) -> int:
    """Resolve steps_per_epoch using the canonical helper (D-001).

    For TFDS this is one-doc-one-window so ``max_samples // batch_size`` is
    the right estimate. For HF Wikipedia we pass the post-filter article
    count from ``load_wikipedia_train_val`` to the chunk-aware helper.
    Users may pass ``--steps-per-epoch`` to override either path.
    """
    if config.dataset_source == "tfds" and config.max_samples and config.steps_per_epoch is None:
        return max(1, config.max_samples // config.batch_size)
    return estimate_clm_steps_per_epoch(
        num_articles=n_train_articles or config.max_train_samples,
        max_seq_length=config.max_seq_length,
        batch_size=config.batch_size,
        override=config.steps_per_epoch,
    )


def train_cliffordnet_nlp_routing(
    config: TrainingConfig,
) -> Tuple[CliffordNetLMRouting, keras.callbacks.History]:
    """Run CliffordNet routing-head NLP CLM pre-training."""
    logger.info("=" * 60)
    logger.info("CliffordNet NLP Routing-head Causal LM Pre-training")
    logger.info("=" * 60)

    tf.random.set_seed(config.seed)
    keras.utils.set_random_seed(config.seed)

    preprocessor = create_tokenizer(
        config.encoding_name,
        config.max_seq_length,
        config.cls_token_id,
        config.sep_token_id,
        config.pad_token_id,
        config.mask_token_id,
    )

    # DECISION D-006: per-resume seed shift. We need initial_step BEFORE
    # building the dataset to derive the data seed, but checkpoint loading
    # also has to happen before fit. Pre-extract initial_step here.
    initial_step = (
        _extract_step_from_checkpoint(config.resume_from)
        if config.resume_from else 0
    )
    data_seed = config.seed + initial_step

    train_dataset, val_dataset, n_train_articles = load_train_val_datasets(
        config, preprocessor, data_seed=data_seed,
    )

    steps_per_epoch = _make_steps_per_epoch(config, n_train_articles)

    if config.resume_from:
        model, initial_step = load_model_from_checkpoint(config.resume_from)
        # Skip recompile so optimizer slots, iteration counter, and the
        # warmup+cosine LR schedule are all preserved. Older checkpoints
        # saved without optimizer state load uncompiled — fall back to
        # recompile and warn that LR/optimizer state is reset.
        if getattr(model, "optimizer", None) is None:
            logger.warning(
                "Checkpoint loaded without compiled optimizer — "
                "recompiling. Optimizer state and LR schedule will reset."
            )
            compile_model(model, config, steps_per_epoch)
        else:
            logger.info(
                "Resumed model arrived compiled; preserving optimizer "
                "state and LR schedule."
            )
    else:
        model = create_model(config)
        compile_model(model, config, steps_per_epoch)

    initial_epoch = initial_step // steps_per_epoch if steps_per_epoch > 0 else 0

    variant_label = config.variant
    if config.variant == "custom":
        variant_label = f"c{config.channels}d{config.depth}"

    # create_nlp_callbacks defaults: monitor='val_loss', patience=15.
    # 1-3 epoch runs effectively never trigger EarlyStopping — accepted.
    callbacks, results_dir = create_nlp_callbacks(
        model_name=f"CliffordNetLMRouting-{variant_label}-{config.routing_mode}",
        results_dir_prefix="cliffordnet_nlp_routing",
        include_analyzer=False,
    )
    # Strip the default per-epoch CSVLogger; StepCheckpointCallback writes
    # both per-step and per-epoch (with val metrics) into a single CSV.
    callbacks = [
        cb for cb in callbacks
        if not isinstance(cb, keras.callbacks.CSVLogger)
    ]

    step_counter = StepCounter(initial_step=initial_step)
    callbacks.insert(0, step_counter)  # must increment before consumers run

    callbacks.append(StepCheckpointCallback(
        save_dir=results_dir,
        step_counter=step_counter,
        save_every_steps=config.checkpoint_every_steps,
        analyze_every_steps=config.analyze_every_steps,
        max_checkpoints=config.max_checkpoints,
        model_name=f"CliffordNetLMRouting-{variant_label}-{config.routing_mode}",
        gc_on_save=True,
        csv_fields=("step", "epoch", "loss", "accuracy", "lr",
                    "val_loss", "val_accuracy"),
    ))

    probe_cb = GenerationProbeCallback(
        step_counter=step_counter,
        probe_every_steps=config.checkpoint_every_steps,
        prompts=config.probe_prompts,
        encoding_name=config.encoding_name,
        max_tokens=config.probe_max_tokens,
        temperature=config.probe_temperature,
        top_p=config.probe_top_p,
        repetition_penalty=config.probe_repetition_penalty,
        pad_token_id=config.pad_token_id,
        ctx_length=config.max_seq_length - 1,
        save_dir=results_dir,
        seed=config.probe_seed,
    )
    probe_cb._post_generate_hook = augment_probe_results
    callbacks.append(probe_cb)

    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"steps_per_epoch={steps_per_epoch:,}, "
        f"initial_epoch={initial_epoch}, initial_step={initial_step:,}, "
        f"batch_size={config.batch_size}, "
        f"routing_mode={config.routing_mode}"
    )
    history = model.fit(
        train_dataset,
        epochs=config.num_epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )
    logger.info("Training completed!")

    generate_training_curves(history, results_dir)

    if "val_loss" in history.history:
        best_epoch = int(np.argmin(history.history["val_loss"]))
        logger.info(
            f"Best epoch: {best_epoch + 1} "
            f"(val_loss: {history.history['val_loss'][best_epoch]:.4f})"
        )

    return model, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CliffordNet routing-head NLP Causal LM Pre-training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--gpu", type=int, default=None, help="GPU device index")

    p.add_argument(
        "--variant", type=str, default="nano",
        choices=list(CliffordNetLMRouting.MODEL_VARIANTS.keys()) + ["custom"],
        help="Model variant (nano/mini/base/large/xl/custom)",
    )
    # Architecture-shaping args default to None so we can tell whether the
    # user explicitly passed them. When non-None, they override the named
    # variant's default; when None, the variant default wins. For
    # --variant custom they fall back to TrainingConfig dataclass defaults.
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--shifts", type=str, default=None,
                    help="Comma-separated ints, e.g. '1,2,4'")
    p.add_argument(
        "--cli-mode", type=str, default=None,
        choices=["inner", "wedge", "full"],
    )
    p.add_argument(
        "--ctx-mode", type=str, default=None,
        choices=["diff", "abs"],
    )
    p.add_argument("--use-global-context", action="store_true", default=None)
    p.add_argument("--stochastic-depth-rate", type=float, default=None)

    # Routing head
    p.add_argument(
        "--routing-mode", type=str, default="trainable",
        choices=["trainable", "deterministic"],
        help="Hierarchical routing layer mode",
    )

    # Token (input) embedding
    p.add_argument(
        "--input-embedding", type=str, default="hce",
        choices=["hce", "albert", "dense"],
        help="Token embedding strategy. hce (default): additive multi-"
             "codebook (~100x param compression). albert: factorized "
             "Embedding(vocab,k)->Dense(D). dense: legacy keras Embedding.",
    )
    p.add_argument(
        "--embedding-bottleneck-dim", type=int, default=None,
        help="ALBERT bottleneck dim k. Default: max(8, min(channels//2, 128)).",
    )
    p.add_argument(
        "--hce-num-chunks", type=int, default=2,
        help="HCE number of codebooks (K).",
    )
    p.add_argument(
        "--hce-chunk-bits", type=int, default=None,
        help="HCE bits per chunk. Default: ceil(log2(vocab) / num_chunks).",
    )

    # Training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--dropout-rate", type=float, default=0.1)

    # Loss
    p.add_argument(
        "--loss-type", type=str, default="ce",
        choices=["ce", "focal"],
    )
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    # Data source
    p.add_argument(
        "--dataset-source", type=str, default="huggingface",
        choices=["tfds", "huggingface"],
    )
    p.add_argument("--dataset-name", type=str, default="imdb_reviews")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--hf-cache-dir", type=str,
                    default="/media/arxwn/data0_4tb/datasets/wikipedia")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument(
        "--min-article-length", type=int, default=0,
        help="HF Wikipedia char-length filter. 0 (default) = no filter "
             "(recommended for packed CLM). 500+ for per-doc consumers.",
    )
    p.add_argument(
        "--shuffle-shards", type=int, default=4,
        help="HF Wikipedia parallel tokenization shards (D-002). 1 = "
             "single-thread, deterministic; >1 reshuffles each epoch and "
             "parallelises tokenization.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global seed for tf/keras + dataset shuffle. On --resume, "
             "data seed is shifted by initial_step (D-006).",
    )

    p.add_argument("--checkpoint-every-steps", type=int, default=25000)
    p.add_argument("--analyze-every-steps", type=int, default=0,
                    help="0 (default) disables blocking spectral analysis")
    p.add_argument("--max-checkpoints", type=int, default=3)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                    help="Override LR-schedule horizon. Required for "
                         "streaming HF data to get correct cosine decay.")
    p.add_argument("--probe-seed", type=int, default=42)

    p.add_argument(
        "--resume", type=str, default=None,
        help="Path to .keras checkpoint to resume training from",
    )

    p.add_argument("--save-dir", type=str,
                    default="results/cliffordnet_nlp_routing")

    return p


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    # Build arch_overrides only from explicitly-set CLI args (non-None).
    # These are forwarded to from_variant() for named variants.
    arch_overrides: Dict[str, Any] = {}
    if args.channels is not None:
        arch_overrides["channels"] = args.channels
    if args.depth is not None:
        arch_overrides["depth"] = args.depth
    if args.shifts is not None:
        arch_overrides["shifts"] = [int(s) for s in args.shifts.split(",")]
    if args.cli_mode is not None:
        arch_overrides["cli_mode"] = args.cli_mode
    if args.ctx_mode is not None:
        arch_overrides["ctx_mode"] = args.ctx_mode
    if args.use_global_context is not None:
        arch_overrides["use_global_context"] = args.use_global_context
    if args.stochastic_depth_rate is not None:
        arch_overrides["stochastic_depth_rate"] = args.stochastic_depth_rate

    # For TrainingConfig dataclass fields (used by --variant custom and for
    # logging), use overrides where present, otherwise the dataclass defaults.
    custom_defaults = TrainingConfig()
    return TrainingConfig(
        variant=args.variant,
        channels=arch_overrides.get("channels", custom_defaults.channels),
        depth=arch_overrides.get("depth", custom_defaults.depth),
        shifts=arch_overrides.get("shifts", custom_defaults.shifts),
        cli_mode=arch_overrides.get("cli_mode", custom_defaults.cli_mode),
        ctx_mode=arch_overrides.get("ctx_mode", custom_defaults.ctx_mode),
        use_global_context=arch_overrides.get(
            "use_global_context", custom_defaults.use_global_context,
        ),
        stochastic_depth_rate=arch_overrides.get(
            "stochastic_depth_rate", custom_defaults.stochastic_depth_rate,
        ),
        dropout_rate=args.dropout_rate,
        routing_mode=args.routing_mode,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        dataset_source=args.dataset_source,
        dataset_name=args.dataset_name,
        max_samples=args.max_samples,
        hf_cache_dir=args.hf_cache_dir,
        max_train_samples=args.max_train_samples,
        val_fraction=args.val_fraction,
        min_article_length=args.min_article_length,
        shuffle_shards=args.shuffle_shards,
        seed=args.seed,
        checkpoint_every_steps=args.checkpoint_every_steps,
        analyze_every_steps=args.analyze_every_steps,
        max_checkpoints=args.max_checkpoints,
        steps_per_epoch=args.steps_per_epoch,
        resume_from=args.resume,
        save_dir=args.save_dir,
        arch_overrides=arch_overrides,
        probe_seed=args.probe_seed,
        input_embedding=args.input_embedding,
        embedding_bottleneck_dim=args.embedding_bottleneck_dim,
        hce_num_chunks=args.hce_num_chunks,
        hce_chunk_bits=args.hce_chunk_bits,
    )


def main() -> None:
    """Main entry point."""
    args = _build_parser().parse_args()
    setup_gpu(gpu_id=args.gpu)

    config = _config_from_args(args)
    logger.info(
        f"Config: variant={config.variant}, "
        f"routing_mode={config.routing_mode}, "
        f"channels={config.channels}, depth={config.depth}, "
        f"shifts={config.shifts}, cli_mode={config.cli_mode}, "
        f"epochs={config.num_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, loss={config.loss_type}, "
        f"source={config.dataset_source}"
    )

    train_cliffordnet_nlp_routing(config)


if __name__ == "__main__":
    main()
