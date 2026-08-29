"""Two-stage trainer for the embeddings_experimental study.

Stage 1 -- masked language modelling on packed Wikipedia characters.
Stage 2 -- contrastive embedding fine-tuning, SimCSE-style: the same sentence
encoded twice under independent dropout gives a positive pair, so no paired
corpus is needed.

One script trains every arm: ``--model`` selects the encoder from
:data:`~config.MODEL_REGISTRY`, and the three study axes (model, variant,
pooling) are plain flags, so ``sweep.py`` drives cells by argv alone.

Two decisions are load-bearing and are not free to change casually.

**Stage 1 is packed, not padded.** Every row is exactly ``max_seq_length`` real
characters and the attention mask is all ones. The Clifford arm's block cannot
honour a padding mask, so a padded batch would make the comparison measure the
padding policy as much as the block. See :mod:`data` for the measurement.

**Weight decay excludes norms and biases.** ``optimizer_builder`` renames the
clipping keys, so a literal ``clipnorm`` in its config dict is silently ignored;
the correct key is ``gradient_clipping_by_norm``.

References:
    - Devlin et al., 2019. BERT: Pre-training of Deep Bidirectional
      Transformers for Language Understanding.
      (https://arxiv.org/abs/1810.04805)
    - Gao et al., 2021. SimCSE: Simple Contrastive Learning of Sentence
      Embeddings. (https://arxiv.org/abs/2104.08821)
    - Reimers and Gurevych, 2019. Sentence-BERT: Sentence Embeddings using
      Siamese BERT-Networks. (https://arxiv.org/abs/1908.10084)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.datasets.nlp import (
    DEFAULT_WIKIPEDIA_CACHE_DIR,
    load_wikipedia_train_val,
)
from dl_techniques.layers.tokenizers.ascii_char import (
    CLS_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
)
from dl_techniques.models.language.masked_language_model.mlm import (
    MaskedLanguageModel,
)
from dl_techniques.optimization import optimizer_builder
from dl_techniques.utils.logger import logger
from train.common import (
    prepare_run_dir,
    save_training_history_json,
    set_seeds,
    setup_gpu,
)
from train.common.nlp import create_warmup_lr_schedule

from .paths import (
    ENCODER_FILENAME,
    REPO_ROOT,
    encoder_path,
    eval_path,
    resolve_output_dir,
    results_path,
)

from .config import (
    BASELINE_MODEL,
    POOLING_STRATEGIES,
    VARIANTS,
    ExperimentConfig,
    available_models,
    build_model,
)
from .data import build_packed_mlm_dataset
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

__all__ = [
    "SimCSELoss",
    "SimCSEModel",
    "main",
    "parse_args",
    "resolve_output_dir",
    "run_study_cell",
]

# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------

def build_optimizer(
    learning_rate: float,
    weight_decay: float,
    clip_norm: float,
    total_steps: int,
    warmup_ratio: float,
) -> keras.optimizers.Optimizer:
    """Build AdamW with warmup + cosine decay and the conventional exclusions.

    :param learning_rate: Peak learning rate.
    :type learning_rate: float
    :param weight_decay: Decoupled weight decay.
    :type weight_decay: float
    :param clip_norm: Global gradient-norm clip.
    :type clip_norm: float
    :param total_steps: Total optimizer steps, for the schedule horizon.
    :type total_steps: int
    :param warmup_ratio: Fraction of ``total_steps`` spent warming up.
    :type warmup_ratio: float
    :return: The configured optimizer.
    :rtype: keras.optimizers.Optimizer
    """
    schedule = create_warmup_lr_schedule(
        learning_rate=learning_rate,
        num_epochs=1,
        steps_per_epoch=max(1, total_steps),
        warmup_ratio=warmup_ratio,
    )
    optimizer = optimizer_builder(
        {
            "type": "adamw",
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": weight_decay,
            # The RENAMED key. A literal "clipnorm" here is silently ignored.
            "gradient_clipping_by_norm": clip_norm,
        },
        schedule,
    )
    if hasattr(optimizer, "exclude_from_weight_decay"):
        # One call: exclude_from_weight_decay OVERWRITES rather than appends.
        optimizer.exclude_from_weight_decay(
            var_names=["bias", "gamma", "beta", "layer_scale"]
        )
        logger.info(
            "Excluded biases, norm gains and layer-scale gammas from weight decay."
        )
    return optimizer


# ---------------------------------------------------------------------
# Stage 1 -- masked language modelling
# ---------------------------------------------------------------------

def run_mlm_stage(
    config: ExperimentConfig,
    encoder: keras.Model,
    train_texts: tf.data.Dataset,
    val_texts: tf.data.Dataset,
    run_dir: str,
) -> Dict[str, Any]:
    """Pretrain the encoder with masked language modelling on packed text.

    :param config: The run configuration.
    :type config: ExperimentConfig
    :param encoder: The encoder to pretrain, modified in place.
    :type encoder: keras.Model
    :param train_texts: Dataset of raw training texts.
    :type train_texts: tf.data.Dataset
    :param val_texts: Dataset of raw validation texts.
    :type val_texts: tf.data.Dataset
    :param run_dir: Directory to write artifacts into.
    :type run_dir: str
    :return: The stage's history dictionary.
    :rtype: dict[str, Any]
    """
    logger.info("=== Stage 1: masked language modelling (packed, no padding) ===")

    train_ds = build_packed_mlm_dataset(
        train_texts,
        seq_len=config.max_seq_length,
        batch_size=config.mlm_batch_size,
        training=True,
        repeat=True,
    )
    val_ds = build_packed_mlm_dataset(
        val_texts,
        seq_len=config.max_seq_length,
        batch_size=config.mlm_batch_size,
        training=False,
    )

    steps_per_epoch = config.steps_per_epoch or 500
    total_steps = steps_per_epoch * config.mlm_epochs

    mlm_model = MaskedLanguageModel(
        encoder=encoder,
        vocab_size=config.vocab_size,
        mask_ratio=config.mask_ratio,
        mask_token_id=MASK_ID,
        random_token_ratio=config.random_token_ratio,
        unchanged_ratio=config.unchanged_ratio,
        special_token_ids=[PAD_ID, CLS_ID, SEP_ID, MASK_ID],
        initializer_range=0.02,
    )
    mlm_model.compile(
        optimizer=build_optimizer(
            learning_rate=config.mlm_learning_rate,
            weight_decay=config.mlm_weight_decay,
            clip_norm=config.mlm_gradient_clip_norm,
            total_steps=total_steps,
            warmup_ratio=config.mlm_warmup_ratio,
        )
    )

    history = mlm_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.mlm_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[keras.callbacks.TerminateOnNaN()],
        verbose=1,
    )

    path = encoder_path(run_dir)
    encoder.save(path)
    logger.info(f"Stage 1 complete; encoder saved to {path}")
    return dict(history.history)


# ---------------------------------------------------------------------
# Stage 2 -- contrastive embedding fine-tuning
# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.train.embeddings_experimental.train_embeddings")
class SimCSELoss(keras.losses.Loss):
    """Symmetric InfoNCE over two dropout views of the same batch.

    ``y_true`` is ignored: the positives are positional (row *i* of view A
    matches row *i* of view B), so the targets are implicit. ``y_pred`` is the
    stacked pair produced by :meth:`SimCSEModel.call`, shape
    ``(batch, 2, embed_dim)``.

    :param temperature: Softmax temperature over cosine similarities.
    :type temperature: float
    :param kwargs: Additional keyword arguments for the Loss base class.
    """

    def __init__(self, temperature: float = 0.05, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        """Compute the symmetric InfoNCE loss.

        :param y_true: Ignored; positives are positional.
        :type y_true: Any
        :param y_pred: ``(batch, 2, embed_dim)`` stacked views.
        :type y_pred: keras.KerasTensor
        :return: Scalar loss.
        :rtype: keras.KerasTensor
        """
        view_a = y_pred[:, 0, :]
        view_b = y_pred[:, 1, :]
        logits = keras.ops.matmul(
            view_a, keras.ops.transpose(view_b)
        ) / self.temperature
        targets = keras.ops.arange(keras.ops.shape(logits)[0])
        forward = keras.losses.sparse_categorical_crossentropy(
            targets, logits, from_logits=True
        )
        backward = keras.losses.sparse_categorical_crossentropy(
            targets, keras.ops.transpose(logits), from_logits=True
        )
        return keras.ops.mean(forward + backward) / 2.0

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Serializable configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


@register_dl_technique("dl_techniques.train.embeddings_experimental.train_embeddings")
class SimCSEModel(keras.Model):
    """SimCSE: two dropout views of one sentence form a positive pair.

    The forward pass encodes the SAME batch twice and stacks the results, so
    the loss is an ordinary ``compile(loss=...)`` function over one output
    tensor and training is stock ``fit()`` -- no custom ``train_step``, per the
    house rule.

    **Dropout is deliberately active in both views regardless of the outer
    training flag.** In SimCSE the positive pair IS the dropout noise: with
    dropout off the two views are identical, the similarity matrix is the
    identity and the loss collapses to a constant that measures nothing. That
    would make a validation number look excellent and mean nothing. Use
    :meth:`embed` for a deterministic embedding.

    :param encoder: The pretrained encoder.
    :type encoder: keras.Model
    :param projection_dim: Width of the projection head.
    :type projection_dim: int
    :param kwargs: Additional keyword arguments for the Model base class.
    """

    def __init__(
        self,
        encoder: keras.Model,
        projection_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projection_dim = projection_dim
        self.projection = keras.layers.Dense(
            projection_dim, use_bias=False, name="projection"
        )

    def _project(self, inputs, training: Optional[bool] = None):
        """Encode, project and L2-normalize one view."""
        pooled = self.encoder(inputs, training=training)["pooled_output"]
        projected = self.projection(pooled)
        return projected / (
            keras.ops.norm(projected, axis=-1, keepdims=True) + 1e-8
        )

    def embed(self, inputs):
        """Return a deterministic embedding, dropout off.

        :param inputs: Encoder inputs.
        :type inputs: Any
        :return: ``(batch, projection_dim)`` unit-norm embeddings.
        :rtype: keras.KerasTensor
        """
        return self._project(inputs, training=False)

    def call(self, inputs, training: Optional[bool] = None):
        """Return two dropout views of the batch, stacked.

        :param inputs: Encoder inputs.
        :type inputs: Any
        :param training: Accepted for the Keras contract; the views are always
            drawn WITH dropout, for the reason in the class docstring.
        :type training: bool | None
        :return: ``(batch, 2, projection_dim)``.
        :rtype: keras.KerasTensor
        """
        view_a = self._project(inputs, training=True)
        view_b = self._project(inputs, training=True)
        return keras.ops.stack([view_a, view_b], axis=1)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return the stacked-view shape.

        :param input_shape: Encoder input shape.
        :type input_shape: Any
        :return: ``(batch, 2, projection_dim)``.
        :rtype: Any
        """
        batch = None
        if isinstance(input_shape, dict):
            ids = input_shape.get("input_ids")
            batch = ids[0] if ids is not None else None
        return (batch, 2, self.projection_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: Serializable configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "encoder": keras.saving.serialize_keras_object(self.encoder),
                "projection_dim": self.projection_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SimCSEModel":
        """Rebuild from a serialized configuration.

        :param config: Configuration produced by :meth:`get_config`.
        :type config: dict[str, Any]
        :return: The reconstructed model.
        :rtype: SimCSEModel
        """
        config = dict(config)
        config["encoder"] = keras.saving.deserialize_keras_object(
            config["encoder"]
        )
        return cls(**config)


def run_contrastive_stage(
    config: ExperimentConfig,
    encoder: keras.Model,
    train_texts: tf.data.Dataset,
    val_texts: tf.data.Dataset,
    run_dir: str,
) -> Dict[str, Any]:
    """Fine-tune the encoder for embedding quality, SimCSE-style.

    :param config: The run configuration.
    :type config: ExperimentConfig
    :param encoder: The pretrained encoder, modified in place.
    :type encoder: keras.Model
    :param train_texts: Dataset of raw training texts.
    :type train_texts: tf.data.Dataset
    :param val_texts: Dataset of raw validation texts.
    :type val_texts: tf.data.Dataset
    :param run_dir: Directory to write artifacts into.
    :type run_dir: str
    :return: The stage's history dictionary.
    :rtype: dict[str, Any]
    """
    logger.info("=== Stage 2: contrastive embedding fine-tuning (SimCSE) ===")

    def with_dummy_targets(dataset: tf.data.Dataset) -> tf.data.Dataset:
        # Stock `fit()` wants (x, y). SimCSE's positives are positional, so the
        # target is unused; the loss ignores it.
        return dataset.map(
            lambda batch: (
                batch,
                tf.zeros(tf.shape(batch["input_ids"])[0], dtype=tf.int32),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = with_dummy_targets(
        build_packed_mlm_dataset(
            train_texts,
            seq_len=config.max_seq_length,
            batch_size=config.contrastive_batch_size,
            training=True,
            repeat=True,
        )
    )
    val_ds = with_dummy_targets(
        build_packed_mlm_dataset(
            val_texts,
            seq_len=config.max_seq_length,
            batch_size=config.contrastive_batch_size,
            training=False,
        )
    )

    steps_per_epoch = config.contrastive_steps_per_epoch or 200
    total_steps = steps_per_epoch * config.contrastive_epochs

    model = SimCSEModel(
        encoder=encoder,
        projection_dim=config.projection_dim,
    )
    model.compile(
        optimizer=build_optimizer(
            learning_rate=config.contrastive_learning_rate,
            weight_decay=config.mlm_weight_decay,
            clip_norm=config.mlm_gradient_clip_norm,
            total_steps=total_steps,
            warmup_ratio=config.mlm_warmup_ratio,
        ),
        loss=SimCSELoss(temperature=config.contrastive_temperature),
        # XLA is OFF here deliberately. Keras defaults `jit_compile="auto"`,
        # which turns it on for a GPU, and the SimCSE step -- two forward passes
        # of one batch feeding a symmetric cross-entropy -- fails to compile on
        # this TF 2.18 build with `FAILED_PRECONDITION: Can not combine dim
        # orders and requirements`. Stage 1 compiles fine, so this is scoped to
        # the stage that needs it rather than disabled globally.
        jit_compile=False,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.contrastive_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[keras.callbacks.TerminateOnNaN()],
        verbose=1,
    )

    encoder.save(encoder_path(run_dir))
    logger.info("Stage 2 complete; encoder updated in place")
    return dict(history.history)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_study_cell(config: ExperimentConfig) -> Dict[str, Any]:
    """Train one cell of the study end to end.

    :param config: The run configuration.
    :type config: ExperimentConfig
    :return: A results dictionary, also written to ``results.json``.
    :rtype: dict[str, Any]
    """
    set_seeds(config.seed)
    if config.mixed_bfloat16:
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        logger.info("Mixed precision enabled: mixed_bfloat16")

    if config.experiment_name is None:
        config.experiment_name = config.cell_id()
    config.output_dir = resolve_output_dir(config.output_dir)
    # No `output_dir=`: that parameter is the FULLY-RESOLVED run dir, so
    # passing config.output_dir would put every run in `results/` itself.
    run_dir = str(prepare_run_dir(config))
    logger.info(f"Run directory: {run_dir}")

    train_texts, val_texts = load_wikipedia_train_val(
        cache_dir=config.wikipedia_cache_dir or DEFAULT_WIKIPEDIA_CACHE_DIR,
        min_article_length=config.min_article_length,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        seed=config.seed,
        num_shards=config.shuffle_shards,
    )

    encoder = build_model(config)
    encoder.build((None, config.max_seq_length))
    logger.info(
        f"Built {config.model}-{config.variant}: "
        f"{encoder.count_params():,} parameters, "
        f"pooling={config.pooling_strategy}"
    )

    results: Dict[str, Any] = {
        "config": config.to_dict(),
        "parameters": int(encoder.count_params()),
        "run_dir": run_dir,
    }

    mlm_history = run_mlm_stage(config, encoder, train_texts, val_texts, run_dir)
    results["mlm"] = {k: [float(v) for v in vs] for k, vs in mlm_history.items()}
    save_training_history_json(mlm_history, run_dir)

    if config.run_contrastive:
        contrastive_history = run_contrastive_stage(
            config, encoder, train_texts, val_texts, run_dir
        )
        results["contrastive"] = {
            k: [float(v) for v in vs] for k, vs in contrastive_history.items()
        }

    if config.run_embedding_eval:
        # An evaluation failure must never kill a training cell that has
        # already produced a checkpoint -- but a SILENT failure would leave an
        # empty results table that reads like a config problem, so `eval_ok` is
        # a first-class field the report counts and prints.
        try:
            from .evaluate_embeddings import EvalConfig, evaluate_run

            results["embedding_eval"] = evaluate_run(
                run_dir,
                EvalConfig(
                    tfds_data_dir=config.tfds_data_dir,
                    max_length=config.max_seq_length,
                    max_queries=config.eval_max_queries,
                    probe_train_n=config.eval_probe_train_n,
                    batch_size=config.eval_batch_size,
                    seed=config.seed,
                ),
            )
        except Exception as exc:
            logger.error("embedding evaluation failed", exc_info=True)
            results["embedding_eval"] = {
                "eval_ok": False, "eval_error": str(exc)[:500]
            }

    with open(results_path(run_dir), "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Wrote {results_path(run_dir)}")
    return results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument vector; ``None`` uses ``sys.argv[1:]``.
    :type argv: Sequence[str] | None
    :return: The parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train one cell of the embeddings_experimental study: MLM "
            "pretraining on packed Wikipedia characters, then SimCSE "
            "contrastive fine-tuning."
        )
    )
    defaults = ExperimentConfig()

    axes = parser.add_argument_group("study axes")
    axes.add_argument(
        "--model", choices=available_models(), default=defaults.model,
        help=f"Encoder arm. Baseline is {BASELINE_MODEL}.",
    )
    axes.add_argument(
        "--variant", choices=list(VARIANTS), default=defaults.variant,
        help="Size ladder rung.",
    )
    axes.add_argument(
        "--pooling-strategy", choices=list(POOLING_STRATEGIES),
        default=defaults.pooling_strategy,
        help="How the sequence collapses to one embedding.",
    )
    axes.add_argument("--seed", type=int, default=defaults.seed)

    data = parser.add_argument_group("data")
    data.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    data.add_argument("--max-train-samples", type=int, default=defaults.max_train_samples)
    data.add_argument("--max-val-samples", type=int, default=defaults.max_val_samples)
    data.add_argument("--min-article-length", type=int, default=defaults.min_article_length)
    data.add_argument("--shuffle-shards", type=int, default=defaults.shuffle_shards)
    data.add_argument("--wikipedia-cache-dir", type=str, default=defaults.wikipedia_cache_dir)

    stage1 = parser.add_argument_group("stage 1: MLM")
    stage1.add_argument("--mlm-epochs", type=int, default=defaults.mlm_epochs)
    stage1.add_argument("--mlm-batch-size", type=int, default=defaults.mlm_batch_size)
    stage1.add_argument("--mlm-learning-rate", type=float, default=defaults.mlm_learning_rate)
    stage1.add_argument("--mlm-warmup-ratio", type=float, default=defaults.mlm_warmup_ratio)
    stage1.add_argument("--mlm-weight-decay", type=float, default=defaults.mlm_weight_decay)
    stage1.add_argument(
        "--mlm-gradient-clip-norm", type=float, default=defaults.mlm_gradient_clip_norm
    )
    stage1.add_argument("--mask-ratio", type=float, default=defaults.mask_ratio)
    stage1.add_argument("--random-token-ratio", type=float, default=defaults.random_token_ratio)
    stage1.add_argument("--unchanged-ratio", type=float, default=defaults.unchanged_ratio)
    stage1.add_argument("--steps-per-epoch", type=int, default=defaults.steps_per_epoch)

    stage2 = parser.add_argument_group("stage 2: contrastive")
    stage2.add_argument(
        "--no-contrastive", dest="run_contrastive", action="store_false",
        default=defaults.run_contrastive,
        help="Stop after MLM pretraining.",
    )
    stage2.add_argument("--contrastive-epochs", type=int, default=defaults.contrastive_epochs)
    stage2.add_argument(
        "--contrastive-batch-size", type=int, default=defaults.contrastive_batch_size
    )
    stage2.add_argument(
        "--contrastive-learning-rate", type=float, default=defaults.contrastive_learning_rate
    )
    stage2.add_argument(
        "--contrastive-temperature", type=float, default=defaults.contrastive_temperature
    )
    stage2.add_argument(
        "--contrastive-steps-per-epoch", type=int,
        default=defaults.contrastive_steps_per_epoch,
    )
    stage2.add_argument("--projection-dim", type=int, default=defaults.projection_dim)

    model_group = parser.add_argument_group("model overrides")
    model_group.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    model_group.add_argument(
        "--hidden-dropout-rate", type=float, default=defaults.hidden_dropout_rate
    )
    model_group.add_argument(
        "--stochastic-depth-rate", type=float, default=defaults.stochastic_depth_rate
    )
    model_group.add_argument(
        "--position-embedding-type", type=str,
        choices=["learned", "sinusoidal"],
        default=defaults.position_embedding_type,
        help=(
            "Default 'sinusoidal'. A learned table inits ~40x smaller than "
            "sinusoidal and does not grow, which left the transformer arm "
            "unable to use position at all; see RESULTS.md."
        ),
    )

    evaluation = parser.add_argument_group("embedding evaluation")
    evaluation.add_argument(
        "--no-embedding-eval", dest="run_embedding_eval", action="store_false",
        default=defaults.run_embedding_eval,
        help=(
            "Skip the SQuAD retrieval / SST-2 probe evaluation after training. "
            "The evaluation is what makes the study measure embedding quality "
            "rather than only optimisation."
        ),
    )
    evaluation.add_argument("--tfds-data-dir", type=str, default=defaults.tfds_data_dir)
    evaluation.add_argument("--eval-max-queries", type=int, default=defaults.eval_max_queries)
    evaluation.add_argument(
        "--eval-probe-train-n", type=int, default=defaults.eval_probe_train_n
    )
    evaluation.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)

    run = parser.add_argument_group("run")
    run.add_argument("--output-dir", type=str, default=defaults.output_dir)
    run.add_argument("--experiment-name", type=str, default=defaults.experiment_name)
    run.add_argument("--gpu", type=int, default=defaults.gpu)
    run.add_argument("--mixed-bfloat16", action="store_true", default=defaults.mixed_bfloat16)

    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from parsed arguments.

    Every field is taken from ``args`` by name rather than listed by hand, so a
    new flag cannot become a silent no-op by being forgotten here -- a defect
    this repository has shipped more than once.

    :param args: Parsed arguments.
    :type args: argparse.Namespace
    :return: The configuration.
    :rtype: ExperimentConfig
    """
    fields = ExperimentConfig().to_dict()
    values = {
        name: getattr(args, name) for name in fields if hasattr(args, name)
    }
    missing = sorted(set(fields) - set(values))
    if missing:
        raise ValueError(
            f"these ExperimentConfig fields have no CLI flag: {missing}"
        )
    return ExperimentConfig(**values)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    :param argv: Argument vector; ``None`` uses ``sys.argv[1:]``.
    :type argv: Sequence[str] | None
    :return: Process exit code.
    :rtype: int
    """
    args = parse_args(argv)
    setup_gpu(args.gpu)
    config = config_from_args(args)
    try:
        run_study_cell(config)
    except Exception:
        logger.error("training cell failed", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
