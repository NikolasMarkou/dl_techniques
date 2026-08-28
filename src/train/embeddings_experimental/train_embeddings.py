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

from .config import (
    BASELINE_MODEL,
    POOLING_STRATEGIES,
    VARIANTS,
    ExperimentConfig,
    available_models,
    build_model,
)
from .data import build_packed_mlm_dataset

# ---------------------------------------------------------------------

__all__ = ["main", "parse_args", "run_study_cell"]

#: Filename of the encoder handed from stage 1 to stage 2 and to evaluation.
ENCODER_FILENAME = "encoder.keras"


def encoder_path(run_dir: str) -> str:
    """Return the encoder checkpoint path for a run directory.

    The ONE producer of this path. Both the stage-1 save site and every reader
    call it, so the two cannot drift -- the failure mode this repo has already
    paid for once, where a filename had several readers and no writer and every
    default run died after training had finished.

    :param run_dir: The run directory.
    :type run_dir: str
    :return: Absolute or relative path to the encoder checkpoint.
    :rtype: str
    """
    return os.path.join(run_dir, ENCODER_FILENAME)


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

@keras.saving.register_keras_serializable()
class SimCSEModel(keras.Model):
    """SimCSE: two dropout views of one sentence form a positive pair.

    The same batch is encoded twice with ``training=True``, so independent
    dropout masks give two different embeddings of the same text. Every other
    sentence in the batch is a negative. This needs no paired corpus, which is
    what makes it usable on the same Wikipedia stream stage 1 uses.

    :param encoder: The pretrained encoder.
    :type encoder: keras.Model
    :param projection_dim: Width of the projection head.
    :type projection_dim: int
    :param temperature: Softmax temperature over cosine similarities.
    :type temperature: float
    :param kwargs: Additional keyword arguments for the Model base class.
    """

    def __init__(
        self,
        encoder: keras.Model,
        projection_dim: int = 256,
        temperature: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.projection = keras.layers.Dense(
            projection_dim, use_bias=False, name="projection"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Return the tracked metrics.

        :return: The running loss tracker.
        :rtype: list[keras.metrics.Metric]
        """
        return [self.loss_tracker]

    def _embed(self, inputs, training: Optional[bool] = None):
        """Encode, project and L2-normalize one view."""
        pooled = self.encoder(inputs, training=training)["pooled_output"]
        projected = self.projection(pooled)
        return projected / (
            keras.ops.norm(projected, axis=-1, keepdims=True) + 1e-8
        )

    def call(self, inputs, training: Optional[bool] = None):
        """Return the normalized embedding of one view.

        :param inputs: Encoder inputs.
        :type inputs: Any
        :param training: Keras training flag.
        :type training: bool | None
        :return: ``(batch, projection_dim)`` unit-norm embeddings.
        :rtype: keras.KerasTensor
        """
        return self._embed(inputs, training=training)

    def _contrastive_loss(self, view_a, view_b):
        """Symmetric InfoNCE over two views of the same batch."""
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

    def train_step(self, data) -> Dict[str, Any]:
        """Run one SimCSE step.

        A custom step is unavoidable here: the loss needs TWO forward passes of
        the same batch, which ``compile(loss=...)`` cannot express.

        :param data: A batch of encoder inputs.
        :type data: Any
        :return: The tracked metrics.
        :rtype: dict[str, Any]
        """
        inputs = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            view_a = self._embed(inputs, training=True)
            view_b = self._embed(inputs, training=True)
            loss = self._contrastive_loss(view_a, view_b)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data) -> Dict[str, Any]:
        """Evaluate one SimCSE step.

        Dropout is still ACTIVE here (``training=True`` in ``_embed``): with it
        off the two views are identical, the loss is trivially near zero, and
        the number means nothing.

        :param data: A batch of encoder inputs.
        :type data: Any
        :return: The tracked metrics.
        :rtype: dict[str, Any]
        """
        inputs = data[0] if isinstance(data, tuple) else data
        view_a = self._embed(inputs, training=True)
        view_b = self._embed(inputs, training=True)
        self.loss_tracker.update_state(self._contrastive_loss(view_a, view_b))
        return {"loss": self.loss_tracker.result()}

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
                "temperature": self.temperature,
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

    train_ds = build_packed_mlm_dataset(
        train_texts,
        seq_len=config.max_seq_length,
        batch_size=config.contrastive_batch_size,
        training=True,
        repeat=True,
    )
    val_ds = build_packed_mlm_dataset(
        val_texts,
        seq_len=config.max_seq_length,
        batch_size=config.contrastive_batch_size,
        training=False,
    )

    steps_per_epoch = config.contrastive_steps_per_epoch or 200
    total_steps = steps_per_epoch * config.contrastive_epochs

    model = SimCSEModel(
        encoder=encoder,
        projection_dim=config.projection_dim,
        temperature=config.contrastive_temperature,
    )
    model.compile(
        optimizer=build_optimizer(
            learning_rate=config.contrastive_learning_rate,
            weight_decay=config.mlm_weight_decay,
            clip_norm=config.mlm_gradient_clip_norm,
            total_steps=total_steps,
            warmup_ratio=config.mlm_warmup_ratio,
        ),
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

    with open(os.path.join(run_dir, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Wrote {os.path.join(run_dir, 'results.json')}")
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
