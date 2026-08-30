"""Model registry and run configuration for the embeddings study.

The registry is the study's model axis. It maps a ``--model`` string to a
builder, so adding an arm to the sweep is one entry here plus the leaf package
itself -- the trainer, the sweep driver and the report all pick it up without
further edits.

Registry keys are public: they are recorded in every run directory's
``config.json`` and in the study's reports, so renaming one invalidates existing
results rather than tidying them. Append; do not rename or reorder.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.tokenizers.ascii_char import VOCAB_SIZE as ASCII_VOCAB_SIZE
from dl_techniques.models.embeddings_experimental.ascii_bert import create_ascii_bert
from dl_techniques.models.embeddings_experimental.ascii_clifford_bert import (
    create_ascii_clifford_bert,
)
from dl_techniques.models.embeddings_experimental.ascii_convnext_bert import (
    create_ascii_convnext_bert,
)

# ---------------------------------------------------------------------

__all__ = [
    "MODEL_REGISTRY",
    "ExperimentConfig",
    "available_models",
    "build_model",
]

#: ``--model`` string -> factory. Append-only; the keys are public API.
MODEL_REGISTRY: Dict[str, Callable[..., keras.Model]] = {
    "ascii_bert": create_ascii_bert,
    "ascii_clifford_bert": create_ascii_clifford_bert,
    "ascii_convnext_bert": create_ascii_convnext_bert,
}

#: The study's baseline. Every paired statistic in the report is computed
#: against this arm at a matched size.
BASELINE_MODEL: str = "ascii_bert"

#: The size axis. Every arm must expose every one of these variants, so the
#: ladder lines up across arms.
VARIANTS: tuple = ("tiny", "small", "base")

#: The pooling axis.
#:
#: ``"max"`` was added 2026-08-30. `EmbeddingEncoder` had supported it all
#: along, but the study's axis excluded it -- and it is the readout that makes a
#: sinusoidal-positioned encoder length-invariant. Measured on the same weights,
#: encoding one repeated sentence at 64 vs 512 real characters, cosine against
#: the 64-character version: ``mean`` gives **0.3805** while ``max`` gives
#: **0.9693**. Since SQuAD queries average 59 characters against contexts of
#: 774, mean pooling displaces a query from its own answer by length alone.
#: See RESULTS.md, "The cause".
POOLING_STRATEGIES: tuple = ("cls", "mean", "attention", "max")


def available_models() -> List[str]:
    """Return the registered model keys, sorted.

    :return: Registry keys.
    :rtype: list[str]
    """
    return sorted(MODEL_REGISTRY)


@dataclass
class ExperimentConfig:
    """One cell of the study: a model, a size, a pooling strategy, a seed.

    Every field here is read by :mod:`train_embeddings`. A field nothing reads
    is a knob the user can set with no effect, which is the defect class
    ``tests/test_train/test_config_fields_are_live.py`` exists to catch -- so
    delete an unused field rather than leaving it for later.
    """

    # -- identity ------------------------------------------------------
    model: str = BASELINE_MODEL
    variant: str = "tiny"
    pooling_strategy: str = "mean"
    seed: int = 0

    # -- data ----------------------------------------------------------
    max_seq_length: int = 256
    max_train_samples: Optional[int] = 20000
    max_val_samples: int = 1000
    min_article_length: int = 0
    shuffle_shards: int = 4
    wikipedia_cache_dir: Optional[str] = None

    # -- stage 1: masked language modelling ----------------------------
    mlm_epochs: int = 1
    mlm_batch_size: int = 32
    mlm_learning_rate: float = 5e-4
    mlm_warmup_ratio: float = 0.06
    mlm_weight_decay: float = 0.01
    mlm_gradient_clip_norm: float = 1.0
    mask_ratio: float = 0.15
    random_token_ratio: float = 0.1
    unchanged_ratio: float = 0.1
    steps_per_epoch: Optional[int] = None

    # -- stage 2: contrastive embedding fine-tuning --------------------
    run_contrastive: bool = True
    contrastive_epochs: int = 1
    contrastive_batch_size: int = 64
    contrastive_learning_rate: float = 1e-4
    contrastive_temperature: float = 0.05
    contrastive_steps_per_epoch: Optional[int] = None
    projection_dim: int = 256

    # -- model overrides ----------------------------------------------
    vocab_size: int = ASCII_VOCAB_SIZE
    hidden_dropout_rate: float = 0.1
    stochastic_depth_rate: float = 0.0

    #: How positional information is produced. ``'sinusoidal'``, NOT the
    #: encoder's own ``'learned'`` default, and the difference is large enough
    #: that it must be recorded per run rather than inherited silently.
    #:
    #: A learned table is initialized at ``initializer_range=0.02`` -- mean row
    #: norm ~0.2 against sinusoidal's ~8 -- and measurably never grows: over
    #: 3000 steps it SHRANK 0.1987 -> 0.1612 while the word table grew to
    #: 0.3283. The transformer arm could not bootstrap position-dependent
    #: attention from it and converged to a bag of characters, scoring the
    #: unigram-plus-copy solution exactly (2.8318 measured against 2.8022
    #: predicted). Switching this one field to sinusoidal bought 0.85 nats at
    #: 256 context. Disabling weight decay does NOT substitute: the cause is
    #: the initial scale of the signal, not its decay.
    #:
    #: See RESULTS.md section "Why the transformer arm was stuck".
    position_embedding_type: str = "sinusoidal"

    # -- embedding evaluation ------------------------------------------
    run_embedding_eval: bool = True
    tfds_data_dir: str = "/media/arxwn/data0_4tb/datasets/tensorflow_datasets"
    eval_max_queries: int = 2000
    eval_probe_train_n: int = 8000
    eval_batch_size: int = 64

    # -- run plumbing --------------------------------------------------
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    gpu: Optional[int] = None
    mixed_bfloat16: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary.

        :return: Field name to value.
        :rtype: dict[str, Any]
        """
        return asdict(self)

    def cell_id(self) -> str:
        """Return the sweep-cell identifier this configuration names.

        Used as the run-directory name so a sweep's cells are addressable and
        a re-run overwrites its own cell rather than accumulating duplicates.

        :return: ``<model>/<variant>/<pooling>/seed_<n>``.
        :rtype: str
        """
        return (
            f"{self.model}/{self.variant}/{self.pooling_strategy}/"
            f"seed_{self.seed}"
        )


def build_model(config: ExperimentConfig) -> keras.Model:
    """Build the encoder named by ``config``.

    :param config: The run configuration.
    :type config: ExperimentConfig
    :return: The configured encoder.
    :rtype: keras.Model
    :raises ValueError: If ``config.model`` is not registered.
    """
    if config.model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {config.model!r}. Available: {available_models()}"
        )
    factory = MODEL_REGISTRY[config.model]
    return factory(
        config.variant,
        vocab_size=config.vocab_size,
        pooling_strategy=config.pooling_strategy,
        max_position_embeddings=config.max_seq_length,
        hidden_dropout_rate=config.hidden_dropout_rate,
        stochastic_depth_rate=config.stochastic_depth_rate,
        position_embedding_type=config.position_embedding_type,
    )
