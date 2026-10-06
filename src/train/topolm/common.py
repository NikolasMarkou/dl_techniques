"""Training pipeline for TopoLM, plus the topographic evaluation it produces.

Shape of a run
--------------
Pretraining is roughly one pass over a text stream, so the validation signal a
run is judged on has to be sampled far more often than once per epoch. The
pipeline therefore runs VIRTUAL EPOCHS: ``fit`` is given
``steps_per_epoch=eval_every_steps``, so every epoch-level callback -- including
the early stopper -- fires on the intended cadence rather than once per pass over
the data. The consequence is stated rather than hidden: the run's optimizer-step
count becomes ``eval_every_steps * epochs``, which is not one pass over the data.
Set ``train_steps`` to pin the step count instead; the cadence is then derived
from it and is unchanged. Both numbers are logged before the first step.

Why not the shared callback factory
-----------------------------------
``train.common.create_nlp_callbacks`` unconditionally installs
``keras.callbacks.EarlyStopping``. The rule the paper trained under is "three
CONSECUTIVE INCREASES on validation loss", which is a different question: the
stock callback counts from the best value seen, so a curve that dips to a new
best and then rises three times is stopped by the paper's rule and tolerated by
the stock one. Both callbacks present would also be two stop signals on one run.
So this module builds its own list and takes ``results_dir`` from
``prepare_run_dir``.

What the run reports
--------------------
``loss`` alone cannot answer the question this trainer exists to ask. The taps
contribute through ``add_loss``, so the reported training loss is
``task + sum(alpha_k * SL_k)``; a rising curve can be a rising task loss with a
falling penalty or the reverse, and the two look identical in one number. So
every run also records, through
:class:`~dl_techniques.callbacks.spatial_loss_logger.SpatialLossLogger`:

- each tap's penalty and the weighted total,
- ``task_loss`` -- the reported loss minus that total,
- ``spatial/unaccounted`` -- how much of the computed penalty is NOT reflected in
  the reported loss.

That last field is the one to watch. A head built without
``aggregate_backbone_losses=True`` looks completely healthy and trains a model
with no topography at all; the run only reveals it through that gap.

The control arm
---------------
``train_topolm.py --paired`` trains the topographic model and then an
``alpha = 0`` control with identical seeds, data order and every other
hyperparameter. The control's taps are still created and still built, so the two
arms hold an identical weight set and differ in their objective alone; the test
suite pins that, and a run where it does not hold is not a comparison.

References:
    - Rathi, Mehrer, AlKhamissi, Binhuraib, Blauch & Schrimpf, 2025. TopoLM.
      ICLR 2025. (https://arxiv.org/abs/2410.11516)
    - Fedorenko, Hsieh, Nieto-Castano & Kanwisher, 2010. New method for fMRI
      investigations of language: defining ROIs functionally in individual
      subjects. (https://arxiv.org/abs/1003.2782)
    - Margalit, Lee, Finzi, DiCarlo, Grill-Spector & Yamins, 2024. A unifying
      framework for functional organization in early and higher ventral visual
      cortex. Neuron 112(14).
"""

import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from train.common import (
    GenerationProbeCallback,
    StepCheckpointCallback,
    run_timestamp,
    save_training_history_json,
    set_seeds,
)
from train.common.clm_pretrain import (
    ClmPretrainConfig,
    create_clm_loss_fn,
    extract_step_from_checkpoint,
    load_train_val_datasets,
    make_clm_steps_per_epoch,
)
from train.common.evaluation import generate_training_curves
from train.common.nlp import (
    augment_probe_results,
    create_tokenizer,
    create_warmup_lr_schedule,
)
from train.common.run_io import prepare_run_dir

from dl_techniques.callbacks.consecutive_increase_early_stopping import (
    ConsecutiveIncreaseEarlyStopping,
)
from dl_techniques.callbacks.spatial_loss_logger import SpatialLossLogger
from dl_techniques.layers.regularization.gaussian_readout import GaussianReadout
from dl_techniques.metrics.spatial_autocorrelation import morans_i_summary
from dl_techniques.metrics.topographic_selectivity import (
    contrast_tmap,
    fdr_across_layers,
    grow_clusters,
)
from dl_techniques.models.common.masked_language_model import (
    CausalLanguageModel,
)
from dl_techniques.models.language.topolm import TopoLM
from dl_techniques.optimization import optimizer_builder
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

#: The paper's four stimulus conditions, in the order its figures report them.
#: Sentences index syntactic AND lexical information; unconnected words are
#: lexical but not syntactic; Jabberwocky sentences are syntactic but not
#: lexical; unconnected nonwords are neither. The expected ordering is
#: ``sentences > {unconnected words, jabberwocky} > nonwords``.
PAPER_CONDITIONS = (
    "sentences",
    "unconnected_words",
    "jabberwocky_sentences",
    "unconnected_nonwords",
)

#: A minimal stand-in stimulus set, so the evaluation block is runnable without a
#: neuroscience dataset. It is NOT a substitute for Fedorenko et al. (2010)'s
#: sentence / nonword materials, and every report it produces is stamped
#: ``stimuli_are_smoke_set: true``.
SMOKE_STIMULI: Dict[str, List[str]] = {
    "a": [
        "the quick brown fox jumps over the lazy dog",
        "she sells sea shells by the seashore",
        "they walked home through the quiet park",
        "he opened the door and stepped outside",
        "we should finish the report before friday",
    ],
    "b": [
        "bright table yellow winter quiet glass",
        "silver morning pocket corner gentle",
        "cloud seven river anchor timber",
        "purple station lantern meadow stone",
        "copper valley whisper candle ridge",
    ],
    "c": [
        "the wug zipped through the blarnket tree",
        "she mibbed a slarn around the glarn",
        "they frolicked past the prarn wall",
        "he glimped the doark and stomped out",
        "we should plarn the plort before frarn",
    ],
    "d": [
        "wug blarnket zipped she mibbed slarn",
        "frarn plort glimped doark glarn starn",
        "she prarn frolicked glimped brarn",
        "zipped starn mibbed wug frolicked",
        "doark glarn slarn she prarn plort",
    ],
}


@dataclass
class TopoLMTrainingConfig(ClmPretrainConfig):
    """Configuration for TopoLM pre-training.

    Additive only: no inherited field is reordered or re-declared except
    ``model_variant`` and ``save_dir``, whose defaults are overridden in place, so
    the positional-construction contract of :class:`ClmPretrainConfig` holds.

    :ivar spatial_alpha: Weight on every tap's spatial loss. ``0`` trains the
        non-topographic control, with an identical weight set.
    :ivar spatial_radius: Neighbourhood radius; a radius-``r`` patch holds
        ``(2 * r + 1) ** 2`` units, so the width must be at least that.
    :ivar spatial_neighborhoods: Neighbourhoods sampled per tap per step, and
        averaged. ``5`` is the paper's value.
    :ivar spatial_distance: Distance metric behind each tap's prior. ``'linf'`` is
        the paper's, and it is the same contiguity the post-hoc cluster growing
        and Moran's I use, so the objective and the measurement agree.
    :ivar spatial_permute: Whether each tap draws its own unit permutation.
        ``False`` is the paper's Fig. 12 ablation and must not be the default.
    :ivar tap_sites: Branch outputs to tap: ``'both'``, ``'attention'`` or
        ``'mlp'``.
    :ivar eval_every_steps: Validation cadence, and the virtual epoch length.
    :ivar eval_batches: Validation batches per evaluation.
    :ivar train_steps: Pin the optimizer-step count, deriving the number of
        virtual epochs from it. ``None`` means ``eval_every_steps *
        num_epochs``, which is NOT one pass over the data -- the run logs which
        of the two it is using.
    :ivar early_stop_patience: Consecutive increases tolerated. The paper's 3.
    :ivar spatial_log_every: Batches between spatial-loss log entries.
    :ivar readout_fwhm: FWHM for the simulated fMRI readout. ``None`` disables
        the smoothed arm of the evaluation.
    :ivar readout_unit_spacing: Inter-unit spacing for the readout kernel.
    :ivar min_cluster_size: Smallest cluster the post-hoc sweep keeps.
    :ivar permutation_p_value: Also score Moran's I against spatial randomness
        by permutation. The paper's third Moran statistic, off by default: it
        costs ``num_permutations`` shuffles per tap per arm.
    :ivar num_permutations: Shuffles behind the permutation p-value, which also
        sets its floor at ``1 / (num_permutations + 1)``.
    :ivar stimuli: Optional ``{condition: [sentence, ...]}``. ``None`` selects
        :data:`SMOKE_STIMULI` and marks every report as such.
    :ivar contrast_conditions: The two conditions the t-map contrasts.
    """

    model_variant: str = "small"
    save_dir: str = "results/topolm_pretrain"

    # Topographic objective
    spatial_alpha: float = 2.5
    spatial_radius: int = 5
    spatial_neighborhoods: int = 5
    spatial_distance: str = "linf"
    spatial_permute: bool = True
    tap_sites: str = "both"

    # Virtual-epoch cadence
    eval_every_steps: int = 2000
    eval_batches: int = 32
    train_steps: Optional[int] = None
    early_stop_patience: int = 3
    spatial_log_every: int = 100

    # Topographic evaluation
    readout_fwhm: Optional[float] = 2.0
    readout_unit_spacing: float = 1.0
    min_cluster_size: int = 10
    permutation_p_value: bool = False
    num_permutations: int = 9999
    stimuli: Optional[Dict[str, List[str]]] = None
    contrast_conditions: Tuple[str, str] = ("a", "b")

    # The paper's optimizer recipe AS THE DEFAULTS, so the shipped configuration
    # is the recipe rather than an approximation of it. AdamW beta (0.9, 0.95),
    # peak lr 6e-4 on warmup+cosine, weight decay 0.1, no dropout.
    learning_rate: float = 6e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    dropout_rate: float = 0.0


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def resolve_tap_sites(tap_sites: str) -> Tuple[str, ...]:
    """Map the config string onto the block's tap-site tuple.

    :param tap_sites: ``'both'``, ``'attention'`` or ``'mlp'``.
    :type tap_sites: str
    :return: The subset to tap.
    :rtype: Tuple[str, ...]
    :raises ValueError: If the string is not one of the three -- listing them.
    """
    if tap_sites == "both":
        return ("attention", "mlp")
    if tap_sites in ("attention", "mlp"):
        return (tap_sites,)
    raise ValueError(
        f"tap_sites must be 'both', 'attention' or 'mlp', got {tap_sites!r}"
    )


def _require_vocab_covers_tokenizer(
    config: TopoLMTrainingConfig, preprocessor: Any
) -> None:
    """Reject a vocabulary too small for the tokenizer that will feed it.

    Checked HERE, immediately after the tokenizer exists and BEFORE the dataset
    is built, because that is the last point at which the message can still name
    the flag that caused it. Left to the forward pass it surfaces as
    ``InvalidArgumentError: indices[..] = 50259 is not in [0, 50257)`` from
    inside ``Embedding.call`` -- which names no argument and arrives only after
    the dataset has been tokenized and cached.

    The bound is the tokenizer's OWN ``vocab_size``, not a constant, so changing
    ``encoding_name`` moves the requirement with it. That matters because the
    natural wrong value is GPT-2's 50257, which is what this CLI used to default
    to against a ``cl100k_base`` tokenizer whose ids reach 100267.

    :param config: The run's configuration.
    :type config: TopoLMTrainingConfig
    :param preprocessor: Tokenizer whose ids must fit the embedding table.
    :type preprocessor: Any
    :raises ValueError: If ``config.vocab_size`` cannot represent every id the
        tokenizer can emit -- naming the flag, the two sizes, and the encoding.
    """
    required = getattr(preprocessor, "vocab_size", None)
    if required is None:
        # A stub or a callable without the attribute. Refusing here would break
        # every test double that stands in for the tokenizer; the forward pass
        # remains the backstop.
        return
    if config.vocab_size >= int(required):
        return
    encoding = getattr(
        getattr(preprocessor, "tokenizer", None), "name", "unknown"
    )
    raise ValueError(
        f"vocab_size {config.vocab_size} cannot hold every id the "
        f"{encoding!r} tokenizer emits (it needs at least {required}; ids run "
        f"to {required - 1}). Raise --vocab-size, or lower --encoding-name to "
        f"a tokenizer with a smaller vocabulary. Left unchecked this fails on "
        f"the first batch as an out-of-range Gather in Embedding.call."
    )


def build_backbone(
    config: TopoLMTrainingConfig,
    required_vocab_size: Optional[int] = None,
) -> TopoLM:
    """Build the bare :class:`TopoLM`, before any training head.

    :param config: The run's configuration.
    :type config: TopoLMTrainingConfig
    :param required_vocab_size: The smallest vocabulary that can represent every
        id the tokenizer will emit, or ``None`` to skip the check. Passed by
        :func:`train_topolm` from the tokenizer it actually built; supplied here
        so a direct caller of this function can get the same early error.
    :type required_vocab_size: Optional[int]
    :return: A randomly initialised TopoLM matching ``config``.
    :rtype: TopoLM
    :raises ValueError: If ``config.tap_sites`` is not a recognised value, or
        ``config.vocab_size`` is below ``required_vocab_size``.
    """
    if required_vocab_size is not None and config.vocab_size < int(
        required_vocab_size
    ):
        raise ValueError(
            f"vocab_size {config.vocab_size} cannot hold every id the tokenizer "
            f"emits (it needs at least {required_vocab_size}); raise "
            f"--vocab-size, or use a tokenizer with a smaller vocabulary"
        )

    variant_kwargs: Dict[str, Any] = {
        "vocab_size": config.vocab_size,
        # The position table's size comes from HERE, not from the variant, because
        # the data pipeline is chunked to config.max_seq_length. Without this the
        # table keeps the variant's length while the batches are longer, and the
        # run dies in `positional_embeddings` with an out-of-range Gather on the
        # first step -- after the dataset has already been built. MEASURED with
        # `--variant tiny --max-seq-length 512` (the CLI default) against the
        # variant's 256: `indices[256] = 256 is not in [0, 256)`.
        "max_seq_len": config.max_seq_length,
        "dropout_rate": config.dropout_rate,
        "attention_dropout_rate": config.attention_dropout_rate,
        "tie_word_embeddings": config.tie_word_embeddings,
        "alpha": config.spatial_alpha,
        "radius": config.spatial_radius,
        "num_neighborhoods": config.spatial_neighborhoods,
        "distance": config.spatial_distance,
        "permute": config.spatial_permute,
        "tap_sites": resolve_tap_sites(config.tap_sites),
        "seed": config.seed,
    }
    if config.num_layers is not None:
        variant_kwargs["depth"] = config.num_layers
    if config.num_heads is not None:
        variant_kwargs["num_heads"] = config.num_heads

    backbone = TopoLM.from_variant(config.model_variant, **variant_kwargs)

    if backbone.max_seq_len != config.max_seq_length:  # pragma: no cover
        raise ValueError(
            f"the position table is {backbone.max_seq_len} but the data pipeline "
            f"emits {config.max_seq_length}-token windows; these must agree or "
            f"the first batch dies in positional_embeddings with an "
            f"out-of-range Gather"
        )
    height, width = backbone.grid_shape
    logger.info(
        f"Backbone: variant={config.model_variant}, "
        f"embed_dim={backbone.embed_dim}, depth={backbone.depth}, "
        f"heads={backbone.num_heads}, grid={height}x{width}, "
        f"{len(backbone.tap_layers)} taps at alpha={config.spatial_alpha} "
        f"(radius {config.spatial_radius}, {config.spatial_distance}, "
        f"permute={config.spatial_permute}, sites={list(backbone.tap_sites)})"
    )
    return backbone


def create_topolm_model(
    config: TopoLMTrainingConfig,
    loss_fn: Optional[keras.losses.Loss] = None,
    required_vocab_size: Optional[int] = None,
) -> Tuple[CausalLanguageModel, TopoLM]:
    """Wrap the backbone in the shared CLM training head.

    ``aggregate_backbone_losses=True`` is LOAD-BEARING and not a default worth
    overriding. The taps reach the objective only through ``backbone.losses``;
    without the flag the head computes every penalty, discards it, and the run
    produces a non-topographic model with a healthy loss curve. Measured on the
    same weights at ``alpha = 2.5``: a reported training loss of 15.31 aggregated
    against 5.36 not, with an EVALUATION loss of 5.31 either way -- the taps are
    training-only, so the validation curve is the pure task loss in both arms.

    An eager forward pass runs here on the WRAPPER, not the bare backbone: it
    forces the head's output-key resolution and its causality probe before
    ``fit`` traces ``train_step``, and an unbuilt head raises from
    ``count_params`` rather than reporting a number.

    :param config: The run's configuration.
    :type config: TopoLMTrainingConfig
    :param loss_fn: Loss override, or ``None`` for the config's CE / focal
        choice.
    :type loss_fn: Optional[keras.losses.Loss]
    :return: ``(training_model, backbone)``.
    :rtype: Tuple[CausalLanguageModel, TopoLM]
    """
    backbone = build_backbone(config, required_vocab_size=required_vocab_size)

    head = CausalLanguageModel(
        backbone=backbone,
        vocab_size=config.vocab_size,
        skip_head=True,
        output_key="logits",
        pre_shifted=True,
        loss_fn=loss_fn if loss_fn is not None else create_clm_loss_fn(config),
        aggregate_backbone_losses=True,
        verify_causality=True,
    )

    probe_length = max(1, min(8, config.max_seq_length - 1))
    head(np.zeros((1, probe_length), dtype="int32"), training=False)

    logger.info(
        f"Training head: {head.count_params():,} parameters, "
        f"{len(head.trainable_variables)} trainable tensors, "
        f"{len(backbone.tap_layers)} taps aggregated into the objective"
    )
    return head, backbone


def compile_model(
    model: CausalLanguageModel,
    config: TopoLMTrainingConfig,
    epochs: int,
    steps_per_epoch: int,
) -> None:
    """Compile with the paper's optimizer recipe.

    AdamW at ``beta_1 = 0.9``, ``beta_2 = 0.95``, weight decay ``0.1``, peak
    learning rate ``6e-4`` on a warmup-then-cosine schedule, and gradient clipping
    at ``1.0`` -- specified as ``gradient_clipping_by_norm``, the GLOBAL norm,
    which is what "gradient clipping at 1.0" means and is distinct from the
    per-variable ``gradient_clipping_by_norm_local``.

    Weight decay is EXCLUDED for biases, the two normalization scales and the
    embedding table. Both exclusions matter: decaying a LayerNorm gain toward zero
    fights the very normalization the taps are trying to make spatially smooth,
    and decaying the tied embedding table decays the output head with it.

    No ``loss=`` or ``metrics=`` are passed. ``CausalLanguageModel`` owns its
    loss and its ``loss`` / ``accuracy`` / ``perplexity`` trackers through a
    hand-rolled ``train_step``, so a compiled loss here would be inert config.
    ``jit_compile`` stays off: ``SpatialSmoothness`` is graph-safe, but the
    whole-step trace is not measured for the fused head, and a run that silently
    refuses to compile is worse than one that runs eagerly.

    :param model: The training head.
    :type model: CausalLanguageModel
    :param config: The run's configuration.
    :type config: TopoLMTrainingConfig
    :param epochs: Number of virtual epochs, which sets the schedule's horizon
        together with ``steps_per_epoch``.
    :type epochs: int
    :param steps_per_epoch: Virtual epoch length in optimizer steps.
    :type steps_per_epoch: int
    :raises ValueError: If either horizon argument is below 1.
    """
    if epochs < 1 or steps_per_epoch < 1:
        raise ValueError(
            f"the LR schedule needs a positive horizon, got epochs={epochs}, "
            f"steps_per_epoch={steps_per_epoch}"
        )

    schedule = create_warmup_lr_schedule(
        config.learning_rate, epochs, steps_per_epoch, config.warmup_ratio
    )
    model.compile(
        optimizer=optimizer_builder(
            {
                "type": "adamw",
                "beta_1": 0.9,
                "beta_2": 0.95,
                "weight_decay": config.weight_decay,
                "gradient_clipping_by_norm": 1.0,
                "exclude_from_weight_decay": [
                    "bias", "gamma", "beta", "embedding",
                ],
            },
            schedule,
        )
    )
    logger.info(
        f"Compiled: AdamW(beta1=0.9, beta2=0.95, wd={config.weight_decay}, "
        f"global_clipnorm=1.0, decay excluded for bias/gamma/beta/embedding), "
        f"peak_lr={config.learning_rate} warmup {config.warmup_ratio:.0%} then "
        f"cosine over {epochs * steps_per_epoch} steps"
    )


# ---------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------


def resolve_cadence(
    config: TopoLMTrainingConfig,
    real_steps_per_epoch: int,
) -> Tuple[int, int]:
    """Resolve ``fit``'s ``(epochs, steps_per_epoch)`` for the virtual cadence.

    Validation has to fire every ``eval_every_steps`` optimizer steps, and Keras
    only evaluates at epoch ends. So ``steps_per_epoch`` IS the cadence, and the
    run's step count becomes ``eval_every_steps * epochs`` -- which is *not* one
    pass over the data. Two ways to pin it:

    - set ``train_steps``, and the epoch count is derived from the cadence, so the
      step count is exact and the cadence is unchanged;
    - leave it ``None`` and the step count is ``eval_every_steps * num_epochs``,
      with ``real_steps_per_epoch`` reported so the discrepancy is visible in the
      log rather than inferred later from a short run.

    :param config: The run's configuration.
    :type config: TopoLMTrainingConfig
    :param real_steps_per_epoch: Steps one real pass over the data would take.
    :type real_steps_per_epoch: int
    :return: ``(epochs, steps_per_epoch)`` for ``fit``.
    :rtype: Tuple[int, int]
    :raises ValueError: If ``eval_every_steps`` is below 1, or if ``train_steps``
        is set and below 1 -- naming the offending value.
    """
    if config.eval_every_steps < 1:
        raise ValueError(
            f"eval_every_steps must be >= 1, got {config.eval_every_steps}"
        )

    cadence = config.eval_every_steps
    if config.train_steps is not None:
        if config.train_steps < 1:
            raise ValueError(
                f"train_steps must be >= 1, got {config.train_steps}"
            )
        epochs = max(1, int(math.ceil(config.train_steps / cadence)))
        source = f"train_steps={config.train_steps} at a {cadence}-step cadence"
    else:
        epochs = max(1, config.num_epochs)
        source = (
            f"num_epochs={config.num_epochs} at a {cadence}-step cadence, "
            f"i.e. {epochs * cadence} steps against {real_steps_per_epoch} for "
            f"one real pass over the data"
        )

    logger.info(f"Cadence: {source}")
    return epochs, cadence


# ---------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------


class TapCapture(keras.layers.Layer):
    """Holds the branch tensor flowing through one tap.

    Reading activations out of a SUBCLASSED model is not what
    ``keras.Model(inputs=..., outputs=layer.output)`` is for: there is no
    ``layer.output`` on a model that was never built functionally, and wrapping
    internal tensors in a Functional graph would mean rebuilding the stack under a
    second model. An identity layer placed on the stack through
    :meth:`TopoLM.call`'s ``taps`` argument is the subclassed-model equivalent,
    and it needs no rebuild because the captures return their inputs unchanged.

    :param name: Optional layer name; defaults to ``'tap_capture'``.
    :type name: Optional[str]
    """

    def __init__(self, name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(name=name or "tap_capture", **kwargs)
        #: The captured tensor, or ``None`` before the first forward pass.
        self.value: Optional[keras.KerasTensor] = None

    def call(
        self, inputs: keras.KerasTensor, training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Record ``inputs`` and return them unchanged.

        :param inputs: The tapped branch tensor.
        :type inputs: keras.KerasTensor
        :param training: Unused; the capture is a pure observer.
        :type training: Optional[bool]
        :return: ``inputs``, unchanged.
        :rtype: keras.KerasTensor
        """
        del training
        self.value = inputs
        return inputs


def extract_tap_activations(
    backbone: TopoLM,
    prompts: Sequence[str],
    preprocessor,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Mean-pooled activations at every tap, keyed by tap path.

    Pooling is a MEAN over each stimulus's token positions, not a sum, so a longer
    encoding does not read as a stronger response. That confound is not
    hypothetical: the paper's own footnote 9 reports it for a response-profile
    analysis run on stimuli that still carried their determiners, and a sum-pool
    would reproduce it exactly.

    :param backbone: The trained model.
    :type backbone: TopoLM
    :param prompts: Stimulus strings.
    :type prompts: Sequence[str]
    :param preprocessor: Tokenizer from ``train.common.nlp``.
    :type preprocessor: Any
    :param batch_size: Forward-pass batch size.
    :type batch_size: int
    :return: ``{tap_path: (num_prompts, num_units)}`` in original unit order, NOT
        grid order -- every consumer maps through the tap's own
        ``layout.cell_to_unit``.
    :rtype: Dict[str, numpy.ndarray]
    :raises ValueError: If ``prompts`` is empty.
    """
    prompts = list(prompts)
    if not prompts:
        raise ValueError("prompts must not be empty")

    # POSITIONAL, not a dict. `TiktokenPreprocessor.__call__` accepts a string
    # or a LIST of strings and raises TypeError on anything else -- the mapping
    # form is HF-dataset-shaped, which is what the *dataset* pipeline uses and
    # not this callable. The padded result is fine here: activations are pooled
    # over positions, so padding contributes an equal share to every condition.
    tokens = np.asarray(preprocessor(prompts)["input_ids"], dtype="int32")

    if not backbone.built:
        # A subclassed model's sub-layers have no `path` until something has built
        # them, so the `tap.path` keys this function returns would all be `None`
        # and every downstream consumer would collide on one entry. One eager
        # forward fixes it, and it is also what the function's contract implies:
        # a caller asking for activations should not have to know that the model
        # has to be built first.
        backbone(tokens[:1], training=False)

    captured: Dict[str, List[np.ndarray]] = {
        tap.path: [] for tap in backbone.tap_layers
    }
    for start in range(0, tokens.shape[0], batch_size):
        chunk = tokens[start:start + batch_size]
        captures = [TapCapture() for _ in backbone.tap_layers]
        backbone(chunk, training=False, taps=tuple(captures))
        for tap, capture in zip(backbone.tap_layers, captures):
            if capture.value is None:
                raise RuntimeError(
                    f"tap {tap.path} produced no activation for a "
                    f"{chunk.shape[0]}-prompt batch"
                )
            pooled = np.mean(
                ops.convert_to_numpy(capture.value).astype("float64"),
                axis=1,
            )
            captured[tap.path].append(pooled)

    return {name: np.concatenate(rows) for name, rows in captured.items()}


# ---------------------------------------------------------------------
# Topographic evaluation
# ---------------------------------------------------------------------


def evaluate_topography(
    backbone: TopoLM,
    stimuli: Mapping[str, Sequence[str]],
    contrast_conditions: Tuple[str, str],
    preprocessor,
    output_dir: Optional[str] = None,
    readout_fwhm: Optional[float] = None,
    readout_unit_spacing: float = 1.0,
    min_cluster_size: int = 10,
    fdr_alpha: float = 0.05,
    permutation_p_value: bool = False,
    num_permutations: int = 9999,
    seed: int = 0,
    is_smoke_set: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Score the model's topographic organisation against one contrast.

    For each tap, in a raw arm and (when configured) a simulated-fMRI-readout arm:

    1. a two-sample t-map between the two conditions, on activations that are
       MEAN-POOLED over token positions;
    2. BH-FDR corrected ACROSS ALL TAPS, then split back;
    3. clusters grown from the surviving cells, both polarities;
    4. Moran's I on the UNTHRESHOLDED map, plus the islands variant.

    Two ordering rules in that list are load-bearing and not interchangeable. The
    correction is joint across taps because the paper's figures read as one
    organism with one map per layer; correcting each tap separately admits
    roughly 470 false positives per 784-unit layer, and the map's threshold stops
    being a statement about the model. And Moran's I is scored on the raw map, not
    the mask: a patch of zeros is a patch of agreement, so thresholding first
    manufactures the clustering the statistic is meant to measure.

    :param backbone: The trained model.
    :type backbone: TopoLM
    :param stimuli: ``{condition: [sentence, ...]}``.
    :type stimuli: Mapping[str, Sequence[str]]
    :param contrast_conditions: The two conditions contrasted.
    :type contrast_conditions: Tuple[str, str]
    :param preprocessor: Tokenizer.
    :type preprocessor: Any
    :param output_dir: Directory for the JSON report, or ``None`` to skip writing.
    :type output_dir: Optional[str]
    :param readout_fwhm: FWHM for the smoothed arm, or ``None`` to skip it.
    :type readout_fwhm: Optional[float]
    :param readout_unit_spacing: Inter-unit spacing for the readout kernel.
    :type readout_unit_spacing: float
    :param min_cluster_size: Smallest cluster the sweep keeps.
    :type min_cluster_size: int
    :param fdr_alpha: BH-FDR rejection threshold, applied ONCE across all taps.
    :type fdr_alpha: float
    :param permutation_p_value: When ``True``, also score Moran's I against
        spatial randomness by permutation and record the p-value per tap. This
        is the third of the paper's three Moran statistics, and it is OFF by
        default because it is not cheap: ``num_permutations`` shuffles of the
        map per tap per arm, so the default 9999 costs 9999 x 24 x arms
        recomputations. The standard and islands values come free with the map.
    :type permutation_p_value: bool
    :param num_permutations: Shuffles behind the permutation p-value. Also sets
        its floor: the smallest attainable p-value is ``1 / (n + 1)``.
    :type num_permutations: int
    :param seed: Base seed for the permutation shuffles, mixed with each tap's
        path so the p-values are reproducible across runs and independent across
        taps.
    :type seed: int
    :param is_smoke_set: Record in the report that the stimuli are the built-in
        stand-in rather than a published set.
    :type is_smoke_set: bool
    :return: ``(report, arrays)``. The report is the JSON summary for every arm.
        ``arrays["arms"][arm]`` holds the t-maps, BH-FDR masks and cluster label
        grids **for that arm specifically**, so :func:`plot_topography` draws
        each figure from the numbers that arm scored rather than recomputing
        them. Returned separately because a report is JSON-serialisable and
        these are not.
    :rtype: Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, numpy.ndarray]]]
    :raises ValueError: If the two contrast conditions are absent from
        ``stimuli`` -- naming them -- or ``num_permutations`` is not positive.
    """
    condition_a, condition_b = contrast_conditions
    missing = [
        name for name in contrast_conditions if name not in stimuli
    ]
    if missing:
        raise ValueError(
            f"contrast conditions {missing} absent from the stimuli "
            f"(available: {sorted(stimuli)})"
        )
    # Validated HERE rather than left to morans_i_permutation_test, so the
    # message names the caller's argument instead of surfacing from inside a
    # 9999-iteration loop one call deeper.
    if permutation_p_value and num_permutations <= 0:
        raise ValueError(
            f"num_permutations must be positive when permutation_p_value is set, "
            f"got {num_permutations}"
        )

    grid_shape = backbone.grid_shape
    report: Dict[str, Any] = {
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "contrast": [condition_a, condition_b],
        "conditions": sorted(stimuli),
        "stimuli_are_smoke_set": bool(is_smoke_set),
        "fdr_alpha": float(fdr_alpha),
        "fdr_scope": "joint across all taps",
        "num_taps": len(backbone.tap_layers),
        "arms": {},
    }

    # PER-ARM. Keyed by arm, because the arms are NOT interchangeable and a
    # single set of arrays cannot serve both: the readout arm's t-maps are a
    # Gaussian-blurred version of the raw ones and differ from them by tens of
    # t-units (MEASURED: max |raw - readout| = 22.99 on a four-prompt-per-
    # condition contrast). Holding only the last arm's arrays and then writing a
    # file per arm produced two byte-identical panels under different names, so
    # `t_maps_raw.png` showed smoothed data labelled "raw". Keyed per arm, each
    # figure is fed the arrays that arm scored.
    plotted_arms: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for arm in (["raw", "readout"] if readout_fwhm else ["raw"]):
        plotted: Dict[str, np.ndarray] = {}
        plotted_sig: Dict[str, np.ndarray] = {}
        plotted_labels: Dict[str, np.ndarray] = {}
        per_condition = {
            condition: extract_tap_activations(
                backbone, prompts, preprocessor
            )
            for condition, prompts in stimuli.items()
        }
        if arm == "readout":
            per_condition = {
                condition: _apply_readout(
                    activations, backbone, readout_fwhm, readout_unit_spacing
                )
                for condition, activations in per_condition.items()
            }

        t_maps: Dict[str, np.ndarray] = {}
        p_maps: Dict[str, np.ndarray] = {}
        for name, condition_a_values in per_condition[condition_a].items():
            t_values, p_values = contrast_tmap(
                condition_a_values, per_condition[condition_b][name]
            )
            t_maps[name] = t_values
            p_maps[name] = p_values

        corrected = fdr_across_layers(p_maps, alpha=fdr_alpha)

        tap_reports: Dict[str, Any] = {}
        for tap in backbone.tap_layers:
            name = tap.path
            sig_grid = np.asarray(
                corrected[name]["reject"], dtype=bool
            ).reshape(grid_shape)
            t_grid = t_maps[name].reshape(grid_shape)

            clusters: Dict[str, List[List[int]]] = {}
            for sign, label in ((1, condition_a), (-1, condition_b)):
                grown, label_grid = grow_clusters(
                    t_grid,
                    sig_grid,
                    sign=sign,
                    min_size=min_cluster_size,
                    connectivity="queen",
                    cell_to_unit=tap.layout.cell_to_unit,
                )
                clusters[label] = [cluster.tolist() for cluster in grown]
                # Keep the dominant polarity's labels for the figure. One map
                # cannot show both polarities' boundaries at once without
                # inventing a distinction the label grid does not carry, so the
                # positive one is drawn and the counts stay in the JSON.
                if sign == 1:
                    plotted[name] = t_grid
                    plotted_sig[name] = sig_grid
                    plotted_labels[name] = label_grid

            tap_reports[name] = {
                # The permutation seed is DERIVED from the run seed and the tap's
                # depth, not left to a global RNG: the p-values then depend only
                # on (seed, map), so re-running one arm reproduces them and two
                # taps are not correlated through a shared shuffle sequence.
                "morans_i": morans_i_summary(
                    t_grid,
                    sig_grid,
                    num_permutations=(
                        num_permutations if permutation_p_value else None
                    ),
                    seed=None
                    if not permutation_p_value
                    else _permutation_seed(seed, name),
                ),
                "num_significant_units": int(sig_grid.sum()),
                "cluster_sizes_a": [len(c) for c in clusters[condition_a]],
                "cluster_sizes_b": [len(c) for c in clusters[condition_b]],
            }

        standard = [
            entry["morans_i"]["standard"] for entry in tap_reports.values()
        ]
        finite = [value for value in standard if not np.isnan(value)]
        significant = [
            entry["num_significant_units"] for entry in tap_reports.values()
        ]
        # Recorded per arm, BEFORE the next iteration rebinds these names. The
        # arrays are snapshots of this arm's own t-maps, masks and labels.
        plotted_arms[arm] = {
            "t_maps": plotted,
            "sig_grids": plotted_sig,
            "label_grids": plotted_labels,
        }

        report["arms"][arm] = {
            "readout_fwhm": readout_fwhm if arm == "readout" else None,
            "per_tap": tap_reports,
            "mean_morans_i": (
                float(np.mean(finite)) if finite else float("nan")
            ),
            "mean_num_significant_units": (
                float(np.mean(significant)) if significant else float("nan")
            ),
            "total_clusters_a": int(
                sum(len(v) for v in _sizes(tap_reports, "cluster_sizes_a"))
            ),
            "total_clusters_b": int(
                sum(len(v) for v in _sizes(tap_reports, "cluster_sizes_b"))
            ),
        }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "topography_report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        logger.info(f"Topographic report written to {path}")

    return report, {"arms": plotted_arms}


def _permutation_seed(base_seed: int, tap_path: str) -> int:
    """A stable per-tap permutation seed, derived rather than drawn.

    ``hash()`` is not used: it is salted per process, so the p-values would
    change between runs of the same command and a report could not be
    reproduced. This is a plain integer mix of the run seed and the tap's path,
    so the same run seed and the same tap always give the same shuffles, and two
    different taps never share a sequence.
    """
    digest = 0
    for character in tap_path:
        digest = (digest * 131 + ord(character)) % (2**31 - 1)
    return int((base_seed * 1_000_003 + digest) % (2**31 - 1))


def plot_topography(
    report: Dict[str, Any],
    arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: str,
) -> List[str]:
    """Render the paper's figure set for every arm in ``report``.

    Called after :func:`evaluate_topography`, and each figure is fed that arm's
    OWN arrays, so a cluster outlined in a figure is the cluster counted in that
    arm's section of the JSON. Keying the arrays by arm is the whole point:
    they are not interchangeable, and feeding one arm's arrays to another's
    figure produced two byte-identical panels under different names --
    ``t_maps_raw.png`` showing readout-smoothed data labelled "raw", wrong by up
    to 22.99 t-units on a four-prompt contrast (MEASURED, on a real run's
    artefacts: the two panels' pixels differed by 0).

    Imported lazily so that importing :mod:`train.topolm.common` does not pull
    in matplotlib; the analysis is usable, and testable, without a display.

    :param report: The report returned by :func:`evaluate_topography`.
    :type report: Dict[str, Any]
    :param arrays: The second return value of :func:`evaluate_topography` --
        ``{"arms": {arm: {"t_maps", "sig_grids", "label_grids"}}}``.
    :type arrays: Dict[str, Dict[str, Dict[str, numpy.ndarray]]]
    :param output_dir: Directory for the PNGs.
    :type output_dir: str
    :return: Paths written.
    :rtype: List[str]
    :raises ValueError: If ``arrays`` carries an arm the report does not, or
        vice versa -- naming the difference, because that mismatch is what makes
        a figure disagree with the report it sits beside.
    """
    from train.topolm.plotting import (  # local: keeps matplotlib off the
        plot_cluster_overlay,  # analysis import path
        plot_morans_i_profile,
        plot_t_maps,
    )

    condition_a, condition_b = report["contrast"]
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []

    arms = arrays["arms"]
    if set(arms) != set(report["arms"]):
        raise ValueError(
            f"arrays carry arms {sorted(arms)} but the report has "
            f"{sorted(report['arms'])}; a figure drawn from one arm's arrays "
            f"under another arm's name would silently misreport the result"
        )

    for arm, values in report["arms"].items():
        arm_arrays = arms[arm]
        written.append(
            plot_t_maps(
                arm_arrays["t_maps"],
                sig_grids=arm_arrays["sig_grids"],
                label_grids=arm_arrays["label_grids"],
                arm=arm,
                condition_a=condition_a,
                condition_b=condition_b,
                output_path=os.path.join(output_dir, f"t_maps_{arm}.png"),
            )
        )
        written.append(
            plot_morans_i_profile(
                {
                    name: entry["morans_i"]["standard"]
                    for name, entry in values["per_tap"].items()
                },
                output_path=os.path.join(output_dir, f"morans_i_{arm}.png"),
                arm=arm,
            )
        )

    # One cluster figure PER ARM, and named for its arm. It used to be a single
    # arm-less `clusters.png` built from whichever arrays survived the loop, so a
    # reader could not tell which arm a categorical map belonged to.
    for arm, arm_arrays in arms.items():
        labels = arm_arrays["label_grids"]
        # The deepest tap that actually found a cluster, rather than the last
        # tap: on an untrained or non-topographic run the last tap's grid is
        # empty and the figure would show nothing.
        populated = [name for name, grid in labels.items() if np.any(grid > 0)]
        if populated:
            deepest = populated[-1]
            written.append(
                plot_cluster_overlay(
                    labels[deepest],
                    output_path=os.path.join(output_dir, f"clusters_{arm}.png"),
                    title=f"{deepest.split('/')[-1]} ({arm})",
                )
            )

    return [path for path in written if path is not None]


def _sizes(tap_reports: Mapping[str, Any], key: str) -> List[List[int]]:
    """Every tap's cluster-size list for one polarity."""
    return [entry[key] for entry in tap_reports.values()]


def _apply_readout(
    activations: Mapping[str, np.ndarray],
    backbone: TopoLM,
    fwhm: Optional[float],
    unit_spacing: float,
) -> Dict[str, np.ndarray]:
    """Blur activations over the grid, BEFORE any statistic is computed.

    Smoothing a t-map is a different operation and inflates apparent structure, so
    the order here is the paper's and is not interchangeable. Each tap gets its
    OWN readout, carrying that tap's layout -- under the default per-tap
    permutation a single shared layout would blur every layer as if they all sat
    on the same grid, which they do not.

    :param activations: ``{tap_path: (num_prompts, num_units)}``.
    :type activations: Mapping[str, numpy.ndarray]
    :param backbone: The model the activations came from.
    :type backbone: TopoLM
    :param fwhm: Full width at half maximum in grid units.
    :type fwhm: Optional[float]
    :param unit_spacing: Inter-unit spacing in the same units.
    :type unit_spacing: float
    :return: The smoothed activations, same shapes.
    :rtype: Dict[str, numpy.ndarray]
    """
    readouts = {
        tap.path: GaussianReadout(
            fwhm=float(fwhm),
            unit_spacing=unit_spacing,
            permute=tap.layout.permute,
            grid_shape=tap.layout.grid_shape,
            seed=tap.layout.seed,
            name=f"readout_{index}",
        )
        for index, tap in enumerate(backbone.tap_layers)
    }

    smoothed: Dict[str, np.ndarray] = {}
    for name, values in activations.items():
        readout = readouts[name]
        smoothed[name] = ops.convert_to_numpy(
            readout(
                ops.convert_to_tensor(np.asarray(values, "float32")),
                training=False,
            )
        ).astype("float64")
    return smoothed


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def build_callbacks(
    config: TopoLMTrainingConfig,
    model: CausalLanguageModel,
    results_dir: str,
    initial_step: int,
) -> List[keras.callbacks.Callback]:
    """Assemble the run's callbacks.

    Built here rather than through ``create_nlp_callbacks`` because that helper
    unconditionally installs ``keras.callbacks.EarlyStopping``, and the paper's
    stopping rule is a different question (see the module docstring). Having both
    would be two independent stop signals on one run.

    :param config: The run's configuration.
    :type config: TopoLMTrainingConfig
    :param model: The training head, for the generation probe's closure.
    :type model: CausalLanguageModel
    :param results_dir: The run directory, for checkpoints and probe records.
    :type results_dir: str
    :param initial_step: Resume point, so the step counters start there.
    :type initial_step: int
    :return: The callback list, in the order they fire.
    :rtype: List[keras.callbacks.Callback]
    """
    probe = GenerationProbeCallback(
        # `CausalLanguageModel` with `output_key="logits"` returns the logits
        # TENSOR directly -- the key is extracted inside `_backbone_forward`, so
        # subscripting `["logits"]` here would raise. What the probe wants is a
        # single row of vocabulary logits for the LAST position of the context,
        # and this closure owns that decision, as it does for every other trainer.
        logits_fn=lambda ctx: ops.convert_to_numpy(
            model(ctx, training=False)
        )[0, -1, :],
        probe_every_steps=config.checkpoint_every_steps,
        prompts=config.probe_prompts,
        encoding_name=config.encoding_name,
        max_tokens=config.probe_max_tokens,
        temperature=config.probe_temperature,
        top_p=config.probe_top_p,
        repetition_penalty=config.probe_repetition_penalty,
        save_dir=results_dir,
        initial_step=initial_step,
        ctx_length=min(511, max(1, config.max_seq_length - 1)),
    )
    probe._post_generate_hook = augment_probe_results

    return [
        ConsecutiveIncreaseEarlyStopping(
            monitor="val_loss",
            patience=config.early_stop_patience,
            restore_weights=True,
            verbose=1,
        ),
        SpatialLossLogger(
            log_every=config.spatial_log_every, verbose=0
        ),
        StepCheckpointCallback(
            save_dir=results_dir,
            save_every_steps=config.checkpoint_every_steps,
            analyze_every_steps=config.analyze_every_steps,
            max_checkpoints=config.max_checkpoints,
            model_name=f"TopoLM-{config.model_variant}",
            initial_step=initial_step,
        ),
        probe,
    ]


def train_topolm(
    config: TopoLMTrainingConfig,
    preprocessor=None,
    run_topography: bool = True,
) -> Dict[str, Any]:
    """Train one TopoLM arm.

    :param config: The run's configuration.
    :type config: TopoLMTrainingConfig
    :param preprocessor: Tokenizer, or ``None`` to build one from ``config``.
    :type preprocessor: Any
    :param run_topography: Run the post-hoc topographic evaluation. ``False``
        skips it, which is the shape a unit test wants.
    :type run_topography: bool
    :return: ``{"model", "backbone", "history", "results_dir",
        "epochs", "steps_per_epoch", "topography"}``.
    :rtype: Dict[str, Any]
    """
    set_seeds(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    if preprocessor is None:
        preprocessor = create_tokenizer(
            config.encoding_name,
            config.max_seq_length,
            config.cls_token_id,
            config.sep_token_id,
            config.pad_token_id,
            config.mask_token_id,
        )

    # Before the dataset is built: the message can only name --vocab-size while
    # the flag is still the obvious suspect. After `preprocess_clm_dataset` has
    # run, the same error is an out-of-range Gather from inside Embedding.call
    # and reads like a model bug.
    _require_vocab_covers_tokenizer(config, preprocessor)

    # Derived BEFORE the dataset is built and shifted by the resume point, so a
    # resumed run sees a new article ordering instead of replaying the first N
    # chunks.
    initial_step = (
        extract_step_from_checkpoint(config.resume_from)
        if config.resume_from
        else 0
    )
    data_seed = config.seed + initial_step

    train_dataset, val_dataset, n_train_articles = load_train_val_datasets(
        config,
        preprocessor,
        data_seed=data_seed,
        # The CLM head is `pre_shifted`, so the label stream is a plain tensor; a
        # dict-keyed label reaches `MaskedCausalLMLoss` as a dict and raises.
        wrap_for_dict_output=False,
    )

    real_steps_per_epoch = make_clm_steps_per_epoch(config, n_train_articles)
    epochs, steps_per_epoch = resolve_cadence(config, real_steps_per_epoch)

    results_dir = str(
        prepare_run_dir(
            config, output_dir=_run_dir(config), config_filename="config.json"
        )
    )

    model, backbone = create_topolm_model(
        config,
        required_vocab_size=getattr(preprocessor, "vocab_size", None),
    )
    compile_model(model, config, epochs, steps_per_epoch)
    callbacks = build_callbacks(config, model, results_dir, initial_step)

    logger.info(
        f"Starting training: source={config.dataset_source}, "
        f"batch_size={config.batch_size}, {epochs} virtual epoch(s) of "
        f"{steps_per_epoch} step(s) = {epochs * steps_per_epoch} total, "
        f"validating every {steps_per_epoch} step(s) over "
        f"{config.eval_batches} batch(es)"
    )

    history = model.fit(
        train_dataset.repeat(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=config.eval_batches,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_history_json(history, results_dir)
    generate_training_curves(history, results_dir)

    summary: Dict[str, Any] = {}
    if "val_loss" in history.history:
        values = history.history["val_loss"]
        best = int(min(range(len(values)), key=lambda i: values[i]))
        summary = {
            "best_evaluation": best,
            "best_val_loss": float(values[best]),
            "final_val_loss": float(values[-1]),
            "evaluations_run": len(values),
            "stopped_early": len(values) < epochs,
        }
        logger.info(
            f"Best evaluation {best}: val_loss={values[best]:.4f} "
            f"of {len(values)} run"
            + (
                " (early stopping fired)"
                if summary["stopped_early"]
                else ""
            )
        )

    result: Dict[str, Any] = {
        "model": model,
        "backbone": backbone,
        "history": history,
        "results_dir": results_dir,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "summary": summary,
    }

    if run_topography:
        result["topography"] = run_topographic_evaluation(
            backbone, config, preprocessor, results_dir
        )

    return result


def run_topographic_evaluation(
    backbone: TopoLM,
    config: TopoLMTrainingConfig,
    preprocessor,
    results_dir: str,
) -> Dict[str, Any]:
    """Score a trained backbone and log the headline numbers.

    :param backbone: The trained model.
    :type backbone: TopoLM
    :param config: The run's configuration.
    :type config: TopoLMTrainingConfig
    :param preprocessor: Tokenizer.
    :type preprocessor: Any
    :param results_dir: Directory for the JSON report.
    :type results_dir: str
    :return: The report dictionary.
    :rtype: Dict[str, Any]
    """
    report, arrays = evaluate_topography(
        backbone,
        config.stimuli if config.stimuli else SMOKE_STIMULI,
        config.contrast_conditions,
        preprocessor,
        output_dir=results_dir,
        readout_fwhm=config.readout_fwhm,
        readout_unit_spacing=config.readout_unit_spacing,
        min_cluster_size=config.min_cluster_size,
        permutation_p_value=config.permutation_p_value,
        num_permutations=config.num_permutations,
        seed=config.seed,
        is_smoke_set=config.stimuli is None,
    )
    for arm, values in report["arms"].items():
        logger.info(
            f"Topography [{arm}]: mean Moran's I = "
            f"{values['mean_morans_i']:.3f}, mean significant units = "
            f"{values['mean_num_significant_units']:.1f}, clusters "
            f"{values['total_clusters_a']} (for '{report['contrast'][0]}') / "
            f"{values['total_clusters_b']} (for '{report['contrast'][1]}')"
        )
    figures = plot_topography(report, arrays, results_dir)
    if figures:
        logger.info(f"Topographic figures written: {', '.join(figures)}")
    if report["stimuli_are_smoke_set"]:
        logger.warning(
            "These topography numbers came from the BUILT-IN smoke stimuli, not "
            "a published stimulus set. They demonstrate that the pipeline runs "
            "end to end; they are not a result."
        )
    return report


def _run_dir(config: TopoLMTrainingConfig) -> str:
    """The run directory: ``<save_dir>/topolm_<variant>_<arm>_alpha<a>_<stamp>``.

    ``alpha`` is in the name because the control arm writes into the same tree,
    and the two must not be confusable when they are read back months apart.
    """
    arm = "topo" if config.spatial_alpha > 0 else "control"
    return os.path.join(
        config.save_dir,
        f"topolm_{config.model_variant}_{arm}"
        f"_alpha{config.spatial_alpha:g}_{run_timestamp()}",
    )


def train_paired(
    config: TopoLMTrainingConfig,
    preprocessor=None,
    run_topography: bool = True,
) -> Dict[str, Any]:
    """Train the topographic model and its ``alpha = 0`` control, back to back.

    The control is the paper's comparison and the point of running it, so the two
    arms share a seed, a data ordering and every hyperparameter except ``alpha``.
    The control's taps are still created and built, which is what makes that true
    at the weight level rather than only at the config level.

    :param config: The topographic arm's configuration.
    :type config: TopoLMTrainingConfig
    :param preprocessor: Tokenizer, forwarded to BOTH arms, or ``None`` to let
        each build its own. Both arms must see the same one, since a tokenizer is
        part of the data pipeline the two arms are supposed to share.
    :type preprocessor: Any
    :param run_topography: Run the post-hoc evaluation on BOTH arms, not just
        the first. It has to reach both or the comparison cannot be made, and
        this is forwarded rather than inferred because ``--no-topography`` is a
        CLI flag that arrives here: without the forwarding, ``--paired
        --no-topography`` still wrote a full report and figure set for the
        topographic arm. MEASURED.
    :type run_topography: bool
    :return: ``{"topographic": <result dict>, "control": <result dict>}``.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``config.spatial_alpha`` is already ``0`` -- there is
        then nothing to pair it against.
    """
    if config.spatial_alpha <= 0.0:
        raise ValueError(
            f"spatial_alpha is {config.spatial_alpha}; a paired run needs a "
            f"positive topographic arm to compare against"
        )

    control = replace(config, spatial_alpha=0.0)

    logger.info("=" * 60)
    logger.info(
        f"ARM 1/2: topographic (alpha = {config.spatial_alpha}, "
        f"{config.spatial_radius}-radius neighbourhoods)"
    )
    topographic = train_topolm(
        replace(config),
        preprocessor=preprocessor,
        run_topography=run_topography,
    )

    logger.info("=" * 60)
    logger.info(
        "ARM 2/2: non-topographic control (alpha = 0) -- identical weights, "
        "no spatial term in the objective"
    )
    control_result = train_topolm(
        control, preprocessor=preprocessor, run_topography=run_topography
    )

    for name, result in (
        ("topographic", topographic), ("control", control_result)
    ):
        summary = result["summary"]
        if summary:
            logger.info(
                f"{name}: best val_loss={summary['best_val_loss']:.4f} at "
                f"evaluation {summary['best_evaluation']} of "
                f"{summary['evaluations_run']}"
            )

    if (
        topographic["summary"] and control_result["summary"]
    ):
        delta = (
            control_result["summary"]["best_val_loss"]
            - topographic["summary"]["best_val_loss"]
        )
        logger.info(
            f"Task cost of topography: control is {delta:+.4f} nats versus the "
            f"topographic model. The paper reports +0.109 for its own pair."
        )

    return {"topographic": topographic, "control": control_result}


# ---------------------------------------------------------------------