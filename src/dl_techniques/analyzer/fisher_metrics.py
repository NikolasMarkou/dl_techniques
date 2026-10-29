"""
Diagonal model-Fisher measurement for Keras models.

This module measures the diagonal of the *model* Fisher information of any
differentiable probabilistic model, following arXiv:2609.36262 (Zhang, Mukherjee,
Peng; "Understanding LLM Parameter Update Sparsity through the Lens of Fisher").
It is deliberately broader than the paper: the math never assumed a language model,
so classifiers, multi-label heads, Gaussian regressors and autoregressive models
share one estimator.

Foundational Mathematics
------------------------
Let theta in R^P be the parameters, x an input from a fixed distribution D_x and
pi_theta(y|x) the model's own output distribution.

-   **Score and Fisher (paper Eq. 2).** With the score
    h_i(x, y) = d_i log pi_theta(y|x) and y sampled from the MODEL (never from
    labels),
        F_i = E_x E_{y~pi_theta}[ h_i(x, y)^2 ].
    Since E_y[h_i] = d_i sum_y pi_theta(y|x) = 0, F_i = Var[h_i]. This is the model
    Fisher, NOT the empirical Fisher (which squares loss gradients at labels).

-   **One backward pass per sample.** differentiating the scalar log pi_theta(y|x)
    w.r.t. every trainable variable at once yields the whole score vector h, so a
    pass of N samples costs N backward passes, not N * P.

-   **Fisher bounds the gradient (paper Prop. 1).** For J = E_x E_{y~mu}[R] with
    gradient g_i = E_{pi_theta}[a(x, y) h_i(x, y)], Cauchy-Schwarz gives
        |g_i| <= sqrt(E[a^2]) * sqrt(F_i).
    Signals a (paper Table 2):
        rl        a = r - mean_group(r)
        sft       a = nu/pi_theta - 1, y ~ nu, so g_i = E_nu[h_i]
        opd_pg    a = R_T - mean_group(R_T),  R_T = sum_t log q_t(y_t)/p_t(y_t)
        opd_full  a = sum_t ( q_t(y_t)/p_t(y_t) - 1 )

-   **Token decomposition (paper Prop. 2).** For y = (y_1..y_T) with prefix s_t,
    token score xi_{t,i} = d_i log p(y_t|s_t) is a martingale difference sequence, so
        F_i = E[ sum_t f_i(s_t) ],   f_i(s) = Var_{a~p}[ j_{i,a}(s) ]
                                           = 1/2 E_{a,b~iid p}[ (j_{i,a} - j_{i,b})^2 ],
    where j_{i,a} = d_i z_a is the logit sensitivity. The cross terms cancel IN
    EXPECTATION, not per sample: h_i^2 != sum_t xi_{t,i}^2 for one trajectory. The
    sequence estimator (mean of h^2) and the token estimator (mean of sum_t f_i)
    estimate the same quantity with different variances, and are published under
    different names.

-   **Gradient-support recall (Eq. 7) and persistence (Eqs. 12-13).** Recall(q) is
    the overlap of the top-q% Fisher and top-q% |g| coordinate sets; a random subset
    scores q. Retention compares the high/low Fisher sets of two policies; the random
    baseline is q for the high set and 100 - q for the low set.

Estimator conventions kept from the paper: no length normalization (scores are SUMS over
tokens), the gradient estimator keeps its sign and ranks by |g| afterwards, ties are
broken by the canonical coordinate order, observed stopping tokens count and a
length-truncated response is neither discarded nor given a synthetic EOS.

Architecture
------------
Pure functions plus small dataclasses; nothing is parked on ``self``. The module
needs TensorFlow's ``GradientTape`` for the per-sample score (imported lazily in
``_tf``), and uses ``keras.ops`` for the tensor math.
"""

import dataclasses
import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .constants import FisherMetricNames as FM
from .utils import (
    classify_output_activation,
    OUTPUT_ACTIVATION_LOGITS,
    OUTPUT_ACTIVATION_SIGMOID,
    OUTPUT_ACTIVATION_SOFTMAX,
)

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

KIND_CATEGORICAL_LOGITS = 'categorical_logits'
KIND_CATEGORICAL_PROBS = 'categorical_probs'
KIND_BERNOULLI = 'bernoulli'
KIND_GAUSSIAN = 'gaussian'
KIND_SEQUENCE = 'sequence'
KIND_CUSTOM = 'custom'

SIGNAL_RL = 'rl'
SIGNAL_SFT = 'sft'
SIGNAL_OPD_PG = 'opd_pg'
SIGNAL_OPD_FULL = 'opd_full'
SIGNAL_NONE = 'none'
SIGNAL_TYPES = (SIGNAL_RL, SIGNAL_SFT, SIGNAL_OPD_PG, SIGNAL_OPD_FULL, SIGNAL_NONE)
# Signals whose advantage is centered by the per-prompt group mean, hence need G >= 2.
GROUP_CENTERED_SIGNALS = (SIGNAL_RL, SIGNAL_OPD_PG)

ESTIMATOR_SEQUENCE = 'sequence_score_squared'
ESTIMATOR_TOKEN = 'mean_token_fisher'

SCORE_SCOPE_RESPONSE = 'response'
SCORE_SCOPE_RESPONSE_PLUS_PROMPT = 'response_plus_prompt'

DEFAULT_SELECTION_PERCENTAGES: Tuple[float, ...] = (0.05, 0.1, 0.5, 1, 5)
# z-score of the tolerance band used by the bound check (see `bound_violation_rate`).
BOUND_Z = 3.0
# Number of non-finite samples logged individually before the log goes quiet.
MAX_LOGGED_SKIPS = 5

# FisherConfig fields whose AnalysisConfig source is NOT spelled ``fisher_<name>``.
UNPREFIXED_CONFIG_SOURCES: Dict[str, str] = {
    'seed': 'random_state',
    'memory_limit_mb': 'memory_limit_mb',
    'output_activation': 'output_activation',
}

# ---------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------


@dataclass
class FisherConfig:
    """Configuration for one Fisher computation.

    Mirrors the ``fisher_*`` fields of ``AnalysisConfig``; build it with
    :meth:`from_analysis_config` so the two cannot drift (a meta-test pins that every
    field here has a source there).

    Attributes:
        n_samples: N_F, the number of inputs the Fisher estimate averages over.
        n_responses: G, responses per prompt in the gradient arm.
        token_level: Run the token-level arm for sequence models.
        compute_gradients: Run the gradient arm (needs a ``FisherSignal``).
        signal_type: One of ``SIGNAL_TYPES``.
        n_gradient_prompts: N_g, prompts for the gradient arm (disjoint from the
            Fisher prompts).
        max_tokens: Cap on response length for BOTH sampling and scoring.
        n_positions_per_response: Token-level positions per response.
        n_token_pairs_per_position: Token-level (a, b) pairs per position and
            condition.
        batch_size: Sampling microbatch. Scoring is always one sample per backward.
        accumulate_dtype: ``'float32'`` or ``'float64'`` accumulator dtype.
        sampling_temperature: Temperature of the response sampler (categorical and
            sequence only). The paper always uses 1.
        regression_variance: sigma^2 of the Gaussian likelihood; setting it selects
            the Gaussian output kind.
        score_scope: ``'response'`` or ``'response_plus_prompt'`` (sequence models).
        selection_percentages: The q grid for recall and retention.
        zero_threshold: Coordinates with Fisher at or below this are Fisher-dead and
            excluded from recall and retention.
        sft_chi2_cap: chi^2(nu || pi) above which the SFT weights are flagged.
        assert_no_weight_drift: Hash the trainable variables before and after the pass
            and raise if they differ.
        force_sequence: ``None`` infers; ``True``/``False`` forces the sequence path.
        output_activation: Same meaning as ``AnalysisConfig.output_activation``.
        seed: Seed, for provenance (the pass-level generator is passed in).
        memory_limit_mb: Budget the sampling microbatch is sized against.
    """
    n_samples: int = 100
    n_responses: int = 1
    token_level: bool = True
    compute_gradients: bool = False
    signal_type: str = SIGNAL_RL
    n_gradient_prompts: int = 100
    max_tokens: int = 512
    n_positions_per_response: int = 8
    n_token_pairs_per_position: int = 8
    batch_size: int = 8
    accumulate_dtype: str = 'float32'
    sampling_temperature: float = 1.0
    regression_variance: Optional[float] = None
    score_scope: str = SCORE_SCOPE_RESPONSE
    selection_percentages: Tuple[float, ...] = DEFAULT_SELECTION_PERCENTAGES
    zero_threshold: float = 1e-12
    sft_chi2_cap: float = 1e4
    assert_no_weight_drift: bool = True
    force_sequence: Optional[bool] = None
    output_activation: Optional[str] = None
    seed: Optional[int] = None
    memory_limit_mb: Optional[int] = 2048

    @classmethod
    def from_analysis_config(cls, config: Any) -> 'FisherConfig':
        """Build from an ``AnalysisConfig``.

        Args:
            config: The analyzer configuration.

        Returns:
            The validated Fisher configuration.

        Raises:
            ValueError: If the resulting configuration is invalid.
        """
        kwargs: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            source = UNPREFIXED_CONFIG_SOURCES.get(f.name, f'fisher_{f.name}')
            kwargs[f.name] = getattr(config, source)
        if kwargs['selection_percentages'] is None:
            kwargs['selection_percentages'] = DEFAULT_SELECTION_PERCENTAGES
        kwargs['selection_percentages'] = tuple(kwargs['selection_percentages'])
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Reject a configuration that cannot produce a meaningful measurement.

        A wrong configuration is a caller bug, not a measurement outcome, so this
        raises rather than degrading to NaN.

        Raises:
            ValueError: On any invalid combination.
        """
        if self.signal_type not in SIGNAL_TYPES:
            raise ValueError(f"signal_type must be one of {SIGNAL_TYPES}, got {self.signal_type!r}")
        if self.accumulate_dtype not in ('float32', 'float64'):
            raise ValueError(f"accumulate_dtype must be float32 or float64, got {self.accumulate_dtype!r}")
        if self.score_scope not in (SCORE_SCOPE_RESPONSE, SCORE_SCOPE_RESPONSE_PLUS_PROMPT):
            raise ValueError(f"score_scope must be 'response' or 'response_plus_prompt', got {self.score_scope!r}")
        for name in ('n_samples', 'n_responses', 'n_gradient_prompts', 'max_tokens',
                     'n_positions_per_response', 'n_token_pairs_per_position', 'batch_size'):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.sampling_temperature <= 0:
            raise ValueError(f"sampling_temperature must be > 0, got {self.sampling_temperature}")
        if self.regression_variance is not None and self.regression_variance <= 0:
            raise ValueError(f"regression_variance must be > 0, got {self.regression_variance}")
        if any(not (0 < q <= 100) for q in self.selection_percentages):
            raise ValueError(f"selection_percentages must lie in (0, 100], got {self.selection_percentages}")
        if self.compute_gradients:
            if self.signal_type == SIGNAL_NONE:
                raise ValueError("compute_gradients=True needs a signal_type other than 'none'")
            # With G == 1 the group mean equals the single reward, a == 0 for every
            # sample and the bound is satisfied with zero information.
            if self.signal_type in GROUP_CENTERED_SIGNALS and self.n_responses < 2:
                raise ValueError(
                    f"signal_type={self.signal_type!r} centers by the per-prompt group mean, so "
                    f"n_responses must be >= 2 (got {self.n_responses}); with G=1 the advantage "
                    f"is identically zero")


@dataclass
class FisherSignal:
    """Caller-supplied callables and models. Deliberately NOT part of ``AnalysisConfig``.

    ``save_results`` serializes every public config field; a callable there would make
    ``json.dump`` fail and lose the whole artifact.

    Attributes:
        reward_fn: ``(x_i, y_i, target_i) -> float``, read by ``rl`` only. ``x_i``
            carries a leading batch axis of 1, ``y_i`` is one sampled response without
            it, ``target_i`` is the pool-aligned target.
        teacher_model: Teacher for ``opd_pg`` and ``opd_full``; it is evaluated outside
            the gradient tape, so its probabilities never receive gradients.
        nu_logprob_fn: ``(x_i, y_i) -> log nu(y|x)`` for ``sft``.
        output_logprob_fn: ``(model, x_i, y_i) -> scalar tensor`` log-likelihood, for
            multi-output models whose likelihood the analyzer cannot infer. Runs inside
            the tape. Requires ``sampler``.
        sampler: ``(model, x_batch, cfg, rng) -> list of responses`` replacing the
            built-in sampler (models with their own decode loop, custom outputs).
        eos_token_id: Stopping token of the built-in sequence sampler. ``None`` runs
            every response to ``max_tokens``.
    """
    reward_fn: Optional[Callable[[Any, Any, Any], float]] = None
    teacher_model: Optional[keras.Model] = None
    nu_logprob_fn: Optional[Callable[[Any, Any], float]] = None
    output_logprob_fn: Optional[Callable[[keras.Model, Any, Any], Any]] = None
    sampler: Optional[Callable[[keras.Model, Any, 'FisherConfig', np.random.Generator], List[np.ndarray]]] = None
    eos_token_id: Optional[int] = None


@dataclass(frozen=True)
class SamplePlan:
    """Which pool rows feed the Fisher arm and which feed the gradient arm.

    The two index sets are disjoint BY CONSTRUCTION (paper B.1: reusing samples couples
    the noise of F-hat and g-hat). The plan is serializable so a later run (another
    checkpoint of the same architecture, another model) replays the same inputs.

    Attributes:
        n_pool: Size of the pool the indices refer to.
        fisher_indices: Rows for the Fisher arm.
        gradient_indices: Rows for the gradient arm.
        seed: Seed that produced the plan, for provenance.
    """
    n_pool: int
    fisher_indices: Tuple[int, ...]
    gradient_indices: Tuple[int, ...]
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if set(self.fisher_indices) & set(self.gradient_indices):
            raise ValueError("fisher_indices and gradient_indices must be disjoint")
        for idx in (*self.fisher_indices, *self.gradient_indices):
            if not 0 <= idx < self.n_pool:
                raise ValueError(f"index {idx} outside the pool of {self.n_pool} rows")

    @classmethod
    def build(cls, n_pool: int, n_fisher: int, n_gradient: int,
              rng: np.random.Generator, seed: Optional[int] = None) -> 'SamplePlan':
        """Draw a plan without replacement from a pool.

        Args:
            n_pool: Pool size.
            n_fisher: Requested Fisher rows. Clipped to the pool (with a warning)
                when no gradient rows are requested.
            n_gradient: Requested gradient rows.
            rng: The pass-level generator.
            seed: Provenance only.

        Returns:
            The plan.

        Raises:
            ValueError: If the gradient arm is requested and the pool cannot hold two
                disjoint sets.
        """
        if n_gradient > 0 and n_fisher + n_gradient > n_pool:
            raise ValueError(
                f"the pool of {n_pool} rows cannot hold {n_fisher} Fisher and {n_gradient} "
                f"gradient rows disjointly")
        if n_fisher > n_pool:
            logger.warning(f"Fisher: requested {n_fisher} samples but the pool has {n_pool}; using {n_pool}")
            n_fisher = n_pool
        order = rng.permutation(n_pool)
        return cls(
            n_pool=n_pool,
            fisher_indices=tuple(int(i) for i in order[:n_fisher]),
            gradient_indices=tuple(int(i) for i in order[n_fisher:n_fisher + n_gradient]),
            seed=seed,
        )

    def fingerprint(self) -> str:
        """SHA-256 over the pool size and both index sets."""
        payload = repr((self.n_pool, self.fisher_indices, self.gradient_indices)).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form."""
        return {
            'n_pool': self.n_pool,
            'fisher_indices': list(self.fisher_indices),
            'gradient_indices': list(self.gradient_indices),
            'seed': self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> 'SamplePlan':
        """Inverse of :meth:`to_dict`."""
        return cls(
            n_pool=int(payload['n_pool']),
            fisher_indices=tuple(int(i) for i in payload['fisher_indices']),
            gradient_indices=tuple(int(i) for i in payload['gradient_indices']),
            seed=payload.get('seed'),
        )


@dataclass
class ParamLayout:
    """The canonical flat coordinate order of a model's trainable parameters.

    Variables in ``model.trainable_variables`` order, each C-order flattened and
    concatenated. Ties in every ranking are broken by this order.

    Attributes:
        variables: The live variables (not serialized).
        paths: ``variable.path`` per variable (unique).
        shapes: Shape per variable.
        dtypes: dtype name per variable.
        slices: ``(start, stop)`` of each variable in the flat vector.
        total: P, the number of unique coordinates.
        fingerprint: SHA-256 over paths and shapes; a mismatch between two runs is a
            loud error, never a silent re-mapping.
    """
    variables: List[Any]
    paths: List[str]
    shapes: List[Tuple[int, ...]]
    dtypes: List[str]
    slices: List[Tuple[int, int]]
    total: int
    fingerprint: str


@dataclass
class GradientArm:
    """What the gradient arm produced, before it is joined to a ``FisherOutput``.

    Attributes:
        gradient_flat: g-hat with its sign intact, shape (P,).
        gradient_se_flat: Per-coordinate standard error of g-hat, over prompts.
        signal_scale: sqrt(E[a^2]) (paper Prop. 1).
        n_prompts: Prompt groups that contributed.
        skipped: Responses dropped for a non-finite score.
        sft_chi2: chi^2(nu || pi) estimate (``sft`` only).
    """
    gradient_flat: np.ndarray
    gradient_se_flat: np.ndarray
    signal_scale: float
    n_prompts: int
    skipped: int
    sft_chi2: Optional[float] = None


@dataclass
class FisherOutput:
    """Everything one model's Fisher pass produces. Returned, never parked on ``self``.

    ``fisher_flat`` is the SEQUENCE-level estimator (mean of squared sequence scores,
    ``ESTIMATOR_SEQUENCE``). ``token_fisher_mean`` is the MEAN-TOKEN estimator
    (``ESTIMATOR_TOKEN``): it averages the token Fisher over sampled prefixes and is not
    the unnormalized sequence Fisher. They agree in expectation (Prop. 2) but not per
    sample, and are never stored in one field.

    Attributes:
        fisher_flat: F-hat, shape (P,), in the canonical order.
        standard_errors: Per-coordinate standard error of F-hat (Welford).
        param_slices: ``(variable_path, start, stop)`` per trainable variable.
        shapes: ``(variable_path, shape, dtype)`` per trainable variable.
        variable_fingerprint: See :class:`ParamLayout`.
        n_samples: Samples that entered the estimate.
        skipped_samples: Samples dropped for a non-finite score (never clipped).
        output_kind: The likelihood used, one of the ``KIND_*`` constants.
        estimator: ``ESTIMATOR_SEQUENCE``.
        gradient_flat: g-hat, sign intact, or ``None``.
        gradient_se_flat: Standard error of g-hat, or ``None``.
        signal_scale: sqrt(E[a^2]), or ``None`` (never 0.0 for "not computed").
        gradient_recall: ``q -> Recall(q)``, or ``None``.
        violation_rate: Fraction of alive coordinates breaking the envelope, or ``None``.
        token_fisher_mean: Mean token Fisher per parameter (Real), or ``None``.
        uniform_reference: Same under uniform token pairs (Eq. 11), or ``None``.
        group_names: Parameter groups of the token-level matrices.
        token_fisher_by_position: ``(n_positions, n_groups)`` Real group means.
        uniform_by_position: Same under Uniform.
        real_uniform_ratio: ``group -> mean Real / mean Uniform``.
        metrics: Summary scalars keyed by ``FisherMetricNames``.
        samples_used: Provenance: pool index and sampled response per sample.
    """
    fisher_flat: np.ndarray
    standard_errors: Optional[np.ndarray]
    param_slices: List[Tuple[str, int, int]]
    shapes: List[Tuple[str, Tuple[int, ...], str]]
    variable_fingerprint: str
    n_samples: int
    skipped_samples: int
    output_kind: str
    estimator: str = ESTIMATOR_SEQUENCE
    gradient_flat: Optional[np.ndarray] = None
    gradient_se_flat: Optional[np.ndarray] = None
    signal_scale: Optional[float] = None
    gradient_recall: Optional[Dict[float, float]] = None
    violation_rate: Optional[float] = None
    token_fisher_mean: Optional[np.ndarray] = None
    uniform_reference: Optional[np.ndarray] = None
    group_names: Optional[List[str]] = None
    token_fisher_by_position: Optional[np.ndarray] = None
    uniform_by_position: Optional[np.ndarray] = None
    real_uniform_ratio: Optional[Dict[str, float]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    samples_used: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_parameters(self) -> int:
        """P, the number of unique coordinates."""
        return int(self.fisher_flat.shape[0])


# ---------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------

def variable_fingerprint(paths: Sequence[str], shapes: Sequence[Tuple[int, ...]]) -> str:
    """SHA-256 over variable paths and shapes, in order.

    Args:
        paths: Variable paths.
        shapes: Variable shapes.

    Returns:
        A 16-hex-digit digest.
    """
    payload = repr([(p, tuple(int(d) for d in s)) for p, s in zip(paths, shapes)]).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def canonical_layout(model: keras.Model) -> ParamLayout:
    """Flatten the trainable variables into the canonical coordinate order.

    Args:
        model: The model.

    Returns:
        The layout.

    Raises:
        ValueError: If the model has no trainable variables or two variables share a
            path (the path is the join key of every downstream table).
    """
    variables = list(model.trainable_variables)
    if not variables:
        raise ValueError("the model has no trainable variables; its Fisher is empty")
    paths = [str(v.path) for v in variables]
    if len(set(paths)) != len(paths):
        raise ValueError("two trainable variables share a path; cannot build a canonical layout")
    shapes = [tuple(int(d) for d in v.shape) for v in variables]
    slices: List[Tuple[int, int]] = []
    start = 0
    for shape in shapes:
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        slices.append((start, start + size))
        start += size
    return ParamLayout(
        variables=variables, paths=paths, shapes=shapes,
        dtypes=[str(v.dtype) for v in variables], slices=slices, total=start,
        fingerprint=variable_fingerprint(paths, shapes))


def unflatten(flat: np.ndarray, output: FisherOutput) -> Dict[str, np.ndarray]:
    """Cut a flat (P,) vector back into one tensor per variable.

    Args:
        flat: A vector in the canonical order.
        output: The output carrying ``param_slices`` and ``shapes``.

    Returns:
        ``variable_path -> array of the variable's shape``.
    """
    result: Dict[str, np.ndarray] = {}
    for (path, start, stop), (_, shape, _) in zip(output.param_slices, output.shapes):
        result[path] = flat[start:stop].reshape(shape)
    return result


def layer_groups(layout: ParamLayout) -> Tuple[List[str], List[List[Tuple[int, int]]]]:
    """Group variables by their owning layer path.

    Args:
        layout: The layout.

    Returns:
        ``(names, spans)``: group names in first-seen order and the ``(start, stop)``
        spans of each group's variables.
    """
    names: List[str] = []
    spans: Dict[str, List[Tuple[int, int]]] = {}
    for path, sl in zip(layout.paths, layout.slices):
        group = path.rsplit('/', 1)[0]
        if group not in spans:
            names.append(group)
            spans[group] = []
        spans[group].append(sl)
    return names, [spans[n] for n in names]


def _weights_digest(layout: ParamLayout) -> str:
    """SHA-256 over the bytes of every trainable variable."""
    digest = hashlib.sha256()
    for v in layout.variables:
        digest.update(np.ascontiguousarray(keras.ops.convert_to_numpy(v)).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------
# backend plumbing
# ---------------------------------------------------------------------

def _tf() -> Any:
    """Import TensorFlow lazily.

    TF-DEPENDENCY: the per-sample score needs ``tf.GradientTape``. Everything else is
    ``keras.ops``. If a non-TensorFlow backend is ever supported, this is the single
    seam to turn into a backend adapter.

    Raises:
        RuntimeError: If the Keras backend is not TensorFlow or the call is traced.
    """
    if keras.backend.backend() != 'tensorflow':
        raise RuntimeError(
            f"the Fisher analyzer needs the TensorFlow Keras backend for its per-sample "
            f"gradient tape; the active backend is {keras.backend.backend()!r}")
    import tensorflow as tf
    if not tf.executing_eagerly():
        raise RuntimeError(
            "the Fisher pass must run eagerly: it needs concrete per-sample scores, so it "
            "cannot be called from inside a tf.function or a jit-compiled path")
    return tf


def _n_rows(x: Any) -> int:
    """Number of rows of an array or dict of arrays."""
    if isinstance(x, dict):
        return len(next(iter(x.values())))
    return len(x)


def _take(x: Any, idx: Sequence[int]) -> Any:
    """Rows ``idx`` of an array or of every array of a dict."""
    idx = np.asarray(idx, dtype=np.int64)
    if isinstance(x, dict):
        return {k: np.asarray(v)[idx] for k, v in x.items()}
    return np.asarray(x)[idx]


def _log_softmax_np(z: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax in float64, via the logsumexp shift."""
    z = np.asarray(z, dtype=np.float64)
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _sample_categorical(log_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF draw of one index per row from row-wise log-probabilities."""
    probs = np.exp(log_probs)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random(probs.shape[0])[:, None]
    return np.minimum((u > cdf).sum(axis=-1), probs.shape[-1] - 1).astype(np.int64)


def infer_output_kind(model: keras.Model, x: Any, cfg: FisherConfig,
                      signal: Optional[FisherSignal] = None) -> str:
    """Decide which likelihood a model's output defines.

    Conservative on purpose: anything ambiguous raises and names the override, instead
    of guessing and silently measuring a different quantity.

    Args:
        model: The model.
        x: Input pool (array or dict); its first rows are used as a probe.
        cfg: Fisher configuration.
        signal: Caller hooks.

    Returns:
        One of the ``KIND_*`` constants.

    Raises:
        ValueError: If the output is ambiguous or unsupported.
    """
    if signal is not None and signal.output_logprob_fn is not None:
        if signal.sampler is None:
            raise ValueError("FisherSignal.output_logprob_fn needs FisherSignal.sampler: the "
                             "analyzer cannot sample a response from a likelihood it cannot see")
        return KIND_CUSTOM
    raw = model(_take(x, list(range(min(8, _n_rows(x))))), training=False)
    if isinstance(raw, (list, tuple, dict)):
        raise ValueError("the model has multiple outputs; supply FisherSignal.output_logprob_fn "
                         "(and a sampler) naming the joint log-likelihood instead of scoring output 0 only")
    probe = np.asarray(keras.ops.convert_to_numpy(raw))
    int_prompts = (not isinstance(x, dict) and np.asarray(x).ndim == 2
                   and np.issubdtype(np.asarray(x).dtype, np.integer))
    if cfg.force_sequence is True:
        if probe.ndim != 3 or not int_prompts:
            raise ValueError("force_sequence=True needs integer (B, L) prompts and (B, L, V) outputs")
        return KIND_SEQUENCE
    if probe.ndim == 3:
        if cfg.force_sequence is None and int_prompts:
            return KIND_SEQUENCE
        raise ValueError(
            f"a rank-3 output {probe.shape} is ambiguous for this input; set "
            f"fisher_force_sequence=True if the model is an autoregressive token model")
    if probe.ndim != 2:
        raise ValueError(f"unsupported output rank {probe.ndim}; expected (B, K) or (B, L, V)")
    if cfg.regression_variance is not None:
        return KIND_GAUSSIAN
    if cfg.output_activation is not None:
        activation = cfg.output_activation
    else:
        activation, reason = classify_output_activation(probe)
        logger.info(f"Fisher: inferred output activation '{activation}' ({reason}).")
    return {
        OUTPUT_ACTIVATION_LOGITS: KIND_CATEGORICAL_LOGITS,
        OUTPUT_ACTIVATION_SOFTMAX: KIND_CATEGORICAL_PROBS,
        OUTPUT_ACTIVATION_SIGMOID: KIND_BERNOULLI,
    }[activation]


def fit_batch_size(requested: int, per_sample_bytes: int, memory_limit_mb: Optional[int]) -> int:
    """Size the sampling microbatch so its logits fit the memory budget.

    Args:
        requested: Requested batch size.
        per_sample_bytes: Estimated bytes one sample holds while sampling.
        memory_limit_mb: Budget; ``None`` means explicitly unbounded.

    Returns:
        A batch size in ``[1, requested]``; logs when it had to shrink.
    """
    if memory_limit_mb is None or per_sample_bytes <= 0:
        return requested
    fits = max(1, int(memory_limit_mb * 1024 * 1024 // per_sample_bytes))
    if fits < requested:
        logger.warning(f"Fisher: sampling batch reduced {requested} -> {fits} to fit "
                       f"memory_limit_mb={memory_limit_mb} ({per_sample_bytes} bytes/sample)")
        return fits
    return requested


# ---------------------------------------------------------------------
# scorer: the likelihood of one model
# ---------------------------------------------------------------------

class _Scorer:
    """Sampling and scoring for one model under one output kind.

    Holds no results; it is the model-specific part of the pass (how to draw y from
    pi_theta, and how to differentiate log pi_theta(y|x)).
    """

    def __init__(self, model: keras.Model, kind: str, cfg: FisherConfig,
                 signal: Optional[FisherSignal], layout: Optional[ParamLayout]) -> None:
        self.model = model
        self.kind = kind
        self.cfg = cfg
        self.signal = signal
        self.layout = layout
        if kind not in (KIND_CATEGORICAL_LOGITS, KIND_CATEGORICAL_PROBS, KIND_SEQUENCE) \
                and cfg.sampling_temperature != 1.0:
            raise ValueError(f"sampling_temperature != 1 is only supported for categorical and "
                             f"sequence outputs, not {kind!r}")
        if kind == KIND_GAUSSIAN and cfg.regression_variance is None:
            raise ValueError("a Gaussian likelihood needs fisher_regression_variance; the analyzer "
                             "refuses to guess sigma^2")

    # -- sampling ------------------------------------------------------

    def sample(self, x_batch: Any, rng: np.random.Generator) -> List[np.ndarray]:
        """Draw one response per row from the model's own distribution.

        Args:
            x_batch: A batch of inputs.
            rng: The pass-level generator.

        Returns:
            One response array per row (no batch axis).
        """
        if self.signal is not None and self.signal.sampler is not None:
            return [np.asarray(y) for y in self.signal.sampler(self.model, x_batch, self.cfg, rng)]
        T = self.cfg.sampling_temperature
        if self.kind == KIND_SEQUENCE:
            return self._sample_sequences(np.asarray(x_batch), rng)
        out = np.asarray(keras.ops.convert_to_numpy(self.model(x_batch, training=False)), dtype=np.float64)
        if self.kind == KIND_CATEGORICAL_LOGITS:
            y = _sample_categorical(_log_softmax_np(out / T), rng)
        elif self.kind == KIND_CATEGORICAL_PROBS:
            with np.errstate(divide='ignore'):
                y = _sample_categorical(_log_softmax_np(np.log(out) / T), rng)
        elif self.kind == KIND_BERNOULLI:
            y = (rng.random(out.shape) < out).astype(np.float32)
        elif self.kind == KIND_GAUSSIAN:
            # y ~ N(f(x), sigma^2). Sampling the MEAN instead would make every score
            # exactly zero: the Fisher is about the spread of y around f(x).
            y = (out + math.sqrt(self.cfg.regression_variance) * rng.standard_normal(out.shape)).astype(np.float32)
        else:
            raise ValueError(f"no built-in sampler for output kind {self.kind!r}")
        return [y[i] for i in range(y.shape[0])]

    def _sample_sequences(self, prompts: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
        """Autoregressive decode at temperature T, no top-k / top-p truncation.

        Truncating the tail would remove low-probability regimes from the sample and
        bias the Fisher (paper B.1). Full recompute per step: correct for any causal
        model that accepts variable length, O(T^2) in forward cost.
        """
        eos = None if self.signal is None else self.signal.eos_token_id
        tokens = prompts.astype(np.int64)
        B = tokens.shape[0]
        alive = np.ones(B, dtype=bool)
        lengths = np.zeros(B, dtype=np.int64)
        columns: List[np.ndarray] = []
        for _ in range(self.cfg.max_tokens):
            logits = keras.ops.convert_to_numpy(self.model(tokens, training=False))[:, -1, :]
            nxt = _sample_categorical(_log_softmax_np(np.asarray(logits, np.float64) / self.cfg.sampling_temperature), rng)
            if eos is not None:
                nxt = np.where(alive, nxt, eos)
            columns.append(nxt)
            lengths += alive
            if eos is not None:
                alive = alive & (nxt != eos)
            tokens = np.concatenate([tokens, nxt[:, None]], axis=1)
            if not alive.any():
                break
        grid = np.stack(columns, axis=1)
        # A response ends at (and includes) its stopping token; a truncated one is kept
        # whole, with no synthetic EOS.
        return [grid[i, :lengths[i]] for i in range(B)]

    # -- scoring -------------------------------------------------------

    def _terms(self, out: Any, x_i: Any, y: np.ndarray) -> Any:
        """Per-term log-probabilities, shape (T,), differentiable in the parameters."""
        ops = keras.ops
        if self.kind == KIND_CATEGORICAL_LOGITS:
            logp = ops.log_softmax(out, axis=-1)
            return ops.sum(logp * ops.one_hot(np.asarray(y, np.int64).reshape(1), out.shape[-1]), axis=-1)
        if self.kind == KIND_CATEGORICAL_PROBS:
            return ops.sum(ops.log(out) * ops.one_hot(np.asarray(y, np.int64).reshape(1), out.shape[-1]), axis=-1)
        if self.kind == KIND_BERNOULLI:
            yy = np.asarray(y, np.float32).reshape(1, -1)
            return ops.sum(yy * ops.log(out) + (1.0 - yy) * ops.log(1.0 - out), axis=-1)
        if self.kind == KIND_GAUSSIAN:
            # The y-only normalizer has zero gradient and is dropped; the score is
            # -(y - f)^T grad f / sigma^2.
            yy = np.asarray(y, np.float32).reshape(1, -1)
            return -0.5 * ops.sum(ops.square(yy - out), axis=-1) / self.cfg.regression_variance
        raise ValueError(f"no built-in likelihood for output kind {self.kind!r}")

    def _sequence_terms(self, out: Any, row: np.ndarray, prompt_len: int) -> Any:
        """Token log-probabilities of a teacher-forced row, shape (T,).

        Args:
            out: Logits of shape (1, L, V) for the full row.
            row: The full token row, prompt followed by response.
            prompt_len: Length of the prompt prefix.
        """
        ops = keras.ops
        total_len = len(row)
        start = 0 if self.cfg.score_scope == SCORE_SCOPE_RESPONSE_PLUS_PROMPT else prompt_len - 1
        # Position j predicts token j + 1.
        logp = ops.log_softmax(out[0, start:total_len - 1, :], axis=-1)
        return ops.sum(logp * ops.one_hot(row[start + 1:total_len], out.shape[-1]), axis=-1)

    def _forward_terms(self, x_i: Any, y: np.ndarray) -> Any:
        """Forward the model and return its per-term log-probabilities (differentiable)."""
        if self.kind == KIND_SEQUENCE:
            prompt = np.asarray(x_i)[0].astype(np.int64)
            row = np.concatenate([prompt, np.asarray(y, np.int64)])
            return self._sequence_terms(self.model(row[None, :], training=False), row, len(prompt))
        return self._terms(self.model(x_i, training=False), x_i, y)

    def log_prob_terms(self, x_i: Any, y: np.ndarray) -> np.ndarray:
        """Per-term log-probabilities as numpy, with NO gradient (teacher / student values)."""
        if self.kind == KIND_CUSTOM:
            value = self.signal.output_logprob_fn(self.model, x_i, y)
            return np.asarray(keras.ops.convert_to_numpy(value), np.float64).reshape(-1)
        terms = self._forward_terms(x_i, y)
        return np.asarray(keras.ops.convert_to_numpy(terms), np.float64).reshape(-1)

    def score(self, x_i: Any, y: np.ndarray, dtype: Any) -> Tuple[float, np.ndarray, np.ndarray]:
        """One forward and ONE backward pass: the score vector of the sample.

        Args:
            x_i: One input with a leading batch axis of 1.
            y: One response.
            dtype: Accumulator dtype of the returned score.

        Returns:
            ``(log_prob, per_term_log_probs, h)`` with ``h = grad_theta log pi(y|x)`` in
            the canonical flat order.
        """
        tf = _tf()
        with tf.GradientTape() as tape:
            if self.kind == KIND_CUSTOM:
                terms = keras.ops.reshape(self.signal.output_logprob_fn(self.model, x_i, y), (-1,))
            else:
                terms = self._forward_terms(x_i, y)
            logp = keras.ops.sum(terms)
        grads = tape.gradient(logp, self.layout.variables,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
        h = np.empty(self.layout.total, dtype=dtype)
        for g, (start, stop) in zip(grads, self.layout.slices):
            if hasattr(g, 'indices'):  # IndexedSlices from an embedding lookup
                g = tf.convert_to_tensor(g)
            h[start:stop] = np.asarray(g.numpy(), dtype=dtype).reshape(-1)
        return (float(logp.numpy()),
                np.asarray(terms.numpy(), np.float64).reshape(-1), h)


# ---------------------------------------------------------------------
# selection helpers (recall, retention, masks)
# ---------------------------------------------------------------------

def _top_k_alive(values: np.ndarray, alive: np.ndarray, k: int) -> np.ndarray:
    """Canonical indices of the k largest ``values`` among alive coordinates.

    Ties break by canonical coordinate order (a stable sort on the negated values).
    """
    idx = np.flatnonzero(alive)
    order = np.argsort(-values[idx], kind='stable')
    return idx[order[:k]]


def selection_size(q: float, n_alive: int) -> int:
    """K_q = floor(q * P_alive / 100); 0 when the set would be empty."""
    return int(math.floor(q * n_alive / 100.0 + 1e-9))


def gradient_recall(fisher: np.ndarray, gradient: np.ndarray, percentages: Sequence[float],
                    zero_threshold: float) -> Dict[float, float]:
    """Gradient-support recall (paper Eq. 7).

    Recall(q) = 100 * |S_F(q) intersect S_g(q)| / K_q, with S_F the K_q largest-F
    coordinates and S_g the K_q largest-|g| ones, both among Fisher-alive coordinates.
    A uniformly random subset scores q.

    Args:
        fisher: F-hat, shape (P,).
        gradient: g-hat (sign intact), shape (P,).
        percentages: The q grid.
        zero_threshold: Coordinates with F-hat at or below it are Fisher-dead.

    Returns:
        ``q -> Recall(q)``; NaN (never 0.0) where K_q < 1.
    """
    alive = fisher > zero_threshold
    n_alive = int(alive.sum())
    recall: Dict[float, float] = {}
    for q in percentages:
        k = selection_size(q, n_alive)
        if k < 1:
            recall[float(q)] = float('nan')
            continue
        top_f = _top_k_alive(fisher, alive, k)
        top_g = _top_k_alive(np.abs(gradient), alive, k)
        recall[float(q)] = 100.0 * len(np.intersect1d(top_f, top_g)) / k
    return recall


def bound_violation_rate(fisher: np.ndarray, fisher_se: np.ndarray, gradient: np.ndarray,
                         gradient_se: np.ndarray, signal_scale: float, zero_threshold: float,
                         z: float = BOUND_Z) -> float:
    """Fraction of alive coordinates breaking |g_i| <= signal_scale * sqrt(F_i).

    The bound is a statement about population quantities; both sides are estimated, so
    a coordinate is a violation only when it breaks the envelope by more than the noise:
    ``|g| - z*se_g > scale * sqrt(F + z*se_F)``.

    Args:
        fisher: F-hat.
        fisher_se: Standard error of F-hat.
        gradient: g-hat.
        gradient_se: Standard error of g-hat.
        signal_scale: sqrt(E[a^2]).
        zero_threshold: Fisher-dead cut-off.
        z: Width of the tolerance band in standard errors.

    Returns:
        The violation fraction in [0, 1], or NaN when no coordinate is alive.
    """
    alive = fisher > zero_threshold
    if not alive.any():
        return float('nan')
    upper = signal_scale * np.sqrt(fisher[alive] + z * np.nan_to_num(fisher_se[alive]))
    lower = np.abs(gradient[alive]) - z * np.nan_to_num(gradient_se[alive])
    return float(np.mean(lower > upper))


def fisher_mask_export(output: FisherOutput, q: float,
                       zero_threshold: float = 1e-12) -> Dict[str, np.ndarray]:
    """Boolean masks selecting the top-K_q Fisher coordinates, per variable.

    This is the artifact the paper's masked-training experiments need. The producer
    lives here; the consumer (a masked fine-tune loop) is ``src/train/`` work.

    Args:
        output: A Fisher output.
        q: Selection percentage in (0, 100].
        zero_threshold: Fisher-dead cut-off; dead coordinates are never selected.

    Returns:
        ``variable_path -> bool array`` of the variable's shape.

    Raises:
        ValueError: If K_q would be empty.
    """
    alive = output.fisher_flat > zero_threshold
    k = selection_size(q, int(alive.sum()))
    if k < 1:
        raise ValueError(f"q={q}% of {int(alive.sum())} alive coordinates selects nothing")
    mask = np.zeros(output.n_parameters, dtype=bool)
    mask[_top_k_alive(output.fisher_flat, alive, k)] = True
    return unflatten(mask, output)


def compute_persistence(reference: FisherOutput, checkpoints: Dict[str, FisherOutput],
                        cfg: FisherConfig) -> List[Dict[str, Any]]:
    """Retention and magnitude diagnostics between a reference policy and later ones.

    Retention_high(q) = 100 |H_0(q) intersect H_k(q)| / K_q,
    Retention_low(q)  = 100 |L_0(q) intersect L_k(q)| / (P - K_q),
    M_k/M_0 = mean F_k over L_0(q) / mean F_0 over L_0(q).

    Random baselines are q for the high set and 100 - q for the low set. Stationarity
    is empirical evidence in a regime, not a theorem: this reports it, never assumes it.
    Coordinates are restricted to those Fisher-alive in the REFERENCE.

    Args:
        reference: The initial policy's output.
        checkpoints: Later policies of the SAME architecture, by name.
        cfg: Supplies the q grid and the dead threshold.

    Returns:
        One record per (checkpoint, q).

    Raises:
        ValueError: If a checkpoint's variable fingerprint differs from the reference.
    """
    alive = reference.fisher_flat > cfg.zero_threshold
    n_alive = int(alive.sum())
    records: List[Dict[str, Any]] = []
    for name, ckpt in checkpoints.items():
        if ckpt.variable_fingerprint != reference.variable_fingerprint:
            raise ValueError(
                f"checkpoint {name!r} has variable fingerprint {ckpt.variable_fingerprint}, the "
                f"reference has {reference.variable_fingerprint}: the coordinates do not align")
        for q in cfg.selection_percentages:
            k = selection_size(q, n_alive)
            rec: Dict[str, Any] = {
                'checkpoint': name, 'q': float(q), 'k': k,
                'random_high': float(q), 'random_low': 100.0 - float(q),
            }
            if k < 1 or k >= n_alive:
                rec.update(retention_high=float('nan'), retention_low=float('nan'), mass_ratio=float('nan'))
            else:
                high0 = _top_k_alive(reference.fisher_flat, alive, k)
                highk = _top_k_alive(ckpt.fisher_flat, alive, k)
                low0 = np.setdiff1d(np.flatnonzero(alive), high0)
                lowk = np.setdiff1d(np.flatnonzero(alive), highk)
                rec['retention_high'] = 100.0 * len(np.intersect1d(high0, highk)) / k
                rec['retention_low'] = 100.0 * len(np.intersect1d(low0, lowk)) / (n_alive - k)
                m0 = float(reference.fisher_flat[low0].mean())
                rec['mass_ratio'] = float(ckpt.fisher_flat[low0].mean() / m0) if m0 > 0 else float('nan')
            records.append(rec)
    return records


# ---------------------------------------------------------------------
# Fisher arm
# ---------------------------------------------------------------------

def _summary_metrics(fisher: np.ndarray, n_used: int, skipped: int, cfg: FisherConfig,
                     kind: str, fingerprint: str) -> Dict[str, Any]:
    """Summary scalars of one Fisher vector, keyed by ``FisherMetricNames``."""
    alive = fisher > cfg.zero_threshold
    return {
        FM.TOTAL_FISHER: float(fisher.sum(dtype=np.float64)),
        FM.MEAN_FISHER: float(fisher.mean(dtype=np.float64)),
        FM.STD_FISHER: float(fisher.std(dtype=np.float64)),
        FM.N_PARAMETERS: int(fisher.shape[0]),
        FM.N_ALIVE_PARAMETERS: int(alive.sum()),
        FM.N_FISHER_DEAD: int((~alive).sum()),
        FM.N_SAMPLES: int(n_used),
        FM.SKIPPED_SAMPLES: int(skipped),
        FM.OUTPUT_KIND: kind,
        FM.ESTIMATOR: ESTIMATOR_SEQUENCE,
        FM.VARIABLE_FINGERPRINT: fingerprint,
    }


def sampling_bytes_per_sample(model: keras.Model, kind: str, x: Any, cfg: FisherConfig) -> int:
    """Lower bound on the bytes one sample holds while it is being sampled.

    The logits dominate: float32 model output plus the float64 copy the sampler makes.
    Activations inside the model are not counted, so this is a floor, not a cap.
    """
    row = _take(x, [0])
    out = keras.ops.convert_to_numpy(model(row, training=False)) if kind != KIND_CUSTOM else np.zeros(1)
    elems = int(np.prod(out.shape[1:])) if out.ndim > 1 else 1
    if kind == KIND_SEQUENCE:
        elems = (np.asarray(row).shape[1] + cfg.max_tokens) * out.shape[-1]
    return elems * (out.dtype.itemsize + 8)


def compute_diagonal_fisher(model: keras.Model, x: Any, cfg: FisherConfig,
                            rng: np.random.Generator,
                            indices: Optional[Sequence[int]] = None,
                            signal: Optional[FisherSignal] = None,
                            layout: Optional[ParamLayout] = None,
                            kind: Optional[str] = None) -> FisherOutput:
    """Diagonal model Fisher of ``model`` on the inputs ``x[indices]``.

    F-hat_i = (1/N) sum_n h_i(x_n, y_n)^2 with y_n ~ pi_theta(.|x_n) drawn on-policy
    and h = grad_theta log pi_theta(y_n|x_n) taken in ONE backward pass. h is the SUM of
    token scores (no length normalization) and is squared per trajectory before
    averaging. A sample with any non-finite score coordinate is dropped whole, counted
    in ``skipped_samples`` and logged; it is never clipped. Welford accumulation
    gives the standard error of F-hat.

    Args:
        model: The model; called with ``training=False``, never modified.
        x: Input pool (array or dict of arrays).
        cfg: Configuration.
        rng: The pass-level generator (one per ``analyze()``, F-009).
        indices: Pool rows to use; defaults to the first ``cfg.n_samples``.
        signal: Optional hooks (custom likelihood, sampler, EOS id).
        layout: Precomputed layout (otherwise built from ``model``).
        kind: Output kind (otherwise inferred).

    Returns:
        The output, with the gradient and token arms unset.

    Raises:
        RuntimeError: If no sample produced a finite score, or if the trainable
            variables changed during the pass and ``cfg.assert_no_weight_drift``.
        ValueError: On an unsupported or ambiguous model / configuration.
    """
    cfg.validate()
    _tf()
    started = time.time()
    layout = layout or canonical_layout(model)
    kind = kind or infer_output_kind(model, x, cfg, signal)
    scorer = _Scorer(model, kind, cfg, signal, layout)
    if indices is None:
        indices = list(range(min(cfg.n_samples, _n_rows(x))))
    indices = [int(i) for i in indices]
    dtype = np.dtype(cfg.accumulate_dtype)
    digest_before = _weights_digest(layout) if cfg.assert_no_weight_drift else None

    batch = fit_batch_size(cfg.batch_size, sampling_bytes_per_sample(model, kind, x, cfg),
                           cfg.memory_limit_mb)
    mean = np.zeros(layout.total, dtype=dtype)
    m2 = np.zeros(layout.total, dtype=dtype)
    n_used = skipped = 0
    samples_used: List[Dict[str, Any]] = []
    for b in range(0, len(indices), batch):
        chunk = indices[b:b + batch]
        x_batch = _take(x, chunk)
        responses = scorer.sample(x_batch, rng)
        for j, idx in enumerate(chunk):
            y = responses[j]
            logp, _, h = scorer.score(_take(x, [idx]), y, dtype)
            if not (np.isfinite(logp) and np.all(np.isfinite(h))):
                skipped += 1
                if skipped <= MAX_LOGGED_SKIPS:
                    logger.warning(f"Fisher: dropped pool row {idx}: non-finite score "
                                   f"(log-prob {logp}); dropped, not clipped")
                continue
            h2 = h * h
            n_used += 1
            delta = h2 - mean
            mean += delta / n_used
            m2 += delta * (h2 - mean)
            samples_used.append({'index': idx, 'response': np.asarray(y).tolist()})
    if n_used == 0:
        raise RuntimeError(f"no sample produced a finite score ({skipped} dropped)")
    if digest_before is not None and _weights_digest(layout) != digest_before:
        raise RuntimeError("the trainable variables changed during the Fisher pass; the "
                           "pass must be read-only")
    se = np.sqrt(m2 / (n_used - 1) / n_used) if n_used > 1 else np.full(layout.total, np.nan, dtype=dtype)
    output = FisherOutput(
        fisher_flat=mean, standard_errors=se,
        param_slices=[(p, s, e) for p, (s, e) in zip(layout.paths, layout.slices)],
        shapes=[(p, s, d) for p, s, d in zip(layout.paths, layout.shapes, layout.dtypes)],
        variable_fingerprint=layout.fingerprint, n_samples=n_used, skipped_samples=skipped,
        output_kind=kind, samples_used=samples_used)
    output.metrics = _summary_metrics(mean, n_used, skipped, cfg, kind, layout.fingerprint)
    output.metrics[FM.COMPUTATION_TIME] = time.time() - started
    return output


# ---------------------------------------------------------------------
# gradient arm
# ---------------------------------------------------------------------

def compute_gradients(model: keras.Model, x: Any, targets: Optional[Any], cfg: FisherConfig,
                      rng: np.random.Generator, signal: FisherSignal,
                      indices: Sequence[int], layout: Optional[ParamLayout] = None,
                      kind: Optional[str] = None) -> GradientArm:
    """Objective-signal gradient g-hat and the signal scale of Prop. 1.

    For each prompt n, G responses y_{n,j} are drawn on-policy (``rl``, ``opd_*``) and
    g-hat is the mean over prompts of (1/G) sum_j a_{n,j} h(x_n, y_{n,j}), sign intact.

    -   ``rl``: a = r - mean_j r (Dr. GRPO, no std normalization). Needs G >= 2.
    -   ``opd_pg``: a = R_T - mean_j R_T, R_T = sum_t [log q_t(y_t) - log p_t(y_t)] with
        the teacher fixed. Needs G >= 2.
    -   ``opd_full``: a = sum_t ( q_t(y_t)/p_t(y_t) - 1 ).
    -   ``sft``: y is the DEMONSTRATION ``targets[n]`` (drawn from nu), and
        g-hat = mean h. The importance weight is already inside the sampling
        distribution: E_pi[(w-1) h] = E_nu[h]. Multiplying by (w-1) here would apply it
        twice. a = w - 1 is used for the scale only, E_pi[a^2] = E_nu[w] - 1 = chi^2.

    The group mean is computed from the same G responses, so g-hat carries a factor
    (G-1)/G and E[a^2] the same, which keeps the envelope valid. Teacher
    probabilities are computed outside the tape and enter only as constants.

    Args:
        model: The student.
        x: Input pool.
        targets: Pool-aligned targets (reward argument; SFT demonstrations).
        cfg: Configuration.
        rng: The pass-level generator.
        signal: Hooks required by ``cfg.signal_type``.
        indices: Gradient-arm pool rows (disjoint from the Fisher rows).
        layout: Precomputed layout.
        kind: Output kind.

    Returns:
        The gradient arm.

    Raises:
        ValueError: If the signal hook the signal type needs is missing.
        RuntimeError: If no prompt produced a finite gradient.
    """
    cfg.validate()
    _tf()
    st = cfg.signal_type
    if signal is None:
        raise ValueError("the gradient arm needs a FisherSignal")
    needs = {SIGNAL_RL: ('reward_fn',), SIGNAL_SFT: ('nu_logprob_fn',),
             SIGNAL_OPD_PG: ('teacher_model',), SIGNAL_OPD_FULL: ('teacher_model',)}.get(st, ())
    for attr in needs:
        if getattr(signal, attr) is None:
            raise ValueError(f"signal_type={st!r} needs FisherSignal.{attr}")
    if st in (SIGNAL_SFT,) and targets is None:
        raise ValueError("signal_type='sft' needs targets (the demonstrations y ~ nu)")
    layout = layout or canonical_layout(model)
    kind = kind or infer_output_kind(model, x, cfg, signal)
    scorer = _Scorer(model, kind, cfg, signal, layout)
    teacher = (_Scorer(signal.teacher_model, kind, cfg, signal, None)
               if st in (SIGNAL_OPD_PG, SIGNAL_OPD_FULL) else None)
    dtype = np.dtype(cfg.accumulate_dtype)
    P = layout.total
    mean = np.zeros(P, dtype=dtype)
    m2 = np.zeros(P, dtype=dtype)
    n_prompts = skipped = 0
    advantages: List[float] = []
    weights: List[float] = []
    G = 1 if st == SIGNAL_SFT else cfg.n_responses

    for idx in indices:
        idx = int(idx)
        x_i = _take(x, [idx])
        target = None if targets is None else np.asarray(targets)[idx]
        if st == SIGNAL_SFT:
            ys = [np.asarray(target)]
        else:
            ys = scorer.sample(_take(x, [idx] * G), rng)
        s_h = np.zeros(P, dtype=dtype)       # sum_j h_j
        s_ah = np.zeros(P, dtype=dtype)      # sum_j a_j h_j (when a_j is known at once)
        s_rh = np.zeros(P, dtype=dtype)      # sum_j r_j h_j (group-centered signals)
        rewards: List[float] = []
        a_known: List[float] = []
        w_known: List[float] = []
        valid = 0
        for y in ys:
            logp, terms, h = scorer.score(x_i, y, dtype)
            if not (np.isfinite(logp) and np.all(np.isfinite(h))):
                skipped += 1
                continue
            if st == SIGNAL_RL:
                r = float(signal.reward_fn(x_i, y, target))
                if not np.isfinite(r):
                    skipped += 1
                    continue
                rewards.append(r)
                s_rh += r * h
            elif st == SIGNAL_OPD_PG:
                log_q = teacher.log_prob_terms(x_i, y)
                r = float(np.sum(log_q - terms))
                if not np.isfinite(r):
                    skipped += 1
                    continue
                rewards.append(r)
                s_rh += r * h
            elif st == SIGNAL_OPD_FULL:
                log_q = teacher.log_prob_terms(x_i, y)
                a = float(np.sum(np.exp(log_q - terms) - 1.0))
                if not np.isfinite(a):
                    skipped += 1
                    continue
                a_known.append(a)
                s_ah += a * h
            else:  # SIGNAL_SFT
                log_nu = float(signal.nu_logprob_fn(x_i, y))
                w = math.exp(log_nu - logp) if np.isfinite(log_nu) else float('nan')
                if not np.isfinite(w):
                    skipped += 1
                    continue
                w_known.append(w)
                s_ah += h
            s_h += h
            valid += 1
        if st in GROUP_CENTERED_SIGNALS:
            if valid < 2:
                skipped += valid  # a lone response has a == 0: no information
                continue
            r_bar = float(np.mean(rewards))
            u = (s_rh - r_bar * s_h) / valid
            advantages.extend(r - r_bar for r in rewards)
        elif valid < 1:
            continue
        else:
            u = s_ah / valid
            advantages.extend(a_known)
            weights.extend(w_known)
        n_prompts += 1
        delta = u - mean
        mean += delta / n_prompts
        m2 += delta * (u - mean)

    if n_prompts == 0:
        raise RuntimeError("no prompt produced a finite gradient estimate")
    se = np.sqrt(m2 / (n_prompts - 1) / n_prompts) if n_prompts > 1 else np.full(P, np.nan, dtype=dtype)
    chi2 = None
    if st == SIGNAL_SFT:
        chi2 = max(float(np.mean(weights)) - 1.0, 0.0)
        scale = math.sqrt(chi2)
        if chi2 > cfg.sft_chi2_cap:
            logger.warning(
                f"Fisher SFT: chi^2(nu || pi) = {chi2:.3g} exceeds fisher_sft_chi2_cap="
                f"{cfg.sft_chi2_cap:g}; the demonstrations put mass where the model has almost "
                f"none. Widen the model's support or raise the temperature before trusting the "
                f"weights.")
    else:
        scale = math.sqrt(float(np.mean(np.square(advantages))))
    return GradientArm(gradient_flat=mean, gradient_se_flat=se, signal_scale=float(scale),
                       n_prompts=n_prompts, skipped=skipped, sft_chi2=chi2)


def attach_gradient_arm(output: FisherOutput, arm: GradientArm, cfg: FisherConfig) -> FisherOutput:
    """Join a gradient arm to a Fisher output and derive recall and the bound check.

    Args:
        output: The Fisher output.
        arm: The gradient arm measured on the same model.
        cfg: Configuration.

    Returns:
        A new ``FisherOutput`` with the gradient fields and metrics filled.
    """
    recall = gradient_recall(output.fisher_flat, arm.gradient_flat, cfg.selection_percentages,
                             cfg.zero_threshold)
    violation = bound_violation_rate(output.fisher_flat, output.standard_errors,
                                     arm.gradient_flat, arm.gradient_se_flat, arm.signal_scale,
                                     cfg.zero_threshold)
    metrics = dict(output.metrics)
    metrics[FM.SIGNAL_SCALE] = arm.signal_scale
    metrics[FM.VIOLATION_RATE] = violation
    metrics[FM.GRADIENT_RECALL] = recall
    if arm.sft_chi2 is not None:
        metrics[FM.SFT_CHI2] = arm.sft_chi2
        metrics[FM.SFT_CHI2_EXCEEDS_CAP] = bool(arm.sft_chi2 > cfg.sft_chi2_cap)
    return dataclasses.replace(
        output, gradient_flat=arm.gradient_flat, gradient_se_flat=arm.gradient_se_flat,
        signal_scale=arm.signal_scale, gradient_recall=recall, violation_rate=violation,
        metrics=metrics)


# ---------------------------------------------------------------------
# token-level arm
# ---------------------------------------------------------------------

def compute_token_fisher(model: keras.Model, x: Any, output: FisherOutput, cfg: FisherConfig,
                         rng: np.random.Generator, layout: Optional[ParamLayout] = None
                         ) -> FisherOutput:
    """Mean token Fisher and the Real-vs-Uniform comparison (paper Eq. 9, Eq. 11, B.3).

    For sampled positions of the responses already drawn by the Fisher arm, with the
    prefix s and logits z = f_theta(s) over V tokens,

        f_i(s) = Var_{a~p}[j_{i,a}] = 1/2 E_{a,b~iid p}[(j_{i,a} - j_{i,b})^2].

    One pair (a, b) is estimated by the squared gradient of (z_a - z_b)/sqrt(2): its
    expectation is exactly f_i. (Backpropagating (z_a - z_b)/2 instead estimates
    f_i / 2.) Pairs are shared by all parameter groups, so there is one backward pass
    per pair. A pair with a == b contributes 0 and STAYS in the count. The Uniform
    reference draws a, b uniformly over the FULL vocabulary, special tokens included.

    This is the MEAN-TOKEN estimator: it averages token Fisher over sampled prefixes
    and is not the unnormalized sequence Fisher of :func:`compute_diagonal_fisher`.

    Args:
        model: The sequence model.
        x: Input pool of integer prompts.
        output: The Fisher output whose ``samples_used`` hold the responses.
        cfg: Configuration.
        rng: The pass-level generator.
        layout: Precomputed layout.

    Returns:
        A new ``FisherOutput`` with the token-level fields filled.

    Raises:
        ValueError: If the output is not from a sequence model.
    """
    if output.output_kind != KIND_SEQUENCE:
        raise ValueError("the token-level arm needs a sequence model's Fisher output")
    tf = _tf()
    layout = layout or canonical_layout(model)
    if layout.fingerprint != output.variable_fingerprint:
        raise ValueError("the model's variable fingerprint differs from the Fisher output's")
    dtype = np.dtype(cfg.accumulate_dtype)
    names, spans = layer_groups(layout)
    n_groups = len(names)
    K = cfg.n_positions_per_response
    sum_real = np.zeros(layout.total, dtype=dtype)
    sum_unif = np.zeros(layout.total, dtype=dtype)
    slot_real = np.zeros((K, n_groups))
    slot_unif = np.zeros((K, n_groups))
    slot_count = np.zeros(K)
    n_pairs = 0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def group_means(g2: np.ndarray) -> np.ndarray:
        return np.array([np.mean(np.concatenate([g2[s:e] for s, e in sp])) for sp in spans])

    for sample in output.samples_used:
        prompt = np.asarray(_take(x, [sample['index']]))[0].astype(np.int64)
        response = np.asarray(sample['response'], dtype=np.int64)
        T = len(response)
        if T == 0:
            continue
        positions = np.sort(rng.choice(T, size=min(K, T), replace=False))
        for slot, t in enumerate(positions):
            prefix = np.concatenate([prompt, response[:t]])[None, :]
            with tf.GradientTape(persistent=True) as tape:
                z = model(prefix, training=False)[0, -1, :]
                V = int(z.shape[0])
                p = np.exp(_log_softmax_np(keras.ops.convert_to_numpy(z)))
                p = p / p.sum()
                pairs = {
                    'real': rng.choice(V, size=(cfg.n_token_pairs_per_position, 2), p=p),
                    'unif': rng.integers(0, V, size=(cfg.n_token_pairs_per_position, 2)),
                }
                scalars = {c: [(z[int(a)] - z[int(b)]) * inv_sqrt2 for a, b in ab]
                           for c, ab in pairs.items()}
            try:
                for cond, target_sum, slot_acc in (('real', sum_real, slot_real),
                                                   ('unif', sum_unif, slot_unif)):
                    for (a, b), scalar in zip(pairs[cond], scalars[cond]):
                        if a != b:
                            grads = tape.gradient(scalar, layout.variables,
                                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
                            g = np.empty(layout.total, dtype=dtype)
                            for gr, (s, e) in zip(grads, layout.slices):
                                if hasattr(gr, 'indices'):
                                    gr = tf.convert_to_tensor(gr)
                                g[s:e] = np.asarray(gr.numpy(), dtype=dtype).reshape(-1)
                            g2 = g * g
                            target_sum += g2
                            slot_acc[slot] += group_means(g2)
                        # a == b: zero contribution, and the pair stays in the count.
                    if cond == 'real':
                        n_pairs += cfg.n_token_pairs_per_position
                        slot_count[slot] += cfg.n_token_pairs_per_position
            finally:
                del tape
    if n_pairs == 0:
        raise RuntimeError("the token-level arm measured no pairs")
    token_mean = sum_real / n_pairs
    uniform = sum_unif / n_pairs
    denom = np.maximum(slot_count, 1)[:, None]
    by_pos, unif_by_pos = slot_real / denom, slot_unif / denom
    real_g, unif_g = group_means(token_mean), group_means(uniform)
    ratio = {n: (float(r / u) if u > 0 else float('nan')) for n, r, u in zip(names, real_g, unif_g)}
    total_unif = float(uniform.mean(dtype=np.float64))
    metrics = dict(output.metrics)
    metrics[FM.TOKEN_FISHER_MEAN] = float(token_mean.mean(dtype=np.float64))
    metrics[FM.UNIFORM_REFERENCE] = total_unif
    metrics[FM.REAL_UNIFORM_RATIO] = (float(token_mean.mean(dtype=np.float64)) / total_unif
                                      if total_unif > 0 else float('nan'))
    return dataclasses.replace(
        output, token_fisher_mean=token_mean, uniform_reference=uniform, group_names=names,
        token_fisher_by_position=by_pos, uniform_by_position=unif_by_pos,
        real_uniform_ratio=ratio, metrics=metrics)
