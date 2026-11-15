"""The forecasting model families of ``python -m train``: adapters, tables, refusals and recipes.

Seven families train the ``time_series_forecasting`` task: ``nbeats``, ``prism`` (point head, or a quantile head when
``model_args.quantile_levels`` is given), ``tirex`` (quantile), ``xlstm`` (point or quantile like prism), ``ets``,
``timestral`` and ``mdn``. Two more are REFUSED by name with the measured technical reason: ``deepar`` and
``adaptive_ema`` (:data:`UNSUPPORTED`). Their library factories do not take the arguments the API calls a factory
with, so the adapters below close that gap and nothing else; the models are not edited. Importing this module imports
no model code and no keras (the adapters import inside the call), so ``python -m train help`` allocates nothing.
The specs are registered with the task by ``recipes.register_builtin_models``, which calls
:func:`builtin_forecast_models` once.

Interface contract
------------------
``create_nbeats / create_prism / create_tirex / create_xlstm / create_ets / create_mdn``
    ``f(*, variant, num_classes=None, input_shape, horizon, **model_args)``
    :param variant: a key of the family's variants table (:data:`NBEATS_VARIANTS` and :data:`MDN_VARIANTS` are
        written here; the others are the library's own, ``ETS_VARIANTS`` a tuple).
    :param num_classes: must be ``None``: a forecaster has no classes.
    :param input_shape: ``(lookback, channels)`` of positive ints.
    :param horizon: the points forecast, a positive int (the task passes ``data.horizon``).
    :param model_args: the keys :data:`ALLOWED` offers; ``quantile_levels`` is read by the adapter (prism and xlstm
        take a quantile head when it is given and a point head otherwise; tirex always has one).
    :return: a model BUILT at ``(None, lookback, channels)`` whose output :mod:`train.common.forecast_run` decodes:
        a tuple for nbeats (forecast, residual), ``(B, H, C)`` or ``(B, H, C, Q)`` for prism, ``(B, H, Q)`` for
        tirex and a quantile xlstm, ``(B, H, 1)`` for ets, packed mixture parameters for mdn (a ``Flatten`` stub in
        front of ``MDNModel``).
    :raises ValueError: for a non-``None`` ``num_classes``, a bad shape or horizon, a key no table knows, a key
        :data:`REFUSED`, :data:`SUPPLIED` or :data:`FORCED` hold, a channel count the family cannot take (tirex, ets
        and mdn take one) or an unknown variant (:func:`train.api.recipes.build_model` turns it into a
        ``ConfigError``).

``create_timestral(*, variant, num_classes=None, input_shape, horizon, batch_size, **model_args)``
    The same, built at a STATIC batch ``(batch_size, lookback, channels)`` (the model packs rows by the batch size;
    the data drops the remainder): the output is ``(B, H, C, 21)`` quantiles. A horizon above ``patch_size`` is
    refused by name (:func:`geometry_problems`).

``geometry_problems(family, channels, horizon, model_args)``
    :return: the refusals (a list of lines, ``[]`` for none) of a channel count above the family's, and of a timestral
        horizon above its patch size. Pure and TensorFlow-free: the adapters and the task (before the claim) share it.

``check_supported_family(model)``
    :param model: the ``model=`` text, ``family`` or ``family/variant``.
    :raises ConfigError: for ``deepar`` and ``adaptive_ema``, naming the family and its reason; it imports no variants
        table and no model code.

``jit_problem(family, jit_compile)``
    :return: the refusal text of ``jit_compile=true`` for a family whose gradient XLA cannot compile (ets), else ``None``.

``flat_target(family, model_args)``
    :return: whether the family's loss wants the target as ``(batch, horizon)`` (tirex, and xlstm with a quantile head);
        pure. :func:`geometry_problems` refuses the quantile xlstm for more than one channel: its output has no channel axis.

``loss_problem(family, model_args, loss_name)``
    :return: the refusal text of ``loss_args.name`` for a head whose loss is fixed (a quantile head, timestral, mdn), else
        ``None``; pure, so the task refuses before the claim. ``train.common.forecast_run`` refuses the same heads when it
        builds the loss (a test holds the two equal).

``problems_of(family, model_args)`` / ``resolve_forecast_args(context, optim_args, model_args)``
    :return: one refusal line per bad key (pure) / the factory keywords, the recipe's then the caller's.
    :raises ConfigValidationError: naming EVERY refused key at once, before the run directory is claimed.
    :raises ConfigError: for an sgd weight decay (no model regularizer is built here).

``factory_signatures(family)`` / ``signature_census(family, signatures=None)``
    The denoising pattern: ``{"unclassified": [...], "stale": [...]}``, the parameters of the callables in
    :data:`FACTORIES` that :data:`ALLOWED`, :data:`REFUSED`, :data:`FORCED` and :data:`SUPPLIED` do not name, and the
    keys those tables name that no callable has. A passing census is both empty.

``builtin_forecast_models()``
    :return: the seven :class:`~train.api.registry.ModelSpec` rows, freshly built (``tasks=("time_series_forecasting",)``).

Recipes
-------
Every family trains AdamW, ``val_mae`` min, 10 epochs, batch 128, lookback 168 and horizon 24, weight decay 0.004, one
warmup epoch, cosine to 1 percent, a GLOBAL clip of 1.0 (never a literal ``clipnorm``), no stacked L2. The numbers are
the legacy trainers' where they agree and the measured ones where they do not (``test_forecast_families`` reads every
legacy value from the legacy modules and names each difference). ets keeps the legacy learning rate (5e-2, no warmup)
and mdn its own (5e-4): one scalar parameter does not move at 1e-4.

Import rule
-----------
Module scope imports the standard library and ``train.api`` only.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import recipes
from .args import AutoContext, OptimArgs, is_count
from .config import unknown_key_message
from .denoising_models import named_parameters
from .errors import ConfigError, ConfigValidationError
from .forecast_args import MIN_LOOKBACK, TASK, WATCH_METRIC
from .registry import ModelSpec

# ---------------------------------------------------------------------

_HERE = "train.api.forecast_models"
_TS = "dl_techniques.models.time_series"

NBEATS, PRISM, TIREX, XLSTM, ETS, TIMESTRAL, MDN = "nbeats", "prism", "tirex", "xlstm", "ets", "timestral", "mdn"

#: ``patch_size`` of ``TimestralModel.from_variant`` (and ``TimestralConfig``); a test holds it equal to the library's.
TIMESTRAL_PATCH = 32

# DECISION plan-2026-11-13T105600-3a8319ef/D-022
# deepar and adaptive_ema are REFUSED by name, with the reasons measured, and are not registered: a refusal
# row is a sentence, not a family. Do NOT train deepar through its teacher-forced `call` (its input holds the horizon,
# so an exported serve function would take the answer as input) and do NOT wrap it for the eager sampling path
# (the legacy `add_loss` wrapper does not reload from .keras); do NOT score adaptive_ema against "naive" (it has no
# horizon axis). The refusal runs BEFORE the variants table or any model module is imported.
UNSUPPORTED: Dict[str, str] = {
    "deepar": (
        "its trained graph is teacher-forced: call takes a window that CONTAINS the horizon (changing the horizon values "
        "moved mu at the horizon positions by up to 0.659), so an exported serve function would take the answer as "
        "input, and its forecast path is an eager Monte-Carlo loop that no export format carries"
    ),
    "adaptive_ema": (
        "it is a regime and slope filter: its outputs (ema, slope, the regime signals) have no horizon axis (their "
        "length follows the input), so a forecast, and the bar 'beats naive', are undefined for it"
    ),
}

# ---------------------------------------------------------------------
# The variants tables the library does not have (written here, like ACCUNET_VARIANTS)
# ---------------------------------------------------------------------

#: ``create_nbeats_model`` keywords by variant. ``base`` is the model the bar was measured on (hidden 256 and
#: the factory's own relu and no dropout, listed here so a change of the library's defaults moves nothing); the
#: legacy trainer's gelu and dropout 0.1 were not measured and are not the default.
NBEATS_VARIANTS: Dict[str, Dict[str, Any]] = {
    "tiny": {"stack_types": ["trend", "seasonality", "generic"], "nb_blocks_per_stack": 1, "hidden_layer_units": 64,
             "activation": "relu", "dropout_rate": 0.0},
    "small": {"stack_types": ["trend", "seasonality", "generic"], "nb_blocks_per_stack": 3, "hidden_layer_units": 128,
              "activation": "relu", "dropout_rate": 0.0},
    "base": {"stack_types": ["trend", "seasonality", "generic"], "nb_blocks_per_stack": 3, "hidden_layer_units": 256,
             "activation": "relu", "dropout_rate": 0.0},
}

#: ``MDNModel`` keywords by variant. ``base`` is the legacy trainer's (``MDNTrainingConfig``: 256-128-64, 12 mixtures,
#: dropout 0.3).
MDN_VARIANTS: Dict[str, Dict[str, Any]] = {
    "tiny": {"hidden_layers": [64, 64], "num_mixtures": 3},
    "small": {"hidden_layers": [256, 256], "num_mixtures": 5},
    "base": {"hidden_layers": [256, 128, 64], "num_mixtures": 12, "dropout_rate": 0.3},
}

# ---------------------------------------------------------------------
# What each family offers, refuses, forces and is given
# ---------------------------------------------------------------------

_DECAY = "the optimizer decays the weights (adamw), a model regularizer would decay twice, and sgd's decay is refused"
_HEAD = "the head follows model_args.quantile_levels: give it for a quantile head, leave it out for a point head"
_NAME = "the model's class is what a record and a reload check, not its name"
_ONE_NORM = (
    "the data layer already standardizes every window by its context statistics, so a model normalization would "
    "normalize twice (D-023)"
)
_TIMESTRAL_FIXED = (
    "it changes what call returns or how it masks (the decode row is held to the released settings, D-009), or only "
    "matters to the multi-block forecaster this task does not run"
)

#: Keys passed through as the caller gave them (they may override a variant's value; the run records the override).
ALLOWED: Dict[str, Tuple[str, ...]] = {
    NBEATS: ("stack_types", "nb_blocks_per_stack", "thetas_dim", "hidden_layer_units", "activation", "dropout_rate",
             "share_weights_in_stack", "kernel_initializer", "use_bias"),
    PRISM: ("hidden_dim", "num_layers", "tree_depth", "overlap_ratio", "num_wavelet_levels", "router_hidden_dim",
            "router_temperature", "dropout_rate", "ffn_expansion", "quantile_levels", "enforce_monotonicity",
            "kernel_initializer"),
    TIREX: ("quantile_levels", "patch_size", "embed_dim", "num_blocks", "num_heads", "lstm_units", "ff_dim", "block_types",
            "dropout_rate", "use_layer_norm", "attention_window_size", "attention_type"),
    XLSTM: ("embed_dim", "num_layers", "mlstm_ratio", "mlstm_num_heads", "mlstm_expansion_factor", "slstm_forget_gate",
            "ffn_type", "ffn_expansion_factor", "normalization_type", "normalization_kwargs", "dropout_rate",
            "quantile_levels", "enforce_monotonicity", "kernel_initializer", "recurrent_initializer", "bias_initializer"),
    ETS: ("seasonal_period", "alpha_init", "beta_init", "gamma_init"),
    TIMESTRAL: ("patch_size", "d_model", "num_heads", "mlp_hidden_dim", "block_schedule", "rope_theta",
                "rope_partial_factor", "norm_eps", "attn_bias", "mlp_bias", "time_attn_gate", "variate_attn_gate",
                "minimum_scale", "use_asinh", "quantile_width_factor"),
    MDN: ("hidden_layers", "num_mixtures", "hidden_activation", "kernel_initializer", "use_batch_norm", "dropout_rate"),
}

#: Keys refused whatever their value: ``{family: {key: why}}``.
REFUSED: Dict[str, Dict[str, str]] = {
    NBEATS: {"kernel_regularizer": _DECAY, "theta_regularizer": _DECAY},
    PRISM: {"use_quantile_head": _HEAD, "num_quantiles": _HEAD, "kernel_regularizer": _DECAY},
    TIREX: {"name": _NAME},
    XLSTM: {"pretrained": "no pretrained weights are shipped", "use_quantile_head": _HEAD, "name": _NAME,
            "kernel_regularizer": _DECAY, "recurrent_regularizer": _DECAY, "bias_regularizer": _DECAY},
    ETS: {},
    TIMESTRAL: {key: _TIMESTRAL_FIXED for key in (
        "quantile_levels", "quantile_knots", "block_size", "context_length", "stabilize_with_global",
        "covariate_enabled", "split_context_horizon_attn", "attend_missing_patches")},
    MDN: {"kernel_regularizer": _DECAY},
}

# DECISION plan-2026-11-13T105600-3a8319ef/D-005
# DECISION plan-2026-11-13T105600-3a8319ef/D-023
# Normalization lives in ONE place, the data layer (each window by its CONTEXT statistics). nbeats, tirex and xlstm each
# have a switch of their own and it is forced OFF here, a user `true` refused by name: two normalizations were the legacy
# xlstm failure ("double normalization collapsed targets"). tirex's own was measured idempotent on standardized input
# (5.5e-5, 3e-5 of the scale), which is the bound if a switch is ever left on, not a reason to leave it. Do NOT drop a row
# to "keep the legacy default", and do NOT add the switch to ALLOWED. prism and mdn have none and need the data layer;
# ets and timestral are equivariant themselves.
#: Keys the adapter sets to one value: ``{family: {key: (value, why)}}``; any other value is refused, the same is a no-op.
FORCED: Dict[str, Dict[str, Tuple[Any, str]]] = {
    NBEATS: {"use_normalization": (False, _ONE_NORM)},
    PRISM: {},
    TIREX: {"use_normalization": (False, _ONE_NORM)},
    XLSTM: {"use_normalization": (False, _ONE_NORM)},
    ETS: {},
    TIMESTRAL: {},
    MDN: {},
}

#: Keywords the adapter takes from the data or the API itself: naming one in ``model_args`` is refused.
SUPPLIED: Dict[str, Dict[str, str]] = {
    NBEATS: {"backcast_length": "it is data.lookback", "forecast_length": "it is data.horizon",
             "input_dim": "it is the number of channels", "output_dim": "it is the number of channels"},
    PRISM: {"variant": "model=prism/<variant> chooses it", "context_len": "it is data.lookback",
            "forecast_len": "it is data.horizon", "num_features": "it is the number of channels"},
    TIREX: {"variant": "model=tirex/<variant> chooses it", "input_length": "it is data.lookback",
            "prediction_length": "it is data.horizon"},
    XLSTM: {"variant": "model=xlstm/<variant> chooses it", "input_length": "it is data.lookback",
            "prediction_length": "it is data.horizon", "num_features": "it is the number of channels"},
    ETS: {"variant": "model=ets/<variant> chooses it", "horizon": "it is data.horizon"},
    TIMESTRAL: {"variant": "model=timestral/<variant> chooses it", "prediction_length": "it is data.horizon"},
    MDN: {"output_dimension": "it is data.horizon"},
}

#: The library callables whose signatures :func:`signature_census` classifies: the factory the adapter calls and the
#: constructor behind it, where the ``**overrides`` of the first land.
FACTORIES: Dict[str, Tuple[str, ...]] = {
    NBEATS: (f"{_TS}.nbeats.nbeats:create_nbeats_model", f"{_TS}.nbeats.nbeats:NBeatsNet"),
    PRISM: (f"{_TS}.prism.model:PRISM.from_variant", f"{_TS}.prism.model:PRISM"),
    TIREX: (f"{_TS}.tirex.model:create_tirex_by_variant", f"{_TS}.tirex.model:TiRexCore"),
    XLSTM: (f"{_TS}.xlstm.forecaster:xLSTMForecaster.from_variant", f"{_TS}.xlstm.forecaster:xLSTMForecaster"),
    ETS: (f"{_TS}.ets.model:create_ets", f"{_TS}.ets.model:ETSModel"),
    TIMESTRAL: (
        f"{_TS}.timestral.model:create_timestral_by_variant", f"{_TS}.timestral.model:TimestralModel.from_variant",
        f"{_TS}.timestral.config:TimestralConfig",
    ),
    MDN: (f"{_TS}.mdn.model:MDNModel",),
}


@dataclass(frozen=True)
class FamilyRow:
    """What the task needs to know about a family besides its tables.

    :param variants: the ``module:attr`` of the variants table (a mapping, or ETS's tuple of names).
    :param max_channels: the most channels the model takes, ``None`` for any; ``why_channels`` is the reason it gives.
    :param static_batch: the model packs rows by the batch size, so it is built at a static batch and the data drops
        the remainder (timestral).
    :param jit_refusal: the reason ``jit_compile=true`` is refused (D-028), ``None`` for a family XLA can compile.
    :param min_lookback: ``{variant: smallest lookback the library builds the variant at}``, MEASURED by building it at
        decreasing lookbacks (a test rebuilds each at the minimum and one
        below). A variant that is not listed builds at :data:`MIN_LOOKBACK`; ``why_lookback`` is the library's own reason.
    :param min_lookback_over: ``{variant: (model_args key, extra)}``: the variant needs a lookback of at least ``model_args[key] +
        extra`` when the key is given (the seasonal ets: a full ``seasonal_period`` plus one step); ``why_over`` is the reason.
    :param min_horizon: the smallest horizon the family builds at (nbeats needs 2: its seasonality basis has half the horizon in
        thetas), ``why_horizon`` the reason; ``1`` for a family that takes any.
    """

    variants: str
    max_channels: Optional[int] = None
    why_channels: str = ""
    static_batch: bool = False
    jit_refusal: Optional[str] = None
    min_lookback: Mapping[str, int] = field(default_factory=dict)
    why_lookback: str = ""
    min_lookback_over: Mapping[str, Tuple[str, int]] = field(default_factory=dict)
    why_over: str = ""
    min_horizon: int = 1
    why_horizon: str = ""


# DECISION plan-2026-11-13T105600-3a8319ef/D-028
# The default is jit_compile=False for every family; true is an opt-in the task accepts for every family but ets, where
# XLA fails the gradient of the unrolled loop (measured on GPU 0: InvalidArgumentError at node
# gradient_tape/ets_model_1/while/.../accumulator at the first step). Do NOT let the opt-in through for ets "to see".
FAMILIES: Dict[str, FamilyRow] = {
    # The minimums are MEASURED (by building each variant at decreasing lookbacks), not derived: the variants listed are the
    # ones small enough to build; tirex large and timestral 28M and up were not built and take the floor of every family.
    NBEATS: FamilyRow(
        f"{_HERE}:NBEATS_VARIANTS", min_horizon=2,
        why_horizon="its seasonality stack has forecast_length // 2 harmonics, which is 0 at a horizon of 1; the library raises "
        "'thetas_dim[1] must be positive'"),
    PRISM: FamilyRow(
        f"{_TS}.prism.model:PRISM.MODEL_VARIANTS", min_lookback={"tiny": 8, "small": 31, "base": 31, "large": 61},
        why_lookback="its wavelet tree, set by the variant's tree_depth and num_wavelet_levels, needs a context long enough for "
        "the deepest band; the library refuses a shorter one as an unsupportable PRISM configuration"),
    TIREX: FamilyRow(f"{_TS}.tirex.model:TiRexCore.MODEL_VARIANTS", 1, "TiRexCore reads one series, (B, L, 1)"),
    XLSTM: FamilyRow(f"{_TS}.xlstm.forecaster:xLSTMForecaster.MODEL_VARIANTS"),
    ETS: FamilyRow(
        f"{_TS}.ets.model:ETS_VARIANTS", 1, "ETSModel is a recursion over one scalar series, (B, L, 1)",
        min_lookback_over={"AAA": ("seasonal_period", 1)},
        why_over="the initial seasonal state needs a full seasonal period plus one step to update",
        jit_refusal=(
            "XLA fails the gradient of the unrolled smoothing loop (InvalidArgumentError at "
            "gradient_tape/ets_model_1/while/.../accumulator, measured on GPU 0)"
        ),
    ),
    TIMESTRAL: FamilyRow(f"{_TS}.timestral.model:TimestralModel.MODEL_VARIANTS", static_batch=True),
    MDN: FamilyRow(f"{_HERE}:MDN_VARIANTS", 1, "the mixture is over the horizon of one series, (B, L, 1)"),
}


def check_supported_family(model: str) -> None:
    """Refuse ``deepar`` and ``adaptive_ema`` by name before anything is imported; see the module contract."""
    family = model.partition("/")[0]
    if family in UNSUPPORTED:
        raise ConfigError(
            f"Invalid value for 'model': the family {family!r} is not trained by {TASK}: {UNSUPPORTED[family]}; the "
            f"families of {TASK} are {', '.join(FAMILIES)}"
        )


def geometry_problems(family: str, channels: int, horizon: int, model_args: Mapping[str, Any]) -> List[str]:
    """Return the lines refusing a channel count or a horizon the family cannot take; see the module contract."""
    lines: List[str] = []
    row = FAMILIES[family]
    if row.max_channels is not None and channels > row.max_channels:
        lines.append(
            f"Invalid value for 'data.value_cols': {family!r} takes {row.max_channels} channel ({row.why_channels}), "
            f"got {channels}; name one column, or choose a family that takes several"
        )
    if family == XLSTM and model_args.get("quantile_levels") is not None and channels > 1:
        lines.append(
            f"Invalid value for 'data.value_cols': an xlstm with a quantile head forecasts one series (its output is "
            f"(batch, horizon, quantiles), with no channel axis, measured at 2 input channels), got {channels} channels; "
            "name one column, or leave model_args.quantile_levels out for a point head"
        )
    # DECISION plan-2026-11-13T105600-3a8319ef/D-003
    # timestral's `call` is the forecaster's first block only up to one patch: later patches are decoded in the anchor
    # frame by TimestralForecaster (measured: horizon 48 at patch 32 differs from the forecaster by 2.54 on a scale of
    # 16, steps 1 to 32 by 0.0). Do NOT lift the limit by decoding more blocks here (that is TimestralForecaster, a
    # second decode row) and do NOT let a longer horizon through unscored: the metric would grade a frame the model
    # was never trained in.
    patch = model_args.get("patch_size", TIMESTRAL_PATCH)
    if family == TIMESTRAL and horizon > patch:
        lines.append(
            f"Invalid value for 'data.horizon': timestral forecasts one patch of {patch} points at a time (its call is the "
            f"forecaster's first block only up to patch_size), so the horizon must be at most {patch}, got {horizon}"
        )
    return lines


def min_lookback(family: str, variant: str) -> int:
    """Return the smallest lookback ``family/variant`` builds at: the row's measured minimum, else :data:`MIN_LOOKBACK`."""
    return FAMILIES[family].min_lookback.get(variant, MIN_LOOKBACK)


def window_problems(
    family: str, variant: str, lookback: int, horizon: int, model_args: Optional[Mapping[str, Any]] = None
) -> List[str]:
    """Return the lines refusing a lookback or horizon below the family's measured minimum; see the module contract."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-042
    # The minimum of each family is a measured row (`FamilyRow.min_lookback`, `min_horizon`), refused BEFORE the claim: a prism
    # that cannot build at the lookback asked, or an nbeats at horizon 1, used to be refused only after the run directory was
    # claimed ("'out' is now used", review W8). Do NOT guess a number from the library's message (it names a wavelet tree, not
    # a lookback) and do NOT raise it to "the largest variant's" for the family: prism/tiny builds at 8 and prism/large at 61, and a
    # family-wide 61 would refuse every tiny run that works.
    row = FAMILIES[family]
    lines: List[str] = []
    floor = min_lookback(family, variant)
    if lookback < floor:
        why = row.why_lookback if floor > MIN_LOOKBACK else "one context value has no spread to standardize by"
        lines.append(
            f"Invalid value for 'data.lookback': {family}/{variant} needs a lookback of at least {floor} ({why}), got {lookback}")
    over = row.min_lookback_over.get(variant)
    period = (model_args or {}).get(over[0]) if over else None
    if over and not (isinstance(period, int) and not isinstance(period, bool) and period >= 2):
        lines.append(
            f"Invalid value for 'model_args.{over[0]}': {family}/{variant} needs model_args.{over[0]}, an int of at least 2 "
            f"({row.why_over}), got {period!r}")
    elif over and lookback < period + over[1]:
        lines.append(
            f"Invalid value for 'data.lookback': {family}/{variant} with model_args.{over[0]}={period} needs a lookback of at "
            f"least {period + over[1]} ({row.why_over}), got {lookback}")
    if horizon < row.min_horizon:
        lines.append(
            f"Invalid value for 'data.horizon': {family} needs a horizon of at least {row.min_horizon} ({row.why_horizon}), "
            f"got {horizon}")
    return lines


def jit_problem(family: str, jit_compile: bool) -> Optional[str]:
    """Return the refusal text of ``jit_compile=true`` for ``family``, or ``None``; see the module contract."""
    reason = FAMILIES[family].jit_refusal
    if jit_compile and reason:
        return f"Invalid value for 'jit_compile': {family!r} cannot be compiled with XLA: {reason}"
    return None


def flat_target(family: str, model_args: Mapping[str, Any]) -> bool:
    """Whether the family's loss wants a ``(batch, horizon)`` target; see the module contract."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-044
    # The ONE rule for the flat target, applied before the model exists (the task flattens the data before it builds anything).
    # Do NOT put it back on `forecast_run.DecodeRow` or `Decoder` as a field: it was there, nothing in production read it, and
    # the two copies could drift with no test holding them equal (the tests now take the rows' flat targets from this function).
    return family == TIREX or (family == XLSTM and model_args.get("quantile_levels") is not None)


def loss_problem(family: str, model_args: Mapping[str, Any], loss_name: Optional[str]) -> Optional[str]:
    """Return the refusal text of ``loss_args.name`` for a head with a fixed loss, or ``None``; see the module contract."""
    point = family in (NBEATS, ETS) or (family in (PRISM, XLSTM) and model_args.get("quantile_levels") is None)
    if loss_name is None or point:
        return None
    return (
        f"Invalid value for 'loss_args.name': {loss_name!r} is not accepted for {family!r} with this head: its loss is fixed "
        "(the pinball loss of a quantile head, the objective of timestral, the likelihood of mdn), not a choice; leave it null"
    )


def _offered(family: str) -> List[str]:
    return sorted({*ALLOWED[family], *REFUSED[family], *FORCED[family]})


def problems_of(family: str, model_args: Mapping[str, Any]) -> List[str]:
    """Return one line per refusal of ``model_args`` for ``family``; see the module contract."""
    lines: List[str] = []
    for key, value in model_args.items():
        name = f"model_args.{key}"
        if key in ALLOWED[family]:
            continue
        if key in FORCED[family]:
            forced, why = FORCED[family][key]
            if value != forced or isinstance(value, bool) != isinstance(forced, bool):
                lines.append(f"Invalid value for {name!r}: the {family!r} adapter sets it to {forced!r}: {why}, got {value!r}")
        elif key in SUPPLIED[family]:
            lines.append(f"Invalid value for {name!r}: {key} is set by the task ({SUPPLIED[family][key]}), got {value!r}")
        elif key in REFUSED[family]:
            lines.append(f"Invalid value for {name!r}: {key} is not offered by {family!r} ({REFUSED[family][key]}), got {value!r}")
        else:
            lines.append(unknown_key_message("model_args", key, _offered(family)))
    return lines


def factory_signatures(family: str) -> Dict[str, List[str]]:
    """Return ``{callable path: named parameters}`` of the library callables behind ``family`` (``**kwargs`` left out)."""
    return {path: named_parameters(path) for path in FACTORIES[family]}


def signature_census(
    family: str, signatures: Optional[Mapping[str, Sequence[str]]] = None
) -> Dict[str, List[str]]:
    """Return the callable parameters no table names and the table keys no callable has; see the module contract."""
    if signatures is None:
        signatures = factory_signatures(family)
    offered = {*ALLOWED[family], *REFUSED[family], *FORCED[family], *SUPPLIED[family]}
    present = {name for names in signatures.values() for name in names}
    return {"unclassified": sorted(present - offered), "stale": sorted(offered - present)}


# ---------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------


def _prepare(
    family: str, variant: str, num_classes: Optional[int], input_shape: Sequence[Any], horizon: Any,
    model_args: Mapping[str, Any],
) -> Tuple[int, int]:
    """Refuse what the family does not take and return ``(lookback, channels)``; the adapters' common door."""
    if num_classes is not None:
        raise ValueError(f"the {family!r} family is a forecaster and has no classes: num_classes must be None, got {num_classes!r}")
    if len(input_shape) != 2 or not all(is_count(side) for side in input_shape):
        raise ValueError(f"input_shape must be (lookback, channels), each a positive int, got {tuple(input_shape)!r}")
    if not is_count(horizon):
        raise ValueError(f"horizon must be a positive int, got {horizon!r}")
    lookback, channels = input_shape
    lines = (problems_of(family, model_args) + geometry_problems(family, channels, horizon, model_args)
             + window_problems(family, variant, lookback, horizon, model_args))
    if lines:
        raise ValueError("; ".join(lines))
    return lookback, channels


def _variant(table: Any, variant: str) -> Any:
    """The row of ``variant`` in a variants mapping, or the name itself for ETS's tuple; ``ValueError`` for an unknown one."""
    if variant not in table:
        raise ValueError(f"Unknown variant {variant!r}. Available variants: {sorted(table)}")
    return table[variant] if isinstance(table, Mapping) else variant


def _plain(family: str, model_args: Mapping[str, Any]) -> Dict[str, Any]:
    """``model_args`` without the keys the adapter sets itself (a user ``use_normalization=false`` equals the forced value)."""
    return {key: value for key, value in model_args.items() if key not in FORCED[family]}


def _head(model_args: Mapping[str, Any], prism_style: bool) -> Dict[str, Any]:
    """The quantile keywords of prism and xlstm: a head iff ``quantile_levels`` is given, else a point head."""
    levels = model_args.get("quantile_levels")
    rest = {key: value for key, value in model_args.items() if key != "quantile_levels"}
    if levels is None:
        return {**rest, "use_quantile_head": False}
    levels = [float(level) for level in levels]
    head: Dict[str, Any] = {**rest, "use_quantile_head": True, "quantile_levels": levels}
    if prism_style:
        head["num_quantiles"] = len(levels)
    return head


def create_nbeats(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], horizon: int, **model_args: Any
) -> Any:
    """Build an N-BEATS forecaster; see the module contract."""
    lookback, channels = _prepare(NBEATS, variant, num_classes, input_shape, horizon, model_args)
    kwargs = {**_variant(NBEATS_VARIANTS, variant), **_plain(NBEATS, model_args), "use_normalization": False}
    from dl_techniques.models.time_series.nbeats.nbeats import create_nbeats_model

    model = create_nbeats_model(
        backcast_length=lookback, forecast_length=horizon, input_dim=channels, output_dim=channels, **kwargs)
    if not model.built:
        model.build((None, lookback, channels))
    return model


def create_prism(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], horizon: int, **model_args: Any
) -> Any:
    """Build a PRISM forecaster, point or quantile head; see the module contract."""
    lookback, channels = _prepare(PRISM, variant, num_classes, input_shape, horizon, model_args)
    from dl_techniques.models.time_series.prism.model import PRISM as Model

    model = Model.from_variant(
        variant, context_len=lookback, forecast_len=horizon, num_features=channels, **_head(model_args, True))
    model.build((None, lookback, channels))
    return model


def create_tirex(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], horizon: int, **model_args: Any
) -> Any:
    """Build a TiRex forecaster (always a quantile head); see the module contract."""
    lookback, _ = _prepare(TIREX, variant, num_classes, input_shape, horizon, model_args)
    from dl_techniques.models.time_series.tirex.model import create_tirex_by_variant

    return create_tirex_by_variant(
        variant, input_length=lookback, prediction_length=horizon, use_normalization=False, **_plain(TIREX, model_args))


def create_xlstm(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], horizon: int, **model_args: Any
) -> Any:
    """Build an xLSTM forecaster, point or quantile head; see the module contract."""
    lookback, channels = _prepare(XLSTM, variant, num_classes, input_shape, horizon, model_args)
    from dl_techniques.models.time_series.xlstm.forecaster import xLSTMForecaster

    model = xLSTMForecaster.from_variant(
        variant, input_length=lookback, prediction_length=horizon, num_features=channels, use_normalization=False,
        **_head(_plain(XLSTM, model_args), False))
    model.build((None, lookback, channels))
    return model


def create_ets(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], horizon: int, **model_args: Any
) -> Any:
    """Build an ETS forecaster from the library's variant tuple; see the module contract."""
    lookback, channels = _prepare(ETS, variant, num_classes, input_shape, horizon, model_args)
    from dl_techniques.models.time_series.ets.model import ETS_VARIANTS, create_ets as create

    _variant(ETS_VARIANTS, variant)
    model = create(variant, horizon=horizon, **model_args)
    model.build((None, lookback, channels))
    return model


def create_timestral(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], horizon: int, batch_size: int,
    **model_args: Any
) -> Any:
    """Build a Timestral forecaster at a static batch; see the module contract."""
    lookback, channels = _prepare(TIMESTRAL, variant, num_classes, input_shape, horizon, model_args)
    if not is_count(batch_size):
        raise ValueError(f"timestral needs a static batch: batch_size must be a positive int, got {batch_size!r}")
    from dl_techniques.models.time_series.timestral.model import create_timestral_by_variant

    model = create_timestral_by_variant(variant, prediction_length=horizon, **model_args)
    model.build((batch_size, lookback, channels))
    return model


def create_mdn(
    *, variant: str, num_classes: Optional[int] = None, input_shape: Sequence[Any], horizon: int, **model_args: Any
) -> Any:
    """Build a mixture-density forecaster: a ``Flatten`` stub in front of ``MDNModel``; see the module contract."""
    lookback, channels = _prepare(MDN, variant, num_classes, input_shape, horizon, model_args)
    kwargs = {**_variant(MDN_VARIANTS, variant), **model_args}
    import keras
    from dl_techniques.models.time_series.mdn.model import MDNModel

    inputs = keras.Input(shape=(lookback, channels))
    outputs = MDNModel(output_dimension=horizon, name="mdn_core", **kwargs)(keras.layers.Flatten()(inputs))
    return keras.Model(inputs, outputs, name="mdn_forecaster")


# ---------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------


def resolve_forecast_args(
    context: AutoContext, optim_args: OptimArgs, model_args: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the adapter's keywords: the recipe's, then the caller's; see the module contract."""
    # The adapters build no kernel regularizer, so sgd's decay would be dropped without a word; the other task
    # resolvers refuse it for the same reason.
    recipes.refuse_dropped_sgd_decay(optim_args, TASK, context.family)
    merged = recipes.merge_model_args(context, optim_args, model_args, {})
    lines = problems_of(context.family, merged)
    if lines:
        raise ConfigValidationError(lines)
    return merged


# ---------------------------------------------------------------------
# The recipes
# ---------------------------------------------------------------------

EPOCHS = 10
BATCH_SIZE = 128
LOOKBACK, HORIZON = 168, 24
WEIGHT_DECAY = 0.004
WARMUP_EPOCHS = 1
GLOBAL_CLIPNORM = 1.0
#: Early stopping counts from the first epoch; the default run is 10 epochs, so it never stops early, and a longer one
#: stops after 10 epochs without a better ``val_mae``.
PATIENCE = 10


def _forecast_recipe(*, lr: float = 1e-4, warmup_epochs: int = WARMUP_EPOCHS) -> Dict[str, Any]:
    """Return a forecasting recipe: every key of the groups the task resolves."""
    # DECISION plan-2026-11-13T105600-3a8319ef/D-032
    # DECISION plan-2026-11-13T105600-3a8319ef/D-034
    # Ten epochs of 500 steps (5,000 steps, val_mae 30 percent under the better baseline for nbeats; 400 steps already
    # beat the bar by 7 to 9 percent), not the legacy 150 or 200: no default shorter than 400 steps meets the bar, and
    # a longer one buys 0.02 of val_mae. The nbeats recipe, one for every family unless a family says otherwise: AdamW lr 1e-4, weight decay 0.004 (what the
    # legacy `optimizer='adamw'` string gave), ONE warmup epoch of 500 steps (the legacy 1000 steps is two epochs at
    # 500; measured equal at 2,000 steps), cosine over the whole run to 1 percent, a GLOBAL clip of 1.0 (never a
    # literal `clipnorm`: the legacy `gradient_clip_norm` is a per-variable one), batch 128, no stacked L2 (the model
    # regularizers are refused). lr 3e-4 was faster in one seed; 1e-4 is a conservative default, not an optimum.
    return {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optim_args": recipes.optim_recipe(
            optimizer="adamw", lr=lr, weight_decay=WEIGHT_DECAY, warmup_epochs=warmup_epochs, decay_over="total",
            global_clipnorm=GLOBAL_CLIPNORM,
        ),
        "metric_args": {"watch_metric": WATCH_METRIC},
        "checkpoint_args": {"early_stopping_patience": PATIENCE},
        "data": {"lookback": LOOKBACK, "horizon": HORIZON},
        "model_args": {},
        "per_dataset": {},
        "per_variant": {},
    }


#: The recipe of each family. ets keeps the legacy 5e-2 with no warmup (a single scalar parameter does not move at 1e-4:
#: 20 steps at 1e-3 moved the bar's metric from 1.0069 to 1.0066, D-029) and mdn its own 5e-4.
RECIPES: Dict[str, Dict[str, Any]] = {
    NBEATS: _forecast_recipe(),
    PRISM: _forecast_recipe(),
    TIREX: _forecast_recipe(),
    XLSTM: _forecast_recipe(),
    ETS: _forecast_recipe(lr=5e-2, warmup_epochs=0),
    TIMESTRAL: _forecast_recipe(),
    MDN: _forecast_recipe(lr=5e-4),
}


def builtin_forecast_models() -> Tuple[ModelSpec, ...]:
    """Return the seven forecasting family specs, freshly built; see the module contract."""
    return tuple(
        ModelSpec(
            family=family,
            factory=f"{_HERE}:create_{family}",
            variants_attr=row.variants,
            tasks=(TASK,),
            defaults=RECIPES[family],
            resolve_model_args=f"{_HERE}:resolve_forecast_args",
        )
        for family, row in FAMILIES.items()
    )
