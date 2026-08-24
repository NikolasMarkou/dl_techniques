"""F-42: ``create_convunext_denoiser``'s norm default is a ``None`` SENTINEL.

The package docstring says "bias-free means no additive constants anywhere in
the forward path, which is what makes the residual degree-1 homogeneous". That
is necessary but not sufficient: homogeneity is also NORM-dependent. LayerNorm
divides by a per-sample std that itself scales with the input, so it is
scale-INVARIANT (degree 0), not degree-1 -- and ``create_convunext_denoiser``
defaulted to it, alone among the package's exported builders.

**Part of F-42 was already fixed and is REFUTED as stated.** ``create_convunext``
has carried a homogeneity warning at ``convunext/model.py:665`` since a previous
plan, and it DOES fire on the raw builder's default path (measured). So the
default was never silent. What was missing is the distinction the warning cannot
draw: it fires identically whether the caller chose ``'layernorm'`` deliberately
or passed nothing at all, which makes it unactionable.

The fix is therefore the ``None`` sentinel from F-42's own suggested shape, in
its WARN form, not its raise form: raising would break every omitted-kwarg
caller (``utils/multiplicative_miyasawa.py:835`` and ~20 test constructions) on
a signature the module docstring declares frozen. The resolved value stays
``'layernorm'``, so no built graph moves.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import inspect
import logging
from collections import Counter

import keras
import numpy as np
import pytest

import dl_techniques.models.bias_free_denoisers as bf_pkg
from dl_techniques.models.bias_free_denoisers.bfconvunext import (
    create_convunext_denoiser,
)

SMALL = dict(initial_filters=4, blocks_per_level=1, depth=2)
INPUT_SHAPE = (32, 32, 1)


def _norm_counts(model):
    return Counter(
        type(layer).__name__ for layer in model._flatten_layers(include_self=False)
    )


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _build_capturing(**kwargs):
    handler = _Capture()
    log = logging.getLogger("dl")
    log.addHandler(handler)
    try:
        model = create_convunext_denoiser(input_shape=INPUT_SHAPE, **SMALL, **kwargs)
    finally:
        log.removeHandler(handler)
    warnings = [r.getMessage() for r in handler.records
                if r.levelno >= logging.WARNING]
    return model, warnings


def _sentinel_warnings(warnings):
    return [w for w in warnings if "no block_normalization was passed" in w]


def _cpu():
    """A CPU device scope for the homogeneity measurements below.

    NOT optional and NOT a speed choice. The homogeneity residual for the
    BATCHNORM stack sits at the float32 floor (1.2e-06 .. 2.6e-06, measured on
    CPU over five seeds). On GPU 1 the SAME comparison measures 1.05e-03 --
    TF32's relative floor, ~1000x higher -- which is indistinguishable from a
    real degree-0 leak and made three of these assertions fail for a reason
    that has nothing to do with the code under test.

    A device scope is used rather than
    ``tf.config.experimental.enable_tensor_float_32_execution(False)`` because
    that setting is PROCESS-GLOBAL: flipping it here would silently retune every
    other test sharing the pytest session (this repository has already been bitten
    by a module doing exactly that).
    """
    import tensorflow as tf

    return tf.device("/CPU:0")


class TestTheSentinelIsTheDefault:

    def test_the_signature_default_is_none_not_a_string(self):
        default = inspect.signature(
            create_convunext_denoiser).parameters["block_normalization"].default
        assert default is None, (
            "a string default cannot distinguish 'the caller chose layernorm' "
            "from 'the caller chose nothing', which is the whole point"
        )

    def test_omitting_it_warns_that_the_choice_was_not_made(self):
        _, warnings = _build_capturing()
        hits = _sentinel_warnings(warnings)
        assert len(hits) == 1, warnings
        assert "not degree-1" in hits[0].lower() or "NOT degree-1" in hits[0]
        assert "batchnorm" in hits[0]

    def test_passing_layernorm_explicitly_does_NOT_warn_about_the_choice(self):
        """The discrimination the pre-existing guard cannot make.

        ``convunext/model.py``'s own homogeneity warning still fires here -- it
        is about the VALUE, and the value is unchanged -- but the sentinel
        warning, which is about the ABSENCE of a choice, must not.
        """
        _, warnings = _build_capturing(block_normalization="layernorm")
        assert _sentinel_warnings(warnings) == [], warnings

    def test_passing_batchnorm_does_not_warn_at_all(self):
        _, warnings = _build_capturing(block_normalization="batchnorm")
        assert _sentinel_warnings(warnings) == [], warnings
        assert not any("degree-1" in w for w in warnings), warnings


class TestTheResolvedGraphIsUnchanged:
    """The sentinel must be a pure observability change, not a numerics one."""

    def test_the_sentinel_resolves_to_layernorm(self):
        model, _ = _build_capturing()
        counts = _norm_counts(model)
        assert counts.get("LayerNormalization", 0) > 0, counts
        assert counts.get("BiasFreeBatchNorm", 0) == 0, counts

    def test_default_and_explicit_layernorm_build_identical_norm_censuses(self):
        default, _ = _build_capturing()
        explicit, _ = _build_capturing(block_normalization="layernorm")
        assert _norm_counts(default) == _norm_counts(explicit)
        assert default.count_params() == explicit.count_params()

    def test_batchnorm_actually_changes_the_graph(self):
        """CONTROL: without this the two assertions above prove nothing."""
        model, _ = _build_capturing(block_normalization="batchnorm")
        counts = _norm_counts(model)
        assert counts.get("BiasFreeBatchNorm", 0) > 0, counts


class TestTheHomogeneityClaimTheWarningIsAbout:
    """The warning is true: measure both norms' degree-1 behaviour.

    NOTE the ``relu`` overrides. Homogeneity has TWO preconditions, not one, and
    this suite isolates the norm: ``block_activation`` defaults to ``'gelu'``,
    which is not positively homogeneous, and a woken gelu stack measures
    5.1e-03 .. 1.2e-02 even under BATCHNORM (measured, five seeds) -- so a
    comparison at the default activation would attribute the activation's error
    to the norm and the batchnorm control below would fail for the wrong
    reason. ``src/train/bfunet/train_convunext_denoiser.py`` pins a leaky-relu
    for the same reason.
    """

    #: A WIDER stack than ``SMALL``. At ``initial_filters=4,
    #: blocks_per_level=1`` the GRN's mean-of-squares denominator is computed
    #: over four channels and the ratio is numerically ragged: the woken
    #: BATCHNORM control measured 3.65e-03 there -- three orders off its true
    #: floor -- which would have been misread as "batchnorm is not homogeneous
    #: either". At 8 channels x 2 blocks the four numbers below are stable to
    #: within a factor of 2 across five `keras.utils.set_random_seed` values
    #: (measured).
    WIDE = dict(initial_filters=8, blocks_per_level=2)
    HOMOGENEOUS = dict(block_activation="relu", stem_activation="relu",
                       drop_path_rate=0.0)

    @staticmethod
    def _homogeneity_error(model, scale=3.0, seed=0):
        """THE GPU CANNOT ANSWER THIS QUESTION -- see ``_cpu`` below."""
        x = np.random.RandomState(seed).randn(2, *INPUT_SHAPE).astype("float32")
        with _cpu():
            f_x = keras.ops.convert_to_numpy(model(x, training=False))
            f_ax = keras.ops.convert_to_numpy(model(scale * x, training=False))
        denom = float(np.max(np.abs(scale * f_x))) + 1e-12
        return float(np.max(np.abs(f_ax - scale * f_x))) / denom

    @staticmethod
    def _wake_the_residual_branches(model, seed=1):
        """MANDATORY: an untrained ConvNeXt V2 cannot answer this question.

        MEASURED over five seeds: at initialization the LAYERNORM stack scores
        1.2e-06 .. 2.8e-06 -- the float32 floor, i.e. it looks perfectly
        degree-1 -- because ConvNeXt V2's GRN ``gamma`` is ZERO-initialized, so
        the residual branch that contains the LayerNorm contributes nothing and
        the model is its skip path. The very defect under test is invisible at
        init. After perturbing the kernels and gammas the same stack scores
        4.3e-01 .. 5.5e-01 while batchnorm stays at 1.2e-06 .. 2.6e-06: a
        ~250,000x separation that only exists once the branch is awake.
        """
        rng = np.random.RandomState(seed)
        for var in model.weights:
            if "gamma" in var.name or "kernel" in var.name:
                var.assign(np.asarray(var)
                           + rng.randn(*var.shape).astype("float32") * 0.5)
        return model

    def _raw(self, **kwargs):
        keras.utils.set_random_seed(0)
        with _cpu():
            return create_convunext_denoiser(
                input_shape=INPUT_SHAPE, depth=SMALL["depth"],
                **self.WIDE, **{**self.HOMOGENEOUS, **kwargs})

    def _woken(self, **kwargs):
        return self._wake_the_residual_branches(self._raw(**kwargs))

    def test_the_defaulted_layernorm_stack_is_NOT_degree_one(self):
        assert self._homogeneity_error(self._woken()) > 1e-1

    def test_the_batchnorm_stack_IS_degree_one(self):
        assert self._homogeneity_error(
            self._woken(block_normalization="batchnorm")) < 1e-4

    @pytest.mark.parametrize("norm", ["layernorm", "batchnorm"])
    def test_at_initialization_the_two_are_INDISTINGUISHABLE(self, norm):
        """The trap, pinned so nobody re-derives the wrong conclusion.

        Without the perturbation both stacks sit at the float32 floor and a
        naive probe concludes the default is fine.
        """
        assert self._homogeneity_error(self._raw(block_normalization=norm)) < 1e-4

    def test_the_activation_is_the_OTHER_precondition(self):
        """Guard against reading the norm result as the whole story.

        A batchnorm stack with the DEFAULT gelu activation is not degree-1
        either, so ``block_normalization='batchnorm'`` alone does not buy the
        Miyasawa reading -- the activation must be positively homogeneous too,
        which is why ``src/train/bfunet/train_convunext_denoiser.py`` pins a
        leaky-relu. MEASURED woken, five seeds: 5.1e-03 .. 1.2e-02.
        """
        model = self._woken(block_normalization="batchnorm",
                            block_activation="gelu", stem_activation="gelu")
        assert self._homogeneity_error(model) > 1e-3


class TestThePackageDocstringNoLongerOverclaims:

    def test_it_names_the_one_builder_whose_default_is_not_homogeneous(self):
        doc = bf_pkg.__doc__
        assert "create_convunext_denoiser" in doc
        assert "NORM-dependent" in doc
        assert "necessary but not sufficient" in doc
