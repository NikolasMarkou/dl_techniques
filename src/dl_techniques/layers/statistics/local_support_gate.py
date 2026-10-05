"""
**What it does:**

`LocalSupportGate` decides, per token, whether a learned weight update is
"local" to the data distribution that produced it. It is the gate half of
*Local Support Learning* (LSL): paired with an additive low-rank adapter, it
restricts that adapter's contribution to inputs that look like the ones it was
trained on, leaving behaviour elsewhere untouched.

The decision is an **input-dependent likelihood ratio between two Gaussian
mixtures**, not a fixed threshold on one density. `Φ_pos` is fitted to the
current phase's data; `Φ_neg` is fitted to a small generic sample standing in
for the pretraining distribution. A token routes to the adapter when
`log Φ_pos(x) > log Φ_neg(x)`:

.. code-block:: text

    x  (..., input_dim)
          │
          ▼  [optional Johnson-Lindenstrauss projection: input_dim → projection_dim]
      z  (..., projection_dim)          projection_dim <= input_dim
          │
          ├──────────────►  log Φ_pos(z)          EM-fitted, then frozen
          │                        │
          └──────────────►  log Φ_neg(z)          EM-fitted, then frozen
                                   │
                                   ▼  score = log Φ_pos(z) − log Φ_neg(z)
          ┌────────────────────────┴─────────────────────┐
          ▼                                              ▼
    output_mode='scores'                        output_mode='hard'
    the raw score                              1[score > threshold]
          │
          ▼
    output_mode='smoothed'   s_t = α·g_t + (1−α)·s_{t−1}
                             (causal EMA along smoothing_axis; α = 1 disables)

**Why two mixtures and not one threshold.** A single density needs a threshold,
and the choice of threshold trades the gate's *deficit* (rejecting
in-distribution tokens) against its *excess* (accepting out-of-distribution
ones) with no principled way to set it — any objective computed on the current
phase's data is blind to the excess, because the current data carries no mass
there. The ratio form takes the reference distribution as the threshold, which
is why `Φ_neg` need only capture the *width* of a generic corpus rather than its
support: empirically ~1M tokens, a negligible fraction of any real pretraining
set, suffices.

**The inductive bias is locality.** A Gaussian density decays exponentially
away from its training data, so `Φ_pos` falls off faster than the wider `Φ_neg`
and the gate tends to stay closed on inputs it has never seen. That property,
not in-domain expressiveness, is what produces retention. The ablations in the
paper make this concrete: a non-local classifier gate reaches comparable
in-distribution accuracy but opens on far more out-of-distribution tokens and
retains markedly less.

**Two fitting paths, because they trade accuracy for memory:**

`:meth:`fit_pos` / :meth:`fit_neg`` run **exact batch EM** over a sample held in
memory. Monotone non-decreasing likelihood, converged parameters, the reference
behaviour.

:meth:`observe` runs **stochastic-approximation EM**, accumulating *sufficient
statistics* (`n_k`, `Σ r_k·z`, `Σ r_k·z²`) into an EMA with step `1/t` and
taking an **exact M-step from the accumulated statistics** on every batch. Its
persistent state is `O(K·d)` and independent of the number of tokens, so a gate
can be fitted during ordinary training with no activation buffer and no
layer-sequential sweep.

Accumulating statistics rather than interpolating parameters is not a stylistic
choice. Step-sizing the parameters toward each batch's M-step — the obvious
reading of "minibatch EM" — converges to "last batch wins": as the step size
approaches 1 the parameters stop carrying any history. Measured on a separable
8-component 32-dimensional mixture, that variant froze at variance 0.12 and
returned *identical* parameters after 1 pass and after 60 passes. Accumulating
statistics keeps a `1/t`-weighted average over the whole stream and is stable
across pass counts, at `O(K·d)` state.

The tradeoff of the streaming path is not free, and it is specific. Measured on
the same mixture, `observe` left the fit's log-likelihood ~20% below exact EM,
and that gap is a **floor, not slow convergence** — flat from 10 passes to 20.
Fitted variances come out inflated for the same reason (statistics computed
under stale parameters get averaged with current ones).

What matters for a gate is the decision, not the likelihood, and the two behave
differently: the in-distribution hit rate stayed ≈0.999, while the
**out-of-distribution hit rate rose roughly 200×** (1e-4 → ~2e-2). Streaming EM
does not weaken the gate uniformly — it erodes precisely the "stays closed on
unseen inputs" property the method depends on. Treat :meth:`observe` as the
memory-constrained option and the exact path as the default.

**Collapsed components are repaired explicitly.** When a component's accumulated
mass falls to zero its mean is undefined and its variance collapses toward the
cancellation floor. Rather than let a floored value propagate, the gate counts
the collapsed components, re-seeds each from the batch that revealed it, and
reports the total through :meth:`num_dead_components` so a caller can fail
loudly or disable the adapter. This is the guard the reference implementation
reaches for with its `skip_on_fail` flag; a ramped step size reaches the same
state by a different route — measured, it drove a whole mixture to NaN.

**References:**
    - Ben-Kish, A., Kumar, A., Glass, J., & Giryes, R., 2026. Local Support
      Learning. (https://arxiv.org/abs/2610.02126)
    - O'Hagan, M., 1995. Minibatch Expectation Maximization. University of
      Sheffield technical report — the stochastic-approximation path.
    - Neal, R. M., & Hinton, G. E., 1998. A View of the EM Algorithm that
      Justifies Incremental, Sparse, and Other Variants. In Learning in
      Graphical Models, pp. 355-368. Kluwer.
      (https://www.cs.toronto.edu/~hinton/absps/em.pdf)
    - Houtko, T., & Kaski, S., 2009. Acceleration of EM Algorithm by Means of
      the Johnson-Lindenstrauss Lemma. (https://arxiv.org/abs/0905.3851)

Examples:
    ```python
    gate = LocalSupportGate(
        input_dim=2048,
        projection_dim=256,
        pos_components=16,
        neg_components=32,
    )

    # Exact fits. Both mixtures are frozen afterwards.
    gate.fit_pos(phase_activations)      # numpy (n, 2048)
    gate.fit_neg(generic_activations)    # numpy (m, 2048)

    # Per-token routing decision at inference time.
    decision = gate(activations, training=False)  # (batch, seq_len)
    ```

    ```python
    # Streaming fit instead: O(K*d) state, no activation buffer.
    gate = LocalSupportGate(input_dim=2048, projection_dim=256)
    for activations in stream_of_activation_batches:
        gate.observe(activations, which="pos")
    ```
"""

import math
from typing import Any, Dict, Literal, Optional, Tuple

import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

#: Covariance structure of both fitted mixtures. Full covariance costs `O(d^2)`
#: parameters per component and `O(d^2)` EM arithmetic; the reference finds
#: diagonal sufficient in practice once the inputs are projected down.
CovarianceType = Literal['diag']

#: Which fitting path this gate is configured for. Recorded and reported rather
#: than silently switching behaviour: the two paths differ in cost AND in
#: out-of-distribution behaviour, so the choice is the caller's to make and see.
FitMode = Literal['batch', 'minibatch']

#: What ``call`` returns.
OutputMode = Literal['hard', 'smoothed', 'scores']

#: Which of the layer's two mixtures a call refers to.
MixtureSide = Literal['pos', 'neg']

#: Floor on a fitted variance. A Gaussian density contains `log(variance)`, so a
#: variance at or below zero is a NaN waiting to happen, and at exactly zero it
#: is a division by zero.
DEFAULT_VARIANCE_FLOOR: float = 1e-6

#: Accumulated component mass below which a component counts as collapsed.
DEFAULT_DEAD_MASS_THRESHOLD: float = 1e-8

#: Default EMA coefficient. `1.0` disables smoothing; lower values smooth more.
DEFAULT_SMOOTHING_ALPHA: float = 0.2

#: `log(2*pi)`, the constant in the diagonal-Gaussian log-density. A module
#: constant rather than `ops.pi`, which `keras.ops` does not expose.
_LOG_2PI: float = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.statistics.local_support_gate")
class LocalSupportGate(keras.layers.Layer):
    """
    Per-token support gate: routes an input to a weight update only when it
    resembles the data that update was trained on.

    Fits two diagonal Gaussian mixtures — one to the current phase's activations
    (`pos`) and one to a small generic sample standing in for the pretraining
    distribution (`neg`) — and emits `1[log Φ_pos(x) > log Φ_neg(x)]` per token.
    The positive mixture is refitted from updated representations as a phase
    proceeds; the negative describes a fixed corpus and is fitted once.

    The layer is **inert until fitted**: an unfitted mixture pair reports every
    token as out-of-support, so a freshly constructed gate routes nothing rather
    than routing by its arbitrary initialisation. Fitting is always an explicit
    call (:meth:`fit_pos`, :meth:`fit_neg`, :meth:`fit`, :meth:`observe`) and
    never a side effect of ``call``.

    Both mixtures are stored as **non-trainable** weights. The gate is fitted by
    expectation-maximization and then frozen; it is not a gradient-trained
    module. That is the opposite lifecycle from
    :class:`dl_techniques.layers.mixtures.gmm.GMMLayer`, which is
    differentiable and trained end to end — its density math is the useful prior
    art here, but its training path is not this layer's.

    Architecture:

    .. code-block:: text

        inputs  (..., input_dim)
              │
              ▼  [projection_dim < input_dim]  fixed seeded random projection
          z  (..., projection_dim)              non-trainable
              │
              ├────────►  Φ_pos: pos_components diagonal Gaussians → log p_pos
              └────────►  Φ_neg: neg_components diagonal Gaussians → log p_neg
                             │
                             ▼  score = log p_pos − log p_neg   (...)
              ┌──────────────┼───────────────────┐
              ▼              ▼                   ▼
          'hard'        'smoothed'            'scores'
         1[· > thr]     causal EMA over        the raw
                        smoothing_axis         score
              │
              ▼  squeeze (or not — see output_shape_reduced)
          gate  (...) or (..., 1)

    :param input_dim: Width of the activations this gate scores. Must be
        positive and statically known.
    :type input_dim: int
    :param projection_dim: Target width of the Johnson-Lindenstrauss projection
        applied before fitting and gating. Reduces the fitted dimensionality
        from `input_dim` to this value, cutting EM arithmetic and stored
        parameters together. `None` (the default) disables projection. Must be
        in `(0, input_dim]` when given.
    :type projection_dim: Optional[int]
    :param pos_components: Number of Gaussian components in the positive
        (in-phase) mixture. Must be positive.
    :type pos_components: int
    :param neg_components: Number of Gaussian components in the negative
        (generic-reference) mixture. Must be positive.
    :type neg_components: int
    :param covariance_type: Covariance structure of both mixtures. Only
        `'diag'` is supported.
    :type covariance_type: CovarianceType
    :param fit_mode: Which fitting path this gate is intended for. `'batch'` for
        the exact `:meth:`fit_pos`/:meth:`fit_neg` path (monotone likelihood,
        the reference behaviour), `'minibatch'` for the streaming
        `:meth:`observe` path, whose `O(K·d)` state lets the gate be fitted
        during ordinary training with no activation buffer. This records and
        reports the intent; it does not silently switch entry points. Streaming
        inflates fitted variances and degrades the out-of-distribution hit rate
        — see the module docstring for the measured magnitude.
    :type fit_mode: FitMode
    :param smoothing_alpha: EMA coefficient for `output_mode='smoothed'`. At
        `1.0` smoothing is disabled and the EMA reduces to the raw decision. At
        `0.0` the state never updates, so the smoothed decision stays at its
        initial value. Must lie in `[0, 1]`.
    :type smoothing_alpha: float
    :param smoothing_axis: Axis along which the causal EMA runs. `-2` is the
        token axis of a `(batch, seq_len, dim)` activation. Ignored unless
        `output_mode='smoothed'`.
    :type smoothing_axis: int
    :param output_mode: What ``call`` returns. `'hard'` the binary decision,
        `'smoothed'` its causal EMA, `'scores'` the raw log-likelihood ratio
        (which is what a threshold sweep needs).
    :type output_mode: OutputMode
    :param decision_threshold: Threshold applied to the score — or, under
        `'smoothed'`, to the EMA of the binary decision. Defaults to `0.0`, the
        likelihood-ratio condition. Negative values open the gate more widely,
        positive values more narrowly.
    :type decision_threshold: float
    :param variance_floor: Floor applied to every fitted variance. Must be
        positive.
    :type variance_floor: float
    :param dead_mass_threshold: Accumulated component mass below which a
        component counts as collapsed and is re-seeded from the current batch.
        Must be positive.
    :type dead_mass_threshold: float
    :param seed: Seed for the projection matrix, mixture initialisation and
        revival draws. Two gates built with the same seed and fitted on the same
        data produce the same parameters. `None` draws from the global RNG,
        which makes the layer unreproducible — set it for anything testable.
    :type seed: Optional[int]
    :param output_shape_reduced: Whether ``call`` drops the trailing axis holding
        the score, returning shape `(...)` instead of `(..., 1)`. `True` by
        default, since a gate is almost always consumed by broadcasting against
        an activation.
    :type output_shape_reduced: bool
    :param name: Optional Keras layer name.
    :type name: Optional[str]
    :param kwargs: Extra arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :raises ValueError: If ``input_dim``, ``pos_components`` or
        ``neg_components`` is not positive; if ``projection_dim`` is given and
        is not in `(0, input_dim]`; if ``covariance_type`` is not `'diag'`; if
        ``fit_mode`` or ``output_mode`` is outside its literal; if
        ``smoothing_alpha`` is outside `[0, 1]`; or if ``variance_floor` /
        ``dead_mass_threshold`` is not positive.

    Example:
        .. code-block:: python

            gate = LocalSupportGate(input_dim=2048, projection_dim=256,
                                    pos_components=16, neg_components=32)

            gate.fit_pos(phase_activations)      # numpy (n, 2048)
            gate.fit_neg(generic_activations)    # numpy (m, 2048)

            delta = gate(some_activation, training=False)  # (batch, seq_len)

    Note:
        The gate decision carries **no gradient path** into the adapter it
        gates: the decision is a hard threshold on a non-differentiable EM fit.
        Gradients still flow through the adapter itself, because the gate
        multiplies the adapter's output rather than replacing it. A caller
        training on a mixture of in- and out-of-distribution data therefore
        trains the adapter only on the tokens the gate opened for — which is the
        intended behaviour, and why the reference implementation holds the gate
        open throughout a phase's own training data.
    """

    def __init__(
        self,
        input_dim: int,
        pos_components: int = 16,
        neg_components: int = 32,
        projection_dim: Optional[int] = None,
        covariance_type: CovarianceType = 'diag',
        fit_mode: FitMode = 'batch',
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        smoothing_axis: int = -2,
        output_mode: OutputMode = 'hard',
        decision_threshold: float = 0.0,
        variance_floor: float = DEFAULT_VARIANCE_FLOOR,
        dead_mass_threshold: float = DEFAULT_DEAD_MASS_THRESHOLD,
        seed: Optional[int] = None,
        output_shape_reduced: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Validate every argument and create every sub-layer and weight.

        All weights are created here rather than in ``build`` because their
        shapes are pure functions of stored config — none of them depends on the
        runtime input width, which is a constructor argument here. That is what
        lets ``compute_output_shape`` answer while the layer is still unbuilt.
        """
        super().__init__(name=name, **kwargs)

        # ---- validation -------------------------------------------------
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if pos_components <= 0:
            raise ValueError(f"pos_components must be positive, got {pos_components}")
        if neg_components <= 0:
            raise ValueError(f"neg_components must be positive, got {neg_components}")

        if projection_dim is not None:
            if projection_dim <= 0:
                raise ValueError(
                    f"projection_dim must be positive, got {projection_dim}"
                )
            if projection_dim > input_dim:
                # A projection WIDER than its input is not a JL sketch: it is an
                # arbitrary linear map that preserves nothing and costs
                # everything. Reject rather than accept a caller who inverted
                # the comparison.
                raise ValueError(
                    f"projection_dim must be at most input_dim ({input_dim}), "
                    f"got {projection_dim}. A projection wider than its input "
                    f"is not a Johnson-Lindenstrauss sketch and reduces nothing; "
                    f"pass None to disable projection."
                )

        if covariance_type != 'diag':
            raise ValueError(
                f"covariance_type must be 'diag', got {covariance_type!r}. Full "
                f"covariance costs O(d^2) parameters per component; the "
                f"reference implementation finds diagonal sufficient once the "
                f"inputs are projected down."
            )
        if fit_mode not in ('batch', 'minibatch'):
            raise ValueError(
                f"fit_mode must be 'batch' or 'minibatch', got {fit_mode!r}"
            )
        if output_mode not in ('hard', 'smoothed', 'scores'):
            raise ValueError(
                f"output_mode must be 'hard', 'smoothed' or 'scores', "
                f"got {output_mode!r}"
            )
        if not (0.0 <= smoothing_alpha <= 1.0):
            raise ValueError(
                f"smoothing_alpha must lie in [0, 1], got {smoothing_alpha}"
            )
        if variance_floor <= 0:
            raise ValueError(f"variance_floor must be positive, got {variance_floor}")
        if dead_mass_threshold <= 0:
            raise ValueError(
                f"dead_mass_threshold must be positive, got {dead_mass_threshold}"
            )

        # ---- stored config ----------------------------------------------
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.pos_components = pos_components
        self.neg_components = neg_components
        self.covariance_type = covariance_type
        self.fit_mode = fit_mode
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_axis = smoothing_axis
        self.output_mode = output_mode
        self.decision_threshold = decision_threshold
        self.variance_floor = variance_floor
        self.dead_mass_threshold = dead_mass_threshold
        self.seed = seed
        self.output_shape_reduced = output_shape_reduced

        #: The width actually operated on: the projection target when one is
        #: configured, else the input width. Resolved once in ``__init__``
        #: because it is a pure function of stored config, never of a weight.
        self.effective_dim = (
            projection_dim if projection_dim is not None else input_dim
        )

        #: Which mixtures have been fitted at least once. Initialised here as a
        #: real dict: it is read by ``is_fitted``, and a class-level default
        #: would be shared across instances (a fitted gate would report every
        #: other gate as fitted too).
        self._fitted_flags: Dict[str, bool] = {'pos': False, 'neg': False}
        self._num_revivals: int = 0

        # ---- sub-layers -------------------------------------------------
        # Created unconditionally per the create-unconditionally rule: the
        # projection is a real module whose absence must not change this layer's
        # weight layout when a caller flips projection_dim. It is NOT trainable
        # — a JL sketch is a fixed random matrix, and training it would forfeit
        # the distance bound the sketch exists to exploit.
        self.projection = keras.layers.Dense(
            units=self.effective_dim,
            use_bias=False,
            trainable=False,
            kernel_initializer=_sketch_initializer(self.seed, self.effective_dim),
            name='projection',
        )

        # ---- weights ----------------------------------------------------
        # Mixture state and the minibatch accumulators are all declared, so the
        # layer's weight layout does not change with fit_mode or with which
        # fitting path a caller has used.
        self._weights: Dict[str, keras.Variable] = {}
        for side, n_components in (
            ('pos', pos_components), ('neg', neg_components)
        ):
            self._weights[f'{side}_log_priors'] = self.add_weight(
                name=f'{side}_log_priors',
                shape=(n_components,),
                dtype='float32',
                initializer='zeros',
                trainable=False,
            )
            self._weights[f'{side}_means'] = self.add_weight(
                name=f'{side}_means',
                shape=(n_components, self.effective_dim),
                dtype='float32',
                initializer='zeros',
                trainable=False,
            )
            self._weights[f'{side}_variances'] = self.add_weight(
                name=f'{side}_variances',
                shape=(n_components, self.effective_dim),
                dtype='float32',
                # Seeded at the floor, not zero: `call` reads variances before
                # any fit, and log(0) there would be a NaN rather than a finite
                # "nothing is in memory yet". `initializers.Constant` rather than
                # the bare float because `add_weight` resolves a string or an
                # initializer instance, not a number.
                initializer=keras.initializers.Constant(self.variance_floor),
                trainable=False,
            )
            self._weights[f'{side}_acc_mass'] = self.add_weight(
                name=f'{side}_acc_mass',
                shape=(n_components,),
                dtype='float32',
                initializer='zeros',
                trainable=False,
            )
            self._weights[f'{side}_acc_weighted_sum'] = self.add_weight(
                name=f'{side}_acc_weighted_sum',
                shape=(n_components, self.effective_dim),
                dtype='float32',
                initializer='zeros',
                trainable=False,
            )
            self._weights[f'{side}_acc_weighted_sq_sum'] = self.add_weight(
                name=f'{side}_acc_weighted_sq_sum',
                shape=(n_components, self.effective_dim),
                dtype='float32',
                initializer='zeros',
                trainable=False,
            )

        #: Denominator of the `1/t` streaming step size. A `keras.Variable`
        #: because a fitting loop may itself be traced, and a Python counter
        #: would not advance under `tf.function` (guide §11.2).
        self._weights['em_step_counter'] = self.add_weight(
            name='em_step_counter',
            shape=(),
            dtype='float32',
            initializer='zeros',
            trainable=False,
        )

        logger.info(
            f"Initialized LocalSupportGate with input_dim={input_dim}, "
            f"effective_dim={self.effective_dim}, "
            f"pos_components={pos_components}, neg_components={neg_components}, "
            f"fit_mode={fit_mode}, output_mode={output_mode}, "
            f"smoothing_alpha={smoothing_alpha}, seed={seed}"
        )

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the projection against the shape it will actually see.

        The mixture weights are created in ``__init__`` because their shapes are
        pure functions of stored config; the projection is the one sub-layer
        whose weight depends on the runtime input width, so it is built here.
        This materializes exactly the tree ``call`` runs -- the projection and
        nothing else -- which is what build-parity by relative ``w.path`` checks
        against.

        :param input_shape: Shape of the input activations; only the last axis is
            used.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return
        # Guarded, not unconditional: `fit_pos`/`fit_neg`/`observe` apply the
        # projection to a sample BEFORE the layer has been called on a real
        # activation, which builds the sub-layer early. An unguarded second
        # `build()` on it raises "cannot add new elements of state to a layer
        # that is already built" the first time the layer is then called.
        if not self.projection.built:
            self.projection.build((None, self.input_dim))
        super().build(input_shape)

    # -----------------------------------------------------------------
    # shape
    # -----------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape of the gate, from stored config, while still unbuilt.

        The gate emits **one decision per token**, so the feature axis is
        consumed, not carried: an input of ``(batch, seq_len, input_dim)``
        produces ``(batch, seq_len)``. That is what makes the result directly
        broadcastable against an activation after a single ``expand_dims``.

        :param input_shape: Shape tuple of the input activations.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``input_shape`` with the feature axis dropped, plus a trailing
            singleton when ``output_shape_reduced`` is ``False``.
        :rtype: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` has rank < 2, which ``call`` also
            rejects.
        """
        shape = tuple(input_shape)
        if len(shape) < 2:
            raise ValueError(
                f"LocalSupportGate expects rank >= 2 inputs of shape "
                f"(..., {self.input_dim}), got shape {shape}"
            )
        per_token = shape[:-1]
        if self.output_shape_reduced:
            return per_token
        return per_token + (1,)

    # -----------------------------------------------------------------
    # fitted-state queries
    # -----------------------------------------------------------------

    def is_fitted(self, which: MixtureSide = 'pos') -> bool:
        """Whether ``which`` mixture has been fitted at least once.

        Deliberately a flag rather than a test on weight values: a fitted
        mixture can legitimately carry near-zero log-priors, and no value test
        distinguishes "fitted with degenerate parameters" from "never fitted".

        :param which: Which mixture to query.
        :type which: MixtureSide
        :return: ``True`` once ``which`` has been fitted.
        :rtype: bool
        """
        return bool(self._fitted_flags.get(which, False))

    def num_dead_components(self) -> int:
        """How many components have been re-seeded since construction.

        A non-zero count means a fit could not support the configured component
        count — most often because ``n_components`` is large relative to the
        sample, so some components capture no mass at all. The fit still
        completes; this is how a caller learns to distrust it.

        :return: The cumulative revival count, or `0` if never fitted.
        :rtype: int
        """
        return self._num_revivals

    # -----------------------------------------------------------------
    # fitting -- exact batch EM
    # -----------------------------------------------------------------

    def fit_pos(self, samples, max_iter: int = 100, tol: float = 1e-4) -> int:
        """Fit the positive mixture to ``samples`` by exact batch EM.

        :param samples: Tensor of shape ``(n, input_dim)``.
        :type samples: Any
        :param max_iter: Iteration cap. EM is monotone so this only bounds a
            slow fit; the tolerance normally stops it first.
        :type max_iter: int
        :param tol: Relative improvement below which to stop.
        :type tol: float
        :return: The number of iterations run.
        :rtype: int
        :raises ValueError: If ``max_iter`` is not positive, ``samples`` is not a
            2-D array of the gate's ``input_dim`` width, or it holds fewer rows
            than ``pos_components``.
        """
        return self._fit_exact(samples, 'pos', max_iter, tol)

    def fit_neg(self, samples, max_iter: int = 100, tol: float = 1e-4) -> int:
        """Fit the negative mixture to ``samples`` by exact batch EM.

        :param samples: Tensor of shape ``(n, input_dim)`` — the generic
            reference corpus sample.
        :type samples: Any
        :param max_iter: Iteration cap.
        :type max_iter: int
        :param tol: Relative improvement below which to stop.
        :type tol: float
        :return: The number of iterations run.
        :rtype: int
        :raises ValueError: If ``max_iter`` is not positive, ``samples`` is
            malformed, or it holds fewer rows than ``neg_components``.
        """
        return self._fit_exact(samples, 'neg', max_iter, tol)

    def fit(self, pos_samples, neg_samples, max_iter: int = 100,
            tol: float = 1e-4) -> Dict[str, int]:
        """Fit both mixtures, for a caller holding both samples at once.

        The two are fitted at different times in the reference implementation —
        the negative once at the start of training, the positive again at the end
        of every epoch — which is why they are separate entry points. This is
        the convenience form.

        :param pos_samples: Tensor of shape ``(n, input_dim)``.
        :type pos_samples: Any
        :param neg_samples: Tensor of shape ``(m, input_dim)``.
        :type neg_samples: Any
        :param max_iter: Iteration cap, per mixture.
        :type max_iter: int
        :param tol: Relative improvement below which to stop, per mixture.
        :type tol: float
        :return: ``{'pos': iterations, 'neg': iterations}``.
        :rtype: Dict[str, int]
        """
        return {
            'pos': self.fit_pos(pos_samples, max_iter=max_iter, tol=tol),
            'neg': self.fit_neg(neg_samples, max_iter=max_iter, tol=tol),
        }

    def _fit_exact(self, samples, which: MixtureSide, max_iter: int,
                   tol: float) -> int:
        """Run monotone batch EM until the tolerance or the iteration cap.

        :param samples: Tensor of shape ``(n, input_dim)``.
        :type samples: Any
        :param which: Which mixture to fit.
        :type which: MixtureSide
        :param max_iter: Iteration cap.
        :type max_iter: int
        :param tol: Relative improvement below which to stop.
        :type tol: float
        :return: The number of iterations run.
        :rtype: int
        :raises ValueError: If ``max_iter`` is not positive or the sample is
            malformed.
        """
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")

        z = self._project_sample(samples)
        self._initialise_mixture(z, which)

        previous = None
        iterations = 0
        mass = None
        for i in range(max_iter):
            responsibilities = self._posterior(z, which)
            mass = ops.sum(responsibilities, axis=0)
            self._m_step_from_stats(
                mass,
                ops.matmul(ops.transpose(responsibilities), z),
                ops.matmul(ops.transpose(responsibilities), z * z),
                which,
            )
            iterations = i + 1

            current = float(ops.mean(self._mixture_log_prob(z, which)))
            if previous is not None:
                improvement = current - previous
                if improvement < tol * abs(previous):
                    break
            previous = current

        # Repair collapsed components ONCE, after the iteration rather than
        # inside it. EM's monotone-likelihood guarantee is a property of the
        # ITERATION, and re-seeding a component mid-loop would forfeit it; a
        # single post-hoc pass leaves the converged parameters free of floored
        # components, which is what actually matters. Done before this, the fit
        # would silently run with fewer components than declared and nothing
        # would report it.
        if mass is not None:
            self._revive_dead(which, z, mass)

        logger.info(
            f"LocalSupportGate.fit_{which}: {iterations} EM iterations over "
            f"{int(ops.shape(z)[0])} samples, "
            f"{self._n_components(which)} components"
        )
        return iterations

    # -----------------------------------------------------------------
    # fitting -- streaming minibatch EM
    # -----------------------------------------------------------------

    def observe(self, samples, which: MixtureSide = 'pos') -> int:
        """Advance ``which`` mixture by one stochastic-approximation EM step.

        Accumulates the sufficient statistics `n_k`, `Σ r_k z`, `Σ r_k z²` into an
        EMA with step `1/t`, then takes an **exact M-step from the accumulated
        statistics**. Persistent state is `O(K·d)`, independent of how many
        tokens have been seen, so a gate can be fitted during ordinary training
        with no activation buffer and no layer-sequential sweep.

        Accumulating statistics rather than interpolating parameters is the load
        -bearing detail. Step-sizing the parameters toward each batch's M-step
        converges to "last batch wins"; measured, that variant froze at variance
        0.12 and returned identical parameters after 1 pass and after 60. See
        the module docstring for the full measurement and for the out-of-
        distribution hit-rate cost of this path.

        :param samples: Tensor of shape ``(n, input_dim)`` — one batch of
            activations.
        :type samples: Any
        :param which: Which mixture to advance.
        :type which: MixtureSide
        :return: The number of components re-seeded on this step.
        :rtype: int
        :raises ValueError: If ``samples`` is not a 2-D array of the gate's
            ``input_dim`` width.
        """
        z = self._project_sample(samples)
        if not self._fitted_flags.get(which, False):
            self._initialise_mixture(z, which)

        responsibilities = self._posterior(z, which)
        transpose = ops.transpose(responsibilities)

        batch_stats = (
            ops.sum(responsibilities, axis=0),
            ops.matmul(transpose, z),
            ops.matmul(transpose, z * z),
        )

        step_index = self._weights['em_step_counter'] + 1.0
        self._weights['em_step_counter'].assign(step_index)
        step = 1.0 / step_index

        for stat, batch_value in zip(
            ('mass', 'weighted_sum', 'weighted_sq_sum'), batch_stats
        ):
            key = f'{which}_acc_{stat}'
            self._weights[key].assign(
                (1.0 - step) * self._weights[key] + step * batch_value
            )

        num_dead = self._m_step_from_stats(
            self._weights[f'{which}_acc_mass'],
            self._weights[f'{which}_acc_weighted_sum'],
            self._weights[f'{which}_acc_weighted_sq_sum'],
            which,
        )
        # The streaming path judges by its ACCUMULATOR, which is what the M-step
        # above consumed; passing anything else would repair the wrong set.
        self._revive_dead(which, z, self._weights[f'{which}_acc_mass'], num_dead)
        return num_dead

    def reset(self, which: MixtureSide = 'pos') -> None:
        """Discard ``which`` mixture's fit so it can be rebuilt from scratch.

        Refitting the positive mixture from *updated* representations after each
        epoch is the point: the representations move as the adapter trains, so a
        support fitted before training describes a region the adapter has since
        left. The negative mixture describes a fixed corpus and is fitted once,
        which is why it is not reset alongside the positive.

        :param which: Which mixture to reset.
        :type which: MixtureSide
        """
        n_components = self._n_components(which)
        for stat, shape in (
            ('mass', (n_components,)),
            ('weighted_sum', (n_components, self.effective_dim)),
            ('weighted_sq_sum', (n_components, self.effective_dim)),
        ):
            # dtype taken from a sibling weight, never a bare 'float32'
            # literal: `ops.zeros(shape, 'float32')` pins backend.floatx()
            # whatever the active dtype policy says. Deriving it keeps this
            # package free of never-narrow dtype sites (the rule the guard in
            # `tests/test_layers/test_statistics/
            # test_the_four_carried_defects_are_fixed.py` enforces package-wide).
            reference = self._weights[f'{which}_acc_mass'].dtype
            self._weights[f'{which}_acc_{stat}'].assign(
                ops.zeros(shape, dtype=reference)
            )
        if which == 'pos':
            self._weights['em_step_counter'].assign(
                ops.zeros((), dtype=self._weights['em_step_counter'].dtype)
            )
        self._fitted_flags[which] = False

    # -----------------------------------------------------------------
    # inference
    # -----------------------------------------------------------------

    def call(self, inputs, training=None):
        """Score ``inputs`` and emit the configured decision.

        :param inputs: Tensor of shape ``(..., input_dim)``.
        :type inputs: Any
        :param training: Unused — the gate is non-differentiable by
            construction. Accepted because every Keras sub-layer must forward
            it and because a caller wiring this into a block will pass it.
        :type training: Optional[bool]
        :return: The binary decision (``'hard'``), its causal EMA
            (``'smoothed'``), or the raw log-likelihood ratio (``'scores'``).
            Shape ``(...,)`` when ``output_shape_reduced``, else ``(..., 1)``.
        :rtype: Any
        :raises ValueError: If ``inputs`` has rank < 2, or its last axis does not
            match ``input_dim``.
        """
        x = ops.cast(inputs, 'float32')
        if len(x.shape) < 2:
            raise ValueError(
                f"LocalSupportGate expects rank >= 2 inputs of shape "
                f"(..., {self.input_dim}), got shape {tuple(x.shape)}"
            )
        if x.shape[-1] is not None and x.shape[-1] != self.input_dim:
            raise ValueError(
                f"input width {x.shape[-1]} does not match the gate's "
                f"input_dim={self.input_dim}"
            )

        leading_shape = ops.shape(x)[:-1]
        flat = ops.reshape(x, (-1, ops.shape(x)[-1]))

        if not (self.is_fitted('pos') and self.is_fitted('neg')):
            # Nothing fitted: report everything as out-of-support, which is the
            # safe direction. Opening on an unfitted mixture would route by the
            # arbitrary initialisation.
            scores = ops.zeros(ops.shape(flat)[:-1], 'float32')
        else:
            z = ops.cast(self.projection(flat), 'float32')
            scores = (
                self._mixture_log_prob(z, 'pos')
                - self._mixture_log_prob(z, 'neg')
            )

        # Back to the caller's own shape before smoothing: the EMA needs a token
        # axis, and folding (batch, seq_len) into one axis here would smooth
        # across the batch boundary.
        scores = ops.reshape(scores, leading_shape)

        if self.output_mode == 'scores':
            outputs = scores
        elif self.output_mode == 'hard':
            outputs = ops.cast(scores > self.decision_threshold, 'float32')
        else:
            decision = ops.cast(scores > self.decision_threshold, 'float32')
            outputs = self._causal_ema(decision)

        if not self.output_shape_reduced:
            outputs = ops.expand_dims(outputs, -1)
        return outputs

    def _causal_ema(self, decision):
        """Causal exponential moving average of ``decision`` along the token axis.

        Implements `s_t = α·g_t + (1−α)·s_{t−1}` with `s_0 = 0`, which the
        closed form turns into one matrix product:

            s_t = Σ_{i≤t} α (1−α)^(t−i) · g_i

        The decay matrix is built per call rather than stored, because it depends
        on the runtime sequence length. It is `O(T²)`; against the
        `O(B·T·D·K)` the two mixtures already spend, that is a small price for an
        exact recurrence, which a `while_loop` version would not have under graph
        mode.

        The exponent is clamped at zero *before* the power, so the `i > t` half
        of the matrix evaluates `r⁰` and is then masked away — rather than
        computing `r^negative`, which overflows to infinity for any `r < 1` once
        the sequence is long.

        :param decision: Tensor of the caller's original shape, with a token axis
            at ``smoothing_axis``.
        :type decision: Any
        :return: Tensor of the same shape as ``decision``.
        :rtype: Any
        """
        if self.smoothing_alpha >= 1.0:
            return decision

        rank = len(decision.shape)
        axis = self.smoothing_axis % rank

        permutation = list(range(rank))
        permutation[axis], permutation[-1] = permutation[-1], permutation[axis]
        moved = ops.transpose(decision, permutation)

        seq_len = ops.shape(moved)[0]
        # dtype from the tensor being smoothed, not a bare 'float32' cast
        # around a scalar-built `arange` (which pins backend.floatx()).
        steps = ops.arange(seq_len, dtype=decision.dtype)
        offsets = ops.expand_dims(steps, -1) - ops.expand_dims(steps, -2)
        decay = 1.0 - self.smoothing_alpha
        weights = ops.where(
            offsets >= 0,
            self.smoothing_alpha * ops.power(decay, ops.maximum(offsets, 0.0)),
            ops.zeros_like(offsets),
        )
        smoothed = ops.tensordot(weights, moved, axes=[[1], [0]])

        inverse = list(range(rank))
        inverse[axis], inverse[-1] = inverse[-1], inverse[axis]
        return ops.transpose(smoothed, inverse)

    # -----------------------------------------------------------------
    # serialization
    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument, plus the fitted-state flags.

        The two flags are written here rather than only by
        :meth:`get_saved_state` because ``get_config`` is the dict Keras
        round-trips through ``save``/``load``. Leaving them out would make
        ``from_config`` receive a dict that never carried them, and every
        reloaded gate would report ``is_fitted() == False`` despite its weights
        holding a completed fit — routing every token closed with nothing to
        indicate why.

        :return: Configuration dict suitable for ``from_config``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'projection_dim': self.projection_dim,
            'pos_components': self.pos_components,
            'neg_components': self.neg_components,
            'covariance_type': self.covariance_type,
            'fit_mode': self.fit_mode,
            'smoothing_alpha': self.smoothing_alpha,
            'smoothing_axis': self.smoothing_axis,
            'output_mode': self.output_mode,
            'decision_threshold': self.decision_threshold,
            'variance_floor': self.variance_floor,
            'dead_mass_threshold': self.dead_mass_threshold,
            'seed': self.seed,
            'output_shape_reduced': self.output_shape_reduced,
            'fitted_pos': self.is_fitted('pos'),
            'fitted_neg': self.is_fitted('neg'),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Rebuild a gate, restoring which mixtures were fitted.

        Keras's own ``from_config`` reconstructs the object but knows nothing
        about this layer's non-config state. The two flags are popped here and
        re-applied after construction: ``__init__`` declares neither, and guide
        §6.2 is explicit that a key read out of the config while also being
        forwarded to ``super().__init__()`` is dead on arrival — so they cannot
        become constructor arguments, and must not be passed through either.

        :param config: The serialized configuration dict.
        :type config: Dict[str, Any]
        :return: A gate whose weights are restored by Keras and whose
            ``is_fitted`` flags match what was saved.
        :rtype: LocalSupportGate
        """
        config = dict(config)
        fitted_pos = bool(config.pop('fitted_pos', False))
        fitted_neg = bool(config.pop('fitted_neg', False))
        layer = super().from_config(config)
        layer._fitted_flags['pos'] = fitted_pos
        layer._fitted_flags['neg'] = fitted_neg
        return layer

    def get_saved_state(self) -> Dict[str, Any]:
        """Return the fitted-state flags, for a caller assembling a save dict.

        The same two keys :meth:`get_config` writes, exposed separately for a
        caller storing a gate's state outside a ``.keras`` archive — a plain
        ``.npz`` of the mixture weights, say — where ``get_config`` is not in
        the path.

        :return: ``{'fitted_pos': bool, 'fitted_neg': bool}``.
        :rtype: Dict[str, Any]
        """
        return {
            'fitted_pos': self.is_fitted('pos'),
            'fitted_neg': self.is_fitted('neg'),
        }

    # -----------------------------------------------------------------
    # internals
    # -----------------------------------------------------------------

    def _n_components(self, which: MixtureSide) -> int:
        """Component count of one mixture.

        :param which: Which mixture.
        :type which: MixtureSide
        :return: The configured component count.
        :rtype: int
        """
        return self.pos_components if which == 'pos' else self.neg_components

    def _project_sample(self, samples):
        """Validate a fitting sample, flatten it and project it.

        :param samples: Tensor of shape ``(n, input_dim)``.
        :type samples: Any
        :return: Tensor of shape ``(n, effective_dim)``, float32.
        :rtype: Any
        :raises ValueError: If ``samples`` is not 2-D of width ``input_dim``.
        """
        x = ops.cast(samples, 'float32')
        if len(x.shape) != 2:
            raise ValueError(
                f"LocalSupportGate fitting expects a 2-D "
                f"(n, {self.input_dim}) sample array, got shape "
                f"{tuple(x.shape)}. Flatten a batch of tokens to 2-D first."
            )
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"sample width {x.shape[-1]} does not match the gate's "
                f"input_dim={self.input_dim}"
            )
        return ops.cast(self.projection(x), 'float32')

    def _component_log_prob(self, z, which: MixtureSide):
        """Per-component log-densities of one mixture at ``z``.

        Split out of :meth:`_mixture_log_prob` so both the E-step and the
        likelihood bookkeeping share one definition of the density — three
        hand-copies of this formula would be three chances to drift.

        :param z: Tensor of shape ``(n, effective_dim)``, float32.
        :type z: Any
        :param which: Which mixture to score.
        :type which: MixtureSide
        :return: Tensor of shape ``(n, K)``, float32.
        :rtype: Any
        """
        means = self._weights[f'{which}_means']
        variances = ops.maximum(
            self._weights[f'{which}_variances'], self.variance_floor
        )
        diff = ops.expand_dims(z, 1) - ops.expand_dims(means, 0)  # (n, K, d)
        return -0.5 * ops.sum(
            diff * diff / variances
            + ops.cast(_LOG_2PI + ops.log(variances), 'float32'),
            axis=-1,
        )

    def _mixture_log_prob(self, z, which: MixtureSide):
        """Mixture log-density of ``which`` at ``z``, all components in one op.

        Scoring components in a Python loop turns a `(n, d)` batch into `K`
        separate `(n, d)` evaluations and dominates the gate's inference cost.

        :param z: Tensor of shape ``(n, effective_dim)``, float32.
        :type z: Any
        :param which: Which mixture to score.
        :type which: MixtureSide
        :return: Tensor of shape ``(n,)``, float32.
        :rtype: Any
        """
        per_component = self._component_log_prob(z, which)
        log_priors = self._weights[f'{which}_log_priors']
        return ops.logsumexp(
            per_component + ops.expand_dims(log_priors, 0), axis=-1
        )

    def _posterior(self, z, which: MixtureSide):
        """E-step responsibilities of one mixture at ``z``.

        :param z: Tensor of shape ``(n, effective_dim)``, float32.
        :type z: Any
        :param which: Which mixture's responsibilities to compute.
        :type which: MixtureSide
        :return: Tensor of shape ``(n, K)`` whose rows sum to 1, float32.
        :rtype: Any
        """
        log_joint = (
            self._component_log_prob(z, which)
            + ops.expand_dims(self._weights[f'{which}_log_priors'], 0)
        )
        return ops.exp(log_joint - ops.expand_dims(
            ops.logsumexp(log_joint, axis=-1), -1
        ))

    def _initialise_mixture(self, z, which: MixtureSide) -> None:
        """Seed one mixture's parameters from a sample.

        Components are seeded from `K` *distinct* shuffled rows, each given the
        sample's overall variance. Distinct-row seeding matters: seeding every
        component from the same row starts EM from `K` identical components
        whose responsibilities are permanently symmetric, so they can never
        separate no matter how long the fit runs.

        :param z: Tensor of shape ``(n, effective_dim)``, float32.
        :type z: Any
        :param which: Which mixture to initialise.
        :type which: MixtureSide
        :raises ValueError: If the sample holds fewer rows than the mixture has
            components, or is empty.
        """
        n_components = self._n_components(which)
        n_samples = int(ops.shape(z)[0])
        if n_samples < n_components:
            raise ValueError(
                f"cannot initialise the {which} mixture: it declares "
                f"{n_components} components but the sample holds only "
                f"{n_samples} rows. Lower the component count or supply a "
                f"larger sample."
            )

        order = keras.random.shuffle(
            ops.arange(n_samples, dtype='int32'),
            seed=self._seed_for(f'{which}_init'),
        )
        chosen = ops.take(order, ops.arange(n_components, dtype='int32'), axis=0)
        means = ops.take(z, chosen, axis=0)

        self._weights[f'{which}_means'].assign(means)
        # Every component starts at the sample's overall variance -- shaped
        # (K, d) by the repeat, since `var(z, axis=0)` is (d,) and a bare
        # expand_dims would assign (1, d) into a (K, d) variable and raise.
        batch_variance = ops.repeat(
            ops.expand_dims(ops.var(z, axis=0) + self.variance_floor, 0),
            n_components,
            axis=0,
        )
        self._weights[f'{which}_variances'].assign(batch_variance)
        self._weights[f'{which}_log_priors'].assign(
            ops.full((n_components,), -math.log(n_components), 'float32')
        )
        self._fitted_flags[which] = True

    def _m_step_from_stats(self, mass, weighted_sum, weighted_sq_sum, which):
        """Exact M-step from accumulated sufficient statistics.

        The single place mixture parameters are written. Both fitting paths
        funnel through it, so exact and streaming mode cannot drift in their
        update rule — they differ only in what they accumulate.

        :param mass: Tensor of shape ``(K,)``, accumulated component mass.
        :type mass: Any
        :param weighted_sum: Tensor of shape ``(K, d)``.
        :type weighted_sum: Any
        :param weighted_sq_sum: Tensor of shape ``(K, d)``.
        :type weighted_sq_sum: Any
        :param which: Which mixture to update.
        :type which: MixtureSide
        :return: The number of components whose mass fell below
            ``dead_mass_threshold``.
        :rtype: int
        """
        collapsed = mass < self.dead_mass_threshold
        safe_mass = ops.expand_dims(
            ops.where(collapsed, ops.ones_like(mass), mass), -1
        )

        means = weighted_sum / safe_mass
        # `E[x²] − E[x]²` is a difference of two similar-sized numbers and can
        # go slightly negative for a near-degenerate component. The floor is
        # what makes it safe to take a log of afterwards.
        variances = ops.maximum(
            weighted_sq_sum / safe_mass - means * means, self.variance_floor
        )

        self._weights[f'{which}_means'].assign(means)
        self._weights[f'{which}_variances'].assign(variances)
        self._weights[f'{which}_log_priors'].assign(
            ops.log(mass + self.variance_floor)
            - ops.log(ops.sum(mass) + self.variance_floor)
        )

        return int(ops.sum(ops.cast(collapsed, 'int32')))

    def _revive_dead(
        self,
        which: MixtureSide,
        z,
        mass,
        num_dead: Optional[int] = None,
    ) -> None:
        """Re-seed collapsed components from the batch that revealed them.

        Without this, a component whose mass fell to zero keeps a floored mean
        and variance, so its log-density is a large finite negative number and
        its responsibilities stay pinned at zero forever -- the fit then silently
        runs with fewer components than configured, and nothing reports it.

        ``mass`` is passed in rather than read from the accumulators, because the
        two fitting paths disagree about which mass the M-step consumed: the
        streaming path uses its EMA accumulator, while exact EM uses the batch's
        own mass and never touches the accumulator at all. Reading the
        accumulator unconditionally would therefore judge every component
        collapsed on the exact path -- and revive the entire mixture on every
        iteration.

        Placement is done with a cumulative rank and a gather rather than a
        scatter: ``keras.ops`` in 3.8 has no ``scatter_nd_update``, and building
        the index array with ``ops.where`` + ``cumsum`` keeps the whole repair in
        one vectorised pass with no Python-side loop over components.

        :param which: Which mixture to repair.
        :type which: MixtureSide
        :param z: The batch, the source of replacement means.
        :type z: Any
        :param mass: The component mass the M-step that just ran was given.
        :type mass: Any
        :param num_dead: A count already computed by the caller, to avoid
            re-evaluating the mask. ``None`` recomputes it.
        :type num_dead: Optional[int]
        """
        collapsed = mass < self.dead_mass_threshold
        if num_dead is None:
            num_dead = int(ops.sum(ops.cast(collapsed, 'int32')))
        if num_dead == 0:
            return

        n_samples = int(ops.shape(z)[0])
        rows = keras.random.randint(
            shape=(num_dead,), minval=0, maxval=n_samples, seed=self._seed_for(
                f'{which}_revive'
            ),
        )
        replacement = ops.take(z, rows, axis=0)  # (num_dead, d)

        # rank among the collapsed components: -1 for every live one, so
        # maximum(., 0) keeps the gather index in range and `where` discards it.
        ranks = ops.cumsum(ops.cast(collapsed, 'int32')) - 1
        gathered = ops.take(replacement, ops.maximum(ranks, 0), axis=0)

        mask = ops.expand_dims(collapsed, -1)
        self._weights[f'{which}_means'].assign(
            ops.where(mask, gathered, self._weights[f'{which}_means'])
        )
        # (1, d), NOT (num_dead, d): `ops.where(mask, …, variances)` broadcasts
        # this against the (K, d) live variances, and repeating it to
        # (num_dead, d) only works when num_dead == K. A PARTIAL revival
        # (0 < num_dead < K — the ordinary case) raised a broadcast error
        # instead of repairing anything.
        batch_variance = ops.expand_dims(
            ops.var(z, axis=0) + self.variance_floor, 0
        )
        self._weights[f'{which}_variances'].assign(
            ops.where(
                mask,
                batch_variance,
                self._weights[f'{which}_variances'],
            )
        )
        # A revived component must not keep its floored log-prior either.
        self._weights[f'{which}_log_priors'].assign(
            ops.where(
                collapsed,
                ops.full_like(mass, -math.log(int(ops.shape(mass)[0]))),
                self._weights[f'{which}_log_priors'],
            )
        )

        self._num_revivals += num_dead
        logger.warning(
            f"LocalSupportGate: re-seeded {num_dead} collapsed {which} "
            f"component(s) from the current batch. The mixture declares more "
            f"components than the sample can support; the fit completes with "
            f"fewer effective components than configured."
        )

    def _seed_for(self, purpose: str) -> Optional[int]:
        """Derive a reproducible per-purpose seed from ``seed``.

        Distinct purposes need distinct draws — otherwise the initialisation
        shuffle and a revival draw would consume the same numbers and produce a
        correlated result. ``None`` propagates, which is the documented
        "unspecified" case.

        :param purpose: A short label distinguishing the draw sites.
        :type purpose: str
        :return: The derived seed, or ``None`` when ``seed`` is ``None``.
        :rtype: Optional[int]
        """
        if self.seed is None:
            return None
        # A cheap stable scramble; not a hash, because it must agree across
        # processes and Python's string hash is salted per process.
        return (self.seed * 1_000_003 + sum(ord(c) * (i + 1) for i, c in enumerate(purpose))) % (2**31 - 1)


def _sketch_initializer(seed: Optional[int], projection_dim: int):
    """Build the projection's initializer: Gaussian, reproducible when seeded.

    A JL sketch needs a fixed random matrix, and the standard construction uses
    entries of scale `1/sqrt(k)` so the map approximately preserves distances.
    ``'glorot_uniform'`` is drawn from the global RNG, so two gates built without
    an explicit seed would draw different sketches and gate differently on
    identical data — which makes the gate untestable and the layer
    unreproducible. With a seed the initializer is recreated identically.

    :param seed: Seed for the sketch, or ``None`` to draw from the global RNG.
    :type seed: Optional[int]
    :param projection_dim: The sketch's target width, fixing the entry scale.
    :type projection_dim: int
    :return: A Keras initializer producing an `(input_dim, projection_dim)`
        matrix.
    :rtype: Any
    """
    return keras.initializers.RandomNormal(
        mean=0.0,
        stddev=1.0 / math.sqrt(projection_dim),
        seed=seed,
    )
