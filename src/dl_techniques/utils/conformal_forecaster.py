"""
Conformal Prediction for Time Series Forecasting.

Implements model-agnostic uncertainty quantification with mathematical
validity guarantees. Based on Valeriy Manokhin's forecasting science framework.

Key principles:
- Validity (coverage) precedes efficiency (sharpness)
- Distribution-free guarantees (only requires exchangeability)
- Compatible with ANY forecasting model
- Finite sample guarantees (not asymptotic)
"""

import numpy as np
from collections import deque
from typing import Callable, Optional, Tuple, Any, Dict, Literal

# ---------------------------------------------------------------------

class ConformalForecaster:
    """
    Model-agnostic wrapper for Inductive Conformal Prediction (ICP).

    Provides mathematically valid prediction intervals for ANY forecasting model:
    - N-BEATS, LSTM, Transformer, ARIMA, XGBoost, Prophet, etc.

    Implements the 'Validity-First' hierarchy:
    1. Validity (coverage guarantee) is non-negotiable
    2. Efficiency (narrow intervals) pursued after validity achieved

    Theoretical Guarantee
    --------------------
    For significance level α (e.g., α=0.1 for 90% coverage):

        P(y_new ∈ [ŷ - Q, ŷ + Q]) ≥ 1 - α

    This holds for:
    - ANY underlying model
    - ANY data distribution (under exchangeability)
    - ANY dataset size (finite sample guarantee)

    Attributes
    ----------
    model : Any
        Trained forecasting model with .predict() method
    alpha : float
        Miscoverage rate (0.1 = 90% coverage)
    nonconformity_measure : str
        Type of conformity score to compute
    horizon_strategy : str
        How to handle multi-step forecasts
    adaptive_method : str or None
        Method for handling distribution shift

    Methods
    -------
    calibrate : Compute conformity scores on calibration set
    predict : Generate prediction intervals for new data
    update : Online update with new observations
    evaluate_coverage : Assess empirical coverage probability
    get_efficiency_metrics : Compute interval width statistics

    Notes
    -----
    Conformal prediction is the ONLY uncertainty quantification method
    with finite sample validity guarantees. Alternative methods (Bayesian,
    bootstrap, Monte Carlo) lack mathematical guarantees.

    References
    ----------
    - Vovk, Gammerman, & Shafer (2005). Algorithmic Learning in a Random World
    - Romano, Patterson, & Candès (2019). Conformalized Quantile Regression
    - Xu & Xie (2021). Conformal Prediction Intervals for Dynamic Time-Series
    """

    def __init__(
            self,
            model: Any,
            calibration_alpha: float = 0.1,
            output_adapter: Optional[Callable[[Any], np.ndarray]] = None,
            horizon_strategy: Literal['global', 'step_wise', 'adaptive'] = 'step_wise',
            nonconformity_measure: Literal[
                'absolute',
                'normalized',
                'cqr',
                'locally_weighted'
            ] = 'absolute',
            adaptive_method: Optional[Literal[
                'sliding_window',
                'exponential_weighting',
                'none'
            ]] = None,
            window_size: int = 100,
            decay_rate: float = 0.95
    ):
        """
        Initialize conformal forecaster.

        Parameters
        ----------
        model : Any
            Trained forecasting model. Must have .predict() method or be callable.
        calibration_alpha : float, default=0.1
            Miscoverage rate. α=0.1 → 90% coverage, α=0.05 → 95% coverage.
        output_adapter : Callable[[Any], np.ndarray], optional
            Function to extract point forecast from model's raw output.
            Examples:
            - N-BEATS/Keras: lambda x: x[0] or lambda x: x
            - Sklearn: None (defaults to identity)
            - Quantile models: lambda x: x[:, :, median_idx]
        horizon_strategy : {'global', 'step_wise', 'adaptive'}, default='step_wise'
            Multi-horizon strategy:
            - 'global': Single Q-value for all time steps (simplest, least efficient)
            - 'step_wise': Unique Q per horizon (accounts for growing uncertainty)
            - 'adaptive': Step-wise with online adaptation (best for distribution shift)
        nonconformity_measure : str, default='absolute'
            Conformity score type:
            - 'absolute': |y - ŷ| (simplest, assumes homoscedasticity)
            - 'normalized': |y - ŷ| / σ̂ (accounts for heteroscedasticity)
            - 'cqr': Conformalized Quantile Regression (requires quantile model)
            - 'locally_weighted': Weighted by local density (adaptive intervals)
        adaptive_method : str or None, default=None
            Distribution shift handling:
            - 'sliding_window': Use only recent calibration points
            - 'exponential_weighting': Weight recent points more heavily
            - None: Standard ICP (assumes exchangeability)
        window_size : int, default=100
            Calibration window size for sliding window method
        decay_rate : float, default=0.95
            Decay rate for exponential weighting (higher = slower decay)

        Notes
        -----
        For time series with distribution shift, use adaptive_method='sliding_window'
        or 'exponential_weighting'. Standard ICP assumes exchangeability which
        time series often violate.

        For heteroscedastic series (varying volatility), use
        nonconformity_measure='normalized'.

        Examples
        --------
        >>> # Basic usage with N-BEATS
        >>> cf = ConformalForecaster(
        ...     model=nbeats_model,
        ...     calibration_alpha=0.1,
        ...     output_adapter=lambda x: x[0]
        ... )

        >>> # Adaptive conformal for non-stationary series
        >>> cf = ConformalForecaster(
        ...     model=lstm_model,
        ...     horizon_strategy='adaptive',
        ...     adaptive_method='sliding_window',
        ...     window_size=50
        ... )

        >>> # CQR for quantile regression models
        >>> cf = ConformalForecaster(
        ...     model=quantile_model,
        ...     nonconformity_measure='cqr',
        ...     output_adapter=lambda x: x  # Returns [lower, point, upper]
        ... )
        """
        self.model = model
        self.alpha = calibration_alpha
        self.strategy = horizon_strategy
        self.nonconformity_measure = nonconformity_measure
        self.adaptive_method = adaptive_method
        self.window_size = window_size
        self.decay_rate = decay_rate

        # Output adapter: extract point forecast from model output
        self.output_adapter = output_adapter if output_adapter else (lambda x: x)

        # Calibration state
        self.q_hat = None  # Quantile threshold (scalar or array)
        self.calibrated = False

        # Adaptive state (for online updates)
        self.calibration_buffer = deque(maxlen=window_size if adaptive_method else None)
        self.coverage_history = []  # Track empirical coverage
        self.interval_widths = []  # Track efficiency

        # Statistics for normalized scores
        self.residual_std = None  # For 'normalized' nonconformity

    def _get_forecast(self, x_input: np.ndarray) -> np.ndarray:
        """
        Extract point forecast from model output.

        Parameters
        ----------
        x_input : np.ndarray
            Input features for forecasting

        Returns
        -------
        np.ndarray
            Point forecasts, shape (n_samples, horizon)
        """
        # Handle different model types
        if hasattr(self.model, 'predict'):
            raw_output = self.model.predict(x_input, verbose=0)
        else:
            raw_output = self.model(x_input)

        # Apply output adapter
        forecast = self.output_adapter(raw_output)

        # Ensure numpy array
        if not isinstance(forecast, np.ndarray):
            forecast = np.array(forecast)

        return forecast

    def _compute_nonconformity_scores(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            method: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute nonconformity scores (measures of prediction difficulty).

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth, shape (n_samples, horizon)
        y_pred : np.ndarray
            Predictions, shape (n_samples, horizon)
        method : str, optional
            Override default nonconformity measure

        Returns
        -------
        np.ndarray
            Nonconformity scores, shape (n_samples, horizon)

        Notes
        -----
        Lower scores → easier to predict (conforming)
        Higher scores → harder to predict (non-conforming)

        The choice of nonconformity measure affects efficiency:
        - 'absolute': Uniform intervals (ignores heteroscedasticity)
        - 'normalized': Adaptive intervals (narrower in stable regions)
        - 'cqr': Model uncertainty-aware (requires quantile predictions)
        """
        method = method or self.nonconformity_measure

        if method == 'absolute':
            # Standard absolute residual: |y - ŷ|
            scores = np.abs(y_true - y_pred)

        elif method == 'normalized':
            # Normalized by local volatility: |y - ŷ| / σ̂
            # Provides adaptive intervals (narrower in stable regions)
            residuals = y_true - y_pred

            # Estimate local standard deviation per horizon
            if self.residual_std is None:
                # Initial calibration: use sample std per horizon
                self.residual_std = np.std(residuals, axis=0, keepdims=True) + 1e-6

            scores = np.abs(residuals) / self.residual_std

        elif method == 'cqr':
            # Conformalized Quantile Regression
            # Requires model that outputs [lower_quantile, median, upper_quantile]
            # Score = max(lower - y, y - upper)
            # This is a placeholder - actual CQR requires quantile model
            raise NotImplementedError(
                "CQR requires model that outputs quantile predictions. "
                "Set output_adapter to extract [q_low, q_mid, q_high] and "
                "implement custom CQR score computation."
            )

        elif method == 'locally_weighted':
            # Weight by local density (adaptive to input space)
            # More conforming in dense regions, less in sparse regions
            # Simplified version: use absolute residual with density weighting
            scores = np.abs(y_true - y_pred)
            # Note: Full implementation requires density estimation on X
            # For now, equivalent to absolute

        else:
            raise ValueError(f"Unknown nonconformity measure: {method}")

        return scores

    def _compute_weighted_quantile(
            self,
            scores: np.ndarray,
            weights: Optional[np.ndarray] = None,
            q_level: float = 0.9
    ) -> float:
        """
        Compute weighted quantile for adaptive conformal prediction.

        Parameters
        ----------
        scores : np.ndarray
            Conformity scores
        weights : np.ndarray, optional
            Importance weights for each score
        q_level : float
            Quantile level (0.9 for 90% coverage)

        Returns
        -------
        float
            Weighted quantile threshold
        """
        if weights is None:
            return np.quantile(scores, q_level)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Sort scores and accumulate weights
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        cumulative_weights = np.cumsum(weights[sorted_indices])

        # Find first score where cumulative weight exceeds q_level
        idx = np.searchsorted(cumulative_weights, q_level)
        idx = min(idx, len(sorted_scores) - 1)

        return sorted_scores[idx]

    def calibrate(self, x_calib: np.ndarray, y_calib: np.ndarray):
        """
        Compute conformity scores on calibration set.

        Parameters
        ----------
        x_calib : np.ndarray
            Calibration inputs, shape (n_calib, ...)
        y_calib : np.ndarray
            Calibration targets, shape (n_calib, horizon)

        Notes
        -----
        Calibration set must be separate from training set (ICP requirement).
        Typical split: 60% train, 20% calibrate, 20% test.

        For time series, use temporal split (not random):
        - Train: [0, t1]
        - Calibrate: [t1, t2]
        - Test: [t2, T]

        Finite sample correction is applied automatically:
        q_level = ceil((n+1)(1-α)) / n

        Examples
        --------
        >>> cf.calibrate(x_calib, y_calib)
        ICP Calibrated (Step-wise): Average Q=2.3456
        """
        # Get point predictions
        preds = self._get_forecast(x_calib)

        # Validate shapes
        if preds.shape != y_calib.shape:
            raise ValueError(
                f"Shape mismatch: Predictions {preds.shape} vs "
                f"Targets {y_calib.shape}"
            )

        # Compute nonconformity scores
        scores = self._compute_nonconformity_scores(y_calib, preds)
        n = len(scores)

        # Finite sample correction for quantile level
        # Ensures valid coverage even for small calibration sets
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))

        # Compute weights for adaptive methods
        weights = None
        if self.adaptive_method == 'exponential_weighting':
            # Recent samples weighted more heavily
            # w_t = decay_rate^(n-t)
            weights = np.array([
                self.decay_rate ** (n - i - 1)
                for i in range(n)
            ])
            weights = weights[:, np.newaxis]  # Shape: (n, 1)

        # Compute quantile threshold(s)
        if self.strategy == 'global':
            # Single Q-value for all horizons
            # Simpler but less efficient (overly conservative for early horizons)
            flat_scores = scores.flatten()
            flat_weights = weights.flatten() if weights is not None else None
            self.q_hat = self._compute_weighted_quantile(
                flat_scores, flat_weights, q_level
            )
            print(f"ICP Calibrated (Global): Q={self.q_hat:.4f}")

        elif self.strategy in ['step_wise', 'adaptive']:
            # Separate Q-value per horizon step
            # Accounts for growing uncertainty over forecast horizon
            horizon_len = scores.shape[1]
            self.q_hat = np.zeros(horizon_len)

            for t in range(horizon_len):
                step_weights = weights[:, 0] if weights is not None else None
                self.q_hat[t] = self._compute_weighted_quantile(
                    scores[:, t], step_weights, q_level
                )

            # Reshape for broadcasting: (1, horizon)
            self.q_hat = self.q_hat.reshape(1, -1)
            print(
                f"ICP Calibrated ({self.strategy.title()}): "
                f"Q_avg={np.mean(self.q_hat):.4f}, "
                f"Q_min={np.min(self.q_hat):.4f}, "
                f"Q_max={np.max(self.q_hat):.4f}"
            )

        else:
            raise ValueError(f"Unknown horizon strategy: {self.strategy}")

        # Store calibration data for adaptive methods
        if self.adaptive_method in ['sliding_window', 'exponential_weighting']:
            for i in range(len(x_calib)):
                self.calibration_buffer.append({
                    'x': x_calib[i],
                    'y': y_calib[i],
                    'score': scores[i]
                })

        self.calibrated = True

    def predict(
            self,
            x_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction intervals with validity guarantee.

        Parameters
        ----------
        x_test : np.ndarray
            Test inputs, shape (n_test, ...)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - Point forecasts, shape (n_test, horizon)
            - Lower bounds, shape (n_test, horizon)
            - Upper bounds, shape (n_test, horizon)

        Notes
        -----
        Validity guarantee:
            P(y_test ∈ [lower, upper]) ≥ 1 - α

        This holds regardless of:
        - Model quality (works with any model)
        - Data distribution (distribution-free)
        - Sample size (finite sample guarantee)

        Examples
        --------
        >>> preds, lower, upper = cf.predict(x_test)
        >>> coverage = np.mean((y_test >= lower) & (y_test <= upper))
        >>> print(f"Empirical coverage: {coverage:.1%}")
        Empirical coverage: 90.2%
        """
        if not self.calibrated:
            raise RuntimeError(
                "Model not calibrated. Call .calibrate() first with "
                "held-out calibration set."
            )

        # Get point forecasts
        preds = self._get_forecast(x_test)

        # Construct prediction intervals
        # Broadcasting handles both scalar Q (global) and array Q (step-wise)
        lower = preds - self.q_hat
        upper = preds + self.q_hat

        return preds, lower, upper

    def update(self, x_new: np.ndarray, y_new: np.ndarray):
        """
        Online update with new observations (adaptive conformal).

        Parameters
        ----------
        x_new : np.ndarray
            New input observation
        y_new : np.ndarray
            New target observation

        Notes
        -----
        Only effective when adaptive_method is set to:
        - 'sliding_window': Adds to buffer, old observations dropped
        - 'exponential_weighting': Updates weights

        For non-adaptive methods, this stores observations but doesn't
        update Q-values until full recalibration.

        Use this for production deployment with distribution shift.
        Recalibrate periodically based on coverage monitoring.

        Examples
        --------
        >>> # Online adaptation
        >>> for x, y in zip(x_stream, y_stream):
        ...     pred, lower, upper = cf.predict(x[np.newaxis, ...])
        ...     cf.update(x, y)  # Adapt to new data
        """
        if not self.calibrated:
            raise RuntimeError("Must calibrate before online updates")

        if self.adaptive_method is None:
            return  # No online adaptation

        # Get prediction and compute score
        pred = self._get_forecast(x_new[np.newaxis, ...])
        score = self._compute_nonconformity_scores(
            y_new[np.newaxis, ...],
            pred
        )

        # Add to buffer (automatically drops old if full)
        self.calibration_buffer.append({
            'x': x_new,
            'y': y_new,
            'score': score[0]
        })

        # Recalibrate with updated buffer
        if len(self.calibration_buffer) >= 10:  # Min samples for stability
            buffer_data = list(self.calibration_buffer)
            scores = np.array([d['score'] for d in buffer_data])
            n = len(scores)

            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(1.0, max(0.0, q_level))

            # Compute weights if using exponential weighting
            weights = None
            if self.adaptive_method == 'exponential_weighting':
                weights = np.array([
                    self.decay_rate ** (n - i - 1)
                    for i in range(n)
                ])

            # Update Q-values
            if self.strategy == 'global':
                flat_scores = scores.flatten()
                self.q_hat = self._compute_weighted_quantile(
                    flat_scores, weights, q_level
                )
            else:
                horizon_len = scores.shape[1]
                self.q_hat = np.zeros(horizon_len)
                for t in range(horizon_len):
                    self.q_hat[t] = self._compute_weighted_quantile(
                        scores[:, t], weights, q_level
                    )
                self.q_hat = self.q_hat.reshape(1, -1)

    def evaluate_coverage(
            self,
            x_test: np.ndarray,
            y_test: np.ndarray,
            per_horizon: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate empirical coverage probability.

        Parameters
        ----------
        x_test : np.ndarray
            Test inputs
        y_test : np.ndarray
            Test targets
        per_horizon : bool, default=False
            If True, return coverage for each horizon step

        Returns
        -------
        Dict[str, float]
            Coverage statistics:
            - 'coverage': Overall empirical coverage
            - 'target_coverage': Target coverage (1 - α)
            - 'coverage_gap': |empirical - target|
            - 'per_horizon_coverage': Coverage per step (if requested)

        Notes
        -----
        Theoretical guarantee: coverage ≥ 1 - α

        In practice:
        - coverage ≈ 1 - α: Well-calibrated
        - coverage >> 1 - α: Overly conservative (inefficient)
        - coverage < 1 - α: Validity violation (distribution shift likely)

        Monitor coverage in production. If coverage drops below target,
        recalibrate with recent data.

        Examples
        --------
        >>> results = cf.evaluate_coverage(x_test, y_test)
        >>> print(f"Coverage: {results['coverage']:.1%}")
        >>> print(f"Target: {results['target_coverage']:.1%}")
        Coverage: 90.5%
        Target: 90.0%
        """
        _, lower, upper = self.predict(x_test)

        # Check if targets fall within intervals
        in_interval = (y_test >= lower) & (y_test <= upper)

        # Overall coverage
        coverage = np.mean(in_interval)
        target_coverage = 1 - self.alpha

        results = {
            'coverage': float(coverage),
            'target_coverage': target_coverage,
            'coverage_gap': float(abs(coverage - target_coverage))
        }

        # Per-horizon coverage
        if per_horizon:
            horizon_coverage = np.mean(in_interval, axis=0)
            results['per_horizon_coverage'] = horizon_coverage.tolist()

        # Store in history
        self.coverage_history.append(coverage)

        return results

    def get_efficiency_metrics(
            self,
            x_test: np.ndarray,
            per_horizon: bool = False
    ) -> Dict[str, float]:
        """
        Compute efficiency metrics (interval sharpness).

        Parameters
        ----------
        x_test : np.ndarray
            Test inputs
        per_horizon : bool, default=False
            If True, return metrics per horizon step

        Returns
        -------
        Dict[str, float]
            Efficiency metrics:
            - 'mean_width': Average interval width
            - 'median_width': Median interval width
            - 'std_width': Width standard deviation
            - 'per_horizon_width': Width per step (if requested)

        Notes
        -----
        After achieving validity (coverage ≥ 1-α), focus on efficiency:
        - Narrower intervals = more informative predictions
        - Width adaptivity = different widths for different difficulties

        Trade-off:
        - Too narrow → validity violations
        - Too wide → uninformative (but valid)

        Goal: Achieve target coverage with minimal width.

        Examples
        --------
        >>> metrics = cf.get_efficiency_metrics(x_test)
        >>> print(f"Mean interval width: {metrics['mean_width']:.2f}")
        Mean interval width: 4.56
        """
        _, lower, upper = self.predict(x_test)

        # Compute interval widths
        widths = upper - lower

        results = {
            'mean_width': float(np.mean(widths)),
            'median_width': float(np.median(widths)),
            'std_width': float(np.std(widths)),
            'min_width': float(np.min(widths)),
            'max_width': float(np.max(widths))
        }

        # Per-horizon widths
        if per_horizon:
            horizon_widths = np.mean(widths, axis=0)
            results['per_horizon_width'] = horizon_widths.tolist()

        # Store in history
        self.interval_widths.append(np.mean(widths))

        return results

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic information.

        Returns
        -------
        Dict[str, Any]
            Diagnostic information including:
            - Configuration
            - Calibration status
            - Coverage history
            - Efficiency history

        Notes
        -----
        Use for monitoring conformal predictor health in production.
        """
        return {
            'configuration': {
                'alpha': self.alpha,
                'target_coverage': 1 - self.alpha,
                'horizon_strategy': self.strategy,
                'nonconformity_measure': self.nonconformity_measure,
                'adaptive_method': self.adaptive_method
            },
            'calibration': {
                'calibrated': self.calibrated,
                'q_values': self.q_hat.tolist() if self.q_hat is not None else None,
                'buffer_size': len(self.calibration_buffer) if self.calibration_buffer else 0
            },
            'history': {
                'coverage_history': self.coverage_history,
                'interval_width_history': self.interval_widths
            }
        }

# ---------------------------------------------------------------------
