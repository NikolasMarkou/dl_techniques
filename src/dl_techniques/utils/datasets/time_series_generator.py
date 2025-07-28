"""
Generic Time Series Generator for Deep Learning Experiments

This module provides a comprehensive time series generator that creates diverse
patterns including trend, seasonal, stochastic, and composite time series for
machine learning and forecasting experiments.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@dataclass
class TimeSeriesConfig:
    """Configuration class for time series generation.

    This configuration class contains all parameters needed for generating
    diverse time series patterns.

    Attributes:
        n_samples: Number of time steps in each generated series.
        random_seed: Random seed for reproducible generation.
        default_noise_level: Default noise level for all series types.

        # Trend parameters
        trend_strengths: Range of trend strengths to sample from.

        # Seasonal parameters
        seasonal_periods: Available seasonal periods for generation.
        seasonal_amplitudes: Range of seasonal amplitudes to sample from.

        # Stochastic parameters
        ar_coeffs_range: Range for AR coefficients.
        ma_coeffs_range: Range for MA coefficients.
        volatility_range: Range for volatility parameters.

        # Advanced pattern parameters
        outlier_prob_range: Range for outlier probabilities.
        regime_switch_prob_range: Range for regime switching probabilities.
        structural_break_points: Possible structural break locations as fractions.
    """

    # Basic configuration
    n_samples: int = 1000
    random_seed: int = 42
    default_noise_level: float = 0.1

    # Trend parameters
    trend_strengths: Tuple[float, float] = (0.0001, 0.005)

    # Seasonal parameters
    seasonal_periods: List[int] = field(default_factory=lambda: [12, 24, 48, 96, 168, 336])
    seasonal_amplitudes: Tuple[float, float] = (0.5, 2.0)

    # Stochastic parameters
    ar_coeffs_range: Tuple[float, float] = (-0.8, 0.8)
    ma_coeffs_range: Tuple[float, float] = (-0.8, 0.8)
    volatility_range: Tuple[float, float] = (0.01, 0.3)

    # Advanced pattern parameters
    outlier_prob_range: Tuple[float, float] = (0.01, 0.1)
    regime_switch_prob_range: Tuple[float, float] = (0.01, 0.05)
    structural_break_points: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])

# ---------------------------------------------------------------------

class TimeSeriesGenerator:
    """Generator for diverse time series patterns for machine learning experiments.

    This class generates a comprehensive set of time series patterns including
    trend, seasonal, stochastic, and composite patterns suitable for various
    forecasting and time series analysis tasks.

    Args:
        config: Configuration object containing generation settings.

    Attributes:
        config: Configuration object.
        task_definitions: Dictionary defining all available time series tasks.
        random_state: Random state for reproducible generation.

    Example:
        >>> config = TimeSeriesConfig(n_samples=500, random_seed=42)
        >>> generator = TimeSeriesGenerator(config)
        >>>
        >>> # Generate a specific pattern
        >>> trend_series = generator.generate_task_data("linear_trend_strong")
        >>>
        >>> # Generate multiple patterns
        >>> all_patterns = generator.generate_all_patterns()
        >>>
        >>> # Get pattern categories
        >>> categories = generator.get_task_categories()
    """

    def __init__(self, config: TimeSeriesConfig) -> None:
        self.config = config
        self.task_definitions = self._define_tasks()
        self.random_state = np.random.RandomState(config.random_seed)

        logger.info(
            f"Initialized time series generator with {len(self.task_definitions)} tasks, "
            f"n_samples={config.n_samples}"
        )

    def _define_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Define comprehensive set of time series generation tasks.

        Returns:
            Dictionary mapping task names to their definitions including
            category, generator function, and parameters.
        """
        tasks = {}

        # === TREND PATTERNS ===
        tasks["linear_trend_strong"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {
                "trend_type": "linear",
                "strength": 0.002,
                "noise_level": 0.05
            }
        }
        tasks["linear_trend_weak"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {
                "trend_type": "linear",
                "strength": 0.0005,
                "noise_level": 0.1
            }
        }
        tasks["exponential_growth"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {
                "trend_type": "exponential",
                "strength": 0.0001,
                "noise_level": 0.08
            }
        }
        tasks["polynomial_trend"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {
                "trend_type": "polynomial",
                "coefficients": [0, 0.001, -2e-7],
                "noise_level": 0.06
            }
        }
        tasks["logistic_growth"] = {
            "category": "trend",
            "generator": self._generate_logistic_growth,
            "params": {
                "carrying_capacity": 10,
                "growth_rate": 0.01,
                "noise_level": 0.1
            }
        }

        # === SEASONAL PATTERNS ===
        tasks["daily_seasonality"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {
                "periods": [24],
                "amplitudes": [1.0],
                "noise_level": 0.08
            }
        }
        tasks["weekly_seasonality"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {
                "periods": [168],
                "amplitudes": [1.2],
                "noise_level": 0.06
            }
        }
        tasks["multi_seasonal"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {
                "periods": [24, 168],
                "amplitudes": [1.0, 0.8],
                "noise_level": 0.1
            }
        }
        tasks["complex_seasonal"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {
                "periods": [12, 24, 168],
                "amplitudes": [0.6, 1.0, 0.7],
                "noise_level": 0.12
            }
        }

        # === TREND + SEASONAL COMBINATIONS ===
        tasks["trend_daily_seasonal"] = {
            "category": "composite",
            "generator": self._generate_trend_seasonal,
            "params": {
                "trend_type": "linear",
                "trend_strength": 0.001,
                "periods": [24],
                "seasonal_amplitudes": [1.0],
                "noise_level": 0.08
            }
        }
        tasks["trend_weekly_seasonal"] = {
            "category": "composite",
            "generator": self._generate_trend_seasonal,
            "params": {
                "trend_type": "linear",
                "trend_strength": 0.0008,
                "periods": [168],
                "seasonal_amplitudes": [1.2],
                "noise_level": 0.1
            }
        }
        tasks["exp_trend_multi_seasonal"] = {
            "category": "composite",
            "generator": self._generate_trend_seasonal,
            "params": {
                "trend_type": "exponential",
                "trend_strength": 0.0001,
                "periods": [24, 168],
                "seasonal_amplitudes": [1.0, 0.6],
                "noise_level": 0.12
            }
        }

        # === STOCHASTIC PROCESSES ===
        tasks["random_walk"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {
                "process_type": "random_walk",
                "drift": 0.001,
                "volatility": 0.05
            }
        }
        tasks["ar_process"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {
                "process_type": "ar",
                "ar_coeffs": [0.7, -0.2],
                "noise_std": 0.1
            }
        }
        tasks["ma_process"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {
                "process_type": "ma",
                "ma_coeffs": [0.8, 0.3],
                "noise_std": 0.1
            }
        }
        tasks["arma_process"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {
                "process_type": "arma",
                "ar_coeffs": [0.6],
                "ma_coeffs": [0.4],
                "noise_std": 0.08
            }
        }
        tasks["mean_reverting"] = {
            "category": "stochastic",
            "generator": self._generate_mean_reverting,
            "params": {
                "theta": 0.05,
                "mu": 0,
                "sigma": 0.2
            }
        }

        # === INTERMITTENT PATTERNS ===
        tasks["intermittent_demand"] = {
            "category": "intermittent",
            "generator": self._generate_intermittent_series,
            "params": {
                "demand_prob": 0.3,
                "demand_mean": 2.0,
                "demand_std": 0.5
            }
        }
        tasks["lumpy_demand"] = {
            "category": "intermittent",
            "generator": self._generate_intermittent_series,
            "params": {
                "demand_prob": 0.1,
                "demand_mean": 5.0,
                "demand_std": 1.0
            }
        }

        # === VOLATILITY CLUSTERING ===
        tasks["garch_low_vol"] = {
            "category": "volatility",
            "generator": self._generate_garch_series,
            "params": {
                "alpha": 0.1,
                "beta": 0.8,
                "omega": 0.01
            }
        }
        tasks["garch_high_vol"] = {
            "category": "volatility",
            "generator": self._generate_garch_series,
            "params": {
                "alpha": 0.2,
                "beta": 0.7,
                "omega": 0.05
            }
        }

        # === REGIME SWITCHING ===
        tasks["regime_switching"] = {
            "category": "regime",
            "generator": self._generate_regime_switching,
            "params": {
                "regimes": 2,
                "switch_prob": 0.02,
                "regime_params": [(0.001, 0.05), (0.005, 0.15)]
            }
        }

        # === STRUCTURAL BREAKS ===
        tasks["level_shift"] = {
            "category": "structural",
            "generator": self._generate_structural_break,
            "params": {
                "break_type": "level",
                "break_magnitude": 2.0,
                "break_points": [0.5]
            }
        }
        tasks["trend_change"] = {
            "category": "structural",
            "generator": self._generate_structural_break,
            "params": {
                "break_type": "trend",
                "break_magnitude": 0.001,
                "break_points": [0.4, 0.7]
            }
        }

        # === OUTLIER PATTERNS ===
        tasks["additive_outliers"] = {
            "category": "outliers",
            "generator": self._generate_outlier_series,
            "params": {
                "outlier_type": "additive",
                "outlier_prob": 0.05,
                "outlier_magnitude": 3.0
            }
        }
        tasks["innovation_outliers"] = {
            "category": "outliers",
            "generator": self._generate_outlier_series,
            "params": {
                "outlier_type": "innovation",
                "outlier_prob": 0.03,
                "outlier_magnitude": 2.0
            }
        }

        # === CHAOTIC PATTERNS ===
        tasks["henon_map"] = {
            "category": "chaotic",
            "generator": self._generate_chaotic_series,
            "params": {
                "system": "henon",
                "a": 1.4,
                "b": 0.3
            }
        }
        tasks["lorenz_x"] = {
            "category": "chaotic",
            "generator": self._generate_chaotic_series,
            "params": {
                "system": "lorenz",
                "component": "x",
                "sigma": 10,
                "rho": 28,
                "beta": 8 / 3
            }
        }

        return tasks

    def get_task_names(self) -> List[str]:
        """Get list of all available task names.

        Returns:
            List of task names.
        """
        return list(self.task_definitions.keys())

    def get_task_categories(self) -> List[str]:
        """Get list of all task categories.

        Returns:
            List of unique task categories.
        """
        return list(set(task["category"] for task in self.task_definitions.values()))

    def get_tasks_by_category(self, category: str) -> List[str]:
        """Get all task names belonging to a specific category.

        Args:
            category: Category name to filter by.

        Returns:
            List of task names in the specified category.
        """
        return [
            name for name, task in self.task_definitions.items()
            if task["category"] == category
        ]

    def generate_task_data(self, task_name: str) -> np.ndarray:
        """Generate time series data for a specific task.

        Args:
            task_name: Name of the task to generate data for.

        Returns:
            Generated time series data as numpy array of shape (n_samples, 1).

        Raises:
            ValueError: If task name is not recognized.
        """
        if task_name not in self.task_definitions:
            raise ValueError(
                f"Unknown task: {task_name}. Available tasks: {self.get_task_names()}"
            )

        task_def = self.task_definitions[task_name]
        return task_def["generator"](**task_def.get("params", {}))

    def generate_all_patterns(self) -> Dict[str, np.ndarray]:
        """Generate all available time series patterns.

        Returns:
            Dictionary mapping task names to their generated time series data.
        """
        logger.info(f"Generating all {len(self.task_definitions)} time series patterns...")

        patterns = {}
        for task_name in self.task_names:
            patterns[task_name] = self.generate_task_data(task_name)

        logger.info("All patterns generated successfully")
        return patterns

    def generate_random_pattern(self, category: Optional[str] = None) -> Tuple[str, np.ndarray]:
        """Generate a random time series pattern, optionally from a specific category.

        Args:
            category: Optional category to sample from. If None, samples from all tasks.

        Returns:
            Tuple of (task_name, generated_series).

        Raises:
            ValueError: If specified category doesn't exist.
        """
        if category is not None:
            if category not in self.get_task_categories():
                raise ValueError(
                    f"Unknown category: {category}. Available: {self.get_task_categories()}"
                )
            available_tasks = self.get_tasks_by_category(category)
        else:
            available_tasks = self.get_task_names()

        task_name = self.random_state.choice(available_tasks)
        series = self.generate_task_data(task_name)

        return task_name, series

    def generate_custom_pattern(
            self,
            pattern_type: str,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate a custom time series pattern with specified parameters.

        Args:
            pattern_type: Type of pattern ('trend', 'seasonal', 'stochastic', etc.).
            **kwargs: Parameters specific to the pattern type.

        Returns:
            Generated time series data.

        Raises:
            ValueError: If pattern type is not supported.
        """
        generator_map = {
            "trend": self._generate_trend_series,
            "seasonal": self._generate_seasonal_series,
            "trend_seasonal": self._generate_trend_seasonal,
            "stochastic": self._generate_stochastic_series,
            "mean_reverting": self._generate_mean_reverting,
            "intermittent": self._generate_intermittent_series,
            "garch": self._generate_garch_series,
            "regime_switching": self._generate_regime_switching,
            "structural_break": self._generate_structural_break,
            "outliers": self._generate_outlier_series,
            "chaotic": self._generate_chaotic_series,
            "logistic": self._generate_logistic_growth
        }

        if pattern_type not in generator_map:
            raise ValueError(
                f"Unsupported pattern type: {pattern_type}. "
                f"Available: {list(generator_map.keys())}"
            )

        return generator_map[pattern_type](**kwargs)

    # ========================================================================
    # GENERATOR METHODS
    # ========================================================================

    def _generate_trend_series(
            self,
            trend_type: str,
            noise_level: Optional[float] = None,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate time series with trend patterns.

        Args:
            trend_type: Type of trend ('linear', 'exponential', 'polynomial').
            noise_level: Standard deviation of additive noise.
            **kwargs: Additional parameters specific to trend type.

        Returns:
            Generated time series with trend pattern.

        Raises:
            ValueError: If trend type is not recognized.
        """
        if noise_level is None:
            noise_level = self.config.default_noise_level

        t = np.arange(self.config.n_samples)

        if trend_type == "linear":
            y = kwargs.get("strength", 0.001) * t
        elif trend_type == "exponential":
            y = np.exp(kwargs.get("strength", 0.0001) * t) - 1
        elif trend_type == "polynomial":
            coeffs = kwargs.get("coefficients", [0, 0.001, -1e-7])
            y = np.polyval(coeffs[::-1], t)
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")

        # Add noise to the trend
        noise = self.random_state.normal(0, noise_level, len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_seasonal_series(
            self,
            periods: List[int],
            amplitudes: List[float],
            noise_level: Optional[float] = None
    ) -> np.ndarray:
        """Generate time series with seasonal patterns.

        Args:
            periods: List of seasonal periods.
            amplitudes: List of seasonal amplitudes.
            noise_level: Standard deviation of additive noise.

        Returns:
            Generated time series with seasonal patterns.
        """
        if noise_level is None:
            noise_level = self.config.default_noise_level

        t = np.arange(self.config.n_samples)
        y = np.zeros_like(t, dtype=float)

        # Add multiple seasonal components
        for period, amplitude in zip(periods, amplitudes):
            y += amplitude * np.sin(2 * np.pi * t / period)

        # Add noise
        noise = self.random_state.normal(0, noise_level, len(y))
        return (y + noise).reshape(-1, 1)

    def _generate_trend_seasonal(
            self,
            trend_type: str,
            trend_strength: float,
            periods: List[int],
            seasonal_amplitudes: List[float],
            noise_level: Optional[float] = None,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate time series combining trend and seasonal patterns.

        Args:
            trend_type: Type of trend component.
            trend_strength: Strength of trend component.
            periods: List of seasonal periods.
            seasonal_amplitudes: List of seasonal amplitudes.
            noise_level: Standard deviation of additive noise.
            **kwargs: Additional parameters for trend generation.

        Returns:
            Generated time series with combined trend and seasonal patterns.
        """
        if noise_level is None:
            noise_level = self.config.default_noise_level

        t = np.arange(self.config.n_samples)

        # Generate trend component
        if trend_type == "linear":
            trend = trend_strength * t
        elif trend_type == "exponential":
            trend = np.exp(trend_strength * t) - 1
        else:
            trend = np.zeros_like(t)

        # Generate seasonal component
        seasonal = sum(
            amp * np.sin(2 * np.pi * t / p)
            for p, amp in zip(periods, seasonal_amplitudes)
        )

        # Combine components and add noise
        noise = self.random_state.normal(0, noise_level, len(t))
        return (trend + seasonal + noise).reshape(-1, 1)

    def _generate_stochastic_series(
            self,
            process_type: str,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate stochastic time series (AR, MA, ARMA, random walk).

        Args:
            process_type: Type of stochastic process.
            **kwargs: Parameters specific to the process type.

        Returns:
            Generated stochastic time series.
        """
        n = self.config.n_samples

        if process_type == "random_walk":
            drift = kwargs.get("drift", 0)
            volatility = kwargs.get("volatility", 0.02)
            # Generate random walk with drift
            increments = self.random_state.normal(drift, volatility, n)
            y = np.cumsum(increments)

        elif process_type == "ar":
            ar_coeffs = kwargs.get("ar_coeffs", [0.7])
            noise_std = kwargs.get("noise_std", 0.1)
            p = len(ar_coeffs)

            # Generate AR(p) process
            y = np.zeros(n)
            for t in range(p, n):
                ar_sum = sum(ar_coeffs[i] * y[t - 1 - i] for i in range(p))
                y[t] = ar_sum + self.random_state.normal(0, noise_std)

        elif process_type == "ma":
            ma_coeffs = kwargs.get("ma_coeffs", [0.8])
            noise_std = kwargs.get("noise_std", 0.1)
            q = len(ma_coeffs)

            # Generate MA(q) process
            noise = self.random_state.normal(0, noise_std, n + q)
            y = np.zeros(n)
            for t in range(n):
                ma_sum = sum(ma_coeffs[i] * noise[t + q - 1 - i] for i in range(q))
                y[t] = noise[t + q] + ma_sum

        elif process_type == "arma":
            ar_coeffs = kwargs.get("ar_coeffs", [0.6])
            ma_coeffs = kwargs.get("ma_coeffs", [0.4])
            noise_std = kwargs.get("noise_std", 0.1)
            p, q = len(ar_coeffs), len(ma_coeffs)

            # Generate ARMA(p,q) process
            y = np.zeros(n)
            noise = self.random_state.normal(0, noise_std, n)

            for t in range(max(p, q), n):
                ar_sum = sum(ar_coeffs[i] * y[t - 1 - i] for i in range(p))
                ma_sum = sum(ma_coeffs[i] * noise[t - 1 - i] for i in range(q))
                y[t] = ar_sum + ma_sum + noise[t]

        return y.reshape(-1, 1)

    def _generate_mean_reverting(
            self,
            theta: float,
            mu: float,
            sigma: float
    ) -> np.ndarray:
        """Generate mean-reverting time series (Ornstein-Uhlenbeck process).

        Args:
            theta: Speed of mean reversion.
            mu: Long-term mean.
            sigma: Volatility parameter.

        Returns:
            Generated mean-reverting time series.
        """
        dt = 0.01  # Time step
        y = np.zeros(self.config.n_samples)

        # Generate Ornstein-Uhlenbeck process
        for t in range(1, self.config.n_samples):
            drift = theta * (mu - y[t - 1]) * dt
            diffusion = sigma * self.random_state.normal(0, np.sqrt(dt))
            y[t] = y[t - 1] + drift + diffusion

        return y.reshape(-1, 1)

    def _generate_intermittent_series(
            self,
            demand_prob: float,
            demand_mean: float,
            demand_std: float
    ) -> np.ndarray:
        """Generate intermittent demand time series.

        Args:
            demand_prob: Probability of demand occurrence.
            demand_mean: Mean demand size when demand occurs.
            demand_std: Standard deviation of demand size.

        Returns:
            Generated intermittent demand time series.
        """
        # Generate binary demand occurrence
        demand_occurs = self.random_state.binomial(1, demand_prob, self.config.n_samples)

        # Generate demand sizes
        demand_sizes = self.random_state.normal(demand_mean, demand_std, self.config.n_samples)

        # Combine occurrence and size (ensure non-negative)
        demand = demand_occurs * np.maximum(demand_sizes, 0)
        return demand.reshape(-1, 1)

    def _generate_garch_series(
            self,
            alpha: float,
            beta: float,
            omega: float
    ) -> np.ndarray:
        """Generate GARCH time series with volatility clustering.

        Args:
            alpha: ARCH coefficient.
            beta: GARCH coefficient.
            omega: Constant term in variance equation.

        Returns:
            Generated GARCH time series.
        """
        n = self.config.n_samples
        y = np.zeros(n)
        sigma2 = np.zeros(n)

        # Initialize variance
        if (1 - alpha - beta) > 0:
            sigma2[0] = omega / (1 - alpha - beta)
        else:
            sigma2[0] = omega

        # Generate GARCH(1,1) process
        for t in range(1, n):
            # Update conditional variance
            sigma2[t] = omega + alpha * y[t - 1] ** 2 + beta * sigma2[t - 1]

            # Generate return
            y[t] = self.random_state.normal(0, np.sqrt(sigma2[t]))

        return y.reshape(-1, 1)

    def _generate_regime_switching(
            self,
            regimes: int,
            switch_prob: float,
            regime_params: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Generate regime-switching time series.

        Args:
            regimes: Number of regimes.
            switch_prob: Probability of regime switch at each time step.
            regime_params: List of (drift, volatility) tuples for each regime.

        Returns:
            Generated regime-switching time series.
        """
        n = self.config.n_samples
        y = np.zeros(n)
        regime = 0

        for t in range(1, n):
            # Check for regime switch
            if self.random_state.rand() < switch_prob:
                regime = (regime + 1) % regimes

            # Generate observation based on current regime
            drift, vol = regime_params[regime]
            y[t] = y[t - 1] + drift + self.random_state.normal(0, vol)

        return y.reshape(-1, 1)

    def _generate_structural_break(
            self,
            break_type: str,
            break_magnitude: float,
            break_points: List[float]
    ) -> np.ndarray:
        """Generate time series with structural breaks.

        Args:
            break_type: Type of structural break ('level' or 'trend').
            break_magnitude: Magnitude of the structural break.
            break_points: List of break points as fractions of series length.

        Returns:
            Generated time series with structural breaks.
        """
        n = self.config.n_samples

        # Start with basic trend and noise
        y = 0.0005 * np.arange(n) + self.random_state.normal(0, 0.1, n)

        # Apply structural breaks
        for bp in break_points:
            idx = int(bp * n)
            if break_type == "level":
                # Level shift
                y[idx:] += break_magnitude
            elif break_type == "trend":
                # Trend change
                y[idx:] += break_magnitude * np.arange(n - idx)

        return y.reshape(-1, 1)

    def _generate_outlier_series(
            self,
            outlier_type: str,
            outlier_prob: float,
            outlier_magnitude: float
    ) -> np.ndarray:
        """Generate time series with outliers.

        Args:
            outlier_type: Type of outliers ('additive' or 'innovation').
            outlier_prob: Probability of outlier occurrence.
            outlier_magnitude: Magnitude of outliers.

        Returns:
            Generated time series with outliers.
        """
        n = self.config.n_samples
        t = np.arange(n)

        # Generate base series with trend and seasonality
        y = 0.001 * t + np.sin(2 * np.pi * t / 24) + self.random_state.normal(0, 0.1, n)

        # Add outliers
        outlier_locations = self.random_state.binomial(1, outlier_prob, n).astype(bool)
        outlier_magnitudes = self.random_state.normal(0, outlier_magnitude, n)

        if outlier_type == "additive":
            y[outlier_locations] += outlier_magnitudes[outlier_locations]

        return y.reshape(-1, 1)

    def _generate_chaotic_series(
            self,
            system: str,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate chaotic time series (Henon map, Lorenz system).

        Args:
            system: Type of chaotic system ('henon' or 'lorenz').
            **kwargs: Parameters specific to the chaotic system.

        Returns:
            Generated chaotic time series.
        """
        n = self.config.n_samples

        if system == "henon":
            # Henon map parameters
            a = kwargs.get("a", 1.4)
            b = kwargs.get("b", 0.3)

            # Initialize
            x, y = 0.1, 0.1
            trajectory = []

            # Generate trajectory (with burn-in period)
            for i in range(n + 100):
                x_new = 1 - a * x ** 2 + y
                y_new = b * x
                x, y = x_new, y_new

                # Skip burn-in period
                if i >= 100:
                    trajectory.append(x)

            data = np.array(trajectory)

        elif system == "lorenz":
            # Lorenz system parameters
            sigma = kwargs.get("sigma", 10)
            rho = kwargs.get("rho", 28)
            beta = kwargs.get("beta", 8 / 3)
            dt = 0.01

            # Initialize
            x, y, z = 1.0, 1.0, 1.0
            trajectory = []

            # Generate trajectory (with burn-in and subsampling)
            for i in range(n * 10 + 1000):
                # Lorenz equations
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z

                # Update state
                x += dx * dt
                y += dy * dt
                z += dz * dt

                # Skip burn-in and subsample
                if i >= 1000 and i % 10 == 0:
                    trajectory.append(x)

            data = np.array(trajectory)

        # Add small amount of noise to make it more realistic
        noise = self.random_state.normal(0, 0.01, len(data))
        return (data + noise).reshape(-1, 1)

    def _generate_logistic_growth(
            self,
            carrying_capacity: float,
            growth_rate: float,
            noise_level: Optional[float] = None
    ) -> np.ndarray:
        """Generate logistic growth time series.

        Args:
            carrying_capacity: Maximum value (K parameter).
            growth_rate: Growth rate parameter.
            noise_level: Standard deviation of additive noise.

        Returns:
            Generated logistic growth time series.
        """
        if noise_level is None:
            noise_level = self.config.default_noise_level

        t = np.arange(self.config.n_samples)

        # Logistic growth equation
        y = carrying_capacity / (1 + np.exp(-growth_rate * (t - self.config.n_samples / 2)))

        # Add noise
        noise = self.random_state.normal(0, noise_level, len(y))
        return (y + noise).reshape(-1, 1)

# ---------------------------------------------------------------------
