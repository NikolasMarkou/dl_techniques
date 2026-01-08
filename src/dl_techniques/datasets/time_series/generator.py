"""
Comprehensive Time Series Generator for Deep Learning and Forecasting Experiments

This module provides a sophisticated time series generator designed to create diverse,
realistic time series patterns for machine learning research, model testing, and
forecasting experiments. The generator supports over 80 different time series patterns
across multiple domains and categories.

Overview
--------
The TimeSeriesGenerator creates synthetic time series data that mimics real-world
patterns found in various domains including finance, weather, industrial sensors,
biomedical signals, network traffic, and more. Each pattern is carefully designed
to exhibit characteristic behaviors that challenge different aspects of forecasting
models and time series analysis techniques.

Key Features
------------
* **Comprehensive Pattern Library**: 80+ predefined time series patterns
* **Data Augmentation**: Includes time warping to increase data diversity.
* **Domain-Specific Patterns**: Specialized generators for finance, weather,
  industrial, biomedical, and network domains
* **Configurable Parameters**: Fully customizable generation parameters
* **Reproducible Generation**: Seed-based random state for consistent results
* **Multiple Pattern Categories**: Organized patterns by type and complexity
* **Custom Pattern Creation**: Build custom patterns with specific parameters

Pattern Categories
------------------
The generator organizes patterns into the following categories:

**Basic Patterns:**
* **trend**: Linear, exponential, polynomial, logarithmic trends
* **seasonal**: Single and multi-period seasonal patterns
* **composite**: Combined trend and seasonal patterns

**Stochastic Processes:**
* **stochastic**: AR, MA, ARMA processes, random walks
* **volatility**: GARCH models with volatility clustering
* **regime**: Regime-switching models
* **structural**: Structural breaks and change points

**Domain-Specific Patterns:**
* **financial**: Stock prices, commodity prices, interest rates, crypto
* **weather**: Temperature, precipitation, humidity, wind patterns
* **network**: Web traffic, server load, bandwidth usage, latency
* **biomedical**: ECG, EEG, blood pressure, respiratory signals
* **industrial**: Motor vibration, HVAC systems, production rates

**Advanced Patterns:**
* **intermittent**: Sparse and lumpy demand patterns
* **outliers**: Additive, innovation, and level outliers
* **chaotic**: Henon map, Lorenz attractor, Mackey-Glass equation

Classes
-------
TimeSeriesConfig
    Configuration dataclass containing all generation parameters including
    sample size, noise levels, trend strengths, seasonal periods, and
    domain-specific parameters.

TimeSeriesGenerator
    Main generator class that creates time series patterns based on the
    provided configuration. Supports both predefined patterns and custom
    pattern generation.

Usage Examples
--------------
Basic Usage:
    >>> from dl_techniques.datasets.time_series import (
    ...     TimeSeriesGenerator, TimeSeriesConfig
    ... )
    >>> config = TimeSeriesGeneratorConfig(n_samples=1000, random_seed=42)
    >>> generator = TimeSeriesGenerator(config)
    >>>
    >>> # Generate specific pattern
    >>> trend_data = generator.generate_task_data("linear_trend_strong")
    >>> seasonal_data = generator.generate_task_data("multi_seasonal")

Generate All Patterns:
    >>> # Generate complete pattern set
    >>> all_patterns = generator.generate_all_patterns()
    >>> print(f"Generated {len(all_patterns)} different patterns")

Category-Based Generation:
    >>> # Get patterns by category
    >>> stochastic_tasks = generator.get_tasks_by_category("stochastic")
    >>>
    >>> # Generate random pattern from category
    >>> task_name, data = generator.generate_random_pattern(category="financial")

Custom Pattern Generation:
    >>> # Create custom trend pattern
    >>> custom_trend = generator.generate_custom_pattern(
    ...     "trend",
    ...     trend_type="polynomial",
    ...     coefficients=[0, 0.002, -1e-6],
    ...     noise_level=0.05
    ... )
    >>>
    >>> # Create custom ARMA process
    >>> custom_arma = generator.generate_custom_pattern(
    ...     "stochastic",
    ...     process_type="arma",
    ...     ar_coeffs=[0.8, -0.3],
    ...     ma_coeffs=[0.4],
    ...     noise_std=0.1
    ... )

Multi-Task Dataset Creation:
    >>> # Create dataset for multiple forecasting tasks
    >>> selected_tasks = [
    ...     "linear_trend_strong", "multi_seasonal", "arma_process",
    ...     "garch_low_vol", "intermittent_demand"
    ... ]
    >>>
    >>> dataset = {}
    >>> for task_name in selected_tasks:
    ...     task_data = []
    ...     for i in range(100):  # 100 series per task
    ...         config = TimeSeriesGeneratorConfig(n_samples=200, random_seed=42 + i)
    ...         gen = TimeSeriesGenerator(config)
    ...         series = gen.generate_task_data(task_name)
    ...         task_data.append(series)
    ...     dataset[task_name] = np.array(task_data)

Augmentation Usage:
    >>> # Generate a base series
    >>> base_series = generator.generate_task_data("trend_daily_seasonal")
    >>>
    >>> # Define and apply augmentations
    >>> augmentations = [
    ...     {"name": "time_warp", "n_knots": 4, "strength": 0.05},
    ... ]
    >>> augmented_series = generator.augment_series(base_series, augmentations)

Available Patterns
------------------
The generator includes the following predefined patterns:

**Trend Patterns (8 patterns):**
* linear_trend_strong, linear_trend_weak
* exponential_growth, polynomial_trend, quadratic_trend, cubic_trend
* logistic_growth, logarithmic_trend

**Seasonal Patterns (7 patterns):**
* daily_seasonality, weekly_seasonality, hourly_seasonality, yearly_seasonality
* multi_seasonal, complex_seasonal, triple_seasonal

**Composite Patterns (4 patterns):**
* trend_daily_seasonal, trend_weekly_seasonal
* exp_trend_multi_seasonal, poly_trend_complex_seasonal

**Stochastic Processes (7 patterns):**
* random_walk, ar_process, ma_process, arma_process
* mean_reverting, ar_high_order, ma_high_order

**Financial Patterns (5 patterns):**
* stock_returns, commodity_prices, interest_rates
* currency_exchange, crypto_prices

**Weather Patterns (4 patterns):**
* temperature_daily, precipitation, humidity, wind_speed

**Network Patterns (4 patterns):**
* web_traffic, server_load, network_latency, bandwidth_usage

**Biomedical Patterns (4 patterns):**
* heartbeat, brain_waves, blood_pressure, respiratory_rate

**Industrial Patterns (4 patterns):**
* machine_vibration, energy_consumption, production_rate, temperature_sensor

**Advanced Patterns:**
* intermittent_demand, lumpy_demand, sparse_events (3 patterns)
* garch_low_vol, garch_high_vol, garch_persistent (3 patterns)
* regime_switching, multi_regime_switching (2 patterns)
* level_shift, trend_change, variance_break (3 patterns)
* additive_outliers, innovation_outliers, level_outliers (3 patterns)
* henon_map, lorenz_x, lorenz_y, mackey_glass (4 patterns)

Configuration Parameters
------------------------
The TimeSeriesConfig class provides extensive customization options:

* **n_samples**: Number of time steps (default: 5000)
* **random_seed**: Random seed for reproducibility (default: 42)
* **default_noise_level**: Base noise level (default: 0.1)
* **trend_strengths**: Range of trend strengths (default: (0.00005, 0.01))
* **seasonal_periods**: Available seasonal periods (default: [12, 24, 48, ...])
* **seasonal_amplitudes**: Range of seasonal amplitudes (default: (0.2, 3.0))
* **ar_coeffs_range**: Range for AR coefficients (default: (-0.8, 0.8))
* **ma_coeffs_range**: Range for MA coefficients (default: (-0.8, 0.8))
* **volatility_range**: Range for volatility parameters (default: (0.01, 0.3))

Notes
-----
* All generated time series are returned as numpy arrays with shape (n_samples, 1)
* The generator uses a seeded random state for reproducible results
* Patterns are designed to be challenging for different types of forecasting models
* Domain-specific patterns include realistic parameter ranges and behaviors
* Custom patterns can be created by directly calling generator methods with parameters

References
----------
The patterns implemented in this generator are based on established time series
models and real-world phenomena documented in forecasting literature, including:

* Box, G. E. P., & Jenkins, G. M. (1976). Time Series Analysis: Forecasting and Control
* Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice
* Hamilton, J. D. (1994). Time Series Analysis
* Tsay, R. S. (2010). Analysis of Financial Time Series

The chaotic systems implementations follow standard formulations:
* Henon, M. (1976). A two-dimensional mapping with a strange attractor
* Lorenz, E. N. (1963). Deterministic nonperiodic flow
* Mackey, M. C., & Glass, L. (1977). Oscillation and chaos in physiological control systems
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@dataclass
class TimeSeriesGeneratorConfig:
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
    n_samples: int = 5000
    random_seed: int = 42
    default_noise_level: float = 0.1

    # Trend parameters
    trend_strengths: Tuple[float, float] = (0.00005, 0.01)

    # Seasonal parameters
    seasonal_periods: List[int] = field(default_factory=lambda: [12, 24, 48, 60, 96, 168, 336, 720, 8760])
    seasonal_amplitudes: Tuple[float, float] = (0.2, 3.0)

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
        >>> config = TimeSeriesGeneratorConfig(n_samples=500, random_seed=42)
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

    def __init__(self, config: TimeSeriesGeneratorConfig) -> None:
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
        tasks["quadratic_trend"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {
                "trend_type": "polynomial",
                "coefficients": [0, 0, 1e-6],
                "noise_level": 0.08
            }
        }
        tasks["cubic_trend"] = {
            "category": "trend",
            "generator": self._generate_trend_series,
            "params": {
                "trend_type": "polynomial",
                "coefficients": [0, 0, 0, 1e-9],
                "noise_level": 0.1
            }
        }
        tasks["logarithmic_trend"] = {
            "category": "trend",
            "generator": self._generate_log_trend,
            "params": {
                "strength": 0.5,
                "noise_level": 0.12
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
        tasks["hourly_seasonality"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {
                "periods": [60],
                "amplitudes": [0.8],
                "noise_level": 0.1
            }
        }
        tasks["yearly_seasonality"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {
                "periods": [8760],
                "amplitudes": [2.0],
                "noise_level": 0.15
            }
        }
        tasks["triple_seasonal"] = {
            "category": "seasonal",
            "generator": self._generate_seasonal_series,
            "params": {
                "periods": [12, 24, 168, 8760],
                "amplitudes": [0.5, 1.0, 0.8, 1.5],
                "noise_level": 0.2
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
        tasks["poly_trend_complex_seasonal"] = {
            "category": "composite",
            "generator": self._generate_trend_seasonal,
            "params": {
                "trend_type": "polynomial",
                "trend_strength": 0.0005,
                "coefficients": [0, 0.001, -1e-7],
                "periods": [12, 24, 168],
                "seasonal_amplitudes": [0.5, 1.0, 0.7],
                "noise_level": 0.15
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
        tasks["ar_high_order"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {
                "process_type": "ar",
                "ar_coeffs": [0.6, -0.3, 0.1, -0.05],
                "noise_std": 0.12
            }
        }
        tasks["ma_high_order"] = {
            "category": "stochastic",
            "generator": self._generate_stochastic_series,
            "params": {
                "process_type": "ma",
                "ma_coeffs": [0.7, 0.2, -0.1, 0.05],
                "noise_std": 0.1
            }
        }

        # === FINANCIAL PATTERNS ===
        tasks["stock_returns"] = {
            "category": "financial",
            "generator": self._generate_financial_series,
            "params": {
                "return_type": "stock",
                "volatility": 0.2,
                "drift": 0.05
            }
        }
        tasks["commodity_prices"] = {
            "category": "financial",
            "generator": self._generate_financial_series,
            "params": {
                "return_type": "commodity",
                "volatility": 0.35,
                "drift": 0.02
            }
        }
        tasks["interest_rates"] = {
            "category": "financial",
            "generator": self._generate_financial_series,
            "params": {
                "return_type": "rates",
                "volatility": 0.1,
                "drift": -0.01
            }
        }
        tasks["currency_exchange"] = {
            "category": "financial",
            "generator": self._generate_financial_series,
            "params": {
                "return_type": "currency",
                "volatility": 0.15,
                "drift": 0.0
            }
        }
        tasks["crypto_prices"] = {
            "category": "financial",
            "generator": self._generate_financial_series,
            "params": {
                "return_type": "crypto",
                "volatility": 0.8,
                "drift": 0.1
            }
        }

        # === WEATHER PATTERNS ===
        tasks["temperature_daily"] = {
            "category": "weather",
            "generator": self._generate_weather_series,
            "params": {
                "weather_type": "temperature",
                "seasonal_strength": 15,
                "noise_level": 2.0
            }
        }
        tasks["precipitation"] = {
            "category": "weather",
            "generator": self._generate_weather_series,
            "params": {
                "weather_type": "precipitation",
                "intermittent_prob": 0.3,
                "noise_level": 0.5
            }
        }
        tasks["humidity"] = {
            "category": "weather",
            "generator": self._generate_weather_series,
            "params": {
                "weather_type": "humidity",
                "base_level": 60,
                "seasonal_variation": 20,
                "noise_level": 5
            }
        }
        tasks["wind_speed"] = {
            "category": "weather",
            "generator": self._generate_weather_series,
            "params": {
                "weather_type": "wind",
                "base_speed": 10,
                "gust_prob": 0.1,
                "noise_level": 2
            }
        }

        # === NETWORK/TRAFFIC PATTERNS ===
        tasks["web_traffic"] = {
            "category": "network",
            "generator": self._generate_network_series,
            "params": {
                "pattern_type": "web_traffic",
                "peak_hours": [9, 14, 20],
                "weekend_factor": 0.6
            }
        }
        tasks["server_load"] = {
            "category": "network",
            "generator": self._generate_network_series,
            "params": {
                "pattern_type": "server_load",
                "spike_prob": 0.02,
                "baseline": 0.3
            }
        }
        tasks["network_latency"] = {
            "category": "network",
            "generator": self._generate_network_series,
            "params": {
                "pattern_type": "latency",
                "base_latency": 50,
                "congestion_periods": [8, 12, 18, 22]
            }
        }
        tasks["bandwidth_usage"] = {
            "category": "network",
            "generator": self._generate_network_series,
            "params": {
                "pattern_type": "bandwidth",
                "daily_cycle": True,
                "burst_prob": 0.05
            }
        }

        # === BIOMEDICAL PATTERNS ===
        tasks["heartbeat"] = {
            "category": "biomedical",
            "generator": self._generate_biomedical_series,
            "params": {
                "signal_type": "ecg",
                "heart_rate": 72,
                "noise_level": 0.1
            }
        }
        tasks["brain_waves"] = {
            "category": "biomedical",
            "generator": self._generate_biomedical_series,
            "params": {
                "signal_type": "eeg",
                "frequency_bands": [8, 13, 30],
                "noise_level": 0.2
            }
        }
        tasks["blood_pressure"] = {
            "category": "biomedical",
            "generator": self._generate_biomedical_series,
            "params": {
                "signal_type": "bp",
                "systolic_base": 120,
                "diastolic_base": 80,
                "noise_level": 5
            }
        }
        tasks["respiratory_rate"] = {
            "category": "biomedical",
            "generator": self._generate_biomedical_series,
            "params": {
                "signal_type": "resp",
                "base_rate": 16,
                "variability": 2,
                "noise_level": 0.5
            }
        }

        # === INDUSTRIAL PATTERNS ===
        tasks["machine_vibration"] = {
            "category": "industrial",
            "generator": self._generate_industrial_series,
            "params": {
                "machine_type": "motor",
                "degradation_rate": 0.001,
                "fault_prob": 0.01
            }
        }
        tasks["energy_consumption"] = {
            "category": "industrial",
            "generator": self._generate_industrial_series,
            "params": {
                "machine_type": "hvac",
                "efficiency_trend": -0.0005,
                "seasonal_periods": [24, 168]
            }
        }
        tasks["production_rate"] = {
            "category": "industrial",
            "generator": self._generate_industrial_series,
            "params": {
                "machine_type": "production",
                "target_rate": 100,
                "maintenance_cycles": [720, 4320],
                "noise_level": 5
            }
        }
        tasks["temperature_sensor"] = {
            "category": "industrial",
            "generator": self._generate_industrial_series,
            "params": {
                "machine_type": "sensor",
                "operating_temp": 85,
                "drift_rate": 0.01,
                "spike_prob": 0.005
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
        tasks["sparse_events"] = {
            "category": "intermittent",
            "generator": self._generate_intermittent_series,
            "params": {
                "demand_prob": 0.05,
                "demand_mean": 10.0,
                "demand_std": 2.0
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
        tasks["garch_persistent"] = {
            "category": "volatility",
            "generator": self._generate_garch_series,
            "params": {
                "alpha": 0.05,
                "beta": 0.93,
                "omega": 0.001
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
        tasks["multi_regime_switching"] = {
            "category": "regime",
            "generator": self._generate_regime_switching,
            "params": {
                "regimes": 3,
                "switch_prob": 0.03,
                "regime_params": [(0.0, 0.05), (0.002, 0.1), (-0.001, 0.2)]
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
        tasks["variance_break"] = {
            "category": "structural",
            "generator": self._generate_structural_break,
            "params": {
                "break_type": "variance",
                "break_magnitude": 2.0,
                "break_points": [0.6]
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
        tasks["level_outliers"] = {
            "category": "outliers",
            "generator": self._generate_outlier_series,
            "params": {
                "outlier_type": "level",
                "outlier_prob": 0.02,
                "outlier_magnitude": 4.0
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
        tasks["lorenz_y"] = {
            "category": "chaotic",
            "generator": self._generate_chaotic_series,
            "params": {
                "system": "lorenz",
                "component": "y",
                "sigma": 10,
                "rho": 28,
                "beta": 8 / 3
            }
        }
        tasks["mackey_glass"] = {
            "category": "chaotic",
            "generator": self._generate_chaotic_series,
            "params": {
                "system": "mackey_glass",
                "a": 0.2,
                "b": 0.1,
                "c": 10,
                "tau": 17
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
        for task_name in self.get_task_names():
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
            "logistic": self._generate_logistic_growth,
            "financial": self._generate_financial_series,
            "weather": self._generate_weather_series,
            "network": self._generate_network_series,
            "biomedical": self._generate_biomedical_series,
            "industrial": self._generate_industrial_series
        }

        if pattern_type not in generator_map:
            raise ValueError(
                f"Unsupported pattern type: {pattern_type}. "
                f"Available: {list(generator_map.keys())}"
            )

        return generator_map[pattern_type](**kwargs)

    # ========================================================================
    # AUGMENTATION METHODS
    # ========================================================================

    def augment_series(
            self,
            series: np.ndarray,
            augmentations: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Apply a sequence of augmentations to a time series.

        Args:
            series: The input time series data of shape (n_samples, 1).
            augmentations: A list of augmentation dictionaries. Each dictionary
                           should have a "name" key and optional parameter keys.
                           Example: [{"name": "time_warp", "n_knots": 4, "strength": 0.1}]

        Returns:
            The augmented time series.
        """
        augmented_series = series.copy()
        dispatcher = {
            "time_warp": self._augment_time_warp,
            # Other augmentation methods can be registered here
        }

        for aug in augmentations:
            aug_name = aug.pop("name", None)
            if not aug_name or aug_name not in dispatcher:
                raise ValueError(f"Unknown or unspecified augmentation: {aug_name}")

            augmented_series = dispatcher[aug_name](augmented_series, **aug)

        return augmented_series

    def _augment_time_warp(
            self,
            series: np.ndarray,
            n_knots: int = 4,
            strength: float = 0.1
    ) -> np.ndarray:
        """Apply time warping to a time series using a smooth cubic spline.

        Args:
            series: The input time series data of shape (n_samples, 1).
            n_knots: The number of knots for the cubic spline.
            strength: The strength of the warping. Represents the standard
                      deviation of the noise added to knots, relative to the
                      average interval length.

        Returns:
            The time-warped series.
        """
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            raise ImportError("scipy is required for time warping. Please install it with 'pip install scipy'")

        if n_knots < 2:
            raise ValueError("n_knots must be greater than 1")

        n_samples = len(series)
        x_original = np.arange(n_samples)

        # Define knots for the spline
        x_knots = np.linspace(0, n_samples - 1, num=n_knots)

        # Generate random perturbations for the knots
        interval_length = (n_samples - 1) / (n_knots - 1) if n_knots > 1 else n_samples -1
        noise = self.random_state.normal(loc=0, scale=interval_length * strength, size=n_knots)
        y_knots = x_knots + noise

        # Fix endpoints and ensure monotonicity
        y_knots[0] = 0
        y_knots[-1] = n_samples - 1
        y_knots = np.sort(y_knots)

        # Create the warping function (spline)
        spline = CubicSpline(x_knots, y_knots)
        warped_indices = spline(x_original)

        # Resample the original series at the warped time indices
        y_warped = np.interp(warped_indices, x_original, series.flatten())

        return y_warped.reshape(-1, 1)

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

    def _generate_log_trend(
            self,
            strength: float,
            noise_level: float
    ) -> np.ndarray:
        """Generate logarithmic trend series."""
        t = np.arange(1, self.config.n_samples + 1)
        y = strength * np.log(t)
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
        elif trend_type == "polynomial":
            coeffs = kwargs.get("coefficients", [0, 0.001, -1e-7])
            trend = np.polyval(coeffs[::-1], t)
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
            break_type: Type of structural break ('level', 'trend', 'variance').
            break_magnitude: Magnitude of the structural break.
            break_points: List of break points as fractions of series length.

        Returns:
            Generated time series with structural breaks.
        """
        n = self.config.n_samples

        # Start with basic trend and noise
        base_noise_std = 0.1
        y = 0.0005 * np.arange(n) + self.random_state.normal(0, base_noise_std, n)

        # Apply structural breaks
        for bp in break_points:
            idx = int(bp * n)
            if break_type == "level":
                # Level shift
                y[idx:] += break_magnitude
            elif break_type == "trend":
                # Trend change
                y[idx:] += break_magnitude * np.arange(n - idx)
            elif break_type == "variance":
                # Variance change
                new_noise = self.random_state.normal(0, base_noise_std * break_magnitude, n - idx)
                y[idx:] = y[idx:] - self.random_state.normal(0, base_noise_std, n - idx) + new_noise

        return y.reshape(-1, 1)

    def _generate_outlier_series(
            self,
            outlier_type: str,
            outlier_prob: float,
            outlier_magnitude: float
    ) -> np.ndarray:
        """Generate time series with outliers.

        Args:
            outlier_type: Type of outliers ('additive', 'innovation', 'level').
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
        elif outlier_type == "innovation":
            # Innovation outliers affect subsequent values through AR structure
            for i in np.where(outlier_locations)[0]:
                for j in range(min(5, n - i)):  # Affect next 5 points
                    if i + j < n:
                        y[i + j] += outlier_magnitudes[i] * (0.5 ** j)
        elif outlier_type == "level":
            # Level outliers create temporary level shifts
            for i in np.where(outlier_locations)[0]:
                duration = min(self.random_state.randint(3, 10), n - i)
                y[i:i + duration] += outlier_magnitudes[i]

        return y.reshape(-1, 1)

    def _generate_chaotic_series(
            self,
            system: str,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate chaotic time series (Henon map, Lorenz system, Mackey-Glass).

        Args:
            system: Type of chaotic system ('henon', 'lorenz', 'mackey_glass').
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
            component = kwargs.get("component", "x")
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
                    if component == "x":
                        trajectory.append(x)
                    elif component == "y":
                        trajectory.append(y)
                    else:
                        trajectory.append(z)

            data = np.array(trajectory)

        elif system == "mackey_glass":
            # Mackey-Glass delay differential equation
            a = kwargs.get("a", 0.2)
            b = kwargs.get("b", 0.1)
            c = kwargs.get("c", 10)
            tau = kwargs.get("tau", 17)

            # Initialize history
            history = np.ones(tau + 1) * 1.2
            trajectory = []

            for i in range(n + 100):
                # Mackey-Glass equation
                current = history[-1]
                delayed = history[-tau] if len(history) > tau else history[0]

                dx = (a * delayed) / (1 + delayed ** c) - b * current
                new_val = current + dx

                history = np.append(history, max(0, new_val))
                if len(history) > tau + 10:
                    history = history[-tau - 5:]

                # Skip burn-in period
                if i >= 100:
                    trajectory.append(new_val)

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

    def _generate_financial_series(
            self,
            return_type: str,
            volatility: float,
            drift: float
    ) -> np.ndarray:
        """Generate financial time series patterns."""
        n = self.config.n_samples

        if return_type == "stock":
            # Geometric Brownian Motion
            dt = 1/252  # Daily returns
            returns = self.random_state.normal(drift * dt, volatility * np.sqrt(dt), n)
            prices = 100 * np.exp(np.cumsum(returns))

        elif return_type == "commodity":
            # Mean-reverting with jumps
            prices = np.zeros(n)
            prices[0] = 50
            for t in range(1, n):
                mean_reversion = 0.1 * (50 - prices[t-1])
                volatility_term = volatility * self.random_state.normal()
                jump = 0 if self.random_state.rand() > 0.05 else self.random_state.normal(0, 5)
                prices[t] = prices[t-1] + mean_reversion + volatility_term + jump

        elif return_type == "rates":
            # Vasicek model for interest rates
            rates = np.zeros(n)
            rates[0] = 0.05  # Start at 5%
            for t in range(1, n):
                dr = 0.1 * (0.05 - rates[t-1]) + volatility * self.random_state.normal()
                rates[t] = max(0, rates[t-1] + dr)
            prices = rates * 100

        elif return_type == "currency":
            # Random walk with mean reversion
            prices = np.zeros(n)
            prices[0] = 1.0
            for t in range(1, n):
                mean_reversion = 0.01 * (1.0 - prices[t-1])
                random_shock = volatility * self.random_state.normal()
                prices[t] = prices[t-1] + mean_reversion + random_shock

        elif return_type == "crypto":
            # High volatility with occasional large jumps
            prices = np.zeros(n)
            prices[0] = 1000
            for t in range(1, n):
                # Base volatility
                base_change = self.random_state.normal(drift/252, volatility/np.sqrt(252))
                # Occasional large jumps
                if self.random_state.rand() < 0.02:
                    jump = self.random_state.normal(0, 0.2)
                    base_change += jump
                prices[t] = prices[t-1] * (1 + base_change)

        return prices.reshape(-1, 1)

    def _generate_weather_series(
            self,
            weather_type: str,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate weather-related time series."""
        t = np.arange(self.config.n_samples)

        if weather_type == "temperature":
            seasonal_strength = kwargs.get("seasonal_strength", 15)
            noise_level = kwargs.get("noise_level", 2.0)

            # Annual cycle + daily cycle + trend + noise
            annual = seasonal_strength * np.sin(2 * np.pi * t / 365.25)
            daily = 5 * np.sin(2 * np.pi * t / 24) if self.config.n_samples > 100 else 0
            trend = 0.01 * t / 365.25  # Slight warming trend
            noise = self.random_state.normal(0, noise_level, len(t))

            y = 20 + annual + daily + trend + noise

        elif weather_type == "precipitation":
            intermittent_prob = kwargs.get("intermittent_prob", 0.3)
            noise_level = kwargs.get("noise_level", 0.5)

            # Seasonal precipitation pattern
            seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * t / 365.25)
            rain_occurs = self.random_state.binomial(1, intermittent_prob * seasonal_factor, len(t))
            rain_amounts = self.random_state.exponential(5, len(t))

            y = rain_occurs * rain_amounts

        elif weather_type == "humidity":
            base_level = kwargs.get("base_level", 60)
            seasonal_variation = kwargs.get("seasonal_variation", 20)
            noise_level = kwargs.get("noise_level", 5)

            seasonal = seasonal_variation * np.sin(2 * np.pi * t / 365.25 + np.pi)
            daily = 10 * np.sin(2 * np.pi * t / 24 + np.pi/2)
            noise = self.random_state.normal(0, noise_level, len(t))

            y = base_level + seasonal + daily + noise
            y = np.clip(y, 0, 100)  # Humidity is 0-100%

        elif weather_type == "wind":
            base_speed = kwargs.get("base_speed", 10)
            gust_prob = kwargs.get("gust_prob", 0.1)
            noise_level = kwargs.get("noise_level", 2)

            # Base wind with seasonal variation
            seasonal = 3 * np.sin(2 * np.pi * t / 365.25)
            base_wind = base_speed + seasonal

            # Add gusts
            gusts = self.random_state.binomial(1, gust_prob, len(t)) * self.random_state.exponential(5, len(t))
            noise = self.random_state.normal(0, noise_level, len(t))

            y = base_wind + gusts + noise
            y = np.maximum(y, 0)

        return y.reshape(-1, 1)

    def _generate_network_series(
            self,
            pattern_type: str,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate network/traffic patterns."""
        t = np.arange(self.config.n_samples)

        if pattern_type == "web_traffic":
            peak_hours = kwargs.get("peak_hours", [9, 14, 20])
            weekend_factor = kwargs.get("weekend_factor", 0.6)

            # Daily pattern with multiple peaks
            daily_pattern = np.zeros_like(t, dtype=float)
            for hour in peak_hours:
                daily_pattern += np.exp(-0.5 * ((t % 24 - hour) / 2) ** 2)

            # Weekly pattern (lower on weekends)
            weekly_factor = np.where((t // 24) % 7 >= 5, weekend_factor, 1.0)

            # Growth trend + seasonality + noise
            growth = 1 + 0.0001 * t
            noise = self.random_state.normal(0, 0.1, len(t))

            y = 1000 * growth * daily_pattern * weekly_factor * (1 + noise)

        elif pattern_type == "server_load":
            spike_prob = kwargs.get("spike_prob", 0.02)
            baseline = kwargs.get("baseline", 0.3)

            # Base load with daily cycle
            daily_cycle = 0.3 + 0.4 * (1 + np.sin(2 * np.pi * t / 24 - np.pi/2)) / 2
            base_load = baseline + daily_cycle

            # Add spikes
            spikes = self.random_state.binomial(1, spike_prob, len(t)) * self.random_state.exponential(0.5, len(t))
            noise = self.random_state.normal(0, 0.05, len(t))

            y = base_load + spikes + noise
            y = np.clip(y, 0, 1)  # Load is 0-100%

        elif pattern_type == "latency":
            base_latency = kwargs.get("base_latency", 50)
            congestion_periods = kwargs.get("congestion_periods", [8, 12, 18, 22])

            # Base latency
            latency = np.full(len(t), base_latency, dtype=float)

            # Add congestion periods
            for period in congestion_periods:
                congestion_mask = np.abs((t % 24) - period) < 2
                latency += congestion_mask * 20

            # Add network jitter
            jitter = self.random_state.exponential(5, len(t))
            occasional_spikes = self.random_state.binomial(1, 0.01, len(t)) * self.random_state.exponential(100, len(t))

            y = latency + jitter + occasional_spikes

        elif pattern_type == "bandwidth":
            daily_cycle = kwargs.get("daily_cycle", True)
            burst_prob = kwargs.get("burst_prob", 0.05)

            if daily_cycle:
                # Business hours pattern
                business_hours = (8 <= (t % 24)) & ((t % 24) <= 18)
                base_usage = np.where(business_hours, 0.7, 0.2)
            else:
                base_usage = 0.4

            # Add bursts
            bursts = self.random_state.binomial(1, burst_prob, len(t)) * self.random_state.exponential(0.3, len(t))
            noise = self.random_state.normal(0, 0.1, len(t))

            y = base_usage + bursts + noise
            y = np.clip(y, 0, 1)

        return np.maximum(y, 0).reshape(-1, 1)

    def _generate_biomedical_series(
            self,
            signal_type: str,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate biomedical signal patterns."""
        t = np.arange(self.config.n_samples)

        if signal_type == "ecg":
            heart_rate = kwargs.get("heart_rate", 72)
            noise_level = kwargs.get("noise_level", 0.1)
            sampling_rate = kwargs.get("sampling_rate", 100) # samples per second

            # ECG waveform approximation
            rr_interval = 60 / heart_rate  # seconds between beats
            beat_times = np.arange(0, len(t), rr_interval * sampling_rate)

            ecg = np.zeros(len(t))
            for beat_time in beat_times:
                if beat_time < len(t):
                    # Simplified QRS complex
                    peak_idx = int(beat_time)
                    if peak_idx < len(t):
                        ecg[peak_idx] = 1.0
                        # Add P and T waves
                        if peak_idx > 20:
                            ecg[peak_idx - 20] = 0.2
                        if peak_idx + 40 < len(t):
                            ecg[peak_idx + 40] = 0.3
            # Smooth and add noise
            ecg = gaussian_filter1d(ecg, sigma=2)
            noise = self.random_state.normal(0, noise_level, len(t))
            y = ecg + noise

        elif signal_type == "eeg":
            frequency_bands = kwargs.get("frequency_bands", [8, 13, 30])
            noise_level = kwargs.get("noise_level", 0.2)

            # Simulate EEG as sum of frequency bands
            y = np.zeros(len(t))
            for freq in frequency_bands:
                amplitude = self.random_state.uniform(0.5, 1.5)
                phase = self.random_state.uniform(0, 2*np.pi)
                y += amplitude * np.sin(2 * np.pi * freq * t / 1000 + phase)

            # Add 1/f noise characteristic of EEG
            noise = self.random_state.normal(0, noise_level, len(t))
            y += noise

        elif signal_type == "bp":
            systolic_base = kwargs.get("systolic_base", 120)
            diastolic_base = kwargs.get("diastolic_base", 80)
            noise_level = kwargs.get("noise_level", 5)

            # Simulate blood pressure with cardiac cycle
            cardiac_cycle = np.sin(2 * np.pi * t / 100)  # ~60 bpm
            systolic_pressure = systolic_base + 10 * np.maximum(cardiac_cycle, 0)
            diastolic_pressure = diastolic_base + 5 * np.minimum(cardiac_cycle, 0)

            # Combine and add noise
            y = (systolic_pressure + diastolic_pressure) / 2
            noise = self.random_state.normal(0, noise_level, len(t))
            y += noise

        elif signal_type == "resp":
            base_rate = kwargs.get("base_rate", 16)
            variability = kwargs.get("variability", 2)
            noise_level = kwargs.get("noise_level", 0.5)

            # Respiratory rate with natural variability
            resp_rate = base_rate + variability * np.sin(2 * np.pi * t / 500)
            y = np.sin(2 * np.pi * resp_rate * t / 3600)

            # Add noise
            noise = self.random_state.normal(0, noise_level, len(t))
            y += noise

        return y.reshape(-1, 1)

    def _generate_industrial_series(
            self,
            machine_type: str,
            **kwargs: Any
    ) -> np.ndarray:
        """Generate industrial sensor/machine patterns."""
        t = np.arange(self.config.n_samples)

        if machine_type == "motor":
            degradation_rate = kwargs.get("degradation_rate", 0.001)
            fault_prob = kwargs.get("fault_prob", 0.01)

            # Base vibration level with degradation
            base_vibration = 1.0 + degradation_rate * t

            # Normal operating vibration
            normal_vib = base_vibration * (1 + 0.1 * np.sin(2 * np.pi * t / 100))

            # Add faults
            faults = self.random_state.binomial(1, fault_prob, len(t))
            fault_amplitude = self.random_state.exponential(2, len(t))

            y = normal_vib + faults * fault_amplitude

        elif machine_type == "hvac":
            efficiency_trend = kwargs.get("efficiency_trend", -0.0005)
            seasonal_periods = kwargs.get("seasonal_periods", [24, 168])

            # Base energy consumption with efficiency trend
            base_consumption = 100 * (1 + efficiency_trend * t)

            # Seasonal patterns (daily and weekly)
            seasonal = 0
            for period in seasonal_periods:
                seasonal += 20 * np.sin(2 * np.pi * t / period)

            # Operating mode switches
            mode_switches = self.random_state.binomial(1, 0.01, len(t))
            mode_changes = np.cumsum(mode_switches) % 3
            mode_factor = 1 + 0.3 * mode_changes

            y = base_consumption + seasonal
            y = y * mode_factor

        elif machine_type == "production":
            target_rate = kwargs.get("target_rate", 100)
            maintenance_cycles = kwargs.get("maintenance_cycles", [720, 4320])
            noise_level = kwargs.get("noise_level", 5)

            # Base production rate
            y = np.full(len(t), target_rate, dtype=float)

            # Maintenance cycles (performance drops periodically)
            for cycle in maintenance_cycles:
                maintenance_times = t % cycle
                performance_drop = 20 * np.exp(-(maintenance_times / 50) ** 2)
                y -= performance_drop

            # Random equipment issues
            issues = self.random_state.binomial(1, 0.005, len(t))
            issue_impact = self.random_state.exponential(15, len(t))
            y -= issues * issue_impact

            # Add noise and ensure non-negative
            noise = self.random_state.normal(0, noise_level, len(t))
            y = np.maximum(y + noise, 0)

        elif machine_type == "sensor":
            operating_temp = kwargs.get("operating_temp", 85)
            drift_rate = kwargs.get("drift_rate", 0.01)
            spike_prob = kwargs.get("spike_prob", 0.005)

            # Base temperature with drift
            y = operating_temp + drift_rate * t

            # Add thermal cycles
            thermal_cycle = 5 * np.sin(2 * np.pi * t / 200)
            y += thermal_cycle

            # Temperature spikes
            spikes = self.random_state.binomial(1, spike_prob, len(t))
            spike_magnitude = self.random_state.exponential(10, len(t))
            y += spikes * spike_magnitude

            # Sensor noise
            noise = self.random_state.normal(0, 1, len(t))
            y += noise

        return y.reshape(-1, 1)