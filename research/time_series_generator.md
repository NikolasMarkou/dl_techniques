# Generic Time Series Generator

A comprehensive time series generator for creating diverse synthetic time series patterns for machine learning, forecasting, and time series analysis experiments.

## Features

### 🎯 **Pattern Categories**
- **Trend Patterns**: Linear, exponential, polynomial, logistic growth
- **Seasonal Patterns**: Single and multi-period seasonality
- **Composite Patterns**: Trend + seasonal combinations
- **Stochastic Processes**: Random walk, AR, MA, ARMA, mean-reverting
- **Intermittent Patterns**: Sparse and lumpy demand
- **Volatility Clustering**: GARCH models
- **Regime Switching**: Multiple regime patterns
- **Structural Breaks**: Level shifts and trend changes
- **Outlier Patterns**: Additive and innovation outliers
- **Chaotic Systems**: Henon map, Lorenz attractor

### 🔧 **Key Capabilities**
- **25+ predefined patterns** covering major time series phenomena
- **Custom pattern generation** with configurable parameters
- **Random pattern sampling** by category or overall
- **Reproducible results** with configurable random seeds
- **Flexible configuration** for different use cases
- **Comprehensive testing** for reliability
- **Type-safe implementation** with full type hints

## Installation

Place the `time_series_generator.py` file in your `dl_techniques/utils/datasets` directory:

```python
from dl_techniques.utils.datasets,time_series_generator import (
    TimeSeriesGenerator,
    TimeSeriesConfig
)
```

## Quick Start

```python
from dl_techniques.utils.datasets,time_series_generator import TimeSeriesGenerator, TimeSeriesConfig

# Create configuration
config = TimeSeriesConfig(n_samples=1000, random_seed=42)
generator = TimeSeriesGenerator(config)

# Generate a specific pattern
trend_data = generator.generate_task_data("linear_trend_strong")
seasonal_data = generator.generate_task_data("multi_seasonal")

# Generate all patterns
all_patterns = generator.generate_all_patterns()

# Generate random pattern from category
task_name, data = generator.generate_random_pattern(category="stochastic")
```

## Configuration

The `TimeSeriesConfig` class provides comprehensive configuration options:

```python
@dataclass
class TimeSeriesConfig:
    # Basic configuration
    n_samples: int = 1000              # Number of time steps
    random_seed: int = 42              # Random seed for reproducibility
    default_noise_level: float = 0.1   # Default noise level
    
    # Pattern-specific parameters
    trend_strengths: Tuple[float, float] = (0.0001, 0.005)
    seasonal_periods: List[int] = [12, 24, 48, 96, 168, 336]
    seasonal_amplitudes: Tuple[float, float] = (0.5, 2.0)
    # ... more parameters
```

## Available Patterns

### Trend Patterns
- `linear_trend_strong` / `linear_trend_weak` - Linear trends with different strengths
- `exponential_growth` - Exponential growth pattern
- `polynomial_trend` - Polynomial trend with configurable coefficients
- `logistic_growth` - S-shaped logistic growth curve

### Seasonal Patterns
- `daily_seasonality` - 24-hour seasonal pattern
- `weekly_seasonality` - 7-day seasonal pattern
- `multi_seasonal` - Multiple overlapping seasonal patterns
- `complex_seasonal` - Complex multi-period seasonality

### Composite Patterns
- `trend_daily_seasonal` - Linear trend + daily seasonality
- `trend_weekly_seasonal` - Linear trend + weekly seasonality
- `exp_trend_multi_seasonal` - Exponential trend + multiple seasonalities

### Stochastic Processes
- `random_walk` - Random walk with drift
- `ar_process` - Autoregressive process
- `ma_process` - Moving average process
- `arma_process` - ARMA process
- `mean_reverting` - Ornstein-Uhlenbeck process

### Advanced Patterns
- `intermittent_demand` / `lumpy_demand` - Intermittent patterns
- `garch_low_vol` / `garch_high_vol` - Volatility clustering
- `regime_switching` - Multiple regime patterns
- `level_shift` / `trend_change` - Structural breaks
- `additive_outliers` / `innovation_outliers` - Outlier patterns
- `henon_map` / `lorenz_x` - Chaotic systems

## Usage Examples

### 1. Basic Pattern Generation

```python
# Create generator
config = TimeSeriesConfig(n_samples=500, random_seed=42)
generator = TimeSeriesGenerator(config)

# Generate specific patterns
trend_data = generator.generate_task_data("linear_trend_strong")
seasonal_data = generator.generate_task_data("multi_seasonal")
stochastic_data = generator.generate_task_data("arma_process")
```

### 2. Custom Pattern Generation

```python
# Custom trend pattern
custom_trend = generator.generate_custom_pattern(
    "trend",
    trend_type="polynomial",
    coefficients=[0, 0.002, -1e-6],
    noise_level=0.05
)

# Custom seasonal pattern
custom_seasonal = generator.generate_custom_pattern(
    "seasonal",
    periods=[24, 168],
    amplitudes=[1.0, 0.8],
    noise_level=0.1
)

# Custom stochastic process
custom_ar = generator.generate_custom_pattern(
    "stochastic",
    process_type="ar",
    ar_coeffs=[0.8, -0.3],
    noise_std=0.1
)
```

### 3. Random Pattern Sampling

```python
# Random pattern from any category
task_name, data = generator.generate_random_pattern()

# Random pattern from specific category
trend_task, trend_data = generator.generate_random_pattern(category="trend")
seasonal_task, seasonal_data = generator.generate_random_pattern(category="seasonal")
```

### 4. Multi-Task Dataset Creation

```python
# Create dataset for multiple tasks
selected_tasks = ["linear_trend_strong", "multi_seasonal", "arma_process"]
dataset = {}

for task_name in selected_tasks:
    task_data = []
    for i in range(100):  # Generate 100 series per task
        config = TimeSeriesConfig(n_samples=200, random_seed=42 + i)
        gen = TimeSeriesGenerator(config)
        series = gen.generate_task_data(task_name)
        task_data.append(series)
    dataset[task_name] = np.array(task_data)
```

### 5. Pattern Analysis

```python
# Analyze pattern properties
patterns = ["linear_trend_strong", "daily_seasonality", "random_walk"]

for pattern_name in patterns:
    data = generator.generate_task_data(pattern_name)
    flat_data = data.flatten()
    
    print(f"{pattern_name}:")
    print(f"  Mean: {np.mean(flat_data):.4f}")
    print(f"  Std:  {np.std(flat_data):.4f}")
    print(f"  Autocorr(1): {np.corrcoef(flat_data[:-1], flat_data[1:])[0,1]:.4f}")
```

## API Reference

### TimeSeriesGenerator

#### Methods

- `get_task_names() -> List[str]` - Get all available task names
- `get_task_categories() -> List[str]` - Get all pattern categories
- `get_tasks_by_category(category: str) -> List[str]` - Get tasks by category
- `generate_task_data(task_name: str) -> np.ndarray` - Generate specific pattern
- `generate_all_patterns() -> Dict[str, np.ndarray]` - Generate all patterns
- `generate_random_pattern(category: Optional[str] = None) -> Tuple[str, np.ndarray]` - Generate random pattern
- `generate_custom_pattern(pattern_type: str, **kwargs) -> np.ndarray` - Generate custom pattern

### TimeSeriesConfig

#### Key Parameters

- `n_samples: int` - Number of time steps in each series
- `random_seed: int` - Random seed for reproducibility
- `default_noise_level: float` - Default noise level for patterns
- `trend_strengths: Tuple[float, float]` - Range of trend strengths
- `seasonal_periods: List[int]` - Available seasonal periods
- `seasonal_amplitudes: Tuple[float, float]` - Range of seasonal amplitudes

## Integration with DL-Techniques

The generator integrates seamlessly with the dl-techniques framework:

```python
# Use with forecasting models
from dl_techniques.models.nbeats import NBeatsNet
from dl_techniques.utils.time_series_generator import TimeSeriesGenerator

# Generate training data
config = TimeSeriesConfig(n_samples=1000)
generator = TimeSeriesGenerator(config)
training_data = generator.generate_all_patterns()

# Train model on synthetic data
model = NBeatsNet(...)
# ... training code
```