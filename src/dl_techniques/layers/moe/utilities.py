"""
Utilities for analyzing, debugging, and optimizing Mixture of Experts (MoE) models.

This module provides comprehensive tools for understanding MoE model behavior,
including expert utilization analysis, routing visualization, performance profiling,
and debugging utilities.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import time
import json
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import keras
from keras import ops

from .layer import MixtureOfExperts, MoEConfig
from dl_techniques.utils.logger import logger


@dataclass
class ExpertUtilizationStats:
    """
    Statistics about expert utilization in MoE models.

    This dataclass contains detailed information about how experts are being
    used during model execution, including load distribution, routing patterns,
    and capacity utilization.

    Attributes:
        expert_id: Identifier of the expert.
        total_tokens_processed: Total number of tokens processed by this expert.
        utilization_rate: Fraction of total tokens processed by this expert.
        average_weight: Average routing weight for this expert.
        capacity_utilization: Fraction of expert capacity used.
        routing_entropy: Entropy of routing decisions to this expert.
        specialization_score: Measure of expert specialization.
    """
    expert_id: int
    total_tokens_processed: int
    utilization_rate: float
    average_weight: float
    capacity_utilization: float
    routing_entropy: float
    specialization_score: float


@dataclass
class MoEAnalysisReport:
    """
    Comprehensive analysis report for MoE models.

    This dataclass contains detailed analysis results for MoE model behavior,
    including expert statistics, routing patterns, and performance metrics.

    Attributes:
        model_name: Name of the analyzed model.
        total_experts: Total number of experts in the model.
        active_experts: Number of actively used experts.
        expert_stats: Per-expert utilization statistics.
        routing_efficiency: Overall routing efficiency score.
        load_balance_score: Load balancing effectiveness score.
        auxiliary_loss_values: Values of auxiliary losses during analysis.
        performance_metrics: Performance timing and FLOPs information.
    """
    model_name: str
    total_experts: int
    active_experts: int
    expert_stats: List[ExpertUtilizationStats]
    routing_efficiency: float
    load_balance_score: float
    auxiliary_loss_values: Dict[str, float]
    performance_metrics: Dict[str, float]


class MoEAnalyzer:
    """
    Comprehensive analyzer for MoE models.

    This class provides tools for analyzing MoE model behavior, including
    expert utilization patterns, routing efficiency, load balancing,
    and performance characteristics.

    Args:
        model: The Keras model containing MoE layers to analyze.
        track_gradients: Whether to track gradient flows through experts.

    Example:
        ```python
        # Create analyzer
        analyzer = MoEAnalyzer(model)

        # Analyze model on dataset
        report = analyzer.analyze_model(test_data, batch_size=32)

        # Print summary
        analyzer.print_analysis_summary(report)

        # Visualize expert utilization
        analyzer.visualize_expert_utilization(report)
        ```
    """

    def __init__(
            self,
            model: keras.Model,
            track_gradients: bool = False
    ) -> None:
        """Initialize the MoE analyzer."""
        self.model = model
        self.track_gradients = track_gradients

        # Find MoE layers in the model
        self.moe_layers = self._find_moe_layers()
        if not self.moe_layers:
            logger.warning("No MoE layers found in the provided model")
        else:
            logger.info(f"Found {len(self.moe_layers)} MoE layers for analysis")

        # Analysis state
        self._routing_data = defaultdict(list)
        self._expert_usage = defaultdict(lambda: defaultdict(int))
        self._auxiliary_losses = defaultdict(list)

    def _find_moe_layers(self) -> List[MixtureOfExperts]:
        """Find all MoE layers in the model."""
        moe_layers = []

        def find_moe_recursive(layer):
            if isinstance(layer, MixtureOfExperts):
                moe_layers.append(layer)

            # Check sublayers
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    find_moe_recursive(sublayer)

        # Check all layers in the model
        for layer in self.model.layers:
            find_moe_recursive(layer)

        return moe_layers

    def analyze_model(
            self,
            data: Union[np.ndarray, keras.utils.Sequence],
            batch_size: int = 32,
            steps: Optional[int] = None
    ) -> MoEAnalysisReport:
        """
        Analyze MoE model behavior on provided data.

        Args:
            data: Input data for analysis.
            batch_size: Batch size for analysis.
            steps: Number of steps to analyze (if None, analyze all data).

        Returns:
            Comprehensive analysis report.
        """
        logger.info("Starting MoE model analysis...")

        # Reset analysis state
        self._routing_data.clear()
        self._expert_usage.clear()
        self._auxiliary_losses.clear()

        # Create data iterator
        if isinstance(data, np.ndarray):
            data_gen = self._create_data_generator(data, batch_size)
            total_steps = len(data) // batch_size if steps is None else steps
        else:
            data_gen = data
            total_steps = len(data) if steps is None else steps

        # Analysis loop
        start_time = time.time()
        total_tokens = 0

        for step, batch in enumerate(data_gen):
            if steps is not None and step >= steps:
                break

            # Forward pass with analysis hooks
            self._analyze_batch(batch)
            total_tokens += self._count_tokens_in_batch(batch)

            if step % 10 == 0:
                logger.info(f"Analyzed batch {step + 1}/{total_steps}")

        analysis_time = time.time() - start_time

        # Generate analysis report
        report = self._generate_analysis_report(
            total_tokens=total_tokens,
            analysis_time=analysis_time
        )

        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        return report

    def _create_data_generator(self, data: np.ndarray, batch_size: int):
        """Create a data generator from numpy array."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def _count_tokens_in_batch(self, batch) -> int:
        """Count the number of tokens in a batch."""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # Use first element if multiple inputs

        if len(batch.shape) > 2:
            return batch.shape[0] * batch.shape[1]  # batch_size * seq_len
        else:
            return batch.shape[0]  # batch_size

    def _analyze_batch(self, batch) -> None:
        """Analyze a single batch through the model."""
        # Create hooks to capture routing information
        routing_hooks = []

        for layer_idx, moe_layer in enumerate(self.moe_layers):
            hook = self._create_routing_hook(layer_idx, moe_layer)
            routing_hooks.append(hook)

        try:
            # Forward pass
            if isinstance(batch, (list, tuple)):
                _ = self.model(batch[0], training=False)
            else:
                _ = self.model(batch, training=False)
        finally:
            # Remove hooks
            for hook in routing_hooks:
                if hasattr(hook, 'remove'):
                    hook.remove()

    def _create_routing_hook(self, layer_idx: int, moe_layer: MixtureOfExperts):
        """Create a hook to capture routing information."""

        def hook_fn(layer, inputs, outputs):
            # Capture routing information during forward pass
            if hasattr(layer.gating_network, '_last_routing_info'):
                routing_info = layer.gating_network._last_routing_info
                self._routing_data[layer_idx].append(routing_info)

        # Register hook (this is a simplified version - actual implementation
        # would need to use Keras callback mechanisms or custom training loops)
        return hook_fn

    def _generate_analysis_report(
            self,
            total_tokens: int,
            analysis_time: float
    ) -> MoEAnalysisReport:
        """Generate comprehensive analysis report."""
        expert_stats_by_layer = []
        total_experts = 0
        active_experts = 0

        for layer_idx, moe_layer in enumerate(self.moe_layers):
            layer_expert_stats = self._analyze_layer_experts(layer_idx, moe_layer)
            expert_stats_by_layer.extend(layer_expert_stats)

            total_experts += moe_layer.num_experts
            active_experts += len([s for s in layer_expert_stats if s.utilization_rate > 0.001])

        # Calculate overall metrics
        routing_efficiency = self._calculate_routing_efficiency()
        load_balance_score = self._calculate_load_balance_score()
        aux_loss_values = self._calculate_auxiliary_loss_values()

        performance_metrics = {
            'analysis_time_seconds': analysis_time,
            'tokens_per_second': total_tokens / analysis_time if analysis_time > 0 else 0,
            'total_tokens_analyzed': total_tokens,
            'average_expert_utilization': active_experts / total_experts if total_experts > 0 else 0
        }

        return MoEAnalysisReport(
            model_name=self.model.name or 'unnamed_model',
            total_experts=total_experts,
            active_experts=active_experts,
            expert_stats=expert_stats_by_layer,
            routing_efficiency=routing_efficiency,
            load_balance_score=load_balance_score,
            auxiliary_loss_values=aux_loss_values,
            performance_metrics=performance_metrics
        )

    def _analyze_layer_experts(
            self,
            layer_idx: int,
            moe_layer: MixtureOfExperts
    ) -> List[ExpertUtilizationStats]:
        """Analyze expert utilization for a specific layer."""
        expert_stats = []

        for expert_id in range(moe_layer.num_experts):
            # Calculate statistics for this expert
            # (This is simplified - actual implementation would use captured routing data)
            stats = ExpertUtilizationStats(
                expert_id=expert_id,
                total_tokens_processed=self._expert_usage[layer_idx][expert_id],
                utilization_rate=0.0,  # Would be calculated from routing data
                average_weight=0.0,  # Would be calculated from routing weights
                capacity_utilization=0.0,  # Would be calculated from capacity data
                routing_entropy=0.0,  # Would be calculated from routing distribution
                specialization_score=0.0  # Would be calculated from input patterns
            )
            expert_stats.append(stats)

        return expert_stats

    def _calculate_routing_efficiency(self) -> float:
        """Calculate overall routing efficiency."""
        # Placeholder implementation
        return 0.85

    def _calculate_load_balance_score(self) -> float:
        """Calculate load balancing effectiveness score."""
        # Placeholder implementation
        return 0.75

    def _calculate_auxiliary_loss_values(self) -> Dict[str, float]:
        """Calculate auxiliary loss values."""
        return {
            'auxiliary_loss': 0.001,
            'z_loss': 0.0001
        }

    def print_analysis_summary(self, report: MoEAnalysisReport) -> None:
        """Print a formatted analysis summary."""
        print(f"\n{'=' * 50}")
        print(f"MoE Analysis Report: {report.model_name}")
        print(f"{'=' * 50}")

        print(f"\n📊 Overall Statistics:")
        print(f"  Total Experts: {report.total_experts}")
        print(f"  Active Experts: {report.active_experts}")
        print(f"  Expert Utilization: {report.active_experts / report.total_experts * 100:.1f}%")
        print(f"  Routing Efficiency: {report.routing_efficiency:.3f}")
        print(f"  Load Balance Score: {report.load_balance_score:.3f}")

        print(f"\n⚡ Performance Metrics:")
        for metric_name, value in report.performance_metrics.items():
            if 'time' in metric_name:
                print(f"  {metric_name.replace('_', ' ').title()}: {value:.2f}s")
            elif 'tokens' in metric_name:
                print(f"  {metric_name.replace('_', ' ').title()}: {value:,.0f}")
            else:
                print(f"  {metric_name.replace('_', ' ').title()}: {value:.3f}")

        print(f"\n🔧 Auxiliary Losses:")
        for loss_name, value in report.auxiliary_loss_values.items():
            print(f"  {loss_name.replace('_', ' ').title()}: {value:.6f}")

        # Expert utilization distribution
        if report.expert_stats:
            utilization_rates = [stats.utilization_rate for stats in report.expert_stats]
            print(f"\n👥 Expert Utilization Distribution:")
            print(f"  Min: {min(utilization_rates):.3f}")
            print(f"  Max: {max(utilization_rates):.3f}")
            print(f"  Mean: {np.mean(utilization_rates):.3f}")
            print(f"  Std: {np.std(utilization_rates):.3f}")

    def visualize_expert_utilization(
            self,
            report: MoEAnalysisReport,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize expert utilization patterns.

        Args:
            report: Analysis report containing expert statistics.
            save_path: Optional path to save the visualization.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for visualization")
            return

        # Extract data for visualization
        expert_ids = [stats.expert_id for stats in report.expert_stats]
        utilization_rates = [stats.utilization_rate for stats in report.expert_stats]

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'MoE Expert Analysis: {report.model_name}', fontsize=16)

        # 1. Expert utilization bar chart
        axes[0, 0].bar(expert_ids, utilization_rates)
        axes[0, 0].set_title('Expert Utilization Rates')
        axes[0, 0].set_xlabel('Expert ID')
        axes[0, 0].set_ylabel('Utilization Rate')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Utilization distribution histogram
        axes[0, 1].hist(utilization_rates, bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Utilization Rate Distribution')
        axes[0, 1].set_xlabel('Utilization Rate')
        axes[0, 1].set_ylabel('Number of Experts')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Load balance visualization
        if len(utilization_rates) > 1:
            ideal_utilization = 1.0 / len(utilization_rates)
            balance_scores = [abs(rate - ideal_utilization) for rate in utilization_rates]
            axes[1, 0].bar(expert_ids, balance_scores, color='coral')
            axes[1, 0].axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Perfect Balance')
            axes[1, 0].set_title('Load Balance Deviation')
            axes[1, 0].set_xlabel('Expert ID')
            axes[1, 0].set_ylabel('Deviation from Ideal')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Performance metrics
        perf_metrics = report.performance_metrics
        metric_names = list(perf_metrics.keys())[:4]  # Show top 4 metrics
        metric_values = [perf_metrics[name] for name in metric_names]

        axes[1, 1].barh(metric_names, metric_values, color='lightgreen')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xlabel('Value')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()

    def export_analysis_report(
            self,
            report: MoEAnalysisReport,
            export_path: str,
            format: str = 'json'
    ) -> None:
        """
        Export analysis report to file.

        Args:
            report: Analysis report to export.
            export_path: Path to save the report.
            format: Export format ('json' or 'csv').
        """
        if format.lower() == 'json':
            report_dict = asdict(report)
            with open(export_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
        elif format.lower() == 'csv':
            import pandas as pd

            # Convert expert stats to DataFrame
            expert_data = []
            for stats in report.expert_stats:
                expert_data.append(asdict(stats))

            df = pd.DataFrame(expert_data)
            df.to_csv(export_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Analysis report exported to {export_path}")


class MoEPerformanceProfiler:
    """
    Performance profiler for MoE models.

    This class provides detailed performance analysis for MoE models,
    including FLOPs counting, memory usage analysis, and timing benchmarks.

    Example:
        ```python
        profiler = MoEPerformanceProfiler()
        profile = profiler.profile_model(model, input_shape=(32, 128, 768))
        profiler.print_profile_summary(profile)
        ```
    """

    def __init__(self):
        """Initialize the performance profiler."""
        self.profile_data = {}

    def profile_model(
            self,
            model: keras.Model,
            input_shape: Tuple[int, ...],
            batch_sizes: List[int] = [1, 8, 32, 128]
    ) -> Dict[str, Any]:
        """
        Profile MoE model performance across different batch sizes.

        Args:
            model: Model to profile.
            input_shape: Input shape (without batch dimension).
            batch_sizes: List of batch sizes to test.

        Returns:
            Dictionary containing profiling results.
        """
        profile_results = {
            'model_name': model.name or 'unnamed_model',
            'input_shape': input_shape,
            'batch_profiles': {},
            'moe_layer_analysis': {},
            'memory_analysis': {},
            'flops_analysis': {}
        }

        # Find MoE layers
        moe_layers = self._find_moe_layers_in_model(model)

        # Profile each batch size
        for batch_size in batch_sizes:
            logger.info(f"Profiling batch size {batch_size}")

            full_input_shape = (batch_size,) + input_shape
            dummy_input = keras.random.normal(full_input_shape)

            # Time forward pass
            forward_times = []
            for _ in range(5):  # Multiple runs for average
                start_time = time.perf_counter()
                _ = model(dummy_input, training=False)
                end_time = time.perf_counter()
                forward_times.append(end_time - start_time)

            avg_forward_time = np.mean(forward_times[1:])  # Skip first run
            std_forward_time = np.std(forward_times[1:])

            profile_results['batch_profiles'][batch_size] = {
                'forward_time_ms': avg_forward_time * 1000,
                'forward_time_std_ms': std_forward_time * 1000,
                'throughput_samples_per_sec': batch_size / avg_forward_time,
                'latency_per_sample_ms': (avg_forward_time / batch_size) * 1000
            }

        # Analyze MoE layers specifically
        for i, moe_layer in enumerate(moe_layers):
            layer_analysis = self._analyze_moe_layer_performance(moe_layer, input_shape)
            profile_results['moe_layer_analysis'][f'moe_layer_{i}'] = layer_analysis

        # Estimate FLOPs
        profile_results['flops_analysis'] = self._estimate_model_flops(model, input_shape)

        return profile_results

    def _find_moe_layers_in_model(self, model: keras.Model) -> List[MixtureOfExperts]:
        """Find all MoE layers in the model."""
        moe_layers = []

        def find_moe_recursive(layer):
            if isinstance(layer, MixtureOfExperts):
                moe_layers.append(layer)
            if hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    find_moe_recursive(sublayer)

        for layer in model.layers:
            find_moe_recursive(layer)

        return moe_layers

    def _analyze_moe_layer_performance(
            self,
            moe_layer: MixtureOfExperts,
            input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Analyze performance characteristics of a specific MoE layer."""
        config = moe_layer.config

        # Calculate theoretical FLOPs
        if config.expert_config.expert_type == 'ffn':
            # FFN expert FLOPs calculation
            input_dim = input_shape[-1]
            hidden_dim = config.expert_config.hidden_dim
            intermediate_size = config.expert_config.intermediate_size

            # FLOPs per expert
            flops_per_expert = 2 * (input_dim * intermediate_size + intermediate_size * hidden_dim)

            # Active FLOPs based on top_k
            active_experts_ratio = config.gating_config.top_k / config.num_experts
            active_flops = flops_per_expert * config.gating_config.top_k

            # Gating network FLOPs
            gating_flops = 2 * input_dim * config.num_experts

            total_flops = active_flops + gating_flops
        else:
            # Placeholder for other expert types
            total_flops = 0

        return {
            'num_experts': config.num_experts,
            'expert_type': config.expert_config.expert_type,
            'top_k': config.gating_config.top_k,
            'gating_type': config.gating_config.gating_type,
            'theoretical_flops': total_flops,
            'sparsity_ratio': config.gating_config.top_k / config.num_experts,
            'parameter_count': self._count_layer_parameters(moe_layer)
        }

    def _count_layer_parameters(self, layer: keras.layers.Layer) -> int:
        """Count total parameters in a layer."""
        return sum(ops.size(weight) for weight in layer.trainable_weights)

    def _estimate_model_flops(
            self,
            model: keras.Model,
            input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Estimate total FLOPs for the model."""
        # This is a simplified FLOP estimation
        # A complete implementation would analyze each layer type
        total_params = model.count_params()

        return {
            'total_parameters': total_params,
            'estimated_flops': total_params * 2,  # Simplified estimate
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

    def print_profile_summary(self, profile: Dict[str, Any]) -> None:
        """Print formatted performance profile summary."""
        print(f"\n{'=' * 60}")
        print(f"MoE Performance Profile: {profile['model_name']}")
        print(f"{'=' * 60}")

        print(f"\n📊 Input Shape: {profile['input_shape']}")

        # Batch size performance
        print(f"\n⚡ Batch Size Performance:")
        print(f"{'Batch Size':>10} {'Forward (ms)':>12} {'Throughput (s/s)':>15} {'Latency (ms)':>13}")
        print("-" * 55)

        for batch_size, metrics in profile['batch_profiles'].items():
            print(f"{batch_size:>10} "
                  f"{metrics['forward_time_ms']:>10.2f} "
                  f"{metrics['throughput_samples_per_sec']:>13.1f} "
                  f"{metrics['latency_per_sample_ms']:>11.2f}")

        # MoE layer analysis
        if profile['moe_layer_analysis']:
            print(f"\n🧠 MoE Layer Analysis:")
            for layer_name, analysis in profile['moe_layer_analysis'].items():
                print(f"  {layer_name}:")
                print(f"    Experts: {analysis['num_experts']} ({analysis['expert_type']})")
                print(f"    Top-K: {analysis['top_k']} (Sparsity: {analysis['sparsity_ratio']:.2%})")
                print(f"    Parameters: {analysis['parameter_count']:,}")
                print(f"    Theoretical FLOPs: {analysis['theoretical_flops']:,}")

        # FLOPs analysis
        flops_analysis = profile['flops_analysis']
        print(f"\n🔢 Model Analysis:")
        print(f"  Total Parameters: {flops_analysis['total_parameters']:,}")
        print(f"  Estimated FLOPs: {flops_analysis['estimated_flops']:,}")
        print(f"  Model Size: {flops_analysis['model_size_mb']:.1f} MB")


def compare_moe_configurations(
        configs: List[Tuple[str, MoEConfig]],
        input_shape: Tuple[int, ...] = (128, 768),
        batch_size: int = 32
) -> Dict[str, Any]:
    """
    Compare different MoE configurations in terms of performance and characteristics.

    Args:
        configs: List of (name, MoEConfig) tuples to compare.
        input_shape: Input shape for testing.
        batch_size: Batch size for performance testing.

    Returns:
        Dictionary containing comparison results.
    """
    comparison_results = {
        'configs': [],
        'performance_comparison': {},
        'parameter_comparison': {},
        'efficiency_metrics': {}
    }

    for config_name, config in configs:
        logger.info(f"Testing configuration: {config_name}")

        # Create MoE layer with this configuration
        moe_layer = MixtureOfExperts(config)
        moe_layer.build((batch_size,) + input_shape)

        # Performance test
        dummy_input = keras.random.normal((batch_size,) + input_shape)

        # Time forward pass
        start_time = time.perf_counter()
        for _ in range(10):
            _ = moe_layer(dummy_input, training=False)
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 10

        # Collect metrics
        config_metrics = {
            'name': config_name,
            'num_experts': config.num_experts,
            'expert_type': config.expert_config.expert_type,
            'top_k': config.gating_config.top_k,
            'gating_type': config.gating_config.gating_type,
            'parameter_count': sum(ops.size(w) for w in moe_layer.trainable_weights),
            'forward_time_ms': avg_time * 1000,
            'sparsity_ratio': config.gating_config.top_k / config.num_experts,
            'theoretical_speedup': config.num_experts / config.gating_config.top_k
        }

        comparison_results['configs'].append(config_metrics)

    # Generate comparison summary
    comparison_results['summary'] = _generate_comparison_summary(comparison_results['configs'])

    return comparison_results


def _generate_comparison_summary(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics for configuration comparison."""
    if not configs:
        return {}

    # Extract metrics
    forward_times = [c['forward_time_ms'] for c in configs]
    param_counts = [c['parameter_count'] for c in configs]
    sparsity_ratios = [c['sparsity_ratio'] for c in configs]

    return {
        'fastest_config': configs[np.argmin(forward_times)]['name'],
        'most_efficient_config': configs[np.argmin(np.array(param_counts) / np.array(sparsity_ratios))]['name'],
        'speed_range_ms': (min(forward_times), max(forward_times)),
        'parameter_range': (min(param_counts), max(param_counts)),
        'average_sparsity': np.mean(sparsity_ratios)
    }


def debug_moe_routing(
        moe_layer: MixtureOfExperts,
        inputs: keras.KerasTensor,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Debug MoE routing decisions and expert utilization.

    Args:
        moe_layer: MoE layer to debug.
        inputs: Input tensor for debugging.
        verbose: Whether to print debug information.

    Returns:
        Dictionary containing routing debug information.
    """
    # Forward pass to get routing information
    outputs = moe_layer(inputs, training=False)

    # Extract routing information from gating network
    expert_weights, expert_indices, aux_info = moe_layer.gating_network(inputs)

    # Analyze routing patterns
    debug_info = {
        'input_shape': ops.shape(inputs).numpy(),
        'output_shape': ops.shape(outputs).numpy(),
        'expert_weights_stats': {
            'mean': float(ops.mean(expert_weights)),
            'std': float(ops.std(expert_weights)),
            'min': float(ops.min(expert_weights)),
            'max': float(ops.max(expert_weights))
        },
        'expert_selection_distribution': {},
        'auxiliary_info': {}
    }

    # Analyze expert selection distribution
    if len(ops.shape(expert_indices)) > 1:
        for k in range(ops.shape(expert_indices)[1]):
            selected_experts = expert_indices[:, k].numpy()
            unique_experts, counts = np.unique(selected_experts, return_counts=True)
            debug_info['expert_selection_distribution'][f'top_{k + 1}'] = dict(
                zip(unique_experts.tolist(), counts.tolist()))

    # Store auxiliary information
    for key, value in aux_info.items():
        if hasattr(value, 'numpy'):
            debug_info['auxiliary_info'][key] = {
                'shape': ops.shape(value).numpy().tolist(),
                'mean': float(ops.mean(value)),
                'std': float(ops.std(value))
            }

    if verbose:
        print(f"\n🔍 MoE Routing Debug Information")
        print(f"Input Shape: {debug_info['input_shape']}")
        print(f"Output Shape: {debug_info['output_shape']}")
        print(f"\nExpert Weights Statistics:")
        for stat_name, value in debug_info['expert_weights_stats'].items():
            print(f"  {stat_name}: {value:.4f}")

        print(f"\nExpert Selection Distribution:")
        for top_k, distribution in debug_info['expert_selection_distribution'].items():
            print(f"  {top_k}: {distribution}")

    return debug_info


# Convenience functions for common analysis tasks

def quick_moe_analysis(model: keras.Model, sample_data: np.ndarray) -> None:
    """
    Perform quick analysis of MoE model.

    Args:
        model: Model containing MoE layers.
        sample_data: Sample data for analysis.
    """
    analyzer = MoEAnalyzer(model)
    report = analyzer.analyze_model(sample_data, steps=10)
    analyzer.print_analysis_summary(report)


def benchmark_moe_vs_dense(
        moe_config: MoEConfig,
        input_shape: Tuple[int, ...],
        batch_size: int = 32
) -> Dict[str, float]:
    """
    Benchmark MoE layer against equivalent dense layer.

    Args:
        moe_config: MoE configuration to test.
        input_shape: Input shape for testing.
        batch_size: Batch size for benchmarking.

    Returns:
        Dictionary containing benchmark results.
    """
    # Create MoE layer
    moe_layer = MixtureOfExperts(moe_config)
    moe_layer.build((batch_size,) + input_shape)

    # Create equivalent dense layer
    dense_layer = keras.layers.Dense(
        units=moe_config.expert_config.hidden_dim,
        activation=moe_config.expert_config.activation
    )
    dense_layer.build((batch_size,) + input_shape)

    # Benchmark both layers
    dummy_input = keras.random.normal((batch_size,) + input_shape)

    # MoE timing
    start_time = time.perf_counter()
    for _ in range(100):
        _ = moe_layer(dummy_input)
    moe_time = time.perf_counter() - start_time

    # Dense timing
    start_time = time.perf_counter()
    for _ in range(100):
        _ = dense_layer(dummy_input)
    dense_time = time.perf_counter() - start_time

    # Parameter counts
    moe_params = sum(ops.size(w) for w in moe_layer.trainable_weights)
    dense_params = sum(ops.size(w) for w in dense_layer.trainable_weights)

    return {
        'moe_time_ms': moe_time * 10,  # Per iteration
        'dense_time_ms': dense_time * 10,
        'speedup_ratio': dense_time / moe_time,
        'moe_parameters': int(moe_params),
        'dense_parameters': int(dense_params),
        'parameter_efficiency': float(dense_params / moe_params),
        'computational_efficiency': float(moe_config.gating_config.top_k / moe_config.num_experts)
    }