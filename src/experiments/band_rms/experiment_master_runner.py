"""
Master Experiment Runner: Comprehensive BandRMS Evaluation

This script orchestrates all three BandRMS experiments and provides
comprehensive analysis and comparison across different architectures
and domains.

Usage:
    python master_experiment.py --experiments all
    python master_experiment.py --experiments transformer,cnn
    python master_experiment.py --experiments mlp --task regression
    python master_experiment.py --analyze-only --results-dir experiments/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

from dl_techniques.utils.logger import logger
from .experiment_1_transformer_language_model import TransformerExperiment
from .experiment_2_cnn_image_classificationl import CNNExperiment
from .experiment_3_deep_mlp_tabular_data import MLPExperiment


class MasterExperimentRunner:
    """
    Master experiment runner for comprehensive BandRMS evaluation.
    """

    def __init__(self, base_output_dir: str = 'experiments'):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f'bandrms_comprehensive_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize experiment runners
        self.experiments = {
            'transformer': TransformerExperiment(
                output_dir=str(self.run_dir / 'transformer')
            ),
            'cnn': CNNExperiment(
                output_dir=str(self.run_dir / 'cnn')
            ),
            'mlp': MLPExperiment(
                output_dir=str(self.run_dir / 'mlp')
            )
        }

        # Common normalization configurations across experiments
        self.normalization_configs = {
            'band_rms_0.05': 'BandRMS (width=0.05)',
            'band_rms_0.1': 'BandRMS (width=0.1)',
            'band_rms_0.2': 'BandRMS (width=0.2)',
            'rms_norm': 'RMSNorm',
            'layer_norm': 'LayerNorm',
            'batch_norm': 'BatchNorm',
            'no_norm': 'No Normalization'
        }

    def run_experiments(
            self,
            experiment_names: List[str],
            mlp_task: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Run specified experiments.

        Args:
            experiment_names: List of experiments to run ['transformer', 'cnn', 'mlp']
            mlp_task: Task type for MLP experiment ('classification' or 'regression')

        Returns:
            Dictionary with all experiment results
        """
        logger.info(f"Starting comprehensive BandRMS evaluation")
        logger.info(f"Running experiments: {experiment_names}")
        logger.info(f"Results will be saved to: {self.run_dir}")

        all_results = {}

        for exp_name in experiment_names:
            logger.info(f"{'=' * 60}")
            logger.info(f"RUNNING {exp_name.upper()} EXPERIMENT")
            logger.info(f"{'=' * 60}")

            try:
                if exp_name == 'transformer':
                    results = self.experiments[exp_name].run_experiment()
                elif exp_name == 'cnn':
                    results = self.experiments[exp_name].run_experiment()
                elif exp_name == 'mlp':
                    results = self.experiments[exp_name].run_experiment(task_type=mlp_task)
                else:
                    logger.error(f"Unknown experiment: {exp_name}")
                    continue

                all_results[exp_name] = results
                logger.info(f"✅ {exp_name.upper()} experiment completed successfully")

            except Exception as e:
                logger.error(f"❌ {exp_name.upper()} experiment failed: {str(e)}")
                all_results[exp_name] = {'error': str(e)}

        # Save combined results
        combined_results_path = self.run_dir / 'all_experiments_results.json'
        with open(combined_results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"All experiment results saved to: {combined_results_path}")

        return all_results

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis across all experiments.

        Args:
            results: Combined results from all experiments

        Returns:
            Analysis summary
        """
        logger.info("Performing comprehensive cross-experiment analysis")

        analysis = {
            'summary': {},
            'best_configs': {},
            'stability_analysis': {},
            'band_parameter_analysis': {},
            'architecture_sensitivity': {}
        }

        # Extract performance metrics across experiments
        performance_data = []

        for exp_name, exp_results in results.items():
            if 'error' in exp_results:
                continue

            if exp_name == 'transformer':
                # Single configuration experiment
                training_results = exp_results.get('training_results', {})
                for config_name, result in training_results.items():
                    if 'error' not in result:
                        performance_data.append({
                            'experiment': 'Transformer',
                            'config': config_name,
                            'test_loss': result['test_loss'],
                            'test_accuracy': result['test_accuracy'],
                            'epochs': result['epochs_trained'],
                            'converged': result['epochs_trained'] < 20
                        })

            elif exp_name == 'cnn':
                # Multiple batch sizes
                for batch_key, batch_results in exp_results.items():
                    if 'batch_size_' in batch_key:
                        batch_size = batch_key.split('_')[2]
                        training_results = batch_results.get('training_results', {})

                        for config_name, result in training_results.items():
                            if 'error' not in result:
                                performance_data.append({
                                    'experiment': f'CNN (BS={batch_size})',
                                    'config': config_name,
                                    'test_loss': result['test_loss'],
                                    'test_accuracy': result['test_accuracy'],
                                    'epochs': result['epochs_trained'],
                                    'converged': result['epochs_trained'] < 30,
                                    'batch_size': int(batch_size)
                                })

            elif exp_name == 'mlp':
                # Multiple depths
                for depth_name, depth_results in exp_results.items():
                    training_results = depth_results.get('training_results', {})

                    for config_name, result in training_results.items():
                        if 'error' not in result:
                            performance_data.append({
                                'experiment': f'MLP ({depth_name})',
                                'config': config_name,
                                'test_loss': result['test_loss'],
                                'test_accuracy': result['test_metric'],
                                'epochs': result['epochs_trained'],
                                'converged': result['converged'],
                                'depth': result['depth'],
                                'parameters': result['total_parameters']
                            })

        # Convert to DataFrame for analysis
        df = pd.DataFrame(performance_data)

        if len(df) > 0:
            # Overall performance summary
            analysis['summary'] = self._create_performance_summary(df)

            # Best configurations per experiment
            analysis['best_configs'] = self._find_best_configurations(df)

            # Stability analysis
            analysis['stability_analysis'] = self._analyze_training_stability(df)

            # Architecture sensitivity
            analysis['architecture_sensitivity'] = self._analyze_architecture_sensitivity(df)

            # Band parameter analysis (for BandRMS variants)
            analysis['band_parameter_analysis'] = self._analyze_band_parameters(results)

        # Save analysis
        analysis_path = self.run_dir / 'comprehensive_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"Comprehensive analysis saved to: {analysis_path}")

        return analysis

    def _create_performance_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create overall performance summary."""

        summary = {}

        # Group by configuration
        config_groups = df.groupby('config')

        summary['by_configuration'] = {}
        for config, group in config_groups:
            summary['by_configuration'][config] = {
                'mean_test_loss': float(group['test_loss'].mean()),
                'std_test_loss': float(group['test_loss'].std()),
                'mean_test_accuracy': float(group['test_accuracy'].mean()),
                'std_test_accuracy': float(group['test_accuracy'].std()),
                'success_rate': float(group['converged'].mean()),
                'experiments_count': len(group)
            }

        # Group by experiment type
        exp_groups = df.groupby('experiment')

        summary['by_experiment'] = {}
        for exp, group in exp_groups:
            best_config = group.loc[group['test_accuracy'].idxmax(), 'config']
            summary['by_experiment'][exp] = {
                'best_config': best_config,
                'best_accuracy': float(group['test_accuracy'].max()),
                'mean_accuracy': float(group['test_accuracy'].mean()),
                'config_count': len(group['config'].unique())
            }

        return summary

    def _find_best_configurations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find best performing configurations."""

        best_configs = {}

        # Overall best configuration
        best_overall = df.loc[df['test_accuracy'].idxmax()]
        best_configs['overall'] = {
            'config': best_overall['config'],
            'experiment': best_overall['experiment'],
            'test_accuracy': float(best_overall['test_accuracy']),
            'test_loss': float(best_overall['test_loss'])
        }

        # Best per experiment type
        best_configs['per_experiment'] = {}
        for exp in df['experiment'].unique():
            exp_df = df[df['experiment'] == exp]
            best_exp = exp_df.loc[exp_df['test_accuracy'].idxmax()]

            best_configs['per_experiment'][exp] = {
                'config': best_exp['config'],
                'test_accuracy': float(best_exp['test_accuracy']),
                'test_loss': float(best_exp['test_loss'])
            }

        # Best BandRMS configuration
        bandrms_df = df[df['config'].str.contains('band_rms')]
        if len(bandrms_df) > 0:
            best_bandrms = bandrms_df.loc[bandrms_df['test_accuracy'].idxmax()]
            best_configs['best_bandrms'] = {
                'config': best_bandrms['config'],
                'experiment': best_bandrms['experiment'],
                'test_accuracy': float(best_bandrms['test_accuracy']),
                'test_loss': float(best_bandrms['test_loss'])
            }

        return best_configs

    def _analyze_training_stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze training stability across configurations."""

        stability = {}

        # Convergence rate by configuration
        config_convergence = df.groupby('config')['converged'].mean()
        stability['convergence_rates'] = config_convergence.to_dict()

        # Training efficiency (epochs to convergence)
        converged_df = df[df['converged'] == True]
        if len(converged_df) > 0:
            config_epochs = converged_df.groupby('config')['epochs'].mean()
            stability['mean_epochs_to_convergence'] = config_epochs.to_dict()

        # Performance consistency (std of test accuracy)
        config_std = df.groupby('config')['test_accuracy'].std()
        stability['performance_consistency'] = config_std.to_dict()

        return stability

    def _analyze_architecture_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how configurations perform across different architectures."""

        sensitivity = {}

        # Performance variance across experiments for each config
        config_variance = {}
        for config in df['config'].unique():
            config_df = df[df['config'] == config]
            if len(config_df) > 1:
                config_variance[config] = {
                    'accuracy_std': float(config_df['test_accuracy'].std()),
                    'accuracy_range': float(config_df['test_accuracy'].max() - config_df['test_accuracy'].min()),
                    'experiments': config_df['experiment'].tolist()
                }

        sensitivity['config_variance'] = config_variance

        # Ranking consistency (how consistent are the rankings across experiments)
        experiment_rankings = {}
        for exp in df['experiment'].unique():
            exp_df = df[df['experiment'] == exp]
            exp_df_sorted = exp_df.sort_values('test_accuracy', ascending=False)
            experiment_rankings[exp] = exp_df_sorted['config'].tolist()

        sensitivity['rankings'] = experiment_rankings

        return sensitivity

    def _analyze_band_parameters(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learned band parameters across experiments."""

        band_analysis = {}

        for exp_name, exp_results in results.items():
            if 'error' in exp_results:
                continue

            exp_band_data = {}

            if exp_name == 'mlp':
                # MLP has band parameters saved per depth
                for depth_name, depth_results in exp_results.items():
                    if 'band_parameters' in depth_results:
                        for config_name, params in depth_results['band_parameters'].items():
                            if 'band_rms' in config_name and params:
                                exp_band_data[f'{depth_name}_{config_name}'] = {
                                    'mean_param': float(np.mean(list(params.values()))),
                                    'std_param': float(np.std(list(params.values()))),
                                    'min_param': float(np.min(list(params.values()))),
                                    'max_param': float(np.max(list(params.values()))),
                                    'layer_count': len(params)
                                }

            if exp_band_data:
                band_analysis[exp_name] = exp_band_data

        return band_analysis

    def create_comprehensive_plots(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Create comprehensive visualization plots."""

        logger.info("Creating comprehensive visualization plots")

        plots_dir = self.run_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("Set2")

        # 1. Overall performance comparison
        self._plot_overall_performance(results, plots_dir)

        # 2. Configuration ranking across experiments
        self._plot_configuration_rankings(analysis, plots_dir)

        # 3. Training stability comparison
        self._plot_training_stability(analysis, plots_dir)

        # 4. Band parameter analysis
        self._plot_band_parameters(analysis, plots_dir)

        # 5. Architecture sensitivity analysis
        self._plot_architecture_sensitivity(analysis, plots_dir)

        logger.info(f"All plots saved to: {plots_dir}")

    def _plot_overall_performance(self, results: Dict[str, Any], plots_dir: Path):
        """Plot overall performance comparison."""

        # Extract data for plotting
        plot_data = []

        for exp_name, exp_results in results.items():
            if 'error' in exp_results:
                continue

            if exp_name == 'transformer':
                training_results = exp_results.get('training_results', {})
                for config_name, result in training_results.items():
                    if 'error' not in result:
                        plot_data.append({
                            'Experiment': 'Transformer',
                            'Configuration': self.normalization_configs.get(config_name, config_name),
                            'Test Accuracy': result['test_accuracy'],
                            'Test Loss': result['test_loss']
                        })

        if not plot_data:
            return

        df_plot = pd.DataFrame(plot_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Test accuracy comparison
        sns.barplot(data=df_plot, x='Configuration', y='Test Accuracy', hue='Experiment', ax=ax1)
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylabel('Test Accuracy')
        ax1.tick_params(axis='x', rotation=45)

        # Test loss comparison
        sns.barplot(data=df_plot, x='Configuration', y='Test Loss', hue='Experiment', ax=ax2)
        ax2.set_title('Test Loss Comparison')
        ax2.set_ylabel('Test Loss')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(plots_dir / 'overall_performance.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_configuration_rankings(self, analysis: Dict[str, Any], plots_dir: Path):
        """Plot configuration rankings across experiments."""

        if 'architecture_sensitivity' not in analysis or 'rankings' not in analysis['architecture_sensitivity']:
            return

        rankings = analysis['architecture_sensitivity']['rankings']

        # Create ranking matrix
        all_configs = set()
        for exp_ranking in rankings.values():
            all_configs.update(exp_ranking)

        all_configs = sorted(list(all_configs))
        ranking_matrix = np.zeros((len(rankings), len(all_configs)))

        for i, (exp, ranking) in enumerate(rankings.items()):
            for j, config in enumerate(all_configs):
                if config in ranking:
                    ranking_matrix[i, j] = ranking.index(config) + 1
                else:
                    ranking_matrix[i, j] = len(ranking) + 1

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            ranking_matrix,
            xticklabels=[self.normalization_configs.get(c, c) for c in all_configs],
            yticklabels=list(rankings.keys()),
            annot=True,
            fmt='.0f',
            cmap='RdYlGn_r',
            ax=ax
        )

        ax.set_title('Configuration Rankings Across Experiments\n(Lower numbers = better performance)')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Experiment')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(plots_dir / 'configuration_rankings.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_training_stability(self, analysis: Dict[str, Any], plots_dir: Path):
        """Plot training stability analysis."""

        if 'stability_analysis' not in analysis:
            return

        stability = analysis['stability_analysis']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Convergence rates
        if 'convergence_rates' in stability:
            conv_rates = stability['convergence_rates']
            configs = list(conv_rates.keys())
            rates = list(conv_rates.values())

            bars = ax1.bar(configs, rates, color=plt.cm.Set2(np.arange(len(configs))))
            ax1.set_title('Convergence Rates by Configuration')
            ax1.set_ylabel('Convergence Rate')
            ax1.set_ylim(0, 1)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

            # Add value labels
            for bar, rate in zip(bars, rates):
                ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                         f'{rate:.2f}', ha='center', va='bottom')

        # Training efficiency
        if 'mean_epochs_to_convergence' in stability:
            epochs_data = stability['mean_epochs_to_convergence']
            configs = list(epochs_data.keys())
            epochs = list(epochs_data.values())

            bars = ax2.bar(configs, epochs, color=plt.cm.Set2(np.arange(len(configs))))
            ax2.set_title('Mean Epochs to Convergence')
            ax2.set_ylabel('Epochs')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

            # Add value labels
            for bar, epoch in zip(bars, epochs):
                ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                         f'{epoch:.1f}', ha='center', va='bottom')

        # Performance consistency
        if 'performance_consistency' in stability:
            consistency = stability['performance_consistency']
            configs = list(consistency.keys())
            stds = list(consistency.values())

            bars = ax3.bar(configs, stds, color=plt.cm.Set2(np.arange(len(configs))))
            ax3.set_title('Performance Consistency (Lower = Better)')
            ax3.set_ylabel('Std of Test Accuracy')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

        # Overall stability score
        ax4.axis('off')
        stability_text = "Stability Analysis Summary:\n\n"

        if 'convergence_rates' in stability:
            best_convergence = max(stability['convergence_rates'], key=stability['convergence_rates'].get)
            stability_text += f"Best Convergence: {best_convergence}\n"

        if 'mean_epochs_to_convergence' in stability:
            fastest_convergence = min(stability['mean_epochs_to_convergence'],
                                      key=stability['mean_epochs_to_convergence'].get)
            stability_text += f"Fastest Training: {fastest_convergence}\n"

        if 'performance_consistency' in stability:
            most_consistent = min(stability['performance_consistency'],
                                  key=stability['performance_consistency'].get)
            stability_text += f"Most Consistent: {most_consistent}\n"

        ax4.text(0.1, 0.9, stability_text, transform=ax4.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))

        plt.tight_layout()
        plt.savefig(plots_dir / 'training_stability.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_band_parameters(self, analysis: Dict[str, Any], plots_dir: Path):
        """Plot band parameter analysis."""

        if 'band_parameter_analysis' not in analysis:
            return

        band_data = analysis['band_parameter_analysis']

        if not band_data:
            return

        # Collect all band parameter data
        plot_data = []
        for exp_name, exp_data in band_data.items():
            for config_variant, params in exp_data.items():
                plot_data.append({
                    'Experiment': exp_name,
                    'Configuration': config_variant,
                    'Mean Parameter': params['mean_param'],
                    'Std Parameter': params['std_param'],
                    'Layer Count': params['layer_count']
                })

        if not plot_data:
            return

        df_band = pd.DataFrame(plot_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Mean band parameters
        sns.barplot(data=df_band, x='Configuration', y='Mean Parameter', hue='Experiment', ax=ax1)
        ax1.set_title('Learned Band Parameters (Mean)')
        ax1.set_ylabel('Band Parameter Value')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Parameter variability
        sns.scatterplot(data=df_band, x='Layer Count', y='Std Parameter',
                        hue='Experiment', size='Mean Parameter', ax=ax2)
        ax2.set_title('Band Parameter Variability vs Network Depth')
        ax2.set_xlabel('Number of Layers')
        ax2.set_ylabel('Std of Band Parameters')

        plt.tight_layout()
        plt.savefig(plots_dir / 'band_parameters.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_architecture_sensitivity(self, analysis: Dict[str, Any], plots_dir: Path):
        """Plot architecture sensitivity analysis."""

        if 'architecture_sensitivity' not in analysis or 'config_variance' not in analysis['architecture_sensitivity']:
            return

        config_variance = analysis['architecture_sensitivity']['config_variance']

        if not config_variance:
            return

        # Extract data
        configs = list(config_variance.keys())
        acc_stds = [config_variance[c]['accuracy_std'] for c in configs]
        acc_ranges = [config_variance[c]['accuracy_range'] for c in configs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy standard deviation
        bars1 = ax1.bar(configs, acc_stds, color=plt.cm.Set2(np.arange(len(configs))))
        ax1.set_title('Configuration Sensitivity\n(Std of Accuracy Across Experiments)')
        ax1.set_ylabel('Standard Deviation of Test Accuracy')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

        # Accuracy range
        bars2 = ax2.bar(configs, acc_ranges, color=plt.cm.Set2(np.arange(len(configs))))
        ax2.set_title('Configuration Robustness\n(Range of Accuracy Across Experiments)')
        ax2.set_ylabel('Range of Test Accuracy')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(plots_dir / 'architecture_sensitivity.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_report(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate comprehensive text report."""

        report_lines = [
            "=" * 80,
            "COMPREHENSIVE BANDRMS EVALUATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Results Directory: {self.run_dir}",
            "",
        ]

        # Executive Summary
        report_lines.extend([
            "EXECUTIVE SUMMARY",
            "-" * 40,
            ""
        ])

        if 'best_configs' in analysis:
            best_configs = analysis['best_configs']

            if 'overall' in best_configs:
                overall_best = best_configs['overall']
                report_lines.extend([
                    f"Best Overall Configuration: {overall_best['config']}",
                    f"  - Experiment: {overall_best['experiment']}",
                    f"  - Test Accuracy: {overall_best['test_accuracy']:.4f}",
                    f"  - Test Loss: {overall_best['test_loss']:.4f}",
                    ""
                ])

            if 'best_bandrms' in best_configs:
                best_bandrms = best_configs['best_bandrms']
                report_lines.extend([
                    f"Best BandRMS Configuration: {best_bandrms['config']}",
                    f"  - Experiment: {best_bandrms['experiment']}",
                    f"  - Test Accuracy: {best_bandrms['test_accuracy']:.4f}",
                    f"  - Test Loss: {best_bandrms['test_loss']:.4f}",
                    ""
                ])

        # Performance by Configuration
        if 'summary' in analysis and 'by_configuration' in analysis['summary']:
            report_lines.extend([
                "PERFORMANCE BY CONFIGURATION",
                "-" * 40,
                ""
            ])

            config_summary = analysis['summary']['by_configuration']
            sorted_configs = sorted(config_summary.items(),
                                    key=lambda x: x[1]['mean_test_accuracy'], reverse=True)

            for config, metrics in sorted_configs:
                report_lines.extend([
                    f"{config}:",
                    f"  - Mean Test Accuracy: {metrics['mean_test_accuracy']:.4f} ± {metrics['std_test_accuracy']:.4f}",
                    f"  - Mean Test Loss: {metrics['mean_test_loss']:.4f} ± {metrics['std_test_loss']:.4f}",
                    f"  - Success Rate: {metrics['success_rate']:.2f}",
                    f"  - Experiments: {metrics['experiments_count']}",
                    ""
                ])

        # Training Stability Analysis
        if 'stability_analysis' in analysis:
            report_lines.extend([
                "TRAINING STABILITY ANALYSIS",
                "-" * 40,
                ""
            ])

            stability = analysis['stability_analysis']

            if 'convergence_rates' in stability:
                sorted_convergence = sorted(stability['convergence_rates'].items(),
                                            key=lambda x: x[1], reverse=True)

                report_lines.append("Convergence Rates (Best to Worst):")
                for config, rate in sorted_convergence[:5]:
                    report_lines.append(f"  - {config}: {rate:.2f}")
                report_lines.append("")

            if 'performance_consistency' in stability:
                sorted_consistency = sorted(stability['performance_consistency'].items(),
                                            key=lambda x: x[1])

                report_lines.append("Most Consistent Configurations (Lower std = Better):")
                for config, std in sorted_consistency[:5]:
                    report_lines.append(f"  - {config}: {std:.4f}")
                report_lines.append("")

        # Band Parameter Analysis
        if 'band_parameter_analysis' in analysis and analysis['band_parameter_analysis']:
            report_lines.extend([
                "BAND PARAMETER ANALYSIS",
                "-" * 40,
                ""
            ])

            for exp_name, exp_data in analysis['band_parameter_analysis'].items():
                report_lines.append(f"{exp_name.upper()} Experiment:")

                for config, params in exp_data.items():
                    report_lines.extend([
                        f"  {config}:",
                        f"    - Mean Parameter: {params['mean_param']:.4f}",
                        f"    - Parameter Range: {params['min_param']:.4f} - {params['max_param']:.4f}",
                        f"    - Std Parameter: {params['std_param']:.4f}",
                        f"    - Layers: {params['layer_count']}",
                        ""
                    ])

        # Conclusions and Recommendations
        report_lines.extend([
            "CONCLUSIONS AND RECOMMENDATIONS",
            "-" * 40,
            ""
        ])

        # Add conclusions based on analysis
        if 'best_configs' in analysis and 'overall' in analysis['best_configs']:
            best_overall = analysis['best_configs']['overall']['config']

            if 'band_rms' in best_overall:
                report_lines.extend([
                    "✅ BandRMS demonstrates superior performance as the best overall configuration.",
                    f"   Specifically, {best_overall} achieved the highest accuracy.",
                    ""
                ])
            else:
                report_lines.extend([
                    f"❌ {best_overall} outperformed BandRMS variants in this evaluation.",
                    "   Consider investigating specific architectural or hyperparameter factors.",
                    ""
                ])

        # Add stability insights
        if ('stability_analysis' in analysis and 'convergence_rates' in analysis['stability_analysis']):
            conv_rates = analysis['stability_analysis']['convergence_rates']
            bandrms_rates = {k: v for k, v in conv_rates.items() if 'band_rms' in k}

            if bandrms_rates:
                avg_bandrms_conv = np.mean(list(bandrms_rates.values()))
                other_rates = {k: v for k, v in conv_rates.items() if 'band_rms' not in k}
                avg_other_conv = np.mean(list(other_rates.values())) if other_rates else 0

                if avg_bandrms_conv > avg_other_conv:
                    report_lines.extend([
                        f"✅ BandRMS shows superior training stability with {avg_bandrms_conv:.2f} average convergence rate",
                        f"   vs {avg_other_conv:.2f} for other normalization techniques.",
                        ""
                    ])
                else:
                    report_lines.extend([
                        f"⚠️  BandRMS shows mixed training stability with {avg_bandrms_conv:.2f} average convergence rate",
                        f"   vs {avg_other_conv:.2f} for other normalization techniques.",
                        ""
                    ])

        report_lines.extend([
            "NEXT STEPS",
            "-" * 20,
            "1. Investigate hyperparameter sensitivity for best-performing configurations",
            "2. Test on larger-scale datasets and architectures",
            "3. Analyze computational overhead and memory usage",
            "4. Explore adaptive max_band_width scheduling during training",
            "",
            "=" * 80
        ])

        report_text = "\n".join(report_lines)

        # Save report
        report_path = self.run_dir / 'comprehensive_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Comprehensive report saved to: {report_path}")

        return report_text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run comprehensive BandRMS evaluation')

    parser.add_argument(
        '--experiments',
        type=str,
        default='all',
        help='Experiments to run: all, transformer, cnn, mlp, or comma-separated list'
    )

    parser.add_argument(
        '--mlp-task',
        type=str,
        default='classification',
        choices=['classification', 'regression'],
        help='Task type for MLP experiment'
    )

    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only perform analysis on existing results'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='experiments',
        help='Directory containing results for analysis-only mode'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments',
        help='Base output directory for results'
    )

    args = parser.parse_args()

    # Create master runner
    runner = MasterExperimentRunner(base_output_dir=args.output_dir)

    if args.analyze_only:
        # Load existing results and analyze
        results_path = Path(args.results_dir) / 'all_experiments_results.json'
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            return

        with open(results_path, 'r') as f:
            results = json.load(f)

        logger.info(f"Loaded results from: {results_path}")
    else:
        # Parse experiment list
        if args.experiments == 'all':
            experiment_names = ['transformer', 'cnn', 'mlp']
        else:
            experiment_names = [exp.strip() for exp in args.experiments.split(',')]

        # Validate experiment names
        valid_experiments = ['transformer', 'cnn', 'mlp']
        for exp in experiment_names:
            if exp not in valid_experiments:
                logger.error(f"Invalid experiment name: {exp}. Valid options: {valid_experiments}")
                return

        # Run experiments
        results = runner.run_experiments(experiment_names, args.mlp_task)

    # Perform analysis
    analysis = runner.analyze_results(results)

    # Create visualizations
    runner.create_comprehensive_plots(results, analysis)

    # Generate report
    report = runner.generate_report(results, analysis)

    # Print summary
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Results Directory: {runner.run_dir}")
    logger.info(f"See comprehensive_report.txt for detailed analysis")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()