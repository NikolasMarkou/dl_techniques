"""
Multi-Task MDN Results Analyzer

This utility script provides functions to analyze and visualize results from
multi-task MDN experiments, including comparative analysis across tasks,
uncertainty decomposition, and performance insights.
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Optional


from dl_techniques.utils.logger import logger

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# ---------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------

class MultiTaskMDNAnalyzer:
    """Analyzer for multi-task MDN experiment results."""

    def __init__(self, results_dir: str):
        """Initialize analyzer with results directory.

        Args:
            results_dir: Path to the directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.metrics_df = None
        self.history_df = None
        self.aggregate_metrics = {}

        self._load_results()

    def _load_results(self):
        """Load results from the results directory."""
        try:
            # Load task metrics
            metrics_file = self.results_dir / 'task_metrics.csv'
            if metrics_file.exists():
                self.metrics_df = pd.read_csv(metrics_file)
                logger.info(f"Loaded task metrics: {self.metrics_df.shape}")

            # Load training history
            history_file = self.results_dir / 'training_history.csv'
            if history_file.exists():
                self.history_df = pd.read_csv(history_file)
                logger.info(f"Loaded training history: {self.history_df.shape}")

            # Load aggregate metrics
            aggregate_file = self.results_dir / 'aggregate_metrics.txt'
            if aggregate_file.exists():
                with open(aggregate_file, 'r') as f:
                    for line in f:
                        key, value = line.strip().split(': ')
                        self.aggregate_metrics[key] = float(value)
                logger.info(f"Loaded aggregate metrics: {len(self.aggregate_metrics)} items")

        except Exception as e:
            logger.error(f"Error loading results: {e}")

    def create_performance_comparison(self, save_path: Optional[str] = None):
        """Create a comprehensive performance comparison plot."""
        if self.metrics_df is None:
            logger.error("No metrics data available")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Task MDN Performance Comparison', fontsize=16, y=0.98)

        # RMSE comparison
        axes[0, 0].bar(self.metrics_df['Task'], self.metrics_df['RMSE'], color='steelblue')
        axes[0, 0].set_title('Root Mean Square Error (RMSE)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # MAE comparison
        axes[0, 1].bar(self.metrics_df['Task'], self.metrics_df['MAE'], color='lightcoral')
        axes[0, 1].set_title('Mean Absolute Error (MAE)')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Coverage comparison
        axes[0, 2].bar(self.metrics_df['Task'], self.metrics_df['Coverage'], color='lightgreen')
        axes[0, 2].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (95%)')
        axes[0, 2].set_title('Prediction Interval Coverage')
        axes[0, 2].set_ylabel('Coverage')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Interval width comparison
        axes[1, 0].bar(self.metrics_df['Task'], self.metrics_df['Interval Width'], color='orange')
        axes[1, 0].set_title('Prediction Interval Width')
        axes[1, 0].set_ylabel('Interval Width')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Uncertainty decomposition
        x = np.arange(len(self.metrics_df))
        width = 0.35
        axes[1, 1].bar(x - width / 2, self.metrics_df['Aleatoric Unc.'], width,
                       label='Aleatoric', color='skyblue')
        axes[1, 1].bar(x + width / 2, self.metrics_df['Epistemic Unc.'], width,
                       label='Epistemic', color='lightcoral')
        axes[1, 1].set_title('Uncertainty Decomposition')
        axes[1, 1].set_ylabel('Uncertainty')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(self.metrics_df['Task'], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Sample counts
        axes[1, 2].bar(self.metrics_df['Task'], self.metrics_df['Samples'], color='mediumpurple')
        axes[1, 2].set_title('Number of Test Samples')
        axes[1, 2].set_ylabel('Sample Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison saved to: {save_path}")

        plt.show()

    def create_training_analysis(self, save_path: Optional[str] = None):
        """Create training analysis plots."""
        if self.history_df is None:
            logger.error("No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Analysis', fontsize=16, y=0.98)

        # Training and validation loss
        axes[0, 0].plot(self.history_df['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(self.history_df['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss smoothing (moving average)
        window_size = max(1, len(self.history_df) // 10)
        train_smooth = self.history_df['loss'].rolling(window=window_size).mean()
        val_smooth = self.history_df['val_loss'].rolling(window=window_size).mean()

        axes[0, 1].plot(train_smooth, label=f'Training (MA-{window_size})', linewidth=2)
        axes[0, 1].plot(val_smooth, label=f'Validation (MA-{window_size})', linewidth=2)
        axes[0, 1].set_title('Smoothed Training Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate (if available)
        if 'lr' in self.history_df.columns:
            axes[1, 0].plot(self.history_df['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)

        # Convergence analysis
        final_epochs = min(50, len(self.history_df))
        recent_loss = self.history_df['val_loss'].iloc[-final_epochs:]

        axes[1, 1].plot(range(len(self.history_df) - final_epochs, len(self.history_df)),
                        recent_loss, linewidth=2, color='red')
        axes[1, 1].set_title(f'Convergence (Last {final_epochs} Epochs)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training analysis saved to: {save_path}")

        plt.show()

    def create_uncertainty_analysis(self, save_path: Optional[str] = None):
        """Create detailed uncertainty analysis."""
        if self.metrics_df is None:
            logger.error("No metrics data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Uncertainty Analysis', fontsize=16, y=0.98)

        # Total uncertainty vs performance
        axes[0, 0].scatter(self.metrics_df['Aleatoric Unc.'] + self.metrics_df['Epistemic Unc.'],
                           self.metrics_df['RMSE'], s=100, alpha=0.7)

        for i, task in enumerate(self.metrics_df['Task']):
            axes[0, 0].annotate(task,
                                (self.metrics_df['Aleatoric Unc.'].iloc[i] +
                                 self.metrics_df['Epistemic Unc.'].iloc[i],
                                 self.metrics_df['RMSE'].iloc[i]),
                                xytext=(5, 5), textcoords='offset points')

        axes[0, 0].set_xlabel('Total Uncertainty')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('Uncertainty vs Performance')
        axes[0, 0].grid(True, alpha=0.3)

        # Aleatoric vs Epistemic uncertainty
        axes[0, 1].scatter(self.metrics_df['Aleatoric Unc.'],
                           self.metrics_df['Epistemic Unc.'], s=100, alpha=0.7)

        for i, task in enumerate(self.metrics_df['Task']):
            axes[0, 1].annotate(task,
                                (self.metrics_df['Aleatoric Unc.'].iloc[i],
                                 self.metrics_df['Epistemic Unc.'].iloc[i]),
                                xytext=(5, 5), textcoords='offset points')

        axes[0, 1].set_xlabel('Aleatoric Uncertainty')
        axes[0, 1].set_ylabel('Epistemic Uncertainty')
        axes[0, 1].set_title('Aleatoric vs Epistemic Uncertainty')
        axes[0, 1].grid(True, alpha=0.3)

        # Uncertainty ratio analysis
        uncertainty_ratio = self.metrics_df['Epistemic Unc.'] / (
                self.metrics_df['Aleatoric Unc.'] + self.metrics_df['Epistemic Unc.'] + 1e-8
        )

        axes[1, 0].bar(self.metrics_df['Task'], uncertainty_ratio, color='teal')
        axes[1, 0].set_title('Epistemic Uncertainty Ratio')
        axes[1, 0].set_ylabel('Epistemic / Total Uncertainty')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Coverage vs Interval Width
        axes[1, 1].scatter(self.metrics_df['Interval Width'],
                           self.metrics_df['Coverage'], s=100, alpha=0.7)

        for i, task in enumerate(self.metrics_df['Task']):
            axes[1, 1].annotate(task,
                                (self.metrics_df['Interval Width'].iloc[i],
                                 self.metrics_df['Coverage'].iloc[i]),
                                xytext=(5, 5), textcoords='offset points')

        axes[1, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target Coverage')
        axes[1, 1].set_xlabel('Prediction Interval Width')
        axes[1, 1].set_ylabel('Coverage')
        axes[1, 1].set_title('Coverage vs Interval Width Trade-off')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Uncertainty analysis saved to: {save_path}")

        plt.show()

    def create_task_comparison_heatmap(self, save_path: Optional[str] = None):
        """Create a heatmap comparing all metrics across tasks."""
        if self.metrics_df is None:
            logger.error("No metrics data available")
            return

        # Select numeric columns for heatmap
        numeric_cols = ['RMSE', 'MAE', 'Coverage', 'Interval Width',
                        'Aleatoric Unc.', 'Epistemic Unc.']

        # Create normalized data for better visualization
        heatmap_data = self.metrics_df[numeric_cols].copy()

        # Normalize each metric to [0, 1] for better comparison
        for col in numeric_cols:
            if col == 'Coverage':
                # For coverage, we want values close to 0.95 to be "good"
                heatmap_data[col] = 1 - np.abs(heatmap_data[col] - 0.95)
            else:
                # For other metrics, lower is generally better (except we need to handle this carefully)
                if col in ['RMSE', 'MAE', 'Interval Width']:
                    # Lower is better - use inverse normalization
                    heatmap_data[col] = 1 - (heatmap_data[col] - heatmap_data[col].min()) / (
                            heatmap_data[col].max() - heatmap_data[col].min() + 1e-8
                    )
                else:
                    # For uncertainty, we just normalize without inverting
                    heatmap_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / (
                            heatmap_data[col].max() - heatmap_data[col].min() + 1e-8
                    )

        # Set task names as index
        heatmap_data.index = self.metrics_df['Task']

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlBu_r', center=0.5,
                    fmt='.3f', cbar_kws={'label': 'Normalized Performance'})
        plt.title('Multi-Task Performance Heatmap\n(Higher values indicate better performance)')
        plt.ylabel('Metrics')
        plt.xlabel('Tasks')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Task comparison heatmap saved to: {save_path}")

        plt.show()

    def create_comprehensive_report(self, save_dir: Optional[str] = None):
        """Create a comprehensive analysis report."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        logger.info("Creating comprehensive analysis report...")

        # Performance comparison
        perf_path = save_dir / 'performance_comparison.png' if save_dir else None
        self.create_performance_comparison(perf_path)

        # Training analysis
        train_path = save_dir / 'training_analysis.png' if save_dir else None
        self.create_training_analysis(train_path)

        # Uncertainty analysis
        unc_path = save_dir / 'uncertainty_analysis.png' if save_dir else None
        self.create_uncertainty_analysis(unc_path)

        # Task comparison heatmap
        heatmap_path = save_dir / 'task_comparison_heatmap.png' if save_dir else None
        self.create_task_comparison_heatmap(heatmap_path)

        # Generate summary statistics
        self.generate_summary_statistics(save_dir)

        logger.info("Comprehensive analysis report completed")

    def generate_summary_statistics(self, save_dir: Optional[Path] = None):
        """Generate and save summary statistics."""
        if self.metrics_df is None:
            logger.error("No metrics data available")
            return

        summary_stats = {}

        # Basic statistics
        numeric_cols = ['RMSE', 'MAE', 'Coverage', 'Interval Width',
                        'Aleatoric Unc.', 'Epistemic Unc.']

        for col in numeric_cols:
            summary_stats[col] = {
                'mean': self.metrics_df[col].mean(),
                'std': self.metrics_df[col].std(),
                'min': self.metrics_df[col].min(),
                'max': self.metrics_df[col].max(),
                'median': self.metrics_df[col].median()
            }

        # Best and worst performing tasks
        best_rmse_task = self.metrics_df.loc[self.metrics_df['RMSE'].idxmin(), 'Task']
        worst_rmse_task = self.metrics_df.loc[self.metrics_df['RMSE'].idxmax(), 'Task']

        best_coverage_task = self.metrics_df.loc[
            np.abs(self.metrics_df['Coverage'] - 0.95).idxmin(), 'Task'
        ]

        # Uncertainty insights
        total_uncertainty = self.metrics_df['Aleatoric Unc.'] + self.metrics_df['Epistemic Unc.']
        avg_aleatoric_ratio = (self.metrics_df['Aleatoric Unc.'] / total_uncertainty).mean()
        avg_epistemic_ratio = (self.metrics_df['Epistemic Unc.'] / total_uncertainty).mean()

        # Create summary text
        summary_text = f"""
Multi-Task MDN Analysis Summary
===============================

Dataset Overview:
- Number of tasks: {len(self.metrics_df)}
- Tasks: {', '.join(self.metrics_df['Task'].tolist())}

Performance Summary:
- Average RMSE: {summary_stats['RMSE']['mean']:.4f} ± {summary_stats['RMSE']['std']:.4f}
- Average MAE: {summary_stats['MAE']['mean']:.4f} ± {summary_stats['MAE']['std']:.4f}
- Average Coverage: {summary_stats['Coverage']['mean']:.4f} ± {summary_stats['Coverage']['std']:.4f}
- Average Interval Width: {summary_stats['Interval Width']['mean']:.4f} ± {summary_stats['Interval Width']['std']:.4f}

Best Performing Tasks:
- Lowest RMSE: {best_rmse_task} (RMSE: {self.metrics_df.loc[self.metrics_df['RMSE'].idxmin(), 'RMSE']:.4f})
- Best Coverage: {best_coverage_task} (Coverage: {self.metrics_df.loc[np.abs(self.metrics_df['Coverage'] - 0.95).idxmin(), 'Coverage']:.4f})

Worst Performing Tasks:
- Highest RMSE: {worst_rmse_task} (RMSE: {self.metrics_df.loc[self.metrics_df['RMSE'].idxmax(), 'RMSE']:.4f})

Uncertainty Analysis:
- Average Aleatoric Uncertainty: {summary_stats['Aleatoric Unc.']['mean']:.4f} ± {summary_stats['Aleatoric Unc.']['std']:.4f}
- Average Epistemic Uncertainty: {summary_stats['Epistemic Unc.']['mean']:.4f} ± {summary_stats['Epistemic Unc.']['std']:.4f}
- Aleatoric Uncertainty Ratio: {avg_aleatoric_ratio:.4f}
- Epistemic Uncertainty Ratio: {avg_epistemic_ratio:.4f}

Training Summary:
"""

        if self.history_df is not None:
            final_train_loss = self.history_df['loss'].iloc[-1]
            final_val_loss = self.history_df['val_loss'].iloc[-1]
            min_val_loss = self.history_df['val_loss'].min()
            epochs_trained = len(self.history_df)

            summary_text += f"""- Epochs trained: {epochs_trained}
- Final training loss: {final_train_loss:.4f}
- Final validation loss: {final_val_loss:.4f}
- Best validation loss: {min_val_loss:.4f}
"""

        if self.aggregate_metrics:
            summary_text += f"""
Aggregate Metrics:
"""
            for key, value in self.aggregate_metrics.items():
                summary_text += f"- {key}: {value:.4f}\n"

        # Print summary
        print(summary_text)

        # Save summary if directory provided
        if save_dir:
            with open(save_dir / 'analysis_summary.txt', 'w') as f:
                f.write(summary_text)

            # Save detailed statistics as JSON
            with open(save_dir / 'detailed_statistics.json', 'w') as f:
                json.dump(summary_stats, f, indent=2)

            logger.info(f"Summary statistics saved to: {save_dir}")

        return summary_stats

    def compare_with_baseline(self, baseline_results: Dict[str, float],
                              save_path: Optional[str] = None):
        """Compare current results with baseline performance."""
        if self.metrics_df is None:
            logger.error("No metrics data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Task MDN vs Baseline Comparison', fontsize=16, y=0.98)

        # RMSE comparison
        x = np.arange(len(self.metrics_df))
        width = 0.35

        baseline_rmse = [baseline_results.get(task, 0) for task in self.metrics_df['Task']]

        axes[0, 0].bar(x - width / 2, baseline_rmse, width, label='Baseline', alpha=0.7)
        axes[0, 0].bar(x + width / 2, self.metrics_df['RMSE'], width, label='Multi-Task MDN', alpha=0.7)
        axes[0, 0].set_xlabel('Task')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.metrics_df['Task'], rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Improvement percentage
        improvement = []
        for i, task in enumerate(self.metrics_df['Task']):
            baseline_val = baseline_results.get(task, 0)
            current_val = self.metrics_df['RMSE'].iloc[i]
            if baseline_val > 0:
                improvement.append((baseline_val - current_val) / baseline_val * 100)
            else:
                improvement.append(0)

        colors = ['green' if imp > 0 else 'red' for imp in improvement]
        axes[0, 1].bar(self.metrics_df['Task'], improvement, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Task')
        axes[0, 1].set_ylabel('Improvement (%)')
        axes[0, 1].set_title('RMSE Improvement over Baseline')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Coverage comparison (if available in baseline)
        if 'coverage' in baseline_results:
            baseline_coverage = [baseline_results['coverage']] * len(self.metrics_df)
            axes[1, 0].bar(x - width / 2, baseline_coverage, width, label='Baseline', alpha=0.7)
            axes[1, 0].bar(x + width / 2, self.metrics_df['Coverage'], width, label='Multi-Task MDN', alpha=0.7)
            axes[1, 0].set_xlabel('Task')
            axes[1, 0].set_ylabel('Coverage')
            axes[1, 0].set_title('Coverage Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(self.metrics_df['Task'], rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Baseline Coverage\nNot Available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)

        # Summary statistics
        avg_improvement = np.mean(improvement)
        best_improvement = max(improvement)
        worst_improvement = min(improvement)

        summary_text = f"""
Comparison Summary:
Average Improvement: {avg_improvement:.2f}%
Best Improvement: {best_improvement:.2f}%
Worst Improvement: {worst_improvement:.2f}%
Tasks Improved: {sum(1 for imp in improvement if imp > 0)}/{len(improvement)}
"""

        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Baseline comparison saved to: {save_path}")

        plt.show()


# ---------------------------------------------------------------------
# Command Line Interface
# ---------------------------------------------------------------------

def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Analyze Multi-Task MDN experiment results')

    parser.add_argument('results_dir', type=str,
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save analysis outputs')
    parser.add_argument('--baseline', type=str, default=None,
                        help='JSON file containing baseline results for comparison')
    parser.add_argument('--report', action='store_true',
                        help='Generate comprehensive report')
    parser.add_argument('--performance', action='store_true',
                        help='Generate performance comparison plots')
    parser.add_argument('--training', action='store_true',
                        help='Generate training analysis plots')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Generate uncertainty analysis plots')
    parser.add_argument('--heatmap', action='store_true',
                        help='Generate task comparison heatmap')

    args = parser.parse_args()

    # Initialize analyzer
    try:
        analyzer = MultiTaskMDNAnalyzer(args.results_dir)
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = None

    # Generate requested analyses
    if args.report:
        analyzer.create_comprehensive_report(output_dir)
    else:
        if args.performance:
            save_path = output_dir / 'performance_comparison.png' if output_dir else None
            analyzer.create_performance_comparison(save_path)

        if args.training:
            save_path = output_dir / 'training_analysis.png' if output_dir else None
            analyzer.create_training_analysis(save_path)

        if args.uncertainty:
            save_path = output_dir / 'uncertainty_analysis.png' if output_dir else None
            analyzer.create_uncertainty_analysis(save_path)

        if args.heatmap:
            save_path = output_dir / 'task_comparison_heatmap.png' if output_dir else None
            analyzer.create_task_comparison_heatmap(save_path)

    # Baseline comparison if provided
    if args.baseline:
        try:
            with open(args.baseline, 'r') as f:
                baseline_results = json.load(f)
            save_path = output_dir / 'baseline_comparison.png' if output_dir else None
            analyzer.compare_with_baseline(baseline_results, save_path)
        except Exception as e:
            logger.error(f"Failed to load baseline results: {e}")

    # Always generate summary statistics
    analyzer.generate_summary_statistics(output_dir)

    logger.info("Analysis completed successfully")


# ---------------------------------------------------------------------
# Example Usage Functions
# ---------------------------------------------------------------------

def example_usage():
    """Show example usage of the analyzer."""
    print("""
Multi-Task MDN Results Analyzer - Example Usage
==============================================

1. Basic analysis:
   python multitask_mdn_analyzer.py /path/to/results

2. Generate comprehensive report:
   python multitask_mdn_analyzer.py /path/to/results --report --output-dir /path/to/output

3. Generate specific analyses:
   python multitask_mdn_analyzer.py /path/to/results --performance --uncertainty --output-dir /path/to/output

4. Compare with baseline:
   python multitask_mdn_analyzer.py /path/to/results --baseline baseline.json --output-dir /path/to/output

5. Programmatic usage:
   ```python
   from multitask_mdn_analyzer import MultiTaskMDNAnalyzer

   analyzer = MultiTaskMDNAnalyzer('/path/to/results')
   analyzer.create_comprehensive_report('/path/to/output')
   ```

Baseline JSON format:
{
    "Sine Wave": 0.15,
    "Noisy Sine": 0.25,
    "Stock Price": 0.35,
    "coverage": 0.90
}
""")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        example_usage()
    else:
        main()