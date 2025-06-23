"""
Black Hole Analyzer - Main interface for detecting computational black holes in neural networks

This module provides a simplified interface to the black hole metrics module,
making it easy to analyze models for computational black holes - individual weights
that concentrate a disproportionate amount of Fisher Information.
"""

import os
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple,  Any
import tempfile
import json

# Import from tf_weightwatcher
from .weightwatcher import WeightWatcher
from .black_hole_metrics import BlackHoleMetrics
from .constants import DEFAULT_FIG_SIZE, DEFAULT_DPI

from dl_techniques.utils.logger import logger


class BlackHoleAnalyzer:
    """
    Analyzer for detecting computational black holes in neural networks.

    This class integrates the standard WeightWatcher analysis with additional
    black hole metrics to identify critical parameters in neural networks.
    """

    def __init__(self, model: keras.Model):
        """
        Initialize the Black Hole Analyzer.

        Args:
            model: Keras model to analyze.
        """
        self.model = model
        self.watcher = WeightWatcher(model)
        self.black_hole_metrics = BlackHoleMetrics(model, self.watcher)
        self.standard_analysis = None
        self.black_hole_analysis = None

    def analyze(self,
                layers: List[int] = None,
                plot: bool = True,
                savedir: str = 'black_hole_analysis',
                detailed: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive analysis of the model to detect computational black holes.

        Args:
            layers: List of layer indices to analyze. If None, analyze all layers.
            plot: Whether to create visualizations.
            savedir: Directory to save analysis results and visualizations.
            detailed: Whether to perform detailed analysis of super weights.

        Returns:
            Dictionary containing standard and black hole analysis DataFrames.
        """
        logger.info("Starting Black Hole Analyzer analysis")

        # Create output directory
        os.makedirs(savedir, exist_ok=True)

        # Standard WeightWatcher analysis
        logger.info("Performing standard WeightWatcher analysis")
        self.standard_analysis = self.watcher.analyze(
            layers=layers,
            plot=plot,
            savefig=os.path.join(savedir, 'standard_analysis') if plot else False
        )

        # Black Hole analysis
        logger.info("Performing Black Hole analysis")
        self.black_hole_analysis = self.black_hole_metrics.analyze(
            layers=layers,
            plot=plot,
            savefig=os.path.join(savedir, 'black_hole_analysis') if plot else False
        )

        # Combine results
        combined_results = self._combine_analyses()

        # Create summary visualization
        if plot:
            self._create_summary_visualization(savedir)

        # Generate report
        self._generate_report(savedir, detailed)

        logger.info(f"Analysis complete. Results saved to {savedir}")

        return {
            'standard': self.standard_analysis,
            'black_hole': self.black_hole_analysis,
            'combined': combined_results
        }

    def _combine_analyses(self) -> pd.DataFrame:
        """
        Combine standard and black hole analyses into a single DataFrame.

        Returns:
            Combined analysis DataFrame.
        """
        if self.standard_analysis is None or self.black_hole_analysis is None:
            logger.warning("Cannot combine analyses: one or both analyses missing")
            return pd.DataFrame()

        # Start with standard analysis
        combined = self.standard_analysis.copy()

        # Add black hole metrics
        for layer_id in self.black_hole_analysis.index:
            if layer_id in combined.index:
                # Add black hole metrics that don't conflict
                for col in self.black_hole_analysis.columns:
                    if col not in combined.columns:
                        combined.at[layer_id, col] = self.black_hole_analysis.at[layer_id, col]

        return combined

    def _create_summary_visualization(self, savedir: str) -> None:
        """
        Create summary visualizations of the analysis results.

        Args:
            savedir: Directory to save visualizations.
        """
        if self.standard_analysis is None or self.black_hole_analysis is None:
            logger.warning("Cannot create visualizations: analyses missing")
            return

        try:
            # Create visualization directory
            vis_dir = os.path.join(savedir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)

            # 1. Plot black hole score vs. alpha (power-law exponent)
            if 'black_hole_score' in self.black_hole_analysis.columns and 'alpha' in self.standard_analysis.columns:
                plt.figure(figsize=DEFAULT_FIG_SIZE)

                # Get data
                layers = sorted(set(self.standard_analysis.index) & set(self.black_hole_analysis.index))
                x = [self.standard_analysis.loc[layer, 'alpha'] for layer in layers]
                y = [self.black_hole_analysis.loc[layer, 'black_hole_score'] for layer in layers]
                labels = [f"{layer}: {self.standard_analysis.loc[layer, 'name']}" for layer in layers]

                # Create scatter plot
                sc = plt.scatter(x, y, c=range(len(x)), cmap='viridis', s=100, alpha=0.7)

                # Add layer IDs as annotations
                for i, layer_id in enumerate(layers):
                    plt.annotate(str(layer_id), (x[i], y[i]),
                                 textcoords="offset points",
                                 xytext=(0, 5),
                                 ha='center')

                # Add trend line
                if len(x) > 1:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(x), "r--", alpha=0.7)

                plt.title("Black Hole Score vs. Power-Law Exponent (α)")
                plt.xlabel("Power-Law Exponent (α)")
                plt.ylabel("Black Hole Score (log scale)")
                plt.yscale('log')
                plt.colorbar(sc, label="Layer Index")
                plt.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'blackhole_vs_alpha.png'), dpi=DEFAULT_DPI)
                plt.close()

            # 2. Plot top layers by black hole score
            if 'black_hole_score' in self.black_hole_analysis.columns:
                plt.figure(figsize=DEFAULT_FIG_SIZE)

                # Get top 10 layers
                top_layers = self.black_hole_analysis.sort_values(
                    'black_hole_score', ascending=False
                ).head(10)

                # Create bar chart
                bars = plt.bar(
                    range(len(top_layers)),
                    top_layers['black_hole_score'],
                    color='darkred'
                )

                # Add layer names
                plt.xticks(
                    range(len(top_layers)),
                    [f"{idx}: {row['name']}" for idx, row in top_layers.iterrows()],
                    rotation=45,
                    ha='right'
                )

                # Add labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height + 0.1,
                        f"{top_layers.iloc[i]['super_weight_count'] if 'super_weight_count' in top_layers.columns else 'N/A'}",
                        ha='center',
                        va='bottom',
                        rotation=0
                    )

                plt.title("Top Layers by Black Hole Score")
                plt.xlabel("Layer")
                plt.ylabel("Black Hole Score")
                plt.yscale('log')
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'top_blackhole_layers.png'), dpi=DEFAULT_DPI)
                plt.close()

            # 3. Plot relationship between black hole metrics
            if ('gini_coefficient' in self.black_hole_analysis.columns and
                    'dominance_ratio' in self.black_hole_analysis.columns and
                    'participation_ratio' in self.black_hole_analysis.columns):

                plt.figure(figsize=DEFAULT_FIG_SIZE)

                # Get data
                x = self.black_hole_analysis['gini_coefficient']
                y = self.black_hole_analysis['dominance_ratio']
                z = 1 / (self.black_hole_analysis['participation_ratio'] + 1e-10)  # Inverse for clearer visualization

                # Create scatter plot
                sc = plt.scatter(x, y, c=z, s=100, alpha=0.7, cmap='plasma')

                # Add layer IDs as annotations
                for i, layer_id in enumerate(self.black_hole_analysis.index):
                    plt.annotate(str(layer_id),
                                 (x.iloc[i], y.iloc[i]),
                                 textcoords="offset points",
                                 xytext=(0, 5),
                                 ha='center')

                plt.title("Relationship Between Black Hole Metrics")
                plt.xlabel("Gini Coefficient")
                plt.ylabel("Dominance Ratio")
                plt.colorbar(sc, label="Inverse Participation Ratio")
                plt.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'blackhole_metrics_relationship.png'), dpi=DEFAULT_DPI)
                plt.close()

        except Exception as e:
            logger.warning(f"Error creating summary visualizations: {e}")

    def _generate_report(self, savedir: str, detailed: bool = False) -> None:
        """
        Generate HTML report of the analysis results.

        Args:
            savedir: Directory to save the report.
            detailed: Whether to include detailed super weight information.
        """
        if self.standard_analysis is None or self.black_hole_analysis is None:
            logger.warning("Cannot generate report: analyses missing")
            return

        try:
            # Prepare report data
            report_data = {
                'model_summary': {
                    'name': self.model.name,
                    'layers': len(self.model.layers),
                    'parameters': self.model.count_params()
                },
                'standard_metrics': self.watcher.get_summary(),
                'black_hole_metrics': self.black_hole_metrics.get_summary(),
                'top_black_hole_layers': []
            }

            # Add top black hole layers
            if 'black_hole_score' in self.black_hole_analysis.columns:
                top_layers = self.black_hole_analysis.sort_values(
                    'black_hole_score', ascending=False
                ).head(5)

                for layer_id, row in top_layers.iterrows():
                    layer_info = {
                        'id': int(layer_id),
                        'name': row['name'],
                        'type': row['layer_type'],
                        'black_hole_score': float(row['black_hole_score']),
                        'gini_coefficient': float(row['gini_coefficient']) if 'gini_coefficient' in row else None,
                        'dominance_ratio': float(row['dominance_ratio']) if 'dominance_ratio' in row else None,
                        'participation_ratio': float(
                            row['participation_ratio']) if 'participation_ratio' in row else None
                    }

                    # Add super weight info if detailed
                    if detailed and 'super_weights' in row:
                        layer_info['super_weights'] = [
                            {'i': int(i), 'j': int(j), 'contribution': float(c)}
                            for i, j, c in row['super_weights']
                        ]

                    report_data['top_black_hole_layers'].append(layer_info)

            # Save report data as JSON
            with open(os.path.join(savedir, 'report_data.json'), 'w') as f:
                json.dump(report_data, f, indent=2)

            # Generate HTML report
            self._create_html_report(report_data, savedir)

        except Exception as e:
            logger.warning(f"Error generating report: {e}")

    def _create_html_report(self, report_data: Dict[str, Any], savedir: str) -> None:
        """
        Create HTML report from report data.

        Args:
            report_data: Dictionary containing report data.
            savedir: Directory to save the report.
        """
        # Basic HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Computational Black Hole Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ color: #0066cc; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .highlight {{ background-color: #ffffcc; }}
                .footer {{ margin-top: 30px; font-size: 0.8em; color: #666; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Computational Black Hole Analysis Report</h1>

                <div class="section">
                    <h2>Model Summary</h2>
                    <div class="metric">
                        <span class="metric-name">Model Name:</span>
                        <span class="metric-value">{report_data['model_summary']['name']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-name">Total Layers:</span>
                        <span class="metric-value">{report_data['model_summary']['layers']}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-name">Total Parameters:</span>
                        <span class="metric-value">{report_data['model_summary']['parameters']:,}</span>
                    </div>
                </div>

                <div class="section">
                    <h2>Black Hole Analysis Summary</h2>
                    <p>This analysis identifies layers with computational black hole properties - 
                    individual weights that concentrate a disproportionate amount of Fisher Information
                    and act as critical control points for the network.</p>

                    <h3>Key Metrics</h3>
        """

        # Add black hole metrics
        for metric, value in report_data['black_hole_metrics'].items():
            html += f"""
                    <div class="metric">
                        <span class="metric-name">{metric.replace('_', ' ').title()}:</span>
                        <span class="metric-value">{value:.4f}</span>
                    </div>
            """

        # Add standard metrics
        html += """
                    <h3>Standard WeightWatcher Metrics</h3>
        """

        for metric, value in report_data['standard_metrics'].items():
            html += f"""
                    <div class="metric">
                        <span class="metric-name">{metric.replace('_', ' ').title()}:</span>
                        <span class="metric-value">{value:.4f}</span>
                    </div>
            """

        # Add top black hole layers
        html += """
                </div>

                <div class="section">
                    <h2>Top Black Hole Layers</h2>
                    <p>These layers show the strongest computational black hole properties.</p>

                    <table>
                        <tr>
                            <th>Layer ID</th>
                            <th>Name</th>
                            <th>Type</th>
                            <th>Black Hole Score</th>
                            <th>Gini Coefficient</th>
                            <th>Dominance Ratio</th>
                            <th>Participation Ratio</th>
                        </tr>
        """

        for layer in report_data['top_black_hole_layers']:
            html += f"""
                        <tr>
                            <td>{layer['id']}</td>
                            <td>{layer['name']}</td>
                            <td>{layer['type']}</td>
                            <td>{layer['black_hole_score']:.4f}</td>
                            <td>{layer['gini_coefficient']:.4f if layer['gini_coefficient'] is not None else 'N/A'}</td>
                            <td>{layer['dominance_ratio']:.4f if layer['dominance_ratio'] is not None else 'N/A'}</td>
                            <td>{layer['participation_ratio']:.4f if layer['participation_ratio'] is not None else 'N/A'}</td>
                        </tr>
            """

        # Add super weights if available
        if 'super_weights' in report_data['top_black_hole_layers'][0]:
            html += """
                    </table>

                    <h3>Super Weights in Top Black Hole Layer</h3>
                    <p>These are the individual weights that concentrate the most Fisher Information.</p>

                    <table>
                        <tr>
                            <th>Index i</th>
                            <th>Index j</th>
                            <th>Contribution</th>
                        </tr>
            """

            for sw in report_data['top_black_hole_layers'][0]['super_weights'][:10]:  # Show top 10
                html += f"""
                        <tr>
                            <td>{sw['i']}</td>
                            <td>{sw['j']}</td>
                            <td>{sw['contribution']:.6f}</td>
                        </tr>
                """

        # Add visualizations
        html += """
                    </table>
                </div>

                <div class="section">
                    <h2>Visualizations</h2>
                    <p>These visualizations show the relationship between different metrics and identify black hole layers.</p>

                    <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                        <div style="flex: 0 0 48%; margin-bottom: 20px;">
                            <h3>Black Hole Score vs. Power-Law Exponent</h3>
                            <img src="visualizations/blackhole_vs_alpha.png" style="width: 100%;" alt="Black Hole Score vs Alpha">
                        </div>
                        <div style="flex: 0 0 48%; margin-bottom: 20px;">
                            <h3>Top Layers by Black Hole Score</h3>
                            <img src="visualizations/top_blackhole_layers.png" style="width: 100%;" alt="Top Black Hole Layers">
                        </div>
                        <div style="flex: 0 0 48%; margin-bottom: 20px;">
                            <h3>Relationship Between Black Hole Metrics</h3>
                            <img src="visualizations/blackhole_metrics_relationship.png" style="width: 100%;" alt="Black Hole Metrics Relationship">
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>Interpretation</h2>
                    <p>The analysis reveals computational black holes in the neural network - individual weights that concentrate a disproportionate amount of Fisher Information and act as critical control points.</p>

                    <h3>Key Findings</h3>
                    <ul>
                        <li><strong>Black Hole Dynamics:</strong> Some layers exhibit extreme concentration of Fisher Information in just a few parameters.</li>
                        <li><strong>Critical Parameters:</strong> These "super weights" behave like computational black holes, absorbing gradient energy and radiating structured activations.</li>
                        <li><strong>Implications:</strong> Model compression, pruning, and quantization should be performed with care to preserve these critical parameters.</li>
                    </ul>

                    <h3>Recommendations</h3>
                    <ul>
                        <li>Preserve super weights during model compression or quantization.</li>
                        <li>Consider targeted distillation of layers with high black hole scores.</li>
                        <li>Monitor these parameters during fine-tuning to prevent catastrophic forgetting.</li>
                    </ul>
                </div>

                <div class="footer">
                    <p>Generated using TensorFlow WeightWatcher with Black Hole Metrics extension</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        with open(os.path.join(savedir, 'report.html'), 'w') as f:
            f.write(html)

    def test_ablation(self,
                      test_data: Tuple[np.ndarray, np.ndarray],
                      metrics: List[str] = ['loss', 'accuracy']) -> Dict[str, float]:
        """
        Test the impact of ablating super weights on model performance.

        Args:
            test_data: Tuple of (x_test, y_test) for evaluation.
            metrics: List of metrics to evaluate.

        Returns:
            Dictionary comparing original and ablated model performance.
        """
        if self.black_hole_analysis is None:
            logger.warning("Cannot perform ablation test: black hole analysis missing")
            return {}

        x_test, y_test = test_data

        # Evaluate original model
        logger.info("Evaluating original model")
        original_results = self.model.evaluate(x_test, y_test, verbose=0)

        # Convert to dictionary if it's a list
        if isinstance(original_results, list):
            original_results = dict(zip(self.model.metrics_names, original_results))

        # Create ablated model
        logger.info("Creating ablated model")
        with tempfile.TemporaryDirectory() as tmpdir:
            ablated_model = self.black_hole_metrics.ablate_super_weights(
                output_model_path=os.path.join(tmpdir, 'ablated_model.keras')
            )

            # Evaluate ablated model
            logger.info("Evaluating ablated model")
            ablated_results = ablated_model.evaluate(x_test, y_test, verbose=0)

            # Convert to dictionary if it's a list
            if isinstance(ablated_results, list):
                ablated_results = dict(zip(self.model.metrics_names, ablated_results))

        # Compare results
        comparison = {}
        for metric in original_results:
            if metric in ablated_results:
                original_val = original_results[metric]
                ablated_val = ablated_results[metric]
                change = ablated_val - original_val
                percent_change = (change / original_val * 100) if original_val != 0 else float('inf')

                comparison[metric] = {
                    'original': original_val,
                    'ablated': ablated_val,
                    'change': change,
                    'percent_change': percent_change
                }

        logger.info(f"Ablation test results: {comparison}")
        return comparison


def analyze_model_black_holes(
        model: keras.Model,
        output_dir: str = 'black_hole_analysis',
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        plot: bool = True,
        detailed: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze a model for computational black holes.

    Args:
        model: Keras model to analyze.
        output_dir: Directory to save analysis results.
        test_data: Optional tuple of (x_test, y_test) for ablation testing.
        plot: Whether to create visualizations.
        detailed: Whether to perform detailed analysis of super weights.

    Returns:
        Dictionary with analysis results.
    """
    analyzer = BlackHoleAnalyzer(model)
    analysis_results = analyzer.analyze(
        plot=plot,
        savedir=output_dir,
        detailed=detailed
    )

    results = {
        'analysis': analysis_results,
        'summary': {
            'standard': analyzer.watcher.get_summary(),
            'black_hole': analyzer.black_hole_metrics.get_summary()
        }
    }

    # Perform ablation test if test data provided
    if test_data is not None:
        results['ablation_test'] = analyzer.test_ablation(test_data)

    return results