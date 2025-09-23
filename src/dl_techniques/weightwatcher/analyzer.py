"""
Main WeightWatcher interface - Comprehensive neural network weight analysis

This module provides the main interface for analyzing Keras models using spectral methods,
power-law analysis, and concentration metrics to assess training quality and model complexity.
"""

import os
import json
import keras
import tempfile
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .weightwatcher import WeightWatcher
from .weights_utils import save_layer_analysis_plots
from .constants import HIGH_CONCENTRATION_PERCENTILE

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def _find_analyzable_layer_indices(model: keras.Model) -> List[int]:
    """
    Finds the indices of all layers in a model that contain weights.

    This function automatically detects which layers are suitable for analysis by checking
    for the presence of weight tensors. This approach works for standard layers, custom
    layers using self.add_weight, and container layers.

    :param model: The Keras model to inspect for analyzable layers
    :type model: keras.Model
    :return: List of integer indices for layers that have weights
    :rtype: List[int]

    .. note::
        The function checks for the presence of weight tensors using the layer's
        get_weights() method, which is the most reliable way to determine if a
        layer contains analyzable parameters.
    """
    analyzable_indices = []

    # Iterate through all layers in the model
    for idx, layer in enumerate(model.layers):
        # Check if layer has any weight tensors - this works for standard layers,
        # custom layers using self.add_weight, and container layers
        if layer.get_weights():
            analyzable_indices.append(idx)

    # Log results for user awareness
    if not analyzable_indices:
        logger.warning(
            "Could not automatically find any layers with weights in the model. "
            "Analysis may not be possible."
        )
    else:
        logger.info(
            f"Automatically found {len(analyzable_indices)} layers with weights "
            "for analysis."
        )

    return analyzable_indices

# ---------------------------------------------------------------------

def analyze_model(
        model: keras.Model,
        layers: Optional[List[int]] = None,
        plot: bool = True,
        concentration_analysis: bool = True,
        randomize: bool = False,
        savedir: str = 'ww_analysis',
        min_evals: int = 10,
        max_evals: int = 15000,
        detailed_plots: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive analysis of a Keras model's weight matrices.

    Performs spectral analysis, power-law fitting, and concentration analysis to assess
    training quality, model complexity, and information concentration. Automatically
    detects standard, custom, and container layers with weights.

    :param model: Keras model to analyze
    :type model: keras.Model
    :param layers: List of layer indices to analyze. If None, automatically finds all
                  layers with weights, including custom layers
    :type layers: Optional[List[int]]
    :param plot: Whether to create analysis visualizations
    :type plot: bool
    :param concentration_analysis: Whether to perform concentration analysis
    :type concentration_analysis: bool
    :param randomize: Whether to analyze randomized weight matrices for comparison
    :type randomize: bool
    :param savedir: Directory to save analysis results and visualizations
    :type savedir: str
    :param min_evals: Minimum number of eigenvalues required for analysis
    :type min_evals: int
    :param max_evals: Maximum number of eigenvalues to analyze
    :type max_evals: int
    :param detailed_plots: Whether to create detailed layer-by-layer plots
    :type detailed_plots: bool
    :return: Dictionary containing analysis results, summary metrics, high concentration
             layers, and recommendations
    :rtype: Dict[str, Any]

    .. note::
        The returned dictionary contains the following keys:

        - 'analysis': DataFrame with detailed layer analysis
        - 'summary': Dictionary with summary metrics
        - 'high_concentration_layers': List of layers with high concentration scores
        - 'recommendations': List of analysis-based recommendations
    """
    logger.info(f"Starting comprehensive analysis of model: {model.name}")

    # Auto-detect analyzable layers if none specified
    # This enables support for custom layers out-of-the-box
    if layers is None:
        layers = _find_analyzable_layer_indices(model)
        if not layers:
            logger.error("No analyzable layers found. Aborting analysis.")
            return {
                'analysis': pd.DataFrame(),
                'summary': {},
                'high_concentration_layers': [],
                'recommendations': ["No analyzable layers were found in the model."]
            }

    # Create output directory structure
    os.makedirs(savedir, exist_ok=True)

    # Initialize WeightWatcher analyzer
    watcher = WeightWatcher(model)

    # Perform comprehensive spectral and concentration analysis
    analysis_df = watcher.analyze(
        layers=layers,
        min_evals=min_evals,
        max_evals=max_evals,
        plot=plot,
        randomize=randomize,
        concentration_analysis=concentration_analysis,
        savefig=os.path.join(savedir, 'plots') if plot else False
    )

    # Handle case where analysis produces no results
    if analysis_df.empty:
        logger.warning("Analysis did not produce any results.")
        return {
            'analysis': pd.DataFrame(),
            'summary': {},
            'high_concentration_layers': [],
            'recommendations': ["Analysis ran but did not yield any metrics. "
                                "Check if layers meet the minimum eigenvalue criteria."]
        }

    # Extract summary metrics from analysis
    summary = watcher.get_summary()

    # Identify layers with high information concentration
    high_concentration_layers = []
    if concentration_analysis and 'concentration_score' in analysis_df.columns:
        # Use percentile-based threshold to identify outliers
        threshold = analysis_df['concentration_score'].quantile(HIGH_CONCENTRATION_PERCENTILE)
        high_conc_mask = analysis_df['concentration_score'] > threshold
        high_concentration_layers = analysis_df[high_conc_mask].index.tolist()

    # Generate analysis-based recommendations
    recommendations = _generate_recommendations(analysis_df, summary)

    # Create detailed layer visualizations if requested
    if detailed_plots and plot:
        plot_dir = os.path.join(savedir, 'detailed_plots')
        # Only plot layers that were successfully analyzed
        analyzed_layer_indices = analysis_df.index.tolist()
        if analyzed_layer_indices:
            save_layer_analysis_plots(
                model,
                analyzed_layer_indices,
                plot_dir,
                plot_types=['weights', 'statistics']
            )

    # Compile all analysis results
    results = {
        'analysis': analysis_df,
        'summary': summary,
        'high_concentration_layers': high_concentration_layers,
        'recommendations': recommendations
    }

    # Save analysis results to CSV format
    analysis_df.to_csv(os.path.join(savedir, 'layer_analysis.csv'))

    # Save summary and recommendations as structured JSON
    summary_data = {
        'model_name': model.name,
        'total_parameters': model.count_params(),
        'layers_analyzed': len(analysis_df),
        'summary_metrics': summary,
        'high_concentration_layers': high_concentration_layers,
        'recommendations': recommendations
    }

    with open(os.path.join(savedir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Generate comprehensive HTML report
    _generate_html_report(results, savedir)

    logger.info(f"Analysis complete. Results saved to {savedir}")

    return results

# ---------------------------------------------------------------------

def compare_models(
        original_model: keras.Model,
        modified_model: keras.Model,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        savedir: str = 'model_comparison'
) -> Dict[str, Any]:
    """
    Compare two models using WeightWatcher analysis.

    Useful for comparing original vs. pruned/quantized/fine-tuned models to assess
    the impact of model modifications on weight structure and performance.

    :param original_model: Original model for comparison baseline
    :type original_model: keras.Model
    :param modified_model: Modified model to compare against original
    :type modified_model: keras.Model
    :param test_data: Optional tuple of (x_test, y_test) for performance evaluation
    :type test_data: Optional[Tuple[np.ndarray, np.ndarray]]
    :param savedir: Directory to save comparison results
    :type savedir: str
    :return: Dictionary containing comparison results and recommendations
    :rtype: Dict[str, Any]

    .. note::
        The comparison includes spectral metrics, concentration analysis, and
        optionally performance metrics if test data is provided.
    """
    logger.info("Starting model comparison analysis")

    # Create output directory
    os.makedirs(savedir, exist_ok=True)

    # Initialize WeightWatcher analyzers for both models
    watcher_original = WeightWatcher(original_model)
    watcher_modified = WeightWatcher(modified_model)

    # Perform analysis on both models without plotting to speed up comparison
    original_analysis = watcher_original.analyze(plot=False, concentration_analysis=True)
    modified_analysis = watcher_modified.analyze(plot=False, concentration_analysis=True)

    # Extract summary metrics from both analyses
    original_summary = watcher_original.get_summary()
    modified_summary = watcher_modified.get_summary()

    # Compare spectral and concentration metrics between models
    metric_comparison = {}
    for metric in original_summary:
        if metric in modified_summary:
            original_val = original_summary[metric]
            modified_val = modified_summary[metric]
            change = modified_val - original_val
            # Calculate percentage change, handling division by zero
            percent_change = (change / original_val * 100) if original_val != 0 else float('inf')

            metric_comparison[metric] = {
                'original': original_val,
                'modified': modified_val,
                'absolute_change': change,
                'percent_change': percent_change
            }

    # Performance comparison if test data is provided
    performance_comparison = {}
    if test_data is not None:
        x_test, y_test = test_data

        # Evaluate both models on test data
        original_metrics = original_model.evaluate(x_test, y_test, verbose=0)
        modified_metrics = modified_model.evaluate(x_test, y_test, verbose=0)

        # Handle both single metric and multiple metrics cases
        if isinstance(original_metrics, list):
            original_metrics = dict(zip(original_model.metrics_names, original_metrics))
            modified_metrics = dict(zip(modified_model.metrics_names, modified_metrics))

        # Compare performance metrics
        for metric_name in original_metrics:
            if metric_name in modified_metrics:
                orig_val = original_metrics[metric_name]
                mod_val = modified_metrics[metric_name]
                change = mod_val - orig_val
                percent_change = (change / orig_val * 100) if orig_val != 0 else float('inf')

                performance_comparison[metric_name] = {
                    'original': orig_val,
                    'modified': mod_val,
                    'absolute_change': change,
                    'percent_change': percent_change
                }

    # Generate recommendations based on the comparison
    comparison_recommendations = _generate_comparison_recommendations(
        metric_comparison, performance_comparison
    )

    # Compile comprehensive comparison results
    comparison_results = {
        'original_analysis': original_analysis,
        'modified_analysis': modified_analysis,
        'metric_comparison': metric_comparison,
        'performance_comparison': performance_comparison,
        'recommendations': comparison_recommendations
    }

    # Save comparison results to JSON
    with open(os.path.join(savedir, 'model_comparison.json'), 'w') as f:
        json.dump({
            'metric_comparison': metric_comparison,
            'performance_comparison': performance_comparison,
            'recommendations': comparison_recommendations
        }, f, indent=2, default=str)

    # Save detailed analysis results for both models
    original_analysis.to_csv(os.path.join(savedir, 'original_model_analysis.csv'))
    modified_analysis.to_csv(os.path.join(savedir, 'modified_model_analysis.csv'))

    logger.info(f"Model comparison complete. Results saved to {savedir}")

    return comparison_results

# ---------------------------------------------------------------------

def create_smoothed_model(
        model: keras.Model,
        method: str = 'detX',
        percent: float = 0.8,
        save_path: Optional[str] = None,
        analyze_smoothed: bool = True
) -> Union[keras.Model, Tuple[keras.Model, Dict[str, Any]]]:
    """
    Create a smoothed version of the model using SVD truncation.

    Applies Singular Value Decomposition (SVD) based smoothing to reduce noise
    and potentially improve model generalization by truncating smaller singular values.

    :param model: Original model to smooth
    :type model: keras.Model
    :param method: Smoothing method - 'svd', 'detX', or 'lambda_min'
    :type method: str
    :param percent: Percentage of singular values to keep (for 'svd' method)
    :type percent: float
    :param save_path: Optional path to save the smoothed model
    :type save_path: Optional[str]
    :param analyze_smoothed: Whether to analyze the smoothed model and return comparison
    :type analyze_smoothed: bool
    :return: Smoothed model alone, or tuple of (smoothed_model, comparison_results)
    :rtype: Union[keras.Model, Tuple[keras.Model, Dict[str, Any]]]

    .. note::
        If analyze_smoothed is True, the function returns a tuple containing both
        the smoothed model and a comparison analysis between original and smoothed versions.
    """
    logger.info(f"Creating smoothed model using method: {method}")

    # Initialize WeightWatcher and create smoothed model
    watcher = WeightWatcher(model)
    smoothed_model = watcher.create_smoothed_model(
        method=method,
        percent=percent,
        save_path=save_path
    )

    # Return only the smoothed model if analysis is not requested
    if not analyze_smoothed:
        return smoothed_model

    # Compare original and smoothed models to assess impact
    with tempfile.TemporaryDirectory() as tmpdir:
        comparison_results = compare_models(
            model,
            smoothed_model,
            savedir=tmpdir
        )

    return smoothed_model, comparison_results

# ---------------------------------------------------------------------

def get_critical_layers(
        model: keras.Model,
        criterion: str = 'concentration',
        top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Identify the most critical layers in the model based on various criteria.

    Analyzes the model to find layers that are most important according to
    different spectral and structural metrics, helping guide optimization decisions.

    :param model: Keras model to analyze
    :type model: keras.Model
    :param criterion: Criterion for identifying critical layers:

                      - 'concentration': Layers with highest concentration scores
                      - 'alpha': Layers with most extreme alpha values
                      - 'entropy': Layers with lowest entropy (most structured)
                      - 'parameters': Layers with most parameters
    :type criterion: str
    :param top_k: Number of top layers to return
    :type top_k: int
    :return: List of dictionaries containing layer information and metrics
    :rtype: List[Dict[str, Any]]

    .. note::
        Each returned dictionary contains layer_id, name, type, rank, and
        relevant metric values for the specified criterion.
    """
    # Perform analysis without plotting for efficiency
    watcher = WeightWatcher(model)
    analysis = watcher.analyze(plot=False, concentration_analysis=True)

    # Return empty list if analysis failed
    if analysis.empty:
        return []

    # Sort layers based on the specified criterion
    if criterion == 'concentration' and 'concentration_score' in analysis.columns:
        # Higher concentration scores indicate more critical layers
        sorted_layers = analysis.sort_values('concentration_score', ascending=False)
    elif criterion == 'alpha' and 'alpha' in analysis.columns:
        # Sort by distance from ideal alpha range (2-6) - more extreme values are more critical
        analysis['alpha_extremity'] = analysis['alpha'].apply(
            lambda x: min(abs(x - 2), abs(x - 6)) if x > 0 else float('inf')
        )
        sorted_layers = analysis.sort_values('alpha_extremity', ascending=False)
    elif criterion == 'entropy' and 'entropy' in analysis.columns:
        # Lower entropy indicates more structured/critical layers
        sorted_layers = analysis.sort_values('entropy', ascending=True)
    elif criterion == 'parameters':
        # More parameters generally indicate more critical layers
        sorted_layers = analysis.sort_values('num_params', ascending=False)
    else:
        logger.warning(f"Criterion '{criterion}' not available or recognized")
        return []

    # Extract information for top critical layers
    critical_layers = []
    for idx, (layer_id, row) in enumerate(sorted_layers.head(top_k).iterrows()):
        # Basic layer information
        layer_info = {
            'layer_id': int(layer_id),
            'name': row['name'],
            'type': row['layer_type'],
            'rank': idx + 1,
            'parameters': int(row['num_params']) if 'num_params' in row else 0
        }

        # Add relevant spectral and concentration metrics
        for metric in ['alpha', 'entropy', 'concentration_score', 'gini_coefficient']:
            if metric in row:
                layer_info[metric] = float(row[metric])

        critical_layers.append(layer_info)

    return critical_layers

# ---------------------------------------------------------------------

def _generate_recommendations(
        analysis_df: pd.DataFrame,
        summary: Dict[str, float]
) -> List[str]:
    """
    Generate analysis-based recommendations for model optimization.

    Analyzes spectral metrics to provide actionable insights about training quality,
    model structure, and potential optimization opportunities.

    :param analysis_df: DataFrame containing detailed layer analysis
    :type analysis_df: pd.DataFrame
    :param summary: Dictionary with summary metrics across all layers
    :type summary: Dict[str, float]
    :return: List of actionable recommendations
    :rtype: List[str]
    """
    recommendations = []

    # Alpha-based training quality recommendations
    if 'alpha' in summary:
        mean_alpha = summary['alpha']
        if mean_alpha < 2.0:
            recommendations.append(
                "Model may be over-trained (low α). Consider early stopping or regularization."
            )
        elif mean_alpha > 6.0:
            recommendations.append(
                "Model may be under-trained (high α). Consider training longer or reducing regularization."
            )
        else:
            recommendations.append(
                f"Model training quality appears good (α = {mean_alpha:.2f})."
            )

    # Concentration-based robustness recommendations
    if 'concentration_score' in summary:
        mean_concentration = summary['concentration_score']
        if mean_concentration > 5.0:  # High concentration threshold
            recommendations.append(
                "High information concentration detected. Be careful with pruning/quantization."
            )

        # Identify specific layers with very high concentration
        if 'concentration_score' in analysis_df.columns:
            high_conc_layers = analysis_df[
                analysis_df['concentration_score'] > analysis_df['concentration_score'].quantile(0.9)
                ]
            if not high_conc_layers.empty:
                layer_names = ', '.join(high_conc_layers['name'].tolist()[:3])
                recommendations.append(
                    f"Layers with highest concentration: {layer_names}. Monitor these during optimization."
                )

    # Entropy-based structure recommendations
    if 'entropy' in summary:
        mean_entropy = summary['entropy']
        if mean_entropy < 0.3:
            recommendations.append(
                "Low entropy detected. Model weights are highly structured."
            )

    # Rank loss based smoothing recommendations
    if 'rank_loss' in analysis_df.columns:
        # Identify layers with significant rank loss (>10% of matrix dimension)
        high_rank_loss = analysis_df[analysis_df['rank_loss'] > 0.1 * analysis_df['M']]
        if not high_rank_loss.empty:
            recommendations.append(
                "Some layers show significant rank loss. Consider SVD smoothing."
            )

    return recommendations


def _generate_comparison_recommendations(
        metric_comparison: Dict[str, Dict[str, float]],
        performance_comparison: Dict[str, Dict[str, float]]
) -> List[str]:
    """
    Generate recommendations based on model comparison results.

    Analyzes changes in spectral metrics and performance to provide guidance
    on the impact of model modifications.

    :param metric_comparison: Dictionary comparing spectral metrics between models
    :type metric_comparison: Dict[str, Dict[str, float]]
    :param performance_comparison: Dictionary comparing performance metrics between models
    :type performance_comparison: Dict[str, Dict[str, float]]
    :return: List of comparison-based recommendations
    :rtype: List[str]
    """
    recommendations = []

    # Analyze alpha (training quality) changes
    if 'alpha' in metric_comparison:
        alpha_change = metric_comparison['alpha']['percent_change']
        if abs(alpha_change) > 20:  # Significant alpha change threshold
            recommendations.append(
                f"Significant alpha change ({alpha_change:.1f}%). Monitor training quality."
            )

    # Analyze concentration changes (model robustness)
    if 'concentration_score' in metric_comparison:
        conc_change = metric_comparison['concentration_score']['percent_change']
        if conc_change > 50:
            recommendations.append(
                "Concentration increased significantly. Model may have become more brittle."
            )
        elif conc_change < -30:
            recommendations.append(
                "Concentration decreased significantly. Model may be more robust."
            )

    # Analyze performance metric changes
    if performance_comparison:
        for metric, values in performance_comparison.items():
            # Check accuracy degradation
            if 'accuracy' in metric.lower():
                acc_change = values['percent_change']
                if acc_change < -5:  # 5% accuracy drop threshold
                    recommendations.append(
                        f"Accuracy decreased by {abs(acc_change):.1f}%. Consider adjusting parameters."
                    )
            # Check loss increase
            elif 'loss' in metric.lower():
                loss_change = values['percent_change']
                if loss_change > 10:  # 10% loss increase threshold
                    recommendations.append(
                        f"Loss increased by {loss_change:.1f}%. Model performance degraded."
                    )

    return recommendations


def _generate_html_report(results: Dict[str, Any], savedir: str) -> None:
    """
    Generate comprehensive HTML report from analysis results.

    Creates a formatted HTML report containing summary metrics, recommendations,
    and high concentration layer information for easy viewing and sharing.

    :param results: Dictionary containing analysis results
    :type results: Dict[str, Any]
    :param savedir: Directory to save the HTML report
    :type savedir: str
    """
    # Extract components from results dictionary
    analysis_df = results['analysis']
    summary = results['summary']
    recommendations = results['recommendations']
    high_concentration_layers = results.get('high_concentration_layers', [])

    # Generate HTML content with embedded CSS styling
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>WeightWatcher Analysis Report</title>
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
            .recommendation {{ background-color: #f0f8ff; padding: 10px; margin: 5px 0; border-left: 4px solid #0066cc; }}
            .footer {{ margin-top: 30px; font-size: 0.8em; color: #666; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>WeightWatcher Analysis Report</h1>

            <div class="section">
                <h2>Summary Metrics</h2>
    """

    # Add summary metrics section
    for metric, value in summary.items():
        if isinstance(value, (int, float)):
            html_content += f"""
                <div class="metric">
                    <span class="metric-name">{metric.replace('_', ' ').title()}:</span>
                    <span class="metric-value">{value:.4f}</span>
                </div>
            """

    # Add recommendations section
    html_content += """
            </div>

            <div class="section">
                <h2>Recommendations</h2>
    """

    for rec in recommendations:
        html_content += f'<div class="recommendation">{rec}</div>'

    # Add high concentration layers section if any exist
    if high_concentration_layers:
        html_content += """
            </div>

            <div class="section">
                <h2>High Concentration Layers</h2>
                <p>These layers show high information concentration and should be monitored during optimization:</p>
                <ul>
        """

        # Show top 5 high concentration layers with details
        for layer_id in high_concentration_layers[:5]:
            if layer_id in analysis_df.index:
                layer_name = analysis_df.loc[layer_id, 'name']
                conc_score = analysis_df.loc[layer_id, 'concentration_score']
                html_content += f'<li>Layer {layer_id}: {layer_name} (Score: {conc_score:.3f})</li>'

        html_content += "</ul>"

    # Close HTML document with footer
    html_content += """
            </div>

            <div class="footer">
                <p>Generated using Enhanced WeightWatcher</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save HTML report to file
    with open(os.path.join(savedir, 'analysis_report.html'), 'w') as f:
        f.write(html_content)

# ---------------------------------------------------------------------

