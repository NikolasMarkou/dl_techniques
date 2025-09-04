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

from .weightwatcher import WeightWatcher
from .weights_utils import save_layer_analysis_plots
from .constants import HIGH_CONCENTRATION_PERCENTILE

from dl_techniques.utils.logger import logger


def _find_analyzable_layer_indices(model: keras.Model) -> List[int]:
    """
    Finds the indices of all layers in a model that contain weights.

    This function is used to automatically detect which layers, including custom
    layers, are suitable for analysis by checking for the presence of weight tensors.

    Args:
        model: The Keras model to inspect.

    Returns:
        A list of integer indices for layers that have weights.
    """
    analyzable_indices = []
    for idx, layer in enumerate(model.layers):
        # The most reliable way to determine if a layer is analyzable is to
        # check if it has any weight tensors. This works for standard layers,
        # custom layers using self.add_weight, and container layers.
        if layer.get_weights():
            analyzable_indices.append(idx)

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

    This function performs spectral analysis, power-law fitting, and concentration
    analysis to assess training quality, model complexity, and information concentration.
    It can automatically detect standard, custom, and container layers with weights.

    Args:
        model: Keras model to analyze.
        layers: Optional list of layer indices to analyze. If None (default), the function
                will automatically find all layers with weights, including custom layers.
        plot: Whether to create analysis visualizations.
        concentration_analysis: Whether to perform concentration analysis.
        randomize: Whether to analyze randomized weight matrices for comparison.
        savedir: Directory to save analysis results and visualizations.
        min_evals: Minimum number of eigenvalues required for analysis.
        max_evals: Maximum number of eigenvalues to analyze.
        detailed_plots: Whether to create detailed layer-by-layer plots.

    Returns:
        Dictionary containing:
        - 'analysis': DataFrame with detailed layer analysis
        - 'summary': Dictionary with summary metrics
        - 'high_concentration_layers': List of layers with high concentration scores
        - 'recommendations': List of analysis-based recommendations
    """
    logger.info(f"Starting comprehensive analysis of model: {model.name}")

    # If no specific layers are requested, automatically find all analyzable layers.
    # This enables support for custom layers out-of-the-box.
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

    # Create output directory
    os.makedirs(savedir, exist_ok=True)

    # Initialize WeightWatcher
    watcher = WeightWatcher(model)

    # Perform analysis
    analysis_df = watcher.analyze(
        layers=layers,
        min_evals=min_evals,
        max_evals=max_evals,
        plot=plot,
        randomize=randomize,
        concentration_analysis=concentration_analysis,
        savefig=os.path.join(savedir, 'plots') if plot else False
    )

    if analysis_df.empty:
        logger.warning("Analysis did not produce any results.")
        # Return a structured empty result
        return {
            'analysis': pd.DataFrame(),
            'summary': {},
            'high_concentration_layers': [],
            'recommendations': ["Analysis ran but did not yield any metrics. "
                                "Check if layers meet the minimum eigenvalue criteria."]
        }

    # Get summary metrics
    summary = watcher.get_summary()

    # Identify high concentration layers
    high_concentration_layers = []
    if concentration_analysis and 'concentration_score' in analysis_df.columns:
        threshold = analysis_df['concentration_score'].quantile(HIGH_CONCENTRATION_PERCENTILE)
        high_conc_mask = analysis_df['concentration_score'] > threshold
        high_concentration_layers = analysis_df[high_conc_mask].index.tolist()

    # Generate recommendations
    recommendations = _generate_recommendations(analysis_df, summary)

    # Create detailed plots if requested
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

    # Save results
    results = {
        'analysis': analysis_df,
        'summary': summary,
        'high_concentration_layers': high_concentration_layers,
        'recommendations': recommendations
    }

    # Save analysis results as CSV and JSON
    analysis_df.to_csv(os.path.join(savedir, 'layer_analysis.csv'))

    # Save summary and recommendations as JSON
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

    # Generate HTML report
    _generate_html_report(results, savedir)

    logger.info(f"Analysis complete. Results saved to {savedir}")

    return results


def compare_models(
        original_model: keras.Model,
        modified_model: keras.Model,
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        savedir: str = 'model_comparison'
) -> Dict[str, Any]:
    """
    Compare two models using WeightWatcher analysis.

    Useful for comparing original vs. pruned/quantized/fine-tuned models.

    Args:
        original_model: Original model for comparison.
        modified_model: Modified model to compare against original.
        test_data: Optional tuple of (x_test, y_test) for performance evaluation.
        savedir: Directory to save comparison results.

    Returns:
        Dictionary containing comparison results and recommendations.
    """
    logger.info("Starting model comparison analysis")

    os.makedirs(savedir, exist_ok=True)

    # Analyze both models
    watcher_original = WeightWatcher(original_model)
    watcher_modified = WeightWatcher(modified_model)

    original_analysis = watcher_original.analyze(plot=False, concentration_analysis=True)
    modified_analysis = watcher_modified.analyze(plot=False, concentration_analysis=True)

    original_summary = watcher_original.get_summary()
    modified_summary = watcher_modified.get_summary()

    # Compare metrics
    metric_comparison = {}
    for metric in original_summary:
        if metric in modified_summary:
            original_val = original_summary[metric]
            modified_val = modified_summary[metric]
            change = modified_val - original_val
            percent_change = (change / original_val * 100) if original_val != 0 else float('inf')

            metric_comparison[metric] = {
                'original': original_val,
                'modified': modified_val,
                'absolute_change': change,
                'percent_change': percent_change
            }

    # Performance comparison if test data provided
    performance_comparison = {}
    if test_data is not None:
        x_test, y_test = test_data

        original_metrics = original_model.evaluate(x_test, y_test, verbose=0)
        modified_metrics = modified_model.evaluate(x_test, y_test, verbose=0)

        if isinstance(original_metrics, list):
            original_metrics = dict(zip(original_model.metrics_names, original_metrics))
            modified_metrics = dict(zip(modified_model.metrics_names, modified_metrics))

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

    # Generate comparison recommendations
    comparison_recommendations = _generate_comparison_recommendations(
        metric_comparison, performance_comparison
    )

    # Compile results
    comparison_results = {
        'original_analysis': original_analysis,
        'modified_analysis': modified_analysis,
        'metric_comparison': metric_comparison,
        'performance_comparison': performance_comparison,
        'recommendations': comparison_recommendations
    }

    # Save results
    with open(os.path.join(savedir, 'model_comparison.json'), 'w') as f:
        json.dump({
            'metric_comparison': metric_comparison,
            'performance_comparison': performance_comparison,
            'recommendations': comparison_recommendations
        }, f, indent=2, default=str)

    original_analysis.to_csv(os.path.join(savedir, 'original_model_analysis.csv'))
    modified_analysis.to_csv(os.path.join(savedir, 'modified_model_analysis.csv'))

    logger.info(f"Model comparison complete. Results saved to {savedir}")

    return comparison_results


def create_smoothed_model(
        model: keras.Model,
        method: str = 'detX',
        percent: float = 0.8,
        save_path: Optional[str] = None,
        analyze_smoothed: bool = True
) -> Union[keras.Model, Tuple[keras.Model, Dict[str, Any]]]:
    """
    Create a smoothed version of the model using SVD truncation.

    Args:
        model: Original model to smooth.
        method: Smoothing method ('svd', 'detX', or 'lambda_min').
        percent: Percentage of singular values to keep (for 'svd' method).
        save_path: Optional path to save the smoothed model.
        analyze_smoothed: Whether to analyze the smoothed model and return comparison.

    Returns:
        If analyze_smoothed is False: Smoothed model
        If analyze_smoothed is True: Tuple of (smoothed_model, comparison_results)
    """
    logger.info(f"Creating smoothed model using method: {method}")

    watcher = WeightWatcher(model)
    smoothed_model = watcher.create_smoothed_model(
        method=method,
        percent=percent,
        save_path=save_path
    )

    if not analyze_smoothed:
        return smoothed_model

    # Compare original and smoothed models
    with tempfile.TemporaryDirectory() as tmpdir:
        comparison_results = compare_models(
            model,
            smoothed_model,
            savedir=tmpdir
        )

    return smoothed_model, comparison_results


def get_critical_layers(
        model: keras.Model,
        criterion: str = 'concentration',
        top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Identify the most critical layers in the model based on various criteria.

    Args:
        model: Keras model to analyze.
        criterion: Criterion for identifying critical layers:
                  - 'concentration': Layers with highest concentration scores
                  - 'alpha': Layers with most extreme alpha values
                  - 'entropy': Layers with lowest entropy (most structured)
                  - 'parameters': Layers with most parameters
        top_k: Number of top layers to return.

    Returns:
        List of dictionaries containing layer information and metrics.
    """
    watcher = WeightWatcher(model)
    analysis = watcher.analyze(plot=False, concentration_analysis=True)

    if analysis.empty:
        return []

    # Sort by criterion
    if criterion == 'concentration' and 'concentration_score' in analysis.columns:
        sorted_layers = analysis.sort_values('concentration_score', ascending=False)
    elif criterion == 'alpha' and 'alpha' in analysis.columns:
        # Sort by distance from ideal alpha range (2-6)
        analysis['alpha_extremity'] = analysis['alpha'].apply(
            lambda x: min(abs(x - 2), abs(x - 6)) if x > 0 else float('inf')
        )
        sorted_layers = analysis.sort_values('alpha_extremity', ascending=False)
    elif criterion == 'entropy' and 'entropy' in analysis.columns:
        sorted_layers = analysis.sort_values('entropy', ascending=True)
    elif criterion == 'parameters':
        sorted_layers = analysis.sort_values('num_params', ascending=False)
    else:
        logger.warning(f"Criterion '{criterion}' not available or recognized")
        return []

    # Extract top layers
    critical_layers = []
    for idx, (layer_id, row) in enumerate(sorted_layers.head(top_k).iterrows()):
        layer_info = {
            'layer_id': int(layer_id),
            'name': row['name'],
            'type': row['layer_type'],
            'rank': idx + 1,
            'parameters': int(row['num_params']) if 'num_params' in row else 0
        }

        # Add relevant metrics
        for metric in ['alpha', 'entropy', 'concentration_score', 'gini_coefficient']:
            if metric in row:
                layer_info[metric] = float(row[metric])

        critical_layers.append(layer_info)

    return critical_layers


def _generate_recommendations(
        analysis_df: pd.DataFrame,
        summary: Dict[str, float]
) -> List[str]:
    """Generate analysis-based recommendations."""
    recommendations = []

    # Alpha-based recommendations
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

    # Concentration-based recommendations
    if 'concentration_score' in summary:
        mean_concentration = summary['concentration_score']
        if mean_concentration > 5.0:  # High concentration
            recommendations.append(
                "High information concentration detected. Be careful with pruning/quantization."
            )

        # Check for layers with very high concentration
        if 'concentration_score' in analysis_df.columns:
            high_conc_layers = analysis_df[
                analysis_df['concentration_score'] > analysis_df['concentration_score'].quantile(0.9)
                ]
            if not high_conc_layers.empty:
                layer_names = ', '.join(high_conc_layers['name'].tolist()[:3])
                recommendations.append(
                    f"Layers with highest concentration: {layer_names}. Monitor these during optimization."
                )

    # Entropy-based recommendations
    if 'entropy' in summary:
        mean_entropy = summary['entropy']
        if mean_entropy < 0.3:
            recommendations.append(
                "Low entropy detected. Model weights are highly structured."
            )

    # Rank loss recommendations
    if 'rank_loss' in analysis_df.columns:
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
    """Generate recommendations based on model comparison."""
    recommendations = []

    # Check alpha changes
    if 'alpha' in metric_comparison:
        alpha_change = metric_comparison['alpha']['percent_change']
        if abs(alpha_change) > 20:
            recommendations.append(
                f"Significant alpha change ({alpha_change:.1f}%). Monitor training quality."
            )

    # Check concentration changes
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

    # Check performance changes
    if performance_comparison:
        for metric, values in performance_comparison.items():
            if 'accuracy' in metric.lower():
                acc_change = values['percent_change']
                if acc_change < -5:
                    recommendations.append(
                        f"Accuracy decreased by {abs(acc_change):.1f}%. Consider adjusting parameters."
                    )
            elif 'loss' in metric.lower():
                loss_change = values['percent_change']
                if loss_change > 10:
                    recommendations.append(
                        f"Loss increased by {loss_change:.1f}%. Model performance degraded."
                    )

    return recommendations


def _generate_html_report(results: Dict[str, Any], savedir: str) -> None:
    """Generate HTML report from analysis results."""
    analysis_df = results['analysis']
    summary = results['summary']
    recommendations = results['recommendations']
    high_concentration_layers = results.get('high_concentration_layers', [])

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

    # Add summary metrics
    for metric, value in summary.items():
        if isinstance(value, (int, float)):
            html_content += f"""
                <div class="metric">
                    <span class="metric-name">{metric.replace('_', ' ').title()}:</span>
                    <span class="metric-value">{value:.4f}</span>
                </div>
            """

    # Add recommendations
    html_content += """
            </div>

            <div class="section">
                <h2>Recommendations</h2>
    """

    for rec in recommendations:
        html_content += f'<div class="recommendation">{rec}</div>'

    # Add high concentration layers
    if high_concentration_layers:
        html_content += """
            </div>

            <div class="section">
                <h2>High Concentration Layers</h2>
                <p>These layers show high information concentration and should be monitored during optimization:</p>
                <ul>
        """

        for layer_id in high_concentration_layers[:5]:  # Show top 5
            if layer_id in analysis_df.index:
                layer_name = analysis_df.loc[layer_id, 'name']
                conc_score = analysis_df.loc[layer_id, 'concentration_score']
                html_content += f'<li>Layer {layer_id}: {layer_name} (Score: {conc_score:.3f})</li>'

        html_content += "</ul>"

    # Close HTML
    html_content += """
            </div>

            <div class="footer">
                <p>Generated using Enhanced WeightWatcher</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Save HTML report
    with open(os.path.join(savedir, 'analysis_report.html'), 'w') as f:
        f.write(html_content)