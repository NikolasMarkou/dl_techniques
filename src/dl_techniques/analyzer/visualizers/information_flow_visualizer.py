"""
Information Flow Visualization Module

Creates visualizations for information flow analysis results with centralized legend management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from typing import Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisualizer
from ..constants import ACTIVATION_MAGNITUDE_NORMALIZER, LAYER_SPECIALIZATION_MAX_RANK

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

# Figure Layout Constants
FIGURE_SIZE = (14, 10)
GRID_HSPACE = 0.3
GRID_WSPACE = 0.3
SUBPLOT_TOP = 0.93
SUBPLOT_BOTTOM = 0.1
SUBPLOT_LEFT = 0.1
SUBPLOT_RIGHT = 0.92  # Increased from 0.85 to bring plots closer to legend

# Text Styling Constants
TITLE_FONT_SIZE = 16
SUBTITLE_FONT_SIZE = 12
SMALL_TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 8
COLORBAR_FONT_SIZE = 7

# Plot Styling Constants
LINE_WIDTH_STANDARD = 2
MARKER_SIZE_SMALL = 6
MARKER_SIZE_LARGE = 8
ALPHA_FILL = 0.2
ALPHA_STANDARD = 0.8
ALPHA_GRID = 0.3

# Subplot Layout Constants
SUB_WSPACE_WIDE = 0.4
SUB_WSPACE_NARROW = 0.1
SUB_HSPACE_STANDARD = 0.4
SUB_HSPACE_NARROW = 0.1
HEIGHT_RATIOS_EQUAL = [1, 1]
HEIGHT_RATIOS_BOTTOM_HEAVY = [1, 1.5]

# Heatmap Constants
HEATMAP_VMIN = 0
HEATMAP_VMAX = 1
# DECISION plan-2026-10-05-analyzer-audit/F-055
# The colour a heatmap cell with NO measurement is drawn in. Matches the
# `BBOX_COLOR_GRAY` convention already used by `weight_visualizer.py`'s N/A cells.
# Do NOT revert to filling those cells with 0.0: under a fixed `vmax=1.0` and a
# one-directional colormap, 0.0 is the HEALTHIEST end of every one of these three scales.
MISSING_DATA_COLOR = 'lightgray'
COLORBAR_SHRINK = 0.6
INTERPOLATION_METHOD = 'nearest'

# Specialization Analysis Constants
BAR_LABEL_OFFSET = 0.02
AXIS_LIMIT_MIN = 0
AXIS_LIMIT_MAX = 1
# DECISION plan-2026-10-05-analyzer-audit/F-053
# `SATURATION_THRESHOLD = 0.9` is GONE, deliberately. It gated
# `saturation = 1 - positive_ratio if positive_ratio > 0.9 else 0.0`, and
# `positive_ratio` counts activations ABOVE ZERO — so a 95%-dead ReLU scored 0.05, failed
# the `> 0.9` test, and was reported as having ZERO saturation: the most saturated layer
# in the network marked healthy. The measure was also bounded above by 0.1 by
# construction, so under the panel's fixed vmax it rendered as a uniformly blank block.
# Saturation is now unconditionally `1 - positive_ratio`, which is the fraction of
# activations that are non-positive — the quantity the panel's name refers to. Do NOT
# reintroduce a threshold here; if a threshold is wanted it belongs on the DISPLAYED
# value, not on the input.
BALANCE_SCORE_MULTIPLIER = 2
SPECIALIZATION_SCORE_COMPONENTS = 3.0

# Health Metrics Constants
ANNOTATION_X_CENTER = 0.5
ANNOTATION_Y_CENTER = 0.5
TITLE_PAD = 20

# ---------------------------------------------------------------------

class InformationFlowVisualizer(BaseVisualizer):
    """Creates information flow visualizations with centralized legend."""

    def create_visualizations(self) -> None:
        """Create information flow visualizations with single legend."""
        if not self.results.information_flow:
            return

        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = plt.GridSpec(2, 2, figure=fig, hspace=GRID_HSPACE, wspace=GRID_WSPACE)

        # Top row: Flow overview
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_activation_flow_overview(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_effective_rank_evolution(ax2)

        # Bottom row: Actionable insights
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_activation_health_dashboard(ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_layer_specialization_analysis(ax4)

        # Add single figure-level legend
        models_with_data = self._get_models_with_data()
        if models_with_data:
            self._create_figure_legend(fig, title="Models", specific_models=models_with_data)

        plt.suptitle('Information Flow and Activation Analysis',
                    fontsize=TITLE_FONT_SIZE, fontweight='bold')
        fig.subplots_adjust(top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM,
                           left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT)

        if self.config.save_plots:
            self._save_figure(fig, 'information_flow_analysis')
        plt.close(fig)

    def _get_models_with_data(self) -> List[str]:
        """Get models that have information flow data."""
        models_with_data = []
        for model_name in self.model_order:
            if model_name in self.results.information_flow and self.results.information_flow[model_name]:
                models_with_data.append(model_name)
        return models_with_data

    def _get_ordered_layer_analysis(self, model_name: str) -> list:
        """
        Get layer analysis ordered by forward-pass depth.

        DECISION plan-2026-09-01T225724-e79ad4bd/D-021: sort on the
        ``capture_index`` that `InformationFlowAnalyzer` records at forward time.
        Do NOT fall back to returning ``layer_analysis.items()`` unsorted and call
        it network order: the analyzer used to key that dict from the static layer
        walk, which is reverse-alphabetical for a custom Layer's sublayers and was
        measured EXACTLY backwards against the forward pass. Entries without a
        ``capture_index`` (results loaded from a pre-fix artifact) keep their
        relative position. See decisions.md D-021.

        Args:
            model_name: Name of the model

        Returns:
            List of (layer_name, analysis) tuples in forward-pass order
        """
        if model_name not in self.results.information_flow:
            return []

        items = list(self.results.information_flow[model_name].items())

        def depth_of(position: int, analysis: Any) -> int:
            if isinstance(analysis, dict):
                return analysis.get('capture_index', position)
            return position

        return [
            item for _, item in sorted(
                enumerate(items),
                key=lambda pair: depth_of(pair[0], pair[1][1]),
            )
        ]

    def _plot_activation_flow_overview(self, ax) -> None:
        """Plot activation statistics evolution through layers."""
        # Use consistent model ordering
        for model_name in self._sort_models_consistently(list(self.results.information_flow.keys())):
            ordered_layers = self._get_ordered_layer_analysis(model_name)

            if not ordered_layers:
                continue

            means = []
            stds = []
            layer_positions = []
            # DECISION plan-2026-10-05-analyzer-audit/F-049
            # Skip entries that carry no measurement, do NOT index them. The analyzer
            # returns `{'error': ...}` for a layer whose activations could not be
            # interpreted, and that dict has NO `mean_activation` key — the direct index
            # raised `KeyError`, which propagated out of `create_visualizations` to
            # `ModelAnalyzer._create_visualizations` (no try there) and aborted the whole
            # `analyze()` BEFORE `save_results()`. One unreadable activation lost every
            # metric every other model produced.
            for i, (layer_name, analysis) in enumerate(ordered_layers):
                mean_value = analysis.get('mean_activation')
                std_value = analysis.get('std_activation')
                if mean_value is None or std_value is None:
                    logger.debug(
                        f"Model '{model_name}' layer '{layer_name}' has no activation "
                        f"statistics ({analysis.get('error', 'keys absent')}); it is "
                        f"omitted from the activation-flow panel.")
                    continue
                mean_value = float(mean_value)
                std_value = float(std_value)
                # DECISION plan-2026-10-05-analyzer-audit/F-050
                # A non-finite activation is not a measurement. The analyzer does no
                # finite check on captured activations, so a NaN reaching here would
                # silently vanish from `min()`/`max()` autoscaling and paint a NaN cell
                # in the health heatmap. Drop the layer and say so.
                if not (np.isfinite(mean_value) and np.isfinite(std_value)):
                    logger.warning(
                        f"Model '{model_name}' layer '{layer_name}' produced "
                        f"non-finite activation statistics "
                        f"(mean={mean_value!r}, std={std_value!r}); omitted from the "
                        f"activation-flow panel.")
                    continue
                means.append(mean_value)
                stds.append(std_value)
                layer_positions.append(i)

            if means:  # Check if we have data
                means = np.array(means)
                stds = np.array(stds)

                color = self._get_model_color(model_name)
                ax.plot(layer_positions, means, 'o-',
                        linewidth=LINE_WIDTH_STANDARD, markersize=MARKER_SIZE_SMALL,
                        color=color)
                ax.fill_between(layer_positions, means - stds, means + stds,
                                alpha=ALPHA_FILL, color=color)

        ax.set_xlabel('Layer Depth (Network Order)')
        ax.set_ylabel('Activation Statistics')
        ax.set_title('Activation Mean ± Std Evolution')
        # REMOVED: Individual legend - will use figure-level legend
        ax.grid(True, alpha=ALPHA_GRID)

    def _plot_effective_rank_evolution(self, ax) -> None:
        """Plot effective rank evolution through network."""
        # Use consistent model ordering
        for model_name in self._sort_models_consistently(list(self.results.information_flow.keys())):
            ordered_layers = self._get_ordered_layer_analysis(model_name)

            if not ordered_layers:
                continue

            ranks = []
            positions = []
            # DECISION plan-2026-10-05-analyzer-audit/F-051
            # Draw the rank as a SCATTER at each layer's OWN position, never a connected
            # line through the layers that were skipped. The old `ax.plot(positions,
            # ranks)` connected consecutive surviving points, so a network whose last four
            # layers collapsed to rank 0 showed a RISING line that appeared to continue
            # straight through them — hiding exactly the gap a reader opens this panel to
            # see. This is the same "fabricated continuity" defect as F-035's phantom
            # reliability points and F-055's heatmap fill, in a third place.
            for i, (layer_name, analysis) in enumerate(ordered_layers):
                rank = analysis.get('effective_rank')
                # `> 0` alone is not enough: `NaN > 0` is False, so a NaN rank would be
                # silently dropped and read as "collapsed" (F-050). Test finiteness first.
                if rank is None or not np.isfinite(float(rank)) or float(rank) <= 0:
                    continue
                ranks.append(float(rank))
                positions.append(i)

            if ranks:
                color = self._get_model_color(model_name)
                ax.plot(positions, ranks, 'o',
                        linewidth=LINE_WIDTH_STANDARD, markersize=MARKER_SIZE_LARGE,
                        color=color)

        ax.set_xlabel('Layer Depth (Network Order)')
        ax.set_ylabel('Effective Rank')
        ax.set_title('Information Dimensionality Evolution')
        # REMOVED: Individual legend - will use figure-level legend
        ax.grid(True, alpha=ALPHA_GRID)

    def _plot_activation_health_dashboard(self, ax) -> None:
        """Create an activation health dashboard showing model health metrics."""
        if not self.results.information_flow:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER, 'No activation data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activation Health Dashboard')
            ax.axis('off')
            return

        # Prepare health metrics data with consistent ordering
        health_data = []

        for model_name in self._sort_models_consistently(list(self.results.information_flow.keys())):
            ordered_layers = self._get_ordered_layer_analysis(model_name)

            for i, (layer_name, analysis) in enumerate(ordered_layers):
                # DECISION plan-2026-10-05-analyzer-audit/F-052
                # SKIP a layer with no measurement. The defaults used here were
                # `sparsity -> 0.0`, `positive_ratio -> 0.5`, `mean_activation -> 0.0`,
                # so an entry carrying only `{'error': ...}` — a layer that produced NO
                # data at all — was rendered as a measured, unremarkable layer reading
                # "zero dead neurons, not saturated, activation level 0.0", and the 0.5
                # default even fed the SATURATION_THRESHOLD test. A missing measurement
                # must be absent from the panel, not drawn as a healthy one.
                sparsity = analysis.get('sparsity')
                positive_ratio = analysis.get('positive_ratio')
                mean_activation = analysis.get('mean_activation')

                if (sparsity is None or positive_ratio is None
                        or mean_activation is None):
                    logger.debug(
                        f"Model '{model_name}' layer '{layer_name}' has no health "
                        f"statistics ({analysis.get('error', 'keys absent')}); omitted "
                        f"from the health dashboard.")
                    continue

                # F-050: a non-finite statistic is not a measurement either.
                if not all(np.isfinite(float(v))
                           for v in (sparsity, positive_ratio, mean_activation)):
                    logger.warning(
                        f"Model '{model_name}' layer '{layer_name}' produced "
                        f"non-finite health statistics; omitted from the dashboard.")
                    continue

                sparsity = float(sparsity)
                positive_ratio = float(positive_ratio)
                mean_activation = abs(float(mean_activation))

                # Health indicators
                dead_neurons = sparsity  # High sparsity indicates dead neurons
                # DECISION plan-2026-10-05-analyzer-audit/F-053
                # Saturation is `1 - positive_ratio` for a layer that is ALMOST ENTIRELY
                # NON-POSITIVE, not `positive_ratio > 0.9`. `positive_ratio` counts
                # activations ABOVE ZERO, so a 95%-dead ReLU has positive_ratio ~= 0.05 and
                # the old test scored its saturation 0.0 — "healthy" — while it is the
                # most saturated layer in the network. The measure was also near-always
                # zero by construction (it is < 0.1 whenever non-zero), which is why the
                # panel rendered as a uniformly blank orange block under a fixed vmax.
                saturation = 1.0 - positive_ratio
                activation_magnitude = min(mean_activation,
                                           ACTIVATION_MAGNITUDE_NORMALIZER) / ACTIVATION_MAGNITUDE_NORMALIZER

                # Use layer index for consistent ordering instead of truncated layer name
                layer_label = f"L{i}"  # This ensures consistent ordering

                health_data.append({
                    'Model': model_name,
                    'Layer': layer_label,
                    'Layer_Index': i,  # Keep index for sorting
                    'Dead Neurons': dead_neurons,
                    'Saturation': saturation,
                    'Activation Level': activation_magnitude
                })

        if health_data:
            df = pd.DataFrame(health_data)

            # Sort by layer index to ensure correct order
            df = df.sort_values(['Model', 'Layer_Index'])

            # Create heatmap data with consistent model ordering
            models = self._sort_models_consistently(list(df['Model'].unique()))
            max_layers = getattr(self.config, 'max_layers_info_flow', 8)

            # Get unique layers and limit based on config, maintaining order
            unique_layers = sorted(df['Layer_Index'].unique())[:max_layers]
            layer_labels = [f"L{i}" for i in unique_layers]

            # Create separate heatmaps for each metric
            metrics = ['Dead Neurons', 'Saturation', 'Activation Level']

            # Create subplots within the main axis
            gs_sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(),
                                             wspace=SUB_WSPACE_WIDE, hspace=SUB_HSPACE_NARROW)

            for idx, metric in enumerate(metrics):
                ax_sub = plt.subplot(gs_sub[0, idx])

                # Filter data for layers within our limit
                df_filtered = df[df['Layer_Index'].isin(unique_layers)]

                # DECISION plan-2026-10-05-analyzer-audit/F-055
                # Fill MISSING cells with NaN, not 0.0. `fill_value=0` rendered a model
                # that failed to capture layers 5-8, or that reported no entry for them,
                # as all-zero under the 'Reds'/'Greens' colormaps with a fixed vmax=1.0 —
                # i.e. ZERO dead neurons and MINIMUM activation magnitude, the healthiest
                # end of the scale. A cell with no measurement now renders as the colormap's
                # "bad data" colour instead, which is the same convention
                # `weight_visualizer.py` already uses for its N/A cells.
                heatmap_data = df_filtered.pivot_table(
                    values=metric,
                    index='Model',
                    columns='Layer_Index'
                )

                # Ensure we have the right models in the right order. `reindex` WITHOUT
                # `fill_value` leaves an absent model as all-NaN rather than as all-0.0
                # for the same reason.
                heatmap_data = heatmap_data.reindex(models)

                # Ensure columns are in the right order. No `fill_value` — see F-055.
                heatmap_data = heatmap_data.reindex(columns=unique_layers)

                # Choose colormap based on metric
                if metric == 'Dead Neurons':
                    cmap_name = 'Reds'
                    vmax = HEATMAP_VMAX
                elif metric == 'Saturation':
                    cmap_name = 'Oranges'
                    vmax = HEATMAP_VMAX
                else:  # Activation Level
                    cmap_name = 'Greens'
                    vmax = HEATMAP_VMAX

                # F-055: give the colormap an explicit "no measurement" colour, the same
                # convention `weight_visualizer.py` uses for its N/A cells. Without it
                # matplotlib renders NaN as transparent, so a missing cell shows the
                # white axes background — which under 'Greens' reads as healthy.
                cmap = plt.get_cmap(cmap_name).copy()
                cmap.set_bad(color=MISSING_DATA_COLOR)

                # Create heatmap
                im = ax_sub.imshow(heatmap_data.values, cmap=cmap, aspect='auto',
                                   vmin=HEATMAP_VMIN, vmax=vmax,
                                   interpolation=INTERPOLATION_METHOD)

                # Set labels
                ax_sub.set_title(metric, fontsize=SMALL_TITLE_FONT_SIZE, fontweight='bold')
                ax_sub.set_xticks(range(len(unique_layers)))
                ax_sub.set_xticklabels(layer_labels, rotation=45, ha='right',
                                      fontsize=LABEL_FONT_SIZE)
                ax_sub.set_yticks(range(len(heatmap_data.index)))

                # Only show model names on the leftmost heatmap (truncated for space)
                if idx == 0:
                    truncated_names = [name[:8] + '...' if len(name) > 8 else name for name in heatmap_data.index]
                    ax_sub.set_yticklabels(truncated_names, fontsize=LABEL_FONT_SIZE)
                else:
                    ax_sub.set_yticklabels([])

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax_sub, shrink=COLORBAR_SHRINK)
                cbar.ax.tick_params(labelsize=COLORBAR_FONT_SIZE)

            ax.axis('off')  # Hide the parent axis
            ax.set_title('Activation Health Dashboard (Network Order)',
                        fontsize=SUBTITLE_FONT_SIZE, fontweight='bold', pad=TITLE_PAD)
        else:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'Insufficient data for health analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Activation Health Dashboard')
            ax.axis('off')

    def _plot_layer_specialization_analysis(self, ax) -> None:
        """Analyze how specialized each model's layers have become."""
        if not self.results.information_flow:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER, 'No activation data available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Specialization Analysis')
            ax.axis('off')
            return

        # Calculate specialization metrics for each model with consistent ordering
        specialization_data = []

        for model_name in self._sort_models_consistently(list(self.results.information_flow.keys())):
            ordered_layers = self._get_ordered_layer_analysis(model_name)

            if not ordered_layers:
                continue

            max_layers = getattr(self.config, 'max_layers_info_flow', 10)
            layer_specializations = []

            for i, (layer_name, analysis) in enumerate(ordered_layers[:max_layers]):
                # Specialization indicators:
                # 1. Low sparsity (neurons are active)
                # 2. Balanced positive ratio (not all saturated)
                # 3. Good effective rank (diverse representations)

                # DECISION plan-2026-10-05-analyzer-audit/F-052
                # Skip a layer with no measurement; do NOT substitute defaults. The old
                # `sparsity -> 1.0` default scored an absent layer's activation_health 0.0
                # and `effective_rank -> 1.0` gave it rank_score 0.1 — a fabricated row in a
                # ranking, not a missing one.
                sparsity = analysis.get('sparsity')
                positive_ratio = analysis.get('positive_ratio')
                effective_rank = analysis.get('effective_rank')

                if (sparsity is None or positive_ratio is None
                        or effective_rank is None):
                    logger.debug(
                        f"Model '{model_name}' layer '{layer_name}' has no "
                        f"specialization statistics ({analysis.get('error', 'keys absent')}); "
                        f"omitted from the specialization panel.")
                    continue

                # F-050: non-finite is not a measurement.
                if not all(np.isfinite(float(v))
                           for v in (sparsity, positive_ratio, effective_rank)):
                    logger.warning(
                        f"Model '{model_name}' layer '{layer_name}' produced non-finite "
                        f"specialization statistics; omitted from the panel.")
                    continue

                sparsity = float(sparsity)
                positive_ratio = float(positive_ratio)
                effective_rank = float(effective_rank)

                # Calculate specialization score (0-1, higher is better)
                activation_health = 1.0 - sparsity
                balance_score = 1.0 - abs(positive_ratio - 0.5) * BALANCE_SCORE_MULTIPLIER
                # DECISION plan-2026-10-05-analyzer-audit/F-054
                # `rank_score` divides by the MODULE constant
                # `LAYER_SPECIALIZATION_MAX_RANK = 10.0`, which is unrelated to the
                # capture batch. `effective_rank` is bounded above by that batch
                # (`results.information_flow_batch_size`), so any run whose memory budget
                # caps the batch below 10 — the DEFAULT 2048 MB does exactly that on a
                # conv net — saturated EVERY conv layer's rank_score at 1.0 and the panel
                # measured nothing. Normalise by the batch actually used, so the score
                # means "fraction of the available dimensionality retained".
                rank_scale = float(getattr(self.results, 'information_flow_batch_size', 0)
                                   or LAYER_SPECIALIZATION_MAX_RANK)
                if rank_scale <= 0:
                    rank_scale = LAYER_SPECIALIZATION_MAX_RANK
                rank_score = min(max(effective_rank, 0.0) / rank_scale, 1.0)

                # Combined specialization score
                layer_spec = (activation_health + balance_score + rank_score) / SPECIALIZATION_SCORE_COMPONENTS
                layer_specializations.append(layer_spec)

            if layer_specializations:
                specialization_data.append({
                    'Model': model_name,
                    'Layer Specializations': layer_specializations
                })

        if specialization_data:
            # Plot layer-by-layer specialization evolution
            for data in specialization_data:
                model_name = data['Model']
                layer_specs = data['Layer Specializations']
                color = self._get_model_color(model_name)

                x_positions = range(len(layer_specs))
                ax.plot(x_positions, layer_specs, 'o-',
                        color=color,
                        linewidth=LINE_WIDTH_STANDARD, markersize=MARKER_SIZE_SMALL)

            ax.set_title('Layer Specialization Evolution (Network Order)',
                        fontsize=SUBTITLE_FONT_SIZE, fontweight='bold')
            ax.set_xlabel('Layer Index (Network Depth)')
            ax.set_ylabel('Specialization Score')
            ax.grid(True, alpha=ALPHA_GRID)
            # DECISION plan-2026-10-05-analyzer-audit/F-056
            # Clamp to the components' actual range instead of pinning [0, 1].
            # `activation_health + balance_score + rank_score` divided by 3 can exceed
            # 1.0 — `balance_score = 1 - 2|posterior - 0.5|` is 1.0 exactly at
            # positive_ratio 0.5 and, unlike the other two, is not bounded above by
            # construction — so `set_ylim(0, 1)` silently cut the top of the score range
            # off the chart. Clamp only the LOWER bound at 0; let the data set the top.
            ax.set_ylim(bottom=AXIS_LIMIT_MIN)
        else:
            ax.text(ANNOTATION_X_CENTER, ANNOTATION_Y_CENTER,
                   'Insufficient data for specialization analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Layer Specialization Analysis')
            ax.axis('off')

# ---------------------------------------------------------------------