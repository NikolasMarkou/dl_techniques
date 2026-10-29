"""
Plots for the diagonal model Fisher analysis.

Every figure names the ESTIMATOR it shows, because the paper's two estimators are easy to
confuse and are not interchangeable: the sequence Fisher (mean of squared sequence scores,
``fisher_value``) and the mean token Fisher (average over sampled prefixes of the token
Fisher, ``token_fisher_mean``). They agree in expectation (Prop. 2) but not per sample, and
the second is not the unnormalized sequence Fisher. Every title also carries the sample
sizes N_F / N_g / G of the pass.

An arm that did not run produces no figure and no placeholder frame: each ``plot_*`` method
returns ``None`` for absent data, and ``create_visualizations`` only saves what it got.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .base import BaseVisualizer
from ..constants import FisherMetricNames as FM
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

LABEL_SEQUENCE = "sequence Fisher (mean of squared sequence scores)"
LABEL_TOKEN = "mean token Fisher (average over sampled prefixes, not the sequence Fisher)"

TITLE_FONT_SIZE = 12
LABEL_MAX_CHARS = 38
TOP_K_COORDINATES = 20
HISTOGRAM_BINS = 80
# A scatter of every parameter is unreadable and slow; the envelope check uses all of them.
MAX_SCATTER_POINTS = 20_000
# Variable groups drawn individually in the concentration bar; the rest are pooled.
MAX_BAR_GROUPS = 12
# A bound-violation fraction above this is stamped on the plot as a failed bound.
VIOLATION_ALERT = 0.01
PANEL_WIDTH = 5.0
PANEL_HEIGHT = 4.0

# ---------------------------------------------------------------------


def _truncate(text: str, limit: int = LABEL_MAX_CHARS) -> str:
    """Keep the tail of a long variable path, which is the part that differs."""
    return text if len(text) <= limit else "..." + text[-(limit - 3):]


class FisherVisualizer(BaseVisualizer):
    """Plots for one or more models' Fisher outputs."""

    def _get_models_with_data(self) -> List[str]:
        """Models whose Fisher pass succeeded, in ``model_order``."""
        have = set(self.results.fisher_outputs)
        ordered = [m for m in self.model_order if m in have]
        return ordered + sorted(have - set(ordered))

    # ------------------------------------------------------------------

    def _header(self, estimator: str) -> str:
        """Estimator label plus the N_F / N_g / G of the pass."""
        summary = self.results.fisher_summary
        sizes = {o.n_samples for o in self.results.fisher_outputs.values()}
        n_f = f"{min(sizes)}" if len(sizes) == 1 else f"{min(sizes)}-{max(sizes)}"
        n_g = summary.get('n_gradient_prompts', 0)
        return f"{estimator}\nN_F={n_f}  N_g={n_g}  G={summary.get('n_responses', 1)}"

    def create_visualizations(self) -> None:
        """Create every Fisher figure whose data exists."""
        if not self.results.fisher_outputs:
            logger.info("No Fisher results to visualize")
            return
        plots = (
            ('fisher_distribution', self.plot_fisher_distribution),
            ('fisher_top_coordinates', self.plot_top_coordinates),
            ('fisher_vs_gradient', self.plot_fisher_vs_gradient),
            ('fisher_gradient_recall', self.plot_gradient_recall),
            ('fisher_token_level', self.plot_token_level),
            ('fisher_persistence', self.plot_persistence),
            ('fisher_variable_summary', self.plot_variable_summary),
        )
        for name, plot in plots:
            try:
                fig = plot()
            except Exception as e:
                logger.error(f"Could not create {name}: {e}", exc_info=True)
                continue
            if fig is not None:
                self._save_figure(fig, name)
                plt.close(fig)

    # ------------------------------------------------------------------

    def plot_fisher_distribution(self) -> Optional[plt.Figure]:
        """Log-scale histogram of F-hat per model, marking the Fisher-dead threshold."""
        models = self._get_models_with_data()
        if not models:
            return None
        fig, ax = plt.subplots(figsize=self.config.get_figure_size(0.7))
        threshold = float(self.config.fisher_zero_threshold)
        for name in models:
            fisher = self.results.fisher_outputs[name].fisher_flat
            alive = fisher[fisher > threshold]
            dead_fraction = 1.0 - len(alive) / max(len(fisher), 1)
            if len(alive):
                ax.hist(np.log10(alive), bins=HISTOGRAM_BINS, histtype='step', density=True,
                        color=self._get_model_color(name), linewidth=2,
                        label=f"{name} (dead {100 * dead_fraction:.1f}%)")
        ax.axvline(np.log10(threshold), color='gray', linestyle='--',
                   label=f"dead threshold {threshold:g}")
        ax.set_xlabel("log10 F-hat")
        ax.set_ylabel("density over Fisher-alive coordinates")
        ax.set_title(self._header(LABEL_SEQUENCE), fontsize=TITLE_FONT_SIZE)
        ax.legend()
        return fig

    def plot_top_coordinates(self, k: int = TOP_K_COORDINATES) -> Optional[plt.Figure]:
        """Lollipop chart of the k largest-Fisher coordinates per model."""
        models = self._get_models_with_data()
        if not models:
            return None
        fig, axes = plt.subplots(len(models), 1, squeeze=False,
                                 figsize=(PANEL_WIDTH * 1.6, PANEL_HEIGHT * 1.3 * len(models)))
        for ax, name in zip(axes[:, 0], models):
            out = self.results.fisher_outputs[name]
            top = np.argsort(-out.fisher_flat, kind='stable')[:k]
            labels = []
            for idx in top:
                path = next(p for p, s, e in out.param_slices if s <= idx < e)
                start = next(s for p, s, e in out.param_slices if p == path)
                labels.append(f"{_truncate(path)}[{idx - start}]")
            y = np.arange(len(top))
            ax.hlines(y, 0, out.fisher_flat[top], color=self._get_model_color(name))
            ax.plot(out.fisher_flat[top], y, 'o', color=self._get_model_color(name))
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_xscale('log')
            ax.set_xlabel("F-hat")
            ax.set_title(f"{name}: top {len(top)} coordinates", fontsize=TITLE_FONT_SIZE - 2)
        fig.suptitle(self._header(LABEL_SEQUENCE) + "\nlabels: variable path (tail kept) [index in variable]",
                     fontsize=TITLE_FONT_SIZE)
        fig.tight_layout()
        return fig

    def plot_fisher_vs_gradient(self) -> Optional[plt.Figure]:
        """|g-hat| against sqrt(F-hat) with the Cauchy-Schwarz envelope signal_scale * sqrt(F)."""
        models = [m for m in self._get_models_with_data()
                  if self.results.fisher_outputs[m].gradient_flat is not None]
        if not models:
            return None
        fig, axes = plt.subplots(1, len(models), squeeze=False,
                                 figsize=(PANEL_WIDTH * len(models) + 1, PANEL_HEIGHT + 1))
        rng = np.random.default_rng(0)
        for ax, name in zip(axes[0], models):
            out = self.results.fisher_outputs[name]
            alive = out.fisher_flat > self.config.fisher_zero_threshold
            x = np.sqrt(out.fisher_flat[alive])
            y = np.abs(out.gradient_flat[alive])
            if len(x) > MAX_SCATTER_POINTS:
                keep = rng.choice(len(x), MAX_SCATTER_POINTS, replace=False)
                x, y = x[keep], y[keep]
            ax.scatter(x, y, s=4, alpha=0.3, color=self._get_model_color(name))
            if len(x) and out.signal_scale is not None:
                line = np.linspace(0, float(x.max()), 50)
                ax.plot(line, out.signal_scale * line, 'k--',
                        label=f"envelope = signal_scale x sqrt(F)  ({out.signal_scale:.3g})")
            rate = out.violation_rate
            if rate is None or not np.isfinite(rate):
                status = "bound check NOT COMPUTED"
            elif rate > VIOLATION_ALERT:
                status = f"BOUND VIOLATED on {100 * rate:.1f}% of coordinates"
            else:
                status = f"violations: {100 * rate:.2f}% of coordinates"
            ax.set_xlabel("sqrt(F-hat)")
            ax.set_ylabel("|g-hat|")
            ax.set_title(f"{name}\n{status}", fontsize=TITLE_FONT_SIZE - 2)
            ax.legend(fontsize=7)
        fig.suptitle(self._header(LABEL_SEQUENCE), fontsize=TITLE_FONT_SIZE)
        fig.tight_layout()
        return fig

    def plot_gradient_recall(self) -> Optional[plt.Figure]:
        """Recall(q) against q, always with the random-subset baseline y = q."""
        data = {m: r for m, r in self.results.fisher_gradient_recall.items() if r}
        if not data:
            return None
        fig, ax = plt.subplots(figsize=self.config.get_figure_size(0.7))
        all_q = sorted({q for r in data.values() for q in r})
        ax.plot(all_q, all_q, 'k--', label="random subset (y = q)")
        for name, recall in data.items():
            qs = sorted(recall)
            ax.plot(qs, [recall[q] for q in qs], 'o-', color=self._get_model_color(name), label=name)
        ax.set_xscale('log')
        # symlog, not log: a recall of exactly 0 is a real value and a log axis would
        # drop it off the bottom of the figure. NaN (an empty K_q) is not plotted.
        ax.set_yscale('symlog', linthresh=max(min(all_q) / 2, 1e-3))
        ax.set_ylim(bottom=0)
        ax.set_xlabel("selection percentage q (%)")
        ax.set_ylabel("Recall(q) (%)")
        ax.set_title(self._header(LABEL_SEQUENCE) + "\ngradient-support recall (Eq. 7)",
                     fontsize=TITLE_FONT_SIZE)
        ax.legend()
        return fig

    def plot_token_level(self) -> Optional[plt.Figure]:
        """Positions x parameter-groups heatmaps, Real beside Uniform, with the ratio per group."""
        models = list(self.results.fisher_token_level)
        if not models:
            return None
        fig, axes = plt.subplots(len(models), 3, squeeze=False,
                                 figsize=(PANEL_WIDTH * 3, PANEL_HEIGHT * len(models) + 1))
        for row, name in zip(axes, models):
            token = self.results.fisher_token_level[name]
            groups = token['group_names']
            for ax, key, title in ((row[0], 'token_fisher_by_position', "Real (p)"),
                                   (row[1], 'uniform_by_position', "Uniform reference (Eq. 11)")):
                image = ax.imshow(np.log10(np.maximum(token[key], 1e-30)), aspect='auto', cmap='viridis')
                ax.set_title(f"{name}: {title}", fontsize=TITLE_FONT_SIZE - 3)
                ax.set_xlabel("parameter group")
                ax.set_ylabel("sampled position slot")
                ax.set_xticks(range(len(groups)))
                ax.set_xticklabels([_truncate(g, 14) for g in groups], rotation=90, fontsize=6)
                fig.colorbar(image, ax=ax, label="log10 token Fisher")
            ratio = token['real_uniform_ratio']
            row[2].bar(range(len(groups)), [ratio[g] for g in groups], color=self._get_model_color(name))
            row[2].axhline(1.0, color='k', linestyle='--')
            row[2].set_xticks(range(len(groups)))
            row[2].set_xticklabels([_truncate(g, 14) for g in groups], rotation=90, fontsize=6)
            row[2].set_ylabel("Real / Uniform")
            row[2].set_title("ratio < 1: likely tokens agree", fontsize=TITLE_FONT_SIZE - 3)
        fig.suptitle(self._header(LABEL_TOKEN), fontsize=TITLE_FONT_SIZE)
        fig.tight_layout()
        return fig

    def plot_persistence(self) -> Optional[plt.Figure]:
        """Retention of the high / low Fisher sets and the low-set magnitude ratio, with baselines."""
        table = self.results.fisher_persistence
        if table is None or len(table) == 0:
            return None
        fig, axes = plt.subplots(1, 3, figsize=(PANEL_WIDTH * 3, PANEL_HEIGHT + 1))
        checkpoints = list(dict.fromkeys(table['checkpoint']))
        for i, name in enumerate(checkpoints):
            sub = table[table['checkpoint'] == name].sort_values('q')
            for ax, column in zip(axes, ('retention_high', 'retention_low', 'mass_ratio')):
                ax.plot(sub['q'], sub[column], 'o-', label=name, color=f"C{i}")
        reference = table[table['checkpoint'] == checkpoints[0]].sort_values('q')
        axes[0].plot(reference['q'], reference['random_high'], 'k--', label="random (q)")
        axes[1].plot(reference['q'], reference['random_low'], 'k--', label="random (100 - q)")
        axes[2].axhline(1.0, color='k', linestyle='--', label="unchanged")
        for ax, title in zip(axes, ("Retention, high set (%)", "Retention, low set (%)",
                                    "M_k / M_0 on the initial low set")):
            ax.set_xscale('log')
            ax.set_xlabel("selection percentage q (%)")
            ax.set_title(title, fontsize=TITLE_FONT_SIZE - 2)
            ax.legend(fontsize=7)
        fig.suptitle(self._header(LABEL_SEQUENCE) + "\npersistence relative to the initial policy",
                     fontsize=TITLE_FONT_SIZE)
        fig.tight_layout()
        return fig

    def plot_variable_summary(self) -> Optional[plt.Figure]:
        """Per-model horizontal bar: each layer's share of the total Fisher."""
        models = self._get_models_with_data()
        if not models:
            return None
        fig, ax = plt.subplots(figsize=(PANEL_WIDTH * 2, 1.2 * len(models) + 2.5))
        cmap = plt.get_cmap('tab20')
        for row, name in enumerate(models):
            out = self.results.fisher_outputs[name]
            totals: Dict[str, float] = {}
            for path, start, stop in out.param_slices:
                layer = path.rsplit('/', 1)[0]
                totals[layer] = totals.get(layer, 0.0) + float(out.fisher_flat[start:stop].sum())
            ranked = sorted(totals.items(), key=lambda kv: -kv[1])
            grand = sum(totals.values()) or 1.0
            shown, rest = ranked[:MAX_BAR_GROUPS], ranked[MAX_BAR_GROUPS:]
            left = 0.0
            for i, (layer, value) in enumerate(shown):
                ax.barh(row, value / grand, left=left, color=cmap(i % 20), edgecolor='white',
                        label=_truncate(layer, 24) if row == 0 else None)
                left += value / grand
            if rest:
                ax.barh(row, sum(v for _, v in rest) / grand, left=left, color='lightgray',
                        label="other layers" if row == 0 else None)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of total Fisher")
        ax.set_title(self._header(LABEL_SEQUENCE) + "\nlayer-level concentration", fontsize=TITLE_FONT_SIZE)
        ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc='upper left')
        return fig
