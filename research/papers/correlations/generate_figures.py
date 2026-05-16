"""Generate figures for the correlations paper."""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# -------------------------------------------------------------------- 1. Anscombe
def fig_anscombe():
    # Classic Anscombe quartet data.
    x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
    y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
    x2 = x1.copy()
    y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
    x3 = x1.copy()
    y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float)
    y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])

    datasets = [(x1, y1, "I"), (x2, y2, "II"), (x3, y3, "III"), (x4, y4, "IV")]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5), sharex=True, sharey=True)
    for ax, (x, y, name) in zip(axes.flat, datasets):
        ax.scatter(x, y, s=30, color="#1f77b4", edgecolor="white", linewidth=0.5, zorder=3)
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(3, 20, 100)
        ax.plot(xs, slope * xs + intercept, color="#d62728", lw=1.5, zorder=2)
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"Dataset {name}  (r = {r:.3f})")
        ax.set_xlim(3, 20)
        ax.set_ylim(2, 14)
    fig.suptitle("Anscombe's Quartet: same r, very different data", y=1.02)
    save(fig, "anscombe.png")


# -------------------------------------------------------------------- 2. Pearson vs Rank
def fig_pearson_vs_rank():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)
    n = 200
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # (a) linear
    x = rng.uniform(-2, 2, n); y = 1.5 * x + rng.normal(0, 0.4, n)
    r = np.corrcoef(x, y)[0, 1]; rho, _ = spearmanr(x, y)
    axes[0].scatter(x, y, s=12, alpha=0.7, color="#1f77b4")
    axes[0].set_title(f"Linear\nPearson r = {r:.2f}, Spearman = {rho:.2f}")

    # (b) tanh(2x)
    x = rng.uniform(-2, 2, n); y = np.tanh(2 * x) + rng.normal(0, 0.08, n)
    r = np.corrcoef(x, y)[0, 1]; rho, _ = spearmanr(x, y)
    axes[1].scatter(x, y, s=12, alpha=0.7, color="#2ca02c")
    axes[1].set_title(f"Monotonic nonlinear (tanh)\nPearson r = {r:.2f}, Spearman = {rho:.2f}")

    # (c) cubic
    x = rng.uniform(-2, 2, n); y = x ** 3 + rng.normal(0, 0.3, n)
    r = np.corrcoef(x, y)[0, 1]; rho, _ = spearmanr(x, y)
    axes[2].scatter(x, y, s=12, alpha=0.7, color="#d62728")
    axes[2].set_title(f"Cubic\nPearson r = {r:.2f}, Spearman = {rho:.2f}")

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    save(fig, "pearson_vs_rank.png")


# -------------------------------------------------------------------- 3. Correlation grid
def fig_correlation_grid():
    rng = np.random.default_rng(1)
    n = 300

    def linear(r):
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        if abs(r) < 1e-6:
            return x, z
        y = r * x + np.sqrt(max(1 - r * r, 0)) * z
        return x, y

    panels = []
    for r in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        if abs(r) >= 0.9999:
            x = rng.uniform(-2, 2, n)
            y = np.sign(r) * x
        else:
            x, y = linear(r)
        panels.append((x, y, f"r ≈ {r:+.1f}"))

    # quadratic
    x = rng.uniform(-2, 2, n); y = x ** 2 + rng.normal(0, 0.1, n)
    panels.append((x, y, "quadratic"))
    # sine
    x = rng.uniform(-np.pi, np.pi, n); y = np.sin(x) + rng.normal(0, 0.1, n)
    panels.append((x, y, "sine"))
    # circle
    t = rng.uniform(0, 2 * np.pi, n); x = np.cos(t) + rng.normal(0, 0.05, n); y = np.sin(t) + rng.normal(0, 0.05, n)
    panels.append((x, y, "circle"))
    # X-shape
    s = rng.choice([-1, 1], size=n); x = rng.uniform(-1, 1, n); y = s * x + rng.normal(0, 0.05, n)
    panels.append((x, y, "X-shape"))

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.5))
    for ax, (x, y, label) in zip(axes.flat, panels):
        ax.scatter(x, y, s=6, alpha=0.6, color="#1f77b4")
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{label}\nPearson r = {r:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    # last panel may be unused; we used 9 - good.
    fig.suptitle("Pearson r = 0 does not imply independence", y=1.01)
    save(fig, "correlation_grid.png")


# -------------------------------------------------------------------- 4. Distance correlation
def _dcor(x, y):
    """U-statistic distance correlation estimator (biased version, simple)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    A = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0)[None, :] - b.mean(1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvarx = (A * A).mean()
    dvary = (B * B).mean()
    denom = np.sqrt(dvarx * dvary)
    return float(np.sqrt(max(dcov2, 0) / denom)) if denom > 0 else 0.0


def fig_distance_correlation():
    rng = np.random.default_rng(2)
    n = 200
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # linear
    x = rng.uniform(-2, 2, n); y = 1.2 * x + rng.normal(0, 0.3, n)
    r = np.corrcoef(x, y)[0, 1]; d = _dcor(x, y)
    axes[0].scatter(x, y, s=12, alpha=0.7, color="#1f77b4")
    axes[0].set_title(f"Linear\nPearson = {r:.2f}, dCor = {d:.2f}")

    # quadratic
    x = rng.uniform(-2, 2, n); y = x ** 2 + rng.normal(0, 0.2, n)
    r = np.corrcoef(x, y)[0, 1]; d = _dcor(x, y)
    axes[1].scatter(x, y, s=12, alpha=0.7, color="#2ca02c")
    axes[1].set_title(f"Quadratic\nPearson = {r:.2f}, dCor = {d:.2f}")

    # sine
    x = rng.uniform(-np.pi, np.pi, n); y = np.sin(x) + rng.normal(0, 0.15, n)
    r = np.corrcoef(x, y)[0, 1]; d = _dcor(x, y)
    axes[2].scatter(x, y, s=12, alpha=0.7, color="#d62728")
    axes[2].set_title(f"Sine\nPearson = {r:.2f}, dCor = {d:.2f}")

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    save(fig, "distance_correlation.png")


# -------------------------------------------------------------------- 5. Chatterjee xi
def _chatterjee_xi(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    y_sorted = y[order]
    # rank y, breaking ties by averaging (use scipy rankdata)
    from scipy.stats import rankdata
    r = rankdata(y_sorted, method="average")
    num = np.sum(np.abs(np.diff(r)))
    return 1.0 - (3.0 * num) / (n * n - 1.0)


def fig_chatterjee_xi():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    # noisy linear
    n = 400
    x = rng.uniform(0, 1, n); y = 0.8 * x + rng.normal(0, 0.2, n)
    xi = _chatterjee_xi(x, y); r = np.corrcoef(x, y)[0, 1]; rho, _ = spearmanr(x, y)
    axes[0].scatter(x, y, s=10, alpha=0.6, color="#1f77b4")
    axes[0].set_title(f"Noisy linear\nPearson = {r:.2f}, Spearman = {rho:.2f}, ξ = {xi:.2f}")

    # sin(4 pi x)
    x = rng.uniform(0, 1, n); y = np.sin(4 * np.pi * x) + rng.normal(0, 0.05, n)
    xi = _chatterjee_xi(x, y); r = np.corrcoef(x, y)[0, 1]; rho, _ = spearmanr(x, y)
    axes[1].scatter(x, y, s=10, alpha=0.6, color="#d62728")
    axes[1].set_title(f"y = sin(4πx)\nPearson = {r:.2f}, Spearman = {rho:.2f}, ξ = {xi:.2f}")

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    save(fig, "chatterjee_xi.png")


# -------------------------------------------------------------------- 6. MI vs Pearson
def fig_mi_vs_pearson():
    from sklearn.feature_selection import mutual_info_regression
    rng = np.random.default_rng(4)
    n = 500
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    # linear
    x = rng.uniform(-2, 2, n); y = 1.2 * x + rng.normal(0, 0.3, n)
    r = np.corrcoef(x, y)[0, 1]
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0]
    axes[0].scatter(x, y, s=10, alpha=0.6, color="#1f77b4")
    axes[0].set_title(f"Linear\nPearson = {r:.2f}, MI = {mi:.2f}")

    # quadratic
    x = rng.uniform(-2, 2, n); y = x ** 2 + rng.normal(0, 0.2, n)
    r = np.corrcoef(x, y)[0, 1]
    mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0]
    axes[1].scatter(x, y, s=10, alpha=0.6, color="#2ca02c")
    axes[1].set_title(f"Quadratic\nPearson = {r:.2f}, MI = {mi:.2f}")

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    save(fig, "mi_vs_pearson.png")


# -------------------------------------------------------------------- 7. SHAP summary
def fig_shap_summary():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.datasets import make_friedman1
    rng = np.random.default_rng(5)
    X, y = make_friedman1(n_samples=800, n_features=8, noise=1.0, random_state=5)
    feat_names = [f"x{i+1}" for i in range(X.shape[1])]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=5)
    model.fit(X, y)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # beeswarm-style: for each feature, plot SHAP values colored by feature value
        order = np.argsort(np.mean(np.abs(sv), axis=0))
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, j in enumerate(order):
            vals = sv[:, j]
            fv = X[:, j]
            # normalize fv for color
            fv_n = (fv - fv.min()) / (fv.max() - fv.min() + 1e-12)
            jitter = rng.uniform(-0.18, 0.18, size=vals.shape)
            sc = ax.scatter(vals, np.full_like(vals, i) + jitter,
                            c=fv_n, cmap="coolwarm", s=8, alpha=0.7, edgecolor="none")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feat_names[j] for j in order])
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.set_title("SHAP beeswarm (TreeExplainer, GradientBoosting on Friedman1)")
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("feature value (normalised)")
        cbar.set_ticks([])
        save(fig, "shap_summary.png")
        return
    except Exception as e:
        print(f"shap path failed: {e}; falling back to permutation importance")

    from sklearn.inspection import permutation_importance
    pi = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    order = np.argsort(pi.importances_mean)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(np.arange(len(order)), pi.importances_mean[order],
            xerr=pi.importances_std[order], color="#1f77b4")
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([feat_names[j] for j in order])
    ax.set_xlabel("Permutation importance")
    ax.set_title("Permutation importance (SHAP fallback)")
    save(fig, "shap_summary.png")


# -------------------------------------------------------------------- 8. PDP + ICE
def fig_pdp_ice():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.datasets import make_friedman1
    from sklearn.inspection import PartialDependenceDisplay
    X, y = make_friedman1(n_samples=800, n_features=8, noise=1.0, random_state=7)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=7)
    model.fit(X, y)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    PartialDependenceDisplay.from_estimator(
        model, X, features=[3], kind="both",
        ice_lines_kw={"color": "lightsteelblue", "alpha": 0.4, "lw": 0.6},
        pd_line_kw={"color": "#d62728", "lw": 2.0},
        ax=ax, random_state=0,
    )
    ax.set_title("Partial Dependence + ICE for feature x4 (Friedman1)")
    ax.set_xlabel("x4"); ax.set_ylabel("partial dependence")
    save(fig, "pdp_ice.png")


def main():
    fig_anscombe()
    fig_pearson_vs_rank()
    fig_correlation_grid()
    fig_distance_correlation()
    fig_chatterjee_xi()
    fig_mi_vs_pearson()
    fig_shap_summary()
    fig_pdp_ice()


if __name__ == "__main__":
    main()
