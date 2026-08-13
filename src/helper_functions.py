import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import anderson_ksamp, ecdf, ks_2samp, wasserstein_distance
from statsmodels.stats.proportion import proportions_ztest

matplotlib.use("Agg")  # headless-safe backend for saving ECDF plots


# for more details; see https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html
def plot_partial_effects(gam, feature_names, width):
    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4))
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), feature_names, strict=True)):
        XX = gam.generate_X_grid(term=i)
        pdep, ci = gam.partial_dependence(term=i, X=XX, width=width)
        ax.plot(XX[:, i], pdep, lw=2, color="C0")
        ax.fill_between(XX[:, i], ci[:, 0], ci[:, 1], alpha=0.25, color="C0")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(name)
    fig.tight_layout()
    return fig


def plot_residuals(gam, X, y, feature_names):
    resid = gam.deviance_residuals(X, y)
    pred = gam.predict(X)

    def binned(xv, rv, bins=15):
        e = np.linspace(xv.min(), xv.max(), bins + 1)
        c = (e[:-1] + e[1:]) / 2
        idx = np.digitize(xv, e[1:-1])
        m = [rv[idx == k].mean() if (idx == k).any() else np.nan for k in range(bins)]
        return c, m

    fig, ax = plt.subplots(1, len(feature_names) + 2, figsize=(4 * (len(feature_names) + 2), 3.2))
    ax[0].scatter(pred, resid, s=6)
    ax[0].axhline(0)
    ax[0].set_title("residuals vs fitted")
    for j, name in enumerate(feature_names):
        a = ax[j + 1]
        xv = X[:, j]
        a.scatter(xv, resid, s=6)
        a.axhline(0)
        cx, cy = binned(xv, resid)
        a.plot(cx, cy, color="k", lw=2)
        a.set_title(f"resid vs {name}")
    ax[-1].hist(resid)
    ax[-1].set_title("residual dist")
    fig.tight_layout()
    return fig


#### More helper functions for monitoring and drift detection;
#  z-test for continuous features, log-odds ratio for categorical features;
#  are used in the monitor.py script to detect drift between reference and current datasets


def ks_test(ref, cur, alpha):
    """Two-sample Kolmogorov-Smirnov test
    H0: the two samples are drawn from the same distribution
    H1: the two samples are drawn from different distributions"""
    stat, p = ks_2samp(cur, ref, alternative="two-sided")  # two-sided test is the default option
    return {
        "test": "ks_2samp",
        "shift": round(float(cur.mean() - ref.mean()), 2),
        "ks_stat": round(float(stat), 3),
        "p_value": round(float(p), 4),
        "significant": bool(p < alpha),
    }

def proptest(ref, cur, positive, alpha):
    """Two-proportion Z-test: did the positive-class rate move?"""
    c = [int((cur == positive).sum()), int((ref == positive).sum())]
    n = [len(cur), len(ref)]
    z, p = proportions_ztest(c, n)
    return {
        "test": "proportions_ztest",
        "ref_rate": round(c[1] / n[1], 3),
        "cur_rate": round(c[0] / n[0], 3),
        "rate_shift": round(c[0] / n[0] - c[1] / n[1], 3),
        "z": round(float(z), 3),
        "p_value": round(float(p), 4),
        "significant": bool(p < alpha),
    }


def ad_test(ref: np.ndarray, cur: np.ndarray, alpha: float) -> dict:
    """Two-sample Anderson-Darling test (AD).
 
    The AD is more sensitive than the KS to differences at the tails.
 
    Notes
    -----
    * The p-value is capped/floored by SciPy (~[0.001, 0.25]); this does NOT
      affect a decision at alpha=0.05, since 0.05 lies inside that range.
    * We read `.statistic` and `.pvalue` only, so this stays compatible with the
      SciPy 1.19 change that removes `critical_values` from the result.
    """
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence tie / p-value-floored warnings
        res = anderson_ksamp([ref, cur])
    p_value = getattr(res, "pvalue", None)
    if p_value is None:  # older SciPy
        p_value = res.significance_level
    return {
        "test": "anderson_darling",
        "statistic": float(res.statistic),
        "p_value": float(p_value),
        "shift": float(np.mean(cur) - np.mean(ref)),  # directional mean shift
        "significant": bool(p_value < alpha),
    }

 
def wasserstein_dist(ref: np.ndarray, cur: np.ndarray) -> float:
    """Raw (UNstandardized) 1-D Wasserstein-1 distance, in the column's own units.
 
    Deliberately NOT divided by the reference SD. Consequence: the value is
    comparable across *batches of the same column*, but NOT across columns    
    """
    return float(wasserstein_distance(np.asarray(ref, dtype=float),
                                      np.asarray(cur, dtype=float)))
 
 
def ecdf_plot(ref, cur, col, path):
    """Reference-vs-current ECDF overlay (the KS picture); returns KS D = max gap."""
    Fr, Fc = ecdf(ref).cdf, ecdf(cur).cdf
    grid = np.union1d(Fr.quantiles, Fc.quantiles)
    D = np.abs(Fr.evaluate(grid) - Fc.evaluate(grid)).max()
    ax = plt.subplots(figsize=(6, 4))[1]
    for F, lab in [(Fr, "reference"), (Fc, "current")]:
        ax.step(F.quantiles, F.probabilities, where="post", label=lab)
    ax.set(title=f"ECDF — {col}  (KS D={D:.3f})", xlabel=col, ylabel="F(x)")
    ax.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return float(D)
 
