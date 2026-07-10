
# src/helper_functions.py
"""Plotting + export helpers for training and monitoring."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.stats.proportion import proportions_ztest
from scipy.spatial.distance import jensenshannon

# for more details; see https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html
def plot_partial_effects(gam, feature_names, width):
    n = len(feature_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4))
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), feature_names)):
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

#### More helper functions for monitoring and drift detection; z-test for continuous features, log-odds ratio for categorical features; these are used in the monitor.py script to detect drift between reference and current datasets

 
def ks(ref, cur, alpha):
    """KS test: whole-distribution drift for a continuous feature (mean shift as effect size)."""
    stat, p = ks_2samp(cur, ref)
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
 
def js_distance(a, b, bins=20):
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), bins + 1)
    p = np.histogram(a, edges)[0] + 1e-9
    q = np.histogram(b, edges)[0] + 1e-9
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))