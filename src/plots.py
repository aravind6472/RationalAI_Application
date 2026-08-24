"""Plotting helpers. Each function saves a figure and returns its path."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def scatter_ec_im(df: pd.DataFrame, out: Path) -> Path:
    """Scatter of EC vs IM with an OLS trend line."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["ec_county"], df["kfr_pooled_pooled_p25"], alpha=0.3, s=12)
    coef = np.polyfit(df["ec_county"], df["kfr_pooled_pooled_p25"], 1)
    xs = np.linspace(df["ec_county"].min(), df["ec_county"].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), color="C1", linewidth=2)
    ax.set_xlabel("Economic Connectedness")
    ax.set_ylabel("Intergenerational Mobility (p25)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def state_boxplot(df: pd.DataFrame, out: Path) -> Path:
    """Boxplot of IM by state, ordered by median, to motivate grouping."""
    order = (df.groupby("state")["kfr_pooled_pooled_p25"].median()
             .sort_values().index)
    data = [df.loc[df["state"] == s, "kfr_pooled_pooled_p25"].values
            for s in order]
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.boxplot(data, vert=False, labels=order, widths=0.6)
    ax.set_xlabel("Intergenerational Mobility (p25)")
    ax.set_ylabel("State")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def ppc(model, idata, out: Path, n_draws: int = 100) -> Path:
    """Posterior predictive check overlay.

    Uses ``az.plot_ppc`` when available (ArviZ < 1.0); otherwise draws the
    overlay directly with matplotlib so the repo runs on ArviZ 1.0+ without the
    separate arviz-plots package.
    """
    model.predict(idata, kind="response", inplace=True)

    if hasattr(az, "plot_ppc"):
        az.plot_ppc(idata, num_pp_samples=n_draws)
        fig = plt.gcf()
    else:
        fig = _ppc_fallback(idata, n_draws)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _to_dataset(group):
    """Return an xarray Dataset from an ArviZ group (Dataset or 1.0 DataTree)."""
    return group.to_dataset() if hasattr(group, "to_dataset") else group


def _stack_vars(group) -> np.ndarray:
    """Stack all data variables of an ArviZ group into a 2D array (samples x obs)."""
    ds = _to_dataset(group)
    arrays = [np.asarray(ds[v].values).ravel() for v in ds.data_vars]
    return np.stack(arrays) if len(arrays) > 1 else arrays[0]


def _ppc_fallback(idata, n_draws: int):
    """Manual PPC overlay: observed KDE vs. a sample of posterior-predictive KDEs."""
    obs = _stack_vars(idata.observed_data).ravel()
    pp = np.asarray(
        list(_to_dataset(idata.posterior_predictive).data_vars.values())[0].values
    ).reshape(-1, obs.size)

    fig, ax = plt.subplots(figsize=(7, 5))
    grid = np.linspace(obs.min(), obs.max(), 200)
    idx = np.random.default_rng(0).choice(pp.shape[0],
                                          size=min(n_draws, pp.shape[0]),
                                          replace=False)
    for i in idx:
        ax.plot(grid, _kde(pp[i], grid), color="C0", alpha=0.15, linewidth=0.6)
    ax.plot(grid, _kde(obs, grid), color="black", linewidth=2, label="Observed")
    ax.set_xlabel("Intergenerational Mobility (p25)")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def _kde(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Lightweight Gaussian KDE to avoid a scipy dependency."""
    samples = samples[np.isfinite(samples)]
    bw = 1.06 * samples.std() * samples.size ** (-1 / 5)  # Silverman's rule
    bw = max(bw, 1e-6)
    u = (grid[:, None] - samples[None, :]) / bw
    return np.exp(-0.5 * u**2).sum(axis=1) / (samples.size * bw * np.sqrt(2 * np.pi))


def trace(idata, var_names: list[str], out: Path) -> Path:
    """Trace + density plot for the named parameters.

    Uses ``az.plot_trace`` when available; otherwise renders per-parameter
    trace and density panels directly with matplotlib.
    """
    if hasattr(az, "plot_trace"):
        az.plot_trace(idata, var_names=var_names)
        fig = plt.gcf()
    else:
        fig = _trace_fallback(idata, var_names)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _trace_fallback(idata, var_names: list[str]):
    """Manual trace + density panels, one row per parameter."""
    post = idata.posterior
    fig, axes = plt.subplots(len(var_names), 2,
                             figsize=(10, 2.5 * len(var_names)),
                             squeeze=False)
    for row, name in enumerate(var_names):
        da = post[name]
        chains = da.values.reshape(da.sizes["chain"], -1)
        flat = chains.ravel()
        grid = np.linspace(flat.min(), flat.max(), 200)
        for ch in chains:
            axes[row, 0].plot(grid, _kde(ch, grid), alpha=0.8, linewidth=0.8)
            axes[row, 1].plot(ch, alpha=0.6, linewidth=0.5)
        axes[row, 0].set_title(f"{name} density")
        axes[row, 1].set_title(f"{name} trace")
    return fig
