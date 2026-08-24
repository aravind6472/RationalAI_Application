"""End-to-end analysis pipeline.

Loads and cleans the data, fits both models, runs convergence diagnostics,
saves figures, and prints the answers to the two research questions:

1. Is there a relationship between economic connectedness (EC) and
   intergenerational mobility (IM), and in which direction?
2. How much IM variation is between-state vs within-state, and does the
   state structure change the EC-IM estimate?

Run from the repo root:

    python -m src.run_analysis --data-dir data --fig-dir figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import data as data_mod
from . import diagnostics as diag
from . import models
from . import plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_analysis")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    p.add_argument("--draws", type=int, default=3000)
    p.add_argument("--tune", type=int, default=3000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--cores", type=int, default=None,
                   help="Cores for parallel sampling; defaults to PyMC's choice.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = data_mod.load_clean(args.data_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Analysis frame: %d counties across %d states",
                len(df), df["state"].nunique())

    plots.scatter_ec_im(df, args.fig_dir / "ec_vs_im.png")
    plots.state_boxplot(df, args.fig_dir / "im_by_state.png")

    fit_kwargs = dict(draws=args.draws, tune=args.tune, chains=args.chains,
                      cores=args.cores)

    # --- Question 1: complete pooling ---
    cp_model, cp_idata = models.fit_complete_pooling(df, **fit_kwargs)
    cp_report = diag.convergence_report(cp_idata)
    n_bad = int((~cp_report["ok"]).sum())
    if n_bad:
        logger.warning("Complete-pooling: %d/%d parameters below convergence "
                       "thresholds (max R-hat %.4f, min ESS %d)", n_bad,
                       len(cp_report), cp_report["r_hat"].max(),
                       int(cp_report["ess_bulk"].min()))
    else:
        logger.info("Complete-pooling: all parameters converged")
    cp_lo, cp_hi = diag.credible_interval(cp_idata, "ec_county")
    plots.ppc(cp_model, cp_idata, args.fig_dir / "ppc_complete_pooling.png")
    plots.trace(cp_idata, ["Intercept", "ec_county", "sigma"],
                args.fig_dir / "trace_complete_pooling.png")

    # --- Question 2: hierarchical ---
    h_model, h_idata = models.fit_hierarchical(df, **fit_kwargs)
    h_report = diag.convergence_report(h_idata)
    n_bad = int((~h_report["ok"]).sum())
    if n_bad:
        logger.warning("Hierarchical: %d/%d parameters below convergence "
                       "thresholds (max R-hat %.4f, min ESS %d) - typically the "
                       "correlated state-level intercepts; increase --draws if "
                       "needed", n_bad, len(h_report), h_report["r_hat"].max(),
                       int(h_report["ess_bulk"].min()))
    else:
        logger.info("Hierarchical: all parameters converged")
    h_lo, h_hi = diag.credible_interval(h_idata, "ec_county")
    var = models.variance_decomposition(h_idata)
    plots.ppc(h_model, h_idata, args.fig_dir / "ppc_hierarchical.png")
    plots.trace(h_idata, ["Intercept", "ec_county", "sigma"],
                args.fig_dir / "trace_hierarchical.png")

    print("\n" + "=" * 60)
    print("RESEARCH QUESTION 1 - EC to IM relationship (complete pooling)")
    print("=" * 60)
    print(f"95% credible interval for EC slope: ({cp_lo:.4f}, {cp_hi:.4f})")
    print("Positive and excludes 0" if cp_lo > 0 else "Interval includes 0")

    print("\n" + "=" * 60)
    print("RESEARCH QUESTION 2 - variance decomposition (hierarchical)")
    print("=" * 60)
    print(f"Between-state variance share: {var['between_state']*100:.1f}%")
    print(f"Within-state variance share:  {var['within_state']*100:.1f}%")
    print(f"95% credible interval for EC slope: ({h_lo:.4f}, {h_hi:.4f})")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
