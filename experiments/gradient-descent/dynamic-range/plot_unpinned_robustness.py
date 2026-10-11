"""Robustness-to-noise figure for the unpinned, physical-moment sweeps.

Reproduces the style of the paper's original panel (80S solid blue circles,
TRPV1 dashed orange squares, categorical SNR axis with an infinity tick) from
the summary.csv files written by plot_unpinned_noise_sweep.m.

The plotted statistic is the MEAN over the 20 restarts, matching the original
figure, which plotted the mean over converged restarts. No restart in these
sweeps hit the iteration cap, so the two coincide. Pass --median to plot the
median instead; for TRPV1 at SNR 30 and 10 the mean is pulled up by a few
outlying restarts (median 0.025 / 0.032 vs mean 0.042 / 0.045).
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SWEEPS = os.path.join(ROOT, "results", "gradient-descent", "noise-sweep")
ORDER = ["Inf", "30", "10", "3", "1", "0.3", "0.1", "0.03"]


def load(folder, column):
    path = os.path.join(SWEEPS, folder, "summary.csv")
    with open(path) as handle:
        rows = {}
        for row in csv.DictReader(handle):
            snr = float(row["snr"])
            key = "Inf" if snr == float("inf") else f"{snr:g}"
            rows[key] = float(row[column])
    return [rows[k] for k in ORDER]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--median", action="store_true")
    parser.add_argument("--metric", choices=["l2", "all"], default="l2",
                        help="l2: degrees l>=2 only; all: every degree, the "
                             "quantity the O(3) alignment minimises")
    args = parser.parse_args()
    stat = "median" if args.median else "mean"
    column = f"{stat}_aligned_{args.metric}"

    s80 = load("80S_unpinned_noise sweep", column)
    trp = load("TRPV1_unpinned_noise sweep", column)

    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#4a4a4a",
                         "axes.labelcolor": "#222222",
                         "xtick.color": "#333333", "ytick.color": "#333333"})
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    x = range(len(ORDER))
    ax.plot(x, s80, "-o", color="#3b6ba5", lw=2.0, ms=6.5,
            mfc="white", mew=1.4, label="80S", zorder=3)
    ax.plot(x, trp, "--s", color="#c06a34", lw=2.0, ms=6.5,
            mfc="white", mew=1.4, label="TRPV1", zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(["∞"] + ORDER[1:])
    ax.set_xlabel("SNR")
    ax.set_ylabel("Relative coefficient error ($\\ell \\geq 2$)"
                  if args.metric == "l2" else "Relative coefficient error")
    ax.set_xlim(-0.35, len(ORDER) - 0.65)
    top = 1.08 * max(max(s80), max(trp))
    ax.set_ylim(0, top)
    # 0.3 spacing as in the original panel; finer when the range is small
    ax.yaxis.set_major_locator(MultipleLocator(0.3 if top > 0.9 else 0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, color="#e3e3e3", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handlelength=2.6)
    fig.tight_layout()

    suffix = ("_alldegrees" if args.metric == "all" else "") + \
             ("_median" if args.median else "")
    out = os.path.join(SWEEPS, f"noise_robustness_unpinned{suffix}")
    fig.savefig(out + ".png", dpi=250)
    fig.savefig(out + ".pdf")
    print("wrote", out + ".png", "and .pdf")
    for name, vals in (("80S", s80), ("TRPV1", trp)):
        print(f"  {name:5s}", " ".join(f"{v:.4f}" for v in vals))


if __name__ == "__main__":
    main()
