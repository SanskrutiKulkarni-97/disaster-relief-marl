"""Figures for the final report.
fig_comparison.pdf      - all algorithms, mean over 3 seeds (100-ep moving avg)
fig_learning_curves.pdf - MAPPO original vs corrected update (mean +/- std)"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e3e3e3"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "pdf.fonttype": 42,
})
_p = argparse.ArgumentParser()
_p.add_argument("--out-dir", default=".", help="where to write the figure files")
OUT = _p.parse_args().out_dir
os.makedirs(OUT, exist_ok=True)
W = 100
res = json.load(open("final_results.json"))

# fixed categorical order: colour follows the algorithm in every figure
SERIES = [
    ("runs_v2", "MAPPO", "#2a78d6"),
    ("cmp_mppo", "Masked PPO", "#eb6834"),
    ("cmp_ppo", "PPO", "#1baf7a"),
    ("cmp_dqn", "DQN", "#eda100"),
    ("cmp_sarsa", "SARSA", "#e87ba4"),
    ("cmp_qlearning", "Q-Learning", "#008300"),
]


def load(prefix, col):
    curves = []
    for path in sorted(glob.glob(f"{prefix}_s*/training_metrics.csv")):
        with open(path) as f:
            v = np.array([float(r[col]) for r in csv.DictReader(f)])
        curves.append(np.convolve(v, np.ones(W) / W, mode="valid"))
    return np.array(curves)


def refs(ax, key):
    for name, lab in [("random", "Random"), ("greedy_heuristic", "Greedy")]:
        y = res[name][key]
        ax.axhline(y, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
        ax.text(4000, y, f" {lab}", va="center", ha="left", fontsize=6.5, color=MUTED)


def style(ax, ylabel, title):
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left")
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 4000)


PANELS = [("reward", "Episode reward", "reward", "(a) Cumulative episode reward"),
          ("unmet_critical_demand", "Critical unmet (units)", "crit_unmet",
           "(b) Unmet critical demand at episode end")]

# ---------- comparison of all algorithms
fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)
for ax, (col, ylabel, key, title) in zip(axes, PANELS):
    for prefix, label, color in reversed(SERIES):  # MAPPO drawn on top
        c = load(prefix, col)
        x = np.arange(W, W + c.shape[1])
        ax.plot(x, c.mean(0), color=color, linewidth=1.2, label=label)
    refs(ax, key)
    style(ax, ylabel, title)
axes[1].set_xlabel("Training episode")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.47, 1.0), columnspacing=1.0, handlelength=1.6)
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.subplots_adjust(right=0.84)
fig.savefig(os.path.join(OUT, "fig_comparison.pdf")); fig.savefig(os.path.join(OUT, "fig_comparison.png"), dpi=200)
plt.close(fig)

# ---------- MAPPO original vs corrected update
fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True)
for ax, (col, ylabel, key, title) in zip(axes, PANELS):
    for prefix, label, color in [("runs_v1", "MAPPO (original update)", "#8a8a8a"),
                                 ("runs_v2", "MAPPO (corrected update)", "#2a78d6")]:
        c = load(prefix, col)
        x = np.arange(W, W + c.shape[1])
        m, s = c.mean(0), c.std(0)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, m, color=color, linewidth=1.3, label=label)
    refs(ax, key)
    style(ax, ylabel, title)
axes[1].set_xlabel("Training episode")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.45, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.subplots_adjust(right=0.84)
fig.savefig(os.path.join(OUT, "fig_learning_curves.pdf")); fig.savefig(os.path.join(OUT, "fig_learning_curves.png"), dpi=200)
print(f"saved figures to {OUT}")
