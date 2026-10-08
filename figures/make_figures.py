"""Generate research figures from real project data. One figures/ dir per repo."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 160,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
})
PALETTE = ['#1f6f9f', '#d96c2c', '#3a9e6e', '#8e5aa8', '#c0a02e', '#4aa3c7', '#e07b7b', '#6e7f80']

REPOS = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def scheduler_figs(d):
    df = pd.read_csv(d / 'data/summary.csv')
    # fig1: NRMSE vs headroom per policy (line + 95% CI)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, (pol, g) in enumerate(df.groupby('policy')):
        g = g.sort_values('headroom')
        ax.errorbar(g['headroom'], g['composite_nrmse_mean'],
                    yerr=[g['composite_nrmse_mean'] - g['composite_nrmse_ci95_low'],
                          g['composite_nrmse_ci95_high'] - g['composite_nrmse_mean']],
                    marker='o', ms=4, lw=1.5, capsize=3, color=PALETTE[i % len(PALETTE)], label=pol)
    ax.set_xlabel('Compute headroom')
    ax.set_ylabel('Composite NRMSE (mean ± 95% CI)')
    ax.set_title('Policy accuracy vs compute headroom')
    ax.legend(ncol=2, loc='upper right', framealpha=0.9)
    savefig(fig, d / 'figures/fig1_policy_nrmse_by_headroom.png')
    # fig2: cost vs accuracy at headroom 0.6
    g = df[df['headroom'] == 0.6].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sc = ax.scatter(g['elapsed_ms_mean'], g['composite_nrmse_mean'],
                    s=60 + 25 * g['switches_mean'], c=range(len(g)),
                    cmap='tab10', alpha=0.85, edgecolors='k', lw=0.5)
    for _, r in g.iterrows():
        ax.annotate(r['policy'], (r['elapsed_ms_mean'], r['composite_nrmse_mean']),
                    fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_xlabel('Elapsed time per decision (ms, log scale)')
    ax.set_ylabel('Composite NRMSE')
    ax.set_title('Cost–accuracy trade-off (headroom 0.6; bubble size = switches)')
    savefig(fig, d / 'figures/fig2_cost_vs_accuracy.png')



if __name__ == '__main__':
    d = REPOS
    (d / 'figures').mkdir(exist_ok=True)
    scheduler_figs(d)
    print('done')
