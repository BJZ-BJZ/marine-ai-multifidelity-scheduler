"""Innovation figure for this project: regenerated from the project's own data files.
Run: python make_innovation_figure.py  (needs matplotlib, numpy, pandas)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, json
from pathlib import Path
plt.rcParams.update({'font.size': 10})
R = Path(__file__).parent.parent

df = pd.read_csv(R / 'data/summary.csv')
d = df[df['headroom'] == 0.9].copy()
order = ['Fixed-L1', 'Fixed-L2', 'Threshold-Projected', 'Greedy-Online',
         'Switch-Aware-Greedy', 'Discounted-LinUCB', 'Contextual-Thompson',
         'Proposed-Exact-Online']
d['policy'] = pd.Categorical(d['policy'], order)
d = d.sort_values('policy')
x = np.arange(len(d)); w = 0.36
fig, ax = plt.subplots(figsize=(11, 5.2))
b1 = ax.bar(x - w/2, d['hydro_l2_share_mean'], w, yerr=[d['hydro_l2_share_mean']-d['hydro_l2_share_ci95_low'], d['hydro_l2_share_ci95_high']-d['hydro_l2_share_mean']],
            capsize=3, label='Hydrodynamics L2 share', color='#2e86c1')
b2 = ax.bar(x + w/2, d['propulsion_l2_share_mean'], w, yerr=[d['propulsion_l2_share_mean']-d['propulsion_l2_share_ci95_low'], d['propulsion_l2_share_ci95_high']-d['propulsion_l2_share_mean']],
            capsize=3, label='Propulsion L2 share', color='#e67e22')
ax.set_xticks(x); ax.set_xticklabels([p.replace('-', '-\n') for p in d['policy']], fontsize=8)
ax.set_ylabel('High-fidelity (L2) share')
ax.set_title('Adaptive fidelity allocation across modules by policy (headroom 0.90)\n'
             'The innovation: policies co-schedule L2 budget between hydrodynamics and propulsion modules',
             fontsize=11)
ax.legend(); ax.set_ylim(0, 1.12)
fig.tight_layout(); fig.savefig(Path(__file__).parent / 'fig7_fidelity_allocation.png', dpi=150)
plt.close(fig); print('saved fig7_fidelity_allocation.png')

