"""Redraws Fig. 4, Fig. 5 (panels B and C data of this study) and Fig. 9 from results/table1_fig9_*.csv. Run after reproduce.py."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results'); os.makedirs(os.path.join(R, 'figs'), exist_ok=True)
T = pd.read_csv(os.path.join(R, 'table1_fig9_trials.csv')); S = pd.read_csv(os.path.join(R, 'table1_fig9_summary.csv'))
order = list(S.condition); lab = {order[0]: 'Braking\n(true D̈=+2.0)', order[1]: 'Constant\n(true D̈=0.0)', order[2]: 'Acceleration\n(true D̈=−2.0)'}
tru = {order[0]: 2.0, order[1]: 0.0, order[2]: -2.0}
# ---- Fig. 4
fig, ax = plt.subplots(figsize=(11.2, 6.72), dpi=100); x = np.arange(3); w = 0.35
for i, c in enumerate(order):
    g = T[T.condition == c]
    ax.bar(i-w/2, g.kf_estimate.mean(), w, yerr=g.kf_estimate.std(ddof=1), color='#1f77b4', capsize=5, ecolor='k', label='Kalman (mean ± std)' if i == 0 else None)
    ax.bar(i+w/2, g.sg_estimate.mean(), w, yerr=g.sg_estimate.std(ddof=1), color='#d94a4a', capsize=5, ecolor='k', label='SG (mean ± std)' if i == 0 else None)
    ax.hlines(tru[c], i-0.4, i+0.4, colors='green', linestyles='--', label='True value' if i == 0 else None)
ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x); ax.set_xticklabels([lab[c] for c in order], fontsize=12); ax.set_ylabel('Estimated D̈ (m/s²)', fontsize=12)
ax.grid(axis='y', alpha=0.3); ax.legend(loc='upper left'); fig.tight_layout(); fig.savefig(os.path.join(R, 'figs', 'fig4_controlled_comparison.png')); plt.close(fig)
# ---- Fig. 9
fig, ax = plt.subplots(figsize=(6.28, 2.9), dpi=200)
for i, c in enumerate(order):
    g = S[S.condition == c].iloc[0]
    for dx, v, col, nm in ((-w/2, g.kf_recon, '#1f77b4', 'Kalman'), (w/2, g.sg_recon, '#d94a4a', 'SG')):
        ax.bar(i+dx, v, w, color=col, label=nm if i == 0 else None); ax.text(i+dx, v*1.08, f'{v:.3g}', ha='center', fontsize=6)
ax.set_yscale('log'); ax.set_ylim(5e-3, 30); ax.set_xticks(x); ax.set_xticklabels(['Braking', 'Constant', 'Acceleration'], fontsize=8)
ax.set_ylabel('Reconstruction RMSE (m, log scale)', fontsize=7); ax.tick_params(axis='y', labelsize=7); ax.grid(axis='y', which='both', alpha=0.25)
ax.legend(fontsize=7, loc='upper center', ncol=2); fig.tight_layout(); fig.savefig(os.path.join(R, 'figs', 'fig9_reconstruction_rmse.png')); plt.close(fig)
# ---- Fig. 5
kb = S.kf_bias.abs().mean(); sb = S.sg_bias.abs().mean(); sp = S.kf_speed_sd_kmh.mean()
fig, axs = plt.subplots(3, 1, figsize=(6.65, 10.5), dpi=100)
a = axs[0]; a.bar([0, 1, 2], [90.0, 95.1, 97.6], 0.55, color='#999999', edgecolor='k')
for i, v in enumerate([90.0, 95.1, 97.6]): a.text(i, v+2, f'{v:.1f}%', ha='center', fontweight='bold')
a.set_xticks([0, 1, 2]); a.set_xticklabels(['Jyothi et al. [25]\n(detection, min)', 'Lian et al. [26]\n(speed, min)', 'Lian et al. [26]\n(speed, average)'], fontsize=8)
a.set_ylabel('Accuracy (%)'); a.set_ylim(0, 125); a.set_yticks([0, 20, 40, 60, 80, 100]); a.set_title('A. Detection/Speed Accuracy (%)', fontweight='bold')
b = axs[1]; b.bar([0, 1, 2], [0.49, 1.77, sp], 0.55, color=['#f5a623', '#f5a623', '#1b2a6b'], edgecolor='k')
b.errorbar([0, 1], [0.49, 1.77], yerr=[[0.08, 1.28], [0.08, 1.28]], fmt='none', ecolor='k', capsize=5)
for i, (v, s) in enumerate([(0.57, '0.41–0.57 km/h'), (3.05, '0.49–3.05 km/h'), (sp, f'{sp:.2f} km/h')]): b.text(i, v+0.1, s, ha='center', fontweight='bold', fontsize=9)
b.set_xticks([0, 1, 2]); b.set_xticklabels(['Choi & Kong [18]\nreal vehicle (range)', 'Choi & Kong [18]\naccident cases (range)', 'This study\nKalman (mean std, sim.)'], fontsize=8)
b.set_ylabel('Speed estimation error (km/h)'); b.set_ylim(0, 3.9); b.set_title('B. Speed Error (km/h) — Real Data vs. Simulation', fontweight='bold')
c = axs[2]; c.bar([0, 1], [kb, sb], 0.55, color=['#1b2a6b', '#c41f5e'], edgecolor='k')
for i, v in enumerate([kb, sb]): c.text(i, v+0.4, f'{v:.2f} m/s²', ha='center', fontweight='bold', fontsize=9)
c.text(2, 1.0, 'No data available\n(research gap)', ha='center', fontsize=7, color='gray', style='italic')
c.set_xticks([0, 1, 2]); c.set_xlim(-0.6, 2.6); c.set_xticklabels(['Kalman filter\n(proposed)', 'Savitzky-Golay\n(baseline)', '6 comparator studies\n[16], [18], [21], [22], [25], [26]'], fontsize=8)
c.set_ylabel('Mean |acceleration bias| (m/s²)'); c.set_ylim(0, 22); c.set_title('C. Acceleration Bias (m/s²) — Gap Filled by This Study', fontweight='bold')
for a_ in axs: a_.spines['top'].set_visible(False); a_.spines['right'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(R, 'figs', 'fig5_comparators.png')); plt.close(fig)
print('mean |bias| KF %.3f SG %.3f; mean Kalman speed SD %.3f km/h' % (kb, sb, sp))
