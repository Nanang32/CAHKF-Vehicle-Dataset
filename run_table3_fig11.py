"""Runs the re-implemented Table 3 ablation and the Section 3.4 / Fig. 11 local-tuning experiments. Writes results/table3_trials.csv, table3_summary.csv, local_tuning_trials.csv, local_tuning_summary.csv."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'src')); os.chdir(os.path.join(HERE, 'results'))
import ablation_and_local_tuning as al
T3 = al.table3(); T3.to_csv('table3_trials.csv', index=False)
V = ['A_constant', 'B_empirical', 'C_calibrated']
def summ(g):
    a = g.true_D2.iloc[0]; r = {}
    for v in V: e = g[v+'_est']; r[v+'_bias'] = e.mean()-a; r[v+'_sd'] = e.std(ddof=1); r[v+'_rmse'] = np.sqrt(np.mean((e-a)**2))
    return pd.Series(r)
S = T3.groupby(['experiment', 'D0', 'maneuver'], sort=False).apply(summ, include_groups=False).reset_index(); S.to_csv('table3_summary.csv', index=False)
print(S.round(3).to_string(index=False))
rng_ = S[S.experiment == 'range']; sd = rng_[[v+'_sd' for v in V]]
print('\nrange: C has the lowest SD at %d of %d distances; mean SD A/B/C = %s; mean |bias| A/B/C = %s' % ((sd.idxmin(axis=1) == 'C_calibrated_sd').sum(), len(sd), sd.mean().round(3).tolist(), rng_[[v+'_bias' for v in V]].abs().mean().round(3).tolist()))
L = al.local_tuning(); L.to_csv('local_tuning_trials.csv', index=False)
LS = L.groupby(['design', 'true_D2', 'strategy'], sort=False).est_mean_over_event.agg(['mean', 'std']).reset_index(); LS.to_csv('local_tuning_summary.csv', index=False); print('\n', LS.round(3).to_string(index=False))
