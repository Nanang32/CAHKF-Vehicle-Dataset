"""Effective-range sweep with many repetitions and bootstrap confidence intervals (reviewer comment R1-3).
Same design as the n = 12 sweep behind Fig. 10(b) (R from the measured distance, D2 = -1.5 m/s^2, v0 = -0.5 m/s, adaptive duration) but n = 200 per distance
on a finer grid. Seeds: 20000 + 1000*grid_index + repetition. Writes results/range_sweep_largeN_trials.csv, range_sweep_largeN_summary.csv and figs/fig10b_largeN.png."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'src'))
import effective_range_analysis as er
F, W, SF, SW, SWP = 800.0, 1.8, 2.0, 0.05, 2.5
GRID = [8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100]; N = 200
def job(a):
    gi, D0, rep = a; dur = er.safe_trial_duration(D0, v0=-0.5, true_a=-1.5)
    e = er.run_single_trial(D0, -1.5, F, W, SF, SW, SWP, duration_s=dur, seed=20000 + 1000*gi + rep)
    return dict(D0=D0, rep=rep, seed=20000 + 1000*gi + rep, duration_s=dur, kf_estimate=e)
if __name__ == '__main__':
    os.chdir(os.path.join(HERE, 'results')); os.makedirs('figs', exist_ok=True)
    with Pool() as p: rows = p.map(job, [(gi, D0, r) for gi, D0 in enumerate(GRID) for r in range(N)], chunksize=20)
    df = pd.DataFrame(rows); df.to_csv('range_sweep_largeN_trials.csv', index=False)
    rng = np.random.default_rng(12345); out = []
    for D0, g in df.dropna().groupby('D0'):
        e = g.kf_estimate.values; rel = lambda x: 100*np.sqrt(np.mean((x + 1.5)**2))/1.5
        bs = [rel(rng.choice(e, len(e))) for _ in range(2000)]
        out.append(dict(D0=D0, n_valid=len(e), bias=e.mean()+1.5, sd=e.std(ddof=1), rel_rmse_pct=rel(e), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), duration_s=g.duration_s.iloc[0]))
    S = pd.DataFrame(out); S.to_csv('range_sweep_largeN_summary.csv', index=False); print(S.round(2).to_string(index=False))
    for thr in (25, 50, 100):
        below = S[S.rel_rmse_pct <= thr]; first_above = S[S.rel_rmse_pct > thr]
        print(f'relRMSE <= {thr}%: largest D0 with value <= {thr}% = {below.D0.max() if len(below) else None}; smallest D0 with CI-lower > {thr}% = {S[S.ci_lo > thr].D0.min() if (S.ci_lo > thr).any() else None}; largest D0 with CI-upper <= {thr}% = {S[S.ci_hi <= thr].D0.max() if (S.ci_hi <= thr).any() else None}')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200); ax.plot(S.D0, S.rel_rmse_pct, 'o-', color='#1f77b4'); ax.fill_between(S.D0, S.ci_lo, S.ci_hi, alpha=0.25, color='#1f77b4', label='95% bootstrap CI')
    ax.axhline(25, color='orange', ls='--', lw=0.8, label='25%'); ax.axvline(16.1, color='red', ls=':', lw=1, label='D_cross = 16.1 m'); ax.set_xlabel('Initial distance D0 (m)'); ax.set_ylabel('Relative RMSE (%)'); ax.set_ylim(0, 300); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig('figs/fig10b_largeN.png'); print('saved')
