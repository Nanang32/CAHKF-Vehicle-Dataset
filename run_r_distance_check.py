"""Sensitivity of the A/C comparison to WHICH distance enters R(t) (reviewer comment R1-6).
Variants: A constant R; C_measured (as used in all other scripts: D_k = fW/w_k from the noisy width); C_smoothedD (two-pass: R from the RTS-smoothed distance of a
constant-R pass); C_trueD (oracle, simulated reference distance). Same noise draws for all variants (paired). Writes results/r_distance_source_trials.csv / _summary.csv."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'src')); os.chdir(os.path.join(HERE, 'results'))
import effective_range_analysis as er
F, W, SF, SW, SWP, FPS = 800., 1.8, 2., .05, 2.5, 15; QG = np.logspace(-3, 3, 20)
def run(D0, rep):
    dur = er.safe_trial_duration(D0, -0.5, -1.5); n = int(FPS*dur); t = np.arange(n)/FPS; Dt = D0 - 0.5*t + 0.5*(-1.5)*t**2
    seed = 9000 + int(10*D0) + 31*rep; rng = np.random.default_rng(seed); w = F*W/Dt + rng.normal(0, SWP, n); D = F*W/w
    Rm = er.estimate_R(D, w, F, W, SF, SW, SWP); Ro = er.estimate_R(Dt, F*W/Dt, F, W, SF, SW, SWP); Ra = np.full(n, Rm.mean())
    q, f = er.tune_q(t, D, Ra, QG); Ds = np.clip(er.rts_smoother(t, f, q)[:, 0], 3, None); Rs = er.estimate_R(Ds, F*W/Ds, F, W, SF, SW, SWP)
    row = dict(D0=D0, rep=rep, seed=seed, duration_s=dur)
    for name, R in (('A_constant', Ra), ('C_measured', Rm), ('C_smoothedD', Rs), ('C_trueD', Ro)):
        q, f = er.tune_q(t, D, R, QG); row[name] = er.rts_smoother(t, f, q)[np.argmin(abs(t - dur/2)), 2]
    return row
df = pd.DataFrame([run(D0, r) for D0 in (15, 20, 30, 40, 50, 60, 80, 100) for r in range(40)]); df.to_csv('r_distance_source_trials.csv', index=False)
V = ['A_constant', 'C_measured', 'C_smoothedD', 'C_trueD']; rows = []
for D0, g in df.groupby('D0'):
    r = dict(D0=D0)
    for v in V: e = g[v]; r[v+'_bias'] = e.mean()+1.5; r[v+'_sd'] = e.std(ddof=1); r[v+'_relRMSE_pct'] = 100*np.sqrt(np.mean((e+1.5)**2))/1.5
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv('r_distance_source_summary.csv', index=False); print(S.round(2).to_string(index=False))
