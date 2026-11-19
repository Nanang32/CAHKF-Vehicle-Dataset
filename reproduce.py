"""Reproduces the per-trial results behind Table 1 (regenerated), Fig. 9 (reconstruction RMSE),
Table 4 / Table S2 (field-informed simulation) and Figs. 8 and 10(b) (effective-range sweep).
All random numbers come from numpy.random.default_rng with fixed integer seeds (listed in the CSV files).
Run:  python reproduce.py        (about 10-20 min on one CPU core; writes results/*.csv)"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'src'))
os.chdir(os.path.join(HERE, 'results'))
import kalman_vs_sg_comparison as kv
import effective_range_analysis as er
import field_informed_simulation as fi

F_PX, W, SF, SW, SWP = 800.0, 1.8, 2.0, 0.05, 2.5      # focal length (px), target width (m), sigma_f (px), sigma_W (m), sigma_w (px)
MARK = 1.1                                              # estimate read at t = 1.1 s

def recon_rmse(t, D_ref, vel, acc):
    """Euler/second-order integration of the estimated acceleration, started from the first valid sample
    (initial D = D_ref[first], initial v = vel[first]); RMSE against D_ref (the estimator's own smoothed D)."""
    valid = ~np.isnan(acc) & ~np.isnan(vel); iv = np.where(valid)[0]
    if len(iv) < 3: return np.nan
    v = np.full_like(t, np.nan); D = np.full_like(t, np.nan); v[iv[0]] = vel[iv[0]]; D[iv[0]] = D_ref[iv[0]]
    for j in range(1, len(iv)):
        i0, i1 = iv[j-1], iv[j]; dt = t[i1] - t[i0]
        v[i1] = v[i0] + acc[i0]*dt; D[i1] = D[i0] + v[i0]*dt + 0.5*acc[i0]*dt**2
    return float(np.sqrt(np.nanmean((D[iv] - D_ref[iv])**2)))

def table1_and_fig9():
    MAN = {'braking (D2=+2.0)': 2.0, 'constant': 0.0, 'acceleration (D2=-2.0)': -2.0}
    SEED0 = {'braking (D2=+2.0)': 1100, 'constant': 2200, 'acceleration (D2=-2.0)': 3300}
    rows = []
    for m, a in MAN.items():
        for tr in range(8):
            seed = SEED0[m] + tr*17
            t, D, _ = kv._synthetic_trial(a, seed=seed)          # D0=30 m, v0=-1.5 m/s, 15 FPS, 4 s, sigma_w=2.5 px
            R = er.estimate_R(D, F_PX*W/D, F_PX, W, SF, SW, SWP)  # R(t) = sigma_D^2 from the measured distance, Eqs. (8)-(9)
            q, f = er.tune_q(t, D, R); xs = er.rts_smoother(t, f, q)
            kf = xs[np.argmin(abs(t - MARK)), 2]; kv_ = xs[np.argmin(abs(t - MARK)), 1]
            k, _ = kv.sg_best_window(t, D); _, _, sg = kv.local_poly_fit(t, D, MARK, k)
            Dsg, vsg, asg = kv.sg_apply(t, D, k)
            rows.append(dict(condition=m, trial=tr, seed=seed, true_D2=a, q_selected=q, sg_k=k, kf_estimate=kf, kf_velocity_mps=kv_, sg_estimate=sg,
                             kf_error=kf-a, sg_error=sg-a,
                             kf_recon_rmse_m=recon_rmse(t, xs[:, 0], xs[:, 1], xs[:, 2]), sg_recon_rmse_m=recon_rmse(t, Dsg, vsg, asg)))
    df = pd.DataFrame(rows); df.to_csv('table1_fig9_trials.csv', index=False)
    S = df.groupby('condition', sort=False).agg(kf_bias=('kf_error', 'mean'), kf_sd=('kf_estimate', lambda x: x.std(ddof=1)),
            sg_bias=('sg_error', 'mean'), sg_sd=('sg_estimate', lambda x: x.std(ddof=1)),
            kf_speed_sd_kmh=('kf_velocity_mps', lambda x: 3.6*x.std(ddof=1)), kf_recon=('kf_recon_rmse_m', 'mean'), sg_recon=('sg_recon_rmse_m', 'mean')).reset_index()
    S.to_csv('table1_fig9_summary.csv', index=False); print(S.round(3).to_string(index=False))

def table4():
    d = pd.read_csv(os.path.join(HERE, 'data', 'gaps_empirical.csv')).gap_s.values
    pool = d[(d > 0) & (d <= 2.0)]; rows = []
    for mode in ('uniform', 'real-gaps'):
        for sw in (2.5, 1.7, 0.96):
            for name, a in (('braking (D2=+2.0)', 2.), ('acceleration (D2=-2.0)', -2.), ('constant', 0.)):
                seed = 1000 + int(sw*100) + int(a*10); rng = np.random.default_rng(seed)
                for rep in range(40):
                    sg, kf, n = fi.trial(a, sw, mode, rng, pool=pool)
                    rows.append(dict(sampling=mode, sigma_w_px=sw, maneuver=name, true_D2=a, rng_seed=seed, repetition=rep,
                                     n_samples=n, kf_estimate=kf, sg_estimate=sg, kf_error=kf-a, sg_error=sg-a))
    df = pd.DataFrame(rows); df.to_csv('table4_trials.csv', index=False)
    c = df.groupby(['sampling', 'sigma_w_px', 'maneuver'], sort=False).agg(
        kf_rmse=('kf_error', lambda x: np.sqrt(np.mean(x**2))), sg_rmse=('sg_error', lambda x: np.sqrt(np.mean(x**2)))).reset_index()
    c['ratio_sg_over_kf'] = c.sg_rmse/c.kf_rmse; c.to_csv('table4_conditions_TableS2.csv', index=False)
    A = c.groupby(['sampling', 'sigma_w_px'], sort=False)[['kf_rmse', 'sg_rmse']].mean().reset_index()   # Table 4 = mean of 3 per-maneuver RMSEs
    A.to_csv('table4_aggregate.csv', index=False); print(A.round(2).to_string(index=False))

def fig8_fig10():
    rows = []
    for D0 in [8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100]:
        dur = er.safe_trial_duration(D0, v0=-0.5, true_a=-1.5)
        for tr in range(12):
            seed = tr*97 + int(D0*10)
            e = er.run_single_trial(D0, -1.5, F_PX, W, SF, SW, SWP, duration_s=dur, seed=seed)
            rows.append(dict(D0=D0, trial=tr, seed=seed, duration_s=dur, true_D2=-1.5, kf_estimate=e))
    df = pd.DataFrame(rows); df.to_csv('fig8_fig10_trials.csv', index=False)
    g = df.dropna().groupby('D0').kf_estimate.agg(['count', 'mean', 'std', lambda x: np.sqrt(np.mean((x+1.5)**2))]).reset_index()
    g.columns = ['D0', 'n_valid', 'mean_est', 'std', 'rmse']; g['rel_rmse_pct'] = 100*g.rmse/1.5
    g.to_csv('fig8_fig10_summary.csv', index=False); print(g.round(2).to_string(index=False))

if __name__ == '__main__':
    for name, fn in (('Table 1 / Fig. 9', table1_and_fig9), ('Table 4 / Table S2', table4), ('Figs. 8 and 10(b)', fig8_fig10)):
        print('\n=== ' + name + ' ==='); fn()
