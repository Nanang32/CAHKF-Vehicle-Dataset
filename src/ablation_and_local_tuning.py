"""Table 3 (R ablation: A constant, B empirical, C calibration-derived R(t)) and Section 3.4 / Fig. 11 (local process-noise tuning).
These two scripts were NOT archived with the original analysis; they are rewritten here from the descriptions in the manuscript.
Settings that the manuscript does not specify are fixed below and listed in README.md ("Assumptions made when re-implementing")."""
import numpy as np, pandas as pd
import effective_range_analysis as er
F_PX, W, SF, SW, SWP, FPS = 800.0, 1.8, 2.0, 0.05, 2.5, 15
QGRID = np.logspace(-3, 3, 20)

# ------------------------------------------------------------------ Table 3: ablation of R
def r_variants(D, w):
    Rc = er.estimate_R(D, w, F_PX, W, SF, SW, SWP)                    # C: calibration-derived R(t), Eqs. (8)-(9)
    A = np.full(len(D), Rc.mean())                                      # A: constant R = trial-average of R(t)
    k = 5; pad = np.pad(D, (k//2, k//2), mode='edge'); trend = np.convolve(pad, np.ones(k)/k, mode='valid')
    B = np.full(len(D), max(np.var(D - trend, ddof=1), 1e-6))           # B: empirical R from residual variance around a 5-point moving average
    return {'A_constant': A, 'B_empirical': B, 'C_calibrated': Rc}

def ablation_trial(D0, a, v0, duration, seed):
    n = int(FPS*duration); t = np.arange(n)/FPS; Dt = D0 + v0*t + 0.5*a*t**2
    if n < 14 or Dt.min() < 3.0: return None
    rng = np.random.default_rng(seed); w = F_PX*W/Dt + rng.normal(0, SWP, n); D = F_PX*W/w   # same noise draw for all three variants (paired)
    out = {}
    for name, R in r_variants(D, w).items():
        q, f = er.tune_q(t, D, R, q_grid=QGRID); xs = er.rts_smoother(t, f, q)
        out[name] = xs[np.argmin(abs(t - duration/2)), 2]
    return out

def table3(n_rep=40):
    rows = []
    for D0 in (30.0,):                                                  # 4-second short trials, near-fixed distance
        for name, a in (('braking (D2=+2.0)', 2.0), ('constant', 0.0), ('acceleration (D2=-2.0)', -2.0)):
            for rep in range(n_rep):
                seed = 7000 + int(10*a) + 31*rep; o = ablation_trial(D0, a, -1.5, 4.0, seed)
                if o: rows.append(dict(experiment='short_4s', D0=D0, maneuver=name, true_D2=a, seed=seed, rep=rep, **{k+'_est': v for k, v in o.items()}))
    for D0 in (15, 20, 25, 30, 40, 50, 60, 80, 100):                    # range simulations; sigma_D(t) grows with distance
        dur = er.safe_trial_duration(D0, v0=-0.5, true_a=-1.5)
        for rep in range(n_rep):
            seed = 9000 + int(10*D0) + 31*rep; o = ablation_trial(D0, -1.5, -0.5, dur, seed)
            if o: rows.append(dict(experiment='range', D0=D0, maneuver='D2=-1.5', true_D2=-1.5, seed=seed, rep=rep, **{k+'_est': v for k, v in o.items()}))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ Section 3.4: local q tuning (single event and Fig. 11)
def make_sequence(events, T=26.0, D0=45.0, v0=-0.5):
    """26-s sequence; every event is a 2-s constant-D2 maneuver. events = [(t_start, D2), ...]. Returns t, D_true, D2_true(t)."""
    t = np.arange(int(FPS*T))/FPS; a = np.zeros_like(t)
    for ts, a_ev in events: a[(t >= ts) & (t < ts+2.0)] = a_ev
    v = v0 + np.cumsum(a)/FPS; D = D0 + np.cumsum(v)/FPS
    return t, D, a

def _kf_forward_q(t, D, R, q): return er.kf_filter(t, D, q, R)

def tune_on(t, D, R, lo, hi):
    """q minimizing innovation RMSE of the filter run on samples [lo, hi)."""
    best = (np.inf, None)
    for q in QGRID:
        f = er.kf_filter(t[lo:hi], D[lo:hi], q, R[lo:hi]); r = er.innovation_rmse(f)
        if r < best[0]: best = (r, q)
    return best[1]

def strategies(t, D, R, events):
    out = {}
    qg = tune_on(t, D, R, 0, len(t)); xs = er.rts_smoother(t, er.kf_filter(t, D, qg, R), qg)
    out['global'] = xs[:, 2]
    loc = np.zeros(len(t)); loc[:] = np.nan                              # event-centered post-hoc: q tuned on +/-6 s around each event, RTS inside that window
    for ts, _ in events:
        c = ts+1.0; lo = int(max(0, (c-6)*FPS)); hi = int(min(len(t), (c+6)*FPS)); ql = tune_on(t, D, R, lo, hi)
        xl = er.rts_smoother(t[lo:hi], er.kf_filter(t[lo:hi], D[lo:hi], ql, R[lo:hi]), ql)[:, 2]; loc[lo:hi] = xl
    out['local_posthoc'] = loc
    caus = np.zeros(len(t)); x = None; q_k = QGRID[len(QGRID)//2]       # causal: forward filter only; q re-tuned each second on the preceding 6 s
    n = len(t); P = None
    xs_c = np.zeros((n, 3)); xs_c[0] = [D[0], 0, 0]; x = xs_c[0].copy(); P = np.diag([R[0]*10, 100., 100.])
    for k in range(1, n):
        if k % FPS == 0 and k >= 6*FPS: q_k = tune_on(t, D, R, k-6*FPS, k)
        dt = t[k]-t[k-1]; F, Q = er._F(dt), er._Q(dt, q_k); xp = F@x; Pp = F@P@F.T + Q; S = Pp[0, 0] + R[k]; K = Pp[:, 0]/S
        x = xp + K*(D[k]-xp[0]); P = Pp - np.outer(K, Pp[0, :]); xs_c[k] = x
    out['causal_trailing'] = xs_c[:, 2]
    return out

def local_tuning(n_rep=40):
    rows = []
    designs = {'single_event_+3.0': [(10.0, 3.0)], 'three_strategy_pm1.0': [(6.0, 1.0), (16.0, -1.0)]}
    for dname, ev in designs.items():
        for rep in range(n_rep):
            seed = 4000 + 17*rep; rng = np.random.default_rng(seed)
            t, Dt, a = make_sequence(ev); w = F_PX*W/Dt + rng.normal(0, SWP, len(t)); D = F_PX*W/w
            R = er.estimate_R(D, w, F_PX, W, SF, SW, SWP); est = strategies(t, D, R, ev)
            for ts, a_ev in ev:
                m = (t >= ts) & (t < ts+2.0)
                for s, e in est.items(): rows.append(dict(design=dname, rep=rep, seed=seed, event_start=ts, true_D2=a_ev, strategy=s, est_mean_over_event=float(np.nanmean(e[m])), est_peak=float(e[m][np.argmax(np.abs(e[m]))])))
    return pd.DataFrame(rows)
