"""Tables S9 and S10 (R2-5) from results/duration_part1_summary.csv and factorial_summary.csv.
Run after run_duration_factorial.py:  python analyze_duration_factorial.py  (seconds)."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, 'results')
P1 = pd.read_csv(os.path.join(R, 'duration_part1_summary.csv')); FA = pd.read_csv(os.path.join(R, 'factorial_summary.csv'))
rows = []
for (T, sW, f), g in P1.groupby(['T', 'sigma_W', 'f']):
    g = g.sort_values('D0'); ok25 = (g.rel_rmse_pct <= 25).values; ok50 = (g.rel_rmse_pct <= 50).values
    def last_run(ok):                       # largest D0 up to which the criterion holds without interruption
        k = 0
        while k < len(ok) and ok[k]: k += 1
        return g.D0.values[k-1] if k else np.nan
    hi = g[g.ci_lo > 25]
    rows.append(dict(T_s=T, sigma_W_m=sW, f_px=f, min_valid_D0=g.D0.min(), last_D0_le25pct=last_run(ok25),
                     first_D0_ci_lower_gt25pct=hi.D0.min() if len(hi) else np.nan, last_D0_le50pct=last_run(ok50)))
S9 = pd.DataFrame(rows).sort_values(['T_s', 'sigma_W_m', 'f_px']); S9.to_csv(os.path.join(R, 'duration_crossings_TableS9.csv'), index=False)
def ols(df, extra, label):
    X = [np.ones(len(df)), np.log(df.D0), np.log(df['T']), np.log(df.f)] + [e[1] for e in extra]
    names = ['intercept', 'ln D0', 'ln T', 'ln f'] + [e[0] for e in extra]
    X = np.column_stack(X); y = np.log(df.rel_rmse_pct.values); b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; s2 = r @ r / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))); r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return [dict(term=n, coef=c, se=s, model=label.format(n=len(df), r2=r2)) for n, c, s in zip(names, b, se)]
p1 = P1.rename(columns={'f': 'f'}); out = ols(p1, [('sigma_W = 0.15 (0/1)', (p1.sigma_W == 0.15).astype(float).values)], 'Part 1 (uniform; {n} cells; R2={r2:.2f})')
fa = FA.rename(columns={'f': 'f'})
out += ols(fa, [('field-like gaps (0/1)', (fa.sampling != 'uniform').astype(float).values), ('sigma_W = 0.15 (0/1)', (fa.sigma_W == 0.15).astype(float).values)], 'Factorial ({n} cells; R2={r2:.2f})')
S10 = pd.DataFrame(out); S10.to_csv(os.path.join(R, 'duration_regression_TableS10.csv'), index=False); print(S9.to_string(index=False)); print(S10.round(3).to_string(index=False))
