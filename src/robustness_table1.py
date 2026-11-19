import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('kv', 'src/kalman_vs_sg_comparison.py')
kv = importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
import effective_range_analysis as er
F, W, SF, SW, SWP = 800., 1.8, 2., .05, 2.5
p = lambda x: (round(float(np.median(x)), 2), round(float(np.percentile(x, 10)), 2), round(float(np.percentile(x, 90)), 2))
for m, a in (('acc', 2.), ('const', 0.), ('brake', -2.)):
    kb, ks, sb, ss = [], [], [], []
    for rep in range(20):
        kf = []; sg = []
        for tr in range(8):
            t, D, _ = kv._synthetic_trial(a, seed=50000 + rep * 100 + tr)
            R = er.estimate_R(D, F * W / D, F, W, SF, SW, SWP)
            q, f = er.tune_q(t, D, R); xs = er.rts_smoother(t, f, q)
            kf.append(xs[np.argmin(abs(t - 1.1)), 2])
            k, _ = kv.sg_best_window(t, D); sg.append(kv.local_poly_fit(t, D, 1.1, k)[2])
        kf = np.array(kf); sg = np.array(sg)
        kb.append(kf.mean() - a); ks.append(kf.std(ddof=1)); sb.append(sg.mean() - a); ss.append(sg.std(ddof=1))
    print(m, 'KF bias med/p10/p90', p(kb), 'KF SD', p(ks), 'SG bias', p(sb), 'SG SD', p(ss), flush=True)
