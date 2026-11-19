"""Reviewer comment R2-4: forward-only filter versus fixed-lag smoothers versus full RTS, on UNSEEN maneuver times.
Test sequences: 26 s, D0=45 m, v0=-0.5 m/s, 15 FPS, 2.5 px width noise, R from the measured distance; three 2-s maneuvers of D2 = +/-1.0 m/s^2 with alternating signs
(random initial sign), start of the first event U(3,6) s and gaps U(6,8) s, so event times differ in every sequence.
q is tuned by innovation RMSE (no ground truth) on a SEPARATE training sequence of the same kind (independent event times and noise) and then fixed for the test sequence.
Estimators: forward-only Kalman (x_filt), fixed-lag RTS with lag 0.5, 1 and 2 s (backward recursion over the lag window), the full RTS smoother (offline upper bound), and, to expose the responsiveness/noise trade-off, forward-only filters with q multiplied by 10 and 100.
Metrics per event (window [ts, ts+4 s]): delay = first time after ts at which the estimate reaches 50% of |true D2| with the correct sign; missed = never reached; peak ratio = peak estimate in the
event direction divided by |true D2|; also mean estimate over the 2-s maneuver and the whole-sequence RMSE of D2 (all in m/s^2). n=200 paired sequences; 95% percentile bootstrap intervals.
Seeds: test 50000 + rep, training 60000 + rep. Writes results/causal_lag_trials.csv, causal_lag_summary.csv, figs/fig_causal_lag.png."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'src')); os.chdir(os.path.join(HERE,'results')); os.makedirs('figs',exist_ok=True)
import ablation_and_local_tuning as al, effective_range_analysis as er
FPS=15; QG=np.logspace(-3,3,20); LAGS=[0.5,1.0,2.0]; METHODS=['forward','forward_q10x','forward_q100x']+[f'lag_{l:g}s' for l in LAGS]+['rts_full']
def make(seed):
    rng=np.random.default_rng(seed); s=rng.choice([-1.0,1.0]); t0=rng.uniform(3,6); ev=[]; 
    for k in range(3):
        ev.append((t0,s*1.0)); s=-s; t0+=rng.uniform(6,8)
    t,Dt,a=al.make_sequence(ev); w=al.F_PX*al.W/Dt+rng.normal(0,al.SWP,len(t)); D=al.F_PX*al.W/w; R=er.estimate_R(D,w,al.F_PX,al.W,al.SF,al.SW,al.SWP); return ev,t,Dt,a,D,R
def fixed_lag(t,filt,q,m):
    xf,Pf=filt['x_filt'],filt['P_filt']; n=len(t); out=np.zeros(n)
    for k in range(n):
        hi=min(n-1,k+m); xs=xf[hi].copy()
        for j in range(hi-1,k-1,-1):
            dt=t[j+1]-t[j]; F,Q=er._F(dt),er._Q(dt,q); Pp=F@Pf[j]@F.T+Q; C=Pf[j]@F.T@np.linalg.pinv(Pp); xs=xf[j]+C@(xs-F@xf[j])
        out[k]=xs[2]
    return out
def job(rep):
    _,tt,_,_,Dtr,Rtr=make(60000+rep); q,_=er.tune_q(tt,Dtr,Rtr,QG)
    ev,t,Dt,a,D,R=make(50000+rep); filt=er.kf_filter(t,D,q,R)
    est={'forward':filt['x_filt'][:,2],'forward_q10x':er.kf_filter(t,D,10*q,R)['x_filt'][:,2],'forward_q100x':er.kf_filter(t,D,100*q,R)['x_filt'][:,2]}
    for l in LAGS: est[f'lag_{l:g}s']=fixed_lag(t,filt,q,int(round(l*FPS)))
    est['rts_full']=er.rts_smoother(t,filt,q)[:,2]; rows=[]
    quiet=np.ones(len(t),bool)
    for ts,_ in ev: quiet&=~((t>=ts-1.5)&(t<=ts+4.5))
    for me,e in est.items():
        rm=float(np.sqrt(np.mean((e-a)**2))); fa=float(np.mean(abs(e[quiet])>=0.5)) if quiet.any() else np.nan; nq=int(quiet.sum())
        for i,(ts,aev) in enumerate(ev):
            sg=np.sign(aev); win=(t>=ts)&(t<=ts+4.0); m2=(t>=ts)&(t<ts+2.0); x=sg*e[win]; tw=t[win]
            hit=np.where(x>=0.5)[0]; delay=float(tw[hit[0]]-ts) if len(hit) else np.nan
            rows.append(dict(rep=rep,event=i,method=me,true_D2=aev,q=q,delay_s=delay,missed=int(len(hit)==0),peak_ratio=float(x.max()/abs(aev)),mean_over_event=float(sg*e[m2].mean()),seq_rmse=rm,false_alarm_frac=fa,n_quiet=nq))
    return rows
def boot(x,rng,B=2000):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    if len(x)==0: return (np.nan,np.nan,np.nan)
    bm=np.array([x[rng.integers(0,len(x),len(x))].mean() for _ in range(B)]); return x.mean(),*np.percentile(bm,[2.5,97.5])
if __name__=='__main__':
    N=200
    with Pool(os.cpu_count()) as p: rows=[x for rs in p.imap(job,range(N),chunksize=4) for x in rs]
    T=pd.DataFrame(rows); T.to_csv('causal_lag_trials.csv',index=False); rng=np.random.default_rng(2468); out=[]
    for me in METHODS:
        g=T[T.method==me]; r=dict(method=me,n_events=len(g),missed_pct=100*g.missed.mean())
        lo=100*boot(g.missed,rng)[1]; hi=100*boot(g.missed,rng)[2]; r['missed_lo'],r['missed_hi']=lo,hi
        for c in ('delay_s','peak_ratio','mean_over_event'): r[c],r[c+'_lo'],r[c+'_hi']=boot(g[c],rng)
        s=g.drop_duplicates(['rep']); r['seq_rmse'],r['seq_rmse_lo'],r['seq_rmse_hi']=boot(s.seq_rmse,rng)
        r['delay_median_s']=float(np.nanmedian(g.delay_s)); qq=s[s.n_quiet>0]; r['false_alarm_pct']=100*float(np.average(qq.false_alarm_frac,weights=qq.n_quiet)); out.append(r)
    S=pd.DataFrame(out); S.to_csv('causal_lag_summary.csv',index=False); pd.set_option('display.width',260); print(S.round(3).to_string(index=False)); print('q range',T.q.min(),T.q.median(),T.q.max())
