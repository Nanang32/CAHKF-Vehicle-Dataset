"""Reviewer comment R2-5: controlled-duration sweep and factorial analysis of the effective range.
Track: v0=-0.5 m/s, true D2=-1.5 m/s^2, D(t)=D0+v0 t+0.5*D2*t^2, 2.5 px width noise (independent per frame), sigma_f=2 px, W=1.8 m, R from the measured distance, q by innovation RMSE
(logspace(-3,3,20)), RTS smoother, estimate at the sample nearest T/2, relative RMSE = RMSE/1.5. A trial is run only if D(T) >= 4 m (the same floor as safe_trial_duration).
Part 1 (duration-controlled sweep, uniform 15 FPS): D0 grid x T in {2,3,4} s x sigma_W in {0.05, 0.15} m x f in {600, 800, 1000} px.
Part 2 (factorial): D0 in {15,30,50,80} x T x f x sampling {uniform 15 FPS, gaps drawn with replacement from data/gaps_empirical.csv (0<gap<=2 s)} x sigma_W.
n=200 paired trials per cell; the pixel-noise and gap draws depend only on (D0, T, rep), so cells that differ in f or sigma_W share them. Seeds: 1000000*kind + 10000*(index of D0 in D0S) + 1000*(index of T in TS) + rep (kind 1 = pixel noise, 2 = gaps).
Writes results/duration_part1_trials.csv, duration_part1_summary.csv, factorial_trials.csv, factorial_summary.csv."""
import os, sys, itertools, numpy as np, pandas as pd
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'src')); os.chdir(os.path.join(HERE,'results'))
import effective_range_analysis as er
W,SF,SWP,FPS=1.8,2.0,2.5,15; QG=np.logspace(-3,3,20); N=200; V0,A=-0.5,-1.5
GAPS=pd.read_csv(os.path.join(HERE,'data','gaps_empirical.csv')).gap_s.values; GAPS=GAPS[(GAPS>0)&(GAPS<=2)]
D0S=[8,10,12,15,18,20,25,30,35,40,45,50,60,80,100]; TS=[2,3,4]
def valid(D0,T): return D0+V0*T+0.5*A*T*T>=4.0
def times(D0,T,rep,sampling):
    if sampling=='uniform': return np.arange(int(FPS*T))/FPS
    rg=np.random.default_rng(seed(D0,T,rep,2)); t=[0.0]
    while t[-1]<T: t.append(t[-1]+rg.choice(GAPS))
    return np.array(t[:-1]) if len(t)>3 else np.array(t)
def seed(D0,T,rep,kind): return 1000000*kind+10000*D0S.index(D0)+1000*TS.index(T)+rep   # kind 1 = pixel noise, 2 = sampling gaps
def job(a):
    part,D0,T,f,sW,sampling,rep=a; t=times(D0,T,rep,sampling); n=len(t); Dt=D0+V0*t+0.5*A*t**2
    w=f*W/Dt+np.random.default_rng(seed(D0,T,rep,1)).normal(0,SWP,max(n,1))[:n]; D=f*W/w; R=er.estimate_R(D,w,f,W,SF,sW,SWP); q,fl=er.tune_q(t,D,R,QG)
    m=int(np.argmin(abs(t-T/2))); est=er.rts_smoother(t,fl,q)[m,2]
    return dict(part=part,D0=D0,T=T,f=f,sigma_W=sW,sampling=sampling,rep=rep,n_samples=n,q=q,est=est)
def summarize(T_,keys,rng):
    out=[]
    for k,g in T_.groupby(keys,sort=False):
        e=g.est.to_numpy()-A; rel=100*np.sqrt(np.mean(e**2))/abs(A)
        idx=rng.integers(0,len(e),(2000,len(e))); bs=100*np.sqrt((e[idx]**2).mean(axis=1))/abs(A)
        out.append(dict(zip(keys,k if isinstance(k,tuple) else (k,)),n=len(e),bias=e.mean(),sd=e.std(ddof=1),rel_rmse_pct=rel,ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5),mean_samples=g.n_samples.mean()))
    return pd.DataFrame(out)
if __name__=='__main__':
    j1=[('p1',D0,T,f,sW,'uniform',r) for D0 in D0S for T in TS if valid(D0,T) for f in (600,800,1000) for sW in (0.05,0.15) for r in range(N)]
    j2=[('p2',D0,T,f,sW,s,r) for D0 in (15,30,50,80) for T in TS if valid(D0,T) for f in (600,800,1000) for s in ('uniform','gaps') for sW in (0.05,0.15) for r in range(N)]
    print(len(j1),len(j2),flush=True)
    with Pool(os.cpu_count()) as p:
        T1=pd.DataFrame(list(p.imap(job,j1,chunksize=50))); T1.to_csv('duration_part1_trials.csv',index=False)
        T2=pd.DataFrame(list(p.imap(job,j2,chunksize=50))); T2.to_csv('factorial_trials.csv',index=False)
    rng=np.random.default_rng(8642)
    summarize(T1,['T','sigma_W','f','D0'],rng).to_csv('duration_part1_summary.csv',index=False)
    summarize(T2,['D0','T','f','sampling','sigma_W'],rng).to_csv('factorial_summary.csv',index=False); print('done')
