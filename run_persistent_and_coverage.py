"""Reviewer comment R1-6 (b) and (c): persistent focal-length / target-width perturbations and empirical coverage of the filter's +/-2 sigma intervals.
Modes: none (as in all other scripts), persistent (f and W errors sampled ONCE per trial: eps_f ~ N(0, sigma_f/f), eps_W ~ N(0, sigma_W/W); the filter keeps the nominal f, W),
per_frame (the same errors drawn independently at every frame). Scenarios: controlled (sigma_W=0.05 m) and naturalistic (sigma_W=0.15 m); R uses the matching sigma_W.
Track: D0 grid, v0=-0.5 m/s, true D2=-1.5 m/s^2, adaptive duration, 15 FPS, 2.5 px width noise, R from the measured distance, q by innovation RMSE (same as the other scripts).
Coverage: fraction of trials in which the true D2 (at the mid sample) lies within +/-2 sqrt(P_smoothed[2,2]) of the RTS estimate; and the pooled fraction of samples where the true D lies within +/-2 sqrt(P_smoothed[0,0]).
Seeds: 31000 + 1000*D0_index + 100*scenario_index + 10*mode_index... see seed(); n=200 paired across modes (same pixel noise draw). Writes results/r16_trials.csv and r16_summary.csv."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'src')); os.chdir(os.path.join(HERE,'results'))
import effective_range_analysis as er
F,W,SF,SWP,FPS=800.,1.8,2.,2.5,15; QG=np.logspace(-3,3,20); N=200
SCEN={'controlled':0.05,'naturalistic':0.15}; MODES=['none','persistent','per_frame']; GRID=[15,30,50,80]
def smoother_cov(t,filt,q):
    n=len(t); xf,Pf=filt['x_filt'],filt['P_filt']; xs=xf.copy(); Ps=Pf.copy()
    for k in range(n-2,-1,-1):
        dt=t[k+1]-t[k]; Fm,Q=er._F(dt),er._Q(dt,q); Pp=Fm@Pf[k]@Fm.T+Q; C=Pf[k]@Fm.T@np.linalg.pinv(Pp)
        xs[k]=xf[k]+C@(xs[k+1]-Fm@xf[k]); Ps[k]=Pf[k]+C@(Ps[k+1]-Pp)@C.T
    return xs,Ps
def job(a):
    D0,scen,rep=a; sW=SCEN[scen]; dur=er.safe_trial_duration(D0,-0.5,-1.5); n=int(FPS*dur); t=np.arange(n)/FPS; Dt=D0-0.5*t+0.5*(-1.5)*t**2
    noise=np.random.default_rng(31000+1000*GRID.index(D0)+rep).normal(0,SWP,n); eps=np.random.default_rng(77000+1000*GRID.index(D0)+rep)
    ef,eW=eps.normal(0,SF/F),eps.normal(0,sW/W); efk,eWk=eps.normal(0,SF/F,n),eps.normal(0,sW/W,n); out=[]
    for mode in MODES:
        if mode=='none': fac=np.ones(n)
        elif mode=='persistent': fac=np.full(n,(1+ef)*(1+eW))
        else: fac=(1+efk)*(1+eWk)
        w=fac*F*W/Dt+noise; D=F*W/w; R=er.estimate_R(D,w,F,W,SF,sW,SWP); q,fl=er.tune_q(t,D,R,QG); xs,Ps=smoother_cov(t,fl,q)
        m=int(np.argmin(abs(t-dur/2))); sd_a=np.sqrt(max(Ps[m,2,2],0)); sd_D=np.sqrt(np.clip(Ps[:,0,0],0,None))
        out.append(dict(D0=D0,scenario=scen,mode=mode,rep=rep,duration_s=dur,q=q,scale_factor=float(fac.mean()),a_est=xs[m,2],a_sd=sd_a,a_cover=int(abs(xs[m,2]+1.5)<=2*sd_a),D_cover_frac=float(np.mean(abs(xs[:,0]-Dt)<=2*sd_D)),D_halfwidth_mean=float(np.mean(2*sd_D))))
    return out
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
if __name__=='__main__':
    jobs=[(D0,s,r) for D0 in GRID for s in SCEN for r in range(N)]
    with Pool(os.cpu_count()) as p: rows=[x for rs in p.imap(job,jobs,chunksize=10) for x in rs]
    T=pd.DataFrame(rows); T.to_csv('r16_trials.csv',index=False); out=[]
    for (D0,s,m),g in T.groupby(['D0','scenario','mode'],sort=False):
        e=g.a_est.to_numpy()+1.5; k=int(g.a_cover.sum()); lo,hi=wilson(k,len(g))
        out.append(dict(D0=D0,scenario=s,mode=m,n=len(g),bias=e.mean(),sd=e.std(ddof=1),rel_rmse_pct=100*np.sqrt(np.mean(e**2))/1.5,mean_a_halfwidth_2sigma=2*g.a_sd.mean(),a_coverage=k/len(g),a_cov_lo=lo,a_cov_hi=hi,D_pooled_coverage=g.D_cover_frac.mean(),mean_scale_factor=g.scale_factor.mean(),sd_scale_factor=g.scale_factor.std(ddof=1)))
    S=pd.DataFrame(out); S.to_csv('r16_summary.csv',index=False); pd.set_option('display.width',250); print(S.round(3).to_string(index=False))
