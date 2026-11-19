"""Reviewer comment R2-3: does the gain of the calibration-derived R(t) (variant C) over constant R (A) and empirical R (B) survive departures from the simulation assumptions?
Base track: D0 in {20,30,40,50} m, v0=-0.5 m/s, true D2=-1.5 m/s^2, 4 s, 15 FPS, f=800 px, W=1.8 m, 2.5 px independent width noise, sigma_f=2 px, sigma_W=0.05 m; same variants, tuning and RTS smoother as Table 3
(src/ablation_and_local_tuning.r_variants); estimate at the sample nearest 2 s. Assumptions chosen by the authors (not specified by the reviewer; stated in Supplementary S15):
 yaw: apparent width w = f*(W*cos(theta)+L*sin(theta))/D with L = 4.5 m (vehicle length) and theta ramping linearly from 0 to 2, 5 or 10 degrees over the trial (the filter keeps the nominal W);
 dropout: one burst of 6 consecutive frames (0.4 s) removed at a random time in [0.5, 3.0] s;
 outlier: one burst of 6 consecutive frames at a random time in [0.5, 3.0] s in which the measured width is multiplied by 0.6 (truncated box), without outlier rejection;
 correlated: width noise is AR(1) with correlation 0.5 or 0.9 between consecutive frames, marginal SD kept at 2.5 px (the filters assume independence).
Pixel noise and burst times depend only on (D0, repetition) so all conditions and all variants are paired. 200 trials per cell. Statistics of C minus A (and B): difference of mean squared error,
difference of |bias| and difference of SD, each with a 95% paired bootstrap interval (2000 resamples). Writes results/robust_trials.csv, robust_summary.csv."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'src')); os.chdir(os.path.join(HERE,'results'))
import effective_range_analysis as er, ablation_and_local_tuning as al
F,W,L,FPS,T,V0,A=800.,1.8,4.5,15,4.0,-0.5,-1.5; SWP=2.5; QG=np.logspace(-3,3,20); N=200; D0S=[20,30,40,50]
CONDS=['base','yaw2','yaw5','yaw10','dropout','outlier','ar0.5','ar0.9']
def ar1(rng,n,rho,sd):
    e=rng.normal(0,sd,n); x=np.zeros(n); x[0]=e[0]
    for k in range(1,n): x[k]=rho*x[k-1]+np.sqrt(1-rho*rho)*e[k]
    return x
def job(a):
    D0,rep=a; n=int(FPS*T); t=np.arange(n)/FPS; Dt=D0+V0*t+0.5*A*t*t; rows=[]
    for cond in CONDS:
        rng=np.random.default_rng(500000+1000*D0S.index(D0)+rep); noise=rng.normal(0,SWP,n); rb=np.random.default_rng(600000+1000*D0S.index(D0)+rep); tb=rb.uniform(0.5,3.0)
        if cond.startswith('ar'): noise=ar1(np.random.default_rng(500000+1000*D0S.index(D0)+rep),n,float(cond[2:]),SWP)
        th=np.deg2rad(float(cond[3:]))*t/T if cond.startswith('yaw') else np.zeros(n)
        w=F*(W*np.cos(th)+L*np.sin(th))/Dt+noise; tt=t.copy()
        if cond=='outlier': m=(t>=tb)&(t<tb+6/FPS); w[m]*=0.6
        if cond=='dropout': keep=~((t>=tb)&(t<tb+6/FPS)); w=w[keep]; tt=t[keep]
        D=F*W/w; out={}
        for name,R in al.r_variants(D,w).items():
            q,fl=er.tune_q(tt,D,R,QG); xs=er.rts_smoother(tt,fl,q); out[name]=float(xs[np.argmin(abs(tt-T/2)),2])
        rows.append(dict(cond=cond,D0=D0,rep=rep,n_samples=len(tt),**{k+'_est':v for k,v in out.items()}))
    return rows
def stats(e):
    return np.abs(e.mean()),e.std(ddof=1),np.mean(e**2)
def boot_diff(eC,eX,rng,B=2000):
    n=len(eC); idx=rng.integers(0,n,(B,n)); c=eC[idx]; x=eX[idx]
    d_mse=(c**2).mean(1)-(x**2).mean(1); d_bias=np.abs(c.mean(1))-np.abs(x.mean(1)); d_sd=c.std(1,ddof=1)-x.std(1,ddof=1)
    f=lambda v:np.percentile(v,[2.5,97.5]); return f(d_mse),f(d_bias),f(d_sd)
if __name__=='__main__':
    with Pool(os.cpu_count()) as p: rows=[x for rs in p.imap(job,[(D0,r) for D0 in D0S for r in range(N)],chunksize=5) for x in rs]
    T_=pd.DataFrame(rows); T_.to_csv('robust_trials.csv',index=False); rng=np.random.default_rng(1357); out=[]
    for (cond,D0),g in T_.groupby(['cond','D0'],sort=False):
        e={v:g[v+'_est'].to_numpy()-A for v in ('A_constant','B_empirical','C_calibrated')}; r=dict(cond=cond,D0=D0,n=len(g),mean_samples=g.n_samples.mean())
        for v,x in e.items(): b,s,m=stats(x); r[v[0]+'_bias']=x.mean(); r[v[0]+'_sd']=s; r[v[0]+'_rmse']=np.sqrt(m)
        for ref in ('A_constant','B_empirical'):
            (m_lo,m_hi),(b_lo,b_hi),(s_lo,s_hi)=boot_diff(e['C_calibrated'],e[ref],rng); k='CA' if ref[0]=='A' else 'CB'
            r[k+'_dMSE']=np.mean(e['C_calibrated']**2)-np.mean(ref and e[ref]**2); r[k+'_dMSE_lo'],r[k+'_dMSE_hi']=m_lo,m_hi
            r[k+'_dAbsBias']=abs(e['C_calibrated'].mean())-abs(e[ref].mean()); r[k+'_dAbsBias_lo'],r[k+'_dAbsBias_hi']=b_lo,b_hi
            r[k+'_dSD']=e['C_calibrated'].std(ddof=1)-e[ref].std(ddof=1); r[k+'_dSD_lo'],r[k+'_dSD_hi']=s_lo,s_hi
        out.append(r)
    S=pd.DataFrame(out); S.to_csv('robust_summary.csv',index=False); pd.set_option('display.width',280)
    print(S[['cond','D0','n_samples'.replace('n_samples','mean_samples'),'A_bias','C_bias','A_sd','C_sd','A_rmse','C_rmse','CA_dMSE','CA_dMSE_lo','CA_dMSE_hi','CA_dSD','CA_dSD_lo','CA_dSD_hi']].round(3).to_string(index=False))
