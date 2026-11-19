"""Reviewer comment R2-2: additional comparators on the SAME detections, timestamps and trajectories.
Estimators: KF_RTS (calibration-derived R(t), q by innovation RMSE, RTS smoother; the proposed estimator), SG (unweighted generalized local quadratic, k by LOO-CV),
HLP (heteroscedastic local polynomial regression = the same local quadratic fitted by weighted least squares with weights 1/R_k, R_k = sigma_D,k^2 of Eqs. (8)-(9); k by weighted LOO-CV,
same candidates {5,7,9,11} as SG), KF_forward (forward-only Kalman, same q), IMM_forward (two-mode interacting multiple model on the same state, q_lo = tuned q, q_hi = 100*q_lo (fixed a priori),
Markov matrix [[0.95,0.05],[0.10,0.90]], initial mode probabilities [0.9,0.1]). KF_RTS, SG and HLP are offline (use the whole trial); KF_forward and IMM_forward are causal (zero lag) and are compared with each other.
Scenarios (200 paired trials each; seeds fixed): S1 Table-1 conditions (D0=30 m, v0=-1.5 m/s, 4 s, 15 FPS, 2.5 px, D2 in {+2, 0, -2}; estimate at the sample nearest 2 s), S2 range (D0 in {20,30,50,80} m, 4 s, v0=-0.5, D2=-1.5),
S3 event sequences (the 26-s sequences of run_causal_lag_tradeoff.py, same seeds 50000+rep; q and k chosen on the separate training sequence 60000+rep).
Metrics: bias, SD, RMSE (m/s^2), RMSE ratio to KF_RTS with a paired bootstrap interval; for S3 also sequence RMSE, mean estimate during the 2-s maneuver (sign-adjusted), delay (first time >= 50 % of |D2|) and false alarms (quiet samples with |estimate| >= 0.5).
Writes results/comparators_trials.csv, comparators_summary.csv, comparators_events_summary.csv."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'src')); os.chdir(os.path.join(HERE,'results'))
import effective_range_analysis as er, ablation_and_local_tuning as al
FPS=15; F,W,SF,SW,SWP=800.,1.8,2.,.05,2.5; QG=np.logspace(-3,3,20); KC=(5,7,9,11); KE=(5,7,9,11,15,21,31,41,61)   # KC: windows of the manuscript's SG baseline; KE: extended windows (up to 4 s at 15 FPS)
def lfit(t,y,t0,k,wts=None,excl=False):
    o=np.argsort(abs(t-t0)); 
    if excl: o=o[t[o]!=t0]
    i=o[:k]; dt=t[i]-t0; A=np.vander(dt,3,increasing=True)
    if wts is None: c=np.linalg.lstsq(A,y[i],rcond=None)[0]
    else: s=np.sqrt(wts[i]); c=np.linalg.lstsq(A*s[:,None],y[i]*s,rcond=None)[0]
    return c[0],2*c[2]
def best_k(t,y,wts=None,cands=KC):
    sc={}
    for k in cands:
        if k>len(t): continue
        e=np.array([lfit(t,y,t[i],k,wts,True)[0]-y[i] for i in range(len(t))])
        sc[k]=np.sqrt(np.mean(e**2)) if wts is None else np.sqrt(np.sum(wts*e**2)/np.sum(wts))
    return min(sc,key=sc.get)
def local_all(t,y,k,wts=None): return np.array([lfit(t,y,t[i],k,wts)[1] for i in range(len(t))])
def imm_forward(t,z,R,q,qhi_ratio=100.):
    n=len(t); H=np.array([1.,0,0]); Pi=np.array([[.95,.05],[.10,.90]]); mu=np.array([.9,.1]); qs=[q,qhi_ratio*q]
    x=[np.array([z[0],0,0.]) for _ in range(2)]; P=[np.diag([R[0]*10,100.,100.]) for _ in range(2)]; out=np.zeros((n,3)); out[0]=x[0]; mus=np.zeros((n,2)); mus[0]=mu
    for k in range(1,n):
        dt=t[k]-t[k-1]; Fm=er._F(dt); c=Pi.T@mu; mix=(Pi*mu[:,None])/c[None,:]   # mix[i,j]=P(i|j)
        xm=[sum(mix[i,j]*x[i] for i in range(2)) for j in range(2)]; Pm=[sum(mix[i,j]*(P[i]+np.outer(x[i]-xm[j],x[i]-xm[j])) for i in range(2)) for j in range(2)]
        lik=np.zeros(2)
        for j in range(2):
            xp=Fm@xm[j]; Pp=Fm@Pm[j]@Fm.T+er._Q(dt,qs[j]); S=Pp[0,0]+R[k]; nu=z[k]-xp[0]; K=Pp[:,0]/S; x[j]=xp+K*nu; P[j]=Pp-np.outer(K,Pp[0,:]); lik[j]=np.exp(-0.5*nu*nu/S)/np.sqrt(2*np.pi*S)
        mu=lik*c; mu=mu/mu.sum() if mu.sum()>0 else c; out[k]=mu[0]*x[0]+mu[1]*x[1]; mus[k]=mu
    return out[:,2]
def estimators(t,D,R,q,ks):
    kS,kH,kSe,kHe=ks; f=er.kf_filter(t,D,q,R); d={'KF_RTS':er.rts_smoother(t,f,q)[:,2],'SG_LOO4':local_all(t,D,kS),'HLP_LOO4':local_all(t,D,kH,1./R),'SG_extLOO':local_all(t,D,kSe),'HLP_extLOO':local_all(t,D,kHe,1./R),'KF_forward':f['x_filt'][:,2],'IMM_forward':imm_forward(t,D,R,q)}
    for k in KE:
        if k<=len(t): d[f'SG_k{k}']=local_all(t,D,k); d[f'HLP_k{k}']=local_all(t,D,k,1./R)
    return d
def pick(t,D,R): return (best_k(t,D),best_k(t,D,1./R),best_k(t,D,None,KE),best_k(t,D,1./R,KE))
def trial_point(a):
    sc,cond,D0,T,v0,acc,rep=a; n=int(FPS*T); t=np.arange(n)/FPS; Dt=D0+v0*t+.5*acc*t*t; seed=400000+1000*cond+rep
    w=F*W/Dt+np.random.default_rng(seed).normal(0,SWP,n); D=F*W/w; R=er.estimate_R(D,w,F,W,SF,SW,SWP); q,_=er.tune_q(t,D,R,QG); ks=pick(t,D,R)
    est=estimators(t,D,R,q,ks); m=int(np.argmin(abs(t-T/2))); return [dict(scenario=sc,cond=f'D0={D0}, D2={acc:+.1f}',D0=D0,true_D2=acc,rep=rep,method=k,est=float(v[m]),q=q,kS=ks[0],kH=ks[1],kSe=ks[2],kHe=ks[3]) for k,v in est.items()]
import run_causal_lag_tradeoff as rc
def trial_event(rep):
    _,tt,_,_,Dtr,Rtr=rc.make(60000+rep); q,_=er.tune_q(tt,Dtr,Rtr,QG); ks=pick(tt,Dtr,Rtr)
    ev,t,Dt,a,D,R=rc.make(50000+rep); est=estimators(t,D,R,q,ks); quiet=np.ones(len(t),bool)
    for ts,_ in ev: quiet&=~((t>=ts-1.5)&(t<=ts+4.5))
    rows=[]
    for me,e in est.items():
        rm=float(np.sqrt(np.mean((e-a)**2))); fa=float(np.mean(abs(e[quiet])>=.5)) if quiet.any() else np.nan
        for i,(ts,aev) in enumerate(ev):
            sg=np.sign(aev); win=(t>=ts)&(t<=ts+4.); m2=(t>=ts)&(t<ts+2.); x=sg*e[win]; hit=np.where(x>=.5)[0]
            rows.append(dict(rep=rep,event=i,method=me,delay_s=float(t[win][hit[0]]-ts) if len(hit) else np.nan,missed=int(len(hit)==0),mean_over_event=float(sg*e[m2].mean()),seq_rmse=rm,false_alarm_frac=fa,n_quiet=int(quiet.sum())))
    return rows
def boot_ratio(e,ref,rng,B=2000):
    e=np.asarray(e);ref=np.asarray(ref);idx=rng.integers(0,len(e),(B,len(e))); r=np.sqrt((e[idx]**2).mean(1))/np.sqrt((ref[idx]**2).mean(1)); return np.percentile(r,[2.5,97.5])
def post(T,E):
    rng=np.random.default_rng(97531); out=[]; orc=[]; ks_chosen={}
    for (sc,cond),g in T.groupby(['scenario','cond'],sort=False):
        a=g.true_D2.iloc[0]
        for fam in ('SG','HLP'):
            ks=[k for k in KE if (g.method==f'{fam}_k{k}').any()]
            best=min(ks,key=lambda k:np.sqrt(np.mean((g[g.method==f'{fam}_k{k}'].est.to_numpy()-a)**2))); ks_chosen[(sc,cond,fam)]=best   # oracle window (uses the true value; not deployable)
            h=g[g.method==f'{fam}_k{best}'].copy(); h['method']=f'{fam}_oracle'; orc.append(h)
    T=pd.concat([T[~T.method.str.contains(r'_k\d+$')]]+orc,ignore_index=True)
    for (sc,cond),g in T.groupby(['scenario','cond'],sort=False):
        a=g.true_D2.iloc[0]; P=g.pivot(index='rep',columns='method',values='est'); err=P-a
        for me in P.columns:
            e=err[me].to_numpy(); r=dict(scenario=sc,cond=cond,method=me,n=len(e),bias=e.mean(),sd=e.std(ddof=1),rmse=np.sqrt(np.mean(e**2)))
            ref='KF_forward' if me in ('KF_forward','IMM_forward') else 'KF_RTS'; r['reference']=ref; r['rmse_ratio_vs_ref']=r['rmse']/np.sqrt(np.mean(err[ref].to_numpy()**2)); r['ratio_lo'],r['ratio_hi']=boot_ratio(e,err[ref].to_numpy(),rng); out.append(r)
    pd.DataFrame(out).to_csv('comparators_summary.csv',index=False)
    sq={me:E[(E.method==me)].drop_duplicates('rep').seq_rmse.mean() for me in E.method.unique()}
    orcE=[]
    for fam in ('SG','HLP'):
        best=min([k for k in KE if f'{fam}_k{k}' in sq],key=lambda k:sq[f'{fam}_k{k}']); ks_chosen[('S3','events',fam)]=best; h=E[E.method==f'{fam}_k{best}'].copy(); h['method']=f'{fam}_oracle'; orcE.append(h)
    E=pd.concat([E[~E.method.str.contains(r'_k\d+$')]]+orcE,ignore_index=True)
    pd.Series({str(k):v for k,v in ks_chosen.items()},name='oracle_window_k').to_csv('comparators_oracle_k.csv')
    ev=[]
    for me,g in E.groupby('method',sort=False):
        s=g.drop_duplicates('rep'); qq=s[s.n_quiet>0]
        ev.append(dict(method=me,n_events=len(g),seq_rmse=s.seq_rmse.mean(),mean_over_event=g.mean_over_event.mean(),delay_s=g.delay_s.mean(),missed_pct=100*g.missed.mean(),false_alarm_pct=100*np.average(qq.false_alarm_frac,weights=qq.n_quiet)))
    pd.DataFrame(ev).to_csv('comparators_events_summary.csv',index=False)
    pd.set_option('display.width',250); print(pd.DataFrame(out).round(3).to_string(index=False)); print(pd.DataFrame(ev).round(3).to_string(index=False))

if __name__=='__main__':
    J=[]; 
    for ci,(D0,T,v0,acc) in enumerate([(30,4,-1.5,2.0),(30,4,-1.5,0.0),(30,4,-1.5,-2.0)]): J+=[('S1',ci,D0,T,v0,acc,r) for r in range(200)]
    for ci,D0 in enumerate([20,30,50,80]): J+=[('S2',10+ci,D0,4,-0.5,-1.5,r) for r in range(200)]
    with Pool(os.cpu_count()) as p:
        rows=[x for rs in p.imap(trial_point,J,chunksize=5) for x in rs]; E=pd.DataFrame([x for rs in p.imap(trial_event,range(200),chunksize=2) for x in rs])
    T=pd.DataFrame(rows); T.to_csv('comparators_raw_trials.csv',index=False); E.to_csv('comparators_raw_events_trials.csv',index=False)
    post(T,E)
