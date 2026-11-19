"""Table 3 (R ablation) regenerated with n=200 paired repetitions and paired bootstrap CIs on the SD ratio C/A and C/B.
Seeds: same formulas as src/ablation_and_local_tuning.table3 (7000+int(10a)+31*rep; 9000+int(10*D0)+31*rep).
Writes results/table3_largeN_trials.csv, table3_largeN_summary.csv, table3_largeN_aggregate.csv."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'src')); os.chdir(os.path.join(HERE,'results'))
import ablation_and_local_tuning as al, effective_range_analysis as er
N=200; V=['A_constant','B_empirical','C_calibrated']
def job(a):
    exp,D0,name,acc,v0,dur,rep=a
    seed=(7000+int(10*acc)+31*rep) if exp=='short_4s' else (9000+int(10*D0)+31*rep)
    o=al.ablation_trial(D0,acc,v0,dur,seed)
    return None if not o else dict(experiment=exp,D0=D0,maneuver=name,true_D2=acc,seed=seed,rep=rep,**{k+'_est':v for k,v in o.items()})
if __name__=='__main__':
    jobs=[]
    for name,a in (('braking (D2=+2.0)',2.0),('constant',0.0),('acceleration (D2=-2.0)',-2.0)):
        jobs+=[('short_4s',30.0,name,a,-1.5,4.0,r) for r in range(N)]
    for D0 in (15,20,25,30,40,50,60,80,100):
        dur=er.safe_trial_duration(D0,v0=-0.5,true_a=-1.5); jobs+=[('range',D0,'D2=-1.5',-1.5,-0.5,dur,r) for r in range(N)]
    with Pool(os.cpu_count()) as p: rows=[r for r in p.imap(job,jobs,chunksize=20) if r]
    T=pd.DataFrame(rows); T.to_csv('table3_largeN_trials.csv',index=False)
    rng=np.random.default_rng(4242); out=[]
    for (exp,D0,man),g in T.groupby(['experiment','D0','maneuver'],sort=False):
        a=g.true_D2.iloc[0]; E={v:g[v+'_est'].to_numpy() for v in V}; n=len(g); r={'experiment':exp,'D0':D0,'maneuver':man,'n_valid':n}
        for v in V: r[v+'_bias']=E[v].mean()-a; r[v+'_sd']=E[v].std(ddof=1); r[v+'_rmse']=np.sqrt(np.mean((E[v]-a)**2))
        idx=rng.integers(0,n,(2000,n))
        for v in ('B_empirical','A_constant'):
            ratio=np.array([E['C_calibrated'][i].std(ddof=1)/E[v][i].std(ddof=1) for i in idx]); r['sdratio_C_over_'+v[0]]=r['C_calibrated_sd']/r[v+'_sd']; r['ratio_lo_'+v[0]],r['ratio_hi_'+v[0]]=np.percentile(ratio,[2.5,97.5])
        out.append(r)
    S=pd.DataFrame(out); S.to_csv('table3_largeN_summary.csv',index=False)
    R=S[S.experiment=='range']; sd=R[[v+'_sd' for v in V]]
    agg=dict(n_dist=len(R),C_lowest_sd=int((sd.idxmin(axis=1)=='C_calibrated_sd').sum()),C_lower_than_A=int((R.C_calibrated_sd<R.A_constant_sd).sum()),mean_sd_A=sd.iloc[:,0].mean(),mean_sd_B=sd.iloc[:,1].mean(),mean_sd_C=sd.iloc[:,2].mean(),
        mean_sd_le60_A=R[R.D0<=60].A_constant_sd.mean(),mean_sd_le60_B=R[R.D0<=60].B_empirical_sd.mean(),mean_sd_le60_C=R[R.D0<=60].C_calibrated_sd.mean(),
        mean_absbias_A=R.A_constant_bias.abs().mean(),mean_absbias_B=R.B_empirical_bias.abs().mean(),mean_absbias_C=R.C_calibrated_bias.abs().mean())
    pd.DataFrame([agg]).to_csv('table3_largeN_aggregate.csv',index=False)
    pd.set_option('display.width',250); print(S.round(3).to_string(index=False)); print(agg)
