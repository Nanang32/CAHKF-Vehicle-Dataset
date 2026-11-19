"""Section 3.4 / Fig. 11 regenerated with n=200 paired sequences (seeds 4000 + 17*rep as in src/ablation_and_local_tuning.local_tuning).
Reported value per event: mean estimated D2 over the 2-s maneuver; attenuation = 1 - mean_estimate/true. 95% percentile bootstrap CI of the mean (2000 resamples).
Writes results/local_tuning_largeN_trials.csv, local_tuning_largeN_summary.csv and figs/fig11_largeN.png."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,'src')); os.chdir(os.path.join(HERE,'results')); os.makedirs('figs',exist_ok=True)
import ablation_and_local_tuning as al, effective_range_analysis as er
DES={'single_event_+3.0':[(10.0,3.0)],'three_strategy_pm1.0':[(6.0,1.0),(16.0,-1.0)]}
def job(a):
    dname,rep=a; ev=DES[dname]; seed=4000+17*rep; rng=np.random.default_rng(seed)
    t,Dt,acc=al.make_sequence(ev); w=al.F_PX*al.W/Dt+rng.normal(0,al.SWP,len(t)); D=al.F_PX*al.W/w
    R=er.estimate_R(D,w,al.F_PX,al.W,al.SF,al.SW,al.SWP); est=al.strategies(t,D,R,ev); rows=[]
    for ts,a_ev in ev:
        m=(t>=ts)&(t<ts+2.0)
        for s,e in est.items(): rows.append(dict(design=dname,rep=rep,seed=seed,event_start=ts,true_D2=a_ev,strategy=s,est_mean_over_event=float(np.nanmean(e[m]))))
    return rows
if __name__=='__main__':
    N=200; jobs=[(d,r) for d in DES for r in range(N)]
    with Pool(os.cpu_count()) as p: rows=[x for rs in p.imap(job,jobs,chunksize=5) for x in rs]
    T=pd.DataFrame(rows); T.to_csv('local_tuning_largeN_trials.csv',index=False); rng=np.random.default_rng(777); out=[]
    for (d,a,s),g in T.groupby(['design','true_D2','strategy'],sort=False):
        e=g.est_mean_over_event.to_numpy(); bm=np.array([e[rng.integers(0,len(e),len(e))].mean() for _ in range(2000)])
        out.append(dict(design=d,true_D2=a,strategy=s,n=len(e),mean=e.mean(),sd=e.std(ddof=1),ci_lo=np.percentile(bm,2.5),ci_hi=np.percentile(bm,97.5),attenuation_pct=100*(1-e.mean()/a),att_lo=100*(1-np.percentile(bm,97.5)/a),att_hi=100*(1-np.percentile(bm,2.5)/a)))
    S=pd.DataFrame(out); S.to_csv('local_tuning_largeN_summary.csv',index=False); pd.set_option('display.width',200); print(S.round(3).to_string(index=False))
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    P=S[S.design=='three_strategy_pm1.0']; names={'global':'Global','local_posthoc':'Event-centered\n(±6 s)','causal_trailing':'Causal\n(preceding 6 s)'}
    fig,ax=plt.subplots(figsize=(5.6,3.3),dpi=200); x=np.arange(3); wdt=0.36
    for k,(a,c) in enumerate(((1.0,'#1f77b4'),(-1.0,'#d62728'))):
        q=P[P.true_D2==a].set_index('strategy').loc[list(names)]; ax.bar(x+(k-.5)*wdt,q['mean'],wdt,yerr=[q['mean']-q.ci_lo,q.ci_hi-q['mean']],capsize=3,color=c,label=f'true D̈ = {a:+.1f} m/s²')
    ax.axhline(1,color='#1f77b4',ls='--',lw=.8); ax.axhline(-1,color='#d62728',ls='--',lw=.8); ax.axhline(0,color='k',lw=.6)
    ax.set_xticks(x); ax.set_xticklabels(list(names.values()),fontsize=8); ax.set_ylabel('Mean estimate over event (m/s²)',fontsize=8); ax.set_ylim(-1.3,1.3); ax.legend(fontsize=7,loc='lower right'); ax.grid(alpha=.3,axis='y')
    fig.tight_layout(); fig.savefig('figs/fig11_largeN.png')
