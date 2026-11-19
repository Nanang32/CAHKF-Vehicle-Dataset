import sys, importlib.util, numpy as np, pandas as pd
import os
HERE=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location('kv',os.path.join(HERE,'kalman_vs_sg_comparison.py'))
kv=importlib.util.module_from_spec(spec); spec.loader.exec_module(kv)
F_PX,W,SF,SW=800.0,1.8,2.0,0.05

def kf_het(t,z,q,rvec):
    n=len(t); x=np.array([z[0],0,0.]); P=np.diag([rvec[0]*10,100.,100.])
    xf=np.zeros((n,3)); Pf=np.zeros((n,3,3)); inn=np.full(n,np.nan); xf[0],Pf[0]=x,P
    for k in range(1,n):
        dt=t[k]-t[k-1]; F=kv._F(dt); Q=kv._Q(dt,q)
        xp=F@x; Pp=F@P@F.T+Q
        S=Pp[0,0]+rvec[k]; e=z[k]-xp[0]; inn[k]=e
        K=Pp[:,0]/S; x=xp+K*e; P=Pp-np.outer(K,Pp[0,:])
        xf[k],Pf[k]=x,P
    return dict(x_filt=xf,P_filt=Pf,innovations=inn)

def run_kf(t,D,w_meas,sw):
    rvec=(D**2)*((SF/F_PX)**2+(SW/W)**2+(sw/w_meas)**2)   # R(t)=sigma_D^2, Eq.(4)
    best=(np.inf,None,None)
    for q in np.logspace(-3,3,25):
        f=kf_het(t,D,q,rvec); r=kv.innovation_rmse(f)
        if r<best[0]: best=(r,q,f)
    return kv.rts_smoother(t,best[2],best[1])

def trial(a,sw,mode,rng,D0=45.,v0=-1.5,T=5.0,pool=None):
    if mode=='uniform': t=np.arange(int(15*T))/15
    else:
        while True:
            t=[0.]
            while t[-1]<T: t.append(t[-1]+rng.choice(pool))
            t=np.array(t[:-1])
            if len(t)>=13: break
    Dt=D0+v0*t+0.5*a*t**2
    wm=F_PX*W/Dt+rng.normal(0,sw,len(t)); D=F_PX*W/wm
    mid=T/2
    k,_=kv.sg_best_window(t,D); _,_,sg=kv.local_poly_fit(t,D,mid,k)
    xs=run_kf(t,D,wm,sw); kf=xs[np.argmin(abs(t-mid)),2]
    return sg,kf,len(t)

