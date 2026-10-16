"""Independent spatial uptake-mixture descriptors."""
import numpy as np
from scipy import ndimage
from scipy.special import logsumexp
AX=(np.arange(24)-11.5)*4
XX,YY,ZZ=np.meshgrid(AX,AX,AX,indexing='ij')
COORD=np.stack([XX,YY,ZZ],axis=-1)

def moments(P,w):
    mu=w@P;d=P-mu;cov=(d*w[:,None]).T@d+np.eye(3)*1.
    return mu,cov

def logpdf(P,mu,cov):
    d=P-mu
    return -.5*(3*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+np.sum((d@np.linalg.inv(cov))*d,axis=1))

def fit(P,w):
    w=w/w.sum();mu,cov=moments(P,w);ev,U=np.linalg.eigh(cov)
    direction=U[:,-1];offset=direction*np.sqrt(ev[-1])*.65
    means=np.stack([mu-offset,mu+offset]);covs=np.stack([cov,cov]);mix=np.array([.5,.5])
    single=logpdf(P,mu,cov)
    for _ in range(25):
        score=np.column_stack([np.log(mix[k])+logpdf(P,means[k],covs[k]) for k in range(2)])
        resp=np.exp(score-logsumexp(score,axis=1)[:,None])*w[:,None]
        mass=resp.sum(0);old=means.copy()
        for k in range(2):means[k],covs[k]=moments(P,resp[:,k]/max(mass[k],1e-12))
        mix=np.maximum(mass,.01);mix/=mix.sum()
        if np.max(abs(means-old))<.005:break
    score=np.column_stack([np.log(mix[k])+logpdf(P,means[k],covs[k]) for k in range(2)])
    gain=float(w@(logsumexp(score,axis=1)-single))
    order=np.argsort(means[:,1]);a,b=order;delta=means[b]-means[a]
    pooled=(mix[:,None,None]*covs).sum(0);distance=float(np.sqrt(delta@np.linalg.solve(pooled,delta)))
    centered=P-mu;project=centered@direction
    skew=float(w@project**3/max(ev[-1]**1.5,1e-6));kurt=float(w@project**4/max(ev[-1]**2,1e-6))
    return dict(gain=max(gain,0),separation=float(np.linalg.norm(delta)),standard_separation=distance,
        posterior_mass=float(mix[a]),component_balance=float(min(mix)),posterior_anterior_ratio=float(mix[a]/mix[b]),
        ap_displacement=float(delta[1]),lateral_displacement=float(abs(delta[0])),si_displacement=float(abs(delta[2])),
        posterior_width=float(np.linalg.det(covs[a])**(1/6)),anterior_width=float(np.linalg.det(covs[b])**(1/6)),
        residual_width=float(np.linalg.det(pooled)**(1/6)),width_reduction=float(np.linalg.det(pooled)/np.linalg.det(cov)),
        absolute_skew=abs(skew),kurtosis=kurt)

def extract(crop):
    # Four-millimetre samples after antialiasing, identical for every case.
    c=ndimage.zoom(ndimage.gaussian_filter(np.asarray(crop,np.float32),.65),.5,order=1)
    all_sides=[]
    for sign in [-1,1]:
        mask=(XX*sign>6)&(XX*sign<38)&(abs(YY)<30)&(abs(ZZ)<24)
        bg=float(np.percentile(c[mask],20));peak=float(c[mask].max());amp=max(peak-bg,1e-6)
        d={}
        for level in [.1,.3]:
            w=np.maximum(c-bg-level*amp,0);valid=mask&(w>1e-8)
            if valid.sum()<10:
                valid=mask;w=np.ones_like(c)
            r=fit(COORD[valid],w[valid].astype(float))
            d.update({f't{level}_{k}':v for k,v in r.items()})
        all_sides.append(d)
    return {key+'_'+which:float(op([d[key] for d in all_sides])) for key in all_sides[0] for which,op in [('min',min),('max',max)]}
