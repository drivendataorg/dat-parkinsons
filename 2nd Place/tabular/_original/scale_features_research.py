"""Local scale-space curvature and contrast-normalized shape descriptors.

Unlike a fixed global intensity contour, these measure shape around each local
striatal maximum and subtract its surrounding background before normalization.
No fitted state or labels are used during extraction.
"""
import numpy as np
from scipy import ndimage as ndi

AX=(np.arange(48)-23.5)*2
XX,YY,ZZ=np.meshgrid(AX,AX,AX,indexing='ij')
MASK=(abs(YY)<30)&(abs(ZZ)<24)&(abs(XX)<38)&(abs(XX)>6)
SIDES=[MASK&(XX<0),MASK&(XX>0)]
ORDERS=[(2,0,0),(0,2,0),(0,0,2),(1,1,0),(1,0,1),(0,1,1)]

def extract_one(crop,intrinsic=False):
    c=np.asarray(crop,dtype=np.float32)
    f=[{},{}]
    for sigma in [1.,2.,3.]:
        sm=ndi.gaussian_filter(c,sigma)
        hs=[ndi.gaussian_filter(c,sigma,order=o) for o in ORDERS]
        for side,mask,out in zip(range(2),SIDES,f):
            idx=np.unravel_index(np.argmax(np.where(mask,sm,-np.inf)),sm.shape)
            bg=float(np.percentile(sm[mask],20))
            peak=float(sm[idx]);amp=max(peak-bg,1e-6)
            hh=[float(h[idx])*sigma**2/amp for h in hs]
            matrix=np.array([[hh[0],hh[3],hh[4]],[hh[3],hh[1],hh[5]],[hh[4],hh[5],hh[2]]])
            eig=-np.linalg.eigvalsh(matrix)
            for i,e in enumerate(eig):out[f's{sigma}_curve{i}']=e
            out[f's{sigma}_curve_ratio']=eig[-1]/max(eig[0],1e-6)
            out[f's{sigma}_contrast']=amp/max(bg,1e-6)
            out[f's{sigma}_peak']=peak
            if intrinsic:
                var=sigma**2/np.maximum(eig,sigma**2/400.)
                out[f's{sigma}_hessian_longdiff']=float(np.sqrt(max(var[-1]-var[0],0))*2)
                out[f's{sigma}_hessian_middiff']=float(np.sqrt(max(var[1]-var[0],0))*2)
            for threshold in [.2,.4,.6,.8]:
                support=mask&(sm>=bg+threshold*amp)
                labs,n=ndi.label(support)
                label=labs[idx]
                support=labs==label if label else support
                pts=np.argwhere(support).astype(float)
                wt=np.maximum(sm[support]-bg,0).astype(float)
                pref=f's{sigma}_t{threshold}'
                out[pref+'_vol']=len(pts)*.008
                out[pref+'_mass']=float(wt.sum())*.008/max(amp,1e-6)
                if len(pts)>=4 and wt.sum()>0:
                    ctr=np.average(pts,axis=0,weights=wt);d=pts-ctr
                    ev=np.maximum(np.linalg.eigvalsh((d*wt[:,None]).T@d/wt.sum()),1e-6)
                    for i,e in enumerate(ev):out[pref+f'_axis{i}']=float(np.sqrt(e)*2)
                    out[pref+'_elong']=float(np.sqrt(ev[-1]/ev[0]))
                    if intrinsic:
                        out[pref+'_longdiff']=float(np.sqrt(max(ev[2]-ev[0],0))*2)
                        out[pref+'_middiff']=float(np.sqrt(max(ev[1]-ev[0],0))*2)
                        out[pref+'_linearity']=float((ev[2]-ev[1])/max(ev[2]-ev[0],1e-6))
                else:
                    for i in range(3):out[pref+f'_axis{i}']=0.
                    out[pref+'_elong']=0.
                    if intrinsic:
                        for key in ['longdiff','middiff','linearity']:out[pref+'_'+key]=0.
    # Independent min/max avoids a single side-ordering error corrupting all descriptors.
    result={}
    for key in f[0]:
        a,b=f[0][key],f[1][key]
        result[key+'_min']=min(a,b);result[key+'_max']=max(a,b)
    return result

if __name__=='__main__':
    import argparse,time
    from pathlib import Path
    HERE=Path(__file__).resolve().parent
    ROOT=HERE.parents[1]
    ap=argparse.ArgumentParser();ap.add_argument('--frame',default='f2');ap.add_argument('--intrinsic',action='store_true');args=ap.parse_args()
    z=np.load(ROOT/'outputs'/f'crops_v3_{args.frame}.npz')
    t0=time.time();rows=[]
    for i,crop in enumerate(z['aligned']):
        rows.append(extract_one(crop,intrinsic=args.intrinsic))
        if i%200==0:print('processed',i,'elapsed',round(time.time()-t0),flush=True)
    import pandas as pd
    d=pd.DataFrame(rows);d.insert(0,'uid',z['uids'])
    assert np.isfinite(d.drop(columns='uid').to_numpy()).all()
    d.to_csv(HERE/f'{"scale2" if args.intrinsic else "scale"}_{args.frame}.csv',index=False)
    print('complete',d.shape,'seconds',round(time.time()-t0),flush=True)
