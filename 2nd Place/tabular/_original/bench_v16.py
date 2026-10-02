"""Fixed v16 tree benchmarks; report aggregates and retain row results locally."""
import os
os.environ.setdefault('OMP_NUM_THREADS','3')
import sys,json,time,argparse,warnings
from pathlib import Path
HERE=Path(__file__).resolve().parent;PREV=HERE.parent/'20260914'
sys.path[:0]=[str(PREV/'vendor'),str(PREV)]
from common import load_data,baseline,metrics,ROOT,expit,logit
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from portable3 import FeatPrep
warnings.filterwarnings('ignore')

def inputs():
    X,y,fold,g,u,cols=load_data()
    shape=pd.read_csv(PREV/'scale_f2.csv').set_index('uid').loc[u]
    meta=pd.read_csv(ROOT/'outputs/features_v13_m2f.csv').set_index('uid').loc[u,['head_ml','inner_ml','ref','native_spacing']]
    return np.column_stack([X,shape,meta]),y,fold,g,u,cols+['shape__'+c for c in shape.columns]+list(meta.columns)

def weights(y,g):
    # Shrink each acquisition group's prevalence toward the training prevalence.
    # Weighting removes much of the reward for learning group prevalence alone.
    prior=y.mean();w=np.ones(len(y))
    for s in np.unique(g):
        sel=g==s;p=(y[sel].sum()+30*prior)/(sel.sum()+30)
        w[sel]=np.where(y[sel],prior/p,(1-prior)/(1-p))
    return w/w.mean()

def make_model(kind,seed):
    if kind.startswith('cat'):
        depth=int(kind[3])
        return CatBoostClassifier(iterations=1600,depth=depth,learning_rate=.025,l2_leaf_reg=12,
            loss_function='Logloss',thread_count=3,random_seed=seed,verbose=False,allow_writing_files=False)
    if kind=='extra':
        return ExtraTreesClassifier(n_estimators=900,max_features=.7,min_samples_leaf=3,n_jobs=3,random_state=seed)
    if kind=='hist':
        return HistGradientBoostingClassifier(max_iter=450,max_leaf_nodes=9,learning_rate=.035,l2_regularization=12,min_samples_leaf=25,early_stopping=False,random_state=seed)
    raise ValueError(kind)

def run():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',default='cat4');ap.add_argument('--protocol',default='original',choices=['original','fresh','family']);ap.add_argument('--balanced',action='store_true');ap.add_argument('--select',type=int,default=0);args=ap.parse_args()
    X,y,fold,g,u,cols=inputs();base,_=baseline(u,fold,y,correct_cnn=True)
    tab=np.mean([pd.read_csv(PREV/f'oof_tabicl_rich_{t}_e8_scale_sel64_metaall.csv').set_index('uid').loc[u,'p'] for t in ['raw','log']],axis=0)
    v15=expit(.65*logit(np.clip(base,1e-6,1-1e-6))/1.4+.35*logit(np.clip(tab,1e-6,1-1e-6))/1.2)
    family=np.array([s.split('/')[0] for s in g])
    if args.protocol=='original':splits=[(fold!=k,fold==k) for k in range(5)]
    elif args.protocol=='fresh':splits=list(StratifiedKFold(5,shuffle=True,random_state=160916).split(X,y))
    else:splits=[(family!=s,family==s) for s in np.unique(family)]
    tag=args.kind+('_balanced' if args.balanced else '')+(f'_sel{args.select}' if args.select else '')+'_'+args.protocol
    p=np.zeros(len(y));t0=time.time()
    for k,(tr,va) in enumerate(splits):
        med=np.nan_to_num(np.nanmedian(X[tr],axis=0));A=np.where(np.isfinite(X[tr]),X[tr],med);B=np.where(np.isfinite(X[va]),X[va],med)
        if args.select:
            prep=FeatPrep().fit(A[:,:-4]);sel=LogisticRegression(C=.2,penalty='l1',solver='liblinear',random_state=0,max_iter=5000).fit(prep.transform(A[:,:-4]),y[tr])
            keep=np.argsort(-abs(sel.coef_[0]))[:args.select]
            A=np.column_stack([A[:,keep],A[:,-4:]]);B=np.column_stack([B[:,keep],B[:,-4:]])
        model=make_model(args.kind,16000+k)
        model.fit(A,y[tr],sample_weight=weights(y[tr],g[tr]) if args.balanced else None)
        p[va]=model.predict_proba(B)[:,1]
        print(tag,'split',k,metrics(y[va],p[va]),'seconds',round(time.time()-t0),flush=True)
    d=pd.DataFrame(dict(uid=u,y=y,fold=fold,p=p));d.to_csv(HERE/f'oof_{tag}.csv',index=False)
    result=dict(name=tag,**metrics(y,p),seconds=round(time.time()-t0))
    if args.protocol=='original':
        result['v15']=metrics(y,v15)
        result['v15_blends']={str(w):metrics(y,expit((1-w)*logit(np.clip(v15,1e-6,1-1e-6))+w*logit(np.clip(p,1e-6,1-1e-6)))) for w in [.2,.4,.6]}
    result['groups']={s:metrics(y[g==s],p[g==s]) for s in np.unique(g) if sum(g==s)>=25}
    (HERE/f'{tag}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)

if __name__=='__main__':run()
