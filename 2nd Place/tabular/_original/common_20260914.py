"""Local-only experiment inputs and aggregate evaluation; never print rows."""
from pathlib import Path
import sys, json, zipfile
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def load_data(which='rich'):
    name = 'lin' if which == 'rich' else 'lin4'
    fn = 'features_v13_m2fnh.csv' if which == 'rich' else 'features_v4_m2fnh.csv'
    d = pd.read_csv(ROOT / 'outputs' / fn).set_index('uid')
    with zipfile.ZipFile(ROOT/'submissions/v13c_clean_2seed/submission.zip') as z:
        cols = json.loads(z.read(f'assets/linear_{name}.json'))['features']
    f = pd.read_csv(ROOT/'outputs/folds.csv').set_index('uid').loc[d.index]
    g = pd.read_csv(ROOT/'outputs/site_diag_rows.csv').set_index('uid').loc[d.index]
    groups = (g.fam2 + '/' + g.scale).to_numpy()
    X = d[cols].to_numpy(float)
    X[~np.isfinite(X)] = np.nan
    y = d.is_pathologic.to_numpy(int)
    assert np.array_equal(y, f.y.to_numpy(int))
    assert len(d.index.unique()) == len(d)
    return X, y, f.fold.to_numpy(int), groups, d.index.to_numpy(), cols

def metrics(y,p):
    q=np.clip(p,0.005,0.995)
    return dict(logloss=float(log_loss(y,q)),auroc=float(roc_auc_score(y,q)))

def baseline(uids,fold,y,version='v14c',correct_cnn=False):
    files = [('oof_feat3_m2fnh.csv','lin'),('oof_feat3_m2fnh.csv','lin4'),
             ('oof_mil48y.csv','mil48y')]
    if version=='v13c':
        files += [('oof_e2dmipx.csv','e2dmipx')]
        ratio = np.array([1,.4,.7,.5])
    else:
        files += [('oof_e2dmipf2x.csv','e2dmipf2x'),('oof_e2db3f2.csv','e2db3f2')]
        ratio = np.array([1,.4,.7,.3,.3])
    Z=np.column_stack([logit(np.clip(pd.read_csv(ROOT/'outputs'/fn).set_index('uid').loc[uids,c].to_numpy(float),1e-6,1-1e-6)) for fn,c in files])
    if correct_cnn:
        d=pd.read_csv(ROOT/'outputs/oof_mil48y6.csv').set_index('uid').loc[uids]
        c='mil48y6' if 'mil48y6' in d else [a for a in d if a not in ['y','fold','is_pathologic']][0]
        Z[:,2]=logit(np.clip(d[c].to_numpy(float),1e-6,1-1e-6))
    z=Z@ratio
    p=np.zeros(len(y))
    for k in np.unique(fold):
        tr=fold!=k; va=~tr
        m=LogisticRegression(C=1e6).fit(z[tr,None],y[tr])
        p[va]=m.predict_proba(z[va,None])[:,1]
    return p,Z

def report(name,y,p,base,groups):
    result={'name':name,**metrics(y,p)}
    result['delta_ll']=result['logloss']-metrics(y,base)['logloss']
    result['logit_corr']=float(np.corrcoef(logit(np.clip(p,1e-6,1-1e-6)),logit(np.clip(base,1e-6,1-1e-6)))[0,1])
    result['blends']={}
    for w in [.15,.3,.5]:
        q=expit((1-w)*logit(np.clip(base,1e-6,1-1e-6))+w*logit(np.clip(p,1e-6,1-1e-6)))
        result['blends'][str(w)]=metrics(y,q)
    result['centres']={str(g):metrics(y[groups==g],p[groups==g]) for g in np.unique(groups) if np.sum(groups==g)>=25 and len(np.unique(y[groups==g]))==2}
    print(json.dumps({k:v for k,v in result.items() if k!='centres'}),flush=True)
    return result
