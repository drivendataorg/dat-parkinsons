"""Prepare a held local v17 candidate with four cross-fitted quantitative priors."""
import os
os.environ['OMP_NUM_THREADS']='3';os.environ['OPENBLAS_NUM_THREADS']='3'
from pathlib import Path
import sys,json,subprocess
HERE=Path(__file__).resolve().parent;PREV=HERE.parent/'20260914'
sys.path[:0]=[str(HERE.parent/'v16'),str(PREV)]
from bench import inputs,weights
from common import load_data
from portable3 import FeatPrep
from sklearn.linear_model import LogisticRegression
import numpy as np,pandas as pd

stage=HERE/'stage';rebuild=HERE/'rebuild';rebuild.mkdir(exist_ok=True)
if stage.exists():raise SystemExit('Candidate stage already exists; refusing to replace it')
subprocess.run(['/bin/cp','-cR',str(HERE.parent/'v16/stage'),str(stage)],check=True)
X,y,fold,g,u,cols=inputs();narrow,*rest=load_data('narrow');narrow_cols=rest[-1]
mixframe=pd.read_csv(HERE/'mixture_shape.csv').set_index('uid').loc[u];mix=mixframe.to_numpy();mixcols=['mixture__'+c for c in mixframe.columns]
family=np.array([s.split('/')[0] for s in g]);anchors=np.zeros((len(y),4));priors=[]
specs=[(X[:,:235],cols[:235],.2),(narrow,narrow_cols,1.),(X[:,:-4],cols[:-4],.2),(np.column_stack([X[:,:235],mix]),cols[:235]+mixcols,.2)]
def fit(F,tr,C):
    prep=FeatPrep().fit(F[tr]);model=LogisticRegression(C=C,penalty='l1',solver='liblinear',random_state=0,max_iter=5000).fit(prep.transform(F[tr]),y[tr],sample_weight=weights(y[tr],g[tr]));return prep,model
for j,(F,names,C) in enumerate(specs):
    for group in np.unique(family):
        tr=np.where(family!=group)[0];va=np.where(family==group)[0];prep,model=fit(F,tr,C)
        anchors[va,j]=model.decision_function(prep.transform(F[va]))
    prep,model=fit(F,np.arange(len(y)),C)
    priors.append(dict(features=names,prep=prep.to_dict(),coef=model.coef_[0].tolist(),intercept=float(model.intercept_[0])))
prep,selector=fit(X[:,:-4],np.arange(len(y)),.2);keep=np.argsort(-abs(selector.coef_[0]))[:64]
F=np.column_stack([X[:,keep],X[:,-4:],anchors]);median=np.nan_to_num(np.nanmedian(F,axis=0));F=np.where(np.isfinite(F),F,median)
chosen=[];rng=np.random.default_rng(16000);prior=y.mean()
for group in np.unique(g):
    ix=np.where(g==group)[0];pos=ix[y[ix]==1];neg=ix[y[ix]==0];n=min(len(pos)/prior,len(neg)/(1-prior))
    if min(len(pos),len(neg))==0:chosen.extend(ix);continue
    chosen.extend(rng.choice(pos,max(1,int(n*prior)),replace=False));chosen.extend(rng.choice(neg,max(1,int(n*(1-prior))),replace=False))
np.savez_compressed(rebuild/'context.npz',X=F,y=y,balanced=np.sort(chosen),median=median)
recipe=dict(features=[cols[i] for i in keep]+cols[-4:],n_selected=64,transforms=['raw','log'],n_estimators=8,
    checkpoint='tabicl-classifier-v2-20260212.ckpt',log_prep=FeatPrep().fit(F[:,:64]).to_dict(),median=median.tolist(),
    frozen_states={kind:f'tabular_{kind}.pkl.xz' for kind in ['raw','log']},blend_weight=.35,temperature=1.2,priors=priors,
    training_policy=dict(context_priors='Four linear models trained with each context case acquisition family excluded',query_priors='Same four models refitted on all training cases',feature_selection='Weighted L1 C=.2; original clinical and curvature features',raw='All training cases',log='Acquisition-balanced subset; seed16000'))
(stage/'assets/tabular_recipe.json').write_text(json.dumps(recipe,indent=2))
cfgpath=stage/'assets/inference_config.json';cfg=json.loads(cfgpath.read_text())
for name in cols+mixcols:
    if name not in cfg['feature_order']:cfg['feature_order'].append(name)
cfgpath.write_text(json.dumps(cfg,indent=2))

# Deploy only the pure feature functions. Research paths and command-line entry
# points must never be evaluated in the shallow platform working directory.
s=(HERE/'mixture_shape.py').read_text();s=s[s.index('AX='):s.index('\ndef main():')]
(stage/'mixture_features.py').write_text('"""Independent spatial uptake-mixture descriptors."""\nimport numpy as np\nfrom scipy import ndimage\nfrom scipy.special import logsumexp\n'+s)
p=stage/'inference_main.py';s=p.read_text().replace('    import scale_features','    import scale_features\n    import mixture_features')
s=s.replace('    absent = [k for k in order if k not in probe]', '    probe.update("mixture__" + k for k in mixture_features.extract(np.ones((48,48,48), np.float32)))\n    absent = [k for k in order if k not in probe]')
s=s.replace('            rows.append([f.get(k, np.nan) for k in order])','            f.update({"mixture__" + k: v for k, v in mixture_features.extract(shape_crop).items()})\n            rows.append([f.get(k, np.nan) for k in order])');p.write_text(s)
p=stage/'tabular_adapter.py';s=p.read_text()
s=s.replace('        self.models=[]','        self.priors=[]\n        for spec in self.spec.get("priors",[]):\n            self.priors.append(([order.index(name) for name in spec["features"]],FeatPrep.from_dict(spec["prep"]),np.asarray(spec["coef"]),float(spec["intercept"])))\n        self.models=[]')
s=s.replace('        X=np.where(np.isfinite(X),X,self.median)\n        p=np.mean', '        if self.priors:\n            full=np.asarray(full_features,dtype=np.float64)\n            scores=np.column_stack([prep.transform(full[:,idx])@coef+intercept for idx,prep,coef,intercept in self.priors])\n            X=np.column_stack([X,scores])\n        X=np.where(np.isfinite(X),X,self.median)\n        p=np.mean');p.write_text(s)
manifest=dict(name='v17-local-candidate',status='HELD: freeze and full pipeline verification pending; not approved for public submission',new_component='Spatial uptake mixtures plus four nested quantitative priors',raw_records_in_deployment=False)
(stage/'CANDIDATE.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'status':manifest['status'],'prior_models':len(priors),'context_features':F.shape[1],'feature_definitions':len(cfg['feature_order'])}),flush=True)
