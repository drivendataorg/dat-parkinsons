"""Offline, per-case-independent TabICL inference for a frozen training context."""
import os,json,warnings,pickle,lzma
from pathlib import Path
import numpy as np
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
warnings.filterwarnings('ignore')

class TabularAdapter:
    def __init__(self,assets,order,device=None,cache=True):
        import torch
        from tabicl import TabICLClassifier
        from portable3 import FeatPrep
        self.spec=json.loads((Path(assets)/'tabular_recipe.json').read_text())
        self.order=[order.index(name) for name in self.spec['features']]
        self.priors=[]
        for spec in self.spec.get("priors",[]):
            self.priors.append(([order.index(name) for name in spec["features"]],FeatPrep.from_dict(spec["prep"]),np.asarray(spec["coef"]),float(spec["intercept"])))
        self.models=[]
        self.prep=FeatPrep.from_dict(self.spec['log_prep'])
        self.n_selected=int(self.spec['n_selected'])
        dev=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.spec.get('frozen_states'):
            self.median=np.asarray(self.spec['median'],dtype=np.float64)
            for transform in self.spec['transforms']:
                path=Path(assets)/self.spec['frozen_states'][transform]
                opener=lzma.open if path.suffix=='.xz' else open
                with opener(path,'rb') as f:state=pickle.load(f)
                state['model_path']=str(Path(assets)/self.spec['checkpoint'])
                state['device']=dev
                model=TabICLClassifier.__new__(TabICLClassifier)
                model.__setstate__(state)
                self.models.append((transform,model))
            return
        context=np.load(Path(assets)/'tabular_context.npz')
        self.median=context['median'].astype(np.float64)
        train=context['X'].astype(np.float64);y=context['y'].astype(int)
        train=np.where(np.isfinite(train),train,self.median)
        for transform in self.spec['transforms']:
            X=self._transform(train,transform)
            model=TabICLClassifier(n_estimators=int(self.spec['n_estimators']),batch_size=1,device=dev,
                use_amp=False,use_fa3=False,kv_cache=cache,allow_auto_download=False,
                model_path=str(Path(assets)/self.spec['checkpoint']),random_state=42,n_jobs=3,verbose=False,
                inference_config={key:dict(safety_factor=.08,min_batch_size=1) for key in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
            model.fit(X,y)
            self.models.append((transform,model))

    def _transform(self,X,kind):
        if kind=='raw':return X
        if kind=='log':return np.column_stack([self.prep.transform(X[:,:self.n_selected]),X[:,self.n_selected:]])
        raise ValueError('Unsupported tabular transform')

    def predict(self,full_features):
        if len(full_features)==0:return np.zeros(0,dtype=np.float64)
        X=np.asarray(full_features,dtype=np.float64)[:,self.order]
        if self.priors:
            full=np.asarray(full_features,dtype=np.float64)
            scores=np.column_stack([prep.transform(full[:,idx])@coef+intercept for idx,prep,coef,intercept in self.priors])
            X=np.column_stack([X,scores])
        X=np.where(np.isfinite(X),X,self.median)
        p=np.mean([model.predict_proba(self._transform(X,kind))[:,1] for kind,model in self.models],axis=0)
        if not np.isfinite(p).all():raise RuntimeError('Tabular inference returned an invalid probability')
        return p.astype(np.float64)

    def blend(self,legacy_probability,tabular_probability):
        eps=1e-6
        def logit(p):
            p=np.clip(np.asarray(p,dtype=np.float64),eps,1-eps)
            return np.log(p/(1-p))
        w=float(self.spec['blend_weight']);T=float(self.spec['temperature'])
        z=(1-w)*logit(legacy_probability)+w*logit(tabular_probability)/T
        return 1/(1+np.exp(-np.clip(z,-30,30)))
