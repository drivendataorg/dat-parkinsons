"""Fit and freeze v17 contexts using the platform's core library versions."""
import os,sys,json,pickle,lzma,gc,hashlib,time
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent;PREV=HERE.parent/'20260914';stage=HERE/'stage';assets=stage/'assets'
sys.path[:0]=[str(PREV/'runtime_scipy115'),str(PREV/'runtime_vendor'),str(stage),str(stage/'tabicl_lib.zip')]
import numpy as np,torch
from portable3 import FeatPrep
from tabicl import TabICLClassifier
torch.set_num_threads(3)
recipe=json.loads((assets/'tabular_recipe.json').read_text());ctx=np.load(HERE/'rebuild/context.npz');X=ctx['X'];y=ctx['y'];idx=ctx['balanced']
prep=FeatPrep.from_dict(recipe['log_prep']);checks={};t0=time.time()
for kind in recipe['transforms']:
    A=X if kind=='raw' else np.column_stack([prep.transform(X[:,:64]),X[:,64:]])
    use=np.arange(len(y)) if kind=='raw' else idx
    model=TabICLClassifier(n_estimators=8,batch_size=1,device='cpu',use_amp=False,use_fa3=False,kv_cache='repr',
        allow_auto_download=False,model_path=str(assets/recipe['checkpoint']),random_state=42,n_jobs=3,verbose=False,
        inference_config={key:dict(safety_factor=.08,min_batch_size=1) for key in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
    model.fit(A[use],y[use]);probe=A[::37];before=model.predict_proba(probe)[:,1]
    reserved=model.model_.col_embedder.reserve_cls_tokens
    assert reserved==4
    for cache in model.model_kv_cache_.values():
        for kv in cache.col_cache.kv.values():kv.key[:,:reserved].zero_();kv.value[:,:reserved].zero_()
    model._save_model_weights=False;model._save_kv_cache=True;model._save_training_data=False
    state=model.__getstate__();state['model_path']=recipe['checkpoint'];eg=state['ensemble_generator_']
    assert eg.X_ is None and eg.y_ is None and all(p.X_transformed_ is None for p in eg.preprocessors_.values())
    path=assets/recipe['frozen_states'][kind];tmp=path.with_suffix('.tmp')
    with lzma.open(tmp,'wb',preset=6) as f:pickle.dump(state,f,protocol=5)
    tmp.replace(path)
    del state,model;gc.collect()
    with lzma.open(path,'rb') as f:state=pickle.load(f)
    state['model_path']=str(assets/recipe['checkpoint']);state['device']='cpu'
    restored=TabICLClassifier.__new__(TabICLClassifier);restored.__setstate__(state);after=restored.predict_proba(probe)[:,1]
    error=float(abs(before-after).max());assert error<1e-6
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    checks[kind]=dict(max_probability_difference=error,bytes=path.stat().st_size,sha256=digest,raw_records_removed=True)
    print(kind,'frozen',json.dumps(checks[kind]),'seconds',round(time.time()-t0),flush=True)
    del restored,state;gc.collect()
(HERE/'persistence_checks.json').write_text(json.dumps(checks,indent=2))
manifest=json.loads((stage/'CANDIDATE.json').read_text());manifest['status']='frozen models verified; pipeline check pending';manifest['persistence_checks']=checks
(stage/'CANDIDATE.json').write_text(json.dumps(manifest,indent=2))
