#!/usr/bin/env python3
"""Unlabelled-train kernel branch with paired MESA mechanism ablation."""
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score
from aeon.transformations.collection.convolution_based import MiniRocket
from bench_neuro_ssl import acquire
from mesa_mr_experiment import train_encoder,encode,indices_for_seed

def normalized(a,b):
    s=StandardScaler().fit(a)
    return np.clip(s.transform(a),-10,10)/np.sqrt(a.shape[1]),np.clip(s.transform(b),-10,10)/np.sqrt(a.shape[1])

def predict(K,Kt,lab,cls,alpha):
    C=K+alpha*np.eye(len(K))
    inv=np.linalg.inv(C)
    Y=(lab[:,None]==cls[None,:]).astype(float)
    coef=inv@Y
    loo=Y-coef/inv.diagonal()[:,None]
    return cls[loo.argmax(1)],cls[(Kt@coef).argmax(1)]

def probe(blocks,labels,testlabels,ids,weights):
    C=np.unique(labels);Y=labels[ids]
    trials=[]
    for w in weights:
        a,b,base,bt=blocks
        f=np.c_[w*a,base]; t=np.c_[w*b,bt]
        gram=f[ids]@f[ids].T; grams=t@f[ids].T
        for lam in (.01,.1,1.,10.):
            cv,pred=predict(gram,grams,Y,C,lam)
            trials.append((f1_score(Y,cv,average='macro'),w,lam,pred))
    # Tie-break in favor of the smaller added learned contribution.
    _,w,lam,pred=sorted(trials,key=lambda r:(-r[0],r[1],r[2]))[0]
    return float(f1_score(testlabels,pred,average='macro')),float(accuracy_score(testlabels,pred)),{'weight':w,'regularization':lam}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[203,204,205])
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--out',default='results')
    o=p.parse_args();path=Path(o.out);path.mkdir(parents=True,exist_ok=True)
    X,T,y,z,cl=acquire(o.dataset);rows=[]
    for seed in o.seeds:
        st=time.time();idx=indices_for_seed(y,seed)
        rocket=MiniRocket(n_kernels=1008,random_state=seed,n_jobs=1)
        r,s=normalized(np.asarray(rocket.fit_transform(X.transpose(0,2,1))),np.asarray(rocket.transform(T.transpose(0,2,1))))
        representations={False:None,True:None}
        for flag in (False,True):
            model,_=train_encoder(X,seed,o.epochs,mechanism=flag)
            representations[flag]=normalized(encode(model,X),encode(model,T))
        configs=[('MiniRocket-only',False,[0.]),('MiniRocket+no-mechanism',False,[.1,.3,1.,3.]),
                 ('MiniRocket+MESA-MR',True,[.1,.3,1.,3.])]
        for name,flag,weights in configs:
            a,b=representations[flag]
            try:
                f,accuracy,meta=probe((a,b,r,s),y,z,idx,weights)
                row=dict(dataset=o.dataset,seed=seed,method=name,macro_f1=f,accuracy=accuracy,
                         epochs=o.epochs if name!='MiniRocket-only' else 0,labels=len(idx),status='EXECUTED',
                         selection='train-only LOO kernel ridge',meta=meta,elapsed=round(time.time()-st,2))
            except Exception as err:
                row=dict(dataset=o.dataset,seed=seed,method=name,status='BLOCKED',error=repr(err))
            rows.append(row);print('RESULT_JSON='+json.dumps(row),flush=True)
        (path/'results.json').write_text(json.dumps(rows,indent=2))
    if any(row['status']!='EXECUTED' for row in rows):
        raise RuntimeError('Incomplete experiment')

if __name__=='__main__':main()
