#!/usr/bin/env python3
"""Train-only gauge-factorized representation ablation, clean and shifted TEST."""
import argparse,json,time
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from aeon.transformations.collection.convolution_based import MiniRocket
from bench_neuro_ssl import acquire
from mesa_mr_experiment import train_encoder,encode,indices_for_seed

def separate(x):
    tau=np.linspace(-1,1,x.shape[1],dtype=np.float32)
    mu=x.mean(axis=1)
    slope=(x*tau[None,:,None]).sum(1)/(tau@tau)
    residual=x-mu[:,None,:]-tau[None,:,None]*slope[:,None,:]
    concept=np.concatenate((mu,slope),axis=1)
    return residual.astype(np.float32),concept.astype(np.float32)

def normalize(a,*others):
    sc=StandardScaler().fit(a)
    return tuple(np.clip(sc.transform(x),-10,10)/np.sqrt(a.shape[1]) for x in (a,)+others)

def repr_rocket(X,Y,Z,seed):
    tr=MiniRocket(n_kernels=1008,random_state=seed,n_jobs=1)
    a=np.asarray(tr.fit_transform(X.transpose(0,2,1)))
    b=np.asarray(tr.transform(Y.transpose(0,2,1)))
    c=np.asarray(tr.transform(Z.transpose(0,2,1)))
    return normalize(a,b,c)

def evaluate(trainblocks,testblocks,testshiftblocks,y,labels,ix,keys):
    cls=np.unique(y); Y=y[ix]; target=(Y[:,None]==cls[None,:]).astype(float)
    configs=[(cw,lw,alpha) for cw in ([0.] if 'concept' not in keys else [0.,.3,1.,3.])
             for lw in ([0.] if 'learned' not in keys else [0.,.3,1.,3.])
             for alpha in [.01,.1,1.,10.]]
    results=[]
    for cw,lw,alpha in configs:
        X=np.concatenate([trainblocks['shape'],cw*trainblocks['concept'],lw*trainblocks['learned']],axis=1)
        idx=X[ix]
        inv=np.linalg.inv(idx@idx.T+alpha*np.eye(len(ix)))
        coef=inv@target
        pred_loo=cls[(target-coef/inv.diagonal()[:,None]).argmax(1)]
        cv=float(f1_score(Y,pred_loo,average='macro'))
        results.append((cv,cw,lw,alpha,coef))
    # All choices use only labeled TRAIN examples, no TEST or shift information.
    cv,cw,lw,alpha,coef=sorted(results,key=lambda s:(-s[0],s[1],s[2],s[3]))[0]
    out={}
    for mode,blocks in [('clean',testblocks),('shifted',testshiftblocks)]:
        U=np.concatenate([blocks['shape'],cw*blocks['concept'],lw*blocks['learned']],axis=1)
        V=np.concatenate([trainblocks['shape'],cw*trainblocks['concept'],lw*trainblocks['learned']],axis=1)
        pred=cls[(U@V[ix].T@coef).argmax(1)]
        out[mode]=float(f1_score(labels,pred,average='macro'))
    return out,{'concept_weight':cw,'learned_weight':lw,'ridge':alpha,'train_loo_f1':cv}

def run(args):
    train,test,y,z,_=acquire(args.dataset)
    test_t=np.linspace(-1,1,test.shape[1],dtype=np.float32)
    rows=[]
    for seed in args.seeds:
        start=time.time();ix=indices_for_seed(y,seed)
        rng=np.random.default_rng(12037+seed)
        offset=rng.normal(scale=.75,size=(len(test),1,test.shape[2])).astype('float32')
        trend=rng.normal(scale=.50,size=(len(test),1,test.shape[2])).astype('float32')
        shifted=test+offset+test_t[None,:,None]*trend
        atr,ctr=separate(train);ate,cte=separate(test);ats,cts=separate(shifted)
        # All transformations and scalers fitted on unlabeled TRAIN only.
        rt,rv,rs=repr_rocket(train,test,shifted,seed)
        ct,cv,cs=repr_rocket(atr,ate,ats,seed)
        mt,mv,ms=normalize(ctr,cte,cts)
        zero=np.zeros((len(train),1));zv=np.zeros((len(test),1))
        learned={}
        for mechanism in [False,True]:
            model,_=train_encoder(atr,seed,args.epochs,mechanism=mechanism)
            ar,br,sr=encode(model,atr),encode(model,ate),encode(model,ats)
            learned[mechanism]=normalize(ar,br,sr)
        configs=[
          ('Raw-MiniRocket',False,False,False),
          ('Canonical-MiniRocket',True,False,False),
          ('Canonical+concept',True,True,False),
          ('Canonical+MR-no-mechanism',True,True,True),
          ('Canonical+MESA-MR',True,True,True)]
        for name,canonical,useconcept,uselearned in configs:
            shape=(ct,cv,cs) if canonical else (rt,rv,rs)
            branch=(learned[name=='Canonical+MESA-MR'] if uselearned else (zero,zv,zv))
            trainblocks={'shape':shape[0],'concept':mt,'learned':branch[0]}
            clean={'shape':shape[1],'concept':mv,'learned':branch[1]}
            shift={'shape':shape[2],'concept':ms,'learned':branch[2]}
            keys=['shape']+(['concept'] if useconcept else [])+(['learned'] if uselearned else [])
            outcome,meta=evaluate(trainblocks,clean,shift,y,z,ix,keys)
            rec=dict(dataset=args.dataset,seed=seed,method=name,clean_macro_f1=outcome['clean'],
                     shifted_macro_f1=outcome['shifted'],n_labeled=len(ix),epochs=args.epochs if uselearned else 0,
                     perturbation='label-preserving synthetic linear offset/trend assumed',
                     source='official UCR/UEA TRAIN-TEST; train-only normalization',meta=meta,status='EXECUTED',
                     runtime_seconds=round(time.time()-start,2))
            rows.append(rec);print('RESULT_JSON='+json.dumps(rec),flush=True)
        Path(args.out,'results.json').write_text(json.dumps(rows,indent=2))
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[206,207,208])
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--out',default='results')
    args=p.parse_args();Path(args.out).mkdir(parents=True,exist_ok=True)
    run(args)
